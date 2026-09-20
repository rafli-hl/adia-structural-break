"""Explicit synthetic tests; no full-data scores or tuning."""
import json
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sbrt.history import ARChannel, residuals, fit_ar, EmpiricalCDF
from sbrt.features import evidence
from sbrt.stage2_detectors import Stage2Detector
from sbrt.logistic import fit_fold as fit_e08, training_weights, standardize_fit, objective
from sbrt.data import Series
from sbrt.replay import replay
from stage4_research.contract import IDS, INPUTS, RAW, BASE, CONFIG, STANDALONE
from stage4_research.scale import ScaleEvidence, lag_one
from stage4_research.features import FeatureState, Preprocessing, contrasts, Detector, infer
from stage4_research.models import fit_fold, Model


def fixture():
    rng = np.random.default_rng(20260922)
    rows = []
    for i in range(50):
        for t in range(40+i%7):
            rows.append(dict(id=f"s{i:03}",fold=i%5,time=200+t,time_online=t,
                target=int(i%2==1 and t>=i%15),**{k:float(rng.uniform()) for k in RAW}))
    return pd.DataFrame(rows)


def history_fixture():
    rng = np.random.default_rng(20260922)
    history = rng.normal(size=601)
    for j in range(1,len(history)):
        history[j] += .4*history[j-1]
    return history,rng.normal(size=301)


class ScaleTests(unittest.TestCase):
    def test_ar_exact_units_coefficients_and_observed_boundary(self):
        h,o = history_fixture()
        state = FeatureState(h,"export")
        ar = ARChannel(h)
        fitted = fit_ar(h[:int(.7*len(h))],5,1.)
        np.testing.assert_array_equal(state.ar.beta,fitted[2])
        self.assertEqual(state.ar.mu,fitted[0]); self.assertEqual(state.ar.sd,fitted[1])
        beta = state.ar.beta.copy()
        for x in o[:12]:
            e,_ = ar.update(x)
            state.observe(x)
            self.assertEqual(state.diagnostic["e_squared"],e*e)
            np.testing.assert_array_equal(state.ar.lags,ar.lags)
            np.testing.assert_array_equal(state.blocks.ar.lags,ar.lags)
            state.commit()
        np.testing.assert_array_equal(state.ar.beta,beta)
        self.assertFalse(state.ar.beta.flags.writeable)

    def test_historical_split_exclusion_recursion_reference_and_terminal_state(self):
        h,_ = history_fixture(); ar = ARChannel(h)
        e = residuals(h,(ar.mu,ar.sd,ar.beta)); k = ar.cut
        for kind,alpha in (("slow",.01),("fast",.05),("robust",.01)):
            scale = ScaleEvidence(e,k,kind,rank=kind=="slow")
            b = max(float(np.mean(e[5:k]**2)),1e-16)
            self.assertEqual(scale.b,b)
            state,ref = b,[]
            c = float(np.quantile(e[5:k]**2,.99,method="linear"))
            mc = float(np.mean(np.minimum(e[5:k]**2,c)))
            for j in range(5,len(e)):
                if j>=k: ref.append(e[j]/np.sqrt(max(state,max(1e-16,1e-4*b))))
                driver = b/mc*min(e[j]**2,c) if kind=="robust" and mc>1e-16 else e[j]**2
                state = (1-alpha)*state+alpha*driver
            self.assertEqual(scale.h,state)
            self.assertEqual(scale.mu,float(np.mean(ref)))
            self.assertEqual(scale.sd,max(float(np.std(ref,ddof=1)),1e-8))
            changed = e.copy(); changed[:5] = [np.inf,np.nan,-np.inf,1e300,-1e300]
            self.assertEqual(ScaleEvidence(changed,k,kind).h,scale.h)
            h_before = scale.h
            row = scale.observe(2.)
            self.assertEqual(row["variance"],max(h_before,scale.v_min))
            self.assertEqual(scale.h,h_before)
            scale.commit()
            driver = b/mc*min(4.,c) if kind=="robust" else 4.
            self.assertEqual(scale.h,(1-alpha)*h_before+alpha*driver)

    def test_forecast_does_not_use_current_or_future_innovation(self):
        h,_ = history_fixture(); ar = ARChannel(h)
        e = residuals(h,(ar.mu,ar.sd,ar.beta))
        for kind in ("slow","fast","robust"):
            a = ScaleEvidence(e,ar.cut,kind)
            b = ScaleEvidence.restore(a.state())
            prior = a.h
            av,bv = a.observe(.2),b.observe(200.)
            self.assertEqual(av["variance"],bv["variance"])
            self.assertEqual(a.h,prior); self.assertEqual(b.h,prior)
            self.assertNotEqual(av["u"],bv["u"])
            a.commit(); b.commit(); self.assertNotEqual(a.h,b.h)
            # A later C observation cannot change any earlier forecast; replay
            # two histories sharing A and prefix C with the fixed AR fit.
            modified = e.copy(); modified[ar.cut+5:] *= 12.
            states = []
            for values in (e,modified):
                s = ScaleEvidence(values[:ar.cut+5],ar.cut,kind)
                states.append(s.h)
            self.assertEqual(states[0],states[1])

    def test_robust_calibration_quantile_normalization_unclipped_numerator(self):
        e = np.r_[np.full(5,np.nan),np.linspace(-3,3,95),np.ones(40)]
        e[90] = 100.
        s = ScaleEvidence(e,100,"robust")
        expected_cap = float(np.quantile(e[5:100]**2,.99,method="linear"))
        self.assertEqual(s.cap,expected_cap)
        self.assertEqual(s.factor,s.b/float(np.mean(np.minimum(e[5:100]**2,expected_cap))))
        self.assertAlmostEqual(float(np.mean(s.factor*np.minimum(e[5:100]**2,s.cap))),s.b,places=12)
        prior = s.h
        v = s.observe(10000.)
        self.assertEqual(v["u"],10000./np.sqrt(max(prior,s.v_min)))
        s.commit()
        self.assertEqual(s.h,.99*prior+.01*s.factor*s.cap)
        self.assertEqual(s.counts["online_clipping"],1)

    def test_robust_unsupported_and_floor_boundaries(self):
        e = np.r_[np.full(5,np.nan),np.zeros(95),np.zeros(40)]
        s = ScaleEvidence(e,100,"robust")
        self.assertFalse(s.robust_supported); self.assertIsNone(s.factor)
        self.assertEqual(s.b,1e-16); self.assertEqual(s.v_min,1e-16)
        self.assertEqual(s.counts["baseline_floor"],1)
        self.assertEqual(s.counts["robust_fallback"],1)
        prior = s.h
        v = s.observe(10.)
        self.assertEqual(v["variance"],1e-16)
        self.assertEqual(s.counts["online_floor"],1)
        s.commit(); self.assertEqual(s.h,.99*prior+.01*100.)
        s.h = s.v_min; s.observe(0.); s.commit()
        self.assertEqual(s.counts["online_floor"],1)
        self.assertGreaterEqual(s.counts["reference_sd_floor"],2)

    def test_all_energy_rank_windows_direct_prefix(self):
        h,o = history_fixture(); ar = ARChannel(h)
        e = residuals(h,(ar.mu,ar.sd,ar.beta))
        for kind in ("slow","fast","robust"):
            scale = ScaleEvidence(e,ar.cut,kind,rank=kind=="slow")
            reference_energy = scale.energy
            squares,v_values,v_squares = [],[],[]
            for residual in o:
                row = scale.observe(float(residual))
                squares.append(row["a"]**2)
                direct = []
                for w in (32,128,256,len(squares)):
                    values = squares[-w:]
                    direct.append(evidence(len(values),float(np.mean(values)),reference_energy.ref_mean,reference_energy.ref_sd))
                np.testing.assert_allclose(row["energy_components"],direct,rtol=0,atol=2e-14)
                if kind=="slow":
                    v = 2*float(scale.cdf.transform(row["a"]))-1
                    v_values.append(v); v_squares.append(v*v)
                    scores = []
                    for observed,reference in ((v_values,scale.location),(v_squares,scale.rank_energy)):
                        for w in (32,128,256,len(observed)):
                            sample = observed[-w:]
                            scores.append(evidence(len(sample),float(np.mean(sample)),reference.ref_mean,reference.ref_sd))
                    self.assertAlmostEqual(row["P"],sum(scores)/8,places=13)
                scale.commit()

    def test_alpha_zero_static_E04_equivalence_no_candidate(self):
        h,o = history_fixture(); ar = ARChannel(h)
        e = residuals(h,(ar.mu,ar.sd,ar.beta))
        s = ScaleEvidence(e,ar.cut,"slow",audit_alpha=0)
        old = Stage2Detector(h,"E04")
        for x in o:
            residual,_ = ar.update(x)
            row = s.observe(residual)
            self.assertAlmostEqual(row["J"],old.update(x),delta=1e-12)
            s.commit()
        self.assertEqual(s.counts["online_floor"],0)
        self.assertEqual(CONFIG["historical"]["slow_alpha"],.01)
        with self.assertRaises(ValueError): ScaleEvidence(e,ar.cut,"slow",audit_alpha=.02)

    def test_cdf_ties_extremes_inclusive_reference_and_no_mutation(self):
        ref = np.array([-2.,0.,0.,3.])
        cdf = EmpiricalCDF(ref)
        np.testing.assert_array_equal(cdf.transform([-100,-2,0,3,100]),[.1,.2,.5,.8,.9])
        h,o = history_fixture(); ar = ARChannel(h)
        innovations = residuals(h,(ar.mu,ar.sd,ar.beta))
        s = ScaleEvidence(innovations,ar.cut,"slow",rank=True)
        before = s.cdf.reference.copy()
        # Preserve chronological reference order: sorting before mean changes
        # floating-point summation, though the mathematical CDF is identical.
        forecast = s.b
        ref = []
        for j in range(5,len(innovations)):
            if j>=ar.cut: ref.append(innovations[j]/np.sqrt(max(forecast,s.v_min)))
            forecast = .99*forecast+.01*innovations[j]**2
        ref = (np.asarray(ref)-s.mu)/s.sd
        v = 2*s.cdf.transform(ref)-1
        self.assertEqual(s.location.ref_mean,float(np.mean(v)))
        for e in o: s.observe(float(e)); s.commit()
        np.testing.assert_array_equal(s.cdf.reference,before)
        self.assertFalse(s.cdf.reference.flags.writeable)

    def test_invalid_support_nonfinite_and_uncommitted_fail_closed(self):
        with self.assertRaises(ValueError): FeatureState(np.ones(100),"E14")
        h,o = history_fixture(); state = FeatureState(h,"E14")
        with self.assertRaises(ValueError): state.observe(np.nan)
        state = FeatureState(h,"E14"); state.observe(1.)
        with self.assertRaises(ValueError): state.observe(2.)
        state.commit()
        with self.assertRaises(ValueError): state.commit()
        ar = ARChannel(h); e = residuals(h,(ar.mu,ar.sd,ar.beta))
        with self.assertRaises(ValueError): ScaleEvidence(e,len(e)-1,"slow")
        e[100] = np.inf
        with self.assertRaises(ValueError): ScaleEvidence(e,ar.cut,"slow")

    def test_historical_squared_acf_support_and_direct_pearson(self):
        h,_ = history_fixture()
        self.assertAlmostEqual(lag_one(h*h),float(np.corrcoef(h[:-1]**2,h[1:]**2)[0,1]),places=14)
        self.assertIsNone(lag_one(np.ones(10)))


class ModelTests(unittest.TestCase):
    def test_allowlists_contrasts_and_standalone_clarification(self):
        frame = fixture()
        self.assertEqual([len(INPUTS[k]) for k in IDS],[5,5,5,6])
        self.assertEqual(STANDALONE,dict(E14="J_s",E15="J_f",E16="J_r",E17="P_s"))
        for name,suffix in (("E14","s"),("E15","f"),("E16","r"),("E17","s")):
            expected = ((frame["J_"+suffix]-frame.I)/np.sqrt(2)).to_numpy()
            np.testing.assert_array_equal(contrasts(name,frame)[:,0],expected)
            if name=="E17": np.testing.assert_array_equal(contrasts(name,frame)[:,1],(frame.P_s-frame.P)/np.sqrt(2))
            model,_,_ = fit_fold(frame,0,name)
            changed = frame.copy()
            for key in ("id","target","fold","time_online","horizon","tau_index","q","acf"):
                changed[key] = -1234
            np.testing.assert_array_equal(model.predict(frame),model.predict(changed))
        self.assertFalse(CONFIG["standalone_selectable"])

    def test_training_only_weights_means_variances_and_eligible_mask(self):
        frame = fixture()
        train = frame[frame.fold!=0]
        eligible,w,audit = training_weights(train.time_online,train.target)
        self.assertLessEqual(max(audit["maximum_absolute_errors"].values()),1e-12)
        x = np.asfortranarray(train.loc[eligible,list(BASE)].to_numpy())
        mean,std,_ = standardize_fit(x,w)
        pre,z = Preprocessing.fit("E17",train.loc[eligible],w)
        np.testing.assert_array_equal(pre.mean,mean); np.testing.assert_array_equal(pre.std,std)
        raw = np.asfortranarray(contrasts("E17",train.loc[eligible]))
        expected = np.array([np.dot(w,raw[:,j])/w.sum() for j in range(2)])
        np.testing.assert_array_equal(pre.extra_mean,expected)
        np.testing.assert_array_equal(pre.extra_std,np.sqrt([np.dot(w,(raw[:,j]-expected[j])**2)/w.sum() for j in range(2)]))
        self.assertEqual(z.shape[1],6)

    def test_heldout_feature_label_age_isolation_and_deterministic_training(self):
        frame = fixture()
        changed = frame.copy(); mask = changed.fold==0
        changed.loc[mask,"target"] = 1-changed.loc[mask,"target"]
        changed.loc[mask,"time_online"] = 99999
        changed.loc[mask,list(RAW)] = 1e20
        for name in IDS:
            first,detail,weights = fit_fold(frame,0,name)
            second,_,again = fit_fold(changed,0,name)
            self.assertEqual(first.dumps(),second.dumps()); self.assertEqual(weights,again)
            repeat,_,_ = fit_fold(frame,0,name)
            self.assertEqual(first.dumps(),repeat.dumps())
            self.assertLessEqual(detail["gradient_infinity_norm"],1e-6)
            self.assertTrue(detail["optimizer_success"])
            restored = Model.loads(first.dumps())
            self.assertEqual(first.dumps(),restored.dumps())
            np.testing.assert_array_equal(first.predict(frame),restored.predict(frame))

    def test_nested_e08_objective_and_predictions_all_experiments(self):
        frame = fixture()
        with threadpool_limits(limits=1):
            old,_,_ = fit_e08(frame,0)
            train = frame[frame.fold!=0]
            eligible,w,_ = training_weights(train.time_online,train.target); train=train.loc[eligible]
            for name in IDS:
                model,_,_ = fit_fold(frame,0,name)
                np.testing.assert_array_equal(old.mean,model.pre.mean); np.testing.assert_array_equal(old.std,model.pre.std)
                beta = np.r_[old.beta,np.zeros(len(INPUTS[name])-4)]
                nested = Model(model.pre,old.intercept,beta)
                np.testing.assert_allclose(nested.predict(frame),old.predict(frame.loc[:,list(BASE)].to_numpy()),atol=1e-14,rtol=0)
                x=model.pre.transform(train); y=train.target.to_numpy()
                self.assertAlmostEqual(objective(np.r_[old.intercept,beta],x,y,w)[0],objective(np.r_[old.intercept,old.beta],x[:,:4],y,w)[0],delta=1e-14)

    def test_nested_e14_into_e17_and_zero_variance_column_policy(self):
        frame = fixture(); a,_,_=fit_fold(frame,0,"E14"); b,_,_=fit_fold(frame,0,"E17")
        np.testing.assert_array_equal(a.pre.extra_mean,b.pre.extra_mean[:1]); np.testing.assert_array_equal(a.pre.extra_std,b.pre.extra_std[:1])
        nested=Model(b.pre,a.intercept,np.r_[a.beta,0.])
        np.testing.assert_allclose(nested.predict(frame),a.predict(frame),atol=1e-14,rtol=0)
        train=frame[frame.fold!=0]; eligible,w,_=training_weights(train.time_online,train.target); train=train.loc[eligible]
        x=b.pre.transform(train); y=train.target.to_numpy()
        with threadpool_limits(limits=1):
            self.assertAlmostEqual(objective(np.r_[a.intercept,a.beta,0.],x,y,w)[0],objective(np.r_[a.intercept,a.beta],x[:,:-1],y,w)[0],delta=1e-14)
        frame["J_s"]=frame.I; frame["R"]=.25
        pre,_=Preprocessing.fit("E14",frame,np.full(len(frame),1/len(frame)))
        frame["J_s"]=99.; frame["R"]=-99.
        x=pre.transform(frame)
        np.testing.assert_array_equal(x[:,0],0); np.testing.assert_array_equal(x[:,-1],0)

    def test_state_exact_resume_model_binding_and_parameter_immutability(self):
        h,o=history_fixture(); frame=fixture()
        for name in IDS:
            model,_,_=fit_fold(frame,0,name)
            state=Detector(h,model); initial=state.dumps(); clone=Detector.loads(initial,model)
            self.assertEqual(initial,clone.dumps())
            for value in o[:129]: self.assertEqual(state.update(value),clone.update(value))
            snapshot=state.dumps(); resumed=Detector.loads(snapshot,Model.loads(model.dumps()))
            self.assertEqual(snapshot,resumed.dumps())
            for value in o[129:]: self.assertEqual(state.update(value),resumed.update(value))
            self.assertEqual(state.dumps(),resumed.dumps())
            bad=Model(model.pre,model.intercept+.1,model.beta)
            with self.assertRaises(ValueError): Detector.loads(snapshot,bad)
            self.assertFalse(resumed.features.ar.beta.flags.writeable)
            if name=="E17": self.assertFalse(resumed.features.scales["slow"].cdf.reference.flags.writeable)

    def test_prefix_truncation_order_handshake_scalar_batch_and_full_key_coverage(self):
        h,o=history_fixture(); frame=fixture()
        records=[Series(str(i),h+i*.03,o+i*.07,np.zeros(len(o),dtype=int),np.arange(len(o)), -1) for i in range(2)]
        for name in IDS:
            model,_,_=fit_fold(frame,0,name)
            results={s.id:p for s,p,_ in replay(records,infer_fn=lambda ds:infer(ds,model))}
            reversed_results={s.id:p for s,p,_ in replay(list(reversed(records)),infer_fn=lambda ds:infer(ds,model))}
            for s in records:
                np.testing.assert_array_equal(results[s.id],reversed_results[s.id])
                state=Detector(s.historical,model)
                short=np.array([state.update(x) for x in s.online[:73]])
                np.testing.assert_array_equal(short,results[s.id][:73])
                f=FeatureState(s.historical,name); rows=[]
                for x in s.online:
                    rows.append(dict(f.observe(x))); f.commit()
                np.testing.assert_array_equal(model.predict(pd.DataFrame(rows)),results[s.id])
                self.assertEqual(len(results[s.id]),len(s.online))
                self.assertTrue(((results[s.id]>=0)&(results[s.id]<=1)).all())
            all_predictions=np.full(len(frame),np.nan); visits=np.zeros(len(frame))
            for fold in range(5):
                fitted,_,_=fit_fold(frame,fold,name); mask=frame.fold==fold
                all_predictions[mask]=fitted.predict(frame[mask]); visits[mask]+=1
            np.testing.assert_array_equal(visits,1); self.assertTrue(np.isfinite(all_predictions).all())

    def test_export_individual_streams_identical_and_pending_serialization(self):
        h,o=history_fixture(); export=FeatureState(h,"export")
        individuals={name:FeatureState(h,name) for name in IDS}
        for x in o[:35]:
            row=export.observe(x)
            pending=FeatureState.loads(export.dumps())
            self.assertEqual(export.dumps(),pending.dumps())
            for name,state in individuals.items():
                other=state.observe(x)
                self.assertEqual(other,{key:row[key] for key in other}); state.commit()
            export.commit(); pending.commit(); self.assertEqual(export.dumps(),pending.dumps())
