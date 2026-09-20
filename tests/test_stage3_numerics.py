"""Explicit synthetic contract tests, never experimental selection data."""
import hashlib
import json
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier
from sbrt.logistic import fit_fold as fit_e08, training_weights, standardize_fit, objective
from sbrt.history import ARChannel
from sbrt.stage2_detectors import Stage2Detector
from sbrt.stage3_features import (INPUT_LISTS, RAW, CONTRASTS, Preprocessing, extension,
    context_q, maturity, raw_values, Stage3Detector, infer)
from sbrt.stage3_models import fit_fold, Model, TREE_PARAMS, coefficient_diagnostics
from sbrt.data import Series
from sbrt.replay import replay
from test_e08 import fixture as base_fixture


def fixture():
    frame = base_fixture()
    rng = np.random.default_rng(309)
    for name in RAW[4:]:
        frame[name] = rng.uniform(size=len(frame))
    frame["q"] = frame.id.map({key:context_q(float(r)) for key,r in zip(frame.id.unique(),rng.uniform(0,1.3,size=frame.id.nunique()))})
    return frame


class Stage3Numerics(unittest.TestCase):
    def test_stage3_historical_ratio_exact_units_clipping_support_and_freezing(self):
        rng = np.random.default_rng(99)
        history = rng.normal(size=150)
        ar = ARChannel(history)
        ref = ar.reference(history)
        z = (history[ar.cut:]-ar.mu)/ar.sd
        self.assertEqual(ar.mse_ratio, float(np.mean(ref*ref))/float(np.mean(z*z)))
        self.assertEqual(context_q(None),0.)
        for r,q in ((0,1),(.7,.3),(1,0),(1.4,0)):
            self.assertAlmostEqual(context_q(r),q)
        for r in (-.01,np.nan,np.inf,-np.inf):
            with self.assertRaises(ValueError):
                context_q(r)
        ratio = ar.mse_ratio
        for x in rng.normal(size=30):
            ar.update(x)
        self.assertEqual(ratio,ar.mse_ratio)
        # Actual zero-denominator history, not a substituted AR implementation.
        h = np.r_[np.tile([-1.,1.],35),np.zeros(30)]
        zero = ARChannel(h)
        self.assertIsNone(zero.mse_ratio)
        self.assertEqual(context_q(zero.mse_ratio),0.)

    def test_stage3_exact_allowlists_and_prohibited_metadata_exclusion(self):
        frame = fixture()
        self.assertEqual([len(INPUT_LISTS[k]) for k in INPUT_LISTS],[6,7,5,7,4])
        for name in INPUT_LISTS:
            with self.subTest(name=name):
                model,_,_ = fit_fold(frame,0,name)
                changed = frame.copy()
                for key in ("id","fold","target","final_horizon","acf","kurtosis","tau_index"):
                    changed[key] = "PROHIBITED_TEST_VALUE"
                if name!="E09":
                    changed["q"] = np.nan
                if name!="E12":
                    changed["time_online"] = np.nan
                if name!="E10":
                    changed[list(RAW[4:8])] = np.nan
                if name!="E11":
                    changed[list(RAW[8:])] = np.nan
                np.testing.assert_array_equal(model.predict(frame),model.predict(changed))
                self.assertEqual(model.pre.as_dict()["inputs"],INPUT_LISTS[name])

    def test_stage3_contrasts_exact_and_zero_sum(self):
        frame = fixture()
        b = np.zeros((len(frame),4))
        a,c,d,e = (frame[k].to_numpy() for k in RAW[4:8])
        expected = np.column_stack(((a-c)/np.sqrt(2),(a+c-2*d)/np.sqrt(6),(a+c+d-3*e)/np.sqrt(12)))
        np.testing.assert_array_equal(extension("E10",frame,b,0),expected)
        np.testing.assert_allclose(CONTRASTS.sum(axis=1),0,atol=1e-15)
        np.testing.assert_allclose(CONTRASTS@CONTRASTS.T,np.eye(3),atol=1e-15)
        np.testing.assert_array_equal(extension("E11",frame,b,0)[:,0],(frame.P_energy-frame.P_location)/np.sqrt(2))

    def test_stage3_training_only_normalization_weights_and_products(self):
        frame = fixture()
        train = frame.loc[frame.fold!=0]
        eligible,w,audit = training_weights(train.time_online,train.target)
        train = train.loc[eligible]
        with threadpool_limits(limits=1):
            mean,std,z = standardize_fit(train[list(RAW[:4])].to_numpy(),w)
            for name in INPUT_LISTS:
                pre,x = Preprocessing.fit(name,train,w)
                np.testing.assert_array_equal(pre.mean,mean)
                np.testing.assert_array_equal(pre.std,std)
                np.testing.assert_array_equal(x[:,:4],z)
                np.testing.assert_array_equal(x,pre.transform(train))
                u = extension(name,train,z,pre.center)
                if u.shape[1]:
                    np.testing.assert_allclose(pre.extra_mean,np.average(u,weights=w,axis=0),atol=2e-15)
                    np.testing.assert_allclose(pre.extra_std,np.sqrt(np.average((u-pre.extra_mean)**2,weights=w,axis=0)),atol=2e-15)
                if name=="E09":
                    self.assertAlmostEqual(pre.center,np.average(train.q,weights=w),places=15)
                if name=="E12":
                    self.assertAlmostEqual(pre.center,np.average(maturity(train.time_online),weights=w),places=15)
        self.assertLess(max(audit["maximum_absolute_errors"].values()),1e-12)

    def test_stage3_heldout_isolation_and_deterministic_fit_all_models(self):
        frame = fixture()
        bad = frame.copy()
        bad.loc[bad.fold==2, [*RAW,"q","time_online","target"]] = np.nan
        for name in INPUT_LISTS:
            with self.subTest(name=name):
                a,da,wa = fit_fold(frame,2,name)
                b,db,wb = fit_fold(bad,2,name)
                self.assertEqual(a.dumps(),b.dumps())
                self.assertEqual(wa,wb)
                np.testing.assert_array_equal(a.predict(frame),b.predict(frame))
                if name!="E13":
                    self.assertLessEqual(da["gradient_infinity_norm"],1e-6)
                    self.assertEqual(da["objective"],db["objective"])

    def test_stage3_nested_e08_zero_extensions_predictions_and_objective(self):
        frame = fixture()
        old,_,old_audit = fit_e08(frame,0)
        train = frame[frame.fold!=0]
        eligible,w,_ = training_weights(train.time_online,train.target)
        train = train.loc[eligible]
        with threadpool_limits(limits=1):
            for name in ("E09","E10","E11","E12"):
                pre,x = Preprocessing.fit(name,train,w)
                model = Model(pre,old.intercept,np.r_[old.beta,np.zeros(x.shape[1]-4)])
                np.testing.assert_array_equal(old.mean,pre.mean)
                np.testing.assert_array_equal(old.std,pre.std)
                np.testing.assert_allclose(old.predict(frame[list(RAW[:4])].to_numpy()),model.predict(frame),rtol=0,atol=1e-14)
                theta = np.r_[model.intercept,model.beta]
                a = objective(theta,x,train.target.to_numpy(),w)[0]
                b = objective(theta[:5],x[:,:4],train.target.to_numpy(),w)[0]
                self.assertLessEqual(abs(a-b),1e-14)

    def test_stage3_constant_context_maturity_and_zero_variance_inference(self):
        frame = fixture()
        w = np.full(len(frame),1/len(frame))
        frame["q"] = .3
        frame["time_online"] = 0
        for name in ("E09","E12"):
            pre,x = Preprocessing.fit(name,frame,w)
            np.testing.assert_array_equal(pre.extra_std,np.zeros(x.shape[1]-4))
            np.testing.assert_array_equal(x[:,4:],0)
            changed = frame.assign(q=.9,time_online=999)
            np.testing.assert_array_equal(pre.transform(changed)[:,4:],0)
        self.assertEqual(maturity([0])[0],np.log(2))
        for bad in ([-1],[.5],[np.nan]):
            with self.assertRaises(ValueError):
                maturity(bad)

    def test_stage3_export_components_and_authoritative_averages(self):
        rng = np.random.default_rng(3)
        state = Stage2Detector(rng.normal(size=150),"E07")
        with self.assertRaises(ValueError):
            raw_values(state)
        for value in rng.normal(size=270):
            state.update(value)
            row = raw_values(state)
            self.assertEqual(tuple(row),RAW)
            self.assertEqual(row["I"],sum(row[k] for k in RAW[4:8])/4)
            for key in (*RAW[:4],*RAW[8:]):
                self.assertEqual(row[key],state.last_blocks[key])

    def test_stage3_prefix_truncation_exact_resume_and_guarded_handshake(self):
        rng = np.random.default_rng(30)
        h,online = rng.normal(size=170),rng.normal(size=45)
        for name in INPUT_LISTS:
            model,_,_ = fit_fold(fixture(),0,name)
            loaded = Model.loads(model.dumps(),native=name=="E13")
            np.testing.assert_array_equal(model.predict(fixture()),loaded.predict(fixture()))
            state = Stage3Detector(h,model)
            q = state.q
            first = [state.update(x) for x in online[:23]]
            resumed = Stage3Detector.loads(state.dumps(),model)
            self.assertEqual(state.dumps(),resumed.dumps())
            tail = [state.update(x) for x in online[23:]]
            np.testing.assert_array_equal(tail,[resumed.update(x) for x in online[23:]])
            self.assertEqual(state.dumps(),resumed.dumps())
            self.assertEqual(q,state.q)
            for n in (1,8,32,45):
                fresh = Stage3Detector(h,model)
                np.testing.assert_array_equal([fresh.update(x) for x in online[:n]],(first+tail)[:n])
            rec = Series("synthetic",h,online,np.zeros(len(online),dtype=int),np.arange(len(online)),-1)
            np.testing.assert_array_equal(list(replay([rec],infer_fn=lambda ds:infer(ds,model)))[0][1],first+tail)

    def test_stage3_e13_parameters_native_bins_weights_iterations_and_split_repeatability(self):
        rng = np.random.default_rng(313)
        n = 50000
        frame = pd.DataFrame(rng.uniform(size=(n,4)),columns=RAW[:4])
        frame["fold"] = np.arange(n)%5
        frame["target"] = (frame.I+frame.R*.1 > .55).astype(int)
        frame["time_online"] = np.arange(n)%103
        a,detail,audit = fit_fold(frame,0,"E13")
        b,_,_ = fit_fold(frame,0,"E13")
        np.testing.assert_array_equal(a.predict(frame),b.predict(frame))
        self.assertEqual(a.dumps(),b.dumps())
        self.assertEqual(a.tree.get_params(),TREE_PARAMS)
        self.assertNotIn("n_jobs",TREE_PARAMS)
        self.assertEqual(a.tree.n_iter_,100)
        self.assertFalse(a.tree.do_early_stopping_)
        self.assertEqual(detail["sample_weight_sha256"],audit["weights_sha256"])
        self.assertAlmostEqual(detail["sample_weight_sum"],1.,places=12)
        self.assertGreater(max(detail["tree_depths"]),0)
        changed = frame.copy()
        changed.loc[changed.fold==0,list(RAW[:4])] = 100000
        changed.loc[changed.fold==0,"target"] = 1-changed.loc[changed.fold==0,"target"]
        c,_,_ = fit_fold(changed,0,"E13")
        for one,two in zip(a.tree._bin_mapper.bin_thresholds_,c.tree._bin_mapper.bin_thresholds_):
            np.testing.assert_array_equal(one,two)
        self.assertEqual(a.dumps(),c.dumps())
        restored = Model.loads(a.dumps(),native=True)
        np.testing.assert_array_equal(a.predict(frame),restored.predict(frame))
        # Native fitting with the exact normalized weights independently reproduces the estimator.
        train = frame.loc[frame.fold!=0]
        eligible,w,_ = training_weights(train.time_online,train.target)
        with threadpool_limits(limits=1):
            native = HistGradientBoostingClassifier(**TREE_PARAMS).fit(a.pre.transform(train.loc[eligible]),train.loc[eligible,"target"],sample_weight=w)
            np.testing.assert_array_equal(a.predict(frame),native.predict_proba(a.pre.transform(frame))[:,1])

    def test_stage3_implied_coefficient_diagnostics_arithmetic(self):
        for name in ("E10","E11","E12"):
            model,_,_ = fit_fold(fixture(),0,name)
            detail = coefficient_diagnostics(model)
            if name=="E10":
                self.assertEqual(len(detail["implied_innovation_scale_slopes"]),4)
            if name=="E11":
                self.assertEqual(len(detail["implied_rank_slopes_location_energy"]),2)
            if name=="E12":
                self.assertEqual(list(detail["implied_raw_block_slopes_by_age"]),["32","128","256","512"])
