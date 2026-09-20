"""Synthetic isolation, deterministic fitting, nesting and guarded replay tests."""
import json
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sbrt.logistic import training_weights, objective
from sbrt.data import Series
from sbrt.replay import replay
from stage4_research.models import fit_fold as fit_e14
from stage5_research.contract import IDS, RAW, BASE, INPUTS, EXTRA
from stage5_research.models import Preprocessing, Model, fit_fold
from stage5_research.features import Detector, infer
from test_numerics import fixture


def frame_fixture():
    rng=np.random.default_rng(20260925); rows=[]
    for i in range(50):
        for t in range(30+i%5):
            rows.append(dict(id=f"test_{i:03}",fold=i%5,time=200+t,time_online=t,
                target=int(i%2 and t>=i%12),**{k:float(rng.uniform()) for k in RAW}))
    return pd.DataFrame(rows)


class Models(unittest.TestCase):
    def test_training_weights_identity_eligibility_and_training_only(self):
        f=frame_fixture(); train=f.loc[f.fold!=0]
        mask,w,a=training_weights(train.time_online,train.target)
        self.assertLessEqual(max(a["maximum_absolute_errors"].values()),1e-12)
        self.assertAlmostEqual(w.sum(),1.,places=12)
        modified=f.copy(); modified.loc[modified.fold==0,"target"]=1
        other=modified.loc[modified.fold!=0]
        mask2,w2,a2=training_weights(other.time_online,other.target)
        np.testing.assert_array_equal(mask,mask2); np.testing.assert_array_equal(w,w2); self.assertEqual(a,a2)

    def test_exact_input_allowlists_dimensions_and_metadata_excluded(self):
        self.assertEqual([len(INPUTS[k]) for k in IDS],[10,7,9,7])
        f=frame_fixture(); mask,w,_=training_weights(f.time_online,f.target)
        for name in IDS:
            pre,x=Preprocessing.fit(name,f.loc[mask],w)
            changed=f.copy()
            for k in ("time_online","target","fold","id","tau","partition","eta","horizon"):
                changed[k]="forbidden"
            np.testing.assert_array_equal(pre.transform(f),pre.transform(changed))
            allowed=f.loc[:,list(BASE)+list(EXTRA[name])]
            np.testing.assert_array_equal(pre.transform(f),pre.transform(allowed))
            with self.assertRaises(KeyError): pre.transform(allowed.drop(columns=EXTRA[name][0]))
            self.assertEqual(x.shape[1],len(INPUTS[name]))

    def test_training_only_preprocessing_and_zero_variance(self):
        f=frame_fixture(); f["M1"]=.3
        train=f.loc[f.fold!=0]; mask,w,_=training_weights(train.time_online,train.target)
        pre,x=Preprocessing.fit("E18",train.loc[mask],w)
        self.assertEqual(pre.std[0],0.); self.assertTrue((x[:,5]==0).all())
        held=f.loc[f.fold==0].copy(); held["M1"]=100.
        self.assertTrue((pre.transform(held)[:,5]==0).all())
        for j,k in enumerate(EXTRA["E18"][1:],1):
            values=train.loc[mask,k].to_numpy()
            self.assertAlmostEqual(pre.mean[j],float(w@values/w.sum()),places=14)
        altered=f.copy(); altered.loc[altered.fold==0,list(RAW)]=1e6
        b,_,_=fit_fold(f,0,"E18"); c,_,_=fit_fold(altered,0,"E18")
        self.assertEqual(b.dumps(),c.dumps())

    def test_heldout_labels_ages_and_keys_do_not_change_model(self):
        f=frame_fixture(); changed=f.copy()
        changed.loc[changed.fold==0,"target"]=1-changed.loc[changed.fold==0,"target"]
        changed.loc[changed.fold==0,"time_online"]=99999
        changed.loc[changed.fold==0,"id"]="changed-evaluator-key"
        for name in IDS:
            a,_,wa=fit_fold(f,0,name); b,_,wb=fit_fold(changed,0,name)
            self.assertEqual(a.dumps(),b.dumps()); self.assertEqual(wa,wb)

    def test_deterministic_model_bytes_save_load_and_finite_outputs(self):
        f=frame_fixture()
        for name in IDS:
            a,da,_=fit_fold(f,0,name); b,db,_=fit_fold(f,0,name)
            self.assertEqual(a.dumps(),b.dumps()); self.assertEqual(a.dumps(),Model.loads(a.dumps()).dumps())
            np.testing.assert_array_equal(a.predict(f),b.predict(f))
            values=a.predict(f); self.assertTrue(np.isfinite(values).all()); self.assertTrue(((values>=0)&(values<=1)).all())
            self.assertLessEqual(da["gradient_infinity_norm"],1e-6); self.assertTrue(da["optimizer_success"])
            self.assertEqual(da["iterations"],db["iterations"])
            bad=json.loads(a.dumps()); bad["preprocessing"]["inputs"].append("age")
            with self.assertRaises(ValueError): Model.loads(json.dumps(bad))

    def test_E14_zero_extension_prediction_objective_1e14(self):
        f=frame_fixture(); old,_,_=fit_e14(f,0,"E14")
        train=f.loc[f.fold!=0]; mask,w,_=training_weights(train.time_online,train.target); train=train.loc[mask]
        for name in IDS:
            model,_,_=fit_fold(f,0,name)
            self.assertEqual(model.pre.base.as_dict(),old.pre.as_dict())
            nested=Model(model.pre,old.intercept,np.r_[old.beta,np.zeros(len(EXTRA[name]))])
            np.testing.assert_allclose(nested.predict(f),old.predict(f),rtol=0,atol=1e-14)
            with threadpool_limits(limits=1):
                x=model.pre.transform(train); y=train.target.to_numpy(dtype=float)
                a=objective(np.r_[nested.intercept,nested.beta],x,y,w)[0]
                b=objective(np.r_[old.intercept,old.beta],old.pre.transform(train),y,w)[0]
            self.assertLessEqual(abs(a-b),1e-14)

    def test_E20_zero_trajectory_nested_E19_objective_prediction(self):
        f=frame_fixture(); old,_,_=fit_fold(f,0,"E19"); new,_,_=fit_fold(f,0,"E20")
        np.testing.assert_array_equal(old.pre.mean,new.pre.mean[:2]); np.testing.assert_array_equal(old.pre.std,new.pre.std[:2])
        nested=Model(new.pre,old.intercept,np.r_[old.beta,0.,0.])
        np.testing.assert_allclose(nested.predict(f),old.predict(f),rtol=0,atol=1e-14)
        train=f.loc[f.fold!=0]; mask,w,_=training_weights(train.time_online,train.target); train=train.loc[mask]
        y=train.target.to_numpy(dtype=float)
        with threadpool_limits(limits=1):
            a=objective(np.r_[nested.intercept,nested.beta],nested.pre.transform(train),y,w)[0]
            b=objective(np.r_[old.intercept,old.beta],old.pre.transform(train),y,w)[0]
        self.assertLessEqual(abs(a-b),1e-14)

    def test_E21_eta_one_preprocessing_model_and_objective_equivalence(self):
        f=frame_fixture(); f["L_plus_calibrated"]=f.L_plus; f["L_minus_calibrated"]=f.L_minus
        old,_,_=fit_fold(f,0,"E19"); new,_,_=fit_fold(f,0,"E21")
        np.testing.assert_array_equal(old.pre.mean,new.pre.mean); np.testing.assert_array_equal(old.pre.std,new.pre.std)
        np.testing.assert_array_equal(old.beta,new.beta); self.assertEqual(old.intercept,new.intercept)
        np.testing.assert_allclose(old.predict(f),new.predict(f),rtol=0,atol=1e-14)
        train=f.loc[f.fold!=0]; mask,w,_=training_weights(train.time_online,train.target); train=train.loc[mask]
        with threadpool_limits(limits=1):
            a=objective(np.r_[old.intercept,old.beta],old.pre.transform(train),train.target.to_numpy(),w)[0]
            b=objective(np.r_[new.intercept,new.beta],new.pre.transform(train),train.target.to_numpy(),w)[0]
        self.assertLessEqual(abs(a-b),1e-14)

    def test_prefix_truncation_order_handshake_complete_and_resume(self):
        f=frame_fixture(); h,o=fixture()
        records=[Series("one",h,o[:63],np.zeros(63),np.arange(63),-1),Series("two",h+.1,o[:38]*1.2,np.zeros(38),np.arange(38),-1)]
        for name in IDS:
            model,_,_=fit_fold(f,0,name)
            run=lambda rr:list(replay(rr,infer_fn=lambda ds:infer(ds,model)))
            full=run(records); rev=run(records[::-1])
            np.testing.assert_array_equal(full[0][1],rev[1][1]); np.testing.assert_array_equal(full[1][1],rev[0][1])
            self.assertEqual(sum(len(r[1]) for r in full),101)
            short=Series("one",h,o[:19],np.zeros(19),np.arange(19),-1)
            np.testing.assert_array_equal(full[0][1][:19],run([short])[0][1])
            suffix=np.r_[o[:19],np.full(25,999.)]
            changed=Series("one",h,suffix,np.zeros(44),np.arange(44),-1)
            np.testing.assert_array_equal(full[0][1][:19],run([changed])[0][1][:19])
            state=Detector(h,model)
            for x in o[:19]: state.update(x)
            loaded=Detector.loads(state.dumps(),Model.loads(model.dumps()))
            self.assertEqual(state.dumps(),loaded.dumps())
            np.testing.assert_array_equal([state.update(x) for x in o[19:63]],[loaded.update(x) for x in o[19:63]])
            self.assertEqual(state.dumps(),loaded.dumps())
            bad=Model(model.pre,model.intercept+.01,model.beta)
            with self.assertRaises(ValueError): Detector.loads(state.dumps(),bad)

    def test_complete_five_fold_synthetic_oof_once_per_key(self):
        f=frame_fixture()
        for name in IDS:
            output=np.full(len(f),np.nan); seen=np.zeros(len(f),dtype=int)
            for fold in range(5):
                model,_,_=fit_fold(f,fold,name)
                ix=f.fold==fold; output[ix]=model.predict(f.loc[ix]); seen[ix]+=1
                self.assertFalse(set(f.loc[ix,"id"])&set(f.loc[~ix,"id"]))
            self.assertTrue((seen==1).all()); self.assertTrue(np.isfinite(output).all())
