"""Synthetic integration fixtures exercise contracts, never research selection."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from sbrt.data import prepare, sha256
from sbrt.profile import run_profile
from sbrt.backtest import run_backtest
from sbrt.folds import create_v2, load_v2, STRATA, stratify
from sbrt.experiments import load_spec, DEFAULT_SPEC, execution_order
from sbrt.provenance import write_json, finalize_artifacts, verify_artifacts
from sbrt.stage2 import rescore_e01, run_one, validate_coverage, load_result, difference
from sbrt.comparison import (gate_conditions, sorted_age, weighted_age_terms,
                             bootstrap_multiplicities, run_comparisons, paired_bootstrap, side_diagnostics)
from sbrt.training_weights import metric_training_weights
from sbrt.metric import ts_auc


def small_manifest():
    rows = []
    for j, tau in enumerate((-1, 30, 100, 200, 400, 550)):
        for k in range(5):
            sid = f"fixture_{j}_{k}"
            rows.append(dict(id=sid, group=sid, tau_index=tau, online_length=560,
                             historical_length=160, has_break=int(tau>=0)))
    return pd.DataFrame(rows)


def manifest_cache(path):
    path.mkdir(parents=True, exist_ok=True)
    frame = small_manifest()
    frame.to_parquet(path/"series_manifest.parquet",index=False)
    write_json(path/"profile_status.json",dict(label_alignment="pass",unresolved_repeated_block_pairs=0))
    write_json(path/"sources.json",dict(test_fixture=True))
    return frame


def integration_cache(root):
    """Actual parquet/loader/profile path, with explicitly synthetic values."""
    rng = np.random.default_rng(450)
    rows, labels, metadata = [], [], []
    for rec in small_manifest().itertuples():
        h = rng.normal(size=160)
        o = rng.normal(size=560)
        if rec.tau_index >= 0:
            o[rec.tau_index:] = .3 + 1.3*o[rec.tau_index:]
        rows.extend((rec.id,t,float(x),1) for t,x in enumerate(h))
        rows.extend((rec.id,len(h)+t,float(x),2) for t,x in enumerate(o))
        labels.extend((rec.id,len(h)+t,int(rec.tau_index>=0 and t>=rec.tau_index)) for t in range(len(o)))
        metadata.append((rec.id,rec.tau_index,rec.tau_index+len(h) if rec.tau_index>=0 else -1))
    pd.DataFrame(rows,columns=["id","time","value","period"]).to_parquet(root/"X.parquet",index=False)
    pd.DataFrame(labels,columns=["id","time","target"]).to_parquet(root/"y.parquet",index=False)
    pd.DataFrame(metadata,columns=["id","tau_index","tau"]).to_parquet(root/"index.parquet",index=False)
    con,source = prepare(root/"X.parquet",root/"y.parquet",root/"index.parquet",root/"cache")
    run_profile(con,source,root/"cache",root/"diagnostics",near_cap=0)
    return con


class FoldContracts(unittest.TestCase):
    def test_exact_strata_boundaries(self):
        frame = pd.DataFrame(dict(tau_index=[-1,0,63,64,127,128,255,256,511,512],
                                  has_break=[0]+[1]*9,online_length=[999]*10))
        self.assertEqual(list(stratify(frame)),[STRATA[0],STRATA[1],STRATA[1],STRATA[2],STRATA[2],
                                               STRATA[3],STRATA[3],STRATA[4],STRATA[4],STRATA[5]])

    def test_immutable_deterministic_fold_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            frame=manifest_cache(root/"cache")
            # A CV-v1 sentinel must survive all CV-v2 actions.
            frame.assign(fold=0).to_parquet(root/"cache/folds.parquet",index=False)
            before=sha256(root/"cache/folds.parquet")
            a,meta=create_v2(root/"cache",root/"cv2")
            b,other=create_v2(root/"cache",root/"another_cv2")
            self.assertTrue(a.equals(b))
            self.assertEqual(meta,other)
            self.assertEqual(before,sha256(root/"cache/folds.parquet"))
            for s,counts in meta["stratum_counts_by_fold"].items():
                self.assertEqual(counts,[1]*5)
            for f in range(5):
                self.assertFalse(set(a[a.fold==f].id)&set(a[a.fold!=f].id))
            self.assertEqual(meta,load_v2(root/"cache",root/"cv2/folds.parquet")[1])
            with self.assertRaises(ValueError):
                create_v2(root/"cache",root/"cv2",seed=123)
            altered=a.copy()
            altered.loc[0,"fold"]=(int(altered.loc[0,"fold"])+1)%5
            altered.to_parquet(root/"cv2/folds.parquet",index=False)
            with self.assertRaisesRegex(ValueError,"modified"):
                load_v2(root/"cache",root/"cv2/folds.parquet")
            with self.assertRaises(ValueError):
                create_v2(root/"cache",root/"cv2")

    def test_manifest_rejects_settings_source_and_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            frame=manifest_cache(root/"cache")
            create_v2(root/"cache",root/"cv2")
            settings=json.loads((root/"cv2/fold_settings.json").read_text())
            settings["seed"]=1
            write_json(root/"cv2/fold_settings.json",settings)
            with self.assertRaises(ValueError):
                load_v2(root/"cache",root/"cv2/folds.parquet")
            frame.loc[1,"group"]=frame.loc[0,"group"]
            frame.to_parquet(root/"cache/series_manifest.parquet",index=False)
            with self.assertRaisesRegex(ValueError,"groups"):
                create_v2(root/"cache",root/"new")

    def test_frozen_config_and_no_e08_execution(self):
        spec=load_spec()
        self.assertEqual(execution_order(["E07"],spec),["E02","E03","E04","E05","E06","E07"])
        with self.assertRaises(ValueError):
            execution_order(["E08"],spec)
        with tempfile.TemporaryDirectory() as tmp:
            spec["windows"][0]=16
            write_json(Path(tmp)/"bad.json",spec)
            with self.assertRaisesRegex(ValueError,"frozen"):
                load_spec(Path(tmp)/"bad.json")


class ComparisonContracts(unittest.TestCase):
    def test_metric_weights_age_class_totals_and_cv_isolation(self):
        frame=pd.DataFrame(dict(time_online=[0,0,0,1,1,1,2,2,0,1],
                                target=[0,0,1,0,1,1,0,0,1,0],fold=[0]*8+[1,1]))
        indices,w=metric_training_weights(frame,heldout_fold=1)
        self.assertEqual(set(indices),set(range(6)))
        self.assertAlmostEqual(w.sum(),1)
        selected=frame.loc[indices].assign(weight=w)
        for t,g in selected.groupby("time_online"):
            self.assertAlmostEqual(g.weight.sum(),.5)
            self.assertAlmostEqual(g[g.target==0].weight.sum(),.25)
            self.assertAlmostEqual(g[g.target==1].weight.sum(),.25)
        altered=frame.copy()
        altered.loc[altered.fold==1,"target"]=1-altered.loc[altered.fold==1,"target"]
        other_indices,other_w=metric_training_weights(altered,heldout_fold=1)
        np.testing.assert_array_equal(indices,other_indices)
        np.testing.assert_array_equal(w,other_w)

    def test_gate_exact_boundaries(self):
        gate=load_spec()["gate"]
        delta=dict(pooled=.002,folds=[.001,.001,.001,.001,0],combined=dict(ge129=0.,le64=-.005))
        self.assertTrue(all(gate_conditions(delta,gate).values()))
        delta["pooled"]=.001999
        self.assertFalse(gate_conditions(delta,gate)["pooled_delta"])
        delta["folds"]=[1,1,1,0,0]
        self.assertFalse(gate_conditions(delta,gate)["positive_folds"])

    def test_series_bootstrap_recomputes_pairs_and_ties(self):
        frame=pd.DataFrame(dict(id=["0","1","2","3","0","1","3"],
                                time_online=[0,0,0,0,1,1,1],target=[0,1,1,0,1,1,0],
                                prediction=[.2,.2,.8,.7,.5,.5,.1]))
        multiplicity=np.array([2,1,3,1])
        numerator=denominator=0.
        repeated=[]
        for t,g in frame.groupby("time_online"):
            prepared=sorted_age(g.id.astype(int),g.target,g.prediction)
            n,d=weighted_age_terms(prepared,multiplicity)
            numerator+=n[0]
            denominator+=d[0]
        for row in frame.itertuples():
            for k in range(multiplicity[int(row.id)]):
                repeated.append(dict(id=f"{row.id}:{k}",time_online=row.time_online,target=row.target,prediction=row.prediction))
        self.assertAlmostEqual(numerator/denominator,ts_auc(pd.DataFrame(repeated))[0],places=15)
        self.assertNotEqual(denominator,sum(int(g.target.sum())*int((1-g.target).sum()) for _,g in frame.groupby("time_online")))

    def test_bootstrap_draws_are_series_level_stratified_and_deterministic(self):
        frame=small_manifest()
        frame["stratum"]=stratify(frame)
        a=bootstrap_multiplicities(frame,19,20260920)
        np.testing.assert_array_equal(a,bootstrap_multiplicities(frame,19,20260920))
        for s in STRATA:
            np.testing.assert_array_equal(a[frame.stratum==s].sum(axis=0),np.full(19,5))

    def test_artifact_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/"result.json",dict(value=1))
            finalize_artifacts(root)
            verify_artifacts(root)
            write_json(root/"result.json",dict(value=2))
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                verify_artifacts(root)


class Stage2Integration(unittest.TestCase):
    def test_complete_statistical_pipeline_and_oof_coverage(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            con=integration_cache(root)
            try:
                run_backtest(con,root/"cache",root/"e01",detectors=["squared"])
                manifest,_=create_v2(root/"cache",root/"cv2")
                folds=root/"cv2/folds.parquet"
                baseline=rescore_e01(root/"cache",folds,DEFAULT_SPEC,root/"e01/oof_squared.parquet",root/"results/E01_squared")
                self.assertEqual(baseline["pooled_invariance_absolute_delta"],0.)
                predictions={}
                for name in ("E02","E03","E04","E05","E06","E07"):
                    result=run_one(root/"cache",folds,DEFAULT_SPEC,name,root/"results")
                    self.assertEqual(result["status"],"valid")
                    self.assertEqual(result["scored_rows"],30*560)
                    self.assertEqual(load_result(root/"results"/name)["prediction_artifact_sha256"],result["prediction_artifact_sha256"])
                    frame=pd.read_parquet(root/"results"/name/"oof.parquet")
                    self.assertFalse(frame[["id","time_online"]].duplicated().any())
                    self.assertEqual(result["pooled_oof_ts_auc"],ts_auc(frame)[0])
                    predictions[name]=frame
                    self.assertEqual(run_one(root/"cache",folds,DEFAULT_SPEC,name,root/"results"),result)
                np.testing.assert_array_equal(predictions["E05"].R,predictions["E03"].prediction)
                np.testing.assert_array_equal(predictions["E05"].I,predictions["E04"].prediction)
                for w,counts in result["support_counts"].items():
                    self.assertEqual(sum(counts.values()),30*560)
                    self.assertEqual(counts["insufficient_pairs"],30*8)
                # A tiny replicate count is test-only; production registry remains 999.
                test_gate=dict(load_spec()["gate"],bootstrap_replicates=3)
                bootstrap=paired_bootstrap(con,root/"results/E02/oof.parquet",root/"results/E02/oof.parquet",manifest,test_gate)
                self.assertEqual(bootstrap["interval_95"],[0.,0.])
                self.assertEqual(bootstrap["delta_replicates"],[0.,0.,0.])
                (root/"side").mkdir()
                side=side_diagnostics(con,root/"results",root/"side")
                self.assertEqual(sum(v["series"] for v in side["historical_ar_mse_subsets"].values()),30)
                self.assertEqual(set(side["rank_subblocks"]),{"P","P_location","P_energy"})
                bad=predictions["E02"].copy()
                bad.loc[0,"time"]=999999
                bad.to_parquet(root/"bad.parquet",index=False)
                with self.assertRaisesRegex(ValueError,"mismatch"):
                    validate_coverage(con,root/"bad.parquet",manifest)
                bad=predictions["E02"].iloc[:-1]
                bad.to_parquet(root/"bad.parquet",index=False)
                with self.assertRaisesRegex(ValueError,"count"):
                    validate_coverage(con,root/"bad.parquet",manifest)
            finally:
                con.close()


if __name__ == "__main__":
    unittest.main()
