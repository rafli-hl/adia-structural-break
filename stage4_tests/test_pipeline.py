"""Stage-4 evaluator/admission tests with explicitly synthetic fixtures."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import duckdb
import numpy as np
import pandas as pd
from sbrt.data import sha256, iter_series
from sbrt.folds import create_v2, load_v2, STRATA
from sbrt.replay import replay
from sbrt.stage2 import score_outputs, validate_coverage
from sbrt.provenance import ROOT, finalize_artifacts, verify_artifacts
from sbrt.comparison import bootstrap_multiplicities, sorted_age, weighted_age_terms
from stage4_research.contract import IDS, RAW, CONFIG, STANDALONE, E08_HASH, CV_HASH
from stage4_research.provenance import protected, source, evaluator
from stage4_research.pipeline import gate, winner, Exporter, subgroup_diagnostics, absorption
from stage4_research.models import fit_fold
sys.path.insert(0,str(ROOT/"tests"))
from test_stage2_pipeline import integration_cache


class PipelineTests(unittest.TestCase):
    def test_g4_all_exact_boundaries_family_four_and_one_percent_quantile(self):
        delta=dict(pooled=.002,folds=[.001]*4+[0.],combined=dict(ge129=0.,le64=-.005))
        self.assertTrue(gate(delta,-.005,replicates=np.ones(999))["advances"])
        self.assertFalse(gate(delta,-.005,replicates=np.zeros(999))["advances"])
        self.assertFalse(gate(delta,-.00500001,replicates=np.ones(999))["advances"])
        self.assertFalse(gate(delta,0,valid=False,replicates=np.ones(999))["advances"])
        draws=np.r_[np.full(15,-.001),np.full(984,.001)]
        self.assertGreater(np.quantile(draws,.025),0)
        result=gate(delta,0,replicates=draws)
        self.assertEqual(result["Q_0_01"],float(np.quantile(draws,.01,method="linear")))
        self.assertFalse(result["advances"]); self.assertEqual(result["family_size"],4)
        for modified in (dict(delta,pooled=.001999),dict(delta,folds=[.001]*3+[0,0]),
                dict(delta,combined=dict(ge129=-1e-9,le64=0)),dict(delta,combined=dict(ge129=0,le64=-.0050001))):
            self.assertFalse(gate(modified,0)["bootstrap_required"])
        with self.assertRaises(ValueError): gate(delta,0,replicates=np.ones(998))

    def test_same_six_strata_whole_series_paired_multiplicities(self):
        manifest=pd.DataFrame(dict(stratum=np.repeat(STRATA,4)))
        a=bootstrap_multiplicities(manifest,999,20260920); b=bootstrap_multiplicities(manifest,999,20260920)
        np.testing.assert_array_equal(a,b)
        for s in STRATA: np.testing.assert_array_equal(a[manifest.stratum==s].sum(axis=0),4)
        labels=np.arange(24)%2
        x=sorted_age(np.arange(24),labels,np.arange(24)); y=sorted_age(np.arange(24),labels,-np.arange(24))
        np.testing.assert_array_equal(weighted_age_terms(x,a)[1],weighted_age_terms(y,a)[1])

    def test_winner_rule_never_selects_standalone(self):
        def candidate(name,score,count,qualifies=True):
            return dict(experiment_id=name,pooled_oof_ts_auc=score,model_input_count=count,status="valid",gate=dict(advances=qualifies))
        self.assertEqual(winner({}),"E08")
        values={n:candidate(n,.6,6 if n=="E17" else 5) for n in IDS}
        self.assertEqual(winner(values),"E14")
        values["E17"]["pooled_oof_ts_auc"]=.61; self.assertEqual(winner(values),"E17")
        for v in values.values(): v["gate"]["advances"]=False
        self.assertEqual(winner(values),"E08")
        self.assertEqual(set(STANDALONE.values()),{"J_s","J_f","J_r","P_s"})
        self.assertFalse(CONFIG["standalone_selectable"])

    def test_protected_research_production_and_actual_oof_coverage(self):
        self.assertTrue(protected()["unchanged"])
        self.assertEqual(sha256(ROOT/"stage2_phase2/E08/oof.parquet"),E08_HASH)
        self.assertEqual(sha256(ROOT/"local_cache/cv2/folds.parquet"),CV_HASH)
        manifest,_=load_v2(ROOT/"local_cache",ROOT/"local_cache/cv2/folds.parquet")
        self.assertEqual(len(manifest),10000); self.assertEqual(int(manifest.online_length.sum()),5036517)
        original=sha256(ROOT/"local_cache/data.duckdb")
        with tempfile.TemporaryDirectory() as tmp:
            con=evaluator(tmp)
            try:
                self.assertEqual(validate_coverage(con,ROOT/"stage2_phase2/E08/oof.parquet",manifest),5036517)
                with self.assertRaises(duckdb.InvalidInputException):
                    con.execute("CREATE TABLE trusted.forbidden_stage4_test AS SELECT 1")
            finally: con.close()
        self.assertEqual(sha256(ROOT/"local_cache/data.duckdb"),original)
        self.assertIn("stage4_research/scale.py",source()["files"])

    def test_tampered_artifacts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"data.txt").write_text("explicit synthetic artifact")
            finalize_artifacts(root); verify_artifacts(root)
            (root/"data.txt").write_text("changed")
            with self.assertRaises(ValueError): verify_artifacts(root)

    def test_causal_export_train_complete_oof_and_official_scoring_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp); con=integration_cache(root)
            try:
                manifest,_=create_v2(root/"cache",root/"cv2"); lookup=dict(zip(manifest.id,manifest.fold))
                observer=Exporter(); frames=[]
                for s,values,_ in replay(iter_series(con),infer_fn=observer.infer):
                    rows,diag,detail=observer.completed.popleft()
                    frame=pd.DataFrame(rows,columns=RAW)
                    frames.append(frame.assign(id=s.id,time=s.time,time_online=np.arange(len(values)),target=s.target,fold=lookup[s.id]))
                    self.assertEqual(len(diag),len(values)); self.assertIn("robust",detail["scales"])
                frame=pd.concat(frames,ignore_index=True)
                for name in IDS:
                    p=np.full(len(frame),np.nan)
                    for fold in range(5):
                        model,_,_=fit_fold(frame,fold,name); mask=frame.fold==fold
                        p[mask]=model.predict(frame[mask])
                    out=root/name; out.mkdir()
                    result=frame.assign(prediction=p); result.to_parquet(out/"oof.parquet",index=False)
                    self.assertEqual(validate_coverage(con,out/"oof.parquet",manifest),len(frame))
                    metrics=score_outputs(con,out/"oof.parquet",manifest,out)
                    self.assertEqual(len(metrics["fold_ts_auc"]),5)
                    self.assertTrue(0<=metrics["pooled_oof_ts_auc"]<=1)
                bad=result.iloc[:-1]; bad.to_parquet(out/"bad.parquet",index=False)
                with self.assertRaises(ValueError): validate_coverage(con,out/"bad.parquet",manifest)
            finally: con.close()

    def test_subgroups_not_confused_with_raw_features_and_absorption_bins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"features").mkdir(); (root/"E14").mkdir(); out=root/"report"; out.mkdir()
            frame=pd.DataFrame([dict(id=f"s{i}",time_online=t,target=i%2,prediction=.8 if i%2 else .2,R=999.) for i in range(6) for t in range(260)])
            frame.to_parquet(root/"E14/oof.parquet",index=False); frame.assign(prediction=.5).to_parquet(root/"parent.parquet",index=False)
            pd.DataFrame(dict(id=[f"s{i}" for i in range(6)],group=np.repeat(["le0.1","gt0.1","unsupported"],2))).to_parquet(root/"features/historical_context.parquet",index=False)
            diag=frame[["id","time_online"]].assign(e_squared=4.,v_s=2.,v_f=4.,v_r=1.)
            diag.to_parquet(root/"features/evaluator_scale.parquet",index=False)
            con=duckdb.connect()
            try:
                result=subgroup_diagnostics(con,root,"E14",out,root/"parent.parquet")
                for group in result["subsets"].values():
                    self.assertEqual(group["series"],2); self.assertEqual(group["pair_weight"],260)
                    self.assertEqual(group["ts_auc"],1.); self.assertEqual(group["reference"],.5)
                idx=pd.DataFrame(dict(id=[f"s{i}" for i in range(6)],tau_index=[-1,0,-1,31,-1,259]))
                con.register("idx",idx); con.execute("CREATE TABLE series_index AS SELECT * FROM idx")
                report=absorption(con,root)
                self.assertEqual(report["summary"]["s"]["1-32"]["series"],3)
                self.assertEqual(report["summary"]["s"]["1-32"]["observations"],65)
                self.assertEqual(report["summary"]["s"]["257+"]["series"],1)
                self.assertEqual(report["summary"]["s"]["257+"]["observations"],4)
                self.assertEqual(report["summary"]["s"]["1-32"]["statistics"]["energy_over_predictive"]["median"],2.)
            finally: con.close()
