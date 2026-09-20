"""Stage-3 admission and evaluator contracts; synthetic tests are explicit."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from sbrt.stage3 import load_spec, protected, gate, winner, IDS, ExportObserver, subgroup_diagnostics, check_evaluation_patch
from sbrt.stage3_models import fit_fold
from sbrt.stage3_features import RAW
from sbrt.provenance import ROOT, finalize_artifacts, verify_artifacts
from sbrt.data import sha256, iter_series
from sbrt.stage2 import validate_coverage
from sbrt.replay import replay
from sbrt.folds import create_v2, load_v2, STRATA
from sbrt.comparison import bootstrap_multiplicities, sorted_age, weighted_age_terms
from test_stage3_numerics import fixture
from test_stage2_pipeline import integration_cache


class Stage3Pipeline(unittest.TestCase):
    def test_stage3_evaluator_correction_preserves_all_model_and_gate_code(self):
        audit = check_evaluation_patch(ROOT/"stage3_results")
        self.assertTrue(audit["model_feature_gate_bootstrap_code_unchanged"])
        self.assertEqual(audit["original_source_sha256"],"756e756a46f975b0bd8c4f1885083bd080d3968992c562cf13061fec8320170b")

    def test_stage3_historical_r_subgroup_query_not_confused_with_raw_R(self):
        import duckdb
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"E09").mkdir()
            (root/"features").mkdir()
            out = root/"comparisons"
            out.mkdir()
            records = [dict(id=f"s{i}",time_online=t,target=i%2,prediction=.8 if i%2 else .2,R=99.)
                       for i in range(8) for t in range(3)]
            frame = pd.DataFrame(records)
            frame.to_parquet(root/"E09/oof.parquet",index=False)
            frame.assign(prediction=.5).to_parquet(root/"parent.parquet",index=False)
            pd.DataFrame(dict(id=[f"s{i}" for i in range(8)],r=[.8,.8,.95,1.,1.1,1.2,None,None],
                             unsupported=[False]*6+[True]*2)).to_parquet(root/"features/context.parquet",index=False)
            con = duckdb.connect()
            try:
                result = subgroup_diagnostics(con,root,out,root/"parent.parquet")
                for subgroup in result["subsets"].values():
                    self.assertEqual(subgroup["series"],2)
                    self.assertEqual(subgroup["pair_weight"],3)
                    self.assertEqual(subgroup["ts_auc"],1.)
                    self.assertEqual(subgroup["reference"],.5)
                    self.assertEqual(subgroup["delta"],.5)
            finally:
                con.close()

    def test_stage3_immutable_spec_cv2_e08_and_all_protected_artifacts(self):
        spec = load_spec()
        self.assertTrue(protected(ROOT/"stage3_results")["unchanged"])
        self.assertEqual(sha256(ROOT/"stage2_phase2/E08/oof.parquet"),spec["parent_oof_sha256"])
        manifest,_ = load_v2(ROOT/"local_cache",ROOT/"local_cache/cv2/folds.parquet")
        self.assertEqual(sha256(ROOT/"local_cache/cv2/folds.parquet"),spec["cv2_sha256"])
        self.assertEqual(len(manifest),10000)
        self.assertEqual(int(manifest.online_length.sum()),5036517)
        self.assertEqual(spec["execution_order"],list(IDS))
        self.assertTrue(all(v["parent"]=="E08" for v in spec["experiments"].values()))

    def test_stage3_five_independent_oof_coverage_and_fold_isolation(self):
        frame = fixture()
        for name in IDS:
            predictions = np.full(len(frame),np.nan)
            count = np.zeros(len(frame),dtype=int)
            for fold in range(5):
                model,_,_ = fit_fold(frame,fold,name)
                mask = frame.fold==fold
                self.assertFalse(set(frame.loc[mask,"id"])&set(frame.loc[~mask,"id"]))
                predictions[mask] = model.predict(frame.loc[mask])
                count[mask] += 1
            np.testing.assert_array_equal(count,1)
            self.assertTrue(np.isfinite(predictions).all())
            self.assertTrue(((predictions>=0)&(predictions<=1)).all())

    def test_stage3_causal_export_and_oof_against_synthetic_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            con = integration_cache(root)
            try:
                manifest,_ = create_v2(root/"cache",root/"cv2")
                observer = ExportObserver()
                frames = []
                lookup = dict(zip(manifest.id,manifest.fold))
                for series,predictions,_ in replay(iter_series(con),detector="E07",state_observer=observer):
                    detail = observer.completed.popleft()
                    frame = pd.DataFrame(detail["rows"],columns=RAW)
                    frame = frame.assign(id=series.id,time=series.time,time_online=np.arange(len(predictions)),
                        target=series.target,fold=lookup[series.id],q=detail["q"])
                    frames.append(frame)
                frame = pd.concat(frames,ignore_index=True)
                p = np.full(len(frame),np.nan)
                for fold in range(5):
                    model,_,_ = fit_fold(frame,fold,"E10")
                    mask = frame.fold==fold
                    p[mask] = model.predict(frame.loc[mask])
                frame["prediction"] = p
                path = root/"oof.parquet"
                frame.to_parquet(path,index=False)
                self.assertEqual(validate_coverage(con,path,manifest),len(frame))
                pd.concat([frame,frame.iloc[:1]]).to_parquet(path,index=False)
                with self.assertRaises(ValueError):
                    validate_coverage(con,path,manifest)
                bad = frame.copy()
                bad.loc[0,"fold"] = (bad.loc[0,"fold"]+1)%5
                bad.to_parquet(path,index=False)
                with self.assertRaises(ValueError):
                    validate_coverage(con,path,manifest)
            finally:
                con.close()

    def test_stage3_g3_exact_boundaries_and_one_percent_guard(self):
        settings = load_spec()["gate"]
        delta = dict(pooled=.002,folds=[.001]*4+[0.],combined=dict(ge129=0.,le64=-.005))
        self.assertTrue(gate(delta,-.005,settings,np.full(999,.001))["advances"])
        self.assertFalse(gate(delta,-.005,settings,np.zeros(999))["advances"])
        self.assertFalse(gate(delta,-.00500001,settings,np.full(999,.001))["advances"])
        draws = np.r_[np.full(15,-.001),np.full(984,.001)]
        self.assertGreater(np.quantile(draws,.025),0)
        result = gate(delta,0,settings,draws)
        self.assertEqual(result["Q_0_01"],float(np.quantile(draws,.01,method="linear")))
        self.assertFalse(result["advances"])
        self.assertEqual(result["family_size"],5)
        for modified in (dict(delta,pooled=.001999),dict(delta,folds=[.001]*3+[0.,0.]),
                dict(delta,combined=dict(ge129=-1e-9,le64=0))):
            self.assertFalse(gate(modified,0,settings,np.ones(999))["bootstrap_required"])
        with self.assertRaises(ValueError):
            gate(delta,0,settings,np.ones(998))

    def test_stage3_same_six_stratum_paired_multiplicities_and_pair_weights(self):
        manifest = pd.DataFrame({"stratum":np.repeat(STRATA,4)})
        a = bootstrap_multiplicities(manifest,999,20260920)
        b = bootstrap_multiplicities(manifest,999,20260920)
        np.testing.assert_array_equal(a,b)
        for s in STRATA:
            np.testing.assert_array_equal(a[manifest.stratum==s].sum(axis=0),4)
        labels = np.arange(24)%2
        x = sorted_age(np.arange(24),labels,np.arange(24))
        y = sorted_age(np.arange(24),labels,-np.arange(24))
        np.testing.assert_array_equal(weighted_age_terms(x,a)[1],weighted_age_terms(y,a)[1])

    def test_stage3_winner_fixed_family_and_tie_breaking(self):
        def candidate(name,score,count,qualifies=True):
            return dict(experiment_id=name,pooled_oof_ts_auc=score,model_input_count=count,status="valid",gate=dict(advances=qualifies))
        self.assertEqual(winner({}),"E08")
        values = dict(E09=candidate("E09",.6,6),E10=candidate("E10",.6,7),E11=candidate("E11",.6,5))
        self.assertEqual(winner(values),"E11")
        values["E09"]["model_input_count"] = 5
        self.assertEqual(winner(values),"E09")
        values["E10"]["pooled_oof_ts_auc"] = .61
        self.assertEqual(winner(values),"E10")
        values["E10"]["gate"]["advances"] = False
        self.assertEqual(winner(values),"E09")

    def test_stage3_artifact_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"model.bin").write_bytes(b"explicit test artifact")
            finalize_artifacts(root)
            verify_artifacts(root)
            (root/"model.bin").write_bytes(b"changed test artifact")
            with self.assertRaises(ValueError):
                verify_artifacts(root)
