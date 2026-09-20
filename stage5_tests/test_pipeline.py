"""Partition, evaluator, immutable admission and full reference coverage tests."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from sbrt.data import sha256
from sbrt.folds import STRATA,load_v2
from sbrt.stage2 import validate_coverage
from sbrt.provenance import ROOT,verify_artifacts
from sbrt.comparison import bootstrap_multiplicities, sorted_age, weighted_age_terms
from stage5_research.contract import IDS,INPUTS,RAW,CONFIG,E14_HASH,CV_HASH,SPEC_HASH
from stage5_research.partition import expected,create
from stage5_research.provenance import protected,source,evaluator,SPEC
from stage5_research.evaluation import gate,winner,partition_scores,correlations,tie_rate
from stage5_research.pipeline import Exporter
from sbrt.data import Series
from sbrt.replay import replay
from stage4_research.features import FeatureState as E14State
from test_numerics import fixture


class Pipeline(unittest.TestCase):
    def test_partition_exact_digest_global_parity_balance_and_repeatability(self):
        manifest=pd.DataFrame([dict(id=f"s{s}-{f}-{j}",stratum=s,fold=f) for s in STRATA for f in range(5) for j in range(3)])
        actual=expected(manifest); shuffled=expected(manifest.sample(frac=1,random_state=20260925))
        self.assertTrue(actual.equals(shuffled)); self.assertEqual(actual.partition.value_counts().to_dict(),{"A":45,"B":45})
        ordered=[]
        for s in STRATA:
            for f in range(5):
                ids=manifest.loc[(manifest.stratum==s)&(manifest.fold==f),"id"]
                ordered.extend(sorted(ids,key=lambda k:(hashlib.sha256(("stage5_v1|20260924|"+k).encode()).digest(),k)))
        self.assertEqual(actual.id.tolist(),ordered)
        self.assertEqual(actual.partition.tolist(),["A","B"]*45)
        self.assertFalse(set(actual.loc[actual.partition=="A","id"])&set(actual.loc[actual.partition=="B","id"]))
        self.assertEqual(set(actual.id),set(manifest.id))
        self.assertEqual(actual.partition.iloc[3],"B") # parity must not reset at a new cell

    def test_partition_immutable_byte_manifest_and_reject_incompatible(self):
        manifest=pd.DataFrame([dict(id=f"{s}-{f}-{j}",stratum=s,fold=f) for s in STRATA for f in range(5) for j in range(2)])
        with tempfile.TemporaryDirectory() as tmp:
            a,ma=create(tmp,manifest,CV_HASH,SPEC_HASH); b,mb=create(tmp,manifest,CV_HASH,SPEC_HASH)
            self.assertTrue(a.equals(b)); self.assertEqual(ma,mb)
            with self.assertRaises(ValueError): create(tmp,manifest,CV_HASH,"different")
            path=Path(tmp)/"partition/manifest.parquet"; data=path.read_bytes(); path.write_bytes(data+b"tamper")
            with self.assertRaises(ValueError): create(tmp,manifest,CV_HASH,SPEC_HASH)

    def test_frozen_actual_partition_precedes_features_and_covers_population(self):
        manifest,_=load_v2(ROOT/"local_cache",ROOT/"local_cache/cv2/folds.parquet")
        a=expected(manifest); self.assertEqual(a.partition.value_counts().to_dict(),{"A":5000,"B":5000})
        if (ROOT/"stage5_results/partition").exists():
            verify_artifacts(ROOT/"stage5_results/partition")
            self.assertTrue(pd.read_parquet(ROOT/"stage5_results/partition/manifest.parquet").equals(a))

    def test_G5_all_boundaries_strict_AB_and_bootstrap_family(self):
        delta=dict(pooled=.002,folds=[.001]*4+[0.],combined=dict(ge129=0.,le64=-.005))
        parts=dict(A=1e-9,B=1e-9); draws=np.ones(999)
        self.assertTrue(gate(delta,-.005,parts,replicates=draws)["advances"])
        for d in (dict(delta,pooled=.0019999),dict(delta,folds=[1,1,1,0,0]),dict(delta,combined=dict(ge129=-1e-9,le64=0)),dict(delta,combined=dict(ge129=0,le64=-.00500001))):
            self.assertFalse(gate(d,0,parts,replicates=draws)["advances"])
        self.assertFalse(gate(delta,-.005000001,parts,replicates=draws)["advances"])
        self.assertFalse(gate(delta,0,parts,valid=False,replicates=draws)["advances"])
        for p in (dict(A=0.,B=1.),dict(A=1.,B=0.),dict(A=-.1,B=.1)):
            self.assertFalse(gate(delta,0,p,replicates=draws)["advances"])
        strict=np.r_[np.full(15,-.001),np.full(984,.001)]
        self.assertGreater(np.quantile(strict,.025),0)
        r=gate(delta,0,parts,replicates=strict); self.assertFalse(r["advances"]); self.assertEqual(r["family_size"],4)
        self.assertEqual(r["Q_0_01"],float(np.quantile(strict,.01,method="linear")))
        with self.assertRaises(ValueError): gate(delta,0,parts,replicates=np.ones(998))

    def test_child_ablation_requires_positive_pooled_and_lower_95(self):
        d=dict(pooled=.003,folds=[.001]*5,combined=dict(ge129=.003,le64=0)); parts=dict(A=.001,B=.002)
        for delta,draws,expected_result in ((.001,np.ones(999),True),(0,np.ones(999),False),(.001,np.zeros(999),False),(None,None,False)):
            self.assertEqual(gate(d,0,parts,child=True,child_delta=delta,replicates=np.ones(999),child_replicates=draws)["advances"],expected_result)
        self.assertFalse(gate(d,0,parts,child=True,child_delta=0)["bootstrap_required"])

    def test_export_guard_coverage_exact_E14_and_history_only_metadata(self):
        h,o=fixture()
        records=[Series("one",h,o[:63],np.zeros(63),np.arange(63),-1),Series("two",h+.1,o[:38],np.ones(38),np.arange(38),0)]
        exporter=Exporter()
        for record,pred,_ in replay(records,infer_fn=exporter.infer):
            values,null,details=exporter.completed.popleft()
            self.assertEqual(np.asarray(values).shape,(len(record.online),16))
            self.assertEqual(len(null),len(h)-int(.7*len(h)))
            self.assertEqual(details["counts"]["transform"]["observations"],len(pred))
            old=E14State(record.historical,"E14")
            for value,features in zip(record.online,values):
                expected=old.observe(value); old.commit()
                self.assertEqual(features[:5],[expected[k] for k in RAW[:5]])
        self.assertFalse(exporter.completed)

    def test_paired_whole_series_same_six_strata_multiplicities(self):
        manifest=pd.DataFrame(dict(stratum=np.repeat(STRATA,4)))
        a=bootstrap_multiplicities(manifest,999,20260920); b=bootstrap_multiplicities(manifest,999,20260920)
        np.testing.assert_array_equal(a,b)
        for s in STRATA: np.testing.assert_array_equal(a[manifest.stratum==s].sum(axis=0),4)
        y=np.arange(24)%2
        x=sorted_age(np.arange(24),y,np.arange(24)); z=sorted_age(np.arange(24),y,-np.arange(24))
        np.testing.assert_array_equal(weighted_age_terms(x,a)[1],weighted_age_terms(z,a)[1])

    def test_selection_exact_ties_inputs_then_ID_no_standalone(self):
        self.assertEqual(winner({}),"E14")
        r={k:dict(experiment_id=k,status="valid",pooled_oof_ts_auc=.6,model_input_count=len(INPUTS[k]),gate=dict(advances=True)) for k in IDS}
        self.assertEqual(winner(r),"E19")
        r["E18"]["pooled_oof_ts_auc"]+=1e-12; self.assertEqual(winner(r),"E18")
        for v in r.values(): v["gate"]["advances"]=False
        self.assertEqual(winner(r),"E14")

    def test_diagnostic_correlation_weights_ties_and_partition_pairs(self):
        f=pd.DataFrame(dict(id=["a","b","c","d"]*2,time_online=[0]*4+[1]*4,target=[0,1,0,1]*2,
            I=[1.,2.,3.,4.]*2,J_s=[1.,2.,3.,4.]*2,P=[1.,2.,3.,4.]*2,M1=[1.,2.,3.,4.]*2))
        c=correlations(f,["M1"]); self.assertAlmostEqual(c["row_weight_total"],1.)
        self.assertTrue(all(v["possible_near_duplication"] for v in c["values"]))
        parts=pd.DataFrame(dict(id=["a","b","c","d"],partition=["A","A","B","B"]))
        s=partition_scores(f,f.target.to_numpy(),1-f.target.to_numpy(),parts)
        self.assertEqual(s["A"]["candidate"]["pair_weight"],2); self.assertEqual(s["A"]["delta"],1.)
        f["prediction"]=.5; self.assertEqual(tie_rate(f)["rate"],1.)

    def test_protected_artifacts_source_spec_hash_and_complete_reference_oof(self):
        self.assertTrue(protected()["unchanged"])
        self.assertEqual(sha256(SPEC),SPEC_HASH)
        self.assertEqual(sha256(ROOT/"stage4_results/E14/oof.parquet"),E14_HASH)
        self.assertEqual(sha256(ROOT/"local_cache/cv2/folds.parquet"),CV_HASH)
        self.assertIn("stage5_research/features.py",source()["files"])
        manifest,_=load_v2(ROOT/"local_cache",ROOT/"local_cache/cv2/folds.parquet")
        self.assertEqual(len(manifest),10000); self.assertEqual(int(manifest.online_length.sum()),5036517)
        with tempfile.TemporaryDirectory() as tmp:
            con=evaluator(tmp)
            try: self.assertEqual(validate_coverage(con,ROOT/"stage4_results/E14/oof.parquet",manifest),5036517)
            finally: con.close()

    def test_export_allowlist_no_standardized_columns_or_context(self):
        self.assertEqual(len(RAW),16)
        self.assertFalse(set(RAW)&{"age","tau","eta","partition","fold","gamma0","z_t","u_t"})
        self.assertEqual(CONFIG["parents"],dict(E18="E14",E19="E14",E20="E19",E21="E19"))
