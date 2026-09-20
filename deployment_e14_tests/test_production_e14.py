"""Deployment contracts. Synthetic trajectories below are explicit test data."""
import ast
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from production_e14 import runtime as rt
from production_e14.local import ROOT, load_module, verify_protected, write_package
from stage4_research.models import fit_fold
from sbrt.data import Series
from sbrt.replay import replay


def fixture():
    rng = np.random.default_rng(20260919)
    return [(i, rng.normal(size=110+i), rng.normal(size=24+i), 3+i if i%2 else None)
            for i in range(8)]


class ProductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture()
        cls.frame, cls.metadata, _ = rt.build_training_frame(cls.data)
        cls.model, cls.fit, cls.weights = rt.fit_full(cls.frame)

    def bundle(self, directory):
        return rt.save_bundle(directory, self.model, self.metadata)

    def predictions(self, records, directory):
        return {r.id: p for r, p, _ in replay(records, infer_fn=lambda ds: rt.infer(ds, directory))}

    def records(self, data=None):
        return [Series(str(i), h, o, np.empty(0), np.arange(len(o)), -1) for i, h, o, _ in (data or self.data)]

    def test_full_population_reuses_exact_E14_fitter_not_fold_average(self):
        reference, _, weights = fit_fold(self.frame.assign(fold=0), -1, "E14")
        self.assertEqual(self.model.dumps(), reference.dumps())
        self.assertEqual(self.weights, weights)
        self.assertEqual(self.weights["training_rows"], len(self.frame))
        self.assertLessEqual(self.fit["gradient_infinity_norm"], 1e-6)

    def test_full_training_weight_identities_and_no_fold_input(self):
        model, _, weights = rt.fit_full(self.frame.assign(fold=np.arange(len(self.frame))%5))
        self.assertEqual(model.dumps(), self.model.dumps())
        self.assertLessEqual(max(weights["maximum_absolute_errors"].values()), 1e-12)
        self.assertAlmostEqual(weights["total_weight"], 1, places=12)

    def test_training_determinism_and_order_canonicalization(self):
        frame, metadata, _ = rt.build_training_frame(reversed(self.data))
        self.assertTrue(frame.equals(self.frame))
        self.assertEqual(metadata, self.metadata)
        model, _, _ = rt.fit_full(frame)
        self.assertEqual(model.dumps(), self.model.dumps())

    def test_save_load_equality_and_bundle_reproducibility(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.assertEqual(self.bundle(a), self.bundle(b))
            for name in ("manifest.json", "e14.json"):
                self.assertEqual((Path(a)/name).read_bytes(), (Path(b)/name).read_bytes())
            loaded = rt.load_bundle(a)
            self.assertEqual(loaded.dumps(), self.model.dumps())
            np.testing.assert_array_equal(loaded.predict(self.frame[list(rt.RAW)]), self.model.predict(self.frame[list(rt.RAW)]))

    def test_artifact_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.bundle(tmp)
            path = Path(tmp)/"e14.json"
            path.write_bytes(path.read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError, "hash"):
                rt.load_bundle(tmp)

    def test_package_reproducibility_allowlist_and_no_research_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.bundle(root/"model")
            first = write_package(root/"one", root/"model")
            second = write_package(root/"two", root/"model")
            self.assertEqual(first, second)
            self.assertEqual(first["zip_sha256"], rt.file_hash(root/"two.zip"))
            self.assertEqual(set(first["files"]), {*rt.RUNTIME_FILES, "main.py", "requirements.txt",
                                                   "resources/e14.json", "resources/manifest.json"})
            self.assertFalse(any("data/" in p or "local.py" in p for p in first["files"]))

    def test_exact_five_input_allowlist(self):
        self.assertEqual(rt.INPUTS, ("z_R","z_I","z_P","z_D","S(C_s)"))
        self.assertEqual(rt.RAW, ("R","I","P","D","J_s"))
        extra = self.frame.assign(age=99, series_id=-100, horizon=500, target_information=1)
        model, _, _ = rt.fit_full(extra)
        self.assertEqual(model.dumps(), self.model.dumps())
        for missing in rt.RAW:
            with self.assertRaises(KeyError):
                model.predict({k:0.5 for k in rt.RAW if k!=missing})

    def test_official_generator_handshake_and_complete_finite_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.bundle(tmp)
            visited = []
            def datasets():
                visited.append(True)
                yield self.data[0][1], iter(self.data[0][2])
            generator = rt.infer(datasets(), tmp)
            self.assertIsNone(next(generator))
            self.assertEqual(visited, [])
            values = np.asarray(list(generator))
            self.assertEqual(len(values), len(self.data[0][2]))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(((0 <= values) & (values <= 1)).all())
            all_values = self.predictions(self.records(), tmp)
            self.assertEqual(sum(map(len, all_values.values())), len(self.frame))

    def test_prefix_and_truncation_invariance(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.bundle(tmp)
            full = self.predictions(self.records(), tmp)
            for cut in (1, 8, 17):
                shortened = [(i, h, o[:cut], tau) for i, h, o, tau in self.data]
                short = self.predictions(self.records(shortened), tmp)
                extended = [(i, h, np.r_[o[:cut], np.full(9, 999.)], tau) for i, h, o, tau in self.data]
                other = self.predictions(self.records(extended), tmp)
                for key in full:
                    np.testing.assert_array_equal(full[key][:cut], short[key])
                    np.testing.assert_array_equal(full[key][:cut], other[key][:cut])

    def test_series_order_independence_and_no_cross_series_mutable_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.bundle(tmp)
            records = self.records()
            normal = self.predictions(records, tmp)
            reverse = self.predictions(list(reversed(records)), tmp)
            for record in records:
                alone = self.predictions([record], tmp)
                np.testing.assert_array_equal(normal[record.id], reverse[record.id])
                np.testing.assert_array_equal(normal[record.id], alone[record.id])

    def test_resumed_state_serialization_exact(self):
        _, h, o, _ = self.data[0]
        state = rt.E14Detector(h, self.model)
        for value in o[:11]:
            state.update(value)
        payload = state.dumps()
        restored = rt.E14Detector.loads(payload, self.model)
        self.assertEqual(payload, restored.dumps())
        for value in o[11:]:
            self.assertEqual(state.update(value), restored.update(value))
        self.assertEqual(state.dumps(), restored.dumps())

    def test_official_main_signatures_train_and_infer(self):
        module = load_module(ROOT/"deployment_e14/main.py", "deployment_contract_test")
        self.assertEqual(list(inspect.signature(module.train).parameters), ["datasets", "model_directory_path"])
        self.assertEqual(list(inspect.signature(module.infer).parameters), ["datasets", "model_directory_path"])
        self.assertEqual(module.INFER_PARALLELISM, 1)
        with tempfile.TemporaryDirectory() as tmp:
            module.train(self.data, tmp)
            self.assertEqual(rt.load_bundle(tmp).dumps(), self.model.dumps())
            values = list(module.infer([(self.data[0][1], iter(self.data[0][2]))], tmp))
            self.assertIsNone(values[0])
            self.assertEqual(len(values)-1, len(self.data[0][2]))

    def test_inference_has_no_evaluator_or_training_imports(self):
        tree = ast.parse(inspect.getsource(rt.infer))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertFalse(names & {"target", "tau", "identifier", "frame", "training_datasets", "fit_full"})
        for path in rt.CORE:
            text = (ROOT/path).read_text()
            self.assertNotIn("from .data import", text)
            self.assertNotIn("from .metric import", text)

    def test_training_rejects_bad_tau_and_duplicate_id(self):
        bad = self.data.copy()
        i, h, o, _ = bad[0]
        for tau in (-1, len(o), .5):
            with self.assertRaises(ValueError):
                rt.build_training_frame([(i, h, o, tau)])
        with self.assertRaises(ValueError):
            rt.build_training_frame([self.data[0], self.data[0]])

    def test_protected_research_source_and_all_stage1_2_3_artifacts(self):
        self.assertTrue(verify_protected()["unchanged"])

    def test_exact_conditional_timing_current_denominator_isolation(self):
        from stage4_research.features import FeatureState
        _, history, online, _ = self.data[0]
        a, b = FeatureState(history,"E14"), FeatureState(history,"E14")
        h = a.scales["slow"].h
        floor = a.scales["slow"].v_min
        first = rt.observe(a,float(online[0]))
        rt.observe(b,1000.0)
        self.assertEqual(a.scales["slow"].h,h)
        self.assertEqual(b.scales["slow"].h,h)
        self.assertEqual(a.diagnostic["v_s"],max(h,floor))
        self.assertEqual(a.diagnostic["v_s"],b.diagnostic["v_s"])
        score = float(self.model.predict(first))
        e = a.scales["slow"].pending
        a.commit()
        self.assertEqual(a.scales["slow"].h,.99*h+.01*(e*e))
        detector = rt.E14Detector(history,self.model)
        self.assertEqual(score,detector.update(float(online[0])))
        self.assertEqual(a.dumps(),detector.features.dumps())
        self.assertEqual(a.scales["slow"].alpha,.01)
        self.assertIsNone(a.scales["slow"].cdf)
        self.assertIsNone(a.scales["slow"].cap)

    def test_production_order_exact_research_state_and_prediction_equivalence(self):
        from stage4_research.features import Detector
        for _,history,online,_ in self.data:
            production = rt.E14Detector(history,self.model)
            research = Detector(history,self.model)
            self.assertEqual(production.dumps(),research.dumps())
            for value in online:
                self.assertEqual(production.update(float(value)),research.update(float(value)))
                self.assertEqual(production.dumps(),research.dumps())

    def test_base_standardization_matches_full_population_E08(self):
        from production_e08.runtime import fit_full as e08_fit
        base,_,audit = e08_fit(self.frame)
        np.testing.assert_array_equal(base.mean,self.model.pre.mean)
        np.testing.assert_array_equal(base.std,self.model.pre.std)
        self.assertEqual(audit,self.weights)

    def test_model_rejects_other_stage4_experiments(self):
        other,_,_ = fit_fold(self.frame.assign(fold=0,J_f=self.frame.J_s),-1,"E15")
        with self.assertRaises(ValueError):
            rt.E14Detector(self.data[0][1],other)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                rt.save_bundle(tmp,other,self.metadata)

    def test_frozen_parameters_survive_online_inference(self):
        state = rt.E14Detector(self.data[0][1],self.model)
        scale = state.features.scales["slow"]
        before = (scale.mu,scale.sd,scale.b,scale.v_min,state.features.ar.beta.copy(),
                  state.features.blocks.cdf.reference.copy())
        model_bytes = self.model.dumps()
        for x in self.data[0][2]:
            state.update(float(x))
        self.assertEqual(before[:4],(scale.mu,scale.sd,scale.b,scale.v_min))
        np.testing.assert_array_equal(before[4],state.features.ar.beta)
        np.testing.assert_array_equal(before[5],state.features.blocks.cdf.reference)
        self.assertEqual(model_bytes,self.model.dumps())

    def test_full_reduced_coverage_and_exact_ten_percent_replay(self):
        from production_e14.local import reduced_records
        records = reduced_records(ROOT/"data")
        self.assertEqual(len(records),100)
        self.assertEqual(sum(len(r.online) for r in records),50983)
        with tempfile.TemporaryDirectory() as tmp:
            self.bundle(tmp)
            full = self.predictions(records,tmp)
            repeat = self.predictions(records[:10],tmp)
            self.assertEqual(sum(map(len,full.values())),50983)
            for key in repeat:
                np.testing.assert_array_equal(full[key],repeat[key])

    def test_local_memory_measurement(self):
        measure = load_module(ROOT/"deployment_e14/measure.py","e14_test_measure")
        with measure.MemoryProbe() as probe:
            self.assertGreater(measure.snapshot()["rss_bytes"],0)
        self.assertGreater(probe.report["sampled_peak_rss_bytes"],0)


if __name__ == "__main__":
    unittest.main()
