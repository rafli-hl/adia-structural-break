"""Explicitly synthetic E08 contract tests; no research/model selection data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sbrt.logistic import (INPUTS, SOLVER_OPTIONS, LogisticModel, E08Detector,
                           fit_fold, objective, standardize_fit, training_weights, infer)
from sbrt.training_weights import metric_training_weights
from sbrt.stage2_detectors import Stage2Detector
from sbrt.e08 import fit_oof, verify_protected, CV2_SHA256
from sbrt.provenance import ROOT, write_json, source_manifest
from sbrt.data import Series, sha256
from sbrt.replay import replay
from sbrt.stage2 import validate_coverage
from sbrt.folds import create_v2
from test_stage2_pipeline import integration_cache


def fixture():
    """Labeled synthetic four-block rows, with unequal lengths and class counts."""
    rng = np.random.default_rng(808)
    rows = []
    for series in range(40):
        for age in range(4 + series % 13):
            y = int(series % 3 != 0 and age >= 2 + series % 4)
            blocks = rng.uniform(size=4)
            blocks[1] = (blocks[1] + y) / 2
            rows.append(dict(id=f"synthetic_{series:02d}", fold=series % 5, time=100 + age,
                             time_online=age, target=y, **dict(zip(INPUTS, blocks))))
    return pd.DataFrame(rows)


class E08Numerics(unittest.TestCase):
    def test_e08_vectorized_weights_equal_frozen_reference_and_identities(self):
        frame = fixture()
        for fold in range(5):
            indices, expected = metric_training_weights(frame, fold)
            train = frame[frame.fold != fold]
            eligible, weights, audit = training_weights(train.time_online, train.target)
            np.testing.assert_array_equal(train.index[eligible], indices)
            np.testing.assert_array_equal(weights, expected)
            self.assertAlmostEqual(weights.sum(), 1., places=14)
            self.assertGreater(audit["excluded_rows"], 0)
            self.assertLess(max(audit["maximum_absolute_errors"].values()), 1e-12)
            for row in audit["ages"]:
                self.assertAlmostEqual(row["actual_mass"], row["pair_weight"] / audit["Z"], places=14)
                self.assertAlmostEqual(row["positive_mass"], row["expected_mass"] / 2, places=14)
                self.assertAlmostEqual(row["negative_mass"], row["expected_mass"] / 2, places=14)

    def test_e08_heldout_labels_features_ages_never_affect_fit_or_weights(self):
        frame = fixture()
        model, detail, audit = fit_fold(frame, 2)
        changed = frame.copy()
        changed.loc[changed.fold == 2, "target"] = 1 - changed.loc[changed.fold == 2, "target"]
        changed.loc[changed.fold == 2, list(INPUTS)] = np.nan
        changed.loc[changed.fold == 2, "time_online"] = -999
        other, other_detail, other_audit = fit_fold(changed, 2)
        self.assertEqual(model.dumps(), other.dumps())
        self.assertEqual(audit, other_audit)
        for key in ("iterations", "objective", "final_gradient", "weighted_mean", "weighted_std"):
            self.assertEqual(detail[key], other_detail[key])

    def test_e08_weighted_preprocessing_training_only_and_constant_zero(self):
        frame = fixture()
        frame["D"] = 0.3
        model, _, _ = fit_fold(frame, 0)
        train = frame[frame.fold != 0]
        eligible, w, _ = training_weights(train.time_online, train.target)
        x = train.loc[eligible, list(INPUTS)].to_numpy()
        expected_mu = np.average(x, weights=w, axis=0)
        expected_sd = np.sqrt(np.average((x - expected_mu) ** 2, weights=w, axis=0))
        np.testing.assert_allclose(model.mean[:3], expected_mu[:3], atol=1e-15)
        np.testing.assert_allclose(model.std[:3], expected_sd[:3], atol=1e-15)
        self.assertEqual(model.std[3], 0.0)
        self.assertEqual(model.beta[3], 0.0)
        xx = x[:10].copy()
        prediction = model.predict(xx)
        xx[:, 3] = 100
        np.testing.assert_array_equal(prediction, model.predict(xx))
        _, std, standardized = standardize_fit(np.full((10, 4), .3), np.full(10, .1))
        np.testing.assert_array_equal(std, np.zeros(4))
        np.testing.assert_array_equal(standardized, np.zeros((10, 4)))

    def test_e08_objective_analytic_gradient_and_unpenalized_intercept(self):
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=(30, 4)), rng.integers(0, 2, size=30)
        w = np.arange(1, 31, dtype=float)
        w /= w.sum()
        theta = np.array([.3, -.5, .1, .2, .4])
        with threadpool_limits(limits=1):
            value, gradient = objective(theta, x, y, w)
            z = theta[0] + x @ theta[1:]
            direct = sum(wi * (np.logaddexp(0, zi) - yi * zi) for wi, zi, yi in zip(w, z, y))
            self.assertAlmostEqual(value, direct + .005 * sum(theta[1:] ** 2), places=14)
            epsilon = 1e-5
            numerical = []
            for j in range(5):
                step = np.zeros(5)
                step[j] = epsilon
                numerical.append((objective(theta + step, x, y, w)[0] - objective(theta - step, x, y, w)[0]) / (2 * epsilon))
            np.testing.assert_allclose(gradient, numerical, atol=1e-10, rtol=1e-7)

    def test_e08_determinism_convergence_and_single_thread(self):
        frame = fixture()
        a, detail, _ = fit_fold(frame, 0)
        b, other, _ = fit_fold(frame, 0)
        self.assertEqual(a.dumps(), b.dumps())
        self.assertEqual(detail["final_gradient"], other["final_gradient"])
        self.assertLessEqual(detail["gradient_infinity_norm"], 1e-6)
        self.assertLessEqual(detail["iterations"], 1000)
        self.assertTrue(detail["optimizer_success"])
        self.assertTrue(all(pool["num_threads"] == 1 for pool in detail["native_threads"]))
        self.assertEqual(SOLVER_OPTIONS["maxiter"], 1000)
        self.assertEqual(SOLVER_OPTIONS["gtol"], 1e-6)

    def test_e08_only_four_blocks_metadata_does_not_enter_model(self):
        frame = fixture()
        a, _, _ = fit_fold(frame, 1)
        augmented = frame.assign(final_horizon=999, historical_acf=10, kurtosis=2,
                                 ar_mse_ratio=.1, interaction=4, age=500, tau_index=1,
                                 primitive_statistic=3, id="unused metadata")
        # Reordering all columns cannot alter the explicit four-column contract.
        augmented = augmented[list(reversed(augmented.columns))]
        b, _, _ = fit_fold(augmented, 1)
        self.assertEqual(a.dumps(), b.dumps())
        self.assertEqual(json.loads(a.dumps())["inputs"], ["R", "I", "P", "D"])
        with self.assertRaisesRegex(ValueError, "four"):
            a.predict(np.zeros((2, 18)))
        with self.assertRaises(ValueError):
            a.predict(np.zeros(5))

    def test_e08_save_load_finite_range_scalar_batch_and_model_immutability(self):
        model, _, _ = fit_fold(fixture(), 3)
        before = model.dumps()
        loaded = LogisticModel.loads(before)
        x = fixture()[list(INPUTS)].to_numpy()
        expected = model.predict(x)
        np.testing.assert_array_equal(expected, loaded.predict(x))
        np.testing.assert_array_equal(expected, [loaded.predict(row) for row in x])
        self.assertTrue(np.isfinite(expected).all())
        self.assertTrue(((expected >= 0) & (expected <= 1)).all())
        self.assertEqual(before, loaded.dumps())
        with self.assertRaises(ValueError):
            loaded.beta[0] = 4
        bad = json.loads(before)
        bad["inputs"] = ["I", "R", "P", "D"]
        with self.assertRaises(ValueError):
            LogisticModel.loads(json.dumps(bad))

    def test_e08_streaming_prefix_truncation_serialization_and_handshake(self):
        rng = np.random.default_rng(55)
        h, online = rng.normal(size=180), rng.normal(size=300)
        model, _, _ = fit_fold(fixture(), 0)
        state, block = E08Detector(h, model), Stage2Detector(h, "E07")
        expected = []
        for x in online:
            block.update(x)
            expected.append(float(model.predict([block.last_blocks[k] for k in INPUTS])))
        actual = [state.update(x) for x in online[:149]]
        restored = E08Detector.loads(state.dumps())
        self.assertEqual(state.dumps(), restored.dumps())
        continuation = [restored.update(x) for x in online[149:]]
        np.testing.assert_array_equal(actual + continuation, expected)
        np.testing.assert_array_equal([state.update(x) for x in online[149:]], continuation)
        self.assertEqual(state.dumps(), restored.dumps())
        for n in (1, 8, 33, 128, 256):
            fresh = E08Detector(h, model)
            np.testing.assert_array_equal([fresh.update(x) for x in online[:n]], expected[:n])
        record = Series("test-only", h, online, np.zeros(len(online), dtype=int), np.arange(len(online)), -1)
        replayed = list(replay([record], infer_fn=lambda ds: infer(ds, model)))
        np.testing.assert_array_equal(replayed[0][1], expected)

    def test_e08_no_eligible_ages_and_invalid_input_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "eligible"):
            training_weights(np.arange(3), np.zeros(3))
        with self.assertRaises(ValueError):
            training_weights(np.array([-1, 0]), np.array([0, 1]))
        bad = fixture()
        bad.loc[bad.fold != 0, "R"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_fold(bad, 0)


class E08Pipeline(unittest.TestCase):
    def test_e08_complete_oof_deterministic_models_and_exact_keys(self):
        frame = fixture()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            p, diagnostics, timing = fit_oof(frame, root / "first")
            q, _, _ = fit_oof(frame, root / "second")
            np.testing.assert_array_equal(p, q)
            self.assertEqual(len(p), len(frame))
            self.assertEqual(len(diagnostics), 5)
            self.assertGreater(timing["inference_rows_per_second"], 0)
            for fold in range(5):
                self.assertEqual((root / f"first/fold{fold}.json").read_bytes(),
                                 (root / f"second/fold{fold}.json").read_bytes())
                model = LogisticModel.loads((root / f"first/fold{fold}.json").read_bytes())
                valid = frame.fold == fold
                np.testing.assert_array_equal(p[valid], model.predict(frame.loc[valid, list(INPUTS)]))
            with self.assertRaisesRegex(ValueError, "overwrite"):
                fit_oof(frame, root / "first")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                fit_oof(pd.concat([frame, frame.iloc[:1]]), root / "bad")

    def test_e08_oof_coverage_against_actual_evaluator_cache(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            con = integration_cache(root)
            try:
                manifest, _ = create_v2(root / "cache", root / "cv2")
                frame = con.execute("""SELECT id,time,target,
                    row_number() OVER(PARTITION BY id ORDER BY ordinal)-1 AS time_online
                    FROM observations WHERE period=2 ORDER BY id,time""").fetchdf()
                frame = frame.merge(manifest[["id", "fold"]], on="id", validate="many_to_one")
                rng = np.random.default_rng(88)
                for k in INPUTS:
                    frame[k] = rng.uniform(size=len(frame))
                frame["prediction"], _, _ = fit_oof(frame, root / "models")
                frame.to_parquet(root / "oof.parquet", index=False)
                self.assertEqual(validate_coverage(con, root / "oof.parquet", manifest), len(frame))
                frame.iloc[:-1].to_parquet(root / "bad.parquet", index=False)
                with self.assertRaisesRegex(ValueError, "count"):
                    validate_coverage(con, root / "bad.parquet", manifest)
                wrong = frame.copy()
                wrong.loc[0, "fold"] = (wrong.loc[0, "fold"] + 1) % 5
                wrong.to_parquet(root / "bad.parquet", index=False)
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    validate_coverage(con, root / "bad.parquet", manifest)
            finally:
                con.close()

    def test_e08_existing_phase1_source_artifacts_and_cv2_unchanged(self):
        baseline = ROOT / "stage2_phase2/before/baseline.json"
        self.assertTrue(baseline.exists(), "E08 requires the pre-edit Phase-1 integrity snapshot")
        verified = verify_protected(baseline)
        self.assertTrue(verified["artifacts_unchanged"])
        self.assertEqual(sha256(ROOT / "local_cache/cv2/folds.parquet"), CV2_SHA256)
        # The source identity is recorded by the final validation runner.
        self.assertIn("sbrt/logistic.py", source_manifest()["files"])


if __name__ == "__main__":
    unittest.main()
