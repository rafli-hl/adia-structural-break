import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sbrt.detectors import OfficialEWMABaseline, E01, infer
from sbrt.metric import ts_auc
from sbrt.replay import replay, ProtocolError
from sbrt.data import Series, prepare, iter_series, get_history
from sbrt.cli import make_synthetic
from sbrt.profile import run_profile, residuals, fit_ar
from sbrt.backtest import e00, run_backtest, fixed_folds


def official_quickstarter_extract(datasets):
    """Independent transcription of executable semantics supplied by user.

    Uses NumPy scalar math as the quickstarter-style implementation; production
    class uses Python floats/math. No claim of downloading notebook source.
    """
    ALPHA = 0.05
    KAPPA = 3.0
    yield
    for x_historical, x_online in datasets:
        x_h = np.asarray(x_historical, dtype=np.float64)
        mu_h = x_h.mean() if len(x_h) else 0.0
        sd_h = x_h.std(ddof=1) if len(x_h) > 1 else 1.0
        sd_h = max(sd_h, 1e-8)
        mu_ewma = mu_h
        n_eff = 0.0
        for x_t in x_online:
            mu_ewma = (1 - ALPHA) * mu_ewma + ALPHA * x_t
            n_eff = (1 - ALPHA) * n_eff + 1
            se = sd_h / np.sqrt(max(n_eff, 1.0))
            z = (mu_ewma - mu_h) / max(se, 1e-8)
            yield float(np.tanh(abs(z) / KAPPA))


def fixture(sid="a", history=None, online=None):
    if history is None:
        history = np.sin(np.arange(100) * .23)
    if online is None:
        online = np.cos(np.arange(30) * .17) + .5
    return Series(sid, np.asarray(history, dtype=float), np.asarray(online, dtype=float),
                  np.zeros(len(online), dtype=np.int8), np.arange(len(online)), -1)


def predict(records, detector):
    return {s.id: scores for s, scores, _ in replay(iter(records), detector)}


class EWMARegression(unittest.TestCase):
    def test_observation_by_observation(self):
        rng = np.random.default_rng(10)
        cases = [([], [0., 1., -1.]), ([7.], [7., 8., -8.]),
                 (np.zeros(50), [0., 1e-12, 1e-7, -1e-7, 1.]),
                 (np.full(100, 5.), np.full(200, 5.)),
                 (rng.normal(size=1000), rng.normal(size=1000)),
                 (rng.normal(size=500), np.r_[rng.normal(size=20), np.full(100, 4.)]),
                 (1e12 + rng.normal(size=100), 1e12 + rng.normal(size=300)),
                 (rng.normal(size=200) * 1e-10, rng.normal(size=100) * 1e-9)]
        for h, o in cases:
            official = official_quickstarter_extract([(h, iter(o))])
            self.assertIsNone(next(official))
            state = OfficialEWMABaseline(h)
            actual = [state.update(x) for x in o]
            np.testing.assert_allclose(actual, list(official), rtol=2e-14, atol=2e-14)

    def test_parameters_initialization_and_recurrence(self):
        s = OfficialEWMABaseline([1., 3.])
        self.assertEqual(s.ALPHA, .05)
        self.assertEqual(s.KAPPA, 3.)
        self.assertEqual(s.mu_ewma, 2.)
        self.assertEqual(s.n_eff, 0.)
        for _ in range(1000):
            s.update(2.)
        self.assertAlmostEqual(s.n_eff, 20., places=11)
        self.assertEqual(s.update(2.), 0.)


class MetricTests(unittest.TestCase):
    def test_e00(self):
        self.assertEqual(e00()["scores"], dict(constant=.5, age_only=.5, perfect_oracle=1., reversed_oracle=0.))

    def test_weighted_hand_calculation(self):
        # age 0 has 2*2=4 pairs and perfect ranking; age 1 has one reversed pair.
        df = pd.DataFrame(dict(id=["a", "b", "c", "d", "a", "b"],
                               time_online=[0, 0, 0, 0, 1, 1], target=[0, 0, 1, 1, 0, 1],
                               prediction=[0., 0., 1., 1., 1., 0.]))
        self.assertEqual(ts_auc(df)[0], .8)

    def test_official_reference_random_ties_and_lengths(self):
        rng = np.random.default_rng(42)
        rows = []
        for sid in range(100):
            n = int(rng.integers(1, 50))
            tau = int(rng.integers(0, 60))
            for t in range(n):
                rows.append((sid, t, int(t >= tau), float(rng.integers(0, 5) / 4)))
        df = pd.DataFrame(rows, columns=["id", "time_online", "target", "prediction"])
        weighted = total = 0.0
        for _, group in df.groupby("time_online"):
            labels, scores = group.target.values, group.prediction.values
            n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
            if n_pos == 0 or n_neg == 0:
                continue
            weight = n_pos * n_neg
            weighted += weight * roc_auc_score(labels, scores)
            total += weight
        self.assertEqual(ts_auc(df)[0], weighted / total)

    def test_invalid_predictions_and_coverage(self):
        df = pd.DataFrame(dict(id=["a", "b"], time_online=[0, 0], target=[0, 1], prediction=[0., 1.]))
        for bad in (df.assign(prediction=[np.nan, 1]), df.assign(prediction=[-1., 1.]),
                    df.assign(target=[2, 1]), pd.concat([df, df]), df.assign(time_online=[-.5, 0])):
            with self.assertRaises(ValueError):
                ts_auc(bad)
        with self.assertRaises(ValueError):
            ts_auc(df.iloc[:1], expected_keys=df[["id", "time_online"]])

    def test_empty_and_single_class(self):
        empty = pd.DataFrame(columns=["id", "time_online", "target", "prediction"])
        self.assertEqual(ts_auc(empty)[0], .5)
        df = pd.DataFrame(dict(id=["a"], time_online=[0], target=[0], prediction=[1.]))
        self.assertEqual(ts_auc(df)[0], .5)


class CausalityTests(unittest.TestCase):
    def test_unchanged_constant_series(self):
        record = fixture(history=np.full(50, 5.), online=np.full(30, 5.))
        for name in E01:
            np.testing.assert_array_equal(predict([record], name)["a"], np.zeros(30))

    def test_prefix_truncation_order_and_determinism(self):
        original = fixture()
        alternate = copy.deepcopy(original)
        alternate.online[12:] += 10000
        truncated = copy.deepcopy(original)
        truncated.online = truncated.online[:12]
        other = fixture("b", online=np.arange(40) * -.1)
        for name in E01 + ("constant", "age_only"):
            baseline = predict([original], name)["a"]
            np.testing.assert_array_equal(baseline, predict([original], name)["a"])
            np.testing.assert_array_equal(baseline[:12], predict([alternate], name)["a"][:12])
            np.testing.assert_array_equal(baseline[:12], predict([truncated], name)["a"])
            forward = predict([original, other], name)
            reverse = predict([other, original], name)
            for sid in forward:
                np.testing.assert_array_equal(forward[sid], reverse[sid])

    def test_actual_infer_generator(self):
        records = [fixture(), fixture("b")]
        for name in E01:
            actual = {s.id: score for s, score, _ in replay(iter(records), name, lambda ds: infer(ds, name))}
            expected = predict(records, name)
            for sid in actual:
                np.testing.assert_array_equal(actual[sid], expected[sid])

    def test_lookahead_rejected(self):
        def bad(ds):
            yield
            for h, online in ds:
                for x in online:
                    next(online)
                    yield .5
        with self.assertRaises(ProtocolError):
            list(replay(iter([fixture()]), infer_fn=bad))

    def test_materialization_rejected(self):
        for convert in (list, len, np.asarray):
            def bad(ds):
                yield
                for h, online in ds:
                    convert(online)
                    yield .5
            with self.assertRaises(ProtocolError):
                list(replay(iter([fixture()]), infer_fn=bad))

    def test_extra_missing_and_skipped_series_rejected(self):
        def extra(ds):
            yield
            for h, online in ds:
                for x in online:
                    yield .5
                    yield .5
        def missing(ds):
            yield
            for h, online in ds:
                for x in online:
                    return
        def skip(ds):
            yield
            for h, online in ds:
                for x in online:
                    yield .5
                return
        for bad in (extra, missing, skip):
            with self.assertRaises(ProtocolError):
                list(replay(iter([fixture(), fixture("b")]), infer_fn=bad))

    def test_handshake_rejected(self):
        def bad(ds):
            for h, online in ds:
                for x in online:
                    yield .5
        with self.assertRaises(ProtocolError):
            list(replay(iter([fixture()]), infer_fn=bad))


class ParquetIntegration(unittest.TestCase):
    def test_full_longform_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, cache, out = root / "data", root / "cache", root / "profile"
            make_synthetic(data)
            con, source = prepare(data / "X_train.parquet", data / "y_train.parquet", data / "y_train_index.parquet", cache)
            try:
                # Tiny batches exercise carry across row groups and series.
                series = list(iter_series(con, batch_size=17))
                self.assertEqual(len(series), 30)
                original = pd.read_parquet(data / "X_train.parquet")
                for s in series:
                    g = original.loc[s.id]
                    np.testing.assert_array_equal(s.historical, g[g.period == 1].value)
                    np.testing.assert_array_equal(s.online, g[g.period == 2].value)
                    np.testing.assert_array_equal(s.target, (np.arange(len(s.online)) >= s.tau_index) if s.tau_index >= 0 else np.zeros(len(s.online)))
                summary = run_profile(con, source, cache, out)
                self.assertEqual(summary["counts"]["series"], 30)
                self.assertEqual(summary["counts"]["no_break"], 15)
                alignment = json.loads((out / "label_alignment.json").read_text())
                self.assertEqual(alignment["status"], "pass")
                self.assertEqual(alignment["first_positive_minus_tau_counts"], {"0": 15})
                audit = json.loads((out / "related_series_audit.json").read_text())
                self.assertEqual(audit["duplicate_histories"], 1)
                folds, _ = fixed_folds(cache)
                self.assertEqual(folds.groupby("group").fold.nunique().max(), 1)
                self.assertEqual(folds.fold.nunique(), 5)
                results = run_backtest(con, cache, root / "evaluation", detectors=E01 + ("constant", "age_only"))
                self.assertEqual(len(results), 9)
                self.assertEqual(results[-1]["pooled_oof_ts_auc"], .5)
                self.assertEqual(results[-2]["pooled_oof_ts_auc"], .5)
                oof = pd.read_parquet(root / "evaluation" / "oof_official_ewma.parquet")
                self.assertEqual(results[0]["pooled_oof_ts_auc"], ts_auc(oof)[0])
                self.assertEqual(len(oof), sum(len(s.online) for s in series))
            finally:
                con.close()

    def test_bad_keys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_synthetic(root)
            y = pd.read_parquet(root / "y_train.parquet")
            pd.concat([y, y.iloc[:1]]).to_parquet(root / "y_train.parquet")
            with self.assertRaisesRegex(ValueError, "integrity"):
                prepare(root / "X_train.parquet", root / "y_train.parquet", root / "y_train_index.parquet", root / "cache")

    def test_one_based_tau_detected_without_relabeling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_synthetic(root)
            metadata = pd.read_parquet(root / "y_train_index.parquet")
            metadata.loc[metadata.tau_index >= 0, "tau_index"] += 1
            metadata.to_parquet(root / "y_train_index.parquet")
            con, source = prepare(root / "X_train.parquet", root / "y_train.parquet", root / "y_train_index.parquet", root / "cache")
            try:
                run_profile(con, source, root / "cache", root / "profile")
                alignment = json.loads((root / "profile" / "label_alignment.json").read_text())
                self.assertEqual(alignment["first_positive_minus_tau_counts"], {"-1": 15})
                self.assertEqual(alignment["status"], "pass")
            finally:
                con.close()

    def test_noncumulative_labels_block_backtest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_synthetic(root)
            y = pd.read_parquet(root / "y_train.parquet")
            sid = "synthetic_001"
            keys = y.loc[sid].sort_index().index
            y.loc[(sid, keys[-1]), "target"] = 0
            y.to_parquet(root / "y_train.parquet")
            con, source = prepare(root / "X_train.parquet", root / "y_train.parquet", root / "y_train_index.parquet", root / "cache")
            try:
                run_profile(con, source, root / "cache", root / "profile")
                with self.assertRaisesRegex(ValueError, "Label audit"):
                    run_backtest(con, root / "cache", root / "evaluation")
            finally:
                con.close()

    def test_historical_ar_diagnostic(self):
        rng = np.random.default_rng(3)
        h = np.zeros(2000)
        for i in range(1, len(h)):
            h[i] = .85 * h[i - 1] + rng.normal()
        model = fit_ar(h[:1400], 1)
        e = residuals(h, model)[1400:]
        baseline = (h[1400:] - model[0]) / model[1]
        self.assertLess(np.var(e) / np.var(baseline), .5)


if __name__ == "__main__":
    unittest.main()
