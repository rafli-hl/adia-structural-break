"""Independent numerical/prefix fixtures, never model-selection data."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import unittest
import numpy as np
from sbrt.detectors import E01, EvidenceDetector, make_detector, infer
from sbrt.history import fit_ar, residuals, ARChannel, EmpiricalCDF
from sbrt.features import (RollingMean, MultiScaleEvidence, SignWindow, SignDependence,
                           historical_sign_correlation)
from sbrt.stage2_detectors import Stage2Detector
from sbrt.data import Series
from sbrt.replay import replay, ProtocolError

ROOT = Path(__file__).resolve().parent.parent


def data():
    rng = np.random.default_rng(771)
    h = rng.standard_t(5, 1000)
    for i in range(5, len(h)):
        h[i] += .45*h[i-1] - .2*h[i-3]
    o = rng.standard_t(4, 310)
    o[130:] = .7 + 1.5*o[130:]
    return h, o


def direct_evidence(values, reference):
    result = []
    for w in (32, 128, 256, len(values)):
        segment = values[-w:]
        q = np.sqrt(len(segment))*abs(np.mean(segment)-np.mean(reference))/max(np.std(reference, ddof=1), 1e-8)
        result.append(float(q/(1+q)))
    return result


def direct_midrank(ref, values):
    return np.array([(np.sum(ref < x)+.5*np.sum(ref == x)+.5)/(len(ref)+1) for x in values])


class FoundationRegression(unittest.TestCase):
    def test_frozen_foundation_ast_and_files(self):
        baseline = json.loads((ROOT / "research_specs/foundation_baseline.json").read_text())
        for name in ("sbrt/metric.py", "sbrt/data.py", "tests/test_foundation.py"):
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), baseline["files"][name])
        cases = {"sbrt/detectors.py": ("OfficialEWMABaseline", "EvidenceDetector", "Constant", "AgeOnly", "infer"),
                 "sbrt/replay.py": ("GuardedOnline", "ProtocolError"),
                 "sbrt/backtest.py": ("e00", "score_file", "run_backtest"),
                 "sbrt/profile.py": ("whitening", "run_profile"),
                 "sbrt/cli.py": ("make_synthetic",)}
        for path, names in cases.items():
            definitions = {n.name:n for n in ast.parse((ROOT/path).read_text()).body if hasattr(n, "name")}
            for name in names:
                digest = hashlib.sha256(ast.dump(definitions[name], include_attributes=False).encode()).hexdigest()
                self.assertEqual(digest, baseline["definitions"][path][name], (path, name))
        for path, origin, names in (("sbrt/history.py", "sbrt/profile.py", ("fit_ar", "residuals")),
                                    ("sbrt/folds.py", "sbrt/backtest.py", ("fixed_folds",))):
            defs = {n.name:n for n in ast.parse((ROOT/path).read_text()).body if hasattr(n, "name")}
            for name in names:
                self.assertEqual(hashlib.sha256(ast.dump(defs[name], include_attributes=False).encode()).hexdigest(),
                                 baseline["definitions"][origin][name])

    def test_observer_does_not_change_e01_or_handshake(self):
        h, o = data()
        s = Series("test", h, o, np.zeros(len(o), dtype=np.int8), np.arange(len(o)), -1)
        for name in E01:
            plain = list(replay(iter([s]), detector=name))[0][1]
            events = []
            observed = list(replay(iter([s]), detector=name, state_observer=lambda event,state: events.append(event)))[0][1]
            np.testing.assert_array_equal(plain, observed)
            self.assertEqual(events, ["initialize"]+["update"]*len(o)+["complete"])


class Stage2Numerics(unittest.TestCase):
    def test_e02_cumulative_exact_e01(self):
        h, o = data()
        original, multi = EvidenceDetector(h, "squared"), Stage2Detector(h, "E02")
        for x in o:
            expected = original.update(x)
            multi.update(x)
            self.assertEqual(expected, multi.last_features[3])

    def test_rolling_means_and_evidence(self):
        h, o = data()
        for w in (32, 128, 256):
            state = RollingMean(w)
            for t, value in enumerate(o, 1):
                self.assertAlmostEqual(state.update(float(value)), float(np.mean(o[max(0,t-w):t])), places=12)
        state = MultiScaleEvidence(h*h)
        for t, value in enumerate(o, 1):
            actual = state.update(float(value*value))
            np.testing.assert_allclose(actual, direct_evidence(o[:t]**2, h*h), rtol=2e-12, atol=2e-12)

    def test_ar_extraction_alias_and_direct_arithmetic(self):
        import sbrt.profile as profile
        self.assertIs(profile.fit_ar, fit_ar)
        self.assertIs(profile.residuals, residuals)
        # Known coefficients are a mathematical test fixture, not a fitted model claim.
        mu, sd = 2.0, 3.0
        beta = np.array([.2, .4, -.1, .2, 0., -.05])
        values = np.array([1., 2., 4., 8., 3., -2., 5., 9.])
        expected = [((values[k]-mu)/sd)-beta[0]-sum(beta[j]*(values[k-j]-mu)/sd for j in range(1,6))
                    for k in range(5, len(values))]
        np.testing.assert_allclose(residuals(values, (mu, sd, beta))[5:], expected, atol=1e-14)

    def test_ar_fit_ridge_and_unpenalized_intercept(self):
        h, _ = data()
        model = fit_ar(h[:700], 5)
        mu, sd, beta = model
        z = (h[:700]-mu)/sd
        X = np.asarray([[1.] + [z[i-j] for j in range(1,6)] for i in range(5,700)])
        gradient = X.T @ (X@beta-z[5:]) + np.r_[0., beta[1:]]
        np.testing.assert_allclose(gradient, 0., atol=1e-10)

    def test_ar_observed_lag_boundary_and_freezing(self):
        h, o = data()
        state = ARChannel(h)
        model = fit_ar(h[:700], 5)
        expected = residuals(np.r_[h, o], model)[len(h):]
        initial = (state.mu, state.sd, state.beta.tobytes(), state.ref_mu, state.ref_sd)
        for x, target in zip(o, expected):
            e, r = state.update(x)
            self.assertAlmostEqual(e, target, places=12)
            self.assertAlmostEqual(r, (e-state.ref_mu)/state.ref_sd, places=14)
        self.assertEqual(initial, (state.mu, state.sd, state.beta.tobytes(), state.ref_mu, state.ref_sd))
        np.testing.assert_allclose(state.lags, ((np.r_[h,o][-5:][::-1]-state.mu)/state.sd), rtol=0, atol=0)
        changed = h.copy()
        changed[700:] += 10
        other = ARChannel(changed)
        np.testing.assert_array_equal(state.beta, other.beta)
        self.assertEqual(state.mu, other.mu)
        self.assertNotEqual(state.ref_mu, other.ref_mu)

    def test_cdf_ties_extremes_constant_and_no_mutation(self):
        ref = np.array([-2., 1., 1., 1., 5.])
        values = np.array([-100., -2., 0., 1., 2., 5., 100.])
        cdf = EmpiricalCDF(ref)
        before = cdf.reference.tobytes()
        np.testing.assert_array_equal(cdf.transform(values), direct_midrank(ref, values))
        self.assertEqual(cdf.transform(-100), .5/6)
        self.assertEqual(cdf.transform(100), 5.5/6)
        self.assertEqual(EmpiricalCDF(np.ones(10)).transform(1), .5)
        self.assertEqual(cdf.reference.tobytes(), before)
        self.assertFalse(cdf.reference.flags.writeable)

    def test_sign_windows_direct_pearson_and_no_crossing(self):
        values = np.array(([1,1,-1,0,-1,1]*70), dtype=float)
        for w in (128,256):
            state = SignWindow(w)
            for t, value in enumerate(values, 1):
                actual = state.update(int(value))
                segment = values[max(0,t-w):t]
                a, b = segment[:-1], segment[1:]
                if len(a) < 2 or np.var(a) == 0 or np.var(b) == 0:
                    self.assertIsNone(actual)
                else:
                    self.assertAlmostEqual(actual, np.corrcoef(a,b)[0,1], places=13)
        state = SignDependence(np.array([-3., 2., -1., 4., 0.]), 0.)
        self.assertEqual(state.update(8), [0.,0.])
        self.assertEqual([w.n-1 for w in state.windows], [0,0])
        self.assertEqual(state.reference_correlation, historical_sign_correlation(np.array([-1.,1.,-1.,1.,0.])))

    def test_sign_support_degeneracy_full_C_reference(self):
        ref = np.r_[np.tile([-1.,1.], 100), np.repeat([-1.,1.], 100)]
        state = SignDependence(ref, 0.)
        expected_ref = np.corrcoef(np.sign(ref[:-1]), np.sign(ref[1:]))[0,1]
        self.assertAlmostEqual(state.reference_correlation, expected_ref, places=14)
        online = np.ones(20)
        for x in online:
            self.assertEqual(state.update(float(x)), [0.,0.])
        for counts in state.counts.values():
            self.assertEqual(counts["insufficient_pairs"], 8)
            self.assertEqual(counts["online_degenerate"], 12)
        disabled = SignDependence(np.ones(300), 1.)
        self.assertEqual(disabled.update(-5), [0.,0.])
        self.assertIsNone(disabled.reference_correlation)

    def test_all_experiments_against_offline_prefix(self):
        h, o = data()
        C = h[700:]
        raw_full = (o-h.mean())/h.std(ddof=1)
        raw_c = (o-C.mean())/C.std(ddof=1)
        ref_full = ((h-h.mean())/h.std(ddof=1))**2
        ref_c = ((C-C.mean())/C.std(ddof=1))**2
        model = fit_ar(h[:700], 5)
        ref_e = residuals(h, model)[700:]
        online_e = residuals(np.r_[h,o], model)[len(h):]
        ref_r = (ref_e-ref_e.mean())/ref_e.std(ddof=1)
        online_r = (online_e-ref_e.mean())/ref_e.std(ddof=1)
        ref_v = 2*direct_midrank(ref_r,ref_r)-1
        online_v = 2*direct_midrank(ref_r,online_r)-1
        med = np.median(ref_e)
        ref_signs = np.sign(ref_e-med)
        rho_ref = np.corrcoef(ref_signs[:-1],ref_signs[1:])[0,1]
        states = {name: Stage2Detector(h, name) for name in ("E02","E03","E04","E05","E06","E07")}
        for t, x in enumerate(o,1):
            output = {name: s.update(x) for name,s in states.items()}
            if t not in (1,8,9,31,32,33,127,128,129,255,256,257,310):
                continue
            a = direct_evidence(raw_full[:t]**2,ref_full)
            b = direct_evidence(raw_c[:t]**2,ref_c)
            c = direct_evidence(online_r[:t]**2,ref_r**2)
            loc = direct_evidence(online_v[:t],ref_v)
            energy = direct_evidence(online_v[:t]**2,ref_v**2)
            d = []
            for w in (128,256):
                signs = np.sign(online_e[max(0,t-w):t]-med)
                m = len(signs)-1
                if m < 8 or np.var(signs[:-1]) == 0 or np.var(signs[1:]) == 0:
                    d.append(0.)
                else:
                    rho = np.corrcoef(signs[:-1],signs[1:])[0,1]
                    q = np.sqrt(m)*abs(rho-rho_ref)
                    d.append(float(q/(1+q)))
            prim = {"E02":a, "E03":b, "E04":c, "E05":b+c, "E06":b+c+loc+energy,"E07":b+c+loc+energy+d}
            R,I,P,D = np.mean(b),np.mean(c),np.mean(loc+energy),np.mean(d)
            expected = {"E02":np.mean(a),"E03":R,"E04":I,"E05":(R+I)/2,"E06":(R+I+P)/3,"E07":(R+I+P+D)/4}
            for name in states:
                np.testing.assert_allclose(states[name].last_features, prim[name], rtol=2e-11, atol=2e-11)
                self.assertAlmostEqual(output[name],expected[name],places=11)

    def test_prefix_truncation_labels_series_order_and_repeats(self):
        h, o = data()
        original = Series("a",h,o,np.zeros(len(o),dtype=np.int8),np.arange(len(o)),-1)
        changed = copy.deepcopy(original)
        changed.online[129:] += 10000
        changed.target[:] = 1
        changed.tau_index = 0
        truncated = copy.deepcopy(original)
        truncated.online = truncated.online[:129]
        truncated.target = truncated.target[:129]
        truncated.time = truncated.time[:129]
        other = copy.deepcopy(original)
        other.id = "b"
        other.historical = h*2 + 1
        for name in ("E02","E03","E04","E05","E06","E07"):
            def run(records):
                return {s.id:p for s,p,_ in replay(iter(records),detector=name)}
            base = run([original,other])
            np.testing.assert_array_equal(base["a"],run([original])["a"])
            np.testing.assert_array_equal(base["a"][:129],run([changed])["a"][:129])
            np.testing.assert_array_equal(base["a"][:129],run([truncated])["a"])
            rev = run([other,original])
            for key in base:
                np.testing.assert_array_equal(base[key],rev[key])
            actual = list(replay(iter([original]), infer_fn=lambda ds: infer(ds,name)))[0][1]
            np.testing.assert_array_equal(base["a"], actual)

    def test_exact_serialization_all_states_and_frozen_cdf(self):
        h, o = data()
        for name in ("E02","E03","E04","E05","E06","E07"):
            for stop in (0,1,8,32,129,257):
                state = Stage2Detector(h,name)
                for x in o[:stop]:
                    state.update(x)
                snapshot = state.dumps()
                restored = Stage2Detector.loads(snapshot)
                self.assertEqual(snapshot,restored.dumps())
                self.assertEqual(state.array_bytes(),restored.array_bytes())
                before_cdf = restored.cdf.reference.tobytes() if restored.cdf else None
                for x in o[stop:]:
                    self.assertEqual(state.update(x),restored.update(x))
                    self.assertEqual(state.last_features,restored.last_features)
                self.assertEqual(state.dumps(),restored.dumps())
                if restored.cdf:
                    self.assertEqual(before_cdf,restored.cdf.reference.tobytes())
                    self.assertFalse(restored.cdf.reference.flags.writeable)
                if restored.ar:
                    self.assertFalse(restored.ar.beta.flags.writeable)

    def test_fail_closed_invalid_ar_and_nonfinite(self):
        with self.assertRaisesRegex(ValueError,"AR"):
            Stage2Detector(np.ones(1000),"E04")
        with self.assertRaisesRegex(ValueError,"Nonfinite"):
            Stage2Detector(data()[0],"E02").update(float("nan"))
        with self.assertRaises(ValueError):
            Stage2Detector(data()[0],"E08")
        with self.assertRaises(ValueError):
            Stage2Detector.loads(b'{"version":2,"state":{}}')


if __name__ == "__main__":
    unittest.main()
