"""Resolved Stage-4 contract. No validation-driven options."""
from sbrt.logistic import SOLVER_OPTIONS

IDS = ("E14", "E15", "E16", "E17")
BASE = ("R", "I", "P", "D")
RAW = (*BASE, "J_s", "J_f", "J_r", "P_s")
STREAMS = {"E14": ("slow",), "E15": ("fast",), "E16": ("robust",), "E17": ("slow",)}
INPUTS = {name: ["z_R", "z_I", "z_P", "z_D", f"S(C_{suffix})"]
          for name, suffix in (("E14", "s"), ("E15", "f"), ("E16", "r"))}
INPUTS["E17"] = ["z_R", "z_I", "z_P", "z_D", "S(C_s)", "S(K_s)"]
PARENTS = {"E14": "E08", "E15": "E14", "E16": "E14", "E17": "E14"}
STANDALONE = dict(E14="J_s", E15="J_f", E16="J_r", E17="P_s")
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
E08_HASH = "99fec6a4d33573f4e86ca86b54e8e277d288896d1db5b5c08bc3cd85e75c0010"
CV_HASH = "9c6b640a0b2c9a5b3fe026d5c28618346c20539d5750e31268802264cdb56434"
GATE = dict(minimum_pooled_delta=.002, minimum_positive_folds=4,
            minimum_age_ge129_delta=0., minimum_age_le64_delta=-.005,
            minimum_age_le64_vs_e01_delta=-.005, bootstrap_replicates=999,
            bootstrap_seed=20260920, bootstrap_interval=[.025, .975],
            bootstrap_scope="paired whole-series conditional, same six CV-v2 strata; fixed OOF, no refit",
            family_size=4, quantile=.01, quantile_method="linear")
CONFIG = dict(version="stage4_v1", execution_order=list(IDS), inputs=INPUTS,
    parents=PARENTS, standalone=STANDALONE, standalone_selectable=False,
    clarification="E14 standalone J_s versus E04; never C_s or J_s-I",
    E08_score=.5846027902494327, E08_oof_sha256=E08_HASH, cv2_sha256=CV_HASH,
    expected_series=10000, expected_rows=5036517,
    historical=dict(split="floor(0.7*H)", ar_order=5, ridge=1., intercept_penalized=False,
        residual_units="uncentered AR training-normalized observations", exclude_first=5,
        baseline="mean(e_A**2)", baseline_floor=1e-16,
        forecast_floor="max(1e-16,1e-4*b)", initial="h_5=b; replay through H-1; retain h_H",
        slow_alpha=.01, fast_alpha=.05, robust_alpha=.01,
        robust_quantile=.99, robust_quantile_method="linear", robust_support=1e-16,
        robust_driver="(b/m_c)*min(e**2,c); ordinary fallback iff m_c<=1e-16"),
    evidence=dict(windows=[32,128,256,"cumulative"], partial_windows=True,
        reference="C only; mean and ddof=1 SD floor 1e-8", energy="unchanged MultiScaleEvidence(a*a)",
        rank="unchanged smoothed empirical midrank; 8 location/energy components", cdf_online_refit=False),
    contrasts=dict(C_s="(J_s-I)/sqrt(2)", C_f="(J_f-I)/sqrt(2)", C_r="(J_r-I)/sqrt(2)", K_s="(P_s-P)/sqrt(2)"),
    training=dict(weighting="unchanged E08 training_weights", base_preprocessing="unchanged E08 separately",
        extension_preprocessing="weighted population variance; constant output zero", l2=.01,
        objective="sum omega*logistic_loss + .01/2*||beta||^2; intercept unpenalized",
        solver="L-BFGS-B, unbounded, float64, zero initialization", options=SOLVER_OPTIONS,
        require_success=True, gradient_infinity_limit=1e-6, threads=1),
    tolerances=dict(weights=1e-12,nested=1e-14,alpha_zero=1e-12),
    diagnostics=dict(historical_acf_groups=["rho<=0.1","rho>0.1","unsupported"],
        correlation_support=1e-12, absorption_bins=["1-32","33-128","129-256","257+"],
        absorption_statistics=["e_squared/v_break","e_squared/v_t","v_t/v_break"]),
    gate=GATE, synthetic_seed=20260922,
    winner="highest pooled qualifying; exact tie fewer inputs, then lower ID; else E08",
    stop="After G4 report; no production refit, cloud submission, or new research")
