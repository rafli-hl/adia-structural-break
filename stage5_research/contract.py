"""Resolved, non-tunable Stage-5 contract; authoritative prose is pinned."""
from sbrt.logistic import SOLVER_OPTIONS
from stage4_research.contract import THREADS, CV_HASH, GATE as G4

IDS = ("E18", "E19", "E20", "E21")
PARENTS = dict(E18="E14", E19="E14", E20="E19", E21="E19")
BASE = ("R", "I", "P", "D", "J_s")
MOMENTS = tuple(f"M{j}" for j in range(1, 6))
LIKELIHOOD = ("L_plus", "L_minus")
TRAJECTORY = ("drawdown", "trajectory_ewma")
CALIBRATED = ("L_plus_calibrated", "L_minus_calibrated")
EXTRA = dict(E18=MOMENTS, E19=LIKELIHOOD, E20=LIKELIHOOD+TRAJECTORY, E21=CALIBRATED)
RAW = BASE+MOMENTS+LIKELIHOOD+TRAJECTORY+CALIBRATED
INPUTS = {k:["z_R", "z_I", "z_P", "z_D", "S(C_s)"]+[f"S({v})" for v in cols] for k,cols in EXTRA.items()}
WINDOWS = (32, 128, 512)
E14_HASH = "f617a259568cba2414f6beea1c7723747e357d00dbfee7ed92a9c8d5d1e70146"
SPEC_HASH = "c6b721fc0fd853bfcdb7bbc3b570d0bd32f9168344887098c589c2b63e6a8730"
GATE = dict(G4, reference="E14", partition_A_minimum_strict=0., partition_B_minimum_strict=0.,
            child_pooled_minimum_strict=0., child_bootstrap_lower_strict=0.)
CONFIG = dict(version="stage5_v1", spec_sha256=SPEC_HASH, execution_order=list(IDS),
    inputs=INPUTS, raw_export=list(RAW), parents=PARENTS, E14_oof_sha256=E14_HASH, cv2_sha256=CV_HASH,
    expected_series=10000, expected_rows=5036517, windows=list(WINDOWS), partial=True,
    stream=dict(parent="unchanged E14 slow uncentered innovation / predictive sigma", alpha=.01,
        CDF="inclusive (L+E/2+1/2)/(M+1) on chronological u_C", inverse="scipy.special.ndtri",
        clipping=None, online_refit=False, global_renormalization=False),
    moments=dict(channels=["z", "(z^2-1)/sqrt(2)", "(z^3-3z)/sqrt(6)",
        "(z^4-6z^2+3)/sqrt(24)", "(z_t-mean(z_C))*(z_previous-mean(z_C))"],
        reference_ddof=1, sd_floor=1e-8, center_before_accumulation=True,
        minimum_observations=1, minimum_internal_pairs=8, history_online_pair=False),
    evidence=dict(map="q/(1+q)", aggregation="(max(32,128,512)+prefix)/2"),
    likelihood=dict(v0="mean((z_C-mean(z_C))^2)", disabled_v0=1e-16, minimum_observations=8,
        kernel="n/2*(r-1-log(r))", near_one=.5, near_one_kernel="delta-log1p(delta)",
        negative_mean_tolerance=1e-12, zero_ratio=[0.,1.], negative_kernel_clamp=0.),
    trajectory=dict(initial_peak=0., initial_ewma=0., alpha=.01, online_only=True,
        inputs=["peak-current", "0.99*previous_ewma+0.01*current"]),
    calibration=dict(lags=16, denominator="M", taper="1-lag/17", minimum_M=34,
        eta="max(1,V/(2*v0^2))", negative_V_tolerance=1e-12, eta_as_input=False),
    training=dict(weights="unchanged E08 training_weights; all eligible rows", base="unchanged E14 separately",
        extras="training weighted population variance; zero variance output zero", l2=.01,
        solver="L-BFGS-B", options=SOLVER_OPTIONS, zero_initialization=True, float64=True,
        unpenalized_intercept=True, gradient_limit=1e-6, success_required=True, native_threads=1),
    partition=dict(seed_identifier=20260924, key="stage5_v1|20260924|normalized_id",
        order="STRATA then fold 0..4 then SHA256 bytes then ID", parity="global even A, odd B"),
    diagnostics=dict(quantiles=[0,.05,.25,.5,.75,.95,1], correlation="W_t/(Z*N_active(t))",
        near_duplicate_threshold=.98, historical_replay="in-sample inclusive C reference; not held out"),
    gate=GATE, synthetic_seed=20260925, tolerances=dict(incremental=1e-10,nested=1e-14,weights=1e-12),
    winner="highest pooled qualifying; exact tie fewer inputs then lower ID; otherwise E14",
    stop="Complete G5 report only; no production, cloud, combination, tau integration or Stage 6")
