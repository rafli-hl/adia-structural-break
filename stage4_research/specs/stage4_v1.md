# Frozen Stage-4 implementation contract

This is the local transcription of the supplied Stage-4 contract, including the
user's E14 diagnostic clarification. It is immutable for the measured batch.
Resolved numerical settings are also recorded from `stage4_research.contract.CONFIG`.

## Scope and references

Execute E14, E15, E16, E17, in that order regardless of prior scores. E08 is the
primary reference (0.5846027902494327); report E04 (0.5818721070282651) as well.
E08 OOF SHA-256: 99fec6a4d33573f4e86ca86b54e8e277d288896d1db5b5c08bc3cd85e75c0010.
CV-v2 SHA-256: 9c6b640a0b2c9a5b3fe026d5c28618346c20539d5750e31268802264cdb56434.
Do not change folds, old sources, numerical semantics, or old artifacts.

E14 parent E08; E15 parent E14 (rate only); E16 parent E14 (robust update only);
E17 parent E14 (one rank contrast only). E17 always uses ordinary slow whitening.
No Stage-3 extensions, additional context, age inputs, new windows/AR orders,
trees, neural models, pairwise losses, candidate-break statistics, tuning,
production refits, or cloud submissions.

## Exact historical and online construction

For H history observations, k=floor(0.7 H), A=[0,k), C=[k,H).
Reuse frozen ARChannel / fit_ar / residuals: AR(5), ridge 1, unpenalized intercept,
fit A only in A mean/sample-SD normalized units, observed lags only.
Use uncentered AR innovations e (first ARChannel.update return), not the second
reference-standardized return. e_A=e[5:k], e_C=e[k:H]; exclude first five positions.

b_raw=mean(e_A^2), b=max(b_raw,1e-16), v_min=max(1e-16,1e-4 b), h_5=b.
Replay j=5,...,H-1 chronologically, retaining terminal h_H for first online point.
A warms state only. For C, u_j=e_j/sqrt(max(h_j,v_min)), then update h.
No current innovation in its denominator. C reference moments/CDF may use all C;
AR parameters, b, robust calibration, and state entering C use A only.

Ordinary slow alpha=.01; ordinary fast alpha=.05:
h_next=(1-alpha)h+alpha e^2, identically in warmup, C, and online.
No cap, reset, stopping rule, refit, or detector feedback. Floor denominator only.

Robust slow alpha=.01: c=quantile(e_A^2,.99,method='linear'),
m_c=mean(min(e_A^2,c)). If m_c>1e-16, g(e)=(b/m_c)min(e^2,c).
Otherwise ordinary g(e)=e^2, record unsupported, retain all rows.
h_next=.99h+.01g(e). Freeze c and b/m_c. Never clip e, u or evidence numerator.

For each online x: frozen AR update -> read old forecast -> u=e/sqrt(v) ->
update evidence -> construct inputs and logistic score -> advance h using e ->
yield score before consuming another observation. After-yield state stores h_next.
No online length, labels, IDs, or cross-series mutable state reaches inference.

## Evidence and inputs

mu_u=mean(u_C), s_u=max(sampleSD(u_C,ddof=1),1e-8), a=(u-mu_u)/s_u.
Reuse exact MultiScaleEvidence with reference a_C^2, online a_t^2:
windows 32,128,256,cumulative; partial windows immediately; online windows contain
online points only. q=sqrt(n)*abs(mean_online-mean_reference)/max(sampleSD_reference,1e-8);
score=q/(1+q). Cumulative mean uses unchanged recursive arithmetic.
J is the arithmetic mean of four components. J_s slow, J_f fast, J_r robust.

For E17, frozen CDF of slow a_C: F(a)=(#less+.5#equal+.5)/(M+1), v=2F(a)-1.
Historical reference v_C uses the same inclusive CDF on a_C itself.
Use unchanged four-scale location v and energy v^2 evidence; P_s is sum of all
eight components / 8 in the original ordering. No online CDF refit or normal scores.

C_s=(J_s-I)/sqrt(2), C_f=(J_f-I)/sqrt(2), C_r=(J_r-I)/sqrt(2),
K_s=(P_s-P)/sqrt(2).
E14: z_R,z_I,z_P,z_D,S(C_s).
E15: z_R,z_I,z_P,z_D,S(C_f).
E16: z_R,z_I,z_P,z_D,S(C_r).
E17: z_R,z_I,z_P,z_D,S(C_s),S(K_s).
No other inputs. Preserve E08 base transformations independently of extensions.
E17's first five transformations must match E14.

### E14 clarification (supersedes only the inconsistent diagnostic sentence)

The standalone diagnostic is **J_s versus E04 in TS-AUC terms**.
C_s remains unchanged as E14's extension. Never score C_s or J_s-I as standalone
competition predictions. Standalone E14=J_s, E15=J_f, E16=J_r, E17=P_s.
These are descriptive only and never selectable. Contrast distribution/correlation
may be descriptive but is not a standalone TS-AUC selection criterion.

## Training and support

For each unchanged outer fold use training rows only. At time t count N_pos,N_neg;
W_t=N_pos*N_neg, Z=sum eligible W. omega=W_t/(2 Z N_y(t)), W=0 rows excluded from
training only. Predict every validation row. No subsampling, inverse-length weights,
class weights, validation calibration, or validation-guided convergence.
Verify weight total, per-age mass and half-class masses within 1e-12.
Weighted means/population variances; exactly constant columns transform to zero
also on validation. Invoke E08's four-column standardization separately and use
separate extension standardization with canonical column-major training arrays.

Reuse objective sum(omega*logistic_loss)+.01/2*dot(beta,beta); intercept unpenalized.
Float64, zeros initialization, unbounded scipy L-BFGS-B, maxiter1000, gtol1e-6,
ftol0, maxcor10, maxls20, maxfun20001, one native numerical thread.
Require optimizer success, finite parameters/objective/gradient and final gradient
infinity norm<=1e-6. Failure invalidates candidate; no settings changes/retry.

AR failure, C<2 or any nonfinite residual, forecast, transformed/reference value,
feature, coefficient or prediction invalidates affected candidate, with no fallback
except robust support rule above. Record all baseline/forecast/reference-SD floors,
clipping, normalization factors and support counts. Zero extension variance -> zero.
Invalid branches must not prevent execution of unaffected preregistered branches.

Nested regression: embedded E08 plus zero extensions reproduces predictions and
objective within 1e-14. Embedded E14 in E17 with zero rank coefficient same tolerance.
Alpha=0 audit only on well-conditioned inactive-floor synthetic data: J versus E04
within 1e-12, no alpha-zero fitted candidate.

## G4 and selection

Gate against frozen E08: valid complete coverage; pooled delta>=.002; at least 4/5
strictly positive fold deltas; age>=129 delta>=0; age<=64 delta>=-.005;
age<=64 versus E01 squared>=-.005; then Q_0.01(bootstrap deltas)>0.
For candidates passing pre-bootstrap conditions run unchanged 999-replicate paired
whole-series conditional bootstrap, seed20260920, same six CV strata, same candidate
and parent multiplicities, recomputed pair weights, fixed OOF, no refitting.
Linear quantiles; report 95% interval, Q_.01, all replicates and multiplicity hash.
Family remains four even for invalid/non-bootstrap candidates; 1% guard approximates
4% familywise conditional protection, not all historical research-selection risk.
E04, age>=257 and ablation-parent comparisons are mandatory diagnostics, not gates.
Winner: highest pooled qualifying score; exact tie fewer model inputs, then lower ID.
If none qualify retain E08. No combinations or ensemble.

## Diagnostics, performance, artifacts

For all four: pooled, five folds, eight existing age buckets, combined <=64, >=129,
>=257, scores/deltas versus E08, E04 and ablation parent. Combined uses exact pair
weights, not bucket averages. Standalone J_s,J_f,J_r versus E04; P_s versus original P.

Historical squared lag-1 Pearson ACF before (E04-standardized r_C) and after (a_C)
using correlation_from_sums and centered-sum support threshold1e-12. Report quantiles
and differences. Fixed original-rho groups <=.1, >.1, unsupported: series counts,
pair weights, candidate/E08 TS-AUC. Descriptive groups are not features or AUC decomposition.

Break absorption only after all frozen predictions: elapsed post-break bins 1-32,
33-128,129-256,257+. v_break is floored forecast before first positive. Per-series
available-bin means of e_t^2/v_break, e_t^2/v_t, v_t/v_break; across-series median,
quartiles, series/observation counts. Use partial bins, no imputation.

Report coefficients, preprocessing means/stds, convergence, contrast correlations,
all support/degeneracy counts, initialization and update timings, genuine end-to-end
stream inference (not cached model throughput), export/training/scoring/bootstrap/total
walls, state array and serialized bytes, shared model sizes. Model input counts5,5,5,6;
primitive counts22,22,22,30. Record source/config/spec/data/fold/cache/model/OOF hashes.

One causal export may contain only R,I,P,D,J_s,J_f,J_r,P_s and evaluator keys/labels/fold.
Historical support/calibration separate per-series; no globally standardized columns.
Separate evaluator-only e^2 and slow/fast/robust pre-update forecasts permitted.
Audit all R/I/P/D exactly against existing E07 artifacts.

## Validation and execution

Test exact AR boundary/units, first-five exclusion, A/C warmup/continuation, C and online
denominator causality, ordinary/robust recursions, quantile/support/floors/unclipped
numerator, direct rolling/cumulative equivalence, CDF ties/extremes/inclusive reference
and immutability, allowlists/contrasts, train-only weights/preprocessing, held-out
isolation, both nested equivalences, alpha-zero audit, determinism/save-load,
prefix/truncation/order/handshake/coverage, exact state resume with model hash,
10,000-series/5,036,517-row OOF coverage, protected artifacts, bootstrap/G4 boundaries.
Synthetic seed20260922. Do not weaken old tests.

Run old and new tests; freeze resolved config/spec/source/dependencies/threads and
protected hashes before full-data measurement. Verify unchanged during/after runs.
Necessary engineering corrections must be audited and revalidated, never score-driven.
Commands: validate, freeze, export, run, batch, compare, report. Never overwrite
completed artifacts; verify compatible reuse. Stop after all four and G4 report.
