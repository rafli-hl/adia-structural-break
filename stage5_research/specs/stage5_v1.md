Freeze **four experiments, E18–E21**, using **normal-score empirical PIT** as the sole new transformed stream.

The batch should test representation content, not model capacity:

1. **E18 — Five compact moment/dependence channels.**
2. **E19 — Two-directional scale likelihood.**
3. **E20 — Scale-evidence trajectory**, as a fixed child of E19.
4. **E21 — Historical null calibration**, as a fixed alternative to E19’s uncalibrated likelihood.

Defer tau integration. Do not combine E18 with the other experiments, introduce another whitening variant, or add a nonlinear combiner.

I found the original public discussion. Its disclosures support compact aggregation and representation-focused hypotheses, but the thread also contains corrections to earlier measurements. These are participant-reported results—not independently established ceilings or evidence that their exact construction transfers to our pipeline. [Original CrunchDAO discussion](https://forum.crunchdao.com/t/where-does-the-remaining-headroom-live-sharing-measured-ceilings-from-a-whitening-based-approach/1206)

**Why normal scores rather than bounded linear PIT:** the smoothed rank convention already prevents probabilities of zero or one. Normal scores therefore remain finite without an extra clipping hyperparameter, while supporting interpretable moment channels and a closed-form Gaussian scale *working likelihood*. Importantly, this transformation does **not** establish independence or an exactly Gaussian null. Historical calibration and diagnostics remain necessary.

The specification below is the complete proposed implementation contract. No code was implemented, no experiments were run, and no existing artifacts were changed.

# CODEX HANDOFF — STAGE 5

## 1. Scope, references and execution

Implement exactly:

```text
E18 compact normal-score moments       parent E14
E19 directional scale likelihood      parent E14
E20 scale-evidence trajectory          parent E19
E21 historical null calibration       parent E19
```

Execution order is **E18 → E19 → E20 → E21**. Run every experiment regardless of earlier scores or advancement results.

E20 and E21 are predetermined E19 ablations. They must not incorporate E18 or whichever preceding experiment performs best.

Primary comparator:

```text
E14 frozen OOF
approximately 0.5871611728

OOF SHA-256:
f617a259568cba2414f6beea1c7723747e357d00dbfee7ed92a9c8d5d1e70146
```

Use its full-precision artifact-derived score for comparisons, not the rounded value above.

Keep CV-v2 unchanged:

```text
9c6b640a0b2c9a5b3fe026d5c28618346c20539d5750e31268802264cdb56434
```

Report comparisons against E08 and E04 descriptively. Neither cloud score enters training, feature construction, selection or gates.

All existing Stage-1–4 and E08/E14 production sources and artifacts remain protected. New work belongs in separate Stage-5 namespaces.

---

## 2. Shared historical normalization and transformed stream

### 2.1 Preserve E14 completely

For history length \(H\), retain:

\[
k=\lfloor0.7H\rfloor,\qquad A=[0,k),\qquad C=[k,H).
\]

Reuse the authoritative historical AR(5), ridge 1, unpenalized intercept, A-fitted normalization and actual-observation lag updates.

Use the **uncentered innovation**, the first return from `ARChannel.update`.

Retain E14’s exact scale initialization:

\[
b_{\rm raw}=\operatorname{mean}(e_5^2,\ldots,e_{k-1}^2),
\]

\[
b=\max(b_{\rm raw},10^{-16}),\qquad
v_{\min}=\max(10^{-16},10^{-4}b),\qquad h_5=b.
\]

Replay historical positions \(j=5,\ldots,H-1\):

\[
u_j=\frac{e_j}{\sqrt{\max(h_j,v_{\min})}},
\qquad
h_{j+1}=0.99h_j+0.01e_j^2.
\]

Retain \(u_j\) for \(j\in C\), in chronological order, and retain terminal \(h_H\) for inference.

Do not change original R/I/P/D, \(J_s\), \(C_s\), their historical references, or their windows.

### 2.2 Exact frozen PIT

Let \(M=|C|\). Fit the new CDF directly to **\(u_C\)**, not to raw observations, original AR residuals, or another rank block.

For any finite \(u\), define:

\[
L(u)=\#\{u_j<u:j\in C\},
\qquad
E(u)=\#\{u_j=u:j\in C\},
\]

\[
p(u)=\frac{L(u)+\tfrac12E(u)+\tfrac12}{M+1}.
\]

Use exact numerical equality for ties. No jitter, randomization, interpolation between order statistics, online updates or leave-one-out ranks.

The sole new stream is:

\[
z_t=\Phi^{-1}(p(u_t)).
\]

Use pinned **`scipy.special.ndtri`**, which is SciPy’s inverse standard-normal CDF. Do not substitute an approximation or upgrade dependencies. [SciPy 1.17 `ndtri` documentation](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.special.ndtri.html)

Because

\[
\frac{1}{2(M+1)}\le p(u)\le1-\frac{1}{2(M+1)},
\]

no probability clipping parameter is required.

Historical \(z_C\) uses the **same inclusive CDF** evaluated on each historical \(u_j\).

Do not further z-normalize this stream globally. The historical centering needed by individual statistics is defined below.

### 2.3 Interpretation and limitations

Call this **Rosenblatt-style normalization**, not proof of complete Rosenblatt whitening.

In particular:

- Remaining temporal dependence is possible.
- Finite-sample ranks are discrete.
- Values beyond a historical endpoint share an endpoint score.
- The transformation loses information about the magnitude of extreme exceedances.

The unchanged E14 inputs retain complementary unranked energy information.

---

## 3. Shared scale ladder, aggregation and support

New Stage-5 trailing windows are exactly:

\[
\mathcal W=\{32,128,512\}.
\]

Also maintain a cumulative online prefix.

This keeps one short window, one medium window and one long window. The long window is justified by the competition’s substantial late-age weight. Do not add 8, 16, 64 or 256 to this new ladder.

**This does not change E14’s existing 32/128/256/cumulative windows.**

For causal online age \(t=1,2,\ldots\):

\[
n_w(t)=\min(t,w),\qquad n_\infty(t)=t.
\]

All windows contain online observations only. Partial windows are active subject to each statistic’s support rule.

Define the bounded evidence map:

\[
B(q)=\frac{q}{1+q},\qquad q\ge0.
\]

For three bounded trailing statistics \(a_{32},a_{128},a_{512}\) and bounded prefix statistic \(a_\infty\), define:

\[
\mathcal A(a)=
\frac{\max(a_{32},a_{128},a_{512})+a_\infty}{2}.
\]

Thus each channel produces **one aggregated feature**, not four window columns.

Unsupported per-window statistics equal zero and increment an explicit support counter. Never omit their rows.

---

## 4. E18 — Compact normal-score moment block

**Parent:** E14.
**Question:** Does the fully transformed stream expose location, shape and short-dependence changes that E14’s energy-heavy representation misses?

### Historical calibration

Define four normalized Hermite-type channels:

\[
\psi_1(z)=z,
\]

\[
\psi_2(z)=\frac{z^2-1}{\sqrt2},
\]

\[
\psi_3(z)=\frac{z^3-3z}{\sqrt6},
\]

\[
\psi_4(z)=\frac{z^4-6z^2+3}{\sqrt{24}}.
\]

These are fixed polynomial channels, **not rolling sample skewness or kurtosis estimates**. Their constants do not imply that the empirical null is exactly Gaussian.

Let:

\[
m_z=\operatorname{mean}(z_C).
\]

Define the fifth, dependence channel:

\[
\psi_{5,t}=(z_t-m_z)(z_{t-1}-m_z).
\]

Its historical reference contains only adjacent pairs wholly within C. Its online stream starts at \(t=2\): **no historical–online pair crossing**.

For each channel \(j\), calculate from its historical reference:

\[
\mu_j=\operatorname{mean}(\psi_{j,C}),
\qquad
s_j=\max\{\operatorname{SD}_{ddof=1}(\psi_{j,C}),10^{-8}\}.
\]

For online updates, first form:

\[
d_{j,t}=\frac{\psi_{j,t}-\mu_j}{s_j}.
\]

Center and divide **before** updating rolling means. This is the frozen numerical order for the new block.

### Statistics

For channels 1–4:

\[
Q_{j,w}(t)=\sqrt{n_w(t)}
\left|\operatorname{mean}_{w}(d_j)\right|.
\]

They are supported immediately at one observation.

For channel 5, a window of \(n_w\) observations contains exactly \(n_w-1\) internal adjacent pairs. Use:

\[
Q_{5,w}(t)=\sqrt{n_w(t)-1}
\left|\operatorname{mean}_{\text{internal pairs}}(d_5)\right|.
\]

Require at least **eight pairs**; otherwise its per-window score is zero. Prefix pair count is \(t-1\).

Aggregate:

\[
M_j(t)=\mathcal A\big(B(Q_{j,\cdot}(t))\big),
\qquad j=1,\ldots,5.
\]

### Model inputs

Let \(X_{14}\) denote the exact five standardized E14 inputs.

E18 inputs are:

\[
[X_{14},S(M_1),S(M_2),S(M_3),S(M_4),S(M_5)].
\]

**Ten total model inputs.** No individual window statistics enter the model.

### Cost, audit and failure mode

- Update: \(O(\log M+5|\mathcal W|)\), plus unchanged E14.
- State: \(O(M+5\sum_{w\in\mathcal W}w)\), with bounded ring buffers and prefix accumulators.
- Leakage audit: frozen CDF/references; online-only windows and pairs; no online recentering.
- Likely failure: endpoint saturation, noisy higher moments, dependence left after whitening, or near-duplication of existing energy/rank evidence.
- Standalone diagnostics: each \(M_j\), plus the fixed omnibus mean \(\frac15\sum_jM_j\). None is selectable.
- Advancement: common G5 against E14.

---

## 5. E19 — Two-directional scale likelihood

**Parent:** E14.
**Question:** Does retaining increase/decrease direction in a scale-focused likelihood add useful information beyond absolute energy evidence?

### Historical reference

Using the shared normal-score stream:

\[
m_z=\operatorname{mean}(z_C),
\qquad
q_j=(z_j-m_z)^2,
\qquad
v_0=\operatorname{mean}(q_C).
\]

This uses a **population second moment**, not sample variance.

If \(v_0\le10^{-16}\), disable the new likelihood block for that series: both new features remain zero. Record the unsupported reference; retain every prediction and all E14 inputs.

### Window likelihood

For supported histories, define:

\[
r_w(t)=\frac{\operatorname{mean}_w(q_t)}{v_0}.
\]

Require at least **eight observations** in the relevant window or prefix.

For \(r>0\):

\[
\ell(n,r)=\frac n2(r-1-\log r),
\]

\[
\ell^+(n,r)=
\begin{cases}
\ell(n,r),&r>1,\\
0,&r\le1,
\end{cases}
\]

\[
\ell^-(n,r)=
\begin{cases}
\ell(n,r),&0<r<1,\\
0,&r\ge1.
\end{cases}
\]

At \(r=1\), both are zero.

At \(r=0\), use the analytic bounded limit:

\[
B(\ell^+)=0,\qquad B(\ell^-)=1.
\]

Do not introduce a tunable ratio floor or materialize an infinite model feature.

Numerical rules:

- Near one, when \(|r-1|\le0.5\), evaluate the kernel as
  \((r-1)-\log1p(r-1)\).
- Otherwise use \(r-1-\log r\).
- Clamp a negative roundoff result for this nonnegative kernel to zero.
- A computed window mean of nonnegative \(q\) may be clamped to zero only if it lies in
  \([-10^{-12}\max(1,v_0),0)\); record it. A more negative value invalidates the implementation/run.

Aggregate directions separately:

\[
L^+(t)=\mathcal A\big(B(\ell^+_\cdot(t))\big),
\qquad
L^-(t)=\mathcal A\big(B(\ell^-_\cdot(t))\big).
\]

This is a Gaussian **working likelihood with fixed historical location**, not a calibrated p-value. A location shift can also increase its energy statistic.

### Model inputs

\[
[X_{14},S(L^+),S(L^-)].
\]

**Seven total inputs.**

### Cost, audit and failure mode

- Update: \(O(\log M+|\mathcal W|)\), plus E14.
- State: frozen CDF plus \(O(\sum w)\) rolling energy state.
- Leakage audit: \(m_z,v_0\) history-only; no rolling location refit; no candidate-tau search.
- Likely failure: likelihood saturation, residual dependence, location/scale confounding, or information already represented by I/\(J_s\).
- Standalone diagnostics: \(L^+\), \(L^-\), and \(\max(L^+,L^-)\).
- Advancement: common G5 against E14.

---

## 6. E20 — Scale-evidence trajectory

**Parent:** E19.
**Question:** Does remembering the trajectory of evidence help when conditional-scale adaptation causes current evidence to subside?

This is **E19 plus two online trajectory features**. It does not include E18.

Define current scale evidence:

\[
A_t=\max(L^+_t,L^-_t).
\]

Initialize before online inference:

\[
P_0=0,\qquad T_0=0.
\]

After calculating current E19 features:

\[
P_t=\max(P_{t-1},A_t),
\]

\[
D_t=P_t-A_t,
\]

\[
T_t=0.99T_{t-1}+0.01A_t.
\]

Here:

- \(D_t\) is drawdown from the online running maximum.
- \(T_t\) is a persistent summary of observed scale evidence.

Both are in \([0,1]\). Use them directly as defined: no bias correction, history warmup, resets, thresholds, age normalization or additional slope features.

### Model inputs

\[
[X_{14},S(L^+),S(L^-),S(D),S(T)].
\]

**Nine total inputs.**

Preserve E19’s first seven transformations exactly. Refit all logistic coefficients using the unchanged objective.

### Cost, audit and failure mode

- Incremental update over E19: \(O(1)\).
- Incremental persistent numeric state: two scalars, \(P_t,T_t\).
- Leakage audit: trajectory starts at the first online observation and never uses historical maxima, labels, tau or eventual horizon.
- Likely failure: retaining false alarms, encoding maturity without useful ranking information, or oversmoothing recent evidence.
- Standalone diagnostics: \(D_t\) and \(T_t\); current \(A_t\) is already an E19 diagnostic.
- Advancement: G5 against E14 **and** the child-ablation guard against E19 defined below.

---

## 7. E21 — Historical null calibration

**Parent:** E19.
**Question:** Does a single history-only correction for variability and serial dependence make scale likelihood more comparable across series?

E21 **replaces** E19’s two likelihood features with calibrated versions. It does not append a context variable or include E20.

### Calibration

Use the same historical \(q_j=(z_j-m_z)^2\), with mean \(v_0\).

Freeze a Bartlett-tapered covariance sum with **16 lags**:

\[
\gamma_\ell=
\frac1M\sum_{j=\ell+1}^{M}
(q_j-v_0)(q_{j-\ell}-v_0),
\qquad \ell=0,\ldots,16,
\]

where the \(\ell=0\) expression is the population variance of \(q_C\).

\[
V_{\rm raw}=
\gamma_0+
2\sum_{\ell=1}^{16}
\left(1-\frac{\ell}{17}\right)\gamma_\ell.
\]

This is a fixed-bandwidth, Bartlett/HAC-style construction. The underlying covariance-estimation approach accounts for serial dependence; the particular bandwidth and application here are our preregistered design choices, not an assertion of exact finite-sample calibration. [Newey–West primary paper](https://www.nber.org/papers/t0055)

Rules:

- Require \(M\ge34\). Otherwise use \(\eta=1\), record calibration unsupported, and retain all rows.
- If \(v_0\le10^{-16}\), use E19’s disabled-block rule.
- If
  \[
  V_{\rm raw}<-10^{-12}\max(1,\gamma_0),
  \]
  invalidate E21.
- Otherwise set \(V=\max(V_{\rm raw},0)\), recording any roundoff clamp.
- Define:
  \[
  \eta=\max\left(1,\frac{V}{2v_0^2}\right).
  \]

The lower bound of one is deliberate: **deflate overconfident evidence, but do not amplify evidence because a short, inclusively transformed historical sample appears unusually quiet**.

### Calibrated likelihood

Before applying \(B\), replace:

\[
\ell^\pm_w\quad\text{with}\quad\frac{\ell^\pm_w}{\eta}.
\]

Apply exactly the same support rules and aggregation as E19, yielding:

\[
L^+_{\rm cal},\quad L^-_{\rm cal}.
\]

The \(r=0\) bounded limit remains one for the decreasing-scale direction.

### Model inputs

\[
[X_{14},S(L^+_{\rm cal}),S(L^-_{\rm cal})].
\]

**Seven total inputs.**

Do not add \(\eta\), autocovariances, or historical descriptors as model inputs.

### Cost, audit and failure mode

- Initialization over E19: \(O(16M)\).
- Incremental online update: constant work for division by the frozen factor.
- Incremental persistent state: one calibration scalar plus diagnostic metadata.
- Leakage audit: covariance estimates and \(\eta\) use C only and never update online.
- Likely failure: noisy finite-history calibration, dependence beyond lag 16, inclusive-CDF estimation effects, or excessive suppression of genuine changes.
- Standalone diagnostics: calibrated up/down features and their maximum. Never score \(\eta\) as a competition predictor.
- Advancement: G5 against E14 and the child-ablation guard against E19.

---

## 8. Common training and nested-model requirements

### Outer CV and weights

Use the existing five CV-v2 folds.

Within each training split, recompute:

\[
W_t=N_+(t)N_-(t),\qquad Z=\sum_{t:W_t>0}W_t,
\]

\[
\omega_{i,t}=\frac{W_t}{2ZN_{y_{i,t}}(t)}.
\]

Exclude \(W_t=0\) rows from fitting only. Predict every validation observation.

No subsampling, class weights, inverse-series-length weights, or partition-specific training.

Reuse the trusted weighting implementation and its \(10^{-12}\) identity checks.

### Standardization

Preserve the E14 five-column transformation exactly for every outer fold:

\[
X_{14}=[z_R,z_I,z_P,z_D,S(C_s)].
\]

Use the same eligible rows, weights, canonical row order and numerical procedures as E14.

Standardize each new feature using training-only weighted mean and population variance:

\[
S(x)=\frac{x-\mu_\omega}{\sigma_\omega}.
\]

Exactly zero variance means an identically zero standardized column, including on validation.

Compute new-column standardization separately so it cannot change existing transformations.

### Objective and solver

For every candidate:

\[
\mathcal L(b,\beta)=
\sum_i\omega_i
\left[\log(1+\exp(s_i))-y_is_i\right]
+\frac{0.01}{2}\|\beta\|_2^2,
\]

\[
s_i=b+X_i^\top\beta.
\]

Reuse the existing stable objective and analytic gradient.

Freeze:

```text
float64
zero initialization
unbounded L-BFGS-B
intercept unpenalized
maxiter = 1000
gtol = 1e-6
ftol = 0
maxcor = 10
maxls = 20
maxfun = 20001
native numerical threads = 1
```

Require optimizer success, finite results and final gradient infinity norm \(\le10^{-6}\). No convergence-driven changes to settings.

### Required nesting tests

Within \(10^{-14}\) for predictions and objective:

- E14 coefficients embedded in each candidate with new coefficients zero reproduce E14.
- E19 embedded in E20 with trajectory coefficients zero reproduces E19.
- With \(\eta=1\), E21’s features, preprocessing and embedded-model calculations reproduce E19.

These tests preserve transformations; the actual candidate fits refit all coefficients.

---

## 9. Exact causal update order

For each online observation:

1. Compute the frozen-AR innovation from actual prior observations.
2. Read pre-update scale state and calculate \(u_t\).
3. Update unchanged E14 conditional-energy evidence.
4. Calculate \(p_t,z_t\) from the frozen CDF.
5. Update the candidate’s new moment/likelihood states.
6. For E20, update \(P_t,D_t,T_t\).
7. Update unchanged R/I/P/D states.
8. Construct E14 inputs and the candidate’s allowed additions.
9. Apply frozen fold preprocessing and logistic scoring.
10. Advance \(h\) using the current innovation.
11. Yield the already-computed prediction.
12. Only then consume the next observation.

Initialization, forecast arithmetic and original E14 outputs must be regression-tested against the existing implementation.

No online CDF/reference refitting, horizon access, ID-derived predictors, labels, tau, partition labels, fold IDs, or cross-series mutable inference state.

---

## 10. Frozen A/B stability partition

Create this **before feature export or any new score is inspected**.

Use normalized string IDs and existing CV-v2 stratum/fold assignments.

Frozen partition seed identifier:

```text
20260924
```

Procedure:

1. Visit the six strata in this order:

   ```text
   no_break
   break_1_64
   break_65_128
   break_129_256
   break_257_512
   break_513_plus
   ```

2. Within each stratum, visit folds 0–4.
3. Within each stratum/fold cell, sort series by:

   ```text
   SHA256(UTF8("stage5_v1|20260924|" + normalized_id))
   ```

   Compare digest bytes lexicographically; break a hash tie by normalized ID.

4. Concatenate those ordered cells.
5. Assign global positions 0,2,4,… to A and 1,3,5,… to B. **Do not restart parity at cell boundaries.**

This yields 5,000 series per partition, with counts differing by at most one within each existing stratum/fold cell and within each stratum.

Persist IDs, stratum, fold, partition, algorithm metadata and hashes. Reject incompatible overwrite.

For each candidate, evaluate the existing OOF predictions separately on A and B. Recompute within-age positive/negative counts and pair weights **inside each partition**.

These are **stability slices, not untouched holdouts**. Earlier research used all series, and the OOF models underlying the slices share training data. Positive results on both halves are not independent replication.

Do not train extra partition models or regenerate partitions.

---

## 11. Frozen G5 advancement and selection

Retain the meaningful **+0.002** minimum effect.

A candidate advances only if all conditions pass:

1. Valid, complete OOF coverage.
2. Pooled delta versus E14 \(\ge+0.002\).
3. At least four of five paired fold deltas versus E14 are strictly positive.
4. Combined ages \(\ge129\): delta versus E14 \(\ge0\).
5. Combined ages \(\le64\): delta versus E14 \(\ge-0.005\).
6. Retained historical early guard: ages \(\le64\) versus E01 squared \(\ge-0.005\).
7. Partition-A delta versus E14 \(>0\).
8. Partition-B delta versus E14 \(>0\).
9. Conditional paired-bootstrap \(Q_{0.01}(\Delta^*)>0\).

The A/B guard is added to reject gains with opposing directions across the two fixed series subsets. The E01 guard prevents cumulative erosion of early performance across successive stages.

### Bootstrap

For candidates passing all non-bootstrap conditions, reuse the existing procedure:

```text
999 replicates
seed 20260920
paired whole-series resampling
same six CV strata
same multiplicities for all compared predictions
fixed OOF predictions; no refitting
pair weights recomputed within each replicate
linear quantiles
```

Report the 95% interval, \(Q_{0.01}\), all replicates and multiplicity hash.

The primary candidate family remains **four**, including invalid and non-advancing candidates. The 1% guard is a strict conditional multiplicity precaution, **not a guarantee that cumulative research-selection bias has been removed**.

### Child-ablation guard

E20 and E21 must additionally demonstrate improvement over E19:

- Pooled delta versus E19 \(>0\).
- Lower endpoint of the **95% paired conditional bootstrap interval** versus E19 \(>0\), using the same 999 multiplicities.

Run this additional comparison only for otherwise serious advancement candidates. It is an intersection requirement, not another selectable experiment.

If E19 has no valid complete OOF artifact, still execute E20/E21, but their child-ablation guard cannot pass.

This prevents a child from receiving advancement credit solely for an inherited E19 improvement.

### Winner

Among qualifying candidates:

1. Highest full-precision pooled OOF TS-AUC.
2. Exact tie: fewer model inputs.
3. Remaining exact tie: lower experiment ID.

If none qualify, retain E14. Never combine candidates.

---

## 12. Required diagnostics

For every candidate, report:

- Pooled score and deltas versus E14, E08, E04 and its ablation parent.
- Five fold scores/deltas.
- Eight existing age buckets.
- Combined \(\le64,\ge129,\ge257\), using exact pair-weighted aggregation.
- A/B scores, deltas, series counts and pair-weight totals.
- All fold coefficients, preprocessing, convergence and objective diagnostics.
- Actual end-to-end replay time, historical initialization, update throughput, training/scoring/bootstrap times, peak memory, state and shared-model sizes.
- Source/config/spec/fold/partition/data/cache/model/OOF hashes.
- Every support, floor, numerical-clamp and degeneracy count.

Additional block diagnostics:

**Standalone scores:** use only the explicitly listed bounded scores. They are descriptive and never selectable. Do not score raw \(C_s\).

**Correlation:** for each new model feature, report correlation with I, \(J_s\) and P, both uncentered and after removing per-age means. Use evaluator-only row weights

\[
a_{i,t}=\frac{W_t}{Z\,N_{\rm active}(t)}
\]

on eligible ages. This gives age mass \(W_t/Z\) without allowing longer age groups to dominate simply through row count. These diagnostic weights never enter fitting.

Flag absolute age-centered correlation \(\ge0.98\) as **possible near-duplication**, not proof of redundancy or a post-hoc selection veto. Gains from a different encoding of the same evidence must not be described as discovery of an independent break mechanism.

**Historical reference distributions:** report per-series reference means/SDs, \(v_0\), and E21’s \(\gamma_0,V,\eta\), summarized at quantiles:

```text
0, .05, .25, .50, .75, .95, 1
```

Also replay the new bounded evidence on chronological \(z_C\), resetting online accumulators at C’s start. Label this explicitly **in-sample historical-reference replay**, because the CDF and references were fitted on that same C. It is not a held-out null-calibration test and cannot supply additional thresholds.

**Saturation and ties:**

- Fractions strictly below/above historical \(u_C\) endpoints.
- Endpoint PIT/normal-score occupancy.
- Feature fractions at zero, one, and \(\ge0.99\).
- Exact within-age prediction tie rate, defined as tied unordered row pairs divided by all unordered row pairs.
- \(r=0\) likelihood-limit counts.
- Unsupported-reference and insufficient-window counts.

No diagnostics may trigger feature revisions or additional experiments.

---

## 13. Engineering, tests and freeze

Use new namespaces, for example:

```text
stage5_research/
  specs/stage5_v1.md
  contract.py
  features.py
  models.py
  pipeline.py
stage5_tests/
run_stage5.py
stage5_results/
```

Reuse trusted AR, E14 state, training weights, objective, scorer and bootstrap routines. Do not modify protected modules or monkey-patch their window constants.

One shared causal export is permitted. Approved causal quantities are:

```text
R, I, P, D, J_s
M1, M2, M3, M4, M5
L_plus, L_minus
drawdown, trajectory_ewma
L_plus_calibrated, L_minus_calibrated
```

Evaluator keys/labels/folds may accompany them. Historical calibration belongs in a separate per-series artifact. No globally standardized model columns may be cached.

Audit exported R/I/P/D/\(J_s\) against frozen Stage-4 values exactly.

Required tests include:

- All existing research and deployment tests.
- PIT ties/endpoints, finite normal scores, inclusive historical ranks and frozen CDF.
- Exact E14 residual units, split, scale timing and terminal history state.
- Hermite formulas and historical channel normalization.
- Internal-pair counts and no historical–online crossing.
- Partial windows and rolling/prefix agreement with direct calculations.
- Likelihood formula, direction, near-one evaluation and zero-energy limit.
- Trajectory recursions and online-only initialization.
- HAC normalization, taper, support rules and \(\eta\ge1\).
- Exact allowlists and training-only preprocessing/weights.
- Held-out-label isolation, deterministic fitting and nesting tests.
- Prefix/truncation/order invariance, handshake and one prediction per observation.
- Exact state serialization/resumption and model binding.
- Partition determinism, balance, disjointness, complete coverage and immutability.
- All G5 boundary conditions and paired bootstrap multiplicities.
- Protected artifact integrity and complete 10,000-series/5,036,517-row OOF coverage.

For new incremental numerical statistics, compare against direct calculations with `atol=rtol=1e-10`; do not weaken existing tests. Serialization, repeated inference, protected feature audits and deterministic model bytes require exact equality. Nesting retains the \(10^{-14}\) requirement.

Use synthetic-test seed **20260925**.

Before full-data execution:

1. Finish implementation corrections.
2. Run all tests.
3. Freeze this specification, resolved configuration, dependency versions, thread environment, source hashes and A/B manifest.
4. Verify E14/CV-v2 and protected production/research hashes.
5. Only then export and measure E18–E21.

No modeling-setting changes after scores are visible. Necessary engineering corrections require an explicit audit and renewed validation; they cannot become a route to score-driven changes.

Provide reproducible commands for validation, freeze, export, individual execution, the fixed batch, comparisons and report generation.

**Stop after the complete G5 report and reference decision. Do not perform a production refit, combine candidates, add tau integration, start a nonlinear-model stage, or submit another cloud model.**