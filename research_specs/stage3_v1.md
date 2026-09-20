I recommend **five experiments, E09–E13**, all independent children of E08. Keep E08 fixed as the primary comparator throughout the batch.

The priority order should be:

1. Historical predictability gating.
2. Innovation-scale decomposition.
3. Empirical-rank decomposition.
4. Age-dependent weighting.
5. One shallow nonlinear combiner.
6. **Defer pairwise ranking.**

Do not combine successful extensions during this batch. A model containing context, scale decomposition, age interactions, and trees would make it difficult to determine which change helped—and would invite selection on the same OOF results.

I inspected the existing definitions and saved diagnostics read-only. **No code was written, experiments run, or Stage-2 artifacts modified.**

## Research decisions

### Start with one historical context variable

Use only the existing AR(5) historical holdout MSE ratio.

This is the most direct measurement of the hypothesized mechanism: whether the frozen predictor improves over a location-only forecast for that series. It is more informative for the first gating experiment than adding ACF, kurtosis, skewness, and coefficient summaries simultaneously.

The saved Stage-2 subgroup diagnostics strengthen this choice:

* Historical MSE ratio `<0.9`: E04 − E03 was approximately **+0.075518**.
* Ratio `0.9–1.0`: approximately **+0.002053**.
* Ratio `>1.0`: approximately **−0.001412**.

These subgroup results are descriptive and do not decompose pooled TS-AUC, but they directly motivate testing predictability-dependent weighting. Saved Stage-2 diagnostics (`D:/adia-structural-break/stage2_results/comparisons/side_diagnostics.json`).

Do not add a standalone context main effect in this experiment. The question is whether context should modify the interpretation of evidence, not whether historical properties predict the series’ break prevalence.

### Decompose blocks without inadvertently duplicating their average

Replacing standardized `I` with four separately standardized scales changes both the information available and the regularization geometry.

Instead, retain E08’s four inputs and add **three zero-sum innovation-scale contrasts**. This exposes exactly the additional information contained in the four scales while preserving E08 as the zero-extension special case.

Use the same principle for the empirical-rank block: retain `P`, then add one energy-versus-location contrast.

This does not make the new columns empirically orthogonal. It avoids the simpler confound of adding redundant copies of the existing average.

### Put rank decomposition before age adaptation

The existing saved diagnostics report:

```text
P_location standalone TS-AUC = 0.5017526392
P_energy   standalone TS-AUC = 0.5739556073
```

That is direct evidence that averaging the two may discard useful information. It justifies one contrast—not automatically discarding location evidence. Saved rank-subblock diagnostics (`D:/adia-structural-break/stage2_results/comparisons/side_diagnostics.json`).

Age-dependent weighting is plausible, but the observed age pattern does not establish that the optimal coefficients vary with age. Longer prefixes can simply contain more informative evidence under constant coefficients. Test age interactions as a separate hypothesis.

### Test the nonlinear model on the original four blocks

E13 should receive **only the established E08 inputs**.

Do not choose its inputs from the winners of E09–E12. This isolates whether a constrained nonlinear mapping improves on logistic combination without introducing a score-dependent feature-selection stage.

### Defer pairwise ranking

The objective is mathematically well motivated:

$$
\mathbb E_{t\sim W_t/Z}
\mathbb E_{\substack{i\sim\mathrm{Uniform}(\mathrm{pos}_t)\\
j\sim\mathrm{Uniform}(\mathrm{neg}_t)}}
\left[\log(1+\exp(-(f_i-f_j)))\right].
$$

Sampling ages by \(W_t/Z\), then a positive and negative series uniformly at that age, would target the desired within-age pair distribution without enumerating all pairs.

Nevertheless, it introduces a new optimization objective, pair-sampling variability, and additional computational choices before we know whether representation aggregation and global weighting are the main limitations. **It is not included in this batch.**

### Retain the +0.002 advancement threshold

Do not lower the threshold. E08 itself exceeded it against E04, and the current evidence does not establish that a smaller gain would be reliably worth advancing.

I would retain the practical threshold and add a **fixed five-comparison uncertainty guard**. The precise rule is below. This addresses the current batch’s multiplicity; it does not erase the broader risk from repeatedly using the same research population.

---

# CODEX HANDOFF — STAGE 3

This section is the frozen engineering and experimental contract for the selected batch.

## 1. Scope and immutable references

Implement and execute exactly:

```text
E09 historical predictability interactions
E10 innovation-scale contrasts
E11 empirical-rank contrast
E12 causal-age interactions
E13 shallow nonlinear four-block combiner
```

Execution order:

```text
E09 → E10 → E11 → E12 → E13
```

**Every experiment’s parent and primary comparator is E08.** This is an execution sequence, not an accumulating feature sequence.

Run all five preregistered experiments regardless of earlier advancement outcomes. Do not merge extensions, change settings, or alter E13’s inputs after seeing results.

Reference:

```text
E08 pooled OOF TS-AUC:
0.5846027902494327

E08 OOF artifact SHA-256:
99fec6a4d33573f4e86ca86b54e8e277d288896d1db5b5c08bc3cd85e75c0010

Immutable CV-v2 manifest SHA-256:
9c6b640a0b2c9a5b3fe026d5c28618346c20539d5750e31268802264cdb56434
```

Preserve:

* All Stage-2 feature and detector definitions.
* Official scorer, replay handshake, no-lookahead guards, label interpretation, and coverage requirements.
* Existing CV-v2 assignments; do not recreate or overwrite the manifest.
* Existing E01–E08 predictions, models, specifications, and reports.

Use the currently pinned numerical environment. Do not upgrade dependencies as part of this batch.

## 2. Common training and preprocessing contract

### Outer CV and weights

Use the existing five outer folds.

For each held-out fold, use only the other four folds to calculate:

$$
N_+(t),\quad N_-(t),\quad W_t=N_+(t)N_-(t),\quad
Z=\sum_{t:W_t>0}W_t.
$$

For each eligible training row:

$$
\omega_{it}=\frac{W_t}{2ZN_{y_{it}}(t)}.
$$

Exclude training ages with \(W_t=0\). Predict all validation rows, including any ages excluded from that fold’s training.

Use all eligible training rows. No row subsampling for logistic fitting, inverse-series-length weighting, additional class weights, or validation-data calibration.

Verify the existing weight identities with absolute tolerance `1e-12`.

### Standardization notation

For any scalar training column \(u\), define:

$$
\mu_u=\frac{\sum\omega u}{\sum\omega},\qquad
\sigma_u^2=\frac{\sum\omega(u-\mu_u)^2}{\sum\omega}.
$$

Define:

$$
S(u)=\frac{u-\mu_u}{\sigma_u}.
$$

If training variance is exactly zero, define the standardized column as zero for both training and inference.

All means, standard deviations, and interaction-centering constants are computed from eligible **training rows only** and persisted with the fold model.

Let:

$$
z_R=S(R),\quad z_I=S(I),\quad z_P=S(P),\quad z_D=S(D).
$$

These four base transformations must match E08’s training-fold transformations. Adding columns must not change the training-row eligibility mask, weights, or base normalization.

### Logistic model contract: E09–E12

For the exact input vector specified for each experiment:

$$
f=b+\beta^\top X,\qquad p=\operatorname{sigmoid}(f).
$$

Fit:

$$
\sum\omega\,[\log(1+\exp(f))-yf]
+\frac{0.01}{2}\sum_j\beta_j^2.
$$

Freeze:

```text
precision                float64
initialization           all coefficients and intercept zero
intercept penalty        none
coefficient penalty      0.01/2 * sum(beta²), including extension coefficients
optimizer                existing deterministic unbounded L-BFGS-B procedure
maxiter                  1000
gtol                     1e-6
ftol                     0
maxcor                   10
maxls                    20
maxfun                   20001
native numerical threads 1
```

Require optimizer success and independently recomputed final gradient infinity norm `<=1e-6`.

A failed fold makes that experiment invalid. Do not change settings or retry with another optimizer.

No feature selection, coefficient-sign constraints, warm starts, regularization search, or validation-guided stopping.

## 3. Frozen experiment definitions

### E09 — Historical predictability gating

**Parent:** E08.

**Hypothesis:** The useful relative weighting of raw and innovation evidence depends on how much the historical AR predictor improves over location-only prediction.

**Historical context: exactly one variable.**

Reuse `ARChannel.mse_ratio` from the existing implementation.

Its definition must remain:

* Fit the existing frozen AR(5) on the first `floor(0.7*H)` historical observations.
* Use the remaining historical segment \(C\).
* Let \(z_C\) be observations normalized using the AR training segment’s mean and sample standard deviation.
* Let \(e_C\) be the corresponding uncentered one-step AR innovations in those same normalized units.

Then:

$$
r=\frac{\operatorname{mean}(e_C^2)}
{\operatorname{mean}(z_C^2)}.
$$

Do not replace the denominator with variance around the mean of \(C\), and do not calculate this ratio using standardized innovation-energy inputs.

Define fractional historical predictive improvement:

$$
q=\operatorname{clip}(1-r,0,1).
$$

Rules:

* Finite \(r\ge0\): use the formula.
* Existing `mse_ratio=None` caused by a zero denominator: set \(q=0\), record an unsupported-context count, and add no missingness feature.
* Negative or nonfinite numeric ratios, or AR fitting failures: invalidate the experiment rather than silently substituting a different AR calculation.
* Freeze \(q\) before online inference.

Using training-row weights, calculate:

$$
\bar q=\frac{\sum\omega q}{\sum\omega}.
$$

**Exact inputs, in order:**

$$
X_{09}=[
z_R,z_I,z_P,z_D,\;
S((q-\bar q)z_R),\;
S((q-\bar q)z_I)
].
$$

**Input count:** 6.

No standalone \(q\), no interactions with `P` or `D`, and no other historical context.

**Model/objective:** Common frozen logistic contract.

**Update complexity and memory:** Baseline E08 feature update plus constant-time interaction calculations and a six-input score. One additional per-series context scalar; no new rolling buffers. Shared model/preprocessing memory is \(O(6)\).

**Leakage audit:** The AR ratio is history-only. Its cross-series centering and both product-column standardizations use training rows only. Online observations never update \(q\).

**Expected failure mode:** Historical predictive skill may not determine the best post-break detector weighting. The scalar context may also be noisy or insufficiently expressive.

**Primary comparison:** E09 versus E08.

**Secondary diagnostics:** Common diagnostics plus:

* Interaction coefficients by fold.
* Context distribution and unsupported count.
* E09 − E08 within the existing historical-ratio subsets:

  * \(r<0.9\);
  * \(0.9\le r\le1\);
  * \(r>1\).
* Report each subset’s series count and metric pair weight. Subset AUCs do not decompose pooled AUC.

**Advancement:** Common G3 gate below.

### E10 — Innovation energy by scale, expressed as contrasts

**Parent:** E08.

**Hypothesis:** The average innovation-energy block hides useful distinctions between recent and persistent evidence.

**Historical context:** None beyond the unchanged Stage-2 feature initialization.

Use the existing bounded innovation-energy scores:

```text
I32
I128
I256
Icumulative
```

Their windows, partial-window behavior, historical innovation reference, and evidence formulas remain unchanged.

Retain the authoritative existing `I` value. Add:

$$
C_1=\frac{I_{32}-I_{128}}{\sqrt2},
$$

$$
C_2=\frac{I_{32}+I_{128}-2I_{256}}{\sqrt6},
$$

$$
C_3=\frac{I_{32}+I_{128}+I_{256}-3I_{\mathrm{cumulative}}}{\sqrt{12}}.
$$

**Exact inputs:**

$$
X_{10}=[z_R,z_I,z_P,z_D,S(C_1),S(C_2),S(C_3)].
$$

**Input count:** 7.

The average plus these three contrasts spans the four scales. Do not also append the four original scale scores, which would introduce redundant inputs.

No additional interactions.

**Model/objective:** Common frozen logistic contract.

**Update complexity and memory:** Baseline feature update plus three constant-size contrasts and a seven-input score. The scale states already exist; no additional rolling buffers. Shared model/preprocessing memory is \(O(7)\).

**Leakage audit:** Extract the existing causal scale statistics. Standardize contrasts using training rows only. No longer windows, future-prefix operations, or horizon-dependent normalization.

**Expected failure mode:** Highly correlated scales may provide little incremental ranking information; the additional coefficients may fit noise.

**Primary comparison:** E10 versus E08.

**Secondary diagnostics:** Common diagnostics plus contrast coefficients and implied coefficients on the four original scales, by fold. These are descriptive, not grounds for dropping a scale.

**Advancement:** Common G3 gate.

### E11 — Separate rank location and rank energy

**Parent:** E08.

**Hypothesis:** Equal averaging of location and energy rank evidence loses information that one additional coefficient can recover.

**Historical context:** None beyond the unchanged frozen historical CDF.

Reuse existing `P_location` and `P_energy`. Retain authoritative `P`; do not reconstruct or overwrite it.

Define:

$$
K=\frac{P_{\mathrm{energy}}-P_{\mathrm{location}}}{\sqrt2}.
$$

**Exact inputs:**

$$
X_{11}=[z_R,z_I,z_P,z_D,S(K)].
$$

**Input count:** 5.

No additional interactions or rank statistics.

**Model/objective:** Common frozen logistic contract.

**Update complexity and memory:** Baseline update plus one subtraction, standardization, and a five-input score. No new historical or rolling state. Shared model/preprocessing memory is \(O(5)\).

**Leakage audit:** Reuse the unchanged empirical-CDF and rank calculations. Only the contrast normalization is fitted, using training rows.

**Expected failure mode:** Rank energy may largely duplicate innovation energy; location evidence may contain conditional information despite its weak standalone AUC.

**Primary comparison:** E11 versus E08.

**Secondary diagnostics:** Common diagnostics plus the contrast coefficient and implied location/energy coefficients by fold.

**Advancement:** Common G3 gate.

### E12 — Causal maturity-dependent evidence weighting

**Parent:** E08.

**Hypothesis:** The relative importance of innovation, rank, and dependence evidence changes with the amount of online evidence observed.

**Historical context:** None.

Let:

```text
t = time_online + 1
```

Thus the first observation has age 1.

Define:

$$
a(t)=\log(1+t),\qquad
\bar a=\frac{\sum\omega a(t)}{\sum\omega}.
$$

**Exact inputs:**

$$
X_{12}=[
z_R,z_I,z_P,z_D,\;
S((a-\bar a)z_I),\;
S((a-\bar a)z_P),\;
S((a-\bar a)z_D)
].
$$

**Input count:** 7.

Exclude:

* Standalone age.
* An age-dependent intercept.
* `R × age`.
* Piecewise age bins or learned knots.
* Relative age, final horizon, or interactions with historical context.

**Model/objective:** Common frozen logistic contract.

**Update complexity and memory:** Baseline update, one logarithm, three products, and a seven-input score. Reuse the existing causal age counter; no prefix storage. Shared model/preprocessing memory is \(O(7)\).

**Leakage audit:** Age is the count already observed. Centering and product normalization use training rows only. Truncating the future stream must not change any existing prediction.

**Expected failure mode:** The observed late-age improvement may reflect stronger inputs rather than changing optimal coefficients. An age-dependent combination can also damage early ranking.

**Primary comparison:** E12 versus E08.

**Secondary diagnostics:** Common diagnostics plus implied block slopes at the fixed ages `32, 128, 256, 512`, by fold. Report within-fold improvements explicitly to distinguish them from changes in cross-fold score calibration.

**Advancement:** Common G3 gate.

### E13 — One constrained nonlinear combiner

**Parent:** E08.

**Hypothesis:** A shallow nonlinear mapping of the established four blocks improves ranking beyond a globally linear logit.

**Exact inputs:**

$$
X_{13}=[z_R,z_I,z_P,z_D].
$$

**Input count:** 4.

No historical context, age, scale contrasts, rank contrast, or manually constructed interactions. Tree splits may express interactions among these four existing inputs.

Use the pinned **scikit-learn 1.8.0** **`HistGradientBoostingClassifier`** with:

```text
loss                   log_loss
learning_rate          0.05
max_iter               100
max_depth              2
max_leaf_nodes         4
min_samples_leaf       10000
max_bins               64
l2_regularization      0.01
max_features           1.0
categorical_features   None
monotonic_cst          None
interaction_cst        None
class_weight           None
early_stopping         False
validation_fraction    None
warm_start             False
random_state           20260921
scoring                loss
n_iter_no_change       10       # inactive because early stopping is disabled
tol                    1e-7     # inactive because early stopping is disabled
verbose                0
native threads         1
```

Explicitly disabling early stopping matters: the library’s default `"auto"` enables it for large training sets. The documented L2 parameter regularizes leaf estimation, not the same coefficient geometry used by E08.

**Training weights/objective:**

* Pass the same normalized \(\omega\), summing to 1, directly as `sample_weight`.
* **Do not rescale weights to sum to the row count.** That would change the effect of the fixed leaf regularization and Hessian thresholds.
* Fit binary log loss using the pinned native histogram-boosting procedure.
* Use training-only weighted standardization of the four inputs.
* Preserve native training-only binning, including its seeded bin-threshold subsampling. Do not implement alternative weighted binning.
* No validation arguments may be passed to `fit`.
* Use the native weighted-prior initialization.
* Supply float64 inputs and weights; retain the pinned implementation’s native internal precision, including its float32 gradient/Hessian arrays. Do not claim this model is entirely float64.

Require `n_iter_ == 100`, finite predictions, and deterministic repeatability. A model containing unsplit trees is valid but may perform poorly. Do not apply the logistic gradient-convergence requirement to boosting.

**Score:** Positive-class `predict_proba`.

**Update complexity and memory:** Baseline feature update plus at most 100 depth-2 tree traversals. At most 700 tree nodes, shared across series, plus preprocessing/bin metadata. No additional per-series learned state.

**Leakage audit:** All bin thresholds, splits, leaves, initialization, and standardization are fitted from outer-training rows only.

**Expected failure mode:** Quantization may create harmful ties; shallow trees may underfit smooth relationships or overfit correlated trajectories. A leaf containing 10,000 rows does not imply 10,000 independent series.

**Primary comparison:** E13 versus E08.

**Secondary diagnostics:** Common diagnostics plus realized tree depths, leaf counts, score-tie rates, training weighted log loss, and model size.

**Advancement:** Common G3 gate.

## 4. Common G3 advancement gate

For every candidate, compare to the **same frozen E08 OOF predictions**.

Require:

1. Pooled TS-AUC delta `>= +0.002`.
2. Strictly positive TS-AUC deltas in at least four of five folds.
3. Combined ages `>=129` delta `>=0`.
4. Combined ages `<=64` delta `>=−0.005`.
5. Retain the historical early guard: combined `<=64` versus E01 squared `>=−0.005`.
6. Pass the uncertainty rule below.

Combined `>=257` remains a required diagnostic, not a new promotion condition.

### Bootstrap and five-comparison adjustment

For candidates passing conditions 1–5:

* Reuse the unchanged whole-series, six-stratum paired bootstrap.
* Exactly `999` replicates.
* Seed `20260920`.
* Same multiplicities for candidate and E08.
* Recompute pair weights in each replicate.
* Keep model predictions fixed; no refitting.
* Report the usual conditional 95% interval using quantiles `0.025` and `0.975`.

For advancement, additionally require:

$$
Q_{0.01}(\Delta^{*})>0.
$$

Use the existing linear-interpolation quantile convention.

This is a one-sided 99% lower percentile bound, assigning `0.05/5 = 0.01` to each of the five preregistered primary comparisons. Keep the family size at five even if an experiment is invalid or does not reach bootstrap.

This is an **approximate conditional multiplicity safeguard**, not an exact finite-sample guarantee. It does not adjust for all previous Stage-1/2 research or establish performance on unseen competition data.

Do not change the replicate count or seed after seeing an interval near zero.

### Decision rules

* Valid but failing any gate: **no advancement**.
* Invalid experiment: report the engineering/numerical reason separately.
* No qualifiers: retain E08.
* Multiple qualifiers: designate the highest pooled-TS-AUC qualifier as the next local reference; exact ties favor fewer model inputs, then lower experiment ID.
* Do not claim that the chosen qualifier is significantly better than other qualifying candidates without another preregistered comparison.
* Do not construct a combined winner model.

## 5. Feature extraction and implementation boundaries

Add Stage-3 modules and a new `research_specs/stage3_v1.json`. Do not edit `stage2_v1.json` or change E08’s numerical implementation.

Reuse existing numerical functions where compatible. In particular:

* Reuse the weighting formula and logistic objective.
* Add dimension-general Stage-3 preprocessing/model wrappers rather than changing E08’s four-input interface.
* Preserve the AR procedure and its historical split exactly.
* Reuse existing comparison/scoring/bootstrap calculations.

A single new **feature-export pass** through the unchanged causal Stage-2 state is allowed during implementation/execution to obtain missing innovation-scale components. It is not a new detector experiment.

The export may contain:

```text
R, I, P, D
I32, I128, I256, Icumulative
P_location, P_energy
```

Store the historical ratio/context separately by series. Evaluator keys and labels may exist in artifacts but must never enter model-input selection.

Verify:

* Exported `R/I/P/D` match the frozen existing artifacts.
* The four innovation scores reproduce authoritative `I`.
* Rank subblocks correspond to the existing implementation; retain authoritative `P` despite any floating-point difference from recomputing an average.
* Historical context matches the existing `ARChannel` calculation.

Cache **raw causal quantities**, not globally standardized interactions. E09 and E12 transformations are fold-specific and must be constructed after the outer split.

Suggested new module boundaries:

```text
sbrt/
  stage3_features.py
  stage3_models.py
  stage3.py
  stage3_cli.py
research_specs/
  stage3_v1.json
tests/
  test_stage3_numerics.py
  test_stage3_pipeline.py
```

Final naming can follow repository conventions; mathematical definitions cannot change.

## 6. Required tests and execution discipline

Run the existing suite before editing; the current expected baseline is 55 passing tests.

Add tests for:

* Historical-ratio arithmetic, clipping, unsupported-context handling, and no online context mutation.
* Every exact input list and exclusion of prohibited fields.
* Training-only weights, base normalization, interaction centering, and product normalization.
* Held-out-label and held-out-feature changes not affecting that fold’s trained model.
* Contrast formulas and their zero-sum coefficients.
* Embedding E08 coefficients with zero extension coefficients reproducing E08 predictions/objective within `1e-14`.
* Constant context or constant maturity producing zero interaction columns.
* Correct first-observation age: `t=1`, hence `a=log(2)`.
* Causal/prefix/truncation invariance and exact resumed state.
* Deterministic training and save/load prediction equivalence.
* E13’s exact parameters, training-only bins, disabled early stopping, normalized weights, and 100 fitted iterations.
* Complete unique OOF coverage: 10,000 series, 5,036,517 rows.
* Immutable Stage-2 files and CV-v2 manifest.
* Bootstrap multiplicity identity and the new `0.01` quantile gate.

Freeze the Stage-3 configuration and source hash after final passing tests and before measured runs. Verify source stability afterward.

No model-setting changes after scores are observed. A necessary implementation correction must be distinguished from a modeling revision, audited, retested, and invalidate affected run artifacts; it must not be used to tune around performance.

## 7. Required outputs

For every experiment, emit:

* Exact parent, input list/order, formulas, resolved configuration and hashes.
* Pooled OOF TS-AUC and delta versus E08.
* Five fold scores and paired deltas.
* Eight age-bucket scores and deltas.
* Combined `<=64`, `>=129`, `>=257` scores and deltas.
* Descriptive pooled delta versus E04.
* Model-specific diagnostics listed above.
* Training seconds per fold and total.
* Initialization, feature-export, inference, scoring, bootstrap, and total wall times.
* Separately labeled cached-model throughput and any measured end-to-end streaming throughput.
* Per-series feature-state size and shared model/preprocessing size.
* Model-input count and any degeneracy/support counts.
* Logistic objective, gradient norm, iterations, and coefficients where applicable.
* Gate conditions, bootstrap intervals/quantiles, seed, and multiplicity hash.
* Source, configuration, dataset, fold, feature-cache, model, and OOF hashes.
* Explicit valid/invalid status and advancement decision.

All subgroup and coefficient diagnostics are descriptive. They cannot trigger new features, revised thresholds, or another fit within this batch.

Provide reproducible individual-experiment, full-batch, and comparison CLI commands. Completed artifacts must be verified before reuse and never silently overwritten.

## 8. Stopping rule

After E09–E13 and their required comparisons finish, **stop**.

Do not automatically run:

* A combined context-plus-scale-plus-age model.
* A tree using newly selected Stage-3 features.
* Pairwise ranking.
* New context variables, penalties, windows, AR variants, or rank statistics.
* Neural models or any hyperparameter search.

The batch’s purpose is to identify **which single limitation of E08 is worth addressing next**, not to maximize the score through accumulated adaptive changes.
