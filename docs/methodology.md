# Methodology

How the numbers are produced and why each choice was made. Formulas are written with a word for every constant; no value is copied from `params.yaml` or `assumptions.yaml`. Key names point to where the value lives.

## Contents

- [Design guardrails](#design-guardrails)
- [Problem framing](#problem-framing)
- [Data preparation](#data-preparation)
- [Features](#features)
- [Leakage defense](#leakage-defense)
- [Data split and cross-validation](#data-split-and-cross-validation)
- [Models](#models)
- [Model selection](#model-selection)
- [Metrics](#metrics)
- [Calibration](#calibration)
- [Business layer: the break-even threshold](#business-layer-the-break-even-threshold)
- [Net value and policy comparison](#net-value-and-policy-comparison)
- [Conditional value](#conditional-value)
- [Ablation](#ablation)
- [Drift monitoring](#drift-monitoring)
- [Report integrity and reproducibility](#report-integrity-and-reproducibility)
- [Temporal mode](#temporal-mode)
- [Assumptions to keep in mind](#assumptions-to-keep-in-mind)

## Design guardrails

Six rules shape all of the code. When something looks odd, one of these is usually the reason.

- **One adapter, public real data as the default.** The input is one row per customer in the Telco schema. `data_source` in `params.yaml` is `real_public_ibm_telco` or `synthetic`; if synthetic, every output carries a banner. A missing file is a hard error.
- **No time axis, so no temporal claims.** The default split is random and stratified, and `temporal` is false. Temporal mode exists, but only for data that carries a date column.
- **The model may lose.** Two baselines (`prior_rate`, `logistic_regression`) compete with the candidate (`gradient_boosting`) under cross-validation. Whichever wins is used, whatever its type.
- **Columns that accumulate until churn is observed are excluded.** `tenure` and `TotalCharges` are flagged in `leakage.flagged_columns` with policy `exclude`.
- **Business figures are always conditional while `save_rate` is an assumption.** No business-value figure appears without that label, and the test set is never extrapolated to the population.
- **Numbers live in four places.** `assumptions.yaml` for business inputs, `reports/metrics.json` and `reports/roi.json` for outputs, and `reports/ablation.json` for the with/without `tenure` and `TotalCharges` diagnostic. Templates contain no numeric literals and are rendered in strict mode; every number in a document, including the README, CHANGELOG, and hand-written `docs/`, must trace to one of those sources, and tests enforce it.

## Problem framing

- **Unit:** one customer, one row, no dates.
- **Label:** one if `Churn` equals the positive label, zero if it equals the negative label; the values are in `schema.positive_label` and `schema.negative_label`.
- **At-risk population:** all rows, because churn status before the data was collected is not observed.
- **Prediction window:** undefined. The model answers "what kind of customer has churn status in the file", not "who will churn within N days".
- **Decision:** contact or do not contact, per customer, subject to the campaign capacity limit.

Consequently, the score is a risk ranking over the file's population. For new customers, the score is treated as a risk estimate, not a time forecast. Deviations from the original time-based rules are recorded one by one in [Path A adaptation](adaptation_path_a.md).

## Data preparation

- **Cleaning** (`data.load.clean`) is idempotent. Strings are trimmed; numeric columns in `schema.int_columns` and `schema.float_columns` are converted, empty strings become missing values, and integer columns stay integer as long as there are no missing values. Values that cannot be converted raise `E_NONNUMERIC`. Cleaning does not change or drop rows; that is the validator's job.
- **Validation** (`data.validate.validate`) collects all error codes without raising. It checks: the data source is known, the table is not empty, all schema columns exist, ids are non-null and unique, the label has only the two valid values, categorical columns are non-null, numeric columns are non-null and non-negative, the leak-prone columns named in the config actually exist, and (temporal mode) the date column is readable.
- **`TotalCharges` may be empty only when `tenure` is zero.** The value is filled with zero when features are built, and only if the column is used; under the default policy this column is excluded.
- **Data hash** (`data.load.data_hash`) is a sha256 over canonical content: rows sorted, fixed float format. The hash therefore does not depend on row order or on how numbers are written.
- **Sample** (`data.sample.make_sample`) is a random subsample stratified on the label, with its seed from `seed`. `data/sample.csv` must equal the generator's output (`python -m src.data.sample --check`).

Broken-file fixtures in `tests/fixtures/broken/` ensure that each data failure mode produces the expected error code.

## Features

Features are pure functions of the columns on that customer's own row. The label is never read when building features, and other rows do not affect a row's features.

- **Binary:** `is_male`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `MultipleLines`, and six add-ons (only `Yes` maps to one).
- **One-hot** for `InternetService`, `Contract`, and `PaymentMethod`, with one reference category each (`No`, `Month-to-month`, `Electronic check`).
- **Numeric:** `MonthlyCharges`; plus `tenure` and `TotalCharges` when the leakage policy allows.
- **Derived:** `n_addons` (number of active add-ons), `has_internet`, and `is_autopay`.
- **Fixed vocabulary.** Categorical values outside the list are rejected with `E_FEATURE_CATEGORY`, not silently encoded.
- **Fixed column order**, independent of the data's contents. The feature list and source columns are generated from code into `docs/feature_spec.md`, not written by hand.
- **No features derived from the label or the id.** The id may serve as the index for alignment but must not be a feature column.

The feature matrix must be finite; NaN or infinity raises `E_FEATURE_NONFINITE`.

## Leakage defense

Leakage here differs from the time-based case: there is no "future" to shift in order to test. What exists are columns whose values already reflect the outcome.

- **Forbidden columns.** The id, the label, and (under policy `exclude`) `tenure` and `TotalCharges` must not become feature columns. `assert_no_forbidden` rejects any matrix containing them (`E_LEAK_FORBIDDEN_COLUMN`).
- **Reason for the policy.** `tenure` and `TotalCharges` accumulate over a customer's lifetime. For customers who churned, they plausibly stop at the point of churn, so a model using them may measure how long customers stayed rather than who will churn. This cannot be proven without dates, and the correlation detector does not flag these columns, so the exclusion is a policy decision with a measured cost (see [Ablation](#ablation)), not a demonstrated leakage.
- **Correlation detector.** Every feature's absolute correlation with the label is checked (point-biserial for a binary label); if any exceeds `leakage.max_abs_corr`, the run stops with `E_LEAK_CORRELATION`. The detector only catches gross leakage, meaning a feature that is nearly a function of the label. It is a guard, not proof of the absence of leakage.
- **Negative control.** `tests/fixtures/sample_leaky.csv` contains a column built as the label times a constant plus small noise; a test verifies that the detector catches it.
- **Policy `include_flagged`** exists only for diagnostics (see [Ablation](#ablation)). The official model always uses `exclude`.

## Data split and cross-validation

Default: `random_stratified`.

- Ids are sorted first, so the split does not depend on the input row order.
- Set sizes are computed from `split.test_frac` and `split.calib_frac` with round-half-up; the remainder is train. The three fractions must sum to one.
- The split is staged with stratification on the label: first the test set is separated, then the calibration set from the rest, using the seed from `seed`.
- Each label class must have enough rows for all three sets, otherwise `E_SPLIT_TOO_SMALL`.
- Ids must be unique and non-null (`E_SPLIT_ID`). Because there is one row per customer, a row that leaks means a customer that leaks: no customer appears in two sets.

**Cross-validation** is stratified k-fold within the train rows only, with the number of folds from `split.cv_folds`. It is used to select the model. The calibration and test sets are invisible to it.

**Touch rules:**

- Preprocessing (mean and standard deviation), class weights, and model selection see only train rows.
- The calibrator sees only raw scores and labels from the calibration set.
- The test set is used once, at the end, to compute metrics. No decision (model choice, calibrator) depends on it. Tests use a spy to verify that every `fit` sees only train rows.

## Models

| Model | Role | Description |
|---|---|---|
| `prior_rate` | Baseline | Always predicts the train positive rate. Sets the floor: a model that cannot beat it is useless. |
| `logistic_regression` | Baseline | Logistic regression on standardized features; regularization strength and iterations from `models.hyperparams`. |
| `gradient_boosting` | Candidate | Boosted trees; number of trees, learning rate, and depth from `models.hyperparams`. |

Features are standardized with the mean and standard deviation from train rows (a zero standard deviation is replaced by one). **Class imbalance** is handled through weights, not resampling: with `models.class_weight` set to `balanced`, a row's weight is the number of rows divided by twice the number of rows in its class, computed only from the train labels (or the training-fold labels inside cross-validation). With `none`, all weights are one. `prior_rate` uses no weights.

`prior_rate` is a scikit-learn `DummyClassifier` with the prior strategy; the other two are also from scikit-learn. The model seed comes from `seed`.

## Model selection

All models, baselines and candidate, are scored by the mean of the primary metric (`models.primary_metric`) across the cross-validation folds on train rows. The highest mean wins. On a tie, priority goes to the candidate first, then the baselines in config order. The hyperparameters used in this comparison come from the tuning step below.

### Hyperparameter tuning

Before selection, `src/models/tuning.py` compares each model's default hyperparameters (`models.hyperparams`) with a small grid (`tuning.grids`). Every candidate is scored by the primary metric under stratified cross-validation on the train rows only; the calibration and test rows are never used. The decision rule is fixed in advance:

1. The default is always one of the candidates and is the reference.
2. The best candidate is the one with the highest mean score; an exact tie goes to the earlier candidate in grid order, so list simpler or more regularized values first.
3. The best candidate replaces the default only if its mean paired gain over the default across folds is positive and larger than `tuning.margin_se` standard errors of that paired gain. Otherwise the default is kept.
4. Tuning folds are seeded with `seed + tuning.seed_offset`, so the scores that later select the model are not computed on the folds that picked the hyperparameters.

The rule is locked. `tuning.rule_lock` holds the sha256 of the grids, the defaults, the margin, the offset, the metric, the number of folds, and the class weighting. If the rule changes and the lock does not, the run stops with `E_TUNE_LOCK`; `python -m src.models.tuning --write-lock` rewrites it, and the change should be committed before training so that a rule altered after seeing results shows up in the history. The result of every candidate is recorded in the `tuning` block of `reports/metrics.json` and rendered in the final report and the model card. With `tuning.enabled` set to false the fixed defaults are used.

This is a light check, not a search for the best possible model: the grids are small, and a gain below the margin is treated as noise.

The primary metric must be "higher is better" (`pr_auc` or `lift_at_k`). For `lift_at_k`, `k` in each fold is derived from `assumptions.yaml`; without assumptions, `E_TRAIN_ASSUMPTIONS` is raised.

All models are trained and calibrated, and all are evaluated on the test set for reporting, but only the selected model goes into the saved `Bundle`, and only it is reported per segment and used for contact decisions. Test-set results never change the choice.

## Metrics

All metrics are computed from calibrated probabilities on the test set.

- **PR-AUC** is average precision. With a low positive rate it is more informative than ROC-AUC; a constant model scores the base rate, so the base rate is the floor.
- **ROC-AUC** as a complement.
- **Precision@k, recall@k, lift@k.** Lift is precision@k divided by the overall positive rate; a constant score gives a lift of exactly one. Tie handling at the `k` boundary is tie-aware and deterministic: if the `k` cutoff splits a group with the same score, that group contributes positives equal to its slot share times the group's positive rate. The result does not depend on row order.
- **Brier** is the mean squared difference between probability and label.
- **ECE** (expected calibration error): probabilities are divided into equal-width bins (`evaluation.ece_bins`) between zero and one; ECE is the sum of the absolute differences between each bin's mean probability and positive rate, weighted by the proportion of rows. Empty bins are skipped.

- **Bootstrap confidence intervals.** For the test-set metrics (PR-AUC, ROC-AUC, precision@k, recall@k, lift@k, Brier), `metrics.json` stores a percentile interval under `test.<model>.ci`. The test rows are resampled with replacement `evaluation.bootstrap_resamples` times, with the same `k`, and the interval is the central `evaluation.ci_level` share of the resampled values. The resampling is seeded from `seed` and depends only on the number of rows, so all models are compared on identical resamples. The interval measures test-set sampling noise only: it does not include variation from training, calibration, or change over time, and overlapping intervals between two models are a visual check, not a significance test. ECE has no interval because resampling biases it upward, so the interval would not contain the point estimate.
- **Per segment** (`segments.by`): count, positives, base rate, mean `p`, Brier, and ROC-AUC per level. A segment is **reliable** if its count is at least `segments.min_n` and both classes are present. Segment counts sum to the total.

`k` for rank-based metrics is not read from `params.yaml`; it is derived from the campaign capacity in `assumptions.yaml`, scaled to the size of the test set.

## Calibration

A model's raw score is not automatically a probability, especially with balanced class weights that deliberately shift scores upward. The calibrator maps raw scores to probabilities.

- **`isotonic`** (default): non-decreasing isotonic regression with output clipped between zero and one.
- **`sigmoid`:** Platt scaling on the logit of the raw score. Rejected if the slope is negative (`E_CALIB_ANTI_MONOTONE`): a score that runs against the label must not be silently repaired.
- The calibrator is fit **only** from raw scores and labels in the calibration set. Calibration labels with a single class are rejected (`E_CALIB_ONE_CLASS`).
- A monotone mapping preserves ranking. Ties that appear in calibrated probabilities are broken using the raw score, then row order, so `rank` is always unique and deterministic.
- Calibration is reported as raw ECE and calibrated ECE against `evaluation.calibration_tolerance`, with the key `within_tolerance`. Being outside tolerance does **not** fail the pipeline; it is reported as-is. On a small test set (for example the sample), ECE can exceed the tolerance because of noise, while a sufficiently large test set is expected to meet it.

Calibration improves the average probability per bin, not discrimination: PR-AUC and ROC-AUC do not improve because of it.

## Business layer: the break-even threshold

Contacting a customer costs a fixed `cost_intervention`. The payoff: if the customer would churn and is successfully retained, the company keeps `clv`. The probability of success is `save_rate`, and the probability the customer churns is the calibrated `p_churn`. The expected value of one contact:

```
value = p_churn * save_rate * clv - cost_intervention
```

A contact is worthwhile when its value is non-negative, so the break-even threshold is:

```
p* = cost_intervention / (save_rate * clv)
```

`p*` is not stored in configuration; it is computed from those three inputs (`config.breakeven_threshold`) and must lie between zero and one, or the configuration is rejected.

**Contact rule.** A customer is contacted if they (a) rank within capacity `k` and (b) have a calibrated probability of at least `p*`. Because rank is consistent with `p_churn`, the contact set is always a prefix of the ranking. Customers below `p*` are not contacted even if capacity remains.

**Capacity.** `campaign_capacity` is the number of contacts per `population_per_period` customers. For a batch of size n, capacity is that proportion times n, rounded half up and capped at n. For a batch equal to the population per period, the result is exactly `campaign_capacity`.

**Scoring domain.** Customers with `tenure` below `domain.min_tenure_months` are flagged `out_of_domain` in the prediction output. `tenure` is read only for this flag, not as a feature.

## Net value and policy comparison

For one policy that contacts some number of customers, of whom some number are churners, over some number of years:

```
net = years * (churners_contacted * save_rate * clv
               - contacts * cost_intervention
               - operating_cost_per_year)
```

`operating_cost_per_year` is optional and is subtracted once per year, independent of the number of contacts. The no-contact policy has value zero, and operating cost applies only when a campaign runs.

Three policies are compared on the test set:

| Policy | Description |
|---|---|
| `no_contact` | Contacts no one; value zero. |
| `model_p_star` | The contact rule above, using the selected model's predictions and the test set's true labels to count reachable churners. |
| `random_k` | Contacts `k` customers at random; reachable churners is the expected value: `k` times the test-set base rate. |

**Incremental value** is the difference in net value between two policies. **Break-even `save_rate`** is the `save_rate` that makes the incremental value exactly zero:

```
s* = extra_cost / (extra_churners_reached * clv)
```

It is empty when the first policy does not reach more churners.

Things to read carefully:

- `random_k` always contacts `k` customers, whereas `model_p_star` may contact fewer when many customers fall below `p*`. The two policies then do not cost the same. The table reports each policy's contact count.
- Reachable churners are counted from the test set's true labels, assuming every contacted churner is saved with probability `save_rate`. There is no per-customer intervention-effect model.
- All figures apply to the test set. Scaling them to the full population adds uncertainty that has not been quantified; the project has no confidence intervals for business value.

## Conditional value

`business_status` is `conditional` as long as `assumptions.yaml` states `save_rate_source` equal to `assumed`, and `measured` once it has been measured. While conditional:

- every rendered document opens with a "Conditional value" box;
- every business-value figure is labeled "conditional value";
- the dashboard shows a warning.

The status changes only by replacing `save_rate` and `save_rate_source` with the results of a controlled trial (an uncontacted holdout group) and re-running `make all`. Lowering `p*` or raising capacity to reach more churners means changing assumptions, and net value must be recomputed.

## Ablation

`python -m src.models.ablation` (`make ablation`) compares the model **without** `tenure` and `TotalCharges` (the official policy) against the model **with** both, on the seeds listed in `ablation.seeds`. On each seed, both variants use the same split, grids, tuning rule, and selection procedure (tuning runs inside each variant, on that variant's train rows); the only difference is `leakage.policy`. Per-seed results, means, and spread are written to `reports/ablation.json`, together with the absolute correlation of those columns with the label.

Ablation is diagnostic only. It changes neither the official policy nor the official model. It is also a numeric source for the rendered documents, so `make all` runs it before rendering, and rendering stops with `E_RENDER_STALE_ABLATION` if `ablation.json` was computed on different data than `metrics.json`. The score increase with those columns is what accumulation until churn is observed would produce, but it is also what legitimately known predictors would produce; without dates the two cannot be told apart, so the gain is reported as the cost of the policy, not as evidence of leakage. The splits are different random partitions of the same rows, so agreement across them is not a significance test.

## Drift monitoring

`reports/monitoring.json` (written by `make train`, code in `src/models/monitoring.py`) lets someone ask whether a batch of customers to be scored looks like the customers the model was trained on. It holds, for every model feature, a reference distribution taken from the **train rows only** and the thresholds that decide when a difference is worth a look.

- **Binning.** A feature with at most `monitoring.max_discrete_levels` distinct train values (every indicator feature and the add-on count) is monitored level by level, with one extra bin for any level that never occurred in training. Any other feature (the monthly charge) is cut at its train quantiles into at most `monitoring.n_bins` bins, with the outer bins open-ended so values beyond the train range are counted, not dropped. The choice depends on the data, not on a list of names, so a new feature is covered without editing the module.
- **Statistic.** The population stability index, summed over bins: the share in the batch minus the share in training, times the natural log of their ratio. A share is floored at `monitoring.psi_eps` so an empty bin stays finite. Zero means identical distributions.
- **Thresholds.** A feature is `warn` at `monitoring.psi_warn` or above and `alert` at `monitoring.psi_alert` or above. These are the customary rule-of-thumb cut-offs, not values fitted to this data; they say how far a distribution has moved, not whether the model got worse. Every feature carries its own copy of both thresholds, and they equal the configured values.
- **Small batches.** PSI on few rows is mostly sampling noise, so `assess` refuses a batch below `monitoring.min_rows`. A test draws repeated batches of that size from the training population and requires that none of them reaches `alert`.
- **Tied to the model's features.** `check_alignment` requires the file to cover exactly the features `features.build` produces for the current `params.yaml` (none missing, none extra), with the same source columns and the same settings; `check_model` requires the same features and train rows as the saved model. It is the counterpart of `E_PREDICT_FEATURES`: when the feature set changes, the file must be regenerated by `make train`, and until then the check fails with a code that lists the missing and extra features. `make train` runs both checks before it writes the file, `make monitoring` repeats them on the written file, and the tests run the first one on the committed file.

- **Reproducible.** Nothing in the file is random. The same train rows and the same rule give the same bins, shares, thresholds, and hashes, byte for byte, whatever the row order of the input; a different seed gives different train rows and so a different file. Tests compare two runs with the same seed, shuffled input, two clean projects, and the committed file against a rebuild from the raw data.
- **Locked and sealed, like the tuning rule.** `monitoring.rule_lock` in `params.yaml` is the sha256 of the rule (thresholds, bin counts, epsilon, minimum batch). Changing the rule without `python -m src.models.monitoring --write-lock` stops the run with `E_MONITOR_LOCK`, before anything is fitted, so a change is a visible decision committed before the file is rebuilt. The file records the `rule_hash` that built it (a difference is `E_MONITOR_STALE`) and a `content_hash` over everything else in it, so an edit to a bin edge, a share, or a threshold is `E_MONITOR_HASH` even when the edited values are still valid. As with the tuning lock, this makes a change deliberate and detectable; it cannot stop someone who rewrites the hashes too, and review of the commit is what catches that.

What it does not do: nothing schedules the comparison, keeps a history of batches, or alerts anyone, and a feature that is stable can still hide a change in how it relates to churn.

## Report integrity and reproducibility

- **Strict rendering.** Documents are rendered with Jinja in a mode that rejects undefined names (`E_RENDER_MISSING_KEY`). A key missing from a source stops the run; there are no silent defaults.
- **Zero placeholders.** Output is scanned; leftover work-in-progress markers or template braces raise `E_RENDER_PLACEHOLDER`. All documents are rendered first and written only if all succeed.
- **Templates without numeric literals.** The only exceptions are section numbers in headings, years, and number formats inside template tags.
- **Number tracing.** Every number in a rendered document, and in hand-written prose in the README, CHANGELOG, and `docs/`, must equal one of the numeric values in `assumptions.yaml`, `reports/metrics.json`, `reports/roi.json`, or `reports/ablation.json` at the displayed precision. Text inside code spans and code blocks is not counted as a numeric claim. `make check-p6` enforces this.
- **Source hash.** The footer of every document contains the sha256 of the four sources and their combined hash, so a stale document is detected.
- **Model hash.** `model_hash` is computed from the model name, seed, `models` block (the tuned hyperparameters), `leakage` block, number of folds, the tuning rule lock (when tuning is enabled), and train-data hash. The calibrator has its own hash from its contents. Changing one hyperparameter changes the hash; `calibration.method` and `paths` are deliberately excluded because they are not part of the base model.
- **Input data hash.** The sha256 of all rows after cleaning is recorded in `reports/metrics.json` (`hashes.input_data`) and shown in the final report, model card, and dashboard.
- **Determinism.** All randomness comes from `seed`. The split does not depend on row order. With the same seed and input, `metrics.json` is identical.
- **Trusted artifacts.** `artifacts/model.joblib` is a pickle; load only files you created yourself. The dashboard rejects an artifact whose `model_hash` differs from `metrics.json`.

## Temporal mode

If the data has a date column per row, set `schema.snapshot_column` (that column must be in `schema.required_columns`), `temporal: true`, `split.type: temporal`, plus `split.window_days` and `split.purge_gap_days`. The configuration enforces that `purge_gap_days` is at least `window_days` and that the window is positive.

The split is then time-ordered: rows are sorted by date (ties broken by id), cut into train, calibration, and test, and train and calibration rows whose distance to the next set is less than the purge gap are dropped. If the purge consumes an entire set, `E_SPLIT_TOO_SMALL` is raised. Without dates, `E_SPLIT_NO_TIME_AXIS`. The validator rejects empty or corrupt dates (`E_SNAPSHOT_INVALID`).

This mode **changes only the split**. The as-of-snapshot label and feature definitions do not change; a genuine label window requires event data.

## Assumptions to keep in mind

- **No window, no temporal validity.** Performance on future customers is untested; a random split can be more optimistic than a time-ordered one.
- **Churn is a status in the file.** Customers who churned before data collection are not excluded because their timing is unknown.
- **`save_rate` is an assumption and is the same for everyone.** There is no uplift model: the score estimates risk, not who can be saved.
- **CLV and intervention cost are fixed** per customer; there is no variation by segment or time.
- **Single test set.** There are no uncertainty intervals for metrics or business value; a small test set makes ECE and per-segment figures wobble.
- **`tenure` and `TotalCharges` are excluded by policy.** This lowers the score compared with a model that uses them; the price is measured in `reports/ablation.json` and shown in the final report. Whether the exclusion is necessary cannot be established without dated data.
- **No operational campaign constraints** beyond capacity: no channels, scheduling, contact fatigue, or effects on uncontacted customers.
- **Public real data does not represent any particular operator.** Results on this file demonstrate the method, not performance on your customers, until it is run on your data.
