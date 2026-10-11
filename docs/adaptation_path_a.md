# Adapting P0 to a Cross-Sectional Dataset (Path A)

The dataset has one row per customer, with no dates and no event log, so P0 rules that depend on time cannot be applied as written. This document records every deviation.

| P0 rule | Status in path A | Notes |
|---|---|---|
| Label from a future window `(snapshot, snapshot + window_days]` | Replaced | `label = (Churn == positive_label)`. There is no window. |
| Snapshots whose window passes `observation_end` are excluded | Not applicable | There is no `observation_end`. |
| Customers who already churned at the snapshot are excluded | Not applicable | Churn time is unknown. Population = all rows. |
| Features as of the snapshot | Replaced | Leakage risk is managed through `leakage.flagged_columns` (`tenure`, `TotalCharges`), with the default policy `exclude`. |
| Temporal split with purge gap | Replaced | `split.type = random_stratified` with `temporal: false`, because Telco has no dates. Temporal mode is available and tested end to end when the data has a date column: set `schema.snapshot_column` (and add it to `schema.required_columns`), set `temporal: true`, `split.type: temporal`, `window_days`, and `purge_gap_days`. `config.py` enforces `purge_gap_days >= window_days`; `make_split` reads dates from that column, and `validate` rejects empty or corrupt dates (`E_SNAPSHOT_INVALID`). This mode changes only the split; the window label and as-of-snapshot features do not follow (see the label and feature rows above). |
| `data.synthetic.generate` | Not used | The data is real. The sample is produced by a seeded generator from the real data. |
| `snapshot_date` in outputs | Constant or omitted | Fixed in the label section. |
| Test set touched once; calibrator only on calibration set; decisions use `p*`; baseline required | Fully applies | Unchanged. |
| Numbers only from yaml/json; one owner per file | Fully applies | Business figures and `k` are forbidden in `params.yaml` (tested). |

## Mapping of P1–P3 test specifications (event-log variant) to path A

The original test specifications were written for event-log data. For Telco, each item is either translated or declared not applicable.

| Specification item | Status in path A | Equivalent |
|---|---|---|
| `target_churn_rate < baseline_churn_rate` within (0, 1); `p*` within (0, 1) | Applies | New keys in `assumptions.yaml`; `config.breakeven_threshold`; tested in `test_assumptions_schema.py` |
| `metric.primary` in {lift_at_k, pr_auc}; `candidates` do not overlap `baselines`; `k` not in params | Applies | Fixed key `models.primary_metric`; `logistic_regression` is only a baseline |
| `split.type` random/group/temporal; columns `group`, `event_date`, `cutoff`, `gap_days` | Not applicable | One row per customer, `random_stratified` split. Temporal splitting is still enforced via `window_days` and `purge_gap_days` |
| `label.definition`, `window_days`, `active_lookback_days`, `churn_date` | Not applicable | `label = (Churn == positive_label)`, no window |
| Fixtures `dup_event_id`, `null_customer`, `future_event`, `forbidden_col` | Partial | `duplicate_id` and `null_id` exist. Events after `observation_end` do not exist (no time). Forbidden columns are tested in `test_leakage.py` |
| `clean` is idempotent; `data_hash` is deterministic and independent of row order | Applies | `data.load.clean`, `data.load.data_hash` |
| Synthetic sample from a logistic hazard; base rate near baseline; segments; outliers | Applies | `data.synthetic.generate`; intercept calibrated to `baseline_churn_rate`; recency and frequency are absent because there are no events |
| Window label `(snapshot, snapshot + window_days]`; churn before snapshot; window past `observation_end` | Not applicable | Tested only for: binary label, no NaN, unique ids, deterministic, matches a hand count |
| As-of-snapshot features, RFM, events after snapshot | Not applicable | Features are purely row attributes; `tenure` and `TotalCharges` are dropped by the `exclude` policy |
| Replacing all labels does not change features; no forbidden columns; correlation detector catches `sample_leaky.csv` | Applies | `tests/test_leakage.py`, `tests/fixtures/sample_leaky.csv` (label × constant + small noise) |

## Mapping of P4 test specifications (event-log variant) to path A

| Specification item | Status in path A | Equivalent |
|---|---|---|
| `test_split`: group = no `customer_id` on both sides | Applies as id | One row per customer, so there is no separate `group` type. Tested: ids are unique, the three sides are disjoint, duplicate or empty ids are rejected (`E_SPLIT_ID`) |
| `test_split`: temporal, train < calib < test by `snapshot_date`, gap >= `label.window_days` | Code branch exists, unused for Telco data | The dataset has no dates. `make_split(..., snapshot=...)` enforces the ordering and a gap >= `split.purge_gap_days` (validated to be >= `split.window_days`), and is tested on synthetic dated data. Without `snapshot`: `E_SPLIT_NO_TIME_AXIS` |
| `test_split`: sizes match params; deterministic with a seed | Applies | Per-side sizes follow `split.*_frac` on both full data and the sample (actual figures are in `reports/metrics.json`); results also do not depend on row order (sorted by id) |
| `test_train`: resampling and class weights do not touch calib or test | Applies to class weights | Resampling is not used (only `models.class_weight`). Tested with a spy: weights, preprocessing state, and every `fit` see only train rows, and never more rows than the train count |
| `test_train`: `model_hash` is the same for the same seed and differs when one param changes | Applies | Hash of model name, seed, `models`, `leakage`, `split.cv_folds`, and the train data hash. `calibration.method` and `paths` are deliberately excluded (not part of the base model); tested as a negative control |
| `test_calibrate`: ECE after <= before, and within `evaluation.calibration_tolerance` | Applies on synthetic data | On sample data (small test set), calibrated ECE can exceed the tolerance because of noise, whereas on full data the test set is large enough to enforce it. `metrics.json` reports `within_tolerance` as-is and does not fail the pipeline |
| `test_evaluate`: lift@k = 1 for a constant score; deterministic tie handling | Applies | Tie-aware top-k: a group of equal scores cut off by the k boundary contributes its expected value. Results do not depend on row order |
| `test_evaluate`: `by_segment` sums to the total; `segments.min_n` | Applies | `segments.by` = Contract, InternetService, SeniorCitizen. A segment is unreliable if n < `min_n` or one label class is absent |
| `test_predict`: flagged as outside `domain.min_tenure_days` | Unit changed | The key becomes `domain.min_tenure_months`, because Telco `tenure` is in months, not days. The `tenure` column is still read, only for this flag (it is not a feature) |
| `test_predict`: label not required | Applies | The label column is not read by the features; `predict` adds a placeholder label column so that it passes the `build_features` schema check |
| `test_targeting`: capacity k | Applies | `k` is not in params; `capacity_for(n, assumptions)` scales `campaign_capacity` to the batch size (n = `population_per_period` gives exactly `campaign_capacity`) |

## Interpretation note

`tenure` and `TotalCharges` accumulate over a customer's lifetime and plausibly stop at the point of churn, so a model that uses them may measure "how long the customer stayed until the data was captured" rather than "who will churn". Without dates this cannot be proven, and the correlation detector does not flag these columns, so the default policy `exclude` is a policy decision, not a demonstrated leakage. The comparison with and without these columns is run by `python -m src.models.ablation` (`make ablation`), written to `reports/ablation.json`, and rendered into the final report as the measured cost of the policy. With dated snapshots, the policy can be tested properly (are the values known at prediction time?) and should be revisited.
