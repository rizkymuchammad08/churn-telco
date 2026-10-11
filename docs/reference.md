# Reference

Every command, file format, configuration key, output, and error code. For how to use them see the [User Guide](user-guide.md); for the reasoning behind the numbers see the [Methodology](methodology.md). This document does not copy configuration values.

## Contents

- [Commands](#commands)
- [Import paths](#import-paths)
- [Data format](#data-format)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Statuses, flags, and markers](#statuses-flags-and-markers)
- [Error codes](#error-codes)
- [Modules](#modules)
- [Tests and lint](#tests-and-lint)
- [Packaging and dependencies](#packaging-and-dependencies)
- [Extending the project](#extending-the-project)
- [Glossary](#glossary)

## Commands

### Make targets

| Target | Function |
|---|---|
| `make install` | `pip install -c constraints.txt -e ".[dev]"`: core packages, the `dashboard` extra, and the test tools, with the numerical stack at the versions in `constraints.txt` |
| `make freeze` | Rewrites `constraints.txt` from the current environment; run it only in a clean virtualenv |
| `make all` | `data`, `features`, `train`, `monitoring`, `ablation`, `roi`, `render` on the full data (`paths.raw`) |
| `make all-sample` | `data-sample`, `features-sample`, `train-sample`, `monitoring`, `ablation-sample`, `roi`, `render` on the sample (`paths.sample`); not a source of official report numbers |
| `make data` / `make data-sample` | Reads, cleans, and validates the data (`src.data.validate`) |
| `make features` / `make features-sample` | Builds features and labels, checks for leakage (`src.features.build`) |
| `make train` / `make train-sample` | Splits, trains, selects, calibrates, evaluates (`src.models.train`); also writes `reports/monitoring.json` |
| `make monitoring` | Checks that `reports/monitoring.json` covers exactly the saved model's features and was built from the same train rows (`src.models.monitoring --require-model`); part of `make all` and `make check-p4` |
| `make roi` | Computes business value from test-set predictions (`src.reporting.roi`) |
| `make render` | Renders the three documents (`src.reporting.render`) |
| `make ablation` | Diagnostic with and without `tenure` and `TotalCharges` (`src.models.ablation`); part of `make all`, before `render`, because `reports/ablation.json` is a number source for the documents |
| `make dashboard` | Opens the read-only dashboard via Streamlit |
| `make lint` | `ruff check` and `ruff format --check` |
| `make test` | `pytest` over all of `tests/` |
| `make clean` | Removes `artifacts`, pytest and ruff caches, and `__pycache__` |
| `make check-p1` to `make check-p6` | Per-part checks (see below) |

`make all-sample` and `make all` write to the same `reports/` and `docs/`. Run `make all` before reading or sharing a report.

**Per-part checks.** Each `check-pN` runs lint, `pytest -m pN`, the part's characteristic step, then `scripts/check_handoff.py handoff/pN.yaml`.

| Target | Part | Characteristic step |
|---|---|---|
| `check-p1` | Foundation (configuration, Makefile, skeleton) | dry run `make -n all` |
| `check-p2` | Data | `make data` |
| `check-p3` | Features and labels | `make features` |
| `check-p4` | Model | `make train`, then `make monitoring` |
| `check-p5` | ROI and recommendations | none |
| `check-p6` | Reporting | `make render`, then `scripts/scan_placeholders.py` over README, CHANGELOG, `docs`, and `reports` |

### Command line

```
python -m src.data.validate [--sample | --path FILE]
python -m src.data.sample [--check]
python -m src.data.dictionary [--check]
python -m src.data.fixtures [--check]
python -m src.features.build [--sample]
python -m src.features.spec [--check]
python -m src.models.train [--sample]
python -m src.models.tuning [--check | --write-lock]
python -m src.models.monitoring [--require-model | --write-lock]
python -m src.models.ablation [--sample]
python -m src.reporting.roi
python -m src.reporting.render
streamlit run src/reporting/dashboard.py
```

| Option | Meaning |
|---|---|
| `--sample` | Use `paths.sample` instead of `paths.raw` |
| `--path FILE` | (`validate`) Validate a specific CSV |
| `--check` | Compare the committed file with the generator's output and exit with status one if they differ; writes nothing |
| `--require-model` | (`monitoring`) Fail when `artifacts/model.joblib` does not exist. Without it the check runs against `params.yaml` alone when there is no saved model (as in CI) |
| `--write-lock` | (`monitoring`) Write the current rule hash into `monitoring.rule_lock` in `params.yaml`; changes only that line |

Modules are run from the project root. Exit status is zero on success. On failure, the module prints `CODE: message` to the screen, exits with status one, and writes nothing after the failed step. Each successful module prints a final marker (`VALID`, `FEATURES OK`, `MODEL OK`, `MONITORING OK`, `ROI OK`, `RENDER OK`). If `data_source` is synthetic, a banner is printed first.

There is no command-line option for the full-data path; it is read from `paths.raw`.

## Import paths

The code uses the `src` package at the project root and is run as a module (`python -m src.models.train`) from the project root, so no `PYTHONPATH` setup is needed. `pyproject.toml` registers `src*` as packages and sets `testpaths`, so `pytest` runs from the project root without extra configuration.

Configuration files (`params.yaml`, `assumptions.yaml`), `templates/`, `data/`, and `reports/` are read relative to the project root (`src.config.ROOT`). The only environment setting is `LOG_LEVEL` in `.env` (example in `.env.example`); it does not affect results.

## Data format

One CSV, one row per customer, with a header. Only schema columns are read; column order is free. Columns are read as text and then converted (see [Methodology](methodology.md#data-preparation)).

| Column | Type | Valid values | Notes |
|---|---|---|---|
| `customerID` | text | unique, non-null | Id; may be the index, not a feature |
| `gender` | categorical | `Female`, `Male` | |
| `SeniorCitizen` | integer | zero or one | |
| `Partner`, `Dependents` | categorical | `Yes`, `No` | |
| `tenure` | integer | non-negative | Leak-prone; excluded from features. Read for the domain flag |
| `PhoneService`, `PaperlessBilling` | categorical | `Yes`, `No` | |
| `MultipleLines` | categorical | `Yes`, `No`, `No phone service` | |
| `InternetService` | categorical | `DSL`, `Fiber optic`, `No` | |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | categorical | `Yes`, `No`, `No internet service` | |
| `Contract` | categorical | `Month-to-month`, `One year`, `Two year` | |
| `PaymentMethod` | categorical | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | |
| `MonthlyCharges` | float | non-negative | |
| `TotalCharges` | float | non-negative | Leak-prone. May be empty only when `tenure` is zero |
| `Churn` | label | `Yes`, `No` | Label; values from `schema.positive_label` and `schema.negative_label` |

Other rules:

- Extra columns outside the schema are ignored by features; a missing schema column is `E_COLUMNS_MISSING`.
- Categorical values outside the vocabulary are rejected when features are built (`E_FEATURE_CATEGORY`).
- Temporal mode adds one date column (`schema.snapshot_column`) that must be readable for every row.
- The full data dictionary, generated from code, is in `docs/data_dictionary.md`.

## Configuration

Two files, read strictly: a missing key, unknown key, or wrong type raises an error naming the key path. Optional keys are marked. Business figures and `k` are forbidden in `params.yaml`.

### `assumptions.yaml`

| Key | Type | Meaning and rules |
|---|---|---|
| `currency` | text | Currency, for labeling |
| `baseline_churn_rate` | float | Baseline churn rate; must be greater than `target_churn_rate`, both between zero and one |
| `target_churn_rate` | float | Target churn rate |
| `clv` | float | Value of a retained customer; positive |
| `cost_intervention` | float | Cost of one contact; positive |
| `save_rate` | float | Probability that a contacted churner is retained; greater than zero and at most one |
| `save_rate_source` | text | `assumed` or `measured`; determines the business-value status |
| `campaign_capacity` | integer | Contacts per period; positive, not above `population_per_period` |
| `population_per_period` | integer | Population size per period; positive |
| `operating_cost_per_year` | float, optional | Fixed campaign cost per year; non-negative; default zero |

Derived rule: `p* = cost_intervention / (save_rate * clv)` must lie between zero and one, or the configuration is rejected.

### `params.yaml`

| Key | Meaning and rules |
|---|---|
| `seed` | The single seed; non-negative integer |
| `data_source`, `data_source_allowed` | Active data source and the list of valid ones (`real_public_ibm_telco`, `synthetic`); the active source must be in the list |
| `paths.raw`, `paths.sample` | Full-data and sample files |
| `paths.artifacts_dir`, `paths.reports_dir` | Output folders |
| `paths.metrics`, `paths.roi`, `paths.ablation`, `paths.monitoring` | Output JSON files |
| `sample.n_rows` | Sample size; must not exceed the population per period and must be large enough for the number of folds |
| `schema.id_column`, `label_column`, `positive_label`, `negative_label` | Column roles; the positive and negative labels must differ |
| `schema.required_columns` | Required columns; must include the id, label, and nullable column |
| `schema.int_columns`, `float_columns` | Columns to convert; subsets of the required columns |
| `schema.nullable_when_tenure_zero` | Column that may be empty only when `tenure` is zero |
| `schema.snapshot_column` (optional) | Date column; temporal mode only; not the id or label |
| `temporal` | Boolean; must be consistent with `split.type` |
| `leakage.flagged_columns` | Leak-prone columns; subset of the required columns |
| `leakage.policy` | `exclude` or `include_flagged` |
| `leakage.max_abs_corr` | Correlation-detector threshold; between zero and one |
| `split.type` | `random_stratified` or `temporal` |
| `split.train_frac`, `calib_frac`, `test_frac` | Each between zero and one; must sum to one |
| `split.cv_folds` | Number of cross-validation folds; at least two |
| `split.window_days`, `split.purge_gap_days` (optional) | Temporal split only; `purge_gap_days` at least `window_days` |
| `models.primary_metric` | `pr_auc` or `lift_at_k` |
| `models.baselines`, `models.candidates` | Model lists drawn from `prior_rate`, `logistic_regression`, `gradient_boosting`; non-empty and disjoint |
| `models.class_weight` | `balanced` or `none` |
| `models.hyperparams` | Logistic regression: `C`, `max_iter`; gradient boosting: `n_estimators`, `learning_rate`, `max_depth` |
| `tuning.enabled` | Switch for hyperparameter tuning; when false the defaults in `models.hyperparams` are used |
| `tuning.method` | `grid` |
| `tuning.seed_offset` | Added to `seed` for the tuning folds; at least one, so they differ from the selection folds |
| `tuning.margin_se` | Standard errors of the paired gain a candidate must exceed to replace the default; between zero and ten |
| `tuning.rule_lock` | sha256 of the tuning rule; must match `python -m src.models.tuning` output when tuning is enabled |
| `tuning.grids` | Candidate values per model and hyperparameter (`C`, `max_iter`; `n_estimators`, `learning_rate`, `max_depth`); the default of each listed key must be in its list; simpler values first |
| `calibration.method` | `isotonic` or `sigmoid` |
| `evaluation.ece_bins` | Number of ECE bins; at least two |
| `evaluation.calibration_tolerance` | Limit on calibrated ECE; between zero and one |
| `evaluation.bootstrap_resamples` | Number of test-row resamples for the confidence intervals; at least one hundred |
| `evaluation.ci_level` | Coverage of the confidence intervals; strictly between zero and one |
| `segments.by` | Segment columns (not the id or label); `segments.min_n` is the reliability threshold |
| `domain.min_tenure_months` | Scoring-domain limit in months |
| `ablation.seeds` | List of seeds for ablation |
| `monitoring.rule_lock` | sha256 of the monitoring rule (the five keys below); must match, or the run stops with `E_MONITOR_LOCK`; rewrite it with `python -m src.models.monitoring --write-lock` |
| `monitoring.psi_warn`, `monitoring.psi_alert` | PSI thresholds written to every feature; `0 < psi_warn < psi_alert` |
| `monitoring.n_bins` | Quantile bins for a feature that is not monitored level by level; at least two |
| `monitoring.max_discrete_levels` | A feature with at most this many distinct train values is monitored level by level; at least two |
| `monitoring.psi_eps` | Floor on a bin share inside the PSI log ratio; strictly between zero and one |
| `monitoring.min_rows` | Smallest batch `assess` will score; at least `monitoring.n_bins` |

Business keys forbidden in `params.yaml`: `k`, `clv`, `cost_intervention`, `save_rate`, `campaign_capacity`, `population_per_period`, `target_churn_rate`, `baseline_churn_rate`. Cross-file check: the sample size must not exceed the population per period and must be large enough for the number of folds.

## Outputs

Generated, not committed (except where noted). Rendered documents are also written to `docs/`.

| File | Contents |
|---|---|
| `artifacts/features.csv`, `artifacts/labels.csv` | Feature matrix and labels |
| `artifacts/model.joblib` | `Bundle`: the selected model and its calibrator (pickle; load only your own files) |
| `artifacts/test_predictions.csv` | Test-set predictions: `label`, `p_raw`, `p_churn`; the only input to ROI |
| `reports/metrics.json` | All model figures |
| `reports/roi.json` | All business figures |
| `reports/ablation.json` | Ablation diagnostic; also the source of the excluded-columns section in rendered documents |
| `reports/monitoring.json` | Reference distribution and drift thresholds for every model feature; written by `make train`, never edited by hand; not a number source for the documents |
| `reports/final_report.md` | Final report, rendered |
| `docs/model_card.md` | Model card, rendered |
| `docs/recommendations.md` | Recommendations, rendered |
| `docs/feature_spec.md`, `docs/data_dictionary.md` | Generated from code; do not edit by hand |

### Layout of `reports/metrics.json`

Extra keys are allowed. Strict rendering fails if a required key is missing.

| Key | Contents |
|---|---|
| `data`, `data_source` | Input marker (`raw` or `sample`) and data source |
| `business_status` | `conditional` or `measured` |
| `seed`, `split`, `split_type` | Seed, rows per set, split type |
| `k`, `p_star` | Capacity on the test set and break-even threshold |
| `primary_metric`, `selected` | Selection metric and selected model |
| `cv` | Mean cross-validated primary metric per model |
| `test` | Per model: `n`, `n_pos`, `base_rate`, `k`, `pr_auc`, `roc_auc`, `precision_at_k`, `recall_at_k`, `lift_at_k`, `brier`, `ece`; for the selected model also `by_segment` |
| `calibration` | `method`, `ece_raw`, `ece_calibrated`, `tolerance`, `within_tolerance` |
| `hashes` | `input_data`, `train_data`, `preprocess`, `model`, `calibrator` |

Per segment: `n`, `n_pos`, `base_rate`, `mean_p`, `brier`, `roc_auc`, `reliable`.

### Layout of `reports/roi.json`

| Key | Contents |
|---|---|
| `data_source`, `business_status`, `save_rate_source`, `currency`, `years` | Markers and units |
| `n_test`, `k_test`, `p_star` | Test-set size, capacity, threshold |
| `inputs` | Copy of the `clv`, `cost_intervention`, `save_rate`, `operating_cost_per_year` used |
| `policies` | `no_contact`, `model_p_star`, `random_k`, each with `n_contacts`, `n_churners_contacted`, `net_value` |
| `incremental` | `model_vs_no_contact`, `model_vs_random_k` |
| `breakeven_save_rate` | `model_vs_no_contact`, `model_vs_random_k`; empty when undefined |

### Layout of `reports/monitoring.json`

Written by `make train` from the train rows only. Extra top-level keys are allowed (`data` records `raw` or `sample`).

| Key | Contents |
|---|---|
| `data_source` | Data source the file was built from |
| `rule_hash` | The monitoring rule hash that built the file; a difference from `monitoring.rule_lock` is `E_MONITOR_STALE` |
| `settings` | Copy of the `monitoring` block of `params.yaml` without the lock; a difference from the current block is `E_MONITOR_STALE` |
| `reference` | `rows` in the train set and `train_data_hash`, which must equal the saved model's train hash |
| `features` | One entry for every model feature, and no other |
| `content_hash` | sha256 of everything else in the file; any edit afterwards is `E_MONITOR_HASH`, even one that leaves the values valid |

Each entry in `features`:

| Key | Contents |
|---|---|
| `sources` | Raw columns the feature reads; equal to `features.build.feature_sources` |
| `kind` | `discrete` (monitored level by level, with a last bin for any level not seen in training) or `continuous` (bins at train quantiles) |
| `levels` or `edges` | The levels, or the strictly increasing bin edges, according to `kind` |
| `ref_props` | Share of train rows in each bin; sums to one |
| `psi_warn`, `psi_alert` | The thresholds for this feature; equal to the `settings` values |

`monitoring.assess(X, monitoring)` returns the PSI and a status (`ok`, `warn`, `alert`) per feature for a scoring-time feature matrix.

### Prediction output

`src.models.predict.predict` returns a table indexed by customer id, with the same row order and count as the input.

| Column | Contents |
|---|---|
| `p_raw` | Raw model score |
| `p_churn` | Calibrated probability, between zero and one |
| `rank` | One means highest risk; unique; ties in `p_churn` are broken by `p_raw`, then row order |
| `out_of_domain` | True when `tenure` is below `domain.min_tenure_months` |

The label column is not needed for prediction. Inputs outside the feature domain (unknown categories, negative numbers, missing values) are rejected.

## Statuses, flags, and markers

The project does not use verdicts or separate flag codes. Status is carried by the following keys.

| Marker | Location | Meaning |
|---|---|---|
| `business_status` | `metrics.json`, `roi.json` | `conditional` while `save_rate_source` equals `assumed`; `measured` once measured |
| "Conditional value" box | All rendered documents | Shown while the status is conditional |
| `calibration.within_tolerance` | `metrics.json` | Whether calibrated ECE is within `evaluation.calibration_tolerance`; being outside tolerance does not fail the run |
| `ci` | `metrics.json`, `test.<model>` | Bootstrap interval `lo`/`hi` per metric, plus `level`, `level_pct`, and `n_resamples`; rendered in brackets after each test-set value |
| `reliable` | `by_segment` | Segment has enough rows and contains both classes |
| `out_of_domain` | Prediction output, contact list | `tenure` below the domain limit; do not contact based on the score |
| Synthetic banner | CLI, reports, dashboard | Shown when `data_source` starts with `synthetic` |
| `source_hash` | Footer of rendered documents | Combined hash of the four number sources; detects stale documents |
| `hashes.input_data` | `metrics.json` | sha256 of the input data after cleaning |

## Error codes

Each layer has its own error class with `code` and `message` attributes. Tests assert the **code**, not the message. `validate` collects data codes without raising; loaders and pipeline steps raise. Configuration mistakes have no code of their own: they appear as value, key, or type errors naming the key path.

### Data

| Code | Triggered by |
|---|---|
| `E_FILE_MISSING` | No file at the data path |
| `E_CSV_PARSE` | CSV unreadable (rows of unequal length, empty, or corrupt encoding) |
| `E_NONNUMERIC` | A numeric column contains a value that cannot be converted |
| `E_DATA_SOURCE` | Unknown `data_source` |
| `E_EMPTY` | No rows |
| `E_COLUMNS_MISSING` | A schema column is missing |
| `E_NULL_ID` | Empty id |
| `E_DUP_ID` | Duplicate id |
| `E_LABEL_DOMAIN` | Label outside the two valid values |
| `E_NULL_CATEGORY` | A categorical column contains empty values |
| `E_NULL_NUMERIC` | A numeric column is empty where not allowed |
| `E_NEGATIVE_NUMERIC` | A numeric column is negative |
| `E_LEAKAGE_COLUMN_UNKNOWN` | A leak-prone column in the config is not in the data |
| `E_SNAPSHOT_INVALID` | Temporal mode: date empty or not a date |

### Features and leakage

| Code | Triggered by |
|---|---|
| `E_FEATURE_COLUMNS_MISSING` | A schema column is missing when building features |
| `E_FEATURE_ID` | Id not unique or empty |
| `E_FEATURE_CATEGORY` | Categorical value outside the vocabulary |
| `E_FEATURE_NONFINITE` | Feature matrix contains NaN or infinity |
| `E_LEAK_FORBIDDEN_COLUMN` | The id, label, or a leak-prone column became a feature column |
| `E_LEAK_CORRELATION` | Feature-label correlation exceeds `leakage.max_abs_corr` |

### Split

| Code | Triggered by |
|---|---|
| `E_SPLIT_COLUMNS` | Id or label column missing |
| `E_SPLIT_ID` | Id not unique or empty |
| `E_SPLIT_TOO_SMALL` | Too few rows or classes for three sets, or the purge gap consumed a set |
| `E_SPLIT_TYPE` | Unknown `split.type` |
| `E_SPLIT_NO_TIME_AXIS` | Temporal split without readable, complete dates |

### Training and calibration

| Code | Triggered by |
|---|---|
| `E_TRAIN_EMPTY` | Train data is empty |
| `E_TRAIN_COLUMNS` | Feature columns differ from those at training time |
| `E_TRAIN_ALIGN` | X and y indexes differ |
| `E_TRAIN_LABEL` | Labels are not binary |
| `E_TRAIN_ONE_CLASS` | Train labels contain only one class |
| `E_TRAIN_MODEL` | Unknown model name |
| `E_TRAIN_ASSUMPTIONS` | `lift_at_k` chosen without assumptions to derive `k` |
| `E_TUNE_LOCK` | `tuning.rule_lock` does not match the tuning rule in `params.yaml` |
| `E_CALIB_INPUT` | Invalid calibration scores or labels (shape, NaN, not binary) |
| `E_CALIB_ONE_CLASS` | Calibration labels contain only one class |
| `E_CALIB_METHOD` | Unknown calibration method |
| `E_CALIB_ANTI_MONOTONE` | Platt scaling with a negative slope |

### Monitoring

| Code | Triggered by |
|---|---|
| `E_MONITOR_FEATURES` | The features in `reports/monitoring.json` differ from the model's (some have no thresholds, or some no longer exist); the message lists both; also raised for an input to `assess` whose columns differ |
| `E_MONITOR_SOURCES` | A feature's source columns in the file differ from `features.build.feature_sources` |
| `E_MONITOR_LOCK` | `monitoring.rule_lock` does not match the monitoring rule in `params.yaml`: a threshold, bin count, epsilon, or minimum batch changed without `--write-lock`; also stops `make train` before anything is fitted |
| `E_MONITOR_STALE` | The file was built with another monitoring rule than `params.yaml` now has, or from other train rows than the saved model |
| `E_MONITOR_HASH` | `content_hash` is missing or does not match the file: it was edited after `make train` wrote it |
| `E_MONITOR_THRESHOLDS` | A feature's `psi_warn` or `psi_alert` is missing, not a finite number, not ordered, or different from the settings |
| `E_MONITOR_REFERENCE` | A feature's `kind`, levels or edges, or `ref_props` are malformed |
| `E_MONITOR_FILE` | The file is missing, unreadable, or lacks a required key; also `--require-model` without a saved model |
| `E_MONITOR_ROWS` | Empty train data, or an `assess` batch smaller than `settings.min_rows` |
| `E_MONITOR_INPUT` | An `assess` input contains a value that is not finite |

### Prediction and contact

| Code | Triggered by |
|---|---|
| `E_PREDICT_COLUMNS` | Columns missing from the prediction input |
| `E_PREDICT_DOMAIN` | Empty or negative values, or `SeniorCitizen` outside zero and one |
| `E_PREDICT_FEATURES` | Current features differ from the model's features |
| `E_TARGET_COLUMNS` | `p_churn` or `rank` column missing |
| `E_TARGET_CAPACITY` | `k` is not a non-negative integer |

### Business value

| Code | Triggered by |
|---|---|
| `E_ROI_INPUT` | `artifacts/test_predictions.csv` does not exist |
| `E_ROI_COLUMNS` | `label` or `p_churn` column missing |
| `E_ROI_LABELS` | Labels are not a binary vector, or the mask length differs |
| `E_ROI_K` | `k` out of range |
| `E_ROI_OUTCOME` | Negative contact or churner count, or churners exceed contacts |
| `E_ROI_YEARS` | `years` is not positive |

### Rendering and dashboard

| Code | Triggered by |
|---|---|
| `E_RENDER_INPUT` | Source file or template missing |
| `E_RENDER_MISSING_KEY` | Missing key or undefined name in a source or template |
| `E_RENDER_TEMPLATE` | Syntax or other error in a template |
| `E_RENDER_PLACEHOLDER` | Leftover work marker or template braces in the output |
| `E_DASH_MISSING` | Artifact or report needed by the dashboard does not exist yet |
| `E_DASH_STALE` | The artifact's `model_hash` differs from `reports/metrics.json` |

The collected list of data codes is in `ALL_CODES` in `src/data/validate.py`.

## Modules

| Module | Responsibility |
|---|---|
| `src/config.py` | Loads and validates configuration, `p*`, business status |
| `src/data/load.py` | Reads CSV, `clean`, `data_hash` |
| `src/data/validate.py` | Validation, synthetic banner, validation CLI |
| `src/data/sample.py` | Seeded sample from the real data |
| `src/data/synthetic.py` | Telco-schema synthetic generator (for tests) |
| `src/data/fixtures.py` | Broken-file fixtures |
| `src/data/dictionary.py` | Data dictionary from code |
| `src/features/build.py` | Feature matrix and labels |
| `src/features/leakage.py` | Forbidden columns and correlation detector |
| `src/features/spec.py` | Feature specification from code |
| `src/features/leaky_fixture.py` | Leaky fixture for the negative control |
| `src/models/split.py` | Train, calibration, test split |
| `src/models/train.py` | Preprocessing, class weights, cross-validation, selection, `Bundle`, `run` |
| `src/models/calibrate.py` | Isotonic and sigmoid calibrators |
| `src/models/evaluate.py` | Metrics and per-segment evaluation |
| `src/models/predict.py` | Customer scoring with `rank` and `out_of_domain` |
| `src/models/targeting.py` | `p*`, capacity, contact plan |
| `src/models/ablation.py` | Diagnostic with and without leak-prone columns |
| `src/models/monitoring.py` | Drift reference and thresholds per feature, the alignment checks, and `assess` |
| `src/reporting/roi.py` | Net value, incremental value, break-even `save_rate` |
| `src/reporting/render.py` | Strict rendering of the three documents (from four number sources) |
| `src/reporting/trace.py` | Tracing document numbers to their sources |
| `src/reporting/dashboard.py` | Read-only dashboard |
| `templates/*.md.j2` | Templates for the final report, model card, recommendations; no numeric literals |

Dependency direction: `predict` imports `train`; `train` imports `evaluate`, `calibrate`, `monitoring`, `split`, and `targeting`; `monitoring` imports only `config` and `features.build`; `roi` imports `targeting`; `config` imports no other project module.

## Tests and lint

Tests carry part markers in `pyproject.toml` (`p1` to `p6`). Each test file declares its part marker and is checked for consistency with the handoff manifest.

| Marker | Test files | Covers |
|---|---|---|
| `p1` | `test_assumptions_schema.py` | Configuration, schema, Makefile, skeleton, markers |
| `p2` | `test_data.py` | Loading, validation, broken fixtures, sample, dictionary, notebook |
| `p3` | `test_features.py`, `test_leakage.py` | Features, labels, specification, leakage |
| `p4` | `test_split.py`, `test_train.py`, `test_calibrate.py`, `test_evaluate.py`, `test_predict.py`, `test_targeting.py`, `test_temporal_mode.py`, `test_ablation.py`, `test_monitoring.py`, `test_p4_handoff.py` | The model and its whole chain |
| `p5` | `test_roi.py`, `test_render.py` | Business value and rendering |
| `p6` | `test_assumptions_traceability.py`, `test_handoff.py`, `test_dashboard_smoke.py` | Number tracing, manifest, dashboard |

Some tests enforce documentation rules:

- The README contains a "Problem" block that the notebook quotes in full; editing it changes the notebook output.
- Hand-written numbers in the README, CHANGELOG, and `docs/` that are not rendered output must trace to a source.
- There are no work markers or template braces in the README, `docs/`, `params.yaml`, or `assumptions.yaml`.
- The latest CHANGELOG version equals the version in `pyproject.toml`.
- Tests that run the `--sample` CLI work on a temporary project copy, so `make test` does not overwrite the full-data `reports/` and `artifacts/`.

Lint configuration is in `[tool.ruff]` in `pyproject.toml` (line length, target Python version, selected rules).

## Packaging and dependencies

- Core packages: `numpy`, `pandas`, `scikit-learn`, `pyyaml`, `jinja2`. The `dashboard` extra: `streamlit` and `openpyxl`, needed only for `make dashboard` and Excel upload or export. The `dev` extra: the dashboard extra plus `pytest`, `ruff`, `nbformat`, `nbclient`, and `ipykernel` (the last three run the notebook test; without them it is skipped).
- `constraints.txt` pins the numerical stack exactly (requires Python `>=3.11`). It is a constraints file, not a second copy of the dependency list: `pyproject.toml` stays the only place packages are declared, and a test checks that every core package is pinned exactly.
- `reports/metrics.json` records the Python and library versions that produced the results (`environment`), so a result can be matched to the environment that made it.
- The minimum Python version is set in `pyproject.toml` (`requires-python`).
- `joblib` is imported directly by the bundle-saving code but is not listed separately; it is installed as a dependency of `scikit-learn`.
- The project is run from the project root, not installed as an ordinary wheel: configuration files, `templates/`, and `data/` are read relative to the project root and are not packaged.
- Generated and not committed: `artifacts/` and packaging output.

## Extending the project

| Task | Where |
|---|---|
| Add a technical parameter | `params.yaml`, plus a spec in `PARAMS_SPEC` and value rules in `src/config.py` |
| Add a business input | `assumptions.yaml`, plus `ASSUMPTIONS_SPEC` and, if needed, rules in `src/config.py`; its key name goes into `BUSINESS_KEYS` so it is forbidden in `params.yaml` |
| Add a model | Register the name in `KNOWN_MODELS` (`src/config.py`), add a branch in `_estimator` (`src/models/train.py`), and its hyperparameters in the config spec |
| Add a feature | Add it to `_registry` in `src/features/build.py` with a vocabulary if categorical; the feature must be a function of its own row and pass the leakage detector; regenerate `docs/feature_spec.md`; run `make train` so `reports/monitoring.json` covers it (until then the monitoring check fails with `E_MONITOR_FEATURES`) |
| Add a leak-prone column | Add it to `leakage.flagged_columns`; the column must be in the required columns |
| Add a segment | Add the column to `segments.by`; not the id or label |
| Add a rendered document | New template in `templates/`, register it in `DOCS` in `src/reporting/render.py` (tests require all templates to be registered), and add context in `build_context` |
| Add an error code | Only deliberately: each code must have a dedicated fixture or test |
| Change report text | Templates in `templates/`; do not put numeric literals in them |
| Change recommendations | Edit `templates/recommendations.md.j2`; there is no automatic recommendation branch (out of scope by decision) |

Rules that are maintained: all randomness comes from `seed`; numbers live only in `assumptions.yaml`, `reports/metrics.json`, `reports/roi.json`, and `reports/ablation.json`; there are no silent defaults for missing keys; every configuration change is followed by `make all`.

## Glossary

| Term | Meaning |
|---|---|
| Churn | A customer with `Churn = Yes` in the file |
| Positive label | The `Churn` value counted as churn |
| p_raw | Raw model score |
| p_churn | Calibrated churn probability |
| p* | Break-even threshold: intervention cost divided by the product of `save_rate` and CLV |
| Capacity k | Maximum number of contacts, scaled to the batch size |
| PSI | Population stability index: how far a feature's distribution in a scoring batch has moved from the train rows; compared with `monitoring.psi_warn` and `monitoring.psi_alert` |
| Baseline | A simple model that must be beaten: `prior_rate`, `logistic_regression` |
| Candidate | The model expected to win: `gradient_boosting` |
| CLV | Value of a retained customer |
| save_rate | Probability that a contacted churner is retained |
| Conditional value | Business figures that hold only if the assumptions are true |
| PR-AUC | Area under the precision-recall curve (average precision) |
| ROC-AUC | Area under the ROC curve |
| Lift@k | Precision@k divided by the overall positive rate |
| Brier | Mean squared difference between probability and label |
| ECE | Average gap between probability and positive rate per bin, weighted by bin size |
| Calibration | Mapping raw scores to probabilities using the calibration set |
| Scoring domain | Customers with `tenure` above the minimum |
| Ablation | Diagnostic comparison with and without leak-prone columns |
| Purge gap | Minimum distance between sets in a temporal split |
| Handoff | Per-part manifest (`pN.yaml`) recording files, keys, APIs, and open issues |
