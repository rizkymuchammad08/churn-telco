# Changelog

The format follows Keep a Changelog. Performance figures and business values are not recorded here: their sources are `reports/metrics.json`, `reports/roi.json`, `reports/ablation.json`, and `assumptions.yaml`, and every document containing numbers is rendered from those four.

## [Unreleased]

### Added

- Drift monitoring thresholds for every model feature: `src/models/monitoring.py`, the `monitoring` block and `paths.monitoring` in `params.yaml`, and `reports/monitoring.json`, which `make train` writes from the train rows. Each feature has a reference distribution, its source columns, and its own `psi_warn` and `psi_alert`; `monitoring.assess` returns the PSI and status per feature for a batch. The file must cover exactly the model's features: `check_alignment` (against `params.yaml`, no model needed) and `check_model` (against the saved model and its train rows) fail with `E_MONITOR_FEATURES`, `E_MONITOR_SOURCES`, `E_MONITOR_STALE`, `E_MONITOR_THRESHOLDS`, `E_MONITOR_REFERENCE`, or `E_MONITOR_FILE` when a feature is added, removed, or redefined, or a monitoring setting changes, until `make train` writes the file again. `make monitoring` runs the check; `make all`, `make all-sample`, and `make check-p4` include it, and CI runs it against `params.yaml`. A changed threshold or bin rule is detected in the pattern of the tuning lock: `monitoring.rule_lock` in `params.yaml` (rewritten with `python -m src.models.monitoring --write-lock`, `E_MONITOR_LOCK` stops training before anything is fitted), the `rule_hash` recorded in the file (`E_MONITOR_STALE`), and a `content_hash` over the whole file (`E_MONITOR_HASH` for any edit after it was written). The file is reproducible: the same train rows and rule give identical bytes. `train.RunResult` has a new `monitoring` field. Tests: `tests/test_monitoring.py`. Model, metrics, and rendered documents are unchanged.
- Bootstrap confidence intervals on the test-set metrics: `evaluate.bootstrap_ci`, the `evaluation.bootstrap_resamples` and `evaluation.ci_level` keys in `params.yaml`, and a `ci` block per model in `reports/metrics.json`. The model tables in the final report and the model card now show each test-set value with its interval in brackets. Point estimates, model selection, and calibration are unchanged. ECE has no interval (a bootstrap of ECE is biased upward).

- Light hyperparameter tuning with a decision rule locked before the run: `src/models/tuning.py`, the `tuning` block in `params.yaml`, a `tuning` block in `reports/metrics.json`, and a tuning section in the final report and the model card. Each grid candidate (and the default) is scored by cross-validation on the train rows only, on folds that differ from the model-selection folds (`tuning.seed_offset`). The best candidate replaces the default in `models.hyperparams` only if its mean paired gain over the default across folds is positive and larger than `tuning.margin_se` standard errors of that gain; otherwise the default stays. The rule is hashed into `tuning.rule_lock`; a run stops with `E_TUNE_LOCK` when the rule and the lock disagree, and `python -m src.models.tuning --write-lock` rewrites the lock. The rule hash also enters `model_hash`. Setting `tuning.enabled` to false restores the fixed hyperparameters.
- `train.cv_fold_scores`, the per-fold scores behind `train.cv_score`.

### Changed

- `train.run` tunes before it selects a model, so every model score in `reports/metrics.json`, the ablation, and the rendered documents now reflects tuned hyperparameters. The ablation tunes inside each variant, on that variant's own train rows.
- The number tracer compares a shown figure with source values by magnitude, because the figure is captured without its sign. A negative change such as an ECE difference was previously looked up as a positive number and could be reported as unsourced.

### Known limitations

- Status: the code is nearly complete but not closed. The main blocker is the data: Telco has no date column, so the existing temporal mode (`schema.snapshot_column`) cannot be used on it.
- The decision on `tenure` and `TotalCharges` is still open. The `exclude` policy stays as the default; its cost is measured in `reports/ablation.json`. Revisit it once dated data is available. See the "Status and open decisions" section of the README.
- `save_rate` is still an assumption (`save_rate_source = assumed`), so business value stays conditional. It becomes `measured` only after a controlled trial, which is work outside the code.
- An uplift model is out of scope and remains separate work, as the documents state.
- Not yet implemented, and possible as later code work: running the drift comparison on real scoring batches with a history and alerts (the reference, thresholds, and `assess` exist), a wider or adaptive tuning search (the grids are deliberately small), and candidate models beyond logistic regression and gradient boosting.

## [0.2.0] - 2026-10-07

### Added

- Release hygiene: GitHub Actions CI (`.github/workflows/ci.yml`: lint, tests, re-render and placeholder scan, package build), `SECURITY.md`, and `.gitignore` entries for `build/`, `dist/`, and `.ipynb_checkpoints/`.
- Long-form documentation: `docs/user-guide.md` (running the pipeline and reading reports), `docs/methodology.md` (methods and formulas), and `docs/reference.md` (commands, configuration, formats, error codes). The README was restructured with a summary, business value levers, business value status, and how to read the report; the Problem section is unchanged because the notebook quotes it.
- Ablation with/without `tenure` and `TotalCharges`: `src/models/ablation.py`, the `make ablation` target, output `reports/ablation.json`, and the `ablation.seeds` and `paths.ablation` keys in `params.yaml`. Diagnostic only; the `leakage.policy` policy and the official model are unchanged.
- End-to-end temporal mode: optional key `schema.snapshot_column`; `make_split` reads dates from that column, `validate` rejects empty or corrupt dates (`E_SNAPSHOT_INVALID`), and `metrics.json` records `split_type`. The default remains `temporal: false` (Telco has no dates). Only the split changes; the label window and as-of-snapshot features do not.
- The sha256 of the input data (`hashes.input_data` in `reports/metrics.json`, `input_data_hash` in `reports/ablation.json`) now appears in the body of the final report, the model card, and the dashboard, no longer only in the footer and the calibrator.
- `LICENSE` (MIT), license, author, and readme metadata in `pyproject.toml` (the build now requires a newer setuptools that supports SPDX license expressions), and `DATA.md` with the origin, checksum, and licensing notes of the IBM Telco data. The README has a License section.

### Changed

- Minimum Python raised to `>=3.11` (`requires-python`, ruff `target-version` set to `py311`), matching `constraints.txt`, the user guide, and the versions CI tests. Support for `3.10` was declared but never tested.
- `tenure` and `TotalCharges` are no longer described as demonstrated leakage. The final report has a new section on excluded columns: it states that the exclusion is a manual policy (the correlation detector does not flag either column), that the reasoning is plausible but unverifiable without dates, and the measured cost of the policy, rendered from `reports/ablation.json` through the template. The model card and recommendations were reworded the same way.
- `reports/ablation.json` is now the fourth number source: it is hashed in every document footer, included in number tracing, and recorded with the detector limit, the number of splits, and the number of splits where the selected model changes. `make all` and `make all-sample` run the ablation before rendering, and rendering fails if the ablation was computed on different data than `metrics.json`.

- Dependencies: `streamlit` and `openpyxl` moved from the core dependencies to the `dashboard` extra; the `dev` extra now includes that extra plus `nbformat`, `nbclient`, and `ipykernel`, so the notebook execution test no longer skips silently. The dashboard tests skip, instead of failing at collection, when the extra is not installed.
- `requirements.txt` removed (it duplicated `pyproject.toml`). `make install` is now `pip install -c constraints.txt -e ".[dev]"`; `constraints.txt` pins the numerical stack to the versions that reproduce `reports/metrics.json`, and `make freeze` regenerates it.
- `reports/metrics.json` records an `environment` block with the Python, numpy, pandas, scikit-learn, scipy, and joblib versions.

### Fixed

- Dashboard exports (CSV and Excel) no longer let a `customerID` from an uploaded file run as a spreadsheet formula. Tests: `tests/test_dashboard_smoke.py`.

## [0.1.0] - 2026-10-01

### Added

- Foundation: `params.yaml`, `assumptions.yaml` with schema validation, a `Makefile` with targets `check-p1` through `check-p6`, a handoff manifest checker, and a placeholder scanner.
- Data: loading and cleaning, validation with corrupt-file fixtures, a seeded sample from the real data, and a data dictionary.
- Features: a non-temporal feature builder, a leakage detector (`tenure` and `TotalCharges` excluded), and a feature spec.
- Model: stratified random split, training with cross-validation, isotonic calibration, per-segment evaluation, prediction with an `out_of_domain` flag, and contact targeting based on the break-even threshold.
- ROI: business value labeled conditional while `save_rate_source` is still `assumed`, and strict rendering of the final report with a source-hash footer.
- Reporting: number tracing back to sources, a read-only dashboard, `docs/model_card.md`, `docs/recommendations.md` (including a "What is NOT recommended" section), and this file.

### Changed

- `make all` (and `make data`, `make features`, `make train`) now use the full data; the sample path was moved to `make all-sample` and the `*-sample` targets.
- Tests that run the `--sample` CLI now work on a temporary copy of the project, so `make test` no longer overwrites the full-data `reports/` and `artifacts/`.
- `make render` now renders three documents from three templates: the final report, the model card, and the recommendations.
- `make check-p6` now also scans for hand-typed numbers in the README, CHANGELOG, `docs/`, and `reports/`.

### Known limitations

- Business value stays conditional while `save_rate_source` has the status `assumed`; the calibration status and conditional values are listed in `reports/final_report.md`.
- Decision: recommendations stay template-rendered; the automatic recommendation branch (`recommend.py`) is out of scope and `open_issues` in `handoff/p5.yaml` is closed.
