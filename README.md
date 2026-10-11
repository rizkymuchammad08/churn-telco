# Telco Churn Prediction (non-temporal variant)

A calibrated churn model for the public IBM Telco dataset, with contact decisions based on a break-even threshold. Business value is reported as conditional value for as long as `save_rate` is still an assumption.

Documentation: [User guide](docs/user-guide.md) (running the pipeline and reading reports) · [Methodology](docs/methodology.md) (why each number has the value it has) · [Reference](docs/reference.md) (configuration, formats, error codes, how to extend) · [Model card](docs/model_card.md) · [Recommendations](docs/recommendations.md) · [Path A adaptation](docs/adaptation_path_a.md).

> **In short.** For each customer, the pipeline estimates the probability of churn, calibrates it, and then decides who is worth contacting in a retention campaign: only customers whose probability is above the break-even threshold and who still fit within campaign capacity. The result is an actionable contact list and conditional business value, not just a score.

| | |
|---|---|
| **Decision supported** | Which customers to contact in a retention campaign in a given period. |
| **Who it is for** | Retention and marketing teams deciding whether the model is fit for use; analysts and developers who maintain the pipeline. |
| **Output per run** | Calibrated churn probability, the threshold `p*`, the contact list, and a final report with a model card and recommendations. |
| **Core rule** | The model must beat simple baselines (base rate and logistic regression) in cross-validation. The selected model is the one that wins, not the most complex one. |
| **Status** | Real public data (IBM Telco), but `save_rate` is still an assumption: all business value is conditional. |

```
raw data (one row per customer)
  -> validation + cleaning (corrupt-file fixtures, data sha256)
  -> non-temporal features (tenure and TotalCharges excluded)
  -> light hyperparameter tuning under a locked decision rule (train rows only)
  -> baselines vs candidates, selected via cross-validation on training data
  -> isotonic calibration (calibration set only)
  -> break-even threshold p* + campaign capacity k
  -> final report, model card, recommendations, dashboard (reports/, docs/)
```

**Where the business value comes from.** Churn without a subscription contract means losing revenue that would otherwise still arrive. Retention campaigns have a cost, so not every customer is worth contacting. This project computes the break-even point from cost inputs and states the result only as conditional value.

| Lever | How the project handles it | Where to read it |
|---|---|---|
| Well-targeted contacts | The threshold `p*` is derived from CLV, intervention cost, and `save_rate`; customers below `p*` are not contacted even when capacity remains. | `assumptions.yaml`, final report |
| No wasteful campaigns | The model policy is compared with no contact and with random contact at the same number of contacts. | contact decision section of the final report, `reports/roi.json` |
| No claims about unmeasured results | The break-even `save_rate` is reported; net value is labeled conditional until `save_rate_source` is no longer `assumed`. | final report, `reports/roi.json` |
| No overconfidence | Calibration is checked against a tolerance, small segments are flagged as unreliable, and customers outside the score domain are flagged. | final report, `reports/metrics.json` |

**How to read the business numbers.** Business numbers are *conditional values*: they hold only if `save_rate` and the cost inputs in `assumptions.yaml` (illustrative) are correct. The score estimates churn risk, not who can be saved; net value is a scenario, not a forecast, until `save_rate` is measured through a controlled trial.

## Problem

- **Domain and relationship.** A subscription telecom service. The relationship is mixed: month-to-month customers (effectively non-contract) and customers on one-year or two-year contracts.
- **Churn definition and window.** Churn is `Churn = Yes` in the file. The data has one row per customer and no dates, so the prediction window is undefined and the project uses a non-temporal variant (see `docs/adaptation_path_a.md`).
- **Population at risk.** All rows in the file, because churn status before the snapshot is not observed.
- **Decision that depends on the score.** Customers whose calibrated `p_churn` is above the break-even threshold `p*` and who rank within campaign capacity `k` are contacted. Capacity and population per period are in `assumptions.yaml`.
- **CLV, intervention cost, save_rate.** All come from `assumptions.yaml`. While `save_rate_source = assumed`, all business value is conditional.

## Results

This README contains no result numbers. Each run writes its results to `reports/` and `docs/`:

- `reports/final_report.md`: model comparison, calibration, per-segment performance, contact decision, conditional business value, and limitations. Rendered from `assumptions.yaml`, `reports/metrics.json`, `reports/roi.json`, and `reports/ablation.json`, with a source-hash footer so stale documents are detected.
- `docs/model_card.md`: the model card (purpose, data, selection, performance, calibration, segments, limitations and risks).
- `docs/recommendations.md`: recommendations, including a "What is NOT recommended" section.
- `reports/metrics.json`: all model numbers. `reports/roi.json`: all business numbers. `reports/ablation.json`: the with/without `tenure` and `TotalCharges` comparison; it feeds the section on excluded columns in the rendered documents, and does not change the official model.

Numbers live in only three places: `assumptions.yaml` (business inputs, assumptions), `params.yaml` (technical parameters), and outputs in `reports/`. Templates contain no numeric literals. The sha256 of the input data is recorded in `reports/metrics.json` (key `hashes.input_data`) and shown in the final report, model card, and dashboard. Hand-typed numbers in the README, `CHANGELOG.md`, or `docs/` fail `make check-p6`; this README deliberately copies no numbers.

### Business value status

| Status | Meaning | When |
|---|---|---|
| `conditional` | All business values are scenarios under assumptions. Do not quote them as forecasts. | `save_rate_source = assumed` (the current state) |
| `measured` | Business value rests on a `save_rate` from a controlled trial. | `save_rate` and `save_rate_source` have been replaced with measured results, then `make all` is run again |

This status is recorded as `business_status` in `reports/metrics.json` and appears as a "Conditional value" box in every document.

### Reading the report

- Start with the conditional-value box and the data source, then the selected model and why it was chosen (cross-validation, not the test set).
- Check performance against the baselines: is the model better than the base rate and logistic regression?
- Check calibration: is the post-calibration ECE within tolerance?
- Read per-segment performance only for segments flagged as reliable.
- Read the contact decision and business value keeping in mind that everything is conditional; compare the break-even `save_rate` with the assumption.
- Read the limitations and the "What is NOT recommended" section before quoting any number.

## Structure

| Path | Contents |
|---|---|
| `params.yaml`, `assumptions.yaml` | The only source of configuration and business numbers |
| `src/config.py` | Configuration loading and validation, break-even threshold |
| `src/data/` | Loading and cleaning (`load.py`), validation (`validate.py`), seeded sampling (`sample.py`), synthetic generator, fixtures, data dictionary |
| `src/features/` | Non-temporal feature builder (`build.py`), leakage detector (`leakage.py`), feature spec |
| `src/models/` | Splitting (`split.py`), training and selection (`train.py`), calibration (`calibrate.py`), evaluation and segments (`evaluate.py`), prediction with `out_of_domain` flag (`predict.py`), contact targeting (`targeting.py`), ablation, drift reference and thresholds per feature (`monitoring.py`) |
| `src/reporting/` | Business value (`roi.py`), strict rendering (`render.py`), number tracing (`trace.py`), read-only dashboard |
| `templates/` | The only place for placeholders used by strict rendering (one template per rendered document) |
| `data/raw/` | Raw data; `data/sample.csv` is produced by the seeded generator |
| `reports/` | `final_report.md`, `metrics.json`, `roi.json`, `ablation.json` (diagnostics), `monitoring.json` (drift reference and thresholds for every model feature) |
| `docs/` | `model_card.md` and `recommendations.md` (rendered), hand-written documents with no claim numbers, and the long-form documentation |
| `handoff/` | Per-part manifests (`pN.yaml`) |
| `tests/` | Per-part tests (markers `p1` to `p6`), including corrupt-file and leakage fixtures |
| `scripts/` | `check_handoff.py`, `scan_placeholders.py`, notebook builder |
| `notebooks/` | `01_data.ipynb`; the Problem block is quoted directly from this README |
| `CHANGELOG.md` | Change history per version; the latest version must match `pyproject.toml` (tested) |
| `LICENSE`, `DATA.md`, `SECURITY.md` | MIT license for the code; origin, checksum, and licensing notes for the data; how to report a vulnerability |
| `.github/workflows/` | CI: lint, tests, document scan, package build |
| `pyproject.toml`, `constraints.txt`, `Makefile` | Package and dependencies (core, `dashboard` and `dev` extras), exact versions of the numerical stack, and work targets |

Generated, not committed: `artifacts/` and packaging output.

## Running

```
make install
make all           # data, features, train, ROI, render reports (on full data)
make all-sample    # fast path on the sample for local iteration (not a source of report numbers)
make dashboard     # read-only dashboard: model_hash, contact list, banner
make ablation      # with/without tenure and TotalCharges -> reports/ablation.json (also run by make all)
make monitoring    # checks reports/monitoring.json covers exactly the model's features (also run by make all)
make check-p6      # lint, P6 tests, re-render, placeholder scan, manifest
```

Each part has its own check: `make check-p1` through `make check-p6`. Each check runs lint, the tests marked for that part, its characteristic step, and then the handoff manifest checker.

Steps can be called directly as modules from the project root:

```
python -m src.data.validate [--sample]
python -m src.features.build [--sample]
python -m src.models.train [--sample]
python -m src.reporting.roi
python -m src.reporting.render
python -m src.models.ablation
```

Modules are run from the project root, so `src` imports without any `PYTHONPATH` setup. `pytest` reads `testpaths` and markers from `pyproject.toml`.

### Changing business assumptions

Edit `assumptions.yaml` only (CLV, intervention cost, `save_rate`, `save_rate_source`, capacity, population), then run `make all` so that `p*` and all documents are recomputed. Business numbers and `k` are forbidden in `params.yaml` (tested). Once `save_rate` has been measured, replace its value and `save_rate_source`; the conditional status goes away only this way.

## Limitations

- The `tenure` and `TotalCharges` columns are excluded from the features by policy, not because leakage was demonstrated: the correlation detector does not flag them, and the data has no dates to show when they were measured. The cost of the policy is measured by `make ablation` (written to `reports/ablation.json`) and reported in the section on excluded columns of the final report; the ablation changes neither the policy nor the official model.
- Without a time axis there is no temporal validation: `temporal: false` for the Telco data. Temporal mode (time-ordered split with a purge gap) is available for data that has a date column via `schema.snapshot_column`; how to enable it is described in `docs/adaptation_path_a.md`. This mode changes only the split, not the label definition or the features.
- In `make all-sample`, the test set is small, so the calibrated ECE can fall outside tolerance; `metrics.json` reports this as it is. The official report comes from `make all` (full data).
- `save_rate` is an assumption and is the same for all customers; the score is not an intervention effect. An uplift model is separate work.
- Test-set results must not be multiplied up to the whole population; extrapolation uncertainty has not been quantified.
- Customers with `tenure` below `domain.min_tenure_months` are flagged `out_of_domain` and must not enter the contact list based on score.
- The split is stratified random, not temporal: performance on future customers is untested.
- Recommendations are rendered from a template (`templates/recommendations.md.j2`) by design; there is no automatic recommendation branch.

## Status and open decisions

The code side is nearly complete but not closed. The pipeline, tests, and documents run end to end; what remains depends on the data, not on the code.

- **No date column.** The Telco file has no dates, so the temporal mode that already exists (`schema.snapshot_column`) cannot be used on it, and performance on future customers stays untested.
- **Open decision: `tenure` and `TotalCharges`.** The current policy is `exclude`. It is a manual policy, not demonstrated leakage, and it has a measured cost: the ablation shows that the two columns raise PR-AUC (see `reports/ablation.json` and the section on excluded columns in `reports/final_report.md`). Without dates, neither keeping nor including them can be verified. The options are to keep `exclude` and report its cost, to include them with the risk stated, or to wait for dated data and test whether the values are known at prediction time. This is a decision for the project owner; none of the three has been applied beyond the current default.
- **`save_rate` is still an assumption.** `save_rate_source` stays `assumed`, so all business value is conditional. The status changes to `measured` only after a controlled trial supplies a measured `save_rate`. That is field work outside the code; once the result exists, replace `save_rate` and `save_rate_source` in `assumptions.yaml` and run `make all`.
- **Uplift model is separate work.** The score estimates churn risk, not who can be saved, and the documents already describe an uplift model as work outside this project. The controlled trial above would also produce the data such a model needs.
- **Next step that unblocks the data questions.** Obtain a snapshot or event date per customer (or a different dataset that has one), set `schema.snapshot_column`, enable temporal mode as described in `docs/adaptation_path_a.md`, and re-run `make all`.

### Code work that can still be added

These are optional extensions; none is needed to close the open decisions above.

- **Uncertainty intervals.** Bootstrap intervals on the test metrics. Today the reports give point values (and the ablation gives spread across seeds), without an interval for the official model.
- **Drift monitoring in operation.** `reports/monitoring.json` holds a reference distribution and thresholds for every model feature, and `monitoring.assess` scores a batch against it; a check fails when the file and the model's features disagree. Nothing yet runs the comparison on real scoring batches, keeps a history, or alerts anyone, and the thresholds are conventional values, not tuned on this data.
- **More candidate models.** The candidates are limited to logistic regression (baseline) and gradient boosting; other model families can be added to `models.candidates`.

## License

The source code, templates, tests, and documentation in this repository are released under the MIT License (see `LICENSE`). The license does not extend to the data: the IBM Telco dataset in `data/raw/` and the files derived from it keep their own terms, and no explicit license for them was found in IBM's published material. The source, checksum, and list of derived files are in `DATA.md`; check them before redistributing.

Data attribution: the IBM Telco Customer Churn sample dataset, published by IBM and available from `https://github.com/IBM/telco-customer-churn-on-icp4d`.
