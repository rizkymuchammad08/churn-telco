# User Guide

How to run the project, read its report, and decide what to do next. The reasoning behind every number is in the [Methodology](methodology.md); every config key, file format, and error code is in the [Reference](reference.md).

This guide contains no result figures and does not copy configuration values. Values live in `params.yaml` and `assumptions.yaml` (inputs) and in `reports/metrics.json`, `reports/roi.json`, and `reports/ablation.json` (outputs).

## Contents

- [What this project answers](#what-this-project-answers)
- [Before you start](#before-you-start)
- [Path A: try it on the sample](#path-a-try-it-on-the-sample)
- [Path B: run on the full data or your own data](#path-b-run-on-the-full-data-or-your-own-data)
- [What happens in one run](#what-happens-in-one-run)
- [Reading the report](#reading-the-report)
- [What to do after reading the report](#what-to-do-after-reading-the-report)
- [Adjusting inputs](#adjusting-inputs)
- [When a run stops](#when-a-run-stops)
- [Frequently asked questions](#frequently-asked-questions)

## What this project answers

One question: **who is worth contacting in a retention campaign, and is there evidence that a model-based contact policy beats contacting no one or contacting at random?**

The project estimates each customer's churn probability, calibrates it, and then applies a single decision rule: contact customers whose calibrated probability is above the break-even threshold `p*` and who rank within the campaign capacity `k`. Prediction quality is compared against simple baselines, and the contact policy is compared against two business baselines. All business value is **conditional** for as long as `save_rate` is an assumption.

The project does not issue a verdict such as "adopt" or "don't adopt". It gives you numbers, trust signals (calibration, segment reliability, business-value status), and a list of things you must not conclude. The decision stays with you.

## Before you start

| You need | Why |
|---|---|
| Python `>=3.11`; packages from `pyproject.toml` | The pipeline is plain Python; `make install` installs the core packages, the dashboard extra, and the test tools at the versions in `constraints.txt`. For the pipeline alone, `pip install -c constraints.txt -e .` is enough; add `".[dashboard]"` for the dashboard. |
| A customer data file | One row per customer with the Telco schema columns (full list in the [Reference](reference.md#data-format)). The public IBM Telco file is already in `data/raw/`. |
| Business assumptions you can defend | CLV, intervention cost, `save_rate`, capacity, and population determine `p*` and every business value. The values shipped with the project are illustrative. |

Terms to understand first:

| Term | Meaning |
|---|---|
| **Churn** | A customer with `Churn = Yes` in the file. There is no time window. |
| **p_churn** | The calibrated churn probability per customer. |
| **p\*** | Break-even threshold: the minimum probability for one contact to have non-negative net value. |
| **Capacity k** | Maximum number of contacts per period, scaled to the size of the batch being scored. |
| **Baselines** | `prior_rate` (base rate) and `logistic_regression`. |
| **Candidate** | `gradient_boosting`, the model expected to beat the baselines. |
| **save_rate** | The probability that a contacted churner is successfully retained. Currently an assumption. |
| **Conditional value** | Business figures that hold only if the assumptions in `assumptions.yaml` are true. |
| **Scoring domain** | Customers with `tenure` above the minimum; those below are flagged `out_of_domain`. |

## Path A: try it on the sample

Use this to see the whole flow once. The sample is a seeded subset of the real data, so its numbers are not the source of the official report.

```
make install
make all-sample
```

This target validates the sample, builds features, trains, computes business value, and renders the documents. Because the test set is small, the calibrated ECE can fall outside tolerance; `reports/metrics.json` reports it as-is, and this is not an error.

Caution: `make all-sample` writes to the same `reports/` and `docs/` as `make all`. Run `make all` again before reading or sharing the official report.

## Path B: run on the full data or your own data

- **Prepare the data.** One CSV, one row per customer, with the Telco schema columns. For the public IBM Telco data, the file is already in `data/raw/`.
- **Point to it.** Set `paths.raw` in `params.yaml`. There is no command-line option for the data path; the pipeline reads it from configuration.
- **Replace the business assumptions.** Edit `assumptions.yaml`: CLV, intervention cost, `save_rate` together with `save_rate_source`, campaign capacity, and population per period. Make sure `p*` stays between zero and one, or the configuration is rejected.
- **Check the leakage policy and segments.** Leak-prone columns are in `leakage.flagged_columns`; reported segments are in `segments.by`.
- **Run.**

  ```
  make all
  ```

  If the file is missing, the first step stops with `E_FILE_MISSING`. The pipeline never silently falls back to other data.
- **Read `reports/final_report.md`.** Start with the conditional-value box.

If your data has a date column, temporal mode is available; see [Path A adaptation](adaptation_path_a.md). That mode changes only how the data is split, not the label or feature definitions.

## What happens in one run

`make all` runs five steps in order. Each step stops at the first error and prints its error code.

| Step | Target | What it does | What it writes |
|---|---|---|---|
| Data | `make data` | Reads the CSV, cleans its shape, validates structure and integrity. | Screen output only, including the data hash. |
| Features | `make features` | Builds the feature matrix and labels, checks for forbidden columns and feature-label correlation. | `artifacts/features.csv`, `artifacts/labels.csv` |
| Train | `make train` | Splits the data, cross-validates all models on train rows, selects one, calibrates on the calibration set, evaluates once on the test set. | `artifacts/model.joblib`, `artifacts/test_predictions.csv`, `reports/metrics.json` |
| ROI | `make roi` | Applies the contact rule to test-set predictions and computes the business value of three policies. | `reports/roi.json` |
| Render | `make render` | Renders the final report, model card, and recommendations from the four number sources. | `reports/final_report.md`, `docs/model_card.md`, `docs/recommendations.md` |

Two other targets are not part of `make all`:

- `make ablation` runs the diagnostic with and without `tenure` and `TotalCharges` and writes `reports/ablation.json`.
- `make dashboard` opens the read-only dashboard.

With the same seed and input, `reports/metrics.json` is identical across runs.

## Reading the report

`reports/final_report.md` always follows the same order. The model card and recommendations are rendered from the same sources, so their numbers cannot disagree.

**Conditional-value box.** Shown at the top while `save_rate_source` is `assumed`. Read this before any other number: every business figure below it is a scenario.

**Data and split.** Train, calibration, and test sizes, the number of churners in the test set, and the base rate. The test set is touched once.

**Model comparison.** The primary metric under cross-validation and on the test set for all three models. The selected model is determined by cross-validation on the train data, not by the test set. If a baseline is selected, that is a legitimate result: a more complex model does not win automatically.

**Calibration.** ECE before and after calibration, against the tolerance. If it is outside tolerance, do not read the probabilities as literal chances; on a small test set this is often just noise.

**Per-segment performance.** Customer count, churners, base rate, and mean `p_churn` per segment, with a "Reliable" column. Unreliable segments are too small or contain only one class; their figures must not be used for decisions.

**Contact decision and business value.** The threshold `p*`, the capacity `k` on the test set, and three policies: `no_contact`, `model_p_star`, `random_k`. The model's incremental value versus no contact and versus random contact, and the break-even `save_rate`. All of it is conditional value.

**Limitations.** A fixed list; see also the README.

Companion documents:

- `docs/model_card.md`: purpose, data, selection, performance, calibration, segments, limitations and risks.
- `docs/recommendations.md`: what is recommended and, just as important, what is **not** recommended. Read the second part before quoting any number.
- Dashboard (`make dashboard`): `model_hash`, input data hash, contact list, and business-value summary. It refuses to display if the model artifact does not match `reports/metrics.json` (`E_DASH_STALE`).

Every rendered document has a `source_hash` footer combining the hashes of the four sources. If you change `assumptions.yaml` or the results and forget to re-render, the tests catch the stale document.

## What to do after reading the report

The project does not produce a single verdict, so use the signals below.

| What you see | What it means | What to do |
|---|---|---|
| A baseline is selected; the more complex model did not win | The candidate was not shown to be better under cross-validation. | Use the selected model as it is; do not force the candidate. Consider extra features or data before a second attempt. |
| Calibration outside tolerance | Probabilities cannot be read as literal chances; `p*` and contact counts are affected. | Run on the full data (`make all`), re-check on a larger test set, and do not quote business values yet. |
| Status `conditional` | `save_rate` has not been measured. | Measure it with a controlled trial that includes an uncontacted holdout group. Then update `save_rate` and `save_rate_source` and run `make all`. |
| Assumed `save_rate` below break-even | Under that assumption the model policy does not cover its costs. | Revisit CLV, intervention cost, or `save_rate` with real data before the campaign. |
| A segment marked unreliable | Its figures are unstable. | Do not use it for segment-level decisions; merge segments or wait for more data. |
| `out_of_domain` customers in a scored list | `tenure` is below the domain limit; the score is not reliable. | Exclude them from score-based contact lists. |

Whatever the outcome, read the "NOT recommended" section of `docs/recommendations.md`. If `data_source` is set to `synthetic`, a synthetic banner appears in the CLI, reports, and dashboard, and no figure supports claims about real customers.

## Adjusting inputs

Change only `params.yaml` and `assumptions.yaml`; the code holds no decision parameters. The full key table is in the [Reference](reference.md#configuration). Common reasons to edit:

| You want to | Change |
|---|---|
| Use your own economics | `clv`, `cost_intervention`, `save_rate`, `save_rate_source` in `assumptions.yaml` |
| Change campaign capacity or population size | `campaign_capacity`, `population_per_period` in `assumptions.yaml` |
| Include fixed campaign cost | `operating_cost_per_year` in `assumptions.yaml` (optional) |
| Change the model-selection metric | `models.primary_metric` (`pr_auc` or `lift_at_k`) |
| Tune the models | `tuning.grids`, `tuning.margin_se`, then `python -m src.models.tuning --write-lock`; defaults in `models.hyperparams`; `models.class_weight` |
| Change the calibration method | `calibration.method` (`isotonic` or `sigmoid`) |
| Tighten or loosen the calibration tolerance | `evaluation.calibration_tolerance` |
| Change the reported segments | `segments.by`, `segments.min_n` |
| Change the scoring-domain limit | `domain.min_tenure_months` |
| Change the drift thresholds or binning | `monitoring.psi_warn`, `monitoring.psi_alert`, `monitoring.n_bins`, `monitoring.max_discrete_levels`, `monitoring.min_rows`; then `python -m src.models.monitoring --write-lock` and `make train` (which rewrites `reports/monitoring.json`) |
| Reproduce or vary a run | `seed` |
| Use data with a date column | `temporal`, `split.type`, `split.window_days`, `split.purge_gap_days`, `schema.snapshot_column` |

Two rules:

- Business figures and `k` must not appear in `params.yaml`; the config loader rejects them.
- `p*` and `k` are not stored; both are derived from `assumptions.yaml`.

Whenever you change configuration, re-run `make all` so every document is recomputed.

## When a run stops

The module prints `CODE: message` to the screen and exits with status one. Nothing is written after the failed step. The most common ones:

| Code | Usual cause | Action |
|---|---|---|
| `E_FILE_MISSING` | No file at `paths.raw` (or `paths.sample`). | Put the file there or fix the path. |
| `E_COLUMNS_MISSING` | A schema column is missing. | Add the column named in the message. |
| `E_DUP_ID` | Duplicate `customerID`. | Remove or merge the duplicates. |
| `E_NULL_NUMERIC` | Empty numeric value, except `TotalCharges` when `tenure` is zero. | Fill the value or drop the row. |
| `E_LABEL_DOMAIN` | A `Churn` value other than the positive and negative labels. | Normalize the labels. |
| `E_LEAK_CORRELATION` | A feature is nearly a function of the label. | Treat it as a real defect; do not loosen the threshold to get past it. |
| `E_ROI_INPUT` | `artifacts/test_predictions.csv` does not exist yet. | Run `make train` first. |
| `E_DASH_STALE` | The model in `artifacts/` does not match `reports/metrics.json`. | Run `make train`, then `make all`. |
| `E_MONITOR_FEATURES` | `reports/monitoring.json` does not cover exactly the model's features: a feature was added, removed, or renamed (or `leakage` changed) since the file was written. | Run `make train`; it writes the file again. |
| `E_MONITOR_LOCK` | A `monitoring` threshold or bin setting in `params.yaml` was changed without rewriting its lock. | If the change is intended, run `python -m src.models.monitoring --write-lock`, commit it, then `make train`; otherwise undo the change. |
| `E_MONITOR_HASH` | `reports/monitoring.json` was edited by hand. | Do not edit it; change `params.yaml` and run `make train`. |
| `E_MONITOR_STALE` | A `monitoring` setting in `params.yaml` changed, or the model was retrained on other rows, without `make train` writing the file again. | Run `make train`. |

Mistakes in `params.yaml` or `assumptions.yaml` (missing key, unknown key, wrong type, out-of-range value) appear as configuration errors that name the key. The full list is in the [Reference](reference.md#error-codes).

## Frequently asked questions

**Why was logistic regression selected instead of gradient boosting?** Selection uses cross-validation on the train data. If a baseline wins or ties, it is used; this is legitimate and by design. Both models' test-set scores are still reported.

**May I quote the net value?** Only labeled "conditional value", only together with the `save_rate` assumption, and only as a scenario. Do not scale test-set results to the whole population: the extrapolation uncertainty has not been quantified.

**Why are `tenure` and `TotalCharges` not used, even though they raise the score?** It is a policy decision, not a demonstrated leakage: both accumulate over a customer's lifetime and plausibly stop at the point of churn, so a model using them may measure how long customers stayed, but the data has no dates to prove it and the correlation detector does not flag them. `make ablation` measures the cost of the policy and the final report shows it; change the policy only with evidence about when these values are measured.

**What is the difference between `random_k` and `model_p_star`?** `random_k` contacts as many customers as the capacity `k`, at random; `model_p_star` contacts those who rank within `k` **and** have a probability above `p*`. If the model contacts fewer than `k`, the two policies do not cost the same; read the contact counts in the table.

**Is `save_rate` the same for all customers?** Yes, it is an assumption. The score estimates churn risk, not who can be saved. An uplift model is separate work.

**Where is the evidence that there is no leakage from the label?** Features are computed only from columns on that customer's own row, the label is never read when building features, leak-prone columns are excluded, and a correlation detector runs on every run. The detector is a coarse guard, not proof; its negative control is in `tests/test_leakage.py`.

**How do I run the dashboard?** `make dashboard`. It reads existing artifacts and reports; it trains nothing.
