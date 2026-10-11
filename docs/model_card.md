# Model Card: Telco Churn Prediction

Model `logistic_regression` with `isotonic` calibration. All figures are rendered from `assumptions.yaml`, `reports/metrics.json`, `reports/roi.json`, and `reports/ablation.json`.

> **Conditional value.** save_rate (0.25) has status `assumed`. The business-value figures on this card hold only if that assumption is true.

## 1. Purpose and intended use

Rank customers by calibrated churn probability to choose whom to contact in a retention campaign. Decision rule: contact customers whose probability is above p* = 0.1333, up to the campaign capacity. Capacity is 700 contacts per period from a population of 7043 customers.

Out of scope: predicting churn within a specific time window, estimating the per-customer effect of an intervention, and scoring customers outside the scoring domain (see Limitations and risks).

## 2. Data

- Source: `real_public_ibm_telco`, file `data/raw`. One row per customer, no dates, so there is no temporal validation.
- Input data sha256: `deae3674baa798c67ebea55e8ed8b59fde59757f7d4d0412f8336abc377aca87` (canonical content after cleaning; recompute with `src.data.load.data_hash`).
- Stratified random split: 4225 train, 1409 calibration, 1409 test rows. The test set contains 374 churners (base rate 0.265).
- The `tenure` and `TotalCharges` columns are excluded from the features by policy, not because leakage was demonstrated. Using both would raise PR-AUC by 0.054 on average (diagnostic; see the final report).

## 3. Model and selection

Primary metric: pr_auc, chosen via cross-validation, not on the test set. The calibrator is trained only on the calibration set; the test set is touched once.

| Model | PR-AUC (CV) | PR-AUC (test) | ROC-AUC (test) | Lift@140 (test) | Brier (test) |
|---|---|---|---|---|---|
| gradient_boosting | 0.613 | 0.578 [0.530, 0.631] | 0.824 [0.801, 0.846] | 2.48 [2.19, 2.80] | 0.145 [0.135, 0.155] |
| prior_rate | 0.265 | 0.265 [0.241, 0.287] | 0.500 [0.500, 0.500] | 1.00 [1.00, 1.00] | 0.195 [0.184, 0.205] |
| logistic_regression | 0.622 | 0.571 [0.522, 0.622] | 0.822 [0.799, 0.843] | 2.58 [2.31, 2.85] | 0.146 [0.136, 0.156] |

Test-set columns show the point estimate and, in brackets, the 95% bootstrap interval (1000 resamples of the test rows). The intervals capture test-set sampling noise only, not variation from training or from changes over time; the CV column is a mean without an interval.

### 3.1 Hyperparameters

Hyperparameters were set by a light grid search under a decision rule locked before the run (rule hash `3438986f4849f68d2d686fab96d7e60cde4d13a99bf52f2f2bc8d2b8c00bea61`), using 5-fold cross-validation on the train rows only. A candidate replaces the default only if its paired gain over the default exceeds 1.0 standard error.

- `gradient_boosting`: replaced the default `learning_rate=0.1, max_depth=3` with `learning_rate=0.1, max_depth=2` (paired gain 0.0087, required more than 0.0026).
- `logistic_regression`: replaced the default `C=1.0` with `C=0.1` (paired gain 0.0006, required more than 0.0004).

## 4. Performance on the test set

| Metric | logistic_regression | prior_rate (baseline) |
|---|---|---|
| PR-AUC | 0.571 [0.522, 0.622] | 0.265 [0.241, 0.287] |
| ROC-AUC | 0.822 [0.799, 0.843] | 0.500 [0.500, 0.500] |
| Precision@140 | 0.684 [0.619, 0.749] | 0.265 [0.241, 0.287] |
| Recall@140 | 0.256 [0.229, 0.283] | 0.099 [0.099, 0.099] |
| Lift@140 | 2.58 [2.31, 2.85] | 1.00 [1.00, 1.00] |
| Brier | 0.146 [0.136, 0.156] | 0.195 [0.184, 0.205] |
| ECE | 0.020 | 0.000 |

Brackets show the 95% bootstrap interval, as in the model table above. ECE has no interval: a bootstrap of ECE is biased upward, so it would not bracket the point estimate.

## 5. Calibration

ECE dropped from 0.150 to 0.020; tolerance is 0.05, so the result is within tolerance.

## 6. Performance by segment

| Segment | n | Churners | Base rate | Mean p | Reliable |
|---|---|---|---|---|---|
| Contract: Month-to-month | 780 | 329 | 0.422 | 0.412 | yes |
| Contract: One year | 295 | 36 | 0.122 | 0.126 | yes |
| Contract: Two year | 334 | 9 | 0.027 | 0.033 | yes |
| InternetService: DSL | 487 | 91 | 0.187 | 0.203 | yes |
| InternetService: Fiber optic | 603 | 260 | 0.431 | 0.404 | yes |
| InternetService: No | 319 | 23 | 0.072 | 0.084 | yes |
| SeniorCitizen: not senior | 1175 | 279 | 0.237 | 0.239 | yes |
| SeniorCitizen: senior | 234 | 95 | 0.406 | 0.378 | yes |

Segments marked as not reliable are too small or contain only one class; do not use them for decisions.

## 7. Limitations and risks

- Evaluation uses a test set of 1409 rows; the intervals reflect test-set sampling noise only (not training variability or change over time), and differences between models have not been formally tested.
- There is no time axis: performance in future periods is not observed.
- Customers with tenure below the `domain.min_tenure_months` limit in `params.yaml` are flagged `out_of_domain` at prediction time and must not be scored by this model.
- Business value depends on save_rate (0.25, `assumed`); the model does not measure the effect of the intervention.
- The model uses customer attributes only; it has no behavioral or interaction-history features.

---

source_hash: `ec4edbad4a54fedb9c8a3cd9b09f0e75e6eebe24ab05da402472a86e2619be1b`
- assumptions.yaml: `01017f1fe28fb85d3a13bb6e94732663b915534052d25d3a3a9af73fa3a66e12`
- metrics.json: `da3c85d6118a608de7e493d08abab5d01ec2ceb357c73ef24ab22dee9be69fd7`
- roi.json: `74511c3bd75ef8cb6469ee92cccfd38ba2fde4a77f0212a5c428eaacb2d3945f`
- ablation.json: `599a8f9148c39b80c99f71c905d05a1def9d89eaeea88b37ecd20df4e6222582`
- train_data: `4d1ea9f3002fe8dbe2c541c6558bb308c1579239a383668397666d0eccdec323`
- preprocess: `eb77a019f364864bde7cb7a79c85503f09f5ba1e7e1f20cc3a47fb067c1dcfa9`
- model: `f0cf7e4bd294fb8ea1a99bbe038b85f011dd14acab7d455e1e06364e871eab10`
- calibrator: `98a8ab80c78f79ae4a6a2d77b50baab4e2e8733bda3d7d06070a2c7a69b8cd39`
