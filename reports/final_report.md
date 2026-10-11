# Final Report: Telco Churn Prediction

Data source: `real_public_ibm_telco` (file `data/raw`). Selected model: `logistic_regression`, with primary metric pr_auc.

Input data sha256: `deae3674baa798c67ebea55e8ed8b59fde59757f7d4d0412f8336abc377aca87` (all file rows after cleaning, canonical content with rows sorted; recompute with `src.data.load.data_hash`).

> **Conditional value.** save_rate (0.25) has status `assumed`; it is not a measured result. All business-value figures in this report hold only if that assumption is true.

## 1. Data and split

Stratified random split: 4225 train, 1409 calibration, 1409 test rows. The test set contains 374 churners (base rate 0.265). The test set is touched once; the calibrator is trained only on the calibration set.

## 2. Model comparison

| Model | PR-AUC (CV) | PR-AUC (test) | ROC-AUC (test) | Lift@140 (test) | Brier (test) |
|---|---|---|---|---|---|
| gradient_boosting | 0.613 | 0.578 [0.530, 0.631] | 0.824 [0.801, 0.846] | 2.48 [2.19, 2.80] | 0.145 [0.135, 0.155] |
| prior_rate | 0.265 | 0.265 [0.241, 0.287] | 0.500 [0.500, 0.500] | 1.00 [1.00, 1.00] | 0.195 [0.184, 0.205] |
| logistic_regression | 0.622 | 0.571 [0.522, 0.622] | 0.822 [0.799, 0.843] | 2.58 [2.31, 2.85] | 0.146 [0.136, 0.156] |

Test-set columns show the point estimate and, in brackets, the 95% bootstrap interval (1000 resamples of the test rows). The intervals capture test-set sampling noise only, not variation from training or from changes over time; the CV column is a mean without an interval.

The model was selected by pr_auc in cross-validation, not on the test set. Overlapping intervals mean the test set cannot separate those models; this is a visual check, not a formal test of the difference.

### 2.1 Hyperparameter tuning

Each model's default hyperparameters were compared with a small grid by 5-fold cross-validation on the train rows only, scored by pr_auc. The calibration and test rows were not used. The rule was fixed and locked before the run (rule hash `3438986f4849f68d2d686fab96d7e60cde4d13a99bf52f2f2bc8d2b8c00bea61`): the best candidate replaces the default only if its mean paired gain over the default across folds is positive and larger than 1.0 standard error of that gain; otherwise the default stays. Tuning folds use a different split from the model-selection folds (seed offset 1000), so the CV column in the table above is not computed on the folds that chose the hyperparameters.

**`gradient_boosting`**

| Candidate | PR-AUC (tuning CV) |
|---|---|
| `learning_rate=0.05, max_depth=2` | 0.607 |
| `learning_rate=0.05, max_depth=3` | 0.607 |
| `learning_rate=0.1, max_depth=2` | 0.611 |
| `learning_rate=0.1, max_depth=3` (default) | 0.602 |

Best candidate: `learning_rate=0.1, max_depth=2`; mean paired gain over the default 0.0087, required more than 0.0026. Decision: replaced the default `learning_rate=0.1, max_depth=3` with `learning_rate=0.1, max_depth=2`.

**`logistic_regression`**

| Candidate | PR-AUC (tuning CV) |
|---|---|
| `C=0.1` | 0.620 |
| `C=0.3` | 0.619 |
| `C=1.0` (default) | 0.619 |
| `C=3.0` | 0.619 |

Best candidate: `C=0.1`; mean paired gain over the default 0.0006, required more than 0.0004. Decision: replaced the default `C=1.0` with `C=0.1`.

A gain this small is within what a different split could change; the tuning is a light check, not a search for a better model, and it does not change how the test set is used.

## 3. Calibration

Method: `isotonic`. ECE dropped from 0.150 to 0.020; tolerance is 0.05, so the result is within tolerance.

## 4. Performance by segment

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

Segments marked as not reliable are too small or contain only one class; their figures must not be used for decisions.

## 5. Contact decision and business value

Break-even threshold p* = 0.1333; campaign capacity on the test set k = 140.

| Policy | Contacts | Churners contacted | Net value (USD) (conditional value) |
|---|---|---|---|
| no_contact | 0 | 0.0 | 0 |
| model_p_star | 140 | 97.0 | 23,500 |
| random_k | 140 | 37.2 | 5,548 |

Incremental value of the model versus no contact: 23,500 USD (conditional value).
Incremental value of the model versus random contact: 17,952 USD (conditional value).
Break-even save_rate versus no contact: 0.048.

## 6. Excluded columns: a policy decision with a measured cost

`tenure` and `TotalCharges` are excluded from the features by manual policy (`leakage.policy = exclude`). This is a policy decision, not a demonstrated case of leakage.

| Column | abs. correlation with label | Detector limit |
|---|---|---|
| tenure | 0.352 | 0.90 |
| TotalCharges | 0.198 | 0.90 |

The correlation detector did not flag either column; the exclusion does not come from the detector.

The reason for excluding them is plausible: both accumulate over a customer's lifetime, and for a churner they stop at the point of churn, so a model that uses them partly measures how long customers stayed. This cannot be verified with this data, because it has no dates (no signup, snapshot, or churn date) from which to show when the values were measured relative to the prediction.

Cost of the policy, on the test set, over 5 random splits (diagnostic only; the official model above does not use these columns):

| Metric | Without (official) | With both columns | Change (mean) | Change (sd) | Splits where "with" is better |
|---|---|---|---|---|---|
| PR-AUC | 0.576 | 0.630 | +0.054 | 0.024 | 5 of 5 |
| ROC-AUC | 0.825 | 0.842 | +0.016 | 0.007 | 5 of 5 |
| Lift@k | 2.555 | 2.864 | +0.308 | 0.097 | 5 of 5 |
| Brier | 0.145 | 0.137 | -0.008 | 0.003 | 5 of 5 |
| ECE | 0.027 | 0.024 | -0.002 | 0.005 | 4 of 5 |

Using both columns raises PR-AUC by 0.054 on average. The selected model changes from `gradient_boosting, logistic_regression` to `gradient_boosting` in 4 of 5 splits. The splits are different random partitions of the same rows, so consistency across them is not an independent replication or a significance test.

If `tenure` and `TotalCharges` are known to be measured at or before the time a prediction would be made (for example, from a dated snapshot), the policy should be revisited. If they are measured at observation time, the higher score would not carry over to real prediction.

## 7. Limitations

- `tenure` and `TotalCharges` are excluded by policy, not because leakage was demonstrated; the measured cost of that policy is in the section on excluded columns.
- The data has one row per customer and no dates: there is no prediction window and no temporal validation.
- Business value depends on save_rate; until it is measured in a controlled trial, the net-value figures are a scenario, not a forecast of results.

---

source_hash: `ec4edbad4a54fedb9c8a3cd9b09f0e75e6eebe24ab05da402472a86e2619be1b`
- assumptions.yaml: `01017f1fe28fb85d3a13bb6e94732663b915534052d25d3a3a9af73fa3a66e12`
- metrics.json: `da3c85d6118a608de7e493d08abab5d01ec2ceb357c73ef24ab22dee9be69fd7`
- roi.json: `74511c3bd75ef8cb6469ee92cccfd38ba2fde4a77f0212a5c428eaacb2d3945f`
- ablation.json: `599a8f9148c39b80c99f71c905d05a1def9d89eaeea88b37ecd20df4e6222582`
