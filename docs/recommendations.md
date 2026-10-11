# Recommendations: Churn Retention Campaign

> **Conditional value.** save_rate (0.25) has status `assumed`; it is not a measured result. All business-value figures in this document are scenarios.

## 1. Decision summary

Contact customers whose calibrated churn probability is above p* = 0.1333, up to 140 contacts on the test set. On the test set, this policy contacts 140 customers, including 97.0 churners; random contact of the same size reaches 37.2 churners.

- Incremental value versus no contact: 23,500 USD (conditional value).
- Incremental value versus random contact: 17,952 USD (conditional value).
- Break-even save_rate versus no contact: 0.048. The assumed save_rate (0.25) is above that limit.

## 2. Recommended

- **Measure save_rate in a controlled trial** (a holdout group that is not contacted) before scaling up the campaign. Once measured, update `save_rate` and `save_rate_source` in `assumptions.yaml`, then run `make all` so that p* and all documents are recomputed.
- **Use the p* threshold, not the ranking alone.** The selected model is `logistic_regression`; customers below p* are not contacted even when capacity remains, because under the current assumptions the net value per contact below p* is negative.
- **Re-evaluate on the full data** (population of 7043 customers per period, capacity 700) before using the performance figures for planning. The current figures come from a test set of 1409 rows with 374 churners.
- **Monitor calibration.** Calibrated ECE is 0.020 against a tolerance of 0.05: within tolerance.
- **Read per-segment figures only for reliable segments.** The base rate and mean p of each segment are in the Performance by segment section of the final report; monitor them per segment every period.

## 3. NOT recommended

- **Treating net value as a forecast of results.** save_rate has status `assumed`; the net-value figures are only a scenario until save_rate is measured.
- **Scaling test-set results up to the whole population.** The test set has only 1409 rows (374 churners); the uncertainty of extrapolating to 7043 customers has not been quantified.
- **Switching the selected model based on test scores.** `gradient_boosting` reaches a test PR-AUC of 0.578, above the selected model `logistic_regression`, but selection was done by pr_auc in cross-validation. Re-selecting from the test set would make the test set no longer independent, and the significance of the difference on 1409 rows has not been tested.
- **Changing the exclusion of `tenure` and `TotalCharges` without dated data.** Using both would raise PR-AUC by 0.054 on average, but the data has no dates, so it cannot be shown whether these values are known at prediction time. Treat the gain as unconfirmed, and decide the policy with the owner of the data.
- **Interpreting the score as an intervention effect.** The score estimates churn risk, not who can be saved; save_rate is assumed to be the same for all customers. An uplift model is a separate piece of work.
- **Lowering p* or raising capacity just to reach more churners.** p* is derived from clv (1,200 USD), intervention cost (40 USD), and save_rate; changing it means changing the assumptions, and net value must be recomputed.
- **Scoring customers outside the scoring domain.** Customers with tenure below `domain.min_tenure_months` in `params.yaml` are flagged `out_of_domain` and must not enter the contact list based on their score.

---

source_hash: `ec4edbad4a54fedb9c8a3cd9b09f0e75e6eebe24ab05da402472a86e2619be1b`
- assumptions.yaml: `01017f1fe28fb85d3a13bb6e94732663b915534052d25d3a3a9af73fa3a66e12`
- metrics.json: `da3c85d6118a608de7e493d08abab5d01ec2ceb357c73ef24ab22dee9be69fd7`
- roi.json: `74511c3bd75ef8cb6469ee92cccfd38ba2fde4a77f0212a5c428eaacb2d3945f`
- ablation.json: `599a8f9148c39b80c99f71c905d05a1def9d89eaeea88b37ecd20df4e6222582`
