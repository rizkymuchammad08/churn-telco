"""Seeded synthetic data generator using the Telco schema. For tests (calibration, baselines).

`p_true` is for tests only and is never written to a data file. It depends only on
columns outside `leakage.flagged_columns`, so a model that excludes those columns can
still recover it. The intercept is solved so that mean(p_true) = baseline_churn_rate
(assumptions.yaml), which makes the synthetic base rate equal to the baseline. A small
fraction of rows gets an extreme MonthlyCharges (above the real-data range) as a hidden
outlier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_assumptions

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
PAYMENT = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
TENURE_SCALE = {"Month-to-month": 15.0, "One year": 35.0, "Two year": 50.0}
INTERNET_BASE = {"No": 20.0, "DSL": 55.0, "Fiber optic": 85.0}
OUTLIER_RATE = 0.01
OUTLIER_MULTIPLIER = 2.5  # outlier MonthlyCharges = real-data upper bound (118.75) x multiplier
MAX_MONTHLY = 118.75


def _solve_intercept(rest: np.ndarray, target: float) -> float:
    """Find b such that mean(sigmoid(b + rest)) = target (bisection; monotonic in b)."""
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if (1.0 / (1.0 + np.exp(-(mid + rest)))).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def generate(seed: int, params: dict, base_rate: float | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (events, p_true). `base_rate` defaults to baseline_churn_rate in assumptions.yaml."""
    if base_rate is None:
        base_rate = load_assumptions()["baseline_churn_rate"]
    sc = params["schema"]
    n = params["sample"]["n_rows"]
    rng = np.random.default_rng(seed)

    def pick(options, probs):
        return rng.choice(options, size=n, p=probs)

    # --- Contract and tenure (longer contracts -> longer tenure) ---
    contract = pick(CONTRACTS, [0.55, 0.21, 0.24])
    scale = np.array([TENURE_SCALE[c] for c in contract])
    tenure = np.clip(np.round(rng.gamma(2.0, scale / 2.0)), 1, 72).astype("int64")
    tenure[rng.random(n) < 0.002] = 0  # a few brand-new customers with tenure 0

    # --- Services ---
    phone = pick(["Yes", "No"], [0.9, 0.1])
    multiple = np.where(phone == "No", "No phone service", pick(["Yes", "No"], [0.45, 0.55]))
    internet = pick(["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22])
    addons = {c: np.where(internet == "No", "No internet service", pick(["Yes", "No"], [0.4, 0.6])) for c in ADDONS}
    n_addons = sum((v == "Yes").astype(int) for v in addons.values())

    # --- Demographics and billing preferences ---
    senior = rng.choice([0, 1], size=n, p=[0.84, 0.16]).astype("int64")
    paperless = pick(["Yes", "No"], [0.59, 0.41])
    payment = pick(PAYMENT, [0.34, 0.23, 0.22, 0.21])

    # --- Charges: monthly depends on internet type and add-ons; a few extreme outliers ---
    monthly = np.array([INTERNET_BASE[i] for i in internet]) + 5.0 * n_addons + rng.normal(0, 8, n)
    monthly = np.clip(monthly, 18.25, MAX_MONTHLY)
    outlier = rng.random(n) < OUTLIER_RATE
    monthly = np.where(outlier, MAX_MONTHLY * OUTLIER_MULTIPLIER, monthly)
    monthly = np.round(monthly, 2)
    total = np.round(tenure * monthly * rng.uniform(0.95, 1.05, n), 2)
    total = np.where(tenure == 0, np.nan, total)  # TotalCharges is empty when tenure = 0

    # --- True churn probability and label ---
    rest = (
        1.2 * (contract == "Month-to-month")
        - 0.8 * (contract == "Two year")
        + 0.8 * (internet == "Fiber optic")
        + 0.3 * senior
        + 0.3 * (paperless == "Yes")
        + 0.5 * (payment == "Electronic check")
        + 0.012 * (monthly - 65.0)
    )
    p_true = 1.0 / (1.0 + np.exp(-(_solve_intercept(rest, base_rate) + rest)))
    churn = np.where(rng.random(n) < p_true, sc["positive_label"], sc["negative_label"])

    cols = {
        sc["id_column"]: [f"SYN-{i:05d}" for i in range(1, n + 1)],
        "gender": pick(["Female", "Male"], [0.5, 0.5]),
        "SeniorCitizen": senior,
        "Partner": pick(["Yes", "No"], [0.48, 0.52]),
        "Dependents": pick(["Yes", "No"], [0.30, 0.70]),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        **addons,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        sc["label_column"]: churn,
    }
    if set(cols) != set(sc["required_columns"]):
        raise ValueError("generator columns do not match schema.required_columns")
    events = pd.DataFrame(cols)[sc["required_columns"]]
    return events, p_true
