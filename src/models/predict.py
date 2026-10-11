"""Score new customers with a trained model and calibrator.

Input: a raw table in the Telco schema (the label column is not required).
Output: a DataFrame indexed by customer id (same row order and count as the input) with columns:
  p_raw         raw model score,
  p_churn       calibrated probability in [0, 1],
  rank          1 = highest risk; unique; ties in p_churn are broken by p_raw, then row order,
  out_of_domain True when tenure < domain.min_tenure_months.
Inputs outside the feature domain (unknown categories, negative numbers, missing values)
are rejected with a clear error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load import clean
from src.features.build import build_features
from src.models.train import Bundle

PRED_COLUMNS = ["p_raw", "p_churn", "rank", "out_of_domain"]


class PredictError(ValueError):
    """Invalid prediction input. `code` is stable for tests and the CLI."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _check_domain(d: pd.DataFrame, params: dict) -> None:
    sc = params["schema"]
    skip = {sc["label_column"], sc["nullable_when_tenure_zero"]}
    nulls = [c for c in sc["required_columns"] if c not in skip and c in d and d[c].isna().any()]
    if nulls:
        raise PredictError("E_PREDICT_DOMAIN", f"missing values in columns: {nulls}")
    negative = [c for c in ("tenure", "MonthlyCharges") if (d[c] < 0).any()]
    if negative:
        raise PredictError("E_PREDICT_DOMAIN", f"negative values in columns: {negative}")
    if not d["SeniorCitizen"].isin([0, 1]).all():
        raise PredictError("E_PREDICT_DOMAIN", "SeniorCitizen must be 0 or 1")


def predict(df: pd.DataFrame, bundle: Bundle, params: dict) -> pd.DataFrame:
    sc = params["schema"]
    d = df.copy()
    if sc["label_column"] not in d.columns:
        # Features never read the label; this column only satisfies the schema check.
        d[sc["label_column"]] = sc["negative_label"]
    missing = [c for c in sc["required_columns"] if c not in d.columns]
    if missing:
        raise PredictError("E_PREDICT_COLUMNS", f"missing columns: {missing}")
    d = clean(d, params, name="prediction input")
    _check_domain(d, params)
    X = build_features(d, params)  # raises FeatureError on duplicate ids or unknown categories
    if list(map(str, X.columns)) != bundle.model.feature_names:
        raise PredictError("E_PREDICT_FEATURES", "current features do not match the model's features")

    p_raw = bundle.model.predict_raw(X)
    p_churn = bundle.calibrator.transform(p_raw)
    n = len(X)
    # np.lexsort sorts by the LAST key first: -p_churn, then -p_raw, then row order.
    order = np.lexsort((np.arange(n), -p_raw, -p_churn))
    rank = np.empty(n, dtype="int64")
    rank[order] = np.arange(1, n + 1)
    out = pd.DataFrame(
        {
            "p_raw": p_raw,
            "p_churn": p_churn,
            "rank": rank,
            "out_of_domain": (d["tenure"].to_numpy() < params["domain"]["min_tenure_months"]),
        },
        index=X.index,
    )
    return out[PRED_COLUMNS]
