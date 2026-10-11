"""Feature matrix and labels for path A (one row per customer, no time axis).

Features are pure functions of the attribute columns in the row itself. The label is
never read while building features. Columns in `leakage.flagged_columns` (and every
feature derived from them) are dropped when `leakage.policy = exclude`. The output
columns are fixed by `feature_names` and do not depend on the data.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import ROOT, load_params
from src.data.load import DataLoadError, data_hash, load_raw, to_csv_text
from src.data.synthetic import ADDONS, CONTRACTS, PAYMENT
from src.data.validate import banner, validate
from src.features.leakage import LeakageError, assert_no_forbidden, check_correlation, forbidden_columns

LABEL_NAME = "label"
YES_NO = {"Yes", "No"}
INTERNET = ["DSL", "Fiber optic", "No"]

# Fixed vocabulary. Values outside it are rejected, not silently encoded.
VOCAB: dict[str, set[str]] = {
    "gender": {"Female", "Male"},
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "PaperlessBilling": YES_NO,
    "MultipleLines": YES_NO | {"No phone service"},
    "InternetService": set(INTERNET),
    **{c: YES_NO | {"No internet service"} for c in ADDONS},
    "Contract": set(CONTRACTS),
    "PaymentMethod": set(PAYMENT),
}

# One-hot levels per column; the omitted level is the baseline.
ONEHOT = {
    "InternetService": ["DSL", "Fiber optic"],  # baseline: No
    "Contract": ["One year", "Two year"],  # baseline: Month-to-month
    # baseline: Electronic check
    "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)", "Mailed check"],
}


class FeatureError(ValueError):
    """The data cannot be turned into a feature matrix. `code` is stable for tests and the CLI."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Feature:
    name: str
    sources: tuple[str, ...]  # raw columns that are read
    fn: Callable[[pd.DataFrame], pd.Series]


def _slug(text: str) -> str:
    return re.sub(r"\W+", "_", text).strip("_")


def _yes(col: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda d: (d[col] == "Yes").astype("float64")


def _registry() -> list[Feature]:
    """All possible features, in their fixed output order."""
    feats = [
        Feature("is_male", ("gender",), lambda d: (d["gender"] == "Male").astype("float64")),
        Feature("SeniorCitizen", ("SeniorCitizen",), lambda d: d["SeniorCitizen"].astype("float64")),
        *[Feature(c, (c,), _yes(c)) for c in ("Partner", "Dependents", "PhoneService", "PaperlessBilling")],
        Feature("MultipleLines", ("MultipleLines",), _yes("MultipleLines")),
        *[Feature(c, (c,), _yes(c)) for c in ADDONS],
    ]
    for col, levels in ONEHOT.items():
        for lv in levels:
            feats.append(
                Feature(f"{col}_{_slug(lv)}", (col,), lambda d, c=col, v=lv: (d[c] == v).astype("float64"))
            )
    feats += [
        Feature("MonthlyCharges", ("MonthlyCharges",), lambda d: d["MonthlyCharges"].astype("float64")),
        Feature("tenure", ("tenure",), lambda d: d["tenure"].astype("float64")),
        # An empty TotalCharges only occurs when tenure is 0 (enforced by the validator): fill with 0.
        Feature("TotalCharges", ("TotalCharges",), lambda d: d["TotalCharges"].fillna(0.0).astype("float64")),
        Feature(
            "n_addons",
            tuple(ADDONS),
            lambda d: sum((d[c] == "Yes").astype("float64") for c in ADDONS),
        ),
        Feature(
            "has_internet",
            ("InternetService",),
            lambda d: (d["InternetService"] != "No").astype("float64"),
        ),
        Feature(
            "is_autopay",
            ("PaymentMethod",),
            lambda d: d["PaymentMethod"].str.contains("(automatic)", regex=False).astype("float64"),
        ),
    ]
    return feats


def _selected(params: dict) -> list[Feature]:
    """Features whose source columns are all allowed under the leakage policy."""
    banned = forbidden_columns(params)
    return [f for f in _registry() if not set(f.sources) & banned]


def feature_names(params: dict) -> list[str]:
    """Output columns of `build_features` (fixed order, independent of the data)."""
    return [f.name for f in _selected(params)]


def feature_sources(params: dict) -> dict[str, tuple[str, ...]]:
    return {f.name: f.sources for f in _selected(params)}


def _prepare(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Check required columns and id integrity, then use the id as the index."""
    sc = params["schema"]
    missing = [c for c in sc["required_columns"] if c not in df.columns]
    if missing:
        raise FeatureError("E_FEATURE_COLUMNS_MISSING", ", ".join(missing))
    idc = sc["id_column"]
    if df[idc].isna().any() or df[idc].duplicated().any():
        raise FeatureError("E_FEATURE_ID", f"{idc} must be unique and non-empty")
    return df.set_index(idc, drop=True)


def build_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Float64 feature matrix, index = customer id, row order = input order.

    Does not read the label column. Does not modify `df`.
    """
    d = _prepare(df, params)
    for col, vocab in VOCAB.items():
        bad = sorted(set(d[col].astype(str)) - vocab)
        if bad:
            raise FeatureError("E_FEATURE_CATEGORY", f"{col}: unknown values {bad}")
    feats = _selected(params)
    cols = {f.name: f.fn(d).to_numpy(dtype="float64") for f in feats}
    X = pd.DataFrame(cols, index=d.index)
    if not np.isfinite(X.to_numpy()).all():
        raise FeatureError("E_FEATURE_NONFINITE", "feature matrix contains NaN or inf")
    return X


def build_labels(df: pd.DataFrame, params: dict) -> pd.Series:
    """Binary int64 labels (1 = positive_label), index = customer id."""
    sc = params["schema"]
    d = _prepare(df, params)
    lab = d[sc["label_column"]]
    bad = sorted(set(lab.astype(str)) - {sc["positive_label"], sc["negative_label"]})
    if bad:
        raise FeatureError("E_LABEL_DOMAIN", f"invalid values: {bad}")
    return (lab == sc["positive_label"]).astype("int64").rename(LABEL_NAME)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the feature matrix and labels")
    ap.add_argument("--sample", action="store_true", help="use data/sample.csv (default: raw)")
    args = ap.parse_args(argv)
    params = load_params()
    if (b := banner(params)) is not None:
        print(b)
    try:
        df = load_raw("sample" if args.sample else "raw", params)
    except DataLoadError as e:
        print(f"{e.code}: {e.message}")
        return 1
    rep = validate(df, params)
    if not rep.ok:
        for code, details in rep.errors.items():
            print(f"{code}: {'; '.join(details)}")
        return 1
    try:
        X, y = build_features(df, params), build_labels(df, params)
        assert_no_forbidden(X, params)
        corr = check_correlation(X, y, params)
    except (FeatureError, LeakageError) as e:
        print(f"{e.code}: {e.message}")
        return 1
    out = ROOT / params["paths"]["artifacts_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "features.csv").write_text(to_csv_text(X.reset_index()), encoding="utf-8", newline="\n")
    (out / "labels.csv").write_text(to_csv_text(y.reset_index()), encoding="utf-8", newline="\n")
    print(
        f"rows={len(X)} features={X.shape[1]} policy={params['leakage']['policy']} "
        f"max_abs_corr={corr.max():.3f} (limit {params['leakage']['max_abs_corr']}) "
        f"features_hash={data_hash(X.reset_index())[:16]}"
    )
    print("FEATURES OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
