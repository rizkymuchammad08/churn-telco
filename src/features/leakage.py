"""Leakage guards: forbidden columns and a feature-label correlation detector.

The correlation detector only catches blatant leakage (a feature that is nearly a
function of the label). It is not proof that no leakage exists; its negative control
lives in tests/test_leakage.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class LeakageError(ValueError):
    """Leakage detected. `code` is stable; `columns` lists the offending columns."""

    def __init__(self, code: str, message: str, columns: list[str]):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.columns = columns


def forbidden_columns(params: dict) -> set[str]:
    """Raw columns that must not become features: id, label, and (policy "exclude") leakage-prone columns."""
    sc = params["schema"]
    out = {sc["id_column"], sc["label_column"]}
    if params["leakage"]["policy"] == "exclude":
        out |= set(params["leakage"]["flagged_columns"])
    return out


def assert_no_forbidden(X: pd.DataFrame, params: dict) -> None:
    """Reject a matrix that has the id, label, or (policy "exclude") leakage-prone columns as columns.

    The id may be the index (for alignment) but must not be a feature column.
    """
    hit = sorted(set(map(str, X.columns)) & forbidden_columns(params))
    if hit:
        raise LeakageError("E_LEAK_FORBIDDEN_COLUMN", f"forbidden columns in feature matrix: {hit}", hit)


def abs_corr(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """|Pearson| of each feature with the label (point-biserial for a binary label). Constant -> 0."""
    y = y.reindex(X.index)
    if y.isna().any():
        raise ValueError("labels are not aligned with the feature index")
    yc = y.to_numpy(dtype="float64") - y.mean()
    Xc = X.to_numpy(dtype="float64") - X.to_numpy(dtype="float64").mean(axis=0)
    denom = np.sqrt((Xc**2).sum(axis=0) * (yc**2).sum())
    cov = np.abs(Xc.T @ yc)
    # Constant column or label: denom is 0, correlation is undefined and treated as 0.
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    return pd.Series(corr, index=X.columns, name="abs_corr")


def check_correlation(X: pd.DataFrame, y: pd.Series, params: dict) -> pd.Series:
    """Return |corr| per feature; raise LeakageError if any exceeds leakage.max_abs_corr."""
    corr = abs_corr(X, y)
    limit = params["leakage"]["max_abs_corr"]
    hit = sorted(corr.index[corr > limit])
    if hit:
        detail = ", ".join(f"{c}={corr[c]:.3f}" for c in hit)
        raise LeakageError("E_LEAK_CORRELATION", f"|corr| > {limit}: {detail}", hit)
    return corr
