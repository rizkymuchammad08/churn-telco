"""Evaluation metrics: PR-AUC, ROC-AUC, precision/recall/lift@k, Brier, ECE, per-segment metrics,
and bootstrap confidence intervals for the headline metrics.

Tie handling at top-k is tie-aware and deterministic: when the k cutoff splits a group of
equal scores, that group contributes the expected number of positives (slots x the group's
positive rate). The result does not depend on row order, and a constant score gives
lift@k = exactly 1. `k` is not read from params (the caller derives it from assumptions).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _arrays(y_true, p) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype="float64")
    s = np.asarray(p, dtype="float64")
    if y.ndim != 1 or y.shape != s.shape:
        raise ValueError("y_true and scores must be vectors of the same length")
    if len(y) == 0:
        raise ValueError("empty data")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN or inf")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_true must be binary 0/1")
    return y, s


def _check_k(k: int, n: int) -> None:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= n:
        raise ValueError(f"k must be an integer in [1, {n}], got {k!r}")


def expected_positives_at_k(y_true, score, k: int) -> float:
    """Expected number of positives in the top-k (tie-aware)."""
    y, s = _arrays(y_true, score)
    _check_k(k, len(y))
    threshold = np.sort(s)[::-1][k - 1]
    above = s > threshold
    tied = s == threshold
    slots = k - int(above.sum())  # slots left for the tied group at the cutoff
    return float(y[above].sum() + slots * y[tied].mean())


def precision_at_k(y_true, score, k: int) -> float:
    return expected_positives_at_k(y_true, score, k) / k


def recall_at_k(y_true, score, k: int) -> float:
    y, _ = _arrays(y_true, score)
    n_pos = y.sum()
    return math.nan if n_pos == 0 else expected_positives_at_k(y_true, score, k) / n_pos


def lift_at_k(y_true, score, k: int) -> float:
    """precision@k divided by the overall positive rate."""
    y, _ = _arrays(y_true, score)
    base = y.mean()
    return math.nan if base == 0 else precision_at_k(y_true, score, k) / base


def brier(y_true, p) -> float:
    y, s = _arrays(y_true, p)
    return float(np.mean((s - y) ** 2))


def expected_calibration_error(y_true, p, bins: int) -> float:
    """ECE with equal-width bins on [0, 1]; empty bins are skipped. p outside [0, 1] is rejected."""
    y, s = _arrays(y_true, p)
    if bins < 1:
        raise ValueError("bins must be >= 1")
    if s.min() < 0 or s.max() > 1:
        raise ValueError("ECE requires probabilities in [0, 1]")
    idx = np.minimum(np.floor(s * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(s[m].mean() - y[m].mean())
    return float(ece)


def pr_auc(y_true, score) -> float:
    y, s = _arrays(y_true, score)
    return math.nan if y.sum() == 0 else float(average_precision_score(y, s))


def roc_auc(y_true, score) -> float:
    y, s = _arrays(y_true, score)
    return math.nan if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def metric_value(name: str, y_true, score, k: int) -> float:
    """Primary metric for model selection (`models.primary_metric`)."""
    if name == "pr_auc":
        return pr_auc(y_true, score)
    if name == "lift_at_k":
        return lift_at_k(y_true, score, k)
    raise ValueError(f"unknown metric: {name}")


# ECE is deliberately absent: resampling inflates it (duplicated rows make every bin noisier), so a
# percentile interval would sit above the point estimate and mislead.
CI_METRICS = ("pr_auc", "roc_auc", "precision_at_k", "recall_at_k", "lift_at_k", "brier")


def bootstrap_ci(y_true, p, params: dict, *, k: int, seed: int) -> dict:
    """Percentile bootstrap intervals over test rows for the headline metrics.

    Rows are resampled with replacement (same n, same `k`) `evaluation.bootstrap_resamples` times.
    The resampled row indices depend only on `seed` and n, so every model evaluated on the same test
    set is compared on identical resamples. The interval covers test-set sampling noise only: it
    does not include variation from training, calibration, or a shift in the data over time.
    """
    y, s = _arrays(y_true, p)
    _check_k(k, len(y))
    ev = params["evaluation"]
    n_boot, level = ev["bootstrap_resamples"], ev["ci_level"]
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in CI_METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, sb = y[idx], s[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):  # one class only: AUCs undefined, skip the draw
            continue
        draws["pr_auc"].append(pr_auc(yb, sb))
        draws["roc_auc"].append(roc_auc(yb, sb))
        draws["precision_at_k"].append(precision_at_k(yb, sb, k))
        draws["recall_at_k"].append(recall_at_k(yb, sb, k))
        draws["lift_at_k"].append(lift_at_k(yb, sb, k))
        draws["brier"].append(brier(yb, sb))
    n_used = len(draws["pr_auc"])
    if n_used == 0:
        raise ValueError("bootstrap produced no usable resamples (one class only)")
    alpha = (1.0 - level) / 2.0
    out = {
        m: {
            "lo": _clean(float(np.quantile(v, alpha))),
            "hi": _clean(float(np.quantile(v, 1.0 - alpha))),
        }
        for m, v in draws.items()
    }
    out["level"] = float(level)
    out["level_pct"] = int(round(level * 100))
    out["n_resamples"] = int(n_used)
    return out


def _clean(v):
    """NaN -> None so the JSON is valid; numpy -> Python types."""
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def by_segment(y_true, p, segments: pd.DataFrame, params: dict) -> dict:
    """Metrics per level for each column in `segments`. n and n_pos per column sum to the totals.

    `reliable` = n >= segments.min_n and both classes are present in the segment.
    """
    y, s = _arrays(y_true, p)
    if len(segments) != len(y):
        raise ValueError("segments must have the same length as y_true")
    min_n = params["segments"]["min_n"]
    out: dict = {}
    for col in segments.columns:
        levels: dict = {}
        for level, pos in segments.reset_index(drop=True).groupby(col, sort=True).indices.items():
            ys, ss = y[pos], s[pos]
            n_pos = int(ys.sum())
            levels[str(level)] = {
                "n": len(pos),
                "n_pos": n_pos,
                "base_rate": float(ys.mean()),
                "mean_p": float(ss.mean()),
                "brier": brier(ys, ss),
                "roc_auc": _clean(roc_auc(ys, ss)),
                "reliable": bool(len(pos) >= min_n and 0 < n_pos < len(pos)),
            }
        out[col] = levels
    return out


def evaluate(
    y_true,
    p,
    params: dict,
    *,
    k: int,
    segments: pd.DataFrame | None = None,
    ci_seed: int | None = None,
) -> dict:
    """Metric summary for one probability vector `p`. JSON-safe.

    With `ci_seed` set, adds `ci`: bootstrap intervals for the headline metrics (see `bootstrap_ci`).
    """
    y, s = _arrays(y_true, p)
    _check_k(k, len(y))
    res = {
        "n": len(y),
        "n_pos": int(y.sum()),
        "base_rate": float(y.mean()),
        "k": int(k),
        "pr_auc": pr_auc(y, s),
        "roc_auc": roc_auc(y, s),
        "precision_at_k": precision_at_k(y, s, k),
        "recall_at_k": recall_at_k(y, s, k),
        "lift_at_k": lift_at_k(y, s, k),
        "brier": brier(y, s),
        "ece": expected_calibration_error(y, s, params["evaluation"]["ece_bins"]),
    }
    res = {key: _clean(v) for key, v in res.items()}
    if ci_seed is not None:
        res["ci"] = bootstrap_ci(y, s, params, k=k, seed=ci_seed)
    if segments is not None:
        res["by_segment"] = by_segment(y, s, segments, params)
    return res
