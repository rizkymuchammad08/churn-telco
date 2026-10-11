"""Probability calibrator. Trained only from raw scores and calib labels.

`isotonic`: isotonic regression, non-decreasing, output in [0, 1].
`sigmoid`: Platt scaling on the logit of the raw score; rejected if the slope is negative
(scores that point the opposite way to the label must not be silently "fixed").
Raw ranking is preserved: the mapping is monotonic, and any ties it creates are broken by
the caller using the raw score (see predict.py).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


class CalibrationError(ValueError):
    """The calibrator cannot be trained. `code` is stable for tests and the CLI."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


@dataclass
class Calibrator:
    method: str
    model: Any  # IsotonicRegression or LogisticRegression (Platt)
    n_fit: int

    def transform(self, p_raw) -> np.ndarray:
        p = np.asarray(p_raw, dtype="float64")
        if p.ndim != 1:
            raise ValueError("scores must be a vector")
        if len(p) == 0:
            return p.copy()
        if not np.isfinite(p).all():
            raise ValueError("scores contain NaN or inf")
        if self.method == "isotonic":
            out = self.model.predict(p)
        else:
            out = self.model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return np.clip(np.asarray(out, dtype="float64"), 0.0, 1.0)

    def state(self) -> dict:
        """The calibrator's contents as a plain structure (for hashing and reports)."""
        if self.method == "isotonic":
            return {
                "method": "isotonic",
                "x": self.model.X_thresholds_.tolist(),
                "y": self.model.y_thresholds_.tolist(),
            }
        return {
            "method": "sigmoid",
            "coef": float(self.model.coef_[0, 0]),
            "intercept": float(self.model.intercept_[0]),
        }

    def state_hash(self) -> str:
        blob = json.dumps(self.state(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def fit_calibrator(p_raw_calib, y_calib, params: dict) -> Calibrator:
    """Train a calibrator from the calib (raw score, label) pairs only."""
    p = np.asarray(p_raw_calib, dtype="float64")
    y = np.asarray(y_calib, dtype="float64")
    if p.ndim != 1 or p.shape != y.shape or len(p) == 0:
        raise CalibrationError(
            "E_CALIB_INPUT", "calib scores and labels must be non-empty vectors of the same length"
        )
    if not np.isfinite(p).all():
        raise CalibrationError("E_CALIB_INPUT", "calib scores contain NaN or inf")
    if not np.isin(y, (0.0, 1.0)).all():
        raise CalibrationError("E_CALIB_INPUT", "calib labels must be binary 0/1")
    if len(np.unique(y)) < 2:
        raise CalibrationError("E_CALIB_ONE_CLASS", "calib labels contain only one class")
    method = params["calibration"]["method"]
    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        iso.fit(p, y)
        return Calibrator("isotonic", iso, len(p))
    if method == "sigmoid":
        lr = LogisticRegression(C=1e6, max_iter=1000)  # effectively unregularized
        lr.fit(_logit(p).reshape(-1, 1), y)
        if lr.coef_[0, 0] < 0:
            raise CalibrationError(
                "E_CALIB_ANTI_MONOTONE", "scores point the opposite way to the label on calib"
            )
        return Calibrator("sigmoid", lr, len(p))
    raise CalibrationError("E_CALIB_METHOD", f"unknown method: {method}")
