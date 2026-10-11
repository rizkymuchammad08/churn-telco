"""Train / calibration / test splitting.

Path A: one row per customer, so a row that leaks means a customer that leaks.
`random_stratified` does a random split stratified on the label (deterministic given the seed).
`temporal` requires a time axis (`snapshot`) and drops rows whose distance to the next side is
less than `split.purge_gap_days`. The Telco dataset has no such axis, so this branch is only
active when the caller supplies one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MIN_PER_CLASS = 3  # each class must be reasonably likely to appear on every side


class SplitError(ValueError):
    """A split could not be created. `code` is stable for tests and the CLI."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Split:
    """Customer ids per side, sorted by id (the result does not depend on input row order).

    `purged` holds the ids dropped by the purge gap (temporal split only).
    """

    train: list
    calib: list
    test: list
    purged: list = field(default_factory=list)

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "calib": len(self.calib), "test": len(self.test)}


def round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def _targets(n: int, params: dict) -> tuple[int, int, int]:
    """Target sizes (train, calib, test) for n rows."""
    s = params["split"]
    n_test = round_half_up(n * s["test_frac"])
    n_calib = round_half_up(n * s["calib_frac"])
    n_train = n - n_test - n_calib
    if min(n_train, n_calib, n_test) < 1:
        raise SplitError("E_SPLIT_TOO_SMALL", f"n={n} is too small for three sides")
    return n_train, n_calib, n_test


def _check_ids(df: pd.DataFrame, params: dict) -> pd.Series:
    sc = params["schema"]
    if sc["id_column"] not in df.columns:
        raise SplitError("E_SPLIT_COLUMNS", f"column {sc['id_column']} is missing")
    ids = df[sc["id_column"]]
    if ids.isna().any() or ids.duplicated().any():
        raise SplitError("E_SPLIT_ID", f"{sc['id_column']} must be unique and non-null")
    return ids.reset_index(drop=True)


def _random_stratified(df: pd.DataFrame, params: dict, seed: int) -> Split:
    sc = params["schema"]
    ids = _check_ids(df, params)
    if sc["label_column"] not in df.columns:
        raise SplitError("E_SPLIT_COLUMNS", f"column {sc['label_column']} is required for stratification")
    y = (df[sc["label_column"]].reset_index(drop=True) == sc["positive_label"]).to_numpy(dtype="int64")

    # Sort by id first so the result does not depend on the input row order.
    order = np.argsort(ids.to_numpy(dtype=object), kind="stable")
    ids, y = ids.iloc[order].reset_index(drop=True), y[order]

    n_train, n_calib, n_test = _targets(len(df), params)
    if np.bincount(y, minlength=2).min() < MIN_PER_CLASS * 3:
        raise SplitError("E_SPLIT_TOO_SMALL", "each label class needs enough rows for three sides")

    positions = np.arange(len(df))
    rest, test = train_test_split(positions, test_size=n_test, stratify=y, random_state=seed)
    train, calib = train_test_split(rest, test_size=n_calib, stratify=y[rest], random_state=seed)

    def take(idx) -> list:
        return ids.iloc[np.sort(idx)].tolist()

    return Split(train=take(train), calib=take(calib), test=take(test))


def _temporal(df: pd.DataFrame, params: dict, snapshot) -> Split:
    s = params["split"]
    ids = _check_ids(df, params)
    if snapshot is None:
        raise SplitError("E_SPLIT_NO_TIME_AXIS", "a temporal split requires a snapshot per row")
    try:
        t = pd.Series(pd.to_datetime(np.asarray(snapshot)))
    except (ValueError, TypeError) as e:
        raise SplitError("E_SPLIT_NO_TIME_AXIS", f"snapshot cannot be parsed as dates: {e}") from e
    if len(t) != len(df) or t.isna().any():
        raise SplitError("E_SPLIT_NO_TIME_AXIS", "snapshot must be complete and the same length as df")

    n_train, n_calib, _ = _targets(len(df), params)
    times = t.to_numpy()
    # np.lexsort sorts by the LAST key first: time, then id (independent of input row order).
    order = np.lexsort((ids.astype(str).to_numpy(), times))
    gap = pd.Timedelta(days=s["purge_gap_days"])

    # Candidate sides before purging.
    train_cand = order[:n_train]
    calib_cand = order[n_train : n_train + n_calib]
    test_cand = order[n_train + n_calib :]
    calib_start, test_start = times[calib_cand].min(), times[test_cand].min()

    # Keep only rows at least `gap` before the start of the next side.
    train_keep = train_cand[times[train_cand] <= calib_start - gap]
    calib_keep = calib_cand[times[calib_cand] <= test_start - gap]
    if min(len(train_keep), len(calib_keep)) < 1:
        raise SplitError("E_SPLIT_TOO_SMALL", "the purge gap consumed an entire side")

    kept = set(train_keep) | set(calib_keep) | set(test_cand)
    purged = [i for i in range(len(df)) if i not in kept]

    def take(idx) -> list:
        return sorted(ids.iloc[idx].tolist())

    return Split(
        train=take(train_keep),
        calib=take(calib_keep),
        test=take(test_cand),
        purged=sorted(ids.iloc[purged].tolist()),
    )


def make_split(df: pd.DataFrame, params: dict, seed: int | None = None, snapshot=None) -> Split:
    """Split `df` into train/calib/test. Does not modify `df`. `seed` defaults to params.seed."""
    seed = params["seed"] if seed is None else seed
    kind = params["split"]["type"]
    if kind == "random_stratified":
        return _random_stratified(df, params, seed)
    if kind == "temporal":
        col = params["schema"].get("snapshot_column")
        if snapshot is None and col is not None and col in df.columns:
            snapshot = df[col]  # temporal mode: dates come from the schema column
        return _temporal(df, params, snapshot)
    raise SplitError("E_SPLIT_TYPE", f"unknown split.type: {kind}")
