from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import pandas as pd

from src.config import load_params
from src.data.load import DataLoadError, data_hash, load_raw

BANNER = "*** SYNTHETIC DATA: results do not represent real customers ***"

# Every error code that can be raised. Tests ensure each code has a fixture or a dedicated test.
ALL_CODES = (
    "E_FILE_MISSING",
    "E_CSV_PARSE",
    "E_NONNUMERIC",
    "E_DATA_SOURCE",
    "E_EMPTY",
    "E_COLUMNS_MISSING",
    "E_NULL_ID",
    "E_DUP_ID",
    "E_LABEL_DOMAIN",
    "E_NULL_CATEGORY",
    "E_NULL_NUMERIC",
    "E_NEGATIVE_NUMERIC",
    "E_LEAKAGE_COLUMN_UNKNOWN",
    "E_SNAPSHOT_INVALID",
)


@dataclass
class ValidationReport:
    errors: dict[str, list[str]] = field(default_factory=dict)
    n_rows: int = 0
    data_source: str = ""

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, code: str, detail: str) -> None:
        self.errors.setdefault(code, []).append(detail)


def banner(params: dict) -> str | None:
    """Return the warning banner automatically for synthetic data."""
    return BANNER if str(params["data_source"]).startswith("synthetic") else None


def _head(items, n=5) -> str:
    """Show the first `n` items, with a count of how many more were omitted."""
    items = list(items)
    extra = f" (+{len(items) - n} more)" if len(items) > n else ""
    return ", ".join(map(str, items[:n])) + extra


def validate(events: pd.DataFrame, params: dict) -> ValidationReport:
    """Check structure and integrity. Does not check outcomes (e.g. the churn rate)."""
    sc = params["schema"]
    rep = ValidationReport(n_rows=len(events), data_source=params["data_source"])

    # --- Source and overall shape ---
    if params["data_source"] not in params["data_source_allowed"]:
        rep.add("E_DATA_SOURCE", f"unknown data_source {params['data_source']!r}")
    if len(events) == 0:
        rep.add("E_EMPTY", "no rows")
        return rep
    missing = [c for c in sc["required_columns"] if c not in events.columns]
    if missing:
        rep.add("E_COLUMNS_MISSING", _head(missing))
        return rep

    # --- ID column: must be present and unique ---
    ids = events[sc["id_column"]]
    bad_id = ids.isna() | (ids.astype(str).str.strip() == "")
    if bad_id.any():
        rep.add("E_NULL_ID", f"{int(bad_id.sum())} rows without an id")
    dup = ids[ids.duplicated(keep=False) & ~bad_id]
    if len(dup):
        rep.add("E_DUP_ID", _head(sorted(dup.unique())))

    # --- Label column: only the two allowed values ---
    lab = events[sc["label_column"]]
    ok_vals = {sc["positive_label"], sc["negative_label"]}
    bad_lab = lab[~lab.isin(ok_vals)]
    if len(bad_lab):
        rep.add("E_LABEL_DOMAIN", f"invalid values: {_head(sorted(set(map(str, bad_lab))))}")

    # --- Categorical columns: no empty values ---
    numeric = set(sc["int_columns"]) | set(sc["float_columns"])
    non_categorical = numeric | {sc["id_column"], sc["label_column"]}
    categorical = [c for c in sc["required_columns"] if c not in non_categorical]
    for c in categorical:
        empty = events[c].isna() | (events[c].astype(str).str.strip() == "")
        if empty.any():
            rep.add("E_NULL_CATEGORY", f"{c}: {int(empty.sum())} empty values")

    # --- Numeric columns: no nulls (one exception) and no negatives ---
    tenure_is_zero = events["tenure"] == 0
    nullable_col = sc.get("nullable_when_tenure_zero")
    for c in sc["int_columns"] + sc["float_columns"]:
        col = events[c]
        is_null = col.isna()
        if c == nullable_col:
            # This column may be empty, but only for customers with tenure = 0.
            is_null = is_null & ~tenure_is_zero
        if is_null.any():
            rep.add("E_NULL_NUMERIC", f"{c}: {int(is_null.sum())} empty values not allowed")
        if (col < 0).any():
            rep.add("E_NEGATIVE_NUMERIC", f"{c}: {int((col < 0).sum())} negative values")

    # --- Temporal mode: every row needs a parseable snapshot date ---
    snap = sc.get("snapshot_column")
    if snap is not None:
        parsed = pd.to_datetime(events[snap], errors="coerce")
        n_bad = int(parsed.isna().sum())
        if n_bad:
            rep.add("E_SNAPSHOT_INVALID", f"{snap}: {n_bad} values are empty or not dates")

    # --- Leakage config: every flagged column must exist in the data ---
    unknown_leak = [c for c in params["leakage"]["flagged_columns"] if c not in events.columns]
    if unknown_leak:
        rep.add("E_LEAKAGE_COLUMN_UNKNOWN", _head(unknown_leak))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate churn data")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--sample", action="store_true", help="validate data/sample.csv")
    group.add_argument("--path", help="validate a specific CSV file")
    args = ap.parse_args(argv)
    params = load_params()
    if (b := banner(params)) is not None:
        print(b)
    source = args.path if args.path else ("sample" if args.sample else "raw")
    try:
        df = load_raw(source, params)
    except DataLoadError as e:
        print(f"{e.code}: {e.message}")
        return 1
    rep = validate(df, params)
    print(f"data_source={rep.data_source} rows={rep.n_rows} data_hash={data_hash(df)[:16]}")
    if rep.ok:
        print("VALID")
        return 0
    for code, details in rep.errors.items():
        print(f"{code}: {'; '.join(details)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
