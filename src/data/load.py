from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.config import ROOT


class DataLoadError(ValueError):
    """The file cannot be read as a valid table. `code` is used by the validator and CLI."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _resolve(source, params) -> Path:
    """Map "raw", "sample", or a path (absolute or relative to ROOT) to a file path."""
    if source in ("raw", "sample"):
        return ROOT / params["paths"][source]
    p = Path(source)
    return p if p.is_absolute() else ROOT / p


def _strip(s: pd.Series) -> pd.Series:
    """Strip whitespace from string values only; leave non-string values untouched."""
    return s.map(lambda v: v.strip() if isinstance(v, str) else v)


def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def clean(df: pd.DataFrame, params: dict, name: str = "data") -> pd.DataFrame:
    """Normalize shape: strip strings and convert numeric columns. Idempotent.

    Empty values in numeric columns become NaN. Int columns stay int64 as long as
    they contain no NaN. Never drops or changes row values (that is the validator's
    job). Raises E_NONNUMERIC.
    """
    out = df.copy()
    for c in out.columns:
        if _is_text(out[c]):
            out[c] = _strip(out[c])

    sc = params["schema"]
    for c in sc["int_columns"] + sc["float_columns"]:
        if c not in out.columns:
            continue
        s = out[c]
        if _is_text(s):
            s = s.map(lambda v: pd.NA if isinstance(v, str) and v == "" else v)
        try:
            num = pd.to_numeric(s, errors="raise")
        except (ValueError, TypeError) as e:
            raise DataLoadError("E_NONNUMERIC", f"{name}: column {c}: {e}") from e
        if c in sc["int_columns"] and num.notna().all():
            out[c] = num.astype("int64")
        else:
            out[c] = num.astype("float64")
    return out


def load_raw(source, params) -> pd.DataFrame:
    """Read a CSV, then `clean` it. `source` is "raw", "sample", or a path.

    Read errors (DataLoadError): E_FILE_MISSING, E_CSV_PARSE, E_NONNUMERIC.
    """
    path = _resolve(source, params)
    if not path.is_file():
        raise DataLoadError("E_FILE_MISSING", f"{path.name} not found")
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise DataLoadError("E_CSV_PARSE", f"{path.name}: {str(e).strip()}") from e
    return clean(df, params, name=path.name)


def to_csv_text(df: pd.DataFrame) -> str:
    return df.to_csv(index=False, float_format="%.10g", lineterminator="\n")


def data_hash(df: pd.DataFrame) -> str:
    """sha256 of the canonical content: rows sorted, float format fixed."""
    canon = df.sort_values(list(df.columns), kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(to_csv_text(canon).encode("utf-8")).hexdigest()
