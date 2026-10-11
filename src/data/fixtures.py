"""Broken-data fixtures: one file per failure mode. Built from sample.csv.

`python -m src.data.fixtures` writes tests/fixtures/broken/; `--check` compares.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import yaml

from src.config import ROOT, load_params

OUT_DIR = ROOT / "tests" / "fixtures" / "broken"
N_BASE = 12  # number of sample rows used as the base for every fixture


def _csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False, lineterminator="\n")


def _positive_tenure_row(df: pd.DataFrame) -> int:
    """Index of the first row whose tenure is greater than 0."""
    return int(df.index[df["tenure"].astype(int) > 0][0])


def build(params: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Return (filename -> CSV text, filename -> expected error code)."""
    sc = params["schema"]
    base = pd.read_csv(ROOT / params["paths"]["sample"], dtype=str, keep_default_na=False).head(N_BASE)
    idc, lab = sc["id_column"], sc["label_column"]
    i = _positive_tenure_row(base)

    def edit(**cells) -> str:
        """Copy the base table and overwrite cells. Keys have the form `<column>__<row>`."""
        df = base.copy()
        for key, value in cells.items():
            col, row = key.rsplit("__", 1)
            df.loc[int(row), col] = value
        return _csv(df)

    # A row with one more field than the header, to trigger a CSV parse error.
    ragged = _csv(base).splitlines()
    ragged[3] += ",extra_column"

    # filename -> (CSV text, expected error code)
    files = {
        "empty_rows.csv": (_csv(base.iloc[0:0]), "E_EMPTY"),
        "missing_column.csv": (_csv(base.drop(columns=["Contract"])), "E_COLUMNS_MISSING"),
        "duplicate_id.csv": (edit(**{f"{idc}__1": base.loc[0, idc]}), "E_DUP_ID"),
        "null_id.csv": (edit(**{f"{idc}__0": ""}), "E_NULL_ID"),
        "bad_label.csv": (edit(**{f"{lab}__0": "Maybe"}), "E_LABEL_DOMAIN"),
        "null_category.csv": (edit(Contract__0=""), "E_NULL_CATEGORY"),
        "null_tenure.csv": (edit(tenure__0=""), "E_NULL_NUMERIC"),
        "null_totalcharges_with_tenure.csv": (edit(**{f"TotalCharges__{i}": ""}), "E_NULL_NUMERIC"),
        "negative_monthly.csv": (edit(MonthlyCharges__0="-20.5"), "E_NEGATIVE_NUMERIC"),
        "nonnumeric_totalcharges.csv": (edit(TotalCharges__0="n/a"), "E_NONNUMERIC"),
        "ragged_row.csv": ("\n".join(ragged) + "\n", "E_CSV_PARSE"),
    }
    texts = {name: text for name, (text, _) in files.items()}
    expected = {name: code for name, (_, code) in files.items()}
    return texts, expected


def expected_text(expected: dict[str, str]) -> str:
    return yaml.safe_dump(dict(sorted(expected.items())), sort_keys=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    texts, expected = build(load_params())
    texts["expected.yaml"] = expected_text(expected)
    if args.check:
        on_disk = {p.name: p.read_text(encoding="utf-8") for p in OUT_DIR.glob("*") if p.is_file()}
        same = on_disk == texts
        print("fixtures MATCH the generator" if same else "fixtures DIFFER from the generator")
        return 0 if same else 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*"):
        old.unlink()
    for name, text in texts.items():
        (OUT_DIR / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(texts)} files to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
