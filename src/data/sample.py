"""Seeded sample generator. The committed data/sample.csv must equal its output."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.config import ROOT, load_params
from src.data.load import load_raw, to_csv_text


def make_sample(raw: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Random subsample stratified on the label, seeded by params.seed."""
    n = params["sample"]["n_rows"]
    if n > len(raw):
        raise ValueError("sample.n_rows exceeds the number of raw rows")
    sc = params["schema"]
    out = raw.groupby(sc["label_column"], group_keys=False).sample(frac=n / len(raw), random_state=params["seed"])
    return out.sort_values(sc["id_column"]).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare sample.csv with the generator output")
    args = ap.parse_args(argv)
    params = load_params()
    text = to_csv_text(make_sample(load_raw("raw", params), params))
    path = ROOT / params["paths"]["sample"]
    if args.check:
        same = path.exists() and path.read_text(encoding="utf-8") == text
        print("sample.csv MATCHES the generator" if same else "sample.csv DIFFERS from the generator")
        return 0 if same else 1
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
