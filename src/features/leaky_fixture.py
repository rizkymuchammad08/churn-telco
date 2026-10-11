"""Negative-control fixture for the leakage detector: tests/fixtures/sample_leaky.csv.

Contains data/sample.csv plus a `leaky_feature` column = label (0/1) x a constant + small
noise, seeded from params.seed. The correlation detector must reject it; without this
column, the normal feature matrix must pass. `python -m src.features.leaky_fixture` writes
the file; `--check` compares.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.config import ROOT, load_params
from src.data.load import load_raw, to_csv_text

OUT_PATH = ROOT / "tests" / "fixtures" / "sample_leaky.csv"
LEAKY_COLUMN = "leaky_feature"
SCALE = 3.7
NOISE_SD = 0.05


def build(params: dict) -> pd.DataFrame:
    """Return the sample data with an added feature that is almost a function of the label."""
    sc = params["schema"]
    df = load_raw("sample", params)
    y = (df[sc["label_column"]] == sc["positive_label"]).to_numpy(dtype="float64")
    rng = np.random.default_rng(params["seed"])
    df[LEAKY_COLUMN] = np.round(SCALE * y + rng.normal(0.0, NOISE_SD, len(df)), 6)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = to_csv_text(build(load_params()))
    if args.check:
        same = OUT_PATH.exists() and OUT_PATH.read_text(encoding="utf-8") == text
        print(
            "sample_leaky.csv MATCHES the generator"
            if same
            else "sample_leaky.csv DIFFERS from the generator"
        )
        return 0 if same else 1
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
