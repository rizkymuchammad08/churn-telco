"""Ablation of `tenure` and `TotalCharges`: model without these columns (official policy) vs with them.

Diagnostic only. The `leakage.policy` setting in params.yaml is not changed, and the official
model (`make train`) is not touched. For every seed in `ablation.seeds`, both variants use the
same split, hyperparameters, and selection procedure; the only difference is `leakage.policy`
(`exclude` vs `include_flagged`). Metrics are computed on test and calibration on calib,
exactly as in `train.run`.

Results are written to `paths.ablation` (JSON), which is the fourth numeric source for rendered documents
(see `src.reporting.render`). The numbers are not copied into hand-written documents.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys

import numpy as np

from src.config import ROOT, load_all
from src.data.load import DataLoadError, data_hash, load_raw
from src.data.validate import banner, validate
from src.features.build import FeatureError, build_features, build_labels
from src.features.leakage import LeakageError, abs_corr
from src.models.calibrate import CalibrationError
from src.models.split import SplitError, make_split
from src.models.train import TrainError, run

VARIANTS = {"without": "exclude", "with": "include_flagged"}
METRICS = ("pr_auc", "roc_auc", "lift_at_k", "brier", "ece")
LOWER_IS_BETTER = {"brier", "ece"}


def _with_policy(params: dict, policy: str) -> dict:
    p = copy.deepcopy(params)
    p["leakage"]["policy"] = policy
    return p


def _summary(values: list[float]) -> dict:
    a = np.asarray(values, dtype="float64")
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def ablate(df, params: dict, assumptions: dict) -> dict:
    """Run both variants on every seed and return a summary (JSON-ready dict)."""
    seeds = params["ablation"]["seeds"]
    per_seed: list[dict] = []
    for seed in seeds:
        split = make_split(df, params, seed=seed)  # shared by both variants
        row = {"seed": seed}
        for label, policy in VARIANTS.items():
            res = run(df, _with_policy(params, policy), assumptions, split=split, seed=seed)
            t = res.metrics["test"][res.selected]
            row[label] = {"selected": res.selected, **{m: t[m] for m in METRICS}}
        per_seed.append(row)

    summary = {}
    for m in METRICS:
        with_vals = [r["with"][m] for r in per_seed]
        without_vals = [r["without"][m] for r in per_seed]
        deltas = [a - b for a, b in zip(with_vals, without_vals, strict=True)]
        # "Better" means higher for ranking metrics, lower for error metrics.
        if m in LOWER_IS_BETTER:
            seeds_with_better = sum(d < 0 for d in deltas)
        else:
            seeds_with_better = sum(d > 0 for d in deltas)
        summary[m] = {
            "without": _summary(without_vals),
            "with": _summary(with_vals),
            "delta_with_minus_without": _summary(deltas),
            "seeds_with_better": int(seeds_with_better),
        }

    # |Pearson| correlation of the flagged columns with the label (a rough guard, not proof),
    # on the full data.
    p_inc = _with_policy(params, "include_flagged")
    corr = abs_corr(build_features(df, p_inc), build_labels(df, p_inc))
    flagged = params["leakage"]["flagged_columns"]
    n_changed = sum(r["with"]["selected"] != r["without"]["selected"] for r in per_seed)
    return {
        "data_source": params["data_source"],
        "input_data_hash": data_hash(df),
        "official_policy": params["leakage"]["policy"],
        "seeds": seeds,
        "flagged_columns": flagged,
        # The detector limit, recorded so documents can state it without reading params.yaml.
        "max_abs_corr": float(params["leakage"]["max_abs_corr"]),
        "n_seeds": len(seeds),
        "n_selection_changed": int(n_changed),
        "abs_corr_with_label": {c: float(corr[c]) for c in flagged if c in corr.index},
        "metrics_on_test": list(METRICS),
        "summary": summary,
        "per_seed": per_seed,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ablation with/without tenure and TotalCharges")
    ap.add_argument("--sample", action="store_true", help="use data/sample.csv (default: raw)")
    args = ap.parse_args(argv)
    params, assumptions = load_all()
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
        out = ablate(df, params, assumptions)
    except (FeatureError, LeakageError, SplitError, TrainError, CalibrationError) as e:
        print(f"{e.code}: {e.message}")
        return 1
    out = {"data": "sample" if args.sample else "raw", **out}
    path = ROOT / params["paths"]["ablation"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    for m in ("pr_auc", "roc_auc", "lift_at_k"):
        s = out["summary"][m]
        d = s["delta_with_minus_without"]
        print(
            f"{m}: without={s['without']['mean']:.3f} with={s['with']['mean']:.3f} "
            f"delta={d['mean']:+.3f} (sd {d['std']:.3f})"
        )
    print("ABLATION OK")
    return 0


if __name__ == "__main__":
    from src.models.ablation import main as _main

    sys.exit(_main())
