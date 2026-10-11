"""Light hyperparameter tuning with a decision rule that is locked before it runs.

What it does
------------
For every model in the pool that has a grid under `tuning.grids`, each grid candidate is scored by
cross-validation on the **train rows only**, using `models.primary_metric`. The test set is never
used, and neither are the calibration rows.

The decision rule (fixed in advance, nothing here depends on results)
---------------------------------------------------------------------
1. The default (`models.hyperparams`) is always one of the candidates and is the reference.
2. The best candidate is the one with the highest mean CV score; an exact tie goes to the earlier
   candidate in grid order (list simpler or more regularized values first).
3. The best candidate replaces the default only if its mean paired gain over the default, across
   folds, is positive and larger than `tuning.margin_se` standard errors of that paired gain.
   Otherwise the default is kept.
4. Tuning folds use `seed + tuning.seed_offset`; the model-selection CV in `train.fit_all` keeps
   `seed`. The scores that select the model are therefore not computed on the folds that picked the
   hyperparameters.

The lock
--------
`rule_hash(params)` is the sha256 of everything that determines the outcome: grids, defaults,
margin, offset, metric, folds, class weighting. `tuning.rule_lock` in params.yaml must equal it
whenever tuning is enabled, otherwise the run stops (`E_TUNE_LOCK`). The lock is written with
`python -m src.models.tuning --write-lock` and committed before `make train` is run, so a rule
changed after seeing results is visible as a changed lock. The rule hash also enters
`train.compute_model_hash`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import re
import sys
from collections.abc import Callable

import numpy as np

from src.config import ROOT, load_params

RULE_VERSION = 1

# Called as score_folds(model_name, params_with_candidate_hyperparams, fold_seed) -> list of fold scores.
ScoreFolds = Callable[[str, dict, int], list]


class TuneError(ValueError):
    """Tuning could not run. `code` is stable for tests and the CLI."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def enabled(params: dict) -> bool:
    return bool(params["tuning"]["enabled"])


def _grids(params: dict) -> dict:
    return {n: g for n, g in params["tuning"]["grids"].items() if g}


def rule_spec(params: dict) -> dict:
    """Everything that determines the tuning outcome, in a canonical form."""
    t = params["tuning"]
    return {
        "version": RULE_VERSION,
        "method": t["method"],
        "seed_offset": t["seed_offset"],
        "margin_se": t["margin_se"],
        "metric": params["models"]["primary_metric"],
        "cv_folds": params["split"]["cv_folds"],
        "class_weight": params["models"]["class_weight"],
        "defaults": params["models"]["hyperparams"],
        "grids": _grids(params),
    }


def rule_hash(params: dict) -> str:
    blob = json.dumps(rule_spec(params), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def verify_lock(params: dict) -> str:
    """Stop unless `tuning.rule_lock` matches the rule actually in force. Returns the rule hash."""
    actual = rule_hash(params)
    if params["tuning"]["rule_lock"] != actual:
        raise TuneError(
            "E_TUNE_LOCK",
            "tuning.rule_lock does not match the tuning rule in params.yaml "
            "(grids, defaults, margin, seed_offset, metric, cv_folds, or class_weight changed). "
            "If the change is intended and was decided before looking at results, run "
            "`python -m src.models.tuning --write-lock` and commit it before training.",
        )
    return actual


def candidates(name: str, params: dict) -> list[dict]:
    """Grid candidates for one model, as dicts over the grid keys only, in grid order."""
    grid = params["tuning"]["grids"].get(name) or {}
    keys = list(grid)
    if not keys:
        return []
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(grid[k] for k in keys))]


def _default_for(name: str, params: dict) -> dict:
    grid = params["tuning"]["grids"][name]
    return {k: params["models"]["hyperparams"][name][k] for k in grid}


def _with_config(params: dict, name: str, config: dict) -> dict:
    p = copy.deepcopy(params)
    p["models"]["hyperparams"][name].update(config)
    return p


def _decide(scores: list[list[float]], default_idx: int, margin_se: float) -> dict:
    """Apply the decision rule to per-candidate fold scores. Pure function of its inputs."""
    arr = np.asarray(scores, dtype="float64")  # candidates x folds
    means = arr.mean(axis=1)
    winner = int(np.argmax(means))  # argmax returns the first maximum: ties go to the earlier candidate
    diffs = arr[winner] - arr[default_idx]
    n = len(diffs)
    gain = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    margin = float(margin_se * se)
    replaced = bool(winner != default_idx and gain > 0 and gain > margin)
    return {
        "means": [float(m) for m in means],
        "winner_index": winner,
        "gain": gain,
        "se": se,
        "margin": margin,
        "replaced": replaced,
    }


def tune(params: dict, score_folds: ScoreFolds, seed: int) -> tuple[dict, dict]:
    """Return (effective_params, report). With tuning disabled the params are returned unchanged."""
    if not enabled(params):
        return params, {"enabled": False}
    lock = verify_lock(params)
    t = params["tuning"]
    fold_seed = seed + t["seed_offset"]
    effective = copy.deepcopy(params)
    models: dict = {}
    pool = [*params["models"]["candidates"], *params["models"]["baselines"]]
    for name in pool:
        if not params["tuning"]["grids"].get(name):
            continue
        cands = candidates(name, params)
        default = _default_for(name, params)
        default_idx = cands.index(default)
        scores = [score_folds(name, _with_config(params, name, c), fold_seed) for c in cands]
        d = _decide(scores, default_idx, t["margin_se"])
        chosen = cands[d["winner_index"]] if d["replaced"] else default
        effective["models"]["hyperparams"][name].update(chosen)
        models[name] = {
            "default": default,
            "chosen": chosen,
            "replaced": d["replaced"],
            "winner": cands[d["winner_index"]],
            "gain_winner_vs_default": d["gain"],
            "se_of_gain": d["se"],
            "required_gain": d["margin"],
            "candidates": [
                {"config": c, "cv_mean": m, "is_default": i == default_idx}
                for i, (c, m) in enumerate(zip(cands, d["means"], strict=True))
            ],
        }
    report = {
        "enabled": True,
        "rule_hash": lock,
        "method": t["method"],
        "margin_se": t["margin_se"],
        "seed_offset": t["seed_offset"],
        "folds": params["split"]["cv_folds"],
        "metric": params["models"]["primary_metric"],
        "models": models,
    }
    return effective, report


# --- CLI: print or write the lock ---------------------------------------------------------------
_LOCK_LINE = re.compile(r'^(\s*rule_lock:\s*)"[^"]*"(.*)$', re.MULTILINE)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Show, check, or write the tuning rule lock.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="exit 1 if tuning.rule_lock is stale")
    g.add_argument("--write-lock", action="store_true", help="write the current rule hash into params.yaml")
    args = ap.parse_args(argv)
    params = load_params()
    h = rule_hash(params)
    path = ROOT / "params.yaml"
    if args.write_lock:
        text = path.read_text(encoding="utf-8")
        new, n = _LOCK_LINE.subn(lambda m: f'{m.group(1)}"{h}"{m.group(2)}', text, count=1)
        if n != 1:
            print("tuning: could not find a `rule_lock: \"...\"` line in params.yaml", file=sys.stderr)
            return 2
        path.write_text(new, encoding="utf-8")
        print(f"tuning: rule_lock written: {h}")
        return 0
    print(f"rule_hash: {h}")
    print(f"rule_lock: {params['tuning']['rule_lock']}")
    if args.check and enabled(params) and params["tuning"]["rule_lock"] != h:
        print("tuning: lock is stale (E_TUNE_LOCK)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
