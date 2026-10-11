"""Loader and validator for params.yaml and assumptions.yaml.

Strict: a missing key, an unknown key, or a wrong type raises an error. Checks that span both
YAML files live in `cross_check` (called by `load_all`).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# --- Allowed values ---------------------------------------------------------------------------

HIGHER_IS_BETTER = {"pr_auc", "lift_at_k"}
KNOWN_MODELS = {"prior_rate", "logistic_regression", "gradient_boosting"}
CLASS_WEIGHTS = {"balanced", "none"}
SPLIT_TYPES = {"random_stratified", "temporal"}

# Business numbers may only live in assumptions.yaml, never in params.yaml.
BUSINESS_KEYS = {
    "k",
    "clv",
    "cost_intervention",
    "save_rate",
    "campaign_capacity",
    "population_per_period",
    "target_churn_rate",
    "baseline_churn_rate",
}


# --- Specs: the expected shape and types of each YAML file ------------------------------------


class Opt:
    """Marks an optional key in a spec."""

    def __init__(self, spec):
        self.spec = spec


PARAMS_SPEC = {
    "seed": int,
    "data_source": str,
    "data_source_allowed": [str],
    "paths": {
        "raw": str,
        "sample": str,
        "artifacts_dir": str,
        "reports_dir": str,
        "metrics": str,
        "roi": str,
        "ablation": str,
        "monitoring": str,
    },
    "sample": {"n_rows": int},
    "ablation": {"seeds": [int]},
    "schema": {
        "id_column": str,
        "label_column": str,
        "positive_label": str,
        "negative_label": str,
        "required_columns": [str],
        "int_columns": [str],
        "float_columns": [str],
        "nullable_when_tenure_zero": str,
        # Temporal mode only: the per-row date column that make_split uses.
        "snapshot_column": Opt(str),
    },
    "temporal": bool,
    "leakage": {"flagged_columns": [str], "policy": str, "max_abs_corr": float},
    "split": {
        "type": str,
        "train_frac": float,
        "calib_frac": float,
        "test_frac": float,
        "cv_folds": int,
        "window_days": Opt(int),
        "purge_gap_days": Opt(int),
    },
    "models": {
        "primary_metric": str,
        "baselines": [str],
        "candidates": [str],
        "class_weight": str,
        "hyperparams": {
            "logistic_regression": {"C": float, "max_iter": int},
            "gradient_boosting": {"n_estimators": int, "learning_rate": float, "max_depth": int},
        },
    },
    # Light hyperparameter tuning. The rule (grids, margin, folds, metric, defaults) is hashed and the
    # hash must be stored in `rule_lock` before a run: see src/models/tuning.py.
    "tuning": {
        "enabled": bool,
        "method": str,
        "seed_offset": int,
        "margin_se": float,
        "rule_lock": str,
        "grids": {
            "logistic_regression": Opt({"C": Opt([float]), "max_iter": Opt([int])}),
            "gradient_boosting": Opt(
                {
                    "n_estimators": Opt([int]),
                    "learning_rate": Opt([float]),
                    "max_depth": Opt([int]),
                }
            ),
        },
    },
    "calibration": {"method": str},
    "evaluation": {
        "ece_bins": int,
        "calibration_tolerance": float,
        "bootstrap_resamples": int,
        "ci_level": float,
    },
    "segments": {"by": [str], "min_n": int},
    "domain": {"min_tenure_months": int},
    # Input-drift monitoring: how reports/monitoring.json is built, and its thresholds.
    "monitoring": {
        "rule_lock": str,
        "psi_warn": float,
        "psi_alert": float,
        "n_bins": int,
        "max_discrete_levels": int,
        "psi_eps": float,
        "min_rows": int,
    },
}

ASSUMPTIONS_SPEC = {
    "currency": str,
    "baseline_churn_rate": float,
    "target_churn_rate": float,
    "clv": float,
    "cost_intervention": float,
    "save_rate": float,
    "save_rate_source": str,
    "campaign_capacity": int,
    "population_per_period": int,
    # Optional (P5): fixed campaign cost per year, charged once per year. Defaults to 0.
    "operating_cost_per_year": Opt(float),
}


# --- Type checking ----------------------------------------------------------------------------


def _tname(t) -> str:
    return t.__name__ if isinstance(t, type) else str(t)


def _check_scalar(value, typ, path):
    """Check one value. Booleans are not accepted as numbers; ints are accepted as floats."""
    if typ is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif typ is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    else:
        ok = isinstance(value, typ)
    if not ok:
        raise TypeError(f"{path}: must be {_tname(typ)}, got {type(value).__name__}")


def check_types(value, spec, path="") -> None:
    """Recursively check `value` against `spec`: required keys, no extra keys, correct types."""
    if isinstance(spec, dict):
        if not isinstance(value, dict):
            raise TypeError(f"{path or '<root>'}: must be a mapping")
        for key, sub in spec.items():
            optional = isinstance(sub, Opt)
            sub = sub.spec if optional else sub
            if key not in value:
                if optional:
                    continue
                raise KeyError(f"{path + '.' if path else ''}{key}: missing key")
            check_types(value[key], sub, f"{path + '.' if path else ''}{key}")
        extra = set(value) - set(spec)
        if extra:
            raise KeyError(f"{path or '<root>'}: unknown keys {sorted(extra)}")
    elif isinstance(spec, list):
        if not isinstance(value, list):
            raise TypeError(f"{path}: must be a list")
        for i, item in enumerate(value):
            _check_scalar(item, spec[0], f"{path}[{i}]")
    else:
        _check_scalar(value, spec, path)


# --- Reading files ----------------------------------------------------------------------------


def _walk_keys(obj):
    """Yield every key in a nested dict, at any depth."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _walk_keys(v)


def _read(path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: content must be a mapping")
    return data


def _forbid_business_keys(p: dict) -> None:
    clash = BUSINESS_KEYS & set(_walk_keys(p))
    if clash:
        raise ValueError(
            f"params.yaml: business numbers belong only in assumptions.yaml, found {sorted(clash)}"
        )


# --- Value checks for params.yaml (one function per section, run in this order) ---------------


def _check_basics(p: dict) -> None:
    if p["seed"] < 0:
        raise ValueError("seed must be >= 0")
    if p["data_source"] not in p["data_source_allowed"]:
        raise ValueError("data_source must be one of data_source_allowed")


def _check_schema(p: dict) -> None:
    sc = p["schema"]
    cols = set(sc["required_columns"])

    for key in ("id_column", "label_column", "nullable_when_tenure_zero"):
        if sc[key] not in cols:
            raise ValueError(f"schema.{key} is not in required_columns")
    for key in ("int_columns", "float_columns"):
        if not set(sc[key]) <= cols:
            raise ValueError(f"schema.{key} contains columns outside required_columns")

    snapshot = sc.get("snapshot_column")
    if snapshot is not None:
        if snapshot not in cols:
            raise ValueError("schema.snapshot_column is not in required_columns")
        if snapshot in {sc["id_column"], sc["label_column"]}:
            raise ValueError("schema.snapshot_column must not equal id_column or label_column")
        if not p["temporal"]:
            raise ValueError("schema.snapshot_column is only for temporal mode (temporal: true)")

    if sc["positive_label"] == sc["negative_label"]:
        raise ValueError("positive_label and negative_label must differ")


def _check_leakage(p: dict) -> None:
    cols = set(p["schema"]["required_columns"])
    leakage = p["leakage"]
    if leakage["policy"] not in {"exclude", "include_flagged"}:
        raise ValueError("leakage.policy must be exclude or include_flagged")
    if not set(leakage["flagged_columns"]) <= cols:
        raise ValueError("leakage.flagged_columns contains columns outside required_columns")
    if not 0 < leakage["max_abs_corr"] < 1:
        raise ValueError("leakage.max_abs_corr must be in (0, 1)")


def _check_split(p: dict) -> None:
    s = p["split"]
    if s["type"] not in SPLIT_TYPES:
        raise ValueError(f"split.type must be one of {sorted(SPLIT_TYPES)}")
    for key in ("train_frac", "calib_frac", "test_frac"):
        if not 0 < s[key] < 1:
            raise ValueError(f"split.{key} must be in (0, 1)")
    if abs(s["train_frac"] + s["calib_frac"] + s["test_frac"] - 1.0) > 1e-9:
        raise ValueError("split: train_frac + calib_frac + test_frac must equal 1")
    if s["cv_folds"] < 2:
        raise ValueError("split.cv_folds must be >= 2")

    # The temporal flag and the split type must agree, and only temporal splits take day settings.
    if s["type"] == "temporal":
        if not p["temporal"]:
            raise ValueError("split.type temporal requires temporal: true")
        if "window_days" not in s or "purge_gap_days" not in s:
            raise ValueError("a temporal split requires window_days and purge_gap_days")
        if s["window_days"] <= 0 or s["purge_gap_days"] < s["window_days"]:
            raise ValueError("purge_gap_days must be >= window_days, and window_days > 0")
    else:
        if p["temporal"]:
            raise ValueError("temporal: true requires split.type temporal")
        if "window_days" in s or "purge_gap_days" in s:
            raise ValueError("window_days/purge_gap_days are only for a temporal split")


def _check_models(p: dict) -> None:
    m = p["models"]
    if m["primary_metric"] not in HIGHER_IS_BETTER:
        raise ValueError(
            f"models.primary_metric must be a 'higher is better' metric: {sorted(HIGHER_IS_BETTER)}"
        )
    for key in ("baselines", "candidates"):
        if not m[key] or not set(m[key]) <= KNOWN_MODELS:
            raise ValueError(f"models.{key} is empty or contains unknown models")
    if set(m["candidates"]) & set(m["baselines"]):
        raise ValueError("models.candidates and models.baselines must not overlap")
    if p["calibration"]["method"] not in {"isotonic", "sigmoid"}:
        raise ValueError("calibration.method must be isotonic or sigmoid")
    if m["class_weight"] not in CLASS_WEIGHTS:
        raise ValueError(f"models.class_weight must be one of {sorted(CLASS_WEIGHTS)}")

    check_hyperparams(m["hyperparams"])


def check_hyperparams(hp: dict) -> None:
    """Value checks for one full set of model hyperparameters (defaults or one tuning candidate)."""
    if hp["logistic_regression"]["C"] <= 0 or hp["logistic_regression"]["max_iter"] < 1:
        raise ValueError("hyperparams.logistic_regression: C > 0 and max_iter >= 1")
    gb = hp["gradient_boosting"]
    if gb["n_estimators"] < 1 or gb["max_depth"] < 1 or not 0 < gb["learning_rate"] <= 1:
        raise ValueError(
            "hyperparams.gradient_boosting: n_estimators and max_depth >= 1, learning_rate in (0, 1]"
        )


def _check_tuning(p: dict) -> None:
    t = p["tuning"]
    if t["method"] != "grid":
        raise ValueError("tuning.method must be grid")
    if t["seed_offset"] < 1:
        raise ValueError("tuning.seed_offset must be >= 1 (tuning folds must differ from selection folds)")
    if not 0 <= t["margin_se"] <= 10:
        raise ValueError("tuning.margin_se must be in [0, 10]")
    if not t["enabled"]:
        return
    pool = set(p["models"]["candidates"]) | set(p["models"]["baselines"])
    grids = t["grids"]
    if not any(grids.get(name) for name in pool):
        raise ValueError("tuning.enabled is true but no model in the pool has a non-empty grid")
    for name, grid in grids.items():
        if name not in pool and grid:
            raise ValueError(f"tuning.grids.{name}: model is not in models.candidates or models.baselines")
        for key, values in grid.items():
            if not values:
                raise ValueError(f"tuning.grids.{name}.{key}: empty list")
            if len(set(values)) != len(values):
                raise ValueError(f"tuning.grids.{name}.{key}: duplicate values")
            default = p["models"]["hyperparams"][name][key]
            if default not in values:
                raise ValueError(
                    f"tuning.grids.{name}.{key}: the default in models.hyperparams must be in the grid "
                    "(the default is always a candidate and the reference for the decision rule)"
                )
            for value in values:  # every candidate must be a valid hyperparameter set
                hp = {m: dict(v) for m, v in p["models"]["hyperparams"].items()}
                hp[name][key] = value
                check_hyperparams(hp)


def _check_evaluation(p: dict) -> None:
    if p["evaluation"]["ece_bins"] < 2:
        raise ValueError("evaluation.ece_bins must be >= 2")
    if not 0 < p["evaluation"]["calibration_tolerance"] < 1:
        raise ValueError("evaluation.calibration_tolerance must be in (0, 1)")
    if p["evaluation"]["bootstrap_resamples"] < 100:
        raise ValueError("evaluation.bootstrap_resamples must be >= 100")
    if not 0 < p["evaluation"]["ci_level"] < 1:
        raise ValueError("evaluation.ci_level must be in (0, 1)")

    sc = p["schema"]
    cols = set(sc["required_columns"])
    seg = p["segments"]
    if seg["min_n"] < 1:
        raise ValueError("segments.min_n must be >= 1")
    if not seg["by"] or not set(seg["by"]) <= cols - {sc["id_column"], sc["label_column"]}:
        raise ValueError(
            "segments.by is empty or contains columns outside required_columns (other than id/label)"
        )

    if p["domain"]["min_tenure_months"] < 0:
        raise ValueError("domain.min_tenure_months must be >= 0")


def _check_monitoring(p: dict) -> None:
    m = p["monitoring"]
    if not 0 < m["psi_warn"] < m["psi_alert"]:
        raise ValueError("monitoring: 0 < psi_warn < psi_alert is required")
    if m["n_bins"] < 2:
        raise ValueError("monitoring.n_bins must be >= 2")
    if m["max_discrete_levels"] < 2:
        raise ValueError("monitoring.max_discrete_levels must be >= 2")
    if not 0 < m["psi_eps"] < 1:
        raise ValueError("monitoring.psi_eps must be in (0, 1)")
    if m["min_rows"] < m["n_bins"]:
        raise ValueError("monitoring.min_rows must be >= monitoring.n_bins")


def _check_params_values(p: dict) -> None:
    _check_basics(p)
    _check_schema(p)
    _check_leakage(p)
    _check_split(p)
    _check_models(p)
    _check_tuning(p)
    _check_evaluation(p)
    _check_monitoring(p)


# --- Business logic and value checks for assumptions.yaml -------------------------------------


def breakeven_threshold(a: dict) -> float:
    """p* = cost_intervention / (save_rate * clv): the minimum churn probability worth contacting."""
    return a["cost_intervention"] / (a["save_rate"] * a["clv"])


def _check_assumptions_values(a: dict) -> None:
    if a["save_rate_source"] not in {"assumed", "measured"}:
        raise ValueError("save_rate_source must be assumed or measured")
    if not 0 < a["save_rate"] <= 1:
        raise ValueError("save_rate must be in (0, 1]")
    for key in ("clv", "cost_intervention", "campaign_capacity", "population_per_period"):
        if not a[key] > 0:
            raise ValueError(f"{key} must be positive")
    if not 0 < a["target_churn_rate"] < a["baseline_churn_rate"] < 1:
        raise ValueError("must satisfy 0 < target_churn_rate < baseline_churn_rate < 1")
    if a.get("operating_cost_per_year", 0) < 0:
        raise ValueError("operating_cost_per_year must not be negative")
    if a["campaign_capacity"] > a["population_per_period"]:
        raise ValueError("campaign_capacity must not exceed population_per_period")
    if not 0 < breakeven_threshold(a) < 1:
        raise ValueError("p* = cost_intervention / (save_rate * clv) must be in (0, 1)")


# --- Public loaders ---------------------------------------------------------------------------


def load_params(path=None) -> dict:
    p = _read(path or ROOT / "params.yaml")
    _forbid_business_keys(p)
    check_types(p, PARAMS_SPEC)
    _check_params_values(p)
    return p


def load_assumptions(path=None) -> dict:
    a = _read(path or ROOT / "assumptions.yaml")
    check_types(a, ASSUMPTIONS_SPEC)
    _check_assumptions_values(a)
    return a


def cross_check(params: dict, assumptions: dict) -> None:
    """Consistency between the two YAML files."""
    n_rows = params["sample"]["n_rows"]
    if n_rows > assumptions["population_per_period"]:
        raise ValueError("sample.n_rows must not exceed population_per_period")
    if n_rows < params["split"]["cv_folds"] * 2:
        raise ValueError("sample.n_rows is too small for cv_folds")


def load_all(params_path=None, assumptions_path=None) -> tuple[dict, dict]:
    """Load and validate both files, then check them against each other."""
    p, a = load_params(params_path), load_assumptions(assumptions_path)
    cross_check(p, a)
    return p, a


def business_status(assumptions: dict) -> str:
    """'conditional' while save_rate_source = assumed (rule P0), otherwise 'measured'."""
    return "conditional" if assumptions["save_rate_source"] == "assumed" else "measured"


def log_level() -> str:
    """The only setting read from .env; it does not affect results."""
    return os.environ.get("LOG_LEVEL", "INFO").upper()
