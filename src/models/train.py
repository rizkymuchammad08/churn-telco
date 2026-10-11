"""Training, model selection, calibration, and one-shot test evaluation.

Rules enforced here:
- Preprocessing, class weights, and model selection (CV) only see train rows.
- The calibrator only sees raw scores and labels from the calibration set.
- Test rows are used only at the end of `run`, to compute metrics. No decision
  (model choice, calibrator) depends on them.
No resampling is used; class imbalance is handled via `models.class_weight`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.config import ROOT, breakeven_threshold, business_status, load_all
from src.data.load import DataLoadError, data_hash, load_raw, to_csv_text
from src.data.validate import banner, validate
from src.features.build import FeatureError, build_features, build_labels, feature_names
from src.features.leakage import LeakageError, assert_no_forbidden, check_correlation
from src.models import evaluate as ev
from src.models import monitoring, tuning
from src.models.calibrate import CalibrationError, Calibrator, fit_calibrator
from src.models.monitoring import MonitorError
from src.models.split import Split, SplitError, make_split
from src.models.targeting import capacity_for


class TrainError(ValueError):
    """Training could not be performed. `code` is stable for tests and the CLI."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _digest(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --- preprocessing: state is learned from train only -------------------------
@dataclass
class Preprocessor:
    columns: list[str]
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: pd.DataFrame) -> Preprocessor:
        if len(X) == 0:
            raise TrainError("E_TRAIN_EMPTY", "train data is empty")
        v = X.to_numpy(dtype="float64")
        sd = v.std(axis=0)
        # Constant columns get scale 1 to avoid division by zero.
        return cls(list(map(str, X.columns)), v.mean(axis=0), np.where(sd > 0, sd, 1.0))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if list(map(str, X.columns)) != self.columns:
            raise TrainError("E_TRAIN_COLUMNS", "feature columns differ from those used in training")
        return (X.to_numpy(dtype="float64") - self.mean) / self.scale

    def state(self) -> dict:
        return {"columns": self.columns, "mean": self.mean.tolist(), "scale": self.scale.tolist()}

    def state_hash(self) -> str:
        return _digest(self.state())


# --- class weights: computed only from the labels passed in (train) ----------
def class_weights(y: pd.Series, params: dict) -> np.ndarray:
    """Per-row weights from `y` (always the train labels or a train fold). 'none' -> all ones."""
    arr = np.asarray(y, dtype="int64")
    if params["models"]["class_weight"] == "none":
        return np.ones(len(arr))
    counts = np.bincount(arr, minlength=2)
    if counts.min() == 0:
        raise TrainError("E_TRAIN_ONE_CLASS", "train labels contain only one class")
    return len(arr) / (2.0 * counts[arr])


def _estimator(name: str, params: dict, seed: int):
    hp = params["models"]["hyperparams"]
    if name == "prior_rate":
        return DummyClassifier(strategy="prior")
    if name == "logistic_regression":
        return LogisticRegression(C=hp[name]["C"], max_iter=hp[name]["max_iter"])
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed, **hp[name])
    raise TrainError("E_TRAIN_MODEL", f"unknown model: {name}")


@dataclass
class TrainedModel:
    name: str
    preprocessor: Preprocessor
    estimator: object
    feature_names: list[str]
    seed: int
    model_hash: str
    train_hash: str
    cv_scores: dict = field(default_factory=dict)

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw (uncalibrated) P(label = 1)."""
        if len(X) == 0:
            return np.empty(0)
        return self.estimator.predict_proba(self.preprocessor.transform(X))[:, 1]


@dataclass
class Bundle:
    """The selected model plus its calibrator. The only object that is saved and loaded."""

    model: TrainedModel
    calibrator: Calibrator
    meta: dict = field(default_factory=dict)


def _check_xy(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    if len(X) != len(y) or not X.index.equals(y.index):
        raise TrainError("E_TRAIN_ALIGN", "X and y must share the same index")
    if not set(np.unique(y)) <= {0, 1}:
        raise TrainError("E_TRAIN_LABEL", "labels must be binary 0/1")
    return y


def _fit_parts(name: str, X: pd.DataFrame, y: pd.Series, params: dict, seed: int):
    prep = Preprocessor.fit(X)
    est = _estimator(name, params, seed)
    if name == "prior_rate":
        est.fit(prep.transform(X), np.asarray(y))
    else:
        est.fit(prep.transform(X), np.asarray(y), sample_weight=class_weights(y, params))
    return prep, est


def train_data_hash(X: pd.DataFrame, y: pd.Series) -> str:
    return data_hash(X.assign(label=np.asarray(y)).reset_index())


def compute_model_hash(name: str, params: dict, seed: int, train_hash: str) -> str:
    """Hash of everything that determines the model: name, seed, model parameters, train data."""
    return _digest(
        {
            "name": name,
            "seed": seed,
            "models": params["models"],
            "leakage": params["leakage"],
            "cv_folds": params["split"]["cv_folds"],
            # The locked tuning rule is part of what produced the hyperparameters (None when tuning is off).
            # `rule_lock` equals the rule hash, verified by `tuning.tune` before any model is fitted.
            "tuning_rule": params["tuning"]["rule_lock"] if tuning.enabled(params) else None,
            "train": train_hash,
        }
    )


def fit_one(name: str, X_train: pd.DataFrame, y_train: pd.Series, params: dict, seed: int | None = None):
    """Train a single model on train data. Accepts no calibration or test data."""
    seed = params["seed"] if seed is None else seed
    y = _check_xy(X_train, y_train)
    prep, est = _fit_parts(name, X_train, y, params, seed)
    th = train_data_hash(X_train, y)
    return TrainedModel(
        name=name,
        preprocessor=prep,
        estimator=est,
        feature_names=list(map(str, X_train.columns)),
        seed=seed,
        model_hash=compute_model_hash(name, params, seed, th),
        train_hash=th,
    )


def _pool(params: dict) -> list[str]:
    m = params["models"]
    return [*m["candidates"], *m["baselines"]]  # order = priority when scores tie


def cv_fold_scores(
    name: str, X: pd.DataFrame, y: pd.Series, params: dict, assumptions: dict | None, seed: int
) -> list[float]:
    """Primary metric on each CV fold inside the train set (folds and model both seeded with `seed`)."""
    metric = params["models"]["primary_metric"]
    if metric == "lift_at_k" and assumptions is None:
        raise TrainError("E_TRAIN_ASSUMPTIONS", "lift_at_k requires assumptions to derive k")
    skf = StratifiedKFold(n_splits=params["split"]["cv_folds"], shuffle=True, random_state=seed)
    scores = []
    for tr, va in skf.split(X, y):
        prep, est = _fit_parts(name, X.iloc[tr], y.iloc[tr], params, seed)
        p = est.predict_proba(prep.transform(X.iloc[va]))[:, 1]
        k = max(1, capacity_for(len(va), assumptions)) if assumptions else 1
        scores.append(ev.metric_value(metric, y.iloc[va], p, k))
    return scores


def cv_score(name: str, X: pd.DataFrame, y: pd.Series, params: dict, assumptions: dict | None, seed: int):
    """Mean primary metric across CV folds inside the train set."""
    return float(np.mean(cv_fold_scores(name, X, y, params, assumptions, seed)))


def fit_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    assumptions: dict | None = None,
    seed: int | None = None,
) -> tuple[dict[str, TrainedModel], dict[str, float], str]:
    """Train all models (candidates + baselines) and pick one via CV on the train set.

    Returns (models, cv_scores, selected). Ties are broken by order: candidates first, then baselines.
    """
    seed = params["seed"] if seed is None else seed
    y = _check_xy(X_train, y_train)
    names = _pool(params)
    cv = {n: cv_score(n, X_train, y, params, assumptions, seed) for n in names}
    selected = max(names, key=lambda n: (cv[n], -names.index(n)))
    models = {n: fit_one(n, X_train, y, params, seed) for n in names}
    for m in models.values():
        m.cv_scores = dict(cv)
    return models, cv, selected


def train_model(X_train, y_train, params, assumptions=None, seed=None) -> TrainedModel:
    """Return only the selected model (chosen by CV on the train set)."""
    models, _, selected = fit_all(X_train, y_train, params, assumptions, seed)
    return models[selected]


# --- save / load -------------------------------------------------------------
def save_bundle(bundle: Bundle, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path) -> Bundle:
    """Load only artifacts you created yourself (joblib/pickle is unsafe for foreign files)."""
    return joblib.load(Path(path))


# --- pipeline ----------------------------------------------------------------
@dataclass
class RunResult:
    split: Split
    bundle: Bundle
    selected: str
    cv_scores: dict
    hashes: dict
    metrics: dict
    test_pred: pd.DataFrame  # indexed by id; columns: label, p_raw, p_churn (test rows only)
    monitoring: dict = field(default_factory=dict)  # reference distribution of the train rows, per feature


def run(
    df: pd.DataFrame,
    params: dict,
    assumptions: dict,
    split: Split | None = None,
    seed: int | None = None,
) -> RunResult:
    """Run the full pipeline on raw `df`. `split` may be supplied (for tests and experiments)."""
    seed = params["seed"] if seed is None else seed
    sc = params["schema"]
    X, y = build_features(df, params), build_labels(df, params)
    assert_no_forbidden(X, params)
    monitoring.verify_lock(params)  # stop before fitting anything if the monitoring rule changed unlocked
    split = split or make_split(df, params, seed=seed)

    # 1. Tune hyperparameters (locked rule, CV on train rows only), then train all models and select
    #    one via CV (train rows only). With tuning disabled `eff_params` is `params`.
    X_tr, y_tr = X.loc[split.train], y.loc[split.train]
    eff_params, tuning_report = tuning.tune(
        params, lambda name, p, fold_seed: cv_fold_scores(name, X_tr, y_tr, p, assumptions, fold_seed), seed
    )
    models, cv, selected = fit_all(X_tr, y_tr, eff_params, assumptions, seed)

    # 2. Fit one calibrator per model (calibration rows only).
    calibrators = {}
    for name, m in models.items():
        calibrators[name] = fit_calibrator(m.predict_raw(X.loc[split.calib]), y.loc[split.calib], params)

    # 3. Evaluate on test: used here once, for metrics only.
    Xt, yt = X.loc[split.test], y.loc[split.test]
    k = min(len(yt), max(1, capacity_for(len(yt), assumptions)))
    seg = df.set_index(sc["id_column"]).loc[split.test, params["segments"]["by"]]
    test_metrics, raw_by_model = {}, {}
    for name, m in models.items():
        p_raw = m.predict_raw(Xt)
        p_cal = calibrators[name].transform(p_raw)
        raw_by_model[name] = (p_raw, p_cal)
        test_metrics[name] = ev.evaluate(
            yt, p_cal, params, k=k, segments=seg if name == selected else None, ci_seed=seed
        )

    p_raw, p_cal = raw_by_model[selected]
    ece_raw = ev.expected_calibration_error(yt, p_raw, params["evaluation"]["ece_bins"])
    ece_cal = test_metrics[selected]["ece"]
    tol = params["evaluation"]["calibration_tolerance"]

    chosen = models[selected]
    bundle = Bundle(
        model=chosen,
        calibrator=calibrators[selected],
        meta={
            "data_source": params["data_source"],
            "seed": seed,
            "calibration_method": params["calibration"]["method"],
            "hyperparams": eff_params["models"]["hyperparams"].get(selected),
        },
    )
    hashes = {
        # sha256 of the entire input data (after clean; canonical content, rows sorted), not just train rows.
        "input_data": data_hash(df),
        "train_data": chosen.train_hash,
        "preprocess": chosen.preprocessor.state_hash(),
        "model": chosen.model_hash,
        "calibrator": calibrators[selected].state_hash(),
    }
    metrics = {
        "data_source": params["data_source"],
        "business_status": business_status(assumptions),
        "seed": seed,
        "split": split.sizes(),
        "split_type": params["split"]["type"],
        "k": k,
        "p_star": breakeven_threshold(assumptions),
        "primary_metric": params["models"]["primary_metric"],
        "selected": selected,
        "cv": cv,
        "tuning": tuning_report,
        "test": test_metrics,
        "calibration": {
            "method": params["calibration"]["method"],
            "ece_raw": ece_raw,
            "ece_calibrated": ece_cal,
            "tolerance": tol,
            "within_tolerance": bool(ece_cal <= tol),
        },
        "hashes": hashes,
        "environment": environment(),
    }
    test_pred = pd.DataFrame({"label": yt.to_numpy(), "p_raw": p_raw, "p_churn": p_cal}, index=Xt.index)
    # Drift reference: train rows only, the same rows the selected model was fitted on.
    reference = monitoring.build_reference(X_tr, params, train_hash=chosen.train_hash)
    return RunResult(split, bundle, selected, cv, hashes, metrics, test_pred, reference)


# Libraries whose versions can change model numbers; recorded in metrics.json next to the results.
ENVIRONMENT_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "joblib")


def environment() -> dict:
    """Python and library versions that produced these results (strings; not numeric sources)."""
    return {"python": platform.python_version(), **{p: version(p) for p in ENVIRONMENT_PACKAGES}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train, calibrate, and evaluate the churn model")
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
        check_correlation(build_features(df, params), build_labels(df, params), params)
        res = run(df, params, assumptions)
        mon = monitoring.seal({"data": "sample" if args.sample else "raw", **res.monitoring})
        monitoring.check_alignment(mon, params)  # every model feature has thresholds, none extra
        monitoring.check_model(mon, res.bundle)
    except (FeatureError, LeakageError, SplitError, TrainError, CalibrationError, MonitorError) as e:
        print(f"{e.code}: {e.message}")
        return 1

    out = ROOT / params["paths"]["artifacts_dir"]
    save_bundle(res.bundle, out / "model.joblib")
    (out / "test_predictions.csv").write_text(
        to_csv_text(res.test_pred.reset_index()), encoding="utf-8", newline="\n"
    )
    mpath = ROOT / params["paths"]["metrics"]
    mpath.parent.mkdir(parents=True, exist_ok=True)
    payload = {"data": "sample" if args.sample else "raw", **res.metrics}
    mpath.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    monitoring.write(mon, ROOT / params["paths"]["monitoring"])

    t, cal = res.metrics["test"][res.selected], res.metrics["calibration"]
    print(
        f"split={res.split.sizes()} features={len(feature_names(params))} selected={res.selected} "
        f"cv={ {n: round(v, 3) for n, v in res.cv_scores.items()} }"
    )
    print(
        f"test: pr_auc={t['pr_auc']:.3f} roc_auc={t['roc_auc']:.3f} lift@{t['k']}={t['lift_at_k']:.2f} "
        f"brier={t['brier']:.3f} ece {cal['ece_raw']:.3f} -> {cal['ece_calibrated']:.3f} "
        f"(tolerance {cal['tolerance']}: {'within' if cal['within_tolerance'] else 'OUTSIDE'})"
    )
    ci = t["ci"]
    print(
        f"{ci['level_pct']}% CI (bootstrap, {ci['n_resamples']} resamples): "
        f"pr_auc [{ci['pr_auc']['lo']:.3f}, {ci['pr_auc']['hi']:.3f}] "
        f"roc_auc [{ci['roc_auc']['lo']:.3f}, {ci['roc_auc']['hi']:.3f}] "
        f"lift@{t['k']} [{ci['lift_at_k']['lo']:.2f}, {ci['lift_at_k']['hi']:.2f}]"
    )
    print(f"model_hash={res.hashes['model'][:16]} calibrator_hash={res.hashes['calibrator'][:16]}")
    print(f"monitoring: {len(mon['features'])} features with thresholds -> {params['paths']['monitoring']}")
    print("MODEL OK")
    return 0


if __name__ == "__main__":
    # Re-import as a module: pickled classes (Bundle etc.) must be named src.models.train.*,
    # not __main__.*, so the bundle can be loaded by other processes (dashboard, predict).
    from src.models.train import main as _main

    sys.exit(_main())
