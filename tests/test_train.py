"""P4: model hash, calib/test isolation from training, class weights from train only, save/load round-trip."""

import inspect
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import override
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.features.build import build_features, build_labels
from src.models import train as train_mod
from src.models.predict import predict
from src.models.train import (
    TrainError,
    class_weights,
    fit_all,
    fit_one,
    load_bundle,
    run,
    save_bundle,
    train_model,
)

pytestmark = pytest.mark.p4


@pytest.fixture(scope="module")
def xy(sample_df, base_split, params):
    X, y = build_features(sample_df, params), build_labels(sample_df, params)
    return X.loc[base_split.train], y.loc[base_split.train]


def _perturb(df, ids, seed=0):
    """Alter features and labels on rows `ids`; leave all other rows untouched."""
    out = df.copy()
    m = out["customerID"].isin(ids)
    rng = np.random.default_rng(seed)
    out.loc[m, "MonthlyCharges"] = out.loc[m, "MonthlyCharges"] * 5.0 + 11.0
    out.loc[m, "PaymentMethod"] = rng.choice(["Mailed check", "Electronic check"], m.sum())
    out.loc[m, "Churn"] = rng.choice(["Yes", "No"], m.sum())
    return out


# --- model_hash --------------------------------------------------------------
def test_model_hash_is_stable_for_same_seed(xy, fast):
    X, y = xy
    a, b = fit_one("gradient_boosting", X, y, fast), fit_one("gradient_boosting", X, y, fast)
    assert a.model_hash == b.model_hash
    assert np.array_equal(a.predict_raw(X), b.predict_raw(X))


def test_model_hash_differs_for_different_seed(xy, fast):
    X, y = xy
    assert (
        fit_one("gradient_boosting", X, y, fast, seed=1).model_hash
        != fit_one("gradient_boosting", X, y, fast, seed=2).model_hash
    )


@pytest.mark.parametrize(
    "path,value",
    [
        (("models", "class_weight"), "none"),
        (("models", "hyperparams", "gradient_boosting", "n_estimators"), 26),
        (("models", "hyperparams", "gradient_boosting", "learning_rate"), 0.05),
        (("models", "hyperparams", "gradient_boosting", "max_depth"), 2),
        (("models", "hyperparams", "logistic_regression", "C"), 0.5),
        (("models", "primary_metric"), "lift_at_k"),
        (("split", "cv_folds"), 4),
    ],
)
def test_model_hash_changes_when_one_param_changes(xy, fast, path, value):
    X, y = xy
    base = fit_one("gradient_boosting", X, y, fast).model_hash
    assert fit_one("gradient_boosting", X, y, override(fast, *path, value=value)).model_hash != base


def test_model_hash_ignores_params_that_do_not_define_the_model(xy, fast):
    """Negative control: calibration and report location are not part of the base model."""
    X, y = xy
    base = fit_one("gradient_boosting", X, y, fast).model_hash
    p = override(fast, "calibration", "method", value="sigmoid")
    p = override(p, "paths", "reports_dir", value="elsewhere")
    assert fit_one("gradient_boosting", X, y, p).model_hash == base


def test_model_hash_depends_on_train_data(xy, fast):
    X, y = xy
    flipped = y.copy()
    flipped.iloc[:20] = 1 - flipped.iloc[:20]
    assert (
        fit_one("logistic_regression", X, flipped, fast).model_hash
        != fit_one("logistic_regression", X, y, fast).model_hash
    )


def test_train_model_is_reproducible(xy, fast, assumptions):
    X, y = xy
    a, b = train_model(X, y, fast, assumptions), train_model(X, y, fast, assumptions)
    assert a.name == b.name and a.model_hash == b.model_hash and a.cv_scores == b.cv_scores


# --- calib and test rows do not influence training ---------------------------
def test_preprocessing_state_ignores_calib_and_test_rows(sample_df, fast, assumptions, base_split, base_run):
    df = _perturb(sample_df, base_split.calib + base_split.test)
    r = run(df, fast, assumptions, split=base_split)
    assert r.hashes["preprocess"] == base_run.hashes["preprocess"]
    assert r.hashes["train_data"] == base_run.hashes["train_data"]
    assert r.hashes["model"] == base_run.hashes["model"]
    pa, pb = r.bundle.model.preprocessor, base_run.bundle.model.preprocessor
    assert np.array_equal(pa.mean, pb.mean) and np.array_equal(pa.scale, pb.scale)


def test_preprocessing_state_changes_when_train_rows_change(
    sample_df, fast, assumptions, base_split, base_run
):
    """Negative control: the preprocessing state is genuinely computed from train."""
    df = _perturb(sample_df, base_split.train[:100])
    r = run(df, fast, assumptions, split=base_split)
    assert r.hashes["preprocess"] != base_run.hashes["preprocess"]


def test_test_labels_do_not_change_selected_model(sample_df, fast, assumptions, base_split, base_run):
    df = sample_df.copy()
    t = df["customerID"].isin(base_split.test)
    df.loc[t, "Churn"] = df.loc[t, "Churn"].map({"Yes": "No", "No": "Yes"})
    r = run(df, fast, assumptions, split=base_split)
    assert r.selected == base_run.selected
    assert r.cv_scores == base_run.cv_scores
    assert r.hashes["model"] == base_run.hashes["model"]
    p = base_split.test
    assert np.array_equal(
        r.bundle.model.predict_raw(build_features(df, fast).loc[p]),
        base_run.bundle.model.predict_raw(build_features(sample_df, fast).loc[p]),
    )
    # Test rows feed the metrics: the metrics change, the decisions do not.
    assert r.metrics["test"][r.selected]["pr_auc"] != base_run.metrics["test"][base_run.selected]["pr_auc"]


def test_train_labels_do_change_the_model(sample_df, fast, assumptions, base_split, base_run):
    """Negative control for the two tests above."""
    df = sample_df.copy()
    t = df["customerID"].isin(base_split.train[:150])
    df.loc[t, "Churn"] = df.loc[t, "Churn"].map({"Yes": "No", "No": "Yes"})
    assert run(df, fast, assumptions, split=base_split).hashes["model"] != base_run.hashes["model"]


def test_fit_apis_accept_only_train_data():
    assert list(inspect.signature(fit_one).parameters) == ["name", "X_train", "y_train", "params", "seed"]
    assert list(inspect.signature(fit_all).parameters)[:2] == ["X_train", "y_train"]
    assert list(inspect.signature(train_model).parameters)[:2] == ["X_train", "y_train"]


# --- resampling and class weights never touch calib/test ---------------------
def test_class_weights_and_preprocessing_see_only_train_rows(
    monkeypatch, sample_df, fast, assumptions, base_split
):
    seen_w, seen_prep = [], []
    orig_cw, orig_fit = train_mod.class_weights, train_mod.Preprocessor.fit

    def spy_cw(y, params):
        seen_w.append(set(y.index))
        return orig_cw(y, params)

    def spy_fit(X):
        seen_prep.append(set(X.index))
        return orig_fit(X)

    monkeypatch.setattr(train_mod, "class_weights", spy_cw)
    monkeypatch.setattr(train_mod.Preprocessor, "fit", staticmethod(spy_fit))
    run(sample_df, fast, assumptions, split=base_split)
    train_ids, held_out = set(base_split.train), set(base_split.calib) | set(base_split.test)
    assert seen_w and seen_prep
    for ids in seen_w + seen_prep:
        assert ids <= train_ids and not ids & held_out


def test_estimators_never_fit_on_more_rows_than_train(monkeypatch, sample_df, fast, assumptions, base_split):
    """Resampling (adding rows) or including calib/test rows would exceed the number of train rows."""
    sizes = []
    for cls in (LogisticRegression, GradientBoostingClassifier, DummyClassifier):
        orig = cls.fit

        def spy(self, X, y, *a, _orig=orig, **k):
            sizes.append(len(X))
            return _orig(self, X, y, *a, **k)

        monkeypatch.setattr(cls, "fit", spy)
    run(sample_df, fast, assumptions, split=base_split)
    n_train = len(base_split.train)
    assert max(sizes) == n_train  # the final refit uses all of train, no more
    assert all(s <= n_train for s in sizes)


def test_class_weights_are_balanced_from_given_labels_only(params):
    y = pd.Series([1] * 10 + [0] * 90)
    w = class_weights(y, params)
    assert w[y.to_numpy() == 1].sum() == pytest.approx(w[y.to_numpy() == 0].sum())
    assert w.sum() == pytest.approx(len(y))
    assert np.all(class_weights(y, override(params, "models", "class_weight", value="none")) == 1.0)
    with pytest.raises(TrainError) as e:
        class_weights(pd.Series([0] * 5), params)
    assert e.value.code == "E_TRAIN_ONE_CLASS"


def test_misaligned_xy_rejected(xy, fast):
    X, y = xy
    with pytest.raises(TrainError) as e:
        fit_one("logistic_regression", X, y.iloc[::-1], fast)
    assert e.value.code == "E_TRAIN_ALIGN"


# --- model selection ---------------------------------------------------------
def test_pool_has_baselines_and_candidates_and_selection_uses_cv(xy, fast, assumptions):
    X, y = xy
    models, cv, selected = fit_all(X, y, fast, assumptions)
    assert set(models) == set(cv) == set(fast["models"]["baselines"]) | set(fast["models"]["candidates"])
    assert selected == max(cv, key=lambda n: (cv[n], -list(cv).index(n)))
    assert cv["prior_rate"] == pytest.approx(y.mean(), abs=0.01)  # baseline: PR-AUC equals the positive rate


def test_selection_follows_cv_score_not_pool_order(xy, fast, assumptions):
    X, y = xy
    ref_models, ref_cv, ref_sel = fit_all(X, y, fast, assumptions)
    best = max(ref_cv, key=ref_cv.get)
    assert ref_sel == best
    # baseline order reversed: same CV scores, same selection
    flipped = override(fast, "models", "baselines", value=list(reversed(fast["models"]["baselines"])))
    _, cv2, sel2 = fit_all(X, y, flipped, assumptions)
    assert cv2 == ref_cv and sel2 == ref_sel
    # only the prior_rate baseline: the candidate must win on CV score, not on position
    only_prior = override(fast, "models", "baselines", value=["prior_rate"])
    _, cv3, sel3 = fit_all(X, y, only_prior, assumptions)
    assert sel3 == "gradient_boosting" and cv3["gradient_boosting"] > cv3["prior_rate"]
    # candidate and baseline swap roles: selection still follows the CV score
    swapped = override(fast, "models", "candidates", value=["logistic_regression"])
    swapped = override(swapped, "models", "baselines", value=["prior_rate", "gradient_boosting"])
    _, cv4, sel4 = fit_all(X, y, swapped, assumptions)
    assert sel4 == max(cv4, key=cv4.get) == ref_sel


def test_selected_model_beats_prior_baseline_on_test(base_run):
    t = base_run.metrics["test"]
    assert t[base_run.selected]["pr_auc"] > t["prior_rate"]["pr_auc"] + 0.1
    assert t["prior_rate"]["lift_at_k"] == pytest.approx(1.0)


def test_lift_primary_metric_works_and_needs_assumptions(xy, fast, assumptions):
    X, y = xy
    p = override(fast, "models", "primary_metric", value="lift_at_k")
    assert fit_all(X, y, p, assumptions)[2] in fast["models"]["baselines"] + fast["models"]["candidates"]
    with pytest.raises(TrainError) as e:
        fit_all(X, y, p, None)
    assert e.value.code == "E_TRAIN_ASSUMPTIONS"


def test_no_forbidden_columns_in_model_features(base_run):
    assert not {"tenure", "TotalCharges", "customerID", "Churn"} & set(base_run.bundle.model.feature_names)


# --- save and load -----------------------------------------------------------
def test_save_then_load_gives_identical_predictions(base_run, sample_df, fast, tmp_path):
    path = save_bundle(base_run.bundle, tmp_path / "sub" / "model.joblib")
    loaded = load_bundle(path)
    X = build_features(sample_df, fast)
    assert np.array_equal(loaded.model.predict_raw(X), base_run.bundle.model.predict_raw(X))
    assert np.array_equal(
        loaded.calibrator.transform(loaded.model.predict_raw(X)),
        base_run.bundle.calibrator.transform(base_run.bundle.model.predict_raw(X)),
    )
    a, b = predict(sample_df, loaded, fast), predict(sample_df, base_run.bundle, fast)
    pd.testing.assert_frame_equal(a, b)
    assert loaded.model.model_hash == base_run.bundle.model.model_hash
    assert loaded.calibrator.state_hash() == base_run.bundle.calibrator.state_hash()


# --- run: determinism and CLI ------------------------------------------------
def test_run_is_deterministic(sample_df, fast, assumptions, base_split, base_run):
    r = run(sample_df, fast, assumptions, split=base_split)
    assert r.hashes == base_run.hashes
    assert r.metrics == base_run.metrics
    pd.testing.assert_frame_equal(r.test_pred, base_run.test_pred)


def test_run_metrics_are_strict_json_and_report_calibration(base_run, assumptions):
    m = json.loads(json.dumps(base_run.metrics, allow_nan=False))
    assert m["split"] == {"train": 600, "calib": 200, "test": 200}
    assert m["p_star"] == pytest.approx(40 / (0.25 * 1200))
    assert m["business_status"] == "conditional"
    assert set(m["calibration"]) >= {"ece_raw", "ece_calibrated", "tolerance", "within_tolerance"}
    assert m["calibration"]["ece_calibrated"] <= m["calibration"]["ece_raw"]
    assert m["k"] == round(200 * assumptions["campaign_capacity"] / assumptions["population_per_period"])


def test_train_cli_runs_on_sample_and_writes_metrics(isolated_project):
    root = isolated_project
    r = subprocess.run(
        [sys.executable, "-m", "src.models.train", "--sample"], cwd=root, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "MODEL OK" in r.stdout
    m = json.loads((root / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert m["data"] == "sample" and m["selected"] in m["test"]
    assert (root / "artifacts" / "model.joblib").is_file()


def test_input_data_hash_is_in_metrics_and_independent_of_row_order(sample_df, base_run):
    from src.data.load import data_hash

    h = base_run.metrics["hashes"]["input_data"]
    assert h == data_hash(sample_df)
    assert h == data_hash(
        sample_df.sample(frac=1.0, random_state=3)
    )  # canonical content: row order has no effect
    assert len(h) == 64
    assert h != base_run.metrics["hashes"]["train_data"]  # the whole input data, not just the train rows


def test_environment_versions_recorded_in_metrics(base_run):
    from importlib.metadata import version

    env = base_run.metrics["environment"]
    assert env["python"].count(".") == 2
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy", "joblib"):
        assert env[pkg] == version(pkg)
