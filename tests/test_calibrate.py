"""P4: monotone calibrator, depends only on calib data, improves ECE, deterministic."""

import inspect

import numpy as np
import pytest
from conftest import override

from src.models.calibrate import CalibrationError, fit_calibrator
from src.models.evaluate import expected_calibration_error
from src.models.train import run

pytestmark = pytest.mark.p4

METHODS = ["isotonic", "sigmoid"]


def _world(n, seed, a=1.8, b=0.9):
    """Known p_true; raw score = monotone distortion (overconfident and shifted)."""
    rng = np.random.default_rng(seed)
    p_true = rng.beta(2, 5, n)
    y = (rng.random(n) < p_true).astype(int)
    logit = np.log(p_true / (1 - p_true))
    return y, p_true, 1 / (1 + np.exp(-(a * logit + b)))


def _with_method(params, method):
    return override(params, "calibration", "method", value=method)


# --- monotone and within [0, 1] ----------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_mapping_is_monotone_non_decreasing_and_in_unit_interval(params, method):
    y, _, raw = _world(5000, seed=1)
    cal = fit_calibrator(raw, y, _with_method(params, method))
    grid = np.linspace(0.0, 1.0, 2001)
    out = cal.transform(grid)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(np.diff(out) >= -1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_ranking_of_raw_scores_is_preserved(params, method):
    y, _, raw = _world(5000, seed=2)
    cal = fit_calibrator(raw, y, _with_method(params, method))
    order = np.argsort(raw, kind="stable")
    assert np.all(np.diff(cal.transform(raw)[order]) >= -1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_scores_outside_calib_range_stay_in_unit_interval(params, method):
    y, _, raw = _world(2000, seed=3)
    cal = fit_calibrator(raw, y, _with_method(params, method))
    out = cal.transform(np.array([0.0, 1e-9, 0.999999, 1.0]))
    assert np.all((out >= 0) & (out <= 1)) and np.all(np.diff(out) >= 0)


def test_isotonic_does_not_collapse_everything_to_a_constant(params):
    """Negative control for the monotone tests: a constant mapping is also monotone, so it must vary."""
    y, _, raw = _world(5000, seed=4)
    out = fit_calibrator(raw, y, params).transform(np.linspace(0.05, 0.95, 50))
    assert out.max() - out.min() > 0.2


# --- depends only on calib data ----------------------------------------------
def test_fit_signature_accepts_only_calib_data():
    assert list(inspect.signature(fit_calibrator).parameters) == ["p_raw_calib", "y_calib", "params"]


def test_pipeline_calibrator_ignores_test_rows(sample_df, fast, assumptions, base_split, base_run):
    """Alter features and labels of the test rows: the calibrator does not change."""
    df = sample_df.copy()
    test = df["customerID"].isin(base_split.test)
    rng = np.random.default_rng(0)
    df.loc[test, "MonthlyCharges"] = df.loc[test, "MonthlyCharges"] * 4.0
    df.loc[test, "Contract"] = rng.choice(["Month-to-month", "One year", "Two year"], test.sum())
    df.loc[test, "Churn"] = rng.choice(["Yes", "No"], test.sum())
    r = run(df, fast, assumptions, split=base_split)
    assert r.hashes["calibrator"] == base_run.hashes["calibrator"]
    assert r.bundle.calibrator.state() == base_run.bundle.calibrator.state()


def test_pipeline_calibrator_changes_when_calib_labels_change(
    sample_df, fast, assumptions, base_split, base_run
):
    """Negative control: without it, the test above also passes for a calibrator that sees no data."""
    df = sample_df.copy()
    calib = df["customerID"].isin(base_split.calib)
    flipped = df.loc[calib, "Churn"].map({"Yes": "No", "No": "Yes"})
    df.loc[calib, "Churn"] = flipped
    r = run(df, fast, assumptions, split=base_split)
    assert r.hashes["calibrator"] != base_run.hashes["calibrator"]
    assert (
        r.hashes["model"] == base_run.hashes["model"]
    )  # the base model stays the same: only the calibrator shifts


@pytest.mark.parametrize("method", METHODS)
def test_changing_calib_labels_changes_calibrator(params, method):
    y, _, raw = _world(3000, seed=5)
    p = _with_method(params, method)
    base = fit_calibrator(raw, y, p)
    y2 = y.copy()
    y2[:300] = 1 - y2[:300]
    assert fit_calibrator(raw, y2, p).state_hash() != base.state_hash()


@pytest.mark.parametrize("method", METHODS)
def test_calibrator_is_a_pure_function_of_calib_arrays(params, method):
    y, _, raw = _world(3000, seed=6)
    p = _with_method(params, method)
    a = fit_calibrator(raw, y, p)
    # other data (e.g. test scores) is only processed via transform and does not change the state
    h = a.state_hash()
    a.transform(np.random.default_rng(0).random(500))
    assert a.state_hash() == h


# --- ECE on synthetic data with p_true ---------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_ece_after_calibration_is_not_worse_and_within_tolerance(params, method):
    p = _with_method(params, method)
    tol = p["evaluation"]["calibration_tolerance"]
    bins = p["evaluation"]["ece_bins"]
    y_c, _, raw_c = _world(20000, seed=10)
    y_t, _, raw_t = _world(20000, seed=11)
    cal = fit_calibrator(raw_c, y_c, p)
    before = expected_calibration_error(y_t, raw_t, bins)
    after = expected_calibration_error(y_t, cal.transform(raw_t), bins)
    assert before > tol  # the raw scores really are miscalibrated, so this test is meaningful
    assert after <= before
    assert after <= tol


@pytest.mark.parametrize("method", METHODS)
def test_calibrated_probability_is_close_to_p_true(params, method):
    p = _with_method(params, method)
    y_c, _, raw_c = _world(20000, seed=12)
    _, p_true_t, raw_t = _world(20000, seed=13)
    cal = fit_calibrator(raw_c, y_c, p)
    err_before = np.abs(raw_t - p_true_t).mean()
    err_after = np.abs(cal.transform(raw_t) - p_true_t).mean()
    assert err_after < err_before
    assert err_after < p["evaluation"]["calibration_tolerance"] / 2


# --- deterministic -----------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_calibration_is_deterministic(params, method):
    y, _, raw = _world(3000, seed=7)
    p = _with_method(params, method)
    a, b = fit_calibrator(raw, y, p), fit_calibrator(raw.copy(), y.copy(), p)
    assert a.state_hash() == b.state_hash()
    grid = np.linspace(0, 1, 101)
    assert np.array_equal(a.transform(grid), b.transform(grid))


# --- errors ------------------------------------------------------------------
def test_one_class_calib_labels_rejected(params):
    with pytest.raises(CalibrationError) as e:
        fit_calibrator(np.linspace(0.1, 0.9, 50), np.zeros(50), params)
    assert e.value.code == "E_CALIB_ONE_CLASS"


def test_sigmoid_rejects_anti_correlated_scores(params):
    y, p_true, _ = _world(3000, seed=8)
    with pytest.raises(CalibrationError) as e:
        fit_calibrator(1 - p_true, y, _with_method(params, "sigmoid"))
    assert e.value.code == "E_CALIB_ANTI_MONOTONE"


@pytest.mark.parametrize(
    "raw,y",
    [
        (np.array([0.1, np.nan, 0.3]), np.array([0, 1, 0])),
        (np.array([0.1, 0.2]), np.array([0, 1, 1])),
        (np.array([0.1, 0.2, 0.3]), np.array([0, 1, 2])),
        (np.array([]), np.array([])),
    ],
)
def test_invalid_calib_input_rejected(params, raw, y):
    with pytest.raises(CalibrationError) as e:
        fit_calibrator(raw, y, params)
    assert e.value.code == "E_CALIB_INPUT"


def test_transform_handles_empty_and_rejects_nan(params):
    y, _, raw = _world(500, seed=9)
    cal = fit_calibrator(raw, y, params)
    assert cal.transform(np.array([])).shape == (0,)
    with pytest.raises(ValueError):
        cal.transform(np.array([0.2, np.nan]))
