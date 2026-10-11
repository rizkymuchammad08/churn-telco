"""P3: labels do not influence features, forbidden columns stay out, the correlation detector works."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.config import ROOT, load_params
from src.data.load import load_raw, to_csv_text
from src.features.build import build_features, build_labels, feature_names
from src.features.leakage import (
    LeakageError,
    abs_corr,
    assert_no_forbidden,
    check_correlation,
    forbidden_columns,
)
from src.features.leaky_fixture import LEAKY_COLUMN
from src.features.leaky_fixture import build as build_leaky
from src.features.leaky_fixture import main as leaky_main

pytestmark = pytest.mark.p3

P = load_params()
SC = P["schema"]
IDC, LAB = SC["id_column"], SC["label_column"]


@pytest.fixture(scope="module")
def sample():
    return load_raw("sample", P)


@pytest.fixture(scope="module")
def leaky():
    return load_raw(ROOT / "tests" / "fixtures" / "sample_leaky.csv", P)


# --- labels do not influence features ----------------------------------------
def test_replacing_all_labels_does_not_change_features(sample):
    base = build_features(sample, P)
    rng = np.random.default_rng(0)
    for new in (
        np.where(rng.random(len(sample)) < 0.5, SC["positive_label"], SC["negative_label"]),
        np.full(len(sample), SC["positive_label"]),
        np.full(len(sample), SC["negative_label"]),
        np.where(sample[LAB] == SC["positive_label"], SC["negative_label"], SC["positive_label"]),
    ):
        swapped = sample.copy()
        swapped[LAB] = new
        assert build_features(swapped, P).equals(base)


def test_features_do_not_read_label_column_at_all(sample):
    # Labels replaced with an invalid value: the feature builder still runs (it never touches that column).
    broken = sample.copy()
    broken[LAB] = "not-a-label"
    assert build_features(broken, P).equals(build_features(sample, P))


def test_excluded_columns_do_not_change_features(sample):
    # Exclude policy: changing tenure/TotalCharges does not change the feature matrix.
    base = build_features(sample, P)
    altered = sample.copy()
    altered["tenure"] = 72 - altered["tenure"]
    altered["TotalCharges"] = altered["TotalCharges"].fillna(0.0) * 3 + 1
    assert build_features(altered, P).equals(base)


def test_included_flagged_columns_do_change_features(sample):
    # Control: with include_flagged, the same change is visible (the test is not vacuous).
    p = {**copy.deepcopy(P), "leakage": {**P["leakage"], "policy": "include_flagged"}}
    altered = sample.copy()
    altered["tenure"] = 72 - altered["tenure"]
    assert not build_features(altered, p).equals(build_features(sample, p))


# --- forbidden columns -------------------------------------------------------
def test_forbidden_columns_cover_id_label_and_flagged():
    assert forbidden_columns(P) == {IDC, LAB, *P["leakage"]["flagged_columns"]}
    p = {**P, "leakage": {**P["leakage"], "policy": "include_flagged"}}
    assert forbidden_columns(p) == {IDC, LAB}


def test_feature_matrix_has_no_forbidden_column(sample):
    X = build_features(sample, P)
    assert not set(X.columns) & forbidden_columns(P)
    assert not set(feature_names(P)) & forbidden_columns(P)
    assert_no_forbidden(X, P)  # does not raise


@pytest.mark.parametrize("col", ["Churn", "customerID", "tenure", "TotalCharges"])
def test_assert_no_forbidden_rejects_each_forbidden_column(sample, col):
    X = build_features(sample, P)
    X[col] = 1.0
    with pytest.raises(LeakageError) as e:
        assert_no_forbidden(X, P)
    assert e.value.code == "E_LEAK_FORBIDDEN_COLUMN" and e.value.columns == [col]


def test_flagged_allowed_under_include_policy(sample):
    p = {**copy.deepcopy(P), "leakage": {**P["leakage"], "policy": "include_flagged"}}
    assert_no_forbidden(build_features(sample, p), p)
    X = build_features(sample, p)
    X[LAB] = 1.0
    with pytest.raises(LeakageError):
        assert_no_forbidden(X, p)  # the label stays forbidden


def test_sensitive_names_never_in_feature_names():
    # Column names that carry the churn outcome (e.g. churn_date, cancellation reason) must not exist.
    banned = ("churn", "reason", "cancel", "termination")
    for name in feature_names(P):
        assert not any(b in name.lower() for b in banned), name


# --- correlation detector ----------------------------------------------------
def test_leaky_fixture_equals_generator(leaky):
    assert leaky_main(["--check"]) == 0
    rebuilt = build_leaky(P)
    assert list(rebuilt.columns) == list(leaky.columns)
    assert to_csv_text(rebuilt) == to_csv_text(leaky)


def test_detector_catches_leaky_fixture(leaky):
    X = build_features(leaky, P)
    X[LEAKY_COLUMN] = leaky.set_index(IDC)[LEAKY_COLUMN]
    y = build_labels(leaky, P)
    assert abs_corr(X, y)[LEAKY_COLUMN] > 0.99
    with pytest.raises(LeakageError) as e:
        check_correlation(X, y, P)
    assert e.value.code == "E_LEAK_CORRELATION" and e.value.columns == [LEAKY_COLUMN]


def test_negative_control_clean_matrix_passes(leaky, sample):
    # The leaky fixture without the leaky column, and the plain sample, must pass.
    for df in (leaky, sample):
        X, y = build_features(df, P), build_labels(df, P)
        corr = check_correlation(X, y, P)
        assert corr.max() < P["leakage"]["max_abs_corr"]
        assert (corr.index == X.columns).all()


def _with_limit(value):
    return {**copy.deepcopy(P), "leakage": {**P["leakage"], "max_abs_corr": value}}


def test_detector_respects_threshold(leaky):
    y = build_labels(leaky, P)
    X = build_features(leaky, P)
    X[LEAKY_COLUMN] = leaky.set_index(IDC)[LEAKY_COLUMN]
    # A loose threshold (above the leaky |corr|) lets it through; a normal threshold catches it.
    assert check_correlation(X, y, _with_limit(0.999999))[LEAKY_COLUMN] < 0.999999
    with pytest.raises(LeakageError):
        check_correlation(X, y, _with_limit(0.9))
    # Genuinely meaningful features are also caught when the threshold is very strict:
    # the threshold applies to all columns.
    with pytest.raises(LeakageError) as e:
        check_correlation(build_features(leaky, P), y, _with_limit(0.05))
    assert "Contract_Two_year" in e.value.columns


def test_detector_sign_and_constant_handling():
    idx = pd.Index(["a", "b", "c", "d"], name=IDC)
    y = pd.Series([0, 1, 0, 1], index=idx)
    X = pd.DataFrame({"pos": [0.0, 2.0, 0.0, 2.0], "neg": [2.0, 0.0, 2.0, 0.0], "const": 1.0}, index=idx)
    c = abs_corr(X, y)
    assert c["pos"] == pytest.approx(1.0) and c["neg"] == pytest.approx(1.0)  # sign ignored
    assert c["const"] == 0.0  # a constant column does not produce NaN


def test_detector_aligns_label_by_index():
    idx = pd.Index(["a", "b", "c", "d"], name=IDC)
    X = pd.DataFrame({"f": [0.0, 1.0, 0.0, 1.0]}, index=idx)
    y = pd.Series([1, 0, 1, 0], index=["b", "a", "d", "c"])  # different order, same content
    assert abs_corr(X, y)["f"] == pytest.approx(1.0)
