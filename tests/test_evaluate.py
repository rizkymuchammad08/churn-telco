"""P4: metrics match hand computation, behaviour on extreme scores, deterministic ties, segments."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from conftest import override

from src.models.evaluate import (
    brier,
    by_segment,
    evaluate,
    expected_calibration_error,
    expected_positives_at_k,
    lift_at_k,
    pr_auc,
    precision_at_k,
    recall_at_k,
    roc_auc,
)

pytestmark = pytest.mark.p4

# Hand fixture: 10 customers, descending scores, 3 positives (indices 0, 2, 5), positive rate 0.3.
S = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
Y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])


# --- hand computation --------------------------------------------------------
def test_precision_recall_lift_at_k_hand_computed():
    # k=4: top-4 = scores 0.9, 0.8, 0.7, 0.6 with labels 1, 0, 1, 0 -> TP = 2
    assert precision_at_k(Y, S, 4) == pytest.approx(2 / 4)
    assert recall_at_k(Y, S, 4) == pytest.approx(2 / 3)
    assert lift_at_k(Y, S, 4) == pytest.approx((2 / 4) / 0.3)
    # k=1: TP = 1
    assert precision_at_k(Y, S, 1) == pytest.approx(1.0)
    assert recall_at_k(Y, S, 1) == pytest.approx(1 / 3)
    assert lift_at_k(Y, S, 1) == pytest.approx(1 / 0.3)
    # k=10: all rows -> precision = positive rate, recall = 1, lift = 1
    assert precision_at_k(Y, S, 10) == pytest.approx(0.3)
    assert recall_at_k(Y, S, 10) == pytest.approx(1.0)
    assert lift_at_k(Y, S, 10) == pytest.approx(1.0)


def test_brier_hand_computed():
    # (0.9-1)^2 + 0.8^2 + (0.7-1)^2 + 0.6^2 + 0.5^2 + (0.4-1)^2 + 0.3^2 + 0.2^2 + 0.1^2 + 0.05^2
    # = 0.01 + 0.64 + 0.09 + 0.36 + 0.25 + 0.36 + 0.09 + 0.04 + 0.01 + 0.0025 = 1.8525
    assert brier(Y, S) == pytest.approx(0.18525)


def test_ece_hand_computed_with_five_bins():
    # bin [0.8,1]: p 0.9,0.8 y 1,0   -> |0.85-0.5|  = 0.35
    # bin [0.6,0.8): p 0.7,0.6 y 1,0 -> |0.65-0.5|  = 0.15
    # bin [0.4,0.6): p 0.5,0.4 y 0,1 -> |0.45-0.5|  = 0.05
    # bin [0.2,0.4): p 0.3,0.2 y 0,0 -> |0.25-0|    = 0.25
    # bin [0,0.2):   p 0.1,0.05 y 0,0 -> |0.075-0|  = 0.075
    # each bin has weight 2/10: ECE = 0.2 * 0.875 = 0.175
    assert expected_calibration_error(Y, S, bins=5) == pytest.approx(0.175)


def test_auc_hand_computed():
    # PR-AUC (average precision): positives at ranks 1, 3, 6 -> (1/1 + 2/3 + 3/6) / 3
    assert pr_auc(Y, S) == pytest.approx((1 + 2 / 3 + 1 / 2) / 3)
    # ROC-AUC: correct pairs 7 + 6 + 4 = 17 out of 3 * 7 = 21
    assert roc_auc(Y, S) == pytest.approx(17 / 21)


def test_evaluate_matches_hand_fixture(params):
    p = override(params, "evaluation", "ece_bins", value=5)
    r = evaluate(Y, S, p, k=4)
    assert r["n"] == 10 and r["n_pos"] == 3 and r["k"] == 4
    assert r["base_rate"] == pytest.approx(0.3)
    assert r["precision_at_k"] == pytest.approx(0.5)
    assert r["recall_at_k"] == pytest.approx(2 / 3)
    assert r["lift_at_k"] == pytest.approx(0.5 / 0.3)
    assert r["brier"] == pytest.approx(0.18525)
    assert r["ece"] == pytest.approx(0.175)
    assert "by_segment" not in r


# --- extreme scores ----------------------------------------------------------
def test_perfect_scores_give_unit_auc():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(int)
    score = y * 0.8 + rng.random(500) * 0.1  # every positive above every negative
    assert pr_auc(y, score) == pytest.approx(1.0)
    assert roc_auc(y, score) == pytest.approx(1.0)
    k = int(y.sum())
    assert precision_at_k(y, score, k) == pytest.approx(1.0)
    assert lift_at_k(y, score, k) == pytest.approx(1 / y.mean())


@pytest.mark.parametrize("k", [1, 7, 50, 250, 499, 500])
def test_constant_scores_give_lift_one(k):
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.27).astype(int)
    assert lift_at_k(y, np.full(500, 0.42), k) == pytest.approx(1.0)


def test_random_scores_give_roc_auc_near_half():
    rng = np.random.default_rng(2)
    y = (rng.random(20000) < 0.3).astype(int)
    assert roc_auc(y, rng.random(20000)) == pytest.approx(0.5, abs=0.02)


def test_reversed_scores_give_roc_auc_zero():
    """Negative control: reversed scores must be detected, not slip through."""
    assert roc_auc(Y, -S) == pytest.approx(4 / 21)


# --- calibration on p_true ---------------------------------------------------
def _beta_world(n=20000, seed=3):
    rng = np.random.default_rng(seed)
    p_true = rng.beta(2, 5, n)
    return (rng.random(n) < p_true).astype(int), p_true


def test_p_true_scores_have_ece_near_zero_within_tolerance(params):
    y, p_true = _beta_world()
    ece = expected_calibration_error(y, p_true, params["evaluation"]["ece_bins"])
    assert ece <= params["evaluation"]["calibration_tolerance"]
    assert ece < 0.02


def test_distorted_scores_exceed_tolerance(params):
    """Negative control: ECE really does catch miscalibrated scores."""
    y, p_true = _beta_world()
    logit = np.log(p_true / (1 - p_true))
    distorted = 1 / (1 + np.exp(-(1.8 * logit + 0.9)))
    ece = expected_calibration_error(y, distorted, params["evaluation"]["ece_bins"])
    assert ece > params["evaluation"]["calibration_tolerance"]


def test_ece_rejects_values_outside_unit_interval():
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [0.2, 1.2], bins=5)
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [-0.1, 0.5], bins=5)


# --- tie ---------------------------------------------------------------------
def test_tie_at_cutoff_uses_expected_positives():
    # top-2: score 0.9 (positive) + 1 slot out of three 0.5 scores containing 1 positive -> 1 + 1/3
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    assert expected_positives_at_k(y, s, 2) == pytest.approx(1 + 1 / 3)
    assert precision_at_k(y, s, 2) == pytest.approx((4 / 3) / 2)
    assert recall_at_k(y, s, 2) == pytest.approx((4 / 3) / 2)
    assert lift_at_k(y, s, 2) == pytest.approx((4 / 3) / 2 / 0.4)


def test_tie_handling_is_deterministic_and_order_free(params):
    rng = np.random.default_rng(4)
    y = (rng.random(300) < 0.3).astype(int)
    s = np.round(rng.random(300), 1)  # many ties
    base = evaluate(y, s, params, k=40)
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(300)
        assert evaluate(y[perm], s[perm], params, k=40) == base
    assert evaluate(y, s, params, k=40) == base


# --- segments ----------------------------------------------------------------
def _segments(n):
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.6, 0.3, 0.1]),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.85, 0.15]),
        }
    )


def test_by_segment_sums_to_total(params):
    n = 1000
    y, p_true = _beta_world(n, seed=6)
    seg = _segments(n)
    res = evaluate(y, p_true, params, k=100, segments=seg)["by_segment"]
    assert set(res) == {"Contract", "SeniorCitizen"}
    for col, levels in res.items():
        assert sum(v["n"] for v in levels.values()) == n
        assert sum(v["n_pos"] for v in levels.values()) == int(y.sum())
        assert set(levels) == set(map(str, seg[col].unique()))


def test_by_segment_per_level_values_match_direct_computation(params):
    n = 1000
    y, p = _beta_world(n, seed=6)
    seg = _segments(n)
    lv = by_segment(y, p, seg, params)["Contract"]["One year"]
    m = (seg["Contract"] == "One year").to_numpy()
    assert lv["n"] == m.sum()
    assert lv["base_rate"] == pytest.approx(y[m].mean())
    assert lv["mean_p"] == pytest.approx(p[m].mean())
    assert lv["brier"] == pytest.approx(brier(y[m], p[m]))
    assert lv["roc_auc"] == pytest.approx(roc_auc(y[m], p[m]))


def test_segments_below_min_n_are_flagged_unreliable(params):
    n = 1000
    y, p = _beta_world(n, seed=6)
    seg = _segments(n)
    min_n = params["segments"]["min_n"]
    res = by_segment(y, p, seg, params)
    for levels in res.values():
        for v in levels.values():
            assert v["reliable"] == (v["n"] >= min_n and 0 < v["n_pos"] < v["n"])
    big = override(params, "segments", "min_n", value=10**6)
    assert not any(v["reliable"] for lv in by_segment(y, p, seg, big).values() for v in lv.values())
    small = override(params, "segments", "min_n", value=1)
    assert by_segment(y, p, seg, small)["Contract"]["Month-to-month"]["reliable"]


def test_small_segment_is_unreliable_even_when_neighbours_are_reliable(params):
    n = 400
    y, p = _beta_world(n, seed=8)
    seg = pd.DataFrame({"g": ["big"] * (n - 5) + ["tiny"] * 5})
    res = by_segment(y, p, seg, params)["g"]
    assert res["big"]["reliable"] is True
    assert res["tiny"]["n"] == 5 and res["tiny"]["reliable"] is False


def test_single_class_segment_is_unreliable_and_auc_is_none(params):
    y = np.array([0] * 60 + [1, 0] * 30)
    p = np.linspace(0.05, 0.95, 120)
    seg = pd.DataFrame({"g": ["allneg"] * 60 + ["mixed"] * 60})
    res = by_segment(y, p, seg, params)["g"]
    assert res["allneg"]["reliable"] is False and res["allneg"]["roc_auc"] is None
    assert res["mixed"]["reliable"] is True


# --- output shape and errors -------------------------------------------------
def test_evaluate_output_is_strict_json(params):
    y = np.array([0] * 100)  # no positives: AUC is undefined -> None, not NaN
    p = np.linspace(0.01, 0.5, 100)
    r = evaluate(y, p, params, k=10, segments=pd.DataFrame({"g": ["a"] * 100}))
    assert r["pr_auc"] is None and r["roc_auc"] is None and r["recall_at_k"] is None
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize("k", [0, -1, 11, 2.5, True])
def test_invalid_k_raises(params, k):
    with pytest.raises(ValueError):
        evaluate(Y, S, params, k=k)


@pytest.mark.parametrize(
    "y,s",
    [([0, 1], [0.1]), ([0, 2], [0.1, 0.2]), ([0, 1], [0.1, math.nan]), ([], [])],
)
def test_invalid_inputs_raise(params, y, s):
    with pytest.raises(ValueError):
        evaluate(y, s, params, k=1)


# --- bootstrap confidence intervals -------------------------------------------
def _scored(n=600, seed=3):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(float)
    s = np.clip(0.3 + 0.25 * (y - 0.3) + rng.normal(0, 0.2, n), 0.01, 0.99)
    return y, s


def test_bootstrap_ci_contains_point_estimate_and_is_ordered(params):
    from src.models.evaluate import CI_METRICS, bootstrap_ci

    y, s = _scored()
    ci = bootstrap_ci(y, s, params, k=60, seed=1)
    point = {
        "pr_auc": pr_auc(y, s),
        "roc_auc": roc_auc(y, s),
        "precision_at_k": precision_at_k(y, s, 60),
        "recall_at_k": recall_at_k(y, s, 60),
        "lift_at_k": lift_at_k(y, s, 60),
        "brier": brier(y, s),
    }
    assert set(CI_METRICS) == set(point)
    for m, v in point.items():
        assert ci[m]["lo"] < ci[m]["hi"]
        assert ci[m]["lo"] <= v <= ci[m]["hi"], m
    assert ci["level"] == params["evaluation"]["ci_level"]
    assert ci["level_pct"] == 95 and ci["n_resamples"] == params["evaluation"]["bootstrap_resamples"]


def test_bootstrap_ci_is_deterministic_and_seed_dependent(params):
    from src.models.evaluate import bootstrap_ci

    y, s = _scored()
    a = bootstrap_ci(y, s, params, k=60, seed=1)
    assert a == bootstrap_ci(y, s, params, k=60, seed=1)
    assert a != bootstrap_ci(y, s, params, k=60, seed=2)


def test_bootstrap_ci_widens_with_level_and_narrows_with_n(params):
    from src.models.evaluate import bootstrap_ci

    y, s = _scored(600)
    narrow = bootstrap_ci(y, s, override(params, "evaluation", "ci_level", value=0.5), k=60, seed=1)
    wide = bootstrap_ci(y, s, override(params, "evaluation", "ci_level", value=0.95), k=60, seed=1)
    assert wide["pr_auc"]["hi"] - wide["pr_auc"]["lo"] > narrow["pr_auc"]["hi"] - narrow["pr_auc"]["lo"]
    y2, s2 = _scored(6000)
    big = bootstrap_ci(y2, s2, params, k=600, seed=1)
    small = bootstrap_ci(y, s, params, k=60, seed=1)
    assert big["roc_auc"]["hi"] - big["roc_auc"]["lo"] < small["roc_auc"]["hi"] - small["roc_auc"]["lo"]


def test_evaluate_adds_ci_only_when_asked(params):
    y, s = _scored()
    assert "ci" not in evaluate(y, s, params, k=60)
    assert "ci" in evaluate(y, s, params, k=60, ci_seed=1)
    json.dumps(evaluate(y, s, params, k=60, ci_seed=1), allow_nan=False)
