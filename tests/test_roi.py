"""P5: business value of the contact policy. All fixtures match hand computation.

Hand fixture (clv 1200, cost 40, save_rate 0.25 -> save*clv = 300 per saved churner):
  y       = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]   n = 10, base rate 0.4
  model   = contacts at indices 0, 1, 2, 4, 5 -> 5 contacts, 3 churners
    net   = 3 * 300 - 5 * 40 = 700
  random k = 5  -> expected churners 5 * 0.4 = 2 -> net = 2 * 300 - 5 * 40 = 400
  small   = contacts at indices 0, 1          -> 2 contacts, 2 churners -> net = 600 - 80 = 520
"""

import copy

import numpy as np
import pandas as pd
import pytest

from src.reporting.roi import (
    Outcome,
    RoiError,
    breakeven_save_rate,
    build_roi,
    incremental_value,
    k_test,
    model_policy,
    net_value,
    no_contact,
    outcome_from_mask,
    random_policy,
)

pytestmark = pytest.mark.p5

Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])


def _mask(*idx, n=10):
    m = np.zeros(n, dtype=bool)
    m[list(idx)] = True
    return m


@pytest.fixture()
def a(assumptions):
    """Fixed assumptions for the hand computations (independent of assumptions.yaml)."""
    d = copy.deepcopy(assumptions)
    d.update(clv=1200.0, cost_intervention=40.0, save_rate=0.25, operating_cost_per_year=0.0)
    return d


def _with(a, **kw):
    d = copy.deepcopy(a)
    d.update(kw)
    return d


# --- base values -------------------------------------------------------------
def test_no_contact_is_zero(a):
    assert net_value(no_contact(), a) == 0.0
    assert outcome_from_mask("none", _mask(), Y).n_contacts == 0
    assert net_value(outcome_from_mask("none", _mask(), Y), a) == 0.0


def test_no_contact_is_zero_even_with_operating_cost(a):
    assert net_value(no_contact(), _with(a, operating_cost_per_year=5000.0), years=3) == 0.0


def test_contacting_only_true_churners_is_n_times_net_per_save(a):
    for n in (1, 4):
        o = outcome_from_mask("c", _mask(*range(n)), Y)  # indices 0..3 are all churners
        assert o.n_churners_contacted == n
        assert net_value(o, a) == pytest.approx(n * (0.25 * 1200 - 40))


def test_zero_intervention_cost_values_only_saved_customers(a):
    z = _with(a, cost_intervention=0.0)
    o = outcome_from_mask("m", _mask(0, 1, 2, 4, 5), Y)  # 3 churners, 2 non-churners
    assert net_value(o, z) == pytest.approx(3 * 0.25 * 1200)
    nonchurn_only = outcome_from_mask("m", _mask(4, 5, 6), Y)
    assert net_value(nonchurn_only, z) == 0.0


def test_hand_computed_model_fixture(a):
    o = outcome_from_mask("model", _mask(0, 1, 2, 4, 5), Y)
    assert (o.n_contacts, o.n_churners_contacted) == (5, 3)
    assert net_value(o, a) == pytest.approx(700.0)
    assert net_value(o, _with(a, operating_cost_per_year=100.0)) == pytest.approx(600.0)


def test_random_policy_hand_computed(a):
    r = random_policy(Y, 5)
    assert (r.n_contacts, r.n_churners_contacted) == (5, pytest.approx(2.0))
    assert net_value(r, a) == pytest.approx(400.0)


@pytest.mark.parametrize("k", [-1, 11, 2.5, True])
def test_random_policy_rejects_bad_k(k):
    with pytest.raises(RoiError) as e:
        random_policy(Y, k)
    assert e.value.code == "E_ROI_K"


# --- incremental -------------------------------------------------------------
def test_identical_policies_have_zero_incremental_value(a):
    for op in (0.0, 250.0):
        aa = _with(a, operating_cost_per_year=op)
        o = outcome_from_mask("m", _mask(0, 1, 2, 4, 5), Y)
        same = outcome_from_mask("copy", _mask(0, 1, 2, 4, 5), Y)
        assert incremental_value(o, o, aa) == 0.0
        assert incremental_value(o, same, aa, years=2) == 0.0
    assert incremental_value(no_contact(), no_contact(), a) == 0.0


def test_incremental_hand_computed(a):
    model = outcome_from_mask("model", _mask(0, 1, 2, 4, 5), Y)
    small = outcome_from_mask("small", _mask(0, 1), Y)
    assert net_value(small, a) == pytest.approx(520.0)
    assert incremental_value(model, small, a) == pytest.approx(180.0)  # 1 * 300 - 3 * 40
    assert incremental_value(model, no_contact(), a) == pytest.approx(700.0)
    assert incremental_value(model, random_policy(Y, 5), a) == pytest.approx(300.0)


def test_incremental_is_antisymmetric(a):
    model = outcome_from_mask("model", _mask(0, 1, 2, 4, 5), Y)
    small = outcome_from_mask("small", _mask(0, 1), Y)
    assert incremental_value(model, small, a) == pytest.approx(-incremental_value(small, model, a))


# --- operating cost per year -------------------------------------------------
def test_operating_cost_subtracted_once_per_year_not_per_contact(a):
    big = outcome_from_mask("big", _mask(0, 1, 2, 3, 4, 5, 6, 7), Y)
    small = outcome_from_mask("small", _mask(0), Y)
    for o in (big, small):
        base = net_value(o, a)
        assert base - net_value(o, _with(a, operating_cost_per_year=100.0)) == pytest.approx(100.0)


def test_operating_cost_scales_with_years_only(a):
    o = outcome_from_mask("m", _mask(0, 1, 2, 4, 5), Y)
    aa = _with(a, operating_cost_per_year=100.0)
    assert net_value(o, aa, years=1) == pytest.approx(600.0)
    assert net_value(o, aa, years=3) == pytest.approx(3 * 600.0)
    assert net_value(o, aa, years=0.5) == pytest.approx(300.0)


@pytest.mark.parametrize("years", [0, -1, True, "1"])
def test_years_must_be_positive_number(a, years):
    with pytest.raises(RoiError) as e:
        net_value(no_contact(), a, years=years)
    assert e.value.code == "E_ROI_YEARS"


def test_operating_cost_drops_out_of_incremental_between_two_campaigns(a):
    model = outcome_from_mask("model", _mask(0, 1, 2, 4, 5), Y)
    small = outcome_from_mask("small", _mask(0, 1), Y)
    d0 = incremental_value(model, small, a)
    d1 = incremental_value(model, small, _with(a, operating_cost_per_year=999.0))
    assert d0 == pytest.approx(d1)
    assert breakeven_save_rate(model, small, a) == pytest.approx(
        breakeven_save_rate(model, small, _with(a, operating_cost_per_year=999.0))
    )


# --- k_test ------------------------------------------------------------------
def test_k_test_scales_capacity_to_batch(assumptions):
    n_pop, cap = assumptions["population_per_period"], assumptions["campaign_capacity"]
    assert k_test(n_pop, assumptions) == cap
    assert k_test(200, assumptions) == 20  # 700 / 7043 * 200 = 19.88 -> 20
    assert k_test(0, assumptions) == 0
    assert k_test(3, assumptions) <= 3
    sizes = [50, 100, 200, 400, 1000]
    ks = [k_test(n, assumptions) for n in sizes]
    assert ks == sorted(ks)
    for n, k in zip(sizes, ks, strict=True):
        assert abs(k - n * cap / n_pop) <= 0.5


# --- breakeven_save_rate ------------------------------------------------------
@pytest.mark.parametrize("op", [0.0, 100.0, 750.0])
@pytest.mark.parametrize(
    "b_name,b_idx",
    [("none", ()), ("small", (0, 1)), ("other", (0, 4, 6, 7))],
)
def test_incremental_is_zero_at_breakeven_save_rate(a, op, b_name, b_idx):
    aa = _with(a, operating_cost_per_year=op)
    model = outcome_from_mask("model", _mask(0, 1, 2, 4, 5), Y)
    b = outcome_from_mask(b_name, _mask(*b_idx), Y)
    s_star = breakeven_save_rate(model, b, aa)
    assert s_star is not None
    assert incremental_value(model, b, aa, save_rate=s_star) == pytest.approx(0.0, abs=1e-9)
    assert incremental_value(model, b, aa, save_rate=s_star * 1.1 + 1e-6) > 0
    assert incremental_value(model, b, aa, years=4, save_rate=s_star) == pytest.approx(0.0, abs=1e-9)


def test_breakeven_hand_computed(a):
    model = outcome_from_mask("model", _mask(0, 1, 2, 4, 5), Y)
    small = outcome_from_mask("small", _mask(0, 1), Y)
    assert breakeven_save_rate(model, small, a) == pytest.approx(3 * 40 / (1 * 1200))  # 0.1
    assert breakeven_save_rate(model, no_contact(), a) == pytest.approx(5 * 40 / (3 * 1200))
    assert breakeven_save_rate(model, no_contact(), _with(a, operating_cost_per_year=100.0)) == pytest.approx(
        (5 * 40 + 100) / (3 * 1200)
    )


def test_breakeven_none_when_no_extra_saves(a):
    model = outcome_from_mask("model", _mask(0, 1, 2, 4, 5), Y)
    assert breakeven_save_rate(model, model, a) is None
    assert breakeven_save_rate(no_contact(), model, a) is None


def test_outcome_rejects_impossible_counts():
    with pytest.raises(RoiError):
        Outcome("x", 2, 3)
    with pytest.raises(RoiError):
        Outcome("x", -1, 0)


def test_mask_and_label_validation():
    with pytest.raises(RoiError):
        outcome_from_mask("x", _mask(0, n=9), Y)
    with pytest.raises(RoiError):
        outcome_from_mask("x", _mask(0), np.array([0, 1, 2, 0, 0, 0, 0, 0, 0, 0]))


# --- model_policy and build_roi ----------------------------------------------
@pytest.fixture()
def pred():
    """p_churn sorted; p* = 0.1333. Only 3 rows are above p*."""
    p = [0.9, 0.6, 0.5, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
    return pd.DataFrame({"label": Y, "p_raw": p, "p_churn": p}, index=[f"C{i}" for i in range(10)])


def test_model_policy_respects_p_star_and_k(pred, a):
    assert (o := model_policy(pred, a, k=10)).n_contacts == 3  # only p >= p*
    assert o.n_churners_contacted == 3
    assert model_policy(pred, a, k=2).n_contacts == 2
    assert model_policy(pred, a, k=0).n_contacts == 0


def test_model_policy_requires_columns(pred, a):
    with pytest.raises(RoiError) as e:
        model_policy(pred.drop(columns="label"), a)
    assert e.value.code == "E_ROI_COLUMNS"


def test_build_roi_fixture_matches_hand_calc(pred, a, params):
    # n = 10 -> k_test = round(10 * cap / pop); force the capacity so that k = 5 (= 10 * 0.5).
    aa = _with(a, campaign_capacity=500, population_per_period=1000)
    roi = build_roi(pred, aa, params)
    assert roi["k_test"] == 5
    assert roi["policies"]["no_contact"]["net_value"] == 0.0
    m = roi["policies"]["model_p_star"]
    assert (m["n_contacts"], m["n_churners_contacted"]) == (3, 3)
    assert m["net_value"] == pytest.approx(3 * 300 - 3 * 40)  # 780
    r = roi["policies"]["random_k"]
    assert (r["n_contacts"], r["n_churners_contacted"]) == (5, pytest.approx(2.0))
    assert r["net_value"] == pytest.approx(400.0)
    assert roi["incremental"]["model_vs_no_contact"] == pytest.approx(780.0)
    assert roi["incremental"]["model_vs_random_k"] == pytest.approx(380.0)
    assert roi["breakeven_save_rate"]["model_vs_no_contact"] == pytest.approx(3 * 40 / (3 * 1200))
    assert roi["business_status"] == "conditional"
    assert roi["data_source"] == params["data_source"]


def test_build_roi_breakeven_zero_incremental(pred, a, params):
    aa = _with(a, campaign_capacity=500, population_per_period=1000, operating_cost_per_year=60.0)
    roi = build_roi(pred, aa, params)
    s = roi["breakeven_save_rate"]["model_vs_no_contact"]
    model = outcome_from_mask("m", _mask(0, 1, 2), Y)
    assert incremental_value(model, no_contact(), aa, save_rate=s) == pytest.approx(0.0, abs=1e-9)


def test_build_roi_status_follows_save_rate_source(pred, a, params):
    assert build_roi(pred, _with(a, save_rate_source="measured"), params)["business_status"] == "measured"
