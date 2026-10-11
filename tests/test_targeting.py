"""P4: break-even threshold p*, capacity limit k, and the contact set as a prefix of the rank."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.config import breakeven_threshold
from src.models.predict import predict
from src.models.targeting import TargetingError, capacity_for, p_star, plan_contacts

pytestmark = pytest.mark.p4


def _pred(p):
    """Synthetic predict output: rank 1 = highest p_churn (input row order is preserved)."""
    p = np.asarray(p, dtype="float64")
    order = np.lexsort((np.arange(len(p)), -p))
    rank = np.empty(len(p), dtype="int64")
    rank[order] = np.arange(1, len(p) + 1)
    return pd.DataFrame({"p_churn": p, "rank": rank}, index=[f"C{i:04d}" for i in range(len(p))])


def _a(assumptions, **kw):
    """Return a deep copy of the assumptions with the given overrides applied."""
    a = copy.deepcopy(assumptions)
    a.update(kw)
    return a


@pytest.fixture(scope="module")
def pop():
    return _pred(np.random.default_rng(0).uniform(0, 0.6, 500))


# --- p* ----------------------------------------------------------------------
def test_p_star_is_cost_over_save_rate_times_clv(assumptions):
    assert p_star(assumptions) == pytest.approx(40 / (0.25 * 1200))
    assert p_star(assumptions) == pytest.approx(0.13333333)
    assert p_star(_a(assumptions, cost_intervention=60.0, save_rate=0.3, clv=1000.0)) == pytest.approx(0.2)
    assert p_star(assumptions) == breakeven_threshold(assumptions)


def test_plan_reports_the_same_p_star(pop, assumptions):
    assert plan_contacts(pop, assumptions).p_star == pytest.approx(p_star(assumptions))


# --- contact limit -----------------------------------------------------------
@pytest.mark.parametrize("k", [0, 1, 5, 50, 499, 500, 10_000])
def test_contacts_never_exceed_k_and_never_fall_below_p_star(pop, assumptions, k):
    plan = plan_contacts(pop, assumptions, k=k)
    assert plan.n_contacts <= min(k, len(pop))
    assert (plan.contacts["p_churn"] >= p_star(assumptions)).all()
    # negative control: when capacity is not binding, only the threshold limits the count
    assert plan.n_contacts == min(k, int((pop["p_churn"] >= p_star(assumptions)).sum()))


def test_default_k_is_scaled_campaign_capacity(pop, assumptions):
    plan = plan_contacts(pop, assumptions)
    assert plan.k == capacity_for(len(pop), assumptions) == 50  # 500 * 700 / 7043 = 49.7 -> 50
    assert plan.n_contacts <= plan.k


def test_capacity_for_scales_to_population(assumptions):
    assert capacity_for(7043, assumptions) == 700
    assert capacity_for(0, assumptions) == 0
    assert capacity_for(200, assumptions) == 20
    assert capacity_for(5, assumptions) == 0
    assert all(capacity_for(n, assumptions) <= n for n in (1, 10, 700, 7043, 20_000))


def test_zero_capacity_gives_zero_contacts(pop, assumptions):
    assert plan_contacts(pop, assumptions, k=0).n_contacts == 0
    assert plan_contacts(pop, _a(assumptions, campaign_capacity=0)).n_contacts == 0


# --- prefix of the rank ------------------------------------------------------
@pytest.mark.parametrize("k", [3, 20, 500])
def test_contact_set_is_a_prefix_of_rank(pop, assumptions, k):
    c = plan_contacts(pop, assumptions, k=k).contacts
    assert c["rank"].tolist() == list(range(1, len(c) + 1))
    assert c["rank"].is_monotonic_increasing


def test_prefix_holds_when_input_rows_are_not_in_rank_order(pop, assumptions):
    shuffled = pop.sample(frac=1.0, random_state=1)
    c = plan_contacts(shuffled, assumptions, k=40).contacts
    assert c["rank"].tolist() == list(range(1, len(c) + 1))
    assert c.index.tolist() == plan_contacts(pop, assumptions, k=40).contacts.index.tolist()


def test_p_star_boundary_is_inclusive(assumptions):
    a = _a(assumptions, cost_intervention=30.0, save_rate=0.5, clv=600.0)  # p* = 0.1
    plan = plan_contacts(_pred([0.3, 0.1, 0.0999, 0.05]), a, k=10)
    assert p_star(a) == 0.1
    assert plan.contacts["p_churn"].tolist() == [0.3, 0.1]


# --- monotonicity with respect to assumptions --------------------------------
def test_raising_save_rate_never_reduces_contacts(pop, assumptions):
    for k in (len(pop), 25):  # loose capacity first, then tight (min(k, eligible count) also never drops)
        counts = [
            plan_contacts(pop, _a(assumptions, save_rate=s), k=k).n_contacts
            for s in np.linspace(0.1, 1.0, 40)
        ]
        assert all(b >= a for a, b in zip(counts, counts[1:], strict=False))


def test_raising_save_rate_adds_contacts_on_loose_capacity(pop, assumptions):
    """Negative control: the monotonicity above is not a coincidence of a constant count."""
    lo = plan_contacts(pop, _a(assumptions, save_rate=0.15), k=len(pop)).n_contacts
    hi = plan_contacts(pop, _a(assumptions, save_rate=0.60), k=len(pop)).n_contacts
    assert hi > lo


def test_raising_cost_never_adds_contacts(pop, assumptions):
    for k in (len(pop), 25):
        counts = [
            plan_contacts(pop, _a(assumptions, cost_intervention=c), k=k).n_contacts
            for c in np.linspace(1, 400, 60)
        ]
        assert all(b <= a for a, b in zip(counts, counts[1:], strict=False))


def test_raising_cost_removes_contacts_on_loose_capacity(pop, assumptions):
    lo = plan_contacts(pop, _a(assumptions, cost_intervention=10.0), k=len(pop)).n_contacts
    hi = plan_contacts(pop, _a(assumptions, cost_intervention=150.0), k=len(pop)).n_contacts
    assert lo > hi


def test_cost_above_value_of_a_save_gives_no_contacts(pop, assumptions):
    # cost >= save_rate * clv -> p* >= 1: nobody is worth contacting, and no error is raised
    assert plan_contacts(pop, _a(assumptions, cost_intervention=400.0), k=len(pop)).n_contacts == 0


# --- hand-computed fixture ---------------------------------------------------
def test_hand_computed_fixture(assumptions):
    # p* = 40 / (0.25 * 1200) = 0.1333...
    pred = _pred([0.05, 0.9, 0.13, 0.5, 0.01, 0.3, 0.2, 0.14])
    # sorted by p_churn: 0.9(C1) 0.5(C3) 0.3(C5) 0.2(C6) 0.14(C7) | 0.13(C2) 0.05(C0) 0.01(C4)
    full = plan_contacts(pred, assumptions, k=8)
    assert full.contacts.index.tolist() == ["C0001", "C0003", "C0005", "C0006", "C0007"]  # 0.13 < p*
    assert plan_contacts(pred, assumptions, k=3).contacts.index.tolist() == ["C0001", "C0003", "C0005"]
    assert plan_contacts(pred, assumptions, k=1).contacts.index.tolist() == ["C0001"]
    # raise save_rate to 0.4 -> p* = 0.0833: 0.13 is now included, 0.05 is not
    up = plan_contacts(pred, _a(assumptions, save_rate=0.4), k=8)
    assert up.p_star == pytest.approx(40 / (0.4 * 1200))
    assert up.contacts.index.tolist() == ["C0001", "C0003", "C0005", "C0006", "C0007", "C0002"]
    # raise cost to 100 -> p* = 0.3333: only 0.9 and 0.5 remain
    dear = plan_contacts(pred, _a(assumptions, cost_intervention=100.0), k=8)
    assert dear.contacts.index.tolist() == ["C0001", "C0003"]


# --- errors ------------------------------------------------------------------
@pytest.mark.parametrize("k", [-1, 2.5, True, "3"])
def test_invalid_k_raises(pop, assumptions, k):
    with pytest.raises(TargetingError) as e:
        plan_contacts(pop, assumptions, k=k)
    assert e.value.code == "E_TARGET_CAPACITY"


def test_missing_columns_raise(pop, assumptions):
    with pytest.raises(TargetingError) as e:
        plan_contacts(pop.drop(columns=["rank"]), assumptions)
    assert e.value.code == "E_TARGET_COLUMNS"


def test_plan_does_not_mutate_input(pop, assumptions):
    before = pop.copy(deep=True)
    plan_contacts(pop, assumptions, k=30)
    pd.testing.assert_frame_equal(pop, before)


# --- integration with predict ------------------------------------------------
def test_contacts_from_real_predictions(base_run, sample_df, fast, assumptions):
    pred = predict(sample_df, base_run.bundle, fast)
    plan = plan_contacts(pred, assumptions)
    assert plan.k == capacity_for(len(pred), assumptions) == 99  # 1000 * 700 / 7043 = 99.4
    assert 0 < plan.n_contacts <= plan.k
    assert plan.contacts["rank"].tolist() == list(range(1, plan.n_contacts + 1))
    assert (plan.contacts["p_churn"] >= plan.p_star).all()
    assert set(plan.contacts.index) <= set(sample_df["customerID"])
