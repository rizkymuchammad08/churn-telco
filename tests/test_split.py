"""P4: train, calib, and test are disjoint; sizes match params; deterministic given a seed."""

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import override

from src.config import load_params
from src.data.synthetic import generate
from src.models.split import SplitError, make_split

pytestmark = pytest.mark.p4


def _ids(split):
    return set(split.train), set(split.calib), set(split.test)


def _synthetic(params, n, seed=3):
    return generate(seed, override(params, "sample", "n_rows", value=n))[0]


# --- disjointness ------------------------------------------------------------
def test_sides_are_pairwise_disjoint_and_cover_everything(sample_df, params):
    s = make_split(sample_df, params)
    tr, ca, te = _ids(s)
    assert not (tr & ca) and not (tr & te) and not (ca & te)
    assert tr | ca | te == set(sample_df["customerID"])
    assert len(s.train) + len(s.calib) + len(s.test) == len(sample_df)


def test_no_customer_id_appears_on_two_sides(sample_df, params):
    """Equivalent of path A's 'group' split: one row per customer, so disjoint ids mean disjoint customers."""
    s = make_split(sample_df, params)
    all_ids = s.train + s.calib + s.test
    assert len(all_ids) == len(set(all_ids))


def test_duplicate_customer_id_is_rejected(sample_df, params):
    dup = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(SplitError) as e:
        make_split(dup, params)
    assert e.value.code == "E_SPLIT_ID"


def test_null_customer_id_is_rejected(sample_df, params):
    bad = sample_df.copy()
    bad.loc[0, "customerID"] = None
    with pytest.raises(SplitError) as e:
        make_split(bad, params)
    assert e.value.code == "E_SPLIT_ID"


# --- sizes -------------------------------------------------------------------
def test_sizes_match_params_exactly_on_sample(sample_df, params):
    assert make_split(sample_df, params).sizes() == {"train": 600, "calib": 200, "test": 200}


@pytest.mark.parametrize("n", [997, 1003, 1500])
def test_sizes_follow_fractions_within_one_row(params, n):
    s = make_split(_synthetic(params, n), params)
    sp = params["split"]
    assert abs(len(s.train) - sp["train_frac"] * n) <= 1
    assert abs(len(s.calib) - sp["calib_frac"] * n) <= 1
    assert abs(len(s.test) - sp["test_frac"] * n) <= 1
    assert len(s.train) + len(s.calib) + len(s.test) == n


def test_sizes_follow_changed_fractions(params, sample_df):
    p = override(params, "split", "train_frac", value=0.5)
    p = override(p, "split", "calib_frac", value=0.3)
    p = override(p, "split", "test_frac", value=0.2)
    assert make_split(sample_df, p).sizes() == {"train": 500, "calib": 300, "test": 200}


def test_stratified_label_rate_is_balanced_across_sides(sample_df, params):
    s = make_split(sample_df, params)
    y = sample_df.set_index("customerID")["Churn"].eq("Yes")
    overall = y.mean()
    for side in (s.train, s.calib, s.test):
        assert abs(y.loc[side].mean() - overall) <= 0.01


def test_too_small_data_is_rejected(params, sample_df):
    with pytest.raises(SplitError) as e:
        make_split(sample_df.head(10), params)
    assert e.value.code == "E_SPLIT_TOO_SMALL"


# --- determinism -------------------------------------------------------------
def test_same_seed_gives_identical_split(sample_df, params):
    assert make_split(sample_df, params, seed=5) == make_split(sample_df, params, seed=5)


def test_default_seed_comes_from_params(sample_df, params):
    assert make_split(sample_df, params) == make_split(sample_df, params, seed=params["seed"])


def test_different_seed_gives_different_split(sample_df, params):
    """Negative control: the determinism tests are only meaningful if the seed actually has an effect."""
    assert make_split(sample_df, params, seed=1) != make_split(sample_df, params, seed=2)


def test_split_does_not_depend_on_row_order(sample_df, params):
    shuffled = sample_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert make_split(shuffled, params) == make_split(sample_df, params)


def test_make_split_does_not_mutate_input(sample_df, params):
    before = sample_df.copy(deep=True)
    make_split(sample_df, params)
    pd.testing.assert_frame_equal(sample_df, before)


# --- temporal ----------------------------------------------------------------
WINDOW, GAP = 30, 45


@pytest.fixture
def temporal_params(params, tmp_path):
    p = override(params, "temporal", value=True)
    p["split"] = {**p["split"], "type": "temporal", "window_days": WINDOW, "purge_gap_days": GAP}
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump(p), encoding="utf-8")
    return load_params(f)  # goes through the validator: the temporal config must be valid


@pytest.fixture
def timed(params):
    df = _synthetic(params, 1200)
    # 2 rows per day over 600 days: date ties straddle the boundaries between sides
    days = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.repeat(np.arange(600), 2), unit="D")
    return df, pd.Series(days)


def _times(df, snap, ids):
    t = pd.Series(snap.to_numpy(), index=df["customerID"].to_numpy())
    return t.loc[ids]


def test_temporal_order_and_gap(temporal_params, timed):
    df, snap = timed
    s = make_split(df, temporal_params, snapshot=snap)
    tr, ca, te = (_times(df, snap, side) for side in (s.train, s.calib, s.test))
    assert tr.max() < ca.min() and ca.max() < te.min()  # train < calib < test
    assert ca.min() - tr.max() >= pd.Timedelta(days=GAP)
    assert te.min() - ca.max() >= pd.Timedelta(days=GAP)
    assert GAP >= WINDOW


def test_temporal_sides_disjoint_and_purged_rows_accounted(temporal_params, timed):
    df, snap = timed
    s = make_split(df, temporal_params, snapshot=snap)
    tr, ca, te, pg = _ids(s) + (set(s.purged),)
    assert not (tr & ca) and not (tr & te) and not (ca & te)
    assert not (pg & (tr | ca | te))
    assert tr | ca | te | pg == set(df["customerID"])


def test_temporal_sizes_are_targets_minus_purge(temporal_params, timed):
    df, snap = timed
    s = make_split(df, temporal_params, snapshot=snap)
    n, sp = len(df), temporal_params["split"]
    assert len(s.test) == round(n * sp["test_frac"])
    # 2 rows/day: purging removes at most 2 * (GAP + 1) rows at each boundary
    assert 0 < n * sp["train_frac"] - len(s.train) <= 2 * (GAP + 1) + 1
    assert 0 < n * sp["calib_frac"] - len(s.calib) <= 2 * (GAP + 1) + 1


def test_temporal_is_deterministic_and_seed_independent(temporal_params, timed):
    df, snap = timed
    a = make_split(df, temporal_params, seed=1, snapshot=snap)
    assert a == make_split(df, temporal_params, seed=1, snapshot=snap)
    assert a == make_split(df, temporal_params, seed=2, snapshot=snap)


def test_temporal_without_gap_violates_distance(temporal_params, timed):
    """Negative control: without purging, the distance between sides is < GAP, so the test above is real."""
    df, snap = timed
    p = override(temporal_params, "split", "purge_gap_days", value=0)
    s = make_split(df, p, snapshot=snap)
    tr, ca = _times(df, snap, s.train), _times(df, snap, s.calib)
    assert ca.min() - tr.max() < pd.Timedelta(days=GAP)


def test_temporal_requires_time_axis(temporal_params, timed):
    df, snap = timed
    with pytest.raises(SplitError) as e:
        make_split(df, temporal_params)
    assert e.value.code == "E_SPLIT_NO_TIME_AXIS"
    with pytest.raises(SplitError) as e:
        make_split(df, temporal_params, snapshot=snap.iloc[:-1])
    assert e.value.code == "E_SPLIT_NO_TIME_AXIS"
