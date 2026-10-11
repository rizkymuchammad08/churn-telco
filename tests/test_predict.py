"""P4: output schema, row order, labels not required, domain flag, and clear errors."""

import numpy as np
import pandas as pd
import pytest
from conftest import override

from src.data.load import DataLoadError
from src.features.build import FeatureError
from src.models.predict import PRED_COLUMNS, PredictError, predict

pytestmark = pytest.mark.p4


@pytest.fixture(scope="module")
def pred(base_run, sample_df, fast):
    return predict(sample_df, base_run.bundle, fast)


# --- output schema -----------------------------------------------------------
def test_output_columns_exist(pred):
    assert list(pred.columns) == PRED_COLUMNS
    assert {"p_raw", "p_churn", "rank"} <= set(pred.columns)


def test_p_churn_is_probability_and_equals_calibrated_p_raw(pred, base_run):
    assert pred["p_churn"].between(0, 1).all() and pred["p_raw"].between(0, 1).all()
    assert np.array_equal(pred["p_churn"].to_numpy(), base_run.bundle.calibrator.transform(pred["p_raw"]))


def test_rank_is_a_permutation_and_consistent_with_p_churn(pred):
    n = len(pred)
    assert sorted(pred["rank"]) == list(range(1, n + 1))
    by_rank = pred.sort_values("rank")
    assert by_rank["p_churn"].is_monotonic_decreasing
    # A higher p_churn must always rank better (ties may get different ranks, but never inverted)
    hi = pred["p_churn"].to_numpy()[:, None] > pred["p_churn"].to_numpy()[None, :]
    r = pred["rank"].to_numpy()
    assert not (hi & (r[:, None] > r[None, :])).any()


def test_ties_in_p_churn_are_broken_by_raw_score(pred):
    """Isotonic calibration creates ties; the raw score is used to break them."""
    ties = pred.groupby("p_churn").filter(lambda g: len(g) > 1)
    assert len(ties) > 0  # isotonic calibration on 200 rows is guaranteed to produce ties
    for _, g in ties.groupby("p_churn"):
        assert g.sort_values("rank")["p_raw"].is_monotonic_decreasing


# --- row count and order, labels ---------------------------------------------
def test_row_count_and_order_are_preserved(pred, sample_df):
    assert len(pred) == len(sample_df)
    assert pred.index.tolist() == sample_df["customerID"].tolist()


def test_shuffled_input_keeps_its_own_order_and_same_scores(pred, sample_df, base_run, fast):
    shuffled = sample_df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    out = predict(shuffled, base_run.bundle, fast)
    assert out.index.tolist() == shuffled["customerID"].tolist()
    pd.testing.assert_frame_equal(
        out.loc[pred.index, ["p_raw", "p_churn"]], pred[["p_raw", "p_churn"]], check_exact=True
    )


def test_single_row_scores_equal_batch_scores(pred, sample_df, base_run, fast):
    one = predict(sample_df.iloc[[7]], base_run.bundle, fast)
    assert one["rank"].tolist() == [1] and len(one) == 1
    assert one.iloc[0]["p_churn"] == pred.iloc[7]["p_churn"]


def test_label_is_not_required_and_does_not_matter(pred, sample_df, base_run, fast):
    no_label = predict(sample_df.drop(columns=["Churn"]), base_run.bundle, fast)
    pd.testing.assert_frame_equal(no_label, pred)
    garbage = sample_df.assign(Churn="???")
    pd.testing.assert_frame_equal(predict(garbage, base_run.bundle, fast), pred)


def test_predict_does_not_mutate_input(sample_df, base_run, fast):
    before = sample_df.copy(deep=True)
    predict(sample_df.drop(columns=["Churn"]), base_run.bundle, fast)
    predict(sample_df, base_run.bundle, fast)
    pd.testing.assert_frame_equal(sample_df, before)


def test_empty_input_gives_empty_output(sample_df, base_run, fast):
    out = predict(sample_df.iloc[0:0], base_run.bundle, fast)
    assert len(out) == 0 and list(out.columns) == PRED_COLUMNS


# --- tenure domain -----------------------------------------------------------
def test_customers_below_min_tenure_are_flagged(pred, sample_df, params):
    thr = params["domain"]["min_tenure_months"]
    expected = (sample_df["tenure"] < thr).to_numpy()
    assert expected.any() and not expected.all()  # the sample contains customers with tenure 0
    assert np.array_equal(pred["out_of_domain"].to_numpy(), expected)


def test_flag_follows_domain_param_and_flagged_rows_still_scored(sample_df, base_run, fast):
    p = override(fast, "domain", "min_tenure_months", value=12)
    out = predict(sample_df, base_run.bundle, p)
    assert np.array_equal(out["out_of_domain"].to_numpy(), (sample_df["tenure"] < 12).to_numpy())
    assert len(out) == len(sample_df) and out["p_churn"].notna().all()
    zero = override(fast, "domain", "min_tenure_months", value=0)
    assert not predict(sample_df, base_run.bundle, zero)["out_of_domain"].any()


# --- values outside the feature domain raise clear errors --------------------
def test_unknown_category_raises_clear_error(sample_df, base_run, fast):
    bad = sample_df.copy()
    bad.loc[3, "Contract"] = "Weekly"
    with pytest.raises(FeatureError) as e:
        predict(bad, base_run.bundle, fast)
    assert e.value.code == "E_FEATURE_CATEGORY"
    assert "Contract" in str(e.value) and "Weekly" in str(e.value)


@pytest.mark.parametrize(
    "col,value",
    [
        ("MonthlyCharges", -5.0),
        ("tenure", -1),
        ("SeniorCitizen", 2),
        ("Contract", None),
        ("MonthlyCharges", np.nan),
    ],
)
def test_out_of_domain_values_raise_clear_error(sample_df, base_run, fast, col, value):
    bad = sample_df.copy()
    bad[col] = bad[col].astype(object)
    bad.loc[3, col] = value
    with pytest.raises(PredictError) as e:
        predict(bad, base_run.bundle, fast)
    assert e.value.code == "E_PREDICT_DOMAIN" and col in str(e.value)


def test_missing_column_raises_naming_it(sample_df, base_run, fast):
    with pytest.raises(PredictError) as e:
        predict(sample_df.drop(columns=["MonthlyCharges"]), base_run.bundle, fast)
    assert e.value.code == "E_PREDICT_COLUMNS" and "MonthlyCharges" in str(e.value)


def test_duplicate_id_raises(sample_df, base_run, fast):
    with pytest.raises(FeatureError) as e:
        predict(pd.concat([sample_df, sample_df.iloc[[0]]]), base_run.bundle, fast)
    assert e.value.code == "E_FEATURE_ID"


def test_non_numeric_value_raises(sample_df, base_run, fast):
    bad = sample_df.copy()
    bad["tenure"] = bad["tenure"].astype(object)
    bad.loc[0, "tenure"] = "abc"
    with pytest.raises(DataLoadError) as e:
        predict(bad, base_run.bundle, fast)
    assert e.value.code == "E_NONNUMERIC"


def test_nullable_total_charges_is_allowed_for_new_customers(sample_df, base_run, fast):
    row = sample_df.iloc[[0]].assign(tenure=0, TotalCharges=np.nan)
    out = predict(row, base_run.bundle, fast)
    assert out["out_of_domain"].tolist() == [True] and 0 <= out["p_churn"].iloc[0] <= 1


def test_feature_mismatch_with_model_raises(sample_df, base_run, fast):
    p = override(fast, "leakage", "policy", value="include_flagged")  # adds tenure and TotalCharges
    with pytest.raises(PredictError) as e:
        predict(sample_df, base_run.bundle, p)
    assert e.value.code == "E_PREDICT_FEATURES"
