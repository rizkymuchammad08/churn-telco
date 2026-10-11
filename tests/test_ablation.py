"""P4: ablation with/without tenure and TotalCharges. Diagnostic only; the official policy does not change."""

import pytest
from conftest import override

from src.features.build import build_features
from src.models.ablation import METRICS, ablate

pytestmark = pytest.mark.p4


@pytest.fixture(scope="module")
def fast2(fast):
    return override(fast, "ablation", "seeds", value=[1, 2])


@pytest.fixture(scope="module")
def out(sample_df, fast2, assumptions):
    return ablate(sample_df, fast2, assumptions)


def test_structure_and_seeds(out, fast2):
    assert out["seeds"] == fast2["ablation"]["seeds"]
    assert [r["seed"] for r in out["per_seed"]] == out["seeds"]
    for m in METRICS:
        assert set(out["summary"][m]) >= {"without", "with", "delta_with_minus_without"}


def test_delta_consistent_with_per_seed(out):
    for m in METRICS:
        d = [r["with"][m] - r["without"][m] for r in out["per_seed"]]
        assert out["summary"][m]["delta_with_minus_without"]["mean"] == pytest.approx(sum(d) / len(d))


def test_official_policy_unchanged(out, sample_df, fast2):
    assert fast2["leakage"]["policy"] == "exclude"
    assert out["official_policy"] == "exclude"
    cols = set(build_features(sample_df, fast2).columns)
    assert not {"tenure", "TotalCharges"} & cols


def test_deterministic(out, sample_df, fast2, assumptions):
    assert ablate(sample_df, fast2, assumptions) == out


def test_leaky_column_correlation_reported(out, fast2):
    assert set(out["abs_corr_with_label"]) == set(fast2["leakage"]["flagged_columns"])


def test_input_data_hash_reported(out, sample_df):
    from src.data.load import data_hash

    assert out["input_data_hash"] == data_hash(sample_df)


def test_detector_limit_and_selection_counts_recorded(out, fast2):
    # Documents state these without reading params.yaml, so they must be in the output.
    assert out["max_abs_corr"] == fast2["leakage"]["max_abs_corr"]
    assert out["n_seeds"] == len(out["seeds"])
    changed = sum(r["with"]["selected"] != r["without"]["selected"] for r in out["per_seed"])
    assert out["n_selection_changed"] == changed
