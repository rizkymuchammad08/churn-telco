"""P4: temporal mode end to end. The real Telco data has no dates, so `temporal: false` stays the default;
this mode is active when `schema.snapshot_column` is set and the data contains that column
(tested on sample + dates)."""

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import override

from src.config import load_params
from src.data.validate import validate
from src.models.ablation import ablate
from src.models.split import SplitError, make_split
from src.models.train import run

pytestmark = pytest.mark.p4

SNAP, WINDOW, GAP = "snapshot_date", 7, 7


@pytest.fixture(scope="module")
def tparams(fast, tmp_path_factory):
    p = override(fast, "temporal", value=True)
    p["split"] = {**p["split"], "type": "temporal", "window_days": WINDOW, "purge_gap_days": GAP}
    p["schema"] = {
        **p["schema"],
        "required_columns": [*p["schema"]["required_columns"], SNAP],
        "snapshot_column": SNAP,
    }
    f = tmp_path_factory.mktemp("tp") / "p.yaml"
    f.write_text(yaml.safe_dump(p), encoding="utf-8")
    return load_params(f)  # goes through the validator


@pytest.fixture(scope="module")
def dated(sample_df):
    df = sample_df.copy()
    days = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(len(df)) // 3, unit="D")
    df[SNAP] = days.strftime("%Y-%m-%d")
    return df


def test_default_stays_non_temporal(params):
    assert params["temporal"] is False
    assert params["split"]["type"] == "random_stratified"
    assert "snapshot_column" not in params["schema"]


def test_split_reads_schema_column(dated, tparams):
    s = make_split(dated, tparams)  # without a snapshot argument
    t = pd.to_datetime(dated.set_index("customerID")[SNAP])
    tr, ca, te = (t.loc[x] for x in (s.train, s.calib, s.test))
    gap = pd.Timedelta(days=GAP)
    assert tr.max() + gap <= ca.min()
    assert ca.max() + gap <= te.min()
    assert s.purged  # rows near the boundary are dropped
    assert len(s.train) + len(s.calib) + len(s.test) + len(s.purged) == len(dated)


def test_full_pipeline_in_temporal_mode(dated, tparams, assumptions):
    res = run(dated, tparams, assumptions)
    assert res.metrics["split_type"] == "temporal"
    assert res.metrics["split"] == res.split.sizes()
    assert set(res.test_pred.index) == set(res.split.test)
    assert SNAP not in res.bundle.model.feature_names  # the date is not a feature


def test_ablation_runs_in_temporal_mode(dated, tparams, assumptions):
    p = override(tparams, "ablation", "seeds", value=[1, 2])
    out = ablate(dated, p, assumptions)
    # the temporal split does not depend on the seed: both variants compare the same data
    assert [r["seed"] for r in out["per_seed"]] == [1, 2]


def test_missing_date_column_errors(sample_df, tparams):
    with pytest.raises(SplitError) as e:
        make_split(sample_df, tparams)
    assert e.value.code == "E_SPLIT_NO_TIME_AXIS"


def test_corrupt_date_rejected_by_validator_and_split(dated, tparams):
    bad = dated.copy()
    bad.loc[bad.index[5], SNAP] = "not-a-date"
    bad.loc[bad.index[6], SNAP] = ""
    rep = validate(bad, tparams)
    assert "E_SNAPSHOT_INVALID" in rep.errors
    with pytest.raises(SplitError):
        make_split(bad, tparams)
    assert "E_SNAPSHOT_INVALID" not in validate(dated, tparams).errors


@pytest.mark.parametrize(
    "mutate,msg",
    [
        (lambda p: p["schema"].update(snapshot_column="nonexistent"), "required_columns"),
        (lambda p: p["schema"].update(snapshot_column="customerID"), "id_column"),
    ],
)
def test_invalid_snapshot_column_config(tparams, tmp_path, mutate, msg):
    import copy

    p = copy.deepcopy(tparams)
    mutate(p)
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump(p), encoding="utf-8")
    with pytest.raises(ValueError, match=msg):
        load_params(f)


def test_snapshot_column_rejected_when_non_temporal(params, tmp_path):
    p = override(params, "schema", "required_columns", value=[*params["schema"]["required_columns"], SNAP])
    p["schema"]["snapshot_column"] = SNAP
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump(p), encoding="utf-8")
    with pytest.raises(ValueError, match="temporal"):
        load_params(f)
