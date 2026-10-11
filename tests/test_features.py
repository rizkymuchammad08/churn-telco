"""P3: feature matrix and labels match hand computation, columns match the contract, deterministic."""

import copy
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.config import ROOT, load_params
from src.data.load import data_hash, load_raw
from src.features.build import (
    FeatureError,
    build_features,
    build_labels,
    feature_names,
    feature_sources,
)
from src.features.spec import render

pytestmark = pytest.mark.p3

P = load_params()
P_INCLUDE = {**copy.deepcopy(P), "leakage": {**P["leakage"], "policy": "include_flagged"}}

# Success marker printed by the CLI.
CLI_OK_MARKER = "FEATURES OK"

BASE = {
    "customerID": "A",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 50.0,
    "TotalCharges": 600.0,
    "Churn": "No",
}
ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

ROW_B = {
    **BASE,
    "customerID": "B",
    "gender": "Male",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "Yes",
    "tenure": 0,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "Fiber optic",
    **{c: "Yes" for c in ADDONS},
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 99.5,
    "TotalCharges": np.nan,
    "Churn": "Yes",
}
ROW_C = {
    **BASE,
    "customerID": "C",
    "MultipleLines": "Yes",
    "InternetService": "No",
    **{c: "No internet service" for c in ADDONS},
    "Contract": "One year",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 20.25,
    "tenure": 5,
    "TotalCharges": 100.0,
}
ROW_D = {**BASE, "customerID": "D", "PaymentMethod": "Mailed check", "tenure": 30, "TotalCharges": 1500.0}

SMALL = pd.DataFrame([BASE, ROW_B, ROW_C, ROW_D])[P["schema"]["required_columns"]]

# Hand computation, per customer, for the exclude policy (without tenure and TotalCharges).
ZERO = {n: 0.0 for n in feature_names(P)}
EXPECTED = {
    "A": {
        **ZERO,
        "Partner": 1,
        "PhoneService": 1,
        "PaperlessBilling": 1,
        "OnlineSecurity": 1,
        "OnlineBackup": 1,
        "InternetService_DSL": 1,
        "MonthlyCharges": 50.0,
        "n_addons": 2,
        "has_internet": 1,
    },
    "B": {
        **ZERO,
        "is_male": 1,
        "SeniorCitizen": 1,
        "Dependents": 1,
        **{c: 1 for c in ADDONS},
        "InternetService_Fiber_optic": 1,
        "Contract_Two_year": 1,
        "PaymentMethod_Credit_card_automatic": 1,
        "MonthlyCharges": 99.5,
        "n_addons": 6,
        "has_internet": 1,
        "is_autopay": 1,
    },
    "C": {
        **ZERO,
        "Partner": 1,
        "PhoneService": 1,
        "PaperlessBilling": 1,
        "MultipleLines": 1,
        "Contract_One_year": 1,
        "PaymentMethod_Bank_transfer_automatic": 1,
        "MonthlyCharges": 20.25,
        "is_autopay": 1,
    },
    "D": {
        **ZERO,
        "Partner": 1,
        "PhoneService": 1,
        "PaperlessBilling": 1,
        "OnlineSecurity": 1,
        "OnlineBackup": 1,
        "InternetService_DSL": 1,
        "PaymentMethod_Mailed_check": 1,
        "MonthlyCharges": 50.0,
        "n_addons": 2,
        "has_internet": 1,
    },
}


@pytest.fixture(scope="module")
def sample():
    return load_raw("sample", P)


# --- matches hand computation ------------------------------------------------
def test_features_match_hand_computation():
    X = build_features(SMALL, P)
    for cid, want in EXPECTED.items():
        got = X.loc[cid]
        for name, value in want.items():
            assert got[name] == pytest.approx(value), f"{cid}.{name}"


def test_flagged_columns_included_only_by_policy():
    X = build_features(SMALL, P_INCLUDE)
    assert X.loc["A", "tenure"] == 12 and X.loc["A", "TotalCharges"] == 600.0
    assert X.loc["D", "tenure"] == 30 and X.loc["D", "TotalCharges"] == 1500.0
    assert X.loc["B", "tenure"] == 0 and X.loc["B", "TotalCharges"] == 0.0  # NaN at tenure 0 becomes 0
    for col in ("tenure", "TotalCharges"):
        assert col not in build_features(SMALL, P).columns


def test_exclude_drops_every_feature_derived_from_flagged_columns():
    flagged = set(P["leakage"]["flagged_columns"])
    for name, sources in feature_sources(P).items():
        assert not set(sources) & flagged, name
    # Flagging another column also drops the features derived from it.
    p = {**copy.deepcopy(P), "leakage": {**P["leakage"], "flagged_columns": ["InternetService"]}}
    names = feature_names(p)
    assert "has_internet" not in names and "InternetService_DSL" not in names
    assert "n_addons" in names  # not derived from InternetService


# --- output columns match the contract --------------------------------------
def test_output_columns_match_declared_names_and_order():
    X = build_features(SMALL, P)
    assert list(X.columns) == feature_names(P)
    assert X.index.name == P["schema"]["id_column"]
    assert list(X.index) == ["A", "B", "C", "D"]
    assert (X.dtypes == "float64").all()


def test_output_columns_do_not_depend_on_data(sample):
    one = build_features(SMALL.iloc[:1], P)
    many = build_features(sample, P)
    assert list(one.columns) == list(many.columns) == feature_names(P)


def test_no_nan_or_inf_on_real_sample(sample):
    X = build_features(sample, P)
    assert np.isfinite(X.to_numpy()).all()
    assert len(X) == len(sample)


def test_feature_spec_doc_equals_generator_and_names():
    committed = (ROOT / "docs" / "feature_spec.md").read_text(encoding="utf-8")
    assert committed == render(P)
    rows = [ln.split("`")[1] for ln in committed.splitlines() if ln.startswith("| `")]
    assert rows == feature_names(P)


# --- deterministic -----------------------------------------------------------
def test_features_deterministic(sample):
    a, b = build_features(sample, P), build_features(sample, P)
    assert a.equals(b)
    assert data_hash(a.reset_index()) == data_hash(b.reset_index())


def test_features_do_not_depend_on_row_order(sample):
    shuffled = sample.sample(frac=1.0, random_state=7)
    a = build_features(sample, P)
    b = build_features(shuffled, P).loc[a.index]
    assert a.equals(b)
    assert data_hash(a.reset_index()) == data_hash(build_features(shuffled, P).reset_index())


def test_build_does_not_mutate_input():
    before = SMALL.copy(deep=True)
    build_features(SMALL, P)
    build_labels(SMALL, P)
    pd.testing.assert_frame_equal(SMALL, before)


def test_each_row_depends_only_on_itself(sample):
    X = build_features(sample, P)
    sub = build_features(sample.iloc[10:20], P)
    assert X.loc[sub.index].equals(sub)


# --- rejections --------------------------------------------------------------
def test_unknown_category_rejected():
    bad = SMALL.copy()
    bad.loc[0, "Contract"] = "Three year"
    with pytest.raises(FeatureError) as e:
        build_features(bad, P)
    assert e.value.code == "E_FEATURE_CATEGORY"


def test_missing_column_rejected():
    with pytest.raises(FeatureError) as e:
        build_features(SMALL.drop(columns=["Contract"]), P)
    assert e.value.code == "E_FEATURE_COLUMNS_MISSING"


def test_duplicate_id_rejected():
    dup = pd.concat([SMALL, SMALL.iloc[:1]], ignore_index=True)
    with pytest.raises(FeatureError) as e:
        build_features(dup, P)
    assert e.value.code == "E_FEATURE_ID"


# --- labels (path A: label = Churn == positive_label, no window) -------------
def test_labels_match_hand_count():
    y = build_labels(SMALL, P)
    assert y.to_dict() == {"A": 0, "B": 1, "C": 0, "D": 0}
    assert y.name == "label" and y.index.name == P["schema"]["id_column"]


def test_labels_binary_no_nan_unique_ids_and_deterministic(sample):
    y = build_labels(sample, P)
    assert set(y.unique()) <= {0, 1} and y.notna().all() and y.dtype == "int64"
    assert not y.index.duplicated().any()
    assert y.equals(build_labels(sample, P))
    assert y.sum() == (sample["Churn"] == "Yes").sum()


def test_label_domain_rejected():
    bad = SMALL.copy()
    bad.loc[0, "Churn"] = "Maybe"
    with pytest.raises(FeatureError) as e:
        build_labels(bad, P)
    assert e.value.code == "E_LABEL_DOMAIN"


# --- CLI ---------------------------------------------------------------------
def test_cli_sample_writes_artifacts_and_passes(isolated_project):
    root = isolated_project
    r = subprocess.run(
        [sys.executable, "-m", "src.features.build", "--sample"], cwd=root, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert CLI_OK_MARKER in r.stdout
    X = pd.read_csv(root / P["paths"]["artifacts_dir"] / "features.csv")
    y = pd.read_csv(root / P["paths"]["artifacts_dir"] / "labels.csv")
    assert list(X.columns) == [P["schema"]["id_column"], *feature_names(P)]
    assert list(y.columns) == [P["schema"]["id_column"], "label"]
    assert X[P["schema"]["id_column"]].equals(y[P["schema"]["id_column"]])


# --- manifest ----------------------------------------------------------------
def test_handoff_p3_manifest_complete_and_consistent():
    import yaml

    m = yaml.safe_load((ROOT / "handoff/p3.yaml").read_text(encoding="utf-8"))
    for k in (
        "part",
        "adapter",
        "files",
        "yaml_keys_consumed",
        "yaml_keys_produced",
        "public_api",
        "check",
        "open_issues",
    ):
        assert k in m
    assert m["part"] == "P3" and m["check"] == "make check-p3" and m["open_issues"] == []
    for f in m["files"]:
        assert (ROOT / f).exists(), f
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    assert set(m["yaml_keys_consumed"]["params"]) <= set(params)
    assert m["yaml_keys_produced"] == {"params": [], "assumptions": []}


def test_p3_does_not_own_p1_or_p2_files():
    import yaml

    owned = {}
    for part in ("p1", "p2", "p3"):
        owned[part] = set(
            yaml.safe_load((ROOT / f"handoff/{part}.yaml").read_text(encoding="utf-8"))["files"]
        )
    assert not owned["p3"] & (owned["p1"] | owned["p2"])
