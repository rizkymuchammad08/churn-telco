"""P4: monitoring.json covers exactly the model's features, and a changed feature set makes the check fail."""

import copy
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml
from conftest import override

from src.config import ROOT, load_params
from src.features.build import build_features, feature_names, feature_sources
from src.models import monitoring
from src.models.monitoring import MonitorError
from src.models.train import run

pytestmark = pytest.mark.p4

NAMES = feature_names(load_params())


def committed(params) -> dict:
    return monitoring.load(ROOT / params["paths"]["monitoring"])


def mutated(params, edit) -> dict:
    """Deep copy of the committed file after `edit(copy)`."""
    data = copy.deepcopy(committed(params))
    edit(data)
    return data


def relocked(params: dict) -> dict:
    """Copy of `params` whose monitoring.rule_lock matches the monitoring rule it carries."""
    p = copy.deepcopy(params)
    p["monitoring"]["rule_lock"] = monitoring.rule_hash(p)
    return p


def resealed(params, edit) -> dict:
    """Like `mutated`, but with content_hash recomputed, so only the edited content is wrong."""
    return monitoring.seal(mutated(params, edit))


def code_of(fn, *args) -> str:
    with pytest.raises(MonitorError) as e:
        fn(*args)
    return e.value.code


# --- the shipped file covers every model feature, nothing more or less -----------------------------
def test_committed_file_has_exactly_the_model_features(params):
    mon = committed(params)
    assert len(NAMES) > 0 and len(NAMES) == len(set(NAMES))
    assert set(mon["features"]) == set(feature_names(params))
    assert list(mon["features"]) == feature_names(params)  # generated in model order


def test_committed_file_has_the_same_source_columns_as_feature_sources(params):
    mon = committed(params)
    for name, sources in feature_sources(params).items():
        assert mon["features"][name]["sources"] == list(sources), name


def test_every_feature_has_valid_thresholds_equal_to_the_settings(params):
    mon = committed(params)
    cfg = params["monitoring"]
    for name, entry in mon["features"].items():
        assert 0 < entry["psi_warn"] < entry["psi_alert"], name
        assert (entry["psi_warn"], entry["psi_alert"]) == (cfg["psi_warn"], cfg["psi_alert"]), name


def test_committed_file_passes_both_validation_levels(params):
    mon = committed(params)
    monitoring.validate(mon)
    monitoring.check_alignment(mon, params)


def test_committed_file_is_strict_json_and_round_trips(params):
    mon = committed(params)
    assert json.loads(monitoring.dumps(mon)) == mon
    assert (ROOT / params["paths"]["monitoring"]).read_text(encoding="utf-8") == monitoring.dumps(mon)


# --- E_MONITOR_FEATURES: a feature without thresholds, or thresholds for a feature that is gone ------
@pytest.mark.parametrize("name", NAMES)
def test_dropping_any_feature_from_monitoring_is_caught(params, name):
    mon = mutated(params, lambda m: m["features"].pop(name))
    with pytest.raises(MonitorError) as e:
        monitoring.check_alignment(mon, params)
    assert e.value.code == "E_MONITOR_FEATURES"
    assert f"missing: ['{name}']" in e.value.message


def test_extra_feature_in_monitoring_is_caught(params):
    def add(m):
        m["features"]["ghost"] = copy.deepcopy(m["features"][NAMES[0]])

    with pytest.raises(MonitorError) as e:
        monitoring.check_alignment(mutated(params, add), params)
    assert e.value.code == "E_MONITOR_FEATURES" and "extra: ['ghost']" in e.value.message


def test_new_model_features_require_new_monitoring(params):
    # include_flagged adds tenure and TotalCharges to the model: the committed file no longer covers it.
    p = override(params, "leakage", "policy", value="include_flagged")
    assert {"tenure", "TotalCharges"} <= set(feature_names(p)) - set(NAMES)
    with pytest.raises(MonitorError) as e:
        monitoring.check_alignment(committed(params), p)
    assert e.value.code == "E_MONITOR_FEATURES"
    assert "tenure" in e.value.message and "TotalCharges" in e.value.message


def test_features_removed_from_the_model_require_new_monitoring(params):
    # Flagging InternetService drops every feature derived from it (see test_features.py).
    p = override(params, "leakage", "flagged_columns", value=["tenure", "TotalCharges", "InternetService"])
    gone = set(NAMES) - set(feature_names(p))
    assert {"has_internet", "InternetService_DSL"} <= gone
    with pytest.raises(MonitorError) as e:
        monitoring.check_alignment(committed(params), p)
    assert e.value.code == "E_MONITOR_FEATURES"
    assert all(f"'{n}'" in e.value.message for n in gone)


def test_feature_order_does_not_matter_but_membership_does(params):
    reordered = mutated(params, lambda m: m.update(features=dict(reversed(list(m["features"].items())))))
    monitoring.check_alignment(reordered, params)


# --- E_MONITOR_SOURCES / E_MONITOR_STALE / E_MONITOR_THRESHOLDS ----------------------------------------
def test_redefined_source_columns_are_caught(params):
    mon = mutated(params, lambda m: m["features"]["is_autopay"].update(sources=["Contract"]))
    assert code_of(monitoring.check_alignment, mon, params) == "E_MONITOR_SOURCES"


@pytest.mark.parametrize(
    "path,value",
    [
        (("monitoring", "psi_alert"), 0.3),
        (("monitoring", "psi_warn"), 0.05),
        (("monitoring", "n_bins"), 5),
        (("monitoring", "max_discrete_levels"), 4),
        (("monitoring", "psi_eps"), 0.001),
        (("monitoring", "min_rows"), 300),
    ],
)
def test_changing_a_monitoring_setting_makes_the_file_stale(params, path, value):
    p = relocked(override(params, *path, value=value))
    assert code_of(monitoring.check_alignment, committed(params), p) == "E_MONITOR_STALE"


@pytest.mark.parametrize(
    "warn,alert",
    [
        (None, 0.25),
        (0.1, None),
        (0.25, 0.1),
        (0.1, 0.1),
        (0.0, 0.25),
        (-0.1, 0.25),
        (float("nan"), 0.25),
        (0.1, float("inf")),
        (True, 0.25),
        ("0.1", 0.25),
    ],
)
def test_invalid_thresholds_on_a_feature_are_caught(params, warn, alert):
    def edit(m):
        entry = m["features"][NAMES[3]]
        for key, value in (("psi_warn", warn), ("psi_alert", alert)):
            if value is None:
                del entry[key]
            else:
                entry[key] = value

    assert code_of(monitoring.check_alignment, mutated(params, edit), params) == "E_MONITOR_THRESHOLDS"


def test_resealed_but_hand_edited_threshold_is_caught(params):
    mon = resealed(params, lambda m: m["features"][NAMES[3]].update(psi_alert=0.5))
    with pytest.raises(MonitorError) as e:
        monitoring.check_alignment(mon, params)
    assert e.value.code == "E_MONITOR_THRESHOLDS" and NAMES[3] in e.value.message


# --- E_MONITOR_REFERENCE / E_MONITOR_FILE -------------------------------------------------------------
@pytest.mark.parametrize(
    "edit",
    [
        lambda e: e.update(kind="ordinal"),
        lambda e: e.update(ref_props=e["ref_props"][:-1]),
        lambda e: e.update(ref_props=[*e["ref_props"][:-1], 0.5]),
        lambda e: e.update(ref_props=[-0.1, *e["ref_props"][1:]]),
        lambda e: e.update(ref_props=None),
        lambda e: e.update(levels=[1.0, 0.0]),
        lambda e: e.update(levels=[]),
    ],
)
def test_broken_reference_is_caught(params, edit):
    mon = mutated(params, lambda m: edit(m["features"][NAMES[0]]))
    assert code_of(monitoring.check_alignment, mon, params) == "E_MONITOR_REFERENCE"


def test_edges_must_increase(params):
    entry = committed(params)["features"]["MonthlyCharges"]
    assert entry["kind"] == "continuous"
    mon = mutated(params, lambda m: m["features"]["MonthlyCharges"].update(edges=entry["edges"][::-1]))
    assert code_of(monitoring.validate, mon) == "E_MONITOR_REFERENCE"


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m.pop("features"),
        lambda m: m.pop("settings"),
        lambda m: m.pop("reference"),
        lambda m: m.update(features={}),
        lambda m: m.update(features=[]),
        lambda m: m["settings"].pop("min_rows"),
        lambda m: m["reference"].update(train_data_hash=""),
        lambda m: m["features"][NAMES[0]].pop("ref_props"),
    ],
)
def test_incomplete_file_is_rejected(params, edit):
    assert code_of(monitoring.validate, mutated(params, edit)) == "E_MONITOR_FILE"


def test_load_reports_missing_and_unreadable_files(tmp_path):
    assert code_of(monitoring.load, tmp_path / "nope.json") == "E_MONITOR_FILE"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert code_of(monitoring.load, bad) == "E_MONITOR_FILE"


def test_write_then_load_round_trips(params, tmp_path):
    mon = committed(params)
    path = monitoring.write(mon, tmp_path / "sub" / "m.json")
    assert monitoring.load(path) == mon


# --- build_reference: from the train rows only, tied to the trained model -----------------------------
def test_run_returns_a_reference_that_passes_both_checks(base_run, fast):
    mon = base_run.monitoring
    assert list(mon["features"]) == feature_names(fast)
    monitoring.check_alignment(mon, fast)
    monitoring.check_model(mon, base_run.bundle)
    assert mon["reference"]["train_data_hash"] == base_run.hashes["train_data"]
    assert mon["reference"]["rows"] == len(base_run.split.train)
    assert json.loads(monitoring.dumps(mon)) == mon


def test_reference_uses_train_rows_only(sample_df, fast, base_run):
    X = build_features(sample_df, fast)
    train = X.loc[base_run.split.train]
    again = monitoring.build_reference(train, fast, train_hash=base_run.hashes["train_data"])
    assert again == base_run.monitoring
    everything = monitoring.build_reference(X, fast, train_hash=base_run.hashes["train_data"])
    assert everything["features"]["MonthlyCharges"] != base_run.monitoring["features"]["MonthlyCharges"]


def test_reference_is_deterministic_and_does_not_modify_input(sample_df, fast):
    X = build_features(sample_df, fast)
    before = X.copy()
    a = monitoring.build_reference(X, fast, train_hash="h")
    b = monitoring.build_reference(X, fast, train_hash="h")
    assert a == b
    assert X.equals(before)


def test_binary_features_are_monitored_by_level_and_charges_by_quantile(sample_df, fast):
    mon = monitoring.build_reference(build_features(sample_df, fast), fast, train_hash="h")
    f = mon["features"]
    assert f["Partner"]["kind"] == "discrete" and f["Partner"]["levels"] == [0.0, 1.0]
    assert len(f["Partner"]["ref_props"]) == 3
    assert f["Partner"]["ref_props"][-1] == 0.0  # no value outside the levels in train
    assert f["n_addons"]["kind"] == "discrete" and len(f["n_addons"]["levels"]) > 2
    assert f["MonthlyCharges"]["kind"] == "continuous"
    assert len(f["MonthlyCharges"]["edges"]) <= fast["monitoring"]["n_bins"] - 1


def test_the_discrete_limit_and_bin_count_come_from_params(sample_df, fast):
    p = override(fast, "monitoring", "max_discrete_levels", value=2)
    p = relocked(override(p, "monitoring", "n_bins", value=4))
    f = monitoring.build_reference(build_features(sample_df, p), p, train_hash="h")["features"]
    assert f["n_addons"]["kind"] == "continuous" and len(f["n_addons"]["ref_props"]) <= 4
    assert len(f["MonthlyCharges"]["edges"]) <= 3
    assert f["Partner"]["kind"] == "discrete"


def test_reference_rejects_empty_or_mismatched_features(sample_df, fast):
    X = build_features(sample_df, fast)
    with pytest.raises(MonitorError) as e:
        monitoring.build_reference(X.iloc[:0], fast, train_hash="h")
    assert e.value.code == "E_MONITOR_ROWS"
    with pytest.raises(MonitorError) as e:
        monitoring.build_reference(X.drop(columns="Partner"), fast, train_hash="h")
    assert e.value.code == "E_MONITOR_FEATURES"


# --- check_model: the saved model's features and train rows ---------------------------------------------
def test_check_model_catches_a_model_with_other_features(base_run):
    other = copy.copy(base_run.bundle)
    other.model = copy.copy(base_run.bundle.model)
    other.model.feature_names = [*base_run.bundle.model.feature_names, "tenure"]
    with pytest.raises(MonitorError) as e:
        monitoring.check_model(base_run.monitoring, other)
    assert e.value.code == "E_MONITOR_FEATURES" and "missing: ['tenure']" in e.value.message


def test_check_model_catches_a_model_trained_on_other_rows(base_run):
    other = copy.copy(base_run.bundle)
    other.model = copy.copy(base_run.bundle.model)
    other.model.train_hash = "0" * 64
    assert code_of(monitoring.check_model, base_run.monitoring, other) == "E_MONITOR_STALE"


# --- assess: the thresholds are used ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def train_X(sample_df, fast, base_run):
    return build_features(sample_df, fast).loc[base_run.split.train]


def test_train_rows_against_their_own_reference_are_clean(train_X, base_run):
    out = monitoring.assess(train_X, base_run.monitoring)
    assert list(out.index) == list(train_X.columns)
    assert (out["psi"].abs() < 1e-12).all() and (out["status"] == "ok").all()


def test_a_batch_from_the_same_population_does_not_alert(train_X, base_run):
    rng = np.random.default_rng(0)
    n = base_run.monitoring["settings"]["min_rows"]
    for _ in range(50):
        batch = train_X.iloc[rng.choice(len(train_X), n, replace=False)]
        assert (monitoring.assess(batch, base_run.monitoring)["status"] != "alert").all()


def test_shifted_features_are_flagged_and_others_stay_ok(train_X, base_run):
    batch = train_X.copy()
    batch["MonthlyCharges"] *= 1.5
    batch["Contract_Two_year"] = 1.0
    out = monitoring.assess(batch, base_run.monitoring)
    assert out.loc["MonthlyCharges", "status"] == "alert"
    assert out.loc["Contract_Two_year", "status"] == "alert"
    assert (out.drop(["MonthlyCharges", "Contract_Two_year"])["status"] == "ok").all()


def test_a_level_never_seen_in_training_is_flagged(train_X, base_run):
    batch = train_X.copy()
    batch["n_addons"] = float(base_run.monitoring["features"]["n_addons"]["levels"][-1] + 1)
    out = monitoring.assess(batch, base_run.monitoring)
    assert out.loc["n_addons", "status"] == "alert"


def test_warn_sits_between_the_two_thresholds(train_X, base_run):
    mon, feature = base_run.monitoring, "Partner"
    entry = mon["features"][feature]
    seen = {}
    for n_flipped in range(0, len(train_X) + 1, 5):
        batch = train_X.copy()
        batch.iloc[:n_flipped, batch.columns.get_loc(feature)] = 1.0
        row = monitoring.assess(batch, mon).loc[feature]
        seen.setdefault(row["status"], row["psi"])
    assert set(seen) == {"ok", "warn", "alert"}
    assert seen["ok"] < entry["psi_warn"] <= seen["warn"] < entry["psi_alert"] <= seen["alert"]


def test_assess_refuses_small_batches_and_wrong_columns(train_X, base_run):
    mon = base_run.monitoring
    small = train_X.iloc[: mon["settings"]["min_rows"] - 1]
    assert code_of(monitoring.assess, small, mon) == "E_MONITOR_ROWS"
    assert code_of(monitoring.assess, train_X.drop(columns="Partner"), mon) == "E_MONITOR_FEATURES"
    assert code_of(monitoring.assess, train_X.assign(extra=1.0), mon) == "E_MONITOR_FEATURES"
    bad = train_X.copy()
    bad.iloc[0, 0] = np.nan
    assert code_of(monitoring.assess, bad, mon) == "E_MONITOR_INPUT"


def test_assess_rejects_a_broken_reference(train_X, base_run):
    mon = copy.deepcopy(base_run.monitoring)
    mon["features"]["Partner"]["ref_props"] = [1.0]
    assert code_of(monitoring.assess, train_X, mon) == "E_MONITOR_REFERENCE"


# --- reproducibility: the same input and seed give the same file -----------------------------------------
def test_two_runs_with_the_same_seed_give_identical_monitoring(sample_df, fast, assumptions):
    a = run(sample_df, fast, assumptions, seed=7).monitoring
    b = run(sample_df, fast, assumptions, seed=7).monitoring
    assert monitoring.dumps(a) == monitoring.dumps(b)  # byte for byte
    for name, entry in a["features"].items():  # the parts that matter, spelled out
        other = b["features"][name]
        assert entry.get("levels") == other.get("levels") and entry.get("edges") == other.get("edges")
        assert entry["ref_props"] == other["ref_props"]
        assert (entry["psi_warn"], entry["psi_alert"]) == (other["psi_warn"], other["psi_alert"])
    assert a["reference"]["train_data_hash"] == b["reference"]["train_data_hash"]
    assert a["rule_hash"] == b["rule_hash"] and a["content_hash"] == b["content_hash"]


def test_a_different_seed_changes_the_train_rows_and_therefore_the_file(sample_df, fast, assumptions):
    a = run(sample_df, fast, assumptions, seed=7).monitoring
    c = run(sample_df, fast, assumptions, seed=8).monitoring
    assert a["reference"]["train_data_hash"] != c["reference"]["train_data_hash"]
    assert a["content_hash"] != c["content_hash"]
    assert a["rule_hash"] == c["rule_hash"]  # the rule is the same, only the rows differ


def test_row_order_of_the_input_does_not_change_the_file(sample_df, fast, assumptions):
    a = run(sample_df, fast, assumptions).monitoring
    shuffled = sample_df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert monitoring.dumps(run(shuffled, fast, assumptions).monitoring) == monitoring.dumps(a)


def test_committed_file_is_what_the_code_builds_from_the_raw_data(params):
    from src.data.load import load_raw
    from src.features.build import build_labels
    from src.models.split import make_split
    from src.models.train import train_data_hash

    df = load_raw("raw", params)
    X, y = build_features(df, params), build_labels(df, params)
    train = make_split(df, params).train
    ref = monitoring.build_reference(
        X.loc[train], params, train_hash=train_data_hash(X.loc[train], y.loc[train])
    )
    rebuilt = monitoring.seal({"data": "raw", **ref})
    assert monitoring.dumps(rebuilt) == (ROOT / params["paths"]["monitoring"]).read_text(encoding="utf-8")


# --- integrity: a changed threshold or bin is detected -----------------------------------------------------
def test_shipped_lock_matches_shipped_rule(params):
    assert params["monitoring"]["rule_lock"] == monitoring.rule_hash(params)
    assert monitoring.verify_lock(params) == monitoring.rule_hash(params)
    assert committed(params)["rule_hash"] == monitoring.rule_hash(params)


def test_rule_hash_covers_the_rule_and_nothing_else(params):
    base = monitoring.rule_hash(params)
    assert len(base) == 64
    assert monitoring.rule_hash(override(params, "monitoring", "rule_lock", value="x")) == base
    assert monitoring.rule_hash(override(params, "seed", value=1)) == base
    assert monitoring.rule_hash(override(params, "tuning", "margin_se", value=2.0)) == base


@pytest.mark.parametrize(
    "key,value",
    [
        ("psi_warn", 0.05),
        ("psi_alert", 0.3),
        ("n_bins", 5),
        ("max_discrete_levels", 4),
        ("psi_eps", 0.001),
        ("min_rows", 300),
    ],
)
def test_changing_the_rule_without_the_lock_stops_everything(fast, sample_df, assumptions, key, value):
    p = override(fast, "monitoring", key, value=value)
    assert monitoring.rule_hash(p) != monitoring.rule_hash(fast)
    assert code_of(monitoring.verify_lock, p) == "E_MONITOR_LOCK"
    assert code_of(monitoring.check_alignment, committed(p), p) == "E_MONITOR_LOCK"
    X = build_features(sample_df, p)
    with pytest.raises(MonitorError) as e:
        monitoring.build_reference(X, p, train_hash="h")
    assert e.value.code == "E_MONITOR_LOCK"
    with pytest.raises(MonitorError) as e:
        run(sample_df, p, assumptions)  # stops before any model is fitted
    assert e.value.code == "E_MONITOR_LOCK"


def test_relocking_a_changed_rule_makes_the_old_file_stale_not_locked(params):
    p = relocked(override(params, "monitoring", "n_bins", value=5))
    assert monitoring.verify_lock(p) == monitoring.rule_hash(p) != monitoring.rule_hash(params)
    assert code_of(monitoring.check_alignment, committed(params), p) == "E_MONITOR_STALE"


def _edge(m):
    m["features"]["MonthlyCharges"]["edges"][0] += 1e-6


def _shares(m):
    props = m["features"]["Partner"]["ref_props"]
    props[0], props[1] = props[1], props[0]


@pytest.mark.parametrize(
    "edit",
    [
        _edge,  # a bin edge moved a little
        _shares,  # shares swapped; they still sum to one
        lambda m: m["features"]["Partner"].update(psi_warn=0.2),  # a threshold, still ordered
        lambda m: m["settings"].update(n_bins=7),
        lambda m: m.update(rule_hash="0" * 64),
        lambda m: m["reference"].update(rows=1),
        lambda m: m["reference"].update(train_data_hash="0" * 64),
        lambda m: m.update(data_source="synthetic"),
        lambda m: m.pop("content_hash"),
        lambda m: m.update(content_hash="0" * 64),
    ],
)
def test_any_edit_to_the_file_is_caught_by_the_content_hash(params, edit):
    assert code_of(monitoring.validate, mutated(params, edit)) == "E_MONITOR_HASH"


def test_an_edit_is_accepted_again_only_after_resealing(params):
    mon = mutated(params, _edge)
    assert code_of(monitoring.validate, mon) == "E_MONITOR_HASH"
    monitoring.validate(monitoring.seal(mon))


def test_hand_edited_file_fails_the_full_alignment_check(params):
    mon = mutated(params, lambda m: m["features"][NAMES[3]].update(psi_alert=0.5))
    assert code_of(monitoring.check_alignment, mon, params) == "E_MONITOR_HASH"


def test_committed_file_is_sealed_and_ties_to_the_committed_metrics(params):
    mon = committed(params)
    assert mon["content_hash"] == monitoring.content_hash(mon)
    metrics = json.loads((ROOT / params["paths"]["metrics"]).read_text(encoding="utf-8"))
    assert mon["reference"]["train_data_hash"] == metrics["hashes"]["train_data"]


def test_seal_is_idempotent_and_ignores_key_order(base_run):
    m = base_run.monitoring
    assert list(m)[-1] == "content_hash" and m["content_hash"] == monitoring.content_hash(m)
    assert monitoring.seal(m) == m
    assert monitoring.seal(dict(reversed(list(m.items())))) == m


def test_write_lock_changes_only_the_monitoring_lock_line(params, tmp_path):
    text = (ROOT / "params.yaml").read_text(encoding="utf-8")
    f = tmp_path / "params.yaml"
    f.write_text(text, encoding="utf-8")
    assert monitoring.write_lock(f, "a" * 64)
    changed = [(a, b) for a, b in zip(text.splitlines(), f.read_text().splitlines(), strict=True) if a != b]
    assert len(changed) == 1 and "a" * 64 in changed[0][1] and "rule_lock" in changed[0][0]
    new = load_params(f)
    assert new["monitoring"]["rule_lock"] == "a" * 64
    assert new["tuning"]["rule_lock"] == params["tuning"]["rule_lock"]  # the tuning lock is not touched
    f.write_text(text.replace("\nmonitoring:", "\nmonit:"), encoding="utf-8")
    assert monitoring.write_lock(f, "b" * 64) is False


# --- configuration -----------------------------------------------------------------------------------------
def _write(tmp_path, data):
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump(data), encoding="utf-8")
    return f


@pytest.mark.parametrize(
    "path,bad,exc",
    [
        (("monitoring", "psi_warn"), 0, ValueError),
        (("monitoring", "psi_warn"), 0.3, ValueError),  # not below psi_alert
        (("monitoring", "psi_alert"), 0.1, ValueError),  # not above psi_warn
        (("monitoring", "psi_warn"), "0.1", TypeError),
        (("monitoring", "n_bins"), 1, ValueError),
        (("monitoring", "n_bins"), 10.5, TypeError),
        (("monitoring", "max_discrete_levels"), 1, ValueError),
        (("monitoring", "psi_eps"), 0, ValueError),
        (("monitoring", "psi_eps"), 1.0, ValueError),
        (("monitoring", "min_rows"), 5, ValueError),  # below n_bins
        (("monitoring", "min_rows"), 200.5, TypeError),
        (("monitoring", "rule_lock"), 5, TypeError),
        (("paths", "monitoring"), 3, TypeError),
    ],
)
def test_monitoring_params_are_validated(params, tmp_path, path, bad, exc):
    with pytest.raises(exc):
        load_params(_write(tmp_path, override(params, *path, value=bad)))


@pytest.mark.parametrize(
    "path",
    [("monitoring",), ("monitoring", "psi_alert"), ("monitoring", "rule_lock"), ("paths", "monitoring")],
)
def test_monitoring_params_are_required(params, tmp_path, path):
    data = copy.deepcopy(params)
    node = data
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    with pytest.raises(KeyError):
        load_params(_write(tmp_path, data))


def test_monitoring_is_a_report_file_next_to_the_other_numbers(params):
    assert params["paths"]["monitoring"] == "reports/monitoring.json"
    assert (ROOT / params["paths"]["monitoring"]).is_file()


# --- command line and Makefile ------------------------------------------------------------------------------
def _project(root):
    shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("params.yaml", "assumptions.yaml"):
        shutil.copy(ROOT / f, root / f)
    (root / "data").mkdir()
    shutil.copy(ROOT / "data" / "sample.csv", root / "data" / "sample.csv")
    return root


def _cli(root, module, *args):
    cmd = [sys.executable, "-m", f"src.models.{module}", *args]
    return subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A project copy in which `train --sample` has run (writes monitoring.json and model.joblib)."""
    root = _project(tmp_path_factory.mktemp("monitoring_cli"))
    r = _cli(root, "train", "--sample")
    assert r.returncode == 0, r.stdout + r.stderr
    return root, r.stdout


def test_train_writes_monitoring_json_next_to_metrics(trained):
    root, out = trained
    params = load_params(root / "params.yaml")
    mon = monitoring.load(root / params["paths"]["monitoring"])
    metrics = json.loads((root / params["paths"]["metrics"]).read_text(encoding="utf-8"))
    assert mon["data"] == metrics["data"] == "sample"
    assert mon["reference"]["train_data_hash"] == metrics["hashes"]["train_data"]
    assert list(mon["features"]) == feature_names(params)
    monitoring.check_alignment(mon, params)
    assert f"monitoring: {len(mon['features'])} features" in out


def test_cli_passes_against_params_and_saved_model(trained):
    r = _cli(trained[0], "monitoring", "--require-model")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "MONITORING OK" in r.stdout and "params and saved model" in r.stdout


def test_cli_fails_when_the_feature_set_changes_until_train_is_rerun(trained, tmp_path):
    root = _project(tmp_path / "changed")
    shutil.copytree(trained[0] / "reports", root / "reports")
    shutil.copytree(trained[0] / "artifacts", root / "artifacts")
    assert _cli(root, "monitoring").returncode == 0
    text = (root / "params.yaml").read_text(encoding="utf-8")
    assert "policy: exclude" in text
    changed = text.replace("policy: exclude", "policy: include_flagged")
    (root / "params.yaml").write_text(changed, encoding="utf-8")
    r = _cli(root, "monitoring")
    assert r.returncode == 1
    assert "E_MONITOR_FEATURES" in r.stdout and "tenure" in r.stdout and "TotalCharges" in r.stdout
    assert "MONITORING OK" not in r.stdout


def test_cli_checks_the_saved_model_when_one_exists(trained, tmp_path):
    root = _project(tmp_path / "other_rows")
    shutil.copytree(trained[0] / "reports", root / "reports")
    shutil.copytree(trained[0] / "artifacts", root / "artifacts")
    path = root / "reports" / "monitoring.json"
    mon = json.loads(path.read_text(encoding="utf-8"))
    mon["reference"]["train_data_hash"] = "0" * 64
    path.write_text(monitoring.dumps(monitoring.seal(mon)), encoding="utf-8")  # a valid file, other rows
    r = _cli(root, "monitoring")
    assert r.returncode == 1 and "E_MONITOR_STALE" in r.stdout


def test_cli_without_a_model_checks_params_only_unless_the_model_is_required(trained, tmp_path):
    root = _project(tmp_path / "no_model")
    shutil.copytree(trained[0] / "reports", root / "reports")
    lenient = _cli(root, "monitoring")
    assert lenient.returncode == 0 and "params only" in lenient.stdout
    strict = _cli(root, "monitoring", "--require-model")
    assert strict.returncode == 1 and "E_MONITOR_FILE" in strict.stdout


def test_cli_reports_a_missing_file(tmp_path):
    root = _project(tmp_path / "empty")
    r = _cli(root, "monitoring")
    assert r.returncode == 1 and "E_MONITOR_FILE" in r.stdout


def test_train_twice_in_clean_projects_gives_identical_monitoring_json(trained, tmp_path):
    other = _project(tmp_path / "second")
    r = _cli(other, "train", "--sample")
    assert r.returncode == 0, r.stdout + r.stderr
    first = (trained[0] / "reports" / "monitoring.json").read_bytes()
    assert (other / "reports" / "monitoring.json").read_bytes() == first


def test_cli_lock_flow_from_changed_threshold_to_rebuilt_file(trained, tmp_path):
    root = _project(tmp_path / "locked")
    shutil.copytree(trained[0] / "reports", root / "reports")
    shutil.copytree(trained[0] / "artifacts", root / "artifacts")
    before = (root / "reports" / "monitoring.json").read_bytes()
    params_file = root / "params.yaml"
    text = params_file.read_text(encoding="utf-8")
    assert "psi_alert: 0.25" in text
    params_file.write_text(text.replace("psi_alert: 0.25", "psi_alert: 0.3"), encoding="utf-8")

    # 1. threshold changed, lock not: check and training both stop, and the file is untouched
    check = _cli(root, "monitoring")
    assert check.returncode == 1 and "E_MONITOR_LOCK" in check.stdout
    stopped = _cli(root, "train", "--sample")
    assert stopped.returncode == 1 and "E_MONITOR_LOCK" in stopped.stdout
    assert (root / "reports" / "monitoring.json").read_bytes() == before

    # 2. lock rewritten: only the lock line changes, and the old file is now stale
    assert _cli(root, "monitoring", "--write-lock").returncode == 0
    now = params_file.read_text(encoding="utf-8")
    assert "psi_alert: 0.3" in now and now != text
    stale = _cli(root, "monitoring")
    assert stale.returncode == 1 and "E_MONITOR_STALE" in stale.stdout

    # 3. training writes the file again with the new threshold on every feature
    rebuilt = _cli(root, "train", "--sample")
    assert rebuilt.returncode == 0, rebuilt.stdout + rebuilt.stderr
    mon = monitoring.load(root / "reports" / "monitoring.json")
    assert {e["psi_alert"] for e in mon["features"].values()} == {0.3}
    assert _cli(root, "monitoring", "--require-model").returncode == 0


def test_cli_catches_a_hand_edited_bin(trained, tmp_path):
    root = _project(tmp_path / "edited")
    shutil.copytree(trained[0] / "reports", root / "reports")
    path = root / "reports" / "monitoring.json"
    mon = json.loads(path.read_text(encoding="utf-8"))
    mon["features"]["MonthlyCharges"]["edges"][0] += 1.0
    path.write_text(json.dumps(mon, indent=2) + "\n", encoding="utf-8")
    r = _cli(root, "monitoring")
    assert r.returncode == 1 and "E_MONITOR_HASH" in r.stdout


def _make(*args):
    return subprocess.run(["make", *args], cwd=ROOT, capture_output=True, text=True)


def test_make_runs_the_monitoring_check_after_training():
    out = _make("-n", "all").stdout
    assert out.index("src.models.train") < out.index("src.models.monitoring --require-model")
    assert "--sample" not in out
    sample = _make("-n", "all-sample").stdout
    assert "src.models.train --sample" in sample and "src.models.monitoring" in sample
    check = _make("-n", "check-p4").stdout
    assert check.index("src.models.train") < check.index("monitoring")
