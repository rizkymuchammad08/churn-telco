"""P4: light hyperparameter tuning. The decision rule is locked, uses train rows only, and is applied as written."""

import shutil
import subprocess
import sys

import pytest
import yaml
from conftest import override, relock

from src.config import ROOT, load_params
from src.models.train import run
from src.models.tuning import TuneError, _decide, candidates, rule_hash, tune, verify_lock
from src.reporting.render import _tuning

pytestmark = pytest.mark.p4


def _write(tmp_path, d):
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump(d), encoding="utf-8")
    return f


# --- the lock ------------------------------------------------------------------
def test_shipped_lock_matches_shipped_rule(params):
    assert params["tuning"]["enabled"] is True
    assert params["tuning"]["rule_lock"] == rule_hash(params)
    assert verify_lock(params) == rule_hash(params)


@pytest.mark.parametrize(
    "path,value",
    [
        (("tuning", "margin_se"), 2.0),
        (("tuning", "seed_offset"), 7),
        (("tuning", "grids", "logistic_regression", "C"), [0.1, 1.0]),
        (("tuning", "grids", "gradient_boosting", "max_depth"), [3]),
        (("split", "cv_folds"), 4),
        (("models", "primary_metric"), "lift_at_k"),
        (("models", "class_weight"), "none"),
        (("models", "hyperparams", "gradient_boosting", "n_estimators"), 99),
    ],
)
def test_every_part_of_the_rule_changes_the_hash(params, path, value):
    assert rule_hash(override(params, *path, value=value)) != rule_hash(params)


def test_hash_ignores_the_lock_itself_and_unrelated_keys(params):
    assert rule_hash(override(params, "tuning", "rule_lock", value="x")) == rule_hash(params)
    assert rule_hash(override(params, "calibration", "method", value="sigmoid")) == rule_hash(params)


def test_stale_lock_stops_the_run(sample_df, fast_tuned, assumptions, base_split):
    stale = override(fast_tuned, "tuning", "margin_se", value=2.0)  # rule changed, lock not rewritten
    with pytest.raises(TuneError) as e:
        run(sample_df, stale, assumptions, split=base_split)
    assert e.value.code == "E_TUNE_LOCK"
    run(sample_df, relock(stale), assumptions, split=base_split)  # after an explicit re-lock it runs


# --- the decision rule -----------------------------------------------------------
def test_decide_replaces_default_when_gain_exceeds_margin():
    default = [0.50, 0.50, 0.50, 0.50, 0.50]
    better = [0.52, 0.51, 0.53, 0.52, 0.52]
    d = _decide([default, better], default_idx=0, margin_se=1.0)
    assert d["winner_index"] == 1 and d["replaced"] is True
    assert d["gain"] > d["margin"] > 0


def test_decide_keeps_default_when_gain_is_within_noise():
    default = [0.50, 0.50, 0.50, 0.50, 0.50]
    noisy = [0.55, 0.47, 0.54, 0.48, 0.53]  # positive mean gain, but smaller than its standard error
    d = _decide([default, noisy], default_idx=0, margin_se=1.0)
    assert d["winner_index"] == 1 and d["gain"] > 0
    assert d["gain"] <= d["margin"] and d["replaced"] is False
    # The same scores with no margin would have replaced the default: the margin is what decides.
    assert _decide([default, noisy], default_idx=0, margin_se=0.0)["replaced"] is True


def test_decide_tie_goes_to_earlier_candidate_and_never_replaces():
    same = [0.5, 0.6, 0.55]
    d = _decide([same, same], default_idx=1, margin_se=1.0)
    assert d["winner_index"] == 0 and d["gain"] == 0 and d["replaced"] is False


def test_decide_default_is_best_means_no_replacement():
    d = _decide([[0.6, 0.6, 0.6], [0.5, 0.5, 0.5]], default_idx=0, margin_se=1.0)
    assert d["winner_index"] == 0 and d["replaced"] is False


# --- tune() ---------------------------------------------------------------------
def test_candidates_follow_grid_order(fast_tuned):
    assert candidates("logistic_regression", fast_tuned) == [{"C": 0.1}, {"C": 1.0}]
    assert candidates("prior_rate", fast_tuned) == []


def test_disabled_tuning_is_a_no_op(fast):
    def boom(*a):
        raise AssertionError("must not score anything")

    eff, report = tune(fast, boom, 42)
    assert eff is fast and report == {"enabled": False}


def test_tune_uses_offset_seed_and_applies_winner(fast_tuned):
    calls = []

    def score(name, p, fold_seed):
        calls.append((name, fold_seed))
        hp = p["models"]["hyperparams"][name]
        good = hp.get("C") == 0.1 or hp.get("max_depth") == 2
        return [0.9, 0.9, 0.9] if good else [0.5, 0.5, 0.5]

    before = repr(fast_tuned)
    eff, report = tune(fast_tuned, score, 42)
    offset = fast_tuned["tuning"]["seed_offset"]
    assert {s for _, s in calls} == {42 + offset}
    assert len(calls) == 4  # two models with two candidates each; prior_rate has no grid
    assert eff["models"]["hyperparams"]["logistic_regression"]["C"] == 0.1
    assert eff["models"]["hyperparams"]["gradient_boosting"]["max_depth"] == 2
    assert repr(fast_tuned) == before  # the input params are not modified
    assert report["models"]["logistic_regression"]["replaced"] is True
    assert report["rule_hash"] == rule_hash(fast_tuned)


def test_tune_keeps_defaults_when_nothing_beats_them(fast_tuned):
    eff, report = tune(fast_tuned, lambda n, p, s: [0.5, 0.5, 0.5], 42)
    assert eff["models"] == fast_tuned["models"]
    assert not any(m["replaced"] for m in report["models"].values())


# --- the pipeline ---------------------------------------------------------------
def test_run_records_the_tuning_block(sample_df, fast_tuned, assumptions, base_split, base_run):
    r = run(sample_df, fast_tuned, assumptions, split=base_split)
    t = r.metrics["tuning"]
    assert t["enabled"] is True and t["rule_hash"] == rule_hash(fast_tuned)
    assert set(t["models"]) == {"logistic_regression", "gradient_boosting"}
    for m in t["models"].values():
        assert len(m["candidates"]) == 2 and sum(c["is_default"] for c in m["candidates"]) == 1
    chosen = t["models"].get(r.selected, {}).get("chosen")
    if chosen:
        used = r.bundle.meta["hyperparams"]
        assert all(used[k] == v for k, v in chosen.items())
    # the locked rule is part of the model hash, so a tuned model never shares a hash with an untuned one
    assert r.hashes["model"] != base_run.hashes["model"]
    assert base_run.metrics["tuning"] == {"enabled": False}


def test_test_rows_do_not_influence_tuning(sample_df, fast_tuned, assumptions, base_split):
    base = run(sample_df, fast_tuned, assumptions, split=base_split)
    df = sample_df.copy()
    t = df["customerID"].isin(base_split.test + base_split.calib)
    df.loc[t, "Churn"] = df.loc[t, "Churn"].map({"Yes": "No", "No": "Yes"})
    r = run(df, fast_tuned, assumptions, split=base_split)
    assert r.metrics["tuning"] == base.metrics["tuning"]
    assert r.hashes["model"] == base.hashes["model"] and r.cv_scores == base.cv_scores


def test_train_labels_do_influence_tuning(sample_df, fast_tuned, assumptions, base_split):
    """Negative control for the test above: the tuning scores do come from the train rows."""
    base = run(sample_df, fast_tuned, assumptions, split=base_split)
    df = sample_df.copy()
    t = df["customerID"].isin(base_split.train[:200])
    df.loc[t, "Churn"] = df.loc[t, "Churn"].map({"Yes": "No", "No": "Yes"})
    r = run(df, fast_tuned, assumptions, split=base_split)
    assert r.metrics["tuning"]["models"] != base.metrics["tuning"]["models"]


# --- configuration checks ---------------------------------------------------------
@pytest.mark.parametrize(
    "mutate,exc",
    [
        (lambda p: p["tuning"]["grids"]["logistic_regression"].update(C=[0.1, 0.5]), ValueError),  # no default
        (lambda p: p["tuning"]["grids"]["logistic_regression"].update(C=[]), ValueError),
        (lambda p: p["tuning"]["grids"]["logistic_regression"].update(C=[1.0, 1.0]), ValueError),
        (lambda p: p["tuning"]["grids"]["logistic_regression"].update(C=[-1.0, 1.0]), ValueError),
        (lambda p: p["tuning"]["grids"]["gradient_boosting"].update(learning_rate=[0.1, 1.5]), ValueError),
        (lambda p: p["tuning"]["grids"]["logistic_regression"].update(C=["1"]), TypeError),
        (lambda p: p["tuning"]["grids"]["logistic_regression"].update(depth=[1]), KeyError),
        (lambda p: p["tuning"].update(method="random"), ValueError),
        (lambda p: p["tuning"].update(seed_offset=0), ValueError),
        (lambda p: p["tuning"].update(margin_se=-1.0), ValueError),
        (lambda p: p["tuning"].update(grids={}), ValueError),
        (lambda p: p["tuning"].pop("rule_lock"), KeyError),
        (lambda p: p["tuning"].update(extra=1), KeyError),
    ],
)
def test_tuning_config_is_validated(params, tmp_path, mutate, exc):
    p = override(params, "seed", value=params["seed"])
    mutate(p)
    with pytest.raises(exc):
        load_params(_write(tmp_path, p))


def test_grid_for_a_model_outside_the_pool_is_rejected(params, tmp_path):
    p = override(params, "models", "candidates", value=["gradient_boosting"])
    p["models"]["baselines"] = ["prior_rate"]
    with pytest.raises(ValueError):
        load_params(_write(tmp_path, p))  # logistic_regression has a grid but is not in the pool


def test_disabled_tuning_accepts_any_lock(params, tmp_path):
    p = override(params, "tuning", "enabled", value=False)
    p["tuning"]["rule_lock"] = "unset"
    assert load_params(_write(tmp_path, p))["tuning"]["enabled"] is False


# --- command line ---------------------------------------------------------------------
def test_cli_check_and_write_lock(isolated_project, tmp_path):
    root = tmp_path / "proj"
    shutil.copytree(isolated_project, root)
    f = root / "params.yaml"
    f.write_text(f.read_text(encoding="utf-8").replace("margin_se: 1.0", "margin_se: 2.0"), encoding="utf-8")

    def cli(*args):
        return subprocess.run(
            [sys.executable, "-m", "src.models.tuning", *args], cwd=root, capture_output=True, text=True
        )

    assert cli("--check").returncode == 1  # the rule changed, the lock did not
    assert cli("--write-lock").returncode == 0
    assert cli("--check").returncode == 0
    assert "margin_se: 2.0" in f.read_text(encoding="utf-8")  # only the lock line changed


# --- rendering ----------------------------------------------------------------------
def test_render_context_for_metrics_without_tuning():
    assert _tuning({}) == {"enabled": False, "models": []}
    assert _tuning({"tuning": {"enabled": False}}) == {"enabled": False, "models": []}


def test_render_context_reads_the_committed_metrics():
    import json

    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    t = _tuning(metrics)
    assert t["enabled"] is True and t["rule_hash"] == metrics["tuning"]["rule_hash"]
    assert [r["name"] for r in t["models"]] == list(metrics["tuning"]["models"])
    for row in t["models"]:
        assert sum(c["is_default"] for c in row["candidates"]) == 1


def test_rendered_documents_show_the_tuning_decisions():
    import json

    from src.reporting.render import render_report

    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    for template in ("final_report.md.j2", "model_card.md.j2"):
        text = render_report(
            ROOT / "assumptions.yaml",
            ROOT / "reports" / "metrics.json",
            ROOT / "reports" / "roi.json",
            ROOT / "templates" / template,
        )
        assert metrics["tuning"]["rule_hash"] in text
        for name, m in metrics["tuning"]["models"].items():
            assert f"`{name}`" in text
            verb = "replaced the default" if m["replaced"] else "kept the default"
            assert verb in text
