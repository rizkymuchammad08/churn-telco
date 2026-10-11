"""P4: honest manifest, new params keys validated, file ownership does not overlap."""

import importlib
import re
import subprocess

import pytest
import yaml
from conftest import override

from src.config import ROOT, load_params

pytestmark = pytest.mark.p4


def _manifest(part):
    return yaml.safe_load((ROOT / f"handoff/{part}.yaml").read_text(encoding="utf-8"))


def _write(tmp_path, data):
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump(data), encoding="utf-8")
    return f


# --- manifest ----------------------------------------------------------------
def test_manifest_complete():
    m = _manifest("p4")
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
    assert m["part"] == "P4" and m["check"] == "make check-p4" and m["open_issues"] == []
    for f in m["files"] + m["modifies"]:
        assert (ROOT / f).exists(), f


def test_manifest_keys_match_params(params):
    m = _manifest("p4")
    assert set(m["yaml_keys_consumed"]["params"]) <= set(params)
    for dotted in m["yaml_keys_produced"]["params"]:
        node = params
        for part in dotted.split("."):
            assert part in node, dotted
            node = node[part]


def test_p4_does_not_own_earlier_files():
    owned_p4 = set(_manifest("p4")["files"])
    for part in ("p1", "p2", "p3"):
        assert not owned_p4 & set(_manifest(part)["files"])


def test_public_api_names_exist():
    for entry in _manifest("p4")["public_api"]:
        module, name = re.match(r"(\w+)\.(\w+)", entry).groups()
        assert hasattr(importlib.import_module(f"src.models.{module}"), name), entry


def test_make_train_dry_run_uses_locked_entry_point():
    full = subprocess.run(["make", "-n", "train"], cwd=ROOT, capture_output=True, text=True)
    assert full.returncode == 0 and "src.models.train" in full.stdout and "--sample" not in full.stdout
    small = subprocess.run(["make", "-n", "train-sample"], cwd=ROOT, capture_output=True, text=True)
    assert small.returncode == 0 and "src.models.train --sample" in small.stdout


# --- new params keys ---------------------------------------------------------
@pytest.mark.parametrize(
    "path,bad,exc",
    [
        (("models", "class_weight"), "auto", ValueError),
        (("models", "hyperparams", "logistic_regression", "C"), 0, ValueError),
        (("models", "hyperparams", "logistic_regression", "max_iter"), 0, ValueError),
        (("models", "hyperparams", "gradient_boosting", "n_estimators"), 0, ValueError),
        (("models", "hyperparams", "gradient_boosting", "learning_rate"), 1.5, ValueError),
        (("models", "hyperparams", "gradient_boosting", "max_depth"), 0, ValueError),
        (("models", "hyperparams", "gradient_boosting", "max_depth"), "3", TypeError),
        (("evaluation", "calibration_tolerance"), 0, ValueError),
        (("evaluation", "calibration_tolerance"), 1.0, ValueError),
        (("evaluation", "calibration_tolerance"), "0.05", TypeError),
        (("evaluation", "bootstrap_resamples"), 10, ValueError),
        (("evaluation", "bootstrap_resamples"), 500.5, TypeError),
        (("evaluation", "ci_level"), 1.0, ValueError),
        (("evaluation", "ci_level"), 0, ValueError),
        (("segments", "min_n"), 0, ValueError),
        (("segments", "min_n"), 3.5, TypeError),
        (("segments", "by"), [], ValueError),
        (("segments", "by"), ["Contrakt"], ValueError),
        (("segments", "by"), ["Churn"], ValueError),
        (("segments", "by"), ["customerID"], ValueError),
        (("domain", "min_tenure_months"), -1, ValueError),
        (("domain", "min_tenure_months"), 1.5, TypeError),
    ],
)
def test_new_param_keys_are_validated(params, tmp_path, path, bad, exc):
    with pytest.raises(exc):
        load_params(_write(tmp_path, override(params, *path, value=bad)))


@pytest.mark.parametrize("key", ["segments", "domain"])
def test_new_top_level_keys_are_required(params, tmp_path, key):
    with pytest.raises(KeyError):
        load_params(_write(tmp_path, {k: v for k, v in params.items() if k != key}))


@pytest.mark.parametrize(
    "path", [("evaluation", "calibration_tolerance"), ("segments", "min_n"), ("models", "hyperparams")]
)
def test_new_nested_keys_are_required(params, tmp_path, path):
    d = override(params, *path[:-1], path[-1], value=None)
    del_node = d
    for key in path[:-1]:
        del_node = del_node[key]
    del del_node[path[-1]]
    with pytest.raises(KeyError):
        load_params(_write(tmp_path, d))


def test_class_weight_none_is_accepted(params, tmp_path):
    load_params(_write(tmp_path, override(params, "models", "class_weight", value="none")))


def test_k_still_absent_from_params_after_p4(params):
    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)

    assert "k" not in set(keys(params))
