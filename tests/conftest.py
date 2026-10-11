"""Shared fixtures for the P4 tests. Session fixtures are read-only: tests that mutate must deepcopy."""

import copy
import shutil

import pytest

from src.config import ROOT, load_all
from src.data.load import load_raw
from src.data.synthetic import generate
from src.models.split import make_split
from src.models.train import run


def override(base: dict, *path, value):
    """Deep copy of `base` with one nested value replaced."""
    d = copy.deepcopy(base)
    node = d
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return d


def relock(base: dict) -> dict:
    """Deep copy of `base` whose tuning.rule_lock matches the tuning rule it carries."""
    from src.models.tuning import rule_hash

    d = copy.deepcopy(base)
    d["tuning"]["rule_lock"] = rule_hash(d)
    return d


@pytest.fixture(scope="session")
def cfg():
    return load_all()


@pytest.fixture(scope="session")
def params(cfg):
    return cfg[0]


@pytest.fixture(scope="session")
def assumptions(cfg):
    return cfg[1]


@pytest.fixture(scope="session")
def fast(params):
    """Params with a small model and 3-fold CV, to keep pipeline tests fast. Tuning is off (see fast_tuned)."""
    p = override(params, "models", "hyperparams", "gradient_boosting", "n_estimators", value=25)
    p = override(p, "split", "cv_folds", value=3)
    return override(p, "tuning", "enabled", value=False)


@pytest.fixture(scope="session")
def fast_tuned(fast):
    """`fast` with tuning on, two candidates per model, and a lock that matches the rule."""
    p = override(fast, "tuning", "enabled", value=True)
    p["tuning"]["grids"] = {
        "logistic_regression": {"C": [0.1, 1.0]},
        "gradient_boosting": {"max_depth": [2, 3]},
    }
    return relock(p)


@pytest.fixture(scope="session")
def sample_df(params):
    return load_raw("sample", params)


@pytest.fixture(scope="session")
def synth(params):
    """Synthetic 2000 rows with known p_true (p_true is for tests only)."""
    p = override(params, "sample", "n_rows", value=2000)
    return generate(7, p)


@pytest.fixture(scope="session")
def base_split(sample_df, params):
    return make_split(sample_df, params)


@pytest.fixture(scope="session")
def base_run(sample_df, fast, assumptions, base_split):
    return run(sample_df, fast, assumptions, split=base_split)


@pytest.fixture(scope="session")
def isolated_project(tmp_path_factory):
    """Minimal project copy (code, config, sample.csv) for CLI tests.

    Tests that run an entry point must work here, not in ROOT: `make all` writes
    reports/ and artifacts/ from the full data, and tests must not overwrite them with sample output.
    """
    root = tmp_path_factory.mktemp("isolated")
    shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("params.yaml", "assumptions.yaml"):
        shutil.copy(ROOT / f, root / f)
    (root / "data").mkdir()
    shutil.copy(ROOT / "data" / "sample.csv", root / "data" / "sample.csv")
    return root
