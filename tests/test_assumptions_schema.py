"""P1: keys present, correctly typed, consistent across YAMLs; scaffold and Makefile intact."""

import copy
import re
import subprocess
import tomllib

import pytest
import yaml

from src.config import (
    ROOT,
    breakeven_threshold,
    business_status,
    load_all,
    load_assumptions,
    load_params,
)

pytestmark = pytest.mark.p1

P = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
A = yaml.safe_load((ROOT / "assumptions.yaml").read_text(encoding="utf-8"))


def _write(tmp_path, name, data):
    f = tmp_path / name
    f.write_text(yaml.safe_dump(data), encoding="utf-8")
    return f


def _mut(base, *path, value):
    d = copy.deepcopy(base)
    node = d
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return d


# --- keys present ------------------------------------------------------------
def test_repo_yaml_loads_clean():
    p, a = load_all()
    assert p["seed"] == P["seed"]
    assert a["clv"] == A["clv"]


@pytest.mark.parametrize("key", list(P))
def test_params_missing_top_key_raises(tmp_path, key):
    d = {k: v for k, v in P.items() if k != key}
    with pytest.raises(KeyError):
        load_params(_write(tmp_path, "p.yaml", d))


@pytest.mark.parametrize("key", list(A))
def test_assumptions_missing_key_raises(tmp_path, key):
    d = {k: v for k, v in A.items() if k != key}
    with pytest.raises(KeyError):
        load_assumptions(_write(tmp_path, "a.yaml", d))


def test_nested_missing_key_raises(tmp_path):
    d = copy.deepcopy(P)
    del d["split"]["cv_folds"]
    with pytest.raises(KeyError):
        load_params(_write(tmp_path, "p.yaml", d))


def test_unknown_key_raises(tmp_path):
    with pytest.raises(KeyError):
        load_params(_write(tmp_path, "p.yaml", {**P, "sedd": 1}))
    with pytest.raises(KeyError):
        load_assumptions(_write(tmp_path, "a.yaml", {**A, "clvv": 1}))


# --- correct types -----------------------------------------------------------
@pytest.mark.parametrize(
    "path,bad",
    [
        (("seed",), "42"),
        (("seed",), True),
        (("temporal",), "false"),
        (("split", "train_frac"), "0.6"),
        (("split", "cv_folds"), 5.5),
        (("schema", "required_columns"), "a,b"),
        (("sample", "n_rows"), 10.0),
    ],
)
def test_params_wrong_type_raises(tmp_path, path, bad):
    with pytest.raises(TypeError):
        load_params(_write(tmp_path, "p.yaml", _mut(P, *path, value=bad)))


@pytest.mark.parametrize(
    "key,bad",
    [
        ("clv", "1200"),
        ("save_rate", True),
        ("campaign_capacity", 700.5),
        ("population_per_period", "7043"),
        ("currency", 1),
    ],
)
def test_assumptions_wrong_type_raises(tmp_path, key, bad):
    with pytest.raises(TypeError):
        load_assumptions(_write(tmp_path, "a.yaml", _mut(A, key, value=bad)))


def test_int_accepted_for_float_field(tmp_path):
    assert load_assumptions(_write(tmp_path, "a.yaml", _mut(A, "clv", value=1000)))["clv"] == 1000


# --- value rules -------------------------------------------------------------
@pytest.mark.parametrize(
    "key,bad",
    [
        ("save_rate", 0),
        ("save_rate", 1.5),
        ("save_rate_source", "guess"),
        ("clv", 0),
        ("cost_intervention", -1),
    ],
)
def test_assumptions_bad_value_raises(tmp_path, key, bad):
    with pytest.raises(ValueError):
        load_assumptions(_write(tmp_path, "a.yaml", _mut(A, key, value=bad)))


def test_capacity_cannot_exceed_population(tmp_path):
    bad = _mut(A, "campaign_capacity", value=A["population_per_period"] + 1)
    with pytest.raises(ValueError):
        load_assumptions(_write(tmp_path, "a.yaml", bad))


def test_split_fractions_sum_to_one(tmp_path):
    with pytest.raises(ValueError):
        load_params(_write(tmp_path, "p.yaml", _mut(P, "split", "test_frac", value=0.5)))


def test_primary_metric_must_be_higher_is_better(tmp_path):
    for bad in ("brier", "ece", "roc_auc"):
        with pytest.raises(ValueError):
            load_params(_write(tmp_path, "p.yaml", _mut(P, "models", "primary_metric", value=bad)))


def test_unknown_model_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_params(_write(tmp_path, "p.yaml", _mut(P, "models", "candidates", value=["magic_forest"])))


def test_leakage_columns_must_exist(tmp_path):
    with pytest.raises(ValueError):
        load_params(_write(tmp_path, "p.yaml", _mut(P, "leakage", "flagged_columns", value=["nope"])))


@pytest.mark.parametrize("good", ["pr_auc", "lift_at_k"])
def test_primary_metric_accepts_allowed_values(tmp_path, good):
    f = _write(tmp_path, "p.yaml", _mut(P, "models", "primary_metric", value=good))
    assert load_params(f)["models"]["primary_metric"] == good


def test_candidates_and_baselines_must_be_disjoint(tmp_path):
    overlap = _mut(P, "models", "candidates", value=[P["models"]["baselines"][0], "gradient_boosting"])
    with pytest.raises(ValueError):
        load_params(_write(tmp_path, "p.yaml", overlap))
    assert not set(P["models"]["candidates"]) & set(P["models"]["baselines"])


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_method_accepts_allowed_values(tmp_path, method):
    f = _write(tmp_path, "p.yaml", _mut(P, "calibration", "method", value=method))
    assert load_params(f)["calibration"]["method"] == method


def test_calibration_method_rejects_unknown(tmp_path):
    with pytest.raises(ValueError):
        load_params(_write(tmp_path, "p.yaml", _mut(P, "calibration", "method", value="platt")))


@pytest.mark.parametrize("bad", [0, 1, 1.5, -0.1])
def test_max_abs_corr_must_be_inside_unit_interval(tmp_path, bad):
    with pytest.raises(ValueError):
        load_params(_write(tmp_path, "p.yaml", _mut(P, "leakage", "max_abs_corr", value=bad)))


def test_max_abs_corr_wrong_type_raises(tmp_path):
    with pytest.raises(TypeError):
        load_params(_write(tmp_path, "p.yaml", _mut(P, "leakage", "max_abs_corr", value="0.9")))


def test_k_is_not_written_in_params_file():
    text = (ROOT / "params.yaml").read_text(encoding="utf-8")
    assert not re.search(r"^\s*k\s*:", text, re.M)


# --- p*, churn rates -----------------------------------------------------------
def test_breakeven_threshold_formula():
    a = {**A, "cost_intervention": 40, "save_rate": 0.25, "clv": 1200}
    assert breakeven_threshold(a) == pytest.approx(40 / (0.25 * 1200))
    assert 0 < breakeven_threshold(A) < 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"cost_intervention": 300},  # p* = 1
        {"cost_intervention": 400},  # p* > 1
        {"save_rate": 0.01},  # p* > 1
        {"clv": 100},  # p* > 1
    ],
)
def test_breakeven_outside_unit_interval_rejected(tmp_path, overrides):
    with pytest.raises(ValueError):
        load_assumptions(_write(tmp_path, "a.yaml", {**A, **overrides}))


@pytest.mark.parametrize(
    "target,baseline",
    [(0.3, 0.265), (0.265, 0.265), (0.0, 0.265), (-0.1, 0.265), (0.2, 1.0), (0.2, 1.2)],
)
def test_churn_rates_must_be_ordered_in_unit_interval(tmp_path, target, baseline):
    bad = {**A, "target_churn_rate": target, "baseline_churn_rate": baseline}
    with pytest.raises(ValueError):
        load_assumptions(_write(tmp_path, "a.yaml", bad))


@pytest.mark.parametrize("key", ["target_churn_rate", "baseline_churn_rate"])
def test_churn_rate_wrong_type_raises(tmp_path, key):
    with pytest.raises(TypeError):
        load_assumptions(_write(tmp_path, "a.yaml", {**A, key: "0.2"}))


def test_capacity_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        load_assumptions(_write(tmp_path, "a.yaml", _mut(A, "campaign_capacity", value=0)))


def test_split_type_must_be_known(tmp_path):
    with pytest.raises(ValueError):
        load_params(_write(tmp_path, "p.yaml", _mut(P, "split", "type", value="group")))


# --- temporal vs non-temporal ------------------------------------------------
def _temporal(window, purge):
    d = copy.deepcopy(P)
    d["temporal"] = True
    d["split"]["type"] = "temporal"
    d["split"]["window_days"] = window
    d["split"]["purge_gap_days"] = purge
    return d


def test_temporal_valid(tmp_path):
    assert load_params(_write(tmp_path, "p.yaml", _temporal(30, 30)))["split"]["type"] == "temporal"


def test_temporal_purge_gap_must_cover_window(tmp_path):
    with pytest.raises(ValueError):
        load_params(_write(tmp_path, "p.yaml", _temporal(30, 29)))


def test_temporal_flag_must_match_split_type(tmp_path):
    with pytest.raises(ValueError):
        load_params(_write(tmp_path, "p.yaml", _mut(P, "temporal", value=True)))
    d = _temporal(30, 30)
    d["temporal"] = False
    with pytest.raises(ValueError):
        load_params(_write(tmp_path, "p.yaml", d))


def test_window_keys_rejected_in_nontemporal(tmp_path):
    with pytest.raises(ValueError):
        load_params(_write(tmp_path, "p.yaml", _mut(P, "split", "window_days", value=30)))


# --- consistency across YAMLs ------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "k",
        "clv",
        "cost_intervention",
        "save_rate",
        "campaign_capacity",
        "population_per_period",
        "target_churn_rate",
        "baseline_churn_rate",
    ],
)
def test_business_numbers_forbidden_in_params(tmp_path, name):
    with pytest.raises(ValueError):
        load_params(_write(tmp_path, "p.yaml", _mut(P, "evaluation", name, value=1)))


def test_no_business_key_leaks_into_params_file():
    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)

    assert not (set(keys(P)) & (set(A) - {"currency", "save_rate_source"}) | ({"k"} & set(keys(P))))


def test_sample_cannot_exceed_population(tmp_path):
    pf = _write(tmp_path, "p.yaml", _mut(P, "sample", "n_rows", value=A["population_per_period"] + 1))
    with pytest.raises(ValueError):
        load_all(pf, ROOT / "assumptions.yaml")


def test_sample_must_fit_cv(tmp_path):
    pf = _write(tmp_path, "p.yaml", _mut(P, "sample", "n_rows", value=P["split"]["cv_folds"]))
    with pytest.raises(ValueError):
        load_all(pf, ROOT / "assumptions.yaml")


def test_business_status_conditional_when_assumed():
    assert business_status({**A, "save_rate_source": "assumed"}) == "conditional"
    assert business_status({**A, "save_rate_source": "measured"}) == "measured"


# --- scaffold ----------------------------------------------------------------
@pytest.mark.parametrize(
    "path",
    [
        "src/data/__init__.py",
        "src/features/__init__.py",
        "src/models/__init__.py",
        "src/reporting/__init__.py",
        "data/raw",
        "handoff",
        "templates",
        "reports",
        "docs",
        "tests",
        "scripts/check_handoff.py",
        "scripts/scan_placeholders.py",
        ".gitignore",
        ".env.example",
        "pyproject.toml",
        "constraints.txt",
        "README.md",
        "Makefile",
    ],
)
def test_scaffold_exists(path):
    assert (ROOT / path).exists()


def test_raw_path_in_params_exists():
    assert (ROOT / P["paths"]["raw"]).is_file()


def test_gitignore_protects_env_not_data():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert ".env" in lines
    assert not any(x.startswith("data") for x in lines)


def test_env_example_has_no_seed_or_business_numbers():
    text = (ROOT / ".env.example").read_text(encoding="utf-8").lower()
    for word in ("seed", "clv", "save_rate", "cost_intervention", "capacity"):
        assert not re.search(rf"^\s*{word}\w*\s*=", text, re.M)


def _dep_names(specs):
    return {re.split(r"[<>=!~ \[]", s.strip(), maxsplit=1)[0].lower() for s in specs}


def _extras():
    py = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return py["project"], py["project"]["optional-dependencies"]


def test_dashboard_dependencies_are_an_extra_not_core():
    project, extras = _extras()
    core = _dep_names(project["dependencies"])
    assert not {"streamlit", "openpyxl"} & core
    assert {"streamlit", "openpyxl"} <= _dep_names(extras["dashboard"])


def test_dev_extra_covers_dashboard_and_notebook_tests():
    _, extras = _extras()
    dev = _dep_names(extras["dev"])
    assert {"pytest", "ruff", "nbformat", "nbclient", "ipykernel"} <= dev
    assert "churn-telco" in dev  # pulls in the dashboard extra, so dashboard tests run under .[dev]


def test_constraints_pin_every_core_dependency_exactly():
    project, _ = _extras()
    pins = {}
    for ln in (ROOT / "constraints.txt").read_text(encoding="utf-8").splitlines():
        if ln.strip() and not ln.startswith("#"):
            assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][^\s]*", ln.strip()), f"not an exact pin: {ln}"
            name, ver = ln.strip().split("==")
            pins[name.lower()] = ver
    assert _dep_names(project["dependencies"]) <= set(pins)


def test_no_requirements_file_duplicates_pyproject():
    assert not (ROOT / "requirements.txt").exists()


def test_pytest_markers_declared_for_all_parts():
    py = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {m.split(":")[0] for m in py["tool"]["pytest"]["ini_options"]["markers"]}
    assert declared == {f"p{n}" for n in range(1, 7)}


# --- Makefile ----------------------------------------------------------------
def _make(*args):
    return subprocess.run(["make", *args], cwd=ROOT, capture_output=True, text=True)


def test_make_dry_run_all_uses_full_data():
    """`make all` uses the full data (paths.raw): no step passes --sample."""
    r = _make("-n", "all")
    assert r.returncode == 0, r.stderr
    for entry in (
        "src.data.validate\n",
        "src.features.build\n",
        "src.models.train\n",
        "src.reporting.roi",
        "src.reporting.render",
    ):
        assert entry in r.stdout
    assert "--sample" not in r.stdout


def test_make_dry_run_all_sample_uses_sample():
    r = _make("-n", "all-sample")
    assert r.returncode == 0, r.stderr
    for entry in (
        "src.data.validate --sample",
        "src.features.build --sample",
        "src.models.train --sample",
        "src.reporting.render",
    ):
        assert entry in r.stdout


@pytest.mark.parametrize("n", range(1, 7))
def test_make_dry_run_each_check_target(n):
    r = _make("-n", f"check-p{n}")
    assert r.returncode == 0, r.stderr
    assert f"pytest -m p{n}" in r.stdout
    assert f"handoff/p{n}.yaml" in r.stdout


# --- hygiene -----------------------------------------------------------------
def test_no_placeholders_in_final_outputs():
    banned = re.compile(r"TODO|TBD|XXX|<\.\.\.>|\{\{")
    files = [ROOT / "README.md", ROOT / "params.yaml", ROOT / "assumptions.yaml"]
    files += sorted((ROOT / "docs").glob("*.md"))
    for f in files:
        assert not banned.search(f.read_text(encoding="utf-8")), f


def test_handoff_manifest_complete():
    m = yaml.safe_load((ROOT / "handoff/p1.yaml").read_text(encoding="utf-8"))
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
    assert m["open_issues"] == []
    for f in m["files"]:
        assert (ROOT / f).exists(), f
    produced = set(m["yaml_keys_produced"]["params"])
    assert produced <= set(P)
    # Params keys added by later parts must be declared in that part's manifest.
    later = set()
    for part in ("p2", "p3", "p4", "p5", "p6"):
        f = ROOT / f"handoff/{part}.yaml"
        if f.exists():
            later |= {
                k.split(".")[0]
                for k in yaml.safe_load(f.read_text(encoding="utf-8"))["yaml_keys_produced"]["params"]
            }
    assert set(P) - produced <= later
    assert set(m["yaml_keys_produced"]["assumptions"]) == set(A)
