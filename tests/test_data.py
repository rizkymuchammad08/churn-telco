"""P2: loader, validator, sample, synthetic generator, broken fixtures, data dictionary, notebook 01."""

import contextlib
import copy
import io
import json
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import ROOT, load_assumptions, load_params
from src.data import validate as validate_mod
from src.data.dictionary import DESCRIPTIONS, build_dictionary
from src.data.fixtures import build as build_fixtures
from src.data.fixtures import expected_text
from src.data.load import DataLoadError, clean, data_hash, load_raw, to_csv_text
from src.data.sample import make_sample
from src.data.synthetic import generate
from src.data.validate import ALL_CODES, banner, validate

pytestmark = pytest.mark.p2

P = load_params()
A = load_assumptions()
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "broken"
EXPECTED = yaml.safe_load((FIXTURE_DIR / "expected.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def raw():
    return load_raw("raw", P)


@pytest.fixture()
def good(raw):
    return raw.head(50).copy()


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "src.data.validate", *args], cwd=ROOT, capture_output=True, text=True
    )


# --- load / hash -------------------------------------------------------------
def test_raw_loads_and_validates(raw):
    rep = validate(raw, P)
    assert rep.ok, rep.errors


def test_totalcharges_null_only_when_tenure_zero(raw):
    nulls = raw[raw["TotalCharges"].isna()]
    assert len(nulls) > 0
    assert (nulls["tenure"] == 0).all()


def test_load_types_are_stable(raw):
    assert raw["tenure"].dtype == np.int64
    assert raw["SeniorCitizen"].dtype == np.int64
    assert raw["MonthlyCharges"].dtype == np.float64
    assert raw["TotalCharges"].dtype == np.float64


def test_hash_deterministic_and_order_invariant(raw):
    h = data_hash(raw)
    assert h == data_hash(raw.copy())
    assert h == data_hash(raw.sample(frac=1.0, random_state=1))


def test_hash_changes_with_data(raw):
    other = raw.copy()
    other.loc[0, "MonthlyCharges"] += 1
    assert data_hash(other) != data_hash(raw)


def test_load_missing_file_raises_coded():
    with pytest.raises(DataLoadError) as e:
        load_raw("does/not/exist.csv", P)
    assert e.value.code == "E_FILE_MISSING"


# --- validator (unit) --------------------------------------------------------
def test_validate_empty(good):
    assert "E_EMPTY" in validate(good.iloc[0:0], P).errors


def test_validate_missing_column(good):
    assert "E_COLUMNS_MISSING" in validate(good.drop(columns=["Contract"]), P).errors


def test_validate_duplicate_id(good):
    good.loc[1, "customerID"] = good.loc[0, "customerID"]
    assert "E_DUP_ID" in validate(good, P).errors


def test_validate_null_id(good):
    good.loc[0, "customerID"] = ""
    assert "E_NULL_ID" in validate(good, P).errors


def test_validate_label_domain(good):
    good.loc[0, "Churn"] = "Maybe"
    assert "E_LABEL_DOMAIN" in validate(good, P).errors


def test_validate_null_category(good):
    good.loc[0, "PaymentMethod"] = ""
    assert "E_NULL_CATEGORY" in validate(good, P).errors


def test_validate_negative(good):
    good.loc[0, "MonthlyCharges"] = -1.0
    assert "E_NEGATIVE_NUMERIC" in validate(good, P).errors


def test_validate_totalcharges_null_with_tenure(good):
    good.loc[0, "tenure"] = 5
    good.loc[0, "TotalCharges"] = float("nan")
    assert "E_NULL_NUMERIC" in validate(good, P).errors


def test_validate_totalcharges_null_ok_at_tenure_zero(good):
    good.loc[0, "tenure"] = 0
    good.loc[0, "TotalCharges"] = float("nan")
    assert validate(good, P).ok


def test_validate_unknown_data_source(good):
    p = copy.deepcopy(P)
    p["data_source"] = "mystery"
    assert "E_DATA_SOURCE" in validate(good, p).errors


def test_validate_leakage_column_unknown(good):
    p = copy.deepcopy(P)
    p["leakage"]["flagged_columns"] = ["nonexistent_column"]
    assert "E_LEAKAGE_COLUMN_UNKNOWN" in validate(good, p).errors


def test_validate_does_not_judge_outcomes(good):
    """Checks test the process: an extreme churn rate is still valid."""
    good["Churn"] = "Yes"
    assert validate(good, P).ok
    good["Churn"] = "No"
    assert validate(good, P).ok


# --- clean -------------------------------------------------------------------
def _messy(raw):
    m = raw.head(30).astype(object).copy()
    m["Contract"] = "  " + m["Contract"] + " "
    m["tenure"] = m["tenure"].astype(str).str.pad(4)
    m["MonthlyCharges"] = m["MonthlyCharges"].astype(str) + " "
    m["TotalCharges"] = m["TotalCharges"].astype(str).replace({"nan": " "})
    return m


def test_clean_idempotent(raw):
    once = clean(_messy(raw), P)
    twice = clean(once, P)
    assert once.equals(twice) and list(once.dtypes) == list(twice.dtypes)
    assert data_hash(once) == data_hash(twice)


def test_clean_idempotent_on_loaded_data(raw):
    assert clean(raw, P).equals(raw)


def test_clean_strips_and_types(raw):
    out = clean(_messy(raw), P)
    assert not out["Contract"].str.startswith(" ").any() and not out["Contract"].str.endswith(" ").any()
    assert out["tenure"].dtype == "int64" and out["MonthlyCharges"].dtype == "float64"
    assert out.equals(clean(raw.head(30), P))


def test_clean_does_not_mutate_input_or_drop_rows(raw):
    m = _messy(raw)
    before = m.copy(deep=True)
    out = clean(m, P)
    assert m.equals(before) and len(out) == len(m)


def test_clean_rejects_nonnumeric():
    df = pd.DataFrame({c: ["1"] for c in P["schema"]["required_columns"]})
    df["MonthlyCharges"] = "abc"
    with pytest.raises(DataLoadError) as e:
        clean(df, P)
    assert e.value.code == "E_NONNUMERIC"


# --- broken fixtures -----------------------------------------------------------
def test_fixtures_cover_every_data_error_code():
    param_or_file_driven = {
        "E_FILE_MISSING",
        "E_DATA_SOURCE",
        "E_LEAKAGE_COLUMN_UNKNOWN",
        "E_SNAPSHOT_INVALID",
    }
    assert set(EXPECTED.values()) | param_or_file_driven == set(ALL_CODES)


def test_committed_fixtures_equal_generator():
    texts, expected = build_fixtures(P)
    texts["expected.yaml"] = expected_text(expected)
    on_disk = {p.name: p.read_text(encoding="utf-8") for p in FIXTURE_DIR.glob("*") if p.is_file()}
    assert on_disk == texts


@pytest.mark.parametrize("name,code", sorted(EXPECTED.items()))
def test_cli_rejects_each_broken_fixture(name, code):
    r = _cli("--path", f"tests/fixtures/broken/{name}")
    assert r.returncode == 1, r.stdout
    assert re.search(rf"^{code}:", r.stdout, re.M), r.stdout
    assert "VALID" not in r.stdout.splitlines()


@pytest.mark.parametrize("name,code", sorted(EXPECTED.items()))
def test_each_fixture_triggers_only_its_own_code(name, code):
    path = f"tests/fixtures/broken/{name}"
    try:
        df = load_raw(path, P)
    except DataLoadError as e:
        assert e.code == code
        return
    assert set(validate(df, P).errors) == {code}


def test_cli_missing_file_rejected():
    r = _cli("--path", "tests/fixtures/broken/does_not_exist.csv")
    assert r.returncode == 1
    assert "E_FILE_MISSING" in r.stdout


def test_cli_sample_passes():
    r = _cli("--sample")
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip().endswith("VALID")


def test_cli_raw_passes():
    assert _cli().returncode == 0


# --- sample ------------------------------------------------------------------
def test_committed_sample_equals_generator(raw):
    committed = (ROOT / P["paths"]["sample"]).read_text(encoding="utf-8")
    assert committed == to_csv_text(make_sample(raw, P))


def test_sample_deterministic_and_seed_sensitive(raw):
    a, b = make_sample(raw, P), make_sample(raw, P)
    assert a.equals(b)
    p2 = copy.deepcopy(P)
    p2["seed"] = P["seed"] + 1
    assert not make_sample(raw, p2)["customerID"].equals(a["customerID"])


def test_sample_is_stratified_and_valid(raw):
    s = make_sample(raw, P)
    assert len(s) == P["sample"]["n_rows"]
    assert s["customerID"].is_unique
    assert abs((s["Churn"] == "Yes").mean() - (raw["Churn"] == "Yes").mean()) < 0.01
    assert validate(s, P).ok


def test_sample_subset_of_raw(raw):
    s = load_raw("sample", P)
    merged = s.merge(raw, on=list(raw.columns), how="inner")
    assert len(merged) == len(s)


def test_sample_larger_than_raw_rejected(raw):
    p = copy.deepcopy(P)
    p["sample"]["n_rows"] = len(raw) + 1
    with pytest.raises(ValueError):
        make_sample(raw, p)


# --- synthetic generator ------------------------------------------------------
def test_synthetic_valid_and_deterministic():
    events, p_true = generate(P["seed"], P)
    p_syn = {**P, "data_source": "synthetic"}
    assert validate(events, p_syn).ok
    e2, p2 = generate(P["seed"], P)
    assert events.equals(e2) and np.array_equal(p_true, p2)
    e3, p3 = generate(P["seed"] + 1, P)
    assert not np.array_equal(p_true, p3)


def test_synthetic_p_true_is_test_only():
    events, p_true = generate(P["seed"], P)
    assert list(events.columns) == P["schema"]["required_columns"]
    assert "p_true" not in events.columns
    assert len(p_true) == len(events)
    assert ((p_true > 0) & (p_true < 1)).all()


def test_synthetic_labels_follow_p_true():
    p = copy.deepcopy(P)
    p["sample"]["n_rows"] = 6000
    events, p_true = generate(P["seed"], p)
    rate = (events["Churn"] == "Yes").mean()
    assert abs(rate - p_true.mean()) < 0.03
    hi = events.loc[p_true > np.median(p_true), "Churn"].eq("Yes").mean()
    lo = events.loc[p_true <= np.median(p_true), "Churn"].eq("Yes").mean()
    assert hi > lo


def test_synthetic_hash_follows_seed():
    def h(seed):
        return data_hash(generate(seed, P)[0])

    assert h(1) == h(1)
    assert h(1) != h(2)


def test_synthetic_base_rate_near_baseline():
    p = copy.deepcopy(P)
    p["sample"]["n_rows"] = 6000
    events, p_true = generate(P["seed"], p)
    base = A["baseline_churn_rate"]
    assert abs(p_true.mean() - base) < 1e-6  # intercept is calibrated exactly
    assert abs((events["Churn"] == "Yes").mean() - base) < 0.03


def test_synthetic_has_several_segments():
    events, _ = generate(P["seed"], P)
    sizes = events.groupby(["Contract", "InternetService"]).size()
    assert len(sizes) >= 6 and sizes.min() >= 1


def test_synthetic_has_planted_outliers_that_still_validate():
    events, _ = generate(P["seed"], P)
    outliers = events["MonthlyCharges"] > 118.75
    assert outliers.sum() >= 1 and not outliers.all()
    assert validate(events, {**P, "data_source": "synthetic"}).ok


def test_synthetic_p_true_ranks_and_calibrates():
    from sklearn.metrics import roc_auc_score

    p = copy.deepcopy(P)
    p["sample"]["n_rows"] = 8000
    events, p_true = generate(P["seed"], p)
    y = (events["Churn"] == "Yes").to_numpy()
    assert roc_auc_score(y, p_true) > 0.65  # p_true is genuinely informative
    bins = np.quantile(p_true, np.linspace(0, 1, 6))
    idx = np.clip(np.digitize(p_true, bins[1:-1]), 0, 4)
    gaps = [abs(y[idx == b].mean() - p_true[idx == b].mean()) for b in range(5)]
    assert max(gaps) < 0.05  # labels are drawn from p_true: perfect calibration up to sampling noise


def test_real_data_base_rate_matches_baseline_assumption(raw):
    rate = (raw["Churn"] == "Yes").mean()
    assert abs(rate - A["baseline_churn_rate"]) < 0.005


def test_synthetic_respects_domain_rules():
    p = copy.deepcopy(P)
    p["sample"]["n_rows"] = 3000
    events, _ = generate(P["seed"], p)
    assert events.loc[events["TotalCharges"].isna(), "tenure"].eq(0).all()
    no_net = events["InternetService"] == "No"
    assert events.loc[no_net, "OnlineSecurity"].eq("No internet service").all()
    no_phone = events["PhoneService"] == "No"
    assert events.loc[no_phone, "MultipleLines"].eq("No phone service").all()


def test_banner_only_for_synthetic():
    assert banner(P) is None
    assert banner({**P, "data_source": "synthetic"}) == validate_mod.BANNER


def test_cli_prints_banner_first_for_synthetic(monkeypatch, capsys):
    monkeypatch.setattr(validate_mod, "load_params", lambda: {**P, "data_source": "synthetic"})
    assert validate_mod.main(["--sample"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == validate_mod.BANNER


def test_synthetic_is_not_written_to_sample_csv():
    header = (ROOT / P["paths"]["sample"]).read_text(encoding="utf-8").splitlines()[0]
    assert "p_true" not in header
    assert "SYN-" not in (ROOT / P["paths"]["sample"]).read_text(encoding="utf-8")


# --- data dictionary --------------------------------------------------------------
def test_data_dictionary_equals_generator(raw):
    committed = (ROOT / "docs" / "data_dictionary.md").read_text(encoding="utf-8")
    assert committed == build_dictionary(raw, P)


def test_data_dictionary_covers_exactly_the_schema(raw):
    assert set(DESCRIPTIONS) == set(P["schema"]["required_columns"])
    text = (ROOT / "docs" / "data_dictionary.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `(\w+)` \|", text, re.M)
    assert rows == P["schema"]["required_columns"]


def test_data_dictionary_marks_roles_and_leakage():
    text = (ROOT / "docs" / "data_dictionary.md").read_text(encoding="utf-8")
    for col in P["leakage"]["flagged_columns"]:
        line = next(ln for ln in text.splitlines() if ln.startswith(f"| `{col}` |"))
        assert "leakage-prone" in line
    assert re.search(rf"\| `{P['schema']['label_column']}` .*\| label \|", text)
    assert re.search(rf"\| `{P['schema']['id_column']}` .*\| id \|", text)


def test_data_dictionary_has_no_placeholders():
    text = (ROOT / "docs" / "data_dictionary.md").read_text(encoding="utf-8")
    assert not re.search(r"TODO|TBD|XXX|<\.\.\.>|\{\{", text)


# --- notebook 01 -------------------------------------------------------------
NB_PATH = ROOT / "notebooks" / "01_data.ipynb"


def _s(x):
    return "".join(x) if isinstance(x, list) else x


def _load_nb():
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


def _code_cells(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _stored_stdout(cell):
    return "".join(_s(o["text"]) for o in cell["outputs"] if o["output_type"] == "stream")


def _norm(text):
    return re.sub(r"\s+", " ", text).strip()


def test_notebook_stored_outputs_are_clean():
    cells = _code_cells(_load_nb())
    assert cells
    assert [c["execution_count"] for c in cells] == list(range(1, len(cells) + 1))
    for c in cells:
        for o in c["outputs"]:
            assert o["output_type"] == "stream" and o["name"] == "stdout", o.get("ename")


def test_notebook_reexecutes_and_matches_stored_outputs(monkeypatch):
    nb = _load_nb()
    monkeypatch.chdir(ROOT / "notebooks")
    monkeypatch.setattr(sys, "path", list(sys.path))
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(_code_cells(nb), 1):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(_s(cell["source"]), f"<cell {i}>", "exec"), ns)
        assert _norm(buf.getvalue()) == _norm(_stored_stdout(cell)), f"cell {i} output is stale"


def test_notebook_quotes_readme_without_rewriting():
    nb = _load_nb()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = re.search(r"## Problem\n(.*?)(?=\n## )", readme, re.S).group(1).strip()
    bullets = [ln for ln in block.splitlines() if ln.startswith("- ")]
    assert bullets
    all_out = _norm("".join(_stored_stdout(c) for c in _code_cells(nb)))
    sources = _norm(" ".join(_s(c["source"]) for c in nb["cells"]))
    for b in bullets:
        assert _norm(b) in all_out, "README excerpt does not appear in the output"
        assert _norm(b)[:50] not in sources, "Problem text was rewritten in the notebook"


def test_notebook_executes_with_nbclient():
    nbformat = pytest.importorskip("nbformat")
    client_mod = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    nb = nbformat.read(NB_PATH, as_version=4)
    client_mod.NotebookClient(
        nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(NB_PATH.parent)}}
    ).execute()
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert not any(o.output_type == "error" for o in cell.outputs)


# --- manifest ----------------------------------------------------------------
def test_handoff_p2_manifest_complete_and_consistent():
    m = yaml.safe_load((ROOT / "handoff/p2.yaml").read_text(encoding="utf-8"))
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
    assert m["part"] == "P2" and m["check"] == "make check-p2"
    assert m["open_issues"] == []
    for f in m["files"]:
        assert (ROOT / f).exists(), f
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    assumptions = yaml.safe_load((ROOT / "assumptions.yaml").read_text(encoding="utf-8"))
    assert set(m["yaml_keys_consumed"]["params"]) <= set(params)
    assert set(m["yaml_keys_consumed"]["assumptions"]) <= set(assumptions)
    assert m["yaml_keys_produced"] == {"params": [], "assumptions": []}


def test_p2_does_not_own_p1_files():
    p1 = yaml.safe_load((ROOT / "handoff/p1.yaml").read_text(encoding="utf-8"))
    p2 = yaml.safe_load((ROOT / "handoff/p2.yaml").read_text(encoding="utf-8"))
    assert not set(p1["files"]) & set(p2["files"])
