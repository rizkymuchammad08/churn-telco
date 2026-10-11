"""P5: strict rendering. Missing key = error, no placeholders, synthetic banner, label, footer hash."""

import copy
import hashlib
import importlib.util
import json
import re
import shutil

import pytest
import yaml

from src.config import ROOT
from src.data.validate import BANNER
from src.reporting.render import (
    DOCS,
    RenderError,
    build_context,
    render_report,
    render_text,
    source_hash,
)
from src.reporting.trace import numeric_leaves, template_literals, untraceable

pytestmark = pytest.mark.p5

TEMPLATE = ROOT / "templates" / "final_report.md.j2"
DOC_TEMPLATES = [ROOT / t for t, _ in DOCS]
PLACEHOLDER = re.compile(r"TODO|TBD|XXX|<\.\.\.>|\{\{|\}\}|\{%")

# Fragments of rendered output. They must match the template text; if the templates change,
# change them here and nowhere else.
CONDITIONAL_LABEL = "conditional value"
SYNTHETIC_MARKER = "SYNTHETIC DATA"
REC_RECOMMENDED_HEADING = "## 2. Recommended"
REC_NOT_RECOMMENDED_HEADING = "## 3. NOT recommended"
REC_ABSOLUTE_FRAGMENT = "absolute values.** Calibrated ECE"
REC_SWAP_FRAGMENT = "Switching the selected model"
REC_UNRELIABLE_FRAGMENT = "unreliable segments"
REC_ABSOLUTE_WARNING = "Reading probabilities as absolute values"
REC_SWAP_WARNING = "Switching the selected model based on test scores"


CI_KEYS = ("pr_auc", "roc_auc", "precision_at_k", "recall_at_k", "lift_at_k", "brier")


def _model_row(pr_auc, k=20):
    return {
        "n": 200,
        "n_pos": 53,
        "base_rate": 0.265,
        "k": k,
        "pr_auc": pr_auc,
        "roc_auc": 0.85,
        "precision_at_k": 0.7,
        "recall_at_k": 0.26,
        "lift_at_k": 2.5,
        "brier": 0.14,
        "ece": 0.04,
        "ci": {
            **{m: {"lo": 0.1, "hi": 0.9} for m in CI_KEYS},
            "level": 0.95,
            "level_pct": 95,
            "n_resamples": 1000,
        },
    }


def _metrics(data_source="real_public_ibm_telco"):
    seg = {"n": 100, "n_pos": 20, "base_rate": 0.2, "mean_p": 0.21, "brier": 0.1, "roc_auc": 0.8}
    chosen = _model_row(0.6)
    chosen["by_segment"] = {
        "Contract": {"Month-to-month": {**seg, "reliable": True}},
        "SeniorCitizen": {"1": {**seg, "n": 10, "reliable": False}},
    }
    return {
        "data": "sample",
        "data_source": data_source,
        "primary_metric": "pr_auc",
        "selected": "logistic_regression",
        "split": {"train": 600, "calib": 200, "test": 200},
        "cv": {"prior_rate": 0.265, "logistic_regression": 0.63},
        "test": {"prior_rate": _model_row(0.265), "logistic_regression": chosen},
        "calibration": {
            "method": "isotonic",
            "ece_raw": 0.16,
            "ece_calibrated": 0.04,
            "tolerance": 0.05,
            "within_tolerance": True,
        },
        "hashes": {
            k: hashlib.sha256(k.encode()).hexdigest()
            for k in ("input_data", "train_data", "preprocess", "model", "calibrator")
        },
    }


def _ablation(data="sample", corr=None):
    """ablation.json consistent with `_metrics()`: same data and input hash."""

    def stat(mean, std=0.0):
        return {"mean": mean, "std": std}

    def row(without, with_, better=2):
        return {
            "without": stat(without),
            "with": stat(with_),
            "delta_with_minus_without": stat(with_ - without, 0.01),
            "seeds_with_better": better,
        }

    def seed(n):
        base = {"pr_auc": 0.5, "roc_auc": 0.8, "lift_at_k": 2.5, "brier": 0.14, "ece": 0.03}
        return {
            "seed": n,
            "without": {"selected": "logistic_regression", **base},
            "with": {"selected": "gradient_boosting", **base},
        }

    return {
        "data": data,
        "data_source": "real_public_ibm_telco",
        "input_data_hash": hashlib.sha256(b"input_data").hexdigest(),
        "official_policy": "exclude",
        "seeds": [1, 2],
        "flagged_columns": ["tenure", "TotalCharges"],
        "max_abs_corr": 0.9,
        "n_seeds": 2,
        "n_selection_changed": 2,
        "abs_corr_with_label": corr or {"tenure": 0.35, "TotalCharges": 0.2},
        "metrics_on_test": ["pr_auc", "roc_auc", "lift_at_k", "brier", "ece"],
        "summary": {
            "pr_auc": row(0.5, 0.56),
            "roc_auc": row(0.8, 0.82),
            "lift_at_k": row(2.5, 2.8),
            "brier": row(0.14, 0.13),
            "ece": row(0.03, 0.02),
        },
        "per_seed": [seed(1), seed(2)],
    }


def _roi():
    def pol(n, c, v):
        return {"n_contacts": n, "n_churners_contacted": c, "net_value": v}

    return {
        "p_star": 0.1333,
        "k_test": 20,
        "policies": {
            "no_contact": pol(0, 0, 0.0),
            "model_p_star": pol(20, 13, 3100.0),
            "random_k": pol(20, 5.3, 790.0),
        },
        "incremental": {"model_vs_no_contact": 3100.0, "model_vs_random_k": 2310.0},
        "breakeven_save_rate": {"model_vs_no_contact": 0.05, "model_vs_random_k": 0.0},
    }


@pytest.fixture()
def src(tmp_path, assumptions):
    """Three sources in tmp_path; `write(**override)` replaces some of them. Returns their paths."""
    paths = {
        "assumptions": tmp_path / "assumptions.yaml",
        "metrics": tmp_path / "metrics.json",
        "roi": tmp_path / "roi.json",
        "ablation": tmp_path / "ablation.json",
        "template": tmp_path / "t.md.j2",
    }
    shutil.copy(TEMPLATE, paths["template"])
    for t in DOC_TEMPLATES:
        shutil.copy(t, tmp_path / t.name)

    def write(*, a=None, m=None, r=None, ab=None):
        paths["assumptions"].write_text(yaml.safe_dump(a or copy.deepcopy(assumptions)), encoding="utf-8")
        paths["metrics"].write_text(json.dumps(m or _metrics()), encoding="utf-8")
        paths["roi"].write_text(json.dumps(r or _roi()), encoding="utf-8")
        paths["ablation"].write_text(json.dumps(ab or _ablation()), encoding="utf-8")
        return paths

    paths["write"] = write
    write()
    return paths


def _render(p):
    return render_report(p["assumptions"], p["metrics"], p["roi"], p["template"], p["ablation"])


def _render_doc(p, template):
    """Render one of the registered templates from its copy in tmp_path."""
    return render_report(
        p["assumptions"], p["metrics"], p["roi"], p["assumptions"].parent / template.name, p["ablation"]
    )


# --- missing key = error ------------------------------------------------------
@pytest.mark.parametrize(
    "where,mutate",
    [
        ("m", lambda d: d.pop("selected")),
        ("m", lambda d: d.pop("data_source")),
        ("m", lambda d: d["calibration"].pop("within_tolerance")),
        ("m", lambda d: d["test"]["logistic_regression"].pop("pr_auc")),
        ("m", lambda d: d.pop("data")),
        ("m", lambda d: d["split"].pop("calib")),
        ("m", lambda d: d["cv"].pop("prior_rate")),
        ("m", lambda d: d["calibration"].pop("ece_raw")),
        ("m", lambda d: d["test"]["logistic_regression"].pop("by_segment")),
        (
            "m",
            lambda d: d["test"]["logistic_regression"]["by_segment"]["Contract"]["Month-to-month"].pop("n"),
        ),
        ("r", lambda d: d.pop("p_star")),
        ("r", lambda d: d["policies"].pop("random_k")),
        ("r", lambda d: d["policies"]["model_p_star"].pop("net_value")),
        ("r", lambda d: d["incremental"].pop("model_vs_random_k")),
        ("a", lambda d: d.pop("save_rate_source")),
        ("a", lambda d: d.pop("currency")),
        ("ab", lambda d: d.pop("max_abs_corr")),
        ("ab", lambda d: d.pop("n_selection_changed")),
        ("ab", lambda d: d["abs_corr_with_label"].pop("tenure")),
        ("ab", lambda d: d["summary"]["pr_auc"].pop("with")),
    ],
)
def test_missing_source_key_is_error(src, assumptions, where, mutate):
    base = {"a": copy.deepcopy(assumptions), "m": _metrics(), "r": _roi(), "ab": _ablation()}
    mutate(base[where])
    src["write"](**base)
    with pytest.raises(RenderError) as e:
        _render(src)
    assert e.value.code == "E_RENDER_MISSING_KEY"


def test_missing_template_variable_is_error(src):
    src["template"].write_text("Hello {{ does_not_exist }}\n", encoding="utf-8")
    with pytest.raises(RenderError) as e:
        _render(src)
    assert e.value.code == "E_RENDER_MISSING_KEY"


def test_missing_nested_template_attribute_is_error():
    with pytest.raises(RenderError) as e:
        render_text("{{ hashes.nope }}", {"hashes": {}})
    assert e.value.code == "E_RENDER_MISSING_KEY"


def test_missing_input_file_is_error(src):
    src["roi"].unlink()
    with pytest.raises(RenderError) as e:
        _render(src)
    assert e.value.code == "E_RENDER_INPUT"


def test_syntax_error_in_template_is_render_error(src):
    src["template"].write_text("{% if x %}no closing tag\n", encoding="utf-8")
    with pytest.raises(RenderError) as e:
        _render(src)
    assert e.value.code == "E_RENDER_TEMPLATE"


# --- zero placeholders --------------------------------------------------------
@pytest.mark.parametrize("source", ["real_public_ibm_telco", "synthetic"])
@pytest.mark.parametrize("sr", ["assumed", "measured"])
def test_output_has_zero_placeholders(src, assumptions, source, sr):
    a = {**copy.deepcopy(assumptions), "save_rate_source": sr}
    src["write"](a=a, m=_metrics(source))
    out = _render(src)
    assert not PLACEHOLDER.search(out)
    assert "None" not in out and "nan" not in out.lower().split()


def test_placeholder_in_output_is_rejected():
    for bad in ("TODO fill in", "TBD", "XXX", "<...>"):
        with pytest.raises(RenderError) as e:
            render_text("{{ x }}", {"x": bad})
        assert e.value.code == "E_RENDER_PLACEHOLDER"


def test_template_dir_is_the_only_place_with_placeholders():
    assert "{{" in TEMPLATE.read_text(encoding="utf-8")
    assert not list((ROOT / "src").rglob("*.j2"))


# --- synthetic banner ----------------------------------------------------------
def test_banner_present_for_synthetic(src):
    src["write"](m=_metrics("synthetic"))
    out = _render(src)
    assert out.startswith(BANNER)
    assert out.count(BANNER) == 1


def test_banner_absent_for_real_data(src):
    assert BANNER not in _render(src)
    assert SYNTHETIC_MARKER not in _render(src)


# --- conditional value label ---------------------------------------------------
def test_conditional_label_when_save_rate_assumed(src, assumptions):
    assert assumptions["save_rate_source"] == "assumed"
    out = _render(src)
    assert CONDITIONAL_LABEL in out.lower()
    # Every business-value figure is labelled: the value column header and both incremental values.
    assert out.lower().count(CONDITIONAL_LABEL) >= 4


def test_conditional_label_absent_when_save_rate_measured(src, assumptions):
    src["write"](a={**copy.deepcopy(assumptions), "save_rate_source": "measured"})
    assert CONDITIONAL_LABEL not in _render(src).lower()


def test_conditional_label_independent_of_data_source(src):
    src["write"](m=_metrics("synthetic"))
    out = _render(src)
    assert CONDITIONAL_LABEL in out.lower() and BANNER in out


# --- footer source_hash --------------------------------------------------------
def test_footer_contains_hash_of_each_source(src):
    out = _render(src)
    footer = out[out.rindex("source_hash") :]
    for name in ("assumptions", "metrics", "roi", "ablation"):
        assert hashlib.sha256(src[name].read_bytes()).hexdigest() in footer
    assert footer.strip().splitlines()[-1].endswith("`")


def test_footer_source_hash_is_combined_digest(src):
    names = ("assumptions", "metrics", "roi", "ablation")
    h = {n: hashlib.sha256(src[n].read_bytes()).hexdigest() for n in names}
    joined = "|".join(f"{n}={h[n]}" for n in names)
    expected = hashlib.sha256(joined.encode()).hexdigest()
    assert f"source_hash: `{expected}`" in _render(src)


@pytest.mark.parametrize("name", ["assumptions", "metrics", "roi", "ablation"])
def test_combined_hash_depends_on_each_source(name):
    base = {"assumptions": "a" * 64, "metrics": "b" * 64, "roi": "c" * 64, "ablation": "e" * 64}
    assert source_hash({**base, name: "d" * 64}) != source_hash(base)


def test_source_hash_changes_when_any_source_changes(src, assumptions):
    h0 = _render(src)
    a = {**copy.deepcopy(assumptions), "clv": 1300}
    src["write"](a=a)
    assert _render(src) != h0

    base = _render(src)
    r = _roi()
    r["k_test"] = 21
    src["write"](a=a, r=r)
    assert _render(src) != base

    base = _render(src)
    m = _metrics()
    m["calibration"]["tolerance"] = 0.06
    src["write"](a=a, r=r, m=m)
    assert _render(src) != base

    base = _render(src)
    ab = _ablation()
    ab["summary"]["pr_auc"]["with"]["mean"] = 0.57
    src["write"](a=a, r=r, m=m, ab=ab)
    assert _render(src) != base


def test_render_is_deterministic(src):
    assert _render(src) == _render(src)


def test_source_hash_covers_raw_bytes_so_whitespace_edits_change_it(src):
    h = _render(src)
    src["roi"].write_text(src["roi"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert _render(src) != h


# --- context -------------------------------------------------------------------
def test_context_numbers_come_from_sources(assumptions):
    hashes = {"assumptions": "a", "metrics": "m", "roi": "r", "ablation": "b"}
    ctx = build_context(assumptions, _metrics(), _roi(), hashes, _ablation())
    assert ctx["k_test"] == 20 and ctx["p_star"] == 0.1333 and ctx["conditional"] is True
    assert [r["name"] for r in ctx["policies"]] == ["no_contact", "model_p_star", "random_k"]
    assert ctx["banner"] is None


def test_real_pipeline_outputs_render_clean(params):
    """Output files of `make train` + `make roi` (when present) render without errors or placeholders."""
    paths = [ROOT / "assumptions.yaml", ROOT / params["paths"]["metrics"], ROOT / params["paths"]["roi"]]
    ablation = ROOT / params["paths"]["ablation"]
    if not all(p.exists() for p in [*paths, ablation]):
        pytest.skip("run make train ablation roi first")
    out = render_report(*paths, TEMPLATE, ablation)
    assert not PLACEHOLDER.search(out)


# --- all registered documents ----------------------------------------------------
IDS = [t.name for t in DOC_TEMPLATES]


@pytest.mark.p6
def test_every_template_is_registered_once_and_outputs_are_unique():
    on_disk = {p.name for p in (ROOT / "templates").glob("*.j2")}
    assert on_disk == {t.name for t in DOC_TEMPLATES}, "template not registered in render.DOCS"
    assert len({o for _, o in DOCS}) == len(DOCS)
    assert {"final_report.md.j2", "model_card.md.j2", "recommendations.md.j2"} <= on_disk


@pytest.mark.parametrize("template", DOC_TEMPLATES, ids=IDS)
@pytest.mark.parametrize("source", ["real_public_ibm_telco", "synthetic"])
@pytest.mark.parametrize("sr", ["assumed", "measured"])
@pytest.mark.p6
def test_every_doc_renders_clean(src, assumptions, template, source, sr):
    src["write"](a={**copy.deepcopy(assumptions), "save_rate_source": sr}, m=_metrics(source))
    out = _render_doc(src, template)
    assert not PLACEHOLDER.search(out)
    assert "None" not in out and "nan" not in out.lower().split()
    assert (BANNER in out) == (source == "synthetic")
    assert (CONDITIONAL_LABEL in out.lower()) == (sr == "assumed")
    assert out.rstrip().splitlines()[-1].startswith(("- ablation.json:", "- calibrator:"))


@pytest.mark.parametrize("template", DOC_TEMPLATES, ids=IDS)
@pytest.mark.p6
def test_every_doc_is_deterministic_and_has_footer_hash(src, template):
    a = _render_doc(src, template)
    assert a == _render_doc(src, template)
    assert re.search(r"^source_hash: `[0-9a-f]{64}`$", a, re.MULTILINE)


@pytest.mark.parametrize("template", DOC_TEMPLATES, ids=IDS)
@pytest.mark.p6
def test_every_doc_missing_source_key_is_error(src, assumptions, template):
    m = _metrics()
    m["calibration"].pop("ece_calibrated")
    src["write"](m=m)
    with pytest.raises(RenderError) as e:
        _render_doc(src, template)
    assert e.value.code == "E_RENDER_MISSING_KEY"


# --- hand-typed numbers fail -------------------------------------------------------
@pytest.mark.parametrize("template", DOC_TEMPLATES, ids=IDS)
@pytest.mark.p6
def test_template_has_no_hand_typed_numbers(template):
    assert template_literals(template.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("template", DOC_TEMPLATES, ids=IDS)
@pytest.mark.p6
def test_every_rendered_number_comes_from_the_four_sources(src, assumptions, template):
    """Hand-typed numbers in a template (that slip past the literal detector) are caught in its output."""
    a, m, r, ab = copy.deepcopy(assumptions), _metrics(), _roi(), _ablation()
    src["write"](a=a, m=m, r=r, ab=ab)
    out = _render_doc(src, template)
    assert untraceable(out, numeric_leaves(a, m, r, ab)) == []


@pytest.mark.p6
def test_hand_typed_number_added_to_template_is_caught(src, assumptions):
    a, m, r, ab = copy.deepcopy(assumptions), _metrics(), _roi(), _ablation()
    src["write"](a=a, m=m, r=r, ab=ab)
    for template in DOC_TEMPLATES:
        typed = template.read_text(encoding="utf-8").replace("\n## 1.", "\nAccuracy 0.777.\n## 1.", 1)
        assert template_literals(typed) == ["0.777"]
        (src["assumptions"].parent / template.name).write_text(typed, encoding="utf-8")
        assert untraceable(_render_doc(src, template), numeric_leaves(a, m, r, ab)) == ["0.777"]


def _scan_module():
    spec = importlib.util.spec_from_file_location(
        "scan_placeholders", ROOT / "scripts" / "scan_placeholders.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.p6
def test_scan_script_flags_hand_typed_number_but_not_code_or_footer(tmp_path):
    scan = _scan_module()
    leaves = numeric_leaves(_roi(), _metrics())
    assert scan.scan_numbers("Net value 3,100 USD, lift 2.50.", leaves) == []
    assert scan.scan_numbers("Net value 4,200 USD.", leaves) == ["4,200"]
    assert scan.scan_numbers("Run `make check-p6` and `k = 77`.\n```\nx = 99\n```\n", leaves) == []
    footer = "source_hash: `1234567890abcdef`\n- model: `7a0330`\n"
    assert scan.scan_numbers("Body.\n\n" + footer, leaves) == []
    doc = tmp_path / "typed.md"
    doc.write_text("Churn dropped 31.4 percent.\n", encoding="utf-8")
    assert scan.main([str(doc)]) == 1
    doc.write_text("No numeric claims.\n", encoding="utf-8")
    assert scan.main([str(doc)]) == 0
    doc.write_text("TODO fill in\n", encoding="utf-8")
    assert scan.main([str(doc)]) == 1


# --- recommendations content ---------------------------------------------------------
RECS = ROOT / "templates" / "recommendations.md.j2"


@pytest.mark.p6
def test_recommendations_always_has_not_recommended_section(src):
    out = _render_doc(src, RECS)
    assert REC_NOT_RECOMMENDED_HEADING in out
    assert REC_RECOMMENDED_HEADING in out
    assert out.index("## 2.") < out.index("## 3.")


@pytest.mark.p6
def test_recommendations_warnings_follow_the_data(src, assumptions):
    base = _render_doc(src, RECS)  # fixture: within tolerance, selected model = best on test
    assert REC_ABSOLUTE_FRAGMENT not in base and REC_SWAP_FRAGMENT not in base
    assert REC_UNRELIABLE_FRAGMENT in base  # SeniorCitizen: senior citizens (reliable False)

    m = _metrics()
    m["calibration"]["within_tolerance"] = False
    m["test"]["gradient_boosting"] = _model_row(0.7)
    m["cv"]["gradient_boosting"] = 0.5
    src["write"](m=m)
    out = _render_doc(src, RECS)
    assert REC_ABSOLUTE_WARNING in out
    assert REC_SWAP_WARNING in out and "gradient_boosting" in out


@pytest.mark.parametrize("template", DOC_TEMPLATES[:2], ids=lambda p: p.name)
def test_input_data_hash_visible_in_body_not_only_footer(src, template):
    h = hashlib.sha256(b"input_data").hexdigest()
    out = render_report(src["assumptions"], src["metrics"], src["roi"], template)
    body = out[: out.rindex("source_hash")]
    assert f"`{h}`" in body


def test_missing_input_data_hash_is_error(src):
    m = _metrics()
    del m["hashes"]["input_data"]
    src["write"](m=m)
    with pytest.raises(RenderError):
        _render(src)


# --- excluded columns: a policy decision with a measured cost -------------------------
@pytest.mark.p6
def test_exclusion_is_described_as_policy_not_as_leakage_fact(src):
    out = _render(src)
    assert "policy decision" in out and "not a demonstrated case of leakage" in out
    assert "leakage risk" not in out.lower()
    assert "did not flag either column" in out
    assert "cannot be verified" in out  # no dates, so the reasoning is plausible but unproven


@pytest.mark.p6
def test_ablation_numbers_are_rendered_from_ablation_json(src):
    ab = _ablation()
    ab["summary"]["pr_auc"]["with"]["mean"] = 0.612
    ab["summary"]["pr_auc"]["delta_with_minus_without"]["mean"] = 0.112
    src["write"](ab=ab)
    out = _render(src)
    assert "0.612" in out and "+0.112" in out and "2 of 2" in out
    assert "`logistic_regression` to `gradient_boosting` in 2 of 2 splits" in out


@pytest.mark.p6
def test_detector_flag_is_reported_when_a_column_exceeds_the_limit(src):
    src["write"](ab=_ablation(corr={"tenure": 0.95, "TotalCharges": 0.2}))
    out = _render(src)
    assert "detector flagged: tenure" in out
    assert "did not flag" not in out


@pytest.mark.p6
def test_unchanged_selection_is_reported_as_unchanged(src):
    ab = _ablation()
    ab["n_selection_changed"] = 0
    src["write"](ab=ab)
    assert "The selected model does not change." in _render(src)


@pytest.mark.p6
@pytest.mark.parametrize("field,value", [("input_data_hash", "f" * 64), ("data", "raw")])
def test_ablation_computed_on_other_data_is_rejected(src, field, value):
    ab = _ablation()
    ab[field] = value
    src["write"](ab=ab)
    with pytest.raises(RenderError) as e:
        _render(src)
    assert e.value.code == "E_RENDER_STALE_ABLATION"


@pytest.mark.p6
def test_missing_ablation_file_is_input_error(src):
    src["ablation"].unlink()
    with pytest.raises(RenderError) as e:
        _render(src)
    assert e.value.code == "E_RENDER_INPUT"


def test_model_tables_show_bootstrap_interval_in_brackets():
    text = (ROOT / "reports" / "final_report.md").read_text(encoding="utf-8")
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    for name, m in metrics["test"].items():
        ci = m["ci"]["pr_auc"]
        assert f"{m['pr_auc']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]" in text, name
    assert f"{metrics['test'][metrics['selected']]['ci']['level_pct']}% bootstrap interval" in text
