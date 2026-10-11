"""P6: templates have no numeric literals; document numbers trace to a source; documents are not stale."""

import hashlib
import json
import re

import pytest
import yaml

from src.config import ROOT
from src.reporting.render import DOCS, render_report, source_hash
from src.reporting.trace import (
    numbers_in,
    numeric_leaves,
    strip_code,
    template_literals,
    untraceable,
)

pytestmark = pytest.mark.p6

TEMPLATES = sorted((ROOT / "templates").glob("*.j2"))
REPORT = ROOT / "reports" / "final_report.md"
README = ROOT / "README.md"
CHANGELOG = ROOT / "CHANGELOG.md"
# (template, output) for every rendered document; its numbers may come only from the four sources.
RENDERED = [(ROOT / t, ROOT / o) for t, o in DOCS]
RENDERED_IDS = [o.name for _, o in RENDERED]
SOURCES = {
    "assumptions": ROOT / "assumptions.yaml",
    "metrics": ROOT / "reports" / "metrics.json",
    "roi": ROOT / "reports" / "roi.json",
    "ablation": ROOT / "reports" / "ablation.json",
}
TEMPLATE = ROOT / "templates" / "final_report.md.j2"
FOOTER = re.compile(r"^source_hash: `([0-9a-f]{64})`$", re.MULTILINE)


def _sources():
    return (
        yaml.safe_load(SOURCES["assumptions"].read_text(encoding="utf-8")),
        json.loads(SOURCES["metrics"].read_text(encoding="utf-8")),
        json.loads(SOURCES["roi"].read_text(encoding="utf-8")),
        json.loads(SOURCES["ablation"].read_text(encoding="utf-8")),
    )


_strip_code = strip_code


def _handwritten_docs():
    """Hand-typed documents: README, CHANGELOG, and docs/*.md files that are not render outputs."""
    rendered = {o for _, o in RENDERED}
    return [README, CHANGELOG, *[p for p in sorted((ROOT / "docs").glob("*.md")) if p not in rendered]]


@pytest.fixture(scope="module")
def leaves():
    return numeric_leaves(*_sources())


# --- templates have no numeric literals --------------------------------------
def test_templates_exist():
    assert TEMPLATE in TEMPLATES


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
def test_template_has_no_numeric_literals(template):
    assert template_literals(template.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "text",
    [
        "Average churn 26,5 percent\n",  # a number in plain text
        "Lift {{ 2.5 }}\n",  # literal in an expression
        '{{ "threshold 0.13" }}\n',  # numeric string that is not a format
        "{{ rows[0] }}\n",  # literal index
        "## Results 3 months\n",  # a number in a heading that is not a section number
    ],
)
def test_literal_detector_flags_numbers(text):
    assert template_literals(text)


@pytest.mark.parametrize(
    "text",
    [
        "## 1. Data\n",
        "### 2.3 Calibration\n",
        "Prepared 2026.\n",
        '{{ "%.3f"|format(x) }} and {{ "{:,.0f}".format(y) }} and {{ "%.2f" % z }}\n',
        "{% for r in rows %}{{ r.name }}{% endfor %}\n",
        "Model sha256sum and p1 are not claim numbers.\n",
    ],
)
def test_literal_detector_allows_section_numbers_years_and_format_specs(text):
    assert template_literals(text) == []


# --- every document number traces to a source ---------------------------------
@pytest.mark.parametrize("output", [o for _, o in RENDERED], ids=RENDERED_IDS)
def test_rendered_doc_exists_and_has_numbers(output):
    assert output.is_file(), "run make all"
    assert len(numbers_in(output.read_text(encoding="utf-8"))) >= 15  # the test is not vacuous


@pytest.mark.parametrize("output", [o for _, o in RENDERED], ids=RENDERED_IDS)
def test_every_number_in_rendered_doc_traces_to_a_source(leaves, output):
    bad = untraceable(output.read_text(encoding="utf-8"), leaves)
    assert bad == [], f"unsourced numbers: {bad}"


def test_report_exists_and_has_numbers():
    assert REPORT.is_file(), "run make all"
    assert len(numbers_in(REPORT.read_text(encoding="utf-8"))) >= 20  # the test is not vacuous


@pytest.mark.parametrize("doc", _handwritten_docs(), ids=lambda p: p.name)
def test_every_number_in_handwritten_prose_traces_to_a_source(leaves, doc):
    """Hand-typed README, CHANGELOG, docs/*.md: claim numbers must trace to a source."""
    bad = untraceable(_strip_code(doc.read_text(encoding="utf-8")), leaves, changelog=doc == CHANGELOG)
    assert bad == [], f"{doc.name}: unsourced hand-typed numbers: {bad}"


def test_handwritten_doc_list_covers_the_known_files():
    names = {p.name for p in _handwritten_docs()}
    assert {"README.md", "CHANGELOG.md", "adaptation_path_a.md"} <= names
    assert not names & {"final_report.md", "model_card.md", "recommendations.md"}


def test_hand_typed_number_in_handwritten_doc_is_caught(leaves):
    for doc in _handwritten_docs():
        text = _strip_code(doc.read_text(encoding="utf-8")) + "\nModel accuracy 0.777.\n"
        assert untraceable(text, leaves, changelog=doc == CHANGELOG) == ["0.777"], doc.name


# --- CHANGELOG ------------------------------------------------------------------
def test_changelog_latest_version_matches_pyproject():
    text = CHANGELOG.read_text(encoding="utf-8")
    versions = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", text, re.MULTILINE)
    declared = re.search(
        r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.MULTILINE
    )
    assert versions and declared
    assert versions[0] == declared.group(1)


def test_changelog_heading_whitelist_does_not_hide_other_numbers(leaves):
    heading = "## [9.9.9] - 2026-10-01\n"
    assert untraceable(heading, leaves, changelog=True) == []
    assert untraceable(heading, leaves)  # without the exception, the release heading counts as a number
    assert untraceable(heading + "Accuracy 0.777 at 9.9.9.\n", leaves, changelog=True) == ["0.777", "9.9"]
    assert untraceable("Version 9.9.9 released.\n", leaves, changelog=True)  # not a heading: not exempted


def test_tracing_detects_fabricated_number(leaves):
    doc = REPORT.read_text(encoding="utf-8")
    assert untraceable(doc, leaves) == []
    assert untraceable(doc.replace("## 1.", "Accuracy 0.777.\n## 1.", 1), leaves) == ["0.777"]
    assert untraceable("Value 9,999 USD", leaves) == ["9,999"]


def test_tracing_whitelists_section_numbers_years_and_footer_hashes(leaves):
    doc = "## 7. Section\nYear 2026.\n\nsource_hash: `" + "1234567890abcdef" * 4 + "`\n"
    assert untraceable(doc, leaves) == []


def test_tracing_depends_on_the_sources():
    # Assumptions are copied into roi.json (inputs), so only metrics and roi are irreplaceable.
    a, m, r, ab = _sources()
    doc = REPORT.read_text(encoding="utf-8")
    assert untraceable(doc, numeric_leaves()) == numbers_in(doc)
    assert untraceable(doc, numeric_leaves(a, r, ab))
    assert untraceable(doc, numeric_leaves(a, m, ab))
    assert untraceable(doc, numeric_leaves(a, m, r))  # the excluded-columns section needs ablation.json
    assert untraceable(doc, numeric_leaves(m, r, ab)) == []


def test_tracing_matches_displayed_precision():
    leaves = [0.0512820512, 3100.0, 5.300000000000001]
    assert untraceable("0.051 and 3,100 and 5.3", leaves) == []
    assert untraceable("0.052", leaves) == ["0.052"]
    assert untraceable("3,101", leaves) == ["3,101"]


# --- documents are not stale -------------------------------------------------
def _file_hashes():
    return {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in SOURCES.items()}


@pytest.mark.parametrize("output", [o for _, o in RENDERED], ids=RENDERED_IDS)
def test_footer_source_hash_equals_current_source_hash(output):
    footers = FOOTER.findall(output.read_text(encoding="utf-8"))
    assert len(footers) == 1
    assert footers[0] == source_hash(_file_hashes()), f"{output.name} is stale: run make all"


@pytest.mark.parametrize("output", [o for _, o in RENDERED], ids=RENDERED_IDS)
def test_footer_lists_current_hash_of_each_source(output):
    footer = output.read_text(encoding="utf-8").split("source_hash:", 1)[1]
    for name, h in _file_hashes().items():
        assert h in footer, f"hash of {name} in the footer of {output.name} differs from the current file"


@pytest.mark.parametrize("template,output", RENDERED, ids=RENDERED_IDS)
def test_doc_equals_fresh_render_of_current_sources(template, output):
    fresh = render_report(
        SOURCES["assumptions"], SOURCES["metrics"], SOURCES["roi"], template, SOURCES["ablation"]
    )
    assert output.read_text(encoding="utf-8") == fresh, (
        f"{output.name} or its template changed without re-rendering"
    )


def test_model_hashes_in_model_card_footer_match_metrics():
    card = (ROOT / "docs" / "model_card.md").read_text(encoding="utf-8")
    hashes = _sources()[1]["hashes"]
    for name in ("train_data", "preprocess", "model", "calibrator"):
        assert f"- {name}: `{hashes[name]}`" in card


def test_stale_report_is_detected_when_any_source_changes(tmp_path):
    footer = FOOTER.findall(REPORT.read_text(encoding="utf-8"))[0]
    for name, path in SOURCES.items():
        copy = tmp_path / path.name
        copy.write_bytes(path.read_bytes() + b"\n")
        changed = {
            n: hashlib.sha256((copy if n == name else p).read_bytes()).hexdigest() for n, p in SOURCES.items()
        }
        assert source_hash(changed) != footer, name


def test_tracing_ignores_sha256_span_in_body(leaves):
    doc = "## 1. Data\n\nsha256 data input: `" + "1234567890abcdef" * 4 + "`.\n"
    assert untraceable(doc, leaves) == []
    assert untraceable(doc + "Accuracy 0.777.\n", leaves) == ["0.777"]


def test_negative_figures_are_traced_by_magnitude():
    """A shown "-0.002" is captured without its sign; it must match a source value of -0.0023."""
    assert untraceable("Change -0.002 in ECE.\n", [-0.0023]) == []
    assert untraceable("Change -0.003 in ECE.\n", [-0.0023]) == ["0.003"]
