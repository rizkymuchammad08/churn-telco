"""P6: manifests p1..p6 complete and honest; files and YAML keys exist; no open_issues or placeholders."""

import ast
import importlib
import io
import re
import tokenize

import pytest
import yaml

from src.config import ASSUMPTIONS_SPEC, ROOT

pytestmark = pytest.mark.p6

PARTS = [f"p{n}" for n in range(1, 7)]
KEYS = [
    "part",
    "adapter",
    "files",
    "yaml_keys_consumed",
    "yaml_keys_produced",
    "public_api",
    "check",
    "open_issues",
]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
ASSUMPTIONS = yaml.safe_load((ROOT / "assumptions.yaml").read_text(encoding="utf-8"))


def manifest(part):
    return yaml.safe_load((ROOT / "handoff" / f"{part}.yaml").read_text(encoding="utf-8"))


MANIFESTS = {p: manifest(p) for p in PARTS}


# --- manifest completeness -----------------------------------------------------
@pytest.mark.parametrize("part", PARTS)
def test_manifest_has_all_keys_and_consistent_identity(part):
    m = MANIFESTS[part]
    assert [k for k in KEYS if k not in m] == []
    assert m["part"] == part.upper()
    assert m["check"] == f"make check-{part}"
    assert m["adapter"] == MANIFESTS["p1"]["adapter"]
    assert m["files"] and all(isinstance(f, str) for f in m["files"])
    for side in ("yaml_keys_consumed", "yaml_keys_produced"):
        assert set(m[side]) >= {"params", "assumptions"}
        assert all(isinstance(v, list) for v in m[side].values())
    assert isinstance(m["public_api"], list) and isinstance(m["open_issues"], list)


@pytest.mark.parametrize("part", PARTS)
def test_registered_files_exist(part):
    m = MANIFESTS[part]
    missing = [f for f in m["files"] + m.get("modifies", []) if not (ROOT / f).is_file()]
    assert missing == []


def test_each_file_has_one_owner():
    owner = {}
    for part, m in MANIFESTS.items():
        for f in m["files"]:
            assert f not in owner, f"{f} is owned by {owner[f]} and {part}"
            owner[f] = part
    for part, m in MANIFESTS.items():
        for f in m.get("modifies", []):
            assert f in owner, f"{part} modifies {f}, which no part owns"
            assert owner[f] != part, f"{part} lists {f} under both files and modifies"


def test_every_test_file_is_registered_and_marked_with_its_part():
    registered = {}
    for part, m in MANIFESTS.items():
        for f in m["files"]:
            if re.fullmatch(r"tests/test_[^/]+\.py", f):
                registered[f] = part
    on_disk = {f"tests/{p.name}" for p in (ROOT / "tests").glob("test_*.py")}
    assert on_disk == set(registered), sorted(on_disk ^ set(registered))
    for f, part in registered.items():
        src = (ROOT / f).read_text(encoding="utf-8")
        assert f"pytestmark = pytest.mark.{part}" in src, f"{f} is not marked {part}"


# --- YAML keys -----------------------------------------------------------------
def _has_path(d, dotted):
    for key in dotted.split("."):
        if not isinstance(d, dict) or key not in d:
            return False
        d = d[key]
    return True


@pytest.mark.parametrize("part", PARTS)
def test_yaml_keys_exist(part):
    m = MANIFESTS[part]
    consumed, produced = m["yaml_keys_consumed"], m["yaml_keys_produced"]
    assert [k for k in consumed["params"] if k not in PARAMS] == []
    assert [k for k in consumed["assumptions"] if k not in ASSUMPTIONS] == []
    assert [k for k in produced["params"] if not _has_path(PARAMS, k)] == []
    assert [k for k in produced["assumptions"] if k not in ASSUMPTIONS] == []


@pytest.mark.parametrize("part", PARTS)
def test_optional_keys_are_declared_in_schema_but_not_required(part):
    optional = MANIFESTS[part].get("yaml_keys_optional", {})
    for key in optional.get("assumptions", []):
        assert key in ASSUMPTIONS_SPEC
        assert key not in MANIFESTS[part]["yaml_keys_consumed"]["assumptions"]


def test_every_params_and_assumptions_key_is_claimed_by_some_part():
    produced_p = {k.split(".")[0] for m in MANIFESTS.values() for k in m["yaml_keys_produced"]["params"]}
    produced_a = {k for m in MANIFESTS.values() for k in m["yaml_keys_produced"]["assumptions"]}
    assert set(PARAMS) <= produced_p
    assert set(ASSUMPTIONS) <= produced_a


# --- public_api really exists ----------------------------------------------------
def _api_entries():
    for part, m in MANIFESTS.items():
        for entry in m["public_api"]:
            yield part, entry


@pytest.mark.parametrize("part,entry", list(_api_entries()), ids=lambda v: v if isinstance(v, str) else None)
def test_public_api_entry_resolves_to_an_object_in_a_registered_file(part, entry):
    dotted = entry.split("(")[0].strip()
    *mod, attr = dotted.split(".")
    suffix = "/".join(mod) + ".py"
    matches = [f for f in MANIFESTS[part]["files"] if f.endswith("/" + suffix)]
    assert len(matches) == 1, f"{entry}: module {'.'.join(mod)} is not in {part} files"
    module = importlib.import_module(matches[0][:-3].replace("/", "."))
    assert hasattr(module, attr), f"{entry}: {attr} does not exist"


# --- open_issues empty ---------------------------------------------------------
@pytest.mark.parametrize("part", PARTS)
def test_open_issues_empty(part):
    assert MANIFESTS[part]["open_issues"] == []


# --- ownership of test_split / test_leakage --------------------------------------
def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_split_tests_belong_to_p4_not_p3():
    assert "tests/test_split.py" in MANIFESTS["p4"]["files"]
    assert "tests/test_split.py" not in MANIFESTS["p3"]["files"]
    assert "src/models/split.py" in MANIFESTS["p4"]["files"]


def test_p3_leakage_tests_do_not_depend_on_p4_code():
    assert "tests/test_leakage.py" in MANIFESTS["p3"]["files"]
    p4_modules = {f[:-3].replace("/", ".") for f in MANIFESTS["p4"]["files"] if f.startswith("src/")}
    used = set(_imports(ROOT / "tests" / "test_leakage.py"))
    assert not {u for u in used if u in p4_modules or u.startswith("src.models")}
    assert "conftest" not in used  # shared P4 fixtures


def test_fit_leakage_tests_live_in_p4_train_and_calibrate():
    for f in ("tests/test_train.py", "tests/test_calibrate.py"):
        assert f in MANIFESTS["p4"]["files"]


# --- zero placeholders across the whole tree ---------------------------------------
BANNED = re.compile(r"TODO|TBD|XXX|<\.\.\.>|\{\{")
# .github: workflow files use the GitHub expression syntax, which looks like a template placeholder.
SKIP_DIRS = {
    ".git",
    ".github",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "artifacts",
    "data",
    "templates",
}
TEXT_SUFFIXES = {
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".toml",
    ".cfg",
    ".ini",
    ".ipynb",
    ".html",
    ".example",
}
TEXT_NAMES = {"Makefile", ".gitignore"}


def _tree_files():
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if not p.is_file() or SKIP_DIRS & set(rel.parts) or any(s.endswith(".egg-info") for s in rel.parts):
            continue
        yield p


def _placeholder_hits(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        # Code: only comments are scanned (the scanner's own patterns live in strings).
        toks = tokenize.generate_tokens(io.StringIO(text).readline)
        return [t.string for t in toks if t.type == tokenize.COMMENT and BANNED.search(t.string)]
    if path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES:
        return [ln for ln in text.splitlines() if BANNED.search(ln)]
    return []


def test_tree_has_zero_placeholders():
    files = list(_tree_files())
    assert len(files) > 50  # the scan is not vacuous
    bad = {str(p.relative_to(ROOT)): hits for p in files if (hits := _placeholder_hits(p))}
    assert bad == {}


def test_placeholder_scan_catches_each_kind(tmp_path):
    for i, bad in enumerate(["TODO fill in", "TBD", "XXX", "<...>", "{{ x }}"]):
        md = tmp_path / f"a{i}.md"
        md.write_text(f"clean line\n{bad}\n", encoding="utf-8")
        assert _placeholder_hits(md) == [bad]
    py = tmp_path / "m.py"
    py.write_text('x = "TODO in string"\n# TODO in comment\n', encoding="utf-8")
    assert _placeholder_hits(py) == ["# TODO in comment"]


def test_placeholders_are_allowed_only_in_templates():
    assert "{{" in (ROOT / "templates" / "final_report.md.j2").read_text(encoding="utf-8")
