"""Tracing document numbers back to their sources (P6).

Two rules:
- Templates must not contain numeric literals. Exceptions: section numbers in headings, and years.
  Inside Jinja tags, a numeric string literal is only allowed if it is a number format (e.g. "%.3f").
- Every number in a rendered document must equal (at the precision shown) one of the numeric values
  in assumptions.yaml, metrics.json, roi.json, or ablation.json.
"""

from __future__ import annotations

import re

# --- Patterns ---------------------------------------------------------------------------------

# A number: plain (12, 0.85) or with thousands separators (1,234.5), not glued to letters/underscores.
NUMBER = re.compile(r"(?<![A-Za-z_\d.,])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")

# Whitelisted: section numbers in headings ("## 2.1 Title") and four-digit years.
SECTION_NUMBER = re.compile(r"^[ \t]*#{1,6}[ \t]+\d+(?:\.\d+)*\.?(?=[ \t])", re.MULTILINE)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

# Template syntax: expression tags (double braces), statement tags (brace-percent), comment tags (brace-hash).
TAG = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}", re.DOTALL)
STRING = re.compile(r"\"[^\"]*\"|'[^']*'")
FORMAT_SPEC = re.compile(r"^(?:%[-+ 0#]*\d*(?:\.\d+)?[dfeEgG]|\{:[,_]?(?:\.\d+)?[dfeEg%]\})$")

# The hash footer starts at "source_hash:" and runs to the end of the document.
FOOTER = re.compile(r"^source_hash:.*\Z", re.MULTILINE | re.DOTALL)

# A sha256 (64 hex chars) in a code span in the body, e.g. the input data hash; not a numeric claim.
HASH_SPAN = re.compile(r"`[0-9a-f]{64}`")

# CHANGELOG only: release headings ("## [0.1.0] - 2026-10-01") are not numeric claims;
# the version itself is checked against pyproject.
CHANGELOG_HEADING = re.compile(
    r"^##[ \t]+\[\d+\.\d+\.\d+\](?:[ \t]+-[ \t]+\d{4}-\d{2}-\d{2})?[ \t]*$", re.MULTILINE
)


def _strip_whitelist(text: str) -> str:
    """Blank out whitelisted numbers (section numbers, years) before looking for numeric claims."""
    return YEAR.sub(" ", SECTION_NUMBER.sub(" ", text))


# --- Rule 1: templates contain no numeric literals --------------------------------------------


def template_literals(template_text: str) -> list[str]:
    """Numeric literals in a template: text outside tags, plus expressions inside tags."""
    found = []

    outside_tags = TAG.sub(" ", template_text)
    found += NUMBER.findall(_strip_whitelist(outside_tags))

    for tag in TAG.findall(template_text):
        body = tag[2:-2]  # drop the opening and closing delimiters
        for s in STRING.findall(body):
            inner = s[1:-1]
            if NUMBER.search(inner) and not FORMAT_SPEC.match(inner):
                found.append(inner)
        found += NUMBER.findall(STRING.sub(" ", body))
    return found


# --- Rule 2: rendered numbers match a source value --------------------------------------------


def numeric_leaves(*sources) -> list[float]:
    """Every numeric value (at any depth) in the given dicts/lists. Booleans are not numbers."""
    out: list[float] = []

    def walk(node):
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            out.append(float(node))
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)

    for s in sources:
        walk(s)
    return out


def body_without_footer(text: str) -> str:
    """The document without its hash footer (hex hashes are not numeric claims)."""
    return FOOTER.sub("", text)


def numbers_in(text: str, *, changelog: bool = False) -> list[str]:
    """Numeric claims in a document: everything except hashes, whitelisted numbers, and the footer."""
    body = HASH_SPAN.sub(" ", body_without_footer(text))
    if changelog:
        body = CHANGELOG_HEADING.sub(" ", body)
    return NUMBER.findall(_strip_whitelist(body))


def _matches(token: str, leaves: list[float]) -> bool:
    """True if some source value equals `token` at the precision the token shows."""
    decimals = len(token.split(".")[1]) if "." in token else 0
    value = float(token.replace(",", ""))
    tolerance = 0.5 * 10**-decimals + 1e-9  # half a unit in the last shown digit
    # NUMBER captures digits without a sign, so a shown "-0.002" is looked up by magnitude.
    return any(abs(abs(v) - value) <= tolerance for v in leaves)


def untraceable(text: str, leaves: list[float], *, changelog: bool = False) -> list[str]:
    """Numbers in `text` that match no source value."""
    return [t for t in numbers_in(text, changelog=changelog) if not _matches(t, leaves)]


# --- Helpers ----------------------------------------------------------------------------------


def strip_code(md: str) -> str:
    """Markdown without code blocks and code spans (file names, commands, keys are not numeric claims)."""
    without_blocks = re.sub(r"```.*?```", " ", md, flags=re.DOTALL)
    return re.sub(r"`[^`\n]*`", " ", without_blocks)


def load_leaves(root) -> list[float]:
    """Numeric values from the four sources under `root`.

    Sources: assumptions.yaml, reports/metrics.json, reports/roi.json, reports/ablation.json.
    """
    import json
    from pathlib import Path

    import yaml

    root = Path(root)
    assumptions = yaml.safe_load((root / "assumptions.yaml").read_text(encoding="utf-8"))
    metrics = json.loads((root / "reports" / "metrics.json").read_text(encoding="utf-8"))
    roi = json.loads((root / "reports" / "roi.json").read_text(encoding="utf-8"))
    ablation = json.loads((root / "reports" / "ablation.json").read_text(encoding="utf-8"))
    return numeric_leaves(assumptions, metrics, roi, ablation)
