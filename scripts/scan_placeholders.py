"""Final output: zero placeholders and zero hand-typed numbers. Used by check-p6; templates/ is not scanned.

Numbers: every number in .md prose (outside code and outside the hash footer) must equal a value in
assumptions.yaml, reports/metrics.json, reports/roi.json, or reports/ablation.json at the displayed precision.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reporting.trace import load_leaves, strip_code, untraceable  # noqa: E402

BANNED = re.compile(r"TODO|TBD|XXX|<\.\.\.>|\{\{")
SUFFIXES = {".md", ".json", ".yaml", ".yml", ".txt", ".csv", ".html"}


def _files(paths: list[str]):
    for name in paths:
        p = ROOT / name
        files = [p] if p.is_file() else sorted(p.rglob("*")) if p.exists() else []
        yield from (f for f in files if f.is_file() and f.suffix in SUFFIXES)


def scan_numbers(text: str, leaves: list[float], *, changelog: bool = False) -> list[str]:
    return untraceable(strip_code(text), leaves, changelog=changelog)


def main(paths: list[str]) -> int:
    bad, bad_numbers = [], []
    leaves = load_leaves(ROOT)
    for f in _files(paths):
        text = f.read_text(encoding="utf-8")
        rel = f.relative_to(ROOT) if f.is_relative_to(ROOT) else f
        for n, line in enumerate(text.splitlines(), 1):
            if BANNED.search(line):
                bad.append(f"{rel}:{n}: {line.strip()[:80]}")
        if f.suffix == ".md":
            bad_numbers += [
                f"{rel}: {t}" for t in scan_numbers(text, leaves, changelog=f.name == "CHANGELOG.md")
            ]
    for b in bad:
        print(b)
    for b in bad_numbers:
        print(f"UNSOURCED NUMBER {b}")
    print("PLACEHOLDERS: FOUND" if bad else "PLACEHOLDERS: NONE")
    print("HAND-TYPED NUMBERS: FOUND" if bad_numbers else "HAND-TYPED NUMBERS: NONE")
    return 1 if bad or bad_numbers else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
