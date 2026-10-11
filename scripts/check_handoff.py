"""Check a handoff manifest: all keys present, files exist, open_issues empty."""

import sys
from pathlib import Path

import yaml

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
ROOT = Path(__file__).resolve().parents[1]


def main(path: str) -> int:
    target = ROOT / path
    if not target.exists():
        print(f"manifest not found: {path}")
        return 1
    manifest = yaml.safe_load(target.read_text(encoding="utf-8"))
    errs = [f"missing key: {k}" for k in KEYS if k not in manifest]
    errs += [f"file not found: {f}" for f in manifest.get("files", []) if not (ROOT / f).exists()]
    if manifest.get("open_issues"):
        errs.append(f"open_issues is not empty: {manifest['open_issues']}")
    for e in errs:
        print(e)
    print("HANDOFF OK" if not errs else "HANDOFF FAILED")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
