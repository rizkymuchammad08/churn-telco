"""Render documents from assumptions.yaml, metrics.json, roi.json, and ablation.json via templates/*.md.j2.

Rendered documents (DOCS): the final report, the model card, and the recommendations.
These four sources are the only place numbers come from: assumptions.yaml, metrics.json, roi.json,
and ablation.json (the with/without `tenure` and `TotalCharges` diagnostic).

Strict rendering: a key missing from the sources or the template raises RenderError, and the output
is scanned for leftover placeholders (there must be none). A banner appears for synthetic data; the
"conditional value" label appears while save_rate_source = assumed. The footer holds the sha256 of
all four sources and their combined source_hash.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import jinja2

from src.config import ROOT, load_all
from src.data.validate import banner

PLACEHOLDER = re.compile(r"TODO|TBD|XXX|<\.\.\.>|\{\{|\}\}|\{%")
TEMPLATE = "templates/final_report.md.j2"
OUTPUT = "reports/final_report.md"

# (template, output). Every template in templates/ must be registered here (tested).
DOCS = (
    (TEMPLATE, OUTPUT),
    ("templates/model_card.md.j2", "docs/model_card.md"),
    ("templates/recommendations.md.j2", "docs/recommendations.md"),
)

# Display labels for segment values that are not self-explanatory in the raw data.
SEGMENT_VALUE_LABELS = {"SeniorCitizen": {"0": "not senior", "1": "senior"}}


class RenderError(ValueError):
    """Rendering failed. `code` is stable for tests and the CLI."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# --- Hashing ----------------------------------------------------------------------------------


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SOURCE_KEYS = ("assumptions", "metrics", "roi", "ablation")


def source_hash(hashes: dict) -> str:
    """One hash over the four source hashes, so a change in any source marks a document stale."""
    joined = "|".join(f"{k}={hashes[k]}" for k in SOURCE_KEYS)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# --- Building the template context ------------------------------------------------------------


def _need(d: dict, path: str):
    """Read a dotted path like "test.logreg.pr_auc"; raise RenderError if any key is missing."""
    node = d
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise RenderError("E_RENDER_MISSING_KEY", f"missing key: {path}")
        node = node[key]
    return node


def _ci(t: dict) -> dict:
    """Bootstrap interval bounds per metric: {metric: {"lo": ..., "hi": ...}}."""
    return {
        k: {"lo": _need(t, f"ci.{k}.lo"), "hi": _need(t, f"ci.{k}.hi")}
        for k in ("pr_auc", "roc_auc", "precision_at_k", "recall_at_k", "lift_at_k", "brier")
    }


def _fmt_config(config: dict) -> str:
    """A hyperparameter set as text, e.g. `C=0.1` or `learning_rate=0.1, max_depth=2`."""
    return ", ".join(f"{k}={v}" for k, v in config.items())


def _tuning(metrics: dict) -> dict:
    """Context for the tuning section. Metrics written before tuning existed render as disabled."""
    t = metrics.get("tuning")
    if not t or not t.get("enabled"):
        return {"enabled": False, "models": []}
    rows = []
    for name, m in _need(t, "models").items():
        rows.append(
            {
                "name": name,
                "default": _fmt_config(_need(m, "default")),
                "chosen": _fmt_config(_need(m, "chosen")),
                "replaced": _need(m, "replaced"),
                "winner": _fmt_config(_need(m, "winner")),
                "gain": _need(m, "gain_winner_vs_default"),
                "se": _need(m, "se_of_gain"),
                "required": _need(m, "required_gain"),
                "candidates": [
                    {
                        "config": _fmt_config(_need(c, "config")),
                        "cv": _need(c, "cv_mean"),
                        "is_default": _need(c, "is_default"),
                    }
                    for c in _need(m, "candidates")
                ],
            }
        )
    return {
        "enabled": True,
        "rule_hash": _need(t, "rule_hash"),
        "margin_se": _need(t, "margin_se"),
        "seed_offset": _need(t, "seed_offset"),
        "folds": _need(t, "folds"),
        "metric": _need(t, "metric"),
        "models": rows,
    }


def _models(metrics: dict) -> list[dict]:
    """One row per model: cross-validation score plus test-set metrics with their intervals."""
    rows = []
    for name in _need(metrics, "test"):
        t = _need(metrics, f"test.{name}")
        rows.append(
            {
                "name": name,
                "cv": _need(metrics, f"cv.{name}"),
                **{k: _need(t, k) for k in ("pr_auc", "roc_auc", "lift_at_k", "brier")},
                "ci": _ci(t),
            }
        )
    return rows


def _segments(by_segment: dict) -> list[dict]:
    """Flatten per-segment metrics into table rows, with readable value labels."""
    rows = []
    for feature, values in by_segment.items():
        for value, stats in values.items():
            label = SEGMENT_VALUE_LABELS.get(feature, {}).get(str(value), str(value))
            rows.append(
                {"feature": feature, "value": label}
                | {k: _need(stats, k) for k in ("n", "n_pos", "base_rate", "mean_p", "reliable")}
            )
    return rows


def _metric_row(t: dict) -> dict:
    return {
        **{
            k: _need(t, k)
            for k in ("pr_auc", "roc_auc", "precision_at_k", "recall_at_k", "lift_at_k", "brier", "ece")
        },
        "ci": _ci(t),
    }


def _policy_rows(roi_policies: dict) -> list[dict]:
    """Rows for the three compared policies, in a fixed order."""
    return [
        {
            "name": name,
            **{
                k: _need(roi_policies, f"{name}.{k}")
                for k in ("n_contacts", "n_churners_contacted", "net_value")
            },
        }
        for name in ("no_contact", "model_p_star", "random_k")
    ]


# Fixed row order for the ablation table; (key in ablation.json, label shown).
ABLATION_ROWS = (
    ("pr_auc", "PR-AUC"),
    ("roc_auc", "ROC-AUC"),
    ("lift_at_k", "Lift@k"),
    ("brier", "Brier"),
    ("ece", "ECE"),
)


def _ablation(ablation: dict, metrics: dict) -> dict:
    """Context for the section on excluded columns, from ablation.json only.

    The ablation must describe the same data as metrics.json; otherwise the report would pair
    a model with a diagnostic about different rows.
    """
    if _need(ablation, "input_data_hash") != _need(metrics, "hashes.input_data") or _need(
        ablation, "data"
    ) != _need(metrics, "data"):
        raise RenderError(
            "E_RENDER_STALE_ABLATION",
            "ablation.json was computed on different data than metrics.json: run make ablation",
        )
    columns = _need(ablation, "flagged_columns")
    corr = _need(ablation, "abs_corr_with_label")
    threshold = _need(ablation, "max_abs_corr")
    summary = _need(ablation, "summary")
    per_seed = _need(ablation, "per_seed")
    rows = [
        {
            "label": label,
            "without": _need(summary, f"{key}.without.mean"),
            "with": _need(summary, f"{key}.with.mean"),
            "delta": _need(summary, f"{key}.delta_with_minus_without.mean"),
            "delta_sd": _need(summary, f"{key}.delta_with_minus_without.std"),
            "seeds_better": _need(summary, f"{key}.seeds_with_better"),
        }
        for key, label in ABLATION_ROWS
    ]
    return {
        "policy": _need(ablation, "official_policy"),
        "columns": columns,
        "corr_rows": [{"column": c, "abs_corr": _need(corr, c)} for c in columns],
        "threshold": threshold,
        "detector_flagged": [c for c in columns if _need(corr, c) > threshold],
        "n_seeds": _need(ablation, "n_seeds"),
        "n_selection_changed": _need(ablation, "n_selection_changed"),
        "selected_without": ", ".join(sorted({r["without"]["selected"] for r in per_seed})),
        "selected_with": ", ".join(sorted({r["with"]["selected"] for r in per_seed})),
        "rows": rows,
        "pr_auc": rows[0],
    }


def build_context(assumptions: dict, metrics: dict, roi: dict, hashes: dict, ablation: dict) -> dict:
    """Collect everything the templates may use. Every value comes from one of the four sources."""
    selected = _need(metrics, "selected")
    test = _need(metrics, f"test.{selected}")
    calibration = _need(metrics, "calibration")
    data_source = _need(metrics, "data_source")

    models = _models(metrics)
    segments = _segments(_need(test, "by_segment"))
    best_on_test = max(models, key=lambda m: m["pr_auc"])
    policies = _policy_rows(_need(roi, "policies"))

    return {
        # Data and status
        "banner": banner({"data_source": data_source}),
        "data_source": data_source,
        "data_file": {"sample": "data/sample.csv", "raw": "data/raw"}.get(
            _need(metrics, "data"), _need(metrics, "data")
        ),
        "conditional": _need(assumptions, "save_rate_source") == "assumed",
        "save_rate_source": assumptions["save_rate_source"],
        # Selected model and test set
        "selected": selected,
        "primary_metric": _need(metrics, "primary_metric"),
        "split": {k: _need(metrics, f"split.{k}") for k in ("train", "calib", "test")},
        "n_test": _need(test, "n"),
        "n_pos": _need(test, "n_pos"),
        "base_rate": _need(test, "base_rate"),
        "k": _need(test, "k"),
        # Bootstrap intervals on the test metrics
        "ci_pct": _need(test, "ci.level_pct"),
        "n_boot": _need(test, "ci.n_resamples"),
        # Model comparison
        "models": models,
        "tuning": _tuning(metrics),
        "sel": _metric_row(test),
        "prior": _metric_row(_need(metrics, "test.prior_rate")),
        "best_test": best_on_test,
        "best_test_differs": best_on_test["name"] != selected,
        # Segments
        "segments": segments,
        "unreliable_segments": [f"{r['feature']}: {r['value']}" for r in segments if not r["reliable"]],
        # Business inputs
        "save_rate": _need(assumptions, "save_rate"),
        "currency": _need(assumptions, "currency"),
        "clv": _need(assumptions, "clv"),
        "cost_intervention": _need(assumptions, "cost_intervention"),
        "campaign_capacity": _need(assumptions, "campaign_capacity"),
        "population_per_period": _need(assumptions, "population_per_period"),
        "baseline_churn_rate": _need(assumptions, "baseline_churn_rate"),
        "target_churn_rate": _need(assumptions, "target_churn_rate"),
        # Hashes
        "input_data_hash": _need(metrics, "hashes.input_data"),
        "model_hashes": {
            k: _need(metrics, f"hashes.{k}") for k in ("train_data", "preprocess", "model", "calibrator")
        },
        "hashes": hashes,
        "source_hash": source_hash(hashes),
        # Excluded columns (policy decision) and its measured cost
        "ablation": _ablation(ablation, metrics),
        # Calibration
        "calibration_method": _need(calibration, "method"),
        "ece_raw": _need(calibration, "ece_raw"),
        "ece": _need(calibration, "ece_calibrated"),
        "tolerance": _need(calibration, "tolerance"),
        "within_tolerance": _need(calibration, "within_tolerance"),
        # Contact decision and business value
        "p_star": _need(roi, "p_star"),
        "k_test": _need(roi, "k_test"),
        "policies": policies,
        "model_policy": next(r for r in policies if r["name"] == "model_p_star"),
        "random_policy": next(r for r in policies if r["name"] == "random_k"),
        "inc_vs_none": _need(roi, "incremental.model_vs_no_contact"),
        "inc_vs_random": _need(roi, "incremental.model_vs_random_k"),
        "be_vs_none": _need(roi, "breakeven_save_rate.model_vs_no_contact"),
    }


# --- Rendering --------------------------------------------------------------------------------


def render_text(template_text: str, context: dict) -> str:
    """Render one template strictly: undefined names fail, and leftover placeholders fail."""
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True, autoescape=False)
    try:
        out = env.from_string(template_text).render(**context)
    except jinja2.UndefinedError as e:
        raise RenderError("E_RENDER_MISSING_KEY", str(e.message)) from e
    except jinja2.TemplateError as e:
        raise RenderError("E_RENDER_TEMPLATE", str(e)) from e

    if (m := PLACEHOLDER.search(out)) is not None:
        raise RenderError("E_RENDER_PLACEHOLDER", f"placeholder left in output: {m.group(0)!r}")
    return out


def render_report(assumptions_path, metrics_path, roi_path, template_path, ablation_path=None) -> str:
    """Render one template. `ablation_path` defaults to ablation.json next to metrics.json."""
    import yaml

    if ablation_path is None:
        ablation_path = Path(metrics_path).parent / "ablation.json"
    for p in (assumptions_path, metrics_path, roi_path, ablation_path, template_path):
        if not Path(p).exists():
            raise RenderError("E_RENDER_INPUT", f"file not found: {p}")

    assumptions = yaml.safe_load(Path(assumptions_path).read_text(encoding="utf-8"))
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    roi = json.loads(Path(roi_path).read_text(encoding="utf-8"))
    ablation = json.loads(Path(ablation_path).read_text(encoding="utf-8"))
    hashes = {
        "assumptions": sha256_file(assumptions_path),
        "metrics": sha256_file(metrics_path),
        "roi": sha256_file(roi_path),
        "ablation": sha256_file(ablation_path),
    }

    context = build_context(assumptions, metrics, roi, hashes, ablation)
    return render_text(Path(template_path).read_text(encoding="utf-8"), context)


def main(argv=None) -> int:
    params, _ = load_all()

    # Render every document first, and write only if all of them succeed.
    texts = {}
    try:
        for template, output in DOCS:
            texts[output] = render_report(
                ROOT / "assumptions.yaml",
                ROOT / params["paths"]["metrics"],
                ROOT / params["paths"]["roi"],
                ROOT / template,
                ROOT / params["paths"]["ablation"],
            )
    except RenderError as e:
        print(f"{e.code}: {e.message}")
        return 1

    for output, text in texts.items():
        out = ROOT / output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
        print(f"{output} written ({len(text.splitlines())} lines)")
    print("RENDER OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
