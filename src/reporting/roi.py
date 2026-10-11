"""Business value of contact policies (P5).

Net value of one policy over `years` years, with s = save_rate:

    net = years * (churners_contacted * s * clv
                   - contacts * cost_intervention
                   - operating_cost_per_year)

Rules:
- No contact: value is 0 (operating cost only applies when a campaign actually runs).
- Operating cost is charged ONCE per year, regardless of the number of contacts.
- Incremental value = net(policy) - net(comparison policy).
- breakeven_save_rate = the s at which incremental value is exactly 0.
- While save_rate_source = assumed, every value is conditional (`business_status`).

Entry point: python -m src.reporting.roi
(reads artifacts/test_predictions.csv, writes paths.roi).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import ROOT, business_status, load_all
from src.data.validate import banner
from src.models.targeting import capacity_for, p_star, plan_contacts


class RoiError(ValueError):
    """Invalid ROI input. `code` is stable for tests and the CLI."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Outcome:
    """Result of one policy on one batch.

    `n_churners_contacted` may be fractional (an expected value, e.g. for random contact).
    """

    name: str
    n_contacts: float
    n_churners_contacted: float

    def __post_init__(self):
        if self.n_contacts < 0 or self.n_churners_contacted < 0:
            raise RoiError("E_ROI_OUTCOME", "contact and churner counts must not be negative")
        if self.n_churners_contacted > self.n_contacts + 1e-9:
            raise RoiError("E_ROI_OUTCOME", "churners contacted exceeds the number of contacts")


# --- Value formulas ---------------------------------------------------------------------------


def operating_cost(assumptions: dict) -> float:
    return float(assumptions.get("operating_cost_per_year", 0.0))


def _check_years(years) -> None:
    if isinstance(years, bool) or not isinstance(years, (int, float)) or not years > 0:
        raise RoiError("E_ROI_YEARS", f"years must be > 0, got {years!r}")


def net_value(o: Outcome, assumptions: dict, *, save_rate: float | None = None, years: float = 1) -> float:
    """Net value of one policy outcome. A policy with zero contacts is worth exactly 0."""
    _check_years(years)
    if o.n_contacts == 0:
        return 0.0

    s = assumptions["save_rate"] if save_rate is None else save_rate
    revenue_saved = o.n_churners_contacted * s * assumptions["clv"]
    intervention_cost = o.n_contacts * assumptions["cost_intervention"]
    per_year = revenue_saved - intervention_cost - operating_cost(assumptions)
    return float(years * per_year)


def incremental_value(
    a: Outcome, b: Outcome, assumptions: dict, *, save_rate: float | None = None, years: float = 1
) -> float:
    """Value of policy `a` minus value of comparison policy `b`."""
    return net_value(a, assumptions, save_rate=save_rate, years=years) - net_value(
        b, assumptions, save_rate=save_rate, years=years
    )


def breakeven_save_rate(a: Outcome, b: Outcome, assumptions: dict) -> float | None:
    """The save_rate s* at which the incremental value of `a` over `b` is 0.

    Returns None when `a` does not contact more churners than `b`: no save_rate can then
    make `a` pay for its extra cost.
    """
    op_a = operating_cost(assumptions) if a.n_contacts > 0 else 0.0
    op_b = operating_cost(assumptions) if b.n_contacts > 0 else 0.0

    # Revenue gained per unit of save_rate, and the extra cost that revenue must cover.
    extra_revenue_per_unit_s = (a.n_churners_contacted - b.n_churners_contacted) * assumptions["clv"]
    if extra_revenue_per_unit_s <= 0:
        return None
    extra_cost = (a.n_contacts - b.n_contacts) * assumptions["cost_intervention"] + (op_a - op_b)
    return float(extra_cost / extra_revenue_per_unit_s)


def k_test(n_test: int, assumptions: dict) -> int:
    """Campaign capacity scaled to the test-set size (n = population per period gives full capacity)."""
    return capacity_for(n_test, assumptions)


# --- Building outcomes for each policy --------------------------------------------------------


def _labels(y_true) -> np.ndarray:
    y = np.asarray(y_true)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise RoiError("E_ROI_LABELS", "labels must be a 0/1 vector")
    return y.astype("int64")


def no_contact() -> Outcome:
    return Outcome("no_contact", 0, 0)


def outcome_from_mask(name: str, contacted, y_true) -> Outcome:
    y, mask = _labels(y_true), np.asarray(contacted, dtype=bool)
    if mask.shape != y.shape:
        raise RoiError("E_ROI_LABELS", "contact mask and labels have different lengths")
    return Outcome(name, int(mask.sum()), int(y[mask].sum()))


def random_policy(y_true, k: int) -> Outcome:
    """Contact k random customers: churners contacted = k * base_rate (expected value)."""
    y = _labels(y_true)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 0 <= k <= len(y):
        raise RoiError("E_ROI_K", f"k must be an integer in [0, {len(y)}], got {k!r}")
    expected_churners = float(k) * float(y.mean()) if len(y) else 0.0
    return Outcome("random_k", int(k), expected_churners)


def _with_rank(pred: pd.DataFrame) -> pd.DataFrame:
    """Add a 1-based `rank` column (highest p_churn first; ties broken by p_raw, then row order)."""
    if "rank" in pred.columns:
        return pred
    order = np.lexsort((np.arange(len(pred)), -pred["p_raw"].to_numpy(), -pred["p_churn"].to_numpy()))
    rank = np.empty(len(pred), dtype="int64")
    rank[order] = np.arange(1, len(pred) + 1)
    return pred.assign(rank=rank)


def model_policy(test_pred: pd.DataFrame, assumptions: dict, k: int | None = None) -> Outcome:
    """Contact by p* and capacity k (default k_test) using labeled test-set predictions."""
    for col in ("label", "p_churn"):
        if col not in test_pred.columns:
            raise RoiError("E_ROI_COLUMNS", f"column {col} is missing")
    plan = plan_contacts(_with_rank(test_pred), assumptions, k=k)
    _labels(test_pred["label"])
    return Outcome("model_p_star", plan.n_contacts, int(plan.contacts["label"].sum()))


# --- Assembling roi.json ----------------------------------------------------------------------


def _policy_dict(o: Outcome, assumptions: dict, years: float) -> dict:
    return {
        "n_contacts": o.n_contacts,
        "n_churners_contacted": o.n_churners_contacted,
        "net_value": net_value(o, assumptions, years=years),
    }


def build_roi(test_pred: pd.DataFrame, assumptions: dict, params: dict, *, years: float = 1) -> dict:
    """Compare the model policy with no contact and with random contact at the same k."""
    n = len(test_pred)
    k = k_test(n, assumptions)

    none = no_contact()
    model = model_policy(test_pred, assumptions, k=k)
    rand = random_policy(test_pred["label"].to_numpy(), k)

    return {
        "data_source": params["data_source"],
        "business_status": business_status(assumptions),
        "save_rate_source": assumptions["save_rate_source"],
        "currency": assumptions["currency"],
        "years": years,
        "n_test": n,
        "k_test": k,
        "p_star": p_star(assumptions),
        "inputs": {
            "clv": assumptions["clv"],
            "cost_intervention": assumptions["cost_intervention"],
            "save_rate": assumptions["save_rate"],
            "operating_cost_per_year": operating_cost(assumptions),
        },
        "policies": {p.name: _policy_dict(p, assumptions, years) for p in (none, model, rand)},
        "incremental": {
            "model_vs_no_contact": incremental_value(model, none, assumptions, years=years),
            "model_vs_random_k": incremental_value(model, rand, assumptions, years=years),
        },
        "breakeven_save_rate": {
            "model_vs_no_contact": breakeven_save_rate(model, none, assumptions),
            "model_vs_random_k": breakeven_save_rate(model, rand, assumptions),
        },
    }


def main(argv=None) -> int:
    params, assumptions = load_all()
    if (b := banner(params)) is not None:
        print(b)

    src = ROOT / params["paths"]["artifacts_dir"] / "test_predictions.csv"
    if not src.exists():
        print(f"E_ROI_INPUT: {src.relative_to(ROOT)} not found; run make train")
        return 1

    try:
        roi = build_roi(pd.read_csv(src), assumptions, params)
    except RoiError as e:
        print(f"{e.code}: {e.message}")
        return 1

    out = ROOT / params["paths"]["roi"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(roi, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")

    m = roi["policies"]["model_p_star"]
    print(
        f"k_test={roi['k_test']} contacts={m['n_contacts']} net={m['net_value']:.0f} {roi['currency']} "
        f"({roi['business_status']})"
    )
    print("ROI OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
