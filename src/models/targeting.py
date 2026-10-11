"""Contact decision: break-even threshold p* and campaign capacity k.

A customer is contacted when (a) they rank within capacity k and (b) p_churn >= p*, where
p* = cost_intervention / (save_rate * clv). Because rank is consistent with p_churn, the
contact set is always a prefix of the ranking. Business value (ROI) is computed in P5, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import breakeven_threshold


class TargetingError(ValueError):
    """Invalid targeting input. `code` is stable for tests and the CLI."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def p_star(assumptions: dict) -> float:
    return breakeven_threshold(assumptions)


def capacity_for(n: int, assumptions: dict) -> int:
    """Campaign capacity for a batch of n rows.

    `campaign_capacity` is scaled to the batch size relative to `population_per_period`,
    rounded half up. A batch of exactly `population_per_period` rows yields `campaign_capacity`.
    """
    frac = assumptions["campaign_capacity"] / assumptions["population_per_period"]
    return min(n, int(n * frac + 0.5))


@dataclass(frozen=True)
class ContactPlan:
    p_star: float
    k: int
    contacts: pd.DataFrame  # rows of `pred` selected for contact, ordered by rank

    @property
    def n_contacts(self) -> int:
        return len(self.contacts)


def plan_contacts(pred: pd.DataFrame, assumptions: dict, k: int | None = None) -> ContactPlan:
    """Select contacts from `predict` output (needs `p_churn` and `rank` columns).

    `k` defaults to the capacity derived from `assumptions`.
    """
    for col in ("p_churn", "rank"):
        if col not in pred.columns:
            raise TargetingError("E_TARGET_COLUMNS", f"column {col} is missing")
    if k is None:
        k = capacity_for(len(pred), assumptions)
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise TargetingError("E_TARGET_CAPACITY", f"k must be an integer >= 0, got {k!r}")
    star = p_star(assumptions)
    top = pred.sort_values("rank", kind="mergesort").head(k)
    return ContactPlan(p_star=star, k=k, contacts=top[top["p_churn"] >= star])
