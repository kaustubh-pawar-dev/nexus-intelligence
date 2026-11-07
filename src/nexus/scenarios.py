"""What-if scenarios on the findings: a retention campaign, fewer refunds and cheaper goods (SYNTHETIC data).

A scenario is a calculation under stated assumptions, not a forecast. Effects and costs are inputs
chosen by the person running the scenario. The data supplies the baseline: churn probabilities, recent
spend and the cost structure of each category.

No Streamlit, Plotly or database imports here, so everything can be tested directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nexus.findings import expected_loss

RANKINGS = ("probability", "expected_loss")
COST_COLUMNS = ("cogs", "shipping", "refunds")


# ---------- retention campaign ----------

def _order(scores: pd.DataFrame, ranking: str) -> np.ndarray:
    if ranking not in RANKINGS:
        raise ValueError(f"ranking must be one of {RANKINGS}, got {ranking!r}")
    key = scores["churn_probability"].to_numpy(float) if ranking == "probability" else expected_loss(scores).to_numpy()
    return np.argsort(-key, kind="stable")


def campaign(
    scores: pd.DataFrame,
    n_contacts: int,
    ranking: str,
    effect: float,
    cost_per_contact: float,
    margin_rate: float,
) -> dict:
    """A retention campaign aimed at the top customers by churn probability or by expected loss.

    effect is the relative reduction of churn probability among contacted customers (0 to 1), so the
    spend retained is effect times their expected loss. Only the contribution margin of that spend counts
    as a benefit.
    """
    if not 0.0 <= effect <= 1.0:
        raise ValueError("effect must be between 0 and 1")
    if n_contacts < 1:
        raise ValueError("n_contacts must be at least 1")
    if cost_per_contact < 0 or not 0.0 <= margin_rate <= 1.0:
        raise ValueError("cost must not be negative and margin_rate must be between 0 and 1")
    scores = scores.reset_index(drop=True)
    n = min(int(n_contacts), len(scores))
    idx = _order(scores, ranking)[:n]
    loss = float(expected_loss(scores).to_numpy()[idx].sum())
    spend = float(scores["monetary_180"].to_numpy(float)[idx].sum())
    retained = effect * loss
    contribution = retained * margin_rate
    cost = n * cost_per_contact
    net = contribution - cost
    at_stake = loss * margin_rate
    return {
        "ranking": ranking,
        "contacts": n,
        "mean_probability": float(scores["churn_probability"].to_numpy(float)[idx].mean()),
        "spend": spend,
        "expected_loss": loss,
        "spend_retained": retained,
        "contribution_retained": contribution,
        "cost": cost,
        "net_benefit": net,
        "return_on_cost": net / cost if cost else None,
        "break_even_effect": cost / at_stake if at_stake else None,
    }


def campaign_sensitivity(
    scores: pd.DataFrame, n_contacts: int, cost_per_contact: float, margin_rate: float, max_effect: float = 0.6
) -> pd.DataFrame:
    """Net benefit of the campaign for effects from 0 to max_effect, for both ranking rules."""
    effects = np.round(np.arange(0.0, max_effect + 1e-9, 0.05), 2)
    rows = []
    for e in effects:
        rows.append(
            {
                "effect": float(e),
                "net_probability": campaign(scores, n_contacts, "probability", float(e), cost_per_contact, margin_rate)["net_benefit"],
                "net_expected_loss": campaign(scores, n_contacts, "expected_loss", float(e), cost_per_contact, margin_rate)["net_benefit"],
            }
        )
    return pd.DataFrame(rows)


# ---------- margin scenarios ----------

def totals(comp: pd.DataFrame) -> dict:
    """Revenue, costs and contribution margin of a category table (revenue, cogs, shipping, refunds)."""
    revenue = float(comp["revenue"].sum())
    cogs, shipping, refunds = (float(comp[c].sum()) for c in COST_COLUMNS)
    contribution = revenue - cogs - shipping - refunds
    return {
        "revenue": revenue,
        "cogs": cogs,
        "shipping": shipping,
        "refunds": refunds,
        "contribution": contribution,
        "contribution_pct": 100 * contribution / revenue if revenue else float("nan"),
    }


def adjust(comp: pd.DataFrame, categories, column: str, factor: float) -> pd.DataFrame:
    """A copy of the category table with one cost column multiplied by factor in the chosen categories."""
    if column not in COST_COLUMNS:
        raise ValueError(f"column must be one of {COST_COLUMNS}")
    if factor < 0:
        raise ValueError("factor must not be negative")
    out = comp.copy()
    out[column] = out[column].astype(float)
    out.loc[out["category"].isin(list(categories)), column] *= factor
    return out


def _compare(base: pd.DataFrame, scenario: pd.DataFrame, column: str) -> dict:
    b, s = totals(base), totals(scenario)
    by_category = pd.DataFrame(
        {
            "category": base["category"].to_numpy(),
            "baseline": base[column].astype(float).to_numpy(),
            "scenario": scenario[column].astype(float).to_numpy(),
        }
    )
    by_category["change"] = by_category["scenario"] - by_category["baseline"]
    return {
        "baseline": b,
        "scenario": s,
        "delta_contribution": s["contribution"] - b["contribution"],
        "delta_points": s["contribution_pct"] - b["contribution_pct"],
        "by_category": by_category,
    }


def refund_scenario(comp: pd.DataFrame, categories, reduction_pct: float) -> dict:
    """Refunds in the chosen categories fall by reduction_pct percent. Volumes and prices stay the same."""
    if not 0.0 <= reduction_pct <= 100.0:
        raise ValueError("reduction_pct must be between 0 and 100")
    scenario = adjust(comp, categories, "refunds", 1 - reduction_pct / 100)
    return _compare(comp, scenario, "refunds")


def cogs_scenario(comp: pd.DataFrame, categories, change_pct: float) -> dict:
    """Cost of goods in the chosen categories changes by change_pct percent (negative is cheaper). Volumes and prices stay the same."""
    if change_pct <= -100.0:
        raise ValueError("change_pct must be above -100")
    scenario = adjust(comp, categories, "cogs", 1 + change_pct / 100)
    return _compare(comp, scenario, "cogs")