"""View model of the Predictive Risk page: cards, band summary, filters and export (SYNTHETIC data).

No Streamlit or Plotly imports here, so everything can be tested directly.
"""

from __future__ import annotations

import pandas as pd

from nexus.dashboard_format import format_number, format_pct
from nexus.dashboard_view import Card

BANDS = ("High", "Medium", "Low")
BAND_COLORS = {"High": "#c62828", "Medium": "#f9a825", "Low": "#1f77b4"}

RISK_DEFINITIONS = {
    "Active customers scored": "Customers with at least one completed order in the 180 days before the scoring date.",
    "High-risk customers": "Customers in the top 20% of predicted churn risk (deciles 1 and 2). Bands are relative to this population, not probability thresholds.",
    "Spend of high-risk customers": "Share of all spend in the 180 days before the scoring date that comes from high-risk customers. Observed spend, not a forecast.",
    "Average predicted churn": "Mean predicted probability of no completed order in the next 180 days. The scores rank customers usefully but are not exact probabilities.",
}


def risk_cards(scores: pd.DataFrame) -> list[Card]:
    d = RISK_DEFINITIONS
    total_spend = scores["monetary_180"].sum()
    high = scores[scores["risk_band"] == "High"]
    spend_share = 100 * high["monetary_180"].sum() / total_spend if total_spend else None
    return [
        Card("Active customers scored", format_number(len(scores)), None, "off", d["Active customers scored"]),
        Card("High-risk customers", format_number(len(high)), None, "off", d["High-risk customers"]),
        Card("Spend of high-risk customers", format_pct(spend_share), None, "off", d["Spend of high-risk customers"]),
        Card("Average predicted churn", format_pct(100 * scores["churn_probability"].mean()), None, "off", d["Average predicted churn"]),
    ]


def band_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per risk band: size, average predicted risk, recent activity and observed spend."""
    total_spend = scores["monetary_180"].sum()
    g = scores.groupby("risk_band").agg(
        customers=("customer_id", "size"),
        mean_probability=("churn_probability", "mean"),
        mean_recency_days=("recency_days", "mean"),
        mean_orders_180=("frequency_180", "mean"),
        spend_180=("monetary_180", "sum"),
    )
    g = g.reindex(list(BANDS)).dropna(how="all")
    g["customers_pct"] = 100 * g["customers"] / len(scores)
    g["spend_share_pct"] = 100 * g["spend_180"] / total_spend if total_spend else float("nan")
    g["mean_probability_pct"] = 100 * g["mean_probability"]
    return g.drop(columns="mean_probability").reset_index().rename(columns={"risk_band": "band"})


def filter_scores(scores: pd.DataFrame, bands, segments) -> pd.DataFrame:
    keep = scores["risk_band"].isin(list(bands)) & scores["segment"].isin(list(segments))
    return scores[keep].sort_values("risk_rank").reset_index(drop=True)


def customer_table(scores: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n highest-risk customers of the given scores, with display column names."""
    out = scores.sort_values("risk_rank").head(n).copy()
    out["churn_probability"] = 100 * out["churn_probability"]
    return out[
        ["risk_rank", "customer_id", "churn_probability", "risk_band", "recency_days", "frequency_180", "monetary_180", "segment"]
    ].rename(
        columns={
            "risk_rank": "Rank",
            "customer_id": "Customer",
            "churn_probability": "Churn probability (%)",
            "risk_band": "Band",
            "recency_days": "Days since last order",
            "frequency_180": "Orders (180 days)",
            "monetary_180": "Spend (180 days)",
            "segment": "Segment",
        }
    ).reset_index(drop=True)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")