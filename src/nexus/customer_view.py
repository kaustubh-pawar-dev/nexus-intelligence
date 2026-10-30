"""View model of the Customer Intelligence page: retention matrix, KPI cards and tables (SYNTHETIC data).

No Streamlit or Plotly imports here, so everything can be tested directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nexus.dashboard_format import format_number, format_pct
from nexus.dashboard_view import Card

CUSTOMER_DEFINITIONS = {
    "Customers with orders": "Distinct customers with at least one completed order up to 2025-12-31.",
    "Champions' share of revenue": "Share of revenue from customers in the Champions RFM segment: frequency and recency both in the top two quartiles.",
    "Average retention after 1 month": "Share of customers who order again one month after their first order month, averaged over cohorts old enough to be observed.",
    "Average retention after 12 months": "Share of customers who order in the twelfth month after their first order month, averaged over cohorts old enough to be observed.",
}


def retention_matrix(cohorts: pd.DataFrame) -> pd.DataFrame:
    """Cohort by months-since-first-order matrix of retention percentages from the cohort_retention query.

    The query returns no row when nobody in a cohort ordered in a month. Such a cell is 0 when the
    month is already observed and stays empty (NaN) when it lies after the last month in the data.
    """
    c = cohorts.copy()
    c["cohort_month"] = pd.to_datetime(c["cohort_month"])
    c["months_since"] = c["months_since"].astype(int)
    idx = c["cohort_month"].dt.year * 12 + c["cohort_month"].dt.month
    last = int((idx + c["months_since"]).max())
    wide = c.pivot(index="cohort_month", columns="months_since", values="retention_pct")
    wide = wide.reindex(columns=range(0, last - int(idx.min()) + 1)).astype(float)
    values = wide.to_numpy(copy=True)
    cohort_idx = wide.index.year * 12 + wide.index.month
    for row, ci in enumerate(cohort_idx):
        observable = last - int(ci)
        values[row, : observable + 1] = np.nan_to_num(values[row, : observable + 1], nan=0.0)
    return pd.DataFrame(values, index=wide.index, columns=wide.columns)


def _curve_value(curve: pd.DataFrame, months: int) -> float | None:
    row = curve.loc[curve["months_since_first_order"] == months, "retention_pct"]
    return None if row.empty or pd.isna(row.iloc[0]) else float(row.iloc[0])


def customer_cards(rfm: pd.DataFrame, curve: pd.DataFrame) -> list[Card]:
    """Four headline cards from the rfm_segments query and the retention curve."""
    champions = rfm.loc[rfm["rfm_segment"] == "Champions", "revenue_share_pct"]
    share = float(champions.iloc[0]) if not champions.empty else None
    d = CUSTOMER_DEFINITIONS
    return [
        Card("Customers with orders", format_number(int(rfm["customers"].sum())), None, "off", d["Customers with orders"]),
        Card("Champions' share of revenue", format_pct(share), None, "off", d["Champions' share of revenue"]),
        Card("Average retention after 1 month", format_pct(_curve_value(curve, 1)), None, "off", d["Average retention after 1 month"]),
        Card("Average retention after 12 months", format_pct(_curve_value(curve, 12)), None, "off", d["Average retention after 12 months"]),
    ]


def rfm_table(rfm: pd.DataFrame) -> pd.DataFrame:
    """RFM segments for display, largest revenue first."""
    out = rfm.copy()
    for col in ("avg_recency_days", "avg_orders", "avg_revenue", "revenue", "revenue_share_pct"):
        out[col] = out[col].astype(float)
    out = out.sort_values("revenue", ascending=False).reset_index(drop=True)
    return out.rename(
        columns={
            "rfm_segment": "Segment",
            "customers": "Customers",
            "avg_recency_days": "Avg days since last order",
            "avg_orders": "Avg orders",
            "avg_revenue": "Avg revenue",
            "revenue": "Revenue",
            "revenue_share_pct": "Revenue share (%)",
        }
    )


def ltv_frame(ltv: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Observed lifetime value rows of one dimension ('segment' or 'channel'), best average first."""
    out = ltv[ltv["dimension"] == dimension].copy()
    for col in ("avg_orders", "avg_revenue", "avg_contribution_ltv", "median_contribution_ltv"):
        out[col] = out[col].astype(float)
    return out.sort_values("avg_contribution_ltv", ascending=False).reset_index(drop=True)