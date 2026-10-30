"""View model of the dashboard pages: KPI cards, period presets and chart preparation (SYNTHETIC data).

No Streamlit or Plotly imports here, so everything can be tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from nexus.dashboard_data import Kpis, _shift_months, month_window, pct_change
from nexus.dashboard_format import format_compact, format_delta, format_number, format_pct

PRESETS = ("Full period", "Last 12 months", "Last 6 months", "Custom range")
_RECENT_MONTHS = {"Last 12 months": 12, "Last 6 months": 6}

DEFINITIONS = {
    "Revenue": "Sum of line totals of completed orders, after order discounts and before refunds.",
    "Orders": "Completed orders placed in the period.",
    "Customers": "Distinct customers with at least one completed order in the period.",
    "Average order value": "Revenue divided by completed orders.",
    "Gross margin": "(Revenue - COGS) / revenue. COGS is quantity times unit cost.",
    "Contribution margin": "(Revenue - COGS - allocated shipping - allocated refunds) / revenue.",
    "Refund rate": "Refunds on the period's orders as a share of revenue. Refunds count in the month of the original order, not the refund date.",
    "On-time delivery": "Share of delivered shipments, for the period's completed orders, delivered on or before the promised date.",
}


@dataclass(frozen=True)
class Card:
    label: str
    value: str
    delta: str | None
    delta_color: str
    help: str


def build_cards(k: Kpis, prev: Kpis | None) -> list[Card]:
    """The eight KPI cards. Changes are relative for amounts and in percentage points for rates."""

    def old(attr: str):
        return getattr(prev, attr) if prev is not None else None

    def rel(cur, before):
        return format_delta(pct_change(cur, before)) if prev is not None else None

    def pts(cur, before):
        if prev is None or cur is None or before is None:
            return None
        return f"{cur - before:+.1f} pp"

    return [
        Card("Revenue", format_compact(k.revenue), rel(k.revenue, old("revenue")), "normal", DEFINITIONS["Revenue"]),
        Card("Orders", format_number(k.orders), rel(k.orders, old("orders")), "normal", DEFINITIONS["Orders"]),
        Card("Customers", format_number(k.customers), rel(k.customers, old("customers")), "normal", DEFINITIONS["Customers"]),
        Card(
            "Average order value",
            format_number(k.aov, 2),
            rel(k.aov, prev.aov if prev is not None else None),
            "normal",
            DEFINITIONS["Average order value"],
        ),
        Card(
            "Gross margin",
            format_pct(k.gross_margin_pct),
            pts(k.gross_margin_pct, prev.gross_margin_pct if prev is not None else None),
            "normal",
            DEFINITIONS["Gross margin"],
        ),
        Card(
            "Contribution margin",
            format_pct(k.contribution_pct),
            pts(k.contribution_pct, prev.contribution_pct if prev is not None else None),
            "normal",
            DEFINITIONS["Contribution margin"],
        ),
        Card(
            "Refund rate",
            format_pct(k.refund_rate_pct, 2),
            pts(k.refund_rate_pct, prev.refund_rate_pct if prev is not None else None),
            "inverse",  # a lower refund rate is better
            DEFINITIONS["Refund rate"],
        ),
        Card(
            "On-time delivery",
            format_pct(k.on_time_pct),
            pts(k.on_time_pct, prev.on_time_pct if prev is not None else None),
            "normal",
            DEFINITIONS["On-time delivery"],
        ),
    ]


def month_options(first: date, last: date) -> list[date]:
    """First day of every month from the first to the last order month."""
    months, m, stop = [], first.replace(day=1), last.replace(day=1)
    while m <= stop:
        months.append(m)
        m = _shift_months(m, 1)
    return months


def month_label(d: date) -> str:
    return f"{d:%Y-%m}"


def preset_window(preset: str, first: date, last: date) -> tuple[date, date] | None:
    """The date window of a period preset, or None for the custom range."""
    if preset == "Full period":
        return month_window(first, last)
    if preset in _RECENT_MONTHS:
        start = max(_shift_months(last.replace(day=1), -(_RECENT_MONTHS[preset] - 1)), first.replace(day=1))
        return month_window(start, last)
    if preset == "Custom range":
        return None
    raise ValueError(f"unknown period preset: {preset!r}")


def trend_for_chart(trend: pd.DataFrame) -> pd.DataFrame:
    """The 3-month average is only shown where three months of data exist."""
    out = trend.copy()
    out.loc[out["window_months"] < 3, "moving_avg_3m"] = float("nan")
    return out


def category_table(mix: pd.DataFrame) -> pd.DataFrame:
    """Category mix with revenue share and contribution margin as percentages."""
    out = mix.copy()
    out["revenue_share_pct"] = 100 * out["revenue"] / out["revenue"].sum()
    out["contribution_pct"] = 100 * out["contribution"] / out["revenue"]
    return out.sort_values("revenue", ascending=False).reset_index(drop=True)