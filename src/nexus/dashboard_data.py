"""Data layer of the dashboard: KPI queries and helpers over the cleaned analytics schema (SYNTHETIC data).

The KPI queries reuse the line_facts CTE of the analytics queries, so the dashboard and the SQL
analytics share one definition of revenue, COGS, shipping and refunds (see the metric dictionary
in docs/data_dictionary.md). Windows are inclusive and based on order date.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from nexus.config import get_settings
from nexus.sql_analytics import INCLUDE_TOKEN, SHARED_FILE, query_dir, run_query


def dashboard_sql_dir() -> Path:
    return get_settings().sql_dir / "dashboard"


def load_dashboard_query(name: str) -> str:
    """Read a dashboard query and expand the shared line_facts CTE if it uses it."""
    sql = (dashboard_sql_dir() / f"{name}.sql").read_text(encoding="utf-8").strip()
    if INCLUDE_TOKEN in sql:
        shared = (query_dir() / SHARED_FILE).read_text(encoding="utf-8").strip()
        sql = sql.replace(INCLUDE_TOKEN, shared)
    return sql


def check_window(start: date, end: date) -> None:
    if start > end:
        raise ValueError(f"start {start} is after end {end}")


def _run(engine: Engine, name: str, start: date, end: date) -> pd.DataFrame:
    check_window(start, end)
    with engine.connect() as conn:
        return pd.read_sql_query(text(load_dashboard_query(name)), conn, params={"start": start, "end": end})


@dataclass(frozen=True)
class Kpis:
    start: date
    end: date
    orders: int
    customers: int
    revenue: float
    cogs: float
    shipping: float
    refunds: float
    delivered: int
    late: int

    @property
    def aov(self) -> float | None:
        return self.revenue / self.orders if self.orders else None

    @property
    def gross_margin_pct(self) -> float | None:
        return 100 * (self.revenue - self.cogs) / self.revenue if self.revenue else None

    @property
    def contribution(self) -> float:
        return self.revenue - self.cogs - self.shipping - self.refunds

    @property
    def contribution_pct(self) -> float | None:
        return 100 * self.contribution / self.revenue if self.revenue else None

    @property
    def refund_rate_pct(self) -> float | None:
        return 100 * self.refunds / self.revenue if self.revenue else None

    @property
    def on_time_pct(self) -> float | None:
        return 100 * (self.delivered - self.late) / self.delivered if self.delivered else None


def fetch_kpis(engine: Engine, start: date, end: date) -> Kpis:
    k = _run(engine, "kpi_summary", start, end).iloc[0]
    d = _run(engine, "delivery_summary", start, end).iloc[0]
    return Kpis(
        start=start,
        end=end,
        orders=int(k["orders"]),
        customers=int(k["customers"]),
        revenue=float(k["revenue"]),
        cogs=float(k["cogs"]),
        shipping=float(k["shipping"]),
        refunds=float(k["refunds"]),
        delivered=int(d["delivered"]),
        late=int(d["late"]),
    )


def category_mix(engine: Engine, start: date, end: date) -> pd.DataFrame:
    df = _run(engine, "category_summary", start, end)
    for col in ("revenue", "contribution"):
        df[col] = df[col].astype(float)
    return df


def data_range(engine: Engine) -> tuple[date, date]:
    with engine.connect() as conn:
        row = pd.read_sql_query(text(load_dashboard_query("data_range")), conn).iloc[0]
    return pd.Timestamp(row["first_order"]).date(), pd.Timestamp(row["last_order"]).date()


def monthly_trend(engine: Engine) -> pd.DataFrame:
    """Full monthly series from the monthly_revenue query. Filter it for display with filter_months, so MoM and YoY stay correct at the window edges."""
    df = run_query(engine, "monthly_revenue")
    df["month"] = pd.to_datetime(df["month"])
    return df


def filter_months(trend: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Calendar months that touch the window."""
    check_window(start, end)
    first = pd.Timestamp(start.replace(day=1))
    return trend[(trend["month"] >= first) & (trend["month"] <= pd.Timestamp(end))].reset_index(drop=True)


def month_window(start_month: date, end_month: date) -> tuple[date, date]:
    """First day of the start month to the last day of the end month."""
    start = start_month.replace(day=1)
    end = end_month.replace(day=calendar.monthrange(end_month.year, end_month.month)[1])
    check_window(start, end)
    return start, end


def _is_whole_months(start: date, end: date) -> bool:
    return start.day == 1 and end.day == calendar.monthrange(end.year, end.month)[1]


def _shift_months(d: date, months: int) -> date:
    year, month0 = divmod(d.year * 12 + d.month - 1 + months, 12)
    return date(year, month0 + 1, 1)


def previous_period(start: date, end: date, first_order: date) -> tuple[date, date] | None:
    """The comparison window just before this one, or None if it does not fit in the data.

    A window of whole calendar months is compared with the same number of months before it, so a
    leap year does not shift the comparison. Any other window is compared with the equal number of
    days before it. The comparison window must start no earlier than the month of the first order.
    """
    check_window(start, end)
    if _is_whole_months(start, end):
        n = (end.year - start.year) * 12 + end.month - start.month + 1
        prev_start, prev_end = _shift_months(start, -n), start - timedelta(days=1)
    else:
        days = (end - start).days + 1
        prev_end = start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=days - 1)
    return (prev_start, prev_end) if prev_start >= first_order.replace(day=1) else None


def pct_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous == 0:
        return None
    return 100 * (current - previous) / previous