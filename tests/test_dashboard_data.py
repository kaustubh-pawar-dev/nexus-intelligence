from datetime import date

import pandas as pd
import pytest
from sqlalchemy import text

from nexus.dashboard_data import (
    Kpis,
    category_mix,
    check_window,
    data_range,
    fetch_kpis,
    filter_months,
    load_dashboard_query,
    month_window,
    monthly_trend,
    pct_change,
    previous_period,
)
from nexus.db import get_engine
from nexus.sql_analytics import INCLUDE_TOKEN, run_query

D = date


def make_kpis(**changes) -> Kpis:
    base = dict(
        start=D(2024, 1, 1), end=D(2024, 12, 31), orders=10, customers=8, revenue=1000.0,
        cogs=600.0, shipping=50.0, refunds=30.0, delivered=9, late=3,
    )
    base.update(changes)
    return Kpis(**base)


# ---- logic that needs no database ----

def test_kpi_derived_metrics_by_hand():
    k = make_kpis()
    assert k.aov == 100.0
    assert k.gross_margin_pct == 40.0
    assert k.contribution == 320.0 and k.contribution_pct == 32.0
    assert k.refund_rate_pct == 3.0
    assert abs(k.on_time_pct - 100 * 6 / 9) < 1e-12


def test_kpi_derived_metrics_are_none_instead_of_dividing_by_zero():
    k = make_kpis(orders=0, revenue=0.0, cogs=0.0, shipping=0.0, refunds=0.0, delivered=0, late=0)
    assert k.aov is None and k.gross_margin_pct is None and k.contribution_pct is None
    assert k.refund_rate_pct is None and k.on_time_pct is None
    assert k.contribution == 0.0


def test_previous_period_compares_whole_months_with_whole_months_and_stays_inside_the_data():
    first = D(2023, 1, 5)  # the first order is not on the first day of the month
    assert previous_period(D(2025, 7, 1), D(2025, 12, 31), first) == (D(2025, 1, 1), D(2025, 6, 30))
    assert previous_period(D(2024, 1, 1), D(2024, 12, 31), first) == (D(2023, 1, 1), D(2023, 12, 31))  # leap year
    assert previous_period(D(2025, 3, 1), D(2025, 3, 31), first) == (D(2025, 2, 1), D(2025, 2, 28))
    assert previous_period(D(2023, 1, 1), D(2023, 6, 30), first) is None
    assert previous_period(D(2023, 7, 1), D(2023, 12, 31), first) == (D(2023, 1, 1), D(2023, 6, 30))


def test_previous_period_for_other_windows_uses_the_equal_number_of_days():
    start, end = D(2025, 3, 10), D(2025, 4, 2)
    ps, pe = previous_period(start, end, D(2023, 1, 5))
    assert (ps, pe) == (D(2025, 2, 14), D(2025, 3, 9)) and (pe - ps) == (end - start)


def test_month_window_handles_leap_february_and_month_ends():
    assert month_window(D(2024, 2, 15), D(2024, 2, 3)) == (D(2024, 2, 1), D(2024, 2, 29))
    assert month_window(D(2025, 1, 20), D(2025, 12, 5)) == (D(2025, 1, 1), D(2025, 12, 31))
    with pytest.raises(ValueError):
        month_window(D(2025, 5, 1), D(2025, 3, 1))


def test_filter_months_keeps_whole_months_touching_the_window():
    trend = pd.DataFrame({"month": pd.date_range("2024-01-01", periods=12, freq="MS"), "revenue": range(12)})
    out = filter_months(trend, D(2024, 3, 15), D(2024, 6, 10))
    assert list(out["month"].dt.month) == [3, 4, 5, 6]
    assert list(out["revenue"]) == [2, 3, 4, 5]


def test_pct_change():
    assert pct_change(110.0, 100.0) == pytest.approx(10.0)
    assert pct_change(90.0, 100.0) == pytest.approx(-10.0)
    assert pct_change(5.0, 0.0) is None and pct_change(None, 5.0) is None and pct_change(5.0, None) is None


def test_check_window_rejects_reversed_dates():
    with pytest.raises(ValueError, match="after"):
        check_window(D(2025, 2, 1), D(2025, 1, 1))
    check_window(D(2025, 1, 1), D(2025, 1, 1))


def test_dashboard_queries_share_the_cte_and_use_window_parameters():
    for name in ("kpi_summary", "category_summary"):
        sql = load_dashboard_query(name)
        assert INCLUDE_TOKEN not in sql and "line_facts AS (" in sql
        assert ":start" in sql and ":end" in sql
    delivery = load_dashboard_query("delivery_summary")
    assert ":start" in delivery and ":end" in delivery
    assert ":" not in load_dashboard_query("data_range").replace("::", "")


# ---- reconciliation against the database (cleaned analytics layer) ----

@pytest.fixture(scope="module")
def engine():
    return get_engine()


@pytest.fixture(scope="module")
def full(engine):
    return data_range(engine)


def test_data_range_is_inside_the_dataset_period(full):
    first, last = full
    assert D(2023, 1, 1) <= first <= last <= D(2025, 12, 31)


def test_full_range_matches_the_monthly_revenue_query(engine, full):
    k = fetch_kpis(engine, *full)
    monthly = run_query(engine, "monthly_revenue")
    assert k.orders == int(monthly["orders"].sum())
    assert abs(k.revenue - float(monthly["revenue"].sum())) < 0.5  # monthly values are rounded to cents


def test_full_range_revenue_matches_the_independent_daily_metrics_table(engine, full):
    k = fetch_kpis(engine, *full)
    with engine.connect() as conn:
        total = float(pd.read_sql_query(text("SELECT sum(gross_revenue) AS r FROM analytics.daily_metrics"), conn)["r"].iloc[0])
    assert abs(k.revenue - total) < 0.05


def test_windows_are_additive_for_revenue_and_orders(engine):
    year = fetch_kpis(engine, D(2024, 1, 1), D(2024, 12, 31))
    h1 = fetch_kpis(engine, D(2024, 1, 1), D(2024, 6, 30))
    h2 = fetch_kpis(engine, D(2024, 7, 1), D(2024, 12, 31))
    assert h1.orders + h2.orders == year.orders
    assert abs(h1.revenue + h2.revenue - year.revenue) < 0.01
    assert abs(h1.refunds + h2.refunds - year.refunds) < 0.01
    assert year.customers <= h1.customers + h2.customers  # a customer can order in both halves
    assert year.customers >= max(h1.customers, h2.customers)


def test_category_mix_adds_up_to_the_kpis_and_the_contribution_query(engine, full):
    k = fetch_kpis(engine, *full)
    mix = category_mix(engine, *full)
    assert abs(mix["revenue"].sum() - k.revenue) < 0.1
    assert abs(mix["contribution"].sum() - k.contribution) < 0.1
    other = run_query(engine, "contribution_by_category")
    assert abs(float(other["contribution"].sum()) - k.contribution) < 0.1


def test_delivery_counts_match_an_independent_calculation(engine):
    start, end = month_window(D(2024, 3, 1), D(2025, 2, 1))
    with engine.connect() as conn:
        s = pd.read_sql_query(
            text(
                "SELECT s.promised_delivery_date AS promised, s.delivered_date AS delivered, s.delivery_delay_days AS delay "
                "FROM analytics.shipments s JOIN analytics.orders o ON o.order_id = s.order_id "
                "WHERE o.order_status = 'Completed' AND s.shipment_status = 'Delivered' "
                "AND o.order_date BETWEEN :start AND :end"
            ),
            conn,
            params={"start": start, "end": end},
        )
    late_by_dates = int((pd.to_datetime(s["delivered"]) > pd.to_datetime(s["promised"])).sum())
    late_by_delay_column = int((s["delay"] > 0).sum())
    k = fetch_kpis(engine, start, end)
    assert k.delivered == len(s) > 0
    assert k.late == late_by_dates == late_by_delay_column


def test_a_window_is_never_larger_than_the_full_range(engine, full):
    whole = fetch_kpis(engine, *full)
    part = fetch_kpis(engine, D(2024, 3, 1), D(2024, 5, 31))
    assert 0 < part.revenue < whole.revenue and part.orders < whole.orders and part.customers <= whole.customers


def test_monthly_trend_keeps_year_over_year_values_outside_the_window(engine):
    trend = monthly_trend(engine)
    shown = filter_months(trend, D(2024, 1, 1), D(2024, 3, 31))
    assert len(shown) == 3 and shown["yoy_pct"].notna().all()