import re

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from nexus.db import get_engine
from nexus.sql_analytics import INCLUDE_TOKEN, list_queries, load_query, run_query

EXPECTED = [
    "contribution_by_category",
    "contribution_by_segment",
    "monthly_revenue",
    "product_rank_in_category",
]
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|truncate)\b", re.IGNORECASE)


@pytest.fixture(scope="module")
def engine():
    try:
        eng = get_engine()
        with eng.connect() as conn:
            ready = conn.execute(
                text("SELECT to_regclass('analytics.orders') IS NOT NULL AND to_regclass('analytics.refunds') IS NOT NULL")
            ).scalar()
    except Exception as exc:
        pytest.skip(f"PostgreSQL is not reachable: {type(exc).__name__}")
    if not ready:
        pytest.skip("run scripts/build_analytics.py first")
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def results(engine):
    return {name: run_query(engine, name) for name in EXPECTED}


@pytest.fixture(scope="module")
def lines(engine):
    """Independent pandas calculation of the line-level facts."""
    names = ["orders", "order_items", "products", "customers", "payments", "refunds", "shipments"]
    with engine.connect() as conn:
        f = {n: pd.read_sql_query(f"SELECT * FROM analytics.{n}", conn) for n in names}
    orders = f["orders"]
    orders = orders[orders["order_status"] == "Completed"][["order_id", "order_date", "customer_id"]]
    li = f["order_items"].merge(orders, on="order_id")
    li["share"] = li["line_total"] / li.groupby("order_id")["line_total"].transform("sum")
    ship = f["shipments"].set_index("order_id")["shipping_cost"]
    refund = f["refunds"].set_index("order_id")["refund_amount"]
    pay = f["payments"].set_index("order_id")
    failed = pay["amount"].where(pay["payment_status"] == "Failed", 0.0)
    li["revenue"] = li["line_total"]
    li["cogs"] = li["quantity"] * li["unit_cost"]
    li["shipping"] = li["order_id"].map(ship).fillna(0.0) * li["share"]
    li["refunded"] = li["order_id"].map(refund).fillna(0.0) * li["share"]
    li["failed"] = li["order_id"].map(failed).fillna(0.0) * li["share"]
    li = li.merge(f["products"][["product_id", "category"]], on="product_id")
    li = li.merge(f["customers"][["customer_id", "segment"]], on="customer_id")
    li["contribution"] = li["revenue"] - li["cogs"] - li["shipping"] - li["refunded"]
    li["month"] = pd.to_datetime(li["order_date"]).dt.to_period("M")
    return li


def test_query_list_is_complete():
    assert set(EXPECTED).issubset(list_queries())


def test_queries_are_read_only_and_use_only_the_analytics_schema():
    for name in EXPECTED:
        sql = load_query(name)
        assert not FORBIDDEN.search(sql), name
        assert ";" not in sql and "%" not in sql, name
        for other in ("core.", "raw.", "staging."):
            assert other not in sql, f"{name} reads {other}"
        assert "analytics." in sql, name


def test_shared_cte_is_expanded():
    for name in ["contribution_by_category", "contribution_by_segment", "product_rank_in_category"]:
        sql = load_query(name)
        assert INCLUDE_TOKEN not in sql
        assert "line_facts AS (" in sql
    assert "line_facts" not in load_query("monthly_revenue")


def test_monthly_revenue_reconciles(results, lines):
    m = results["monthly_revenue"]
    assert len(m) == 36
    assert abs(m["revenue"].sum() - lines["revenue"].sum()) < 0.5
    assert m["orders"].sum() == lines["order_id"].nunique()
    independent = lines.groupby("month")["revenue"].sum().sort_index().to_numpy()
    assert np.allclose(m["revenue"].to_numpy(dtype=float), independent, atol=0.01)


def test_period_over_period_changes(results):
    m = results["monthly_revenue"]
    rev = m["revenue"].to_numpy(dtype=float)
    mom = pd.to_numeric(m["mom_pct"]).to_numpy(dtype=float)
    yoy = pd.to_numeric(m["yoy_pct"]).to_numpy(dtype=float)
    assert np.isnan(mom[0])
    assert np.isnan(yoy[:12]).all()
    for k in range(1, 36):
        assert abs(mom[k] - round(100 * (rev[k] / rev[k - 1] - 1), 1)) <= 0.15
    for k in range(12, 36):
        assert abs(yoy[k] - round(100 * (rev[k] / rev[k - 12] - 1), 1)) <= 0.15


def test_moving_average_uses_a_three_month_window(results, lines):
    m = results["monthly_revenue"]
    independent = lines.groupby("month")["revenue"].sum().sort_index().rolling(3).mean().to_numpy()
    full = m["window_months"].to_numpy() == 3
    assert full.sum() == 34
    assert m["window_months"].iloc[0] == 1 and m["window_months"].iloc[1] == 2
    assert np.allclose(m["moving_avg_3m"].to_numpy(dtype=float)[full], independent[full], atol=0.01)


def test_contribution_totals_reconcile_with_the_source_tables(results, lines):
    for name in ["contribution_by_category", "contribution_by_segment"]:
        r = results[name]
        assert abs(r["revenue"].sum() - lines["revenue"].sum()) < 0.05
        assert abs(r["cogs"].sum() - lines["cogs"].sum()) < 0.05
        assert abs(r["shipping"].sum() - lines["shipping"].sum()) < 0.05
        assert abs(r["refunds"].sum() - lines["refunded"].sum()) < 0.05
        assert abs(r["uncollected_failed_payments"].sum() - lines["failed"].sum()) < 0.05
    cat = results["contribution_by_category"]["contribution"].sum()
    seg = results["contribution_by_segment"]["contribution"].sum()
    assert abs(cat - seg) < 0.05


def test_contribution_identity_and_ordering(results):
    for name in ["contribution_by_category", "contribution_by_segment"]:
        r = results[name]
        parts = r["revenue"] - r["cogs"] - r["shipping"] - r["refunds"]
        assert ((parts - r["contribution"]).abs() < 0.03).all()
        assert (r["contribution"].diff().dropna() <= 0).all()
        assert (r["contribution_pct"] < 100).all()


def test_category_and_segment_match_pandas(results, lines):
    by_cat = lines.groupby("category")[["revenue", "contribution"]].sum()
    cat = results["contribution_by_category"].set_index("category")
    assert len(cat) == 6
    assert np.allclose(cat["revenue"], by_cat.loc[cat.index, "revenue"], atol=0.01)
    assert np.allclose(cat["contribution"], by_cat.loc[cat.index, "contribution"], atol=0.02)
    by_seg = lines.groupby("segment")["contribution"].sum()
    seg = results["contribution_by_segment"].set_index("segment")
    assert set(seg.index) == {"Consumer", "Small Business", "Enterprise"}
    assert np.allclose(seg["contribution"], by_seg.loc[seg.index], atol=0.02)


def test_product_rank_matches_pandas(results, lines):
    r = results["product_rank_in_category"]
    assert r["category"].nunique() == 6
    assert (r["rank_in_category"] <= 3).all()
    for category, group in r.groupby("category"):
        assert group["rank_in_category"].is_monotonic_increasing
        assert group["contribution"].is_monotonic_decreasing
    prod = lines.groupby(["category", "product_id"])["contribution"].sum()
    best = prod.groupby(level=0).idxmax()
    top = r[r["rank_in_category"] == 1].set_index("category")
    for category, (_, product_id) in best.items():
        assert top.loc[category, "product_id"] == product_id
        assert abs(top.loc[category, "contribution"] - prod.loc[(category, product_id)]) < 0.02