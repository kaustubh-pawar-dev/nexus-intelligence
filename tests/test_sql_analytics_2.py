import re

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from nexus.db import get_engine
from nexus.sql_analytics import SHARED_FILE, list_queries, load_query, query_dir, retention_curve, run_query

NEW = [
    "cohort_retention",
    "customer_ltv",
    "inventory_turnover",
    "revenue_leakage",
    "rfm_segments",
    "supplier_performance",
]
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|truncate)\b", re.IGNORECASE)


@pytest.fixture(scope="module")
def engine():
    try:
        eng = get_engine()
        with eng.connect() as conn:
            ready = conn.execute(
                text("SELECT to_regclass('analytics.orders') IS NOT NULL AND to_regclass('analytics.inventory_daily') IS NOT NULL")
            ).scalar()
    except Exception as exc:
        pytest.skip(f"PostgreSQL is not reachable: {type(exc).__name__}")
    if not ready:
        pytest.skip("run scripts/build_analytics.py first")
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def results(engine):
    return {name: run_query(engine, name) for name in NEW}


@pytest.fixture(scope="module")
def data(engine):
    names = ["orders", "order_items", "products", "customers", "payments", "refunds",
             "shipments", "purchase_orders", "suppliers", "inventory_daily"]
    with engine.connect() as conn:
        return {n: pd.read_sql_query(f"SELECT * FROM analytics.{n}", conn) for n in names}


@pytest.fixture(scope="module")
def lines(data):
    """Independent pandas calculation of line-level facts (same allocation rule as the SQL)."""
    orders = data["orders"]
    orders = orders[orders["order_status"] == "Completed"][["order_id", "order_date", "customer_id"]]
    li = data["order_items"].merge(orders, on="order_id")
    li["share"] = li["line_total"] / li.groupby("order_id")["line_total"].transform("sum")
    ship = data["shipments"].set_index("order_id")["shipping_cost"]
    refund = data["refunds"].set_index("order_id")["refund_amount"]
    li["revenue"] = li["line_total"]
    li["cogs"] = li["quantity"] * li["unit_cost"]
    li["shipping"] = li["order_id"].map(ship).fillna(0.0) * li["share"]
    li["refunded"] = li["order_id"].map(refund).fillna(0.0) * li["share"]
    li["contribution"] = li["revenue"] - li["cogs"] - li["shipping"] - li["refunded"]
    return li


def _ntile(n, k=4):
    base, extra = divmod(n, k)
    out = []
    for b in range(1, k + 1):
        out += [b] * (base + (1 if b <= extra else 0))
    return np.array(out)


def _rfm_frame(data):
    o = data["orders"]
    o = o[o["order_status"] == "Completed"]
    value = data["order_items"].groupby("order_id")["line_total"].sum().rename("v")
    per = o.merge(value, on="order_id")
    g = per.groupby("customer_id").agg(last=("order_date", "max"), frequency=("order_id", "nunique"), monetary=("v", "sum"))
    g["monetary"] = g["monetary"].round(2)
    g["recency"] = (pd.Timestamp("2025-12-31") - pd.to_datetime(g["last"])).dt.days
    g = g.reset_index()
    buckets = _ntile(len(g))

    def score(cols, ascending):
        ordered = g.sort_values(cols, ascending=ascending)
        return pd.Series(buckets, index=ordered.index)

    g["r"] = score(["recency", "customer_id"], [False, True])
    g["f"] = score(["frequency", "customer_id"], [True, True])
    g["m"] = score(["monetary", "customer_id"], [True, True])

    def label(r, f):
        if f >= 3:
            return "Champions" if r >= 3 else "At risk"
        return "New / promising" if r >= 3 else "Dormant"

    g["segment"] = [label(r, f) for r, f in zip(g["r"], g["f"])]
    return g


def test_new_queries_are_listed():
    assert set(NEW).issubset(list_queries())


def test_new_queries_are_read_only_and_use_only_the_analytics_schema():
    for name in NEW:
        sql = load_query(name)
        assert not FORBIDDEN.search(sql), name
        assert ";" not in sql and "%" not in sql, name
        for other in ("core.", "raw.", "staging."):
            assert other not in sql, f"{name} reads {other}"
        assert "analytics." in sql, name


def test_shared_cte_exposes_customer_id():
    shared = (query_dir() / SHARED_FILE).read_text(encoding="utf-8")
    assert "o.customer_id" in shared
    assert "@LINE_FACTS@" not in load_query("customer_ltv")


def test_retention_curve_counts_only_observable_cohorts():
    cohorts = pd.DataFrame(
        {
            "cohort_month": [pd.Timestamp("2025-01-01").date()] * 2 + [pd.Timestamp("2025-02-01").date()],
            "months_since": [0, 1, 0],
            "active_customers": [10, 5, 20],
            "cohort_size": [10, 10, 20],
        }
    )
    curve = retention_curve(cohorts, offsets=(0, 1)).set_index("months_since_first_order")
    assert curve.loc[0, "retention_pct"] == 100.0
    assert curve.loc[0, "customers_in_cohorts"] == 30
    assert curve.loc[1, "cohorts_observed"] == 1
    assert curve.loc[1, "customers_in_cohorts"] == 10
    assert curve.loc[1, "retention_pct"] == 50.0


def test_cohort_retention_matches_pandas(results, data):
    r = results["cohort_retention"].copy()
    o = data["orders"]
    o = o[o["order_status"] == "Completed"].copy()
    d = pd.to_datetime(o["order_date"])
    o["idx"] = d.dt.year * 12 + d.dt.month
    act = o.drop_duplicates(["customer_id", "idx"])
    first = act.groupby("customer_id")["idx"].min().rename("cohort_idx")
    act = act.join(first, on="customer_id")
    act["since"] = act["idx"] - act["cohort_idx"]
    expected = act.groupby(["cohort_idx", "since"])["customer_id"].nunique().sort_index()

    m = pd.to_datetime(r["cohort_month"])
    r["cohort_idx"] = m.dt.year * 12 + m.dt.month
    got = r.set_index(["cohort_idx", "months_since"])["active_customers"].sort_index()
    assert list(got.index) == list(expected.index)
    assert (got.to_numpy() == expected.to_numpy()).all()
    assert (r.loc[r["months_since"] == 0, "retention_pct"] == 100.0).all()
    assert r["retention_pct"].between(0, 100).all()
    assert r.drop_duplicates("cohort_month")["cohort_size"].sum() == o["customer_id"].nunique()


def test_retention_curve_from_the_query_is_plausible(results):
    curve = retention_curve(results["cohort_retention"])
    assert curve.loc[0, "retention_pct"] == 100.0
    assert curve["retention_pct"].between(0, 100).all()
    assert curve.loc[1, "retention_pct"] < 100.0
    assert curve["cohorts_observed"].is_monotonic_decreasing


def test_ltv_matches_pandas(results, lines, data):
    r = results["customer_ltv"]
    per = lines.groupby("customer_id").agg(
        orders=("order_id", "nunique"), revenue=("revenue", "sum"), contribution=("contribution", "sum")
    )
    per = per.join(data["customers"].set_index("customer_id")[["segment", "acquisition_channel"]])
    assert set(r["dimension"]) == {"segment", "channel"}
    for dim, col in (("segment", "segment"), ("channel", "acquisition_channel")):
        sub = r[r["dimension"] == dim].set_index("value")
        assert set(sub.index) == set(per[col].unique())
        assert sub["customers"].sum() == len(per)
        for value, row in sub.iterrows():
            grp = per[per[col] == value]
            assert row["customers"] == len(grp)
            assert abs(row["avg_orders"] - grp["orders"].mean()) < 0.006
            assert abs(row["avg_revenue"] - grp["revenue"].mean()) < 0.02
            assert abs(row["avg_contribution_ltv"] - grp["contribution"].mean()) < 0.02
            assert abs(row["median_contribution_ltv"] - grp["contribution"].median()) < 0.02


def test_rfm_matches_pandas(results, data):
    g = _rfm_frame(data)
    r = results["rfm_segments"].set_index("rfm_segment")
    assert set(r.index) <= {"Champions", "At risk", "New / promising", "Dormant"}
    for label, grp in g.groupby("segment"):
        assert r.loc[label, "customers"] == len(grp), label
        assert abs(r.loc[label, "revenue"] - grp["monetary"].sum()) < 0.5, label
        assert abs(r.loc[label, "avg_recency_days"] - grp["recency"].mean()) < 0.06, label
    assert set(g["segment"]) == set(r.index)


def test_rfm_segments_cover_every_customer_once(results, data, lines):
    r = results["rfm_segments"]
    assert r["customers"].sum() == lines["customer_id"].nunique()
    assert abs(r["revenue"].sum() - lines["revenue"].sum()) < 0.5
    assert abs(r["revenue_share_pct"].sum() - 100) < 0.5
    assert r["revenue"].is_monotonic_decreasing


def test_supplier_performance_matches_pandas(results, data):
    po = data["purchase_orders"]
    d = po[po["delivery_status"] == "Delivered"].copy()
    d["delay"] = (pd.to_datetime(d["actual_delivery_date"]) - pd.to_datetime(d["expected_delivery_date"])).dt.days
    r = results["supplier_performance"].set_index("supplier_id")
    assert len(r) == d["supplier_id"].nunique()
    for sid, grp in d.groupby("supplier_id"):
        assert r.loc[sid, "deliveries"] == len(grp)
        assert abs(r.loc[sid, "on_time_pct"] - 100 * (grp["delay"] <= 0).mean()) < 0.06
        defect = 100 * grp["defective_units"].sum() / grp["quantity_received"].sum()
        assert abs(r.loc[sid, "defect_pct"] - defect) < 0.01
        short = 100 * (grp["quantity_received"] < grp["quantity_ordered"]).mean()
        assert abs(r.loc[sid, "short_shipment_pct"] - short) < 0.06
    ordered = results["supplier_performance"]
    assert ordered["on_time_pct"].is_monotonic_decreasing
    assert ordered["on_time_rank"].iloc[0] == 1


def test_inventory_turnover_matches_pandas(results, data):
    inv = data["inventory_daily"].merge(data["products"][["product_id", "category", "unit_cost"]], on="product_id")
    inv["year"] = pd.to_datetime(inv["date"]).dt.year
    inv["cogs"] = inv["units_fulfilled"] * inv["unit_cost"]
    daily = inv.groupby(["year", "category", "date"]).agg(v=("inventory_value", "sum"), cogs=("cogs", "sum")).reset_index()
    g = daily.groupby(["year", "category"]).agg(cogs=("cogs", "sum"), avg_v=("v", "mean"))
    g["turnover"] = g["cogs"] / g["avg_v"]
    r = results["inventory_turnover"].set_index(["year", "category"])
    assert len(r) == 18
    assert set(r.index) == set(g.index)
    for key, row in r.iterrows():
        assert abs(row["turnover"] - g.loc[key, "turnover"]) < 0.01
        assert abs(row["days_of_inventory"] - 365 / g.loc[key, "turnover"]) < 0.2


def test_revenue_leakage_matches_pandas(results, data):
    value = data["order_items"].groupby("order_id")["line_total"].sum().rename("total")
    x = data["orders"].merge(value, on="order_id")
    x = x.merge(data["payments"][["order_id", "payment_status"]], on="order_id", how="left")
    x = x.merge(data["refunds"][["order_id", "refund_amount"]], on="order_id", how="left")
    x["refund_amount"] = x["refund_amount"].fillna(0.0)
    x["year"] = pd.to_datetime(x["order_date"]).dt.year
    r = results["revenue_leakage"].set_index("year")
    assert list(r.index) == [2023, 2024, 2025]
    for year, g in x.groupby("year"):
        cancelled = g.loc[g["order_status"] == "Cancelled", "total"].sum()
        failed = g.loc[g["payment_status"] == "Failed", "total"].sum()
        refunded = g["refund_amount"].sum()
        assert r.loc[year, "orders_placed"] == len(g)
        assert r.loc[year, "completed"] == (g["order_status"] == "Completed").sum()
        assert r.loc[year, "paid"] == (g["payment_status"] == "Captured").sum()
        assert r.loc[year, "paid_not_refunded"] == ((g["payment_status"] == "Captured") & (g["refund_amount"] == 0)).sum()
        assert abs(r.loc[year, "placed_value"] - g["total"].sum()) < 0.02
        assert abs(r.loc[year, "cancelled_value"] - cancelled) < 0.02
        assert abs(r.loc[year, "failed_value"] - failed) < 0.02
        assert abs(r.loc[year, "refunded_value"] - refunded) < 0.02
        assert abs(r.loc[year, "leakage_pct"] - round(100 * (cancelled + failed + refunded) / g["total"].sum(), 1)) < 0.06
    funnel = r[["orders_placed", "completed", "paid", "paid_not_refunded"]]
    assert (funnel["orders_placed"] >= funnel["completed"]).all()
    assert (funnel["completed"] >= funnel["paid"]).all()
    assert (funnel["paid"] >= funnel["paid_not_refunded"]).all()