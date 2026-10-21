from pathlib import Path

import pytest
from sqlalchemy import text

import nexus.promotion as promotion
from nexus.db import get_engine
from nexus.loader import LOAD_ORDER
from nexus.promotion import analytics_ddl, build_analytics, check_analytics

UNCHANGED = ["suppliers", "products", "purchase_orders", "inventory_daily"]


@pytest.fixture(scope="module")
def engine():
    try:
        eng = get_engine()
        with eng.connect() as conn:
            ready = conn.execute(
                text(
                    "SELECT to_regclass('staging.orders') IS NOT NULL AND to_regclass('raw.suppliers') IS NOT NULL "
                    "AND to_regclass('core.orders') IS NOT NULL AND to_regclass('staging.row_issues') IS NOT NULL"
                )
            ).scalar()
    except Exception as exc:
        pytest.skip(f"PostgreSQL is not reachable: {type(exc).__name__}")
    if not ready:
        pytest.skip("run load_core.py, load_raw.py and build_staging.py first")
    build_analytics(eng)
    yield eng
    eng.dispose()


def _scalar(engine, sql):
    with engine.connect() as conn:
        return conn.execute(text(sql)).scalar()


def _except_count(engine, table, left, right):
    return _scalar(engine, f"SELECT count(*) FROM (SELECT * FROM {left}.{table} EXCEPT SELECT * FROM {right}.{table}) x")


def test_ddl_is_a_rewrite_of_the_core_schema():
    statements = analytics_ddl()
    assert len(statements) == 26
    assert statements[0] == "CREATE SCHEMA IF NOT EXISTS analytics"
    for sql in statements:
        assert "core." not in sql
    assert any("analytics.order_items" in s for s in statements)


def test_promotion_does_not_read_ground_truth_or_the_manifest():
    source = Path(promotion.__file__).read_text(encoding="utf-8")
    assert "defect_manifest" not in source
    assert "FROM core" not in source
    assert "JOIN core" not in source


def test_tables_and_constraints_exist(engine):
    with engine.connect() as conn:
        names = {r[0] for r in conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'analytics'"))}
        counts = {r[0]: r[1] for r in conn.execute(text(
            "SELECT constraint_type, count(*) FROM information_schema.table_constraints "
            "WHERE table_schema = 'analytics' AND constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY') GROUP BY 1"))}
    assert names == {*LOAD_ORDER, "promotion_log"}
    assert counts["PRIMARY KEY"] == 11
    assert counts["FOREIGN KEY"] == 11


def test_counts_follow_staging_and_raw(engine):
    for t in ["customers", "orders", "order_items", "payments"]:
        assert _scalar(engine, f"SELECT count(*) FROM analytics.{t}") == _scalar(engine, f"SELECT count(*) FROM staging.{t}")
    for t in UNCHANGED + ["daily_metrics"]:
        expected = 1096 if t == "daily_metrics" else _scalar(engine, f"SELECT count(*) FROM raw.{t}")
        assert _scalar(engine, f"SELECT count(*) FROM analytics.{t}") == expected


def test_check_analytics_reports_nothing(engine):
    assert check_analytics(engine) == []


def test_cleaned_tables_equal_ground_truth(engine):
    for t in ["orders", "order_items", "payments"]:
        assert _except_count(engine, t, "analytics", "core") == 0, t
    imputed = _scalar(
        engine,
        "SELECT count(*) FROM staging.row_issues WHERE table_name = 'customers' "
        "AND rule = 'signup_date_imputed_from_neighbours'",
    )
    assert _except_count(engine, "customers", "analytics", "core") <= imputed


def test_unchanged_tables_equal_ground_truth(engine):
    for t in UNCHANGED:
        assert _except_count(engine, t, "analytics", "core") == 0, t
        assert _except_count(engine, t, "core", "analytics") == 0, t
    for t in ["refunds", "shipments"]:
        assert _except_count(engine, t, "analytics", "core") == 0, t


def test_quarantine_log_explains_every_missing_row(engine):
    for t in ["refunds", "shipments"]:
        missing = _scalar(engine, f"SELECT count(*) FROM core.{t}") - _scalar(engine, f"SELECT count(*) FROM analytics.{t}")
        logged = _scalar(engine, f"SELECT count(*) FROM analytics.promotion_log WHERE table_name = '{t}'")
        assert missing == logged > 0, t
    quarantined_payments = _scalar(
        engine,
        "SELECT count(*) FROM staging.row_issues WHERE table_name = 'payments' AND rule = 'order_not_in_staging'",
    )
    assert _scalar(engine, "SELECT count(*) FROM analytics.promotion_log WHERE table_name = 'shipments'") == quarantined_payments


def test_one_payment_and_one_shipment_per_completed_order(engine):
    completed = _scalar(engine, "SELECT count(*) FROM analytics.orders WHERE order_status = 'Completed'")
    assert _scalar(engine, "SELECT count(*) FROM analytics.payments") == completed
    assert _scalar(engine, "SELECT count(*) FROM analytics.shipments") == completed


def test_daily_metrics_reconcile_with_the_tables(engine):
    revenue = float(_scalar(
        engine,
        "SELECT sum(i.line_total) FROM analytics.order_items i JOIN analytics.orders o USING (order_id) "
        "WHERE o.order_status = 'Completed'",
    ))
    assert abs(float(_scalar(engine, "SELECT sum(gross_revenue) FROM analytics.daily_metrics")) - revenue) < 0.05
    assert abs(float(_scalar(engine, "SELECT sum(amount) FROM analytics.payments")) - revenue) < 0.05


def test_daily_metrics_differ_from_ground_truth_only_by_quarantined_revenue(engine):
    truth = float(_scalar(engine, "SELECT sum(gross_revenue) FROM core.daily_metrics"))
    cleaned = float(_scalar(engine, "SELECT sum(gross_revenue) FROM analytics.daily_metrics"))
    held_out = float(_scalar(
        engine,
        "SELECT sum(i.line_total) FROM core.order_items i JOIN core.orders o USING (order_id) "
        "WHERE o.order_status = 'Completed' AND o.order_id IN "
        "(SELECT row_key FROM staging.row_issues WHERE table_name = 'orders' AND action = 'quarantined')",
    ))
    assert held_out > 0
    assert abs((truth - cleaned) - held_out) < 0.05


def test_rebuild_is_repeatable(engine):
    before = {t: _scalar(engine, f"SELECT count(*) FROM analytics.{t}") for t in LOAD_ORDER}
    assert build_analytics(engine) == before
    assert check_analytics(engine) == []