import pandas as pd
import pytest
from sqlalchemy import text

from nexus.config import get_settings
from nexus.db import apply_schema, get_engine
from nexus.loader import LOAD_ORDER, ORPHAN_CHECKS, csv_row_count, load_core, reconcile

RAW = get_settings().raw_dir


@pytest.fixture(scope="module")
def engine():
    missing = [t for t in LOAD_ORDER if not (RAW / f"{t}.csv").exists()]
    if missing:
        pytest.skip(f"clean CSV files are missing: {missing}")
    try:
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL is not reachable: {type(exc).__name__}")
    apply_schema(eng)
    load_core(eng, RAW)
    yield eng
    eng.dispose()


def _scalar(engine, sql):
    with engine.connect() as conn:
        return conn.execute(text(sql)).scalar()


def test_load_order_covers_every_table_once():
    assert len(LOAD_ORDER) == 11
    assert len(set(LOAD_ORDER)) == 11
    assert LOAD_ORDER.index("suppliers") < LOAD_ORDER.index("products")
    assert LOAD_ORDER.index("orders") < LOAD_ORDER.index("order_items") < LOAD_ORDER.index("payments")
    assert LOAD_ORDER.index("payments") < LOAD_ORDER.index("refunds")


def test_csv_row_count_ignores_the_header():
    assert csv_row_count(RAW / "suppliers.csv") == 24 if (RAW / "suppliers.csv").exists() else True


def test_row_counts_match_the_csv_files(engine):
    for table in LOAD_ORDER:
        assert _scalar(engine, f"SELECT count(*) FROM core.{table}") == csv_row_count(RAW / f"{table}.csv")


def test_reconcile_reports_nothing(engine):
    assert reconcile(engine, RAW) == []


def test_no_orphans(engine):
    for name, sql in ORPHAN_CHECKS:
        assert _scalar(engine, sql) == 0, name


def test_database_totals_match_daily_metrics(engine):
    dm = pd.read_csv(RAW / "daily_metrics.csv")
    revenue = float(
        _scalar(
            engine,
            "SELECT sum(i.line_total) FROM core.order_items i JOIN core.orders o USING (order_id) "
            "WHERE o.order_status = 'Completed'",
        )
    )
    assert abs(revenue - dm["gross_revenue"].sum()) < 0.05
    refunds = float(_scalar(engine, "SELECT sum(refund_amount) FROM core.refunds"))
    assert abs(refunds - dm["refunds_amount"].sum()) < 0.05


def test_nulls_only_where_allowed(engine):
    assert _scalar(engine, "SELECT count(*) FROM core.customers WHERE email IS NULL") == 0
    assert _scalar(engine, "SELECT count(*) FROM core.purchase_orders WHERE delivery_status = 'In Transit'") > 0
    assert _scalar(
        engine,
        "SELECT count(*) FROM core.purchase_orders WHERE delivery_status = 'In Transit' "
        "AND (actual_delivery_date IS NOT NULL OR quantity_received IS NOT NULL)",
    ) == 0


def test_booleans_and_types_loaded_correctly(engine):
    assert _scalar(engine, "SELECT count(*) FROM core.products WHERE is_active") > 0
    assert _scalar(engine, "SELECT count(*) FROM core.products WHERE NOT is_active") > 0
    assert _scalar(engine, "SELECT min(order_date)::text FROM core.orders") >= "2023-01-01"
    assert _scalar(engine, "SELECT pg_typeof(line_total)::text FROM core.order_items LIMIT 1") == "numeric"


def test_reloading_gives_identical_counts(engine):
    before = {t: _scalar(engine, f"SELECT count(*) FROM core.{t}") for t in LOAD_ORDER}
    load_core(engine, RAW)
    after = {t: _scalar(engine, f"SELECT count(*) FROM core.{t}") for t in LOAD_ORDER}
    assert before == after
    assert reconcile(engine, RAW) == []


def test_a_failed_load_rolls_back(engine, tmp_path):
    for table in LOAD_ORDER:
        (tmp_path / f"{table}.csv").write_bytes((RAW / f"{table}.csv").read_bytes())
    products = pd.read_csv(tmp_path / "products.csv")
    products.loc[0, "list_price"] = products.loc[0, "unit_cost"] - 1
    products.to_csv(tmp_path / "products.csv", index=False)
    before = _scalar(engine, "SELECT count(*) FROM core.orders")
    assert before > 0
    with pytest.raises(Exception):
        load_core(engine, tmp_path)
    assert _scalar(engine, "SELECT count(*) FROM core.orders") == before