import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from nexus.db import (
    apply_schema,
    get_db_settings,
    get_engine,
    list_core_tables,
    schema_path,
    split_sql,
)

EXPECTED_TABLES = [
    "customers",
    "daily_metrics",
    "inventory_daily",
    "order_items",
    "orders",
    "payments",
    "products",
    "purchase_orders",
    "refunds",
    "shipments",
    "suppliers",
]
EXPECTED_INDEXES = [
    "ix_products_supplier_id",
    "ix_customers_signup_date",
    "ix_orders_customer_id",
    "ix_orders_order_date",
    "ix_order_items_product_id",
    "ix_payments_payment_date",
    "ix_refunds_order_id",
    "ix_refunds_refund_date",
    "ix_purchase_orders_supplier_id",
    "ix_purchase_orders_product_id",
    "ix_purchase_orders_order_date",
    "ix_inventory_daily_date",
    "ix_inventory_daily_stockouts",
    "ix_shipments_carrier",
]

BAD_SUPPLIER = (
    "INSERT INTO core.suppliers (supplier_id, supplier_name, country, region, lead_time_mean_days, "
    "on_time_delivery_rate, defect_rate, payment_terms_days, onboarded_date) "
    "VALUES ('SUP-TEST', 'Test Co', 'India', 'Asia', 10, 0.9, 2, 30, '2020-01-01')"
)
GOOD_SUPPLIER = BAD_SUPPLIER.replace("0.9, 2, 30", "0.9, 0.02, 30")
PRODUCT_TEMPLATE = (
    "INSERT INTO core.products (product_id, sku, product_name, category, supplier_id, unit_cost, "
    "list_price, launch_date, is_active) "
    "VALUES ('PRD-TEST', 'SKU-TEST', 'Test Product', 'Electronics', '{supplier}', 10.00, 15.00, "
    "'2022-01-01', true)"
)
BAD_SHIPMENT = (
    "INSERT INTO core.shipments (shipment_id, order_id, carrier, ship_date, promised_delivery_date, "
    "delivered_date, shipment_status, had_backorder, handling_days, transit_days, delivery_delay_days, "
    "shipping_cost) "
    "VALUES ('SHP-TEST', 'ORD-NONE', 'SwiftShip', NULL, '2025-01-05', '2025-01-04', 'Delivered', 0, "
    "NULL, NULL, NULL, 5.00)"
)


@pytest.fixture(scope="module")
def engine():
    try:
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL is not reachable: {type(exc).__name__}")
    apply_schema(eng)
    yield eng
    eng.dispose()


def _expect_violation(engine, sql, hint):
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            with pytest.raises(IntegrityError) as excinfo:
                conn.execute(text(sql))
            assert hint in str(excinfo.value)
        finally:
            trans.rollback()


def test_split_sql_strips_comments_and_blanks():
    sql = "-- a comment\nCREATE TABLE a (x int);\n\n-- another\nCREATE TABLE b (y int);\n"
    assert split_sql(sql) == ["CREATE TABLE a (x int)", "CREATE TABLE b (y int)"]


def test_schema_file_is_splittable():
    raw = schema_path().read_text(encoding="utf-8")
    assert "%" not in raw
    statements = split_sql(raw)
    assert len(statements) == 26  # 1 schema, 11 tables, 14 indexes
    assert statements[0] == "CREATE SCHEMA IF NOT EXISTS core"


def test_db_settings_from_environment(monkeypatch):
    monkeypatch.setenv("NEXUS_DB_HOST", "dbhost")
    monkeypatch.setenv("NEXUS_DB_PORT", "5544")
    monkeypatch.setenv("NEXUS_DB_NAME", "somedb")
    monkeypatch.setenv("NEXUS_DB_USER", "someone")
    monkeypatch.setenv("NEXUS_DB_PASSWORD", "pw123")
    s = get_db_settings()
    assert (s.host, s.port, s.name, s.user, s.password) == ("dbhost", 5544, "somedb", "someone", "pw123")


def test_missing_password_is_rejected(monkeypatch):
    monkeypatch.delenv("NEXUS_DB_PASSWORD", raising=False)
    with pytest.raises(RuntimeError):
        get_db_settings()
    monkeypatch.setenv("NEXUS_DB_PASSWORD", "x")
    monkeypatch.setenv("NEXUS_DB_PORT", "abc")
    with pytest.raises(ValueError):
        get_db_settings()


def test_engine_url_hides_password(monkeypatch):
    monkeypatch.setenv("NEXUS_DB_PASSWORD", "s3cret!@:value")
    monkeypatch.setenv("NEXUS_DB_PORT", "5432")
    eng = get_engine()
    try:
        assert eng.url.password == "s3cret!@:value"
        shown = eng.url.render_as_string(hide_password=True)
        assert "s3cret" not in shown
        assert "***" in shown
    finally:
        eng.dispose()


def test_connects_as_the_project_role(engine):
    with engine.connect() as conn:
        row = conn.execute(text("SELECT current_user, current_database()")).one()
    assert row[0] == get_db_settings().user
    assert row[1] == get_db_settings().name


def test_core_tables_exist(engine):
    assert list_core_tables(engine) == EXPECTED_TABLES


def test_every_table_has_a_primary_key_and_foreign_keys_exist(engine):
    query = text(
        "SELECT constraint_type, count(*) FROM information_schema.table_constraints "
        "WHERE table_schema = 'core' AND constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY') "
        "GROUP BY constraint_type"
    )
    with engine.connect() as conn:
        counts = {row[0]: row[1] for row in conn.execute(query)}
    assert counts["PRIMARY KEY"] == 11
    assert counts["FOREIGN KEY"] == 11


def test_expected_indexes_exist(engine):
    with engine.connect() as conn:
        found = {r[0] for r in conn.execute(text("SELECT indexname FROM pg_indexes WHERE schemaname = 'core'"))}
    assert set(EXPECTED_INDEXES).issubset(found)


def test_apply_schema_is_idempotent(engine):
    first = apply_schema(engine)
    second = apply_schema(engine)
    assert first == second == 26
    assert list_core_tables(engine) == EXPECTED_TABLES


def test_check_constraint_rejects_a_bad_rate(engine):
    _expect_violation(engine, BAD_SUPPLIER, "suppliers_defect_rate_check")


def test_foreign_key_rejects_an_unknown_supplier(engine):
    _expect_violation(engine, PRODUCT_TEMPLATE.format(supplier="SUP-NONE"), "products_supplier_id_fkey")


def test_shipment_status_must_match_its_fields(engine):
    _expect_violation(engine, BAD_SHIPMENT, "shipments_status_fields")


def test_valid_rows_are_accepted_and_rolled_back(engine):
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(text(GOOD_SUPPLIER))
            conn.execute(text(PRODUCT_TEMPLATE.format(supplier="SUP-TEST")))
            n = conn.execute(text("SELECT count(*) FROM core.products WHERE product_id = 'PRD-TEST'")).scalar()
            assert n == 1
        finally:
            trans.rollback()
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM core.suppliers WHERE supplier_id = 'SUP-TEST'")).scalar() == 0