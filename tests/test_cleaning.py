from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

import nexus.cleaning as cleaning
from nexus.cleaning import (
    CANONICAL_COUNTRIES,
    PAYMENT_METHODS,
    TABLES,
    build_staging,
    check_staging,
    country_map,
    infer_slash_date_format,
    method_map,
    staging_statements,
)
from nexus.config import get_settings
from nexus.db import apply_schema, get_engine
from nexus.loader import LOAD_ORDER, load_core
from nexus.raw_loader import load_raw

RAW = get_settings().raw_dir
DIRTY = RAW / "dirty"


@pytest.fixture(scope="module")
def engine():
    missing = [t for t in LOAD_ORDER if not (RAW / f"{t}.csv").exists() or not (DIRTY / f"{t}.csv").exists()]
    if missing:
        pytest.skip(f"CSV files are missing: {missing}")
    try:
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL is not reachable: {type(exc).__name__}")
    apply_schema(eng)
    load_core(eng, RAW)
    load_raw(eng, DIRTY)
    build_staging(eng)
    yield eng
    eng.dispose()


def _scalar(engine, sql, **params):
    with engine.connect() as conn:
        return conn.execute(text(sql), params).scalar()


def _logged(engine, table, rule, action):
    return _scalar(
        engine,
        "SELECT count(*) FROM staging.row_issues WHERE table_name = :t AND rule = :r AND action = :a",
        t=table, r=rule, a=action,
    )


def test_country_and_method_maps_are_complete():
    countries = country_map()
    for canonical in CANONICAL_COUNTRIES:
        assert countries[canonical.lower()] == canonical
    assert countries["usa"] == "United States"
    assert countries["uk"] == "United Kingdom"
    assert countries["uae"] == "United Arab Emirates"
    assert set(method_map().values()) == set(PAYMENT_METHODS)
    assert method_map()["bank transfer"] == "Bank Transfer"


def test_statements_are_safe_and_format_aware():
    dmy = staging_statements("dmy")
    mdy = staging_statements("mdy")
    assert len(dmy) == len(mdy) > 30
    for sql in dmy + mdy:
        assert "%" not in sql
        assert "@" not in sql
    assert "b.p2 BETWEEN 1 AND 12" in " ".join(dmy)
    assert "b.p1 BETWEEN 1 AND 12" in " ".join(mdy)
    assert dmy != mdy


def test_pipeline_does_not_read_the_manifest_or_ground_truth():
    source = Path(cleaning.__file__).read_text(encoding="utf-8")
    assert "defect_manifest" not in source
    assert "core." not in source
    assert "datagen" not in source


def test_slash_format_is_inferred_from_the_data(engine):
    with engine.connect() as conn:
        assert infer_slash_date_format(conn) == "dmy"


def test_every_raw_row_is_accounted_for(engine):
    for table in TABLES:
        raw_n = _scalar(engine, f"SELECT count(*) FROM raw.{table}")
        staged = _scalar(engine, f"SELECT count(*) FROM staging.{table}")
        removed = _scalar(
            engine,
            "SELECT count(DISTINCT source_row_id) FROM staging.row_issues "
            "WHERE table_name = :t AND action IN ('dropped', 'quarantined')",
            t=table,
        )
        assert raw_n == staged + removed, table


def test_staging_types_and_keys(engine):
    expected = [
        ("customers", "signup_date", "date"),
        ("customers", "marketing_opt_in", "boolean"),
        ("orders", "order_date", "date"),
        ("orders", "discount_pct", "numeric"),
        ("order_items", "quantity", "integer"),
        ("order_items", "line_total", "numeric"),
        ("payments", "payment_date", "date"),
        ("payments", "amount", "numeric"),
    ]
    for table, column, data_type in expected:
        found = _scalar(
            engine,
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_schema = 'staging' AND table_name = :t AND column_name = :c",
            t=table, c=column,
        )
        assert found == data_type, f"{table}.{column}"
    for table in TABLES:
        assert _scalar(engine, f"SELECT count(*) - count(DISTINCT source_row_id) FROM staging.{table}") == 0


def test_staging_has_no_orphans_or_gaps(engine):
    assert check_staging(engine) == []


def test_values_are_standardised(engine):
    with engine.connect() as conn:
        countries = {r[0] for r in conn.execute(text("SELECT DISTINCT country FROM staging.customers"))}
        statuses = {r[0] for r in conn.execute(text("SELECT DISTINCT order_status FROM staging.orders"))}
        methods = {r[0] for r in conn.execute(text("SELECT DISTINCT payment_method FROM staging.payments"))}
    assert countries <= set(CANONICAL_COUNTRIES)
    assert statuses == {"Completed", "Cancelled"}
    assert methods <= set(PAYMENT_METHODS)
    assert _scalar(engine, "SELECT max(discount_pct) FROM staging.orders") <= 0.20
    assert _scalar(engine, "SELECT min(quantity) FROM staging.order_items") >= 1
    assert _scalar(engine, "SELECT min(unit_price) FROM staging.order_items") > 0


def test_orders_and_customers_match_ground_truth(engine):
    assert _scalar(
        engine,
        "SELECT count(*) FROM staging.orders s LEFT JOIN core.orders c USING (order_id) WHERE c.order_id IS NULL",
    ) == 0
    assert _scalar(
        engine,
        "SELECT count(*) FROM staging.orders s JOIN core.orders c USING (order_id) "
        "WHERE s.customer_id <> c.customer_id OR s.order_date <> c.order_date "
        "OR s.order_status <> c.order_status OR s.discount_pct <> c.discount_pct "
        "OR s.sales_channel <> c.sales_channel",
    ) == 0
    assert _scalar(
        engine,
        "SELECT count(*) FROM staging.customers s JOIN core.customers c USING (customer_id) "
        "WHERE s.country <> c.country OR s.region <> c.region OR s.segment <> c.segment "
        "OR (s.email IS NOT NULL AND s.email <> c.email)",
    ) == 0
    different = _scalar(
        engine,
        "SELECT count(*) FROM staging.customers s JOIN core.customers c USING (customer_id) "
        "WHERE s.signup_date <> c.signup_date",
    )
    imputed = _logged(engine, "customers", "signup_date_imputed_from_neighbours", "fixed")
    assert different <= imputed
    assert _scalar(
        engine,
        "SELECT coalesce(max(abs(s.signup_date - c.signup_date)), 0) FROM staging.customers s "
        "JOIN core.customers c USING (customer_id)",
    ) <= 30


def test_items_and_payments_match_ground_truth(engine):
    assert _scalar(
        engine,
        "SELECT count(*) FROM staging.order_items s JOIN core.order_items c USING (order_id, line_number) "
        "WHERE s.product_id <> c.product_id OR s.quantity <> c.quantity OR s.unit_price <> c.unit_price "
        "OR s.unit_cost <> c.unit_cost OR s.discount_pct <> c.discount_pct OR s.line_total <> c.line_total",
    ) == 0
    assert _scalar(
        engine,
        "SELECT count(*) FROM staging.payments s JOIN core.payments c USING (payment_id) "
        "WHERE s.order_id <> c.order_id OR s.payment_date <> c.payment_date "
        "OR s.payment_method <> c.payment_method OR s.payment_status <> c.payment_status OR s.amount <> c.amount",
    ) == 0


def test_issue_counts_match_the_injected_defects(engine):
    path = DIRTY / "defect_manifest.csv"
    if not path.exists():
        pytest.skip("defect manifest is missing")
    manifest = pd.read_csv(path, keep_default_na=False)

    def injected(table, defect):
        return manifest[(manifest["table"] == table) & (manifest["defect_type"] == defect)]

    for table in TABLES:
        assert _logged(engine, table, "duplicate_key", "dropped") == len(injected(table, "exact_duplicate_row"))

    with engine.connect() as conn:
        quarantined_orders = {
            r[0] for r in conn.execute(
                text("SELECT row_key FROM staging.row_issues WHERE table_name = 'orders' AND action = 'quarantined'")
            )
        }
        pay_order = dict(conn.execute(text("SELECT payment_id, order_id FROM core.payments")).all())
    unlinkable = set(injected("orders", "missing_customer_id")["row_key"]) | set(
        injected("orders", "unknown_customer_id")["row_key"]
    )
    assert quarantined_orders == unlinkable

    assert _logged(engine, "customers", "country_standardised", "fixed") == len(injected("customers", "country_variant"))
    assert _logged(engine, "customers", "missing_email", "flagged") == len(injected("customers", "missing_email"))
    assert _logged(engine, "customers", "signup_date_imputed_from_neighbours", "fixed") == len(
        injected("customers", "future_signup_date")
    )
    assert _logged(engine, "orders", "date_format_standardised", "fixed") == len(injected("orders", "date_format_dmy"))
    assert _logged(engine, "orders", "status_case_standardised", "fixed") == len(injected("orders", "status_case_variant"))
    assert _logged(engine, "orders", "discount_recovered_from_items", "fixed") == len(
        injected("orders", "invalid_discount_pct")
    )

    def kept_items(defect):
        keys = injected("order_items", defect)["row_key"]
        return sum(1 for k in keys if k.split("|")[0] not in quarantined_orders)

    assert _logged(engine, "order_items", "quantity_sign_corrected", "fixed") == kept_items("negative_quantity")
    assert _logged(engine, "order_items", "unit_price_from_catalogue", "fixed") == kept_items("zero_unit_price")
    assert _logged(engine, "order_items", "line_total_recomputed", "fixed") == kept_items("missing_line_total")

    def kept_payments(defect):
        keys = injected("payments", defect)["row_key"]
        return sum(1 for k in keys if pay_order[k] not in quarantined_orders)

    assert _logged(engine, "payments", "payment_method_case_standardised", "fixed") == kept_payments("method_case_variant")
    assert _logged(engine, "payments", "amount_recomputed_from_items", "fixed") == kept_payments("missing_amount")


def test_build_is_repeatable(engine):
    before = {t: _scalar(engine, f"SELECT count(*) FROM staging.{t}") for t in TABLES}
    issues = _scalar(engine, "SELECT count(*) FROM staging.row_issues")
    counts, fmt = build_staging(engine)
    assert fmt == "dmy"
    assert counts == before
    assert _scalar(engine, "SELECT count(*) FROM staging.row_issues") == issues
    assert check_staging(engine) == []