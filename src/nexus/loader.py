"""Load the clean CSV files into the core schema and reconcile them (SYNTHETIC data)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from nexus.db import CORE_SCHEMA

# Foreign-key order: parents before children.
LOAD_ORDER = [
    "suppliers",
    "products",
    "customers",
    "orders",
    "order_items",
    "payments",
    "refunds",
    "purchase_orders",
    "inventory_daily",
    "shipments",
    "daily_metrics",
]

# Orphan checks: (name, SQL that must return 0).
ORPHAN_CHECKS = [
    ("products without a supplier",
     "SELECT count(*) FROM core.products p LEFT JOIN core.suppliers s USING (supplier_id) WHERE s.supplier_id IS NULL"),
    ("orders without a customer",
     "SELECT count(*) FROM core.orders o LEFT JOIN core.customers c USING (customer_id) WHERE c.customer_id IS NULL"),
    ("order lines without an order",
     "SELECT count(*) FROM core.order_items i LEFT JOIN core.orders o USING (order_id) WHERE o.order_id IS NULL"),
    ("order lines without a product",
     "SELECT count(*) FROM core.order_items i LEFT JOIN core.products p USING (product_id) WHERE p.product_id IS NULL"),
    ("payments without an order",
     "SELECT count(*) FROM core.payments p LEFT JOIN core.orders o USING (order_id) WHERE o.order_id IS NULL"),
    ("refunds without a payment",
     "SELECT count(*) FROM core.refunds r LEFT JOIN core.payments p USING (payment_id) WHERE p.payment_id IS NULL"),
    ("shipments without an order",
     "SELECT count(*) FROM core.shipments s LEFT JOIN core.orders o USING (order_id) WHERE o.order_id IS NULL"),
    ("completed orders without a payment",
     "SELECT count(*) FROM core.orders o LEFT JOIN core.payments p USING (order_id) WHERE o.order_status = 'Completed' AND p.payment_id IS NULL"),
    ("completed orders without a shipment",
     "SELECT count(*) FROM core.orders o LEFT JOIN core.shipments s USING (order_id) WHERE o.order_status = 'Completed' AND s.shipment_id IS NULL"),
]


def csv_row_count(path: Path) -> int:
    """Number of data rows in a CSV file (the file may not end with a newline)."""
    return len(pd.read_csv(path, usecols=[0]))


def _copy_file(raw_conn, table: str, path: Path) -> None:
    columns = list(pd.read_csv(path, nrows=0).columns)
    column_sql = ", ".join(columns)
    statement = f"COPY {CORE_SCHEMA}.{table} ({column_sql}) FROM STDIN WITH (FORMAT csv, HEADER true)"
    with raw_conn.cursor() as cur:
        with cur.copy(statement) as copy:
            with open(path, "rb") as handle:
                while chunk := handle.read(1 << 20):
                    copy.write(chunk)


def load_core(engine: Engine, raw_dir: Path) -> dict[str, int]:
    """Truncate and load all tables in one transaction. Returns rows loaded per table.

    Any error rolls the whole load back, leaving the previous contents untouched.
    """
    missing = [t for t in LOAD_ORDER if not (raw_dir / f"{t}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Missing CSV files in {raw_dir}: {', '.join(missing)}")

    loaded: dict[str, int] = {}
    with engine.begin() as conn:
        tables_sql = ", ".join(f"{CORE_SCHEMA}.{t}" for t in LOAD_ORDER)
        conn.exec_driver_sql(f"TRUNCATE {tables_sql} CASCADE")
        raw_conn = conn.connection.driver_connection
        for table in LOAD_ORDER:
            _copy_file(raw_conn, table, raw_dir / f"{table}.csv")
            loaded[table] = int(conn.execute(text(f"SELECT count(*) FROM {CORE_SCHEMA}.{table}")).scalar())
    return loaded


def reconcile(engine: Engine, raw_dir: Path) -> list[str]:
    """Compare the database with the CSV files. Returns problems (empty means reconciled)."""
    problems: list[str] = []
    with engine.connect() as conn:
        for table in LOAD_ORDER:
            expected = csv_row_count(raw_dir / f"{table}.csv")
            actual = int(conn.execute(text(f"SELECT count(*) FROM {CORE_SCHEMA}.{table}")).scalar())
            if actual != expected:
                problems.append(f"{table}: {actual} rows in the database, {expected} in the CSV")

        for name, sql in ORPHAN_CHECKS:
            n = int(conn.execute(text(sql)).scalar())
            if n:
                problems.append(f"{n} {name}")

        dm = pd.read_csv(raw_dir / "daily_metrics.csv")
        totals = [
            ("gross revenue", "SELECT coalesce(sum(i.line_total), 0) FROM core.order_items i "
                              "JOIN core.orders o USING (order_id) WHERE o.order_status = 'Completed'",
             dm["gross_revenue"].sum()),
            ("cost of goods sold", "SELECT coalesce(sum(i.quantity * i.unit_cost), 0) FROM core.order_items i "
                                   "JOIN core.orders o USING (order_id) WHERE o.order_status = 'Completed'",
             dm["cogs"].sum()),
            ("captured payments", "SELECT coalesce(sum(amount), 0) FROM core.payments WHERE payment_status = 'Captured'",
             dm["payments_captured"].sum()),
            ("failed payments", "SELECT coalesce(sum(amount), 0) FROM core.payments WHERE payment_status = 'Failed'",
             dm["payments_failed"].sum()),
            ("refunds", "SELECT coalesce(sum(refund_amount), 0) FROM core.refunds", dm["refunds_amount"].sum()),
            ("shipping cost", "SELECT coalesce(sum(shipping_cost), 0) FROM core.shipments", dm["shipping_cost"].sum()),
            ("purchase spend", "SELECT coalesce(sum(order_cost), 0) FROM core.purchase_orders", dm["po_spend"].sum()),
            ("inventory value", "SELECT coalesce(sum(inventory_value), 0) FROM core.inventory_daily",
             dm["inventory_value"].sum()),
        ]
        for name, sql, expected in totals:
            actual = float(conn.execute(text(sql)).scalar())
            if abs(actual - float(expected)) > 0.05:
                problems.append(f"{name}: database {actual:,.2f}, daily_metrics {float(expected):,.2f}")
    return problems