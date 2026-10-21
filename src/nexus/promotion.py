"""Promote the cleaned data into the analytics schema (SYNTHETIC data).

analytics has the same constrained DDL as core (copied from sql/schema.sql). It is filled from
the cleaned staging tables and from the unchanged raw tables (cast to proper types). Rows that
lose their parent to quarantine are quarantined too and logged in analytics.promotion_log.
The promotion reads only raw, staging and schema.sql, never the ground-truth core schema.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

from nexus.db import schema_path, split_sql
from nexus.loader import LOAD_ORDER

ANALYTICS_SCHEMA = "analytics"
CLEANED = ["customers", "orders", "order_items", "payments"]
FILTERED = {"refunds": "payment_not_in_analytics", "shipments": "order_not_in_analytics"}


def analytics_ddl() -> list[str]:
    """The core DDL statements rewritten for the analytics schema."""
    out = []
    for statement in split_sql(schema_path().read_text(encoding="utf-8")):
        statement = statement.replace(
            "CREATE SCHEMA IF NOT EXISTS core", f"CREATE SCHEMA IF NOT EXISTS {ANALYTICS_SCHEMA}"
        )
        out.append(statement.replace("core.", f"{ANALYTICS_SCHEMA}."))
    return out


PROMOTION_LOG_SQL = (
    "CREATE TABLE analytics.promotion_log ("
    "table_name text NOT NULL, row_key text NOT NULL, rule text NOT NULL, action text NOT NULL)"
)

_DAILY_METRICS_SQL = """
INSERT INTO analytics.daily_metrics (
    date, orders_placed, orders_completed, orders_cancelled, new_customers, active_customers,
    units_sold, gross_revenue, discount_amount, cogs, gross_margin, shipping_cost,
    payments_captured, payments_failed, refunds_amount, po_spend, listed_products,
    products_in_stockout, backorder_units, inventory_value)
WITH days AS (
    SELECT g::date AS dt FROM generate_series(DATE '2023-01-01', DATE '2025-12-31', INTERVAL '1 day') g
), od AS (
    SELECT order_date AS dt, count(*) AS placed,
           count(*) FILTER (WHERE order_status = 'Completed') AS completed,
           count(*) FILTER (WHERE order_status = 'Cancelled') AS cancelled,
           count(DISTINCT customer_id) FILTER (WHERE order_status = 'Completed') AS active
    FROM analytics.orders GROUP BY 1
), nc AS (
    SELECT signup_date AS dt, count(*) AS n FROM analytics.customers GROUP BY 1
), ln AS (
    SELECT o.order_date AS dt, sum(i.quantity) AS units, sum(i.line_total) AS revenue,
           sum(i.quantity * i.unit_price) AS list_value, sum(i.quantity * i.unit_cost) AS cogs
    FROM analytics.order_items i JOIN analytics.orders o USING (order_id)
    WHERE o.order_status = 'Completed' GROUP BY 1
), sh AS (
    SELECT o.order_date AS dt, sum(s.shipping_cost) AS cost
    FROM analytics.shipments s JOIN analytics.orders o USING (order_id) GROUP BY 1
), cp AS (
    SELECT payment_date AS dt, sum(amount) AS amt FROM analytics.payments
    WHERE payment_status = 'Captured' GROUP BY 1
), fl AS (
    SELECT payment_date AS dt, sum(amount) AS amt FROM analytics.payments
    WHERE payment_status = 'Failed' GROUP BY 1
), rf AS (
    SELECT refund_date AS dt, sum(refund_amount) AS amt FROM analytics.refunds GROUP BY 1
), pq AS (
    SELECT order_date AS dt, sum(order_cost) AS amt FROM analytics.purchase_orders GROUP BY 1
), iv AS (
    SELECT "date" AS dt, count(*) AS listed, sum(stockout_flag) AS so,
           sum(backorder_units) AS bo, sum(inventory_value) AS val
    FROM analytics.inventory_daily GROUP BY 1
)
SELECT d.dt,
    coalesce(od.placed, 0), coalesce(od.completed, 0), coalesce(od.cancelled, 0),
    coalesce(nc.n, 0), coalesce(od.active, 0), coalesce(ln.units, 0), coalesce(ln.revenue, 0),
    coalesce(ln.list_value, 0) - coalesce(ln.revenue, 0), coalesce(ln.cogs, 0),
    coalesce(ln.revenue, 0) - coalesce(ln.cogs, 0), coalesce(sh.cost, 0),
    coalesce(cp.amt, 0), coalesce(fl.amt, 0), coalesce(rf.amt, 0), coalesce(pq.amt, 0),
    coalesce(iv.listed, 0), coalesce(iv.so, 0), coalesce(iv.bo, 0), coalesce(iv.val, 0)
FROM days d
LEFT JOIN od ON od.dt = d.dt LEFT JOIN nc ON nc.dt = d.dt LEFT JOIN ln ON ln.dt = d.dt
LEFT JOIN sh ON sh.dt = d.dt LEFT JOIN cp ON cp.dt = d.dt LEFT JOIN fl ON fl.dt = d.dt
LEFT JOIN rf ON rf.dt = d.dt LEFT JOIN pq ON pq.dt = d.dt LEFT JOIN iv ON iv.dt = d.dt
ORDER BY d.dt
"""

_INSERTS = [
    """INSERT INTO analytics.suppliers
       SELECT supplier_id, supplier_name, country, region, lead_time_mean_days::integer,
              on_time_delivery_rate::numeric, defect_rate::numeric, payment_terms_days::integer,
              onboarded_date::date
       FROM raw.suppliers ORDER BY _row_id""",
    """INSERT INTO analytics.products
       SELECT product_id, sku, product_name, category, supplier_id, unit_cost::numeric,
              list_price::numeric, launch_date::date, is_active::boolean
       FROM raw.products ORDER BY _row_id""",
    """INSERT INTO analytics.customers (customer_id, email, country, region, segment,
           acquisition_channel, signup_date, marketing_opt_in)
       SELECT customer_id, email, country, region, segment, acquisition_channel, signup_date,
              marketing_opt_in
       FROM staging.customers ORDER BY customer_id""",
    """INSERT INTO analytics.orders (order_id, customer_id, order_date, sales_channel,
           order_status, discount_pct)
       SELECT order_id, customer_id, order_date, sales_channel, order_status, discount_pct
       FROM staging.orders ORDER BY order_id""",
    """INSERT INTO analytics.order_items (order_id, line_number, product_id, quantity,
           unit_price, unit_cost, discount_pct, line_total)
       SELECT order_id, line_number, product_id, quantity, unit_price, unit_cost, discount_pct,
              line_total
       FROM staging.order_items ORDER BY order_id, line_number""",
    """INSERT INTO analytics.payments (payment_id, order_id, payment_date, payment_method,
           payment_status, amount)
       SELECT payment_id, order_id, payment_date, payment_method, payment_status, amount
       FROM staging.payments ORDER BY payment_id""",
    """INSERT INTO analytics.promotion_log (table_name, row_key, rule, action)
       SELECT 'refunds', r.refund_id, 'payment_not_in_analytics', 'quarantined'
       FROM raw.refunds r
       WHERE NOT EXISTS (SELECT 1 FROM analytics.payments p WHERE p.payment_id = r.payment_id)""",
    """INSERT INTO analytics.refunds
       SELECT r.refund_id, r.payment_id, r.order_id, r.refund_date::date, r.refund_type,
              r.refund_reason, r.refund_amount::numeric
       FROM raw.refunds r
       WHERE EXISTS (SELECT 1 FROM analytics.payments p WHERE p.payment_id = r.payment_id)
       ORDER BY r._row_id""",
    """INSERT INTO analytics.purchase_orders
       SELECT po_id, supplier_id, product_id, order_date::date, expected_delivery_date::date,
              actual_delivery_date::date, delivery_status, quantity_ordered::integer,
              quantity_received::integer, defective_units::integer, unit_cost::numeric,
              order_cost::numeric
       FROM raw.purchase_orders ORDER BY _row_id""",
    """INSERT INTO analytics.inventory_daily
       SELECT product_id, "date"::date, opening_stock::integer, initial_stock_units::integer,
              po_receipts_units::integer, defective_writeoff_units::integer,
              units_demanded::integer, units_fulfilled::integer, closing_stock::integer,
              backorder_units::integer, stockout_flag::smallint, inventory_value::numeric
       FROM raw.inventory_daily ORDER BY _row_id""",
    """INSERT INTO analytics.promotion_log (table_name, row_key, rule, action)
       SELECT 'shipments', s.shipment_id, 'order_not_in_analytics', 'quarantined'
       FROM raw.shipments s
       WHERE NOT EXISTS (SELECT 1 FROM analytics.orders o WHERE o.order_id = s.order_id)""",
    """INSERT INTO analytics.shipments
       SELECT s.shipment_id, s.order_id, s.carrier, s.ship_date::date,
              s.promised_delivery_date::date, s.delivered_date::date, s.shipment_status,
              s.had_backorder::smallint, s.handling_days::integer, s.transit_days::integer,
              s.delivery_delay_days::integer, s.shipping_cost::numeric
       FROM raw.shipments s
       WHERE EXISTS (SELECT 1 FROM analytics.orders o WHERE o.order_id = s.order_id)
       ORDER BY s._row_id""",
    _DAILY_METRICS_SQL,
]


def build_analytics(engine: Engine) -> dict[str, int]:
    """Rebuild the analytics schema in one transaction. Returns rows per table.

    Any error (including a constraint violation) rolls everything back, leaving the previous
    analytics schema untouched.
    """
    with engine.begin() as conn:
        conn.exec_driver_sql(f"DROP SCHEMA IF EXISTS {ANALYTICS_SCHEMA} CASCADE")
        for statement in analytics_ddl():
            conn.exec_driver_sql(statement)
        conn.exec_driver_sql(PROMOTION_LOG_SQL)
        for statement in _INSERTS:
            conn.exec_driver_sql(" ".join(statement.split()))
        return {
            t: int(conn.execute(text(f"SELECT count(*) FROM {ANALYTICS_SCHEMA}.{t}")).scalar())
            for t in LOAD_ORDER
        }


_METRIC_CHECKS = [
    ("orders_placed", "SELECT count(*) FROM analytics.orders"),
    ("orders_completed", "SELECT count(*) FROM analytics.orders WHERE order_status = 'Completed'"),
    ("new_customers", "SELECT count(*) FROM analytics.customers "
                      "WHERE signup_date BETWEEN DATE '2023-01-01' AND DATE '2025-12-31'"),
    ("units_sold", "SELECT sum(i.quantity) FROM analytics.order_items i "
                   "JOIN analytics.orders o USING (order_id) WHERE o.order_status = 'Completed'"),
    ("gross_revenue", "SELECT sum(i.line_total) FROM analytics.order_items i "
                      "JOIN analytics.orders o USING (order_id) WHERE o.order_status = 'Completed'"),
    ("cogs", "SELECT sum(i.quantity * i.unit_cost) FROM analytics.order_items i "
             "JOIN analytics.orders o USING (order_id) WHERE o.order_status = 'Completed'"),
    ("shipping_cost", "SELECT sum(shipping_cost) FROM analytics.shipments"),
    ("payments_captured", "SELECT sum(amount) FROM analytics.payments WHERE payment_status = 'Captured'"),
    ("payments_failed", "SELECT sum(amount) FROM analytics.payments WHERE payment_status = 'Failed'"),
    ("refunds_amount", "SELECT sum(refund_amount) FROM analytics.refunds"),
    ("po_spend", "SELECT sum(order_cost) FROM analytics.purchase_orders"),
    ("listed_products", "SELECT count(*) FROM analytics.inventory_daily"),
    ("products_in_stockout", "SELECT sum(stockout_flag) FROM analytics.inventory_daily"),
    ("backorder_units", "SELECT sum(backorder_units) FROM analytics.inventory_daily"),
    ("inventory_value", "SELECT sum(inventory_value) FROM analytics.inventory_daily"),
]

_INTEGRITY = [
    ("completed orders without a payment",
     "SELECT count(*) FROM analytics.orders o LEFT JOIN analytics.payments p USING (order_id) "
     "WHERE o.order_status = 'Completed' AND p.payment_id IS NULL"),
    ("completed orders without a shipment",
     "SELECT count(*) FROM analytics.orders o LEFT JOIN analytics.shipments s USING (order_id) "
     "WHERE o.order_status = 'Completed' AND s.shipment_id IS NULL"),
    ("payments whose amount differs from their order lines",
     "SELECT count(*) FROM analytics.payments p JOIN (SELECT order_id, sum(line_total) AS total "
     "FROM analytics.order_items GROUP BY order_id) t USING (order_id) WHERE abs(p.amount - t.total) > 0.01"),
]


def check_analytics(engine: Engine) -> list[str]:
    """Accounting, integrity and metric reconciliation checks. Empty list means all good."""
    problems: list[str] = []
    with engine.connect() as conn:
        def scalar(sql: str) -> float:
            return float(conn.execute(text(sql)).scalar() or 0)

        for table in LOAD_ORDER:
            actual = int(scalar(f"SELECT count(*) FROM analytics.{table}"))
            if table in CLEANED:
                expected = int(scalar(f"SELECT count(*) FROM staging.{table}"))
            elif table in FILTERED:
                logged = int(scalar(f"SELECT count(*) FROM analytics.promotion_log WHERE table_name = '{table}'"))
                expected = int(scalar(f"SELECT count(*) FROM raw.{table}")) - logged
            else:
                expected = int(scalar(f"SELECT count(*) FROM raw.{table}"))
            if actual != expected:
                problems.append(f"{table}: {actual} rows in analytics, {expected} expected")

        for name, sql in _INTEGRITY:
            n = int(scalar(sql))
            if n:
                problems.append(f"{n} {name}")

        for column, source_sql in _METRIC_CHECKS:
            in_metrics = scalar(f"SELECT sum({column}) FROM analytics.daily_metrics")
            in_source = scalar(source_sql)
            if abs(in_metrics - in_source) > 0.05:
                problems.append(f"daily_metrics.{column}: {in_metrics:,.2f} against {in_source:,.2f} in the tables")
    return problems