"""Profile the as-delivered data in the raw schema with SQL (reads only raw tables; SYNTHETIC data)."""

from __future__ import annotations

import pandas as pd

from nexus.db import get_engine

QUERIES = [
    (
        "Rows against distinct keys (a gap means duplicate rows)",
        """
        SELECT 'customers' AS table_name, count(*) AS rows, count(DISTINCT customer_id) AS distinct_keys FROM raw.customers
        UNION ALL SELECT 'orders', count(*), count(DISTINCT order_id) FROM raw.orders
        UNION ALL SELECT 'order_items', count(*), count(DISTINCT order_id || '|' || line_number) FROM raw.order_items
        UNION ALL SELECT 'payments', count(*), count(DISTINCT payment_id) FROM raw.payments
        """,
    ),
    (
        "Missing values",
        """
        SELECT 'customers.email' AS column_name, count(*) FILTER (WHERE email IS NULL) AS missing FROM raw.customers
        UNION ALL SELECT 'orders.customer_id', count(*) FILTER (WHERE customer_id IS NULL) FROM raw.orders
        UNION ALL SELECT 'order_items.line_total', count(*) FILTER (WHERE line_total IS NULL) FROM raw.order_items
        UNION ALL SELECT 'payments.amount', count(*) FILTER (WHERE amount IS NULL) FROM raw.payments
        """,
    ),
    (
        "customers.country values",
        "SELECT country, count(*) AS rows FROM raw.customers GROUP BY 1 ORDER BY 2 DESC, 1",
    ),
    (
        "orders.order_status values",
        "SELECT order_status, count(*) AS rows FROM raw.orders GROUP BY 1 ORDER BY 2 DESC, 1",
    ),
    (
        "payments.payment_method values",
        "SELECT payment_method, count(*) AS rows FROM raw.payments GROUP BY 1 ORDER BY 2 DESC, 1",
    ),
    (
        "orders.order_date formats",
        """
        SELECT CASE WHEN order_date ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' THEN 'yyyy-mm-dd'
                    WHEN order_date ~ '^[0-9]{2}/[0-9]{2}/[0-9]{4}$' THEN 'dd/mm/yyyy'
                    ELSE 'other' END AS format,
               count(*) AS rows
        FROM raw.orders GROUP BY 1 ORDER BY 2 DESC
        """,
    ),
    (
        "Invalid or out-of-range values",
        """
        SELECT 'customers: signup_date after 2025-12-31' AS check_name, count(*) AS rows FROM raw.customers WHERE signup_date > '2025-12-31'
        UNION ALL SELECT 'orders: discount_pct above 0.20', count(*) FROM raw.orders WHERE discount_pct::numeric > 0.20
        UNION ALL SELECT 'orders: customer_id not in raw.customers', count(*) FROM raw.orders WHERE customer_id NOT IN (SELECT customer_id FROM raw.customers)
        UNION ALL SELECT 'order_items: quantity below 1', count(*) FROM raw.order_items WHERE quantity::integer < 1
        UNION ALL SELECT 'order_items: unit_price is 0', count(*) FROM raw.order_items WHERE unit_price::numeric = 0
        """,
    ),
    (
        "Load log (lineage)",
        "SELECT table_name, source_file, rows_loaded, left(file_sha256, 12) AS sha256_start FROM raw.load_log ORDER BY table_name",
    ),
]


def main() -> int:
    engine = get_engine()
    print("SYNTHETIC DATA: PROFILE OF THE RAW LAYER (AS DELIVERED, NO CLEANING YET)")
    with engine.connect() as conn:
        for title, sql in QUERIES:
            print(f"\n{title}:")
            print(pd.read_sql_query(sql, conn).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())