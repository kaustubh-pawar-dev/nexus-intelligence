"""Run a few business queries against the core schema (SYNTHETIC data)."""

from __future__ import annotations

import pandas as pd

from nexus.db import get_engine

QUERIES = [
    (
        "Row counts",
        """
        SELECT 'customers' AS table_name, count(*) AS rows FROM core.customers
        UNION ALL SELECT 'orders', count(*) FROM core.orders
        UNION ALL SELECT 'order_items', count(*) FROM core.order_items
        UNION ALL SELECT 'payments', count(*) FROM core.payments
        UNION ALL SELECT 'refunds', count(*) FROM core.refunds
        UNION ALL SELECT 'purchase_orders', count(*) FROM core.purchase_orders
        UNION ALL SELECT 'inventory_daily', count(*) FROM core.inventory_daily
        UNION ALL SELECT 'shipments', count(*) FROM core.shipments
        UNION ALL SELECT 'daily_metrics', count(*) FROM core.daily_metrics
        """,
    ),
    (
        "Revenue, cost and gross margin by year (completed orders)",
        """
        SELECT extract(year FROM o.order_date)::int AS year,
               round(sum(i.line_total), 0) AS revenue,
               round(sum(i.quantity * i.unit_cost), 0) AS cogs,
               round(100 * (sum(i.line_total) - sum(i.quantity * i.unit_cost)) / sum(i.line_total), 1) AS gross_margin_pct
        FROM core.order_items i JOIN core.orders o USING (order_id)
        WHERE o.order_status = 'Completed'
        GROUP BY 1 ORDER BY 1
        """,
    ),
    (
        "Revenue by customer segment (completed orders)",
        """
        SELECT c.segment, count(DISTINCT o.order_id) AS orders,
               round(sum(i.line_total), 0) AS revenue
        FROM core.order_items i
        JOIN core.orders o USING (order_id)
        JOIN core.customers c USING (customer_id)
        WHERE o.order_status = 'Completed'
        GROUP BY 1 ORDER BY revenue DESC
        """,
    ),
    (
        "Top 5 products by revenue",
        """
        SELECT p.product_id, p.product_name, p.category, round(sum(i.line_total), 0) AS revenue
        FROM core.order_items i
        JOIN core.orders o USING (order_id)
        JOIN core.products p USING (product_id)
        WHERE o.order_status = 'Completed'
        GROUP BY 1, 2, 3 ORDER BY revenue DESC LIMIT 5
        """,
    ),
    (
        "Refund rate by order defect category: refunds by reason",
        """
        SELECT refund_reason, count(*) AS refunds, round(sum(refund_amount), 0) AS refunded
        FROM core.refunds GROUP BY 1 ORDER BY refunds DESC
        """,
    ),
    (
        "On-time purchase-order delivery by supplier region (delivered only)",
        """
        SELECT s.region, count(*) AS deliveries,
               round(100.0 * avg((po.actual_delivery_date <= po.expected_delivery_date)::int), 1) AS on_time_pct
        FROM core.purchase_orders po JOIN core.suppliers s USING (supplier_id)
        WHERE po.delivery_status = 'Delivered'
        GROUP BY 1 ORDER BY on_time_pct
        """,
    ),
    (
        "Stockout days by year",
        """
        SELECT extract(year FROM date)::int AS year, sum(stockout_flag) AS stockout_product_days,
               round(100.0 * avg(stockout_flag), 2) AS stockout_pct
        FROM core.inventory_daily GROUP BY 1 ORDER BY 1
        """,
    ),
]


def main() -> int:
    engine = get_engine()
    print("SYNTHETIC DATA: BUSINESS QUERIES RUN IN POSTGRESQL")
    with engine.connect() as conn:
        for title, sql in QUERIES:
            print(f"\n{title}:")
            print(pd.read_sql_query(sql, conn).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())