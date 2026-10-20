"""Summarize the staging layer and measure it against ground truth (SYNTHETIC data).

The comparison with the core schema is an evaluation of the pipeline. The pipeline itself
never reads core or the defect manifest.
"""

from __future__ import annotations

import pandas as pd

from nexus.db import get_engine

ACCOUNTING = """
SELECT t.table_name, t.raw_rows, t.staged_rows,
       coalesce(i.dropped, 0) AS dropped_duplicates, coalesce(i.quarantined, 0) AS quarantined,
       t.raw_rows - t.staged_rows - coalesce(i.dropped, 0) - coalesce(i.quarantined, 0) AS unaccounted
FROM (
    SELECT 'customers' AS table_name, (SELECT count(*) FROM raw.customers) AS raw_rows,
           (SELECT count(*) FROM staging.customers) AS staged_rows
    UNION ALL SELECT 'orders', (SELECT count(*) FROM raw.orders), (SELECT count(*) FROM staging.orders)
    UNION ALL SELECT 'order_items', (SELECT count(*) FROM raw.order_items), (SELECT count(*) FROM staging.order_items)
    UNION ALL SELECT 'payments', (SELECT count(*) FROM raw.payments), (SELECT count(*) FROM staging.payments)
) t
LEFT JOIN (
    SELECT table_name,
           count(DISTINCT source_row_id) FILTER (WHERE action = 'dropped') AS dropped,
           count(DISTINCT source_row_id) FILTER (WHERE action = 'quarantined') AS quarantined
    FROM staging.row_issues GROUP BY table_name
) i USING (table_name)
ORDER BY t.table_name
"""

GROUND_TRUTH = """
SELECT 'orders: staged rows missing from ground truth' AS check_name, count(*) AS rows
    FROM staging.orders s LEFT JOIN core.orders c USING (order_id) WHERE c.order_id IS NULL
UNION ALL SELECT 'orders: any column differs', count(*)
    FROM staging.orders s JOIN core.orders c USING (order_id)
    WHERE s.customer_id <> c.customer_id OR s.order_date <> c.order_date OR s.order_status <> c.order_status
       OR s.discount_pct <> c.discount_pct OR s.sales_channel <> c.sales_channel
UNION ALL SELECT 'customers: country, region or segment differs', count(*)
    FROM staging.customers s JOIN core.customers c USING (customer_id)
    WHERE s.country <> c.country OR s.region <> c.region OR s.segment <> c.segment
UNION ALL SELECT 'customers: email present but different', count(*)
    FROM staging.customers s JOIN core.customers c USING (customer_id)
    WHERE s.email IS NOT NULL AND s.email <> c.email
UNION ALL SELECT 'customers: signup_date differs (imputed dates)', count(*)
    FROM staging.customers s JOIN core.customers c USING (customer_id) WHERE s.signup_date <> c.signup_date
UNION ALL SELECT 'customers: largest signup_date error in days', coalesce(max(abs(s.signup_date - c.signup_date)), 0)
    FROM staging.customers s JOIN core.customers c USING (customer_id)
UNION ALL SELECT 'order_items: any column differs', count(*)
    FROM staging.order_items s JOIN core.order_items c USING (order_id, line_number)
    WHERE s.product_id <> c.product_id OR s.quantity <> c.quantity OR s.unit_price <> c.unit_price
       OR s.unit_cost <> c.unit_cost OR s.discount_pct <> c.discount_pct OR s.line_total <> c.line_total
UNION ALL SELECT 'payments: any column differs', count(*)
    FROM staging.payments s JOIN core.payments c USING (payment_id)
    WHERE s.order_id <> c.order_id OR s.payment_date <> c.payment_date OR s.payment_method <> c.payment_method
       OR s.payment_status <> c.payment_status OR s.amount <> c.amount
"""

REVENUE = """
SELECT
  round((SELECT sum(i.line_total) FROM staging.order_items i JOIN staging.orders o USING (order_id)
         WHERE o.order_status = 'Completed'), 2) AS staged_revenue,
  round((SELECT sum(i.line_total) FROM core.order_items i JOIN core.orders o USING (order_id)
         WHERE o.order_status = 'Completed'), 2) AS ground_truth_revenue,
  round((SELECT coalesce(sum(i.line_total), 0) FROM core.order_items i JOIN core.orders o USING (order_id)
         WHERE o.order_status = 'Completed'
           AND o.order_id IN (SELECT row_key FROM staging.row_issues
                              WHERE table_name = 'orders' AND action = 'quarantined')), 2)
         AS revenue_in_quarantined_orders
"""


def main() -> int:
    engine = get_engine()
    print("SYNTHETIC DATA: STAGING LAYER (CLEANED FROM RAW)")
    with engine.connect() as conn:
        print("\nEvery raw row is staged, a dropped duplicate or quarantined:")
        print(pd.read_sql_query(ACCOUNTING, conn).to_string(index=False))

        print("\nCleaning log (row-level actions grouped by rule):")
        log = pd.read_sql_query(
            "SELECT table_name, action, rule, rows FROM staging.cleaning_summary ORDER BY table_name, action, rule",
            conn,
        )
        print(log.to_string(index=False))

        print("\nAgainst ground truth (an evaluation only; every difference should be explained):")
        print(pd.read_sql_query(GROUND_TRUTH, conn).to_string(index=False))

        revenue = pd.read_sql_query(REVENUE, conn)
        revenue["unexplained_gap"] = (
            revenue["ground_truth_revenue"] - revenue["staged_revenue"] - revenue["revenue_in_quarantined_orders"]
        ).round(2)
        print("\nRevenue held back by quarantine (completed orders):")
        print(revenue.T.to_string(header=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())