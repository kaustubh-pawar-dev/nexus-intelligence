"""Summarize the analytics schema and measure it against ground truth (SYNTHETIC data).

The comparison with core is an evaluation. The promotion itself never reads core.
"""

from __future__ import annotations

import pandas as pd

from nexus.db import get_engine

TABLES = ["suppliers", "products", "customers", "orders", "order_items", "payments",
          "refunds", "purchase_orders", "inventory_daily", "shipments", "daily_metrics"]


def _except(table: str, left: str, right: str) -> str:
    return f"SELECT count(*) FROM (SELECT * FROM {left}.{table} EXCEPT SELECT * FROM {right}.{table}) x"


def main() -> int:
    engine = get_engine()
    print("SYNTHETIC DATA: ANALYTICS LAYER (CLEANED, CONSTRAINT-CHECKED)")
    with engine.connect() as conn:
        rows = []
        for t in TABLES:
            rows.append(
                {
                    "table": t,
                    "analytics_rows": conn.exec_driver_sql(f"SELECT count(*) FROM analytics.{t}").scalar(),
                    "ground_truth_rows": conn.exec_driver_sql(f"SELECT count(*) FROM core.{t}").scalar(),
                    "in_analytics_not_in_truth": conn.exec_driver_sql(_except(t, "analytics", "core")).scalar(),
                    "in_truth_not_in_analytics": conn.exec_driver_sql(_except(t, "core", "analytics")).scalar(),
                }
            )
        print("\nRow counts and differences against ground truth (rows that differ in any column):")
        print(pd.DataFrame(rows).to_string(index=False))

        print("\nKnock-on quarantine log:")
        print(pd.read_sql_query(
            "SELECT table_name, rule, action, count(*) AS rows FROM analytics.promotion_log GROUP BY 1, 2, 3 ORDER BY 1",
            conn).to_string(index=False))

        print("\nYearly revenue and gross margin (completed orders, cleaned layer):")
        print(pd.read_sql_query(
            """
            SELECT extract(year FROM o.order_date)::int AS year, round(sum(i.line_total), 0) AS revenue,
                   round(100 * (sum(i.line_total) - sum(i.quantity * i.unit_cost)) / sum(i.line_total), 1) AS gross_margin_pct
            FROM analytics.order_items i JOIN analytics.orders o USING (order_id)
            WHERE o.order_status = 'Completed' GROUP BY 1 ORDER BY 1
            """, conn).to_string(index=False))

        print("\nRevenue against ground truth:")
        print(pd.read_sql_query(
            """
            SELECT round((SELECT sum(gross_revenue) FROM analytics.daily_metrics), 2) AS analytics_revenue,
                   round((SELECT sum(gross_revenue) FROM core.daily_metrics), 2) AS ground_truth_revenue,
                   round((SELECT sum(gross_revenue) FROM core.daily_metrics)
                       - (SELECT sum(gross_revenue) FROM analytics.daily_metrics), 2) AS held_out_by_quarantine
            """, conn).T.to_string(header=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())