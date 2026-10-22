"""Run the analytic SQL queries and print the results (cleaned layer; SYNTHETIC data)."""

from __future__ import annotations

from nexus.db import get_engine
from nexus.sql_analytics import list_queries, run_query

TITLES = {
    "monthly_revenue": "Monthly revenue with MoM, YoY and a 3-month moving average",
    "contribution_by_category": "Contribution margin by category (revenue - COGS - shipping - refunds)",
    "contribution_by_segment": "Contribution margin by customer segment",
    "product_rank_in_category": "Top 3 products per category by contribution margin",
}


def main() -> int:
    engine = get_engine()
    print("SYNTHETIC DATA: ADVANCED SQL ANALYTICS ON THE CLEANED LAYER")
    print("(revenue held out by quarantine is not included; failed payments are shown separately)")
    for name in list_queries():
        print(f"\n{TITLES.get(name, name)}:")
        print(run_query(engine, name).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())