"""Run the part 2 analytic SQL queries and print the results (cleaned layer; SYNTHETIC data)."""

from __future__ import annotations

from nexus.db import get_engine
from nexus.sql_analytics import retention_curve, run_query

TITLES = {
    "customer_ltv": "Observed lifetime value (contribution margin to date) by segment and channel",
    "rfm_segments": "RFM segments as of 2025-12-31 (NTILE quartiles)",
    "supplier_performance": "Supplier scorecard (delivered purchase orders)",
    "inventory_turnover": "Inventory turnover by year and category",
    "revenue_leakage": "Order funnel and revenue leakage by year",
}


def main() -> int:
    engine = get_engine()
    print("SYNTHETIC DATA: ADVANCED SQL ANALYTICS, PART 2 (CLEANED LAYER)")

    cohorts = run_query(engine, "cohort_retention")
    print(f"\nCohort retention: {len(cohorts)} cohort-month rows, {cohorts['cohort_month'].nunique()} cohorts")
    print("Retention by months since first order (only cohorts old enough to be observed):")
    print(retention_curve(cohorts).to_string(index=False))

    for name, title in TITLES.items():
        print(f"\n{title}:")
        print(run_query(engine, name).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())