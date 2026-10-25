"""Profile the cleaned layer and run the statistical tests (SYNTHETIC data)."""

from __future__ import annotations

import pandas as pd

from nexus.db import get_engine
from nexus.sql_analytics import run_query
from nexus.stats_analysis import TABLE_NAMES, load_tables, profile_frame, run_hypothesis_tests, seasonal_index

PROFILE_TABLES = ["customers", "orders", "order_items", "payments", "shipments", "purchase_orders"]


def main() -> int:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    engine = get_engine()
    t = load_tables(engine, TABLE_NAMES)
    print("SYNTHETIC DATA: PROFILING AND STATISTICAL TESTS ON THE CLEANED LAYER")

    prof = pd.concat([profile_frame(t[n], n) for n in PROFILE_TABLES], ignore_index=True)
    numeric = prof.dropna(subset=["mean"])
    print("\nNumeric columns (skew above 1 means a long right tail; outlier_pct uses the 1.5 x IQR rule):")
    cols = ["table", "column", "mean", "median", "std", "min", "max", "skew", "outlier_pct"]
    print(numeric[cols].round(2).to_string(index=False))

    missing = prof[prof["missing"] > 0]
    print("\nColumns with missing values (all are intentional or flagged earlier):")
    print(missing[["table", "column", "missing", "missing_pct"]].to_string(index=False))

    print("\nSeasonal index (month revenue / that year's average month, averaged over years):")
    print(seasonal_index(t["daily_metrics"]).to_string())

    cust = run_query(engine, "customer_level")
    res = run_hypothesis_tests(t, cust)
    shown = res.drop(columns=["question", "detail"]).copy()
    for col in ("p_value", "p_holm"):
        shown[col] = shown[col].map("{:.2e}".format)
    shown["effect"] = shown["effect"].round(3)
    print("\nHypothesis tests (Holm-adjusted across all seven):")
    print(shown.to_string(index=False))
    print("\nDetails:")
    for r in res.itertuples(index=False):
        print(f"  {r.id} {r.question}\n      {r.detail}")
    print(f"\nTests consistent with the simulation design: {int(res['consistent_with_design'].sum())} of {len(res)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())