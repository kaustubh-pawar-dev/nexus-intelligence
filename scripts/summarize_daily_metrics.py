"""Print a summary of the generated daily business metrics (reads data/raw; SYNTHETIC data)."""

from __future__ import annotations

import pandas as pd

from nexus.config import get_settings

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def main() -> int:
    raw = get_settings().raw_dir
    dm = pd.read_csv(raw / "daily_metrics.csv", parse_dates=["date"])
    payments = pd.read_csv(raw / "payments.csv")
    refunds = pd.read_csv(raw / "refunds.csv")
    shipments = pd.read_csv(raw / "shipments.csv")
    orders = pd.read_csv(raw / "orders.csv")
    dm["year"] = dm["date"].dt.year

    print("SYNTHETIC DATA SUMMARY: DAILY BUSINESS METRICS")
    print(f"rows: {len(dm):,}   dates: {dm['date'].min().date()} to {dm['date'].max().date()}")
    print(f"days with zero orders: {int((dm['orders_placed'] == 0).sum())}")

    yearly = dm.groupby("year").agg(
        orders=("orders_completed", "sum"),
        units=("units_sold", "sum"),
        revenue=("gross_revenue", "sum"),
        discounts=("discount_amount", "sum"),
        cogs=("cogs", "sum"),
        shipping=("shipping_cost", "sum"),
        captured=("payments_captured", "sum"),
        failed=("payments_failed", "sum"),
        refunds=("refunds_amount", "sum"),
    )
    yearly["gross_margin_pct"] = (yearly["revenue"] - yearly["cogs"]) / yearly["revenue"] * 100
    yearly["failed_pct_of_payments"] = yearly["failed"] / (yearly["captured"] + yearly["failed"]) * 100
    yearly["refunds_pct_of_captured"] = yearly["refunds"] / yearly["captured"] * 100
    print("\nYearly KPIs (order basis for orders to shipping, cash basis for captured, failed and refunds):")
    print(yearly.round(1).T.to_string())

    print("\nAverage daily gross revenue by calendar month (all three years):")
    print(dm.groupby(dm["date"].dt.month)["gross_revenue"].mean().round(0).to_string())

    weekday = dm.groupby(dm["date"].dt.dayofweek)["orders_placed"].mean().round(1)
    weekday.index = [WEEKDAYS[i] for i in weekday.index]
    print("\nAverage orders placed by weekday:")
    print(weekday.to_string())

    top = dm.sort_values("gross_revenue", ascending=False).head(5)[["date", "orders_placed", "gross_revenue"]]
    print("\nFive highest revenue days:")
    print(top.round(0).to_string(index=False))
    low = dm.sort_values("gross_revenue").head(5)[["date", "orders_placed", "gross_revenue"]]
    print("\nFive lowest revenue days:")
    print(low.round(0).to_string(index=False))

    print("\nProducts in stockout per day (average) by year:")
    print(dm.groupby("year")["products_in_stockout"].mean().round(2).to_string())

    print("\nReconciliation against the source CSVs (every difference should be about 0):")
    diffs = {
        "orders_placed vs orders.csv rows": dm["orders_placed"].sum() - len(orders),
        "gross_revenue vs payments.csv amount": dm["gross_revenue"].sum() - payments["amount"].sum(),
        "payments_captured + failed vs payments.csv amount": dm["payments_captured"].sum() + dm["payments_failed"].sum() - payments["amount"].sum(),
        "refunds_amount vs refunds.csv": dm["refunds_amount"].sum() - refunds["refund_amount"].sum(),
        "shipping_cost vs shipments.csv": dm["shipping_cost"].sum() - shipments["shipping_cost"].sum(),
    }
    for name, value in diffs.items():
        print(f"  {name}: {value:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())