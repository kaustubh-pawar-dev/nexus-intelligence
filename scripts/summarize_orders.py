"""Print a summary of the generated orders (reads data/raw; SYNTHETIC data)."""

from __future__ import annotations

import pandas as pd

from nexus.config import get_settings


def main() -> int:
    raw = get_settings().raw_dir
    customers = pd.read_csv(raw / "customers.csv", parse_dates=["signup_date"])
    orders = pd.read_csv(raw / "orders.csv", parse_dates=["order_date"])
    items = pd.read_csv(raw / "order_items.csv")

    print("SYNTHETIC DATA SUMMARY")
    print(f"orders: {len(orders):,}   order_items: {len(items):,}")
    print(f"average lines per order: {len(items) / len(orders):.2f}")
    print(
        f"customers with at least one order: {orders['customer_id'].nunique():,} "
        f"of {len(customers):,}"
    )

    print("\nOrders per year:")
    print(orders.groupby(orders["order_date"].dt.year).size().to_string())

    print("\nOrder status share:")
    print(orders["order_status"].value_counts(normalize=True).round(3).to_string())

    per_customer = orders.groupby("customer_id").size()
    per_customer = per_customer.reindex(customers["customer_id"], fill_value=0)
    print("\nMean orders per customer by segment (2023-2025):")
    print(per_customer.groupby(customers["segment"].to_numpy()).mean().round(2).to_string())

    print("\nOrders per calendar month (all three years combined):")
    print(orders.groupby(orders["order_date"].dt.month).size().to_string())

    completed = orders.loc[orders["order_status"] == "Completed", ["order_id", "order_date"]]
    lines = items.merge(completed, on="order_id")
    lines["cogs"] = lines["quantity"] * lines["unit_cost"]
    by_year = lines.groupby(lines["order_date"].dt.year)[["line_total", "cogs"]].sum()
    by_year["gross_margin_pct"] = ((by_year["line_total"] - by_year["cogs"]) / by_year["line_total"] * 100).round(1)
    print("\nSYNTHETIC revenue and cost of goods sold by year (completed orders, simulated units):")
    print(by_year.round(0).assign(gross_margin_pct=by_year["gross_margin_pct"]).to_string())

    print("\nShare of orders with a discount:", round((orders["discount_pct"] > 0).mean(), 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())