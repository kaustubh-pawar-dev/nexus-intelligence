"""Print a summary of the generated shipments (reads data/raw; SYNTHETIC data)."""

from __future__ import annotations

import pandas as pd

from nexus.config import get_settings


def main() -> int:
    raw = get_settings().raw_dir
    customers = pd.read_csv(raw / "customers.csv")
    orders = pd.read_csv(raw / "orders.csv", parse_dates=["order_date"])
    items = pd.read_csv(raw / "order_items.csv")
    refunds = pd.read_csv(raw / "refunds.csv")
    sh = pd.read_csv(
        raw / "shipments.csv",
        parse_dates=["ship_date", "promised_delivery_date", "delivered_date"],
    )
    sh = sh.merge(orders[["order_id", "customer_id", "order_date"]], on="order_id").merge(
        customers[["customer_id", "region"]], on="customer_id"
    )
    sh["year"] = sh["order_date"].dt.year

    print("SYNTHETIC DATA SUMMARY: SHIPMENTS")
    print(f"shipments: {len(sh):,}")
    print("\nShipment status share:")
    print(sh["shipment_status"].value_counts(normalize=True).round(4).to_string())

    d = sh[sh["shipment_status"] == "Delivered"].copy()
    d["late"] = d["delivery_delay_days"] > 0
    print(f"\nDelivered: {len(d):,}   late share: {d['late'].mean():.3f}")
    print(f"mean delay of late deliveries (days): {d.loc[d['late'], 'delivery_delay_days'].mean():.2f}")
    print(f"mean handling days: {d['handling_days'].mean():.2f}   mean transit days: {d['transit_days'].mean():.2f}")
    print("late share by order year:")
    print(d.groupby("year")["late"].mean().round(3).to_string())

    print("\nBy carrier:")
    by_carrier = d.groupby("carrier").agg(
        delivered=("shipment_id", "count"),
        late_share=("late", "mean"),
        mean_transit=("transit_days", "mean"),
    )
    by_carrier["mean_cost"] = sh.groupby("carrier")["shipping_cost"].mean()
    print(by_carrier.round(3).to_string())

    print("\nBy customer region:")
    by_region = d.groupby("region").agg(
        delivered=("shipment_id", "count"),
        late_share=("late", "mean"),
        mean_transit=("transit_days", "mean"),
    )
    by_region["mean_cost"] = sh.groupby("region")["shipping_cost"].mean()
    print(by_region.round(3).to_string())

    print("\nBackorder effect (any product on the order in backorder on the order date):")
    shipped = sh[sh["handling_days"].notna()]
    by_flag = shipped.groupby("had_backorder").agg(
        orders=("shipment_id", "count"), mean_handling_days=("handling_days", "mean")
    )
    by_flag["late_share_delivered"] = d.groupby("had_backorder")["late"].mean()
    print(by_flag.round(3).to_string())
    print(f"share of orders with a backorder line: {sh['had_backorder'].mean():.4f}")

    completed = orders.loc[orders["order_status"] == "Completed", ["order_id", "order_date"]]
    revenue = items.merge(completed, on="order_id").groupby(completed["order_date"].dt.year.rename("y").reindex(range(0)).index if False else items.merge(completed, on="order_id")["order_date"].dt.year)["line_total"].sum()
    cost = sh.groupby("year")["shipping_cost"].sum()
    table = pd.DataFrame({"revenue": revenue, "shipping_cost": cost})
    table["shipping_cost_share_of_revenue"] = table["shipping_cost"] / table["revenue"]
    print("\nSYNTHETIC shipping cost against revenue by order year (cost to the business):")
    print(table.round(4).to_string())

    d["refunded"] = d["order_id"].isin(refunds["order_id"])
    print("\nRefund rate of delivered orders (the Late Delivery refund reason is NOT linked to delay in the simulation):")
    print(d.groupby("late")["refunded"].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())