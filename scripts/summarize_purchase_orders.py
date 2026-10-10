"""Print a summary of the generated purchase orders (reads data/raw; SYNTHETIC data)."""

from __future__ import annotations

import pandas as pd

from nexus.config import get_settings


def main() -> int:
    raw = get_settings().raw_dir
    suppliers = pd.read_csv(raw / "suppliers.csv")
    po = pd.read_csv(
        raw / "purchase_orders.csv",
        parse_dates=["order_date", "expected_delivery_date", "actual_delivery_date"],
    )

    print("SYNTHETIC DATA SUMMARY: PURCHASE ORDERS")
    print(f"purchase orders: {len(po):,}   products: {po['product_id'].nunique()}   suppliers: {po['supplier_id'].nunique()}")

    print("\nDelivery status share:")
    print(po["delivery_status"].value_counts(normalize=True).round(3).to_string())

    print("\nPurchase orders and order cost by order year (SYNTHETIC cost):")
    by_year = po.groupby(po["order_date"].dt.year).agg(
        orders=("po_id", "count"), units=("quantity_ordered", "sum"), cost=("order_cost", "sum")
    )
    print(by_year.round(0).to_string())

    d = po[po["delivery_status"] == "Delivered"].copy()
    d["delay_days"] = (d["actual_delivery_date"] - d["expected_delivery_date"]).dt.days
    d["on_time"] = d["delay_days"] <= 0

    print(f"\nOverall on-time share (delivered only): {d['on_time'].mean():.3f}")
    print(f"Mean delay of late deliveries (days): {d.loc[d['delay_days'] > 0, 'delay_days'].mean():.2f}")

    print("\nOn-time share by order quarter (delivered only; built-in Q4 congestion):")
    print(d.groupby(d["order_date"].dt.quarter)["on_time"].mean().round(3).to_string())

    sup = d.groupby("supplier_id").agg(pos=("po_id", "count"), observed_on_time=("on_time", "mean"))
    sup = sup.join(
        suppliers.set_index("supplier_id")[["on_time_delivery_rate", "region"]]
    ).rename(columns={"on_time_delivery_rate": "stated_on_time"})
    print("\nFive least reliable suppliers by observed on-time share:")
    print(sup.sort_values("observed_on_time").head(5).round(3).to_string())
    print(f"\nCorrelation of observed and stated supplier on-time rate: {sup['observed_on_time'].corr(sup['stated_on_time']):.3f}")

    print("\nOn-time share by supplier region:")
    print(sup.groupby("region")["observed_on_time"].mean().round(3).to_string())

    short = d["quantity_received"] < d["quantity_ordered"]
    print(f"\nShort-shipment share (delivered): {short.mean():.3f}")
    units_short = (d["quantity_ordered"] - d["quantity_received"]).sum()
    print(f"Units short as a share of units ordered: {units_short / d['quantity_ordered'].sum():.3f}")
    print(f"Defective units as a share of units received: {d['defective_units'].sum() / d['quantity_received'].sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())