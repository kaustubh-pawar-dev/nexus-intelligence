"""Print a summary of the generated purchase orders (reads data/raw; SYNTHETIC data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nexus.config import get_settings
from nexus.datagen.constants import DATA_START
from nexus.datagen.purchasing import daily_units_sold, initial_stock


def main() -> int:
    raw = get_settings().raw_dir
    suppliers = pd.read_csv(raw / "suppliers.csv")
    products = pd.read_csv(raw / "products.csv", parse_dates=["launch_date"])
    orders = pd.read_csv(raw / "orders.csv", parse_dates=["order_date"])
    items = pd.read_csv(raw / "order_items.csv")
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

    sales = daily_units_sold(orders, items, products)
    print(f"\nUnits ordered / units sold (all years): {po['quantity_ordered'].sum() / sales.sum():.3f}")

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

    short = d["quantity_received"] < d["quantity_ordered"]
    print(f"\nShort-shipment share (delivered): {short.mean():.3f}")
    units_short = (d["quantity_ordered"] - d["quantity_received"]).sum()
    print(f"Units short as a share of units ordered: {units_short / d['quantity_ordered'].sum():.3f}")
    print(f"Defective units as a share of units received: {d['defective_units'].sum() / d['quantity_received'].sum():.4f}")

    # ---- Preview of the inventory implied by the purchase orders (no write-offs) ----
    start = pd.Timestamp(DATA_START)
    n_prod, n_days = sales.shape
    start_idx, init_qty = initial_stock(sales, products)
    flow = -sales
    flow[np.arange(n_prod), start_idx] += init_qty
    prod_pos = pd.Index(products["product_id"]).get_indexer(d["product_id"])
    arrival = (d["actual_delivery_date"] - start).dt.days.to_numpy()
    np.add.at(flow, (prod_pos, arrival), d["quantity_received"].astype("int64").to_numpy())
    net = np.cumsum(flow, axis=1)  # negative values mean units sold but not yet in stock

    launch_idx = (products["launch_date"] - start).dt.days.to_numpy()
    listed = np.arange(n_days)[None, :] >= launch_idx[:, None]
    backorder = (net < 0) & listed
    on_hand = np.maximum(net, 0)
    unit_cost = products["unit_cost"].to_numpy()
    value_by_day = (on_hand * unit_cost[:, None]).sum(axis=0)
    cogs_by_day = (sales * unit_cost[:, None]).sum(axis=0)
    years = (start + pd.to_timedelta(np.arange(n_days), unit="D")).year.to_numpy()

    print("\nPREVIEW of the inventory implied by the purchase orders (no write-offs; SYNTHETIC):")
    print(f"share of listed product-days with backorders: {backorder.sum() / listed.sum():.4f}")
    print(f"products with at least one backorder day: {int(backorder.any(axis=1).sum())} of {n_prod}")

    rows = []
    for yr in np.unique(years):
        m = years == yr
        cogs = cogs_by_day[m].sum()
        avg_value = value_by_day[m].mean()
        rows.append(
            {
                "year": int(yr),
                "backorder_day_share": backorder[:, m].sum() / listed[:, m].sum(),
                "avg_on_hand_units": on_hand[:, m].sum(axis=0).mean(),
                "cogs": cogs,
                "avg_inventory_value": avg_value,
                "turnover": cogs / avg_value,
            }
        )
    print(pd.DataFrame(rows).set_index("year").round(3).to_string())

    region = products["supplier_id"].map(suppliers.set_index("supplier_id")["region"]).to_numpy()
    by_region = {
        str(r): backorder[region == r].sum() / listed[region == r].sum()
        for r in sorted(set(region))
    }
    print("\nBackorder-day share by supplier region:")
    print(pd.Series(by_region).round(4).to_string())

    end_units = int(on_hand[:, -1].sum())
    end_value = float((on_hand[:, -1] * unit_cost).sum())
    print(f"\nOn hand at the end of the window: {end_units:,} units, {end_value:,.0f} at cost")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())