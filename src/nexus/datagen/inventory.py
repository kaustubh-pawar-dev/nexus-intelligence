"""Daily inventory snapshots (SYNTHETIC data).

The net position of each product is the running total of initial stock plus delivered
units, minus defective write-offs, minus demand. A positive net position is stock on hand.
A negative one is a backlog of unmet demand (backorders), filled first by later receipts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nexus.datagen.constants import DATA_END, DATA_START
from nexus.datagen.purchasing import daily_units_sold, initial_stock

INVENTORY_COLUMNS = [
    "product_id",
    "date",
    "opening_stock",
    "initial_stock_units",
    "po_receipts_units",
    "defective_writeoff_units",
    "units_demanded",
    "units_fulfilled",
    "closing_stock",
    "backorder_units",
    "stockout_flag",
    "inventory_value",
]
UNIT_COLUMNS = [
    "opening_stock",
    "initial_stock_units",
    "po_receipts_units",
    "defective_writeoff_units",
    "units_demanded",
    "units_fulfilled",
    "closing_stock",
    "backorder_units",
]


def build_inventory_daily(
    products: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    purchase_orders: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per product and listed day, sorted by product then date."""
    sales = daily_units_sold(orders, order_items, products)
    n_prod, n_days = sales.shape
    start_idx, init_qty = initial_stock(sales, products)
    rows = np.arange(n_prod)

    initial = np.zeros((n_prod, n_days), dtype=np.int64)
    initial[rows, start_idx] = init_qty

    delivered = purchase_orders[purchase_orders["delivery_status"] == "Delivered"]
    prod_pos = pd.Index(products["product_id"]).get_indexer(delivered["product_id"])
    arrival = (delivered["actual_delivery_date"] - pd.Timestamp(DATA_START)).dt.days.to_numpy()
    arrival = np.maximum(arrival, start_idx[prod_pos])  # early arrivals count on the first listed day

    received = np.zeros((n_prod, n_days), dtype=np.int64)
    defective = np.zeros((n_prod, n_days), dtype=np.int64)
    np.add.at(received, (prod_pos, arrival), delivered["quantity_received"].astype("int64").to_numpy())
    np.add.at(defective, (prod_pos, arrival), delivered["defective_units"].astype("int64").to_numpy())

    net = np.cumsum(initial + (received - defective) - sales, axis=1)
    closing = np.maximum(net, 0)
    backorder = np.maximum(-net, 0)
    opening = np.zeros_like(closing)
    opening[:, 1:] = closing[:, :-1]
    fulfilled = opening + initial + received - defective - closing

    listed = np.arange(n_days)[None, :] >= start_idx[:, None]
    p_i, d_i = np.nonzero(listed)  # row-major: product, then date
    unit_cost = products["unit_cost"].to_numpy()[p_i]
    dates = (pd.Timestamp(DATA_START) + pd.to_timedelta(d_i, unit="D")).to_numpy()

    return pd.DataFrame(
        {
            "product_id": products["product_id"].to_numpy()[p_i],
            "date": dates,
            "opening_stock": opening[p_i, d_i],
            "initial_stock_units": initial[p_i, d_i],
            "po_receipts_units": received[p_i, d_i],
            "defective_writeoff_units": defective[p_i, d_i],
            "units_demanded": sales[p_i, d_i],
            "units_fulfilled": fulfilled[p_i, d_i],
            "closing_stock": closing[p_i, d_i],
            "backorder_units": backorder[p_i, d_i],
            "stockout_flag": (backorder[p_i, d_i] > 0).astype(np.int64),
            "inventory_value": np.round(closing[p_i, d_i] * unit_cost, 2),
        }
    )


def validate_inventory(
    inv: pd.DataFrame,
    products: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    purchase_orders: pd.DataFrame,
) -> list[str]:
    """Return a list of problems found. An empty list means the table is valid."""
    if list(inv.columns) != INVENTORY_COLUMNS:
        return ["unexpected columns"]
    if inv.duplicated(["product_id", "date"]).any():
        return ["duplicate (product_id, date)"]
    if not inv["product_id"].isin(products["product_id"]).all():
        return ["inventory references unknown products"]

    problems: list[str] = []
    null_count = int(inv.isna().sum().sum())
    if null_count:
        problems.append(f"{null_count} unexpected null values")
    if not inv["date"].between(pd.Timestamp(DATA_START), pd.Timestamp(DATA_END)).all():
        problems.append("date outside the observation window")
    if (inv[UNIT_COLUMNS] < 0).any().any():
        problems.append("negative unit values")

    inv = inv.sort_values(["product_id", "date"]).reset_index(drop=True)
    g = inv.groupby("product_id")

    span = g["date"].agg(["min", "max", "count"])
    if len(span) != len(products):
        problems.append("products without inventory rows")
    if (((span["max"] - span["min"]).dt.days + 1) != span["count"]).any():
        problems.append("missing days within a product's series")
    if (span["max"] != pd.Timestamp(DATA_END)).any():
        problems.append("series does not end at the window end")
    expected_start = (
        products.set_index("product_id")["launch_date"]
        .clip(lower=pd.Timestamp(DATA_START))
        .reindex(span.index)
    )
    if (span["min"] != expected_start).any():
        problems.append("series does not start on the first listed day")

    prev_close = g["closing_stock"].shift(1).fillna(0)
    prev_back = g["backorder_units"].shift(1).fillna(0)
    if (inv["opening_stock"] != prev_close).any():
        problems.append("opening_stock differs from the previous closing_stock")
    stock_rhs = (
        inv["opening_stock"]
        + inv["initial_stock_units"]
        + inv["po_receipts_units"]
        - inv["defective_writeoff_units"]
        - inv["units_fulfilled"]
    )
    if (inv["closing_stock"] != stock_rhs).any():
        problems.append("stock identity broken")
    if (inv["backorder_units"] != prev_back + inv["units_demanded"] - inv["units_fulfilled"]).any():
        problems.append("backorder identity broken")
    if (inv["defective_writeoff_units"] > inv["po_receipts_units"]).any():
        problems.append("write-off exceeds receipts")
    if ((inv["closing_stock"] > 0) & (inv["backorder_units"] > 0)).any():
        problems.append("stock on hand and backorders at the same time")
    if (inv["stockout_flag"] != (inv["backorder_units"] > 0).astype(int)).any():
        problems.append("stockout_flag does not match backorder_units")
    cost = inv["product_id"].map(products.set_index("product_id")["unit_cost"])
    if ((inv["inventory_value"] - (inv["closing_stock"] * cost).round(2)).abs() > 0.005).any():
        problems.append("inventory_value does not match closing_stock * unit_cost")

    ids = products["product_id"]
    sales = daily_units_sold(orders, order_items, products)
    sales_total = pd.Series(sales.sum(axis=1), index=ids)
    _, init_qty = initial_stock(sales, products)
    initial_total = pd.Series(init_qty, index=ids)
    delivered = purchase_orders[purchase_orders["delivery_status"] == "Delivered"]
    receipts_total = (
        delivered.groupby("product_id")["quantity_received"].sum().astype("int64").reindex(ids).fillna(0)
    )
    defective_total = (
        delivered.groupby("product_id")["defective_units"].sum().astype("int64").reindex(ids).fillna(0)
    )
    checks = [
        ("units_demanded", sales_total, "sales"),
        ("initial_stock_units", initial_total, "the initial stock rule"),
        ("po_receipts_units", receipts_total, "purchase orders"),
        ("defective_writeoff_units", defective_total, "purchase orders"),
    ]
    for column, expected, source in checks:
        actual = inv.groupby("product_id")[column].sum().reindex(ids).fillna(0)
        if (actual.to_numpy() != expected.to_numpy()).any():
            problems.append(f"{column} do not reconcile with {source}")

    return problems