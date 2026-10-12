"""Daily business metrics (SYNTHETIC data).

A deterministic aggregation of the other tables: no random draws. Order-basis columns are
dated by order_date. Cash-basis columns (payments, refunds) are dated by their own event
dates. Every column total reconciles to its source table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nexus.datagen.constants import DATA_END, DATA_START

DAILY_COLUMNS = [
    "date",
    "orders_placed",
    "orders_completed",
    "orders_cancelled",
    "new_customers",
    "active_customers",
    "units_sold",
    "gross_revenue",
    "discount_amount",
    "cogs",
    "gross_margin",
    "shipping_cost",
    "payments_captured",
    "payments_failed",
    "refunds_amount",
    "po_spend",
    "listed_products",
    "products_in_stockout",
    "backorder_units",
    "inventory_value",
]
COUNT_COLUMNS = [
    "orders_placed",
    "orders_completed",
    "orders_cancelled",
    "new_customers",
    "active_customers",
    "units_sold",
    "listed_products",
    "products_in_stockout",
    "backorder_units",
]
MONEY_COLUMNS = [
    "gross_revenue",
    "discount_amount",
    "cogs",
    "gross_margin",
    "shipping_cost",
    "payments_captured",
    "payments_failed",
    "refunds_amount",
    "po_spend",
    "inventory_value",
]


def _daily(series: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Align a date-indexed series to every day of the window, filling gaps with 0."""
    return series.reindex(dates, fill_value=0).to_numpy()


def build_daily_metrics(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    payments: pd.DataFrame,
    refunds: pd.DataFrame,
    shipments: pd.DataFrame,
    purchase_orders: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per calendar day of the observation window."""
    dates = pd.date_range(DATA_START, DATA_END, freq="D")

    completed = orders.loc[orders["order_status"] == "Completed"]
    cancelled = orders.loc[orders["order_status"] != "Completed"]

    lines = order_items.merge(completed[["order_id", "order_date"]], on="order_id")
    lines = lines.assign(
        list_value=lines["quantity"] * lines["unit_price"],
        cost_value=lines["quantity"] * lines["unit_cost"],
    )
    by_day = lines.groupby("order_date")
    revenue = by_day["line_total"].sum()
    list_value = by_day["list_value"].sum()

    shipping = (
        shipments[["order_id", "shipping_cost"]]
        .merge(orders[["order_id", "order_date"]], on="order_id")
        .groupby("order_date")["shipping_cost"]
        .sum()
    )
    captured = payments.loc[payments["payment_status"] == "Captured"].groupby("payment_date")["amount"].sum()
    failed = payments.loc[payments["payment_status"] == "Failed"].groupby("payment_date")["amount"].sum()
    refunded = refunds.groupby("refund_date")["refund_amount"].sum()
    po_spend = purchase_orders.groupby("order_date")["order_cost"].sum()
    inv = inventory.groupby("date")

    gross_revenue = _daily(revenue, dates)
    cogs = _daily(by_day["cost_value"].sum(), dates)

    frame = pd.DataFrame(
        {
            "date": dates,
            "orders_placed": _daily(orders.groupby("order_date").size(), dates),
            "orders_completed": _daily(completed.groupby("order_date").size(), dates),
            "orders_cancelled": _daily(cancelled.groupby("order_date").size(), dates),
            "new_customers": _daily(customers.groupby("signup_date").size(), dates),
            "active_customers": _daily(completed.groupby("order_date")["customer_id"].nunique(), dates),
            "units_sold": _daily(by_day["quantity"].sum(), dates),
            "gross_revenue": gross_revenue,
            "discount_amount": _daily(list_value, dates) - gross_revenue,
            "cogs": cogs,
            "gross_margin": gross_revenue - cogs,
            "shipping_cost": _daily(shipping, dates),
            "payments_captured": _daily(captured, dates),
            "payments_failed": _daily(failed, dates),
            "refunds_amount": _daily(refunded, dates),
            "po_spend": _daily(po_spend, dates),
            "listed_products": _daily(inv.size(), dates),
            "products_in_stockout": _daily(inv["stockout_flag"].sum(), dates),
            "backorder_units": _daily(inv["backorder_units"].sum(), dates),
            "inventory_value": _daily(inv["inventory_value"].sum(), dates),
        }
    )
    frame[COUNT_COLUMNS] = frame[COUNT_COLUMNS].astype(np.int64)
    frame[MONEY_COLUMNS] = frame[MONEY_COLUMNS].astype(float).round(2)
    return frame


def validate_daily_metrics(
    dm: pd.DataFrame,
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    payments: pd.DataFrame,
    refunds: pd.DataFrame,
    shipments: pd.DataFrame,
    purchase_orders: pd.DataFrame,
    inventory: pd.DataFrame,
) -> list[str]:
    """Return a list of problems found. An empty list means the table is valid."""
    if list(dm.columns) != DAILY_COLUMNS:
        return ["unexpected columns"]
    if dm["date"].duplicated().any():
        return ["duplicate dates"]

    problems: list[str] = []
    expected_dates = pd.date_range(DATA_START, DATA_END, freq="D")
    if len(dm) != len(expected_dates) or not (
        dm["date"].sort_values().to_numpy() == expected_dates.to_numpy()
    ).all():
        problems.append("dates are not the complete observation window")

    null_count = int(dm.isna().sum().sum())
    if null_count:
        problems.append(f"{null_count} unexpected null values")
    numeric = dm.drop(columns="date")
    if (numeric < -0.01).any().any():
        problems.append("negative values")

    if (dm["orders_placed"] != dm["orders_completed"] + dm["orders_cancelled"]).any():
        problems.append("orders_placed differs from orders_completed + orders_cancelled")
    if ((dm["gross_margin"] - (dm["gross_revenue"] - dm["cogs"])).abs() > 0.02).any():
        problems.append("gross_margin differs from gross_revenue - cogs")
    if (dm["active_customers"] > dm["orders_completed"]).any():
        problems.append("active_customers exceeds orders_completed")
    if (dm["products_in_stockout"] > dm["listed_products"]).any():
        problems.append("products_in_stockout exceeds listed_products")

    completed = orders.loc[orders["order_status"] == "Completed", ["order_id"]]
    units = order_items.merge(completed, on="order_id")["quantity"].sum()
    new_customers = customers["signup_date"].between(pd.Timestamp(DATA_START), pd.Timestamp(DATA_END)).sum()
    payment_total = payments["amount"].sum()
    checks = [
        ("orders_placed", dm["orders_placed"].sum(), len(orders)),
        ("orders_completed", dm["orders_completed"].sum(), len(completed)),
        ("new_customers", dm["new_customers"].sum(), new_customers),
        ("units_sold", dm["units_sold"].sum(), units),
        ("gross_revenue", dm["gross_revenue"].sum(), payment_total),
        ("payments_captured + payments_failed", dm["payments_captured"].sum() + dm["payments_failed"].sum(), payment_total),
        ("payments_captured", dm["payments_captured"].sum(), payments.loc[payments["payment_status"] == "Captured", "amount"].sum()),
        ("refunds_amount", dm["refunds_amount"].sum(), refunds["refund_amount"].sum()),
        ("shipping_cost", dm["shipping_cost"].sum(), shipments["shipping_cost"].sum()),
        ("po_spend", dm["po_spend"].sum(), purchase_orders["order_cost"].sum()),
        ("listed_products", dm["listed_products"].sum(), len(inventory)),
        ("products_in_stockout", dm["products_in_stockout"].sum(), inventory["stockout_flag"].sum()),
        ("backorder_units", dm["backorder_units"].sum(), inventory["backorder_units"].sum()),
        ("inventory_value", dm["inventory_value"].sum(), inventory["inventory_value"].sum()),
    ]
    for name, actual, expected in checks:
        if abs(float(actual) - float(expected)) > 0.05:
            problems.append(f"{name} do not reconcile with the source tables")

    return problems