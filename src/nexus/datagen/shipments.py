"""Shipments, the operational events of each completed order (SYNTHETIC data).

Each completed order has one shipment. Dispatch is delayed when a product on the order is
in backorder on the order date. Transit time depends on the customer's region and the
carrier. The promised delivery date depends only on the region.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nexus.datagen.constants import (
    BACKORDER_EXTRA_BASE_DAYS,
    BACKORDER_EXTRA_MEAN_DAYS,
    CARRIER_COST_MULTIPLIER,
    CARRIER_LATE_PROB,
    CARRIER_PROBS,
    CARRIER_SPEED_MULTIPLIER,
    CARRIERS,
    DATA_END,
    DATA_START,
    EXTRA_LINE_COST_SHARE,
    HANDLING_DAYS,
    HANDLING_PROBS,
    PROMISE_BUFFER_DAYS,
    PROMISE_PROCESSING_DAYS,
    REGION_BASE_SHIPPING_COST,
    REGION_BASE_TRANSIT_DAYS,
    SHIPMENT_STATUSES,
    STREAM_SHIPMENTS,
    TRANSIT_JITTER,
    TRANSIT_LATE_EXTRA_MEAN,
)

SHIPMENT_COLUMNS = [
    "shipment_id",
    "order_id",
    "carrier",
    "ship_date",
    "promised_delivery_date",
    "delivered_date",
    "shipment_status",
    "had_backorder",
    "handling_days",
    "transit_days",
    "delivery_delay_days",
    "shipping_cost",
]
NULLABLE_SHIPMENT_COLUMNS = [
    "ship_date",
    "delivered_date",
    "handling_days",
    "transit_days",
    "delivery_delay_days",
]


def _rng(seed: int, stream: int) -> np.random.Generator:
    """Independent, reproducible random stream for one table."""
    return np.random.default_rng([seed, stream])


def generate_shipments(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    customers: pd.DataFrame,
    inventory: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Generate one shipment per completed order, in order-id order."""
    rng = _rng(seed, STREAM_SHIPMENTS)
    start = pd.Timestamp(DATA_START)
    end = pd.Timestamp(DATA_END)

    completed = orders.loc[orders["order_status"] == "Completed"].reset_index(drop=True)
    n = len(completed)
    region = completed["customer_id"].map(customers.set_index("customer_id")["region"]).to_numpy()

    # Backorder flag: any line product has backorders at the end of the order day.
    order_day = (completed["order_date"] - start).dt.days.to_numpy()
    lines = order_items[["order_id", "product_id"]].merge(
        completed[["order_id"]].assign(day=order_day), on="order_id"
    )
    stockouts = inventory.loc[inventory["stockout_flag"] == 1, ["product_id", "date"]].copy()
    stockouts["day"] = (stockouts["date"] - start).dt.days
    stockouts["so"] = 1
    lines = lines.merge(stockouts[["product_id", "day", "so"]], on=["product_id", "day"], how="left")
    flag = lines.groupby("order_id")["so"].max()
    had_backorder = completed["order_id"].map(flag).fillna(0).astype(np.int64).to_numpy()

    base_transit = np.array([REGION_BASE_TRANSIT_DAYS[str(r)] for r in region])
    base_cost = np.array([REGION_BASE_SHIPPING_COST[str(r)] for r in region])

    carrier = rng.choice(CARRIERS, size=n, p=CARRIER_PROBS)
    speed = np.array([CARRIER_SPEED_MULTIPLIER[str(c)] for c in carrier])
    late_prob = np.array([CARRIER_LATE_PROB[str(c)] for c in carrier])
    cost_mult = np.array([CARRIER_COST_MULTIPLIER[str(c)] for c in carrier])

    handling_base = rng.choice(HANDLING_DAYS, size=n, p=HANDLING_PROBS)
    backorder_extra = BACKORDER_EXTRA_BASE_DAYS + rng.poisson(BACKORDER_EXTRA_MEAN_DAYS, size=n)
    handling = handling_base + np.where(had_backorder == 1, backorder_extra, 0)

    jitter = rng.choice(TRANSIT_JITTER, size=n)
    disrupted = rng.random(n) < late_prob
    disruption_days = 1 + rng.poisson(TRANSIT_LATE_EXTRA_MEAN, size=n)
    core = np.rint(base_transit * speed).astype(np.int64) + jitter
    transit = np.maximum(1, core + np.where(disrupted, disruption_days, 0)).astype(np.int64)

    order_date = completed["order_date"]
    ship = order_date + pd.to_timedelta(handling, unit="D")
    delivered_dt = ship + pd.to_timedelta(transit, unit="D")
    promise_days = PROMISE_PROCESSING_DAYS + base_transit + PROMISE_BUFFER_DAYS
    promised = order_date + pd.to_timedelta(promise_days, unit="D")

    awaiting = np.asarray(ship > end)
    delivered_mask = np.asarray(delivered_dt <= end)
    status = np.where(awaiting, "Awaiting Shipment", np.where(delivered_mask, "Delivered", "In Transit"))
    shipped_s = pd.Series(~awaiting)
    delivered_s = pd.Series(delivered_mask)

    delay = (delivered_dt - promised).dt.days.to_numpy()
    n_lines = completed["order_id"].map(order_items.groupby("order_id").size()).to_numpy()
    cost = np.round(base_cost * cost_mult * (1.0 + EXTRA_LINE_COST_SHARE * (n_lines - 1)), 2)

    return pd.DataFrame(
        {
            "shipment_id": [f"SHP-{i:06d}" for i in range(1, n + 1)],
            "order_id": completed["order_id"].to_numpy(),
            "carrier": [str(c) for c in carrier],
            "ship_date": pd.Series(ship.to_numpy()).where(shipped_s),
            "promised_delivery_date": promised.to_numpy(),
            "delivered_date": pd.Series(delivered_dt.to_numpy()).where(delivered_s),
            "shipment_status": status,
            "had_backorder": had_backorder,
            "handling_days": pd.Series(handling, dtype="Int64").where(shipped_s),
            "transit_days": pd.Series(transit, dtype="Int64").where(delivered_s),
            "delivery_delay_days": pd.Series(delay, dtype="Int64").where(delivered_s),
            "shipping_cost": cost,
        }
    )


def validate_shipments(sh: pd.DataFrame, orders: pd.DataFrame) -> list[str]:
    """Return a list of problems found. An empty list means the table is valid."""
    if list(sh.columns) != SHIPMENT_COLUMNS:
        return ["unexpected columns"]
    if not sh["shipment_id"].is_unique:
        return ["shipment_id is not unique"]
    if not sh["order_id"].is_unique:
        return ["shipments contain more than one row per order"]

    order = orders.set_index("order_id")
    if not sh["order_id"].isin(order.index).all():
        return ["shipments reference unknown orders"]

    problems: list[str] = []
    status = order["order_status"]
    if (sh["order_id"].map(status) != "Completed").any():
        problems.append("shipments exist for orders that are not completed")
    completed_ids = status.index[(status == "Completed").to_numpy()]
    if not completed_ids.isin(sh["order_id"]).all():
        problems.append("completed orders without a shipment")

    if not sh["carrier"].isin(CARRIERS).all():
        problems.append("unknown carrier")
    if not sh["shipment_status"].isin(SHIPMENT_STATUSES).all():
        problems.append("unknown shipment_status value")
    if not sh["had_backorder"].isin([0, 1]).all():
        problems.append("had_backorder is not 0 or 1")
    if (sh["shipping_cost"] <= 0).any():
        problems.append("shipping_cost is not positive")

    end = pd.Timestamp(DATA_END)
    order_date = sh["order_id"].map(order["order_date"])
    if (sh["promised_delivery_date"] <= order_date).any():
        problems.append("promised_delivery_date not after order_date")

    delivered = sh["shipment_status"] == "Delivered"
    transit = sh["shipment_status"] == "In Transit"
    awaiting = sh["shipment_status"] == "Awaiting Shipment"
    delivery_fields = ["ship_date", "delivered_date", "handling_days", "transit_days", "delivery_delay_days"]
    if sh.loc[delivered, delivery_fields].isna().any().any():
        problems.append("delivered rows with missing fields")
    if sh.loc[~delivered, ["delivered_date", "transit_days", "delivery_delay_days"]].notna().any().any():
        problems.append("non-delivered rows have delivery fields")
    if sh.loc[transit, ["ship_date", "handling_days"]].isna().any().any():
        problems.append("in-transit rows missing ship fields")
    if sh.loc[awaiting, ["ship_date", "handling_days"]].notna().any().any():
        problems.append("awaiting-shipment rows have ship fields")

    shipped = sh["ship_date"].notna() & sh["handling_days"].notna()
    s = sh[shipped]
    od = order_date[shipped]
    if (s["ship_date"] < od).any():
        problems.append("ship_date before order_date")
    if (s["ship_date"] > end).any():
        problems.append("ship_date after the observation window")
    if ((s["ship_date"] - od).dt.days != s["handling_days"].astype("int64")).any():
        problems.append("handling_days does not match ship_date - order_date")

    full = delivered & sh[delivery_fields].notna().all(axis=1)
    dd = sh[full]
    if (dd["delivered_date"] > end).any():
        problems.append("delivered_date after the observation window")
    if (dd["transit_days"].astype("int64") < 1).any():
        problems.append("transit_days below 1")
    if ((dd["delivered_date"] - dd["ship_date"]).dt.days != dd["transit_days"].astype("int64")).any():
        problems.append("transit_days does not match delivered_date - ship_date")
    delay = (dd["delivered_date"] - dd["promised_delivery_date"]).dt.days
    if (delay != dd["delivery_delay_days"].astype("int64")).any():
        problems.append("delivery_delay_days does not match delivered_date - promised_delivery_date")

    other_nulls = int(sh.drop(columns=NULLABLE_SHIPMENT_COLUMNS).isna().sum().sum())
    if other_nulls:
        problems.append(f"{other_nulls} unexpected null values")

    return problems