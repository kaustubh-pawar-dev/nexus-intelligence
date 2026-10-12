"""Purchase orders and supplier deliveries (SYNTHETIC data).

Replenishment is stock-aware. At each order date the buyer targets the demand over the
planned lead time plus PO_REVIEW_DAYS days, times a safety buffer and a forecast error,
and orders the difference to the inventory position (everything received or scheduled,
minus past sales). Months where the position already covers the target get no order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nexus.datagen.constants import (
    DATA_END,
    DATA_START,
    DELIVERY_STATUSES,
    EARLY_DELIVERY_MAX_DAYS,
    INACTIVE_PRODUCT_CUTOFF_DAYS,
    INITIAL_STOCK_BUFFER,
    INITIAL_STOCK_COVER_DAYS,
    LATE_DELAY_EXTRA_MEAN,
    MIN_PO_QTY,
    ON_TIME_PROB_MAX,
    ON_TIME_PROB_MIN,
    PEAK_SEASON_MONTHS,
    PEAK_SEASON_ON_TIME_PENALTY,
    PO_FORECAST_NOISE_SIGMA,
    PO_REVIEW_DAYS,
    PO_SAFETY_BUFFER,
    SHORT_SHIPMENT_PROB,
    SHORT_SHIPMENT_RANGE,
    STREAM_PURCHASE_ORDERS,
)

NULLABLE_DELIVERY_COLUMNS = ["actual_delivery_date", "quantity_received", "defective_units"]
# Extra days of padded demand so that horizons running past the window end can be sized.
_DEMAND_TAIL_DAYS = 90


def _rng(seed: int, stream: int) -> np.random.Generator:
    """Independent, reproducible random stream for one table."""
    return np.random.default_rng([seed, stream])


def _month_starts() -> pd.DatetimeIndex:
    return pd.date_range(DATA_START, DATA_END, freq="MS")


def _n_days() -> int:
    return (DATA_END - DATA_START).days + 1


def daily_units_sold(
    orders: pd.DataFrame, order_items: pd.DataFrame, products: pd.DataFrame
) -> np.ndarray:
    """Units sold per product (rows, in products order) and day (columns).

    Only completed orders count as demand. Day 0 is DATA_START.
    """
    completed = orders.loc[orders["order_status"] == "Completed", ["order_id", "order_date"]]
    lines = order_items[["order_id", "product_id", "quantity"]].merge(completed, on="order_id")
    day_idx = (lines["order_date"] - pd.Timestamp(DATA_START)).dt.days.to_numpy()
    prod_pos = pd.Index(products["product_id"]).get_indexer(lines["product_id"])
    units = np.zeros((len(products), _n_days()), dtype=np.int64)
    np.add.at(units, (prod_pos, day_idx), lines["quantity"].to_numpy())
    return units


def monthly_units_sold(
    orders: pd.DataFrame, order_items: pd.DataFrame, products: pd.DataFrame
) -> np.ndarray:
    """Units sold per product (rows, in products order) and month (columns).

    Only completed orders count as demand.
    """
    months = _month_starts()
    completed = orders.loc[orders["order_status"] == "Completed", ["order_id", "order_date"]]
    lines = order_items[["order_id", "product_id", "quantity"]].merge(completed, on="order_id")
    month_idx = (
        (lines["order_date"].dt.year - DATA_START.year) * 12
        + lines["order_date"].dt.month
        - DATA_START.month
    ).to_numpy()
    prod_pos = pd.Index(products["product_id"]).get_indexer(lines["product_id"])
    units = np.zeros((len(products), len(months)), dtype=np.int64)
    np.add.at(units, (prod_pos, month_idx), lines["quantity"].to_numpy())
    return units


def initial_stock(sales: np.ndarray, products: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Day index and quantity of each product's initial stock.

    The stock arrives on the product's first listed day (day 0 for products launched before
    the window) and covers INITIAL_STOCK_COVER_DAYS days of demand times INITIAL_STOCK_BUFFER,
    with a minimum of MIN_PO_QTY units.
    """
    n_days = sales.shape[1]
    launch_idx = (products["launch_date"] - pd.Timestamp(DATA_START)).dt.days.to_numpy()
    start = np.clip(launch_idx, 0, n_days - 1).astype(np.int64)
    qty = np.zeros(len(products), dtype=np.int64)
    for i, s in enumerate(start):
        window = sales[i, s : s + INITIAL_STOCK_COVER_DAYS]
        qty[i] = max(MIN_PO_QTY, int(np.ceil(window.sum() * INITIAL_STOCK_BUFFER)))
    return start, qty


def generate_purchase_orders(
    products: pd.DataFrame,
    suppliers: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Generate stock-aware replenishment purchase orders.

    Candidate orders are one per product and month when the product has launched and has
    demand in the following month. Delivery timing follows the supplier's lead time and
    on-time rate, and is drawn before sizing so it does not depend on quantities. Quantities
    are then set product by product in date order from the inventory position. Deliveries
    that would land after the end of the window are marked 'In Transit' with empty delivery
    fields.
    """
    rng = _rng(seed, STREAM_PURCHASE_ORDERS)
    months = _month_starts()
    n_days = _n_days()
    start = pd.Timestamp(DATA_START)

    sales = daily_units_sold(orders, order_items, products)
    units = monthly_units_sold(orders, order_items, products)
    next_units = units.copy()
    next_units[:, :-1] = units[:, 1:]  # demand of the following month; the last month reuses its own

    days_in_month = months.days_in_month.to_numpy()
    month_start_day = (months - start).days.to_numpy()
    month_end = (months + pd.to_timedelta(days_in_month - 1, unit="D")).to_numpy()
    month_end = month_end.astype("datetime64[ns]")
    launch = products["launch_date"].to_numpy().astype("datetime64[ns]")
    eligible = (launch[:, None] <= month_end[None, :]) & (next_units > 0)

    p_idx, m_idx = np.nonzero(eligible)  # grouped by product, months ascending
    n = len(p_idx)
    if n == 0:
        raise ValueError("No purchase orders were generated; check the configuration")

    # Timing draws (same order as before, independent of quantities).
    noise = rng.lognormal(0.0, PO_FORECAST_NOISE_SIGMA, size=n)
    offset = np.floor(rng.random(n) * days_in_month[m_idx]).astype(np.int64)
    order_day = month_start_day[m_idx] + offset
    order_date = months[m_idx] + pd.to_timedelta(offset, unit="D")

    # Inactive products are not replenished in the last N days of the window.
    cutoff = pd.Timestamp(DATA_END) - pd.Timedelta(days=INACTIVE_PRODUCT_CUTOFF_DAYS)
    is_active = products["is_active"].to_numpy().astype(bool)
    drop = (~is_active[p_idx]) & np.asarray(order_date > cutoff)

    sup = suppliers.set_index("supplier_id")
    supplier_id = products["supplier_id"].to_numpy()[p_idx]
    planned_lead = sup["lead_time_mean_days"].reindex(supplier_id).to_numpy().astype(np.int64)
    on_time_rate = sup["on_time_delivery_rate"].reindex(supplier_id).to_numpy(dtype=float)
    defect_rate = sup["defect_rate"].reindex(supplier_id).to_numpy(dtype=float)

    peak = np.isin(months.month.to_numpy()[m_idx], PEAK_SEASON_MONTHS)
    on_time_p = np.clip(
        on_time_rate - np.where(peak, PEAK_SEASON_ON_TIME_PENALTY, 0.0),
        ON_TIME_PROB_MIN,
        ON_TIME_PROB_MAX,
    )
    is_late = rng.random(n) >= on_time_p
    early = rng.integers(0, EARLY_DELIVERY_MAX_DAYS + 1, size=n)
    late = 1 + rng.poisson(LATE_DELAY_EXTRA_MEAN, size=n)
    delay = np.where(is_late, late, -early).astype(np.int64)

    arrival_day = order_day + planned_lead + delay
    expected = order_date + pd.to_timedelta(planned_lead, unit="D")
    actual = expected + pd.to_timedelta(delay, unit="D")
    delivered = arrival_day <= n_days - 1

    short = rng.random(n) < SHORT_SHIPMENT_PROB
    fraction = rng.uniform(SHORT_SHIPMENT_RANGE[0], SHORT_SHIPMENT_RANGE[1], size=n)

    # Sequential sizing from the inventory position, product by product.
    start_idx, init_qty = initial_stock(sales, products)
    keep = ~drop
    quantity = np.zeros(n, dtype=np.int64)
    received = np.zeros(n, dtype=np.int64)
    defective = np.zeros(n, dtype=np.int64)

    for p in np.unique(p_idx):
        rows = np.flatnonzero((p_idx == p) & keep)
        if len(rows) == 0:
            continue
        tail_rate = float(sales[p, -30:].mean())
        padded = np.concatenate([sales[p], np.full(_DEMAND_TAIL_DAYS, tail_rate)])
        cum_sales = np.concatenate([[0.0], np.cumsum(padded)])
        scheduled = float(init_qty[p])  # everything received or scheduled so far
        for r in rows:
            t = int(order_day[r])
            end = min(t + int(planned_lead[r]) + PO_REVIEW_DAYS, len(padded))
            demand = cum_sales[end] - cum_sales[t]
            position = scheduled - cum_sales[t]  # received or scheduled minus past sales
            target = demand * PO_SAFETY_BUFFER * noise[r]
            qty = int(np.ceil(target - position))
            if qty <= 0:
                continue
            qty = max(MIN_PO_QTY, qty)
            short_units = int(np.floor(qty * fraction[r])) if short[r] else 0
            quantity[r] = qty
            received[r] = qty - short_units
            defective[r] = rng.binomial(received[r], defect_rate[r])
            scheduled += received[r] - defective[r]  # defective units are written off on arrival

    valid = keep & (quantity > 0)

    unit_cost = products["unit_cost"].to_numpy()[p_idx]
    delivered_s = pd.Series(delivered)

    frame = pd.DataFrame(
        {
            "supplier_id": supplier_id,
            "product_id": products["product_id"].to_numpy()[p_idx],
            "order_date": order_date.to_numpy(),
            "expected_delivery_date": expected.to_numpy(),
            "actual_delivery_date": pd.Series(actual.to_numpy()).where(delivered_s),
            "delivery_status": np.where(delivered, "Delivered", "In Transit"),
            "quantity_ordered": quantity,
            "quantity_received": pd.Series(received, dtype="Int64").where(delivered_s),
            "defective_units": pd.Series(defective, dtype="Int64").where(delivered_s),
            "unit_cost": unit_cost,
            "order_cost": np.round(quantity * unit_cost, 2),
        }
    )
    frame = frame.loc[valid]
    frame = frame.sort_values(["order_date", "product_id"], kind="stable").reset_index(drop=True)
    frame.insert(0, "po_id", [f"PO-{i:06d}" for i in range(1, len(frame) + 1)])
    return frame


def validate_purchase_orders(
    po: pd.DataFrame, products: pd.DataFrame, suppliers: pd.DataFrame
) -> list[str]:
    """Return a list of problems found. An empty list means the table is valid."""
    problems: list[str] = []

    if not po["po_id"].is_unique:
        return ["po_id is not unique"]

    if not po["supplier_id"].isin(suppliers["supplier_id"]).all():
        problems.append("purchase orders reference unknown suppliers")

    prod = products.set_index("product_id")
    if not po["product_id"].isin(prod.index).all():
        problems.append("purchase orders reference unknown products")
    else:
        if (po["supplier_id"] != po["product_id"].map(prod["supplier_id"])).any():
            problems.append("purchase order supplier differs from the product's supplier")
        if not np.isclose(po["unit_cost"], po["product_id"].map(prod["unit_cost"])).all():
            problems.append("unit_cost differs from product unit_cost")
        launch = po["product_id"].map(prod["launch_date"])
        if (po["order_date"] + pd.Timedelta(days=31) < launch).any():
            problems.append("purchase order placed more than a month before product launch")

    if not po["order_date"].between(pd.Timestamp(DATA_START), pd.Timestamp(DATA_END)).all():
        problems.append("order_date outside the observation window")
    if (po["expected_delivery_date"] <= po["order_date"]).any():
        problems.append("expected_delivery_date not after order_date")
    if not po["delivery_status"].isin(DELIVERY_STATUSES).all():
        problems.append("unknown delivery_status value")
    if (po["quantity_ordered"] < MIN_PO_QTY).any():
        problems.append("quantity_ordered below the minimum")
    if (po["order_cost"] - (po["quantity_ordered"] * po["unit_cost"]).round(2)).abs().max() > 0.005:
        problems.append("order_cost does not match quantity_ordered * unit_cost")

    delivered = po["delivery_status"] == "Delivered"
    d = po[delivered]
    t = po[~delivered]
    if d[NULLABLE_DELIVERY_COLUMNS].isna().any().any():
        problems.append("delivered rows with missing delivery fields")
    else:
        received = d["quantity_received"].astype("int64")
        defective = d["defective_units"].astype("int64")
        if ((received < 1) | (received > d["quantity_ordered"])).any():
            problems.append("quantity_received outside 1..quantity_ordered")
        if ((defective < 0) | (defective > received)).any():
            problems.append("defective_units outside 0..quantity_received")
        if (d["actual_delivery_date"] > pd.Timestamp(DATA_END)).any():
            problems.append("delivered rows with actual_delivery_date after the window")
        if (d["actual_delivery_date"] <= d["order_date"]).any():
            problems.append("actual_delivery_date not after order_date")
    if t[NULLABLE_DELIVERY_COLUMNS].notna().any().any():
        problems.append("in-transit rows have delivery fields")

    other_nulls = int(po.drop(columns=NULLABLE_DELIVERY_COLUMNS).isna().sum().sum())
    if other_nulls:
        problems.append(f"{other_nulls} unexpected null values")

    return problems