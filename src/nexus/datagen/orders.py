"""Sales orders and order lines (SYNTHETIC data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nexus.datagen.constants import (
    CHANNEL_LIFETIME_MULTIPLIER,
    CHANNEL_RATE_MULTIPLIER,
    CUSTOMER_RATE_GAMMA_SHAPE,
    DATA_END,
    DATA_START,
    DISCOUNT_LEVELS,
    DISCOUNT_PROB_BASE,
    DISCOUNT_PROB_NOVEMBER,
    DISCOUNT_PROBS,
    INACTIVE_PRODUCT_CUTOFF_DAYS,
    MAX_ITEMS_PER_ORDER,
    MAX_QUANTITY,
    MONTH_SEASONALITY,
    OPT_IN_RATE_MULTIPLIER,
    ORDER_CANCEL_RATE,
    ORDER_STATUSES,
    PRODUCT_POPULARITY_SIGMA,
    SALES_CHANNEL_PROBS,
    SALES_CHANNELS,
    SEGMENT_ITEMS_BASE,
    SEGMENT_ITEMS_EXTRA_MEAN,
    SEGMENT_MEAN_LIFETIME_DAYS,
    SEGMENT_MONTHLY_ORDER_RATE,
    SEGMENT_QTY_BASE,
    SEGMENT_QTY_EXTRA_MEAN,
    STREAM_ORDER_ITEMS,
    STREAM_ORDERS,
    WEEKDAY_FACTOR,
)

DAYS_PER_MONTH = 30.44


def _rng(seed: int, stream: int) -> np.random.Generator:
    """Independent, reproducible random stream for one table."""
    return np.random.default_rng([seed, stream])


def _n_days() -> int:
    return (DATA_END - DATA_START).days + 1


def _cumulative_demand() -> np.ndarray:
    """Cumulative demand weights over the window (length n_days + 1, starts at 0).

    The daily weight is month seasonality times weekday factor, scaled to mean 1.
    """
    dates = pd.date_range(DATA_START, DATA_END, freq="D")
    month_factor = np.array(MONTH_SEASONALITY)[dates.month.to_numpy() - 1]
    weekday_factor = np.array(WEEKDAY_FACTOR)[dates.dayofweek.to_numpy()]
    weights = month_factor * weekday_factor
    weights = weights / weights.mean()
    return np.concatenate([[0.0], np.cumsum(weights)])


def generate_orders(customers: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Generate the orders table (one row per order, sorted by date).

    Each customer is active from max(signup, window start) until signup plus an
    exponential lifetime (their churn point). Order counts are Poisson with a rate
    that depends on segment, acquisition channel, marketing opt-in and noise.
    """
    rng = _rng(seed, STREAM_ORDERS)
    n_days = _n_days()
    cum = _cumulative_demand()
    n_cust = len(customers)

    segment = customers["segment"].to_numpy()
    channel = customers["acquisition_channel"].to_numpy()
    opt_in = customers["marketing_opt_in"].to_numpy().astype(bool)
    signup_idx = (customers["signup_date"] - pd.Timestamp(DATA_START)).dt.days.to_numpy()

    seg_rate = np.array([SEGMENT_MONTHLY_ORDER_RATE[s] for s in segment])
    chan_rate = np.array([CHANNEL_RATE_MULTIPLIER[c] for c in channel])
    seg_life = np.array([SEGMENT_MEAN_LIFETIME_DAYS[s] for s in segment], dtype=float)
    chan_life = np.array([CHANNEL_LIFETIME_MULTIPLIER[c] for c in channel])

    noise = rng.gamma(
        shape=CUSTOMER_RATE_GAMMA_SHAPE, scale=1.0 / CUSTOMER_RATE_GAMMA_SHAPE, size=n_cust
    )
    rate_per_day = (
        seg_rate / DAYS_PER_MONTH
        * chan_rate
        * np.where(opt_in, OPT_IN_RATE_MULTIPLIER, 1.0)
        * noise
    )
    lifetime = rng.exponential(seg_life * chan_life)

    first_day = np.clip(signup_idx, 0, n_days - 1).astype(np.int64)
    last_day = np.clip(np.floor(signup_idx + lifetime), -1, n_days - 1).astype(np.int64)
    active = last_day >= first_day
    expected = np.where(active, rate_per_day * (cum[last_day + 1] - cum[first_day]), 0.0)

    counts = rng.poisson(expected)
    total = int(counts.sum())
    if total == 0:
        raise ValueError("No orders were generated; check the configuration")

    cust_idx = np.repeat(np.arange(n_cust), counts)
    lo = cum[first_day][cust_idx]
    hi = cum[last_day + 1][cust_idx]
    u = lo + rng.random(total) * (hi - lo)
    day = np.searchsorted(cum, u, side="right") - 1
    day = np.clip(day, first_day[cust_idx], last_day[cust_idx])

    order = np.lexsort((cust_idx, day))
    cust_idx = cust_idx[order]
    day = day[order]
    order_dates = pd.Timestamp(DATA_START) + pd.to_timedelta(day, unit="D")
    month = order_dates.month.to_numpy()

    cancelled = rng.random(total) < ORDER_CANCEL_RATE
    sales_channel = rng.choice(SALES_CHANNELS, size=total, p=SALES_CHANNEL_PROBS)
    discount_prob = np.where(month == 11, DISCOUNT_PROB_NOVEMBER, DISCOUNT_PROB_BASE)
    has_discount = rng.random(total) < discount_prob
    level = rng.choice(DISCOUNT_LEVELS, size=total, p=DISCOUNT_PROBS)
    discount = np.where(has_discount, level, 0.0)

    return pd.DataFrame(
        {
            "order_id": [f"ORD-{i:06d}" for i in range(1, total + 1)],
            "customer_id": customers["customer_id"].to_numpy()[cust_idx],
            "order_date": order_dates,
            "sales_channel": [str(c) for c in sales_channel],
            "order_status": np.where(cancelled, "Cancelled", "Completed"),
            "discount_pct": discount,
        }
    )


def generate_order_items(
    orders: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """Generate order lines. Products in one order are distinct.

    A product can only be sold on or after its launch date. Inactive products are
    not sold during the last INACTIVE_PRODUCT_CUTOFF_DAYS days of the window.
    Prices are the product list price (constant over time) and unit_cost is the
    cost of goods sold snapshot at the time of sale.
    """
    rng = _rng(seed, STREAM_ORDER_ITEMS)
    n_orders = len(orders)
    n_days = _n_days()

    seg = orders["customer_id"].map(customers.set_index("customer_id")["segment"]).to_numpy()
    popularity = rng.lognormal(0.0, PRODUCT_POPULARITY_SIGMA, size=len(products))

    items_base = np.array([SEGMENT_ITEMS_BASE[s] for s in seg])
    items_extra = np.array([SEGMENT_ITEMS_EXTRA_MEAN[s] for s in seg])
    k_items = np.minimum(MAX_ITEMS_PER_ORDER, items_base + rng.poisson(items_extra))

    start = pd.Timestamp(DATA_START)
    order_idx = (orders["order_date"] - start).dt.days.to_numpy()
    launch_idx = (products["launch_date"] - start).dt.days.to_numpy()
    is_active = products["is_active"].to_numpy().astype(bool)
    cutoff_idx = (n_days - 1) - INACTIVE_PRODUCT_CUTOFF_DAYS

    line_order: list[int] = []
    line_prod: list[int] = []
    line_no: list[int] = []

    current_day = None
    avail = np.array([], dtype=np.int64)
    probs = np.array([])
    for i in range(n_orders):
        d = int(order_idx[i])
        if d != current_day:
            mask = launch_idx <= d
            if d > cutoff_idx:
                mask = mask & is_active
            avail = np.flatnonzero(mask)
            weights = popularity[avail]
            probs = weights / weights.sum()
            current_day = d
        k = min(int(k_items[i]), len(avail))
        chosen = rng.choice(avail, size=k, replace=False, p=probs)
        for j, prod in enumerate(chosen, start=1):
            line_order.append(i)
            line_prod.append(int(prod))
            line_no.append(j)

    line_order_arr = np.array(line_order, dtype=np.int64)
    line_prod_arr = np.array(line_prod, dtype=np.int64)

    line_seg = seg[line_order_arr]
    qty_base = np.array([SEGMENT_QTY_BASE[s] for s in line_seg])
    qty_extra = np.array([SEGMENT_QTY_EXTRA_MEAN[s] for s in line_seg])
    quantity = np.minimum(MAX_QUANTITY, qty_base + rng.poisson(qty_extra))

    unit_price = products["list_price"].to_numpy()[line_prod_arr]
    unit_cost = products["unit_cost"].to_numpy()[line_prod_arr]
    discount = orders["discount_pct"].to_numpy()[line_order_arr]
    line_total = np.round(quantity * unit_price * (1.0 - discount), 2)

    return pd.DataFrame(
        {
            "order_id": orders["order_id"].to_numpy()[line_order_arr],
            "line_number": np.array(line_no, dtype=np.int64),
            "product_id": products["product_id"].to_numpy()[line_prod_arr],
            "quantity": quantity.astype(np.int64),
            "unit_price": unit_price,
            "unit_cost": unit_cost,
            "discount_pct": discount,
            "line_total": line_total,
        }
    )


def validate_orders(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
) -> list[str]:
    """Return a list of problems found. An empty list means the tables are valid."""
    problems: list[str] = []

    if not orders["order_id"].is_unique:
        problems.append("order_id is not unique")
        return problems

    if not orders["customer_id"].isin(customers["customer_id"]).all():
        problems.append("orders reference unknown customers")

    in_window = orders["order_date"].between(pd.Timestamp(DATA_START), pd.Timestamp(DATA_END))
    if not in_window.all():
        problems.append("order_date outside the observation window")

    signup = orders["customer_id"].map(customers.set_index("customer_id")["signup_date"])
    if (orders["order_date"] < signup).any():
        problems.append("orders placed before customer signup")

    if not orders["order_status"].isin(ORDER_STATUSES).all():
        problems.append("unknown order_status value")
    if not orders["discount_pct"].isin([0.0, *DISCOUNT_LEVELS]).all():
        problems.append("unexpected discount_pct value")

    if not order_items["order_id"].isin(orders["order_id"]).all():
        problems.append("order_items reference unknown orders")
    if order_items.duplicated(["order_id", "line_number"]).any():
        problems.append("duplicate (order_id, line_number)")
    if order_items.duplicated(["order_id", "product_id"]).any():
        problems.append("duplicate product within an order")
    if not order_items["product_id"].isin(products["product_id"]).all():
        problems.append("order_items reference unknown products")
    if not order_items["quantity"].between(1, MAX_QUANTITY).all():
        problems.append("quantity outside allowed range")
    if not orders["order_id"].isin(order_items["order_id"]).all():
        problems.append("orders without items")

    merged = order_items.merge(
        products[["product_id", "list_price", "unit_cost", "launch_date"]],
        on="product_id",
        how="left",
        suffixes=("", "_product"),
    )
    if not np.isclose(merged["unit_price"], merged["list_price"]).all():
        problems.append("unit_price differs from product list_price")
    if not np.isclose(merged["unit_cost"], merged["unit_cost_product"]).all():
        problems.append("unit_cost differs from product unit_cost")

    order_date = merged["order_id"].map(orders.set_index("order_id")["order_date"])
    if (order_date < merged["launch_date"]).any():
        problems.append("item sold before product launch")

    order_discount = merged["order_id"].map(orders.set_index("order_id")["discount_pct"])
    if not np.isclose(merged["discount_pct"], order_discount).all():
        problems.append("item discount differs from order discount")

    expected_total = np.round(
        merged["quantity"] * merged["unit_price"] * (1.0 - merged["discount_pct"]), 2
    )
    if ((expected_total - merged["line_total"]).abs() > 0.005).any():
        problems.append("line_total does not match quantity * unit_price * (1 - discount)")

    null_count = int(orders.isna().sum().sum() + order_items.isna().sum().sum())
    if null_count:
        problems.append(f"{null_count} unexpected null values")

    return problems