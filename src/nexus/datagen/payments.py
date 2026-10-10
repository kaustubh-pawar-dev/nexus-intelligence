"""Payments and refunds (SYNTHETIC data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nexus.datagen.constants import (
    BANK_TRANSFER_EXTRA_DELAY_MAX,
    CUSTOMER_SEGMENTS,
    DATA_END,
    DEFECTIVE_REASON,
    DEFECTIVE_REASON_BASE,
    DEFECTIVE_REASON_MAX,
    DEFECTIVE_REASON_SLOPE,
    OTHER_REFUND_REASON_PROBS,
    OTHER_REFUND_REASONS,
    PAYMENT_DELAY_DAYS,
    PAYMENT_DELAY_PROBS,
    PAYMENT_FAILURE_RATE,
    PAYMENT_METHODS,
    PAYMENT_STATUSES,
    REFUND_BASE_PROB,
    REFUND_DEFECT_SENSITIVITY,
    REFUND_FULL_SHARE,
    REFUND_MAX_DAYS,
    REFUND_MIN_DAYS,
    REFUND_PARTIAL_RANGE,
    REFUND_REASONS,
    REFUND_TYPES,
    SEGMENT_PAYMENT_METHOD_PROBS,
    STREAM_PAYMENTS,
    STREAM_REFUNDS,
)


def _rng(seed: int, stream: int) -> np.random.Generator:
    """Independent, reproducible random stream for one table."""
    return np.random.default_rng([seed, stream])


def generate_payments(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    customers: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Generate one payment per completed order (cancelled orders get none).

    The amount is the sum of the order's line totals. Payment method depends on
    the customer segment, and the failure probability depends on the method.
    Payment dates are clipped at the end of the observation window.
    """
    rng = _rng(seed, STREAM_PAYMENTS)

    completed = orders.loc[orders["order_status"] == "Completed"].reset_index(drop=True)
    n = len(completed)

    totals = order_items.groupby("order_id")["line_total"].sum()
    amount = completed["order_id"].map(totals).round(2).to_numpy()

    segment = (
        completed["customer_id"].map(customers.set_index("customer_id")["segment"]).to_numpy()
    )
    methods = np.empty(n, dtype=object)
    for seg in CUSTOMER_SEGMENTS:
        idx = np.flatnonzero(segment == seg)
        methods[idx] = rng.choice(PAYMENT_METHODS, size=len(idx), p=SEGMENT_PAYMENT_METHOD_PROBS[seg])

    failure_p = np.array([PAYMENT_FAILURE_RATE[str(m)] for m in methods])
    failed = rng.random(n) < failure_p

    delay = rng.choice(PAYMENT_DELAY_DAYS, size=n, p=PAYMENT_DELAY_PROBS)
    extra = rng.integers(1, BANK_TRANSFER_EXTRA_DELAY_MAX + 1, size=n)
    is_bank = np.array([str(m) == "Bank Transfer" for m in methods])
    delay = delay + np.where(is_bank, extra, 0)

    payment_date = completed["order_date"] + pd.to_timedelta(delay, unit="D")
    payment_date = payment_date.clip(upper=pd.Timestamp(DATA_END))

    return pd.DataFrame(
        {
            "payment_id": [f"PAY-{i:06d}" for i in range(1, n + 1)],
            "order_id": completed["order_id"].to_numpy(),
            "payment_date": payment_date.to_numpy(),
            "payment_method": [str(m) for m in methods],
            "payment_status": np.where(failed, "Failed", "Captured"),
            "amount": amount,
        }
    )


def generate_refunds(
    payments: pd.DataFrame,
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    suppliers: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Generate refunds for captured payments (at most one refund per payment).

    The refund probability rises with the order's defect score, the line-value-weighted
    defect rate of the suppliers behind the items. Refunds that would be dated after the
    end of the observation window are not generated (right-censoring).
    """
    rng = _rng(seed, STREAM_REFUNDS)

    defect_by_product = products[["product_id", "supplier_id"]].merge(
        suppliers[["supplier_id", "defect_rate"]], on="supplier_id", how="left"
    )
    lines = order_items[["order_id", "product_id", "line_total"]].merge(
        defect_by_product[["product_id", "defect_rate"]], on="product_id", how="left"
    )
    lines["weighted"] = lines["line_total"] * lines["defect_rate"]
    grouped = lines.groupby("order_id")
    score_by_order = grouped["weighted"].sum() / grouped["line_total"].sum()

    captured = payments.loc[payments["payment_status"] == "Captured"].reset_index(drop=True)
    n = len(captured)
    defect_score = captured["order_id"].map(score_by_order).to_numpy()

    probability = REFUND_BASE_PROB + REFUND_DEFECT_SENSITIVITY * defect_score
    refunded = rng.random(n) < probability

    selected = captured.loc[refunded].reset_index(drop=True)
    score = defect_score[refunded]
    m = len(selected)

    is_full = rng.random(m) < REFUND_FULL_SHARE
    fraction = rng.uniform(REFUND_PARTIAL_RANGE[0], REFUND_PARTIAL_RANGE[1], size=m)
    pay_amount = selected["amount"].to_numpy()
    refund_amount = np.where(is_full, pay_amount, np.round(pay_amount * fraction, 2))

    days = rng.integers(REFUND_MIN_DAYS, REFUND_MAX_DAYS + 1, size=m)
    refund_date = selected["payment_date"] + pd.to_timedelta(days, unit="D")

    p_defective = np.minimum(
        DEFECTIVE_REASON_MAX, DEFECTIVE_REASON_BASE + DEFECTIVE_REASON_SLOPE * score
    )
    is_defective = rng.random(m) < p_defective
    other = rng.choice(OTHER_REFUND_REASONS, size=m, p=OTHER_REFUND_REASON_PROBS)
    reason = np.where(is_defective, DEFECTIVE_REASON, other)

    frame = pd.DataFrame(
        {
            "payment_id": selected["payment_id"].to_numpy(),
            "order_id": selected["order_id"].to_numpy(),
            "refund_date": refund_date.to_numpy(),
            "refund_type": np.where(is_full, "Full", "Partial"),
            "refund_reason": [str(r) for r in reason],
            "refund_amount": refund_amount,
        }
    )
    frame = frame[frame["refund_date"] <= pd.Timestamp(DATA_END)]
    frame = frame.sort_values(["refund_date", "payment_id"], kind="stable").reset_index(drop=True)
    frame.insert(0, "refund_id", [f"REF-{i:06d}" for i in range(1, len(frame) + 1)])
    return frame


def validate_payments_refunds(
    payments: pd.DataFrame,
    refunds: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
) -> list[str]:
    """Return a list of problems found. An empty list means the tables are valid."""
    problems: list[str] = []

    if not payments["payment_id"].is_unique:
        return ["payment_id is not unique"]
    if not payments["order_id"].is_unique:
        return ["payments contain more than one row per order"]
    if not refunds["refund_id"].is_unique:
        return ["refund_id is not unique"]

    order_status = orders.set_index("order_id")["order_status"]
    order_date = orders.set_index("order_id")["order_date"]

    if not payments["order_id"].isin(order_status.index).all():
        problems.append("payments reference unknown orders")
    if (payments["order_id"].map(order_status) != "Completed").any():
        problems.append("payments exist for orders that are not completed")
    completed_ids = order_status.index[(order_status == "Completed").to_numpy()]
    if not completed_ids.isin(payments["order_id"]).all():
        problems.append("completed orders without a payment")

    totals = order_items.groupby("order_id")["line_total"].sum()
    expected = payments["order_id"].map(totals)
    bad_amount = ((expected - payments["amount"]).abs() > 0.005) | expected.isna()
    if bad_amount.any():
        problems.append("payment amount does not match order line totals")

    if (payments["payment_date"] < payments["order_id"].map(order_date)).any():
        problems.append("payment_date before order_date")
    if (payments["payment_date"] > pd.Timestamp(DATA_END)).any():
        problems.append("payment_date after the observation window")
    if not payments["payment_method"].isin(PAYMENT_METHODS).all():
        problems.append("unknown payment_method value")
    if not payments["payment_status"].isin(PAYMENT_STATUSES).all():
        problems.append("unknown payment_status value")

    pay = payments.set_index("payment_id")
    if refunds["payment_id"].duplicated().any():
        problems.append("more than one refund per payment")
    if not refunds["payment_id"].isin(pay.index).all():
        problems.append("refunds reference unknown payments")
    else:
        pay_status = refunds["payment_id"].map(pay["payment_status"])
        pay_amount = refunds["payment_id"].map(pay["amount"])
        pay_date = refunds["payment_id"].map(pay["payment_date"])
        if (pay_status != "Captured").any():
            problems.append("refund for a failed payment")
        if (refunds["order_id"] != refunds["payment_id"].map(pay["order_id"])).any():
            problems.append("refund order_id differs from payment order_id")
        if (refunds["refund_amount"] > pay_amount + 0.005).any():
            problems.append("refund_amount exceeds payment amount")
        full = refunds["refund_type"] == "Full"
        if ((refunds.loc[full, "refund_amount"] - pay_amount[full]).abs() > 0.005).any():
            problems.append("full refund amount differs from payment amount")
        if (refunds["refund_date"] <= pay_date).any():
            problems.append("refund_date not after payment_date")

    if (refunds["refund_amount"] <= 0).any():
        problems.append("refund_amount is not positive")
    if (refunds["refund_date"] > pd.Timestamp(DATA_END)).any():
        problems.append("refund_date after the observation window")
    if not refunds["refund_type"].isin(REFUND_TYPES).all():
        problems.append("unknown refund_type value")
    if not refunds["refund_reason"].isin(REFUND_REASONS).all():
        problems.append("unknown refund_reason value")

    null_count = int(payments.isna().sum().sum() + refunds.isna().sum().sum())
    if null_count:
        problems.append(f"{null_count} unexpected null values")

    return problems