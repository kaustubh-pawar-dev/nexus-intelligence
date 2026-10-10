import pandas as pd
import pytest

from nexus.datagen.constants import (
    DATA_END,
    DATA_START,
    INACTIVE_PRODUCT_CUTOFF_DAYS,
    MAX_QUANTITY,
)
from nexus.datagen.customers import generate_customers
from nexus.datagen.orders import generate_order_items, generate_orders, validate_orders
from nexus.datagen.reference import generate_products, generate_suppliers

SEED = 42


@pytest.fixture(scope="module")
def customers():
    return generate_customers(SEED)


@pytest.fixture(scope="module")
def products():
    return generate_products(generate_suppliers(SEED), SEED)


@pytest.fixture(scope="module")
def orders(customers):
    return generate_orders(customers, SEED)


@pytest.fixture(scope="module")
def order_items(orders, customers, products):
    return generate_order_items(orders, customers, products, SEED)


def test_orders_shape_and_ids(orders):
    assert orders.shape[1] == 6
    assert 10_000 < len(orders) < 35_000
    assert orders["order_id"].is_unique
    assert orders["order_id"].iloc[0] == "ORD-000001"
    assert orders["order_id"].iloc[-1] == f"ORD-{len(orders):06d}"


def test_orders_in_window_sorted_and_after_signup(orders, customers):
    assert orders["order_date"].between(pd.Timestamp(DATA_START), pd.Timestamp(DATA_END)).all()
    assert orders["order_date"].is_monotonic_increasing
    signup = orders["customer_id"].map(customers.set_index("customer_id")["signup_date"])
    assert (orders["order_date"] >= signup).all()


def test_share_of_customers_with_orders(orders, customers):
    share = orders["customer_id"].nunique() / len(customers)
    assert 0.35 < share < 0.90


def test_enterprise_orders_more_than_others(orders, customers):
    counts = orders.groupby("customer_id").size().reindex(customers["customer_id"], fill_value=0)
    by_segment = counts.groupby(customers["segment"].to_numpy()).mean()
    assert by_segment["Enterprise"] > 3 * by_segment["Consumer"]
    assert by_segment["Small Business"] > by_segment["Consumer"]


def test_november_seasonal_peak(orders):
    month = orders["order_date"].dt.month
    november_per_day = (month == 11).sum() / 90
    february_per_day = (month == 2).sum() / 85
    assert november_per_day > 1.2 * february_per_day


def test_status_and_discount_rates(orders):
    cancelled = (orders["order_status"] == "Cancelled").mean()
    assert 0.02 < cancelled < 0.07
    discounted = (orders["discount_pct"] > 0).mean()
    assert 0.18 < discounted < 0.30


def test_items_shape_and_lines_per_order(orders, order_items):
    assert order_items.shape[1] == 8
    assert order_items["order_id"].nunique() == len(orders)
    assert 1.8 < len(order_items) / len(orders) < 3.6
    assert order_items["quantity"].between(1, MAX_QUANTITY).all()
    assert not order_items.duplicated(["order_id", "product_id"]).any()


def test_line_totals_reconcile(order_items):
    expected = (
        order_items["quantity"] * order_items["unit_price"] * (1 - order_items["discount_pct"])
    ).round(2)
    assert ((expected - order_items["line_total"]).abs() <= 0.005).all()


def test_inactive_products_not_sold_in_final_months(products, orders, order_items):
    cutoff = pd.Timestamp(DATA_END) - pd.Timedelta(days=INACTIVE_PRODUCT_CUTOFF_DAYS)
    inactive_ids = set(products.loc[~products["is_active"], "product_id"])
    joined = order_items.merge(orders[["order_id", "order_date"]], on="order_id")
    late = joined[joined["order_date"] > cutoff]
    assert len(late) > 0
    assert not late["product_id"].isin(inactive_ids).any()
    assert order_items["product_id"].isin(inactive_ids).any()


def test_generation_is_deterministic(customers, products):
    o1 = generate_orders(customers, SEED)
    o2 = generate_orders(customers, SEED)
    pd.testing.assert_frame_equal(o1, o2)
    pd.testing.assert_frame_equal(
        generate_order_items(o1, customers, products, SEED),
        generate_order_items(o2, customers, products, SEED),
    )


def test_different_seed_changes_output(customers):
    assert not generate_orders(customers, 1).equals(generate_orders(customers, 2))


def test_validation_passes_on_clean_data(orders, order_items, customers, products):
    assert validate_orders(orders, order_items, customers, products) == []


def test_validation_detects_broken_orders(orders, order_items, customers, products):
    broken = orders.copy()
    broken.loc[0, "customer_id"] = "CUST-99999"
    broken.loc[1, "order_status"] = "Lost"
    problems = validate_orders(broken, order_items, customers, products)
    assert any("unknown customers" in p for p in problems)
    assert any("unknown order_status" in p for p in problems)


def test_validation_detects_broken_items(orders, order_items, customers, products):
    broken = order_items.copy()
    broken.loc[0, "quantity"] = 0
    broken.loc[1, "line_total"] = broken.loc[1, "line_total"] + 5.0
    problems = validate_orders(orders, broken, customers, products)
    assert any("quantity outside" in p for p in problems)
    assert any("line_total" in p for p in problems)

    duplicated = pd.concat([order_items, order_items.iloc[[0]]], ignore_index=True)
    problems = validate_orders(orders, duplicated, customers, products)
    assert any("duplicate (order_id, line_number)" in p for p in problems)