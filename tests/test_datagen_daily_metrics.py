import numpy as np
import pandas as pd
import pytest

from nexus.datagen.constants import DATA_END, DATA_START
from nexus.datagen.customers import generate_customers
from nexus.datagen.daily_metrics import (
    DAILY_COLUMNS,
    build_daily_metrics,
    validate_daily_metrics,
)
from nexus.datagen.inventory import build_inventory_daily
from nexus.datagen.orders import generate_order_items, generate_orders
from nexus.datagen.payments import generate_payments, generate_refunds
from nexus.datagen.purchasing import generate_purchase_orders
from nexus.datagen.reference import generate_products, generate_suppliers
from nexus.datagen.shipments import generate_shipments

SEED = 42
N_DAYS = (DATA_END - DATA_START).days + 1


@pytest.fixture(scope="module")
def suppliers():
    return generate_suppliers(SEED)


@pytest.fixture(scope="module")
def products(suppliers):
    return generate_products(suppliers, SEED)


@pytest.fixture(scope="module")
def customers():
    return generate_customers(SEED)


@pytest.fixture(scope="module")
def orders(customers):
    return generate_orders(customers, SEED)


@pytest.fixture(scope="module")
def order_items(orders, customers, products):
    return generate_order_items(orders, customers, products, SEED)


@pytest.fixture(scope="module")
def payments(orders, order_items, customers):
    return generate_payments(orders, order_items, customers, SEED)


@pytest.fixture(scope="module")
def refunds(payments, order_items, products, suppliers):
    return generate_refunds(payments, order_items, products, suppliers, SEED)


@pytest.fixture(scope="module")
def purchase_orders(products, suppliers, orders, order_items):
    return generate_purchase_orders(products, suppliers, orders, order_items, SEED)


@pytest.fixture(scope="module")
def inventory(products, orders, order_items, purchase_orders):
    return build_inventory_daily(products, orders, order_items, purchase_orders)


@pytest.fixture(scope="module")
def shipments(orders, order_items, customers, inventory):
    return generate_shipments(orders, order_items, customers, inventory, SEED)


@pytest.fixture(scope="module")
def daily(customers, orders, order_items, payments, refunds, shipments, purchase_orders, inventory):
    return build_daily_metrics(
        customers, orders, order_items, payments, refunds, shipments, purchase_orders, inventory
    )


def _validate(dm, customers, orders, order_items, payments, refunds, shipments, purchase_orders, inventory):
    return validate_daily_metrics(
        dm, customers, orders, order_items, payments, refunds, shipments, purchase_orders, inventory
    )


def test_shape_and_calendar(daily):
    assert list(daily.columns) == DAILY_COLUMNS
    assert daily.shape == (N_DAYS, 20)
    assert daily["date"].is_unique
    assert daily["date"].is_monotonic_increasing
    assert daily["date"].iloc[0] == pd.Timestamp(DATA_START)
    assert daily["date"].iloc[-1] == pd.Timestamp(DATA_END)
    assert daily.isna().sum().sum() == 0


def test_identities(daily):
    assert (daily["orders_placed"] == daily["orders_completed"] + daily["orders_cancelled"]).all()
    margin = daily["gross_revenue"] - daily["cogs"]
    assert ((daily["gross_margin"] - margin).abs() <= 0.02).all()
    assert (daily["active_customers"] <= daily["orders_completed"]).all()
    assert (daily["products_in_stockout"] <= daily["listed_products"]).all()


def test_values_never_negative(daily):
    assert (daily.drop(columns="date") >= -0.01).all().all()


def test_order_and_unit_totals_reconcile(daily, orders, order_items, customers):
    completed = orders.loc[orders["order_status"] == "Completed", "order_id"]
    units = order_items.loc[order_items["order_id"].isin(completed), "quantity"].sum()
    in_window = customers["signup_date"].between(pd.Timestamp(DATA_START), pd.Timestamp(DATA_END)).sum()
    assert daily["orders_placed"].sum() == len(orders)
    assert daily["orders_completed"].sum() == len(completed)
    assert daily["units_sold"].sum() == units
    assert daily["new_customers"].sum() == in_window


def test_revenue_and_cash_reconcile(daily, payments, refunds):
    total = payments["amount"].sum()
    captured = payments.loc[payments["payment_status"] == "Captured", "amount"].sum()
    assert np.isclose(daily["gross_revenue"].sum(), total, atol=0.05)
    assert np.isclose(daily["payments_captured"].sum() + daily["payments_failed"].sum(), total, atol=0.05)
    assert np.isclose(daily["payments_captured"].sum(), captured, atol=0.05)
    assert np.isclose(daily["refunds_amount"].sum(), refunds["refund_amount"].sum(), atol=0.05)


def test_cost_and_operations_reconcile(daily, shipments, purchase_orders, inventory):
    assert np.isclose(daily["shipping_cost"].sum(), shipments["shipping_cost"].sum(), atol=0.05)
    assert np.isclose(daily["po_spend"].sum(), purchase_orders["order_cost"].sum(), atol=0.05)
    assert daily["listed_products"].sum() == len(inventory)
    assert daily["products_in_stockout"].sum() == inventory["stockout_flag"].sum()
    assert np.isclose(daily["inventory_value"].sum(), inventory["inventory_value"].sum(), atol=0.05)


def test_gross_margin_is_plausible(daily):
    pct = daily["gross_margin"].sum() / daily["gross_revenue"].sum()
    assert 0.20 < pct < 0.45


def test_november_is_seasonal_peak(daily):
    by_month = daily.groupby(daily["date"].dt.month)["gross_revenue"].mean()
    assert by_month[11] > 1.2 * by_month[2]


def test_business_is_growing(daily):
    by_year = daily.groupby(daily["date"].dt.year)["gross_revenue"].sum()
    assert by_year[2025] > by_year[2024] > by_year[2023]


def test_saturday_busier_than_tuesday(daily):
    by_day = daily.groupby(daily["date"].dt.dayofweek)["orders_placed"].mean()
    assert by_day[5] > by_day[1]


def test_generation_is_deterministic(
    customers, orders, order_items, payments, refunds, shipments, purchase_orders, inventory
):
    a = build_daily_metrics(customers, orders, order_items, payments, refunds, shipments, purchase_orders, inventory)
    b = build_daily_metrics(customers, orders, order_items, payments, refunds, shipments, purchase_orders, inventory)
    pd.testing.assert_frame_equal(a, b)


def test_validation_passes_on_clean_data(
    daily, customers, orders, order_items, payments, refunds, shipments, purchase_orders, inventory
):
    problems = _validate(daily, customers, orders, order_items, payments, refunds, shipments, purchase_orders, inventory)
    assert problems == []


def test_validation_detects_broken_values(
    daily, customers, orders, order_items, payments, refunds, shipments, purchase_orders, inventory
):
    broken = daily.copy()
    broken.loc[5, "gross_revenue"] = broken.loc[5, "gross_revenue"] + 100.0
    problems = _validate(broken, customers, orders, order_items, payments, refunds, shipments, purchase_orders, inventory)
    assert any("gross_revenue do not reconcile" in p for p in problems)

    broken = daily.copy()
    broken.loc[3, "orders_cancelled"] = broken.loc[3, "orders_cancelled"] + 1
    problems = _validate(broken, customers, orders, order_items, payments, refunds, shipments, purchase_orders, inventory)
    assert any("orders_placed differs" in p for p in problems)


def test_validation_detects_missing_and_duplicate_days(
    daily, customers, orders, order_items, payments, refunds, shipments, purchase_orders, inventory
):
    missing = daily.drop(index=10).reset_index(drop=True)
    problems = _validate(missing, customers, orders, order_items, payments, refunds, shipments, purchase_orders, inventory)
    assert any("complete observation window" in p for p in problems)

    duplicated = pd.concat([daily, daily.iloc[[0]]], ignore_index=True)
    problems = _validate(duplicated, customers, orders, order_items, payments, refunds, shipments, purchase_orders, inventory)
    assert any("duplicate dates" in p for p in problems)