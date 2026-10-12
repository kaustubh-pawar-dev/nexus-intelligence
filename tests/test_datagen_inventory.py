import pandas as pd
import pytest

from nexus.datagen.constants import DATA_END, DATA_START, N_PRODUCTS
from nexus.datagen.customers import generate_customers
from nexus.datagen.inventory import (
    INVENTORY_COLUMNS,
    UNIT_COLUMNS,
    build_inventory_daily,
    validate_inventory,
)
from nexus.datagen.orders import generate_order_items, generate_orders
from nexus.datagen.purchasing import daily_units_sold, generate_purchase_orders
from nexus.datagen.reference import generate_products, generate_suppliers

SEED = 42


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
def purchase_orders(products, suppliers, orders, order_items):
    return generate_purchase_orders(products, suppliers, orders, order_items, SEED)


@pytest.fixture(scope="module")
def inventory(products, orders, order_items, purchase_orders):
    return build_inventory_daily(products, orders, order_items, purchase_orders)


def _validate(inv, products, orders, order_items, purchase_orders):
    return validate_inventory(inv, products, orders, order_items, purchase_orders)


def test_shape_and_keys(inventory):
    assert list(inventory.columns) == INVENTORY_COLUMNS
    assert 80_000 < len(inventory) <= N_PRODUCTS * ((pd.Timestamp(DATA_END) - pd.Timestamp(DATA_START)).days + 1)
    assert not inventory.duplicated(["product_id", "date"]).any()
    assert inventory["product_id"].nunique() == N_PRODUCTS
    assert inventory.isna().sum().sum() == 0


def test_series_start_on_first_listed_day_and_end_at_window_end(inventory, products):
    start = inventory.groupby("product_id")["date"].min()
    expected = (
        products.set_index("product_id")["launch_date"]
        .clip(lower=pd.Timestamp(DATA_START))
        .reindex(start.index)
    )
    assert (start == expected).all()
    assert (inventory.groupby("product_id")["date"].max() == pd.Timestamp(DATA_END)).all()


def test_stock_and_backorder_identities(inventory):
    inv = inventory.sort_values(["product_id", "date"]).reset_index(drop=True)
    g = inv.groupby("product_id")
    prev_close = g["closing_stock"].shift(1).fillna(0)
    prev_back = g["backorder_units"].shift(1).fillna(0)
    assert (inv["opening_stock"] == prev_close).all()
    stock = (
        inv["opening_stock"]
        + inv["initial_stock_units"]
        + inv["po_receipts_units"]
        - inv["defective_writeoff_units"]
        - inv["units_fulfilled"]
    )
    assert (inv["closing_stock"] == stock).all()
    assert (inv["backorder_units"] == prev_back + inv["units_demanded"] - inv["units_fulfilled"]).all()


def test_units_never_negative_and_exclusive(inventory):
    assert (inventory[UNIT_COLUMNS] >= 0).all().all()
    assert not ((inventory["closing_stock"] > 0) & (inventory["backorder_units"] > 0)).any()


def test_totals_reconcile_with_sources(inventory, products, orders, order_items, purchase_orders):
    sales = daily_units_sold(orders, order_items, products)
    delivered = purchase_orders[purchase_orders["delivery_status"] == "Delivered"]
    assert inventory["units_demanded"].sum() == sales.sum()
    assert inventory["po_receipts_units"].sum() == delivered["quantity_received"].astype("int64").sum()
    assert inventory["defective_writeoff_units"].sum() == delivered["defective_units"].astype("int64").sum()


def test_stockout_flag_matches_backorders(inventory):
    assert set(inventory["stockout_flag"].unique()).issubset({0, 1})
    assert (inventory["stockout_flag"] == (inventory["backorder_units"] > 0).astype(int)).all()


def test_inventory_value_matches_cost(inventory, products):
    cost = inventory["product_id"].map(products.set_index("product_id")["unit_cost"])
    assert ((inventory["inventory_value"] - (inventory["closing_stock"] * cost).round(2)).abs() <= 0.005).all()


def test_writeoffs_never_exceed_receipts(inventory):
    assert (inventory["defective_writeoff_units"] <= inventory["po_receipts_units"]).all()
    assert inventory["defective_writeoff_units"].sum() > 0


def test_stockouts_are_rare_but_present(inventory):
    share = inventory["stockout_flag"].mean()
    assert 0.001 < share < 0.05
    assert inventory.loc[inventory["stockout_flag"] == 1, "product_id"].nunique() >= 20


def test_generation_is_deterministic(products, orders, order_items, purchase_orders):
    a = build_inventory_daily(products, orders, order_items, purchase_orders)
    b = build_inventory_daily(products, orders, order_items, purchase_orders)
    pd.testing.assert_frame_equal(a, b)


def test_validation_passes_on_clean_data(inventory, products, orders, order_items, purchase_orders):
    assert _validate(inventory, products, orders, order_items, purchase_orders) == []


def test_validation_detects_broken_identity_and_reconciliation(
    inventory, products, orders, order_items, purchase_orders
):
    broken = inventory.copy()
    broken.loc[10, "closing_stock"] = broken.loc[10, "closing_stock"] + 7
    problems = _validate(broken, products, orders, order_items, purchase_orders)
    assert any("stock identity broken" in p for p in problems)

    broken = inventory.copy()
    broken.loc[20, "units_demanded"] = broken.loc[20, "units_demanded"] + 3
    problems = _validate(broken, products, orders, order_items, purchase_orders)
    assert any("units_demanded do not reconcile" in p for p in problems)


def test_validation_detects_duplicates_missing_days_and_unknown_products(
    inventory, products, orders, order_items, purchase_orders
):
    duplicated = pd.concat([inventory, inventory.iloc[[0]]], ignore_index=True)
    problems = _validate(duplicated, products, orders, order_items, purchase_orders)
    assert any("duplicate (product_id, date)" in p for p in problems)

    missing = inventory.drop(index=5).reset_index(drop=True)
    problems = _validate(missing, products, orders, order_items, purchase_orders)
    assert any("missing days" in p for p in problems)

    unknown = inventory.copy()
    unknown.loc[0, "product_id"] = "PRD-9999"
    problems = _validate(unknown, products, orders, order_items, purchase_orders)
    assert any("unknown products" in p for p in problems)