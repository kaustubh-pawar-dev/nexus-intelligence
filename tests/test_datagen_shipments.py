import pandas as pd
import pytest

from nexus.datagen.customers import generate_customers
from nexus.datagen.inventory import build_inventory_daily
from nexus.datagen.orders import generate_order_items, generate_orders
from nexus.datagen.purchasing import generate_purchase_orders
from nexus.datagen.reference import generate_products, generate_suppliers
from nexus.datagen.shipments import SHIPMENT_COLUMNS, generate_shipments, validate_shipments

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


@pytest.fixture(scope="module")
def shipments(orders, order_items, customers, inventory):
    return generate_shipments(orders, order_items, customers, inventory, SEED)


def _delivered(sh):
    return sh[sh["shipment_status"] == "Delivered"]


def test_shape_and_ids(shipments):
    assert list(shipments.columns) == SHIPMENT_COLUMNS
    assert shipments["shipment_id"].is_unique
    assert shipments["shipment_id"].iloc[0] == "SHP-000001"
    assert shipments["shipment_id"].iloc[-1] == f"SHP-{len(shipments):06d}"


def test_one_shipment_per_completed_order(shipments, orders):
    completed = orders.loc[orders["order_status"] == "Completed", "order_id"]
    assert len(shipments) == len(completed)
    assert shipments["order_id"].is_unique
    assert set(shipments["order_id"]) == set(completed)


def test_date_identities_and_nullability(shipments, orders):
    sh = shipments
    od = sh["order_id"].map(orders.set_index("order_id")["order_date"])
    delivered = sh[sh["shipment_status"] == "Delivered"]
    transit = sh[sh["shipment_status"] == "In Transit"]
    awaiting = sh[sh["shipment_status"] == "Awaiting Shipment"]
    assert len(transit) > 0 and len(awaiting) > 0

    assert delivered[["ship_date", "delivered_date", "handling_days", "transit_days", "delivery_delay_days"]].notna().all().all()
    assert transit[["ship_date", "handling_days"]].notna().all().all()
    assert transit[["delivered_date", "transit_days", "delivery_delay_days"]].isna().all().all()
    assert awaiting[["ship_date", "handling_days", "delivered_date", "transit_days", "delivery_delay_days"]].isna().all().all()

    shipped = sh[sh["ship_date"].notna()]
    assert ((shipped["ship_date"] - od[shipped.index]).dt.days == shipped["handling_days"].astype("int64")).all()
    assert (shipped["ship_date"] >= od[shipped.index]).all()
    assert ((delivered["delivered_date"] - delivered["ship_date"]).dt.days == delivered["transit_days"].astype("int64")).all()
    assert (delivered["transit_days"].astype("int64") >= 1).all()
    delay = (delivered["delivered_date"] - delivered["promised_delivery_date"]).dt.days
    assert (delay == delivered["delivery_delay_days"].astype("int64")).all()
    assert (sh["promised_delivery_date"] > od).all()


def test_slow_carrier_is_late_more_often(shipments):
    d = _delivered(shipments)
    late = d["delivery_delay_days"].astype("int64") > 0
    rates = late.groupby(d["carrier"]).mean()
    assert rates["GlobalPost"] > rates["SwiftShip"] + 0.05


def test_overall_late_share_is_moderate(shipments):
    d = _delivered(shipments)
    share = (d["delivery_delay_days"].astype("int64") > 0).mean()
    assert 0.03 < share < 0.40


def test_backorders_delay_dispatch(shipments):
    s = shipments[shipments["handling_days"].notna()]
    handling = s["handling_days"].astype("int64")
    had = s["had_backorder"] == 1
    assert had.sum() > 50
    assert handling[had].mean() - handling[~had].mean() > 2


def test_far_regions_take_longer(shipments, orders, customers):
    d = _delivered(shipments)
    cust = shipments["order_id"].map(orders.set_index("order_id")["customer_id"])
    region = cust.map(customers.set_index("customer_id")["region"])
    transit = d["transit_days"].astype("int64").groupby(region[d.index]).mean()
    assert transit["Oceania"] > transit["North America"] + 1.5


def test_shipping_cost_reflects_carrier(shipments):
    assert (shipments["shipping_cost"] > 0).all()
    by_carrier = shipments.groupby("carrier")["shipping_cost"].mean()
    assert by_carrier["GlobalPost"] < by_carrier["SwiftShip"]


def test_generation_is_deterministic(orders, order_items, customers, inventory):
    a = generate_shipments(orders, order_items, customers, inventory, SEED)
    b = generate_shipments(orders, order_items, customers, inventory, SEED)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_output(orders, order_items, customers, inventory):
    a = generate_shipments(orders, order_items, customers, inventory, 1)
    b = generate_shipments(orders, order_items, customers, inventory, 2)
    assert not a.equals(b)


def test_validation_passes_on_clean_data(shipments, orders):
    assert validate_shipments(shipments, orders) == []


def test_validation_detects_broken_dates_status_and_carrier(shipments, orders):
    sh = shipments
    delivered_idx = sh.index[sh["shipment_status"] == "Delivered"]
    transit_idx = sh.index[sh["shipment_status"] == "In Transit"]

    broken = sh.copy()
    broken.loc[delivered_idx[0], "ship_date"] = pd.Timestamp("2020-01-01")
    problems = validate_shipments(broken, orders)
    assert any("ship_date before order_date" in p for p in problems)

    broken = sh.copy()
    broken.loc[transit_idx[0], "delivered_date"] = pd.Timestamp("2025-12-30")
    problems = validate_shipments(broken, orders)
    assert any("non-delivered rows have delivery fields" in p for p in problems)

    broken = sh.copy()
    broken.loc[delivered_idx[1], "carrier"] = "PigeonPost"
    problems = validate_shipments(broken, orders)
    assert any("unknown carrier" in p for p in problems)


def test_validation_detects_duplicates_and_missing_shipments(shipments, orders):
    duplicated = pd.concat([shipments, shipments.iloc[[0]]], ignore_index=True)
    assert any("shipment_id is not unique" in p for p in validate_shipments(duplicated, orders))

    missing = shipments.iloc[1:].reset_index(drop=True)
    assert any("completed orders without a shipment" in p for p in validate_shipments(missing, orders))