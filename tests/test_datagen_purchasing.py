import pandas as pd
import pytest

from nexus.datagen.constants import DATA_END, DATA_START, MIN_PO_QTY, N_PRODUCTS
from nexus.datagen.customers import generate_customers
from nexus.datagen.orders import generate_order_items, generate_orders
from nexus.datagen.purchasing import (
    daily_units_sold,
    generate_purchase_orders,
    initial_stock,
    validate_purchase_orders,
)
from nexus.datagen.reference import generate_products, generate_suppliers

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
def purchase_orders(products, suppliers, orders, order_items):
    return generate_purchase_orders(products, suppliers, orders, order_items, SEED)


def _delivered_before_2025(po):
    mask = (po["delivery_status"] == "Delivered") & (po["order_date"] < pd.Timestamp("2025-01-01"))
    d = po[mask].copy()
    d["on_time"] = d["actual_delivery_date"] <= d["expected_delivery_date"]
    return d


def test_po_shape_and_ids(purchase_orders):
    assert purchase_orders.shape[1] == 12
    assert 2_000 < len(purchase_orders) <= N_PRODUCTS * 36
    assert purchase_orders["po_id"].is_unique
    assert purchase_orders["po_id"].iloc[0] == "PO-000001"
    assert purchase_orders["po_id"].iloc[-1] == f"PO-{len(purchase_orders):06d}"


def test_po_supplier_and_cost_consistent_with_products(purchase_orders, products, suppliers):
    prod = products.set_index("product_id")
    assert purchase_orders["supplier_id"].isin(suppliers["supplier_id"]).all()
    assert (
        purchase_orders["supplier_id"] == purchase_orders["product_id"].map(prod["supplier_id"])
    ).all()
    assert (purchase_orders["unit_cost"] == purchase_orders["product_id"].map(prod["unit_cost"])).all()
    expected = (purchase_orders["quantity_ordered"] * purchase_orders["unit_cost"]).round(2)
    assert ((expected - purchase_orders["order_cost"]).abs() <= 0.005).all()


def test_po_dates_valid(purchase_orders):
    po = purchase_orders
    assert po["order_date"].between(pd.Timestamp(DATA_START), pd.Timestamp(DATA_END)).all()
    assert po["order_date"].is_monotonic_increasing
    assert (po["expected_delivery_date"] > po["order_date"]).all()


def test_delivered_and_in_transit_fields(purchase_orders):
    po = purchase_orders
    delivered = po["delivery_status"] == "Delivered"
    d, t = po[delivered], po[~delivered]
    assert len(t) > 0
    assert 0.001 < len(t) / len(po) < 0.04
    assert d["actual_delivery_date"].notna().all()
    assert (d["actual_delivery_date"] <= pd.Timestamp(DATA_END)).all()
    assert (d["actual_delivery_date"] > d["order_date"]).all()
    received = d["quantity_received"].astype("int64")
    assert ((received >= 1) & (received <= d["quantity_ordered"])).all()
    assert (d["defective_units"].astype("int64") <= received).all()
    assert t[["actual_delivery_date", "quantity_received", "defective_units"]].isna().all().all()
    assert (t["order_date"] >= pd.Timestamp("2025-11-01")).all()


def test_ordered_quantity_tracks_demand(purchase_orders, orders, order_items):
    completed = orders.loc[orders["order_status"] == "Completed", "order_id"]
    sold = order_items.loc[order_items["order_id"].isin(completed), "quantity"].sum()
    ratio = purchase_orders["quantity_ordered"].sum() / sold
    assert 0.90 < ratio < 1.25
    assert (purchase_orders["quantity_ordered"] >= MIN_PO_QTY).all()


def test_on_time_rate_tracks_supplier_rate(purchase_orders, suppliers):
    d = _delivered_before_2025(purchase_orders)
    observed = d.groupby("supplier_id")["on_time"].mean()
    stated = suppliers.set_index("supplier_id")["on_time_delivery_rate"].reindex(observed.index)
    assert observed.corr(stated) > 0.6
    assert abs(d["on_time"].mean() - stated.mean()) < 0.10


def test_peak_season_has_lower_on_time_rate(purchase_orders):
    d = _delivered_before_2025(purchase_orders)
    q4 = d["order_date"].dt.month >= 10
    assert d.loc[~q4, "on_time"].mean() - d.loc[q4, "on_time"].mean() > 0.02


def test_delay_distribution(purchase_orders):
    d = purchase_orders[purchase_orders["delivery_status"] == "Delivered"]
    delay = (d["actual_delivery_date"] - d["expected_delivery_date"]).dt.days
    assert delay.min() >= -2
    late = delay[delay > 0]
    assert len(late) > 100
    assert 4.0 < late.mean() < 6.5


def test_defective_units_track_supplier_defect_rate(purchase_orders, suppliers):
    d = purchase_orders[purchase_orders["delivery_status"] == "Delivered"]
    rate = d["supplier_id"].map(suppliers.set_index("supplier_id")["defect_rate"])
    received = d["quantity_received"].astype("int64")
    defective = d["defective_units"].astype("int64")
    expected = (received * rate).sum() / received.sum()
    observed = defective.sum() / received.sum()
    assert defective.sum() > 0
    assert abs(observed - expected) < 0.002


def test_short_shipment_share(purchase_orders):
    d = purchase_orders[purchase_orders["delivery_status"] == "Delivered"]
    short = (d["quantity_received"].astype("int64") < d["quantity_ordered"]).mean()
    assert 0.02 < short < 0.08


def test_generation_is_deterministic(products, suppliers, orders, order_items):
    p1 = generate_purchase_orders(products, suppliers, orders, order_items, SEED)
    p2 = generate_purchase_orders(products, suppliers, orders, order_items, SEED)
    pd.testing.assert_frame_equal(p1, p2)


def test_different_seed_changes_output(products, suppliers, orders, order_items):
    p1 = generate_purchase_orders(products, suppliers, orders, order_items, 1)
    p2 = generate_purchase_orders(products, suppliers, orders, order_items, 2)
    assert not p1.equals(p2)


def test_validation_passes_on_clean_data(purchase_orders, products, suppliers):
    assert validate_purchase_orders(purchase_orders, products, suppliers) == []


def test_validation_detects_broken_links_and_dates(purchase_orders, products, suppliers):
    broken = purchase_orders.copy()
    broken.loc[0, "supplier_id"] = "SUP-9999"
    broken.loc[1, "product_id"] = "PRD-9999"
    broken.loc[2, "expected_delivery_date"] = broken.loc[2, "order_date"]
    problems = validate_purchase_orders(broken, products, suppliers)
    assert any("unknown suppliers" in p for p in problems)
    assert any("unknown products" in p for p in problems)
    assert any("expected_delivery_date not after" in p for p in problems)


def test_validation_detects_broken_quantities_and_transit(purchase_orders, products, suppliers):
    po = purchase_orders
    delivered_idx = po.index[po["delivery_status"] == "Delivered"]
    transit_idx = po.index[po["delivery_status"] == "In Transit"]

    broken = po.copy()
    broken.loc[delivered_idx[0], "quantity_received"] = 10_000_000
    broken.loc[delivered_idx[1], "defective_units"] = 10_000_000
    broken.loc[transit_idx[0], "quantity_received"] = 5
    problems = validate_purchase_orders(broken, products, suppliers)
    assert any("quantity_received outside" in p for p in problems)
    assert any("defective_units outside" in p for p in problems)
    assert any("in-transit rows have delivery fields" in p for p in problems)

    duplicated = pd.concat([po, po.iloc[[0]]], ignore_index=True)
    problems = validate_purchase_orders(duplicated, products, suppliers)
    assert any("po_id is not unique" in p for p in problems)


def test_daily_units_sold_matches_completed_lines(products, orders, order_items):
    sales = daily_units_sold(orders, order_items, products)
    completed = orders.loc[orders["order_status"] == "Completed", "order_id"]
    expected = order_items.loc[order_items["order_id"].isin(completed), "quantity"].sum()
    assert sales.shape == (N_PRODUCTS, N_DAYS)
    assert sales.sum() == expected
    assert (sales >= 0).all()


def test_initial_stock_rules(products, orders, order_items):
    sales = daily_units_sold(orders, order_items, products)
    start_idx, qty = initial_stock(sales, products)
    launch = (products["launch_date"] - pd.Timestamp(DATA_START)).dt.days.to_numpy()
    assert (start_idx == launch.clip(0, N_DAYS - 1)).all()
    assert (qty >= MIN_PO_QTY).all()
    assert len(qty) == N_PRODUCTS


def test_inventory_position_stays_bounded(purchase_orders, products, orders, order_items):
    """Stock-aware ordering must not let stock pile up (or collapse) over the window."""
    sales = daily_units_sold(orders, order_items, products)
    _, init_qty = initial_stock(sales, products)
    delivered = purchase_orders[purchase_orders["delivery_status"] == "Delivered"]
    received = delivered["quantity_received"].astype("int64").sum()
    on_hand_end = init_qty.sum() + received - sales.sum()
    ratio = on_hand_end / sales.sum()
    assert -0.05 < ratio < 0.30