import pandas as pd
import pytest

from nexus.datagen.constants import DATA_START, N_PRODUCTS, N_SUPPLIERS
from nexus.datagen.reference import (
    generate_products,
    generate_suppliers,
    validate_reference_tables,
)

SEED = 42


@pytest.fixture(scope="module")
def suppliers():
    return generate_suppliers(SEED)


@pytest.fixture(scope="module")
def products(suppliers):
    return generate_products(suppliers, SEED)


def test_supplier_shape_and_ids(suppliers):
    assert len(suppliers) == N_SUPPLIERS
    assert suppliers["supplier_id"].is_unique
    assert suppliers["supplier_id"].iloc[0] == "SUP-0001"
    assert suppliers["supplier_name"].is_unique


def test_supplier_value_ranges(suppliers):
    assert suppliers["on_time_delivery_rate"].between(0.70, 1.0).all()
    assert suppliers["defect_rate"].between(0.0, 0.08).all()
    assert (suppliers["lead_time_mean_days"] >= 3).all()
    assert set(suppliers["payment_terms_days"]).issubset({15, 30, 45, 60, 90})
    assert (suppliers["onboarded_date"] < pd.Timestamp(DATA_START)).all()


def test_product_shape_and_ids(products):
    assert len(products) == N_PRODUCTS
    assert products["product_id"].is_unique
    assert products["sku"].is_unique
    assert products["product_id"].iloc[0] == "PRD-0001"
    assert products["category"].nunique() == 6


def test_product_foreign_keys(suppliers, products):
    assert products["supplier_id"].isin(suppliers["supplier_id"]).all()


def test_price_exceeds_cost(products):
    assert (products["list_price"] > products["unit_cost"]).all()
    assert (products["unit_cost"] >= 2.0).all()


def test_generation_is_deterministic():
    s1, s2 = generate_suppliers(SEED), generate_suppliers(SEED)
    pd.testing.assert_frame_equal(s1, s2)
    pd.testing.assert_frame_equal(generate_products(s1, SEED), generate_products(s2, SEED))


def test_different_seed_changes_output():
    assert not generate_suppliers(1).equals(generate_suppliers(2))


def test_validation_passes_on_clean_data(suppliers, products):
    assert validate_reference_tables(suppliers, products) == []


def test_validation_detects_orphan_supplier(suppliers, products):
    broken = products.copy()
    broken.loc[0, "supplier_id"] = "SUP-9999"
    problems = validate_reference_tables(suppliers, broken)
    assert any("unknown suppliers" in p for p in problems)


def test_validation_detects_price_below_cost(suppliers, products):
    broken = products.copy()
    broken.loc[0, "list_price"] = broken.loc[0, "unit_cost"] - 1.0
    problems = validate_reference_tables(suppliers, broken)
    assert any("list_price <= unit_cost" in p for p in problems)