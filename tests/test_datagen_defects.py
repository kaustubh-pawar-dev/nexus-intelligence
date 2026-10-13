import pandas as pd
import pytest

from nexus.datagen.constants import DATA_END, N_CUSTOMERS, PAYMENT_METHODS
from nexus.datagen.customers import generate_customers
from nexus.datagen.defects import (
    DEFECT_RATES,
    DUPLICATE,
    MANIFEST_COLUMNS,
    UNKNOWN_CUSTOMER_ID,
    inject_defects,
    validate_defects,
)
from nexus.datagen.orders import generate_order_items, generate_orders
from nexus.datagen.payments import generate_payments
from nexus.datagen.reference import generate_products, generate_suppliers

SEED = 42


@pytest.fixture(scope="module")
def clean():
    suppliers = generate_suppliers(SEED)
    products = generate_products(suppliers, SEED)
    customers = generate_customers(SEED)
    orders = generate_orders(customers, SEED)
    order_items = generate_order_items(orders, customers, products, SEED)
    payments = generate_payments(orders, order_items, customers, SEED)
    return {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "payments": payments,
    }


@pytest.fixture(scope="module")
def result(clean):
    return inject_defects(clean, SEED)


def _count(manifest, table, defect):
    return int(((manifest["table"] == table) & (manifest["defect_type"] == defect)).sum())


def test_manifest_columns_and_content(result):
    _, manifest = result
    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert len(manifest) > 1_000
    assert set(manifest["table"]) == set(DEFECT_RATES)
    dups = manifest[manifest["defect_type"] == DUPLICATE]
    assert (dups["column"] == "").all()
    assert (manifest.loc[manifest["defect_type"] != DUPLICATE, "column"] != "").all()


def test_row_counts_are_clean_plus_duplicates(result, clean):
    dirty, manifest = result
    for table in DEFECT_RATES:
        n_dup = _count(manifest, table, DUPLICATE)
        assert n_dup > 0
        assert len(dirty[table]) == len(clean[table]) + n_dup


def test_defect_counts_follow_the_rates(result, clean):
    _, manifest = result
    total = 0
    for table, rates in DEFECT_RATES.items():
        for defect, rate in rates.items():
            expected = int(round(len(clean[table]) * rate))
            assert _count(manifest, table, defect) == expected
            total += expected
    assert len(manifest) == total


def test_validation_passes_on_injected_data(result, clean):
    dirty, manifest = result
    assert validate_defects(clean, dirty, manifest) == []


def test_manifest_originals_match_clean_values(result, clean):
    _, manifest = result
    m = manifest[(manifest["table"] == "customers") & (manifest["defect_type"] == "missing_email")]
    emails = clean["customers"].set_index("customer_id")["email"]
    assert len(m) == int(round(N_CUSTOMERS * 0.02))
    assert (m["row_key"].map(emails) == m["original_value"]).all()


def test_each_defect_is_found_by_a_simple_rule(result, clean):
    dirty, manifest = result
    c, o, i, p = dirty["customers"], dirty["orders"], dirty["order_items"], dirty["payments"]
    known_countries = set(clean["customers"]["country"])
    known_customers = set(clean["customers"]["customer_id"])
    statuses = {"Completed", "Cancelled"}
    end = pd.Timestamp(DATA_END).strftime("%Y-%m-%d")

    assert c.duplicated().sum() == _count(manifest, "customers", DUPLICATE)
    assert c["email"].isna().sum() == _count(manifest, "customers", "missing_email")
    assert (~c["country"].isin(known_countries)).sum() == _count(manifest, "customers", "country_variant")
    assert (c["signup_date"] > end).sum() == _count(manifest, "customers", "future_signup_date")

    assert o.duplicated().sum() == _count(manifest, "orders", DUPLICATE)
    assert (~o["order_date"].str.match(r"^\d{4}-\d{2}-\d{2}$")).sum() == _count(manifest, "orders", "date_format_dmy")
    assert o["customer_id"].isna().sum() == _count(manifest, "orders", "missing_customer_id")
    assert (~o["customer_id"].dropna().isin(known_customers)).sum() == _count(manifest, "orders", "unknown_customer_id")
    assert (o["customer_id"] == UNKNOWN_CUSTOMER_ID).sum() == _count(manifest, "orders", "unknown_customer_id")
    assert ((o["discount_pct"] < 0) | (o["discount_pct"] > 0.2)).sum() == _count(manifest, "orders", "invalid_discount_pct")
    assert (~o["order_status"].isin(statuses)).sum() == _count(manifest, "orders", "status_case_variant")

    assert i.duplicated().sum() == _count(manifest, "order_items", DUPLICATE)
    assert (i["quantity"] < 1).sum() == _count(manifest, "order_items", "negative_quantity")
    assert i["line_total"].isna().sum() == _count(manifest, "order_items", "missing_line_total")
    assert (i["unit_price"] == 0).sum() == _count(manifest, "order_items", "zero_unit_price")

    assert p.duplicated().sum() == _count(manifest, "payments", DUPLICATE)
    assert p["amount"].isna().sum() == _count(manifest, "payments", "missing_amount")
    assert (~p["payment_method"].isin(PAYMENT_METHODS)).sum() == _count(manifest, "payments", "method_case_variant")


def test_cell_defects_never_share_a_row(result):
    _, manifest = result
    cells = manifest[manifest["defect_type"] != DUPLICATE]
    assert not cells.duplicated(["table", "row_key"]).any()
    dups = manifest[manifest["defect_type"] == DUPLICATE]
    merged = dups.merge(cells, on=["table", "row_key"], how="inner")
    assert len(merged) == 0


def test_dates_are_text_and_injected_formats_differ(result):
    dirty, manifest = result
    orders = dirty["orders"]
    assert orders["order_date"].str.len().min() == 10
    m = manifest[manifest["defect_type"] == "date_format_dmy"]
    assert m["injected_value"].str.match(r"^\d{2}/\d{2}/\d{4}$").all()
    assert m["original_value"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()


def test_injection_is_deterministic(clean):
    d1, m1 = inject_defects(clean, SEED)
    d2, m2 = inject_defects(clean, SEED)
    pd.testing.assert_frame_equal(m1, m2)
    for table in DEFECT_RATES:
        pd.testing.assert_frame_equal(d1[table], d2[table])


def test_different_seed_changes_the_manifest(clean):
    _, m1 = inject_defects(clean, 1)
    _, m2 = inject_defects(clean, 2)
    assert not m1.equals(m2)


def test_clean_inputs_are_not_modified(clean):
    before = {t: f.copy() for t, f in clean.items()}
    inject_defects(clean, SEED)
    for table in DEFECT_RATES:
        pd.testing.assert_frame_equal(before[table], clean[table])


def test_validation_detects_tampering(result, clean):
    dirty, manifest = result

    broken = {t: f.copy() for t, f in dirty.items()}
    broken["customers"].loc[0, "country"] = "Narnia"
    problems = validate_defects(clean, broken, manifest)
    assert any(p.startswith("customers") for p in problems)

    broken = {t: f.copy() for t, f in dirty.items()}
    broken["orders"] = broken["orders"].iloc[:-1]
    problems = validate_defects(clean, broken, manifest)
    assert any("row count differs" in p for p in problems)

    problems = validate_defects(clean, dirty, manifest.iloc[1:])
    assert len(problems) > 0