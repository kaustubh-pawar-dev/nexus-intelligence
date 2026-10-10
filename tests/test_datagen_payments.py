import pandas as pd
import pytest

from nexus.datagen.constants import DATA_END
from nexus.datagen.customers import generate_customers
from nexus.datagen.orders import generate_order_items, generate_orders
from nexus.datagen.payments import (
    generate_payments,
    generate_refunds,
    validate_payments_refunds,
)
from nexus.datagen.reference import generate_products, generate_suppliers

SEED = 42


@pytest.fixture(scope="module")
def customers():
    return generate_customers(SEED)


@pytest.fixture(scope="module")
def suppliers():
    return generate_suppliers(SEED)


@pytest.fixture(scope="module")
def products(suppliers):
    return generate_products(suppliers, SEED)


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


def test_payments_shape_and_ids(payments):
    assert payments.shape[1] == 6
    assert payments["payment_id"].is_unique
    assert payments["payment_id"].iloc[0] == "PAY-000001"
    assert payments["payment_id"].iloc[-1] == f"PAY-{len(payments):06d}"


def test_one_payment_per_completed_order(payments, orders):
    completed = set(orders.loc[orders["order_status"] == "Completed", "order_id"])
    assert payments["order_id"].is_unique
    assert set(payments["order_id"]) == completed


def test_payment_amounts_reconcile_with_order_lines(payments, order_items):
    totals = order_items.groupby("order_id")["line_total"].sum()
    expected = payments["order_id"].map(totals)
    assert ((expected - payments["amount"]).abs() <= 0.005).all()


def test_payment_dates_valid(payments, orders):
    order_date = payments["order_id"].map(orders.set_index("order_id")["order_date"])
    assert (payments["payment_date"] >= order_date).all()
    assert (payments["payment_date"] <= pd.Timestamp(DATA_END)).all()


def test_payment_method_mix_by_segment(payments, orders, customers):
    customer_id = payments["order_id"].map(orders.set_index("order_id")["customer_id"])
    segment = customer_id.map(customers.set_index("customer_id")["segment"])
    share = pd.crosstab(segment, payments["payment_method"], normalize="index")
    assert share.loc["Enterprise", "Bank Transfer"] > 0.55
    assert share.loc["Consumer", "Bank Transfer"] < 0.10


def test_payment_failure_rates(payments):
    failed = payments["payment_status"] == "Failed"
    assert 0.01 < failed.mean() < 0.05
    by_method = failed.groupby(payments["payment_method"]).mean()
    assert by_method["Net Banking"] > by_method["Credit Card"]


def test_refunds_shape_and_keys(refunds, payments):
    captured_ids = set(payments.loc[payments["payment_status"] == "Captured", "payment_id"])
    assert refunds.shape[1] == 7
    assert refunds["refund_id"].is_unique
    assert refunds["refund_id"].iloc[0] == "REF-000001"
    assert refunds["payment_id"].is_unique
    assert set(refunds["payment_id"]).issubset(captured_ids)


def test_refund_amounts_and_dates_valid(refunds, payments):
    pay = payments.set_index("payment_id")
    amount = refunds["payment_id"].map(pay["amount"])
    pay_date = refunds["payment_id"].map(pay["payment_date"])
    assert (refunds["refund_amount"] > 0).all()
    assert (refunds["refund_amount"] <= amount + 0.005).all()
    assert (refunds["refund_date"] > pay_date).all()
    assert (refunds["refund_date"] <= pd.Timestamp(DATA_END)).all()
    full = refunds["refund_type"] == "Full"
    assert full.any() and (~full).any()
    assert ((refunds.loc[full, "refund_amount"] - amount[full]).abs() <= 0.005).all()
    assert (refunds.loc[~full, "refund_amount"] < amount[~full]).all()


def test_overall_refund_rate(refunds, payments):
    rate = len(refunds) / (payments["payment_status"] == "Captured").sum()
    assert 0.03 < rate < 0.10


def test_refund_rate_rises_with_supplier_defect_rate(
    refunds, payments, order_items, products, suppliers
):
    detail = order_items.merge(products[["product_id", "supplier_id"]], on="product_id").merge(
        suppliers[["supplier_id", "defect_rate"]], on="supplier_id"
    )
    detail["weighted"] = detail["line_total"] * detail["defect_rate"]
    grouped = detail.groupby("order_id")
    score = grouped["weighted"].sum() / grouped["line_total"].sum()

    captured = payments[payments["payment_status"] == "Captured"].copy()
    captured["score"] = captured["order_id"].map(score)
    captured["refunded"] = captured["payment_id"].isin(refunds["payment_id"])
    captured["band"] = pd.qcut(captured["score"], 3, labels=["low", "mid", "high"])
    rates = captured.groupby("band", observed=True)["refunded"].mean()
    assert rates["high"] > 1.3 * rates["low"]


def test_generation_is_deterministic(
    orders, order_items, customers, products, suppliers
):
    p1 = generate_payments(orders, order_items, customers, SEED)
    p2 = generate_payments(orders, order_items, customers, SEED)
    pd.testing.assert_frame_equal(p1, p2)
    pd.testing.assert_frame_equal(
        generate_refunds(p1, order_items, products, suppliers, SEED),
        generate_refunds(p2, order_items, products, suppliers, SEED),
    )


def test_different_seed_changes_output(orders, order_items, customers):
    p1 = generate_payments(orders, order_items, customers, 1)
    p2 = generate_payments(orders, order_items, customers, 2)
    assert not p1.equals(p2)


def test_validation_passes_on_clean_data(payments, refunds, orders, order_items):
    assert validate_payments_refunds(payments, refunds, orders, order_items) == []


def test_validation_detects_broken_payments(payments, refunds, orders, order_items):
    cancelled_id = orders.loc[orders["order_status"] == "Cancelled", "order_id"].iloc[0]
    broken = payments.copy()
    broken.loc[0, "amount"] = broken.loc[0, "amount"] + 5.0
    broken.loc[1, "order_id"] = cancelled_id
    problems = validate_payments_refunds(broken, refunds, orders, order_items)
    assert any("amount does not match" in p for p in problems)
    assert any("not completed" in p for p in problems)

    missing = payments.iloc[1:].reset_index(drop=True)
    problems = validate_payments_refunds(missing, refunds, orders, order_items)
    assert any("without a payment" in p for p in problems)


def test_validation_detects_broken_refunds(payments, refunds, orders, order_items):
    failed_id = payments.loc[payments["payment_status"] == "Failed", "payment_id"].iloc[0]
    broken = refunds.copy()
    broken.loc[0, "refund_amount"] = 1_000_000.0
    broken.loc[1, "payment_id"] = failed_id
    problems = validate_payments_refunds(payments, broken, orders, order_items)
    assert any("exceeds payment amount" in p for p in problems)
    assert any("failed payment" in p for p in problems)