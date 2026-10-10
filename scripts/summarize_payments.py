"""Print a summary of the generated payments and refunds (reads data/raw; SYNTHETIC data)."""

from __future__ import annotations

import pandas as pd

from nexus.config import get_settings


def main() -> int:
    raw = get_settings().raw_dir
    customers = pd.read_csv(raw / "customers.csv")
    orders = pd.read_csv(raw / "orders.csv", parse_dates=["order_date"])
    items = pd.read_csv(raw / "order_items.csv")
    products = pd.read_csv(raw / "products.csv")
    suppliers = pd.read_csv(raw / "suppliers.csv")
    payments = pd.read_csv(raw / "payments.csv", parse_dates=["payment_date"])
    refunds = pd.read_csv(raw / "refunds.csv", parse_dates=["refund_date"])

    completed = int((orders["order_status"] == "Completed").sum())
    captured = payments[payments["payment_status"] == "Captured"]

    print("SYNTHETIC DATA SUMMARY: PAYMENTS AND REFUNDS")
    print(f"completed orders: {completed:,}   payments: {len(payments):,}   refunds: {len(refunds):,}")

    print("\nPayment status share:")
    print(payments["payment_status"].value_counts(normalize=True).round(3).to_string())

    print("\nFailure rate by payment method:")
    failed = payments["payment_status"] == "Failed"
    print(failed.groupby(payments["payment_method"]).mean().round(3).to_string())

    pay = payments.merge(orders[["order_id", "customer_id"]], on="order_id").merge(
        customers[["customer_id", "segment"]], on="customer_id"
    )
    print("\nPayment method share within each segment:")
    print(pd.crosstab(pay["segment"], pay["payment_method"], normalize="index").round(2).to_string())

    by_year = (
        payments.assign(year=payments["payment_date"].dt.year)
        .pivot_table(index="year", columns="payment_status", values="amount", aggfunc="sum")
        .round(0)
    )
    print("\nSYNTHETIC payment amounts by year and status (Failed = simulated uncollected revenue):")
    print(by_year.to_string())

    print("\nRefunds:")
    print(f"refund rate among captured payments: {len(refunds) / len(captured):.3f}")
    print(f"refunded amount as a share of captured amount: {refunds['refund_amount'].sum() / captured['amount'].sum():.3f}")
    print("refund type share:")
    print(refunds["refund_type"].value_counts(normalize=True).round(3).to_string())
    print("refund reason share:")
    print(refunds["refund_reason"].value_counts(normalize=True).round(3).to_string())
    print("refunds by year of refund date:")
    print(refunds.groupby(refunds["refund_date"].dt.year).size().to_string())

    detail = items[["order_id", "product_id", "line_total"]].merge(
        products[["product_id", "supplier_id"]], on="product_id"
    ).merge(suppliers[["supplier_id", "defect_rate"]], on="supplier_id")
    detail["weighted"] = detail["line_total"] * detail["defect_rate"]
    grouped = detail.groupby("order_id")
    score = grouped["weighted"].sum() / grouped["line_total"].sum()

    cap = captured.copy()
    cap["defect_score"] = cap["order_id"].map(score)
    cap["refunded"] = cap["payment_id"].isin(refunds["payment_id"])
    cap["defect_band"] = pd.qcut(cap["defect_score"], 3, labels=["low", "mid", "high"])
    table = cap.groupby("defect_band", observed=True).agg(
        mean_defect_score=("defect_score", "mean"), refund_rate=("refunded", "mean")
    )
    print("\nRefund rate by order defect-score tercile (built into the simulation):")
    print(table.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())