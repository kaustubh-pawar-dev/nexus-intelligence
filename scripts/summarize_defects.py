"""Summarize the injected defects and check that simple rules find them (SYNTHETIC data)."""

from __future__ import annotations

import pandas as pd

from nexus.config import get_settings
from nexus.datagen.constants import DATA_END
from nexus.datagen.defects import DUPLICATE, UNKNOWN_CUSTOMER_ID


def main() -> int:
    raw = get_settings().raw_dir
    out = raw / "dirty"
    manifest = pd.read_csv(out / "defect_manifest.csv", keep_default_na=False)
    tables = ["customers", "orders", "order_items", "payments"]
    clean = {t: pd.read_csv(raw / f"{t}.csv") for t in tables}
    d = {t: pd.read_csv(out / f"{t}.csv") for t in tables}

    print("SYNTHETIC DATA SUMMARY: INJECTED DEFECTS")
    rows = []
    for t in tables:
        affected = manifest.loc[manifest["table"] == t, "row_key"].nunique()
        rows.append(
            {
                "table": t,
                "clean_rows": len(clean[t]),
                "dirty_rows": len(d[t]),
                "manifest_entries": int((manifest["table"] == t).sum()),
                "rows_affected_pct": round(affected / len(clean[t]) * 100, 2),
            }
        )
    print(pd.DataFrame(rows).set_index("table").to_string())

    def injected(table: str, defect: str) -> int:
        return int(((manifest["table"] == table) & (manifest["defect_type"] == defect)).sum())

    cust, orders, items, pay = d["customers"], d["orders"], d["order_items"], d["payments"]
    known_countries = set(clean["customers"]["country"])
    known_customers = set(clean["customers"]["customer_id"])
    methods = set(clean["payments"]["payment_method"])
    statuses = set(clean["orders"]["order_status"])
    end = pd.Timestamp(DATA_END).strftime("%Y-%m-%d")

    checks = [
        ("customers", DUPLICATE, "fully duplicated rows", int(cust.duplicated().sum())),
        ("customers", "missing_email", "email is empty", int(cust["email"].isna().sum())),
        ("customers", "country_variant", "country not in the known list", int((~cust["country"].isin(known_countries)).sum())),
        ("customers", "future_signup_date", "signup_date after 2025-12-31", int((cust["signup_date"] > end).sum())),
        ("orders", DUPLICATE, "fully duplicated rows", int(orders.duplicated().sum())),
        ("orders", "date_format_dmy", "order_date is not YYYY-MM-DD", int((~orders["order_date"].str.match(r"^\d{4}-\d{2}-\d{2}$")).sum())),
        ("orders", "missing_customer_id", "customer_id is empty", int(orders["customer_id"].isna().sum())),
        ("orders", "unknown_customer_id", f"customer_id not in customers (e.g. {UNKNOWN_CUSTOMER_ID})", int((~orders["customer_id"].dropna().isin(known_customers)).sum())),
        ("orders", "invalid_discount_pct", "discount_pct outside 0 to 0.2", int(((orders["discount_pct"] < 0) | (orders["discount_pct"] > 0.2)).sum())),
        ("orders", "status_case_variant", "order_status not in the known list", int((~orders["order_status"].isin(statuses)).sum())),
        ("order_items", DUPLICATE, "fully duplicated rows", int(items.duplicated().sum())),
        ("order_items", "negative_quantity", "quantity below 1", int((items["quantity"] < 1).sum())),
        ("order_items", "missing_line_total", "line_total is empty", int(items["line_total"].isna().sum())),
        ("order_items", "zero_unit_price", "unit_price is 0", int((items["unit_price"] == 0).sum())),
        ("payments", DUPLICATE, "fully duplicated rows", int(pay.duplicated().sum())),
        ("payments", "missing_amount", "amount is empty", int(pay["amount"].isna().sum())),
        ("payments", "method_case_variant", "payment_method not in the known list", int((~pay["payment_method"].isin(methods)).sum())),
    ]
    result = pd.DataFrame(
        [
            {"table": t, "defect": defect, "rule": rule, "found": found, "injected": injected(t, defect)}
            for t, defect, rule, found in checks
        ]
    )
    result["match"] = result["found"] == result["injected"]
    print("\nSimple rule against injected count (found should equal injected):")
    print(result.to_string(index=False))
    print(f"\nRules that match the manifest: {int(result['match'].sum())} of {len(result)}")
    print(f"Total manifest entries: {len(manifest):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())