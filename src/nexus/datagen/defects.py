"""Injected data-quality defects (SYNTHETIC data).

The clean tables stay in data/raw as ground truth. This module builds an "as delivered" copy
of four tables with known defects, plus a manifest that records every injected change.
Each row gets at most one cell defect, and duplicates are copies of untouched rows, so the
manifest is exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nexus.datagen.constants import DATA_END

STREAM_DEFECTS = 10
DUPLICATE = "exact_duplicate_row"
MANIFEST_COLUMNS = ["table", "row_key", "defect_type", "column", "original_value", "injected_value"]

# Share of the clean row count affected by each defect type (simulation settings).
DEFECT_RATES: dict[str, dict[str, float]] = {
    "customers": {
        DUPLICATE: 0.012,
        "missing_email": 0.02,
        "country_variant": 0.015,
        "future_signup_date": 0.003,
    },
    "orders": {
        DUPLICATE: 0.008,
        "date_format_dmy": 0.01,
        "missing_customer_id": 0.004,
        "unknown_customer_id": 0.004,
        "invalid_discount_pct": 0.006,
        "status_case_variant": 0.015,
    },
    "order_items": {
        DUPLICATE: 0.005,
        "negative_quantity": 0.002,
        "missing_line_total": 0.003,
        "zero_unit_price": 0.002,
    },
    "payments": {
        DUPLICATE: 0.004,
        "missing_amount": 0.003,
        "method_case_variant": 0.02,
    },
}
KEY_COLUMNS = {
    "customers": ["customer_id"],
    "orders": ["order_id"],
    "order_items": ["order_id", "line_number"],
    "payments": ["payment_id"],
}
UNKNOWN_CUSTOMER_ID = "CUST-99999"
INVALID_DISCOUNT_VALUE = 1.5
COUNTRY_ABBREVIATIONS = {"United States": "USA", "United Kingdom": "UK", "United Arab Emirates": "UAE"}


def _text_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of the frame with datetime columns as ISO text, as they appear in a CSV file."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out


def _row_keys(df: pd.DataFrame, table: str) -> np.ndarray:
    cols = KEY_COLUMNS[table]
    if len(cols) == 1:
        return df[cols[0]].astype(str).to_numpy()
    return (df[cols[0]].astype(str) + "|" + df[cols[1]].astype(str)).to_numpy()


def _fmt(value) -> str:
    return "" if pd.isna(value) else str(value)


def _carve(rng: np.random.Generator, n: int, rates: dict[str, float]) -> dict[str, np.ndarray]:
    """Disjoint random row positions for each defect type."""
    perm = rng.permutation(n)
    out: dict[str, np.ndarray] = {}
    pos = 0
    for name, rate in rates.items():
        k = int(round(n * rate))
        out[name] = np.sort(perm[pos : pos + k])
        pos += k
    return out


def _set(df, table, keys, rows, defect, column, new_values, manifest) -> None:
    if len(rows) == 0:
        return
    original = df[column].iloc[rows].tolist()
    new_list = list(new_values)
    df.loc[rows, column] = new_list
    for key, old, new in zip(keys[rows], original, new_list):
        manifest.append(
            {
                "table": table,
                "row_key": key,
                "defect_type": defect,
                "column": column,
                "original_value": _fmt(old),
                "injected_value": _fmt(new),
            }
        )


def _inject_customers(df, keys, rows, rng, manifest) -> None:
    t = "customers"
    r = rows["missing_email"]
    _set(df, t, keys, r, "missing_email", "email", [np.nan] * len(r), manifest)

    r = rows["country_variant"]
    new = [COUNTRY_ABBREVIATIONS.get(c, c.lower()) for c in df["country"].iloc[r]]
    _set(df, t, keys, r, "country_variant", "country", new, manifest)

    r = rows["future_signup_date"]
    days = rng.integers(30, 400, size=len(r))
    new = (pd.Timestamp(DATA_END) + pd.to_timedelta(days, unit="D")).strftime("%Y-%m-%d").tolist()
    _set(df, t, keys, r, "future_signup_date", "signup_date", new, manifest)


def _inject_orders(df, keys, rows, rng, manifest) -> None:
    t = "orders"
    r = rows["date_format_dmy"]
    new = pd.to_datetime(df["order_date"].iloc[r]).dt.strftime("%d/%m/%Y").tolist()
    _set(df, t, keys, r, "date_format_dmy", "order_date", new, manifest)

    r = rows["missing_customer_id"]
    _set(df, t, keys, r, "missing_customer_id", "customer_id", [np.nan] * len(r), manifest)

    r = rows["unknown_customer_id"]
    _set(df, t, keys, r, "unknown_customer_id", "customer_id", [UNKNOWN_CUSTOMER_ID] * len(r), manifest)

    r = rows["invalid_discount_pct"]
    _set(df, t, keys, r, "invalid_discount_pct", "discount_pct", [INVALID_DISCOUNT_VALUE] * len(r), manifest)

    r = rows["status_case_variant"]
    new = [s.lower() for s in df["order_status"].iloc[r]]
    _set(df, t, keys, r, "status_case_variant", "order_status", new, manifest)


def _inject_order_items(df, keys, rows, rng, manifest) -> None:
    t = "order_items"
    r = rows["negative_quantity"]
    new = [-int(q) for q in df["quantity"].iloc[r]]
    _set(df, t, keys, r, "negative_quantity", "quantity", new, manifest)

    r = rows["missing_line_total"]
    _set(df, t, keys, r, "missing_line_total", "line_total", [np.nan] * len(r), manifest)

    r = rows["zero_unit_price"]
    _set(df, t, keys, r, "zero_unit_price", "unit_price", [0.0] * len(r), manifest)


def _inject_payments(df, keys, rows, rng, manifest) -> None:
    t = "payments"
    r = rows["missing_amount"]
    _set(df, t, keys, r, "missing_amount", "amount", [np.nan] * len(r), manifest)

    r = rows["method_case_variant"]
    new = [m.lower() for m in df["payment_method"].iloc[r]]
    _set(df, t, keys, r, "method_case_variant", "payment_method", new, manifest)


_INJECTORS = {
    "customers": _inject_customers,
    "orders": _inject_orders,
    "order_items": _inject_order_items,
    "payments": _inject_payments,
}


def inject_defects(
    clean: dict[str, pd.DataFrame], seed: int
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return the dirty tables and the defect manifest. The clean frames are not modified."""
    missing = [t for t in DEFECT_RATES if t not in clean]
    if missing:
        raise ValueError(f"Missing clean tables: {missing}")

    rng = np.random.default_rng([seed, STREAM_DEFECTS])
    manifest: list[dict] = []
    dirty: dict[str, pd.DataFrame] = {}
    for table, rates in DEFECT_RATES.items():
        df = _text_dates(clean[table]).reset_index(drop=True)
        keys = _row_keys(df, table)
        rows = _carve(rng, len(df), rates)
        _INJECTORS[table](df, keys, rows, rng, manifest)

        dup = rows[DUPLICATE]
        for key in keys[dup]:
            manifest.append(
                {
                    "table": table,
                    "row_key": key,
                    "defect_type": DUPLICATE,
                    "column": "",
                    "original_value": "",
                    "injected_value": "",
                }
            )
        dirty[table] = pd.concat([df, df.iloc[dup]], ignore_index=True)
    return dirty, pd.DataFrame(manifest, columns=MANIFEST_COLUMNS)


def validate_defects(
    clean: dict[str, pd.DataFrame], dirty: dict[str, pd.DataFrame], manifest: pd.DataFrame
) -> list[str]:
    """Return a list of problems found. An empty list means dirty tables and manifest agree."""
    if list(manifest.columns) != MANIFEST_COLUMNS:
        return ["unexpected manifest columns"]

    problems: list[str] = []
    for table, rates in DEFECT_RATES.items():
        if table not in clean or table not in dirty:
            problems.append(f"{table}: table missing")
            continue
        c = _text_dates(clean[table]).reset_index(drop=True)
        d = dirty[table].reset_index(drop=True)
        m = manifest[manifest["table"] == table]

        for defect, rate in rates.items():
            expected = int(round(len(c) * rate))
            actual = int((m["defect_type"] == defect).sum())
            if actual != expected:
                problems.append(f"{table}: {actual} {defect} entries recorded, expected {expected}")

        n_dup = int((m["defect_type"] == DUPLICATE).sum())
        if len(d) != len(c) + n_dup:
            problems.append(f"{table}: row count differs from clean rows plus duplicates")
            continue
        if list(d.columns) != list(c.columns):
            problems.append(f"{table}: columns differ from the clean table")
            continue

        clean_keys = _row_keys(c, table)
        pos = pd.Series(np.arange(len(c)), index=clean_keys)
        if not pos.index.is_unique:
            problems.append(f"{table}: clean primary key is not unique")
            continue

        extra = d.iloc[len(c):].reset_index(drop=True)
        extra_keys = _row_keys(extra, table)
        dup_keys = m.loc[m["defect_type"] == DUPLICATE, "row_key"].tolist()
        if sorted(extra_keys.tolist()) != sorted(dup_keys):
            problems.append(f"{table}: appended rows do not match the duplicate entries")
        else:
            expected_rows = c.iloc[pos.loc[extra_keys].to_numpy()].reset_index(drop=True)
            if not extra.equals(expected_rows):
                problems.append(f"{table}: appended rows are not exact copies of clean rows")

        base = d.iloc[: len(c)]
        changed = 0
        for col in c.columns:
            same = (base[col] == c[col]) | (base[col].isna() & c[col].isna())
            changed += int((~same).sum())
        cells = m[m["defect_type"] != DUPLICATE]
        if changed != len(cells):
            problems.append(f"{table}: {changed} changed cells but {len(cells)} manifest entries")

        bad_key = bad_original = bad_injected = 0
        for row in cells.itertuples(index=False):
            if row.row_key not in pos.index:
                bad_key += 1
                continue
            p = int(pos.loc[row.row_key])
            if _fmt(c[row.column].iloc[p]) != _fmt(row.original_value):
                bad_original += 1
            if _fmt(base[row.column].iloc[p]) != _fmt(row.injected_value):
                bad_injected += 1
        if bad_key:
            problems.append(f"{table}: {bad_key} manifest keys not found in the clean table")
        if bad_original:
            problems.append(f"{table}: {bad_original} manifest original values differ from the clean table")
        if bad_injected:
            problems.append(f"{table}: {bad_injected} manifest injected values differ from the dirty table")

    return problems