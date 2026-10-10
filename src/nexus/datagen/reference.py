"""Reference tables: suppliers and products (SYNTHETIC data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nexus.datagen.constants import (
    CATEGORY_PROFILES,
    COUNTRY_PROFILES,
    DATA_START,
    N_PRODUCTS,
    N_SUPPLIERS,
    PAYMENT_TERMS_DAYS,
    PAYMENT_TERMS_PROBS,
    PRODUCT_ADJECTIVES,
    STREAM_PRODUCTS,
    STREAM_SUPPLIERS,
    SUPPLIER_NAME_PREFIXES,
    SUPPLIER_NAME_SUFFIXES,
)


def _rng(seed: int, stream: int) -> np.random.Generator:
    """Independent, reproducible random stream for one table."""
    return np.random.default_rng([seed, stream])


def generate_suppliers(seed: int, n: int = N_SUPPLIERS) -> pd.DataFrame:
    """Generate the suppliers table."""
    rng = _rng(seed, STREAM_SUPPLIERS)
    start = pd.Timestamp(DATA_START)

    combos = [f"{p} {s}" for p in SUPPLIER_NAME_PREFIXES for s in SUPPLIER_NAME_SUFFIXES]
    if n > len(combos):
        raise ValueError(f"At most {len(combos)} unique supplier names are available")
    name_idx = rng.choice(len(combos), size=n, replace=False)

    countries = rng.choice(list(COUNTRY_PROFILES), size=n)
    lead_time = [
        max(3, int(round(COUNTRY_PROFILES[str(c)].base_lead_days + rng.normal(0, 2))))
        for c in countries
    ]
    on_time = np.clip(rng.beta(18, 2, size=n), 0.70, 0.995).round(3)
    defect = np.clip(rng.beta(2, 98, size=n), 0.0, 0.08).round(4)
    terms = rng.choice(PAYMENT_TERMS_DAYS, size=n, p=PAYMENT_TERMS_PROBS)
    onboard_days = rng.integers(30, 365 * 6, size=n)

    return pd.DataFrame(
        {
            "supplier_id": [f"SUP-{i:04d}" for i in range(1, n + 1)],
            "supplier_name": [combos[int(i)] for i in name_idx],
            "country": [str(c) for c in countries],
            "region": [COUNTRY_PROFILES[str(c)].region for c in countries],
            "lead_time_mean_days": lead_time,
            "on_time_delivery_rate": on_time,
            "defect_rate": defect,
            "payment_terms_days": terms.astype(int),
            "onboarded_date": start - pd.to_timedelta(onboard_days, unit="D"),
        }
    )


def generate_products(suppliers: pd.DataFrame, seed: int, n: int = N_PRODUCTS) -> pd.DataFrame:
    """Generate the products table. Every product references an existing supplier."""
    rng = _rng(seed, STREAM_PRODUCTS)
    start = pd.Timestamp(DATA_START)
    supplier_ids = suppliers["supplier_id"].to_numpy()

    categories = list(CATEGORY_PROFILES)
    cycle = [categories[i % len(categories)] for i in range(n)]
    category_order = rng.permutation(cycle)

    rows = []
    for i in range(n):
        category = str(category_order[i])
        prof = CATEGORY_PROFILES[category]
        unit_cost = float(
            np.round(np.exp(rng.uniform(np.log(prof.cost_low), np.log(prof.cost_high))), 2)
        )
        markup = rng.uniform(prof.markup_low, prof.markup_high)
        list_price = round(unit_cost * (1.0 + markup), 2)
        adjective = PRODUCT_ADJECTIVES[int(rng.integers(len(PRODUCT_ADJECTIVES)))]
        noun = prof.nouns[int(rng.integers(len(prof.nouns)))]
        model = int(rng.integers(100, 1000))
        rows.append(
            {
                "product_id": f"PRD-{i + 1:04d}",
                "sku": f"SKU-{prof.code}-{i + 1:04d}",
                "product_name": f"{adjective} {noun} M{model}",
                "category": category,
                "supplier_id": str(rng.choice(supplier_ids)),
                "unit_cost": unit_cost,
                "list_price": list_price,
                "launch_date": start + pd.Timedelta(days=int(rng.integers(-1095, 540))),
                "is_active": bool(rng.random() > 0.08),
            }
        )
    return pd.DataFrame(rows)


def validate_reference_tables(suppliers: pd.DataFrame, products: pd.DataFrame) -> list[str]:
    """Return a list of problems found. An empty list means the tables are valid."""
    problems: list[str] = []

    if not suppliers["supplier_id"].is_unique:
        problems.append("supplier_id is not unique")
    if not products["product_id"].is_unique:
        problems.append("product_id is not unique")
    if not products["sku"].is_unique:
        problems.append("sku is not unique")

    orphans = ~products["supplier_id"].isin(suppliers["supplier_id"])
    if orphans.any():
        problems.append(f"{int(orphans.sum())} products reference unknown suppliers")

    if (products["list_price"] <= products["unit_cost"]).any():
        problems.append("some products have list_price <= unit_cost")

    if not suppliers["on_time_delivery_rate"].between(0, 1).all():
        problems.append("on_time_delivery_rate outside [0, 1]")
    if not suppliers["defect_rate"].between(0, 1).all():
        problems.append("defect_rate outside [0, 1]")
    if (suppliers["lead_time_mean_days"] < 1).any():
        problems.append("lead_time_mean_days below 1")

    null_count = int(suppliers.isna().sum().sum() + products.isna().sum().sum())
    if null_count:
        problems.append(f"{null_count} unexpected null values")

    return problems