"""Customer master table (SYNTHETIC data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nexus.datagen.constants import (
    ACQUISITION_CHANNELS,
    CHANNEL_PROBS,
    CUSTOMER_COUNTRY_PROFILES,
    CUSTOMER_SEGMENTS,
    CUSTOMER_SIGNUP_START,
    DATA_END,
    MARKETING_OPT_IN_RATE,
    N_CUSTOMERS,
    SEGMENT_PROBS,
    STREAM_CUSTOMERS,
)


def _rng(seed: int, stream: int) -> np.random.Generator:
    """Independent, reproducible random stream for one table."""
    return np.random.default_rng([seed, stream])


def generate_customers(seed: int, n: int = N_CUSTOMERS) -> pd.DataFrame:
    """Generate the customers table.

    Signup dates follow a linearly increasing daily weight, so later days have
    more signups (a growing business). Customer IDs are assigned in signup order.
    """
    rng = _rng(seed, STREAM_CUSTOMERS)
    start = pd.Timestamp(CUSTOMER_SIGNUP_START)

    n_days = (DATA_END - CUSTOMER_SIGNUP_START).days + 1
    weights = 1.0 + 2.0 * np.arange(n_days) / (n_days - 1)
    day_idx = np.sort(rng.choice(n_days, size=n, p=weights / weights.sum()))

    countries = list(CUSTOMER_COUNTRY_PROFILES)
    country_p = np.array([CUSTOMER_COUNTRY_PROFILES[c].weight for c in countries])
    country = rng.choice(countries, size=n, p=country_p / country_p.sum())
    segment = rng.choice(CUSTOMER_SEGMENTS, size=n, p=SEGMENT_PROBS)
    channel = rng.choice(ACQUISITION_CHANNELS, size=n, p=CHANNEL_PROBS)
    opt_in = rng.random(n) < MARKETING_OPT_IN_RATE

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(1, n + 1)],
            "email": [f"customer{i:05d}@example.com" for i in range(1, n + 1)],
            "country": [str(c) for c in country],
            "region": [CUSTOMER_COUNTRY_PROFILES[str(c)].region for c in country],
            "segment": [str(s) for s in segment],
            "acquisition_channel": [str(c) for c in channel],
            "signup_date": start + pd.to_timedelta(day_idx, unit="D"),
            "marketing_opt_in": opt_in,
        }
    )


def validate_customers(customers: pd.DataFrame) -> list[str]:
    """Return a list of problems found. An empty list means the table is valid."""
    problems: list[str] = []

    if not customers["customer_id"].is_unique:
        problems.append("customer_id is not unique")
    if not customers["email"].is_unique:
        problems.append("email is not unique")

    if not customers["segment"].isin(CUSTOMER_SEGMENTS).all():
        problems.append("unknown segment value")
    if not customers["acquisition_channel"].isin(ACQUISITION_CHANNELS).all():
        problems.append("unknown acquisition_channel value")
    if not customers["country"].isin(list(CUSTOMER_COUNTRY_PROFILES)).all():
        problems.append("unknown country value")

    expected_region = customers["country"].map(
        {c: p.region for c, p in CUSTOMER_COUNTRY_PROFILES.items()}
    )
    if not (customers["region"] == expected_region).all():
        problems.append("region does not match country")

    in_window = customers["signup_date"].between(
        pd.Timestamp(CUSTOMER_SIGNUP_START), pd.Timestamp(DATA_END)
    )
    if not in_window.all():
        problems.append("signup_date outside the allowed window")

    null_count = int(customers.isna().sum().sum())
    if null_count:
        problems.append(f"{null_count} unexpected null values")

    return problems