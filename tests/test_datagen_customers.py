import pandas as pd
import pytest

from nexus.datagen.constants import (
    ACQUISITION_CHANNELS,
    CUSTOMER_SEGMENTS,
    CUSTOMER_SIGNUP_START,
    DATA_END,
    DATA_START,
    N_CUSTOMERS,
)
from nexus.datagen.customers import generate_customers, validate_customers

SEED = 42


@pytest.fixture(scope="module")
def customers():
    return generate_customers(SEED)


def test_customer_shape_and_ids(customers):
    assert len(customers) == N_CUSTOMERS
    assert customers.shape[1] == 8
    assert customers["customer_id"].is_unique
    assert customers["email"].is_unique
    assert customers["customer_id"].iloc[0] == "CUST-00001"
    assert customers["customer_id"].iloc[-1] == f"CUST-{N_CUSTOMERS:05d}"


def test_signup_dates_in_window_and_chronological(customers):
    assert customers["signup_date"].between(
        pd.Timestamp(CUSTOMER_SIGNUP_START), pd.Timestamp(DATA_END)
    ).all()
    assert customers["signup_date"].is_monotonic_increasing


def test_existing_customer_share(customers):
    share = (customers["signup_date"] < pd.Timestamp(DATA_START)).mean()
    assert 0.20 < share < 0.40


def test_business_is_growing(customers):
    by_year = customers["signup_date"].dt.year.value_counts()
    assert by_year[2025] > by_year[2021]


def test_segment_mix(customers):
    shares = customers["segment"].value_counts(normalize=True)
    assert 0.66 < shares["Consumer"] < 0.74
    assert 0.18 < shares["Small Business"] < 0.26
    assert 0.05 < shares["Enterprise"] < 0.11


def test_categorical_values_valid(customers):
    assert customers["segment"].isin(CUSTOMER_SEGMENTS).all()
    assert customers["acquisition_channel"].isin(ACQUISITION_CHANNELS).all()
    assert customers["acquisition_channel"].nunique() == len(ACQUISITION_CHANNELS)


def test_generation_is_deterministic():
    pd.testing.assert_frame_equal(generate_customers(SEED), generate_customers(SEED))


def test_different_seed_changes_output():
    assert not generate_customers(1).equals(generate_customers(2))


def test_validation_passes_on_clean_data(customers):
    assert validate_customers(customers) == []


def test_validation_detects_duplicate_email_and_bad_segment(customers):
    broken = customers.copy()
    broken.loc[1, "email"] = broken.loc[0, "email"]
    broken.loc[2, "segment"] = "VIP"
    problems = validate_customers(broken)
    assert any("email is not unique" in p for p in problems)
    assert any("unknown segment" in p for p in problems)