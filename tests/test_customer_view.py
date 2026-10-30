from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from nexus.customer_view import CUSTOMER_DEFINITIONS, customer_cards, ltv_frame, retention_matrix, rfm_table
from nexus.dashboard_data import data_range, fetch_kpis
from nexus.db import get_engine
from nexus.sql_analytics import retention_curve, run_query

M = pd.Timestamp


def cohort_rows():
    """Three cohorts; the data ends in 2024-02. Rows with zero active customers are absent, as in the query."""
    rows = [
        (M("2023-12-01"), 0, 8, 8, 100.0),
        (M("2023-12-01"), 2, 4, 8, 50.0),  # month 1 is observed but nobody ordered: no row
        (M("2024-01-01"), 0, 10, 10, 100.0),
        (M("2024-01-01"), 1, 5, 10, 50.0),
        (M("2024-02-01"), 0, 20, 20, 100.0),
    ]
    return pd.DataFrame(rows, columns=["cohort_month", "months_since", "active_customers", "cohort_size", "retention_pct"])


def test_retention_matrix_fills_observed_gaps_with_zero_and_leaves_unobserved_cells_empty():
    m = retention_matrix(cohort_rows())
    assert list(m.columns) == [0, 1, 2]
    assert m.loc[M("2023-12-01")].tolist() == [100.0, 0.0, 50.0]  # observed gap is 0, not empty
    assert m.loc[M("2024-01-01")].iloc[:2].tolist() == [100.0, 50.0] and np.isnan(m.loc[M("2024-01-01"), 2])
    assert m.loc[M("2024-02-01"), 0] == 100.0 and m.loc[M("2024-02-01")].iloc[1:].isna().all()


def test_retention_matrix_agrees_with_the_existing_retention_curve():
    cohorts = cohort_rows()
    m = retention_matrix(cohorts)
    sizes = cohorts.drop_duplicates("cohort_month").set_index("cohort_month")["cohort_size"]
    curve = retention_curve(cohorts, offsets=(0, 1, 2)).set_index("months_since_first_order")
    for k in (0, 1, 2):
        observed = m[k].dropna()
        weighted = (observed * sizes.loc[observed.index]).sum() / sizes.loc[observed.index].sum()
        assert weighted == pytest.approx(curve.loc[k, "retention_pct"], abs=0.1)


def test_customer_cards_use_the_segment_and_retention_results():
    rfm = pd.DataFrame(
        {
            "rfm_segment": ["Champions", "At risk", "Dormant"],
            "customers": [400, 300, 300],
            "revenue_share_pct": [52.3, 30.0, 17.7],
        }
    )
    curve = pd.DataFrame({"months_since_first_order": [0, 1, 12], "retention_pct": [100.0, 21.4, float("nan")]})
    cards = {c.label: c for c in customer_cards(rfm, curve)}
    assert list(cards) == list(CUSTOMER_DEFINITIONS)
    assert cards["Customers with orders"].value == "1,000"
    assert cards["Champions' share of revenue"].value == "52.3%"
    assert cards["Average retention after 1 month"].value == "21.4%"
    assert cards["Average retention after 12 months"].value == "n/a"
    assert all(c.delta is None and c.delta_color == "off" for c in cards.values())


def test_rfm_and_ltv_tables_are_sorted_and_numeric():
    rfm = pd.DataFrame(
        {
            "rfm_segment": ["B", "A"],
            "customers": [10, 20],
            "avg_recency_days": ["5.0", "9.5"],
            "avg_orders": ["1.5", "2.5"],
            "avg_revenue": ["100.00", "300.00"],
            "revenue": ["1000.00", "6000.00"],
            "revenue_share_pct": ["14.3", "85.7"],
        }
    )
    t = rfm_table(rfm)
    assert list(t["Segment"]) == ["A", "B"] and t["Revenue"].dtype == float
    ltv = pd.DataFrame(
        {
            "dimension": ["segment", "segment", "channel"],
            "value": ["Small", "Large", "Email"],
            "customers": [5, 6, 7],
            "avg_orders": [1.0, 2.0, 3.0],
            "avg_revenue": [10.0, 20.0, 30.0],
            "avg_contribution_ltv": [2.0, 5.0, 9.0],
            "median_contribution_ltv": [1.0, 4.0, 8.0],
        }
    )
    assert list(ltv_frame(ltv, "segment")["value"]) == ["Large", "Small"]
    assert list(ltv_frame(ltv, "channel")["value"]) == ["Email"]


# ---- reconciliation against the database (cleaned analytics layer) ----

@pytest.fixture(scope="module")
def engine():
    return get_engine()


@pytest.fixture(scope="module")
def totals(engine):
    return fetch_kpis(engine, *data_range(engine))


def test_rfm_segments_cover_every_customer_and_all_revenue(engine, totals):
    rfm = run_query(engine, "rfm_segments")
    assert int(rfm["customers"].sum()) == totals.customers
    assert abs(float(rfm["revenue"].sum()) - totals.revenue) < 0.5  # segment revenue is rounded to cents
    assert abs(float(rfm["revenue_share_pct"].sum()) - 100.0) < 0.5


def test_cohort_sizes_add_up_to_the_customers_and_the_matrix_matches_the_query(engine, totals):
    cohorts = run_query(engine, "cohort_retention")
    sizes = cohorts.drop_duplicates("cohort_month")["cohort_size"]
    assert int(sizes.sum()) == totals.customers
    m = retention_matrix(cohorts)
    assert len(m) == cohorts["cohort_month"].nunique()
    assert (m[0] == 100.0).all()
    assert int((m.to_numpy() > 0).sum()) == len(cohorts)  # every query row is one non-zero cell
    values = m.to_numpy()
    assert np.nanmin(values) >= 0 and np.nanmax(values) <= 100


def test_ltv_rows_cover_every_customer_and_add_up_to_the_contribution(engine, totals):
    ltv = run_query(engine, "customer_ltv")
    for dimension in ("segment", "channel"):
        frame = ltv_frame(ltv, dimension)
        assert int(frame["customers"].sum()) == totals.customers
        weighted = float((frame["avg_contribution_ltv"] * frame["customers"]).sum())
        assert abs(weighted - totals.contribution) < 0.005 * totals.customers + 1  # averages are rounded to cents