import math

import pandas as pd
import pytest

from nexus.churn_scoring import AS_OF_CUTOFF, read_scores
from nexus.dashboard_data import fetch_kpis
from nexus.db import get_engine
from nexus.risk_view import RISK_DEFINITIONS, band_summary, csv_bytes, customer_table, filter_scores, risk_cards


def sample_scores() -> pd.DataFrame:
    """Ten customers, highest risk first: 2 High, 3 Medium, 5 Low."""
    return pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(10)],
            "churn_probability": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
            "risk_rank": list(range(1, 11)),
            "risk_decile": list(range(1, 11)),
            "risk_band": ["High"] * 2 + ["Medium"] * 3 + ["Low"] * 5,
            "recency_days": [170, 160, 100, 90, 80, 60, 50, 40, 30, 20],
            "frequency_180": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "monetary_180": [100.0, 100.0, 200.0, 200.0, 300.0, 300.0, 400.0, 400.0, 500.0, 500.0],
            "segment": ["Consumer", "Enterprise"] * 5,
        }
    )


def test_band_summary_by_hand():
    s = band_summary(sample_scores()).set_index("band")
    assert list(s.index) == ["High", "Medium", "Low"]
    assert s["customers"].tolist() == [2, 3, 5] and s["customers_pct"].tolist() == [20.0, 30.0, 50.0]
    assert s["mean_probability_pct"].round(6).tolist() == [85.0, 60.0, 21.0]
    assert s["mean_recency_days"].tolist() == [165.0, pytest.approx(90.0), pytest.approx(40.0)]
    assert s["spend_180"].tolist() == [200.0, 700.0, 2100.0]
    assert s["spend_share_pct"].round(3).tolist() == [6.667, 23.333, 70.0]


def test_cards_by_hand_and_with_a_missing_band():
    cards = {c.label: c for c in risk_cards(sample_scores())}
    assert list(cards) == list(RISK_DEFINITIONS)
    assert cards["Active customers scored"].value == "10" and cards["High-risk customers"].value == "2"
    assert cards["Spend of high-risk customers"].value == "6.7%" and cards["Average predicted churn"].value == "45.5%"
    no_high = sample_scores().assign(risk_band="Low")
    assert {c.label: c for c in risk_cards(no_high)}["High-risk customers"].value == "0"
    assert list(band_summary(no_high)["band"]) == ["Low"]


def test_filters_table_and_csv_export():
    scores = sample_scores()
    assert list(filter_scores(scores, ["High"], ["Consumer", "Enterprise"])["customer_id"]) == ["C0", "C1"]
    assert list(filter_scores(scores, ["High", "Medium", "Low"], ["Consumer"])["customer_id"]) == ["C0", "C2", "C4", "C6", "C8"]
    assert filter_scores(scores, [], ["Consumer"]).empty
    table = customer_table(scores, 3)
    assert list(table["Customer"]) == ["C0", "C1", "C2"] and table.loc[0, "Churn probability (%)"] == pytest.approx(90.0)
    assert list(table.columns)[:3] == ["Rank", "Customer", "Churn probability (%)"]
    data = csv_bytes(table)
    assert data.startswith(b"Rank,Customer,Churn probability (%)") and data.count(b"\n") == 4


# ---- reconciliation against the database ----

@pytest.fixture(scope="module")
def engine():
    return get_engine()


@pytest.fixture(scope="module")
def scores(engine):
    return read_scores(engine)


def test_stored_scores_are_complete_ranked_and_banded(scores):
    n = len(scores)
    assert n > 0 and scores["customer_id"].is_unique
    assert scores["risk_rank"].tolist() == list(range(1, n + 1))
    assert scores["churn_probability"].is_monotonic_decreasing
    assert scores["churn_probability"].between(0, 1).all()
    counts = scores["risk_band"].value_counts()
    assert counts["High"] == math.ceil(n / 5)
    assert counts["High"] + counts["Medium"] == math.ceil(n / 2)


def test_scores_reconcile_with_the_kpi_layer_for_the_same_window(engine, scores):
    start = (AS_OF_CUTOFF - pd.Timedelta(days=180)).date()
    end = (AS_OF_CUTOFF - pd.Timedelta(days=1)).date()
    k = fetch_kpis(engine, start, end)
    assert len(scores) == k.customers
    assert int(scores["frequency_180"].sum()) == k.orders
    assert abs(float(scores["monetary_180"].sum()) - k.revenue) < 0.005 * len(scores) + 1  # spend is rounded to cents