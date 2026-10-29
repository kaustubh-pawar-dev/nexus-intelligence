import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from nexus.churn import CHANNELS, SEGMENTS, fit_logistic
from nexus.churn_scoring import (
    AS_OF_CUTOFF,
    SCORE_COLUMNS,
    current_features,
    read_scores,
    risk_band,
    score_frame,
    write_scores,
)
from nexus.db import get_engine

T = pd.Timestamp


def fake_rows(n: int, seed: int, unique: bool = False) -> pd.DataFrame:
    """Synthetic feature rows where churn rises with recency."""
    rng = np.random.default_rng(seed)
    recency = rng.integers(1, 180, n).astype(float)
    freq = rng.integers(1, 12, n).astype(float)
    p = 1 / (1 + np.exp(-(recency - 90) / 25))
    ids = np.arange(n).astype(str) if unique else rng.integers(0, 400, n).astype(str)
    return pd.DataFrame(
        {
            "customer_id": ids,
            "frequency_180": freq,
            "monetary_180": freq * 40.0,
            "avg_order_value": np.full(n, 40.0),
            "recency_days": recency,
            "refunds_180": rng.integers(0, 2, n),
            "late_orders_180": rng.integers(0, 3, n),
            "orders_total": freq + 2,
            "tenure_days": rng.integers(30, 600, n),
            "days_since_signup": rng.integers(30, 700, n),
            "marketing_opt_in": rng.random(n) < 0.5,
            "segment": rng.choice(SEGMENTS, n),
            "channel": rng.choice(CHANNELS, n),
            "churned": (rng.random(n) < p).astype(int),
        }
    )


@pytest.fixture(scope="module")
def model():
    return fit_logistic(fake_rows(4000, 1))


def test_current_features_ignore_everything_dated_after_the_cutoff():
    t = {
        "orders": pd.DataFrame(
            {
                "order_id": ["a1", "a2", "a3", "b1", "c1"],
                "customer_id": ["A", "A", "A", "B", "C"],
                "order_date": ["2025-09-01", "2026-01-20", "2025-10-01", "2025-02-01", "2026-01-10"],
                "order_status": ["Completed", "Completed", "Cancelled", "Completed", "Completed"],
            }
        ),
        "order_items": pd.DataFrame(
            {"order_id": ["a1", "a2", "a3", "b1", "c1"], "line_total": [100.0, 500.0, 70.0, 30.0, 20.0]}
        ),
        "refunds": pd.DataFrame({"order_id": ["a1"], "refund_date": ["2026-01-05"]}),
        "shipments": pd.DataFrame(
            {"order_id": ["a1"], "delivered_date": ["2025-09-05"], "delivery_delay_days": [3.0]}
        ),
        "customers": pd.DataFrame(
            {
                "customer_id": ["A", "B", "C"],
                "signup_date": ["2024-01-01"] * 3,
                "segment": ["Consumer", "Enterprise", "Consumer"],
                "acquisition_channel": ["Direct", "Referral", "Email"],
                "marketing_opt_in": [True, False, True],
            }
        ),
    }
    f = current_features(t)
    assert list(f["customer_id"]) == ["A"]  # B ordered too long ago, C only after the cutoff
    row = f.iloc[0]
    assert row["frequency_180"] == 1 and row["monetary_180"] == 100.0
    assert row["recency_days"] == (AS_OF_CUTOFF - T("2025-09-01")).days
    assert row["refunds_180"] == 0  # the refund is dated after the cutoff
    assert row["late_orders_180"] == 1
    assert "churned" not in f.columns
    assert f["cutoff"].iloc[0] == AS_OF_CUTOFF


def test_risk_band_boundaries():
    assert [risk_band(d) for d in range(1, 11)] == ["High"] * 2 + ["Medium"] * 3 + ["Low"] * 5


def test_score_frame_ranks_bands_and_columns(model):
    feats = fake_rows(1000, 2, unique=True)
    s = score_frame(model, feats)
    assert list(s.columns) == SCORE_COLUMNS
    assert sorted(s["risk_rank"]) == list(range(1, 1001))
    assert s["churn_probability"].between(0, 1).all()
    assert s["churn_probability"].is_monotonic_decreasing
    assert (s["risk_decile"].value_counts() == 100).all()
    assert s["risk_band"].value_counts().to_dict() == {"Low": 500, "Medium": 300, "High": 200}
    assert set(s["cutoff"]) == {AS_OF_CUTOFF.date()}
    assert s["customer_id"].is_unique


def test_score_frame_is_deterministic(model):
    feats = fake_rows(500, 3, unique=True)
    pd.testing.assert_frame_equal(score_frame(model, feats), score_frame(model, feats))


def test_score_frame_rejects_duplicates_and_missing_values(model):
    feats = fake_rows(200, 4, unique=True)
    with pytest.raises(ValueError, match="one row per customer"):
        score_frame(model, pd.concat([feats, feats.iloc[:1]]))
    broken = feats.copy()
    broken.loc[0, "tenure_days"] = np.nan
    with pytest.raises(ValueError, match="tenure_days"):
        score_frame(model, broken)


def test_write_scores_is_repeatable_and_round_trips(model):
    scores = score_frame(model, fake_rows(300, 5, unique=True))
    engine = get_engine()
    schema = "ml_test"
    try:
        assert write_scores(engine, scores, schema=schema) == 300
        assert write_scores(engine, scores, schema=schema) == 300  # repeating replaces, never duplicates
        back = read_scores(engine, schema=schema)
        assert len(back) == 300
        assert list(back["customer_id"]) == list(scores["customer_id"])
        assert np.allclose(back["churn_probability"], scores["churn_probability"])
        assert np.allclose(back["monetary_180"], scores["monetary_180"])
        assert list(back["risk_band"]) == list(scores["risk_band"])
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))


def test_write_scores_rejects_unsafe_schema_and_mixed_cutoffs(model):
    scores = score_frame(model, fake_rows(50, 6, unique=True))
    with pytest.raises(ValueError, match="unsafe"):
        write_scores(None, scores, schema="ml; DROP TABLE x")
    mixed = scores.copy()
    mixed.loc[0, "cutoff"] = T("2025-12-01").date()
    with pytest.raises(ValueError, match="one cutoff"):
        write_scores(None, mixed)