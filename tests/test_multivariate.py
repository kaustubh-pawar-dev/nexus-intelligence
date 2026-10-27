import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from nexus.db import get_engine
from nexus.multivariate import (
    CHANNEL_TERMS,
    SEGMENT_TERMS,
    channel_segment_mix,
    coef_table,
    customer_model_frame,
    design_matrix,
    fit_ltv_model,
    fit_order_rate_model,
    referral_vs_paid_by_segment,
    spearman_matrix,
)
from nexus.sql_analytics import run_query

CHANNELS = ["Direct", "Email", "Organic Search", "Paid Search", "Referral", "Social Media"]


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(n)],
            "segment": rng.choice(["Consumer", "Small Business", "Enterprise"], n),
            "channel": rng.choice(CHANNELS, n),
            "marketing_opt_in": rng.random(n) < 0.6,
            "exposure_days": rng.integers(100, 1097, n),
        }
    )


def test_exposure_days_known_dates():
    cust = pd.DataFrame({"signup_date": [dt.date(2021, 5, 1), dt.date(2023, 1, 1), dt.date(2024, 1, 1), dt.date(2025, 12, 31)]})
    out = customer_model_frame(cust)["exposure_days"].tolist()
    assert out == [1096, 1096, 731, 1]


def test_design_matrix_columns_and_baselines():
    x = design_matrix(synthetic(50))
    assert list(x.columns) == ["const", *[f"segment: {s}" for s in SEGMENT_TERMS],
                               *[f"channel: {c}" for c in CHANNEL_TERMS], "opt_in", "log_exposure"]
    assert "segment: Consumer" not in x.columns and "channel: Paid Search" not in x.columns
    assert (x["const"] == 1.0).all() and len(x) == 50
    assert "log_exposure" not in design_matrix(synthetic(10), include_exposure=False).columns


def test_ltv_model_recovers_known_coefficients():
    df = synthetic()
    rng = np.random.default_rng(1)
    df["revenue"] = np.exp(
        2 + 1.0 * (df["segment"] == "Enterprise") + 0.3 * (df["channel"] == "Referral")
        + 0.5 * np.log(df["exposure_days"]) + rng.normal(0, 0.5, len(df))
    )
    df["orders"], df["contribution"] = 1, df["revenue"]
    _, table = fit_ltv_model(df)
    t = table.set_index("term")
    assert abs(t.loc["segment: Enterprise", "coef"] - 1.0) < 0.10
    assert abs(t.loc["channel: Referral", "coef"] - 0.3) < 0.10
    assert abs(t.loc["log_exposure", "coef"] - 0.5) < 0.08
    assert abs(t.loc["opt_in", "coef"]) < 0.10


def test_order_rate_model_recovers_a_known_rate_ratio():
    df = synthetic()
    rng = np.random.default_rng(2)
    df["orders"] = rng.poisson(df["exposure_days"] * 0.01 * np.exp(0.4 * df["marketing_opt_in"]))
    _, table = fit_order_rate_model(df)
    t = table.set_index("term")
    assert abs(t.loc["opt_in", "coef"] - 0.4) < 0.10
    assert t.loc["opt_in", "ci_low"] < 0.4 < t.loc["opt_in", "ci_high"]


def test_coef_table_intervals_and_percentage_effect():
    df = synthetic()
    df["orders"] = np.random.default_rng(3).poisson(3, len(df))
    _, t = fit_order_rate_model(df)
    assert (t["ci_low"] < t["coef"]).all() and (t["coef"] < t["ci_high"]).all()
    assert np.allclose(t["pct_effect"], 100 * (np.exp(t["coef"]) - 1))
    assert (t["pct_low"] < t["pct_high"]).all()
    assert t["p_value"].between(0, 1).all()


def test_spearman_matrix_on_monotone_data():
    df = pd.DataFrame({"orders": [1, 2, 3, 4], "revenue": [10, 20, 30, 40],
                       "contribution": [1, 3, 5, 9], "exposure_days": [5, 6, 7, 8]})
    m = spearman_matrix(df)
    assert np.allclose(m.to_numpy(), 1.0)
    assert list(m.index) == list(m.columns)


def test_within_segment_comparison_skips_small_groups_and_finds_a_shift():
    rng = np.random.default_rng(4)
    rows = []
    for ch, level in (("Referral", 2.0), ("Paid Search", 0.0)):
        for _ in range(40):
            rows.append({"segment": "Consumer", "channel": ch, "orders": 1, "contribution": rng.normal(level, 0.5)})
        for _ in range(5):
            rows.append({"segment": "Small Business", "channel": ch, "orders": 1, "contribution": rng.normal(level, 0.5)})
    out = referral_vs_paid_by_segment(pd.DataFrame(rows)).set_index("segment")
    assert out.loc["Consumer", "cliffs_delta"] > 0.9 and out.loc["Consumer", "p_holm"] < 1e-6
    assert np.isnan(out.loc["Small Business", "cliffs_delta"]) and np.isnan(out.loc["Small Business", "p_holm"])
    assert np.isnan(out.loc["Enterprise", "cliffs_delta"]) and out.loc["Enterprise", "n_referral"] == 0


@pytest.fixture(scope="module")
def engine():
    try:
        eng = get_engine()
        with eng.connect() as conn:
            ready = conn.execute(text("SELECT to_regclass('analytics.orders') IS NOT NULL")).scalar()
    except Exception as exc:
        pytest.skip(f"PostgreSQL is not reachable: {type(exc).__name__}")
    if not ready:
        pytest.skip("run scripts/build_analytics.py first")
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def df(engine):
    return customer_model_frame(run_query(engine, "customer_level"))


def test_customer_frame_exposure_on_real_data(df):
    assert len(df) == 5000
    assert df["exposure_days"].between(1, 1096).all()
    assert (df["exposure_days"] == 1096).sum() > 1000
    assert df["marketing_opt_in"].dtype == bool


def test_ltv_model_uses_every_buyer_and_enterprise_dominates(df):
    res, table = fit_ltv_model(df)
    assert int(res.nobs) == int((df["orders"] > 0).sum())
    t = table.set_index("term")
    assert t.loc["segment: Enterprise", "coef"] > 2.0 and t.loc["segment: Enterprise", "p_value"] < 1e-6
    assert t.loc["segment: Enterprise", "coef"] > t.loc["segment: Small Business", "coef"] > 0
    assert 0.3 < res.rsquared < 1.0


def test_order_rate_model_uses_every_customer(df):
    res, table = fit_order_rate_model(df)
    assert int(res.nobs) == len(df)
    t = table.set_index("term")
    assert t.loc["segment: Enterprise", "coef"] > t.loc["segment: Small Business", "coef"] > 0
    assert t.loc["opt_in", "ci_low"] < t.loc["opt_in", "ci_high"]


def test_correlation_matrix_and_channel_mix_on_real_data(df):
    m = spearman_matrix(df)
    assert np.allclose(np.diag(m.to_numpy()), 1.0) and np.allclose(m, m.T)
    assert m.loc["orders", "revenue"] > 0.8
    mix = channel_segment_mix(df)
    assert set(mix.index) == set(CHANNELS)
    assert np.allclose(mix.sum(axis=1), 1.0, atol=0.01)


def test_within_segment_table_reconciles_with_buyer_counts(df):
    out = referral_vs_paid_by_segment(df)
    b = df[df["orders"] > 0]
    assert list(out["segment"]) == ["Consumer", "Small Business", "Enterprise"]
    assert out["n_referral"].sum() == (b["channel"] == "Referral").sum()
    assert out["n_paid_search"].sum() == (b["channel"] == "Paid Search").sum()
    assert out["cliffs_delta"].dropna().between(-1, 1).all()