import numpy as np
import pandas as pd

from nexus.churn import CHANNELS, SEGMENTS, design
from nexus.churn_diagnostics import (
    calibration_gap,
    flag_table,
    logistic_effects,
    metrics_by_cutoff,
    risk_deciles,
)
from nexus.churn_models import evaluate_full


def fake_rows(n: int, seed: int, customers: int = 400) -> pd.DataFrame:
    """Synthetic churn rows where churn rises with recency."""
    rng = np.random.default_rng(seed)
    recency = rng.integers(1, 180, n).astype(float)
    freq = rng.integers(1, 12, n).astype(float)
    p = 1 / (1 + np.exp(-(recency - 90) / 25))
    return pd.DataFrame(
        {
            "customer_id": rng.integers(0, customers, n).astype(str),
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


def test_risk_deciles_on_a_perfect_ranking():
    y = np.array([1] * 40 + [0] * 60)
    score = np.linspace(1, 0, 100)
    d = risk_deciles(y, score)
    assert d["rows"].sum() == 100 and len(d) == 10
    assert d.loc[0, "observed_churn"] == 1.0 and abs(d.loc[0, "lift"] - 2.5) < 1e-12
    assert d.loc[4, "observed_churn"] == 0.0
    assert abs(d.loc[3, "cum_recall"] - 1.0) < 1e-12
    assert d["mean_predicted"].is_monotonic_decreasing


def test_calibration_gap_is_small_for_true_probabilities_and_large_for_inverted_ones():
    rng = np.random.default_rng(3)
    p = rng.random(40000)
    y = (rng.random(40000) < p).astype(int)
    assert calibration_gap(risk_deciles(y, p)) < 0.02
    assert calibration_gap(risk_deciles(y, 1 - p)) > 0.2


def test_flag_table_matches_evaluate_and_recall_never_falls():
    rng = np.random.default_rng(4)
    s = rng.random(500)
    y = (rng.random(500) < s).astype(int)
    t = flag_table(y, s)
    ref = evaluate_full(y, s, top_share=0.2)
    row = t[t["flagged_share"] == 0.2].iloc[0]
    assert abs(row["precision"] - ref["precision_top"]) < 1e-12
    assert abs(row["f1"] - ref["f1_top"]) < 1e-12
    assert t["recall"].is_monotonic_increasing


def test_metrics_by_cutoff_skips_single_class_cutoffs():
    test = pd.DataFrame(
        {
            "cutoff": pd.to_datetime(["2024-07-01"] * 4 + ["2024-08-01"] * 3),
            "churned": [1, 0, 1, 0, 1, 1, 1],
        }
    )
    out = metrics_by_cutoff(test, [0.9, 0.2, 0.8, 0.1, 0.5, 0.5, 0.5])
    assert len(out) == 1 and out.loc[0, "roc_auc"] == 1.0 and out.loc[0, "rows"] == 4


def test_logistic_effects_find_the_planted_feature_and_are_reproducible():
    train = fake_rows(3000, 8)
    one = logistic_effects(train, n_boot=30, seed=5)
    two = logistic_effects(train, n_boot=30, seed=5)
    pd.testing.assert_frame_equal(one, two)
    assert sorted(one["feature"]) == sorted(design(train).columns)
    r = one[one["feature"] == "recency_days"].iloc[0]
    assert r["odds_ratio_per_sd"] > 1 and r["ci_low"] > 1
    assert (one["ci_low"] <= one["ci_high"]).all()