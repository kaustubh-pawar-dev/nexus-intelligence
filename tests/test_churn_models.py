import numpy as np
import pandas as pd

from nexus.churn import CHANNELS, SEGMENTS, design
from nexus.churn_models import (
    bootstrap_intervals,
    comparison_table,
    customer_blocks,
    evaluate_full,
    fit_gradient_boosting,
    model_scores,
)


def fake_rows(n: int, seed: int, customers: int = 400) -> pd.DataFrame:
    """Synthetic churn rows where churn depends on recency, and on frequency only through an interaction."""
    rng = np.random.default_rng(seed)
    recency = rng.integers(1, 180, n).astype(float)
    freq = rng.integers(1, 12, n).astype(float)
    p = 1 / (1 + np.exp(-((recency - 90) / 25 - 0.8 * (freq > 6) * (recency < 60))))
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


def test_f1_matches_hand_calculation():
    y = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    score = [10, 9, 8, 1, 1, 1, 1, 1, 1, 1]
    out = evaluate_full(y, score, top_share=0.2)  # top 2 rows: one churner of two
    assert out["precision_top"] == 0.5
    assert out["recall_top"] == 0.5
    assert abs(out["f1_top"] - 0.5) < 1e-12


def test_gradient_boosting_learns_a_planted_signal_and_is_deterministic():
    train, test = fake_rows(4000, 1), fake_rows(2000, 2)
    a = fit_gradient_boosting(train, seed=7).predict_proba(design(test))[:, 1]
    b = fit_gradient_boosting(train, seed=7).predict_proba(design(test))[:, 1]
    assert np.array_equal(a, b)
    assert evaluate_full(test["churned"], a)["roc_auc"] > 0.75


def test_comparison_table_has_all_models_and_brier_only_for_probabilities():
    train, test = fake_rows(3000, 3), fake_rows(1500, 4)
    table = comparison_table(test, model_scores(train, test))
    assert list(table["model"]) == ["rule: long recency", "logistic regression", "gradient boosting"]
    assert table.loc[table["model"] == "rule: long recency", "brier"].isna().all()
    assert table.loc[table["model"] != "rule: long recency", "brier"].between(0, 1).all()


def test_customer_blocks_partition_every_row_once():
    ids = np.array(["a", "b", "a", "c", "b", "a"])
    blocks = customer_blocks(ids)
    assert sorted(np.concatenate(blocks).tolist()) == list(range(len(ids)))
    assert sorted(len(b) for b in blocks) == [1, 2, 3]
    for b in blocks:
        assert len(set(ids[b])) == 1


def test_bootstrap_intervals_are_ordered_reproducible_and_detect_a_real_gap():
    test = fake_rows(3000, 5)
    rng = np.random.default_rng(0)
    scores = {
        "logistic regression": rng.random(len(test)),  # pure noise as the reference
        "informative": test["recency_days"].to_numpy(float),
    }
    one = bootstrap_intervals(test, scores, n_boot=60, seed=11)
    two = bootstrap_intervals(test, scores, n_boot=60, seed=11)
    pd.testing.assert_frame_equal(one, two)
    assert (one["ci_low"] <= one["ci_high"]).all()
    gap = one[(one["comparison"] == "informative minus logistic regression") & (one["metric"] == "roc_auc")]
    assert len(gap) == 1 and gap["ci_low"].iloc[0] > 0