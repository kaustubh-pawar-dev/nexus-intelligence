import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from nexus.churn import (
    DATA_END,
    FEATURE_COLUMNS,
    HORIZON_DAYS,
    TABLES_NEEDED,
    TEST_FIRST_CUTOFF,
    TRAIN_LAST_CUTOFF,
    baseline_scores,
    baseline_table,
    build_cutoff,
    build_dataset,
    cutoff_dates,
    design,
    evaluate,
    order_frame,
    split_by_cutoff,
)
from nexus.db import get_engine
from nexus.stats_analysis import load_tables

T = pd.Timestamp
CUTOFF = T("2023-09-01")


def orders_frame(rows):
    cols = ["order_id", "customer_id", "order_date", "value", "refund_date", "delivered_date", "delivery_delay_days"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ("order_date", "refund_date", "delivered_date"):
        df[c] = pd.to_datetime(df[c])
    return df


def customers_frame():
    return pd.DataFrame(
        {
            "customer_id": ["A", "B", "C"],
            "signup_date": [T("2023-01-01")] * 3,
            "segment": ["Consumer", "Enterprise", "Consumer"],
            "channel": ["Direct", "Referral", "Email"],
            "marketing_opt_in": [True, False, True],
        }
    )


ROWS = [
    ("o1", "A", T("2023-05-01"), 100.0, pd.NaT, pd.NaT, np.nan),
    ("o2", "A", T("2023-08-01"), 50.0, T("2023-08-20"), T("2023-08-05"), 2.0),
    ("o3", "A", T("2023-10-01"), 10.0, pd.NaT, pd.NaT, np.nan),
    ("o4", "B", T("2023-06-01"), 30.0, T("2023-09-15"), T("2023-09-10"), 3.0),
    ("o5", "C", T("2023-01-01"), 20.0, pd.NaT, pd.NaT, np.nan),
]


def test_cutoff_dates_cover_the_window_and_fit_the_data():
    c = cutoff_dates()
    assert c[0] == T("2023-07-01") and c[-1] == T("2025-07-01") and len(c) == 25
    assert all(x + pd.Timedelta(days=HORIZON_DAYS - 1) <= DATA_END for x in c)
    assert all(x.day == 1 for x in c)


def test_split_has_a_gap_and_no_overlap():
    assert TRAIN_LAST_CUTOFF + pd.Timedelta(days=HORIZON_DAYS) <= TEST_FIRST_CUTOFF
    df = pd.DataFrame({"cutoff": cutoff_dates(), "churned": 0})
    train, test = split_by_cutoff(df)
    assert len(train) == 7 and len(test) == 13
    assert train["cutoff"].max() < test["cutoff"].min()


def test_features_and_label_on_handcrafted_data():
    out = build_cutoff(orders_frame(ROWS), customers_frame(), CUTOFF).set_index("customer_id")
    assert set(out.index) == {"A", "B"}
    a, b = out.loc["A"], out.loc["B"]
    assert (a["frequency_180"], a["monetary_180"], a["recency_days"], a["avg_order_value"]) == (2, 150.0, 31, 75.0)
    assert (a["refunds_180"], a["late_orders_180"], a["orders_total"], a["tenure_days"]) == (1, 1, 2, 123)
    assert a["churned"] == 0
    assert (b["frequency_180"], b["recency_days"], b["churned"]) == (1, 92, 1)


def test_nothing_dated_after_the_cutoff_leaks_into_features():
    full = build_cutoff(orders_frame(ROWS), customers_frame(), CUTOFF).set_index("customer_id")
    assert full.loc["B", "refunds_180"] == 0 and full.loc["B", "late_orders_180"] == 0
    no_future = build_cutoff(orders_frame([r for r in ROWS if r[0] != "o3"]), customers_frame(), CUTOFF).set_index("customer_id")
    pd.testing.assert_frame_equal(full[FEATURE_COLUMNS], no_future[FEATURE_COLUMNS])
    assert full.loc["A", "churned"] == 0 and no_future.loc["A", "churned"] == 1


def test_customers_without_a_recent_order_are_not_in_the_population():
    out = build_cutoff(orders_frame(ROWS), customers_frame(), CUTOFF)
    assert "C" not in set(out["customer_id"])
    assert build_cutoff(orders_frame(ROWS), customers_frame(), T("2023-02-01")).empty is False


def test_design_matrix_is_numeric_and_complete():
    out = build_cutoff(orders_frame(ROWS), customers_frame(), CUTOFF)
    x = design(out)
    assert x.shape == (2, 4 + 5 + 1 + 2 + 5)
    assert np.isfinite(x.to_numpy()).all()
    assert set(x.columns) >= {"log_monetary_180", "recency_days", "opt_in", "segment: Enterprise", "channel: Referral"}


def test_evaluate_known_and_random_scores():
    perfect = evaluate([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], top_share=0.5)
    assert perfect["roc_auc"] == 1.0 and perfect["pr_auc"] == 1.0
    assert perfect["precision_top"] == 1.0 and perfect["recall_top"] == 1.0 and perfect["lift_top"] == 2.0
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.3).astype(int)
    r = evaluate(y, rng.random(20000))
    assert abs(r["roc_auc"] - 0.5) < 0.03
    assert abs(r["pr_auc"] - r["prevalence"]) < 0.03
    assert abs(r["lift_top"] - 1.0) < 0.1


@pytest.fixture(scope="module")
def tables():
    try:
        eng = get_engine()
        with eng.connect() as conn:
            ready = conn.execute(text("SELECT to_regclass('analytics.orders') IS NOT NULL")).scalar()
    except Exception as exc:
        pytest.skip(f"PostgreSQL is not reachable: {type(exc).__name__}")
    if not ready:
        pytest.skip("run scripts/build_analytics.py first")
    t = load_tables(eng, TABLES_NEEDED)
    eng.dispose()
    return t


@pytest.fixture(scope="module")
def data(tables):
    return build_dataset(tables)


def test_dataset_shape_keys_and_cutoffs(data):
    assert data.groupby(["customer_id", "cutoff"]).size().max() == 1
    assert set(data["cutoff"].unique()) == set(cutoff_dates())
    assert set(data["churned"].unique()) == {0, 1}
    assert data.groupby("cutoff").size().min() >= 100
    assert data[FEATURE_COLUMNS].notna().all().all()
    assert (data["recency_days"].between(1, HORIZON_DAYS)).all()


def test_churn_rate_is_plausible_and_counts_reconcile(data, tables):
    assert 0.2 < data["churned"].mean() < 0.95
    buyers = tables["orders"].loc[tables["orders"]["order_status"] == "Completed", "customer_id"].nunique()
    assert data["customer_id"].nunique() <= buyers
    assert (data["frequency_180"] >= 1).all() and (data["monetary_180"] > 0).all()


def test_rebuilding_from_data_truncated_at_the_cutoff_gives_the_same_features(tables):
    orders = order_frame(tables)
    cust = tables["customers"].rename(columns={"acquisition_channel": "channel"})
    cutoff = T("2024-07-01")
    full = build_cutoff(orders, cust, cutoff).set_index("customer_id").sort_index()
    cut = orders[orders["order_date"] < cutoff].copy()
    cut.loc[cut["refund_date"] >= cutoff, "refund_date"] = pd.NaT
    cut.loc[cut["delivered_date"] >= cutoff, "delivered_date"] = pd.NaT
    trunc = build_cutoff(cut, cust, cutoff).set_index("customer_id").sort_index()
    assert list(full.index) == list(trunc.index)
    pd.testing.assert_frame_equal(full[FEATURE_COLUMNS], trunc[FEATURE_COLUMNS])


def test_baselines_rank_sensibly_on_unseen_cutoffs(data):
    train, test = split_by_cutoff(data)
    assert len(train) > 1000 and len(test) > 1000
    scores = baseline_scores(train, test)
    assert all(len(s) == len(test) and np.isfinite(s).all() for s in scores.values())
    assert 0 <= scores["logistic regression"].min() and scores["logistic regression"].max() <= 1
    table = baseline_table(train, test).set_index("model")
    assert abs(table.loc["random", "roc_auc"] - 0.5) < 0.05
    assert table.loc["logistic regression", "roc_auc"] > 0.60
    assert table.loc["logistic regression", "roc_auc"] > table.loc["random", "roc_auc"] + 0.05
    assert table.loc["logistic regression", "brier"] < table.loc["reference: constant train rate", "brier"]