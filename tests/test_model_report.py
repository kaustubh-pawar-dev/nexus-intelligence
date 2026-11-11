import numpy as np
import pandas as pd
import pytest

from nexus.db import get_engine
from nexus.model_report import backtest_intervals, load_report, targeting_backtest


def two_cutoffs() -> tuple[pd.DataFrame, list]:
    test = pd.DataFrame(
        {
            "customer_id": list("abcdefgh"),
            "cutoff": pd.to_datetime(["2024-07-01"] * 4 + ["2024-08-01"] * 4),
            "churned": [1, 1, 1, 0, 0, 1, 1, 1],
            "monetary_180": [10, 10, 100, 200, 50, 50, 50, 50],
        }
    )
    return test, [0.9, 0.8, 0.2, 0.1, 0.7, 0.6, 0.5, 0.4]


def test_backtest_by_hand():
    test, p = two_cutoffs()
    b = targeting_backtest(test, p, share=0.25)  # the top 1 of 4 customers at each cutoff
    per = b["per_cutoff"]
    assert per["rows"].tolist() == [4, 4]
    assert per["by_probability"].round(2).tolist() == [8.33, 0.0]  # customer a lost 10 of 120; customer e lost 0 of 150
    assert per["by_expected_loss"].round(2).tolist() == [83.33, 0.0]  # customer c lost 100 of 120; customer e again
    assert per["random_list"].tolist() == [25.0, 25.0]
    assert b["pooled"]["by_probability"] == pytest.approx(100 * 10 / 270)
    assert b["pooled"]["by_expected_loss"] == pytest.approx(100 * 100 / 270)
    assert b["pooled"]["random_list"] == 25.0


def test_a_cutoff_without_any_churn_gives_nan_for_itself_but_does_not_distort_the_pool():
    test, p = two_cutoffs()
    test.loc[4:, "churned"] = 0  # nobody churned at the second cutoff
    b = targeting_backtest(test, p, share=0.25)
    assert b["per_cutoff"]["by_probability"].isna().tolist() == [False, True]
    assert b["pooled"]["by_expected_loss"] == pytest.approx(100 * 100 / 120)


def planted_gap_data():
    rng = np.random.default_rng(1)
    n_customers, cutoffs = 300, 3
    spend = rng.lognormal(mean=4.0, sigma=1.2, size=n_customers)
    rows = []
    for c in range(cutoffs):
        rows.append(
            pd.DataFrame(
                {
                    "customer_id": np.arange(n_customers).astype(str),
                    "cutoff": pd.Timestamp("2024-07-01") + pd.DateOffset(months=c),
                    "churned": (rng.random(n_customers) < 0.4).astype(int),
                    "monetary_180": spend,
                }
            )
        )
    test = pd.concat(rows, ignore_index=True)
    return test, rng.random(len(test))  # the score is pure noise, so only spend can make a ranking useful


def test_backtest_intervals_are_reproducible_ordered_and_detect_a_real_gap():
    test, noise = planted_gap_data()
    one = backtest_intervals(test, noise, n_boot=60, seed=3)
    two = backtest_intervals(test, noise, n_boot=60, seed=3)
    pd.testing.assert_frame_equal(one, two)
    assert (one["ci_low"] <= one["ci_high"]).all()
    gap = one[one["rule"] == "Expected loss minus probability"].iloc[0]
    assert gap["ci_low"] > 0  # ranking by probability x spend finds the valuable churners even when the probability is noise
    assert one[one["rule"] == "Top list by churn probability"].iloc[0]["estimate"] == pytest.approx(20.0, abs=6.0)  # about random


# ---- on the database ----

@pytest.fixture(scope="module")
def report():
    return load_report(get_engine())


def test_split_sizes_and_headline_metrics_reproduce_the_documented_results(report):
    assert report["train_rows"] == 5368 and report["test_rows"] == 13794
    lr = report["comparison"].set_index("model").loc["logistic regression"]
    assert lr["roc_auc"] == pytest.approx(0.735, abs=0.002) and lr["pr_auc"] == pytest.approx(0.605, abs=0.002)
    assert report["calibration_gap"] == pytest.approx(0.032, abs=0.003)
    ci = report["intervals"].set_index(["comparison", "metric"]).loc[("logistic regression", "roc_auc")]
    assert ci["ci_low"] == pytest.approx(0.718, abs=0.003) and ci["ci_high"] == pytest.approx(0.752, abs=0.003)


def test_logistic_regression_beats_the_recency_rule_and_gradient_boosting_does_not_beat_it(report):
    d = report["intervals"].set_index(["comparison", "metric"])
    rule = d.loc[("rule: long recency minus logistic regression", "roc_auc")]
    boost = d.loc[("gradient boosting minus logistic regression", "pr_auc")]
    assert rule["ci_high"] < 0  # the simple rule is clearly worse
    assert boost["ci_high"] < 0  # boosting is clearly worse on PR-AUC
    assert d.loc[("gradient boosting minus logistic regression", "roc_auc")]["ci_low"] < 0 < d.loc[("gradient boosting minus logistic regression", "roc_auc")]["ci_high"]


def test_expected_loss_ranking_beats_probability_ranking_in_every_cutoff_and_overall(report):
    per = report["backtest"]["per_cutoff"]
    assert len(per) == 13
    assert (per["by_expected_loss"] > per["by_probability"]).all()
    assert (per["by_expected_loss"] > per["random_list"]).all()
    gap = report["backtest_intervals"].set_index("rule").loc["Expected loss minus probability"]
    assert gap["ci_low"] > 0