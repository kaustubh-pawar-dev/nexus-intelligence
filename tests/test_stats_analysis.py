import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from nexus.db import get_engine
from nexus.sql_analytics import run_query
from nexus.stats_analysis import (
    ALPHA,
    RESULT_COLUMNS,
    cliffs_delta,
    holm_adjust,
    kruskal_epsilon,
    load_tables,
    mann_whitney,
    profile_frame,
    refund_frame_by_defect,
    run_hypothesis_tests,
    seasonal_index,
    two_proportions,
)


def test_cliffs_delta_known_values():
    assert cliffs_delta([1, 2, 3], [4, 5, 6]) == -1.0
    assert cliffs_delta([4, 5, 6], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def test_mann_whitney_detects_a_shift():
    rng = np.random.default_rng(0)
    r = mann_whitney(rng.normal(0, 1, 500), rng.normal(1, 1, 500))
    assert r["p_value"] < 1e-6
    assert -0.65 < r["cliffs_delta"] < -0.40
    same = mann_whitney(rng.normal(0, 1, 500), rng.normal(0, 1, 500))
    assert abs(same["cliffs_delta"]) < 0.12


def test_kruskal_epsilon_separates_and_mixes():
    separated = kruskal_epsilon([range(0, 10), range(10, 20), range(20, 30)])
    assert separated["epsilon_squared"] > 0.8 and separated["p_value"] < 1e-4
    rng = np.random.default_rng(1)
    mixed = kruskal_epsilon([rng.normal(size=200) for _ in range(3)])
    assert mixed["epsilon_squared"] < 0.02


def test_two_proportions_known_risk_ratio():
    r = two_proportions(100, 1000, 50, 1000)
    assert abs(r["risk_ratio"] - 2.0) < 1e-9
    assert r["p_value"] < 1e-4
    assert r["ci_low"] < 2.0 < r["ci_high"]
    flat = two_proportions(50, 1000, 50, 1000)
    assert flat["risk_ratio"] == 1.0 and flat["p_value"] > 0.9


def test_holm_adjust_known_values():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])
    assert (holm_adjust([0.5, 0.9]) <= 1.0).all()


def test_seasonal_index_removes_growth_between_years():
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    revenue = np.where(dates.month == 11, 2.0, 1.0) * np.where(dates.year == 2024, 3.0, 1.0)
    idx = seasonal_index(pd.DataFrame({"date": dates, "gross_revenue": revenue}))
    assert idx.idxmax() == 11
    assert abs(idx.sum() - 12) < 0.01


def test_profile_frame_reports_missing_skew_and_outliers():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 100], "b": ["x", None, "y", "x", "x"]})
    p = profile_frame(df, "demo").set_index("column")
    assert p.loc["b", "missing"] == 1 and p.loc["b", "unique"] == 2
    assert p.loc["a", "outlier_pct"] == 20.0
    assert p.loc["a", "skew"] > 1
    assert np.isnan(p.loc["b", "mean"])


@pytest.fixture(scope="module")
def engine():
    try:
        eng = get_engine()
        with eng.connect() as conn:
            ready = conn.execute(
                text("SELECT to_regclass('analytics.orders') IS NOT NULL AND to_regclass('analytics.daily_metrics') IS NOT NULL")
            ).scalar()
    except Exception as exc:
        pytest.skip(f"PostgreSQL is not reachable: {type(exc).__name__}")
    if not ready:
        pytest.skip("run scripts/build_analytics.py first")
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def tables(engine):
    return load_tables(engine)


@pytest.fixture(scope="module")
def cust(engine):
    return run_query(engine, "customer_level")


@pytest.fixture(scope="module")
def results(tables, cust):
    return run_hypothesis_tests(tables, cust)


def test_customer_level_reconciles_with_the_tables(cust, tables, engine):
    assert len(cust) == len(tables["customers"]) == 5000
    assert cust["customer_id"].is_unique
    completed = tables["orders"].loc[tables["orders"]["order_status"] == "Completed", "order_id"]
    revenue = tables["order_items"].loc[tables["order_items"]["order_id"].isin(completed), "line_total"].sum()
    assert abs(cust["revenue"].sum() - revenue) < 0.05
    seg = run_query(engine, "contribution_by_segment")
    assert abs(cust["contribution"].sum() - seg["contribution"].sum()) < 0.05
    assert (cust["orders"] == 0).sum() == len(cust) - cust.loc[cust["orders"] > 0, "customer_id"].nunique()


def test_results_table_is_well_formed(results):
    assert list(results.columns) == RESULT_COLUMNS
    assert list(results["id"]) == [f"T{i}" for i in range(1, 8)]
    assert (results["p_holm"] >= results["p_value"] - 1e-15).all()
    assert results["p_holm"].between(0, 1).all()
    assert set(results["design"]) == {"built in", "built in (small)", "not built in"}
    assert results["effect"].notna().all()


def test_results_are_consistent_with_the_simulation_design(results):
    assert results["consistent_with_design"].all(), results[~results["consistent_with_design"]][["id", "effect", "p_holm", "verdict"]]
    built_in = results[results["design"] == "built in"]
    assert built_in["significant"].all()
    null = results[results["design"] == "not built in"]
    assert not null["practical"].any()


def test_refund_frame_reconciles_with_the_tables(tables):
    f = refund_frame_by_defect(tables)
    captured = (tables["payments"]["payment_status"] == "Captured").sum()
    assert len(f) == captured
    assert f["refunded"].sum() == len(tables["refunds"])
    assert f["band"].value_counts().min() > 0.9 * len(f) / 3
    assert f["score"].between(0, 0.1).all()


def test_seasonal_index_on_real_data_shows_a_november_peak(tables):
    idx = seasonal_index(tables["daily_metrics"])
    assert len(idx) == 12
    assert abs(idx.sum() - 12) < 0.01
    assert idx[11] > 1.2 * idx[2]