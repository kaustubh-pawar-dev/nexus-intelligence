import math

import numpy as np
import pandas as pd
import pytest

from nexus.dashboard_data import fetch_kpis
from nexus.db import get_engine
from nexus.findings import (
    CALIBRATION_SHIFT,
    build_findings,
    capture,
    component_table,
    expected_loss,
    finding_concentration,
    finding_lapsed,
    finding_margin,
    finding_refunds,
    finding_retention,
    summary_table,
)
from nexus.findings_data import category_components, findings_from_database, load_inputs


def five_customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": list("abcde"),
            "churn_probability": [0.9, 0.8, 0.5, 0.2, 0.1],
            "monetary_180": [10.0, 20.0, 100.0, 500.0, 1000.0],
            "risk_band": ["High", "High", "Medium", "Low", "Low"],
            "segment": ["Consumer", "Consumer", "Small Business", "Enterprise", "Enterprise"],
        }
    )


def rfm_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rfm_segment": ["Champions", "At risk", "Dormant", "New / promising"],
            "customers": [932, 570, 934, 570],
            "avg_recency_days": [57.3, 501.1, 637.0, 73.5],
            "avg_orders": [12.3, 8.0, 1.6, 1.4],
            "avg_revenue": [10072.7, 4291.8, 306.4, 381.9],
            "revenue": [9387742.0, 2446339.0, 286166.2, 217706.8],
            "revenue_share_pct": [76.1, 19.8, 2.3, 1.8],
        }
    )


def components(a, b, c) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "revenue": [a[0], b[0], c[0]],
            "cogs": [a[1], b[1], c[1]],
            "shipping": [a[2], b[2], c[2]],
            "refunds": [a[3], b[3], c[3]],
        }
    )


CUR = components((1000, 700, 50, 40), (1000, 600, 50, 30), (1000, 600, 40, 20))


def test_expected_loss_and_the_shift_is_clipped_to_valid_probabilities():
    s = five_customers()
    assert expected_loss(s).round(6).tolist() == [9.0, 16.0, 50.0, 100.0, 100.0]
    shifted = expected_loss(s, 0.2)
    assert shifted.iloc[0] == pytest.approx(1.0 * 10.0)  # 0.9 + 0.2 is clipped to 1
    assert expected_loss(s, -0.5).iloc[4] == 0.0  # 0.1 - 0.5 is clipped to 0


def test_capture_compares_the_two_rankings_by_hand():
    c = capture(five_customers(), top_share=0.4)
    assert c["k"] == 2
    assert c["by_probability"]["loss_share"] == pytest.approx(100 * 25 / 275)
    assert c["by_probability"]["spend_share"] == pytest.approx(100 * 30 / 1630)
    assert c["by_probability"]["top_segment"] == "Consumer"
    assert c["by_expected_loss"]["loss_share"] == pytest.approx(100 * 200 / 275)
    assert c["by_expected_loss"]["spend_share"] == pytest.approx(100 * 1500 / 1630)
    assert c["by_expected_loss"]["top_segment"] == "Enterprise"


def test_retention_finding_numbers_match_the_inputs():
    f = finding_retention(five_customers(), "01 Jan 2026")
    assert f.kind == "Prediction" and "alone misses most" in f.title
    assert f.exposure == pytest.approx(275.0)
    assert f.exposure_low == pytest.approx(expected_loss(five_customers(), -CALIBRATION_SHIFT).sum())
    assert f.exposure_low < f.exposure < f.exposure_high
    assert "top 1 customers by churn probability account for 3.3% of expected loss" in f.statement  # 20% of 5 customers is 1
    assert "account for 36.4%" in f.statement  # the single highest expected loss is 100 of 275
    assert dict(f.evidence)["Share of 180-day spend held by High-risk customers"] == "1.8%"
    assert f.assumptions and f.limits and f.recommendation and f.evaluation


def test_customer_value_findings_by_hand_and_skipped_when_the_segment_is_absent():
    lapsed = finding_lapsed(rfm_table())
    assert lapsed.exposure is None and lapsed.kind == "Observation"
    assert "570 customers (19% of customers" in lapsed.statement and "501 days" in lapsed.statement and "19.8%" in lapsed.statement
    assert dict(lapsed.evidence)["Revenue to date"].startswith("2,446,339")
    champions = finding_concentration(rfm_table())
    assert "932 Champions (31% of customers" in champions.statement and "76.1%" in champions.statement
    assert finding_lapsed(rfm_table().iloc[[0, 2, 3]]) is None
    assert finding_concentration(rfm_table().assign(revenue_share_pct=[40.0, 30.0, 20.0, 10.0])) is None


def test_margin_finding_decomposes_the_gap_by_hand():
    t = component_table(CUR).set_index("category")
    assert t.loc["A", "contribution_pct"] == pytest.approx(21.0) and t.loc["B", "contribution_pct"] == pytest.approx(32.0)
    f = finding_margin(CUR, "Jan 2025 to Dec 2025")
    assert f.title.startswith("A earns a lower") and "21.0%" in f.statement and "33.0%" in f.statement
    assert "12.0-point gap, gross margin accounts for +10.0 points, shipping for +0.5 and refunds for +1.5" in f.factors
    assert f.exposure == pytest.approx(1000 * 12 / 100)
    assert "supplier costs" in f.recommendation  # gross margin is the largest driver
    assert finding_margin(components((1000, 600, 50, 30), (1000, 600, 50, 30), (1000, 600, 50, 30)), "x") is None  # no gap
    small = components((50, 40, 5, 4), (1000, 600, 50, 30), (1000, 600, 40, 20))  # A is under 10% of revenue
    assert finding_margin(small, "x") is None or not finding_margin(small, "x").title.startswith("A ")


def test_refund_finding_values():
    prev = components((800, 500, 40, 16), (800, 500, 40, 16), (800, 500, 40, 16))  # 2.0% everywhere
    cur = components((1000, 600, 50, 40), (1000, 600, 50, 20), (1000, 600, 50, 20))  # 4.0%, 2.0%, 2.0%
    f = finding_refunds(cur, prev, "2025", "2024")
    assert f.title == "The refund rate rose by 0.7 points"
    assert "2.67% of revenue in 2025, against 2.00% in 2024 (+0.67 points)" in f.statement
    assert f.exposure == pytest.approx(0.0066667 * 3000, rel=1e-3)
    assert "A (+20)" in f.factors
    assert finding_refunds(cur, None, "2025", "2024") is None
    assert finding_refunds(prev, prev, "2025", "2024") is None  # no change
    fell = finding_refunds(prev, cur, "2025", "2024")
    assert fell.title.startswith("The refund rate fell") and fell.exposure_label.endswith("below the previous-period rate")


def test_build_findings_skips_what_cannot_be_built_and_summarises():
    prev = components((800, 500, 40, 16), (800, 500, 40, 16), (800, 500, 40, 16))
    all_five = build_findings(five_customers(), rfm_table(), CUR, prev, "2025", "2024", "01 Jan 2026")
    assert [f.key for f in all_five] == ["retention_targeting", "category_margin", "refund_rate", "lapsed_value", "revenue_concentration"]
    fewer = build_findings(five_customers(), rfm_table().iloc[[1, 2, 3]], CUR, None, "2025", "2024", "01 Jan 2026")
    assert [f.key for f in fewer] == ["retention_targeting", "category_margin", "lapsed_value"]
    table = summary_table(all_five)
    assert list(table.columns)[:4] == ["Finding", "Type", "Exposure (estimate)", "What it measures"]
    assert table.loc[3, "Exposure (estimate)"] == "n/a" and "to" in table.loc[0, "Exposure (estimate)"]


# ---- reconciliation against the database ----

@pytest.fixture(scope="module")
def engine():
    return get_engine()


@pytest.fixture(scope="module")
def built(engine):
    return findings_from_database(engine)


def test_category_components_add_up_to_the_kpi_layer(engine):
    window = load_inputs(engine)["window"]
    comp = category_components(engine, *window)
    k = fetch_kpis(engine, *window)
    assert abs(comp["revenue"].sum() - k.revenue) < 0.05
    assert abs(comp["cogs"].sum() - k.cogs) < 0.05
    assert abs(comp["shipping"].sum() - k.shipping) < 0.05
    assert abs(comp["refunds"].sum() - k.refunds) < 0.05


def test_findings_are_built_from_the_real_data_with_consistent_numbers(built):
    findings, inputs = built
    keys = [f.key for f in findings]
    assert len(findings) >= 3 and keys[0] == "retention_targeting"
    scores = inputs["scores"]
    independent = float((scores["churn_probability"].to_numpy() * scores["monetary_180"].to_numpy()).sum())
    retention = findings[0]
    assert retention.exposure == pytest.approx(independent)
    assert retention.exposure_low < retention.exposure < retention.exposure_high
    c = capture(scores)
    assert c["by_expected_loss"]["loss_share"] >= c["by_probability"]["loss_share"]  # true by construction
    assert all(math.isfinite(f.exposure) and f.exposure >= 0 for f in findings if f.exposure is not None)
    assert all(f.assumptions and f.limits and f.recommendation and f.evaluation for f in findings)