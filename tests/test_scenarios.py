import math

import pandas as pd
import pytest

from nexus.dashboard_data import fetch_kpis
from nexus.db import get_engine
from nexus.findings import expected_loss
from nexus.findings_data import findings_from_database, load_inputs
from nexus.scenarios import adjust, campaign, campaign_sensitivity, cogs_scenario, refund_scenario, totals


def five_customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "churn_probability": [0.9, 0.8, 0.5, 0.2, 0.1],
            "monetary_180": [10.0, 20.0, 100.0, 500.0, 1000.0],
        }
    )


def components() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "revenue": [1000.0, 1000.0, 1000.0],
            "cogs": [700.0, 600.0, 600.0],
            "shipping": [50.0, 50.0, 40.0],
            "refunds": [40.0, 30.0, 20.0],
        }
    )


def test_campaign_by_hand_for_both_ranking_rules():
    p = campaign(five_customers(), 2, "probability", 0.2, 1.0, 0.25)  # contacts customers a and b
    assert p["contacts"] == 2 and p["mean_probability"] == pytest.approx(0.85) and p["spend"] == pytest.approx(30.0)
    assert p["expected_loss"] == pytest.approx(25.0) and p["spend_retained"] == pytest.approx(5.0)
    assert p["contribution_retained"] == pytest.approx(1.25) and p["cost"] == pytest.approx(2.0)
    assert p["net_benefit"] == pytest.approx(-0.75) and p["return_on_cost"] == pytest.approx(-0.375)
    assert p["break_even_effect"] == pytest.approx(2 / (25 * 0.25))
    v = campaign(five_customers(), 2, "expected_loss", 0.2, 1.0, 0.25)  # contacts customers d and e
    assert v["spend"] == pytest.approx(1500.0) and v["mean_probability"] == pytest.approx(0.15)
    assert v["expected_loss"] == pytest.approx(200.0) and v["net_benefit"] == pytest.approx(8.0)
    assert v["return_on_cost"] == pytest.approx(4.0) and v["break_even_effect"] == pytest.approx(0.04)


def test_the_campaign_breaks_even_exactly_at_the_break_even_effect():
    r = campaign(five_customers(), 2, "expected_loss", 0.5, 1.0, 0.25)
    at_break_even = campaign(five_customers(), 2, "expected_loss", r["break_even_effect"], 1.0, 0.25)
    assert at_break_even["net_benefit"] == pytest.approx(0.0, abs=1e-9)
    free = campaign(five_customers(), 2, "probability", 0.2, 0.0, 0.25)
    assert free["return_on_cost"] is None and free["break_even_effect"] == pytest.approx(0.0)


def test_campaign_validation_and_caps():
    s = five_customers()
    assert campaign(s, 99, "probability", 0.1, 1.0, 0.2)["contacts"] == 5  # capped at the customers available
    for bad in (
        lambda: campaign(s, 2, "probability", 1.5, 1.0, 0.2),
        lambda: campaign(s, 2, "probability", -0.1, 1.0, 0.2),
        lambda: campaign(s, 0, "probability", 0.1, 1.0, 0.2),
        lambda: campaign(s, 2, "probability", 0.1, -1.0, 0.2),
        lambda: campaign(s, 2, "probability", 0.1, 1.0, 1.5),
        lambda: campaign(s, 2, "alphabetical", 0.1, 1.0, 0.2),
    ):
        with pytest.raises(ValueError):
            bad()


def test_sensitivity_agrees_with_the_campaign_and_zero_effect_only_costs_money():
    s = five_customers()
    grid = campaign_sensitivity(s, 2, 1.0, 0.25)
    assert len(grid) == 13 and grid["effect"].iloc[0] == 0.0 and grid["effect"].iloc[-1] == pytest.approx(0.6)
    assert grid.loc[0, "net_probability"] == pytest.approx(-2.0) and grid.loc[0, "net_expected_loss"] == pytest.approx(-2.0)
    row = grid[grid["effect"] == 0.2].iloc[0]
    assert row["net_probability"] == pytest.approx(campaign(s, 2, "probability", 0.2, 1.0, 0.25)["net_benefit"])
    assert row["net_expected_loss"] == pytest.approx(campaign(s, 2, "expected_loss", 0.2, 1.0, 0.25)["net_benefit"])
    assert grid["net_expected_loss"].is_monotonic_increasing and grid["net_probability"].is_monotonic_increasing


def test_baseline_totals_and_the_refund_scenario_by_hand():
    t = totals(components())
    assert t["revenue"] == 3000 and t["cogs"] == 1900 and t["shipping"] == 140 and t["refunds"] == 90
    assert t["contribution"] == 870 and t["contribution_pct"] == pytest.approx(29.0)
    r = refund_scenario(components(), ["A"], 50)
    assert r["scenario"]["refunds"] == pytest.approx(70.0) and r["delta_contribution"] == pytest.approx(20.0)
    assert r["scenario"]["contribution_pct"] == pytest.approx(890 / 30) and r["delta_points"] == pytest.approx(20 / 30)
    by = r["by_category"].set_index("category")
    assert by.loc["A", ["baseline", "scenario", "change"]].tolist() == [40.0, 20.0, -20.0] and by.loc["B", "change"] == 0.0


def test_cost_of_goods_scenario_by_hand_in_both_directions():
    down = cogs_scenario(components(), ["A", "B", "C"], -10)
    assert down["scenario"]["cogs"] == pytest.approx(1710.0) and down["delta_contribution"] == pytest.approx(190.0)
    assert down["scenario"]["contribution_pct"] == pytest.approx(1060 / 30) and down["delta_points"] == pytest.approx(190 / 30)
    up = cogs_scenario(components(), ["B"], 10)
    assert up["delta_contribution"] == pytest.approx(-60.0)
    assert up["by_category"].set_index("category").loc["B", "scenario"] == pytest.approx(660.0)


def test_no_change_reproduces_the_baseline_and_the_input_is_never_modified():
    comp = components()
    before = comp.copy()
    for res in (refund_scenario(comp, ["A", "B", "C"], 0), cogs_scenario(comp, ["A", "B", "C"], 0)):
        assert res["delta_contribution"] == 0 and res["delta_points"] == 0
    refund_scenario(comp, ["A"], 50)
    cogs_scenario(comp, ["B"], 10)
    pd.testing.assert_frame_equal(comp, before)
    with pytest.raises(ValueError):
        refund_scenario(comp, ["A"], 120)
    with pytest.raises(ValueError):
        cogs_scenario(comp, ["A"], -100)
    with pytest.raises(ValueError):
        adjust(comp, ["A"], "revenue", 1.0)


# ---- reconciliation against the database ----

@pytest.fixture(scope="module")
def inputs():
    return load_inputs(get_engine())


def test_scenario_baseline_matches_the_kpi_layer(inputs):
    k = fetch_kpis(get_engine(), *inputs["window"])
    t = totals(inputs["comp_cur"])
    assert abs(t["revenue"] - k.revenue) < 0.05 and abs(t["contribution"] - k.contribution) < 0.05
    assert abs(t["contribution_pct"] - k.contribution_pct) < 0.001


def test_contacting_everyone_for_free_retains_exactly_the_exposure_in_the_decision_findings(inputs):
    scores = inputs["scores"]
    everyone = campaign(scores, len(scores), "expected_loss", 1.0, 0.0, 1.0)
    findings, _ = findings_from_database(get_engine())
    assert findings[0].key == "retention_targeting"
    assert everyone["spend_retained"] == pytest.approx(findings[0].exposure)
    assert everyone["spend_retained"] == pytest.approx(float(expected_loss(scores).sum()))
    assert everyone["net_benefit"] == pytest.approx(everyone["spend_retained"])
    n = math.ceil(0.2 * len(scores))
    by_value, by_probability = campaign(scores, n, "expected_loss", 0.2, 10.0, 0.25), campaign(scores, n, "probability", 0.2, 10.0, 0.25)
    assert by_value["expected_loss"] >= by_probability["expected_loss"]  # true by construction