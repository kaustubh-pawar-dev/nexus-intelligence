import math
from pathlib import Path

from streamlit.testing.v1 import AppTest

from nexus.dashboard_format import format_number
from nexus.db import get_engine
from nexus.findings_data import load_inputs
from nexus.scenarios import campaign, totals

PAGE = str(Path(__file__).resolve().parents[1] / "app" / "views" / "scenario_laboratory.py")

EXPECTED_METRICS = [
    "Net benefit (by probability)",
    "Break-even effect (by probability)",
    "Net benefit (by expected loss)",
    "Break-even effect (by expected loss)",
    "Contribution margin, baseline",
    "Contribution margin, fewer refunds",
    "Contribution gained, fewer refunds",
    "Contribution margin, baseline margin",
    "Contribution margin, new cost of goods",
    "Contribution gained, new cost of goods",
]


def run_page() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=180)
    at.run()
    return at


def expected_net_benefit(effect_percent: int) -> str:
    inputs = load_inputs(get_engine())
    scores = inputs["scores"]
    margin = round(totals(inputs["comp_cur"])["contribution_pct"], 1)
    contacts = min(len(scores), math.ceil(0.2 * len(scores)))
    result = campaign(scores, contacts, "expected_loss", effect_percent / 100, 10.0, margin / 100)
    return format_number(result["net_benefit"], 0)


def test_page_renders_without_errors_and_shows_every_result():
    at = run_page()
    assert not at.exception
    assert not at.error
    assert [m.label for m in at.metric] == EXPECTED_METRICS


def test_default_campaign_matches_the_pure_calculation():
    at = run_page()
    shown = {m.label: m.value for m in at.metric}
    assert shown["Net benefit (by expected loss)"] == expected_net_benefit(20)


def test_changing_the_assumed_effect_changes_the_net_benefit():
    at = run_page()
    at.slider(key="effect").set_value(50).run()
    assert not at.exception
    shown = {m.label: m.value for m in at.metric}
    assert shown["Net benefit (by expected loss)"] == expected_net_benefit(50)