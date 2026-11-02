from pathlib import Path

from streamlit.testing.v1 import AppTest

from nexus.churn_scoring import read_scores
from nexus.dashboard_format import format_number
from nexus.db import get_engine
from nexus.risk_view import RISK_DEFINITIONS

PAGE = str(Path(__file__).resolve().parents[1] / "app" / "views" / "predictive_risk.py")


def run_page() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    return at


def test_page_renders_without_errors_and_shows_every_card():
    at = run_page()
    assert not at.exception
    assert not at.error
    assert [m.label for m in at.metric] == list(RISK_DEFINITIONS)


def test_cards_match_the_stored_scores():
    at = run_page()
    scores = read_scores(get_engine())
    shown = {m.label: m.value for m in at.metric}
    assert shown["Active customers scored"] == format_number(len(scores))
    assert shown["High-risk customers"] == format_number(int((scores["risk_band"] == "High").sum()))


def test_filtering_to_the_high_band_updates_the_customer_count():
    at = run_page()
    at.sidebar.multiselect[0].set_value(["High"]).run()
    assert not at.exception
    expected = int((read_scores(get_engine())["risk_band"] == "High").sum())
    assert any(f"{expected:,} customers match" in c.value for c in at.caption)