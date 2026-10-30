from pathlib import Path

from streamlit.testing.v1 import AppTest

from nexus.customer_view import CUSTOMER_DEFINITIONS
from nexus.db import get_engine
from nexus.dashboard_format import format_number
from nexus.sql_analytics import run_query

PAGE = str(Path(__file__).resolve().parents[1] / "app" / "views" / "customer_intelligence.py")


def run_page() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    return at


def test_page_renders_without_errors_and_shows_every_card():
    at = run_page()
    assert not at.exception
    assert not at.error
    assert [m.label for m in at.metric] == list(CUSTOMER_DEFINITIONS)


def test_customer_card_matches_the_query():
    at = run_page()
    rfm = run_query(get_engine(), "rfm_segments")
    shown = next(m for m in at.metric if m.label == "Customers with orders")
    assert shown.value == format_number(int(rfm["customers"].sum()))