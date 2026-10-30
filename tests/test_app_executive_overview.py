from pathlib import Path

from streamlit.testing.v1 import AppTest

from nexus.dashboard_data import data_range, fetch_kpis, month_window
from nexus.dashboard_format import format_compact
from nexus.dashboard_view import DEFINITIONS, preset_window
from nexus.db import get_engine

PAGE = str(Path(__file__).resolve().parents[1] / "app" / "views" / "executive_overview.py")


def run_page() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.run()
    return at


def metric(at: AppTest, label: str):
    return next(m for m in at.metric if m.label == label)


def test_page_renders_without_errors_and_shows_every_kpi():
    at = run_page()
    assert not at.exception
    assert not at.error
    assert [m.label for m in at.metric] == list(DEFINITIONS)


def test_revenue_card_matches_the_data_layer_for_the_full_period():
    at = run_page()
    engine = get_engine()
    first, last = data_range(engine)
    expected = fetch_kpis(engine, *month_window(first, last))
    assert metric(at, "Revenue").value == format_compact(expected.revenue)


def test_switching_to_the_last_six_months_changes_the_window_and_shows_a_change():
    at = run_page()
    at.sidebar.radio[0].set_value("Last 6 months").run()
    assert not at.exception
    engine = get_engine()
    first, last = data_range(engine)
    expected = fetch_kpis(engine, *preset_window("Last 6 months", first, last))
    revenue = metric(at, "Revenue")
    assert revenue.value == format_compact(expected.revenue)
    assert revenue.delta is not None and revenue.delta.endswith("%")