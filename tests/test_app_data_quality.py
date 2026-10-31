from pathlib import Path

from streamlit.testing.v1 import AppTest

from nexus.quality_view import QUALITY_DEFINITIONS

PAGE = str(Path(__file__).resolve().parents[1] / "app" / "views" / "data_quality.py")


def run_page() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=180)
    at.run()
    return at


def test_page_renders_without_errors_and_shows_every_card():
    at = run_page()
    assert not at.exception
    assert not at.error
    assert [m.label for m in at.metric] == list(QUALITY_DEFINITIONS)


def test_no_raw_row_is_unaccounted_for():
    at = run_page()
    unaccounted = next(m for m in at.metric if m.label == "Unaccounted rows")
    assert unaccounted.value == "0"