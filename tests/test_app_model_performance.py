from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parents[1] / "app" / "views" / "model_performance.py")

EXPECTED_METRICS = [
    "ROC-AUC (logistic regression)",
    "PR-AUC (logistic regression)",
    "Calibration gap",
    "Lost spend captured by expected-loss ranking",
]


def run_page() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=300)
    at.run()
    return at


def test_page_renders_without_errors_and_shows_the_headline_metrics():
    at = run_page()
    assert not at.exception
    assert not at.error
    assert [m.label for m in at.metric] == EXPECTED_METRICS


def test_headline_numbers_match_the_documented_results():
    at = run_page()
    shown = {m.label: m.value for m in at.metric}
    assert shown["ROC-AUC (logistic regression)"] == "0.735"
    assert shown["PR-AUC (logistic regression)"] == "0.605"