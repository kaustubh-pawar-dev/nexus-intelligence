from pathlib import Path

from streamlit.testing.v1 import AppTest

from nexus.db import get_engine
from nexus.findings_data import findings_from_database

PAGE = str(Path(__file__).resolve().parents[1] / "app" / "views" / "decision_findings.py")


def run_page() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=180)
    at.run()
    return at


def test_page_renders_without_errors_and_shows_every_finding():
    at = run_page()
    assert not at.exception
    assert not at.error
    findings, _ = findings_from_database(get_engine())
    labels = [e.label for e in at.expander if e.label[:1].isdigit()]
    assert labels == [f"{i}. {f.title}" for i, f in enumerate(findings, start=1)]


def test_the_retention_exposure_shown_matches_the_stored_scores():
    at = run_page()
    findings, inputs = findings_from_database(get_engine())
    scores = inputs["scores"]
    total = float((scores["churn_probability"] * scores["monetary_180"]).sum())
    assert any(f"Exposure (estimate): {total:,.0f}" in m.value for m in at.markdown)