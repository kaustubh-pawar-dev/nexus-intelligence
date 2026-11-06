"""Data layer of the Decision Findings page: loads the inputs and builds the findings (SYNTHETIC data)."""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy.engine import Engine

from nexus.churn_scoring import AS_OF_CUTOFF, read_scores
from nexus.dashboard_data import _run, data_range, previous_period
from nexus.dashboard_view import preset_window
from nexus.findings import Finding, build_findings
from nexus.sql_analytics import run_query


def category_components(engine: Engine, start: date, end: date) -> pd.DataFrame:
    df = _run(engine, "category_components", start, end)
    for col in ("revenue", "cogs", "shipping", "refunds"):
        df[col] = df[col].astype(float)
    return df


def _label(window: tuple[date, date]) -> str:
    return f"{window[0]:%b %Y} to {window[1]:%b %Y}"


def load_inputs(engine: Engine) -> dict:
    first, last = data_range(engine)
    window = preset_window("Last 12 months", first, last)
    prev = previous_period(window[0], window[1], first)
    return {
        "scores": read_scores(engine),
        "rfm": run_query(engine, "rfm_segments"),
        "comp_cur": category_components(engine, *window),
        "comp_prev": category_components(engine, *prev) if prev else None,
        "window": window,
        "prev_window": prev,
        "cur_label": _label(window),
        "prev_label": _label(prev) if prev else "the previous period",
        "as_of": f"{AS_OF_CUTOFF:%d %b %Y}",
    }


def findings_from_database(engine: Engine) -> tuple[list[Finding], dict]:
    inputs = load_inputs(engine)
    findings = build_findings(
        inputs["scores"],
        inputs["rfm"],
        inputs["comp_cur"],
        inputs["comp_prev"],
        inputs["cur_label"],
        inputs["prev_label"],
        inputs["as_of"],
    )
    return findings, inputs