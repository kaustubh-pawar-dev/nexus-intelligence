"""Load and run the analytic SQL queries in sql/analytics (SYNTHETIC data)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from nexus.config import get_settings

INCLUDE_TOKEN = "@LINE_FACTS@"
SHARED_FILE = "_line_facts.sql"


def query_dir() -> Path:
    return get_settings().sql_dir / "analytics"


def list_queries() -> list[str]:
    """Names of the runnable queries (files not starting with an underscore)."""
    return sorted(p.stem for p in query_dir().glob("*.sql") if not p.stem.startswith("_"))


def load_query(name: str) -> str:
    """Read a query and expand the shared line_facts CTE if it uses it."""
    text = (query_dir() / f"{name}.sql").read_text(encoding="utf-8").strip()
    if INCLUDE_TOKEN in text:
        shared = (query_dir() / SHARED_FILE).read_text(encoding="utf-8").strip()
        text = text.replace(INCLUDE_TOKEN, shared)
    return text


def run_query(engine: Engine, name: str) -> pd.DataFrame:
    """Run one named query and return the result as a DataFrame."""
    sql = load_query(name)
    with engine.connect() as conn:
        return pd.read_sql_query(sql, conn)