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

def retention_curve(cohorts: pd.DataFrame, offsets=(0, 1, 2, 3, 6, 9, 12)) -> pd.DataFrame:
    """Retention by months since first order, using only cohorts old enough to be observed.

    Input is the cohort_retention query result. The last observed month is the maximum of
    cohort month plus months since first order.
    """
    c = cohorts.copy()
    month = pd.to_datetime(c["cohort_month"])
    c["cohort_idx"] = month.dt.year * 12 + month.dt.month
    last_idx = int((c["cohort_idx"] + c["months_since"].astype(int)).max())
    sizes = c.drop_duplicates("cohort_month")[["cohort_idx", "cohort_size"]]
    rows = []
    for k in offsets:
        eligible = sizes[sizes["cohort_idx"] + k <= last_idx]
        total = int(eligible["cohort_size"].sum())
        active = int(c.loc[c["months_since"] == k, "active_customers"].sum())
        rows.append(
            {
                "months_since_first_order": k,
                "cohorts_observed": len(eligible),
                "customers_in_cohorts": total,
                "active_customers": active,
                "retention_pct": round(100 * active / total, 1) if total else float("nan"),
            }
        )
    return pd.DataFrame(rows)