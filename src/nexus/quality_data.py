"""Data layer of the Data Quality page: cleaning accounting, ground-truth evaluation and defect manifest (SYNTHETIC data).

The ground-truth (core) schema and the defect manifest are read only to evaluate the cleaning
pipeline. The pipeline itself never reads either of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from nexus.config import get_settings
from nexus.dashboard_data import load_dashboard_query

ANALYTICS_TABLES = [
    "suppliers", "products", "customers", "orders", "order_items", "payments",
    "refunds", "purchase_orders", "inventory_daily", "shipments", "daily_metrics",
]


def _read(engine: Engine, name: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(load_dashboard_query(name), conn)


def accounting(engine: Engine) -> pd.DataFrame:
    return _read(engine, "quality_accounting")


def cleaning_log(engine: Engine) -> pd.DataFrame:
    return _read(engine, "quality_cleaning_log")


def ground_truth_checks(engine: Engine) -> pd.DataFrame:
    return _read(engine, "quality_ground_truth")


def promotion_log(engine: Engine) -> pd.DataFrame:
    return _read(engine, "quality_promotion_log")


@dataclass(frozen=True)
class RevenueCheck:
    staged_revenue: float
    ground_truth_revenue: float
    quarantined_revenue: float

    @property
    def unexplained_gap(self) -> float:
        return round(self.ground_truth_revenue - self.staged_revenue - self.quarantined_revenue, 2) + 0.0  # turns -0.0 into 0.0

    @property
    def held_back_pct(self) -> float | None:
        return 100 * self.quarantined_revenue / self.ground_truth_revenue if self.ground_truth_revenue else None


def revenue_check(engine: Engine) -> RevenueCheck:
    row = _read(engine, "quality_revenue").iloc[0]
    return RevenueCheck(
        staged_revenue=float(row["staged_revenue"]),
        ground_truth_revenue=float(row["ground_truth_revenue"]),
        quarantined_revenue=float(row["revenue_in_quarantined_orders"]),
    )


def _except(table: str, left: str, right: str) -> str:
    return f"SELECT count(*) FROM (SELECT * FROM {left}.{table} EXCEPT SELECT * FROM {right}.{table}) x"


def analytics_vs_ground_truth(engine: Engine) -> pd.DataFrame:
    """Row counts and rows that differ in any column, per table, between the analytics layer and the ground truth."""
    rows = []
    with engine.connect() as conn:

        def count(sql: str) -> int:
            return int(conn.exec_driver_sql(sql).scalar())

        for t in ANALYTICS_TABLES:
            rows.append(
                {
                    "table": t,
                    "analytics_rows": count(f"SELECT count(*) FROM analytics.{t}"),
                    "ground_truth_rows": count(f"SELECT count(*) FROM core.{t}"),
                    "in_analytics_not_in_truth": count(_except(t, "analytics", "core")),
                    "in_truth_not_in_analytics": count(_except(t, "core", "analytics")),
                }
            )
    return pd.DataFrame(rows)


def manifest_path() -> Path:
    return get_settings().raw_dir / "dirty" / "defect_manifest.csv"


def load_manifest() -> pd.DataFrame | None:
    """The record of every injected defect, or None if the dirty data has not been generated."""
    path = manifest_path()
    if not path.exists():
        return None
    return pd.read_csv(path, usecols=["table", "defect_type"], keep_default_na=False)