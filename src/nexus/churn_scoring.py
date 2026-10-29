"""Score currently active customers with the chosen churn model and store the scores (SYNTHETIC data).

The model is logistic regression, chosen in the model comparison. For scoring it is refitted on
every labelled cutoff. The last labelled outcome window ends on 2025-12-27, so no label uses
information from the scoring date. Customers are scored as of AS_OF_CUTOFF with features built
only from events dated before it. Scores go to their own schema (ml) so that rebuilding the
analytics schema does not delete them.

Risk deciles are relative to the scored population: decile 1 holds the highest predicted risk.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from nexus.churn import DATA_END, build_cutoff, design, fit_logistic, order_frame

AS_OF_CUTOFF = DATA_END + pd.Timedelta(days=1)
MODEL_NAME = "logistic_regression_v1"
SCHEMA = "ml"
TABLE = "churn_scores"
HIGH_MAX_DECILE = 2
MEDIUM_MAX_DECILE = 5
SCORE_COLUMNS = [
    "cutoff", "customer_id", "model_name", "churn_probability", "risk_rank", "risk_decile",
    "risk_band", "recency_days", "frequency_180", "monetary_180", "segment",
]

DDL = """
CREATE TABLE IF NOT EXISTS {schema}.churn_scores (
    cutoff            date             NOT NULL,
    customer_id       text             NOT NULL,
    model_name        text             NOT NULL,
    churn_probability double precision NOT NULL CHECK (churn_probability BETWEEN 0 AND 1),
    risk_rank         integer          NOT NULL CHECK (risk_rank >= 1),
    risk_decile       smallint         NOT NULL CHECK (risk_decile BETWEEN 1 AND 10),
    risk_band         text             NOT NULL CHECK (risk_band IN ('High', 'Medium', 'Low')),
    recency_days      integer          NOT NULL,
    frequency_180     integer          NOT NULL,
    monetary_180      numeric(14, 2)   NOT NULL,
    segment           text,
    scored_at         timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (cutoff, customer_id, model_name)
)
"""

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


def _checked(schema: str) -> str:
    if not _IDENTIFIER.match(schema):
        raise ValueError(f"unsafe schema name: {schema!r}")
    return schema


def risk_band(decile: int) -> str:
    if decile <= HIGH_MAX_DECILE:
        return "High"
    if decile <= MEDIUM_MAX_DECILE:
        return "Medium"
    return "Low"


def fit_final_model(data: pd.DataFrame):
    """Logistic regression fitted on all labelled cutoffs (train, gap and test)."""
    return fit_logistic(data)


def current_features(t: dict[str, pd.DataFrame], cutoff: pd.Timestamp = AS_OF_CUTOFF) -> pd.DataFrame:
    """Feature rows of the customers active in the 180 days before the cutoff. The label is undefined, so it is dropped."""
    orders = order_frame(t)
    customers = t["customers"].rename(columns={"acquisition_channel": "channel"})
    feats = build_cutoff(orders, customers, cutoff)
    if feats.empty:
        raise ValueError("no active customers before the cutoff")
    return feats.drop(columns=["churned"])


def score_frame(model, feats: pd.DataFrame, cutoff: pd.Timestamp = AS_OF_CUTOFF, model_name: str = MODEL_NAME) -> pd.DataFrame:
    """Churn probability, risk rank, decile and band for every customer, highest risk first."""
    if feats["customer_id"].duplicated().any():
        raise ValueError("expected one row per customer")
    x = design(feats)
    missing = x.columns[x.isna().any()].tolist()
    if missing:
        raise ValueError("missing feature values in: " + ", ".join(missing))

    prob = model.predict_proba(x)[:, 1]
    order = np.argsort(-prob, kind="stable")
    rank = np.empty(len(prob), dtype=int)
    rank[order] = np.arange(1, len(prob) + 1)
    decile = (rank - 1) * 10 // len(prob) + 1
    out = pd.DataFrame(
        {
            "cutoff": pd.Timestamp(cutoff).date(),
            "customer_id": feats["customer_id"].to_numpy(),
            "model_name": model_name,
            "churn_probability": prob,
            "risk_rank": rank,
            "risk_decile": decile,
            "risk_band": [risk_band(int(d)) for d in decile],
            "recency_days": feats["recency_days"].astype(int).to_numpy(),
            "frequency_180": feats["frequency_180"].astype(int).to_numpy(),
            "monetary_180": feats["monetary_180"].astype(float).round(2).to_numpy(),
            "segment": feats["segment"].to_numpy(),
        }
    )
    return out.sort_values("risk_rank").reset_index(drop=True)[SCORE_COLUMNS]


def write_scores(engine: Engine, scores: pd.DataFrame, schema: str = SCHEMA) -> int:
    """Replace the stored scores of one cutoff and model, in a single transaction. Safe to repeat."""
    schema = _checked(schema)
    if scores.empty:
        raise ValueError("no scores to write")
    cutoffs, models = scores["cutoff"].unique(), scores["model_name"].unique()
    if len(cutoffs) != 1 or len(models) != 1:
        raise ValueError("write one cutoff and one model at a time")
    with engine.begin() as conn:
        conn.exec_driver_sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        conn.exec_driver_sql(DDL.format(schema=schema))
        conn.execute(
            text(f"DELETE FROM {schema}.{TABLE} WHERE cutoff = :c AND model_name = :m"),
            {"c": cutoffs[0], "m": models[0]},
        )
        scores.to_sql(TABLE, conn, schema=schema, if_exists="append", index=False, method="multi", chunksize=500)
    return len(scores)


def read_scores(
    engine: Engine, cutoff: pd.Timestamp = AS_OF_CUTOFF, schema: str = SCHEMA, model_name: str = MODEL_NAME
) -> pd.DataFrame:
    schema = _checked(schema)
    sql = text(
        f"SELECT {', '.join(SCORE_COLUMNS)} FROM {schema}.{TABLE} "
        "WHERE cutoff = :c AND model_name = :m ORDER BY risk_rank"
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params={"c": pd.Timestamp(cutoff).date(), "m": model_name})
    df["monetary_180"] = df["monetary_180"].astype(float)
    return df