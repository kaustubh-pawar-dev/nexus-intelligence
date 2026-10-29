"""Score the currently active customers with the chosen churn model and store them in ml.churn_scores (SYNTHETIC data)."""

from __future__ import annotations

import pandas as pd

from nexus.churn import TABLES_NEEDED, build_dataset
from nexus.churn_scoring import (
    AS_OF_CUTOFF,
    MODEL_NAME,
    current_features,
    fit_final_model,
    read_scores,
    score_frame,
    write_scores,
)
from nexus.db import get_engine
from nexus.stats_analysis import load_tables


def main() -> int:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    engine = get_engine()
    tables = load_tables(engine, TABLES_NEEDED)

    data = build_dataset(tables)
    model = fit_final_model(data)
    scores = score_frame(model, current_features(tables))
    written = write_scores(engine, scores)
    back = read_scores(engine)

    print("SYNTHETIC DATA: CHURN SCORES FOR ACTIVE CUSTOMERS")
    print(f"model {MODEL_NAME} fitted on {len(data)} labelled rows ({data['cutoff'].min().date()} to {data['cutoff'].max().date()})")
    print(f"scored as of {AS_OF_CUTOFF.date()}: {written} active customers written to ml.churn_scores, {len(back)} read back")

    by_band = back.groupby("risk_band").agg(
        customers=("customer_id", "size"),
        mean_probability=("churn_probability", "mean"),
        mean_recency_days=("recency_days", "mean"),
        mean_frequency_180=("frequency_180", "mean"),
    )
    print("\nBy risk band (High = deciles 1-2, Medium = 3-5, Low = 6-10, relative to this population):")
    print(by_band.reindex(["High", "Medium", "Low"]).round(3).to_string())
    print("\nHighest-risk customers:")
    print(back.head(5)[["customer_id", "churn_probability", "risk_decile", "recency_days", "frequency_180", "segment"]].round(3).to_string(index=False))

    ok = len(back) == written and back["churn_probability"].between(0, 1).all() and back["customer_id"].is_unique
    print("\nCHECK:", "scores stored and read back consistently" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())