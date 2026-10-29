"""Compare the churn baselines with gradient boosting on the unseen test cutoffs (SYNTHETIC data)."""

from __future__ import annotations

import pandas as pd

from nexus.churn import TABLES_NEEDED, build_dataset, split_by_cutoff
from nexus.churn_models import GB_PARAMS, bootstrap_intervals, comparison_table, model_scores
from nexus.db import get_engine
from nexus.stats_analysis import load_tables


def main() -> int:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    data = build_dataset(load_tables(get_engine(), TABLES_NEEDED))
    train, test = split_by_cutoff(data)
    print("SYNTHETIC DATA: CHURN MODEL COMPARISON (CLEANED LAYER)")
    print(f"train: {len(train)} rows (churn {train['churned'].mean():.3f})   test: {len(test)} rows (churn {test['churned'].mean():.3f})")
    print(f"gradient boosting settings (fixed, not tuned on test): {GB_PARAMS}")

    scores = model_scores(train, test)
    print("\nTest metrics (top 20% of scores flagged as churners):")
    print(comparison_table(test, scores).round(3).to_string(index=False))

    print("\n95% customer-level bootstrap intervals (300 resamples):")
    print(bootstrap_intervals(test, scores).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())