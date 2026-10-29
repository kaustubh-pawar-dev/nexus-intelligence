"""Build the churn dataset and score the baselines (cleaned layer; SYNTHETIC data)."""

from __future__ import annotations

import pandas as pd

from nexus.churn import TABLES_NEEDED, baseline_table, build_dataset, split_by_cutoff
from nexus.db import get_engine
from nexus.stats_analysis import load_tables


def main() -> int:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    data = build_dataset(load_tables(get_engine(), TABLES_NEEDED))
    train, test = split_by_cutoff(data)
    print("SYNTHETIC DATA: CHURN DATASET AND BASELINES (CLEANED LAYER)")
    print(f"rows: {len(data)}   customers: {data['customer_id'].nunique()}   cutoffs: {data['cutoff'].nunique()}")

    by_cut = data.groupby("cutoff").agg(rows=("churned", "size"), churn_rate=("churned", "mean")).round(3)
    print("\nRows and churn rate per cutoff:")
    print(by_cut.to_string())

    print(f"\ntrain: {len(train)} rows, cutoffs {train['cutoff'].min().date()} to {train['cutoff'].max().date()}, churn {train['churned'].mean():.3f}")
    print(f"test : {len(test)} rows, cutoffs {test['cutoff'].min().date()} to {test['cutoff'].max().date()}, churn {test['churned'].mean():.3f}")
    print(f"customers in both train and test: {len(set(train['customer_id']) & set(test['customer_id']))}")

    table = baseline_table(train, test)
    print("\nBaselines on the unseen test cutoffs (top 20% of scores flagged):")
    print(table.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())