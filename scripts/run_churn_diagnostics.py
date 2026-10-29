"""Diagnostics of the chosen churn model (logistic regression) on the unseen test cutoffs (SYNTHETIC data)."""

from __future__ import annotations

import pandas as pd

from nexus.churn import TABLES_NEEDED, build_dataset, design, fit_logistic, split_by_cutoff
from nexus.churn_diagnostics import (
    calibration_gap,
    flag_table,
    logistic_effects,
    metrics_by_cutoff,
    risk_deciles,
)
from nexus.db import get_engine
from nexus.stats_analysis import load_tables


def main() -> int:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    data = build_dataset(load_tables(get_engine(), TABLES_NEEDED))
    train, test = split_by_cutoff(data)
    model = fit_logistic(train)
    score = model.predict_proba(design(test))[:, 1]
    y = test["churned"].to_numpy(int)

    print("SYNTHETIC DATA: CHURN MODEL DIAGNOSTICS (logistic regression, unseen test cutoffs)")
    deciles = risk_deciles(y, score)
    print("\nRisk deciles (1 = highest predicted risk):")
    print(deciles.round(3).to_string(index=False))
    print(f"\ncalibration gap (weighted mean |predicted - observed| per decile): {calibration_gap(deciles):.3f}")

    print("\nFlagging the top share of customers by risk:")
    print(flag_table(y, score).round(3).to_string(index=False))

    print("\nStability across test cutoffs:")
    print(metrics_by_cutoff(test, score).round({"churn_rate": 3, "roc_auc": 3}).to_string(index=False))

    print("\nOdds ratio per +1 SD of each feature (train data, 95% customer-level bootstrap, 200 resamples):")
    print(logistic_effects(train).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())