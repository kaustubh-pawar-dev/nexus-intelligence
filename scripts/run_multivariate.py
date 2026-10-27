"""Multivariate analysis and confounding checks (cleaned layer; SYNTHETIC data)."""

from __future__ import annotations

import pandas as pd

from nexus.db import get_engine
from nexus.multivariate import (
    channel_segment_mix,
    customer_model_frame,
    fit_ltv_model,
    fit_order_rate_model,
    referral_vs_paid_by_segment,
    spearman_matrix,
)
from nexus.sql_analytics import run_query


def _show(table: pd.DataFrame) -> str:
    t = table.copy()
    for col in ("coef", "ci_low", "ci_high", "pct_effect", "pct_low", "pct_high"):
        t[col] = t[col].round(3)
    t["p_value"] = t["p_value"].map("{:.1e}".format)
    return t.to_string(index=False)


def main() -> int:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    df = customer_model_frame(run_query(get_engine(), "customer_level"))
    buyers = int((df["orders"] > 0).sum())
    print("SYNTHETIC DATA: MULTIVARIATE ANALYSIS (CLEANED LAYER; ASSOCIATIONS, NOT CAUSES)")
    print(f"customers: {len(df)}   with orders: {buyers}   exposure days: {df['exposure_days'].min()} to {df['exposure_days'].max()}")

    print("\nSpearman correlations (customers with orders):")
    print(spearman_matrix(df).round(3).to_string())

    print("\nChannel by segment mix among buyers (rows sum to 1; a large gap would mean confounding):")
    print(channel_segment_mix(df).to_string())

    ltv, ltv_table = fit_ltv_model(df)
    print(f"\nLTV model: OLS on log revenue, buyers only (n={int(ltv.nobs)}, R-squared {ltv.rsquared:.3f}); pct_effect is exp(coef) - 1:")
    print(_show(ltv_table))

    rate, rate_table = fit_order_rate_model(df)
    print(f"\nOrder-rate model: Poisson with exposure offset, all customers (n={int(rate.nobs)}); pct_effect is the change in order rate:")
    print(_show(rate_table))

    print("\nReferral vs Paid Search contribution within each segment:")
    print(referral_vs_paid_by_segment(df).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())