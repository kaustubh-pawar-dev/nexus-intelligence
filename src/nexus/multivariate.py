"""Multivariate analysis and confounding checks on the cleaned layer (SYNTHETIC data).

Inputs are the customer_level query result (one row per customer). All models are
associations, not causal claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from nexus.stats_analysis import holm_adjust, mann_whitney

WINDOW_START = pd.Timestamp("2023-01-01")
WINDOW_END = pd.Timestamp("2025-12-31")
SEGMENTS = ("Consumer", "Small Business", "Enterprise")
SEGMENT_TERMS = ("Small Business", "Enterprise")  # baseline: Consumer
CHANNEL_TERMS = ("Direct", "Email", "Organic Search", "Referral", "Social Media")  # baseline: Paid Search
CORRELATION_COLUMNS = ("orders", "revenue", "contribution", "exposure_days")


def customer_model_frame(cust: pd.DataFrame) -> pd.DataFrame:
    """Add exposure_days: days from max(signup, window start) to the window end, inclusive."""
    df = cust.copy()
    signup = pd.to_datetime(df["signup_date"]).clip(lower=WINDOW_START)
    df["exposure_days"] = (WINDOW_END - signup).dt.days + 1
    return df


def design_matrix(df: pd.DataFrame, include_exposure: bool = True) -> pd.DataFrame:
    """Constant plus 0/1 columns for segment, channel and opt-in (and log exposure)."""
    x = pd.DataFrame(index=df.index)
    for s in SEGMENT_TERMS:
        x[f"segment: {s}"] = (df["segment"] == s).astype(float)
    for c in CHANNEL_TERMS:
        x[f"channel: {c}"] = (df["channel"] == c).astype(float)
    x["opt_in"] = df["marketing_opt_in"].astype(float)
    if include_exposure:
        x["log_exposure"] = np.log(df["exposure_days"].astype(float))
    return sm.add_constant(x, has_constant="add")


def coef_table(res) -> pd.DataFrame:
    """Tidy coefficients with 95% intervals and the percentage effect exp(coef) - 1."""
    ci = np.asarray(res.conf_int())
    out = pd.DataFrame(
        {
            "term": list(res.params.index),
            "coef": res.params.to_numpy(),
            "ci_low": ci[:, 0],
            "ci_high": ci[:, 1],
            "p_value": res.pvalues.to_numpy(),
        }
    )
    out["pct_effect"] = 100 * (np.exp(out["coef"]) - 1)
    out["pct_low"] = 100 * (np.exp(out["ci_low"]) - 1)
    out["pct_high"] = 100 * (np.exp(out["ci_high"]) - 1)
    return out


def fit_ltv_model(df: pd.DataFrame):
    """OLS of log revenue on segment, channel, opt-in and log exposure, among customers with orders."""
    buyers = df[df["orders"] > 0]
    y = np.log(buyers["revenue"].astype(float))
    res = sm.OLS(y, design_matrix(buyers)).fit(cov_type="HC3")
    return res, coef_table(res)


def fit_order_rate_model(df: pd.DataFrame):
    """Poisson regression of order counts with an exposure offset, over all customers."""
    x = design_matrix(df, include_exposure=False)
    offset = np.log(df["exposure_days"].astype(float))
    res = sm.GLM(df["orders"].astype(float), x, family=sm.families.Poisson(), offset=offset).fit(cov_type="HC1")
    return res, coef_table(res)


def spearman_matrix(df: pd.DataFrame, cols=CORRELATION_COLUMNS) -> pd.DataFrame:
    """Spearman correlations among customers with at least one order."""
    return df.loc[df["orders"] > 0, list(cols)].astype(float).corr(method="spearman")


def referral_vs_paid_by_segment(df: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    """Referral vs Paid Search contribution within each segment (customers with orders)."""
    b = df[df["orders"] > 0]
    rows = []
    for seg in SEGMENTS:
        a = b[(b["segment"] == seg) & (b["channel"] == "Referral")]["contribution"].to_numpy(float)
        p = b[(b["segment"] == seg) & (b["channel"] == "Paid Search")]["contribution"].to_numpy(float)
        row = {"segment": seg, "n_referral": len(a), "n_paid_search": len(p),
               "cliffs_delta": np.nan, "p_value": np.nan}
        if len(a) >= min_n and len(p) >= min_n:
            r = mann_whitney(a, p)
            row.update(cliffs_delta=r["cliffs_delta"], p_value=r["p_value"])
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    ok = out["p_value"].notna()
    if ok.any():
        out.loc[ok, "p_holm"] = holm_adjust(out.loc[ok, "p_value"].to_numpy())
    return out


def channel_segment_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Share of each segment within each channel, among customers with orders."""
    b = df[df["orders"] > 0]
    return pd.crosstab(b["channel"], b["segment"], normalize="index").round(3)