"""Diagnostics for the chosen churn model: risk deciles, flag rules, stability, effect sizes (SYNTHETIC data).

Everything here describes associations in simulated data. None of it is a causal claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nexus.churn import design, fit_logistic
from nexus.churn_models import customer_blocks, evaluate_full


def risk_deciles(y, score, n_bins: int = 10) -> pd.DataFrame:
    """Rows ranked from highest to lowest predicted risk and cut into equal-sized groups.

    Comparing mean_predicted with observed_churn per group is the calibration check.
    """
    y = np.asarray(y, int)
    s = np.asarray(score, float)
    order = np.argsort(-s, kind="stable")
    base, total, cum = y.mean(), y.sum(), 0
    rows = []
    for i, g in enumerate(np.array_split(order, n_bins), start=1):
        cum += int(y[g].sum())
        observed = float(y[g].mean())
        rows.append(
            {
                "decile": i,
                "rows": len(g),
                "mean_predicted": float(s[g].mean()),
                "observed_churn": observed,
                "lift": observed / float(base),
                "cum_recall": cum / float(total),
            }
        )
    return pd.DataFrame(rows)


def calibration_gap(deciles: pd.DataFrame) -> float:
    """Row-weighted mean absolute difference between predicted and observed churn per decile."""
    w = deciles["rows"] / deciles["rows"].sum()
    return float((w * (deciles["mean_predicted"] - deciles["observed_churn"]).abs()).sum())


def flag_table(y, score, shares=(0.1, 0.2, 0.3, 0.4, 0.5)) -> pd.DataFrame:
    """Precision, recall, F1 and lift when the top share of customers by risk is flagged."""
    rows = []
    for share in shares:
        out = evaluate_full(y, score, top_share=share)
        rows.append(
            {
                "flagged_share": share,
                "precision": out["precision_top"],
                "recall": out["recall_top"],
                "f1": out["f1_top"],
                "lift": out["lift_top"],
            }
        )
    return pd.DataFrame(rows)


def metrics_by_cutoff(test: pd.DataFrame, score) -> pd.DataFrame:
    """ROC-AUC and churn rate per cutoff, to see whether performance is stable over time."""
    df = pd.DataFrame(
        {"cutoff": test["cutoff"].to_numpy(), "y": test["churned"].to_numpy(int), "s": np.asarray(score, float)}
    )
    rows = []
    for cutoff, g in df.groupby("cutoff"):
        if g["y"].nunique() < 2:
            continue
        rows.append(
            {
                "cutoff": cutoff,
                "rows": len(g),
                "churn_rate": float(g["y"].mean()),
                "roc_auc": float(roc_auc_score(g["y"], g["s"])),
            }
        )
    return pd.DataFrame(rows)


def logistic_effects(train: pd.DataFrame, n_boot: int = 200, seed: int = 42) -> pd.DataFrame:
    """Odds ratio per +1 standard deviation of each feature, with customer-level bootstrap 95% intervals.

    Correlated features (for example frequency and monetary value) share effects, so read the
    table as associations, not as the separate effect of one feature.
    """
    columns = list(design(train).columns)
    blocks = customer_blocks(train["customer_id"])
    rng = np.random.default_rng(seed)

    def coefficients(frame: pd.DataFrame) -> np.ndarray:
        return fit_logistic(frame).steps[-1][1].coef_[0]

    estimate = coefficients(train)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        draws.append(coefficients(train.iloc[idx]))
    draws = np.array(draws)
    out = pd.DataFrame(
        {
            "feature": columns,
            "odds_ratio_per_sd": np.exp(estimate),
            "ci_low": np.exp(np.percentile(draws, 2.5, axis=0)),
            "ci_high": np.exp(np.percentile(draws, 97.5, axis=0)),
        }
    )
    out["abs_log_or"] = np.abs(np.log(out["odds_ratio_per_sd"]))
    return out.sort_values("abs_log_or", ascending=False).drop(columns="abs_log_or").reset_index(drop=True)