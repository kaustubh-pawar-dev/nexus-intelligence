"""Model Performance report: the churn model measured on unseen cutoffs, plus a backtest of the targeting rule (SYNTHETIC data).

Everything is computed from the data each time the report is built. The backtest asks: among the customers
who really churned in the unseen cutoffs, how much of their spend did each ranking rule put in its top list?
Lost spend is a proxy: the spend in the 180 days before the cutoff, for customers with no order in the 180 days after.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from nexus.churn import TABLES_NEEDED, build_dataset, split_by_cutoff
from nexus.churn_diagnostics import calibration_gap, flag_table, metrics_by_cutoff, risk_deciles
from nexus.churn_models import REFERENCE, bootstrap_intervals, comparison_table, customer_blocks, model_scores
from nexus.stats_analysis import load_tables

TOP_SHARE = 0.2


def _arrays(test: pd.DataFrame, prob) -> dict:
    p = np.asarray(prob, float)
    spend = test["monetary_180"].to_numpy(float)
    codes, cutoffs = pd.factorize(test["cutoff"].to_numpy(), sort=True)
    return {
        "p": p,
        "el": p * spend,
        "lost": test["churned"].to_numpy(float) * spend,
        "codes": codes,
        "cutoffs": list(cutoffs),
    }


def _captured(a: dict, idx: np.ndarray, share: float) -> dict:
    """Realised lost spend captured by the top share of each cutoff, by each rule, summed over the cutoffs."""
    out = {"probability": 0.0, "expected_loss": 0.0, "total": 0.0}
    per = []
    for c in range(len(a["cutoffs"])):
        rows = idx[a["codes"][idx] == c]
        if len(rows) == 0:
            continue
        k = max(1, math.ceil(share * len(rows)))
        total = a["lost"][rows].sum()
        by_p = a["lost"][rows[np.argsort(-a["p"][rows], kind="stable")[:k]]].sum()
        by_e = a["lost"][rows[np.argsort(-a["el"][rows], kind="stable")[:k]]].sum()
        out["probability"] += by_p
        out["expected_loss"] += by_e
        out["total"] += total
        per.append((c, len(rows), total, by_p, by_e))
    out["per_cutoff"] = per
    return out


def targeting_backtest(test: pd.DataFrame, prob, share: float = TOP_SHARE) -> dict:
    """Per-cutoff and pooled share of realised lost spend captured by the top list of each ranking rule."""
    a = _arrays(test, prob)
    cap = _captured(a, np.arange(len(test)), share)
    rows = [
        {
            "cutoff": a["cutoffs"][c],
            "rows": n,
            "by_probability": 100 * by_p / total if total else float("nan"),
            "by_expected_loss": 100 * by_e / total if total else float("nan"),
            "random_list": 100 * share,
        }
        for c, n, total, by_p, by_e in cap["per_cutoff"]
    ]
    total = cap["total"]
    return {
        "per_cutoff": pd.DataFrame(rows),
        "pooled": {
            "by_probability": 100 * cap["probability"] / total if total else float("nan"),
            "by_expected_loss": 100 * cap["expected_loss"] / total if total else float("nan"),
            "random_list": 100 * share,
        },
    }


def backtest_intervals(test: pd.DataFrame, prob, share: float = TOP_SHARE, n_boot: int = 200, seed: int = 42) -> pd.DataFrame:
    """95% customer-level bootstrap intervals for the pooled capture of each rule and for their difference."""
    a = _arrays(test, prob)
    blocks = customer_blocks(test["customer_id"])
    rng = np.random.default_rng(seed)
    draws = {"probability": [], "expected_loss": [], "difference": []}
    for _ in range(n_boot):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        cap = _captured(a, idx, share)
        if cap["total"] == 0:
            continue
        p, e = 100 * cap["probability"] / cap["total"], 100 * cap["expected_loss"] / cap["total"]
        draws["probability"].append(p)
        draws["expected_loss"].append(e)
        draws["difference"].append(e - p)
    point = _captured(a, np.arange(len(test)), share)
    est = {
        "probability": 100 * point["probability"] / point["total"],
        "expected_loss": 100 * point["expected_loss"] / point["total"],
    }
    est["difference"] = est["expected_loss"] - est["probability"]
    labels = {
        "probability": "Top list by churn probability",
        "expected_loss": "Top list by expected loss",
        "difference": "Expected loss minus probability",
    }
    rows = []
    for key in ("probability", "expected_loss", "difference"):
        d = np.array(draws[key])
        rows.append(
            {
                "rule": labels[key],
                "estimate": float(est[key]),
                "ci_low": float(np.percentile(d, 2.5)),
                "ci_high": float(np.percentile(d, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def build_report(data: pd.DataFrame) -> dict:
    """The whole report from the churn dataset: train, then evaluate on the unseen cutoffs."""
    train, test = split_by_cutoff(data)
    test = test.reset_index(drop=True)
    scores = model_scores(train, test)
    lr = scores[REFERENCE]
    y = test["churned"].to_numpy(int)
    deciles = risk_deciles(y, lr)
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "test_cutoffs": f"{test['cutoff'].min():%b %Y} to {test['cutoff'].max():%b %Y}",
        "churn_rate": float(y.mean()),
        "comparison": comparison_table(test, scores),
        "intervals": bootstrap_intervals(test, scores),
        "deciles": deciles,
        "calibration_gap": calibration_gap(deciles),
        "flags": flag_table(y, lr),
        "by_cutoff": metrics_by_cutoff(test, lr),
        "backtest": targeting_backtest(test, lr),
        "backtest_intervals": backtest_intervals(test, lr),
    }


def load_report(engine: Engine) -> dict:
    return build_report(build_dataset(load_tables(engine, TABLES_NEEDED)))