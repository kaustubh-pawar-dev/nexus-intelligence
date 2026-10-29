"""Gradient boosting churn model, extended metrics and bootstrap uncertainty (SYNTHETIC data).

Builds on nexus.churn. The hyperparameters are fixed and conservative. Nothing is tuned on the
test cutoffs, so the test set is scored once. Intervals come from a customer-level bootstrap
because the same customer appears at several cutoffs and its rows are not independent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from nexus.churn import design, evaluate, fit_logistic

GB_PARAMS = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": False,
}
REFERENCE = "logistic regression"
PROBABILITY_MODELS = ("logistic regression", "gradient boosting")


def fit_gradient_boosting(train: pd.DataFrame, seed: int = 42) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(random_state=seed, **GB_PARAMS)
    model.fit(design(train), train["churned"])
    return model


def model_scores(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> dict[str, np.ndarray]:
    """Churn scores on the test rows: the best simple rule, logistic regression, gradient boosting."""
    x_test = design(test)
    return {
        "rule: long recency": test["recency_days"].to_numpy(float),
        REFERENCE: fit_logistic(train).predict_proba(x_test)[:, 1],
        "gradient boosting": fit_gradient_boosting(train, seed).predict_proba(x_test)[:, 1],
    }


def evaluate_full(y, score, top_share: float = 0.2) -> dict:
    """evaluate() plus F1 of the flagged top share (the flagged customers are the predicted churners)."""
    out = evaluate(y, score, top_share)
    p, r = out["precision_top"], out["recall_top"]
    out["f1_top"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return out


def comparison_table(test: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    y = test["churned"].to_numpy(int)
    rows = []
    for name, s in scores.items():
        row = {"model": name, **evaluate_full(y, s)}
        row["brier"] = float(brier_score_loss(y, s)) if name in PROBABILITY_MODELS else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def customer_blocks(customer_ids) -> list[np.ndarray]:
    """Row positions grouped by customer. Every row belongs to exactly one block."""
    codes, uniques = pd.factorize(np.asarray(customer_ids))
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
    return [order[bounds[i] : bounds[i + 1]] for i in range(len(uniques))]


def bootstrap_intervals(
    test: pd.DataFrame,
    scores: dict[str, np.ndarray],
    n_boot: int = 300,
    seed: int = 42,
    reference: str = REFERENCE,
) -> pd.DataFrame:
    """95% percentile intervals for ROC-AUC and PR-AUC, plus paired differences to the reference.

    Each draw resamples whole customers with replacement and scores every model on the same draw,
    so the differences are paired.
    """
    y = test["churned"].to_numpy(int)
    blocks = customer_blocks(test["customer_id"])
    rng = np.random.default_rng(seed)
    metrics = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}
    names = list(scores)
    draws: dict[tuple[str, str], list[float]] = {(n, m): [] for n in names for m in metrics}
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for n in names:
            for m, fn in metrics.items():
                draws[(n, m)].append(float(fn(yb, scores[n][idx])))

    rows = []
    for n in names:
        for m, fn in metrics.items():
            d = np.array(draws[(n, m)])
            rows.append(
                {
                    "comparison": n,
                    "metric": m,
                    "estimate": float(fn(y, scores[n])),
                    "ci_low": float(np.percentile(d, 2.5)),
                    "ci_high": float(np.percentile(d, 97.5)),
                }
            )
        if n == reference:
            continue
        for m, fn in metrics.items():
            diff = np.array(draws[(n, m)]) - np.array(draws[(reference, m)])
            rows.append(
                {
                    "comparison": f"{n} minus {reference}",
                    "metric": m,
                    "estimate": float(fn(y, scores[n]) - fn(y, scores[reference])),
                    "ci_low": float(np.percentile(diff, 2.5)),
                    "ci_high": float(np.percentile(diff, 97.5)),
                }
            )
    return pd.DataFrame(rows)