"""Churn dataset, leak-free splits and baseline models (cleaned layer; SYNTHETIC data).

One row per customer per monthly cutoff. Population: customers with a completed order in the
180 days before the cutoff. Label: churned = no completed order in the 180 days from the cutoff.
Every feature uses only events dated before the cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON_DAYS = 180
FIRST_CUTOFF = pd.Timestamp("2023-07-01")
DATA_END = pd.Timestamp("2025-12-31")
TRAIN_LAST_CUTOFF = pd.Timestamp("2024-01-01")
TEST_FIRST_CUTOFF = pd.Timestamp("2024-07-01")
SEGMENTS = ("Consumer", "Small Business", "Enterprise")
CHANNELS = ("Direct", "Email", "Organic Search", "Paid Search", "Referral", "Social Media")
FEATURE_COLUMNS = [
    "frequency_180", "monetary_180", "avg_order_value", "recency_days", "refunds_180",
    "late_orders_180", "orders_total", "tenure_days", "days_since_signup",
]
TABLES_NEEDED = ["customers", "orders", "order_items", "refunds", "shipments"]


def cutoff_dates() -> list[pd.Timestamp]:
    """Month starts from FIRST_CUTOFF whose full 180-day outcome window fits inside the data."""
    out = []
    for c in pd.date_range(FIRST_CUTOFF, DATA_END, freq="MS"):
        if c + pd.Timedelta(days=HORIZON_DAYS - 1) <= DATA_END:
            out.append(c)
    return out


def order_frame(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Completed orders with value, refund date, delivery date and delivery delay."""
    o = t["orders"]
    o = o.loc[o["order_status"] == "Completed", ["order_id", "customer_id", "order_date"]].copy()
    o["order_date"] = pd.to_datetime(o["order_date"])
    o["value"] = o["order_id"].map(t["order_items"].groupby("order_id")["line_total"].sum()).astype(float)
    ref = t["refunds"][["order_id", "refund_date"]].copy()
    ref["refund_date"] = pd.to_datetime(ref["refund_date"])
    sh = t["shipments"][["order_id", "delivered_date", "delivery_delay_days"]].copy()
    sh["delivered_date"] = pd.to_datetime(sh["delivered_date"])
    sh["delivery_delay_days"] = pd.to_numeric(sh["delivery_delay_days"], errors="coerce")
    return o.merge(ref, on="order_id", how="left").merge(sh, on="order_id", how="left")


def build_cutoff(orders: pd.DataFrame, customers: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Feature and label rows for one cutoff. Features use only events dated before the cutoff."""
    start = cutoff - pd.Timedelta(days=HORIZON_DAYS)
    end = cutoff + pd.Timedelta(days=HORIZON_DAYS)
    past = orders[orders["order_date"] < cutoff]
    win = past[past["order_date"] >= start]
    if win.empty:
        return pd.DataFrame()

    g = win.groupby("customer_id")
    feat = pd.DataFrame(
        {
            "last_order": g["order_date"].max(),
            "frequency_180": g.size(),
            "monetary_180": g["value"].sum(),
        }
    )
    feat["recency_days"] = (cutoff - feat["last_order"]).dt.days
    feat["avg_order_value"] = feat["monetary_180"] / feat["frequency_180"]

    refunded = win[win["refund_date"].notna() & (win["refund_date"] < cutoff)]
    feat["refunds_180"] = refunded.groupby("customer_id").size().reindex(feat.index, fill_value=0)
    late = win[
        win["delivered_date"].notna() & (win["delivered_date"] < cutoff) & (win["delivery_delay_days"] > 0)
    ]
    feat["late_orders_180"] = late.groupby("customer_id").size().reindex(feat.index, fill_value=0)

    allp = past.groupby("customer_id").agg(orders_total=("order_id", "size"), first_order=("order_date", "min"))
    feat = feat.join(allp)
    feat["tenure_days"] = (cutoff - feat["first_order"]).dt.days

    cust = customers.set_index("customer_id")
    signup = pd.to_datetime(cust["signup_date"]).reindex(feat.index)
    feat["days_since_signup"] = (cutoff - signup).dt.days
    feat["segment"] = cust["segment"].reindex(feat.index)
    feat["channel"] = cust["channel"].reindex(feat.index)
    feat["marketing_opt_in"] = cust["marketing_opt_in"].reindex(feat.index).astype(bool)

    future = orders.loc[(orders["order_date"] >= cutoff) & (orders["order_date"] < end), "customer_id"].unique()
    feat["churned"] = (~feat.index.isin(future)).astype(int)
    feat["cutoff"] = cutoff
    return feat.drop(columns=["last_order", "first_order"]).reset_index()


def build_dataset(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """All cutoffs stacked into one frame (customer_id, cutoff, features, churned)."""
    orders = order_frame(t)
    cust = t["customers"].rename(columns={"acquisition_channel": "channel"})
    frames = [build_cutoff(orders, cust, c) for c in cutoff_dates()]
    return pd.concat([f for f in frames if len(f)], ignore_index=True)


def split_by_cutoff(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on early cutoffs and test on later ones. Cutoffs in between are dropped (the gap)."""
    train = df[df["cutoff"] <= TRAIN_LAST_CUTOFF]
    test = df[df["cutoff"] >= TEST_FIRST_CUTOFF]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def design(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric model matrix: log-scaled counts and amounts, dates as days, 0/1 dummies."""
    x = pd.DataFrame(index=df.index)
    for col in ("frequency_180", "monetary_180", "avg_order_value", "orders_total"):
        x[f"log_{col}"] = np.log1p(df[col].astype(float))
    for col in ("recency_days", "tenure_days", "days_since_signup", "refunds_180", "late_orders_180"):
        x[col] = df[col].astype(float)
    x["opt_in"] = df["marketing_opt_in"].astype(float)
    for s in SEGMENTS[1:]:
        x[f"segment: {s}"] = (df["segment"] == s).astype(float)
    for c in CHANNELS[1:]:
        x[f"channel: {c}"] = (df["channel"] == c).astype(float)
    return x


def evaluate(y, score, top_share: float = 0.2) -> dict:
    """Ranking metrics plus precision, recall and lift among the top share of scores."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    k = max(1, int(round(top_share * len(y))))
    top = np.argsort(-score, kind="stable")[:k]
    precision = float(y[top].mean())
    return {
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "precision_top": precision,
        "recall_top": float(y[top].sum() / y.sum()),
        "lift_top": precision / float(y.mean()),
    }


def fit_logistic(train: pd.DataFrame):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(design(train), train["churned"])
    return model


def baseline_scores(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> dict[str, np.ndarray]:
    """Churn scores on the test rows (higher means more likely to churn)."""
    rng = np.random.default_rng(seed)
    model = fit_logistic(train)
    return {
        "random": rng.random(len(test)),
        "rule: long recency": test["recency_days"].to_numpy(float),
        "rule: low frequency": -test["frequency_180"].to_numpy(float),
        "logistic regression": model.predict_proba(design(test))[:, 1],
    }


def baseline_table(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Metrics of every baseline on the test cutoffs. Brier score only for probabilities."""
    scores = baseline_scores(train, test, seed)
    y = test["churned"].to_numpy()
    rows = []
    for name, s in scores.items():
        row = {"model": name, **evaluate(y, s)}
        row["brier"] = float(brier_score_loss(y, s)) if name == "logistic regression" else float("nan")
        rows.append(row)
    ref = float(brier_score_loss(y, np.full(len(y), train["churned"].mean())))
    rows.append({"model": "reference: constant train rate", "brier": ref, "prevalence": float(y.mean())})
    return pd.DataFrame(rows)