"""Profiling and statistical tests on the cleaned analytics layer (SYNTHETIC data).

Every test reports an effect size, a Holm-adjusted p-value and a verdict that separates
statistical from practical significance. Each test is labelled by how the simulation was built:
'built in', 'built in (small)' or 'not built in'.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy.engine import Engine

ALPHA = 0.05
SMALL_DELTA = 0.147          # |Cliff's delta| at or above this counts as a practical effect
MEANINGFUL_RISK_RATIO = 1.25  # risk ratio at or above this (or at or below its inverse)
SMALL_EPSILON = 0.01          # epsilon-squared at or above this

TABLE_NAMES = [
    "customers", "orders", "order_items", "payments", "refunds",
    "shipments", "products", "suppliers", "daily_metrics", "purchase_orders",
]

RESULT_COLUMNS = [
    "id", "question", "design", "test", "n", "effect_name", "effect", "p_value", "p_holm",
    "significant", "practical", "verdict", "consistent_with_design", "detail",
]


def load_tables(engine: Engine, names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Read analytics tables into DataFrames."""
    with engine.connect() as conn:
        return {n: pd.read_sql_query(f"SELECT * FROM analytics.{n}", conn) for n in (names or TABLE_NAMES)}


# --------------------------------------------------------------------------- profiling
def profile_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """One row per column: missingness, uniqueness and, for numeric columns, spread and outliers."""
    rows = []
    for col in df.columns:
        s = df[col]
        n = len(s)
        missing = int(s.isna().sum())
        row = {
            "table": name, "column": col, "dtype": str(s.dtype), "n": n, "missing": missing,
            "missing_pct": round(100 * missing / n, 2) if n else float("nan"),
            "unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            x = s.dropna().astype(float)
            if len(x) > 0:
                q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
                iqr = q3 - q1
                outliers = ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean()
                row.update(
                    mean=x.mean(), std=x.std(), min=x.min(), p25=q1, median=med, p75=q3, max=x.max(),
                    skew=float(x.skew()) if len(x) > 2 else float("nan"),
                    outlier_pct=round(100 * outliers, 2),
                )
        rows.append(row)
    return pd.DataFrame(rows)


def seasonal_index(daily: pd.DataFrame) -> pd.Series:
    """Revenue of each calendar month relative to its own year's average month, averaged over years.

    Dividing by the same year's average removes growth between years. Growth inside a year
    remains, so the index is an approximation.
    """
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["year"], d["month"] = d["date"].dt.year, d["date"].dt.month
    m = d.groupby(["year", "month"])["gross_revenue"].sum().reset_index()
    m["index"] = m["gross_revenue"] / m.groupby("year")["gross_revenue"].transform("mean")
    return m.groupby("month")["index"].mean().round(3)


# ------------------------------------------------------------------ effect sizes and tests
def cliffs_delta(x, y) -> float:
    """P(x > y) - P(x < y), from the Mann-Whitney U statistic. Ranges from -1 to 1."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2 * u / (len(x) * len(y)) - 1)


def mann_whitney(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    res = stats.mannwhitneyu(x, y, alternative="two-sided")
    return {
        "p_value": float(res.pvalue),
        "cliffs_delta": float(2 * res.statistic / (len(x) * len(y)) - 1),
        "n_x": len(x), "n_y": len(y),
    }


def kruskal_epsilon(groups) -> dict:
    """Kruskal-Wallis test with epsilon-squared, H * (n + 1) / (n^2 - 1), as the effect size."""
    arrays = [np.asarray(g, float) for g in groups]
    h, p = stats.kruskal(*arrays)
    n = sum(len(a) for a in arrays)
    return {"p_value": float(p), "epsilon_squared": float(np.clip(h * (n + 1) / (n ** 2 - 1), 0, 1)), "n": n}


def two_proportions(events_a: int, n_a: int, events_b: int, n_b: int) -> dict:
    """Chi-square test of two proportions with the risk ratio a / b and its 95% interval."""
    table = [[events_a, n_a - events_a], [events_b, n_b - events_b]]
    _, p, _, _ = stats.chi2_contingency(table)
    ra, rb = events_a / n_a, events_b / n_b
    rr = ra / rb if rb > 0 else float("nan")
    if events_a > 0 and events_b > 0:
        se = math.sqrt(1 / events_a - 1 / n_a + 1 / events_b - 1 / n_b)
        lo, hi = math.exp(math.log(rr) - 1.96 * se), math.exp(math.log(rr) + 1.96 * se)
    else:
        lo = hi = float("nan")
    return {"p_value": float(p), "risk_ratio": rr, "ci_low": lo, "ci_high": hi, "rate_a": ra, "rate_b": rb}


def holm_adjust(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the original order."""
    p = np.asarray(pvalues, float)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


# ------------------------------------------------------------------------ test inputs
def _captured(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    p = t["payments"]
    return p.loc[p["payment_status"] == "Captured", ["payment_id", "order_id"]].copy()


def refund_frame_by_defect(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Captured payments with the order's value-weighted supplier defect score, tercile and refund flag."""
    items = (
        t["order_items"][["order_id", "product_id", "line_total"]]
        .merge(t["products"][["product_id", "supplier_id"]], on="product_id")
        .merge(t["suppliers"][["supplier_id", "defect_rate"]], on="supplier_id")
    )
    items["w"] = items["line_total"] * items["defect_rate"]
    g = items.groupby("order_id")
    score = (g["w"].sum() / g["line_total"].sum()).rename("score")
    cap = _captured(t)
    cap["score"] = cap["order_id"].map(score)
    cap["refunded"] = cap["payment_id"].isin(t["refunds"]["payment_id"])
    cap["band"] = pd.qcut(cap["score"], 3, labels=["low", "mid", "high"])
    return cap


def refund_frame_by_lateness(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Captured payments of delivered orders with a late flag (delivered after the promised date)."""
    ship = t["shipments"]
    ship = ship.loc[ship["shipment_status"] == "Delivered", ["order_id", "delivery_delay_days"]].copy()
    ship["late"] = ship["delivery_delay_days"].astype(float) > 0
    x = _captured(t).merge(ship, on="order_id")
    x["refunded"] = x["payment_id"].isin(t["refunds"]["payment_id"])
    return x


def order_value_frame(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Completed orders with their value and sales channel."""
    totals = t["order_items"].groupby("order_id")["line_total"].sum()
    o = t["orders"]
    o = o.loc[o["order_status"] == "Completed", ["order_id", "sales_channel"]].copy()
    o["value"] = o["order_id"].map(totals)
    return o


# ------------------------------------------------------------------------- the seven tests
def _mw_row(rid, question, design, x, y, names) -> dict:
    r = mann_whitney(x, y)
    return {
        "id": rid, "question": question, "design": design, "test": "Mann-Whitney U",
        "n": r["n_x"] + r["n_y"], "effect_name": "Cliff's delta", "effect": r["cliffs_delta"],
        "p_value": r["p_value"], "practical": abs(r["cliffs_delta"]) >= SMALL_DELTA,
        "detail": (
            f"{names[0]}: mean {np.mean(x):.2f}, median {np.median(x):.2f} (n={len(x)}); "
            f"{names[1]}: mean {np.mean(y):.2f}, median {np.median(y):.2f} (n={len(y)})"
        ),
    }


def _prop_row(rid, question, design, a, b, names) -> dict:
    r = two_proportions(int(a.sum()), len(a), int(b.sum()), len(b))
    rr = r["risk_ratio"]
    return {
        "id": rid, "question": question, "design": design, "test": "Chi-square (2x2)",
        "n": len(a) + len(b), "effect_name": "Risk ratio", "effect": rr, "p_value": r["p_value"],
        "practical": bool(rr >= MEANINGFUL_RISK_RATIO or rr <= 1 / MEANINGFUL_RISK_RATIO),
        "detail": (
            f"{names[0]}: {100 * r['rate_a']:.2f}% (n={len(a)}); {names[1]}: {100 * r['rate_b']:.2f}% "
            f"(n={len(b)}); risk ratio 95% interval {r['ci_low']:.2f} to {r['ci_high']:.2f}"
        ),
    }


def _kw_row(rid, question, design, groups: dict) -> dict:
    r = kruskal_epsilon(list(groups.values()))
    medians = ", ".join(f"{k} {np.median(v):.2f}" for k, v in groups.items())
    return {
        "id": rid, "question": question, "design": design, "test": "Kruskal-Wallis",
        "n": r["n"], "effect_name": "Epsilon-squared", "effect": r["epsilon_squared"],
        "p_value": r["p_value"], "practical": r["epsilon_squared"] >= SMALL_EPSILON,
        "detail": f"{len(groups)} groups; medians: {medians}",
    }


def _verdict(significant: bool, practical: bool) -> str:
    if significant and practical:
        return "significant and practically meaningful"
    if significant:
        return "statistically significant but practically small"
    if practical:
        return "meaningful size but not significant (low power)"
    return "no evidence of an effect"


def _consistent(design: str, significant: bool, practical: bool) -> bool:
    if design == "built in":
        return bool(significant)
    if design == "not built in":
        return not practical
    return True


def run_hypothesis_tests(t: dict[str, pd.DataFrame], cust: pd.DataFrame) -> pd.DataFrame:
    """Run the seven tests and return one row per test (cust is the customer_level query result)."""
    buyers = cust[cust["orders"] > 0]
    ltv = lambda col, value: buyers.loc[buyers[col] == value, "contribution"].to_numpy(float)
    rows = [
        _mw_row("T1", "Does Referral LTV differ from Paid Search LTV (customers with orders)?",
                "built in", ltv("channel", "Referral"), ltv("channel", "Paid Search"),
                ("Referral", "Paid Search")),
        _mw_row("T2", "Does Enterprise LTV differ from Consumer LTV (customers with orders)?",
                "built in", ltv("segment", "Enterprise"), ltv("segment", "Consumer"),
                ("Enterprise", "Consumer")),
    ]
    by_defect = refund_frame_by_defect(t)
    rows.append(_prop_row(
        "T3", "Is the refund rate higher on orders from high-defect suppliers (top vs bottom tercile)?",
        "built in", by_defect.loc[by_defect["band"] == "high", "refunded"],
        by_defect.loc[by_defect["band"] == "low", "refunded"], ("high-defect", "low-defect")))
    rows.append(_mw_row(
        "T4", "Do marketing opt-in customers place more orders (all customers)?",
        "built in (small)", cust.loc[cust["marketing_opt_in"], "orders"].to_numpy(float),
        cust.loc[~cust["marketing_opt_in"], "orders"].to_numpy(float), ("opt-in", "not opted in")))
    by_late = refund_frame_by_lateness(t)
    rows.append(_prop_row(
        "T5", "Is the refund rate higher after a late delivery?", "not built in",
        by_late.loc[by_late["late"], "refunded"], by_late.loc[~by_late["late"], "refunded"],
        ("late", "on time")))
    ov = order_value_frame(t)
    rows.append(_kw_row(
        "T6", "Does order value differ by sales channel?", "not built in",
        {k: g["value"].to_numpy(float) for k, g in ov.groupby("sales_channel")}))
    rows.append(_kw_row(
        "T7", "Do orders per customer differ by country?", "not built in",
        {k: g["orders"].to_numpy(float) for k, g in cust.groupby("country")}))

    df = pd.DataFrame(rows)
    df["p_holm"] = holm_adjust(df["p_value"])
    df["significant"] = df["p_holm"] < ALPHA
    df["practical"] = df["practical"].astype(bool)
    df["verdict"] = [_verdict(s, p) for s, p in zip(df["significant"], df["practical"])]
    df["consistent_with_design"] = [
        _consistent(d, s, p) for d, s, p in zip(df["design"], df["significant"], df["practical"])
    ]
    return df[RESULT_COLUMNS]