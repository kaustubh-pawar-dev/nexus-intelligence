"""Decision findings: evidence-backed statements with estimated exposure and recommendations (SYNTHETIC data).

Every number in a finding is computed from the data when the findings are built. A finding is either
an observation (read directly from the data) or a prediction (built on the churn model). None is a
causal claim, and every exposure figure is an estimate with its assumptions stated.

No Streamlit, Plotly or database imports here, so everything can be tested directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

CALIBRATION_SHIFT = 0.05  # the size of the calibration error found in the middle risk deciles
TOP_SHARE = 0.2
MIN_CATEGORY_SHARE_PCT = 10.0
MIN_MARGIN_GAP_PP = 1.0
MIN_REFUND_CHANGE_PP = 0.25


@dataclass(frozen=True)
class Finding:
    key: str
    title: str
    kind: str  # "Observation" or "Prediction"
    statement: str
    evidence: tuple
    method: str
    baseline: str
    factors: str
    exposure_label: str | None
    exposure: float | None
    exposure_low: float | None
    exposure_high: float | None
    assumptions: str
    limits: str
    recommendation: str
    evaluation: str


# ---------- churn risk and value at risk ----------

def expected_loss(scores: pd.DataFrame, shift: float = 0.0) -> pd.Series:
    """Churn probability times spend in the 180 days before the scoring date, with probabilities optionally shifted."""
    p = np.clip(scores["churn_probability"].to_numpy(float) + shift, 0.0, 1.0)
    return pd.Series(p * scores["monetary_180"].to_numpy(float), index=scores.index)


def capture(scores: pd.DataFrame, top_share: float = TOP_SHARE) -> dict:
    """What a list of the top customers captures, ranked by churn probability and by expected loss."""
    scores = scores.reset_index(drop=True)
    n = len(scores)
    k = max(1, math.ceil(top_share * n))
    loss = expected_loss(scores).to_numpy()
    spend = scores["monetary_180"].to_numpy(float)
    prob = scores["churn_probability"].to_numpy(float)
    total_loss, total_spend = loss.sum(), spend.sum()

    def pack(order: np.ndarray) -> dict:
        idx = order[:k]
        seg = scores["segment"].iloc[idx].value_counts()
        return {
            "loss_share": 100 * loss[idx].sum() / total_loss if total_loss else float("nan"),
            "spend_share": 100 * spend[idx].sum() / total_spend if total_spend else float("nan"),
            "mean_probability": float(prob[idx].mean()),
            "top_segment": str(seg.index[0]),
            "top_segment_pct": 100 * float(seg.iloc[0]) / k,
        }

    return {
        "k": k,
        "by_probability": pack(np.argsort(-prob, kind="stable")),
        "by_expected_loss": pack(np.argsort(-loss, kind="stable")),
    }


def finding_retention(scores: pd.DataFrame, as_of: str) -> Finding:
    n = len(scores)
    high = scores[scores["risk_band"] == "High"]
    spend = scores["monetary_180"].astype(float)
    loss = expected_loss(scores)
    total_spend, total_loss = spend.sum(), loss.sum()
    spend_share_high = 100 * high["monetary_180"].sum() / total_spend if total_spend else float("nan")
    loss_share_high = 100 * loss[high.index].sum() / total_loss if total_loss else float("nan")
    c = capture(scores)
    p, v, k = c["by_probability"], c["by_expected_loss"], c["k"]
    gap = v["loss_share"] - p["loss_share"]
    title = (
        "Ranking customers by churn probability alone misses most of the value at risk"
        if gap >= 20
        else "Churn probability and expected loss rank customers differently"
    )
    low = expected_loss(scores, -CALIBRATION_SHIFT).sum()
    high_case = expected_loss(scores, CALIBRATION_SHIFT).sum()
    return Finding(
        key="retention_targeting",
        title=title,
        kind="Prediction",
        statement=(
            f"The top {k:,} customers by churn probability account for {p['loss_share']:.1f}% of expected loss. "
            f"The top {k:,} by expected loss (probability x recent spend) account for {v['loss_share']:.1f}%. "
            f"The {len(high):,} High-risk customers hold {spend_share_high:.1f}% of the spend of the {n:,} scored customers."
        ),
        evidence=(
            ("Customers scored", f"{n:,}"),
            ("High-risk customers", f"{len(high):,} ({100 * len(high) / n:.0f}% of customers)"),
            ("Share of 180-day spend held by High-risk customers", f"{spend_share_high:.1f}%"),
            ("Share of expected loss held by High-risk customers", f"{loss_share_high:.1f}%"),
            ("Expected loss, all customers", f"{total_loss:,.0f} ({100 * total_loss / total_spend:.1f}% of spend)"),
            (
                f"Top {k:,} by churn probability",
                f"{p['loss_share']:.1f}% of expected loss, {p['spend_share']:.1f}% of spend, {p['top_segment_pct']:.0f}% {p['top_segment']}",
            ),
            (
                f"Top {k:,} by expected loss",
                f"{v['loss_share']:.1f}% of expected loss, {v['spend_share']:.1f}% of spend, {v['top_segment_pct']:.0f}% {v['top_segment']}",
            ),
        ),
        method=f"Churn probabilities from the logistic regression, multiplied by each customer's spend in the 180 days before {as_of}.",
        baseline="Ranking customers by churn probability alone.",
        factors=(
            "Recency and order frequency drive the model. Enterprise and Small Business customers have lower churn odds "
            "but far higher spend, so the two rankings pick different customers."
        ),
        exposure_label="Expected spend at risk over the next 180 days",
        exposure=float(total_loss),
        exposure_low=float(low),
        exposure_high=float(high_case),
        assumptions=(
            "A customer who churns would have spent as much as in the last 180 days, and an outreach lowers churn probability "
            f"in proportion to it. The range shifts every probability by {CALIBRATION_SHIFT * 100:.0f} points."
        ),
        limits=(
            "These are predictions, not outcomes. The model under-predicts churn in the middle risk deciles and ranks customers "
            "only moderately well. Whether outreach changes behaviour is not measured. The data is synthetic."
        ),
        recommendation=(
            "Rank retention outreach by expected loss (probability x recent spend) and review high-spend customers with moderate risk first. "
            "Keep probability-only lists for low-cost automated messages."
        ),
        evaluation=(
            "Contact a random half of the top expected-loss customers and compare repeat orders over the next 180 days "
            "with the other half."
        ),
    )


# ---------- customer value ----------

def _segment(rfm: pd.DataFrame, name: str):
    row = rfm.loc[rfm["rfm_segment"] == name]
    return None if row.empty else row.iloc[0]


def finding_lapsed(rfm: pd.DataFrame) -> Finding | None:
    r = _segment(rfm, "At risk")
    if r is None:
        return None
    total = float(rfm["customers"].sum())
    customers, recency = int(r["customers"]), float(r["avg_recency_days"])
    revenue, share = float(r["revenue"]), float(r["revenue_share_pct"])
    return Finding(
        key="lapsed_value",
        title="Frequent buyers who have stopped ordering still account for a large share of revenue",
        kind="Observation",
        statement=(
            f"{customers:,} customers ({100 * customers / total:.0f}% of customers with orders) bought often but have not ordered for "
            f"{recency:.0f} days on average. They account for {share:.1f}% of all revenue ({revenue:,.0f})."
        ),
        evidence=(
            ("Customers in the At risk segment", f"{customers:,} of {int(total):,}"),
            ("Average days since last order", f"{recency:.0f}"),
            ("Average orders per customer", f"{float(r['avg_orders']):.1f}"),
            ("Revenue to date", f"{revenue:,.0f} ({share:.1f}% of all revenue)"),
        ),
        method="RFM segmentation: recency, frequency and monetary value split into quartiles. At risk means high frequency but not recent.",
        baseline="The other RFM segments in the same table.",
        factors="Not analysed here. The segment is defined by rules, so it shows who lapsed but not why.",
        exposure_label=None,
        exposure=None,
        exposure_low=None,
        exposure_high=None,
        assumptions="No loss is estimated. The revenue shown is what these customers already generated, not what could be won back.",
        limits=(
            "These customers are outside the churn model, which scores only customers with an order in the last 180 days. "
            "The segment is a rule, not a prediction. The data is synthetic."
        ),
        recommendation="Run a win-back campaign for this segment, separate from the churn-risk outreach, and track reactivation.",
        evaluation="Randomly hold out part of the segment and compare the share who place an order in the following 90 days.",
    )


def finding_concentration(rfm: pd.DataFrame) -> Finding | None:
    r = _segment(rfm, "Champions")
    if r is None:
        return None
    total = float(rfm["customers"].sum())
    customers, share = int(r["customers"]), float(r["revenue_share_pct"])
    if share < 50:
        return None
    return Finding(
        key="revenue_concentration",
        title="Most revenue comes from a small group of Champions",
        kind="Observation",
        statement=(
            f"{customers:,} Champions ({100 * customers / total:.0f}% of customers with orders) account for {share:.1f}% of all revenue "
            f"({float(r['revenue']):,.0f})."
        ),
        evidence=(
            ("Champions", f"{customers:,} of {int(total):,} customers"),
            ("Share of all revenue", f"{share:.1f}%"),
            ("Average orders per Champion", f"{float(r['avg_orders']):.1f}"),
            ("Average days since last order", f"{float(r['avg_recency_days']):.0f}"),
        ),
        method="RFM segmentation: Champions have frequency and recency both in the top two quartiles.",
        baseline="Their share of customers.",
        factors="Not analysed here.",
        exposure_label=None,
        exposure=None,
        exposure_low=None,
        exposure_high=None,
        assumptions="No loss is estimated. The finding describes dependency, not a forecast.",
        limits="Revenue to date, not future value. Segment boundaries are quartiles of this population. The data is synthetic.",
        recommendation="Protect this group first: monitor their order frequency every month and escalate when it drops.",
        evaluation="Track the Champions' monthly retention and their share of revenue over time.",
    )


# ---------- margin and refunds ----------

def component_table(comp: pd.DataFrame) -> pd.DataFrame:
    """Revenue, COGS, shipping and refunds by category as percentages of revenue."""
    out = comp.copy()
    for col in ("revenue", "cogs", "shipping", "refunds"):
        out[col] = out[col].astype(float)
    r = out["revenue"]
    out["revenue_share_pct"] = 100 * r / r.sum()
    out["gross_margin_pct"] = 100 * (r - out["cogs"]) / r
    out["shipping_pct"] = 100 * out["shipping"] / r
    out["refund_pct"] = 100 * out["refunds"] / r
    out["contribution_pct"] = out["gross_margin_pct"] - out["shipping_pct"] - out["refund_pct"]
    return out


def _pct(frame: pd.DataFrame) -> dict:
    r = frame["revenue"].sum()
    return {
        "revenue": r,
        "gm": 100 * (r - frame["cogs"].sum()) / r,
        "ship": 100 * frame["shipping"].sum() / r,
        "refund": 100 * frame["refunds"].sum() / r,
    }


def finding_margin(comp: pd.DataFrame, period: str) -> Finding | None:
    t = component_table(comp)
    candidates = t[t["revenue_share_pct"] >= MIN_CATEGORY_SHARE_PCT]
    if candidates.empty:
        return None
    row = candidates.sort_values("contribution_pct").iloc[0]
    name = str(row["category"])
    rest = _pct(t[t["category"] != name])
    cat = _pct(t[t["category"] == name])
    rest_contribution = rest["gm"] - rest["ship"] - rest["refund"]
    gap = rest_contribution - float(row["contribution_pct"])
    if gap < MIN_MARGIN_GAP_PP:
        return None
    parts = {
        "gross margin": rest["gm"] - cat["gm"],
        "shipping": cat["ship"] - rest["ship"],
        "refunds": cat["refund"] - rest["refund"],
    }
    driver = max(parts, key=lambda k: parts[k])
    advice = {
        "gross margin": "Review pricing and supplier costs for the category, starting with its lowest-margin products.",
        "shipping": "Review carrier rates, packaging and free-shipping thresholds for the category.",
        "refunds": "Investigate refund reasons and product quality in the category.",
    }[driver]
    exposure = cat["revenue"] * gap / 100
    return Finding(
        key="category_margin",
        title=f"{name} earns a lower contribution margin than the other categories",
        kind="Observation",
        statement=(
            f"{name} is {float(row['revenue_share_pct']):.1f}% of revenue in {period} but earns a contribution margin of "
            f"{float(row['contribution_pct']):.1f}%, against {rest_contribution:.1f}% for the other categories combined."
        ),
        evidence=(
            ("Revenue share", f"{float(row['revenue_share_pct']):.1f}%"),
            ("Contribution margin", f"{float(row['contribution_pct']):.1f}% (other categories {rest_contribution:.1f}%)"),
            ("Gross margin", f"{cat['gm']:.1f}% (other categories {rest['gm']:.1f}%)"),
            ("Shipping cost as % of revenue", f"{cat['ship']:.1f}% (other categories {rest['ship']:.1f}%)"),
            ("Refunds as % of revenue", f"{cat['refund']:.1f}% (other categories {rest['refund']:.1f}%)"),
        ),
        method=f"Revenue, COGS, allocated shipping and allocated refunds by category for {period}, from the shared line-level revenue definition.",
        baseline="The other categories combined.",
        factors=(
            f"Of the {gap:.1f}-point gap, gross margin accounts for {parts['gross margin']:+.1f} points, shipping for "
            f"{parts['shipping']:+.1f} and refunds for {parts['refunds']:+.1f}. A negative figure means the category does better "
            "than the rest on that item."
        ),
        exposure_label="Contribution gap versus the other categories",
        exposure=float(exposure),
        exposure_low=None,
        exposure_high=None,
        assumptions=f"What {name} would add if it earned the same contribution margin as the rest. A benchmark, not a forecast of savings.",
        limits=(
            "Shipping and refunds are allocated to lines in proportion to revenue, so category figures are approximations. "
            "The data is synthetic."
        ),
        recommendation=advice,
        evaluation="Track the category's gross margin, shipping cost and refund rate monthly after the change.",
    )


def finding_refunds(cur: pd.DataFrame, prev: pd.DataFrame | None, cur_label: str, prev_label: str) -> Finding | None:
    if prev is None:
        return None
    c, p = component_table(cur), component_table(prev)
    cur_rate = 100 * c["refunds"].sum() / c["revenue"].sum()
    prev_rate = 100 * p["refunds"].sum() / p["revenue"].sum()
    change = cur_rate - prev_rate
    if abs(change) < MIN_REFUND_CHANGE_PP:
        return None
    merged = c.merge(p[["category", "refund_pct"]], on="category", how="left", suffixes=("", "_prev"))
    merged["excess"] = (merged["refund_pct"] - merged["refund_pct_prev"].fillna(merged["refund_pct"])) / 100 * merged["revenue"]
    up = change > 0
    ranked = merged.sort_values("excess", ascending=not up).head(3)
    drivers = "; ".join(f"{r.category} ({r.excess:+,.0f})" for r in ranked.itertuples())
    exposure = abs(change) / 100 * c["revenue"].sum()
    word = "rose" if up else "fell"
    return Finding(
        key="refund_rate",
        title=f"The refund rate {word} by {abs(change):.1f} points",
        kind="Observation",
        statement=(
            f"Refunds were {cur_rate:.2f}% of revenue in {cur_label}, against {prev_rate:.2f}% in {prev_label} "
            f"({change:+.2f} points)."
        ),
        evidence=(
            (f"Refund rate, {cur_label}", f"{cur_rate:.2f}%"),
            (f"Refund rate, {prev_label}", f"{prev_rate:.2f}%"),
            ("Change", f"{change:+.2f} points"),
            ("Categories with the largest change in refunds", drivers),
        ),
        method=f"Allocated refunds as a share of revenue for {cur_label} compared with {prev_label}.",
        baseline=f"The previous period, {prev_label}.",
        factors=f"Largest contributions to the change (refund amount against the previous-period rate): {drivers}.",
        exposure_label="Refunds above the previous-period rate" if up else "Refunds below the previous-period rate",
        exposure=float(exposure),
        exposure_low=None,
        exposure_high=None,
        assumptions="The amount refunded in the period minus what the previous-period refund rate would have produced on the same revenue.",
        limits="Refunds are attributed to the month of the original order, and allocated to lines in proportion to revenue. The data is synthetic.",
        recommendation=(
            "Look at refund reasons in the categories listed above and check for product, description or delivery problems."
            if up
            else "Identify what changed in the categories listed above and keep doing it."
        ),
        evaluation="Track the refund rate monthly for the categories listed above.",
    )


# ---------- assembly ----------

def build_findings(
    scores: pd.DataFrame,
    rfm: pd.DataFrame,
    comp_cur: pd.DataFrame,
    comp_prev: pd.DataFrame | None,
    cur_label: str,
    prev_label: str,
    as_of: str,
) -> list[Finding]:
    candidates = [
        finding_retention(scores, as_of) if len(scores) else None,
        finding_margin(comp_cur, cur_label) if len(comp_cur) else None,
        finding_refunds(comp_cur, comp_prev, cur_label, prev_label) if len(comp_cur) else None,
        finding_lapsed(rfm),
        finding_concentration(rfm),
    ]
    return [f for f in candidates if f is not None]


def summary_table(findings: list[Finding]) -> pd.DataFrame:
    rows = []
    for f in findings:
        if f.exposure is None:
            shown = "n/a"
        elif f.exposure_low is not None:
            shown = f"{f.exposure:,.0f} ({f.exposure_low:,.0f} to {f.exposure_high:,.0f})"
        else:
            shown = f"{f.exposure:,.0f}"
        rows.append(
            {
                "Finding": f.title,
                "Type": f.kind,
                "Exposure (estimate)": shown,
                "What it measures": f.exposure_label or "No exposure estimated",
                "exposure": f.exposure,
            }
        )
    return pd.DataFrame(rows)