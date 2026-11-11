"""Model Performance page (SYNTHETIC data): the churn model measured on unseen cutoffs, and a backtest of the targeting rule."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from nexus.dashboard_format import format_pct
from nexus.db import get_engine
from nexus.model_report import load_report


@st.cache_resource
def cached_engine():
    return get_engine()


@st.cache_data(ttl=3600, show_spinner="Evaluating the model on the unseen cutoffs (this takes about a minute the first time)...")
def cached_report(_engine):
    return load_report(_engine)


def show_database_error(exc: Exception) -> None:
    st.error("The evaluation could not be built. Start PostgreSQL and check the NEXUS_DB_* settings in .env.")
    with st.expander("Technical detail"):
        st.code(f"{type(exc).__name__}: {exc}")
    st.stop()


st.title("Model Performance")
st.caption(
    "All data is synthetic. The churn model is trained on early cutoffs and evaluated only on later, unseen ones, with a gap equal to the "
    "outcome window so no label leaks. Every figure on this page is computed from an actual run."
)

try:
    report = cached_report(cached_engine())
except Exception as exc:
    show_database_error(exc)

comparison, intervals = report["comparison"], report["intervals"]
lr = comparison[comparison["model"] == "logistic regression"].iloc[0]
pooled = report["backtest"]["pooled"]
st.caption(
    f"Trained on {report['train_rows']:,} customer-months; evaluated on {report['test_rows']:,} customer-months from cutoffs "
    f"{report['test_cutoffs']}. {100 * report['churn_rate']:.1f}% of them churned (no order in the next 180 days)."
)

cards = [
    ("ROC-AUC (logistic regression)", f"{lr['roc_auc']:.3f}", "How well the model ranks churners above non-churners. 0.5 is chance and 1.0 is perfect."),
    ("PR-AUC (logistic regression)", f"{lr['pr_auc']:.3f}", "Precision across all thresholds. Compare with the churn rate above, which is what a random ranking scores."),
    ("Calibration gap", f"{report['calibration_gap']:.3f}", "Average difference between predicted and observed churn across risk deciles. Lower is better."),
    ("Lost spend captured by expected-loss ranking", format_pct(pooled["by_expected_loss"], 1), "Share of the lost spend of customers who really churned that falls in the top 20% by expected loss."),
]
for column, (label, value, help_text) in zip(st.columns(4), cards):
    with column, st.container(border=True):
        st.metric(label, value, help=help_text)

st.subheader("Model comparison")
st.caption("Top 20% of scores flagged as churners. Brier score only applies to models that output probabilities.")
numeric = [c for c in comparison.columns if c != "model"]
st.dataframe(
    comparison.style.format({c: "{:.3f}" for c in numeric}, na_rep="n/a"),
    hide_index=True,
)
models = list(comparison["model"])
left, right = st.columns(2)
for column, metric, title in ((left, "roc_auc", "ROC-AUC with 95% interval"), (right, "pr_auc", "PR-AUC with 95% interval")):
    rows = intervals[(intervals["metric"] == metric) & (intervals["comparison"].isin(models))].set_index("comparison").loc[models]
    fig = go.Figure(
        go.Bar(
            y=models,
            x=rows["estimate"],
            orientation="h",
            error_x={
                "type": "data",
                "symmetric": False,
                "array": rows["ci_high"] - rows["estimate"],
                "arrayminus": rows["estimate"] - rows["ci_low"],
            },
            hovertemplate="%{y}<br>%{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=metric.upper().replace("_", "-"),
        xaxis_range=[0.4, 0.85],
        yaxis={"autorange": "reversed", "title": ""},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    with column:
        st.plotly_chart(fig, width="stretch")
diffs = intervals[intervals["comparison"].str.contains(" minus ")].rename(
    columns={"comparison": "Comparison", "metric": "Metric", "estimate": "Difference", "ci_low": "95% low", "ci_high": "95% high"}
)
st.markdown("**Paired differences.** An interval that does not include 0 is a real difference.")
st.dataframe(diffs.style.format({c: "{:+.3f}" for c in ("Difference", "95% low", "95% high")}), hide_index=True)

st.subheader("Calibration")
st.caption("Customers are cut into ten equal groups by predicted risk. A well-calibrated model lies on the dotted line.")
deciles = report["deciles"]
fig_cal = go.Figure()
fig_cal.add_scatter(
    x=100 * deciles["mean_predicted"],
    y=100 * deciles["observed_churn"],
    mode="lines+markers",
    name="Risk deciles",
    text=deciles["decile"],
    hovertemplate="Decile %{text}<br>Predicted %{x:.1f}%<br>Observed %{y:.1f}%<extra></extra>",
)
fig_cal.add_scatter(x=[0, 80], y=[0, 80], mode="lines", name="Perfect calibration", line={"dash": "dot"})
fig_cal.update_layout(
    xaxis_title="Mean predicted churn (%)",
    yaxis_title="Observed churn (%)",
    legend={"orientation": "h", "y": -0.25},
    margin={"l": 10, "r": 10, "t": 30, "b": 60},
)
st.plotly_chart(fig_cal, width="stretch")

st.subheader("Flagging customers")
st.dataframe(
    report["flags"].rename(
        columns={"flagged_share": "Share flagged", "precision": "Precision", "recall": "Recall", "f1": "F1", "lift": "Lift"}
    ).style.format({"Share flagged": "{:.0%}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "Lift": "{:.2f}"}),
    hide_index=True,
)
st.caption("F1 keeps rising as more customers are flagged, because churn is common. It cannot choose a threshold. Cost and benefit do, which is what the Scenario Laboratory is for.")

st.subheader("Stability across cutoffs")
by_cutoff = report["by_cutoff"]
fig_stab = go.Figure(
    go.Scatter(
        x=by_cutoff["cutoff"],
        y=by_cutoff["roc_auc"],
        mode="lines+markers",
        hovertemplate="%{x|%b %Y}<br>ROC-AUC %{y:.3f}<extra></extra>",
    )
)
fig_stab.update_layout(xaxis_title="Cutoff", yaxis_title="ROC-AUC", margin={"l": 10, "r": 10, "t": 30, "b": 10})
st.plotly_chart(fig_stab, width="stretch")

st.subheader("Does ranking by expected loss work? A backtest on real outcomes")
st.caption(
    "On each unseen cutoff, the top 20% of customers are picked by each rule. The chart shows what share of the spend of customers who really "
    "churned sits in that top list. Lost spend is a proxy: spend in the 180 days before the cutoff, for customers with no order in the 180 days after."
)
bt = report["backtest"]["per_cutoff"]
fig_bt = go.Figure()
fig_bt.add_scatter(x=bt["cutoff"], y=bt["by_probability"], mode="lines+markers", name="Top 20% by churn probability")
fig_bt.add_scatter(x=bt["cutoff"], y=bt["by_expected_loss"], mode="lines+markers", name="Top 20% by expected loss")
fig_bt.add_scatter(x=bt["cutoff"], y=bt["random_list"], mode="lines", name="Random 20%", line={"dash": "dot"})
fig_bt.update_layout(
    xaxis_title="Cutoff",
    yaxis_title="Share of lost spend captured (%)",
    yaxis_range=[0, 100],
    legend={"orientation": "h", "y": -0.25},
    margin={"l": 10, "r": 10, "t": 30, "b": 60},
)
st.plotly_chart(fig_bt, width="stretch")
st.dataframe(
    report["backtest_intervals"].rename(columns={"rule": "Rule", "estimate": "Share captured (%)", "ci_low": "95% low", "ci_high": "95% high"}).style.format(
        {"Share captured (%)": "{:.1f}", "95% low": "{:.1f}", "95% high": "{:.1f}"}
    ),
    hide_index=True,
)
st.caption(
    "Intervals come from resampling customers, because the same customer appears at several cutoffs. A list built from churn probability alone "
    "can capture less than a random list when high-risk customers are low-value ones."
)

with st.expander("Limits of this evaluation"):
    st.markdown(
        "The data is simulated, so the scores demonstrate the method. Churn means no completed order in 180 days, which counts infrequent buyers who are "
        "simply between orders. The same customers can appear in the training and test periods at different dates. The backtest measures where lost spend "
        "sits, not whether contacting those customers would have kept it: that needs a controlled test, which the intervention tracker is for. "
        "Gradient boosting used fixed settings that were not tuned on the test cutoffs."
    )