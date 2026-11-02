"""Predictive Risk page (SYNTHETIC data): churn risk of the currently active customers from ml.churn_scores."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from nexus.churn_scoring import AS_OF_CUTOFF, MODEL_NAME, read_scores
from nexus.db import get_engine
from nexus.risk_view import (
    BAND_COLORS,
    BANDS,
    RISK_DEFINITIONS,
    band_summary,
    csv_bytes,
    customer_table,
    filter_scores,
    risk_cards,
)


@st.cache_resource
def cached_engine():
    return get_engine()


@st.cache_data(ttl=600, show_spinner="Loading churn scores...")
def cached_scores(_engine):
    return read_scores(_engine)


def show_database_error(exc: Exception) -> None:
    st.error(
        "The scores could not be loaded. Start PostgreSQL, check the NEXUS_DB_* settings in .env, "
        "and run python scripts/score_customers.py if the customers have not been scored yet."
    )
    with st.expander("Technical detail"):
        st.code(f"{type(exc).__name__}: {exc}")
    st.stop()


st.title("Predictive Risk")
st.caption(
    f"All data is synthetic. Customers with an order in the 180 days before {AS_OF_CUTOFF:%d %b %Y}, scored with {MODEL_NAME}. "
    "Risk is the predicted chance of no completed order in the following 180 days."
)

try:
    scores = cached_scores(cached_engine())
except Exception as exc:
    show_database_error(exc)

if scores.empty:
    st.warning("There are no stored scores. Run python scripts/score_customers.py first.")
    st.stop()

for column, card in zip(st.columns(4), risk_cards(scores)):
    with column, st.container(border=True):
        st.metric(card.label, card.value, help=card.help)
st.caption("The cards cover all scored customers. The filters in the sidebar apply to the charts and the customer list.")

segments = sorted(scores["segment"].dropna().unique())
with st.sidebar:
    st.subheader("Filters")
    chosen_bands = st.multiselect("Risk band", list(BANDS), default=list(BANDS), key="bands")
    chosen_segments = st.multiselect("Customer segment", segments, default=segments, key="segments")
    show_n = st.slider("Customers to show", min_value=10, max_value=100, value=25, step=5, key="top_n")

filtered = filter_scores(scores, chosen_bands, chosen_segments)
if filtered.empty:
    st.warning("No customers match these filters. Choose at least one risk band and one segment.")
    st.stop()

st.subheader("Risk bands")
summary = band_summary(scores)
left, right = st.columns(2)
with left:
    fig_share = go.Figure()
    fig_share.add_bar(
        y=summary["band"],
        x=summary["customers_pct"],
        name="Share of customers",
        orientation="h",
        hovertemplate="%{y}<br>%{x:.1f}% of customers<extra></extra>",
    )
    fig_share.add_bar(
        y=summary["band"],
        x=summary["spend_share_pct"],
        name="Share of spend",
        orientation="h",
        hovertemplate="%{y}<br>%{x:.1f}% of spend in the last 180 days<extra></extra>",
    )
    fig_share.update_layout(
        title="Customers and spend by risk band",
        barmode="group",
        xaxis_title="Share (%)",
        yaxis={"autorange": "reversed", "title": ""},
        legend={"orientation": "h", "y": -0.25},
        margin={"l": 10, "r": 10, "t": 50, "b": 60},
    )
    st.plotly_chart(fig_share, width="stretch")
with right:
    fig_hist = go.Figure()
    for band in BANDS:
        values = filtered.loc[filtered["risk_band"] == band, "churn_probability"] * 100
        if values.empty:
            continue
        fig_hist.add_histogram(
            x=values,
            name=band,
            xbins={"start": 0, "end": 100, "size": 5},
            marker_color=BAND_COLORS[band],
            hovertemplate="%{x}% predicted churn<br>%{y} customers<extra></extra>",
        )
    fig_hist.update_layout(
        title="Distribution of predicted churn risk",
        barmode="stack",
        xaxis_title="Predicted probability of no order in the next 180 days (%)",
        yaxis_title="Customers",
        legend={"orientation": "h", "y": -0.25},
        margin={"l": 10, "r": 10, "t": 50, "b": 60},
    )
    st.plotly_chart(fig_hist, width="stretch")

st.dataframe(
    summary.rename(
        columns={
            "band": "Band",
            "customers": "Customers",
            "customers_pct": "Share of customers (%)",
            "mean_probability_pct": "Average predicted churn (%)",
            "mean_recency_days": "Avg days since last order",
            "mean_orders_180": "Avg orders (180 days)",
            "spend_180": "Spend (180 days)",
            "spend_share_pct": "Share of spend (%)",
        }
    ).style.format(
        {
            "Customers": "{:,.0f}",
            "Share of customers (%)": "{:.1f}",
            "Average predicted churn (%)": "{:.1f}",
            "Avg days since last order": "{:.0f}",
            "Avg orders (180 days)": "{:.1f}",
            "Spend (180 days)": "{:,.0f}",
            "Share of spend (%)": "{:.1f}",
        }
    ),
    hide_index=True,
)

st.subheader("Highest-risk customers")
st.caption(f"{len(filtered):,} customers match the filters. Showing the {min(show_n, len(filtered)):,} with the highest predicted risk.")
table = customer_table(filtered, show_n)
st.dataframe(
    table.style.format(
        {
            "Churn probability (%)": "{:.1f}",
            "Days since last order": "{:,.0f}",
            "Orders (180 days)": "{:,.0f}",
            "Spend (180 days)": "{:,.0f}",
        }
    ),
    hide_index=True,
)
st.download_button(
    "Download the customers matching the filters (CSV)",
    data=csv_bytes(customer_table(filtered, len(filtered))),
    file_name="churn_risk_customers.csv",
    mime="text/csv",
)

with st.expander("How to read these scores"):
    for label, definition in RISK_DEFINITIONS.items():
        st.markdown(f"**{label}.** {definition}")
    st.markdown(
        "**Limits.** The model ranks customers better than chance, but it is not precise. Use the bands to prioritise outreach, "
        "not to make a judgement about an individual customer. Churn here means no completed order in 180 days, which counts "
        "customers who order rarely and are simply between orders. The data is simulated, so the scores demonstrate the method "
        "and say nothing about real customers."
    )