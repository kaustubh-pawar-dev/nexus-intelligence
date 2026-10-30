"""Customer Intelligence page (SYNTHETIC data): retention cohorts, RFM segments and observed lifetime value."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from nexus.customer_view import (
    CUSTOMER_DEFINITIONS,
    customer_cards,
    ltv_frame,
    retention_matrix,
    rfm_table,
)
from nexus.db import get_engine
from nexus.sql_analytics import retention_curve, run_query


@st.cache_resource
def cached_engine():
    return get_engine()


@st.cache_data(ttl=600, show_spinner="Loading customer data...")
def cached_query(_engine, name: str):
    return run_query(_engine, name)


def show_database_error(exc: Exception) -> None:
    st.error("The data could not be loaded. Start PostgreSQL, check the NEXUS_DB_* settings in .env and reload the page.")
    with st.expander("Technical detail"):
        st.code(f"{type(exc).__name__}: {exc}")
    st.stop()


st.title("Customer Intelligence")
st.caption(
    "All data is synthetic. These views cover the whole period up to 2025-12-31 and do not follow the Executive Overview filter."
)

try:
    engine = cached_engine()
    cohorts = cached_query(engine, "cohort_retention")
    rfm = cached_query(engine, "rfm_segments")
    ltv = cached_query(engine, "customer_ltv")
except Exception as exc:
    show_database_error(exc)

if rfm.empty or cohorts.empty:
    st.warning("There is no customer data to show yet. Build the analytics layer first.")
    st.stop()

curve = retention_curve(cohorts)
for column, card in zip(st.columns(4), customer_cards(rfm, curve)):
    with column, st.container(border=True):
        st.metric(card.label, card.value, help=card.help)

st.subheader("Retention by cohort")
st.caption("A cohort is the month of a customer's first completed order. Empty cells are months that lie after the end of the data.")
matrix = retention_matrix(cohorts)
heatmap = go.Figure(
    go.Heatmap(
        z=matrix.to_numpy(),
        x=list(matrix.columns),
        y=[f"{m:%Y-%m}" for m in matrix.index],
        colorscale="Blues",
        zmin=0,
        zmax=100,
        colorbar={"title": "Retention %"},
        hovertemplate="Cohort %{y}<br>Months since first order: %{x}<br>Retention: %{z:.1f}%<extra></extra>",
    )
)
heatmap.update_layout(
    xaxis_title="Months since first order",
    yaxis_title="Cohort (month of first order)",
    yaxis={"autorange": "reversed"},
    margin={"l": 10, "r": 10, "t": 30, "b": 10},
    height=560,
)
st.plotly_chart(heatmap, width="stretch")

fig_curve = go.Figure(
    go.Scatter(
        x=curve["months_since_first_order"],
        y=curve["retention_pct"],
        mode="lines+markers",
        hovertemplate="%{x} months after first order<br>Average retention: %{y:.1f}%<extra></extra>",
    )
)
fig_curve.update_layout(
    title="Average retention by months since first order",
    xaxis_title="Months since first order",
    yaxis_title="Customers active (%)",
    yaxis={"range": [0, 105]},
    margin={"l": 10, "r": 10, "t": 50, "b": 10},
)
st.plotly_chart(fig_curve, width="stretch")
st.caption("Each point averages only cohorts old enough to be observed at that distance, weighted by cohort size.")

st.subheader("RFM segments")
segments = rfm_table(rfm)
left, right = st.columns(2)
with left:
    fig_customers = go.Figure(
        go.Bar(
            x=segments["Customers"],
            y=segments["Segment"],
            orientation="h",
            hovertemplate="%{y}<br>Customers: %{x:,.0f}<extra></extra>",
        )
    )
    fig_customers.update_layout(
        title="Customers by segment",
        xaxis_title="Customers",
        yaxis={"autorange": "reversed", "title": ""},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    st.plotly_chart(fig_customers, width="stretch")
with right:
    fig_share = go.Figure(
        go.Bar(
            x=segments["Revenue share (%)"],
            y=segments["Segment"],
            orientation="h",
            hovertemplate="%{y}<br>Revenue share: %{x:.1f}%<extra></extra>",
        )
    )
    fig_share.update_layout(
        title="Revenue share by segment",
        xaxis_title="Share of revenue (%)",
        yaxis={"autorange": "reversed", "title": ""},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    st.plotly_chart(fig_share, width="stretch")
st.dataframe(segments.round(1), hide_index=True)

st.subheader("Observed lifetime value")
st.caption("Contribution margin per customer to 2025-12-31 (revenue minus COGS, allocated shipping and allocated refunds). It is observed to date, not a forecast.")
left, right = st.columns(2)
for column, dimension, title in ((left, "segment", "By customer segment"), (right, "channel", "By acquisition channel")):
    frame = ltv_frame(ltv, dimension)
    with column:
        fig_ltv = go.Figure()
        fig_ltv.add_bar(x=frame["value"], y=frame["avg_contribution_ltv"], name="Average")
        fig_ltv.add_bar(x=frame["value"], y=frame["median_contribution_ltv"], name="Median")
        fig_ltv.update_layout(
            title=title,
            barmode="group",
            xaxis_title="",
            yaxis_title="Contribution margin per customer",
            legend={"orientation": "h", "y": 1.15},
            margin={"l": 10, "r": 10, "t": 70, "b": 10},
        )
        st.plotly_chart(fig_ltv, width="stretch")
with st.expander("Lifetime value details"):
    st.dataframe(ltv_frame(ltv, "segment").round(2), hide_index=True)
    st.dataframe(ltv_frame(ltv, "channel").round(2), hide_index=True)

with st.expander("Metric definitions"):
    for label, definition in CUSTOMER_DEFINITIONS.items():
        st.markdown(f"**{label}.** {definition}")
    st.markdown(
        "**RFM.** Recency, frequency and monetary value are each split into quartiles (4 is best). "
        "Champions: frequency and recency quartile 3 or 4. At risk: frequency quartile 3 or 4 but not recent. "
        "New / promising: lower frequency but recent. Dormant: lower frequency and not recent."
    )