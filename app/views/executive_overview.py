"""Executive Overview page (SYNTHETIC data): KPIs, revenue trend and category mix from the cleaned analytics layer."""

from __future__ import annotations

from datetime import date

import plotly.graph_objects as go
import streamlit as st

from nexus.dashboard_data import (
    category_mix,
    data_range,
    fetch_kpis,
    filter_months,
    month_window,
    monthly_trend,
    previous_period,
)
from nexus.dashboard_view import (
    DEFINITIONS,
    PRESETS,
    build_cards,
    category_table,
    month_label,
    month_options,
    preset_window,
    trend_for_chart,
)
from nexus.db import get_engine


@st.cache_resource
def cached_engine():
    return get_engine()


@st.cache_data(ttl=600, show_spinner=False)
def cached_range(_engine):
    return data_range(_engine)


@st.cache_data(ttl=600, show_spinner="Loading KPIs...")
def cached_kpis(_engine, start: date, end: date):
    return fetch_kpis(_engine, start, end)


@st.cache_data(ttl=600, show_spinner=False)
def cached_trend(_engine):
    return monthly_trend(_engine)


@st.cache_data(ttl=600, show_spinner=False)
def cached_mix(_engine, start: date, end: date):
    return category_mix(_engine, start, end)


def show_database_error(exc: Exception) -> None:
    st.error("The data could not be loaded. Start PostgreSQL, check the NEXUS_DB_* settings in .env and reload the page.")
    with st.expander("Technical detail"):
        st.code(f"{type(exc).__name__}: {exc}")
    st.stop()


st.title("Executive Overview")
st.caption(
    "All data is synthetic. Figures come from the cleaned analytics layer, which holds out about 0.9% of revenue in quarantine."
)

try:
    engine = cached_engine()
    first, last = cached_range(engine)
except Exception as exc:  # the database is the only external dependency
    show_database_error(exc)

months = month_options(first, last)
with st.sidebar:
    st.subheader("Filters")
    preset = st.radio("Period", PRESETS, key="period")
    window = preset_window(preset, first, last)
    if window is None:
        low, high = st.select_slider(
            "Months", options=months, value=(months[0], months[-1]), format_func=month_label, key="months"
        )
        window = month_window(low, high)
    start, end = window
    st.caption(f"Showing {start:%d %b %Y} to {end:%d %b %Y}")

try:
    kpis = cached_kpis(engine, start, end)
    prev_window = previous_period(start, end, first)
    prev_kpis = cached_kpis(engine, *prev_window) if prev_window else None
    trend = filter_months(cached_trend(engine), start, end)
    mix = cached_mix(engine, start, end)
except Exception as exc:
    show_database_error(exc)

if kpis.orders == 0:
    st.warning("There are no completed orders in this period. Choose a different period.")
    st.stop()

if prev_window:
    st.caption(f"Changes compare with the previous period ({prev_window[0]:%b %Y} to {prev_window[1]:%b %Y}).")
else:
    st.caption("No change figures: the comparison period would start before the first order in the data.")

cards = build_cards(kpis, prev_kpis)
for row in (cards[:4], cards[4:]):
    for column, card in zip(st.columns(4), row):
        with column, st.container(border=True):
            st.metric(card.label, card.value, delta=card.delta, delta_color=card.delta_color, help=card.help)

chart_df = trend_for_chart(trend)
st.subheader("Revenue trend")
fig = go.Figure()
fig.add_bar(
    x=chart_df["month"],
    y=chart_df["revenue"],
    name="Monthly revenue",
    hovertemplate="%{x|%b %Y}<br>Revenue: %{y:,.0f}<extra></extra>",
)
fig.add_scatter(
    x=chart_df["month"],
    y=chart_df["moving_avg_3m"],
    mode="lines+markers",
    name="3-month average",
    hovertemplate="%{x|%b %Y}<br>3-month average: %{y:,.0f}<extra></extra>",
)
fig.update_layout(
    xaxis_title="Month",
    yaxis_title="Revenue",
    legend={"orientation": "h", "y": 1.12},
    margin={"l": 10, "r": 10, "t": 30, "b": 10},
    hovermode="x unified",
)
st.plotly_chart(fig, width="stretch")
st.caption("The 3-month average appears once three months of data are available.")

st.subheader("Year-over-year growth")
yoy = chart_df.dropna(subset=["yoy_pct"])
if yoy.empty:
    st.info("Year-over-year growth needs data from the same months one year earlier.")
else:
    colors = ["#2e7d32" if v >= 0 else "#c62828" for v in yoy["yoy_pct"]]
    fig_yoy = go.Figure(
        go.Bar(
            x=yoy["month"],
            y=yoy["yoy_pct"],
            marker_color=colors,
            hovertemplate="%{x|%b %Y}<br>%{y:+.1f}% vs same month last year<extra></extra>",
        )
    )
    fig_yoy.update_layout(
        xaxis_title="Month",
        yaxis_title="Change vs same month last year (%)",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    st.plotly_chart(fig_yoy, width="stretch")

table = category_table(mix)
left, right = st.columns(2)
with left:
    st.subheader("Revenue by category")
    fig_rev = go.Figure(
        go.Bar(
            x=table["revenue"],
            y=table["category"],
            orientation="h",
            text=table["revenue"],
            texttemplate="%{text:.3s}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Revenue: %{x:,.0f}<extra></extra>",
        )
    )
    fig_rev.update_layout(
        xaxis_title="Revenue",
        xaxis_range=[0, table["revenue"].max() * 1.2],
        yaxis={"autorange": "reversed", "title": ""},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    st.plotly_chart(fig_rev, width="stretch")
with right:
    st.subheader("Contribution margin by category")
    fig_con = go.Figure(
        go.Bar(
            x=table["contribution_pct"],
            y=table["category"],
            orientation="h",
            text=table["contribution_pct"],
            texttemplate="%{text:.1f}%",
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Contribution margin: %{x:.1f}% of revenue<extra></extra>",
        )
    )
    fig_con.update_layout(
        xaxis_title="Contribution margin (% of revenue)",
        xaxis_range=[0, table["contribution_pct"].max() * 1.2],
        yaxis={"autorange": "reversed", "title": ""},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    st.plotly_chart(fig_con, width="stretch")

with st.expander("Category details"):
    st.dataframe(
        table.rename(
            columns={
                "category": "Category",
                "revenue": "Revenue",
                "contribution": "Contribution",
                "revenue_share_pct": "Revenue share (%)",
                "contribution_pct": "Contribution margin (%)",
            }
        ).style.format(
            {
                "Revenue": "{:,.0f}",
                "Contribution": "{:,.0f}",
                "Revenue share (%)": "{:.1f}",
                "Contribution margin (%)": "{:.1f}",
            }
        ),
        hide_index=True,
    )

with st.expander("Metric definitions"):
    for label, definition in DEFINITIONS.items():
        st.markdown(f"**{label}.** {definition}")