"""Data Quality page (SYNTHETIC data): how the raw data was cleaned and how that compares with a known answer."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from nexus.db import get_engine
from nexus.quality_data import (
    accounting,
    analytics_vs_ground_truth,
    cleaning_log,
    ground_truth_checks,
    load_manifest,
    promotion_log,
    revenue_check,
)
from nexus.quality_view import (
    QUALITY_DEFINITIONS,
    accounting_shares,
    action_chart_frame,
    defect_comparison,
    knock_on_rows,
    quality_cards,
)


@st.cache_resource
def cached_engine():
    return get_engine()


@st.cache_data(ttl=600, show_spinner="Loading data quality results...")
def load_all(_engine):
    return {
        "accounting": accounting(_engine),
        "log": cleaning_log(_engine),
        "checks": ground_truth_checks(_engine),
        "analytics": analytics_vs_ground_truth(_engine),
        "promotion": promotion_log(_engine),
        "revenue": revenue_check(_engine),
        "manifest": load_manifest(),
    }


def show_database_error(exc: Exception) -> None:
    st.error("The data could not be loaded. Start PostgreSQL, check the NEXUS_DB_* settings in .env and reload the page.")
    with st.expander("Technical detail"):
        st.code(f"{type(exc).__name__}: {exc}")
    st.stop()


st.title("Data Quality")
st.caption(
    "All data is synthetic. Defects were injected into the raw data on purpose and recorded, so the cleaning pipeline "
    "can be measured against a known answer. The pipeline never reads the ground truth or the defect record."
)

try:
    data = load_all(cached_engine())
except Exception as exc:
    show_database_error(exc)

acc, log, checks = data["accounting"], data["log"], data["checks"]
manifest, revenue = data["manifest"], data["revenue"]

for column, card in zip(st.columns(4), quality_cards(acc, None if manifest is None else len(manifest), revenue.held_back_pct)):
    with column, st.container(border=True):
        st.metric(card.label, card.value, help=card.help)

st.subheader("Every raw row is accounted for")
st.caption("Each raw row ends up staged, dropped as an exact duplicate, or quarantined. Nothing disappears silently.")
shares = accounting_shares(acc)
fig_acc = go.Figure()
for column_name, label, color in (
    ("staged_rows", "Staged", "#1f77b4"),
    ("dropped_duplicates", "Dropped duplicates", "#9e9e9e"),
    ("quarantined", "Quarantined", "#c62828"),
):
    fig_acc.add_bar(
        y=shares["table_name"],
        x=shares[f"{column_name}_pct"],
        name=label,
        orientation="h",
        marker_color=color,
        customdata=shares[column_name],
        hovertemplate="%{y}<br>" + label + ": %{customdata:,.0f} rows (%{x:.1f}%)<extra></extra>",
    )
fig_acc.update_layout(
    barmode="stack",
    xaxis_title="Share of raw rows (%)",
    xaxis_range=[0, 100],
    yaxis={"autorange": "reversed", "title": ""},
    legend={"orientation": "h", "y": -0.25},
    margin={"l": 10, "r": 10, "t": 30, "b": 60},
)
st.plotly_chart(fig_acc, width="stretch")
st.dataframe(
    acc.rename(
        columns={
            "table_name": "Table",
            "raw_rows": "Raw rows",
            "staged_rows": "Staged",
            "dropped_duplicates": "Dropped duplicates",
            "quarantined": "Quarantined",
            "unaccounted": "Unaccounted",
        }
    ).style.format({c: "{:,.0f}" for c in ("Raw rows", "Staged", "Dropped duplicates", "Quarantined", "Unaccounted")}),
    hide_index=True,
)

st.subheader("What the pipeline did")
st.caption("Rows touched by each cleaning rule. Fixed rows were repaired, flagged rows were kept with a note, dropped rows were exact duplicates, quarantined rows were kept out of the cleaned layer.")
frame = action_chart_frame(log)
fig_log = go.Figure()
for action, color in (("fixed", "#2e7d32"), ("flagged", "#f9a825"), ("dropped", "#9e9e9e"), ("quarantined", "#c62828")):
    part = frame[frame["action"] == action]
    if part.empty:
        continue
    fig_log.add_bar(
        y=part["label"],
        x=part["rows"],
        name=action.capitalize(),
        orientation="h",
        marker_color=color,
        hovertemplate="%{y}<br>" + action.capitalize() + ": %{x:,.0f} rows<extra></extra>",
    )
fig_log.update_layout(
    xaxis_title="Rows",
    yaxis={"autorange": "reversed", "title": "", "categoryorder": "array", "categoryarray": list(frame["label"])},
    legend={"orientation": "h", "y": -0.1},
    margin={"l": 10, "r": 10, "t": 30, "b": 60},
    height=560,
)
st.plotly_chart(fig_log, width="stretch")

st.subheader("Injected defects against pipeline actions")
if manifest is None:
    st.info("The defect record was not found. Run scripts/generate_dirty_data.py to create it.")
else:
    st.caption(
        "One row per defect type. A small positive difference means a defect sat on a row that was also dropped or quarantined, "
        "so it could not be repaired."
    )
    comparison = defect_comparison(manifest, log)
    st.dataframe(
        comparison.rename(
            columns={
                "table": "Table",
                "defect": "Injected defect",
                "pipeline_rule": "Pipeline rule",
                "action": "Action",
                "injected": "Injected",
                "pipeline_rows": "Pipeline rows",
                "difference": "Difference",
            }
        ).style.format({"Injected": "{:,.0f}", "Pipeline rows": "{:,.0f}", "Difference": "{:,.0f}"}),
        hide_index=True,
    )

st.subheader("Cleaned staging data against the ground truth")
st.caption(
    "A zero means every compared row matches the known clean value. Imputed signup dates are estimates by design, "
    "so a small difference there is expected."
)
st.dataframe(
    checks.rename(columns={"check_name": "Check", "rows": "Result"}).style.format({"Result": "{:,.0f}"}),
    hide_index=True,
)

st.subheader("Analytics layer against the ground truth")
st.caption("Rows that differ in any column. Differences are expected where data could not be recovered or where quarantine removed rows.")
st.dataframe(
    data["analytics"]
    .rename(
        columns={
            "table": "Table",
            "analytics_rows": "Analytics rows",
            "ground_truth_rows": "Ground-truth rows",
            "in_analytics_not_in_truth": "Only in analytics",
            "in_truth_not_in_analytics": "Only in ground truth",
        }
    )
    .style.format({c: "{:,.0f}" for c in ("Analytics rows", "Ground-truth rows", "Only in analytics", "Only in ground truth")}),
    hide_index=True,
)

st.subheader("Knock-on quarantine")
st.caption("Rows removed because something they depend on was removed, for example an order item whose order was quarantined.")
left, right = st.columns(2)
with left:
    st.markdown("**Staging step**")
    st.dataframe(knock_on_rows(log).style.format({"rows": "{:,.0f}"}), hide_index=True)
with right:
    st.markdown("**Promotion to the analytics layer**")
    st.dataframe(data["promotion"].style.format({"rows": "{:,.0f}"}), hide_index=True)

st.subheader("Revenue held back by quarantine")
st.dataframe(
    {
        "Measure": [
            "Revenue in the cleaned staging data",
            "Revenue in the ground truth",
            "Revenue inside quarantined orders",
            "Unexplained gap",
        ],
        "Value": [
            f"{revenue.staged_revenue:,.2f}",
            f"{revenue.ground_truth_revenue:,.2f}",
            f"{revenue.quarantined_revenue:,.2f}",
            f"{revenue.unexplained_gap:,.2f}",
        ],
    },
    hide_index=True,
)
st.caption("The unexplained gap must be 0.00: every missing unit of revenue is accounted for by a quarantined order.")

with st.expander("Metric definitions and method"):
    for label, definition in QUALITY_DEFINITIONS.items():
        st.markdown(f"**{label}.** {definition}")
    st.markdown(
        "**Method.** The generator writes clean tables (the ground truth), then injects defects into four of them and records every change. "
        "The cleaning pipeline works only from the dirty data. This page compares its output with the ground truth afterwards."
    )