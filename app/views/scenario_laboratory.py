"""Scenario Laboratory page (SYNTHETIC data): what-if calculations on the findings, with assumptions stated."""

from __future__ import annotations

import math

import plotly.graph_objects as go
import streamlit as st

from nexus.dashboard_format import format_number, format_pct
from nexus.db import get_engine
from nexus.findings_data import load_inputs
from nexus.scenarios import campaign, campaign_sensitivity, cogs_scenario, refund_scenario, totals


@st.cache_resource
def cached_engine():
    return get_engine()


@st.cache_data(ttl=600, show_spinner="Loading the baseline...")
def cached_inputs(_engine):
    return load_inputs(_engine)


def show_database_error(exc: Exception) -> None:
    st.error(
        "The baseline could not be loaded. Start PostgreSQL, check the NEXUS_DB_* settings in .env, and make sure the "
        "customers have been scored (python scripts/score_customers.py)."
    )
    with st.expander("Technical detail"):
        st.code(f"{type(exc).__name__}: {exc}")
    st.stop()


def break_even_text(value) -> str:
    if value is None:
        return "n/a"
    return ">100%" if value > 1 else format_pct(100 * value, 1)


st.title("Scenario Laboratory")
st.caption(
    "All data is synthetic. A scenario is a what-if calculation, not a forecast. The data supplies the baseline; the effects and costs "
    "are assumptions you set. The defaults for effect and cost are illustrative placeholders, not measurements."
)

try:
    inputs = cached_inputs(cached_engine())
except Exception as exc:
    show_database_error(exc)

scores, comp = inputs["scores"], inputs["comp_cur"]
base = totals(comp)
categories = sorted(comp["category"])
st.caption(
    f"Baseline: churn scores as of {inputs['as_of']} for the retention campaign, and {inputs['cur_label']} for the margin scenarios. "
    "No currency is defined in the data, so amounts are shown without a unit."
)

tab_campaign, tab_refunds, tab_cogs = st.tabs(["Retention campaign", "Fewer refunds", "Cost of goods"])

with tab_campaign:
    st.markdown(
        "Contact a number of active customers and compare two ways of choosing them: highest churn probability, or highest "
        "expected loss (probability x recent spend)."
    )
    n_scored = len(scores)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        contacts = st.number_input(
            "Customers to contact", min_value=1, max_value=n_scored, value=min(n_scored, math.ceil(0.2 * n_scored)), step=10, key="contacts"
        )
    with c2:
        effect = st.slider("Reduction in churn probability (%)", min_value=0, max_value=100, value=20, step=1, key="effect")
    with c3:
        cost = st.number_input("Cost per contact", min_value=0.0, max_value=1000000.0, value=10.0, step=1.0, key="cost")
    with c4:
        margin = st.number_input(
            "Contribution margin (%)", min_value=0.0, max_value=100.0, value=round(base["contribution_pct"], 1), step=0.5, key="margin"
        )
    by_p = campaign(scores, contacts, "probability", effect / 100, cost, margin / 100)
    by_l = campaign(scores, contacts, "expected_loss", effect / 100, cost, margin / 100)

    left, right = st.columns(2)
    with left:
        st.markdown("**Choose by churn probability**")
        st.metric("Net benefit (by probability)", format_number(by_p["net_benefit"], 0), help="Contribution kept minus campaign cost.")
        st.metric(
            "Break-even effect (by probability)",
            break_even_text(by_p["break_even_effect"]),
            help="The smallest reduction in churn probability at which the campaign pays for itself.",
        )
    with right:
        st.markdown("**Choose by expected loss**")
        st.metric("Net benefit (by expected loss)", format_number(by_l["net_benefit"], 0), help="Contribution kept minus campaign cost.")
        st.metric(
            "Break-even effect (by expected loss)",
            break_even_text(by_l["break_even_effect"]),
            help="The smallest reduction in churn probability at which the campaign pays for itself.",
        )

    def row(label, key, formatter):
        return {"Measure": label, "By probability": formatter(by_p[key]), "By expected loss": formatter(by_l[key])}

    st.dataframe(
        [
            row("Customers contacted", "contacts", lambda v: format_number(v, 0)),
            row("Average churn probability of the list", "mean_probability", lambda v: format_pct(100 * v, 1)),
            row("Their spend in the last 180 days", "spend", lambda v: format_number(v, 0)),
            row("Expected loss without the campaign", "expected_loss", lambda v: format_number(v, 0)),
            row("Spend retained by the campaign", "spend_retained", lambda v: format_number(v, 0)),
            row("Contribution retained", "contribution_retained", lambda v: format_number(v, 0)),
            row("Campaign cost", "cost", lambda v: format_number(v, 0)),
            row("Net benefit", "net_benefit", lambda v: format_number(v, 0)),
            row("Return on cost", "return_on_cost", lambda v: "n/a" if v is None else format_pct(100 * v, 0)),
        ],
        hide_index=True,
    )

    sens = campaign_sensitivity(scores, contacts, cost, margin / 100)
    fig = go.Figure()
    fig.add_scatter(
        x=100 * sens["effect"],
        y=sens["net_probability"],
        mode="lines+markers",
        name="Choose by churn probability",
        hovertemplate="Effect %{x:.0f}%<br>Net benefit %{y:,.0f}<extra></extra>",
    )
    fig.add_scatter(
        x=100 * sens["effect"],
        y=sens["net_expected_loss"],
        mode="lines+markers",
        name="Choose by expected loss",
        hovertemplate="Effect %{x:.0f}%<br>Net benefit %{y:,.0f}<extra></extra>",
    )
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(
        title="Net benefit by how much the campaign reduces churn probability",
        xaxis_title="Reduction in churn probability (%)",
        yaxis_title="Net benefit",
        legend={"orientation": "h", "y": -0.25},
        margin={"l": 10, "r": 10, "t": 50, "b": 60},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Assumptions: contacted customers would otherwise spend as much as in the last 180 days; the campaign lowers each contacted "
        "customer's churn probability by the same relative amount; only the contribution margin of retained spend counts; "
        "uncontacted customers are unaffected. Whether any campaign reaches the effect assumed is not measured."
    )

with tab_refunds:
    st.markdown("Refunds fall in the chosen categories while volumes and prices stay the same.")
    chosen_r = st.multiselect("Categories", categories, default=categories, key="refund_categories")
    reduction = st.slider("Reduction in refunds (%)", min_value=0, max_value=50, value=10, step=1, key="refund_reduction")
    if not chosen_r:
        st.warning("Choose at least one category.")
    else:
        res = refund_scenario(comp, chosen_r, reduction)
        a, b, c = st.columns(3)
        a.metric("Contribution margin, baseline", format_pct(res["baseline"]["contribution_pct"], 2))
        b.metric("Contribution margin, fewer refunds", format_pct(res["scenario"]["contribution_pct"], 2), delta=f"{res['delta_points']:+.2f} pp")
        c.metric("Contribution gained, fewer refunds", format_number(res["delta_contribution"], 0))
        table = res["by_category"].rename(columns={"category": "Category", "baseline": "Refunds, baseline", "scenario": "Refunds, scenario", "change": "Change"})
        st.dataframe(table.style.format({c_: "{:,.0f}" for c_ in ("Refunds, baseline", "Refunds, scenario", "Change")}), hide_index=True)
        st.caption(f"Baseline: {inputs['cur_label']}. Allocated refunds are cut by the chosen share; nothing else changes.")

with tab_cogs:
    st.markdown("The cost of goods changes in the chosen categories while volumes and prices stay the same.")
    chosen_c = st.multiselect("Categories", categories, default=categories, key="cogs_categories")
    change = st.slider("Change in cost of goods (%)", min_value=-20, max_value=20, value=-5, step=1, key="cogs_change")
    if not chosen_c:
        st.warning("Choose at least one category.")
    else:
        res = cogs_scenario(comp, chosen_c, change)
        a, b, c = st.columns(3)
        a.metric("Contribution margin, baseline margin", format_pct(res["baseline"]["contribution_pct"], 2))
        b.metric("Contribution margin, new cost of goods", format_pct(res["scenario"]["contribution_pct"], 2), delta=f"{res['delta_points']:+.2f} pp")
        c.metric("Contribution gained, new cost of goods", format_number(res["delta_contribution"], 0))
        table = res["by_category"].rename(columns={"category": "Category", "baseline": "Cost of goods, baseline", "scenario": "Cost of goods, scenario", "change": "Change"})
        st.dataframe(
            table.style.format({c_: "{:,.0f}" for c_ in ("Cost of goods, baseline", "Cost of goods, scenario", "Change")}),
            hide_index=True,
        )
        st.caption(f"Baseline: {inputs['cur_label']}. Only the cost of goods changes; a price or volume response is not modelled.")