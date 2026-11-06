"""Decision Findings page (SYNTHETIC data): evidence-backed findings with estimated exposure and recommended actions."""

from __future__ import annotations

import streamlit as st

from nexus.db import get_engine
from nexus.findings import summary_table
from nexus.findings_data import findings_from_database


@st.cache_resource
def cached_engine():
    return get_engine()


@st.cache_data(ttl=600, show_spinner="Building findings from the data...")
def cached_findings(_engine):
    return findings_from_database(_engine)


def show_database_error(exc: Exception) -> None:
    st.error(
        "The findings could not be built. Start PostgreSQL, check the NEXUS_DB_* settings in .env, and make sure the "
        "customers have been scored (python scripts/score_customers.py)."
    )
    with st.expander("Technical detail"):
        st.code(f"{type(exc).__name__}: {exc}")
    st.stop()


st.title("Decision Findings")
st.caption(
    "All data is synthetic. Each finding is computed from the data when the page loads. Observations are read directly from the data; "
    "predictions rest on the churn model. None is a causal claim, and every exposure figure is an estimate with stated assumptions."
)

try:
    findings, inputs = cached_findings(cached_engine())
except Exception as exc:
    show_database_error(exc)

if not findings:
    st.warning("No findings could be built from the current data.")
    st.stop()

st.caption(
    f"Periods: {inputs['cur_label']} compared with {inputs['prev_label']}. Churn scores as of {inputs['as_of']}. "
    "No currency is defined in the data, so amounts are shown without a unit."
)
st.subheader("Summary")
st.dataframe(summary_table(findings).drop(columns="exposure"), hide_index=True)

for number, f in enumerate(findings, start=1):
    with st.expander(f"{number}. {f.title}", expanded=(number == 1)):
        st.markdown(f"**Type:** {f.kind}")
        st.markdown(f"**Finding.** {f.statement}")
        st.markdown("**Evidence**")
        st.dataframe({"Measure": [e[0] for e in f.evidence], "Value": [e[1] for e in f.evidence]}, hide_index=True)
        st.markdown(f"**Method.** {f.method}")
        st.markdown(f"**Compared with.** {f.baseline}")
        st.markdown(f"**Contributing factors.** {f.factors}")
        if f.exposure is not None:
            line = f"**Exposure (estimate): {f.exposure:,.0f}** - {f.exposure_label}."
            if f.exposure_low is not None:
                line += f" Range: {f.exposure_low:,.0f} to {f.exposure_high:,.0f}."
            st.markdown(line)
        st.markdown(f"**Assumptions.** {f.assumptions}")
        st.markdown(f"**Limits.** {f.limits}")
        st.markdown(f"**Recommended action.** {f.recommendation}")
        st.markdown(f"**How to evaluate it.** {f.evaluation}")

with st.expander("How findings are built"):
    st.markdown(
        "**Observation.** A number read directly from the cleaned data, such as a margin or a refund rate. "
        "**Prediction.** A number that depends on the churn model, such as expected loss. "
        "**Exposure.** An estimate of the amount at stake, always with the assumptions behind it. It is a size, not a forecast of savings. "
        "**Evaluation.** Every recommendation names a way to test it, so the result can later be recorded and compared with the estimate."
    )