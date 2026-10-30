"""NEXUS Intelligence dashboard (SYNTHETIC data).

Run from the project root:  streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="NEXUS Intelligence", page_icon=":material/insights:", layout="wide")
st.sidebar.caption("NEXUS Intelligence. All data is synthetic.")

pages = [
    st.Page("views/executive_overview.py", title="Executive Overview", icon=":material/dashboard:", default=True),
    st.Page("views/customer_intelligence.py", title="Customer Intelligence", icon=":material/group:"),
]
st.navigation(pages).run()