"""Demo app entry point: two pages.

1. Real data (app/real_data.py): the KKBox churn model, the telecom retention experiment and the
   offer decision, read from the committed records in reports/metrics/.
2. StreamFlix case study (app/streamflix_app.py): the project's simulated first version, where the
   true effects are known.
"""
import streamlit as st

st.set_page_config(page_title="Churn and retention", page_icon="💸", layout="wide")
page = st.navigation(
    [st.Page("real_data.py", title="Real data: KKBox and a retention experiment", icon="📉", default=True),
     st.Page("streamflix_app.py", title="StreamFlix case study (simulated)", icon="🎬")],
    position="top",
)
page.run()
