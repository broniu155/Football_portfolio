import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from app.components.ui import setup_page

setup_page(page_title="Football Portfolio", page_icon="⚽")

st.title("Football Data Analyst Portfolio")
st.caption("StatsBomb open data • ETL to star schema • Streamlit + Plotly + DuckDB")
st.markdown('<span class="context-chip">Dark tactical dashboard mode enabled</span>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Focus", "Match & Player Insights")
k2.metric("Model", "Star Schema")
k3.metric("Engine", "DuckDB")
k4.metric("Visuals", "Coach-ready reports")

st.markdown('<div class="section-title">Pages</div>', unsafe_allow_html=True)
st.markdown(
    """
- `Match Report`: pitch shot map, xG timeline, filtered event table
- `Team Dashboard`: team-level KPIs and outcomes
- `Player Report`: player shot profile in selected match context
- `Data Model`: star-schema table previews
"""
)
