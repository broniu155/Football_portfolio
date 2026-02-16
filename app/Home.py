import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Football Portfolio", page_icon="⚽", layout="wide")

css = Path("app/assets/style.css")
if css.exists():
    st.markdown(f"<style>{css.read_text()}</style>", unsafe_allow_html=True)

st.title("⚽ Football Data Analyst Portfolio")
st.caption("StatsBomb open data • ETL → Star Schema → Interactive dashboards")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Focus", "Match & Player Insights")
c2.metric("Data Model", "Star Schema")
c3.metric("Stack", "Python + Streamlit")
c4.metric("Visuals", "Shots, xG, Maps")

st.markdown("---")

st.subheader("Pages")
st.markdown("""
Use the sidebar to navigate:
- **Match Report**: shot map + xG timeline + key shots table  
- **Team Dashboard**: team shot profile + event summaries  
- **Player Report**: player shot profile + involvement  
- **Data Model**: preview star schema tables  
""")
