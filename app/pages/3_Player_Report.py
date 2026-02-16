import streamlit as st
import plotly.express as px
from app.components.data import load_star_schema
from app.components.filters import sidebar_filters

st.set_page_config(page_title="Player Report", page_icon="👤", layout="wide")

dim_match, dim_team, dim_player, fact_events, fact_shots = load_star_schema()
match_id, team_name, player_name = sidebar_filters(dim_match, dim_team, dim_player)

st.title("👤 Player Report")

if not player_name:
    st.info("Select a player from the sidebar to view a player report.")
    st.stop()

shots = fact_shots[fact_shots["match_id"] == match_id].copy()
if "player_name" in shots.columns:
    shots = shots[shots["player_name"] == player_name]

c1, c2, c3 = st.columns(3)
c1.metric("Shots", len(shots))
if "xg" in shots.columns: c2.metric("Total xG", f"{shots['xg'].sum():.2f}")
if "shot_outcome" in shots.columns: c3.metric("Goals", int((shots["shot_outcome"] == "Goal").sum()))

if {"x","y"}.issubset(shots.columns):
    fig = px.scatter(shots, x="x", y="y", size="xg" if "xg" in shots.columns else None,
                     color="shot_outcome" if "shot_outcome" in shots.columns else None)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Shots table")
st.dataframe(shots.head(200), use_container_width=True)
