import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from app.components.data import load_dimensions, load_fact_shots
from app.components.filters import sidebar_filters
from app.components.model_views import get_shots_view

st.set_page_config(page_title="Match Report", page_icon="📊", layout="wide")

dim_match, dim_team, dim_player = load_dimensions()
match_id, team_name, player_name = sidebar_filters(dim_match, dim_team, dim_player)

st.title("📊 Match Report")

with st.spinner("Loading match data..."):
    fact_shots = load_fact_shots(match_id)

shots_view = get_shots_view(fact_shots, dim_team=dim_team, dim_player=dim_player)
shots = shots_view[shots_view["match_id"] == match_id].copy()
if team_name and "team_name" in shots.columns:
    shots = shots[shots["team_name"] == team_name]
if player_name and "player_name" in shots.columns:
    shots = shots[shots["player_name"] == player_name]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Shots", len(shots))
if "xg" in shots.columns:
    k2.metric("Total xG", f"{shots['xg'].sum():.2f}")
if "shot_outcome" in shots.columns:
    k3.metric("Goals", int((shots["shot_outcome"] == "Goal").sum()))
if "minute" in shots.columns and len(shots):
    k4.metric("Last Shot Min", int(shots["minute"].max()))
else:
    k4.metric("Last Shot Min", 0)

st.markdown("---")
left, right = st.columns([1.3, 1])

with left:
    st.subheader("Shot map")
    needed = {"x", "y"}
    if needed.issubset(shots.columns):
        size_col = "xg" if "xg" in shots.columns else None
        color_col = "shot_outcome" if "shot_outcome" in shots.columns else None
        fig = px.scatter(
            shots,
            x="x",
            y="y",
            size=size_col,
            color=color_col,
            hover_data=[c for c in ["minute", "player_name", "xg", "shot_outcome"] if c in shots.columns],
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("fact_shots.csv needs x/y columns to plot a shot map.")

with right:
    st.subheader("xG timeline")
    if {"minute", "xg"}.issubset(shots.columns):
        s = shots.groupby("minute", as_index=False)["xg"].sum()
        s["cum_xg"] = s["xg"].cumsum()
        fig2 = px.line(s, x="minute", y="cum_xg")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Need minute + xg columns for timeline.")

st.subheader("Shots table")
st.dataframe(shots.head(200), use_container_width=True)
