import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from app.components.data import get_events, get_shots, load_dimensions
from app.components.filters import sidebar_filters_cascading
from app.components.model_views import get_events_view, get_shots_view
from app.components.ui import setup_page

setup_page(page_title="Team Dashboard", page_icon="🧠")

dim_match, dim_team, dim_player = load_dimensions()
selection = sidebar_filters_cascading(dim_match)
match_id = selection["match_id"]
team_id = selection["team_id"]
player_id = selection["player_id"]

if match_id is None:
    st.warning("No match available for the current filter context.")
    st.stop()

st.title("Team Dashboard")
st.caption(selection["match_label"] or "")

with st.spinner("Loading team context..."):
    fact_shots = get_shots(match_id=match_id, team_id=team_id, player_id=player_id)
    fact_events = get_events(match_id=match_id, team_id=team_id, player_id=player_id)

shots = get_shots_view(fact_shots, dim_team=dim_team, dim_player=dim_player)
events = get_events_view(fact_events, dim_team=dim_team, dim_player=dim_player)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Shots", len(shots))
k2.metric("Total xG", f"{shots['xg'].fillna(0).sum():.2f}" if "xg" in shots.columns else "-")
k3.metric(
    "Goals",
    int(shots["is_goal"].sum())
    if "is_goal" in shots.columns
    else int(shots["shot_outcome"].astype(str).str.strip().str.lower().eq("goal").sum())
    if "shot_outcome" in shots.columns
    else 0,
)
k4.metric("Events", len(events))

left, right = st.columns(2)
with left:
    st.markdown('<div class="section-title">Shot Outcomes</div>', unsafe_allow_html=True)
    if "shot_outcome" in shots.columns and len(shots):
        outcomes = shots["shot_outcome"].fillna("Unknown").value_counts().reset_index()
        outcomes.columns = ["shot_outcome", "count"]
        fig = px.bar(outcomes, x="shot_outcome", y="count", color="shot_outcome")
        fig.update_layout(
            showlegend=False,
            paper_bgcolor="#0b1220",
            plot_bgcolor="#111a2b",
            font=dict(color="#e7edf7"),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No shot outcomes available in selected context.")

with right:
    st.markdown('<div class="section-title">Top Shooters</div>', unsafe_allow_html=True)
    if {"player_name", "xg"}.issubset(shots.columns) and len(shots):
        top_players = (
            shots.groupby("player_name", as_index=False)
            .agg(shots=("player_name", "size"), total_xg=("xg", "sum"))
            .sort_values(["shots", "total_xg"], ascending=False)
            .head(10)
        )
        fig2 = px.bar(top_players, x="player_name", y="shots", hover_data=["total_xg"])
        fig2.update_layout(
            paper_bgcolor="#0b1220",
            plot_bgcolor="#111a2b",
            font=dict(color="#e7edf7"),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Player names are required for top-shooter chart.")

st.markdown('<div class="section-title">Event Table (Top 200)</div>', unsafe_allow_html=True)
st.dataframe(events.head(200), use_container_width=True)
