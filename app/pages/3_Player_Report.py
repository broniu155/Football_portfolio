import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from app.components.data import get_shots, load_dimensions
from app.components.filters import sidebar_filters_cascading
from app.components.model_views import get_shots_view
from app.components.ui import setup_page
from app.components.viz import draw_pitch_figure

setup_page(page_title="Player Report", page_icon="👤")

dim_match, dim_team, dim_player = load_dimensions()
selection = sidebar_filters_cascading(dim_match)
match_id = selection["match_id"]
team_id = selection["team_id"]
player_id = selection["player_id"]

st.title("Player Report")
st.caption(selection["match_label"] or "")

if player_id is None:
    st.info("Select a player from the sidebar to view the player report.")
    st.stop()

with st.spinner("Loading player context..."):
    raw_shots = get_shots(match_id=match_id, team_id=team_id, player_id=player_id)
shots = get_shots_view(raw_shots, dim_team=dim_team, dim_player=dim_player)

k1, k2, k3 = st.columns(3)
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

left, right = st.columns([1.35, 1])
with left:
    st.markdown('<div class="section-title">Player Shot Map</div>', unsafe_allow_html=True)
    fig = draw_pitch_figure(
        shots,
        title=f"{selection['player_name']} Shot Map",
        subtitle=selection["match_label"] or "",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown('<div class="section-title">Shot Timing</div>', unsafe_allow_html=True)
    if "minute" in shots.columns and len(shots):
        minute_counts = shots["minute"].value_counts().sort_index().reset_index()
        minute_counts.columns = ["minute", "shots"]
        fig2 = px.bar(minute_counts, x="minute", y="shots")
        fig2.update_layout(
            paper_bgcolor="#0b1220",
            plot_bgcolor="#111a2b",
            font=dict(color="#e7edf7"),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Minute column not available for timing chart.")

st.markdown('<div class="section-title">Shots Table (Top 200)</div>', unsafe_allow_html=True)
st.dataframe(shots.head(200), use_container_width=True)
