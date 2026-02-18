import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from app.components.data import load_dimensions, load_fact_events, load_fact_shots
from app.components.filters import sidebar_filters
from app.components.model_views import get_events_view, get_shots_view

st.set_page_config(page_title="Team Dashboard", page_icon="🧠", layout="wide")

dim_match, dim_team, dim_player = load_dimensions()
match_id, team_name, player_name = sidebar_filters(dim_match, dim_team, dim_player)

st.title("🧠 Team Dashboard")

with st.spinner("Loading match data..."):
    fact_shots = load_fact_shots(match_id)
    fact_events = load_fact_events(match_id)

shots_view = get_shots_view(fact_shots, dim_team=dim_team, dim_player=dim_player)
events_view = get_events_view(fact_events, dim_team=dim_team, dim_player=dim_player)

shots = shots_view[shots_view["match_id"] == match_id].copy()
if team_name and "team_name" in shots.columns:
    shots = shots[shots["team_name"] == team_name]

st.markdown("### Team shot profile")
c1, c2, c3 = st.columns(3)
c1.metric("Shots", len(shots))
if "xg" in shots.columns:
    c2.metric("Total xG", f"{shots['xg'].sum():.2f}")
if "shot_outcome" in shots.columns:
    c3.metric("Goals", int((shots["shot_outcome"] == "Goal").sum()))

if "shot_outcome" in shots.columns:
    fig = px.histogram(shots, x="shot_outcome")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Event summary (match/team)")
events = events_view[events_view["match_id"] == match_id].copy()
if team_name and "team_name" in events.columns:
    events = events[events["team_name"] == team_name]
st.dataframe(events.head(200), use_container_width=True)
