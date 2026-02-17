import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from app.components.data import load_star_schema
from app.components.filters import sidebar_filters

st.set_page_config(page_title="Team Dashboard", page_icon="🧠", layout="wide")

dim_match, dim_team, dim_player, fact_events, fact_shots = load_star_schema()
match_id, team_name, player_name = sidebar_filters(dim_match, dim_team, dim_player)

st.title("🧠 Team Dashboard")

shots = fact_shots[fact_shots["match_id"] == match_id].copy()
if "team_name" in shots.columns:
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
events = fact_events[fact_events["match_id"] == match_id].copy()
if "team_name" in events.columns:
    events = events[events["team_name"] == team_name]
st.dataframe(events.head(200), use_container_width=True)
