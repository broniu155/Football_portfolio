import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from app.components.data import get_events, get_shots, load_dimensions
from app.components.filters import sidebar_filters_cascading
from app.components.match_stats import (
    compute_match_stats,
    render_match_score_header,
    render_match_stats_panel,
    validate_goals_consistency,
)
from app.components.model_views import get_shots_view
from app.components.ui import setup_page
from app.components.viz import draw_pitch_figure

setup_page(page_title="Match Report", page_icon=":bar_chart:")


def _apply_context_filters(df: pd.DataFrame, team_id: int | None, player_id: int | None) -> pd.DataFrame:
    out = df.copy()
    if team_id is not None and "team_id" in out.columns:
        out = out[pd.to_numeric(out["team_id"], errors="coerce") == int(team_id)]
    if player_id is not None and "player_id" in out.columns:
        out = out[pd.to_numeric(out["player_id"], errors="coerce") == int(player_id)]
    return out


dim_match, dim_team, dim_player = load_dimensions()
selection = sidebar_filters_cascading(dim_match)
match_id = selection["match_id"]
team_id = selection["team_id"]
player_id = selection["player_id"]

if match_id is None:
    st.warning("No match available for the current filter context.")
    st.stop()

st.title("Match Report")
st.caption(selection["match_label"] or "")

with st.spinner("Loading match context..."):
    match_events = get_events(match_id=match_id)
    match_shots = get_shots(match_id=match_id)
    context_shots = get_shots(match_id=match_id, team_id=team_id, player_id=player_id)

shots = get_shots_view(context_shots, dim_team=dim_team, dim_player=dim_player)

official_stats = compute_match_stats(
    fact_events=match_events,
    fact_shots=match_shots,
    dim_match=dim_match,
    match_id=int(match_id),
)

st.markdown('<div class="section-title">Match Score</div>', unsafe_allow_html=True)
render_match_score_header(official_stats)

apply_stats_filters = st.toggle(
    "Apply current filters to stats",
    value=False,
    help="OFF shows full match stats for both teams.",
)
stats_events = _apply_context_filters(match_events, team_id=team_id, player_id=player_id) if apply_stats_filters else match_events
stats_shots = _apply_context_filters(match_shots, team_id=team_id, player_id=player_id) if apply_stats_filters else match_shots
stats_payload = compute_match_stats(
    fact_events=stats_events,
    fact_shots=stats_shots,
    dim_match=dim_match,
    match_id=int(match_id),
)
if not apply_stats_filters:
    stats_payload["metrics"]["Goals"] = (official_stats.get("home_score"), official_stats.get("away_score"))

try:
    validate_goals_consistency(stats_payload, apply_filtered_stats=apply_stats_filters)
except AssertionError as err:
    st.error(str(err))

st.markdown('<div class="section-title">Match Stats</div>', unsafe_allow_html=True)
render_match_stats_panel(stats_payload, filtered=apply_stats_filters)

st.markdown('<div class="section-title">Context</div>', unsafe_allow_html=True)
chips = [selection["competition_name"], selection["season_name"], selection["team_name"], selection["player_name"]]
chips = [c for c in chips if c]
if chips:
    st.markdown(" ".join([f'<span class="context-chip">{c}</span>' for c in chips]), unsafe_allow_html=True)

with st.expander("Shot filters", expanded=False):
    col_a, col_b, col_c, col_d = st.columns(4)
    fixed_direction = col_a.checkbox("Attacking direction fixed (left to right)", value=True)
    only_goals = col_b.checkbox("Show only goals", value=False)
    big_chances = col_c.checkbox("Show big chances (xG >= 0.30)", value=False)
    open_play_only = col_d.checkbox("Show open play only", value=False)

filtered = shots.copy()
if only_goals and "shot_outcome" in filtered.columns:
    filtered = filtered[filtered["shot_outcome"].astype(str).str.lower() == "goal"]
if big_chances and "xg" in filtered.columns:
    filtered = filtered[filtered["xg"].fillna(0) >= 0.3]
if open_play_only:
    if "play_pattern_name" in filtered.columns:
        filtered = filtered[filtered["play_pattern_name"].astype(str).str.lower() == "regular play"]
    elif "shot_type" in filtered.columns:
        filtered = filtered[~filtered["shot_type"].astype(str).str.lower().isin(["penalty", "free kick"])]
    else:
        st.info("Open-play filtering is unavailable because play-pattern columns are not present.")

if fixed_direction and team_id is not None and {"home_team_id", "away_team_id", "match_id"}.issubset(dim_match.columns):
    dm = dim_match[pd.to_numeric(dim_match["match_id"], errors="coerce") == int(match_id)]
    if not dm.empty:
        home_team_id = int(dm.iloc[0]["home_team_id"])
        if int(team_id) != home_team_id and {"x", "y"}.issubset(filtered.columns):
            filtered["x"] = 120 - filtered["x"]
            filtered["y"] = 80 - filtered["y"]
elif fixed_direction and team_id is None:
    st.info("Attacking-direction normalization is most reliable when one team is selected.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Shots", len(filtered))
k2.metric("Total xG", f"{filtered['xg'].fillna(0).sum():.2f}" if "xg" in filtered.columns else "-")
k3.metric("Goals", int((filtered["shot_outcome"] == "Goal").sum()) if "shot_outcome" in filtered.columns else 0)
k4.metric("Players", filtered["player_id"].nunique() if "player_id" in filtered.columns else 0)

left, right = st.columns([1.35, 1])
with left:
    st.markdown('<div class="section-title">Shot Map</div>', unsafe_allow_html=True)
    fig = draw_pitch_figure(
        filtered,
        title="Coach Shot Map",
        subtitle=selection["match_label"] or "",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown('<div class="section-title">Cumulative xG Timeline</div>', unsafe_allow_html=True)
    if {"minute", "xg"}.issubset(filtered.columns) and len(filtered):
        timeline = filtered.groupby("minute", as_index=False)["xg"].sum().sort_values("minute")
        timeline["cum_xg"] = timeline["xg"].cumsum()
        fig2 = px.line(timeline, x="minute", y="cum_xg", markers=True)
        fig2.update_layout(
            paper_bgcolor="#0b1220",
            plot_bgcolor="#111a2b",
            font=dict(color="#e7edf7"),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
            margin=dict(l=10, r=10, t=30, b=80),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Minute + xG columns are required for timeline rendering.")

st.markdown('<div class="section-title">Shots Table (Top 200)</div>', unsafe_allow_html=True)
st.dataframe(filtered.head(200), use_container_width=True)
