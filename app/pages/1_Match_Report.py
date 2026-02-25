import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

try:
    from app.components.analysis_navigation import render_analysis_nav
    from app.components.data import (
        get_active_data_mode,
        get_lineup_events,
        get_lineup_players,
        get_shots,
        load_dimensions,
        load_match_events,
    )
    from app.components.event_classification import SET_PIECE_PATTERNS, derive_counterpress_regains, derive_event_labels
    from app.components.filters import top_filters_cascading
    from app.components.lineups import get_formation, get_starting_positions, get_starting_xi, get_unmapped_position_names
    from app.components.match_stats import (
        compute_match_stats,
        render_match_score_header,
        render_match_stats_panel,
        validate_goals_consistency,
    )
    from app.components.model_views import get_shots_view
    from app.components.passes import get_filtered_events, render_passes_section
    from app.components.ui import setup_page
    from app.components.viz import draw_pitch_figure, draw_split_lineup_pitch
except (ModuleNotFoundError, KeyError):
    from components.analysis_navigation import render_analysis_nav
    from components.data import (
        get_active_data_mode,
        get_lineup_events,
        get_lineup_players,
        get_shots,
        load_dimensions,
        load_match_events,
    )
    from components.event_classification import SET_PIECE_PATTERNS, derive_counterpress_regains, derive_event_labels
    from components.filters import top_filters_cascading
    from components.lineups import get_formation, get_starting_positions, get_starting_xi, get_unmapped_position_names
    from components.match_stats import (
        compute_match_stats,
        render_match_score_header,
        render_match_stats_panel,
        validate_goals_consistency,
    )
    from components.model_views import get_shots_view
    from components.passes import get_filtered_events, render_passes_section
    from components.ui import setup_page
    from components.viz import draw_pitch_figure, draw_split_lineup_pitch

setup_page(page_title="Match Report", page_icon=":bar_chart:")

MATCH_EVENT_COLUMNS = [
    "event_id",
    "match_id",
    "team_id",
    "player_id",
    "type_id",
    "type_name",
    "period",
    "minute",
    "second",
    "timestamp",
    "event_index",
    "location_x",
    "location_y",
    "pass_end_location_x",
    "pass_end_location_y",
    "carry_end_location_x",
    "carry_end_location_y",
    "under_pressure",
    "counterpress",
    "play_pattern_name",
    "pass_height",
    "pass_height_name",
    "pass_cross",
    "pass_outcome",
    "pass_outcome_name",
    "duel_outcome_name",
    "duel_outcome",
    "team_name",
    "player_name",
]


def _apply_team_player_filters(df: pd.DataFrame, team_id: int | None, player_id: int | None) -> pd.DataFrame:
    out = df.copy()
    if team_id is not None and "team_id" in out.columns:
        out = out[pd.to_numeric(out["team_id"], errors="coerce") == int(team_id)]
    if player_id is not None and "player_id" in out.columns:
        out = out[pd.to_numeric(out["player_id"], errors="coerce") == int(player_id)]
    return out


def _render_context_chips(selection: dict[str, object]) -> None:
    st.markdown('<div class="section-title">Context</div>', unsafe_allow_html=True)
    chips = [selection["competition_name"], selection["season_name"], selection["team_name"], selection["player_name"]]
    chips = [c for c in chips if c]
    if chips:
        st.markdown(" ".join([f'<span class="context-chip">{c}</span>' for c in chips]), unsafe_allow_html=True)


def _render_score_header(official_stats: dict[str, object]) -> None:
    st.markdown('<div class="section-title">Match Score</div>', unsafe_allow_html=True)
    render_match_score_header(official_stats)


def _render_stats_section(
    selection: dict[str, object],
    dim_match: pd.DataFrame,
    dim_player: pd.DataFrame,
    match_events: pd.DataFrame,
    match_shots: pd.DataFrame,
    official_stats: dict[str, object],
) -> None:
    match_id = int(selection["match_id"])
    team_id = selection["team_id"]
    player_id = selection["player_id"]

    apply_stats_filters = st.toggle(
        "Apply current filters to stats",
        value=False,
        help="OFF shows full match stats for both teams.",
        key="apply_stats_filters",
    )
    if apply_stats_filters:
        stats_events = get_filtered_events(match_id=match_id, team_id=team_id, player_id=player_id, events=match_events)
        stats_shots = _apply_team_player_filters(match_shots, team_id=team_id, player_id=player_id)
        stats_payload = compute_match_stats(
            fact_events=stats_events,
            fact_shots=stats_shots,
            dim_match=dim_match,
            match_id=match_id,
        )
    else:
        stats_payload = official_stats

    try:
        validate_goals_consistency(stats_payload, apply_filtered_stats=apply_stats_filters)
    except AssertionError as err:
        st.error(str(err))

    st.markdown('<div class="section-title">Match Stats</div>', unsafe_allow_html=True)
    render_match_stats_panel(stats_payload, filtered=apply_stats_filters)

    with st.spinner("Loading lineup context..."):
        lineup_events = get_lineup_events(match_id=match_id)
        lineup_players = get_lineup_players(match_id=match_id)

    st.markdown('<div class="section-title">Starting XI & Formation</div>', unsafe_allow_html=True)
    match_row = (
        dim_match[pd.to_numeric(dim_match["match_id"], errors="coerce") == match_id].iloc[0]
        if "match_id" in dim_match.columns and not dim_match.empty
        else pd.Series(dtype="object")
    )
    home_team_id = int(match_row["home_team_id"]) if "home_team_id" in match_row and pd.notna(match_row["home_team_id"]) else None
    away_team_id = int(match_row["away_team_id"]) if "away_team_id" in match_row and pd.notna(match_row["away_team_id"]) else None
    home_team_name = str(match_row.get("home_team_name") or "Home")
    away_team_name = str(match_row.get("away_team_name") or "Away")

    home_xi = get_starting_xi(
        lineup_events,
        match_id=match_id,
        team_id=home_team_id,
        team_name=home_team_name,
        lineup_players=lineup_players,
        dim_player=dim_player,
    )
    away_xi = get_starting_xi(
        lineup_events,
        match_id=match_id,
        team_id=away_team_id,
        team_name=away_team_name,
        lineup_players=lineup_players,
        dim_player=dim_player,
    )
    home_formation = get_formation(
        lineup_events,
        match_id=match_id,
        team_id=home_team_id,
        team_name=home_team_name,
        lineup_players=lineup_players,
        dim_player=dim_player,
    )
    away_formation = get_formation(
        lineup_events,
        match_id=match_id,
        team_id=away_team_id,
        team_name=away_team_name,
        lineup_players=lineup_players,
        dim_player=dim_player,
    )
    home_positions = get_starting_positions(
        lineup_events,
        match_id=match_id,
        team_id=home_team_id,
        team_name=home_team_name,
        formation=home_formation,
        selected_home_team_id=home_team_id,
        selected_away_team_id=away_team_id,
        lineup_players=lineup_players,
        dim_player=dim_player,
    )
    away_positions = get_starting_positions(
        lineup_events,
        match_id=match_id,
        team_id=away_team_id,
        team_name=away_team_name,
        formation=away_formation,
        selected_home_team_id=home_team_id,
        selected_away_team_id=away_team_id,
        lineup_players=lineup_players,
        dim_player=dim_player,
    )

    hdr_home, hdr_away = st.columns(2)
    with hdr_home:
        st.markdown(f"**{home_team_name} - {home_formation or 'Formation unknown'}**")
    with hdr_away:
        st.markdown(f"**{away_team_name} - {away_formation or 'Formation unknown'}**")
    st.plotly_chart(
        draw_split_lineup_pitch(
            home_positions=home_positions,
            away_positions=away_positions,
            subtitle=f"{home_team_name} vs {away_team_name}",
        ),
        use_container_width=True,
    )

    unmapped = sorted(
        set(
            get_unmapped_position_names(
                match_id,
                lineup_events,
                team_id=home_team_id,
                lineup_players=lineup_players,
                dim_player=dim_player,
            )
        )
        | set(
            get_unmapped_position_names(
                match_id,
                lineup_events,
                team_id=away_team_id,
                lineup_players=lineup_players,
                dim_player=dim_player,
            )
        )
    )
    if unmapped:
        st.warning("Unmapped positions placed in fallback midfield zone: " + ", ".join(unmapped))

    if bool(st.session_state.get("debug_lineup")):
        home_gk = next((p for p in home_positions if str(p.get("position_name") or "").strip().lower() == "goalkeeper"), None)
        away_gk = next((p for p in away_positions if str(p.get("position_name") or "").strip().lower() == "goalkeeper"), None)
        st.caption(
            "Lineup debug | "
            f"Home GK: ({home_gk.get('x') if home_gk else 'n/a'}, {home_gk.get('y') if home_gk else 'n/a'}) | "
            f"Away GK: ({away_gk.get('x') if away_gk else 'n/a'}, {away_gk.get('y') if away_gk else 'n/a'})"
        )

    if "source" in home_xi.columns and not home_xi.empty and not (home_xi["source"] == "data_raw_lineups").any():
        st.info("Home lineup JSON not found; showing fallback lineup from event participation.")
    if "source" in away_xi.columns and not away_xi.empty and not (away_xi["source"] == "data_raw_lineups").any():
        st.info("Away lineup JSON not found; showing fallback lineup from event participation.")

    col_home, col_away = st.columns(2)
    with col_home:
        st.markdown(f"**{home_team_name} • {home_formation or 'Formation unknown'}**")
        if home_xi.empty:
            st.warning("Starting XI unavailable for this team/match.")
        else:
            home_tbl = home_xi.copy()
            if "jersey_number" in home_tbl.columns:
                home_tbl["jersey_number"] = home_tbl["jersey_number"].fillna("?")
            display_cols = [c for c in ("jersey_number", "player_name", "position_name") if c in home_tbl.columns]
            st.dataframe(home_tbl[display_cols].head(11), use_container_width=True, hide_index=True)

    with col_away:
        st.markdown(f"**{away_team_name} • {away_formation or 'Formation unknown'}**")
        if away_xi.empty:
            st.warning("Starting XI unavailable for this team/match.")
        else:
            away_tbl = away_xi.copy()
            if "jersey_number" in away_tbl.columns:
                away_tbl["jersey_number"] = away_tbl["jersey_number"].fillna("?")
            display_cols = [c for c in ("jersey_number", "player_name", "position_name") if c in away_tbl.columns]
            st.dataframe(away_tbl[display_cols].head(11), use_container_width=True, hide_index=True)


def _render_shots_section(
    selection: dict[str, object],
    dim_match: pd.DataFrame,
    dim_team: pd.DataFrame,
    dim_player: pd.DataFrame,
    match_shots: pd.DataFrame,
) -> None:
    match_id = int(selection["match_id"])
    team_id = selection["team_id"]
    player_id = selection["player_id"]

    context_shots = _apply_team_player_filters(match_shots, team_id=team_id, player_id=player_id)
    shots = get_shots_view(context_shots, dim_team=dim_team, dim_player=dim_player)

    with st.expander("Shot filters", expanded=False):
        col_a, col_b, col_c, col_d = st.columns(4)
        fixed_direction = col_a.checkbox("Attacking direction fixed (left to right)", value=True, key="shots_fixed_direction")
        only_goals = col_b.checkbox("Show only goals", value=False, key="shots_only_goals")
        big_chances = col_c.checkbox("Show big chances (xG >= 0.30)", value=False, key="shots_big_chances")
        open_play_only = col_d.checkbox("Show open play only", value=False, key="shots_open_play_only")

    filtered = shots.copy()
    if only_goals:
        if "is_goal" in filtered.columns:
            filtered = filtered[filtered["is_goal"]]
        elif "shot_outcome" in filtered.columns:
            filtered = filtered[filtered["shot_outcome"].astype(str).str.strip().str.lower() == "goal"]
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
        dm = dim_match[pd.to_numeric(dim_match["match_id"], errors="coerce") == match_id]
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
    k3.metric(
        "Goals",
        int(filtered["is_goal"].sum())
        if "is_goal" in filtered.columns
        else int(filtered["shot_outcome"].astype(str).str.strip().str.lower().eq("goal").sum())
        if "shot_outcome" in filtered.columns
        else 0,
    )
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


def _render_duels_section(match_id: int, team_id: int | None, player_id: int | None, match_events: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Duels / Recoveries</div>', unsafe_allow_html=True)
    events = get_filtered_events(match_id=match_id, team_id=team_id, player_id=player_id, events=match_events)
    if events.empty or "type_name" not in events.columns:
        st.info("No events available for the current filter context.")
        return

    type_name = events["type_name"].astype("string").str.strip().str.lower()
    recoveries = int((type_name == "ball recovery").sum())
    interceptions = int((type_name == "interception").sum())
    duels = int((type_name == "duel").sum())
    tackles = int((type_name == "tackle").sum()) if (type_name == "tackle").any() else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recoveries", recoveries)
    c2.metric("Interceptions", interceptions)
    c3.metric("Duels", duels)
    c4.metric("Tackles", tackles)

    st.markdown(
        """
        <div class="coming-soon-card">
          <div class="coming-soon-title">Positional Duel Map</div>
          <div class="coming-soon-subtitle">Coming soon: zone-level defensive event maps and trend splits.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _event_type_mask(df: pd.DataFrame, event_type: str) -> pd.Series:
    if "type_name" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["type_name"].astype("string").str.strip().str.lower().eq(event_type.strip().lower())


def _duel_won_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(False, index=df.index)
    is_duel = _event_type_mask(df, "Duel") | _event_type_mask(df, "50/50")
    outcome = (
        df["duel_outcome_name"]
        if "duel_outcome_name" in df.columns
        else df["duel_outcome"]
        if "duel_outcome" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    won = outcome.astype("string").str.strip().str.lower().isin({"won", "win", "success", "success in play", "success out"})
    return is_duel & won.fillna(False)


def _pass_completion_mask(df: pd.DataFrame) -> pd.Series:
    outcome = (
        df["pass_outcome"]
        if "pass_outcome" in df.columns
        else df["pass_outcome_name"]
        if "pass_outcome_name" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    norm = outcome.astype("string").str.strip().str.lower()
    return norm.isna() | norm.eq("") | norm.eq("none")


def _progressive_pass_mask(df: pd.DataFrame) -> pd.Series:
    if not {"location_x", "pass_end_location_x"}.issubset(df.columns):
        return pd.Series(False, index=df.index)
    start_x = pd.to_numeric(df["location_x"], errors="coerce")
    end_x = pd.to_numeric(df["pass_end_location_x"], errors="coerce")
    return (end_x - start_x).ge(15).fillna(False)


def _final_third_pass_mask(df: pd.DataFrame) -> pd.Series:
    if "pass_end_location_x" not in df.columns:
        return pd.Series(False, index=df.index)
    end_x = pd.to_numeric(df["pass_end_location_x"], errors="coerce")
    if end_x.dropna().empty:
        return pd.Series(False, index=df.index)
    threshold = 80.0 if float(end_x.max()) > 101.0 else (2.0 / 3.0) * 100.0
    return end_x.ge(threshold).fillna(False)


def _render_bucket_debug(events: pd.DataFrame, team_id: int | None, player_id: int | None) -> None:
    with st.expander("Developer tools", expanded=False):
        show_debug = st.checkbox(
            "Show event bucket counts (debug)",
            value=False,
            help="Developer-only debug table for bucket classification counts in current filter context.",
        )
        if not show_debug:
            return
        scoped = _apply_team_player_filters(events, team_id=team_id, player_id=player_id)
        if "bucket" not in scoped.columns or scoped.empty:
            st.info("No bucket data available.")
            return
        counts = scoped["bucket"].astype("string").value_counts(dropna=False).rename_axis("bucket").reset_index(name="count")
        st.dataframe(counts, use_container_width=True, hide_index=True)


def _render_offensive_panel(events: pd.DataFrame, shots_df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Offensive - Chance Creation & Progression</div>', unsafe_allow_html=True)
    st.caption("Plain-language view of how your team progressed the ball and created chances.")
    if events.empty:
        st.info("No offensive events in current context.")
        return

    offensive = events[events["bucket"].astype("string") == "OFFENSIVE"] if "bucket" in events.columns else events.copy()
    pass_events = offensive[_event_type_mask(offensive, "Pass")]
    dribbles = offensive[_event_type_mask(offensive, "Dribble")]
    shots = shots_df.copy()

    completed = int(_pass_completion_mask(pass_events).sum()) if not pass_events.empty else 0
    progressive = int(_progressive_pass_mask(pass_events).sum()) if not pass_events.empty else 0
    final_third = int(_final_third_pass_mask(pass_events).sum()) if not pass_events.empty else 0
    dribble_success = int(_duel_won_mask(dribbles).sum()) if not dribbles.empty else 0
    xg_total = float(pd.to_numeric(shots["xg"], errors="coerce").fillna(0).sum()) if "xg" in shots.columns else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total passes", int(len(pass_events)))
    k2.metric("Completed passes", completed)
    k3.metric("Progressive passes", progressive)
    k4.metric("Passes to final third", final_third)

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Dribbles attempted", int(len(dribbles)))
    k6.metric("Dribbles success", dribble_success)
    k7.metric("Shots", int(len(shots)))
    k8.metric("xG", f"{xg_total:.2f}")

    if {"minute", "type_name", "team_name", "player_name"}.issubset(offensive.columns):
        cols = ["minute", "type_name", "team_name", "player_name"]
        st.dataframe(offensive[cols].sort_values("minute").head(30), use_container_width=True, hide_index=True)


def _render_defensive_panel(events: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Defensive - Regains & Disruption</div>', unsafe_allow_html=True)
    st.caption("Tracks how often your team disrupted attacks and recovered possession.")
    if events.empty:
        st.info("No defensive events in current context.")
        return

    defensive = events[events["bucket"].astype("string") == "DEFENSIVE"] if "bucket" in events.columns else events.copy()
    pressures = int(_event_type_mask(defensive, "Pressure").sum())
    counterpress = int(defensive["is_counterpress"].sum()) if "is_counterpress" in defensive.columns else 0
    interceptions = int(_event_type_mask(defensive, "Interception").sum())
    recoveries = int(_event_type_mask(defensive, "Ball Recovery").sum())
    duels_won = int(_duel_won_mask(defensive).sum())
    blocks = int(_event_type_mask(defensive, "Block").sum())
    clearances = int(_event_type_mask(defensive, "Clearance").sum())
    fouls = int(_event_type_mask(defensive, "Foul Committed").sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pressures", pressures)
    k2.metric("Counterpress actions", counterpress)
    k3.metric("Interceptions", interceptions)
    k4.metric("Ball recoveries", recoveries)

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Duels won", duels_won)
    k6.metric("Blocks", blocks)
    k7.metric("Clearances", clearances)
    k8.metric("Fouls committed", fouls)


def _render_transitions_panel(events: pd.DataFrame, shots_df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Transitions - Turnovers & Counterpress</div>', unsafe_allow_html=True)
    st.caption("Shows what happens immediately after possession changes.")
    if events.empty:
        st.info("No transition events in current context.")
        return

    transition = events[events["bucket"].astype("string") == "TRANSITION"] if "bucket" in events.columns else events.copy()
    turnovers = int(transition["is_turnover"].sum()) if "is_turnover" in transition.columns else 0
    counterpress = int(transition["is_counterpress"].sum()) if "is_counterpress" in transition.columns else 0
    counter_regains = int(transition["is_counterpress_regain"].sum()) if "is_counterpress_regain" in transition.columns else 0
    counters = int(transition["is_counter"].sum()) if "is_counter" in transition.columns else 0

    shots_counter = pd.DataFrame()
    if "play_pattern_name" in shots_df.columns:
        shots_counter = shots_df[
            shots_df["play_pattern_name"].astype("string").str.strip().str.lower().eq("from counter")
        ].copy()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Turnovers", turnovers)
    k2.metric("Counterpress", counterpress)
    k3.metric("Regains in 6s", counter_regains)
    k4.metric("Counters", counters)
    k5.metric("Shots from counters", int(len(shots_counter)))

    if {"minute", "type_name", "is_turnover", "is_counterpress_regain"}.issubset(transition.columns):
        cols = ["minute", "type_name", "is_turnover", "is_counterpress_regain"]
        st.dataframe(transition[cols].sort_values("minute").head(30), use_container_width=True, hide_index=True)


def _render_set_piece_panel(events: pd.DataFrame, shots_df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Set Pieces</div>', unsafe_allow_html=True)
    st.caption("Set-piece view for corners, free kicks, throw-ins, goal kicks, and kick-offs.")
    if events.empty:
        st.info("No set-piece events in current context.")
        return

    set_piece_events = events[events["bucket"].astype("string") == "SET_PIECE"] if "bucket" in events.columns else events.copy()
    set_piece_passes = int(_event_type_mask(set_piece_events, "Pass").sum())
    set_piece_shots = shots_df[
        shots_df["play_pattern_name"].astype("string").str.strip().str.lower().isin(SET_PIECE_PATTERNS)
    ] if "play_pattern_name" in shots_df.columns else shots_df.iloc[0:0].copy()
    set_piece_goals = (
        int(set_piece_shots["shot_outcome"].astype("string").str.strip().str.lower().eq("goal").sum())
        if "shot_outcome" in set_piece_shots.columns
        else 0
    )
    set_piece_xg = float(pd.to_numeric(set_piece_shots["xg"], errors="coerce").fillna(0).sum()) if "xg" in set_piece_shots.columns else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Set-piece passes", set_piece_passes)
    k2.metric("Set-piece shots", int(len(set_piece_shots)))
    k3.metric("Set-piece xG", f"{set_piece_xg:.2f}")
    k4.metric("Set-piece goals", set_piece_goals)

    if "play_pattern_name" in set_piece_events.columns:
        breakdown = (
            set_piece_events["play_pattern_name"]
            .astype("string")
            .value_counts()
            .rename_axis("set_piece_type")
            .reset_index(name="events")
            .head(12)
        )
        st.dataframe(breakdown, use_container_width=True, hide_index=True)


def _render_more_section() -> None:
    st.markdown('<div class="section-title">More</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="coming-soon-card">
          <div class="coming-soon-title">Additional Analysis Modules</div>
          <div class="coming-soon-subtitle">Future additions: pass networks, pressing waves, and possession phases.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


dim_match, dim_team, dim_player = load_dimensions()
st.title("Match Report")

with st.expander("Filters", expanded=True):
    selection = top_filters_cascading(dim_match)

match_id = selection["match_id"]
if match_id is None:
    st.warning("No match available for the current filter context.")
    st.stop()

st.caption(selection["match_label"] or "")
with st.spinner("Loading match context..."):
    active_data_mode = get_active_data_mode()
    match_events = load_match_events(match_id=int(match_id), data_mode=active_data_mode, columns=MATCH_EVENT_COLUMNS)
    match_events = derive_event_labels(match_events)
    match_events = derive_counterpress_regains(match_events, window_seconds=6.0)
    match_shots = get_shots(match_id=match_id)

official_stats = compute_match_stats(
    fact_events=match_events,
    fact_shots=match_shots,
    dim_match=dim_match,
    match_id=int(match_id),
)
_render_score_header(official_stats)

analysis_view = render_analysis_nav(current_view=str(st.session_state.get("analysis_view", "Stats")))
_render_context_chips(selection)
_render_bucket_debug(match_events, team_id=selection["team_id"], player_id=selection["player_id"])

if analysis_view == "Stats":
    _render_stats_section(
        selection=selection,
        dim_match=dim_match,
        dim_player=dim_player,
        match_events=match_events,
        match_shots=match_shots,
        official_stats=official_stats,
    )
elif analysis_view == "Shots":
    _render_shots_section(
        selection=selection,
        dim_match=dim_match,
        dim_team=dim_team,
        dim_player=dim_player,
        match_shots=match_shots,
    )
elif analysis_view == "Passes":
    match_row = (
        dim_match[pd.to_numeric(dim_match["match_id"], errors="coerce") == int(match_id)].iloc[0]
        if "match_id" in dim_match.columns and not dim_match.empty
        else pd.Series(dtype="object")
    )
    home_team_id = int(match_row["home_team_id"]) if "home_team_id" in match_row and pd.notna(match_row["home_team_id"]) else None
    away_team_id = int(match_row["away_team_id"]) if "away_team_id" in match_row and pd.notna(match_row["away_team_id"]) else None
    home_team_name = str(match_row.get("home_team_name") or "Home")
    away_team_name = str(match_row.get("away_team_name") or "Away")
    render_passes_section(
        match_id=int(match_id),
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_team_name=home_team_name,
        away_team_name=away_team_name,
        selected_team_id=selection["team_id"],
        selected_player_id=selection["player_id"],
        events=match_events,
    )
elif analysis_view == "Duels / Recoveries":
    _render_duels_section(
        match_id=int(match_id),
        team_id=selection["team_id"],
        player_id=selection["player_id"],
        match_events=match_events,
    )
elif analysis_view == "Offensive":
    filtered_events = _apply_team_player_filters(match_events, team_id=selection["team_id"], player_id=selection["player_id"])
    filtered_shots = _apply_team_player_filters(match_shots, team_id=selection["team_id"], player_id=selection["player_id"])
    offensive_shots = get_shots_view(filtered_shots, dim_team=dim_team, dim_player=dim_player)
    _render_offensive_panel(filtered_events, offensive_shots)
elif analysis_view == "Defensive":
    filtered_events = _apply_team_player_filters(match_events, team_id=selection["team_id"], player_id=selection["player_id"])
    _render_defensive_panel(filtered_events)
elif analysis_view == "Transitions":
    filtered_events = _apply_team_player_filters(match_events, team_id=selection["team_id"], player_id=selection["player_id"])
    filtered_shots = _apply_team_player_filters(match_shots, team_id=selection["team_id"], player_id=selection["player_id"])
    transition_shots = get_shots_view(filtered_shots, dim_team=dim_team, dim_player=dim_player)
    _render_transitions_panel(filtered_events, transition_shots)
elif analysis_view == "Set Pieces":
    filtered_events = _apply_team_player_filters(match_events, team_id=selection["team_id"], player_id=selection["player_id"])
    filtered_shots = _apply_team_player_filters(match_shots, team_id=selection["team_id"], player_id=selection["player_id"])
    set_piece_shots = get_shots_view(filtered_shots, dim_team=dim_team, dim_player=dim_player)
    _render_set_piece_panel(filtered_events, set_piece_shots)
else:
    _render_more_section()
