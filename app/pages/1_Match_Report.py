import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
        load_match_passes,
    )
    from app.components.event_classification import SET_PIECE_PATTERNS, derive_counterpress_regains, derive_event_labels
    from app.components.export import ESSENTIAL_COLUMNS, build_match_events_export_df, events_df_to_csv_bytes
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
    from app.components.passes_metrics import PROGRESSIVE_THRESHOLD_DEFAULT, summarize_channels, top_progressive_passers, with_pass_features
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
        load_match_passes,
    )
    from components.event_classification import SET_PIECE_PATTERNS, derive_counterpress_regains, derive_event_labels
    from components.export import ESSENTIAL_COLUMNS, build_match_events_export_df, events_df_to_csv_bytes
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
    from components.passes_metrics import PROGRESSIVE_THRESHOLD_DEFAULT, summarize_channels, top_progressive_passers, with_pass_features
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

MATCH_PASS_COLUMNS = [
    "match_id",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
    "type_name",
    "location_x",
    "location_y",
    "pass_end_location_x",
    "pass_end_location_y",
    "pass_outcome_name",
    "pass_outcome_id",
    "minute",
    "second",
    "event_index",
    "index",
]

EXPORT_HINT_COLUMNS = [
    "event_id",
    "match_id",
    "event_index",
    "index",
    "period",
    "timestamp",
    "minute",
    "second",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
    "position_name",
    "type_id",
    "type_name",
    "sub_type",
    "play_pattern_name",
    "possession",
    "possession_team_name",
    "location_x",
    "location_y",
    "shot_outcome_name",
    "shot_outcome_id",
    "shot_type_name",
    "shot_type_id",
    "shot_body_part_name",
    "shot_body_part_id",
    "shot_end_location_x",
    "shot_end_location_y",
    "shot_end_location_z",
    "shot_statsbomb_xg",
    "pass_outcome_name",
    "pass_outcome_id",
    "pass_height_name",
    "pass_height_id",
    "pass_cross",
    "pass_length",
    "pass_angle",
    "pass_end_location_x",
    "pass_end_location_y",
    "pass_recipient_name",
    "pass_recipient_id",
    "assisted_shot_id",
    "duel_type_name",
    "duel_type_id",
    "duel_outcome_name",
    "duel_outcome_id",
    "foul_committed",
    "foul_won",
    "card_name",
    "offside",
]


def _apply_team_player_filters(df: pd.DataFrame, team_id: int | None, player_id: int | None) -> pd.DataFrame:
    out = df.copy()
    if team_id is not None and "team_id" in out.columns:
        out = out[pd.to_numeric(out["team_id"], errors="coerce") == int(team_id)]
    if player_id is not None and "player_id" in out.columns:
        out = out[pd.to_numeric(out["player_id"], errors="coerce") == int(player_id)]
    return out


@st.cache_data(show_spinner=False, ttl=600, max_entries=64)
def _build_match_export_cached(
    match_id: int,
    data_mode: str,
    include_derived: bool,
    essential_only: bool,
    apply_filters: bool,
    team_id: int | None,
    player_id: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    raw = load_match_events(match_id=int(match_id), data_mode=data_mode, columns=None)
    if raw.empty:
        return raw.copy(), []
    if apply_filters:
        raw = _apply_team_player_filters(raw, team_id=team_id, player_id=player_id)
    missing_hint_cols = [col for col in EXPORT_HINT_COLUMNS if col not in raw.columns]
    export_df = build_match_events_export_df(raw, include_derived=include_derived, essential_only=essential_only)
    return export_df, missing_hint_cols


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

    _render_match_export_panel(
        match_id=match_id,
        match_label=str(selection.get("match_label") or f"match_{match_id}"),
        data_mode=get_active_data_mode(),
        team_id=team_id,
        player_id=player_id,
    )


def _render_match_export_panel(
    match_id: int,
    match_label: str,
    data_mode: str,
    team_id: int | None,
    player_id: int | None,
) -> None:
    with st.expander("Debug / Export", expanded=False):
        st.caption("Export match events from the active data source for validation and reconciliation.")
        include_derived = st.checkbox(
            "Include derived helper columns",
            value=True,
            help="Adds helper flags like pass_completed, progressive_pass, attack_channel, and shot_is_goal.",
            key="export_include_derived",
        )
        essential_only = st.checkbox(
            "Limit columns to essentials (smaller file)",
            value=False,
            help="Exports a compact validation-focused column set.",
            key="export_essential_only",
        )
        apply_filters = st.checkbox(
            "Apply current team/player filters",
            value=False,
            help="OFF exports all events for the selected match. ON applies current team/player filters.",
            key="export_apply_filters",
        )

        params_sig = (
            int(match_id),
            str(data_mode),
            bool(include_derived),
            bool(essential_only),
            bool(apply_filters),
            int(team_id) if team_id is not None else None,
            int(player_id) if player_id is not None else None,
        )
        if st.button("Generate CSV", key="export_generate_csv"):
            with st.spinner("Generating match event CSV..."):
                export_df, missing_cols = _build_match_export_cached(
                    match_id=int(match_id),
                    data_mode=str(data_mode),
                    include_derived=bool(include_derived),
                    essential_only=bool(essential_only),
                    apply_filters=bool(apply_filters),
                    team_id=team_id,
                    player_id=player_id,
                )
                csv_bytes = events_df_to_csv_bytes(export_df)
                safe_label = (
                    match_label.replace(" ", "_")
                    .replace("/", "-")
                    .replace("\\", "-")
                    .replace(":", "-")
                    .replace("(", "")
                    .replace(")", "")
                )
                st.session_state["match_export_state"] = {
                    "params_sig": params_sig,
                    "export_df": export_df,
                    "missing_cols": missing_cols,
                    "csv_bytes": csv_bytes,
                    "file_name": f"match_events_{safe_label}.csv",
                }

        payload = st.session_state.get("match_export_state")
        if not payload or payload.get("params_sig") != params_sig:
            st.info("Click Generate CSV to create an export for the current options.")
            return

        export_df = payload.get("export_df", pd.DataFrame())
        missing_cols = payload.get("missing_cols", [])
        st.success(f"Ready: {len(export_df):,} rows x {len(export_df.columns):,} columns.")
        if {"attack_channel", "channel_source", "channel_reason"}.issubset(export_df.columns):
            st.caption("Attack channel summary")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.dataframe(
                    export_df["attack_channel"].astype("string").value_counts(dropna=False).rename_axis("attack_channel").reset_index(name="count"),
                    use_container_width=True,
                    hide_index=True,
                )
            with c2:
                st.dataframe(
                    export_df["channel_source"].astype("string").value_counts(dropna=False).rename_axis("channel_source").reset_index(name="count"),
                    use_container_width=True,
                    hide_index=True,
                )
            with c3:
                st.dataframe(
                    export_df["channel_reason"].astype("string").value_counts(dropna=False).rename_axis("channel_reason").reset_index(name="count"),
                    use_container_width=True,
                    hide_index=True,
                )
        if essential_only:
            missing_essentials = [col for col in ESSENTIAL_COLUMNS if col not in export_df.columns]
            if missing_essentials:
                st.warning("Some essential columns are unavailable in this dataset: " + ", ".join(missing_essentials[:12]))
        if missing_cols:
            st.warning("Some optional validation columns are unavailable: " + ", ".join(missing_cols[:15]))

        st.download_button(
            "Download CSV",
            data=payload.get("csv_bytes", b""),
            file_name=str(payload.get("file_name", f"match_events_{match_id}.csv")),
            mime="text/csv",
            key="export_download_csv",
        )
        st.dataframe(export_df.head(50), use_container_width=True, hide_index=True)
        st.caption("Preview: first 50 rows")


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


def _team_passes_subset(pass_df: pd.DataFrame, team_id: int | None) -> pd.DataFrame:
    if team_id is None or "team_id" not in pass_df.columns:
        return pass_df.iloc[0:0].copy() if team_id is not None else pass_df.copy()
    return pass_df[pd.to_numeric(pass_df["team_id"], errors="coerce") == int(team_id)].copy()


def _render_attack_distribution_chart(summary: dict[str, int], title: str) -> None:
    left = int(summary.get("Left", 0))
    centre = int(summary.get("Centre", 0))
    right = int(summary.get("Right", 0))
    total = max(1, left + centre + right)
    fig = go.Figure()
    fig.add_bar(y=[""], x=[left], name="Left", orientation="h", marker_color="#64748B", text=[f"{left/total:.0%}"])
    fig.add_bar(y=[""], x=[centre], name="Centre", orientation="h", marker_color="#38BDF8", text=[f"{centre/total:.0%}"])
    fig.add_bar(y=[""], x=[right], name="Right", orientation="h", marker_color="#94A3B8", text=[f"{right/total:.0%}"])
    fig.update_layout(
        title=title,
        barmode="stack",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#111a2b",
        font=dict(color="#e7edf7"),
        margin=dict(l=10, r=10, t=45, b=70),
        height=220,
        legend=dict(orientation="h", yanchor="top", y=-0.35, xanchor="left", x=0),
    )
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Left {left} | Centre {centre} | Right {right} ({left + centre + right} total)")


def _build_passers_table(pass_df: pd.DataFrame, selected_player_id: int | None, top_n: int) -> pd.DataFrame:
    top = top_progressive_passers(pass_df, top_n=top_n)
    if top.empty:
        return top
    table = top.copy()
    table["progressive_completion_pct"] = table["progressive_completion_pct"].map(lambda v: f"{float(v):.1f}%")
    table["avg_progressive_gain"] = table["avg_progressive_gain"].map(lambda v: f"{float(v):.1f}m")
    if selected_player_id is not None and "player_id" in table.columns:
        is_selected = pd.to_numeric(table["player_id"], errors="coerce") == int(selected_player_id)
        table["selected"] = is_selected.map(lambda v: "*" if bool(v) else "")
    show_cols = [c for c in ("selected", "player_name", "successful_progressive", "progressive_completion_pct", "avg_progressive_gain") if c in table.columns]
    return table[show_cols]


def _render_offensive_panel(
    events: pd.DataFrame,
    shots_df: pd.DataFrame,
    pass_df: pd.DataFrame,
    home_team_id: int | None,
    away_team_id: int | None,
    home_team_name: str,
    away_team_name: str,
    selected_team_id: int | None,
    selected_player_id: int | None,
) -> None:
    st.markdown('<div class="section-title">Offensive - Chance Creation & Progression</div>', unsafe_allow_html=True)
    st.caption("Coach-friendly view of attacking behaviour, attack channels, and progressive pass leaders.")

    apply_filters = st.checkbox(
        "Apply current filters to offensive metrics",
        value=False,
        help="OFF compares both teams for the full match. ON applies current team/player filters.",
        key="offensive_apply_filters",
    )
    prog_threshold = st.slider(
        "Progressive pass threshold (meters gained toward goal)",
        min_value=8,
        max_value=15,
        value=int(PROGRESSIVE_THRESHOLD_DEFAULT),
        step=1,
        key="offensive_progressive_threshold",
        help="A pass is progressive when it moves the ball this many meters closer to goal.",
    )

    passes_scope = pass_df.copy()
    events_scope = events.copy()
    shots_scope = shots_df.copy()
    if apply_filters:
        passes_scope = _apply_team_player_filters(passes_scope, team_id=selected_team_id, player_id=selected_player_id)
        events_scope = _apply_team_player_filters(events_scope, team_id=selected_team_id, player_id=selected_player_id)
        shots_scope = _apply_team_player_filters(shots_scope, team_id=selected_team_id, player_id=selected_player_id)

    if passes_scope.empty:
        st.info("No pass events available for the current context.")
        return
    if not {"location_x", "pass_end_location_x", "pass_end_location_y"}.issubset(passes_scope.columns):
        st.info("Pass locations not available in this dataset.")
        return

    passes_scope, completion_available = with_pass_features(passes_scope, threshold=float(prog_threshold))
    if not completion_available:
        st.warning("Completion unavailable: outcome columns are missing, so passes are treated as completed.")

    offensive = events_scope[events_scope["bucket"].astype("string") == "OFFENSIVE"] if "bucket" in events_scope.columns else events_scope.copy()
    dribbles = offensive[_event_type_mask(offensive, "Dribble")]
    shots = shots_scope.copy()
    total_passes = int(len(passes_scope))
    completed_passes = int(passes_scope["is_completed"].sum())
    progressive_success = int(passes_scope["is_successful_progressive"].sum())
    final_third = int(_final_third_pass_mask(passes_scope).sum())
    dribble_success = int(_duel_won_mask(dribbles).sum()) if not dribbles.empty else 0
    xg_total = float(pd.to_numeric(shots["xg"], errors="coerce").fillna(0).sum()) if "xg" in shots.columns else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total passes", total_passes)
    k2.metric("Completed passes", completed_passes)
    k3.metric("Successful progressive passes", progressive_success)
    k4.metric("Passes to final third", final_third)

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Dribbles attempted", int(len(dribbles)))
    k6.metric("Dribbles success", dribble_success)
    k7.metric("Shots", int(len(shots)))
    k8.metric("xG", f"{xg_total:.2f}")

    st.markdown('<div class="section-title">Attack distribution</div>', unsafe_allow_html=True)
    st.caption("Distribution of pass end locations by channel: Left / Centre / Right.")
    home_passes = _team_passes_subset(passes_scope, home_team_id)
    away_passes = _team_passes_subset(passes_scope, away_team_id)
    if (home_team_id is None or away_team_id is None) and "team_id" in passes_scope.columns:
        team_ids = [int(t) for t in pd.to_numeric(passes_scope["team_id"], errors="coerce").dropna().drop_duplicates().tolist()]
        if len(team_ids) >= 2:
            home_team_id = team_ids[0] if home_team_id is None else home_team_id
            away_team_id = team_ids[1] if away_team_id is None else away_team_id
            home_passes = _team_passes_subset(passes_scope, home_team_id)
            away_passes = _team_passes_subset(passes_scope, away_team_id)
        if "team_name" in passes_scope.columns:
            if home_team_name in {"", "Home"} and home_team_id is not None:
                home_name_series = passes_scope.loc[pd.to_numeric(passes_scope["team_id"], errors="coerce") == int(home_team_id), "team_name"]
                if not home_name_series.empty:
                    home_team_name = str(home_name_series.iloc[0] or home_team_name)
            if away_team_name in {"", "Away"} and away_team_id is not None:
                away_name_series = passes_scope.loc[pd.to_numeric(passes_scope["team_id"], errors="coerce") == int(away_team_id), "team_name"]
                if not away_name_series.empty:
                    away_team_name = str(away_name_series.iloc[0] or away_team_name)
    home_summary = summarize_channels(home_passes)
    away_summary = summarize_channels(away_passes)

    dist_home, dist_away = st.columns(2)
    with dist_home:
        _render_attack_distribution_chart(
            {"Left": home_summary.left, "Centre": home_summary.centre, "Right": home_summary.right},
            f"{home_team_name} attack channels",
        )
    with dist_away:
        _render_attack_distribution_chart(
            {"Left": away_summary.left, "Centre": away_summary.centre, "Right": away_summary.right},
            f"{away_team_name} attack channels",
        )

    st.markdown('<div class="section-title">Top progressive passers (successful)</div>', unsafe_allow_html=True)
    passers_home, passers_away = st.columns(2)
    with passers_home:
        st.markdown(f"**{home_team_name}**")
        top3_home = _build_passers_table(home_passes, selected_player_id=selected_player_id, top_n=3)
        if top3_home.empty:
            st.info("No progressive passers found.")
        else:
            st.dataframe(top3_home, use_container_width=True, hide_index=True)
            with st.expander("Show top 10", expanded=False):
                st.dataframe(_build_passers_table(home_passes, selected_player_id=selected_player_id, top_n=10), use_container_width=True, hide_index=True)
    with passers_away:
        st.markdown(f"**{away_team_name}**")
        top3_away = _build_passers_table(away_passes, selected_player_id=selected_player_id, top_n=3)
        if top3_away.empty:
            st.info("No progressive passers found.")
        else:
            st.dataframe(top3_away, use_container_width=True, hide_index=True)
            with st.expander("Show top 10", expanded=False):
                st.dataframe(_build_passers_table(away_passes, selected_player_id=selected_player_id, top_n=10), use_container_width=True, hide_index=True)


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
home_team_id = official_stats.get("home_team_id")
away_team_id = official_stats.get("away_team_id")
home_team_name = str(official_stats.get("home_team_name") or "Home")
away_team_name = str(official_stats.get("away_team_name") or "Away")

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
    with st.spinner("Loading pass actions..."):
        offensive_passes = load_match_passes(match_id=int(match_id), data_mode=active_data_mode, columns=MATCH_PASS_COLUMNS)
    offensive_shots = get_shots_view(match_shots, dim_team=dim_team, dim_player=dim_player)
    _render_offensive_panel(
        events=match_events,
        shots_df=offensive_shots,
        pass_df=offensive_passes,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_team_name=home_team_name,
        away_team_name=away_team_name,
        selected_team_id=selection["team_id"],
        selected_player_id=selection["player_id"],
    )
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
