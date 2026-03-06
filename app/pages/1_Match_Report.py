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
    from app.components.analysis_registry import OFFENSIVE_COMPARISON_METRICS, classify_analysis_groups
    from app.components.comparison_cards import render_comparison_panel
    from app.components.data import (
        get_active_data_mode,
        get_lineup_events,
        get_lineup_players,
        get_shots,
        load_dimensions,
        load_match_events,
    )
    from app.components.event_classification import derive_counterpress_regains, derive_event_labels
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
    from app.components.replay_model import build_replay_segments
    from app.components.set_pieces import render_set_piece_tactical_view
    from app.components.dribbles import compute_offensive_team_stats, summarize_dribbles, top_dribble_players
    from app.components.ui import setup_page
    from app.components.viz import draw_pitch_figure, draw_split_lineup_pitch
    from app.visualizations.pitch_replay import build_replay_figure
except (ModuleNotFoundError, KeyError):
    from components.analysis_navigation import render_analysis_nav
    from components.analysis_registry import OFFENSIVE_COMPARISON_METRICS, classify_analysis_groups
    from components.comparison_cards import render_comparison_panel
    from components.data import (
        get_active_data_mode,
        get_lineup_events,
        get_lineup_players,
        get_shots,
        load_dimensions,
        load_match_events,
    )
    from components.event_classification import derive_counterpress_regains, derive_event_labels
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
    from components.replay_model import build_replay_segments
    from components.set_pieces import render_set_piece_tactical_view
    from components.dribbles import compute_offensive_team_stats, summarize_dribbles, top_dribble_players
    from components.ui import setup_page
    from components.viz import draw_pitch_figure, draw_split_lineup_pitch
    from visualizations.pitch_replay import build_replay_figure

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
    "pass_recipient_id",
    "pass_recipient_name",
    "dribble_outcome_id",
    "dribble_outcome_name",
    "dribble_no_touch",
    "dribble_nutmeg",
    "dribble_overrun",
    "duel_outcome_name",
    "duel_outcome",
    "shot_outcome_name",
    "shot_outcome_id",
    "shot_outcome",
    "shot_type_name",
    "shot_statsbomb_xg",
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
    "dribble_outcome_name",
    "dribble_outcome_id",
    "dribble_no_touch",
    "dribble_nutmeg",
    "dribble_overrun",
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
        if not export_df.empty:
            if "dribble_is_attempt" in export_df.columns:
                dribble_mask = export_df["dribble_is_attempt"].fillna(False).astype(bool)
            else:
                event_type = export_df.get("type_name", pd.Series("", index=export_df.index)).astype("string").str.strip().str.lower()
                dribble_mask = event_type.eq("dribble")
            st.caption(f"Export summary: total rows={len(export_df):,}, dribble rows={int(dribble_mask.sum()):,}")
            if dribble_mask.any():
                outcome_col = (
                    export_df.get("dribble_outcome_name", pd.Series(pd.NA, index=export_df.index))
                    .astype("string")
                    .str.strip()
                )
                outcome_dist = (
                    outcome_col.loc[dribble_mask]
                    .fillna("Unknown")
                    .replace("", "Unknown")
                    .value_counts(dropna=False)
                    .rename_axis("dribble_outcome")
                    .reset_index(name="count")
                )
                st.dataframe(outcome_dist, use_container_width=True, hide_index=True)

                player_col = export_df.get("player_name", pd.Series("Unknown Player", index=export_df.index)).astype("string").fillna("Unknown Player")
                top_players = (
                    player_col.loc[dribble_mask]
                    .value_counts(dropna=False)
                    .rename_axis("player_name")
                    .reset_index(name="attempts")
                    .head(10)
                )
                st.dataframe(top_players, use_container_width=True, hide_index=True)
            if {"analysis_group", "analysis_subgroup"}.issubset(export_df.columns):
                st.caption("Analysis grouping summary")
                group_counts = (
                    export_df["analysis_group"]
                    .astype("string")
                    .fillna("other")
                    .value_counts(dropna=False)
                    .rename_axis("analysis_group")
                    .reset_index(name="count")
                )
                subgroup_counts = (
                    export_df["analysis_subgroup"]
                    .astype("string")
                    .fillna("other")
                    .value_counts(dropna=False)
                    .rename_axis("analysis_subgroup")
                    .reset_index(name="count")
                )
                total_groups = int(group_counts["count"].sum()) if not group_counts.empty else 0
                st.caption(f"Sanity check: total rows={len(export_df):,}, sum(groups)={total_groups:,}")
                sg1, sg2 = st.columns(2)
                with sg1:
                    st.dataframe(group_counts, use_container_width=True, hide_index=True)
                with sg2:
                    st.dataframe(subgroup_counts.head(20), use_container_width=True, hide_index=True)

                event_type = export_df.get("type_name", pd.Series("", index=export_df.index)).astype("string").str.strip().str.lower()
                counts_focus = pd.DataFrame(
                    {
                        "metric": ["shots", "dribbles", "duels", "recoveries", "carries", "passes"],
                        "count": [
                            int(event_type.eq("shot").sum()),
                            int(event_type.eq("dribble").sum()),
                            int(event_type.isin({"duel", "50/50"}).sum()),
                            int(event_type.isin({"ball recovery", "interception"}).sum()),
                            int(event_type.eq("carry").sum()),
                            int(event_type.eq("pass").sum()),
                        ],
                    }
                )
                st.dataframe(counts_focus, use_container_width=True, hide_index=True)
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


def _render_offensive_comparison_card(
    events_scope: pd.DataFrame,
    shots_scope: pd.DataFrame,
    match_id: int,
    home_team_id: int | None,
    away_team_id: int | None,
    home_team_name: str,
    away_team_name: str,
) -> None:
    stats = compute_offensive_team_stats(
        events_df=events_scope,
        shots_df=shots_scope,
        match_id=match_id,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
    )
    home = stats["home"]
    away = stats["away"]
    rows = [(label, home.get(metric_id), away.get(metric_id), is_pct) for metric_id, label, is_pct in OFFENSIVE_COMPARISON_METRICS]
    render_comparison_panel(rows=rows, home_name=home_team_name, away_name=away_team_name)


def _render_top_dribblers(
    events_scope: pd.DataFrame,
    home_team_id: int | None,
    away_team_id: int | None,
    home_team_name: str,
    away_team_name: str,
) -> None:
    st.markdown('<div class="section-title">Top Dribble Attempts</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown(f"**{home_team_name}**")
        top_home = top_dribble_players(events_scope, team_id=home_team_id, top_n=3)
        if top_home.empty:
            st.info("No dribble attempts found.")
        else:
            tbl_home = top_home.copy()
            tbl_home["success_pct"] = tbl_home["success_pct"].map(lambda v: f"{float(v):.1f}%")
            st.dataframe(tbl_home, use_container_width=True, hide_index=True)
    with right:
        st.markdown(f"**{away_team_name}**")
        top_away = top_dribble_players(events_scope, team_id=away_team_id, top_n=3)
        if top_away.empty:
            st.info("No dribble attempts found.")
        else:
            tbl_away = top_away.copy()
            tbl_away["success_pct"] = tbl_away["success_pct"].map(lambda v: f"{float(v):.1f}%")
            st.dataframe(tbl_away, use_container_width=True, hide_index=True)


def _render_offensive_panel(
    events: pd.DataFrame,
    shots_df: pd.DataFrame,
    home_team_id: int | None,
    away_team_id: int | None,
    home_team_name: str,
    away_team_name: str,
    selected_team_id: int | None,
    selected_player_id: int | None,
    match_id: int,
) -> None:
    st.markdown('<div class="section-title">Offensive - Chance Creation & Progression</div>', unsafe_allow_html=True)
    st.caption("Coach-friendly view of shot creation, attacking duels, and chance quality.")

    apply_filters = st.checkbox(
        "Apply current filters to offensive metrics",
        value=False,
        help="OFF compares both teams for the full match. ON applies current team/player filters.",
        key="offensive_apply_filters",
    )
    events_scope = events.copy()
    shots_scope = shots_df.copy()
    if apply_filters:
        events_scope = _apply_team_player_filters(events_scope, team_id=selected_team_id, player_id=selected_player_id)
        shots_scope = _apply_team_player_filters(shots_scope, team_id=selected_team_id, player_id=selected_player_id)

    classified_events = classify_analysis_groups(events_scope)
    offensive = classified_events[classified_events["analysis_group"].astype("string") == "offensive"].copy()
    shots = shots_scope.copy()

    st.markdown('<div class="section-title">Offensive Comparison Stats</div>', unsafe_allow_html=True)
    _render_offensive_comparison_card(
        events_scope=offensive,
        shots_scope=shots,
        match_id=match_id,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_team_name=home_team_name,
        away_team_name=away_team_name,
    )

    offensive_duels = offensive[offensive["analysis_subgroup"].astype("string") == "duels_offensive"].copy() if not offensive.empty else offensive
    fouls_won = int(_event_type_mask(offensive, "Foul Won").sum()) if not offensive.empty else 0
    k1, k2, k3 = st.columns(3)
    k1.metric("Offensive duels/actions", int(len(offensive_duels)))
    k2.metric("Fouls won (attacking phase)", fouls_won)
    k3.metric("Shot events", int(_event_type_mask(shots, "Shot").sum()) if "type_name" in shots.columns else int(len(shots)))

    dribble_debug = summarize_dribbles(events_scope, team_id=selected_team_id, player_id=selected_player_id)
    with st.expander("Dribble Debug (selected context)", expanded=False):
        st.write(
            {
                "total_dribble_events": dribble_debug["total_dribble_events"],
                "counts_by_outcome": dribble_debug["outcomes"],
                "missing_dribble_object": dribble_debug["missing_dribble_object"],
                "missing_outcome": dribble_debug["missing_outcome"],
                "duplicates_removed_by_event_id": dribble_debug["duplicates_removed"],
                "filtered_out_by_current_filters": dribble_debug["filtered_out"],
            }
        )

    if {"x", "y"}.issubset(shots.columns) and not shots.empty:
        st.markdown('<div class="section-title">Shot Map</div>', unsafe_allow_html=True)
        shot_map_fig = draw_pitch_figure(shots, title="Offensive Shot Map", subtitle=f"{home_team_name} vs {away_team_name}")
        st.plotly_chart(shot_map_fig, use_container_width=True)

    if "shot_outcome" in shots.columns and not shots.empty:
        st.markdown('<div class="section-title">Shot Outcomes</div>', unsafe_allow_html=True)
        outcome_dist = (
            shots["shot_outcome"]
            .astype("string")
            .str.strip()
            .fillna("Unknown")
            .replace("", "Unknown")
            .value_counts(dropna=False)
            .rename_axis("shot_outcome")
            .reset_index(name="count")
        )
        st.dataframe(outcome_dist, use_container_width=True, hide_index=True)

    if {"minute", "xg"}.issubset(shots.columns) and not shots.empty:
        st.markdown('<div class="section-title">Cumulative xG Timeline</div>', unsafe_allow_html=True)
        timeline = shots.groupby("minute", as_index=False)["xg"].sum().sort_values("minute")
        timeline["cum_xg"] = timeline["xg"].cumsum()
        fig_timeline = px.line(timeline, x="minute", y="cum_xg", markers=True)
        fig_timeline.update_layout(
            paper_bgcolor="#0b1220",
            plot_bgcolor="#111a2b",
            font=dict(color="#e7edf7"),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
            margin=dict(l=10, r=10, t=30, b=80),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    _render_top_dribblers(
        events_scope=offensive,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_team_name=home_team_name,
        away_team_name=away_team_name,
    )


def _render_defensive_panel(events: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Defensive - Regains & Disruption</div>', unsafe_allow_html=True)
    st.caption("Tracks how often your team disrupted attacks and recovered possession.")
    if events.empty:
        st.info("No defensive events in current context.")
        return

    classified = classify_analysis_groups(events)
    defensive = classified[classified["analysis_group"].astype("string") == "defensive"].copy()
    recoveries_df = defensive[defensive["analysis_subgroup"].astype("string") == "recoveries"].copy()
    defensive_duels_df = defensive[defensive["analysis_subgroup"].astype("string") == "duels_defensive"].copy()
    pressures = int(_event_type_mask(defensive, "Pressure").sum())
    counterpress = int(defensive["is_counterpress"].sum()) if "is_counterpress" in defensive.columns else 0
    interceptions = int(_event_type_mask(defensive, "Interception").sum())
    recoveries = int(len(recoveries_df))
    duels_won = int(_duel_won_mask(defensive_duels_df).sum()) if not defensive_duels_df.empty else 0
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


def _clock_seconds_to_label(value: int) -> str:
    minutes = max(0, int(value)) // 60
    seconds = max(0, int(value)) % 60
    return f"{minutes:02d}:{seconds:02d}"


def _render_live_pitch_replay(
    match_id: int,
    data_mode: str,
    home_team_name: str,
    away_team_name: str,
) -> None:
    with st.expander("Live Pitch Replay (Beta)", expanded=False):
        st.caption("Event-tied animation on a 120x80 pitch with optional 360 freeze-frame overlays.")

        try:
            segments, coverage = build_replay_segments(match_id=int(match_id), data_mode=str(data_mode))
        except Exception as exc:
            st.warning(f"Replay data could not be loaded for this match: {exc}")
            return

        if not segments:
            st.info("No replay segments available for this match.")
            return

        all_types = sorted({str(seg.get("event_type") or "Unknown") for seg in segments})
        default_candidates = ["Pass", "Carry", "Shot", "Pressure", "Ball Receipt*", "Ball Receipt"]
        default_types = [item for item in default_candidates if item in all_types]
        if not default_types:
            default_types = all_types[: min(5, len(all_types))]

        control1, control2, control3 = st.columns(3)
        with control1:
            team_filter = st.selectbox("Team", options=["Both", "Home", "Away"], index=0)
        with control2:
            period_filter = st.selectbox("Period", options=["Both", "1H", "2H"], index=0)
        with control3:
            time_window = st.selectbox(
                "Time window",
                options=["Full match", "0-15", "15-30", "30-45", "45-60", "60-75", "75-90", "90+"],
                index=0,
            )

        event_types = st.multiselect(
            "Event types",
            options=all_types,
            default=default_types,
            help="Replay includes only selected event types.",
        )
        if not event_types:
            st.warning("Select at least one event type to render the replay.")
            return

        timing_col, perf_col = st.columns(2)
        with timing_col:
            min_clock = int(max(0.0, min(float(seg["t0"]) for seg in segments)))
            max_clock = int(max(float(seg["t1"]) for seg in segments))
            clock_range = st.slider(
                "Match clock range (seconds)",
                min_value=min_clock,
                max_value=max_clock,
                value=(min_clock, max_clock),
                step=1,
                format="%d",
            )
            st.caption(f"Selected clock: {_clock_seconds_to_label(clock_range[0])} - {_clock_seconds_to_label(clock_range[1])}")
        with perf_col:
            fps = st.slider("FPS", min_value=5, max_value=25, value=12, step=1)
            max_frames = st.slider("Max frames", min_value=200, max_value=2000, value=900, step=100)
            show_paths = st.toggle("Show event paths", value=True)
            has_visible_area = bool(coverage.get("has_visible_area"))
            show_visible_area = False
            if bool(coverage.get("has_360")):
                show_visible_area = st.toggle(
                    "Show 360 visible area",
                    value=False,
                    disabled=not has_visible_area,
                )

        def _window_ok(seg: dict[str, object]) -> bool:
            t0 = float(seg["t0"])
            minute_abs = t0 / 60.0
            if time_window == "Full match":
                return True
            if time_window == "0-15":
                return 0.0 <= minute_abs < 15.0
            if time_window == "15-30":
                return 15.0 <= minute_abs < 30.0
            if time_window == "30-45":
                return 30.0 <= minute_abs < 45.0
            if time_window == "45-60":
                return 45.0 <= minute_abs < 60.0
            if time_window == "60-75":
                return 60.0 <= minute_abs < 75.0
            if time_window == "75-90":
                return 75.0 <= minute_abs < 90.0
            return minute_abs >= 90.0

        filtered_segments = []
        for seg in segments:
            t0 = float(seg["t0"])
            if not (float(clock_range[0]) <= t0 <= float(clock_range[1])):
                continue
            if seg["event_type"] not in event_types:
                continue
            if team_filter == "Home" and str(seg["team"]) != str(home_team_name):
                continue
            if team_filter == "Away" and str(seg["team"]) != str(away_team_name):
                continue
            if period_filter == "1H" and int(seg.get("period") or 0) != 1:
                continue
            if period_filter == "2H" and int(seg.get("period") or 0) != 2:
                continue
            if not _window_ok(seg):
                continue
            filtered_segments.append(seg)

        if not filtered_segments:
            st.info("No replay segments match the current filters.")
            return

        fig = build_replay_figure(
            segments=filtered_segments,
            fps=int(fps),
            max_frames=int(max_frames),
            show_visible_area=bool(show_visible_area),
            show_paths=bool(show_paths),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Data coverage: "
            f"{coverage.get('coverage', 'Events-only')} | "
            f"Segments: {len(filtered_segments):,} | "
            "Legend: white=ball, green=actor, blue/orange=360 players."
        )
        if not bool(coverage.get("has_360")):
            st.info("360 snapshots were not found for this match/data mode. Running in Events-only replay.")


def _render_transitions_panel(
    events: pd.DataFrame,
    shots_df: pd.DataFrame,
    match_id: int,
    data_mode: str,
    home_team_name: str,
    away_team_name: str,
) -> None:
    st.markdown('<div class="section-title">Transitions - Turnovers & Counterpress</div>', unsafe_allow_html=True)
    st.caption("Shows what happens immediately after possession changes.")
    if events.empty:
        st.info("No transition events in current context.")
        return

    classified = classify_analysis_groups(events)
    transition = classified[classified["analysis_group"].astype("string") == "transitions"].copy()
    carries = transition[transition["analysis_subgroup"].astype("string") == "carries"].copy() if not transition.empty else transition
    turnovers = int(transition["is_turnover"].sum()) if "is_turnover" in transition.columns else 0
    counterpress = int(transition["is_counterpress"].sum()) if "is_counterpress" in transition.columns else 0
    counter_regains = int(transition["is_counterpress_regain"].sum()) if "is_counterpress_regain" in transition.columns else 0
    counters = int(transition["is_counter"].sum()) if "is_counter" in transition.columns else 0
    carry_count = int(_event_type_mask(carries, "Carry").sum()) if not carries.empty else 0
    progressive_carries = 0
    if not carries.empty and {"location_x", "carry_end_location_x"}.issubset(carries.columns):
        start_x = pd.to_numeric(carries["location_x"], errors="coerce")
        end_x = pd.to_numeric(carries["carry_end_location_x"], errors="coerce")
        progressive_carries = int((end_x - start_x).ge(10.0).fillna(False).sum())

    shots_counter = pd.DataFrame()
    if "play_pattern_name" in shots_df.columns:
        shots_counter = shots_df[
            shots_df["play_pattern_name"].astype("string").str.strip().str.lower().eq("from counter")
        ].copy()

    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("Turnovers", turnovers)
    k2.metric("Counterpress", counterpress)
    k3.metric("Regains in 6s", counter_regains)
    k4.metric("Counters", counters)
    k5.metric("Shots from counters", int(len(shots_counter)))
    k6.metric("Carries", carry_count)
    k7.metric("Progressive carries", progressive_carries)

    if {"minute", "type_name", "is_turnover", "is_counterpress_regain"}.issubset(transition.columns):
        cols = ["minute", "type_name", "is_turnover", "is_counterpress_regain"]
        st.dataframe(transition[cols].sort_values("minute").head(30), use_container_width=True, hide_index=True)

    _render_live_pitch_replay(
        match_id=match_id,
        data_mode=data_mode,
        home_team_name=home_team_name,
        away_team_name=away_team_name,
    )


def _render_set_piece_panel(events: pd.DataFrame, shots_df: pd.DataFrame) -> None:
    del shots_df  # Phase 1 set-piece tactical view is derived from events-only pipeline.
    render_set_piece_tactical_view(events)


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
elif analysis_view == "Offensive":
    offensive_shots = get_shots_view(match_shots, dim_team=dim_team, dim_player=dim_player)
    _render_offensive_panel(
        events=match_events,
        shots_df=offensive_shots,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_team_name=home_team_name,
        away_team_name=away_team_name,
        selected_team_id=selection["team_id"],
        selected_player_id=selection["player_id"],
        match_id=int(match_id),
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
elif analysis_view == "Transitions":
    filtered_events = _apply_team_player_filters(match_events, team_id=selection["team_id"], player_id=selection["player_id"])
    filtered_shots = _apply_team_player_filters(match_shots, team_id=selection["team_id"], player_id=selection["player_id"])
    transition_shots = get_shots_view(filtered_shots, dim_team=dim_team, dim_player=dim_player)
    _render_transitions_panel(
        filtered_events,
        transition_shots,
        match_id=int(match_id),
        data_mode=active_data_mode,
        home_team_name=home_team_name,
        away_team_name=away_team_name,
    )
elif analysis_view == "Defensive":
    filtered_events = _apply_team_player_filters(match_events, team_id=selection["team_id"], player_id=selection["player_id"])
    _render_defensive_panel(filtered_events)
elif analysis_view == "Set Pieces":
    filtered_events = _apply_team_player_filters(match_events, team_id=selection["team_id"], player_id=selection["player_id"])
    filtered_shots = _apply_team_player_filters(match_shots, team_id=selection["team_id"], player_id=selection["player_id"])
    set_piece_shots = get_shots_view(filtered_shots, dim_team=dim_team, dim_player=dim_player)
    _render_set_piece_panel(filtered_events, set_piece_shots)
else:
    _render_more_section()
