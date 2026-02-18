from __future__ import annotations

import pandas as pd
import streamlit as st


def _coalesce(df: pd.DataFrame, candidates: list[str], default: object = pd.NA) -> pd.Series:
    existing = [column for column in candidates if column in df.columns]
    if not existing:
        return pd.Series(default, index=df.index, dtype="object")
    series = df[existing[0]]
    for column in existing[1:]:
        series = series.combine_first(df[column])
    return series


def _safe_count(df: pd.DataFrame, mask: pd.Series | None) -> int | str:
    if df.empty or mask is None:
        return "n/a"
    try:
        return int(mask.fillna(False).sum())
    except Exception:
        return "n/a"


def _value_or_na(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and pd.isna(value):
        return "n/a"
    text = str(value).strip()
    return text if text else "n/a"


def render_match_summary(
    dim_match: pd.DataFrame,
    fact_events_match: pd.DataFrame,
    fact_shots_match: pd.DataFrame,
    fact_shots_context: pd.DataFrame,
    selection: dict[str, object],
) -> None:
    match_id = selection.get("match_id")
    if match_id is None:
        return

    match_rows = dim_match[pd.to_numeric(dim_match["match_id"], errors="coerce") == int(match_id)] if "match_id" in dim_match.columns else pd.DataFrame()
    if match_rows.empty:
        st.markdown('<div class="section-title">Match Summary</div>', unsafe_allow_html=True)
        st.info("Match metadata not available.")
        return

    row = match_rows.iloc[0]
    home_team = _value_or_na(row.get("home_team_name"))
    away_team = _value_or_na(row.get("away_team_name"))
    home_score = _value_or_na(row.get("home_score"))
    away_score = _value_or_na(row.get("away_score"))
    match_date = _value_or_na(row.get("match_date"))
    kick_off = _value_or_na(row.get("kick_off"))
    competition = _value_or_na(selection.get("competition_name") or row.get("competition_name"))
    season = _value_or_na(selection.get("season_name") or row.get("season_name"))

    st.markdown(
        (
            '<div class="match-header-card">'
            f'<div class="match-header-title">{home_team} {home_score} - {away_score} {away_team}</div>'
            f'<div class="match-header-subtitle">{match_date} • {kick_off} • {competition} • {season}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    team_id = selection.get("team_id")
    player_id = selection.get("player_id")
    player_name = selection.get("player_name")
    team_name = selection.get("team_name")

    events_stats = fact_events_match.copy()
    shots_stats = fact_shots_match.copy()
    context_note = "Match totals"

    if player_id is None and team_id is not None:
        if "team_id" in events_stats.columns:
            events_stats = events_stats[pd.to_numeric(events_stats["team_id"], errors="coerce") == int(team_id)]
        if "team_id" in shots_stats.columns:
            shots_stats = shots_stats[pd.to_numeric(shots_stats["team_id"], errors="coerce") == int(team_id)]
        context_note = f"Team totals: {team_name or team_id}"
    elif player_id is not None:
        context_note = "Match totals (player selected)"

    type_name_series = _coalesce(events_stats, ["type_name"], default="")
    type_name_norm = type_name_series.astype(str).str.lower()

    passes = _safe_count(events_stats, type_name_norm.eq("pass") if "type_name" in events_stats.columns else None)
    offsides = _safe_count(
        events_stats,
        (
            type_name_norm.eq("offside")
            | _coalesce(events_stats, ["pass_outcome_name"], default="").astype(str).str.lower().eq("offside")
        ) if ("type_name" in events_stats.columns or "pass_outcome_name" in events_stats.columns) else None,
    )

    card_series = _coalesce(events_stats, ["foul_committed_card_name", "card_name"], default="")
    card_norm = card_series.astype(str).str.lower()
    yellow_cards = _safe_count(events_stats, card_norm.str.contains("yellow", na=False) if len(card_norm) else None)
    red_cards = _safe_count(events_stats, card_norm.str.contains("red", na=False) if len(card_norm) else None)

    shots_count = len(shots_stats)
    xg_series = pd.to_numeric(_coalesce(shots_stats, ["xg", "shot_statsbomb_xg"], default=0), errors="coerce")
    total_xg = float(xg_series.fillna(0).sum()) if len(xg_series) else 0.0
    shot_outcome_norm = _coalesce(shots_stats, ["shot_outcome", "shot_outcome_name"], default="").astype(str).str.lower()
    goals = int(shot_outcome_norm.eq("goal").sum()) if len(shot_outcome_norm) else 0

    st.markdown(
        (
            '<div class="match-summary-note">'
            f"{context_note}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(7)
    metric_cols[0].metric("Passes", passes)
    metric_cols[1].metric("Shots", shots_count)
    metric_cols[2].metric("xG", f"{total_xg:.2f}")
    metric_cols[3].metric("Goals", goals)
    metric_cols[4].metric("Offsides", offsides)
    metric_cols[5].metric("Yellow", yellow_cards)
    metric_cols[6].metric("Red", red_cards)

    if player_id is not None:
        player_shots = fact_shots_context.copy()
        player_xg_series = pd.to_numeric(_coalesce(player_shots, ["xg", "shot_statsbomb_xg"], default=0), errors="coerce")
        player_goals = int(
            _coalesce(player_shots, ["shot_outcome", "shot_outcome_name"], default="")
            .astype(str)
            .str.lower()
            .eq("goal")
            .sum()
        )
        player_xg = float(player_xg_series.fillna(0).sum()) if len(player_xg_series) else 0.0
        st.markdown(
            (
                '<div class="match-summary-player-line">'
                f"Selected player: {player_name or player_id} • Shots: {len(player_shots)} • xG: {player_xg:.2f} • Goals: {player_goals}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
