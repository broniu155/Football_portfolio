from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.data import get_matches, get_players_for_match, get_teams_for_match


def _to_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _init_or_reset_selectbox(key: str, options: list[object], default: object | None = None) -> object | None:
    if not options:
        st.session_state[key] = None
        return None

    if default not in options:
        default = options[0]

    current = st.session_state.get(key, default)
    if current not in options:
        current = default
        st.session_state[key] = current
    return current


def sidebar_filters_cascading(dim_match: pd.DataFrame) -> dict[str, object]:
    st.sidebar.header("Filters")
    dm = dim_match.copy()

    for column in ("match_id", "competition_id", "season_id", "home_team_id", "away_team_id"):
        if column in dm.columns:
            dm[column] = pd.to_numeric(dm[column], errors="coerce")

    competition_id: int | None = None
    competition_name: str | None = None
    season_id: int | None = None
    season_name: str | None = None

    if {"competition_id", "competition_name"}.issubset(dm.columns):
        comp_df = (
            dm.dropna(subset=["competition_id"])
            .drop_duplicates(subset=["competition_id"])[["competition_id", "competition_name"]]
            .sort_values("competition_name")
        )
        comp_ids = [int(c) for c in comp_df["competition_id"].tolist()]
        if comp_ids:
            selected_comp = _init_or_reset_selectbox("flt_competition_id", comp_ids)
            selected_comp = st.sidebar.selectbox(
                "Competition",
                options=comp_ids,
                index=comp_ids.index(selected_comp) if selected_comp in comp_ids else 0,
                format_func=lambda c: comp_df.loc[comp_df["competition_id"] == c, "competition_name"].iloc[0],
                key="flt_competition_id",
            )
            competition_id = int(selected_comp)
            competition_name = str(comp_df.loc[comp_df["competition_id"] == selected_comp, "competition_name"].iloc[0])
            dm = dm[dm["competition_id"] == competition_id]

    if {"season_id", "season_name"}.issubset(dm.columns):
        season_df = (
            dm.dropna(subset=["season_id"])
            .drop_duplicates(subset=["season_id"])[["season_id", "season_name"]]
            .sort_values("season_name")
        )
        season_ids = [int(s) for s in season_df["season_id"].tolist()]
        if season_ids:
            selected_season = _init_or_reset_selectbox("flt_season_id", season_ids)
            selected_season = st.sidebar.selectbox(
                "Season",
                options=season_ids,
                index=season_ids.index(selected_season) if selected_season in season_ids else 0,
                format_func=lambda s: season_df.loc[season_df["season_id"] == s, "season_name"].iloc[0],
                key="flt_season_id",
            )
            season_id = int(selected_season)
            season_name = str(season_df.loc[season_df["season_id"] == selected_season, "season_name"].iloc[0])

    matches = get_matches(competition_id=competition_id, season_id=season_id)
    if matches.empty:
        st.sidebar.warning("No matches available for current filters.")
        return {
            "competition_id": competition_id,
            "competition_name": competition_name,
            "season_id": season_id,
            "season_name": season_name,
            "match_id": None,
            "match_label": None,
            "team_id": None,
            "team_name": None,
            "player_id": None,
            "player_name": None,
        }

    if "match_label" not in matches.columns:
        matches["match_label"] = matches["match_id"].astype(str)

    match_lookup = matches.drop_duplicates(subset=["match_id"]).set_index("match_id")
    match_ids = [int(mid) for mid in match_lookup.index.tolist()]
    selected_match = _init_or_reset_selectbox("flt_match_id", match_ids)
    selected_match = st.sidebar.selectbox(
        "Match",
        options=match_ids,
        index=match_ids.index(selected_match) if selected_match in match_ids else 0,
        format_func=lambda mid: str(match_lookup.loc[mid, "match_label"]),
        key="flt_match_id",
    )
    match_id = int(selected_match)
    match_label = str(match_lookup.loc[match_id, "match_label"])

    teams = get_teams_for_match(match_id)
    team_options: list[int | None] = [None]
    team_lookup: dict[int | None, str] = {None: "(All)"}
    if not teams.empty and "team_id" in teams.columns:
        for _, row in teams.iterrows():
            tid = _to_int(row.get("team_id"))
            if tid is None:
                continue
            if tid not in team_options:
                team_options.append(tid)
                team_lookup[tid] = str(row.get("team_name") or tid)

    selected_team = _init_or_reset_selectbox("flt_team_id", team_options, default=None)
    selected_team = st.sidebar.selectbox(
        "Team",
        options=team_options,
        index=team_options.index(selected_team) if selected_team in team_options else 0,
        format_func=lambda tid: team_lookup.get(tid, "(All)"),
        key="flt_team_id",
    )
    team_id = _to_int(selected_team)
    team_name = team_lookup.get(team_id)

    players = get_players_for_match(match_id=match_id, team_id=team_id)
    player_options: list[int | None] = [None]
    player_lookup: dict[int | None, str] = {None: "(All)"}
    if not players.empty and "player_id" in players.columns:
        for _, row in players.iterrows():
            pid = _to_int(row.get("player_id"))
            if pid is None:
                continue
            if pid not in player_options:
                player_options.append(pid)
                player_lookup[pid] = str(row.get("player_name") or pid)

    selected_player = _init_or_reset_selectbox("flt_player_id", player_options, default=None)
    selected_player = st.sidebar.selectbox(
        "Player",
        options=player_options,
        index=player_options.index(selected_player) if selected_player in player_options else 0,
        format_func=lambda pid: player_lookup.get(pid, "(All)"),
        key="flt_player_id",
    )
    player_id = _to_int(selected_player)
    player_name = player_lookup.get(player_id)

    return {
        "competition_id": competition_id,
        "competition_name": competition_name,
        "season_id": season_id,
        "season_name": season_name,
        "match_id": match_id,
        "match_label": match_label,
        "team_id": team_id,
        "team_name": team_name if team_id is not None else None,
        "player_id": player_id,
        "player_name": player_name if player_id is not None else None,
    }


def sidebar_filters(dim_match: pd.DataFrame, dim_team: pd.DataFrame | None = None, dim_player: pd.DataFrame | None = None):
    selected = sidebar_filters_cascading(dim_match=dim_match)
    return selected["match_id"], selected["team_name"], selected["player_name"]
