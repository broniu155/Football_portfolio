from __future__ import annotations

import pandas as pd


def _first_existing(columns: pd.Index, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in columns:
            return column
    return None


def _coalesce(df: pd.DataFrame, candidates: list[str], default: object = pd.NA) -> pd.Series:
    existing = [column for column in candidates if column in df.columns]
    if not existing:
        return pd.Series(default, index=df.index, dtype="object")

    series = df[existing[0]]
    for column in existing[1:]:
        series = series.combine_first(df[column])
    return series


def _with_names_from_dims(
    view: pd.DataFrame, dim_team: pd.DataFrame | None, dim_player: pd.DataFrame | None
) -> pd.DataFrame:
    if "team_name" not in view.columns or view["team_name"].isna().any():
        if dim_team is not None and {"team_id", "team_name"}.issubset(dim_team.columns) and "team_id" in view.columns:
            team_lookup = dim_team[["team_id", "team_name"]].drop_duplicates(subset=["team_id"])
            view = view.merge(team_lookup, on="team_id", how="left", suffixes=("", "_dim"))
            if "team_name_dim" in view.columns:
                if "team_name" in view.columns:
                    view["team_name"] = view["team_name"].combine_first(view["team_name_dim"])
                else:
                    view["team_name"] = view["team_name_dim"]
                view = view.drop(columns=["team_name_dim"])

    if "player_name" not in view.columns or view["player_name"].isna().any():
        if (
            dim_player is not None
            and {"player_id", "player_name"}.issubset(dim_player.columns)
            and "player_id" in view.columns
        ):
            player_lookup = dim_player[["player_id", "player_name"]].drop_duplicates(subset=["player_id"])
            view = view.merge(player_lookup, on="player_id", how="left", suffixes=("", "_dim"))
            if "player_name_dim" in view.columns:
                if "player_name" in view.columns:
                    view["player_name"] = view["player_name"].combine_first(view["player_name_dim"])
                else:
                    view["player_name"] = view["player_name_dim"]
                view = view.drop(columns=["player_name_dim"])

    return view


def get_shots_view(
    fact_shots: pd.DataFrame, dim_team: pd.DataFrame | None = None, dim_player: pd.DataFrame | None = None
) -> pd.DataFrame:
    view = fact_shots.copy()

    view["x"] = pd.to_numeric(_coalesce(view, ["x", "location_x"]), errors="coerce")
    view["y"] = pd.to_numeric(_coalesce(view, ["y", "location_y"]), errors="coerce")
    view["xg"] = pd.to_numeric(_coalesce(view, ["xg", "shot_statsbomb_xg"]), errors="coerce")
    view["shot_outcome"] = _coalesce(view, ["shot_outcome", "shot_outcome_name", "shot_outcome_id"])
    view["shot_type"] = _coalesce(view, ["shot_type", "shot_type_name"])
    view["body_part"] = _coalesce(view, ["body_part", "body_part_name", "shot_body_part_name", "shot_body_part"])
    outcome_norm = view["shot_outcome"].astype("string").str.strip().str.lower()
    goal_mask = outcome_norm.eq("goal")
    if "shot_outcome_id" in view.columns:
        # StatsBomb canonical goal outcome id.
        goal_mask = goal_mask | (pd.to_numeric(view["shot_outcome_id"], errors="coerce") == 97)
    view["is_goal"] = goal_mask.fillna(False)

    view = _with_names_from_dims(view, dim_team=dim_team, dim_player=dim_player)

    if "team_name" not in view.columns:
        view["team_name"] = pd.NA
    if "player_name" not in view.columns:
        view["player_name"] = pd.NA

    return view


def get_events_view(
    fact_events: pd.DataFrame, dim_team: pd.DataFrame | None = None, dim_player: pd.DataFrame | None = None
) -> pd.DataFrame:
    view = fact_events.copy()

    x_col = _first_existing(view.columns, ["x", "location_x"])
    y_col = _first_existing(view.columns, ["y", "location_y"])
    xg_col = _first_existing(view.columns, ["xg", "shot_statsbomb_xg"])

    view["x"] = pd.to_numeric(view[x_col], errors="coerce") if x_col else pd.NA
    view["y"] = pd.to_numeric(view[y_col], errors="coerce") if y_col else pd.NA
    view["xg"] = pd.to_numeric(view[xg_col], errors="coerce") if xg_col else pd.NA
    view["shot_outcome"] = _coalesce(
        view,
        ["shot_outcome", "shot_outcome_id", "shot_outcome_name", "outcome_name", "outcome_id"],
    )

    view = _with_names_from_dims(view, dim_team=dim_team, dim_player=dim_player)

    if "team_name" not in view.columns:
        view["team_name"] = pd.NA
    if "player_name" not in view.columns:
        view["player_name"] = pd.NA

    return view
