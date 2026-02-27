from __future__ import annotations

from typing import Any

import pandas as pd

DRIBBLE_TYPE_IDS = {14}
DRIBBLE_COMPLETE = {"complete"}
DRIBBLE_INCOMPLETE = {"incomplete"}
DRIBBLE_OBJECT_COLUMNS = (
    "dribble_no_touch",
    "dribble_nutmeg",
    "dribble_outcome_id",
    "dribble_outcome_name",
    "dribble_overrun",
)


def _coalesce(df: pd.DataFrame, candidates: list[str], default: object = pd.NA) -> pd.Series:
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return pd.Series(default, index=df.index, dtype="object")
    out = df[cols[0]]
    for col in cols[1:]:
        out = out.combine_first(df[col])
    return out


def _as_bool(series: pd.Series) -> pd.Series:
    if str(series.dtype).lower() in {"bool", "boolean"}:
        return series.fillna(False).astype(bool)
    as_text = series.astype("string").str.strip().str.lower()
    as_num = pd.to_numeric(series, errors="coerce")
    return as_text.isin({"true", "t", "1", "yes", "y"}) | as_num.eq(1)


def dribble_attempt_mask(events_df: pd.DataFrame) -> pd.Series:
    if events_df.empty:
        return pd.Series(dtype="bool")
    mask = pd.Series(False, index=events_df.index)
    if "type_name" in events_df.columns:
        mask = mask | events_df["type_name"].astype("string").str.strip().str.lower().eq("dribble")
    if "type_id" in events_df.columns:
        mask = mask | pd.to_numeric(events_df["type_id"], errors="coerce").isin(DRIBBLE_TYPE_IDS)
    return mask.fillna(False)


def prepare_dribble_events(events_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if events_df.empty:
        return events_df.copy(), 0

    work = events_df.copy()
    work = work[dribble_attempt_mask(work)].copy()
    if work.empty:
        return work, 0

    duplicates = 0
    if "event_id" in work.columns:
        before = len(work)
        subset = ["event_id"] if "match_id" not in work.columns else ["match_id", "event_id"]
        work = work.drop_duplicates(subset=subset, keep="first").copy()
        duplicates = max(0, before - len(work))

    outcome = _coalesce(work, ["dribble_outcome_name", "dribble_outcome"], default=pd.NA).astype("string").str.strip()
    outcome_norm = outcome.str.lower()
    is_complete = outcome_norm.isin(DRIBBLE_COMPLETE).fillna(False)
    is_incomplete = outcome_norm.isin(DRIBBLE_INCOMPLETE).fillna(False)
    is_unknown = ~(is_complete | is_incomplete)

    work["dribble_outcome_name_norm"] = outcome_norm
    work["dribble_outcome_resolved"] = "Unknown"
    work.loc[is_complete, "dribble_outcome_resolved"] = "Complete"
    work.loc[is_incomplete, "dribble_outcome_resolved"] = "Incomplete"
    work["dribble_is_attempt"] = True
    work["dribble_is_complete"] = is_complete.astype(bool)
    work["dribble_is_incomplete"] = is_incomplete.astype(bool)
    work["dribble_is_unknown"] = is_unknown.astype(bool)

    return work, duplicates


def summarize_dribbles(
    events_df: pd.DataFrame,
    team_id: int | None = None,
    player_id: int | None = None,
) -> dict[str, Any]:
    base = events_df.copy()
    total_before_filters = int(dribble_attempt_mask(base).sum()) if not base.empty else 0
    if team_id is not None and "team_id" in base.columns:
        base = base[pd.to_numeric(base["team_id"], errors="coerce") == int(team_id)]
    if player_id is not None and "player_id" in base.columns:
        base = base[pd.to_numeric(base["player_id"], errors="coerce") == int(player_id)]

    dribbles, duplicates = prepare_dribble_events(base)
    if dribbles.empty:
        return {
            "total_dribble_events": 0,
            "total_before_filters": total_before_filters,
            "duplicates_removed": duplicates,
            "missing_dribble_object": 0,
            "missing_outcome": 0,
            "outcomes": {},
            "filtered_out": max(0, total_before_filters),
        }

    has_object_fields = pd.Series(False, index=dribbles.index)
    for col in DRIBBLE_OBJECT_COLUMNS:
        if col in dribbles.columns:
            has_object_fields = has_object_fields | _as_bool(dribbles[col]) | dribbles[col].notna()

    outcome_raw = _coalesce(dribbles, ["dribble_outcome_name", "dribble_outcome"], default=pd.NA).astype("string").str.strip()
    missing_outcome = outcome_raw.isna() | outcome_raw.eq("")
    outcome_counts = (
        dribbles["dribble_outcome_resolved"]
        .astype("string")
        .value_counts(dropna=False)
        .rename_axis("outcome")
        .to_dict()
    )
    return {
        "total_dribble_events": int(len(dribbles)),
        "total_before_filters": total_before_filters,
        "duplicates_removed": duplicates,
        "missing_dribble_object": int((~has_object_fields).sum()),
        "missing_outcome": int(missing_outcome.sum()),
        "outcomes": {str(k): int(v) for k, v in outcome_counts.items()},
        "filtered_out": max(0, total_before_filters - int(len(dribbles))),
    }


def _event_type_mask(events_df: pd.DataFrame, name: str) -> pd.Series:
    if "type_name" not in events_df.columns:
        return pd.Series(False, index=events_df.index)
    return events_df["type_name"].astype("string").str.strip().str.lower().eq(name.strip().lower())


def _team_pair_count(df: pd.DataFrame, mask: pd.Series, home_team_id: int | None, away_team_id: int | None) -> tuple[int, int]:
    if df.empty or "team_id" not in df.columns or home_team_id is None or away_team_id is None:
        return 0, 0
    team_ids = pd.to_numeric(df["team_id"], errors="coerce")
    use = mask.fillna(False)
    home = int((use & team_ids.eq(int(home_team_id))).sum())
    away = int((use & team_ids.eq(int(away_team_id))).sum())
    return home, away


def _team_pair_sum(df: pd.DataFrame, value_col: str, home_team_id: int | None, away_team_id: int | None) -> tuple[float, float]:
    if df.empty or "team_id" not in df.columns or value_col not in df.columns or home_team_id is None or away_team_id is None:
        return 0.0, 0.0
    work = df[["team_id", value_col]].copy()
    work["team_id"] = pd.to_numeric(work["team_id"], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    home = float(work.loc[work["team_id"] == int(home_team_id), value_col].fillna(0).sum())
    away = float(work.loc[work["team_id"] == int(away_team_id), value_col].fillna(0).sum())
    return home, away


def compute_offensive_team_stats(
    events_df: pd.DataFrame,
    shots_df: pd.DataFrame,
    match_id: int,
    home_team_id: int | None,
    away_team_id: int | None,
) -> dict[str, dict[str, float | int]]:
    events = events_df.copy()
    shots = shots_df.copy()

    if "match_id" in events.columns:
        event_match_ids = pd.to_numeric(events["match_id"], errors="coerce")
        events = events[event_match_ids == int(match_id)]
    if "match_id" in shots.columns:
        shot_match_ids = pd.to_numeric(shots["match_id"], errors="coerce")
        shots = shots[shot_match_ids == int(match_id)]

    shot_outcome = _coalesce(shots, ["shot_outcome", "shot_outcome_name"], default=pd.NA).astype("string").str.strip().str.lower()
    shots_on_target_mask = shot_outcome.isin({"goal", "saved", "saved to post"}).fillna(False)
    shots_off_target_mask = shot_outcome.isin({"off t", "wayward", "post"}).fillna(False)
    blocked_shots_mask = shot_outcome.eq("blocked").fillna(False)
    goals_mask = shot_outcome.eq("goal").fillna(False)
    if "shot_outcome_id" in shots.columns:
        goals_mask = goals_mask | pd.to_numeric(shots["shot_outcome_id"], errors="coerce").eq(97)
    shot_x = pd.to_numeric(_coalesce(shots, ["x", "location_x"], default=pd.NA), errors="coerce")
    shot_y = pd.to_numeric(_coalesce(shots, ["y", "location_y"], default=pd.NA), errors="coerce")
    shots_in_box = shot_x.ge(102.0) & shot_y.between(18.0, 62.0, inclusive="both")

    shot_totals = _team_pair_count(shots, pd.Series(True, index=shots.index), home_team_id, away_team_id)
    shot_on_target = _team_pair_count(shots, shots_on_target_mask, home_team_id, away_team_id)
    shot_off_target = _team_pair_count(shots, shots_off_target_mask, home_team_id, away_team_id)
    blocked_shots = _team_pair_count(shots, blocked_shots_mask, home_team_id, away_team_id)
    goals = _team_pair_count(shots, goals_mask, home_team_id, away_team_id)
    shots_box = _team_pair_count(shots, shots_in_box.fillna(False), home_team_id, away_team_id)
    xg_home, xg_away = _team_pair_sum(shots.assign(xg_value=pd.to_numeric(_coalesce(shots, ["xg", "shot_statsbomb_xg"], default=0.0), errors="coerce")), "xg_value", home_team_id, away_team_id)

    crosses = _team_pair_count(
        events,
        _event_type_mask(events, "Pass") & _as_bool(_coalesce(events, ["pass_cross"], default=False)),
        home_team_id,
        away_team_id,
    )

    dribble_events, _ = prepare_dribble_events(events)
    dribble_attempts = _team_pair_count(dribble_events, pd.Series(True, index=dribble_events.index), home_team_id, away_team_id)
    dribble_complete = _team_pair_count(dribble_events, dribble_events["dribble_is_complete"], home_team_id, away_team_id)

    return {
        "home": {
            "total_shots": shot_totals[0],
            "shots_on_target": shot_on_target[0],
            "shots_off_target": shot_off_target[0],
            "blocked_shots": blocked_shots[0],
            "goals": goals[0],
            "total_xg": round(xg_home, 2),
            "xg_per_shot": round((xg_home / shot_totals[0]), 3) if shot_totals[0] else 0.0,
            "dribble_attempts": dribble_attempts[0],
            "dribbles_completed": dribble_complete[0],
            "dribble_success_pct": round((dribble_complete[0] / dribble_attempts[0] * 100.0), 1) if dribble_attempts[0] else 0.0,
            "shots_in_box": shots_box[0],
            "crosses": crosses[0],
        },
        "away": {
            "total_shots": shot_totals[1],
            "shots_on_target": shot_on_target[1],
            "shots_off_target": shot_off_target[1],
            "blocked_shots": blocked_shots[1],
            "goals": goals[1],
            "total_xg": round(xg_away, 2),
            "xg_per_shot": round((xg_away / shot_totals[1]), 3) if shot_totals[1] else 0.0,
            "dribble_attempts": dribble_attempts[1],
            "dribbles_completed": dribble_complete[1],
            "dribble_success_pct": round((dribble_complete[1] / dribble_attempts[1] * 100.0), 1) if dribble_attempts[1] else 0.0,
            "shots_in_box": shots_box[1],
            "crosses": crosses[1],
        },
    }


def top_dribble_players(events_df: pd.DataFrame, team_id: int | None, top_n: int = 3) -> pd.DataFrame:
    if team_id is None or events_df.empty:
        return pd.DataFrame(columns=["player_name", "attempts", "completed", "incomplete", "success_pct"])

    dribbles, _ = prepare_dribble_events(events_df)
    if dribbles.empty or "team_id" not in dribbles.columns:
        return pd.DataFrame(columns=["player_name", "attempts", "completed", "incomplete", "success_pct"])

    scoped = dribbles[pd.to_numeric(dribbles["team_id"], errors="coerce") == int(team_id)].copy()
    if scoped.empty:
        return pd.DataFrame(columns=["player_name", "attempts", "completed", "incomplete", "success_pct"])

    if "player_name" not in scoped.columns:
        scoped["player_name"] = pd.NA
    if "player_nickname" in scoped.columns:
        nickname = scoped["player_nickname"].astype("string").str.strip()
        scoped["player_name"] = nickname.where(nickname.ne(""), pd.NA).combine_first(scoped["player_name"])
    scoped["player_name"] = scoped["player_name"].astype("string").fillna("Unknown Player")

    group_cols = [c for c in ("player_id", "player_name") if c in scoped.columns]
    out = (
        scoped.assign(
            attempts=1,
            completed=scoped["dribble_is_complete"].astype(int),
            incomplete=scoped["dribble_is_incomplete"].astype(int),
        )
        .groupby(group_cols, dropna=False, as_index=False)
        .agg(attempts=("attempts", "sum"), completed=("completed", "sum"), incomplete=("incomplete", "sum"))
    )
    out["success_pct"] = ((out["completed"] / out["attempts"].where(out["attempts"] > 0, 1)) * 100.0).round(1)
    out.loc[out["attempts"] == 0, "success_pct"] = 0.0
    out = out.sort_values(["attempts", "completed", "player_name"], ascending=[False, False, True])
    return out.head(int(top_n)).reset_index(drop=True)
