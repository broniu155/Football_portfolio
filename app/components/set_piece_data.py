from __future__ import annotations

from typing import Any

import pandas as pd

SET_PIECE_OUTPUT_COLUMNS: list[str] = [
    "event_id",
    "match_id",
    "team_id",
    "team",
    "player_id",
    "player",
    "taker",
    "minute",
    "period",
    "set_piece_type",
    "side",
    "subtype",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "target_zone",
    "outcome",
    "recipient",
    "linked_shot",
    "linked_goal",
    "short_set_piece",
]


def _coalesce(df: pd.DataFrame, candidates: list[str], default: Any = pd.NA) -> pd.Series:
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return pd.Series(default, index=df.index, dtype="object")
    out = df[cols[0]]
    for col in cols[1:]:
        out = out.combine_first(df[col])
    return out


def _as_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _norm(series: pd.Series) -> pd.Series:
    return _as_text(series).str.lower()


def _scalar_text(value: Any, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _is_truthy(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def classify_corner_side(start_y: float | int | None) -> str:
    y = pd.to_numeric(pd.Series([start_y]), errors="coerce").iloc[0]
    if pd.isna(y):
        return "Unknown"
    return "Left" if float(y) < 40.0 else "Right"


def classify_free_kick_type(event: pd.Series) -> str:
    event_type = _scalar_text(event.get("type_name")).lower()
    if event_type == "shot":
        return "Direct"
    pass_cross = _scalar_text(event.get("pass_cross")).lower()
    if pass_cross in {"true", "1", "yes"} or _is_truthy(event.get("pass_cross")):
        return "Crossed"
    start_x = pd.to_numeric(pd.Series([event.get("start_x")]), errors="coerce").iloc[0]
    end_x = pd.to_numeric(pd.Series([event.get("end_x")]), errors="coerce").iloc[0]
    if pd.notna(start_x) and pd.notna(end_x) and float(end_x) - float(start_x) < 10.0:
        return "Short"
    return "Indirect"


def classify_target_zone(end_x: float | int | None, end_y: float | int | None) -> str:
    x = pd.to_numeric(pd.Series([end_x]), errors="coerce").iloc[0]
    y = pd.to_numeric(pd.Series([end_y]), errors="coerce").iloc[0]
    if pd.isna(x) or pd.isna(y):
        return "Unknown"
    x_num = float(x)
    y_num = float(y)
    if x_num < 90.0:
        return "Recycled/Short"
    if x_num >= 114.0 and 30.0 <= y_num <= 50.0:
        return "Six-yard central"
    if x_num >= 108.0 and y_num < 30.0:
        return "Near-post"
    if x_num >= 108.0 and y_num > 50.0:
        return "Far-post"
    if x_num >= 100.0 and 18.0 <= y_num <= 62.0:
        return "Penalty area"
    return "Edge/Other"


def classify_delivery_subtype(row: pd.Series) -> str:
    set_piece_type = str(row.get("set_piece_type") or "")
    fk_type = str(row.get("free_kick_type") or "")
    zone = str(row.get("target_zone") or "")
    short_flag = bool(row.get("short_set_piece"))
    if short_flag:
        return "Short routine"
    if set_piece_type == "Corner":
        if zone in {"Near-post", "Far-post"}:
            return "Post delivery"
        if zone in {"Six-yard central", "Penalty area"}:
            return "Box delivery"
        return "Recycled"
    if set_piece_type == "Free Kick":
        if fk_type == "Direct":
            return "Direct shot"
        if fk_type == "Crossed":
            return "Crossed delivery"
        return "Indirect routine"
    return "Other"


def _set_piece_mask(events: pd.DataFrame) -> pd.Series:
    play_pattern = _norm(_coalesce(events, ["play_pattern_name", "play_pattern"], default=""))
    return play_pattern.isin({"from corner", "from free kick"})


def _resolve_side(set_piece_type: str, start_y: float | int | None) -> str:
    if set_piece_type == "Corner":
        return classify_corner_side(start_y)
    y = pd.to_numeric(pd.Series([start_y]), errors="coerce").iloc[0]
    if pd.isna(y):
        return "Unknown"
    if float(y) < 26.67:
        return "Left"
    if float(y) > 53.33:
        return "Right"
    return "Centre"


def _resolve_outcome(row: pd.Series) -> str:
    event_type = _scalar_text(row.get("type_name")).lower()
    if event_type == "pass":
        out = _scalar_text(row.get("pass_outcome_name")) or _scalar_text(row.get("pass_outcome"))
        return "Complete" if out == "" else out
    if event_type == "shot":
        out = _scalar_text(row.get("shot_outcome_name")) or _scalar_text(row.get("shot_outcome"))
        return "Unknown" if out == "" else out
    return "Unknown"


def _build_empty_output() -> pd.DataFrame:
    return pd.DataFrame(columns=SET_PIECE_OUTPUT_COLUMNS)


def extract_set_piece_events(
    events: pd.DataFrame,
    include_follow_up: bool = True,
    follow_up_seconds: int = 15,
    next_n_actions: int = 5,
) -> pd.DataFrame:
    if events.empty:
        return _build_empty_output()

    work = events.copy()
    work["event_type_norm"] = _norm(_coalesce(work, ["type_name", "type"], default=""))
    sp_mask = _set_piece_mask(work) & work["event_type_norm"].isin({"pass", "shot"})
    if not sp_mask.any():
        return _build_empty_output()

    sp = work.loc[sp_mask].copy()
    sp["event_index_num"] = pd.to_numeric(_coalesce(sp, ["event_index", "index"]), errors="coerce")
    sp["minute_num"] = pd.to_numeric(_coalesce(sp, ["minute"], default=0), errors="coerce").fillna(0)
    sp["second_num"] = pd.to_numeric(_coalesce(sp, ["second"], default=0), errors="coerce").fillna(0)
    sp["time_seconds"] = sp["minute_num"] * 60 + sp["second_num"]

    play_pattern_norm = _norm(_coalesce(sp, ["play_pattern_name", "play_pattern"], default=""))
    sp["set_piece_type"] = pd.Series("Free Kick", index=sp.index, dtype="string")
    sp.loc[play_pattern_norm.eq("from corner"), "set_piece_type"] = "Corner"

    sp["start_x"] = pd.to_numeric(_coalesce(sp, ["location_x", "x"]), errors="coerce")
    sp["start_y"] = pd.to_numeric(_coalesce(sp, ["location_y", "y"]), errors="coerce")
    sp["end_x"] = pd.to_numeric(_coalesce(sp, ["pass_end_location_x", "shot_end_location_x"]), errors="coerce")
    sp["end_y"] = pd.to_numeric(_coalesce(sp, ["pass_end_location_y", "shot_end_location_y"]), errors="coerce")
    sp["end_x"] = sp["end_x"].combine_first(sp["start_x"])
    sp["end_y"] = sp["end_y"].combine_first(sp["start_y"])

    sp["side"] = [
        _resolve_side(str(set_piece_type), start_y)
        for set_piece_type, start_y in zip(sp["set_piece_type"], sp["start_y"], strict=False)
    ]
    sp["target_zone"] = [classify_target_zone(end_x, end_y) for end_x, end_y in zip(sp["end_x"], sp["end_y"], strict=False)]
    sp["short_set_piece"] = (
        pd.to_numeric(sp["end_x"], errors="coerce") - pd.to_numeric(sp["start_x"], errors="coerce")
    ).lt(10.0).fillna(False)
    sp["free_kick_type"] = [classify_free_kick_type(row) for _, row in sp.iterrows()]
    sp["subtype"] = [classify_delivery_subtype(row) for _, row in sp.iterrows()]
    sp["outcome"] = [_resolve_outcome(row) for _, row in sp.iterrows()]

    sp["team"] = _as_text(_coalesce(sp, ["team_name"], default="Unknown")).fillna("Unknown")
    sp["player"] = _as_text(_coalesce(sp, ["player_name"], default="Unknown")).fillna("Unknown")
    sp["taker"] = sp["player"]
    sp["recipient"] = _as_text(_coalesce(sp, ["pass_recipient_name"], default=pd.NA))
    sp["team_id"] = pd.to_numeric(_coalesce(sp, ["team_id"]), errors="coerce")
    sp["player_id"] = pd.to_numeric(_coalesce(sp, ["player_id"]), errors="coerce")

    sp["linked_shot"] = False
    sp["linked_goal"] = False
    if include_follow_up:
        lookup = work.copy()
        lookup["event_index_num"] = pd.to_numeric(_coalesce(lookup, ["event_index", "index"]), errors="coerce")
        lookup["minute_num"] = pd.to_numeric(_coalesce(lookup, ["minute"], default=0), errors="coerce").fillna(0)
        lookup["second_num"] = pd.to_numeric(_coalesce(lookup, ["second"], default=0), errors="coerce").fillna(0)
        lookup["time_seconds"] = lookup["minute_num"] * 60 + lookup["second_num"]
        lookup["event_type_norm"] = _norm(_coalesce(lookup, ["type_name", "type"], default=""))
        lookup["shot_outcome_norm"] = _norm(_coalesce(lookup, ["shot_outcome_name", "shot_outcome"], default=""))
        lookup_team = pd.to_numeric(_coalesce(lookup, ["team_id"]), errors="coerce")

        linked_shot: list[bool] = []
        linked_goal: list[bool] = []
        for _, row in sp.iterrows():
            idx = row["event_index_num"]
            t0 = row["time_seconds"]
            team_id = row["team_id"]
            if pd.isna(idx):
                linked_shot.append(False)
                linked_goal.append(False)
                continue
            candidates = lookup[
                lookup["event_index_num"].gt(idx)
                & lookup["event_index_num"].le(idx + float(next_n_actions))
                & lookup["time_seconds"].le(float(t0) + float(follow_up_seconds))
            ].copy()
            if pd.notna(team_id):
                candidates = candidates[lookup_team.loc[candidates.index] == float(team_id)]
            shots = candidates[candidates["event_type_norm"] == "shot"]
            has_shot = not shots.empty
            has_goal = bool(has_shot and shots["shot_outcome_norm"].eq("goal").any())
            linked_shot.append(has_shot)
            linked_goal.append(has_goal)

        sp["linked_shot"] = linked_shot
        sp["linked_goal"] = linked_goal

    out = sp[SET_PIECE_OUTPUT_COLUMNS].copy()
    return out.reset_index(drop=True)
