from __future__ import annotations

# Match stats definitions and sources:
# - Official scoreline and "Goals" default values come from dim_match (home_score/away_score).
# - Shot-family metrics are computed per team using fact_shots when available (preferred source),
#   otherwise fallback to fact_events rows where type_name == "Shot".
# - "Shots on target", "Shots off target", and "Blocked shots" are derived from normalized
#   shot outcome names. If outcome data is unavailable, these metrics are returned as None ("—" in UI).

import html
from typing import Any

import pandas as pd
import streamlit as st

ON_TARGET_OUTCOMES = {"goal", "saved", "saved to post"}
OFF_TARGET_OUTCOMES = {"off t", "wayward", "post"}
BLOCKED_OUTCOMES = {"blocked"}


def _to_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coalesce(df: pd.DataFrame, candidates: list[str], default: object = pd.NA) -> pd.Series:
    existing = [column for column in candidates if column in df.columns]
    if not existing:
        return pd.Series(default, index=df.index, dtype="object")
    series = df[existing[0]]
    for column in existing[1:]:
        series = series.combine_first(df[column])
    return series


def _filter_match(df: pd.DataFrame, match_id: int) -> pd.DataFrame:
    if df.empty or "match_id" not in df.columns:
        return df.iloc[0:0].copy()
    work = df.copy()
    work["match_id"] = pd.to_numeric(work["match_id"], errors="coerce")
    return work[work["match_id"] == int(match_id)].copy()


def _extract_shot_events(fact_events_match: pd.DataFrame) -> pd.DataFrame:
    if fact_events_match.empty:
        return fact_events_match.iloc[0:0].copy()
    events = fact_events_match.copy()
    shot_mask = pd.Series(False, index=events.index)
    if "type_name" in events.columns:
        shot_mask = shot_mask | events["type_name"].astype(str).str.strip().str.lower().eq("shot")
    if "type_id" in events.columns:
        shot_mask = shot_mask | pd.to_numeric(events["type_id"], errors="coerce").eq(16)
    return events[shot_mask].copy()


def _build_outcome_map_from_events(shot_events: pd.DataFrame) -> dict[int, str]:
    if shot_events.empty or "shot_outcome_id" not in shot_events.columns or "shot_outcome_name" not in shot_events.columns:
        return {}
    mapping_df = shot_events[["shot_outcome_id", "shot_outcome_name"]].copy()
    mapping_df["shot_outcome_id"] = pd.to_numeric(mapping_df["shot_outcome_id"], errors="coerce")
    mapping_df = mapping_df.dropna(subset=["shot_outcome_id"])
    mapping_df["shot_outcome_id"] = mapping_df["shot_outcome_id"].astype(int)
    mapping_df["shot_outcome_name"] = mapping_df["shot_outcome_name"].astype(str).str.strip()
    mapping_df = mapping_df[mapping_df["shot_outcome_name"] != ""]
    mapping_df = mapping_df.drop_duplicates(subset=["shot_outcome_id"])
    return dict(zip(mapping_df["shot_outcome_id"], mapping_df["shot_outcome_name"]))


def _prepare_shot_rows(
    fact_events_match: pd.DataFrame,
    fact_shots_match: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    shots = fact_shots_match.copy()
    events_shots = _extract_shot_events(fact_events_match)

    if shots.empty or "team_id" not in shots.columns:
        shots = events_shots.copy()
    if shots.empty:
        shots["shot_outcome_norm"] = pd.Series(dtype="string")
        shots["xg_value"] = pd.Series(dtype="float64")
        return shots, False

    shots["team_id"] = pd.to_numeric(shots["team_id"], errors="coerce")
    shots = shots.dropna(subset=["team_id"])
    shots["team_id"] = shots["team_id"].astype(int)

    outcome = _coalesce(shots, ["shot_outcome", "shot_outcome_name"], default=pd.NA)
    if outcome.isna().all() and "shot_outcome_id" in shots.columns:
        outcome_id = pd.to_numeric(shots["shot_outcome_id"], errors="coerce")
        outcome_map = _build_outcome_map_from_events(events_shots)
        if outcome_map:
            mapped = outcome_id.map(outcome_map)
            outcome = outcome.combine_first(mapped)
    shots["shot_outcome_norm"] = outcome.astype("string").str.strip().str.lower()
    shots["xg_value"] = pd.to_numeric(_coalesce(shots, ["xg", "shot_statsbomb_xg"], default=pd.NA), errors="coerce")

    outcome_available = shots["shot_outcome_norm"].notna().any()
    return shots, bool(outcome_available)


def _team_metric_int(series: pd.Series, team_id: int) -> int:
    team_series = pd.to_numeric(series, errors="coerce")
    return int(team_series.eq(int(team_id)).sum())


def _pair_int(df: pd.DataFrame, home_team_id: int, away_team_id: int, predicate: pd.Series | None = None) -> tuple[int, int]:
    if df.empty or "team_id" not in df.columns:
        return 0, 0
    work = df
    if predicate is not None:
        work = work[predicate.fillna(False)]
    return _team_metric_int(work["team_id"], home_team_id), _team_metric_int(work["team_id"], away_team_id)


def _pair_float_sum(df: pd.DataFrame, value_col: str, home_team_id: int, away_team_id: int) -> tuple[float | None, float | None]:
    if df.empty or "team_id" not in df.columns or value_col not in df.columns:
        return None, None
    work = df[["team_id", value_col]].copy()
    work["team_id"] = pd.to_numeric(work["team_id"], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["team_id"])
    if work.empty:
        return None, None
    home = float(work.loc[work["team_id"] == int(home_team_id), value_col].fillna(0).sum())
    away = float(work.loc[work["team_id"] == int(away_team_id), value_col].fillna(0).sum())
    return home, away


def _value_text(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        txt = value.strip()
        return txt if txt else "—"
    if isinstance(value, float):
        if pd.isna(value):
            return "—"
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.2f}"
    return str(value)


def _bar_widths(left: object, right: object) -> tuple[float, float]:
    left_num = _to_float(left)
    right_num = _to_float(right)
    if left_num is None or right_num is None:
        return 50.0, 50.0
    total = left_num + right_num
    if total <= 0:
        return 50.0, 50.0
    return (left_num / total) * 100.0, (right_num / total) * 100.0


def compute_match_stats(
    fact_events: pd.DataFrame,
    fact_shots: pd.DataFrame,
    dim_match: pd.DataFrame,
    match_id: int,
) -> dict[str, Any]:
    match_rows = (
        dim_match[pd.to_numeric(dim_match["match_id"], errors="coerce") == int(match_id)]
        if "match_id" in dim_match.columns
        else pd.DataFrame()
    )
    if match_rows.empty:
        return {
            "home_team_id": None,
            "away_team_id": None,
            "home_team_name": "Home",
            "away_team_name": "Away",
            "home_score": None,
            "away_score": None,
            "match_date": None,
            "metrics": {},
        }

    row = match_rows.iloc[0]
    home_team_id = _to_int(row.get("home_team_id"))
    away_team_id = _to_int(row.get("away_team_id"))
    home_team_name = str(row.get("home_team_name") or "Home")
    away_team_name = str(row.get("away_team_name") or "Away")
    home_score = _to_int(row.get("home_score"))
    away_score = _to_int(row.get("away_score"))
    match_date = row.get("match_date")

    events_match = _filter_match(fact_events, match_id)
    shots_match = _filter_match(fact_shots, match_id)
    shots_rows, has_outcomes = _prepare_shot_rows(events_match, shots_match)

    metrics: dict[str, tuple[object, object]] = {}
    if home_team_id is None or away_team_id is None:
        metrics["Goals"] = (home_score, away_score)
        return {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_team_name": home_team_name,
            "away_team_name": away_team_name,
            "home_score": home_score,
            "away_score": away_score,
            "match_date": match_date,
            "metrics": metrics,
        }

    metrics["Shots"] = _pair_int(shots_rows, home_team_id, away_team_id)

    if has_outcomes and "shot_outcome_norm" in shots_rows.columns:
        outcome_norm = shots_rows["shot_outcome_norm"].fillna("").astype(str)
        metrics["Goals"] = _pair_int(
            shots_rows,
            home_team_id,
            away_team_id,
            predicate=outcome_norm.eq("goal"),
        )
        metrics["Shots on target"] = _pair_int(
            shots_rows,
            home_team_id,
            away_team_id,
            predicate=outcome_norm.isin(ON_TARGET_OUTCOMES),
        )
        metrics["Shots off target"] = _pair_int(
            shots_rows,
            home_team_id,
            away_team_id,
            predicate=outcome_norm.isin(OFF_TARGET_OUTCOMES),
        )
        metrics["Blocked shots"] = _pair_int(
            shots_rows,
            home_team_id,
            away_team_id,
            predicate=outcome_norm.isin(BLOCKED_OUTCOMES),
        )
    else:
        metrics["Goals"] = (None, None)
        metrics["Shots on target"] = (None, None)
        metrics["Shots off target"] = (None, None)
        metrics["Blocked shots"] = (None, None)

    home_xg, away_xg = _pair_float_sum(shots_rows, "xg_value", home_team_id, away_team_id)
    metrics["xG"] = (
        None if home_xg is None else round(home_xg, 2),
        None if away_xg is None else round(away_xg, 2),
    )

    return {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_team_name": home_team_name,
        "away_team_name": away_team_name,
        "home_score": home_score,
        "away_score": away_score,
        "match_date": match_date,
        "metrics": metrics,
    }


def validate_goals_consistency(stats: dict[str, Any], apply_filtered_stats: bool) -> None:
    if apply_filtered_stats:
        return
    goals = stats.get("metrics", {}).get("Goals", (None, None))
    home_goals = _to_int(goals[0] if isinstance(goals, tuple) else None)
    away_goals = _to_int(goals[1] if isinstance(goals, tuple) else None)
    home_score = _to_int(stats.get("home_score"))
    away_score = _to_int(stats.get("away_score"))
    assert home_goals == home_score and away_goals == away_score, (
        f"Goals metric mismatch: ({home_goals}, {away_goals}) vs official score ({home_score}, {away_score})"
    )


def render_match_score_header(stats: dict[str, Any]) -> None:
    home_name = html.escape(str(stats.get("home_team_name") or "Home"))
    away_name = html.escape(str(stats.get("away_team_name") or "Away"))
    match_date = html.escape(_value_text(stats.get("match_date")))
    home_score = html.escape(_value_text(stats.get("home_score")))
    away_score = html.escape(_value_text(stats.get("away_score")))
    st.markdown(
        (
            '<div class="match-score-card">'
            f'<div class="match-score-teams"><span>{home_name}</span><span>{away_name}</span></div>'
            f'<div class="match-scoreline">{home_score}<span class="match-score-sep">–</span>{away_score}</div>'
            f'<div class="match-score-meta">{home_name} vs {away_name} ({match_date})</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_match_stats_panel(stats: dict[str, Any], filtered: bool) -> None:
    metrics = stats.get("metrics", {})
    if not metrics:
        st.info("Match stats are unavailable for this match.")
        return

    context_label = "Filtered stats" if filtered else "Match-level stats"
    rows: list[str] = [f'<div class="match-stats-context">{html.escape(context_label)}</div>']

    for label, pair in metrics.items():
        left_val = pair[0] if isinstance(pair, tuple) and len(pair) > 0 else None
        right_val = pair[1] if isinstance(pair, tuple) and len(pair) > 1 else None
        left_txt = html.escape(_value_text(left_val))
        right_txt = html.escape(_value_text(right_val))
        lbl_txt = html.escape(str(label))
        left_w, right_w = _bar_widths(left_val, right_val)
        rows.append(
            (
                '<div class="match-stats-row">'
                f'<div class="match-stats-values"><span class="home">{left_txt}</span>'
                f'<span class="label">{lbl_txt}</span><span class="away">{right_txt}</span></div>'
                '<div class="match-stats-bars">'
                f'<div class="track left"><div class="fill home" style="width:{left_w:.1f}%"></div></div>'
                f'<div class="track right"><div class="fill away" style="width:{right_w:.1f}%"></div></div>'
                "</div>"
                "</div>"
            )
        )

    st.markdown(f'<div class="match-stats-panel">{"".join(rows)}</div>', unsafe_allow_html=True)
