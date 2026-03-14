from __future__ import annotations

from typing import Any
from typing import Literal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from app.components.set_piece_data import (
        classify_corner_side,
        classify_delivery_subtype,
        classify_free_kick_type,
        classify_target_zone,
        compute_set_piece_sanity_checks,
        extract_defensive_corner_clearances,
        extract_set_piece_events,
        summarize_defensive_corner_clearances,
    )
except (ModuleNotFoundError, KeyError):
    from components.set_piece_data import (
        classify_corner_side,
        classify_delivery_subtype,
        classify_free_kick_type,
        classify_target_zone,
        compute_set_piece_sanity_checks,
        extract_defensive_corner_clearances,
        extract_set_piece_events,
        summarize_defensive_corner_clearances,
    )


PRESET_OPTIONS: tuple[str, ...] = ("(None)", "Attacking FKs", "Corners leading to shots", "Short routines")


def _coalesce(df: pd.DataFrame, candidates: list[str], default: Any = pd.NA) -> pd.Series:
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return pd.Series(default, index=df.index, dtype="object")
    out = df[cols[0]]
    for col in cols[1:]:
        out = out.combine_first(df[col])
    return out


def _norm(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def _apply_context_filters(events: pd.DataFrame, team_id: int | None, player_id: int | None) -> pd.DataFrame:
    out = events.copy()
    if team_id is not None and "team_id" in out.columns:
        out = out[pd.to_numeric(out["team_id"], errors="coerce") == int(team_id)]
    if player_id is not None and "player_id" in out.columns:
        out = out[pd.to_numeric(out["player_id"], errors="coerce") == int(player_id)]
    return out


def _pitch_shapes(line_color: str = "#6c8f78") -> list[dict[str, Any]]:
    shapes: list[dict[str, Any]] = [
        dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color=line_color, width=2)),
        dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color=line_color, width=2)),
        dict(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color=line_color, width=2)),
        dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color=line_color, width=2)),
        dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color=line_color, width=2)),
        dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color=line_color, width=2)),
        dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color=line_color, width=2)),
    ]
    for shape in shapes:
        shape["layer"] = "below"
    return shapes


def _base_pitch_layout(title: str, height: int = 520) -> dict[str, Any]:
    return dict(
        title=title,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#111a2b",
        font=dict(color="#e7edf7"),
        margin=dict(l=10, r=10, t=60, b=10),
        height=height,
        shapes=_pitch_shapes(),
    )


def _follow_up_actions(
    events: pd.DataFrame,
    set_piece_row: pd.Series,
    follow_up_seconds: int = 15,
    next_n_actions: int = 5,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    work = events.copy()
    work["event_index_num"] = pd.to_numeric(_coalesce(work, ["event_index", "index"]), errors="coerce")
    work["minute_num"] = pd.to_numeric(_coalesce(work, ["minute"], default=0), errors="coerce").fillna(0)
    work["second_num"] = pd.to_numeric(_coalesce(work, ["second"], default=0), errors="coerce").fillna(0)
    work["time_seconds"] = work["minute_num"] * 60 + work["second_num"]
    work["team_id_num"] = pd.to_numeric(_coalesce(work, ["team_id"]), errors="coerce")
    work["event_type_norm"] = _norm(_coalesce(work, ["type_name", "type"], default=""))
    work["shot_outcome_norm"] = _norm(_coalesce(work, ["shot_outcome_name", "shot_outcome"], default=""))
    work["start_x"] = pd.to_numeric(_coalesce(work, ["location_x", "x"]), errors="coerce")
    work["start_y"] = pd.to_numeric(_coalesce(work, ["location_y", "y"]), errors="coerce")
    work["end_x"] = pd.to_numeric(
        _coalesce(work, ["pass_end_location_x", "carry_end_location_x", "shot_end_location_x", "location_x", "x"]),
        errors="coerce",
    )
    work["end_y"] = pd.to_numeric(
        _coalesce(work, ["pass_end_location_y", "carry_end_location_y", "shot_end_location_y", "location_y", "y"]),
        errors="coerce",
    )

    event_id = set_piece_row.get("event_id")
    anchor = work[work["event_id"].astype("string") == str(event_id)].head(1)
    if anchor.empty:
        return pd.DataFrame()

    idx = pd.to_numeric(anchor["event_index_num"], errors="coerce").iloc[0]
    t0 = pd.to_numeric(anchor["time_seconds"], errors="coerce").fillna(0).iloc[0]
    team_id = pd.to_numeric(anchor["team_id_num"], errors="coerce").iloc[0]
    if pd.isna(idx):
        return pd.DataFrame()

    candidates = work[
        work["event_index_num"].gt(float(idx))
        & work["event_index_num"].le(float(idx) + float(next_n_actions))
        & work["time_seconds"].le(float(t0) + float(follow_up_seconds))
    ].copy()
    if pd.notna(team_id):
        candidates = candidates[candidates["team_id_num"] == float(team_id)]

    if candidates.empty:
        return candidates
    candidates["is_shot"] = candidates["event_type_norm"].eq("shot")
    candidates["is_goal"] = candidates["is_shot"] & candidates["shot_outcome_norm"].eq("goal")
    return candidates


def apply_set_piece_preset(sp_df: pd.DataFrame, preset: str) -> pd.DataFrame:
    if sp_df.empty or preset == "(None)":
        return sp_df.copy()

    filtered = sp_df.copy()
    if preset == "Attacking FKs":
        set_piece_type = filtered["set_piece_type"].astype("string")
        start_x = pd.to_numeric(filtered["start_x"], errors="coerce")
        filtered = filtered[set_piece_type.eq("Free Kick") & start_x.ge(80.0).fillna(False)]
    elif preset == "Corners leading to shots":
        set_piece_type = filtered["set_piece_type"].astype("string")
        linked = filtered["linked_shot"].fillna(False).astype(bool) | filtered["linked_goal"].fillna(False).astype(bool)
        filtered = filtered[set_piece_type.eq("Corner") & linked]
    elif preset == "Short routines":
        filtered = filtered[filtered["short_set_piece"].fillna(False).astype(bool)]
    return filtered.reset_index(drop=True)


def _read_filter_state(sp_df: pd.DataFrame) -> tuple[dict[str, Any], bool]:
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    c7, c8, c9, c10 = st.columns(4)

    team_options = sorted(sp_df["team"].astype("string").fillna("Unknown").unique().tolist())
    type_options = sorted(sp_df["set_piece_type"].astype("string").fillna("Unknown").unique().tolist())
    taker_options = sorted(sp_df["taker"].astype("string").fillna("Unknown").unique().tolist())
    half_options = ["(All)", "First Half", "Second Half", "Other"]
    subtype_options = sorted(sp_df["subtype"].astype("string").fillna("Unknown").unique().tolist())
    outcome_options = sorted(sp_df["outcome"].astype("string").fillna("Unknown").unique().tolist())

    preset = c1.selectbox("Preset", options=list(PRESET_OPTIONS), index=0, key="sp_preset_filter")
    team_filter = c2.multiselect("Team", options=team_options, default=[], key="sp_team_filter")
    type_filter = c3.multiselect("Set Piece Type", options=type_options, default=[], key="sp_type_filter")
    taker_filter = c4.multiselect("Taker", options=taker_options, default=[], key="sp_taker_filter")
    half_filter = c5.selectbox("Half", options=half_options, index=0, key="sp_half_filter")
    subtype_filter = c6.multiselect("Subtype", options=subtype_options, default=[], key="sp_subtype_filter")
    outcome_filter = c7.multiselect("Outcome", options=outcome_options, default=[], key="sp_outcome_filter")
    include_follow_up_only = c8.toggle("Include follow-up actions", value=False, key="sp_follow_up_filter")
    show_follow_up_overlay = c9.toggle("Show linked actions on Single Event", value=True, key="sp_follow_up_overlay")
    taker_search = c10.text_input("Search taker", value="", key="sp_taker_search").strip().lower()

    state = {
        "preset": str(preset),
        "team_filter": team_filter,
        "type_filter": type_filter,
        "taker_filter": taker_filter,
        "half_filter": str(half_filter),
        "subtype_filter": subtype_filter,
        "outcome_filter": outcome_filter,
        "include_follow_up_only": bool(include_follow_up_only),
        "taker_search": taker_search,
    }
    return state, bool(show_follow_up_overlay)


def apply_set_piece_filter_state(sp_df: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
    filtered = apply_set_piece_preset(sp_df, str(state.get("preset", "(None)")))
    if filtered.empty:
        return filtered.copy()

    team_filter = state.get("team_filter", [])
    type_filter = state.get("type_filter", [])
    taker_filter = state.get("taker_filter", [])
    subtype_filter = state.get("subtype_filter", [])
    outcome_filter = state.get("outcome_filter", [])
    taker_search = str(state.get("taker_search", "")).strip().lower()
    half_filter = str(state.get("half_filter", "(All)"))

    if team_filter:
        filtered = filtered[filtered["team"].astype("string").isin(team_filter)]
    if type_filter:
        filtered = filtered[filtered["set_piece_type"].astype("string").isin(type_filter)]
    if taker_filter:
        filtered = filtered[filtered["taker"].astype("string").isin(taker_filter)]
    if subtype_filter:
        filtered = filtered[filtered["subtype"].astype("string").isin(subtype_filter)]
    if outcome_filter:
        filtered = filtered[filtered["outcome"].astype("string").isin(outcome_filter)]
    if taker_search:
        taker_norm = filtered["taker"].astype("string").fillna("").str.lower()
        filtered = filtered[taker_norm.str.contains(taker_search, regex=False)]

    period_num = pd.to_numeric(filtered["period"], errors="coerce")
    if half_filter == "First Half":
        filtered = filtered[period_num.eq(1)]
    elif half_filter == "Second Half":
        filtered = filtered[period_num.eq(2)]
    elif half_filter == "Other":
        filtered = filtered[~period_num.isin([1, 2])]

    if bool(state.get("include_follow_up_only", False)):
        linked = filtered["linked_shot"].fillna(False).astype(bool) | filtered["linked_goal"].fillna(False).astype(bool)
        filtered = filtered[linked]

    return filtered.reset_index(drop=True)


def build_set_piece_event_options(sp_df: pd.DataFrame) -> pd.DataFrame:
    if sp_df.empty:
        return pd.DataFrame(columns=["event_key", "event_label"])
    work = sp_df.copy()
    minute = pd.to_numeric(work["minute"], errors="coerce").fillna(-1).astype(int).astype("string")
    period = pd.to_numeric(work["period"], errors="coerce").fillna(-1).astype(int).astype("string")
    team = work["team"].astype("string").fillna("Unknown")
    taker = work["taker"].astype("string").fillna("Unknown")
    set_piece_type = work["set_piece_type"].astype("string").fillna("Unknown")
    subtype = work["subtype"].astype("string").fillna("Unknown")
    event_id = work["event_id"].astype("string").fillna("n/a")

    work["event_label"] = (
        "P"
        + period
        + " "
        + minute
        + "' | "
        + team
        + " | "
        + taker
        + " | "
        + set_piece_type
        + " | "
        + subtype
        + " | id:"
        + event_id
    )
    return work[["event_key", "event_label"]].drop_duplicates(subset=["event_key"]).reset_index(drop=True)


def build_set_piece_compare_table(restart_df: pd.DataFrame, phase_df: pd.DataFrame) -> pd.DataFrame:
    if restart_df.empty and phase_df.empty:
        return pd.DataFrame(columns=["team", "set_piece_type", "restart_only_events", "phase_events", "delta"])

    def _team_type_counts(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["team", "set_piece_type", value_name])
        return (
            df.assign(
                team=df["team"].astype("string").fillna("Unknown"),
                set_piece_type=df["set_piece_type"].astype("string").fillna("Unknown"),
            )
            .groupby(["team", "set_piece_type"], dropna=False)
            .size()
            .reset_index(name=value_name)
        )

    restart_counts = _team_type_counts(restart_df, "restart_only_events")
    phase_counts = _team_type_counts(phase_df, "phase_events")
    merged = restart_counts.merge(phase_counts, on=["team", "set_piece_type"], how="outer").fillna(0)
    merged["restart_only_events"] = pd.to_numeric(merged["restart_only_events"], errors="coerce").fillna(0).astype(int)
    merged["phase_events"] = pd.to_numeric(merged["phase_events"], errors="coerce").fillna(0).astype(int)
    merged["delta"] = merged["phase_events"] - merged["restart_only_events"]
    return merged.sort_values(["team", "set_piece_type"]).reset_index(drop=True)


def apply_corner_exit_filters(
    clearance_df: pd.DataFrame,
    team_filter: list[str] | None = None,
    half_filter: str = "(All)",
    winner_filter: str = "(All)",
) -> pd.DataFrame:
    if clearance_df.empty:
        return clearance_df.copy()

    filtered = clearance_df.copy()
    if team_filter:
        filtered = filtered[filtered["team"].astype("string").isin(team_filter)]

    period_num = pd.to_numeric(filtered["period"], errors="coerce")
    if half_filter == "First Half":
        filtered = filtered[period_num.eq(1)]
    elif half_filter == "Second Half":
        filtered = filtered[period_num.eq(2)]
    elif half_filter == "Other":
        filtered = filtered[~period_num.isin([1, 2])]

    if winner_filter != "(All)":
        filtered = filtered[filtered["first_ball_winner"].astype("string") == winner_filter]
    return filtered.reset_index(drop=True)


def _render_single_event_view(sp_df: pd.DataFrame, raw_events: pd.DataFrame, show_follow_up_overlay: bool) -> None:
    if sp_df.empty:
        st.info("No set-piece events for current filter selection.")
        return

    work = sp_df.copy()
    options_df = build_set_piece_event_options(work)
    key_to_label = dict(zip(options_df["event_key"], options_df["event_label"], strict=False))
    option_keys = options_df["event_key"].tolist()
    selected_key = st.selectbox(
        "Select Event",
        options=option_keys,
        format_func=lambda key: key_to_label.get(str(key), str(key)),
        index=0,
        key="sp_single_event_selector",
    )
    selected_row = work[work["event_key"].astype("string") == str(selected_key)]
    if selected_row.empty:
        st.warning("Selected event is not available in current filter context.")
        return
    row = selected_row.iloc[0]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=[row["start_x"], row["end_x"]],
            y=[row["start_y"], row["end_y"]],
            mode="lines",
            line=dict(width=4, color="#42d392"),
            name="Set-piece delivery",
            hovertemplate="Delivery<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=[row["start_x"]],
            y=[row["start_y"]],
            mode="markers",
            marker=dict(size=10, color="#6aa6ff"),
            name="Origin",
            hovertemplate="Origin (%{x:.1f}, %{y:.1f})<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=[row["end_x"]],
            y=[row["end_y"]],
            mode="markers",
            marker=dict(size=12, color="#42d392"),
            name="End point",
            hovertemplate="End (%{x:.1f}, %{y:.1f})<extra></extra>",
        )
    )

    if show_follow_up_overlay:
        follow = _follow_up_actions(raw_events, row, follow_up_seconds=15, next_n_actions=5)
        if not follow.empty:
            for _, action in follow.iterrows():
                fig.add_trace(
                    go.Scattergl(
                        x=[action.get("start_x"), action.get("end_x")],
                        y=[action.get("start_y"), action.get("end_y")],
                        mode="lines",
                        line=dict(width=2, color="#f59e0b", dash="dot"),
                        name="Linked action",
                        showlegend=False,
                        hovertemplate=f"{str(action.get('type_name') or 'Action')}<extra></extra>",
                    )
                )
            goals = int(follow["is_goal"].fillna(False).astype(bool).sum()) if "is_goal" in follow.columns else 0
            shots = int(follow["is_shot"].fillna(False).astype(bool).sum()) if "is_shot" in follow.columns else 0
            st.caption(f"Linked follow-up actions in window: {len(follow)} (shots: {shots}, goals: {goals})")

    fig.update_layout(**_base_pitch_layout(title="Single Event Tactical View", height=500))
    fig.update_xaxes(range=[-2, 122], visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(range=[82, -2], visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)

    meta_cols = st.columns(5)
    meta_cols[0].metric("Type", str(row["set_piece_type"]))
    meta_cols[1].metric("Side", str(row["side"]))
    meta_cols[2].metric("Subtype", str(row["subtype"]))
    meta_cols[3].metric("Target zone", str(row["target_zone"]))
    meta_cols[4].metric("Outcome", str(row["outcome"]))


def _line_trace(df: pd.DataFrame, color: str, name: str, opacity: float = 0.22) -> go.Scattergl:
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    for row in df.itertuples(index=False):
        sx = getattr(row, "start_x")
        sy = getattr(row, "start_y")
        ex = getattr(row, "end_x")
        ey = getattr(row, "end_y")
        if pd.isna(sx) or pd.isna(sy) or pd.isna(ex) or pd.isna(ey):
            continue
        x_values.extend([float(sx), float(ex), None])
        y_values.extend([float(sy), float(ey), None])
    return go.Scattergl(x=x_values, y=y_values, mode="lines", line=dict(width=1.4, color=color), name=name, opacity=opacity)


def _render_pattern_view(sp_df: pd.DataFrame) -> None:
    if sp_df.empty:
        st.info("No set-piece patterns for current filter selection.")
        return

    work = sp_df.copy()
    if len(work) > 500:
        work = work.sample(n=500, random_state=42)

    fig = go.Figure()
    corner_df = work[work["set_piece_type"].astype("string") == "Corner"]
    fk_df = work[work["set_piece_type"].astype("string") == "Free Kick"]
    if not corner_df.empty:
        fig.add_trace(_line_trace(corner_df, color="#42d392", name="Corner deliveries"))
    if not fk_df.empty:
        fig.add_trace(_line_trace(fk_df, color="#6aa6ff", name="Free-kick deliveries"))

    zone_colors = {
        "Near-post": "#f59e0b",
        "Six-yard central": "#ef4444",
        "Far-post": "#8b5cf6",
        "Penalty area": "#22c55e",
        "Recycled/Short": "#94a3b8",
        "Edge/Other": "#eab308",
        "Unknown": "#cbd5e1",
    }
    for zone, chunk in work.groupby(work["target_zone"].astype("string").fillna("Unknown")):
        fig.add_trace(
            go.Scattergl(
                x=pd.to_numeric(chunk["end_x"], errors="coerce"),
                y=pd.to_numeric(chunk["end_y"], errors="coerce"),
                mode="markers",
                marker=dict(size=7, color=zone_colors.get(str(zone), "#cbd5e1"), opacity=0.85),
                name=str(zone),
                hovertemplate=f"{zone}<extra></extra>",
            )
        )

    fig.update_layout(**_base_pitch_layout(title="Pattern View", height=560))
    fig.update_xaxes(range=[-2, 122], visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(range=[82, -2], visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Lines show delivery direction; markers show delivery end points (target areas).")


def _render_summary_view(
    sp_df: pd.DataFrame,
    restart_filtered: pd.DataFrame,
    phase_filtered: pd.DataFrame,
    counting_mode: Literal["restart_only", "phase_events"],
) -> None:
    if sp_df.empty:
        st.info("No summary available for current filter selection.")
        return

    checks = compute_set_piece_sanity_checks(sp_df)
    total = int(checks["total_rows"])
    linked_shots = int(checks["linked_shot_total"])
    linked_goals = int(checks["linked_goal_total"])
    shot_rate = (100.0 * linked_shots / total) if total else 0.0
    goal_rate = (100.0 * linked_goals / total) if total else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total events", total)
    m2.metric("Linked shots", linked_shots)
    m3.metric("Linked goals", linked_goals)
    m4.metric("Shot/Goal rate", f"{shot_rate:.1f}% / {goal_rate:.1f}%")

    taker_dist = (
        sp_df["taker"]
        .astype("string")
        .fillna("Unknown")
        .value_counts()
        .rename_axis("taker")
        .reset_index(name="events")
        .head(12)
    )
    subtype_dist = (
        sp_df["subtype"]
        .astype("string")
        .fillna("Unknown")
        .value_counts()
        .rename_axis("subtype")
        .reset_index(name="events")
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Side split**")
        st.dataframe(checks["side_distribution"], use_container_width=True, hide_index=True)
        st.markdown("**Target zone split**")
        st.dataframe(checks["target_zone_distribution"], use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Subtype split**")
        st.dataframe(subtype_dist, use_container_width=True, hide_index=True)
        st.markdown("**Taker distribution**")
        st.dataframe(taker_dist, use_container_width=True, hide_index=True)

    reason_dist = (
        sp_df["restart_event_reason"]
        .astype("string")
        .fillna("Unknown")
        .value_counts()
        .rename_axis("restart_event_reason")
        .reset_index(name="events")
    )
    st.markdown("**Restart event reason**")
    st.dataframe(reason_dist, use_container_width=True, hide_index=True)

    st.markdown("**Set-piece type counts**")
    st.dataframe(checks["counts_by_set_piece_type"], use_container_width=True, hide_index=True)

    compare_enabled = st.toggle("Compare restart-only vs phase-events per team", value=False, key="sp_compare_modes")
    if compare_enabled:
        compare_table = build_set_piece_compare_table(restart_filtered, phase_filtered)
        if compare_table.empty:
            st.info("No comparison data available for current filters.")
            return
        left, right = st.columns(2)
        with left:
            st.markdown("**`restart_only` distribution**")
            st.dataframe(
                compare_table[["team", "set_piece_type", "restart_only_events"]],
                use_container_width=True,
                hide_index=True,
            )
        with right:
            st.markdown("**`phase_events` distribution**")
            st.dataframe(
                compare_table[["team", "set_piece_type", "phase_events", "delta"]],
                use_container_width=True,
                hide_index=True,
            )
        st.caption(
            f"Comparison uses the same preset and manual filters while the active mode stays `{counting_mode}`."
        )


def _render_corner_exit_view(clearance_df: pd.DataFrame) -> None:
    st.markdown("**Defensive corner exit lanes**")
    st.caption(
        "Lane is estimated from the first post-clearance actionable event with a valid location. "
        "Use this to position rest-defense players where corners are most often cleared."
    )
    if clearance_df.empty:
        st.info("No defensive corner clearances found in the current context.")
        return

    team_options = sorted(clearance_df["team"].astype("string").fillna("Unknown").unique().tolist())
    c1, c2, c3 = st.columns(3)
    team_filter = c1.multiselect("Defending team", options=team_options, default=[], key="sp_corner_exit_team_filter")
    half_filter = c2.selectbox(
        "Half",
        options=["(All)", "First Half", "Second Half", "Other"],
        index=0,
        key="sp_corner_exit_half_filter",
    )
    winner_filter = c3.selectbox(
        "First ball after clearance",
        options=["(All)", "Defending team", "Attacking team", "Unknown"],
        index=0,
        key="sp_corner_exit_winner_filter",
    )

    filtered = apply_corner_exit_filters(
        clearance_df,
        team_filter=team_filter,
        half_filter=str(half_filter),
        winner_filter=str(winner_filter),
    )
    if filtered.empty:
        st.info("No defensive corner clearances match the current filters.")
        return

    summary = summarize_defensive_corner_clearances(filtered)
    total_clearances = int(len(filtered))
    located_exits = int(filtered["exit_lane"].astype("string").ne("Unknown").sum())
    defending_team_first_ball = int(filtered["first_ball_winner"].astype("string").eq("Defending team").sum())
    defending_team_first_ball_pct = (100.0 * defending_team_first_ball / total_clearances) if total_clearances else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Corner clearances", total_clearances)
    m2.metric("Located exit lanes", f"{located_exits}/{total_clearances}")
    m3.metric("Defenders won first ball", f"{defending_team_first_ball_pct:.1f}%")

    lane_colors = {"Left": "#38bdf8", "Centre": "#f59e0b", "Right": "#22c55e", "Unknown": "#94a3b8"}
    bar = go.Figure(
        data=[
            go.Bar(
                x=summary["share_pct"],
                y=summary["exit_lane"],
                orientation="h",
                marker=dict(color=[lane_colors.get(str(lane), "#94a3b8") for lane in summary["exit_lane"]]),
                text=summary["share_pct"].map(lambda value: f"{float(value):.1f}%"),
                textposition="outside",
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            )
        ]
    )
    bar.update_layout(
        title="Where defenders clear the first ball from corners",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#111a2b",
        font=dict(color="#e7edf7"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=360,
        xaxis=dict(title="Share of clearances (%)", range=[0, max(100.0, float(summary["share_pct"].max()) + 5.0)]),
        yaxis=dict(title="Exit lane"),
    )

    left, right = st.columns([1.4, 1.0])
    with left:
        st.plotly_chart(bar, use_container_width=True)
    with right:
        st.markdown("**Lane summary**")
        st.dataframe(
            summary.rename(
                columns={
                    "exit_lane": "Exit lane",
                    "clearances": "Clearances",
                    "share_pct": "Share %",
                    "defending_team_first_ball": "Defenders first ball",
                    "defending_team_first_ball_pct": "Defenders first ball %",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    detail = filtered[
        [
            "minute",
            "team",
            "player",
            "exit_lane",
            "first_ball_winner",
            "exit_event_type",
            "exit_team",
            "seconds_to_first_action",
        ]
    ].copy()
    detail = detail.rename(
        columns={
            "minute": "Minute",
            "team": "Defending team",
            "player": "Clearance player",
            "exit_lane": "Exit lane",
            "first_ball_winner": "First ball winner",
            "exit_event_type": "Next action",
            "exit_team": "Next action team",
            "seconds_to_first_action": "Seconds",
        }
    )
    st.markdown("**Clearance log**")
    st.dataframe(detail, use_container_width=True, hide_index=True)


def render_set_piece_tactical_view(
    events: pd.DataFrame,
    context_team_id: int | None = None,
    context_player_id: int | None = None,
) -> None:
    st.markdown('<div class="section-title">Set Piece Tactical View</div>', unsafe_allow_html=True)
    st.caption("Tactical analysis for attacking set pieces and defensive corner exits.")
    if events.empty:
        st.info("No events in current context.")
        return

    analysis_events = _apply_context_filters(events, team_id=context_team_id, player_id=context_player_id)

    counting_mode = st.radio(
        "Set-piece counting logic",
        options=["restart_only", "phase_events"],
        index=0,
        horizontal=True,
        format_func=lambda value: "Restart events (recommended)" if value == "restart_only" else "All phase events",
        key="sp_counting_mode",
    )

    sp_df = extract_set_piece_events(
        analysis_events,
        include_follow_up=True,
        follow_up_seconds=15,
        next_n_actions=5,
        counting_mode=str(counting_mode),
    )
    restart_df = extract_set_piece_events(
        analysis_events,
        include_follow_up=True,
        follow_up_seconds=15,
        next_n_actions=5,
        counting_mode="restart_only",
    )
    phase_df = extract_set_piece_events(
        analysis_events,
        include_follow_up=True,
        follow_up_seconds=15,
        next_n_actions=5,
        counting_mode="phase_events",
    )
    clearance_df = extract_defensive_corner_clearances(
        events,
        follow_up_seconds=12,
        next_n_actions=6,
        team_id=context_team_id,
        player_id=context_player_id,
    )

    show_follow_up_overlay = False
    filtered = sp_df.copy()
    restart_filtered = restart_df.copy()
    phase_filtered = phase_df.copy()
    if not sp_df.empty:
        filter_state, show_follow_up_overlay = _read_filter_state(sp_df)
        filtered = apply_set_piece_filter_state(sp_df, filter_state)
        restart_filtered = apply_set_piece_filter_state(restart_df, filter_state)
        phase_filtered = apply_set_piece_filter_state(phase_df, filter_state)

    tab_single, tab_pattern, tab_summary, tab_corner_exits = st.tabs(
        ["Single Event", "Pattern View", "Summary", "Corner Exits"]
    )
    with tab_single:
        _render_single_event_view(filtered, raw_events=events, show_follow_up_overlay=show_follow_up_overlay)
    with tab_pattern:
        _render_pattern_view(filtered)
    with tab_summary:
        _render_summary_view(
            filtered,
            restart_filtered=restart_filtered,
            phase_filtered=phase_filtered,
            counting_mode=str(counting_mode),
        )
    with tab_corner_exits:
        _render_corner_exit_view(clearance_df)
