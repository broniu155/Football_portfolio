from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


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


def classify_corner_side(start_y: float | int | None) -> str:
    y = pd.to_numeric(pd.Series([start_y]), errors="coerce").iloc[0]
    if pd.isna(y):
        return "Unknown"
    return "Left" if float(y) < 40.0 else "Right"


def classify_free_kick_type(event: pd.Series) -> str:
    event_type = str(event.get("type_name") or "").strip().lower()
    if event_type == "shot":
        return "Direct"
    pass_cross = str(event.get("pass_cross") or "").strip().lower()
    if pass_cross in {"true", "1", "yes"} or event.get("pass_cross") is True:
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
    event_type = str(row.get("type_name") or "").strip().lower()
    if event_type == "pass":
        out = str(row.get("pass_outcome_name") or row.get("pass_outcome") or "").strip()
        return "Complete" if out == "" else out
    if event_type == "shot":
        out = str(row.get("shot_outcome_name") or row.get("shot_outcome") or "").strip()
        return "Unknown" if out == "" else out
    return "Unknown"


def extract_set_piece_events(events: pd.DataFrame, follow_up_seconds: int = 15, next_n_actions: int = 5) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "match_id",
                "team",
                "player",
                "recipient",
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
                "linked_shot",
                "linked_goal",
                "short_set_piece",
            ]
        )

    work = events.copy()
    work["event_type_norm"] = _norm(_coalesce(work, ["type_name", "type"], default=""))
    sp_mask = _set_piece_mask(work) & work["event_type_norm"].isin({"pass", "shot"})
    if not sp_mask.any():
        return pd.DataFrame()

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
    sp["recipient"] = _as_text(_coalesce(sp, ["pass_recipient_name"], default=pd.NA))

    lookup = work.copy()
    lookup["event_index_num"] = pd.to_numeric(_coalesce(lookup, ["event_index", "index"]), errors="coerce")
    lookup["minute_num"] = pd.to_numeric(_coalesce(lookup, ["minute"], default=0), errors="coerce").fillna(0)
    lookup["second_num"] = pd.to_numeric(_coalesce(lookup, ["second"], default=0), errors="coerce").fillna(0)
    lookup["time_seconds"] = lookup["minute_num"] * 60 + lookup["second_num"]
    lookup["event_type_norm"] = _norm(_coalesce(lookup, ["type_name", "type"], default=""))
    lookup["shot_outcome_norm"] = _norm(_coalesce(lookup, ["shot_outcome_name", "shot_outcome"], default=""))
    lookup_team = pd.to_numeric(_coalesce(lookup, ["team_id"]), errors="coerce")

    sp_team = pd.to_numeric(_coalesce(sp, ["team_id"]), errors="coerce")
    linked_shot: list[bool] = []
    linked_goal: list[bool] = []
    for i, (_, row) in enumerate(sp.iterrows()):
        idx = row["event_index_num"]
        t0 = row["time_seconds"]
        team_id = sp_team.iloc[i] if i < len(sp_team) else pd.NA
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

    out = sp[
        [
            "event_id",
            "match_id",
            "team",
            "player",
            "recipient",
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
            "linked_shot",
            "linked_goal",
            "short_set_piece",
        ]
    ].copy()
    return out.reset_index(drop=True)


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


def _render_single_event_pitch(sp_df: pd.DataFrame) -> None:
    if sp_df.empty:
        st.info("No corner/free-kick deliveries available for tactical plot.")
        return
    choice_df = sp_df.copy()
    choice_df["event_label"] = (
        choice_df["minute"].astype("string").fillna("?")
        + "' | "
        + choice_df["team"].astype("string").fillna("Unknown")
        + " | "
        + choice_df["player"].astype("string").fillna("Unknown")
        + " | "
        + choice_df["set_piece_type"].astype("string").fillna("Unknown")
    )
    labels = choice_df["event_label"].tolist()
    selected = st.selectbox("Single Event Tactical View", options=labels, index=0, key="set_piece_single_event")
    row = choice_df[choice_df["event_label"] == selected].iloc[0]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=[row["start_x"], row["end_x"]],
            y=[row["start_y"], row["end_y"]],
            mode="lines+markers",
            line=dict(color="#42d392", width=3),
            marker=dict(size=[9, 11], color=["#7de8b8", "#42d392"]),
            name="Delivery",
            hovertemplate="x=%{x:.1f}, y=%{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Set Piece Delivery: {row['set_piece_type']} ({row['subtype']})",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#111a2b",
        font=dict(color="#e7edf7"),
        margin=dict(l=10, r=10, t=60, b=10),
        height=460,
        shapes=_pitch_shapes(),
    )
    fig.update_xaxes(range=[-2, 122], visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(range=[82, -2], visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)


def _render_aggregate_patterns(sp_df: pd.DataFrame) -> None:
    if sp_df.empty:
        return
    agg = (
        sp_df.groupby(["set_piece_type", "side", "target_zone", "subtype"], dropna=False)
        .size()
        .reset_index(name="events")
        .sort_values("events", ascending=False)
    )
    st.markdown("**Aggregate Pattern View**")
    st.dataframe(agg.head(15), use_container_width=True, hide_index=True)
    zone_counts = (
        sp_df.groupby(["set_piece_type", "target_zone"], dropna=False)
        .size()
        .reset_index(name="events")
        .sort_values(["set_piece_type", "events"], ascending=[True, False])
    )
    fig = px.bar(
        zone_counts,
        x="target_zone",
        y="events",
        color="set_piece_type",
        barmode="group",
        title="Delivery Target Zones",
    )
    fig.update_layout(margin=dict(l=8, r=8, t=42, b=8), height=360)
    st.plotly_chart(fig, use_container_width=True)


def render_set_piece_tactical_view(events: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Set Pieces</div>', unsafe_allow_html=True)
    st.caption("Corners and free kicks tactical view (Phase 1).")
    if events.empty:
        st.info("No events in current context.")
        return

    sp_df = extract_set_piece_events(events)
    if sp_df.empty:
        st.info("No corner or free-kick events in current context.")
        return

    type_filter = st.selectbox("Set Piece Type", options=["Both", "Corner", "Free Kick"], index=0, key="set_piece_type_filter")
    filtered = sp_df.copy()
    if type_filter != "Both":
        filtered = filtered[filtered["set_piece_type"] == type_filter]

    metrics = filtered.copy()
    total = int(len(metrics))
    linked_shots = int(metrics["linked_shot"].sum()) if "linked_shot" in metrics.columns else 0
    linked_goals = int(metrics["linked_goal"].sum()) if "linked_goal" in metrics.columns else 0
    short_count = int(metrics["short_set_piece"].sum()) if "short_set_piece" in metrics.columns else 0
    short_pct = (100.0 * short_count / total) if total else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Set pieces", total)
    k2.metric("Linked shots", linked_shots)
    k3.metric("Linked goals", linked_goals)
    k4.metric("Short routines %", f"{short_pct:.1f}%")

    _render_single_event_pitch(filtered)
    _render_aggregate_patterns(filtered)
