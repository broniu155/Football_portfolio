from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.data import get_active_data_mode, get_events


def get_filtered_events(
    match_id: int,
    team_id: int | None = None,
    player_id: int | None = None,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    work = events.copy() if events is not None else get_events(match_id=int(match_id))
    if work.empty:
        return work
    if "match_id" in work.columns:
        work = work[pd.to_numeric(work["match_id"], errors="coerce") == int(match_id)]
    if team_id is not None and "team_id" in work.columns:
        work = work[pd.to_numeric(work["team_id"], errors="coerce") == int(team_id)]
    if player_id is not None and "player_id" in work.columns:
        work = work[pd.to_numeric(work["player_id"], errors="coerce") == int(player_id)]
    return work


def _first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for col in names:
        if col in df.columns:
            return col
    return None


def _is_completed(pass_df: pd.DataFrame) -> pd.Series:
    if "pass_outcome_name" not in pass_df.columns:
        return pd.Series(True, index=pass_df.index)
    outcome = pass_df["pass_outcome_name"].astype("string").str.strip().str.lower()
    return outcome.isna() | (outcome == "") | (outcome == "none")


def get_pass_events(
    match_id: int,
    team_id: int | None = None,
    player_id: int | None = None,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = get_filtered_events(match_id=match_id, team_id=team_id, player_id=player_id, events=events)
    if base.empty or "type_name" not in base.columns:
        return base.iloc[0:0].copy()
    pass_df = base[base["type_name"].astype("string").str.strip().str.lower() == "pass"].copy()
    pass_df["is_completed"] = _is_completed(pass_df)
    return pass_df


@st.cache_data(show_spinner=False)
def get_pass_summary(match_id: int, team_id: int | None, player_id: int | None, data_mode: str) -> dict[str, Any]:
    del data_mode  # cache key only
    pass_df = get_pass_events(match_id=match_id, team_id=team_id, player_id=player_id, events=None)
    if pass_df.empty:
        return {"total": 0, "completed": 0, "completion_pct": 0.0, "progressive": 0, "key_passes": 0}

    start_x_col = _first_col(pass_df, ["location_x", "x"])
    end_x_col = _first_col(pass_df, ["pass_end_location_x", "pass_end_x", "end_x"])
    progressive = 0
    if start_x_col and end_x_col:
        sx = pd.to_numeric(pass_df[start_x_col], errors="coerce")
        ex = pd.to_numeric(pass_df[end_x_col], errors="coerce")
        progressive = int(((ex - sx) >= 15.0).fillna(False).sum())

    key_cols = [col for col in ("pass_shot_assist", "pass_goal_assist", "pass_assisted_shot_id") if col in pass_df.columns]
    if key_cols:
        key_mask = pd.Series(False, index=pass_df.index)
        for col in key_cols:
            if pass_df[col].dtype == "bool":
                key_mask = key_mask | pass_df[col].fillna(False)
            else:
                as_text = pass_df[col].astype("string").str.strip().str.lower()
                key_mask = key_mask | pass_df[col].notna() | as_text.isin({"true", "1", "yes", "y"})
        key_passes = int(key_mask.sum())
    else:
        key_passes = 0

    total = int(len(pass_df))
    completed = int(pass_df["is_completed"].sum())
    completion_pct = (completed / total * 100.0) if total else 0.0
    return {
        "total": total,
        "completed": completed,
        "completion_pct": completion_pct,
        "progressive": progressive,
        "key_passes": key_passes,
    }


def _pitch_shapes(line_color: str = "#6c8f78") -> list[dict[str, Any]]:
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color=line_color, width=2)),
        dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color=line_color, width=2)),
        dict(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color=line_color, width=2)),
        dict(type="circle", x0=59, y0=39, x1=61, y1=41, fillcolor=line_color, line=dict(color=line_color)),
        dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color=line_color, width=2)),
        dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color=line_color, width=2)),
        dict(type="circle", x0=11, y0=39, x1=13, y1=41, fillcolor=line_color, line=dict(color=line_color)),
        dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color=line_color, width=2)),
        dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color=line_color, width=2)),
        dict(type="circle", x0=107, y0=39, x1=109, y1=41, fillcolor=line_color, line=dict(color=line_color)),
    ]
    for shape in shapes:
        shape["layer"] = "below"
    return shapes


def _line_trace(df: pd.DataFrame, sx_col: str, sy_col: str, ex_col: str, ey_col: str, color: str, name: str) -> go.Scattergl:
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    for row in df.itertuples(index=False):
        sx = getattr(row, sx_col)
        sy = getattr(row, sy_col)
        ex = getattr(row, ex_col)
        ey = getattr(row, ey_col)
        if pd.isna(sx) or pd.isna(sy) or pd.isna(ex) or pd.isna(ey):
            continue
        x_values.extend([float(sx), float(ex), None])
        y_values.extend([float(sy), float(ey), None])
    return go.Scattergl(x=x_values, y=y_values, mode="lines", line=dict(width=1.4, color=color), name=name, opacity=0.62)


def _endpoint_markers(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color: str,
    name: str,
    size: float,
    opacity: float,
) -> go.Scattergl:
    x_values = pd.to_numeric(df[x_col], errors="coerce")
    y_values = pd.to_numeric(df[y_col], errors="coerce")
    keep = x_values.notna() & y_values.notna()
    return go.Scattergl(
        x=x_values[keep],
        y=y_values[keep],
        mode="markers",
        marker=dict(size=size, color=color, opacity=opacity, line=dict(width=0)),
        name=name,
        hoverinfo="skip",
        showlegend=False,
    )


def draw_pass_map(pass_df: pd.DataFrame, pass_status: str = "All", max_lines: int = 1400) -> go.Figure:
    fig = go.Figure()
    if pass_df.empty:
        fig.update_layout(
            paper_bgcolor="#0b1220",
            plot_bgcolor="#111a2b",
            font=dict(color="#e7edf7"),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=560,
        )
        return fig

    start_x_col = _first_col(pass_df, ["location_x", "x"])
    start_y_col = _first_col(pass_df, ["location_y", "y"])
    end_x_col = _first_col(pass_df, ["pass_end_location_x", "pass_end_x", "end_x"])
    end_y_col = _first_col(pass_df, ["pass_end_location_y", "pass_end_y", "end_y"])
    if not all([start_x_col, start_y_col, end_x_col, end_y_col]):
        fig.update_layout(
            paper_bgcolor="#0b1220",
            plot_bgcolor="#111a2b",
            font=dict(color="#e7edf7"),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=560,
            title="Pass Map",
        )
        return fig

    work = pass_df.copy()
    for col in [start_x_col, start_y_col, end_x_col, end_y_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[start_x_col, start_y_col, end_x_col, end_y_col])
    if work.empty:
        return fig

    if pass_status == "Completed":
        work = work[work["is_completed"]]
    elif pass_status == "Unsuccessful":
        work = work[~work["is_completed"]]

    if work.empty:
        return fig

    if len(work) > max_lines:
        work = work.sample(n=max_lines, random_state=42)

    completed = work[work["is_completed"]]
    unsuccessful = work[~work["is_completed"]]
    if not completed.empty:
        fig.add_trace(_line_trace(completed, start_x_col, start_y_col, end_x_col, end_y_col, "#42d392", "Completed"))
        fig.add_trace(_endpoint_markers(completed, start_x_col, start_y_col, "#7de8b8", "Completed start", size=4.5, opacity=0.45))
        fig.add_trace(_endpoint_markers(completed, end_x_col, end_y_col, "#42d392", "Completed end", size=8.5, opacity=0.85))
    if not unsuccessful.empty:
        fig.add_trace(_line_trace(unsuccessful, start_x_col, start_y_col, end_x_col, end_y_col, "#f59e0b", "Unsuccessful"))
        fig.add_trace(
            _endpoint_markers(unsuccessful, start_x_col, start_y_col, "#f8c46d", "Unsuccessful start", size=4.5, opacity=0.45)
        )
        fig.add_trace(_endpoint_markers(unsuccessful, end_x_col, end_y_col, "#f59e0b", "Unsuccessful end", size=8.5, opacity=0.85))

    fig.update_layout(
        title="Pass Map",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#111a2b",
        font=dict(color="#e7edf7"),
        margin=dict(l=10, r=10, t=48, b=15),
        height=560,
        shapes=_pitch_shapes(),
        legend=dict(orientation="h", yanchor="top", y=1.04, xanchor="left", x=0),
    )
    fig.update_xaxes(range=[-2, 122], visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(range=[-2, 82], visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
    return fig


def render_passes_section(match_id: int, team_id: int | None, player_id: int | None, events: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Passes</div>', unsafe_allow_html=True)
    pass_df = get_pass_events(match_id=match_id, team_id=team_id, player_id=player_id, events=events)
    summary = get_pass_summary(
        match_id=int(match_id),
        team_id=team_id,
        player_id=player_id,
        data_mode=get_active_data_mode(),
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Passes", summary["total"])
    m2.metric("Completed", summary["completed"])
    m3.metric("Completion %", f"{summary['completion_pct']:.1f}%")
    m4.metric("Progressive", summary["progressive"])
    m5.metric("Key Passes", summary["key_passes"])

    f1, f2 = st.columns([1, 1])
    with f1:
        status = st.selectbox("Pass Result", ["All", "Completed", "Unsuccessful"], index=0, key="passes_result_filter")
    with f2:
        max_lines = st.slider("Max Pass Lines", min_value=200, max_value=3000, value=1400, step=100, key="passes_line_cap")

    fig = draw_pass_map(pass_df, pass_status=status, max_lines=int(max_lines))
    st.plotly_chart(fig, use_container_width=True)

    if pass_df.empty:
        st.info("No pass events available for the current filter context.")
