from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.data import get_events


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


def _normalize_height(pass_df: pd.DataFrame) -> pd.Series:
    if "pass_height_name" not in pass_df.columns:
        return pd.Series("unknown", index=pass_df.index, dtype="string")
    normalized = pass_df["pass_height_name"].astype("string").str.strip().str.lower()
    normalized = normalized.fillna("unknown").replace("", "unknown")
    return normalized


def _cross_mask(pass_df: pd.DataFrame) -> pd.Series:
    if "pass_cross" not in pass_df.columns:
        return pd.Series(False, index=pass_df.index)
    col = pass_df["pass_cross"]
    if str(col.dtype).lower() in {"bool", "boolean"}:
        return col.fillna(False).astype(bool)
    text = col.astype("string").str.strip().str.lower()
    numeric = pd.to_numeric(col, errors="coerce")
    return text.isin({"true", "t", "1", "yes", "y"}) | numeric.eq(1)


def _pass_longitudinal_cols(pass_df: pd.DataFrame) -> tuple[str | None, str | None]:
    start_long = _first_col(pass_df, ["location_x", "x", "location_y", "y"])
    end_long = _first_col(pass_df, ["pass_end_location_x", "pass_end_x", "end_x", "pass_end_location_y", "pass_end_y", "end_y"])
    return start_long, end_long


def _final_third_mask(pass_df: pd.DataFrame, is_home: bool) -> pd.Series:
    _, end_long_col = _pass_longitudinal_cols(pass_df)
    if end_long_col is None:
        return pd.Series(False, index=pass_df.index)
    end_long = pd.to_numeric(pass_df[end_long_col], errors="coerce")
    if end_long.dropna().empty:
        return pd.Series(False, index=pass_df.index)

    pitch_len = 120.0 if float(end_long.max()) > 101.0 else 100.0
    high_threshold = (2.0 / 3.0) * pitch_len
    low_threshold = (1.0 / 3.0) * pitch_len
    if is_home:
        return end_long >= high_threshold
    return end_long <= low_threshold


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
    if pass_df.empty:
        return pass_df
    pass_df["is_completed"] = _is_completed(pass_df)
    pass_df["pass_height_norm"] = _normalize_height(pass_df)
    pass_df["is_cross"] = _cross_mask(pass_df)
    return pass_df


def filter_passes(
    pass_df: pd.DataFrame,
    pass_height_filter: str = "All",
    cross_filter: str = "All",
    final_third_only: bool = False,
    is_home: bool = True,
) -> pd.DataFrame:
    work = pass_df.copy()
    if work.empty:
        return work

    if "pass_height_norm" not in work.columns:
        work["pass_height_norm"] = _normalize_height(work)
    if "is_cross" not in work.columns:
        work["is_cross"] = _cross_mask(work)

    height_lookup = {
        "Ground": "ground pass",
        "Low": "low pass",
        "High": "high pass",
    }
    target_height = height_lookup.get(pass_height_filter)
    if target_height is not None:
        work = work[work["pass_height_norm"] == target_height]

    if cross_filter == "Crosses Only":
        work = work[work["is_cross"]]
    elif cross_filter == "Exclude Crosses":
        work = work[~work["is_cross"]]

    if final_third_only:
        work = work[_final_third_mask(work, is_home=is_home)]
    return work


def compute_pass_stats(pass_df: pd.DataFrame, is_home: bool) -> dict[str, float]:
    if pass_df.empty:
        return {
            "total_passes": 0,
            "completed_passes": 0,
            "completion_pct": 0.0,
            "progressive_passes": 0,
            "final_third_passes": 0,
            "ground_passes": 0,
            "low_passes": 0,
            "high_passes": 0,
            "crosses": 0,
            "cross_completion_pct": 0.0,
        }

    work = pass_df.copy()
    if "pass_height_norm" not in work.columns:
        work["pass_height_norm"] = _normalize_height(work)
    if "is_cross" not in work.columns:
        work["is_cross"] = _cross_mask(work)

    start_long_col, end_long_col = _pass_longitudinal_cols(work)
    progressive = 0
    if start_long_col and end_long_col:
        start_long = pd.to_numeric(work[start_long_col], errors="coerce")
        end_long = pd.to_numeric(work[end_long_col], errors="coerce")
        delta = end_long - start_long if is_home else start_long - end_long
        progressive = int((delta >= 15.0).fillna(False).sum())

    total = int(len(work))
    completed = int(work["is_completed"].sum()) if "is_completed" in work.columns else 0
    completion_pct = (completed / total * 100.0) if total else 0.0
    final_third = int(_final_third_mask(work, is_home=is_home).sum())
    ground = int((work["pass_height_norm"] == "ground pass").sum())
    low = int((work["pass_height_norm"] == "low pass").sum())
    high = int((work["pass_height_norm"] == "high pass").sum())
    crosses = int(work["is_cross"].sum())
    cross_completed = int((work["is_cross"] & work["is_completed"]).sum())
    cross_completion_pct = (cross_completed / crosses * 100.0) if crosses else 0.0

    return {
        "total_passes": total,
        "completed_passes": completed,
        "completion_pct": completion_pct,
        "progressive_passes": progressive,
        "final_third_passes": final_third,
        "ground_passes": ground,
        "low_passes": low,
        "high_passes": high,
        "crosses": crosses,
        "cross_completion_pct": cross_completion_pct,
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
        hoverinfo="skip",
        showlegend=False,
    )


def render_pass_map(pass_df: pd.DataFrame, title: str, max_lines: int = 1100) -> go.Figure:
    fig = go.Figure()
    if pass_df.empty:
        fig.update_layout(
            title=title,
            paper_bgcolor="#0b1220",
            plot_bgcolor="#111a2b",
            font=dict(color="#e7edf7"),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=500,
            shapes=_pitch_shapes(),
            margin=dict(l=10, r=10, t=55, b=10),
        )
        return fig

    start_x_col = _first_col(pass_df, ["location_x", "x"])
    start_y_col = _first_col(pass_df, ["location_y", "y"])
    end_x_col = _first_col(pass_df, ["pass_end_location_x", "pass_end_x", "end_x"])
    end_y_col = _first_col(pass_df, ["pass_end_location_y", "pass_end_y", "end_y"])
    if not all([start_x_col, start_y_col, end_x_col, end_y_col]):
        return fig

    work = pass_df.copy()
    for col in [start_x_col, start_y_col, end_x_col, end_y_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[start_x_col, start_y_col, end_x_col, end_y_col])
    if work.empty:
        return fig
    if len(work) > max_lines:
        work = work.sample(n=max_lines, random_state=42)

    completed = work[work["is_completed"]]
    unsuccessful = work[~work["is_completed"]]
    if not completed.empty:
        fig.add_trace(_line_trace(completed, start_x_col, start_y_col, end_x_col, end_y_col, "#42d392", "Completed"))
        fig.add_trace(_endpoint_markers(completed, start_x_col, start_y_col, "#7de8b8", size=4.2, opacity=0.45))
        fig.add_trace(_endpoint_markers(completed, end_x_col, end_y_col, "#42d392", size=8.4, opacity=0.85))
    if not unsuccessful.empty:
        fig.add_trace(_line_trace(unsuccessful, start_x_col, start_y_col, end_x_col, end_y_col, "#f59e0b", "Unsuccessful"))
        fig.add_trace(_endpoint_markers(unsuccessful, start_x_col, start_y_col, "#f8c46d", size=4.2, opacity=0.45))
        fig.add_trace(_endpoint_markers(unsuccessful, end_x_col, end_y_col, "#f59e0b", size=8.4, opacity=0.85))

    fig.update_layout(
        title=title,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#111a2b",
        font=dict(color="#e7edf7"),
        margin=dict(l=10, r=10, t=55, b=10),
        height=500,
        shapes=_pitch_shapes(),
        legend=dict(orientation="h", yanchor="top", y=1.04, xanchor="left", x=0),
    )
    fig.update_xaxes(range=[-2, 122], visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(range=[-2, 82], visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
    return fig


def render_pass_comparison_panel(home_stats: dict[str, float], away_stats: dict[str, float], home_name: str, away_name: str) -> None:
    rows = [
        ("Total Passes", home_stats["total_passes"], away_stats["total_passes"], False),
        ("Completed Passes", home_stats["completed_passes"], away_stats["completed_passes"], False),
        ("Pass Completion %", home_stats["completion_pct"], away_stats["completion_pct"], True),
        ("Progressive Passes", home_stats["progressive_passes"], away_stats["progressive_passes"], False),
        ("Passes into Final Third", home_stats["final_third_passes"], away_stats["final_third_passes"], False),
        ("Ground Passes", home_stats["ground_passes"], away_stats["ground_passes"], False),
        ("Low Passes", home_stats["low_passes"], away_stats["low_passes"], False),
        ("High Passes", home_stats["high_passes"], away_stats["high_passes"], False),
        ("Crosses", home_stats["crosses"], away_stats["crosses"], False),
        ("Cross Completion %", home_stats["cross_completion_pct"], away_stats["cross_completion_pct"], True),
    ]
    html = ['<div class="match-stats-panel">']
    html.append(f'<div class="match-stats-context">{home_name} vs {away_name}</div>')
    for label, hv, av, is_pct in rows:
        left = f"{hv:.1f}%" if is_pct else str(int(hv))
        right = f"{av:.1f}%" if is_pct else str(int(av))
        html.append(
            "<div class='match-stats-row'>"
            "<div class='match-stats-values'>"
            f"<div class='home'>{left}</div>"
            f"<div class='label'>{label}</div>"
            f"<div class='away'>{right}</div>"
            "</div>"
            "</div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _team_subset(
    pass_df: pd.DataFrame,
    team_id: int,
    selected_team_id: int | None,
    selected_player_id: int | None,
) -> pd.DataFrame:
    work = pass_df.copy()
    if work.empty:
        return work
    if selected_team_id is not None and int(selected_team_id) != int(team_id):
        return work.iloc[0:0].copy()
    if "team_id" in work.columns:
        work = work[pd.to_numeric(work["team_id"], errors="coerce") == int(team_id)]
    if selected_player_id is not None and "player_id" in work.columns:
        work = work[pd.to_numeric(work["player_id"], errors="coerce") == int(selected_player_id)]
    return work


def _debug_pass_validation(pass_df: pd.DataFrame) -> None:
    if pass_df.empty:
        st.caption("Pass debug: no pass events in current context.")
        return
    height_values = (
        pass_df["pass_height_name"].astype("string").dropna().str.strip().drop_duplicates().sort_values().tolist()
        if "pass_height_name" in pass_df.columns
        else []
    )
    if "is_cross" in pass_df.columns:
        cross_true = int(pass_df["is_cross"].sum())
        cross_false = int((~pass_df["is_cross"]).sum())
    else:
        cross_true = 0
        cross_false = int(len(pass_df))
    st.caption(
        "Pass debug | "
        f"pass_height_name values: {height_values if height_values else 'unavailable'} | "
        f"crosses: {cross_true}, non-crosses: {cross_false}"
    )


def render_passes_section(
    match_id: int,
    home_team_id: int | None,
    away_team_id: int | None,
    home_team_name: str,
    away_team_name: str,
    selected_team_id: int | None,
    selected_player_id: int | None,
    events: pd.DataFrame,
) -> None:
    pass_df = get_pass_events(match_id=match_id, team_id=None, player_id=None, events=events)
    if home_team_id is None or away_team_id is None:
        st.info("Home/Away team metadata is unavailable for this match.")
        return

    home_passes = _team_subset(pass_df, team_id=int(home_team_id), selected_team_id=selected_team_id, selected_player_id=selected_player_id)
    away_passes = _team_subset(pass_df, team_id=int(away_team_id), selected_team_id=selected_team_id, selected_player_id=selected_player_id)

    f1, f2 = st.columns(2)
    with f1:
        height_filter = st.selectbox(
            "Pass Height Filter",
            options=["All", "Ground", "Low", "High"],
            index=0,
            key="passes_height_filter",
        )
    with f2:
        cross_filter = st.selectbox(
            "Cross Filter",
            options=["All", "Crosses Only", "Exclude Crosses"],
            index=0,
            key="passes_cross_filter",
        )

    filtered_home = filter_passes(home_passes, pass_height_filter=height_filter, cross_filter=cross_filter, final_third_only=False, is_home=True)
    filtered_away = filter_passes(away_passes, pass_height_filter=height_filter, cross_filter=cross_filter, final_third_only=False, is_home=False)

    st.markdown('<div class="section-title">Pass Comparison Stats</div>', unsafe_allow_html=True)
    home_stats = compute_pass_stats(filtered_home, is_home=True)
    away_stats = compute_pass_stats(filtered_away, is_home=False)
    render_pass_comparison_panel(home_stats, away_stats, home_team_name, away_team_name)

    st.markdown('<div class="section-title">Home vs Away Pass Maps</div>', unsafe_allow_html=True)
    c_home, c_away = st.columns(2)
    with c_home:
        st.plotly_chart(render_pass_map(filtered_home, title="Home Passes"), use_container_width=True)
    with c_away:
        st.plotly_chart(render_pass_map(filtered_away, title="Away Passes"), use_container_width=True)

    st.markdown('<div class="section-title">Passes into Final Third</div>', unsafe_allow_html=True)
    ft_home, ft_away = st.columns(2)
    with ft_home:
        final_home = filter_passes(
            home_passes,
            pass_height_filter=height_filter,
            cross_filter=cross_filter,
            final_third_only=True,
            is_home=True,
        )
        st.plotly_chart(render_pass_map(final_home, title="Home Final Third Passes"), use_container_width=True)
    with ft_away:
        final_away = filter_passes(
            away_passes,
            pass_height_filter=height_filter,
            cross_filter=cross_filter,
            final_third_only=True,
            is_home=False,
        )
        st.plotly_chart(render_pass_map(final_away, title="Away Final Third Passes"), use_container_width=True)

    if bool(st.session_state.get("debug_passes")):
        _debug_pass_validation(pass_df)
