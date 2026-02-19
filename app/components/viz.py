from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def _coalesce_column(df: pd.DataFrame, candidates: list[str], default: object = pd.NA) -> pd.Series:
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        return pd.Series(default, index=df.index, dtype="object")
    result = df[existing[0]]
    for col in existing[1:]:
        result = result.combine_first(df[col])
    return result


def _pitch_shapes(line_color: str = "#6c8f78") -> list[dict]:
    shapes: list[dict] = []
    outer = dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color=line_color, width=2))
    halfway = dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color=line_color, width=2))
    center_circle = dict(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color=line_color, width=2))
    left_box = dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color=line_color, width=2))
    right_box = dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color=line_color, width=2))
    left_six = dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color=line_color, width=2))
    right_six = dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color=line_color, width=2))
    left_goal = dict(type="rect", x0=-2, y0=36, x1=0, y1=44, line=dict(color=line_color, width=2))
    right_goal = dict(type="rect", x0=120, y0=36, x1=122, y1=44, line=dict(color=line_color, width=2))
    left_spot = dict(type="circle", x0=11.5, y0=39.5, x1=12.5, y1=40.5, fillcolor=line_color, line=dict(color=line_color))
    right_spot = dict(type="circle", x0=107.5, y0=39.5, x1=108.5, y1=40.5, fillcolor=line_color, line=dict(color=line_color))
    shapes.extend(
        [outer, halfway, center_circle, left_box, right_box, left_six, right_six, left_goal, right_goal, left_spot, right_spot]
    )
    return shapes


def draw_pitch_figure(shots_df: pd.DataFrame, title: str, subtitle: str | None = None) -> go.Figure:
    shots = shots_df.copy()
    shots["x"] = pd.to_numeric(_coalesce_column(shots, ["x", "location_x"]), errors="coerce")
    shots["y"] = pd.to_numeric(_coalesce_column(shots, ["y", "location_y"]), errors="coerce")
    shots["xg"] = pd.to_numeric(_coalesce_column(shots, ["xg", "shot_statsbomb_xg"]), errors="coerce")
    shots["shot_outcome"] = _coalesce_column(shots, ["shot_outcome_name", "shot_outcome"], default="Unknown")
    shots["shot_type"] = _coalesce_column(shots, ["shot_type_name", "shot_type"])
    shots["body_part"] = _coalesce_column(
        shots,
        ["body_part_name", "shot_body_part_name", "body_part", "shot_body_part"],
    )

    shots = shots.dropna(subset=["x", "y"])
    if shots.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0f1a2d",
            font=dict(color="#e7edf7"),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    colors = {
        "Goal": "#42d392",
        "Saved": "#6aa6ff",
        "Off T": "#f6ad55",
        "Blocked": "#b794f4",
        "Post": "#f56565",
        "Wayward": "#f6ad55",
        "Unknown": "#9aa4b2",
    }

    shots["shot_outcome"] = shots["shot_outcome"].fillna("Unknown").astype(str)
    shots["marker_size"] = 7 + (shots["xg"].fillna(0) * 26)
    shots["minute_text"] = _coalesce_column(shots, ["minute"], default="-").astype(str)
    shots["player_text"] = _coalesce_column(shots, ["player_name"], default="-").astype(str)
    shots["xg_text"] = shots["xg"].fillna(0).round(2).astype(str)
    shots["shot_type_text"] = shots["shot_type"].fillna("-").astype(str)
    shots["body_part_text"] = shots["body_part"].fillna("-").astype(str)

    fig = go.Figure()
    for outcome in sorted(shots["shot_outcome"].unique()):
        chunk = shots[shots["shot_outcome"] == outcome]
        fig.add_trace(
            go.Scatter(
                x=chunk["x"],
                y=chunk["y"],
                mode="markers",
                name=outcome,
                marker=dict(
                    size=chunk["marker_size"],
                    color=colors.get(outcome, "#9aa4b2"),
                    opacity=0.9,
                    line=dict(color="#0b1220", width=1),
                ),
                customdata=chunk[["minute_text", "player_text", "xg_text", "shot_outcome", "shot_type_text", "body_part_text"]],
                hovertemplate=(
                    "Minute: %{customdata[0]}<br>"
                    "Player: %{customdata[1]}<br>"
                    "xG: %{customdata[2]}<br>"
                    "Outcome: %{customdata[3]}<br>"
                    "Shot type: %{customdata[4]}<br>"
                    "Body part: %{customdata[5]}<extra></extra>"
                ),
            )
        )

    if subtitle:
        full_title = f"{title}<br><sup>{subtitle}</sup>"
    else:
        full_title = title

    fig.update_layout(
        title=full_title,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0f1a2d",
        font=dict(color="#e7edf7"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.14,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=12, r=12, t=110, b=96),
        height=560,
        shapes=_pitch_shapes(),
    )
    fig.update_xaxes(range=[-2, 122], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[82, -2], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1)
    return fig
