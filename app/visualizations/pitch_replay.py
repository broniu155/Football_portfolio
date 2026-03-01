from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

PITCH_BG = "#0f1a2d"
PAPER_BG = "#0b1220"
LINE_COLOR = "#6c8f78"


def _pitch_shapes(line_color: str = LINE_COLOR) -> list[dict[str, Any]]:
    shapes: list[dict[str, Any]] = [
        dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color=line_color, width=2)),
        dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color=line_color, width=2)),
        dict(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color=line_color, width=2)),
        dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color=line_color, width=2)),
        dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color=line_color, width=2)),
        dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color=line_color, width=2)),
        dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color=line_color, width=2)),
        dict(type="rect", x0=-2, y0=36, x1=0, y1=44, line=dict(color=line_color, width=2)),
        dict(type="rect", x0=120, y0=36, x1=122, y1=44, line=dict(color=line_color, width=2)),
        dict(type="circle", x0=11.5, y0=39.5, x1=12.5, y1=40.5, fillcolor=line_color, line=dict(color=line_color)),
        dict(type="circle", x0=107.5, y0=39.5, x1=108.5, y1=40.5, fillcolor=line_color, line=dict(color=line_color)),
    ]
    for shape in shapes:
        shape["layer"] = "below"
    return shapes


def build_pitch_base_figure() -> go.Figure:
    """Create a 120x80 football pitch base figure for replay layers."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PITCH_BG,
        font=dict(color="#e7edf7"),
        margin=dict(l=10, r=10, t=36, b=20),
        height=620,
        shapes=_pitch_shapes(),
    )
    fig.update_xaxes(range=[-2, 122], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[82, -2], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1)
    return fig


def _clock_label(seconds_abs: float) -> str:
    seconds_abs = max(0, int(round(seconds_abs)))
    mm = seconds_abs // 60
    ss = seconds_abs % 60
    return f"{mm:02d}:{ss:02d}"


def _segment_for_time(segments: list[dict[str, Any]], t_value: float, start_idx: int) -> tuple[dict[str, Any] | None, int]:
    idx = max(0, start_idx)
    while idx < len(segments) - 1 and float(segments[idx]["t1"]) < t_value:
        idx += 1
    seg = segments[idx] if idx < len(segments) else None
    if seg is not None and float(seg["t0"]) <= t_value <= float(seg["t1"]):
        return seg, idx
    return None, idx


def _lerp_xy(p0: tuple[float, float] | None, p1: tuple[float, float] | None, alpha: float) -> tuple[float, float] | None:
    if p0 is None and p1 is None:
        return None
    if p0 is None:
        return p1
    if p1 is None:
        return p0
    return (p0[0] + (p1[0] - p0[0]) * alpha, p0[1] + (p1[1] - p0[1]) * alpha)


def _build_frame_data(
    seg: dict[str, Any] | None,
    t_value: float,
    show_visible_area: bool,
    show_paths: bool,
) -> list[go.Scatter]:
    visible_x: list[float] = []
    visible_y: list[float] = []
    teammates_x: list[float] = []
    teammates_y: list[float] = []
    opponents_x: list[float] = []
    opponents_y: list[float] = []
    actor_x: list[float] = []
    actor_y: list[float] = []
    ball_x: list[float] = []
    ball_y: list[float] = []
    path_x: list[float] = []
    path_y: list[float] = []

    if seg is not None:
        t0 = float(seg.get("t0", t_value))
        t1 = max(t0 + 0.0001, float(seg.get("t1", t_value)))
        alpha = min(1.0, max(0.0, (t_value - t0) / (t1 - t0)))

        ball0 = seg.get("ball0")
        ball1 = seg.get("ball1")
        ball_xy = _lerp_xy(ball0, ball1, alpha)
        if ball_xy is not None:
            ball_x = [float(ball_xy[0])]
            ball_y = [float(ball_xy[1])]
        if show_paths and ball0 is not None and ball1 is not None:
            path_x = [float(ball0[0]), float(ball1[0])]
            path_y = [float(ball0[1]), float(ball1[1])]

        actor_xy = seg.get("actor_xy")
        if actor_xy is not None:
            actor_x = [float(actor_xy[0])]
            actor_y = [float(actor_xy[1])]

        freeze = seg.get("freeze_frame", [])
        if isinstance(freeze, list):
            for p in freeze:
                x = p.get("location_x")
                y = p.get("location_y")
                if x is None or y is None:
                    continue
                if bool(p.get("teammate")):
                    teammates_x.append(float(x))
                    teammates_y.append(float(y))
                else:
                    opponents_x.append(float(x))
                    opponents_y.append(float(y))

        if show_visible_area:
            poly = seg.get("visible_area", [])
            if isinstance(poly, list) and len(poly) >= 3:
                visible_x = [float(pt[0]) for pt in poly]
                visible_y = [float(pt[1]) for pt in poly]

    return [
        go.Scatter(
            x=visible_x,
            y=visible_y,
            mode="lines",
            fill="toself",
            fillcolor="rgba(66, 211, 146, 0.16)",
            line=dict(color="rgba(66, 211, 146, 0.44)", width=1),
            name="Visible area",
            hoverinfo="skip",
            showlegend=True,
        ),
        go.Scatter(
            x=teammates_x,
            y=teammates_y,
            mode="markers",
            marker=dict(size=8, color="#6aa6ff", line=dict(color="#e7edf7", width=0.6)),
            name="360 teammates",
            hoverinfo="skip",
            showlegend=True,
        ),
        go.Scatter(
            x=opponents_x,
            y=opponents_y,
            mode="markers",
            marker=dict(size=8, color="#f6ad55", line=dict(color="#e7edf7", width=0.6)),
            name="360 opponents",
            hoverinfo="skip",
            showlegend=True,
        ),
        go.Scatter(
            x=actor_x,
            y=actor_y,
            mode="markers",
            marker=dict(size=11, color="#42d392", line=dict(color="#0b1220", width=1.0)),
            name="Actor",
            hoverinfo="skip",
            showlegend=True,
        ),
        go.Scatter(
            x=ball_x,
            y=ball_y,
            mode="markers",
            marker=dict(size=10, color="#ffffff", line=dict(color="#111", width=1.0)),
            name="Ball",
            hoverinfo="skip",
            showlegend=True,
        ),
        go.Scatter(
            x=path_x,
            y=path_y,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.55)", width=2, dash="dot"),
            name="Event path",
            hoverinfo="skip",
            showlegend=True,
        ),
    ]


def build_replay_figure(
    segments: list[dict[str, Any]],
    fps: int,
    max_frames: int,
    show_visible_area: bool,
    show_paths: bool,
) -> go.Figure:
    """Build animated replay figure from normalized segments."""
    fig = build_pitch_base_figure()
    if not segments:
        fig.add_annotation(
            x=60,
            y=40,
            text="No replay data available for current filters.",
            showarrow=False,
            font=dict(size=14, color="#d0d8e6"),
        )
        return fig

    sorted_segments = sorted(segments, key=lambda row: (float(row["t0"]), float(row["t1"])))
    t_min = float(sorted_segments[0]["t0"])
    t_max = float(max(row["t1"] for row in sorted_segments))
    fps_safe = max(5, min(25, int(fps)))
    max_frames_safe = max(1, int(max_frames))

    timeline: list[float] = []
    if t_max <= t_min:
        timeline = [t_min]
    else:
        step = 1.0 / float(fps_safe)
        current = t_min
        while current <= t_max and len(timeline) < max_frames_safe:
            timeline.append(current)
            current += step
        if len(timeline) >= max_frames_safe:
            # Keep first/last coverage when frame cap is reached.
            interval = (t_max - t_min) / float(max_frames_safe - 1) if max_frames_safe > 1 else 0.0
            timeline = [t_min + interval * idx for idx in range(max_frames_safe)]
        elif timeline[-1] < t_max:
            timeline.append(t_max)

    first_data = _build_frame_data(sorted_segments[0], timeline[0], show_visible_area=show_visible_area, show_paths=show_paths)
    for trace in first_data:
        fig.add_trace(trace)

    slider_steps: list[dict[str, Any]] = []
    frames: list[go.Frame] = []
    seek_idx = 0
    for idx, t_value in enumerate(timeline):
        seg, seek_idx = _segment_for_time(sorted_segments, t_value=t_value, start_idx=seek_idx)
        frame_data = _build_frame_data(seg, t_value=t_value, show_visible_area=show_visible_area, show_paths=show_paths)
        frame_name = f"f{idx}"
        frames.append(go.Frame(name=frame_name, data=frame_data))
        slider_steps.append(
            {
                "args": [[frame_name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                "label": _clock_label(t_value),
                "method": "animate",
            }
        )

    frame_duration_ms = int(round(1000 / float(fps_safe)))
    fig.frames = frames
    fig.update_layout(
        title="Live Pitch Replay (Beta)",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.07,
                "showactive": True,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.0,
                "y": 1.01,
                "len": 1.0,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"t": 4, "b": 0},
                "currentvalue": {"prefix": "Clock ", "font": {"size": 12}},
                "steps": slider_steps,
            }
        ],
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=90, b=50),
    )
    return fig
