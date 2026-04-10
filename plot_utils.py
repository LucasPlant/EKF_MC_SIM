"""
Plotting utilities for rigid-body SE(2) trajectories.

All plots use Plotly and return go.Figure objects so callers can
further customise or write them to disk.

Public API
----------
plot_trajectory    — static path with body-axes arrows at sampled poses
animate_trajectory — frame-by-frame animation
plot_mc_paths      — overlaid paths for all MC trials
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

from sim import RigidBodyTrajectory, SE2


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_BG       = "#0d1117"
_GRID     = "#21262d"
_ACCENT   = "#58a6ff"
_WARM     = "#f78166"
_GREEN    = "#3fb950"
_PURPLE   = "#bc8cff"
_ORANGE   = "#e3b341"
_FG       = "#c9d1d9"

_MC_COLOURS = [
    "#58a6ff", "#3fb950", "#f78166", "#bc8cff",
    "#e3b341", "#79c0ff", "#56d364", "#ffa198",
]

_AXIS_LAYOUT = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(color=_FG, family="monospace"),
    xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID, color=_FG),
    legend=dict(bgcolor="#161b22", bordercolor=_GRID, borderwidth=1),
)

_YAXIS_BASE = dict(gridcolor=_GRID, zerolinecolor=_GRID, color=_FG)
_YAXIS_EQUAL = dict(**_YAXIS_BASE, scaleanchor="x", scaleratio=1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _body_axes(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    scale: float = 0.4,
) -> tuple[list, list]:
    """
    Build Plotly arrow traces for body x-axis (forward, red) and
    y-axis (left, green) at each pose.

    Returns two lists of go.Scatter traces (one per axis).
    """
    c, s = np.cos(theta), np.sin(theta)

    # x-axis (forward) — warm red
    x_arrow_x = np.stack([x, x + scale * c, np.full_like(x, None)], axis=-1).ravel()
    x_arrow_y = np.stack([y, y + scale * s, np.full_like(y, None)], axis=-1).ravel()

    # y-axis (left) — green
    y_arrow_x = np.stack([x, x - scale * s, np.full_like(x, None)], axis=-1).ravel()
    y_arrow_y = np.stack([y, y + scale * c, np.full_like(y, None)], axis=-1).ravel()

    trace_x = go.Scatter(
        x=x_arrow_x, y=x_arrow_y,
        mode="lines",
        line=dict(color=_WARM, width=2),
        name="body x-axis",
        showlegend=False,
    )
    trace_y = go.Scatter(
        x=y_arrow_x, y=y_arrow_y,
        mode="lines",
        line=dict(color=_GREEN, width=2),
        name="body y-axis",
        showlegend=False,
    )
    return [trace_x, trace_y]


def _extract_path(traj: RigidBodyTrajectory, trial: int = 0):
    """Return (xs, ys, thetas) for one MC trial across all timesteps."""
    xs     = np.array([p.x[trial]     for p in traj.poses])
    ys     = np.array([p.y[trial]     for p in traj.poses])
    thetas = np.array([p.theta[trial] for p in traj.poses])
    return xs, ys, thetas


# ---------------------------------------------------------------------------
# 1. Static trajectory with body axes at sampled poses
# ---------------------------------------------------------------------------

def plot_trajectory(
    traj: RigidBodyTrajectory,
    trial: int = 0,
    n_poses: int = 12,
    axis_scale: float = 0.4,
    title: str = "Trajectory",
) -> go.Figure:
    """
    Static plot of one trial's path with body-frame axes and pose markers
    drawn at equally-spaced intervals.

    Parameters
    ----------
    traj       : RigidBodyTrajectory
    trial      : which MC trial to plot (default 0)
    n_poses    : number of poses at which to draw body axes and markers
    axis_scale : length of axis arrows in world units
    title      : figure title
    """
    xs, ys, thetas = _extract_path(traj, trial)
    idx = np.linspace(0, len(xs) - 1, n_poses, dtype=int)

    fig = go.Figure()

    # Path
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(color=_ACCENT, width=2),
        name="path",
    ))

    # Start / end markers
    fig.add_trace(go.Scatter(
        x=[xs[0]], y=[ys[0]],
        mode="markers",
        marker=dict(color=_GREEN, size=12, symbol="circle"),
        name="start",
    ))
    fig.add_trace(go.Scatter(
        x=[xs[-1]], y=[ys[-1]],
        mode="markers",
        marker=dict(color=_WARM, size=12, symbol="square"),
        name="end",
    ))

    # Body axes and pose markers at sampled steps
    for ax_trace in _body_axes(xs[idx], ys[idx], thetas[idx], axis_scale):
        fig.add_trace(ax_trace)
    fig.add_trace(go.Scatter(
        x=xs[idx], y=ys[idx],
        mode="markers",
        marker=dict(color=_ORANGE, size=8, symbol="circle",
                    line=dict(color=_FG, width=1)),
        name="sampled poses",
    ))

    # Legend-only proxies for axis colours
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
        line=dict(color=_WARM, width=2), name="x-axis (fwd)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
        line=dict(color=_GREEN, width=2), name="y-axis (left)"))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=_FG)),
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        yaxis=_YAXIS_EQUAL,
        **_AXIS_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Animation
# ---------------------------------------------------------------------------

def animate_trajectory(
    traj: RigidBodyTrajectory,
    trial: int = 0,
    step: int = 5,
    axis_scale: float = 0.5,
    title: str = "Trajectory Animation",
    frame_duration_ms: int = 40,
) -> go.Figure:
    """
    Animated trajectory showing the body moving through time.

    Parameters
    ----------
    traj              : RigidBodyTrajectory
    trial             : MC trial index
    step              : subsample timesteps for smoother animation
    axis_scale        : body-axis arrow length
    frame_duration_ms : milliseconds per frame
    """
    xs, ys, thetas = _extract_path(traj, trial)
    indices = np.arange(0, len(xs), step)

    # Axis extents with padding
    pad = 0.5
    xmin, xmax = xs.min() - pad, xs.max() + pad
    ymin, ymax = ys.min() - pad, ys.max() + pad

    frames = []
    for i in indices:
        xi, yi, ti = xs[i], ys[i], thetas[i]
        c, s        = np.cos(ti), np.sin(ti)

        frame_data = [
            # Full path (faded)
            go.Scatter(x=xs[:i+1], y=ys[:i+1],
                       mode="lines",
                       line=dict(color=_ACCENT, width=1.5),
                       name="path"),
            # Body position
            go.Scatter(x=[xi], y=[yi],
                       mode="markers",
                       marker=dict(color=_ORANGE, size=14, symbol="circle"),
                       name="body"),
            # x-axis
            go.Scatter(
                x=[xi, xi + axis_scale * c, None],
                y=[yi, yi + axis_scale * s, None],
                mode="lines", line=dict(color=_WARM, width=3),
                name="x-axis"),
            # y-axis
            go.Scatter(
                x=[xi, xi - axis_scale * s, None],
                y=[yi, yi + axis_scale * c, None],
                mode="lines", line=dict(color=_GREEN, width=3),
                name="y-axis"),
        ]
        frames.append(go.Frame(data=frame_data, name=str(i)))

    # Initial frame data
    i0 = indices[0]
    xi, yi, ti = xs[i0], ys[i0], thetas[i0]
    c, s = np.cos(ti), np.sin(ti)
    init_data = [
        go.Scatter(x=xs[:1], y=ys[:1], mode="lines",
                   line=dict(color=_ACCENT, width=1.5), name="path"),
        go.Scatter(x=[xi], y=[yi], mode="markers",
                   marker=dict(color=_ORANGE, size=14), name="body"),
        go.Scatter(x=[xi, xi + axis_scale*c, None],
                   y=[yi, yi + axis_scale*s, None],
                   mode="lines", line=dict(color=_WARM, width=3), name="x-axis"),
        go.Scatter(x=[xi, xi - axis_scale*s, None],
                   y=[yi, yi + axis_scale*c, None],
                   mode="lines", line=dict(color=_GREEN, width=3), name="y-axis"),
    ]

    fig = go.Figure(data=init_data, frames=frames)
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=_FG)),
        xaxis=dict(range=[xmin, xmax], gridcolor=_GRID, color=_FG),
        yaxis=dict(range=[ymin, ymax], **_YAXIS_EQUAL),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_FG, family="monospace"),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0.02, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶  Play",
                     method="animate",
                     args=[None, dict(
                         frame=dict(duration=frame_duration_ms, redraw=True),
                         fromcurrent=True,
                         transition=dict(duration=0),
                     )]),
                dict(label="⏸  Pause",
                     method="animate",
                     args=[[None], dict(
                         frame=dict(duration=0, redraw=False),
                         mode="immediate",
                     )]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate",
                        args=[[f.name], dict(mode="immediate",
                                             frame=dict(duration=0, redraw=True),
                                             transition=dict(duration=0))],
                        label=str(i))
                   for i, f in zip(indices, frames)],
            transition=dict(duration=0),
            x=0.05, y=0.0, len=0.9,
            currentvalue=dict(prefix="step: ", font=dict(color=_FG)),
            font=dict(color=_FG),
            bgcolor=_GRID,
            bordercolor=_GRID,
        )],
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Monte Carlo path overlay
# ---------------------------------------------------------------------------

def plot_mc_paths(
    traj: RigidBodyTrajectory,
    title: str = "Monte Carlo Trials",
    max_trials: Optional[int] = None,
    alpha: float = 0.5,
) -> go.Figure:
    """
    Overlay path lines for every (or up to max_trials) MC trials.

    Parameters
    ----------
    traj       : RigidBodyTrajectory with n > 1
    max_trials : cap number of plotted trials (None = all)
    alpha      : line opacity
    """
    n_plot = traj.n if max_trials is None else min(max_trials, traj.n)
    fig    = go.Figure()

    for i in range(n_plot):
        xs, ys, _ = _extract_path(traj, i)
        colour    = _MC_COLOURS[i % len(_MC_COLOURS)]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=colour, width=1.2),
            opacity=alpha,
            name=f"trial {i}",
            showlegend=(n_plot <= 12),
        ))

    # Mean path
    mean_xs = np.array([[p.x[i] for p in traj.poses] for i in range(traj.n)]).mean(0)
    mean_ys = np.array([[p.y[i] for p in traj.poses] for i in range(traj.n)]).mean(0)
    fig.add_trace(go.Scatter(
        x=mean_xs, y=mean_ys,
        mode="lines",
        line=dict(color="white", width=2.5, dash="dash"),
        name="mean path",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=_FG)),
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        yaxis=_YAXIS_EQUAL,
        **_AXIS_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. IMU measurement time series  (ax_B, ay_B, ω_B)
# ---------------------------------------------------------------------------

def plot_imu_measurements(
    traj: RigidBodyTrajectory,
    title: str = "IMU Measurements",
    alpha: float = 0.25,
) -> go.Figure:
    """
    Three stacked subplots — body-frame accel x, accel y, and angular rate —
    one line per MC trial with a dotted mean line and dotted ±1σ bounds.

    Parameters
    ----------
    traj  : RigidBodyTrajectory
    title : figure title
    alpha : opacity for individual trial lines
    """
    times = traj.timestamps
    ax_B  = traj.accel_meas_B[:, :, 0]    # (n, nt)
    ay_B  = traj.accel_meas_B[:, :, 1]    # (n, nt)
    omega = traj.gyro_meas_B               # (n, nt)

    panels = [
        (ax_B,  "a_x [m/s²]",  "Accel x — body frame"),
        (ay_B,  "a_y [m/s²]",  "Accel y — body frame"),
        (omega, "ω [rad/s]",   "Angular rate — body frame"),
    ]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=[p[2] for p in panels],
    )

    _ax_style = dict(gridcolor=_GRID, zerolinecolor=_GRID, color=_FG)

    for row, (data, ylabel, _) in enumerate(panels, start=1):
        mean_  = data.mean(axis=0)   # (nt,)
        sigma_ = data.std(axis=0)    # (nt,)

        # Individual trials
        for i in range(traj.n):
            colour = _MC_COLOURS[i % len(_MC_COLOURS)]
            fig.add_trace(go.Scatter(
                x=times, y=data[i],
                mode="lines",
                line=dict(color=colour, width=0.8),
                opacity=alpha,
                name=f"trial {i}",
                showlegend=(row == 1),
                legendgroup=f"trial_{i}",
            ), row=row, col=1)

        # ±1σ filled region
        fig.add_trace(go.Scatter(
            x=np.concatenate([times, times[::-1]]),
            y=np.concatenate([mean_ + sigma_, (mean_ - sigma_)[::-1]]),
            fill="toself",
            fillcolor="rgba(200,200,200,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name="±1σ region",
            showlegend=(row == 1),
            legendgroup="sigma_fill",
        ), row=row, col=1)

        # ±1σ dotted boundary lines
        for sign, sname in [(1, "+1σ"), (-1, "−1σ")]:
            fig.add_trace(go.Scatter(
                x=times, y=mean_ + sign * sigma_,
                mode="lines",
                line=dict(color=_FG, width=1.0, dash="dot"),
                name=sname,
                showlegend=(row == 1),
                legendgroup=sname,
            ), row=row, col=1)

        # Mean dotted line
        fig.add_trace(go.Scatter(
            x=times, y=mean_,
            mode="lines",
            line=dict(color="white", width=2.0, dash="dot"),
            name="mean",
            showlegend=(row == 1),
            legendgroup="mean",
        ), row=row, col=1)

        fig.update_yaxes(title_text=ylabel, row=row, col=1, **_ax_style)
        fig.update_xaxes(row=row, col=1, **_ax_style)

    fig.update_xaxes(title_text="time [s]", row=3, col=1)

    for ann in fig.layout.annotations:
        ann.font.color = _FG
        ann.font.size  = 12

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=_FG)),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_FG, family="monospace"),
        legend=dict(bgcolor="#161b22", bordercolor=_GRID, borderwidth=1),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. MC trajectories with mean path and ±1σ spatial tube
# ---------------------------------------------------------------------------

def plot_trajectory_with_bounds(
    traj: RigidBodyTrajectory,
    title: str = "MC Trajectories — Mean & ±1σ Bounds",
    alpha: float = 0.35,
) -> go.Figure:
    """
    Parametric 2D trajectory for every MC trial, overlaid with a dotted mean
    path and dotted ±1σ boundary curves.

    The ±1σ tube is computed as sqrt(std_x² + std_y²) offset perpendicular
    to the mean path direction at each timestep.

    Parameters
    ----------
    traj  : RigidBodyTrajectory with n > 1
    title : figure title
    alpha : opacity for individual trial lines
    """
    n = traj.n

    # Position arrays: (n, nt)
    xs = np.array([[p.x[i] for p in traj.poses] for i in range(n)])
    ys = np.array([[p.y[i] for p in traj.poses] for i in range(n)])

    mean_x = xs.mean(axis=0)
    mean_y = ys.mean(axis=0)
    std_x  = xs.std(axis=0)
    std_y  = ys.std(axis=0)

    # Unit normal perpendicular to the mean path (left-hand side)
    dx  = np.gradient(mean_x)
    dy  = np.gradient(mean_y)
    mag = np.hypot(dx, dy) + 1e-12
    nx  = -dy / mag
    ny  =  dx / mag

    # Scalar spread → tube half-width
    sigma = np.sqrt(std_x**2 + std_y**2)

    upper_x = mean_x + sigma * nx
    upper_y = mean_y + sigma * ny
    lower_x = mean_x - sigma * nx
    lower_y = mean_y - sigma * ny

    fig = go.Figure()

    # Individual MC trial paths
    for i in range(n):
        colour = _MC_COLOURS[i % len(_MC_COLOURS)]
        fig.add_trace(go.Scatter(
            x=xs[i], y=ys[i],
            mode="lines",
            line=dict(color=colour, width=1.0),
            opacity=alpha,
            name=f"trial {i}",
            showlegend=(n <= 12),
        ))

    # ±1σ filled tube
    fig.add_trace(go.Scatter(
        x=np.concatenate([upper_x, lower_x[::-1]]),
        y=np.concatenate([upper_y, lower_y[::-1]]),
        fill="toself",
        fillcolor="rgba(200,200,200,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1σ region",
    ))

    # ±1σ dotted boundary lines
    for bx, by, label in [(upper_x, upper_y, "+1σ"), (lower_x, lower_y, "−1σ")]:
        fig.add_trace(go.Scatter(
            x=bx, y=by,
            mode="lines",
            line=dict(color=_FG, width=1.2, dash="dot"),
            name=label,
        ))

    # Mean path dotted
    fig.add_trace(go.Scatter(
        x=mean_x, y=mean_y,
        mode="lines",
        line=dict(color="white", width=2.5, dash="dot"),
        name="mean",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=_FG)),
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        yaxis=_YAXIS_EQUAL,
        **_AXIS_LAYOUT,
    )
    return fig
