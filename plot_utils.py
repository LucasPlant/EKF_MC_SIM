"""
Plotting utilities for rigid-body SE(2) trajectories.

All plots use Plotly and return go.Figure objects so callers can
further customise or write them to disk.

Public API
----------
plot_trajectory    — static path with body-axes arrows at sampled poses
animate_trajectory — frame-by-frame animation
plot_mc_paths      — overlaid paths for all MC trials
plot_imu_measurements     — IMU time-series subplots
plot_trajectory_with_bounds — MC paths with ±1σ spatial tube
confidence_ellipse — closed (x, y) curve for a 2-D Gaussian
animate_estimate   — animated EKF estimate for a single MC trial
plot_mc_estimates  — all MC trial EKF estimates overlaid with ground truth
plot_ekf_states    — time-series of all 5 KF states vs GT and measurements
plot_mc_mse        — trace(P) over time for every MC trial
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

# Like _AXIS_LAYOUT but without the xaxis key — safe to use alongside explicit xaxis=
_LAYOUT_BASE = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(color=_FG, family="monospace"),
    legend=dict(bgcolor="#161b22", bordercolor=_GRID, borderwidth=1),
)

_YAXIS_BASE  = dict(gridcolor=_GRID, zerolinecolor=_GRID, color=_FG)
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
    Three stacked subplots — body-frame accel x, accel y, and angular rate.

    One line per MC trial.  The ground truth (trial 0, noise-free) is shown
    as a solid orange line.  The ±1σ bounds are the theoretical noise standard
    deviation from noise_cfg.imu_cov (constant band around ground truth).

    Parameters
    ----------
    traj  : RigidBodyTrajectory
    title : figure title
    alpha : opacity for individual trial lines
    """
    times = traj.timestamps
    nt    = traj.nt

    # --- Ground truth body-frame signals from trial 0 ---
    # Rotate world-frame acceleration into body frame using trial-0 heading
    theta_gt = np.array([traj.poses[k].theta[0] for k in range(nt)])  # (nt,)
    c, s     = np.cos(theta_gt), np.sin(theta_gt)
    ax_W_gt  = traj.acceleration_W[0, :, 0]
    ay_W_gt  = traj.acceleration_W[0, :, 1]
    ax_B_gt  =  c * ax_W_gt + s * ay_W_gt
    ay_B_gt  = -s * ax_W_gt + c * ay_W_gt
    om_gt    = traj.angular_velocity[0, :]                             # (nt,)

    # --- Theoretical ±1σ from noise_cfg.imu_cov ---
    sigma_ax = np.sqrt(traj.noise_cfg.imu_cov[0, 0])   # scalar
    sigma_ay = np.sqrt(traj.noise_cfg.imu_cov[1, 1])
    sigma_om = np.sqrt(traj.noise_cfg.imu_cov[2, 2])

    # Measurement arrays
    ax_B  = traj.accel_meas_B[:, :, 0]    # (n, nt)
    ay_B  = traj.accel_meas_B[:, :, 1]
    omega = traj.gyro_meas_B

    panels = [
        (ax_B,  ax_B_gt, sigma_ax, "a_x [m/s²]", "Accel x — body frame"),
        (ay_B,  ay_B_gt, sigma_ay, "a_y [m/s²]", "Accel y — body frame"),
        (omega, om_gt,   sigma_om, "ω [rad/s]",  "Angular rate — body frame"),
    ]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=[p[4] for p in panels],
    )

    _ax_style = dict(gridcolor=_GRID, zerolinecolor=_GRID, color=_FG)

    for row, (data, gt, sigma_, ylabel, _) in enumerate(panels, start=1):

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

        # ±1σ theoretical filled region
        fig.add_trace(go.Scatter(
            x=np.concatenate([times, times[::-1]]),
            y=np.concatenate([gt + sigma_, (gt - sigma_)[::-1]]),
            fill="toself",
            fillcolor="rgba(188,140,255,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="±1σ region",
            showlegend=(row == 1),
            legendgroup="sigma_fill",
        ), row=row, col=1)

        # ±1σ dotted boundary lines
        for sign, sname in [(1, "+1σ"), (-1, "−1σ")]:
            fig.add_trace(go.Scatter(
                x=times, y=gt + sign * sigma_,
                mode="lines",
                line=dict(color=_PURPLE, width=2.0, dash="dot"),
                name=sname,
                showlegend=(row == 1),
                legendgroup=sname,
            ), row=row, col=1)

        # Ground truth solid line
        fig.add_trace(go.Scatter(
            x=times, y=gt,
            mode="lines",
            line=dict(color=_ORANGE, width=2.5),
            name="ground truth",
            showlegend=(row == 1),
            legendgroup="gt",
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
    title: str = "MC Trajectories — Ground Truth & ±1σ Bounds",
    alpha: float = 0.35,
) -> go.Figure:
    """
    Noisy position measurements for every MC trial, overlaid with the ground
    truth path (trial 0, solid orange) and theoretical ±1σ tube from
    noise_cfg.pos_cov (dotted purple, offset perpendicular to the GT path).

    The tube half-width at each point is the 1σ uncertainty projected onto
    the path-normal direction: sqrt(nx²·σ_x² + ny²·σ_y²).

    Parameters
    ----------
    traj  : RigidBodyTrajectory with n > 1
    title : figure title
    alpha : opacity for individual trial lines
    """
    n  = traj.n
    nt = traj.nt

    # Noisy position measurements: (n, nt)
    xs = traj.pos_meas_W[:, :, 0]
    ys = traj.pos_meas_W[:, :, 1]

    # Ground truth from trial 0 (noiseless poses)
    gt_x = np.array([traj.poses[k].x[0] for k in range(nt)])
    gt_y = np.array([traj.poses[k].y[0] for k in range(nt)])

    # Theoretical noise std from pos_cov
    sigma_x = np.sqrt(traj.noise_cfg.pos_cov[0, 0])   # scalar
    sigma_y = np.sqrt(traj.noise_cfg.pos_cov[1, 1])

    # Unit normal perpendicular to the ground truth path
    dx  = np.gradient(gt_x)
    dy  = np.gradient(gt_y)
    mag = np.hypot(dx, dy) + 1e-12
    nx  = -dy / mag
    ny  =  dx / mag

    # 1σ tube half-width: project noise ellipse onto normal direction
    sigma_n = np.sqrt(nx**2 * sigma_x**2 + ny**2 * sigma_y**2)

    upper_x = gt_x + sigma_n * nx
    upper_y = gt_y + sigma_n * ny
    lower_x = gt_x - sigma_n * nx
    lower_y = gt_y - sigma_n * ny

    fig = go.Figure()

    # Individual MC trial measurement paths
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
        fillcolor="rgba(188,140,255,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1σ region",
    ))

    # ±1σ dotted boundary lines
    for bx, by, label in [(upper_x, upper_y, "+1σ"), (lower_x, lower_y, "−1σ")]:
        fig.add_trace(go.Scatter(
            x=bx, y=by,
            mode="lines",
            line=dict(color=_PURPLE, width=2.0, dash="dot"),
            name=label,
        ))

    # Ground truth path — solid orange
    fig.add_trace(go.Scatter(
        x=gt_x, y=gt_y,
        mode="lines",
        line=dict(color=_ORANGE, width=3.0),
        name="ground truth",
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
# 6. Confidence ellipse helper
# ---------------------------------------------------------------------------

def confidence_ellipse(
    cx: float,
    cy: float,
    cov_2x2: np.ndarray,
    n_std: float = 2.0,
    n_points: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Closed-curve (x, y) for an `n_std`-sigma confidence ellipse of `cov_2x2`
    centred at (cx, cy).
    """
    eigvals, eigvecs = np.linalg.eigh(cov_2x2)
    eigvals = np.clip(eigvals, 0.0, None)
    angle   = np.arctan2(eigvecs[1, 1], eigvecs[0, 1])
    a = n_std * np.sqrt(eigvals[1])   # major semi-axis
    b = n_std * np.sqrt(eigvals[0])   # minor semi-axis

    t = np.linspace(0.0, 2.0 * np.pi, n_points)
    xl = a * np.cos(t)
    yl = b * np.sin(t)
    c, s = np.cos(angle), np.sin(angle)
    return cx + c * xl - s * yl, cy + s * xl + c * yl


# ---------------------------------------------------------------------------
# 7. Animated EKF estimate — single trial
# ---------------------------------------------------------------------------

def animate_estimate(
    traj:       RigidBodyTrajectory,
    s_hist:     np.ndarray,
    P_hist:     np.ndarray,
    trial:      int = 0,
    step:       int = 4,
    n_std:      float = 2.0,
    heading_len: float = 1.0,
    vel_scale:  float = 0.5,
    title:      str = "EKF — Single Trial Estimate",
    frame_duration_ms: int = 40,
) -> go.Figure:
    """
    Animate the EKF estimate for one MC trial.

    Each frame shows the estimated path so far, the ground-truth path, an
    `n_std`-sigma position-variance ellipse, heading ±n_std·σ_θ rays, and a
    mean-velocity arrow.

    Parameters
    ----------
    s_hist : (n_trials, nt, 5)  EKF state history [x, y, θ, vx, vy]
    P_hist : (n_trials, nt, 5, 5)  EKF covariance history
    step   : timestep stride for animation frames
    n_std  : confidence level (e.g. 2.0 ≈ 95 %)
    heading_len : length of heading rays in world units
    vel_scale   : multiplier on velocity vector for the arrow length
    """
    s = s_hist[trial]                     # (nt, 5)
    P = P_hist[trial]                     # (nt, 5, 5)
    gt_xs, gt_ys, _ = _extract_path(traj, trial)
    nt      = s.shape[0]
    indices = np.arange(0, nt, step)

    pad  = 1.0
    xmin = min(s[:, 0].min(), gt_xs.min()) - pad
    xmax = max(s[:, 0].max(), gt_xs.max()) + pad
    ymin = min(s[:, 1].min(), gt_ys.min()) - pad
    ymax = max(s[:, 1].max(), gt_ys.max()) + pad

    meas_xs = traj.pos_meas_W[trial, :, 0]
    meas_ys = traj.pos_meas_W[trial, :, 1]

    def frame_traces(i: int) -> list[go.Scatter]:
        cx, cy, th, vx, vy = s[i]
        sigma_th = n_std * np.sqrt(max(P[i, 2, 2], 0.0))
        ex, ey   = confidence_ellipse(cx, cy, P[i, 0:2, 0:2], n_std=n_std)

        rays_x, rays_y = [], []
        for sign in (+1, -1):
            ang = th + sign * sigma_th
            rays_x.extend([cx, cx + heading_len * np.cos(ang), None])
            rays_y.extend([cy, cy + heading_len * np.sin(ang), None])

        return [
            go.Scatter(
                x=gt_xs[: i + 1], y=gt_ys[: i + 1],
                mode="lines",
                line=dict(color=_ORANGE, width=2, dash="dot"),
                name="ground truth",
            ),
            go.Scatter(
                x=meas_xs[: i + 1], y=meas_ys[: i + 1],
                mode="lines",
                line=dict(color=_WARM, width=0.8),
                opacity=0.6,
                name="measurement",
            ),
            go.Scatter(
                x=s[: i + 1, 0], y=s[: i + 1, 1],
                mode="lines",
                line=dict(color=_ACCENT, width=2),
                name="estimate",
            ),
            go.Scatter(
                x=ex, y=ey,
                mode="lines",
                line=dict(color=_PURPLE, width=1.5),
                fill="toself", fillcolor="rgba(188,140,255,0.12)",
                name=f"position ±{n_std:g}σ",
            ),
            go.Scatter(
                x=rays_x, y=rays_y,
                mode="lines",
                line=dict(color=_PURPLE, width=2, dash="dash"),
                name=f"heading ±{n_std:g}σ",
            ),
            go.Scatter(
                x=[cx, cx + vel_scale * vx],
                y=[cy, cy + vel_scale * vy],
                mode="lines+markers",
                line=dict(color=_GREEN, width=3),
                marker=dict(symbol=["circle", "triangle-up"],
                            size=[8, 12], color=_GREEN,
                            angle=[0, np.degrees(np.arctan2(vy, vx)) - 90]),
                name="velocity",
            ),
            go.Scatter(
                x=[cx], y=[cy],
                mode="markers",
                marker=dict(color=_ACCENT, size=10, symbol="circle",
                            line=dict(color=_FG, width=1)),
                name="current estimate",
                showlegend=False,
            ),
        ]

    frames = [go.Frame(data=frame_traces(i), name=str(i)) for i in indices]
    fig    = go.Figure(data=frame_traces(indices[0]), frames=frames)
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=_FG)),
        xaxis=dict(range=[xmin, xmax], **_YAXIS_BASE, title="x [m]"),
        yaxis=dict(range=[ymin, ymax], scaleanchor="x", scaleratio=1,
                   **_YAXIS_BASE, title="y [m]"),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0.02, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶  Play", method="animate",
                     args=[None, dict(
                         frame=dict(duration=frame_duration_ms, redraw=True),
                         fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸  Pause", method="animate",
                     args=[[None], dict(
                         frame=dict(duration=0, redraw=False),
                         mode="immediate")]),
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
            bgcolor=_GRID, bordercolor=_GRID,
        )],
        **_LAYOUT_BASE,
    )
    return fig


# ---------------------------------------------------------------------------
# 8. EKF MC estimates — all trials overlaid with ground truth
# ---------------------------------------------------------------------------

def plot_mc_estimates(
    traj:   RigidBodyTrajectory,
    s_hist: np.ndarray,
    title:  str = "EKF — MC Trial Estimates vs Ground Truth",
    alpha:  float = 0.5,
) -> go.Figure:
    """
    Overlay every MC trial's estimated (x, y) path with the ground-truth path.

    Parameters
    ----------
    s_hist : (n_trials, nt, 5)  EKF state history [x, y, θ, vx, vy]
    """
    n = s_hist.shape[0]
    gt_xs, gt_ys, _ = _extract_path(traj, trial=0)

    fig = go.Figure()
    for i in range(n):
        colour = _MC_COLOURS[i % len(_MC_COLOURS)]
        fig.add_trace(go.Scatter(
            x=s_hist[i, :, 0], y=s_hist[i, :, 1],
            mode="lines",
            line=dict(color=colour, width=1.2),
            opacity=alpha,
            name=f"trial {i}",
            showlegend=(n <= 12),
        ))

    fig.add_trace(go.Scatter(
        x=gt_xs, y=gt_ys,
        mode="lines",
        line=dict(color=_ORANGE, width=3.0, dash="dash"),
        name="ground truth",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=_FG)),
        xaxis=dict(title="x [m]", **_YAXIS_BASE),
        yaxis=dict(title="y [m]", scaleanchor="x", scaleratio=1, **_YAXIS_BASE),
        **_LAYOUT_BASE,
    )
    return fig


# ---------------------------------------------------------------------------
# 9. EKF state time series — estimate vs ground truth vs measurements
# ---------------------------------------------------------------------------

def plot_ekf_states(
    traj:   RigidBodyTrajectory,
    s_hist: np.ndarray,
    trial:  int = 0,
    title:  str = "EKF — State Estimates vs Ground Truth & Measurements",
) -> go.Figure:
    """
    Five stacked subplots (x, y, θ, vx, vy) comparing the KF estimate against
    the ground truth and, where available, the raw noisy measurements.

    Parameters
    ----------
    s_hist : (n_trials, nt, 5)  EKF state history [x, y, θ, vx, vy]
    trial  : which MC trial to plot
    """
    times = traj.timestamps
    nt    = traj.nt

    est = s_hist[trial]                          # (nt, 5)

    gt_x  = np.array([traj.poses[k].x[trial]     for k in range(nt)])
    gt_y  = np.array([traj.poses[k].y[trial]     for k in range(nt)])
    gt_th = np.unwrap(np.array([traj.poses[k].theta[trial] for k in range(nt)]))
    gt_vx = traj.velocity_W[trial, :, 0]
    gt_vy = traj.velocity_W[trial, :, 1]

    meas_x  = traj.pos_meas_W[trial, :, 0]
    meas_y  = traj.pos_meas_W[trial, :, 1]
    meas_th = np.unwrap(traj.heading_meas_W[trial, :])

    panels = [
        ("x [m]",     est[:, 0],              gt_x,  meas_x),
        ("y [m]",     est[:, 1],              gt_y,  meas_y),
        ("θ [rad]",   np.unwrap(est[:, 2]),   gt_th, meas_th),
        ("vx [m/s]",  est[:, 3], gt_vx, None),
        ("vy [m/s]",  est[:, 4], gt_vy, None),
    ]

    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[p[0] for p in panels],
    )

    _ax_style = dict(gridcolor=_GRID, zerolinecolor=_GRID, color=_FG)

    for row, (ylabel, est_vals, gt_vals, meas_vals) in enumerate(panels, start=1):
        if meas_vals is not None:
            fig.add_trace(go.Scatter(
                x=times, y=meas_vals,
                mode="lines",
                line=dict(color=_WARM, width=0.8),
                opacity=0.55,
                name="measurement",
                showlegend=(row == 1),
                legendgroup="meas",
            ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=times, y=gt_vals,
            mode="lines",
            line=dict(color=_ORANGE, width=2.0, dash="dash"),
            name="ground truth",
            showlegend=(row == 1),
            legendgroup="gt",
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=times, y=est_vals,
            mode="lines",
            line=dict(color=_ACCENT, width=2.0),
            name="KF estimate",
            showlegend=(row == 1),
            legendgroup="est",
        ), row=row, col=1)

        fig.update_yaxes(title_text=ylabel, row=row, col=1, **_ax_style)
        fig.update_xaxes(row=row, col=1, **_ax_style)

    fig.update_xaxes(title_text="time [s]", row=5, col=1)

    for ann in fig.layout.annotations:
        ann.font.color = _FG
        ann.font.size  = 12

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=_FG)),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_FG, family="monospace"),
        legend=dict(bgcolor="#161b22", bordercolor=_GRID, borderwidth=1),
        height=900,
    )
    return fig


# ---------------------------------------------------------------------------
# 10. MC trial total MSE — trace(P) over time
# ---------------------------------------------------------------------------

def plot_mc_mse(
    P_hist: np.ndarray,
    traj:   RigidBodyTrajectory,
    s_hist: np.ndarray,
    title:  str = "EKF — Uncertainty & Position Error over Time",
    alpha:  float = 0.35,
) -> go.Figure:
    """
    Three overlaid series on a log-scale axis:
      - trace(P) per trial                        (dashed, faint)
      - total squared error per trial vs GT       (solid, faint, same units)
      - MC variance of squared error across trials (bold white)

    Parameters
    ----------
    P_hist : (n, nt, 5, 5)  EKF covariance history
    s_hist : (n, nt, 5)     EKF state history  [x, y, theta, vx, vy]
    """
    n, nt = P_hist.shape[:2]
    times = traj.timestamps

    trace_P = P_hist[:, :, np.arange(5), np.arange(5)].sum(axis=-1)  # (n, nt)

    # ground truth for all 5 EKF states
    gt_x     = np.stack([traj.poses[k].x         for k in range(nt)], axis=1)  # (n, nt)
    gt_y     = np.stack([traj.poses[k].y         for k in range(nt)], axis=1)
    gt_theta = np.stack([traj.poses[k].theta     for k in range(nt)], axis=1)
    gt_vx    = traj.velocity_W[:, :, 0]                                         # (n, nt)
    gt_vy    = traj.velocity_W[:, :, 1]

    sq_err = (
        (s_hist[:, :, 0] - gt_x)     ** 2
      + (s_hist[:, :, 1] - gt_y)     ** 2
      + (s_hist[:, :, 2] - gt_theta) ** 2
      + (s_hist[:, :, 3] - gt_vx)    ** 2
      + (s_hist[:, :, 4] - gt_vy)    ** 2
    )  # (n, nt)

    mc_var = sq_err.var(axis=0)  # (nt,)

    fig = go.Figure()

    for i in range(n):
        colour = _MC_COLOURS[i % len(_MC_COLOURS)]
        fig.add_trace(go.Scatter(
            x=times, y=trace_P[i],
            mode="lines",
            line=dict(color=colour, width=1, dash="dash"),
            opacity=alpha,
            legendgroup="traceP",
            name="trace(P)" if i == 0 else f"trace(P) {i}",
            showlegend=(i == 0),
        ))

    for i in range(n):
        colour = _MC_COLOURS[i % len(_MC_COLOURS)]
        fig.add_trace(go.Scatter(
            x=times, y=sq_err[i],
            mode="lines",
            line=dict(color=colour, width=1),
            opacity=alpha,
            legendgroup="sqerr",
            name="total sq. error" if i == 0 else f"sq. error {i}",
            showlegend=(i == 0),
        ))

    fig.add_trace(go.Scatter(
        x=times, y=mc_var,
        mode="lines",
        line=dict(color="#ffffff", width=2.5),
        name="MC var(sq. error)",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=_FG)),
        xaxis=dict(title="time [s]", **_YAXIS_BASE),
        yaxis=dict(title="value (log scale)", type="log", **_YAXIS_BASE),
        **_LAYOUT_BASE,
    )
    return fig
