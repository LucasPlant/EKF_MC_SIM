"""
EKF demo — runs the vectorized Extended Kalman Filter on a Monte Carlo
simulation and produces two visualizations.

Plots
-----
1. Animated single trial: estimate path, position-variance ellipse,
   heading ±1σ bounds, mean velocity arrow, dotted ground truth.
2. All-trial overlay: every MC trial's position estimate path with
   the ground truth path.

Usage
-----
    python kf_demo.py                  # circle tangent demo, open in browser
    python kf_demo.py sinusoid         # sinusoid demo
    python kf_demo.py --html           # save to ./output/
"""

from __future__ import annotations

import os
import sys
import tempfile
import webbrowser

import numpy as np
import plotly.graph_objects as go

from sim import (
    NoiseConfig,
    CircleCenterFacing,
    CircleTangentFacing,
    SinusoidForward,
    RandomWalk,
    RigidBodyTrajectory,
)
from plot_utils import (
    plot_trajectory,
    animate_trajectory,
    plot_mc_paths,
    plot_imu_measurements,
    plot_trajectory_with_bounds,
    animate_estimate,
    plot_mc_estimates,
    plot_ekf_states,
    plot_mc_mse,
)
from EKF import EKF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_ekf_inputs(traj: RigidBodyTrajectory) -> tuple[np.ndarray, np.ndarray]:
    """
    Pack noisy simulation measurements into (z_seq, u_seq) for the EKF.

    z_seq : (n, nt, 3)  [x, y, theta] world-frame measurements
    u_seq : (n, nt, 3)  [a_xB, a_yB, omega] body-frame IMU readings
    """
    z = np.concatenate(
        [traj.pos_meas_W, traj.heading_meas_W[:, :, None]], axis=-1
    )
    u = np.concatenate(
        [traj.accel_meas_B, traj.gyro_meas_B[:, :, None]], axis=-1
    )
    return z, u


# ---------------------------------------------------------------------------
# Demo registry
# ---------------------------------------------------------------------------

N_TRIALS = 25
DT       = 0.02
T_SPAN   = (0.0, 15.0)
MASS     = 1.0
INERTIA  = 0.5

NOISE_CFG = NoiseConfig(
    imu_cov     = np.diag([0.10**2, 0.10**2, 0.05**2]),
    pos_cov     = np.diag([0.20**2, 0.20**2]),
    heading_var = 0.10**2,
    seed        = 7,
)

SIGMA0_INIT  = 5.0      # [m/s] initial velocity std dev


def _initial_state(x0=5.0, y0=0.0, theta0=np.pi/2, vx0=0.0, vy0=2.0, w0=0.0):
    return np.array([x0, y0, theta0, vx0, vy0, w0])


DEMOS: dict[str, dict] = {
    "circle_tangent": dict(
        label   = "Circle — Tangent Facing",
        sim_cls = CircleTangentFacing,
        sim_kw  = dict(radius=5.0, speed=2.0),
        init    = _initial_state(x0=5.0, y0=0.0, theta0=np.pi/2, vx0=0.0, vy0=2.0),
    ),
    "circle_center": dict(
        label   = "Circle — Center Facing",
        sim_cls = CircleCenterFacing,
        sim_kw  = dict(radius=5.0, speed=2.0),
        init    = _initial_state(x0=5.0, y0=0.0, theta0=np.pi, vx0=0.0, vy0=2.0),
    ),
    "sinusoid": dict(
        label   = "Forward Sinusoid",
        sim_cls = SinusoidForward,
        sim_kw  = dict(speed=2.0, lat_amp=1.0, lat_freq=0.3),
        init    = _initial_state(x0=0.0, y0=0.0, theta0=np.pi/2, vx0=0.0, vy0=2.0),
    ),
    "random_walk": dict(
        label   = "Random Walk",
        sim_cls = RandomWalk,
        sim_kw  = dict(force_std=2.0, torque_std=1.5),
        init    = _initial_state(x0=0.0, y0=0.0, theta0=0.0, vx0=0.0, vy0=0.0),
    ),
}


# ---------------------------------------------------------------------------
# HTML page assembly
# ---------------------------------------------------------------------------

_PAGE_STYLE = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; font-family: monospace; color: #c9d1d9; }
  h1   { padding: 28px 32px 8px; font-size: 1.4rem; color: #58a6ff;
         border-bottom: 1px solid #21262d; }
  h2   { padding: 24px 32px 0; font-size: 1.05rem; color: #8b949e; }
  .plot-wrap { padding: 12px 24px; }
</style>
"""

def _build_page(title: str, sections: list[tuple[str, go.Figure]]) -> str:
    divs = []
    for i, (heading, fig) in enumerate(sections):
        include_js = "cdn" if i == 0 else False
        div = fig.to_html(full_html=False, include_plotlyjs=include_js,
                          div_id=f"plot-{i}", default_height="700px")
        divs.append(f'<h2>{heading}</h2><div class="plot-wrap">{div}</div>')

    return (
        f'<!DOCTYPE html>\n<html><head><meta charset="utf-8">'
        f'<title>{title}</title>{_PAGE_STYLE}</head>'
        f'<body><h1>{title}</h1>{"".join(divs)}</body></html>'
    )


def _open_in_browser(html: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html",
                                     delete=False, encoding="utf-8") as f:
        f.write(html)
        path = f.name
    webbrowser.open(f"file://{path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_demo(name: str, cfg: dict, save_html: bool) -> None:
    label   = cfg["label"]
    sim_cls = cfg["sim_cls"]
    sim_kw  = cfg["sim_kw"]
    init    = cfg["init"]

    print(f"\n{'='*60}\n  {label}\n{'='*60}")

    sim = sim_cls(
        mass=MASS, inertia=INERTIA, noise_cfg=NOISE_CFG,
        dt=DT, n=N_TRIALS, **sim_kw,
    )

    print(f"  Simulating {N_TRIALS} trials × {(T_SPAN[1]-T_SPAN[0])/DT:.0f} steps …")
    traj = sim.simulate(T_SPAN, init)

    print(f"  Running EKF on all {N_TRIALS} trials …")
    z_seq, u_seq = build_ekf_inputs(traj)

    ekf = EKF(
        z0          = z_seq[:, 0, :],
        sigma0      = SIGMA0_INIT,
        dt          = DT,
        Sigma_imu   = NOISE_CFG.imu_cov,
        Sigma_pos   = NOISE_CFG.pos_cov,
        sigma_theta = np.sqrt(NOISE_CFG.heading_var),
    )
    s_hist, P_hist = ekf.run(z_seq, u_seq)

    print(f"  Generating plots …")
    sections = [
        # --- Simulation context (from plot_utils) -----------------------
        ("Trajectory",
            plot_trajectory(traj, trial=0, n_poses=12, axis_scale=0.4,
                            title=f"{label} — Trajectory")),
        ("Animation",
            animate_trajectory(traj, trial=0, step=4, axis_scale=0.4,
                               title=f"{label} — Animation",
                               frame_duration_ms=30)),
        (f"Monte Carlo Paths ({N_TRIALS} trials)",
            plot_mc_paths(traj,
                          title=f"{label} — Monte Carlo Paths ({N_TRIALS} trials)")),
        ("IMU Measurements",
            plot_imu_measurements(traj,
                                  title=f"{label} — IMU Measurements")),
        ("MC Trajectories with ±1σ Bounds",
            plot_trajectory_with_bounds(traj,
                                        title=f"{label} — MC Trajectories ±1σ")),
        # --- EKF results -------------------------------------------------
        ("EKF — Single-Trial Animation",
            animate_estimate(traj, s_hist, P_hist, trial=0,
                             title=f"{label} — EKF Single Trial")),
        ("EKF — State Estimates vs Ground Truth & Measurements",
            plot_ekf_states(traj, s_hist, trial=0,
                            title=f"{label} — EKF States")),
        (f"EKF — All MC Trial Estimates ({N_TRIALS} trials)",
            plot_mc_estimates(traj, s_hist,
                              title=f"{label} — EKF Estimates vs Ground Truth")),
        (f"EKF — Total MSE per MC Trial ({N_TRIALS} trials)",
            plot_mc_mse(P_hist, traj,
                        title=f"{label} — Total MSE (trace P) per Trial")),
    ]

    page = _build_page(label, sections)
    if save_html:
        os.makedirs("output", exist_ok=True)
        path = f"output/kf_{name}.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write(page)
        print(f"  Saved {path}")
    else:
        _open_in_browser(page)
        print(f"  Opened in browser.")


def main():
    args      = sys.argv[1:]
    save_html = "--html" in args
    requested = [a for a in args if a != "--html"] or ["circle_tangent"]

    unknown = [k for k in requested if k not in DEMOS]
    if unknown:
        print(f"Unknown demo(s): {unknown}\nAvailable: {list(DEMOS.keys())}")
        sys.exit(1)

    for name in requested:
        run_demo(name, DEMOS[name], save_html)


if __name__ == "__main__":
    main()
