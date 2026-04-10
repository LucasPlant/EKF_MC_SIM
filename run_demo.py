"""
Demo runner — generates trajectories for each simulation type.

All plots for a run appear on a single scrollable page.

Usage
-----
    python run_demo.py                         # all demos, open in browser
    python run_demo.py circle_center           # one demo, open in browser
    python run_demo.py sinusoid --html         # one demo, save to HTML
    python run_demo.py --html                  # all demos, save to HTML
"""

import sys
import os
import tempfile
import webbrowser
import numpy as np

from sim import (
    NoiseConfig,
    CircleCenterFacing,
    CircleTangentFacing,
    SinusoidForward,
    RandomWalk,
)
from plot_utils import (
    plot_trajectory,
    animate_trajectory,
    plot_mc_paths,
    plot_imu_measurements,
    plot_trajectory_with_bounds,
)

OUTPUT_DIR = "output"

# ---------------------------------------------------------------------------
# Shared simulation parameters
# ---------------------------------------------------------------------------

N_TRIALS   = 30        # Monte Carlo trials
DT         = 0.02      # timestep [s]
T_SPAN     = (0.0, 8.0)

MASS       = 1.0       # [kg]
INERTIA    = 0.5       # [kg·m²]

NOISE_CFG  = NoiseConfig(
    imu_cov = np.diag([0.1**2, 0.1**2, 0.05**2]),
    pos_cov = np.diag([0.2**2, 0.2**2]),
    seed    = 42,
)

# ---------------------------------------------------------------------------
# Demo registry
# ---------------------------------------------------------------------------

def _initial_state(x0=5.0, y0=0.0, theta0=np.pi/2, vx0=0.0, vy0=2.0, w0=0.0):
    return np.array([x0, y0, theta0, vx0, vy0, w0])


DEMOS: dict[str, dict] = {
    "circle_center": dict(
        label   = "Circle — Center Facing",
        sim_cls = CircleCenterFacing,
        sim_kw  = dict(radius=5.0, speed=2.0),
        init    = _initial_state(x0=5.0, y0=0.0, theta0=np.pi, vx0=0.0, vy0=2.0),
    ),
    "circle_tangent": dict(
        label   = "Circle — Tangent Facing",
        sim_cls = CircleTangentFacing,
        sim_kw  = dict(radius=5.0, speed=2.0),
        init    = _initial_state(x0=5.0, y0=0.0, theta0=np.pi/2, vx0=0.0, vy0=2.0),
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
# Page builder
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

def _build_page(title: str, sections: list[tuple[str, object]]) -> str:
    """
    Combine multiple (heading, figure) pairs into one scrollable HTML page.

    sections : list of (section_title, go.Figure)
    """
    divs = []
    for i, (heading, fig) in enumerate(sections):
        include_js = "cdn" if i == 0 else False
        div = fig.to_html(full_html=False, include_plotlyjs=include_js,
                          div_id=f"plot-{i}", default_height="650px")
        divs.append(f'<h2>{heading}</h2><div class="plot-wrap">{div}</div>')

    body = "\n".join(divs)
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{title}</title>{_PAGE_STYLE}</head>
<body>
<h1>{title}</h1>
{body}
</body>
</html>"""


def _open_in_browser(html: str) -> None:
    """Write to a temp file and open it; file persists until the OS cleans /tmp."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(html)
        path = f.name
    webbrowser.open(f"file://{path}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_demo(name: str, cfg: dict, save_html: bool) -> None:
    label   = cfg["label"]
    sim_cls = cfg["sim_cls"]
    sim_kw  = cfg["sim_kw"]
    init    = cfg["init"]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    sim = sim_cls(
        mass      = MASS,
        inertia   = INERTIA,
        noise_cfg = NOISE_CFG,
        dt        = DT,
        n         = N_TRIALS,
        **sim_kw,
    )

    print(f"  Simulating {N_TRIALS} trials × {(T_SPAN[1]-T_SPAN[0])/DT:.0f} steps …")
    traj = sim.simulate(T_SPAN, init)
    print(f"  Generating plots …")

    sections = [
        ("Trajectory",
             plot_trajectory(traj, trial=0, n_poses=12, axis_scale=0.4,
                             title=f"{label} — Trajectory")),
        ("Animation",
             animate_trajectory(traj, trial=0, step=4, axis_scale=0.4,
                                title=f"{label} — Animation", frame_duration_ms=30)),
        (f"Monte Carlo Paths ({N_TRIALS} trials)",
             plot_mc_paths(traj,
                           title=f"{label} — Monte Carlo Paths ({N_TRIALS} trials)")),
        ("IMU Measurements",
             plot_imu_measurements(traj,
                                   title=f"{label} — IMU Measurements")),
        ("MC Trajectories with ±1σ Bounds",
             plot_trajectory_with_bounds(traj,
                                         title=f"{label} — MC Trajectories ±1σ")),
    ]

    page = _build_page(label, sections)

    if save_html:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = f"{OUTPUT_DIR}/{name}.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write(page)
        print(f"  Saved {path}")
    else:
        _open_in_browser(page)
        print(f"  Opened in browser.")


def main():
    args      = sys.argv[1:]
    save_html = "--html" in args
    requested = [a for a in args if a != "--html"] or list(DEMOS.keys())

    unknown = [k for k in requested if k not in DEMOS]
    if unknown:
        print(f"Unknown demo(s): {unknown}")
        print(f"Available: {list(DEMOS.keys())}")
        sys.exit(1)

    for name in requested:
        run_demo(name, DEMOS[name], save_html)

    if save_html:
        print(f"\nAll plots saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
