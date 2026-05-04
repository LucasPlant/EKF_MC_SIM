"""
Rigid body simulation framework in SE(2) for EKF testing.

Notation: A_T_B is the transform from frame B to frame A.
  - Composition: A_T_C = A_T_B * B_T_C
  - In math: T_AC = T_AB * T_BC
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# SE(2)
# ---------------------------------------------------------------------------

class SE2:
    """
    Vectorized SE(2) — stores n transforms as (x, y, theta) arrays.

    Coordinate convention
    ---------------------
    A point p in frame B expressed in frame A:
        p_A = T_AB @ p_B     (homogeneous)

    Composition
    -----------
    A_T_C = A_T_B * B_T_C   (Python __mul__)
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray):
        self.x     = np.asarray(x,     dtype=float)
        self.y     = np.asarray(y,     dtype=float)
        self.theta = np.asarray(theta, dtype=float)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @staticmethod
    def identity(n: int = 1) -> SE2:
        return SE2(np.zeros(n), np.zeros(n), np.zeros(n))

    @staticmethod
    def from_matrix(mat: np.ndarray) -> SE2:
        """mat : (..., 3, 3) homogeneous matrices → SE2 of shape (...)"""
        x     = mat[..., 0, 2]
        y     = mat[..., 1, 2]
        theta = np.arctan2(mat[..., 1, 0], mat[..., 0, 0])
        return SE2(x.ravel(), y.ravel(), theta.ravel())

    # ------------------------------------------------------------------
    # Homogeneous matrix representation
    # ------------------------------------------------------------------

    def as_matrix(self) -> np.ndarray:
        """Returns (n, 3, 3) array of homogeneous matrices."""
        n  = len(self.x)
        c  = np.cos(self.theta)
        s  = np.sin(self.theta)
        T  = np.zeros((n, 3, 3))
        T[:, 0, 0] = c;  T[:, 0, 1] = -s;  T[:, 0, 2] = self.x
        T[:, 1, 0] = s;  T[:, 1, 1] =  c;  T[:, 1, 2] = self.y
        T[:, 2, 2] = 1.0
        return T

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def __mul__(self, other: SE2) -> SE2:
        """A_T_C = A_T_B * B_T_C  (vectorized, broadcasts n)."""
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        x_new     = c * other.x - s * other.y + self.x
        y_new     = s * other.x + c * other.y + self.y
        theta_new = self.theta + other.theta
        return SE2(x_new, y_new, _wrap(theta_new))

    def apply(self, other: SE2) -> SE2:
        """Alias for self * other."""
        return self.__mul__(other)

    def inverse(self) -> SE2:
        """T^{-1}"""
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        x_new = -(c * self.x + s * self.y)
        y_new = -(-s * self.x + c * self.y)
        return SE2(x_new, y_new, _wrap(-self.theta))

    # ------------------------------------------------------------------
    # Lie algebra  (se(2) ↔ SE(2))
    # ------------------------------------------------------------------

    def log(self) -> np.ndarray:
        """
        Logarithmic map: SE(2) → se(2).

        Returns xi = (rho_x, rho_y, theta) of shape (n, 3).

        For the SE(2) log map, the translational component is:

            V^{-1} = (1/theta) * [ sin(theta)   1-cos(theta) ]^{-1}
                                  [ cos(theta)-1   sin(theta) ]

            rho = V^{-1} @ t
        """
        theta = self.theta
        n     = len(theta)
        xi    = np.zeros((n, 3))
        xi[:, 2] = theta

        small = np.abs(theta) < 1e-8
        big   = ~small

        # Small angle: V^{-1} ≈ I
        xi[small, 0] = self.x[small]
        xi[small, 1] = self.y[small]

        # General case
        if big.any():
            th  = theta[big]
            c   = np.cos(th)
            s   = np.sin(th)
            half_th = th / 2.0
            A   = s / th
            B   = (1.0 - c) / th
            det = A**2 + B**2          # = 1 always, kept for clarity
            Vinv_00 =  A / det
            Vinv_01 =  B / det
            Vinv_10 = -B / det
            Vinv_11 =  A / det
            xi[big, 0] = Vinv_00 * self.x[big] + Vinv_01 * self.y[big]
            xi[big, 1] = Vinv_10 * self.x[big] + Vinv_11 * self.y[big]

        return xi

    @staticmethod
    def exp(xi: np.ndarray) -> SE2:
        """
        Exponential map: se(2) → SE(2).

        xi : (n, 3) array  [rho_x, rho_y, theta]

        Translation recovered via:
            t = V @ rho
        where
            V = (sin(theta)/theta) I  +  ((1-cos)/theta) J
            J = [[0,-1],[1,0]]
        """
        xi    = np.atleast_2d(xi)
        theta = xi[:, 2]
        n     = len(theta)
        x_out = np.zeros(n)
        y_out = np.zeros(n)

        small = np.abs(theta) < 1e-8
        big   = ~small

        x_out[small] = xi[small, 0]
        y_out[small] = xi[small, 1]

        if big.any():
            th = theta[big]
            c  = np.cos(th)
            s  = np.sin(th)
            A  = s / th
            B  = (1.0 - c) / th
            x_out[big] = A * xi[big, 0] - B * xi[big, 1]
            y_out[big] = B * xi[big, 0] + A * xi[big, 1]

        return SE2(x_out, y_out, _wrap(theta))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> SE2:
        return SE2(self.x[idx], self.y[idx], self.theta[idx])

    def __repr__(self) -> str:
        return f"SE2(x={self.x}, y={self.y}, theta={self.theta})"


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to (-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Noise configuration
# ---------------------------------------------------------------------------

@dataclass
class NoiseConfig:
    """
    Measurement noise parameters.

    imu_cov     : (3, 3) covariance for [accel_x, accel_y, omega]
    pos_cov     : (2, 2) covariance for [x, y] position measurement
    heading_var : scalar variance for heading (theta) measurement [rad²]
    seed        : optional RNG seed for reproducibility
    """
    imu_cov: np.ndarray = field(
        default_factory=lambda: np.diag([0.1**2, 0.1**2, 0.01**2])
    )
    pos_cov: np.ndarray = field(
        default_factory=lambda: np.diag([0.05**2, 0.05**2])
    )
    heading_var: float = 0.05**2
    seed: Optional[int] = None

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


# ---------------------------------------------------------------------------
# Trajectory container
# ---------------------------------------------------------------------------

@dataclass
class RigidBodyTrajectory:
    """
    Stores the full simulated trajectory for n Monte Carlo trials
    over nt time steps.

    Shape conventions
    -----------------
    - poses            : list of nt SE2 objects, each of length n
    - timestamps       : (nt,)
    - velocity_W       : (n, nt, 2)   [vx, vy] in world frame
    - acceleration_W   : (n, nt, 2)   [ax, ay] in world frame
    - force_total_W_arr: (n, nt, 2)   [Fx, Fy] total force in world frame
    - angular_velocity : (n, nt)
    - angular_accel    : (n, nt)
    - torque           : (n, nt)
    - accel_meas_B     : (n, nt, 2)   noisy body-frame acceleration measurement
    - gyro_meas_B      : (n, nt)      noisy body-frame angular velocity measurement
    - pos_meas_W       : (n, nt, 2)   noisy world-frame position measurement
    - heading_meas_W   : (n, nt)      noisy world-frame heading measurement [rad]
    """
    n:   int
    nt:  int
    mass:    np.ndarray            # (n,)
    inertia: np.ndarray            # (n,)
    noise_cfg: NoiseConfig

    poses:             list        = field(default_factory=list)   # nt × SE2(n)
    timestamps:        np.ndarray  = field(default_factory=lambda: np.array([]))

    velocity_W:         np.ndarray  = field(init=False)
    acceleration_W:     np.ndarray  = field(init=False)
    force_total_W_arr:  np.ndarray  = field(init=False)
    angular_velocity:   np.ndarray  = field(init=False)
    angular_accel:      np.ndarray  = field(init=False)
    torque:             np.ndarray  = field(init=False)

    accel_meas_B:       np.ndarray  = field(init=False)
    gyro_meas_B:        np.ndarray  = field(init=False)
    pos_meas_W:         np.ndarray  = field(init=False)
    heading_meas_W:     np.ndarray  = field(init=False)

    def __post_init__(self):
        n, nt = self.n, self.nt
        self.velocity_W        = np.zeros((n, nt, 2))
        self.acceleration_W    = np.zeros((n, nt, 2))
        self.force_total_W_arr = np.zeros((n, nt, 2))
        self.angular_velocity  = np.zeros((n, nt))
        self.angular_accel     = np.zeros((n, nt))
        self.torque            = np.zeros((n, nt))
        self.accel_meas_B      = np.zeros((n, nt, 2))
        self.gyro_meas_B       = np.zeros((n, nt))
        self.pos_meas_W        = np.zeros((n, nt, 2))
        self.heading_meas_W    = np.zeros((n, nt))


# ---------------------------------------------------------------------------
# Base simulator
# ---------------------------------------------------------------------------

class RigidBodySim:
    """
    Base class for SE(2) rigid body simulation.

    Subclasses override force_body_B (body-frame forces), force_world_W (world-frame forces),
    and/or torque to define different motion profiles. force_total_W combines both.

    State vector  s = [x, y, theta, vx, vy, omega]  shape (n, 6)
    """

    def __init__(
        self,
        mass:      float | np.ndarray,
        inertia:   float | np.ndarray,
        noise_cfg: NoiseConfig,
        dt:        float = 0.01,
        n:         int   = 1,
    ):
        self.mass      = np.broadcast_to(np.asarray(mass,    float), (n,)).copy()
        self.inertia   = np.broadcast_to(np.asarray(inertia, float), (n,)).copy()
        self.noise_cfg = noise_cfg
        self.dt        = dt
        self.n         = n

    # ------------------------------------------------------------------
    # Callbacks — override in subclasses
    # ------------------------------------------------------------------

    def force_body_B(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Body-frame force component (e.g. thrust, propulsion).
        Returns (n, 2) array [Fx_B, Fy_B].
        Override in subclasses that apply forces in the body frame.
        """
        return np.zeros((self.n, 2))

    def force_world_W(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        World-frame force component (e.g. gravity, drag, centripetal).
        Returns (n, 2) array [Fx_W, Fy_W].
        Override in subclasses that apply forces directly in the world frame.
        """
        return np.zeros((self.n, 2))

    def torque(self, t: float, state: np.ndarray) -> float | np.ndarray:
        """Torque about z-axis (frame-independent in 2D). Returns (n,) array."""
        return np.zeros(self.n)

    def force_total_W(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Total force in the world frame used by the integrator.
        Rotates force_body_B into the world frame and adds force_world_W.
        Returns (n, 2) array [Fx_W, Fy_W].
        """
        theta = state[:, 2]
        c, s  = np.cos(theta), np.sin(theta)
        fb    = self.force_body_B(t, state)
        fb_to_W_x = c * fb[:, 0] - s * fb[:, 1]
        fb_to_W_y = s * fb[:, 0] + c * fb[:, 1]
        fw        = self.force_world_W(t, state)
        return np.stack([fb_to_W_x + fw[:, 0], fb_to_W_y + fw[:, 1]], axis=-1)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def _state_derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        state  : (n, 6)  [x, y, theta, vx, vy, omega]
        returns: (n, 6)  [vx, vy, omega, ax, ay, alpha]
        """
        fw    = self.force_total_W(t, state)         # (n, 2)
        tau   = self.torque(t, state)               # (n,)
        ax    = fw[:, 0] / self.mass
        ay    = fw[:, 1] / self.mass
        alpha = tau / self.inertia
        dsdt  = np.empty_like(state)
        dsdt[:, 0] = state[:, 3]   # dx/dt  = vx
        dsdt[:, 1] = state[:, 4]   # dy/dt  = vy
        dsdt[:, 2] = state[:, 5]   # dθ/dt  = ω
        dsdt[:, 3] = ax
        dsdt[:, 4] = ay
        dsdt[:, 5] = alpha
        return dsdt

    def _rk4_step(self, t: float, state: np.ndarray) -> np.ndarray:
        """Single RK4 step. state : (n, 6)"""
        dt = self.dt
        k1 = self._state_derivative(t,          state)
        k2 = self._state_derivative(t + dt/2,   state + dt/2 * k1)
        k3 = self._state_derivative(t + dt/2,   state + dt/2 * k2)
        k4 = self._state_derivative(t + dt,     state + dt   * k3)
        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        new_state[:, 2] = _wrap(new_state[:, 2])
        return new_state

    # ------------------------------------------------------------------
    # Noise injection
    # ------------------------------------------------------------------

    def _add_noise(self, traj: RigidBodyTrajectory) -> None:
        """Populate measurement arrays with additive Gaussian noise."""
        rng = self.noise_cfg.rng()
        n, nt = traj.n, traj.nt

        imu_cov = self.noise_cfg.imu_cov
        pos_cov = self.noise_cfg.pos_cov

        imu_noise     = rng.multivariate_normal(np.zeros(3), imu_cov, size=(n, nt))
        pos_noise     = rng.multivariate_normal(np.zeros(2), pos_cov, size=(n, nt))
        heading_noise = rng.normal(0.0, np.sqrt(self.noise_cfg.heading_var), size=(n, nt))

        # Body-frame acceleration = R^T * world_accel
        for k in range(nt):
            theta = traj.poses[k].theta          # (n,)
            c, s  = np.cos(theta), np.sin(theta)
            ax_w  = traj.acceleration_W[:, k, 0]
            ay_w  = traj.acceleration_W[:, k, 1]
            # Rotate to body frame
            ax_b  =  c * ax_w + s * ay_w
            ay_b  = -s * ax_w + c * ay_w
            traj.accel_meas_B[:, k, 0] = ax_b + imu_noise[:, k, 0]
            traj.accel_meas_B[:, k, 1] = ay_b + imu_noise[:, k, 1]
            traj.gyro_meas_B[:, k]     = traj.angular_velocity[:, k] + imu_noise[:, k, 2]

        for k in range(nt):
            traj.pos_meas_W[:, k, 0] = traj.poses[k].x + pos_noise[:, k, 0]
            traj.pos_meas_W[:, k, 1] = traj.poses[k].y + pos_noise[:, k, 1]
            traj.heading_meas_W[:, k] = _wrap(traj.poses[k].theta + heading_noise[:, k])

    # ------------------------------------------------------------------
    # Main simulate entry point
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_span:        tuple[float, float],
        initial_state: np.ndarray,
    ) -> RigidBodyTrajectory:
        """
        Run vectorized RK4 simulation.

        Parameters
        ----------
        t_span        : (t0, tf)
        initial_state : (n, 6) array [x0, y0, theta0, vx0, vy0, omega0]
                        or (6,) broadcast to all n trials.

        Returns
        -------
        RigidBodyTrajectory fully populated including noisy measurements.
        """
        t0, tf = t_span
        dt     = self.dt
        times  = np.arange(t0, tf + 1e-12, dt)
        nt     = len(times)
        n      = self.n

        state = np.broadcast_to(
            np.atleast_2d(np.asarray(initial_state, float)), (n, 6)
        ).copy()

        traj = RigidBodyTrajectory(
            n=n, nt=nt,
            mass=self.mass.copy(),
            inertia=self.inertia.copy(),
            noise_cfg=self.noise_cfg,
            timestamps=times,
        )

        # Allocate force/torque cache reused inside loop
        fw_cache = np.zeros((n, 2))

        for k, t in enumerate(times):
            # Cache dynamics at current state
            fw    = self.force_total_W(t, state)
            tau   = self.torque(t, state)
            ax    = fw[:, 0] / self.mass
            ay    = fw[:, 1] / self.mass
            alpha = tau / self.inertia

            # Store
            traj.poses.append(SE2(state[:, 0].copy(),
                                   state[:, 1].copy(),
                                   state[:, 2].copy()))
            traj.velocity_W[:, k, :]          = state[:, 3:5]
            traj.angular_velocity[:, k]       = state[:, 5]
            traj.acceleration_W[:, k, 0]      = ax
            traj.acceleration_W[:, k, 1]      = ay
            traj.angular_accel[:, k]          = alpha
            traj.force_total_W_arr[:, k, :]   = fw
            traj.torque[:, k]                 = tau

            # Step forward (except at last step)
            if k < nt - 1:
                state = self._rk4_step(t, state)

        self._add_noise(traj)
        return traj


# ---------------------------------------------------------------------------
# Concrete simulations
# ---------------------------------------------------------------------------

class CircleCenterFacing(RigidBodySim):
    """
    Circular orbit — body always faces the center of the circle.

    The net centripetal force points toward the origin.  Orientation
    θ is set so the body x-axis faces inward (toward center).

    Parameters
    ----------
    radius   : orbit radius [m]
    speed    : tangential speed [m/s]
    """

    def __init__(self, radius: float = 5.0, speed: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius
        self.speed  = speed

    def force_world_W(self, t: float, state: np.ndarray) -> np.ndarray:
        """Centripetal force directed toward the origin (world frame)."""
        x, y = state[:, 0], state[:, 1]
        r    = np.hypot(x, y) + 1e-12
        # Centripetal acceleration magnitude: v²/r
        a_c  = self.speed**2 / self.radius
        fx   = -a_c * self.mass * x / r
        fy   = -a_c * self.mass * y / r
        return np.stack([fx, fy], axis=-1)

    def torque(self, t: float, state: np.ndarray) -> np.ndarray:
        # Drive theta to point toward center: desired = atan2(-y, -x) + π
        x, y   = state[:, 0], state[:, 1]
        theta  = state[:, 2]
        desired = np.arctan2(-y, -x)
        err     = _wrap(desired - theta)
        kp, kd  = 10.0, 2.0
        omega   = state[:, 5]
        return self.inertia * (kp * err - kd * omega)


class CircleTangentFacing(RigidBodySim):
    """
    Circular orbit — body faces the direction of travel (tangent).
    """

    def __init__(self, radius: float = 5.0, speed: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius
        self.speed  = speed

    def force_world_W(self, t: float, state: np.ndarray) -> np.ndarray:
        """Centripetal force directed toward the origin (world frame)."""
        x, y = state[:, 0], state[:, 1]
        r    = np.hypot(x, y) + 1e-12
        a_c  = self.speed**2 / self.radius
        fx   = -a_c * self.mass * x / r
        fy   = -a_c * self.mass * y / r
        return np.stack([fx, fy], axis=-1)

    def torque(self, t: float, state: np.ndarray) -> np.ndarray:
        x, y   = state[:, 0], state[:, 1]
        theta  = state[:, 2]
        # Tangent direction: perpendicular to radius, in direction of travel
        desired = np.arctan2(-x, y)   # tangent for CCW orbit
        err     = _wrap(desired - theta)
        kp, kd  = 10.0, 2.0
        omega   = state[:, 5]
        return self.inertia * (kp * err - kd * omega)


class SinusoidForward(RigidBodySim):
    """
    Traces a vertical sine wave: x oscillates, y advances at constant average
    speed, body heading always aligns with the direction of travel.

    Desired velocity in world frame:
        vx_d(t) = lat_amp * cos(2π * lat_freq * t)
        vy_d(t) = sqrt(speed² − vx_d²)   ← keeps |v| = speed exactly

    A proportional velocity-tracking force drives the body to follow this
    profile.  The parametric curve (x, y) is an upright sine wave.

    Parameters
    ----------
    speed    : constant linear speed [m/s]  (must satisfy lat_amp ≤ speed)
    lat_amp  : lateral (x) velocity amplitude [m/s]
    lat_freq : lateral oscillation frequency [Hz]
    kv       : velocity tracking gain [1/s]
    """

    def __init__(
        self,
        speed:    float = 2.0,
        lat_amp:  float = 1.0,
        lat_freq: float = 0.3,
        kv:       float = 8.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.speed    = speed
        self.lat_amp  = min(lat_amp, speed)   # clamp so vy_d stays real
        self.lat_freq = lat_freq
        self.kv       = kv

    def force_world_W(self, t: float, state: np.ndarray) -> np.ndarray:
        """Velocity-tracking force in world frame."""
        vx, vy  = state[:, 3], state[:, 4]
        vx_d    = self.lat_amp * np.cos(2 * np.pi * self.lat_freq * t)
        vy_d    = np.sqrt(max(self.speed**2 - vx_d**2, 0.0))
        fx      = self.mass * self.kv * (vx_d - vx)
        fy      = self.mass * self.kv * (vy_d - vy)
        return np.stack([fx, fy], axis=-1)

    def torque(self, t: float, state: np.ndarray) -> np.ndarray:
        # Align heading with the current velocity direction
        vx, vy  = state[:, 3], state[:, 4]
        theta   = state[:, 2]
        speed   = np.hypot(vx, vy)
        desired = np.where(speed > 0.1, np.arctan2(vy, vx), theta)
        err     = _wrap(desired - theta)
        kp, kd  = 12.0, 3.0
        return self.inertia * (kp * err - kd * state[:, 5])


class RandomWalk(RigidBodySim):
    """
    Random walk — stochastic force and torque at each timestep.

    Parameters
    ----------
    force_std  : std dev of random world-frame force [N]
    torque_std : std dev of random torque [N·m]
    """

    def __init__(
        self,
        force_std:  float = 2.0,
        torque_std: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.force_std  = force_std
        self.torque_std = torque_std
        self._rng       = np.random.default_rng(self.noise_cfg.seed)

    def force_world_W(self, t: float, state: np.ndarray) -> np.ndarray:
        """Stochastic force applied directly in the world frame."""
        f = self._rng.normal(0.0, self.force_std, size=(self.n, 2))
        return f * self.mass[:, None]

    def torque(self, t: float, state: np.ndarray) -> np.ndarray:
        return self._rng.normal(0.0, self.torque_std, size=self.n) * self.inertia
