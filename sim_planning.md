# SE(2) Rigid Body Simulation Framework

A vectorized simulation framework for testing Extended Kalman Filters on 2-D rigid body motion.
All classes operate over **n** parallel Monte Carlo trials simultaneously using NumPy arrays —
no Python loops over trials.

---

## Table of Contents

1. [Notation](#notation)
2. [SE(2) Lie Group](#se2-lie-group)
3. [Dynamics Model](#dynamics-model)
4. [Classes](#classes)
5. [Simulation Implementations](#simulation-implementations)
6. [Measurement Model](#measurement-model)
7. [Plotting Utilities](#plotting-utilities)
8. [Running the Demos](#running-the-demos)

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $n$    | Number of Monte Carlo trials (batch dimension) |
| $n_t$  | Number of timesteps |
| `A_T_B` | Transform from frame **B** to frame **A** (code convention) |
| $T_{AB}$ | Same transform in math notation |
| $\xi$  | Element of $\mathfrak{se}(2)$ — the Lie algebra twist |
| $\mathbf{s}$ | State vector $[x,\, y,\, \theta,\, \dot x_W,\, \dot y_W,\, \dot\theta]^\top$ |
| `_W`   | Quantity expressed in the **world frame** |
| `_B`   | Quantity expressed in the **body frame** |

**Composition rule** (code mirrors math):

```
A_T_C = A_T_B * B_T_C        # Python
```
$$T_{AC} = T_{AB}\, T_{BC}$$

---

## SE(2) Lie Group

### Group Element

An element of $SE(2)$ is a rigid planar transform represented by a $3\times 3$ homogeneous matrix:

$$T = \begin{bmatrix} \cos\theta & -\sin\theta & x \\ \sin\theta & \cos\theta & y \\ 0 & 0 & 1 \end{bmatrix}$$

In code each `SE2` instance stores three length-$n$ arrays `(x, y, theta)`.

### Composition

$$T_{AC} = T_{AB}\, T_{BC}$$

Expanded (avoids forming full matrices):

$$x_{AC} = x_{AB} + \cos\theta_{AB}\, x_{BC} - \sin\theta_{AB}\, y_{BC}$$
$$y_{AC} = y_{AB} + \sin\theta_{AB}\, x_{BC} + \cos\theta_{AB}\, y_{BC}$$
$$\theta_{AC} = \theta_{AB} + \theta_{BC}$$

### Inverse

$$T^{-1} = \begin{bmatrix} \cos\theta & \sin\theta & -\cos\theta\, x - \sin\theta\, y \\ -\sin\theta & \cos\theta & \sin\theta\, x - \cos\theta\, y \\ 0 & 0 & 1 \end{bmatrix}$$

### Exponential Map — $\exp: \mathfrak{se}(2) \to SE(2)$

Given a twist $\xi = (\rho_x,\, \rho_y,\, \theta)^\top \in \mathfrak{se}(2)$:

$$T = \exp(\xi^\wedge) = \begin{bmatrix} R(\theta) & V(\theta)\,\boldsymbol{\rho} \\ \mathbf{0} & 1 \end{bmatrix}$$

where the $V$ matrix recovers translation from the rotational component:

$$V(\theta) = \frac{\sin\theta}{\theta}\,I + \frac{1-\cos\theta}{\theta}\,J, \qquad J = \begin{bmatrix}0 & -1 \\ 1 & 0\end{bmatrix}$$

For $|\theta| < 10^{-8}$, the small-angle limit $V \approx I$ is used.

### Logarithmic Map — $\log: SE(2) \to \mathfrak{se}(2)$

$$\xi = \log(T) \implies \boldsymbol{\rho} = V^{-1}(\theta)\,\mathbf{t}, \quad \theta = \theta$$

$$V^{-1}(\theta) = \frac{1}{A^2+B^2}\begin{bmatrix} A & B \\ -B & A \end{bmatrix}, \quad A = \frac{\sin\theta}{\theta},\quad B = \frac{1-\cos\theta}{\theta}$$

---

## Dynamics Model

### State Vector

$$\mathbf{s} = \begin{bmatrix} x \\ y \\ \theta \\ \dot x_W \\ \dot y_W \\ \dot\theta \end{bmatrix} \in \mathbb{R}^{n \times 6}$$

### Force Architecture

Forces are split into two independently overridable components that are summed by the integrator:

| Method | Frame | Purpose |
|--------|-------|---------|
| `force_body_B(t, state)` | Body | Thrust, propulsion, or any force expressed in the body frame |
| `force_world_W(t, state)` | World | Gravity, centripetal, drag, or any force already in world frame |
| `force_total_W(t, state)` | World | **Used by integrator.** Rotates `force_body_B` into world frame and adds `force_world_W` |

The rotation from body to world frame:

$$\mathbf{F}^W = R(\theta)\,\mathbf{F}^B + \mathbf{F}^W_{\text{direct}}, \qquad R(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}$$

Subclasses override `force_body_B`, `force_world_W`, or both. The base class `force_total_W` combines them automatically.

### Equations of Motion

Newton–Euler on a planar rigid body:

$$\ddot x_W = \frac{F^W_{x,\text{total}}}{m}, \qquad \ddot y_W = \frac{F^W_{y,\text{total}}}{m}, \qquad \ddot\theta = \frac{\tau}{J}$$

> **Note:** Torque is frame-independent in 2D (always the z-component of the moment vector). A single `torque(t, state)` method is used with no frame suffix.

### State Derivative

$$\dot{\mathbf{s}} = \begin{bmatrix} \dot x_W \\ \dot y_W \\ \dot\theta \\ F^W_{x,\text{total}}/m \\ F^W_{y,\text{total}}/m \\ \tau/J \end{bmatrix}$$

### RK4 Integration

Each timestep uses the classical 4th-order Runge–Kutta rule:

$$k_1 = f(t,\, \mathbf{s})$$
$$k_2 = f\!\left(t+\tfrac{h}{2},\, \mathbf{s}+\tfrac{h}{2}k_1\right)$$
$$k_3 = f\!\left(t+\tfrac{h}{2},\, \mathbf{s}+\tfrac{h}{2}k_2\right)$$
$$k_4 = f\!\left(t+h,\, \mathbf{s}+h\,k_3\right)$$
$$\mathbf{s}_{k+1} = \mathbf{s}_k + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

Heading $\theta$ is wrapped to $(-\pi, \pi]$ after each step.

---

## Classes

### `SE2`

Vectorized batch of $n$ SE(2) transforms. Core storage: three `(n,)` arrays.

| Method | Description |
|--------|-------------|
| `SE2(x, y, theta)` | Construct from arrays |
| `SE2.identity(n)` | Batch of identity transforms |
| `SE2.from_matrix(mat)` | From `(n,3,3)` homogeneous matrices |
| `T.as_matrix()` | Returns `(n,3,3)` matrices |
| `A * B` | Composition $T_{AC} = T_{AB}\,T_{BC}$ |
| `T.apply(other)` | Alias for `T * other` |
| `T.inverse()` | Batch inverse |
| `T.log()` | Returns `(n,3)` twist array |
| `SE2.exp(xi)` | Returns SE2 from `(n,3)` twist array |

---

### `NoiseConfig`

```python
@dataclass
class NoiseConfig:
    imu_cov: np.ndarray   # (3,3) — [accel_x_B, accel_y_B, omega_B]
    pos_cov: np.ndarray   # (2,2) — [x_W, y_W]
    seed:    Optional[int]
```

---

### `RigidBodyTrajectory`

Container for all trajectory data across $n$ trials and $n_t$ timesteps.

| Field | Shape | Frame | Description |
|-------|-------|-------|-------------|
| `poses` | `list[SE2]` length $n_t$ | W | Each SE2 holds $n$ trials |
| `timestamps` | $(n_t,)$ | — | Time values |
| `velocity_W` | $(n, n_t, 2)$ | W | $[\dot x_W,\, \dot y_W]$ |
| `acceleration_W` | $(n, n_t, 2)$ | W | $[\ddot x_W,\, \ddot y_W]$ |
| `force_total_W_arr` | $(n, n_t, 2)$ | W | Total applied force |
| `angular_velocity` | $(n, n_t)$ | — | $\dot\theta$ |
| `angular_accel` | $(n, n_t)$ | — | $\ddot\theta$ |
| `torque` | $(n, n_t)$ | — | Applied torque |
| `accel_meas_B` | $(n, n_t, 2)$ | B | Noisy body-frame accelerometer |
| `gyro_meas_B` | $(n, n_t)$ | B | Noisy body-frame gyroscope |
| `pos_meas_W` | $(n, n_t, 2)$ | W | Noisy world-frame position |

---

### `RigidBodySim`

Base class. Subclasses override `force_body_B`, `force_world_W`, and/or `torque`.
`force_total_W` (called by the integrator) combines both force components automatically.

```python
sim = MySim(mass=1.0, inertia=0.5, noise_cfg=cfg, dt=0.02, n=20)
traj = sim.simulate(t_span=(0.0, 8.0), initial_state=s0)
```

**Override contract:**

| Method | Signature | Default | Typical use |
|--------|-----------|---------|-------------|
| `force_body_B(t, state)` | `→ (n,2)` | zeros | Thrust, propulsion |
| `force_world_W(t, state)` | `→ (n,2)` | zeros | Gravity, centripetal, drag |
| `force_total_W(t, state)` | `→ (n,2)` | rotates `force_body_B` + `force_world_W` | **Called by integrator — do not override** |
| `torque(t, state)` | `→ (n,)` | zeros | Heading controller |

---

## Simulation Implementations

### `CircleCenterFacing`

The body orbits at radius $r$ with tangential speed $v$. Heading is servo-controlled to always
point inward (toward the circle center).

**Centripetal force** (`force_world_W`):

$$\mathbf{F}^W = -\frac{m v^2}{r} \frac{[x,\, y]^\top}{\|[x,\, y]\|}$$

**Orientation controller** (PD on heading error):

$$\theta_{\text{des}} = \text{atan2}(-y, -x)$$
$$\tau = J\bigl(k_p\,\Delta\theta - k_d\,\dot\theta\bigr), \qquad k_p=10,\; k_d=2$$

---

### `CircleTangentFacing`

Same centripetal force (`force_world_W`) as above. Heading servo-controlled to face the velocity
direction (tangent to orbit, CCW):

$$\theta_{\text{des}} = \text{atan2}(-x,\; y)$$

---

### `SinusoidForward`

Traces a **vertical sine wave** in world space: the body advances in $+y$ on average while
oscillating in $x$, maintaining constant linear speed throughout.

**Desired velocity profile** (world frame):

$$\dot x_W^{\text{des}}(t) = A\cos(2\pi f t)$$
$$\dot y_W^{\text{des}}(t) = \sqrt{V^2 - \bigl(\dot x_W^{\text{des}}\bigr)^2}$$

where $V$ is the constant target speed and $A \leq V$ is the lateral velocity amplitude.
This decomposition guarantees $\|\mathbf{v}\| = V$ exactly at every instant.

**Velocity-tracking force** (`force_world_W`):

$$\mathbf{F}^W = m\, k_v\bigl(\mathbf{v}^{\text{des}} - \mathbf{v}\bigr), \qquad k_v = 8\;\text{s}^{-1}$$

**Heading controller** — aligns body x-axis with velocity direction:

$$\theta_{\text{des}} = \text{atan2}(\dot y_W,\, \dot x_W)$$
$$\tau = J\bigl(k_p\,\Delta\theta - k_d\,\dot\theta\bigr), \qquad k_p=12,\; k_d=3$$

The parametric curve $(x(t),\, y(t))$ converges to an upright sine wave as the
velocity controller tracks the desired profile.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `speed` | 2.0 m/s | Constant linear speed $V$ |
| `lat_amp` | 1.0 m/s | Lateral velocity amplitude $A$ (clamped to `speed`) |
| `lat_freq` | 0.3 Hz | Oscillation frequency $f$ |
| `kv` | 8.0 1/s | Velocity tracking gain |

---

### `RandomWalk`

Independent Gaussian force and torque drawn at every timestep (`force_world_W`):

$$\mathbf{F}^W \sim \mathcal{N}\!\left(\mathbf{0},\; \sigma_F^2 I\right)\cdot m$$
$$\tau \sim \mathcal{N}(0,\; \sigma_\tau^2)\cdot J$$

Each trial uses a shared RNG seeded from `NoiseConfig.seed` so runs are reproducible.

---

## Measurement Model

### IMU — body-frame accelerometer + gyroscope

The accelerometer measures specific force in the **body frame** (`accel_meas_B`):

$$\tilde{\mathbf{a}}_B = R(\theta)^\top\, \mathbf{a}^W + \boldsymbol{\eta}_a, \qquad \boldsymbol{\eta}_a \sim \mathcal{N}(\mathbf{0},\, \Sigma_{\text{imu}}[0{:}2,\, 0{:}2])$$

The gyroscope measures angular rate in the **body frame** (`gyro_meas_B`):

$$\tilde\omega_B = \dot\theta + \eta_\omega, \qquad \eta_\omega \sim \mathcal{N}(0,\, \Sigma_{\text{imu}}[2,2])$$

The full $3\times 3$ IMU covariance $\Sigma_{\text{imu}}$ is drawn jointly from `NoiseConfig.imu_cov`.

### Position — GPS / external (`pos_meas_W`)

$$\tilde{\mathbf{p}}_W = \begin{bmatrix}x\\y\end{bmatrix} + \boldsymbol{\eta}_p, \qquad \boldsymbol{\eta}_p \sim \mathcal{N}(\mathbf{0},\, \Sigma_{\text{pos}})$$

---

## Plotting Utilities

All functions in `plot_utils.py` return `plotly.graph_objects.Figure` objects and use a shared dark theme.

| Function | Description |
|----------|-------------|
| `plot_trajectory(traj)` | Static path for one trial with body-axis arrows at sampled poses |
| `animate_trajectory(traj)` | Frame-by-frame animation with play/pause slider |
| `plot_mc_paths(traj)` | All MC trial paths overlaid with a dashed mean path |
| `plot_imu_measurements(traj)` | 3-panel time series: $\tilde a_{x,B}$, $\tilde a_{y,B}$, $\tilde\omega_B$ |
| `plot_trajectory_with_bounds(traj)` | Noisy position measurements for all trials with ground truth and ±1σ tube |

### `plot_imu_measurements`

Three stacked subplots sharing a time axis. Each panel shows:
- **Thin coloured lines** — individual MC trial measurements
- **Orange solid line** — ground truth (trial 0, noise-free, rotated to body frame)
- **Purple dotted lines** — theoretical ±1σ bounds: `gt ± sqrt(imu_cov[i,i])`
- **Light purple fill** — ±1σ region

### `plot_trajectory_with_bounds`

Plots `pos_meas_W` (noisy position measurements) for all MC trials in world-frame x–y space:
- **Thin coloured lines** — individual trial measurement paths
- **Orange solid line** — ground truth trajectory (trial 0, from `poses`)
- **Purple dotted lines** — theoretical ±1σ tube boundary
- **Light purple fill** — ±1σ region

The tube half-width is the 1σ noise projected onto the path-normal direction at each timestep:

$$\sigma_n(t) = \sqrt{n_x^2\,\sigma_{p_x}^2 + n_y^2\,\sigma_{p_y}^2}$$

where $(n_x, n_y)$ is the unit normal to the ground truth path and $\sigma_{p_x}$, $\sigma_{p_y}$ are from `NoiseConfig.pos_cov`.

---

## Running the Demos

```bash
# All four demos, open in browser
python run_demo.py

# Individual demos
python run_demo.py circle_center
python run_demo.py circle_tangent
python run_demo.py sinusoid
python run_demo.py random_walk

# Save to HTML instead of opening browser
python run_demo.py --html
python run_demo.py sinusoid --html
```

Each demo produces one scrollable HTML page with five sections:

| Section | Plot function | Contents |
|---------|--------------|----------|
| Trajectory | `plot_trajectory` | Static path with body-axis arrows and pose markers |
| Animation | `animate_trajectory` | Animated playback with play/pause and slider |
| Monte Carlo Paths | `plot_mc_paths` | All MC trial paths + dashed mean path |
| IMU Measurements | `plot_imu_measurements` | Body-frame accel and gyro time series with ground truth and ±1σ |
| MC Trajectories ±1σ | `plot_trajectory_with_bounds` | Noisy position measurements with ground truth and theoretical ±1σ tube |

### Dependencies

```
numpy
plotly
```

Install with `pip install numpy plotly`.
