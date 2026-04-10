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
7. [Running the Demos](#running-the-demos)

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $n$    | Number of Monte Carlo trials (batch dimension) |
| $n_t$  | Number of timesteps |
| `A_T_B` | Transform from frame **B** to frame **A** (code convention) |
| $T_{AB}$ | Same transform in math notation |
| $\xi$  | Element of $\mathfrak{se}(2)$ — the Lie algebra twist |
| $\mathbf{s}$ | State vector $[x,\, y,\, \theta,\, \dot x,\, \dot y,\, \dot\theta]^\top$ |

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

$$\mathbf{s} = \begin{bmatrix} x \\ y \\ \theta \\ \dot x \\ \dot y \\ \dot\theta \end{bmatrix} \in \mathbb{R}^{n \times 6}$$

### Equations of Motion (world frame)

Newton–Euler on a planar rigid body:

$$\ddot x = \frac{F_x^w}{m}, \qquad \ddot y = \frac{F_y^w}{m}, \qquad \ddot\theta = \frac{\tau}{J}$$

Force in world frame from body-frame input:

$$\mathbf{F}^w = R(\theta)\,\mathbf{F}^b, \qquad R(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}$$

### State Derivative

$$\dot{\mathbf{s}} = \begin{bmatrix} \dot x \\ \dot y \\ \dot\theta \\ F_x^w/m \\ F_y^w/m \\ \tau/J \end{bmatrix}$$

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
    imu_cov: np.ndarray   # (3,3) — [accel_x, accel_y, omega]
    pos_cov: np.ndarray   # (2,2) — [x, y]
    seed:    Optional[int]
```

---

### `RigidBodyTrajectory`

Container for all trajectory data across $n$ trials and $n_t$ timesteps.

| Field | Shape | Description |
|-------|-------|-------------|
| `poses` | `list[SE2]` length $n_t$ | Each SE2 holds $n$ trials |
| `timestamps` | $(n_t,)$ | Time values |
| `velocity` | $(n, n_t, 2)$ | World-frame $[\dot x,\, \dot y]$ |
| `acceleration` | $(n, n_t, 2)$ | World-frame $[\ddot x,\, \ddot y]$ |
| `force_world_arr` | $(n, n_t, 2)$ | Applied world-frame force |
| `angular_velocity` | $(n, n_t)$ | $\dot\theta$ |
| `angular_accel` | $(n, n_t)$ | $\ddot\theta$ |
| `torque` | $(n, n_t)$ | Applied torque |
| `accel_meas` | $(n, n_t, 2)$ | Noisy **body-frame** acceleration |
| `gyro_meas` | $(n, n_t)$ | Noisy angular velocity |
| `pos_meas` | $(n, n_t, 2)$ | Noisy world-frame position |

---

### `RigidBodySim`

Base class. Subclasses override `force_body` / `torque` (or `force_world`).

```python
sim = MySim(mass=1.0, inertia=0.5, noise_cfg=cfg, dt=0.02, n=20)
traj = sim.simulate(t_span=(0.0, 8.0), initial_state=s0)
```

**Override contract:**

| Method | Signature | Default |
|--------|-----------|---------|
| `force_body(t, state)` | `→ (n,2)` | zeros |
| `torque(t, state)` | `→ (n,)` | zeros |
| `force_world(t, state)` | `→ (n,2)` | rotates `force_body` |

> **Note:** Torque is frame-independent in 2D (it is always the z-component of the moment vector), so `torque_body` and `torque_world` are unified into a single `torque` method.

---

## Simulation Implementations

### `CircleCenterFacing`

The body orbits at radius $r$ with tangential speed $v$. Heading is servo-controlled to always
point inward (toward the circle center).

**Centripetal force:**

$$\mathbf{F}^w = -\frac{m v^2}{r} \frac{[x,\, y]^\top}{\|[x,\, y]\|}$$

**Orientation controller** (PD on heading error):

$$\theta_{\text{des}} = \text{atan2}(-y, -x)$$
$$\tau = J\bigl(k_p\,\Delta\theta - k_d\,\dot\theta\bigr), \qquad k_p=10,\; k_d=2$$

---

### `CircleTangentFacing`

Same centripetal force as above. Heading servo-controlled to face the velocity direction
(tangent to orbit, CCW):

$$\theta_{\text{des}} = \text{atan2}(-x,\; y)$$

---

### `SinusoidForward`

Constant body-x thrust with sinusoidal body-y lateral force. Heading tracks velocity direction.

$$\mathbf{F}^b = \begin{bmatrix} F_{\text{fwd}} \\ A\sin(2\pi f t) \end{bmatrix}$$

$$\theta_{\text{des}} = \text{atan2}(\dot y,\, \dot x)$$

---

### `RandomWalk`

Independent Gaussian force and torque drawn at every timestep:

$$\mathbf{F}^w \sim \mathcal{N}\!\left(\mathbf{0},\; \sigma_F^2 I\right)\cdot m$$
$$\tau \sim \mathcal{N}(0,\; \sigma_\tau^2)\cdot J$$

Each trial uses a shared RNG seeded from `NoiseConfig.seed` so runs are reproducible.

---

## Measurement Model

### IMU (body-frame accelerometer + gyro)

$$\tilde{\mathbf{a}}_b = R(\theta)^\top\, \mathbf{a}^w + \boldsymbol{\eta}_a, \qquad \boldsymbol{\eta}_a \sim \mathcal{N}(\mathbf{0},\, \Sigma_{\text{imu}}[0{:}2,\, 0{:}2])$$

$$\tilde\omega = \dot\theta + \eta_\omega, \qquad \eta_\omega \sim \mathcal{N}(0,\, \Sigma_{\text{imu}}[2,2])$$

The full $3\times 3$ IMU covariance $\Sigma_{\text{imu}}$ is drawn jointly.

### Position (GPS / external)

$$\tilde{\mathbf{p}} = \begin{bmatrix}x\\y\end{bmatrix} + \boldsymbol{\eta}_p, \qquad \boldsymbol{\eta}_p \sim \mathcal{N}(\mathbf{0},\, \Sigma_{\text{pos}})$$

---

## Running the Demos

```bash
# All four demos
python run_demo.py

# Individual demos
python run_demo.py circle_center
python run_demo.py circle_tangent
python run_demo.py sinusoid
python run_demo.py random_walk
```

Each demo produces one scrollable HTML page with three sections:

| Section | Contents |
|---------|----------|
| Trajectory | Static path with body-axis arrows and pose markers |
| Animation | Animated playback with play/pause and slider |
| Monte Carlo Paths | All MC trial paths + mean path |

### Dependencies

```
numpy
plotly
```

Install with `pip install numpy plotly`.
