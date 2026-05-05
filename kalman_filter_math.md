# Kalman Filtering Math
A document describing the mathematics behind the Kalman filters implemented for this project. This will describe the EKF (Extended Kalman Filter) and the MEKF/IEKF (Multiplicative/Invariant Kalman Filter).

---

## Table of Contents

1. [Notation](#notation)
2. [Dynamics Model](#dynamics-model)
3. [Extended Kalman Filter Design](#extended-kalman-filter-design)
4. [Extended Kalman Filter Class](#extended-kalman-filter-class)

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $k$ | Discrete time index |
| $\Delta t$ | Timestep size [s] |
| $n$ | Number of Monte Carlo trials |
| $\mathbf{s}_k$ | State vector at step $k$ |
| $\mathbf{u}_k$ | Control input (IMU reading) at step $k$ |
| $\mathbf{z}_k$ | GPS measurement vector at step $k$ |
| $\hat{\mathbf{s}}_k^-$ | Prior (predicted) state estimate — before the measurement update |
| $\hat{\mathbf{s}}_k^+$ | Posterior state estimate — after the measurement update |
| $P_k^-$ | Prior state covariance |
| $P_k^+$ | Posterior state covariance |
| $R^W_B(\theta)$ | $2\times 2$ rotation matrix mapping vectors from body frame $B$ to world frame $W$ |
| $\tilde{(\cdot)}$ | Noisy (measured) version of a quantity |
| `_W` | Quantity expressed in the world frame |
| `_B` | Quantity expressed in the body frame |
| $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ | Gaussian distribution with mean $\boldsymbol{\mu}$ and covariance $\Sigma$ |

---

## Dynamics Model

### Coordinate Frames and Rotation

$R^W_B(\theta)$ is the 2D rotation matrix that maps vectors from body frame $B$ into world frame $W$:

$$R^W_B(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

Its transpose $\bigl(R^W_B\bigr)^\top = R^B_W$ maps from world frame to body frame.

### State, Input, and Measurement Vectors

$$\mathbf{s} = \begin{bmatrix} x\\ y\\ \theta \\ \dot x_W \\ \dot y_W \end{bmatrix} \in \mathbb{R}^5, \qquad
\mathbf{u} = \begin{bmatrix} a_{x,B} \\ a_{y,B} \\ \dot\theta \end{bmatrix} = \begin{bmatrix} \mathbf{a}_B \\ \dot\theta \end{bmatrix} \in \mathbb{R}^3, \qquad
\mathbf{z} = \begin{bmatrix} x \\ y \\ \theta \end{bmatrix} \in \mathbb{R}^3$$

The input $\mathbf{u}$ is the raw IMU reading: body-frame linear acceleration $\mathbf{a}_B$ from the accelerometer and angular rate $\dot\theta$ from the gyroscope. Both are measured in the body frame.

The measurement $\mathbf{z}$ combines the GPS position and a heading sensor (e.g. magnetometer or compass-derived heading) — both observe state components directly.

### Deterministic Continuous-Time Dynamics

$$\dot{\mathbf{s}} = f(\mathbf{s}, \mathbf{u}) = A_c\, \mathbf{s} + B_c(\theta)\, \mathbf{u}$$

where the continuous-time matrices are:

$$A_c = \begin{bmatrix} 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 \end{bmatrix}, \qquad
B_c(\theta) = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 1 \\ \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \end{bmatrix}$$

In expanded form the state derivative is:

$$\dot{\mathbf{s}} = \begin{bmatrix} \dot x_W \\ \dot y_W \\ \dot\theta \\ R^W_B(\theta)\, \mathbf{a}_B \end{bmatrix}$$

The top two rows of $A_c$ couple $[\dot x_W, \dot y_W]$ from the velocity states; the input matrix $B_c$ applies $\dot\theta$ directly to $\dot\theta$ (row 3) and rotates the body-frame acceleration into world-frame acceleration (rows 4–5).

### Measurement Model

The GPS sensor measures $x, y$ and the heading sensor measures $\theta$ directly. All three are linear functions of the state, so the measurement Jacobian is exact:

$$\mathbf{z} = H\, \mathbf{s}, \qquad H = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 \end{bmatrix}$$

### Discrete-Time Dynamics (Forward Euler)

Applying forward Euler ($\mathbf{s}_k \approx \mathbf{s}_{k-1} + \Delta t\, \dot{\mathbf{s}}_{k-1}$):

$$\mathbf{s}_k = \mathbf{A}\, \mathbf{s}_{k-1} + \mathbf{B}(\theta_{k-1})\, \mathbf{u}_{k-1}$$

$$\mathbf{A} = I_5 + \Delta t\, A_c = \begin{bmatrix} 1 & 0 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & 0 & \Delta t \\ 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 1 \end{bmatrix}$$

$$\mathbf{B}(\theta_{k-1}) = \Delta t\, B_c(\theta_{k-1}) = \Delta t \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 1 \\ \cos\theta_{k-1} & -\sin\theta_{k-1} & 0 \\ \sin\theta_{k-1} & \cos\theta_{k-1} & 0 \end{bmatrix}$$

$$\mathbf{z}_k = H\, \mathbf{s}_k$$

> **Note:** The discrete state matrix $\mathbf{A} = I + \Delta t\, A_c$ includes the identity term that carries the current state forward — this is the standard result of discretizing $\dot{\mathbf{s}} = A_c \mathbf{s} + \ldots$ via forward Euler.

### Stochastic Dynamics

The IMU readings used as control inputs are corrupted by additive Gaussian noise. The 3D input noise $\mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \Sigma_{\text{imu}})$ enters through the $\mathbf{B}$ matrix:

$$\mathbf{s}_k = \mathbf{A}\, \mathbf{s}_{k-1} + \mathbf{B}(\theta_{k-1})\bigl(\mathbf{u}_{k-1} + \mathbf{w}_{k-1}\bigr), \qquad
\mathbf{w}_{k} \sim \mathcal{N}(\mathbf{0},\, \Sigma_{\text{imu}})$$

$$\Sigma_{\text{imu}} = \begin{bmatrix} \sigma_{a_x}^2 & 0 & 0 \\ 0 & \sigma_{a_y}^2 & 0 \\ 0 & 0 & \sigma_\omega^2 \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$

This is equivalent to an additive state-space noise $\tilde{\mathbf{w}}_k = \mathbf{B}_{k-1}\, \mathbf{w}_{k-1} \sim \mathcal{N}(\mathbf{0},\, Q_k)$, where the process noise covariance in state space is:

$$Q_k = \mathbf{B}(\theta_{k-1})\, \Sigma_{\text{imu}}\, \mathbf{B}(\theta_{k-1})^\top \in \mathbb{R}^{5 \times 5}$$

The GPS and heading measurements are corrupted by independent noise. Stacking position and heading into a single 3D measurement gives a block-diagonal covariance:

$$\mathbf{z}_k = H\, \mathbf{s}_k + \mathbf{v}_k, \qquad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0},\, R)$$

$$R = \begin{bmatrix} \Sigma_{\text{pos}} & \mathbf{0}_{2\times 1} \\ \mathbf{0}_{1\times 2} & \sigma_\theta^2 \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$

The cross-terms are zero because the position and heading sensors are independent. $\sigma_\theta^2$ is the heading-measurement variance (`heading_var` in `NoiseConfig`), kept separate from `pos_cov` so the rotation update can be toggled independently in EKF experiments.

### Sensor Measurement Models

**Accelerometer** — measures specific force in the body frame (`accel_meas_B`):

$$\tilde{\mathbf{a}}_B = R^W_B(\theta)^\top\, \mathbf{a}_W + \boldsymbol{\eta}_a, \qquad \boldsymbol{\eta}_a \sim \mathcal{N}\!\left(\mathbf{0},\, \Sigma_{\text{imu}}[0{:}2,\, 0{:}2]\right)$$

**Gyroscope** — measures angular rate in the body frame (`gyro_meas_B`):

$$\tilde\omega_B = \dot\theta + \eta_\omega, \qquad \eta_\omega \sim \mathcal{N}\!\left(0,\, \Sigma_{\text{imu}}[2,2]\right)$$

**GPS/position** — measures position in the world frame (`pos_meas_W`):

$$\tilde{\mathbf{p}}_W = \begin{bmatrix}x \\ y\end{bmatrix} + \boldsymbol{\eta}_p, \qquad \boldsymbol{\eta}_p \sim \mathcal{N}(\mathbf{0},\, \Sigma_{\text{pos}})$$

**Heading** — measures absolute heading in the world frame (`heading_meas_W`):

$$\tilde\theta_W = \theta + \eta_\theta, \qquad \eta_\theta \sim \mathcal{N}\!\left(0,\, \sigma_\theta^2\right)$$

The heading measurement is wrapped to $(-\pi, \pi]$ after the noise is added.

---

## Extended Kalman Filter Design

The dynamics $g(\mathbf{s}_{k-1}, \mathbf{u}_{k-1}) = \mathbf{A}\,\mathbf{s}_{k-1} + \mathbf{B}(\theta_{k-1})\,\mathbf{u}_{k-1}$ are nonlinear because $\mathbf{B}$ depends on $\theta \in \mathbf{s}$. The EKF handles this by linearizing around the current estimate at each step.

### EKF Jacobians

**State (process) Jacobian** — linearization of the predict step with respect to the state:

$$F_k = \frac{\partial g}{\partial \mathbf{s}}\bigg|_{\hat{\mathbf{s}}_k^+,\, \mathbf{u}_k} = \mathbf{A} + \Delta t \begin{bmatrix} 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & {-\sin\hat\theta\, a_{x,B} - \cos\hat\theta\, a_{y,B}} & 0 & 0 \\ 0 & 0 & {\phantom{-}\cos\hat\theta\, a_{x,B} - \sin\hat\theta\, a_{y,B}} & 0 & 0 \end{bmatrix}$$

where $\hat\theta = [\hat{\mathbf{s}}_k^+]_3$ is the current heading estimate and $[a_{x,B},\, a_{y,B}] = [\mathbf{u}_k]_{1:2}$ are the accelerometer readings used as input. The extra column-3 terms come from $\partial(R^W_B(\theta)\,\mathbf{a}_B)/\partial\theta$.

**Measurement Jacobian** — the GPS + heading observation is linear, so no approximation is required:

$$H = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 \end{bmatrix}$$

### Initialization

Given the first GPS measurement $\tilde{\mathbf{p}}_0$ and heading measurement $\tilde\theta_0$, the initial posterior is:

$$\hat{\mathbf{s}}_0^+ = \begin{bmatrix} \tilde p_{x,0} \\ \tilde p_{y,0} \\ \tilde\theta_0 \\ 0 \\ 0 \end{bmatrix}, \qquad
P_0^+ = \begin{bmatrix} \Sigma_{\text{pos}} & \mathbf{0}_{2\times 3} \\ \mathbf{0}_{3\times 2} & \sigma_0^2\, I_3 \end{bmatrix}$$

The velocity states $[\dot x_W, \dot y_W]$ and heading $\theta$ are not directly observed from GPS alone, so they are initialized to zero with a large variance $\sigma_0^2 \gg 1$ that encodes our initial ignorance. After the first GPS update the filter is run normally.

### Predict / Propagate Step

Given posterior $(\hat{\mathbf{s}}_k^+,\, P_k^+)$ and IMU input $\mathbf{u}_k$:

**Mean propagation:**
$$\hat{\mathbf{s}}_{k+1}^- = \mathbf{A}\, \hat{\mathbf{s}}_k^+ + \mathbf{B}(\hat\theta_k^+)\, \mathbf{u}_k$$

**Covariance propagation:**
$$P_{k+1}^- = F_k\, P_k^+\, F_k^\top + Q_k$$

where $Q_k = \mathbf{B}(\hat\theta_k^+)\, \Sigma_{\text{imu}}\, \mathbf{B}(\hat\theta_k^+)^\top$ maps the IMU input noise into state space.

### Update Step and Kalman Gain

Given prior $(\hat{\mathbf{s}}_k^-,\, P_k^-)$ and stacked measurement $\mathbf{z}_k = [\tilde p_x, \tilde p_y, \tilde\theta]^\top$:

**Innovation:**
$$\boldsymbol{\nu}_k = \mathbf{z}_k - H\, \hat{\mathbf{s}}_k^-$$

The third (heading) component must be wrapped to $(-\pi, \pi]$ to handle the angular discontinuity at $\pm\pi$:

$$\nu_{\theta,k} \leftarrow \text{wrap}(\nu_{\theta,k}), \qquad \text{wrap}(\alpha) = ((\alpha + \pi) \bmod 2\pi) - \pi$$

Without this step a tiny true error across the wrap boundary (e.g. $\hat\theta = \pi - 0.01$ vs. $\tilde\theta = -\pi + 0.01$) would be interpreted as a $\sim 2\pi$ innovation and corrupt the update.

**Innovation Covariance:**
$$S_k = H\, P_k^-\, H^\top + R$$

**Kalman Gain:**
$$K_k = P_k^-\, H^\top\, S_k^{-1}$$

**State Update:**
$$\hat{\mathbf{s}}_k^+ = \hat{\mathbf{s}}_k^- + K_k\, \boldsymbol{\nu}_k$$

**Covariance Update (Joseph form — numerically stable):**
$$P_k^+ = (I - K_k H)\, P_k^-\, (I - K_k H)^\top + K_k\, R\, K_k^\top$$

The Joseph form is preferred over the simpler $(I - K_k H) P_k^-$ because it maintains symmetry and positive semi-definiteness even under finite-precision arithmetic.

---

## Extended Kalman Filter Class

The EKF is implemented in `EKF.py`. The class is **vectorized over $n$ Monte Carlo trials** — every state, covariance, and Jacobian carries a leading batch axis so the filter can be run on the output of a Monte Carlo simulation in a single call.

### Vectorized Shapes

| Quantity | Shape | Notes |
|----------|-------|-------|
| `s_hat` | $(n, 5)$ | Per-trial state estimate |
| `P` | $(n, 5, 5)$ | Per-trial covariance |
| `A`, `H` | $(5, 5)$, $(3, 5)$ | Constant — broadcast across trials |
| `B(θ)`, `F` | $(n, 5, 3)$, $(n, 5, 5)$ | Depend on per-trial $\theta$ |
| `Q = B Σ_imu Bᵀ` | $(n, 5, 5)$ | Per-trial process noise covariance |
| `R` | $(3, 3)$ | Constant — block diagonal of `Σ_pos` and `σ_θ²` |
| `z_seq` | $(n, n_t, 3)$ | Stacked measurements: `[x, y, θ]` per trial per step |
| `u_seq` | $(n, n_t, 3)$ | IMU readings: `[a_xB, a_yB, ω]` per trial per step |

### Class Interface

```python
class EKF:
    # Per-trial running state — leading axis is the MC batch
    s_hat: np.ndarray    # (n, 5)    state estimate per trial
    P:     np.ndarray    # (n, 5, 5) covariance per trial

    def __init__(
        self,
        z0:        np.ndarray,   # (n, 3)   first measurement per trial [x, y, θ]
        sigma0:    float,         # large initial std dev for velocity states
        dt:        float,
        Sigma_imu: np.ndarray,   # (3, 3)   IMU noise covariance
        Sigma_pos: np.ndarray,   # (2, 2)   GPS noise covariance
        sigma_theta: float,       # heading-measurement std dev [rad]
    ): ...

    def run(
        self,
        z_seq: np.ndarray,   # (n, nt, 3)   measurements
        u_seq: np.ndarray,   # (n, nt, 3)   IMU readings
    ) -> tuple[np.ndarray, np.ndarray]:
        # Returns (s_hist, P_hist) of shape (n, nt, 5) and (n, nt, 5, 5)
        ...

    def _propagate(self, u: np.ndarray) -> None:
        # u: (n, 3)  — advances (s_hat, P) using batched IMU input
        ...

    def _update(self, z: np.ndarray) -> None:
        # z: (n, 3)  — corrects (s_hat, P) using batched measurement
        ...
```

### Method Summaries

| Method | Role |
|--------|------|
| `__init__` | Initialize $\hat{\mathbf{s}}_0^+$ from first measurement (per trial); set $P_0^+$ with `Sigma_pos` for position states, `sigma_theta²` for heading, and $\sigma_0^2$ for the unobserved velocity states |
| `run` | Loop over time steps: call `_propagate` with the batched IMU reading, then `_update` with the batched measurement; record full $(n, n_t, \cdot)$ history |
| `_propagate` | Build batched $F_k, \mathbf{B}_k$; advance $\hat{\mathbf{s}}$ via $\mathbf{A}\hat{\mathbf{s}} + \mathbf{B}_k \mathbf{u}$; advance $P$ via $F_k P F_k^\top + \mathbf{B}_k \Sigma_{\text{imu}} \mathbf{B}_k^\top$, all with `np.einsum` |
| `_update` | Compute innovation $\boldsymbol{\nu}_k$ (with heading wrap), innovation covariance $S_k$, Kalman gain $K_k = P H^\top S^{-1}$ via batched solve, then Joseph-form covariance update |
