# Kalman Filtering Math
A document describing the mathematics behind the Kalman filters implemented for this project. This will describe the EKF (Extended Kalman Filter) and the MEKF/IEKF (Multiplicative/Invariant Kalman Filter).

---

## Table of Contents

TODO

---

## Notation

TODO

---

## Dynamics Model

### Measurement model
### Inertial (IMU)
The accelerometer measures specific force in the **body frame** (`accel_meas_B`):

$$\tilde{\mathbf{a}}_B = R(\theta)^\top\, \mathbf{a}^W + \boldsymbol{\eta}_a, \qquad \boldsymbol{\eta}_a \sim \mathcal{N}(\mathbf{0},\, \Sigma_{\text{imu}}[0{:}2,\, 0{:}2])$$

The gyroscope measures angular rate in the **body frame** (`gyro_meas_B`):

$$\tilde\omega_B = \dot\theta + \eta_\omega, \qquad \eta_\omega \sim \mathcal{N}(0,\, \Sigma_{\text{imu}}[2,2])$$

The full $3\times 3$ IMU covariance $\Sigma_{\text{imu}}$ is drawn jointly from `NoiseConfig.imu_cov`.

### Position — GPS / external (`pos_meas_W`)

$$\tilde{\mathbf{p}}_W = \begin{bmatrix}x\\y\end{bmatrix} + \boldsymbol{\eta}_p, \qquad \boldsymbol{\eta}_p \sim \mathcal{N}(\mathbf{0},\, \Sigma_{\text{pos}})$$

### Deterministic Continuous Time Dynamics
We will use the following vectorization for our state $s_k$ inputs and measurements. The input u is in the "Body" frame $B$:

$$\mathbf{s} = \begin{bmatrix} x\\ y\\ \theta \\ \dot x \\ \dot y \end{bmatrix}$$

$$\mathbf{u} = \begin{bmatrix} \ddot x_B \\ \ddot y_B \\ \dot \theta \end{bmatrix} = \begin{bmatrix} \mathbf{a}_B \\ \dot \theta \end{bmatrix}$$

$$\mathbf{z} = \begin{bmatrix} x \\ y \end{bmatrix}$$

With the deterministic continuous time dynamics as follows:

$$\dot{\mathbf{s}} = f(\mathbf{u}, \mathbf{s}) = \begin{bmatrix} \dot x \\ \dot y \\ \dot \theta \\ \ddot x \\ \ddot y \end{bmatrix} = \begin{bmatrix} \dot x \\ \dot y \\ \dot \theta \\ R^W_B(\theta) \mathbf{a}_B \end{bmatrix} = \begin{bmatrix} \mathbf{0}_{1 \times 3} \ 1 \ 0 \\ \mathbf{0}_{1 \times 3} \ 0 \ 1 \\ \mathbf{0}_{3 \times 5}\end{bmatrix} \mathbf{s} + \begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \\ 1\end{bmatrix} R^W_B(\theta) \mathbf{a}_B + \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 0 \end{bmatrix} \dot \theta$$

$$\dot{\mathbf{s}} = \begin{bmatrix} \mathbf{0}_{1 \times 3} \ 1 \ 0 \\ \mathbf{0}_{1 \times 3} \ 0 \ 1 \\ \mathbf{0}_{3 \times 5}\end{bmatrix} \mathbf{s} + \begin{bmatrix} \mathbf{0}_{2 \times 2} \ \mathbf{0}_{2 \times 1}\\ 0 \ \ \ \ \ \ 1 \\ R^W_B(\theta) \ \mathbf{0}_{2 \times 2} \end{bmatrix} \mathbf{u}$$

Measurement model:

$$\mathbf{z} = \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 1 \ 0 \ 0 \ 0 \ 0  \\ 0 \ 1 \ 0 \ 0 \ 0\end{bmatrix} \mathbf{s}$$

### Putting in discrete time with forward euler

$$\mathbf{s}_k = \mathbf{A} \mathbf{s}_{k-1} + \mathbf{B}(\mathbf{s}_{k-1}) \mathbf{u}_{k-1} = \Delta t \begin{bmatrix} \mathbf{0}_{1 \times 3} \ 1 \ 0 \\ \mathbf{0}_{1 \times 3} \ 0 \ 1 \\ \mathbf{0}_{3 \times 5}\end{bmatrix} \mathbf{s}_{k-1} + \Delta t \begin{bmatrix} \mathbf{0}_{2 \times 2} \ \mathbf{0}_{2 \times 1}\\ 0 \ \ \ \ \ \ 1 \\ R^W_B(\theta _ {k-1}) \ \mathbf{0}_{2 \times 2} \end{bmatrix} \mathbf{u}_{k-1}$$

$$$$

$$\mathbf{z}_k = H \mathbf{s}_k = \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 1 \ 0 \ 0 \ 0 \ 0  \\ 0 \ 1 \ 0 \ 0 \ 0\end{bmatrix} \mathbf{s}_k$$

### Add in stochasticicity
$$\mathbf{s}_k = \mathbf{A} \mathbf{s}_{k-1} + \mathbf{B}(\mathbf{s}_{k-1}) (\mathbf{u}_{k-1} + \mathbf{w}_{k-1}) = \Delta t \begin{bmatrix} \mathbf{0}_{1 \times 3} \ 1 \ 0 \\ \mathbf{0}_{1 \times 3} \ 0 \ 1 \\ \mathbf{0}_{3 \times 5}\end{bmatrix} \mathbf{s}_{k-1} + \Delta t \begin{bmatrix} \mathbf{0}_{2 \times 2} \ \mathbf{0}_{2 \times 1}\\ 0 \ \ \ \ \ \ 1 \\ R^W_B(\theta) \ \mathbf{0}_{2 \times 2} \end{bmatrix} (\mathbf{u}_{k-1} + \mathbf{w}_{k-1})$$

$$\mathbf{w}_k = \begin{bmatrix} 0 \\ 0 \\\mathbf{\mathbf{w}_{\text{imu}}} \end{bmatrix} \qquad \mathbf{w}_{\text{imu}} \sim \mathcal{N}(\mathbf{0},\, \Sigma_{\text{imu}})$$

$$\mathbf{w}_k \sim \mathcal{N}(\mathbf{0}_{5 \times 1}, Q) \qquad Q = \begin{bmatrix} \mathbf{0}_{2 \times 2} \ \mathbf{0}_{2 \times 3} \\ \mathbf{0}_{3 \times 2} \Sigma_{\text{imu}} \end{bmatrix}$$

$$\mathbf{z}_k = \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 1 \ 0 \ 0 \ 0 \ 0  \\ 0 \ 1 \ 0 \ 0 \ 0\end{bmatrix} \mathbf{s}_k + \mathbf{v}_k$$

$$\boldsymbol{\eta}_p = \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, R) \qquad R = \Sigma_{\text{pos}}$$

---

## Extended Kalman Filter Design
Bellow details the math used in the Kalman Filter.

Notation: ' indicates the value before the Kalman filter update. Often denoted elsewhere using a superscript -.

### Important Matrix Definitions
Matrices used in calculation: A, B, H

### Initialization
The Kalman filter will be initialized using the very first measurement step and the initial estimated covariance will be very high. The velocity terms / not measured terms will be set to 0. Include a parameter on how large the variance should be set to. Then run the first update step on the measurements.

TODO EQUATION

### Update Step and Kalman Gain
Innovation:
$$\mathbf{v}_k = \mathbf{z}_k - \mathbf{H} \mathbf{s}_k'$$

Innovation Covariance:
$$\mathbf{S_k} = \mathbf{H} \mathbf{P}'_k \mathbf{H}^T + \mathbf{R}_k$$

Kalman Gain:
$$\mathbf{K}_k = \mathbf{P}'_k \mathbf{H}^T \mathbf{S}_k^{-1}$$

Update:
$$\mathbf{s}_k = \mathbf{s}_k' + \mathbf{K}_k \mathbf{v}_k$$

Corrected Covariance (Numerically Stable Form):
$$\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_k' (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^T + \mathbf{K}_k \mathbf{R_k} \mathbf{K}_k^T$$

### Predict / Propagate Step
Mean update:
$$\mathbf{s}_{k + 1}' = \mathbf{A}_k \mathbf{s}_k+ \mathbf{B}_k \mathbf{u}_k$$

Covariance update:
$$\mathbf{P}_{k + 1}' = \mathbf{A}_k \mathbf{P}_k \mathbf{A}_k^T + \mathbf{B}_k \mathbf{Q}_k \mathbf{B}_k^T$$

---

## Extended Kalman Filter Class
The EKF will be implemented in `EKF.py` and will define a class for keeping track of the state of the EKF and functions to call to evaluate the EKF in a vectorized way.

Init method:
run_ekf

private:
ekf update
ekf propagate

