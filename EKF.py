"""
Extended Kalman Filter for SE(2) rigid body motion, vectorized over n MC trials.

State : s = [x, y, theta, vx_W, vy_W]
Input : u = [a_xB, a_yB, omega]                (IMU)
Meas  : z = [x, y, theta]                       (GPS + heading)

All filter quantities carry a leading batch axis of size n so a full
Monte Carlo sweep runs in a single `run()` call.
"""

from __future__ import annotations

import numpy as np


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class EKF:
    """
    Vectorized EKF — runs n filters in parallel via batched numpy ops.

    Shapes
    ------
    s_prior : (n, 5)        predicted state  (after propagation, before update)
    s_hat   : (n, 5)        corrected state  (after measurement update)
    P_prior : (n, 5, 5)     predicted covariance
    P       : (n, 5, 5)     corrected covariance
    """

    def __init__(
        self,
        z0:          np.ndarray,    # (n, 3)  first measurement per trial [x, y, theta]
        sigma0:      float,          # large std dev for unobserved velocity states
        dt:          float,
        Sigma_imu:   np.ndarray,    # (3, 3) IMU noise covariance
        Sigma_pos:   np.ndarray,    # (2, 2) GPS noise covariance
        sigma_theta: float,          # heading-measurement std dev [rad]
    ):
        z0 = np.atleast_2d(np.asarray(z0, float))
        n  = z0.shape[0]

        self.n         = n
        self.dt        = dt
        self.Sigma_imu = np.asarray(Sigma_imu, float)
        self.Sigma_pos = np.asarray(Sigma_pos, float)
        self.sigma_theta = float(sigma_theta)

        # Initial posterior state: position and heading from first measurement, velocities = 0
        self.s_hat = np.zeros((n, 5))
        self.s_hat[:, 0:2] = z0[:, 0:2]
        self.s_hat[:, 2]   = _wrap(z0[:, 2])

        # Initial posterior covariance: known position+heading, large variance on velocity
        P0 = np.zeros((5, 5))
        P0[0:2, 0:2] = self.Sigma_pos
        P0[2, 2]     = self.sigma_theta ** 2
        P0[3, 3]     = sigma0 ** 2
        P0[4, 4]     = sigma0 ** 2
        self.P = np.broadcast_to(P0, (n, 5, 5)).copy()

        # Constant matrices
        self.A = np.eye(5)
        self.A[0, 3] = dt
        self.A[1, 4] = dt

        self.H = np.zeros((3, 5))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.R = np.zeros((3, 3))
        self.R[0:2, 0:2] = self.Sigma_pos
        self.R[2, 2]     = self.sigma_theta ** 2

    # ------------------------------------------------------------------
    # Internal helpers — build per-trial Jacobians
    # ------------------------------------------------------------------

    def _B_matrix(self, theta: np.ndarray) -> np.ndarray:
        """
        Discrete-time input matrix B(theta) = dt * B_c(theta).
        theta : (n,)   per-trial heading
        returns: (n, 5, 3)
        """
        n  = theta.shape[0]
        c  = np.cos(theta)
        s  = np.sin(theta)
        dt = self.dt

        B = np.zeros((n, 5, 3))
        B[:, 2, 2] = dt
        B[:, 3, 0] =  dt * c
        B[:, 3, 1] = -dt * s
        B[:, 4, 0] =  dt * s
        B[:, 4, 1] =  dt * c
        return B

    def _F_jacobian(self, theta: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        State Jacobian F = A + dt * d(B(theta) u)/d(theta)  evaluated per trial.
        theta : (n,)
        u     : (n, 3)   [a_xB, a_yB, omega]
        returns: (n, 5, 5)
        """
        n  = theta.shape[0]
        c  = np.cos(theta)
        s  = np.sin(theta)
        dt = self.dt
        ax = u[:, 0]
        ay = u[:, 1]

        F = np.broadcast_to(self.A, (n, 5, 5)).copy()
        # d(R(theta) a_B)/d(theta) goes into column 3 of rows 4 and 5
        F[:, 3, 2] = dt * (-s * ax - c * ay)
        F[:, 4, 2] = dt * ( c * ax - s * ay)
        return F

    # ------------------------------------------------------------------
    # Predict: writes s_hat, P  (the prior for the next update)
    # ------------------------------------------------------------------

    def _propagate(self, u: np.ndarray) -> None:
        """
        Predict step.  u : (n, 3)
        Reads  self.s_hat, self.P  (posterior from previous step).
        Writes self.s_hat, self.P  (prior, ready for _update).
        """
        theta = self.s_hat[:, 2]
        B = self._B_matrix(theta)              # (n, 5, 3)
        F = self._F_jacobian(theta, u)         # (n, 5, 5)

        # s⁻ = A s + B u
        self.s_hat = (
            np.einsum("ij,nj->ni", self.A, self.s_hat)   # A s
            + np.einsum("nij,nj->ni", B, u)               # B u
        )
        self.s_hat[:, 2] = _wrap(self.s_hat[:, 2])

        # P- = F P F^T + B Σ_imu B^T
        FP    = np.einsum("nij,njk->nik", F, self.P)          # F P
        FPFt  = np.einsum("nij,nkj->nik", FP, F)              # F P F^T
        BSig  = np.einsum("nij,jk->nik", B, self.Sigma_imu)   # B Σ_imu
        Q     = np.einsum("nij,nkj->nik", BSig, B)            # B Σ_imu B^T
        self.P = FPFt + Q

    # ------------------------------------------------------------------
    # Update — reads s_hat, P  (prior); writes s_hat, P  (posterior)
    # ------------------------------------------------------------------

    def _update(self, z: np.ndarray) -> None:
        """
        Measurement update.  z : (n, 3)  [x, y, theta]
        Reads  self.s_hat, self.P  (prior from _propagate).
        Writes self.s_hat, self.P  (posterior).
        """
        H, R = self.H, self.R

        # ν = z - H s-  (innovation, heading wrapped)
        nu = z - np.einsum("ij,nj->ni", H, self.s_hat)   # H s-
        nu[:, 2] = _wrap(nu[:, 2])

        # S = H P- H^T + R  (innovation covariance)
        HP = np.einsum("ij,njk->nik", H, self.P)          # H P-
        S  = np.einsum("nij,kj->nik", HP, H) + R          # H P- H^T + R   (n, 3, 3)

        # K = P- H^T S^-1, via batched solve:
        #   S K^T = H P-  K = solve(S, H P-)^T
        K = np.linalg.solve(S, HP).transpose(0, 2, 1)     # (n, 5, 3)

        # s = s- + K ν
        self.s_hat = self.s_hat + np.einsum("nij,nj->ni", K, nu)   # K ν
        self.s_hat[:, 2] = _wrap(self.s_hat[:, 2])

        # joseph form: P = (I - KH) P- (I - KH)^T + K R K^T
        I    = np.eye(5)
        KH   = np.einsum("nij,jk->nik", K, H)             # K H
        IKH  = I - KH                                       # I - K H
        P1   = np.einsum("nij,njk->nik", IKH, self.P)     # (I-KH) P-
        P1   = np.einsum("nij,nkj->nik", P1, IKH)         # (I-KH) P- (I-KH)^T
        KR   = np.einsum("nij,jk->nik", K, R)             # K R
        KRKt = np.einsum("nij,nkj->nik", KR, K)           # K R K^T
        self.P = P1 + KRKt

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run(
        self,
        z_seq: np.ndarray,    # (n, nt, 3)
        u_seq: np.ndarray,    # (n, nt, 3)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run propagate and update for nt steps across all n trials.

        The first measurement is consumed by the constructor, so step k = 0
        of the loop performs only an update (no propagation yet) using z_seq[:, 0]
        — this anchors the filter to the very first observation.  Subsequent
        steps propagate with u_seq[:, k-1] and then update with z_seq[:, k].

        Returns
        -------
        s_prior_hist : (n, nt, 5)       predicted state  (before update)
        P_prior_hist : (n, nt, 5, 5)    predicted covariance
        s_hist       : (n, nt, 5)       corrected state  (after update)
        P_hist       : (n, nt, 5, 5)    corrected covariance
        """
        z_seq = np.asarray(z_seq, float)
        u_seq = np.asarray(u_seq, float)
        n, nt, _ = z_seq.shape
        assert n == self.n, f"batch mismatch: filter n={self.n}, z n={n}"

        s_prior_hist = np.zeros((n, nt, 5))
        P_prior_hist = np.zeros((n, nt, 5, 5))
        s_hist       = np.zeros((n, nt, 5))
        P_hist       = np.zeros((n, nt, 5, 5))

        for k in range(nt):
            if k > 0:
                self._propagate(u_seq[:, k - 1, :])
            # snapshot s_hat/P before update — these are the priors at step k
            s_prior_hist[:, k, :]    = self.s_hat
            P_prior_hist[:, k, :, :] = self.P

            self._update(z_seq[:, k, :])

            s_hist[:, k, :]    = self.s_hat
            P_hist[:, k, :, :] = self.P

        # return s_prior_hist, P_prior_hist, s_hist, P_hist
        return s_hist, P_hist
