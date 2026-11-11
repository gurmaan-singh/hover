"""Error-state Kalman filter (ESKF) for position, velocity, attitude and sensor biases.

The nominal state is propagated by integrating the IMU. The filter tracks a small
15-element error around it and corrects the nominal state when GPS, barometer or
compass readings arrive.

Error state: [position, velocity, attitude, accel bias, gyro bias], each 3 long.
The attitude error is a small rotation in body axes: q_true = q_est (x) exp(dtheta).
"""

import numpy as np

from .dynamics import G_WORLD
from .math3d import qmul, quat_exp, quat_from_euler, quat_to_rot, qnormalize, skew, wrap, yaw_of
from .params import Params

P_, V_, T_, BA, BG = (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15))


class Estimator:
    def __init__(self, params: Params, position=(0, 0, 0), velocity=(0, 0, 0), quat=(1, 0, 0, 0)):
        self.params = params
        self.p = np.array(position, float)
        self.v = np.array(velocity, float)
        self.q = np.array(quat, float)
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        self.P = np.diag([1.0] * 3 + [0.5] * 3 + [0.1] * 3 + [0.05] * 3 + [0.02] * 3) ** 2
        self.omega = np.zeros(3)  # latest bias-corrected body rate

    # -- prediction from the IMU
    def predict(self, gyro, accel, dt):
        pr = self.params
        w = gyro - self.bg
        f = accel - self.ba
        R = quat_to_rot(self.q)
        a = R @ f + G_WORLD
        self.p = self.p + self.v * dt + 0.5 * a * dt * dt
        self.v = self.v + a * dt
        self.q = qnormalize(qmul(self.q, quat_exp(w * dt)))
        self.omega = w

        F = np.zeros((15, 15))
        F[P_, V_] = np.eye(3)
        F[V_, T_] = -R @ skew(f)
        F[V_, BA] = -R
        F[T_, T_] = -skew(w)
        F[T_, BG] = -np.eye(3)
        Phi = np.eye(15) + F * dt + 0.5 * (F @ F) * dt * dt
        Q = np.zeros((15, 15))
        Q[V_, V_] = np.eye(3) * (pr.accel_noise * dt) ** 2
        Q[T_, T_] = np.eye(3) * (pr.gyro_noise * dt) ** 2
        Q[BA, BA] = np.eye(3) * pr.accel_bias_walk ** 2 * dt
        Q[BG, BG] = np.eye(3) * pr.gyro_bias_walk ** 2 * dt
        self.P = Phi @ self.P @ Phi.T + Q

    # -- measurement updates
    def _update(self, y, H, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form keeps P symmetric and positive
        self.p = self.p + dx[P_]
        self.v = self.v + dx[V_]
        self.q = qnormalize(qmul(self.q, quat_exp(dx[T_])))
        self.ba = self.ba + dx[BA]
        self.bg = self.bg + dx[BG]
        # Reset: the attitude error was folded into q, so re-express P in the new frame.
        G = np.eye(15)
        G[T_, T_] = np.eye(3) - skew(0.5 * dx[T_])
        self.P = G @ self.P @ G.T

    def update_gps(self, pos, vel):
        pr = self.params
        H = np.zeros((6, 15))
        H[0:3, P_] = np.eye(3)
        H[3:6, V_] = np.eye(3)
        R = np.diag([pr.gps_pos_noise ** 2] * 3 + [pr.gps_vel_noise ** 2] * 3)
        self._update(np.concatenate([pos - self.p, vel - self.v]), H, R)

    def update_baro(self, z):
        H = np.zeros((1, 15))
        H[0, 2] = 1.0
        self._update(np.array([z - self.p[2]]), H, np.array([[self.params.baro_noise ** 2]]))

    def update_heading(self, yaw):
        # d(yaw)/d(attitude error), found numerically because the yaw formula is awkward to differentiate.
        eps = 1e-6
        y0 = yaw_of(self.q)
        H = np.zeros((1, 15))
        for i in range(3):
            e = np.zeros(3)
            e[i] = eps
            H[0, 6 + i] = wrap(yaw_of(qmul(self.q, quat_exp(e))) - y0) / eps
        self._update(np.array([wrap(yaw - y0)]), H, np.array([[self.params.mag_noise ** 2]]))

    def align(self, accel, yaw):
        """Initial attitude from a still vehicle: tilt from gravity, heading from the compass."""
        f = accel / np.linalg.norm(accel)
        roll = np.arctan2(f[1], f[2])
        pitch = np.arctan2(-f[0], np.hypot(f[1], f[2]))
        self.q = quat_from_euler(roll, pitch, yaw)
