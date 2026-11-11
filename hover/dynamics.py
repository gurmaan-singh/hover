"""Rigid-body quadrotor: position, velocity, attitude, body rates and four lagged motors."""

import numpy as np

from .math3d import qmul, qnormalize, quat_to_rot
from .params import G, Params

G_WORLD = np.array([0.0, 0.0, -G])  # world frame is east-north-up


class Quadrotor:
    """State vector: position (3), velocity (3), quaternion (4), body rates (3), motor thrusts (4)."""

    def __init__(self, params: Params, position=(0, 0, 0), quat=(1, 0, 0, 0), velocity=(0, 0, 0)):
        self.p = params
        self.s = np.zeros(17)
        self.s[0:3] = position
        self.s[3:6] = velocity
        self.s[6:10] = quat
        self.motor_effectiveness = np.ones(4)  # 1 is healthy; below 1 is a weakened motor
        self.torque_disturbance = np.zeros(3)
        self._Jinv = np.linalg.inv(params.J)
        self._A = params.allocation
        self.accel_nongrav = np.zeros(3)  # world-frame acceleration without gravity, for the IMU model
        self.grounded = position[2] <= 1e-6
        if self.grounded:
            self.accel_nongrav = -G_WORLD  # the floor pushes up with exactly one g

    # -- accessors
    @property
    def pos(self):
        return self.s[0:3]

    @property
    def vel(self):
        return self.s[3:6]

    @property
    def quat(self):
        return self.s[6:10]

    @property
    def rates(self):
        return self.s[10:13]

    @property
    def thrusts(self):
        return self.s[13:17]

    def specific_force_body(self):
        """What an accelerometer measures: acceleration minus gravity, in body axes."""
        return quat_to_rot(self.quat).T @ self.accel_nongrav

    def _deriv(self, s, cmd, wind):
        p = self.p
        q = s[6:10]
        w = s[10:13]
        T = s[13:17]
        R = quat_to_rot(q)
        v_rel = s[3:6] - wind
        drag = -(p.drag_lin * v_rel + p.drag_quad * np.linalg.norm(v_rel) * v_rel)
        a_ng = (R @ np.array([0.0, 0.0, T.sum()]) + drag) / p.mass
        tau = (self._A @ T)[1:4] + self.torque_disturbance
        d = np.empty(17)
        d[0:3] = s[3:6]
        d[3:6] = a_ng + G_WORLD
        d[6:10] = 0.5 * qmul(q, np.array([0.0, *w]))
        d[10:13] = self._Jinv @ (tau - np.cross(w, p.J @ w))
        d[13:17] = (self.motor_effectiveness * cmd - T) / p.motor_tau
        self.accel_nongrav = a_ng
        return d

    def step(self, motor_cmd, wind=np.zeros(3), dt=None):
        """Advance one physics step with fourth-order Runge-Kutta."""
        dt = dt or self.p.dt_phys
        cmd = np.clip(motor_cmd, 0.0, self.p.max_thrust)
        s = self.s
        k1 = self._deriv(s, cmd, wind)
        k2 = self._deriv(s + 0.5 * dt * k1, cmd, wind)
        k3 = self._deriv(s + 0.5 * dt * k2, cmd, wind)
        k4 = self._deriv(s + dt * k3, cmd, wind)
        self.s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.s[6:10] = qnormalize(self.s[6:10])
        self._ground()
        self._deriv(self.s, cmd, wind)  # refresh accel_nongrav at the new state
        if self.grounded:
            self.accel_nongrav = -G_WORLD

    def _ground(self):
        """A flat floor at z = 0. Not a contact model: it only stops the vehicle falling through."""
        s = self.s
        if s[2] < 0.0:
            s[2] = 0.0
            if s[5] < 0.0:
                s[5] = 0.0
        self.grounded = bool(s[2] <= 1e-6 and s[13:17].sum() < self.p.hover_thrust)
        if self.grounded:
            # Sitting on the floor: no lift-off yet, so it cannot tilt or slide.
            s[3:6] = 0.0
            s[10:13] = 0.0
            w, x, y, z = s[6:10]
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            s[6:10] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
