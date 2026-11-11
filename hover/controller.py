"""Cascaded flight controller.

    position error --P--> velocity command --PI--> acceleration command
        --> thrust vector --> attitude command --P--> body-rate command
        --PID--> torque --> mixer --> four motor thrusts

The outer loops decide where the thrust must point; the inner loops tilt the
vehicle to point it there. Each loop is about three to five times faster than
the one outside it, which is what lets them be tuned one at a time.
"""

import numpy as np

from .dynamics import G_WORLD
from .math3d import qconj, qmul, quat_to_rot, rot_to_quat
from .params import Gains, Params


class Controller:
    def __init__(self, params: Params, gains: Gains | None = None):
        self.params = params
        self.g = gains or Gains()
        self._Ainv = np.linalg.inv(params.allocation)
        self.vel_int = np.zeros(3)
        self.rate_int = np.zeros(3)
        self._omega_f = None
        self.last = {}  # intermediate signals from the latest update, for plotting

    def reset(self):
        self.vel_int[:] = 0
        self.rate_int[:] = 0
        self._omega_f = None

    # -- outer loops: where should the thrust point, and how hard?
    def _force(self, p, v, ref, dt):
        g = self.g
        v_cmd = ref.v + np.array(g.pos_p) * (ref.p - p)
        h = np.linalg.norm(v_cmd[:2])
        if h > g.max_speed_xy:
            v_cmd[:2] *= g.max_speed_xy / h
        v_cmd[2] = np.clip(v_cmd[2], -g.max_speed_z, g.max_speed_z)

        e_v = v_cmd - v
        if ref.p[2] < 0.02:
            self.vel_int[:] = 0  # on the ground: do not wind up while waiting to lift off
        else:
            self.vel_int = np.clip(self.vel_int + np.array(g.vel_i) * e_v * dt, -3.0, 3.0)
        a_cmd = ref.a + np.array(g.vel_p) * e_v + self.vel_int
        h = np.linalg.norm(a_cmd[:2])
        if h > g.max_accel_xy:
            a_cmd[:2] *= g.max_accel_xy / h
        self.last["v_cmd"] = v_cmd
        self.last["a_cmd"] = a_cmd
        return self.params.mass * (a_cmd - G_WORLD)

    # -- attitude: build the rotation that points the body z axis along the force
    def _desired_attitude(self, F, yaw):
        g = self.g
        n = np.linalg.norm(F)
        z_d = F / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
        tilt = np.arccos(np.clip(z_d[2], -1, 1))
        if tilt > g.max_tilt:
            hz = z_d[:2] / np.linalg.norm(z_d[:2])
            z_d = np.array([np.sin(g.max_tilt) * hz[0], np.sin(g.max_tilt) * hz[1], np.cos(g.max_tilt)])
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y_d = np.cross(z_d, x_c)
        y_d /= np.linalg.norm(y_d)
        x_d = np.cross(y_d, z_d)
        return rot_to_quat(np.column_stack([x_d, y_d, z_d]))

    def _mix(self, thrust, torque):
        """Turn collective thrust and torques into motor thrusts, giving up torque before thrust when saturated."""
        p = self.params
        thrust = np.clip(thrust, 0.05 * p.hover_thrust, 4 * p.max_thrust * 0.95)
        base = self._Ainv @ np.array([thrust, 0, 0, 0])
        delta = self._Ainv @ np.array([0, *torque])
        scale = 1.0
        for b, d in zip(base, delta):
            if d > 1e-9:
                scale = min(scale, (p.max_thrust - b) / d)
            elif d < -1e-9:
                scale = min(scale, (0.0 - b) / d)
        scale = max(0.0, min(1.0, scale))
        self.last["torque_scale"] = scale
        return np.clip(base + scale * delta, 0.0, p.max_thrust)

    def update(self, p, v, q, omega, ref, dt):
        """One control step. Returns the four motor thrust commands in newtons."""
        g = self.g
        if ref.p[2] < 0.02 and p[2] < 0.3:
            # Told to be on the floor and nearly there: motors idle, so sensor noise cannot lift it off.
            self.reset()
            return np.zeros(4)
        F = self._force(p, v, ref, dt)
        R = quat_to_rot(q)
        thrust = float(F @ R[:, 2])

        q_d = self._desired_attitude(F, ref.yaw)
        q_e = qmul(qconj(q), q_d)
        if q_e[0] < 0:
            q_e = -q_e
        omega_cmd = 2.0 * np.array(g.att_p) * q_e[1:]

        # Low-pass the rate for the derivative term only, so noise does not reach the motors.
        alpha = dt / (dt + 1.0 / (2 * np.pi * g.d_filter_hz))
        prev = omega.copy() if self._omega_f is None else self._omega_f
        self._omega_f = prev + alpha * (omega - prev)
        d_omega = (self._omega_f - prev) / dt

        e_w = omega_cmd - omega
        if ref.p[2] < 0.02:
            self.rate_int[:] = 0
        else:
            self.rate_int = np.clip(self.rate_int + np.array(g.rate_i) * e_w * dt, -4.0, 4.0)
        alpha_cmd = np.array(g.rate_p) * e_w + self.rate_int - np.array(g.rate_d) * d_omega
        torque = self.params.J @ alpha_cmd
        self.last["omega_cmd"] = omega_cmd
        self.last["q_d"] = q_d
        return self._mix(thrust, torque)
