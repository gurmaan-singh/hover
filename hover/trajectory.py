"""Reference trajectories. Each one returns position, velocity, acceleration and yaw at time t."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Ref:
    p: np.ndarray
    v: np.ndarray
    a: np.ndarray
    yaw: float = 0.0


class Hold:
    """Stay at one point."""

    def __init__(self, point, yaw=0.0):
        self.point = np.asarray(point, float)
        self.yaw = yaw

    def at(self, t):
        return Ref(self.point.copy(), np.zeros(3), np.zeros(3), self.yaw)


class Step:
    """Jump from one point to another at time t0. The controller sees a hard step."""

    def __init__(self, a, b, t0=1.0, yaw=0.0):
        self.a, self.b, self.t0, self.yaw = np.asarray(a, float), np.asarray(b, float), t0, yaw

    def at(self, t):
        return Ref((self.b if t >= self.t0 else self.a).copy(), np.zeros(3), np.zeros(3), self.yaw)


def _quintic(s):
    """Minimum-jerk blend: position, first and second derivative of 10s^3 - 15s^4 + 6s^5."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 - 60 * s**3 + 30 * s**4, 60 * s - 180 * s**2 + 120 * s**3


class Waypoints:
    """Visit points in order, coming to rest at each one, with smooth (minimum-jerk) motion between them."""

    def __init__(self, points, max_speed=2.5, pause=0.6, yaw=0.0, start_time=0.0):
        self.points = [np.asarray(p, float) for p in points]
        self.yaw = yaw
        self.legs = []  # (t_start, t_end, a, b)
        t = start_time
        for a, b in zip(self.points[:-1], self.points[1:]):
            dur = max(1.5, 1.875 * np.linalg.norm(b - a) / max_speed)  # 1.875 is the peak/mean speed of a quintic
            self.legs.append((t, t + dur, a, b))
            t += dur + pause
        self.end_time = t

    def at(self, t):
        for t0, t1, a, b in self.legs:
            if t < t0:
                return Ref(a.copy(), np.zeros(3), np.zeros(3), self.yaw)
            if t <= t1:
                T = t1 - t0
                s, ds, dds = _quintic((t - t0) / T)
                d = b - a
                return Ref(a + d * s, d * ds / T, d * dds / T**2, self.yaw)
        return Ref(self.points[-1].copy(), np.zeros(3), np.zeros(3), self.yaw)


class Figure8:
    """A lemniscate flown at a steady pace, eased in so it starts from rest.

    x = A sin(phi), y = B sin(phi) cos(phi), z = z0 + C sin(phi)
    """

    def __init__(self, centre=(0, 0, 1.5), A=4.0, B=3.0, C=0.0, omega=0.8, ramp=4.0, yaw=0.0):
        self.c = np.asarray(centre, float)
        self.A, self.B, self.C, self.w, self.ramp, self.yaw = A, B, C, omega, ramp, yaw

    def _phase(self, t):
        """phase, phase rate, phase acceleration with a smooth speed-up over `ramp` seconds."""
        w, T = self.w, self.ramp
        if t >= T:
            return w * (0.5 * T + (t - T)), w, 0.0
        x = max(t, 0.0) / T
        phi = w * T * (x**6 - 3 * x**5 + 2.5 * x**4)
        dphi = w * (6 * x**5 - 15 * x**4 + 10 * x**3)
        ddphi = w * (30 * x**4 - 60 * x**3 + 30 * x**2) / T
        return phi, dphi, ddphi

    def at(self, t):
        phi, dphi, ddphi = self._phase(t)
        A, B, C = self.A, self.B, self.C
        s, c = np.sin(phi), np.cos(phi)
        p = self.c + np.array([A * s, B * s * c, C * s])
        d1 = np.array([A * c, B * (c * c - s * s), C * c])
        d2 = np.array([-A * s, -4 * B * s * c, -C * s])
        v = d1 * dphi
        a = d2 * dphi**2 + d1 * ddphi
        return Ref(p, v, a, self.yaw)
