"""The simulation loop: physics at 1 kHz, sensors and controller at 250 Hz, GPS at 10 Hz."""

from dataclasses import dataclass

import numpy as np

from .controller import Controller
from .dynamics import Quadrotor
from .estimator import Estimator
from .math3d import angle_between, euler_of, quat_from_euler
from .params import Gains, Params
from .scenarios import Scenario
from .sensors import Sensors


@dataclass
class Log:
    t: np.ndarray
    pos: np.ndarray          # truth
    vel: np.ndarray
    quat: np.ndarray
    pos_est: np.ndarray
    vel_est: np.ndarray
    quat_est: np.ndarray
    ref_pos: np.ndarray
    motors: np.ndarray
    wind: np.ndarray
    gyro_bias: np.ndarray
    gyro_bias_est: np.ndarray
    scenario: str

    # -- numbers that summarise a flight
    def tracking_rms(self, t_from=0.0):
        m = self.t >= t_from
        return float(np.sqrt(np.mean(np.sum((self.pos[m] - self.ref_pos[m]) ** 2, axis=1))))

    def attitude_error_deg(self):
        return np.degrees([angle_between(a, b) for a, b in zip(self.quat, self.quat_est)])

    def position_error_est(self):
        return np.linalg.norm(self.pos - self.pos_est, axis=1)

    def max_tilt_deg(self):
        return float(np.degrees(max(np.hypot(*euler_of(q)[:2]) for q in self.quat)))

    def summary(self):
        att = self.attitude_error_deg()
        half = len(self.t) // 2
        return {
            "scenario": self.scenario,
            "tracking_rms_m": round(self.tracking_rms(), 3),
            "estimate_pos_rms_m": round(float(np.sqrt(np.mean(self.position_error_est()[half:] ** 2))), 3),
            "estimate_att_rms_deg": round(float(np.sqrt(np.mean(att[half:] ** 2))), 2),
            "max_tilt_deg": round(self.max_tilt_deg(), 1),
            "final_error_m": round(float(np.linalg.norm(self.pos[-1] - self.ref_pos[-1])), 3),
        }


def run(scn: Scenario, seed=0, params: Params | None = None, gains: Gains | None = None, use_truth=False) -> Log:
    """Fly a scenario. With use_truth the controller is given the true state instead of the estimate."""
    params = params or Params()
    rng = np.random.default_rng(seed)
    quad = Quadrotor(params, position=scn.start, quat=quat_from_euler(0, 0, 0))
    sensors = Sensors(params, rng)
    est = Estimator(params, position=scn.start)
    ctrl = Controller(params, gains)

    in_air = scn.start[2] > 0.05
    cmd = np.full(4, params.hover_thrust / 4 if in_air else 0.0)
    quad.s[13:17] = cmd
    if in_air:
        quad.step(cmd, np.zeros(3), dt=1e-6)  # settle the acceleration the IMU model reads
    gyro, accel = sensors.imu(quad, params.dt_phys)
    est.align(accel, sensors.mag_heading(quad))

    dt = params.dt_phys
    dt_c = dt * params.control_every
    n = int(round(scn.duration / dt))
    events = sorted(scn.events, key=lambda e: e[0])
    rows = {k: [] for k in ("t", "pos", "vel", "quat", "pos_est", "vel_est", "quat_est", "ref_pos", "motors", "wind", "gb", "gbe")}
    wind = np.zeros(3)

    for k in range(n):
        t = k * dt
        while events and t >= events[0][0]:
            events.pop(0)[1](quad)
        if scn.wind:
            wind = scn.wind(t, rng)

        if k % params.control_every == 0:
            gyro, accel = sensors.imu(quad, dt_c)
            est.predict(gyro, accel, dt_c)
            if k % params.gps_every == 0:
                est.update_gps(*sensors.gps(quad))
            if k % params.baro_every == 0:
                est.update_baro(sensors.baro(quad))
            if k % params.mag_every == 0:
                est.update_heading(sensors.mag_heading(quad))
            ref = scn.trajectory.at(t)
            if use_truth:
                cmd = ctrl.update(quad.pos, quad.vel, quad.quat, quad.rates, ref, dt_c)
            else:
                cmd = ctrl.update(est.p, est.v, est.q, est.omega, ref, dt_c)
            for key, val in (("t", t), ("pos", quad.pos), ("vel", quad.vel), ("quat", quad.quat), ("pos_est", est.p),
                             ("vel_est", est.v), ("quat_est", est.q), ("ref_pos", ref.p), ("motors", quad.thrusts),
                             ("wind", wind), ("gb", sensors.gyro_bias), ("gbe", est.bg)):
                rows[key].append(np.copy(val))

        quad.step(cmd, wind)

    a = {k: np.array(v) for k, v in rows.items()}
    return Log(a["t"], a["pos"], a["vel"], a["quat"], a["pos_est"], a["vel_est"], a["quat_est"], a["ref_pos"],
               a["motors"], a["wind"], a["gb"], a["gbe"], scn.name)
