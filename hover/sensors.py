"""Sensor models: gyro and accelerometer with random-walk bias, GPS, barometer, magnetometer."""

import numpy as np

from .dynamics import Quadrotor
from .math3d import wrap, yaw_of
from .params import Params


class Sensors:
    def __init__(self, params: Params, rng: np.random.Generator):
        self.p = params
        self.rng = rng
        self.gyro_bias = rng.normal(0, params.gyro_bias_init, 3)
        self.accel_bias = rng.normal(0, params.accel_bias_init, 3)

    def imu(self, quad: Quadrotor, dt: float):
        p, r = self.p, self.rng
        self.gyro_bias += r.normal(0, p.gyro_bias_walk * np.sqrt(dt), 3)
        self.accel_bias += r.normal(0, p.accel_bias_walk * np.sqrt(dt), 3)
        gyro = quad.rates + self.gyro_bias + r.normal(0, p.gyro_noise, 3)
        accel = quad.specific_force_body() + self.accel_bias + r.normal(0, p.accel_noise, 3)
        return gyro, accel

    def gps(self, quad: Quadrotor):
        r, p = self.rng, self.p
        return quad.pos + r.normal(0, p.gps_pos_noise, 3), quad.vel + r.normal(0, p.gps_vel_noise, 3)

    def baro(self, quad: Quadrotor):
        return quad.pos[2] + self.rng.normal(0, self.p.baro_noise)

    def mag_heading(self, quad: Quadrotor):
        return wrap(yaw_of(quad.quat) + self.rng.normal(0, self.p.mag_noise))
