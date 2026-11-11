"""All physical constants and tuning in one place."""

from dataclasses import dataclass, field

import numpy as np

G = 9.80665


@dataclass
class Params:
    # Airframe: a 1.2 kg quad with 34 cm motor-to-motor diagonal (X layout).
    mass: float = 1.2
    arm: float = 0.17                       # centre to motor, metres
    inertia: tuple = (0.0123, 0.0123, 0.0224)
    torque_ratio: float = 0.016             # propeller drag torque per newton of thrust, metres
    max_thrust: float = 8.0                 # per motor, newtons
    motor_tau: float = 0.02                 # motor + propeller lag, seconds
    drag_lin: float = 0.10                  # N per m/s
    drag_quad: float = 0.04                 # N per (m/s)^2

    # Timing.
    dt_phys: float = 0.001                  # physics step, 1 kHz
    control_every: int = 4                  # controller and IMU at 250 Hz
    gps_every: int = 25                     # 10 Hz
    baro_every: int = 10                    # 25 Hz
    mag_every: int = 10                     # 25 Hz

    # Sensor noise (1 sigma).
    gyro_noise: float = 0.004               # rad/s per sample
    gyro_bias_walk: float = 2e-4            # rad/s per sqrt(s)
    gyro_bias_init: float = 0.01            # rad/s
    accel_noise: float = 0.06               # m/s^2 per sample
    accel_bias_walk: float = 2e-3           # m/s^2 per sqrt(s)
    accel_bias_init: float = 0.05           # m/s^2
    gps_pos_noise: float = 0.25             # m
    gps_vel_noise: float = 0.06             # m/s
    baro_noise: float = 0.15                # m
    mag_noise: float = 0.03                 # rad of heading

    @property
    def J(self):
        return np.diag(self.inertia)

    @property
    def motor_xy(self):
        d = self.arm / np.sqrt(2)
        # Front-right, front-left, rear-left, rear-right. x forward, y left.
        return np.array([[d, -d], [d, d], [-d, d], [-d, -d]])

    @property
    def spin(self):
        # +1 counter-clockwise seen from above.
        return np.array([1.0, -1.0, 1.0, -1.0])

    @property
    def allocation(self):
        """Matrix A with [thrust, tau_x, tau_y, tau_z] = A @ motor_thrusts."""
        xy = self.motor_xy
        return np.vstack([
            np.ones(4),
            xy[:, 1],
            -xy[:, 0],
            -self.spin * self.torque_ratio,
        ])

    @property
    def hover_thrust(self):
        return self.mass * G


@dataclass
class Gains:
    pos_p: tuple = (1.4, 1.4, 2.0)          # position error to velocity command, 1/s
    vel_p: tuple = (3.2, 3.2, 4.5)          # velocity error to acceleration, 1/s
    vel_i: tuple = (0.9, 0.9, 1.5)
    max_speed_xy: float = 6.0
    max_speed_z: float = 3.0
    max_accel_xy: float = 7.0
    max_tilt: float = np.radians(40.0)
    att_p: tuple = (9.0, 9.0, 3.5)          # attitude error to rate command, 1/s
    rate_p: tuple = (26.0, 26.0, 11.0)      # scaled by inertia, so units are 1/s
    rate_i: tuple = (14.0, 14.0, 6.0)
    rate_d: tuple = (0.35, 0.35, 0.0)
    d_filter_hz: float = 40.0
