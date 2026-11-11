import numpy as np
import pytest

from hover.dynamics import G_WORLD, Quadrotor
from hover.math3d import (angle_between, euler_of, quat_exp, quat_from_euler, quat_to_rot, qmul, rot_to_quat, skew, wrap)
from hover.params import G, Params


def make(params=None, position=(0, 0, 100.0)):
    return Quadrotor(params or Params(), position=position)


def test_free_fall_matches_the_formula():
    q = make(Params(drag_lin=0.0, drag_quad=0.0))
    for _ in range(1000):
        q.step(np.zeros(4))
    assert q.pos[2] == pytest.approx(100 - 0.5 * G * 1.0**2, abs=1e-6)
    assert q.vel[2] == pytest.approx(-G, abs=1e-6)


def test_hover_thrust_balances_weight():
    p = Params()
    q = make(p)
    q.s[13:17] = p.hover_thrust / 4
    for _ in range(2000):
        q.step(np.full(4, p.hover_thrust / 4))
    assert np.linalg.norm(q.vel) < 1e-6
    assert q.pos[2] == pytest.approx(100.0, abs=1e-6)


def test_differential_thrust_rolls_the_vehicle_as_expected():
    p = Params()
    q = make(p)
    base = p.hover_thrust / 4
    thrust = np.array([base - 0.2, base + 0.2, base + 0.2, base - 0.2])  # more thrust on the left motors
    q.s[13:17] = thrust
    for _ in range(100):  # 0.1 s; command equals state so the motors do not lag
        q.step(thrust)
    torque_x = (p.allocation @ thrust)[1]
    assert torque_x > 0
    assert q.rates[0] == pytest.approx(torque_x * 0.1 / p.inertia[0], rel=1e-2)
    assert abs(q.rates[1]) < 1e-3 and abs(q.rates[2]) < 1e-3


def test_yaw_torque_comes_from_motor_direction():
    p = Params()
    q = make(p)
    base = p.hover_thrust / 4
    thrust = np.array([base + 0.3, base - 0.3, base + 0.3, base - 0.3])  # speed up the counter-clockwise motors
    q.s[13:17] = thrust
    for _ in range(100):
        q.step(thrust)
    assert q.rates[2] < 0  # the body turns clockwise, opposite to the faster propellers


def test_torque_free_motion_keeps_momentum_and_energy():
    p = Params(inertia=(0.01, 0.02, 0.03))
    q = make(p)
    q.s[10:13] = [1.0, 5.0, 2.0]
    J = p.J
    L0 = np.linalg.norm(J @ q.rates)
    E0 = 0.5 * q.rates @ J @ q.rates
    for _ in range(2000):
        q.step(np.zeros(4))
    w = q.rates
    # Momentum is conserved in the world frame, so its length is the same in body axes.
    assert np.linalg.norm(J @ w) == pytest.approx(L0, rel=1e-6)
    assert 0.5 * w @ J @ w == pytest.approx(E0, rel=1e-6)


def test_quaternion_stays_unit_length_under_fast_spin():
    q = make()
    q.s[10:13] = [8.0, -6.0, 9.0]
    for _ in range(3000):
        q.step(np.zeros(4))
    assert np.linalg.norm(q.quat) == pytest.approx(1.0, abs=1e-12)


def test_resting_on_the_floor_reads_one_g():
    q = Quadrotor(Params(), position=(0, 0, 0))
    for _ in range(50):
        q.step(np.zeros(4))
    assert q.pos[2] == 0.0
    assert np.allclose(q.specific_force_body(), [0, 0, G], atol=1e-9)


def test_wind_pushes_the_vehicle():
    q = make()
    q.s[13:17] = Params().hover_thrust / 4
    for _ in range(1000):
        q.step(np.full(4, Params().hover_thrust / 4), wind=np.array([10.0, 0, 0]))
    assert q.vel[0] > 0.5


def test_allocation_matrix_is_invertible_and_symmetric_in_the_right_places():
    A = Params().allocation
    assert abs(np.linalg.det(A)) > 1e-6
    thrust_only = np.linalg.solve(A, [4.0, 0, 0, 0])
    assert np.allclose(thrust_only, 1.0)


# -- math helpers
def test_quaternion_rotation_round_trips():
    rng = np.random.default_rng(1)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        back = rot_to_quat(quat_to_rot(q))
        assert angle_between(q, back) < 1e-6


def test_euler_round_trip():
    for roll, pitch, yaw in [(0.1, -0.2, 0.3), (-0.5, 0.4, 2.5), (0.0, 0.0, -3.0)]:
        r, p, y = euler_of(quat_from_euler(roll, pitch, yaw))
        assert (r, p, wrap(y)) == pytest.approx((roll, pitch, wrap(yaw)), abs=1e-12)


def test_exp_matches_rotation_matrix_of_the_same_vector():
    v = np.array([0.3, -0.2, 0.5])
    R = quat_to_rot(quat_exp(v))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    # Rodrigues' formula.
    th = np.linalg.norm(v)
    K = skew(v / th)
    assert np.allclose(R, np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K, atol=1e-12)


def test_quaternion_product_composes_rotations():
    a, b = quat_exp([0.2, 0.1, -0.3]), quat_exp([-0.4, 0.5, 0.1])
    assert np.allclose(quat_to_rot(qmul(a, b)), quat_to_rot(a) @ quat_to_rot(b), atol=1e-12)
