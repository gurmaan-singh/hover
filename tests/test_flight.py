"""Closed-loop flights. Each test runs the whole stack: physics, sensors, filter, controller."""

import numpy as np
import pytest

from hover import SCENARIOS, run
from hover.scenarios import HOVER_Z, Scenario
from hover.trajectory import Hold


@pytest.fixture(scope="module")
def flights():
    return {name: run(SCENARIOS[name](), seed=3) for name in ("takeoff", "step", "figure8", "gusts", "motor_loss", "square")}


def test_takeoff_settles_at_the_target_height(flights):
    log = flights["takeoff"]
    late = log.t > 8
    assert np.abs(log.pos[late] - log.ref_pos[late]).max() < 0.12
    assert log.pos[-1, 2] == pytest.approx(HOVER_Z, abs=0.1)


def test_step_response_is_quick_and_does_not_ring(flights):
    log = flights["step"]
    x, t = log.pos[:, 0], log.t
    assert x.max() - 3.0 < 0.45                       # overshoot under 15 % of the step
    assert t[np.argmax(x >= 2.7)] - 2.0 < 2.5         # reaches 90 % within 2.5 s
    outside = np.where(np.abs(x - 3.0) > 0.15)[0]
    assert t[outside[-1]] - 2.0 < 5.0                 # settled within 5 % inside 5 s


def test_figure_eight_is_tracked_closely(flights):
    log = flights["figure8"]
    assert log.tracking_rms() < 0.25
    assert log.max_tilt_deg() < 40.0


def test_square_visits_the_corners(flights):
    log = flights["square"]
    assert log.tracking_rms() < 0.25
    assert np.linalg.norm(log.pos[-1] - np.array([0, 0, HOVER_Z])) < 0.15


def test_gust_pushes_it_off_but_it_recovers(flights):
    log = flights["gusts"]
    dev = np.linalg.norm(log.pos - log.ref_pos, axis=1)
    assert 0.1 < dev.max() < 1.0                      # the wind does move it, and not far
    assert dev[log.t > 18].max() < 0.25               # and it is back on station afterwards


def test_one_weak_motor_is_compensated(flights):
    log = flights["motor_loss"]
    assert log.pos[:, 2].min() > 1.0                  # never drops
    assert np.linalg.norm(log.pos[-1] - log.ref_pos[-1]) < 0.2
    assert log.max_tilt_deg() < 15.0


def test_the_estimator_tracks_the_truth_in_every_flight(flights):
    for name, log in flights.items():
        half = len(log.t) // 2
        att = log.attitude_error_deg()[half:]
        assert np.sqrt(np.mean(att**2)) < 1.0, name
        assert np.sqrt(np.mean(log.position_error_est()[half:] ** 2)) < 0.2, name


def test_gyro_bias_is_learned(flights):
    log = flights["figure8"]
    err = np.abs(log.gyro_bias[-1] - log.gyro_bias_est[-1])
    assert err.max() < 0.003                          # starting bias was about 0.01 rad/s


def test_same_seed_same_flight_different_seed_different_noise():
    scn = lambda: Scenario("hold", "", 3.0, Hold((0, 0, 1.5)), start=(0, 0, 1.5))
    a, b, c = run(scn(), seed=1), run(scn(), seed=1), run(scn(), seed=2)
    assert np.array_equal(a.pos, b.pos)
    assert not np.array_equal(a.pos_est, c.pos_est)


def test_resting_on_the_ground_does_not_wander():
    log = run(Scenario("rest", "", 6.0, Hold((0, 0, 0)), start=(0, 0, 0)), seed=4)
    assert np.abs(log.pos).max() < 1e-6
    assert log.attitude_error_deg()[-1] < 1.0
