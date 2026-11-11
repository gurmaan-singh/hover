"""Ready-made flights. Each scenario says what to fly and what goes wrong during it."""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .trajectory import Figure8, Hold, Step, Waypoints

HOVER_Z = 1.5


@dataclass
class Scenario:
    name: str
    description: str
    duration: float
    trajectory: object
    start: tuple = (0.0, 0.0, 0.0)
    wind: Callable | None = None                 # (t, rng) -> wind vector, may keep state in the closure
    events: list = field(default_factory=list)   # (time, function(quad)) applied once


def _gusty(t_gust=5.0, length=4.0, peak=6.0, turbulence=0.8):
    """A one-off gust from the west plus slowly varying turbulence."""
    state = {"u": np.zeros(3), "last": 0.0}

    def wind(t, rng):
        dt = max(t - state["last"], 1e-3)
        state["last"] = t
        tau = 2.0
        state["u"] += (-state["u"] / tau) * dt + turbulence * np.sqrt(2 * dt / tau) * rng.normal(size=3)
        gust = 0.0
        if t_gust <= t <= t_gust + length:
            gust = peak * 0.5 * (1 - np.cos(2 * np.pi * (t - t_gust) / length))
        return state["u"] * np.array([1, 1, 0.3]) + np.array([gust, 0.0, 0.0])

    return wind


def _weak_motor(index, effectiveness):
    def event(quad):
        quad.motor_effectiveness[index] = effectiveness
    return event


def takeoff():
    z = HOVER_Z
    traj = Waypoints([(0, 0, 0), (0, 0, z)], max_speed=0.8, pause=0.0)
    return Scenario("takeoff", "Lift off from the ground and settle into a hover at 1.5 m.", 12.0, traj)


def step():
    a, b = (0.0, 0.0, HOVER_Z), (3.0, 0.0, HOVER_Z)
    return Scenario("step", "Hover, then a hard 3 m step in position. Shows overshoot and settling time.", 12.0,
                    Step(a, b, t0=2.0), start=a)


def square():
    z = HOVER_Z
    pts = [(0, 0, z), (4, 0, z), (4, 4, z), (0, 4, z), (0, 0, z)]
    traj = Waypoints(pts, max_speed=2.5, pause=0.6)
    return Scenario("square", "Fly a 4 m square, stopping at each corner.", traj.end_time + 2.0, traj, start=pts[0])


def figure8():
    traj = Figure8(centre=(0, 0, HOVER_Z), A=4.0, B=3.0, C=0.3, omega=0.85, ramp=4.0)
    return Scenario("figure8", "A figure of eight at up to about 3.5 m/s and 3 m/s^2 sideways acceleration.", 32.0, traj,
                    start=tuple(traj.at(0).p))


def gusts():
    a = (0.0, 0.0, HOVER_Z)
    return Scenario("gusts", "Hover through a 6 m/s gust and continuous turbulence.", 22.0, Hold(a), start=a, wind=_gusty())


def motor_loss():
    a = (0.0, 0.0, HOVER_Z)
    return Scenario("motor_loss", "Hover, then one motor drops to 55 % of its thrust at t = 6 s.", 18.0, Hold(a), start=a,
                    events=[(6.0, _weak_motor(0, 0.55))])


SCENARIOS = {f.__name__: f for f in (takeoff, step, square, figure8, gusts, motor_loss)}
