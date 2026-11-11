"""Fly the scenarios and write the figures and the animation used in the README.

    python scripts/make_figures.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from hover import SCENARIOS, run
from hover.math3d import euler_of, quat_to_rot

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

BG, FG, DIM, RED = "#070707", "#e8e8e8", "#7a7a7a", "#ff2d2d"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": "#333", "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": "#1c1c1c", "axes.grid": True,
    "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def figure_eight(log):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax[0].plot(log.ref_pos[:, 0], log.ref_pos[:, 1], color="#555", lw=3, label="reference")
    ax[0].plot(log.pos[:, 0], log.pos[:, 1], color=RED, lw=1.4, label="flown")
    ax[0].set_aspect("equal")
    ax[0].set(xlabel="east (m)", ylabel="north (m)", title="Figure of eight, seen from above")
    ax[0].legend(loc="lower center", ncol=2)
    err = np.linalg.norm(log.pos - log.ref_pos, axis=1)
    ax[1].plot(log.t, err, color=RED, lw=1.2)
    ax[1].set(xlabel="time (s)", ylabel="distance from reference (m)",
              title=f"Tracking error, rms {log.tracking_rms():.2f} m")
    save(fig, "figure8.png")


def step_response(log):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(log.t, log.ref_pos[:, 0], color="#666", lw=2, drawstyle="steps-post", label="command")
    ax.plot(log.t, log.pos[:, 0], color=RED, lw=1.6, label="vehicle")
    ax.set(xlabel="time (s)", ylabel="east position (m)", title="3 m step in position")
    x = log.pos[:, 0]
    ax.annotate(f"overshoot {x.max() - 3.0:.2f} m", xy=(log.t[np.argmax(x)], x.max()), xytext=(5.5, 3.5),
                color=FG, arrowprops={"arrowstyle": "-", "color": DIM})
    ax.legend(loc="lower right")
    save(fig, "step.png")


def gust(log):
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(log.t, log.wind[:, 0], color=DIM, lw=1.2)
    ax[0].set(ylabel="wind from the west (m/s)", title="A 6 m/s gust and turbulence, hovering at 1.5 m")
    dev = log.pos - log.ref_pos
    ax[1].plot(log.t, dev[:, 0], color=RED, lw=1.3, label="east")
    ax[1].plot(log.t, dev[:, 1], color=FG, lw=1.0, alpha=0.7, label="north")
    ax[1].plot(log.t, dev[:, 2], color=DIM, lw=1.0, label="up")
    ax[1].set(xlabel="time (s)", ylabel="position error (m)")
    ax[1].legend(ncol=3, loc="upper right")
    save(fig, "gust.png")


def estimator(log):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    ax[0].plot(log.t, log.attitude_error_deg(), color=RED, lw=1.0)
    ax[0].set(xlabel="time (s)", ylabel="degrees", title="Attitude estimate error")
    ax[1].plot(log.t, log.position_error_est(), color=RED, lw=1.0)
    ax[1].set(xlabel="time (s)", ylabel="metres", title="Position estimate error")
    for i, c in enumerate([RED, FG, DIM]):
        ax[2].plot(log.t, log.gyro_bias[:, i], color=c, lw=1.4, label="true" if i == 0 else None)
        ax[2].plot(log.t, log.gyro_bias_est[:, i], color=c, lw=1.0, ls="--", label="estimated" if i == 0 else None)
    ax[2].set(xlabel="time (s)", ylabel="rad/s", title="Gyro bias, three axes")
    ax[2].legend()
    fig.tight_layout()
    save(fig, "estimator.png")


def motor_loss(log):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, c in zip(range(4), [RED, FG, DIM, "#b0b0b0"]):
        ax.plot(log.t, log.motors[:, i], color=c, lw=1.3, label=f"motor {i}")
    ax.axvline(6.0, color="#555", ls=":")
    ax.text(6.1, ax.get_ylim()[1] * 0.92, "motor 0 loses 45 % of its thrust", color=DIM)
    ax.set(xlabel="time (s)", ylabel="thrust (N)", title="Thrust the motors actually deliver")
    ax.legend(ncol=4, loc="lower right")
    save(fig, "motor_loss.png")


def animate(log, name="flight.gif", step=24, fps=20):
    idx = np.arange(0, len(log.t), step)
    fig, (top, side) = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1.2, 1]})
    top.plot(log.ref_pos[:, 0], log.ref_pos[:, 1], color="#333", lw=3)
    top.set_aspect("equal")
    top.set(xlabel="east (m)", ylabel="north (m)", xlim=(-5.2, 5.2), ylim=(-4.2, 4.2))
    side.plot(log.t, log.ref_pos[:, 2], color="#333", lw=3)
    side.set(xlabel="time (s)", ylabel="height (m)", xlim=(0, log.t[-1]), ylim=(0, 2.4))
    trail, = top.plot([], [], color=RED, lw=1.2)
    arms = [top.plot([], [], color=FG, lw=2)[0] for _ in range(2)]
    hub, = top.plot([], [], "o", color=RED, ms=6)
    hline, = side.plot([], [], color=RED, lw=1.3)
    hdot, = side.plot([], [], "o", color=RED, ms=5)
    label = top.text(0.02, 0.95, "", transform=top.transAxes, color=DIM, va="top")
    arm = 0.17 * 3  # drawn larger than real so it is visible at this scale
    local = np.array([[1, -1, 0], [-1, 1, 0], [1, 1, 0], [-1, -1, 0]]) * arm / np.sqrt(2)

    def draw(n):
        k = idx[n]
        R = quat_to_rot(log.quat[k])
        pts = (R @ local.T).T + log.pos[k]
        arms[0].set_data([pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]])
        arms[1].set_data([pts[2, 0], pts[3, 0]], [pts[2, 1], pts[3, 1]])
        hub.set_data([log.pos[k, 0]], [log.pos[k, 1]])
        trail.set_data(log.pos[: k + 1, 0], log.pos[: k + 1, 1])
        hline.set_data(log.t[: k + 1], log.pos[: k + 1, 2])
        hdot.set_data([log.t[k]], [log.pos[k, 2]])
        roll, pitch, _ = np.degrees(euler_of(log.quat[k]))
        label.set_text(f"t = {log.t[k]:4.1f} s   roll {roll:+5.1f}   pitch {pitch:+5.1f}")
        return arms + [hub, trail, hline, hdot, label]

    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=len(idx), blit=False)
    anim.save(OUT / name, writer=PillowWriter(fps=fps), dpi=55)
    plt.close(fig)
    print("wrote", name)


if __name__ == "__main__":
    logs = {n: run(SCENARIOS[n](), seed=3) for n in ("figure8", "step", "gusts", "motor_loss")}
    figure_eight(logs["figure8"])
    step_response(logs["step"])
    gust(logs["gusts"])
    estimator(logs["figure8"])
    motor_loss(logs["motor_loss"])
    animate(logs["figure8"])
    for n, log in logs.items():
        print(log.summary())
