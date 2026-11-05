'''
Animated version of hexapod_camera_fusion_inekf.py, one run of experiment 2
(one pose jump and one dropout of the camera): a 3D hexapod walks (tripod
gait, 0.2 m/s) while contact-aided invariant EKFs track it from an IMU plus
leg kinematics, and three of them also use a tracking camera that reports its
pose in its own, drifting frame. They see identical leg, IMU and camera data:
  - "legs": no camera;
  - "cam_pose": the camera pose used as if it were the world pose;
  - "cam_pose gated": the same, with the chi-squared gate on camera readings;
  - "cam_pose_drift": the camera frame's position and yaw offsets are states.

Three panels, drawn progressively:
  1. top-down view, a window that follows the robot: the true body, legs and
     stance feet (black), the camera's reported position and heading (purple
     diamond, trail of the last 3 s: its frame drifts and steps at the jump),
     and each filter's estimated body outline and trail;
  2. position error of each filter and of the camera's own reading, with the
     jump time and the dropout window marked;
  3. position NEES of each filter (log scale; consistent value 3, with the 95%
     band of a single NEES sample), then a strip showing when the camera
     reading was available and when the gated cam_pose used it.

Message (doc docs/filtering/hexapod_camera_fusion_inekf.md, sections 3-4): a
tracking camera's pose lives in its own frame, which drifts from the world
(scale error, yaw drift) and can jump. Used as the world pose, it makes the
filter follow the drift while believing 5 mm (NEES in the thousands); the
gate then locks it out after the jump. With the frame's offset as states the
filter stays consistent and absorbs the jump.

Seed 2 by default (not main()'s seed 0): the seed sets the jump size, and
seed 0 draws a 33 cm jump (the doc's jump std is 0.1 m per axis, so about 12 cm
is typical). Seed 2 is the first seed from 0 whose camera draw gives a jump of
8-13 cm (11 cm, at 15.1 s; dropout 41.7-44.7 s); the rule uses the camera draws
only, not the filter results (though the filter results of seeds 0-3 had been
looked at before the rule was set). This is one draw of experiment 2's model, not one
of the doc's 30 Monte Carlo trials (main() seeds experiment 2 with 100), so its
numbers differ from the doc's tables. Checked on seeds 0-3: cam_pose_drift
position NEES 1.1-3.5 and cam_pose 1300-47000 every time (gated cam_pose
35-870), but the error ordering is seed-dependent: cam_pose's mean error is
above legs' in seeds 0, 2 and 3 and below it in seed 1. In seed 2 the gate
rejects 74% of cam_pose's readings after the jump, short of the doc's lockout
(under 5% used). Swing legs are drawn faded at their nominal offsets (the
script does not simulate swing feet; illustrative only). Everything else is
the simulation's own simulate_trial / Camera / make_filter / camera_update
output.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse

from scipy.stats import chi2
from state_estimation_simulations.use_numpy.hexapod_camera_fusion_inekf import (
    Camera, build_parser, camera_settings, make_filter)
from state_estimation_simulations.use_numpy.hexapod_contact_inekf import (
    Config, NOMINAL_FOOT_OFFSETS, simulate_trial, world_from_invariant)

# (name, filter kind, gated)
RUNS = (("legs", "legs", False), ("cam_pose", "cam_pose", False),
        ("cam_pose_gated", "cam_pose", True), ("cam_pose_drift", "cam_pose_drift", False))
COLORS = {"legs": "0.35", "cam_pose": "tab:red", "cam_pose_gated": "tab:orange", "cam_pose_drift": "tab:blue"}
LABELS = {"legs": "legs only", "cam_pose": "camera pose as world pose", "cam_pose_gated": "same, gated",
          "cam_pose_drift": "camera pose, drift as states"}
STYLES = {"legs": ("--", 1.6), "cam_pose": ("--", 2.6), "cam_pose_gated": (":", 2.0), "cam_pose_drift": ("--", 1.6)}
CAM_COLOR = "tab:purple"
GAIT = "tripod"


def heading(R):
    return np.arctan2(R[1, 0], R[0, 0])


def run(kind, gating, trial, camera, cfg, cam):
    """Drives one filter like hexapod_camera_fusion_inekf.run_filter, also
    keeping the estimated position and heading for drawing."""
    stance, y = trial.feet[GAIT]
    ekf = make_filter(kind, trial, cfg, cam, gating)
    n = len(stance)
    out = dict(p=np.zeros((n, 3)), yaw=np.zeros(n), pos_err=np.zeros(n), nees=np.zeros(n), used=np.zeros(n, dtype=bool))
    for k in range(n):
        if k > 0:
            ekf.predict(trial.gyro_meas[k - 1], trial.accel_meas[k - 1])
        ekf.step(stance[k], y[k])
        if kind != "legs" and camera.ok[k]:
            out["used"][k] = ekf.camera_update((camera.p[k], camera.R[k], camera.v[k]))
        T = world_from_invariant(ekf.v, ekf.p)
        e_p = ekf.p - trial.p[k]
        out["p"][k], out["yaw"][k] = ekf.p, heading(ekf.R)
        out["pos_err"][k] = np.linalg.norm(e_p)
        out["nees"][k] = e_p @ np.linalg.solve((T @ ekf.P[:9, :9] @ T.T)[6:9, 6:9], e_p)
    return out


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument("--frame-step", type=int, default=28, help="Filter ticks per GIF frame")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.set_defaults(out="hexapod_camera_fusion_inekf.gif", seed=2)
    args = parser.parse_args()

    d = Config()
    cfg = replace(d, n_steps=int(round(args.duration / d.dt)), twist0=(args.speed, 0.0, args.turn_rate))
    cam = camera_settings(args, True)
    rng = np.random.default_rng(args.seed)
    trial = simulate_trial(cfg, [GAIT], rng)
    camera = Camera(trial, cfg, rng, cam)
    stance, _ = trial.feet[GAIT]
    est = {name: run(kind, gating, trial, camera, cfg, cam) for name, kind, gating in RUNS}
    n_steps = cfg.n_steps
    t = np.arange(n_steps + 1) * cfg.dt
    true_yaw = np.array([heading(R) for R in trial.R])
    cam_yaw = np.array([heading(R) for R in camera.R])
    cam_err = np.linalg.norm(camera.p - trial.p, axis=1)
    t_jump = camera.jump_tick * cfg.dt
    kj = camera.jump_tick
    jump_cm = 100 * np.linalg.norm((camera.p - trial.p)[kj:kj + 5].mean(axis=0) - (camera.p - trial.p)[kj - 5:kj].mean(axis=0))
    dropped = np.where(~camera.ok)[0]
    t_drop = (dropped[0] * cfg.dt, (dropped[-1] + 1) * cfg.dt)

    skip = int(round(1.0 / cfg.dt))
    print(f"Seed {args.seed}, experiment 2 settings, {args.duration:g} s tripod; camera scale error {100 * camera.scale:+.2f}%, "
          f"pose jump about {jump_cm:.0f} cm at {t_jump:.1f} s, "
          f"dropout {t_drop[0]:.1f}-{t_drop[1]:.1f} s")
    print(f"  camera reading error: {100 * cam_err[-1]:.1f} cm at the end (mean {100 * cam_err[skip:].mean():.1f} cm)")
    lo, hi = chi2.ppf([0.025, 0.975], 3)   # one NEES sample (3 DoF), exact
    print(f"  consistent position NEES 3 (one sample: 95% in [{lo:.2f}, {hi:.2f}]); averages after the first second:")
    for name, kind, gating in RUNS:
        e = est[name]
        after = e["used"][camera.jump_tick:][camera.ok[camera.jump_tick:]]
        used = f" | readings used after the jump {100 * after.mean():.0f}%" if kind != "legs" else ""
        print(f"  {name:<15s} pos err {100 * e['pos_err'][skip:].mean():6.2f} cm (end {100 * e['pos_err'][-1]:6.2f}) | "
              f"pos NEES {e['nees'][skip:].mean():10.2f}{used}")

    resumed = kj + np.argmax(est["cam_pose_gated"]["used"][kj:])
    print(f"  gated cam_pose used no reading from the jump until {t[resumed]:.1f} s (the dropout ends at {t_drop[1]:.1f} s)")

    fig = plt.figure(figsize=(12, 7.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.1], height_ratios=[3, 3, 1.3], hspace=0.5, wspace=0.22)
    ax_m = fig.add_subplot(gs[:, 0])
    ax_e = fig.add_subplot(gs[0, 1])
    ax_n = fig.add_subplot(gs[1, 1], sharex=ax_e)
    ax_u = fig.add_subplot(gs[2, 1], sharex=ax_e)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.25)
    fig.suptitle("Tracking camera on a hexapod: its pose drifts and jumps, so the filter must model the drift",
                 fontsize=12, fontweight="bold")

    # map panel: a window that follows the robot
    half, trail = 0.6, int(round(3.0 / cfg.dt))
    ax_m.set_aspect("equal")
    ax_m.set(xlabel="x (m)", ylabel="y (m)")
    true_trail, = ax_m.plot([], [], color="green", lw=2, label="true path", zorder=6)
    cam_trail, = ax_m.plot([], [], color=CAM_COLOR, lw=1.5, label="camera reading (position, heading)", zorder=6)
    cam_dot, = ax_m.plot([], [], "D", color=CAM_COLOR, ms=7, zorder=8)
    cam_nose, = ax_m.plot([], [], color=CAM_COLOR, lw=2, zorder=8)
    est_trails = {n: ax_m.plot([], [], color=COLORS[n], lw=1.2, alpha=0.8, zorder=5)[0] for n, _, _ in RUNS}
    body = Ellipse((0, 0), 0.42, 0.24, fc="#e8eef6", ec="#5b6b7f", lw=1.4, alpha=0.6, zorder=3)
    ax_m.add_patch(body)
    est_bodies = {n: Ellipse((0, 0), 0.42, 0.24, fc="none", ec=COLORS[n], lw=STYLES[n][1], ls=STYLES[n][0], zorder=7) for n, _, _ in RUNS}
    for e in est_bodies.values():
        ax_m.add_patch(e)
    nose, = ax_m.plot([], [], color="#5b6b7f", lw=1.4, zorder=4)
    legs = [ax_m.plot([], [], color="#5b6b7f", lw=1.6, zorder=2)[0] for _ in range(6)]
    foot_marks = [ax_m.plot([], [], "o", ms=7, mec="black", zorder=5)[0] for _ in range(6)]
    ax_m.plot([], [], "o", ms=7, mfc="black", mec="black", label="true stance foot")
    ax_m.plot([], [], "o", ms=7, mfc="white", mec="black", alpha=0.4, label="swing leg (illustrative)")
    for n, _, _ in RUNS:
        ax_m.plot([], [], color=COLORS[n], lw=STYLES[n][1], ls=STYLES[n][0], label=LABELS[n])
    ax_m.legend(title="estimates (dashed outline, trail, error and NEES curves)", title_fontsize=8, alignment="left", loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, ncol=2, frameon=False)
    status = ax_m.text(0.02, 0.98, "", transform=ax_m.transAxes, va="top", fontsize=9, fontweight="bold")
    event = ax_m.text(0.5, 0.03, "", transform=ax_m.transAxes, ha="center", va="bottom", fontsize=10,
                      fontweight="bold", color=CAM_COLOR, bbox=dict(fc="white", ec=CAM_COLOR, alpha=0.9))

    # error panel
    ax_e.plot(t, 100 * cam_err, color=CAM_COLOR, lw=1, alpha=0.2)
    cam_err_line, = ax_e.plot([], [], color=CAM_COLOR, lw=1.8, ls=":")
    err_lines = {}
    for n, _, _ in RUNS:
        ax_e.plot(t, 100 * est[n]["pos_err"], color=COLORS[n], lw=1, alpha=0.2)
        err_lines[n], = ax_e.plot([], [], color=COLORS[n], lw=1.8, label=LABELS[n])
    top = 100 * max(cam_err.max(), *(est[n]["pos_err"].max() for n, _, _ in RUNS)) * 1.1
    ax_e.set(ylim=(0, top), ylabel="position error (cm)", title="Position error (purple dotted: the camera reading itself)")

    # NEES panel
    nees_lines = {}
    for n, _, _ in RUNS:
        ax_n.plot(t, est[n]["nees"], color=COLORS[n], lw=1, alpha=0.2)
        nees_lines[n], = ax_n.plot([], [], color=COLORS[n], lw=1.2)
    ax_n.axhspan(lo, hi, color="0.85")
    ax_n.axhline(3, color="black", lw=0.8)
    ax_n.text(t[-1], 3.3, "consistent: 3 ", va="bottom", ha="right", fontsize=8)
    ax_n.set(yscale="log", ylabel="position NEES", title="Position NEES (one run; grey band: 95% of a consistent filter)")
    ax_n.set_ylim(0.05, max(est[n]["nees"].max() for n, _, _ in RUNS) * 3)

    # camera availability strip
    ax_u.set(ylim=(-0.5, 1.5), yticks=[0, 1], yticklabels=["gated\nused", "camera\navailable"],
             xlabel="time (s)")
    ax_u.tick_params(axis="y", labelsize=7)
    avail_line, = ax_u.plot([], [], "|", color=CAM_COLOR, ms=8, mew=1)
    used_line, = ax_u.plot([], [], "|", color=COLORS["cam_pose_gated"], ms=8, mew=1)
    used_g = est["cam_pose_gated"]["used"]

    for ax in (ax_e, ax_n, ax_u):
        ax.axvspan(*t_drop, color="0.5", alpha=0.18, lw=0)
        ax.axvline(t_jump, color=CAM_COLOR, lw=1.2, ls="--")
        ax.set_xlim(0, t[-1])
    ax_e.text(t_jump, top * 0.97, " pose jump", color=CAM_COLOR, fontsize=8, va="top")
    ax_e.text(sum(t_drop) / 2, top * 0.97, "dropout", color="0.3", fontsize=8, va="top", ha="center")
    plt.setp(ax_e.get_xticklabels(), visible=False)
    plt.setp(ax_n.get_xticklabels(), visible=False)
    kj_ok = camera.ok[kj:]
    rejected = 100 * (1 - used_g[kj:][kj_ok].mean())
    fig.text(0.5, 0.012, f"The camera's frame drifts (scale error, yaw drift) and jumps {jump_cm:.0f} cm. Used as the world pose, its readings drag "
             f"the filter along (NEES in the thousands);\nthe gate rejects {rejected:.0f}% of them after the jump. With the frame's offset as "
             "states, the filter stays consistent and absorbs the jump.", ha="center", fontsize=9)

    def update(frame):
        k = min(frame * args.frame_step, n_steps)
        sl, tr = slice(0, k + 1), slice(max(0, k - trail), k + 1)
        p, th = trial.p[k, :2], true_yaw[k]
        Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        ax_m.set_xlim(p[0] - half, p[0] + half)
        ax_m.set_ylim(p[1] - half, p[1] + half)
        body.set_center(p)
        body.set_angle(np.degrees(th))
        tip = p + Rz @ np.array([0.17, 0.0])
        nose.set_data([p[0], tip[0]], [p[1], tip[1]])
        d_k = trial.foot_pos[GAIT][k]
        for i in range(6):
            hip = p + Rz @ np.array([NOMINAL_FOOT_OFFSETS[i, 0] * 0.6, np.sign(NOMINAL_FOOT_OFFSETS[i, 1]) * 0.1])
            if stance[k, i]:
                f, fc, alpha = d_k[i, :2], "black", 1.0
            else:
                f, fc, alpha = p + Rz @ NOMINAL_FOOT_OFFSETS[i], "white", 0.4
            legs[i].set_data([hip[0], f[0]], [hip[1], f[1]])
            legs[i].set_alpha(alpha)
            foot_marks[i].set_data([f[0]], [f[1]])
            foot_marks[i].set_markerfacecolor(fc)
            foot_marks[i].set_alpha(alpha)
        true_trail.set_data(trial.p[tr, 0], trial.p[tr, 1])
        cam_trail.set_data(camera.p[tr, 0], camera.p[tr, 1])
        cp, ch = camera.p[k, :2], cam_yaw[k]
        cam_dot.set_data([cp[0]], [cp[1]])
        cam_dot.set_markerfacecolor(CAM_COLOR if camera.ok[k] else "white")
        cam_nose.set_data([cp[0], cp[0] + 0.17 * np.cos(ch)], [cp[1], cp[1] + 0.17 * np.sin(ch)])
        for n, _, _ in RUNS:
            e = est[n]
            est_trails[n].set_data(e["p"][tr, 0], e["p"][tr, 1])
            est_bodies[n].set_center(e["p"][k, :2])
            est_bodies[n].set_angle(np.degrees(e["yaw"][k]))
            err_lines[n].set_data(t[sl], 100 * e["pos_err"][sl])
            nees_lines[n].set_data(t[sl], e["nees"][sl])
        cam_err_line.set_data(t[sl], 100 * cam_err[sl])
        avail = np.where(camera.ok[sl])[0]
        used = np.where(used_g[sl])[0]
        avail_line.set_data(t[avail], np.ones(len(avail)))
        used_line.set_data(t[used], np.zeros(len(used)))
        status.set_text(f"t = {t[k]:.1f} s")
        if not camera.ok[k]:
            event.set_text("camera dropout: no readings")
        elif 0 <= t[k] - t_jump < 3.0:
            event.set_text(f"camera pose jumped (about {jump_cm:.0f} cm)")
        else:
            event.set_text("")
        event.set_visible(bool(event.get_text()))
        return []

    n_frames = n_steps // args.frame_step + 1
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {n_frames} frames to {args.out}")


if __name__ == "__main__":
    main()
