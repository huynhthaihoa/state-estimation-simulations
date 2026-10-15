'''
Animated version of hexapod_contact_inekf.py, one run: a 3D hexapod walks its
turning path (tripod gait) while two contact-aided filters track it from an
IMU plus leg kinematics, with each stance foot's world position in the state.
They see identical data and start from the same estimate, 0.1 rad initial
rotation std; one is the right-invariant EKF on SE_{K+2}(3), the other the
ordinary world-frame error-state EKF.

Three panels, drawn progressively:
  1. top-down view: the true body, its legs and the true stance feet (black),
     and each filter's estimated body outline and estimated stance feet
     (the feet in its state);
  2. yaw error of each filter with its reported +-2 sigma band;
  3. position error of each filter.

Message (doc docs/filtering/hexapod_contact_inekf.md section 3): yaw cannot be
observed from an IMU plus leg kinematics. The InEKF knows that: its yaw
error stays where it started and its band never shrinks. The world EKF's
band shrinks as if the feet told it the heading; its updates then pull the
yaw off and the path rotates away.

Seed 4 by default (not main()'s seed 0): seed 0 draws an initial yaw error of
only 0.08 deg, too small to see. Over 50 trials the two filters' yaw RMSE is
similar (6.2 vs 6.8 deg at 0.1 rad, see main()); what differs is the reported
sigma. Swing legs are drawn faded at their nominal offsets (the script does
not simulate swing feet; illustrative only). Everything else is the
simulation's own simulate_trial / make_filter / step output.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse

from hexapod_contact_inekf import FILTERS, Config, NOMINAL_FOOT_OFFSETS, make_filter, simulate_trial, so3_log


def heading(R):
    return np.arctan2(R[1, 0], R[0, 0])


def run(kind, trial, gait, cfg):
    """Drives one filter like run_filter, also keeping the estimated heading
    and the estimated stance feet for drawing."""
    stance, y = trial.feet[gait]
    ekf = make_filter(kind, trial, cfg)
    out = dict(p=[], yaw=[], feet=[], yaw_err=[], yaw_sigma=[], pos_err=[])
    for k in range(len(stance)):
        if k > 0:
            ekf.predict(trial.gyro_meas[k - 1], trial.accel_meas[k - 1])
        ekf.step(stance[k], y[k])
        out["p"].append(ekf.p.copy())
        out["yaw"].append(heading(ekf.R))
        out["feet"].append(dict(zip(ekf.legs, ekf.d.copy())))
        out["yaw_err"].append(np.degrees(so3_log(ekf.R @ trial.R[k].T)[2]))
        out["yaw_sigma"].append(np.degrees(np.sqrt(ekf.P[2, 2])))
        out["pos_err"].append(np.linalg.norm(ekf.p - trial.p[k]))
    return {key: (val if key == "feet" else np.array(val)) for key, val in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gait", type=str, default="tripod", choices=["tripod", "ripple", "wave"], help="Gait")
    parser.add_argument("--init-rot-std", type=float, default=0.1, help="Initial rotation std, per axis (rad)")
    parser.add_argument("--seed", type=int, default=4, help="RNG seed (see the module docstring)")
    parser.add_argument("--frame-step", type=int, default=3, help="Filter ticks per GIF frame")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.add_argument("--out", type=str, default="hexapod_contact_inekf.gif", help="Output GIF path")
    args = parser.parse_args()

    cfg = replace(Config(), init_rot_std=args.init_rot_std)
    trial = simulate_trial(cfg, [args.gait], np.random.default_rng(args.seed))
    stance, _ = trial.feet[args.gait]
    est = {kind: run(kind, trial, args.gait, cfg) for kind in FILTERS}
    n_steps = cfg.n_steps
    t = np.arange(n_steps + 1) * cfg.dt
    true_yaw = np.array([heading(R) for R in trial.R])
    print(f"{args.gait}, init rotation std {args.init_rot_std} rad, seed {args.seed}:")
    for kind in FILTERS:
        e = est[kind]
        print(f"  {kind:<6s} yaw err {e['yaw_err'][0]:+.2f} -> {e['yaw_err'][-1]:+.2f} deg "
              f"(reported 2 sigma {2 * e['yaw_sigma'][-1]:.2f}) | final pos err {e['pos_err'][-1]:.3f} m")

    colors = {"inekf": "tab:blue", "world": "tab:red"}
    labels = {"inekf": "invariant EKF", "world": "world-frame EKF"}
    fig = plt.figure(figsize=(12, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], hspace=0.45, wspace=0.25)
    ax_m = fig.add_subplot(gs[:, 0])
    ax_y = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.24)
    fig.suptitle(f"Contact-aided EKF on a hexapod ({args.gait} gait): yaw is unobservable, "
                 f"only one filter knows it", fontsize=12, fontweight="bold")

    # map panel
    pad = 0.3
    ax_m.set_aspect("equal")
    ax_m.set_xlim(trial.p[:, 0].min() - pad, trial.p[:, 0].max() + pad)
    ax_m.set_ylim(trial.p[:, 1].min() - pad, trial.p[:, 1].max() + pad)
    ax_m.set(xlabel="x (m)", ylabel="y (m)")
    ax_m.plot(trial.p[:, 0], trial.p[:, 1], color="green", lw=1, ls="--", alpha=0.3)
    truth_path, = ax_m.plot([], [], color="green", lw=2, label="true path", zorder=6)
    est_paths = {k: ax_m.plot([], [], color=colors[k], lw=1.5, label=f"{labels[k]} path", zorder=6)[0] for k in FILTERS}
    body = Ellipse((0, 0), 0.42, 0.24, fc="#e8eef6", ec="#5b6b7f", lw=1.4, alpha=0.6, zorder=3)
    ax_m.add_patch(body)
    est_bodies = {k: Ellipse((0, 0), 0.42, 0.24, fc="none", ec=colors[k], lw=1.4, ls="--", zorder=4) for k in FILTERS}
    for e in est_bodies.values():
        ax_m.add_patch(e)
    nose, = ax_m.plot([], [], color="#5b6b7f", lw=1.4, zorder=4)
    legs = [ax_m.plot([], [], color="#5b6b7f", lw=1.6, zorder=2)[0] for _ in range(6)]
    foot_marks = [ax_m.plot([], [], "o", ms=7, mec="black", zorder=5)[0] for _ in range(6)]
    est_feet = {k: ax_m.plot([], [], "x", ms=7, mew=2, color=colors[k], zorder=7)[0] for k in FILTERS}
    ax_m.plot([], [], "o", ms=7, mfc="black", mec="black", label="true stance foot")
    ax_m.plot([], [], "o", ms=7, mfc="white", mec="black", alpha=0.4, label="swing leg (illustrative)")
    for k in FILTERS:
        ax_m.plot([], [], "x", ms=7, mew=2, color=colors[k], label=f"{labels[k]}: body + feet in state")
    ax_m.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8, ncol=2, frameon=False)
    status = ax_m.text(0.02, 0.98, "", transform=ax_m.transAxes, va="top", fontsize=9, fontweight="bold")

    # yaw panel
    yaw_lines, yaw_bands = {}, {}
    for k in FILTERS:
        ax_y.plot(t, est[k]["yaw_err"], color=colors[k], lw=1, alpha=0.2)
        yaw_lines[k], = ax_y.plot([], [], color=colors[k], lw=1.8, label=f"{labels[k]}: yaw error")
        yaw_bands[k] = None
    ax_y.axhline(0.0, color="black", lw=0.8)
    lim = max(max(np.abs(est[k]["yaw_err"]).max(), 2 * est[k]["yaw_sigma"].max()) for k in FILTERS) * 1.1
    ax_y.set(xlim=(0, t[-1]), ylim=(-lim, lim), xlabel="time (s)", ylabel="yaw error (deg)",
             title="Yaw error and the filter's own ±2σ (shaded)")
    ax_y.legend(loc="lower left", fontsize=8)

    # position panel
    pos_lines = {}
    for k in FILTERS:
        ax_p.plot(t, est[k]["pos_err"], color=colors[k], lw=1, alpha=0.2)
        pos_lines[k], = ax_p.plot([], [], color=colors[k], lw=1.8, label=labels[k])
    ax_p.set(xlim=(0, t[-1]), ylim=(0, max(est[k]["pos_err"].max() for k in FILTERS) * 1.1), xlabel="time (s)",
             ylabel="position error (m)", title="Position error")
    ax_p.legend(loc="upper left", fontsize=8)
    fig.text(0.5, 0.012, "Rotating the whole world about gravity changes no IMU or leg reading, so yaw can only lose "
             "information.\nInvariant EKF: band never shrinks, error stays at its start. World EKF: band shrinks "
             "as if the feet fixed the heading, and its updates drag the yaw (and the path) off.",
             ha="center", fontsize=9)

    def update(frame):
        k = min(frame * args.frame_step, n_steps)
        sl = slice(0, k + 1)
        p, th = trial.p[k, :2], true_yaw[k]
        Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        body.set_center(p)
        body.set_angle(np.degrees(th))
        tip = p + Rz @ np.array([0.17, 0.0])
        nose.set_data([p[0], tip[0]], [p[1], tip[1]])
        d_k = trial.foot_pos[args.gait][k]
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
        truth_path.set_data(trial.p[sl, 0], trial.p[sl, 1])
        for kind in FILTERS:
            e = est[kind]
            est_paths[kind].set_data(e["p"][sl, 0], e["p"][sl, 1])
            est_bodies[kind].set_center(e["p"][k, :2])
            est_bodies[kind].set_angle(np.degrees(e["yaw"][k]))
            fe = np.array(list(e["feet"][k].values())) if e["feet"][k] else np.zeros((0, 3))
            est_feet[kind].set_data(fe[:, 0], fe[:, 1])
            yaw_lines[kind].set_data(t[sl], e["yaw_err"][sl])
            if yaw_bands[kind] is not None:
                yaw_bands[kind].remove()
            yaw_bands[kind] = ax_y.fill_between(t[sl], -2 * e["yaw_sigma"][sl], 2 * e["yaw_sigma"][sl],
                                                color=colors[kind], alpha=0.15, lw=0)
            pos_lines[kind].set_data(t[sl], e["pos_err"][sl])
        status.set_text(f"t = {t[k]:.1f} s")
        return []

    n_frames = n_steps // args.frame_step + 1
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {n_frames} frames to {args.out}")


if __name__ == "__main__":
    main()
