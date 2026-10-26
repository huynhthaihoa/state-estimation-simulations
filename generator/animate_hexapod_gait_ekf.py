'''
Animated version of hexapod_gait_ekf.py's slip experiment, one run: a planar
hexapod walks its random-walk path while leg L2 slips sideways for a time
window, and two EKFs track it from the stance feet's relative-velocity
readings alone. They are identical except that one trusts every stance foot
and the other drops any foot whose innovation fails a per-foot chi-squared
test (99%, 2 DoF).

Two panels, drawn progressively:
  1. top-down view: the true body and its legs, with each stance foot at its
     simulated foothold (filled; red while L2 slips), the true path, and both
     filters' estimated paths;
  2. sideways body velocity v: the truth and both estimates (the slip window
     shaded).

Message (doc docs/filtering/hexapod_gait_ekf.md section 3): while L2 slips,
trusting every foot drags the velocity estimate off whenever L2 is down and
the position estimate drifts away; the gated filter drops L2's readings and
stays on the truth.

Swing feet are drawn faded, at their simulated positions (simulate_feet
moves each one along a straight line between footholds with a smooth speed
profile). The truth, footholds, readings and estimates are the simulation's own
simulate_trial / run_ekf outputs. This is the single run from main()'s
seed (same rng streams), with the slip switched on.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse

from state_estimation_simulations.use_numpy.hexapod_gait_ekf import (GAITS, LEG_NAMES, NOMINAL_FOOT_OFFSETS, process_noise_covariance, rot, run_ekf,
                              simulate_trial, state_errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gait", type=str, default="tripod", choices=list(GAITS), help="Gait to animate")
    parser.add_argument("--duration", type=float, default=12.0, help="Simulation length in seconds")
    parser.add_argument("--dt", type=float, default=0.02, help="Filter/measurement step interval in seconds")
    parser.add_argument("--cycle-period", type=float, default=1.2, help="Gait cycle period (s)")
    parser.add_argument("--forward-speed", type=float, default=0.1, help="Initial forward speed u (m/s)")
    parser.add_argument("--turn-rate", type=float, default=0.2, help="Initial turn rate omega (rad/s)")
    parser.add_argument("--lin-accel-std", type=float, default=0.05, help="Linear-acceleration std-dev (m/s^2)")
    parser.add_argument("--ang-accel-std", type=float, default=0.1, help="Angular-acceleration std-dev (rad/s^2)")
    parser.add_argument("--meas-noise-std", type=float, default=0.02, help="Foot reading std-dev per axis (m/s)")
    parser.add_argument("--slip-leg", type=str, default="L2", choices=LEG_NAMES, help="Leg that slips")
    parser.add_argument("--slip-start", type=float, default=4.0, help="Slip window start (s)")
    parser.add_argument("--slip-end", type=float, default=8.0, help="Slip window end (s)")
    parser.add_argument("--slip-speed", type=float, default=0.1, help="Outward sideways slip speed (m/s)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--frame-step", type=int, default=2, help="Filter ticks per GIF frame")
    parser.add_argument("--fps", type=int, default=20, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.add_argument("--out", type=str, default="hexapod_gait_ekf.gif", help="Output GIF path")
    args = parser.parse_args()

    tpc = int(round(args.cycle_period / args.dt))
    n_steps = int(round(args.duration / args.dt))
    twist0 = np.array([args.forward_speed, 0.0, args.turn_rate])
    init_std = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])  # hexapod_gait_ekf.main defaults
    slip_leg = LEG_NAMES.index(args.slip_leg)
    side = 1.0 if args.slip_leg.startswith("L") else -1.0
    slip = (slip_leg, args.slip_start, args.slip_end, np.array([0.0, side * args.slip_speed]))

    # Same rng streams as hexapod_gait_ekf.main's single run, plus the slip
    x_true, x_init, feet = simulate_trial(n_steps, args.dt, tpc, [args.gait], twist0, args.lin_accel_std,
                                          args.ang_accel_std, args.meas_noise_std, init_std, args.forward_speed,
                                          np.random.default_rng(args.seed), slip)
    stance, r, z = feet[args.gait]
    Q = process_noise_covariance(args.dt, args.lin_accel_std, args.ang_accel_std)
    P_init = np.diag(init_std ** 2)
    est, acc = {}, {}
    for gating in (False, True):
        est[gating], _, acc[gating] = run_ekf(x_init, P_init, z, r, stance, args.dt, Q, args.meas_noise_std ** 2,
                                              gating)

    t = np.arange(n_steps + 1) * args.dt
    slipping = stance[:, slip_leg] & (t >= args.slip_start) & (t < args.slip_end)
    print(f"{args.gait}, {args.slip_leg} slipping at {args.slip_speed} m/s for t in "
          f"[{args.slip_start}, {args.slip_end}) s, seed {args.seed}:")
    for gating in (False, True):
        err = state_errors(x_true, est[gating])
        name = "per-foot gate" if gating else "trust every foot"
        print(f"  {name:<16s} final pos err={np.linalg.norm(err[-1, :2]):.4f} m | "
              f"in-window RMS v err={np.sqrt(np.mean(err[(t >= args.slip_start) & (t < args.slip_end), 4] ** 2)):.4f} m/s"
              f" | slip readings used {acc[gating][slipping, slip_leg].mean():.0%}")

    colors = {False: "tab:red", True: "tab:blue"}
    labels = {False: "trust every foot", True: "per-foot χ² gate"}
    fig = plt.figure(figsize=(11, 7.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.25)
    ax_m = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1])
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.27)
    fig.suptitle(f"Hexapod leg odometry ({args.gait} gait): {args.slip_leg} slips for "
                 f"{args.slip_end - args.slip_start:.0f} s", fontsize=12, fontweight="bold")

    # map panel
    pad = 0.3
    ax_m.set_aspect("equal")
    ax_m.set_xlim(x_true[:, 0].min() - pad, x_true[:, 0].max() + pad)
    ax_m.set_ylim(x_true[:, 1].min() - pad, x_true[:, 1].max() + pad)
    ax_m.set(xlabel="x (m)", ylabel="y (m)")
    ax_m.plot(x_true[:, 0], x_true[:, 1], color="green", lw=1, ls="--", alpha=0.3)
    truth_path, = ax_m.plot([], [], color="green", lw=2, label="true path", zorder=6)
    est_paths = {g: ax_m.plot([], [], color=colors[g], lw=1.6, label=f"estimate: {labels[g]}", zorder=6)[0] for g in est}
    body = Ellipse((0, 0), 0.42, 0.24, fc="#e8eef6", ec="#5b6b7f", lw=1.4, alpha=0.55, zorder=3)
    ax_m.add_patch(body)
    heading, = ax_m.plot([], [], color="#5b6b7f", lw=1.4, zorder=4)
    legs = [ax_m.plot([], [], color="#5b6b7f", lw=1.6, zorder=2)[0] for _ in range(6)]
    foot_marks = [ax_m.plot([], [], "o", ms=7, mec="black", zorder=5)[0] for _ in range(6)]
    ax_m.plot([], [], "o", ms=7, mfc="black", mec="black", label="stance foot (simulated foothold)")
    ax_m.plot([], [], "o", ms=7, mfc="white", mec="black", alpha=0.4, label="swing foot (simulated)")
    ax_m.plot([], [], "o", ms=7, mfc="red", mec="black", label=f"{args.slip_leg} slipping")
    ax_m.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8, ncol=2, frameon=False)
    status = ax_m.text(0.02, 0.98, "", transform=ax_m.transAxes, va="top", fontsize=9, fontweight="bold")

    # velocity panel
    ax_v.axvspan(args.slip_start, args.slip_end, color="gold", alpha=0.25, lw=0, label=f"{args.slip_leg} slipping")
    ax_v.plot(t, x_true[:, 4], color="green", lw=1.2, ls="--", alpha=0.3)
    v_true, = ax_v.plot([], [], color="green", lw=2, label="true v")
    v_est = {g: ax_v.plot([], [], color=colors[g], lw=1.4, label=labels[g])[0] for g in est}
    lo = min(x_true[:, 4].min(), min(e[1:, 4].min() for e in est.values()))
    hi = max(x_true[:, 4].max(), max(e[1:, 4].max() for e in est.values()))
    ax_v.set(xlim=(0, t[-1]), ylim=(lo - 0.01, hi + 0.01), xlabel="time (s)",
             ylabel="sideways body velocity v (m/s)", title="What the two filters believe")
    ax_v.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8, ncol=2, frameon=False)
    fig.text(0.5, 0.02, "Trusting every foot: L2's sliding reading pulls v off by ~1/3 of the slip whenever L2 is down "
             "(tripod),\nand the estimated path drifts away. The gate drops those readings and stays on the truth.", ha="center", fontsize=9)

    def update(frame):
        k = min(frame * args.frame_step, n_steps)
        sl = slice(0, k + 1)
        p, th = x_true[k, :2], x_true[k, 2]
        R = rot(th)
        body.set_center(p)
        body.set_angle(np.degrees(th))
        nose = p + R @ np.array([0.17, 0.0])
        heading.set_data([p[0], nose[0]], [p[1], nose[1]])
        for i in range(6):
            hip = p + R @ np.array([NOMINAL_FOOT_OFFSETS[i, 0] * 0.6, np.sign(NOMINAL_FOOT_OFFSETS[i, 1]) * 0.1])
            if stance[k, i]:
                f = p + R @ r[k, i]
                fc = "red" if slipping[k] and i == slip_leg else "black"
                alpha = 1.0
            else:
                f = p + R @ r[k, i]
                fc, alpha = "white", 0.4
            legs[i].set_data([hip[0], f[0]], [hip[1], f[1]])
            legs[i].set_alpha(alpha)
            foot_marks[i].set_data([f[0]], [f[1]])
            foot_marks[i].set_markerfacecolor(fc)
            foot_marks[i].set_alpha(alpha)
        truth_path.set_data(x_true[sl, 0], x_true[sl, 1])
        for g in est:
            est_paths[g].set_data(est[g][sl, 0], est[g][sl, 1])
            v_est[g].set_data(t[sl], est[g][sl, 4])
        v_true.set_data(t[sl], x_true[sl, 4])
        if args.slip_start <= t[k] < args.slip_end:
            status.set_text(f"t = {t[k]:.1f} s   slip window: " + (f"{args.slip_leg} down and SLIDING" if slipping[k]
                                                                    else f"{args.slip_leg} in swing, no reading"))
            status.set_color("red")
        else:
            status.set_text(f"t = {t[k]:.1f} s")
            status.set_color("black")
        return []

    n_frames = n_steps // args.frame_step + 1
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {n_frames} frames to {args.out}")


if __name__ == "__main__":
    main()
