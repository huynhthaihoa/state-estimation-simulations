'''
Animated version of inchworm_zupt_ekf.py's default run: a 1D point mass crawls
in five anchor (dwell, exactly stationary) / extend (ramp up, cruise, ramp
down) cycles while three EKFs track it from noisy position measurements. They
share the predict step and the position update and differ only in when the
zero-velocity pseudo-measurement (ZUPT) is also applied: never, always, or
phase_conditional (only on the known anchor ticks).

Two panels share the time axis and are drawn progressively:
  1. velocity: the true velocity and the three estimates, each with a +-2 sigma_v
     band (shaded background = anchor phases);
  2. consistency: the Monte Carlo averaged NEES of each filter (log scale) with
     the expected value 2 (2 DoF) and the 95% band for an average over the trials.

Message (doc docs/filtering/inchworm_zupt_ekf.md section 3): `always` collapses
sigma_v during motion and its NEES grows cycle after cycle, far worse than the
others in both the anchor and the cruise windows. `phase_conditional` tightens
sigma_v only on anchor ticks; at the defaults that extra confidence makes the
ramp-up hurt more than `never` (the ramps, not ZUPT, keep every variant
inconsistent at the defaults: true ramp acceleration 0.5 m/s^2 vs the assumed
0.15).

Not claimed: that any variant is best. At the defaults `never` is better
calibrated in cruise and more accurate in position than `phase_conditional`.

The per-step estimates are the simulation's own run_ekf_* outputs (same rng
order as inchworm_zupt_ekf.main); the Monte Carlo curve is its
run_monte_carlo_consistency. The default Monte Carlo (500 trials) takes ~45 s.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

from inchworm_zupt_ekf import (generate_ground_truth_and_data, run_ekf_never_zupt,
                               run_ekf_always_zupt, run_ekf_phase_conditional_zupt,
                               run_monte_carlo_consistency, averaged_nees_bounds, state_errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation length in seconds")
    parser.add_argument("--dt", type=float, default=0.05, help="Filter/measurement step interval in seconds")
    parser.add_argument("--t-anchor", type=float, default=1.0, help="Anchor/dwell phase duration per cycle (s)")
    parser.add_argument("--t-extend", type=float, default=1.0, help="Total extend phase duration per cycle (s)")
    parser.add_argument("--t-ramp", type=float, default=0.2, help="Ramp-up/ramp-down duration (s)")
    parser.add_argument("--v-extend", type=float, default=0.1, help="Commanded cruise speed (m/s)")
    parser.add_argument("--pos-noise-std", type=float, default=0.02, help="Position measurement noise std-dev (m)")
    parser.add_argument("--process-noise-std", type=float, default=0.15,
                        help="Assumed acceleration-disturbance noise std-dev (m/s^2)")
    parser.add_argument("--zupt-noise-std", type=float, default=0.01, help="ZUPT pseudo-measurement noise std-dev (m/s)")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.05, help="Initial position error std-dev (m)")
    parser.add_argument("--init-vel-noise-std", type=float, default=0.05, help="Initial velocity error std-dev (m/s)")
    parser.add_argument("--n-mc-trials", type=int, default=500, help="Number of Monte Carlo consistency trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--fps", type=int, default=20, help="GIF frames per second (one frame per filter tick)")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.add_argument("--out", type=str, default="inchworm_zupt_ekf.gif", help="Output GIF path")
    args = parser.parse_args()

    # Same rng consumption order as inchworm_zupt_ekf.main
    rng = np.random.default_rng(args.seed)
    x_true, z, is_anchor, is_cruise = generate_ground_truth_and_data(
        args.duration, args.dt, args.t_anchor, args.t_extend, args.t_ramp, args.v_extend,
        args.pos_noise_std, rng)
    n_steps = len(x_true) - 1
    init_std = np.array([args.init_pos_noise_std, args.init_vel_noise_std])
    x_init = x_true[0] + rng.normal(0.0, 1.0, 2) * init_std
    P_init = np.diag(init_std ** 2)
    R_pos, R_zupt = args.pos_noise_std ** 2, args.zupt_noise_std ** 2

    runners = {"never": run_ekf_never_zupt, "always": run_ekf_always_zupt,
               "phase_conditional": run_ekf_phase_conditional_zupt}
    est, sig = {}, {}
    for name, fn in runners.items():
        x_list, P_list = fn(x_init, P_init, z, args.dt, n_steps, args.process_noise_std, R_pos, R_zupt, is_anchor)
        est[name] = x_list
        sig[name] = np.sqrt(np.array([P[1, 1] for P in P_list]))

    print(f"Running Monte Carlo ({args.n_mc_trials} trials)...")
    nees_arrs = run_monte_carlo_consistency(
        x_true, args.dt, args.process_noise_std, R_pos, R_zupt, is_anchor,
        args.init_pos_noise_std, args.init_vel_noise_std, args.n_mc_trials, rng)
    nees = dict(zip(runners, nees_arrs))
    lo, hi = averaged_nees_bounds(2, args.n_mc_trials)

    print("\nRMS errors (single run) / Monte Carlo NEES (overall | anchor | cruise):")
    for name in runners:
        pe, _ = state_errors(x_true, est[name])
        n = nees[name]
        print(f"  {name:<18s} RMS pos={np.sqrt(np.mean(pe ** 2)):.4f} m | NEES {n.mean():7.2f} | "
              f"{n[is_anchor].mean():7.2f} | {n[is_cruise].mean():7.2f}")
    print(f"  95% bounds for the {args.n_mc_trials}-trial average: [{lo:.2f}, {hi:.2f}]")

    t = np.arange(n_steps + 1) * args.dt
    colors = {"never": "tab:gray", "always": "tab:red", "phase_conditional": "tab:blue"}

    fig = plt.figure(figsize=(10, 9.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.55, 1.0, 1.0], hspace=0.22)
    ax_t = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1])
    ax_n = fig.add_subplot(gs[2], sharex=ax_v)
    plt.setp(ax_v.get_xticklabels(), visible=False)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.10)

    # track panel: the point mass crawling along its 1D track, and where each filter puts it
    p_lo, p_hi = x_true[:, 0].min(), x_true[:, 0].max()
    pad = 0.06 * (p_hi - p_lo)
    ax_t.set_xlim(p_lo - pad, p_hi + pad)
    ax_t.set_ylim(-1.75, 1.05)
    ax_t.set_yticks([])
    for side in ("left", "right", "top"):
        ax_t.spines[side].set_visible(False)
    ax_t.set_xlabel("Position along the track (m)", fontsize=9, labelpad=1)
    ax_t.axhline(0.0, color="saddlebrown", lw=3)
    body_w = 0.035 * (p_hi - p_lo + 2 * pad)
    body = Rectangle((x_true[0, 0] - body_w / 2, 0.02), body_w, 0.55, color="gold", ec="black", lw=1, zorder=4)
    ax_t.add_patch(body)
    phase_text = ax_t.text(0.99, 0.97, "", transform=ax_t.transAxes, ha="right", va="top", fontsize=10,
                           fontweight="bold")
    footprints, = ax_t.plot([], [], "|", color="0.45", markersize=10, mew=2, zorder=3)
    rows = {"never": -0.55, "always": -0.95, "phase_conditional": -1.35}
    est_marks = {}
    for n, y in rows.items():
        ax_t.axhline(y, color=colors[n], lw=0.5, alpha=0.3)
        est_marks[n], = ax_t.plot([], [], "v", color=colors[n], markersize=8, zorder=4, label=f"{n} estimate")
    ax_t.legend(loc="upper left", fontsize=7.5, ncol=3, frameon=False, borderaxespad=0.1)
    truth_tick, = ax_t.plot([], [], color="green", lw=1, ls=":", zorder=2)
    anchor_starts = [k for k in range(n_steps + 1) if is_anchor[k] and (k == 0 or not is_anchor[k - 1])]

    # anchor shading on both panels
    edges = np.diff(np.concatenate([[False], is_anchor, [False]]).astype(int))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    for ax in (ax_v, ax_n):
        for s, e in zip(starts, ends):
            ax.axvspan(t[s] - args.dt / 2, t[e - 1] + args.dt / 2, color="gold", alpha=0.18, lw=0)
        ax.set_xlim(0.0, t[-1])

    # velocity panel
    v_lim = 1.1 * max(np.max(np.abs(est[n][:, 1]) + 2 * sig[n]) for n in ("never", "phase_conditional"))
    v_lim = max(v_lim, 0.20)
    ax_v.set_ylim(-0.09, v_lim)
    ax_v.set_ylabel("Velocity (m/s)")
    ax_t.set_title("Inchworm: a point mass crawling by anchor/extend steps, tracked with three ZUPT policies", fontsize=11)
    ax_v.text(0.01, 0.95, "shaded = anchor phase", transform=ax_v.transAxes, fontsize=8, va="top")
    ax_v.plot(t, x_true[:, 1], color="green", lw=1.5, ls="--", alpha=0.35)
    truth_line, = ax_v.plot([], [], color="green", lw=2, ls="--", label="Ground truth")
    lines = {n: ax_v.plot([], [], color=colors[n], lw=1.8, label=f"{n} (±2σ band)")[0] for n in runners}
    bands = {}

    # NEES panel
    ax_n.set_yscale("log")
    ax_n.set_ylim(0.8, 1.0e4)
    ax_n.set_ylabel(f"Monte Carlo avg. NEES ({args.n_mc_trials} trials)")
    ax_n.set_xlabel("Time (s)")
    ax_n.axhline(2.0, color="black", lw=1, label="Expected NEES (2 DoF)")
    ax_n.axhspan(lo, hi, color="black", alpha=0.10, lw=0, label="95% band for the average")
    nlines = {n: ax_n.plot([], [], color=colors[n], lw=2, label=n)[0] for n in runners}

    ax_v.legend(loc="upper right", fontsize=8, ncol=2)
    ax_n.legend(loc="upper left", fontsize=8, ncol=3)

    fig.text(0.5, 0.015,
             "always: ZUPT during motion collapses σ_v and its NEES grows cycle after cycle. phase_conditional tightens σ_v only on anchor ticks,\n"
             "which makes the unmodeled ramp-up hurt more than never at these defaults (true ramp accel 0.5 vs assumed 0.15 m/s²).",
             ha="center", fontsize=8.5)

    def update(k):
        sl = slice(0, k + 1)
        p = x_true[k, 0]
        body.set_x(p - body_w / 2)
        body.set_color("gold" if is_anchor[k] else "tab:green")
        body.set_edgecolor("black")
        phase_text.set_text("ANCHORED (v = 0)" if is_anchor[k] else "EXTENDING")
        phase_text.set_color("darkgoldenrod" if is_anchor[k] else "darkgreen")
        steps = [x_true[a, 0] for a in anchor_starts if a <= k]
        footprints.set_data(steps, [0.0] * len(steps))
        truth_tick.set_data([p, p], [-1.55, 0.0])
        for n in runners:
            est_marks[n].set_data([est[n][k, 0]], [rows[n]])
        truth_line.set_data(t[sl], x_true[sl, 1])
        for n in runners:
            lines[n].set_data(t[sl], est[n][sl, 1])
            nlines[n].set_data(t[sl], nees[n][sl])
            if n in bands:
                bands[n].remove()
            bands[n] = ax_v.fill_between(t[sl], (est[n][:, 1] - 2 * sig[n])[sl], (est[n][:, 1] + 2 * sig[n])[sl],
                                         color=colors[n], alpha=0.18, lw=0)
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_steps + 1, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {n_steps + 1} frames to {args.out}")


if __name__ == "__main__":
    main()
