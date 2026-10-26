'''
Animated version of inchworm_zupt_zaru_ekf.py: a planar inchworm crawls five
anchor/extend cycles, turning at omega_extend while it extends, and four EKFs
track it from noisy position measurements only. They share the predict step
and the position update and differ only in when the zero-velocity (ZUPT,
v = 0) and zero-angular-rate (ZARU, omega = 0) pseudo-measurements also run:
never, always, phase_conditional (both on the known anchor ticks), and the
ablation phase_conditional_zupt_only (ZUPT on anchor ticks, no ZARU).

Unlike animate_inchworm_zupt_ekf.py, the defaults here are NOT the script's
defaults: the assumed accelerations are the matched ones (0.5 m/s^2 and
1.5 rad/s^2, equal to the true ramp accelerations), the regime where ZARU's
effect on the heading is not masked by an under-sized process model. Pass
--process-noise-std 0.15 --process-noise-ang-std 0.45 for the script defaults.

Three panels, drawn progressively:
  1. top-down view: the true path (body arrow gold when anchored, green when
     extending) and each filter's position and heading arrow;
  2. heading: truth and each estimate with a +-2 sigma_theta band (shaded =
     anchor phases);
  3. consistency: Monte Carlo averaged NEES (log scale) against the expected
     5 (5 DoF) and the 95% band for the average.

Message (doc docs/filtering/inchworm_zupt_ekf.md, side-extension section):
with position-only measurements the heading is unobserved at rest, so without
ZARU sigma_theta grows through every dwell; with ZARU it shrinks. `always`
pins v and omega to zero, so its heading is never informed and its NEES grows
cycle after cycle.

The per-step estimates are the simulation's own run_ekf outputs (same rng
order as inchworm_zupt_zaru_ekf.main); the Monte Carlo curve is its
run_monte_carlo_consistency. The default Monte Carlo (500 trials) takes ~50 s.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from state_estimation_simulations.use_numpy.inchworm_zupt_zaru_ekf import (GATING_MODES, STATE_DIM, generate_ground_truth_and_data, run_ekf,
                                    run_monte_carlo_consistency, state_errors)
from state_estimation_simulations.utils import averaged_nees_bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation length in seconds")
    parser.add_argument("--dt", type=float, default=0.05, help="Filter/measurement step interval in seconds")
    parser.add_argument("--t-anchor", type=float, default=1.0, help="Anchor/dwell phase duration per cycle (s)")
    parser.add_argument("--t-extend", type=float, default=1.0, help="Total extend phase duration per cycle (s)")
    parser.add_argument("--t-ramp", type=float, default=0.2, help="Ramp-up/ramp-down duration (s)")
    parser.add_argument("--v-extend", type=float, default=0.1, help="Commanded cruise speed (m/s)")
    parser.add_argument("--omega-extend", type=float, default=0.3, help="Commanded cruise turn rate (rad/s)")
    parser.add_argument("--pos-noise-std", type=float, default=0.02, help="Position noise std-dev per axis (m)")
    parser.add_argument("--process-noise-std", type=float, default=0.5,
                        help="Assumed linear-acceleration std-dev (m/s^2); matched default, script default 0.15")
    parser.add_argument("--process-noise-ang-std", type=float, default=1.5,
                        help="Assumed angular-acceleration std-dev (rad/s^2); matched default, script default 0.45")
    parser.add_argument("--zupt-noise-std", type=float, default=0.01, help="ZUPT noise std-dev (m/s)")
    parser.add_argument("--zaru-noise-std", type=float, default=0.02, help="ZARU noise std-dev (rad/s)")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.05, help="Initial position error std-dev (m)")
    parser.add_argument("--init-heading-noise-std", type=float, default=0.1, help="Initial heading error std-dev (rad)")
    parser.add_argument("--init-vel-noise-std", type=float, default=0.05, help="Initial speed error std-dev (m/s)")
    parser.add_argument("--init-omega-noise-std", type=float, default=0.05, help="Initial turn-rate error std-dev (rad/s)")
    parser.add_argument("--n-mc-trials", type=int, default=500, help="Number of Monte Carlo consistency trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--fps", type=int, default=20, help="GIF frames per second (one frame per filter tick)")
    parser.add_argument("--dpi", type=int, default=70, help="GIF resolution")
    parser.add_argument("--out", type=str, default="inchworm_zupt_zaru_ekf.gif", help="Output GIF path")
    args = parser.parse_args()

    # Same rng consumption order as inchworm_zupt_zaru_ekf.main
    rng = np.random.default_rng(args.seed)
    x_true, z, is_anchor, is_cruise = generate_ground_truth_and_data(
        args.duration, args.dt, args.t_anchor, args.t_extend, args.t_ramp, args.v_extend,
        args.omega_extend, args.pos_noise_std, rng)
    n_steps = len(x_true) - 1
    init_std = np.array([args.init_pos_noise_std, args.init_pos_noise_std, args.init_heading_noise_std,
                         args.init_vel_noise_std, args.init_omega_noise_std])
    x_init = x_true[0] + rng.normal(0.0, 1.0, STATE_DIM) * init_std
    P_init = np.diag(init_std ** 2)
    R = (args.pos_noise_std ** 2, args.zupt_noise_std ** 2, args.zaru_noise_std ** 2)

    est, sig = {}, {}
    for name in GATING_MODES:
        x_list, P_list = run_ekf(x_init, P_init, z, args.dt, n_steps, args.process_noise_std,
                                 args.process_noise_ang_std, *R, is_anchor, name)
        est[name] = x_list
        sig[name] = np.sqrt(np.array([P[2, 2] for P in P_list]))

    print(f"Running Monte Carlo ({args.n_mc_trials} trials)...")
    nees = run_monte_carlo_consistency(x_true, args.dt, args.process_noise_std, args.process_noise_ang_std, *R,
                                       is_anchor, init_std, args.n_mc_trials, rng)
    lo, hi = averaged_nees_bounds(STATE_DIM, args.n_mc_trials)

    print("\nRMS heading error (single run) / Monte Carlo NEES (overall | anchor | cruise):")
    for name in GATING_MODES:
        _, he, _, _ = state_errors(x_true, est[name])
        n = nees[name]
        print(f"  {name:<28s} RMS heading={np.sqrt(np.mean(he ** 2)):.4f} rad | NEES {n.mean():8.2f} | "
              f"{n[is_anchor].mean():8.2f} | {n[is_cruise].mean():8.2f}")
    print(f"  95% bounds for the {args.n_mc_trials}-trial average: [{lo:.2f}, {hi:.2f}]")

    t = np.arange(n_steps + 1) * args.dt
    colors = {"never": "tab:gray", "always": "tab:red", "phase_conditional": "tab:blue",
              "phase_conditional_zupt_only": "tab:orange"}
    short = {"never": "never", "always": "always", "phase_conditional": "phase_conditional (ZUPT+ZARU)",
             "phase_conditional_zupt_only": "zupt_only (no ZARU)"}

    fig = plt.figure(figsize=(11, 8.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], hspace=0.25, wspace=0.22)
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    ax_n = fig.add_subplot(gs[1, 1], sharex=ax_h)
    plt.setp(ax_h.get_xticklabels(), visible=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.11)
    fig.suptitle("Planar inchworm: ZUPT + ZARU gated by the anchor schedule (matched process noise)", fontsize=12)

    # top-down panel
    pts = np.vstack([x_true[:, :2]] + [est[n][:, :2] for n in GATING_MODES if n != "always"])
    c, span = 0.5 * (pts.min(0) + pts.max(0)), 0.6 * np.max(pts.max(0) - pts.min(0)) + 0.05
    ax_xy.set_xlim(c[0] - span, c[0] + span)
    ax_xy.set_ylim(c[1] - span, c[1] + span)
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.plot(x_true[:, 0], x_true[:, 1], color="green", lw=1, ls="--", alpha=0.35)
    truth_path, = ax_xy.plot([], [], color="green", lw=2, label="true path")
    arrow_len = 0.22 * span
    arrows = {}
    phase_text = ax_xy.text(0.02, 0.98, "", transform=ax_xy.transAxes, ha="left", va="top", fontsize=10,
                            fontweight="bold")
    est_paths = {n: ax_xy.plot([], [], color=colors[n], lw=1.0, alpha=0.8, label=short[n])[0] for n in GATING_MODES}
    ax_xy.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax_xy.set_title("Top view: position and heading arrow", fontsize=10)

    edges = np.diff(np.concatenate([[False], is_anchor, [False]]).astype(int))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    for ax in (ax_h, ax_n):
        for s, e in zip(starts, ends):
            ax.axvspan(t[s] - args.dt / 2, t[e - 1] + args.dt / 2, color="gold", alpha=0.18, lw=0)
        ax.set_xlim(0.0, t[-1])

    # heading panel
    h_lo = min(x_true[:, 2].min(), min((est[n][:, 2] - 2 * sig[n]).min() for n in GATING_MODES)) - 0.1
    h_hi = max(x_true[:, 2].max(), max((est[n][:, 2] + 2 * sig[n]).max() for n in GATING_MODES)) + 0.1
    ax_h.set_ylim(max(h_lo, -1.5), min(h_hi, 3.0))
    ax_h.set_ylabel("Heading θ (rad)")
    ax_h.set_title("Heading with ±2σ_θ bands (shaded = anchor phase)", fontsize=10)
    ax_h.plot(t, x_true[:, 2], color="green", lw=1.5, ls="--", alpha=0.35)
    truth_line, = ax_h.plot([], [], color="green", lw=2, ls="--", label="Ground truth")
    lines = {n: ax_h.plot([], [], color=colors[n], lw=1.6)[0] for n in GATING_MODES}
    bands = {}

    # NEES panel
    ax_n.set_yscale("log")
    ax_n.set_ylim(1.0, 2.0e4)
    ax_n.set_ylabel(f"Monte Carlo avg. NEES ({args.n_mc_trials} trials)")
    ax_n.set_xlabel("Time (s)")
    ax_n.axhline(STATE_DIM, color="black", lw=1, label=f"Expected NEES ({STATE_DIM} DoF)")
    ax_n.axhspan(lo, hi, color="black", alpha=0.10, lw=0, label="95% band for the average")
    nlines = {n: ax_n.plot([], [], color=colors[n], lw=2)[0] for n in GATING_MODES}
    ax_n.legend(loc="upper left", fontsize=8)

    fig.text(0.5, 0.015,
             "Position-only measurements: heading is unobserved at rest. Without ZARU, σ_θ grows through each dwell; with ZARU it shrinks.\n"
             "always pins v and ω to zero, so its heading is never informed and its NEES grows cycle after cycle.",
             ha="center", fontsize=8.5)

    def update(k):
        sl = slice(0, k + 1)
        truth_path.set_data(x_true[sl, 0], x_true[sl, 1])
        for a in arrows.values():
            a.remove()
        arrows.clear()
        th = x_true[k, 2]
        arrows["truth"] = ax_xy.arrow(x_true[k, 0], x_true[k, 1], arrow_len * np.cos(th), arrow_len * np.sin(th),
                                      width=0.014 * span, color="gold" if is_anchor[k] else "tab:green",
                                      ec="black", lw=0.6, zorder=5, length_includes_head=True)
        for n in GATING_MODES:
            xe = est[n][k]
            est_paths[n].set_data(est[n][sl, 0], est[n][sl, 1])
            arrows[n] = ax_xy.arrow(xe[0], xe[1], 0.8 * arrow_len * np.cos(xe[2]), 0.8 * arrow_len * np.sin(xe[2]),
                                    width=0.008 * span, color=colors[n], zorder=4, length_includes_head=True)
        phase_text.set_text("ANCHORED (v = ω = 0)" if is_anchor[k] else "EXTENDING (turning)")
        phase_text.set_color("darkgoldenrod" if is_anchor[k] else "darkgreen")
        truth_line.set_data(t[sl], x_true[sl, 2])
        for n in GATING_MODES:
            lines[n].set_data(t[sl], est[n][sl, 2])
            nlines[n].set_data(t[sl], nees[n][sl])
            if n in bands:
                bands[n].remove()
            bands[n] = ax_h.fill_between(t[sl], (est[n][:, 2] - 2 * sig[n])[sl], (est[n][:, 2] + 2 * sig[n])[sl],
                                         color=colors[n], alpha=0.15, lw=0)
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_steps + 1, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {n_steps + 1} frames to {args.out}")


if __name__ == "__main__":
    main()
