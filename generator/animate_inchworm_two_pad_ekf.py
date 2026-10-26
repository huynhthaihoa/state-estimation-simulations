'''
Animated version of inchworm_two_pad_ekf.py: a planar inchworm with a back and
a front pad crawls ten extend/contract cycles, steering while it extends. Its
anchored pad creeps, and in some phases slips backward in a short pulse. Four
EKFs estimate both pads from a gyro, a kinematic closure and the displacement
reading of the anchored pad's optical sensor. They share the predict and update
code and differ only in how they treat the anchored pad: never (loose random
walk), schedule ("anchored, so still"), schedule_and_sensor (still, unless the
reading |d| exceeds the veto threshold) and pad_fused (move by the reading).

Three panels, drawn progressively (gold = slip pulses):
  1. top view (left): the true robot as two pads joined by a body line, the
     anchored pad filled (gold; red while a slip pulse is active) and the
     moving pad hollow, a fading trail of the true body midpoint, and each
     policy's estimated midpoint;
  2. the anchored-pad sensor reading |d| with the veto threshold line;
  3. consistency: Monte Carlo averaged NEES (log scale) against the expected
     5 (5 DoF) and the 95% band for the average.

Message: the schedule is blind to slip pulses, so its NEES jumps by orders of
magnitude whenever the pad slips; the veto (schedule_and_sensor) reacts when
|d| crosses the threshold, and pad_fused follows the pad's actual motion.

The per-step estimates are the simulation's own run_ekf outputs (same rng order
as inchworm_two_pad_ekf.main); the Monte Carlo loop mirrors that script's. The
default Monte Carlo (500 trials) takes about a minute.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D

from state_estimation_simulations.use_numpy.inchworm_two_pad_ekf import (B, F, POLICIES, STATE_DIM, TH, draw_slip_episodes, run_ekf, simulate,
                                  state_error, window_flags)
from state_estimation_simulations.utils import averaged_nees_bounds
from state_estimation_simulations.utils import nees as nees_of


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation length (s); 20 s = 10 cycles")
    parser.add_argument("--dt", type=float, default=0.05, help="Filter/sensor step interval (s)")
    parser.add_argument("--t-phase", type=float, default=1.0, help="Duration of each extend or contract phase (s)")
    parser.add_argument("--t-ramp", type=float, default=0.2, help="Ramp at each end of a stroke and of a slip pulse (s)")
    parser.add_argument("--v-stroke", type=float, default=0.1, help="Peak stroke rate dl/dt (m/s)")
    parser.add_argument("--l-min", type=float, default=0.15, help="Body length at the start of extend (m)")
    parser.add_argument("--steer", type=float, default=0.3, help="Steering angle during extend (rad)")
    parser.add_argument("--p-slip", type=float, default=0.5, help="Probability that a phase contains a slip pulse")
    parser.add_argument("--v-slip", type=float, default=0.08, help="Peak backward slip speed (m/s)")
    parser.add_argument("--t-slip", type=float, default=0.6, help="Slip pulse duration including both ramps (s)")
    parser.add_argument("--micro-slip-std", type=float, default=0.01, help="White creep of the anchored pad (m/s per axis)")
    parser.add_argument("--gyro-noise-std", type=float, default=0.02, help="Gyro noise std-dev (rad/s)")
    parser.add_argument("--sensor-noise-std", type=float, default=0.0005, help="Pad sensor noise per reading and axis (m)")
    parser.add_argument("--closure-noise-std", type=float, default=0.001, help="Closure noise along the body axis (m)")
    parser.add_argument("--closure-lat-noise-std", type=float, default=0.001, help="Closure noise across the body axis (m)")
    parser.add_argument("--gate-sigmas", type=float, default=3.0, help="Veto threshold in std-devs of a still pad's reading")
    parser.add_argument("--free-noise-std", type=float, default=0.1, help="Anchored-pad random walk with no anchor knowledge (m/s)")
    parser.add_argument("--move-noise-std", type=float, default=0.02, help="Moving-pad random walk per tick (m)")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.005, help="Initial pad position error (m)")
    parser.add_argument("--init-heading-noise-std", type=float, default=0.02, help="Initial heading error (rad)")
    parser.add_argument("--n-mc-trials", type=int, default=500, help="Number of Monte Carlo consistency trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--stride", type=int, default=2, help="Filter ticks per GIF frame")
    parser.add_argument("--fps", type=int, default=10, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.add_argument("--out", type=str, default="inchworm_two_pad_ekf.gif", help="Output GIF path")
    args = parser.parse_args()
    if not 2.0 * args.t_ramp <= args.t_slip <= args.t_phase:
        raise SystemExit("need 2*t-ramp <= t-slip <= t-phase")

    # Same rng consumption order as inchworm_two_pad_ekf.main
    rng = np.random.default_rng(args.seed)
    n_steps = int(round(args.duration / args.dt))
    n_phases = int(np.ceil(args.duration / args.t_phase))
    episodes = draw_slip_episodes(n_phases, args.t_phase, args.t_slip, args.p_slip, rng)
    is_slip, is_clean = window_flags(n_steps, args.dt, args.t_phase, episodes)

    still_std = np.hypot(args.sensor_noise_std, args.micro_slip_std * args.dt)
    noise = dict(micro_slip=args.micro_slip_std, free=args.free_noise_std, sensor=args.sensor_noise_std,
                 move=args.move_noise_std, gyro=args.gyro_noise_std, closure=args.closure_noise_std,
                 closure_lat=args.closure_lat_noise_std, d_thr=args.gate_sigmas * still_std)
    init_std = np.array([args.init_pos_noise_std] * 4 + [args.init_heading_noise_std])
    P_init = np.diag(init_std ** 2)
    sim_args = (n_steps, args.dt, args.t_phase, args.t_ramp, args.v_stroke, args.l_min, args.steer, episodes,
                args.v_slip, args.micro_slip_std, args.gyro_noise_std, args.sensor_noise_std,
                args.closure_noise_std, args.closure_lat_noise_std)

    def one_trial(trial_rng):
        x_true, gyro, d, z, back = simulate(*sim_args, trial_rng)
        x_init = x_true[0] + trial_rng.normal(0.0, 1.0, STATE_DIM) * init_std
        return x_true, gyro, d, z, back, x_init

    x_true, gyro, d, z, back, x_init = one_trial(rng)
    est = {p: run_ekf(x_init, P_init, gyro, d, z, back, args.dt, p, noise)[0] for p in POLICIES}

    print(f"Running Monte Carlo ({args.n_mc_trials} trials)...")
    sums = {p: np.zeros(n_steps + 1) for p in POLICIES}
    for _ in range(args.n_mc_trials):
        xt, g, dd, zz, bb, xi = one_trial(rng)
        for p in POLICIES:
            xe, Pe = run_ekf(xi, P_init, g, dd, zz, bb, args.dt, p, noise)
            err = state_error(xt, xe)
            for k in range(n_steps + 1):
                sums[p][k] += nees_of(err[k], Pe[k])
    nees = {p: s / args.n_mc_trials for p, s in sums.items()}
    lo, hi = averaged_nees_bounds(STATE_DIM, args.n_mc_trials)

    print("\nMonte Carlo NEES (overall | clean | slip):")
    for p in POLICIES:
        n = nees[p]
        print(f"  {p:<20s} {n.mean():10.2f} | {n[is_clean].mean():10.2f} | "
              f"{(n[is_slip].mean() if is_slip.any() else float('nan')):10.2f}")
    print(f"  95% bounds for the {args.n_mc_trials}-trial average: [{lo:.2f}, {hi:.2f}]")

    t = np.arange(n_steps + 1) * args.dt
    d_mag = np.linalg.norm(d, axis=1) * 1e3
    d_thr_mm = noise["d_thr"] * 1e3
    mid_true = 0.5 * (x_true[:, B] + x_true[:, F])
    mid_est = {p: 0.5 * (est[p][:, B] + est[p][:, F]) for p in POLICIES}
    # Reading k is the interval ending at tick k + 1; hold it for the tick it starts at
    anchored_back = np.concatenate([back, back[-1:]])
    colors = {"never": "tab:gray", "schedule": "tab:blue", "schedule_and_sensor": "tab:purple", "pad_fused": "tab:green"}

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.12, wspace=0.18)
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_n = fig.add_subplot(gs[1, 1], sharex=ax_d)
    plt.setp(ax_d.get_xticklabels(), visible=False)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.08)
    fig.suptitle("Two-pad inchworm with anchor slip: four anchor policies, one pad sensor", fontsize=13)

    # Top view
    pts = np.vstack([x_true[:, B], x_true[:, F]] + [mid_est[p] for p in ("schedule_and_sensor", "pad_fused")])
    lo_xy, hi_xy = pts.min(0), pts.max(0)
    c, half = 0.5 * (lo_xy + hi_xy), 0.5 * np.max(hi_xy - lo_xy) * 1.12 + 0.02
    ax_xy.set_xlim(c[0] - half, c[0] + half)
    ax_xy.set_ylim(c[1] - half, c[1] + half)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("Top view: true robot and estimated body midpoints", fontsize=11)
    ax_xy.grid(alpha=0.25)
    ax_xy.plot(mid_true[:, 0], mid_true[:, 1], color="black", lw=1, ls="--", alpha=0.15)

    trail_n = 30
    trail = [ax_xy.plot([], [], color="black", lw=3, solid_capstyle="round")[0] for _ in range(trail_n)]
    est_trails = {p: ax_xy.plot([], [], color=colors[p], lw=1.0, alpha=0.5)[0] for p in POLICIES}
    est_dots = {p: ax_xy.plot([], [], "D", color=colors[p], ms=7, mec="white", mew=0.8, zorder=6)[0] for p in POLICIES}
    body, = ax_xy.plot([], [], color="black", lw=4, solid_capstyle="round", zorder=4)
    pad_back, = ax_xy.plot([], [], "o", ms=16, mec="black", mew=1.8, zorder=5)
    pad_front, = ax_xy.plot([], [], "o", ms=16, mec="black", mew=1.8, zorder=5)
    status = ax_xy.text(0.02, 0.98, "", transform=ax_xy.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")

    handles = [Line2D([], [], color="black", lw=2.2, label="true body midpoint (trail)"),
               Line2D([], [], marker="o", ls="", ms=10, mfc="gold", mec="black", label="anchored pad"),
               Line2D([], [], marker="o", ls="", ms=10, mfc="white", mec="black", label="moving pad"),
               Line2D([], [], marker="o", ls="", ms=10, mfc="tab:red", mec="black", label="anchored pad, slipping")]
    handles += [Line2D([], [], marker="D", ls="", ms=7, color=colors[p], label=p) for p in POLICIES]
    ax_xy.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.94), fontsize=8.5, framealpha=0.9, title="estimated midpoint",
                 title_fontsize=8.5)

    # Slip pulses shaded on the time panels
    for ax in (ax_d, ax_n):
        for _, a, b in episodes:
            ax.axvspan(a, b, color="gold", alpha=0.25, lw=0)
        ax.set_xlim(0.0, t[-1])
    cursors = [ax.axvline(0.0, color="black", lw=1.2) for ax in (ax_d, ax_n)]

    # Sensor panel
    ax_d.set_ylim(0.0, 1.45 * d_mag.max())
    ax_d.set_ylabel("Anchored-pad reading |d| (mm/tick)")
    ax_d.set_title("Pad sensor reading (gold = slip pulse)", fontsize=11)
    ax_d.axhline(d_thr_mm, color="tab:red", ls="--", lw=1.3, label=f"veto threshold ({d_thr_mm:.2f} mm)")
    d_line, = ax_d.plot([], [], color="tab:orange", lw=1.4, label="|d|")
    ax_d.legend(loc="upper right", fontsize=9)

    # NEES panel
    ax_n.set_yscale("log")
    all_n = np.concatenate([nees[p] for p in POLICIES])
    ax_n.set_ylim(min(0.5 * lo, 0.8 * all_n.min()), 8.0 * all_n.max())
    ax_n.set_ylabel(f"Monte Carlo avg. NEES ({args.n_mc_trials} trials)")
    ax_n.set_xlabel("Time (s)")
    ax_n.axhline(STATE_DIM, color="black", lw=1, label=f"Expected NEES ({STATE_DIM} DoF)")
    ax_n.axhspan(lo, hi, color="black", alpha=0.10, lw=0, label="95% band for the average")
    nlines = {p: ax_n.plot([], [], color=colors[p], lw=1.8, label=p)[0] for p in POLICIES}
    ax_n.legend(loc="upper left", fontsize=8.5, ncol=2)

    frames = list(range(0, n_steps + 1, args.stride))
    if frames[-1] != n_steps:
        frames.append(n_steps)

    def update(k):
        sl = slice(0, k + 1)
        pb, pf = x_true[k, B], x_true[k, F]
        body.set_data([pb[0], pf[0]], [pb[1], pf[1]])
        slipping = bool(is_slip[k])
        anch_col = "tab:red" if slipping else "gold"
        for pad, p, is_anch in ((pad_back, pb, anchored_back[k]), (pad_front, pf, not anchored_back[k])):
            pad.set_data([p[0]], [p[1]])
            pad.set_markerfacecolor(anch_col if is_anch else "white")
        # Fading trail of the true midpoint
        i0 = max(0, k - trail_n)
        for j, ln in enumerate(trail):
            a, b = i0 + j, i0 + j + 1
            if b <= k:
                ln.set_data(mid_true[a:b + 1, 0], mid_true[a:b + 1, 1])
                ln.set_alpha(0.08 + 0.7 * (j + 1) / trail_n)
            else:
                ln.set_data([], [])
        for p in POLICIES:
            est_trails[p].set_data(mid_est[p][max(0, k - 2 * trail_n):k + 1, 0], mid_est[p][max(0, k - 2 * trail_n):k + 1, 1])
            est_dots[p].set_data([mid_est[p][k, 0]], [mid_est[p][k, 1]])
            nlines[p].set_data(t[sl], nees[p][sl])
        if slipping:
            status.set_text("ANCHORED PAD SLIPPING")
            status.set_color("tab:red")
        elif anchored_back[k]:
            status.set_text("EXTEND (back pad anchored, turning)")
            status.set_color("darkgreen")
        else:
            status.set_text("CONTRACT (front pad anchored)")
            status.set_color("tab:blue")
        kd = min(k, n_steps - 1)
        d_line.set_data(t[1:kd + 2], d_mag[:kd + 1])
        for cur in cursors:
            cur.set_xdata([t[k], t[k]])
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
