'''
Animated version of inchworm_slip_gate_ekf.py: a 1D inchworm crawls ten
anchor/extend cycles, some anchor phases contain a backward slip pulse, and
five EKFs track it from noisy position measurements plus a pad displacement
sensor. They share the predict step and the position update and differ only
in the extra velocity information: never, schedule (ZUPT on the known anchor
ticks), sensor_gate (ZUPT when the pad reading is quiet), schedule_and_sensor
(both must agree) and sensor_fused (the reading d / dt as a velocity
measurement).

Defaults are the matched process noise (0.5 m/s^2), as in the script's own
defaults, the regime where the slip effect is not masked by an under-sized
process model.

Four panels sharing the time axis, drawn progressively (gold = slip pulses):
  1. velocity: truth and four estimates;
  2. the pad sensor reading with the quiet band |d| <= threshold;
  3. when each gated policy applies ZUPT (a bar on every tick it runs);
  4. consistency: Monte Carlo averaged NEES (log scale) against the expected
     2 (2 DoF) and the 95% band for the average.

Message (doc docs/filtering/inchworm_zupt_ekf.md, anchor-slip section): the
schedule applies "v = 0" through every slip pulse, so its NEES jumps by more
than an order of magnitude; the sensor-based policies stop applying it as soon
as the reading leaves the quiet band. sensor_gate also applies it at both ends
of each extend phase, where the ramp speed is still below the threshold, which
schedule_and_sensor avoids.

The per-step estimates are the simulation's own run_ekf outputs (same rng
order as inchworm_slip_gate_ekf.main); the Monte Carlo curve is its
run_monte_carlo_consistency. The default Monte Carlo (500 trials) takes about
a minute.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from state_estimation_simulations.use_numpy.inchworm_slip_gate_ekf import (GATING_MODES, STATE_DIM, generate_ground_truth_and_data, run_ekf,
                                    run_monte_carlo_consistency)
from state_estimation_simulations.utils import averaged_nees_bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation length in seconds")
    parser.add_argument("--dt", type=float, default=0.05, help="Filter/measurement step interval in seconds")
    parser.add_argument("--t-anchor", type=float, default=1.0, help="Anchor/dwell phase duration per cycle (s)")
    parser.add_argument("--t-extend", type=float, default=1.0, help="Total extend phase duration per cycle (s)")
    parser.add_argument("--t-ramp", type=float, default=0.2, help="Ramp duration (s)")
    parser.add_argument("--v-extend", type=float, default=0.1, help="Commanded cruise speed (m/s)")
    parser.add_argument("--p-slip", type=float, default=0.5, help="Probability that an anchor phase slips")
    parser.add_argument("--v-slip", type=float, default=0.08, help="Peak backward slip speed (m/s)")
    parser.add_argument("--t-slip", type=float, default=0.6, help="Slip pulse duration (s)")
    parser.add_argument("--pos-noise-std", type=float, default=0.02, help="Position noise std-dev (m)")
    parser.add_argument("--sensor-noise-std", type=float, default=0.0005, help="Pad sensor noise std-dev (m)")
    parser.add_argument("--gate-sigmas", type=float, default=3.0, help="Gate threshold in sensor noise std-devs")
    parser.add_argument("--process-noise-std", type=float, default=0.5, help="Assumed acceleration std-dev (m/s^2)")
    parser.add_argument("--zupt-noise-std", type=float, default=0.01, help="ZUPT noise std-dev (m/s)")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.05, help="Initial position error std-dev (m)")
    parser.add_argument("--init-vel-noise-std", type=float, default=0.05, help="Initial velocity error std-dev (m/s)")
    parser.add_argument("--n-mc-trials", type=int, default=500, help="Number of Monte Carlo consistency trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--stride", type=int, default=2, help="Filter ticks per GIF frame")
    parser.add_argument("--fps", type=int, default=12, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=70, help="GIF resolution")
    parser.add_argument("--out", type=str, default="inchworm_slip_gate_ekf.gif", help="Output GIF path")
    args = parser.parse_args()

    # Same rng consumption order as inchworm_slip_gate_ekf.main
    rng = np.random.default_rng(args.seed)
    x_true, z, d, is_anchor, is_cruise, is_slip, is_clean = generate_ground_truth_and_data(
        args.duration, args.dt, args.t_anchor, args.t_extend, args.t_ramp, args.v_extend,
        args.pos_noise_std, args.sensor_noise_std, args.p_slip, args.v_slip, args.t_slip, rng)
    n_steps = len(x_true) - 1
    init_std = np.array([args.init_pos_noise_std, args.init_vel_noise_std])
    x_init = x_true[0] + rng.normal(0.0, 1.0, 2) * init_std
    P_init = np.diag(init_std ** 2)
    R_pos, R_zupt, R_vel = args.pos_noise_std ** 2, args.zupt_noise_std ** 2, (args.sensor_noise_std / args.dt) ** 2
    d_thr = args.gate_sigmas * args.sensor_noise_std

    est = {m: run_ekf(x_init, P_init, z, d, args.dt, n_steps, args.process_noise_std, R_pos, R_zupt, R_vel,
                      d_thr, is_anchor, m)[0] for m in GATING_MODES}

    print(f"Running Monte Carlo ({args.n_mc_trials} trials)...")
    nees = run_monte_carlo_consistency(x_true, args.dt, args.process_noise_std, R_pos, R_zupt, R_vel, d_thr,
                                       is_anchor, args.init_pos_noise_std, args.init_vel_noise_std,
                                       args.sensor_noise_std, args.n_mc_trials, rng)
    lo, hi = averaged_nees_bounds(STATE_DIM, args.n_mc_trials)
    print("\nMonte Carlo NEES (clean anchor | slip | cruise):")
    window_mean = lambda n, w: n[w].mean() if w.any() else float("nan")  # noqa: E731
    for m in GATING_MODES:
        print(f"  {m:<20s} {window_mean(nees[m], is_clean):8.2f} | {window_mean(nees[m], is_slip):8.2f} | "
              f"{window_mean(nees[m], is_cruise):8.2f}")
    print(f"  95% bounds for the {args.n_mc_trials}-trial average: [{lo:.2f}, {hi:.2f}]")

    quiet = np.abs(d) <= d_thr
    applied = {"schedule": is_anchor, "sensor_gate": quiet, "schedule_and_sensor": is_anchor & quiet}
    applied = {m: np.concatenate([[False], a[1:]]) for m, a in applied.items()}  # run_ekf updates from tick 1

    t = np.arange(n_steps + 1) * args.dt
    colors = {"never": "tab:gray", "schedule": "tab:blue", "sensor_gate": "tab:red",
              "schedule_and_sensor": "tab:purple", "sensor_fused": "tab:green"}
    short = {"never": "never", "schedule": "schedule", "sensor_gate": "sensor_gate",
             "schedule_and_sensor": "schedule_and_sensor", "sensor_fused": "sensor_fused (d/dt)"}

    fig, (ax_v, ax_d, ax_z, ax_n) = plt.subplots(4, 1, figsize=(10, 9.6), sharex=True,
                                                  gridspec_kw=dict(height_ratios=[1.3, 1.0, 0.8, 1.3], hspace=0.12))
    fig.subplots_adjust(left=0.17, right=0.98, top=0.90, bottom=0.10)
    fig.suptitle("1D inchworm with anchor slip: ZUPT gated by schedule vs. by a pad sensor "
                 "(matched process noise)", fontsize=12)
    status = fig.text(0.5, 0.925, "", ha="center", fontsize=12, fontweight="bold")

    edges = np.diff(np.concatenate([[False], is_slip, [False]]).astype(int))
    for ax in (ax_v, ax_d, ax_z, ax_n):
        for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            ax.axvspan(t[a], t[b - 1] + args.dt, color="gold", alpha=0.25, lw=0)
        ax.set_xlim(0.0, t[-1])

    ax_v.plot(t, x_true[:, 1], color="black", lw=1.2, ls="--", alpha=0.3)
    truth_line, = ax_v.plot([], [], color="black", lw=2, ls="--", label="Ground truth")
    vlines = {m: ax_v.plot([], [], color=colors[m], lw=1.3, label=short[m])[0]
              for m in ("schedule", "sensor_gate", "schedule_and_sensor", "sensor_fused")}
    ax_v.set_ylim(-0.12, 0.2)
    ax_v.set_ylabel("Velocity (m/s)")
    ax_v.legend(loc="upper right", fontsize=7.5, ncol=3)

    ax_d.axhspan(-d_thr * 1e3, d_thr * 1e3, color="tab:red", alpha=0.14, label="quiet band |d| <= threshold")
    ax_d.set_ylim(-6.5, 8.0)
    ax_d.set_ylabel("Pad reading d (mm/tick)")
    d_line, = ax_d.plot([], [], color="tab:orange", lw=1.2, label="pad sensor reading d")
    ax_d.legend(loc="upper right", fontsize=7.5)

    rows = {"schedule": 2, "sensor_gate": 1, "schedule_and_sensor": 0}
    ax_z.set_ylim(-0.6, 2.6)
    ax_z.set_yticks(list(rows.values()))
    ax_z.set_yticklabels(list(rows.keys()), fontsize=8)
    ax_z.set_ylabel("ZUPT applied")
    z_marks = {m: ax_z.plot([], [], "|", color=colors[m], ms=11, mew=2)[0] for m in rows}

    ax_n.set_yscale("log")
    ax_n.set_ylim(0.5, 150)
    ax_n.set_ylabel(f"Monte Carlo avg. NEES ({args.n_mc_trials} trials)")
    ax_n.set_xlabel("Time (s)")
    ax_n.axhline(STATE_DIM, color="black", lw=1, label=f"Expected NEES ({STATE_DIM} DoF)")
    ax_n.axhspan(lo, hi, color="black", alpha=0.10, lw=0, label="95% band for the average")
    nlines = {m: ax_n.plot([], [], color=colors[m], lw=1.8, label=short[m])[0] for m in GATING_MODES}
    ax_n.legend(loc="upper right", fontsize=7.5, ncol=2)

    fig.text(0.5, 0.012,
             "schedule keeps applying v = 0 through every slip pulse and its NEES jumps; the sensor policies stop as soon as the reading leaves the band.\n"
             "sensor_gate also applies it at both ends of each extend phase (ramp speed still below the threshold); schedule_and_sensor does not.",
             ha="center", fontsize=8.2)

    frames = list(range(0, n_steps + 1, args.stride))

    def update(k):
        sl = slice(0, k + 1)
        truth_line.set_data(t[sl], x_true[sl, 1])
        for m, ln in vlines.items():
            ln.set_data(t[sl], est[m][sl, 1])
        d_line.set_data(t[sl], d[sl] * 1e3)
        for m, mk in z_marks.items():
            idx = np.flatnonzero(applied[m][sl])
            mk.set_data(t[idx], np.full(idx.size, rows[m]))
        for m, ln in nlines.items():
            ln.set_data(t[sl], nees[m][sl])
        if is_slip[k]:
            status.set_text("SLIPPING (true v < 0)")
            status.set_color("darkgoldenrod")
        elif is_anchor[k]:
            status.set_text("ANCHORED (true v = 0)")
            status.set_color("tab:blue")
        else:
            status.set_text("EXTENDING")
            status.set_color("darkgreen")
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
