'''
Animated version of rts_smoother.py's default run: a 1D point mass crawls in
five anchor (dwell, exactly stationary) / extend (ramp up, cruise, ramp down)
cycles. The same noisy position measurements and ZUPTs are processed twice:
a forward Kalman filter, then the backward Rauch-Tung-Striebel (RTS) pass.

The animation has two acts on one time axis:
  1. FORWARD PASS: the cursor sweeps left to right; the filter's estimate at
     each tick (with its +-2 sigma band) is drawn using only the data up to the
     cursor (the data to its right is greyed out: not measured yet);
  2. BACKWARD PASS: the cursor sweeps right to left; the RTS smoother's
     estimate at each tick (smaller band) appears, now using all the data.

Three panels:
  1. the track: the point mass crawling along its 1D track (gold while anchored,
     green while extending) with its anchor footprints, the true position, and
     the estimate of the pass being drawn (blue triangle = filter, red = smoother);
  2. velocity: truth, filter and smoother estimates with +-2 sigma bands (one run,
     the same seed as the script);
  3. Monte Carlo velocity RMSE per tick (the script's scenario `inchworm`): the
     filter's curve in the forward pass, the smoother's in the backward pass.

The estimates are rts_smoother.py's own run_forward_filter / rts_smoother
outputs (same rng order as its main) and the Monte Carlo curve is its
run_monte_carlo (default 500 trials, about 10 s).

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

from state_estimation_simulations.use_numpy.inchworm_zupt_ekf import generate_ground_truth_and_data
from state_estimation_simulations.use_numpy.rts_smoother import run_forward_filter, rts_smoother, run_monte_carlo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation length in seconds")
    parser.add_argument("--dt", type=float, default=0.05, help="Filter/measurement step interval in seconds")
    parser.add_argument("--t-anchor", type=float, default=1.0, help="Anchor/dwell phase duration per cycle (s)")
    parser.add_argument("--t-extend", type=float, default=1.0, help="Total extend phase duration per cycle (s)")
    parser.add_argument("--t-ramp", type=float, default=0.2, help="Ramp-up/ramp-down duration (s)")
    parser.add_argument("--v-extend", type=float, default=0.1, help="Commanded cruise speed (m/s)")
    parser.add_argument("--pos-noise-std", type=float, default=0.02, help="Position measurement noise std-dev (m)")
    parser.add_argument("--process-noise-std", type=float, default=0.5, help="Assumed acceleration std-dev (m/s^2)")
    parser.add_argument("--zupt-noise-std", type=float, default=0.01, help="ZUPT noise std-dev (m/s)")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.05, help="Initial position guess std-dev (m)")
    parser.add_argument("--init-vel-noise-std", type=float, default=0.05, help="Initial velocity guess std-dev (m/s)")
    parser.add_argument("--n-mc-trials", type=int, default=500, help="Monte Carlo trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--stride", type=int, default=2, help="Ticks per GIF frame")
    parser.add_argument("--hold", type=int, default=12, help="Extra frames held at the end of each pass")
    parser.add_argument("--fps", type=int, default=16, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=70, help="GIF resolution")
    parser.add_argument("--out", type=str, default="rts_smoother.gif", help="Output GIF path")
    args = parser.parse_args()

    # Same rng consumption order as rts_smoother.main
    rng = np.random.default_rng(args.seed)
    x_true, z, is_anchor, is_cruise = generate_ground_truth_and_data(
        args.duration, args.dt, args.t_anchor, args.t_extend, args.t_ramp, args.v_extend,
        args.pos_noise_std, rng)
    n_steps = len(x_true) - 1
    init_std = np.array([args.init_pos_noise_std, args.init_vel_noise_std])
    x_init = x_true[0] + rng.normal(0.0, 1.0, 2) * init_std
    P_init = np.diag(init_std ** 2)
    R_pos, R_vel = args.pos_noise_std ** 2, args.zupt_noise_std ** 2
    z_vel = np.where(is_anchor, 0.0, np.nan)
    q = args.process_noise_std

    fwd = run_forward_filter(x_init, P_init, z, z_vel, args.dt, n_steps, q, R_pos, R_vel)
    x_s, P_s, _ = rts_smoother(fwd)
    x_f, sig_f = fwd["x_f"], np.sqrt(fwd["P_f"].diagonal(axis1=1, axis2=2))
    sig_s = np.sqrt(P_s.diagonal(axis1=1, axis2=2))

    print(f"Running Monte Carlo ({args.n_mc_trials} trials)...")
    mc = run_monte_carlo("inchworm", x_true, args.dt, q, args.pos_noise_std, args.zupt_noise_std, is_anchor,
                         init_std, args.n_mc_trials, (), np.random.default_rng([args.seed, 1]))
    rm_f, rm_s = np.sqrt(mc["sq_vel_f"]), np.sqrt(mc["sq_vel_s"])
    print(f"Velocity RMSE over ticks 1..N (Monte Carlo): filter {np.sqrt(np.mean(mc['sq_vel_f'][1:])):.4f} m/s, "
          f"smoother {np.sqrt(np.mean(mc['sq_vel_s'][1:])):.4f} m/s")

    t = np.arange(n_steps + 1) * args.dt
    blue, red = "tab:blue", "tab:red"

    fig = plt.figure(figsize=(10, 9.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.55, 1.0, 0.8], hspace=0.25)
    ax_t = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1])
    ax_r = fig.add_subplot(gs[2], sharex=ax_v)
    plt.setp(ax_v.get_xticklabels(), visible=False)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.11)
    status = fig.text(0.5, 0.955, "", ha="center", fontsize=12, fontweight="bold")

    # track panel
    p_lo, p_hi = x_true[:, 0].min(), x_true[:, 0].max()
    pad = 0.06 * (p_hi - p_lo)
    ax_t.set_xlim(p_lo - pad, p_hi + pad)
    ax_t.set_ylim(-1.1, 1.05)
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
    est_mark, = ax_t.plot([], [], "v", color=blue, markersize=9, zorder=4)
    ax_t.axhline(-0.6, color="0.5", lw=0.5, alpha=0.4)
    est_label = ax_t.text(0.01, 0.05, "", transform=ax_t.transAxes, fontsize=8.5)
    anchor_starts = [k for k in range(n_steps + 1) if is_anchor[k] and (k == 0 or not is_anchor[k - 1])]
    ax_t.set_title("Inchworm: one set of measurements, processed forward (filter) then backward (RTS smoother)",
                   fontsize=10.5)

    edges = np.diff(np.concatenate([[False], is_anchor, [False]]).astype(int))
    for ax in (ax_v, ax_r):
        for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            ax.axvspan(t[s] - args.dt / 2, t[e - 1] + args.dt / 2, color="gold", alpha=0.18, lw=0)
        ax.set_xlim(0.0, t[-1])

    v_lim = max(0.20, 1.05 * float(np.max(np.abs(x_f[:, 1]) + 2 * sig_f[:, 1])))
    ax_v.set_ylim(-0.12, min(v_lim, 0.40))
    ax_v.set_ylabel("Velocity (m/s), band = ±2σ")
    ax_v.text(0.01, 0.95, "shaded gold = anchor phase", transform=ax_v.transAxes, fontsize=8, va="top")
    ax_v.plot(t, x_true[:, 1], color="green", lw=1.5, ls="--", alpha=0.3)
    truth_line, = ax_v.plot([], [], color="green", lw=2, ls="--", label="Ground truth")
    f_line, = ax_v.plot([], [], color=blue, lw=1.6, label="filter (data up to the cursor)")
    s_line, = ax_v.plot([], [], color=red, lw=1.8, label="RTS smoother (all data)")
    ax_v.legend(loc="upper right", fontsize=8, ncol=1)

    r_max = 1.1 * float(max(rm_f.max(), rm_s.max()))
    ax_r.set_ylim(0.0, r_max)
    ax_r.set_ylabel("Velocity RMSE (m/s)")
    ax_r.set_xlabel("Time (s)")
    ax_r.set_title(f"Monte Carlo ({args.n_mc_trials} trials), RMSE per tick", fontsize=9.5, pad=3)
    rf_line, = ax_r.plot([], [], color=blue, lw=2, label="filter")
    rs_line, = ax_r.plot([], [], color=red, lw=2, label="RTS smoother")
    ax_r.legend(loc="upper right", fontsize=8, ncol=2)

    fig.text(0.5, 0.012,
             "The smoother cuts the velocity error most in the extend phase, above all just before the next anchor phase, where the coming ZUPT\n"
             "tells it the body stops. Inside the anchor phase the filter's ZUPT already pins v, so the smoother is no better there. Both agree at the last tick.",
             ha="center", fontsize=8.5)

    frames = [("fwd", k) for k in range(0, n_steps + 1, args.stride)]
    frames += [("fwd", n_steps)] * args.hold
    frames += [("bwd", k) for k in range(n_steps, -1, -args.stride)]
    frames += [("bwd", 0)] * args.hold
    per_frame = []  # artists redrawn every frame (bands, the "not measured yet" cover)

    def update(frame):
        mode, k = frame
        for a in per_frame:
            a.remove()
        per_frame.clear()
        body.set_x(x_true[k, 0] - body_w / 2)
        body.set_color("gold" if is_anchor[k] else "tab:green")
        body.set_edgecolor("black")
        phase_text.set_text("ANCHORED (v = 0)" if is_anchor[k] else "EXTENDING")
        phase_text.set_color("darkgoldenrod" if is_anchor[k] else "darkgreen")
        steps = [x_true[a, 0] for a in anchor_starts if a <= k]
        footprints.set_data(steps, [0.0] * len(steps))
        if mode == "fwd":
            status.set_text(f"FORWARD PASS: the filter at t = {t[k]:.2f} s has seen the data up to now only")
            status.set_color(blue)
            est_mark.set_data([x_f[k, 0]], [-0.6])
            est_mark.set_color(blue)
            est_label.set_text("blue: the filter's position estimate")
            truth_line.set_data(t[:k + 1], x_true[:k + 1, 1])
            f_line.set_data(t[:k + 1], x_f[:k + 1, 1])
            s_line.set_data([], [])
            rf_line.set_data(t[:k + 1], rm_f[:k + 1])
            rs_line.set_data([], [])
            per_frame.append(ax_v.fill_between(t[:k + 1], x_f[:k + 1, 1] - 2 * sig_f[:k + 1, 1],
                                               x_f[:k + 1, 1] + 2 * sig_f[:k + 1, 1], color=blue, alpha=0.18, lw=0))
            for ax in (ax_v, ax_r):
                per_frame.append(ax.axvspan(t[k], t[-1] + args.dt / 2, color="0.55", alpha=0.25, lw=0, zorder=5))
        else:
            status.set_text(f"BACKWARD PASS (RTS): the smoother at t = {t[k]:.2f} s uses all the data, past and future")
            status.set_color(red)
            est_mark.set_data([x_s[k, 0]], [-0.6])
            est_mark.set_color(red)
            est_label.set_text("red: the smoother's position estimate")
            truth_line.set_data(t, x_true[:, 1])
            f_line.set_data(t, x_f[:, 1])
            s_line.set_data(t[k:], x_s[k:, 1])
            rf_line.set_data(t, rm_f)
            rs_line.set_data(t[k:], rm_s[k:])
            per_frame.append(ax_v.fill_between(t, x_f[:, 1] - 2 * sig_f[:, 1], x_f[:, 1] + 2 * sig_f[:, 1],
                                               color=blue, alpha=0.10, lw=0))
            per_frame.append(ax_v.fill_between(t[k:], x_s[k:, 1] - 2 * sig_s[k:, 1], x_s[k:, 1] + 2 * sig_s[k:, 1],
                                               color=red, alpha=0.25, lw=0))
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
