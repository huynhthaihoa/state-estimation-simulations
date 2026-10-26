'''
Animated version of inchworm_two_pad_fusion_ekf.py: the planar two-pad inchworm
of inchworm_two_pad_ekf.py, now reading both pad sensors, with a stroke error
and sensor faults (spikes and dropouts). Five EKFs differ only in how they move
the moving pad: random_walk (anchored sensor only), encoder_model,
moving_sensor, both (combined) and cross_check (combined after a disagreement
test).

Three panels, drawn progressively (gold = slip pulses):
  1. top view (left): the true robot as two pads joined by a body line, the
     anchored pad filled (gold; red while a slip pulse is active) and the
     moving pad hollow; a pad's outline turns red while its sensor spikes and
     grey while it drops out; a fading trail of the true body midpoint and the
     estimated midpoints of random_walk, moving_sensor and cross_check;
  2. the two readings that show a slip: |anchored reading| and |moving
     reading - commanded stroke|, with the slip threshold, the fault episodes
     as a strip at the top, and crosses where cross_check's disagreement test
     fired;
  3. consistency: Monte Carlo averaged NEES (log scale) of all five variants
     against the expected 5 (5 DoF) and the 95% band for the average.

Message: a real slip moves both pads, so both readings rise together; a fault
makes them disagree. Reading the moving sensor without a check makes faults
worse; the cross-check catches most of them.

The per-step estimates are the simulation's own run_trial outputs (same rng
order as inchworm_two_pad_fusion_ekf.main). The default Monte Carlo (300
trials) takes about a minute.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D

import _paths  # noqa: F401  (puts use_numpy/ and the repo root on sys.path)
from inchworm_two_pad_fusion_ekf import (B, F, SPIKE, STATE_DIM, VARIANTS, build_parser, build_setup,
                                         commanded_change, run_monte_carlo, run_trial, simulate_trial,
                                         window_flags)
from utils import averaged_nees_bounds

SHOWN = ("random_walk", "moving_sensor", "cross_check")


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--stride", type=int, default=2, help="Filter ticks per GIF frame")
    parser.add_argument("--fps", type=int, default=10, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.set_defaults(out="inchworm_two_pad_fusion_ekf.gif")
    args = parser.parse_args()

    # Same rng consumption order as inchworm_two_pad_fusion_ekf.main
    rng = np.random.default_rng(args.seed)
    setup = build_setup(args, rng)
    n_steps, episodes, noise = setup["n_steps"], setup["episodes"], setup["noise"]
    is_slip, _ = window_flags(n_steps, args.dt, args.t_phase, episodes)
    trial = simulate_trial(args, setup, rng)
    runs = {v: run_trial(trial, args, setup, v) for v in VARIANTS}

    print(f"Running Monte Carlo ({args.n_mc_trials} trials)...")
    stats, _ = run_monte_carlo(args, setup, args.n_mc_trials, rng)
    lo, hi = averaged_nees_bounds(STATE_DIM, args.n_mc_trials)
    for v in VARIANTS:
        print(f"  {v:<14s} NEES {stats[v]['nees'].mean():6.2f}")

    t = np.arange(n_steps + 1) * args.dt
    x_true = trial["x_true"]
    s = np.where(trial["back"], 1.0, -1.0)[:, None]
    stroke = np.array([commanded_change(trial["e_cmd"][k], trial["z"][k][0]) for k in range(n_steps)])
    a_mag = np.linalg.norm(trial["d_a"], axis=1) * 1e3
    m_mag = np.linalg.norm(trial["d_m"] - s * stroke, axis=1) * 1e3
    d_thr_mm = noise["d_thr"] * 1e3
    fired = np.array([vd is not None for vd in runs["cross_check"][3]])
    mid_true = 0.5 * (x_true[:, B] + x_true[:, F])
    mid_est = {v: 0.5 * (runs[v][0][:, B] + runs[v][0][:, F]) for v in SHOWN}
    # Reading k is the interval ending at tick k + 1; hold it for the tick it starts at
    anchored_back = np.concatenate([trial["back"], trial["back"][-1:]])
    colors = {"random_walk": "tab:gray", "encoder_model": "tab:blue", "moving_sensor": "tab:orange",
              "both": "tab:purple", "cross_check": "tab:green"}

    def fault_on(pad, time):
        for p, kind, a, b, _ in trial["faults"]:
            if p == pad and a <= time < b:
                return kind
        return None

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.12, wspace=0.18)
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_n = fig.add_subplot(gs[1, 1], sharex=ax_d)
    plt.setp(ax_d.get_xticklabels(), visible=False)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.08)
    fig.suptitle("Two-pad inchworm reading both pad sensors, with sensor faults", fontsize=13)

    # Top view
    pts = np.vstack([x_true[:, B], x_true[:, F]] + [mid_est[v] for v in SHOWN])
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
    est_trails = {v: ax_xy.plot([], [], color=colors[v], lw=1.0, alpha=0.5)[0] for v in SHOWN}
    est_dots = {v: ax_xy.plot([], [], "D", color=colors[v], ms=7, mec="white", mew=0.8, zorder=6)[0] for v in SHOWN}
    body, = ax_xy.plot([], [], color="black", lw=4, solid_capstyle="round", zorder=4)
    pad_back, = ax_xy.plot([], [], "o", ms=16, mew=2.5, zorder=5)
    pad_front, = ax_xy.plot([], [], "o", ms=16, mew=2.5, zorder=5)
    status = ax_xy.text(0.02, 0.98, "", transform=ax_xy.transAxes, ha="left", va="top", fontsize=11,
                        fontweight="bold")

    handles = [Line2D([], [], color="black", lw=2.2, label="true body midpoint (trail)"),
               Line2D([], [], marker="o", ls="", ms=10, mfc="gold", mec="black", label="anchored pad"),
               Line2D([], [], marker="o", ls="", ms=10, mfc="white", mec="black", label="moving pad"),
               Line2D([], [], marker="o", ls="", ms=10, mfc="tab:red", mec="black", label="anchored pad, slipping"),
               Line2D([], [], marker="o", ls="", ms=10, mfc="white", mec="tab:red", mew=2.5,
                      label="sensor spike (outline)"),
               Line2D([], [], marker="o", ls="", ms=10, mfc="white", mec="tab:gray", mew=2.5,
                      label="sensor dropout (outline)")]
    handles += [Line2D([], [], marker="D", ls="", ms=7, color=colors[v], label=v) for v in SHOWN]
    ax_xy.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.94), fontsize=8.5, framealpha=0.9,
                 title="estimated midpoint", title_fontsize=8.5)

    for ax in (ax_d, ax_n):
        for _, a, b in episodes:
            ax.axvspan(a, b, color="gold", alpha=0.25, lw=0)
        ax.set_xlim(0.0, t[-1])
    cursors = [ax.axvline(0.0, color="black", lw=1.2) for ax in (ax_d, ax_n)]

    # Sensor panel
    top = 1.9 * max(a_mag.max(), m_mag.max())
    ax_d.set_ylim(0.0, top)
    ax_d.set_ylabel("Per tick (mm)")
    ax_d.set_title("Readings that show a slip (gold = slip pulse, top strip = faults)", fontsize=11)
    ax_d.axhline(d_thr_mm, color="black", ls="--", lw=1.1, label=f"slip threshold ({d_thr_mm:.2f} mm)")
    for pad, kind, a, b, _ in trial["faults"]:
        ax_d.axvspan(a, b, ymin=0.93, ymax=1.0, color="tab:red" if kind == SPIKE else "tab:gray", lw=0)
    a_line, = ax_d.plot([], [], color="tab:orange", lw=1.4, label="|anchored reading|")
    m_line, = ax_d.plot([], [], color="tab:blue", lw=1.4, label="|moving reading - commanded stroke|")
    x_marks, = ax_d.plot([], [], "x", color="tab:green", ms=6, mew=1.6, label="cross_check: sensors disagree")
    ax_d.legend(loc="upper center", bbox_to_anchor=(0.5, 0.86), fontsize=8, ncol=2)

    # NEES panel
    ax_n.set_yscale("log")
    all_n = np.concatenate([stats[v]["nees"] for v in VARIANTS])
    ax_n.set_ylim(min(0.5 * lo, 0.8 * all_n.min()), 6.0 * all_n.max())
    ax_n.set_ylabel(f"Monte Carlo avg. NEES ({args.n_mc_trials} trials)")
    ax_n.set_xlabel("Time (s)")
    ax_n.axhline(STATE_DIM, color="black", lw=1, label=f"Expected NEES ({STATE_DIM} DoF)")
    ax_n.axhspan(lo, hi, color="black", alpha=0.10, lw=0, label="95% band for the average")
    nlines = {v: ax_n.plot([], [], color=colors[v], lw=1.8, label=v)[0] for v in VARIANTS}
    ax_n.legend(loc="upper left", fontsize=8, ncol=2)

    frames = list(range(0, n_steps + 1, args.stride))
    if frames[-1] != n_steps:
        frames.append(n_steps)

    def update(k):
        sl = slice(0, k + 1)
        pb, pf = x_true[k, B], x_true[k, F]
        body.set_data([pb[0], pf[0]], [pb[1], pf[1]])
        slipping = bool(is_slip[k])
        anch_col = "tab:red" if slipping else "gold"
        tm = min(k, n_steps - 1) * args.dt + 0.5 * args.dt
        for marker, name, p, is_anch in ((pad_back, "back", pb, anchored_back[k]),
                                         (pad_front, "front", pf, not anchored_back[k])):
            marker.set_data([p[0]], [p[1]])
            marker.set_markerfacecolor(anch_col if is_anch else "white")
            kind = fault_on(name, tm)
            marker.set_markeredgecolor("black" if kind is None else ("tab:red" if kind == SPIKE else "tab:gray"))
        i0 = max(0, k - trail_n)
        for j, ln in enumerate(trail):
            a, b = i0 + j, i0 + j + 1
            if b <= k:
                ln.set_data(mid_true[a:b + 1, 0], mid_true[a:b + 1, 1])
                ln.set_alpha(0.08 + 0.7 * (j + 1) / trail_n)
            else:
                ln.set_data([], [])
        for v in SHOWN:
            est_trails[v].set_data(mid_est[v][max(0, k - 2 * trail_n):k + 1, 0],
                                   mid_est[v][max(0, k - 2 * trail_n):k + 1, 1])
            est_dots[v].set_data([mid_est[v][k, 0]], [mid_est[v][k, 1]])
        for v in VARIANTS:
            nlines[v].set_data(t[sl], stats[v]["nees"][sl])
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
        a_line.set_data(t[1:kd + 2], a_mag[:kd + 1])
        m_line.set_data(t[1:kd + 2], m_mag[:kd + 1])
        idx = np.nonzero(fired[:kd + 1])[0]
        x_marks.set_data(t[idx + 1], np.full(len(idx), 0.895 * top))
        for cur in cursors:
            cur.set_xdata([t[k], t[k]])
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
