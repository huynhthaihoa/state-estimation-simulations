'''
Animated version of inchworm_slip_imm.py's default run: a 1D inchworm crawls
ten anchor/extend cycles, two anchor phases contain a backward slip pulse, and
interacting multiple model (IMM) filters keep a probability P(M) that the body
is moving, from noisy position measurements with the pad displacement reading
(imm_schedule) or without it (imm_schedule_pos).

Four panels, drawn progressively (gold = slip pulses, light blue = scheduled
anchor phases):
  1. the body on its track: the true body is colored by the position+pad IMM's
     P(M) (blue = still, red = moving); the triangles below mark where each IMM
     puts the body, colored by its own P(M);
  2. velocity: truth, schedule_and_sensor, sensor_fused and imm_schedule;
  3. P(M) of imm_schedule and imm_schedule_pos against the 0.5 line;
  4. consistency: Monte Carlo averaged NEES (log scale) of schedule,
     sensor_fused, imm_schedule and imm_schedule_pos against the expected 2
     (2 DoF) and the 95% band for the average.

Message (doc docs/filtering/inchworm_slip_imm.md): with the pad reading the
IMM's P(M) rises inside the slip pulses and its NEES stays near 2, like
sensor_fused but with a lower velocity error; from position alone the slip is
too small to see (4 mm per tick against 20 mm noise), P(M) stays low through
the pulses and the NEES rises, though less than the schedule's.

The per-step estimates and the Monte Carlo curves are the simulation's own
(make_setup and evaluate of inchworm_slip_imm.py, same rng order). The default
Monte Carlo (500 trials) takes about a minute.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from state_estimation_simulations.use_numpy.inchworm_slip_imm import (STATE_DIM, add_model_arguments, evaluate, imm_spec,
                                                                     make_setup, policy_spec, slip_pulses)
from state_estimation_simulations.utils import averaged_nees_bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--stride", type=int, default=2, help="Filter ticks per GIF frame")
    parser.add_argument("--fps", type=int, default=12, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=70, help="GIF resolution")
    parser.add_argument("--out", type=str, default="inchworm_slip_imm.gif", help="Output GIF path")
    args = parser.parse_args()

    setup = make_setup(args, args.sensor_noise_std)
    specs = {"schedule": policy_spec("schedule"), "schedule_and_sensor": policy_spec("schedule_and_sensor"),
             "sensor_fused": policy_spec("sensor_fused"), "imm_schedule": imm_spec("schedule", True),
             "imm_schedule_pos": imm_spec("schedule", False)}
    print(f"Running single runs and Monte Carlo ({args.n_mc_trials} trials)...")
    res = evaluate(args, setup, specs, args.sensor_noise_std)
    lo, hi = averaged_nees_bounds(STATE_DIM, args.n_mc_trials)
    x_true, is_anchor, is_slip, is_clean, is_cruise = (setup.x_true, setup.is_anchor, setup.is_slip, setup.is_clean,
                                                       setup.is_cruise)
    print("\nMonte Carlo NEES (clean anchor | slip | cruise), RMSE vel (single run):")
    for m in ("schedule", "sensor_fused", "imm_schedule", "imm_schedule_pos"):
        n = res[m]["nees"]
        print(f"  {m:<18s} {n[is_clean].mean():7.2f} | {n[is_slip].mean():7.2f} | {n[is_cruise].mean():7.2f}"
              f" | {res[m]['rmse_vel']:.4f} m/s")
    print(f"  95% bounds for the {args.n_mc_trials}-trial average: [{lo:.2f}, {hi:.2f}]")

    n_steps = len(x_true) - 1
    t = np.arange(n_steps + 1) * args.dt
    pm_pad, pm_pos = res["imm_schedule"]["p_move"], res["imm_schedule_pos"]["p_move"]
    cmap = LinearSegmentedColormap.from_list("still_moving", ["tab:blue", "lightgray", "tab:red"])

    fig = plt.figure(figsize=(10, 11.5))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.75, 1.2, 0.9, 1.2], hspace=0.26)
    ax_t = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1])
    ax_p = fig.add_subplot(gs[2], sharex=ax_v)
    ax_n = fig.add_subplot(gs[3], sharex=ax_v)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.09)
    fig.suptitle("1D inchworm with anchor slip: an IMM keeps a probability of 'moving'", fontsize=12)
    for ax in (ax_v, ax_p):
        plt.setp(ax.get_xticklabels(), visible=False)

    # body panel
    p_lo, p_hi = x_true[:, 0].min(), x_true[:, 0].max()
    pad = 0.06 * (p_hi - p_lo)
    ax_t.set_xlim(p_lo - pad, p_hi + pad)
    ax_t.set_ylim(-1.45, 1.1)
    ax_t.set_yticks([])
    for side in ("left", "right", "top"):
        ax_t.spines[side].set_visible(False)
    ax_t.set_xlabel("Position along the track (m)", fontsize=9, labelpad=1)
    ax_t.axhline(0.0, color="saddlebrown", lw=3)
    body_w = 0.035 * (p_hi - p_lo + 2 * pad)
    body = Rectangle((x_true[0, 0] - body_w / 2, 0.02), body_w, 0.55, color=cmap(pm_pad[0]), ec="black", lw=1, zorder=4)
    ax_t.add_patch(body)
    status = ax_t.text(0.99, 0.97, "", transform=ax_t.transAxes, ha="right", va="top", fontsize=10, fontweight="bold")
    pm_text = ax_t.text(0.01, 0.97, "", transform=ax_t.transAxes, ha="left", va="top", fontsize=9)
    rows = {"imm_schedule": -0.6, "imm_schedule_pos": -1.1}
    est_marks = {}
    for lab, y in rows.items():
        ax_t.axhline(y, color="0.7", lw=0.5)
        est_marks[lab], = ax_t.plot([], [], "v", markersize=11, mec="black", zorder=4)
        ax_t.text(p_hi + pad, y, lab, fontsize=7.5, ha="right", va="bottom")
    truth_tick, = ax_t.plot([], [], color="green", lw=1, ls=":", zorder=2)

    anchor_edges = np.diff(np.concatenate([[False], is_anchor, [False]]).astype(int))
    for ax in (ax_v, ax_p, ax_n):
        for a, b in zip(np.where(anchor_edges == 1)[0], np.where(anchor_edges == -1)[0]):
            ax.axvspan(t[a] - args.dt / 2, t[b - 1] + args.dt / 2, color="tab:blue", alpha=0.06, lw=0)
        for a, b in slip_pulses(is_slip):
            ax.axvspan(t[a], t[b - 1] + args.dt, color="gold", alpha=0.3, lw=0)
        ax.set_xlim(0.0, t[-1])

    colors = {"schedule": "tab:blue", "schedule_and_sensor": "tab:purple", "sensor_fused": "tab:green",
              "imm_schedule": "tab:red", "imm_schedule_pos": "tab:cyan"}
    ax_v.set_ylim(-0.12, 0.2)
    ax_v.set_ylabel("Velocity (m/s)")
    ax_v.plot(t, x_true[:, 1], color="black", lw=1.2, ls="--", alpha=0.3)
    truth_line, = ax_v.plot([], [], color="black", lw=2, ls="--", label="Ground truth")
    vlines = {m: ax_v.plot([], [], color=colors[m], lw=1.4, label=m)[0]
              for m in ("schedule_and_sensor", "sensor_fused", "imm_schedule")}
    ax_v.legend(loc="upper right", fontsize=7.5, ncol=2)

    ax_p.set_ylim(-0.05, 1.5)
    ax_p.set_ylabel("P(moving)")
    ax_p.axhline(0.5, color="black", lw=0.8, ls=":")
    plines = {"imm_schedule": ax_p.plot([], [], color=colors["imm_schedule"], lw=1.6,
                                        label="imm_schedule (position + pad)")[0],
              "imm_schedule_pos": ax_p.plot([], [], color=colors["imm_schedule_pos"], lw=1.6,
                                            label="imm_schedule_pos (position only)")[0]}
    ax_p.legend(loc="upper center", fontsize=7.5, ncol=2)

    ax_n.set_yscale("log")
    ax_n.set_ylim(0.4, 1500)
    ax_n.set_ylabel(f"Monte Carlo avg. NEES ({args.n_mc_trials} trials)")
    ax_n.set_xlabel("Time (s)")
    ax_n.axhline(STATE_DIM, color="black", lw=1, label=f"Expected NEES ({STATE_DIM} DoF)")
    ax_n.axhspan(lo, hi, color="black", alpha=0.10, lw=0, label="95% band for the average")
    nlines = {m: ax_n.plot([], [], color=colors[m], lw=1.8, label=m)[0]
              for m in ("schedule", "sensor_fused", "imm_schedule", "imm_schedule_pos")}
    ax_n.legend(loc="upper center", fontsize=7.5, ncol=3)

    fig.text(0.5, 0.012,
             "With the pad reading, P(moving) rises inside each slip pulse and the NEES stays near 2, like sensor_fused but with a lower velocity error.\n"
             "From position alone a 4 mm per tick slip is buried in 20 mm noise: P(moving) stays low through the pulse and the NEES rises, less than the schedule's.",
             ha="center", fontsize=8)

    frames = list(range(0, n_steps + 1, args.stride))

    def update(k):
        sl = slice(0, k + 1)
        p = x_true[k, 0]
        body.set_x(p - body_w / 2)
        body.set_color(cmap(pm_pad[k]))
        body.set_edgecolor("black")
        if is_slip[k]:
            status.set_text("SLIPPING (true v < 0)")
            status.set_color("darkgoldenrod")
        elif is_anchor[k]:
            status.set_text("ANCHORED (true v = 0)")
            status.set_color("tab:blue")
        else:
            status.set_text("EXTENDING")
            status.set_color("darkgreen")
        pm_text.set_text(f"P(moving): pad {pm_pad[k]:.2f}, position only {pm_pos[k]:.2f}\n"
                         "body and triangles: blue = still, red = moving")
        truth_tick.set_data([p, p], [-1.3, 0.0])
        for lab, pm in (("imm_schedule", pm_pad), ("imm_schedule_pos", pm_pos)):
            est_marks[lab].set_data([res[lab]["x_est"][k, 0]], [rows[lab]])
            est_marks[lab].set_color(cmap(pm[k]))
        truth_line.set_data(t[sl], x_true[sl, 1])
        for m, ln in vlines.items():
            ln.set_data(t[sl], res[m]["x_est"][sl, 1])
        plines["imm_schedule"].set_data(t[sl], pm_pad[sl])
        plines["imm_schedule_pos"].set_data(t[sl], pm_pos[sl])
        for m, ln in nlines.items():
            ln.set_data(t[sl], res[m]["nees"][sl])
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
