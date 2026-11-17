'''
Animated version of use_numpy/particle_filter_localization.py: a robot drives
right along a 30 m corridor with five doors, starting from an unknown position.
A particle filter (PF), an EKF and a UKF estimate where it is from the odometry
and a door-profile sensor that does not say WHICH door it sees.

Two panels, drawn progressively over the 48 steps (24 s) of example trial 0
(default flags of the script):
  1. left: the corridor. Grey = the door profile f(x) the sensor measures, blue
     bars = the PF belief (weight of the particles in each 0.25 m bin), red and
     purple curves = the EKF and UKF Gaussian beliefs (one hill each), black
     dashed line and robot = the true position. The title shows N_eff, the
     effective number of particles (1 / sum of squared weights). A strip under
     the axis marks the doors and the corridor walls; positions outside the
     corridor are shaded;
  2. right: the Monte Carlo result: the fraction of trials whose estimate is
     more than half the smallest door spacing (2 m) from the truth ("wrong
     door") for PF / EKF / UKF over time (static curves plus a moving time
     marker), and N_eff / N of the example trial with the resampling threshold.

Message (doc docs/filtering/particle_filter_localization.md): at t = 0 the PF
belief has a hill at every position that looks like a door position, one hill
per door by t = 4 s, and the hills merge to one around t = 13 s, so almost no
trial ends at the wrong door; the EKF and UKF hold one Gaussian, so most of
their trials end at a wrong door (the EKF mean even leaves the corridor in
the example trial).

Everything shown is the script's own output: draw_noise, generate_trial,
pf_on_trial(record=True) for the example trial, run_monte_carlo for the curves
(default 400 trials, a few seconds). Writes a GIF with matplotlib's pillow
writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Patch, Rectangle

from state_estimation_simulations.use_numpy.particle_filter_localization import (
    COLORS, Config, door_profile, draw_noise, generate_trial, pf_on_trial, run_monte_carlo)

BIN_W = 0.25   # histogram bin width of the PF belief (m)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed of all random streams")
    parser.add_argument("--trials", type=int, default=400, help="Monte Carlo trials for the right panel")
    parser.add_argument("--particles", type=int, default=Config().n_particles, help="number of particles N")
    parser.add_argument("--example-trial", type=int, default=0, help="trial index animated in the left panel")
    parser.add_argument("--hold", type=int, default=8, help="extra frames that hold the last step")
    parser.add_argument("--fps", type=int, default=5, help="GIF frames per second (one frame per step)")
    parser.add_argument("--dpi", type=int, default=70, help="GIF resolution")
    parser.add_argument("--out", type=str, default="particle_filter_localization.gif", help="Output GIF path")
    args = parser.parse_args()

    cfg = Config(n_particles=args.particles)
    noises = draw_noise(args.seed, args.trials, cfg.steps)
    x_true, res = run_monte_carlo(cfg, noises)
    ex = args.example_trial
    data = generate_trial(cfg, noises[ex])
    pf = pf_on_trial(cfg, noises[ex], data, record=True)
    assert np.allclose(pf["mean"], res["PF"]["mean"][ex]), "example trial differs from the Monte Carlo trial"
    xt = x_true[ex]
    n_steps = cfg.steps
    t = np.arange(n_steps + 1) * cfg.dt
    names = ("PF", "EKF", "UKF")
    wrong = {m: np.mean(np.abs(res[m]["mean"] - x_true) > cfg.wrong_thr, axis=0) for m in names}
    ess_frac = pf["ess"] / cfg.n_particles
    print(f"{args.trials} trials, example trial {ex}: start {xt[0]:.2f} m, end {xt[-1]:.2f} m; final wrong-door "
          + ", ".join(f"{m} {wrong[m][-1]:.3f}" for m in names)
          + f"; EKF mean at the end {res['EKF']['mean'][ex, -1]:.1f} m")

    x_lo, x_hi = -2.0, cfg.length + 10.0
    bins = np.arange(x_lo, x_hi + BIN_W, BIN_W)
    centres = 0.5 * (bins[:-1] + bins[1:])
    heights = np.stack([np.histogram(pf["x_hist"][k], bins=bins, weights=pf["w_hist"][k])[0]
                        for k in range(n_steps + 1)])
    xs = np.linspace(x_lo, x_hi, 1000)

    def gauss(name, k):
        m, v = res[name]["mean"][ex, k], res[name]["var"][ex, k]
        return BIN_W * np.exp(-0.5 * (xs - m) ** 2 / v) / np.sqrt(2.0 * np.pi * v)

    fig = plt.figure(figsize=(14.5, 6.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1.0], height_ratios=[1.0, 1.0],
                          left=0.075, right=0.985, top=0.80, bottom=0.17, wspace=0.17, hspace=0.42)
    gs_left = gs[:, 0].subgridspec(2, 1, height_ratios=[6.0, 1.0], hspace=0.06)
    ax = fig.add_subplot(gs_left[0])
    ax_c = fig.add_subplot(gs_left[1], sharex=ax)
    ax_w = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[1, 1])
    title = fig.text(0.06, 0.945, "", fontsize=17, fontweight="bold", ha="left")
    fig.text(0.06, 0.895, "N_eff = effective number of particles (how many particles effectively count; "
             "N_eff = N when all weigh the same)", fontsize=11, ha="left", color="0.25")
    status = fig.text(0.06, 0.845, "", fontsize=13, fontweight="bold", ha="left")

    # --- left panel: belief over the corridor (the y axis rescales to the tallest belief peak) ---
    prof = door_profile(xs, cfg.doors, cfg.width)
    for a in (ax, ax_c):
        a.axvspan(x_lo, 0.0, color="tab:red", alpha=0.07, lw=0)
        a.axvspan(cfg.length, x_hi, color="tab:red", alpha=0.07, lw=0)
    ax_c.text(cfg.length + 0.7, 0.5, "outside the\ncorridor", ha="left", va="center", fontsize=10, color="tab:red")
    ax_c.text(cfg.length / 2, 0.5, "corridor (brown = doors)", ha="center", va="center", fontsize=10, color="0.3", zorder=1)
    prof_art = [ax.fill_between(xs, 0, prof, color="0.80")]
    bars = ax.bar(centres, heights[0], width=BIN_W, color=COLORS["PF"], alpha=0.75, label="PF belief (particles)")
    gl = {m: ax.plot(xs, gauss(m, 0), color=COLORS[m], lw=2.6, label=f"{m} hill (one Gaussian)")[0] for m in ("EKF", "UKF")}
    truth_line = ax.axvline(xt[0], color="k", ls="--", lw=1.8, label="true position")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylabel("belief per 0.25 m bin\n(axis rescales to the tallest peak)", fontsize=11.5)
    ax.tick_params(labelsize=11, labelbottom=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([Patch(fc="0.80")] + handles, ["door profile f(x) the sensor measures (height not to scale)"] + labels,
              loc="upper left", fontsize=10.5, ncol=2, framealpha=0.92)
    # --- corridor strip: walls, doors, robot ---
    ax_c.set_ylim(0, 1)
    ax_c.add_patch(Rectangle((0.0, 0.1), cfg.length, 0.8, fc="0.93", ec="0.35", lw=1.2, zorder=2))
    for d in cfg.doors:
        ax_c.add_patch(Rectangle((d - 0.4, 0.1), 0.8, 0.8, fc="saddlebrown", ec="none", zorder=3))
    robot = Rectangle((0, 0.2), 3.0, 0.6, fc="k", ec="k", zorder=5)
    ax_c.add_patch(robot)
    robot_lab = ax_c.text(0, 0.5, "robot", ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=6)
    ax_c.set_yticks([])
    ax_c.set_xlabel("position along the corridor (m)", fontsize=12)
    ax_c.tick_params(labelsize=11)
    truth_line_c = ax_c.axvline(xt[0], color="k", ls="--", lw=1.8, zorder=1)

    # --- right panels ---
    for m in names:
        ax_w.plot(t, wrong[m], color=COLORS[m], lw=2.4, label=m)
    ax_w.set_xlim(0, t[-1])
    ax_w.set_ylim(-0.02, 1.02)
    ax_w.set_title(f"Monte Carlo: trials at the wrong door ({args.trials} trials)", fontsize=12.5)
    ax_w.set_ylabel(f"fraction with error > {cfg.wrong_thr:g} m", fontsize=11)
    ax_w.tick_params(labelsize=10.5)
    ax_w.legend(loc="center right", fontsize=11)
    w_mark = ax_w.axvline(0.0, color="k", lw=1.2)
    w_dots = {m: ax_w.plot([], [], "o", color=COLORS[m], ms=7)[0] for m in names}

    ax_e.plot(t, ess_frac, color=COLORS["PF"], lw=2.0, label="N_eff / N (example trial)")
    ax_e.axhline(cfg.resample_threshold, color="k", ls=":", lw=1.4, label="resample below this")
    ax_e.set_xlim(0, t[-1])
    ax_e.set_ylim(0, 1.05)
    ax_e.set_title("PF: effective particles (example trial)", fontsize=12.5)
    ax_e.set_xlabel("time (s)", fontsize=11)
    ax_e.set_ylabel("N_eff / N", fontsize=11)
    ax_e.tick_params(labelsize=10.5)
    ax_e.legend(loc="lower right", fontsize=10)
    e_mark = ax_e.axvline(0.0, color="k", lw=1.2)
    e_dot, = ax_e.plot([], [], "o", color=COLORS["PF"], ms=7)

    fig.text(0.06, 0.035,
             "Same data for all three filters. The sensor reads the same bump at every door, so at t = 0 the PF keeps a hill at every position that "
             "could be a door position.\n"
             "The EKF and UKF can hold only one hill; after the PF has seen the pattern of door spacings its hills merge into one at the true position.",
             fontsize=10.5, ha="left", va="bottom", color="0.2")

    frames = list(range(n_steps + 1)) + [n_steps] * args.hold
    hill_thr = cfg.hill_mass

    def update(k):
        for b, h in zip(bars, heights[k]):
            b.set_height(h)
        g = {m: gauss(m, k) for m in gl}
        for m, ln in gl.items():
            ln.set_ydata(g[m])
        # a Gaussian with a tiny variance would exceed a probability of 1 per bin: clip it for the scale
        peak = max(heights[k].max(), *(min(1.0, v.max()) for v in g.values()))
        ax.set_ylim(0.0, 1.55 * peak)
        prof_art[0].remove()
        prof_art[0] = ax.fill_between(xs, 0, prof * peak, color="0.80", zorder=0)
        x = xt[k]
        truth_line.set_xdata([x, x])
        truth_line_c.set_xdata([x, x])
        robot.set_x(x - 1.5)
        robot_lab.set_position((x, 0.5))
        title.set_text(f"t = {t[k]:4.1f} s     N_eff = {pf['ess'][k]:4.0f} of N = {cfg.n_particles}")
        one = pf["hill_mass"][k] >= hill_thr
        em = res["EKF"]["mean"][ex, k]
        txt = "PF belief: " + ("ONE hill, at the true position" if one and abs(pf["mean"][k] - x) <= cfg.wrong_thr
                               else "ONE hill" if one else "SEVERAL hills")
        txt += "     EKF: one hill at %.0f m" % em + (" (outside the corridor)" if em < 0 or em > cfg.length else "")
        status.set_text(txt)
        status.set_color("darkgreen" if one else "darkorange")
        for m in names:
            w_dots[m].set_data([t[k]], [wrong[m][k]])
        w_mark.set_xdata([t[k], t[k]])
        e_mark.set_xdata([t[k], t[k]])
        e_dot.set_data([t[k]], [ess_frac[k]])
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
