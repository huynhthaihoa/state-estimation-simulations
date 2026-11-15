'''
Animated version of observability_fej.py: one robot drives ten laps of a circle
among 20 unknown landmarks, and three EKF-SLAM filters process the SAME noisy
odometry and landmark measurements: the standard EKF, the FEJ-EKF and the
invariant EKF (invariant EKF-SLAM of Barrau & Bonnabel).

Top row, one panel per filter (the scene as that filter sees it):
  dashed grey = true path, black triangles = true landmarks, coloured dots with
  ellipses = the filter's landmark estimates with their 99% uncertainty
  ellipses (drawn 10 times larger, they are about 0.1 m at the scale of the
  map), coloured line = estimated robot path, arrow = estimated pose, black circle = true position, thin
  lines = the landmarks measured now. The three scenes look alike at this scale;
  the difference is in the bottom row.
Bottom row, sharing the time axis:
  1. consistency: robot pose NEES averaged over the Monte Carlo trials against
     the consistent value 3 and the 95% band of the average;
  2. heading error of the displayed trial (solid) and the filter's reported
     +-3 standard deviation envelope (dashed);
  3. spurious information: the information the filter adds along the global
     rotation, summed over time (symlog scale); exactly 0 for FEJ and the
     invariant EKF, growing for the standard EKF.

The panels reuse the simulation's own run_filter / run_monte_carlo, with the
same random streams as observability_fej.main (same seed -> the displayed trial
is its Monte Carlo trial 0). The default Monte Carlo (50 trials, 3 filters)
takes about 40 s.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse

from state_estimation_simulations.use_numpy.observability_fej import (
    build_parser, params_from_args, make_scene, visibility, draw_trial, run_filter, run_monte_carlo)
from state_estimation_simulations.utils import averaged_nees_bounds

FILTERS = ("standard", "fej", "iekf")
TITLES = {"standard": "standard EKF", "fej": "FEJ-EKF", "iekf": "invariant EKF"}
COLORS = {"standard": "tab:red", "fej": "tab:blue", "iekf": "tab:green"}


def cov_ellipse(ax, mean, cov, color, chi2=9.21, scale=10.0):
    """Draws the 99% uncertainty ellipse (chi-squared quantile 9.21, 2 DoF) of a 2D Gaussian, enlarged by `scale`."""
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 1e-12, None)
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ax.add_patch(Ellipse(mean, 2 * scale * np.sqrt(chi2 * w[1]), 2 * scale * np.sqrt(chi2 * w[0]), angle=ang, fill=False,
                         edgecolor=color, lw=0.8, alpha=0.9))


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.formatter_class = __import__("argparse").RawDescriptionHelpFormatter
    parser.set_defaults(out="observability_fej.gif")
    parser.add_argument("--stride", type=int, default=5, help="Filter steps per GIF frame")
    parser.add_argument("--fps", type=int, default=10, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=60, help="GIF resolution")
    args = parser.parse_args()
    p = params_from_args(args)

    # same streams as observability_fej.main
    rng_map, rng_mc, _, _ = np.random.default_rng(args.seed).spawn(4)
    scene = make_scene(p, rng_map)
    vis = visibility(p, scene)
    N = scene["n_steps"]
    print(f"Running {args.n_trials} Monte Carlo trials x {len(FILTERS)} filters ...")
    res = run_monte_carlo(p, scene, vis, FILTERS, args.n_trials, rng_mc)
    trial0 = draw_trial(p, scene, np.random.default_rng(args.seed).spawn(4)[1].spawn(args.n_trials)[0])
    run = {k: run_filter(k, p, scene, vis, trial0, keep_states=True) for k in FILTERS}
    lo, hi = averaged_nees_bounds(3, args.n_trials)
    t = np.arange(N + 1) * p.dt
    poses, lms = scene["poses"], scene["lms"]

    fig = plt.figure(figsize=(12, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.7, 1.0])
    ax_sc = {k: fig.add_subplot(gs[0, i]) for i, k in enumerate(FILTERS)}
    ax_n, ax_h, ax_i = (fig.add_subplot(gs[1, i]) for i in range(3))
    lim = scene["radius"] + p.ring_halfwidth + 1.0
    cx, cy = scene["center"]

    ax_n.axhline(3, color="black", lw=1)
    ax_n.axhline(lo, color="black", ls="--", lw=0.8)
    ax_n.axhline(hi, color="black", ls="--", lw=0.8)
    ax_n.set_xlim(0, t[-1])
    finite = np.concatenate([res[k]["nees_robot"][np.isfinite(res[k]["nees_robot"])] for k in FILTERS])
    ax_n.set_ylim(0.0, min(finite.max() * 1.05, 12.0))
    ax_n.set_ylabel(f"robot pose NEES ({args.n_trials} trials)")
    ax_n.set_xlabel("Time (s)")
    ax_n.set_title("robot pose NEES; dashed: 95% band", fontsize=9)
    ax_h.set_xlim(0, t[-1])
    hmax = 3.2 * np.rad2deg(max(run[k]["sig_theta"][2:].max() for k in FILTERS))
    ax_h.set_ylim(-min(hmax, 40), min(hmax, 40))
    ax_h.axhline(0, color="black", lw=0.6)
    ax_h.set_ylabel("heading error (deg)")
    ax_h.set_xlabel("Time (s)")
    ax_h.set_title("one trial: heading error, +-3 std dashed", fontsize=9)
    ax_i.set_xlim(0, t[-1])
    cum = {k: np.cumsum(run[k]["info_gain"]) for k in FILTERS}
    ax_i.set_yscale("symlog", linthresh=1.0)
    ax_i.set_ylim(-0.5, 3 * max(c.max() for c in cum.values()))
    ax_i.set_ylabel("information along rotation")
    ax_i.set_xlabel("Time (s)")
    ax_i.set_title("spurious information, summed", fontsize=9)
    for k in FILTERS:
        ax_n.plot(t, res[k]["nees_robot"], color=COLORS[k], lw=0.4, alpha=0.25)
    lines = {}
    for k in FILTERS:
        lines[("n", k)], = ax_n.plot([], [], color=COLORS[k], lw=1.6, label=TITLES[k])
        lines[("e", k)], = ax_h.plot([], [], color=COLORS[k], lw=1.0)
        lines[("u", k)], = ax_h.plot([], [], color=COLORS[k], lw=0.8, ls="--")
        lines[("l", k)], = ax_h.plot([], [], color=COLORS[k], lw=0.8, ls="--")
        lines[("i", k)], = ax_i.plot([], [], color=COLORS[k], lw=1.6, label=TITLES[k])
    ax_n.legend(fontsize=7, loc="upper left")
    fig.tight_layout()

    frames = list(range(2, N + 1, args.stride)) + [N] * 6

    def draw(k):
        for name, ax in ax_sc.items():
            ax.cla()
            r = run[name]
            ax.plot(poses[:, 0], poses[:, 1], color="0.6", ls="--", lw=0.8)
            ax.plot(lms[:, 0], lms[:, 1], "k^", ms=4)
            ax.plot(r["x_est"][:k + 1, 0], r["x_est"][:k + 1, 1], color=COLORS[name], lw=1.0)
            for j in np.where(np.isfinite(r["lm_hist"][k, :, 0]))[0]:
                m = r["lm_hist"][k, j]
                ax.plot(m[0], m[1], ".", color=COLORS[name], ms=4)
                cov_ellipse(ax, m, r["lmP_hist"][k, j], COLORS[name])
            for j in np.where(vis[k])[0]:
                if np.isfinite(r["lm_hist"][k, j, 0]):
                    ax.plot([r["x_est"][k, 0], r["lm_hist"][k, j, 0]], [r["x_est"][k, 1], r["lm_hist"][k, j, 1]],
                            color="0.5", lw=0.4)
            ax.plot(poses[k, 0], poses[k, 1], "ko", ms=5, mfc="none")
            x, y, th = r["x_est"][k]
            ax.annotate("", xy=(x + 1.4 * np.cos(th), y + 1.4 * np.sin(th)), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>", color=COLORS[name], lw=1.6))
            ax.set_xlim(cx - lim, cx + lim)
            ax.set_ylim(cy - lim, cy + lim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{TITLES[name]}: t = {t[k]:.0f} s, heading std {np.rad2deg(r['sig_theta'][k]):.1f} deg", fontsize=9)
        for name in FILTERS:
            r = run[name]
            lines[("n", name)].set_data(t[:k + 1], res[name]["nees_robot"][:k + 1])
            sig3 = 3 * np.rad2deg(r["sig_theta"][:k + 1])
            lines[("e", name)].set_data(t[:k + 1], np.rad2deg(r["err"][:k + 1, 2]))
            lines[("u", name)].set_data(t[:k + 1], sig3)
            lines[("l", name)].set_data(t[:k + 1], -sig3)
            lines[("i", name)].set_data(t[:k + 1], cum[name][:k + 1])
        return []

    anim = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved animation to {args.out}")


if __name__ == "__main__":
    main()
