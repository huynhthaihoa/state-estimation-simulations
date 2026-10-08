'''
Animated version of friction_anisotropic_ekf.py's default run: a robot crawls
one full loop on a friction-anisotropic pad (grips along its body axis, slips
sideways) while three EKFs track it. They differ only in the shape of the
position block of the process noise Q (isotropic / fixed_anisotropic /
heading_aware, see docs/filtering/friction_anisotropic_ekf.md sections 2-3).

Their position tracks overlap (RMS about 0.011-0.013 m), so the animation
shows what actually differs, the noise model and its consistency:
  1. left: the circular path with the robot and, at the robot, each policy's
     process-noise ellipse (2 sigma of its Q position block) plus the true
     slip covariance ellipse (anisotropic Q at the TRUE heading). The ellipses
     are about 0.005 m wide against a ~1.3 m circle, so they are drawn
     magnified by a fixed factor (--ellipse-scale);
  2. right: Monte Carlo averaged NEES vs time (log scale) with the 3-DoF
     expected value and the 95% band for the averaged NEES, plus the ellipse
     misalignment between the fixed reference heading and the robot heading
     (axis period pi, so it peaks at 90 degrees, not 180).

fixed_anisotropic keeps the right shape but stops matching the pad as the
robot turns; its NEES is worst near the 90 degree mismatch and only partly
recovers afterwards because error has leaked into the heading estimate.
isotropic stays mildly overconfident; heading_aware stays consistent. This
does not claim the tracks differ (they do not), nor that the NEES curves of
the animation come from the single run shown (they are the doc's Monte Carlo
average over --n-mc-trials independent trials).

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse

from friction_anisotropic_ekf import (
    generate_ground_truth_and_data, run_ekf_isotropic, run_ekf_fixed_anisotropic,
    run_ekf_heading_aware, run_monte_carlo_consistency, isotropic_Q_pos, anisotropic_Q_pos,
    averaged_nees_bounds, state_errors)

COLORS = {"isotropic": "tab:gray", "fixed_anisotropic": "tab:red", "heading_aware": "tab:blue"}


def q_ellipse(Q, center, scale, n_sigma=2.0):
    """n-sigma ellipse of a 2x2 covariance, magnified by `scale`."""
    w, V = np.linalg.eigh(Q)
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    return center, 2 * n_sigma * scale * np.sqrt(w[1]), 2 * n_sigma * scale * np.sqrt(w[0]), ang


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--v-cmd", type=float, default=0.2)
    parser.add_argument("--omega-cmd", type=float, default=None, help="Default 2*pi/duration (one loop)")
    parser.add_argument("--sigma-grip", type=float, default=0.02)
    parser.add_argument("--sigma-slip", type=float, default=0.1)
    parser.add_argument("--sigma-theta", type=float, default=0.01)
    parser.add_argument("--pos-noise-std", type=float, default=0.02)
    parser.add_argument("--init-pos-noise-std", type=float, default=0.05)
    parser.add_argument("--init-theta-noise-std", type=float, default=0.05)
    parser.add_argument("--n-mc-trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ellipse-scale", type=float, default=30.0, help="Magnification of the Q ellipses")
    parser.add_argument("--stride", type=int, default=3, help="Filter ticks per GIF frame")
    parser.add_argument("--fps", type=int, default=20, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.add_argument("--out", type=str, default="friction_anisotropic_ekf.gif", help="Output GIF path")
    parser.add_argument("--frames-png", type=str, default=None,
                        help="Optional prefix: also save sample frames <prefix>_<k>.png")
    args = parser.parse_args()
    omega = args.omega_cmd if args.omega_cmd is not None else 2.0 * np.pi / args.duration
    sg, ss, dt = args.sigma_grip, args.sigma_slip, args.dt

    # Same call order / RNG use as friction_anisotropic_ekf.main()
    rng = np.random.default_rng(args.seed)
    x_true, z = generate_ground_truth_and_data(args.duration, dt, args.v_cmd, omega, sg, ss,
                                               args.pos_noise_std, rng)
    n_steps = len(x_true) - 1
    theta_ref = x_true[0, 2]
    init_std = np.array([args.init_pos_noise_std, args.init_pos_noise_std, args.init_theta_noise_std])
    x_init = x_true[0] + rng.normal(0.0, 1.0, 3) * init_std
    P_init = np.diag(init_std ** 2)
    R_pos = args.pos_noise_std ** 2 * np.eye(2)

    common = (x_init, P_init, z, dt, n_steps, args.v_cmd, omega, sg, ss, args.sigma_theta, R_pos)
    x_iso, _ = run_ekf_isotropic(*common)
    x_fix, _ = run_ekf_fixed_anisotropic(*common, theta_ref)
    x_awr, _ = run_ekf_heading_aware(*common)

    print(f"Monte Carlo ({args.n_mc_trials} trials)...")
    t0 = time.time()
    nees_iso, nees_fix, nees_awr = run_monte_carlo_consistency(
        x_true, dt, args.v_cmd, omega, sg, ss, args.sigma_theta, R_pos, theta_ref,
        args.init_pos_noise_std, args.init_theta_noise_std, args.n_mc_trials, rng)
    print(f"  done in {time.time() - t0:.0f} s")
    lo, hi = averaged_nees_bounds(3, args.n_mc_trials)

    # Ellipse misalignment between the fixed reference and the true heading, axis period pi -> [0, 90] deg
    d = np.abs(((x_true[:, 2] - theta_ref + np.pi) % (2 * np.pi)) - np.pi)
    mis = np.degrees(np.minimum(d, np.pi - d))

    # ---- numbers shown vs the simulation's own printout ----
    rot = d
    q = np.pi / 4
    bins = [("0-quarter turn", rot < q), ("quarter-half turn", (rot >= q) & (rot < 2 * q)),
            ("half-3quarter turn", (rot >= 2 * q) & (rot < 3 * q)), ("3quarter-full turn", rot >= 3 * q)]
    print(f"95% band for the averaged NEES: [{lo:.2f}, {hi:.2f}]")
    for lab, m in bins:
        print(f"  {lab:<20s} iso={nees_iso[m].mean():8.2f} | fixed={nees_fix[m].mean():8.2f} | aware={nees_awr[m].mean():8.2f}")
    k_peak = int(np.argmax(nees_fix))
    print(f"fixed_anisotropic peak NEES {nees_fix[k_peak]:.1f} at t={k_peak * dt:.1f} s, "
          f"heading {np.degrees(x_true[k_peak, 2]):.0f} deg, ellipse mismatch {mis[k_peak]:.0f} deg")
    for name, xe in [("isotropic", x_iso), ("fixed_anisotropic", x_fix), ("heading_aware", x_awr)]:
        pe, _ = state_errors(x_true, xe)
        print(f"  RMS pos {name:<18s} {np.sqrt(np.mean(pe ** 2)):.4f} m")

    # ---- figure ----
    t_hist = np.arange(n_steps + 1) * dt
    S = args.ellipse_scale
    fig = plt.figure(figsize=(10, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], left=0.09, right=0.93, top=0.90, bottom=0.24, wspace=0.28)
    ax_p = fig.add_subplot(gs[0])
    ax_n = fig.add_subplot(gs[1])

    ax_p.plot(x_true[:, 0], x_true[:, 1], color="0.6", lw=1.2, ls="--", label="true path")
    trail, = ax_p.plot([], [], color="black", lw=1.5, label="driven so far")
    robot, = ax_p.plot([], [], "ko", ms=6, zorder=6)
    head, = ax_p.plot([], [], "k-", lw=2, zorder=6)
    pad = 0.45
    ax_p.set_xlim(x_true[:, 0].min() - pad, x_true[:, 0].max() + pad)
    ax_p.set_ylim(x_true[:, 1].min() - pad, x_true[:, 1].max() + pad)
    ax_p.set_aspect("equal")
    ax_p.set_xlabel("p_x (m)")
    ax_p.set_ylabel("p_y (m)")
    ax_p.set_title(f"Process-noise ellipses at the robot (2 sigma, x{S:g})", fontsize=10)

    def mk(color, ls, lw, label):
        e = Ellipse((0, 0), 0, 0, fill=False, ec=color, ls=ls, lw=lw, label=label, zorder=5)
        ax_p.add_patch(e)
        return e
    e_true = mk("green", "-", 2.5, "true slip (true heading)")
    e_iso = mk(COLORS["isotropic"], "-", 1.8, "isotropic")
    e_fix = mk(COLORS["fixed_anisotropic"], "-", 1.8, "fixed_anisotropic")
    e_awr = mk(COLORS["heading_aware"], "--", 1.8, "heading_aware")
    ax_p.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, framealpha=0.9)

    for name, y in [("isotropic", nees_iso), ("fixed_anisotropic", nees_fix), ("heading_aware", nees_awr)]:
        ax_n.plot(t_hist, y, color=COLORS[name], lw=0.6, alpha=0.15)
    lines = {n: ax_n.plot([], [], color=COLORS[n], lw=2, label=n)[0] for n in COLORS}
    ax_n.axhline(3.0, color="black", lw=1, label="expected NEES (3 DoF)")
    ax_n.axhspan(lo, hi, color="black", alpha=0.12, label=f"95% band, {args.n_mc_trials}-trial avg")
    ax_n.set_yscale("log")
    ax_n.set_xlim(0, t_hist[-1])
    ymax = max(nees_iso.max(), nees_fix.max(), nees_awr.max())
    ax_n.set_ylim(1.5, ymax * 1.6)
    ax_n.set_xlabel("Time (s)")
    ax_n.set_ylabel(f"Monte Carlo avg. NEES ({args.n_mc_trials} trials)")
    ax_n.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax_m = ax_n.twinx()
    ax_m.plot(t_hist, mis, color="tab:orange", lw=1, ls=":", alpha=0.7)
    mline, = ax_m.plot([], [], color="tab:orange", lw=1.8, ls=":")
    ax_m.set_ylim(0, 270)
    ax_m.set_yticks([0, 45, 90])
    ax_m.set_ylabel("fixed-ellipse misalignment (deg)", color="tab:orange", fontsize=8, loc="bottom")
    ax_m.tick_params(axis="y", colors="tab:orange", labelsize=8)
    vline = ax_n.axvline(0, color="k", lw=0.8, alpha=0.5)
    status = ax_n.text(0.98, 0.04, "", transform=ax_n.transAxes, ha="right", va="bottom", fontsize=8,
                       bbox=dict(fc="white", ec="0.7", alpha=0.9))

    fig.suptitle("Friction-anisotropic crawl: does Q's shape track the heading?", fontsize=12)
    fig.text(0.5, 0.035,
             "Fixed Q keeps the right shape but stops matching the pad as the robot turns (worst near 90 deg, ellipse period pi);\n"
             "isotropic Q ignores direction (mildly overconfident); heading-aware Q rotates with the heading and stays consistent.",
             ha="center", va="center", fontsize=8.5)

    def update(k):
        c = x_true[k, 0:2]
        trail.set_data(x_true[:k + 1, 0], x_true[:k + 1, 1])
        robot.set_data([c[0]], [c[1]])
        head.set_data([c[0], c[0] + 0.12 * np.cos(x_true[k, 2])], [c[1], c[1] + 0.12 * np.sin(x_true[k, 2])])
        # Q used by the predict step leaving tick k: heading_aware uses its own pre-propagation estimate x[2]
        for e, Q in [(e_true, anisotropic_Q_pos(x_true[k, 2], sg, ss, dt)),
                     (e_iso, isotropic_Q_pos(sg, ss, dt)),
                     (e_fix, anisotropic_Q_pos(theta_ref, sg, ss, dt)),
                     (e_awr, anisotropic_Q_pos(x_awr[k, 2], sg, ss, dt))]:
            e.set_center(c)
            _, w, h, a = q_ellipse(Q, c, S)
            e.set_width(w); e.set_height(h); e.set_angle(a)
        for n, y in [("isotropic", nees_iso), ("fixed_anisotropic", nees_fix), ("heading_aware", nees_awr)]:
            lines[n].set_data(t_hist[:k + 1], y[:k + 1])
        mline.set_data(t_hist[:k + 1], mis[:k + 1])
        vline.set_xdata([t_hist[k]])
        status.set_text(f"t = {t_hist[k]:4.1f} s   heading {np.degrees(x_true[k, 2]):5.1f} deg   "
                        f"mismatch {mis[k]:4.1f} deg\nNEES  iso {nees_iso[k]:5.1f} | fixed {nees_fix[k]:5.1f} | aware {nees_awr[k]:4.1f}")
        return []

    frames = list(range(0, n_steps + 1, args.stride))
    if frames[-1] != n_steps:
        frames.append(n_steps)
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    t0 = time.time()
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {args.out}: {len(frames)} frames, render {time.time() - t0:.0f} s")
    if args.frames_png:
        for k in [0, n_steps // 4, n_steps // 2, n_steps]:
            update(k)
            fig.savefig(f"{args.frames_png}_{k}.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
