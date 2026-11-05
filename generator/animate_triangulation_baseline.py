'''
Animated version of the static figure triangulation_baseline_concept
(make_figures.py, docs/frontend/triangulation_pnp.md Section 2 "tiny example"):
two rectified cameras (parallel optical axes, offset sideways) triangulate one
point Z = 10 m ahead with use_numpy.bundle_adjustment_advanced.triangulate_landmark,
focal length f = 800 px, 1 px disparity noise sigma. The baseline B (distance
between the two camera centers) sweeps from 1 m down to 0.1 m, log-spaced,
then the last frame holds.

Two panels, drawn progressively:
  1. left: top view (x sideways, depth z ahead; the two axes are drawn at
     different scales). The two cameras (squares), their rays to the true
     point (star) and the cloud of N noisy triangulations (dots). The cloud
     stretches along the viewing direction as B shrinks;
  2. right: the measured depth std of the N triangulations against B (dots,
     added one baseline at a time) next to the doc's first-order formula
     Z^2 sigma / (f B) (line), with a readout of B, measured std, formula std
     and the 5-95% range of the depth.

Message: depth error grows as 1/B, so a 10 times shorter baseline gives about
10 times the depth error, and the spread is lopsided (a longer far-side tail).

Data: every dot is a triangulate_landmark return value, called on the camera
poses and noisy pixels exactly as in the static figure (no copied solver). The
pixel noise is N(0, (sigma/sqrt(2))^2) per coordinate per image, so the
disparity noise is sigma. Common random numbers: the N x 2 x 2 noise array is
drawn ONCE and reused at every baseline, so the cloud morphs smoothly instead
of re-rolling. The draw layout matches the static figure's per-solve draws,
and the generator asserts that the B = 1 m cloud equals one made with the
static figure's loop (same seed), so the first frame is the static figure's
B = 1 m cloud. The formula is the doc's exact expression Z^2 sigma / (f B).

Illustrative: the choice of 40 log-spaced baselines between 1 m and 0.1 m and
the hold at the end; the figure is a top view with x and z at different
scales (points outside the drawn window are counted in the readout, not
drawn). Seed 0 by default (the static figure's seed). N = 2000 solves
(the static figure's count).

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from state_estimation_simulations.use_numpy.bundle_adjustment_advanced import triangulate_landmark

K = (800.0, 800.0, 320.0, 240.0)    # fx, fy, cx, cy (the static figure's intrinsics)
F, Z, SIGMA = K[0], 10.0, 1.0       # focal length (px), true depth (m), disparity noise std (px)
XLIM, ZLIM = (-0.8, 0.8), (-0.5, 15.0)


def cloud(B, noise):
    """Triangulate every noise draw for baseline B; returns (N, 3) points. Cameras at x = -B/2, +B/2, looking along +z."""
    T_obs = []
    for cx in (-B / 2, B / 2):          # camera-to-world poses
        T = np.eye(4)
        T[0, 3] = cx
        T_obs.append(T)
    z_true = [np.array([F * (0.0 - T[0, 3]) / Z + K[2], K[3]]) for T in T_obs]
    return np.array([triangulate_landmark(T_obs, [z_true[0] + e[0], z_true[1] + e[1]], K) for e in noise])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (the static figure's seed)")
    parser.add_argument("--n", type=int, default=2000, help="Noisy triangulations per baseline (the static figure's count)")
    parser.add_argument("--frames", type=int, default=40, help="Baselines in the sweep (log-spaced, 1 m to 0.1 m)")
    parser.add_argument("--hold", type=int, default=15, help="Extra frames holding the last baseline")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.add_argument("--out", type=str, required=True, help="Output GIF path")
    args = parser.parse_args()

    noise = np.random.default_rng(args.seed).normal(0.0, SIGMA / np.sqrt(2), (args.n, 2, 2))   # common random numbers
    Bs = np.logspace(0.0, -1.0, args.frames)
    clouds = [cloud(B, noise) for B in Bs]

    # the B = 1 m cloud must equal the static figure's loop (same seed, per-solve draws)
    rng = np.random.default_rng(args.seed)
    T_obs = [np.eye(4), np.eye(4)]
    T_obs[0][0, 3], T_obs[1][0, 3] = -0.5, 0.5
    z_true = [np.array([F * (0.0 - T[0, 3]) / Z + K[2], K[3]]) for T in T_obs]
    ref = np.array([triangulate_landmark(T_obs, [zt + rng.normal(0.0, SIGMA / np.sqrt(2), 2) for zt in z_true], K)
                    for _ in range(args.n)])
    assert np.allclose(ref, clouds[0]), "B = 1 m cloud differs from the static figure's loop"

    std = np.array([c[:, 2].std() for c in clouds])
    formula = Z ** 2 * SIGMA / (F * Bs)
    p5, p95 = (np.array([np.percentile(c[:, 2], q) for c in clouds]) for q in (5, 95))
    off = np.array([np.sum((c[:, 2] < ZLIM[0]) | (c[:, 2] > ZLIM[1]) | (c[:, 0] < XLIM[0]) | (c[:, 0] > XLIM[1])) for c in clouds])
    print(f"seed {args.seed}, N = {args.n}, f = {F:g} px, Z = {Z:g} m, sigma = {SIGMA:g} px, {args.frames} baselines + {args.hold} hold frames")
    for B in (1.0, 0.5, 0.1):
        i = int(np.argmin(np.abs(Bs - B)))
        print(f"  B = {Bs[i]:.3f} m: depth std {std[i]:.4f} m, formula {formula[i]:.4f} m, 5-95% [{p5[i]:.2f}, {p95[i]:.2f}] m, "
              f"near side {Z - p5[i]:.2f} m, far side {p95[i] - Z:.2f} m, off-window {off[i]}")
    print(f"  std ratio B = 0.1 m over B = 1 m: {std[-1] / std[0]:.2f} (formula 10)")

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.25, wspace=0.18)
    fig.suptitle("Shorter baseline, longer depth error: two cameras triangulating a point 10 m away",
                 fontsize=12, fontweight="bold")

    ax_t.set(xlim=XLIM, ylim=ZLIM, xlabel="x, sideways (m)", ylabel="depth z (m)",
             title="Top view (x and z at different scales)")
    rays = [ax_t.plot([], [], color="tab:orange", lw=1.2)[0] for _ in range(2)]
    cams = ax_t.scatter([], [], marker="s", s=60, color="k", zorder=4, label="two cameras (squares)")
    pts = ax_t.scatter([], [], s=5, color="tab:blue", alpha=0.35, lw=0, zorder=5)
    ax_t.plot([], [], "o", ms=3, color="tab:blue", label=f"{args.n} noisy triangulations")
    ax_t.scatter([0], [Z], marker="*", s=220, facecolor="none", edgecolor="k", lw=1.2, zorder=6, label=f"true point (Z = {Z:g} m)")
    ax_t.plot([], [], color="tab:orange", lw=1.2, label="rays to the true point")
    ax_t.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5, ncol=2, frameon=False)
    readout = ax_t.text(0.03, 0.97, "", transform=ax_t.transAxes, va="top", fontsize=8.5, family="monospace",
                        bbox=dict(fc="white", ec="0.7", alpha=0.9))

    Bg = np.logspace(-1.0, 0.0, 100)
    ax_s.plot(Bg, Z ** 2 * SIGMA / (F * Bg), color="tab:green", lw=1.8, label=r"formula $Z^2\sigma/(fB)$")
    ax_s.plot(Bs, std, color="tab:blue", lw=1, alpha=0.2)
    meas, = ax_s.plot([], [], "o", color="tab:blue", ms=5, label=f"measured depth std ({args.n} solves)")
    now, = ax_s.plot([], [], "o", mfc="none", mec="k", ms=11, mew=1.5, label="current frame")
    ax_s.set(xscale="log", yscale="log", xlim=(1.12, 0.09), ylim=(0.08, 1.8),
             xlabel="baseline B (m), shrinking to the right", ylabel="depth error (m)",
             title="Depth std of the triangulations against B")
    ax_s.set_xticks([1.0, 0.7, 0.5, 0.3, 0.2, 0.1], ["1", "0.7", "0.5", "0.3", "0.2", "0.1"])
    ax_s.set_yticks([0.1, 0.2, 0.5, 1.0], ["0.1", "0.2", "0.5", "1"])
    ax_s.minorticks_off()
    ax_s.grid(True, which="both", alpha=0.3)
    ax_s.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5, ncol=2, frameon=False)

    fig.text(0.5, 0.012,
             f"Same {args.n} pixel-noise draws at every B (seed {args.seed}, common random numbers), so the cloud morphs smoothly; "
             "1 px disparity noise, f = 800 px.\n"
             f"Depth std {std[0]:.2f} m at B = 1 m, {std[-1]:.2f} m at B = 0.1 m ({std[-1] / std[0]:.1f}x), tracking Z²σ/(fB). "
             f"At B = 0.1 m the cloud is lopsided: {Z - p5[-1]:.1f} m to the near side, {p95[-1] - Z:.1f} m to the far side (5-95%).",
             ha="center", fontsize=9)

    n_frames = args.frames + args.hold

    def update(frame):
        i = min(frame, args.frames - 1)
        B = Bs[i]
        for r, cx in zip(rays, (-B / 2, B / 2)):
            r.set_data([cx, 0.0], [0.0, Z])
        cams.set_offsets([[-B / 2, 0.0], [B / 2, 0.0]])
        pts.set_offsets(clouds[i][:, [0, 2]])
        meas.set_data(Bs[:i + 1], std[:i + 1])
        now.set_data([B], [std[i]])
        readout.set_text(f"B = {B:.3f} m\nmeasured std {std[i]:.3f} m\nformula     {formula[i]:.3f} m\n"
                         f"5-95%: {p5[i]:.2f}-{p95[i]:.2f} m\noutside window: {off[i]}")
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {n_frames} frames to {args.out}")


if __name__ == "__main__":
    main()
