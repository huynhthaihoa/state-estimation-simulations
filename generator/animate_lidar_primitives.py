'''
Animated version of lidar_primitives.py experiment 2, one scan pair: the same
two LiDAR scans of the corridor with pillars, the same initial guess (0.5 m
and 10 deg off the truth), registered by the three methods, one Gauss-Newton
iteration per frame.

Five panels:
  1-3. (left column) top-down view in world coordinates, one per method: the
       target scan (grey) and the source scan moved by the method's current
       estimate (colored; for the primitive method, the plane and cylinder
       inliers it actually uses are drawn darker);
  4.   translation error vs. iteration for the three methods (log scale);
  5.   zoom on the pillar nearest the target sensor, all three methods'
       source points overlaid on the target scan: at corridor scale a 0.3 m
       error is hard to see, at pillar scale it is not.

Message (doc docs/frontend/lidar_primitives.md section 4): from this start,
point-to-point ICP stalls about 0.3 m and 4.6 deg off, because its nearest
neighbors are not the same surface points; point-to-plane converges in 12
iterations to ~0.4 mm; the primitive method in 4 iterations to ~0.2 mm.

Pair 1, start 2 of run_basin_experiment's draws at level (0.5 m, 10 deg),
seed 0: chosen (by running all 20 pair/start combinations at that level for
the first 4 pairs) as a start where point-to-point visibly fails while the
other two succeed, which is what happens in 78% of the starts at that level.
Everything drawn is the simulation's own make_scan_pair / register output.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from matplotlib.gridspec import GridSpec


from state_estimation_simulations.use_numpy.lidar_primitives import (METHODS, ExtractionConfig, LidarConfig, RegistrationConfig, corridor_scene,
                              inlier_threshold, make_scan_pair, pose_error, random_perturbation, register, se3_exp)

LABELS = {"point_to_point": "point-to-point ICP", "point_to_plane": "point-to-plane ICP",
          "primitive": "primitives (planes + cylinders)"}
COLORS = {"point_to_point": "#d1495b", "point_to_plane": "#2e86de", "primitive": "#2a9d8f"}


def run(seed, pair, start, level, noise):
    """Scan pair, initial guess and every method's per-iteration estimates,
    drawn exactly as run_basin_experiment draws them."""
    lcfg = LidarConfig(range_noise_std=noise)
    ecfg = ExtractionConfig(inlier_dist=inlier_threshold(noise))
    src, tgt, T_true, T_tgt = make_scan_pair(corridor_scene(pillars=True), [seed, pair], lcfg, ecfg)
    rng = np.random.default_rng([seed, pair, 2])           # level index 2 = (0.5 m, 10 deg)
    for _ in range(start + 1):
        xi = random_perturbation(rng, level[0], np.deg2rad(level[1]))
    T0 = T_true @ se3_exp(xi)
    hist = {}
    for m in METHODS:
        h = [T0]
        register(m, T0, src, tgt, RegistrationConfig(), history=h)
        hist[m] = h
    return src, tgt, T_true, T_tgt, hist


def nearest_pillar(T_tgt):
    """World xy of the pillar axis nearest the target sensor."""
    bases = np.array([cyl.base[:2] for cyl in corridor_scene(pillars=True).cylinders])
    return bases[np.argmin(np.linalg.norm(bases - T_tgt[:2, 3], axis=1))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair", type=int, default=1)
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--step", type=int, default=3, help="draw every step-th point")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--out", type=str, default="assets/lidar_primitives.gif")
    args = parser.parse_args()

    src, tgt, T_true, T_tgt, hist = run(args.seed, args.pair, args.start, (0.5, 10.0), args.noise)
    errs = {m: np.array([pose_error(T, T_true) for T in h]) for m, h in hist.items()}
    for m in METHODS:
        print(f"{m}: {len(hist[m]) - 1} iterations, final {1e3 * errs[m][-1, 0]:.1f} mm, "
              f"{np.rad2deg(errs[m][-1, 1]):.2f} deg")

    def world(T_rel, P):
        T = T_tgt @ T_rel
        return P[::args.step] @ T[:3, :3].T + T[:3, 3]

    tgt_w = world(np.eye(4), tgt.points)
    used = np.zeros(len(src.points), bool)
    for prim in src.planes + src.cylinders:
        used[prim["inliers"]] = True
    used = used[::args.step]
    c = T_tgt[:2, 3]
    n_frames = max(len(h) for h in hist.values())
    frames = list(range(min(n_frames, 21))) + list(range(25, n_frames, 5)) + [n_frames - 1]
    frames = sorted(set(frames))

    fig = plt.figure(figsize=(15, 9.6))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[2.1, 1], hspace=0.55, wspace=0.18)
    scan_axes = [fig.add_subplot(gs[r, 0]) for r in range(3)]
    ax_e = fig.add_subplot(gs[0:2, 1])
    ax_z = fig.add_subplot(gs[2, 1])
    pz = nearest_pillar(T_tgt)
    artists = {}
    for ax, m in zip(scan_axes, METHODS):
        ax.scatter(tgt_w[:, 0], tgt_w[:, 1], s=1, c="#9a9a9a", lw=0)
        src_w = world(hist[m][0], src.points)
        sc = ax.scatter(src_w[:, 0], src_w[:, 1], s=1.5, c=COLORS[m], lw=0, alpha=0.55)
        sc_used = (ax.scatter(src_w[used, 0], src_w[used, 1], s=2, c="#14574f", lw=0)
                   if m == "primitive" else None)
        ax.plot(*c, marker="^", color="k", ms=7)
        ax.add_patch(plt.Rectangle(pz - 0.8, 1.6, 1.6, fill=False, ls=":", lw=1.0, color="k"))
        ax.set(xlim=(c[0] - 14, c[0] + 14), ylim=(-3.8, 3.8), aspect="equal", ylabel="y (m)")
        title = ax.set_title("", fontsize=10.5)
        artists[m] = (sc, sc_used, title)
    scan_axes[-1].set_xlabel("x (m)")
    scan_axes[0].annotate("wall", (c[0] - 11, 3.0), (c[0] - 11, 3.55), fontsize=8.5, ha="center", va="bottom",
                          arrowprops=dict(arrowstyle="-", lw=0.6))
    scan_axes[0].annotate("pillar (zoomed at right)", pz + [0.3, 0.3], pz + [2.5, 1.2], fontsize=8.5,
                          arrowprops=dict(arrowstyle="-", lw=0.6))
    ax_z.scatter(tgt_w[:, 0], tgt_w[:, 1], s=4, c="#9a9a9a", lw=0, label="target scan")
    zoom = {m: ax_z.scatter([], [], s=6, c=COLORS[m], lw=0, label=LABELS[m]) for m in METHODS}
    ax_z.add_patch(plt.Circle(pz, 0.25, fill=False, ls="--", lw=0.8, color="k"))
    ax_z.set(xlim=(pz[0] - 0.8, pz[0] + 0.8), ylim=(pz[1] - 0.8, pz[1] + 0.8), aspect="equal", xlabel="x (m)",
             title="Zoom: nearest pillar (dashed: true outline)")
    ax_z.title.set_fontsize(10.5)
    for m in METHODS:
        ax_e.plot([], [], color=COLORS[m], lw=1.8, label=LABELS[m])
    lines = {m: ax_e.plot([], [], color=COLORS[m], lw=1.8)[0] for m in METHODS}
    ax_e.set(yscale="log", xlim=(0, n_frames - 1), ylim=(1e-4, 2.0), xlabel="Gauss-Newton iteration",
             ylabel="translation error (m)", title="Error vs. iteration")
    ax_e.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax_e.grid(alpha=0.25)
    fig.suptitle("Registering two LiDAR scans of a pillared corridor, same start (0.5 m, 10 deg off)",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.945, "Seen from above. Grey: target scan; colored: source scan moved by the current estimate; "
             "triangle: target sensor. lidar_primitives.py, seed 0, pair 1.", fontsize=9.5, color="#555555")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.07)

    def draw(k):
        out = []
        for m in METHODS:
            i = min(k, len(hist[m]) - 1)
            sc, sc_used, title = artists[m]
            src_w = world(hist[m][i], src.points)
            sc.set_offsets(src_w[:, :2])
            out.append(sc)
            if sc_used is not None:
                sc_used.set_offsets(src_w[used, :2])
                out.append(sc_used)
            et, er = errs[m][i]
            done = " (done)" if i == len(hist[m]) - 1 else ""
            title.set_text(f"{LABELS[m]}: iteration {i}{done}\nerror {1e3 * et:.1f} mm, {np.rad2deg(er):.2f} deg")
            lines[m].set_data(np.arange(i + 1), np.maximum(errs[m][:i + 1, 0], 1e-4))
            zoom[m].set_offsets(src_w[:, :2])
            out.extend([title, lines[m], zoom[m]])
        return out

    anim = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps))
    print(f"saved {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
