'''
Animated version of pose_graph_robust.py, one run at its defaults (seed 0, 20%
false loop closures = 8 false of 48): a robot drives a 2 m square for three laps,
so the graph has 59 odometry edges, 40 true loop closures (same place, one lap
apart) and 8 false ones (random node pairs, random measurement). The poses start
from dead reckoning and Levenberg-Marquardt moves them, once with plain least
squares (l2) and once with a robust kernel (cauchy by default).

Frame k shows the poses after accepted step k of each run (k = 0 is the dead
reckoning start); the run that converges first is held at its final state while
the other one finishes. Three panels:
  1. left: top view of the l2 estimate (blue) against the true square (black) and
     the dead reckoning (gray dotted), with every loop-closure edge drawn between
     its two estimated nodes and colored by its CURRENT weight w(e) (green = 1,
     red = 0); false closures are dashed and thick, true ones thin;
  2. right: the same for the robust method;
  3. bottom: position RMSE over all nodes (m) against the iteration, both
     methods, log scale.

Message (docs/optimization/robust_pose_graph.md): under l2 the false closures
pull the map into a crumpled shape (every edge keeps weight 1); the robust kernel
drives their weights to ~0 within the first iterations and the map relaxes onto the
square. A true closure whose weight is low at the start (its residual is large
from the dead-reckoning guess) recovers as the map straightens.

Data source: only pose_graph_robust.py's own functions (build_scene, solve_scene
with record=True, kernel_threshold) at its defaults; every drawn pose is a pose
the solver returned after an accepted step, nothing is interpolated. The
iteration count of each run and its RMSE are printed and asserted against a
fresh solve without recording.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from state_estimation_simulations.use_numpy import pose_graph_robust as prg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    prg.add_arguments(parser)
    parser.add_argument("--robust-method", choices=("huber", "cauchy", "dcs"), default="cauchy",
                        help="Robust method shown next to l2 (cauchy has the lowest position RMSE on seed 0)")
    parser.add_argument("--hold", type=int, default=3, help="Frames per iteration")
    parser.add_argument("--fps", type=int, default=5, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=70, help="GIF resolution")
    parser.add_argument("--gif-out", type=str, required=True, help="Output GIF path")
    p = parser.parse_args()

    scene = prg.build_scene(prg, p, p.seed, p.single_outlier_frac)
    tau = prg.base_tau(p)
    methods = ["l2", p.robust_method]
    res = {m: prg.solve_scene(prg, scene, m, tau, p, record=True) for m in methods}
    gt, loops, is_false = scene["gt"], scene["loops"], scene["is_false"]
    n_lap = scene["nodes_per_lap"]

    hist = {m: res[m]["info"]["history"] for m in methods}
    n_it = {m: len(hist[m]["poses"]) - 1 for m in methods}
    n_frames_it = max(n_it.values()) + 1                  # iteration states 0..max
    pos_rmse = {m: [prg.pose_rmse(prg, gt, T)[0] for T in hist[m]["poses"]] for m in methods}
    for m in methods:      # the recorded final state is the one a plain solve returns
        assert np.isclose(pos_rmse[m][-1], res[m]["pos_rmse"]) and n_it[m] == res[m]["iterations"]
    print(f"seed {p.seed}: {len(gt)} nodes, {len(scene['true_loops'])} true + {len(scene['false_loops'])} false loop closures, tau={tau:.3f}")
    for m in methods:
        print(f"  {m}: {n_it[m]} iterations, position RMSE {pos_rmse[m][0]:.4f} m (start) -> {pos_rmse[m][-1]:.4f} m; "
              f"final mean weight of the false closures {np.mean(hist[m]['weights'][-1][is_false]):.3g}, of the true closures {np.mean(hist[m]['weights'][-1][~is_false]):.3g}; "
              f"true closures with weight < 0.5: {int(np.sum(hist[m]['weights'][0][~is_false] < 0.5))} of {int((~is_false).sum())} at the start, "
              f"{int(np.sum(hist[m]['weights'][-1][~is_false] < 0.5))} at the end; false closures with weight < 0.1: "
              f"{int(np.sum(hist[m]['weights'][0][is_false] < 0.1))} of {int(is_false.sum())} at the start")

    gt_xy = prg.positions(gt[:n_lap] + gt[:1])[:, 0:2]      # one lap, closed
    dr_xy = prg.positions(scene["T_dr"])[:, 0:2]
    cmap = plt.get_cmap("RdYlGn")
    norm = Normalize(0.0, 1.0)

    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.0], hspace=0.32, wspace=0.12, left=0.07, right=0.89, top=0.90, bottom=0.08)
    ax = {m: fig.add_subplot(gs[0, k]) for k, m in enumerate(methods)}
    ax_r = fig.add_subplot(gs[1, :])
    cax = fig.add_axes([0.915, 0.40, 0.012, 0.42])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="loop-edge weight w(e)")
    title = fig.suptitle("", fontsize=11)
    lim = (-0.9, 2.6)
    colors = {"l2": "tab:blue", p.robust_method: "tab:blue"}

    def draw(frame):
        k = min(frame // p.hold, n_frames_it - 1)
        for m in methods:
            a = ax[m]
            a.clear()
            kk = min(k, n_it[m])
            T = hist[m]["poses"][kk]
            w = hist[m]["weights"][kk]
            xy = prg.positions(T)[:, 0:2]
            a.plot(*gt_xy.T, color="black", lw=2.2, zorder=1)
            a.plot(*dr_xy.T, color="tab:gray", lw=0.8, ls=":", zorder=1)
            a.plot(*xy.T, color=colors[m], lw=1.3, marker="o", ms=2.5, zorder=3)
            for e, (i, j, _) in enumerate(loops):
                seg = xy[[i, j]]
                if is_false[e]:
                    a.plot(seg[:, 0], seg[:, 1], color=cmap(norm(w[e])), lw=2.2, ls="--", zorder=4)
                    a.plot(seg[:, 0], seg[:, 1], color="black", lw=0.6, ls="--", zorder=5)
                else:
                    a.plot(seg[:, 0], seg[:, 1], color=cmap(norm(w[e])), lw=1.0, zorder=2)
            a.set_xlim(*lim); a.set_ylim(*lim); a.set_aspect("equal")
            a.set_xlabel("x (m)")
            if m == methods[0]:
                a.set_ylabel("y (m)")
            else:
                a.set_yticklabels([])
            done = " (converged)" if kk == n_it[m] and kk < k + 1 and kk > 0 else ""
            a.set_title(f"{m}: iteration {kk}{done}\nposition RMSE {pos_rmse[m][kk]:.3f} m; mean weight of false closures {np.mean(w[is_false]):.3f}",
                        fontsize=9.5)
        ax_r.clear()
        for m, c in zip(methods, ("tab:red", "tab:blue")):
            kk = min(k, n_it[m])
            ax_r.plot(range(n_it[m] + 1), pos_rmse[m], color=c, lw=1, alpha=0.35)
            ax_r.plot(range(kk + 1), pos_rmse[m][:kk + 1], color=c, lw=2, marker="o", ms=3, label=m)
        ax_r.axhline(pos_rmse["l2"][0], color="tab:gray", ls=":", lw=1)
        ax_r.text(n_frames_it - 1, pos_rmse["l2"][0], "dead reckoning ", ha="right", va="bottom", fontsize=7, color="tab:gray")
        ax_r.set_yscale("log"); ax_r.set_xlim(-0.3, n_frames_it - 0.7)
        ax_r.set_xlabel("accepted Levenberg-Marquardt step"); ax_r.set_ylabel("position RMSE (m)")
        ax_r.legend(loc="upper right", fontsize=8); ax_r.grid(alpha=0.3, which="both")
        title.set_text(f"Pose graph, 3 laps, {len(scene['false_loops'])} false of {len(loops)} loop closures (dashed = false; black = truth, gray = dead reckoning)")

    n_frames = n_frames_it * p.hold + 2 * p.hold          # hold the last state
    anim = animation.FuncAnimation(fig, draw, frames=n_frames, interval=1000 // p.fps)
    anim.save(p.gif_out, writer=animation.PillowWriter(fps=p.fps), dpi=p.dpi)
    print(f"Saved {p.gif_out} ({n_frames} frames, {n_frames_it} iteration states)")


if __name__ == "__main__":
    main()
