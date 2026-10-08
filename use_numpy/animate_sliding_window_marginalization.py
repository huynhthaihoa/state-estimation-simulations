'''
Animated version of sliding_window_marginalization.py's streaming run: an
odometry-only square path of 64 poses streams in one pose at a time while a
sliding window keeps at most 10 poses live and folds the oldest into a prior
(docs/optimization/marginalization.md, sections 8 and 9).

Panels:
  1. x-y path: faint ground truth, marginalized poses frozen in gray (the
     estimate they had when dropped), the live window highlighted, and the pose
     being marginalized at this step flashed with a red ring;
  2. system size (number of unknowns, 6 per pose) against poses streamed: a full
     batch re-solve of the whole graph (6n, analytic, not re-run) against the
     window (capped at 6 x window = 60), drawn progressively;
  3. the window's information matrix at the current step, block by block
     (6x6 blocks): block-tridiagonal chain plus the prior block on the oldest
     pose that earlier marginalizations left behind.

Message: the solve's memory/system size is bounded by the window, not just
faster. Accuracy is NOT claimed to differ: with no loop closures both solvers
return the dead-reckoned trajectory (doc section 9), so they tie by construction.

The simulation function returns only the final result, so its ~25-line loop is
copied here (calling the module's own marginalize_oldest and
solve_to_convergence) to record per-step state; the copy's x_final and max_dof
are checked against run_sliding_window_pose_graph's.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap

from sliding_window_marginalization import (
    assemble_window_system, solve_to_convergence, marginalize_oldest,
    run_sliding_window_pose_graph, generate_ground_truth_trajectory, simulate_noisy_edges)


def record_sliding_window(gt_poses, odometry_constraints, info_matrix, window_size,
                          anchor_weight, gn_tol, gn_max_iters):
    """Copy of run_sliding_window_pose_graph's loop that also records, after
    each step's solve: the window poses, their global indices, the dropped
    global index (or None), the system size, and the block-occupancy pattern
    of the window's information matrix.
    Returns:
        x_final, max_dof (as the original), steps (list of dicts)
    """
    n_poses = len(gt_poses)
    x_final = [None] * n_poses
    x_window, global_idx, edges_window = [gt_poses[0]], [0], []
    prior_ref, prior_info = gt_poses[0], anchor_weight * np.eye(6)
    max_dof = 6
    steps = [dict(k=0, xw=list(x_window), gidx=list(global_idx), dropped=None, dof=6,
                  pattern=np.ones((1, 1), bool))]

    for k in range(1, n_poses):
        idx_i, idx_j, Z_ij = odometry_constraints[k - 1]
        dropped = None
        if len(x_window) >= window_size:
            new_prior_ref, new_prior_info, x_dropped = marginalize_oldest(
                x_window, prior_ref, prior_info, edges_window, info_matrix)
            dropped = global_idx[0]
            x_final[dropped] = x_dropped
            prior_ref, prior_info = new_prior_ref, new_prior_info
            x_window = x_window[1:]
            global_idx = global_idx[1:]
            edges_window = [(i - 1, j - 1, Z) for (i, j, Z) in edges_window[1:]]
        x_window.append(x_window[-1] @ Z_ij)
        global_idx.append(k)
        edges_window.append((len(x_window) - 2, len(x_window) - 1, Z_ij))
        x_window, dof = solve_to_convergence(x_window, prior_ref, prior_info, edges_window,
                                              info_matrix, gn_tol, gn_max_iters)
        max_dof = max(max_dof, dof)

        H, _ = assemble_window_system(x_window, prior_ref, prior_info, edges_window, info_matrix)
        W = len(x_window)
        blocks = np.abs(H).reshape(W, 6, W, 6).max(axis=(1, 3))
        steps.append(dict(k=k, xw=list(x_window), gidx=list(global_idx), dropped=dropped, dof=dof,
                          pattern=blocks > 1e-9, x_dropped=x_final[dropped] if dropped is not None else None))

    for local_i, g_i in enumerate(global_idx):
        x_final[g_i] = x_window[local_i]
    return x_final, max_dof, steps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes-per-side", type=int, default=16, help="64 poses total = 4x this")
    p.add_argument("--side-length", type=float, default=2.0)
    p.add_argument("--window-size", type=int, default=10)
    p.add_argument("--pos-noise-std", type=float, default=0.05)
    p.add_argument("--rot-noise-std", type=float, default=0.01)
    p.add_argument("--anchor-weight", type=float, default=1e6)
    p.add_argument("--gn-tol", type=float, default=1e-6)
    p.add_argument("--gn-max-iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hold", type=int, default=12, help="frames to hold on the last step")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--dpi", type=int, default=80)
    p.add_argument("--out", type=str, default="sliding_window_marginalization.gif")
    p.add_argument("--png-frames", type=int, nargs="*", default=[], help="also save these frame indices as PNG next to --out")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    info = np.eye(6)
    gt = generate_ground_truth_trajectory(a.side_length, nodes_per_side=a.nodes_per_side)
    n = len(gt)
    odo, _ = simulate_noisy_edges(gt, a.pos_noise_std, a.rot_noise_std, loop_noise_scale=1.0, rng=rng)

    x_ref, dof_ref = run_sliding_window_pose_graph(gt, odo, info, a.window_size, a.anchor_weight, a.gn_tol, a.gn_max_iters)
    x_fin, max_dof, steps = record_sliding_window(gt, odo, info, a.window_size, a.anchor_weight, a.gn_tol, a.gn_max_iters)
    same = all(np.array_equal(u, v) for u, v in zip(x_ref, x_fin)) and dof_ref == max_dof
    print(f"copied loop == run_sliding_window_pose_graph: x_final identical and max_dof equal: {same} "
          f"(max_dof {max_dof} vs {dof_ref}, expected 6*{a.window_size}={6 * a.window_size})")
    print(f"poses: {n}; final window dof {steps[-1]['dof']}; batch dof at n={n} (analytic 6n): {6 * n}")

    gt_xy = np.array([T[:2, 3] for T in gt])
    xy = lambda poses: np.array([T[:2, 3] for T in poses])
    # latest estimate of every pose, frozen once dropped
    est = np.full((n, 2), np.nan)
    frames = list(range(n + a.hold))  # indices >= n hold on the last step

    ns = np.arange(1, n + 1)
    dof_batch = 6 * ns
    dof_win = np.array([s["dof"] for s in steps])

    fig = plt.figure(figsize=(10, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], width_ratios=[1.5, 1], left=0.07, right=0.97,
                          top=0.94, bottom=0.16, hspace=0.34, wspace=0.28)
    ax_p = fig.add_subplot(gs[0, :])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_h = fig.add_subplot(gs[1, 1])

    pad = 0.25
    allxy = np.vstack([gt_xy, xy(x_fin)])
    ax_p.set(xlim=(allxy[:, 0].min() - pad, allxy[:, 0].max() + pad), ylim=(allxy[:, 1].min() - pad, allxy[:, 1].max() + pad),
             aspect="equal", xlabel="x (m)", ylabel="y (m)")
    ax_p.set_title("Odometry-only path: estimate frozen once marginalized", fontsize=10)
    ax_p.plot(gt_xy[:, 0], gt_xy[:, 1], "-", color="0.8", lw=3, zorder=1, label="ground truth")
    l_frozen, = ax_p.plot([], [], "o-", color="0.5", ms=4, lw=1, zorder=2, label="marginalized (frozen)")
    l_win, = ax_p.plot([], [], "o-", color="tab:blue", ms=6, lw=2, zorder=3, label="live window")
    l_flash, = ax_p.plot([], [], "o", mfc="none", mec="tab:red", mew=2.5, ms=16, zorder=4, label="marginalized this step")
    l_new, = ax_p.plot([], [], "o", color="tab:orange", ms=8, zorder=5, label="newest pose")
    ax_p.legend(loc="center left", fontsize=8, framealpha=0.9, bbox_to_anchor=(1.02, 0.5))
    txt = ax_p.text(1.02, 0.97, "", transform=ax_p.transAxes, va="top", fontsize=10)

    ax_d.set(xlim=(0, n + 1), ylim=(0, 6 * n * 1.05), xlabel="poses streamed", ylabel="unknowns in the solve (6 per pose)")
    ax_d.set_title("System size of each solve", fontsize=10)
    ax_d.grid(alpha=0.3)
    ax_d.axhline(6 * a.window_size, color="tab:blue", ls=":", lw=1)
    l_b, = ax_d.plot([], [], color="tab:red", lw=2, label="full batch re-solve: 6n (analytic)")
    l_w, = ax_d.plot([], [], color="tab:blue", lw=2.5, label=f"sliding window: capped at {6 * a.window_size}")
    ax_d.legend(loc="upper left", fontsize=8)
    t_b = ax_d.text(0.98, 0.45, "", transform=ax_d.transAxes, ha="right", fontsize=9, color="tab:red")
    t_w = ax_d.text(0.98, 0.19, "", transform=ax_d.transAxes, ha="right", fontsize=9, color="tab:blue")

    cmap = ListedColormap(["white", "tab:blue", "tab:orange"])
    W = a.window_size
    im = ax_h.imshow(np.zeros((W, W)), cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax_h.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax_h.set_yticks(np.arange(-0.5, W, 1), minor=True)
    ax_h.grid(which="minor", color="0.85", lw=0.5)
    ax_h.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    ax_h.set_title("Window information matrix\n(6x6 blocks; orange = prior on oldest)", fontsize=9)

    fig.text(0.5, 0.025,
             "Memory is bounded by the window (60 unknowns vs 6n for batch), not just faster. Accuracy ties by construction here:\n"
             "no loop closures, so both solvers return the dead-reckoned path (doc section 9) - the window is not claimed more accurate.",
             ha="center", va="bottom", fontsize=9)
    ax_d.xaxis.labelpad = 2

    def draw(fi):
        s = steps[min(fi, n - 1)]
        k = s["k"]
        if fi < n:
            if s["dropped"] is not None:
                est[s["dropped"]] = s["x_dropped"][:2, 3]
        w = xy(s["xw"])
        fz = est[: s["gidx"][0]] if s["gidx"][0] > 0 else np.empty((0, 2))
        l_frozen.set_data(fz[:, 0], fz[:, 1])
        l_win.set_data(w[:, 0], w[:, 1])
        l_new.set_data(w[-1:, 0], w[-1:, 1])
        if s["dropped"] is not None and fi < n:
            d = est[s["dropped"]]
            l_flash.set_data([d[0]], [d[1]])
        else:
            l_flash.set_data([], [])
        txt.set_text(f"pose {k + 1} of {n} streamed\nwindow = poses {s['gidx'][0]}..{s['gidx'][-1]}")
        l_b.set_data(ns[: k + 1], dof_batch[: k + 1])
        l_w.set_data(ns[: k + 1], dof_win[: k + 1])
        t_b.set_text(f"batch: {dof_batch[k]}")
        t_w.set_text(f"window: {dof_win[k]}")
        img = np.zeros((W, W))
        pat = s["pattern"]
        m = pat.shape[0]
        img[:m, :m] = pat
        img[0, 0] = 2 if m > 1 or s["gidx"][0] > 0 else 1  # prior block (anchor at start)
        im.set_data(img)
        return []

    # est must be filled cumulatively when frames are rendered out of order (PNG saves)
    def prefill(fi):
        est[:] = np.nan
        for t in steps[: min(fi, n - 1) + 1]:
            if t["dropped"] is not None:
                est[t["dropped"]] = t["x_dropped"][:2, 3]

    def update(fi):
        prefill(fi)
        return draw(fi)

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 // a.fps, blit=False)
    anim.save(a.out, writer=animation.PillowWriter(fps=a.fps), dpi=a.dpi)
    print(f"saved {a.out}: {len(frames)} frames")
    base = a.out.rsplit(".", 1)[0]
    for fi in a.png_frames:
        update(frames[fi])
        fig.savefig(f"{base}_frame{fi:03d}.png", dpi=a.dpi)


if __name__ == "__main__":
    main()
