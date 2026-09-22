'''
Gives docs/optimization/marginalization.md its first accompanying script --
its own §8 names this exact gap: "there is no accompanying script... a
reader wanting to exercise §4 for real would extend [pose_graph_incremental.
py] to marginalize its oldest node once a fixed window size is exceeded."
This is that extension, addressing `unified_phd_plan.md`'s "resource
constraints" row: "algorithm-level efficiency (sparse factor graphs,
hierarchical pose pruning) on commodity embedded hardware" is the core
commitment there (GAP9's <500kB target is explicitly Stretch, not core) --
this script demonstrates the specific algorithmic property that matters for
that claim: bounded, not merely fast, memory/compute as a trajectory grows
without bound.

A robot drives a growing chain of noisy odometry edges (no loop closure --
see below for why) and two solvers process the same stream:

  - run_full_batch_growing: re-solves the *entire* graph from scratch every
    time a new node arrives -- the unbounded baseline. Problem size (and
    therefore the dense normal-equations system that must be held in memory
    to solve it) grows every step, forever.

  - run_sliding_window_pose_graph: keeps at most `window_size` poses in
    memory at once. Once a new node would exceed that, the oldest pose is
    permanently marginalized out (marginalization.md §4's Schur complement,
    $\\Lambda_b' = \\Lambda_{bb} - \\Lambda_{ba}\\Lambda_{aa}^{-1}\\Lambda_{ab}$)
    and folded into a new unary prior factor over whatever it was still
    connected to. The marginal is represented as a genuine prior factor (a
    frozen reference pose + information matrix, re-linearized against the
    *current* estimate every solve, exactly like an ordinary edge) rather
    than a frozen linear term -- the textbook-correct way real systems
    (GTSAM included) represent a marginal, and the only representation that
    stays correct as the surviving poses keep moving across later windows.

Deliberately a pure odometry chain, no loop closures: the oldest pose in a
chain window is connected to exactly one surviving neighbor, so
marginalizing it produces a *provably unary* prior (verified directly by a
test) -- no marginalization.md §5-style fill-in to manage. That's a real,
different problem (already covered by bayes_tree.md/isam2_optimization.md,
and by pose_graph_incremental.py's own loop-closure handling); this script
isolates the memory-*bounding* property alone, uncomplicated by fill-in
management.

Deliberately does *not* implement First-Estimate Jacobians (FEJ,
marginalization.md §6): the surviving poses' Jacobians are re-evaluated at
their newest estimate on every solve, including the prior factor's own,
which is the textbook source of the mild overconfidence FEJ exists to fix --
flagged here as a known, unaddressed simplification, the same way
pose_graph_incremental.py flags what it doesn't implement relative to iSAM2.

A note on measuring "bounded memory" honestly: this script's own output
(final pose estimates for every node, for plotting/error-reporting) is
necessarily O(trajectory length) for *both* solvers alike, regardless of
which algorithm produced it -- a real streaming deployment wouldn't
necessarily retain that history at all. Comparing raw whole-process peak
memory (via utils.measure_performance, as every other script in this repo
does) would bury the actual algorithmic signal under that shared, benchmark-
only bookkeeping cost. So both run_* functions additionally track and return
`max_dof`, the size of the largest dense information matrix ever actually
assembled and solved during the run -- for full-batch this grows with
trajectory length without bound; for the sliding-window approach it's capped
at `6 * window_size` forever. That's the number this script's docs and plots
treat as the headline resource-constraint metric; measure_performance's
whole-call timing/memory is still reported alongside it for consistency with
the rest of this repo, with this same caveat applying to its memory column.

Poses are (4,4) numpy homogeneous transforms; tangent vectors follow this
codebase's [vx,vy,vz,wx,wy,wz] convention throughout.
'''

import argparse
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import measure_performance
from lie_utils import se3_exp, se3_log, se3_inv, compute_se3_inv_right_jacobian
from pose_graph import (generate_ground_truth_trajectory, simulate_noisy_edges, run_dead_reckoning,
                         run_pose_graph_optimization, pose_errors)
from pose_graph_incremental import linearize_edge


def assemble_window_system(x_window, prior_ref, prior_info, edges_window, info_matrix):
    """Dense Gauss-Newton normal-equation system (H, g) for the current
    window: an optional unary prior factor on the oldest pose (from a
    previous marginalization, or the initial gauge-fixing anchor) plus every
    odometry edge inside the window, all linearized at x_window. Same
    accumulation pattern as pose_graph.run_pose_graph_optimization's inner
    loop, scoped to local window indices, with the prior treated as a
    genuine factor (its own residual/Jacobian against a frozen reference
    pose) rather than a precomputed linear term -- see the module docstring
    for why that distinction matters once poses keep moving across windows.
    Arguments:
        x_window: list of current pose estimates (4,4), oldest first
        prior_ref: (4,4) frozen reference pose for the prior factor on
            x_window[0], or None if there is no prior yet
        prior_info: (6,6) information matrix for the prior factor, or None
        edges_window: list of (local_i, local_j, Z_ij) edges, indices local to x_window
        info_matrix: (6,6) information matrix shared by every edge
    Returns:
        H: (6*W, 6*W) information (Gauss-Newton Hessian) matrix
        g: (6*W,) right-hand side (H @ delta = g gives the GN correction step)
    """
    W = len(x_window)
    dof = 6 * W
    H = np.zeros((dof, dof))
    g = np.zeros(dof)

    if prior_info is not None:
        e_prior = se3_log(se3_inv(prior_ref) @ x_window[0])
        J_prior = compute_se3_inv_right_jacobian(e_prior)
        H[0:6, 0:6] += J_prior.T @ prior_info @ J_prior
        g[0:6] += -J_prior.T @ prior_info @ e_prior

    for idx_i, idx_j, Z_ij in edges_window:
        J_i, J_j, e_vec = linearize_edge(x_window[idx_i], x_window[idx_j], Z_ij)
        c0, c1 = 6 * idx_i, 6 * idx_j
        H[c0:c0 + 6, c0:c0 + 6] += J_i.T @ info_matrix @ J_i
        H[c1:c1 + 6, c1:c1 + 6] += J_j.T @ info_matrix @ J_j
        H[c0:c0 + 6, c1:c1 + 6] += J_i.T @ info_matrix @ J_j
        H[c1:c1 + 6, c0:c0 + 6] += J_j.T @ info_matrix @ J_i
        g[c0:c0 + 6] += -J_i.T @ info_matrix @ e_vec
        g[c1:c1 + 6] += -J_j.T @ info_matrix @ e_vec

    return H, g


def solve_to_convergence(x_window, prior_ref, prior_info, edges_window, info_matrix, gn_tol, gn_max_iters):
    """Repeated assemble_window_system + Gauss-Newton step, to convergence --
    used for a single window's per-step solve, and directly reusable as a
    plain batch solver over any window (including a full graph) for testing.
    Arguments: see assemble_window_system, plus gn_tol/gn_max_iters (Gauss-
        Newton convergence tolerance and iteration cap).
    Returns:
        x: converged pose estimates (4,4 each)
        max_dof: the size of the largest system assembled during this call
            (always 6*len(x_window) here, exposed for the caller's running max)
    """
    x = list(x_window)
    max_dof = 6 * len(x)
    for _ in range(gn_max_iters):
        H, g = assemble_window_system(x, prior_ref, prior_info, edges_window, info_matrix)
        delta = np.linalg.solve(H, g)
        x = [x[k] @ se3_exp(delta[6 * k:6 * k + 6]) for k in range(len(x))]
        if np.linalg.norm(delta) < gn_tol:
            break
    return x, max_dof


def marginalize_oldest(x_window, prior_ref, prior_info, edges_window, info_matrix):
    """Marginalizes x_window[0] out of a *converged* window via the Schur
    complement (marginalization.md §4), returning a new unary prior factor
    over the next-oldest surviving pose. In a pure odometry chain, x_window[0]
    is connected to exactly one other still-alive variable (x_window[1]), so
    the correction term is provably nonzero only in that single 6x6 block --
    verified directly by this module's tests, not merely assumed.
    Arguments: same x_window/prior_ref/prior_info/edges_window/info_matrix as
        assemble_window_system, with x_window already solved to convergence
        for the current prior/edges.
    Returns:
        new_prior_ref: (4,4) x_window[1]'s current estimate, frozen as the
            new prior's reference pose (its residual is exactly 0 there,
            since we're marginalizing at the joint optimum)
        new_prior_info: (6,6) information matrix of the new prior
        x_dropped: (4,4) the marginalized pose's final, frozen estimate
    """
    H, _ = assemble_window_system(x_window, prior_ref, prior_info, edges_window, info_matrix)
    Lambda_aa = H[0:6, 0:6]
    Lambda_ab = H[0:6, 6:]
    Lambda_ba = H[6:, 0:6]
    correction = Lambda_ba @ np.linalg.solve(Lambda_aa, Lambda_ab)
    new_prior_info = correction[0:6, 0:6]
    new_prior_ref = x_window[1]
    x_dropped = x_window[0]
    return new_prior_ref, new_prior_info, x_dropped


def run_sliding_window_pose_graph(gt_poses, odometry_constraints, info_matrix, window_size,
                                   anchor_weight, gn_tol, gn_max_iters):
    """Streams nodes one at a time, keeping at most window_size poses live at
    once and marginalizing the oldest whenever a new node would exceed that.
    Arguments:
        gt_poses: list of ground-truth poses (4,4), used only for their count
        odometry_constraints: list of (idx_i, idx_j, Z_ij) sequential edges
        info_matrix: (6,6) information matrix shared by every edge
        window_size: maximum number of poses kept live at once
        anchor_weight: information weight of the node-0 gauge-fixing prior
            (the seed prior every later marginalization's prior chain builds on)
        gn_tol, gn_max_iters: Gauss-Newton convergence tolerance/iteration cap
    Returns:
        x_final: list of frozen final pose estimates (4,4), one per node --
            marginalized poses keep whatever estimate they had at the moment
            they were dropped (marginalization.md §4: "it's never coming back")
        max_dof: largest dense system size (rows of H) ever assembled during
            the run -- capped at 6*window_size here, by construction
    """
    n_poses = len(gt_poses)
    x_final = [None] * n_poses

    x_window = [gt_poses[0]]
    global_idx = [0]
    edges_window = []
    prior_ref, prior_info = gt_poses[0], anchor_weight * np.eye(6)
    max_dof = 6

    for k in range(1, n_poses):
        idx_i, idx_j, Z_ij = odometry_constraints[k - 1]

        # Marginalize first, *before* the new node arrives, whenever the window is already
        # at capacity -- using the linearization point it already converged to on the
        # previous iteration (still valid; no new information has arrived yet). This is what
        # makes the cap a true hard cap: the solve below never runs at more than window_size
        # poses, not window_size+1-then-trimmed.
        if len(x_window) >= window_size:
            new_prior_ref, new_prior_info, x_dropped = marginalize_oldest(
                x_window, prior_ref, prior_info, edges_window, info_matrix)
            x_final[global_idx[0]] = x_dropped
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

    for local_i, g_i in enumerate(global_idx):
        x_final[g_i] = x_window[local_i]

    return x_final, max_dof


def run_full_batch_growing(gt_poses, odometry_constraints, info_matrix, gn_tol, gn_max_iters):
    """The unbounded baseline: re-solves the entire graph from scratch every
    time a new node streams in, via pose_graph.run_pose_graph_optimization
    (reused unmodified) -- mirrors pose_graph_incremental.run_batch_streaming's
    structure, without loop closures, for a matched comparison against the
    sliding-window approach specifically.
    Arguments:
        gt_poses: list of ground-truth poses (4,4), used only for their count
        odometry_constraints: list of (idx_i, idx_j, Z_ij) sequential edges
        info_matrix: (6,6) information matrix shared by every edge
        gn_tol, gn_max_iters: Gauss-Newton convergence tolerance/iteration cap
    Returns:
        x_final: list of estimated poses (4,4) after all nodes have streamed in
        max_dof: largest dense system size ever assembled -- always
            6*n_poses here, growing without bound as the trajectory grows
    """
    n_poses = len(gt_poses)
    x_final = None
    max_dof = 6
    for k in range(1, n_poses + 1):
        edges = odometry_constraints[:k - 1]
        T_init = run_dead_reckoning(gt_poses[0], odometry_constraints[:k - 1])
        max_dof = max(max_dof, 6 * k)
        with contextlib.redirect_stdout(io.StringIO()):
            x_final = run_pose_graph_optimization(T_init, edges, info_matrix, damping=0.0,
                                                   gn_tol=gn_tol, gn_max_iters=gn_max_iters)
    return x_final, max_dof


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--nodes-per-side-sweep", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64],
                         help="Sweep of nodes-per-side values (trajectory has 4x this many total "
                              "poses); the actual resource-constraint result is how cost scales "
                              "across this sweep, not any single run")
    parser.add_argument("--side-length", type=float, default=2.0, help="Side length of the square ground-truth path (m)")
    parser.add_argument("--window-size", type=int, default=10, help="Sliding-window size (poses kept live at once)")

    parser.add_argument("--pos-noise-std", type=float, default=0.05, help="Odometry-edge translation noise std-dev (m)")
    parser.add_argument("--rot-noise-std", type=float, default=0.01, help="Odometry-edge rotation noise std-dev (rad)")

    parser.add_argument("--anchor-weight", type=float, default=1e6, help="Information weight of the node-0 gauge-fixing prior")
    parser.add_argument("--gn-tol", type=float, default=1e-6, help="Gauss-Newton convergence tolerance")
    parser.add_argument("--gn-max-iters", type=int, default=10, help="Maximum Gauss-Newton iterations per solve")

    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    info_matrix = np.eye(6)

    results = []
    for nodes_per_side in args.nodes_per_side_sweep:
        gt_poses = generate_ground_truth_trajectory(args.side_length, nodes_per_side=nodes_per_side)
        n_poses = len(gt_poses)
        odometry_constraints, _ = simulate_noisy_edges(
            gt_poses, args.pos_noise_std, args.rot_noise_std, loop_noise_scale=1.0, rng=rng)

        print(f"n_poses={n_poses}: running full-batch-growing and sliding-window "
              f"(window_size={args.window_size})...")

        (x_batch, dof_batch), t_batch, mem_batch = measure_performance(
            run_full_batch_growing, gt_poses, odometry_constraints, info_matrix,
            args.gn_tol, args.gn_max_iters, n_steps=n_poses)

        (x_window, dof_window), t_window, mem_window = measure_performance(
            run_sliding_window_pose_graph, gt_poses, odometry_constraints, info_matrix,
            args.window_size, args.anchor_weight, args.gn_tol, args.gn_max_iters, n_steps=n_poses)

        _, pos_err_batch = pose_errors(gt_poses, x_batch)
        _, pos_err_window = pose_errors(gt_poses, x_window)

        results.append({
            "n_poses": n_poses,
            "dof_batch": dof_batch, "dof_window": dof_window,
            "t_batch": t_batch, "t_window": t_window,
            "mem_batch": mem_batch, "mem_window": mem_window,
            "rms_batch": np.sqrt(np.mean(pos_err_batch ** 2)),
            "rms_window": np.sqrt(np.mean(pos_err_window ** 2)),
        })

    bytes_per_dof2 = 8  # float64
    print("\nPeak information-matrix size (the headline resource-constraint metric -- see module "
          "docstring for why this, not whole-process memory, is what's reported here):")
    print(f"  {'n_poses':>8s} | {'batch max_dof':>14s} {'(~bytes)':>12s} | "
          f"{'window max_dof':>15s} {'(~bytes)':>12s}")
    for r in results:
        print(f"  {r['n_poses']:>8d} | {r['dof_batch']:>14d} "
              f"{r['dof_batch']**2 * bytes_per_dof2:>12,d} | "
              f"{r['dof_window']:>15d} {r['dof_window']**2 * bytes_per_dof2:>12,d}")

    print("\nWhole-call timing/memory (utils.measure_performance; memory column includes this "
          "script's own O(n_poses) output bookkeeping, shared by both -- see module docstring):")
    for r in results:
        print(f"  n_poses={r['n_poses']:<6d} batch: {r['t_batch']*1e6:8.2f} us/step, "
              f"{r['mem_batch']/1024:8.2f} KB/step | window: {r['t_window']*1e6:8.2f} us/step, "
              f"{r['mem_window']/1024:8.2f} KB/step")

    print("\nFinal RMS position error (accuracy isn't sacrificed for bounded memory):")
    for r in results:
        print(f"  n_poses={r['n_poses']:<6d} batch={r['rms_batch']:.4f} m, "
              f"window={r['rms_window']:.4f} m")

    n_poses_arr = np.array([r["n_poses"] for r in results])
    dof_batch_arr = np.array([r["dof_batch"] for r in results])
    dof_window_arr = np.array([r["dof_window"] for r in results])
    t_batch_arr = np.array([r["t_batch"] for r in results])
    t_window_arr = np.array([r["t_window"] for r in results])

    fig, (ax_dof, ax_time) = plt.subplots(2, 1, figsize=(8, 9))

    ax_dof.plot(n_poses_arr, dof_batch_arr ** 2 * bytes_per_dof2 / 1024, label="full-batch (unbounded)",
                color="tab:red", marker="o")
    ax_dof.plot(n_poses_arr, dof_window_arr ** 2 * bytes_per_dof2 / 1024, label="sliding-window (bounded)",
                color="tab:blue", marker="o")
    ax_dof.set_ylabel("Peak information-matrix size (KB)")
    ax_dof.set_title("Sliding-window marginalization: bounded vs. unbounded resource growth")
    ax_dof.set_yscale("log")
    ax_dof.legend()

    ax_time.plot(n_poses_arr, t_batch_arr * 1e6, label="full-batch (unbounded)", color="tab:red", marker="o")
    ax_time.plot(n_poses_arr, t_window_arr * 1e6, label="sliding-window (bounded)", color="tab:blue", marker="o")
    ax_time.set_xlabel("Trajectory length (n_poses)")
    ax_time.set_ylabel("Avg. wall-clock time per step (us)")
    ax_time.set_yscale("log")
    ax_time.legend()

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
