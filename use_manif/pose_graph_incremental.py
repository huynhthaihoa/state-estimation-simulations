'''
Incremental (square-root SAM) pose-graph solving using manifpy, contrasted
with the batch Levenberg-Marquardt/Gauss-Newton relaxation in pose_graph.py
-- the mechanism docs/optimization/isam_optimization.md Section 3 describes
(Dellaert & Kaess 2006's square-root-information factorization, Kaess et al.
2008's iSAM) but pose_graph.py's batch solver doesn't: absorbing one new
measurement into an existing factorization via Givens-rotation row insertion
(`qr_insert_row`, utils.py) instead of rebuilding the whole linear system
from scratch.

A robot streams around a longer square loop one node at a time (odometry
edge, then the next node). Two solvers process the same stream:

  - `run_batch_streaming`: the expensive baseline. Every new node triggers a
    full re-solve of the *entire* graph so far, from a fresh dead-reckoning
    guess, via pose_graph.py's own `run_pose_graph_optimization` (reused
    unmodified) -- exactly the "throw everything away and start over"
    approach Section 1/Section 8 of the doc contrasts with incremental
    smoothing.
  - `run_incremental_pose_graph`: absorbs each new odometry edge into a
    running upper-triangular square-root-information matrix R (and
    right-hand side d) via `qr_insert_row`, touching only the new node's
    columns -- no other row is re-evaluated. Every `--relinearize-every`
    nodes, and unconditionally on the loop-closure edge (Section 6/Section 9:
    "loop closure can affect many previous poses", exactly the case that
    would touch nearly every column anyway), it instead does a full
    relinearization: adopt the current estimate as the new linearization
    point and rebuild R/d from scratch via one `np.linalg.qr` over every
    factor seen so far, iterating to Gauss-Newton convergence the same way
    pose_graph.py's batch solver does.

What this deliberately does *not* implement: Section 7's Bayes tree
(iSAM2's further refinement -- selective relinearization of only the
affected subtree) and variable reordering (COLAMD) to bound fill-in. Both
are explicitly out of scope here; this script implements iSAM's original
QR/square-root-information mechanism (Kaess et al. 2008), not iSAM2. The
incremental solver also runs plain Gauss-Newton (no Levenberg-Marquardt
damping) between/at relinearizations -- the always-present anchor prior on
node 0 already keeps the system well-conditioned.
'''

import argparse
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import manifpy as manif
from scipy.linalg import solve_triangular

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import qr_insert_row, measure_performance
from pose_graph import (generate_ground_truth_trajectory, simulate_noisy_edges, run_dead_reckoning,
                         run_pose_graph_optimization, pose_errors)


def linearize_edge(Xi, Xj, Z_ij):
    """Residual and Jacobians of e_ij = Xj.rminus(Xi.compose(Z_ij)) -- the
    same Jacobian-chaining pattern as pose_graph.py's
    run_pose_graph_optimization, factored out here since both the
    incremental-insertion path and the full-relinearization path need it.
    Arguments:
        Xi, Xj: current pose estimates at the edge's two endpoints (manif.SE3)
        Z_ij: measured relative pose (manif.SE3)
    Returns:
        J_i: (6,6) d e_ij / d Xi
        J_j: (6,6) d e_ij / d Xj
        e_vec: (6,) residual
    """
    Jc_self = np.zeros((6, 6))
    T_pred = Xi.compose(Z_ij, Jc_self)

    Ja, Jb = np.zeros((6, 6)), np.zeros((6, 6))
    e_vec = Xj.rminus(T_pred, Ja, Jb).coeffs()

    J_i = Jb @ Jc_self  # d e_ij / d Xi, chained through T_pred
    J_j = Ja            # d e_ij / d Xj
    return J_i, J_j, e_vec


def edge_whitened_block(idx_i, idx_j, Z_ij, x_lin, dof, sqrt_info):
    """(6,dof) whitened Jacobian block and (6,) whitened target for one edge,
    ready to insert into the sqrt-information system: solving
    (whitened block) @ delta = target drives this edge's residual toward zero.
    Arguments:
        idx_i, idx_j: node indices this edge connects
        Z_ij: measured relative pose (manif.SE3)
        x_lin: list of pose estimates, the current linearization point
        dof: total system size (6 * number of poses currently in x_lin)
        sqrt_info: (6,6) upper-triangular sqrt of the shared information matrix
    Returns:
        A_block: (6,dof) whitened Jacobian rows (zero outside columns idx_i, idx_j)
        b_block: (6,) whitened right-hand side (= sqrt_info @ (-residual))
    """
    J_i, J_j, e_vec = linearize_edge(x_lin[idx_i], x_lin[idx_j], Z_ij)
    A_block = np.zeros((6, dof))
    A_block[:, 6 * idx_i:6 * idx_i + 6] = sqrt_info @ J_i
    A_block[:, 6 * idx_j:6 * idx_j + 6] = sqrt_info @ J_j
    b_block = sqrt_info @ (-e_vec)
    return A_block, b_block


def full_relinearize(x_lin, odom_seen, loop_seen, info_matrix, anchor_weight):
    """Rebuilds the sqrt-information system (R, d) completely from scratch:
    stacks the node-0 anchor prior plus every odometry/loop-closure factor
    seen so far, linearized at x_lin, and QR-factorizes the whole thing in
    one shot. This is the expensive operation `qr_insert_row` exists to
    avoid running on every single new edge.
    Arguments:
        x_lin: list of pose estimates, the linearization point (manif.SE3 each)
        odom_seen, loop_seen: lists of (idx_i, idx_j, Z_ij) factors absorbed so far
        info_matrix: 6x6 information matrix shared by every edge
        anchor_weight: information weight of the node-0 gauge-fixing prior
    Returns:
        R: (dof,dof) upper-triangular sqrt-information matrix
        d: (dof,) transformed right-hand side
    """
    n_poses = len(x_lin)
    dof = 6 * n_poses
    sqrt_info = np.linalg.cholesky(info_matrix).T

    A_rows = [np.sqrt(anchor_weight) * np.eye(6, dof)]
    b_rows = [np.zeros(6)]
    for idx_i, idx_j, Z_ij in odom_seen + loop_seen:
        A_block, b_block = edge_whitened_block(idx_i, idx_j, Z_ij, x_lin, dof, sqrt_info)
        A_rows.append(A_block)
        b_rows.append(b_block)

    A_full = np.vstack(A_rows)
    b_full = np.concatenate(b_rows)
    Q, R = np.linalg.qr(A_full)
    d = Q.T @ b_full
    return R, d


def relinearize_to_convergence(x_current, odom_seen, loop_seen, info_matrix, anchor_weight,
                                gn_tol, gn_max_iters):
    """Repeated full_relinearize + solve + retract, to Gauss-Newton
    convergence -- the same iterate-until-converged pattern as
    pose_graph.py's batch solver, just built on QR instead of normal
    equations. Returns the converged linearization point together with the
    R, d already built at that point, so the caller can resume incremental
    insertion from a fully up-to-date factorization.
    Arguments:
        x_current: list of pose estimates to relinearize around (manif.SE3 each)
        odom_seen, loop_seen: lists of (idx_i, idx_j, Z_ij) factors to include
        info_matrix: 6x6 information matrix shared by every edge
        anchor_weight: information weight of the node-0 gauge-fixing prior
        gn_tol: convergence tolerance on the correction step norm
        gn_max_iters: maximum number of relinearization iterations
    Returns:
        x_current: converged pose estimates (manif.SE3 each)
        R, d: sqrt-information system at the converged linearization point
    """
    R, d = None, None
    for _ in range(gn_max_iters):
        R, d = full_relinearize(x_current, odom_seen, loop_seen, info_matrix, anchor_weight)
        delta = solve_triangular(R, d)
        x_current = [x_current[k] + manif.SE3Tangent(delta[6 * k:6 * k + 6]) for k in range(len(x_current))]
        if np.linalg.norm(delta) < gn_tol:
            break
    return x_current, R, d


def run_incremental_pose_graph(gt_poses, odometry_constraints, loop_constraints, info_matrix,
                                anchor_weight, relinearize_every, gn_tol, gn_max_iters):
    """Incremental square-root SAM: streams in one node/edge at a time,
    absorbing each new odometry edge via Givens-rotation row insertion
    (qr_insert_row) instead of a full re-solve, with periodic and
    loop-closure-triggered full relinearization. See the module docstring
    for what this deliberately does/doesn't implement relative to iSAM2.
    Arguments:
        gt_poses: list of ground-truth poses (manif.SE3), used only for their count
        odometry_constraints: list of (idx_i, idx_j, Z_ij) sequential edges
        loop_constraints: list with the single (idx_i, idx_j, Z_ij) loop-closure edge
        info_matrix: 6x6 information matrix shared by every edge
        anchor_weight: information weight of the node-0 gauge-fixing prior
        relinearize_every: force a full relinearization every this many new nodes
                            (the loop-closure edge always forces one too)
        gn_tol: convergence tolerance used during each relinearization
        gn_max_iters: maximum Gauss-Newton iterations per relinearization
    Returns:
        x_est: list of estimated poses (manif.SE3) after all nodes/edges have streamed in
        n_relinearizations: total number of full relinearizations run
    """
    n_poses = len(gt_poses)
    sqrt_info = np.linalg.cholesky(info_matrix).T

    x_lin = [gt_poses[0]]
    odom_seen, loop_seen = [], []
    x_lin, R, d = relinearize_to_convergence(x_lin, odom_seen, loop_seen, info_matrix,
                                              anchor_weight, gn_tol, gn_max_iters)
    n_relinearizations = 1

    def read_out():
        delta = solve_triangular(R, d)
        return [x_lin[k] + manif.SE3Tangent(delta[6 * k:6 * k + 6]) for k in range(len(x_lin))]

    x_est = read_out()

    for k in range(1, n_poses):
        idx_i, idx_j, Z_ij = odometry_constraints[k - 1]
        x_lin.append(x_lin[-1] * Z_ij)  # dead-reckon the new node's linearization point
        odom_seen.append((idx_i, idx_j, Z_ij))

        dof = 6 * len(x_lin)
        R_grown, d_grown = np.zeros((dof, dof)), np.zeros(dof)
        R_grown[:dof - 6, :dof - 6], d_grown[:dof - 6] = R, d
        R, d = R_grown, d_grown

        A_block, b_block = edge_whitened_block(idx_i, idx_j, Z_ij, x_lin, dof, sqrt_info)
        for r in range(6):
            qr_insert_row(R, d, A_block[r], b_block[r])

        x_est = read_out()

        is_last_node = (k == n_poses - 1)
        if is_last_node and loop_constraints:
            loop_seen.append(loop_constraints[0])

        if (k % relinearize_every == 0) or (is_last_node and loop_constraints):
            x_lin, R, d = relinearize_to_convergence(x_est, odom_seen, loop_seen, info_matrix,
                                                       anchor_weight, gn_tol, gn_max_iters)
            n_relinearizations += 1
            x_est = read_out()

    return x_est, n_relinearizations


def run_batch_streaming(gt_poses, odometry_constraints, loop_constraints, info_matrix,
                         damping, gn_tol, gn_max_iters):
    """The expensive incremental baseline: every time a new node streams in,
    rebuild the constraint list and re-run the existing, unmodified
    pose_graph.run_pose_graph_optimization from a fresh dead-reckoning guess
    -- the "throw everything away and re-solve" approach
    docs/optimization/isam_optimization.md Section 1/Section 8 contrasts
    with incremental smoothing.
    Arguments:
        gt_poses: list of ground-truth poses (manif.SE3), used only for their count
        odometry_constraints: list of (idx_i, idx_j, Z_ij) sequential edges
        loop_constraints: list with the single (idx_i, idx_j, Z_ij) loop-closure edge
        info_matrix: 6x6 information matrix shared by every edge
        damping: Levenberg-Marquardt damping passed through to the batch solver
        gn_tol: convergence tolerance passed through to the batch solver
        gn_max_iters: maximum iterations passed through to the batch solver
    Returns:
        T_final: list of estimated poses (manif.SE3) after all nodes/edges have streamed in
        n_relinearizations: total number of from-scratch batch solves run
    """
    n_poses = len(gt_poses)
    T_final = None
    for k in range(2, n_poses + 1):
        edges = odometry_constraints[:k - 1]
        if k == n_poses and loop_constraints:
            edges = edges + loop_constraints
        T_init = run_dead_reckoning(gt_poses[0], odometry_constraints[:k - 1])
        with contextlib.redirect_stdout(io.StringIO()):
            T_final = run_pose_graph_optimization(T_init, edges, info_matrix, damping, gn_tol, gn_max_iters)
    return T_final, n_poses - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--side-length", type=float, default=2.0, help="Side length of the square ground-truth loop (m)")
    parser.add_argument("--nodes-per-side", type=int, default=16, help="Nodes per side of the loop (streamed in one at a time)")

    parser.add_argument("--pos-noise-std", type=float, default=0.05, help="Odometry-edge translation noise std-dev (m)")
    parser.add_argument("--rot-noise-std", type=float, default=0.01, help="Odometry-edge rotation noise std-dev (rad)")
    parser.add_argument("--loop-noise-scale", type=float, default=0.5, help="Noise std-dev multiplier for the loop-closure edge")

    parser.add_argument("--damping", type=float, default=0.01, help="Levenberg-Marquardt damping for the batch-streaming baseline only")
    parser.add_argument("--gn-tol", type=float, default=1e-6, help="Gauss-Newton convergence tolerance")
    parser.add_argument("--gn-max-iters", type=int, default=10, help="Maximum Gauss-Newton iterations per solve/relinearization")

    parser.add_argument("--anchor-weight", type=float, default=1e6, help="Information weight of the node-0 gauge-fixing prior")
    parser.add_argument("--relinearize-every", type=int, default=8, help="Force a full relinearization every this many new nodes")

    parser.add_argument("--seed", type=int, default=0, help="RNG seed")

    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    gt_poses = generate_ground_truth_trajectory(args.side_length, args.nodes_per_side)
    n_poses = len(gt_poses)

    odometry_constraints, loop_constraints = simulate_noisy_edges(
        gt_poses, args.pos_noise_std, args.rot_noise_std, args.loop_noise_scale, rng)

    print(f"Generated {n_poses} ground-truth poses ({args.nodes_per_side} per side), "
          f"streamed in one node at a time.")

    T_dr = run_dead_reckoning(gt_poses[0], odometry_constraints)

    info_matrix = np.eye(6)

    print("Running batch-streaming baseline (full re-solve from scratch at every new node)...")
    (T_batch, n_relin_batch), t_batch, _ = measure_performance(
        run_batch_streaming, gt_poses, odometry_constraints, loop_constraints, info_matrix,
        args.damping, args.gn_tol, args.gn_max_iters, n_steps=n_poses)

    print("Running incremental square-root SAM (Givens-rotation QR updates)...")
    (T_incr, n_relin_incr), t_incr, _ = measure_performance(
        run_incremental_pose_graph, gt_poses, odometry_constraints, loop_constraints, info_matrix,
        args.anchor_weight, args.relinearize_every, args.gn_tol, args.gn_max_iters, n_steps=n_poses)

    rot_err_dr, pos_err_dr = pose_errors(gt_poses, T_dr)
    rot_err_batch, pos_err_batch = pose_errors(gt_poses, T_batch)
    rot_err_incr, pos_err_incr = pose_errors(gt_poses, T_incr)

    print("\nFinal / RMS errors:")
    for name, rot_err, pos_err in [
        ("Uncorrected (odometry)", rot_err_dr, pos_err_dr),
        ("Batch streaming", rot_err_batch, pos_err_batch),
        ("Incremental sqrt-SAM", rot_err_incr, pos_err_incr),
    ]:
        print(f"  {name:<24s} final rot={rot_err[-1]:7.3f} deg, pos={pos_err[-1]:7.4f} m | "
              f"RMS rot={np.sqrt(np.mean(rot_err**2)):7.3f} deg, pos={np.sqrt(np.mean(pos_err**2)):7.4f} m")

    print(f"\nRelinearizations: batch-streaming={n_relin_batch} (one full solve per new node) "
          f"vs. incremental={n_relin_incr} (every {args.relinearize_every} nodes + loop closure)")
    speedup = t_batch / t_incr if t_incr > 0 else float("inf")
    print(f"Avg wall-clock per node: batch-streaming={t_batch * 1e3:.3f} ms, "
          f"incremental={t_incr * 1e3:.3f} ms ({speedup:.1f}x faster)")

    def xy(T_list):
        pts = np.array([T.translation() for T in T_list] + [T_list[0].translation()])
        return pts[:, 0], pts[:, 1]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(*xy(gt_poses), label="Ground truth", color="black", linewidth=2, marker="o", markersize=3)
    ax.plot(*xy(T_dr), label="Uncorrected (odometry)", color="tab:gray", linestyle="--", marker="o", markersize=3)
    ax.plot(*xy(T_batch), label="Batch streaming", color="tab:blue", linestyle="--", marker="o", markersize=3)
    ax.plot(*xy(T_incr), label="Incremental sqrt-SAM", color="tab:orange", linestyle=":", marker="o", markersize=3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Incremental vs. batch pose-graph solving:\nsame stream of nodes/edges, same final answer, different cost")
    ax.axis("equal")
    ax.legend()

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
