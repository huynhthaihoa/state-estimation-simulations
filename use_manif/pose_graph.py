'''
3D pose-graph relaxation (Levenberg-Marquardt / Gauss-Newton) using manifpy.

Simulates a small closed-loop SE(3) pose graph: a robot drives around a
square, accumulating drift from noisy relative-pose ("odometry") edges
between consecutive nodes, then detects it has returned to the start and adds
one loop-closure edge back to node 0. The graph is relaxed by jointly
optimizing every node pose against each edge's residual
e_ij = Log(Z_ij^-1 * X_i^-1 * X_j) (Z_ij the measured relative pose, X_i/X_j
the current pose estimates), using manif's analytical `compose`/`rminus`
Jacobians -- the same Jacobian-chaining pattern as the motion factors in
`run_batch_gn` (pointcloud_pose_tracking_manif.py): a factor's Jacobian wrt
the upstream node is the downstream node's "other" Jacobian chained through
the upstream node's own Jacobian in the predicted-pose composition.

A pure dead-reckoning trajectory (sequential noisy-edge composition, no loop
closure) is carried along as the uncorrected baseline, and also doubles as
the optimizer's initial guess -- the same role it plays in
pointcloud_pose_tracking.py.
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
import manifpy as manif


def generate_ground_truth_trajectory(side_length, nodes_per_side=1):
    """A closed square loop in the XY plane (start -> +X -> +Y -> -X), with an
    implicit loop-closure edge back from the last node to the first.
    Arguments:
        side_length: length of each side of the square (m)
        nodes_per_side: number of edge-segments per side (1 = corners only,
                        the original 4-node loop; >1 subdivides each side
                        with evenly-spaced intermediate nodes, for a longer
                        streaming trajectory)
    Returns:
        gt_poses: list of 4*nodes_per_side ground-truth poses (manif.SE3)
    """
    corners = [(0.0, 0.0), (side_length, 0.0), (side_length, side_length), (0.0, side_length)]
    gt_poses = []
    for k in range(len(corners)):
        x0, y0 = corners[k]
        x1, y1 = corners[(k + 1) % len(corners)]
        for step in range(nodes_per_side):
            frac = step / nodes_per_side
            x, y = x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)
            gt_poses.append(manif.SE3(np.array([x, y, 0.0, 0.0, 0.0, 0.0, 1.0])))
    return gt_poses


def simulate_noisy_edges(gt_poses, pos_noise_std, rot_noise_std, loop_noise_scale, rng):
    """Builds noisy relative-pose constraints: one odometry edge between each
    pair of consecutive nodes, plus one loop-closure edge from the last node
    back to the first.
    Arguments:
        gt_poses: list of ground-truth poses (manif.SE3)
        pos_noise_std: std-dev of Gaussian noise on each edge's translation (m)
        rot_noise_std: std-dev of Gaussian noise on each edge's rotation (rad)
        loop_noise_scale: multiplier applied to the noise std-devs for the loop-closure edge
                           only (a loop-closure detector is usually more precise than raw odometry)
        rng: numpy random number generator
    Returns:
        odometry_constraints: list of (idx_i, idx_j, Z_ij) edges between consecutive nodes
        loop_constraints: list of (idx_i, idx_j, Z_ij) edges closing the loop back to node 0
                           (Z_ij is the measured relative pose, a manif.SE3)
    """
    def noisy_edge(T_i, T_j, scale):
        Z_true = T_i.inverse() * T_j
        noise = np.concatenate([
            rng.normal(0.0, pos_noise_std * scale, 3),
            rng.normal(0.0, rot_noise_std * scale, 3),
        ])
        return Z_true + manif.SE3Tangent(noise)

    n_poses = len(gt_poses)
    odometry_constraints = [(k, k + 1, noisy_edge(gt_poses[k], gt_poses[k + 1], 1.0))
                             for k in range(n_poses - 1)]
    loop_constraints = [(n_poses - 1, 0, noisy_edge(gt_poses[-1], gt_poses[0], loop_noise_scale))]
    return odometry_constraints, loop_constraints


def run_dead_reckoning(T_init, odometry_constraints):
    """Prior-only baseline: composes each odometry edge sequentially from the
    anchored first pose, never looking at the loop-closure edge -- the
    uncorrected trajectory a robot with no loop-closure detection would end
    up with, and also the optimizer's initial guess.
    Arguments:
        T_init: anchored pose of node 0 (manif.SE3)
        odometry_constraints: list of (idx_i, idx_j, Z_ij) edges between consecutive nodes
    Returns:
        T_list: list of estimated poses (manif.SE3)
    """
    T_list = [T_init]
    for _, _, Z_ij in odometry_constraints:
        T_list.append(T_list[-1] * Z_ij)
    return T_list


def run_pose_graph_optimization(T_init_list, constraints, info_matrix, damping, gn_tol, gn_max_iters):
    """Batch Gauss-Newton / Levenberg-Marquardt pose-graph relaxation: jointly
    optimizes every node pose against all edge residuals
    e_ij = Xj.rminus(Xi.compose(Z_ij)) = Log(Z_ij^-1 * Xi^-1 * Xj), fixing the
    gauge freedom by heavily anchoring node 0.

    Each edge's Jacobian wrt Xj comes straight from rminus's "self" Jacobian;
    its Jacobian wrt Xi is that same rminus "other" Jacobian chained through
    compose's "self" Jacobian (the predicted pose T_pred = Xi.compose(Z_ij)'s
    own sensitivity to Xi) -- exactly the motion-factor Jacobian-chaining
    pattern used in run_batch_gn (pointcloud_pose_tracking_manif.py),
    generalized from a twist-based motion model to a directly-measured
    relative pose.
    Arguments:
        T_init_list: list of initial pose guesses (manif.SE3)
        constraints: list of (idx_i, idx_j, Z_ij) edges (manif.SE3 measurements)
        info_matrix: 6x6 information (inverse covariance) matrix shared by every edge (numpy array)
        damping: Levenberg-Marquardt damping added to the normal equations' diagonal
        gn_tol: convergence tolerance on the correction step norm
        gn_max_iters: maximum number of iterations
    Returns:
        T_est: list of optimized poses (manif.SE3)
    """
    n_poses = len(T_init_list)
    dof = 6 * n_poses
    T_est = list(T_init_list)

    for it in range(gn_max_iters):
        H = np.zeros((dof, dof))
        g = np.zeros(dof)
        H[0:6, 0:6] += np.eye(6) * 1e6  # anchor node 0 (gauge freedom)

        total_error = 0.0
        for idx_i, idx_j, Z_ij in constraints:
            Xi, Xj = T_est[idx_i], T_est[idx_j]

            Jc_self = np.zeros((6, 6))
            T_pred = Xi.compose(Z_ij, Jc_self)

            Ja, Jb = np.zeros((6, 6)), np.zeros((6, 6))
            e_vec = Xj.rminus(T_pred, Ja, Jb).coeffs()

            J_i = Jb @ Jc_self  # d e_ij / d Xi, chained through T_pred
            J_j = Ja            # d e_ij / d Xj

            total_error += e_vec @ info_matrix @ e_vec

            c0, c1 = 6 * idx_i, 6 * idx_j
            H[c0:c0 + 6, c0:c0 + 6] += J_i.T @ info_matrix @ J_i
            H[c1:c1 + 6, c1:c1 + 6] += J_j.T @ info_matrix @ J_j
            H[c0:c0 + 6, c1:c1 + 6] += J_i.T @ info_matrix @ J_j
            H[c1:c1 + 6, c0:c0 + 6] += J_j.T @ info_matrix @ J_i
            g[c0:c0 + 6] += -J_i.T @ info_matrix @ e_vec
            g[c1:c1 + 6] += -J_j.T @ info_matrix @ e_vec

        print(f"    Iteration {it + 1}: total chi-squared error = {total_error:.6f}")

        H += np.eye(dof) * damping
        delta = np.linalg.solve(H, g)

        for k in range(n_poses):
            T_est[k] = T_est[k] + manif.SE3Tangent(delta[6 * k:6 * k + 6])

        step_norm = np.linalg.norm(delta)
        if step_norm < gn_tol:
            print(f"    Converged after {it + 1} iteration(s) (|delta| < {gn_tol})")
            break
    else:
        print(f"    Reached max iterations ({gn_max_iters}) without full convergence")

    return T_est


def pose_errors(T_true_list, T_est_list):
    """Rotation geodesic error (deg) and position error (m), per node.
    Arguments:
        T_true_list: list of true poses (manif.SE3)
        T_est_list: list of estimated poses (manif.SE3)
    Returns:
        rot_err: (N,) array of rotation errors (deg)
        pos_err: (N,) array of position errors (m)
    """
    n = len(T_true_list)
    rot_err, pos_err = np.zeros(n), np.zeros(n)
    for k in range(n):
        R_true = manif.SO3(T_true_list[k].coeffs()[3:7])
        R_est = manif.SO3(T_est_list[k].coeffs()[3:7])
        rot_err[k] = np.degrees(np.linalg.norm(R_est.rminus(R_true).coeffs()))
        pos_err[k] = np.linalg.norm(T_true_list[k].translation() - T_est_list[k].translation())
    return rot_err, pos_err


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--side-length", type=float, default=2.0, help="Side length of the square ground-truth loop (m)")

    parser.add_argument("--pos-noise-std", type=float, default=0.05, help="Odometry-edge translation noise std-dev (m)")
    parser.add_argument("--rot-noise-std", type=float, default=0.01, help="Odometry-edge rotation noise std-dev (rad)")
    parser.add_argument("--loop-noise-scale", type=float, default=0.5, help="Noise std-dev multiplier for the loop-closure edge")

    parser.add_argument("--damping", type=float, default=0.01, help="Levenberg-Marquardt damping factor")
    parser.add_argument("--gn-tol", type=float, default=1e-6, help="Pose-graph optimization convergence tolerance")
    parser.add_argument("--gn-max-iters", type=int, default=10, help="Maximum pose-graph optimization iterations")

    parser.add_argument("--seed", type=int, default=0, help="RNG seed")

    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    gt_poses = generate_ground_truth_trajectory(args.side_length)
    n_poses = len(gt_poses)

    odometry_constraints, loop_constraints = simulate_noisy_edges(
        gt_poses, args.pos_noise_std, args.rot_noise_std, args.loop_noise_scale, rng)
    all_constraints = odometry_constraints + loop_constraints

    print(f"Generated {n_poses} ground-truth poses, {len(all_constraints)} constraints "
          f"({len(odometry_constraints)} odometry + {len(loop_constraints)} loop closure).")

    T_dr = run_dead_reckoning(gt_poses[0], odometry_constraints)
    print(f"Uncorrected drift at the loop-closure node: "
          f"{np.linalg.norm(T_dr[-1].translation() - gt_poses[-1].translation()):.4f} m")

    info_matrix = np.eye(6)  # equal weight on every DoF for every edge

    print("Running pose-graph relaxation (Levenberg-Marquardt)...")
    T_opt = run_pose_graph_optimization(T_dr, all_constraints, info_matrix, args.damping,
                                         args.gn_tol, args.gn_max_iters)

    rot_err_dr, pos_err_dr = pose_errors(gt_poses, T_dr)
    rot_err_opt, pos_err_opt = pose_errors(gt_poses, T_opt)

    print("\nFinal / RMS errors:")
    for name, rot_err, pos_err in [
        ("Uncorrected (odometry)", rot_err_dr, pos_err_dr),
        ("Pose-graph optimized", rot_err_opt, pos_err_opt),
    ]:
        print(f"  {name:<24s} final rot={rot_err[-1]:7.3f} deg, pos={pos_err[-1]:7.4f} m | "
              f"RMS rot={np.sqrt(np.mean(rot_err**2)):7.3f} deg, pos={np.sqrt(np.mean(pos_err**2)):7.4f} m")

    def xy(T_list):
        pts = np.array([T.translation() for T in T_list] + [T_list[0].translation()])
        return pts[:, 0], pts[:, 1]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(*xy(gt_poses), label="Ground truth", color="black", linewidth=2, marker="o")
    ax.plot(*xy(T_dr), label="Uncorrected (odometry)", color="tab:gray", linestyle="--", marker="o")
    ax.plot(*xy(T_opt), label="Pose-graph optimized", color="tab:blue", linestyle="--", marker="o")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Pose-graph relaxation:\nodometry drift vs. loop-closure correction")
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
