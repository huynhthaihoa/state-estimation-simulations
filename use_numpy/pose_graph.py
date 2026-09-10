'''
3D pose-graph relaxation (Levenberg-Marquardt / Gauss-Newton).

Simulates a small closed-loop SE(3) pose graph: a robot drives around a
square, accumulating drift from noisy relative-pose ("odometry") edges
between consecutive nodes, then detects it has returned to the start and adds
one loop-closure edge back to node 0. The graph is relaxed by jointly
optimizing every node pose against each edge's residual
e_ij = Log(Z_ij^-1 * X_i^-1 * X_j) (Z_ij the measured relative pose, X_i/X_j
the current pose estimates), using closed-form analytical compose/rminus
Jacobians (hand-rolled skew/exp/log/adjoint math, no external Lie-theory
library) -- the same Jacobian-chaining pattern as the motion factors in
`run_batch_gn` (pointcloud_pose_tracking.py): a factor's Jacobian wrt the
upstream node is the downstream node's "other" Jacobian chained through the
upstream node's own Jacobian in the predicted-pose composition.

A pure dead-reckoning trajectory (sequential noisy-edge composition, no loop
closure) is carried along as the uncorrected baseline, and also doubles as
the optimizer's initial guess -- the same role it plays in
pointcloud_pose_tracking.py.

Poses are represented as (4,4) numpy homogeneous transforms; tangent vectors
follow the [vx,vy,vz,wx,wy,wz] (translation-first) convention used throughout
this codebase.
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt

from lie_utils import se3_exp, se3_log, se3_inv, se3_adjoint, compute_se3_inv_right_jacobian, rotation_geodesic_error


# def skew(v):
#     """
#     Returns the 3x3 skew-symmetric matrix of a 3D vector.
#     Arguments:
#         v: 3D vector (numpy array)
#     Returns:
#         3x3 skew-symmetric matrix (numpy array)
#     """
#     return np.array([
#         [0, -v[2], v[1]],
#         [v[2], 0, -v[0]],
#         [-v[1], v[0], 0],
#     ])


# def se3_exp(xi):
#     """
#     The Exponential Map for SE(3). Maps a 6-vector [v,omega] to a 4x4 matrix.
#     Arguments:
#         xi: 6-vector [v,omega] (numpy array)
#     Returns:
#         4x4 SE(3) transform (numpy array)
#     """
#     v = xi[0:3]
#     omega = xi[3:6]
#     theta = np.linalg.norm(omega)
#     I3 = np.eye(3)
#     T = np.eye(4)

#     if theta < 1e-6:
#         R = I3 + skew(omega)
#         V = I3 + 0.5 * skew(omega)
#     else:
#         omega_skew = skew(omega)
#         omega_skew_sq = np.dot(omega_skew, omega_skew)
#         R = I3 + (np.sin(theta) / theta) * omega_skew + ((1.0 - np.cos(theta)) / (theta ** 2)) * omega_skew_sq
#         V = I3 + ((1.0 - np.cos(theta)) / (theta ** 2)) * omega_skew + ((theta - np.sin(theta)) / (theta ** 3)) * omega_skew_sq

#     T[0:3, 0:3] = R
#     T[0:3, 3] = np.dot(V, v)
#     return T


# def se3_log(T):
#     """
#     The Logarithmic Map for SE(3). Extracts a 6-vector [v,omega] from a 4x4 matrix.
#     Arguments:
#         T: 4x4 SE(3) transform (numpy array)
#     Returns:
#         6-vector [v,omega] (numpy array)
#     """
#     R = T[0:3, 0:3]
#     t = T[0:3, 3]
#     I3 = np.eye(3)

#     cos_theta = (np.trace(R) - 1.0) / 2.0
#     cos_theta = np.clip(cos_theta, -1.0, 1.0)
#     theta = np.arccos(cos_theta)

#     if theta < 1e-6:
#         omega = np.zeros(3)
#         V_inv = I3
#     else:
#         omega_hat = (theta / (2.0 * np.sin(theta))) * (R - R.T)
#         omega = np.array([-omega_hat[1, 2], omega_hat[0, 2], -omega_hat[0, 1]])

#         omega_skew = skew(omega)
#         omega_skew_sq = np.dot(omega_skew, omega_skew)
#         V_inv = I3 - 0.5 * omega_skew + (1.0 / (theta ** 2) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))) * omega_skew_sq

#     rho = np.dot(V_inv, t)
#     return np.concatenate([rho, omega])


# def se3_inv(T):
#     """
#     Rigid inverse of a 4x4 SE(3) transform.
#     Arguments:
#         T: 4x4 SE(3) transform (numpy array)
#     Returns:
#         4x4 inverted SE(3) transform (numpy array)
#     """
#     R, t = T[0:3, 0:3], T[0:3, 3]
#     T_inv = np.eye(4)
#     T_inv[0:3, 0:3] = R.T
#     T_inv[0:3, 3] = -R.T @ t
#     return T_inv


# def se3_adjoint(T):
#     """
#     The 6x6 SE(3) adjoint, block form [[R, skew(t)@R],[0, R]] for the
#     [v,omega]-ordered tangent convention used throughout this codebase.
#     Arguments:
#         T: 4x4 SE(3) transform (numpy array)
#     Returns:
#         6x6 SE(3) adjoint (numpy array)
#     """
#     R, t = T[0:3, 0:3], T[0:3, 3]
#     Adj = np.zeros((6, 6))
#     Adj[0:3, 0:3] = R
#     Adj[0:3, 3:6] = skew(t) @ R
#     Adj[3:6, 3:6] = R
#     return Adj


# def compute_so3_inv_right_jacobian(theta_vec):
#     """
#     Computes the 3x3 inverse right Jacobian of SO(3).
#     Arguments:
#         theta_vec: 3D vector (numpy array)
#     Returns:
#         3x3 inverse right Jacobian (numpy array)
#     """
#     theta = np.linalg.norm(theta_vec)
#     I3 = np.eye(3)
#     if theta < 1e-6:
#         return I3 + 0.5 * skew(theta_vec) + (1.0 / 12.0) * np.dot(skew(theta_vec), skew(theta_vec))
#     theta_skew = skew(theta_vec)
#     theta_skew_sq = np.dot(theta_skew, theta_skew)
#     coeff = (1.0 / (theta ** 2)) - ((1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta)))
#     return I3 + 0.5 * theta_skew + coeff * theta_skew_sq


# def compute_se3_inv_right_jacobian(error_vector):
#     """
#     Computes the 6x6 analytical inverse right Jacobian for an SE(3) error vector.
#     Arguments:
#         error_vector: 6-vector [rho, theta_vec] (numpy array)
#     Returns:
#         6x6 inverse right Jacobian (numpy array)
#     """
#     rho = error_vector[0:3]
#     theta_vec = error_vector[3:6]
#     theta = np.linalg.norm(theta_vec)

#     J_r_inv_so3 = compute_so3_inv_right_jacobian(theta_vec)

#     if theta < 1e-6:
#         Q = 0.5 * skew(rho)
#     else:
#         theta_skew = skew(theta_vec)
#         theta_skew_sq = np.dot(theta_skew, theta_skew)

#         coeff_2 = (theta - np.sin(theta)) / (theta ** 3)
#         rho_skew = skew(rho)
#         theta_rho_skew = skew(np.cross(theta_vec, rho))

#         coeff_q1 = (theta * np.sin(theta) + 2 * np.cos(theta) - 2) / (2 * theta ** 4 * (np.cos(theta) - 1))
#         if np.isnan(coeff_q1) or np.isinf(coeff_q1):
#             coeff_q1 = -1.0 / 12.0

#         Q = (0.5 * rho_skew +
#              (coeff_2 * (np.dot(theta_skew, rho_skew) + np.dot(rho_skew, theta_skew) + np.dot(theta_skew, np.dot(rho_skew, theta_skew)))) +
#              coeff_q1 * np.dot(theta_skew_sq, theta_rho_skew))

#     J_inv_se3 = np.zeros((6, 6))
#     J_inv_se3[0:3, 0:3] = J_r_inv_so3
#     J_inv_se3[0:3, 3:6] = Q
#     J_inv_se3[3:6, 3:6] = J_r_inv_so3
#     return J_inv_se3

# def se3_right_jacobian(xi):
#     """
#     The direct (non-inverse) SE(3) right Jacobian Jr(xi), obtained by inverting
#     the closed-form Jr_inv(xi) above rather than re-deriving a second formula.
#     Arguments:
#         xi: 6-vector [v,omega] (numpy array)
#     Returns:
#         6x6 SE(3) right Jacobian (numpy array)
#     """
#     return np.linalg.inv(compute_se3_inv_right_jacobian(xi))


# def rotation_geodesic_error(R_a, R_b):
#     """
#     Angle (rad) of the relative rotation between two rotation matrices.
#     Arguments:
#         R_a: first rotation matrix (numpy array)
#         R_b: second rotation matrix (numpy array)
#     Returns:
#         angle: angle (rad) of the relative rotation
#     """
#     c = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
#     return np.arccos(np.clip(c, -1.0, 1.0))


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
        gt_poses: list of 4*nodes_per_side ground-truth poses (4,4)
    """
    corners = [(0.0, 0.0), (side_length, 0.0), (side_length, side_length), (0.0, side_length)]
    gt_poses = []
    for k in range(len(corners)):
        x0, y0 = corners[k]
        x1, y1 = corners[(k + 1) % len(corners)]
        for step in range(nodes_per_side):
            frac = step / nodes_per_side
            T = np.eye(4)
            T[0:3, 3] = [x0 + frac * (x1 - x0), y0 + frac * (y1 - y0), 0.0]
            gt_poses.append(T)
    return gt_poses


def simulate_noisy_edges(gt_poses, pos_noise_std, rot_noise_std, loop_noise_scale, rng):
    """Builds noisy relative-pose constraints: one odometry edge between each
    pair of consecutive nodes, plus one loop-closure edge from the last node
    back to the first.
    Arguments:
        gt_poses: list of ground-truth poses (4,4)
        pos_noise_std: std-dev of Gaussian noise on each edge's translation (m)
        rot_noise_std: std-dev of Gaussian noise on each edge's rotation (rad)
        loop_noise_scale: multiplier applied to the noise std-devs for the loop-closure edge
                           only (a loop-closure detector is usually more precise than raw odometry)
        rng: numpy random number generator
    Returns:
        odometry_constraints: list of (idx_i, idx_j, Z_ij) edges between consecutive nodes
        loop_constraints: list of (idx_i, idx_j, Z_ij) edges closing the loop back to node 0
                           (Z_ij is the measured relative pose, a (4,4) transform)
    """
    def noisy_edge(T_i, T_j, scale):
        Z_true = se3_inv(T_i) @ T_j
        noise = np.concatenate([
            rng.normal(0.0, pos_noise_std * scale, 3),
            rng.normal(0.0, rot_noise_std * scale, 3),
        ])
        return Z_true @ se3_exp(noise)

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
        T_init: anchored pose of node 0 (4,4)
        odometry_constraints: list of (idx_i, idx_j, Z_ij) edges between consecutive nodes
    Returns:
        T_list: list of estimated poses (4,4)
    """
    T_list = [T_init]
    for _, _, Z_ij in odometry_constraints:
        T_list.append(T_list[-1] @ Z_ij)
    return T_list


def run_pose_graph_optimization(T_init_list, constraints, info_matrix, damping, gn_tol, gn_max_iters):
    """Batch Gauss-Newton / Levenberg-Marquardt pose-graph relaxation: jointly
    optimizes every node pose against all edge residuals
    e_ij = Log((Xi @ Z_ij)^-1 @ Xj) = Log(Z_ij^-1 * Xi^-1 * Xj), fixing the
    gauge freedom by heavily anchoring node 0.

    Each edge's Jacobian wrt Xj comes straight from the residual's own
    inverse-right Jacobian (the "self" term); its Jacobian wrt Xi is that
    same residual "other" Jacobian chained through the predicted pose
    T_pred = Xi @ Z_ij's own sensitivity to Xi (the SE(3) adjoint of
    Z_ij^-1) -- exactly the motion-factor Jacobian-chaining pattern used in
    run_batch_gn (pointcloud_pose_tracking.py), generalized from a
    twist-based motion model to a directly-measured relative pose.
    Arguments:
        T_init_list: list of initial pose guesses (4,4)
        constraints: list of (idx_i, idx_j, Z_ij) edges ((4,4) measurements)
        info_matrix: 6x6 information (inverse covariance) matrix shared by every edge (numpy array)
        damping: Levenberg-Marquardt damping added to the normal equations' diagonal
        gn_tol: convergence tolerance on the correction step norm
        gn_max_iters: maximum number of iterations
    Returns:
        T_est: list of optimized poses (4,4)
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

            Jc_self = se3_adjoint(se3_inv(Z_ij))  # d(Xi @ Z_ij) / d Xi
            T_pred = Xi @ Z_ij

            e_vec = se3_log(se3_inv(T_pred) @ Xj)
            Ja = compute_se3_inv_right_jacobian(e_vec)     # d e_ij / d Xj
            Jb = -compute_se3_inv_right_jacobian(-e_vec)   # d e_ij / d T_pred

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
            T_est[k] = T_est[k] @ se3_exp(delta[6 * k:6 * k + 6])

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
        T_true_list: list of true poses (4,4)
        T_est_list: list of estimated poses (4,4)
    Returns:
        rot_err: (N,) array of rotation errors (deg)
        pos_err: (N,) array of position errors (m)
    """
    n = len(T_true_list)
    rot_err, pos_err = np.zeros(n), np.zeros(n)
    for k in range(n):
        rot_err[k] = np.degrees(rotation_geodesic_error(T_true_list[k][0:3, 0:3], T_est_list[k][0:3, 0:3]))
        pos_err[k] = np.linalg.norm(T_true_list[k][0:3, 3] - T_est_list[k][0:3, 3])
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
          f"{np.linalg.norm(T_dr[-1][0:3, 3] - gt_poses[-1][0:3, 3]):.4f} m")

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
        pts = np.array([T[0:3, 3] for T in T_list] + [T_list[0][0:3, 3]])
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
