'''
Robust 3D pose-graph optimization against false loop closures (IRLS inside
Levenberg-Marquardt) using manifpy.

The manifpy twin of use_numpy/pose_graph_robust.py: same scenario (several laps
of the 2 m square, odometry edges, true and false loop closures), same four
approaches (l2, huber, cauchy, dcs), same flags and same printed output; read
that file's docstring for the model, the false-closure generator
(Suenderhauf & Protzel / Vertigo), the kernels and the scores.

What differs here is only the Lie-group layer: poses are manif.SE3, the
residual is e_ij = Xj.rminus(Xi.compose(Z_ij)) = Log(Z_ij^-1 * Xi^-1 * Xj) and
its Jacobians come from manif's analytical `compose` / `rminus` Jacobians,
chained exactly as in use_manif/pose_graph.py (the Jacobian wrt Xj is rminus's
"self" Jacobian; wrt Xi it is rminus's "other" Jacobian chained through
compose's "self" Jacobian). The update is X + SE3Tangent(delta). Everything that
does not touch the pose type (kernels, false-pair drawing, the scene builder, the
Levenberg-Marquardt loop, the experiments) is imported from the numpy module, so
both backends run the same code on the same random draws.
'''

import sys

import numpy as np
import manifpy as manif

from state_estimation_simulations.use_manif.manif_utils import pose_errors
from state_estimation_simulations.use_manif.pose_graph import (
    generate_ground_truth_trajectory, simulate_noisy_edges, run_dead_reckoning)
from state_estimation_simulations.use_numpy.pose_graph_robust import (  # noqa: F401  (re-exported backend-independent parts)
    METHODS, kernel_weight, kernel_cost, kernel_threshold, chi2_quantile_6dof, euler_to_quat, draw_false_event, add_arguments,
    information_matrix, build_scene, solve_robust, classify_loops, pose_rmse, solve_scene, run_single,
    run_monte_carlo, run_scale_sweep, plot_single, run_main)


def noisy_relative_pose(T_i, T_j, pos_std, rot_std, rng):
    """Noisy measurement Z = (Ti^-1 Tj) Exp(noise); same draws as pose_graph.py's simulate_noisy_edges.
    Arguments:
        T_i, T_j: true poses (manif.SE3)
        pos_std: translation noise std-dev (m)
        rot_std: rotation noise std-dev (rad)
        rng: numpy random number generator
    Returns:
        Z: measured relative pose (manif.SE3)
    """
    noise = np.concatenate([rng.normal(0.0, pos_std, 3), rng.normal(0.0, rot_std, 3)])
    return (T_i.inverse() * T_j) + manif.SE3Tangent(noise)


def make_pose(t, quat_wxyz):
    """manif.SE3 from a translation and a (w, x, y, z) unit quaternion.
    Arguments:
        t: translation (3,) (m)
        quat_wxyz: unit quaternion (w, x, y, z)
    Returns:
        manif.SE3
    """
    w, x, y, z = quat_wxyz
    return manif.SE3(np.array([t[0], t[1], t[2], x, y, z, w]))


def edge_residual(Xi, Xj, Z):
    """Residual e_ij = Xj.rminus(Xi.compose(Z)) of one edge.
    Arguments:
        Xi, Xj: current poses (manif.SE3)
        Z: measured relative pose (manif.SE3)
    Returns:
        e_vec: (6,) residual (translation first)
    """
    return Xj.rminus(Xi.compose(Z)).coeffs()


def edge_residual_and_jacobians(Xi, Xj, Z):
    """Residual and its Jacobians wrt right perturbations of Xi and Xj, from manif's compose / rminus
    Jacobians (as in use_manif/pose_graph.py).
    Arguments:
        Xi, Xj: current poses (manif.SE3)
        Z: measured relative pose (manif.SE3)
    Returns:
        e_vec: (6,) residual
        J_i: (6,6) d e / d Xi
        J_j: (6,6) d e / d Xj
    """
    Jc_self = np.zeros((6, 6))
    T_pred = Xi.compose(Z, Jc_self)
    Ja, Jb = np.zeros((6, 6)), np.zeros((6, 6))
    e_vec = Xj.rminus(T_pred, Ja, Jb).coeffs()
    return e_vec, Jb @ Jc_self, Ja


def retract(X, delta):
    """Right-perturbation update X + SE3Tangent(delta).
    Arguments:
        X: pose (manif.SE3)
        delta: (6,) tangent vector
    Returns:
        updated pose (manif.SE3)
    """
    return X + manif.SE3Tangent(delta)


def positions(T_list):
    """Node positions as an (n,3) array (m)."""
    return np.array([T.translation() for T in T_list])


def run_robust_pose_graph_optimization(T_init_list, odometry, loops, info_odo, info_loop, method="l2", tau=3.55,
                                       damping=0.01, gn_tol=1e-6, gn_max_iters=50, record=False):
    """Batch Levenberg-Marquardt pose-graph relaxation with a robust kernel on the loop closures (IRLS),
    manifpy version; see use_numpy/pose_graph_robust.py for the method. With method "l2" and identity
    information matrices it reproduces use_manif/pose_graph.py's run_pose_graph_optimization.
    Arguments:
        T_init_list: list of initial poses (manif.SE3), node 0 is anchored where it is
        odometry: list of (i, j, Z) odometry edges (always squared)
        loops: list of (i, j, Z) loop-closure edges (robust kernel)
        info_odo: (6,6) information matrix of the odometry edges
        info_loop: (6,6) information matrix of the loop-closure edges
        method: "l2", "huber", "cauchy" or "dcs"
        tau: kernel scale (Huber delta, Cauchy k, sqrt of DCS Phi), in standard deviations
        damping: initial Levenberg-Marquardt damping
        gn_tol: stop when the step norm is below this
        gn_max_iters: maximum number of accepted steps
        record: keep the cost, poses and loop weights after every accepted step
    Returns:
        T_est: list of optimized poses (manif.SE3)
        info: dict with iterations, converged, cost, weights (final weight of every loop edge) and history
    """
    return solve_robust(T_init_list, odometry, loops, info_odo, info_loop, method, tau, damping, gn_tol,
                        gn_max_iters, edge_residual, edge_residual_and_jacobians, retract, record)


def main():
    run_main(sys.modules[__name__], __doc__)


if __name__ == "__main__":
    main()
