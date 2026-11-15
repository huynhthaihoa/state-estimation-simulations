'''
Robust 3D pose-graph optimization against false loop closures (IRLS inside
Levenberg-Marquardt).

pose_graph.py relaxes a square loop that has ONE loop closure, too few to show
what a wrong one does. Here the robot drives the same square (side 2 m,
subdivided into nodes) for several laps, so later laps revisit earlier places.
The graph has

  - odometry edges between consecutive nodes (noise 0.05 m / 0.01 rad, as in
    pose_graph.py), always kept squared (L2);
  - TRUE loop closures: every node of lap >= 2 to the node at the same place one
    lap earlier (noise = loop_noise_scale x odometry noise, as in pose_graph.py);
  - FALSE loop closures: a fraction `--outlier-frac` of the true count, between
    node pairs that are NOT at the same place. They follow the outlier
    generator of Suenderhauf & Protzel (Vertigo's generateDataset.py): by
    default the "random" policy (both nodes drawn uniformly, never consecutive),
    a Gaussian relative pose (translation 0.3 per axis, roll/pitch/yaw 10 deg;
    the numbers of that script, applied unchanged) and the same information
    matrix as the true closures. "local", "grouped" and a "perfect match"
    (identity) measurement are labeled variants, see the flags.

Each loop closure has a residual r = Log(Z^-1 Xi^-1 Xj) (as pose_graph.py) and
a scalar Mahalanobis residual e = sqrt(r^T Omega r), Omega the edge's
information matrix (inverse of the noise covariance, so for a true closure at
the truth e^2 follows a chi-squared law with 6 degrees of freedom). Four
approaches minimize  sum_odometry 1/2 r^T Omega r + sum_loops rho(e):

  l2      rho = 1/2 e^2 (plain least squares, what pose_graph.py does)
  huber   quadratic up to e = tau, linear beyond; weight tau / e
  cauchy  logarithmic tail; weight 1 / (1 + (e/tau)^2)
  dcs     dynamic covariance scaling (Agarwal et al. 2013): information matrix
          scaled by s^2, s = min(1, 2 Phi / (Phi + e^2)), Phi = tau^2

Each is solved by iteratively reweighted least squares: at the start of every
accepted-step iteration each loop edge's information matrix is multiplied by
its weight w(e) at the current estimate; the Levenberg-Marquardt step
accept/reject then uses the matching robust cost sum rho(e), not the squared
cost. The Jacobians, the anchoring of node 0 (gauge) and the adaptive damping
follow run_pose_graph_optimization of pose_graph.py. The cost whose IRLS weight
is exactly the DCS weight s^2 is rho(e) = 1/2 [3 Phi - 4 Phi^2 / (Phi + e^2)]
for e^2 > Phi (obtained by integrating the weight; a Geman-McClure-shaped
bounded cost), which is what the accept/reject uses for dcs.

Kernel scale tau (design choice, not from the sources): e^2 = tau^2 is the
`--kernel-quantile` quantile (default 0.95) of the chi-squared law with 6
degrees of freedom, tau = 3.55; Huber's delta and Cauchy's k equal tau and DCS's
Phi equals tau^2 = 12.59. `--kernel-tau` sets tau directly (Agarwal et al.'s
default Phi = 1 is `--kernel-tau 1`); `--kernel-scale-mult` multiplies tau.

Experiments (`--experiment`):
  single       one run (seed `--seed`) at `--single-outlier-frac`: table plus figure
  montecarlo   `--n-seeds` seeds x `--outlier-fracs` x methods: position and
               rotation RMSE over all nodes (node 0 is anchored at the truth, so
               no alignment is needed), iterations and loop-edge
               classification (false closures with final weight < 0.1 are
               "rejected", true closures with weight >= 0.5 are "kept")
  sweep        kernel scale multipliers at one outlier fraction
  diagnostics  not in "all": the residual e of true and false closures at the
               dead-reckoning start, and the l2 position RMSE per lap without outliers
  all          single + montecarlo + sweep (the default run)

Every Monte Carlo trial redraws the odometry noise, the true-closure noise and
the false closures; their random streams are spawned from the trial seed
(numpy SeedSequence), so the data of a trial do not change with the outlier
fraction or the method, and the false closures at a smaller fraction are the
first ones of those at a larger fraction.

This is not a filter, so there is no NEES; the scores are RMSE and the
classification counts. Poses are (4,4) numpy transforms; tangent vectors are
[vx,vy,vz,wx,wy,wz] (translation first). use_manif/pose_graph_robust.py is the
manifpy twin (same flags and output); it reuses the backend-independent parts
of this file.
'''

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

from state_estimation_simulations.use_numpy.lie_utils import (
    se3_exp, se3_log, se3_inv, se3_adjoint, compute_se3_inv_right_jacobian)
from state_estimation_simulations.use_numpy.numpy_utils import pose_errors
from state_estimation_simulations.use_numpy.pose_graph import (
    generate_ground_truth_trajectory, simulate_noisy_edges, run_dead_reckoning)

METHODS = ("l2", "huber", "cauchy", "dcs")
ANCHOR_WEIGHT = 1e6          # node-0 gauge anchor, times the largest information entry (1e6 for identity, as in pose_graph.py)
REJECT_BELOW = 0.1           # false closure counted "rejected" if final weight < this
KEEP_FROM = 0.5              # true closure counted "kept" if final weight >= this


# --------------------------------------------------------------------------- kernels
def chi2_quantile_6dof(p):
    """Quantile of the chi-squared law with 6 degrees of freedom (closed-form CDF,
    inverted by bisection; no scipy needed).
    Arguments:
        p: probability in (0, 1)
    Returns:
        x such that P(chi2_6 <= x) = p
    """
    def cdf(x):
        return 1.0 - np.exp(-x / 2.0) * (1.0 + x / 2.0 + x * x / 8.0)
    lo, hi = 0.0, 100.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if cdf(mid) < p else (lo, mid)
    return 0.5 * (lo + hi)


def kernel_threshold(quantile, mult=1.0):
    """Kernel scale tau (in units of the scalar Mahalanobis residual e): the square
    root of the chi-squared(6) quantile, times `mult`. Huber's delta and Cauchy's k
    equal tau; DCS's Phi equals tau^2.
    Arguments:
        quantile: chi-squared(6) probability (e.g. 0.95)
        mult: multiplier on tau (1 = the quantile itself)
    Returns:
        tau (dimensionless, in units of standard deviations)
    """
    return mult * np.sqrt(chi2_quantile_6dof(quantile))


def base_tau(p):
    """Kernel scale of a run: `--kernel-tau` if given, else the `--kernel-quantile` chi-squared(6) quantile's
    square root, times `--kernel-scale-mult`.
    Arguments:
        p: parameter namespace
    Returns:
        tau (dimensionless, in standard deviations)
    """
    if p.kernel_tau is not None:
        return p.kernel_tau * p.kernel_scale_mult
    return kernel_threshold(p.kernel_quantile, p.kernel_scale_mult)


def kernel_weight(method, e, tau):
    """IRLS weight w(e) = (1/e) d rho / d e applied to a loop edge's information matrix
    (rho in the 1/2 convention of docs/optimization/pose_graph_optimization.md section 16).
    Arguments:
        method: "l2", "huber", "cauchy" or "dcs"
        e: scalar Mahalanobis residual(s) sqrt(r^T Omega r) (float or array)
        tau: kernel scale (Huber delta, Cauchy k, sqrt(DCS Phi)); ignored for l2
    Returns:
        weight(s) in (0, 1], same shape as e
    """
    e = np.asarray(e, dtype=float)
    if method == "l2":
        return np.ones_like(e)
    if method == "huber":
        return np.where(e <= tau, 1.0, tau / np.maximum(e, 1e-300))
    if method == "cauchy":
        return 1.0 / (1.0 + (e / tau) ** 2)
    if method == "dcs":
        phi = tau ** 2
        s = np.minimum(1.0, 2.0 * phi / (phi + e ** 2))
        return s ** 2
    raise ValueError(f"unknown method {method!r}")


def kernel_cost(method, e, tau):
    """Robust cost rho(e) of one loop edge (1/2 convention), the function whose IRLS weight is
    kernel_weight. For dcs it is the integral of the DCS weight s^2 (see the module docstring).
    Arguments:
        method: "l2", "huber", "cauchy" or "dcs"
        e: scalar Mahalanobis residual(s) (float or array)
        tau: kernel scale
    Returns:
        cost(s), same shape as e
    """
    e = np.asarray(e, dtype=float)
    if method == "l2":
        return 0.5 * e ** 2
    if method == "huber":
        return np.where(e <= tau, 0.5 * e ** 2, tau * (e - 0.5 * tau))
    if method == "cauchy":
        return 0.5 * tau ** 2 * np.log1p((e / tau) ** 2)
    if method == "dcs":
        phi = tau ** 2
        t = e ** 2
        return np.where(t <= phi, 0.5 * t, 0.5 * (3.0 * phi - 4.0 * phi ** 2 / (phi + t)))
    raise ValueError(f"unknown method {method!r}")


# --------------------------------------------------------------------------- graph generation
def euler_to_quat(roll, pitch, yaw):
    """Quaternion of R = Rz(yaw) Ry(pitch) Rx(roll), same formula as Vertigo's
    generateDataset.py euler_to_quat.
    Arguments:
        roll, pitch, yaw: angles (rad)
    Returns:
        (w, x, y, z) unit quaternion
    """
    sy, cy = np.sin(yaw * 0.5), np.cos(yaw * 0.5)
    sp, cp = np.sin(pitch * 0.5), np.cos(pitch * 0.5)
    sr, cr = np.sin(roll * 0.5), np.cos(roll * 0.5)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def make_pose(t, quat_wxyz):
    """(4,4) transform from a translation and a (w, x, y, z) unit quaternion.
    Arguments:
        t: translation (3,) (m)
        quat_wxyz: unit quaternion (w, x, y, z)
    Returns:
        T: (4,4) homogeneous transform
    """
    w, x, y, z = quat_wxyz
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


def noisy_relative_pose(T_i, T_j, pos_std, rot_std, rng):
    """Noisy measurement Z = (Ti^-1 Tj) Exp(noise), the noise draw of pose_graph.py's simulate_noisy_edges
    (translation 3 draws, then rotation 3 draws).
    Arguments:
        T_i, T_j: true poses (4,4)
        pos_std: translation noise std-dev (m)
        rot_std: rotation noise std-dev (rad)
        rng: numpy random number generator
    Returns:
        Z: measured relative pose (4,4)
    """
    noise = np.concatenate([rng.normal(0.0, pos_std, 3), rng.normal(0.0, rot_std, 3)])
    return se3_inv(T_i) @ T_j @ se3_exp(noise)


def draw_false_event(n_nodes, policy, group_size, window, nodes_per_lap, rng):
    """Node pair of one false-closure event, following Vertigo's generateDataset.py:
    "random": both indices uniform over the nodes; "local": the second index within
    `window` nodes after the first (the script's -l option); the pair is sorted and never
    consecutive (j == i + 1 becomes i + 2). With groups, (i + k, j + k) for k < group_size are
    consecutive pairs that share one measurement. Extra rule of this file: a pair at the same
    place (j - i a multiple of nodes_per_lap) is redrawn, so every drawn edge is false.
    Arguments:
        n_nodes: number of nodes
        policy: "random" or "local"
        group_size: edges per group (1 = no grouping)
        window: local window in nodes (the script uses 20)
        nodes_per_lap: nodes in one lap (same-place pairs differ by a multiple of it)
        rng: numpy random number generator
    Returns:
        (i, j): node indices of the first edge of the group, i < j
    """
    hi = n_nodes - 1 - group_size
    while True:
        v1 = int(rng.integers(0, hi + 1))
        v2 = int(rng.integers(0, hi + 1)) if policy == "random" else int(rng.integers(v1, min(hi, v1 + window) + 1))
        v1, v2 = min(v1, v2), max(v1, v2)
        if v2 == v1 + 1:
            v2 = v1 + 2
        if v1 == v2 or v2 + group_size - 1 > n_nodes - 1 or (v2 - v1) % nodes_per_lap == 0:
            continue
        return v1, v2


def information_matrix(pos_std, rot_std):
    """Information matrix (inverse covariance) of an edge with the given noise std-devs.
    Arguments:
        pos_std: translation std-dev (m)
        rot_std: rotation std-dev (rad)
    Returns:
        Omega: (6,6) diagonal matrix, translation first
    """
    return np.diag([1.0 / pos_std ** 2] * 3 + [1.0 / rot_std ** 2] * 3)


def build_scene(be, p, trial_seed, outlier_frac):
    """One simulated graph: multi-lap truth, odometry, true and false loop closures, dead reckoning.
    The odometry, true-closure and false-closure random streams are spawned from the trial seed,
    so changing `outlier_frac` changes nothing but the number of false closures.
    Arguments:
        be: backend module (this module or use_manif.pose_graph_robust)
        p: namespace with side_length, nodes_per_side, n_laps, pos_noise_std, rot_noise_std,
           loop_noise_scale, loop_stride, outlier_policy, outlier_meas, outlier_group, outlier_window,
           outlier_pos_std, outlier_rot_std_deg
        trial_seed: integer seed of this trial
        outlier_frac: number of false closures as a fraction of the number of true closures
    Returns:
        dict with gt, nodes_per_lap, odometry, true_loops, false_loops, loops, is_false (bool array),
        T_dr, info_odo, info_loop
    """
    gt_lap = be.generate_ground_truth_trajectory(p.side_length, p.nodes_per_side)
    L = len(gt_lap)
    gt = gt_lap * p.n_laps
    n = len(gt)
    ss_odo, ss_true, ss_false = np.random.SeedSequence(trial_seed).spawn(3)
    rng_odo, rng_true, rng_false = (np.random.default_rng(s) for s in (ss_odo, ss_true, ss_false))

    odometry = be.simulate_noisy_edges(gt, p.pos_noise_std, p.rot_noise_std, p.loop_noise_scale, rng_odo)[0]
    pos_l, rot_l = p.pos_noise_std * p.loop_noise_scale, p.rot_noise_std * p.loop_noise_scale
    true_loops = [(k - L, k, be.noisy_relative_pose(gt[k - L], gt[k], pos_l, rot_l, rng_true))
                  for k in range(L, n) if k % p.loop_stride == 0]

    n_false = int(round(outlier_frac * len(true_loops)))
    false_loops = []
    while len(false_loops) < n_false:      # one event = pair, then measurement, then group_size edges (as in the script)
        i, j = draw_false_event(n, p.outlier_policy, p.outlier_group, p.outlier_window, L, rng_false)
        if p.outlier_meas == "identity":
            Z = be.make_pose(np.zeros(3), (1.0, 0.0, 0.0, 0.0))
        else:
            t = rng_false.normal(0.0, p.outlier_pos_std, 3)
            rpy = rng_false.normal(0.0, np.radians(p.outlier_rot_std_deg), 3)
            Z = be.make_pose(t, euler_to_quat(*rpy))
        false_loops += [(i + k, j + k, Z) for k in range(p.outlier_group)]
    false_loops = false_loops[:n_false]

    loops = true_loops + false_loops
    is_false = np.array([False] * len(true_loops) + [True] * len(false_loops), dtype=bool)
    return dict(gt=gt, nodes_per_lap=L, odometry=odometry, true_loops=true_loops, false_loops=false_loops,
                loops=loops, is_false=is_false, T_dr=be.run_dead_reckoning(gt[0], odometry),
                info_odo=information_matrix(p.pos_noise_std, p.rot_noise_std),
                info_loop=information_matrix(pos_l, rot_l))


# --------------------------------------------------------------------------- SE(3) residuals (numpy backend)
def edge_residual(Xi, Xj, Z):
    """Residual e_ij = Log((Xi Z)^-1 Xj) of one edge.
    Arguments:
        Xi, Xj: current poses (4,4)
        Z: measured relative pose (4,4)
    Returns:
        e_vec: (6,) residual (translation first)
    """
    return se3_log(se3_inv(Xi @ Z) @ Xj)


def edge_residual_and_jacobians(Xi, Xj, Z):
    """Residual and its Jacobians wrt right perturbations of Xi and Xj. Same chain as
    pose_graph.py's run_pose_graph_optimization (d e / d T_pred = -invJr(-e), chained through
    Ad(Z^-1)); -invJr(-e) is evaluated as -invJr(e) Ad((T_pred^-1 Xj)^-1), using Jl(e) = Ad(Exp e) Jr(e),
    which saves one matrix inversion per edge (the test compares it with the original form).
    Arguments:
        Xi, Xj: current poses (4,4)
        Z: measured relative pose (4,4)
    Returns:
        e_vec: (6,) residual
        J_i: (6,6) d e / d Xi
        J_j: (6,6) d e / d Xj
    """
    T_pred = Xi @ Z
    T_err = se3_inv(T_pred) @ Xj
    e_vec = se3_log(T_err)
    Ja = compute_se3_inv_right_jacobian(e_vec)          # d e / d Xj
    Jb = -Ja @ se3_adjoint(se3_inv(T_err))              # d e / d T_pred
    return e_vec, Jb @ se3_adjoint(se3_inv(Z)), Ja


def retract(X, delta):
    """Right-perturbation update X Exp(delta).
    Arguments:
        X: pose (4,4)
        delta: (6,) tangent vector
    Returns:
        updated pose (4,4)
    """
    return X @ se3_exp(delta)


def positions(T_list):
    """Node positions as an (n,3) array (m)."""
    return np.array([T[0:3, 3] for T in T_list])


# --------------------------------------------------------------------------- the solver
def solve_robust(T_init_list, odometry, loops, info_odo, info_loop, method, tau, damping, gn_tol,
                 gn_max_iters, residual_fn, jacobian_fn, retract_fn, record=False):
    """Backend-independent IRLS + Levenberg-Marquardt loop (the numpy and manifpy modules pass
    their own residual, Jacobian and retraction functions).
    See run_robust_pose_graph_optimization for the arguments and the return value."""
    n_poses = len(T_init_list)
    dof = 6 * n_poses
    anchor = ANCHOR_WEIGHT * max(info_odo.max(), info_loop.max())   # = 1e6 for identity information, as in pose_graph.py
    edges = [(i, j, Z, info_odo, False) for i, j, Z in odometry] + [(i, j, Z, info_loop, True) for i, j, Z in loops]

    def mahalanobis(e_vec, info):
        return float(np.sqrt(e_vec @ info @ e_vec))

    def total_cost(T_list):
        total = 0.0
        for i, j, Z, info, is_loop in edges:
            e_vec = residual_fn(T_list[i], T_list[j], Z)
            e = mahalanobis(e_vec, info)
            total += float(kernel_cost(method, e, tau)) if is_loop else 0.5 * e * e
        return total

    def loop_weights(T_list):
        return np.array([float(kernel_weight(method, mahalanobis(residual_fn(T_list[i], T_list[j], Z), info_loop), tau))
                         for i, j, Z in loops])

    T_est = list(T_init_list)
    cur_cost = total_cost(T_est)
    lam = damping
    history = dict(cost=[cur_cost], poses=[list(T_est)], weights=[loop_weights(T_est)]) if record else None
    n_iter, converged = 0, False

    for _ in range(gn_max_iters):
        H = np.zeros((dof, dof))
        g = np.zeros(dof)
        H[0:6, 0:6] += np.eye(6) * anchor      # anchor node 0 (gauge freedom)
        for i, j, Z, info, is_loop in edges:
            e_vec, J_i, J_j = jacobian_fn(T_est[i], T_est[j], Z)
            w = float(kernel_weight(method, mahalanobis(e_vec, info), tau)) if is_loop else 1.0
            W = w * info
            c0, c1 = 6 * i, 6 * j
            H[c0:c0 + 6, c0:c0 + 6] += J_i.T @ W @ J_i
            H[c1:c1 + 6, c1:c1 + 6] += J_j.T @ W @ J_j
            H[c0:c0 + 6, c1:c1 + 6] += J_i.T @ W @ J_j
            H[c1:c1 + 6, c0:c0 + 6] += J_j.T @ W @ J_i
            g[c0:c0 + 6] += -J_i.T @ W @ e_vec
            g[c1:c1 + 6] += -J_j.T @ W @ e_vec
        diag_H = np.diag(H).copy()
        diag_H[diag_H < 1e-12] = 1e-12

        accepted = False
        delta = np.zeros(dof)
        for _retry in range(10):
            delta = np.linalg.solve(H + lam * np.diag(diag_H), g)
            T_trial = [retract_fn(T_est[k], delta[6 * k:6 * k + 6]) for k in range(n_poses)]
            trial_cost = total_cost(T_trial)
            if trial_cost < cur_cost:
                T_est, cur_cost = T_trial, trial_cost
                lam = max(lam * 0.5, 1e-7)
                accepted = True
                break
            lam = max(lam * 2.0, 1e-6)
        if not accepted:
            break
        n_iter += 1
        if record:
            history["cost"].append(cur_cost)
            history["poses"].append(list(T_est))
            history["weights"].append(loop_weights(T_est))
        if np.linalg.norm(delta) < gn_tol:
            converged = True
            break

    info = dict(iterations=n_iter, converged=converged, cost=cur_cost, weights=loop_weights(T_est), history=history)
    return T_est, info


def run_robust_pose_graph_optimization(T_init_list, odometry, loops, info_odo, info_loop, method="l2", tau=3.55,
                                       damping=0.01, gn_tol=1e-6, gn_max_iters=50, record=False):
    """Batch Levenberg-Marquardt pose-graph relaxation with a robust kernel on the loop closures
    (IRLS): the same Jacobians, node-0 anchoring and adaptive damping as run_pose_graph_optimization
    in pose_graph.py, with a weight w(e) on every loop edge's information matrix and a step
    accept/reject on the matching robust cost sum_odometry 1/2 e^2 + sum_loops rho(e).
    With method "l2" and identity information matrices it reproduces run_pose_graph_optimization.
    Arguments:
        T_init_list: list of initial poses (4,4), node 0 is anchored where it is
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
        T_est: list of optimized poses (4,4)
        info: dict with iterations (accepted steps), converged (bool), cost (final robust cost),
              weights (final weight of every loop edge, at T_est) and history (None unless record)
    """
    return solve_robust(T_init_list, odometry, loops, info_odo, info_loop, method, tau, damping, gn_tol,
                        gn_max_iters, edge_residual, edge_residual_and_jacobians, retract, record)


# --------------------------------------------------------------------------- scores and experiments
def classify_loops(weights, is_false):
    """Loop-edge classification at convergence.
    Arguments:
        weights: final weight of every loop edge
        is_false: bool array, True for false closures
    Returns:
        (rejected_false, n_false, kept_true, n_true): false closures with weight < REJECT_BELOW, their number,
        true closures with weight >= KEEP_FROM, their number
    """
    weights = np.asarray(weights)
    return (int(np.sum(weights[is_false] < REJECT_BELOW)), int(is_false.sum()),
            int(np.sum(weights[~is_false] >= KEEP_FROM)), int((~is_false).sum()))


def pose_rmse(be, gt, T_est):
    """RMSE over all nodes of the position error (m) and of the rotation geodesic error (deg).
    Node 0 is anchored at the truth, so no trajectory alignment is applied.
    Arguments:
        be: backend module
        gt, T_est: lists of true and estimated poses
    Returns:
        (position RMSE (m), rotation RMSE (deg))
    """
    rot_err, pos_err = be.pose_errors(gt, T_est)
    return float(np.sqrt(np.mean(pos_err ** 2))), float(np.sqrt(np.mean(rot_err ** 2)))


def solve_scene(be, scene, method, tau, p, record=False):
    """Run one method on a scene and score it.
    Arguments:
        be: backend module
        scene: dict from build_scene
        method: one of METHODS
        tau: kernel scale
        p: namespace with damping, gn_tol, gn_max_iters
        record: keep the per-iteration history
    Returns:
        dict with T, info, pos_rmse (m), rot_rmse (deg), iterations, time (s), classification
        (rejected_false, n_false, kept_true, n_true)
    """
    t0 = time.perf_counter()
    T, info = be.run_robust_pose_graph_optimization(
        scene["T_dr"], scene["odometry"], scene["loops"], scene["info_odo"], scene["info_loop"], method, tau,
        p.damping, p.gn_tol, p.gn_max_iters, record)
    dt = time.perf_counter() - t0
    pos, rot = pose_rmse(be, scene["gt"], T)
    return dict(T=T, info=info, pos_rmse=pos, rot_rmse=rot, iterations=info["iterations"], time=dt,
                cls=classify_loops(info["weights"], scene["is_false"]))


def method_taus(p, mult=1.0):
    """Kernel scale of every method for a multiplier (l2 ignores it)."""
    return {m: base_tau(p) * mult for m in METHODS}


def stats(x):
    """Mean, median and worst (max) of a list."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.median(x)), float(np.max(x))


def print_setup(p, scene, tau):
    """One-line setup summary."""
    print(f"Setup: {len(scene['gt'])} nodes ({p.n_laps} laps x {scene['nodes_per_lap']}), "
          f"{len(scene['odometry'])} odometry edges, {len(scene['true_loops'])} true + {len(scene['false_loops'])} false loop closures "
          f"(policy={p.outlier_policy}, meas={p.outlier_meas}, group={p.outlier_group}); noise {p.pos_noise_std} m / {p.rot_noise_std} rad, "
          f"loop scale {p.loop_noise_scale}; kernel tau={tau:.3f} (Phi={tau ** 2:.2f})")


def run_single(be, p, verbose=True):
    """One run at p.single_outlier_frac with every method; prints the table.
    Arguments:
        be: backend module
        p: parameter namespace (see main)
        verbose: print the table
    Returns:
        (scene, results dict method -> solve_scene result)
    """
    scene = be.build_scene(be, p, p.seed, p.single_outlier_frac)
    taus = method_taus(p)
    results = {m: solve_scene(be, scene, m, taus[m], p, record=True) for m in METHODS}
    if verbose:
        print_setup(p, scene, taus["huber"])
        dr_pos, dr_rot = pose_rmse(be, scene["gt"], scene["T_dr"])
        print(f"\nSingle run (seed {p.seed}, outlier fraction {p.single_outlier_frac}):")
        print(f"  {'method':<8s} {'pos RMSE (m)':>12s} {'rot RMSE (deg)':>14s} {'iters':>6s} {'false rejected':>15s} {'true kept':>10s} {'max false w':>12s} {'time (s)':>9s}")
        print(f"  {'dead-reck':<8s} {dr_pos:12.4f} {dr_rot:14.3f}")
        for m in METHODS:
            r = results[m]
            rf, nf, kt, nt = r["cls"]
            max_fw = float(np.max(r["info"]["weights"][scene["is_false"]])) if nf else float("nan")
            print(f"  {m:<8s} {r['pos_rmse']:12.4f} {r['rot_rmse']:14.3f} {r['iterations']:6d} {rf:>10d}/{nf:<4d} {kt:>5d}/{nt:<4d} {max_fw:12.2e} {r['time']:9.2f}")
    return scene, results


def run_monte_carlo(be, p, verbose=True):
    """Monte Carlo over seeds x outlier fractions x methods.
    Arguments:
        be: backend module
        p: parameter namespace
        verbose: print the tables
    Returns:
        rows: dict (frac, method) -> dict of lists (pos, rot, iters, rej, nfalse, kept, ntrue, time)
        dr: dict frac -> list of dead-reckoning position RMSE per seed (identical across fractions by construction)
    """
    taus = method_taus(p)
    rows = {(f, m): dict(pos=[], rot=[], iters=[], rej=[], nfalse=[], kept=[], ntrue=[], time=[]) for f in p.outlier_fracs for m in METHODS}
    dr = {f: [] for f in p.outlier_fracs}
    for f in p.outlier_fracs:
        for t in range(p.n_seeds):
            scene = be.build_scene(be, p, p.seed + t, f)
            dr[f].append(pose_rmse(be, scene["gt"], scene["T_dr"])[0])
            for m in METHODS:
                r = solve_scene(be, scene, m, taus[m], p)
                d = rows[(f, m)]
                d["pos"].append(r["pos_rmse"]); d["rot"].append(r["rot_rmse"]); d["iters"].append(r["iterations"]); d["time"].append(r["time"])
                rf, nf, kt, nt = r["cls"]
                d["rej"].append(rf); d["nfalse"].append(nf); d["kept"].append(kt); d["ntrue"].append(nt)
    if verbose:
        same = all(np.array_equal(dr[p.outlier_fracs[0]], dr[f]) for f in p.outlier_fracs)
        print(f"\nMonte Carlo: {p.n_seeds} seeds (seed {p.seed}..{p.seed + p.n_seeds - 1}), tau={taus['huber']:.3f}; "
              f"check, dead-reckoning RMSE identical across outlier fractions: {same}")
        print(f"  dead reckoning (all fractions): pos RMSE mean {np.mean(dr[p.outlier_fracs[0]]):.4f} m")
        print(f"  {'frac':>5s} {'method':<7s} | {'pos RMSE (m): mean':>18s} {'median':>8s} {'worst':>8s} | {'rot RMSE (deg): mean':>20s} {'median':>8s} {'worst':>8s} | "
              f"{'iters':>6s} | {'false rejected':>14s} {'true kept':>10s} | {'time (s)':>8s}")
        for f in p.outlier_fracs:
            for m in METHODS:
                d = rows[(f, m)]
                pm, pmed, pmax = stats(d["pos"])
                rm, rmed, rmax = stats(d["rot"])
                rej = f"{np.sum(d['rej'])}/{np.sum(d['nfalse'])}" if np.sum(d["nfalse"]) else "-"
                print(f"  {f:5.2f} {m:<7s} | {pm:18.4f} {pmed:8.4f} {pmax:8.4f} | {rm:20.3f} {rmed:8.3f} {rmax:8.3f} | "
                      f"{np.mean(d['iters']):6.1f} | {rej:>14s} {np.sum(d['kept'])}/{np.sum(d['ntrue']):<5d} | {np.mean(d['time']):8.2f}")
    return rows, dr


def run_scale_sweep(be, p, verbose=True):
    """Kernel-scale sweep at p.sweep_outlier_frac: the three robust kernels at tau x each multiplier,
    plus DCS with Phi = 1 (the value Agarwal et al. use unless stated otherwise), on the same scenes
    (same seeds) for every setting; l2 shown once for reference.
    Arguments:
        be: backend module
        p: parameter namespace
        verbose: print the table
    Returns:
        rows: dict (method, label) -> dict with tau and lists pos, rot, iters, rej, nfalse, kept, ntrue
    """
    configs = [("l2", "-", float("nan"))]
    for m in METHODS[1:]:
        configs += [(m, f"x{k:g}", base_tau(p) * k) for k in p.sweep_mults]
    configs.append(("dcs", "Phi=1", 1.0))
    rows = {(m, lab): dict(tau=tau, pos=[], rot=[], iters=[], rej=[], nfalse=[], kept=[], ntrue=[]) for m, lab, tau in configs}
    for t in range(p.n_sweep_seeds):
        scene = be.build_scene(be, p, p.seed + t, p.sweep_outlier_frac)
        for m, lab, tau in configs:
            r = solve_scene(be, scene, m, tau, p)
            d = rows[(m, lab)]
            d["pos"].append(r["pos_rmse"]); d["rot"].append(r["rot_rmse"]); d["iters"].append(r["iterations"])
            rf, nf, kt, nt = r["cls"]
            d["rej"].append(rf); d["nfalse"].append(nf); d["kept"].append(kt); d["ntrue"].append(nt)
    if verbose:
        print(f"\nKernel-scale sweep: outlier fraction {p.sweep_outlier_frac}, {p.n_sweep_seeds} seeds (seed {p.seed}..{p.seed + p.n_sweep_seeds - 1}); "
              f"x1 is tau = {base_tau(p):.3f} (Phi = {base_tau(p) ** 2:.2f}):")
        print(f"  {'method':<7s} {'scale':>6s} {'tau':>6s} | {'pos RMSE (m): mean':>18s} {'median':>8s} {'worst':>8s} | {'rot RMSE (deg) mean':>19s} | {'iters':>6s} | {'false rejected':>14s} {'true kept':>10s}")
        for (m, lab), d in rows.items():
            pm, pmed, pmax = stats(d["pos"])
            print(f"  {m:<7s} {lab:>6s} {d['tau']:6.2f} | {pm:18.4f} {pmed:8.4f} {pmax:8.4f} | {np.mean(d['rot']):19.3f} | {np.mean(d['iters']):6.1f} | "
                  f"{np.sum(d['rej'])}/{np.sum(d['nfalse']):<12d} {np.sum(d['kept'])}/{np.sum(d['ntrue'])}")
    return rows


def run_diagnostics(be, p, verbose=True):
    """Where the numbers of the doc come from: the scalar residual e of the loop closures at the dead-reckoning
    start (true vs false) and at the l2 solution of the outlier-free graph, and the l2 position RMSE per lap of
    the outlier-free graph. Seeds p.seed .. p.seed + p.n_seeds - 1; the false closures use p.single_outlier_frac.
    Arguments:
        be: backend module
        p: parameter namespace
        verbose: print the summary
    Returns:
        dict with e_true_start, e_false_start, e_true_l2 (arrays) and lap_rmse (list of per-lap position RMSE, m)
    """
    tau = base_tau(p)
    e_true_start, e_false_start, e_true_l2, lap_sq = [], [], [], []
    for t in range(p.n_seeds):
        scene = be.build_scene(be, p, p.seed + t, p.single_outlier_frac)
        for (i, j, Z), is_f in zip(scene["loops"], scene["is_false"]):
            r = be.edge_residual(scene["T_dr"][i], scene["T_dr"][j], Z)
            (e_false_start if is_f else e_true_start).append(float(np.sqrt(r @ scene["info_loop"] @ r)))
        clean = be.build_scene(be, p, p.seed + t, 0.0)
        T, _ = be.run_robust_pose_graph_optimization(clean["T_dr"], clean["odometry"], clean["loops"], clean["info_odo"],
                                                     clean["info_loop"], "l2", tau, p.damping, p.gn_tol, p.gn_max_iters)
        for (i, j, Z) in clean["loops"]:
            r = be.edge_residual(T[i], T[j], Z)
            e_true_l2.append(float(np.sqrt(r @ clean["info_loop"] @ r)))
        _, pos = be.pose_errors(clean["gt"], T)
        L = clean["nodes_per_lap"]
        lap_sq.append([float(np.mean(pos[k * L:(k + 1) * L] ** 2)) for k in range(p.n_laps)])
    out = dict(e_true_start=np.array(e_true_start), e_false_start=np.array(e_false_start), e_true_l2=np.array(e_true_l2),
               lap_rmse=[float(np.sqrt(np.mean(np.array(lap_sq)[:, k]))) for k in range(p.n_laps)])
    if verbose:
        print(f"\nDiagnostics: seeds {p.seed}..{p.seed + p.n_seeds - 1}, tau={tau:.3f}; e = sqrt(r^T Omega r) of the loop closures")
        for name, key in (("true closures at the dead-reckoning start", "e_true_start"),
                          ("false closures at the dead-reckoning start", "e_false_start"),
                          ("true closures at the l2 solution of the outlier-free graph", "e_true_l2")):
            e = out[key]
            print(f"  {name}: n={len(e)}, median {np.median(e):.1f}, min {e.min():.1f}, max {e.max():.1f}, share with e <= tau {np.mean(e <= tau):.2f}")
        print("  l2 position RMSE per lap on the outlier-free graph (m): " + ", ".join(f"{v:.4f}" for v in out["lap_rmse"]))
    return out


def plot_single(be, p, scene, results, out):
    """Figure of the single run: top views of the l2 and the best robust estimate, loop weights per edge.
    Arguments:
        be: backend module
        p: parameter namespace
        scene: dict from build_scene
        results: dict method -> solve_scene result
        out: output path (None = show)
    Returns:
        None
    """
    robust = [m for m in METHODS if m != "l2"]
    best = min(robust, key=lambda m: results[m]["pos_rmse"])
    gt_xy = be.positions(scene["gt"])[:, 0:2]
    dr_xy = be.positions(scene["T_dr"])[:, 0:2]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    for ax, m in zip(axes[:2], ("l2", best)):
        est_xy = be.positions(results[m]["T"])[:, 0:2]
        w = results[m]["info"]["weights"]
        ax.plot(*gt_xy[:p.nodes_per_side * 4 + 1].T, color="black", lw=2.5, label="truth (one lap)", zorder=1)
        ax.plot(*dr_xy.T, color="tab:gray", lw=1, ls=":", label="dead reckoning", zorder=1)
        ax.plot(*est_xy.T, color="tab:blue", lw=1.6, marker="o", ms=3, label=f"{m} estimate", zorder=3)
        for k, (i, j, _) in enumerate(scene["loops"]):
            if scene["is_false"][k]:
                ax.plot([est_xy[i, 0], est_xy[j, 0]], [est_xy[i, 1], est_xy[j, 1]], color="tab:red", ls="--",
                        lw=1.0 + 2.0 * w[k], alpha=0.35 + 0.65 * w[k], zorder=2)
        ax.plot([], [], color="tab:red", ls="--", label="false closure (width = weight)")
        r = results[m]
        ax.set_title(f"{m}: position RMSE {r['pos_rmse']:.3f} m, {r['iterations']} iterations")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.axis("equal")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax = axes[2]
    idx = np.arange(len(scene["loops"]))
    colors = dict(huber="tab:orange", cauchy="tab:green", dcs="tab:purple")
    for m in robust:
        w = results[m]["info"]["weights"]
        ax.plot(idx[~scene["is_false"]], w[~scene["is_false"]], ".", color=colors[m], ms=5, label=f"{m}, true")
        ax.plot(idx[scene["is_false"]], w[scene["is_false"]], "x", color=colors[m], ms=7, label=f"{m}, false")
    ax.axhline(REJECT_BELOW, color="gray", ls=":", lw=1); ax.axhline(KEEP_FROM, color="gray", ls=":", lw=1)
    ax.set_xlabel("loop-closure edge (true first, then false)"); ax.set_ylabel("final weight w(e)")
    ax.set_title("final weights (dotted: rejected < 0.1, kept >= 0.5)")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.suptitle(f"Robust pose graph, seed {p.seed}: {len(scene['false_loops'])} false of {len(scene['loops'])} loop closures "
                 f"(policy {p.outlier_policy}); best robust by position RMSE: {best}")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=130)
        print(f"\nSaved figure to {out}")
    else:
        plt.show()


def add_arguments(parser):
    """Add the shared command-line flags (every model parameter, units in the help text)."""
    parser.add_argument("--experiment", choices=("single", "montecarlo", "sweep", "diagnostics", "all"), default="all",
                        help="Which experiment to run (all = single + montecarlo + sweep, the default run; diagnostics is separate)")
    parser.add_argument("--side-length", type=float, default=2.0, help="Side length of the square (m)")
    parser.add_argument("--nodes-per-side", type=int, default=5, help="Odometry segments per side (nodes per lap = 4x)")
    parser.add_argument("--n-laps", type=int, default=3, help="Number of laps driven along the same square")
    parser.add_argument("--pos-noise-std", type=float, default=0.05, help="Odometry translation noise std-dev (m)")
    parser.add_argument("--rot-noise-std", type=float, default=0.01, help="Odometry rotation noise std-dev (rad)")
    parser.add_argument("--loop-noise-scale", type=float, default=0.5, help="Noise std-dev multiplier of the loop closures (true and false)")
    parser.add_argument("--loop-stride", type=int, default=1, help="Keep a true closure for every m-th node of laps >= 2 (1 = every node)")
    parser.add_argument("--outlier-policy", choices=("random", "local"), default="random",
                        help="False-closure node pairs: random = Vertigo default, local = second node within --outlier-window nodes (variant)")
    parser.add_argument("--outlier-meas", choices=("gaussian", "identity"), default="gaussian",
                        help="False-closure measurement: gaussian = Vertigo default (random relative pose), identity = '-p' perfect match (variant)")
    parser.add_argument("--outlier-group", type=int, default=1, help="Consecutive false closures sharing one measurement (1 = no grouping; sources use 10 or 20)")
    parser.add_argument("--outlier-window", type=int, default=20, help="Window of the local policy (nodes), the Vertigo value")
    parser.add_argument("--outlier-pos-std", type=float, default=0.3, help="Std-dev of the false-closure translation, per axis (m, Vertigo value)")
    parser.add_argument("--outlier-rot-std-deg", type=float, default=10.0, help="Std-dev of the false-closure roll/pitch/yaw (deg, Vertigo value)")
    parser.add_argument("--kernel-quantile", type=float, default=0.95, help="Chi-squared(6) quantile whose square root is the kernel scale tau")
    parser.add_argument("--kernel-tau", type=float, default=None, help="Kernel scale tau directly (Huber delta, Cauchy k, sqrt(DCS Phi)); overrides --kernel-quantile")
    parser.add_argument("--kernel-scale-mult", type=float, default=1.0, help="Multiplier on tau")
    parser.add_argument("--damping", type=float, default=0.01, help="Initial Levenberg-Marquardt damping")
    parser.add_argument("--gn-tol", type=float, default=1e-6, help="Stop when the step norm is below this")
    parser.add_argument("--gn-max-iters", type=int, default=50, help="Maximum accepted-step iterations")
    parser.add_argument("--single-outlier-frac", type=float, default=0.2, help="Outlier fraction of the single run")
    parser.add_argument("--outlier-fracs", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3], help="Outlier fractions of the Monte Carlo")
    parser.add_argument("--n-seeds", type=int, default=20, help="Monte Carlo seeds (seed, seed+1, ...)")
    parser.add_argument("--sweep-outlier-frac", type=float, default=0.2, help="Outlier fraction of the kernel-scale sweep")
    parser.add_argument("--sweep-mults", type=float, nargs="+", default=[0.5, 1.0, 2.0], help="Kernel-scale multipliers of the sweep")
    parser.add_argument("--n-sweep-seeds", type=int, default=5, help="Seeds of the kernel-scale sweep")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (first seed of the Monte Carlo)")
    parser.add_argument("--out", type=str, default=None, help="Save the single-run figure to this path instead of showing it")


def run_main(be, description):
    """Parse the flags and run the chosen experiments with backend `be` (shared by both backends)."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    p = parser.parse_args()
    if p.experiment in ("single", "all"):
        scene, results = run_single(be, p)
        plot_single(be, p, scene, results, p.out)
    if p.experiment in ("montecarlo", "all"):
        run_monte_carlo(be, p)
    if p.experiment in ("sweep", "all"):
        run_scale_sweep(be, p)
    if p.experiment == "diagnostics":
        run_diagnostics(be, p)


def main():
    import sys
    run_main(sys.modules[__name__], __doc__)


if __name__ == "__main__":
    main()
