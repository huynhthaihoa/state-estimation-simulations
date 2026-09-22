import numpy as np
import pytest


@pytest.fixture
def pose_graph(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "pose_graph")


@pytest.fixture
def swm(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "sliding_window_marginalization")


def _chain_scenario(pose_graph, seed, n_poses=3, side_length=2.0, pos_noise_std=0.05,
                     rot_noise_std=0.01):
    # nodes_per_side=1 gives a 4-node square; slice down to n_poses<=4 for a short chain.
    gt_poses = pose_graph.generate_ground_truth_trajectory(side_length, nodes_per_side=1)[:n_poses]
    rng = np.random.default_rng(seed)
    odometry_constraints, _ = pose_graph.simulate_noisy_edges(
        gt_poses, pos_noise_std, rot_noise_std, loop_noise_scale=1.0, rng=rng)
    odometry_constraints = odometry_constraints[:n_poses - 1]
    return gt_poses, odometry_constraints


# --- the critical correctness test: marginalized-window matches joint batch ---

def test_marginalized_window_matches_joint_batch_solve(pose_graph, swm):
    gt_poses, odometry_constraints = _chain_scenario(pose_graph, seed=7, n_poses=3)
    info_matrix = np.eye(6)
    anchor_weight = 1e6
    gn_tol, gn_max_iters = 1e-12, 30

    # Joint batch solve over all 3 nodes together (self-consistent reference: built from
    # the exact same assemble_window_system/solve_to_convergence this script's sliding-window
    # path uses, not pose_graph.py's own hardcoded-anchor solver, to avoid a fragile
    # cross-module constant dependency).
    x_init = pose_graph.run_dead_reckoning(gt_poses[0], odometry_constraints)
    edges_all = [(0, 1, odometry_constraints[0][2]), (1, 2, odometry_constraints[1][2])]
    x_batch, _ = swm.solve_to_convergence(x_init, gt_poses[0], anchor_weight * np.eye(6),
                                           edges_all, info_matrix, gn_tol, gn_max_iters)

    # Sliding-window path with window_size=2: node 0 anchored, node 1 added and converged,
    # node 2 added (window now size 3 > window_size) -> marginalize node 0 -> re-converge.
    prior_ref, prior_info = gt_poses[0], anchor_weight * np.eye(6)
    x_window = [gt_poses[0]]
    x_window.append(x_window[-1] @ odometry_constraints[0][2])
    x_window, _ = swm.solve_to_convergence(x_window, prior_ref, prior_info,
                                            [(0, 1, odometry_constraints[0][2])],
                                            info_matrix, gn_tol, gn_max_iters)
    x_window.append(x_window[-1] @ odometry_constraints[1][2])
    edges_window = [(0, 1, odometry_constraints[0][2]), (1, 2, odometry_constraints[1][2])]
    x_window, _ = swm.solve_to_convergence(x_window, prior_ref, prior_info, edges_window,
                                            info_matrix, gn_tol, gn_max_iters)

    new_prior_ref, new_prior_info, x_dropped = swm.marginalize_oldest(
        x_window, prior_ref, prior_info, edges_window, info_matrix)
    x_window_final = [x_window[1], x_window[2]]
    x_window_final, _ = swm.solve_to_convergence(
        x_window_final, new_prior_ref, new_prior_info, [(0, 1, odometry_constraints[1][2])],
        info_matrix, gn_tol, gn_max_iters)

    for T_batch, T_window in zip(x_batch[1:], x_window_final):
        assert np.allclose(T_batch[0:3, 0:3], T_window[0:3, 0:3], atol=1e-6)
        assert np.allclose(T_batch[0:3, 3], T_window[0:3, 3], atol=1e-6)


def test_marginalization_produces_no_fill_in_on_a_chain(pose_graph, swm):
    # Structural check: the Schur-complement correction term, before extracting just the
    # (0,0) block, must be exactly zero everywhere else -- a chain's oldest pose only ever
    # touches one surviving neighbor, so there is nothing to fill in.
    gt_poses, odometry_constraints = _chain_scenario(pose_graph, seed=3, n_poses=4)
    info_matrix = np.eye(6)
    anchor_weight = 1e6
    gn_tol, gn_max_iters = 1e-12, 30

    prior_ref, prior_info = gt_poses[0], anchor_weight * np.eye(6)
    x_window = [gt_poses[0]]
    edges_window = []
    for k in range(1, 4):
        x_window.append(x_window[-1] @ odometry_constraints[k - 1][2])
        edges_window.append((k - 1, k, odometry_constraints[k - 1][2]))
    x_window, _ = swm.solve_to_convergence(x_window, prior_ref, prior_info, edges_window,
                                            info_matrix, gn_tol, gn_max_iters)

    H, _ = swm.assemble_window_system(x_window, prior_ref, prior_info, edges_window, info_matrix)
    Lambda_aa = H[0:6, 0:6]
    Lambda_ab = H[0:6, 6:]
    Lambda_ba = H[6:, 0:6]
    correction = Lambda_ba @ np.linalg.solve(Lambda_aa, Lambda_ab)

    assert np.max(np.abs(correction[6:, 6:])) < 1e-8  # zero everywhere outside the (0,0) block
    assert np.max(np.abs(correction[0:6, 6:])) < 1e-8
    assert np.max(np.abs(correction[6:, 0:6])) < 1e-8
    assert np.max(np.abs(correction[0:6, 0:6])) > 1e-8  # the one nonzero block actually exists


# --- window never exceeds its configured size ---

def test_sliding_window_never_exceeds_configured_size(pose_graph, swm):
    gt_poses = pose_graph.generate_ground_truth_trajectory(2.0, nodes_per_side=8)  # 32 poses
    rng = np.random.default_rng(0)
    odometry_constraints, _ = pose_graph.simulate_noisy_edges(
        gt_poses, 0.05, 0.01, loop_noise_scale=1.0, rng=rng)

    x_final, max_dof = swm.run_sliding_window_pose_graph(
        gt_poses, odometry_constraints, np.eye(6), window_size=5, anchor_weight=1e6,
        gn_tol=1e-6, gn_max_iters=10)

    assert max_dof <= 6 * 5
    assert len(x_final) == len(gt_poses)
    assert all(T is not None for T in x_final)


# --- the real scaling finding, run for real ---

def test_sliding_window_max_dof_stays_flat_while_batch_grows(pose_graph, swm):
    rng = np.random.default_rng(0)
    info_matrix = np.eye(6)
    window_size = 5

    dof_batch_list, dof_window_list = [], []
    for nodes_per_side in (2, 4, 8, 16):
        gt_poses = pose_graph.generate_ground_truth_trajectory(2.0, nodes_per_side=nodes_per_side)
        odometry_constraints, _ = pose_graph.simulate_noisy_edges(
            gt_poses, 0.05, 0.01, loop_noise_scale=1.0, rng=rng)

        _, dof_batch = swm.run_full_batch_growing(gt_poses, odometry_constraints, info_matrix,
                                                   gn_tol=1e-6, gn_max_iters=10)
        _, dof_window = swm.run_sliding_window_pose_graph(
            gt_poses, odometry_constraints, info_matrix, window_size=window_size,
            anchor_weight=1e6, gn_tol=1e-6, gn_max_iters=10)
        dof_batch_list.append(dof_batch)
        dof_window_list.append(dof_window)

    assert dof_batch_list == sorted(dof_batch_list)  # batch's system size grows monotonically
    assert dof_batch_list[-1] > dof_batch_list[0]  # ...and actually grows, not flat
    assert all(d == 6 * window_size for d in dof_window_list)  # window's stays exactly capped
