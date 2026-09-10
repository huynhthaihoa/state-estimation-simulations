import numpy as np
import pytest


@pytest.fixture
def pose_graph(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "pose_graph")


@pytest.fixture
def lie_utils(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "lie_utils")


def test_generate_ground_truth_trajectory_corners_only(pose_graph):
    side_length = 2.0
    gt_poses = pose_graph.generate_ground_truth_trajectory(side_length, nodes_per_side=1)
    assert len(gt_poses) == 4

    expected_xy = [(0.0, 0.0), (side_length, 0.0), (side_length, side_length), (0.0, side_length)]
    for T, (x, y) in zip(gt_poses, expected_xy):
        assert np.allclose(T[0:3, 0:3], np.eye(3), atol=1e-12)
        assert np.allclose(T[0:3, 3], [x, y, 0.0], atol=1e-12)
        assert np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    # loop closes: last node is exactly one edge (side_length) away from the first
    closing_dist = np.linalg.norm(gt_poses[-1][0:3, 3] - gt_poses[0][0:3, 3])
    assert np.isclose(closing_dist, side_length, atol=1e-12)


def test_generate_ground_truth_trajectory_subdivided(pose_graph):
    side_length = 4.0
    nodes_per_side = 3
    gt_poses = pose_graph.generate_ground_truth_trajectory(side_length, nodes_per_side=nodes_per_side)
    assert len(gt_poses) == 4 * nodes_per_side
    # consecutive nodes along a side are evenly spaced
    step = side_length / nodes_per_side
    for k in range(nodes_per_side - 1):
        dist = np.linalg.norm(gt_poses[k + 1][0:3, 3] - gt_poses[k][0:3, 3])
        assert np.isclose(dist, step, atol=1e-10)


def test_simulate_noisy_edges_reproducible_with_same_seed(pose_graph):
    gt_poses = pose_graph.generate_ground_truth_trajectory(2.0)

    rng1 = np.random.default_rng(123)
    odom1, loop1 = pose_graph.simulate_noisy_edges(gt_poses, 0.05, 0.01, 0.5, rng1)

    rng2 = np.random.default_rng(123)
    odom2, loop2 = pose_graph.simulate_noisy_edges(gt_poses, 0.05, 0.01, 0.5, rng2)

    for (i1, j1, Z1), (i2, j2, Z2) in zip(odom1, odom2):
        assert i1 == i2 and j1 == j2
        assert np.array_equal(Z1, Z2)
    for (i1, j1, Z1), (i2, j2, Z2) in zip(loop1, loop2):
        assert i1 == i2 and j1 == j2
        assert np.array_equal(Z1, Z2)


def test_simulate_noisy_edges_zero_noise_matches_exact_relative_pose(pose_graph, lie_utils):
    gt_poses = pose_graph.generate_ground_truth_trajectory(2.0)
    rng = np.random.default_rng(0)
    odom, loop = pose_graph.simulate_noisy_edges(gt_poses, 0.0, 0.0, 0.5, rng)

    for idx_i, idx_j, Z_ij in odom:
        Z_true = lie_utils.se3_inv(gt_poses[idx_i]) @ gt_poses[idx_j]
        assert np.allclose(Z_ij, Z_true, atol=1e-12)

    idx_i, idx_j, Z_ij = loop[0]
    Z_true = lie_utils.se3_inv(gt_poses[idx_i]) @ gt_poses[idx_j]
    assert np.allclose(Z_ij, Z_true, atol=1e-12)


def test_run_dead_reckoning_zero_noise_recovers_ground_truth(pose_graph):
    gt_poses = pose_graph.generate_ground_truth_trajectory(2.0)
    rng = np.random.default_rng(0)
    odom, _ = pose_graph.simulate_noisy_edges(gt_poses, 0.0, 0.0, 0.5, rng)

    T_dr = pose_graph.run_dead_reckoning(gt_poses[0], odom)
    assert len(T_dr) == len(gt_poses)
    for T_est, T_true in zip(T_dr, gt_poses):
        assert np.allclose(T_est, T_true, atol=1e-10)


def test_pose_errors_zero_for_identical_lists(pose_graph):
    gt_poses = pose_graph.generate_ground_truth_trajectory(2.0)
    rot_err, pos_err = pose_graph.pose_errors(gt_poses, gt_poses)
    assert np.allclose(rot_err, 0.0, atol=1e-10)
    assert np.allclose(pos_err, 0.0, atol=1e-10)


def test_pose_graph_optimization_beats_dead_reckoning_baseline(pose_graph):
    # Fixed-seed noisy square-loop scenario: this is the module's whole point --
    # loop-closure-aware batch optimization should correct the accumulated
    # dead-reckoning drift, ending up strictly closer to ground truth.
    pos_noise_std, rot_noise_std, loop_noise_scale = 0.05, 0.01, 0.5
    gt_poses = pose_graph.generate_ground_truth_trajectory(2.0)
    rng = np.random.default_rng(10)
    odom, loop = pose_graph.simulate_noisy_edges(gt_poses, pos_noise_std, rot_noise_std, loop_noise_scale, rng)
    all_constraints = odom + loop

    T_dr = pose_graph.run_dead_reckoning(gt_poses[0], odom)

    info_matrix = np.eye(6)
    T_opt = pose_graph.run_pose_graph_optimization(
        T_dr, all_constraints, info_matrix, damping=0.01, gn_tol=1e-6, gn_max_iters=10)

    rot_err_dr, pos_err_dr = pose_graph.pose_errors(gt_poses, T_dr)
    rot_err_opt, pos_err_opt = pose_graph.pose_errors(gt_poses, T_opt)

    rms_pos_dr = np.sqrt(np.mean(pos_err_dr ** 2))
    rms_pos_opt = np.sqrt(np.mean(pos_err_opt ** 2))
    rms_rot_dr = np.sqrt(np.mean(rot_err_dr ** 2))
    rms_rot_opt = np.sqrt(np.mean(rot_err_opt ** 2))

    # optimization must be strictly better than the uncorrected baseline
    assert rms_pos_opt < rms_pos_dr
    assert rms_rot_opt < rms_rot_dr

    # and stay within a reasonable multiple of the injected per-edge noise std-devs
    assert rms_pos_opt < 10 * pos_noise_std
    assert rms_rot_opt < 10 * np.degrees(rot_noise_std)
