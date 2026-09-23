import numpy as np
import pytest


@pytest.fixture
def pose_graph(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "pose_graph")


@pytest.fixture
def pose_graph_incremental(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "pose_graph_incremental")


def _noisy_scenario(pose_graph, seed, nodes_per_side=4, side_length=2.0,
                     pos_noise_std=0.05, rot_noise_std=0.01, loop_noise_scale=0.5):
    gt_poses = pose_graph.generate_ground_truth_trajectory(side_length, nodes_per_side=nodes_per_side)
    rng = np.random.default_rng(seed)
    odometry_constraints, loop_constraints = pose_graph.simulate_noisy_edges(
        gt_poses, pos_noise_std, rot_noise_std, loop_noise_scale, rng)
    return gt_poses, odometry_constraints, loop_constraints


def test_incremental_matches_batch_on_same_constraint_set(pose_graph, pose_graph_incremental):
    # The module's own stated thesis: incremental square-root SAM and batch
    # Gauss-Newton relaxation solve the same nonlinear least-squares problem,
    # so given the same constraints and enough relinearization/iteration they
    # must converge to (numerically) the same final answer -- only the cost of
    # getting there differs, which this test isn't responsible for checking.
    gt_poses, odometry_constraints, loop_constraints = _noisy_scenario(pose_graph, seed=7)
    info_matrix = np.eye(6)
    gn_tol, gn_max_iters = 1e-10, 30

    x_incr, n_relinearizations = pose_graph_incremental.run_incremental_pose_graph(
        gt_poses, odometry_constraints, loop_constraints, info_matrix,
        anchor_weight=1e6, relinearize_every=5, gn_tol=gn_tol, gn_max_iters=gn_max_iters)

    T_init = pose_graph.run_dead_reckoning(gt_poses[0], odometry_constraints)
    T_batch = pose_graph.run_pose_graph_optimization(
        T_init, odometry_constraints + loop_constraints, info_matrix,
        damping=0.0, gn_tol=gn_tol, gn_max_iters=gn_max_iters)

    assert len(x_incr) == len(T_batch) == len(gt_poses)
    assert n_relinearizations >= 1

    for T_i, T_b in zip(x_incr, T_batch):
        assert np.allclose(T_i[0:3, 0:3], T_b[0:3, 0:3], atol=1e-6)
        assert np.allclose(T_i[0:3, 3], T_b[0:3, 3], atol=1e-6)

    rot_incr, pos_incr = pose_graph.pose_errors(gt_poses, x_incr)
    rot_batch, pos_batch = pose_graph.pose_errors(gt_poses, T_batch)
    assert np.allclose(rot_incr, rot_batch, atol=1e-5)
    assert np.allclose(pos_incr, pos_batch, atol=1e-6)

    # sanity: loop-closure-aware solving still beats the uncorrected baseline
    rot_dr, pos_dr = pose_graph.pose_errors(gt_poses, T_init)
    assert np.sqrt(np.mean(pos_incr ** 2)) < np.sqrt(np.mean(pos_dr ** 2))


def test_relinearize_to_convergence_R_d_matches_returned_x(pose_graph, pose_graph_incremental):
    # relinearize_to_convergence's docstring promises R, d are "already built at
    # that point" (the returned x_current). Checked directly against
    # full_relinearize -- the ground-truth definition of "R, d built at a given
    # x" -- at low gn_max_iters, where a version that relinearized before each
    # retraction (rather than after) would return R, d from the *previous*
    # iteration's x, one retraction stale.
    gt_poses, odometry_constraints, loop_constraints = _noisy_scenario(pose_graph, seed=0, nodes_per_side=4)
    info_matrix = np.eye(6)

    x_init = [gt_poses[0]]
    for (_, _, Z) in odometry_constraints:
        x_init.append(x_init[-1] @ Z)

    for gn_max_iters in (1, 2, 3):
        x_out, R, d = pose_graph_incremental.relinearize_to_convergence(
            x_init, odometry_constraints, loop_constraints, info_matrix,
            anchor_weight=1e6, gn_tol=1e-12, gn_max_iters=gn_max_iters)
        R_expected, d_expected = pose_graph_incremental.full_relinearize(
            x_out, odometry_constraints, loop_constraints, info_matrix, anchor_weight=1e6)
        assert np.allclose(R, R_expected, atol=1e-8)
        assert np.allclose(d, d_expected, atol=1e-8)


def test_incremental_matches_batch_across_relinearize_schedules(pose_graph, pose_graph_incremental):
    # Regardless of how often periodic relinearization fires, the
    # loop-closure edge always forces a final full relinearization to
    # convergence -- so the schedule shouldn't change the converged answer.
    gt_poses, odometry_constraints, loop_constraints = _noisy_scenario(pose_graph, seed=3, nodes_per_side=3)
    info_matrix = np.eye(6)
    gn_tol, gn_max_iters = 1e-10, 30

    T_init = pose_graph.run_dead_reckoning(gt_poses[0], odometry_constraints)
    T_batch = pose_graph.run_pose_graph_optimization(
        T_init, odometry_constraints + loop_constraints, info_matrix,
        damping=0.0, gn_tol=gn_tol, gn_max_iters=gn_max_iters)
    _, pos_batch = pose_graph.pose_errors(gt_poses, T_batch)

    for relinearize_every in (1, 2, 100):
        x_incr, _ = pose_graph_incremental.run_incremental_pose_graph(
            gt_poses, odometry_constraints, loop_constraints, info_matrix,
            anchor_weight=1e6, relinearize_every=relinearize_every, gn_tol=gn_tol, gn_max_iters=gn_max_iters)
        _, pos_incr = pose_graph.pose_errors(gt_poses, x_incr)
        assert np.allclose(pos_incr, pos_batch, atol=1e-6)
