import numpy as np
import pytest


@pytest.fixture
def pnp_estimation(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "pnp_estimation")


def test_linear_pnp_dlt_zero_noise_recovers_ground_truth(pnp_estimation):
    # Noiseless correspondences are exactly consistent with T_true, so the
    # closed-form DLT solve should recover it to near machine precision --
    # the strong correctness check for the new linear-algebra derivation.
    rng = np.random.default_rng(0)
    K = (800.0, 800.0, 320.0, 240.0)
    T_true, P_list = pnp_estimation.generate_scene(20, K, rng)
    z_list = [pnp_estimation.camera_project(T_true, P, K)[0] for P in P_list]

    T_dlt = pnp_estimation.linear_pnp_dlt(P_list, z_list, K)

    rot_err, pos_err = pnp_estimation.pose_error(T_dlt, T_true)
    assert rot_err < 1e-8
    assert pos_err < 1e-8


def test_refine_pose_gn_improves_on_noisy_linear_pnp_dlt(pnp_estimation):
    # Fixed-seed noisy scenario: Gauss-Newton refinement against the true
    # reprojection error should end up strictly closer to ground truth than
    # the linear DLT initial guess alone.
    rng = np.random.default_rng(0)
    K = (800.0, 800.0, 320.0, 240.0)
    T_true, P_list = pnp_estimation.generate_scene(20, K, rng)
    z_list = [pnp_estimation.camera_project(T_true, P, K)[0] + rng.normal(0.0, 1.0, 2) for P in P_list]

    T_dlt = pnp_estimation.linear_pnp_dlt(P_list, z_list, K)
    T_gn = pnp_estimation.refine_pose_gn(T_dlt, P_list, z_list, K, gn_tol=1e-8, gn_max_iters=20)

    rot_err_dlt, pos_err_dlt = pnp_estimation.pose_error(T_dlt, T_true)
    rot_err_gn, pos_err_gn = pnp_estimation.pose_error(T_gn, T_true)

    assert rot_err_gn < rot_err_dlt
    assert pos_err_gn < pos_err_dlt


def test_generate_scene_reproducible_with_same_seed(pnp_estimation):
    K = (800.0, 800.0, 320.0, 240.0)

    rng1 = np.random.default_rng(7)
    T1, P1 = pnp_estimation.generate_scene(10, K, rng1)

    rng2 = np.random.default_rng(7)
    T2, P2 = pnp_estimation.generate_scene(10, K, rng2)

    assert np.array_equal(T1, T2)
    for p1, p2 in zip(P1, P2):
        assert np.array_equal(p1, p2)
