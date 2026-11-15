import argparse
import contextlib
import io

import numpy as np
import pytest
manif = pytest.importorskip("manifpy")
from state_estimation_simulations.use_manif import pose_graph as _pose_graph_manif  # noqa: E402
from state_estimation_simulations.use_manif import pose_graph_robust as _prg_manif  # noqa: E402
from state_estimation_simulations.use_numpy import pose_graph_robust as _prg_numpy  # noqa: E402


def _to_matrix(T):
    M = np.eye(4)
    M[0:3, 0:3] = T.rotation()
    M[0:3, 3] = T.translation()
    return M


@pytest.fixture
def prg():
    return _prg_manif


@pytest.fixture
def params():
    parser = argparse.ArgumentParser()
    _prg_manif.add_arguments(parser)
    return parser.parse_args([])


def _random_se3(rng, scale):
    return manif.SE3Tangent(rng.normal(0, scale, 6)).exp()


def test_residual_jacobians_match_finite_differences(prg):
    rng = np.random.default_rng(1)
    h = 1e-6
    for _ in range(5):
        Xi, Z = _random_se3(rng, 0.5), _random_se3(rng, 0.5)
        Xj = Xi * Z * _random_se3(rng, 0.3)
        e0, J_i, J_j = prg.edge_residual_and_jacobians(Xi, Xj, Z)
        assert np.allclose(e0, prg.edge_residual(Xi, Xj, Z), atol=1e-12)
        J_i_fd, J_j_fd = np.zeros((6, 6)), np.zeros((6, 6))
        for k in range(6):
            d = np.zeros(6)
            d[k] = h
            J_i_fd[:, k] = (prg.edge_residual(prg.retract(Xi, d), Xj, Z) - prg.edge_residual(prg.retract(Xi, -d), Xj, Z)) / (2 * h)
            J_j_fd[:, k] = (prg.edge_residual(Xi, prg.retract(Xj, d), Z) - prg.edge_residual(Xi, prg.retract(Xj, -d), Z)) / (2 * h)
        assert np.allclose(J_i, J_i_fd, atol=1e-6)
        assert np.allclose(J_j, J_j_fd, atol=1e-6)


def test_manif_residual_and_jacobians_equal_the_numpy_ones(prg):
    rng = np.random.default_rng(2)
    for _ in range(5):
        Xi, Z = _random_se3(rng, 0.5), _random_se3(rng, 0.5)
        Xj = Xi * Z * _random_se3(rng, 0.3)
        e_m, Ji_m, Jj_m = prg.edge_residual_and_jacobians(Xi, Xj, Z)
        e_n, Ji_n, Jj_n = _prg_numpy.edge_residual_and_jacobians(_to_matrix(Xi), _to_matrix(Xj), _to_matrix(Z))
        assert np.allclose(e_m, e_n, atol=1e-9)
        assert np.allclose(Ji_m, Ji_n, atol=1e-9)
        assert np.allclose(Jj_m, Jj_n, atol=1e-9)


@pytest.mark.parametrize("method", ["l2", "huber", "cauchy", "dcs"])
def test_zero_noise_no_outliers_recovers_the_truth(prg, method):
    gt = prg.generate_ground_truth_trajectory(2.0, 3) * 2
    L = 12
    odo = [(k, k + 1, gt[k].inverse() * gt[k + 1]) for k in range(len(gt) - 1)]
    loops = [(k - L, k, gt[k - L].inverse() * gt[k]) for k in range(L, len(gt))]
    rng = np.random.default_rng(0)
    init = [gt[0]] + [T + manif.SE3Tangent(np.concatenate([rng.normal(0, 0.03, 3), rng.normal(0, 0.01, 3)])) for T in gt[1:]]
    T_est, info = prg.run_robust_pose_graph_optimization(init, odo, loops, np.eye(6), np.eye(6), method, 3.5, 0.01, 1e-10, 100)
    for T, T_true in zip(T_est, gt):
        assert np.allclose(_to_matrix(T), _to_matrix(T_true), atol=1e-5)
    assert np.all(info["weights"] > 0.99)


def test_l2_without_outliers_equals_pose_graph_py(prg):
    rng = np.random.default_rng(4)
    gt = _pose_graph_manif.generate_ground_truth_trajectory(2.0, 3)
    odo, loop = _pose_graph_manif.simulate_noisy_edges(gt, 0.05, 0.01, 0.5, rng)
    T_dr = _pose_graph_manif.run_dead_reckoning(gt[0], odo)
    with contextlib.redirect_stdout(io.StringIO()):
        T_ref = _pose_graph_manif.run_pose_graph_optimization(T_dr, odo + loop, np.eye(6), 0.01, 1e-6, 10)
    T_new, _ = prg.run_robust_pose_graph_optimization(T_dr, odo, loop, np.eye(6), np.eye(6), "l2", 3.5, 0.01, 1e-6, 10)
    for a, b in zip(T_ref, T_new):
        assert np.allclose(_to_matrix(a), _to_matrix(b), atol=1e-9, rtol=0)


def test_numpy_and_manif_agree_on_the_same_seed(prg, params):
    scene_m = prg.build_scene(prg, params, 0, 0.2)
    scene_n = _prg_numpy.build_scene(_prg_numpy, params, 0, 0.2)
    assert len(scene_m["loops"]) == len(scene_n["loops"]) == 48
    for (i1, j1, Zm), (i2, j2, Zn) in zip(scene_m["odometry"] + scene_m["loops"], scene_n["odometry"] + scene_n["loops"]):
        assert (i1, j1) == (i2, j2)
        assert np.allclose(_to_matrix(Zm), Zn, atol=1e-10)
    assert np.allclose(np.array([_to_matrix(T) for T in scene_m["T_dr"]]), np.array(scene_n["T_dr"]), atol=1e-10)
    for method in prg.METHODS:
        tau = prg.kernel_threshold(0.95)
        rm = prg.solve_scene(prg, scene_m, method, tau, params)
        rn = _prg_numpy.solve_scene(_prg_numpy, scene_n, method, tau, params)
        assert rm["iterations"] == rn["iterations"]
        for Tm, Tn in zip(rm["T"], rn["T"]):
            assert np.allclose(_to_matrix(Tm), Tn, atol=1e-8)
        assert np.allclose(rm["info"]["weights"], rn["info"]["weights"], atol=1e-8)
        assert np.isclose(rm["pos_rmse"], rn["pos_rmse"], atol=1e-8)


def test_robust_methods_beat_l2_on_a_fixed_seed_with_20_percent_outliers(prg, params):
    scene = prg.build_scene(prg, params, 0, 0.2)
    res = {m: prg.solve_scene(prg, scene, m, prg.kernel_threshold(0.95), params) for m in prg.METHODS}
    assert res["l2"]["pos_rmse"] > 0.8
    assert res["huber"]["pos_rmse"] < 0.6 * res["l2"]["pos_rmse"]
    assert res["cauchy"]["pos_rmse"] < 0.15 and res["dcs"]["pos_rmse"] < 0.15
    for m in ("cauchy", "dcs"):
        assert res[m]["cls"][0] == 8 and res[m]["cls"][2] == 40
