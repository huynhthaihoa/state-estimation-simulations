import argparse
import contextlib
import io

import numpy as np
import pytest

from state_estimation_simulations.use_numpy import pose_graph as _pose_graph
from state_estimation_simulations.use_numpy import pose_graph_robust as _prg
from state_estimation_simulations.use_numpy.lie_utils import se3_exp, se3_log, se3_inv, se3_adjoint, compute_se3_inv_right_jacobian


@pytest.fixture
def prg():
    return _prg


@pytest.fixture
def params():
    parser = argparse.ArgumentParser()
    _prg.add_arguments(parser)
    return parser.parse_args([])


# ------------------------------------------------------------------ kernels
def test_kernel_weights_match_the_doc_worked_example(prg):
    # docs/optimization/pose_graph_optimization.md section 16.1: delta = k = 1, e = 10
    assert np.isclose(prg.kernel_weight("l2", 10.0, 1.0), 1.0)
    assert np.isclose(prg.kernel_weight("huber", 10.0, 1.0), 0.1)
    assert np.isclose(prg.kernel_weight("cauchy", 10.0, 1.0), 1.0 / 101.0)
    # costs in the 1/2 convention: L2 50, Huber 9.5, Cauchy 1/2 ln(101) = 2.31
    assert np.isclose(prg.kernel_cost("l2", 10.0, 1.0), 50.0)
    assert np.isclose(prg.kernel_cost("huber", 10.0, 1.0), 9.5)
    assert np.isclose(prg.kernel_cost("cauchy", 10.0, 1.0), 0.5 * np.log(101.0))
    # good edge e = 0.5: L2 and Huber keep weight 1, Cauchy gives 1 / 1.25 = 0.8
    assert np.isclose(prg.kernel_weight("huber", 0.5, 1.0), 1.0)
    assert np.isclose(prg.kernel_weight("cauchy", 0.5, 1.0), 0.8)


def test_dcs_weight_matches_the_doc_and_the_paper_formula(prg):
    phi = 12.59
    tau = np.sqrt(phi)
    # section 16.3: s = 1 up to e^2 = Phi, 0.7726 at e^2 = 20, 0.2236 at 100, 0.002515 at 1e4 ; weight = s^2
    for e2, s in [(10.0, 1.0), (12.59, 1.0), (20.0, 0.7726), (100.0, 0.2236), (1e4, 0.002515)]:
        assert np.isclose(prg.kernel_weight("dcs", np.sqrt(e2), tau), s ** 2, rtol=2e-3)
    # s = min(1, 2 Phi / (Phi + e^2)) written out
    e = 7.0
    assert np.isclose(prg.kernel_weight("dcs", e, tau), min(1.0, 2 * phi / (phi + e ** 2)) ** 2)


@pytest.mark.parametrize("method", ["l2", "huber", "cauchy", "dcs"])
def test_kernel_weight_is_the_irls_weight_of_the_cost(prg, method):
    # w(e) = (1/e) d rho / d e, checked by central differences (away from the kinks)
    tau = 3.5
    h = 1e-6
    for e in [0.3, 1.0, 2.0, 5.0, 9.0, 40.0, 300.0]:
        drho = (prg.kernel_cost(method, e + h, tau) - prg.kernel_cost(method, e - h, tau)) / (2 * h)
        assert np.isclose(drho / e, prg.kernel_weight(method, e, tau), rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize("method", ["huber", "cauchy", "dcs"])
def test_kernels_are_quadratic_for_small_residuals_and_continuous(prg, method):
    tau = 3.5
    assert np.isclose(prg.kernel_cost(method, 0.1, tau), 0.005, rtol=1e-3 if method == "cauchy" else 1e-12)
    e = np.array([tau - 1e-9, tau + 1e-9])
    c = prg.kernel_cost(method, e, tau)
    assert abs(c[1] - c[0]) < 1e-7


def test_chi2_quantile_matches_scipy_and_the_doc(prg):
    scipy_stats = pytest.importorskip("scipy.stats")
    for q in (0.5, 0.9, 0.95, 0.99):
        assert np.isclose(prg.chi2_quantile_6dof(q), scipy_stats.chi2.ppf(q, 6), rtol=1e-9)
    assert np.isclose(prg.chi2_quantile_6dof(0.95), 12.59, atol=5e-3)    # section 16.5 of the doc
    assert np.isclose(prg.kernel_threshold(0.95, 2.0), 2.0 * np.sqrt(12.5916), atol=1e-3)


# ------------------------------------------------------------------ Jacobians
def _random_edge(rng):
    Xi = se3_exp(rng.normal(0, 0.5, 6))
    Z = se3_exp(rng.normal(0, 0.5, 6))
    Xj = Xi @ Z @ se3_exp(rng.normal(0, 0.3, 6))
    return Xi, Xj, Z


def test_residual_jacobians_match_finite_differences(prg):
    rng = np.random.default_rng(1)
    h = 1e-6
    for _ in range(5):
        Xi, Xj, Z = _random_edge(rng)
        e0, J_i, J_j = prg.edge_residual_and_jacobians(Xi, Xj, Z)
        assert np.allclose(e0, prg.edge_residual(Xi, Xj, Z), atol=1e-12)
        J_i_fd, J_j_fd = np.zeros((6, 6)), np.zeros((6, 6))
        for k in range(6):
            d = np.zeros(6)
            d[k] = h
            J_i_fd[:, k] = (prg.edge_residual(Xi @ se3_exp(d), Xj, Z) - prg.edge_residual(Xi @ se3_exp(-d), Xj, Z)) / (2 * h)
            J_j_fd[:, k] = (prg.edge_residual(Xi, Xj @ se3_exp(d), Z) - prg.edge_residual(Xi, Xj @ se3_exp(-d), Z)) / (2 * h)
        assert np.allclose(J_i, J_i_fd, atol=1e-6)
        assert np.allclose(J_j, J_j_fd, atol=1e-6)


def test_jacobians_equal_the_pose_graph_py_form(prg):
    # pose_graph.py: Jb = -invJr(-e), J_i = Jb @ Ad(Z^-1); here Jb = -invJr(e) Ad((T_pred^-1 Xj)^-1)
    rng = np.random.default_rng(2)
    for _ in range(5):
        Xi, Xj, Z = _random_edge(rng)
        e_vec = se3_log(se3_inv(Xi @ Z) @ Xj)
        J_i_ref = -compute_se3_inv_right_jacobian(-e_vec) @ se3_adjoint(se3_inv(Z))
        J_j_ref = compute_se3_inv_right_jacobian(e_vec)
        _, J_i, J_j = prg.edge_residual_and_jacobians(Xi, Xj, Z)
        assert np.allclose(J_i, J_i_ref, atol=1e-10)
        assert np.allclose(J_j, J_j_ref, atol=1e-10)


# ------------------------------------------------------------------ graph generation
def test_euler_to_quat_and_make_pose_give_a_rotation_matrix(prg):
    roll, pitch, yaw = 0.2, -0.4, 1.1
    T = prg.make_pose(np.array([1.0, 2.0, 3.0]), prg.euler_to_quat(roll, pitch, yaw))
    R = T[0:3, 0:3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(R), 1.0)
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    assert np.allclose(R, Rz @ Ry @ Rx, atol=1e-12)
    assert np.allclose(T[0:3, 3], [1.0, 2.0, 3.0])


def test_false_events_are_never_same_place_or_consecutive(prg):
    rng = np.random.default_rng(0)
    for policy in ("random", "local"):
        for _ in range(300):
            i, j = prg.draw_false_event(60, policy, 1, 20, 20, rng)
            assert 0 <= i < j <= 59 and j - i >= 2 and (j - i) % 20 != 0
            if policy == "local":
                assert j - i <= 20 + 2


def test_grouped_false_closures_are_consecutive_and_share_a_measurement(prg, params):
    params.outlier_group = 10
    scene = prg.build_scene(prg, params, 3, 0.5)           # 20 false closures = 2 groups of 10
    f = scene["false_loops"]
    assert len(f) == 20
    for g in (f[:10], f[10:]):
        assert all((g[k + 1][0], g[k + 1][1]) == (g[k][0] + 1, g[k][1] + 1) for k in range(9))
        assert all(np.array_equal(g[0][2], e[2]) for e in g) and max(e[1] for e in g) <= 59


def test_scene_structure_and_counts(prg, params):
    scene = prg.build_scene(prg, params, 0, 0.2)
    assert len(scene["gt"]) == 60 and len(scene["odometry"]) == 59
    assert len(scene["true_loops"]) == 40 and len(scene["false_loops"]) == 8
    assert scene["is_false"].sum() == 8 and len(scene["loops"]) == 48
    # true closures connect the same place one lap apart; the dead reckoning starts at the truth
    for i, j, _ in scene["true_loops"]:
        assert j - i == 20 and np.allclose(scene["gt"][i], scene["gt"][j])
    assert np.allclose(scene["T_dr"][0], scene["gt"][0])


def test_data_do_not_change_with_the_outlier_fraction(prg, params):
    a = prg.build_scene(prg, params, 5, 0.1)
    b = prg.build_scene(prg, params, 5, 0.3)
    for (i1, j1, Z1), (i2, j2, Z2) in zip(a["odometry"] + a["true_loops"], b["odometry"] + b["true_loops"]):
        assert (i1, j1) == (i2, j2) and np.array_equal(Z1, Z2)
    assert np.array_equal(a["T_dr"][-1], b["T_dr"][-1])
    # the false closures of the smaller fraction are the first ones of the larger
    assert len(a["false_loops"]) == 4 and len(b["false_loops"]) == 12
    for (i1, j1, Z1), (i2, j2, Z2) in zip(a["false_loops"], b["false_loops"]):
        assert (i1, j1) == (i2, j2) and np.array_equal(Z1, Z2)


def test_different_seeds_redraw_odometry_true_closures_and_false_closures(prg, params):
    a = prg.build_scene(prg, params, 1, 0.2)
    b = prg.build_scene(prg, params, 2, 0.2)
    assert not np.allclose(a["odometry"][0][2], b["odometry"][0][2])
    assert not np.allclose(a["true_loops"][0][2], b["true_loops"][0][2])
    assert [(i, j) for i, j, _ in a["false_loops"]] != [(i, j) for i, j, _ in b["false_loops"]]


def test_identity_measurement_variant(prg, params):
    params.outlier_meas = "identity"
    scene = prg.build_scene(prg, params, 0, 0.2)
    for _, _, Z in scene["false_loops"]:
        assert np.allclose(Z, np.eye(4))


def test_classify_loops(prg):
    w = np.array([1.0, 0.4, 0.6, 0.05, 0.2, 0.0])
    is_false = np.array([False, False, False, True, True, True])
    assert prg.classify_loops(w, is_false) == (2, 3, 2, 3)     # false < 0.1: 0.05, 0.0 ; true >= 0.5: 1.0, 0.6


# ------------------------------------------------------------------ solver
def _perfect_graph(prg, n_laps=2, nodes_per_side=3, side=2.0):
    gt = prg.generate_ground_truth_trajectory(side, nodes_per_side) * n_laps
    L = 4 * nodes_per_side
    odo = [(k, k + 1, se3_inv(gt[k]) @ gt[k + 1]) for k in range(len(gt) - 1)]
    loops = [(k - L, k, se3_inv(gt[k - L]) @ gt[k]) for k in range(L, len(gt))]
    return gt, odo, loops


@pytest.mark.parametrize("method", ["l2", "huber", "cauchy", "dcs"])
def test_zero_noise_no_outliers_recovers_the_truth(prg, method):
    gt, odo, loops = _perfect_graph(prg)
    rng = np.random.default_rng(0)
    init = [gt[0]] + [T @ se3_exp(np.concatenate([rng.normal(0, 0.03, 3), rng.normal(0, 0.01, 3)])) for T in gt[1:]]
    T_est, info = prg.run_robust_pose_graph_optimization(init, odo, loops, np.eye(6), np.eye(6), method, 3.5,
                                                         0.01, 1e-10, 100)
    for T, T_true in zip(T_est, gt):
        assert np.allclose(T, T_true, atol=1e-5)
    assert np.all(info["weights"] > 0.99)
    # from the truth itself nothing moves
    T_same, info_same = prg.run_robust_pose_graph_optimization(gt, odo, loops, np.eye(6), np.eye(6), method, 3.5)
    assert all(np.allclose(a, b, atol=1e-9) for a, b in zip(T_same, gt))


def test_l2_without_outliers_equals_pose_graph_py(prg):
    rng = np.random.default_rng(4)
    gt = _pose_graph.generate_ground_truth_trajectory(2.0, 3)
    odo, loop = _pose_graph.simulate_noisy_edges(gt, 0.05, 0.01, 0.5, rng)
    T_dr = _pose_graph.run_dead_reckoning(gt[0], odo)
    with contextlib.redirect_stdout(io.StringIO()):
        T_ref = _pose_graph.run_pose_graph_optimization(T_dr, odo + loop, np.eye(6), 0.01, 1e-6, 10)
    T_new, info = prg.run_robust_pose_graph_optimization(T_dr, odo, loop, np.eye(6), np.eye(6), "l2", 3.5, 0.01, 1e-6, 10)
    for a, b in zip(T_ref, T_new):
        assert np.allclose(a, b, atol=1e-9, rtol=0)
    assert np.allclose(info["weights"], 1.0)


def test_node_zero_stays_at_the_truth_so_no_alignment_is_needed(prg, params):
    scene = prg.build_scene(prg, params, 0, 0.2)
    for method in prg.METHODS:
        T, _ = prg.run_robust_pose_graph_optimization(scene["T_dr"], scene["odometry"], scene["loops"], scene["info_odo"],
                                                      scene["info_loop"], method, 3.55)
        assert np.allclose(T[0], scene["gt"][0], atol=1e-6)


def test_lm_never_increases_the_robust_cost(prg, params):
    scene = prg.build_scene(prg, params, 0, 0.2)
    for method in ("huber", "cauchy", "dcs"):
        _, info = prg.run_robust_pose_graph_optimization(scene["T_dr"], scene["odometry"], scene["loops"], scene["info_odo"],
                                                         scene["info_loop"], method, 3.55, record=True)
        c = np.array(info["history"]["cost"])
        assert np.all(np.diff(c) < 0) and len(c) == info["iterations"] + 1


def test_robust_methods_beat_l2_on_a_fixed_seed_with_20_percent_outliers(prg, params):
    # numbers seen on seed 0 (position RMSE, m): dead reckoning 0.355, l2 1.259, huber 0.563, cauchy 0.082, dcs 0.091
    scene = prg.build_scene(prg, params, 0, 0.2)
    res = {m: prg.solve_scene(prg, scene, m, prg.kernel_threshold(0.95), params) for m in prg.METHODS}
    assert res["l2"]["pos_rmse"] > 0.8
    assert res["huber"]["pos_rmse"] < 0.6 * res["l2"]["pos_rmse"]
    assert res["cauchy"]["pos_rmse"] < 0.15 and res["dcs"]["pos_rmse"] < 0.15
    assert res["cauchy"]["rot_rmse"] < 2.0 and res["dcs"]["rot_rmse"] < 2.0
    assert res["l2"]["cls"][0] == 0                       # l2 never down-weights anything
    for m in ("cauchy", "dcs"):
        assert res[m]["cls"][0] == 8 and res[m]["cls"][2] == 40    # all 8 false rejected, all 40 true kept


def test_robust_kernels_cost_little_when_nothing_is_wrong(prg, params):
    scene = prg.build_scene(prg, params, 0, 0.0)
    res = {m: prg.solve_scene(prg, scene, m, prg.kernel_threshold(0.95), params)["pos_rmse"] for m in prg.METHODS}
    for m in ("huber", "cauchy", "dcs"):
        assert abs(res[m] - res["l2"]) < 0.01
