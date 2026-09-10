import importlib
import sys

import numpy as np
import pytest
manif = pytest.importorskip("manifpy")


@pytest.fixture
def pointcloud_pose_tracking(import_module, use_manif_dir):
    return import_module(use_manif_dir, "pointcloud_pose_tracking")


@pytest.fixture(scope="module")
def scenario(use_manif_dir):
    # Built once (module scope) and shared across the scenario tests below --
    # run_batch_gn alone takes ~1s, so recomputing per-test would be wasteful.
    # `import_module` is function-scoped (by design, see conftest.py), so it
    # can't be depended on here; replicate its isolated-import logic directly.
    directory = str(use_manif_dir)
    modules_before = set(sys.modules)
    sys.path.insert(0, directory)
    try:
        importlib.invalidate_caches()
        m = importlib.import_module("pointcloud_pose_tracking")
    finally:
        sys.path.remove(directory)
        for name in set(sys.modules) - modules_before:
            del sys.modules[name]

    rng = np.random.default_rng(0)
    duration, dt, n_points = 5.0, 0.1, 20
    vel_noise_std, gyro_noise_std, point_noise_std = 0.05, 0.02, 0.03
    init_pose_noise_std = 0.1

    body_points, T_true, u_meas, z = m.generate_ground_truth_and_data(
        duration, dt, n_points, vel_noise_std, gyro_noise_std, point_noise_std, rng)

    init_offset = rng.normal(0.0, init_pose_noise_std, 6)
    T_init = T_true[0].rplus(manif.SE3Tangent(init_offset))
    P_init = init_pose_noise_std ** 2 * np.eye(6)
    Q_rate = np.diag(np.concatenate([[vel_noise_std ** 2] * 3, [gyro_noise_std ** 2] * 3]))
    Q_tangent = dt ** 2 * Q_rate

    T_dr = m.run_dead_reckoning(T_init, u_meas, dt)
    T_ekf = m.run_ekf(T_init, P_init, u_meas, z, body_points, dt, Q_tangent, point_noise_std)
    T_iekf = m.run_iekf(T_init, P_init, u_meas, z, body_points, dt, Q_tangent, point_noise_std)
    T_ukf = m.run_ukf(T_init, P_init, u_meas, z, body_points, dt, Q_tangent, point_noise_std,
                       1.0, 2.0, -3.0)
    T_gn = m.run_batch_gn(T_dr, T_init, P_init, u_meas, z, body_points, dt, Q_tangent,
                           point_noise_std, 1e-6, 20)

    return {
        "m": m, "T_true": T_true,
        "T_dr": T_dr, "T_ekf": T_ekf, "T_iekf": T_iekf, "T_ukf": T_ukf, "T_gn": T_gn,
    }


def _rms(arr):
    return np.sqrt(np.mean(arr ** 2))


# --- make_body_point_cloud / generate_ground_truth_and_data ---

def test_make_body_point_cloud_shape_and_bounds(pointcloud_pose_tracking):
    rng = np.random.default_rng(0)
    pts = pointcloud_pose_tracking.make_body_point_cloud(15, rng, half_extent=0.5)
    assert pts.shape == (15, 3)
    assert np.all(np.abs(pts) <= 0.5)


def test_generate_ground_truth_and_data_shapes_and_initial_pose(pointcloud_pose_tracking):
    rng = np.random.default_rng(1)
    duration, dt, n_points = 1.0, 0.1, 8
    body_points, T_true, u_meas, z = pointcloud_pose_tracking.generate_ground_truth_and_data(
        duration, dt, n_points, 0.05, 0.02, 0.03, rng)

    n_steps = int(duration / dt)
    assert body_points.shape == (n_points, 3)
    assert len(T_true) == n_steps + 1
    assert np.allclose(T_true[0].translation(), np.zeros(3), atol=1e-12)
    assert np.allclose(T_true[0].rotation(), np.eye(3), atol=1e-12)
    assert u_meas.shape == (n_steps, 6)
    assert len(z) == n_steps + 1
    for zk in z:
        assert zk.shape == (n_points, 3)
    for T in T_true:
        R = T.rotation()
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-8)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)


# --- Jacobian finite-difference checks ---

def test_motion_model_jacobians_match_finite_difference(pointcloud_pose_tracking):
    m = pointcloud_pose_tracking
    rng = np.random.default_rng(7)
    xi0 = np.concatenate([rng.normal(size=3), rng.normal(size=3) * 0.3])
    T_prev = manif.SE3.Identity().rplus(manif.SE3Tangent(xi0))
    # A small-magnitude twist*dt keeps the perturbation near the SE(3) identity,
    # where manif's right Jacobian is at its most numerically accurate; this
    # keeps the check tight without depending on behavior far from zero.
    twist = rng.normal(size=6) * 0.05
    dt = 0.02

    J_self, J_tau = np.zeros((6, 6)), np.zeros((6, 6))
    T_pred = m.motion_model(T_prev, twist, dt, J_self, J_tau)

    eps = 1e-6
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        T_prev_pert = T_prev.rplus(manif.SE3Tangent(d))
        T_pred_pert = m.motion_model(T_prev_pert, twist, dt)
        numeric_col = T_pred_pert.rminus(T_pred).coeffs() / eps
        assert np.allclose(numeric_col, J_self[:, i], atol=1e-4)

    for i in range(6):
        dtau = np.zeros(6)
        dtau[i] = eps
        twist_pert = twist + dtau / dt  # perturbs tau = twist*dt by exactly dtau
        T_pred_pert = m.motion_model(T_prev, twist_pert, dt)
        numeric_col = T_pred_pert.rminus(T_pred).coeffs() / eps
        assert np.allclose(numeric_col, J_tau[:, i], atol=1e-4)


def test_observation_model_jacobian_matches_finite_difference(pointcloud_pose_tracking):
    m = pointcloud_pose_tracking
    rng = np.random.default_rng(8)
    xi = np.concatenate([rng.normal(size=3), rng.normal(size=3) * 0.5])
    T = manif.SE3.Identity().rplus(manif.SE3Tangent(xi))
    body_points = m.make_body_point_cloud(10, rng)

    pred, J = m.observation_model(T, body_points, with_jacobian=True)
    assert pred.shape == (10, 3)
    assert J.shape == (30, 6)

    eps = 1e-6
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        T_pert = T.rplus(manif.SE3Tangent(d))
        pred_pert, _ = m.observation_model(T_pert, body_points)
        numeric_col = (pred_pert - pred).reshape(-1) / eps
        assert np.allclose(numeric_col, J[:, i], atol=1e-4)


def test_observation_model_without_jacobian_returns_none(pointcloud_pose_tracking):
    m = pointcloud_pose_tracking
    rng = np.random.default_rng(9)
    body_points = m.make_body_point_cloud(5, rng)
    pred, J = m.observation_model(manif.SE3.Identity(), body_points, with_jacobian=False)
    assert J is None
    assert np.allclose(pred, body_points, atol=1e-12)  # identity pose: pred == body_points


# --- End-to-end scenario checks ---

def test_all_filters_beat_dead_reckoning_baseline(scenario):
    m = scenario["m"]
    rot_err_dr, pos_err_dr = m.pose_errors(scenario["T_true"], scenario["T_dr"])
    rms_rot_dr, rms_pos_dr = _rms(rot_err_dr), _rms(pos_err_dr)

    for key in ("T_ekf", "T_iekf", "T_ukf", "T_gn"):
        rot_err, pos_err = m.pose_errors(scenario["T_true"], scenario[key])
        assert _rms(rot_err) < rms_rot_dr
        assert _rms(pos_err) < rms_pos_dr


def test_filters_achieve_low_absolute_error(scenario):
    m = scenario["m"]
    for key in ("T_ekf", "T_iekf", "T_ukf"):
        rot_err, pos_err = m.pose_errors(scenario["T_true"], scenario[key])
        assert _rms(rot_err) < 3.0    # degrees
        assert _rms(pos_err) < 0.15   # meters

    rot_err_gn, pos_err_gn = m.pose_errors(scenario["T_true"], scenario["T_gn"])
    assert _rms(rot_err_gn) < 2.0
    assert _rms(pos_err_gn) < 0.1


def test_ekf_and_iekf_agree_to_tight_tolerance(scenario):
    # The run_iekf docstring explicitly claims it produces the *exact same*
    # corrections as run_ekf on this problem (isotropic point-noise covariance
    # makes the world-frame vs. body-frame residual/Jacobian pairs cancel
    # exactly in the Kalman gain) -- verify that algebraic claim directly.
    for T_e, T_i in zip(scenario["T_ekf"], scenario["T_iekf"]):
        assert np.allclose(T_e.coeffs(), T_i.coeffs(), atol=1e-9)


def test_batch_gn_rms_error_beats_ekf(scenario):
    # Batch smoothing has access to the whole trajectory (non-causal), so it
    # should do at least as well as the purely causal recursive EKF.
    m = scenario["m"]
    rot_err_ekf, pos_err_ekf = m.pose_errors(scenario["T_true"], scenario["T_ekf"])
    rot_err_gn, pos_err_gn = m.pose_errors(scenario["T_true"], scenario["T_gn"])
    assert _rms(rot_err_gn) <= _rms(rot_err_ekf)
    assert _rms(pos_err_gn) <= _rms(pos_err_ekf)


def test_pose_errors_zero_for_identical_trajectories(pointcloud_pose_tracking):
    m = pointcloud_pose_tracking
    rng = np.random.default_rng(2)
    T_list = [manif.SE3.Identity().rplus(manif.SE3Tangent(rng.normal(size=6) * 0.3)) for _ in range(5)]
    rot_err, pos_err = m.pose_errors(T_list, T_list)
    assert np.allclose(rot_err, 0.0, atol=1e-8)
    assert np.allclose(pos_err, 0.0, atol=1e-10)
