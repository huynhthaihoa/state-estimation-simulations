import numpy as np
import pytest

from state_estimation_simulations.use_numpy import eskf_imu_position_fix as _eskf_imu_position_fix
from state_estimation_simulations.use_numpy.lie_utils import so3_exp, so3_log


@pytest.fixture
def eskf_imu_position_fix():
    return _eskf_imu_position_fix


def _trial_kwargs(**overrides):
    kw = dict(duration=10.0, dt=0.01, fix_rate=1.0, rate_amp=0.4, accel_amp=0.5, gyro_noise_std=0.01,
              accel_noise_std=0.05, gyro_bias_std=0.01, accel_bias_std=0.05, pos_noise_std=0.1,
              init_pos_noise_std=0.1, init_vel_noise_std=0.1, init_att_noise_std=np.deg2rad(0.5))
    kw.update(overrides)
    return kw


# --- quaternion helpers ---

def test_quaternion_rotation_matches_so3_exp(eskf_imu_position_fix):
    m = eskf_imu_position_fix
    phi = np.array([0.3, -0.7, 1.1])
    assert np.allclose(m.quat_to_rotation(m.quat_exp(phi)), so3_exp(phi), atol=1e-12)
    assert np.allclose(m.quat_to_rotation(m.rotation_to_quat(so3_exp(phi))), so3_exp(phi), atol=1e-12)


def test_quaternion_multiplication_matrices(eskf_imu_position_fix):
    m = eskf_imu_position_fix
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=4), rng.normal(size=4)
    assert np.allclose(m.quat_left_matrix(a) @ b, m.quat_multiply(a, b))
    assert np.allclose(m.quat_right_matrix(b) @ a, m.quat_multiply(a, b))


def test_d_rotate_dq_matches_finite_difference_off_the_unit_sphere(eskf_imu_position_fix):
    m = eskf_imu_position_fix
    rng = np.random.default_rng(1)
    q, a, eps = 1.3 * m.quat_exp(rng.normal(size=3)), rng.normal(size=3), 1e-6
    numeric = np.column_stack([(m.quat_to_rotation(q + eps * e) @ a - m.quat_to_rotation(q - eps * e) @ a)
                               / (2 * eps) for e in np.eye(4)])
    assert np.allclose(m.d_rotate_dq(q, a), numeric, atol=1e-8)


def test_quat_error_jacobian_recovers_small_rotation(eskf_imu_position_fix):
    m = eskf_imu_position_fix
    q_hat = m.quat_exp(np.array([0.4, 0.2, -0.9]))
    dtheta = np.array([1e-5, -2e-5, 3e-5])
    q = m.quat_multiply(q_hat, m.quat_exp(dtheta))
    assert np.allclose(m.quat_error_jacobian(q_hat) @ (q - q_hat), dtheta, atol=1e-9)


# --- predict-step Jacobians against finite differences ---

def _eskf_inject(m, nominal, dx):
    return dict(p=nominal["p"] + dx[0:3], v=nominal["v"] + dx[3:6], R=nominal["R"] @ so3_exp(dx[6:9]),
                b_a=nominal["b_a"] + dx[9:12], b_g=nominal["b_g"] + dx[12:15])


def _eskf_difference(true, nominal):
    return np.r_[true["p"] - nominal["p"], true["v"] - nominal["v"], so3_log(nominal["R"].T @ true["R"]),
                 true["b_a"] - nominal["b_a"], true["b_g"] - nominal["b_g"]]


def test_eskf_transition_matches_finite_difference(eskf_imu_position_fix):
    m = eskf_imu_position_fix
    rng = np.random.default_rng(2)
    nominal = dict(R=so3_exp(rng.normal(size=3)), v=rng.normal(size=3), p=rng.normal(size=3),
                   b_a=0.05 * rng.normal(size=3), b_g=0.01 * rng.normal(size=3))
    gyro, accel, dt, eps, Z = 0.5 * rng.normal(size=3), np.array([0.3, -0.2, 9.81]), 0.01, 1e-6, np.zeros((12, 12))
    nom_next, _ = m.eskf_predict(nominal, np.eye(15), gyro, accel, dt, Z)
    numeric = np.zeros((15, 15))
    for i in range(15):
        plus, _ = m.eskf_predict(_eskf_inject(m, nominal, eps * np.eye(15)[i]), np.eye(15), gyro, accel, dt, Z)
        minus, _ = m.eskf_predict(_eskf_inject(m, nominal, -eps * np.eye(15)[i]), np.eye(15), gyro, accel, dt, Z)
        numeric[:, i] = (_eskf_difference(plus, nom_next) - _eskf_difference(minus, nom_next)) / (2 * eps)
    F, _ = m.eskf_transition(nominal, gyro, accel, dt)
    assert np.allclose(F, numeric, atol=1e-8)


def test_qekf_transition_matches_finite_difference(eskf_imu_position_fix):
    m = eskf_imu_position_fix
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(size=3), rng.normal(size=3), m.quat_exp(rng.normal(size=3)),
              0.05 * rng.normal(size=3), 0.01 * rng.normal(size=3)]
    gyro, accel, dt, eps, Z = 0.5 * rng.normal(size=3), np.array([0.3, -0.2, 9.81]), 0.01, 1e-6, np.zeros((12, 12))
    numeric = np.column_stack([(m.qekf_predict(x + eps * e, np.eye(16), gyro, accel, dt, Z)[0]
                                - m.qekf_predict(x - eps * e, np.eye(16), gyro, accel, dt, Z)[0]) / (2 * eps)
                               for e in np.eye(16)])
    F, _ = m.qekf_transition(x, gyro, accel, dt)
    # dq/db_g uses the first-order increment Jacobian, accurate to O((w dt)^2)
    assert np.allclose(F, numeric, atol=1e-5)


# --- update and reset ---

def test_projected_update_removes_the_norm_direction(eskf_imu_position_fix):
    m = eskf_imu_position_fix
    rng = np.random.default_rng(4)
    x = np.r_[np.zeros(6), m.quat_exp(rng.normal(size=3)), np.zeros(6)]
    A = rng.normal(size=(16, 16))
    P = A @ A.T * 1e-2
    x_new, P_new = m.qekf_update(x, P, np.array([0.3, -0.1, 0.2]), 0.1, project_covariance=True)
    u = x_new[6:10]
    assert np.isclose(np.linalg.norm(u), 1.0)
    assert np.allclose(P_new[6:10, 6:10] @ u, 0.0, atol=1e-6)


def test_eskf_without_noise_or_error_stays_on_the_truth(eskf_imu_position_fix):
    m = eskf_imu_position_fix
    kw = _trial_kwargs(gyro_noise_std=0.0, accel_noise_std=0.0, gyro_bias_std=0.0, accel_bias_std=0.0,
                       pos_noise_std=0.0, init_pos_noise_std=0.0, init_vel_noise_std=0.0, init_att_noise_std=0.0)
    trial = m.generate_trial(rng=np.random.default_rng(5), **kw)
    P0 = m.initial_covariance(0.1, 0.1, 0.01, 0.05, 0.01)
    Q = m.imu_noise_covariance(0.05, 0.01, 1e-4, 1e-5)
    att, pos, _ = m.trial_errors(trial, m.run_eskf(trial, P0, Q, 0.1), 15)
    assert att.max() < 1e-9 and pos.max() < 1e-9


# --- Monte Carlo behaviour (small, fast versions of the script's findings) ---

def _mc(m, att_deg, n_trials, seed):
    kw = _trial_kwargs(init_att_noise_std=np.deg2rad(att_deg))
    P0 = m.initial_covariance(0.1, 0.1, np.deg2rad(att_deg), 0.05, 0.01)
    Q = m.imu_noise_covariance(0.05, 0.01, 1e-4, 1e-5)
    return m.run_monte_carlo(n_trials, kw, P0, Q, 0.1, np.random.default_rng(seed))


def test_eskf_is_consistent_and_no_bias_filter_is_not(eskf_imu_position_fix):
    m = eskf_imu_position_fix
    res = _mc(m, 0.5, 8, 6)
    eskf_nees = m.second_half_mean(res["eskf"]["nees"])
    assert 10.0 < eskf_nees < 22.0
    assert m.second_half_mean(res["eskf_no_bias"]["nees"]) > 10 * 9
    assert m.second_half_mean(res["eskf_no_bias"]["att_rmse"]) > m.second_half_mean(res["eskf"]["att_rmse"])
