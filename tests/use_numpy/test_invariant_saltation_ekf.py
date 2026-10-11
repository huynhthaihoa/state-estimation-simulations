import numpy as np
import pytest
from scipy.linalg import expm


@pytest.fixture
def ise(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "invariant_saltation_ekf")


G_VEC = np.array([0.0, 0.0, -9.81])
BODY_RATE = np.array([1.0, -0.6, 2.0])


def hat9(m, xi):
    M = np.zeros((5, 5))
    M[0:3, 0:3] = m.skew(xi[0:3])
    M[0:3, 3] = xi[3:6]
    M[0:3, 4] = xi[6:9]
    return M


def random_state(m, rng, height=1.0):
    return m.se23_make(m.so3_exp(rng.normal(0.0, 0.5, 3)), rng.normal(0.0, 1.0, 3),
                       np.array([rng.normal(), rng.normal(), height]))


# --- SE_2(3) helpers ---

def test_se23_exp_matches_matrix_exponential_and_log_round_trips(ise):
    rng = np.random.default_rng(0)
    for _ in range(5):
        xi = rng.normal(0.0, 1.0, 9)
        assert np.allclose(ise.se23_exp(xi), expm(hat9(ise, xi)), atol=1e-12)
        assert np.allclose(ise.se23_log(ise.se23_exp(xi)), xi, atol=1e-10)


def test_se23_inverse_and_adjoint(ise):
    rng = np.random.default_rng(1)
    X = random_state(ise, rng)
    assert np.allclose(X @ ise.se23_inv(X), np.eye(5), atol=1e-12)
    xi = rng.normal(0.0, 1.0, 9)
    lhs = hat9(ise, ise.se23_adjoint(X) @ xi)
    rhs = X @ hat9(ise, xi) @ ise.se23_inv(X)
    assert np.allclose(lhs, rhs, atol=1e-12)


# --- Propagation: group-affine, so the invariant error evolves exactly linearly ---

def test_propagation_matrix_is_expm_of_constant_A(ise):
    h = 0.37
    A = np.zeros((9, 9))
    A[3:6, 0:3] = ise.skew(G_VEC)
    A[6:9, 3:6] = np.eye(3)
    assert np.allclose(ise.propagation_matrix(h, G_VEC), expm(A * h), atol=1e-12)


def test_invariant_error_propagates_exactly_even_for_large_errors(ise):
    # log(f(exp(xi) X) f(X)^-1) == Phi xi for a finite (not infinitesimal) xi:
    # the defining property of a group-affine system, independent of X.
    rng = np.random.default_rng(2)
    X = random_state(ise, rng, height=50.0)
    gyro, accel, h = BODY_RATE, rng.normal(0.0, 1.0, 3), 0.2
    xi = rng.normal(0.0, 0.5, 9)
    X_next = ise.propagate_mean(X, gyro, accel, h, G_VEC)
    X_pert_next = ise.propagate_mean(ise.se23_exp(xi) @ X, gyro, accel, h, G_VEC)
    xi_next = ise.se23_log(X_pert_next @ ise.se23_inv(X_next))
    assert np.allclose(xi_next, ise.propagation_matrix(h, G_VEC) @ xi, atol=1e-10)


# --- Guard / reset ---

def test_crossing_time_known_vertical_drop(ise):
    X = ise.se23_make(np.eye(3), np.zeros(3), np.array([0.0, 0.0, 5.0]))
    tau = ise.crossing_time(X, np.zeros(3), G_VEC)
    assert np.isclose(tau, np.sqrt(2 * 5.0 / 9.81), atol=1e-12)
    X_minus = ise.propagate_mean(X, BODY_RATE, np.zeros(3), tau, G_VEC)
    assert np.isclose(X_minus[2, 4], 0.0, atol=1e-12)


def test_reset_reflects_normal_velocity_only(ise):
    X = ise.se23_make(np.eye(3), np.array([1.0, 2.0, -3.0]), np.zeros(3))
    _, v_plus, _ = ise.se23_parts(ise.reset_map(X, 0.8))
    assert np.allclose(v_plus, [1.0, 2.0, 2.4])


def test_reset_is_not_an_automorphism_except_for_vertical_axis_rotations(ise):
    rng = np.random.default_rng(3)
    X1, X2 = random_state(ise, rng), random_state(ise, rng)
    assert ise.reset_is_automorphism_gap(X1, X2, 0.8) > 0.1
    # M = diag(1, 1, -e) commutes with rotations about e_z, so then it is one.
    Rz = ise.so3_exp(np.array([0.0, 0.0, 0.7]))
    X1z = ise.se23_make(Rz, rng.normal(0.0, 1.0, 3), rng.normal(0.0, 1.0, 3))
    assert ise.reset_is_automorphism_gap(X1z, X2, 0.8) < 1e-12


# --- The saltation matrix on the group, against a finite-difference ground truth ---

def hybrid_fixed_time_map(m, X, T_total, e):
    accel = np.zeros(3)
    tau = m.crossing_time(X, accel, G_VEC)
    assert tau is not None and tau < T_total
    X_minus = m.propagate_mean(X, BODY_RATE, accel, tau, G_VEC)
    return m.propagate_mean(m.reset_map(X_minus, e), BODY_RATE, accel, T_total - tau, G_VEC), tau


def fd_jacobian_ri(m, X0, T_total, e, step=1e-6):
    Y0, _ = hybrid_fixed_time_map(m, X0, T_total, e)
    J = np.zeros((9, 9))
    for i in range(9):
        d = np.zeros(9)
        d[i] = step
        Yp, _ = hybrid_fixed_time_map(m, m.se23_exp(d) @ X0, T_total, e)
        Ym, _ = hybrid_fixed_time_map(m, m.se23_exp(-d) @ X0, T_total, e)
        J[:, i] = (m.se23_log(Yp @ m.se23_inv(Y0)) - m.se23_log(Ym @ m.se23_inv(Y0))) / (2 * step)
    return J


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_saltation_matrix_ri_matches_finite_differences(ise, seed):
    rng = np.random.default_rng(seed)
    e, T_total = 0.8, 0.8
    X0 = ise.se23_make(ise.so3_exp(rng.normal(0.0, 0.5, 3)),
                       np.array([rng.normal(), rng.normal(), -1.0]),
                       np.array([rng.normal(), rng.normal(), 1.0]))
    J_fd = fd_jacobian_ri(ise, X0, T_total, e)
    _, tau = hybrid_fixed_time_map(ise, X0, T_total, e)
    X_minus = ise.propagate_mean(X0, BODY_RATE, np.zeros(3), tau, G_VEC)
    Phi_before = ise.propagation_matrix(tau, G_VEC)
    Phi_after = ise.propagation_matrix(T_total - tau, G_VEC)

    def composed(Xi):
        return Phi_after @ Xi @ Phi_before

    salt = composed(ise.saltation_matrix_ri(X_minus, BODY_RATE, np.zeros(3), e, G_VEC))
    reset_only = composed(ise.reset_jacobian_ri(X_minus, e))
    naive = composed(ise.invariant_naive_reset(e))
    assert np.max(np.abs(salt - J_fd)) < 1e-6
    assert np.max(np.abs(reset_only - J_fd)) > 0.5
    assert np.max(np.abs(naive - J_fd)) > 0.5


def test_saltation_matrix_ri_is_state_dependent(ise):
    rng = np.random.default_rng(4)
    X_minus = ise.se23_make(ise.so3_exp(rng.normal(0.0, 0.5, 3)), np.array([1.0, 0.5, -5.0]),
                            np.array([0.3, -0.2, 0.0]))
    R, v, p = ise.se23_parts(X_minus)
    X_shifted = ise.se23_make(R, v + np.array([0.5, 0.0, 0.0]), p + np.array([0.3, 0.2, 0.0]))
    Xi_a = ise.saltation_matrix_ri(X_minus, BODY_RATE, np.zeros(3), 0.8, G_VEC)
    Xi_b = ise.saltation_matrix_ri(X_shifted, BODY_RATE, np.zeros(3), 0.8, G_VEC)
    assert np.max(np.abs(Xi_a - Xi_b)) > 0.1


# --- Measurement Jacobians ---

def test_measurement_jacobian_ri_matches_finite_differences_and_is_state_free(ise):
    rng = np.random.default_rng(5)
    landmarks = ise.default_landmarks()
    H = ise.measurement_jacobian_ri(landmarks)
    for _ in range(2):
        X = random_state(ise, rng)
        R, _, p = ise.se23_parts(X)
        y = (landmarks - p) @ R  # noise-free

        def residual(xi):
            Rh, _, ph = ise.se23_parts(ise.se23_exp(xi) @ X)
            return (y @ Rh.T + ph - landmarks).reshape(-1)

        J = np.column_stack([(residual(d) - residual(-d)) / 2e-6 for d in 1e-6 * np.eye(9)])
        assert np.allclose(J, H, atol=1e-6)


def test_world_measurement_jacobian_matches_finite_differences(ise):
    rng = np.random.default_rng(6)
    landmarks = ise.default_landmarks()
    X = random_state(ise, rng)
    R, v, p = ise.se23_parts(X)

    def predict(d):
        Rd = ise.so3_exp(d[0:3]) @ R
        return ((landmarks - (p + d[6:9])) @ Rd).reshape(-1)

    J = np.column_stack([(predict(d) - predict(-d)) / 2e-6 for d in 1e-6 * np.eye(9)])
    assert np.allclose(J, ise.measurement_jacobian_world(X, landmarks), atol=1e-6)


# --- End-to-end regression ---

@pytest.fixture
def default_scenario(ise):
    X0 = ise.se23_make(ise.so3_exp(np.array([0.2, -0.1, 0.3])), np.array([1.0, 0.5, 0.0]),
                       np.array([0.0, 0.0, 5.0]))
    P0 = np.diag([np.radians(5.0) ** 2] * 3 + [0.2 ** 2] * 3 + [0.1 ** 2] * 3)
    return X0, P0


def test_truth_has_four_bounces_with_correct_restitution(ise, default_scenario):
    X0, _ = default_scenario
    rng = np.random.default_rng(0)
    X_true, gyro, accel, meas, impact_ticks = ise.generate_ground_truth_and_data(
        5.0, 0.01, 0.8, G_VEC, X0, BODY_RATE, ise.default_landmarks(), 10, 0.02, 0.3, 0.05, rng)
    assert impact_ticks == [101, 263, 392, 496]
    assert len(gyro) == 500 and len(meas) == 50
    # Apex heights shrink by e^2 per bounce (energy loss), since horizontal motion is decoupled.
    heights = np.array([X[2, 4] for X in X_true])
    apex_1 = heights[impact_ticks[0]:impact_ticks[1]].max()
    apex_2 = heights[impact_ticks[1]:impact_ticks[2]].max()
    assert np.isclose(apex_2 / apex_1, 0.8 ** 2, rtol=0.01)


def test_monte_carlo_saltation_is_consistent_and_naive_is_overconfident(ise, default_scenario):
    X0, P0 = default_scenario
    rng = np.random.default_rng(0)
    landmarks = ise.default_landmarks()
    X_true, _, _, _, impact_ticks = ise.generate_ground_truth_and_data(
        5.0, 0.01, 0.8, G_VEC, X0, BODY_RATE, landmarks, 10, 0.02, 0.3, 0.05, rng)
    avg, mismatch = ise.run_monte_carlo_consistency(
        X_true, impact_ticks, BODY_RATE, 0.01, 0.8, G_VEC, landmarks, 10, P0, 0.02, 0.3, 0.05, 30, rng)

    def post_bounce(name):
        return np.nanmean(np.concatenate([avg[name][k + 1:k + 11] for k in impact_ticks]))

    assert post_bounce("invariant_naive") > 1.3 * post_bounce("saltation")
    assert 7.0 < post_bounce("saltation") < 11.0
    # The non-invariant world-frame EKF is the inconsistent one before any bounce.
    pre = slice(1, impact_ticks[0] - 1)
    assert np.nanmean(avg["world_saltation"][pre]) > 1.2 * np.nanmean(avg["saltation"][pre])
    assert all(rate < 0.02 for rate in mismatch.values())


# --- Contact-detection jitter ---

def test_detected_crossing_time_applies_bias_noise_and_clips_into_the_tick(ise):
    assert ise.detected_crossing_time(0.004, 0.01, 0.0, 0.0, None) == 0.004
    assert np.isclose(ise.detected_crossing_time(0.004, 0.01, 0.002, 0.0, None), 0.006)
    assert ise.detected_crossing_time(0.004, 0.01, 0.02, 0.0, None) == 0.01
    assert ise.detected_crossing_time(0.004, 0.01, -0.02, 0.0, None) == 0.0
    draws = [ise.detected_crossing_time(0.5, 1.0, 0.0, 0.01, np.random.default_rng(s)) for s in range(200)]
    assert abs(np.mean(draws) - 0.5) < 0.003 and abs(np.std(draws) - 0.01) < 0.002


def test_impact_timing_direction_matches_finite_difference_of_a_late_reset(ise):
    # Resetting delta seconds after the crossing moves the fixed-time hybrid
    # map's output by -Phi_after w delta, to first order (in RI coordinates).
    rng = np.random.default_rng(7)
    e, T_total = 0.8, 0.8
    accel = np.zeros(3)
    X0 = ise.se23_make(ise.so3_exp(rng.normal(0.0, 0.5, 3)), np.array([rng.normal(), rng.normal(), -1.0]),
                       np.array([rng.normal(), rng.normal(), 1.0]))
    tau = ise.crossing_time(X0, accel, G_VEC)

    def late_reset_map(delta):
        X_reset = ise.reset_map(ise.propagate_mean(X0, BODY_RATE, accel, tau + delta, G_VEC), e)
        return ise.propagate_mean(X_reset, BODY_RATE, accel, T_total - tau - delta, G_VEC)

    Y0 = late_reset_map(0.0)
    d = 1e-6
    J_fd = (ise.se23_log(late_reset_map(d) @ ise.se23_inv(Y0))
            - ise.se23_log(late_reset_map(-d) @ ise.se23_inv(Y0))) / (2 * d)
    X_minus = ise.propagate_mean(X0, BODY_RATE, accel, tau, G_VEC)
    w = ise.impact_timing_direction_ri(X_minus, BODY_RATE, accel, e, G_VEC)
    assert np.allclose(J_fd, -ise.propagation_matrix(T_total - tau, G_VEC) @ w, atol=1e-6)


def test_saltation_timing_is_identical_to_saltation_with_exact_detection(ise, default_scenario):
    X0, P0 = default_scenario
    landmarks = ise.default_landmarks()
    _, gyro, accel, meas, _ = ise.generate_ground_truth_and_data(
        5.0, 0.01, 0.8, G_VEC, X0, BODY_RATE, landmarks, 10, 0.02, 0.3, 0.05, np.random.default_rng(0))
    runs = [ise.run_filter(name, X0, P0, gyro, accel, meas, 0.01, 0.8, G_VEC, landmarks, 0.02, 0.3, 0.05)
            for name in ("saltation", "saltation_timing")]
    assert all(np.array_equal(a, b) for a, b in zip(runs[0][1], runs[1][1]))
    assert all(np.array_equal(a, b) for a, b in zip(runs[0][0], runs[1][0]))


def test_monte_carlo_timing_noise_restores_consistency_under_jitter(ise, default_scenario):
    X0, P0 = default_scenario
    rng = np.random.default_rng(0)
    landmarks = ise.default_landmarks()
    X_true, _, _, _, impact_ticks = ise.generate_ground_truth_and_data(
        5.0, 0.01, 0.8, G_VEC, X0, BODY_RATE, landmarks, 10, 0.02, 0.3, 0.05, rng)
    avg, mismatch = ise.run_monte_carlo_consistency(
        X_true, impact_ticks, BODY_RATE, 0.01, 0.8, G_VEC, landmarks, 10, P0, 0.02, 0.3, 0.05, 30, rng,
        detect_time_bias=0.0, detect_time_noise_std=0.003)

    def post_bounce(name):
        return np.nanmean(np.concatenate([avg[name][k + 1:k + 11] for k in impact_ticks]))

    assert 7.0 < post_bounce("saltation_timing") < 11.0
    assert post_bounce("saltation") > 1.4 * post_bounce("saltation_timing")
    # The point-mass finding (hybrid_saltation_ekf.md section 8) carries over: under
    # jitter, saltation without the timing term is more overconfident than the
    # reset Jacobian alone.
    assert post_bounce("saltation") > post_bounce("reset_jacobian")
    assert all(rate < 0.02 for rate in mismatch.values())
