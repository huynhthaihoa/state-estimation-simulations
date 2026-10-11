import numpy as np
import pytest
manif = pytest.importorskip("manifpy")


@pytest.fixture
def ise(import_module, use_manif_dir):
    return import_module(use_manif_dir, "invariant_saltation_ekf")


@pytest.fixture
def ise_np(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "invariant_saltation_ekf")


G_VEC = np.array([0.0, 0.0, -9.81])
BODY_RATE = np.array([1.0, -0.6, 2.0])
# manif tangent index i holds use_numpy tangent index TO_MANIF[i]:
# manif (rho, theta, nu) = use_numpy (xi_p, xi_R, xi_v).
TO_MANIF = np.array([6, 7, 8, 0, 1, 2, 3, 4, 5])


def tangent(xi):
    return manif.SE_2_3Tangent(np.asarray(xi, dtype=float))


def random_state(m, rng, height=1.0):
    rot = manif.SO3Tangent(rng.normal(0.0, 0.5, 3)).exp()
    return m.se23_from(rot, rng.normal(0.0, 1.0, 3), np.array([rng.normal(), rng.normal(), height]))


def to_numpy_state(m_np, X):
    return m_np.se23_make(X.rotation(), X.linearVelocity(), X.translation())


def permute(P):
    return P[np.ix_(TO_MANIF, TO_MANIF)]


# --- manif conventions this script relies on, against the use_numpy helpers ---

def test_manif_tangent_layout_and_adjoint_match_use_numpy(ise, ise_np):
    rng = np.random.default_rng(0)
    for _ in range(3):
        xi_np = rng.normal(0.0, 1.0, 9)
        X = tangent(xi_np[TO_MANIF]).exp()
        X_np = ise_np.se23_exp(xi_np)
        assert np.allclose(to_numpy_state(ise_np, X), X_np, atol=1e-12)
        assert np.allclose(X.adj(), permute(ise_np.se23_adjoint(X_np)), atol=1e-12)


def test_lminus_is_the_right_invariant_error(ise, ise_np):
    rng = np.random.default_rng(1)
    X1, X2 = random_state(ise, rng), random_state(ise, rng)
    xi_np = ise_np.se23_log(to_numpy_state(ise_np, X1) @ ise_np.se23_inv(to_numpy_state(ise_np, X2)))
    assert np.allclose(X1.lminus(X2).coeffs(), xi_np[TO_MANIF], atol=1e-10)


def test_propagate_mean_matches_use_numpy(ise, ise_np):
    rng = np.random.default_rng(2)
    X = random_state(ise, rng)
    gyro, accel, h = BODY_RATE, rng.normal(0.0, 1.0, 3), 0.2
    expected = ise_np.propagate_mean(to_numpy_state(ise_np, X), gyro, accel, h, G_VEC)
    got = ise.propagate_mean(X, gyro, accel, h, G_VEC)
    assert np.allclose(to_numpy_state(ise_np, got), expected, atol=1e-12)


# --- Propagation: group-affine, so the invariant error evolves exactly linearly ---

def test_invariant_error_propagates_exactly_even_for_large_errors(ise):
    rng = np.random.default_rng(3)
    X = random_state(ise, rng, height=50.0)
    gyro, accel, h = BODY_RATE, rng.normal(0.0, 1.0, 3), 0.2
    xi = rng.normal(0.0, 0.5, 9)
    X_next = ise.propagate_mean(X, gyro, accel, h, G_VEC)
    X_pert_next = ise.propagate_mean(X.lplus(tangent(xi)), gyro, accel, h, G_VEC)
    xi_next = X_pert_next.lminus(X_next).coeffs()
    assert np.allclose(xi_next, ise.propagation_matrix(h, G_VEC) @ xi, atol=1e-10)


# --- Guard / reset ---

def test_crossing_time_and_reset(ise):
    X = ise.se23_from(manif.SO3.Identity(), np.zeros(3), np.array([0.0, 0.0, 5.0]))
    tau = ise.crossing_time(X, np.zeros(3), G_VEC)
    assert np.isclose(tau, np.sqrt(2 * 5.0 / 9.81), atol=1e-12)
    assert np.isclose(ise.propagate_mean(X, BODY_RATE, np.zeros(3), tau, G_VEC).translation()[2], 0.0,
                      atol=1e-12)
    X_hit = ise.se23_from(manif.SO3.Identity(), np.array([1.0, 2.0, -3.0]), np.zeros(3))
    assert np.allclose(ise.reset_map(X_hit, 0.8).linearVelocity(), [1.0, 2.0, 2.4])


def test_reset_is_not_an_automorphism_except_for_vertical_axis_rotations(ise):
    rng = np.random.default_rng(4)
    X1, X2 = random_state(ise, rng), random_state(ise, rng)
    assert ise.reset_is_automorphism_gap(X1, X2, 0.8) > 0.1
    X1z = ise.se23_from(manif.SO3Tangent(np.array([0.0, 0.0, 0.7])).exp(),
                        rng.normal(0.0, 1.0, 3), rng.normal(0.0, 1.0, 3))
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
        Yp, _ = hybrid_fixed_time_map(m, X0.lplus(tangent(d)), T_total, e)
        Ym, _ = hybrid_fixed_time_map(m, X0.lplus(tangent(-d)), T_total, e)
        J[:, i] = (Yp.lminus(Y0).coeffs() - Ym.lminus(Y0).coeffs()) / (2 * step)
    return J


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_saltation_matrix_ri_matches_finite_differences(ise, seed):
    rng = np.random.default_rng(seed)
    e, T_total = 0.8, 0.8
    X0 = ise.se23_from(manif.SO3Tangent(rng.normal(0.0, 0.5, 3)).exp(),
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


# --- Measurement Jacobians ---

def test_measurement_jacobian_ri_equals_manifs_act_jacobian_at_a_noise_free_measurement(ise):
    # r_i = X_hat.act(y_i) - l_i. manif's act Jacobian is wrt a right
    # perturbation; X Exp(Ad(X^-1) xi) = Exp(xi) X moves it to the left
    # (right-invariant) error, and at y_i = X^-1.act(l_i) it is H_i exactly.
    rng = np.random.default_rng(5)
    landmarks = ise.default_landmarks()
    H = ise.measurement_jacobian_ri(landmarks)
    for _ in range(2):
        X = random_state(ise, rng)
        y = ise.measure_landmarks(X, landmarks)
        J_left = []
        for y_i in y:
            J_act = np.zeros((3, 9))
            X.act(y_i, J_act)
            J_left.append(J_act @ X.inverse().adj())
        assert np.allclose(np.vstack(J_left), H, atol=1e-10)


def test_impact_timing_direction_matches_use_numpy(ise, ise_np):
    rng = np.random.default_rng(8)
    X = random_state(ise, rng, height=0.0)
    w = ise.impact_timing_direction_ri(X, BODY_RATE, np.zeros(3), 0.8, G_VEC)
    w_np = ise_np.impact_timing_direction_ri(to_numpy_state(ise_np, X), BODY_RATE, np.zeros(3), 0.8, G_VEC)
    assert np.allclose(w, w_np[TO_MANIF], atol=1e-12)


def test_world_measurement_jacobian_matches_finite_differences(ise):
    rng = np.random.default_rng(6)
    landmarks = ise.default_landmarks()
    X = random_state(ise, rng)

    def predict(d):
        rot = manif.SO3Tangent(d[3:6]).exp() * ise.rotation_of(X)
        Xd = ise.se23_from(rot, X.linearVelocity() + d[6:9], X.translation() + d[0:3])
        return ise.measure_landmarks(Xd, landmarks).reshape(-1)

    J = np.column_stack([(predict(d) - predict(-d)) / 2e-6 for d in 1e-6 * np.eye(9)])
    assert np.allclose(J, ise.measurement_jacobian_world(X, landmarks), atol=1e-6)


# --- Cross-backend equivalence and end-to-end regression ---

@pytest.fixture
def default_scenario(ise):
    X0 = ise.se23_from(manif.SO3Tangent(np.array([0.2, -0.1, 0.3])).exp(), np.array([1.0, 0.5, 0.0]),
                       np.array([0.0, 0.0, 5.0]))
    P0 = np.diag([0.1 ** 2] * 3 + [np.radians(5.0) ** 2] * 3 + [0.2 ** 2] * 3)
    return X0, P0


@pytest.mark.parametrize("jitter", [0.0, 0.003])
@pytest.mark.parametrize("name", ["invariant_naive", "reset_jacobian", "saltation", "saltation_timing",
                                  "world_saltation"])
def test_every_filter_matches_use_numpy_on_the_same_data(ise, ise_np, default_scenario, name, jitter):
    # With jitter, both backends get identically seeded detection Generators.
    X0, P0 = default_scenario
    landmarks = ise.default_landmarks()
    X_true, gyro, accel, meas, impact_ticks = ise.generate_ground_truth_and_data(
        5.0, 0.01, 0.8, G_VEC, X0, BODY_RATE, landmarks, 10, 0.02, 0.3, 0.05, np.random.default_rng(0))
    X_init, P_init = ise.initial_estimate(X0, P0, np.random.default_rng(1))
    X_np_init = to_numpy_state(ise_np, X_init)
    P_np_init = P_init[name][np.ix_(np.argsort(TO_MANIF), np.argsort(TO_MANIF))]
    detect = (0.0, jitter)
    X_list, P_list, counts = ise.run_filter(name, X_init, P_init[name], gyro, accel, meas, 0.01, 0.8, G_VEC,
                                            landmarks, 0.02, 0.3, 0.05, *detect, np.random.default_rng(2))
    X_np_list, P_np_list, counts_np = ise_np.run_filter(name, X_np_init, P_np_init, gyro, accel, meas, 0.01,
                                                        0.8, G_VEC, landmarks, 0.02, 0.3, 0.05, *detect,
                                                        np.random.default_rng(2))
    assert counts[-1] == len(impact_ticks) and np.array_equal(counts, counts_np)
    for k in range(0, len(X_list), 25):
        assert np.allclose(to_numpy_state(ise_np, X_list[k]), X_np_list[k], atol=1e-8)
        assert np.allclose(P_list[k], permute(P_np_list[k]), rtol=1e-6, atol=1e-12)


def test_truth_matches_use_numpy(ise, ise_np, default_scenario):
    X0, _ = default_scenario
    landmarks = ise.default_landmarks()
    args = (5.0, 0.01, 0.8, G_VEC)
    rest = (BODY_RATE, landmarks, 10, 0.02, 0.3, 0.05)
    X_true, gyro, _, meas, impact_ticks = ise.generate_ground_truth_and_data(
        *args, X0, *rest, np.random.default_rng(0))
    X_np, gyro_np, _, meas_np, impact_np = ise_np.generate_ground_truth_and_data(
        *args, to_numpy_state(ise_np, X0), *rest, np.random.default_rng(0))
    assert impact_ticks == impact_np == [101, 263, 392, 496]
    assert np.array_equal(gyro, gyro_np)
    assert np.allclose(to_numpy_state(ise_np, X_true[-1]), X_np[-1], atol=1e-10)
    assert all(np.allclose(meas[k], meas_np[k], atol=1e-12) for k in meas)


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
    pre = slice(1, impact_ticks[0] - 1)
    assert np.nanmean(avg["world_saltation"][pre]) > 1.2 * np.nanmean(avg["saltation"][pre])
    assert all(rate < 0.02 for rate in mismatch.values())


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
    assert all(rate < 0.02 for rate in mismatch.values())
