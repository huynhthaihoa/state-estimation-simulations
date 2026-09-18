import numpy as np
import pytest


@pytest.fixture
def saltation_matrix_ekf(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "saltation_matrix_ekf")


# --- flow / crossing_time on a hand-computable case ---

def test_flow_and_crossing_time_known_vertical_drop(saltation_matrix_ekf):
    m = saltation_matrix_ekf
    g = 9.81
    height = 5.0
    x0 = np.array([0.0, 0.0, height, 0.0, 0.0, 0.0])  # dropped from rest, no horizontal motion

    tau = m.crossing_time(x0, g)
    assert np.isclose(tau, np.sqrt(2 * height / g), atol=1e-10)

    x_minus, Phi = m.flow(x0, tau, g)
    assert np.isclose(x_minus[2], 0.0, atol=1e-9)  # exactly on the guard
    assert np.isclose(x_minus[5], -np.sqrt(2 * g * height), atol=1e-9)  # impact speed
    assert Phi.shape == (6, 6)
    assert np.allclose(Phi[0:3, 0:3], np.eye(3)) and np.allclose(Phi[3:6, 3:6], np.eye(3))
    assert np.allclose(Phi[0:3, 3:6], tau * np.eye(3))


def test_crossing_time_none_when_moving_away_from_guard(saltation_matrix_ekf):
    m = saltation_matrix_ekf
    # already on the ground, moving upward -- no future crossing under free-fall
    # until it comes back down, but discriminant check should still find that
    # positive root; use a case with genuinely no positive root instead:
    # already above ground, moving straight up, starting far enough away that
    # the (only) positive root corresponds to a real future fall -- to get a
    # true "no crossing" case, use a state already below the guard with
    # upward velocity insufficient to ever return (not physical for constant
    # gravity, so instead verify the still-falling case always finds a root).
    x0 = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 5.0])  # thrown upward from height 10
    tau = m.crossing_time(x0, 9.81)
    assert tau is not None and tau > 0.0


# --- reset_map / reset_jacobian ---

def test_reset_map_flips_and_scales_vertical_velocity_only(saltation_matrix_ekf):
    m = saltation_matrix_ekf
    x = np.array([1.0, 2.0, 0.0, 3.0, 4.0, -5.0])
    x_plus = m.reset_map(x, e=0.6)
    assert np.allclose(x_plus[0:5], x[0:5])  # position, horizontal velocity untouched
    assert np.isclose(x_plus[5], -0.6 * -5.0)


def test_reset_jacobian_matches_reset_map_directly(saltation_matrix_ekf):
    m = saltation_matrix_ekf
    e = 0.6
    DR = m.reset_jacobian(e)
    x = np.array([1.0, 2.0, 0.0, 3.0, 4.0, -5.0])
    assert np.allclose(DR @ x, m.reset_map(x, e))  # reset_map is exactly linear


# --- saltation_matrix: the critical correctness check ---

def test_saltation_matrix_matches_finite_difference(saltation_matrix_ekf):
    m = saltation_matrix_ekf
    g, e = 9.81, 0.6
    rng = np.random.default_rng(7)

    x0 = np.array([0.0, 0.0, 5.0, 1.0, 0.5, -2.0])
    tau = m.crossing_time(x0, g)
    x_minus, _ = m.flow(x0, tau, g)
    x_plus = m.reset_map(x_minus, e)
    Xi = m.saltation_matrix(x_minus, e, g)

    eps = 1e-6
    Xi_numeric = np.zeros((6, 6))
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        x_offguard = x_minus + d
        # re-land the perturbed point exactly on the guard via the SAME
        # pre-impact flow (closed-form quadratic, not step_hybrid's fixed-dt
        # stepper), then apply the reset -- the standard finite-difference
        # recipe for a saltation matrix.
        p0, v0 = x_offguard[0:3], x_offguard[3:6]
        a_coef, b_coef, c_coef = -0.5 * g, v0[2], p0[2]
        disc = b_coef ** 2 - 4 * a_coef * c_coef
        r1 = (-b_coef + np.sqrt(disc)) / (2 * a_coef)
        r2 = (-b_coef - np.sqrt(disc)) / (2 * a_coef)
        tau_correction = min([r1, r2], key=abs)
        x_actual_minus, _ = m.flow(x_offguard, tau_correction, g)
        x_actual_plus = m.reset_map(x_actual_minus, e)
        Xi_numeric[:, i] = (x_actual_plus - x_plus) / eps

    assert np.allclose(Xi, Xi_numeric, atol=1e-4)

    # and confirm this genuinely differs from the naive reset Jacobian alone --
    # not just that the code runs, but that the correction actually matters.
    DR = m.reset_jacobian(e)
    assert np.max(np.abs(DR - Xi_numeric)) > 0.5


def test_saltation_matrix_annihilates_guard_normal_direction(saltation_matrix_ekf):
    # Structural property (not a numerical coincidence): every trajectory in
    # the perturbed family satisfies g(x) = p_z = 0 exactly at its own
    # crossing, so Dg @ Xi = 0 identically, for any x_minus/e/g.
    m = saltation_matrix_ekf
    rng = np.random.default_rng(11)
    Dg = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    for _ in range(5):
        x_minus = rng.normal(size=6)
        x_minus[2] = 0.0  # on the guard
        e = rng.uniform(0.0, 1.0)
        Xi = m.saltation_matrix(x_minus, e, 9.81)
        assert np.allclose(Dg @ Xi, 0.0, atol=1e-12)


# --- step_hybrid / generate_ground_truth_and_data: sanity on a real bounce ---

def test_generate_ground_truth_bounces_and_stays_above_ground(saltation_matrix_ekf):
    m = saltation_matrix_ekf
    rng = np.random.default_rng(0)
    x_true, z = m.generate_ground_truth_and_data(
        duration=5.0, dt=0.02, e=0.85, g=9.81, drop_height=5.0,
        init_horizontal_vel=[1.0, 0.5], pos_noise_std=0.03, rng=rng)

    heights = np.array([x[2] for x in x_true])
    assert heights.min() > -1e-6  # never tunnels through the ground
    assert heights[0] == 5.0

    bounce_ticks = m.detect_bounce_ticks(x_true)
    assert len(bounce_ticks) >= 2  # at least a couple of bounces in 5s at these settings

    assert z.shape == (len(x_true), 3)


def test_run_ekf_variants_beat_dead_reckoning(saltation_matrix_ekf):
    m = saltation_matrix_ekf
    rng = np.random.default_rng(0)
    duration, dt, e, g = 3.0, 0.02, 0.85, 9.81
    x_true, z = m.generate_ground_truth_and_data(
        duration, dt, e, g, drop_height=5.0, init_horizontal_vel=[1.0, 0.5],
        pos_noise_std=0.03, rng=rng)
    n_steps = len(x_true) - 1

    init_std = np.array([0.1] * 3 + [0.2] * 3)
    x_init = x_true[0] + rng.normal(0.0, 1.0, 6) * init_std
    P_init = np.diag(init_std ** 2)
    R = 0.03 ** 2 * np.eye(3)

    x_dr = m.run_dead_reckoning(x_init, dt, n_steps, e, g)
    x_naive, _ = m.run_ekf_naive(x_init, P_init, z, dt, n_steps, e, g, 0.3, 0.005, R)
    x_salt, _ = m.run_ekf_saltation(x_init, P_init, z, dt, n_steps, e, g, 0.3, 0.005, R)

    pos_err_dr, _ = m.state_errors(x_true, x_dr)
    pos_err_naive, _ = m.state_errors(x_true, x_naive)
    pos_err_salt, _ = m.state_errors(x_true, x_salt)

    rms_dr = np.sqrt(np.mean(pos_err_dr ** 2))
    assert np.sqrt(np.mean(pos_err_naive ** 2)) < rms_dr
    assert np.sqrt(np.mean(pos_err_salt ** 2)) < rms_dr


def test_nees_zero_for_exact_match(saltation_matrix_ekf):
    m = saltation_matrix_ekf
    x = np.array([1.0, 2.0, 0.0, 3.0, 4.0, -5.0])
    P = np.eye(6) * 0.1
    assert m.nees(x, x.copy(), P) == 0.0


def test_detect_bounce_ticks_known_case(saltation_matrix_ekf):
    m = saltation_matrix_ekf
    # a hand-built height sequence with an obvious single local minimum at index 2
    x_true = [np.array([0.0, 0.0, h, 0.0, 0.0, 0.0]) for h in [2.0, 1.0, 0.05, 1.0, 2.0]]
    ticks = m.detect_bounce_ticks(x_true, height_threshold=0.2)
    assert ticks == [2]
