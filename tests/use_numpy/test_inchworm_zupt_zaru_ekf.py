import numpy as np
import pytest


@pytest.fixture
def zaru(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "inchworm_zupt_zaru_ekf")


def _setup(m, seed=0, omega_extend=0.3, accel_std=0.15, ang_accel_std=0.45):
    rng = np.random.default_rng(seed)
    dt = 0.05
    x_true, z, is_anchor, is_cruise = m.generate_ground_truth_and_data(
        10.0, dt, 1.0, 1.0, 0.2, 0.1, omega_extend, 0.02, rng)
    init_std = np.array([0.05, 0.05, 0.1, 0.05, 0.05])
    x_init = x_true[0] + rng.normal(0.0, 1.0, 5) * init_std
    P_init = np.diag(init_std ** 2)
    return dict(x_true=x_true, z=z, is_anchor=is_anchor, is_cruise=is_cruise, dt=dt,
                x_init=x_init, P_init=P_init, n_steps=len(x_true) - 1,
                accel_std=accel_std, ang_accel_std=ang_accel_std,
                R=(0.02 ** 2, 0.01 ** 2, 0.02 ** 2))


def _run(m, s, mode):
    return m.run_ekf(s["x_init"], s["P_init"], s["z"], s["dt"], s["n_steps"], s["accel_std"],
                     s["ang_accel_std"], *s["R"], s["is_anchor"], mode)


# --- model pieces ---

def test_wrap_angle_range_and_identity(zaru):
    a = np.array([0.0, 3.0, np.pi + 0.1, -np.pi - 0.1, 7.0])
    w = zaru.wrap_angle(a)
    assert np.all(w > -np.pi - 1e-12) and np.all(w <= np.pi + 1e-12)
    np.testing.assert_allclose(np.cos(w), np.cos(a), atol=1e-12)
    np.testing.assert_allclose(np.sin(w), np.sin(a), atol=1e-12)
    assert zaru.wrap_angle(0.5) == pytest.approx(0.5)


def test_motion_jacobian_matches_central_finite_differences(zaru):
    x = np.array([0.3, -0.2, 0.7, 0.4, -0.25])
    dt, eps = 0.05, 1e-6
    F = zaru.motion_jacobian(x, dt)
    F_fd = np.zeros((5, 5))
    for j in range(5):
        d = np.zeros(5)
        d[j] = eps
        F_fd[:, j] = (zaru.propagate(x + d, dt) - zaru.propagate(x - d, dt)) / (2 * eps)
    np.testing.assert_allclose(F, F_fd, atol=1e-8)


def test_process_noise_is_symmetric_psd_with_no_sideways_position_noise(zaru):
    x = np.array([0.0, 0.0, 0.6, 0.1, 0.0])
    Q = zaru.process_noise_covariance(x, 0.05, 0.15, 0.45)
    np.testing.assert_allclose(Q, Q.T, atol=1e-15)
    assert np.linalg.eigvalsh(Q).min() > -1e-15
    heading = np.array([np.cos(0.6), np.sin(0.6)])
    sideways = np.array([-np.sin(0.6), np.cos(0.6)])
    assert sideways @ Q[:2, :2] @ sideways == pytest.approx(0.0, abs=1e-15)
    assert heading @ Q[:2, :2] @ heading > 0.0


# --- truth ---

def test_truth_is_still_in_anchor_and_turns_in_extend(zaru):
    s = _setup(zaru)
    x_true, is_anchor, is_cruise = s["x_true"], s["is_anchor"], s["is_cruise"]
    assert np.all(x_true[is_anchor, 3] == 0.0) and np.all(x_true[is_anchor, 4] == 0.0)
    assert np.all(np.isclose(x_true[is_cruise, 3], 0.1)) and np.all(np.isclose(x_true[is_cruise, 4], 0.3))
    # heading only changes while omega != 0; one anchor tick pair has identical position and heading
    first_anchor_pair = np.where(is_anchor[1:] & is_anchor[:-1])[0][0]
    np.testing.assert_allclose(x_true[first_anchor_pair + 1, :3], x_true[first_anchor_pair, :3])
    assert x_true[-1, 2] != pytest.approx(x_true[0, 2], abs=0.1)


def test_straight_truth_keeps_heading(zaru):
    s = _setup(zaru, omega_extend=0.0)
    np.testing.assert_allclose(s["x_true"][:, 2], 0.0, atol=1e-12)


# --- updates ---

def test_zaru_update_pulls_omega_toward_zero_and_shrinks_its_variance(zaru):
    x = np.array([0.0, 0.0, 0.2, 0.1, 0.4])
    P = np.diag([0.01, 0.01, 0.04, 0.01, 0.09])
    x_new, P_new = zaru.measurement_update_zaru(x, P, 0.02 ** 2)
    assert abs(x_new[4]) < abs(x[4])
    assert P_new[4, 4] < P[4, 4]
    # independent states stay exactly as they were
    np.testing.assert_allclose(x_new[[0, 1, 2, 3]], x[[0, 1, 2, 3]])


def test_zupt_update_pulls_v_toward_zero_without_touching_omega(zaru):
    x = np.array([0.0, 0.0, 0.2, 0.3, 0.4])
    P = np.diag([0.01, 0.01, 0.04, 0.04, 0.09])
    x_new, P_new = zaru.measurement_update_zupt(x, P, 0.01 ** 2)
    assert abs(x_new[3]) < abs(x[3])
    assert x_new[4] == pytest.approx(x[4])


def test_position_update_pulls_position_toward_measurement(zaru):
    x = np.zeros(5)
    P = np.eye(5) * 0.04
    x_new, _ = zaru.measurement_update_position(x, P, np.array([0.1, -0.1]), 0.02 ** 2)
    assert 0.0 < x_new[0] <= 0.1 and -0.1 <= x_new[1] < 0.0


def test_state_error_vector_wraps_heading(zaru):
    err = zaru.state_error_vector(np.array([0, 0, np.pi - 0.05, 0, 0]), np.array([0, 0, -np.pi + 0.05, 0, 0]))
    assert err[2] == pytest.approx(-0.1)


# --- filter ---

def test_run_ekf_rejects_unknown_mode(zaru):
    s = _setup(zaru)
    with pytest.raises(ValueError):
        _run(zaru, s, "sometimes")


def test_gating_policies_apply_the_expected_pseudo_measurements(zaru):
    # At the last tick of an anchor phase: ZARU tightens omega, ZUPT tightens v.
    s = _setup(zaru)
    _, P_never = _run(zaru, s, "never")
    _, P_zupt = _run(zaru, s, "phase_conditional_zupt_only")
    _, P_phase = _run(zaru, s, "phase_conditional")
    k = int(np.where(s["is_anchor"])[0][-1])
    assert P_phase[k][4, 4] < P_zupt[k][4, 4]
    assert P_zupt[k][3, 3] < P_never[k][3, 3]


def test_zaru_stops_heading_uncertainty_growing_during_an_anchor_when_process_noise_covers_the_ramps(zaru):
    # Heading is unobserved at rest (position-only measurements), so without ZARU its variance grows
    # on omega's uncertainty through the dwell; with ZARU it shrinks. Measured at the matched noise
    # level (0.5 m/s^2, 1.5 rad/s^2 = the true ramp accelerations) over anchor phases 2-4.
    s = _setup(zaru, accel_std=0.5, ang_accel_std=1.5)
    _, P_zupt = _run(zaru, s, "phase_conditional_zupt_only")
    _, P_phase = _run(zaru, s, "phase_conditional")
    idx = np.where(s["is_anchor"])[0]
    phases = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)[1:4]
    for r in phases:
        assert P_zupt[r[-1]][2, 2] > P_zupt[r[0]][2, 2]
        assert P_phase[r[-1]][2, 2] < P_phase[r[0]][2, 2]


def test_always_gating_is_dramatically_inconsistent_and_matched_noise_makes_phase_conditional_best(zaru):
    # Monte Carlo (40 trials; the gaps are an order of magnitude, far outside trial noise).
    s = _setup(zaru, accel_std=0.5, ang_accel_std=1.5)
    init_std = np.array([0.05, 0.05, 0.1, 0.05, 0.05])
    nees_by = zaru.run_monte_carlo_consistency(
        s["x_true"], s["dt"], s["accel_std"], s["ang_accel_std"], *s["R"], s["is_anchor"], init_std,
        40, np.random.default_rng(1))
    mean = {k: v.mean() for k, v in nees_by.items()}
    assert mean["always"] > 20 * mean["phase_conditional"]
    assert mean["phase_conditional"] < mean["phase_conditional_zupt_only"] < mean["never"]


def test_ekf_tracks_position_in_every_mode(zaru):
    s = _setup(zaru)
    for mode in zaru.GATING_MODES:
        x_est, _ = _run(zaru, s, mode)
        pos_err, *_ = zaru.state_errors(s["x_true"], x_est)
        assert np.sqrt(np.mean(pos_err ** 2)) < 0.15, mode
