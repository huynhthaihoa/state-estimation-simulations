import numpy as np
import pytest


@pytest.fixture
def friction_anisotropic_ekf(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "friction_anisotropic_ekf")


# --- rotation_2d / exact_arc_step on hand-computable cases ---

def test_rotation_2d_known_angles(friction_anisotropic_ekf):
    m = friction_anisotropic_ekf
    assert np.allclose(m.rotation_2d(0.0), np.eye(2))
    assert np.allclose(m.rotation_2d(np.pi / 2), np.array([[0.0, -1.0], [1.0, 0.0]]), atol=1e-9)


def test_exact_arc_step_quarter_circle(friction_anisotropic_ekf):
    m = friction_anisotropic_ekf
    # v=1, omega=pi/2, dt=1 -> a quarter-circle of radius r=v/omega=2/pi, turning left
    # from the origin facing +x: center at (0, r), ends up at (r, r), heading pi/2.
    x0 = np.array([0.0, 0.0, 0.0])
    v, omega, dt = 1.0, np.pi / 2, 1.0
    r = v / omega

    x_new, F = m.exact_arc_step(x0, v, omega, dt)
    assert np.allclose(x_new, [r, r, np.pi / 2], atol=1e-9)
    assert F.shape == (3, 3)


def test_exact_arc_step_straight_line_when_omega_zero(friction_anisotropic_ekf):
    m = friction_anisotropic_ekf
    x0 = np.array([1.0, 2.0, np.pi / 4])
    v, dt = 2.0, 0.5
    x_new, F = m.exact_arc_step(x0, v, 0.0, dt)
    expected = x0 + np.array([v * dt * np.cos(x0[2]), v * dt * np.sin(x0[2]), 0.0])
    assert np.allclose(x_new, expected, atol=1e-9)
    assert x_new[2] == x0[2]  # heading unchanged with no turn rate


def test_exact_arc_step_agrees_near_omega_zero_threshold(friction_anisotropic_ekf):
    # The straight-line branch (|omega| < 1e-9) and the general arc formula must agree
    # to machine precision right at the threshold, so there's no discontinuity in F/x_new
    # for a trajectory that happens to pass through omega ~= 0.
    m = friction_anisotropic_ekf
    x0 = np.array([0.0, 0.0, 0.3])
    v, dt = 1.0, 0.2
    omega_tiny = 2e-9  # just above the 1e-9 threshold, so it takes the general-formula branch
    x_new, _ = m.exact_arc_step(x0, v, omega_tiny, dt)
    x_straight, _ = m.exact_arc_step(x0, v, 0.0, dt)
    assert np.allclose(x_new, x_straight, atol=1e-6)


# --- isotropic_Q_pos / anisotropic_Q_pos: fairness and orientation ---

def test_isotropic_and_anisotropic_q_pos_share_the_same_total_budget(friction_anisotropic_ekf):
    m = friction_anisotropic_ekf
    sigma_grip, sigma_slip, dt = 0.02, 0.1, 0.05
    Q_iso = m.isotropic_Q_pos(sigma_grip, sigma_slip, dt)
    Q_aniso = m.anisotropic_Q_pos(0.7, sigma_grip, sigma_slip, dt)  # any heading -- rotation preserves trace
    assert np.isclose(np.trace(Q_iso), np.trace(Q_aniso))
    assert np.allclose(Q_iso, Q_iso[0, 0] * np.eye(2))  # genuinely direction-blind


def test_anisotropic_q_pos_aligns_with_world_axes_at_zero_heading(friction_anisotropic_ekf):
    m = friction_anisotropic_ekf
    sigma_grip, sigma_slip, dt = 0.02, 0.1, 0.05
    Q = m.anisotropic_Q_pos(0.0, sigma_grip, sigma_slip, dt)
    assert np.allclose(Q, np.diag([sigma_grip ** 2, sigma_slip ** 2]) * dt * dt)


def test_predict_rejects_unknown_q_policy(friction_anisotropic_ekf):
    m = friction_anisotropic_ekf
    x, P = np.zeros(3), np.eye(3)
    with pytest.raises(ValueError):
        m.predict(x, P, 0.1, 0.1, 0.05, "bogus", 0.02, 0.1, 0.01)


# --- the actual finding, run for real ---

def test_fixed_anisotropic_degrades_with_rotation_but_heading_aware_does_not(friction_anisotropic_ekf):
    m = friction_anisotropic_ekf
    rng = np.random.default_rng(0)
    duration, dt = 20.0, 0.05
    v_cmd, omega_cmd = 0.2, 2.0 * np.pi / duration  # exactly one full loop
    sigma_grip, sigma_slip, sigma_theta = 0.02, 0.1, 0.01
    pos_noise_std = 0.02

    x_true, z = m.generate_ground_truth_and_data(
        duration, dt, v_cmd, omega_cmd, sigma_grip, sigma_slip, pos_noise_std, rng)
    theta_ref = x_true[0, 2]
    R_pos = pos_noise_std ** 2 * np.eye(2)

    nees_iso, nees_fixed, nees_aware = m.run_monte_carlo_consistency(
        x_true, dt, v_cmd, omega_cmd, sigma_grip, sigma_slip, sigma_theta, R_pos, theta_ref,
        init_pos_noise_std=0.05, init_theta_noise_std=0.05, n_trials=200, rng=rng)

    # Angular distance under wraparound saturates at pi (opposite heading), not 2*pi, and a
    # covariance ellipse R(theta) @ diag(a,b) @ R(theta).T has period pi in theta (it's the
    # same ellipse at a 180-degree difference) -- so the worst *orientation* mismatch for
    # fixed_anisotropic's Q is expected near a 90-degree heading difference, not 180. Verified
    # robust across seeds 0-3 (this exact bucket comparison holds in all four); what happens
    # beyond the 90-degree peak, out toward a full 180-degree difference, was NOT robust across
    # seeds in manual testing (recovers in some, keeps climbing in others -- likely accumulated
    # trajectory drift competing with the pure orientation-mismatch effect at longer horizons) and
    # is intentionally not asserted here.
    rotation_away = np.abs(((x_true[:, 2] - theta_ref + np.pi) % (2 * np.pi)) - np.pi)
    near_reference = rotation_away < (np.pi / 4)
    near_90_mismatch = (rotation_away >= np.pi / 2) & (rotation_away < 3 * np.pi / 4)

    assert np.mean(nees_fixed[near_90_mismatch]) > np.mean(nees_fixed[near_reference])
    # heading_aware doesn't degrade the same way -- it tracks its own current heading estimate.
    assert np.mean(nees_aware[near_90_mismatch]) <= np.mean(nees_fixed[near_90_mismatch])
    # at its worst, fixed_anisotropic (pointing the wrong way) should be at least as
    # inconsistent as isotropic (which never claimed a direction to begin with).
    assert np.mean(nees_fixed[near_90_mismatch]) >= np.mean(nees_iso[near_90_mismatch])


def test_nees_zero_for_exact_match(friction_anisotropic_ekf):
    m = friction_anisotropic_ekf
    x = np.array([1.0, 2.0, 0.5])
    P = np.eye(3) * 0.1
    assert m.nees(x, x.copy(), P) == 0.0
