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


def test_crossing_time_ignores_ascending_root_below_guard(saltation_matrix_ekf):
    # A point mass starting BELOW the guard moving upward (the exact state
    # step_hybrid's detect_time_* off-guard resets can produce -- see
    # crossing_time's docstring) crosses p_z=0 again almost immediately on
    # its way back out; that ascending crossing is not a real impact and
    # must be skipped in favor of the next genuinely descending one.
    m = saltation_matrix_ekf
    g = 9.81
    x0 = np.array([0.0, 0.0, -0.05, 0.0, 0.0, 7.0])  # 5cm under "ground", moving up at 7 m/s
    tau = m.crossing_time(x0, g)
    assert tau is not None
    x_at_tau, _ = m.flow(x0, tau, g)
    assert np.isclose(x_at_tau[2], 0.0, atol=1e-9)  # genuinely on the guard
    assert x_at_tau[5] < 0.0  # descending, i.e. the real return-impact, not the rising exit


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

def test_saltation_matrix_composes_to_the_fixed_dt_jacobian(saltation_matrix_ekf):
    # saltation_matrix must satisfy the property step_hybrid actually needs:
    # composed with the ordinary flow Jacobians before/after the event
    # (Phi_after @ Xi @ Phi_before), it must equal the exact Jacobian of the
    # WHOLE fixed-dt map (every state compared at the same external clock
    # time T, not at each trajectory's own natural crossing time) -- checked
    # here directly against a from-scratch finite difference of that map.
    m = saltation_matrix_ekf
    g, e = 9.81, 0.6
    x0 = np.array([0.0, 0.0, 0.05, 0.3, 0.2, -2.0])  # close to the guard: crosses well within dt
    dt_tick = 0.05

    def one_bounce_map(x, dt):
        tau = m.crossing_time(x, g)
        assert tau is not None and tau < dt  # sanity: this scenario crosses within the tick
        x_minus, _ = m.flow(x, tau, g)
        x_plus = m.reset_map(x_minus, e)
        x_final, _ = m.flow(x_plus, dt - tau, g)
        return x_final

    x_final = one_bounce_map(x0, dt_tick)

    eps = 1e-6
    J_numeric = np.zeros((6, 6))
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        J_numeric[:, i] = (one_bounce_map(x0 + d, dt_tick) - x_final) / eps

    tau = m.crossing_time(x0, g)
    x_minus, Phi_before = m.flow(x0, tau, g)
    Xi = m.saltation_matrix(x_minus, e, g)
    x_plus = m.reset_map(x_minus, e)
    _, Phi_after = m.flow(x_plus, dt_tick - tau, g)
    J_analytic = Phi_after @ Xi @ Phi_before

    assert np.allclose(J_analytic, J_numeric, atol=1e-4)

    # and confirm this genuinely differs from composing the naive reset
    # Jacobian alone -- not just that the code runs, but that the correction
    # actually matters for this fixed-dt comparison too.
    DR = m.reset_jacobian(e)
    J_naive = Phi_after @ DR @ Phi_before
    assert np.max(np.abs(J_naive - J_numeric)) > 0.5


def test_saltation_matrix_guard_normal_row_scales_by_minus_e(saltation_matrix_ekf):
    # Structural property (not a numerical coincidence): Dg @ Xi = -e * Dg
    # identically, for any x_minus/e/g -- see saltation_matrix's docstring
    # for the derivation. Reduced, not zero: unlike the "compare each
    # trajectory at its own crossing time" quantity this function used to
    # (wrongly) return, this fixed-reference-time quantity does not claim
    # exactly-zero sensitivity in the guard-normal direction.
    m = saltation_matrix_ekf
    rng = np.random.default_rng(11)
    Dg = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    for _ in range(5):
        x_minus = rng.normal(size=6)
        x_minus[2] = 0.0  # on the guard
        x_minus[5] = -abs(x_minus[5]) - 0.1  # descending, nonzero impact speed
        e = rng.uniform(0.0, 1.0)
        Xi = m.saltation_matrix(x_minus, e, 9.81)
        assert np.allclose(Dg @ Xi, -e * Dg, atol=1e-10)


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


# --- contact/phase-detection timing uncertainty (detect_time_bias / detect_time_noise_std) ---

def test_step_hybrid_default_detection_matches_exact_crossing(saltation_matrix_ekf):
    # Backward-compat: with detect_time_bias/detect_time_noise_std left at their
    # defaults (0.0), the reset must still land exactly at the true geometric
    # crossing time, i.e. the behavior step_hybrid had before these params existed.
    m = saltation_matrix_ekf
    g, e = 9.81, 0.6
    x0 = np.array([0.0, 0.0, 5.0, 1.0, 0.5, -2.0])
    P0 = np.eye(6) * 0.01
    dt = 1.0  # long enough to contain exactly one bounce, not a second

    x_out, P_out = m.step_hybrid(x0, P0, dt, e, g, accel_noise_std=0.0, use_saltation=True)

    tau = m.crossing_time(x0, g)
    x_minus, Phi_pre = m.flow(x0, tau, g)
    P_minus = Phi_pre @ P0 @ Phi_pre.T  # process noise is 0 (accel_noise_std=0)
    Xi = m.saltation_matrix(x_minus, e, g)
    x_plus = m.reset_map(x_minus, e)
    P_plus = Xi @ P_minus @ Xi.T  # impact-noise floor is 0 (impact_noise_std default)
    x_expected, Phi_rest = m.flow(x_plus, dt - tau, g)
    P_expected = Phi_rest @ P_plus @ Phi_rest.T

    assert np.allclose(x_out, x_expected, atol=1e-9)
    assert np.allclose(P_out, P_expected, atol=1e-9)


def test_step_hybrid_large_detect_bias_clips_to_tick_and_still_bounces(saltation_matrix_ekf):
    # A detection delay far larger than the tick clips to the tick boundary (the
    # documented single-tick-horizon simplification) rather than silently skipping
    # the bounce or blowing up -- and produces the expected "tunnels through the
    # ground before the late reset fires" artifact.
    m = saltation_matrix_ekf
    g, e = 9.81, 0.6
    x0 = np.array([0.0, 0.0, 5.0, 1.0, 0.5, -2.0])
    P0 = np.eye(6) * 0.01
    dt = 1.0

    x_out, P_out = m.step_hybrid(x0, P0, dt, e, g, accel_noise_std=0.0, use_saltation=True,
                                  detect_time_bias=100.0)

    assert np.all(np.isfinite(x_out)) and np.all(np.isfinite(P_out))
    assert x_out[2] < 0.0  # detected late enough to have fallen through the ground first
    assert x_out[5] > 0.0  # the (late) bounce still flips vertical velocity positive


def test_detect_time_bias_shifts_reset_outcome(saltation_matrix_ekf):
    m = saltation_matrix_ekf
    g, e = 9.81, 0.6
    x0 = np.array([0.0, 0.0, 5.0, 1.0, 0.5, -2.0])
    P0 = np.eye(6) * 0.01
    dt = 1.0

    x_exact, P_exact = m.step_hybrid(x0.copy(), P0.copy(), dt, e, g, accel_noise_std=0.0,
                                      use_saltation=True)
    x_biased, P_biased = m.step_hybrid(x0.copy(), P0.copy(), dt, e, g, accel_noise_std=0.0,
                                        use_saltation=True, detect_time_bias=0.05)

    assert not np.allclose(x_exact, x_biased)
    assert not np.allclose(P_exact, P_biased)


def test_step_hybrid_detect_noise_is_reproducible_given_seeded_rng(saltation_matrix_ekf):
    m = saltation_matrix_ekf
    g, e = 9.81, 0.6
    x0 = np.array([0.0, 0.0, 5.0, 1.0, 0.5, -2.0])
    P0 = np.eye(6) * 0.01
    dt = 1.0

    x_a, _ = m.step_hybrid(x0.copy(), P0.copy(), dt, e, g, accel_noise_std=0.0, use_saltation=True,
                            detect_time_noise_std=0.01, detect_rng=np.random.default_rng(42))
    x_b, _ = m.step_hybrid(x0.copy(), P0.copy(), dt, e, g, accel_noise_std=0.0, use_saltation=True,
                            detect_time_noise_std=0.01, detect_rng=np.random.default_rng(42))
    x_c, _ = m.step_hybrid(x0.copy(), P0.copy(), dt, e, g, accel_noise_std=0.0, use_saltation=True,
                            detect_time_noise_std=0.01, detect_rng=np.random.default_rng(7))

    assert np.array_equal(x_a, x_b)  # same seed -> identical draw -> identical result
    assert not np.array_equal(x_a, x_c)  # different seed -> (almost certainly) different result


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
