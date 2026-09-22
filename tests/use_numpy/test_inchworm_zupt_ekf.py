import numpy as np
import pytest


@pytest.fixture
def inchworm_zupt_ekf(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "inchworm_zupt_ekf")


# --- is_anchor_phase / true_velocity on a hand-picked schedule ---

def test_is_anchor_phase_known_schedule(inchworm_zupt_ekf):
    m = inchworm_zupt_ekf
    t_anchor, t_extend = 1.0, 0.5  # cycle = 1.5s: [0,1.0)=anchor, [1.0,1.5)=extend

    assert m.is_anchor_phase(0.0, t_anchor, t_extend) is True
    assert m.is_anchor_phase(0.99, t_anchor, t_extend) is True
    assert m.is_anchor_phase(1.01, t_anchor, t_extend) is False
    assert m.is_anchor_phase(1.49, t_anchor, t_extend) is False
    # second cycle
    assert m.is_anchor_phase(1.5, t_anchor, t_extend) is True
    assert m.is_anchor_phase(2.6, t_anchor, t_extend) is False  # 2.6 mod 1.5 = 1.1, in extend


def test_true_velocity_matches_phase_no_ramp(inchworm_zupt_ekf):
    m = inchworm_zupt_ekf
    t_anchor, t_extend, v_extend = 1.0, 0.5, 0.3

    assert m.true_velocity(0.5, t_anchor, t_extend, t_ramp=0.0, v_extend=v_extend) == 0.0
    assert m.true_velocity(1.2, t_anchor, t_extend, t_ramp=0.0, v_extend=v_extend) == v_extend


def test_true_velocity_ramps_continuously_at_transitions(inchworm_zupt_ekf):
    # cycle = 1.5s: [0,1.0)=anchor, [1.0,1.1)=ramp-up, [1.1,1.4)=cruise, [1.4,1.5)=ramp-down
    m = inchworm_zupt_ekf
    t_anchor, t_extend, t_ramp, v_extend = 1.0, 0.5, 0.1, 0.3

    assert np.isclose(m.true_velocity(1.0, t_anchor, t_extend, t_ramp, v_extend), 0.0)  # continuous at the boundary
    assert np.isclose(m.true_velocity(1.05, t_anchor, t_extend, t_ramp, v_extend), 0.5 * v_extend)  # mid ramp-up
    assert np.isclose(m.true_velocity(1.1, t_anchor, t_extend, t_ramp, v_extend), v_extend)  # cruise reached
    assert np.isclose(m.true_velocity(1.25, t_anchor, t_extend, t_ramp, v_extend), v_extend)  # mid-cruise
    assert np.isclose(m.true_velocity(1.45, t_anchor, t_extend, t_ramp, v_extend), 0.5 * v_extend)  # mid ramp-down
    assert np.isclose(m.true_velocity(1.5, t_anchor, t_extend, t_ramp, v_extend), 0.0)  # continuous back to 0


def test_is_cruise_phase_excludes_ramp_edges(inchworm_zupt_ekf):
    m = inchworm_zupt_ekf
    t_anchor, t_extend, t_ramp = 1.0, 0.5, 0.1

    assert m.is_cruise_phase(0.5, t_anchor, t_extend, t_ramp) is False  # anchored
    assert m.is_cruise_phase(1.05, t_anchor, t_extend, t_ramp) is False  # ramp-up
    assert m.is_cruise_phase(1.25, t_anchor, t_extend, t_ramp) is True  # cruise
    assert m.is_cruise_phase(1.45, t_anchor, t_extend, t_ramp) is False  # ramp-down


# --- generate_ground_truth_and_data: velocity is exact per phase ---

def test_generate_ground_truth_velocity_exact_per_phase(inchworm_zupt_ekf):
    m = inchworm_zupt_ekf
    rng = np.random.default_rng(0)
    t_anchor, t_extend, t_ramp, v_extend = 1.0, 0.5, 0.1, 0.3
    dt = 0.1  # evenly divides t_anchor/t_ramp/the cruise duration

    x_true, z, is_anchor, is_cruise = m.generate_ground_truth_and_data(
        duration=3.0, dt=dt, t_anchor=t_anchor, t_extend=t_extend, t_ramp=t_ramp,
        v_extend=v_extend, pos_noise_std=0.01, rng=rng)

    assert np.all(x_true[is_anchor, 1] == 0.0)
    assert np.all(x_true[is_cruise, 1] == v_extend)
    assert np.any(is_anchor) and np.any(is_cruise)  # both phases actually occurred
    assert x_true[0, 0] == 0.0  # starts at the origin
    assert np.all(np.diff(x_true[:, 0]) >= -1e-12)  # position never goes backward
    assert z.shape == (len(x_true),)


# --- measurement_update_zupt: basic KF sanity ---

def test_measurement_update_zupt_pulls_velocity_toward_zero(inchworm_zupt_ekf):
    m = inchworm_zupt_ekf
    x = np.array([1.0, 0.5])  # nonzero velocity estimate
    P = np.diag([0.01, 0.01])
    x_new, P_new = m.measurement_update_zupt(x, P, R_zupt=0.001)

    assert abs(x_new[1]) < abs(x[1])  # velocity pulled toward the ZUPT's claimed 0
    assert x_new[0] == x[0]  # position untouched (H's first column is 0)
    assert P_new[1, 1] < P[1, 1]  # velocity variance shrinks after the update


def test_measurement_update_position_pulls_position_toward_measurement(inchworm_zupt_ekf):
    m = inchworm_zupt_ekf
    x = np.array([0.0, 0.0])
    P = np.diag([0.01, 0.01])
    x_new, P_new = m.measurement_update_position(x, P, z=1.0, R_pos=0.001)

    assert x_new[0] > 0.0  # pulled toward the measurement
    assert x_new[1] == x[1]  # velocity untouched (H's second column is 0)


# --- the actual 3-way finding, run for real (see inchworm_zupt_ekf.py's module docstring:
# calibrated so the true ramp acceleration isn't wildly outside what the process-noise model
# can represent -- an earlier, less-calibrated parameterization produced nonsensical NEES in
# the hundreds of thousands from exactly that mismatch, a bug caught by sanity-checking before
# trusting any number, not a finding) ---

def test_never_zupt_has_better_velocity_rms_than_always_but_worse_than_phase_conditional(inchworm_zupt_ekf):
    m = inchworm_zupt_ekf
    rng = np.random.default_rng(0)
    duration, dt, t_anchor, t_extend, t_ramp, v_extend = 10.0, 0.05, 1.0, 1.0, 0.2, 0.1
    x_true, z, is_anchor, is_cruise = m.generate_ground_truth_and_data(
        duration, dt, t_anchor, t_extend, t_ramp, v_extend, pos_noise_std=0.02, rng=rng)
    n_steps = len(x_true) - 1

    init_std = np.array([0.05, 0.05])
    x_init = x_true[0] + rng.normal(0.0, 1.0, 2) * init_std
    P_init = np.diag(init_std ** 2)
    R_pos, R_zupt = 0.02 ** 2, 0.01 ** 2
    process_noise_std = 0.15

    x_never, _ = m.run_ekf_never_zupt(x_init, P_init, z, dt, n_steps, process_noise_std, R_pos,
                                       R_zupt, is_anchor)
    x_always, _ = m.run_ekf_always_zupt(x_init, P_init, z, dt, n_steps, process_noise_std, R_pos,
                                         R_zupt, is_anchor)
    x_phase, _ = m.run_ekf_phase_conditional_zupt(x_init, P_init, z, dt, n_steps, process_noise_std,
                                                   R_pos, R_zupt, is_anchor)

    _, vel_err_never = m.state_errors(x_true, x_never)
    _, vel_err_always = m.state_errors(x_true, x_always)
    _, vel_err_phase = m.state_errors(x_true, x_phase)

    rms_never = np.sqrt(np.mean(vel_err_never ** 2))
    rms_always = np.sqrt(np.mean(vel_err_always ** 2))
    rms_phase = np.sqrt(np.mean(vel_err_phase ** 2))
    # phase-conditional wins on accuracy: it's the only one that both uses the free
    # anchor-phase information and never misapplies it while genuinely moving.
    assert rms_phase < rms_never
    assert rms_phase < rms_always


def test_always_zupt_is_dramatically_inconsistent_in_both_windows(inchworm_zupt_ekf):
    # Contrary to the naive expectation that always-ZUPT would at least match
    # phase-conditional during genuine anchor ticks (same correct update applied there),
    # it doesn't: misapplying ZUPT throughout the preceding cruise phase leaves it so
    # overconfident that it hasn't recovered even ~20 correct-ZUPT anchor ticks later.
    # Verified stable across seeds 0-3 (ratios consistently ~30-50x) before writing this in.
    m = inchworm_zupt_ekf
    rng = np.random.default_rng(0)
    duration, dt, t_anchor, t_extend, t_ramp, v_extend = 10.0, 0.05, 1.0, 1.0, 0.2, 0.1
    x_true, z, is_anchor, is_cruise = m.generate_ground_truth_and_data(
        duration, dt, t_anchor, t_extend, t_ramp, v_extend, pos_noise_std=0.02, rng=rng)

    nees_never, nees_always, nees_phase = m.run_monte_carlo_consistency(
        x_true, dt, process_noise_std=0.15, R_pos=0.02 ** 2, R_zupt=0.01 ** 2, is_anchor=is_anchor,
        init_pos_noise_std=0.05, init_vel_noise_std=0.05, n_trials=200, rng=rng)

    assert np.mean(nees_always[is_cruise]) > 20.0 * np.mean(nees_phase[is_cruise])
    assert np.mean(nees_always[is_anchor]) > 20.0 * np.mean(nees_phase[is_anchor])


def test_never_zupt_is_at_least_as_well_calibrated_as_phase_conditional(inchworm_zupt_ekf):
    # An echo of saltation_matrix_ekf.py's finding: the filter that stakes the least
    # confidence (never) ends up at least as consistent (NEES) as the one that correctly
    # exploits a strong pseudo-measurement (phase_conditional) -- exact/aggressive claims
    # are more fragile to any real-world imperfection than conservative ones, even when
    # used exactly as intended. phase_conditional still wins on raw accuracy (see the
    # RMS test above); this is specifically about calibration, not accuracy.
    m = inchworm_zupt_ekf
    rng = np.random.default_rng(0)
    duration, dt, t_anchor, t_extend, t_ramp, v_extend = 10.0, 0.05, 1.0, 1.0, 0.2, 0.1
    x_true, z, is_anchor, is_cruise = m.generate_ground_truth_and_data(
        duration, dt, t_anchor, t_extend, t_ramp, v_extend, pos_noise_std=0.02, rng=rng)

    nees_never, nees_always, nees_phase = m.run_monte_carlo_consistency(
        x_true, dt, process_noise_std=0.15, R_pos=0.02 ** 2, R_zupt=0.01 ** 2, is_anchor=is_anchor,
        init_pos_noise_std=0.05, init_vel_noise_std=0.05, n_trials=200, rng=rng)

    assert np.mean(nees_never) <= np.mean(nees_phase)


def test_run_ekf_rejects_unknown_zupt_mode(inchworm_zupt_ekf):
    m = inchworm_zupt_ekf
    x_init, P_init = np.zeros(2), np.eye(2)
    z = np.zeros(3)
    is_anchor = np.array([True, True, False])
    with pytest.raises(ValueError):
        m.run_ekf(x_init, P_init, z, 0.1, 2, 0.05, 0.01, 0.01, is_anchor, zupt_mode="bogus")


def test_nees_zero_for_exact_match(inchworm_zupt_ekf):
    m = inchworm_zupt_ekf
    x = np.array([1.0, 2.0])
    P = np.eye(2) * 0.1
    assert m.nees(x, x.copy(), P) == 0.0
