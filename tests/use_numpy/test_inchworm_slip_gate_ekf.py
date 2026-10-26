import numpy as np
import pytest

from state_estimation_simulations.use_numpy import inchworm_slip_gate_ekf as _inchworm_slip_gate_ekf
from state_estimation_simulations.use_numpy import inchworm_zupt_ekf as _inchworm_zupt_ekf


@pytest.fixture
def slip():
    return _inchworm_slip_gate_ekf


@pytest.fixture
def base():
    return _inchworm_zupt_ekf


def _setup(m, seed=0, p_slip=0.5, v_slip=0.08, sensor_noise_std=0.0005, duration=20.0):
    rng = np.random.default_rng(seed)
    dt = 0.05
    out = m.generate_ground_truth_and_data(
        duration, dt, 1.0, 1.0, 0.2, 0.1, 0.02, sensor_noise_std, p_slip, v_slip, 0.6, rng)
    x_true, z, d, is_anchor, is_cruise, is_slip, is_clean = out
    init_std = np.array([0.05, 0.05])
    return dict(x_true=x_true, z=z, d=d, is_anchor=is_anchor, is_cruise=is_cruise, is_slip=is_slip,
                is_clean=is_clean, dt=dt, x_init=x_true[0] + rng.normal(0.0, 1.0, 2) * init_std,
                P_init=np.diag(init_std ** 2), n_steps=len(x_true) - 1,
                R=(0.02 ** 2, 0.01 ** 2, (sensor_noise_std / dt) ** 2 if sensor_noise_std else 1e-8),
                d_thr=3.0 * sensor_noise_std)


def _run(m, s, mode, d_thr=None):
    return m.run_ekf(s["x_init"], s["P_init"], s["z"], s["d"], s["dt"], s["n_steps"], 0.5, *s["R"],
                     s["d_thr"] if d_thr is None else d_thr, s["is_anchor"], mode)


# --- slip schedule and truth ---

def test_slip_episodes_deterministic_and_inside_anchor_phases(slip):
    a = slip.draw_slip_episodes(20.0, 1.0, 1.0, 0.6, 0.5, np.random.default_rng(3))
    b = slip.draw_slip_episodes(20.0, 1.0, 1.0, 0.6, 0.5, np.random.default_rng(3))
    assert a == b and len(a) > 0
    for start, end in a:
        cycle = int(start // 2.0)
        assert end - start == pytest.approx(0.6)
        assert cycle * 2.0 <= start and end <= cycle * 2.0 + 1.0 + 1e-12


def test_slip_probability_zero_and_one(slip):
    assert slip.draw_slip_episodes(20.0, 1.0, 1.0, 0.6, 0.0, np.random.default_rng(0)) == []
    assert len(slip.draw_slip_episodes(20.0, 1.0, 1.0, 0.6, 1.0, np.random.default_rng(0))) == 10


def test_rng_stream_does_not_depend_on_p_slip(slip):
    r0, r1 = np.random.default_rng(5), np.random.default_rng(5)
    slip.draw_slip_episodes(20.0, 1.0, 1.0, 0.6, 0.0, r0)
    slip.draw_slip_episodes(20.0, 1.0, 1.0, 0.6, 1.0, r1)
    assert r0.random() == r1.random()


def test_slip_velocity_is_a_backward_trapezoid(slip):
    ep = [(1.0, 1.6)]
    assert slip.slip_velocity(0.9, ep, 0.2, 0.08) == 0.0
    assert slip.slip_velocity(1.0, ep, 0.2, 0.08) == pytest.approx(0.0)
    assert slip.slip_velocity(1.1, ep, 0.2, 0.08) == pytest.approx(-0.04)
    assert slip.slip_velocity(1.3, ep, 0.2, 0.08) == pytest.approx(-0.08)
    assert slip.slip_velocity(1.6, ep, 0.2, 0.08) == 0.0


def test_truth_is_still_on_clean_anchor_and_slides_backward_in_slip(slip):
    s = _setup(slip)
    v = s["x_true"][:, 1]
    assert s["is_slip"].any() and s["is_clean"].any()
    assert np.all(v[s["is_clean"]] == 0.0)
    assert np.all(v[s["is_slip"]] <= 0.0) and v[s["is_slip"]].min() == pytest.approx(-0.08)
    assert not np.any(s["is_slip"] & ~s["is_anchor"])
    assert not np.any(s["is_slip"] & s["is_clean"])
    assert np.all(v[~s["is_anchor"]] >= 0.0)


def test_position_is_trapezoidal_integral_of_velocity(slip):
    s = _setup(slip)
    p, v = s["x_true"][:, 0], s["x_true"][:, 1]
    np.testing.assert_allclose(np.diff(p), 0.5 * (v[:-1] + v[1:]) * s["dt"], atol=1e-12)


def test_generate_rejects_slip_pulse_that_does_not_fit(slip):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        slip.generate_ground_truth_and_data(10.0, 0.05, 1.0, 1.0, 0.2, 0.1, 0.02, 0.0005, 0.5, 0.08, 1.2, rng)
    with pytest.raises(ValueError):
        slip.generate_ground_truth_and_data(10.0, 0.05, 1.0, 1.0, 0.2, 0.1, 0.02, 0.0005, 0.5, 0.08, 0.3, rng)


def test_sensor_reads_dt_times_velocity_plus_noise(slip):
    s = _setup(slip, p_slip=0.0, duration=200.0)
    resid = s["d"] - s["dt"] * s["x_true"][:, 1]
    assert np.std(resid) == pytest.approx(0.0005, rel=0.1)
    assert abs(np.mean(resid)) < 5 * 0.0005 / np.sqrt(len(resid))
    assert np.mean(s["d"][s["is_cruise"]]) == pytest.approx(0.05 * 0.1, rel=0.02)


# --- filter wiring ---

def test_never_and_schedule_match_the_1d_script_exactly(slip, base):
    s = _setup(slip, p_slip=0.0)
    for mode, fn in (("never", base.run_ekf_never_zupt), ("schedule", base.run_ekf_phase_conditional_zupt)):
        x_new, _ = _run(slip, s, mode)
        x_old, _ = fn(s["x_init"], s["P_init"], s["z"], s["dt"], s["n_steps"], 0.5, s["R"][0], s["R"][1],
                      s["is_anchor"])
        np.testing.assert_allclose(x_new, x_old, atol=1e-12)


def test_gate_threshold_extremes_reduce_to_other_policies(slip):
    s = _setup(slip)
    never = _run(slip, s, "never")[0]
    sched = _run(slip, s, "schedule")[0]
    np.testing.assert_allclose(_run(slip, s, "sensor_gate", d_thr=0.0)[0], never, atol=1e-12)
    np.testing.assert_allclose(_run(slip, s, "schedule_and_sensor", d_thr=0.0)[0], never, atol=1e-12)
    np.testing.assert_allclose(_run(slip, s, "schedule_and_sensor", d_thr=np.inf)[0], sched, atol=1e-12)


def test_slip_below_the_gate_speed_is_never_vetoed(slip):
    s = _setup(slip, v_slip=0.02, sensor_noise_std=0.0)
    s["d_thr"] = 0.0015  # 0.03 m/s per 0.05 s tick, above the 0.02 m/s slip
    assert s["is_slip"].any()
    np.testing.assert_allclose(_run(slip, s, "schedule_and_sensor")[0], _run(slip, s, "schedule")[0],
                               atol=1e-12)


def test_unknown_mode_is_rejected(slip):
    s = _setup(slip)
    with pytest.raises(ValueError):
        _run(slip, s, "bogus")


def test_sensor_fused_tracks_velocity_better_than_never(slip):
    s = _setup(slip)
    err = {m: slip.state_errors(s["x_true"], _run(slip, s, m)[0])[1] for m in ("never", "sensor_fused")}
    assert np.sqrt(np.mean(err["sensor_fused"] ** 2)) < 0.5 * np.sqrt(np.mean(err["never"] ** 2))


# --- the finding ---

def test_schedule_is_overconfident_in_slip_but_the_sensor_veto_is_not(slip):
    s = _setup(slip)
    nees = slip.run_monte_carlo_consistency(
        s["x_true"], s["dt"], 0.5, *s["R"], s["d_thr"], s["is_anchor"], 0.05, 0.05, 0.0005, 80,
        np.random.default_rng(1), modes=("schedule", "schedule_and_sensor"))
    sl, clean = s["is_slip"], s["is_clean"]
    assert np.mean(nees["schedule"][sl]) > 10.0
    assert np.mean(nees["schedule_and_sensor"][sl]) < 0.4 * np.mean(nees["schedule"][sl])
    assert np.mean(nees["schedule"][clean]) < 2.2 and np.mean(nees["schedule_and_sensor"][clean]) < 2.2


def _two_tick_run(m, d1, mode, is_anchor=(True, True)):
    """One filter step from a state with velocity 1 m/s; a ZUPT pulls the velocity down, a skipped one does not."""
    x, P = np.array([0.0, 1.0]), np.diag([1.0, 1.0])
    return m.run_ekf(x, P, np.zeros(2), np.array([0.0, d1]), 0.05, 1, 0.5, 0.02 ** 2, 0.01 ** 2, 0.01 ** 2,
                     0.0015, np.array(is_anchor), mode)[0][-1, 1]


def test_gate_is_inclusive_at_the_threshold(slip):
    at = _two_tick_run(slip, 0.0015, "sensor_gate")
    above = _two_tick_run(slip, 0.0015 * (1 + 1e-9), "sensor_gate")
    assert at < 0.5 * above  # a ZUPT ran at |d| == d_thr, none just above it


def test_sensor_gate_ignores_the_schedule(slip):
    never = _two_tick_run(slip, 0.0, "never", is_anchor=(False, False))
    assert _two_tick_run(slip, 0.0, "schedule", is_anchor=(False, False)) == pytest.approx(never)
    assert _two_tick_run(slip, 0.0, "schedule_and_sensor", is_anchor=(False, False)) == pytest.approx(never)
    assert _two_tick_run(slip, 0.0, "sensor_gate", is_anchor=(False, False)) < 0.5 * never


def test_monte_carlo_draws_sensor_noise_and_position_noise_at_the_stated_sizes(slip):
    s = _setup(slip)
    nees = slip.run_monte_carlo_consistency(
        s["x_true"], s["dt"], 0.5, *s["R"], s["d_thr"], s["is_anchor"], 0.05, 0.05, 0.0005, 100,
        np.random.default_rng(2), modes=("never", "sensor_fused"))
    assert 1.1 < np.mean(nees["never"][s["is_clean"]]) < 1.9
    assert 1.2 < np.mean(nees["sensor_fused"][s["is_clean"]]) < 2.2
