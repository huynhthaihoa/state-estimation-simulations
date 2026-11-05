import numpy as np

from state_estimation_simulations.use_numpy import gait_phase_ekf as g
from state_estimation_simulations.use_numpy import gait_phase_torque_ekf as m


def default_args(**kw):
    args = m.build_parser().parse_args([])
    args.duration = 6.0
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def test_force_per_torque_from_daisy_stance():
    # vertical force std per 1 Nm of torque noise on each joint, Daisy nominal stance
    assert abs(m.force_per_torque() - 4.27) < 0.01


def test_force_signal_parts():
    """Noise-free and error-free: 0 before touchdown, linear ramp, then the load share."""
    dt, sw = 0.02, 30
    steps = [(0, 0, 50, 80)]
    td = [52]
    windows = m.step_windows(steps, td, sw)
    draws = [(1.0, np.ones(int(round((80 - windows[0][0]) * dt / m.SAMPLE_DT))))]
    (t, f), = m.force_signals(windows, td, dt, sw, draws, 100.0, 0.05, 0.0, 0.0)
    t_td = 52 * dt
    assert np.all(f[t < t_td] == 0.0)
    assert np.allclose(f[t >= t_td + 0.05], 100.0)
    mid = np.argmin(abs(t - (t_td + 0.025)))
    assert abs(f[mid] - 50.0) < 1e-6


def test_model_error_pulse_window():
    """The pulse lives in the first half of the swing-down only, peak = amplitude."""
    dt, sw = 0.02, 30
    steps, td = [(0, 0, 50, 80)], [60]
    windows = m.step_windows(steps, td, sw)
    n = int(round((80 - windows[0][0]) * dt / m.SAMPLE_DT))
    (t, f), = m.force_signals(windows, td, dt, sw, [(2.0, np.zeros(n))], 100.0, 0.05, 10.0, 0.0)
    start = (50 - sw // 3) * dt
    inside = (t >= start) & (t <= start + (sw // 6) * dt)
    assert abs(f[inside].max() - 20.0) < 1e-3
    assert np.all(f[~inside & (t < 60 * dt)] == 0.0)


def test_threshold_triggers_tick_hysteresis_and_cutoff():
    dt = 0.02
    t = np.arange(0.0, 1.0, m.SAMPLE_DT)
    f = np.zeros_like(t)
    f[(t > 0.105) & (t < 0.2)] = 50.0     # crossing at 0.106 s -> tick 6
    f[(t >= 0.2) & (t < 0.3)] = 20.0      # stays above half the threshold: no re-arm
    f[t >= 0.3] = 50.0
    steps = [(0, 0, 0, 40)]
    assert m.threshold_triggers([(t, f)], steps, dt, 30.0) == [[6]]
    f[(t >= 0.4) & (t < 0.5)] = 0.0       # re-arm, cross again at 0.5 s -> tick 25
    assert m.threshold_triggers([(t, f)], steps, dt, 30.0) == [[6, 25]]
    assert m.threshold_triggers([(t, f)], [(0, 0, 0, 25)], dt, 30.0) == [[6]]


def test_position_rows_unchanged_and_torque_draws_last():
    args = default_args()
    model = m.TorqueModel(args)
    *_, f0, _ = g.simulate_trial(args, np.random.default_rng(4))
    *_, f1, _ = m.torque_trial(args, np.random.default_rng(4), model)
    for a in g.ROWS:
        assert np.array_equal(f0[a], f1[a])


def test_noise_free_torque_delay_is_the_known_delay():
    args = default_args(model_err_std=0.0, torque_noise_std=0.0, blocked_delay=0.0)
    model = m.TorqueModel(args)
    *_, info = m.torque_trial(args, np.random.default_rng(1), model)
    delay = int(round(model.known_delay(args.dt) / args.dt))
    for trig, k, (_, _, _, nxt) in zip(info["torque"]["triggers"], info["td"], info["steps"]):
        assert trig == ([k + delay] if k + delay < nxt else [])


def test_fit_logistic_recovers_a_known_logistic():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 2.0, 20000)
    y = (rng.random(x.size) < 1.0 / (1.0 + np.exp(-(4.0 * x - 2.0)))).astype(float)
    # pack as one "step": times so that label = sample at or after td reproduces y
    order = np.argsort(y, kind="stable")
    t = np.arange(x.size) * 1.0
    k_td = int((y == 0).sum())
    b0, b1, base = m.fit_logistic([(t, x[order])], [k_td], 1.0, 1.0, l2=0.0)
    assert abs(b1 - 4.0) < 0.3 and abs(b0 + 2.0) < 0.2
    assert abs(base - y.mean()) < 1e-12
