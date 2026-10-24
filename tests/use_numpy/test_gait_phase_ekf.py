import numpy as np
import pytest

DT = 0.02
TICKS_PER_CYCLE = 60
SWING_TICKS = 30  # tripod, default cycle period: half of 60 ticks
SWING_TIME = SWING_TICKS * DT


_MODULE = {}


@pytest.fixture
def gait(import_module, use_numpy_dir):
    # imported once per file: each isolated import re-loads matplotlib (~1 s)
    if "m" not in _MODULE:
        _MODULE["m"] = import_module(use_numpy_dir, "gait_phase_ekf")
    return _MODULE["m"]


def _args(m, **overrides):
    """Default command-line arguments, with the given fields overridden."""
    args = m.build_parser().parse_args([])
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _cmd_steps(m, n_ticks=401):
    """Commanded tripod stance flags and their steps."""
    stance_cmd = m.stance_mask(n_ticks, TICKS_PER_CYCLE, m.GAIT)
    return stance_cmd, m.commanded_steps(stance_cmd)


def _td(m, steps, lag_max=0.0, terrain_std=0.0, blocked=None, seed=0):
    return m.true_touchdowns(steps, DT, SWING_TIME, 8.0, lag_max, terrain_std, blocked, np.random.default_rng(seed))


def _synthetic(n, leg=0, spacing=60):
    """n steps of one leg with commanded touchdowns every `spacing` ticks and a liftoff 30 ticks later."""
    return [(leg, k * spacing + 10, k * spacing + 40, k * spacing + 70) for k in range(n)]


# --- commanded schedule ---

def test_commanded_steps_sorted_and_consistent_with_stance_mask(gait):
    stance_cmd, steps = _cmd_steps(gait)
    assert [s[2] for s in steps] == sorted(s[2] for s in steps)
    for i, up, d, nxt in steps:
        assert not stance_cmd[d - 1, i] and stance_cmd[d, i]  # touchdown = first stance tick
        assert up < d < nxt
        assert up < 0 or (stance_cmd[up - 1, i] and not stance_cmd[up, i])  # liftoff = first swing tick
        assert nxt == len(stance_cmd) or (stance_cmd[nxt - 1, i] and not stance_cmd[nxt, i])
        assert stance_cmd[d:nxt, i].all()


# --- true touchdowns ---

def test_true_touchdowns_equal_commanded_without_perturbation(gait):
    _, steps = _cmd_steps(gait)
    td, tau, lag, is_blocked = _td(gait, steps)
    assert np.array_equal(td, [d for (_, _, d, _) in steps])
    assert np.all(tau == 0) and np.all(lag == 0) and not is_blocked.any()


def test_blocked_delay_only_hits_that_leg_inside_the_window(gait):
    _, steps = _cmd_steps(gait)
    td, _, _, is_blocked = _td(gait, steps, blocked=(1, 3.0, 6.0, 0.1))
    for j, (i, up, d, nxt) in enumerate(steps):
        inside = i == 1 and 3.0 <= d * DT < 6.0
        assert is_blocked[j] == inside
        assert td[j] - d == (5 if inside else 0)  # 0.1 s = 5 ticks
    assert is_blocked.sum() > 0


def test_touchdown_clipped_between_min_swing_and_next_liftoff(gait):
    _, steps = _cmd_steps(gait)
    td_hi, _, _, _ = _td(gait, steps, blocked=(1, 0.0, 8.0, 2.0))
    td_lo, _, _, _ = _td(gait, steps, blocked=(1, 0.0, 8.0, -2.0))
    td_rand, _, _, _ = _td(gait, steps, terrain_std=0.5)
    for j, (i, up, d, nxt) in enumerate(steps):
        if i == 1:
            assert td_hi[j] == nxt - 1 and td_lo[j] == up + gait.MIN_SWING_TICKS
        assert up + gait.MIN_SWING_TICKS <= td_rand[j] <= nxt - 1


def test_true_touchdowns_draw_one_normal_per_step_whatever_terrain_std(gait):
    _, steps = _cmd_steps(gait)
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    gait.true_touchdowns(steps, DT, SWING_TIME, 8.0, 0.0, 0.0, None, rng_a)
    gait.true_touchdowns(steps, DT, SWING_TIME, 8.0, 0.0, 0.05, None, rng_b)
    assert rng_a.normal() == rng_b.normal()


def test_systematic_lag_grows_linearly_to_lag_max(gait):
    _, steps = _cmd_steps(gait)
    _, tau, lag, _ = _td(gait, steps, lag_max=0.2)
    d = np.array([s[2] for s in steps])
    assert np.allclose(lag, 0.2 * SWING_TIME * d * DT / 8.0) and np.array_equal(tau, lag)


# --- detector ---

def _triggers(m, steps, td, delay=0.0, delay_std=0.0, false_rate=0.0, seed=0):
    return m.detector_triggers(steps, td, DT, SWING_TICKS, delay, delay_std, false_rate, np.random.default_rng(seed))


def test_detector_without_delay_or_false_alarms_triggers_at_touchdown(gait):
    _, steps = _cmd_steps(gait)
    td, _, _, _ = _td(gait, steps, terrain_std=0.03)
    trig, false_ticks = _triggers(gait, steps, td)
    assert trig == [[t] for t in td]
    assert all(f is None for f in false_ticks)


def test_false_trigger_in_first_half_of_swing_down_and_before_true_one(gait):
    _, steps = _cmd_steps(gait)
    td = np.array([d for (_, _, d, _) in steps])
    trig, false_ticks = _triggers(gait, steps, td, false_rate=1.0)
    for (i, up, d, nxt), t, f in zip(steps, trig, false_ticks):
        lo = d - SWING_TICKS // 3
        assert lo <= f < lo + SWING_TICKS // 6
        assert t == [f, d]


def test_detection_at_or_after_next_liftoff_is_dropped(gait):
    _, steps = _cmd_steps(gait)
    td = np.array([d for (_, _, d, _) in steps])
    full = [j for j, (_, _, d, nxt) in enumerate(steps) if nxt - d == 30]
    kept, _ = _triggers(gait, steps, td, delay=29 * DT)
    dropped, _ = _triggers(gait, steps, td, delay=30 * DT)
    assert all(kept[j] == [steps[j][2] + 29] for j in full)
    assert all(dropped[j] == [] for j in full)


# --- stance flags from start ticks ---

def test_stance_from_commanded_touchdowns_reproduces_commanded_flags(gait):
    stance_cmd, steps = _cmd_steps(gait)
    out = gait.stance_from_starts(stance_cmd, steps, [d for (_, _, d, _) in steps])
    assert np.array_equal(out, stance_cmd)


def test_stance_from_starts_none_removes_that_steps_stance_only(gait):
    stance_cmd, steps = _cmd_steps(gait)
    starts = [d for (_, _, d, _) in steps]
    starts[4] = None
    i, up, d, nxt = steps[4]
    out = gait.stance_from_starts(stance_cmd, steps, starts)
    assert not out[d:nxt, i].any()
    assert np.array_equal(np.delete(out, i, axis=1), np.delete(stance_cmd, i, axis=1))
    assert np.array_equal(out[:d, i], stance_cmd[:d, i]) and np.array_equal(out[nxt:, i], stance_cmd[nxt:, i])


def test_late_start_shortens_stance_early_start_lengthens_it(gait):
    stance_cmd, steps = _cmd_steps(gait)
    i, up, d, nxt = steps[4]
    starts = [s[2] for s in steps]
    starts[4] = d + 3
    late = gait.stance_from_starts(stance_cmd, steps, starts)
    starts[4] = d - 3
    early = gait.stance_from_starts(stance_cmd, steps, starts)
    assert late[:, i].sum() == stance_cmd[:, i].sum() - 3
    assert early[:, i].sum() == stance_cmd[:, i].sum() + 3


# --- CPG touchdown offsets ---

def _cpg(m, steps, triggers, delay=0.04, gain=0.3, window=0.1, mistime=0.1):
    return m.starts_cpg(steps, triggers, DT, delay, gain, window, mistime)


def test_cpg_rejects_trigger_earlier_than_offset_minus_window(gait):
    steps = _synthetic(2)
    d = steps[0][2]
    # window 0.1 s around mu = 0.04 s: a trigger before -0.06 s (3 ticks early) is rejected
    starts, flags, _ = _cpg(gait, steps, [[d - 4, d + 2], steps[1][2:3]])
    assert starts[0] == d + 2 and not flags[0]
    starts, _, _ = _cpg(gait, steps, [[d - 4], [steps[1][2]]])
    assert starts[0] is None
    starts, _, _ = _cpg(gait, steps, [[d - 3], [steps[1][2]]])
    assert starts[0] == d - 3


def test_cpg_late_trigger_is_flagged_and_not_learned(gait):
    steps = _synthetic(3)
    d = [s[2] for s in steps]
    # 10 ticks = 0.2 s late: error 0.16 s > mistime 0.1 s
    starts, flags, est = _cpg(gait, steps, [[d[0] + 10], [d[1] + 2], [d[2] + 2]])
    assert starts[0] == d[0] + 10 and flags == [True, False, False]
    assert est[1] == pytest.approx(0.0)  # mu unchanged: lag estimate still 0
    # the same trigger within the mistime threshold is learned from: mu moves by gain * error
    _, flags, est = _cpg(gait, steps, [[d[0] + 4], [d[1] + 2], [d[2] + 2]])
    assert not flags[0] and est[1] == pytest.approx(0.3 * (0.08 - 0.04))


def test_cpg_groups_learn_independently(gait):
    steps = [(0, 10, 40, 70), (1, 10, 40, 70), (0, 70, 100, 130), (1, 70, 100, 130)]
    assert gait.GROUP_OF[0] != gait.GROUP_OF[1]
    _, _, est = _cpg(gait, steps, [[44], [42], [104], [102]])
    assert est[1] == 0.0 and est[2] > 0.0 and est[3] == 0.0


# --- phase EKF ---

def _ekf(m, steps, triggers, delay=0.04, R=1e-4, q=1e-5, gate=3.0):
    P0 = np.diag([0.05 ** 2, 0.01 ** 2])
    return m.starts_phase_ekf(steps, triggers, DT, delay, R, q, P0, gate)


def test_phase_ekf_rejects_early_trigger_beyond_the_gate(gait):
    steps = _synthetic(2)
    d = [s[2] for s in steps]
    # first step: sqrt(P00 + R) ~ 51 ms, so the gate is ~153 ms around the predicted 40 ms offset
    starts, flags, _, _ = _ekf(gait, steps, [[d[0] - 10, d[0] + 2], [d[1] + 2]])
    assert starts[0] == d[0] + 2 and not flags[0]
    starts, _, _, _ = _ekf(gait, steps, [[d[0] - 10], [d[1] + 2]])
    assert starts[0] is None


def test_phase_ekf_late_trigger_is_flagged_and_not_used(gait):
    steps = _synthetic(3)
    d = [s[2] for s in steps]
    starts, flags, est, std = _ekf(gait, steps, [[d[0] + 12], [d[1] + 2], [d[2] + 2]])
    assert starts[0] == d[0] + 12 and flags == [True, False, False]
    assert est[1] == 0.0 and std[1] > std[0]  # state untouched and uncertainty grew with the prediction
    _, flags, est, std = _ekf(gait, steps, [[d[0] + 4], [d[1] + 2], [d[2] + 2]])
    assert not flags[0] and est[1] > 0.0 and std[1] < std[0]  # a good trigger moves the estimate and shrinks std


def test_phase_ekf_lag_estimate_converges_to_constant_true_lag(gait):
    n, lag, delay = 25, 0.1, 0.04
    steps = _synthetic(n)
    triggers = [[s[2] + int(round((lag + delay) / DT))] for s in steps]
    starts, flags, est, std = _ekf(gait, steps, triggers, delay=delay)
    assert not any(flags) and all(s is not None for s in starts)
    assert est[-1] == pytest.approx(lag, abs=0.01)
    assert abs(est[-1] - lag) < abs(est[0] - lag) and std[-1] < std[0]


# --- Monte Carlo ---

def test_trials_with_no_perturbation_make_every_approach_match_the_oracle(gait):
    args = _args(gait, duration=6.0, lag_max=0.0, terrain_std=0.0, blocked_delay=0.0, det_delay=0.0,
                 det_delay_std=0.0, false_rate=0.0)
    mc = gait.run_monte_carlo(args, 10, np.random.default_rng(0))
    for a in gait.APPROACHES:
        assert mc[a]["swing"] == 0 and mc[a]["missed"] == 0
        assert np.array_equal(mc[a]["nees"], mc["oracle"]["nees"])
    assert 4.0 < mc["oracle"]["nees"][1:].mean() < 9.0


def test_perturbed_trials_mislead_schedule_and_detector_but_not_cpg(gait):
    args = _args(gait, duration=8.0, false_rate=0.3, lag_max=0.3)
    mc = gait.run_monte_carlo(args, 8, np.random.default_rng(0))
    assert mc["oracle"]["swing"] == 0 and mc["oracle"]["missed"] == 0
    assert mc["schedule"]["swing"] > 0 and mc["detector"]["swing"] > 0
    assert mc["cpg"]["swing"] < 0.2 * mc["schedule"]["swing"]
    assert mc["schedule"]["nees"][1:].mean() > 2 * mc["cpg"]["nees"][1:].mean()
