import numpy as np
import pytest

from state_estimation_simulations.use_numpy import inchworm_two_pad_fusion_ekf as _inchworm_two_pad_fusion_ekf
from state_estimation_simulations.use_numpy import inchworm_two_pad_ekf as _inchworm_two_pad_ekf

DT = 0.05
T_PHASE = 1.0


@pytest.fixture
def fus():
    return _inchworm_two_pad_fusion_ekf


@pytest.fixture
def two():
    return _inchworm_two_pad_ekf


def _args(m, extra=""):
    return m.build_parser().parse_args(extra.split())


def _sim(m, gains=None, faults=(), episodes=(), noise_std=0.0005, closure_std=0.001, duration=4.0, seed=0):
    """One simulate() call with the script's default geometry; returns its outputs."""
    rng = np.random.default_rng(seed)
    n_steps = int(round(duration / DT))
    n_phases = int(np.ceil(duration / T_PHASE))
    gains = np.ones(n_phases) if gains is None else gains
    return m.simulate(n_steps, DT, T_PHASE, 0.2, 0.1, 0.15, 0.3, list(episodes), 0.08, gains, list(faults),
                      0.01 if noise_std else 0.0, 0.02 if noise_std else 0.0, noise_std, noise_std, closure_std,
                      closure_std, rng)


def _mc_nees(m, extra, variants, n_trials=30, duration=8.0):
    args = _args(m, f"--duration {duration} {extra}")
    rng = np.random.default_rng(args.seed)
    setup = m.build_setup(args, rng)
    stats, det = m.run_monte_carlo(args, setup, n_trials, rng, variants=variants)
    return {v: float(np.mean(stats[v]["nees"])) for v in variants}, det


# --- truth model ---

def test_faults_deterministic_inside_their_phase_and_stream_independent_of_p_fault(fus):
    a = fus.draw_faults(20, T_PHASE, 0.3, 0.3, np.random.default_rng(2))
    b = fus.draw_faults(20, T_PHASE, 0.3, 0.3, np.random.default_rng(2))
    assert len(a) == len(b) > 0
    assert all(f[:4] == g[:4] and (f[4] is g[4] is None or np.array_equal(f[4], g[4])) for f, g in zip(a, b))
    for pad, kind, start, end, v in a:
        assert pad in fus.PADS and kind in (fus.SPIKE, fus.DROPOUT)
        assert (v is None) == (kind == fus.DROPOUT)
        assert int(np.floor(start / T_PHASE)) == int(np.floor((end - 1e-9) / T_PHASE))
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    fus.draw_faults(20, T_PHASE, 0.3, 0.0, r1)
    fus.draw_faults(20, T_PHASE, 0.3, 0.9, r2)
    assert r1.random() == r2.random()


def test_fault_type_forces_the_kind_without_moving_the_episodes(fus):
    both = fus.draw_faults(20, T_PHASE, 0.3, 0.4, np.random.default_rng(1))
    spikes = fus.draw_faults(20, T_PHASE, 0.3, 0.4, np.random.default_rng(1), "spike")
    drops = fus.draw_faults(20, T_PHASE, 0.3, 0.4, np.random.default_rng(1), "dropout")
    assert [(f[0], f[2]) for f in both] == [(f[0], f[2]) for f in spikes] == [(f[0], f[2]) for f in drops]
    assert all(f[1] == fus.SPIKE for f in spikes) and all(f[1] == fus.DROPOUT for f in drops)


def test_random_spike_speed_draws_speed_and_direction(fus):
    fixed = fus.draw_faults(40, T_PHASE, 0.3, 0.5, np.random.default_rng(4), "spike", 0.08)
    assert all(np.allclose(f[4], [-0.08, 0.0]) for f in fixed)
    rand = fus.draw_faults(40, T_PHASE, 0.3, 0.5, np.random.default_rng(4), "spike", 0.08, True)
    speeds = np.array([np.linalg.norm(f[4]) for f in rand])
    angles = np.array([np.arctan2(f[4][1], f[4][0]) for f in rand])
    assert np.all((speeds >= 0.04) & (speeds <= 0.12)) and np.ptp(speeds) > 0.03
    assert np.ptp(angles) > np.pi


def test_stroke_gains_are_one_without_error(fus):
    assert np.all(fus.draw_stroke_gains(8, 0.0, np.random.default_rng(0)) == 1.0)


def test_per_cycle_gains_share_the_extend_gain_and_keep_the_stream(fus):
    per_phase = fus.draw_stroke_gains(9, 0.05, np.random.default_rng(3))
    per_cycle = fus.draw_stroke_gains(9, 0.05, np.random.default_rng(3), per_cycle=True)
    assert np.array_equal(per_cycle[0::2], per_phase[0::2])
    assert np.array_equal(per_cycle[1::2], per_cycle[0::2][:4])


def test_moving_reading_equals_anchored_plus_commanded_change_when_noise_free(fus):
    x, gyro, d_a, d_m, z, back, e_cmd, fa, fm = _sim(fus, noise_std=0.0, closure_std=0.0)
    for k in range(len(gyro)):
        s = 1.0 if back[k] else -1.0
        e = fus.commanded_change(e_cmd[k], z[k][0])
        assert np.allclose(d_m[k], d_a[k] + s * e, atol=1e-12)


def test_stroke_gain_scales_the_executed_stroke(fus):
    gains = np.array([1.2, 1.0, 1.0, 1.0])
    z1 = _sim(fus, noise_std=0.0, closure_std=0.0)[4]
    z2 = _sim(fus, gains=gains, noise_std=0.0, closure_std=0.0)[4]
    stroke = z1[20][0] - z1[0][0]
    assert z2[20][0] - z2[0][0] == pytest.approx(1.2 * stroke)


def test_faults_change_only_the_faulty_readings_and_not_the_truth(fus):
    faults = [("back", fus.SPIKE, 0.2, 0.5, np.array([-0.08, 0.0])), ("front", fus.DROPOUT, 1.2, 1.5, None)]
    clean = _sim(fus)
    bad = _sim(fus, faults=faults)
    assert np.array_equal(clean[0], bad[0]) and np.array_equal(clean[4], bad[4])
    d_a0, d_m0, d_a1, d_m1 = clean[2], clean[3], bad[2], bad[3]
    spike = (np.arange(len(d_a0)) + 0.5) * DT
    in_spike = (spike >= 0.2) & (spike < 0.5)
    in_drop = (spike >= 1.2) & (spike < 1.5)
    # back pad anchored in phase 0 (extend), front pad anchored in phase 1 (contract)
    assert np.allclose(d_a1[in_spike] - d_a0[in_spike], [-0.08 * DT, 0.0])
    assert np.all(d_a1[in_drop] == 0.0)
    others = ~(in_spike | in_drop)
    assert np.array_equal(d_a0[others], d_a1[others]) and np.array_equal(d_m0, d_m1)
    assert np.all(bad[7][in_spike] == fus.SPIKE) and np.all(bad[7][in_drop] == fus.DROPOUT)


# --- filter ---

def test_random_walk_variant_equals_two_pad_pad_fused(fus, two):
    args = _args(fus)
    rng = np.random.default_rng(0)
    setup = fus.build_setup(args, rng)
    tr = fus.simulate_trial(args, setup, rng)
    x1, P1, _, _ = fus.run_trial(tr, args, setup, "random_walk")
    n = setup["noise"]
    noise2 = dict(micro_slip=args.micro_slip_std, free=0.1, sensor=n["sensor"], move=n["move"], gyro=n["gyro"],
                  closure=n["closure"], closure_lat=n["closure_lat"], d_thr=n["d_thr"])
    x2, P2 = two.run_ekf(tr["x_init"], setup["P_init"], tr["gyro"], tr["d_a"], tr["z"], tr["back"], args.dt,
                         "pad_fused", noise2)
    assert np.allclose(x1, x2, atol=1e-12)
    assert np.allclose(np.array(P1), np.array(P2), atol=1e-15)


def _noise(fus):
    args = _args(fus)
    return fus.build_setup(args, np.random.default_rng(0))["noise"]


def test_both_lies_between_the_two_moving_estimates(fus):
    noise = _noise(fus)
    d_a, e = np.array([0.0001, 0.0]), np.array([0.005, 0.0005])
    d_m = d_a + e + np.array([0.0006, -0.0002])
    _, u_m, _, C_m, _, verdict = fus.increments("both", d_a, d_m, e, 1.0, noise)
    enc = d_a + e
    assert verdict is None
    for i in range(2):
        assert min(d_m[i], enc[i]) <= u_m[i] <= max(d_m[i], enc[i])
    assert np.all(np.diag(C_m) < noise["moving_sensor"] ** 2)


def test_cross_check_equals_both_when_the_sensors_agree(fus):
    noise = _noise(fus)
    d_a, e = np.array([0.0002, 0.0001]), np.array([0.005, 0.0004])
    d_m = d_a + e + np.array([0.0003, 0.0001])
    a = fus.increments("both", d_a, d_m, e, 1.0, noise)
    b = fus.increments("cross_check", d_a, d_m, e, 1.0, noise)
    assert all(np.allclose(p, q) for p, q in zip(a[:5], b[:5])) and b[5] is None


def test_cross_check_blames_the_spiking_sensor(fus):
    noise = _noise(fus)
    e, spike = np.array([0.005, 0.0004]), np.array([-0.004, 0.0])
    # anchored spike, no slip
    u_a, u_m, *_, verdict = fus.increments("cross_check", spike, e.copy(), e, 1.0, noise)
    assert verdict == "anchored" and np.allclose(u_a, 0.0) and np.allclose(u_m, e)
    # moving spike during a slip: the anchored reading shows the slip, the moving one slip + spike
    slip = np.array([-0.004, 0.0])
    u_a, u_m, *_, verdict = fus.increments("cross_check", slip, slip + e + spike, e, 1.0, noise)
    assert verdict == "moving" and np.allclose(u_a, slip) and np.allclose(u_m, slip + e)


def test_inflation_adds_the_disagreement_outer_product(fus):
    noise = _noise(fus)
    e, spike = np.array([0.005, 0.0004]), np.array([-0.004, 0.0])
    d_a, d_m = spike, e.copy()
    r = d_m - (d_a + e)
    on = fus.increments("cross_check", d_a, d_m, e, 1.0, noise)
    off = fus.increments("cross_check", d_a, d_m, e, 1.0, dict(noise, inflate=False))
    assert np.allclose(on[0], off[0]) and np.allclose(on[1], off[1])
    for i in (2, 3, 4):
        assert np.allclose(on[i] - off[i], np.outer(r, r))


def test_slip_flags_threshold_reads_the_anchored_reading(fus):
    d_a = np.array([[0.003, 0.0], [0.0, 0.0]])
    u_a = np.array([[0.0, 0.0], [0.003, 0.0]])
    assert list(fus.slip_flags("threshold", d_a, u_a, 0.002)) == [True, False]
    assert list(fus.slip_flags("cross_check", d_a, u_a, 0.002)) == [False, True]


def test_detection_counts(fus):
    tick_mid = (np.arange(10) + 0.5) * 0.1
    episodes = [(0, 0.2, 0.5)]
    flags = np.array([0, 0, 1, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
    fault_a = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    true_step = np.array([0, 0, 3, 3, 0, 0, 0, 0, 0, 0]) * 1e-3
    assert fus.detection_counts(flags, tick_mid, episodes, fault_a, true_step, 0.002) == (1, 2, 1, 1, 1)


def test_build_setup_rejects_bad_fault_duration(fus):
    with pytest.raises(SystemExit):
        fus.build_setup(_args(fus, "--t-fault 2.0"), np.random.default_rng(0))


# --- Monte Carlo ---

def test_all_variants_consistent_without_slip_faults_or_stroke_error(fus):
    nees, _ = _mc_nees(fus, "--p-slip 0 --p-fault 0 --stroke-err-std 0", fus.VARIANTS)
    for v, n in nees.items():
        assert 3.8 < n < 6.3, (v, n)


def test_faults_hurt_unchecked_fusion_and_cross_check_limits_the_damage(fus):
    nees, det = _mc_nees(fus, "--p-fault 0.3", ("random_walk", "moving_sensor", "cross_check"))
    assert nees["cross_check"] < nees["random_walk"] < nees["moving_sensor"]
    assert det["cross_check"]["false_fault"] < 0.3 * det["threshold"]["false_fault"]


def test_stroke_error_makes_the_encoder_model_overconfident(fus):
    nees, _ = _mc_nees(fus, "--p-slip 0 --p-fault 0 --stroke-err-std 0.2", ("encoder_model", "moving_sensor"))
    assert nees["encoder_model"] > 7.0 > nees["moving_sensor"]
