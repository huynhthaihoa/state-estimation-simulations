import numpy as np
import pytest

from state_estimation_simulations.use_numpy import imu_allan_variance as _imu_allan_variance
from state_estimation_simulations.use_numpy.eskf_imu_position_fix import generate_trial


@pytest.fixture
def imu_allan_variance():
    return _imu_allan_variance


def _stream(seed):
    return np.random.default_rng(seed)


# --- Allan deviation of the two noise terms against the closed forms ---

def test_white_noise_allan_deviation_follows_n_over_sqrt_tau(imu_allan_variance):
    m = imu_allan_variance
    dt, N = 0.01, 0.02
    y = m.simulate_static_axis(400_000, dt, N, 0.0, _stream(0), _stream(1))
    tau, adev = m.allan_deviation(y, dt, [1, 10, 100, 1000])
    assert np.allclose(adev, N / np.sqrt(tau), rtol=0.08)


def test_random_walk_allan_deviation_follows_k_sqrt_tau_over_3(imu_allan_variance):
    m = imu_allan_variance
    dt, K = 0.01, 0.003
    y = m.simulate_static_axis(400_000, dt, 0.0, K, _stream(0), _stream(1))
    tau, adev = m.allan_deviation(y, dt, [100, 300, 1000])
    assert np.allclose(adev, K * np.sqrt(tau / 3.0), rtol=0.15)
    assert np.allclose(m.model_allan_deviation(tau, 0.0, K, dt), K * np.sqrt(tau / 3.0), rtol=1e-3)


def test_discrete_random_walk_formula_matches_brute_force_variance(imu_allan_variance):
    # exact variance of the estimator's term x[2m] - 2 x[m] + x[0] for a bias walk with per-step std s
    m_mod = imu_allan_variance
    dt, K = 0.01, 0.5
    s = K * np.sqrt(dt)
    for m in (1, 2, 5, 40):
        k = np.arange(1, 2 * m + 1)

        def coef(j):
            return dt * np.clip(j - k + 1, 0, None)

        c = coef(2 * m) - 2.0 * coef(m)
        var = s ** 2 * np.sum(c ** 2) / (2.0 * (m * dt) ** 2)
        assert np.isclose(m_mod.model_allan_deviation(m * dt, 0.0, K, dt) ** 2, var, rtol=1e-12)


def test_white_term_is_exact_in_discrete_time(imu_allan_variance):
    m = imu_allan_variance
    dt, N = 0.005, 0.003
    assert np.allclose(m.model_allan_deviation(np.array([dt, 0.1]), N, 0.0, dt) ** 2, N ** 2 / np.array([dt, 0.1]))


# --- estimator vs slow reference ---

def test_estimator_matches_slow_reference(imu_allan_variance):
    m = imu_allan_variance
    rng = _stream(3)
    y = rng.normal(size=300) + np.cumsum(rng.normal(size=300) * 0.1)
    sizes = [1, 2, 3, 7, 20, 50, 100]
    tau, adev = m.allan_deviation(y, 0.05, sizes)
    tau_ref, adev_ref = m.allan_deviation_reference(y, 0.05, sizes)
    assert np.allclose(tau, tau_ref)
    assert np.allclose(adev, adev_ref, rtol=1e-10, atol=0.0)


def test_cluster_sizes_leave_nine_clusters(imu_allan_variance):
    m = imu_allan_variance
    sizes = m.cluster_sizes(10_000)
    assert sizes[0] == 1 and sizes[-1] == 10_000 // 9 and np.all(np.diff(sizes) > 0)


# --- simulation streams ---

def test_shorter_record_is_a_prefix_of_a_longer_one(imu_allan_variance):
    m = imu_allan_variance
    short = m.simulate_static_axis(1000, 0.01, 0.02, 0.003, _stream(0), _stream(1))
    long = m.simulate_static_axis(4000, 0.01, 0.02, 0.003, _stream(0), _stream(1))
    assert np.array_equal(short, long[:1000])


# --- reading the parameters off the curve ---

def test_fit_on_the_noise_free_curve_is_within_a_few_percent(imu_allan_variance):
    m = imu_allan_variance
    dt = 0.005
    for sensor in ("gyro", "accel"):
        p = m.PRESETS["euroc"][sensor]
        tau = m.cluster_sizes(int(4 * 3600 / dt)) * dt
        fit = m.fit_noise_parameters(tau, m.model_allan_deviation(tau, p["density"], p["walk"], dt))
        assert abs(fit["density"] / p["density"] - 1.0) < 0.015
        assert abs(fit["walk"] / p["walk"] - 1.0) < 0.03
        assert fit["white_range"][0] == tau[0] and fit["walk_range"][0] > fit["white_range"][1]


def test_parameter_recovery_on_a_fixed_seed(imu_allan_variance):
    m = imu_allan_variance
    p = m.PRESETS["euroc"]["accel"]
    dt = 1.0 / m.PRESETS["euroc"]["rate_hz"]
    n = int(0.25 * 3600 / dt)
    y = m.simulate_static_axis(n, dt, p["density"], p["walk"], _stream(0), _stream(1))
    tau, adev = m.allan_deviation(y, dt, m.cluster_sizes(n))
    fit = m.fit_noise_parameters(tau, adev)
    assert abs(fit["density"] / p["density"] - 1.0) < 0.03
    assert abs(fit["walk"] / p["walk"] - 1.0) < 0.20


def test_unresolved_random_walk_is_nan_not_a_guess(imu_allan_variance):
    m = imu_allan_variance
    dt = 0.01
    y = m.simulate_static_axis(20_000, dt, 0.02, 0.0, _stream(0), _stream(1))   # white noise only
    tau, adev = m.allan_deviation(y, dt, m.cluster_sizes(20_000))
    assert np.isnan(m.fit_noise_parameters(tau, adev)["walk"])


# --- Part 2: conversion and ESKF ---

def test_conversion_stds(imu_allan_variance):
    m = imu_allan_variance
    dt = 0.01
    base = np.array(m.conversion_noise_stds(m.PRESETS["euroc"], dt, "correct"))
    assert np.allclose(base[0], m.PRESETS["euroc"]["gyro"]["density"] / np.sqrt(dt))
    assert np.allclose(m.conversion_noise_stds(m.PRESETS["euroc"], dt, "density_as_std"), base * np.sqrt(dt))
    assert np.allclose(m.conversion_noise_stds(m.PRESETS["euroc"], dt, "double_conversion"), base / np.sqrt(dt))


def test_bias_walk_adds_nothing_without_walk_and_leaves_the_main_stream_alone(imu_allan_variance):
    m = imu_allan_variance
    kw = dict(duration=5.0, dt=0.01, fix_rate=1.0, rate_amp=0.4, accel_amp=0.5, gyro_noise_std=0.002,
              accel_noise_std=0.02, gyro_bias_std=0.01, accel_bias_std=0.05, pos_noise_std=0.1,
              init_pos_noise_std=0.1, init_vel_noise_std=0.1, init_att_noise_std=0.03)
    rng = _stream(0)
    trial = generate_trial(rng=rng, **kw)
    before = rng.bit_generator.state
    zero = m.add_bias_walk(trial, m.scaled_preset(m.PRESETS["euroc"], 1.0, 0.0), rng)
    assert np.array_equal(zero["gyro"], trial["gyro"]) and np.array_equal(zero["accel"], trial["accel"])
    walk = m.add_bias_walk(trial, m.PRESETS["euroc"], rng)
    assert not np.array_equal(walk["accel"], trial["accel"])
    assert rng.bit_generator.state == before


def test_correct_conversion_is_consistent_and_density_as_std_is_not_without_bias_walk(imu_allan_variance):
    m = imu_allan_variance
    from state_estimation_simulations.utils import averaged_nees_bounds
    n_trials = 6
    res = m.eskf_conversion_experiment(m.scaled_preset(m.PRESETS["euroc"], 1.0, 0.0), 0.01, 20.0, n_trials,
                                       _stream(5))
    lo, hi = averaged_nees_bounds(15, n_trials)
    correct = m.second_half_mean(res["correct"]["nees"])
    assert 0.5 * lo < correct < 2.0 * hi               # loose band: short run, few trials
    assert m.second_half_mean(res["density_as_std"]["nees"]) > 3.0 * correct
    assert m.second_half_mean(res["double_conversion"]["nees"]) < lo


def test_block_check_joint_nees_equals_the_experiment_nees_on_the_same_trials(imu_allan_variance):
    m = imu_allan_variance
    preset = m.PRESETS["euroc"]

    def fresh():
        return np.random.default_rng(np.random.SeedSequence(7))

    res = m.eskf_conversion_experiment(preset, 0.01, 6.0, 2, fresh())
    blocks = m.block_nees_experiment(preset, 0.01, 6.0, 2, fresh())
    for c in m.CONVERSIONS:
        assert np.isclose(blocks[c]["joint"], m.second_half_mean(res[c]["nees"]), rtol=1e-9)
        assert blocks[c]["position"] > 0.0
