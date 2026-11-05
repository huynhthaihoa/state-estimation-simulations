import numpy as np
from dataclasses import replace

from state_estimation_simulations.use_numpy import skid_steer_odometry_ekf as m


def test_path_turns_one_loop_each_way():
    cfg = m.Config()
    w = m.yaw_rate_profile(cfg)
    half = len(w) // 2
    assert np.isclose(w[:half].sum() * m.DT, 2 * np.pi) and np.isclose(w[half:].sum() * m.DT, -2 * np.pi)
    assert np.all(w[:int(cfg.straight / m.DT)] == 0.0)


def test_truth_follows_the_skid_steer_model():
    """Noise-free: the wheel-speed difference over chi * B is the yaw rate."""
    cfg = replace(m.Config(), chi=1.6)
    trial = m.simulate_trial(cfg, np.random.default_rng(0))
    w = m.yaw_rate_profile(cfg)
    assert np.allclose(np.diff(trial.x[:, 2]) / m.DT, w)
    # quantized encoders: (v_r - v_l) / (chi B) within one tick step of the true rate
    step = 1.0 / m.TICKS_PER_M / m.DT
    assert np.max(np.abs((trial.v_r - trial.v_l) / (cfg.chi * m.TRACK) - w)) <= 2 * step / (cfg.chi * m.TRACK) + 1e-12


def test_encoder_distance_is_exact_to_one_tick():
    cfg = m.Config()
    trial = m.simulate_trial(cfg, np.random.default_rng(1))
    true_dist = cfg.speed * cfg.duration
    meas = 0.5 * (trial.v_l + trial.v_r).sum() * m.DT
    assert abs(meas - true_dist) <= 1.0 / m.TICKS_PER_M


def test_dead_reckoning_with_the_true_chi_tracks_the_truth():
    cfg = m.Config()
    trial = m.simulate_trial(cfg, np.random.default_rng(2))
    x = m.dead_reckoning(trial, cfg.chi)
    assert np.max(np.linalg.norm(x[:, :2] - trial.x[:, :2], axis=1)) < 1e-3


def test_measurement_jacobian():
    cfg = m.Config()
    f = m.SkidSteerEKF(cfg, True)
    f.x = np.array([1.0, -2.0, 0.3, 0.12, 0.004, 1.7])
    h0, H = f.measurement()
    eps = 1e-7
    for i in range(6):
        f.x[i] += eps
        h1, _ = f.measurement()
        f.x[i] -= eps
        assert np.allclose((h1 - h0) / eps, H[:, i], atol=1e-6)


def test_ekf_learns_the_bias_and_chi():
    cfg = replace(m.Config(), chi=1.3)
    trial = m.simulate_trial(cfg, np.random.default_rng(3))
    xs, Ps = m.run_ekf(trial, cfg, True)
    assert abs(xs[-1, 5] - 1.3) < 0.01
    assert abs(xs[-1, 4] - trial.bias) < np.radians(0.01)


def test_ekf_is_consistent_with_white_wheel_noise():
    """With white wheel-speed noise of the modeled size (not quantization),
    the averaged 3-DoF NEES of ekf_chi lies in its 95% band."""
    cfg = m.Config()
    rng = np.random.default_rng(4)
    sd = np.sqrt(m.wheel_speed_var())
    w = m.yaw_rate_profile(cfg)
    vals = []
    n_tr = 6
    for _ in range(n_tr):
        tr = m.simulate_trial(cfg, rng)
        vr = cfg.speed + w * cfg.chi * m.TRACK / 2 + rng.normal(size=len(w)) * sd
        vl = cfg.speed - w * cfg.chi * m.TRACK / 2 + rng.normal(size=len(w)) * sd
        t = m.Trial(tr.x, vl, vr, tr.gyro, tr.bias)
        xs, Ps = m.run_ekf(t, cfg, True)
        e = t.x - xs[:, :3]
        e[:, 2] = m.wrap(e[:, 2])
        vals.append(np.mean([m.nees(e[k], Ps[k, :3, :3]) for k in range(50, len(e))]))
    lo, hi = m.averaged_nees_bounds(3, n_tr)
    assert lo < np.mean(vals) < hi
