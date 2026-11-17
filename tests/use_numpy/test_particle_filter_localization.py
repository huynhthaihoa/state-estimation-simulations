import numpy as np
import pytest

from state_estimation_simulations.use_numpy import particle_filter_localization as _pf


@pytest.fixture
def pf():
    return _pf


DOORS = np.array([3.0, 7.0, 12.0])


# --- sensor profile ---

def test_profile_gradient_matches_finite_difference(pf):
    x = np.linspace(-1.0, 15.0, 400)
    for width in (0.15, 0.3, 0.6):
        h = 1e-6
        fd = (pf.door_profile(x + h, DOORS, width) - pf.door_profile(x - h, DOORS, width)) / (2 * h)
        np.testing.assert_allclose(pf.door_profile_grad(x, DOORS, width), fd, atol=1e-6)


def test_profile_peaks_at_doors_and_is_zero_far_away(pf):
    assert pf.door_profile(3.0, DOORS, 0.3) == pytest.approx(1.0)
    assert pf.door_profile(5.0, DOORS, 0.3) < 1e-9
    assert pf.door_profile_grad(3.0, DOORS, 0.3) == pytest.approx(0.0)


# --- weights, ESS, resampler ---

def test_ess_uniform_is_n_and_one_hot_is_one(pf):
    n = 50
    assert pf.effective_sample_size(np.full(n, 1.0 / n)) == pytest.approx(n)
    w = np.zeros(n)
    w[7] = 1.0
    assert pf.effective_sample_size(w) == pytest.approx(1.0)


def test_low_variance_resampler_copy_counts(pf):
    w = np.array([0.05, 0.15, 0.30, 0.0, 0.50])
    n = len(w)
    # every copy count is within one of n * w (systematic resampling), for any offset
    for r in (0.0, 0.05, 0.1, 0.1999):
        counts = np.bincount(pf.low_variance_resample(w, r), minlength=n)
        assert counts.sum() == n
        assert np.all(np.abs(counts - n * w) < 1.0)
        assert counts[3] == 0
    # expected copy count equals n * w when the offset is uniform
    rng = np.random.default_rng(0)
    big_n = 40
    w = rng.random(big_n)
    w /= w.sum()
    acc = np.zeros(big_n)
    draws = 4000
    for _ in range(draws):
        acc += np.bincount(pf.low_variance_resample(w, rng.random() / big_n), minlength=big_n)
    np.testing.assert_allclose(acc / draws, big_n * w, atol=0.05)


def test_pf_weights_normalize_and_resampling_resets(pf):
    cfg = pf.Config()
    noise = pf.draw_noise(1, 1, cfg.steps)[0]
    out = pf.pf_on_trial(cfg, noise, pf.generate_trial(cfg, noise), record=True)
    np.testing.assert_allclose(out["w_hist"].sum(axis=1), 1.0, atol=1e-12)
    assert np.all(out["ess"] <= cfg.n_particles + 1e-9) and np.all(out["ess"] >= 1.0 - 1e-9)
    assert out["resampled"].sum() > 0


def test_pf_is_deterministic_for_fixed_streams(pf):
    cfg = pf.Config(n_particles=200)
    noise = pf.draw_noise(3, 1, cfg.steps)[0]
    data = pf.generate_trial(cfg, noise)
    a, b = pf.pf_on_trial(cfg, noise, data), pf.pf_on_trial(cfg, noise, data)
    np.testing.assert_array_equal(a["mean"], b["mean"])


# --- PF vs KF on a linear Gaussian model ---

def test_pf_matches_kalman_filter_on_linear_gaussian_model(pf):
    rng = np.random.default_rng(5)
    t, a, b = 12, 1.5, 0.3
    u, sigma_u, sigma_z = np.full(t, 0.4), np.full(t, 0.2), 0.3
    x = np.empty(t + 1)
    x[0] = 0.0
    x[1:] = np.cumsum(u + sigma_u * rng.standard_normal(t))
    z = a * x + b + sigma_z * rng.standard_normal(t + 1)
    m0, p0 = 0.2, 1.0
    h = lambda s: a * s + b
    mean_kf, var_kf = pf.run_ekf(z, u, sigma_u, h, lambda s: a, sigma_z, m0, p0)   # exact KF: h linear
    n = 40000
    x_init = m0 + np.sqrt(p0) * np.random.default_rng(6).standard_normal(n)
    out = pf.run_pf(z, u, sigma_u, h, sigma_z, x_init, np.random.default_rng(7), np.random.default_rng(8),
                    0.5, 1.0)
    np.testing.assert_allclose(out["mean"], mean_kf, atol=0.03)
    np.testing.assert_allclose(out["var"], var_kf, rtol=0.06)


# --- EKF / UKF ---

def test_ukf_equals_ekf_for_linear_measurement(pf):
    t = 10
    u, sigma_u = np.full(t, 0.3), np.full(t, 0.1)
    z = np.random.default_rng(1).standard_normal(t + 1)
    h = lambda s: 2.0 * s - 1.0
    e = pf.run_ekf(z, u, sigma_u, h, lambda s: 2.0, 0.4, 0.5, 2.0)
    for kappa in (2.0, 0.0):
        k = pf.run_ukf(z, u, sigma_u, h, 0.4, 0.5, 2.0, 1.0, 2.0, kappa)
        np.testing.assert_allclose(k[0], e[0], atol=1e-9)
        np.testing.assert_allclose(k[1], e[1], atol=1e-9)


def test_ukf_agrees_with_ekf_when_sigma_small_near_truth(pf):
    cfg = pf.Config(sigma_z=0.05, motion_floor=0.005, motion_factor=0.0)
    noise = pf.draw_noise(2, 1, cfg.steps)[0]
    x_true, z, _, u, sigma_u = pf.generate_trial(cfg, noise)
    h = lambda s: pf.door_profile(s, cfg.doors, cfg.width)
    dh = lambda s: pf.door_profile_grad(s, cfg.doors, cfg.width)
    start = x_true[0] + 0.05          # near the truth, narrow prior
    e = pf.run_ekf(z, u, sigma_u, h, dh, cfg.sigma_z, start, 0.05 ** 2)
    k = pf.run_ukf(z, u, sigma_u, h, cfg.sigma_z, start, 0.05 ** 2, cfg.ukf_alpha, cfg.ukf_beta, cfg.ukf_kappa)
    assert np.max(np.abs(e[0] - k[0])) < 0.05
    assert np.max(np.abs(e[0] - x_true)) < 0.3


def test_zero_noise_filters_stay_on_truth(pf):
    t = 20
    u, sigma_u = np.full(t, 0.25), np.zeros(t)
    doors = np.array([2.0, 5.0])
    x = 1.0 + np.concatenate([[0.0], np.cumsum(u)])
    h = lambda s: pf.door_profile(s, doors, 0.3)
    dh = lambda s: pf.door_profile_grad(s, doors, 0.3)
    z = h(x)
    m, p = pf.run_ekf(z, u, sigma_u, h, dh, 0.05, 1.0, 1e-4)
    np.testing.assert_allclose(m, x, atol=1e-6)
    m, p = pf.run_ukf(z, u, sigma_u, h, 0.05, 1.0, 1e-4, 1.0, 2.0, 2.0)
    np.testing.assert_allclose(m, x, atol=1e-4)   # sigma points sample f over +-0.017 m: tiny nonlinearity error
    x_init = 1.0 + 0.01 * np.random.default_rng(0).standard_normal(2000)
    out = pf.run_pf(z, u, sigma_u, h, 0.05, x_init, np.random.default_rng(1), np.random.default_rng(2), 0.5, 1.0)
    np.testing.assert_allclose(out["mean"], x, atol=0.01)


# --- ML association variant ---

def test_ml_door_index_picks_nearest_predicted_offset(pf):
    predicted = np.array([-4.0, 1.0, 6.0])
    s = np.full(3, 0.5)
    assert pf.ml_door_index(0.8, predicted, s) == 1
    assert pf.ml_door_index(5.0, predicted, s) == 2


def test_ukf_ml_equals_ekf_ml(pf):
    cfg = pf.Config()
    noises = pf.draw_noise(4, 5, cfg.steps)
    m0, p0 = pf.prior_gaussian(cfg)
    for noise in noises:
        _, _, z_ml, u, sigma_u = pf.generate_trial(cfg, noise)
        a = pf.run_ekf_ml(z_ml, u, sigma_u, cfg.doors, cfg.sigma_c, m0, p0)
        b = pf.run_ukf_ml(z_ml, u, sigma_u, cfg.doors, cfg.sigma_c, m0, p0, 1.0, 2.0, 2.0)
        np.testing.assert_allclose(a[0], b[0], atol=1e-9)
        np.testing.assert_allclose(a[1], b[1], atol=1e-9)
        np.testing.assert_array_equal(a[2], b[2])


# --- data generation and scoring ---

def test_noise_draws_are_independent_of_sweep_settings(pf):
    noises = pf.draw_noise(0, 3, 48)
    c1, c2 = pf.Config(sigma_z=0.05), pf.Config(sigma_z=0.4)
    x1 = [pf.generate_trial(c1, n)[0] for n in noises]
    x2 = [pf.generate_trial(c2, n)[0] for n in noises]
    np.testing.assert_array_equal(np.stack(x1), np.stack(x2))   # truth does not depend on the sensor noise


def test_truth_starts_in_zone_and_ends_near_corridor_end(pf):
    cfg = pf.Config()
    lo, hi = cfg.zone
    for noise in pf.draw_noise(0, 50, cfg.steps):
        x = pf.generate_trial(cfg, noise)[0]
        assert lo <= x[0] <= hi
        assert x.max() < cfg.length + 0.5     # start at the zone edge + travel + motion noise: may pass L by ~0.2 m


def test_one_hill_step(pf):
    mass = np.array([[0.2, 0.95, 0.5, 0.95, 0.99], [0.1, 0.2, 0.3, 0.4, 0.5], [0.95, 0.95, 0.95, 0.95, 0.95]])
    np.testing.assert_array_equal(pf.one_hill_step(mass, 0.9), [3, -1, 0])


def test_nees_is_consistent_for_the_pf_on_a_small_run(pf):
    cfg = pf.Config(n_particles=500)
    noises = pf.draw_noise(0, 30, cfg.steps)
    x_true, res = pf.run_monte_carlo(cfg, noises)
    sc = pf.score(cfg, x_true, res)
    nees, steps_in, lo, hi = sc["PF"]["nees_right"]
    assert 0.5 < nees < 2.0
    assert sc["PF"]["wrong"] < 0.2
    assert sc["EKF"]["wrong"] > sc["PF"]["wrong"]


def test_known_start_diagnostic_all_filters_consistent(pf):
    cfg = pf.Config(prior_std=0.1, n_particles=500)
    noises = pf.draw_noise(0, 60, cfg.steps)
    x_true, res = pf.run_monte_carlo(cfg, noises)
    sc = pf.score(cfg, x_true, res)
    for name in ("PF", "EKF", "UKF"):
        assert sc[name]["wrong"] == 0.0
        assert sc[name]["final_rmse"] < 0.15
        assert 0.5 < sc[name]["nees_all"][0] < 2.0     # narrow prior: linearization is good, all consistent
