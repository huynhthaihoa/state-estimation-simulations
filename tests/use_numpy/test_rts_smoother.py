import numpy as np
import pytest

from state_estimation_simulations.use_numpy import rts_smoother as _rts_smoother
from state_estimation_simulations.use_numpy import inchworm_zupt_ekf as _inchworm_zupt_ekf

DT = 0.05
Q_STD = 0.5
POS_STD = 0.02
ZUPT_STD = 0.01
INIT_STD = np.array([0.05, 0.05])


@pytest.fixture
def m():
    return _rts_smoother


@pytest.fixture
def base():
    return _inchworm_zupt_ekf


def _setup(base, seed=0, duration=10.0, pos_std=POS_STD):
    rng = np.random.default_rng(seed)
    x_true, z, is_anchor, is_cruise = base.generate_ground_truth_and_data(
        duration, DT, 1.0, 1.0, 0.2, 0.1, pos_std, rng)
    x_init = x_true[0] + rng.normal(0.0, 1.0, 2) * INIT_STD
    return dict(x_true=x_true, z=z, is_anchor=is_anchor, is_cruise=is_cruise, x_init=x_init,
                P_init=np.diag(INIT_STD ** 2), z_vel=np.where(is_anchor, 0.0, np.nan),
                n_steps=len(x_true) - 1, R_pos=pos_std ** 2, R_vel=ZUPT_STD ** 2)


def _forward(m, s, q=Q_STD):
    return m.run_forward_filter(s["x_init"], s["P_init"], s["z"], s["z_vel"], DT, s["n_steps"], q,
                                s["R_pos"], s["R_vel"])


# --- model pieces: finite-difference check of the linear maps ---

def test_transition_matrix_matches_parent_predict_by_finite_difference(m, base):
    x0 = np.array([0.3, -0.2])
    h = 1e-6
    J = np.zeros((2, 2))
    for i in range(2):
        d = np.zeros(2)
        d[i] = h
        xp, _ = base.predict(x0 + d, np.eye(2), DT, Q_STD)
        xm, _ = base.predict(x0 - d, np.eye(2), DT, Q_STD)
        J[:, i] = (xp - xm) / (2 * h)
    np.testing.assert_allclose(J, m.transition_matrix(DT), atol=1e-8)


def test_process_noise_covariance_matches_parent_predict(m, base):
    _, P = base.predict(np.zeros(2), np.zeros((2, 2)), DT, Q_STD)
    np.testing.assert_allclose(P, m.process_noise_covariance(DT, Q_STD), atol=1e-15)


# --- forward filter ---

def test_forward_filter_reproduces_parent_phase_conditional_filter(m, base):
    s = _setup(base)
    fwd = _forward(m, s)
    x_ref, P_ref = base.run_ekf_phase_conditional_zupt(
        s["x_init"], s["P_init"], s["z"], DT, s["n_steps"], Q_STD, s["R_pos"], s["R_vel"], s["is_anchor"])
    np.testing.assert_allclose(fwd["x_f"], x_ref, atol=1e-12)
    np.testing.assert_allclose(fwd["P_f"], np.array(P_ref), atol=1e-12)


def test_forward_filter_without_history_returns_the_final_state(m, base):
    s = _setup(base)
    fwd = _forward(m, s)
    x, P = m.run_forward_filter(s["x_init"], s["P_init"], s["z"], s["z_vel"], DT, s["n_steps"], Q_STD,
                                s["R_pos"], s["R_vel"], store_history=False)
    np.testing.assert_allclose(x, fwd["x_f"][-1], atol=1e-14)
    np.testing.assert_allclose(P, fwd["P_f"][-1], atol=1e-14)


# --- RTS smoother vs batch least squares ---

def test_rts_means_and_covariances_equal_batch_least_squares(m, base):
    s = _setup(base)
    fwd = _forward(m, s)
    x_s, P_s, _ = m.rts_smoother(fwd)
    X, P_blocks = m.batch_least_squares(s["x_init"], s["P_init"], s["z"], s["z_vel"], DT, Q_STD,
                                        s["R_pos"], s["R_vel"])
    assert np.max(np.abs(x_s - X)) < 1e-9
    assert np.max(np.abs(P_s - P_blocks)) < 1e-9


def test_rts_equals_batch_with_a_different_seed_and_process_noise(m, base):
    s = _setup(base, seed=3)
    fwd = _forward(m, s, q=0.15)
    x_s, _, _ = m.rts_smoother(fwd)
    X, _ = m.batch_least_squares(s["x_init"], s["P_init"], s["z"], s["z_vel"], DT, 0.15, s["R_pos"], s["R_vel"])
    assert np.max(np.abs(x_s - X)) < 1e-9


def test_batch_matches_normal_equations_gradient(m, base):
    # the batch solution is a stationary point of the cost: perturbing it never lowers the cost
    s = _setup(base, duration=2.0)
    X, _ = m.batch_least_squares(s["x_init"], s["P_init"], s["z"], s["z_vel"], DT, Q_STD, s["R_pos"], s["R_vel"])
    F = m.transition_matrix(DT)
    Qi = np.linalg.inv(m.process_noise_covariance(DT, Q_STD))
    P0i = np.linalg.inv(s["P_init"])

    def cost(X):
        c = (X[0] - s["x_init"]) @ P0i @ (X[0] - s["x_init"])
        for k in range(len(X) - 1):
            w = X[k + 1] - F @ X[k]
            c += w @ Qi @ w
        for k in range(1, len(X)):
            c += (X[k, 0] - s["z"][k]) ** 2 / s["R_pos"]
            if not np.isnan(s["z_vel"][k]):
                c += (X[k, 1] - s["z_vel"][k]) ** 2 / s["R_vel"]
        return c

    rng = np.random.default_rng(1)
    c0 = cost(X)
    for _ in range(5):
        assert cost(X + 1e-4 * rng.normal(size=X.shape)) > c0


# --- covariance ordering ---

def test_smoother_covariance_not_larger_than_filter_and_equal_at_last_tick(m, base):
    s = _setup(base)
    fwd = _forward(m, s)
    _, P_s, _ = m.rts_smoother(fwd)
    for k in range(s["n_steps"] + 1):
        assert np.linalg.eigvalsh(fwd["P_f"][k] - P_s[k]).min() > -1e-12
    np.testing.assert_allclose(P_s[-1], fwd["P_f"][-1], atol=1e-15)
    # and it is strictly smaller somewhere (the smoother does add information)
    assert P_s[10, 1, 1] < 0.9 * fwd["P_f"][10, 1, 1]


def test_smoother_mean_equals_filter_mean_at_last_tick(m, base):
    s = _setup(base)
    fwd = _forward(m, s)
    x_s, _, _ = m.rts_smoother(fwd)
    np.testing.assert_allclose(x_s[-1], fwd["x_f"][-1], atol=1e-15)


# --- zero noise recovers the truth ---

def test_noise_free_data_recovers_truth(m, base):
    # exact positions on the inchworm truth: positions come back to well below the 2 cm noise; the velocity
    # is not pinned that tightly because the model's own process noise (q = 0.5 m/s^2) lets p drift 6e-4 m per tick
    s = _setup(base, pos_std=1e-6)
    s["z"] = s["x_true"][:, 0].copy()
    s["x_init"] = s["x_true"][0].copy()
    s["P_init"] = np.diag([1e-8, 1e-8])
    fwd = m.run_forward_filter(s["x_init"], s["P_init"], s["z"], s["z_vel"], DT, s["n_steps"], Q_STD,
                               1e-12, 1e-6 ** 2)
    x_s, _, _ = m.rts_smoother(fwd)
    assert np.max(np.abs(x_s[:, 0] - s["x_true"][:, 0])) < 1e-4


def test_noise_free_constant_velocity_truth_is_recovered_exactly(m):
    n = 60
    t = np.arange(n + 1) * DT
    x_true = np.stack([0.3 + 0.1 * t, np.full(n + 1, 0.1)], axis=1)
    fwd = m.run_forward_filter(x_true[0].copy(), np.diag([1e-8, 1e-8]), x_true[:, 0].copy(), np.full(n + 1, np.nan),
                               DT, n, 1e-3, 1e-12, 1e-4)
    x_s, _, _ = m.rts_smoother(fwd)
    np.testing.assert_allclose(x_s, x_true, atol=1e-6)
    np.testing.assert_allclose(fwd["x_f"], x_true, atol=1e-6)


# --- fixed lag ---

def test_fixed_lag_limits_are_filter_and_full_smoother(m, base):
    s = _setup(base)
    fwd = _forward(m, s)
    x_s, _, G = m.rts_smoother(fwd)
    np.testing.assert_allclose(m.fixed_lag_estimates(fwd, G, 0), fwd["x_f"], atol=1e-15)
    np.testing.assert_allclose(m.fixed_lag_estimates(fwd, G, s["n_steps"]), x_s, atol=1e-12)


def test_fixed_lag_equals_rts_on_the_truncated_data(m, base):
    s = _setup(base)
    fwd = _forward(m, s)
    _, _, G = m.rts_smoother(fwd)
    lag, k = 7, 30
    x_lag = m.fixed_lag_estimates(fwd, G, lag)[k]
    top = k + lag
    fwd_cut = m.run_forward_filter(s["x_init"], s["P_init"], s["z"][:top + 1], s["z_vel"][:top + 1], DT, top,
                                   Q_STD, s["R_pos"], s["R_vel"])
    x_cut, _, _ = m.rts_smoother(fwd_cut)
    np.testing.assert_allclose(x_lag, x_cut[k], atol=1e-12)


# --- windows ---

def test_windows_partition_ticks_after_zero(m, base):
    s = _setup(base)
    w = m.phase_windows(s["is_anchor"], s["is_cruise"])
    total = sum(w[n].astype(int) for n in m.WINDOWS)
    assert total[0] == 0
    assert np.all(total[1:] == 1)
    # each extend phase: 4 ramp-up ticks then 4 ramp-down ticks (t_ramp 0.2 s, dt 0.05 s)
    assert w["ramp-up"].sum() == 20 and w["ramp-down"].sum() == 20


# --- Monte Carlo (regression, loose bands) ---

def _mc(m, base, scenario, q, n_trials=60):
    s = _setup(base)
    return m.run_monte_carlo(scenario, s["x_true"], DT, q, POS_STD, ZUPT_STD, s["is_anchor"], INIT_STD,
                             n_trials, (0, 10), np.random.default_rng([0, 2]))


def test_matched_truth_filter_and_smoother_nees_near_two(m, base):
    mc = _mc(m, base, "matched", Q_STD)
    lo, hi = 1.80, 2.20  # loose band around the expected 2 for a long-run mean of a 60-trial average
    assert lo < np.mean(mc["nees_f"][1:]) < hi
    assert lo < np.mean(mc["nees_s"][1:]) < hi


def test_inchworm_truth_smoother_is_more_accurate_and_not_overconfident(m, base):
    mc = _mc(m, base, "inchworm", Q_STD)
    sl = slice(1, None)
    assert np.sqrt(np.mean(mc["sq_pos_s"][sl])) < 0.7 * np.sqrt(np.mean(mc["sq_pos_f"][sl]))
    assert np.sqrt(np.mean(mc["sq_vel_s"][sl])) < 0.6 * np.sqrt(np.mean(mc["sq_vel_f"][sl]))
    assert np.mean(mc["nees_s"][1:]) < 2.5


def test_fixed_lag_zero_in_monte_carlo_equals_filter_error(m, base):
    mc = _mc(m, base, "inchworm", Q_STD, n_trials=10)
    np.testing.assert_allclose(mc["sq_pos_lag"][0], mc["sq_pos_f"], atol=1e-15)
    np.testing.assert_allclose(mc["sq_vel_lag"][0], mc["sq_vel_f"], atol=1e-15)


def test_nees_per_tick_matches_utils_nees(m):
    from state_estimation_simulations.utils import nees
    rng = np.random.default_rng(0)
    err = rng.normal(size=(6, 2))
    A = rng.normal(size=(6, 2, 2))
    P = A @ A.transpose(0, 2, 1) + np.eye(2)
    np.testing.assert_allclose(m.nees_per_tick(err, P), [nees(err[k], P[k]) for k in range(6)], rtol=1e-12)


def test_use_zupt_false_removes_the_zero_velocity_information(m, base):
    s = _setup(base)
    kw = dict(scenario="inchworm", x_nominal=s["x_true"], dt=DT, process_noise_std=Q_STD, pos_noise_std=POS_STD,
              zupt_noise_std=ZUPT_STD, is_anchor=s["is_anchor"], init_std=INIT_STD, n_trials=10, lags=())
    with_z = m.run_monte_carlo(rng=np.random.default_rng(1), **kw)
    without = m.run_monte_carlo(rng=np.random.default_rng(1), use_zupt=False, **kw)
    anchor = m.phase_windows(s["is_anchor"], s["is_cruise"])["anchor"]
    # without ZUPT the filter's velocity in the anchor phase is far less accurate
    assert np.mean(without["sq_vel_f"][anchor]) > 20 * np.mean(with_z["sq_vel_f"][anchor])
