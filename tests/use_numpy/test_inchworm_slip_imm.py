import argparse
import copy

import numpy as np
import pytest

from state_estimation_simulations.use_numpy import inchworm_slip_gate_ekf as _gate
from state_estimation_simulations.use_numpy import inchworm_slip_imm as _imm
from state_estimation_simulations.use_numpy import inchworm_zupt_ekf as _base
from state_estimation_simulations.utils import nees as _nees


@pytest.fixture
def imm():
    return _imm


@pytest.fixture
def gate():
    return _gate


@pytest.fixture
def base():
    return _base


def _args(**over):
    p = argparse.ArgumentParser()
    _imm.add_model_arguments(p)
    a = p.parse_args([])
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _data(imm, seed=0, **over):
    a = _args(seed=seed, **over)
    s = imm.make_setup(a, a.sensor_noise_std)
    R = (a.pos_noise_std ** 2, a.zupt_noise_std ** 2, (a.sensor_noise_std / a.dt) ** 2)
    return a, s, R


def _always_moving(n):
    return np.broadcast_to(np.array([[0.0, 1.0], [0.0, 1.0]]), (n, 2, 2)).copy(), np.array([0.0, 1.0])


# --- measurement update with a likelihood ---

def test_update_state_and_covariance_equal_kf_update(imm, base):
    x, P = np.array([0.3, -0.2]), np.array([[0.04, 0.01], [0.01, 0.09]])
    H, R = np.array([[1.0, 0.0]]), np.array([[0.02 ** 2]])
    x1, P1, lik = imm.kf_update_with_likelihood(x, P, np.array([0.31]), H, R)
    x2, P2 = base._kf_update(x, P, np.array([0.31]), H, R)
    np.testing.assert_allclose(x1, x2, atol=1e-12)
    np.testing.assert_allclose(P1, P2, atol=1e-12)
    assert lik > 0


def test_stacked_position_and_velocity_update_equals_the_sequential_parent_updates(imm, base, gate):
    x, P = np.array([0.1, 0.05]), np.array([[0.04, 0.01], [0.01, 0.09]])
    z, v, R_pos, R_vel = 0.13, -0.02, 0.02 ** 2, 0.01 ** 2
    xs, Ps, _ = imm.kf_update_with_likelihood(x, P, np.array([z, v]), np.eye(2), np.diag([R_pos, R_vel]))
    xq, Pq = base.measurement_update_position(x, P, z, R_pos)
    xq, Pq = gate.measurement_update_velocity(xq, Pq, v, R_vel)
    np.testing.assert_allclose(xs, xq, atol=1e-12)
    np.testing.assert_allclose(Ps, Pq, atol=1e-12)


def test_likelihood_is_the_gaussian_density_of_the_innovation(imm):
    x, P = np.array([0.3, -0.2]), np.array([[0.04, 0.01], [0.01, 0.09]])
    H, R = np.eye(2), np.diag([0.02 ** 2, 0.01 ** 2])
    z = np.array([0.32, -0.19])
    S = H @ P @ H.T + R
    r = z - H @ x
    expected = np.exp(-0.5 * r @ np.linalg.solve(S, r)) / (2.0 * np.pi * np.sqrt(np.linalg.det(S)))
    _, _, lik = imm.kf_update_with_likelihood(x, P, z, H, R)
    _, _, log_lik = imm.kf_update_loglik(x, P, z, H, R)
    assert lik == pytest.approx(expected, rel=1e-12)
    assert log_lik == pytest.approx(np.log(expected), rel=1e-12)
    # scalar case: 1D normal density
    _, _, l1 = imm.kf_update_with_likelihood(np.array([0.0, 0.0]), np.diag([1.0, 1.0]), np.array([0.5]),
                                             np.array([[1.0, 0.0]]), np.array([[0.25]]))
    assert l1 == pytest.approx(np.exp(-0.5 * 0.25 / 1.25) / np.sqrt(2.0 * np.pi * 1.25), rel=1e-12)


def test_batched_update_matches_one_at_a_time(imm):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 2, 2))
    A = rng.normal(size=(5, 2, 2, 2))
    P = A @ np.swapaxes(A, -1, -2) + np.eye(2)
    z = rng.normal(size=(5, 1, 2))
    H, R = np.eye(2), np.diag([0.04, 0.01])
    xb, Pb, lb = imm.kf_update_loglik(x, P, z, H, R)
    for i in range(5):
        for j in range(2):
            x1, P1, l1 = imm.kf_update_loglik(x[i, j], P[i, j], z[i, 0], H, R)
            np.testing.assert_allclose(xb[i, j], x1, atol=1e-12)
            np.testing.assert_allclose(Pb[i, j], P1, atol=1e-12)
            assert lb[i, j] == pytest.approx(l1, abs=1e-12)


# --- mode dynamics and Jacobians ---

def _fd_jacobian(f, x, eps=1e-6):
    cols = [(f(x + eps * e) - f(x - eps * e)) / (2 * eps) for e in np.eye(len(x))]
    return np.stack(cols, axis=1)


@pytest.mark.parametrize("integrates", [False, True])
def test_mode_predict_jacobians_match_finite_differences(imm, integrates):
    dt, q, sig = 0.05, 0.5, 0.01
    x0 = np.array([[[0.3, 0.1], [0.3, 0.1]]])
    P0 = np.array([[np.diag([0.04, 0.09])] * 2])
    for mode in (0, 1):
        f = lambda v, mode=mode: imm.predict_modes(np.stack([v, v])[None], P0, dt, q, sig, integrates)[0][0, mode]  # noqa: E731
        J = _fd_jacobian(f, x0[0, mode])
        P_zero = imm.predict_modes(x0, np.zeros_like(P0), dt, q, sig, integrates)[1][0, mode]
        P_pred = imm.predict_modes(x0, P0, dt, q, sig, integrates)[1][0, mode]
        np.testing.assert_allclose(P_pred - P_zero, J @ P0[0, mode] @ J.T, atol=1e-10)


def test_still_mode_belief_is_v_zero_with_the_zupt_noise(imm):
    x0 = np.array([[[0.5, 0.2], [0.5, 0.2]]])
    P0 = np.array([[np.diag([0.04, 0.09])] * 2])
    x_p, P_p = imm.predict_modes(x0, P0, 0.05, 0.5, 0.01)
    np.testing.assert_allclose(x_p[0, 0], [0.5, 0.0], atol=1e-15)
    np.testing.assert_allclose(P_p[0, 0], np.diag([0.04, 0.01 ** 2]), atol=1e-15)
    x_i, _ = imm.predict_modes(x0, P0, 0.05, 0.5, 0.01, still_integrates=True)
    np.testing.assert_allclose(x_i[0, 0], [0.5 + 0.05 * 0.2, 0.0], atol=1e-15)


def test_moving_mode_is_the_parent_constant_velocity_predict(imm, base):
    x0 = np.array([[[0.5, 0.2], [0.5, 0.2]]])
    P0 = np.array([[np.diag([0.04, 0.09])] * 2])
    x_p, P_p = imm.predict_modes(x0, P0, 0.05, 0.5, 0.01)
    xr, Pr = base.predict(x0[0, 1], P0[0, 1], 0.05, 0.5)
    np.testing.assert_allclose(x_p[0, 1], xr, atol=1e-15)
    np.testing.assert_allclose(P_p[0, 1], Pr, atol=1e-15)


def test_measurement_matrices_match_finite_differences(imm):
    # the measurement is linear: z = H x; its finite-difference Jacobian is the H the IMM stacks
    meas = lambda x: np.array([x[0], x[1]])  # noqa: E731  (position, then d / dt = velocity)
    np.testing.assert_allclose(_fd_jacobian(meas, np.array([0.2, 0.1])), np.eye(2), atol=1e-9)


# --- transition design ---

def test_sojourn_times_match_the_simulated_truth(imm):
    a = _args(duration=4000.0, seed=3)
    s = imm.make_setup(a, a.sensor_noise_std)
    moving = s.x_true[:, 1] != 0.0

    def mean_run(flag):
        edges = np.diff(np.concatenate([[False], flag, [False]]).astype(int))
        return np.mean(np.where(edges == -1)[0] - np.where(edges == 1)[0]) * a.dt

    T_s, T_m = imm.mean_sojourn_times(a.t_anchor, a.t_extend, a.t_slip, a.p_slip)
    assert (T_s, T_m) == pytest.approx((0.7 / 1.5, 1.3 / 1.5))
    assert mean_run(~moving) == pytest.approx(T_s, rel=0.12)
    assert mean_run(moving) == pytest.approx(T_m, rel=0.12)


def test_const_transition_matrix_and_stationary_start(imm):
    a, s, _ = _data(imm)
    trans, mu0, info = imm.design_transitions("const", s.is_anchor, a.dt, a.t_anchor, a.t_extend, a.t_slip, a.p_slip)
    np.testing.assert_allclose(trans[5], [[1 - 0.05 / (0.7 / 1.5), 0.05 / (0.7 / 1.5)],
                                          [0.05 / (1.3 / 1.5), 1 - 0.05 / (1.3 / 1.5)]], atol=1e-12)
    np.testing.assert_allclose(trans.sum(axis=2), 1.0, atol=1e-12)
    np.testing.assert_allclose(mu0 @ trans[1], mu0, atol=1e-12)
    assert mu0[0] == pytest.approx(0.35)


def test_schedule_matrices_follow_the_schedule(imm):
    a, s, _ = _data(imm)
    trans, mu0, _ = imm.design_transitions("schedule_next", s.is_anchor, a.dt, a.t_anchor, a.t_extend, a.t_slip, a.p_slip)
    ext, anc = np.where(~s.is_anchor)[0][3], np.where(s.is_anchor)[0][3]
    np.testing.assert_allclose(trans[ext], [[0.05, 0.95], [0.05, 0.95]], atol=1e-12)
    np.testing.assert_allclose(trans[anc], [[1 - 0.05 * 0.5 / 0.7, 0.05 * 0.5 / 0.7], [0.05 / 0.6, 1 - 0.05 / 0.6]],
                               atol=1e-12)
    assert mu0[0] == pytest.approx(0.7)


def test_default_schedule_differs_from_next_tick_rule_only_on_the_first_anchor_tick_after_extend(imm):
    a, s, _ = _data(imm)
    args = (s.is_anchor, a.dt, a.t_anchor, a.t_extend, a.t_slip, a.p_slip)
    nxt = imm.design_transitions("schedule_next", *args)[0]
    default = imm.design_transitions("schedule", *args)[0]
    first = np.where(np.concatenate([[False], s.is_anchor[1:] & ~s.is_anchor[:-1]]))[0]
    assert len(first) == 10  # ticks 40, 80, ..., 400 (the last tick, t = 20 s, starts a new anchor phase)
    assert set(np.where(np.any(nxt != default, axis=(1, 2)))[0]) == set(first)
    np.testing.assert_allclose(default[first], np.tile([[1.0, 0.0], [1.0, 0.0]], (len(first), 1, 1)))
    assert nxt[first[0], 1, 0] == pytest.approx(0.05 / 0.6)  # leaves the moving mode with probability 0.9167


def test_scale_multiplies_switch_probabilities_and_clips(imm):
    a, s, _ = _data(imm)
    args = (s.is_anchor, a.dt, a.t_anchor, a.t_extend, a.t_slip, a.p_slip)
    p1 = imm.design_transitions("const", *args)[0][1]
    p5 = imm.design_transitions("const", *args, scale=5.0)[0][1]
    assert p5[0, 1] == pytest.approx(5 * p1[0, 1]) and p5[1, 0] == pytest.approx(5 * p1[1, 0])
    assert imm.design_transitions("const", *args, scale=10.0)[0][1][0, 1] == 1.0  # 10 * 0.107 clipped to 1
    huge = imm.design_transitions("const", *args, scale=1e6)[0][1]
    assert np.all(huge >= 0) and np.all(huge <= 1)
    only_ext = imm.design_transitions("schedule", *args, scale=1.0, scale_extend=10.0)[0]
    ext = np.where(~s.is_anchor)[0][3]
    assert only_ext[ext][0, 0] == pytest.approx(0.5)
    anc = np.where(s.is_anchor)[0][3]
    np.testing.assert_allclose(only_ext[anc], imm.design_transitions("schedule", *args)[0][anc])


def test_unknown_transition_kind_is_rejected(imm):
    a, s, _ = _data(imm)
    with pytest.raises(ValueError):
        imm.design_transitions("bogus", s.is_anchor, a.dt, a.t_anchor, a.t_extend, a.t_slip, a.p_slip)


# --- IMM cycle ---

def test_mixing_with_an_asymmetric_matrix_equals_eq_12_to_14_written_literally(imm):
    # asymmetric on purpose: row- and column-stochastic conventions give different answers
    Pi = np.array([[0.9, 0.1], [0.3, 0.7]])          # row = previous mode (this code)
    H = Pi.T                                         # paper: H_ij = P(theta_t = i | theta_{t-1} = j), columns sum to 1
    np.testing.assert_allclose(H.sum(axis=0), 1.0)
    rng = np.random.default_rng(2)
    mu = np.array([[0.8, 0.2], [0.3, 0.7]])
    xs = rng.normal(size=(2, 2, 2))
    A = rng.normal(size=(2, 2, 2, 2))
    Ps = A @ np.swapaxes(A, -1, -2) + 0.1 * np.eye(2)
    c, x0, P0 = imm.mix_modes(mu, Pi, xs, Ps)
    for b in range(2):
        for i in range(2):
            mu_bar = sum(H[i, j] * mu[b, j] for j in range(2))                                   # (12)
            x_mix = sum(H[i, j] * mu[b, j] * xs[b, j] for j in range(2)) / mu_bar                # (13)
            R_mix = sum(H[i, j] * mu[b, j] * (Ps[b, j] + np.outer(xs[b, j] - x_mix, xs[b, j] - x_mix))
                        for j in range(2)) / mu_bar                                              # (14)
            assert c[b, i] == pytest.approx(mu_bar, abs=1e-14)
            np.testing.assert_allclose(x0[b, i], x_mix, atol=1e-12)
            np.testing.assert_allclose(P0[b, i], R_mix, atol=1e-12)
    # the transposed (wrong-convention) matrix would give a different mixing
    c_wrong = imm.mix_modes(mu, H, xs, Ps)[0]
    assert not np.allclose(c, c_wrong)



def _reference_imm(x_init, P_init, z, d, dt, q, R_pos, R_vel, sig, trans, mu0, use_pad):
    """Plain-loop IMM written literally from Blom & Bar-Shalom (1988) eq. (12)-(19), in the paper's notation:
    H_ij = P(theta_t = i | theta_{t-1} = j) (columns sum to 1), so H = trans[k].T; eq. (15) without the
    (2 pi)^(-m/2) factor. Independent of run_imm's einsums and of the row-stochastic convention."""
    F = [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[1.0, dt], [0.0, 1.0]])]
    Q = [np.diag([0.0, sig ** 2]), np.diag([(0.5 * q * dt * dt) ** 2, (q * dt) ** 2])]
    h = np.eye(2) if use_pad else np.array([[1.0, 0.0]])
    R = np.diag([R_pos, R_vel]) if use_pad else np.array([[R_pos]])
    xh = [x_init.copy(), x_init.copy()]
    Rh = [P_init.copy(), P_init.copy()]
    mu = mu0.copy()
    out = [(x_init.copy(), P_init.copy(), mu.copy())]
    for k in range(len(z) - 1):
        Hm = trans[k + 1].T
        mu_bar = [sum(Hm[i, j] * mu[j] for j in range(2)) for i in range(2)]                       # (12)
        unnorm, xn, Pn = [], [], []
        for i in range(2):
            if mu_bar[i] == 0.0:                                                                   # (12): prune hypothesis i
                unnorm.append(0.0)
                xn.append(xh[i])
                Pn.append(Rh[i])
                continue
            x_mix = sum(Hm[i, j] * mu[j] * xh[j] for j in range(2)) / mu_bar[i]                    # (13)
            R_mix = sum(Hm[i, j] * mu[j] * (Rh[j] + np.outer(xh[j] - x_mix, xh[j] - x_mix))
                        for j in range(2)) / mu_bar[i]                                             # (14)
            xp, Pp = F[i] @ x_mix, F[i] @ R_mix @ F[i].T + Q[i]
            y = np.array([z[k + 1], d[k + 1] / dt]) if use_pad else np.array([z[k + 1]])
            s_i = y - h @ xp                                                                       # (16)
            Q_i = h @ Pp @ h.T + R                                                                 # (17)
            Kg = Pp @ h.T @ np.linalg.inv(Q_i)
            xn.append(xp + Kg @ s_i)
            Pn.append((np.eye(2) - Kg @ h) @ Pp)
            unnorm.append(mu_bar[i] * np.linalg.det(Q_i) ** -0.5 * np.exp(-0.5 * s_i @ np.linalg.solve(Q_i, s_i)))  # (15)
        mu = np.array(unnorm) / sum(unnorm)
        xh, Rh = xn, Pn
        xc = sum(mu[i] * xh[i] for i in range(2))                                                  # (18)
        Pc = sum(mu[i] * (Rh[i] + np.outer(xh[i] - xc, xh[i] - xc)) for i in range(2))             # (19)
        out.append((xc, Pc, mu.copy()))
    return out


@pytest.mark.parametrize("use_pad", [True, False])
@pytest.mark.parametrize("kind", ["const", "schedule", "schedule_next"])
def test_run_imm_matches_a_plain_loop_reference(imm, kind, use_pad):
    a, s, (R_pos, _, R_vel) = _data(imm, duration=6.0)
    trans, mu0, _ = imm.design_transitions(kind, s.is_anchor, a.dt, a.t_anchor, a.t_extend, a.t_slip, a.p_slip)
    x, P, mu = imm.run_imm(s.x_init, s.P_init, s.z, s.d, a.dt, 0.5, R_pos, R_vel, 0.01, trans, mu0, use_pad)
    ref = _reference_imm(s.x_init, s.P_init, s.z, s.d, a.dt, 0.5, R_pos, R_vel, 0.01, trans, mu0, use_pad)
    for k, (xr, Pr, mur) in enumerate(ref):
        np.testing.assert_allclose(x[k], xr, atol=1e-10)
        np.testing.assert_allclose(P[k], Pr, atol=1e-10)
        np.testing.assert_allclose(mu[k], mur, atol=1e-10)


def test_mode_probabilities_are_valid_and_covariances_symmetric_positive(imm):
    a, s, (R_pos, _, R_vel) = _data(imm)
    trans, mu0, _ = imm.design_transitions("schedule", s.is_anchor, a.dt, a.t_anchor, a.t_extend, a.t_slip, a.p_slip)
    x, P, mu = imm.run_imm(s.x_init, s.P_init, s.z, s.d, a.dt, 0.5, R_pos, R_vel, 0.01, trans, mu0, True)
    assert np.all(mu >= 0) and np.all(mu <= 1)
    np.testing.assert_allclose(mu.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(P, np.swapaxes(P, -1, -2), atol=1e-12)
    assert np.all(np.linalg.eigvalsh(0.5 * (P + np.swapaxes(P, -1, -2))) > 0)


def test_combined_covariance_includes_the_spread_of_the_means(imm):
    # two modes with very different means and mu = (0.5, 0.5): P_combined = mean(P) + spread term
    xs = np.array([[0.0, 0.0], [0.0, 0.1]])
    Ps = np.array([np.eye(2) * 0.01, np.eye(2) * 0.01])
    mu = np.array([0.5, 0.5])
    xc = mu @ xs
    Pc = sum(mu[j] * (Ps[j] + np.outer(xs[j] - xc, xs[j] - xc)) for j in range(2))
    assert Pc[1, 1] == pytest.approx(0.01 + 0.05 ** 2)
    assert Pc[0, 0] == pytest.approx(0.01)


def test_always_moving_imm_equals_sensor_fused_and_never(imm, gate):
    a, s, (R_pos, R_zupt, R_vel) = _data(imm)
    trans, mu0 = _always_moving(len(s.z))
    n_steps = len(s.z) - 1
    for use_pad, mode in ((True, "sensor_fused"), (False, "never")):
        x, P, mu = imm.run_imm(s.x_init, s.P_init, s.z, s.d, a.dt, 0.5, R_pos, R_vel, 0.01, trans, mu0, use_pad)
        xe, Pe = gate.run_ekf(s.x_init, s.P_init, s.z, s.d, a.dt, n_steps, 0.5, R_pos, R_zupt, R_vel,
                              3 * a.sensor_noise_std, s.is_anchor, mode)
        np.testing.assert_allclose(x, xe, atol=1e-9)
        np.testing.assert_allclose(P, np.array(Pe), atol=1e-9)
        np.testing.assert_allclose(mu[:, 1], 1.0)


def test_zero_noise_recovers_the_truth_with_a_perfect_pad_reading(imm):
    a, s, (R_pos, _, R_vel) = _data(imm, pos_noise_std=1e-6, sensor_noise_std=1e-8, p_slip=0.5)
    R_pos, R_vel = 1e-12, (1e-8 / a.dt) ** 2
    trans, mu0, _ = imm.design_transitions("schedule", s.is_anchor, a.dt, a.t_anchor, a.t_extend, a.t_slip, a.p_slip)
    x, _, _ = imm.run_imm(s.x_true[0], np.diag([1e-4, 1e-4]), s.z, s.d, a.dt, 0.5, R_pos, R_vel, 0.01, trans, mu0, True)
    assert np.max(np.abs(x[5:, 0] - s.x_true[5:, 0])) < 5e-4
    assert np.max(np.abs(x[5:, 1] - s.x_true[5:, 1])) < 5e-3


def test_batched_run_equals_single_runs(imm):
    a, s, (R_pos, _, R_vel) = _data(imm, duration=5.0)
    trans, mu0, _ = imm.design_transitions("schedule", s.is_anchor, a.dt, a.t_anchor, a.t_extend, a.t_slip, a.p_slip)
    x0, zz, dd = imm.draw_trials(s.x_true, a.dt, R_pos, a.sensor_noise_std, np.array([0.05, 0.05]), 3,
                                 np.random.default_rng(4))
    xb, Pb, mub = imm.run_imm(x0, s.P_init, zz, dd, a.dt, 0.5, R_pos, R_vel, 0.01, trans, mu0, True)
    for i in range(3):
        x1, P1, mu1 = imm.run_imm(x0[i], s.P_init, zz[i], dd[i], a.dt, 0.5, R_pos, R_vel, 0.01, trans, mu0, True)
        np.testing.assert_allclose(xb[i], x1, atol=1e-12)
        np.testing.assert_allclose(Pb[i], P1, atol=1e-12)
        np.testing.assert_allclose(mub[i], mu1, atol=1e-12)


def test_unreachable_mode_does_not_produce_nan(imm):
    a, s, (R_pos, _, R_vel) = _data(imm, duration=3.0)
    trans, mu0 = _always_moving(len(s.z))
    x, P, mu = imm.run_imm(s.x_init, s.P_init, s.z, s.d, a.dt, 0.5, R_pos, R_vel, 0.01, trans, mu0, True)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(P)) and np.all(np.isfinite(mu))


def test_a_surprising_measurement_does_not_underflow_the_mode_probability(imm):
    # velocity reading 100 sigma away from the still-mode belief: exp(-5000) is 0 in float64
    a, s, _ = _data(imm, duration=2.0)
    trans, mu0, _ = imm.design_transitions("const", s.is_anchor, a.dt, a.t_anchor, a.t_extend, a.t_slip, a.p_slip)
    d = np.full_like(s.d, 100 * 0.01 * a.dt)
    _, _, mu = imm.run_imm(np.zeros(2), np.diag([1e-4, 1e-4]), np.zeros_like(s.z), d, a.dt, 0.5, 4e-4, 1e-4,
                           0.01, trans, mu0, True)
    assert np.all(np.isfinite(mu)) and mu[-1, 1] > 0.99


# --- reuse of the parent script: same truth, same Monte Carlo draws ---

def test_seed_zero_gives_the_parents_two_slip_pulses(imm):
    a, s, _ = _data(imm)
    assert len(imm.slip_pulses(s.is_slip)) == 2
    assert int(s.is_slip.sum()) == 24 and int(s.is_clean.sum()) == 161 and int(s.is_cruise.sum()) == 119


def test_setup_matches_the_parents_main_rng_order(imm, gate):
    a, s, _ = _data(imm)
    rng = np.random.default_rng(0)
    x_true, z, d, *_ = gate.generate_ground_truth_and_data(20.0, 0.05, 1.0, 1.0, 0.2, 0.1, 0.02, 0.0005, 0.5, 0.08,
                                                           0.6, rng)
    x_init = x_true[0] + rng.normal(0.0, 1.0, 2) * np.array([0.05, 0.05])
    np.testing.assert_array_equal(s.z, z)
    np.testing.assert_array_equal(s.d, d)
    np.testing.assert_array_equal(s.x_init, x_init)
    assert s.rng.random() == rng.random()


def test_sensor_noise_changes_only_the_pad_readings(imm):
    a = _args()
    s1, s2 = imm.make_setup(a, 0.0005), imm.make_setup(a, 0.005)
    np.testing.assert_array_equal(s1.z, s2.z)
    np.testing.assert_array_equal(s1.is_slip, s2.is_slip)
    np.testing.assert_array_equal(s1.x_init, s2.x_init)
    assert not np.array_equal(s1.d, s2.d)


def test_evaluate_reproduces_the_parent_monte_carlo(imm, gate):
    a = _args(n_mc_trials=6)
    s = imm.make_setup(a, a.sensor_noise_std)
    res = imm.evaluate(a, s, {"never": imm.policy_spec("never"), "sensor_fused": imm.policy_spec("sensor_fused")},
                       a.sensor_noise_std)
    R_pos, R_zupt, R_vel = a.pos_noise_std ** 2, a.zupt_noise_std ** 2, (a.sensor_noise_std / a.dt) ** 2
    ref = gate.run_monte_carlo_consistency(
        s.x_true, a.dt, 0.5, R_pos, R_zupt, R_vel, 3 * a.sensor_noise_std, s.is_anchor, 0.05, 0.05,
        a.sensor_noise_std, 6, copy.deepcopy(s.rng), modes=("never", "sensor_fused"))
    for m in ("never", "sensor_fused"):
        np.testing.assert_allclose(res[m]["nees"], ref[m], atol=1e-9)


def test_batched_nees_equals_utils_nees(imm):
    rng = np.random.default_rng(1)
    x_true = rng.normal(size=(7, 2))
    x_est = x_true + rng.normal(size=(3, 7, 2)) * 0.1
    A = rng.normal(size=(3, 7, 2, 2))
    P = A @ np.swapaxes(A, -1, -2) + np.eye(2)
    got = imm.batched_nees(x_true, x_est, P)
    for i in range(3):
        for k in range(7):
            assert got[i, k] == pytest.approx(_nees(x_true[k] - x_est[i, k], P[i, k]), abs=1e-12)


def test_invariant_quantity_is_identical_across_the_sensor_noise_sweep(imm):
    # imm_pos ignores the pad sensor, so its NEES cannot change when only the sensor noise changes
    a = _args(n_mc_trials=5, duration=6.0)
    out = []
    for sd in (0.0005, 0.005):
        s = imm.make_setup(a, sd)
        out.append(imm.evaluate(a, s, {"imm_pos": imm.imm_spec("const", False)}, sd)["imm_pos"]["nees"])
    np.testing.assert_array_equal(out[0], out[1])


def test_evaluate_mse_is_the_monte_carlo_mean_squared_error(imm):
    a = _args(n_mc_trials=4, duration=4.0)
    s = imm.make_setup(a, a.sensor_noise_std)
    res = imm.evaluate(a, s, {"imm": imm.imm_spec("const", True), "never": imm.policy_spec("never")},
                       a.sensor_noise_std)
    R_pos, R_vel = a.pos_noise_std ** 2, (a.sensor_noise_std / a.dt) ** 2
    x0, zz, dd = imm.draw_trials(s.x_true, a.dt, R_pos, a.sensor_noise_std, np.array([0.05, 0.05]), 4,
                                 copy.deepcopy(s.rng))
    trans, mu0, _ = imm.design_transitions("const", s.is_anchor, a.dt, a.t_anchor, a.t_extend, a.t_slip, a.p_slip)
    x, _, _ = imm.run_imm(x0, s.P_init, zz, dd, a.dt, 0.5, R_pos, R_vel, 0.01, trans, mu0, True)
    np.testing.assert_allclose(res["imm"]["mse"], ((x - s.x_true) ** 2).mean(axis=0), atol=1e-15)
    assert res["never"]["mse"].shape == (len(s.x_true), 2)


# --- detection statistics ---

def test_detection_stats_on_hand_built_flags(imm):
    is_slip = np.zeros(12, bool)
    is_slip[3:7] = True
    is_clean = np.zeros(12, bool)
    is_clean[8:12] = True
    flag = np.zeros((3, 12), bool)
    flag[0, 3:7] = True            # already flagged on the pulse's first tick
    flag[1, 5:7] = True            # detected 2 ticks after the first slip tick
    flag[2, :] = False             # missed
    flag[2, 9] = True              # one false alarm on a clean tick
    st = imm.detection_stats(flag, is_slip, is_clean, 0.05)
    assert st["hit"] == pytest.approx(6 / 12)
    assert st["false_alarm"] == pytest.approx(1 / 12)
    assert st["pre_flagged"] == pytest.approx(1 / 3)
    assert st["delay_s"] == pytest.approx(0.1)
    assert st["missed"] == pytest.approx(1 / 2)


def test_slip_pulses_are_the_contiguous_slip_runs(imm):
    f = np.zeros(10, bool)
    f[2:4] = True
    f[7:10] = True
    assert imm.slip_pulses(f) == [(2, 4), (7, 10)]


# --- the finding (regression; written after the numbers were in) ---

def test_imm_with_the_pad_reading_is_consistent_in_slip_where_the_schedule_is_not(imm):
    a = _args(n_mc_trials=60)
    s = imm.make_setup(a, a.sensor_noise_std)
    res = imm.evaluate(a, s, {"schedule": imm.policy_spec("schedule"), "imm_schedule": imm.imm_spec("schedule", True),
                              "sensor_fused": imm.policy_spec("sensor_fused")}, a.sensor_noise_std)
    sl = s.is_slip
    assert np.mean(res["schedule"]["nees"][sl]) > 10.0
    assert np.mean(res["imm_schedule"]["nees"][sl]) < 3.5
    assert res["imm_schedule"]["detect"]["hit"] > 0.8
    assert res["imm_schedule"]["rmse_vel"] < 1.5 * res["sensor_fused"]["rmse_vel"]


def test_position_only_imm_cannot_see_the_slip_and_the_next_tick_rule_hides_that_with_leftover_motion(imm):
    a = _args(n_mc_trials=60)
    s = imm.make_setup(a, a.sensor_noise_std)
    res = imm.evaluate(a, s, {"default": imm.imm_spec("schedule", False), "next": imm.imm_spec("schedule_next", False)},
                       a.sensor_noise_std)
    assert res["next"]["detect"]["pre_flagged"] > 0.5      # mostly leftover motion from the extend phase
    assert res["default"]["detect"]["false_alarm"] < 0.05
    assert res["default"]["detect"]["hit"] < 0.3           # a 4 mm per tick slip is buried in 20 mm noise
    assert np.mean(res["default"]["nees"][s.is_slip]) > 2.0 * np.mean(res["next"]["nees"][s.is_slip])


def test_cli_seeds_table_without_seeds_is_an_error_not_a_full_run(imm, monkeypatch):
    monkeypatch.setattr("sys.argv", ["inchworm_slip_imm.py", "--seeds-table"])
    with pytest.raises(SystemExit):
        imm.main()
