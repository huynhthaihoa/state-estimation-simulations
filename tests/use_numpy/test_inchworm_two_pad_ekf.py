import numpy as np
import pytest

DT = 0.05
T_PHASE = 1.0
T_RAMP = 0.2
T_SLIP = 0.6
L_MIN = 0.15


@pytest.fixture
def two(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "inchworm_two_pad_ekf")


def _noise(micro_slip=0.01, sensor=0.0005, gate_sigmas=3.0):
    """Filter noise dict, built the same way main() does with its default arguments."""
    still_std = np.hypot(sensor, micro_slip * DT)
    return dict(micro_slip=micro_slip, free=0.1, sensor=sensor, move=0.02, gyro=0.02, closure=0.001,
                closure_lat=0.001, d_thr=gate_sigmas * still_std)


def _sim(m, duration=10.0, p_slip=0.5, micro_slip_std=0.01, steer=0.3, seed=0, episodes=None):
    rng = np.random.default_rng(seed)
    n_steps = int(round(duration / DT))
    if episodes is None:
        episodes = m.draw_slip_episodes(int(np.ceil(duration / T_PHASE)), T_PHASE, T_SLIP, p_slip, rng)
    out = m.simulate(n_steps, DT, T_PHASE, T_RAMP, 0.1, L_MIN, steer, episodes, 0.08, micro_slip_std,
                     0.02, 0.0005, 0.001, 0.001, rng)
    return out, episodes


def _monte_carlo_nees(m, p_slip, policies, n_trials=40, duration=6.0, seed=0):
    """Mean NEES over trials and ticks per policy, same trial setup as main()."""
    rng = np.random.default_rng(seed)
    n_steps = int(round(duration / DT))
    episodes = m.draw_slip_episodes(int(np.ceil(duration / T_PHASE)), T_PHASE, T_SLIP, p_slip, rng)
    noise = _noise()
    init_std = np.array([0.005] * 4 + [0.02])
    P_init = np.diag(init_std ** 2)
    sums = {p: 0.0 for p in policies}
    for _ in range(n_trials):
        x_true, gyro, d, z, back = m.simulate(n_steps, DT, T_PHASE, T_RAMP, 0.1, L_MIN, 0.3, episodes, 0.08,
                                              0.01, 0.02, 0.0005, 0.001, 0.001, rng)
        x_init = x_true[0] + rng.normal(0.0, 1.0, m.STATE_DIM) * init_std
        for p in policies:
            x_est, P_est = m.run_ekf(x_init, P_init, gyro, d, z, back, DT, p, noise)
            err = m.state_error(x_true, x_est)
            sums[p] += np.mean([m.nees(err[k], P_est[k]) for k in range(n_steps + 1)])
    return {p: s / n_trials for p, s in sums.items()}


# --- slip schedule ---

def test_slip_episodes_deterministic_and_inside_their_phase(two):
    a = two.draw_slip_episodes(20, T_PHASE, T_SLIP, 0.5, np.random.default_rng(3))
    b = two.draw_slip_episodes(20, T_PHASE, T_SLIP, 0.5, np.random.default_rng(3))
    assert a == b and len(a) > 0
    for j, start, end in a:
        assert end - start == pytest.approx(T_SLIP)
        assert j * T_PHASE <= start and end <= (j + 1) * T_PHASE + 1e-12


def test_slip_probability_zero_and_one(two):
    assert two.draw_slip_episodes(10, T_PHASE, T_SLIP, 0.0, np.random.default_rng(0)) == []
    eps = two.draw_slip_episodes(10, T_PHASE, T_SLIP, 1.0, np.random.default_rng(0))
    assert [j for j, _, _ in eps] == list(range(10))


# --- truth ---

def test_truth_satisfies_the_kinematic_closure(two):
    (x, *_), _ = _sim(two)
    rel = x[:, two.F] - x[:, two.B]
    length = np.linalg.norm(rel, axis=1)
    th = x[:, two.TH]
    np.testing.assert_allclose(rel, length[:, None] * np.column_stack([np.cos(th), np.sin(th)]), atol=1e-12)
    local = np.einsum("kji,kj->ki", np.array([two.rot(t) for t in th]), rel)
    assert np.all(local[:, 0] > 0.0)
    np.testing.assert_allclose(local[:, 1], 0.0, atol=1e-12)


def test_heading_changes_only_while_the_back_pad_is_anchored(two):
    (x, _, _, _, back), _ = _sim(two)
    dth = np.diff(x[:, two.TH])
    assert np.all(dth[~back] == 0.0)
    assert np.any(dth[back] > 0.0)
    assert np.all(dth[back] >= 0.0)  # steer > 0 and dl >= 0 in extend


def test_anchored_pad_is_still_without_slip_and_micro_slip(two):
    (x, _, _, _, back), eps = _sim(two, p_slip=0.0, micro_slip_std=0.0)
    assert eps == []
    assert back[:20].all() and not back[20:40].any()
    step = np.diff(x, axis=0)
    assert np.all(step[back][:, two.B] == 0.0)
    assert np.all(step[~back][:, two.F] == 0.0)
    assert np.any(step[back][:, two.F] != 0.0)  # the moving pad does move


def test_body_length_returns_to_l_min_after_each_full_cycle(two):
    (x, *_), _ = _sim(two, duration=10.0)
    length = np.linalg.norm(x[:, two.F] - x[:, two.B], axis=1)
    cycle = int(round(2 * T_PHASE / DT))
    np.testing.assert_allclose(length[::cycle], L_MIN, atol=1e-12)
    assert length.max() > L_MIN + 0.02


# --- predict ---

def test_predict_rejects_unknown_policy(two):
    with pytest.raises(ValueError):
        two.predict(np.zeros(5), np.eye(5), DT, 0.0, np.zeros(2), True, "bogus", _noise())


def test_pad_fused_moves_the_anchored_pad_by_the_rotated_reading(two):
    x = np.array([1.0, 2.0, 1.2, 2.1, 0.7])
    d_k = np.array([0.002, -0.001])
    for back in (True, False):
        x_new, _ = two.predict(x, np.eye(5), DT, 0.0, d_k, back, "pad_fused", _noise())
        anch, mov = (two.B, two.F) if back else (two.F, two.B)
        np.testing.assert_allclose(x_new[anch], x[anch] + two.rot(0.7) @ d_k)
        np.testing.assert_allclose(x_new[mov], x[mov])


def test_theta_integrates_the_gyro(two):
    x = np.array([0.0, 0.0, 0.15, 0.0, 0.3])
    x_new, _ = two.predict(x, np.eye(5), DT, 0.4, np.zeros(2), True, "schedule", _noise())
    assert x_new[two.TH] == pytest.approx(0.3 + 0.4 * DT)


def test_schedule_and_sensor_uses_loose_variance_only_above_the_threshold(two):
    noise = _noise()
    still_var, free_var = (noise["micro_slip"] * DT) ** 2, (noise["free"] * DT) ** 2
    x = np.zeros(5)
    for d_norm, var in ((0.5 * noise["d_thr"], still_var), (noise["d_thr"], still_var),
                        (1.01 * noise["d_thr"], free_var), (5.0 * noise["d_thr"], free_var)):
        _, P = two.predict(x, np.zeros((5, 5)), DT, 0.0, np.array([d_norm, 0.0]), True, "schedule_and_sensor", noise)
        np.testing.assert_allclose(P[two.B, two.B], var * np.eye(2))


# --- update_closure ---

def _closure_h(m, x):
    return m.rot(x[m.TH]).T @ (x[m.F] - x[m.B])


def _fd_jacobian(m, x, eps=1e-6):
    H = np.zeros((2, 5))
    for i in range(5):
        e = np.zeros(5)
        e[i] = eps
        H[:, i] = (_closure_h(m, x + e) - _closure_h(m, x - e)) / (2 * eps)
    return H


def test_update_closure_matches_the_finite_difference_jacobian(two):
    x = np.array([0.3, -0.2, 0.42, -0.05, 0.8])
    P = np.diag([0.01, 0.02, 0.015, 0.01, 0.05]) ** 2 + 1e-5 * np.ones((5, 5))
    R = np.diag([0.001, 0.002]) ** 2
    z = _closure_h(two, x) + np.array([0.003, -0.002])
    H = _fd_jacobian(two, x)
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x_new, P_new = two.update_closure(x, P, z, R)
    np.testing.assert_allclose(x_new, x + K @ (z - _closure_h(two, x)), atol=1e-8)
    np.testing.assert_allclose(P_new, P - K @ H @ P, atol=1e-10)


def test_update_closure_keeps_the_covariance_symmetric_positive_definite(two):
    x = np.array([0.0, 0.0, 0.15, 0.0, 0.2])
    P = np.diag([0.005, 0.005, 0.02, 0.02, 0.02]) ** 2
    R = np.diag([0.001, 0.001]) ** 2
    for _ in range(20):
        x, P = two.update_closure(x, P, np.array([0.15, 0.0]), R)
        np.testing.assert_allclose(P, P.T, atol=1e-15)
        assert np.linalg.eigvalsh(P).min() > 0.0


# --- consistency ---

def test_without_slip_schedule_and_pad_fused_are_consistent(two):
    nees = _monte_carlo_nees(two, 0.0, ("schedule", "pad_fused"))
    assert 3.5 < nees["schedule"] < 6.5
    assert 3.5 < nees["pad_fused"] < 6.5


def test_schedule_is_overconfident_under_slip_but_pad_fused_is_not(two):
    nees = _monte_carlo_nees(two, 1.0, ("schedule", "pad_fused"))
    assert nees["schedule"] > 3.0 * nees["pad_fused"]
    assert 3.5 < nees["pad_fused"] < 6.5
