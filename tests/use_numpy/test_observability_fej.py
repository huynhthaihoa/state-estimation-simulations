import numpy as np
import pytest

from state_estimation_simulations.use_numpy import observability_fej as _m


@pytest.fixture
def m():
    return _m


def _numeric_jac(fun, x, eps=1e-6):
    cols = []
    for i in range(len(x)):
        e = np.zeros(len(x))
        e[i] = eps
        cols.append((fun(x + e) - fun(x - e)) / (2 * eps))
    return np.stack(cols, axis=1)


def _state(K=2, seed=1):
    rng = np.random.default_rng(seed)
    return np.concatenate([[1.0, 2.0, 0.7], rng.normal(size=2 * K) * 3.0])


def _small(m, **kw):
    return m.Params(loops=1, n_landmarks=6, **kw)


def _data(m, p, seed=0):
    rng_map, rng_trial = np.random.default_rng(seed).spawn(2)
    scene = m.make_scene(p, rng_map)
    return scene, m.visibility(p, scene), m.draw_trial(p, scene, rng_trial)


# --- finite-difference checks of every Jacobian ---

def test_standard_propagation_jacobian_matches_finite_differences(m):
    p = _small(m)
    scene, vis, trial = _data(m, p)
    o = m.run_filter("standard", p, scene, vis, trial, record=True)
    for k in (1, 5, 20):
        d, w = trial["odom"][k - 1]
        x_prev = np.zeros(3 + 2 * p.n_landmarks)
        x_prev[:3] = o["x_est"][k - 1]
        Phi_num = _numeric_jac(lambda x: m.f_motion(x, d, w), x_prev)
        assert np.allclose(o["Phi"][k - 1], Phi_num, atol=1e-6)


def test_process_noise_jacobian_matches_finite_differences(m):
    x = _state()
    d, w = 1.0, 0.15

    def f_noise(wn):
        y = x.copy()
        y[:2] = x[:2] + m.rot(x[2]) @ (np.array([d, 0.0]) + wn[:2])
        y[2] = x[2] + w + wn[2]
        return y

    G_num = _numeric_jac(f_noise, np.zeros(3))
    G = m.noise_jacobian("standard", x[2], x[:2], x[3:])
    assert np.allclose(G, G_num, atol=1e-6)


def test_invariant_noise_jacobian_matches_finite_differences_at_the_propagated_estimate(m):
    # error xi of the true state after the noisy step, measured against the propagated estimate (Barrau & Bonnabel eq. 16)
    x = _state()
    d, w = 1.0, 0.15
    xp = m.f_motion(x, d, w)

    def xi_after_noise(wn):
        y = x.copy()
        y[:2] = x[:2] + m.rot(x[2]) @ (np.array([d, 0.0]) + wn[:2])
        y[2] = x[2] + w + wn[2]
        return m.xi_error(y, xp)

    G_num = _numeric_jac(xi_after_noise, np.zeros(3))
    G = m.noise_jacobian("iekf", x[2], xp[:2], x[3:])
    assert np.allclose(G, G_num, atol=1e-6)
    # with x^_{n-1|n-1} in the rotation column (the other parametrization) it would not match this process model
    G_prev = G.copy()
    G_prev[:2, 2] = -m.J2 @ x[:2]
    assert not np.allclose(G_prev, G_num, atol=1e-3)


@pytest.mark.parametrize("meas", ["position", "range_bearing"])
def test_standard_measurement_jacobian_matches_finite_differences(m, meas):
    K = 2
    x = _state(K)
    j = 1
    H = m.jac_H("standard", meas, x[2], x[:2], x[3 + 2 * j:5 + 2 * j], m.rot(x[2]).T @ (x[3 + 2 * j:5 + 2 * j] - x[:2]), j, 3 + 2 * K)
    H_num = _numeric_jac(lambda s: m.meas_state(meas, s, j), x)
    assert np.allclose(H, H_num, atol=1e-6)


@pytest.mark.parametrize("meas", ["position", "range_bearing"])
def test_invariant_measurement_jacobian_matches_finite_differences_in_xi(m, meas):
    K = 2
    x = _state(K)
    j = 0
    rel = m.rot(x[2]).T @ (x[3:5] - x[:2])
    H = m.jac_H("iekf", meas, x[2], x[:2], x[3:5], rel, j, 3 + 2 * K)
    H_num = _numeric_jac(lambda xi: m.meas_state(meas, m.group_act(xi, x), j), np.zeros(3 + 2 * K))
    assert np.allclose(H, H_num, atol=1e-6)


def test_group_action_commutes_with_the_motion_and_inverts_the_error(m):
    x = _state()
    xi = np.random.default_rng(2).normal(size=len(x)) * 0.05
    assert np.allclose(m.f_motion(m.group_act(xi, x), 1.0, 0.15), m.group_act(xi, m.f_motion(x, 1.0, 0.15)), atol=1e-12)
    # the first-order error xi of the group-acted state is xi up to second order
    small = xi * 1e-3
    assert np.allclose(m.xi_error(m.group_act(small, x), x), small, atol=1e-5)


def test_fej_measurement_rule_changes_only_the_rotation_column(m):
    K = 2
    x = _state(K)
    rel = m.rot(x[2]).T @ (x[3:5] - x[:2])
    H_latest = m.jac_H("fej", "position", x[2], x[:2], x[3:5], rel, 0, 3 + 2 * K)
    H_first = m.jac_H("fej", "position", x[2], x[:2], x[3:5] + np.array([0.3, -0.2]), rel, 0, 3 + 2 * K)
    diff = H_first - H_latest
    assert np.allclose(diff[:, [0, 1, 3, 4, 5, 6]], 0.0) and np.abs(diff[:, 2]).max() > 1e-3


# --- the observability matrix and its null space ---

def test_ideal_linearization_has_the_three_dimensional_null_space_with_the_expected_basis(m):
    res = m.observability_demo(m.Params(), seed=0, n_window=10)
    r = res["ideal"]
    assert r["null_dim"] == 3 and r["n_cols"] == 9
    assert r["rot_resid"] < 1e-12 and r["trans_resid"] < 1e-12


def test_standard_ekf_linearization_loses_the_rotation_direction(m):
    res = m.observability_demo(m.Params(), seed=0, n_window=10)
    s = res["standard"]
    assert s["null_dim"] == 2                                    # 2 of the true 3
    assert s["trans_resid"] < 1e-12                              # translations stay unobservable
    assert s["rot_resid"] > 1e-3                                 # the rotation is seen as observable
    assert s["sv"][-3] / s["sv"][0] > 1e-4                       # the third-smallest singular value is clearly nonzero


@pytest.mark.parametrize("meas", ["position", "range_bearing"])
def test_fej_and_invariant_ekf_keep_the_three_dimensional_null_space(m, meas):
    res = m.observability_demo(m.Params(meas=meas), seed=0, n_window=10)
    for kind in ("fej", "iekf"):
        assert res[kind]["null_dim"] == 3
        assert res[kind]["rot_resid"] < 1e-10 and res[kind]["trans_resid"] < 1e-10
    if meas == "position":
        assert res["standard"]["null_dim"] == 2


def test_each_fej_rule_alone_is_not_enough(m):
    res = m.observability_demo(m.Params(), seed=0, n_window=10)
    assert res["fej_phi"]["null_dim"] == 2 and res["fej_h"]["null_dim"] == 2


def test_null_space_report_on_a_known_matrix(m):
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    nd, s, V = m.null_space_report(M)
    assert nd == 1 and np.allclose(np.abs(V[:, -1]), [0, 0, 1])


def test_observability_matrix_stacks_h_times_cumulative_phi(m):
    Phis = [np.array([[1.0, 1.0], [0.0, 1.0]])] * 2
    Hs = [np.array([[1.0, 0.0]])] * 3
    M = m.observability_matrix(Phis, Hs)
    assert np.allclose(M, [[1, 0], [1, 1], [1, 2]])


# --- information gained along the unobservable rotation ---

def test_spurious_information_is_zero_for_fej_ideal_and_invariant_but_not_for_standard(m):
    p = _small(m)
    scene, vis, trial = _data(m, p)
    gain = {k: m.run_filter(k, p, scene, vis, trial)["info_gain"].sum() for k in m.KINDS}
    assert gain["standard"] > 1e2
    assert gain["fej"] < 1e-9 and gain["ideal"] < 1e-9 and gain["iekf"] == 0.0


# --- zero noise: everyone recovers the truth ---

@pytest.mark.parametrize("meas", ["position", "range_bearing"])
def test_all_filters_recover_the_truth_without_noise(m, meas):
    p = m.Params(loops=1, n_landmarks=6, wheel_noise_frac=1e-9, pos_noise_std=1e-6, range_noise_std=1e-6,
                 bearing_noise_std_deg=1e-6 * 180 / np.pi, meas=meas)
    scene, vis, trial = _data(m, p)
    for kind in m.ALL_KINDS:
        o = m.run_filter(kind, p, scene, vis, trial)
        assert np.abs(o["err"]).max() < 1e-3, kind
        seen = o["seen"]
        assert np.abs(o["lm_est"][seen] - scene["lms"][seen]).max() < 1e-2, kind


# --- data generation and the sweep rule ---

def test_scene_is_the_regular_polygon_with_the_stated_radius(m):
    p = m.Params()
    scene = m.make_scene(p, np.random.default_rng(0))
    assert scene["n_steps"] == 400
    r = np.linalg.norm(scene["poses"][:, :2] - scene["center"], axis=1)
    assert np.allclose(r, scene["radius"], atol=1e-9)
    assert abs(2 * scene["radius"] - 12.75) < 0.02                 # 1 m/s and 9 deg/s imply 12.7 m, not the paper's "15 m"


def test_changing_the_heading_noise_scale_leaves_other_draws_unchanged(m):
    p1, p2 = m.Params(loops=1, n_landmarks=8), m.Params(loops=1, n_landmarks=8, heading_noise_scale=2.0)
    scene, vis, _ = _data(m, p1)
    t1 = m.draw_trial(p1, scene, np.random.default_rng(5))
    t2 = m.draw_trial(p2, scene, np.random.default_rng(5))
    assert np.array_equal(t1["z"], t2["z"])
    assert np.allclose(t1["odom"][:, 0], t2["odom"][:, 0])
    # landmarks first seen at step 0 start from the exactly known pose, so the rotation noise cannot touch them
    f1 = m.run_filter("standard", p1, scene, vis, t1)["p_first"][vis[0]]
    f2 = m.run_filter("standard", p2, scene, vis, t2)["p_first"][vis[0]]
    assert np.array_equal(f1, f2) and f1.size > 0


# --- Monte Carlo regression (fixed seeds) ---

def test_monte_carlo_fej_invariant_and_ideal_are_near_consistent_and_standard_is_not_better(m):
    p = m.Params(loops=5)
    rng_map, rng_mc = np.random.default_rng(0).spawn(2)
    scene = m.make_scene(p, rng_map)
    vis = m.visibility(p, scene)
    res = m.run_monte_carlo(p, scene, vis, m.KINDS, 20, rng_mc)
    sm = m.summarize(res, 40, 20)
    for kind in ("fej", "iekf", "ideal"):
        assert 2.0 < sm[kind]["nees_all"] < 5.0, kind
        assert 1.5 < sm[kind]["nees_lm"] < 4.0, kind
    assert 2.0 < sm["standard"]["nees_all"] < 6.0
    assert sm["standard"]["info_mean"] > 1e3 and sm["fej"]["info_mean"] < 1e-9 and sm["iekf"]["info_mean"] == 0.0


def test_monte_carlo_standard_ekf_is_overconfident_on_a_sparse_map(m):
    p = m.Params(loops=5, n_landmarks=12)
    rng_map, rng_mc = np.random.default_rng(3).spawn(2)
    scene = m.make_scene(p, rng_map)
    vis = m.visibility(p, scene)
    res = m.run_monte_carlo(p, scene, vis, ("standard", "iekf", "ideal"), 24, rng_mc)
    sm = m.summarize(res, 40, 24)
    assert 2.0 < sm["iekf"]["nees_all"] < 5.0 and 2.0 < sm["ideal"]["nees_all"] < 6.0
    assert sm["standard"]["nees_all"] > 2.0 * sm["iekf"]["nees_all"]
