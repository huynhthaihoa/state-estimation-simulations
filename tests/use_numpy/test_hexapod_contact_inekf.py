import numpy as np
import pytest


@pytest.fixture
def hexapod_contact_inekf(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "hexapod_contact_inekf")


EPS = 1e-6
N_FEET = 3


def make_state(m, rng):
    R = m.so3_exp(rng.normal(size=3))
    return R, rng.normal(size=3), rng.normal(size=3), rng.normal(size=(N_FEET, 3)), 0.1 * rng.normal(size=6)


def build(m, kind, cfg, R, v, p, d, b, biases):
    """A filter whose estimate is exactly (R, v, p, d, b), feet added through augment."""
    cls = m.InvariantContactEKF if kind == "inekf" else m.WorldContactEKF
    f = cls(R, v, p, np.eye(9), cfg, biases, np.eye(6))
    for j, dj in enumerate(d):
        f.augment(j, R.T @ (dj - p))
    f.b = b.copy()
    return f


def perturbed(m, kind, cfg, f, delta):
    """Copy of f whose error relative to f (in kind's own coordinates) is delta."""
    g = build(m, kind, cfg, f.R, f.v, f.p, f.d, f.b, f.biases)
    g.correct(-delta)
    return g


def error(m, kind, est, tru):
    """Full error [R, v, p, (bias), feet] of est relative to tru, in kind's coordinates."""
    bias = (est.b - tru.b if kind == "inekf" else tru.b - est.b) if est.biases else []
    if kind == "inekf":
        xi = m.sek3_log(est.group() @ m.sek3_inv(tru.group()))
        return np.concatenate([xi[:9], bias, xi[9:]])
    return np.concatenate([m.so3_log(tru.R @ est.R.T), tru.v - est.v, tru.p - est.p, bias, (tru.d - est.d).ravel()])


def numerical_transition(m, kind, cfg, f, gyro, accel):
    """Finite-difference transition of the error over one predict; the truth
    propagates with the true inputs, the estimate with true + true bias."""
    def propagated(g):
        h = build(m, kind, cfg, g.R, g.v, g.p, g.d, g.b, g.biases)
        h.predict(gyro + f.b[:3], accel + f.b[3:])
        return h
    t1 = propagated(f)
    n = len(f.P)
    F = np.zeros((n, n))
    for c in range(n):
        delta = np.zeros(n)
        delta[c] = EPS
        F[:, c] = error(m, kind, propagated(perturbed(m, kind, cfg, f, delta)), t1) / EPS
    return F


def analytic_transition(m, f, gyro, accel):
    R_prev = f.R
    f.R, f.v, f.p = m.propagate_base(f.R, f.v, f.p, gyro, accel, f.cfg.dt)
    return f.predict_jacobians(R_prev, accel, f.cfg.dt)[0]


# --- Jacobians ---

@pytest.mark.parametrize("kind", ["inekf", "world"])
def test_transition_matches_finite_difference_without_biases(hexapod_contact_inekf, kind):
    m = hexapod_contact_inekf
    cfg = m.Config()
    rng = np.random.default_rng(0)
    R, v, p, d, b = make_state(m, rng)
    gyro, accel = rng.normal(size=3), rng.normal(size=3) + [0.0, 0.0, 9.81]
    f = build(m, kind, cfg, R, v, p, d, np.zeros(6), False)
    F_num = numerical_transition(m, kind, cfg, f, gyro, accel)
    assert np.allclose(analytic_transition(m, f, gyro, accel), F_num, atol=1e-6)


def test_invariant_transition_is_state_independent(hexapod_contact_inekf):
    m = hexapod_contact_inekf
    cfg = m.Config()
    rng = np.random.default_rng(1)
    Fs = []
    for _ in range(2):
        R, v, p, d, _ = make_state(m, rng)
        f = build(m, "inekf", cfg, R, v, p, d, np.zeros(6), False)
        Fs.append(analytic_transition(m, f, rng.normal(size=3), rng.normal(size=3)))
    assert np.array_equal(Fs[0], Fs[1])


@pytest.mark.parametrize("kind", ["inekf", "world"])
def test_bias_transition_is_first_order_accurate(hexapod_contact_inekf, kind):
    """The bias columns are a first-order discretization: error O(h^2), so
    halving dt cuts the mismatch by about 4."""
    m = hexapod_contact_inekf
    rng = np.random.default_rng(2)
    R, v, p, d, b = make_state(m, rng)
    gyro, accel = rng.normal(size=3), rng.normal(size=3) + [0.0, 0.0, 9.81]
    mismatch = []
    for dt in (0.02, 0.01):
        cfg = m.Config(dt=dt)
        f = build(m, kind, cfg, R, v, p, d, b, True)
        F_num = numerical_transition(m, kind, cfg, f, gyro, accel)
        mismatch.append(np.abs(analytic_transition(m, f, gyro, accel) - F_num).max())
    assert mismatch[0] < 10.0 * 0.02 ** 2
    assert 3.0 < mismatch[0] / mismatch[1] < 5.0


@pytest.mark.parametrize("kind", ["inekf", "world"])
def test_measurement_jacobian_matches_finite_difference(hexapod_contact_inekf, kind):
    m = hexapod_contact_inekf
    cfg = m.Config()
    rng = np.random.default_rng(3)
    R, v, p, d, _ = make_state(m, rng)
    f = build(m, kind, cfg, R, v, p, d, np.zeros(6), False)
    y = R.T @ (d[1] - p)
    _, H = f.measurement(1, y)
    H_num = np.zeros_like(H)
    for c in range(len(f.P)):
        delta = np.zeros(len(f.P))
        delta[c] = EPS
        H_num[:, c] = perturbed(m, kind, cfg, f, delta).measurement(1, y)[0] / EPS
    assert np.allclose(H, H_num, atol=1e-6)


def test_invariant_measurement_jacobian_is_constant(hexapod_contact_inekf):
    m = hexapod_contact_inekf
    cfg = m.Config()
    rng = np.random.default_rng(4)
    Hs = [build(m, "inekf", cfg, *make_state(m, rng)[:4], np.zeros(6), False).measurement(2, rng.normal(size=3))[1]
          for _ in range(2)]
    assert np.array_equal(Hs[0], Hs[1])


@pytest.mark.parametrize("kind", ["inekf", "world"])
def test_touchdown_jacobian_matches_finite_difference(hexapod_contact_inekf, kind):
    m = hexapod_contact_inekf
    cfg = m.Config()
    rng = np.random.default_rng(5)
    R, v, p, d, _ = make_state(m, rng)
    f = build(m, kind, cfg, R, v, p, d, np.zeros(6), False)
    y = 0.2 * rng.normal(size=3)
    M = f.touchdown_jacobian(y)
    M_num = np.zeros_like(M)
    for c in range(len(f.P)):
        delta = np.zeros(len(f.P))
        delta[c] = EPS
        g = perturbed(m, kind, cfg, f, delta)
        t, e = build(m, kind, cfg, f.R, f.v, f.p, f.d, f.b, False), g
        t.augment(9, y)
        e.augment(9, y)
        M_num[:, c] = error(m, kind, e, t)[-3:] / EPS
    assert np.allclose(M, M_num, atol=1e-6)


# --- Contact bookkeeping ---

def test_remove_keeps_the_other_blocks(hexapod_contact_inekf):
    m = hexapod_contact_inekf
    cfg = m.Config()
    rng = np.random.default_rng(6)
    f = build(m, "inekf", cfg, *make_state(m, rng)[:4], np.zeros(6), True)
    A = rng.normal(size=(len(f.P), len(f.P)))
    f.P = A @ A.T
    P_before, d_before = f.P.copy(), f.d.copy()
    f.remove(1)
    keep = np.r_[0:15, 15:18, 21:24]
    assert f.legs == [0, 2]
    assert np.array_equal(f.P, P_before[np.ix_(keep, keep)])
    assert np.array_equal(f.d, d_before[[0, 2]])


@pytest.mark.parametrize("gait", ["tripod", "ripple", "wave"])
def test_feet_in_state_follow_the_stance_schedule(hexapod_contact_inekf, gait):
    m = hexapod_contact_inekf
    cfg = m.Config(n_steps=150)
    trial = m.simulate_trial(cfg, [gait], np.random.default_rng(7))
    stance, y = trial.feet[gait]
    f = m.make_filter("inekf", trial, cfg)
    for k in range(len(stance)):
        if k > 0:
            f.predict(trial.gyro_meas[k - 1], trial.accel_meas[k - 1])
        f.step(stance[k], y[k])
        assert sorted(f.legs) == list(np.flatnonzero(stance[k]))
        assert len(f.P) == 9 + 3 * len(f.legs)


# --- Truth and whole runs ---

def test_truth_is_reproduced_by_the_imu_inputs(hexapod_contact_inekf):
    m = hexapod_contact_inekf
    cfg = m.Config(n_steps=200)
    R, v, p, _, gyro, accel = m.simulate_body(cfg, np.random.default_rng(8))
    for k in range(cfg.n_steps):
        R1, v1, p1 = m.propagate_base(R[k], v[k], p[k], gyro[k], accel[k], cfg.dt)
        assert np.allclose(R1, R[k + 1]) and np.allclose(v1, v[k + 1]) and np.allclose(p1, p[k + 1])


def test_noise_free_stance_feet_stay_planted_on_the_ground(hexapod_contact_inekf):
    m = hexapod_contact_inekf
    cfg = m.Config(n_steps=200)
    R, v, p, yaw, _, _ = m.simulate_body(cfg, np.random.default_rng(9))
    stance, d = m.simulate_feet(cfg, R, p, yaw, "tripod", np.zeros((cfg.n_steps + 1, 6, 3)))
    held = stance[1:] & stance[:-1]
    assert np.allclose(d[1:][held], d[:-1][held])
    assert np.allclose(d[stance][:, 2], 0.0)


@pytest.mark.parametrize("kind", ["inekf", "world"])
def test_noise_free_run_tracks_the_truth(hexapod_contact_inekf, kind):
    m = hexapod_contact_inekf
    zero = m.Config(n_steps=300, gyro_noise_std=0.0, accel_noise_std=0.0, contact_noise_std=0.0, kin_noise_std=0.0,
                    init_rot_std=0.0, init_vel_std=0.0, init_pos_std=0.0)
    trial = m.simulate_trial(zero, ["tripod"], np.random.default_rng(10))
    o = m.run_filter(kind, trial, "tripod", m.Config(n_steps=300))
    assert np.abs(o["pos_err"]).max() < 1e-10
    assert np.abs(o["rot_err"]).max() < 1e-10


def test_invariant_yaw_variance_never_shrinks_but_world_does(hexapod_contact_inekf):
    """Yaw is unobservable from IMU + leg kinematics. The InEKF's yaw
    variance can only grow (gravity has no yaw component, so no update ever
    reaches it); the world EKF's linearization lets it shrink."""
    m = hexapod_contact_inekf
    cfg = m.Config(n_steps=300, init_rot_std=0.1)
    trial = m.simulate_trial(cfg, ["tripod"], np.random.default_rng(11))
    yaw_in = m.run_filter("inekf", trial, "tripod", cfg)["yaw_var"]
    yaw_w = m.run_filter("world", trial, "tripod", cfg)["yaw_var"]
    assert np.all(np.diff(yaw_in) >= -1e-15)
    assert yaw_w[-1] < 0.5 * cfg.init_rot_std ** 2


def test_gating_rejects_a_slipping_foot(hexapod_contact_inekf):
    m = hexapod_contact_inekf
    cfg = m.Config()
    rng = np.random.default_rng(12)
    R, v, p, d, _ = make_state(m, rng)
    f = build(m, "inekf", cfg, R, v, p, d, np.zeros(6), False)
    f.P = 1e-6 * np.eye(len(f.P))
    f.gating = True
    ys = np.array([R.T @ (dj - p) for dj in d])
    ys[1] += [0.0, 0.1, 0.0]  # foot 1 has slid 10 cm
    assert list(f.update([0, 1, 2], ys)) == [True, False, True]


def test_options_do_not_change_the_other_draws(hexapod_contact_inekf):
    m = hexapod_contact_inekf
    cfg = m.Config(n_steps=200)
    slip = (1, 1.0, 2.0, np.array([0.0, 0.1, 0.0]))
    a = m.simulate_trial(cfg, ["tripod", "wave"], np.random.default_rng(13))
    b = m.simulate_trial(cfg, ["tripod", "wave"], np.random.default_rng(13), slip=slip, biases=True)
    assert np.array_equal(a.R, b.R) and np.array_equal(a.p, b.p) and np.array_equal(a.xi0, b.xi0)
    assert np.allclose(b.gyro_meas - a.gyro_meas, b.bias[:3])
    stance, ya = a.feet["tripod"]
    _, yb = b.feet["tripod"]
    other = np.ones(6, dtype=bool)
    other[1] = False
    assert np.array_equal(ya[:, other][stance[:, other]], yb[:, other][stance[:, other]])
