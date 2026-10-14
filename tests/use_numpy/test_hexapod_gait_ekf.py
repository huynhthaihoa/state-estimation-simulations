import numpy as np
import pytest


@pytest.fixture
def hexapod_gait_ekf(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "hexapod_gait_ekf")


DT = 0.02
TICKS_PER_CYCLE = 60


# --- gait schedules ---

@pytest.mark.parametrize("gait, n_down", [("tripod", 3), ("ripple", 4), ("wave", 5)])
def test_stance_count_constant_per_gait(hexapod_gait_ekf, gait, n_down):
    stance = hexapod_gait_ekf.stance_mask(3 * TICKS_PER_CYCLE, TICKS_PER_CYCLE, gait)
    assert np.all(stance.sum(axis=1) == n_down)


@pytest.mark.parametrize("gait", ["tripod", "ripple", "wave"])
def test_duty_factor_per_leg(hexapod_gait_ekf, gait):
    m = hexapod_gait_ekf
    duty, _ = m.GAITS[gait]
    stance = m.stance_mask(TICKS_PER_CYCLE, TICKS_PER_CYCLE, gait)
    assert np.allclose(stance.mean(axis=0), duty)


def test_tripod_alternates_two_triangles(hexapod_gait_ekf):
    stance = hexapod_gait_ekf.stance_mask(TICKS_PER_CYCLE, TICKS_PER_CYCLE, "tripod")
    a, b = [0, 2, 4], [1, 3, 5]  # L1, L3, R2 and L2, R1, R3
    for row in stance:
        assert row[a].all() != row[b].all()
        assert row[a].all() or row[b].all()


def test_unknown_gait_raises(hexapod_gait_ekf):
    with pytest.raises(ValueError):
        hexapod_gait_ekf.stance_mask(10, TICKS_PER_CYCLE, "gallop")


# --- truth ---

def _truth(m, n_steps=300, seed=0):
    rng = np.random.default_rng(seed)
    return m.simulate_body_trajectory(n_steps, DT, np.array([0.1, 0.0, 0.2]), 0.05, 0.1, rng)


def test_stance_foot_fixed_in_world(hexapod_gait_ekf):
    m = hexapod_gait_ekf
    x_true = _truth(m)
    stance, r, _ = m.simulate_feet(x_true, DT, TICKS_PER_CYCLE, "ripple", 0.1)
    for i in range(6):
        for k in range(1, len(x_true)):
            if stance[k, i] and stance[k - 1, i]:
                f_prev = x_true[k - 1, :2] + m.rot(x_true[k - 1, 2]) @ r[k - 1, i]
                f_now = x_true[k, :2] + m.rot(x_true[k, 2]) @ r[k, i]
                assert np.allclose(f_prev, f_now, atol=1e-12)


def test_relative_velocity_matches_finite_difference(hexapod_gait_ekf):
    # the instantaneous -(b + omega x r) should match how r actually moves between ticks
    m = hexapod_gait_ekf
    x_true = m.simulate_body_trajectory(300, DT, np.array([0.1, 0.0, 0.2]), 0.0, 0.0,
                                        np.random.default_rng(0))
    stance, r, rel_vel = m.simulate_feet(x_true, DT, TICKS_PER_CYCLE, "wave", 0.1)
    k, i = 100, int(np.flatnonzero(stance[100] & stance[101])[0])
    fd = (r[k + 1, i] - r[k, i]) / DT
    assert np.allclose(fd, rel_vel[k, i], atol=2e-3)  # first-order in dt


def test_slipping_foot_moves_in_world(hexapod_gait_ekf):
    m = hexapod_gait_ekf
    x_true = _truth(m)
    slip = (1, 0.0, 10.0, np.array([0.0, 0.1]))
    stance, r, rel_vel = m.simulate_feet(x_true, DT, TICKS_PER_CYCLE, "wave", 0.1, slip)
    _, _, rel_clean = m.simulate_feet(x_true, DT, TICKS_PER_CYCLE, "wave", 0.1)
    k = int(np.flatnonzero(stance[1:, 1] & stance[:-1, 1])[0]) + 1
    f_prev = x_true[k - 1, :2] + m.rot(x_true[k - 1, 2]) @ r[k - 1, 1]
    f_now = x_true[k, :2] + m.rot(x_true[k, 2]) @ r[k, 1]
    assert np.linalg.norm(f_now - f_prev) > 0.5 * 0.1 * DT
    assert np.allclose(rel_vel[:, [0, 2, 3, 4, 5]], rel_clean[:, [0, 2, 3, 4, 5]])


# --- filter pieces ---

def test_predict_jacobian_matches_numerical(hexapod_gait_ekf):
    m = hexapod_gait_ekf
    x = np.array([0.3, -0.2, 0.7, 0.1, 0.05, 0.2])
    F = m.predict_jacobian(x, DT)
    eps = 1e-6
    F_num = np.column_stack([(m.propagate(x + eps * e, DT) - m.propagate(x - eps * e, DT)) / (2 * eps)
                             for e in np.eye(6)])
    assert np.allclose(F, F_num, atol=1e-8)


def test_measurement_jacobian_reproduces_model(hexapod_gait_ekf):
    m = hexapod_gait_ekf
    x = np.array([0.3, -0.2, 0.7, 0.1, 0.05, 0.2])
    r_i = np.array([0.2, 0.15])
    expected = -(x[3:5] + x[5] * np.array([-r_i[1], r_i[0]]))
    assert np.allclose(m.foot_measurement_jacobian(r_i) @ x, expected)


def test_noise_free_run_recovers_twist(hexapod_gait_ekf):
    m = hexapod_gait_ekf
    x_true = _truth(m)
    stance, r, rel_vel = m.simulate_feet(x_true, DT, TICKS_PER_CYCLE, "tripod", 0.1)
    Q = m.process_noise_covariance(DT, 0.05, 0.1)
    x_init = x_true[0] + np.array([0.0, 0.0, 0.0, 0.05, -0.05, 0.05])
    P_init = np.diag([1e-4, 1e-4, 1e-4, 0.05 ** 2, 0.05 ** 2, 0.05 ** 2])
    x_est, _, _ = m.run_ekf(x_init, P_init, rel_vel, r, stance, DT, Q, 1e-8)
    assert np.allclose(x_est[-1, 3:], x_true[-1, 3:], atol=1e-3)


def test_pose_covariance_grows(hexapod_gait_ekf):
    # leg odometry never measures px, py, theta, so their variance only grows
    m = hexapod_gait_ekf
    x_true = _truth(m)
    stance, r, rel_vel = m.simulate_feet(x_true, DT, TICKS_PER_CYCLE, "wave", 0.1)
    Q = m.process_noise_covariance(DT, 0.05, 0.1)
    _, P_list, _ = m.run_ekf(x_true[0], np.eye(6) * 1e-4, rel_vel, r, stance, DT, Q, 0.02 ** 2)
    var = np.array([np.diag(P)[:3] for P in P_list])
    assert np.all(np.diff(var, axis=0) >= -1e-15)
    assert np.all(var[-1] > var[0])


def test_gating_rejects_slip_and_keeps_good_feet(hexapod_gait_ekf):
    m = hexapod_gait_ekf
    x = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.2])
    P = np.diag([1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5])
    r_feet = m.NOMINAL_FOOT_OFFSETS[[0, 4, 2]]
    z = np.array([m.foot_measurement_jacobian(ri) @ x for ri in r_feet])
    z[1] += np.array([0.0, -0.1])  # R2 slipping outward
    _, _, accepted = m.measurement_update_feet(x, P, z, r_feet, 0.02 ** 2, gating=True)
    assert accepted.tolist() == [True, False, True]
    _, _, accepted = m.measurement_update_feet(x, P, z, r_feet, 0.02 ** 2, gating=False)
    assert accepted.all()


def test_trial_shares_truth_and_noise_across_gaits(hexapod_gait_ekf):
    # controlled comparison: adding a gait must not change any other gait's draws
    m = hexapod_gait_ekf
    args = (200, DT, TICKS_PER_CYCLE)
    rest = (np.array([0.1, 0.0, 0.2]), 0.05, 0.1, 0.02, np.full(6, 0.01), 0.1)
    xa, ia, fa = m.simulate_trial(*args, ["tripod"], *rest, np.random.default_rng(3))
    xb, ib, fb = m.simulate_trial(*args, ["wave", "tripod", "ripple"], *rest, np.random.default_rng(3))
    assert np.array_equal(xa, xb) and np.array_equal(ia, ib)
    for a, b in zip(fa["tripod"], fb["tripod"]):
        assert np.array_equal(a, b)


def test_monte_carlo_consistent_without_slip(hexapod_gait_ekf):
    m = hexapod_gait_ekf
    from utils import averaged_nees_bounds
    n_trials = 30
    acc = m.run_monte_carlo(n_trials, 300, DT, TICKS_PER_CYCLE, ["tripod"], np.array([0.1, 0.0, 0.2]),
                            0.05, 0.1, 0.02, np.full(6, 0.01), 0.1, np.random.default_rng(0))
    lo, hi = averaged_nees_bounds(3, n_trials)
    nees_twist = acc[("tripod", False)]["nees_twist"][1:].mean()
    assert lo < nees_twist < hi
