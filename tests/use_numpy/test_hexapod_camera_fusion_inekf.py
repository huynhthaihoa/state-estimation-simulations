import copy

import numpy as np
import pytest
from dataclasses import replace

from state_estimation_simulations.use_numpy import hexapod_camera_fusion_inekf as m
from state_estimation_simulations.use_numpy import hexapod_contact_inekf as c
from state_estimation_simulations.use_numpy.lie_utils import sek3_exp, sek3_inv, sek3_log, sek3_make, so3_exp


def setup(failures=False, n_steps=100, **cam_kw):
    args = m.build_parser().parse_args([])
    args.duration = n_steps * 0.02
    for k, v in cam_kw.items():
        setattr(args, k, v)
    cfg = replace(c.Config(), n_steps=n_steps, twist0=(args.speed, 0.0, args.turn_rate))
    cam = m.camera_settings(args, failures)
    return cfg, cam


def test_camera_draws_come_after_the_contact_trial():
    """The legs / IMU data are those of hexapod_contact_inekf.simulate_trial
    for the same seed; the camera draws do not change them."""
    cfg, cam = setup()
    rng = np.random.default_rng(3)
    t0 = c.simulate_trial(cfg, ["tripod"], rng)
    m.Camera(t0, cfg, rng, cam)
    t1 = c.simulate_trial(cfg, ["tripod"], np.random.default_rng(3))
    assert np.array_equal(t0.gyro_meas, t1.gyro_meas) and np.array_equal(t0.feet["tripod"][1], t1.feet["tripod"][1], equal_nan=True)


def test_noise_free_camera_without_drift_reads_the_truth():
    cfg, cam = setup(pos_std=0.0, rot_std=0.0, vel_std=0.0, scale_std=0.0, yaw_drift_std=0.0)
    trial = c.simulate_trial(cfg, ["tripod"], np.random.default_rng(0))
    cam_data = m.Camera(trial, cfg, np.random.default_rng(1), cam)
    assert np.allclose(cam_data.p, trial.p) and np.allclose(cam_data.R, trial.R)
    assert np.allclose(cam_data.v, np.einsum("kji,kj->ki", trial.R, trial.v))
    assert cam_data.ok.all() and cam_data.jump_tick is None


def test_failures_add_a_persistent_jump_and_a_dropout():
    """Position jump only (a yaw jump would also rotate the displacement about the start)."""
    cfg, cam = setup(failures=True, n_steps=3000, pos_std=0.0, rot_std=0.0, scale_std=0.0, yaw_drift_std=0.0,
                     jump_yaw_deg=0.0)
    trial = c.simulate_trial(cfg, ["tripod"], np.random.default_rng(0))
    cd = m.Camera(trial, cfg, np.random.default_rng(2), cam)
    k = cd.jump_tick
    off = cd.p - trial.p
    assert np.allclose(off[:k], 0.0) and np.allclose(off[k:], off[k]) and not np.allclose(off[k], 0.0)
    assert (~cd.ok).sum() == int(round(cam.drop_len / cfg.dt))


@pytest.mark.parametrize("kind", m.CAM_KINDS)
def test_camera_measurement_jacobian(kind):
    """r(X_hat) - r(X) ~ H xi for a small right-invariant error xi of
    (R, v, p) (and the offsets for cam_pose_drift), noise-free readings."""
    cfg, cam = setup(pos_std=0.0, rot_std=0.0, vel_std=0.0, scale_std=0.0, yaw_drift_std=0.0)
    rng = np.random.default_rng(5)
    R, v, p = so3_exp(rng.normal(size=3) * 0.3), rng.normal(size=3), rng.normal(size=3)
    y = (p, R, R.T @ v)
    f = m.CameraFusionInEKF(R, v, p, np.eye(9), cfg, kind, cam, False)
    r0, H, _ = f.camera_measurement(y)
    assert np.allclose(r0, 0.0, atol=1e-12)
    xi = rng.normal(size=9) * 1e-6
    Xh = sek3_exp(xi) @ sek3_make(R, [v, p])
    f.R, f.v, f.p = Xh[0:3, 0:3], Xh[0:3, 3], Xh[0:3, 4]
    full = np.zeros(len(f.P))
    full[:9] = xi
    if kind == "cam_pose_drift":
        f.b_off = rng.normal(size=4) * 1e-6
        full[9:13] = f.b_off
    r1, _, _ = f.camera_measurement(y)
    assert np.allclose(r1, H @ full, atol=1e-10)


def test_velocity_jacobian_is_constant():
    cfg, cam = setup()
    f = m.CameraFusionInEKF(so3_exp([0.3, -0.2, 1.0]), np.ones(3), np.ones(3) * 5, np.eye(9), cfg, "cam_vel", cam,
                            False)
    _, H, _ = f.camera_measurement((np.zeros(3), np.eye(3), np.zeros(3)))
    expected = np.zeros((3, 9))
    expected[:, 3:6] = -np.eye(3)
    assert np.array_equal(H, expected)


def test_offset_states_drift_with_distance():
    cfg, cam = setup()
    trial = c.simulate_trial(cfg, ["tripod"], np.random.default_rng(0))
    f = m.make_filter("cam_pose_drift", trial, cfg, cam, False)
    P0 = f.P[9:13, 9:13].copy()
    f.predict(trial.gyro_meas[0], trial.accel_meas[0])
    ds = np.linalg.norm(f.v) * cfg.dt
    assert np.allclose(np.diag(f.P[9:13, 9:13] - P0), [cam.off_pos_q * ds] * 3 + [cam.yaw_drift_std ** 2 * ds])


def test_gate_rejects_a_large_jump():
    cfg, cam = setup()
    R, v, p = np.eye(3), np.zeros(3), np.zeros(3)
    f = m.CameraFusionInEKF(R, v, p, np.eye(9) * 1e-4, cfg, "cam_pose", cam, True)
    assert not f.camera_update((p + [1.0, 0.0, 0.0], R, v))
    assert f.camera_update((p + [0.001, 0.0, 0.0], R, v))


# --- camera delay (--delay-experiment) ---------------------------------------

def noise_free(**kw):
    return dict(pos_std=0.0, rot_std=0.0, vel_std=0.0, scale_std=0.0, yaw_drift_std=0.0, **kw)


def true_imu(trial, cfg, k):
    """The true (gyro, accel) of the steps 0..k-1, from the truth arrays."""
    return [(m.so3_log(trial.R[j].T @ trial.R[j + 1]) / cfg.dt,
             trial.R[j].T @ ((trial.v[j + 1] - trial.v[j]) / cfg.dt - c.G_VEC)) for j in range(k)]


def test_delayed_camera_reads_the_body_at_the_earlier_time():
    cfg, cam = setup(n_steps=60, **noise_free())
    cam.delay = 0.06
    trial = c.simulate_trial(cfg, ["tripod"], np.random.default_rng(0))
    cd = m.Camera(trial, cfg, np.random.default_rng(1), cam)
    assert np.allclose(cd.p[3:], trial.p[:-3]) and np.allclose(cd.R[3:], trial.R[:-3])
    assert not cd.ok[:3].any() and cd.ok[3:].all()


@pytest.mark.parametrize("delay", [0.01, 0.03, 0.1, 0.2])
def test_known_delay_residual_vanishes_at_the_truth(delay):
    """The estimate rewound through the true IMU readings is the body at the
    sample time the camera used, so a filter at the truth has a zero residual
    (also for delays that are not a whole number of ticks)."""
    cfg, cam = setup(n_steps=60, **noise_free())
    cam.delay = delay
    trial = c.simulate_trial(cfg, ["tripod"], np.random.default_rng(0))
    cd = m.Camera(trial, cfg, np.random.default_rng(1), cam)
    k = 40
    f = m.CameraFusionInEKF(trial.R[k], trial.v[k], trial.p[k], np.eye(9), cfg, "cam_pose_drift", cam, False,
                            "known", delay)
    f.imu.extend(true_imu(trial, cfg, k))
    r, _, _ = f.camera_measurement((cd.p[k], cd.R[k], cd.v[k]))
    assert np.allclose(r, 0.0, atol=1e-12)


def test_error_transition_inverse_maps_the_error_back():
    """An error xi at time t, propagated through the same noise-free steps
    with no update, is Phi xi; the map goes back by the lag."""
    cfg, cam = setup(n_steps=60, **noise_free())
    trial = c.simulate_trial(cfg, ["tripod"], np.random.default_rng(0))
    xi0 = np.random.default_rng(2).normal(size=9) * 1e-6
    X = sek3_make(trial.R[0], [trial.v[0], trial.p[0]])
    Xh = sek3_exp(xi0) @ X
    for gyro, accel in true_imu(trial, cfg, 10):
        X, Xh = [sek3_make(*(lambda R, v, p: (R, [v, p]))(*c.propagate_base(Y[0:3, 0:3], Y[0:3, 3], Y[0:3, 4],
                                                                         gyro, accel, cfg.dt))) for Y in (X, Xh)]
    xi_now = sek3_log(Xh @ sek3_inv(X))
    assert np.allclose(m.error_transition_inverse(10 * cfg.dt) @ xi_now, xi0, atol=1e-10)


def delayed_filter(delay_hat, mode="estimated", n_ticks=50, b_yaw=0.004):
    """A filter after n_ticks of prediction only (random-looking IMU history), plus a camera reading."""
    cfg, cam = setup(n_steps=n_ticks + 5)
    trial = c.simulate_trial(cfg, ["tripod"], np.random.default_rng(4))
    cd = m.Camera(trial, cfg, np.random.default_rng(5), cam)
    f = m.make_filter("cam_pose_drift", trial, cfg, cam, False, mode, delay_hat, 0.1)
    for k in range(n_ticks):
        f.predict(trial.gyro_meas[k], trial.accel_meas[k])
    f.b_off = np.array([0.003, -0.002, 0.001, b_yaw])
    f.d_hat = delay_hat
    # a reading that nearly agrees with the filter (small residual: the Jacobians drop second order in it)
    R_then, _, p_then, _ = m.rewind(f.R, f.v, f.p, f.imu, delay_hat, cfg.dt)
    y_R = c.rot_z(f.b_off[3]) @ so3_exp(np.array([1e-4, -2e-4, 1.5e-4])) @ R_then
    return f, (p_then + f.b_off[:3] + [1e-3, -2e-3, 1e-3], y_R, cd.v[n_ticks])


@pytest.mark.parametrize("delay", [0.0437, 0.0987, -0.013])
def test_delay_jacobian_column_matches_finite_differences(delay):
    f, y = delayed_filter(delay)
    _, H, _ = f.camera_measurement(y)
    eps = 1e-6
    f.d_hat = delay + eps
    r_plus = f.camera_measurement(y)[0]
    f.d_hat = delay - eps
    r_minus = f.camera_measurement(y)[0]
    assert np.allclose((r_plus - r_minus) / (2 * eps), H[:, 13], atol=2e-3)


@pytest.mark.parametrize("delay", [0.0437, 0.0987])
def test_delayed_residual_jacobian_on_the_state_error(delay):
    """r(X_hat) - r(X) ~ H xi for a small right-invariant error of (R, v, p),
    the offsets b (4) and the delay: the same data, the perturbed estimate.
    The yaw offset is tiny: like the undelayed Jacobian, H takes R_z(b_psi) as the identity."""
    f, y = delayed_filter(delay, b_yaw=1e-6)
    r0, H, _ = f.camera_measurement(y)
    rng = np.random.default_rng(6)
    xi = rng.normal(size=9) * 1e-6
    full = np.zeros(len(f.P))
    full[:9] = xi
    Xh = sek3_exp(xi) @ sek3_make(f.R, [f.v, f.p])
    f.R, f.v, f.p = Xh[0:3, 0:3], Xh[0:3, 3], Xh[0:3, 4]
    full[9:13] = rng.normal(size=4) * 1e-6
    f.b_off = f.b_off + full[9:13]
    full[13] = 1e-6
    f.d_hat += full[13]
    r1, _, _ = f.camera_measurement(y)
    assert np.allclose(r1 - r0, H @ full, atol=1e-9)


def test_delay_machinery_changes_nothing_without_delay():
    """No delay: "known" is "ignore", and "estimated" with a zero prior std on the delay too."""
    cfg, cam = setup(n_steps=150)
    trial = c.simulate_trial(cfg, ["tripod"], np.random.default_rng(0))
    cd = m.Camera(trial, cfg, np.random.default_rng(1), cam)
    base = m.run_filter("cam_pose_drift", trial, cd, "tripod", cfg, cam)
    for mode, std in (("known", 0.1), ("estimated", 0.0)):
        o = m.run_filter("cam_pose_drift", trial, cd, "tripod", cfg, cam, False, mode, 0.0, std)
        assert np.allclose(o["pos_err"], base["pos_err"], atol=1e-10)
        assert np.allclose(o["nees_pos"], base["nees_pos"], rtol=1e-8)


def test_default_camera_settings_have_no_delay():
    cfg, cam = setup()
    assert cam.delay == 0.0


def replay_known(trial, camera, cfg, cam, m_ticks, until):
    """Reference for the integer-tick delay: a filter that runs m_ticks
    behind, takes each reading at the tick it was sampled, and is replayed
    forward through the leg and IMU data to the present. Returns the position
    error (until, 3) of the replayed estimate at every tick."""
    stance, y = trial.feet["tripod"]
    behind = m.make_filter("cam_pose_drift", trial, cfg, cam, False)
    errs = np.zeros((until, 3))
    for k in range(until):
        j = k - m_ticks
        if j >= 0:
            if j > 0:
                behind.predict(trial.gyro_meas[j - 1], trial.accel_meas[j - 1])
            behind.step(stance[j], y[j])
            behind.camera_update((camera.p[k], camera.R[k], camera.v[k]))
            now = copy.deepcopy(behind)
            for i in range(j + 1, k + 1):
                now.predict(trial.gyro_meas[i - 1], trial.accel_meas[i - 1])
                now.step(stance[i], y[i])
        else:
            now = behind
            if k > 0:
                now.predict(trial.gyro_meas[k - 1], trial.accel_meas[k - 1])
            now.step(stance[k], y[k])
        errs[k] = now.p - trial.p[k]
    return errs


def test_known_delay_matches_the_replay_on_one_run():
    """100 ms (5 ticks): the rewind filter's position error stays close to the exact replay's."""
    cfg, cam = setup(n_steps=300)
    cam.delay = 0.1
    trial = c.simulate_trial(cfg, ["tripod"], np.random.default_rng(0))
    cd = m.Camera(trial, cfg, np.random.default_rng(1), cam)
    rep = replay_known(trial, cd, cfg, cam, 5, 301)
    o = m.run_filter("cam_pose_drift", trial, cd, "tripod", cfg, cam, False, "known", 0.1)
    ign = m.run_filter("cam_pose_drift", trial, cd, "tripod", cfg, cam)
    rmse = lambda e: np.sqrt(np.mean(np.sum(e[50:] ** 2, axis=1)))
    assert rmse(o["pos_err"]) < 1.2 * rmse(rep)
    assert rmse(rep) < rmse(ign["pos_err"])


def test_estimated_delay_converges():
    """Regression: 100 ms, 20 s straight walk, seed 0."""
    cfg, cam = setup(n_steps=1000)
    cam.delay = 0.1
    trial = c.simulate_trial(cfg, ["tripod"], np.random.default_rng(0))
    cd = m.Camera(trial, cfg, np.random.default_rng(1), cam)
    o = m.run_filter("cam_pose_drift", trial, cd, "tripod", cfg, cam, False, "estimated", 0.1, 0.1)
    assert abs(o["d_hat"][-1] - 0.1) < 0.002 and o["d_std"][-1] < 0.002
