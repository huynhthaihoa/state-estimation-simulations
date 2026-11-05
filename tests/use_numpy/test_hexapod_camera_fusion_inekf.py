import numpy as np
import pytest
from dataclasses import replace

from state_estimation_simulations.use_numpy import hexapod_camera_fusion_inekf as m
from state_estimation_simulations.use_numpy import hexapod_contact_inekf as c
from state_estimation_simulations.use_numpy.lie_utils import sek3_exp, sek3_make, so3_exp


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
