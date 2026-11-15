'''
hexapod_contact_inekf.py's contact-aided invariant EKF (legs + IMU) fused with
a tracking camera that reports its own visual-inertial odometry, like the
Intel RealSense T265 on the CTU Computational Robotics Lab's hexapods.

Legs plus an IMU cannot observe absolute position or yaw; a tracking camera
reports a pose, but in its own odometry frame, which drifts away from the
world as the robot travels. Three ways to use it are compared with legs only:

  - "legs": hexapod_contact_inekf's InvariantContactEKF, no camera.
  - "cam_vel": the camera's velocity, in the body frame, y = R^T v + n. For
    the right-invariant EKF this is an invariant observation: residual
    R_hat y - v_hat has the constant Jacobian [0, -I, 0].
  - "cam_pose": the camera's position and rotation as if they were the world
    pose, y_p = p + n_p, y_R = Exp(n_R) R (white noise only).
  - "cam_pose_drift": the same readings, with the camera frame's drift as
    filter states: a position offset b_p (3) and a yaw offset b_psi,
    y_p = p + b_p + n_p, y_R = Rz(b_psi) Exp(n_R) R. Both are random walks
    whose variance grows with the distance travelled (estimated speed x dt).

Camera truth (assumed, T265-style; see the doc for sources):
  - pose readings every tick (the filter rate; the T265 reports at 200 Hz),
    in a frame that starts at the world pose and drifts: a scale error s
    (one draw per run, std `scale_std`) on the displacement from the start,
    and a yaw drift that random-walks with the distance travelled (std
    `yaw_drift_std` per sqrt(m)), about the start position;
  - velocity readings (1 + s) R^T v + n_v in the body frame (rs2_pose reports
    velocity);
  - optional failures: one pose jump per run (the camera frame shifts by
    N(0, jump_pos_std^2) per horizontal axis and N(0, jump_yaw_std^2) in yaw,
    and stays shifted; the default yaw std is 0, since the T265 doc says a pose
    jump changes only the translation) and one dropout window with no readings.

Gating (optional, camera readings only): normalized innovation squared
against the chi-squared 99% bound of the reading's dimension (3: 11.34,
6: 16.81).

Camera delay (--delay-experiment, replaces experiments 1 and 2; cam_pose_drift
only, no jump or dropout): the reading of tick k describes the body at time
t_k - d, with d the true delay (constant per run; a sweep, not a T265 value;
the body in between ticks follows the IMU model with constant inputs per tick).
Three ways to use it, on a straight and on a turning walk:
  - "ignore": the filter above, which compares the reading with its current
    estimate;
  - "known": d given. The estimate d seconds ago is the current estimate run
    backwards through the stored IMU readings (the exact inverse of the
    propagation); the residual compares the reading with that. The Jacobian on
    the current error is the one at that time times the inverse of the
    contact filter's state-independent error transition over d. Not modeled:
    the process noise accumulated over d and the leg updates since, which a
    full replay (Pronto) would redo;
  - "estimated": d is one more state (start 0 s, prior std --delay-init-std-ms),
    constant. The residual is the same function of d_hat; its derivative with
    respect to d is minus the rate of the predicted reading (rotation: the
    gyro; position: the velocity at that time), the first-order form of Li &
    Mourikis 2014 (their t_d is minus this d: their reading is taken at t + t_d).
A final block checks when d is identifiable: motions with no body bob, with
constant velocity and no rotation, and with constant velocity and turn rate.

Truth body, legs and IMU: hexapod_contact_inekf.simulate_trial, unchanged
(isotropic foot-position noise, no biases, true stance schedule), with a
straighter and faster default walk (0.2 m/s, no mean turn) so the run covers
some distance. Camera draws come after all of its draws.

Scores: physical position error (NEES with the covariance mapped to world
coordinates, and RMSE), yaw error (NEES, RMSE), velocity NEES, averaged after
the first second.
'''

import argparse
import copy
import time
from collections import deque
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from state_estimation_simulations.utils import averaged_nees_bounds  # noqa: E402
from state_estimation_simulations.use_numpy.lie_utils import sek3_exp, sek3_make, skew, so3_exp, so3_log  # noqa: E402
from state_estimation_simulations.use_numpy.hexapod_contact_inekf import (  # noqa: E402
    G_VEC, Config, InvariantContactEKF, initial_covariance, propagate_base, simulate_trial, world_from_invariant, rot_z)

KINDS = ("legs", "cam_vel", "cam_pose", "cam_pose_drift")
CAM_KINDS = KINDS[1:]
GATE = {3: 11.34, 6: 16.81}
LABELS = {"legs": "legs + IMU only", "cam_vel": "+ camera velocity", "cam_pose": "+ camera pose as world pose",
          "cam_pose_drift": "+ camera pose, drift as states"}


def split_lag(lag_s, dt):
    """Splits a lag into whole ticks and a forward remainder: the state
    `lag_s` seconds ago is the state m = ceil(lag_s / dt) ticks ago, advanced
    by h = m dt - lag_s with that step's constant IMU inputs (0 <= h < dt).
    Arguments:
        lag_s: time lag (s), >= 0
        dt: tick length (s)
    Returns:
        m: whole ticks to go back (int)
        h: forward remainder (s)"""
    ticks = lag_s / dt
    if abs(ticks - round(ticks)) < 1e-9:
        return int(round(ticks)), 0.0
    m = int(np.ceil(ticks))
    return m, (m - ticks) * dt


def delayed_truth(trial, dt, delay, psi, off_p):
    """The body at the time each reading was taken, `delay` seconds before its
    tick: the tick m = ceil(delay / dt) back, advanced by the IMU model
    (propagate_base, constant true gyro and accel over the tick, as the truth
    itself is built) for the remainder h. Ticks with k < m have no sample yet.
    Arguments:
        trial: truth (Trial)
        dt: tick length (s)
        delay: true camera delay (s), > 0
        psi: (N,) camera-frame yaw at each tick (rad)
        off_p: (N, 3) camera-frame position jump at each tick (m)
    Returns:
        p, R, v: (N, 3), (N, 3, 3), (N, 3) body position, rotation, world velocity at the sample times
        psi_t, off_p_t: the camera frame's yaw (N,) and jump (N, 3) at the sample times
        valid: (N,) bool, False where the sample time is before the start"""
    n = len(trial.p)
    m, h = split_lag(delay, dt)
    i = np.maximum(np.arange(n) - m, 0)
    j = np.minimum(i + 1, n - 1)
    p, R, v = np.zeros((n, 3)), np.zeros((n, 3, 3)), np.zeros((n, 3))
    for k in range(n):
        gyro = so3_log(trial.R[i[k]].T @ trial.R[j[k]]) / dt
        accel = trial.R[i[k]].T @ ((trial.v[j[k]] - trial.v[i[k]]) / dt - G_VEC)
        R[k], v[k], p[k] = propagate_base(trial.R[i[k]], trial.v[i[k]], trial.p[i[k]], gyro, accel, h)
    psi_t = psi[i] + (psi[j] - psi[i]) * (h / dt)
    return p, R, v, psi_t, off_p[i], np.arange(n) >= m


def rewind(R, v, p, imu, lag_s, dt):
    """The filter's (R, v, p) `lag_s` seconds ago: the current estimate run
    backwards through the IMU readings of the steps since (the exact inverse
    of propagate_base), then forward over the remainder (split_lag). A
    negative lag (an estimate of a delay below 0) runs the estimate forward
    with the latest IMU reading.
    Arguments:
        R, v, p: current rotation (3, 3), world velocity (m/s), position (m)
        imu: sequence of (gyro (rad/s), accel (m/s^2)) of the steps so far, latest last
        lag_s: time lag (s)
        dt: tick length (s)
    Returns:
        R, v, p: the state then
        gyro: gyro reading (rad/s, body frame) of the step that holds that instant
        None if the IMU history is shorter than the lag"""
    if lag_s < 0.0:
        if not imu:
            return R, v, p, np.zeros(3)
        gyro, accel = imu[-1]
        R, v, p = propagate_base(R, v, p, gyro, accel, -lag_s)
        return R, v, p, gyro
    m, h = split_lag(lag_s, dt)
    if m > len(imu):
        return None
    for j in range(m):
        gyro, accel = imu[-1 - j]
        R = R @ so3_exp(-gyro * dt)
        v = v - G_VEC * dt - R @ (accel * dt)
        p = p - v * dt - 0.5 * G_VEC * dt * dt - R @ (0.5 * accel * dt * dt)
    if not imu:
        return R, v, p, np.zeros(3)
    gyro, accel = imu[-max(m, 1)]
    if h > 0.0:
        R, v, p = propagate_base(R, v, p, gyro, accel, h)
    return R, v, p, gyro


def error_transition_inverse(lag_s):
    """Maps the (R, v, p) right-invariant error now to the error `lag_s` ago
    (the inverse of the contact filter's state-independent 9x9 transition,
    for the gravity-coupled part: velocity error += g x rotation error, position error += velocity error).
    Arguments:
        lag_s: time lag (s)
    Returns: (9, 9) matrix"""
    gx = skew(G_VEC)
    Phi = np.eye(9)
    Phi[3:6, 0:3] = -lag_s * gx
    Phi[6:9, 0:3] = 0.5 * lag_s ** 2 * gx
    Phi[6:9, 3:6] = -lag_s * np.eye(3)
    return Phi


class Camera:
    """One run's camera readings (see the module docstring).
    Attributes (N = ticks): p (N, 3), R (N, 3, 3) pose readings, v (N, 3)
    body-frame velocity readings, ok (N,) False in the dropout window, and
    the truth drift: scale, yaw_drift (N,), jump_tick (or None). With
    cam.delay > 0 (s), the reading of tick k describes the body at time
    t_k - delay (see delayed_truth) and ok is False before the first such time."""

    def __init__(self, trial, cfg, rng, cam):
        n = len(trial.p)
        # draws (always the same number, whatever the settings)
        scale = rng.normal() * cam.scale_std
        dpsi = rng.normal(size=n)
        n_p, n_R, n_v = rng.normal(size=(n, 3)), rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
        u_jump, jump = rng.random(), rng.normal(size=3)
        u_drop = rng.random()
        step = np.r_[0.0, np.linalg.norm(np.diff(trial.p, axis=0), axis=1)]
        self.yaw_drift = np.cumsum(cam.yaw_drift_std * np.sqrt(step) * dpsi)
        off_p, off_psi = np.zeros((n, 3)), np.zeros(n)
        self.jump_tick = None
        if cam.failures:
            k = int(cam.jump_window[0] / cfg.dt + u_jump * (cam.jump_window[1] - cam.jump_window[0]) / cfg.dt)
            self.jump_tick = k
            off_p[k:, :2] = jump[:2] * cam.jump_pos_std
            off_psi[k:] = jump[2] * cam.jump_yaw_std
        psi = self.yaw_drift + off_psi
        p0 = trial.p[0]
        # the body at the time of each reading: this tick's, unless the camera's readings are late
        p_body, R_body, v_body = trial.p, trial.R, trial.v
        valid = np.ones(n, dtype=bool)
        if getattr(cam, "delay", 0.0) > 0.0:
            p_body, R_body, v_body, psi, off_p, valid = delayed_truth(trial, cfg.dt, cam.delay, psi, off_p)
        self.p = np.array([p0 + rot_z(psi[k]) @ ((1.0 + scale) * (p_body[k] - p0)) for k in range(n)])
        self.p += off_p + n_p * cam.pos_std
        self.R = np.array([rot_z(psi[k]) @ so3_exp(n_R[k] * cam.rot_std) @ R_body[k] for k in range(n)])
        self.v = np.array([(1.0 + scale) * R_body[k].T @ v_body[k] for k in range(n)]) + n_v * cam.vel_std
        self.ok = valid
        if cam.failures:
            k0 = int(cam.drop_window[0] / cfg.dt + u_drop * (cam.drop_window[1] - cam.drop_window[0]) / cfg.dt)
            self.ok[k0:k0 + int(round(cam.drop_len / cfg.dt))] = False
        self.scale = scale
        self.delay = getattr(cam, "delay", 0.0)


class CameraFusionInEKF(InvariantContactEKF):
    """InvariantContactEKF plus camera updates (kind in KINDS). For
    cam_pose_drift the P layout is [rotation, velocity, position,
    b_p (3), b_psi (1), feet]; the offset error is zeta = b_hat - b.
    Late readings (kind cam_pose_drift only), delay_mode:
      "ignore": the reading is compared with the current estimate (the default);
      "known": compared with the estimate delay_known seconds ago (rewind);
      "estimated": as "known", with the delay as one more state, d_hat, after
        b_psi in P ([..., b_p, b_psi, d, feet]); d_init_std is its prior std (s)."""

    def __init__(self, R, v, p, P_base, cfg, kind, cam, gating, delay_mode="ignore", delay_known=0.0,
                 d_init_std=0.1):
        super().__init__(R, v, p, P_base, cfg)
        self.kind, self.cam, self.cam_gating = kind, cam, gating
        self.b_off = np.zeros(4)
        self.delay_mode, self.delay_known, self.d_hat = delay_mode, delay_known, 0.0
        self.imu = deque(maxlen=100)    # (gyro, accel) of the latest steps, for rewind
        if kind == "cam_pose_drift":
            self.nb = 13
            self.P = np.block([[P_base, np.zeros((9, 4))], [np.zeros((4, 9)), cam.off_init_std ** 2 * np.eye(4)]])
            if delay_mode == "estimated":
                self.nb = 14
                self.P = np.block([[self.P, np.zeros((13, 1))], [np.zeros((1, 13)), np.full((1, 1), d_init_std ** 2)]])

    def predict(self, gyro, accel):
        super().predict(gyro, accel)
        self.imu.append((gyro, accel))
        if self.kind == "cam_pose_drift":
            ds = np.linalg.norm(self.v) * self.cfg.dt
            self.P[9:12, 9:12] += self.cam.off_pos_q * ds * np.eye(3)
            self.P[12, 12] += self.cam.yaw_drift_std ** 2 * ds

    def correct(self, delta):
        super().correct(delta)
        if self.kind == "cam_pose_drift":
            self.b_off = self.b_off - delta[9:13]
            if self.delay_mode == "estimated":
                self.d_hat -= delta[13]

    def delay_now(self):
        """Delay (s) the filter assumes for the next reading (estimate clipped to [-0.2, 1] s)."""
        return self.delay_known if self.delay_mode == "known" else float(np.clip(self.d_hat, -0.2, 1.0))

    def delayed_measurement(self, cam_k):
        """cam_pose_drift residual and Jacobian for a reading taken delay_now()
        ago: the body's pose then is the current estimate rewound through the
        IMU readings; the Jacobian on the current error is the one at that
        time times error_transition_inverse; for "estimated" the column of d is
        the rate of the predicted reading, minus the body's rotation and
        velocity then (the prediction moves back along the trajectory as d
        grows). Returns None if the IMU history is shorter than the lag."""
        y_p, y_R, _ = cam_k
        c, lag = self.cam, self.delay_now()
        back = rewind(self.R, self.v, self.p, self.imu, lag, self.cfg.dt)
        if back is None:
            return None
        R_then, v_then, p_then, gyro = back
        Rz_hat = rot_z(self.b_off[3])
        r_R = so3_log(Rz_hat @ R_then @ y_R.T)
        r_p = p_then + self.b_off[:3] - y_p
        H = np.zeros((6, len(self.P)))
        H_then = np.zeros((6, 9))
        H_then[0:3, 0:3] = np.eye(3)
        H_then[3:6, 0:3] = -skew(p_then)
        H_then[3:6, 6:9] = np.eye(3)
        H[:, 0:9] = H_then @ error_transition_inverse(lag)
        H[0:3, 12] = [0.0, 0.0, 1.0]
        H[3:6, 9:12] = np.eye(3)
        if self.delay_mode == "estimated":
            H[0:3, 13] = -Rz_hat @ R_then @ gyro
            H[3:6, 13] = -v_then
        N = np.diag([c.rot_std ** 2] * 3 + [c.pos_std ** 2] * 3)
        return np.concatenate([r_R, r_p]), H, N

    def camera_measurement(self, cam_k):
        """Residual r ~ H xi (+ noise) and noise covariance N for this
        filter's kind; cam_k = (p, R, v) readings of one tick."""
        if self.kind == "cam_pose_drift" and self.delay_mode != "ignore":
            return self.delayed_measurement(cam_k)
        y_p, y_R, y_v = cam_k
        n, c = len(self.P), self.cam
        if self.kind == "cam_vel":
            H = np.zeros((3, n))
            H[:, 3:6] = -np.eye(3)
            return self.R @ y_v - self.v, H, c.vel_std ** 2 * np.eye(3)
        H = np.zeros((6, n))
        Rz_hat = rot_z(self.b_off[3])
        r_R = so3_log(Rz_hat @ self.R @ y_R.T)
        r_p = self.p + self.b_off[:3] - y_p
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 0:3] = -skew(self.p)
        H[3:6, 6:9] = np.eye(3)
        if self.kind == "cam_pose_drift":
            H[0:3, 12] = [0.0, 0.0, 1.0]
            H[3:6, 9:12] = np.eye(3)
        N = np.diag([c.rot_std ** 2] * 3 + [c.pos_std ** 2] * 3)
        return np.concatenate([r_R, r_p]), H, N

    def camera_update(self, cam_k):
        """Returns True if the reading was used."""
        meas = self.camera_measurement(cam_k)
        if meas is None:
            return False
        r, H, N = meas
        S = H @ self.P @ H.T + N
        if self.cam_gating and r @ np.linalg.solve(S, r) > GATE[len(r)]:
            return False
        K = np.linalg.solve(S, H @ self.P).T
        self.correct(K @ r)
        I_KH = np.eye(len(self.P)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ N @ K.T
        return True


def make_filter(kind, trial, cfg, cam, gating, delay_mode="ignore", delay_known=0.0, d_init_std=0.1):
    X0 = sek3_make(trial.R[0], [trial.v[0], trial.p[0]])
    Xh = sek3_exp(trial.xi0) @ X0
    return CameraFusionInEKF(Xh[0:3, 0:3], Xh[0:3, 3], Xh[0:3, 4], initial_covariance(cfg), cfg, kind, cam, gating,
                             delay_mode, delay_known, d_init_std)


def run_filter(kind, trial, camera, gait, cfg, cam, gating=False, delay_mode="ignore", delay_known=0.0,
               d_init_std=0.1):
    """Per tick: pos_err, vel_err (N, 3), yaw_err (N,), nees_pos (3 DoF,
    world-coordinate covariance), nees_yaw, nees_vel, and used (N,) camera
    readings used. For delay_mode "estimated" also d_hat and d_std (N,), the
    delay estimate and its std (s)."""
    stance, y = trial.feet[gait]
    ekf = make_filter(kind, trial, cfg, cam, gating, delay_mode, delay_known, d_init_std)
    n_pts = len(stance)
    out = dict(pos_err=np.zeros((n_pts, 3)), vel_err=np.zeros((n_pts, 3)), yaw_err=np.zeros(n_pts),
               nees_pos=np.zeros(n_pts), nees_yaw=np.zeros(n_pts), nees_vel=np.zeros(n_pts),
               used=np.zeros(n_pts, dtype=bool))
    if delay_mode == "estimated":
        out["d_hat"], out["d_std"] = np.zeros(n_pts), np.zeros(n_pts)
    for k in range(n_pts):
        if k > 0:
            ekf.predict(trial.gyro_meas[k - 1], trial.accel_meas[k - 1])
        ekf.step(stance[k], y[k])
        if kind != "legs" and camera.ok[k]:
            out["used"][k] = ekf.camera_update((camera.p[k], camera.R[k], camera.v[k]))
        T = world_from_invariant(ekf.v, ekf.p)
        Pw = T @ ekf.P[:9, :9] @ T.T
        e_p, e_v = ekf.p - trial.p[k], ekf.v - trial.v[k]
        e_yaw = so3_log(ekf.R @ trial.R[k].T)[2]
        out["pos_err"][k], out["vel_err"][k], out["yaw_err"][k] = e_p, e_v, e_yaw
        out["nees_pos"][k] = e_p @ np.linalg.solve(Pw[6:9, 6:9], e_p)
        out["nees_vel"][k] = e_v @ np.linalg.solve(Pw[3:6, 3:6], e_v)
        out["nees_yaw"][k] = e_yaw ** 2 / Pw[2, 2]
        if delay_mode == "estimated":
            out["d_hat"][k], out["d_std"][k] = ekf.d_hat, np.sqrt(ekf.P[13, 13])
    return out


def run_monte_carlo(cfg, cam, n_trials, rng, kinds, gating_modes=(False,), gait="tripod"):
    """Per (kind, gating): per-tick averages of nees_pos, nees_yaw, nees_vel,
    mse_pos (summed over axes), mse_yaw, the fraction of available camera
    readings used before and after the jump, and lockouts: the number of runs
    that used under 5% of the available readings after the jump."""
    n_pts = cfg.n_steps + 1
    keys = [(kd, gm) for kd in kinds for gm in (gating_modes if kd != "legs" else (False,))]
    acc = {key: dict(nees_pos=np.zeros(n_pts), nees_yaw=np.zeros(n_pts), nees_vel=np.zeros(n_pts),
                     mse_pos=np.zeros(n_pts), mse_yaw=np.zeros(n_pts), used_before=[0, 0], used_after=[0, 0],
                     lockouts=0)
           for key in keys}
    for _ in range(n_trials):
        trial = simulate_trial(cfg, [gait], rng)
        camera = Camera(trial, cfg, rng, cam)
        for kind, gating in keys:
            o = run_filter(kind, trial, camera, gait, cfg, cam, gating)
            a = acc[(kind, gating)]
            for name in ("nees_pos", "nees_yaw", "nees_vel"):
                a[name] += o[name]
            a["mse_pos"] += np.sum(o["pos_err"] ** 2, axis=1)
            a["mse_yaw"] += o["yaw_err"] ** 2
            if kind != "legs":
                jt = camera.jump_tick if camera.jump_tick is not None else n_pts
                before, after = camera.ok.copy(), camera.ok.copy()
                before[jt:], after[:jt] = False, False
                a["used_before"][0] += int(o["used"][before].sum())
                a["used_before"][1] += int(before.sum())
                a["used_after"][0] += int(o["used"][after].sum())
                a["used_after"][1] += int(after.sum())
                if camera.jump_tick is not None and o["used"][after].sum() < 0.05 * after.sum():
                    a["lockouts"] += 1
    for a in acc.values():
        for name in ("nees_pos", "nees_yaw", "nees_vel", "mse_pos", "mse_yaw"):
            a[name] /= n_trials
    return acc


DELAY_MODES = ("ignore", "known", "estimated")
DELAY_LABELS = {"ignore": "ignore the delay", "known": "delay known", "estimated": "delay estimated (state)"}


def run_delay_monte_carlo(cfg, cam, n_trials, rng, delays, d_init_std):
    """The delay experiment on cam_pose_drift (no jump or dropout): per delay
    and delay_mode, per-tick averages as in run_monte_carlo. Per trial one
    truth and one set of camera draws; each delay changes only the time at
    which the camera samples the body (every delay's Camera is built from the
    same random state, the delay-0 one from `rng` itself, so the trials match
    run_monte_carlo's). "known" at delay 0 is "ignore" (not run again).
    Arguments:
        cfg: Config, cam: camera settings (cam.delay is overwritten), n_trials, rng
        delays: true delays (s), the first must be 0
        d_init_std: prior std of the delay state for "estimated" (s)
    Returns:
        acc: {(delay, mode): dict} with nees_pos, nees_yaw, nees_vel, mse_pos, mse_yaw (N,),
            and for "estimated" d_hat, d_std (n_trials, N); and for every delay
            shift_pos (m), shift_tilt, shift_yaw (rad), RMS shift of the reading against the
            delay-0 one (same noise), and time (s per tick) per mode."""
    n_pts = cfg.n_steps + 1
    acc = {}
    for d in delays:
        for mode in DELAY_MODES:
            if mode == "known" and d == 0.0:
                continue
            acc[(d, mode)] = dict(nees_pos=np.zeros(n_pts), nees_yaw=np.zeros(n_pts), nees_vel=np.zeros(n_pts),
                                  mse_pos=np.zeros(n_pts), mse_yaw=np.zeros(n_pts), time=0.0)
            if mode == "estimated":
                acc[(d, mode)].update(d_hat=np.zeros((n_trials, n_pts)), d_std=np.zeros((n_trials, n_pts)))
        acc[(d, "shift")] = dict(pos=0.0, tilt=0.0, yaw=0.0)
    for t in range(n_trials):
        trial = simulate_trial(cfg, ["tripod"], rng)
        state = copy.deepcopy(rng.bit_generator.state)
        cams = {}
        for d in delays:
            g = rng
            if d != delays[0]:
                g = np.random.Generator(np.random.PCG64())
                g.bit_generator.state = copy.deepcopy(state)
            cams[d] = Camera(trial, cfg, g, replace_delay(cam, d))
        for d in delays:
            ok = cams[d].ok[::10]
            sh = acc[(d, "shift")]
            sh["pos"] += np.mean(np.sum((cams[d].p - cams[delays[0]].p)[::10][ok] ** 2, axis=1)) / n_trials
            dR = np.array([so3_log(cams[d].R[k] @ cams[delays[0]].R[k].T) for k in range(0, n_pts, 10)])[ok]
            sh["tilt"] += np.mean(np.sum(dR[:, :2] ** 2, axis=1)) / n_trials
            sh["yaw"] += np.mean(dR[:, 2] ** 2) / n_trials
            for mode in DELAY_MODES:
                if (d, mode) not in acc:
                    continue
                t0 = time.perf_counter()
                o = run_filter("cam_pose_drift", trial, cams[d], "tripod", cfg, cam, False, mode, d, d_init_std)
                a = acc[(d, mode)]
                a["time"] += (time.perf_counter() - t0) / n_pts / n_trials
                for name in ("nees_pos", "nees_yaw", "nees_vel"):
                    a[name] += o[name] / n_trials
                a["mse_pos"] += np.sum(o["pos_err"] ** 2, axis=1) / n_trials
                a["mse_yaw"] += o["yaw_err"] ** 2 / n_trials
                if mode == "estimated":
                    a["d_hat"][t], a["d_std"][t] = o["d_hat"], o["d_std"]
    for d in delays:
        sh = acc[(d, "shift")]
        sh["pos"], sh["tilt"], sh["yaw"] = np.sqrt(sh["pos"]), np.sqrt(sh["tilt"]), np.sqrt(sh["yaw"])
        if d == 0.0:
            acc[(d, "known")] = acc[(d, "ignore")]
    return acc


def replace_delay(cam, delay):
    """Copy of the camera settings with the true delay (s) set."""
    c = copy.copy(cam)
    c.delay = delay
    return c


def convergence_time(d_hat, d_true, dt, tol=0.005):
    """Median over trials of the first time after which |d_hat - d_true| stays below tol.
    Arguments:
        d_hat: (n_trials, N) delay estimates (s), d_true: true delay (s), dt: tick (s), tol: bound (s)
    Returns: time (s), or nan if fewer than half of the trials get there"""
    times = []
    for tr in d_hat:
        bad = np.flatnonzero(np.abs(tr - d_true) >= tol)
        times.append((bad[-1] + 1) * dt if len(bad) else 0.0)
        if len(bad) and bad[-1] == len(tr) - 1:
            times[-1] = np.inf
    times = np.array(times)
    return float(np.median(times)) if np.mean(np.isfinite(times)) >= 0.5 else float("nan")


IDENT_CASES = (
    ("default walk", dict()),
    ("no body bob", dict(tilt_amp=0.0, heave_amp=0.0)),
    ("constant velocity, no rotation", dict(tilt_amp=0.0, heave_amp=0.0, lin_accel_std=0.0, ang_accel_std=0.0)),
    ("constant velocity and turn rate", dict(tilt_amp=0.0, heave_amp=0.0, lin_accel_std=0.0, ang_accel_std=0.0,
                                             turn=True)))


def run_identifiability(cfg, cam, n_trials, rng, delay, d_init_std, turn_rate):
    """Delay estimate of the "estimated" filter under motions with less and
    less variation: per case, the final delay error of each trial and the
    final std of the delay estimate.
    Arguments:
        cfg, cam: as in run_delay_monte_carlo, delay: true delay (s), n_trials, rng,
        d_init_std: prior std of the delay (s), turn_rate: initial turn rate (rad/s) of the cases with "turn"
    Returns:
        {case name: (final errors (n_trials,) s, final std (n_trials,) s)}"""
    out = {}
    for name, kw in IDENT_CASES:
        kw = dict(kw)
        c = replace(cfg, **kw, twist0=(cfg.twist0[0], 0.0, turn_rate)) if kw.pop("turn", False) else replace(cfg, **kw)
        errs, stds = [], []
        for _ in range(n_trials):
            trial = simulate_trial(c, ["tripod"], rng)
            cam_d = Camera(trial, c, rng, replace_delay(cam, delay))
            o = run_filter("cam_pose_drift", trial, cam_d, "tripod", c, cam, False, "estimated", delay, d_init_std)
            errs.append(o["d_hat"][-1] - delay)
            stds.append(o["d_std"][-1])
        out[name] = (np.array(errs), np.array(stds))
    return out


def summarize(a, skip):
    """Run averages after `skip` ticks: position NEES, yaw NEES, velocity NEES,
    position RMSE (m), yaw RMSE (deg), and final position RMSE (m)."""
    return (a["nees_pos"][skip:].mean(), a["nees_yaw"][skip:].mean(), a["nees_vel"][skip:].mean(),
            np.sqrt(a["mse_pos"][skip:].mean()), np.degrees(np.sqrt(a["mse_yaw"][skip:].mean())),
            np.sqrt(a["mse_pos"][-1]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation length (s)")
    parser.add_argument("--speed", type=float, default=0.2, help="Initial forward speed (m/s)")
    parser.add_argument("--turn-rate", type=float, default=0.0, help="Initial turn rate (rad/s)")
    parser.add_argument("--pos-std", type=float, default=0.005, help="Camera position noise per axis (m) (assumed)")
    parser.add_argument("--rot-std", type=float, default=0.005, help="Camera rotation noise per axis (rad) (assumed)")
    parser.add_argument("--vel-std", type=float, default=0.02, help="Camera velocity noise per axis (m/s) (assumed)")
    parser.add_argument("--scale-std", type=float, default=0.01,
                        help="Camera scale error std, one draw per run (0.01 = 1%% of the displacement)")
    parser.add_argument("--yaw-drift-std", type=float, default=0.005,
                        help="Camera yaw drift random walk (rad per sqrt(m travelled)) (assumed)")
    parser.add_argument("--off-pos-q", type=float, default=None,
                        help="cam_pose_drift: position-offset random walk (m^2 per m travelled); "
                             "default scale_std^2 x speed x duration")
    parser.add_argument("--off-init-std", type=float, default=1e-3, help="cam_pose_drift: initial offset std")
    parser.add_argument("--jump-pos-std", type=float, default=0.1, help="Pose jump std per horizontal axis (m) (assumed)")
    parser.add_argument("--jump-yaw-deg", type=float, default=0.0,
                        help="Pose jump yaw std (deg); 0 matches t265.md (pose jumps change translation only)")
    parser.add_argument("--drop-len", type=float, default=3.0, help="Dropout window length (s)")
    parser.add_argument("--n-mc-trials", type=int, default=30, help="Monte Carlo trials per experiment")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")
    parser.add_argument("--delay-experiment", action="store_true",
                        help="Run only the camera-delay experiment (cam_pose_drift, no jump or dropout) instead of "
                             "experiments 1 and 2")
    parser.add_argument("--delays-ms", type=float, nargs="+", default=[0.0, 10.0, 30.0, 100.0, 200.0],
                        help="Delay experiment: true camera delays (ms), a sweep, not a T265 value; the first must be 0")
    parser.add_argument("--delay-duration", type=float, default=20.0,
                        help="Delay experiment: simulation length (s) (shorter than --duration, for run time)")
    parser.add_argument("--delay-turn-rate", type=float, default=0.2,
                        help="Delay experiment: initial turn rate (rad/s) of the turning walk "
                             "(hexapod_contact_inekf's default)")
    parser.add_argument("--delay-init-std-ms", type=float, default=100.0,
                        help="Delay experiment: prior std of the delay state (ms) (design choice)")
    parser.add_argument("--ident-trials", type=int, default=10,
                        help="Delay experiment: Monte Carlo trials of the identifiability cases")
    return parser


def camera_settings(args, failures):
    """Namespace of the camera settings used by Camera and the filters."""
    dur = args.duration
    return argparse.Namespace(
        pos_std=args.pos_std, rot_std=args.rot_std, vel_std=args.vel_std, scale_std=args.scale_std,
        yaw_drift_std=args.yaw_drift_std, off_init_std=args.off_init_std,
        off_pos_q=args.off_pos_q if args.off_pos_q is not None else args.scale_std ** 2 * args.speed * dur,
        failures=failures, jump_pos_std=args.jump_pos_std, jump_yaw_std=np.radians(args.jump_yaw_deg),
        jump_window=(0.2 * dur, 0.6 * dur), drop_window=(0.6 * dur, 0.8 * dur), drop_len=args.drop_len,
        delay=0.0)


def delay_main(args):
    """The delay experiment: sweep of true delays x {ignore, known, estimated},
    on a straight and on a turning walk, then the identifiability cases."""
    if args.delays_ms[0] != 0.0:
        raise SystemExit("--delays-ms must start with 0")
    delays = [x / 1000.0 for x in args.delays_ms]
    d_init = args.delay_init_std_ms / 1000.0
    d = Config()
    n = args.n_mc_trials
    a2 = argparse.Namespace(**vars(args))
    a2.duration = args.delay_duration
    cam = camera_settings(a2, False)
    skip = int(round(1.0 / d.dt))
    bands = {k: averaged_nees_bounds(k, n) for k in (1, 3)}
    print(f"Consistent run averages ({n} trials): 3-DoF NEES in [{bands[3][0]:.2f}, {bands[3][1]:.2f}], "
          f"yaw (1 DoF) in [{bands[1][0]:.2f}, {bands[1][1]:.2f}]; averages leave out the first second")
    print(f"Delay experiment: cam_pose_drift, no jump or dropout, {args.delay_duration:g} s, tripod, delays "
          f"{', '.join(f'{x:g}' for x in args.delays_ms)} ms (a sweep), prior std of the delay state "
          f"{args.delay_init_std_ms:g} ms")
    scenarios = (("straight walk", 0.0), (f"turning walk ({args.delay_turn_rate:g} rad/s)", args.delay_turn_rate))
    results = {}
    for e, (name, turn) in enumerate(scenarios):
        cfg = replace(d, n_steps=int(round(args.delay_duration / d.dt)), twist0=(args.speed, 0.0, turn))
        t0 = time.perf_counter()
        results[e] = run_delay_monte_carlo(cfg, cam, n, np.random.default_rng(args.seed + 100 * e), delays, d_init)
        print(f"\n{name} ({time.perf_counter() - t0:.0f} s)")
        print("  reading shifted by the delay (RMS over the run, same noise): "
              + "; ".join(f"{1000 * x:g} ms: {1000 * results[e][(x, 'shift')]['pos']:.1f} mm, "
                          f"roll/pitch {np.degrees(results[e][(x, 'shift')]['tilt']):.2f} deg, "
                          f"yaw {np.degrees(results[e][(x, 'shift')]['yaw']):.3f} deg" for x in delays[1:]))
        for x in delays:
            for mode in DELAY_MODES:
                a = results[e][(x, mode)]
                npos, nyaw, nvel, rpos, ryaw, rend = summarize(a, skip)
                print(f"  delay {1000 * x:5.0f} ms  {mode:<9s} pos NEES {npos:8.2f}  yaw NEES {nyaw:7.2f}  "
                      f"vel NEES {nvel:5.2f} | pos RMSE {100 * rpos:6.2f} cm (end {100 * rend:6.2f})  "
                      f"yaw RMSE {ryaw:5.2f} deg | {1000 * a['time']:.2f} ms/tick")
        for x in delays:
            a = results[e][(x, "estimated")]
            err = a["d_hat"][:, -1] - x
            tc = convergence_time(a["d_hat"], x, d.dt)
            print(f"  delay {1000 * x:5.0f} ms  estimated: final error mean {1000 * err.mean():6.2f} ms, std "
                  f"{1000 * err.std(ddof=1):5.2f} ms, filter std at the end {1000 * a['d_std'][:, -1].mean():5.2f} ms"
                  f", median time to stay within 5 ms {tc:.1f} s")
            peak = a["d_hat"].mean(axis=0)
            late = int(round(5.0 / d.dt))
            print(f"{'':20s}mean estimate peaks at {1000 * peak.max():.0f} ms ({peak.argmax() * d.dt:.2f} s); "
                  f"position NEES after 5 s: known {results[e][(x, 'known')]['nees_pos'][late:].mean():.2f}, "
                  f"estimated {a['nees_pos'][late:].mean():.2f}")
    cfg = replace(d, n_steps=int(round(args.delay_duration / d.dt)), twist0=(args.speed, 0.0, 0.0))
    dd = 0.1
    print(f"\nIdentifiability of the delay ({args.ident_trials} trials, true delay {1000 * dd:g} ms, "
          f"{args.delay_duration:g} s; error = estimate - truth at the end)")
    ident = run_identifiability(cfg, cam, args.ident_trials, np.random.default_rng(args.seed + 200), dd, d_init,
                                args.delay_turn_rate)
    for name, (err, std) in ident.items():
        print(f"  {name:<34s} final error mean {1000 * err.mean():7.2f} ms, std {1000 * err.std(ddof=1):6.2f} ms, "
              f"RMS {1000 * np.sqrt(np.mean(err ** 2)):6.2f} ms | filter std {1000 * std.mean():6.2f} ms "
              f"(prior {args.delay_init_std_ms:g})")

    t = np.arange(cfg.n_steps + 1) * d.dt
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    cols = {"ignore": "tab:red", "known": "tab:green", "estimated": "tab:blue"}
    for e, ls in ((0, "-"), (1, "--")):
        for mode in DELAY_MODES:
            sums = [summarize(results[e][(x, mode)], skip) for x in delays]
            kw = dict(color=cols[mode], ls=ls, marker="o", ms=4, label=f"{DELAY_LABELS[mode]}, {scenarios[e][0].split(' (')[0]}")
            axs[0, 0].plot(args.delays_ms, [100 * q[3] for q in sums], **kw)
            axs[0, 1].plot(args.delays_ms, [q[0] for q in sums], **kw)
            axs[0, 2].plot(args.delays_ms, [q[4] for q in sums], **kw)
    axs[0, 1].set_yscale("log")
    axs[0, 1].axhspan(*bands[3], color="0.85")
    axs[0, 1].axhline(3, color="black", lw=0.8)
    axs[0, 0].set_ylabel(f"position RMSE (cm, {n} trials)")
    axs[0, 1].set_ylabel(f"position NEES ({n} trials, band: 95%, consistent)")
    axs[0, 2].set_ylabel("yaw RMSE (deg)")
    for ax in axs[0]:
        ax.set_xlabel("true camera delay (ms)")
    axs[0, 0].legend(fontsize=6)
    axs[0, 0].text(0.02, 0.5, "solid: straight walk, dashed: turning walk", transform=axs[0, 0].transAxes, fontsize=7)
    for e, ax in ((0, axs[1, 0]), (1, axs[1, 1])):
        for i, x in enumerate(delays[1:]):
            a = results[e][(x, "estimated")]
            line, = ax.plot(t, 1000 * a["d_hat"].mean(axis=0), label=f"true {1000 * x:g} ms")
            ax.axhline(1000 * x, color=line.get_color(), lw=0.8, ls=":")
            ax.fill_between(t, 1000 * (a["d_hat"].mean(axis=0) - a["d_std"].mean(axis=0)),
                            1000 * (a["d_hat"].mean(axis=0) + a["d_std"].mean(axis=0)), color=line.get_color(), alpha=0.12)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("delay estimate (ms): mean over trials, +- filter std")
        ax.set_title(f"estimated delay, {scenarios[e][0]} (first 5 s)")
    if len(delays) > 1:
        axs[1, 0].legend(fontsize=7)
    names = list(ident)
    xs = np.arange(len(names))
    axs[1, 2].bar(xs - 0.2, [1000 * ident[k][1].mean() for k in names], 0.4, color="tab:blue", label="filter std")
    axs[1, 2].bar(xs + 0.2, [1000 * np.sqrt(np.mean(ident[k][0] ** 2)) for k in names], 0.4, color="tab:red",
                  label="RMS error over trials")
    axs[1, 2].axhline(args.delay_init_std_ms, color="black", lw=0.8, ls=":", label="prior std")
    axs[1, 2].set_yscale("log")
    axs[1, 2].set_xticks(xs, [k.replace(", ", ",\n").replace(" and ", " and\n") for k in names], fontsize=7)
    axs[1, 2].set_ylabel("delay, end of the run (ms)")
    axs[1, 2].set_title(f"is the delay identifiable? (true {1000 * dd:g} ms, {args.ident_trials} trials)")
    axs[1, 2].legend(fontsize=7)
    for ax in axs[1, :2]:
        ax.set_xlim(0, 5)
    axs[0, 0].set_title("position error")
    axs[0, 1].set_title("position consistency")
    axs[0, 2].set_title("yaw error")
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=120)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


def main():
    args = build_parser().parse_args()
    if args.delay_experiment:
        delay_main(args)
        return
    d = Config()
    cfg = replace(d, n_steps=int(round(args.duration / d.dt)), twist0=(args.speed, 0.0, args.turn_rate))
    skip = int(round(1.0 / cfg.dt))
    n = args.n_mc_trials
    bands = {k: averaged_nees_bounds(k, n) for k in (1, 3)}
    print(f"Consistent run averages ({n} trials): 3-DoF NEES in [{bands[3][0]:.2f}, {bands[3][1]:.2f}], "
          f"yaw (1 DoF) in [{bands[1][0]:.2f}, {bands[1][1]:.2f}]; averages leave out the first second")

    results = {}
    for e, (failures, gm) in enumerate([(False, (False,)), (True, (False, True))]):
        cam = camera_settings(args, failures)
        title = "Experiment 2: one pose jump + one dropout per run" if failures else "Experiment 1: drift and noise only"
        print(f"\n{title} ({args.duration:g} s, tripod; position-offset random walk {cam.off_pos_q:.2e} m^2/m)")
        results[e] = run_monte_carlo(cfg, cam, n, np.random.default_rng(args.seed + 100 * e), KINDS, gm)
        for (kind, gating), a in results[e].items():
            npos, nyaw, nvel, rpos, ryaw, rend = summarize(a, skip)
            used = ""
            if kind != "legs" and failures:
                ub, ua = a["used_before"], a["used_after"]
                used = (f" | readings used before / after the jump {100 * ub[0] / ub[1]:.0f}% / {100 * ua[0] / ua[1]:.0f}%"
                        f", locked out in {a['lockouts']} of {n} runs")
            print(f"  {kind:<15s} {'gated' if gating else '     '}  pos NEES {npos:7.2f}  yaw NEES {nyaw:7.2f}  "
                  f"vel NEES {nvel:5.2f} | pos RMSE {100 * rpos:6.2f} cm (end {100 * rend:6.2f})  "
                  f"yaw RMSE {ryaw:5.2f} deg{used}")

    t = np.arange(cfg.n_steps + 1) * cfg.dt
    colors = {"legs": "black", "cam_vel": "tab:green", "cam_pose": "tab:red", "cam_pose_drift": "tab:blue"}
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for (kind, gating), a in results[0].items():
        axs[0, 0].plot(t, 100 * np.sqrt(a["mse_pos"]), color=colors[kind], label=LABELS[kind])
        axs[0, 1].plot(t, a["nees_pos"], color=colors[kind], label=LABELS[kind])
    for (kind, gating), a in results[1].items():
        if kind == "legs":
            continue
        ls = "-" if gating else "--"
        axs[1, 0].plot(t, 100 * np.sqrt(a["mse_pos"]), color=colors[kind], ls=ls,
                       label=f"{LABELS[kind]}, {'gated' if gating else 'not gated'}")
        axs[1, 1].plot(t, a["nees_pos"], color=colors[kind], ls=ls)
    for ax in axs[:, 1]:
        ax.axhspan(*bands[3], color="0.85")
        ax.axhline(3, color="black", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylabel(f"position NEES ({n} trials, 3 DoF)")
    for ax in axs[:, 0]:
        ax.set_ylabel(f"position RMSE (cm, {n} trials)")
    for ax in axs.ravel():
        ax.set_xlabel("time (s)")
    axs[0, 0].set_title("Experiment 1: drift and noise only")
    axs[0, 1].set_title("Experiment 1: position consistency (band: 95%, consistent)")
    axs[1, 0].set_title("Experiment 2: one pose jump + one dropout per run")
    axs[1, 1].set_title("Experiment 2: position consistency")
    axs[0, 0].legend(fontsize=8)
    axs[1, 0].legend(fontsize=7)
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=120)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
