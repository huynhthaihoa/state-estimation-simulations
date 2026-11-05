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

Truth body, legs and IMU: hexapod_contact_inekf.simulate_trial, unchanged
(isotropic foot-position noise, no biases, true stance schedule), with a
straighter and faster default walk (0.2 m/s, no mean turn) so the run covers
some distance. Camera draws come after all of its draws.

Scores: physical position error (NEES with the covariance mapped to world
coordinates, and RMSE), yaw error (NEES, RMSE), velocity NEES, averaged after
the first second.
'''

import argparse
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from state_estimation_simulations.utils import averaged_nees_bounds  # noqa: E402
from state_estimation_simulations.use_numpy.lie_utils import sek3_exp, sek3_make, skew, so3_exp, so3_log  # noqa: E402
from state_estimation_simulations.use_numpy.hexapod_contact_inekf import (  # noqa: E402
    Config, InvariantContactEKF, initial_covariance, simulate_trial, world_from_invariant, rot_z)

KINDS = ("legs", "cam_vel", "cam_pose", "cam_pose_drift")
CAM_KINDS = KINDS[1:]
GATE = {3: 11.34, 6: 16.81}
LABELS = {"legs": "legs + IMU only", "cam_vel": "+ camera velocity", "cam_pose": "+ camera pose as world pose",
          "cam_pose_drift": "+ camera pose, drift as states"}


class Camera:
    """One run's camera readings (see the module docstring).
    Attributes (N = ticks): p (N, 3), R (N, 3, 3) pose readings, v (N, 3)
    body-frame velocity readings, ok (N,) False in the dropout window, and
    the truth drift: scale, yaw_drift (N,), jump_tick (or None)."""

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
        self.p = np.array([p0 + rot_z(psi[k]) @ ((1.0 + scale) * (trial.p[k] - p0)) for k in range(n)])
        self.p += off_p + n_p * cam.pos_std
        self.R = np.array([rot_z(psi[k]) @ so3_exp(n_R[k] * cam.rot_std) @ trial.R[k] for k in range(n)])
        self.v = np.array([(1.0 + scale) * trial.R[k].T @ trial.v[k] for k in range(n)]) + n_v * cam.vel_std
        self.ok = np.ones(n, dtype=bool)
        if cam.failures:
            k0 = int(cam.drop_window[0] / cfg.dt + u_drop * (cam.drop_window[1] - cam.drop_window[0]) / cfg.dt)
            self.ok[k0:k0 + int(round(cam.drop_len / cfg.dt))] = False
        self.scale = scale


class CameraFusionInEKF(InvariantContactEKF):
    """InvariantContactEKF plus camera updates (kind in KINDS). For
    cam_pose_drift the P layout is [rotation, velocity, position,
    b_p (3), b_psi (1), feet]; the offset error is zeta = b_hat - b."""

    def __init__(self, R, v, p, P_base, cfg, kind, cam, gating):
        super().__init__(R, v, p, P_base, cfg)
        self.kind, self.cam, self.cam_gating = kind, cam, gating
        self.b_off = np.zeros(4)
        if kind == "cam_pose_drift":
            self.nb = 13
            self.P = np.block([[P_base, np.zeros((9, 4))], [np.zeros((4, 9)), cam.off_init_std ** 2 * np.eye(4)]])

    def predict(self, gyro, accel):
        super().predict(gyro, accel)
        if self.kind == "cam_pose_drift":
            ds = np.linalg.norm(self.v) * self.cfg.dt
            self.P[9:12, 9:12] += self.cam.off_pos_q * ds * np.eye(3)
            self.P[12, 12] += self.cam.yaw_drift_std ** 2 * ds

    def correct(self, delta):
        super().correct(delta)
        if self.kind == "cam_pose_drift":
            self.b_off = self.b_off - delta[9:13]

    def camera_measurement(self, cam_k):
        """Residual r ~ H xi (+ noise) and noise covariance N for this
        filter's kind; cam_k = (p, R, v) readings of one tick."""
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
        r, H, N = self.camera_measurement(cam_k)
        S = H @ self.P @ H.T + N
        if self.cam_gating and r @ np.linalg.solve(S, r) > GATE[len(r)]:
            return False
        K = np.linalg.solve(S, H @ self.P).T
        self.correct(K @ r)
        I_KH = np.eye(len(self.P)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ N @ K.T
        return True


def make_filter(kind, trial, cfg, cam, gating):
    X0 = sek3_make(trial.R[0], [trial.v[0], trial.p[0]])
    Xh = sek3_exp(trial.xi0) @ X0
    return CameraFusionInEKF(Xh[0:3, 0:3], Xh[0:3, 3], Xh[0:3, 4], initial_covariance(cfg), cfg, kind, cam, gating)


def run_filter(kind, trial, camera, gait, cfg, cam, gating=False):
    """Per tick: pos_err, vel_err (N, 3), yaw_err (N,), nees_pos (3 DoF,
    world-coordinate covariance), nees_yaw, nees_vel, and used (N,) camera
    readings used."""
    stance, y = trial.feet[gait]
    ekf = make_filter(kind, trial, cfg, cam, gating)
    n_pts = len(stance)
    out = dict(pos_err=np.zeros((n_pts, 3)), vel_err=np.zeros((n_pts, 3)), yaw_err=np.zeros(n_pts),
               nees_pos=np.zeros(n_pts), nees_yaw=np.zeros(n_pts), nees_vel=np.zeros(n_pts),
               used=np.zeros(n_pts, dtype=bool))
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
    return parser


def camera_settings(args, failures):
    """Namespace of the camera settings used by Camera and the filters."""
    dur = args.duration
    return argparse.Namespace(
        pos_std=args.pos_std, rot_std=args.rot_std, vel_std=args.vel_std, scale_std=args.scale_std,
        yaw_drift_std=args.yaw_drift_std, off_init_std=args.off_init_std,
        off_pos_q=args.off_pos_q if args.off_pos_q is not None else args.scale_std ** 2 * args.speed * dur,
        failures=failures, jump_pos_std=args.jump_pos_std, jump_yaw_std=np.radians(args.jump_yaw_deg),
        jump_window=(0.2 * dur, 0.6 * dur), drop_window=(0.6 * dur, 0.8 * dur), drop_len=args.drop_len)


def main():
    args = build_parser().parse_args()
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
