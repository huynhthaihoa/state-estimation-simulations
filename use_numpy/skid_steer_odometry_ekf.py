'''
Wheel odometry of a skid-steer robot (Clearpath Husky A200) whose wheels
slip in turns, with and without a gyro, and with the slip as a filter state.

A skid-steer robot turns by driving its left and right wheels at different
speeds, and its wheels must skid sideways to do it. It turns slower than
plain differential drive predicts. The ICR model of skid steering (Wang et
al. 2015, after Mandow et al. 2007) captures this, in its symmetric case, as
a wider effective track:

    v = (v_l + v_r) / 2,    omega = (v_r - v_l) / (chi * B),    chi >= 1

with B the track and chi the ICR coefficient (chi = 1: no slip).

Husky numbers (all sourced; see docs/filtering/skid_steer_odometry_ekf.md):
  - track B = 0.555 m (husky_description URDF; the user manual's effective
    track); max speed 1 m/s (user manual);
  - wheel encoders 78,000 ticks per meter (user manual), read at 50 Hz (the
    driver's diff_drive_controller publish_rate);
  - the driver multiplies the track by wheel_separation_multiplier = 1.875
    (husky_control control.yaml; the file does not say why, so reading it as
    a slip factor is an inference);
  - gyro: Xsens MTi-30 (the IMU on the CTU lab's Husky, Hulchuk et al. 2023),
    noise density 0.03 deg/s/sqrt(Hz), bias repeatability 0.2 deg/s (MTi
    10-series leaflet).

Assumptions (approved, stated in the doc): the true chi is constant per run
and swept (default 1.875); no sideways slip (symmetric ICRs); the wheel
radius is exact; the gyro has a constant turn-on bias of std 0.2 deg/s (the
18 deg/h in-run drift is left out); the path (a straight, a full left loop, a straight, a full right loop)
runs at 0.5 m/s.

Rows:
  - odo_chi1: dead reckoning with chi = 1 (plain differential drive);
  - odo_driver: dead reckoning with the driver's chi = 1.875;
  - ekf_fixed: EKF on wheels + gyro, chi fixed at 1.875;
  - ekf_chi: the same EKF with chi as a state (online calibration).

EKF state [x, y, theta, omega, b] (+ chi): the forward speed (wheel average)
drives the position; the yaw rate omega is a random walk; each tick two
measurements, gyro = omega + b and wheels (v_r - v_l) / B = chi * omega.
On straight parts (omega ~ 0) the gyro shows its bias; in turns the wheels
show chi.
'''

import argparse
from dataclasses import dataclass, replace

import numpy as np
import matplotlib.pyplot as plt

from state_estimation_simulations.utils import averaged_nees_bounds, nees  # noqa: E402

TRACK = 0.555                 # m, husky_description URDF "track"
TICKS_PER_M = 78000.0         # Husky user manual, wheel encoders
DT = 0.02                     # s, control.yaml publish_rate 50 Hz
DRIVER_CHI = 1.875            # control.yaml wheel_separation_multiplier
P3AT_CHI = (1.4115, 1.4662)   # Wang et al. 2015, Table 3 (Pioneer P3-AT)
GYRO_ND = np.radians(0.03)    # rad/s/sqrt(Hz), MTi 10-series leaflet (typ)
GYRO_BIAS = np.radians(0.2)   # rad/s, MTi 10-series bias repeatability
ROWS = ("odo_chi1", "odo_driver", "ekf_fixed", "ekf_chi")


@dataclass
class Config:
    chi: float = DRIVER_CHI       # true ICR coefficient
    speed: float = 0.5            # m/s
    duration: float = 120.0       # s, two opposite loops of duration / 2
    straight: float = 8.0         # s of straight driving before each loop
    ramp: float = 2.0             # s, raised-cosine ramp in and out of a turn
    chi_prior: float = 1.5        # ekf_chi prior mean
    chi_prior_std: float = 0.3    # ekf_chi prior std
    yaw_accel_std: float = 0.1    # omega random walk, rad/s per sqrt(s) (filter tuning)
    white_wheel_noise: bool = False  # check: white wheel noise of the modeled size instead of encoder rounding


def yaw_rate_profile(cfg):
    """Two opposite loops: per half, a straight part then one full loop
    (+2 pi, then -2 pi), with raised-cosine ramps; returns omega at each tick."""
    n_half = int(round(cfg.duration / 2 / DT))
    t = np.arange(n_half) * DT
    turn = t - cfg.straight
    T = cfg.duration / 2 - cfg.straight
    shape = np.where(turn < 0, 0.0, 1.0)
    up = (turn >= 0) & (turn < cfg.ramp)
    down = (turn > T - cfg.ramp)
    shape[up] = 0.5 * (1 - np.cos(np.pi * turn[up] / cfg.ramp))
    shape[down] = 0.5 * (1 - np.cos(np.pi * (T - turn[down]) / cfg.ramp))
    omega0 = 2 * np.pi / (shape.sum() * DT)
    half = omega0 * shape
    return np.concatenate([half, -half])


@dataclass
class Trial:
    x: np.ndarray        # (n+1, 3) true x, y, theta
    v_l: np.ndarray      # (n,) measured wheel speeds (quantized encoders)
    v_r: np.ndarray
    gyro: np.ndarray     # (n,) measured yaw rate
    bias: float


def simulate_trial(cfg, rng):
    omega = yaw_rate_profile(cfg)
    n = len(omega)
    v = np.full(n, cfg.speed)
    vr_true = v + omega * cfg.chi * TRACK / 2
    vl_true = v - omega * cfg.chi * TRACK / 2
    x = np.zeros((n + 1, 3))
    for k in range(n):
        th = x[k, 2] + 0.5 * omega[k] * DT
        x[k + 1] = x[k] + DT * np.array([v[k] * np.cos(th), v[k] * np.sin(th), omega[k]])

    def encoder(vs):
        ticks = np.floor((np.concatenate([[0.0], np.cumsum(vs * DT)]) + rng.uniform(0, 1 / TICKS_PER_M)) * TICKS_PER_M)
        return np.diff(ticks) / TICKS_PER_M / DT

    if cfg.white_wheel_noise:
        sd = np.sqrt(wheel_speed_var())
        v_l, v_r = vl_true + rng.normal(size=n) * sd, vr_true + rng.normal(size=n) * sd
    else:
        v_l, v_r = encoder(vl_true), encoder(vr_true)
    bias = rng.normal() * GYRO_BIAS
    gyro = omega + bias + rng.normal(size=n) * GYRO_ND / np.sqrt(DT)
    return Trial(x, v_l, v_r, gyro, bias)


def wheel_speed_var():
    """Per-side speed variance from the encoder step (difference of two
    roundings): 2 (step / DT)^2 / 12."""
    step = 1.0 / TICKS_PER_M / DT
    return 2 * step ** 2 / 12


def dead_reckoning(trial, chi_hat):
    n = len(trial.v_l)
    x = np.zeros((n + 1, 3))
    for k in range(n):
        v = 0.5 * (trial.v_l[k] + trial.v_r[k])
        w = (trial.v_r[k] - trial.v_l[k]) / (chi_hat * TRACK)
        th = x[k, 2] + 0.5 * w * DT
        x[k + 1] = x[k] + DT * np.array([v * np.cos(th), v * np.sin(th), w])
    return x


class SkidSteerEKF:
    """State [x, y, theta, omega, b] (+ chi if estimate_chi)."""

    def __init__(self, cfg, estimate_chi):
        self.estimate_chi = estimate_chi
        nx = 6 if estimate_chi else 5
        self.x = np.zeros(nx)
        self.P = np.diag([1e-8, 1e-8, 1e-8, 1e-2, GYRO_BIAS ** 2] + ([cfg.chi_prior_std ** 2] if estimate_chi else []))
        self.chi_fixed = DRIVER_CHI
        if estimate_chi:
            self.x[5] = cfg.chi_prior
        self.q_omega = cfg.yaw_accel_std ** 2 * DT
        self.r_v = wheel_speed_var()
        self.r_gyro = GYRO_ND ** 2 / DT
        self.r_wheel = 2 * self.r_v / TRACK ** 2

    @property
    def chi(self):
        return self.x[5] if self.estimate_chi else self.chi_fixed

    def predict(self, v):
        x, y, th, w = self.x[:4]
        thm = th + 0.5 * w * DT
        c, s = np.cos(thm), np.sin(thm)
        self.x[0] += DT * v * c
        self.x[1] += DT * v * s
        self.x[2] += DT * w
        F = np.eye(len(self.x))
        F[0, 2], F[1, 2] = -DT * v * s, DT * v * c
        F[0, 3], F[1, 3] = -0.5 * DT ** 2 * v * s, 0.5 * DT ** 2 * v * c
        F[2, 3] = DT
        Q = np.zeros_like(self.P)
        G = np.array([DT * c, DT * s])          # position from the speed noise
        Q[:2, :2] = np.outer(G, G) * self.r_v / 2
        Q[3, 3] = self.q_omega
        self.P = F @ self.P @ F.T + Q

    def measurement(self):
        """Predicted (gyro, wheel yaw reading (v_r - v_l) / B) and Jacobian."""
        w, b = self.x[3], self.x[4]
        h = np.array([w + b, self.chi * w])
        H = np.zeros((2, len(self.x)))
        H[0, 3], H[0, 4] = 1.0, 1.0
        H[1, 3] = self.chi
        if self.estimate_chi:
            H[1, 5] = w
        return h, H

    def update(self, gyro, dv):
        z = np.array([gyro, dv / TRACK])
        h, H = self.measurement()
        S = H @ self.P @ H.T + np.diag([self.r_gyro, self.r_wheel])
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ (z - h)
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ np.diag([self.r_gyro, self.r_wheel]) @ K.T


def run_ekf(trial, cfg, estimate_chi):
    f = SkidSteerEKF(cfg, estimate_chi)
    n = len(trial.v_l)
    xs = np.zeros((n + 1, len(f.x)))
    Ps = np.zeros((n + 1, len(f.x), len(f.x)))
    xs[0], Ps[0] = f.x, f.P
    for k in range(n):
        f.update(trial.gyro[k], trial.v_r[k] - trial.v_l[k])
        f.predict(0.5 * (trial.v_l[k] + trial.v_r[k]))
        xs[k + 1], Ps[k + 1] = f.x, f.P
    return xs, Ps


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_trial(cfg, rng):
    trial = simulate_trial(cfg, rng)
    out = {}
    for row, chi_hat in (("odo_chi1", 1.0), ("odo_driver", DRIVER_CHI)):
        out[row] = dict(x=dead_reckoning(trial, chi_hat))
    for row, est in (("ekf_fixed", False), ("ekf_chi", True)):
        xs, Ps = run_ekf(trial, cfg, est)
        out[row] = dict(x=xs[:, :3], P=Ps[:, :3, :3], full=xs, Pfull=Ps)
    for r in out.values():
        e = trial.x - r["x"]
        e[:, 2] = wrap(e[:, 2])
        r["err"] = e
        if "P" in r:
            r["nees"] = np.array([nees(e[k], r["P"][k]) for k in range(len(e))])
    return trial, out


def run_monte_carlo(cfg, n_trials, seed):
    """Per row: position error norm (trials x ticks), heading error, NEES."""
    rng = np.random.default_rng(seed)
    acc = {r: dict(pos=[], yaw=[], nees=[], chi=[], chi_std=[], bias_err=[]) for r in ROWS}
    example = None
    for i in range(n_trials):
        trial, out = run_trial(cfg, rng)
        if example is None:
            example = (trial, out)
        for r in ROWS:
            o = out[r]
            acc[r]["pos"].append(np.linalg.norm(o["err"][:, :2], axis=1))
            acc[r]["yaw"].append(o["err"][:, 2])
            if "nees" in o:
                acc[r]["nees"].append(o["nees"])
                acc[r]["bias_err"].append(trial.bias - o["full"][-1, 4])
            if r == "ekf_chi":
                acc[r]["chi"].append(o["full"][:, 5])
                acc[r]["chi_std"].append(np.sqrt(o["Pfull"][:, 5, 5]))
    return {r: {k: np.array(v) for k, v in a.items()} for r, a in acc.items()}, example


def summarize(acc, cfg):
    first = int(round(1.0 / DT))
    dist = cfg.speed * cfg.duration
    res = {}
    for r in ROWS:
        a = acc[r]
        s = dict(pos_rmse=np.sqrt(np.mean(a["pos"][:, first:] ** 2)),
                 max_pct=100 * np.sqrt(np.mean(a["pos"].max(axis=1) ** 2)) / dist,
                 yaw_rmse=np.degrees(np.sqrt(np.mean(a["yaw"][:, first:] ** 2))))
        if len(a["nees"]):
            s["nees"] = np.mean(a["nees"][:, first:])
        if len(a["chi"]):
            s["chi_end"] = np.mean(a["chi"][:, -1])
            s["chi_rmse"] = np.sqrt(np.mean((a["chi"][:, -1] - cfg.chi) ** 2))
            s["chi_std"] = np.mean(a["chi_std"][:, -1])
        res[r] = s
    return res


def print_rows(res):
    for r in ROWS:
        s = res[r]
        line = f"  {r:<11} pos RMSE {100 * s['pos_rmse']:7.2f} cm  worst error {s['max_pct']:6.2f} % of distance  yaw RMSE {s['yaw_rmse']:7.2f} deg"
        if "nees" in s:
            line += f" | NEES {s['nees']:7.2f}"
        if "chi_end" in s:
            line += f" | chi at end {s['chi_end']:.3f} (RMSE {s['chi_rmse']:.3f}, reported std {s['chi_std']:.3f})"
        print(line)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--chi", type=float, default=DRIVER_CHI, help="True ICR coefficient (assumed; default = the driver's multiplier)")
    p.add_argument("--speed", type=float, default=0.5, help="Forward speed (m/s) (assumed)")
    p.add_argument("--duration", type=float, default=120.0, help="Run length (s) (assumed)")
    p.add_argument("--sweep-chi", type=float, nargs="+", default=[1.0, 1.25, 1.47, 1.875, 2.0],
                   help="True chi values for experiment 2")
    p.add_argument("--chi-prior", type=float, default=1.5, help="ekf_chi prior mean (design choice)")
    p.add_argument("--chi-prior-std", type=float, default=0.3, help="ekf_chi prior std (design choice)")
    p.add_argument("--white-wheel-noise", action="store_true",
                   help="Check: white wheel-speed noise of the modeled size instead of encoder rounding")
    p.add_argument("--n-mc-trials", type=int, default=30)
    p.add_argument("--n-sweep-trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    return p


def main(args=None):
    args = build_parser().parse_args(args)
    cfg = Config(chi=args.chi, speed=args.speed, duration=args.duration, chi_prior=args.chi_prior,
                 chi_prior_std=args.chi_prior_std, white_wheel_noise=args.white_wheel_noise)
    lo3, hi3 = averaged_nees_bounds(3, args.n_mc_trials)
    print(f"Consistent run average ({args.n_mc_trials} trials): 3-DoF NEES (x, y, heading) in [{lo3:.2f}, {hi3:.2f}]; "
          "averages leave out the first second")
    omega = yaw_rate_profile(cfg)
    print(f"Path: straight, left loop, straight, right loop; {cfg.speed} m/s, {cfg.duration:.0f} s ({cfg.speed * cfg.duration:.0f} m), "
          f"loop yaw rate {omega.max():.3f} rad/s (radius {cfg.speed / omega.max():.2f} m)")

    print(f"\nExperiment 1: true chi = {cfg.chi} ({args.n_mc_trials} trials)")
    acc, example = run_monte_carlo(cfg, args.n_mc_trials, args.seed)
    res = summarize(acc, cfg)
    print_rows(res)
    print(f"  bias error at the end (deg/s RMS): ekf_fixed {np.degrees(np.sqrt(np.mean(acc['ekf_fixed']['bias_err'] ** 2))):.4f}, "
          f"ekf_chi {np.degrees(np.sqrt(np.mean(acc['ekf_chi']['bias_err'] ** 2))):.4f} (prior {np.degrees(GYRO_BIAS):.2f})")

    print(f"\nExperiment 2: sweep of the true chi ({args.n_sweep_trials} trials each)")
    lo3s, hi3s = averaged_nees_bounds(3, args.n_sweep_trials)
    print(f"  ({args.n_sweep_trials} trials: consistent NEES in [{lo3s:.2f}, {hi3s:.2f}])")
    sweep = {}
    for i, chi in enumerate(args.sweep_chi):
        c = replace(cfg, chi=chi)
        a, _ = run_monte_carlo(c, args.n_sweep_trials, args.seed + 100 + i)
        sweep[chi] = summarize(a, c)
        print(f" true chi = {chi}")
        print_rows(sweep[chi])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    trial, out = example
    colors = dict(odo_chi1="tab:red", odo_driver="tab:orange", ekf_fixed="tab:green", ekf_chi="tab:blue")
    labels = dict(odo_chi1="wheels, chi = 1", odo_driver="wheels, chi = 1.875 (driver)",
                  ekf_fixed="EKF wheels + gyro, chi = 1.875", ekf_chi="EKF wheels + gyro, chi as a state")
    ax = axes[0, 0]
    ax.plot(trial.x[:, 0], trial.x[:, 1], "k", lw=2, label="truth")
    for r in ROWS:
        ax.plot(out[r]["x"][:, 0], out[r]["x"][:, 1], color=colors[r], label=labels[r])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)")
    ax.set_title(f"One run, true chi = {cfg.chi}")
    ax.legend(fontsize=8)
    t = np.arange(acc["ekf_chi"]["pos"].shape[1]) * DT
    ax = axes[0, 1]
    for r in ROWS:
        ax.plot(t, 100 * np.sqrt(np.mean(acc[r]["pos"] ** 2, axis=0)), color=colors[r], label=labels[r])
    ax.set_yscale("log")
    ax.set_xlabel("time (s)"), ax.set_ylabel(f"position RMSE (cm, {args.n_mc_trials} trials)")
    ax.set_title("Experiment 1: position error")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    chis = acc["ekf_chi"]["chi"]
    ax.plot(t, chis.mean(axis=0), color="tab:blue", label="ekf_chi: mean estimate")
    ax.fill_between(t, chis.mean(axis=0) - 2 * acc["ekf_chi"]["chi_std"].mean(axis=0),
                    chis.mean(axis=0) + 2 * acc["ekf_chi"]["chi_std"].mean(axis=0), color="tab:blue", alpha=0.2,
                    label="+- 2 reported std")
    ax.axhline(cfg.chi, color="k", ls="--", label="truth")
    ax.set_xlabel("time (s)"), ax.set_ylabel("chi")
    ax.set_title("Experiment 1: learning the slip coefficient")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    xs = list(sweep)
    for r in ROWS:
        ax.plot(xs, [100 * sweep[c][r]["pos_rmse"] for c in xs], "o", ms=8, color=colors[r], label=labels[r])
    ax.axvspan(*P3AT_CHI, color="0.85", label="P3-AT measured chi (Wang 2015)")
    ax.axvline(DRIVER_CHI, color="0.5", ls=":", label="Husky driver multiplier")
    ax.set_yscale("log")
    ax.set_xlabel("true chi"), ax.set_ylabel(f"position RMSE (cm, {args.n_sweep_trials} trials)")
    ax.set_title("Experiment 2: sweep of the true slip")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
