'''
Servo-only odometry of a magnetic climbing inchworm (CTU MinchW) whose
anchored pad slides sideways on a wall, and what a slip flag, an external
position fix and a slip-bias state change.

MinchW (Pham 2025, CTU bachelor thesis) is a two-footed inchworm with 6
servos (3 per leg), about 0.49 m long, holding on with one hybrid magnet per
foot. It has no IMU and no external sensors: it "relies solely on
proprioceptive feedback from its servo motors". On a vertical wall its heavy
head makes it "slide laterally", toward one side; untextured silicone pads
"do not provide enough friction to fully prevent slippage".

The robot walks as an inchworm (the thesis's gait B): the rear foot holds
while the front foot extends by one step, then the front foot holds while
the rear foot closes up. At the end of each step both feet hold, and the
servo angles give the relative pose of the two feet through the leg
kinematics (FK). Chaining these gives odometry of the feet on the surface.

A slide of the holding pad moves the whole chain rigidly: no servo angle (or
torque, in a quasi-static model) changes, so servo-only odometry cannot see
it.

Surface coordinates are "unrolled": x along the path (along the floor, then
up the wall), y sideways, psi the heading about the surface normal. The
corner crossing between floor and wall is not modeled kinematically.

Sourced (see docs/filtering/magnetic_inchworm_odometry_ekf.md):
  - 6 servos, 3 per leg; about 0.49 m long; step length up to 0.3282 m
    (vertical and horizontal gait B), Pham 2025;
  - XM430-W350 / XM540-W270 position resolution 4096 counts per revolution
    (ROBOTIS e-manual); a reading is modeled as the true angle rounded to a
    count (uniform error over one count, std = count / sqrt(12));
  - the robot slides laterally on the wall, countered by turning a joint
    toward one side (Pham 2025, qualitative).
Assumed (stated in the doc): no slip on the floor; the slip is one-sided; the chain's geometry (yaw-pitch-pitch
per leg, h 0.05 m, leg links 0.20 m, body 0.09 m: 0.49 m stretched); 10
floor steps then 40 wall steps; a slip happens on a random fraction of wall
steps, with a one-sided (half-normal) size toward the heavy side; the Hall
slip flag (an experimental pad in the thesis) is perfect unless a miss rate is
set; an external position fix (like the thesis's PTI motion-tracking markers) of
std 1 mm every 10 steps; the filters' slip std 5 mm.

Rows (all start from the true pose; all use the same FK step readings):
  - servo_only: chains the FK steps; no slip noise;
  - wall_noise: adds the slip std on every wall step;
  - hall_flag: adds it only on steps the Hall pad flags;
  - ext_fix: hall_flag + the external position fix;
  - ext_fix_bias: ext_fix + a slip-bias state (mean sideways slide per
    flagged slip), learned from the fixes.
Control (skip with --no-control): experiment 1 again with each slide's side
drawn at random (same sizes), to show that the one-sided direction, not the
slide size, makes hall_flag overconfident.
'''

import argparse
from dataclasses import dataclass, replace

import numpy as np
import matplotlib.pyplot as plt

from state_estimation_simulations.utils import averaged_nees_bounds, nees  # noqa: E402

COUNT = 2 * np.pi / 4096      # rad, Dynamixel XM430 / XM540 resolution (ROBOTIS e-manual)
STEP_LEN = 0.3282             # m, gait B maximum step length (Pham 2025)
PAD_H = 0.05                  # m, foot to first pitch joint (assumed)
LEG = 0.20                    # m, leg link (assumed)
BODY = 0.09                   # m, body link (assumed): stretched length 2 * 0.20 + 0.09 = 0.49 m
D_MIN = 0.12                  # m, feet separation when closed (assumed); open = D_MIN + STEP_LEN
ROWS = ("servo_only", "wall_noise", "hall_flag", "ext_fix", "ext_fix_bias")


@dataclass
class Config:
    n_floor: int = 10             # floor steps (assumed)
    n_wall: int = 40              # wall steps (assumed)
    slip_rate: float = 0.3        # fraction of wall steps with a slip (assumed)
    slip_std: float = 0.005       # m, half-normal scale of a slip (assumed)
    miss_rate: float = 0.0        # Hall flag misses this fraction of slips (assumed)
    filter_slip_std: float = 0.005  # m, slip std the filters assume (design choice)
    fix_every: int = 10           # steps between external position fixes (assumed)
    fix_std: float = 0.001        # m, external fix std per axis (assumed)
    bias_prior_std: float = 0.005  # m, ext_fix_bias prior on the mean slide per slip
    slip_sign: str = "one_sided"  # "one_sided" (toward the heavy side) or "random" (control)


def fk(q):
    """Pose of foot B relative to foot A (x, y, psi) from the six joint
    angles [yaw A, pitch A1, pitch A2, pitch B2, pitch B1, yaw B]."""
    q1, q2, q3, q4, q5, q6 = q
    a, s = 0.0, np.zeros(2)                       # sagittal plane: (along, up)
    s = s + [0.0, PAD_H]
    for ang, length in ((q2, LEG), (q3, BODY), (q4, LEG)):
        a += ang
        s = s + length * np.array([np.cos(a), np.sin(a)])
    a += q5
    s = s + PAD_H * np.array([np.cos(a - np.pi / 2), np.sin(a - np.pi / 2)])
    d = s[0]
    return np.array([d * np.cos(q1), d * np.sin(q1), q1 + q6])


def ik(d):
    """Symmetric arch with the feet d apart, body parallel to the surface."""
    th = np.arccos((d - BODY) / (2 * LEG))
    return np.array([0.0, th, -th, -th, th, 0.0])


def fk_jacobian(q, h=1e-7):
    return np.column_stack([(fk(q + e) - fk(q - e)) / (2 * h) for e in np.eye(6) * h])


def se2_compose(x, u):
    c, s = np.cos(x[2]), np.sin(x[2])
    return np.array([x[0] + c * u[0] - s * u[1], x[1] + s * u[0] + c * u[1], x[2] + u[2]])


def se2_inverse(u):
    c, s = np.cos(u[2]), np.sin(u[2])
    return np.array([-c * u[0] - s * u[1], s * u[0] - c * u[1], -u[2]])


def step_plan(cfg):
    """Per step: (extend?, on the wall?). Extend: rear foot holds, front foot
    moves to D_MIN + STEP_LEN ahead; close: front foot holds, rear foot moves
    to D_MIN behind it."""
    n = cfg.n_floor + cfg.n_wall
    return [(k % 2 == 0, k >= cfg.n_floor) for k in range(n)]


@dataclass
class Trial:
    anchor: np.ndarray      # (n+1, 3) true pose of the holding foot before each step / at the end
    q_meas: np.ndarray      # (n, 6) servo readings at the end of each step
    slip: np.ndarray        # (n,) sideways slide of the holding pad during each step (m)
    flag: np.ndarray        # (n,) Hall flag
    fix: np.ndarray         # (n, 2) external position fix of the new holding foot (nan if none)


def simulate_trial(cfg, rng):
    plan = step_plan(cfg)
    n = len(plan)
    anchor = np.zeros((n + 1, 3))
    q_meas = np.zeros((n, 6))
    slip = np.zeros(n)
    flag = np.zeros(n, bool)
    fix = np.full((n, 2), np.nan)
    u_slip = rng.uniform(size=n)
    size = np.abs(rng.normal(size=n)) * cfg.slip_std
    u_miss = rng.uniform(size=n)
    q_noise = rng.uniform(-0.5, 0.5, (n, 6)) * COUNT
    fix_noise = rng.normal(size=(n, 2)) * cfg.fix_std
    sign = rng.choice([-1.0, 1.0], size=n) if cfg.slip_sign == "random" else np.ones(n)  # drawn last
    for k, (extend, wall) in enumerate(plan):
        a = anchor[k].copy()
        if wall and cfg.slip_std > 0 and u_slip[k] < cfg.slip_rate:
            slip[k] = sign[k] * size[k]           # one-sided: toward the heavy side, drawn as +y
            flag[k] = u_miss[k] >= cfg.miss_rate
            a[1] += slip[k]
        q_true = ik(D_MIN + STEP_LEN) if extend else ik(D_MIN)
        T_ab = fk(q_true)                         # foot B relative to foot A
        if extend:                                # A (rear) holds, B lands ahead
            rel = T_ab
        else:                                     # B (front) holds, A lands behind
            rel = se2_inverse(T_ab)
        anchor[k + 1] = se2_compose(a, rel)
        q_meas[k] = q_true + q_noise[k]
        if (k + 1) % cfg.fix_every == 0:
            fix[k] = anchor[k + 1, :2] + fix_noise[k]
    return Trial(anchor, q_meas, slip, flag, fix)


def run_filter(row, trial, cfg):
    """Returns the estimated pose of the holding foot after each step and its
    3x3 covariance."""
    plan = step_plan(cfg)
    has_bias = row == "ext_fix_bias"
    x = np.zeros(4 if has_bias else 3)
    P = np.diag([1e-12, 1e-12, 1e-12] + ([cfg.bias_prior_std ** 2] if has_bias else []))
    r_q = COUNT ** 2 / 12 * np.eye(6)
    xs, Ps = [x[:3].copy()], [P[:3, :3].copy()]
    for k, (extend, wall) in enumerate(plan):
# Predict: the holding foot is x[:3] with covariance P[:3, :3]
        slip_noise = (row == "wall_noise" and wall) or (row in ("hall_flag", "ext_fix", "ext_fix_bias")
                                                        and trial.flag[k])
        if slip_noise:
            if has_bias:
                x[1] += x[3]
                F = np.eye(4)
                F[1, 3] = 1.0
                P = F @ P @ F.T
            P[1, 1] += cfg.filter_slip_std ** 2
        q = trial.q_meas[k]
        T_ab, J = fk(q), fk_jacobian(q)
        if extend:
            rel, Jr = T_ab, J
        else:
            rel = se2_inverse(T_ab)
            Jr = np.column_stack([(se2_inverse(T_ab + e) - se2_inverse(T_ab - e)) / 2e-7 for e in np.eye(3) * 1e-7]) @ J
        c, s = np.cos(x[2]), np.sin(x[2])
        Jx = np.eye(len(x))
        Jx[0, 2] = -s * rel[0] - c * rel[1]
        Jx[1, 2] = c * rel[0] - s * rel[1]
        Ju = np.zeros((len(x), 3))
        Ju[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        x[:3] = se2_compose(x[:3], rel)
        P = Jx @ P @ Jx.T + Ju @ (Jr @ r_q @ Jr.T) @ Ju.T
# Update: the holding foot is x[:3] with covariance P[:3, :3]        
        if row in ("ext_fix", "ext_fix_bias") and not np.isnan(trial.fix[k, 0]):
            H = np.zeros((2, len(x)))
            H[0, 0] = H[1, 1] = 1.0
            R = cfg.fix_std ** 2 * np.eye(2)
            S = H @ P @ H.T + R
            K = np.linalg.solve(S, H @ P).T
            x = x + K @ (trial.fix[k] - H @ x)
            IKH = np.eye(len(x)) - K @ H
            P = IKH @ P @ IKH.T + K @ R @ K.T
        xs.append(x[:3].copy())
        Ps.append(P[:3, :3].copy())
    return np.array(xs), np.array(Ps)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_monte_carlo(cfg, n_trials, seed):
    rng = np.random.default_rng(seed)
    acc = {r: dict(pos=[], lat=[], nees=[]) for r in ROWS}
    example = None
    for _ in range(n_trials):
        trial = simulate_trial(cfg, rng)
        out = {}
        for r in ROWS:
            xs, Ps = run_filter(r, trial, cfg)
            e = trial.anchor - xs
            e[:, 2] = wrap(e[:, 2])
            acc[r]["pos"].append(np.linalg.norm(e[:, :2], axis=1))
            acc[r]["lat"].append(e[:, 1])
            acc[r]["nees"].append([nees(e[k], Ps[k]) for k in range(1, len(e))])
            out[r] = (xs, Ps)
        if example is None:
            example = (trial, out)
    return {r: {k: np.array(v) for k, v in a.items()} for r, a in acc.items()}, example


def summarize(acc, cfg):
    wall0 = cfg.n_floor + 1
    res = {}
    for r in ROWS:
        a = acc[r]
        res[r] = dict(pos_rmse=np.sqrt(np.mean(a["pos"][:, 1:] ** 2)),
                      end_lat=np.sqrt(np.mean(a["lat"][:, -1] ** 2)),
                      end_lat_mean=np.mean(a["lat"][:, -1]),
                      nees=np.mean(a["nees"]),
                      nees_wall=np.mean(a["nees"][:, wall0 - 1:]))
    return res


def print_rows(res):
    for r in ROWS:
        s = res[r]
        print(f"  {r:<13} pos RMSE {1000 * s['pos_rmse']:7.2f} mm | sideways error at the end: RMS {1000 * s['end_lat']:6.2f} mm, "
              f"mean {1000 * s['end_lat_mean']:+6.2f} mm | NEES {s['nees']:9.2f} (wall steps {s['nees_wall']:9.2f})")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--slip-rate", type=float, default=0.3, help="Fraction of wall steps with a slip (assumed)")
    p.add_argument("--slip-std", type=float, default=0.005, help="Half-normal scale of a slip, m (assumed)")
    p.add_argument("--miss-rate", type=float, default=0.0, help="Fraction of slips the Hall flag misses (assumed)")
    p.add_argument("--fix-every", type=int, default=10, help="Steps between external position fixes (assumed)")
    p.add_argument("--no-control", action="store_true",
                   help="Skip the control run with random-sign slides (same sizes, either side)")
    p.add_argument("--sweep-slip-std", type=float, nargs="+", default=[0.0, 0.001, 0.002, 0.005, 0.01])
    p.add_argument("--sweep-miss-rate", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5])
    p.add_argument("--sweep-fix-every", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--n-mc-trials", type=int, default=200)
    p.add_argument("--n-sweep-trials", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    return p


def main(args=None):
    args = build_parser().parse_args(args)
    cfg = Config(slip_rate=args.slip_rate, slip_std=args.slip_std, miss_rate=args.miss_rate, fix_every=args.fix_every)
    n_steps = cfg.n_floor + cfg.n_wall
    lo, hi = averaged_nees_bounds(3, args.n_mc_trials)
    print(f"{cfg.n_floor} floor + {cfg.n_wall} wall steps (one foot move each; each pair advances {STEP_LEN} m, "
          f"{n_steps // 2 * STEP_LEN:.2f} m in all); "
          f"encoder count {np.degrees(COUNT):.4f} deg; consistent 3-DoF NEES 3; at one step, the average over "
          f"{args.n_mc_trials} trials falls in [{lo:.2f}, {hi:.2f}] 95% of the time")

    print(f"\nExperiment 1: slips on {100 * cfg.slip_rate:.0f}% of wall steps, half-normal scale {1000 * cfg.slip_std:.0f} mm "
          f"({args.n_mc_trials} trials)")
    acc, example = run_monte_carlo(cfg, args.n_mc_trials, args.seed)
    res = summarize(acc, cfg)
    print_rows(res)

    if not args.no_control:
        print(f"\nControl: as experiment 1, but each slide goes to either side with equal chance "
              f"({args.n_mc_trials} trials)")
        a, _ = run_monte_carlo(replace(cfg, slip_sign="random"), args.n_mc_trials, args.seed)
        print_rows(summarize(a, cfg))

    sweeps = {}
    for name, values, field in (("slip size", args.sweep_slip_std, "slip_std"),
                                ("Hall miss rate", args.sweep_miss_rate, "miss_rate"),
                                ("fix interval", args.sweep_fix_every, "fix_every")):
        print(f"\nSweep: {name} ({args.n_sweep_trials} trials each)")
        sweeps[name] = {}
        for i, v in enumerate(values):
            c = replace(cfg, **{field: v})
            a, _ = run_monte_carlo(c, args.n_sweep_trials, args.seed + 100 * (len(sweeps) + 1) + i)
            sweeps[name][v] = summarize(a, c)
            print(f" {field} = {v}")
            print_rows(sweeps[name][v])

    colors = dict(servo_only="tab:red", wall_noise="tab:orange", hall_flag="tab:green", ext_fix="tab:blue",
                  ext_fix_bias="tab:purple")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    trial, out = example
    ax = axes[0, 0]
    ax.plot(trial.anchor[:, 1] * 1000, trial.anchor[:, 0], "k.-", lw=2, label="truth (holding foot)")
    for r in ROWS:
        ax.plot(out[r][0][:, 1] * 1000, out[r][0][:, 0], ".-", color=colors[r], label=r)
    ax.axhline(trial.anchor[cfg.n_floor, 0], color="0.6", ls=":", label="floor -> wall")
    ax.set_xlabel("sideways y (mm)"), ax.set_ylabel("along the path x (m)")
    ax.set_title("One run: holding-foot positions")
    ax.legend(fontsize=8)
    steps = np.arange(n_steps + 1)
    ax = axes[0, 1]
    for r in ROWS:
        ax.plot(steps, 1000 * np.sqrt(np.mean(acc[r]["lat"] ** 2, axis=0)), color=colors[r], label=r)
    ax.axvline(cfg.n_floor, color="0.6", ls=":")
    ax.set_xlabel("step"), ax.set_ylabel(f"sideways error RMS (mm, {args.n_mc_trials} trials)")
    ax.set_title("Experiment 1: sideways error")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    for r in ROWS:
        ax.plot(steps[1:], np.mean(acc[r]["nees"], axis=0), color=colors[r], label=r)
    ax.axhline(3.0, color="k", lw=0.8)
    ax.axvline(cfg.n_floor, color="0.6", ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("step"), ax.set_ylabel("NEES of (x, y, psi), trial average")
    ax.set_title("Experiment 1: consistency (3 = consistent)")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    sw = sweeps["slip size"]
    xs = list(sw)
    for r in ROWS:
        ax.plot([1000 * v for v in xs], [sw[v][r]["nees_wall"] for v in xs], "o", ms=7, color=colors[r], label=r)
    ax.axhline(3.0, color="k", lw=0.8)
    ax.axvline(1000 * cfg.filter_slip_std, color="0.6", ls=":", label="filters' slip std")
    ax.set_yscale("log")
    ax.set_xlabel("true slip scale (mm)"), ax.set_ylabel("NEES on wall steps")
    ax.set_title("Sweep: true slip size")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
