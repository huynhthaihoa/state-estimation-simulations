'''
Follow-up to gait_phase_ekf.py: the same hexapod, the same uncertain
touchdowns, but the contact detector reads joint torque instead of servo
position error, as on HEBI's series-elastic X-Series actuators (Daisy).

gait_phase_ekf.py models its position-error detector by its outputs only (a
delay and a false-trigger rate, both assumed). Here the torque detector is
modeled one level down, as a signal: the vertical foot force estimated from
the joint torques, f = -(J^T)^-1 (tau - h(q, dq)) (Camurri et al. 2017),
sampled at 1 kHz (HEBI feedback rate), made of

  - load: 0 before the true touchdown, then a linear ramp to the tripod load
    share (body weight / 3) over `load_ramp` s (assumed: how fast the
    series-elastic leg takes the weight).
  - model error: the leg dynamics that h(q, dq) misses, as one half-sine
    pulse per step over the first half of the swing-down (the window where
    gait_phase_ekf.py puts its false triggers), with a normal amplitude of
    std `model_err_std` N (assumed; its sign is random).
  - noise: white torque noise of std `torque_noise_std` Nm, mapped to the
    vertical force through Daisy's stance Jacobian (4.27 N per Nm).

Daisy numbers: geometry and stance from hexapod_joint_kinematics_inekf.py
(HEBI's Daisy HRDF); body mass 31 kg (HEBI hex_config.xml "body mass");
torque resolution 0.01 Nm and 1 kHz feedback (HEBI X-Series datasheet / docs).
HEBI publishes no torque noise, model error or load-ramp values; the
defaults are assumptions and are swept.

Torque trigger: the force rises above `torque_threshold` x the load share
(re-armed below half of that), available at the next filter tick. Rows:

  - schedule, detector, cpg, phase_ekf, oracle: gait_phase_ekf.py's, on its
    position-error triggers (unchanged).
  - torque_detector: the first torque trigger of the step.
  - torque_phase_ekf: gait_phase_ekf.py's phase EKF on the torque triggers
    (known mean delay = the noise-free delay, ramp x threshold, in ticks).
  - torque_fused: a simplified version of the idea of Bledt et al. 2018:
    contact log-odds = gait-phase prior (touchdown predicted by
    torque_phase_ekf, std sqrt(P_lag + terrain^2)) + force evidence, a
    logistic in the force (Camurri et al. 2017) fitted on one separate
    training trial; stance starts at the first sample above 0.5.

Draw order per trial: everything gait_phase_ekf.simulate_trial draws, then per
step one pulse amplitude and the noise samples (standard normals, scaled
after, so the sweeps reuse the same draws).
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, logit
from scipy.stats import norm

from state_estimation_simulations.utils import averaged_nees_bounds, nees  # noqa: E402
from state_estimation_simulations.use_numpy import gait_phase_ekf as g  # noqa: E402
from state_estimation_simulations.use_numpy import hexapod_joint_kinematics_inekf as daisy  # noqa: E402
from state_estimation_simulations.use_numpy.hexapod_gait_ekf import GAITS, LEG_NAMES, run_ekf, state_errors  # noqa: E402

SAMPLE_DT = 1e-3            # HEBI feedback rate, 1 kHz
GRAVITY = 9.81
TORQUE_ROWS = ("torque_detector", "torque_phase_ekf", "torque_fused")
ROWS = g.ROWS + TORQUE_ROWS


def force_per_torque():
    """Std of the vertical foot-force estimate per 1 Nm of white torque noise
    (same std on the three joints), at Daisy's nominal stance, f = -(J^T)^-1 tau.
    The largest over the six legs (they differ by < 1%)."""
    k = 0.0
    for leg in range(6):
        J = daisy.leg_jacobian(daisy.stance_ik(leg), leg)
        k = max(k, np.linalg.norm(np.linalg.inv(J.T)[2]))
    return k


def step_windows(steps, td, swing_ticks):
    """Per step, (first tick of the signal window, start-of-swing-down tick,
    next liftoff tick): the window runs from the start of the swing-down (or
    the true touchdown, if earlier) to the next liftoff."""
    return [(min(d - swing_ticks // 3, k), d - swing_ticks // 3, nxt) for (_, _, d, nxt), k in zip(steps, td)]


def draw_torque(windows, dt, rng):
    """Standard-normal draws per step: (pulse amplitude, noise samples)."""
    out = []
    for k0, _, nxt in windows:
        n = int(round((nxt - k0) * dt / SAMPLE_DT))
        out.append((rng.normal(), rng.normal(size=n)))
    return out


def force_signals(windows, td, dt, swing_ticks, draws, share, ramp, model_err_std, sigma_f):
    """Estimated vertical foot force per step: (sample times (s), force (N)).
    See the module docstring for the three parts."""
    pulse_len = (swing_ticks // 6) * dt
    out = []
    for (k0, kd, nxt), k_td, (a, n) in zip(windows, td, draws):
        t = k0 * dt + SAMPLE_DT * np.arange(len(n))
        load = share * np.clip((t - k_td * dt) / ramp, 0.0, 1.0) if ramp > 0 else share * (t >= k_td * dt)
        s = (t - kd * dt) / pulse_len
        pulse = np.where((s >= 0) & (s <= 1), np.sin(np.pi * np.clip(s, 0, 1)), 0.0)
        out.append((t, load + model_err_std * a * pulse + sigma_f * n))
    return out


def threshold_triggers(signals, steps, dt, threshold):
    """Torque triggers per step: ticks of each upward crossing of `threshold`
    (re-armed below threshold / 2), at the first tick at or after the
    crossing sample; crossings at or after the next liftoff are dropped."""
    out = []
    for (t, f), (_, _, _, nxt) in zip(signals, steps):
        trig, armed = [], True
        for ti, fi in zip(t, f):
            if armed and fi > threshold:
                k = int(np.ceil(ti / dt - 1e-9))
                if k < nxt:
                    trig.append(k)
                armed = False
            elif not armed and fi < 0.5 * threshold:
                armed = True
        out.append(sorted(set(trig)))
    return out


def fit_logistic(signals, td, dt, share, n_iter=50, l2=1e-3):
    """Contact probability P(contact | f) = expit(b1 f / share + b0), fitted
    by Newton's method (small L2 on b1) on every sample of `signals`, label
    = sample at or after the true touchdown. Returns (b0, b1, base rate)."""
    x = np.concatenate([f for _, f in signals]) / share
    y = np.concatenate([(t >= k * dt).astype(float) for (t, _), k in zip(signals, td)])
    b = np.zeros(2)
    X = np.column_stack([np.ones_like(x), x])
    for _ in range(n_iter):
        p = expit(X @ b)
        grad = X.T @ (p - y) + l2 * np.array([0.0, b[1]])
        Hs = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.diag([0.0, 1.0])
        b = b - np.linalg.solve(Hs, grad)
    return b[0], b[1], y.mean()


def starts_fused(signals, steps, dt, lag_est, lag_std, terrain_std, logistic, share):
    """First tick at or after the first sample whose fused contact
    probability exceeds 0.5 (None if none before the next liftoff). Prior:
    P(touchdown by t) = Phi((t - (d dt + lag)) / sqrt(lag_std^2 + terrain^2));
    evidence: the logistic's log-odds minus its training base-rate log-odds."""
    b0, b1, base = logistic
    starts = []
    for (t, f), (_, _, d, nxt), lag, s in zip(signals, steps, lag_est, lag_std):
        prior = np.clip(norm.cdf((t - (d * dt + lag)) / np.hypot(s, terrain_std)), 1e-12, 1 - 1e-12)
        post = logit(prior) + b0 + b1 * f / share - logit(base)
        hit = np.flatnonzero(post > 0.0)
        k = int(np.ceil(t[hit[0]] / dt - 1e-9)) if len(hit) else None
        starts.append(k if k is not None and k < nxt else None)
    return starts


class TorqueModel:
    """The torque-signal settings of one run (see build_parser)."""

    def __init__(self, args):
        self.share = args.body_mass * GRAVITY / 3.0
        self.k_f = force_per_torque()
        self.set(args.load_ramp, args.model_err_std, args.torque_noise_std, args.torque_threshold)

    def set(self, ramp, model_err_std, torque_noise_std, threshold_frac):
        self.ramp, self.model_err_std = ramp, model_err_std
        self.sigma_f = torque_noise_std * self.k_f
        self.threshold = threshold_frac * self.share
        return self

    def known_delay(self, dt):
        """Noise-free delay from touchdown to the trigger tick (s): the
        crossing at ramp x threshold fraction, rounded up to a tick."""
        return np.ceil(self.ramp * self.threshold / self.share / dt - 1e-9) * dt

    def signals(self, info, dt, swing_ticks, draws, windows):
        return force_signals(windows, info["td"], dt, swing_ticks, draws, self.share, self.ramp,
                             self.model_err_std, self.sigma_f)


def swing_ticks_of(args):
    return int(round((1.0 - GAITS[g.GAIT][0]) * args.cycle_period / args.dt))


def torque_trial(args, rng, model, logistic=None):
    """One trial: gait_phase_ekf.simulate_trial, then the torque draws and the
    torque rows' stance flags. Returns (x_true, x_init, r, z, flags, info)
    with info["torque"] = dict(signals, triggers, draws, windows)."""
    dt = args.dt
    xt, xi, r, z, flags, info = g.simulate_trial(args, rng)
    sw = swing_ticks_of(args)
    windows = step_windows(info["steps"], info["td"], sw)
    draws = draw_torque(windows, dt, rng)
    add_torque_rows(args, model, info, flags, draws, windows, logistic)
    return xt, xi, r, z, flags, info


def add_torque_rows(args, model, info, flags, draws, windows, logistic):
    """Fills flags[row] for the torque rows (logistic None: torque_fused left out)."""
    dt, steps = args.dt, info["steps"]
    sw = swing_ticks_of(args)
    sig = model.signals(info, dt, sw, draws, windows)
    trig = threshold_triggers(sig, steps, dt, model.threshold)
    delay = model.known_delay(dt)
    R = args.terrain_std ** 2 + dt ** 2 / 12.0
    P0 = np.diag([args.ekf_init_lag_std ** 2, args.ekf_init_rate_std ** 2])
    starts = {"torque_detector": [t[0] if t else None for t in trig]}
    starts["torque_phase_ekf"], _, est, std = g.starts_phase_ekf(steps, trig, dt, delay, R, args.ekf_rate_q, P0,
                                                                 args.ekf_gate)
    if logistic is not None:
        starts["torque_fused"] = starts_fused(sig, steps, dt, est, std, args.terrain_std, logistic, model.share)
    for a, s in starts.items():
        flags[a] = g.stance_from_starts(info["stance_cmd"], steps, s)
    info["torque"] = dict(signals=sig, triggers=trig, draws=draws, windows=windows)


def train_logistic(args, model, seed):
    """Fit the force logistic on one trial with its own seed (not reused)."""
    rng = np.random.default_rng(seed)
    _, _, _, _, _, info = torque_trial(args, rng, model)
    return fit_logistic(info["torque"]["signals"], info["td"], args.dt, model.share)


def false_trigger_rate(triggers, steps, td):
    """Fraction of steps with a trigger before the true touchdown."""
    return float(np.mean([any(k < t for k in trig) for trig, t in zip(triggers, td)]))


def run_monte_carlo(args, n_trials, rng, model, logistic, rows=ROWS):
    """gait_phase_ekf.run_monte_carlo for every row, position and torque.
    Returns dict row -> dict(nees (N,), rmse (6,), swing, missed), plus
    "false" -> fraction of steps with an early trigger per detector."""
    Q, R_meas, P_init = g.filter_setup(args)
    out, early = None, {"position": [], "torque": []}
    for _ in range(n_trials):
        xt, xi, rr, zz, fl, info = torque_trial(args, rng, model, logistic)
        n_pts = len(xt)
        if out is None:
            out = {a: dict(nees=np.zeros(n_pts), mse=np.zeros(6), swing=0.0, missed=0.0) for a in rows}
        early["position"].append(false_trigger_rate(info["triggers"], info["steps"], info["td"]))
        early["torque"].append(false_trigger_rate(info["torque"]["triggers"], info["steps"], info["td"]))
        for a in rows:
            xe, Pl, _ = run_ekf(xi, P_init, zz, rr, fl[a], args.dt, Q, R_meas)
            e = state_errors(xt, xe)
            o = out[a]
            o["mse"] += np.mean(e ** 2, axis=0)
            o["nees"] += [nees(e[k], Pl[k]) for k in range(n_pts)]
            sw, ms = g.flag_errors(fl[a], fl["oracle"])
            o["swing"] += sw
            o["missed"] += ms
    for o in out.values():
        o["nees"] /= n_trials
        o["rmse"] = np.sqrt(o.pop("mse") / n_trials)
        o["swing"] /= n_trials
        o["missed"] /= n_trials
    out["false"] = {k: float(np.mean(v)) for k, v in early.items()}
    return out


def run_sweep(args, n_trials, seed, train_seed, settings):
    """Flag scores only (no body EKF): for each setting (ramp, model_err_std,
    torque_noise_std), mean swing-read and missed ticks per trial of the
    torque rows, and the torque false-trigger rate. The same trials and draws
    for every setting (common random numbers); the force logistic is refitted
    per setting on the training trial `train_seed`. The position rows do not
    depend on the setting; their scores are returned once under None."""
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n_trials):
        _, _, _, _, flags, info = g.simulate_trial(args, rng)
        windows = step_windows(info["steps"], info["td"], swing_ticks_of(args))
        trials.append((flags, info, windows, draw_torque(windows, args.dt, rng)))
    res = {None: {a: np.mean([g.flag_errors(fl[a], fl["oracle"]) for fl, *_ in trials], axis=0)
                  for a in ("detector", "phase_ekf")}}
    res[None]["false"] = np.mean([false_trigger_rate(i["triggers"], i["steps"], i["td"]) for _, i, _, _ in trials])
    for st in settings:
        m = TorqueModel(args).set(st[0], st[1], st[2], args.torque_threshold)
        logistic = train_logistic(args, m, train_seed)
        sc = {a: [] for a in TORQUE_ROWS}
        fr = []
        for flags, info, windows, draws in trials:
            fl = dict(flags)
            add_torque_rows(args, m, info, fl, draws, windows, logistic)
            fr.append(false_trigger_rate(info["torque"]["triggers"], info["steps"], info["td"]))
            for a in TORQUE_ROWS:
                sc[a].append(g.flag_errors(fl[a], fl["oracle"]))
        res[tuple(st)] = {a: np.mean(v, axis=0) for a, v in sc.items()}
        res[tuple(st)]["false"] = float(np.mean(fr))
    return res


def build_parser():
    parser = g.build_parser()
    parser.description = __doc__
    parser.add_argument("--body-mass", type=float, default=31.0, help="Robot mass (kg); HEBI hex_config.xml")
    parser.add_argument("--load-ramp", type=float, default=0.05,
                        help="Time for a landed foot's load to reach its share (s) (assumption)")
    parser.add_argument("--model-err-std", type=float, default=15.0,
                        help="Std of the swing-down model-error pulse in the force estimate (N) (assumption)")
    parser.add_argument("--torque-noise-std", type=float, default=0.1,
                        help="White joint-torque noise std (Nm) (assumption; resolution is 0.01 Nm)")
    parser.add_argument("--torque-threshold", type=float, default=0.3,
                        help="Torque trigger threshold, as a fraction of the load share")
    parser.add_argument("--sweep-model-err", type=float, nargs="+", default=[0.0, 7.5, 15.0, 22.5, 30.0, 45.0],
                        help="Model-error stds to sweep (N)")
    parser.add_argument("--sweep-ramp", type=float, nargs="+", default=[0.01, 0.025, 0.05, 0.1, 0.2],
                        help="Load-ramp times to sweep (s)")
    parser.add_argument("--sweep-torque-noise", type=float, nargs="+", default=[0.01, 0.1, 1.0, 3.0],
                        help="Torque-noise stds to sweep (Nm)")
    parser.add_argument("--n-sweep-trials", type=int, default=30, help="Trials per sweep setting (flag scores only)")
    return parser


def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    model = TorqueModel(args)
    logistic = train_logistic(args, model, args.seed + 1000)
    print(f"Daisy: load share {model.share:.1f} N per foot (tripod, {args.body_mass:.0f} kg); "
          f"force noise {model.k_f:.2f} N per Nm of torque noise -> {model.sigma_f:.2f} N at "
          f"{args.torque_noise_std} Nm; threshold {model.threshold:.1f} N; known torque delay "
          f"{1e3 * model.known_delay(args.dt):.0f} ms")
    print(f"Force logistic (training trial, seed {args.seed + 1000}): P = expit({logistic[1]:.2f} f/share "
          f"{logistic[0]:+.2f}); 0.5 at {-logistic[0] / logistic[1] * model.share:.1f} N")

    *_, info = torque_trial(args, rng, model, logistic)

    print(f"\nRunning Monte Carlo ({args.n_mc_trials} trials)...")
    mc = run_monte_carlo(args, args.n_mc_trials, rng, model, logistic)
    lo, hi = averaged_nees_bounds(6, args.n_mc_trials)
    print(f"Steps with a trigger before the true touchdown: position {100 * mc['false']['position']:.1f}%, "
          f"torque {100 * mc['false']['torque']:.1f}%")
    print(f"\nMonte Carlo (consistent 6-DoF NEES ~6; 95% bounds for a {args.n_mc_trials}-trial average "
          f"[{lo:.2f}, {hi:.2f}]):")
    print(f"  {'':<17s} {'NEES':>8s} {'u RMSE':>10s} {'omega RMSE':>12s} {'swing read/trial':>18s} "
          f"{'missed/trial':>14s}")
    for a in ROWS:
        o = mc[a]
        print(f"  {a:<17s} {np.mean(o['nees']):8.2f} {o['rmse'][3]:10.4f} {o['rmse'][5]:12.4f} "
              f"{o['swing']:18.1f} {o['missed']:14.1f}")

    base = (args.load_ramp, args.model_err_std, args.torque_noise_std)
    sweeps = {"model_err": [(base[0], v, base[2]) for v in args.sweep_model_err],
              "ramp": [(v, base[1], base[2]) for v in args.sweep_ramp],
              "noise": [(base[0], base[1], v) for v in args.sweep_torque_noise]}
    labels = {"model_err": ("model-error std", "N", 1.0), "ramp": ("load ramp", "ms", 1e3),
              "noise": ("torque noise", "Nm", 1.0)}
    res = {}
    for name, settings in sweeps.items():
        res[name] = run_sweep(args, args.n_sweep_trials, args.seed + 2000, args.seed + 1000, settings)
        pos = res[name][None]
        lab, unit, sc = labels[name]
        print(f"\nSweep: {lab} ({args.n_sweep_trials} trials; swing read / missed ticks per trial; early-trigger "
              f"steps). Position: detector {pos['detector'][0]:.1f} / {pos['detector'][1]:.1f}, phase_ekf "
              f"{pos['phase_ekf'][0]:.1f} / {pos['phase_ekf'][1]:.1f}, early {100 * pos['false']:.1f}%")
        idx = 1 if name == "model_err" else (0 if name == "ramp" else 2)
        for st in settings:
            o = res[name][tuple(st)]
            print(f"  {lab} {sc * st[idx]:6.3g} {unit:<3s} " + "  ".join(
                f"{a} {o[a][0]:5.1f} / {o[a][1]:5.1f}" for a in TORQUE_ROWS) + f"  early {100 * o['false']:5.1f}%")

    # Figure
    dt = args.dt
    colors = {"detector": "tab:red", "phase_ekf": "tab:green", "torque_detector": "tab:orange",
              "torque_phase_ekf": "tab:olive", "torque_fused": "tab:blue"}
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    ax = axs[0, 0]
    sig, steps, td = info["torque"]["signals"], info["steps"], info["td"]
    j = int(np.argmax([a for a, _ in info["torque"]["draws"]]))
    t, f = sig[j]
    t0 = td[j] * dt
    ax.plot(1e3 * (t - t0), f, color="tab:orange", lw=1, label="estimated vertical foot force")
    ax.axhline(model.threshold, color="black", ls="--", lw=1, label="torque threshold")
    ax.axvline(0.0, color="gray", lw=1, label="true touchdown")
    for k in info["torque"]["triggers"][j]:
        ax.axvline(1e3 * (k * dt - t0), color="tab:red", ls=":", lw=1.5)
    ax.plot([], [], color="tab:red", ls=":", label="torque triggers (filter ticks)")
    ax.set_xlabel("time from true touchdown (ms)")
    ax.set_ylabel("force (N)")
    ax.set_title(f"Torque signal: the step with the largest upward model-error pulse (leg {LEG_NAMES[steps[j][0]]})",
                 fontsize=10)
    ax.legend(fontsize=8)

    for ax, name in ((axs[0, 1], "model_err"), (axs[1, 0], "ramp"), (axs[1, 1], "noise")):
        lab, unit, sc = labels[name]
        idx = 1 if name == "model_err" else (0 if name == "ramp" else 2)
        xs = [sc * st[idx] for st in sweeps[name]]
        r = res[name]
        for a in ("detector", "phase_ekf"):
            ax.axhline(r[None][a][0] + r[None][a][1], color=colors[a], ls="--", lw=1,
                       label=f"position {a}")
        for a in TORQUE_ROWS:
            ax.plot(xs, [r[tuple(st)][a][0] + r[tuple(st)][a][1] for st in sweeps[name]], "o-",
                    color=colors[a], label=a)
        ax.set_xlabel(f"{lab} ({unit})")
        ax.set_ylabel("wrong stance ticks per trial (swing read + missed)")
        if name == "noise":
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_title(f"Sweep: {lab}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
