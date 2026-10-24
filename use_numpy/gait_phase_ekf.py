'''
Follow-up to hexapod_gait_ekf.py, whose filters are told exactly when each
foot is on the ground. Here they are not: the planar hexapod walks a tripod
gait whose liftoffs are commanded (so known), but each foot lands at the
commanded touchdown time plus an unknown offset tau. Which feet the body
filter reads on each tick (the stance flags) must be decided from the
commanded schedule and a contact detector, and a wrong "planted" flag feeds
it a swinging foot's velocity.

The body, the foot readings and the body EKF are hexapod_gait_ekf.py's,
unchanged (a cascade: a touchdown-timing estimator produces stance flags,
the body EKF consumes them).

True touchdown offset of each foot, per step (tau, s; may be negative):

  - lag: a systematic lag shared by all feet, growing linearly from 0 to
    `lag_max` x the swing time over the run (e.g. a swing that slows under
    load).
  - terrain: an independent N(0, terrain_std^2) offset per foot and step
    (uneven ground: higher ground is touched early, lower ground late).
  - blocked leg: one leg lands `blocked_delay` late on every step whose
    commanded touchdown falls in [blocked_start, blocked_end).

A foot always swings at least `MIN_SWING_TICKS` and stands at least one tick
before its commanded liftoff.

Contact detector (modeled on the servo position-error detector of Faigl &
Cizek 2019; the rates are this script's assumptions, not the paper's):
each true touchdown is flagged after a delay N(det_delay, det_delay_std^2)
(>= 0, whole ticks) -- the estimators know the mean delay, not each draw.
With probability `false_rate` per step, a false trigger also fires early in
the swing-down, the last third of the swing (assumed), within its first half:
the start-of-swing-down error peak that a fixed threshold can mistake for
contact (Faigl & Cizek 2019). A detection that would come after the next
liftoff is lost.

Four ways to set the stance flags (stance always ends at the commanded
liftoff; it starts at a touchdown tick chosen as below):

  - schedule: the commanded touchdown (assumes tau = 0).
  - detector: the first trigger of the step.
  - cpg: per tripod group, a learned touchdown offset mu (the phase at which
    contact usually comes, as in Szadkowski et al. 2021, much simplified).
    Triggers earlier than mu - `cpg_window` are rejected as false; the first
    accepted one starts stance and nudges mu += cpg_gain * error. An error
    above `cpg_mistime` flags a mistiming (and is not learned from).
  - phase_ekf: per tripod group, a Kalman filter on [lag, lag rate] (s, s/s),
    one prediction per step and one measurement per detected touchdown
    (detection offset minus the known mean delay; noise = terrain spread +
    delay spread). A trigger more than `ekf_gate` sigma earlier than
    predicted is rejected; one more than `ekf_gate` sigma later is a
    mistiming (and is not used to update).

`oracle` (the true stance) is printed as a reference, not as an approach.
Scores: ticks where a swinging foot is read as planted, ticks of true stance
missed, body Monte Carlo NEES (6 DoF) and twist RMSE, mistiming hits and
false alarms, and the lag-estimate error of cpg and phase_ekf.
'''

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import averaged_nees_bounds, nees  # noqa: E402
from hexapod_gait_ekf import (  # noqa: E402
    LEG_NAMES, GAITS, stance_mask, simulate_body_trajectory, simulate_feet, process_noise_covariance,
    run_ekf, state_errors)

GAIT = "tripod"
APPROACHES = ("schedule", "detector", "cpg", "phase_ekf")
ROWS = APPROACHES + ("oracle",)
MIN_SWING_TICKS = 5
# Tripod groups: legs whose swing starts at phase 0.0, and at phase 0.5.
GROUPS = [[i for i, s in enumerate(GAITS[GAIT][1]) if s == 0.0],
          [i for i, s in enumerate(GAITS[GAIT][1]) if s == 0.5]]
GROUP_OF = {i: g for g, legs in enumerate(GROUPS) for i in legs}


def commanded_steps(stance_cmd):
    """Every commanded step after t = 0, per leg: (leg, previous liftoff tick,
    commanded touchdown tick, next liftoff tick), sorted by touchdown tick.
    The next liftoff is capped at the run length."""
    n_pts = len(stance_cmd)
    steps = []
    for i in range(6):
        col = stance_cmd[:, i].astype(int)
        downs = np.flatnonzero(np.diff(col) == 1) + 1
        ups = np.flatnonzero(np.diff(col) == -1) + 1
        for d in downs:
            prev_up = ups[ups < d].max() if (ups < d).any() else -1
            next_up = ups[ups > d].min() if (ups > d).any() else n_pts
            steps.append((i, int(prev_up), int(d), int(next_up)))
    return sorted(steps, key=lambda s: (s[2], s[0]))


def true_touchdowns(steps, dt, swing_time, duration, lag_max, terrain_std, blocked, rng):
    """True touchdown tick of every commanded step (see the module docstring).
    Draws one standard normal per step, whatever terrain_std is.
    Arguments:
        steps: from commanded_steps
        dt, swing_time, duration: tick, nominal swing time and run length (s)
        lag_max: final systematic lag as a fraction of the swing time
        terrain_std: per-step touchdown offset std (s)
        blocked: None, or (leg, t_start, t_end, delay_s)
        rng: numpy random number generator
    Returns:
        td: (n_steps,) true touchdown ticks
        tau: (n_steps,) true offsets (s), before clipping to whole ticks
        lag: (n_steps,) the systematic part of tau (s)
        is_blocked: (n_steps,) bool
    """
    n = len(steps)
    td, tau, lag, is_blocked = np.zeros(n, dtype=int), np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
    for j, (i, up, d, nxt) in enumerate(steps):
        t = d * dt
        lag[j] = lag_max * swing_time * t / duration
        tau[j] = lag[j] + terrain_std * rng.normal()
        if blocked is not None and blocked[0] == i and blocked[1] <= t < blocked[2]:
            tau[j] += blocked[3]
            is_blocked[j] = True
        td[j] = int(np.clip(d + int(round(tau[j] / dt)), up + MIN_SWING_TICKS, nxt - 1))
    return td, tau, lag, is_blocked


def detector_triggers(steps, td, dt, swing_ticks, det_delay, det_delay_std, false_rate, rng):
    """Contact-detector triggers per step: the delayed true detection (None if
    it would land at or after the next liftoff) and, with probability
    false_rate, a false trigger in the first half of the swing-down (the last
    third of the nominal swing). Draws three numbers per step, whatever the
    settings.
    Returns:
        list (per step) of sorted trigger ticks
        list (per step) of the false trigger tick or None
    """
    triggers, false_ticks = [], []
    for j, (i, up, d, nxt) in enumerate(steps):
        n, u, u2 = rng.normal(), rng.random(), rng.random()
        delay = max(0, int(round((det_delay + det_delay_std * n) / dt)))
        trig = [td[j] + delay] if td[j] + delay < nxt else []
        f = None
        if u < false_rate:
            down_start = d - swing_ticks // 3
            f = down_start + int(u2 * (swing_ticks // 6))
            trig.append(f)
        triggers.append(sorted(trig))
        false_ticks.append(f)
    return triggers, false_ticks


def stance_from_starts(stance_cmd, steps, starts):
    """Stance flags: the commanded flags with each step's stance run from its
    chosen start tick (None = no stance this step) to its commanded liftoff."""
    out = stance_cmd.copy()
    for (i, up, d, nxt), s in zip(steps, starts):
        out[up + 1:nxt, i] = False
        if s is not None:
            out[s:nxt, i] = True
    return out


def starts_cpg(steps, triggers, dt, det_delay, gain, window, mistime):
    """Learned-offset (CPG-style) touchdown starts. Returns starts, the
    mistiming flag per step and the per-step lag estimate mu - det_delay (s)
    before that step's update."""
    mu = [det_delay, det_delay]
    starts, flags, est = [], [], []
    for (i, up, d, nxt), trig in zip(steps, triggers):
        g = GROUP_OF[i]
        est.append(mu[g] - det_delay)
        ok = [k for k in trig if (k - d) * dt >= mu[g] - window]
        if not ok:
            starts.append(None)
            flags.append(False)
            continue
        e = (ok[0] - d) * dt - mu[g]
        mistimed = abs(e) > mistime
        if not mistimed:
            mu[g] += gain * e
        starts.append(ok[0])
        flags.append(mistimed)
    return starts, flags, np.array(est)


def starts_phase_ekf(steps, triggers, dt, det_delay, R, q, P0, gate):
    """Kalman-filter touchdown starts: per tripod group a state [lag, lag rate]
    predicted once per step (constant-rate model, white-noise rate change of
    intensity q) and updated with each detected touchdown, y = offset - mean
    delay, noise variance R. Returns starts, the mistiming flag per step, and
    the per-step lag estimate and its std before that step's update (s)."""
    s = [np.zeros(2), np.zeros(2)]
    P = [P0.copy(), P0.copy()]
    last_d = [None, None]
    H = np.array([1.0, 0.0])
    starts, flags, est, est_std = [], [], [], []
    for (i, up, d, nxt), trig in zip(steps, triggers):
        g = GROUP_OF[i]
        if last_d[g] is not None and d != last_d[g]:
            dtau = (d - last_d[g]) * dt
            F = np.array([[1.0, dtau], [0.0, 1.0]])
            Qd = q * np.array([[dtau ** 3 / 3.0, dtau ** 2 / 2.0], [dtau ** 2 / 2.0, dtau]])
            s[g], P[g] = F @ s[g], F @ P[g] @ F.T + Qd
        last_d[g] = d
        est.append(s[g][0])
        est_std.append(np.sqrt(P[g][0, 0]))
        S = P[g][0, 0] + R
        ok = [k for k in trig if ((k - d) * dt - det_delay - s[g][0]) / np.sqrt(S) >= -gate]
        if not ok:
            starts.append(None)
            flags.append(False)
            continue
        nu = (ok[0] - d) * dt - det_delay - s[g][0]
        mistimed = nu / np.sqrt(S) > gate
        if not mistimed:
            K = P[g] @ H / S
            s[g] = s[g] + K * nu
            P[g] = P[g] - np.outer(K, H @ P[g])
        starts.append(ok[0])
        flags.append(mistimed)
    return starts, flags, np.array(est), np.array(est_std)


def simulate_trial(args, rng):
    """One trial: true body, foot readings under the true (perturbed) stance,
    initial guess, and every approach's stance flags. Draw order: body,
    reading noise, initial error, touchdown offsets, detector."""
    dt = args.dt
    n_steps = int(round(args.duration / dt))
    tpc = int(round(args.cycle_period / dt))
    duty = GAITS[GAIT][0]
    swing_time = (1.0 - duty) * args.cycle_period
    twist0 = np.array([args.forward_speed, 0.0, args.turn_rate])
    init_std = np.array([args.init_pos_noise_std] * 2 + [args.init_heading_noise_std] + [args.init_twist_noise_std] * 3)

    x_true = simulate_body_trajectory(n_steps, dt, twist0, args.lin_accel_std, args.ang_accel_std, rng)
    noise = rng.normal(0.0, args.meas_noise_std, (n_steps + 1, 6, 2))
    x_init = x_true[0] + rng.normal(0.0, 1.0, 6) * init_std

    stance_cmd = stance_mask(n_steps + 1, tpc, GAIT)
    steps = commanded_steps(stance_cmd)
    blocked = None
    if args.blocked_delay > 0:
        blocked = (LEG_NAMES.index(args.blocked_leg), args.blocked_start, args.blocked_end, args.blocked_delay)
    td, tau, lag, is_blocked = true_touchdowns(steps, dt, swing_time, args.duration, args.lag_max,
                                               args.terrain_std, blocked, rng)
    stance_true = stance_from_starts(stance_cmd, steps, td)
    stance_true_sim, r, rel_vel = simulate_feet(x_true, dt, tpc, GAIT, args.forward_speed, stance=stance_true)
    z = rel_vel + noise

    triggers, false_ticks = detector_triggers(steps, td, dt, int(round(swing_time / dt)), args.det_delay,
                                              args.det_delay_std, args.false_rate, rng)
    R_ekf = args.terrain_std ** 2 + args.det_delay_std ** 2 + dt ** 2 / 12.0
    P0 = np.diag([args.ekf_init_lag_std ** 2, args.ekf_init_rate_std ** 2])
    starts = {"schedule": [d for (_, _, d, _) in steps],
              "detector": [t[0] if t else None for t in triggers]}
    starts["cpg"], cpg_flags, cpg_est = starts_cpg(steps, triggers, dt, args.det_delay, args.cpg_gain,
                                                   args.cpg_window, args.cpg_mistime)
    starts["phase_ekf"], ekf_flags, ekf_est, ekf_std = starts_phase_ekf(
        steps, triggers, dt, args.det_delay, R_ekf, args.ekf_rate_q, P0, args.ekf_gate)
    flags = {a: stance_from_starts(stance_cmd, steps, starts[a]) for a in APPROACHES}
    flags["oracle"] = stance_true
    info = dict(steps=steps, td=td, tau=tau, lag=lag, is_blocked=is_blocked, false_ticks=false_ticks,
                triggers=triggers,
                mistimed={"cpg": np.array(cpg_flags), "phase_ekf": np.array(ekf_flags)},
                lag_est={"cpg": cpg_est, "phase_ekf": ekf_est}, ekf_std=ekf_std, stance_cmd=stance_cmd)
    return x_true, x_init, r, z, flags, info


def flag_errors(flags, stance_true):
    """(swinging foot read as planted, true stance missed) tick counts, t > 0."""
    f, s = flags[1:], stance_true[1:]
    return int((f & ~s).sum()), int((s & ~f).sum())


def filter_setup(args):
    """Body-EKF settings shared by every approach: (Q, per-axis reading
    variance, initial covariance), matched to the truth as in hexapod_gait_ekf.py."""
    init_std = np.array([args.init_pos_noise_std] * 2 + [args.init_heading_noise_std] + [args.init_twist_noise_std] * 3)
    return (process_noise_covariance(args.dt, args.lin_accel_std, args.ang_accel_std), args.meas_noise_std ** 2,
            np.diag(init_std ** 2))


def run_monte_carlo(args, n_trials, rng, rows=ROWS):
    """Repeats simulate_trial n_trials times (fresh body, noise, touchdown
    offsets and detector draws each time; the blocked window and the lag
    profile are deterministic) and runs the body EKF on each row's flags.
    Returns:
        dict row -> dict with nees (N,) per-tick average NEES (6 DoF), rmse (6,)
            per-state RMSE over trials and ticks, swing and missed (mean ticks
            per trial of a swinging foot read as planted / true stance missed),
            and for cpg / phase_ekf mistime_hits, mistime_false (totals over
            trials) and blocked_total, other_total
    """
    Q, R_meas, P_init = filter_setup(args)
    out = None
    for _ in range(n_trials):
        xt, xi, rr, zz, fl, info = simulate_trial(args, rng)
        n_pts = len(xt)
        if out is None:
            out = {a: dict(nees=np.zeros(n_pts), mse=np.zeros(6), swing=0.0, missed=0.0, mistime_hits=0,
                           mistime_false=0, blocked_total=0, other_total=0) for a in rows}
        blk = info["is_blocked"]
        for a in rows:
            xe, Pl, _ = run_ekf(xi, P_init, zz, rr, fl[a], args.dt, Q, R_meas)
            e = state_errors(xt, xe)
            o = out[a]
            o["mse"] += np.mean(e ** 2, axis=0)
            o["nees"] += [nees(e[k], Pl[k]) for k in range(n_pts)]
            sw, ms = flag_errors(fl[a], fl["oracle"])
            o["swing"] += sw
            o["missed"] += ms
            if a in info["mistimed"]:
                m = info["mistimed"][a]
                o["mistime_hits"] += int((m & blk).sum())
                o["mistime_false"] += int((m & ~blk).sum())
                o["blocked_total"] += int(blk.sum())
                o["other_total"] += int((~blk).sum())
    for o in out.values():
        o["nees"] /= n_trials
        o["rmse"] = np.sqrt(o.pop("mse") / n_trials)
        o["swing"] /= n_trials
        o["missed"] /= n_trials
    return out


def build_parser():
    """Command-line flags (also used by the figure and animation scripts for
    the defaults: build_parser().parse_args([]))."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=24.0, help="Simulation length (s); 24 s = 20 cycles")
    parser.add_argument("--dt", type=float, default=0.02, help="Filter/measurement step interval (s)")
    parser.add_argument("--cycle-period", type=float, default=1.2, help="Commanded gait cycle period (s)")
    parser.add_argument("--forward-speed", type=float, default=0.1, help="Initial forward speed u (m/s)")
    parser.add_argument("--turn-rate", type=float, default=0.2, help="Initial turn rate omega (rad/s)")
    parser.add_argument("--lin-accel-std", type=float, default=0.05, help="Linear-acceleration random walk (m/s^2)")
    parser.add_argument("--ang-accel-std", type=float, default=0.1, help="Angular-acceleration random walk (rad/s^2)")
    parser.add_argument("--meas-noise-std", type=float, default=0.02, help="Foot velocity reading noise (m/s)")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.01, help="Initial position std (m)")
    parser.add_argument("--init-heading-noise-std", type=float, default=0.01, help="Initial heading std (rad)")
    parser.add_argument("--init-twist-noise-std", type=float, default=0.05, help="Initial twist std (m/s, rad/s)")

    parser.add_argument("--lag-max", type=float, default=0.2,
                        help="Final systematic touchdown lag, as a fraction of the swing time (grows linearly from 0)")
    parser.add_argument("--terrain-std", type=float, default=0.03, help="Per-step touchdown offset std (s)")
    parser.add_argument("--blocked-leg", type=str, default="L2", choices=LEG_NAMES, help="Leg that gets blocked")
    parser.add_argument("--blocked-start", type=float, default=10.0, help="Blocked window start (s)")
    parser.add_argument("--blocked-end", type=float, default=14.0, help="Blocked window end (s)")
    parser.add_argument("--blocked-delay", type=float, default=0.2, help="Extra landing delay when blocked (s); 0 = off")

    parser.add_argument("--det-delay", type=float, default=0.04, help="Mean detection delay (s), known to the estimators")
    parser.add_argument("--det-delay-std", type=float, default=0.02, help="Detection delay std (s)")
    parser.add_argument("--false-rate", type=float, default=0.1,
                        help="Probability of a false trigger early in a swing-down (assumption)")

    parser.add_argument("--cpg-gain", type=float, default=0.3, help="CPG learning gain on the touchdown offset")
    parser.add_argument("--cpg-window", type=float, default=0.1,
                        help="CPG: reject triggers earlier than the learned offset minus this (s)")
    parser.add_argument("--cpg-mistime", type=float, default=0.1, help="CPG: mistiming threshold on the error (s)")
    parser.add_argument("--ekf-rate-q", type=float, default=1e-5, help="Phase EKF: lag-rate change intensity (s^2/s^3)")
    parser.add_argument("--ekf-init-lag-std", type=float, default=0.05, help="Phase EKF: initial lag std (s)")
    parser.add_argument("--ekf-init-rate-std", type=float, default=0.01, help="Phase EKF: initial lag-rate std (s/s)")
    parser.add_argument("--ekf-gate", type=float, default=3.0, help="Phase EKF: early-reject / mistiming gate (sigma)")

    parser.add_argument("--n-mc-trials", type=int, default=100, help="Number of Monte Carlo trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")
    return parser


def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    dt = args.dt
    Q, R_meas, P_init = filter_setup(args)
    rows = ROWS

    x_true, x_init, r, z, flags, info = simulate_trial(args, rng)
    stance_true = flags["oracle"]
    n_pts = len(x_true)
    runs = {a: run_ekf(x_init, P_init, z, r, flags[a], dt, Q, R_meas)[:2] for a in rows}

    blk, fls = info["is_blocked"], np.array([f is not None for f in info["false_ticks"]])
    print(f"{len(info['steps'])} steps; {int(blk.sum())} blocked, {int(fls.sum())} with a false trigger; "
          f"final systematic lag {1e3 * info['lag'][-1]:.0f} ms")
    print("\nStance flags (single run): swing read as planted / stance missed, ticks")
    for a in rows:
        sw, ms = flag_errors(flags[a], stance_true)
        print(f"  {a:<10s} {sw:5d} / {ms:5d}")
    print("\nMistiming flags: hits on blocked steps / false alarms on other steps")
    for a in ("cpg", "phase_ekf"):
        m = info["mistimed"][a]
        print(f"  {a:<10s} {int((m & blk).sum())} of {int(blk.sum())} / {int((m & ~blk).sum())} of {int((~blk).sum())}")
    print("\nLag estimate error (RMS over steps, vs the true systematic lag)")
    for a in ("cpg", "phase_ekf"):
        print(f"  {a:<10s} {1e3 * np.sqrt(np.mean((info['lag_est'][a] - info['lag']) ** 2)):6.1f} ms")

    print(f"\nRunning Monte Carlo ({args.n_mc_trials} trials)...")
    mc = run_monte_carlo(args, args.n_mc_trials, rng, rows)
    lo, hi = averaged_nees_bounds(6, args.n_mc_trials)
    print(f"\nMonte Carlo (consistent 6-DoF NEES ~6; 95% bounds for a {args.n_mc_trials}-trial average "
          f"[{lo:.2f}, {hi:.2f}]):")
    print(f"  {'':<10s} {'NEES':>8s} {'u RMSE':>10s} {'omega RMSE':>12s} {'swing read/trial':>18s} {'missed/trial':>14s}")
    for a in rows:
        o = mc[a]
        print(f"  {a:<10s} {np.mean(o['nees']):8.2f} {o['rmse'][3]:10.4f} {o['rmse'][5]:12.4f} "
              f"{o['swing']:18.1f} {o['missed']:14.1f}")
    print("\nMonte Carlo mistiming flags: hits on blocked steps / false alarms on other steps (totals)")
    for a in ("cpg", "phase_ekf"):
        o = mc[a]
        print(f"  {a:<10s} {o['mistime_hits']} of {o['blocked_total']} / {o['mistime_false']} of {o['other_total']}")

    t = np.arange(n_pts) * dt
    colors = {"schedule": "tab:blue", "detector": "tab:red", "cpg": "tab:purple", "phase_ekf": "tab:green",
              "oracle": "black"}
    fig, (ax_g, ax_l, ax_n) = plt.subplots(3, 1, figsize=(10, 11))
    leg = LEG_NAMES.index(args.blocked_leg)
    for row, a in enumerate(rows):
        ax_g.fill_between(t, row - 0.4, row + 0.4, where=flags[a][:, leg], step="post", color=colors[a], alpha=0.7)
        bad = flags[a][:, leg] & ~stance_true[:, leg]
        ax_g.fill_between(t, row - 0.4, row + 0.4, where=bad, step="post", color="gold")
    ax_g.set_yticks(range(len(rows)), rows)
    ax_g.set_title(f"Stance flags of leg {args.blocked_leg} (gold = swinging foot read as planted)")
    ax_g.set_xlim(args.blocked_start - 3.0, args.blocked_end + 3.0)

    td_t = np.array([d for (_, _, d, _) in info["steps"]]) * dt
    ax_l.plot(td_t, 1e3 * info["tau"], ".", color="gray", markersize=3, label="true offset tau, per step")
    ax_l.plot(td_t, 1e3 * info["lag"], color="black", label="true systematic lag")
    for a in ("cpg", "phase_ekf"):
        ax_l.plot(td_t, 1e3 * info["lag_est"][a], color=colors[a], label=f"{a} estimate")
    ax_l.set_ylabel("touchdown offset (ms)")
    ax_l.set_xlabel("time (s)")
    ax_l.legend(fontsize=8)

    for a in rows:
        ax_n.plot(t, mc[a]["nees"], color=colors[a], label=a)
    ax_n.axhline(6, color="black", lw=1)
    ax_n.axhspan(lo, hi, color="black", alpha=0.1)
    ax_n.set_yscale("log")
    ax_n.set_xlabel("time (s)")
    ax_n.set_ylabel(f"NEES ({args.n_mc_trials} trials)")
    ax_n.legend(fontsize=8)
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
