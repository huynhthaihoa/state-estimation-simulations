'''
Follow-up to inchworm_slip_gate_ekf.py (1D point mass, one sensor that reads
the body in every phase): a planar inchworm with two pads, turning, one
optical slip sensor per pad, and no absolute position measurement.

Loosely modeled on the inchworm robot of Kubik & Faigl (CTU, ICRA 2026),
which reads a laser-mouse sensor in the anchored pad only and lists Kalman
fusion with motor and inertial data as future work. The numbers here are
not that robot's, and the filter design is this script's, not the paper's.

The gait alternates two phases of length `t_phase`:

  - extend: the back pad is anchored; the body length l grows by a
    trapezoidal rate (peak `v_stroke`, ramps `t_ramp`) and the body turns
    at d(theta)/dt = dl/dt * tan(steer) / l_min (bicycle-style steering,
    wheelbase l_min).
  - contract: the front pad is anchored; l shrinks back, no turning.

The anchored pad is never perfectly still: it creeps with small white
"micro-slip" (std `micro_slip_std` m/s per axis), and with probability
`p_slip` per phase it slips backward along the body axis in one trapezoidal
pulse (peak `v_slip`, duration `t_slip`). The pulses are drawn once per seed
and held fixed over the Monte Carlo trials; everything else is redrawn.

Sensors, every tick:

  - gyro: body turn rate + white noise (`gyro_noise_std`).
  - pad sensor: displacement of the anchored pad over the tick, in the
    body frame, + white noise (`sensor_noise_std` per axis). The moving
    pad's sensor is not used (as in the paper).
  - kinematic closure: the motor encoders give the front pad's position
    relative to the back pad, in the body frame, [l, 0] + noise
    (`closure_noise_std` along the axis, `closure_lat_noise_std` across).

Filter state x = [b_x, b_y, f_x, f_y, theta] (back pad, front pad, heading).
Predict: theta integrates the gyro; the moving pad gets a large random walk
(`move_noise_std` per tick: no motion model, the closure update places it);
the anchored pad is handled by the policy. Update: kinematic closure (an EKF
update, nonlinear in theta).

Four policies for the anchored pad:

  - never: no anchor knowledge, a loose random walk (`free_noise_std` m/s).
  - schedule: "anchored, so still": random walk at the micro-slip level.
    Blind to slip pulses.
  - schedule_and_sensor: as `schedule`, unless the pad sensor reads more
    than `d_thr` = gate_sigmas * (std of a still pad's reading); then
    the loose random walk of `never` for that tick (a veto).
  - pad_fused: the anchored pad moves by the sensor's reading, rotated to
    the world frame, with the sensor noise as process noise.

Monte Carlo NEES (5 DoF) is reported per window: `clean` (ticks of phases
without a slip pulse), `slip` (ticks inside a pulse), and overall.
'''

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import measure_performance, averaged_nees_bounds, nees  # noqa: E402

STATE_DIM = 5
B, F, TH = slice(0, 2), slice(2, 4), 4
POLICIES = ("never", "schedule", "schedule_and_sensor", "pad_fused")


def rot(theta):
    """2D rotation matrix, body frame -> world frame."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def drot(theta):
    """Derivative of rot(theta) with respect to theta."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[-s, -c], [c, -s]])


def wrap(a):
    """Wraps an angle to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def trapezoid(tau, duration, t_ramp, peak):
    """Trapezoidal profile at time tau into a window of length duration:
    0 outside [0, duration], linear ramps of t_ramp at both ends."""
    if tau < 0.0 or tau > duration:
        return 0.0
    if t_ramp <= 0.0:
        return peak
    return peak * min(1.0, tau / t_ramp, (duration - tau) / t_ramp)


def back_is_anchored(phase):
    """Even phases extend (back pad anchored), odd phases contract (front pad anchored)."""
    return phase % 2 == 0


def draw_slip_episodes(n_phases, t_phase, t_slip, p_slip, rng):
    """Draws the slip pulses: each phase gets one with probability p_slip,
    starting uniformly so the whole pulse fits inside the phase. Always
    draws two numbers per phase so the rng stream does not depend on p_slip.
    Arguments:
        n_phases: number of phases
        t_phase: phase duration (s)
        t_slip: pulse duration including both ramps (s); at most t_phase
        p_slip: probability that a phase contains a slip pulse
        rng: numpy random number generator
    Returns:
        list of (phase, t_start, t_end)
    """
    episodes = []
    for j in range(n_phases):
        has_slip, offset = rng.random(), rng.random()
        if has_slip < p_slip:
            start = j * t_phase + offset * (t_phase - t_slip)
            episodes.append((j, start, start + t_slip))
    return episodes


def slip_speed(t, episodes, t_ramp, v_slip):
    """Backward slip speed of the anchored pad at time t (>= 0)."""
    for _, a, b in episodes:
        if a <= t <= b:
            return trapezoid(t - a, b - a, t_ramp, v_slip)
    return 0.0


def simulate(n_steps, dt, t_phase, t_ramp, v_stroke, l_min, steer, episodes, v_slip,
             micro_slip_std, gyro_noise_std, sensor_noise_std, closure_noise_std,
             closure_lat_noise_std, rng):
    """Builds one true trajectory and its sensor readings. Rates are
    integrated with the trapezoidal rule inside each tick, using the phase
    the tick starts in (phase boundaries fall on ticks when t_phase is a
    multiple of dt).
    Arguments:
        n_steps: number of steps
        dt: step interval (s)
        t_phase, t_ramp, v_stroke: gait timing and peak stroke rate (s, s, m/s)
        l_min: body length at the start of each extend, also the steering wheelbase (m)
        steer: steering angle (rad)
        episodes: slip pulses from draw_slip_episodes (fixed across trials)
        v_slip: peak slip speed (m/s)
        micro_slip_std: white creep of the anchored pad (m/s per axis)
        gyro_noise_std: gyro noise (rad/s)
        sensor_noise_std: pad sensor noise per reading and axis (m)
        closure_noise_std, closure_lat_noise_std: closure noise along / across the body (m)
        rng: numpy random number generator
    Returns:
        x_true: (N, 5) true [b, f, theta]
        gyro: (n_steps,) turn-rate readings, one per tick interval
        d: (n_steps, 2) anchored-pad displacement readings, body frame
        z: (N, 2) closure readings, body frame (z[0] unused)
        back_anchored: (n_steps,) bool, which pad is anchored in each interval
    """
    x = np.zeros((n_steps + 1, STATE_DIM))
    x[0, F] = [l_min, 0.0]
    length = l_min
    gyro = np.zeros(n_steps)
    d = np.zeros((n_steps, 2))
    back_anchored = np.zeros(n_steps, dtype=bool)
    for k in range(n_steps):
        t0 = k * dt
        phase = int(np.floor(t0 / t_phase + 1e-9))
        tau0, tau1 = t0 - phase * t_phase, t0 + dt - phase * t_phase
        sign = 1.0 if back_is_anchored(phase) else -1.0
        dl = sign * 0.5 * (trapezoid(tau0, t_phase, t_ramp, v_stroke)
                           + trapezoid(tau1, t_phase, t_ramp, v_stroke)) * dt
        dtheta = dl * np.tan(steer) / l_min if back_is_anchored(phase) else 0.0
        ds = 0.5 * (slip_speed(t0, episodes, t_ramp, v_slip)
                    + slip_speed(t0 + dt, episodes, t_ramp, v_slip)) * dt

        theta = x[k, TH]
        anch = B if back_is_anchored(phase) else F
        step = -ds * rot(theta)[:, 0] + rng.normal(0.0, micro_slip_std * dt, 2)
        a_new = x[k, anch] + step
        length += dl
        theta_new = theta + dtheta
        axis = length * rot(theta_new)[:, 0]
        x[k + 1, TH] = theta_new
        if back_is_anchored(phase):
            x[k + 1, B], x[k + 1, F] = a_new, a_new + axis
        else:
            x[k + 1, F], x[k + 1, B] = a_new, a_new - axis

        back_anchored[k] = back_is_anchored(phase)
        gyro[k] = dtheta / dt + rng.normal(0.0, gyro_noise_std)
        d[k] = rot(theta).T @ step + rng.normal(0.0, sensor_noise_std, 2)

    rel = x[:, F] - x[:, B]
    z = np.einsum("kji,kj->ki", np.array([rot(th) for th in x[:, TH]]), rel)
    z = z + rng.normal(0.0, 1.0, (n_steps + 1, 2)) * [closure_noise_std, closure_lat_noise_std]
    return x, gyro, d, z, back_anchored


def predict(x, P, dt, omega, d_k, back_anchored, policy, noise):
    """Predict step. theta integrates the gyro, the moving pad gets a loose
    random walk, and the anchored pad follows `policy` (module docstring).
    Arguments:
        x, P: state (5,) and covariance (5,5)
        dt: step interval (s)
        omega: gyro reading (rad/s)
        d_k: pad sensor reading for this tick (2,), body frame
        back_anchored: True if the back pad is anchored
        policy: one of POLICIES
        noise: dict of filter noise std-devs (see main)
    Returns:
        x_new, P_new
    """
    anch, mov = (B, F) if back_anchored else (F, B)
    x_new = x.copy()
    Fm = np.eye(STATE_DIM)
    Q = np.zeros((STATE_DIM, STATE_DIM))

    still_var = (noise["micro_slip"] * dt) ** 2
    free_var = (noise["free"] * dt) ** 2
    if policy == "never":
        Q[anch, anch] = free_var * np.eye(2)
    elif policy == "schedule":
        Q[anch, anch] = still_var * np.eye(2)
    elif policy == "schedule_and_sensor":
        quiet = np.linalg.norm(d_k) <= noise["d_thr"]
        Q[anch, anch] = (still_var if quiet else free_var) * np.eye(2)
    elif policy == "pad_fused":
        x_new[anch] = x[anch] + rot(x[TH]) @ d_k
        Fm[anch, TH] = drot(x[TH]) @ d_k
        Q[anch, anch] = noise["sensor"] ** 2 * np.eye(2)
    else:
        raise ValueError(f"unknown policy: {policy!r}")

    Q[mov, mov] = noise["move"] ** 2 * np.eye(2)
    x_new[TH] = x[TH] + omega * dt
    Q[TH, TH] = (noise["gyro"] * dt) ** 2
    return x_new, Fm @ P @ Fm.T + Q


def update_closure(x, P, z, R_cl):
    """EKF update against the kinematic closure z = rot(theta)^T (f - b) + noise.
    Arguments:
        x, P: predicted state (5,) and covariance (5,5)
        z: closure reading (2,), body frame
        R_cl: (2,2) closure noise covariance
    Returns:
        x_new, P_new (Joseph form)
    """
    rel = x[F] - x[B]
    Rt = rot(x[TH]).T
    H = np.zeros((2, STATE_DIM))
    H[:, B], H[:, F], H[:, TH] = -Rt, Rt, drot(x[TH]).T @ rel
    innov = z - Rt @ rel
    S = H @ P @ H.T + R_cl
    K = np.linalg.solve(S, H @ P).T
    x_new = x + K @ innov
    x_new[TH] = wrap(x_new[TH])
    I_KH = np.eye(STATE_DIM) - K @ H
    return x_new, I_KH @ P @ I_KH.T + K @ R_cl @ K.T


def run_ekf(x_init, P_init, gyro, d, z, back_anchored, dt, policy, noise):
    """Recursive EKF over one run: predict, then the closure update.
    Returns:
        x_list: (N, 5) estimates
        P_list: list of (5,5) covariances
    """
    R_cl = np.diag([noise["closure"] ** 2, noise["closure_lat"] ** 2])
    x, P = x_init.copy(), P_init.copy()
    x_list, P_list = [x], [P]
    for k in range(len(gyro)):
        x, P = predict(x, P, dt, gyro[k], d[k], back_anchored[k], policy, noise)
        x, P = update_closure(x, P, z[k + 1], R_cl)
        x_list.append(x)
        P_list.append(P)
    return np.array(x_list), P_list


def state_error(x_true, x_est):
    """x_true - x_est with the heading error wrapped."""
    e = x_true - x_est
    e[..., TH] = wrap(e[..., TH])
    return e


def window_flags(n_steps, dt, t_phase, episodes):
    """Per-tick flags (N,): ticks inside a slip pulse, and ticks of phases with no pulse."""
    ticks = np.arange(n_steps + 1) * dt
    is_slip = np.array([any(a <= t <= b for _, a, b in episodes) for t in ticks])
    slip_phases = {j for j, _, _ in episodes}
    phase_of = np.floor(ticks / t_phase - 1e-9).astype(int)
    is_clean = np.array([j not in slip_phases for j in phase_of])
    is_clean[0] = True
    return is_slip, is_clean


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--duration", type=float, default=20.0, help="Simulation length (s); 20 s = 10 cycles")
    parser.add_argument("--dt", type=float, default=0.05, help="Filter/sensor step interval (s)")
    parser.add_argument("--t-phase", type=float, default=1.0, help="Duration of each extend or contract phase (s)")
    parser.add_argument("--t-ramp", type=float, default=0.2,
                        help="Ramp at each end of a stroke and of a slip pulse (s)")
    parser.add_argument("--v-stroke", type=float, default=0.1, help="Peak stroke rate dl/dt (m/s)")
    parser.add_argument("--l-min", type=float, default=0.15,
                        help="Body length at the start of extend, also the steering wheelbase (m)")
    parser.add_argument("--steer", type=float, default=0.3, help="Steering angle during extend (rad)")

    parser.add_argument("--p-slip", type=float, default=0.5,
                        help="Probability that a phase contains a slip pulse; 0 = never slips")
    parser.add_argument("--v-slip", type=float, default=0.08, help="Peak backward slip speed (m/s)")
    parser.add_argument("--t-slip", type=float, default=0.6,
                        help="Slip pulse duration including both ramps (s); 2*t-ramp <= t-slip <= t-phase")
    parser.add_argument("--micro-slip-std", type=float, default=0.01,
                        help="White creep of the anchored pad, true and assumed by the filter (m/s per axis)")

    parser.add_argument("--gyro-noise-std", type=float, default=0.02, help="Gyro noise std-dev (rad/s)")
    parser.add_argument("--sensor-noise-std", type=float, default=0.0005,
                        help="Pad sensor noise std-dev per reading and axis (m)")
    parser.add_argument("--closure-noise-std", type=float, default=0.001,
                        help="Kinematic closure noise along the body axis (m)")
    parser.add_argument("--closure-lat-noise-std", type=float, default=0.001,
                        help="Kinematic closure noise across the body axis (m)")
    parser.add_argument("--gate-sigmas", type=float, default=3.0,
                        help="Veto threshold in std-devs of a still pad's reading (sensor noise plus micro-slip)")
    parser.add_argument("--free-noise-std", type=float, default=0.1,
                        help="Anchored-pad random walk with no anchor knowledge, used by never and the veto (m/s)")
    parser.add_argument("--move-noise-std", type=float, default=0.02,
                        help="Moving-pad random walk per tick (m); loose, the closure update places the pad")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.005, help="Initial pad position error (m)")
    parser.add_argument("--init-heading-noise-std", type=float, default=0.02, help="Initial heading error (rad)")

    parser.add_argument("--n-mc-trials", type=int, default=500, help="Number of Monte Carlo consistency trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")

    args = parser.parse_args()
    if not 2.0 * args.t_ramp <= args.t_slip <= args.t_phase:
        raise SystemExit("need 2*t-ramp <= t-slip <= t-phase")
    rng = np.random.default_rng(args.seed)

    n_steps = int(round(args.duration / args.dt))
    n_phases = int(np.ceil(args.duration / args.t_phase))
    episodes = draw_slip_episodes(n_phases, args.t_phase, args.t_slip, args.p_slip, rng)
    is_slip, is_clean = window_flags(n_steps, args.dt, args.t_phase, episodes)

    still_std = np.hypot(args.sensor_noise_std, args.micro_slip_std * args.dt)
    noise = dict(micro_slip=args.micro_slip_std, free=args.free_noise_std, sensor=args.sensor_noise_std,
                 move=args.move_noise_std, gyro=args.gyro_noise_std, closure=args.closure_noise_std,
                 closure_lat=args.closure_lat_noise_std, d_thr=args.gate_sigmas * still_std)
    init_std = np.array([args.init_pos_noise_std] * 4 + [args.init_heading_noise_std])
    P_init = np.diag(init_std ** 2)
    sim_args = (n_steps, args.dt, args.t_phase, args.t_ramp, args.v_stroke, args.l_min, args.steer, episodes,
                args.v_slip, args.micro_slip_std, args.gyro_noise_std, args.sensor_noise_std,
                args.closure_noise_std, args.closure_lat_noise_std)

    def one_trial(trial_rng):
        x_true, gyro, d, z, back = simulate(*sim_args, trial_rng)
        x_init = x_true[0] + trial_rng.normal(0.0, 1.0, STATE_DIM) * init_std
        return x_true, gyro, d, z, back, x_init

    print(f"Slip pulses: {len(episodes)} of {n_phases} phases ({int(is_slip.sum())} slip ticks); "
          f"veto threshold {noise['d_thr'] * 1e3:.2f} mm per tick")

    x_true, gyro, d, z, back, x_init = one_trial(rng)
    runs = {}
    for p in POLICIES:
        (x_est, P_est), t_step, mem = measure_performance(
            run_ekf, x_init, P_init, gyro, d, z, back, args.dt, p, noise, n_steps=n_steps)
        runs[p] = (x_est, P_est, t_step, mem)

    print(f"Running Monte Carlo consistency check ({args.n_mc_trials} trials)...")
    sums = {p: np.zeros(n_steps + 1) for p in POLICIES}
    for _ in range(args.n_mc_trials):
        xt, g, dd, zz, bb, xi = one_trial(rng)
        for p in POLICIES:
            xe, Pe = run_ekf(xi, P_init, g, dd, zz, bb, args.dt, p, noise)
            err = state_error(xt, xe)
            for k in range(n_steps + 1):
                sums[p][k] += nees(err[k], Pe[k])
    nees_by_policy = {p: s / args.n_mc_trials for p, s in sums.items()}

    mid_true = 0.5 * (x_true[:, B] + x_true[:, F])
    path = np.sum(np.linalg.norm(np.diff(mid_true, axis=0), axis=1))
    print(f"\nSingle run (path length of the body midpoint {path:.3f} m):")
    for p in POLICIES:
        mid_est = 0.5 * (runs[p][0][:, B] + runs[p][0][:, F])
        pos_err = np.linalg.norm(mid_true - mid_est, axis=1)
        th_err = state_error(x_true, runs[p][0])[:, TH]
        print(f"  {p:<20s} midpoint RMSE={np.sqrt(np.mean(pos_err ** 2)):7.4f} m, "
              f"final={pos_err[-1]:7.4f} m ({100 * pos_err[-1] / path:5.1f}% of path), "
              f"heading RMSE={np.sqrt(np.mean(th_err ** 2)):6.4f} rad")

    lo, hi = averaged_nees_bounds(STATE_DIM, args.n_mc_trials)
    print(f"\nMonte Carlo NEES (consistent {STATE_DIM}-DoF filter ~{STATE_DIM}; 95% bounds for a "
          f"{args.n_mc_trials}-trial average: [{lo:.2f}, {hi:.2f}]):")
    windows = [("overall", np.ones_like(is_slip)), ("clean", is_clean), ("slip", is_slip)]
    print("  " + " " * 20 + "".join(f"{name:>12s}" for name, _ in windows))
    for p in POLICIES:
        n = nees_by_policy[p]
        cells = "".join(f"{np.mean(n[w]) if w.any() else float('nan'):12.2f}" for _, w in windows)
        print(f"  {p:<20s}{cells}")

    print("\nAverage time (per-step) and peak memory (whole-run) per policy, empirical:")
    for p in POLICIES:
        print(f"  {p:<20s} avg time={runs[p][2] * 1e6:9.2f} us/step | peak mem={runs[p][3] / 1024.0:9.3f} KB")

    t_hist = np.arange(n_steps + 1) * args.dt
    colors = {"never": "tab:gray", "schedule": "tab:blue", "schedule_and_sensor": "tab:purple",
              "pad_fused": "tab:green"}
    fig = plt.figure(figsize=(12, 9))
    ax_top = fig.add_subplot(1, 2, 1)
    ax_d = fig.add_subplot(2, 2, 2)
    ax_nees = fig.add_subplot(2, 2, 4, sharex=ax_d)

    ax_top.plot(mid_true[:, 0], mid_true[:, 1], color="black", linestyle="--", linewidth=2,
                label="Ground truth (body midpoint)")
    for p in POLICIES:
        mid_est = 0.5 * (runs[p][0][:, B] + runs[p][0][:, F])
        ax_top.plot(mid_est[:, 0], mid_est[:, 1], color=colors[p], label=p)
    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("y (m)")
    ax_top.set_aspect("equal")
    ax_top.set_title("Two-pad inchworm, top view")
    ax_top.legend(fontsize=8)

    ax_d.plot(t_hist[1:], np.linalg.norm(d, axis=1) * 1e3, color="tab:orange", linewidth=1,
              label="|anchored-pad reading|")
    ax_d.axhline(noise["d_thr"] * 1e3, color="tab:red", linestyle="--", linewidth=1, label="veto threshold")
    ax_d.set_ylabel("Displacement per tick (mm)")
    ax_d.legend(fontsize=8)

    for p in POLICIES:
        ax_nees.plot(t_hist, nees_by_policy[p], color=colors[p], label=p)
    ax_nees.axhline(STATE_DIM, color="black", linewidth=1, label=f"Expected NEES ({STATE_DIM} DoF)")
    ax_nees.axhline(lo, color="black", linestyle="--", linewidth=1,
                    label=f"95% bounds for a {args.n_mc_trials}-trial average")
    ax_nees.axhline(hi, color="black", linestyle="--", linewidth=1)
    ax_nees.set_yscale("log")
    ax_nees.set_xlabel("Time (s)")
    ax_nees.set_ylabel(f"Monte Carlo avg. NEES ({args.n_mc_trials} trials)")
    ax_nees.legend(fontsize=8)
    for ax in (ax_d, ax_nees):
        for _, a, b in episodes:
            ax.axvspan(a, b, color="gold", alpha=0.25, linewidth=0)

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
