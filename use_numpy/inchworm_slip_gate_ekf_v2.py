'''
Side extension of inchworm_zupt_ekf.py (1D point mass, state `x = [p, v]`):
what if the anchor phase is not always stationary, and the filter decides
when to trust "v = 0" from a pad sensor instead of the known schedule?

Two additions to the 1D toy:

  - Anchor slip. In each anchor (dwell) phase the body slides backward with
    probability `p_slip`: a trapezoidal velocity pulse of peak `v_slip` and
    duration `t_slip` (ramps of `t_ramp`, like the extend phase, so slip onset
    is a finite acceleration, not an unmodeled step). The slip pattern is
    drawn once per seed and held fixed over the Monte Carlo trials, so the NEES
    windows below are the same ticks in every trial.

  - A pad displacement sensor. Every tick it reports `d_k = dt * v_k + noise`
    (noise std `sensor_noise_std`, independent per tick): the current velocity
    times the tick length, a noisy velocity in displacement units (a
    body-to-ground displacement over the tick if the velocity were constant).
    This is a simplification of a pad-mounted optical sensor, which sees the
    pad's displacement; here it reads the body's.

Five ways to decide when the zero-velocity pseudo-measurement (ZUPT) runs.
All share the same predict step and the same noisy position measurement every
tick; they differ *only* in the extra velocity information they apply:

  - never: position only.
  - schedule: ZUPT on ticks the known schedule marks as anchor. This is
    `phase_conditional` of inchworm_zupt_ekf.py, blind to slip.
  - sensor_gate: ZUPT on ticks where `|d_k| <= d_thr` (no schedule). The
    threshold is `gate_sigmas * sensor_noise_std`.
  - schedule_and_sensor: ZUPT only when the schedule says anchor AND the
    sensor is quiet.
  - sensor_fused: no gate and no ZUPT; the reading `d_k / dt` is applied
    every tick as a velocity measurement with noise `sensor_noise_std / dt`.

Monte Carlo NEES is reported per window: `clean anchor` (anchor ticks of
phases with no slip), `slip` (ticks inside a slip pulse) and `cruise`.
Ramp ticks and the anchor ticks of slip phases outside the pulse count only
in the overall average.
'''

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import measure_performance, averaged_nees_bounds, nees  # noqa: E402
from inchworm_zupt_ekf import (  # noqa: E402
    is_anchor_phase, is_cruise_phase, true_velocity, state_errors)

STATE_DIM = 2
GATING_MODES = ("never", "schedule", "sensor_gate", "schedule_and_sensor", "sensor_fused")


def draw_slip_episodes(duration, t_anchor, t_extend, t_slip, p_slip, rng):
    """Draws the slip pulses: each anchor phase gets one with probability
    p_slip, starting uniformly so the whole pulse fits inside the phase.
    Always draws two numbers per anchor phase so the rng stream does not
    depend on p_slip.
    Arguments:
        duration: total simulation time (s)
        t_anchor, t_extend: phase durations per cycle (s)
        t_slip: duration of one slip pulse including both ramps (s); at most t_anchor
        p_slip: probability that an anchor phase contains a slip pulse
        rng: numpy random number generator
    Returns:
        list of (t_start, t_end) pairs in seconds
    """
    period = t_anchor + t_extend
    episodes = []
    for c in range(int(np.ceil(duration / period))):
        has_slip, offset = rng.random(), rng.random()
        if has_slip < p_slip:
            start = c * period + offset * (t_anchor - t_slip)
            episodes.append((start, start + t_slip))
    return episodes


def slip_velocity(t, episodes, t_ramp, v_slip):
    """Backward slip velocity at time t: 0 outside the pulses, a trapezoid of
    peak -v_slip inside (linear ramps of t_ramp at both ends).
    Arguments:
        t: time (s)
        episodes: list of (t_start, t_end) from draw_slip_episodes
        t_ramp: ramp duration at each end of a pulse (s)
        v_slip: peak slip speed (m/s)
    Returns:
        v: slip velocity at time t (m/s), <= 0
    """
    for a, b in episodes:
        if a <= t < b:
            if t_ramp <= 0.0:
                return -v_slip
            return -v_slip * min(1.0, (t - a) / t_ramp, (b - t) / t_ramp)
    return 0.0


def generate_ground_truth_and_data(duration, dt, t_anchor, t_extend, t_ramp, v_extend,
                                   pos_noise_std, sensor_noise_std, p_slip, v_slip, t_slip, rng):
    """Builds the true trajectory (anchor with optional slip pulses, extend
    as in inchworm_zupt_ekf.py), the noisy position measurements, the noisy
    pad sensor readings, and the per-tick window flags.
    Position is integrated with the trapezoidal rule, as in the 1D script.
    Arguments:
        duration: total simulation time (s)
        dt: step interval (s)
        t_anchor, t_extend, t_ramp, v_extend: gait parameters as in inchworm_zupt_ekf.py
        pos_noise_std: position measurement noise std-dev (m)
        sensor_noise_std: pad sensor noise std-dev per reading (m)
        p_slip: probability that an anchor phase contains a slip pulse
        v_slip: peak slip speed (m/s)
        t_slip: duration of one slip pulse including both ramps (s)
        rng: numpy random number generator
    Returns:
        x_true: (N, 2) true [p, v]
        z: (N,) noisy position measurements
        d: (N,) pad sensor readings dt * v + noise (d[0] is unused)
        is_anchor: (N,) bool, scheduled anchor ticks
        is_cruise: (N,) bool, steady-cruise ticks
        is_slip: (N,) bool, ticks inside a slip pulse
        is_clean_anchor: (N,) bool, anchor ticks of phases with no slip pulse
    """
    if not 2.0 * t_ramp <= t_slip <= t_anchor:
        raise ValueError("need 2*t_ramp <= t_slip <= t_anchor")
    episodes = draw_slip_episodes(duration, t_anchor, t_extend, t_slip, p_slip, rng)
    n_steps = int(duration / dt)
    ticks = np.arange(n_steps + 1) * dt
    v_true = np.array([
        slip_velocity(t, episodes, t_ramp, v_slip) if is_anchor_phase(t, t_anchor, t_extend)
        else true_velocity(t, t_anchor, t_extend, t_ramp, v_extend) for t in ticks])
    p_true = np.zeros(n_steps + 1)
    p_true[1:] = np.cumsum(0.5 * (v_true[:-1] + v_true[1:]) * dt)
    x_true = np.stack([p_true, v_true], axis=1)

    is_anchor = np.array([is_anchor_phase(t, t_anchor, t_extend) for t in ticks])
    is_cruise = np.array([is_cruise_phase(t, t_anchor, t_extend, t_ramp) for t in ticks])
    is_slip = np.array([any(a <= t < b for a, b in episodes) for t in ticks])
    period = t_anchor + t_extend
    slip_cycles = {int(a // period) for a, _ in episodes}
    is_clean_anchor = is_anchor & np.array([int(t // period) not in slip_cycles for t in ticks])

    z = p_true + rng.normal(0.0, pos_noise_std, n_steps + 1)
    d = dt * v_true + rng.normal(0.0, sensor_noise_std, n_steps + 1)
    return x_true, z, d, is_anchor, is_cruise, is_slip, is_clean_anchor


class InchwormSlipGateEKF:
    """Encapsulates the EKF state and covariance, and provides a single-step
    update method that applies the predict step, the position measurement
    update, and the extra velocity information chosen by the gating mode
    (see the module docstring).
    """

    def __init__(self, x_init, P_init, dt, process_noise_std, R_pos, R_zupt, R_vel, d_thr, mode):
        """
        Arguments:
            x_init: initial state estimate (2,)
            P_init: initial covariance (2,2)
            dt: propagation interval (s)
            process_noise_std: std-dev of the assumed acceleration disturbance (m/s^2)
            R_pos, R_zupt, R_vel: measurement noise variances (m^2, (m/s)^2, (m/s)^2)
            d_thr: gate threshold on |d| (m)
            mode: one of GATING_MODES
        """
        if mode not in GATING_MODES:
            raise ValueError(f"unknown mode: {mode!r}")
        self.x = x_init.copy()
        self.P = P_init.copy()
        self.dt = dt
        self.R_pos = R_pos
        self.R_zupt = R_zupt
        self.R_vel = R_vel
        self.d_thr = d_thr
        self.mode = mode

        self.Phi = np.array([[1.0, dt], [0.0, 1.0]])
        self.Q = np.array([[(0.5 * process_noise_std * dt * dt) ** 2, 0.0],
                           [0.0, (process_noise_std * dt) ** 2]])

    def predict(self):
        """Ordinary constant-velocity EKF predict step -- exactly linear, so no
        linearization error, and identical for all 5 gating modes and both
        gait phases: the filter does not know the true commanded velocity, only
        that it's approximately constant between measurements (a standard
        dead-reckoning assumption), regardless of phase. The same "simple
        heuristic, not textbook continuous white-noise-acceleration" Q as
        saltation_matrix_ekf.process_noise_covariance.
        """
        self.x = self.Phi @ self.x
        self.P = self.Phi @ self.P @ self.Phi.T + self.Q

    def _kf_update(self, z, H, R):
        """Shared linear-Gaussian KF update math for the three measurement
        methods below -- kept private since their fixed H/z are the actual
        interface this script's callers reason about.
        """
        r = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ r
        self.P = (np.eye(2) - K @ H) @ self.P

    def measurement_update_position(self, z):
        """Linear-Gaussian update against a direct noisy position observation
        z = p + noise (H = [1, 0]).
        Arguments:
            z: position measurement (scalar)
        """
        self._kf_update(np.array([z]), np.array([[1.0, 0.0]]), np.array([[self.R_pos]]))

    def measurement_update_zupt(self):
        """Zero-velocity pseudo-measurement update: z = 0 = v + noise (H = [0, 1]).
        Only valid while the point mass is actually stationary -- applying this
        while genuinely moving (the `schedule` mode's mistake inside a slip
        pulse) tells the filter, with high confidence, something false.
        """
        self._kf_update(np.array([0.0]), np.array([[0.0, 1.0]]), np.array([[self.R_zupt]]))

    def measurement_update_velocity(self, z_v):
        """Linear-Gaussian update against a direct noisy velocity reading
        z_v = v + noise (H = [0, 1]); the ZUPT is the special case z_v = 0.
        Arguments:
            z_v: velocity measurement (m/s)
        """
        self._kf_update(np.array([z_v]), np.array([[0.0, 1.0]]), np.array([[self.R_vel]]))

    def update(self, z, d, is_anchor_tick):
        """Position update, then the extra velocity information `mode` calls
        for on this tick: a ZUPT (schedule, sensor gate, or both) or the
        sensor reading as a velocity measurement (sensor_fused).
        Arguments:
            z: position measurement for this tick (scalar)
            d: pad sensor reading for this tick (m)
            is_anchor_tick: whether the known schedule marks this tick as anchor
                (only consulted by "schedule" and "schedule_and_sensor")
        """
        self.measurement_update_position(z)
        quiet = abs(d) <= self.d_thr
        if self.mode == "schedule" and is_anchor_tick:
            self.measurement_update_zupt()
        elif self.mode == "sensor_gate" and quiet:
            self.measurement_update_zupt()
        elif self.mode == "schedule_and_sensor" and is_anchor_tick and quiet:
            self.measurement_update_zupt()
        elif self.mode == "sensor_fused":
            self.measurement_update_velocity(d / self.dt)

    def step(self, z, d, is_anchor_tick):
        """One filter tick: predict, then update (see update).
        Arguments:
            z: position measurement for this tick (scalar)
            d: pad sensor reading for this tick (m)
            is_anchor_tick: whether the known schedule marks this tick as anchor
        """
        self.predict()
        self.update(z, d, is_anchor_tick)

    def get_state(self):
        """Returns copies of (x, P), safe to store as history."""
        return self.x.copy(), self.P.copy()


def run_ekf(x_init, P_init, z, d, dt, n_steps, process_noise_std, R_pos, R_zupt, R_vel, d_thr,
            is_anchor, mode):
    """Recursive EKF: predict, position update every tick, then the extra
    velocity information chosen by `mode` (see the module docstring).
    Arguments:
        x_init, P_init: initial state (2,) and covariance (2,2)
        z: (n_steps+1,) position measurements
        d: (n_steps+1,) pad sensor readings
        dt: step interval (s)
        n_steps: number of steps
        process_noise_std: assumed acceleration disturbance std-dev (m/s^2)
        R_pos, R_zupt, R_vel: measurement noise variances (m^2, (m/s)^2, (m/s)^2)
        d_thr: gate threshold on |d| (m)
        is_anchor: (n_steps+1,) bool scheduled anchor flags
        mode: one of GATING_MODES
    Returns:
        x_list: (n_steps+1, 2) estimated states
        P_list: list of (2,2) covariances
    """
    ekf = InchwormSlipGateEKF(x_init, P_init, dt, process_noise_std, R_pos, R_zupt, R_vel, d_thr, mode)
    x, P = ekf.get_state()
    x_list, P_list = [x], [P]
    for k in range(n_steps):
        ekf.step(z[k + 1], d[k + 1], is_anchor[k + 1])
        x, P = ekf.get_state()
        x_list.append(x)
        P_list.append(P)
    return np.array(x_list), P_list


def run_monte_carlo_consistency(x_true, dt, process_noise_std, R_pos, R_zupt, R_vel, d_thr,
                                is_anchor, init_pos_noise_std, init_vel_noise_std,
                                sensor_noise_std, n_trials, rng, modes=GATING_MODES):
    """Repeats every mode over n_trials noise realizations of the same true
    trajectory (fresh initial error, position noise and sensor noise each
    trial) and returns the per-step average NEES.
    Arguments:
        x_true: (N, 2) true states
        dt: step interval (s)
        process_noise_std: assumed acceleration disturbance std-dev (m/s^2)
        R_pos, R_zupt, R_vel: measurement noise variances
        d_thr: gate threshold on |d| (m)
        is_anchor: (N,) bool scheduled anchor flags
        init_pos_noise_std, init_vel_noise_std: initial error std-devs
        sensor_noise_std: pad sensor noise std-dev (m)
        n_trials: number of trials
        rng: numpy random number generator
        modes: policies to run
    Returns:
        dict mode -> (N,) average NEES per step (2 DoF)
    """
    n_steps = len(x_true) - 1
    n_pts = n_steps + 1
    init_std = np.array([init_pos_noise_std, init_vel_noise_std])
    P_init = np.diag(init_std ** 2)
    sums = {m: np.zeros(n_pts) for m in modes}
    for _ in range(n_trials):
        x_init = x_true[0] + rng.normal(0.0, 1.0, 2) * init_std
        z_trial = x_true[:, 0] + rng.normal(0.0, np.sqrt(R_pos), n_pts)
        d_trial = dt * x_true[:, 1] + rng.normal(0.0, sensor_noise_std, n_pts)
        for m in modes:
            x_est, P_est = run_ekf(x_init, P_init, z_trial, d_trial, dt, n_steps, process_noise_std,
                                   R_pos, R_zupt, R_vel, d_thr, is_anchor, m)
            for k in range(n_pts):
                sums[m][k] += nees(x_true[k] - x_est[k], P_est[k])
    return {m: s / n_trials for m, s in sums.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--duration", type=float, default=20.0,
                        help="Simulation length in seconds (20 s = 10 cycles, so several slip pulses)")
    parser.add_argument("--dt", type=float, default=0.05, help="Filter/measurement step interval in seconds")
    parser.add_argument("--t-anchor", type=float, default=1.0, help="Anchor/dwell phase duration per cycle (s)")
    parser.add_argument("--t-extend", type=float, default=1.0,
                        help="Total extend phase duration per cycle (s), including both ramps")
    parser.add_argument("--t-ramp", type=float, default=0.2,
                        help="Ramp duration at each end of the extend phase and of a slip pulse (s)")
    parser.add_argument("--v-extend", type=float, default=0.1, help="Commanded cruise speed during extend (m/s)")

    parser.add_argument("--p-slip", type=float, default=0.5,
                        help="Probability that an anchor phase contains a slip pulse; 0 = never slips")
    parser.add_argument("--v-slip", type=float, default=0.08, help="Peak backward slip speed (m/s)")
    parser.add_argument("--t-slip", type=float, default=0.6,
                        help="Slip pulse duration including both ramps (s); 2*t-ramp <= t-slip <= t-anchor")

    parser.add_argument("--pos-noise-std", type=float, default=0.02, help="Position measurement noise std-dev (m)")
    parser.add_argument("--sensor-noise-std", type=float, default=0.0005,
                        help="Pad sensor noise std-dev per reading (m); 0.0005 m / 0.05 s = 0.01 m/s")
    parser.add_argument("--gate-sigmas", type=float, default=3.0,
                        help="Gate threshold in units of the sensor noise std-dev: d-thr = gate-sigmas * "
                             "sensor-noise-std (speed threshold d-thr / dt)")
    parser.add_argument("--process-noise-std", type=float, default=0.5,
                        help="Assumed acceleration disturbance std-dev (m/s^2); 0.5 covers the true ramps, "
                             "the 1D script's 0.15 does not")
    parser.add_argument("--zupt-noise-std", type=float, default=0.01, help="ZUPT noise std-dev (m/s)")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.05, help="Initial position error std-dev (m)")
    parser.add_argument("--init-vel-noise-std", type=float, default=0.05, help="Initial velocity error std-dev (m/s)")

    parser.add_argument("--n-mc-trials", type=int, default=500, help="Number of Monte Carlo consistency trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    x_true, z, d, is_anchor, is_cruise, is_slip, is_clean = generate_ground_truth_and_data(
        args.duration, args.dt, args.t_anchor, args.t_extend, args.t_ramp, args.v_extend,
        args.pos_noise_std, args.sensor_noise_std, args.p_slip, args.v_slip, args.t_slip, rng)
    n_steps = len(x_true) - 1

    init_std = np.array([args.init_pos_noise_std, args.init_vel_noise_std])
    x_init = x_true[0] + rng.normal(0.0, 1.0, 2) * init_std
    P_init = np.diag(init_std ** 2)
    R_pos, R_zupt = args.pos_noise_std ** 2, args.zupt_noise_std ** 2
    R_vel = (args.sensor_noise_std / args.dt) ** 2
    d_thr = args.gate_sigmas * args.sensor_noise_std

    print(f"Slip ticks: {int(is_slip.sum())}, clean-anchor ticks: {int(is_clean.sum())}, "
          f"cruise ticks: {int(is_cruise.sum())}; gate threshold {d_thr * 1e3:.2f} mm per tick "
          f"= {d_thr / args.dt:.3f} m/s")

    runs = {}
    for m in GATING_MODES:
        print(f"Running EKF with mode='{m}'...")
        (x_est, P_est), t_step, mem = measure_performance(
            run_ekf, x_init, P_init, z, d, args.dt, n_steps, args.process_noise_std, R_pos, R_zupt,
            R_vel, d_thr, is_anchor, m, n_steps=n_steps)
        runs[m] = (x_est, P_est, t_step, mem)

    print(f"Running Monte Carlo consistency check ({args.n_mc_trials} trials)...")
    nees_by_mode = run_monte_carlo_consistency(
        x_true, args.dt, args.process_noise_std, R_pos, R_zupt, R_vel, d_thr, is_anchor,
        args.init_pos_noise_std, args.init_vel_noise_std, args.sensor_noise_std, args.n_mc_trials, rng)

    print("\nRMS errors (single run):")
    for m in GATING_MODES:
        pe, ve = state_errors(x_true, runs[m][0])
        print(f"  {m:<20s} pos={np.sqrt(np.mean(pe ** 2)):7.4f} m, v={np.sqrt(np.mean(ve ** 2)):7.4f} m/s")

    lo, hi = averaged_nees_bounds(STATE_DIM, args.n_mc_trials)
    print(f"\nMonte Carlo NEES (consistent {STATE_DIM}-DoF filter ~{STATE_DIM}; 95% bounds for a "
          f"{args.n_mc_trials}-trial average: [{lo:.2f}, {hi:.2f}]):")
    windows = [("overall", np.ones_like(is_slip)), ("clean anchor", is_clean), ("slip", is_slip),
               ("cruise", is_cruise)]
    print("  " + " " * 20 + "".join(f"{name:>14s}" for name, _ in windows))
    for m in GATING_MODES:
        n = nees_by_mode[m]
        cells = "".join(f"{np.mean(n[w]) if w.any() else float('nan'):14.2f}" for _, w in windows)
        print(f"  {m:<20s}{cells}")

    print("\nAverage time (per-step) and peak memory (whole-run) per approach, empirical:")
    for m in GATING_MODES:
        print(f"  {m:<20s} avg time={runs[m][2] * 1e6:9.2f} us/step | peak mem={runs[m][3] / 1024.0:9.3f} KB")

    t_hist = np.arange(len(x_true)) * args.dt
    colors = {"never": "tab:gray", "schedule": "tab:blue", "sensor_gate": "tab:red",
              "schedule_and_sensor": "tab:purple", "sensor_fused": "tab:green"}
    fig, (ax_v, ax_d, ax_nees) = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    ax_v.plot(t_hist, x_true[:, 1], label="Ground truth", color="black", linewidth=2, linestyle="--")
    for m in ("schedule", "sensor_gate", "sensor_fused"):
        ax_v.plot(t_hist, runs[m][0][:, 1], label=m, color=colors[m])
    ax_v.set_ylabel("Velocity (m/s)")
    ax_v.set_title("Inchworm with anchor slip: ZUPT gated by schedule vs. by a pad sensor")
    ax_v.legend()

    ax_d.plot(t_hist, d * 1e3, color="tab:orange", linewidth=1, label="pad sensor reading d")
    ax_d.axhspan(-d_thr * 1e3, d_thr * 1e3, color="tab:red", alpha=0.15, label="quiet band |d| <= threshold")
    ax_d.set_ylabel("Displacement per tick (mm)")
    ax_d.legend()

    for m in GATING_MODES:
        ax_nees.plot(t_hist, nees_by_mode[m], label=m, color=colors[m])
    ax_nees.axhline(STATE_DIM, color="black", linestyle="-", linewidth=1, label=f"Expected NEES ({STATE_DIM} DoF)")
    ax_nees.axhline(lo, color="black", linestyle="--", linewidth=1,
                    label=f"95% bounds for a {args.n_mc_trials}-trial average")
    ax_nees.axhline(hi, color="black", linestyle="--", linewidth=1)
    ax_nees.set_ylabel(f"Monte Carlo avg. NEES ({args.n_mc_trials} trials)")
    ax_nees.set_xlabel("Time (s)")
    ax_nees.set_yscale("log")
    ax_nees.legend(fontsize=8)
    for ax in (ax_v, ax_d, ax_nees):
        edges = np.diff(np.concatenate([[False], is_slip, [False]]).astype(int))
        for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            ax.axvspan(t_hist[a], t_hist[b - 1] + args.dt, color="gold", alpha=0.25, linewidth=0)

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
