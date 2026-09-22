'''
Tracks a 1D point mass crawling like an inchworm: a repeating anchor (dwell,
exactly stationary) / extend (moving at a constant commanded speed) gait
cycle, with a *known, deterministic* phase schedule -- the simplest possible
model of `unified_phd_plan.md`'s "the anchored/dwell portion of an inchworm
cycle is, by definition, stationary" observation. State `x = [p, v]`, plain
R^2 -- deliberately scoped down to translation-only (no ZARU/orientation/
accel-bias: those need an orientation state this toy doesn't carry) and a
known schedule (phase-*detection* uncertainty is a separate, already-studied
problem -- see docs/filtering/hybrid_saltation_ekf.md's §8 for what happens
once the transition time itself is uncertain).

Unlike saltation_matrix_ekf.py, this is *not* a discontinuous-state-reset
problem: nothing here needs a guard, a reset map, or a saltation matrix. The
ground truth is an ordinary switched-linear system (velocity is externally
commanded per phase, not an instantaneous jump the filter's covariance has
to be linearized through), and every filter variant below shares the exact
same linear-Gaussian predict step. The entire story is on the *measurement*
side: whether the filter exploits the free zero-velocity information the
anchor phase provides, correctly, always, or never.

Three ways to fuse the zero-velocity pseudo-measurement (ZUPT) are compared,
all sharing the same predict step and the same noisy position measurement
each tick -- they differ *only* in when `measurement_update_zupt` also runs:

  - never: only the position measurement is ever used. Never wrong, but
    leaves free information on the table during every anchor tick.

  - always: the position measurement plus an (unconditional) ZUPT are both
    applied every tick, regardless of the true phase -- the natural mistake
    a schedule-unaware implementation would make.

  - phase_conditional: the position measurement every tick, plus ZUPT only
    on ticks the known schedule marks as anchor -- the correct policy.

A Monte Carlo consistency check (`run_monte_carlo_consistency`, following
saltation_matrix_ekf.py's NEES-based pattern) reports NEES not just overall
but split by phase (anchor-only vs. cruise-only ticks, excluding the ramp
edges both share -- see true_velocity's docstring). The going-in assumption
was that `always` would at least match `phase_conditional` during genuine
anchor ticks, since both apply the identical correct update there, and only
diverge during motion. That is *not* what happens: `always`'s NEES is
dramatically worse than `phase_conditional`'s in *both* windows (~20-50x, not
just during cruise) -- misapplying ZUPT throughout every cruise phase leaves
it so overconfident that ~20 ticks of genuinely correct ZUPT evidence at the
start of the next anchor phase isn't enough to recover before that phase
ends. A second, more subtle finding echoes saltation_matrix_ekf.py's own:
`never`, despite discarding real information, ends up at least as well
*calibrated* (NEES) as `phase_conditional`, even though `phase_conditional`
has the best raw *accuracy* (lowest velocity RMS) of the three -- correctly
staking confidence on a strong pseudo-measurement is still measurably more
fragile than never staking it at all, the same theme as saltation matrices'
`Dg @ Xi = 0` exact-zero claim in the sibling script's §8. Verified stable
across seeds 0-3 before being written up here or in the doc.
'''

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import measure_performance

# chi-squared, 2 degrees of freedom, 95th percentile (scipy.stats.chi2.ppf(0.95, 2));
# hardcoded rather than importing scipy, since this script is otherwise pure numpy
# and this is only needed as a reference line on the consistency plot.
CHI2_2DOF_95 = 5.9915


def is_anchor_phase(t, t_anchor, t_extend):
    """Whether the known, deterministic gait schedule says the point mass is
    anchored (dwelling, exactly stationary) at time t, vs. extending. The
    cycle is [0, t_anchor) = anchor, [t_anchor, t_anchor+t_extend) = extend,
    repeating with period t_anchor+t_extend.
    Arguments:
        t: time (s)
        t_anchor: duration of the anchor/dwell phase per cycle (s)
        t_extend: duration of the extend phase per cycle (s)
    Returns:
        True if anchored at time t, False if extending
    """
    phase_t = t % (t_anchor + t_extend)
    return phase_t < t_anchor


def is_cruise_phase(t, t_anchor, t_extend, t_ramp):
    """Whether time t falls in the steady constant-v_extend part of the
    extend phase, excluding its ramp-up/ramp-down edges. Used to window the
    headline NEES comparison away from the (shared, uninteresting, all-3-
    variants-affected) transition ticks, the same way
    saltation_matrix_ekf.py's post-bounce window excludes the tick shared
    spike at the bounce itself.
    Arguments:
        t: time (s)
        t_anchor, t_extend: phase durations per cycle (s)
        t_ramp: ramp-up/ramp-down duration at each end of the extend phase (s)
    Returns:
        True if t is in the steady-cruise part of the extend phase
    """
    phase_t = t % (t_anchor + t_extend) - t_anchor
    if phase_t < 0.0:
        return False  # anchored
    return t_ramp <= phase_t < (t_extend - t_ramp)


def true_velocity(t, t_anchor, t_extend, t_ramp, v_extend):
    """The true (deterministic, externally-commanded) velocity at time t:
    exactly 0 while anchored, ramping linearly up to v_extend over the first
    t_ramp seconds of the extend phase, holding at v_extend through the
    steady "cruise" middle, then ramping back down to exactly 0 over the
    last t_ramp seconds -- continuous everywhere (no instantaneous jump).

    This is a switched-linear system, not a hybrid reset -- there is no
    discontinuous *state* jump to linearize through, unlike
    saltation_matrix_ekf.py's bounce, where the reset itself is the object
    of study. An earlier version of this function used a hard step (0 to
    v_extend instantaneously) instead of a ramp; that produced a true
    acceleration far outside anything the filters' constant-velocity predict
    step could represent (an effectively-infinite jump vs. the model's
    process-noise budget of a few cm/s per tick), which swamped every
    transition tick with a huge NEES spike shared identically by all 3
    filter variants regardless of ZUPT policy -- not the phenomenon this
    script is about. The ramp keeps the true dynamics within a rate of
    change the predict step can plausibly track, so the comparison isolates
    ZUPT-policy differences instead of "can any of these filters track an
    infinite-acceleration jump" (none can, uninterestingly).
    Arguments:
        t: time (s)
        t_anchor: duration of the anchor/dwell phase per cycle (s)
        t_extend: total duration of the extend phase per cycle (s), including
            both ramps
        t_ramp: ramp-up/ramp-down duration at each end of the extend phase (s);
            0.0 recovers the original instantaneous-jump behavior
        v_extend: commanded cruise speed during the extend phase (m/s)
    Returns:
        v: true velocity at time t (m/s)
    """
    if is_anchor_phase(t, t_anchor, t_extend):
        return 0.0
    phase_t = t % (t_anchor + t_extend) - t_anchor  # time since extend phase began, in [0, t_extend)
    if t_ramp <= 0.0:
        return v_extend
    if phase_t < t_ramp:
        return v_extend * (phase_t / t_ramp)
    remaining = t_extend - phase_t  # time left until the next anchor phase begins
    if remaining < t_ramp:
        return v_extend * (remaining / t_ramp)
    return v_extend


def generate_ground_truth_and_data(duration, dt, t_anchor, t_extend, t_ramp, v_extend,
                                    pos_noise_std, rng):
    """Builds the true anchor/extend gait trajectory (exact, deterministic --
    no process noise in the true dynamics themselves, matching
    saltation_matrix_ekf.py's convention) and noisy position measurements a
    tracker would actually receive, plus the per-tick ground-truth anchor and
    cruise flags (used by the phase_conditional filter and by NEES windowing,
    not by the never/always variants).

    Position is integrated with the trapezoidal rule (not left-Euler), which
    is exact here as long as dt evenly divides t_anchor/t_ramp/the cruise
    duration (true by construction for this script's defaults) -- true
    velocity is piecewise *linear* now (see true_velocity), so a left-Euler
    step would pick up a small first-order error during every ramp tick that
    trapezoidal integration avoids.
    Arguments:
        duration: total simulation time (s)
        dt: measurement/filter step interval (s)
        t_anchor: duration of the anchor/dwell phase per cycle (s)
        t_extend: total duration of the extend phase per cycle (s), including both ramps
        t_ramp: ramp-up/ramp-down duration at each end of the extend phase (s)
        v_extend: commanded cruise speed during the extend phase (m/s)
        pos_noise_std: std-dev of Gaussian noise added to position measurements (m)
        rng: numpy random number generator
    Returns:
        x_true: (n_steps+1, 2) array of true [p, v] at each tick (including t=0)
        z: (n_steps+1,) array of noisy position measurements
        is_anchor: (n_steps+1,) bool array, True where that tick is an anchor tick
        is_cruise: (n_steps+1,) bool array, True where that tick is a steady-cruise
            extend tick (excludes the ramp-up/ramp-down edges)
    """
    n_steps = int(duration / dt)
    ticks = np.arange(n_steps + 1) * dt

    v_true = np.array([true_velocity(t, t_anchor, t_extend, t_ramp, v_extend) for t in ticks])
    p_true = np.zeros(n_steps + 1)
    p_true[1:] = np.cumsum(0.5 * (v_true[:-1] + v_true[1:]) * dt)
    x_true = np.stack([p_true, v_true], axis=1)
    is_anchor = np.array([is_anchor_phase(t, t_anchor, t_extend) for t in ticks])
    is_cruise = np.array([is_cruise_phase(t, t_anchor, t_extend, t_ramp) for t in ticks])
    z = x_true[:, 0] + rng.normal(0.0, pos_noise_std, n_steps + 1)
    return x_true, z, is_anchor, is_cruise


def predict(x, P, dt, process_noise_std):
    """Ordinary constant-velocity EKF predict step -- exactly linear, so no
    linearization error, and identical for all 3 filter variants and both
    gait phases: the filter does not know the true commanded velocity, only
    that it's approximately constant between measurements (a standard
    dead-reckoning assumption), regardless of phase. The same "simple
    heuristic, not textbook continuous white-noise-acceleration" Q as
    saltation_matrix_ekf.process_noise_covariance.
    Arguments:
        x: state [p, v] (2,)
        P: (2,2) covariance
        dt: propagation interval (s)
        process_noise_std: std-dev of the assumed acceleration disturbance (m/s^2)
    Returns:
        x_new: predicted state (2,)
        P_new: predicted covariance (2,2)
    """
    Phi = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.array([[(0.5 * process_noise_std * dt * dt) ** 2, 0.0],
                  [0.0, (process_noise_std * dt) ** 2]])
    x_new = Phi @ x
    P_new = Phi @ P @ Phi.T + Q
    return x_new, P_new


def measurement_update_position(x, P, z, R_pos):
    """Linear-Gaussian update against a direct noisy position observation
    z = p + noise (H = [1, 0]).
    Arguments:
        x: predicted state (2,)
        P: predicted covariance (2,2)
        z: position measurement (scalar)
        R_pos: position measurement noise variance (scalar)
    Returns:
        x_new: updated state (2,)
        P_new: updated covariance (2,2)
    """
    H = np.array([[1.0, 0.0]])
    return _kf_update(x, P, np.array([z]), H, np.array([[R_pos]]))


def measurement_update_zupt(x, P, R_zupt):
    """Zero-velocity pseudo-measurement update: z = 0 = v + noise (H = [0, 1]).
    Only valid while the point mass is actually stationary -- applying this
    while genuinely moving (the `always` variant's mistake, or a
    schedule-unaware real implementation's) tells the filter, with high
    confidence, something false.
    Arguments:
        x: predicted state (2,)
        P: predicted covariance (2,2)
        R_zupt: ZUPT pseudo-measurement noise variance (scalar)
    Returns:
        x_new: updated state (2,)
        P_new: updated covariance (2,2)
    """
    H = np.array([[0.0, 1.0]])
    return _kf_update(x, P, np.array([0.0]), H, np.array([[R_zupt]]))


def _kf_update(x, P, z, H, R):
    """Shared linear-Gaussian KF update math for both measurement functions
    above -- kept private since the two public functions' fixed H/z are the
    actual interface this script's callers reason about.
    """
    r = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x_new = x + K @ r
    P_new = (np.eye(2) - K @ H) @ P
    return x_new, P_new


def run_ekf(x_init, P_init, z, dt, n_steps, process_noise_std, R_pos, R_zupt, is_anchor, zupt_mode):
    """Recursive EKF alternating predict with a position update every tick
    and, depending on zupt_mode, an additional ZUPT update. Shared by the
    three run_ekf_* wrappers below -- they differ only in zupt_mode, so any
    difference in their output is attributable entirely to ZUPT usage
    policy, nothing else.
    Arguments:
        x_init: initial state estimate (2,)
        P_init: initial covariance (2,2)
        z: (n_steps+1,) array of noisy position measurements
        dt: step interval (s)
        n_steps: number of steps
        process_noise_std: std-dev of the assumed acceleration disturbance (m/s^2)
        R_pos: position measurement noise variance (scalar)
        R_zupt: ZUPT pseudo-measurement noise variance (scalar)
        is_anchor: (n_steps+1,) bool array, True where that tick is an anchor tick
            (only consulted when zupt_mode == "phase_conditional")
        zupt_mode: "never", "always", or "phase_conditional"
    Returns:
        x_list: (n_steps+1, 2) array of estimated states
        P_list: list of (2,2) estimated covariances
    """
    if zupt_mode not in ("never", "always", "phase_conditional"):
        raise ValueError(f"unknown zupt_mode: {zupt_mode!r}")

    x, P = x_init.copy(), P_init.copy()
    x_list, P_list = [x], [P]
    for k in range(n_steps):
        x, P = predict(x, P, dt, process_noise_std)
        x, P = measurement_update_position(x, P, z[k + 1], R_pos)
        apply_zupt = (zupt_mode == "always"
                      or (zupt_mode == "phase_conditional" and is_anchor[k + 1]))
        if apply_zupt:
            x, P = measurement_update_zupt(x, P, R_zupt)
        x_list.append(x)
        P_list.append(P)
    return np.array(x_list), P_list


def run_ekf_never_zupt(x_init, P_init, z, dt, n_steps, process_noise_std, R_pos, R_zupt, is_anchor):
    """run_ekf with zupt_mode='never' -- see run_ekf's docstring."""
    return run_ekf(x_init, P_init, z, dt, n_steps, process_noise_std, R_pos, R_zupt, is_anchor,
                    zupt_mode="never")


def run_ekf_always_zupt(x_init, P_init, z, dt, n_steps, process_noise_std, R_pos, R_zupt, is_anchor):
    """run_ekf with zupt_mode='always' -- see run_ekf's docstring."""
    return run_ekf(x_init, P_init, z, dt, n_steps, process_noise_std, R_pos, R_zupt, is_anchor,
                    zupt_mode="always")


def run_ekf_phase_conditional_zupt(x_init, P_init, z, dt, n_steps, process_noise_std, R_pos,
                                    R_zupt, is_anchor):
    """run_ekf with zupt_mode='phase_conditional' -- see run_ekf's docstring."""
    return run_ekf(x_init, P_init, z, dt, n_steps, process_noise_std, R_pos, R_zupt, is_anchor,
                    zupt_mode="phase_conditional")


def state_errors(x_true, x_est):
    """Position and velocity absolute error, per step.
    Arguments:
        x_true: (N, 2) array of true states
        x_est: (N, 2) array of estimated states
    Returns:
        pos_err: (N,) array of position error magnitudes (m)
        vel_err: (N,) array of velocity error magnitudes (m/s)
    """
    pos_err = np.abs(x_true[:, 0] - x_est[:, 0])
    vel_err = np.abs(x_true[:, 1] - x_est[:, 1])
    return pos_err, vel_err


def nees(x_true, x_est, P):
    """Normalized Estimation Error Squared: (x_true-x_est)^T @ P^-1 @ (x_true-x_est).
    A consistent filter's NEES should average to the state dimension (2 here)
    across many independent trials; a value that's persistently much larger
    means the filter's reported covariance P is too small (overconfident)
    for the errors it is actually making.
    Arguments:
        x_true: true state (2,)
        x_est: estimated state (2,)
        P: estimated covariance (2,2)
    Returns:
        nees: scalar NEES value
    """
    err = x_true - x_est
    return float(err @ np.linalg.inv(P) @ err)


def run_monte_carlo_consistency(x_true, dt, process_noise_std, R_pos, R_zupt, is_anchor,
                                 init_pos_noise_std, init_vel_noise_std, n_trials, rng):
    """Repeats the three run_ekf_* variants over n_trials independent noise
    realizations of the same nominal true trajectory (fresh initial-condition
    error and fresh measurement noise each trial), returning the per-step
    average NEES for all three.
    Arguments:
        x_true: (N, 2) array of true states at each tick
        dt: step interval (s)
        process_noise_std: std-dev of the assumed acceleration disturbance (m/s^2)
        R_pos: position measurement noise variance (scalar)
        R_zupt: ZUPT pseudo-measurement noise variance (scalar)
        is_anchor: (N,) bool array, True where that tick is an anchor tick
        init_pos_noise_std: std-dev used to perturb the initial position guess
        init_vel_noise_std: std-dev used to perturb the initial velocity guess
        n_trials: number of Monte Carlo trials
        rng: numpy random number generator
    Returns:
        nees_never, nees_always, nees_phase: (N,) average NEES per step, one
            per zupt_mode
    """
    n_steps = len(x_true) - 1
    n_pts = n_steps + 1
    P_init = np.diag([init_pos_noise_std ** 2, init_vel_noise_std ** 2])
    init_std = np.array([init_pos_noise_std, init_vel_noise_std])
    R_pos_std = np.sqrt(R_pos)
    true_pos = x_true[:, 0]

    sum_never = np.zeros(n_pts)
    sum_always = np.zeros(n_pts)
    sum_phase = np.zeros(n_pts)
    for _ in range(n_trials):
        x_init = x_true[0] + rng.normal(0.0, 1.0, 2) * init_std
        z_trial = true_pos + rng.normal(0.0, R_pos_std, n_pts)

        x_never, P_never = run_ekf_never_zupt(x_init, P_init, z_trial, dt, n_steps,
                                               process_noise_std, R_pos, R_zupt, is_anchor)
        x_always, P_always = run_ekf_always_zupt(x_init, P_init, z_trial, dt, n_steps,
                                                  process_noise_std, R_pos, R_zupt, is_anchor)
        x_phase, P_phase = run_ekf_phase_conditional_zupt(x_init, P_init, z_trial, dt, n_steps,
                                                           process_noise_std, R_pos, R_zupt,
                                                           is_anchor)

        for k in range(n_pts):
            sum_never[k] += nees(x_true[k], x_never[k], P_never[k])
            sum_always[k] += nees(x_true[k], x_always[k], P_always[k])
            sum_phase[k] += nees(x_true[k], x_phase[k], P_phase[k])

    return sum_never / n_trials, sum_always / n_trials, sum_phase / n_trials


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--duration", type=float, default=10.0, help="Simulation length in seconds")
    parser.add_argument("--dt", type=float, default=0.05, help="Filter/measurement step interval in seconds")
    parser.add_argument("--t-anchor", type=float, default=1.0, help="Anchor/dwell phase duration per cycle (s)")
    parser.add_argument("--t-extend", type=float, default=1.0,
                         help="Total extend phase duration per cycle (s), including both ramps")
    parser.add_argument("--t-ramp", type=float, default=0.2,
                         help="Ramp-up/ramp-down duration at each end of the extend phase (s) -- "
                              "keeps the true velocity continuous (finite acceleration) instead of "
                              "an instantaneous jump; must satisfy 2*t-ramp < t-extend")
    parser.add_argument("--v-extend", type=float, default=0.1, help="Commanded cruise speed during the extend phase (m/s)")

    parser.add_argument("--pos-noise-std", type=float, default=0.02, help="Position measurement noise std-dev (m)")
    parser.add_argument("--process-noise-std", type=float, default=0.15,
                         help="Assumed acceleration-disturbance noise std-dev (m/s^2)")
    parser.add_argument("--zupt-noise-std", type=float, default=0.01,
                         help="ZUPT pseudo-measurement noise std-dev (m/s) -- how confidently the "
                              "filter trusts 'velocity is exactly zero' when it applies a ZUPT update")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.05,
                         help="Std-dev used to perturb the initial position guess (m)")
    parser.add_argument("--init-vel-noise-std", type=float, default=0.05,
                         help="Std-dev used to perturb the initial velocity guess (m/s)")

    parser.add_argument("--n-mc-trials", type=int, default=500, help="Number of Monte Carlo consistency trials")

    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    x_true, z, is_anchor, is_cruise = generate_ground_truth_and_data(
        args.duration, args.dt, args.t_anchor, args.t_extend, args.t_ramp, args.v_extend,
        args.pos_noise_std, rng)
    n_steps = len(x_true) - 1

    init_std = np.array([args.init_pos_noise_std, args.init_vel_noise_std])
    x_init = x_true[0] + rng.normal(0.0, 1.0, 2) * init_std
    P_init = np.diag(init_std ** 2)
    R_pos = args.pos_noise_std ** 2
    R_zupt = args.zupt_noise_std ** 2

    print("Running EKF with zupt_mode='never' (position-only, ignores the anchor phase entirely)...")
    (x_never, _), time_never, mem_never = measure_performance(
        run_ekf_never_zupt, x_init, P_init, z, args.dt, n_steps, args.process_noise_std, R_pos,
        R_zupt, is_anchor, n_steps=n_steps)

    print("Running EKF with zupt_mode='always' (ZUPT applied every tick, regardless of phase)...")
    (x_always, _), time_always, mem_always = measure_performance(
        run_ekf_always_zupt, x_init, P_init, z, args.dt, n_steps, args.process_noise_std, R_pos,
        R_zupt, is_anchor, n_steps=n_steps)

    print("Running EKF with zupt_mode='phase_conditional' (ZUPT only on known anchor ticks)...")
    (x_phase, _), time_phase, mem_phase = measure_performance(
        run_ekf_phase_conditional_zupt, x_init, P_init, z, args.dt, n_steps, args.process_noise_std,
        R_pos, R_zupt, is_anchor, n_steps=n_steps)

    print(f"Running Monte Carlo consistency check ({args.n_mc_trials} trials)...")
    nees_never, nees_always, nees_phase = run_monte_carlo_consistency(
        x_true, args.dt, args.process_noise_std, R_pos, R_zupt, is_anchor,
        args.init_pos_noise_std, args.init_vel_noise_std, args.n_mc_trials, rng)

    pos_err_never, vel_err_never = state_errors(x_true, x_never)
    pos_err_always, vel_err_always = state_errors(x_true, x_always)
    pos_err_phase, vel_err_phase = state_errors(x_true, x_phase)

    print("\nRMS errors (single run):")
    for name, pos_err, vel_err in [
        ("never", pos_err_never, vel_err_never),
        ("always", pos_err_always, vel_err_always),
        ("phase_conditional", pos_err_phase, vel_err_phase),
    ]:
        print(f"  {name:<18s} RMS pos={np.sqrt(np.mean(pos_err**2)):7.4f} m, "
              f"vel={np.sqrt(np.mean(vel_err**2)):7.4f} m/s")

    print(f"\nMonte Carlo NEES (consistent 2-DoF filter should average ~2.0 everywhere):")
    print("  (anchor-only/cruise-only exclude ramp ticks -- the shared, all-3-variants transition")
    print("  spike that isn't the point here; see true_velocity's docstring)")
    for name, nees_arr in [("never", nees_never), ("always", nees_always), ("phase_conditional", nees_phase)]:
        print(f"  {name:<18s} overall={np.mean(nees_arr):8.2f} | "
              f"anchor-only={np.mean(nees_arr[is_anchor]):8.2f} | "
              f"cruise-only={np.mean(nees_arr[is_cruise]):8.2f}")

    print("\nAverage time complexity + space complexity per approach (per-step, empirical):")
    for name, avg_time, avg_mem in [
        ("never", time_never, mem_never),
        ("always", time_always, mem_always),
        ("phase_conditional", time_phase, mem_phase),
    ]:
        print(f"  {name:<18s} avg time={avg_time * 1e6:9.2f} us/step | avg peak mem={avg_mem / 1024.0:9.3f} KB/step")

    t_hist = np.arange(len(x_true)) * args.dt
    fig, (ax_vel, ax_err, ax_nees) = plt.subplots(3, 1, figsize=(9, 11))

    ax_vel.plot(t_hist, x_true[:, 1], label="Ground truth", color="green", linewidth=2, linestyle="--")
    ax_vel.plot(t_hist, x_never[:, 1], label="never", color="tab:gray")
    ax_vel.plot(t_hist, x_always[:, 1], label="always", color="tab:red")
    ax_vel.plot(t_hist, x_phase[:, 1], label="phase_conditional", color="tab:blue")
    ax_vel.set_ylabel("Velocity v (m/s)")
    ax_vel.set_title("Phase-conditional ZUPT: inchworm anchor/extend gait")
    ax_vel.legend()

    ax_err.plot(t_hist, vel_err_never, label="never", color="tab:gray")
    ax_err.plot(t_hist, vel_err_always, label="always", color="tab:red")
    ax_err.plot(t_hist, vel_err_phase, label="phase_conditional", color="tab:blue")
    ax_err.set_ylabel("Velocity error (m/s)")
    ax_err.set_yscale("log")
    ax_err.legend()

    ax_nees.plot(t_hist, nees_never, label="never", color="tab:gray")
    ax_nees.plot(t_hist, nees_always, label="always", color="tab:red")
    ax_nees.plot(t_hist, nees_phase, label="phase_conditional", color="tab:blue")
    ax_nees.axhline(2.0, color="black", linestyle="-", linewidth=1, label="Expected NEES (2 DoF)")
    ax_nees.axhline(CHI2_2DOF_95, color="black", linestyle="--", linewidth=1, label="Chi-squared 95% bound")
    ax_nees.set_ylabel(f"Monte Carlo avg. NEES ({args.n_mc_trials} trials)")
    ax_nees.set_xlabel("Time (s)")
    ax_nees.set_yscale("log")
    ax_nees.legend()

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
