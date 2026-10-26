'''
Side extension of inchworm_zupt_ekf.py: a planar inchworm with a heading. The
state is `x = [px, py, theta, v, omega]` (position, heading, forward speed,
turn rate), a unicycle, and the same known anchor/extend schedule gates TWO
free pseudo-measurements: the zero-velocity update (ZUPT, v = 0) and the
zero-angular-rate update (ZARU, omega = 0). The 1D script's Scope section
leaves ZARU out because it needs an orientation state; this one carries it.

Why this is not just "the 1D script twice": with isotropic noise a 2D
constant-velocity filter splits into two independent 1D filters and shows
nothing new. The heading couples the axes (px' = v cos(theta), py' = v
sin(theta)), which makes the model nonlinear, so this one is a real EKF with
an analytic Jacobian (`motion_jacobian`), unlike the 1D script, which its
doc calls "really a plain KF".

Only the position (px, py) is measured; there is no heading sensor. So theta
is observable only through the direction of travel while extending, and is
unobservable at rest. During an anchor, ZARU pins omega to zero, which is what
could keep theta from drifting on omega's uncertainty.

The truth turns during extend (omega follows the same trapezoid as v), so an
unconditional ZARU is a false claim for part of every cycle, the same way an
unconditional ZUPT is.

Gating policies, each applied jointly to ZUPT and ZARU on the same ticks:

  - never: only the position measurement.
  - always: position + ZUPT + ZARU every tick, whatever the true phase.
  - phase_conditional: position every tick, ZUPT + ZARU only on the known
    anchor ticks.
  - phase_conditional_zupt_only: an ablation (not one of the three policies of
    the 1D script): ZUPT only on anchor ticks, no ZARU, to isolate what ZARU
    itself adds.

Truth follows the repo's convention: it is integrated with the filter's own
discrete motion model (`propagate`) using the true (v, omega), with no process
noise, and the Monte Carlo redraws the initial error and the measurement noise.
NEES uses all 5 states with the heading error wrapped to (-pi, pi].
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt

from state_estimation_simulations.utils import measure_performance, averaged_nees_bounds, nees  # noqa: E402
from state_estimation_simulations.use_numpy.inchworm_zupt_ekf import is_anchor_phase, is_cruise_phase, true_velocity  # noqa: E402

GATING_MODES = ("never", "always", "phase_conditional", "phase_conditional_zupt_only")
STATE_DIM = 5  # [px, py, theta, v, omega]


def wrap_angle(a):
    """Wraps an angle (or array of angles) to (-pi, pi].
    Arguments:
        a: angle(s) in radians
    Returns:
        the equivalent angle(s) in (-pi, pi]
    """
    return np.pi - np.mod(np.pi - a, 2.0 * np.pi)


def propagate(x, dt):
    """The unicycle motion model over one tick (left-Euler): position moves
    along the current heading at speed v, heading turns at omega, and v and
    omega stay constant (they wander only through the process noise).
    Arguments:
        x: state [px, py, theta, v, omega] (5,)
        dt: propagation interval (s)
    Returns:
        x_new: propagated state (5,)
    """
    px, py, th, v, om = x
    return np.array([px + v * np.cos(th) * dt,
                     py + v * np.sin(th) * dt,
                     wrap_angle(th + om * dt),
                     v,
                     om])


def motion_jacobian(x, dt):
    """Jacobian F = d propagate / d x at x. Only the theta column is
    state-dependent (through cos/sin of the heading); the wrap in `propagate`
    does not change the derivative.
    Arguments:
        x: state [px, py, theta, v, omega] (5,)
        dt: propagation interval (s)
    Returns:
        F: (5,5) Jacobian
    """
    th, v = x[2], x[3]
    F = np.eye(STATE_DIM)
    F[0, 2] = -v * np.sin(th) * dt
    F[0, 3] = np.cos(th) * dt
    F[1, 2] = v * np.cos(th) * dt
    F[1, 3] = np.sin(th) * dt
    F[2, 4] = dt
    return F


def process_noise_covariance(x, dt, accel_std, ang_accel_std):
    """Q from a random linear acceleration (along the heading) and a random
    angular acceleration, both held over one tick: Q = G diag(sa^2, sα^2) G^T.
    Position gets 1/2 dt^2 of the linear acceleration along the heading (no
    sideways noise: a unicycle cannot accelerate sideways), theta gets 1/2 dt^2
    of the angular acceleration, and v and omega get dt of theirs. Unlike the 1D
    script's diagonal heuristic this keeps the correlation between a state and
    its rate.
    Arguments:
        x: state at the start of the tick (5,); only theta is used
        dt: propagation interval (s)
        accel_std: std-dev of the assumed linear acceleration disturbance (m/s^2)
        ang_accel_std: std-dev of the assumed angular acceleration disturbance (rad/s^2)
    Returns:
        Q: (5,5) process-noise covariance
    """
    th = x[2]
    G = np.zeros((STATE_DIM, 2))
    G[0, 0] = 0.5 * dt * dt * np.cos(th)
    G[1, 0] = 0.5 * dt * dt * np.sin(th)
    G[3, 0] = dt
    G[2, 1] = 0.5 * dt * dt
    G[4, 1] = dt
    return G @ np.diag([accel_std ** 2, ang_accel_std ** 2]) @ G.T


def generate_ground_truth_and_data(duration, dt, t_anchor, t_extend, t_ramp, v_extend, omega_extend,
                                   pos_noise_std, rng, theta0=0.0):
    """Builds the true gait trajectory and noisy position measurements.
    v follows the 1D script's trapezoid (0 in anchor, ramp up, cruise at
    v_extend, ramp down), and omega follows the same trapezoid with peak
    omega_extend, so the body turns while it extends and is exactly still
    while anchored. The state is advanced with `propagate` using the true
    (v, omega) of the current tick; the new tick's v and omega are then set
    from the schedule.
    Arguments:
        duration, dt: total simulation time and step (s)
        t_anchor, t_extend, t_ramp: phase durations as in inchworm_zupt_ekf.py (s)
        v_extend: commanded cruise speed (m/s)
        omega_extend: commanded cruise turn rate (rad/s)
        pos_noise_std: std-dev of the Gaussian noise on each position coordinate (m)
        rng: numpy random number generator
        theta0: initial heading (rad)
    Returns:
        x_true: (n_steps+1, 5) true [px, py, theta, v, omega]
        z: (n_steps+1, 2) noisy position measurements
        is_anchor: (n_steps+1,) bool, True on anchor ticks
        is_cruise: (n_steps+1,) bool, True on steady-cruise extend ticks
    """
    n_steps = int(duration / dt)
    ticks = np.arange(n_steps + 1) * dt
    v_true = np.array([true_velocity(t, t_anchor, t_extend, t_ramp, v_extend) for t in ticks])
    om_true = np.array([true_velocity(t, t_anchor, t_extend, t_ramp, omega_extend) for t in ticks])

    x_true = np.zeros((n_steps + 1, STATE_DIM))
    x_true[0] = [0.0, 0.0, theta0, v_true[0], om_true[0]]
    for k in range(n_steps):
        x_next = propagate(x_true[k], dt)
        x_next[3], x_next[4] = v_true[k + 1], om_true[k + 1]
        x_true[k + 1] = x_next

    is_anchor = np.array([is_anchor_phase(t, t_anchor, t_extend) for t in ticks])
    is_cruise = np.array([is_cruise_phase(t, t_anchor, t_extend, t_ramp) for t in ticks])
    z = x_true[:, :2] + rng.normal(0.0, pos_noise_std, (n_steps + 1, 2))
    return x_true, z, is_anchor, is_cruise


def predict(x, P, dt, accel_std, ang_accel_std):
    """EKF predict: x- = f(x), P- = F P F^T + Q, with F and Q evaluated at the
    current estimate. Identical for every gating policy and both gait phases.
    Arguments:
        x: state estimate (5,)
        P: (5,5) covariance
        dt: propagation interval (s)
        accel_std: std-dev of the assumed linear acceleration disturbance (m/s^2)
        ang_accel_std: std-dev of the assumed angular acceleration disturbance (rad/s^2)
    Returns:
        x_new: predicted state (5,)
        P_new: predicted covariance (5,5)
    """
    F = motion_jacobian(x, dt)
    Q = process_noise_covariance(x, dt, accel_std, ang_accel_std)
    return propagate(x, dt), F @ P @ F.T + Q


def _kf_update(x, P, z, H, R):
    """Shared linear-Gaussian update used by all three measurement functions
    (the measurement models here are linear in the state, so H is exact).
    """
    r = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x_new = x + K @ r
    x_new[2] = wrap_angle(x_new[2])
    return x_new, (np.eye(STATE_DIM) - K @ H) @ P


def measurement_update_position(x, P, z, R_pos):
    """Position update: z = [px, py] + noise, H = [I2 0].
    Arguments:
        x, P: predicted state (5,) and covariance (5,5)
        z: noisy position measurement (2,)
        R_pos: position measurement noise variance (scalar, same on both axes)
    Returns:
        updated state and covariance
    """
    H = np.zeros((2, STATE_DIM))
    H[0, 0] = H[1, 1] = 1.0
    return _kf_update(x, P, z, H, R_pos * np.eye(2))


def measurement_update_zupt(x, P, R_zupt):
    """ZUPT pseudo-measurement: "v = 0", H = e_v^T.
    Arguments:
        x, P: predicted state (5,) and covariance (5,5)
        R_zupt: ZUPT noise variance (scalar, (m/s)^2)
    Returns:
        updated state and covariance
    """
    H = np.zeros((1, STATE_DIM))
    H[0, 3] = 1.0
    return _kf_update(x, P, np.array([0.0]), H, np.array([[R_zupt]]))


def measurement_update_zaru(x, P, R_zaru):
    """ZARU pseudo-measurement: "omega = 0", H = e_omega^T.
    Arguments:
        x, P: predicted state (5,) and covariance (5,5)
        R_zaru: ZARU noise variance (scalar, (rad/s)^2)
    Returns:
        updated state and covariance
    """
    H = np.zeros((1, STATE_DIM))
    H[0, 4] = 1.0
    return _kf_update(x, P, np.array([0.0]), H, np.array([[R_zaru]]))


def run_ekf(x_init, P_init, z, dt, n_steps, accel_std, ang_accel_std, R_pos, R_zupt, R_zaru,
            is_anchor, gating_mode):
    """Recursive EKF: predict, position update every tick, then (by policy)
    the ZUPT and ZARU updates. The policies differ only in which of the two
    pseudo-measurements run on which ticks, so any difference in their output
    is attributable to that gating alone.
    Arguments:
        x_init: initial state estimate (5,)
        P_init: initial covariance (5,5)
        z: (n_steps+1, 2) noisy position measurements
        dt: step interval (s)
        n_steps: number of steps
        accel_std, ang_accel_std: assumed acceleration disturbances (m/s^2, rad/s^2)
        R_pos, R_zupt, R_zaru: measurement noise variances (scalars)
        is_anchor: (n_steps+1,) bool, True where that tick is an anchor tick
        gating_mode: one of GATING_MODES
    Returns:
        x_list: (n_steps+1, 5) estimated states
        P_list: list of (5,5) estimated covariances
    """
    if gating_mode not in GATING_MODES:
        raise ValueError(f"unknown gating_mode: {gating_mode!r}")

    x, P = x_init.copy(), P_init.copy()
    x_list, P_list = [x], [P]
    for k in range(n_steps):
        x, P = predict(x, P, dt, accel_std, ang_accel_std)
        x, P = measurement_update_position(x, P, z[k + 1], R_pos)
        anchored = bool(is_anchor[k + 1])
        if gating_mode == "always" or (gating_mode != "never" and anchored):
            x, P = measurement_update_zupt(x, P, R_zupt)
        if gating_mode == "always" or (gating_mode == "phase_conditional" and anchored):
            x, P = measurement_update_zaru(x, P, R_zaru)
        x_list.append(x)
        P_list.append(P)
    return np.array(x_list), P_list


def state_error_vector(x_true, x_est):
    """Estimation error x_true - x_est with the heading component wrapped to
    (-pi, pi], for NEES and for the per-state error magnitudes.
    Arguments:
        x_true, x_est: (5,) or (N,5) true and estimated states
    Returns:
        the same-shaped error
    """
    err = np.array(x_true - x_est, dtype=float)
    err[..., 2] = wrap_angle(err[..., 2])
    return err


def state_errors(x_true, x_est):
    """Per-step error magnitudes.
    Arguments:
        x_true, x_est: (N,5) true and estimated states
    Returns:
        pos_err: (N,) Euclidean position error (m)
        heading_err: (N,) absolute wrapped heading error (rad)
        vel_err: (N,) absolute speed error (m/s)
        omega_err: (N,) absolute turn-rate error (rad/s)
    """
    err = state_error_vector(x_true, x_est)
    return np.linalg.norm(err[:, :2], axis=1), np.abs(err[:, 2]), np.abs(err[:, 3]), np.abs(err[:, 4])


def run_monte_carlo_consistency(x_true, dt, accel_std, ang_accel_std, R_pos, R_zupt, R_zaru, is_anchor,
                                init_std, n_trials, rng, modes=GATING_MODES):
    """Repeats every gating policy over n_trials independent noise
    realizations of the same true trajectory (fresh initial error and fresh
    measurement noise each trial) and returns the per-step average NEES.
    Arguments:
        x_true: (N,5) true states
        dt: step interval (s)
        accel_std, ang_accel_std: assumed acceleration disturbances
        R_pos, R_zupt, R_zaru: measurement noise variances
        is_anchor: (N,) bool anchor flags
        init_std: (5,) std-dev of the initial error and of P_init
        n_trials: number of Monte Carlo trials
        rng: numpy random number generator
        modes: gating policies to run
    Returns:
        dict mode -> (N,) average NEES per step (5 DoF)
    """
    n_steps = len(x_true) - 1
    n_pts = n_steps + 1
    P_init = np.diag(init_std ** 2)
    R_pos_std = np.sqrt(R_pos)
    sums = {m: np.zeros(n_pts) for m in modes}
    for _ in range(n_trials):
        x_init = x_true[0] + rng.normal(0.0, 1.0, STATE_DIM) * init_std
        z_trial = x_true[:, :2] + rng.normal(0.0, R_pos_std, (n_pts, 2))
        for m in modes:
            x_est, P_est = run_ekf(x_init, P_init, z_trial, dt, n_steps, accel_std, ang_accel_std,
                                   R_pos, R_zupt, R_zaru, is_anchor, m)
            for k in range(n_pts):
                sums[m][k] += nees(state_error_vector(x_true[k], x_est[k]), P_est[k])
    return {m: s / n_trials for m, s in sums.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--duration", type=float, default=10.0, help="Simulation length in seconds")
    parser.add_argument("--dt", type=float, default=0.05, help="Filter/measurement step interval in seconds")
    parser.add_argument("--t-anchor", type=float, default=1.0, help="Anchor/dwell phase duration per cycle (s)")
    parser.add_argument("--t-extend", type=float, default=1.0,
                        help="Total extend phase duration per cycle (s), including both ramps")
    parser.add_argument("--t-ramp", type=float, default=0.2,
                        help="Ramp-up/ramp-down duration at each end of the extend phase (s)")
    parser.add_argument("--v-extend", type=float, default=0.1, help="Commanded cruise speed during extend (m/s)")
    parser.add_argument("--omega-extend", type=float, default=0.3,
                        help="Commanded cruise turn rate during extend (rad/s); 0 makes the body go straight, "
                             "where an unconditional ZARU is harmless")

    parser.add_argument("--pos-noise-std", type=float, default=0.02, help="Position noise std-dev per axis (m)")
    parser.add_argument("--process-noise-std", type=float, default=0.15,
                        help="Assumed linear-acceleration disturbance std-dev (m/s^2)")
    parser.add_argument("--process-noise-ang-std", type=float, default=0.45,
                        help="Assumed angular-acceleration disturbance std-dev (rad/s^2); the true ramp "
                             "angular acceleration is omega-extend / t-ramp")
    parser.add_argument("--zupt-noise-std", type=float, default=0.01, help="ZUPT noise std-dev (m/s)")
    parser.add_argument("--zaru-noise-std", type=float, default=0.02, help="ZARU noise std-dev (rad/s)")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.05, help="Initial position error std-dev (m)")
    parser.add_argument("--init-heading-noise-std", type=float, default=0.1,
                        help="Initial heading error std-dev (rad)")
    parser.add_argument("--init-vel-noise-std", type=float, default=0.05, help="Initial speed error std-dev (m/s)")
    parser.add_argument("--init-omega-noise-std", type=float, default=0.05,
                        help="Initial turn-rate error std-dev (rad/s)")

    parser.add_argument("--n-mc-trials", type=int, default=500, help="Number of Monte Carlo consistency trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    x_true, z, is_anchor, is_cruise = generate_ground_truth_and_data(
        args.duration, args.dt, args.t_anchor, args.t_extend, args.t_ramp, args.v_extend,
        args.omega_extend, args.pos_noise_std, rng)
    n_steps = len(x_true) - 1

    init_std = np.array([args.init_pos_noise_std, args.init_pos_noise_std, args.init_heading_noise_std,
                         args.init_vel_noise_std, args.init_omega_noise_std])
    x_init = x_true[0] + rng.normal(0.0, 1.0, STATE_DIM) * init_std
    P_init = np.diag(init_std ** 2)
    R_pos, R_zupt, R_zaru = args.pos_noise_std ** 2, args.zupt_noise_std ** 2, args.zaru_noise_std ** 2

    runs = {}
    for m in GATING_MODES:
        print(f"Running EKF with gating_mode='{m}'...")
        (x_est, P_est), t_step, mem = measure_performance(
            run_ekf, x_init, P_init, z, args.dt, n_steps, args.process_noise_std,
            args.process_noise_ang_std, R_pos, R_zupt, R_zaru, is_anchor, m, n_steps=n_steps)
        runs[m] = (x_est, P_est, t_step, mem)

    print(f"Running Monte Carlo consistency check ({args.n_mc_trials} trials)...")
    nees_by_mode = run_monte_carlo_consistency(
        x_true, args.dt, args.process_noise_std, args.process_noise_ang_std, R_pos, R_zupt, R_zaru,
        is_anchor, init_std, args.n_mc_trials, rng)

    print("\nRMS errors (single run):")
    for m in GATING_MODES:
        pe, he, ve, oe = state_errors(x_true, runs[m][0])
        rms = lambda e: np.sqrt(np.mean(e ** 2))  # noqa: E731
        print(f"  {m:<28s} pos={rms(pe):7.4f} m, heading={rms(he):7.4f} rad, "
              f"v={rms(ve):7.4f} m/s, omega={rms(oe):7.4f} rad/s")

    nees_lo, nees_hi = averaged_nees_bounds(STATE_DIM, args.n_mc_trials)
    print(f"\nMonte Carlo NEES (consistent {STATE_DIM}-DoF filter should average ~{STATE_DIM}; "
          f"95% bounds for a {args.n_mc_trials}-trial average: [{nees_lo:.2f}, {nees_hi:.2f}]):")
    print("  (anchor-only/cruise-only exclude ramp ticks, as in inchworm_zupt_ekf.py)")
    for m in GATING_MODES:
        n = nees_by_mode[m]
        print(f"  {m:<28s} overall={np.mean(n):9.2f} | anchor-only={np.mean(n[is_anchor]):9.2f} | "
              f"cruise-only={np.mean(n[is_cruise]):9.2f}")

    print("\nAverage time (per-step) and peak memory (whole-run) per approach, empirical:")
    for m in GATING_MODES:
        print(f"  {m:<28s} avg time={runs[m][2] * 1e6:9.2f} us/step | peak mem={runs[m][3] / 1024.0:9.3f} KB")

    t_hist = np.arange(len(x_true)) * args.dt
    colors = {"never": "tab:gray", "always": "tab:red", "phase_conditional": "tab:blue",
              "phase_conditional_zupt_only": "tab:orange"}
    fig, (ax_th, ax_om, ax_nees) = plt.subplots(3, 1, figsize=(9, 11))

    ax_th.plot(t_hist, x_true[:, 2], label="Ground truth", color="green", linewidth=2, linestyle="--")
    ax_om.plot(t_hist, x_true[:, 4], label="Ground truth", color="green", linewidth=2, linestyle="--")
    for m in GATING_MODES:
        ax_th.plot(t_hist, runs[m][0][:, 2], label=m, color=colors[m])
        ax_om.plot(t_hist, runs[m][0][:, 4], label=m, color=colors[m])
        ax_nees.plot(t_hist, nees_by_mode[m], label=m, color=colors[m])
    ax_th.set_ylabel("Heading (rad)")
    ax_th.set_title("Planar inchworm: ZUPT + ZARU gated by the anchor schedule")
    ax_th.legend()
    ax_om.set_ylabel("Turn rate omega (rad/s)")
    ax_om.legend()
    ax_nees.axhline(STATE_DIM, color="black", linestyle="-", linewidth=1, label=f"Expected NEES ({STATE_DIM} DoF)")
    ax_nees.axhline(nees_lo, color="black", linestyle="--", linewidth=1,
                    label=f"95% bounds for a {args.n_mc_trials}-trial average")
    ax_nees.axhline(nees_hi, color="black", linestyle="--", linewidth=1)
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
