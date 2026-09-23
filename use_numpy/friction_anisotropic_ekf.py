'''
Tracks a 2D unicycle robot crawling on a friction-anisotropic pad (grips well
along one axis, slides easily along the other -- "the way a snake's belly
scales do") that drives at a *known* constant commanded forward speed v_cmd
and turn rate omega_cmd, so its true path is an exact circular arc and its
heading increases smoothly and continuously throughout -- deliberately not
hybrid/switched, unlike saltation_matrix_ekf.py's bounce or
inchworm_zupt_ekf.py's anchor/extend gait. State `x = [p_x, p_y, theta]`.

This is the simplest concrete form of a claim about friction-anisotropic
locomotion: it needs heading-dependent process noise, not just another
hybrid mode. The pad's slip is anisotropic in its own body
frame (low variance along the grip/forward axis, high variance along the
slip/lateral axis), not in the world frame -- so a real slip event, expressed
in world coordinates, is a random displacement drawn in the body frame and
then rotated by the *true* heading at that instant. `pointcloud_pose_tracking_
empirical_note.md`'s §A.1/§2 already established the general shape of this
argument (a noise ellipsoid fixed in an object's own frame looks anisotropic-
and-rotating in the world frame; what matters is whether the noise model
tracks that rotation, not "rigid vs. non-rigid") for *measurement* noise in a
point-cloud registration EKF/IEKF -- this script is the first to apply the
same argument to *process* noise for a moving-and-turning robot instead, and
deliberately stays an ordinary EKF throughout (no new IEKF/Lie-group
derivation) so the comparison isolates one thing: does Q's *shape* correctly
track heading, not any linearization-frame question the sibling doc's own
EKF/IEKF split already covers.

Three ways of building the position block of the process-noise covariance Q
are compared, all sharing the same predict step (exact-arc propagation) and
the same noisy position measurement each tick -- they differ *only* in
`anisotropic_Q_pos`'s reference heading, or whether it's used at all:

  - isotropic: direction-blind, `(sigma_grip^2+sigma_slip^2)/2 * I` -- same
    *total* noise budget as the other two (a fair comparison, not a rigged
    one), just spread uniformly instead of along the pad's real axes.

  - fixed_anisotropic: correctly anisotropic in shape, but oriented to a
    *fixed* reference heading (theta at t=0) and never updated as the robot
    turns -- the mistake of knowing the pad is anisotropic without re-
    deriving Q as heading changes.

  - heading_aware: anisotropic and re-oriented every predict step using the
    filter's own current heading estimate -- the correct policy.

A Monte Carlo consistency check (`run_monte_carlo_consistency`, following the
sibling scripts' NEES-based pattern) reports NEES as a function of how far
the true heading has rotated away from the reference heading `fixed_
anisotropic` is stuck with. `fixed_anisotropic` is dramatically worse than
both `isotropic` and `heading_aware` throughout -- robust across seeds, at
every checkpoint. Its *own* worst checkpoint is consistently a ~90-degree
heading mismatch, not the largest possible (~180-degree) one: a covariance
ellipse `R(theta) @ diag(a,b) @ R(theta).T` has period pi in theta (it's the
same ellipse at a 180-degree difference as at 0), so the orientation mismatch
peaks at 90 degrees, not 180. What happens *beyond* that peak, out toward a
full 180-degree difference, was not robust across seeds when checked by hand
(NEES partially recovers in some seeds, keeps climbing in others) -- most
likely accumulated trajectory drift competing with the pure orientation-
mismatch effect over a longer run, not a clean single-cause story, and this
script deliberately doesn't oversell it as one. `isotropic` stays uniformly
mediocre throughout (never claims a direction to be wrong about, but never
gets the anisotropy's benefit either); `heading_aware` stays uniformly good.
'''

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import measure_performance

# chi-squared, 3 degrees of freedom, 95th percentile (scipy.stats.chi2.ppf(0.95, 3));
# hardcoded rather than importing scipy, since this script is otherwise pure numpy
# and this is only needed as a reference line on the consistency plot.
CHI2_3DOF_95 = 7.8147


def rotation_2d(theta):
    """2x2 rotation matrix for angle theta (radians), world_vec = R @ body_vec.
    Arguments:
        theta: rotation angle (rad)
    Returns:
        R: (2,2) rotation matrix
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def exact_arc_step(x, v, omega, dt):
    """Exact closed-form unicycle propagation over an interval dt at constant
    commanded forward speed v and turn rate omega, plus its exact Jacobian.
    Exact for any dt (a genuine circular arc, or a straight line in the
    omega->0 limit), not a linearization of the mean -- the EKF still needs
    the Jacobian F below since theta enters the position update nonlinearly,
    unlike saltation_matrix_ekf.flow's fully-linear free-fall dynamics.
    Arguments:
        x: state [p_x, p_y, theta] (3,)
        v: commanded forward speed (m/s)
        omega: commanded turn rate (rad/s)
        dt: propagation interval (s)
    Returns:
        x_new: propagated state (3,)
        F: (3,3) exact Jacobian dx_new/dx
    """
    p_x, p_y, theta = x
    if abs(omega) < 1e-9:
        # straight-line limit: the arc formulas below divide by omega, so
        # fall back to the omega->0 Taylor limit directly (exact for omega=0,
        # and avoids a 0/0 for the tiny-but-nonzero omega this threshold lets
        # through, where the two forms agree to machine precision anyway).
        dx, dy = v * dt * np.cos(theta), v * dt * np.sin(theta)
        theta_new = theta
        F = np.array([[1.0, 0.0, -v * dt * np.sin(theta)],
                      [0.0, 1.0, v * dt * np.cos(theta)],
                      [0.0, 0.0, 1.0]])
    else:
        theta_new = theta + omega * dt
        r = v / omega
        dx = r * (np.sin(theta_new) - np.sin(theta))
        dy = -r * (np.cos(theta_new) - np.cos(theta))
        F = np.array([[1.0, 0.0, r * (np.cos(theta_new) - np.cos(theta))],
                      [0.0, 1.0, r * (np.sin(theta_new) - np.sin(theta))],
                      [0.0, 0.0, 1.0]])
    x_new = np.array([p_x + dx, p_y + dy, theta_new])
    return x_new, F


def generate_ground_truth_and_data(duration, dt, v_cmd, omega_cmd, sigma_grip, sigma_slip,
                                    pos_noise_std, rng):
    """Builds the true trajectory (exact commanded arc, plus a random
    friction-anisotropic slip disturbance added to position each tick) and
    noisy position measurements a tracker would actually receive.

    The slip disturbance is drawn in the pad's own body frame each tick --
    `N(0, diag(sigma_grip^2, sigma_slip^2)) * dt` (the same "simple
    heuristic, std times dt" scaling as saltation_matrix_ekf.
    process_noise_covariance, not textbook continuous white noise -- flagged
    there as a deliberate simplification, and reused here for the same
    reason: this script's process-noise scaling convention is a tuning knob,
    not the object of study) -- then rotated into world coordinates by the
    *true* heading at that instant before being added to position. Heading
    itself is exact/noise-free, driven only by the known commanded omega_cmd
    -- an explicit simplification that isolates the position-slip-anisotropy
    question from a second one about heading noise, which this script isn't
    about.
    Arguments:
        duration: total simulation time (s)
        dt: measurement/filter step interval (s)
        v_cmd: constant commanded forward speed (m/s)
        omega_cmd: constant commanded turn rate (rad/s)
        sigma_grip: std-dev of body-frame slip along the grip/forward axis (m/s)
        sigma_slip: std-dev of body-frame slip along the slip/lateral axis (m/s)
        pos_noise_std: std-dev of Gaussian noise added to position measurements (m)
        rng: numpy random number generator
    Returns:
        x_true: (n_steps+1, 3) array of true [p_x, p_y, theta] at each tick
        z: (n_steps+1, 2) array of noisy position measurements
    """
    n_steps = int(duration / dt)
    x_true = np.zeros((n_steps + 1, 3))
    x = np.zeros(3)
    x_true[0] = x
    for k in range(n_steps):
        x, _ = exact_arc_step(x, v_cmd, omega_cmd, dt)
        slip_body = rng.normal(0.0, 1.0, 2) * np.array([sigma_grip, sigma_slip]) * dt
        x[0:2] += rotation_2d(x[2]) @ slip_body
        x_true[k + 1] = x

    z = x_true[:, 0:2] + rng.normal(0.0, pos_noise_std, (n_steps + 1, 2))
    return x_true, z


def isotropic_Q_pos(sigma_grip, sigma_slip, dt):
    """Direction-blind position process-noise block, carrying the *same
    total* variance (trace) as anisotropic_Q_pos -- a fair baseline, not one
    handicapped by an unfair overall noise-magnitude difference.
    Arguments:
        sigma_grip, sigma_slip: body-frame slip std-devs (m/s)
        dt: propagation interval (s)
    Returns:
        Q_pos: (2,2) covariance
    """
    sigma_iso_sq = 0.5 * (sigma_grip ** 2 + sigma_slip ** 2)
    return sigma_iso_sq * dt * dt * np.eye(2)


def anisotropic_Q_pos(theta, sigma_grip, sigma_slip, dt):
    """Position process-noise block, anisotropic in the pad's body frame
    (low variance along the grip/forward axis, high along the slip/lateral
    axis) and rotated into world coordinates by theta.
    Arguments:
        theta: reference heading to orient the anisotropy by (rad) -- the
            filter's current heading estimate for "heading_aware", or a
            fixed reference heading for "fixed_anisotropic"
        sigma_grip, sigma_slip: body-frame slip std-devs (m/s)
        dt: propagation interval (s)
    Returns:
        Q_pos: (2,2) covariance
    """
    Q_body = np.diag([sigma_grip ** 2, sigma_slip ** 2]) * dt * dt
    R = rotation_2d(theta)
    return R @ Q_body @ R.T


def predict(x, P, v, omega, dt, q_policy, sigma_grip, sigma_slip, sigma_theta, theta_ref=None):
    """EKF predict step: exact_arc_step for the mean/Jacobian, plus a
    process-noise covariance Q whose position block is built according to
    q_policy -- the single toggle distinguishing the three filter variants
    below; everything else in the predict step is shared.
    Arguments:
        x: state [p_x, p_y, theta] (3,)
        P: (3,3) covariance
        v: commanded forward speed (m/s)
        omega: commanded turn rate (rad/s)
        dt: propagation interval (s)
        q_policy: "isotropic", "fixed_anisotropic", or "heading_aware"
        sigma_grip, sigma_slip: assumed body-frame slip std-devs (m/s)
        sigma_theta: assumed heading process-noise std-dev (rad/s)
        theta_ref: fixed reference heading (rad), required for
            "fixed_anisotropic"; ignored otherwise
    Returns:
        x_new: predicted state (3,)
        P_new: predicted covariance (3,3)
    """
    x_new, F = exact_arc_step(x, v, omega, dt)

    if q_policy == "isotropic":
        Q_pos = isotropic_Q_pos(sigma_grip, sigma_slip, dt)
    elif q_policy == "fixed_anisotropic":
        Q_pos = anisotropic_Q_pos(theta_ref, sigma_grip, sigma_slip, dt)
    elif q_policy == "heading_aware":
        Q_pos = anisotropic_Q_pos(x[2], sigma_grip, sigma_slip, dt)
    else:
        raise ValueError(f"unknown q_policy: {q_policy!r}")

    Q = np.zeros((3, 3))
    Q[0:2, 0:2] = Q_pos
    Q[2, 2] = (sigma_theta * dt) ** 2

    P_new = F @ P @ F.T + Q
    return x_new, P_new


def measurement_update(x, P, z, R_pos):
    """Linear-Gaussian update against a direct noisy position observation
    z = [p_x, p_y] + noise (H = [[1,0,0],[0,1,0]]).
    Arguments:
        x: predicted state (3,)
        P: predicted covariance (3,3)
        z: position measurement (2,)
        R_pos: (2,2) position measurement noise covariance
    Returns:
        x_new: updated state (3,)
        P_new: updated covariance (3,3)
    """
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    r = z - H @ x
    S = H @ P @ H.T + R_pos
    K = P @ H.T @ np.linalg.inv(S)
    x_new = x + K @ r
    P_new = (np.eye(3) - K @ H) @ P
    return x_new, P_new


def run_ekf(x_init, P_init, z, dt, n_steps, v_cmd, omega_cmd, sigma_grip, sigma_slip, sigma_theta,
            R_pos, q_policy, theta_ref=None):
    """Recursive EKF alternating predict with a position measurement update
    every tick. Shared by the three run_ekf_* wrappers below -- they differ
    only in q_policy, so any difference in their output is attributable
    entirely to the process-noise-shape policy, nothing else.
    Arguments:
        x_init: initial state estimate (3,)
        P_init: initial covariance (3,3)
        z: (n_steps+1, 2) array of noisy position measurements
        dt: step interval (s)
        n_steps: number of steps
        v_cmd: constant commanded forward speed (m/s)
        omega_cmd: constant commanded turn rate (rad/s)
        sigma_grip, sigma_slip: assumed body-frame slip std-devs (m/s)
        sigma_theta: assumed heading process-noise std-dev (rad/s)
        R_pos: (2,2) position measurement noise covariance
        q_policy: "isotropic", "fixed_anisotropic", or "heading_aware"
        theta_ref: fixed reference heading (rad), required for "fixed_anisotropic"
    Returns:
        x_list: (n_steps+1, 3) array of estimated states
        P_list: list of (3,3) estimated covariances
    """
    x, P = x_init.copy(), P_init.copy()
    x_list, P_list = [x], [P]
    for k in range(n_steps):
        x, P = predict(x, P, v_cmd, omega_cmd, dt, q_policy, sigma_grip, sigma_slip, sigma_theta,
                        theta_ref)
        x, P = measurement_update(x, P, z[k + 1], R_pos)
        x_list.append(x)
        P_list.append(P)
    return np.array(x_list), P_list


def run_ekf_isotropic(x_init, P_init, z, dt, n_steps, v_cmd, omega_cmd, sigma_grip, sigma_slip,
                       sigma_theta, R_pos, theta_ref=None):
    """run_ekf with q_policy='isotropic' -- see run_ekf's docstring."""
    return run_ekf(x_init, P_init, z, dt, n_steps, v_cmd, omega_cmd, sigma_grip, sigma_slip,
                    sigma_theta, R_pos, q_policy="isotropic", theta_ref=theta_ref)


def run_ekf_fixed_anisotropic(x_init, P_init, z, dt, n_steps, v_cmd, omega_cmd, sigma_grip,
                               sigma_slip, sigma_theta, R_pos, theta_ref):
    """run_ekf with q_policy='fixed_anisotropic' -- see run_ekf's docstring."""
    return run_ekf(x_init, P_init, z, dt, n_steps, v_cmd, omega_cmd, sigma_grip, sigma_slip,
                    sigma_theta, R_pos, q_policy="fixed_anisotropic", theta_ref=theta_ref)


def run_ekf_heading_aware(x_init, P_init, z, dt, n_steps, v_cmd, omega_cmd, sigma_grip, sigma_slip,
                           sigma_theta, R_pos, theta_ref=None):
    """run_ekf with q_policy='heading_aware' -- see run_ekf's docstring."""
    return run_ekf(x_init, P_init, z, dt, n_steps, v_cmd, omega_cmd, sigma_grip, sigma_slip,
                    sigma_theta, R_pos, q_policy="heading_aware", theta_ref=theta_ref)


def state_errors(x_true, x_est):
    """Position and heading error, per step.
    Arguments:
        x_true: (N, 3) array of true states
        x_est: (N, 3) array of estimated states
    Returns:
        pos_err: (N,) array of position error norms (m)
        theta_err: (N,) array of absolute heading error (rad)
    """
    pos_err = np.linalg.norm(x_true[:, 0:2] - x_est[:, 0:2], axis=1)
    theta_err = np.abs(x_true[:, 2] - x_est[:, 2])
    return pos_err, theta_err


def nees(x_true, x_est, P):
    """Normalized Estimation Error Squared: (x_true-x_est)^T @ P^-1 @ (x_true-x_est).
    A consistent filter's NEES should average to the state dimension (3 here)
    across many independent trials; a value that's persistently much larger
    means the filter's reported covariance P is too small (overconfident)
    for the errors it is actually making.
    Arguments:
        x_true: true state (3,)
        x_est: estimated state (3,)
        P: estimated covariance (3,3)
    Returns:
        nees: scalar NEES value
    """
    err = x_true - x_est
    return float(err @ np.linalg.inv(P) @ err)


def run_monte_carlo_consistency(x_true, dt, v_cmd, omega_cmd, sigma_grip, sigma_slip, sigma_theta,
                                 R_pos, theta_ref, init_pos_noise_std, init_theta_noise_std,
                                 n_trials, rng):
    """Repeats the three run_ekf_* variants over n_trials independent noise
    realizations of the same nominal true trajectory (fresh initial-condition
    error and fresh measurement noise each trial), returning the per-step
    average NEES for all three.
    Arguments:
        x_true: (N, 3) array of true states at each tick
        dt: step interval (s)
        v_cmd, omega_cmd: constant commanded forward speed/turn rate
        sigma_grip, sigma_slip: assumed body-frame slip std-devs (m/s)
        sigma_theta: assumed heading process-noise std-dev (rad/s)
        R_pos: (2,2) position measurement noise covariance
        theta_ref: fixed reference heading (rad) for fixed_anisotropic
        init_pos_noise_std: std-dev used to perturb the initial position guess (m)
        init_theta_noise_std: std-dev used to perturb the initial heading guess (rad)
        n_trials: number of Monte Carlo trials
        rng: numpy random number generator
    Returns:
        nees_iso, nees_fixed, nees_aware: (N,) average NEES per step, one
            per q_policy
    """
    n_steps = len(x_true) - 1
    n_pts = n_steps + 1
    init_std = np.array([init_pos_noise_std, init_pos_noise_std, init_theta_noise_std])
    P_init = np.diag(init_std ** 2)
    pos_noise_std = np.sqrt(R_pos[0, 0])

    sum_iso = np.zeros(n_pts)
    sum_fixed = np.zeros(n_pts)
    sum_aware = np.zeros(n_pts)
    for _ in range(n_trials):
        x_init = x_true[0] + rng.normal(0.0, 1.0, 3) * init_std
        z_trial = x_true[:, 0:2] + rng.normal(0.0, pos_noise_std, (n_pts, 2))

        x_iso, P_iso = run_ekf_isotropic(x_init, P_init, z_trial, dt, n_steps, v_cmd, omega_cmd,
                                          sigma_grip, sigma_slip, sigma_theta, R_pos)
        x_fixed, P_fixed = run_ekf_fixed_anisotropic(x_init, P_init, z_trial, dt, n_steps, v_cmd,
                                                      omega_cmd, sigma_grip, sigma_slip,
                                                      sigma_theta, R_pos, theta_ref)
        x_aware, P_aware = run_ekf_heading_aware(x_init, P_init, z_trial, dt, n_steps, v_cmd,
                                                  omega_cmd, sigma_grip, sigma_slip, sigma_theta,
                                                  R_pos)

        for k in range(n_pts):
            sum_iso[k] += nees(x_true[k], x_iso[k], P_iso[k])
            sum_fixed[k] += nees(x_true[k], x_fixed[k], P_fixed[k])
            sum_aware[k] += nees(x_true[k], x_aware[k], P_aware[k])

    return sum_iso / n_trials, sum_fixed / n_trials, sum_aware / n_trials


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--duration", type=float, default=20.0, help="Simulation length in seconds")
    parser.add_argument("--dt", type=float, default=0.05, help="Filter/measurement step interval in seconds")
    parser.add_argument("--v-cmd", type=float, default=0.2, help="Constant commanded forward speed (m/s)")
    parser.add_argument("--omega-cmd", type=float, default=None,
                         help="Constant commanded turn rate (rad/s). Default: 2*pi/duration, i.e. "
                              "exactly one full loop over the simulation, so the heading sweeps "
                              "through every orientation relative to the fixed reference heading")

    parser.add_argument("--sigma-grip", type=float, default=0.02,
                         help="True body-frame slip std-dev along the grip/forward axis (m/s)")
    parser.add_argument("--sigma-slip", type=float, default=0.1,
                         help="True body-frame slip std-dev along the slip/lateral axis (m/s)")
    parser.add_argument("--sigma-theta", type=float, default=0.01,
                         help="Assumed heading process-noise std-dev, all 3 filter variants (rad/s)")
    parser.add_argument("--pos-noise-std", type=float, default=0.02, help="Position measurement noise std-dev (m)")

    parser.add_argument("--init-pos-noise-std", type=float, default=0.05,
                         help="Std-dev used to perturb the initial position guess (m)")
    parser.add_argument("--init-theta-noise-std", type=float, default=0.05,
                         help="Std-dev used to perturb the initial heading guess (rad)")

    parser.add_argument("--n-mc-trials", type=int, default=500, help="Number of Monte Carlo consistency trials")

    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")

    args = parser.parse_args()
    omega_cmd = args.omega_cmd if args.omega_cmd is not None else 2.0 * np.pi / args.duration

    rng = np.random.default_rng(args.seed)

    x_true, z = generate_ground_truth_and_data(
        args.duration, args.dt, args.v_cmd, omega_cmd, args.sigma_grip, args.sigma_slip,
        args.pos_noise_std, rng)
    n_steps = len(x_true) - 1
    theta_ref = x_true[0, 2]  # reference heading fixed_anisotropic is stuck with (0.0 here)

    init_std = np.array([args.init_pos_noise_std, args.init_pos_noise_std, args.init_theta_noise_std])
    x_init = x_true[0] + rng.normal(0.0, 1.0, 3) * init_std
    P_init = np.diag(init_std ** 2)
    R_pos = args.pos_noise_std ** 2 * np.eye(2)

    print("Running EKF with q_policy='isotropic' (direction-blind, same total noise budget)...")
    (x_iso, _), time_iso, mem_iso = measure_performance(
        run_ekf_isotropic, x_init, P_init, z, args.dt, n_steps, args.v_cmd, omega_cmd,
        args.sigma_grip, args.sigma_slip, args.sigma_theta, R_pos, n_steps=n_steps)

    print("Running EKF with q_policy='fixed_anisotropic' (correct shape, fixed to the initial heading)...")
    (x_fixed, _), time_fixed, mem_fixed = measure_performance(
        run_ekf_fixed_anisotropic, x_init, P_init, z, args.dt, n_steps, args.v_cmd, omega_cmd,
        args.sigma_grip, args.sigma_slip, args.sigma_theta, R_pos, theta_ref, n_steps=n_steps)

    print("Running EKF with q_policy='heading_aware' (correct shape, tracks the filter's own heading estimate)...")
    (x_aware, _), time_aware, mem_aware = measure_performance(
        run_ekf_heading_aware, x_init, P_init, z, args.dt, n_steps, args.v_cmd, omega_cmd,
        args.sigma_grip, args.sigma_slip, args.sigma_theta, R_pos, n_steps=n_steps)

    print(f"Running Monte Carlo consistency check ({args.n_mc_trials} trials)...")
    nees_iso, nees_fixed, nees_aware = run_monte_carlo_consistency(
        x_true, args.dt, args.v_cmd, omega_cmd, args.sigma_grip, args.sigma_slip, args.sigma_theta,
        R_pos, theta_ref, args.init_pos_noise_std, args.init_theta_noise_std, args.n_mc_trials, rng)

    pos_err_iso, _ = state_errors(x_true, x_iso)
    pos_err_fixed, _ = state_errors(x_true, x_fixed)
    pos_err_aware, _ = state_errors(x_true, x_aware)

    print("\nRMS position error (single run):")
    for name, pos_err in [("isotropic", pos_err_iso), ("fixed_anisotropic", pos_err_fixed),
                           ("heading_aware", pos_err_aware)]:
        print(f"  {name:<18s} RMS pos={np.sqrt(np.mean(pos_err**2)):7.4f} m")

    # Angular distance under wraparound saturates at pi (a heading exactly opposite the
    # reference), not 2*pi -- so "quarter of the way to maximally wrong" is pi/4, not pi/2.
    rotation_away = np.abs(((x_true[:, 2] - theta_ref + np.pi) % (2 * np.pi)) - np.pi)
    quarter = np.pi / 4
    checkpoints = [("0-quarter turn", rotation_away < quarter),
                   ("quarter-half turn", (rotation_away >= quarter) & (rotation_away < 2 * quarter)),
                   ("half-3quarter turn", (rotation_away >= 2 * quarter) & (rotation_away < 3 * quarter)),
                   ("3quarter-full turn", rotation_away >= 3 * quarter)]
    print("\nMonte Carlo NEES by rotation away from the reference heading "
          "(consistent 3-DoF filter should average ~3.0 everywhere):")
    for label, mask in checkpoints:
        if not np.any(mask):
            continue
        print(f"  {label:<20s} isotropic={np.mean(nees_iso[mask]):8.2f} | "
              f"fixed_anisotropic={np.mean(nees_fixed[mask]):8.2f} | "
              f"heading_aware={np.mean(nees_aware[mask]):8.2f}")

    print("\nAverage time (per-step) and peak memory (whole-run) per approach, empirical:")
    for name, avg_time, avg_mem in [
        ("isotropic", time_iso, mem_iso),
        ("fixed_anisotropic", time_fixed, mem_fixed),
        ("heading_aware", time_aware, mem_aware),
    ]:
        print(f"  {name:<18s} avg time={avg_time * 1e6:9.2f} us/step | peak mem={avg_mem / 1024.0:9.3f} KB")

    t_hist = np.arange(len(x_true)) * args.dt
    fig, (ax_path, ax_err, ax_nees) = plt.subplots(3, 1, figsize=(9, 12))

    ax_path.plot(x_true[:, 0], x_true[:, 1], label="Ground truth", color="green", linewidth=2, linestyle="--")
    ax_path.plot(x_iso[:, 0], x_iso[:, 1], label="isotropic", color="tab:gray")
    ax_path.plot(x_fixed[:, 0], x_fixed[:, 1], label="fixed_anisotropic", color="tab:red")
    ax_path.plot(x_aware[:, 0], x_aware[:, 1], label="heading_aware", color="tab:blue")
    ax_path.set_xlabel("p_x (m)")
    ax_path.set_ylabel("p_y (m)")
    ax_path.set_title("Heading-dependent process noise: friction-anisotropic crawl")
    ax_path.axis("equal")
    ax_path.legend()

    ax_err.plot(t_hist, pos_err_iso, label="isotropic", color="tab:gray")
    ax_err.plot(t_hist, pos_err_fixed, label="fixed_anisotropic", color="tab:red")
    ax_err.plot(t_hist, pos_err_aware, label="heading_aware", color="tab:blue")
    ax_err.set_ylabel("Position error (m)")
    ax_err.set_yscale("log")
    ax_err.legend()

    ax_nees.plot(t_hist, nees_iso, label="isotropic", color="tab:gray")
    ax_nees.plot(t_hist, nees_fixed, label="fixed_anisotropic", color="tab:red")
    ax_nees.plot(t_hist, nees_aware, label="heading_aware", color="tab:blue")
    ax_nees.axhline(3.0, color="black", linestyle="-", linewidth=1, label="Expected NEES (3 DoF)")
    ax_nees.axhline(CHI2_3DOF_95, color="black", linestyle="--", linewidth=1, label="Chi-squared 95% bound")
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
