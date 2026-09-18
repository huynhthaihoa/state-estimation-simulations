'''
Tracks a 3D point mass in free-fall under gravity that bounces (inelastically,
restitution `e`) off the ground `p_z = 0` -- the textbook canonical hybrid
dynamical system, and the simplest analog of a foot-strike / ground-contact
impact (a discrete velocity reset at a discrete contact event) without any of
SE(3)'s rotational complexity. State `x = [p (3,), v (3,)]`, plain R^6 -- no
Lie group is involved anywhere in this script (the guard/reset act on flat
position/velocity coordinates), so unlike every other script in this repo
there is no `use_manif/` counterpart: nothing here would change if it were
rewritten against manifpy, since there is no SE(3)/SO(3) object to delegate
to in the first place.

Between bounces the dynamics `dx/dt = [v, (0,0,-g)]` is exactly linear, so its
mean and covariance propagate exactly (no linearization at all) -- every
scrap of disagreement between the two EKF variants below is therefore
attributable entirely to how each one handles the bounce itself, nothing
else.

Three ways to track the trajectory are implemented:

  - Dead-reckoning (baseline): propagates the exact deterministic dynamics
    from an uncertain initial guess, never looking at position measurements.
    Bounce timing is sensitive to the state, so an initial-condition error
    makes the predicted bounce *times* drift from the true ones, producing
    growing position/velocity error even though the dynamics model itself
    (gravity, restitution) is known exactly.

  - EKF, naive bounce handling: predicts through each bounce by resetting the
    mean with `reset_map` and the covariance with `reset_jacobian(e)` alone
    (`P+ = DR @ P- @ DR.T`) -- the natural first guess, and the common
    mistake: it is *not* the correct linearization of "post-impact state as a
    function of pre-impact state", because a perturbed trajectory reaches the
    guard at a slightly different time, and during that extra sliver of time
    it is still governed by the pre-impact flow.

  - EKF, saltation-corrected: identical in every other respect, but corrects
    the covariance at each bounce with the true `saltation_matrix` instead of
    the reset Jacobian alone. See `saltation_matrix`'s own docstring for the
    derivation (verified against a from-scratch finite-difference ground
    truth to ~2.9e-8, i.e. machine precision, after an initially-recalled,
    plausible-looking formula was checked and found wrong).

A Monte Carlo consistency check (`run_monte_carlo_consistency`, new to this
repo -- there is no existing NEES/NIS/chi-squared helper anywhere else here)
repeats the naive and saltation-corrected EKFs over many independent noise
realizations of the same nominal trajectory and averages each one's NEES
(Normalized Estimation Error Squared) per step. The finding here is more
interesting, and more useful, than "naive is overconfident, saltation fixes
it": `saltation_matrix` provably drives the guard-normal (height) direction's
*reported* variance to exactly zero at every bounce (it is derived to do
exactly that -- see its own docstring's `Dg @ Xi = 0` identity), which is
correct *only* if the filter's own estimated bounce time exactly coincides
with the true one. It generally does not, once there is any tracking error
at all -- and empirically (verified across multiple seeds, see
`docs/filtering/hybrid_saltation_ekf.md`), this makes the saltation-corrected
EKF's post-bounce NEES consistently *higher* than the naive EKF's at the same
regularization level, not lower: the mathematically-exact local correction is
also the more fragile one once event-detection itself carries uncertainty, a
concrete instance of exactly the gap between "saltation matrices assume a
known transition time" and "contact/phase detection is itself uncertain"
already flagged as Open Consideration #1 in `unified_phd_plan.md`. Both EKFs'
*mean* trajectories still look nearly identical throughout regardless: this
entire effect is invisible in the point estimate and only shows up in
whether the reported uncertainty can be trusted.
'''

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import measure_performance

# chi-squared, 6 degrees of freedom, 95th percentile (scipy.stats.chi2.ppf(0.95, 6));
# hardcoded rather than importing scipy, since this script is otherwise pure numpy
# and this is only needed as a reference line on the consistency plot.
CHI2_6DOF_95 = 12.5916


def flow(x, dt, g):
    """Exact closed-form propagation of the free-fall (no-contact) dynamics
    dx/dt = [v, (0,0,-g)] over an interval dt, plus its exact state-transition
    Jacobian. Since the dynamics are linear in x, this is exact for any dt
    (including negative dt, i.e. propagating backward), not a linearization.
    Arguments:
        x: state [p(3), v(3)] (6,)
        dt: propagation interval (s)
        g: gravitational acceleration magnitude (m/s^2)
    Returns:
        x_new: propagated state (6,)
        Phi: (6,6) exact Jacobian dx_new/dx
    """
    p, v = x[0:3], x[3:6]
    a = np.array([0.0, 0.0, -g])
    p_new = p + v * dt + 0.5 * a * dt * dt
    v_new = v + a * dt
    x_new = np.concatenate([p_new, v_new])
    Phi = np.eye(6)
    Phi[0:3, 3:6] = dt * np.eye(3)
    return x_new, Phi


def flow_resting(x, dt):
    """Fallback dynamics for a point mass already at rest on the ground
    (p_z = v_z = 0 held fixed): only horizontal motion continues, as a
    frictionless slide. Used inside step_hybrid once a bounce's impact speed
    drops below min_bounce_speed, to avoid the classic "Zeno" pathology of a
    lossy bounce (e < 1) producing infinitely many, ever-faster bounces in a
    finite time. Not a general contact-mode model -- a deliberately simple
    fallback for a rare edge case, not the object of this script's comparison.
    Arguments:
        x: state [p(3), v(3)] (6,), assumed to already have p_z = v_z = 0
        dt: propagation interval (s)
    Returns:
        x_new: propagated state (6,)
        Phi: (6,6) exact Jacobian dx_new/dx
    """
    p, v = x[0:3].copy(), x[3:6].copy()
    p[0:2] = p[0:2] + v[0:2] * dt
    x_new = np.concatenate([p, v])
    Phi = np.eye(6)
    Phi[0, 3] = dt
    Phi[1, 4] = dt
    return x_new, Phi


def crossing_time(x, g):
    """Closed-form time until the point mass's height reaches the ground
    (p_z = 0): solves the exact quadratic p_z(t) = p_z0 + v_z0*t - 0.5*g*t^2 = 0
    for the smallest strictly-positive root. No bisection/root-finding is
    needed since the free-fall flow is exactly quadratic in time.
    Arguments:
        x: state [p(3), v(3)] (6,)
        g: gravitational acceleration magnitude (m/s^2)
    Returns:
        tau: time to the next ground crossing (s), or None if there is none
             in the future
    """
    p_z0, v_z0 = x[2], x[5]
    a_coef, b_coef, c_coef = -0.5 * g, v_z0, p_z0
    disc = b_coef ** 2 - 4 * a_coef * c_coef
    if disc < 0.0:
        return None
    sqrt_disc = np.sqrt(disc)
    r1 = (-b_coef + sqrt_disc) / (2 * a_coef)
    r2 = (-b_coef - sqrt_disc) / (2 * a_coef)
    positive_roots = [r for r in (r1, r2) if r > 1e-12]
    return min(positive_roots) if positive_roots else None


def reset_map(x, e):
    """Instantaneous inelastic bounce: reflects and scales the vertical
    velocity by the restitution coefficient e, leaving position and
    horizontal velocity unchanged. Assumes x is already at the guard
    (p_z = 0).
    Arguments:
        x: pre-impact state (6,)
        e: restitution coefficient (0 = fully inelastic, 1 = perfectly elastic)
    Returns:
        x_plus: post-impact state (6,)
    """
    x_plus = x.copy()
    x_plus[5] = -e * x[5]
    return x_plus


def reset_jacobian(e):
    """(6,6) Jacobian of reset_map wrt x -- constant, since reset_map is
    linear. This is the covariance-propagation matrix a naive implementation
    reaches for (P+ = DR @ P- @ DR.T); saltation_matrix below is the correct
    one.
    Arguments:
        e: restitution coefficient
    Returns:
        DR: (6,6) Jacobian, diag(1,1,1,1,1,-e)
    """
    DR = np.eye(6)
    DR[5, 5] = -e
    return DR


def saltation_matrix(x_minus, e, g):
    """The saltation matrix: the correct linearization of the post-impact
    state (at its own natural post-impact time) with respect to the
    pre-impact state (at its own natural pre-impact time) -- NOT simply
    reset_jacobian(e), because a perturbed trajectory reaches the guard
    p_z = 0 at a slightly different time, and during that extra sliver of
    time it is still governed by the free-fall flow.

    Derivation: for a family of trajectories x(t; p) governed by dx/dt =
    f(x) up to a guard g(x) = 0 crossed at a p-dependent time t*(p), with a
    reset x+ = R(x-) applied there, implicit differentiation of
    g(x(t*(p); p)) = 0 gives dt*/dp = -(Dg . S(t*)) / (Dg . f(x-)), where
    S(t) = dx(t;p)/dp. Substituting back into d/dp[x(t*(p); p)] and then
    into d/dp[R(x(t*(p); p))] gives

        Xi = DR(x-) @ [ I - f(x-) (outer) Dg / (Dg . f(x-)) ]

    -- DR corrected by a rank-1 term that removes exactly the
    "along-the-flow" component of a perturbation (the piece that would
    otherwise double-count as a pure time-shift of when the guard is
    crossed, rather than a genuine change in the crossing state itself).

    Verified against a from-scratch finite-difference ground truth (perturb
    x- directly, re-land it on the guard via the pre-impact flow, apply
    reset_map, compare to nominal) to ~2.9e-8, i.e. machine precision -- see
    docs/filtering/hybrid_saltation_ekf.md for the full derivation and for a
    different, plausible-looking formula (DR + [f+(x+) - DR@f-(x-)] (outer)
    Dg / (Dg.f-(x-)), commonly seen written down from memory) that was
    checked against the same finite-difference ground truth and found wrong
    by a wide margin (max abs diff ~1.28, not floating-point noise) --
    keeping that failed check is a deliberate teaching point, not an
    afterthought.

    A structural property worth knowing before using this in an EKF: since
    every trajectory in the family satisfies g(x) = 0 exactly at its own
    crossing, Dg @ Xi = 0 identically (checked: `Dg @ saltation_matrix(...)`
    is exactly the zero vector for any x_minus/e/g) -- the guard-normal
    output direction (p_z here) has *exactly* zero sensitivity to any input
    perturbation. That is mathematically correct, but it drives the p_z
    row/column of `Xi @ P @ Xi.T` to exactly zero every bounce, which is only
    trustworthy if the filter's own estimated crossing time exactly matches
    the true one -- it generally won't, once there is any state-estimation
    error at all. `step_hybrid` below adds a small regularizing "impact
    noise" floor after every bounce (both branches, so the comparison stays
    fair) specifically to keep this exact, correct projection numerically
    usable once compared against a true trajectory sampled at fixed ticks
    rather than at its own exact crossing times.
    Arguments:
        x_minus: pre-impact state (6,), assumed to already satisfy p_z = 0
        e: restitution coefficient
        g: gravitational acceleration magnitude (m/s^2)
    Returns:
        Xi: (6,6) saltation matrix
    """
    f_minus = np.concatenate([x_minus[3:6], [0.0, 0.0, -g]])
    Dg = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    denom = Dg @ f_minus
    B = np.eye(6) - np.outer(f_minus, Dg) / denom
    return reset_jacobian(e) @ B


def process_noise_covariance(h, accel_noise_std):
    """Simple diagonal process-noise covariance for one interval of length h,
    representing an unmodeled small constant-acceleration disturbance:
    velocity variance scales as (accel_noise_std*h)^2, position variance as
    the resulting position uncertainty (0.5*accel_noise_std*h^2)^2, no
    cross-terms. A deliberately simple heuristic (not the textbook continuous
    white-noise-acceleration covariance), since this filter's process-noise
    design is a tuning knob for the demo, not the object of the
    saltation-matrix comparison itself.
    Arguments:
        h: interval length (s)
        accel_noise_std: std-dev of the assumed acceleration disturbance (m/s^2)
    Returns:
        Q: (6,6) process-noise covariance for this interval
    """
    Q = np.zeros((6, 6))
    Q[0:3, 0:3] = np.eye(3) * (0.5 * accel_noise_std * h * h) ** 2
    Q[3:6, 3:6] = np.eye(3) * (accel_noise_std * h) ** 2
    return Q


def step_hybrid(x, P, dt, e, g, accel_noise_std, use_saltation, impact_noise_std=0.0,
                 min_bounce_speed=1e-3, max_bounces=20):
    """Predicts (x, P) forward by one fixed interval dt through the hybrid
    free-fall/bounce dynamics, splitting the step at every ground-contact
    event it contains: propagate the continuous flow exactly up to each
    crossing (adding process noise for the sub-interval actually integrated),
    apply reset_map to the mean and either reset_jacobian(e) alone (if
    use_saltation is False) or saltation_matrix (if True) to the covariance,
    then continue with the remainder of the interval. use_saltation is the
    single toggle distinguishing the naive EKF from the saltation-corrected
    one below -- everything else in the predict step is shared.

    A small isotropic "impact noise" floor (impact_noise_std**2 * I) is added
    to P immediately after every bounce's covariance update, applied
    identically regardless of use_saltation. This exists specifically to
    counter saltation_matrix's own Dg@Xi=0 property (see its docstring): the
    exact saltation projection drives the guard-normal (p_z) row/column of P
    to exactly zero, which is only trustworthy if the filter's own estimated
    crossing time exactly matches the true one -- with any state-estimation
    error this won't hold, and comparing a near-singular P against a true
    trajectory sampled at a fixed tick (not at its own exact crossing) blows
    up the NEES/Mahalanobis distance purely numerically, not because the
    saltation-corrected filter is somehow worse. Real contact-aided
    EKF/InEKF implementations handle this with an explicit impact-timing or
    model-uncertainty term; this isotropic floor is a simplified stand-in for
    that, not a claim of matching that rigor.

    Once the pre-impact vertical speed at a crossing falls below
    min_bounce_speed, the point mass is treated as having come to rest
    (v_z clamped to 0, remaining time integrated via flow_resting) rather
    than continuing to resolve ever-smaller, ever-faster bounces.
    Arguments:
        x: state [p(3), v(3)] (6,)
        P: (6,6) covariance
        dt: full step interval (s)
        e: restitution coefficient
        g: gravitational acceleration magnitude (m/s^2)
        accel_noise_std: std-dev of the assumed acceleration disturbance (m/s^2)
        use_saltation: if True, correct the covariance at each bounce with
            the saltation matrix; if False, use reset_jacobian(e) alone
        impact_noise_std: std-dev of the isotropic regularizing floor added
            to P at each bounce (both branches)
        min_bounce_speed: impact speed (m/s) below which the point mass is
            treated as at rest
        max_bounces: safety cap on bounces processed within one step
    Returns:
        x: state after the full interval dt
        P: covariance after the full interval dt
    """
    remaining = dt
    at_rest = False
    for _ in range(max_bounces):
        if remaining <= 0.0:
            break
        if at_rest:
            x, Phi = flow_resting(x, remaining)
            P = Phi @ P @ Phi.T + process_noise_covariance(remaining, accel_noise_std)
            remaining = 0.0
            break

        tau = crossing_time(x, g)
        if tau is None or tau >= remaining:
            x, Phi = flow(x, remaining, g)
            P = Phi @ P @ Phi.T + process_noise_covariance(remaining, accel_noise_std)
            remaining = 0.0
            break

        x, Phi = flow(x, tau, g)
        P = Phi @ P @ Phi.T + process_noise_covariance(tau, accel_noise_std)
        remaining -= tau

        if abs(x[5]) < min_bounce_speed:
            x[5] = 0.0
            at_rest = True
            continue

        Xi = saltation_matrix(x, e, g) if use_saltation else reset_jacobian(e)
        x = reset_map(x, e)
        P = Xi @ P @ Xi.T + impact_noise_std ** 2 * np.eye(6)

    return x, P


def generate_ground_truth_and_data(duration, dt, e, g, drop_height, init_horizontal_vel,
                                    pos_noise_std, rng):
    """Builds the true multi-bounce trajectory (exact, deterministic -- no
    process noise in the true dynamics themselves) and noisy position
    measurements a tracker would actually receive.
    Arguments:
        duration: total simulation time (s)
        dt: measurement/filter step interval (s)
        e: restitution coefficient
        g: gravitational acceleration magnitude (m/s^2)
        drop_height: initial height p_z (m)
        init_horizontal_vel: initial horizontal velocity [vx, vy] (m/s)
        pos_noise_std: std-dev of Gaussian noise added to position measurements (m)
        rng: numpy random number generator
    Returns:
        x_true: list of true states (6,) at each tick (including t=0)
        z: (n_steps+1, 3) array of noisy position measurements
    """
    n_steps = int(duration / dt)
    x0 = np.array([0.0, 0.0, drop_height, init_horizontal_vel[0], init_horizontal_vel[1], 0.0])

    x_true = [x0]
    for _ in range(n_steps):
        x_next, _ = step_hybrid(x_true[-1], np.zeros((6, 6)), dt, e, g, 0.0, use_saltation=True)
        x_true.append(x_next)

    z = np.array([x[0:3] for x in x_true]) + rng.normal(0.0, pos_noise_std, (n_steps + 1, 3))
    return x_true, z


def run_dead_reckoning(x_init, dt, n_steps, e, g):
    """Prior-only baseline: propagate the exact deterministic dynamics from
    the (uncertain) initial guess, never looking at position measurements.
    Arguments:
        x_init: initial state guess (6,)
        dt: step interval (s)
        n_steps: number of steps
        e: restitution coefficient
        g: gravitational acceleration magnitude (m/s^2)
    Returns:
        x_list: list of predicted states (6,)
    """
    x_list = [x_init]
    for _ in range(n_steps):
        x_next, _ = step_hybrid(x_list[-1], np.zeros((6, 6)), dt, e, g, 0.0, use_saltation=True)
        x_list.append(x_next)
    return x_list


def measurement_update(x, P, z, R):
    """Linear-Gaussian EKF update against a direct noisy position observation
    z = p + noise (H = [I_3 | 0_3]) -- deliberately the simplest possible
    measurement model, to keep focus on step_hybrid's bounce-covariance
    mechanics rather than the measurement side.
    Arguments:
        x: predicted state (6,)
        P: predicted covariance (6,6)
        z: position measurement (3,)
        R: measurement noise covariance (3,3)
    Returns:
        x_new: updated state (6,)
        P_new: updated covariance (6,6)
    """
    H = np.zeros((3, 6))
    H[:, 0:3] = np.eye(3)
    r = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x_new = x + K @ r
    P_new = (np.eye(6) - K @ H) @ P
    return x_new, P_new


def run_ekf(x_init, P_init, z, dt, n_steps, e, g, accel_noise_std, impact_noise_std, R, use_saltation):
    """Recursive EKF alternating step_hybrid's hybrid predict with
    measurement_update. Shared by run_ekf_naive/run_ekf_saltation below --
    they differ only in the use_saltation flag, so any difference in their
    output is attributable entirely to the bounce-covariance handling,
    nothing else.
    Arguments:
        x_init: initial state estimate (6,)
        P_init: initial covariance (6,6)
        z: (n_steps+1, 3) array of noisy position measurements
        dt: step interval (s)
        n_steps: number of steps
        e: restitution coefficient
        g: gravitational acceleration magnitude (m/s^2)
        accel_noise_std: std-dev of the assumed acceleration disturbance (m/s^2)
        impact_noise_std: std-dev of the regularizing floor added to P at
            each bounce -- see step_hybrid's docstring
        R: measurement noise covariance (3,3)
        use_saltation: if True, correct bounce covariance with the saltation
            matrix; if False, use reset_jacobian(e) alone
    Returns:
        x_list: list of estimated states (6,)
        P_list: list of estimated covariances (6,6)
    """
    x, P = x_init.copy(), P_init.copy()
    x_list, P_list = [x], [P]
    for k in range(n_steps):
        x, P = step_hybrid(x, P, dt, e, g, accel_noise_std, use_saltation, impact_noise_std)
        x, P = measurement_update(x, P, z[k + 1], R)
        x_list.append(x)
        P_list.append(P)
    return x_list, P_list


def run_ekf_naive(x_init, P_init, z, dt, n_steps, e, g, accel_noise_std, impact_noise_std, R):
    """run_ekf with use_saltation=False -- see run_ekf's docstring."""
    return run_ekf(x_init, P_init, z, dt, n_steps, e, g, accel_noise_std, impact_noise_std, R,
                    use_saltation=False)


def run_ekf_saltation(x_init, P_init, z, dt, n_steps, e, g, accel_noise_std, impact_noise_std, R):
    """run_ekf with use_saltation=True -- see run_ekf's docstring."""
    return run_ekf(x_init, P_init, z, dt, n_steps, e, g, accel_noise_std, impact_noise_std, R,
                    use_saltation=True)


def state_errors(x_true_list, x_est_list):
    """Position and velocity error norms, per step.
    Arguments:
        x_true_list: list of true states (6,)
        x_est_list: list of estimated states (6,)
    Returns:
        pos_err: (N,) array of position error norms (m)
        vel_err: (N,) array of velocity error norms (m/s)
    """
    n = len(x_true_list)
    pos_err, vel_err = np.zeros(n), np.zeros(n)
    for k in range(n):
        pos_err[k] = np.linalg.norm(x_true_list[k][0:3] - x_est_list[k][0:3])
        vel_err[k] = np.linalg.norm(x_true_list[k][3:6] - x_est_list[k][3:6])
    return pos_err, vel_err


def nees(x_true, x_est, P):
    """Normalized Estimation Error Squared: (x_true-x_est)^T @ P^-1 @ (x_true-x_est).
    A consistent filter's NEES should average to the state dimension (6 here)
    across many independent trials; a value that's persistently much larger
    means the filter's reported covariance P is too small (overconfident)
    for the errors it is actually making.
    Arguments:
        x_true: true state (6,)
        x_est: estimated state (6,)
        P: estimated covariance (6,6)
    Returns:
        nees: scalar NEES value
    """
    err = x_true - x_est
    return float(err @ np.linalg.inv(P) @ err)


def detect_bounce_ticks(x_true_list, height_threshold=0.2):
    """Approximate indices of the tick closest to each bounce in a true
    trajectory, found as a local minimum in height below height_threshold.
    Used only for reporting (windowing the NEES trace around each bounce),
    not by any of the filters themselves.
    Arguments:
        x_true_list: list of true states (6,)
        height_threshold: only consider local minima below this height (m)
    Returns:
        ticks: list of tick indices
    """
    ticks = []
    for k in range(1, len(x_true_list) - 1):
        p_z = x_true_list[k][2]
        if p_z < height_threshold and p_z <= x_true_list[k - 1][2] and p_z <= x_true_list[k + 1][2]:
            ticks.append(k)
    return ticks


def run_monte_carlo_consistency(x_true_list, dt, e, g, accel_noise_std, impact_noise_std,
                                 pos_noise_std, init_pos_noise_std, init_vel_noise_std,
                                 n_trials, rng):
    """Repeats run_ekf_naive/run_ekf_saltation over n_trials independent
    noise realizations of the same nominal true trajectory (fresh
    initial-condition error and fresh measurement noise each trial),
    returning the per-step average NEES for both.
    Arguments:
        x_true_list: list of true states (6,) at each tick
        dt: step interval (s)
        e: restitution coefficient
        g: gravitational acceleration magnitude (m/s^2)
        accel_noise_std: std-dev of the assumed acceleration disturbance (m/s^2)
        impact_noise_std: std-dev of the regularizing floor added to P at
            each bounce -- see step_hybrid's docstring
        pos_noise_std: std-dev of position measurement noise (m)
        init_pos_noise_std: std-dev used to perturb the initial position guess
        init_vel_noise_std: std-dev used to perturb the initial velocity guess
        n_trials: number of Monte Carlo trials
        rng: numpy random number generator
    Returns:
        nees_naive: (N,) average NEES per step, naive covariance handling
        nees_saltation: (N,) average NEES per step, saltation-corrected
    """
    n_steps = len(x_true_list) - 1
    n_pts = n_steps + 1
    P_init = np.diag([init_pos_noise_std ** 2] * 3 + [init_vel_noise_std ** 2] * 3)
    R = pos_noise_std ** 2 * np.eye(3)
    init_std = np.sqrt(np.diag(P_init))
    true_pos = np.array([xt[0:3] for xt in x_true_list])

    sum_naive = np.zeros(n_pts)
    sum_salt = np.zeros(n_pts)
    for _ in range(n_trials):
        x_init = x_true_list[0] + rng.normal(0.0, 1.0, 6) * init_std
        z_trial = true_pos + rng.normal(0.0, pos_noise_std, (n_pts, 3))

        x_naive, P_naive = run_ekf_naive(x_init, P_init, z_trial, dt, n_steps, e, g,
                                          accel_noise_std, impact_noise_std, R)
        x_salt, P_salt = run_ekf_saltation(x_init, P_init, z_trial, dt, n_steps, e, g,
                                            accel_noise_std, impact_noise_std, R)

        for k in range(n_pts):
            sum_naive[k] += nees(x_true_list[k], x_naive[k], P_naive[k])
            sum_salt[k] += nees(x_true_list[k], x_salt[k], P_salt[k])

    return sum_naive / n_trials, sum_salt / n_trials


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--duration", type=float, default=5.0,
                         help="Simulation length in seconds. Keep this comfortably below the "
                              "trajectory's own Zeno settling time (for the defaults below, an "
                              "e=0.85 restitution bounce dropped from 5m settles around ~12.4s, "
                              "but bounce intervals shrink quickly enough that the 5th+ bounce "
                              "already shows Zeno-adjacent instability well before that -- both "
                              "filters become unreliable there for reasons unrelated to the "
                              "saltation matrix itself, since mode-matching a fixed-dt-sampled "
                              "estimate against a true trajectory bouncing many times within one "
                              "dt breaks the whole comparison's premise); the default gives 3 "
                              "well-separated, energetic bounces and stops before the 4th")
    parser.add_argument("--dt", type=float, default=0.02, help="Filter/measurement step interval in seconds")
    parser.add_argument("--restitution", type=float, default=0.85, help="Bounce restitution coefficient (0-1)")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravitational acceleration magnitude (m/s^2)")
    parser.add_argument("--drop-height", type=float, default=5.0, help="Initial height (m)")
    parser.add_argument("--init-horizontal-vel", type=float, nargs=2, default=[1.0, 0.5],
                         metavar=("VX", "VY"), help="Initial horizontal velocity (m/s)")

    parser.add_argument("--pos-noise-std", type=float, default=0.03, help="Position measurement noise std-dev (m)")
    parser.add_argument("--process-noise-std", type=float, default=0.3,
                         help="Assumed acceleration-disturbance noise std-dev (m/s^2)")
    parser.add_argument("--impact-noise-std", type=float, default=0.005,
                         help="Std-dev of the regularizing floor added to P at each bounce, both EKF "
                              "variants (m-scale; needed because the exact saltation projection drives "
                              "the guard-normal direction's variance to exactly zero -- see step_hybrid). "
                              "This value is not just a numerical-stability knob: which filter's "
                              "post-bounce NEES comes out higher is sensitive to it -- see the module "
                              "docstring and docs/filtering/hybrid_saltation_ekf.md")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.1,
                         help="Std-dev used to perturb the initial position guess (m)")
    parser.add_argument("--init-vel-noise-std", type=float, default=0.2,
                         help="Std-dev used to perturb the initial velocity guess (m/s)")

    parser.add_argument("--n-mc-trials", type=int, default=500, help="Number of Monte Carlo consistency trials")

    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    x_true, z = generate_ground_truth_and_data(
        args.duration, args.dt, args.restitution, args.gravity, args.drop_height,
        args.init_horizontal_vel, args.pos_noise_std, rng)
    n_steps = len(x_true) - 1

    init_std = np.array([args.init_pos_noise_std] * 3 + [args.init_vel_noise_std] * 3)
    x_init = x_true[0] + rng.normal(0.0, 1.0, 6) * init_std
    P_init = np.diag(init_std ** 2)
    R = args.pos_noise_std ** 2 * np.eye(3)

    print("Running dead-reckoning baseline (exact dynamics, no measurement correction)...")
    x_dr, time_dr, mem_dr = measure_performance(
        run_dead_reckoning, x_init, args.dt, n_steps, args.restitution, args.gravity, n_steps=n_steps)

    print("Running EKF with naive bounce-covariance handling (reset Jacobian only)...")
    (x_naive, P_naive), time_naive, mem_naive = measure_performance(
        run_ekf_naive, x_init, P_init, z, args.dt, n_steps, args.restitution, args.gravity,
        args.process_noise_std, args.impact_noise_std, R, n_steps=n_steps)

    print("Running EKF with saltation-corrected bounce-covariance handling...")
    (x_salt, P_salt), time_salt, mem_salt = measure_performance(
        run_ekf_saltation, x_init, P_init, z, args.dt, n_steps, args.restitution, args.gravity,
        args.process_noise_std, args.impact_noise_std, R, n_steps=n_steps)

    print(f"Running Monte Carlo consistency check ({args.n_mc_trials} trials)...")
    nees_naive, nees_salt = run_monte_carlo_consistency(
        x_true, args.dt, args.restitution, args.gravity, args.process_noise_std, args.impact_noise_std,
        args.pos_noise_std, args.init_pos_noise_std, args.init_vel_noise_std, args.n_mc_trials, rng)

    pos_err_dr, vel_err_dr = state_errors(x_true, x_dr)
    pos_err_naive, vel_err_naive = state_errors(x_true, x_naive)
    pos_err_salt, vel_err_salt = state_errors(x_true, x_salt)

    print("\nFinal / RMS errors:")
    for name, pos_err, vel_err in [
        ("Dead-reckoning", pos_err_dr, vel_err_dr),
        ("EKF (naive)", pos_err_naive, vel_err_naive),
        ("EKF (saltation)", pos_err_salt, vel_err_salt),
    ]:
        print(f"  {name:<18s} final pos={pos_err[-1]:7.4f} m, vel={vel_err[-1]:7.4f} m/s | "
              f"RMS pos={np.sqrt(np.mean(pos_err**2)):7.4f} m, vel={np.sqrt(np.mean(vel_err**2)):7.4f} m/s")

    print("\nMonte Carlo NEES (consistent 6-DoF filter should average ~6.0 everywhere):")
    print(f"  EKF (naive)      mean NEES={np.mean(nees_naive):8.2f} | max NEES={np.max(nees_naive):8.2f}")
    print(f"  EKF (saltation)  mean NEES={np.mean(nees_salt):8.2f} | max NEES={np.max(nees_salt):8.2f}")

    # Both filters spike momentarily AT the exact tick nearest each bounce (an artifact of
    # comparing a fixed-dt-sampled estimate against a true trajectory that is itself close to
    # its own bounce at that same tick, not a difference between the two filters) -- the actual
    # naive-vs-saltation difference this script is about shows up in how fast NEES *recovers*
    # in the few ticks immediately after that shared spike.
    bounce_ticks = detect_bounce_ticks(x_true)
    recovery_window = 5
    post_bounce_naive, post_bounce_salt = [], []
    for tick in bounce_ticks:
        lo, hi = tick + 1, min(tick + 1 + recovery_window, len(nees_naive))
        post_bounce_naive.extend(nees_naive[lo:hi])
        post_bounce_salt.extend(nees_salt[lo:hi])
    if post_bounce_naive:
        print(f"\nMean NEES over the {recovery_window} ticks immediately after each of the "
              f"{len(bounce_ticks)} detected bounces (excludes the shared spike at the bounce "
              f"tick itself -- this is the actual naive-vs-saltation comparison):")
        print(f"  EKF (naive)      mean NEES={np.mean(post_bounce_naive):8.2f}")
        print(f"  EKF (saltation)  mean NEES={np.mean(post_bounce_salt):8.2f}")

    print("\nAverage time complexity + space complexity per approach (per-step, empirical):")
    for name, avg_time, avg_mem in [
        ("Dead-reckoning", time_dr, mem_dr),
        ("EKF (naive)", time_naive, mem_naive),
        ("EKF (saltation)", time_salt, mem_salt),
    ]:
        print(f"  {name:<18s} avg time={avg_time * 1e6:9.2f} us/step | avg peak mem={avg_mem / 1024.0:9.3f} KB/step")

    t_hist = np.arange(len(x_true)) * args.dt
    fig, (ax_height, ax_err, ax_nees) = plt.subplots(3, 1, figsize=(9, 11))

    ax_height.plot(t_hist, [x[2] for x in x_true], label="Ground truth", color="green", linewidth=2, linestyle="--")
    ax_height.plot(t_hist, [x[2] for x in x_dr], label="Dead-reckoning (prior only)", color="tab:gray", linestyle=":")
    ax_height.plot(t_hist, [x[2] for x in x_naive], label="EKF (naive)", color="tab:red", linestyle="-")
    ax_height.plot(t_hist, [x[2] for x in x_salt], label="EKF (saltation)", color="tab:blue", linestyle="-")
    ax_height.set_ylabel("Height p_z (m)")
    ax_height.set_title("Saltation-matrix EKF: bouncing point mass")
    ax_height.legend()

    ax_err.plot(t_hist, pos_err_dr, label="Dead-reckoning (prior only)", color="tab:gray")
    ax_err.plot(t_hist, pos_err_naive, label="EKF (naive)", color="tab:red")
    ax_err.plot(t_hist, pos_err_salt, label="EKF (saltation)", color="tab:blue")
    ax_err.set_ylabel("Position error (m)")
    ax_err.set_yscale("log")
    ax_err.legend()

    ax_nees.plot(t_hist, nees_naive, label="EKF (naive)", color="tab:red")
    ax_nees.plot(t_hist, nees_salt, label="EKF (saltation)", color="tab:blue")
    ax_nees.axhline(6.0, color="black", linestyle="-", linewidth=1, label="Expected NEES (6 DoF)")
    ax_nees.axhline(CHI2_6DOF_95, color="black", linestyle="--", linewidth=1, label="Chi-squared 95% bound")
    ax_nees.set_ylabel("Monte Carlo avg. NEES")
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
