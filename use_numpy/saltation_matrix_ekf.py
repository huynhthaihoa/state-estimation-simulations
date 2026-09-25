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
    truth of the exact quantity `step_hybrid` needs -- composed with the
    ordinary flow Jacobians before/after the event, to ~7e-6 -- after a
    different, also-standard-looking formula, correct for a *different*
    question, was checked against the same ground truth and found wrong by a
    wide margin for *this* one).

A Monte Carlo consistency check (`run_monte_carlo_consistency`, new to this
repo -- there is no existing NEES/NIS/chi-squared helper anywhere else here)
repeats the naive and saltation-corrected EKFs over many independent noise
realizations of the same nominal trajectory and averages each one's NEES
(Normalized Estimation Error Squared) per step. The finding here is modest,
not dramatic: the saltation-corrected EKF's post-bounce NEES runs slightly
*higher* than the naive EKF's (~2.5 vs. ~2.4 at this script's defaults, a
~5% gap -- see docs/filtering/hybrid_saltation_ekf.md §6 for the full
numbers), the opposite direction from the naive "saltation fixes naive's
overconfidence" expectation, but nowhere near large enough to call either
filter's calibration meaningfully broken. `saltation_matrix` reduces (but,
unlike the rejected formula, does not zero out) the guard-normal (height)
direction's *reported* variance at every bounce -- see its own docstring's
`Dg @ Xi = -e * Dg` identity -- which is *most* accurate when the filter's
own estimated bounce time coincides with the true one, and tracking error
generally keeps that from being exact; the small residual gap above is that
effect, still present but far more muted than the near-singular-covariance
version an incorrect formula without the `f+` term would produce. Both EKFs'
*mean* trajectories still look nearly identical throughout regardless: this
entire effect is invisible in the point estimate and only shows up in
whether the reported uncertainty can be trusted.

A real contact sensor adds a second, independent source of timing error on
top of that: its own detection latency/jitter, beyond whatever the state
estimate already gets wrong about the geometric crossing time.
`step_hybrid`'s `detect_time_bias`/`detect_time_noise_std` parameters (both
default 0.0, reproducing the exact-detection behavior above) model that
directly, by applying the bounce reset at a `tau_detect` offset from the
true geometric crossing instead of at the crossing itself. Quantified (see
docs/filtering/hybrid_saltation_ekf.md §8 for the full sweep): the
naive-vs-saltation post-bounce NEES gap above (already present at exact
detection, ~1.05x) widens further as --detect-time-noise-std grows, reaching
~1.6x once jitter reaches half the measurement interval -- both filters
degrade (unsurprising, since neither one's P accounts for detection
uncertainty at all), and saltation degrades somewhat faster, a real if
modest instance of the general gap between "saltation matrices assume a
known transition time" and "contact/phase detection is itself uncertain".
Getting this measurement right required a real bug fix, not just a new
parameter: `crossing_time` originally accepted any future zero-crossing,
which was harmless as long as every reset landed exactly on the guard, but
once detect_time_* can leave a reset off-guard, the point mass can end up
slightly below `p_z = 0` and immediately re-cross it on the way back *up*
-- an ascending, non-physical "impact" that (before the fix) triggered a
cascade of spurious re-bounces and inflated NEES/position error by 4-5
orders of magnitude for reasons having nothing to do with the phenomenon
being modeled. `crossing_time` now only returns a *descending* crossing --
see its own docstring.
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
    (p_z = 0) *while descending* (impact from above, dp_z/dt < 0 at that
    instant): solves the exact quadratic p_z(t) = p_z0 + v_z0*t - 0.5*g*t^2 = 0
    for the smallest strictly-positive, descending root. No bisection/
    root-finding is needed since the free-fall flow is exactly quadratic in
    time.

    The descending-only filter matters once `x` can start off-guard (e.g.
    step_hybrid's detect_time_bias/detect_time_noise_std placing a reset
    slightly below p_z=0): a point mass that starts under the guard moving
    upward crosses p_z=0 again almost immediately on its way back out, but
    that crossing is ascending, not a real impact -- without this filter,
    step_hybrid would treat rising back out of that detection-induced
    "dip" as a fresh bounce, triggering a cascade of spurious re-bounces
    within a single step (verified this was happening: NEES/RMS position
    error exploded by orders of magnitude the instant any detect_time_*
    noise was introduced, traced to exactly this). Every crossing this
    script called before that feature existed was already the descending
    one (falling from p_z0>0, or starting exactly at p_z=0 with v_z0>0 where
    the only positive root left after excluding t=0 is the next *descending*
    return), so this filter changes no prior behavior.
    Arguments:
        x: state [p(3), v(3)] (6,)
        g: gravitational acceleration magnitude (m/s^2)
    Returns:
        tau: time to the next descending ground crossing (s), or None if
             there is none in the future
    """
    p_z0, v_z0 = x[2], x[5]
    a_coef, b_coef, c_coef = -0.5 * g, v_z0, p_z0
    disc = b_coef ** 2 - 4 * a_coef * c_coef
    if disc < 0.0:
        return None
    sqrt_disc = np.sqrt(disc)
    r1 = (-b_coef + sqrt_disc) / (2 * a_coef)
    r2 = (-b_coef - sqrt_disc) / (2 * a_coef)
    descending_roots = [r for r in (r1, r2) if r > 1e-12 and (v_z0 - g * r) < 0.0]
    return min(descending_roots) if descending_roots else None


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
    state, propagated forward to a *fixed* later reference time shared by
    every trajectory in the family (as `step_hybrid` below needs: it always
    compares states at fixed-size dt ticks, not at each trajectory's own
    natural crossing time) -- NOT simply reset_jacobian(e), because a
    perturbed trajectory reaches the guard p_z = 0 at a slightly different
    time, and during that extra sliver of time it is still governed by the
    free-fall flow, both before *and after* the reset.

    Derivation: for a family of trajectories x(t; p) governed by dx/dt =
    f(x) up to a guard g(x) = 0 crossed at a p-dependent time t*(p), with a
    reset x+ = R(x-) applied there and the same flow resumed afterward,
    fix a later reference time T (independent of p) and ask for
    d/dp[x(T;p)]. Implicit differentiation of g(x(t*(p);p))=0 gives the
    familiar crossing-time sensitivity dt*/dp = -(Dg.S(t*))/(Dg.f(x-)),
    where S(t) = dx(t;p)/dp; the state at T is x(T;p) = Phi(x+(p), T-t*(p))
    (the flow map, run for the *remaining* time T-t*(p) starting from the
    post-reset state). Differentiating that through the chain rule, and
    using the standard flow identity D_xPhi(x,s).f(x) = f(Phi(x,s)) (an
    autonomous flow's own linearization always carries its generating
    vector field forward to the vector field at the flowed-to point), the
    two time-dependent pieces combine into

        Xi = DR(x-) + [f+(x+) - DR(x-).f-(x-)] (outer) Dg / (Dg . f-(x-))

    (f- = f(x-), f+ = f(x+) = f(R(x-)) -- the SAME vector field, just
    evaluated pre- vs. post-reset). This composes with the ordinary flow
    Jacobians before and after the event -- Phi_after @ Xi @ Phi_before --
    to give the exact Jacobian of the whole fixed-time map, matching a
    from-scratch finite-difference ground truth of that exact quantity to
    ~7e-6 (limited by the finite-difference step itself, not this formula).

    A different, also-standard-looking formula, DR(x-) @ [I - f-(x-) (outer)
    Dg/(Dg.f-(x-))], answers a *different* question -- the sensitivity of
    the post-impact state at its *own* natural post-impact time (no shared
    reference time at all) with respect to the pre-impact state at its own
    natural pre-impact time. That quantity is correct for what it measures
    (verified separately, to ~2.9e-8, in this module's tests) and is
    exactly the building block composed-across-events for Poincare-map
    stability analysis (docs/filtering/hybrid_saltation_ekf.md's §9) -- but
    it is missing the f+ term entirely, and using it here, where every
    comparison is against a fixed dt tick rather than each trajectory's own
    crossing time, was checked against this function's own finite-difference
    ground truth and found wrong by a wide margin (max abs diff ~4.4, not
    floating-point noise) -- this module's docs/tests keep that failed
    check as a deliberate point: getting the *shape* of a saltation-matrix
    formula right is not the same as getting the right saltation matrix for
    the specific comparison at hand.

    A structural property worth knowing before using this in an EKF:
    Dg @ Xi = -e * Dg identically (checked: `Dg @ saltation_matrix(...)`
    always equals `-e` times the guard gradient, for any x_minus/e/g) --
    the guard-normal (p_z) output direction has a *reduced*, not zero,
    sensitivity to perturbations along the guard-normal input direction,
    scaled by the restitution coefficient. Unlike the rejected formula
    above, this one does not drive P's guard-normal row/column to exactly
    zero every bounce, so it needs no artificial regularizing floor to stay
    numerically well-behaved against a true trajectory sampled at fixed
    ticks -- see `step_hybrid` below and docs/filtering/hybrid_saltation_ekf.md
    §5 for what that changes about this script's own empirical findings.
    Arguments:
        x_minus: pre-impact state (6,), assumed to already satisfy p_z = 0
        e: restitution coefficient
        g: gravitational acceleration magnitude (m/s^2)
    Returns:
        Xi: (6,6) saltation matrix
    """
    f_minus = np.concatenate([x_minus[3:6], [0.0, 0.0, -g]])
    x_plus = reset_map(x_minus, e)
    f_plus = np.concatenate([x_plus[3:6], [0.0, 0.0, -g]])
    Dg = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    DR = reset_jacobian(e)
    denom = Dg @ f_minus
    return DR + np.outer(f_plus - DR @ f_minus, Dg) / denom


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
                 min_bounce_speed=1e-3, max_bounces=20,
                 detect_time_bias=0.0, detect_time_noise_std=0.0, detect_rng=None):
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
    identically regardless of use_saltation. saltation_matrix's `Dg @ Xi =
    -e * Dg` property (see its docstring) means the guard-normal (p_z)
    row/column of P is *reduced*, not driven to exactly zero, at each bounce
    -- unlike a formula without the `f+` correction term, which would zero
    it out and make P numerically fragile against a true trajectory sampled
    at fixed ticks rather than at its own exact crossing. This floor is
    consequently no longer strictly required for numerical sanity here (with
    `impact_noise_std=0`, NEES stays modest rather than exploding -- see
    docs/filtering/hybrid_saltation_ekf.md §7); it remains as a simplified
    stand-in for the explicit impact-timing/model-uncertainty term a real
    contact-aided EKF/InEKF implementation would carry, not a numerical
    necessity for this script's own formula.

    `detect_time_bias`/`detect_time_noise_std` model a second, independent
    source of bounce-timing error: a real contact sensor (IMU spike, force
    threshold -- see Cizek et al. 2018) has its own detection latency/jitter
    on top of whatever the filter's state estimate already gets wrong about
    the geometric crossing time. This models directly the gap between
    "saltation matrices assume a known transition time" and "contact/phase
    detection is itself uncertain": the reset (mean and covariance both) is
    applied at a *detected* crossing time
    `tau_detect = clip(tau + detect_time_bias [+ N(0, detect_time_noise_std)],
    0, remaining)` instead of the true geometric `tau`. Since `x` is only
    flowed forward to `tau_detect`, not `tau`, its height is generally
    nonzero there (early detection: still above ground; late detection: the
    free-fall model has it slightly "through" the ground) -- exactly the
    interpenetration/anticipation artifact a real delayed or jittery contact
    detector produces. Clipping into the current `dt` tick is a deliberate
    scope simplification (no multi-tick carryover for detections large
    enough to miss the tick entirely), consistent with `flow_resting`'s and
    the impact-noise floor's own documented simplifications above. Both
    default to 0.0, which reproduces the exact-detection behavior this
    function had before this parameter existed.

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
        detect_time_bias: systematic offset (s) applied to the detected
            crossing time relative to the true one (positive = late detection)
        detect_time_noise_std: std-dev (s) of Gaussian jitter added on top of
            detect_time_bias each time a crossing is detected
        detect_rng: numpy Generator used to draw the jitter; required if
            detect_time_noise_std > 0
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

        tau_detect = tau + detect_time_bias
        if detect_time_noise_std > 0.0:
            tau_detect += detect_rng.normal(0.0, detect_time_noise_std)
        tau_detect = float(np.clip(tau_detect, 0.0, remaining))

        x, Phi = flow(x, tau_detect, g)
        P = Phi @ P @ Phi.T + process_noise_covariance(tau_detect, accel_noise_std)
        remaining -= tau_detect

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


def run_ekf(x_init, P_init, z, dt, n_steps, e, g, accel_noise_std, impact_noise_std, R, use_saltation,
            detect_time_bias=0.0, detect_time_noise_std=0.0, detect_rng=None):
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
        detect_time_bias, detect_time_noise_std, detect_rng: contact-detection
            timing-uncertainty parameters, passed straight through to
            step_hybrid -- see its docstring
    Returns:
        x_list: list of estimated states (6,)
        P_list: list of estimated covariances (6,6)
    """
    x, P = x_init.copy(), P_init.copy()
    x_list, P_list = [x], [P]
    for k in range(n_steps):
        x, P = step_hybrid(x, P, dt, e, g, accel_noise_std, use_saltation, impact_noise_std,
                            detect_time_bias=detect_time_bias,
                            detect_time_noise_std=detect_time_noise_std, detect_rng=detect_rng)
        x, P = measurement_update(x, P, z[k + 1], R)
        x_list.append(x)
        P_list.append(P)
    return x_list, P_list


def run_ekf_naive(x_init, P_init, z, dt, n_steps, e, g, accel_noise_std, impact_noise_std, R,
                   detect_time_bias=0.0, detect_time_noise_std=0.0, detect_rng=None):
    """run_ekf with use_saltation=False -- see run_ekf's docstring."""
    return run_ekf(x_init, P_init, z, dt, n_steps, e, g, accel_noise_std, impact_noise_std, R,
                    use_saltation=False, detect_time_bias=detect_time_bias,
                    detect_time_noise_std=detect_time_noise_std, detect_rng=detect_rng)


def run_ekf_saltation(x_init, P_init, z, dt, n_steps, e, g, accel_noise_std, impact_noise_std, R,
                       detect_time_bias=0.0, detect_time_noise_std=0.0, detect_rng=None):
    """run_ekf with use_saltation=True -- see run_ekf's docstring."""
    return run_ekf(x_init, P_init, z, dt, n_steps, e, g, accel_noise_std, impact_noise_std, R,
                    use_saltation=True, detect_time_bias=detect_time_bias,
                    detect_time_noise_std=detect_time_noise_std, detect_rng=detect_rng)


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
                                 n_trials, rng, detect_time_bias=0.0, detect_time_noise_std=0.0):
    """Repeats run_ekf_naive/run_ekf_saltation over n_trials independent
    noise realizations of the same nominal true trajectory (fresh
    initial-condition error and fresh measurement noise each trial),
    returning the per-step average NEES for both.

    When detect_time_noise_std > 0, each filter variant draws its own
    bounce-detection jitter from an independent child RNG (derived once from
    `rng` before the trial loop, not `rng` itself) -- this repo's numpy
    (1.22) predates `Generator.spawn()`, so the child streams are seeded
    directly from a draw of the parent instead. Using two independent
    streams (rather than one shared `rng`) avoids naive's and saltation's
    detection draws being coupled through call-order-dependent consumption
    of a single stream, since the two variants' trajectories -- and
    therefore how many bounces/crossings each one's step_hybrid resolves --
    can differ slightly once detection noise is involved.
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
        detect_time_bias, detect_time_noise_std: contact-detection
            timing-uncertainty parameters, passed to both filter variants
            identically -- see step_hybrid's docstring
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

    detect_rng_naive = np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))
    detect_rng_salt = np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))

    sum_naive = np.zeros(n_pts)
    sum_salt = np.zeros(n_pts)
    for _ in range(n_trials):
        x_init = x_true_list[0] + rng.normal(0.0, 1.0, 6) * init_std
        z_trial = true_pos + rng.normal(0.0, pos_noise_std, (n_pts, 3))

        x_naive, P_naive = run_ekf_naive(x_init, P_init, z_trial, dt, n_steps, e, g,
                                          accel_noise_std, impact_noise_std, R,
                                          detect_time_bias=detect_time_bias,
                                          detect_time_noise_std=detect_time_noise_std,
                                          detect_rng=detect_rng_naive)
        x_salt, P_salt = run_ekf_saltation(x_init, P_init, z_trial, dt, n_steps, e, g,
                                            accel_noise_std, impact_noise_std, R,
                                            detect_time_bias=detect_time_bias,
                                            detect_time_noise_std=detect_time_noise_std,
                                            detect_rng=detect_rng_salt)

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
                         help="Std-dev of the isotropic floor (impact_noise_std**2 * I) added to P at "
                              "each bounce, both EKF variants. Not needed for numerical stability "
                              "(saltation_matrix scales the guard-normal row by -e rather than "
                              "zeroing it -- see step_hybrid); it stands in for the impact-timing/model "
                              "uncertainty a real contact-aided filter would carry. It shifts both "
                              "filters' NEES level together but does not flip which one is higher -- "
                              "see docs/filtering/hybrid_saltation_ekf.md §5 and §7")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.1,
                         help="Std-dev used to perturb the initial position guess (m)")
    parser.add_argument("--init-vel-noise-std", type=float, default=0.2,
                         help="Std-dev used to perturb the initial velocity guess (m/s)")

    parser.add_argument("--detect-time-bias", type=float, default=0.0,
                         help="Systematic contact-detection timing offset (s), positive = late "
                              "detection. Models a real contact sensor's own detection latency, "
                              "on top of whatever the filter's state estimate already gets wrong about the geometric "
                              "crossing time -- see step_hybrid's docstring. 0.0 reproduces "
                              "exact-detection behavior (the only behavior this script had before "
                              "this parameter existed)")
    parser.add_argument("--detect-time-noise-std", type=float, default=0.0,
                         help="Std-dev (s) of Gaussian contact-detection timing jitter, added on "
                              "top of --detect-time-bias each time a bounce is detected. See "
                              "step_hybrid's docstring and docs/filtering/hybrid_saltation_ekf.md")

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

    # Independent detection-jitter streams per filter variant -- see
    # run_monte_carlo_consistency's docstring for why naive/saltation don't share one.
    detect_rng_naive = np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))
    detect_rng_salt = np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))

    print("Running EKF with naive bounce-covariance handling (reset Jacobian only)...")
    (x_naive, P_naive), time_naive, mem_naive = measure_performance(
        run_ekf_naive, x_init, P_init, z, args.dt, n_steps, args.restitution, args.gravity,
        args.process_noise_std, args.impact_noise_std, R, n_steps=n_steps,
        detect_time_bias=args.detect_time_bias, detect_time_noise_std=args.detect_time_noise_std,
        detect_rng=detect_rng_naive)

    print("Running EKF with saltation-corrected bounce-covariance handling...")
    (x_salt, P_salt), time_salt, mem_salt = measure_performance(
        run_ekf_saltation, x_init, P_init, z, args.dt, n_steps, args.restitution, args.gravity,
        args.process_noise_std, args.impact_noise_std, R, n_steps=n_steps,
        detect_time_bias=args.detect_time_bias, detect_time_noise_std=args.detect_time_noise_std,
        detect_rng=detect_rng_salt)

    print(f"Running Monte Carlo consistency check ({args.n_mc_trials} trials)...")
    nees_naive, nees_salt = run_monte_carlo_consistency(
        x_true, args.dt, args.restitution, args.gravity, args.process_noise_std, args.impact_noise_std,
        args.pos_noise_std, args.init_pos_noise_std, args.init_vel_noise_std, args.n_mc_trials, rng,
        detect_time_bias=args.detect_time_bias, detect_time_noise_std=args.detect_time_noise_std)

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

    print("\nAverage time (per-step) and peak memory (whole-run) per approach, empirical:")
    for name, avg_time, avg_mem in [
        ("Dead-reckoning", time_dr, mem_dr),
        ("EKF (naive)", time_naive, mem_naive),
        ("EKF (saltation)", time_salt, mem_salt),
    ]:
        print(f"  {name:<18s} avg time={avg_time * 1e6:9.2f} us/step | peak mem={avg_mem / 1024.0:9.3f} KB")

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
