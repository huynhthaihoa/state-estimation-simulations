'''
Tracks a tumbling rigid body carrying an IMU (gyroscope + accelerometer) that
falls under gravity and bounces off the ground `p_z = 0` -- the same hybrid
system as saltation_matrix_ekf.py, lifted from a point mass in plain R^6 to a
rigid body whose state is an element of the Lie group SE_2(3):

    X = [[R, v, p],
         [0, 1, 0],
         [0, 0, 1]]        (R: body-to-world rotation, v: velocity, p: position)

Between bounces the IMU drives the motion: the body spins at a prescribed
constant body rate, and in free flight the accelerometer reads zero specific
force, so translation is exactly ballistic and the time to the next ground
crossing is closed-form. At a bounce, the velocity is reflected off the ground
with restitution `e` -- `v+ = M v`, `M = I - (1 + e) n n^T`, `n = e_z` --
while R and p are unchanged.

The filters are right-invariant EKFs (RIEKF): the error is `eta = X_hat @
inv(X)`, parameterized by its 9-vector logarithm `xi = (xi_R, xi_v, xi_p)`.
Two properties make the RIEKF attractive, and both hold exactly here between
bounces:
  - Propagation: IMU kinematics are "group-affine", so the error evolves
    independently of the trajectory, with a *constant* matrix
    `A = [[0,0,0],[g^,0,0],[0,I,0]]` and an exact discrete transition
    `Phi(h) = expm(A h)` (`propagation_matrix`).
  - Measurement: body-frame observations of known landmarks,
    `y = R^T (l - p) + noise`, are right-invariant observations, so the
    measurement Jacobian `H = [-l^, 0, I]` is constant too
    (`measurement_jacobian_ri`).

The bounce breaks this. The reset `(R, v, p) -> (R, M v, p)` is not a group
automorphism of SE_2(3) (`M R != R M` in general -- see
`reset_is_automorphism_gap`), so its effect on the invariant error depends on
the state. Three ways to push the covariance through a bounce are compared,
identical in every other respect:

  - "invariant-naive": treat the reset as if it acted on the error the way it
    acts on the state, `Xi = blockdiag(I, M, I)` -- the tempting shortcut;
    wrong for two reasons (state dependence and impact timing).
  - "reset-Jacobian": the exact Jacobian of the reset map in invariant
    coordinates, `T(X+)^-1 DR T(X-)` -- fixes the state dependence, but still
    ignores that a perturbed trajectory reaches the ground at a slightly
    different time (the naive EKF of saltation_matrix_ekf.py, on the group).
  - "saltation": the saltation matrix, computed in world-perturbation
    coordinates (where it has the textbook form) and mapped into invariant
    coordinates, `Xi_RI = T(X+)^-1 Xi_world T(X-)` (`saltation_matrix_ri`).
    Verified against a finite-difference ground truth of the exact fixed-time
    hybrid map on SE_2(3) (tests/use_numpy/test_invariant_saltation_ekf.py),
    where the other two are wrong by O(1).

A further RIEKF, "saltation_timing", adds what the saltation matrix leaves
out: the contact detector's own timing error. At each detected bounce it adds
`s^2 w w^T` to the covariance, where `w = T(X+)^-1 (f+ - DR f-)` is how much
the post-impact state moves per second of timing error
(`impact_timing_direction_ri`) and `s` is the detector's timing-error RMS.
With exact detection (the default) it is identical to "saltation".

One more filter -- an ordinary error-state EKF in world-perturbation
coordinates (`R = Exp(dtheta) R_hat, v = v_hat + dv, p = p_hat + dp`), with
the saltation correction -- is the non-invariant baseline: its propagation and
measurement Jacobians depend on the estimate.

A Monte Carlo consistency check (`run_monte_carlo_consistency`) averages each
filter's NEES (9 DoF) per step over many independent noise realizations of
the same nominal trajectory, scoring only mode-matched ticks (filter and truth
have resolved the same number of bounces; see its docstring). See docs/filtering/invariant_saltation_ekf.md
for the derivation and the numbers.

Scope simplifications, all deliberate: no IMU biases (they break the exact
group-affine property and would blur the comparison); the impact does not
change the body rate (the reset acts on velocity only); bounce timing is
predicted from each filter's own estimate, and contact-detection error
(`--detect-time-bias`/`--detect-time-noise-std`, off by default) is the
point-mass model of saltation_matrix_ekf.py (`detected_crossing_time`); and
the run stops well
before the bounces become small (`step_hybrid_ri`/`step_hybrid_world` raise if an impact
is slower than `min_bounce_speed`, instead of modeling the settling phase).
'''

import argparse
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from state_estimation_simulations.utils import measure_performance, averaged_nees_bounds, nees
from state_estimation_simulations.use_numpy.lie_utils import skew, so3_exp, so3_log, se23_make, se23_parts, se23_inv, se23_exp, se23_log, se23_adjoint

# --- Continuous dynamics, guard and reset -------------------------------------

def restitution_matrix(e, normal=np.array([0.0, 0.0, 1.0])):
    """Velocity reset at a bounce: M = I - (1 + e) n n^T reflects the normal
    component and scales it by e, leaving the tangential components unchanged.
    Arguments:
        e: restitution coefficient (0-1)
        normal: (3,) unit ground normal
    Returns:
        M: (3,3) reset matrix
    """
    return np.eye(3) - (1.0 + e) * np.outer(normal, normal)


def reset_map(X, e):
    """Bounce reset on SE_2(3): (R, v, p) -> (R, M v, p). Assumes X is at the
    guard (p_z = 0).
    Arguments:
        X: (5,5) pre-impact state
        e: restitution coefficient
    Returns:
        X_plus: (5,5) post-impact state
    """
    R, v, p = se23_parts(X)
    return se23_make(R, restitution_matrix(e) @ v, p)


def reset_is_automorphism_gap(X1, X2, e):
    """How far the reset is from being a group automorphism:
    max |reset(X1 X2) - reset(X1) reset(X2)|. Zero for every pair only if the
    reset commutes with the group product; for the bounce it does not
    (M R != R M), which is why its invariant-error Jacobian is state-dependent.
    Arguments:
        X1, X2: (5,5) SE_2(3) elements
        e: restitution coefficient
    Returns:
        gap: scalar
    """
    return float(np.max(np.abs(reset_map(X1 @ X2, e) - reset_map(X1, e) @ reset_map(X2, e))))


def propagate_mean(X, gyro, accel, h, g_vec):
    """Mean propagation over h with constant IMU readings, in the group-affine
    form X+ = Gamma(h) f_h(X) Upsilon(h):
        R+ = R Exp(gyro h)
        v+ = v + g h + R (accel h)
        p+ = p + v h + 0.5 g h^2 + R (0.5 accel h^2)
    Exact for the true motion in this script (constant body rate, zero
    specific force in flight); first order in h for a noisy accel reading.
    Arguments:
        X: (5,5) state
        gyro: (3,) body angular rate (rad/s)
        accel: (3,) body specific force (m/s^2)
        h: interval (s)
        g_vec: (3,) gravity vector in world frame
    Returns:
        X_new: (5,5) propagated state
    """
    R, v, p = se23_parts(X)
    R_new = R @ so3_exp(gyro * h)
    v_new = v + g_vec * h + R @ (accel * h)
    p_new = p + v * h + 0.5 * g_vec * h * h + R @ (0.5 * accel * h * h)
    return se23_make(R_new, v_new, p_new)


def crossing_time(X, accel, g_vec):
    """Time until the next *descending* ground crossing (p_z = 0), assuming
    the world-frame acceleration R accel + g stays constant over the interval
    (exact for the truth, where accel = 0).
    Arguments:
        X: (5,5) state
        accel: (3,) body specific force (m/s^2)
        g_vec: (3,) gravity vector in world frame
    Returns:
        tau: time to the crossing (s), or None if there is none ahead
    """
    R, v, p = se23_parts(X)
    a_z = (R @ accel + g_vec)[2]
    a_c, b_c, c_c = 0.5 * a_z, v[2], p[2]
    if abs(a_c) < 1e-12:
        roots = [-c_c / b_c] if abs(b_c) > 1e-12 else []
    else:
        disc = b_c * b_c - 4.0 * a_c * c_c
        if disc < 0.0:
            return None
        s = np.sqrt(disc)
        roots = [(-b_c + s) / (2.0 * a_c), (-b_c - s) / (2.0 * a_c)]
    descending = [r for r in roots if r > 1e-12 and (b_c + a_z * r) < 0.0]
    return min(descending) if descending else None


# --- Error-coordinate Jacobians ----------------------------------------------

def propagation_matrix(h, g_vec):
    """Exact transition matrix of the right-invariant error over h,
    Phi(h) = expm(A h) with A = [[0,0,0],[g^,0,0],[0,I,0]] -- the same for
    every trajectory (the group-affine property).
    Arguments:
        h: interval (s)
        g_vec: (3,) gravity vector in world frame
    Returns:
        Phi: (9,9)
    """
    Phi = np.eye(9)
    Phi[3:6, 0:3] = h * skew(g_vec)
    Phi[6:9, 0:3] = 0.5 * h * h * skew(g_vec)
    Phi[6:9, 3:6] = h * np.eye(3)
    return Phi


def body_noise_covariance(h, gyro_noise_std, accel_noise_std):
    """Covariance of the body-frame increment noise over one interval h:
    rotation (gyro_noise_std h)^2, velocity (accel_noise_std h)^2, position
    (0.5 accel_noise_std h^2)^2 -- the same simple per-sample heuristic as
    saltation_matrix_ekf.py's process_noise_covariance.
    Arguments:
        h: interval (s)
        gyro_noise_std: gyro noise std-dev per sample (rad/s)
        accel_noise_std: accel noise std-dev per sample (m/s^2)
    Returns:
        Q_b: (9,9)
    """
    return np.diag([(gyro_noise_std * h) ** 2] * 3
                   + [(accel_noise_std * h) ** 2] * 3
                   + [(0.5 * accel_noise_std * h * h) ** 2] * 3)


def world_to_invariant_map(X):
    """T(X): maps a right-invariant error xi (X' = exp(xi) X) to the
    world-perturbation coordinates (dtheta, dv, dp) of the same perturbed
    state (R' = Exp(dtheta) R, v' = v + dv, p' = p + dp), to first order:
    dtheta = xi_R, dv = xi_v - v^ xi_R, dp = xi_p - p^ xi_R.
    Arguments:
        X: (5,5) base state
    Returns:
        T: (9,9), with d_world = T @ xi
    """
    _, v, p = se23_parts(X)
    T = np.eye(9)
    T[3:6, 0:3] = -skew(v)
    T[6:9, 0:3] = -skew(p)
    return T


def vector_field_world(X, gyro, accel, g_vec):
    """The continuous dynamics in world-perturbation coordinates at X:
    (R gyro, R accel + g, v).
    Arguments:
        X: (5,5) state
        gyro, accel: (3,) body IMU readings
        g_vec: (3,) gravity vector
    Returns:
        f: (9,)
    """
    R, v, _ = se23_parts(X)
    return np.concatenate([R @ gyro, R @ accel + g_vec, v])


@dataclass(frozen=True)
class BounceConfig:
    """Settings that bound how step_hybrid_ri and step_hybrid_world resolve
    bounces within one step.
    Fields:
        max_bounces: safety cap on bounces processed within one step
        min_bounce_speed: impact speed (m/s) below which a bounce raises
            RuntimeError instead of being resolved (the slow-bounce settling
            phase is not modelled)
    """
    max_bounces: int = 5
    min_bounce_speed: float = 0.5


def saltation_matrix_world(X_minus, gyro, accel, e, g_vec):
    """Saltation matrix in world-perturbation coordinates (Kong et al. 2024,
    Def. 2, time-invariant guard and reset):
        Xi = DR + (f+ - DR f-) Dg^T / (Dg . f-)
    with DR = blockdiag(I, M, I), guard g(X) = p_z (Dg picks dp_z), and f-/f+
    the vector field before/after the reset (only the position rate changes:
    v -> M v). Assumes X_minus is on the guard.
    Arguments:
        X_minus: (5,5) pre-impact state
        gyro, accel: (3,) IMU readings at the impact
        e: restitution coefficient
        g_vec: (3,) gravity vector
    Returns:
        Xi: (9,9)
    """
    M = restitution_matrix(e)
    DR = np.eye(9)
    DR[3:6, 3:6] = M
    f_minus = vector_field_world(X_minus, gyro, accel, g_vec)
    f_plus = vector_field_world(reset_map(X_minus, e), gyro, accel, g_vec)
    Dg = np.zeros(9)
    Dg[8] = 1.0
    return DR + np.outer(f_plus - DR @ f_minus, Dg) / (Dg @ f_minus)


def saltation_matrix_ri(X_minus, gyro, accel, e, g_vec):
    """Saltation matrix in right-invariant error coordinates:
    Xi_RI = T(X+)^-1 Xi_world T(X-). State-dependent through T, because the
    reset is not a group automorphism.
    Arguments:
        X_minus: (5,5) pre-impact state (on the guard)
        gyro, accel: (3,) IMU readings at the impact
        e: restitution coefficient
        g_vec: (3,) gravity vector
    Returns:
        Xi: (9,9)
    """
    X_plus = reset_map(X_minus, e)
    return (np.linalg.inv(world_to_invariant_map(X_plus))
            @ saltation_matrix_world(X_minus, gyro, accel, e, g_vec)
            @ world_to_invariant_map(X_minus))


def reset_jacobian_ri(X_minus, e):
    """Jacobian of the reset map alone in right-invariant coordinates,
    T(X+)^-1 DR T(X-) -- no impact-timing correction.
    Arguments:
        X_minus: (5,5) pre-impact state
        e: restitution coefficient
    Returns:
        J: (9,9)
    """
    DR = np.eye(9)
    DR[3:6, 3:6] = restitution_matrix(e)
    return (np.linalg.inv(world_to_invariant_map(reset_map(X_minus, e)))
            @ DR @ world_to_invariant_map(X_minus))


def invariant_naive_reset(e):
    """The shortcut blockdiag(I, M, I): applies the reset to xi_v as if the
    reset were an automorphism. State-independent, and wrong.
    Arguments:
        e: restitution coefficient
    Returns:
        J: (9,9)
    """
    J = np.eye(9)
    J[3:6, 3:6] = restitution_matrix(e)
    return J


def impact_timing_direction_ri(X_minus, gyro, accel, e, g_vec):
    """How a bounce applied delta seconds late moves the post-impact state,
    per second, in right-invariant coordinates: w = T(X+)^-1 (f+ - DR f-),
    the same vector the saltation correction is built from. Flowing delta
    past the crossing with f- and then resetting gives R(x-) + DR f- delta,
    the exact hybrid flow gives R(x-) + f+ delta, so a late reset is off by
    -w delta (finite-difference checked in the tests). A detection-time error
    with RMS s therefore adds s^2 w w^T to the post-impact covariance -- the
    "saltation_timing" filter.
    Arguments:
        X_minus: (5,5) pre-impact state
        gyro, accel: (3,) IMU readings at the impact
        e: restitution coefficient
        g_vec: (3,) gravity vector
    Returns:
        w: (9,)
    """
    X_plus = reset_map(X_minus, e)
    DR = np.eye(9)
    DR[3:6, 3:6] = restitution_matrix(e)
    w_world = (vector_field_world(X_plus, gyro, accel, g_vec)
               - DR @ vector_field_world(X_minus, gyro, accel, g_vec))
    return np.linalg.solve(world_to_invariant_map(X_plus), w_world)


def detected_crossing_time(tau, remaining, detect_time_bias, detect_time_noise_std, detect_rng):
    """Contact-detection model of saltation_matrix_ekf.py's step_hybrid: the
    bounce is applied at tau_detect = clip(tau + bias [+ N(0, std^2)], 0,
    remaining) instead of the estimate's own crossing time tau. Clipping into
    the current tick is a deliberate simplification (no carry-over of a late
    detection into the next tick).
    Arguments:
        tau: crossing time predicted from the estimate (s), within this tick
        remaining: time left in the tick (s)
        detect_time_bias: systematic offset (s), positive = late detection
        detect_time_noise_std: std-dev of Gaussian jitter (s)
        detect_rng: numpy Generator for the jitter; required if the std is > 0
    Returns:
        tau_detect: (s)
    """
    tau_detect = tau + detect_time_bias
    if detect_time_noise_std > 0.0:
        tau_detect += detect_rng.normal(0.0, detect_time_noise_std)
    return float(np.clip(tau_detect, 0.0, remaining))


def measurement_jacobian_ri(landmarks):
    """Stacked right-invariant measurement Jacobian for residuals
    r_i = R_hat y_i + p_hat - l_i: H_i = [-l_i^, 0, I] (constant).
    Arguments:
        landmarks: (L,3) known landmark positions (world)
    Returns:
        H: (3L, 9)
    """
    H = np.zeros((3 * len(landmarks), 9))
    for i, l in enumerate(landmarks):
        H[3 * i:3 * i + 3, 0:3] = -skew(l)
        H[3 * i:3 * i + 3, 6:9] = np.eye(3)
    return H


# --- Ground truth and sensor data ---------------------------------------------

def step_truth(X, dt, gyro, e, g_vec):
    """Exact hybrid flow of the true body over dt (zero specific force,
    constant body rate), resolving every bounce inside the step.
    Arguments:
        X: (5,5) state
        dt: step (s)
        gyro: (3,) true body rate (rad/s)
        e: restitution coefficient
        g_vec: (3,) gravity vector
    Returns:
        X_new: (5,5) state after dt
        impacts: list of impact times relative to the start of the step (s)
    """
    accel = np.zeros(3)
    remaining, elapsed, impacts = dt, 0.0, []
    while remaining > 0.0:
        tau = crossing_time(X, accel, g_vec)
        if tau is None or tau >= remaining:
            return propagate_mean(X, gyro, accel, remaining, g_vec), impacts
        X = reset_map(propagate_mean(X, gyro, accel, tau, g_vec), e)
        elapsed += tau
        remaining -= tau
        impacts.append(elapsed)
    return X, impacts


def generate_ground_truth_and_data(duration, dt, e, g_vec, X0, body_rate, landmarks, meas_every,
                                   gyro_noise_std, accel_noise_std, meas_noise_std, rng):
    """True trajectory plus noisy IMU readings (every step) and body-frame
    landmark measurements y_i = R^T (l_i - p) + noise (every meas_every steps).
    Arguments:
        duration: total time (s)
        dt: IMU step (s)
        e: restitution coefficient
        g_vec: (3,) gravity vector
        X0: (5,5) initial true state
        body_rate: (3,) true constant body rate (rad/s)
        landmarks: (L,3) landmark positions
        meas_every: landmark measurement period, in IMU steps
        gyro_noise_std, accel_noise_std: IMU noise std-devs per sample
        meas_noise_std: landmark measurement noise std-dev (m)
        rng: numpy Generator
    Returns:
        X_true: list of (5,5) true states at each tick (including t=0)
        gyro: (N,3) gyro readings, accel: (N,3) accel readings
        meas: dict tick -> (L,3) landmark measurements
        impact_ticks: list of the first tick index after each true impact
    """
    n_steps = int(round(duration / dt))
    X_true, impact_ticks = [X0], []
    for k in range(n_steps):
        X_next, impacts = step_truth(X_true[-1], dt, body_rate, e, g_vec)
        impact_ticks.extend([k + 1] * len(impacts))
        X_true.append(X_next)
    gyro = body_rate + rng.normal(0.0, gyro_noise_std, (n_steps, 3))
    accel = rng.normal(0.0, accel_noise_std, (n_steps, 3))
    meas = {}
    for k in range(meas_every, n_steps + 1, meas_every):
        R, _, p = se23_parts(X_true[k])
        meas[k] = (landmarks - p) @ R + rng.normal(0.0, meas_noise_std, landmarks.shape)
    return X_true, gyro, accel, meas, impact_ticks


# --- Filters ------------------------------------------------------------------

def step_hybrid_ri(X, P, gyro, accel, dt, e, g_vec, gyro_noise_std, accel_noise_std, reset_mode,
                   detect_time_bias=0.0, detect_time_noise_std=0.0, detect_rng=None, timing_noise_std=0.0,
                   bounce_cfg=BounceConfig()):
    """RIEKF predict over dt through the hybrid dynamics, splitting the step at
    every ground crossing predicted from the estimate (applied at the detected
    time, see detected_crossing_time). reset_mode selects the covariance
    treatment at a bounce: "invariant_naive", "reset_jacobian", "saltation",
    or "saltation_timing" (saltation plus timing_noise_std^2 w w^T, see
    impact_timing_direction_ri).
    Arguments:
        X, P: (5,5) estimate, (9,9) covariance of xi = log(X_hat X^-1)
        gyro, accel: (3,) IMU readings for this step
        dt: step (s)
        e: restitution coefficient
        g_vec: (3,) gravity vector
        gyro_noise_std, accel_noise_std: IMU noise std-devs per sample
        reset_mode: see above
        detect_time_bias, detect_time_noise_std, detect_rng: contact-detection
            error (see detected_crossing_time); all default to exact detection
        timing_noise_std: RMS detection-time error the "saltation_timing"
            filter assumes (s); ignored by the other modes
        bounce_cfg: BounceConfig instance with max_bounces and min_bounce_speed
    Returns:
        X, P after dt
        n_bounces: number of bounces resolved in this step
    """
    remaining, n_bounces = dt, 0
    for _ in range(bounce_cfg.max_bounces + 1):
        tau = crossing_time(X, accel, g_vec)
        impact = tau is not None and tau < remaining
        h = (detected_crossing_time(tau, remaining, detect_time_bias, detect_time_noise_std, detect_rng)
             if impact else remaining)
        X = propagate_mean(X, gyro, accel, h, g_vec)
        Phi = propagation_matrix(h, g_vec)
        Ad = se23_adjoint(X)
        P = Phi @ P @ Phi.T + Ad @ body_noise_covariance(h, gyro_noise_std, accel_noise_std) @ Ad.T
        remaining -= h
        if not impact:
            return X, P, n_bounces
        if abs(X[2, 3]) < bounce_cfg.min_bounce_speed:
            raise RuntimeError("impact slower than min_bounce_speed; shorten --duration")
        Q_timing = np.zeros((9, 9))
        if reset_mode in ("saltation", "saltation_timing"):
            Xi = saltation_matrix_ri(X, gyro, accel, e, g_vec)
            if reset_mode == "saltation_timing":
                w = impact_timing_direction_ri(X, gyro, accel, e, g_vec)
                Q_timing = timing_noise_std ** 2 * np.outer(w, w)
        elif reset_mode == "reset_jacobian":
            Xi = reset_jacobian_ri(X, e)
        elif reset_mode == "invariant_naive":
            Xi = invariant_naive_reset(e)
        else:
            raise ValueError(f"unknown reset_mode {reset_mode!r}")
        X = reset_map(X, e)
        P = Xi @ P @ Xi.T + Q_timing
        n_bounces += 1
        if remaining <= 0.0:
            return X, P, n_bounces
    raise RuntimeError("too many bounces in one step")


def update_ri(X, P, y, landmarks, meas_noise_std):
    """RIEKF update with body-frame landmark measurements. Residual
    r = R_hat y_i + p_hat - l_i ~ H xi, noise R_hat N R_hat^T; correction
    X_hat <- exp(-K r) X_hat (since X = exp(-xi) X_hat). Joseph form.
    Arguments:
        X, P: (5,5) estimate, (9,9) covariance
        y: (L,3) measurements
        landmarks: (L,3) landmark positions
        meas_noise_std: measurement noise std-dev (m)
    Returns:
        X, P after the update
    """
    R, _, p = se23_parts(X)
    r = (y @ R.T + p - landmarks).reshape(-1)
    H = measurement_jacobian_ri(landmarks)
    N_block = meas_noise_std ** 2 * R @ R.T
    N = np.kron(np.eye(len(landmarks)), N_block)
    S = H @ P @ H.T + N
    K = P @ H.T @ np.linalg.inv(S)
    X = se23_exp(-K @ r) @ X
    I_KH = np.eye(9) - K @ H
    return X, I_KH @ P @ I_KH.T + K @ N @ K.T


def step_hybrid_world(X, P, gyro, accel, dt, e, g_vec, gyro_noise_std, accel_noise_std,
                      detect_time_bias=0.0, detect_time_noise_std=0.0, detect_rng=None,
                      bounce_cfg=BounceConfig()):
    """Ordinary error-state EKF predict (world-perturbation coordinates) with
    the saltation correction at each bounce. Its propagation Jacobian depends
    on the estimate:
        F = [[I,0,0], [-(R accel h)^, I, 0], [-(R 0.5 accel h^2)^, h I, I]].
    Arguments:
        X, P: (5,5) estimate, (9,9) covariance of (dtheta, dv, dp)
        gyro, accel: (3,) IMU readings
        dt: step (s)
        e: restitution coefficient
        g_vec: (3,) gravity vector
        gyro_noise_std, accel_noise_std: IMU noise std-devs per sample
        detect_time_bias, detect_time_noise_std, detect_rng, bounce_cfg: as step_hybrid_ri
    Returns:
        X, P after dt
        n_bounces: number of bounces resolved in this step
    """
    remaining, n_bounces = dt, 0
    for _ in range(bounce_cfg.max_bounces + 1):
        tau = crossing_time(X, accel, g_vec)
        impact = tau is not None and tau < remaining
        h = (detected_crossing_time(tau, remaining, detect_time_bias, detect_time_noise_std, detect_rng)
             if impact else remaining)
        R, _, _ = se23_parts(X)
        F = np.eye(9)
        F[3:6, 0:3] = -skew(R @ (accel * h))
        F[6:9, 0:3] = -skew(R @ (0.5 * accel * h * h))
        F[6:9, 3:6] = h * np.eye(3)
        G = np.kron(np.eye(3), R)
        X = propagate_mean(X, gyro, accel, h, g_vec)
        P = F @ P @ F.T + G @ body_noise_covariance(h, gyro_noise_std, accel_noise_std) @ G.T
        remaining -= h
        if not impact:
            return X, P, n_bounces
        if abs(X[2, 3]) < bounce_cfg.min_bounce_speed:
            raise RuntimeError("impact slower than min_bounce_speed; shorten --duration")
        Xi = saltation_matrix_world(X, gyro, accel, e, g_vec)
        X = reset_map(X, e)
        P = Xi @ P @ Xi.T
        n_bounces += 1
        if remaining <= 0.0:
            return X, P, n_bounces
    raise RuntimeError("too many bounces in one step")


def measurement_jacobian_world(X, landmarks):
    """World-perturbation measurement Jacobian for y_i = R^T (l_i - p):
    H_i = [R_hat^T (l_i - p_hat)^, 0, -R_hat^T] -- depends on the estimate,
    unlike measurement_jacobian_ri.
    Arguments:
        X: (5,5) estimate
        landmarks: (L,3) landmark positions
    Returns:
        H: (3L, 9)
    """
    R, _, p = se23_parts(X)
    H = np.zeros((3 * len(landmarks), 9))
    for i, l in enumerate(landmarks):
        H[3 * i:3 * i + 3, 0:3] = R.T @ skew(l - p)
        H[3 * i:3 * i + 3, 6:9] = -R.T
    return H


def update_world(X, P, y, landmarks, meas_noise_std):
    """World-perturbation EKF update: y_hat_i = R_hat^T (l_i - p_hat), H from
    measurement_jacobian_world; correction
    R <- Exp(dtheta) R, v += dv, p += dp. Joseph form.
    Arguments:
        X, P: (5,5) estimate, (9,9) covariance
        y: (L,3) measurements
        landmarks: (L,3) landmark positions
        meas_noise_std: measurement noise std-dev (m)
    Returns:
        X, P after the update
    """
    R, v, p = se23_parts(X)
    y_hat = (landmarks - p) @ R
    H = measurement_jacobian_world(X, landmarks)
    N = meas_noise_std ** 2 * np.eye(3 * len(landmarks))
    S = H @ P @ H.T + N
    K = P @ H.T @ np.linalg.inv(S)
    d = K @ (y - y_hat).reshape(-1)
    X = se23_make(so3_exp(d[0:3]) @ R, v + d[3:6], p + d[6:9])
    I_KH = np.eye(9) - K @ H
    return X, I_KH @ P @ I_KH.T + K @ N @ K.T


FILTERS = ("invariant_naive", "reset_jacobian", "saltation", "saltation_timing", "world_saltation")


def run_filter(name, X_init, P_init, gyro, accel, meas, dt, e, g_vec, landmarks,
               gyro_noise_std, accel_noise_std, meas_noise_std,
               detect_time_bias=0.0, detect_time_noise_std=0.0, detect_rng=None,
               bounce_cfg=BounceConfig()):
    """Runs one of FILTERS over the whole data set. "saltation_timing" assumes
    a detection-time error RMS of hypot(detect_time_bias,
    detect_time_noise_std), i.e. it is tuned to the true detector.
    Arguments:
        name: one of FILTERS
        X_init: (5,5) initial estimate
        P_init: (9,9) initial covariance, in that filter's own error coordinates
        gyro, accel: (N,3) IMU readings
        meas: dict tick -> (L,3) landmark measurements
        dt, e, g_vec, landmarks, *_noise_std: as above
        detect_time_bias, detect_time_noise_std, detect_rng: contact-detection
            error, see detected_crossing_time
        bounce_cfg: BounceConfig instance with max_bounces and min_bounce_speed
    Returns:
        X_list: list of (5,5) estimates, P_list: list of (9,9) covariances
        bounce_count: (N,) cumulative number of bounces the filter has resolved, per tick
    """
    X, P = X_init.copy(), P_init.copy()
    X_list, P_list, counts = [X], [P], [0]
    for k in range(len(gyro)):
        if name == "world_saltation":
            X, P, nb = step_hybrid_world(X, P, gyro[k], accel[k], dt, e, g_vec,
                                         gyro_noise_std, accel_noise_std,
                                         detect_time_bias, detect_time_noise_std, detect_rng, bounce_cfg)
            if k + 1 in meas:
                X, P = update_world(X, P, meas[k + 1], landmarks, meas_noise_std)
        else:
            X, P, nb = step_hybrid_ri(X, P, gyro[k], accel[k], dt, e, g_vec,
                                      gyro_noise_std, accel_noise_std, reset_mode=name,
                                      detect_time_bias=detect_time_bias,
                                      detect_time_noise_std=detect_time_noise_std, detect_rng=detect_rng,
                                      timing_noise_std=np.hypot(detect_time_bias, detect_time_noise_std),
                                      bounce_cfg=bounce_cfg)
            if k + 1 in meas:
                X, P = update_ri(X, P, meas[k + 1], landmarks, meas_noise_std)
        X_list.append(X)
        P_list.append(P)
        counts.append(counts[-1] + nb)
    return X_list, P_list, np.array(counts)


# --- Metrics ------------------------------------------------------------------

def error_vector(name, X_true, X_est):
    """Estimation error in the named filter's own coordinates: xi =
    log(X_hat X^-1) for the RIEKFs, (log(R R_hat^T), v - v_hat, p - p_hat) for
    the world EKF.
    Arguments:
        name: one of FILTERS
        X_true, X_est: (5,5)
    Returns:
        err: (9,)
    """
    if name == "world_saltation":
        R, v, p = se23_parts(X_true)
        R_h, v_h, p_h = se23_parts(X_est)
        return np.concatenate([so3_log(R @ R_h.T), v - v_h, p - p_h])
    return se23_log(X_est @ se23_inv(X_true))


def state_errors(X_true_list, X_est_list):
    """Rotation (deg), velocity (m/s) and position (m) error norms per tick.
    Arguments:
        X_true_list, X_est_list: lists of (5,5)
    Returns:
        rot_err, vel_err, pos_err: (N,) arrays
    """
    rot, vel, pos = [], [], []
    for Xt, Xe in zip(X_true_list, X_est_list):
        Rt, vt, pt = se23_parts(Xt)
        Re, ve, pe = se23_parts(Xe)
        rot.append(np.degrees(np.linalg.norm(so3_log(Re @ Rt.T))))
        vel.append(np.linalg.norm(ve - vt))
        pos.append(np.linalg.norm(pe - pt))
    return np.array(rot), np.array(vel), np.array(pos)


def initial_estimate(X_true0, P0_ri, rng):
    """Draws the shared initial estimate X_hat = exp(xi0) X_true0, xi0 ~
    N(0, P0_ri), and returns each filter's initial covariance in its own
    coordinates (world EKF: T(X_hat) P0_ri T(X_hat)^T, sign-free since T is
    applied symmetrically).
    Arguments:
        X_true0: (5,5) true initial state
        P0_ri: (9,9) initial covariance of the right-invariant error
        rng: numpy Generator
    Returns:
        X_init: (5,5), P_init: dict name -> (9,9)
    """
    xi0 = np.linalg.cholesky(P0_ri) @ rng.normal(0.0, 1.0, 9)
    X_init = se23_exp(xi0) @ X_true0
    T = world_to_invariant_map(X_init)
    P_init = {name: P0_ri.copy() for name in FILTERS}
    P_init["world_saltation"] = T @ P0_ri @ T.T
    return X_init, P_init


def true_bounce_counts(impact_ticks, n_ticks):
    """Cumulative number of true bounces at each tick.
    Arguments:
        impact_ticks: first tick after each true impact
        n_ticks: number of ticks (including t=0)
    Returns:
        counts: (n_ticks,) int array
    """
    return np.searchsorted(np.sort(impact_ticks), np.arange(n_ticks), side="right")


def run_monte_carlo_consistency(X_true_list, impact_ticks, gyro_true, dt, e, g_vec, landmarks, meas_every,
                                P0_ri, gyro_noise_std, accel_noise_std, meas_noise_std, n_trials, rng,
                                detect_time_bias=0.0, detect_time_noise_std=0.0, bounce_cfg=BounceConfig()):
    """Average NEES per tick for every filter, over n_trials independent
    realizations of the IMU noise, measurement noise and initial error on the
    same nominal true trajectory.

    Mode matching: a tick is scored only when the filter has resolved the same
    number of bounces as the truth. Around each impact there are usually one
    or two ticks where one of them has bounced and the other has not; there
    the velocity error is of the order of (1+e)|v_z| -- tens of m/s -- and
    NEES reaches ~1e5 for every filter alike. That is a property of comparing
    a hybrid estimate to a hybrid truth at fixed ticks, not of the covariance
    treatment being compared (saltation_matrix_ekf.py reports the same shared
    spike), so it is excluded and reported separately as a mismatch rate.

    Detection jitter comes from one child Generator per filter
    (child_detect_rngs), so one filter's draws never shift another's, and
    rng's own stream -- IMU noise, measurements, initial errors -- is the same
    with or without jitter: a jitter sweep at a fixed seed differs only in
    the jitter.
    Arguments:
        X_true_list: list of (5,5) true states
        gyro_true: (3,) true body rate
        dt, e, g_vec, landmarks, meas_every: as above
        P0_ri: (9,9) initial right-invariant covariance
        *_noise_std: noise std-devs
        n_trials: number of trials
        rng: numpy Generator
        detect_time_bias, detect_time_noise_std: contact-detection error
        bounce_cfg: BounceConfig, see step_hybrid_ri
    Returns:
        avg_nees: dict name -> (N,) average NEES per tick over mode-matched
            trials (nan where no trial is mode-matched)
        mismatch_rate: dict name -> fraction of (trial, tick) pairs excluded
    """
    n_steps = len(X_true_list) - 1
    counts_true = true_bounce_counts(impact_ticks, n_steps + 1)
    sums = {name: np.zeros(n_steps + 1) for name in FILTERS}
    valid = {name: np.zeros(n_steps + 1) for name in FILTERS}
    detect_rngs = child_detect_rngs(rng)
    for _ in range(n_trials):
        gyro = gyro_true + rng.normal(0.0, gyro_noise_std, (n_steps, 3))
        accel = rng.normal(0.0, accel_noise_std, (n_steps, 3))
        meas = {}
        for k in range(meas_every, n_steps + 1, meas_every):
            R, _, p = se23_parts(X_true_list[k])
            meas[k] = (landmarks - p) @ R + rng.normal(0.0, meas_noise_std, landmarks.shape)
        X_init, P_init = initial_estimate(X_true_list[0], P0_ri, rng)
        for name in FILTERS:
            X_list, P_list, counts = run_filter(name, X_init, P_init[name], gyro, accel, meas, dt, e, g_vec,
                                                landmarks, gyro_noise_std, accel_noise_std, meas_noise_std,
                                                detect_time_bias, detect_time_noise_std, detect_rngs[name], bounce_cfg)
            for k in np.flatnonzero(counts == counts_true):
                sums[name][k] += nees(error_vector(name, X_true_list[k], X_list[k]), P_list[k])
                valid[name][k] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = {name: sums[name] / valid[name] for name in FILTERS}
    mismatch = {name: 1.0 - valid[name].sum() / (n_trials * (n_steps + 1)) for name in FILTERS}
    return avg, mismatch


def child_detect_rngs(rng):
    """One independent detection-jitter Generator per filter, spawned from
    rng's seed sequence. Spawning does not advance rng itself, so every other
    draw is unchanged by whether jitter is on.
    Arguments:
        rng: numpy Generator
    Returns:
        dict name -> Generator
    """
    return dict(zip(FILTERS, rng.spawn(len(FILTERS))))


def default_landmarks():
    """Four known landmarks around the drop zone (m)."""
    return np.array([[-3.0, -3.0, 1.0], [4.0, -2.0, 3.0], [3.0, 4.0, 0.5], [-2.0, 3.0, 4.0]])


LABELS = {
    "invariant_naive": "RIEKF, naive reset blockdiag(I,M,I)",
    "reset_jacobian": "RIEKF, reset Jacobian only",
    "saltation": "RIEKF, saltation",
    "saltation_timing": "RIEKF, saltation + timing noise",
    "world_saltation": "World-frame EKF, saltation",
}
COLORS = {"invariant_naive": "tab:red", "reset_jacobian": "tab:orange",
          "saltation": "tab:blue", "saltation_timing": "tab:cyan", "world_saltation": "tab:purple"}
# Dashed, so it stays visible where it coincides with "saltation" (exact detection).
LINESTYLES = {"saltation_timing": "--"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Simulation length (s). The defaults give 4 bounces; keep it short of "
                             "the slow-bounce regime (see min_bounce_speed)")
    parser.add_argument("--dt", type=float, default=0.01, help="IMU step (s)")
    parser.add_argument("--meas-every", type=int, default=10, help="Landmark measurement period (IMU steps)")
    parser.add_argument("--restitution", type=float, default=0.8, help="Bounce restitution coefficient (0-1)")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity magnitude (m/s^2)")
    parser.add_argument("--drop-height", type=float, default=5.0, help="Initial height (m)")
    parser.add_argument("--init-velocity", type=float, nargs=3, default=[1.0, 0.5, 0.0],
                        metavar=("VX", "VY", "VZ"), help="Initial velocity (m/s)")
    parser.add_argument("--body-rate", type=float, nargs=3, default=[1.0, -0.6, 2.0],
                        metavar=("WX", "WY", "WZ"), help="Constant true body rate (rad/s)")
    parser.add_argument("--gyro-noise-std", type=float, default=0.02, help="Gyro noise std-dev per sample (rad/s)")
    parser.add_argument("--accel-noise-std", type=float, default=0.3, help="Accel noise std-dev per sample (m/s^2)")
    parser.add_argument("--meas-noise-std", type=float, default=0.05, help="Landmark measurement noise std-dev (m)")
    parser.add_argument("--init-rot-std-deg", type=float, default=5.0, help="Initial rotation error std-dev (deg)")
    parser.add_argument("--init-vel-std", type=float, default=0.2, help="Initial velocity error std-dev (m/s)")
    parser.add_argument("--init-pos-std", type=float, default=0.1, help="Initial position error std-dev (m)")
    parser.add_argument("--detect-time-bias", type=float, default=0.0,
                        help="Contact-detection time offset (s), positive = late; 0 = exact detection")
    parser.add_argument("--detect-time-noise-std", type=float, default=0.0,
                        help="Contact-detection time jitter std-dev (s); 0 = exact detection")
    parser.add_argument("--n-mc-trials", type=int, default=200, help="Monte Carlo consistency trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure here instead of showing it")
    parser.add_argument("--max-bounces", type=int, default=5,
                        help="Safety cap on bounces processed within one step (see BounceConfig)")
    parser.add_argument("--min-bounce-speed", type=float, default=0.5,
                        help="Impact speed (m/s) below which a bounce raises RuntimeError "
                             "(settling phase not modelled; see BounceConfig)")
    args = parser.parse_args()

    bounce_cfg = BounceConfig(max_bounces=args.max_bounces, min_bounce_speed=args.min_bounce_speed)
    rng = np.random.default_rng(args.seed)
    g_vec = np.array([0.0, 0.0, -args.gravity])
    landmarks = default_landmarks()
    body_rate = np.array(args.body_rate)
    X0 = se23_make(so3_exp(np.array([0.2, -0.1, 0.3])), np.array(args.init_velocity),
                   np.array([0.0, 0.0, args.drop_height]))
    P0_ri = np.diag([np.radians(args.init_rot_std_deg) ** 2] * 3 + [args.init_vel_std ** 2] * 3
                    + [args.init_pos_std ** 2] * 3)

    X_true, gyro, accel, meas, impact_ticks = generate_ground_truth_and_data(
        args.duration, args.dt, args.restitution, g_vec, X0, body_rate, landmarks, args.meas_every,
        args.gyro_noise_std, args.accel_noise_std, args.meas_noise_std, rng)
    n_steps = len(gyro)
    X_init, P_init = initial_estimate(X0, P0_ri, rng)
    print(f"{len(impact_ticks)} true bounces at t = "
          + ", ".join(f"{k * args.dt:.2f}" for k in impact_ticks) + " s")

    detect = (args.detect_time_bias, args.detect_time_noise_std)
    detect_rngs = child_detect_rngs(rng)
    results, perf = {}, {}
    for name in FILTERS:
        print(f"Running {LABELS[name]}...")
        (X_list, P_list, _), t_step, mem = measure_performance(
            run_filter, name, X_init, P_init[name], gyro, accel, meas, args.dt, args.restitution, g_vec,
            landmarks, args.gyro_noise_std, args.accel_noise_std, args.meas_noise_std, *detect, detect_rngs[name],
            n_steps=n_steps, bounce_cfg=bounce_cfg)
        results[name] = X_list
        perf[name] = (t_step, mem)

    print(f"Running Monte Carlo consistency check ({args.n_mc_trials} trials)...")
    avg_nees, mismatch = run_monte_carlo_consistency(
        X_true, impact_ticks, body_rate, args.dt, args.restitution, g_vec, landmarks, args.meas_every, P0_ri,
        args.gyro_noise_std, args.accel_noise_std, args.meas_noise_std, args.n_mc_trials, rng, *detect,
        bounce_cfg=bounce_cfg)

    print("\nRMS errors (single run):")
    errs = {}
    for name in FILTERS:
        errs[name] = state_errors(X_true, results[name])
        rot, vel, pos = errs[name]
        print(f"  {LABELS[name]:<36s} rot={np.sqrt(np.mean(rot ** 2)):6.3f} deg  "
              f"vel={np.sqrt(np.mean(vel ** 2)):6.4f} m/s  pos={np.sqrt(np.mean(pos ** 2)):6.4f} m")

    lo, hi = averaged_nees_bounds(9, args.n_mc_trials)
    window = args.meas_every
    print(f"\nMonte Carlo NEES over mode-matched ticks (consistent 9-DoF filter ~9; 95% bounds for a "
          f"{args.n_mc_trials}-trial average: [{lo:.2f}, {hi:.2f}]):")
    for name in FILTERS:
        a = avg_nees[name]
        post = np.concatenate([a[k + 1:k + 1 + window] for k in impact_ticks])
        print(f"  {LABELS[name]:<36s} before 1st bounce={np.nanmean(a[1:impact_ticks[0] - 1]):7.2f}  "
              f"after bounces (ticks k+1..k+{window})={np.nanmean(post):7.2f}  overall={np.nanmean(a):7.2f}  "
              f"mismatched ticks={100 * mismatch[name]:4.2f}%")

    print("\nAverage time (per step) and peak memory (whole run):")
    for name in FILTERS:
        t_step, mem = perf[name]
        print(f"  {LABELS[name]:<36s} {t_step * 1e6:9.2f} us/step | {mem / 1024.0:9.3f} KB")

    t = np.arange(n_steps + 1) * args.dt
    fig, (ax_h, ax_pos, ax_nees) = plt.subplots(3, 1, figsize=(9, 11))
    ax_h.plot(t, [X[2, 4] for X in X_true], "g--", linewidth=2, label="Ground truth")
    for name in FILTERS:
        ax_h.plot(t, [X[2, 4] for X in results[name]], color=COLORS[name], linestyle=LINESTYLES.get(name, "-"),
                  label=LABELS[name])
    ax_h.set_ylabel("Height p_z (m)")
    title = "Right-invariant EKF on SE_2(3) through bounces"
    if args.detect_time_bias or args.detect_time_noise_std:
        title += (f"\ncontact detection: bias {args.detect_time_bias * 1e3:g} ms, "
                  f"jitter std {args.detect_time_noise_std * 1e3:g} ms")
    ax_h.set_title(title)
    ax_h.legend(fontsize=8)
    for name in FILTERS:
        ax_pos.plot(t, errs[name][2], color=COLORS[name], linestyle=LINESTYLES.get(name, "-"),
                    label=LABELS[name])
    ax_pos.set_ylabel("Position error (m)")
    ax_pos.set_yscale("log")
    for name in FILTERS:
        ax_nees.plot(t, avg_nees[name], color=COLORS[name], linestyle=LINESTYLES.get(name, "-"),
                     label=LABELS[name])
    ax_nees.axhline(9.0, color="black", linewidth=1, label="Expected NEES (9 DoF)")
    ax_nees.axhline(lo, color="black", linestyle="--", linewidth=1)
    ax_nees.axhline(hi, color="black", linestyle="--", linewidth=1)
    for k in impact_ticks:
        for ax in (ax_h, ax_pos, ax_nees):
            ax.axvline(k * args.dt, color="gray", linestyle=":", linewidth=1)
    ax_nees.set_yscale("log")
    ax_nees.set_ylabel(f"Monte Carlo avg. NEES ({args.n_mc_trials} trials)")
    ax_nees.set_xlabel("Time (s)")
    ax_nees.legend(fontsize=8)
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
