'''
Same simulation as use_numpy/invariant_saltation_ekf.py -- a tumbling,
IMU-carrying rigid body on SE_2(3) bouncing off the ground `p_z = 0`, tracked
by right-invariant EKFs that differ only in how the covariance crosses a
bounce (invariant-naive, reset-Jacobian, saltation, saltation + timing noise
for contact-detection error), plus a world-frame EKF baseline -- but with the
group math delegated to the manif library (https://github.com/artivis/manif).
See the use_numpy twin's module docstring and
docs/filtering/invariant_saltation_ekf.md for the full story.

What manif does here: the state is a `manif.SE_2_3` object throughout; its
Exp/Log, composition and inverse, `adj()` (for mapping body-frame IMU noise
into the right-invariant error), `act()` (landmark measurements), and
`lplus`/`lminus` (the right-invariant error `xi = log(X_hat X^-1) =
X_hat.lminus(X)` and the correction `X_hat <- Exp(-K r) X_hat`). Mean
propagation is written in the group-affine form `X+ = Gamma f_h(X) Upsilon`
with manif products.

What stays hand-derived, because manif has no notion of it: the
trajectory-independent transition matrix Phi, the world-to-invariant map T(X),
the saltation matrix, the reset Jacobians, the impact-timing direction w and
the constant measurement Jacobian H. tests/use_manif/test_invariant_saltation_ekf.py
checks them against finite differences, against manif's own `act` Jacobian
and against the use_numpy version.

Tangent layout: manif's SE_2_3 tangent is [rho, theta, nu] = (position,
rotation, velocity) -- the use_numpy twin's (xi_R, xi_v, xi_p) reordered to
(xi_p, xi_R, xi_v). Every matrix below uses manif's order, as do the
world-frame EKF's error coordinates (dp, dtheta, dv). The numbers are the same
as the use_numpy version's up to that reordering (NEES is unaffected by it),
except where a random draw is shaped by the layout (`initial_estimate`).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
import manifpy as manif

from state_estimation_simulations.utils import measure_performance, averaged_nees_bounds, nees

# Blocks of a 9-vector in manif's SE_2_3 tangent order.
POS, ROT, VEL = slice(0, 3), slice(3, 6), slice(6, 9)
IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def skew(w):
    """3x3 skew-symmetric (hat) matrix of a 3-vector, via manif.
    Arguments:
        w: (3,) vector
    Returns:
        (3,3) matrix with skew(w) @ x == cross(w, x)
    """
    return manif.SO3Tangent(np.asarray(w, dtype=float)).hat()


# --- SE_2(3) helpers ----------------------------------------------------------

def se23_from(rot, v, p):
    """Builds an SE_2_3 element from an SO3 rotation, a velocity and a position
    (manif's data vector is [p, quaternion (x, y, z, w), v]).
    Arguments:
        rot: manif.SO3 rotation, v: (3,) velocity, p: (3,) position
    Returns:
        X: manif.SE_2_3
    """
    return manif.SE_2_3(np.concatenate([p, rot.coeffs(), v]))


def rotation_of(X):
    """The rotation part of X as a manif.SO3 (avoids a matrix-to-quaternion
    conversion, since manif has no rotation-matrix constructor).
    Arguments:
        X: manif.SE_2_3
    Returns:
        manif.SO3
    """
    return manif.SO3(X.coeffs_copy()[3:7])


def with_velocity_position(X, v, p):
    """X with its velocity and position replaced, rotation unchanged.
    Arguments:
        X: manif.SE_2_3, v: (3,) velocity, p: (3,) position
    Returns:
        manif.SE_2_3
    """
    return se23_from(rotation_of(X), v, p)


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
        X: manif.SE_2_3 pre-impact state
        e: restitution coefficient
    Returns:
        X_plus: manif.SE_2_3 post-impact state
    """
    return with_velocity_position(X, restitution_matrix(e) @ X.linearVelocity(), X.translation())


def reset_is_automorphism_gap(X1, X2, e):
    """How far the reset is from being a group automorphism:
    |Log(reset(X1 X2) (reset(X1) reset(X2))^-1)|_inf. Zero for every pair only
    if the reset commutes with the group product; for the bounce it does not
    (M R != R M), which is why its invariant-error Jacobian is state-dependent.
    Arguments:
        X1, X2: manif.SE_2_3
        e: restitution coefficient
    Returns:
        gap: scalar
    """
    lhs = reset_map(X1 * X2, e)
    rhs = reset_map(X1, e) * reset_map(X2, e)
    return float(np.max(np.abs(lhs.lminus(rhs).coeffs())))


def propagate_mean(X, gyro, accel, h, g_vec):
    """Mean propagation over h with constant IMU readings, as the group-affine
    product X+ = Gamma(h) f_h(X) Upsilon(h):
        Upsilon = (Exp(gyro h), accel h, 0.5 accel h^2)   body-frame increment
        f_h(X)  = (R, v, p + v h)                          (not a group product)
        Gamma   = (I, g h, 0.5 g h^2)                      gravity
    Exact for the true motion in this script (constant body rate, zero
    specific force in flight); first order in h for a noisy accel reading.
    Arguments:
        X: manif.SE_2_3 state
        gyro: (3,) body angular rate (rad/s)
        accel: (3,) body specific force (m/s^2)
        h: interval (s)
        g_vec: (3,) gravity vector in world frame
    Returns:
        X_new: manif.SE_2_3 propagated state
    """
    upsilon = se23_from(manif.SO3Tangent(gyro * h).exp(), accel * h, 0.5 * accel * h * h)
    gamma = manif.SE_2_3(np.concatenate([0.5 * g_vec * h * h, IDENTITY_QUAT, g_vec * h]))
    v = X.linearVelocity()
    return gamma * with_velocity_position(X, v, X.translation() + v * h) * upsilon


def crossing_time(X, accel, g_vec):
    """Time until the next *descending* ground crossing (p_z = 0), assuming
    the world-frame acceleration R accel + g stays constant over the interval
    (exact for the truth, where accel = 0).
    Arguments:
        X: manif.SE_2_3 state
        accel: (3,) body specific force (m/s^2)
        g_vec: (3,) gravity vector in world frame
    Returns:
        tau: time to the crossing (s), or None if there is none ahead
    """
    a_z = (X.rotation() @ accel + g_vec)[2]
    a_c, b_c, c_c = 0.5 * a_z, X.linearVelocity()[2], X.translation()[2]
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


# --- Error-coordinate Jacobians (hand-derived, manif tangent order) -----------

def propagation_matrix(h, g_vec):
    """Exact transition matrix of the right-invariant error over h, the same
    for every trajectory (the group-affine property). In (rho, theta, nu)
    order: theta is constant, nu gains h g^ theta, rho gains h nu +
    0.5 h^2 g^ theta.
    Arguments:
        h: interval (s)
        g_vec: (3,) gravity vector in world frame
    Returns:
        Phi: (9,9)
    """
    Phi = np.eye(9)
    Phi[VEL, ROT] = h * skew(g_vec)
    Phi[POS, ROT] = 0.5 * h * h * skew(g_vec)
    Phi[POS, VEL] = h * np.eye(3)
    return Phi


def body_noise_covariance(h, gyro_noise_std, accel_noise_std):
    """Covariance of the body-frame increment noise over one interval h, in
    (position, rotation, velocity) order: (0.5 accel_noise_std h^2)^2,
    (gyro_noise_std h)^2, (accel_noise_std h)^2 -- the same heuristic as the
    use_numpy version.
    Arguments:
        h: interval (s)
        gyro_noise_std: gyro noise std-dev per sample (rad/s)
        accel_noise_std: accel noise std-dev per sample (m/s^2)
    Returns:
        Q_b: (9,9)
    """
    return np.diag([(0.5 * accel_noise_std * h * h) ** 2] * 3
                   + [(gyro_noise_std * h) ** 2] * 3
                   + [(accel_noise_std * h) ** 2] * 3)


def world_to_invariant_map(X):
    """T(X): maps a right-invariant error xi (X' = Exp(xi) X) to the
    world-perturbation coordinates (dp, dtheta, dv) of the same perturbed
    state (p' = p + dp, R' = Exp(dtheta) R, v' = v + dv), to first order:
    dp = rho - p^ theta, dtheta = theta, dv = nu - v^ theta.
    Arguments:
        X: manif.SE_2_3 base state
    Returns:
        T: (9,9), with d_world = T @ xi
    """
    T = np.eye(9)
    T[POS, ROT] = -skew(X.translation())
    T[VEL, ROT] = -skew(X.linearVelocity())
    return T


def vector_field_world(X, gyro, accel, g_vec):
    """The continuous dynamics in world-perturbation coordinates at X:
    (v, R gyro, R accel + g).
    Arguments:
        X: manif.SE_2_3 state
        gyro, accel: (3,) body IMU readings
        g_vec: (3,) gravity vector
    Returns:
        f: (9,)
    """
    R = X.rotation()
    return np.concatenate([X.linearVelocity(), R @ gyro, R @ accel + g_vec])


def saltation_matrix_world(X_minus, gyro, accel, e, g_vec):
    """Saltation matrix in world-perturbation coordinates (Kong et al. 2024,
    Def. 2, time-invariant guard and reset):
        Xi = DR + (f+ - DR f-) Dg^T / (Dg . f-)
    with DR = I except its velocity block M, guard g(X) = p_z (Dg picks dp_z,
    index 2 in this order), and f-/f+ the vector field before/after the reset.
    Assumes X_minus is on the guard.
    Arguments:
        X_minus: manif.SE_2_3 pre-impact state
        gyro, accel: (3,) IMU readings at the impact
        e: restitution coefficient
        g_vec: (3,) gravity vector
    Returns:
        Xi: (9,9)
    """
    DR = np.eye(9)
    DR[VEL, VEL] = restitution_matrix(e)
    f_minus = vector_field_world(X_minus, gyro, accel, g_vec)
    f_plus = vector_field_world(reset_map(X_minus, e), gyro, accel, g_vec)
    Dg = np.zeros(9)
    Dg[2] = 1.0
    return DR + np.outer(f_plus - DR @ f_minus, Dg) / (Dg @ f_minus)


def saltation_matrix_ri(X_minus, gyro, accel, e, g_vec):
    """Saltation matrix in right-invariant error coordinates:
    Xi_RI = T(X+)^-1 Xi_world T(X-). State-dependent through T, because the
    reset is not a group automorphism.
    Arguments:
        X_minus: manif.SE_2_3 pre-impact state (on the guard)
        gyro, accel: (3,) IMU readings at the impact
        e: restitution coefficient
        g_vec: (3,) gravity vector
    Returns:
        Xi: (9,9)
    """
    return (np.linalg.inv(world_to_invariant_map(reset_map(X_minus, e)))
            @ saltation_matrix_world(X_minus, gyro, accel, e, g_vec)
            @ world_to_invariant_map(X_minus))


def reset_jacobian_ri(X_minus, e):
    """Jacobian of the reset map alone in right-invariant coordinates,
    T(X+)^-1 DR T(X-) -- no impact-timing correction.
    Arguments:
        X_minus: manif.SE_2_3 pre-impact state
        e: restitution coefficient
    Returns:
        J: (9,9)
    """
    DR = np.eye(9)
    DR[VEL, VEL] = restitution_matrix(e)
    return (np.linalg.inv(world_to_invariant_map(reset_map(X_minus, e)))
            @ DR @ world_to_invariant_map(X_minus))


def invariant_naive_reset(e):
    """The shortcut: M on the velocity block of xi, identity elsewhere, as if
    the reset were an automorphism. State-independent, and wrong.
    Arguments:
        e: restitution coefficient
    Returns:
        J: (9,9)
    """
    J = np.eye(9)
    J[VEL, VEL] = restitution_matrix(e)
    return J


def impact_timing_direction_ri(X_minus, gyro, accel, e, g_vec):
    """How a bounce applied delta seconds late moves the post-impact state,
    per second, in right-invariant coordinates: w = T(X+)^-1 (f+ - DR f-)
    (a late reset is off by -w delta). A detection-time error with RMS s
    adds s^2 w w^T to the post-impact covariance -- the "saltation_timing"
    filter.
    Arguments:
        X_minus: manif.SE_2_3 pre-impact state
        gyro, accel: (3,) IMU readings at the impact
        e: restitution coefficient
        g_vec: (3,) gravity vector
    Returns:
        w: (9,)
    """
    X_plus = reset_map(X_minus, e)
    DR = np.eye(9)
    DR[VEL, VEL] = restitution_matrix(e)
    w_world = (vector_field_world(X_plus, gyro, accel, g_vec)
               - DR @ vector_field_world(X_minus, gyro, accel, g_vec))
    return np.linalg.solve(world_to_invariant_map(X_plus), w_world)


def detected_crossing_time(tau, remaining, detect_time_bias, detect_time_noise_std, detect_rng):
    """Contact-detection model: the bounce is applied at tau_detect =
    clip(tau + bias [+ N(0, std^2)], 0, remaining) instead of the estimate's
    own crossing time tau (no carry-over into the next tick).
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
    r_i = X_hat.act(y_i) - l_i: H_i = [I, -l_i^, 0] (constant). It equals
    manif's own `act` Jacobian moved to the left (J_act Ad(X_hat)^-1) at a
    noise-free measurement -- checked in the tests.
    Arguments:
        landmarks: (L,3) known landmark positions (world)
    Returns:
        H: (3L, 9)
    """
    H = np.zeros((3 * len(landmarks), 9))
    for i, l in enumerate(landmarks):
        H[3 * i:3 * i + 3, POS] = np.eye(3)
        H[3 * i:3 * i + 3, ROT] = -skew(l)
    return H


# --- Ground truth and sensor data ---------------------------------------------

def measure_landmarks(X, landmarks):
    """Noise-free body-frame landmark observations y_i = X^-1.act(l_i) =
    R^T (l_i - p).
    Arguments:
        X: manif.SE_2_3 state
        landmarks: (L,3)
    Returns:
        y: (L,3)
    """
    X_inv = X.inverse()
    return np.array([X_inv.act(l) for l in landmarks])


def step_truth(X, dt, gyro, e, g_vec):
    """Exact hybrid flow of the true body over dt (zero specific force,
    constant body rate), resolving every bounce inside the step.
    Arguments:
        X: manif.SE_2_3 state
        dt: step (s)
        gyro: (3,) true body rate (rad/s)
        e: restitution coefficient
        g_vec: (3,) gravity vector
    Returns:
        X_new: manif.SE_2_3 state after dt
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
    landmark measurements (every meas_every steps), drawn in the same order as
    the use_numpy version, so the same seed gives the same data.
    Arguments:
        duration: total time (s)
        dt: IMU step (s)
        e: restitution coefficient
        g_vec: (3,) gravity vector
        X0: manif.SE_2_3 initial true state
        body_rate: (3,) true constant body rate (rad/s)
        landmarks: (L,3) landmark positions
        meas_every: landmark measurement period, in IMU steps
        gyro_noise_std, accel_noise_std: IMU noise std-devs per sample
        meas_noise_std: landmark measurement noise std-dev (m)
        rng: numpy Generator
    Returns:
        X_true: list of manif.SE_2_3 true states at each tick (including t=0)
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
        meas[k] = measure_landmarks(X_true[k], landmarks) + rng.normal(0.0, meas_noise_std, landmarks.shape)
    return X_true, gyro, accel, meas, impact_ticks


# --- Filters ------------------------------------------------------------------

def step_hybrid_ri(X, P, gyro, accel, dt, e, g_vec, gyro_noise_std, accel_noise_std, reset_mode,
                   detect_time_bias=0.0, detect_time_noise_std=0.0, detect_rng=None, timing_noise_std=0.0,
                   min_bounce_speed=0.5, max_bounces=5):
    """RIEKF predict over dt through the hybrid dynamics, splitting the step at
    every ground crossing predicted from the estimate (applied at the detected
    time, see detected_crossing_time). reset_mode selects the covariance
    treatment at a bounce: "invariant_naive", "reset_jacobian", "saltation",
    or "saltation_timing" (saltation plus timing_noise_std^2 w w^T).
    Body-frame IMU noise enters the invariant error through manif's adjoint,
    X.adj().
    Arguments:
        X, P: manif.SE_2_3 estimate, (9,9) covariance of xi = X_hat.lminus(X)
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
        min_bounce_speed: raise if an impact is slower than this (m/s)
        max_bounces: safety cap per step
    Returns:
        X, P after dt
        n_bounces: number of bounces resolved in this step
    """
    remaining, n_bounces = dt, 0
    for _ in range(max_bounces + 1):
        tau = crossing_time(X, accel, g_vec)
        impact = tau is not None and tau < remaining
        h = (detected_crossing_time(tau, remaining, detect_time_bias, detect_time_noise_std, detect_rng)
             if impact else remaining)
        X = propagate_mean(X, gyro, accel, h, g_vec)
        Phi = propagation_matrix(h, g_vec)
        Ad = X.adj()
        P = Phi @ P @ Phi.T + Ad @ body_noise_covariance(h, gyro_noise_std, accel_noise_std) @ Ad.T
        remaining -= h
        if not impact:
            return X, P, n_bounces
        if abs(X.linearVelocity()[2]) < min_bounce_speed:
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
    r_i = X_hat.act(y_i) - l_i ~ H xi, noise R_hat N R_hat^T; correction
    X_hat <- X_hat.lplus(-K r) = Exp(-K r) X_hat. Joseph form.
    Arguments:
        X, P: manif.SE_2_3 estimate, (9,9) covariance
        y: (L,3) measurements
        landmarks: (L,3) landmark positions
        meas_noise_std: measurement noise std-dev (m)
    Returns:
        X, P after the update
    """
    r = np.concatenate([X.act(y_i) - l for y_i, l in zip(y, landmarks)])
    H = measurement_jacobian_ri(landmarks)
    R = X.rotation()
    N = np.kron(np.eye(len(landmarks)), meas_noise_std ** 2 * R @ R.T)
    S = H @ P @ H.T + N
    K = P @ H.T @ np.linalg.inv(S)
    X = X.lplus(manif.SE_2_3Tangent(-K @ r))
    I_KH = np.eye(9) - K @ H
    return X, I_KH @ P @ I_KH.T + K @ N @ K.T


def step_hybrid_world(X, P, gyro, accel, dt, e, g_vec, gyro_noise_std, accel_noise_std,
                      detect_time_bias=0.0, detect_time_noise_std=0.0, detect_rng=None,
                      min_bounce_speed=0.5, max_bounces=5):
    """Ordinary error-state EKF predict (world-perturbation coordinates
    (dp, dtheta, dv)) with the saltation correction at each bounce. Its
    propagation Jacobian depends on the estimate:
        dp     += -(R 0.5 accel h^2)^ dtheta + h dv
        dv     += -(R accel h)^ dtheta
    Arguments:
        X, P: manif.SE_2_3 estimate, (9,9) covariance of (dp, dtheta, dv)
        gyro, accel: (3,) IMU readings
        dt: step (s)
        e: restitution coefficient
        g_vec: (3,) gravity vector
        gyro_noise_std, accel_noise_std: IMU noise std-devs per sample
        detect_time_bias, detect_time_noise_std, detect_rng, min_bounce_speed,
            max_bounces: as step_hybrid_ri
    Returns:
        X, P after dt
        n_bounces: number of bounces resolved in this step
    """
    remaining, n_bounces = dt, 0
    for _ in range(max_bounces + 1):
        tau = crossing_time(X, accel, g_vec)
        impact = tau is not None and tau < remaining
        h = (detected_crossing_time(tau, remaining, detect_time_bias, detect_time_noise_std, detect_rng)
             if impact else remaining)
        R = X.rotation()
        F = np.eye(9)
        F[VEL, ROT] = -skew(R @ (accel * h))
        F[POS, ROT] = -skew(R @ (0.5 * accel * h * h))
        F[POS, VEL] = h * np.eye(3)
        G = np.kron(np.eye(3), R)
        X = propagate_mean(X, gyro, accel, h, g_vec)
        P = F @ P @ F.T + G @ body_noise_covariance(h, gyro_noise_std, accel_noise_std) @ G.T
        remaining -= h
        if not impact:
            return X, P, n_bounces
        if abs(X.linearVelocity()[2]) < min_bounce_speed:
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
    H_i = [-R_hat^T, R_hat^T (l_i - p_hat)^, 0] -- depends on the estimate,
    unlike measurement_jacobian_ri.
    Arguments:
        X: manif.SE_2_3 estimate
        landmarks: (L,3) landmark positions
    Returns:
        H: (3L, 9)
    """
    R, p = X.rotation(), X.translation()
    H = np.zeros((3 * len(landmarks), 9))
    for i, l in enumerate(landmarks):
        H[3 * i:3 * i + 3, POS] = -R.T
        H[3 * i:3 * i + 3, ROT] = R.T @ skew(l - p)
    return H


def update_world(X, P, y, landmarks, meas_noise_std):
    """World-perturbation EKF update: y_hat_i = X_hat^-1.act(l_i), H from
    measurement_jacobian_world; correction p += dp, R <- Exp(dtheta) R,
    v += dv. Joseph form.
    Arguments:
        X, P: manif.SE_2_3 estimate, (9,9) covariance
        y: (L,3) measurements
        landmarks: (L,3) landmark positions
        meas_noise_std: measurement noise std-dev (m)
    Returns:
        X, P after the update
    """
    y_hat = measure_landmarks(X, landmarks)
    H = measurement_jacobian_world(X, landmarks)
    N = meas_noise_std ** 2 * np.eye(3 * len(landmarks))
    S = H @ P @ H.T + N
    K = P @ H.T @ np.linalg.inv(S)
    d = K @ (y - y_hat).reshape(-1)
    rot = manif.SO3Tangent(d[ROT]).exp() * rotation_of(X)
    X = se23_from(rot, X.linearVelocity() + d[VEL], X.translation() + d[POS])
    I_KH = np.eye(9) - K @ H
    return X, I_KH @ P @ I_KH.T + K @ N @ K.T


FILTERS = ("invariant_naive", "reset_jacobian", "saltation", "saltation_timing", "world_saltation")


def run_filter(name, X_init, P_init, gyro, accel, meas, dt, e, g_vec, landmarks,
               gyro_noise_std, accel_noise_std, meas_noise_std,
               detect_time_bias=0.0, detect_time_noise_std=0.0, detect_rng=None):
    """Runs one of FILTERS over the whole data set. "saltation_timing" assumes
    a detection-time error RMS of hypot(detect_time_bias,
    detect_time_noise_std), i.e. it is tuned to the true detector.
    Arguments:
        name: one of FILTERS
        X_init: manif.SE_2_3 initial estimate
        P_init: (9,9) initial covariance, in that filter's own error coordinates
        gyro, accel: (N,3) IMU readings
        meas: dict tick -> (L,3) landmark measurements
        dt, e, g_vec, landmarks, *_noise_std: as above
        detect_time_bias, detect_time_noise_std, detect_rng: contact-detection
            error, see detected_crossing_time
    Returns:
        X_list: list of manif.SE_2_3 estimates, P_list: list of (9,9) covariances
        bounce_count: (N,) cumulative number of bounces the filter has resolved, per tick
    """
    X, P = X_init, P_init.copy()
    X_list, P_list, counts = [X], [P], [0]
    for k in range(len(gyro)):
        if name == "world_saltation":
            X, P, nb = step_hybrid_world(X, P, gyro[k], accel[k], dt, e, g_vec,
                                         gyro_noise_std, accel_noise_std,
                                         detect_time_bias, detect_time_noise_std, detect_rng)
            if k + 1 in meas:
                X, P = update_world(X, P, meas[k + 1], landmarks, meas_noise_std)
        else:
            X, P, nb = step_hybrid_ri(X, P, gyro[k], accel[k], dt, e, g_vec,
                                      gyro_noise_std, accel_noise_std, reset_mode=name,
                                      detect_time_bias=detect_time_bias,
                                      detect_time_noise_std=detect_time_noise_std, detect_rng=detect_rng,
                                      timing_noise_std=np.hypot(detect_time_bias, detect_time_noise_std))
            if k + 1 in meas:
                X, P = update_ri(X, P, meas[k + 1], landmarks, meas_noise_std)
        X_list.append(X)
        P_list.append(P)
        counts.append(counts[-1] + nb)
    return X_list, P_list, np.array(counts)


# --- Metrics ------------------------------------------------------------------

def error_vector(name, X_true, X_est):
    """Estimation error in the named filter's own coordinates: xi =
    X_hat.lminus(X) = Log(X_hat X^-1) for the RIEKFs, (p - p_hat,
    Log(R R_hat^T), v - v_hat) for the world EKF.
    Arguments:
        name: one of FILTERS
        X_true, X_est: manif.SE_2_3
    Returns:
        err: (9,)
    """
    if name == "world_saltation":
        return np.concatenate([X_true.translation() - X_est.translation(),
                               rotation_of(X_true).lminus(rotation_of(X_est)).coeffs(),
                               X_true.linearVelocity() - X_est.linearVelocity()])
    return X_est.lminus(X_true).coeffs_copy()


def state_errors(X_true_list, X_est_list):
    """Rotation (deg), velocity (m/s) and position (m) error norms per tick.
    Arguments:
        X_true_list, X_est_list: lists of manif.SE_2_3
    Returns:
        rot_err, vel_err, pos_err: (N,) arrays
    """
    rot, vel, pos = [], [], []
    for Xt, Xe in zip(X_true_list, X_est_list):
        rot.append(np.degrees(np.linalg.norm(rotation_of(Xe).lminus(rotation_of(Xt)).coeffs())))
        vel.append(np.linalg.norm(Xe.linearVelocity() - Xt.linearVelocity()))
        pos.append(np.linalg.norm(Xe.translation() - Xt.translation()))
    return np.array(rot), np.array(vel), np.array(pos)


def initial_estimate(X_true0, P0_ri, rng):
    """Draws the shared initial estimate X_hat = X_true0.lplus(xi0), xi0 ~
    N(0, P0_ri) in manif's tangent order, and returns each filter's initial
    covariance in its own coordinates (world EKF: T(X_hat) P0_ri T(X_hat)^T).
    Arguments:
        X_true0: manif.SE_2_3 true initial state
        P0_ri: (9,9) initial covariance of the right-invariant error
        rng: numpy Generator
    Returns:
        X_init: manif.SE_2_3, P_init: dict name -> (9,9)
    """
    xi0 = np.linalg.cholesky(P0_ri) @ rng.normal(0.0, 1.0, 9)
    X_init = X_true0.lplus(manif.SE_2_3Tangent(xi0))
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
                                detect_time_bias=0.0, detect_time_noise_std=0.0):
    """Average NEES per tick for every filter over n_trials independent noise
    realizations of the same nominal true trajectory, scoring only
    mode-matched ticks (filter and truth have resolved the same number of
    bounces) -- see the use_numpy version's docstring for why. Detection
    jitter comes from one spawned child Generator per filter
    (child_detect_rngs), exactly as in the use_numpy version.
    Arguments:
        X_true_list: list of manif.SE_2_3 true states
        impact_ticks: first tick after each true impact
        gyro_true: (3,) true body rate
        dt, e, g_vec, landmarks, meas_every: as above
        P0_ri: (9,9) initial right-invariant covariance
        *_noise_std: noise std-devs
        n_trials: number of trials
        rng: numpy Generator
        detect_time_bias, detect_time_noise_std: contact-detection error
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
            meas[k] = (measure_landmarks(X_true_list[k], landmarks)
                       + rng.normal(0.0, meas_noise_std, landmarks.shape))
        X_init, P_init = initial_estimate(X_true_list[0], P0_ri, rng)
        for name in FILTERS:
            X_list, P_list, counts = run_filter(name, X_init, P_init[name], gyro, accel, meas, dt, e, g_vec,
                                                landmarks, gyro_noise_std, accel_noise_std, meas_noise_std,
                                                detect_time_bias, detect_time_noise_std, detect_rngs[name])
            for k in np.flatnonzero(counts == counts_true):
                sums[name][k] += nees(error_vector(name, X_true_list[k], X_list[k]), P_list[k])
                valid[name][k] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = {name: sums[name] / valid[name] for name in FILTERS}
    mismatch = {name: 1.0 - valid[name].sum() / (n_trials * (n_steps + 1)) for name in FILTERS}
    return avg, mismatch


def child_detect_rngs(rng):
    """One independent detection-jitter Generator per filter, spawned from
    rng's seed sequence (does not advance rng itself).
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
    "invariant_naive": "RIEKF, naive reset (M on xi_v)",
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
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    g_vec = np.array([0.0, 0.0, -args.gravity])
    landmarks = default_landmarks()
    body_rate = np.array(args.body_rate)
    X0 = se23_from(manif.SO3Tangent(np.array([0.2, -0.1, 0.3])).exp(), np.array(args.init_velocity),
                   np.array([0.0, 0.0, args.drop_height]))
    # manif tangent order: position, rotation, velocity.
    P0_ri = np.diag([args.init_pos_std ** 2] * 3 + [np.radians(args.init_rot_std_deg) ** 2] * 3
                    + [args.init_vel_std ** 2] * 3)

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
            n_steps=n_steps)
        results[name] = X_list
        perf[name] = (t_step, mem)

    print(f"Running Monte Carlo consistency check ({args.n_mc_trials} trials)...")
    avg_nees, mismatch = run_monte_carlo_consistency(
        X_true, impact_ticks, body_rate, args.dt, args.restitution, g_vec, landmarks, args.meas_every, P0_ri,
        args.gyro_noise_std, args.accel_noise_std, args.meas_noise_std, args.n_mc_trials, rng, *detect)

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
    ax_h.plot(t, [X.translation()[2] for X in X_true], "g--", linewidth=2, label="Ground truth")
    for name in FILTERS:
        ax_h.plot(t, [X.translation()[2] for X in results[name]], color=COLORS[name],
                  linestyle=LINESTYLES.get(name, "-"), label=LABELS[name])
    ax_h.set_ylabel("Height p_z (m)")
    title = "Right-invariant EKF on SE_2(3) through bounces (manif)"
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
