'''
Tracks a planar hexapod body from its legs alone: each foot on the ground
reports its velocity relative to the body (as joint encoders would), and the
gait decides which feet those are on every tick. State
`x = [px, py, theta, u, v, omega]`: world position, heading, body-frame
velocity (u forward, v left) and turn rate. Plain R^6 with an unwrapped
heading -- no SE(2) helpers needed.

Why planar and not 1D: in 1D every stance foot measures the same scalar
velocity, so six legs only average noise (variance ~ sigma^2 / n). In the
plane a foot at body-frame offset r_i sees -(b + omega x r_i), b = [u, v],
so feet at *different places* see different velocities whenever the body
turns. That is what lets the filter separate forward speed from turn rate,
and it makes the gait matter through both *how many* feet are down and
*where* they are.

Three gaits, each a known stance schedule with a constant number of feet
down:

  - tripod (duty factor 1/2): 3 feet down -- L1, R2, L3 alternate with R1, L2, R3.
  - ripple (duty factor 2/3): 4 feet down -- one leg per side in swing,
    back-to-front on each side, the two sides half a cycle apart.
  - wave (duty factor 5/6): 5 feet down -- one leg in swing at a time.

Every gait walks the *same* true body trajectory with the *same* noise draws
(one fixed noise stream per foot), so the measurement set is the only thing
that changes between gaits.

The truth twist is a random walk with exactly the noise the filter assumes
(Q matched, as in friction_anisotropic_ekf.py), and the pose is integrated
with the filter's own discrete model, so NEES is a fair consistency test.
No measurement touches px, py or theta: leg odometry alone never fixes
absolute position or heading, so those drift and their covariance grows.
That is expected and shown, not hidden.

Experiment 2 makes one leg (L2 by default) slip sideways for a time window.
A slipping stance foot breaks the "planted foot is not moving" model, so its
reading is biased. Two filters are compared: `gating=False` trusts every
stance foot; `gating=True` tests each foot's innovation against the
predicted state (chi-squared, 2 DoF, 99%) and drops the ones that fail
before the joint update.

Experiments 3 and 4 run on the tripod and wave gaits only:

  - Experiment 3, random foot drift (--foot-drift-std): every stance foot
    also slides at a small white random velocity. In this velocity-reading
    model that is extra reading noise, so the comparison is a filter that
    ignores it (R = sigma^2) vs. one that includes it (R = sigma^2 +
    sigma_drift^2).
  - Experiment 4, contact-detection jitter (--detect-time-noise-std): the
    filter uses detected touchdown/liftoff times, each off by its own
    Gaussian error, so it sometimes reads a foot that is still swinging.
    Three filters: trust the detected flags, the per-foot gate, or skip
    readings within 2 sigma of a detected touchdown/liftoff.

No saltation matrix: a touchdown changes which feet are measured, not the
body state (the legs are massless and the schedule is time-triggered), so
the reset map is the identity, the flow is the same on both sides and the
guard has no state gradient -- the saltation matrix is exactly I, and the
timing-noise term of invariant_saltation_ekf.py (s^2 w w^T, w = f+ - DR f-)
is exactly 0. Detection jitter shows up on the measurement side instead.

Simplifications: kinematics are perfect (the filter knows each foot's true
offset r_i; only the velocity reading is noisy), a swing foot moves in a
straight line between footholds with a smooth (1 - cos) speed profile, there
is no IMU, and except in experiment 4 the stance schedule is known. See
docs/filtering/hexapod_gait_ekf.md.
'''

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import measure_performance, averaged_nees_bounds, nees  # noqa: E402


LEG_NAMES = ["L1", "L2", "L3", "R1", "R2", "R3"]  # 1 = front, 3 = rear

# Nominal foot offsets from the body centre, body frame (m): x forward, y left.
NOMINAL_FOOT_OFFSETS = np.array([
    [0.20, 0.15], [0.00, 0.20], [-0.20, 0.15],
    [0.20, -0.15], [0.00, -0.20], [-0.20, -0.15],
])

# Per gait: duty factor (fraction of the cycle a foot is down) and, per leg in
# LEG_NAMES order, the fraction of the cycle at which its swing starts.
GAITS = {
    "tripod": (1.0 / 2.0, [0.5, 0.0, 0.5, 0.0, 0.5, 0.0]),
    "ripple": (2.0 / 3.0, [2.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 6.0, 5.0 / 6.0, 0.5]),
    "wave": (5.0 / 6.0, [5.0 / 6.0, 4.0 / 6.0, 3.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0, 0.0]),
}

# Planar cross product as a matrix: omega x r = omega * J @ r.
J = np.array([[0.0, -1.0], [1.0, 0.0]])


def rot(theta):
    """2D rotation matrix R(theta), body frame -> world frame."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def wrap_angle(a):
    """Wraps an angle (or array of angles) to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def gait_phase(n_ticks, ticks_per_cycle, gait):
    """Each leg's phase since its swing started, per tick, in [0, 1).
    Computed from integer ticks (rounded to 9 decimals) so stance/swing
    boundaries land exactly on ticks instead of flickering with float error.
    Arguments:
        n_ticks: number of ticks (including t=0)
        ticks_per_cycle: gait cycle length in ticks
        gait: "tripod", "ripple" or "wave"
    Returns:
        psi: (n_ticks, 6) array of phases
    """
    if gait not in GAITS:
        raise ValueError(f"unknown gait: {gait!r}")
    _, swing_start = GAITS[gait]
    s = np.arange(n_ticks)[:, None] / ticks_per_cycle
    return np.round((s - np.array(swing_start)[None, :]) % 1.0, 9) % 1.0


def stance_mask(n_ticks, ticks_per_cycle, gait):
    """True where a leg is in stance (foot planted) on that tick.
    A leg swings for the first (1 - duty) of its phase, then stands.
    Arguments:
        n_ticks: number of ticks (including t=0)
        ticks_per_cycle: gait cycle length in ticks
        gait: "tripod", "ripple" or "wave"
    Returns:
        (n_ticks, 6) bool array
    """
    psi = gait_phase(n_ticks, ticks_per_cycle, gait)
    duty, _ = GAITS[gait]
    return psi >= round(1.0 - duty, 9)


def simulate_body_trajectory(n_steps, dt, twist0, lin_accel_std, ang_accel_std, rng):
    """True body states: twist [u, v, omega] is a random walk with per-tick
    increments of std (lin_accel_std * dt, lin_accel_std * dt,
    ang_accel_std * dt) -- the exact noise `predict` assumes -- and the pose
    is integrated with `predict`'s own discrete model, starting at the origin.
    Arguments:
        n_steps: number of steps
        dt: step interval (s)
        twist0: initial [u, v, omega] (m/s, m/s, rad/s)
        lin_accel_std: std-dev of the linear-acceleration disturbance (m/s^2)
        ang_accel_std: std-dev of the angular-acceleration disturbance (rad/s^2)
        rng: numpy random number generator
    Returns:
        x_true: (n_steps+1, 6) array of true states
    """
    x_true = np.zeros((n_steps + 1, 6))
    x_true[0, 3:] = twist0
    w_std = np.array([lin_accel_std, lin_accel_std, ang_accel_std]) * dt
    w = rng.normal(0.0, 1.0, (n_steps, 3)) * w_std
    for k in range(n_steps):
        x_true[k + 1] = propagate(x_true[k], dt)
        x_true[k + 1, 3:] += w[k]
    return x_true


def simulate_feet(x_true, dt, ticks_per_cycle, gait, nominal_speed, slip=None, drift=None,
                  stance=None):
    """Footholds and true foot velocities relative to the body, per tick.
    A foot lands at its nominal offset shifted forward by half a stride, then
    stays fixed in the world for the rest of its stance (unless slipping or
    drifting), so its body-frame offset is r_i = R(theta)^T (f_i - p) and it
    moves in the body frame at -(b + omega x r_i). Feet already down at t=0
    are placed as if they had landed earlier at the nominal stride.
    Swing: the foot travels in a straight line from where it lifted off to
    its next foothold, with a smooth (1 - cos) profile, so its speed is zero
    at liftoff and at touchdown and peaks mid-swing. In the body frame it
    moves at R^T df/dt - (b + omega x r_i). Only a filter that wrongly treats
    a swinging foot as planted ever reads these values (detection jitter).
    A swing at the very start or end of the run, with no foothold on one
    side, is given a virtual one a stride away along the body's heading.
    Arguments:
        x_true: (N, 6) array of true states
        dt: step interval (s)
        ticks_per_cycle: gait cycle length in ticks
        gait: "tripod", "ripple" or "wave"
        nominal_speed: forward speed (m/s) used to place footholds: a foot
            lands half a stride ahead of its nominal offset, stride =
            nominal_speed x stance time, so it sweeps roughly symmetrically
            about the nominal offset
        slip: None, or (leg_index, t_start, t_end, slip_vel_body (2,)): while
            in stance and t_start <= t < t_end, that foot slides at
            slip_vel_body (body frame, m/s) instead of staying planted
        drift: None, or (N, 6, 2) body-frame velocities (m/s): every stance
            foot also slides by drift[k, i] on tick k (random foot drift)
        stance: None (default) for the fixed gait schedule, or an (N, 6) bool
            array of true stance flags to use instead (e.g. touchdowns that
            land early or late, gait_phase_ekf.py); footholds and swing paths
            follow it. Its first tick should match the schedule, since the
            initial foot placement assumes it.
    Returns:
        stance: (N, 6) bool array
        r: (N, 6, 2) body-frame foot offsets
        rel_vel: (N, 6, 2) true body-frame foot velocity relative to the body
    """
    n_pts = len(x_true)
    duty, _ = GAITS[gait]
    stance = stance_mask(n_pts, ticks_per_cycle, gait) if stance is None else stance
    psi = gait_phase(n_pts, ticks_per_cycle, gait)
    stride = nominal_speed * duty * ticks_per_cycle * dt
    touchdown = NOMINAL_FOOT_OFFSETS + np.array([stride / 2.0, 0.0])

    r = np.zeros((n_pts, 6, 2))
    rel_vel = np.zeros((n_pts, 6, 2))
    f_world = np.zeros((n_pts, 6, 2))  # foothold at each stance tick (before that tick's slide)
    footholds = np.zeros((6, 2))
    for k in range(n_pts):
        p, theta, b, omega = x_true[k, :2], x_true[k, 2], x_true[k, 3:5], x_true[k, 5]
        R = rot(theta)
        for i in range(6):
            if not stance[k, i]:
                continue
            if k == 0:
                since_td = (psi[0, i] - (1.0 - duty)) * ticks_per_cycle * dt
                r0 = touchdown[i] - np.array([nominal_speed * since_td, 0.0])
                footholds[i] = p + R @ r0
            elif not stance[k - 1, i]:
                footholds[i] = p + R @ touchdown[i]
            f_world[k, i] = footholds[i]
            r[k, i] = R.T @ (footholds[i] - p)
            rel_vel[k, i] = -(b + omega * J @ r[k, i])
            if slip is not None and slip[0] == i and slip[1] <= k * dt < slip[2]:
                rel_vel[k, i] += slip[3]
                footholds[i] = footholds[i] + R @ slip[3] * dt
            if drift is not None:
                rel_vel[k, i] += drift[k, i]
                footholds[i] = footholds[i] + R @ drift[k, i] * dt

    swing_ticks = int(round((1.0 - duty) * ticks_per_cycle))
    for i in range(6):
        k = 0
        while k < n_pts:
            if stance[k, i]:
                k += 1
                continue
            c = k
            while c < n_pts and not stance[c, i]:
                c += 1
            a = k - 1  # last stance tick before this swing (-1 if none)
            if a >= 0:
                start = f_world[a, i] + rot(x_true[a, 2]) @ (rel_vel[a, i] + x_true[a, 3:5]
                                                            + x_true[a, 5] * J @ r[a, i]) * dt
            if c < n_pts:
                end = f_world[c, i]
            if a < 0:
                a = c - swing_ticks - 1
                start = end - rot(x_true[c, 2]) @ np.array([stride, 0.0])
            if c >= n_pts:
                c = a + swing_ticks + 1
                end = start + rot(x_true[-1, 2]) @ np.array([stride, 0.0])
            span = (c - a) * dt
            for j in range(k, min(c, n_pts)):
                s = (j - a) / (c - a)
                f = start + (end - start) * 0.5 * (1.0 - np.cos(np.pi * s))
                f_dot = (end - start) * 0.5 * np.pi * np.sin(np.pi * s) / span
                p, theta, b, omega = x_true[j, :2], x_true[j, 2], x_true[j, 3:5], x_true[j, 5]
                R = rot(theta)
                r[j, i] = R.T @ (f - p)
                rel_vel[j, i] = R.T @ f_dot - (b + omega * J @ r[j, i])
            k = c
    return stance, r, rel_vel


def detected_stance(stance, dt, detect_noise_std, event_noise):
    """Stance flags as a contact detector would report them: every touchdown
    and liftoff is shifted by its own timing error, N(0, detect_noise_std^2)
    rounded to whole ticks. The run's first and last ticks are not events.
    Arguments:
        stance: (N, 6) true stance flags
        dt: step interval (s)
        detect_noise_std: std-dev of each event's timing error (s)
        event_noise: (6, M) standard-normal draws, consumed in order per leg
            (one per event); shared across gaits for a controlled comparison
    Returns:
        (N, 6) bool detected stance flags
    """
    n_pts = len(stance)
    out = np.zeros_like(stance)
    for i in range(6):
        edges = np.diff(np.concatenate([[False], stance[:, i], [False]]).astype(int))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        e = 0
        for s, t in zip(starts, ends):
            if s > 0:
                s = s + int(round(event_noise[i, e] * detect_noise_std / dt))
                e += 1
            if t < n_pts:
                t = t + int(round(event_noise[i, e] * detect_noise_std / dt))
                e += 1
            out[max(s, 0):min(t, n_pts), i] = True
    return out


def trim_transitions(mask, n_ticks):
    """Drops readings within n_ticks of any switch in `mask` (per leg): the
    filter skips the ticks right after a detected touchdown and right before
    a detected liftoff, where a timing error would make it read a moving foot."""
    out = mask.copy()
    n_pts = len(mask)
    for i in range(mask.shape[1]):
        edges = np.flatnonzero(np.diff(mask[:, i].astype(int)) != 0) + 1  # first tick after a switch
        for e in edges:
            out[max(e - n_ticks, 0):min(e + n_ticks, n_pts), i] = False
    return out


def propagate(x, dt):
    """Noise-free discrete motion model shared by truth and filter:
    p += R(theta) b dt, theta += omega dt, twist constant."""
    x_new = x.copy()
    x_new[:2] = x[:2] + rot(x[2]) @ x[3:5] * dt
    x_new[2] = x[2] + x[5] * dt
    return x_new


def predict_jacobian(x, dt):
    """Jacobian of `propagate` w.r.t. the state.
    dp'/dtheta = J R(theta) b dt, dp'/db = R(theta) dt, dtheta'/domega = dt."""
    F = np.eye(6)
    R = rot(x[2])
    F[:2, 2] = J @ R @ x[3:5] * dt
    F[:2, 3:5] = R * dt
    F[2, 5] = dt
    return F


def process_noise_covariance(dt, lin_accel_std, ang_accel_std):
    """Q: random-walk noise on the twist only, matched to
    simulate_body_trajectory's per-tick increments."""
    return np.diag([0.0, 0.0, 0.0,
                    (lin_accel_std * dt) ** 2, (lin_accel_std * dt) ** 2, (ang_accel_std * dt) ** 2])


def predict(x, P, dt, Q):
    """EKF predict: propagate the mean, linearize for the covariance.
    Arguments:
        x: state (6,)
        P: (6,6) covariance
        dt: step interval (s)
        Q: (6,6) process noise covariance
    Returns:
        x_new, P_new
    """
    F = predict_jacobian(x, dt)
    return propagate(x, dt), F @ P @ F.T + Q


def foot_measurement_jacobian(r_i):
    """H_i for one stance foot: h_i(x) = -(b + omega J r_i), linear in the
    state. Only the twist columns are nonzero -- no foot reading touches the
    pose, which is why px, py, theta are unobservable here."""
    H = np.zeros((2, 6))
    H[:, 3:5] = -np.eye(2)
    H[:, 5] = -J @ r_i
    return H


def measurement_update_feet(x, P, z_feet, r_feet, R_meas, gating, gate=9.21):
    """Joint update with every stance foot's relative-velocity reading.
    With gating, each foot is first tested on its own against the predicted
    state (NIS = nu^T S^-1 nu vs. the chi-squared 2-DoF 99% value 9.21), and
    only the feet that pass enter the joint update -- testing against the
    prediction keeps the result independent of foot order.
    Arguments:
        x: predicted state (6,)
        P: predicted covariance (6,6)
        z_feet: (m, 2) readings of the m stance feet
        r_feet: (m, 2) body-frame offsets of those feet
        R_meas: per-axis measurement noise variance (scalar)
        gating: whether to drop feet that fail the per-foot test
        gate: NIS threshold
    Returns:
        x_new, P_new: updated state and covariance
        accepted: (m,) bool array, which feet were used
    """
    m = len(z_feet)
    accepted = np.ones(m, dtype=bool)
    if m == 0:
        return x, P, accepted
    Hs = [foot_measurement_jacobian(r_feet[j]) for j in range(m)]
    if gating:
        R2 = R_meas * np.eye(2)
        for j in range(m):
            nu = z_feet[j] - Hs[j] @ x
            S = Hs[j] @ P @ Hs[j].T + R2
            accepted[j] = nu @ np.linalg.solve(S, nu) <= gate
        if not accepted.any():
            return x, P, accepted
    H = np.vstack([Hs[j] for j in range(m) if accepted[j]])
    z = z_feet[accepted].reshape(-1)
    S = H @ P @ H.T + R_meas * np.eye(len(z))
    K = np.linalg.solve(S, H @ P).T
    x_new = x + K @ (z - H @ x)
    I_KH = np.eye(6) - K @ H
    P_new = I_KH @ P @ I_KH.T + R_meas * K @ K.T  # Joseph form
    return x_new, P_new, accepted


def run_ekf(x_init, P_init, z, r, stance, dt, Q, R_meas, gating=False):
    """Recursive EKF: predict, then one joint update with the feet in stance.
    Arguments:
        x_init: initial state estimate (6,)
        P_init: initial covariance (6,6)
        z: (N, 6, 2) foot readings (only stance entries are read)
        r: (N, 6, 2) body-frame foot offsets
        stance: (N, 6) bool stance schedule
        dt: step interval (s)
        Q: (6,6) process noise covariance
        R_meas: per-axis measurement noise variance (scalar)
        gating: per-foot chi-squared gating on/off
    Returns:
        x_list: (N, 6) estimates
        P_list: list of (6,6) covariances
        accepted: (N, 6) bool, True where a stance foot's reading was used
    """
    n_pts = len(stance)
    x, P = x_init.copy(), P_init.copy()
    x_list, P_list = [x], [P]
    accepted = np.zeros((n_pts, 6), dtype=bool)
    for k in range(1, n_pts):
        x, P = predict(x, P, dt, Q)
        idx = np.flatnonzero(stance[k])
        x, P, acc = measurement_update_feet(x, P, z[k, idx], r[k, idx], R_meas, gating)
        accepted[k, idx] = acc
        x_list.append(x)
        P_list.append(P)
    return np.array(x_list), P_list, accepted


def state_errors(x_true, x_est):
    """Error x_true - x_est with the heading wrapped. (N, 6)."""
    err = x_true - x_est
    err[:, 2] = wrap_angle(err[:, 2])
    return err


def twist_geometry_std(r, stance, meas_noise_std):
    """How well the stance feet alone pin down the twist, per tick: the std
    of u, v and omega from one tick's readings with no prior,
    sqrt(diag((H_b^T H_b)^-1)) * sigma, H_b = the stacked twist columns.
    Depends on how many feet are down and where they are.
    Returns:
        (N, 3) array (NaN on ticks with fewer than 2 feet down)
    """
    out = np.full((len(stance), 3), np.nan)
    for k in range(len(stance)):
        idx = np.flatnonzero(stance[k])
        if len(idx) < 2:
            continue
        Hb = np.vstack([foot_measurement_jacobian(r[k, i])[:, 3:] for i in idx])
        out[k] = np.sqrt(np.diag(np.linalg.inv(Hb.T @ Hb))) * meas_noise_std
    return out


def simulate_trial(n_steps, dt, ticks_per_cycle, gaits, twist0, lin_accel_std, ang_accel_std,
                   meas_noise_std, init_std, nominal_speed, rng, slip=None, foot_drift_std=0.0):
    """One trial: one true body trajectory, one set of per-foot noise draws
    and one initial-guess error, shared by every gait in `gaits` -- so
    gaits differ only in which feet are down and where. With
    foot_drift_std > 0, every stance foot also drifts at a white body-frame
    velocity of that std (drawn after the other draws, so the rest of the
    trial is unchanged).
    Returns:
        x_true: (N, 6)
        x_init: (6,)
        feet: dict gait -> (stance, r, z)
    """
    x_true = simulate_body_trajectory(n_steps, dt, twist0, lin_accel_std, ang_accel_std, rng)
    noise = rng.normal(0.0, meas_noise_std, (n_steps + 1, 6, 2))
    x_init = x_true[0] + rng.normal(0.0, 1.0, 6) * init_std
    drift = rng.normal(0.0, foot_drift_std, (n_steps + 1, 6, 2)) if foot_drift_std > 0 else None
    feet = {}
    for gait in gaits:
        stance, r, rel_vel = simulate_feet(x_true, dt, ticks_per_cycle, gait, nominal_speed, slip, drift)
        feet[gait] = (stance, r, rel_vel + noise)
    return x_true, x_init, feet


def run_monte_carlo(n_trials, n_steps, dt, ticks_per_cycle, gaits, twist0, lin_accel_std,
                    ang_accel_std, meas_noise_std, init_std, nominal_speed, rng, slip=None,
                    gating_modes=(False,), foot_drift_std=0.0, R_meas_filter=None,
                    detect_noise_std=0.0, trim_ticks=0):
    """Monte Carlo over fresh truths and noise. For each (gait, gating) pair:
    per-tick average NEES (full 6-DoF and twist-only 3-DoF), per-tick mean
    squared error (N, 6), and acceptance counts of the slipping foot's
    readings (slip ticks), of every other stance reading, and of readings
    taken from a foot that was really in swing (only possible with
    detection jitter).
    Extensions (all off by default):
        foot_drift_std: random drift of every stance foot (see simulate_trial)
        R_meas_filter: the filter's per-axis reading variance; default
            meas_noise_std^2 (i.e. a filter that ignores any drift)
        detect_noise_std: contact-detection timing jitter (s); the filter then
            uses detected_stance flags instead of the true schedule. Event
            draws come after the trial's other draws.
        trim_ticks: skip readings within this many ticks of a detected
            touchdown or liftoff (trim_transitions)
    Returns:
        dict (gait, gating) -> dict with keys nees_full, nees_twist, mse,
            slip_used, slip_total, other_used, other_total, swing_used
    """
    Q = process_noise_covariance(dt, lin_accel_std, ang_accel_std)
    R_meas = meas_noise_std ** 2 if R_meas_filter is None else R_meas_filter
    P_init = np.diag(init_std ** 2)
    n_pts = n_steps + 1
    keys = [(g, gm) for g in gaits for gm in gating_modes]
    acc = {key: dict(nees_full=np.zeros(n_pts), nees_twist=np.zeros(n_pts), mse=np.zeros((n_pts, 6)),
                     slip_used=0, slip_total=0, other_used=0, other_total=0, swing_used=0) for key in keys}
    t = np.arange(n_pts) * dt
    for _ in range(n_trials):
        x_true, x_init, feet = simulate_trial(n_steps, dt, ticks_per_cycle, gaits, twist0,
                                              lin_accel_std, ang_accel_std, meas_noise_std,
                                              init_std, nominal_speed, rng, slip, foot_drift_std)
        if detect_noise_std > 0:
            event_noise = rng.normal(0.0, 1.0, (6, 2 * (n_pts // ticks_per_cycle + 2)))
        for gait, gating in keys:
            stance, r, z = feet[gait]
            used = detected_stance(stance, dt, detect_noise_std, event_noise) if detect_noise_std > 0 else stance
            if trim_ticks > 0:
                used = trim_transitions(used, trim_ticks)
            x_est, P_list, accepted = run_ekf(x_init, P_init, z, r, used, dt, Q, R_meas, gating)
            err = state_errors(x_true, x_est)
            a = acc[(gait, gating)]
            a["mse"] += err ** 2
            for k in range(n_pts):
                a["nees_full"][k] += nees(err[k], P_list[k])
                a["nees_twist"][k] += nees(err[k, 3:], P_list[k][3:, 3:])
            slipping = np.zeros_like(stance)
            if slip is not None:
                slipping[:, slip[0]] = stance[:, slip[0]] & (t >= slip[1]) & (t < slip[2])
            others = stance & ~slipping
            others[0] = False  # no update at t=0
            a["slip_used"] += int(accepted[slipping].sum())
            a["slip_total"] += int(slipping.sum())
            a["other_used"] += int(accepted[others].sum())
            a["other_total"] += int(others.sum())
            a["swing_used"] += int(accepted[~stance].sum())
    for a in acc.values():
        a["nees_full"] /= n_trials
        a["nees_twist"] /= n_trials
        a["mse"] /= n_trials
    return acc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--duration", type=float, default=12.0, help="Simulation length in seconds")
    parser.add_argument("--dt", type=float, default=0.02, help="Filter/measurement step interval in seconds")
    parser.add_argument("--cycle-period", type=float, default=1.2,
                         help="Gait cycle period (s); must be a multiple of 6*dt so every gait's "
                              "stance/swing boundaries land on ticks")
    parser.add_argument("--forward-speed", type=float, default=0.1, help="Initial forward speed u (m/s)")
    parser.add_argument("--turn-rate", type=float, default=0.2, help="Initial turn rate omega (rad/s)")
    parser.add_argument("--lin-accel-std", type=float, default=0.05,
                         help="Linear-acceleration disturbance std-dev (m/s^2), truth and filter")
    parser.add_argument("--ang-accel-std", type=float, default=0.1,
                         help="Angular-acceleration disturbance std-dev (rad/s^2), truth and filter")
    parser.add_argument("--meas-noise-std", type=float, default=0.02,
                         help="Per-axis std-dev of each foot's relative-velocity reading (m/s)")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.01, help="Initial position std-dev (m)")
    parser.add_argument("--init-heading-noise-std", type=float, default=0.01, help="Initial heading std-dev (rad)")
    parser.add_argument("--init-twist-noise-std", type=float, default=0.05,
                         help="Initial u, v std-dev (m/s); omega uses the same number in rad/s")

    parser.add_argument("--slip-leg", type=str, default="L2", choices=LEG_NAMES, help="Leg that slips in experiment 2")
    parser.add_argument("--slip-start", type=float, default=4.0, help="Slip window start (s)")
    parser.add_argument("--slip-end", type=float, default=8.0, help="Slip window end (s)")
    parser.add_argument("--slip-speed", type=float, default=0.1,
                         help="Sideways (outward) slip speed of the slipping foot (m/s)")

    parser.add_argument("--foot-drift-std", type=float, default=0.02,
                         help="Experiment 3: std-dev of every stance foot's random drift velocity (m/s)")
    parser.add_argument("--detect-time-noise-std", type=float, default=0.05,
                         help="Experiment 4: std-dev of each touchdown/liftoff detection time error (s)")
    parser.add_argument("--n-mc-trials", type=int, default=100, help="Number of Monte Carlo trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")

    args = parser.parse_args()

    ticks_per_cycle = int(round(args.cycle_period / args.dt))
    if abs(ticks_per_cycle * args.dt - args.cycle_period) > 1e-9 or ticks_per_cycle % 6 != 0:
        raise ValueError("--cycle-period must be a multiple of 6*dt")
    n_steps = int(round(args.duration / args.dt))
    gaits = list(GAITS)
    twist0 = np.array([args.forward_speed, 0.0, args.turn_rate])
    init_std = np.array([args.init_pos_noise_std, args.init_pos_noise_std, args.init_heading_noise_std,
                         args.init_twist_noise_std, args.init_twist_noise_std, args.init_twist_noise_std])
    slip_leg = LEG_NAMES.index(args.slip_leg)
    side = 1.0 if args.slip_leg.startswith("L") else -1.0
    slip = (slip_leg, args.slip_start, args.slip_end, np.array([0.0, side * args.slip_speed]))
    Q = process_noise_covariance(args.dt, args.lin_accel_std, args.ang_accel_std)
    R_meas = args.meas_noise_std ** 2
    P_init = np.diag(init_std ** 2)
    sim_args = (n_steps, args.dt, ticks_per_cycle, gaits, twist0, args.lin_accel_std, args.ang_accel_std,
                args.meas_noise_std, init_std, args.forward_speed)

    # --- single run, no slip ---
    rng = np.random.default_rng(args.seed)
    x_true, x_init, feet = simulate_trial(*sim_args, rng)
    single = {}
    print("Single run (no slip):")
    for gait in gaits:
        stance, r, z = feet[gait]
        (x_est, _, _), time_s, mem = measure_performance(run_ekf, x_init, P_init, z, r, stance, args.dt, Q,
                                                         R_meas, n_steps=n_steps)
        single[gait] = x_est
        err = state_errors(x_true, x_est)
        print(f"  {gait:<7s} feet down={stance[1:].sum(axis=1).mean():.1f} | "
              f"RMS u={np.sqrt(np.mean(err[1:, 3] ** 2)):.4f} m/s, "
              f"omega={np.sqrt(np.mean(err[1:, 5] ** 2)):.4f} rad/s | "
              f"final pos err={np.linalg.norm(err[-1, :2]):.4f} m, heading err={abs(err[-1, 2]):.4f} rad | "
              f"{time_s * 1e6:.1f} us/step, {mem / 1024.0:.1f} KB")

    print(f"\nStance geometry: std of u, v, omega from one tick's readings alone "
          f"(no prior; meas std {args.meas_noise_std} m/s), mean over ticks:")
    for gait in gaits:
        stance, r, _ = feet[gait]
        g = np.nanmean(twist_geometry_std(r[1:], stance[1:], args.meas_noise_std), axis=0)
        n_down = stance[1:].sum(axis=1).mean()
        print(f"  {gait:<7s} u={g[0]:.4f} m/s, v={g[1]:.4f} m/s, omega={g[2]:.4f} rad/s "
              f"(sigma/sqrt(n) = {args.meas_noise_std / np.sqrt(n_down):.4f})")
    print("  Same count, different places (3 feet at nominal offsets, no gait):")
    for label, legs in (("tripod triangle L1, R2, L3", [0, 4, 2]), ("one side L1, L2, L3", [0, 1, 2])):
        n_legs = len(legs)
        g = twist_geometry_std(NOMINAL_FOOT_OFFSETS[legs][None], np.ones((1, n_legs), dtype=bool),
                               args.meas_noise_std)[0]
        print(f"    {label:<27s} u={g[0]:.4f} m/s, v={g[1]:.4f} m/s, omega={g[2]:.4f} rad/s")

    # --- Monte Carlo, no slip ---
    print(f"\nMonte Carlo, no slip ({args.n_mc_trials} trials)...")
    mc = run_monte_carlo(args.n_mc_trials, *sim_args, np.random.default_rng(args.seed + 1))
    lo6, hi6 = averaged_nees_bounds(6, args.n_mc_trials)
    lo3, hi3 = averaged_nees_bounds(3, args.n_mc_trials)
    print(f"  NEES: full expects 6 [{lo6:.2f}, {hi6:.2f}], twist-only expects 3 [{lo3:.2f}, {hi3:.2f}]")
    for gait in gaits:
        a = mc[(gait, False)]
        rmse = np.sqrt(a["mse"][1:].mean(axis=0))
        print(f"  {gait:<7s} NEES full={a['nees_full'][1:].mean():.2f}, twist={a['nees_twist'][1:].mean():.2f} | "
              f"RMSE u={rmse[3]:.4f}, v={rmse[4]:.4f} m/s, omega={rmse[5]:.4f} rad/s | "
              f"final RMSE pos={np.sqrt(a['mse'][-1, :2].sum()):.4f} m, "
              f"heading={np.sqrt(a['mse'][-1, 2]):.4f} rad")

    # --- Monte Carlo, slip ---
    print(f"\nMonte Carlo, {args.slip_leg} slipping sideways at {args.slip_speed} m/s "
          f"for t in [{args.slip_start}, {args.slip_end}) s ({args.n_mc_trials} trials)...")
    mc_slip = run_monte_carlo(args.n_mc_trials, *sim_args, np.random.default_rng(args.seed + 2), slip=slip,
                              gating_modes=(False, True))
    t_hist = np.arange(n_steps + 1) * args.dt
    win = (t_hist >= args.slip_start) & (t_hist < args.slip_end)
    for gait in gaits:
        for gating in (False, True):
            a = mc_slip[(gait, gating)]
            rmse_win = np.sqrt(a["mse"][win].mean(axis=0))
            label = "gated" if gating else "naive"
            print(f"  {gait:<7s} {label}: in window RMSE v={rmse_win[4]:.4f} m/s, omega={rmse_win[5]:.4f} rad/s, "
                  f"twist NEES={a['nees_twist'][win].mean():.2f} | "
                  f"final RMSE pos={np.sqrt(a['mse'][-1, :2].sum()):.4f} m | "
                  f"slip readings used {a['slip_used'] / max(a['slip_total'], 1):.1%}, "
                  f"good readings used {a['other_used'] / a['other_total']:.1%}")

    # --- Monte Carlo, random foot drift (tripod and wave) ---
    ext_gaits = ["tripod", "wave"]
    ext_args = (n_steps, args.dt, ticks_per_cycle, ext_gaits) + sim_args[4:]
    print(f"\nMonte Carlo, every stance foot drifting at a random {args.foot_drift_std} m/s (white, per axis) "
          f"({args.n_mc_trials} trials, same draws for both filters)...")
    drift_var = args.meas_noise_std ** 2 + args.foot_drift_std ** 2
    for label, R_filter in (("ignores drift", None), ("models drift", drift_var)):
        mc_d = run_monte_carlo(args.n_mc_trials, *ext_args, np.random.default_rng(args.seed + 3),
                               foot_drift_std=args.foot_drift_std, R_meas_filter=R_filter)
        for gait in ext_gaits:
            a = mc_d[(gait, False)]
            rmse = np.sqrt(a["mse"][1:].mean(axis=0))
            print(f"  {gait:<7s} {label}: NEES full={a['nees_full'][1:].mean():.2f}, twist={a['nees_twist'][1:].mean():.2f} | "
                  f"RMSE u={rmse[3]:.4f} m/s, omega={rmse[5]:.4f} rad/s | "
                  f"final RMSE pos={np.sqrt(a['mse'][-1, :2].sum()):.4f} m")

    # --- Monte Carlo, contact-detection jitter (tripod and wave) ---
    trim = int(np.ceil(2.0 * args.detect_time_noise_std / args.dt))
    print(f"\nMonte Carlo, touchdown/liftoff detected with {args.detect_time_noise_std} s timing jitter "
          f"({args.n_mc_trials} trials, same draws for all three filters; trim = {trim} ticks)...")
    mc_j = run_monte_carlo(args.n_mc_trials, *ext_args, np.random.default_rng(args.seed + 4),
                           detect_noise_std=args.detect_time_noise_std, gating_modes=(False, True))
    mc_t = run_monte_carlo(args.n_mc_trials, *ext_args, np.random.default_rng(args.seed + 4),
                           detect_noise_std=args.detect_time_noise_std, trim_ticks=trim)
    for gait in ext_gaits:
        for label, a in (("trust flags", mc_j[(gait, False)]), ("per-foot gate", mc_j[(gait, True)]),
                         ("trim edges", mc_t[(gait, False)])):
            rmse = np.sqrt(a["mse"][1:].mean(axis=0))
            print(f"  {gait:<7s} {label:<13s}: NEES full={a['nees_full'][1:].mean():.2f}, "
                  f"twist={a['nees_twist'][1:].mean():.2f} | RMSE u={rmse[3]:.4f} m/s, omega={rmse[5]:.4f} rad/s | "
                  f"final RMSE pos={np.sqrt(a['mse'][-1, :2].sum()):.4f} m | "
                  f"swing-foot readings used {a['swing_used'] / args.n_mc_trials:.1f} per run")

    colors = {"tripod": "tab:red", "ripple": "tab:orange", "wave": "tab:blue"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.plot(x_true[:, 0], x_true[:, 1], color="green", linestyle="--", linewidth=2, label="Ground truth")
    for gait in gaits:
        ax.plot(single[gait][:, 0], single[gait][:, 1], color=colors[gait], label=gait)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Single run: body path from leg odometry")
    ax.legend()

    ax = axes[0, 1]
    for gait in gaits:
        ax.plot(t_hist, np.sqrt(mc[(gait, False)]["mse"][:, 5]), color=colors[gait], label=gait)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Turn-rate RMSE (rad/s)")
    ax.set_title(f"Monte Carlo ({args.n_mc_trials} trials), no slip")
    ax.legend()

    ax = axes[1, 0]
    for gait in gaits:
        ax.plot(t_hist[1:], mc[(gait, False)]["nees_twist"][1:], color=colors[gait], label=gait)
    ax.axhline(3.0, color="black", linewidth=1, label="Expected (3 DoF)")
    ax.axhline(lo3, color="black", linestyle="--", linewidth=1, label="95% bounds")
    ax.axhline(hi3, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Twist NEES")
    ax.set_title("Twist consistency, no slip")
    ax.legend()

    ax = axes[1, 1]
    for gait in ("tripod", "wave"):
        for gating, ls in ((False, "-"), (True, "--")):
            ax.plot(t_hist, np.sqrt(mc_slip[(gait, gating)]["mse"][:, 4]), color=colors[gait], linestyle=ls,
                    label=f"{gait}, {'gated' if gating else 'naive'}")
    ax.axvspan(args.slip_start, args.slip_end, color="gray", alpha=0.2, label=f"{args.slip_leg} slipping")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sideways velocity v RMSE (m/s)")
    ax.set_title("One slipping leg: naive vs. per-foot gating")
    ax.legend()

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
