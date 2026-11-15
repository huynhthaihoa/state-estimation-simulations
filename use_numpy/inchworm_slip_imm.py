'''
Follow-up of inchworm_slip_gate_ekf.py (1D point mass, state `x = [p, v]`,
anchor phases that may slip, a pad displacement sensor `d_k = dt * v_k + noise`):
instead of deciding "is the body still?" with a hard rule (schedule, threshold
gate) or fusing the pad reading as a velocity, keep a probability for each of
two motion modes and blend two mode-matched Kalman filters with an interacting
multiple model (IMM) filter (Blom & Bar-Shalom 1988, IEEE TAC 33(8), 780-783).
Truth, noise and defaults are the parent's (same random-number order, so seed 0
has the same two slip pulses).

Modes (both filters see the same measurements, so their likelihoods compare):
  - still (S): v' = 0 + w, w ~ N(0, zupt_noise^2), p' = p. The ZUPT belief
    "v = 0" is the dynamics instead of a pseudo-measurement.
  - moving (M): the parent's constant-velocity predict, acceleration noise 0.5 m/s^2.
Measurements per tick: position z (R = pos_noise^2) and, for the variants that
use the pad, d / dt as a velocity (R = (sensor_noise / dt)^2), stacked.

IMM cycle (run_imm): mixing -> mode-matched predict and update -> mode
probability update from the innovation likelihoods -> moment-matched combined
estimate (with the spread-of-means term). Equations (12)-(19) of the paper;
the paper's transition matrix H is column-stochastic, the code's is its
transpose (row = previous mode).

Mode transition matrices are filter design choices derived from the
simulation's own parameters (design_transitions, printed at run time):
  - imm / imm_pos: one constant matrix from the mean sojourn times.
  - imm_schedule / imm_schedule_pos: the matrix follows the known schedule
    (scheduled extend: strongly moving; scheduled anchor: slip start/end
    rates; first anchor tick after an extend phase: exactly still).
  - imm_schedule_next: a labeled variant that uses only the next tick's
    scheduled phase. It leaves the moving mode lingering after each extend
    phase, which looks like slip detection in seed 0 (both pulses start within
    0.2 s of the extend phase) but is leftover motion; see the doc.

Regimes: A = pad available (the parent's five policies + imm + imm_schedule),
B = position only (never, schedule, imm_pos, imm_schedule_pos), C = pad noise
sweep (schedule_and_sensor, sensor_fused, imm_schedule). Scores: single-run
RMSE, Monte Carlo averaged NEES (2 DoF) in the windows overall / clean anchor /
slip / cruise against the 95% bound, slip detection of P(M) > 0.5 (fraction of
slip ticks, false alarms on clean-anchor ticks, delay), time per step. All
variants share the same data in every trial. Use --seeds-table to repeat A and
B over several seeds; --p-slip 0 and --still-model integrate are the ablations.

Findings at the defaults (seed 0, 500 trials; numbers in the doc
docs/filtering/inchworm_slip_imm.md, written after the runs):
  - With the pad reading, imm and imm_schedule match sensor_fused's NEES in the
    slip window (1.83 / 1.94 vs 1.95) and lower its Monte Carlo velocity RMS
    error by about 13% (imm_schedule 0.0081 vs 0.0093 m/s), all of it in the
    clean-anchor window (0.0066 vs 0.0097; the schedule's plain ZUPT reaches
    0.0040 there); imm_schedule has no false alarms on clean anchors.
  - Without the pad, position noise (20 mm) hides a 4 mm per tick slip:
    imm_schedule_pos does not detect it (7.9% of slip ticks) and its slip NEES
    is 9.3; imm_pos has P(M) > 0.5 on 99% of clean-anchor ticks, so it does not tell still from moving.
  - As the pad gets noisier, sensor_fused stays consistent in the slip window
    while imm_schedule's slip NEES rises (4.27 at 2 mm, 8.26 at 5 mm).
'''

import argparse
import copy
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

from state_estimation_simulations.utils import measure_performance, averaged_nees_bounds
from state_estimation_simulations.use_numpy.inchworm_zupt_ekf import predict, state_errors, _kf_update  # noqa: E402
from state_estimation_simulations.use_numpy.inchworm_slip_gate_ekf import (  # noqa: E402
    GATING_MODES, generate_ground_truth_and_data, run_ekf, draw_slip_episodes)

STATE_DIM = 2
STILL, MOVING = 0, 1  # mode indices; P(M) is mu[..., MOVING]
WINDOWS = ("overall", "clean anchor", "slip", "cruise")


# --- measurement update with a likelihood -----------------------------------------------------------

def kf_update_loglik(x, P, z, H, R):
    """Linear-Gaussian update that also returns the log of the Gaussian
    likelihood of the innovation, N(z - H x; 0, H P H^T + R). The state and
    covariance come from inchworm_zupt_ekf._kf_update (so they are identical
    to its output); this adds only the likelihood. Works on a leading batch
    shape: x (..., n), P (..., n, n), z (..., m) broadcast against each other.
    Arguments:
        x: predicted state (..., n)
        P: predicted covariance (..., n, n)
        z: measurement (..., m)
        H: (m, n) measurement matrix
        R: (m, m) measurement noise covariance
    Returns:
        x_new: updated state (..., n)
        P_new: updated covariance (..., n, n)
        log_lik: natural log of the innovation likelihood (...)
    """
    xc, zc = x[..., None], z[..., None]
    r = zc - H @ xc
    S = H @ P @ H.T + R
    x_new, P_new = _kf_update(xc, P, zc, H, R)
    _, logdet = np.linalg.slogdet(S)
    maha = (np.swapaxes(r, -1, -2) @ np.linalg.solve(S, r))[..., 0, 0]
    return x_new[..., 0], P_new, -0.5 * (maha + logdet + z.shape[-1] * np.log(2.0 * np.pi))


def kf_update_with_likelihood(x, P, z, H, R):
    """Same as kf_update_loglik but returns the likelihood itself (the IMM
    uses the log form so that a very surprising measurement does not
    underflow to 0).
    Arguments:
        x, P, z, H, R: as in kf_update_loglik
    Returns:
        x_new: updated state (..., n)
        P_new: updated covariance (..., n, n)
        likelihood: Gaussian density of the innovation (...)
    """
    x_new, P_new, log_lik = kf_update_loglik(x, P, z, H, R)
    return x_new, P_new, np.exp(log_lik)


# --- mode transition matrices (filter design choices) ----------------------------------------------

def switch_matrix(p_still_to_moving, p_moving_to_still):
    """2x2 Markov mode-transition matrix, row = mode at the previous tick,
    column = mode at this tick, order (still, moving).
    Arguments:
        p_still_to_moving: probability still -> moving in one tick
        p_moving_to_still: probability moving -> still in one tick
    Returns:
        (2, 2) row-stochastic matrix
    """
    return np.array([[1.0 - p_still_to_moving, p_still_to_moving],
                     [p_moving_to_still, 1.0 - p_moving_to_still]])


def mean_sojourn_times(t_anchor, t_extend, t_slip, p_slip):
    """Mean time the truth spends in each mode before leaving it, from the
    simulation's own parameters (rate balance over one gait cycle). Per cycle
    the body is still for t_anchor - p_slip * t_slip seconds and moving for
    t_extend + p_slip * t_slip seconds, and each mode is entered 1 + p_slip
    times (the extend phase once, each slip pulse once; a slipped anchor
    phase is still, slip, still). The mean sojourn is time in the mode per
    cycle divided by entries per cycle. Assumes a slip pulse does not touch
    the extend phase.
    Arguments:
        t_anchor, t_extend: phase durations per cycle (s)
        t_slip: slip pulse duration (s)
        p_slip: probability an anchor phase contains a slip pulse
    Returns:
        T_still, T_moving: mean sojourn times (s)
    """
    entries = 1.0 + p_slip
    return (t_anchor - p_slip * t_slip) / entries, (t_extend + p_slip * t_slip) / entries


def stationary_still_probability(P):
    """Long-run probability of the still mode of a 2x2 transition matrix.
    Arguments:
        P: (2, 2) row-stochastic matrix, order (still, moving)
    Returns:
        pi_still: float
    """
    q_sm, q_ms = P[STILL, MOVING], P[MOVING, STILL]
    return q_ms / (q_sm + q_ms)


def design_transitions(kind, is_anchor, dt, t_anchor, t_extend, t_slip, p_slip, scale=1.0, scale_extend=None):
    """Per-tick transition matrices of the IMM variants, derived from the
    simulation's own parameters (filter design choices, not measured values).
    Convention: row = mode at the previous tick, column = mode at this tick
    (row-stochastic); Blom & Bar-Shalom's H is the transpose of this matrix.
      - "const" (schedule-free): one matrix with p_still_to_moving = dt / T_still
        and p_moving_to_still = dt / T_moving, T from mean_sojourn_times.
      - "schedule_next" (the rule "the matrix depends on the scheduled phase of
        the next tick"): for the step into tick k, by whether the known
        schedule marks tick k as anchor.
          scheduled extend: both rows [e, 1 - e] with e = dt / t_extend, the
              share of extend ticks on which the true speed is exactly 0 (the
              first one), so the next mode is moving with probability 1 - e.
          scheduled anchor: still -> moving (a slip starts) with probability
              dt * p_slip / (t_anchor - p_slip * t_slip), the slip rate per
              still second; moving -> still (the slip ends) with probability
              dt / t_slip, one over the slip duration.
        This leaves the moving mode with probability 0.9167 per tick on the
        first anchor tick after an extend phase, where the body is in fact
        exactly still, so the moving mode lingers there (see the doc).
      - "schedule" (the default schedule-aware rule): as "schedule_next", except
        on the first scheduled anchor tick after an extend tick both rows are
        [1, 0]. The gait ends the extend ramp at v = 0 and a slip pulse starts
        from v = 0, so the body is exactly still on that tick.
    `scale` multiplies every switch probability (clipped to 1) for the
    sensitivity sweep (the [1, 0] boundary rows are exact and not scaled);
    `scale_extend` (default: scale) overrides it for the scheduled-extend
    matrix only, to isolate which matrix a sweep result comes from. The
    initial mode probabilities are the stationary distribution of the matrix
    used at tick 1.
    Arguments:
        kind: "const", "schedule" or "schedule_next"
        is_anchor: (N,) bool scheduled anchor flags
        dt: step interval (s)
        t_anchor, t_extend, t_slip, p_slip: as in mean_sojourn_times
        scale: multiplier on the switch probabilities
        scale_extend: multiplier for the scheduled-extend matrix (None = scale)
    Returns:
        trans: (N, 2, 2) transition matrix for the step into each tick (row 0 unused)
        mu0: (2,) initial mode probabilities (still, moving)
        info: dict of the derived numbers, for printing
    """
    n = len(is_anchor)
    scale_ext = scale if scale_extend is None else scale_extend
    clip = lambda p, sc=scale: float(min(max(sc * p, 0.0), 1.0))  # noqa: E731
    if kind == "const":
        T_s, T_m = mean_sojourn_times(t_anchor, t_extend, t_slip, p_slip)
        P = switch_matrix(clip(dt / T_s), clip(dt / T_m))
        trans = np.broadcast_to(P, (n, 2, 2)).copy()
        info = dict(T_still=T_s, T_moving=T_m, P=P)
        mu0_P = P
    elif kind in ("schedule", "schedule_next"):
        free_still = t_anchor - p_slip * t_slip
        if free_still <= 0.0:
            raise ValueError("need p_slip * t_slip < t_anchor")
        e = dt / t_extend
        P_ext = switch_matrix(1.0 - clip(e, scale_ext), clip(e, scale_ext))
        P_ext[MOVING] = P_ext[STILL]  # both rows [e, 1 - e]
        P_anc = switch_matrix(clip(dt * p_slip / free_still), clip(dt / t_slip))
        trans = np.where(is_anchor[:, None, None], P_anc, P_ext)
        if kind == "schedule":
            edge = np.concatenate([[False], is_anchor[1:] & ~is_anchor[:-1]])
            trans[edge] = switch_matrix(0.0, 1.0)
        info = dict(P_extend=P_ext, P_anchor=P_anc)
        mu0_P = trans[1]
    else:
        raise ValueError(f"unknown transition kind: {kind!r}")
    pi_s = stationary_still_probability(mu0_P) if mu0_P[STILL, MOVING] + mu0_P[MOVING, STILL] > 0 else 0.5
    return trans, np.array([pi_s, 1.0 - pi_s]), info


# --- the IMM filter ------------------------------------------------------------------------------

def mix_modes(mu, Pi, xs, Ps):
    """IMM mixing step, Blom & Bar-Shalom (1988) eq. (12)-(14), batched.
    Convention: Pi[i, j] = P(mode at this tick = j | mode at the previous tick = i)
    (rows sum to 1), the transpose of the paper's H_ij = P(theta_t = i | theta_{t-1} = j)
    (columns sum to 1, eq. 9). So the paper's H_ij is Pi[j, i] here, and
      mu_bar_i  = sum_j H_ij mu_j                       (12)  -> c[i] = sum_j Pi[j, i] mu[j]
      x0_i      = sum_j H_ij mu_j x_j / mu_bar_i        (13)
      P0_i      = sum_j H_ij mu_j [P_j + (x_j - x0_i)(x_j - x0_i)^T] / mu_bar_i   (14).
    A mode with mu_bar_i = 0 is pruned in the paper; here it keeps the
    previous mode probabilities as weights, so its (unused) state stays
    finite, and its probability stays 0.
    Arguments:
        mu: (B, 2) mode probabilities after the previous tick
        Pi: (2, 2) row-stochastic transition matrix for this step
        xs: (B, 2, 2) mode estimates, axis 1 = mode, axis 2 = [p, v]
        Ps: (B, 2, 2, 2) mode covariances
    Returns:
        c: (B, 2) predicted mode probabilities mu_bar
        x0: (B, 2, 2) mixed initial states
        P0: (B, 2, 2, 2) mixed initial covariances
    """
    c = mu @ Pi
    w = np.where(c[:, None, :] > 0.0, mu[:, :, None] * Pi / np.where(c > 0.0, c, 1.0)[:, None, :],
                 mu[:, :, None])                                           # (B, i, j) = H_ji mu_i / mu_bar_j
    x0 = np.einsum("bij,bis->bjs", w, xs)
    dx = xs[:, :, None, :] - x0[:, None, :, :]
    P0 = np.einsum("bij,bist->bjst", w, Ps) + np.einsum("bij,bijs,bijt->bjst", w, dx, dx)
    return c, x0, P0


def predict_modes(x0, P0, dt, process_noise_std, sigma_still, still_integrates=False):
    """Mode-matched predict step for both modes at once.
      - still: v' = 0 + w with std sigma_still (the "v = 0" belief as dynamics,
        F = [[1, 0], [0, 0]], Q = diag(0, sigma_still^2)); with
        still_integrates the position also advances by the carried velocity,
        F = [[1, dt], [0, 0]].
      - moving: inchworm_zupt_ekf.predict, the constant-velocity model with
        acceleration noise process_noise_std.
    Arguments:
        x0: (B, 2, 2) mixed states, axis 1 = mode (still, moving), axis 2 = [p, v]
        P0: (B, 2, 2, 2) mixed covariances
        dt: step interval (s)
        process_noise_std: moving-mode acceleration noise std (m/s^2)
        sigma_still: still-mode velocity noise std (m/s)
        still_integrates: use p' = p + v dt in the still mode
    Returns:
        x_pred: (B, 2, 2), P_pred: (B, 2, 2, 2)
    """
    F_s = np.array([[1.0, dt if still_integrates else 0.0], [0.0, 0.0]])
    Q_s = np.diag([0.0, sigma_still ** 2])
    x_s = x0[:, STILL] @ F_s.T
    P_s = F_s @ P0[:, STILL] @ F_s.T + Q_s
    x_m, P_m = predict(x0[:, MOVING, :, None], P0[:, MOVING], dt, process_noise_std)
    return np.stack([x_s, x_m[..., 0]], axis=1), np.stack([P_s, P_m], axis=1)


def run_imm(x_init, P_init, z, d, dt, process_noise_std, R_pos, R_vel, sigma_still, trans, mu0,
            use_pad, still_integrates=False):
    """Interacting multiple model filter over two modes (still, moving) that
    both process the same measurements: position z every tick and, when
    use_pad, the pad reading d / dt as a velocity. One cycle per tick
    (Blom & Bar-Shalom 1988, Sec. III, eq. (12)-(19)):
      1. mixing: predicted mode probabilities mu_bar, mixed initial states and
         covariances, eq. (12)-(14), see mix_modes for the index convention;
      2. mode-matched predict and update (a Kalman filter per mode);
      3. mode probabilities mu_i = c mu_bar_i |Q_i|^(-1/2) exp(-1/2 s_i^T Q_i^-1 s_i),
         eq. (15)-(17), with s_i the innovation and Q_i its covariance. The paper
         drops the (2 pi)^(-m/2) factor of the Gaussian; it is common to both
         modes here (same measurement vector), so keeping it (as this code does)
         changes nothing after normalization;
      4. combined estimate x = sum_i mu_i x_i, P = sum_i mu_i [P_i + (x_i - x)(x_i - x)^T],
         eq. (18)-(19).
    Both modes start from x_init, P_init. Runs a batch of trials at once
    when z has a leading trial axis.
    Arguments:
        x_init: initial state (2,) or (B, 2)
        P_init: initial covariance (2, 2)
        z: position measurements (N,) or (B, N)
        d: pad readings (N,) or (B, N) (displacement per tick, m); ignored unless use_pad
        dt: step interval (s)
        process_noise_std: moving-mode acceleration noise std (m/s^2)
        R_pos: position noise variance (m^2)
        R_vel: velocity-reading noise variance ((m/s)^2), = (sensor noise / dt)^2
        sigma_still: still-mode velocity noise std (m/s)
        trans: (N, 2, 2) per-tick row-stochastic transition matrices, row = previous mode (design_transitions)
        mu0: (2,) initial mode probabilities (still, moving)
        use_pad: stack the velocity reading with the position
        still_integrates: see predict_modes
    Returns:
        x_est: (N, 2) or (B, N, 2) combined estimates
        P_est: (N, 2, 2) or (B, N, 2, 2) combined covariances
        mu: (N, 2) or (B, N, 2) mode probabilities after each tick's update
            (row 0 is mu0); column MOVING is P(M)
    """
    single = np.ndim(z) == 1
    z = np.atleast_2d(z)
    d = np.atleast_2d(d)
    x_init = np.atleast_2d(x_init)
    B, N = z.shape
    if use_pad:
        Y = np.stack([z, d / dt], axis=-1)
        H, R = np.eye(2), np.diag([R_pos, R_vel])
    else:
        Y = z[..., None]
        H, R = np.array([[1.0, 0.0]]), np.array([[R_pos]])

    xs = np.repeat(x_init[:, None, :], 2, axis=1)
    Ps = np.broadcast_to(P_init, (B, 2, 2, 2)).copy()
    mu = np.broadcast_to(mu0, (B, 2)).copy()
    x_est = np.empty((B, N, 2))
    P_est = np.empty((B, N, 2, 2))
    mu_hist = np.empty((B, N, 2))
    x_est[:, 0], P_est[:, 0], mu_hist[:, 0] = x_init, P_init, mu
    for k in range(N - 1):
        c, x0, P0 = mix_modes(mu, trans[k + 1], xs, Ps)
        x_p, P_p = predict_modes(x0, P0, dt, process_noise_std, sigma_still, still_integrates)
        xs, Ps, log_lik = kf_update_loglik(x_p, P_p, Y[:, k + 1, None, :], H, R)
        with np.errstate(divide="ignore"):
            logw = np.log(c) + log_lik
        mu = np.exp(logw - logw.max(axis=1, keepdims=True))
        mu /= mu.sum(axis=1, keepdims=True)
        x_c = np.einsum("bj,bjs->bs", mu, xs)
        dxc = xs - x_c[:, None, :]
        P_c = np.einsum("bj,bjst->bst", mu, Ps) + np.einsum("bj,bjs,bjt->bst", mu, dxc, dxc)
        x_est[:, k + 1], P_est[:, k + 1], mu_hist[:, k + 1] = x_c, P_c, mu
    if single:
        return x_est[0], P_est[0], mu_hist[0]
    return x_est, P_est, mu_hist


# --- scoring -----------------------------------------------------------------------------------------

def batched_nees(x_true, x_est, P_est):
    """NEES err^T P^-1 err at every tick of every trial (same quantity as
    utils.nees, vectorized).
    Arguments:
        x_true: (N, 2) true states
        x_est: (B, N, 2) estimates
        P_est: (B, N, 2, 2) covariances
    Returns:
        (B, N) NEES
    """
    err = (x_true - x_est)[..., None]
    return (np.swapaxes(err, -1, -2) @ np.linalg.solve(P_est, err))[..., 0, 0]


def slip_pulses(is_slip):
    """Index ranges (first tick, one past the last tick) of each slip pulse.
    Arguments:
        is_slip: (N,) bool
    Returns:
        list of (a, b)
    """
    edges = np.diff(np.concatenate([[False], is_slip, [False]]).astype(int))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def detection_stats(flag, is_slip, is_clean, dt, p_move=None):
    """Slip detection of a per-tick flag (trials, ticks), e.g. P(M) > 0.5.
    Arguments:
        flag: (B, N) bool, True where the method says "moving / slipping"
        is_slip, is_clean: (N,) bool windows
        dt: step interval (s)
        p_move: optional (B, N) P(M), for the window means below
    Returns:
        dict with
          hit: fraction of slip ticks flagged
          false_alarm: fraction of clean-anchor ticks flagged
          pre_flagged: fraction of (trial, pulse) pairs already flagged on the pulse's first tick
          delay_s: mean delay (s) from a pulse's first tick to its first flagged tick, over
              the pairs not pre-flagged and detected inside the pulse
          missed: fraction of not-pre-flagged pairs never flagged inside the pulse
          pm_slip, pm_clean: mean P(M) over the slip / clean-anchor ticks (nan without p_move)
    """
    hit = flag[:, is_slip].mean() if is_slip.any() else float("nan")
    false_alarm = flag[:, is_clean].mean() if is_clean.any() else float("nan")
    delays, pre, missed, total = [], 0, 0, 0
    for a, b in slip_pulses(is_slip):
        seg = flag[:, a:b]
        total += seg.shape[0]
        already = seg[:, 0]
        pre += int(already.sum())
        rest = seg[~already]
        found = rest.any(axis=1)
        missed += int((~found).sum())
        delays.extend((np.argmax(rest[found], axis=1) * dt).tolist())
    pm = lambda w: float(p_move[:, w].mean()) if p_move is not None and w.any() else float("nan")  # noqa: E731
    return dict(hit=hit, false_alarm=false_alarm, pm_slip=pm(is_slip), pm_clean=pm(is_clean),
                pre_flagged=pre / total if total else float("nan"),
                delay_s=float(np.mean(delays)) if delays else float("nan"),
                missed=missed / max(total - pre, 1) if total else float("nan"))


# --- experiment ---------------------------------------------------------------------------------------

def policy_spec(mode):
    """Spec of one of the parent script's five policies (inchworm_slip_gate_ekf.GATING_MODES)."""
    return dict(type="policy", mode=mode)


def imm_spec(kind, use_pad, scale=1.0, still_integrates=False, scale_extend=None):
    """Spec of one IMM variant: transition kind ("const", "schedule" or "schedule_next"), whether the pad reading is
    used, the switch-rate scale factor (and a separate one for the scheduled-extend matrix) and the still-mode
    position dynamics."""
    return dict(type="imm", kind=kind, use_pad=use_pad, scale=scale, scale_extend=scale_extend,
                still_integrates=still_integrates)


def make_setup(args, sensor_noise_std):
    """Truth, one data set and the initial guess, with the same random-number order as
    inchworm_slip_gate_ekf.main (slip pulses, position noise, sensor noise, then the
    initial error), so seed 0 gives the same two slip pulses. The returned rng is the
    state main() hands to its Monte Carlo loop.
    Arguments:
        args: namespace of model parameters (add_model_arguments)
        sensor_noise_std: pad sensor noise std (m), overriding args.sensor_noise_std
    Returns:
        SimpleNamespace with x_true, z, d, is_anchor, is_cruise, is_slip, is_clean, x_init, P_init, rng
    """
    rng = np.random.default_rng(args.seed)
    x_true, z, d, is_anchor, is_cruise, is_slip, is_clean = generate_ground_truth_and_data(
        args.duration, args.dt, args.t_anchor, args.t_extend, args.t_ramp, args.v_extend,
        args.pos_noise_std, sensor_noise_std, args.p_slip, args.v_slip, args.t_slip, rng)
    init_std = np.array([args.init_pos_noise_std, args.init_vel_noise_std])
    x_init = x_true[0] + rng.normal(0.0, 1.0, 2) * init_std
    return SimpleNamespace(x_true=x_true, z=z, d=d, is_anchor=is_anchor, is_cruise=is_cruise, is_slip=is_slip,
                           is_clean=is_clean, x_init=x_init, P_init=np.diag(init_std ** 2), rng=rng)


def draw_trials(x_true, dt, R_pos, sensor_noise_std, init_std, n_trials, rng):
    """Monte Carlo data in the order of inchworm_slip_gate_ekf.run_monte_carlo_consistency
    (per trial: initial error, position noise, sensor noise), so the same rng state gives
    the same trials.
    Arguments:
        x_true: (N, 2) true states
        dt: step interval (s)
        R_pos: position noise variance (m^2)
        sensor_noise_std: pad sensor noise std (m)
        init_std: (2,) initial error stds
        n_trials: number of trials
        rng: generator (advanced)
    Returns:
        x_init (T, 2), z (T, N), d (T, N)
    """
    n_pts = len(x_true)
    x0, zz, dd = np.empty((n_trials, 2)), np.empty((n_trials, n_pts)), np.empty((n_trials, n_pts))
    for i in range(n_trials):
        x0[i] = x_true[0] + rng.normal(0.0, 1.0, 2) * init_std
        zz[i] = x_true[:, 0] + rng.normal(0.0, np.sqrt(R_pos), n_pts)
        dd[i] = dt * x_true[:, 1] + rng.normal(0.0, sensor_noise_std, n_pts)
    return x0, zz, dd


def evaluate(args, setup, specs, sensor_noise_std):
    """Runs every spec once on the setup's data (RMSE, time, memory, mode probability) and
    over args.n_mc_trials Monte Carlo trials that all specs share (NEES and, for IMMs,
    slip detection). The Monte Carlo rng is a copy of the setup's, so repeated calls with
    different sensor noise see the same position noise and initial errors.
    Arguments:
        args: model parameters
        setup: from make_setup
        specs: dict label -> policy_spec / imm_spec
        sensor_noise_std: pad sensor noise std (m); sets R_vel and the gate threshold
    Returns:
        dict label -> dict(x_est, p_move (IMM only), rmse_pos, rmse_vel (single run), t_step, mem,
        nees (N,), mse (N, 2) Monte Carlo mean squared [p, v] error per tick, detect (IMM only));
        plus the key "_gate": detection stats of the sensor veto |d| > threshold
    """
    dt = args.dt
    n_steps = len(setup.x_true) - 1
    R_pos, R_zupt = args.pos_noise_std ** 2, args.zupt_noise_std ** 2
    R_vel = (sensor_noise_std / dt) ** 2
    d_thr = args.gate_sigmas * sensor_noise_std
    still_int = args.still_model == "integrate"
    designs = {lab: design_transitions(s["kind"], setup.is_anchor, dt, args.t_anchor, args.t_extend, args.t_slip,
                                       args.p_slip, s["scale"], s["scale_extend"])[:2] for lab, s in specs.items() if s["type"] == "imm"}

    def run(lab, x_init, P_init, z, d):
        s = specs[lab]
        if s["type"] == "policy":
            x, P = run_ekf(x_init, P_init, z, d, dt, n_steps, args.process_noise_std, R_pos, R_zupt, R_vel,
                           d_thr, setup.is_anchor, s["mode"])
            return x, np.array(P), None
        trans, mu0 = designs[lab]
        return run_imm(x_init, P_init, z, d, dt, args.process_noise_std, R_pos, R_vel, args.zupt_noise_std,
                       trans, mu0, s["use_pad"], s["still_integrates"] or still_int)

    out = {}
    for lab in specs:
        (x_est, P_est, mu), t_step, mem = measure_performance(
            run, lab, setup.x_init, setup.P_init, setup.z, setup.d, n_steps=n_steps)
        pe, ve = state_errors(setup.x_true, x_est)
        out[lab] = dict(x_est=x_est, p_move=None if mu is None else mu[:, MOVING], t_step=t_step, mem=mem,
                        rmse_pos=float(np.sqrt(np.mean(pe ** 2))), rmse_vel=float(np.sqrt(np.mean(ve ** 2))))

    init_std = np.array([args.init_pos_noise_std, args.init_vel_noise_std])
    x0, zz, dd = draw_trials(setup.x_true, dt, R_pos, sensor_noise_std, init_std, args.n_mc_trials,
                             copy.deepcopy(setup.rng))
    for lab, s in specs.items():
        if s["type"] == "policy":
            total, sq = np.zeros(len(setup.x_true)), np.zeros((len(setup.x_true), 2))
            for i in range(args.n_mc_trials):
                x, P, _ = run(lab, x0[i], setup.P_init, zz[i], dd[i])
                total += batched_nees(setup.x_true, x[None], P[None])[0]
                sq += (x - setup.x_true) ** 2
            out[lab]["nees"] = total / args.n_mc_trials
            out[lab]["mse"] = sq / args.n_mc_trials
        else:
            x, P, mu = run(lab, x0, setup.P_init, zz, dd)
            out[lab]["nees"] = batched_nees(setup.x_true, x, P).mean(axis=0)
            out[lab]["mse"] = ((x - setup.x_true) ** 2).mean(axis=0)
            out[lab]["detect"] = detection_stats(mu[:, :, MOVING] > 0.5, setup.is_slip, setup.is_clean, dt,
                                                   mu[:, :, MOVING])
    out["_gate"] = detection_stats(np.abs(dd) > d_thr, setup.is_slip, setup.is_clean, dt)
    return out


# --- printing and plotting ---------------------------------------------------------------------------

def window_masks(setup):
    return {"overall": np.ones_like(setup.is_slip), "clean anchor": setup.is_clean, "slip": setup.is_slip,
            "cruise": setup.is_cruise}


def print_results_table(title, res, setup, labels):
    """Prints RMSE, windowed NEES and time per step for the given labels."""
    masks = window_masks(setup)
    print(f"\n{title}")
    print(f"  {'':22s}{'RMSE pos (m)':>13s}{'RMSE vel (m/s)':>16s}" + "".join(f"{w:>14s}" for w in WINDOWS)
          + f"{'us/step':>10s}")
    for lab in labels:
        r = res[lab]
        cells = "".join(f"{np.mean(r['nees'][masks[w]]) if masks[w].any() else float('nan'):14.2f}" for w in WINDOWS)
        print(f"  {lab:<22s}{r['rmse_pos']:13.4f}{r['rmse_vel']:16.4f}{cells}{r['t_step'] * 1e6:10.1f}")


def print_window_rmse_table(title, res, setup, labels):
    """Prints the Monte Carlo RMS velocity error (m/s) per window: the square root of the mean over the trials
    and the window's ticks of the squared velocity error."""
    masks = window_masks(setup)
    print(f"\n{title}")
    print(f"  {'':22s}" + "".join(f"{w:>14s}" for w in WINDOWS))
    for lab in labels:
        cells = "".join(f"{np.sqrt(np.mean(res[lab]['mse'][masks[w], 1])) if masks[w].any() else float('nan'):14.4f}"
                        for w in WINDOWS)
        print(f"  {lab:<22s}{cells}")


def print_detection_table(title, res, labels):
    """Prints the slip-detection statistics of the IMM labels and of the sensor veto."""
    print(f"\n{title}")
    print(f"  {'':22s}{'slip ticks hit':>16s}{'clean false alarm':>19s}{'mean P(M) slip':>16s}{'mean P(M) clean':>17s}"
          f"{'pre-flagged':>13s}{'delay (s)':>11s}{'missed':>9s}")
    rows = [(lab, res[lab]["detect"]) for lab in labels if "detect" in res[lab]] + [("sensor veto |d|>thr", res["_gate"])]
    for lab, s in rows:
        print(f"  {lab:<22s}{s['hit']:16.3f}{s['false_alarm']:19.3f}{s['pm_slip']:16.3f}{s['pm_clean']:17.3f}"
              f"{s['pre_flagged']:13.3f}{s['delay_s']:11.3f}{s['missed']:9.3f}")


def add_model_arguments(parser):
    """Adds every model parameter (same names and defaults as inchworm_slip_gate_ekf.py) plus the IMM options."""
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation length in seconds")
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
                        help="Gate threshold of the parent policies, in sensor noise std-devs")
    parser.add_argument("--process-noise-std", type=float, default=0.5,
                        help="Assumed acceleration disturbance std-dev of the moving mode / parent EKFs (m/s^2)")
    parser.add_argument("--zupt-noise-std", type=float, default=0.01,
                        help="ZUPT noise std-dev of the parent policies, and velocity noise std of the IMM "
                             "still mode (m/s)")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.05, help="Initial position error std-dev (m)")
    parser.add_argument("--init-vel-noise-std", type=float, default=0.05, help="Initial velocity error std-dev (m/s)")
    parser.add_argument("--still-model", choices=("hold", "integrate"), default="hold",
                        help="Still-mode position dynamics: hold = p' = p (default), integrate = p' = p + v dt")
    parser.add_argument("--n-mc-trials", type=int, default=500, help="Number of Monte Carlo consistency trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")


REGIME_A = tuple(GATING_MODES) + ("imm", "imm_schedule")
REGIME_B = ("never", "schedule", "imm_pos", "imm_schedule_pos")
SCALE_SWEEP = (0.1, 10.0)


def main_specs():
    """All variants run on the shared data of the main setup (regimes A and B and the scale sweep)."""
    specs = {m: policy_spec(m) for m in GATING_MODES}
    specs["imm"] = imm_spec("const", True)
    specs["imm_schedule"] = imm_spec("schedule", True)
    specs["imm_pos"] = imm_spec("const", False)
    specs["imm_schedule_pos"] = imm_spec("schedule", False)
    specs["imm_schedule_next"] = imm_spec("schedule_next", True)
    specs["imm_schedule_next_pos"] = imm_spec("schedule_next", False)
    for s in SCALE_SWEEP:
        specs[f"imm_schedule x{s:g}"] = imm_spec("schedule", True, s)
        specs[f"imm_schedule_pos x{s:g}"] = imm_spec("schedule", False, s)
    # which matrix drives the x10 collapse of the position-only variant: scale only one of the two
    specs["imm_schedule_pos x10 anchor only"] = imm_spec("schedule", False, 10.0, scale_extend=1.0)
    specs["imm_schedule_pos x10 extend only"] = imm_spec("schedule", False, 1.0, scale_extend=10.0)
    return specs


def print_transition_design(args, is_anchor):
    for kind in ("const", "schedule_next", "schedule"):
        _, mu0, info = design_transitions(kind, is_anchor, args.dt, args.t_anchor, args.t_extend, args.t_slip,
                                          args.p_slip)
        print(f"Transition design '{kind}' (rows = from still, moving; columns = to still, moving), "
              f"mu0 = {np.round(mu0, 4).tolist()}:")
        if kind == "const":
            print(f"  mean sojourn: still {info['T_still']:.4f} s, moving {info['T_moving']:.4f} s")
            print(f"  P = {np.round(info['P'], 4).tolist()}")
        else:
            print(f"  scheduled extend: {np.round(info['P_extend'], 4).tolist()}")
            print(f"  scheduled anchor: {np.round(info['P_anchor'], 4).tolist()}")
            if kind == "schedule":
                print("  first scheduled anchor tick after an extend tick: [[1, 0], [1, 0]]")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--noise-sweep", type=float, nargs="*", default=[0.002, 0.005],
                        help="Extra pad sensor noise stds (m) for regime C (besides --sensor-noise-std); "
                             "pass the flag with no values to skip")
    parser.add_argument("--seeds-table", type=int, nargs="+", default=None,
                        help="Instead of the main run, repeat regimes A and B for these seeds and print the "
                             "summary rows (e.g. --seeds-table 0 1 2 3)")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")
    args = parser.parse_args()

    lo, hi = averaged_nees_bounds(STATE_DIM, args.n_mc_trials)
    if args.seeds_table:
        run_seeds_table(args, lo, hi)
        return

    setup = make_setup(args, args.sensor_noise_std)
    n_pulses = len(slip_pulses(setup.is_slip))
    print(f"Setup: seed {args.seed}, {n_pulses} slip pulses, slip ticks: {int(setup.is_slip.sum())}, "
          f"clean-anchor ticks: {int(setup.is_clean.sum())}, cruise ticks: {int(setup.is_cruise.sum())}; "
          f"{args.n_mc_trials} Monte Carlo trials; still-mode position dynamics: {args.still_model}")
    print(f"Truth vs model: extend ramp acceleration {args.v_extend / args.t_ramp:.2f} m/s^2, slip ramp "
          f"acceleration {args.v_slip / args.t_ramp:.2f} m/s^2, moving-mode process noise "
          f"{args.process_noise_std:.2f} m/s^2")
    print_transition_design(args, setup.is_anchor)

    res = evaluate(args, setup, main_specs(), args.sensor_noise_std)
    print(f"\nMonte Carlo NEES (a consistent {STATE_DIM}-DoF filter averages ~{STATE_DIM}; 95% bounds for a "
          f"{args.n_mc_trials}-trial average: [{lo:.2f}, {hi:.2f}])")
    print_results_table("Regime A: pad sensor available (position + pad reading)", res, setup, REGIME_A)
    print_results_table("Regime B: no pad sensor (position only)", res, setup, REGIME_B)
    print_window_rmse_table("Monte Carlo RMS velocity error by window (m/s), regimes A and B", res, setup,
                            ("never", "schedule", "schedule_and_sensor", "sensor_fused", "imm", "imm_schedule",
                             "imm_pos", "imm_schedule_pos"))
    nxt = ["imm_schedule_next", "imm_schedule_next_pos"]
    print_results_table("Variant: next-tick-only schedule rule, no forced-still first anchor tick (imm_schedule_next)",
                        res, setup, ["imm_schedule", "imm_schedule_pos"] + nxt)
    sweep = [f"imm_schedule x{s:g}" for s in SCALE_SWEEP] + [f"imm_schedule_pos x{s:g}" for s in SCALE_SWEEP]
    isolate = ["imm_schedule_pos x10 anchor only", "imm_schedule_pos x10 extend only"]
    print_results_table("Switch-rate scale sweep (x1 rows are imm_schedule and imm_schedule_pos above)", res, setup,
                        ["imm_schedule", "imm_schedule_pos"] + sweep + isolate)
    print_detection_table("Slip detection, P(M) > 0.5 (Monte Carlo average), regime A and B",
                          res, ["imm", "imm_schedule", "imm_pos", "imm_schedule_pos"] + nxt + sweep + isolate)

    c_specs = {"never": policy_spec("never"), "schedule_and_sensor": policy_spec("schedule_and_sensor"),
               "sensor_fused": policy_spec("sensor_fused"), "imm_schedule": imm_spec("schedule", True),
               "imm_pos": imm_spec("const", False)}
    res_c = {args.sensor_noise_std: evaluate(args, setup, c_specs, args.sensor_noise_std)}
    for sd in args.noise_sweep or []:
        res_c[sd] = evaluate(args, make_setup(args, sd), c_specs, sd)
    for sd, r in sorted(res_c.items()):
        s_setup = setup if sd == args.sensor_noise_std else make_setup(args, sd)
        print_results_table(f"Regime C: sensor noise {sd * 1e3:.2f} mm per reading = {sd / args.dt:.3f} m/s", r,
                            s_setup, ["schedule_and_sensor", "sensor_fused", "imm_schedule", "imm_pos"])
        print_detection_table(f"  slip detection at sensor noise {sd * 1e3:.2f} mm", r, ["imm_schedule"])
    ref = res_c[args.sensor_noise_std]["imm_pos"]["nees"]
    print("\nInvariance check (imm_pos ignores the pad sensor, so its NEES must not change with the sensor "
          "noise): max |difference| from the first level = "
          + ", ".join(f"{np.max(np.abs(r['imm_pos']['nees'] - ref)):.1e}" for _, r in sorted(res_c.items())))

    print("\nAverage time (per step) and peak memory (whole run), empirical (batch of one trial):")
    for lab in REGIME_A + ("imm_pos", "imm_schedule_pos"):
        print(f"  {lab:<22s} avg time={res[lab]['t_step'] * 1e6:9.2f} us/step | peak mem={res[lab]['mem'] / 1024.0:9.3f} KB")

    t = np.arange(len(setup.x_true)) * args.dt
    fig, (ax_v, ax_p, ax_n) = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    ax_v.plot(t, setup.x_true[:, 1], label="Ground truth", color="black", linewidth=2, linestyle="--")
    for lab, col in (("schedule_and_sensor", "tab:purple"), ("sensor_fused", "tab:green"), ("imm_schedule", "tab:red")):
        r = res[lab] if lab in res else res_c[args.sensor_noise_std][lab]
        ax_v.plot(t, r["x_est"][:, 1], label=lab, color=col)
    ax_v.set_ylabel("Velocity (m/s)")
    ax_v.set_title("Inchworm with anchor slip: IMM vs. gated ZUPT")
    ax_v.legend()
    ax_p.plot(t, res["imm_schedule"]["p_move"], color="tab:red", label="imm_schedule (position + pad)")
    ax_p.plot(t, res["imm_schedule_pos"]["p_move"], color="tab:blue", label="imm_schedule_pos (position only)")
    ax_p.plot(t, res["imm_pos"]["p_move"], color="tab:gray", linewidth=1, label="imm_pos (position only, no schedule)")
    ax_p.axhline(0.5, color="black", linewidth=0.8, linestyle=":")
    ax_p.set_ylabel("P(moving)")
    ax_p.set_ylim(-0.05, 1.45)
    ax_p.legend(fontsize=7.5, loc="upper center", ncol=2)
    for lab, col in (("never", "tab:gray"), ("schedule", "tab:blue"), ("schedule_and_sensor", "tab:purple"),
                     ("sensor_fused", "tab:green"), ("imm", "tab:orange"), ("imm_schedule", "tab:red"),
                     ("imm_schedule_pos", "tab:cyan")):
        ax_n.plot(t, res[lab]["nees"], label=lab, color=col)
    ax_n.axhline(STATE_DIM, color="black", linewidth=1, label=f"Expected NEES ({STATE_DIM} DoF)")
    ax_n.axhline(lo, color="black", linestyle="--", linewidth=1, label=f"95% bounds for a {args.n_mc_trials}-trial average")
    ax_n.axhline(hi, color="black", linestyle="--", linewidth=1)
    ax_n.set_ylabel(f"Monte Carlo avg. NEES ({args.n_mc_trials} trials)")
    ax_n.set_xlabel("Time (s)")
    ax_n.set_yscale("log")
    ax_n.set_ylim(0.4, 2000.0)
    ax_n.legend(fontsize=7, ncol=3, loc="upper center")
    for ax in (ax_v, ax_p, ax_n):
        for a, b in slip_pulses(setup.is_slip):
            ax.axvspan(t[a], t[b - 1] + args.dt, color="gold", alpha=0.25, linewidth=0)
        for a, b in zip(np.where(np.diff(np.concatenate([[False], setup.is_anchor, [False]]).astype(int)) == 1)[0],
                        np.where(np.diff(np.concatenate([[False], setup.is_anchor, [False]]).astype(int)) == -1)[0]):
            ax.axvspan(t[a], t[b - 1] + args.dt, color="tab:blue", alpha=0.05, linewidth=0)
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


def run_seeds_table(args, lo, hi):
    """Regimes A and B for several seeds (each seed redraws the slip pulses)."""
    specs = {m: policy_spec(m) for m in ("never", "schedule", "schedule_and_sensor", "sensor_fused")}
    specs.update(imm=imm_spec("const", True), imm_schedule=imm_spec("schedule", True),
                 imm_pos=imm_spec("const", False), imm_schedule_pos=imm_spec("schedule", False))
    print(f"Seeds table, {args.n_mc_trials} trials, 95% bounds [{lo:.2f}, {hi:.2f}]")
    for seed in args.seeds_table:
        a = copy.copy(args)
        a.seed = seed
        setup = make_setup(a, a.sensor_noise_std)
        res = evaluate(a, setup, specs, a.sensor_noise_std)
        print(f"\nSeed {seed}: {len(slip_pulses(setup.is_slip))} pulses, {int(setup.is_slip.sum())} slip ticks")
        print_results_table(f"seed {seed}", res, setup, list(specs))
        print_detection_table(f"seed {seed} slip detection", res, ["imm", "imm_schedule", "imm_pos", "imm_schedule_pos"])


if __name__ == "__main__":
    main()
