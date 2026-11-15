'''
What a smoother adds over a filter, on one data set: the 1D inchworm of
inchworm_zupt_ekf.py (state `x = [p, v]`, noisy position measurement every
tick, zero-velocity update (ZUPT) on the scheduled anchor ticks) estimated
three ways from the same measurements:

  - filter: the forward Kalman filter, `phase_conditional` of
    inchworm_zupt_ekf.py. The estimate at tick k uses the measurements up to
    tick k only. This script re-writes its loop only to store the predicted
    and updated means and covariances and each step's transition matrix F
    (a test checks it reproduces run_ekf_phase_conditional_zupt exactly).

  - smoother: the Rauch-Tung-Striebel (RTS) backward pass over the stored
    forward results (Sarkka 2013, Theorem 8.2, Eq. 8.6; Rauch, Tung &
    Striebel 1965). The estimate at tick k uses all the measurements, before
    and after k, and the backward pass starts from the last filter tick.

  - batch: the whole run solved as one linear least-squares problem over all
    ticks' states at once (prior on x_0, one dynamics factor per tick, one
    position factor per tick, one ZUPT factor per anchor tick, each whitened
    by its own covariance). For a linear Gaussian model, the RTS means (and
    covariances) equal the batch solution to round-off; this is checked, not
    assumed (`check_against_batch` in main, and the tests).

Model. The filter's constant-velocity predict step (acceleration-noise
std-dev `process_noise_std`) and its position / ZUPT updates are imported
from inchworm_zupt_ekf.py, unchanged. Default `process_noise_std` is 0.5
m/s^2, not the parent's 0.15: the true ramp acceleration is exactly 0.5 m/s^2
(v_extend / t_ramp), and the parent's truth has no random disturbance, so with
0.15 the truth departs from the filter's model and the NEES of every
estimator measures that mismatch, not its own quality (the parent's doc §3
and inchworm_slip_gate_ekf.py make the same choice). Even at 0.5 the parent's
truth stays deterministic, so three Monte Carlo scenarios are reported:

  - inchworm: the parent's deterministic truth, process noise 0.5 (default);
  - inchworm_q015: the same data with the parent's under-sized 0.15;
  - matched: a truth drawn from the filter's own model each trial (random
    process noise, random initial state, and a noisy velocity reading
    `v + noise` on the anchor ticks instead of the exact zero). This one has
    no model mismatch at all, so a correct filter and a correct smoother must
    both average NEES = 2; it checks the code, not the inchworm.

Scores: position / velocity RMSE and NEES of filter and smoother per gait
window (anchor, ramp-up, cruise, ramp-down), a fixed-lag version (smooth tick
k with the measurements up to tick k + L only), time per step of the forward
and backward passes and the memory the history takes. Monte Carlo trials redraw
the initial error and all measurement noise (and, for `matched`, the process
noise and the initial truth as well).

Findings (documented in docs/filtering_smoothing.md, last part of §8) are
written there from the printed numbers.
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt

from state_estimation_simulations.utils import measure_performance, averaged_nees_bounds  # noqa: E402
from state_estimation_simulations.use_numpy.inchworm_zupt_ekf import (  # noqa: E402
    generate_ground_truth_and_data, predict, measurement_update_position, _kf_update,
    run_ekf_phase_conditional_zupt)

STATE_DIM = 2
H_VEL = np.array([[0.0, 1.0]])
SCENARIOS = ("inchworm", "inchworm_q015", "matched")
WINDOWS = ("anchor", "ramp-up", "cruise", "ramp-down")


def transition_matrix(dt):
    """Constant-velocity transition matrix F of inchworm_zupt_ekf.predict.
    Arguments:
        dt: step interval (s)
    Returns:
        F: (2,2) matrix mapping [p, v] at tick k to [p, v] at tick k+1
    """
    return np.array([[1.0, dt], [0.0, 1.0]])


def process_noise_covariance(dt, process_noise_std):
    """Process noise Q of inchworm_zupt_ekf.predict (diagonal, one step).
    Arguments:
        dt: step interval (s)
        process_noise_std: std-dev of the assumed acceleration disturbance (m/s^2)
    Returns:
        Q: (2,2) covariance added to the state each step (m^2, m^2/s^2)
    """
    return np.diag([(0.5 * process_noise_std * dt * dt) ** 2, (process_noise_std * dt) ** 2])


def run_forward_filter(x_init, P_init, z, z_vel, dt, n_steps, process_noise_std, R_pos, R_vel,
                       store_history=True):
    """Forward Kalman filter of inchworm_zupt_ekf.run_ekf (predict, position
    update, then a velocity update where z_vel is not NaN), additionally
    storing the predicted and updated means and covariances and each F. With
    z_vel = 0 on the anchor ticks it is run_ekf_phase_conditional_zupt.
    Arguments:
        x_init: initial state estimate [p, v] (2,) (m, m/s)
        P_init: initial covariance (2,2)
        z: (n_steps+1,) position measurements (m); z[0] is unused
        z_vel: (n_steps+1,) velocity pseudo-measurements (m/s), NaN on ticks without one
            (0 on anchor ticks for the ZUPT); z_vel[0] is unused
        dt: step interval (s)
        n_steps: number of filter steps
        process_noise_std: std-dev of the assumed acceleration disturbance (m/s^2)
        R_pos: position measurement noise variance (m^2)
        R_vel: velocity pseudo-measurement noise variance (m^2/s^2)
        store_history: False keeps only the current state (what a plain filter needs)
    Returns:
        With store_history, a dict of arrays indexed by tick k = 0..n_steps:
            x_f (N+1,2), P_f (N+1,2,2): updated mean/covariance (row 0 = the prior);
            x_p (N+1,2), P_p (N+1,2,2): predicted mean/covariance at tick k from tick k-1
                (row 0 = the prior, unused); F (N,2,2): F[k] maps tick k to tick k+1.
        Without it, the final (x, P) only.
    """
    F = transition_matrix(dt)
    x, P = x_init.copy(), P_init.copy()
    if store_history:
        x_f, P_f, x_p, P_p = [x], [P], [x], [P]
    for k in range(n_steps):
        x, P = predict(x, P, dt, process_noise_std)
        if store_history:
            x_p.append(x)
            P_p.append(P)
        x, P = measurement_update_position(x, P, z[k + 1], R_pos)
        if not np.isnan(z_vel[k + 1]):
            x, P = _kf_update(x, P, np.array([z_vel[k + 1]]), H_VEL, np.array([[R_vel]]))
        if store_history:
            x_f.append(x)
            P_f.append(P)
    if not store_history:
        return x, P
    return dict(x_f=np.array(x_f), P_f=np.array(P_f), x_p=np.array(x_p), P_p=np.array(P_p),
                F=np.repeat(F[None], n_steps, axis=0))


def rts_smoother(fwd):
    """Rauch-Tung-Striebel backward pass (Sarkka 2013, Thm 8.2, Eq. 8.6):
        G_k = P_k F_k^T (P^-_{k+1})^-1
        m^s_k = m_k + G_k (m^s_{k+1} - m^-_{k+1})
        P^s_k = P_k + G_k (P^s_{k+1} - P^-_{k+1}) G_k^T
    started at the last tick with m^s_N = m_N, P^s_N = P_N, where m_k, P_k are
    the filter's updated mean/covariance and m^-_{k+1}, P^-_{k+1} its
    prediction of tick k+1. The smoothed covariance is symmetrized
    ((P + P^T)/2) to remove round-off asymmetry.
    Arguments:
        fwd: dict from run_forward_filter
    Returns:
        x_s: (N+1,2) smoothed means
        P_s: (N+1,2,2) smoothed covariances
        G: (N,2,2) smoother gains (G[k] links tick k+1 back to tick k)
    """
    x_f, P_f, x_p, P_p, F = fwd["x_f"], fwd["P_f"], fwd["x_p"], fwd["P_p"], fwd["F"]
    n_steps = len(F)
    x_s, P_s = x_f.copy(), P_f.copy()
    G = np.zeros((n_steps, 2, 2))
    for k in range(n_steps - 1, -1, -1):
        G[k] = np.linalg.solve(P_p[k + 1], F[k] @ P_f[k]).T  # P_k F^T P^-1 (P^-1 symmetric)
        x_s[k] = x_f[k] + G[k] @ (x_s[k + 1] - x_p[k + 1])
        P_s[k] = P_f[k] + G[k] @ (P_s[k + 1] - P_p[k + 1]) @ G[k].T
        P_s[k] = 0.5 * (P_s[k] + P_s[k].T)
    return x_s, P_s, G


def fixed_lag_estimates(fwd, G, lag):
    """Fixed-lag smoothed means: the estimate of tick k uses the measurements
    up to tick min(k + lag, N) only. It is the RTS recursion started at tick
    k + lag from the filter mean there (the filter and the gains G do not
    depend on where the data end), run back to k.
    Arguments:
        fwd: dict from run_forward_filter
        G: (N,2,2) smoother gains from rts_smoother
        lag: number of future ticks used (ticks, >= 0)
    Returns:
        x_lag: (N+1,2) fixed-lag smoothed means; lag = 0 gives the filter, lag >= N the RTS smoother
    """
    x_f, x_p = fwd["x_f"], fwd["x_p"]
    n_steps = len(G)
    x_lag = x_f.copy()
    for k in range(n_steps + 1):
        top = min(k + lag, n_steps)
        x = x_f[top]
        for j in range(top - 1, k - 1, -1):
            x = x_f[j] + G[j] @ (x - x_p[j + 1])
        x_lag[k] = x
    return x_lag


def batch_least_squares(x_init, P_init, z, z_vel, dt, process_noise_std, R_pos, R_vel):
    """Solves the whole run as one linear least-squares problem over the stacked
    states X = [x_0, ..., x_N]: minimize the sum of squared whitened residuals
        prior:    L0^-1 (x_0 - x_init)                     (P_init = L0 L0^T)
        dynamics: Lq^-1 (x_{k+1} - F x_k), k = 0..N-1      (Q = Lq Lq^T)
        position: (p_k - z_k) / sqrt(R_pos), k = 1..N
        velocity: (v_k - z_vel_k) / sqrt(R_vel), where z_vel_k is not NaN, k >= 1
    The same factors the forward filter uses. The posterior covariance is
    (A^T A)^-1 for the stacked whitened Jacobian A; its diagonal 2x2 blocks
    are returned.
    Arguments:
        x_init, P_init: prior mean (2,) and covariance (2,2) of x_0
        z: (N+1,) position measurements (m)
        z_vel: (N+1,) velocity pseudo-measurements (m/s), NaN where none
        dt: step interval (s)
        process_noise_std: assumed acceleration disturbance std-dev (m/s^2)
        R_pos: position measurement noise variance (m^2)
        R_vel: velocity pseudo-measurement noise variance (m^2/s^2)
    Returns:
        X: (N+1,2) least-squares states [p, v] (m, m/s)
        P_blocks: (N+1,2,2) posterior covariance of each tick's state
    """
    n_steps = len(z) - 1
    n = 2 * (n_steps + 1)
    F = transition_matrix(dt)
    Lq_inv = np.linalg.inv(np.linalg.cholesky(process_noise_covariance(dt, process_noise_std)))
    L0_inv = np.linalg.inv(np.linalg.cholesky(P_init))
    rows, rhs = [], []

    def add(block, cols, b):
        row = np.zeros((block.shape[0], n))
        row[:, cols] = block
        rows.append(row)
        rhs.append(b)

    add(L0_inv, slice(0, 2), L0_inv @ x_init)
    for k in range(n_steps):
        add(np.hstack([-Lq_inv @ F, Lq_inv]), slice(2 * k, 2 * k + 4), np.zeros(2))
    for k in range(1, n_steps + 1):
        add(np.array([[1.0 / np.sqrt(R_pos), 0.0]]), slice(2 * k, 2 * k + 2), np.array([z[k] / np.sqrt(R_pos)]))
        if not np.isnan(z_vel[k]):
            add(np.array([[0.0, 1.0 / np.sqrt(R_vel)]]), slice(2 * k, 2 * k + 2),
                np.array([z_vel[k] / np.sqrt(R_vel)]))
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    X = np.linalg.lstsq(A, b, rcond=None)[0]
    cov = np.linalg.inv(A.T @ A)
    P_blocks = np.array([cov[2 * k:2 * k + 2, 2 * k:2 * k + 2] for k in range(n_steps + 1)])
    return X.reshape(-1, 2), P_blocks


def phase_windows(is_anchor, is_cruise):
    """Splits the ticks into the gait windows used for the tables. Tick 0 is
    excluded from all of them (the filter's estimate there is the prior; it is
    reported on its own row).
    Arguments:
        is_anchor: (N+1,) bool, scheduled anchor ticks
        is_cruise: (N+1,) bool, steady-cruise ticks
    Returns:
        dict window name -> (N+1,) bool mask, for the names in WINDOWS
    """
    n = len(is_anchor)
    extend = ~is_anchor
    ramp = extend & ~is_cruise
    # ramp ticks before the cruise ticks of their extend phase are ramp-up, after are ramp-down
    ramp_up = np.zeros(n, dtype=bool)
    for k in np.flatnonzero(ramp):
        j = k - 1
        while j >= 0 and ramp[j]:
            j -= 1
        ramp_up[k] = j < 0 or is_anchor[j]  # the run of ramp ticks directly after an anchor tick
    masks = dict(anchor=is_anchor.copy(), **{"ramp-up": ramp_up, "cruise": is_cruise.copy(),
                                              "ramp-down": ramp & ~ramp_up})
    for m in masks.values():
        m[0] = False
    return masks


def simulate_trial(scenario, x_nominal, pos_noise_std, is_anchor, dt, process_noise_std, R_pos,
                   R_vel, init_std, rng):
    """Draws one Monte Carlo trial's truth, initial guess and measurements.
    "inchworm*": the nominal (deterministic) truth, a fresh initial error and
    fresh position noise; ZUPT value 0 on anchor ticks.
    "matched": a truth drawn from the filter's model (fresh initial state,
    process noise each step) around the same initial guess, fresh position
    noise, and an anchor-tick velocity reading v + noise with variance R_vel.
    Arguments:
        scenario: one of SCENARIOS
        x_nominal: (N+1,2) nominal true [p, v] per tick (used by "inchworm*")
        pos_noise_std: position measurement noise std-dev (m)
        is_anchor: (N+1,) bool, scheduled anchor ticks
        dt: step interval (s)
        process_noise_std: acceleration std-dev used to draw the "matched" truth (m/s^2)
        R_pos: position measurement noise variance (m^2)
        R_vel: velocity pseudo-measurement noise variance (m^2/s^2)
        init_std: (2,) std-dev of the initial guess error [p, v] (m, m/s)
        rng: numpy random number generator
    Returns:
        x_true: (N+1,2) truth for this trial
        x_init: (2,) initial guess
        z: (N+1,) position measurements
        z_vel: (N+1,) velocity pseudo-measurements, NaN off the anchor ticks
    """
    n_pts = len(x_nominal)
    if scenario == "matched":
        x_init = x_nominal[0].copy()
        x_true = np.zeros((n_pts, 2))
        x_true[0] = x_init + rng.normal(0.0, 1.0, 2) * init_std
        F = transition_matrix(dt)
        w_std = np.array([0.5 * process_noise_std * dt * dt, process_noise_std * dt])
        for k in range(n_pts - 1):
            x_true[k + 1] = F @ x_true[k] + rng.normal(0.0, 1.0, 2) * w_std
        z = x_true[:, 0] + rng.normal(0.0, np.sqrt(R_pos), n_pts)
        reading = x_true[:, 1] + rng.normal(0.0, np.sqrt(R_vel), n_pts)
        z_vel = np.where(is_anchor, reading, np.nan)
    else:
        x_true = x_nominal
        x_init = x_nominal[0] + rng.normal(0.0, 1.0, 2) * init_std
        z = x_nominal[:, 0] + rng.normal(0.0, pos_noise_std, n_pts)
        z_vel = np.where(is_anchor, 0.0, np.nan)
    return x_true, x_init, z, z_vel


def run_monte_carlo(scenario, x_nominal, dt, process_noise_std, pos_noise_std, zupt_noise_std, is_anchor,
                    init_std, n_trials, lags, rng, use_zupt=True):
    """Monte Carlo of filter and smoother (and fixed-lag smoothers) on the same
    data. For "matched" the truth is redrawn each trial too; see simulate_trial.
    Arguments:
        scenario: one of SCENARIOS; "inchworm_q015" is meant to be called with the parent's
            process_noise_std = 0.15 (the assumed value; nothing else differs)
        x_nominal: (N+1,2) nominal true states
        dt: step interval (s)
        process_noise_std: assumed (and, for "matched", true) acceleration std-dev (m/s^2)
        pos_noise_std: position noise std-dev (m)
        zupt_noise_std: ZUPT / velocity-reading noise std-dev (m/s)
        is_anchor: (N+1,) bool, anchor ticks
        init_std: (2,) initial guess std-dev [p, v]
        n_trials: number of trials
        lags: tuple of fixed-lag values (ticks) to score
        rng: numpy random number generator
        use_zupt: False drops every velocity pseudo-measurement from both the filter and the smoother
            (the data and noise draws are unchanged); used to isolate what the ZUPTs contribute
    Returns:
        dict of (N+1,) per-tick averages over trials: nees_f, nees_s; mean squared position
        error sq_pos_f, sq_pos_s (m^2); mean squared velocity error sq_vel_f, sq_vel_s (m^2/s^2);
        mean_sigma_v_f, mean_sigma_v_s (velocity std-dev the estimator reports, m/s); and
        sq_pos_lag, sq_vel_lag: dicts lag -> (N+1,) mean squared errors of the fixed-lag mean
    """
    n_steps = len(x_nominal) - 1
    n_pts = n_steps + 1
    P_init = np.diag(init_std ** 2)
    R_pos, R_vel = pos_noise_std ** 2, zupt_noise_std ** 2
    keys = ("nees_f", "nees_s", "sq_pos_f", "sq_pos_s", "sq_vel_f", "sq_vel_s", "mean_sigma_v_f",
            "mean_sigma_v_s")
    acc = {k: np.zeros(n_pts) for k in keys}
    acc["sq_pos_lag"] = {lag: np.zeros(n_pts) for lag in lags}
    acc["sq_vel_lag"] = {lag: np.zeros(n_pts) for lag in lags}
    for _ in range(n_trials):
        x_true, x_init, z, z_vel = simulate_trial(scenario, x_nominal, pos_noise_std, is_anchor, dt,
                                                  process_noise_std, R_pos, R_vel, init_std, rng)
        if not use_zupt:
            z_vel = np.full_like(z_vel, np.nan)
        fwd = run_forward_filter(x_init, P_init, z, z_vel, dt, n_steps, process_noise_std, R_pos, R_vel)
        x_s, P_s, G = rts_smoother(fwd)
        err_f, err_s = x_true - fwd["x_f"], x_true - x_s
        acc["nees_f"] += nees_per_tick(err_f, fwd["P_f"])
        acc["nees_s"] += nees_per_tick(err_s, P_s)
        acc["sq_pos_f"] += err_f[:, 0] ** 2
        acc["sq_vel_f"] += err_f[:, 1] ** 2
        acc["sq_pos_s"] += err_s[:, 0] ** 2
        acc["sq_vel_s"] += err_s[:, 1] ** 2
        acc["mean_sigma_v_f"] += np.sqrt(fwd["P_f"][:, 1, 1])
        acc["mean_sigma_v_s"] += np.sqrt(P_s[:, 1, 1])
        for lag in lags:
            x_lag = x_s if lag >= n_steps else fixed_lag_estimates(fwd, G, lag)  # lag >= N is the full RTS
            err_l = x_true - x_lag
            acc["sq_pos_lag"][lag] += err_l[:, 0] ** 2
            acc["sq_vel_lag"][lag] += err_l[:, 1] ** 2
    out = {k: acc[k] / n_trials for k in keys}
    out["sq_pos_lag"] = {lag: v / n_trials for lag, v in acc["sq_pos_lag"].items()}
    out["sq_vel_lag"] = {lag: v / n_trials for lag, v in acc["sq_vel_lag"].items()}
    return out


def nees_per_tick(err, P):
    """NEES err_k^T P_k^-1 err_k for every tick at once (the same quantity as
    utils.nees, evaluated with one batched solve instead of a Python loop).
    Arguments:
        err: (N+1,2) estimation errors x_true - x_est
        P: (N+1,2,2) reported covariances
    Returns:
        (N+1,) NEES values
    """
    return np.einsum("ki,ki->k", err, np.linalg.solve(P, err[:, :, None])[:, :, 0])


COST_REPEATS = 15


def repeated_cost(fn, *args, n_steps):
    """measure_performance repeated COST_REPEATS times (one call of a few ms is
    too short for a stable single timing).
    Arguments:
        fn, *args: callable and its arguments
        n_steps: steps per call, to normalize the time
    Returns:
        median per-step time (s), peak traced memory (bytes, same in every repeat)
    """
    runs = [measure_performance(fn, *args, n_steps=n_steps)[1:] for _ in range(COST_REPEATS)]
    return float(np.median([r[0] for r in runs])), max(r[1] for r in runs)


def window_rmse(sq, mask):
    """RMS over the masked ticks of a per-tick mean squared error."""
    return float(np.sqrt(np.mean(sq[mask])))


def print_window_table(title, mc, windows, n_trials):
    """Prints RMSE (position, velocity) and NEES of filter and smoother per window."""
    lo, hi = averaged_nees_bounds(STATE_DIM, n_trials)
    print(f"\n{title}")
    print(f"  95% bounds for the {n_trials}-trial averaged NEES (2 DoF): [{lo:.2f}, {hi:.2f}]")
    print(f"  {'window':<10s} {'ticks':>5s} | {'pos RMSE filter':>15s} {'smoother':>9s} | "
          f"{'vel RMSE filter':>15s} {'smoother':>9s} | {'NEES filter':>11s} {'smoother':>9s}")
    n_pts = len(mc["nees_f"])
    last = np.zeros(n_pts, dtype=bool)
    last[-1] = True
    first = np.zeros(n_pts, dtype=bool)
    first[0] = True
    rows = [(name, windows[name]) for name in WINDOWS]
    rows += [("ticks 1..N", np.concatenate([[False], np.ones(n_pts - 1, dtype=bool)])),
             ("tick 0", first), ("tick N", last)]
    for name, m in rows:
        print(f"  {name:<10s} {int(m.sum()):5d} | {window_rmse(mc['sq_pos_f'], m):15.4f} "
              f"{window_rmse(mc['sq_pos_s'], m):9.4f} | {window_rmse(mc['sq_vel_f'], m):15.4f} "
              f"{window_rmse(mc['sq_vel_s'], m):9.4f} | {np.mean(mc['nees_f'][m]):11.2f} "
              f"{np.mean(mc['nees_s'][m]):9.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation length in seconds")
    parser.add_argument("--dt", type=float, default=0.05, help="Filter/measurement step interval in seconds")
    parser.add_argument("--t-anchor", type=float, default=1.0, help="Anchor/dwell phase duration per cycle (s)")
    parser.add_argument("--t-extend", type=float, default=1.0,
                        help="Total extend phase duration per cycle (s), including both ramps")
    parser.add_argument("--t-ramp", type=float, default=0.2, help="Ramp-up/ramp-down duration of the extend phase (s)")
    parser.add_argument("--v-extend", type=float, default=0.1, help="Commanded cruise speed during extend (m/s)")
    parser.add_argument("--pos-noise-std", type=float, default=0.02, help="Position measurement noise std-dev (m)")
    parser.add_argument("--process-noise-std", type=float, default=0.5,
                        help="Assumed acceleration-disturbance std-dev (m/s^2); 0.5 equals the true ramp "
                             "acceleration v_extend / t_ramp (the parent's default is 0.15)")
    parser.add_argument("--under-sized-process-noise-std", type=float, default=0.15,
                        help="Assumed acceleration std-dev of the under-sized scenario (m/s^2), the parent's default")
    parser.add_argument("--zupt-noise-std", type=float, default=0.01, help="ZUPT noise std-dev (m/s)")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.05, help="Initial position guess std-dev (m)")
    parser.add_argument("--init-vel-noise-std", type=float, default=0.05, help="Initial velocity guess std-dev (m/s)")
    parser.add_argument("--n-mc-trials", type=int, default=500, help="Monte Carlo trials per scenario")
    parser.add_argument("--lags", type=int, nargs="+", default=[0, 5, 10, 20, 40],
                        help="Fixed-lag values to score (ticks); the full smoother is always added")
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
    R_pos, R_vel = args.pos_noise_std ** 2, args.zupt_noise_std ** 2
    z_vel = np.where(is_anchor, 0.0, np.nan)
    windows = phase_windows(is_anchor, is_cruise)
    t = np.arange(n_steps + 1) * args.dt
    q = args.process_noise_std

    print(f"Setup: {n_steps} ticks of {args.dt} s ({args.duration} s), cycle {args.t_anchor} s anchor + "
          f"{args.t_extend} s extend (ramps {args.t_ramp} s, cruise {args.v_extend} m/s), position noise "
          f"{args.pos_noise_std} m, ZUPT noise {args.zupt_noise_std} m/s, assumed process noise {q} m/s^2, "
          f"initial std {args.init_pos_noise_std} m / {args.init_vel_noise_std} m/s, seed {args.seed}")
    print(f"Ticks per window (tick 0 excluded): " + ", ".join(f"{n} {int(windows[n].sum())}" for n in WINDOWS))

    # Single run: filter, smoother, batch
    fwd = run_forward_filter(x_init, P_init, z, z_vel, args.dt, n_steps, q, R_pos, R_vel)
    smooth = rts_smoother(fwd)
    x_s, P_s, G = smooth
    x_f, P_f = fwd["x_f"], fwd["P_f"]
    X_batch, P_batch = batch_least_squares(x_init, P_init, z, z_vel, args.dt, q, R_pos, R_vel)

    ref_x, ref_P = run_ekf_phase_conditional_zupt(x_init, P_init, z, args.dt, n_steps, q, R_pos, R_vel, is_anchor)
    print("\nChecks (single run):")
    print(f"  filter vs inchworm_zupt_ekf.run_ekf_phase_conditional_zupt: max |dx| = "
          f"{np.max(np.abs(x_f - ref_x)):.2e}, max |dP| = {np.max(np.abs(P_f - np.array(ref_P))):.2e}")
    print(f"  RTS vs batch least squares: max |dx| = {np.max(np.abs(x_s - X_batch)):.2e}, "
          f"max |dP| = {np.max(np.abs(P_s - P_batch)):.2e}")
    min_eig = min(np.linalg.eigvalsh(P_f[k] - P_s[k]).min() for k in range(n_steps + 1))
    print(f"  smoothed covariance <= filter covariance at every tick: min eigenvalue of (P_f - P_s) = {min_eig:.2e}")
    print(f"  last tick: smoother = filter: max |dx| = {np.max(np.abs(x_s[-1] - x_f[-1])):.2e}, "
          f"max |dP| = {np.max(np.abs(P_s[-1] - P_f[-1])):.2e}")
    sig_f, sig_s = np.sqrt(P_f[:, :, :].diagonal(axis1=1, axis2=2)), np.sqrt(P_s.diagonal(axis1=1, axis2=2))
    ratio = sig_s[:, 1] / sig_f[:, 1]
    print(f"  velocity sigma, smoother / filter: min {ratio.min():.3f} (tick {int(ratio.argmin())}), "
          f"median {np.median(ratio):.3f}, max {ratio.max():.3f}")

    err_f, err_s = x_true - x_f, x_true - x_s
    print("\nSingle run (seed %d), RMSE over ticks 1..N and over all ticks:" % args.seed)
    for label, sl in (("ticks 1..N", slice(1, None)), ("all ticks", slice(0, None))):
        print(f"  {label:<10s} filter pos {np.sqrt(np.mean(err_f[sl, 0] ** 2)):.4f} m, vel "
              f"{np.sqrt(np.mean(err_f[sl, 1] ** 2)):.4f} m/s | smoother pos "
              f"{np.sqrt(np.mean(err_s[sl, 0] ** 2)):.4f} m, vel {np.sqrt(np.mean(err_s[sl, 1] ** 2)):.4f} m/s")
    print(f"  tick 0 (prior vs smoothed): velocity error filter {err_f[0, 1]:+.4f} m/s (sigma {sig_f[0, 1]:.4f}), "
          f"smoother {err_s[0, 1]:+.4f} m/s (sigma {sig_s[0, 1]:.4f})")
    k_star = 1 + int(np.argmin(ratio[1:]))
    print(f"  tick with the largest velocity-sigma gain (ticks >= 1): k = {k_star} (t = {t[k_star]:.2f} s): "
          f"filter v error {err_f[k_star, 1]:+.4f} m/s (sigma {sig_f[k_star, 1]:.4f}), smoother "
          f"{err_s[k_star, 1]:+.4f} m/s (sigma {sig_s[k_star, 1]:.4f}); position error filter "
          f"{err_f[k_star, 0]:+.4f} m (sigma {sig_f[k_star, 0]:.4f}), smoother {err_s[k_star, 0]:+.4f} m "
          f"(sigma {sig_s[k_star, 0]:.4f})")

    # Monte Carlo (common random numbers between the two inchworm scenarios)
    lags = tuple(sorted(set(args.lags) | {n_steps}))
    print(f"\nMonte Carlo, {args.n_mc_trials} trials per scenario (initial error and all measurement noise redrawn; "
          f"'matched' also redraws the process noise and the initial truth)...")
    mc = {}
    for name, q_assumed, stream in (("inchworm", q, 1), ("inchworm_q015", args.under_sized_process_noise_std, 1),
                                    ("matched", q, 2)):
        mc[name] = run_monte_carlo(name, x_true, args.dt, q_assumed, args.pos_noise_std, args.zupt_noise_std,
                                   is_anchor, init_std, args.n_mc_trials, lags if name == "inchworm" else (),
                                   np.random.default_rng([args.seed, stream]))
    titles = {"inchworm": f"Scenario inchworm: deterministic truth, assumed process noise {q} m/s^2 (default)",
              "inchworm_q015": f"Scenario inchworm_q015: same data, assumed process noise "
                               f"{args.under_sized_process_noise_std} m/s^2 (the parent's under-sized value)",
              "matched": f"Scenario matched: truth drawn from the filter's own model (process noise {q} m/s^2, "
                         f"velocity readings v + noise on anchor ticks)"}
    for name in SCENARIOS:
        print_window_table(titles[name], mc[name], windows, args.n_mc_trials)

    m_all = np.concatenate([[False], np.ones(n_steps, dtype=bool)])
    print(f"\nFixed-lag smoothing, scenario inchworm: RMSE over ticks 1..N (lag in ticks; 1 tick = {args.dt} s)")
    print(f"  {'lag':>5s} {'lag (s)':>8s} | {'pos RMSE (m)':>12s} {'vel RMSE (m/s)':>14s}")
    for lag in lags:
        tag = "  (full RTS)" if lag == n_steps else ("  (= filter)" if lag == 0 else "")
        print(f"  {lag:5d} {lag * args.dt:8.2f} | {window_rmse(mc['inchworm']['sq_pos_lag'][lag], m_all):12.4f} "
              f"{window_rmse(mc['inchworm']['sq_vel_lag'][lag], m_all):14.4f}{tag}")

    # Where in the gait cycle the smoother gains most (scenario inchworm): velocity RMSE per
    # tick-in-cycle, over interior ticks only (not the first cycle's start-up, not the last cycle,
    # where the data end and the smoother has no future to use)
    period = int(round((args.t_anchor + args.t_extend) / args.dt))
    interior = np.arange(period, n_steps + 1 - period)
    if len(interior) >= period:
        m0 = mc["inchworm"]
        in_cycle = interior % period
        rm_f = np.array([np.sqrt(np.mean(m0["sq_vel_f"][interior[in_cycle == i]])) for i in range(period)])
        rm_s = np.array([np.sqrt(np.mean(m0["sq_vel_s"][interior[in_cycle == i]])) for i in range(period)])
        print(f"\nVelocity RMSE by tick within the {period}-tick cycle, interior ticks {interior[0]}..{interior[-1]} "
              f"(tick 0 of a cycle = first anchor tick), scenario inchworm:")
        print(f"  {'tick':>4s} {'t (s)':>6s} {'phase':<9s} | {'filter':>8s} {'smoother':>9s} {'reduction':>9s} {'ratio':>6s}")
        cycle_t = np.arange(period) * args.dt
        for i in range(period):
            ph = ("anchor" if cycle_t[i] < args.t_anchor - 1e-9 else
                  "ramp-up" if cycle_t[i] - args.t_anchor < args.t_ramp - 1e-9 else
                  "ramp-down" if cycle_t[i] - args.t_anchor >= args.t_extend - args.t_ramp - 1e-9 else "cruise")
            print(f"  {i:4d} {cycle_t[i]:6.2f} {ph:<9s} | {rm_f[i]:8.4f} {rm_s[i]:9.4f} {rm_f[i] - rm_s[i]:9.4f} "
                  f"{rm_s[i] / rm_f[i]:6.2f}")
        top = np.argsort(rm_f - rm_s)[::-1][:3]
        print("  largest absolute reduction at in-cycle ticks " + ", ".join(
            f"{int(i)} (t = {cycle_t[i]:.2f} s: {rm_f[i]:.4f} -> {rm_s[i]:.4f})" for i in top))
        print("  in-cycle ticks (of %d) where the smoother's RMSE is larger than the filter's (interior ticks):" % period)
        for name in SCENARIOS:
            mm = mc[name]
            worse = {}
            for qty, key in (("velocity", "sq_vel"), ("position", "sq_pos")):
                f_i = np.array([np.sqrt(np.mean(mm[key + "_f"][interior[in_cycle == i]])) for i in range(period)])
                s_i = np.array([np.sqrt(np.mean(mm[key + "_s"][interior[in_cycle == i]])) for i in range(period)])
                worse[qty] = np.flatnonzero(s_i > f_i)
            print(f"    {name:<14s} velocity: {len(worse['velocity']):2d} {worse['velocity'].tolist()} | "
                  f"position: {len(worse['position']):2d} {worse['position'].tolist()}")

        # What the coming ZUPT contributes: the same trials with the ZUPTs removed from both estimators
        mc_nz = run_monte_carlo("inchworm", x_true, args.dt, q, args.pos_noise_std, args.zupt_noise_std, is_anchor,
                                init_std, args.n_mc_trials, (), np.random.default_rng([args.seed, 1]), use_zupt=False)
        nz_f = np.array([np.sqrt(np.mean(mc_nz["sq_vel_f"][interior[in_cycle == i]])) for i in range(period)])
        nz_s = np.array([np.sqrt(np.mean(mc_nz["sq_vel_s"][interior[in_cycle == i]])) for i in range(period)])
        print("  same trials with the ZUPTs removed from both estimators (velocity RMSE filter -> smoother):")
        for i in top:
            print(f"    in-cycle tick {int(i):2d}: with ZUPT {rm_f[i]:.4f} -> {rm_s[i]:.4f}, without {nz_f[i]:.4f} -> {nz_s[i]:.4f}")

    # Cost
    time_nostore, mem_nostore = repeated_cost(
        run_forward_filter, x_init, P_init, z, z_vel, args.dt, n_steps, q, R_pos, R_vel, False, n_steps=n_steps)
    time_fwd, mem_fwd = repeated_cost(
        run_forward_filter, x_init, P_init, z, z_vel, args.dt, n_steps, q, R_pos, R_vel, True, n_steps=n_steps)
    time_bwd, mem_bwd = repeated_cost(rts_smoother, fwd, n_steps=n_steps)
    floats_per_tick = 2 + 4 + 2 + 4 + 4  # x_f, P_f, x_p, P_p, F
    print(f"\nCost, single run ({COST_REPEATS} repeats each: median per-step time, peak traced memory of one call):")
    print(f"  forward filter, current state only : {time_nostore * 1e6:8.2f} us/step | peak mem "
          f"{mem_nostore / 1024.0:8.3f} KB")
    print(f"  forward filter, history stored     : {time_fwd * 1e6:8.2f} us/step | peak mem "
          f"{mem_fwd / 1024.0:8.3f} KB")
    print(f"  RTS backward pass                  : {time_bwd * 1e6:8.2f} us/step | peak mem "
          f"{mem_bwd / 1024.0:8.3f} KB (on top of the stored history)")
    print(f"  stored history: {floats_per_tick} floats per tick x {n_steps + 1} ticks = "
          f"{floats_per_tick * (n_steps + 1) * 8 / 1024.0:.1f} KB; a plain filter keeps 6 floats (x, P)")

    # Figure
    fig, ((ax_v, ax_p), (ax_r, ax_n)) = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    edges = np.diff(np.concatenate([[False], is_anchor, [False]]).astype(int))
    for ax in (ax_v, ax_p, ax_r, ax_n):
        for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            ax.axvspan(t[a], t[b - 1] + args.dt, color="gold", alpha=0.18, lw=0)
        ax.set_xlim(0.0, t[-1])
    ax_v.plot(t, x_true[:, 1], "g--", lw=2, label="Ground truth")
    ax_v.plot(t, x_f[:, 1], color="tab:blue", label="filter")
    ax_v.fill_between(t, x_f[:, 1] - 2 * sig_f[:, 1], x_f[:, 1] + 2 * sig_f[:, 1], color="tab:blue", alpha=0.18)
    ax_v.plot(t, x_s[:, 1], color="tab:red", label="RTS smoother")
    ax_v.fill_between(t, x_s[:, 1] - 2 * sig_s[:, 1], x_s[:, 1] + 2 * sig_s[:, 1], color="tab:red", alpha=0.25)
    ax_v.set_ylabel("Velocity v (m/s), band = ±2σ")
    ax_v.set_title("One run (shaded gold = anchor ticks)")
    ax_v.legend(loc="upper right", fontsize=8, ncol=3)

    ax_p.plot(t, err_f[:, 0], color="tab:blue", label="filter")
    ax_p.fill_between(t, -2 * sig_f[:, 0], 2 * sig_f[:, 0], color="tab:blue", alpha=0.15)
    ax_p.plot(t, err_s[:, 0], color="tab:red", label="RTS smoother")
    ax_p.fill_between(t, -2 * sig_s[:, 0], 2 * sig_s[:, 0], color="tab:red", alpha=0.25)
    ax_p.set_ylabel("Position error (m), band = ±2σ")
    ax_p.set_title("One run: position error")
    ax_p.legend(loc="upper right", fontsize=8)

    m = mc["inchworm"]
    ax_r.plot(t, np.sqrt(m["sq_vel_f"]), color="tab:blue", label="filter")
    ax_r.plot(t, np.sqrt(m["sq_vel_s"]), color="tab:red", label="RTS smoother")
    ax_r.set_ylabel("Velocity RMSE over trials (m/s)")
    ax_r.set_xlabel("Time (s)")
    ax_r.set_title(f"Monte Carlo ({args.n_mc_trials} trials): velocity RMSE per tick")
    ax_r.legend(loc="upper right", fontsize=8)

    lo, hi = averaged_nees_bounds(STATE_DIM, args.n_mc_trials)
    ax_n.plot(t, m["nees_f"], color="tab:blue", label="filter")
    ax_n.plot(t, m["nees_s"], color="tab:red", label="RTS smoother")
    ax_n.plot(t, mc["matched"]["nees_f"], color="tab:blue", ls=":", label="filter, matched truth")
    ax_n.plot(t, mc["matched"]["nees_s"], color="tab:red", ls=":", label="smoother, matched truth")
    ax_n.axhline(STATE_DIM, color="black", lw=1)
    ax_n.axhspan(lo, hi, color="black", alpha=0.10, lw=0, label=f"95% band, {args.n_mc_trials}-trial average")
    ax_n.set_ylabel("Monte Carlo avg. NEES (2 DoF)")
    ax_n.set_xlabel("Time (s)")
    ax_n.set_title("Consistency: expected value 2")
    ax_n.legend(loc="upper right", fontsize=7, ncol=2)

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
