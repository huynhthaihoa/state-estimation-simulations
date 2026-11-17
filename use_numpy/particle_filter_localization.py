'''
1D corridor global localization: a particle filter (PF) against an EKF and a
UKF, on the same data.

Scene. A robot drives right along a corridor of length `L` at a commanded speed
`v` for `T` steps of `dt` seconds (open-loop odometry `u = v * dt` per step, known
to every filter). It does not know where it started: the start is drawn uniformly
in the **start zone** `[0, L - D]`, where `D = v * dt * T` is the total commanded
travel, so the robot never leaves the corridor. Every filter starts from the same
belief about the start (the prior).

Model (the truth follows the PF's model exactly; the noise sources are all drawn
again in every Monte Carlo trial):

  - Motion:  x_{k+1} = x_k + u + w_k,  w_k ~ N(0, sigma_u^2),
             sigma_u = sqrt((motion_noise_factor * |u|)^2 + motion_noise_floor^2).
  - Profile sensor (main sensor): z_k = f(x_k) + n_k,  n_k ~ N(0, sigma_z^2), with
             f(x) = sum_k exp(-(x - m_k)^2 / (2 s^2)),
    one Gaussian bump of width `s` at each door position `m_k`. It is the same
    scalar `z = h(x) + noise` for all three filters and never says WHICH door it
    sees. `f` repeats at every door, so the posterior has several hills until the
    pattern of door spacings has been seen. Door positions, `s` and `sigma_z` are
    design choices (no source gives them).

The filters:

  - PF: the sampling-importance-resampling filter of Thrun et al. (2005) Table
    4.3 (Gordon et al. 1993 bootstrap filter): move each particle with the motion
    model, weight it by `N(z; f(x), sigma_z^2)`, resample. Resampling uses the
    low-variance sampler (Thrun Table 4.4) and runs when the effective sample
    size `N_eff = 1 / sum(w_i^2)` drops below `resample_threshold * N` (default
    N/2); `--resample-threshold 1.01` resamples every step (Gordon's version).
    Initial particles: uniform over the start zone. No random-particle injection.
  - EKF: scalar Kalman filter, `H = f'(x)` evaluated at the predicted mean.
  - UKF: scalar unscented transform with the repo's sigma-point convention
    (`unscented_weights`, `unscented_sigma_offsets` from utils.py; alpha = 1,
    beta = 2, kappa = 3 - n as in pointcloud_pose_tracking.py).
  - EKF and UKF start from the Gaussian that matches the uniform prior: mean =
    start-zone centre, variance = (zone width)^2 / 12. A design choice.

Labeled variant (`--ml-variant`): the textbook workaround for unknown door
identity, EKF with maximum-likelihood correspondence (Thrun Sec. 7.5.1, Table
7.3), and a UKF version of the same rule. It needs a different sensor: the signed
offset to the nearest door, `z = m_j - x + n`, `n ~ N(0, sigma_c^2)`, `j` the door
nearest to the TRUE position, and only reported if `|m_j - x| <= ml_range`
(no detection otherwise: no update). In 1D, Table 7.3 reduces to: for each door
`k`, `zhat_k = m_k - x`, `H = -1`, `S = P + sigma_c^2`; use the door
`j = argmax_k N(z; zhat_k, S)` only (no gate). Because the measurement is linear
given `j`, the UKF-ML result must equal the EKF-ML result.

Scores (all numbers come from the Monte Carlo runs below):

  - final error: `x_hat_T - x_T` at the last step (m); `x_hat` is the filter's
    mean (the PF's weighted mean). RMSE and median |error| over trials.
  - wrong-door rate: fraction of trials with `|final error| > wrong_door_threshold`
    (default: half the smallest door spacing, so an error beyond it could be
    a shift to a neighbouring door).
  - one hill (PF): at step k the belief is one hill if at least `hill_mass` of the
    weight (default 0.9) lies within `hill_halfwidth` (default a quarter of the
    smallest door spacing) of the weighted mean. Time to one hill = first step
    from which this holds at every later step, in seconds; trials that never get
    there are counted as not converged (censored), not averaged in.
  - "right-door" NEES: the same NEES over the trials in which that filter's final
    error is within the wrong-door threshold (a different subset per filter,
    its size is printed); it separates "consistent when right" from the rare
    confident wrong-door trial that dominates an all-trials average.
  - "true mass" (PF): weight within `wrong_door_threshold` of the truth, at the
    final step.
  - NEES in 1D: `(x_T - x_hat)^2 / P` (consistent filters average 1). Averaged
    over trials per step, then reported as the mean over the last `tail_steps`
    steps, with the 95% bound for one averaged step from `averaged_nees_bounds`
    (approximate for the mean of several correlated steps). Reported over all
    trials, and over the "converged" trials: those where the PF is one hill from
    the start of the tail window on (the same trial subset for all filters).
    PF variance = weighted variance of the particles.
  - ESS: `N_eff / N` before resampling, mean and minimum over steps and trials;
    resampling count per trial.

Known-start diagnostic (`--prior-std S`): every filter starts from a Gaussian of
std `S` centred on the true start plus a draw of that std (so the truth is a draw
from the prior) instead of the uniform start-zone prior. It isolates how much of
a filter's behaviour comes from the global start rather than from the profile
sensor itself.

Sweeps (`--sweep NAME`, repeatable; none run unless asked): particle count,
bump width `s`, `sigma_z`, motion-noise factor, resampling rule, door layout.
Run `--help` for every flag; no flag turns on "everything". Sources: Thrun,
Burgard, Fox (2005) Probabilistic Robotics Tables 4.3, 4.4, 7.3; Gordon, Salmond,
Smith (1993) IEE Proc. F 140(2) Sec. 3.1; Arulampalam et al. (2002) eq. 51 (N_eff);
Doucet, Johansen (2008) for the N/2 resampling trigger; Julier, Uhlmann (1997) for
the UKF.
All sweeps reuse the same standard-normal noise draws for every setting (common
random numbers), so a setting changes results only through the knob itself.
'''

import argparse
import dataclasses
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from state_estimation_simulations.utils import (
    measure_performance, averaged_nees_bounds, unscented_weights, unscented_sigma_offsets)

DEFAULT_DOORS = (3.0, 7.0, 12.0, 18.0, 25.0)   # spacings 4, 5, 6, 7 m: every spacing is different
LAYOUTS = {"5 doors, all spacings differ": DEFAULT_DOORS,
           "5 doors, spacing 4 m twice": (4.0, 9.0, 13.0, 20.0, 24.0),
           "3 doors (4, 10, 22)": (4.0, 10.0, 22.0)}
SWEEPS = ("particles", "width", "sigma_z", "motion", "resample", "layout")
COLORS = {"PF": "tab:blue", "EKF": "tab:red", "UKF": "tab:purple",
          "EKF-ML": "tab:orange", "UKF-ML": "tab:green"}


@dataclass(frozen=True)
class Config:
    """All model, filter and scoring parameters of one setting (units in the
    CLI help of `build_parser`)."""
    length: float = 30.0
    doors: tuple = DEFAULT_DOORS
    width: float = 0.3
    sigma_z: float = 0.1
    speed: float = 0.5
    dt: float = 0.5
    steps: int = 48
    motion_factor: float = 0.05
    motion_floor: float = 0.01
    n_particles: int = 1000
    resample_threshold: float = 0.5
    ukf_alpha: float = 1.0
    ukf_beta: float = 2.0
    ukf_kappa: float = 2.0
    hill_mass: float = 0.9
    hill_halfwidth: float = None   # None: a quarter of the smallest door spacing (m)
    wrong_door_threshold: float = None   # None: half the smallest door spacing (m)
    tail_steps: int = 10
    ml_range: float = 3.0
    sigma_c: float = 0.1
    prior_std: float = None   # None: global start (uniform over the zone); else known-start diagnostic (m)

    @property
    def travel(self):
        return self.speed * self.dt * self.steps

    @property
    def zone(self):
        return 0.0, self.length - self.travel

    @property
    def min_gap(self):
        return float(np.min(np.diff(np.sort(np.asarray(self.doors, dtype=float)))))

    @property
    def hill_h(self):
        return self.hill_halfwidth if self.hill_halfwidth is not None else 0.25 * self.min_gap

    @property
    def wrong_thr(self):
        return self.wrong_door_threshold if self.wrong_door_threshold is not None else 0.5 * self.min_gap


# --- sensor model -----------------------------------------------------------------------

def door_profile(x, doors, width):
    """Profile sensor mean f(x): one unit-height Gaussian bump per door.
    Arguments:
        x: positions (m), any shape
        doors: (K,) door positions (m)
        width: bump standard deviation s (m)
    Returns:
        f: same shape as x (dimensionless, 0 far from doors, 1 at a door)
    """
    d = np.asarray(x, dtype=float)[..., None] - np.asarray(doors, dtype=float)
    return np.exp(-d ** 2 / (2.0 * width ** 2)).sum(axis=-1)


def door_profile_grad(x, doors, width):
    """Analytic derivative f'(x) of door_profile (the EKF's H).
    Arguments:
        x: positions (m), any shape
        doors: (K,) door positions (m)
        width: bump standard deviation s (m)
    Returns:
        df/dx: same shape as x (1/m)
    """
    d = np.asarray(x, dtype=float)[..., None] - np.asarray(doors, dtype=float)
    return (-d / width ** 2 * np.exp(-d ** 2 / (2.0 * width ** 2))).sum(axis=-1)


def motion_noise_std(u, factor, floor):
    """Per-step motion noise sigma_u = sqrt((factor*|u|)^2 + floor^2).
    Arguments:
        u: commanded displacements (m), array
        factor: proportional noise (m of noise per m travelled)
        floor: noise that does not scale with travel (m)
    Returns:
        sigma_u: same shape as u (m)
    """
    return np.sqrt((factor * np.abs(u)) ** 2 + floor ** 2)


# --- data generation --------------------------------------------------------------------

def draw_noise(seed, n_trials, steps):
    """Standard (unit-scale) random draws for every Monte Carlo trial. Scaling
    them by sigma later keeps the draws identical across sweep settings.
    One independent stream per trial and per source (SeedSequence.spawn), so no
    source's draws depend on how another is used.
    Arguments:
        seed: integer seed
        n_trials: number of Monte Carlo trials
        steps: number of motion steps T
    Returns:
        list of dicts with keys u0 (uniform start in [0,1)), eps_w (T,) motion,
        eps_z (T+1,) profile sensor, eps_c (T+1,) offset sensor, eps_p (prior-mean
        error of the known-start diagnostic), pf (3 child SeedSequences: initial
        particles, particle motion, resampling offset)
    """
    trials = []
    for ss in np.random.SeedSequence(seed).spawn(n_trials):
        s_x0, s_w, s_z, s_c, s_pf, s_p = ss.spawn(6)
        trials.append(dict(
            u0=np.random.default_rng(s_x0).random(),
            eps_w=np.random.default_rng(s_w).standard_normal(steps),
            eps_z=np.random.default_rng(s_z).standard_normal(steps + 1),
            eps_c=np.random.default_rng(s_c).standard_normal(steps + 1),
            pf=s_pf.spawn(3),
            eps_p=np.random.default_rng(s_p).standard_normal()))
    return trials


def generate_trial(cfg, noise):
    """Truth and measurements of one trial.
    Arguments:
        cfg: Config
        noise: one entry of draw_noise
    Returns:
        x_true: (T+1,) true positions (m)
        z: (T+1,) profile sensor readings
        z_ml: (T+1,) signed offset to the nearest door (m), NaN where nothing is within ml_range
        u: (T,) commanded displacements per step (m)
        sigma_u: (T,) motion noise std per step (m)
    """
    lo, hi = cfg.zone
    u = np.full(cfg.steps, cfg.speed * cfg.dt)
    sigma_u = motion_noise_std(u, cfg.motion_factor, cfg.motion_floor)
    x_true = np.empty(cfg.steps + 1)
    x_true[0] = lo + (hi - lo) * noise["u0"]
    x_true[1:] = x_true[0] + np.cumsum(u + sigma_u * noise["eps_w"])
    z = door_profile(x_true, cfg.doors, cfg.width) + cfg.sigma_z * noise["eps_z"]
    doors = np.asarray(cfg.doors, dtype=float)
    offsets = doors[np.argmin(np.abs(doors[None, :] - x_true[:, None]), axis=1)] - x_true
    z_ml = np.where(np.abs(offsets) <= cfg.ml_range, offsets + cfg.sigma_c * noise["eps_c"], np.nan)
    return x_true, z, z_ml, u, sigma_u


# --- filters ----------------------------------------------------------------------------

def effective_sample_size(w):
    """N_eff = 1 / sum(w_i^2) for normalized weights (Arulampalam et al. 2002, eq. 51).
    Arguments:
        w: (N,) normalized weights
    Returns:
        N_eff in [1, N]
    """
    return 1.0 / float(np.sum(w ** 2))


def low_variance_resample(w, r):
    """Low-variance (systematic) resampler, Thrun et al. Table 4.4: one random
    offset r in [0, 1/N), then N equally spaced pointers on the cumulative weights.
    Arguments:
        w: (N,) normalized weights
        r: offset in [0, 1/N)
    Returns:
        idx: (N,) indices of the selected particles
    """
    n = len(w)
    c = np.cumsum(w)
    u = r + np.arange(n) / n
    return np.minimum(np.searchsorted(c, u, side="left"), n - 1)


def run_pf(z, u, sigma_u, h_fn, sigma_z, x_init, rng_motion, rng_resample, resample_threshold,
           hill_halfwidth, x_true=None, true_halfwidth=None, record=False):
    """Bootstrap / SIR particle filter (Thrun Table 4.3 with Table 4.4 resampler).
    At step 0 the measurement is applied to the initial particles; at step k > 0
    the particles are first moved with the motion model. Weights carry over
    between resampling events (w_k proportional to w_{k-1} p(z_k|x_k), Thrun eq.
    4.36) and are reset to 1/N after resampling. The random offset of the
    resampler is drawn every step (also when no resampling happens), so the
    stream does not depend on the resampling rule.
    Arguments:
        z: (T+1,) measurements
        u: (T,) commanded displacements (m)
        sigma_u: (T,) motion noise std (m)
        h_fn: measurement function x -> h(x), vectorized over particles
        sigma_z: measurement noise std (same units as z)
        x_init: (N,) initial particle positions (m)
        rng_motion: Generator for the particle motion noise
        rng_resample: Generator for the resampler offset
        resample_threshold: resample when N_eff < resample_threshold * N (> 1: every step)
        hill_halfwidth: half-width of the one-hill window around the weighted mean (m)
        x_true: optional (T+1,) truth, for the true-mass score
        true_halfwidth: half-width of the window around the truth for true mass (m)
        record: if True also return per-step particles and weights
    Returns:
        dict with mean (T+1,), var (T+1,) (weighted, before resampling), ess (T+1,),
        resampled (T+1,) bool, hill_mass (T+1,), true_mass (T+1,) (NaN without x_true)
        and, with record, x_hist (T+1,N) and w_hist (T+1,N)
    """
    n = len(x_init)
    t1 = len(z)
    x = np.array(x_init, dtype=float)
    logw = np.full(n, -np.log(n))
    out = dict(mean=np.empty(t1), var=np.empty(t1), ess=np.empty(t1), resampled=np.zeros(t1, bool),
               hill_mass=np.empty(t1), true_mass=np.full(t1, np.nan))
    if record:
        out["x_hist"], out["w_hist"] = np.empty((t1, n)), np.empty((t1, n))
    for k in range(t1):
        if k > 0:
            x = x + u[k - 1] + sigma_u[k - 1] * rng_motion.standard_normal(n)
        r = rng_resample.random() / n
        logw = logw - 0.5 * ((z[k] - h_fn(x)) / sigma_z) ** 2
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        logw = np.log(np.maximum(w, 1e-300))
        m = float(np.sum(w * x))
        out["mean"][k] = m
        out["var"][k] = float(np.sum(w * (x - m) ** 2))
        out["ess"][k] = effective_sample_size(w)
        out["hill_mass"][k] = float(w[np.abs(x - m) <= hill_halfwidth].sum())
        if x_true is not None:
            out["true_mass"][k] = float(w[np.abs(x - x_true[k]) <= true_halfwidth].sum())
        if record:
            out["x_hist"][k], out["w_hist"][k] = x, w
        if out["ess"][k] < resample_threshold * n:
            x = x[low_variance_resample(w, r)]
            logw = np.full(n, -np.log(n))
            out["resampled"][k] = True
    return out


def run_ekf(z, u, sigma_u, h_fn, dh_fn, sigma_z, mean0, var0):
    """Scalar EKF: predict with x + u, P + sigma_u^2; update with the
    linearization H = h'(x_pred) (Joseph form of the covariance update).
    Arguments:
        z: (T+1,) measurements; u: (T,) displacements (m); sigma_u: (T,) motion noise std (m)
        h_fn, dh_fn: measurement function and its derivative
        sigma_z: measurement noise std
        mean0, var0: initial mean (m) and variance (m^2)
    Returns:
        mean (T+1,), var (T+1,) after each measurement update
    """
    t1 = len(z)
    m, p = mean0, var0
    mean, var = np.empty(t1), np.empty(t1)
    for k in range(t1):
        if k > 0:
            m, p = m + u[k - 1], p + sigma_u[k - 1] ** 2
        h = dh_fn(m)
        s = h * p * h + sigma_z ** 2
        gain = p * h / s
        m = m + gain * (z[k] - h_fn(m))
        p = (1.0 - gain * h) ** 2 * p + gain ** 2 * sigma_z ** 2
        mean[k], var[k] = m, p
    return mean, var


def run_ukf(z, u, sigma_u, h_fn, sigma_z, mean0, var0, alpha, beta, kappa):
    """Scalar UKF (scaled unscented transform, n = 1). The motion model is
    linear, so the prediction is exact; the update draws 2n+1 sigma points
    around the predicted mean, passes them through h and forms the usual
    innovation covariance and cross covariance.
    Arguments:
        z, u, sigma_u, h_fn, sigma_z, mean0, var0: as in run_ekf
        alpha, beta, kappa: unscented-transform parameters (utils.unscented_weights)
    Returns:
        mean (T+1,), var (T+1,) after each measurement update
    """
    t1 = len(z)
    lam, w_m, w_c = unscented_weights(1, alpha, beta, kappa)
    m, p = mean0, var0
    mean, var = np.empty(t1), np.empty(t1)
    for k in range(t1):
        if k > 0:
            m, p = m + u[k - 1], p + sigma_u[k - 1] ** 2
        off = unscented_sigma_offsets(np.array([[p]]), lam)[:, 0]
        zs = h_fn(m + off)
        zhat = float(w_m @ zs)
        pzz = float(w_c @ (zs - zhat) ** 2) + sigma_z ** 2
        pxz = float(w_c @ (off * (zs - zhat)))
        gain = pxz / pzz
        m = m + gain * (z[k] - zhat)
        p = p - gain ** 2 * pzz
        mean[k], var[k] = m, p
    return mean, var


def ml_door_index(z, predicted, s):
    """Maximum-likelihood correspondence (Thrun Table 7.3, line 16) in 1D:
    argmax_k N(z; zhat_k, S_k).
    Arguments:
        z: measured signed offset to a door (m)
        predicted: (K,) predicted offsets zhat_k (m)
        s: (K,) innovation variances S_k (m^2)
    Returns:
        index j of the most likely door
    """
    return int(np.argmax(-0.5 * np.log(2.0 * np.pi * s) - 0.5 * (z - predicted) ** 2 / s))


def run_ekf_ml(z_ml, u, sigma_u, doors, sigma_c, mean0, var0):
    """EKF with ML door association (Thrun Table 7.3 in 1D) on the offset
    sensor z = m_j - x + n. NaN readings (no door in range) skip the update.
    Arguments:
        z_ml: (T+1,) offsets (m), NaN = no detection
        u, sigma_u: motion as in run_ekf; doors: (K,) positions (m); sigma_c: offset noise std (m)
        mean0, var0: initial mean (m) and variance (m^2)
    Returns:
        mean (T+1,), var (T+1,), chosen door index (T+1,) (-1 where no update)
    """
    t1 = len(z_ml)
    doors = np.asarray(doors, dtype=float)
    m, p = mean0, var0
    mean, var, picked = np.empty(t1), np.empty(t1), np.full(t1, -1)
    for k in range(t1):
        if k > 0:
            m, p = m + u[k - 1], p + sigma_u[k - 1] ** 2
        if not np.isnan(z_ml[k]):
            s = np.full(len(doors), p + sigma_c ** 2)
            j = ml_door_index(z_ml[k], doors - m, s)
            gain = -p / s[j]
            m = m + gain * (z_ml[k] - (doors[j] - m))
            p = (1.0 + gain) ** 2 * p + gain ** 2 * sigma_c ** 2
            picked[k] = j
        mean[k], var[k] = m, p
    return mean, var, picked


def run_ukf_ml(z_ml, u, sigma_u, doors, sigma_c, mean0, var0, alpha, beta, kappa):
    """UKF with the same ML association rule: for each door the sigma points
    give zhat_k and S_k; the best door's cross covariance updates the state.
    Arguments:
        as run_ekf_ml, plus alpha, beta, kappa (utils.unscented_weights)
    Returns:
        mean (T+1,), var (T+1,), chosen door index (T+1,) (-1 where no update)
    """
    t1 = len(z_ml)
    doors = np.asarray(doors, dtype=float)
    lam, w_m, w_c = unscented_weights(1, alpha, beta, kappa)
    m, p = mean0, var0
    mean, var, picked = np.empty(t1), np.empty(t1), np.full(t1, -1)
    for k in range(t1):
        if k > 0:
            m, p = m + u[k - 1], p + sigma_u[k - 1] ** 2
        if not np.isnan(z_ml[k]):
            off = unscented_sigma_offsets(np.array([[p]]), lam)[:, 0]
            zs = doors[:, None] - (m + off)[None, :]          # (K, 2n+1)
            zhat = zs @ w_m
            s = (w_c * (zs - zhat[:, None]) ** 2).sum(axis=1) + sigma_c ** 2
            j = ml_door_index(z_ml[k], zhat, s)
            pxz = float(w_c @ (off * (zs[j] - zhat[j])))
            gain = pxz / s[j]
            m = m + gain * (z_ml[k] - zhat[j])
            p = p - gain ** 2 * s[j]
            picked[k] = j
        mean[k], var[k] = m, p
    return mean, var, picked


# --- Monte Carlo ------------------------------------------------------------------------

def prior_gaussian(cfg, noise=None, x_start=None):
    """Gaussian prior of the EKF/UKF. Global start: matches the uniform start-zone
    prior, (mean, variance) = (centre, width^2/12). Known-start diagnostic
    (cfg.prior_std set): mean = true start + prior_std * eps_p, variance prior_std^2,
    so the truth is a draw from the prior.
    Arguments:
        cfg: Config; noise: one entry of draw_noise (needed with prior_std); x_start: true start (m)
    Returns:
        (mean (m), variance (m^2))
    """
    if cfg.prior_std is not None:
        return x_start + cfg.prior_std * noise["eps_p"], cfg.prior_std ** 2
    lo, hi = cfg.zone
    return 0.5 * (lo + hi), (hi - lo) ** 2 / 12.0


def pf_on_trial(cfg, noise, data, record=False):
    """Runs the PF on one trial; every random stream is rebuilt from the trial's
    SeedSequences, so calling it twice gives identical results.
    Arguments:
        cfg: Config; noise: one entry of draw_noise; data: output of generate_trial; record: keep particle history
    Returns:
        run_pf output dict
    """
    x_true, z, _, u, sigma_u = data
    s_init, s_mot, s_res = noise["pf"]
    rng_init = np.random.default_rng(s_init)
    if cfg.prior_std is None:
        x_init = rng_init.uniform(*cfg.zone, cfg.n_particles)
    else:
        m0, p0 = prior_gaussian(cfg, noise, x_true[0])
        x_init = m0 + np.sqrt(p0) * rng_init.standard_normal(cfg.n_particles)
    return run_pf(z, u, sigma_u, lambda x: door_profile(x, cfg.doors, cfg.width), cfg.sigma_z, x_init,
                  np.random.default_rng(s_mot), np.random.default_rng(s_res), cfg.resample_threshold, cfg.hill_h,
                  x_true=x_true, true_halfwidth=cfg.wrong_thr, record=record)


def run_filters_on_trial(cfg, noise, data, ml_variant=False):
    """Runs every filter on one trial's data.
    Arguments:
        cfg: Config
        noise: one entry of draw_noise (provides the PF streams)
        data: output of generate_trial
        ml_variant: also run EKF-ML and UKF-ML
    Returns:
        dict name -> dict(mean, var, ...); the PF entry also has ess, resampled, hill_mass, true_mass
    """
    x_true, z, z_ml, u, sigma_u = data
    doors, s = np.asarray(cfg.doors, dtype=float), cfg.width
    h_fn = lambda x: door_profile(x, doors, s)
    dh_fn = lambda x: door_profile_grad(x, doors, s)
    m0, p0 = prior_gaussian(cfg, noise, x_true[0])
    res = {"PF": pf_on_trial(cfg, noise, data)}
    mean, var = run_ekf(z, u, sigma_u, h_fn, dh_fn, cfg.sigma_z, m0, p0)
    res["EKF"] = dict(mean=mean, var=var)
    mean, var = run_ukf(z, u, sigma_u, h_fn, cfg.sigma_z, m0, p0, cfg.ukf_alpha, cfg.ukf_beta, cfg.ukf_kappa)
    res["UKF"] = dict(mean=mean, var=var)
    if ml_variant:
        mean, var, pk = run_ekf_ml(z_ml, u, sigma_u, doors, cfg.sigma_c, m0, p0)
        res["EKF-ML"] = dict(mean=mean, var=var, picked=pk)
        mean, var, pk = run_ukf_ml(z_ml, u, sigma_u, doors, cfg.sigma_c, m0, p0,
                                   cfg.ukf_alpha, cfg.ukf_beta, cfg.ukf_kappa)
        res["UKF-ML"] = dict(mean=mean, var=var, picked=pk)
    return res


def run_monte_carlo(cfg, noises, ml_variant=False):
    """All trials, all filters.
    Arguments:
        cfg: Config
        noises: list from draw_noise
        ml_variant: include the ML-association variant
    Returns:
        x_true: (n, T+1) truths
        res: dict name -> dict of stacked (n, T+1) arrays (mean, var and the PF extras)
    """
    xs, per = [], []
    for noise in noises:
        data = generate_trial(cfg, noise)
        xs.append(data[0])
        per.append(run_filters_on_trial(cfg, noise, data, ml_variant))
    res = {name: {key: np.stack([r[name][key] for r in per]) for key in per[0][name]} for name in per[0]}
    return np.stack(xs), res


# --- scores -----------------------------------------------------------------------------

def one_hill_step(hill_mass, frac):
    """First step from which hill_mass >= frac holds at every later step.
    Arguments:
        hill_mass: (n, T+1) PF weight within the hill window around the weighted mean
        frac: required weight fraction
    Returns:
        (n,) int array of steps, -1 where the belief never stays one hill
    """
    stays = np.logical_and.accumulate((hill_mass >= frac)[:, ::-1], axis=1)[:, ::-1]
    first = np.argmax(stays, axis=1)
    return np.where(stays[:, -1], first, -1)


def tail_nees(err, var, mask, tail):
    """NEES averaged over the trials in `mask` per step, then over the last `tail` steps.
    Arguments:
        err: (n, T+1) errors x_hat - x (m); var: (n, T+1) filter variances (m^2)
        mask: (n,) bool trial subset; tail: number of final steps
    Returns:
        (mean over the tail of the trial-averaged NEES, number of tail steps inside the 95% bound,
         lower bound, upper bound); NaNs if the subset is empty
    """
    n = int(mask.sum())
    if n == 0:
        return np.nan, 0, np.nan, np.nan
    per_step = np.mean(err[mask][:, -tail:] ** 2 / np.maximum(var[mask][:, -tail:], 1e-12), axis=0)
    lo, hi = averaged_nees_bounds(1, n)
    return float(per_step.mean()), int(np.sum((per_step >= lo) & (per_step <= hi))), lo, hi


def score(cfg, x_true, res):
    """Scores of every filter (definitions in the module docstring).
    Arguments:
        cfg: Config; x_true: (n, T+1) truths; res: output of run_monte_carlo
    Returns:
        dict name -> dict of scalars; PF also has one-hill, ESS and resampling scores
    """
    tail, thr = cfg.tail_steps, cfg.wrong_thr
    hill = one_hill_step(res["PF"]["hill_mass"], cfg.hill_mass)
    conv = (hill >= 0) & (hill <= cfg.steps + 1 - tail)
    out = {}
    for name, r in res.items():
        err = r["mean"] - x_true
        sc = dict(final_rmse=float(np.sqrt(np.mean(err[:, -1] ** 2))),
                  final_median=float(np.median(np.abs(err[:, -1]))),
                  wrong=float(np.mean(np.abs(err[:, -1]) > thr)),
                  tail_rmse=float(np.sqrt(np.mean(err[:, -tail:] ** 2))))
        sc["nees_all"] = tail_nees(err, r["var"], np.ones(len(err), bool), tail)
        sc["nees_conv"] = tail_nees(err, r["var"], conv, tail)
        right = np.abs(err[:, -1]) <= thr
        sc["nees_right"] = tail_nees(err, r["var"], right, tail)
        sc["n_right"] = int(right.sum())
        sc["conv_rmse"] = float(np.sqrt(np.mean(err[conv][:, -tail:] ** 2))) if conv.any() else np.nan
        sc["conv_wrong"] = float(np.mean(np.abs(err[conv][:, -1]) > thr)) if conv.any() else np.nan
        out[name] = sc
    pf = res["PF"]
    out["PF"].update(
        hill_frac=float(np.mean(hill >= 0)), n_conv=int(conv.sum()),
        hill_time_median=float(np.median(hill[hill >= 0]) * cfg.dt) if (hill >= 0).any() else np.nan,
        hill_time_mean=float(np.mean(hill[hill >= 0]) * cfg.dt) if (hill >= 0).any() else np.nan,
        true_mass=float(np.mean(pf["true_mass"][:, -1])),
        ess_mean=float(np.mean(pf["ess"]) / cfg.n_particles), ess_min=float(np.min(pf["ess"]) / cfg.n_particles),
        resamples=float(np.mean(pf["resampled"].sum(axis=1))))
    return out


# --- printing ---------------------------------------------------------------------------

def print_setup(cfg, trials, seed):
    lo, hi = cfg.zone
    print(f"Setup: corridor {cfg.length:g} m, doors at {', '.join(f'{d:g}' for d in cfg.doors)} m "
          f"(smallest spacing {cfg.min_gap:g} m); start zone [{lo:g}, {hi:g}] m; "
          f"{cfg.steps} steps x {cfg.dt:g} s at {cfg.speed:g} m/s = {cfg.travel:g} m travel")
    print(f"  profile sensor: bump width s={cfg.width:g} m, sigma_z={cfg.sigma_z:g}; "
          f"motion noise sigma_u=sqrt(({cfg.motion_factor:g}*|u|)^2+{cfg.motion_floor:g}^2) "
          f"= {motion_noise_std(cfg.speed * cfg.dt, cfg.motion_factor, cfg.motion_floor):.4f} m/step")
    if cfg.prior_std is None:
        m0, p0 = prior_gaussian(cfg)
        prior = f"EKF/UKF prior N({m0:g}, {p0:.3f}) (std {np.sqrt(p0):.2f} m)"
    else:
        prior = (f"KNOWN-START DIAGNOSTIC: every filter starts from N(true start + noise, {cfg.prior_std:g}^2) "
                 f"(PF particles drawn from it too)")
    print(f"  PF: N={cfg.n_particles}, resample when N_eff < {cfg.resample_threshold:g} N, low-variance resampler; "
          f"{prior}; "
          f"UKF alpha={cfg.ukf_alpha:g} beta={cfg.ukf_beta:g} kappa={cfg.ukf_kappa:g}")
    print(f"  scoring: wrong door if |final error| > {cfg.wrong_thr:g} m; one hill = >= {cfg.hill_mass:g} of the weight "
          f"within {cfg.hill_h:g} m of the weighted mean; NEES over the last {cfg.tail_steps} steps; "
          f"{trials} Monte Carlo trials, seed {seed}")


def print_results(cfg, sc, n_trials):
    print("\nFinal-step error and wrong-door rate (all trials)")
    print(f"  {'filter':8s} {'RMSE (m)':>9s} {'median |err| (m)':>17s} {'wrong-door':>11s} {'tail RMSE (m)':>14s}")
    for name, s in sc.items():
        print(f"  {name:8s} {s['final_rmse']:9.3f} {s['final_median']:17.3f} {s['wrong']:11.3f} {s['tail_rmse']:14.3f}")
    lo, hi = averaged_nees_bounds(1, n_trials)
    print(f"\nNEES (1D; 1 = consistent), mean over the last {cfg.tail_steps} steps of the trial-averaged NEES")
    print(f"  all trials: 95% bound for one averaged step [{lo:.2f}, {hi:.2f}] (n={n_trials})")
    n_conv = sc["PF"]["n_conv"]
    print(f"  {'filter':8s} {'NEES all':>10s} {'steps in bound':>15s} | {'NEES conv':>10s} {'steps in bound':>15s}"
          f" {'conv RMSE (m)':>14s} {'conv wrong':>11s}")
    for name, s in sc.items():
        na, ia = s["nees_all"][:2]
        nc, ic = s["nees_conv"][:2]
        print(f"  {name:8s} {na:10.2f} {ia:12d}/{cfg.tail_steps:<2d} | {nc:10.2f} {ic:12d}/{cfg.tail_steps:<2d}"
              f" {s['conv_rmse']:14.3f} {s['conv_wrong']:11.3f}")
    print(f"  NEES over the trials where that filter ends at the right door (subset differs per filter):")
    for name, s in sc.items():
        nr, ir, rlo, rhi = s["nees_right"]
        print(f"    {name:8s} n={s['n_right']:3d}  NEES {nr:8.2f}  steps in bound {ir:2d}/{cfg.tail_steps}  "
              f"(95% bound [{rlo:.2f}, {rhi:.2f}])")
    if n_conv:
        blo, bhi = sc["PF"]["nees_conv"][2:]
        print(f"  converged subset = trials where the PF is one hill from the start of the tail window on: "
              f"n={n_conv}, 95% bound [{blo:.2f}, {bhi:.2f}]")
    p = sc["PF"]
    print(f"\nPF one-hill and sample statistics")
    print(f"  one hill by the end: {p['hill_frac']:.3f} of trials; time to one hill (those trials): "
          f"median {p['hill_time_median']:.1f} s, mean {p['hill_time_mean']:.1f} s; "
          f"weight within {cfg.wrong_thr:g} m of the truth at the end: {p['true_mass']:.3f}")
    print(f"  N_eff/N before resampling: mean {p['ess_mean']:.3f}, min {p['ess_min']:.3f}; "
          f"resamplings per trial: {p['resamples']:.1f} of {cfg.steps + 1} steps")


SWEEP_HEADER = (f"  {'setting':32s} {'wrong PF':>9s} {'EKF':>6s} {'UKF':>6s} {'PF 1-hill':>10s} "
                f"{'t_hill med (s)':>15s} {'PF RMSE':>8s} {'PF med|e|':>10s} {'PF true mass':>13s} {'ESS/N':>6s}")


def print_sweep_row(label, sc):
    p = sc["PF"]
    print(f"  {label:32s} {p['wrong']:9.3f} {sc['EKF']['wrong']:6.3f} {sc['UKF']['wrong']:6.3f} {p['hill_frac']:10.3f} "
          f"{p['hill_time_median']:15.1f} {p['final_rmse']:8.3f} {p['final_median']:10.3f} {p['true_mass']:13.3f} {p['ess_mean']:6.3f}")


def run_sweep(name, cfg, noises):
    """One sweep: same noise draws at every setting. Prints a table.
    Arguments:
        name: one of SWEEPS; cfg: base Config; noises: draw_noise output
    """
    if name == "particles":
        settings = [(f"N = {n}", dict(n_particles=n)) for n in (50, 100, 500, 1000, 2000, 5000)]
    elif name == "width":
        settings = [(f"bump width s = {v:g} m", dict(width=v)) for v in (0.15, 0.3, 0.6)]
    elif name == "sigma_z":
        settings = [(f"sigma_z = {v:g}", dict(sigma_z=v)) for v in (0.05, 0.1, 0.2, 0.4)]
    elif name == "motion":
        settings = [(f"motion factor = {v:g}", dict(motion_factor=v)) for v in (0.02, 0.05, 0.10)]
    elif name == "resample":
        settings = [("resample if N_eff < N/3", dict(resample_threshold=1 / 3)),
                    ("resample if N_eff < N/2", dict(resample_threshold=0.5)),
                    ("resample every step", dict(resample_threshold=1.01))]
    elif name == "layout":
        settings = [(k, dict(doors=v)) for k, v in LAYOUTS.items()]
    else:
        raise ValueError(f"unknown sweep {name!r}; choose from {SWEEPS}")
    print(f"\nSweep: {name} ({len(noises)} trials per setting, same noise draws; other parameters at the defaults)")
    print(SWEEP_HEADER)
    checks = []
    for label, change in settings:
        c = dataclasses.replace(cfg, **change)
        x_true, res = run_monte_carlo(c, noises)
        print_sweep_row(label, score(c, x_true, res))
        checks.append(x_true[:, -1].sum())
        if name == "layout":
            print(f"    wrong-door threshold {c.wrong_thr:g} m, hill half-width {c.hill_h:g} m for this layout")
    if name != "motion":     # every other knob leaves the truth untouched
        if len(set(np.round(checks, 9))) != 1:
            raise RuntimeError("truth changed across a sweep that must not change it")
        print(f"  check: the truth is identical across these settings (sum of final truth {checks[0]:.6f}); "
              f"the EKF/UKF columns change only where the knob enters their model")


# --- figure -----------------------------------------------------------------------------

def make_figure(cfg, noises, x_true, res, example, out_path):
    """Corridor snapshots of the beliefs at four times for one example trial,
    plus the Monte Carlo metrics over time.
    Arguments:
        cfg: Config; noises: draw_noise output; x_true: (n, T+1) truths;
        res: run_monte_carlo output; example: trial index shown in the snapshots; out_path: PNG path
    """
    data = generate_trial(cfg, noises[example])
    pf = pf_on_trial(cfg, noises[example], data, record=True)
    assert np.allclose(pf['mean'], res['PF']['mean'][example])
    t1 = cfg.steps + 1
    times = np.arange(t1) * cfg.dt
    bin_w = 0.25
    x_lo, x_hi = -2.0, cfg.length + 10.0
    xs = np.linspace(x_lo, x_hi, 800)
    snaps = [0, cfg.steps // 6, cfg.steps // 3, cfg.steps]
    fig = plt.figure(figsize=(17, 8.5))
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.28)
    for i, k in enumerate(snaps):
        ax = fig.add_subplot(gs[0, i])
        ax.fill_between(xs, 0, door_profile(xs, cfg.doors, cfg.width), color="0.85", label="profile f(x)")
        ax.hist(pf["x_hist"][k], bins=np.arange(x_lo, x_hi + bin_w, bin_w), weights=pf["w_hist"][k],
                color=COLORS["PF"], alpha=0.7, label="PF belief")
        for name in ("EKF", "UKF"):
            m, v = res[name]["mean"][example, k], res[name]["var"][example, k]
            ax.plot(xs, bin_w * np.exp(-0.5 * (xs - m) ** 2 / v) / np.sqrt(2 * np.pi * v), color=COLORS[name], label=name)
        ax.axvline(x_true[example, k], color="k", linestyle="--", label="truth")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel("position (m)")
        ax.set_title(f"t = {times[k]:.1f} s (step {k}), N_eff = {pf['ess'][k]:.0f}")
        if i == 0:
            ax.set_ylabel(f"probability per {bin_w:g} m bin")
            ax.legend(fontsize=7, loc="upper right")
    ax = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(0)
    for k in range(t1):
        idx = rng.choice(cfg.n_particles, 100, p=pf["w_hist"][k])
        ax.plot(np.full(100, times[k]), pf["x_hist"][k][idx], ".", color=COLORS["PF"], alpha=0.15, markersize=2)
    ax.plot(times, x_true[example], "k--", label="truth")
    for name in ("EKF", "UKF"):
        ax.plot(times, res[name]["mean"][example], color=COLORS[name], label=f"{name} mean")
    for d in cfg.doors:
        ax.axhline(d, color="0.7", linewidth=0.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position (m)")
    ax.set_title("example trial: PF particles over time")
    ax.legend(fontsize=7, loc="upper left")
    n = len(x_true)
    ax = fig.add_subplot(gs[1, 1])
    for name in ("PF", "EKF", "UKF"):
        ax.plot(times, np.mean(np.abs(res[name]["mean"] - x_true) > cfg.wrong_thr, axis=0), color=COLORS[name], label=name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"fraction of trials with |error| > {cfg.wrong_thr:g} m")
    ax.set_title(f"wrong-door fraction ({n} trials)")
    ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[1, 2])
    for name in ("PF", "EKF", "UKF"):
        ax.semilogy(times, np.mean((res[name]["mean"] - x_true) ** 2 / np.maximum(res[name]["var"], 1e-12), axis=0),
                    color=COLORS[name], label=name)
    lo, hi = averaged_nees_bounds(1, n)
    ax.axhspan(lo, hi, color="0.85", label="95% bound")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("trial-averaged NEES (1 = consistent)")
    ax.set_title("1D NEES, all trials")
    ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[1, 3])
    ess = res["PF"]["ess"] / cfg.n_particles
    ax.plot(times, ess.mean(axis=0), color=COLORS["PF"], label="mean")
    ax.fill_between(times, np.percentile(ess, 10, axis=0), np.percentile(ess, 90, axis=0), color=COLORS["PF"], alpha=0.2,
                    label="10-90 percentile")
    ax.axhline(cfg.resample_threshold, color="k", linestyle=":", label="resample threshold")
    ax.plot(times, res["PF"]["hill_mass"].mean(axis=0), color="tab:green", label="weight in one-hill window")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("N_eff / N  or  one-hill weight")
    ax.set_title("PF effective sample size")
    ax.legend(fontsize=7)
    fig.suptitle(f"PF vs EKF vs UKF, 1D door corridor (N={cfg.n_particles}, sigma_z={cfg.sigma_z:g}, s={cfg.width:g} m)")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --- CLI --------------------------------------------------------------------------------

def build_parser():
    d = Config()
    p = argparse.ArgumentParser(description="1D door-corridor global localization: PF vs EKF vs UKF")
    p.add_argument("--seed", type=int, default=0, help="integer seed of all random streams")
    p.add_argument("--trials", type=int, default=400, help="number of Monte Carlo trials")
    p.add_argument("--corridor-length", type=float, default=d.length, help="corridor length L (m)")
    p.add_argument("--doors", type=float, nargs="+", default=list(d.doors), help="door positions m_k (m)")
    p.add_argument("--door-width", type=float, default=d.width, help="profile bump width s, std-dev of each door's Gaussian bump (m)")
    p.add_argument("--sensor-noise", type=float, default=d.sigma_z, help="profile sensor noise std-dev sigma_z (profile height is 1)")
    p.add_argument("--speed", type=float, default=d.speed, help="commanded speed (m/s)")
    p.add_argument("--dt", type=float, default=d.dt, help="step interval (s)")
    p.add_argument("--steps", type=int, default=d.steps, help="number of motion steps T")
    p.add_argument("--motion-noise-factor", type=float, default=d.motion_factor, help="motion noise per metre travelled (m/m)")
    p.add_argument("--motion-noise-floor", type=float, default=d.motion_floor, help="motion noise independent of travel (m per step)")
    p.add_argument("--particles", type=int, default=d.n_particles, help="number of particles N")
    p.add_argument("--resample-threshold", type=float, default=d.resample_threshold,
                   help="resample when N_eff < threshold * N (fraction of N; above 1 resamples every step)")
    p.add_argument("--ukf-alpha", type=float, default=d.ukf_alpha, help="UKF unscented-transform spread parameter")
    p.add_argument("--ukf-beta", type=float, default=d.ukf_beta, help="UKF prior-knowledge parameter (2 is optimal for Gaussian priors)")
    p.add_argument("--ukf-kappa", type=float, default=d.ukf_kappa, help="UKF secondary scaling parameter (default gives n+kappa=3 for 1 state)")
    p.add_argument("--hill-mass", type=float, default=d.hill_mass, help="weight fraction that must lie in the one-hill window")
    p.add_argument("--hill-halfwidth", type=float, default=None, help="one-hill window half-width (m); default a quarter of the smallest door spacing")
    p.add_argument("--wrong-door-threshold", type=float, default=None, help="|final error| above this is a wrong door (m); default half the smallest door spacing")
    p.add_argument("--tail-steps", type=int, default=d.tail_steps, help="number of final steps for RMSE and NEES windows")
    p.add_argument("--prior-std", type=float, default=None,
                   help="known-start diagnostic: all filters start from N(true start + noise, std^2) with this std (m) instead of the uniform start-zone prior")
    p.add_argument("--ml-variant", action="store_true", help="also run EKF-ML and UKF-ML (offset-to-nearest-door sensor, ML association)")
    p.add_argument("--ml-range", type=float, default=d.ml_range, help="offset sensor range: nearest door reported only within this distance (m)")
    p.add_argument("--ml-noise", type=float, default=d.sigma_c, help="offset sensor noise std-dev sigma_c (m)")
    p.add_argument("--sweep", action="append", choices=SWEEPS, default=[], help="run this sweep (repeatable); none run by default")
    p.add_argument("--example-trial", type=int, default=0, help="trial index shown in the figure snapshots")
    p.add_argument("--out", type=str, default=None, help="save the figure to this PNG path instead of showing it")
    return p


def config_from_args(a):
    return Config(length=a.corridor_length, doors=tuple(a.doors), width=a.door_width, sigma_z=a.sensor_noise,
                  speed=a.speed, dt=a.dt, steps=a.steps, motion_factor=a.motion_noise_factor,
                  motion_floor=a.motion_noise_floor, n_particles=a.particles, resample_threshold=a.resample_threshold,
                  ukf_alpha=a.ukf_alpha, ukf_beta=a.ukf_beta, ukf_kappa=a.ukf_kappa, hill_mass=a.hill_mass,
                  hill_halfwidth=a.hill_halfwidth, wrong_door_threshold=a.wrong_door_threshold,
                  tail_steps=a.tail_steps, ml_range=a.ml_range, sigma_c=a.ml_noise, prior_std=a.prior_std)


def main():
    args = build_parser().parse_args()
    cfg = config_from_args(args)
    if cfg.zone[1] <= cfg.zone[0]:
        raise SystemExit("travel distance must be shorter than the corridor")
    if not 0 <= args.example_trial < args.trials:
        raise SystemExit("--example-trial must be in [0, trials)")
    print_setup(cfg, args.trials, args.seed)
    noises = draw_noise(args.seed, args.trials, cfg.steps)
    x_true, res = run_monte_carlo(cfg, noises, ml_variant=args.ml_variant)
    sc = score(cfg, x_true, res)
    print_results(cfg, sc, args.trials)
    if args.ml_variant:
        d = np.max(np.abs(res["EKF-ML"]["mean"] - res["UKF-ML"]["mean"]))
        same = np.array_equal(res["EKF-ML"]["picked"], res["UKF-ML"]["picked"])
        print(f"\nEKF-ML vs UKF-ML (linear measurement given the door): max |mean difference| = {d:.2e} m, "
              f"same door chosen at every step: {same}")
        det = np.mean(~np.isnan(np.stack([generate_trial(cfg, nz)[2] for nz in noises])))
        print(f"  offset sensor reports a detection on {det:.3f} of the steps (range {cfg.ml_range:g} m)")

    data = generate_trial(cfg, noises[0])
    h_fn = lambda x: door_profile(x, cfg.doors, cfg.width)
    dh_fn = lambda x: door_profile_grad(x, cfg.doors, cfg.width)
    print("\nTiming (trial 0, per step) and peak memory")
    for name, fn in (
            ("PF", lambda: pf_on_trial(cfg, noises[0], data)),
            ("EKF", lambda: run_ekf(data[1], data[3], data[4], h_fn, dh_fn, cfg.sigma_z, *prior_gaussian(cfg, noises[0], data[0][0]))),
            ("UKF", lambda: run_ukf(data[1], data[3], data[4], h_fn, cfg.sigma_z, *prior_gaussian(cfg, noises[0], data[0][0]),
                                    cfg.ukf_alpha, cfg.ukf_beta, cfg.ukf_kappa))):
        _, t_step, mem = measure_performance(fn, n_steps=cfg.steps + 1)
        print(f"  {name:4s} {t_step * 1e6:9.1f} us/step   peak {mem / 1024:9.1f} KiB")

    for name in args.sweep:
        run_sweep(name, cfg, noises)

    if args.out:
        make_figure(cfg, noises, x_true, res, args.example_trial, args.out)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
