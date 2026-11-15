'''
Where an IMU's noise values come from: reading white-noise density and bias
random walk off a static recording with the Allan deviation, and what a wrong
density -> per-sample conversion does inside the ESKF of
`eskf_imu_position_fix.py`.

Part 1, identification. One gyroscope axis and one accelerometer axis of a
motionless IMU are simulated with the noise model of Kalibr's "IMU Noise
Model" page (ref. 1 in docs/foundations/noise_identification.md):

    y[k] = b[k] + n[k]
    n[k] = (N / sqrt(dt)) w[k]            white noise, noise density N
    b[k] = b[k-1] + K sqrt(dt) w'[k]      bias random walk, density K

with w, w' ~ N(0, 1), dt the sample interval, N in (rad/s)/sqrt(Hz) for the
gyroscope and (m/s^2)/sqrt(Hz) for the accelerometer, and K in
(rad/s^2)/sqrt(Hz) and (m/s^3)/sqrt(Hz). The default values N, K are the
`sensor.yaml` values of the EuRoC MAV dataset's ADIS16448 IMU (200 Hz), see
PRESETS: they come from a copy of the dataset's imu0/sensor.yaml in the EuRoC-rs
crate's test data, not from the original dataset archive. Bias instability (flicker noise) is not simulated: no source with a
simulation model and its Allan-variance formula was found.

The overlapping Allan variance is computed with the integrated-data form of
NIST SP 1065 eq. (11): with x[j] = dt * sum_{i<=j} y[i] (x[0] = 0), cluster
size m, cluster time tau = m dt and n samples,

    AVAR(tau) = sum_{j=0}^{n-2m} (x[j+2m] - 2 x[j+m] + x[j])^2
                / (2 (n - 2m + 1) tau^2)

(the same number as eq. (10) of that source on the cluster averages of y).
One cumulative sum serves every tau, so each tau costs O(n). The expected
value of the continuous-time model is N^2 / tau for the white term (IEEE Std
952-1997 eq. C.3, El-Sheimy et al. 2008 eq. 17) and K^2 tau / 3 for the
random-walk term (IEEE 952 eq. C.7, El-Sheimy eq. 21), slopes -1/2 and +1/2,
read at tau = 1 s and tau = 3 s. The discrete-time value of the random-walk
term is K^2 dt (2 m^2 + 1) / (6 m), derived here from the covariance of the
estimator's term and checked in the tests against the sum above; it tends to
the quoted K^2 tau / 3 for large m, and the white term is exactly N^2 / tau:

    ADEV(tau) = sqrt(N^2 / tau + K^2 tau / 3)         (continuous-time model)

Reading the parameters (Kalibr page and Woodman 2007): N is the value at
tau = 1 s of a slope -1/2 line fitted to the white-noise part of the curve,
K is the value at tau = 3 s of a slope +1/2 line fitted to the random-walk
part. Both fit regions are chosen automatically from the local slope of the
curve (fit_noise_parameters) and printed.

Experiment 1: record length (--hours) x seeds (--n-seeds): one nested record
per seed and sensor (the 0.25 h record is the first 0.25 h of the 1 h record),
the estimates N_hat, K_hat against the truth, as median and spread of the
percent error.

Part 2, the conversion mistake. The truth of `eskf_imu_position_fix.py`
(generate_trial) is run with per-sample noise stds N / sqrt(dt) from the same
preset. Its parameters mean: `*_noise_std` is a PER-SAMPLE std (it enters Q
squared and is mapped by dt-scaled blocks of L), `*_bias_rw` is a CONTINUOUS
density (it enters L through sqrt(dt)), and the true biases are constants
(`*_bias_std` is the std of their draw and the filter's prior). Because that
truth has no bias walk, add_bias_walk adds the preset's walk (per-step std
K sqrt(dt)) on top of the constant biases from a spawned stream, and
trial_errors_walk scores the bias error against the drifting truth. Three ESKFs
see the same trials and differ only in the per-sample std put in Q:

  - correct: N / sqrt(dt)
  - density_as_std: N (the density used directly as the per-sample std)
  - double_conversion: N / dt (a per-sample std converted a second time)

All three use the same K as the bias-walk density. Scenarios (--scenarios,
noise_scale:bias_walk_scale): 1:1 is the preset; 1:0 removes the bias walk from
truth and from every Q (the script's own constant-bias truth, so the filters'
Q contains nothing that is not in the truth); 10:1 and 100:1 multiply N and K
by 10 and 100 (Kalibr's page says the static-data values of low-cost IMUs may
need to be increased by 10x or more; it is a what-if, not a measured value).
Scores: Monte Carlo NEES of the 15-state error, averaged over trials, with its
95% band from `averaged_nees_bounds`, plus attitude and position RMSE. Every
trial redraws motion, biases, bias walk, IMU noise, position-fix noise and the
initial error.

Findings are in docs/foundations/noise_identification.md (written after the
numbers were in).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt

from state_estimation_simulations.use_numpy.eskf_imu_position_fix import (
    generate_trial, imu_noise_covariance, initial_covariance, run_eskf)
from state_estimation_simulations.use_numpy.lie_utils import so3_log
from state_estimation_simulations.utils import averaged_nees_bounds, measure_performance, nees

# EuRoC MAV dataset, imu0/sensor.yaml ("VI-Sensor IMU (ADIS16448)", rate_hz: 200). These values come
# from a copy of the dataset's imu0/sensor.yaml in the EuRoC-rs crate's test data, NOT from the
# original dataset archive: https://github.com/eduidl/EuRoC-rs/blob/main/test_data/imu0/sensor.yaml
#   gyroscope_noise_density: 1.6968e-04     # [ rad / s / sqrt(Hz) ]
#   gyroscope_random_walk: 1.9393e-05       # [ rad / s^2 / sqrt(Hz) ]
#   accelerometer_noise_density: 2.0000e-3  # [ m / s^2 / sqrt(Hz) ]
#   accelerometer_random_walk: 3.0000e-3    # [ m / s^3 / sqrt(Hz) ]
PRESETS = {
    "euroc": dict(rate_hz=200.0,
                  gyro=dict(density=1.6968e-4, walk=1.9393e-5),
                  accel=dict(density=2.0e-3, walk=3.0e-3)),
}
SENSORS = ("gyro", "accel")
UNITS = {"gyro": dict(density="rad/s/sqrt(Hz)", walk="rad/s^2/sqrt(Hz)", adev="rad/s"),
         "accel": dict(density="m/s^2/sqrt(Hz)", walk="m/s^3/sqrt(Hz)", adev="m/s^2")}
CONVERSIONS = ("correct", "density_as_std", "double_conversion")

# --------------------------------------------------------------------------- Part 1

def simulate_static_axis(n, dt, density, walk, rng_white, rng_walk):
    """One axis of a motionless IMU: white noise plus a bias random walk (zero start).
    Arguments:
        n: number of samples
        dt: sample interval (s)
        density: white-noise density N (unit/sqrt(Hz); per-sample std is N / sqrt(dt))
        walk: bias random-walk density K (unit/s/sqrt(Hz); per-step std is K sqrt(dt))
        rng_white, rng_walk: independent numpy Generators for the two processes (the first
            n draws do not depend on n, so a shorter record is a prefix of a longer one)
    Returns:
        y: (n,) readings (unit of the sensor: rad/s or m/s^2)
    """
    white = (density / np.sqrt(dt)) * rng_white.standard_normal(n)
    bias = np.cumsum((walk * np.sqrt(dt)) * rng_walk.standard_normal(n))
    return white + bias


def cluster_sizes(n, per_decade=12, min_clusters=9):
    """Log-spaced cluster sizes m (samples per cluster) from 1 up to n / min_clusters.
    Arguments:
        n: number of samples
        per_decade: number of cluster sizes per decade
        min_clusters: fewest clusters a record must hold at the largest tau
            (Woodman 2007: at least 9 bins)
    Returns:
        (k,) sorted unique integer cluster sizes
    """
    m_max = max(1, n // min_clusters)
    count = max(2, int(np.ceil(np.log10(m_max) * per_decade)) + 1)
    return np.unique(np.round(np.logspace(0.0, np.log10(m_max), count)).astype(int))


def allan_deviation(y, dt, m_list):
    """Overlapping Allan deviation by the cumulative-sum form of NIST SP 1065 eq. (11).
    Arguments:
        y: (n,) readings of one axis (rad/s or m/s^2)
        dt: sample interval (s)
        m_list: (k,) integer cluster sizes, each with 2 m <= n
    Returns:
        tau: (k,) cluster times m dt (s)
        adev: (k,) Allan deviation (unit of y)
    """
    x = np.concatenate(([0.0], np.cumsum(y))) * dt
    n_x = len(x)
    adev = np.empty(len(m_list))
    for i, m in enumerate(m_list):
        d = x[2 * m:] - 2.0 * x[m:n_x - m] + x[:n_x - 2 * m]
        adev[i] = np.sqrt(np.dot(d, d) / (2.0 * (n_x - 2 * m) * (m * dt) ** 2))
    return np.asarray(m_list) * dt, adev


def allan_deviation_reference(y, dt, m_list):
    """Slow reference: eq. (10) of NIST SP 1065 on explicit overlapping cluster averages.
    Arguments:
        y: (n,) readings
        dt: sample interval (s)
        m_list: (k,) integer cluster sizes
    Returns:
        tau, adev as allan_deviation
    """
    adev = []
    for m in m_list:
        avg = np.array([np.mean(y[j:j + m]) for j in range(len(y) - m + 1)])
        diffs = avg[m:] - avg[:-m]
        adev.append(np.sqrt(0.5 * np.mean(diffs ** 2)))
    return np.asarray(m_list) * dt, np.array(adev)


def model_allan_deviation(tau, density, walk, dt=None):
    """Expected Allan deviation of white noise (density N) plus a bias random walk (density K).
    Arguments:
        tau: (k,) cluster times (s)
        density: white-noise density N
        walk: bias random-walk density K
        dt: None for the continuous-time model sqrt(N^2/tau + K^2 tau/3); a sample interval (s)
            for the exact discrete-time value sqrt(N^2/tau + K^2 dt (2m^2+1)/(6m)), m = tau/dt
    Returns:
        (k,) expected Allan deviation (unit of the sensor)
    """
    tau = np.asarray(tau, dtype=float)
    if dt is None:
        walk_var = walk ** 2 * tau / 3.0
    else:
        m = tau / dt
        walk_var = walk ** 2 * dt * (2.0 * m ** 2 + 1.0) / (6.0 * m)
    return np.sqrt(density ** 2 / tau + walk_var)


def local_slope(tau, adev, half_window=2):
    """Slope of log(adev) versus log(tau), by least squares over +-half_window neighbors.
    Arguments:
        tau, adev: (k,) Allan curve
        half_window: neighbors on each side (fewer at the ends)
    Returns:
        (k,) local log-log slope
    """
    lt, la = np.log(tau), np.log(adev)
    slope = np.empty(len(tau))
    for i in range(len(tau)):
        lo, hi = max(0, i - half_window), min(len(tau), i + half_window + 1)
        slope[i] = np.polyfit(lt[lo:hi], la[lo:hi], 1)[0]
    return slope


def fit_noise_parameters(tau, adev, band=0.1, min_points=4):
    """Reads N and K off an Allan curve with fixed-slope lines on automatically chosen regions.
    White region: the run of points from the shortest tau on whose local slope is within
    [-0.5 - band, -0.5 + band]. Random-walk region: the longest run of points after the curve's
    minimum whose local slope is within [0.5 - band, 0.5 + band] (at least min_points points).
    A slope -1/2 line through the white region read at tau = 1 s gives N; a slope +1/2 line
    through the random-walk region read at tau = 3 s gives K (AVAR = K^2 tau / 3 -> ADEV(3 s) = K).
    Arguments:
        tau, adev: (k,) Allan curve (s, unit of the sensor)
        band: half-width of the accepted slope window around -1/2 and +1/2
        min_points: fewest points a region needs; otherwise that parameter is NaN
    Returns:
        dict: density (N or NaN), walk (K or NaN), white_range, walk_range (tau of the first and
        last point of each region, or None)
    """
    slope = local_slope(tau, adev)
    white_ok = np.abs(slope + 0.5) <= band
    n_white = 0
    while n_white < len(tau) and white_ok[n_white]:
        n_white += 1
    out = dict(density=np.nan, walk=np.nan, white_range=None, walk_range=None)
    if n_white >= min_points:
        out["density"] = float(np.exp(np.mean(np.log(adev[:n_white] * np.sqrt(tau[:n_white])))))
        out["white_range"] = (float(tau[0]), float(tau[n_white - 1]))
    walk_ok = np.abs(slope - 0.5) <= band
    walk_ok[:int(np.argmin(adev)) + 1] = False
    best, run_start = (0, 0), None
    for i in range(len(tau) + 1):
        if i < len(tau) and walk_ok[i]:
            run_start = i if run_start is None else run_start
        else:
            if run_start is not None and i - run_start > best[1] - best[0]:
                best = (run_start, i)
            run_start = None
    if best[1] - best[0] >= min_points:
        sl = slice(*best)
        out["walk"] = float(np.sqrt(3.0) * np.exp(np.mean(np.log(adev[sl] / np.sqrt(tau[sl])))))
        out["walk_range"] = (float(tau[best[0]]), float(tau[best[1] - 1]))
    return out


def record_length_experiment(preset, hours, n_seeds, per_decade, band, seed, keep_curve=True):
    """Estimates N and K from nested records of increasing length, for every seed and sensor.
    Arguments:
        preset: dict from PRESETS
        hours: record lengths (h), the longest record is generated once per seed and sensor
        n_seeds: number of independent records per sensor
        per_decade: cluster sizes per decade
        band: slope window of fit_noise_parameters
        seed: root seed (each seed and sensor gets its own spawned streams)
        keep_curve: store the Allan curve and fit of seed 0 for every length (for the figure)
    Returns:
        dict sensor -> dict length (h) -> dict with density, walk (arrays over seeds),
        and, for seed 0, tau, adev, white_range, walk_range
    """
    dt = 1.0 / preset["rate_hz"]
    n_max = int(round(max(hours) * 3600.0 / dt))
    roots = np.random.SeedSequence(seed).spawn(n_seeds)
    results = {s: {h: dict(density=[], walk=[], regions=[]) for h in hours} for s in SENSORS}
    for i, root in enumerate(roots):
        streams = root.spawn(2 * len(SENSORS))
        for j, s in enumerate(SENSORS):
            p = preset[s]
            y = simulate_static_axis(n_max, dt, p["density"], p["walk"],
                                     np.random.default_rng(streams[2 * j]),
                                     np.random.default_rng(streams[2 * j + 1]))
            for h in hours:
                n = int(round(h * 3600.0 / dt))
                tau, adev = allan_deviation(y[:n], dt, cluster_sizes(n, per_decade))
                fit = fit_noise_parameters(tau, adev, band)
                r = results[s][h]
                r["density"].append(fit["density"])
                r["walk"].append(fit["walk"])
                r["regions"].append((fit["white_range"], fit["walk_range"]))
                if keep_curve and i == 0:
                    r.update(tau=tau, adev=adev, white_range=fit["white_range"], walk_range=fit["walk_range"],
                             fit_density=fit["density"], fit_walk=fit["walk"])
    for s in SENSORS:
        for h in hours:
            for key in ("density", "walk"):
                results[s][h][key] = np.array(results[s][h][key])
    return results


def percent_error_summary(estimates, truth):
    """Median and spread of the percent error of an estimate, ignoring unresolved (NaN) runs.
    Arguments:
        estimates: (n_seeds,) estimates, NaN where the parameter could not be read
        truth: true value
    Returns:
        median_pct, std_pct (percent), n_resolved
    """
    ok = ~np.isnan(estimates)
    if not ok.any():
        return np.nan, np.nan, 0
    err = 100.0 * (estimates[ok] / truth - 1.0)
    return float(np.median(err)), float(np.std(err)), int(ok.sum())

# --------------------------------------------------------------------------- Part 2

def conversion_noise_stds(preset, dt, conversion):
    """Per-sample noise stds that a filter puts in Q for each way of converting the density.
    Arguments:
        preset: dict from PRESETS
        dt: IMU sample interval of the filter (s)
        conversion: 'correct' (N / sqrt(dt)), 'density_as_std' (N) or 'double_conversion' (N / dt)
    Returns:
        gyro_noise_std (rad/s), accel_noise_std (m/s^2)
    """
    factor = dict(correct=1.0 / np.sqrt(dt), density_as_std=1.0, double_conversion=1.0 / dt)[conversion]
    return preset["gyro"]["density"] * factor, preset["accel"]["density"] * factor


def scaled_preset(preset, noise_scale, walk_scale=1.0):
    """Copy of a preset with N multiplied by noise_scale and K by noise_scale * walk_scale.
    Arguments:
        preset: dict from PRESETS
        noise_scale: factor on every noise parameter (1 = the preset as is)
        walk_scale: extra factor on K only (0 = no bias walk in truth or filters)
    Returns:
        new preset dict
    """
    out = dict(preset)
    for sensor in SENSORS:
        out[sensor] = dict(density=preset[sensor]["density"] * noise_scale,
                           walk=preset[sensor]["walk"] * noise_scale * walk_scale)
    return out


def add_bias_walk(trial, preset, rng):
    """Adds the preset's bias random walk to the IMU readings of a generate_trial trial.
    generate_trial draws constant biases only; the walk (per-step std K sqrt(dt), zero start) is
    added on top of them, from a stream spawned from rng so the main stream is not advanced.
    Arguments:
        trial: dict from generate_trial
        preset: dict from PRESETS
        rng: numpy Generator (used through rng.spawn only)
    Returns:
        trial copy with shifted gyro and accel, and walk_g, walk_a: (n,3) bias offsets per step
    """
    n, dt = len(trial["gyro"]), trial["dt"]
    child = rng.spawn(1)[0]
    walk_g = np.cumsum(preset["gyro"]["walk"] * np.sqrt(dt) * child.standard_normal((n, 3)), axis=0)
    walk_a = np.cumsum(preset["accel"]["walk"] * np.sqrt(dt) * child.standard_normal((n, 3)), axis=0)
    out = dict(trial)
    out["gyro"] = trial["gyro"] + walk_g
    out["accel"] = trial["accel"] + walk_a
    out["walk_g"], out["walk_a"] = walk_g, walk_a
    return out


def state_error_walk(trial, k, R, v, p, b_a, b_g):
    """15-vector error (dp, dv, dtheta, db_a, db_g) after step k when the true biases drift.
    Arguments:
        trial: dict from add_bias_walk
        k: step index (the estimate is of the state at step k + 1)
        R, v, p, b_a, b_g: the filter's estimate after step k
    Returns:
        (15,) error, truth minus estimate (rotation error Log(R_hat^T R))
    """
    return np.r_[trial["p"][k + 1] - p, trial["v"][k + 1] - v, so3_log(R.T @ trial["R"][k + 1]),
                 trial["b_a"] + trial["walk_a"][k] - b_a, trial["b_g"] + trial["walk_g"][k] - b_g]


def trial_errors_walk(trial, out):
    """Attitude error (deg), position error (m) and 15-state NEES per step when the true biases
    drift (same error definition as eskf_imu_position_fix.trial_errors, bias truth = constant
    draw + walk at that step).
    Arguments:
        trial: dict from add_bias_walk
        out: list from run_eskf
    Returns:
        att_err, pos_err, nees_values: (n,) arrays
    """
    n = len(out)
    att, pos, ne = np.zeros(n), np.zeros(n), np.zeros(n)
    for k, (R, v, p, b_a, b_g, P) in enumerate(out):
        err = state_error_walk(trial, k, R, v, p, b_a, b_g)
        att[k] = np.rad2deg(np.linalg.norm(err[6:9]))
        pos[k] = np.linalg.norm(err[0:3])
        ne[k] = nees(err, P)
    return att, pos, ne


def eskf_conversion_experiment(preset, dt, duration, n_trials, rng, fix_rate=1.0):
    """Monte Carlo of the 15-state ESKF fed with the three density -> per-sample conversions.
    Arguments:
        preset: dict from PRESETS (truth white noise = N / sqrt(dt), truth bias walk K; the same K
            is in every filter's Q)
        dt: IMU sample interval of the ESKF run (s)
        duration: run length (s)
        n_trials: Monte Carlo trials (each redraws motion, biases, bias walk, noise, fixes, initial error)
        rng: numpy random Generator
        fix_rate: position-fix rate (Hz)
    Returns:
        dict conversion -> dict with nees (n_steps,), att_rmse (deg), pos_rmse (m), all per step
    """
    g_true, a_true = conversion_noise_stds(preset, dt, "correct")
    kw = dict(duration=duration, dt=dt, fix_rate=fix_rate, rate_amp=0.4, accel_amp=0.5,
              gyro_noise_std=g_true, accel_noise_std=a_true, gyro_bias_std=0.01, accel_bias_std=0.05,
              pos_noise_std=0.1, init_pos_noise_std=0.1, init_vel_noise_std=0.1,
              init_att_noise_std=np.deg2rad(2.0))
    P0 = initial_covariance(kw["init_pos_noise_std"], kw["init_vel_noise_std"], kw["init_att_noise_std"],
                            kw["accel_bias_std"], kw["gyro_bias_std"])
    Q = {}
    for c in CONVERSIONS:
        g, a = conversion_noise_stds(preset, dt, c)
        Q[c] = imu_noise_covariance(a, g, preset["accel"]["walk"], preset["gyro"]["walk"])
    acc = {c: dict(nees=[], att=[], pos=[]) for c in CONVERSIONS}
    for _ in range(n_trials):
        trial = add_bias_walk(generate_trial(rng=rng, **kw), preset, rng)
        for c in CONVERSIONS:
            att, pos, ne = trial_errors_walk(trial, run_eskf(trial, P0, Q[c], kw["pos_noise_std"]))
            acc[c]["att"].append(att)
            acc[c]["pos"].append(pos)
            acc[c]["nees"].append(ne)
    return {c: dict(nees=np.mean(a["nees"], axis=0), att_rmse=np.sqrt(np.mean(np.square(a["att"]), axis=0)),
                    pos_rmse=np.sqrt(np.mean(np.square(a["pos"]), axis=0))) for c, a in acc.items()}


BLOCKS = dict(position=slice(0, 3), velocity=slice(3, 6), attitude=slice(6, 9),
              accel_bias=slice(9, 12), gyro_bias=slice(12, 15), joint=slice(0, 15))


def block_nees_experiment(preset, dt, duration, n_trials, rng, fix_rate=1.0):
    """Marginal NEES of each 3-state block and the joint 15-state NEES, second half of the run,
    for the three conversions on the same trials (same truth as eskf_conversion_experiment).
    Arguments:
        preset, dt, duration, n_trials, rng, fix_rate: as eskf_conversion_experiment
    Returns:
        dict conversion -> dict block name -> NEES averaged over trials and the second half
        (consistent: 3 for each 3-state block, 15 for the joint)
    """
    g_true, a_true = conversion_noise_stds(preset, dt, "correct")
    kw = dict(duration=duration, dt=dt, fix_rate=fix_rate, rate_amp=0.4, accel_amp=0.5,
              gyro_noise_std=g_true, accel_noise_std=a_true, gyro_bias_std=0.01, accel_bias_std=0.05,
              pos_noise_std=0.1, init_pos_noise_std=0.1, init_vel_noise_std=0.1,
              init_att_noise_std=np.deg2rad(2.0))
    P0 = initial_covariance(kw["init_pos_noise_std"], kw["init_vel_noise_std"], kw["init_att_noise_std"],
                            kw["accel_bias_std"], kw["gyro_bias_std"])
    Q = {c: imu_noise_covariance(*reversed(conversion_noise_stds(preset, dt, c)),
                                 preset["accel"]["walk"], preset["gyro"]["walk"]) for c in CONVERSIONS}
    acc = {c: {b: [] for b in BLOCKS} for c in CONVERSIONS}
    for _ in range(n_trials):
        trial = add_bias_walk(generate_trial(rng=rng, **kw), preset, rng)
        for c in CONVERSIONS:
            out = run_eskf(trial, P0, Q[c], kw["pos_noise_std"])
            half = len(out) // 2
            for b, sl in BLOCKS.items():
                acc[c][b].append(np.mean([nees(state_error_walk(trial, k, *out[k][:5])[sl], out[k][5][sl, sl])
                                          for k in range(half, len(out))]))
    return {c: {b: float(np.mean(v)) for b, v in blocks.items()} for c, blocks in acc.items()}


def second_half_mean(series):
    """Mean of the second half of a per-step series.
    Arguments:
        series: (n,) array
    Returns:
        scalar mean of series[n//2:]
    """
    return float(np.mean(series[len(series) // 2:]))

# --------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=("1", "2", "both"), default="both",
                        help="1 = Allan identification, 2 = ESKF conversion mistake, both = the two")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="euroc", help="IMU noise values")
    parser.add_argument("--hours", type=float, nargs="+", default=[0.25, 1.0, 4.0],
                        help="Static record lengths (h); each is a prefix of the longest")
    parser.add_argument("--n-seeds", type=int, default=10, help="Independent records per length and sensor")
    parser.add_argument("--per-decade", type=int, default=12, help="Cluster sizes per decade of tau")
    parser.add_argument("--slope-band", type=float, default=0.1,
                        help="Half-width of the local-slope window around -1/2 and +1/2 for the fit regions")
    parser.add_argument("--dt", type=float, default=0.01, help="ESKF IMU sample interval (s) for Part 2")
    parser.add_argument("--duration", type=float, default=30.0, help="ESKF run length (s) for Part 2")
    parser.add_argument("--fix-rate", type=float, default=1.0, help="Position-fix rate (Hz) for Part 2")
    parser.add_argument("--mc-trials", type=int, default=20, help="Monte Carlo trials per scenario for Part 2")
    parser.add_argument("--block-check", action="store_true",
                        help="Part 2: also print the NEES of each 3-state block next to the joint NEES "
                             "(runs every scenario once more)")
    parser.add_argument("--scenarios", nargs="+", default=["1:1", "1:0", "10:1", "100:1"],
                        help="Part 2 scenarios as noise_scale:bias_walk_scale; noise_scale multiplies N and K "
                             "of the preset, bias_walk_scale multiplies K only (0 = no bias walk in truth or Q)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    dt_imu = 1.0 / preset["rate_hz"]
    hours = sorted(args.hours)
    root_allan, root_eskf = np.random.SeedSequence(args.seed).spawn(2)
    do1, do2 = args.part in ("1", "both"), args.part in ("2", "both")

    print(f"Setup: preset {args.preset} ({preset['rate_hz']:g} Hz); gyro N={preset['gyro']['density']:.4e} "
          f"{UNITS['gyro']['density']}, K={preset['gyro']['walk']:.4e} {UNITS['gyro']['walk']}; accel "
          f"N={preset['accel']['density']:.4e} {UNITS['accel']['density']}, K={preset['accel']['walk']:.4e} "
          f"{UNITS['accel']['walk']}")

    res, nees_res = None, None
    if do1:
        n_max = int(round(hours[-1] * 3600.0 / dt_imu))
        y = simulate_static_axis(n_max, dt_imu, preset["gyro"]["density"], preset["gyro"]["walk"],
                                 np.random.default_rng(1), np.random.default_rng(2))
        _, t_step, mem = measure_performance(allan_deviation, y, dt_imu, cluster_sizes(n_max, args.per_decade),
                                             n_steps=n_max)
        print(f"\nPart 1: {args.n_seeds} seeds x records of {hours} h, {args.per_decade} cluster sizes per decade, "
              f"slope band +-{args.slope_band:g}")
        print(f"  Allan deviation of one {hours[-1]:g} h record ({n_max} samples): avg time={t_step * 1e9:.1f} "
              f"ns/sample | peak mem={mem / 1e6:.1f} MB")
        del y
        res = record_length_experiment(preset, hours, args.n_seeds, args.per_decade, args.slope_band,
                                       int(root_allan.generate_state(1)[0]))
        for s in SENSORS:
            for key, name in (("density", "N"), ("walk", "K")):
                true = preset[s][key if key == "density" else "walk"]
                print(f"\n  {s} {name} (true {true:.4e} {UNITS[s][key]}):")
                print(f"    {'record (h)':<11s}{'median est':>13s}{'median err %':>14s}{'std err %':>11s}{'resolved':>10s}")
                for h in hours:
                    est = res[s][h][key]
                    med, std, ok = percent_error_summary(est, true)
                    med_est = np.nanmedian(est) if ok else np.nan
                    print(f"    {h:<11g}{med_est:13.4e}{med:14.2f}{std:11.2f}{ok:7d}/{len(est)}")
        print("\n  Bias of the read-off itself: the same fit on the noise-free expected curve (exact discrete model), "
              "error % of N, K:")
        for s in SENSORS:
            for h in hours:
                n = int(round(h * 3600.0 / dt_imu))
                tau = cluster_sizes(n, args.per_decade) * dt_imu
                fit = fit_noise_parameters(tau, model_allan_deviation(tau, preset[s]["density"], preset[s]["walk"],
                                                                      dt_imu), args.slope_band)
                print(f"    {s:<6s}{h:<6g}h  N {100.0 * (fit['density'] / preset[s]['density'] - 1.0):+6.2f}   "
                      f"K {100.0 * (fit['walk'] / preset[s]['walk'] - 1.0):+6.2f}")
        print("\n  Fit regions of seed 0, tau from-to (s), white | random walk:")
        for s in SENSORS:
            for h in hours:
                r = res[s][h]
                w = "none" if r["white_range"] is None else f"{r['white_range'][0]:.3g}-{r['white_range'][1]:.3g}"
                k = "none" if r["walk_range"] is None else f"{r['walk_range'][0]:.3g}-{r['walk_range'][1]:.3g}"
                print(f"    {s:<6s}{h:<6g}h  white {w:<14s} walk {k}")

    if do2:
        n_steps = int(round(args.duration / args.dt))
        lo, hi = averaged_nees_bounds(15, args.mc_trials)
        print(f"\nPart 2: ESKF, {args.duration:g} s at {1.0 / args.dt:g} Hz, {args.fix_rate:g} Hz position fix, "
              f"{args.mc_trials} Monte Carlo trials; truth white noise = N / sqrt(dt), truth bias walk = K "
              f"(K also in every Q); 95% band of the averaged NEES (15 states): [{lo:.1f}, {hi:.1f}]")
        g0, a0 = conversion_noise_stds(preset, args.dt, "correct")
        trial = generate_trial(rng=np.random.default_rng(root_eskf), duration=args.duration, dt=args.dt,
                               fix_rate=args.fix_rate, rate_amp=0.4, accel_amp=0.5, gyro_noise_std=g0,
                               accel_noise_std=a0, gyro_bias_std=0.01, accel_bias_std=0.05, pos_noise_std=0.1,
                               init_pos_noise_std=0.1, init_vel_noise_std=0.1, init_att_noise_std=np.deg2rad(2.0))
        _, t_step, mem = measure_performance(
            run_eskf, trial, initial_covariance(0.1, 0.1, np.deg2rad(2.0), 0.05, 0.01),
            imu_noise_covariance(a0, g0, preset["accel"]["walk"], preset["gyro"]["walk"]), 0.1, n_steps=n_steps)
        print(f"  one ESKF run: avg time={t_step * 1e6:.1f} us/step | peak mem={mem / 1024.0:.1f} KB")
        nees_res = {}
        streams = root_eskf.spawn(len(args.scenarios))
        for spec, stream in zip(args.scenarios, streams):
            def fresh_rng(stream=stream):
                # a new SeedSequence with the same state: Generator.spawn advances the shared one's counter
                return np.random.default_rng(np.random.SeedSequence(stream.entropy, spawn_key=stream.spawn_key))

            noise_scale, walk_scale = (float(v) for v in spec.split(":"))
            sc_preset = scaled_preset(preset, noise_scale, walk_scale)
            nees_res[spec] = eskf_conversion_experiment(sc_preset, args.dt, args.duration, args.mc_trials,
                                                        fresh_rng(), args.fix_rate)
            print(f"\n  scenario {spec} (noise scale {noise_scale:g}, bias-walk scale {walk_scale:g}): truth "
                  f"per-sample std gyro {conversion_noise_stds(sc_preset, args.dt, 'correct')[0]:.3e} rad/s, "
                  f"accel {conversion_noise_stds(sc_preset, args.dt, 'correct')[1]:.3e} m/s^2; "
                  f"K gyro {sc_preset['gyro']['walk']:.3e}, accel {sc_preset['accel']['walk']:.3e}")
            print(f"    {'':<18s}{'Q std gyro':>12s}{'Q std accel':>13s}{'NEES':>9s}{'att RMSE (deg)':>16s}"
                  f"{'pos RMSE (m)':>14s}  verdict (second half)")
            for c in CONVERSIONS:
                r = nees_res[spec][c]
                g, a = conversion_noise_stds(sc_preset, args.dt, c)
                n_mean = second_half_mean(r["nees"])
                verdict = "inside band" if lo <= n_mean <= hi else ("overconfident" if n_mean > hi else "underconfident")
                print(f"    {c:<18s}{g:12.3e}{a:13.3e}{n_mean:9.2f}{second_half_mean(r['att_rmse']):16.4f}"
                      f"{second_half_mean(r['pos_rmse']):14.4f}  {verdict}")
            if args.block_check:
                blocks = block_nees_experiment(sc_preset, args.dt, args.duration, args.mc_trials,
                                               fresh_rng(), args.fix_rate)
                print("    block NEES, second half, same trials (3-state blocks: consistent = 3; joint: consistent = 15):")
                print(f"    {'':<18s}" + "".join(f"{b:>13s}" for b in BLOCKS))
                for c in CONVERSIONS:
                    print(f"    {c:<18s}" + "".join(f"{blocks[c][b]:13.2f}" for b in BLOCKS))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    if res is not None:
        h = hours[-1]
        for ax, s in zip(axes[0], SENSORS):
            r, p = res[s][h], preset[s]
            tau_fine = np.logspace(np.log10(r["tau"][0]), np.log10(r["tau"][-1]), 200)
            ax.loglog(r["tau"], r["adev"], "o-", ms=3, color="tab:blue", label=f"simulated ({h:g} h, seed 0)")
            ax.loglog(tau_fine, model_allan_deviation(tau_fine, p["density"], p["walk"]), "k--",
                      label=r"model $\sqrt{N^2/\tau + K^2\tau/3}$")
            if r["white_range"] is not None:
                t = np.geomspace(r["white_range"][0], max(1.0, r["white_range"][1]), 50)
                ax.loglog(t, r["fit_density"] / np.sqrt(t), color="tab:green", lw=1.5, label="slope -1/2 fit")
                ax.loglog([1.0], [r["fit_density"]], "s", color="tab:green", ms=8)
                ax.axvspan(*r["white_range"], color="tab:green", alpha=0.08)
            if r["walk_range"] is not None:
                t = np.geomspace(min(3.0, r["walk_range"][0]), r["walk_range"][1], 50)
                ax.loglog(t, r["fit_walk"] * np.sqrt(t / 3.0), color="tab:red", lw=1.5, label="slope +1/2 fit")
                ax.loglog([3.0], [r["fit_walk"]], "s", color="tab:red", ms=8)
                ax.axvspan(*r["walk_range"], color="tab:red", alpha=0.08)
            ax.set_title(f"{s}: Allan deviation (squares: read-off at 1 s and 3 s)")
            ax.set_xlabel(r"cluster time $\tau$ (s)")
            ax.set_ylabel(f"Allan deviation ({UNITS[s]['adev']})")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
        axes[1, 0].axis("off")
        rows = []
        for s in SENSORS:
            for key, name in (("density", "N"), ("walk", "K")):
                est = res[s][h][key]
                med, std, ok = percent_error_summary(est, preset[s][key])
                rows.append([f"{s} {name}", f"{preset[s][key]:.3e}", f"{np.nanmedian(est) if ok else np.nan:.3e}",
                             f"{med:+.1f} +- {std:.1f}"])
        tab = axes[1, 0].table(cellText=rows, colLabels=["parameter", "true", "median est.", "error % (med. +- std)"], loc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)
        tab.scale(1.0, 1.8)
        axes[1, 0].set_title(f"True vs estimated, {h:g} h records, {args.n_seeds} seeds")
    else:
        for ax in axes[0]:
            ax.axis("off")
        axes[1, 0].axis("off")
    if nees_res is not None:
        t_axis = np.arange(1, int(round(args.duration / args.dt)) + 1) * args.dt
        colors = dict(correct="tab:blue", density_as_std="tab:red", double_conversion="tab:orange")
        ax = axes[1, 1]
        for spec, style in zip(args.scenarios[:2], ("-", "--")):
            for c in CONVERSIONS:
                ax.semilogy(t_axis, nees_res[spec][c]["nees"], style, color=colors[c],
                            label=f"{c} (scenario {spec})")
        ax.axhline(15, color="k", ls=":", lw=1)
        ax.axhspan(*averaged_nees_bounds(15, args.mc_trials), color="k", alpha=0.08, label="95% band")
        ax.set_title(f"ESKF NEES vs conversion ({args.mc_trials} trials, consistent = 15)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("average NEES")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    else:
        axes[1, 1].axis("off")
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
