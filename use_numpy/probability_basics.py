'''
Probability basics for state estimation: three small experiments, each one a
"predict, run, check" exercise on an idea the Kalman filter uses.

Part 1, fusion. Two independent 1D estimates of one quantity x, say a
prediction (mean mu1, variance var1) and a measurement (mean mu2, variance
var2). Fusing them gives the inverse-variance weighted mean and a variance
smaller than either input:

    1/var = 1/var1 + 1/var2            (Barfoot 2017, eq. 2.69a, p. 22)
    mu/var = mu1/var1 + mu2/var2       (Barfoot 2017, eq. 2.69b, p. 22; 1D form in Fig. 2.3, p. 23)

so the weights are w1 = var2/(var1+var2) and w2 = var1/(var1+var2). With
var1 = 3 and var2 = 7 they are exactly 0.7 and 0.3 (a design choice, so the
numbers are easy to check by hand). Four checks:
  - closed form above;
  - Bayes' rule on a grid: posterior = prior pdf * likelihood, normalized to
    area 1 (Thrun et al. 2005, eqs. 2.14-2.15, p. 13; Barfoot 2017, eqs.
    2.8-2.9, p. 11), with mean and variance by numerical integration. The prior
    is N(mu1, var1) over x and the likelihood of the reading mu2 is N(mu2; x,
    var2), which as a function of x is the Gaussian N(x; mu2, var2);
  - the 1D Kalman gain K = var1/(var1+var2) and fused = mu1 + K (mu2 - mu1),
    var = (1 - K) var1 (Thrun et al. 2005, Table 3.1, lines 4-6, p. 36, with
    state-to-measurement map C = 1, prior variance var1, measurement variance
    var2);
  - Monte Carlo: in each trial draw both estimates around a fixed true value
    (independent noise), fuse them, and compare the variance of the fused
    estimate across trials with the formula, with each input alone and with a
    naive 50/50 average. The variance of a weighted sum of independent
    estimates, w1^2 var1 + w2^2 var2, is the linear-map rule G Sigma G^T of
    Barfoot 2017, eq. 2.58b (p. 20), with G = [w1 w2] and Sigma = diag(var1,
    var2) (independent means uncorrelated: Barfoot 2017, eqs. 2.17-2.18, p. 12).
    For the 50/50 average that is (var1+var2)/4.
The true value only sets where the Monte Carlo histogram sits: the fused
estimate is unbiased for any value (the weights sum to 1). The default is the
fused mean of the example (14.7 m), so the histogram can be drawn on the same
axis as the two input densities.

Part 2, propagation. A 2D Gaussian x ~ N(mean, P) pushed through a map y = g(x):
  - (a) linear map y = A x: the result is Gaussian with mean A mean and
    covariance A P A^T (Barfoot 2017, eqs. 2.58a-b, p. 20), compared with the
    sample mean and covariance of A applied to many samples (sample
    covariance with the N-1 divisor: Barfoot 2017, eq. 2.16b, p. 12);
  - (b) nonlinear map: range-bearing (polar) to Cartesian, x = r cos(theta),
    y = r sin(theta), the model of a range sensor with a bearing (angle)
    sensor. The linearization replaces g by its tangent at the mean,
    g(x) ~ g(mean) + G (x - mean), with G the Jacobian dg/dx at the mean
    (Barfoot 2017, eq. 2.85, p. 26; Thrun et al. 2005, eq. 3.50, p. 50), and
    passes the Gaussian through that tangent, so the linearized result is
    N(g(mean), G P G^T) (eq. 2.58 applied to the tangent). Here the Jacobian is
        G = [[cos(theta), -r sin(theta)],
             [sin(theta),  r cos(theta)]]     (derivative of (r cos, r sin)).
    The range r and bearing theta are independent with standard deviations
    range_std (m) and bearing_std (rad). The experiment sweeps bearing_std at
    fixed range, range_std and mean bearing, and draws the same underlying
    standard-normal numbers for every sweep value (common random numbers: only
    the scale of the bearing noise changes, so the range samples are identical
    across the sweep). Mismatch metrics between the linearized Gaussian
    N(mu_lin, P_lin) and the Monte Carlo samples (sample mean mu_s, sample
    covariance P_s):
      offset (m)        = |mu_lin - mu_s|, the distance between the two means;
      offset (sigma)    = sqrt(d^T P_s^-1 d) with d = mu_lin - mu_s, the same
                          offset as a Mahalanobis distance (Barfoot 2017,
                          p. 28) in units of the sample spread;
      cov err           = |P_lin - P_s|_F / |P_s|_F, the relative Frobenius norm
                          of the covariance difference (0 = identical);
      sample NEES       = the average over the samples y_i of
                          (y_i - mu_lin)^T P_lin^-1 (y_i - mu_lin), the same
                          quadratic form as the NEES of part 3 but with the
                          linearized Gaussian as the "reported" distribution. If
                          the samples really followed N(mu_lin, P_lin) it would
                          equal the dimension 2 (Bar-Shalom et al. 2001, eq.
                          1.4.17-8, p. 59). Equivalently it is
                          tr(P_lin^-1 (S + d d^T)) with S the sample covariance
                          using the N divisor (Barfoot 2017, eq. 2.92, p. 28),
                          so it rises when the linearized covariance is too
                          small or its mean is off. A value above 2 means the
                          linearized Gaussian is overconfident about where the
                          samples fall.
    The numbers are reported as measured; nothing is assumed about which sigma
    is "small enough".

Part 3, NEES. A stand-in for an estimator (no filter is run) of a 2D state:
in every trial the estimation error e is drawn fresh from the true covariance
P_true, N(0, P_true), but the estimator REPORTS P_rep = c * P_true. For each
trial the Normalized Estimation Error Squared is NEES = e^T P_rep^-1 e
(Bar-Shalom et al. 2001, eq. 5.4.2-2, p. 234; the quadratic form of
Barfoot 2017, p. 28). With c = 1 (honest) the NEES of one trial is chi-square
distributed with n = 2 degrees of freedom and averages n (Bar-Shalom et al.
2001, eqs. 5.4.1-4, 5.4.2-3, 1.4.17-7 to -9, pp. 59, 233-234). Averaged over
M independent trials, M * (average NEES) is chi-square with M n degrees of
freedom (Bar-Shalom et al. 2001, p. 235), which gives the 95% acceptance
interval of `averaged_nees_bounds` in utils.py. With c != 1, e^T (c P_true)^-1 e
= (e^T P_true^-1 e) / c, so the average NEES is n / c (algebra, not from a
source): c < 1 (overconfident, reported covariance too small) gives more than
n, c > 1 (underconfident) gives less. The fraction of single trials above the
95% point of the chi-square with 2 degrees of freedom, 5.991, is printed next
to its prediction chi2.sf(c * 5.991, 2) (same algebra). For 2 degrees of freedom
the chi-square density is q -> 0.5 exp(-q/2) (Bar-Shalom et al. 2001, eq.
1.4.17-10, p. 59, with n = 2), so P(q > t) = exp(-t/2) and the 95% point is
-2 ln 0.05 = 5.991; scipy (already a dependency) is used for the general case
and the script checks the two agree.

Scores printed: part 1 fused mean/variance from the three methods and the
Monte Carlo variances with their standard errors; part 2 the mismatch table;
part 3 average NEES against n and the 95% bounds, and the single-trial
fractions. Time per Monte Carlo trial and peak memory per part come from
`measure_performance` in utils.py.

Thrun et al. (2005) page, equation and table numbers above are from the
1999-2000 early-draft PDF that was read; they may differ from the 2005 print.

Run: python use_numpy/probability_basics.py --out assets/probability_basics.png
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from state_estimation_simulations.utils import measure_performance, averaged_nees_bounds, nees  # noqa: E402

PARTS = ("fusion", "propagation", "nees")
STATE_DIM = 2
# np.trapezoid exists from numpy 2.0; older numpy only has np.trapz.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------- part 1: fusion

def gaussian_pdf(x, mu, var):
    """1D Gaussian density N(x; mu, var) (Barfoot 2017, eq. 2.35).
    Arguments:
        x: points where to evaluate (any shape)
        mu: mean (same unit as x)
        var: variance (unit of x squared)
    Returns:
        density at x (1 / unit of x)
    """
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def fuse_inverse_variance(mu1, var1, mu2, var2):
    """Closed-form fusion of two independent 1D Gaussian estimates of one
    quantity (Barfoot 2017, eqs. 2.69a-b, p. 22, with K = 2 in 1D).
    Arguments:
        mu1, mu2: the two estimates (m)
        var1, var2: their variances (m^2)
    Returns:
        mu: fused mean (m)
        var: fused variance (m^2), smaller than both var1 and var2
        w1, w2: weights of mu1 and mu2 in the fused mean (unitless, sum to 1)
    """
    var = 1.0 / (1.0 / var1 + 1.0 / var2)
    w1, w2 = var / var1, var / var2
    return w1 * mu1 + w2 * mu2, var, w1, w2


def fuse_kalman_gain(mu1, var1, mu2, var2):
    """The same fusion written as a Kalman update (Thrun et al. 2005, Table 3.1,
    lines 4-6, p. 36, with C = 1): the first estimate is the prediction, the
    second one the measurement.
    Arguments:
        mu1, var1: prediction mean (m) and variance (m^2)
        mu2, var2: measurement (m) and its variance (m^2)
    Returns:
        K: gain (unitless), the fraction of the innovation mu2 - mu1 that is added
        mu: updated mean mu1 + K (mu2 - mu1) (m)
        var: updated variance (1 - K) var1 (m^2)
    """
    K = var1 / (var1 + var2)
    return K, mu1 + K * (mu2 - mu1), (1.0 - K) * var1


def fuse_grid_bayes(mu1, var1, mu2, var2, n_grid=20001, n_sigma=10.0):
    """Bayes' rule on a grid: posterior = prior * likelihood / evidence, where
    the prior is N(x; mu1, var1) and the likelihood of the reading mu2 is
    N(mu2; x, var2) = N(x; mu2, var2). The evidence (the area under
    prior * likelihood) is a numerical integral, and so are the posterior
    mean and variance (Thrun et al. 2005, eq. 2.14).
    Arguments:
        mu1, var1: prior mean (m) and variance (m^2)
        mu2, var2: measurement (m) and its variance (m^2)
        n_grid: number of grid points
        n_sigma: grid half-width beyond the two means, in largest standard deviations
    Returns:
        x: grid (m), prior pdf, likelihood, posterior pdf (all 1/m, on the grid),
        mean (m) and variance (m^2) of the posterior by numerical integration
    """
    half = n_sigma * np.sqrt(max(var1, var2))
    x = np.linspace(min(mu1, mu2) - half, max(mu1, mu2) + half, n_grid)
    prior = gaussian_pdf(x, mu1, var1)
    likelihood = gaussian_pdf(mu2, x, var2)
    unnormalized = prior * likelihood
    posterior = unnormalized / _trapezoid(unnormalized, x)
    mean = _trapezoid(x * posterior, x)
    var = _trapezoid((x - mean) ** 2 * posterior, x)
    return x, prior, likelihood, posterior, mean, var


def run_fusion(mu1, var1, mu2, var2, true_value, n_trials, seed_seq):
    """Part 1: the closed form, the grid Bayes posterior, the Kalman-gain form
    and a Monte Carlo of the fused estimate.
    Arguments:
        mu1, var1: first estimate (the "prediction") mean (m) and variance (m^2)
        mu2, var2: second estimate (the "measurement") mean (m) and variance (m^2)
        true_value: true quantity that the Monte Carlo trials estimate (m)
        n_trials: number of Monte Carlo trials
        seed_seq: numpy SeedSequence; the generator is built inside so a second
            call (measure_performance runs every function twice) gives the same numbers
    Returns:
        dict with the closed form ('mu', 'var', 'w1', 'w2'), the Kalman form
        ('K', 'mu_gain', 'var_gain'), the grid posterior ('grid', 'prior',
        'likelihood', 'posterior', 'mu_grid', 'var_grid'), and the Monte Carlo
        sample variances ('var_mc': dict name -> variance across trials (m^2)),
        the fused estimates ('fused') and 'max_gain_form_diff', the largest
        |fused - (x1 + K (x2 - x1))| over the trials (m)
    """
    rng = np.random.default_rng(seed_seq)
    mu, var, w1, w2 = fuse_inverse_variance(mu1, var1, mu2, var2)
    K, mu_gain, var_gain = fuse_kalman_gain(mu1, var1, mu2, var2)
    grid, prior, likelihood, posterior, mu_grid, var_grid = fuse_grid_bayes(mu1, var1, mu2, var2)

    # Fresh independent noise on both estimates in every trial.
    x1 = true_value + np.sqrt(var1) * rng.standard_normal(n_trials)
    x2 = true_value + np.sqrt(var2) * rng.standard_normal(n_trials)
    fused = w1 * x1 + w2 * x2
    naive = 0.5 * (x1 + x2)
    var_mc = {"input 1 alone": np.var(x1, ddof=1), "input 2 alone": np.var(x2, ddof=1),
              "naive 50/50": np.var(naive, ddof=1), "fused": np.var(fused, ddof=1)}
    return {"mu": mu, "var": var, "w1": w1, "w2": w2, "K": K, "mu_gain": mu_gain, "var_gain": var_gain,
            "grid": grid, "prior": prior, "likelihood": likelihood, "posterior": posterior,
            "mu_grid": mu_grid, "var_grid": var_grid, "var_mc": var_mc, "fused": fused,
            "mean_error_fused": float(np.mean(fused) - true_value),
            "max_gain_form_diff": float(np.max(np.abs(fused - (x1 + K * (x2 - x1)))))}


def print_fusion(res, mu1, var1, mu2, var2, n_trials):
    """Prints the part 1 results.
    Arguments:
        res: dict from run_fusion
        mu1, var1, mu2, var2: the two estimates (m, m^2)
        n_trials: number of Monte Carlo trials used
    Returns:
        None
    """
    print(f"\n=== Part 1: fusion of two independent estimates: {mu1:g} m (var {var1:g} m^2) "
          f"and {mu2:g} m (var {var2:g} m^2) ===")
    print(f"inverse-variance weights: w1 = {res['w1']:.4f}, w2 = {res['w2']:.4f}")
    print(f"{'method':<24s}{'mean (m)':>12s}{'variance (m^2)':>16s}")
    print(f"{'closed form':<24s}{res['mu']:12.6f}{res['var']:16.6f}")
    print(f"{'grid Bayes':<24s}{res['mu_grid']:12.6f}{res['var_grid']:16.6f}")
    print(f"{'Kalman gain form':<24s}{res['mu_gain']:12.6f}{res['var_gain']:16.6f}")
    print(f"Kalman gain K = var1/(var1+var2) = {res['K']:.4f}: fused = {mu1:g} + {res['K']:.4f} "
          f"* ({mu2:g} - {mu1:g}) = {res['mu_gain']:.4f} m; "
          f"largest |fused - (x1 + K (x2 - x1))| over trials = {res['max_gain_form_diff']:.2e} m")
    predicted = {"input 1 alone": var1, "input 2 alone": var2, "naive 50/50": 0.25 * (var1 + var2),
                 "fused": res["var"]}
    # The variance of a sample variance of N Gaussian draws is 2 var^2 / (N - 1): a chi-square with N - 1
    # degrees of freedom has variance 2 (N - 1) (Bar-Shalom et al. 2001, eq. 1.4.17-9, p. 59).
    print(f"Monte Carlo variance of the estimate across {n_trials} trials "
          f"(one standard error is {np.sqrt(2.0 / (n_trials - 1)):.2%} of each value):")
    print(f"  {'estimate':<16s}{'formula (m^2)':>15s}{'Monte Carlo (m^2)':>19s}")
    for name, v in res["var_mc"].items():
        print(f"  {name:<16s}{predicted[name]:15.4f}{v:19.4f}")
    print(f"  mean error of the fused estimate over the trials: {res['mean_error_fused']:+.4f} m "
          f"(standard error {np.sqrt(res['var'] / n_trials):.4f} m)")


# --------------------------------------------------------------------------- part 2: propagation

def polar_to_cartesian(r, theta):
    """Range-bearing to Cartesian: (r cos theta, r sin theta).
    Arguments:
        r: range (m), scalar or array
        theta: bearing (rad), same shape as r
    Returns:
        array of shape (..., 2): x and y (m)
    """
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=-1)


def polar_jacobian(r, theta):
    """Jacobian of polar_to_cartesian with respect to (r, theta).
    Arguments:
        r: range (m)
        theta: bearing (rad)
    Returns:
        2x2 matrix [[dx/dr, dx/dtheta], [dy/dr, dy/dtheta]] (columns: unitless, m/rad)
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -r * s], [s, r * c]])


def propagate_linear(mean, cov, A):
    """Exact Gaussian through y = A x (Barfoot 2017, eqs. 2.58a-b, p. 20).
    Arguments:
        mean: input mean (2,)
        cov: input covariance (2, 2)
        A: linear map (2, 2)
    Returns:
        mean A mean (2,) and covariance A cov A^T (2, 2)
    """
    return A @ mean, A @ cov @ A.T


def propagate_polar_linearized(r0, theta0, range_std, bearing_std):
    """Linearized propagation of independent (range, bearing) noise through
    polar_to_cartesian: the Gaussian N(g(mean), G P G^T) with G the Jacobian at
    the mean (Barfoot 2017, eq. 2.85, p. 26).
    Arguments:
        r0: mean range (m)
        theta0: mean bearing (rad)
        range_std: range standard deviation (m)
        bearing_std: bearing standard deviation (rad)
    Returns:
        mean (2,) in m and covariance (2, 2) in m^2 of the linearized Cartesian Gaussian
    """
    G = polar_jacobian(r0, theta0)
    P = np.diag([range_std ** 2, bearing_std ** 2])
    return polar_to_cartesian(r0, theta0), G @ P @ G.T


def gaussian_mismatch(mu_lin, P_lin, samples):
    """Mismatch between a Gaussian N(mu_lin, P_lin) and a sample cloud: the
    four metrics defined in the module docstring (part 2).
    Arguments:
        mu_lin: mean of the Gaussian (2,) in m
        P_lin: covariance of the Gaussian (2, 2) in m^2
        samples: points (n, 2) in m
    Returns:
        dict: 'mu_s', 'P_s' (sample mean in m, sample covariance in m^2, N-1 divisor),
        'offset' (m), 'offset_sigma', 'cov_err', 'sample_nees' (unitless)
    """
    mu_s = samples.mean(axis=0)
    P_s = np.cov(samples, rowvar=False)
    d = mu_lin - mu_s
    dev = samples - mu_lin
    return {"mu_s": mu_s, "P_s": P_s, "offset": float(np.linalg.norm(d)),
            "offset_sigma": float(np.sqrt(d @ np.linalg.solve(P_s, d))),
            "cov_err": float(np.linalg.norm(P_lin - P_s) / np.linalg.norm(P_s)),
            "sample_nees": float(np.mean(np.einsum("ij,ij->i", dev, np.linalg.solve(P_lin, dev.T).T)))}


def run_propagation(lin_mean, lin_cov, A, r0, theta0, range_std, bearing_stds, n_samples, seed_seq):
    """Part 2: (a) a linear map, (b) the polar sweep.
    Arguments:
        lin_mean: mean of the input Gaussian of (a), shape (2,)
        lin_cov: covariance of the input Gaussian of (a), shape (2, 2)
        A: linear map of (a), shape (2, 2)
        r0: mean range of (b) (m)
        theta0: mean bearing of (b) (rad)
        range_std: range standard deviation of (b) (m)
        bearing_stds: bearing standard deviations to sweep in (b) (rad)
        n_samples: Monte Carlo samples per case
        seed_seq: numpy SeedSequence (the generator is built inside; two child
            streams: one for (a), one for (b))
    Returns:
        dict: 'linear' (closed form 'mean', 'cov', sample 'mu_s', 'P_s'),
        'polar' (list of dicts, one per bearing std, with 'bearing_std', the
        linearized 'mu_lin', 'P_lin', the 'samples' and the gaussian_mismatch metrics)
    """
    ss_lin, ss_pol = seed_seq.spawn(2)
    rng_lin, rng_pol = np.random.default_rng(ss_lin), np.random.default_rng(ss_pol)

    # (a) linear map
    x = rng_lin.multivariate_normal(lin_mean, lin_cov, size=n_samples)
    mean_cf, cov_cf = propagate_linear(lin_mean, lin_cov, A)
    y = x @ A.T
    lin = {"mean": mean_cf, "cov": cov_cf, "mu_s": y.mean(axis=0), "P_s": np.cov(y, rowvar=False)}

    # (b) polar sweep: ONE set of standard-normal draws, rescaled by each bearing std.
    z = rng_pol.standard_normal((n_samples, 2))
    r = r0 + range_std * z[:, 0]
    polar = []
    for bearing_std in bearing_stds:
        theta = theta0 + bearing_std * z[:, 1]
        mu_lin, P_lin = propagate_polar_linearized(r0, theta0, range_std, bearing_std)
        samples = polar_to_cartesian(r, theta)
        polar.append({"bearing_std": bearing_std, "mu_lin": mu_lin, "P_lin": P_lin, "samples": samples,
                      **gaussian_mismatch(mu_lin, P_lin, samples)})
    return {"linear": lin, "polar": polar}


def print_propagation(res, A, lin_mean, lin_cov, r0, theta0, range_std, n_samples):
    """Prints the part 2 results.
    Arguments:
        res: dict from run_propagation
        A: linear map of (a), (2, 2)
        lin_mean, lin_cov: input Gaussian of (a), (2,) and (2, 2)
        r0: mean range (m); theta0: mean bearing (rad); range_std: range std (m)
        n_samples: samples per case
    Returns:
        None
    """
    lin = res["linear"]
    print(f"\n=== Part 2: a 2D Gaussian through a map, {n_samples} samples per case ===")
    print(f"(a) linear map y = A x, A = {A.tolist()}, input mean {lin_mean.tolist()}, "
          f"input covariance {lin_cov.round(4).tolist()}")
    print(f"    closed form  mean {np.round(lin['mean'], 4).tolist()}  cov {np.round(lin['cov'], 4).tolist()}")
    print(f"    sample       mean {np.round(lin['mu_s'], 4).tolist()}  cov {np.round(lin['P_s'], 4).tolist()}")
    print(f"    relative Frobenius error of the sample covariance: "
          f"{np.linalg.norm(lin['cov'] - lin['P_s']) / np.linalg.norm(lin['cov']):.4f}")
    print(f"(b) polar -> Cartesian, range {r0:g} m +/- {range_std:g} m, bearing {np.degrees(theta0):g} deg "
          f"+/- bearing std; linearized Gaussian vs sample")
    print(f"    {'bearing std':>11s}{'offset (m)':>12s}{'offset (sigma)':>16s}{'cov err':>10s}{'sample NEES':>13s}")
    for c in res["polar"]:
        print(f"    {np.degrees(c['bearing_std']):9.1f} deg{c['offset']:12.4f}{c['offset_sigma']:16.3f}"
              f"{c['cov_err']:10.3f}{c['sample_nees']:13.3f}")
    print("    (sample NEES = 2.000 would mean the linearized Gaussian describes the cloud exactly; "
          "offset in units of the sample standard deviation)")


# --------------------------------------------------------------------------- part 3: NEES

def run_nees(P_true, c_values, n_trials, seed_seq, threshold=None):
    """Part 3: errors from P_true, covariance reported as c * P_true.
    The same errors are scored against every c (the c values change only the
    reported covariance, not the draws).
    Arguments:
        P_true: true error covariance (2, 2), m^2
        c_values: scale factors c of the reported covariance (unitless)
        n_trials: number of Monte Carlo trials
        seed_seq: numpy SeedSequence (the generator is built inside)
        threshold: single-trial NEES threshold; default the 95% point of the
            chi-square with STATE_DIM degrees of freedom
    Returns:
        dict: 'threshold', 'chi2_check' (|threshold - (-2 ln 0.05)|, valid for 2 dof),
        'cases' (list of dicts per c: 'c', 'nees' (per-trial array), 'avg',
        'predicted_avg' (= n / c), 'frac_above', 'predicted_frac_above')
    """
    rng = np.random.default_rng(seed_seq)
    if threshold is None:
        threshold = float(chi2.ppf(0.95, STATE_DIM))
    L = np.linalg.cholesky(P_true)
    errors = rng.standard_normal((n_trials, STATE_DIM)) @ L.T  # e ~ N(0, P_true), fresh each trial
    cases = []
    for c in c_values:
        P_rep = c * P_true
        values = np.array([nees(e, P_rep) for e in errors])
        cases.append({"c": c, "nees": values, "avg": float(values.mean()), "predicted_avg": STATE_DIM / c,
                      "frac_above": float(np.mean(values > threshold)),
                      "predicted_frac_above": float(chi2.sf(c * threshold, STATE_DIM))})
    return {"threshold": threshold, "chi2_check": abs(threshold + 2.0 * np.log(0.05)), "cases": cases}


def print_nees(res, P_true, n_trials):
    """Prints the part 3 results.
    Arguments:
        res: dict from run_nees
        P_true: true error covariance (2, 2), m^2
        n_trials: number of Monte Carlo trials
    Returns:
        None
    """
    lo, hi = averaged_nees_bounds(STATE_DIM, n_trials)
    print(f"\n=== Part 3: NEES of an estimator that reports c * P_true, {n_trials} trials ===")
    print(f"P_true = {P_true.round(4).tolist()} m^2; single-trial threshold {res['threshold']:.3f} "
          f"(95% point of chi-square with {STATE_DIM} dof; -2 ln 0.05 = {-2.0 * np.log(0.05):.3f}, "
          f"difference {res['chi2_check']:.1e})")
    print(f"dimension n = {STATE_DIM}; 95% bounds for the {n_trials}-trial average NEES: [{lo:.3f}, {hi:.3f}]")
    print(f"  {'c':>6s}{'reported':>12s}{'avg NEES':>10s}{'n / c':>8s}{'in bounds':>11s}"
          f"{'frac > thr':>12s}{'predicted':>11s}")
    for k in res["cases"]:
        label = "honest" if k["c"] == 1 else ("overconf." if k["c"] < 1 else "underconf.")
        print(f"  {k['c']:6.2f}{label:>12s}{k['avg']:10.3f}{k['predicted_avg']:8.3f}"
              f"{'yes' if lo <= k['avg'] <= hi else 'no':>11s}{k['frac_above']:12.3f}{k['predicted_frac_above']:11.3f}")


# --------------------------------------------------------------------------- figure

def ellipse_points(mean, cov, n_sigma, n_pts=200):
    """Points of the n_sigma uncertainty ellipse {x : (x-mean)^T cov^-1 (x-mean) = n_sigma^2}
    (Barfoot 2017, Fig. 2.6, p. 29).
    Arguments:
        mean: center (2,)
        cov: covariance (2, 2)
        n_sigma: ellipse size in standard deviations
        n_pts: points on the curve
    Returns:
        array (n_pts, 2)
    """
    t = np.linspace(0.0, 2.0 * np.pi, n_pts)
    return mean + n_sigma * (np.linalg.cholesky(cov) @ np.stack([np.cos(t), np.sin(t)])).T


def make_figure(mu1, var1, mu2, var2, fusion, propagation, nees_res, n_trials_nees, plot_stds):
    """Three-part figure: fusion (top), propagation at a small and a large
    bearing std (bottom left, middle), NEES (bottom right).
    Arguments:
        mu1, var1, mu2, var2: part 1 inputs (m, m^2)
        fusion, propagation, nees_res: dicts from run_fusion, run_propagation, run_nees
        n_trials_nees: Monte Carlo trials of part 3
        plot_stds: indices into propagation['polar'] of the two cases drawn
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.38, wspace=0.28)

    # (1) fusion
    ax = fig.add_subplot(gs[0, :])
    x = fusion["grid"]
    ax.hist(fusion["fused"], bins=80, density=True, color="tab:green", alpha=0.30,
            label=f"Monte Carlo: fused estimate of {len(fusion['fused'])} trials")
    ax.plot(x, gaussian_pdf(x, mu1, var1), color="tab:blue", label=f"input 1: N({mu1:g}, {var1:g})")
    ax.plot(x, gaussian_pdf(x, mu2, var2), color="tab:orange", label=f"input 2: N({mu2:g}, {var2:g})")
    ax.plot(x, gaussian_pdf(x, fusion["mu"], fusion["var"]), color="tab:green", linewidth=2.5,
            label=f"fused, closed form: N({fusion['mu']:.1f}, {fusion['var']:.1f})")
    ax.plot(x, fusion["posterior"], color="black", linestyle=":", linewidth=2.5, label="fused, Bayes on a grid")
    ax.set_xlim(min(mu1, mu2) - 3.0 * np.sqrt(max(var1, var2)), max(mu1, mu2) + 3.0 * np.sqrt(max(var1, var2)))
    ax.set_xlabel("quantity x (m)")
    ax.set_ylabel("probability density (1/m)")
    ax.set_title("1. Fusion: two estimates give one narrower estimate, nearer the more certain one")
    ax.legend(fontsize=8, loc="upper left")

    # (2) propagation: small and large bearing std
    for col, idx in enumerate(plot_stds):
        ax = fig.add_subplot(gs[1, col])
        c = propagation["polar"][idx]
        pts = c["samples"][:2000]
        ax.scatter(pts[:, 0], pts[:, 1], s=2, color="0.7", label="samples (2000 shown)")
        for k, ls in ((1, "-"), (2, "-")):
            e_lin = ellipse_points(c["mu_lin"], c["P_lin"], k)
            e_s = ellipse_points(c["mu_s"], c["P_s"], k)
            ax.plot(e_lin[:, 0], e_lin[:, 1], color="tab:red", linestyle="--", linewidth=1.6,
                    label="linearized 1σ, 2σ" if k == 1 else None)
            ax.plot(e_s[:, 0], e_s[:, 1], color="tab:blue", linestyle=ls, linewidth=1.6,
                    label="sample covariance 1σ, 2σ" if k == 1 else None)
        ax.plot(*c["mu_lin"], marker="x", color="tab:red", markersize=9, linestyle="none", label="linearized mean")
        ax.plot(*c["mu_s"], marker="+", color="tab:blue", markersize=11, linestyle="none", label="sample mean")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"2{'ab'[col]}. Polar → Cartesian, bearing std {np.degrees(c['bearing_std']):g}°\n"
                     f"cov err {c['cov_err']:.2f}, offset {c['offset_sigma']:.2f}σ", fontsize=10)
        if col == 0:
            handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, fontsize=8, loc="lower center", bbox_to_anchor=(0.33, -0.045), ncol=3,
               title="panels 2a and 2b", title_fontsize=8)

    # (3) NEES
    ax = fig.add_subplot(gs[1, 2])
    cases = nees_res["cases"]
    lo, hi = averaged_nees_bounds(STATE_DIM, n_trials_nees)
    labels = [f"c = {k['c']:g}\n" + ("honest" if k["c"] == 1 else "over-\nconfident" if k["c"] < 1 else "under-\nconfident")
              for k in cases]
    colors = ["tab:green" if lo <= k["avg"] <= hi else "tab:red" for k in cases]
    bars = ax.bar(labels, [k["avg"] for k in cases], color=colors, alpha=0.8)
    ax.axhline(STATE_DIM, color="black", linewidth=1.2, label=f"expected for a consistent estimator, n = {STATE_DIM}")
    ax.axhspan(lo, hi, color="black", alpha=0.15, label=f"95% bounds, {n_trials_nees}-trial average")
    for b, k in zip(bars, cases):
        ax.text(b.get_x() + b.get_width() / 2, k["avg"], f"{k['avg']:.2f}\n{k['frac_above']:.1%} of trials\n> {nees_res['threshold']:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(f"average NEES over {n_trials_nees} trials")
    ax.set_ylim(0, max(k["avg"] for k in cases) * 1.35)
    ax.set_title("3. NEES: reported covariance c·P_true", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    return fig


# --------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", action="append", choices=PARTS, default=None,
                        help="Part to run; repeat the flag to choose several (default: all three)")
    # part 1
    parser.add_argument("--mu1", type=float, default=15.0, help="Part 1: first estimate, the prediction (m)")
    parser.add_argument("--var1", type=float, default=3.0, help="Part 1: variance of the first estimate (m^2)")
    parser.add_argument("--mu2", type=float, default=14.0, help="Part 1: second estimate, the measurement (m)")
    parser.add_argument("--var2", type=float, default=7.0, help="Part 1: variance of the second estimate (m^2)")
    parser.add_argument("--true-value", type=float, default=14.7,
                        help="Part 1: true quantity the Monte Carlo trials estimate (m); default is the "
                             "fused mean so the histogram sits under the fused curve")
    parser.add_argument("--n-fusion-trials", type=int, default=100000, help="Part 1: Monte Carlo trials")
    # part 2
    parser.add_argument("--n-samples", type=int, default=200000, help="Part 2: Monte Carlo samples per case")
    parser.add_argument("--lin-mean", type=float, nargs=2, default=[1.0, 2.0], metavar=("X", "Y"),
                        help="Part 2a: mean of the input Gaussian (m)")
    parser.add_argument("--lin-std", type=float, nargs=2, default=[1.0, 0.5], metavar=("SX", "SY"),
                        help="Part 2a: standard deviations of the input Gaussian (m)")
    parser.add_argument("--lin-corr", type=float, default=0.5, help="Part 2a: correlation of the input Gaussian (-)")
    parser.add_argument("--lin-map", type=float, nargs=4, default=[1.0, 0.5, 0.0, 2.0],
                        metavar=("A11", "A12", "A21", "A22"), help="Part 2a: linear map A, row by row (-)")
    parser.add_argument("--range", type=float, default=10.0, help="Part 2b: mean range (m)")
    parser.add_argument("--range-std", type=float, default=0.2, help="Part 2b: range standard deviation (m)")
    parser.add_argument("--bearing-deg", type=float, default=0.0, help="Part 2b: mean bearing (deg)")
    parser.add_argument("--bearing-std-deg", type=float, nargs="+", default=[1.0, 5.0, 15.0, 30.0],
                        help="Part 2b: bearing standard deviations to sweep (deg); the figure draws the "
                             "first and the last")
    # part 3
    parser.add_argument("--nees-std", type=float, nargs=2, default=[0.5, 0.2], metavar=("S1", "S2"),
                        help="Part 3: true error standard deviations of the 2D state (m)")
    parser.add_argument("--nees-corr", type=float, default=0.6, help="Part 3: true error correlation (-)")
    parser.add_argument("--c-values", type=float, nargs="+", default=[1.0, 0.25, 4.0],
                        help="Part 3: scale c of the reported covariance, P_rep = c P_true "
                             "(1 honest, <1 overconfident, >1 underconfident)")
    parser.add_argument("--n-nees-trials", type=int, default=1000, help="Part 3: Monte Carlo trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")
    args = parser.parse_args()

    parts = set(args.part) if args.part else set(PARTS)
    # One child stream per part, always spawned, so a part's numbers do not depend on which parts are selected.
    ss = dict(zip(PARTS, np.random.SeedSequence(args.seed).spawn(len(PARTS))))
    print(f"Probability basics, seed {args.seed}, parts: {', '.join(p for p in PARTS if p in parts)}")

    results, perf = {}, {}
    if "fusion" in parts:
        res, t, mem = measure_performance(run_fusion, args.mu1, args.var1, args.mu2, args.var2, args.true_value,
                                          args.n_fusion_trials, ss["fusion"], n_steps=args.n_fusion_trials)
        print_fusion(res, args.mu1, args.var1, args.mu2, args.var2, args.n_fusion_trials)
        results["fusion"], perf["fusion"] = res, (t, mem, "trial")

    if "propagation" in parts:
        lin_mean = np.array(args.lin_mean)
        lin_std = np.array(args.lin_std)
        lin_cov = np.array([[lin_std[0] ** 2, args.lin_corr * lin_std[0] * lin_std[1]],
                            [args.lin_corr * lin_std[0] * lin_std[1], lin_std[1] ** 2]])
        A = np.array(args.lin_map).reshape(2, 2)
        bearing_stds = np.radians(args.bearing_std_deg)
        n_cases = 1 + len(bearing_stds)
        res, t, mem = measure_performance(
            run_propagation, lin_mean, lin_cov, A, args.range, np.radians(args.bearing_deg), args.range_std,
            bearing_stds, args.n_samples, ss["propagation"], n_steps=n_cases * args.n_samples)
        print_propagation(res, A, lin_mean, lin_cov, args.range, np.radians(args.bearing_deg), args.range_std,
                          args.n_samples)
        results["propagation"], perf["propagation"] = res, (t, mem, "sample")

    if "nees" in parts:
        s = np.array(args.nees_std)
        P_true = np.array([[s[0] ** 2, args.nees_corr * s[0] * s[1]], [args.nees_corr * s[0] * s[1], s[1] ** 2]])
        res, t, mem = measure_performance(run_nees, P_true, args.c_values, args.n_nees_trials, ss["nees"],
                                          n_steps=args.n_nees_trials * len(args.c_values))
        print_nees(res, P_true, args.n_nees_trials)
        results["nees"], perf["nees"] = res, (t, mem, "NEES evaluation")

    print("\nAverage time per unit of work and peak memory (whole part), empirical:")
    for name, (t, mem, unit) in perf.items():
        print(f"  {name:<12s} avg time={t * 1e6:9.3f} us per {unit:<16s} | peak mem={mem / 1024.0:10.1f} KB")

    if len(parts) < len(PARTS):
        if args.out:
            print("\nNo figure saved: the figure shows all three parts; run without --part to get it.")
        return
    fig = make_figure(args.mu1, args.var1, args.mu2, args.var2, results["fusion"], results["propagation"],
                      results["nees"], args.n_nees_trials, [0, len(args.bearing_std_deg) - 1])
    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
