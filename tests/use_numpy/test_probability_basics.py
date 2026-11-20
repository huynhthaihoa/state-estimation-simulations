import sys

import numpy as np
import pytest

from state_estimation_simulations.use_numpy import probability_basics as _probability_basics
from state_estimation_simulations.utils import averaged_nees_bounds


@pytest.fixture
def pb():
    return _probability_basics


def _seed(i=0, seed=0):
    """Child i of SeedSequence(seed): the same stream the script gives part i."""
    return np.random.SeedSequence(seed).spawn(3)[i]


# --- part 1: fusion ---

def test_fusion_weights_are_70_30_for_variances_3_and_7(pb):
    mu, var, w1, w2 = pb.fuse_inverse_variance(15.0, 3.0, 14.0, 7.0)
    assert w1 == pytest.approx(0.7) and w2 == pytest.approx(0.3)
    assert mu == pytest.approx(14.7)
    assert var == pytest.approx(2.1)


def test_fused_variance_below_both_inputs(pb):
    for v1, v2 in [(3.0, 7.0), (1.0, 1.0), (0.01, 100.0)]:
        _, var, _, _ = pb.fuse_inverse_variance(0.0, v1, 1.0, v2)
        assert var < min(v1, v2)


def test_closed_form_equals_grid_bayes(pb):
    mu, var, _, _ = pb.fuse_inverse_variance(15.0, 3.0, 14.0, 7.0)
    _, _, _, posterior, mu_g, var_g = pb.fuse_grid_bayes(15.0, 3.0, 14.0, 7.0)
    assert mu_g == pytest.approx(mu, abs=1e-8)
    assert var_g == pytest.approx(var, abs=1e-8)
    assert posterior.min() >= 0.0


def test_grid_posterior_is_the_fused_gaussian_pointwise(pb):
    mu, var, _, _ = pb.fuse_inverse_variance(15.0, 3.0, 14.0, 7.0)
    x, _, _, posterior, _, _ = pb.fuse_grid_bayes(15.0, 3.0, 14.0, 7.0)
    assert np.max(np.abs(posterior - pb.gaussian_pdf(x, mu, var))) < 1e-8


def test_kalman_gain_form_equals_inverse_variance(pb):
    for mu1, v1, mu2, v2 in [(15.0, 3.0, 14.0, 7.0), (0.0, 0.5, 2.0, 4.0)]:
        mu, var, _, _ = pb.fuse_inverse_variance(mu1, v1, mu2, v2)
        K, mu_k, var_k = pb.fuse_kalman_gain(mu1, v1, mu2, v2)
        assert K == pytest.approx(v1 / (v1 + v2))
        assert mu_k == pytest.approx(mu) and var_k == pytest.approx(var)
    assert pb.fuse_kalman_gain(15.0, 3.0, 14.0, 7.0)[0] == pytest.approx(0.3)


def test_fusion_monte_carlo_variances_match_formulas(pb):
    n = 100000
    res = pb.run_fusion(15.0, 3.0, 14.0, 7.0, 14.7, n, _seed(0))
    tol = 5.0 * np.sqrt(2.0 / (n - 1))  # five standard errors of a sample variance
    for name, v in {"input 1 alone": 3.0, "input 2 alone": 7.0, "naive 50/50": 2.5, "fused": 2.1}.items():
        assert res["var_mc"][name] == pytest.approx(v, rel=tol)
    assert res["var_mc"]["fused"] < res["var_mc"]["naive 50/50"] < res["var_mc"]["input 1 alone"]
    assert abs(res["mean_error_fused"]) < 5.0 * np.sqrt(2.1 / n)  # unbiased
    assert res["max_gain_form_diff"] < 1e-12


def test_fusion_is_deterministic_for_a_seed(pb):
    a = pb.run_fusion(15.0, 3.0, 14.0, 7.0, 14.7, 1000, _seed(0))
    b = pb.run_fusion(15.0, 3.0, 14.0, 7.0, 14.7, 1000, _seed(0))
    assert np.array_equal(a["fused"], b["fused"])


# --- part 2: propagation ---

def test_polar_jacobian_matches_finite_differences(pb):
    for r, th in [(10.0, 0.0), (3.0, 0.7), (25.0, -2.1)]:
        G = pb.polar_jacobian(r, th)
        eps = 1e-6
        G_fd = np.column_stack([
            (pb.polar_to_cartesian(r + eps, th) - pb.polar_to_cartesian(r - eps, th)) / (2 * eps),
            (pb.polar_to_cartesian(r, th + eps) - pb.polar_to_cartesian(r, th - eps)) / (2 * eps)])
        assert np.allclose(G, G_fd, atol=1e-7)


def test_linear_propagation_closed_form_matches_samples(pb):
    mean, cov = np.array([1.0, 2.0]), np.array([[1.0, 0.25], [0.25, 0.25]])
    A = np.array([[1.0, 0.5], [0.0, 2.0]])
    res = pb.run_propagation(mean, cov, A, 10.0, 0.0, 0.2, np.radians([1.0]), 100000, _seed(1))
    lin = res["linear"]
    assert np.allclose(lin["mean"], A @ mean) and np.allclose(lin["cov"], A @ cov @ A.T)
    assert np.allclose(lin["mu_s"], lin["mean"], atol=0.03)
    assert np.allclose(lin["P_s"], lin["cov"], atol=0.03)


def test_linearized_polar_equals_sample_for_tiny_bearing_std(pb):
    mean, cov, A = np.zeros(2), np.eye(2), np.eye(2)
    res = pb.run_propagation(mean, cov, A, 10.0, 0.0, 0.2, np.radians([0.1]), 100000, _seed(1))
    c = res["polar"][0]
    assert c["cov_err"] < 0.02
    assert c["offset_sigma"] < 0.01
    assert c["sample_nees"] == pytest.approx(2.0, abs=0.05)


def test_polar_mismatch_grows_with_bearing_std(pb):
    mean, cov, A = np.zeros(2), np.eye(2), np.eye(2)
    res = pb.run_propagation(mean, cov, A, 10.0, 0.0, 0.2, np.radians([1.0, 5.0, 15.0, 30.0]), 50000, _seed(1))
    # At 1 deg the covariance error is sampling noise (~1%), so the ordering is only claimed from 5 deg up.
    errs = [c["cov_err"] for c in res["polar"]]
    assert errs[1] < errs[2] < errs[3] and errs[3] > 10 * errs[0]
    nees_vals = [c["sample_nees"] for c in res["polar"]]
    assert nees_vals == sorted(nees_vals)


def test_sweep_values_do_not_change_other_draws(pb):
    # Common random numbers: dropping sweep values must not change the others' samples.
    mean, cov, A = np.zeros(2), np.eye(2), np.eye(2)
    a = pb.run_propagation(mean, cov, A, 10.0, 0.0, 0.2, np.radians([5.0, 30.0]), 2000, _seed(1))
    b = pb.run_propagation(mean, cov, A, 10.0, 0.0, 0.2, np.radians([5.0]), 2000, _seed(1))
    assert np.array_equal(a["polar"][0]["samples"], b["polar"][0]["samples"])
    # The radius of each sample is the drawn range r, so it must be identical at every bearing std.
    r_small = np.hypot(*a["polar"][0]["samples"].T)
    r_large = np.hypot(*a["polar"][1]["samples"].T)
    assert np.allclose(r_small, r_large)


# --- part 3: NEES ---

P_TRUE = np.array([[0.25, 0.06], [0.06, 0.04]])


def test_chi2_95_point_for_2_dof_is_minus_2_ln_0_05(pb):
    res = pb.run_nees(P_TRUE, [1.0], 10, _seed(2))
    assert res["threshold"] == pytest.approx(5.991465, abs=1e-5)
    assert res["chi2_check"] < 1e-9


def test_averaged_nees_bounds_match_bar_shalom_example():
    # Bar-Shalom et al. 2001, p. 235: n = 2, N = 50 runs, two-sided 95% interval [1.5, 2.6].
    lo, hi = averaged_nees_bounds(2, 50)
    assert round(lo, 1) == 1.5 and round(hi, 1) == 2.6


def test_nees_honest_over_and_underconfident(pb):
    n = 1000
    res = pb.run_nees(P_TRUE, [1.0, 0.25, 4.0], n, _seed(2))
    lo, hi = averaged_nees_bounds(2, n)
    honest, over, under = res["cases"]
    assert lo <= honest["avg"] <= hi
    assert over["avg"] > 2.0 * hi       # clearly above
    assert under["avg"] < 0.5 * lo      # clearly below


def test_nees_scales_exactly_as_one_over_c(pb):
    # e^T (c P)^-1 e = (e^T P^-1 e) / c, on the same errors.
    res = pb.run_nees(P_TRUE, [1.0, 0.25, 4.0], 200, _seed(2))
    base = res["cases"][0]["nees"]
    for k in res["cases"]:
        assert np.allclose(k["nees"] * k["c"], base)


def test_nees_averages_and_tail_fractions_match_theory_for_many_trials(pb):
    n = 40000
    res = pb.run_nees(P_TRUE, [1.0, 0.25, 4.0], n, _seed(2, seed=1))
    for k in res["cases"]:
        lo, hi = averaged_nees_bounds(2, n)
        # the c != 1 averages are n/c times the honest one: compare with the scaled honest bounds
        assert k["avg"] * k["c"] == pytest.approx(2.0, abs=(hi - lo) / 2)
        assert k["frac_above"] == pytest.approx(k["predicted_frac_above"], abs=0.01)


# --- command line ---

def test_part_flag_runs_only_the_selected_part(pb, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["probability_basics.py", "--part", "nees", "--n-nees-trials", "50"])
    pb.main()
    out = capsys.readouterr().out
    assert "Part 3" in out and "Part 1" not in out and "Part 2" not in out
