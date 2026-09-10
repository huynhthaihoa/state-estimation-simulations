import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture
def lie_utils(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "lie_utils")


# A few representative omega/xi magnitudes: exactly zero, deep inside the
# small-angle branch (theta << 1e-6), just past the branch threshold, and a
# generic moderate rotation -- so every branch in lie_utils.py gets exercised.
OMEGAS = [
    np.array([0.0, 0.0, 0.0]),
    np.array([1e-8, -2e-8, 5e-9]),
    np.array([2e-6, -1e-6, 1e-6]),
    np.array([0.3, -0.5, 0.7]),
    np.array([1.2, 0.4, -0.9]),
]

XIS = [np.concatenate([t, o]) for t, o in zip(
    [np.array([0.0, 0.0, 0.0]), np.array([0.1, -0.2, 0.05]),
     np.array([1.0, -2.0, 0.5]), np.array([0.3, 0.1, -0.4]), np.array([2.0, -1.0, 0.3])],
    OMEGAS,
)]


def test_skew_is_antisymmetric_and_matches_cross_product(lie_utils):
    rng = np.random.default_rng(0)
    v, w = rng.normal(size=3), rng.normal(size=3)
    S = lie_utils.skew(v)
    assert np.allclose(S, -S.T, atol=1e-12)
    assert np.allclose(S @ w, np.cross(v, w), atol=1e-12)


@pytest.mark.parametrize("omega", OMEGAS)
def test_so3_exp_is_a_valid_rotation(lie_utils, omega):
    R = lie_utils.so3_exp(omega)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)


@pytest.mark.parametrize("omega", [o for o in OMEGAS if np.linalg.norm(o) > 1e-6])
def test_so3_exp_matches_scipy_rotvec(lie_utils, omega):
    R = lie_utils.so3_exp(omega)
    expected = Rotation.from_rotvec(omega).as_matrix()
    assert np.allclose(R, expected, atol=1e-8)


@pytest.mark.parametrize("omega", OMEGAS)
def test_so3_jacobian_inverse_pair(lie_utils, omega):
    Jr = lie_utils.so3_right_jacobian(omega)
    Jr_inv = lie_utils.compute_so3_inv_right_jacobian(omega)
    assert np.allclose(Jr @ Jr_inv, np.eye(3), atol=1e-6)
    assert np.allclose(Jr_inv @ Jr, np.eye(3), atol=1e-6)


def test_so3_right_jacobian_matches_finite_difference(lie_utils):
    # Defining property: so3_exp(omega + d) ~= so3_exp(omega) @ so3_exp(Jr(omega) @ d)
    # for small d (first-order right-perturbation model).
    omega = np.array([0.4, -0.2, 0.6])
    Jr = lie_utils.so3_right_jacobian(omega)
    rng = np.random.default_rng(1)
    d = rng.normal(size=3) * 1e-5
    lhs = lie_utils.so3_exp(omega + d)
    rhs = lie_utils.so3_exp(omega) @ lie_utils.so3_exp(Jr @ d)
    assert np.allclose(lhs, rhs, atol=1e-8)


def test_so3_inv_right_jacobian_matches_defining_identity(lie_utils):
    # Defining property: Log(Exp(a) Exp(eps)) - a ~= Jr_inv(a) @ eps for small
    # eps. Stronger than test_so3_jacobian_inverse_pair above, which only
    # checks Jr @ Jr_inv == I -- true by construction whenever one is
    # literally computed as the matrix inverse of the other, even if both
    # are simultaneously wrong in a self-consistent way. This test checks
    # compute_so3_inv_right_jacobian against its actual mathematical
    # definition instead.
    omega = np.array([0.4, -0.2, 0.6])
    Jr_inv = lie_utils.compute_so3_inv_right_jacobian(omega)
    rng = np.random.default_rng(5)
    eps = rng.normal(size=3) * 1e-5

    def rotation_log(R):
        T = np.eye(4)
        T[0:3, 0:3] = R
        return lie_utils.se3_log(T)[3:6]  # no standalone so3_log in this codebase

    lhs = rotation_log(lie_utils.so3_exp(omega) @ lie_utils.so3_exp(eps)) - omega
    rhs = Jr_inv @ eps
    assert np.allclose(lhs, rhs, atol=1e-8)


def test_rotation_geodesic_error_properties(lie_utils):
    rng = np.random.default_rng(2)
    R = lie_utils.so3_exp(rng.normal(size=3) * 0.5)
    assert np.isclose(lie_utils.rotation_geodesic_error(R, R), 0.0, atol=1e-10)

    small_omega = np.array([0.001, -0.0005, 0.0002])
    R2 = lie_utils.so3_exp(small_omega) @ R
    assert np.isclose(lie_utils.rotation_geodesic_error(R, R2), np.linalg.norm(small_omega), atol=1e-6)

    R_b = lie_utils.so3_exp(rng.normal(size=3) * 0.3)
    assert np.isclose(
        lie_utils.rotation_geodesic_error(R, R_b),
        lie_utils.rotation_geodesic_error(R_b, R),
        atol=1e-10,
    )


@pytest.mark.parametrize("xi", XIS)
def test_se3_exp_top_left_block_is_so3_exp(lie_utils, xi):
    T = lie_utils.se3_exp(xi)
    assert np.allclose(T[0:3, 0:3], lie_utils.so3_exp(xi[3:6]), atol=1e-10)
    assert np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-12)


@pytest.mark.parametrize("xi", XIS)
def test_se3_log_se3_exp_roundtrip(lie_utils, xi):
    T = lie_utils.se3_exp(xi)
    xi_recovered = lie_utils.se3_log(T)
    assert np.allclose(xi_recovered, xi, atol=1e-6)


@pytest.mark.parametrize("xi", XIS)
def test_se3_exp_se3_log_roundtrip(lie_utils, xi):
    T = lie_utils.se3_exp(xi)  # guaranteed-valid SE(3), rather than a hand-built matrix
    T_recovered = lie_utils.se3_exp(lie_utils.se3_log(T))
    assert np.allclose(T_recovered, T, atol=1e-6)


@pytest.mark.parametrize("xi", XIS)
def test_se3_inv(lie_utils, xi):
    T = lie_utils.se3_exp(xi)
    T_inv = lie_utils.se3_inv(T)
    assert np.allclose(T @ T_inv, np.eye(4), atol=1e-8)
    assert np.allclose(T_inv @ T, np.eye(4), atol=1e-8)
    assert np.allclose(lie_utils.se3_inv(T_inv), T, atol=1e-8)


def test_se3_adjoint_conjugation_identity(lie_utils):
    # Defining property of the adjoint: T @ exp(xi) @ T^-1 == exp(Adj(T) @ xi)
    rng = np.random.default_rng(3)
    T = lie_utils.se3_exp(np.concatenate([rng.normal(size=3), rng.normal(size=3) * 0.5]))
    xi = np.concatenate([rng.normal(size=3) * 0.1, rng.normal(size=3) * 0.1])

    lhs = T @ lie_utils.se3_exp(xi) @ lie_utils.se3_inv(T)
    rhs = lie_utils.se3_exp(lie_utils.se3_adjoint(T) @ xi)
    assert np.allclose(lhs, rhs, atol=1e-6)


def test_se3_adjoint_is_a_group_homomorphism(lie_utils):
    rng = np.random.default_rng(4)
    T1 = lie_utils.se3_exp(np.concatenate([rng.normal(size=3), rng.normal(size=3) * 0.3]))
    T2 = lie_utils.se3_exp(np.concatenate([rng.normal(size=3), rng.normal(size=3) * 0.3]))
    assert np.allclose(lie_utils.se3_adjoint(T1 @ T2), lie_utils.se3_adjoint(T1) @ lie_utils.se3_adjoint(T2), atol=1e-8)
    assert np.allclose(lie_utils.se3_adjoint(lie_utils.se3_inv(T1)), np.linalg.inv(lie_utils.se3_adjoint(T1)), atol=1e-6)


@pytest.mark.parametrize("xi", XIS)
def test_se3_jacobian_inverse_pair(lie_utils, xi):
    Jr = lie_utils.se3_right_jacobian(xi)
    Jr_inv = lie_utils.compute_se3_inv_right_jacobian(xi)
    assert np.allclose(Jr @ Jr_inv, np.eye(6), atol=1e-5)
    assert np.allclose(Jr_inv @ Jr, np.eye(6), atol=1e-5)


def test_se3_inv_right_jacobian_matches_defining_identity(lie_utils):
    # Defining property: Log(Exp(xi) Exp(eps)) - xi ~= Jr_inv(xi) @ eps for
    # small eps. This is the test that actually catches Jacobian formula
    # bugs -- the inverse-pair check above only proves internal self-
    # consistency (in this codebase Jr and Jr_inv are literally each other's
    # matrix inverse by construction), which holds even if both are
    # simultaneously wrong in the same self-consistent way. A prior version
    # of compute_se3_inv_right_jacobian's translation/rotation coupling (Q)
    # term passed the inverse-pair check while failing this one -- confirmed
    # via a convergence-order sweep (error should shrink by ~4x each time
    # `scale` halves for a correct first-order Jacobian; the old formula only
    # shrank by ~2x, meaning it was zeroth-order accurate, not first-order).
    xi = np.array([1.0, -2.0, 0.5, 0.3, -0.5, 0.7])
    Jr_inv = lie_utils.compute_se3_inv_right_jacobian(xi)
    rng = np.random.default_rng(6)
    eps = rng.normal(size=6) * 1e-5

    lhs = lie_utils.se3_log(lie_utils.se3_exp(xi) @ lie_utils.se3_exp(eps)) - xi
    rhs = Jr_inv @ eps
    assert np.allclose(lhs, rhs, atol=1e-7)


def test_compute_se3_inv_right_jacobian_near_zero_theta_has_no_nan(lie_utils):
    # theta just below the small-angle threshold triggers the NaN/inf guard
    # (coeff_q1 fallback to -1/12) -- confirm it's actually finite there and
    # still satisfies the inverse-pair identity, not just "doesn't crash".
    xi = np.array([0.5, -0.3, 0.2, 1e-9, -2e-9, 5e-10])
    J_inv = lie_utils.compute_se3_inv_right_jacobian(xi)
    assert np.all(np.isfinite(J_inv))
    Jr = lie_utils.se3_right_jacobian(xi)
    assert np.allclose(Jr @ J_inv, np.eye(6), atol=1e-4)
