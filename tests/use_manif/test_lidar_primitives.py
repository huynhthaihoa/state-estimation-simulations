import numpy as np
import pytest

manif = pytest.importorskip("manifpy")

from state_estimation_simulations.lidar_utils import ExtractionConfig, LidarConfig, RegistrationConfig, corridor_scene  # noqa: E402
from state_estimation_simulations.use_manif import lidar_primitives as _lidar_primitives  # noqa: E402
from state_estimation_simulations.use_numpy import lidar_primitives as _lidar_primitives_np  # noqa: E402


@pytest.fixture
def lm():
    return _lidar_primitives


@pytest.fixture
def ln():
    return _lidar_primitives_np


EPS = 1e-6


def random_pose(rng):
    return manif.SE3Tangent(rng.normal(size=6)).exp()


def numerical_jacobian(f, T):
    """d f(T.rplus(xi)) / d xi at xi = 0, by forward differences."""
    e0 = f(T)
    return np.column_stack([(f(T.rplus(manif.SE3Tangent(EPS * np.eye(6)[k]))) - e0) / EPS for k in range(6)])


_PAIRS = {}


def scan_pair(lm, seed):
    """Noise-free scan pair in the corridor with pillars, built by the manif script (cached)."""
    if seed not in _PAIRS:
        _PAIRS[seed] = lm.make_scan_pair(corridor_scene(pillars=True), [seed, 0],
                                         LidarConfig(n_azimuth=180, range_noise_std=0.005),
                                         ExtractionConfig(inlier_dist=0.05))
    return _PAIRS[seed]


def test_vectorized_point_action_jacobian_matches_manif_act(lm):
    rng = np.random.default_rng(0)
    T, P = random_pose(rng), rng.normal(size=(5, 3))
    J = lm.point_action_jacobian(T, P)
    for k, p in enumerate(P):
        J_act = np.zeros((3, 6))
        T.act(p, J_act)
        assert np.allclose(J[k], J_act, atol=1e-12)


def test_plane_factor_jacobians_match_finite_differences_and_numpy(lm, ln):
    rng = np.random.default_rng(1)
    T = random_pose(rng)
    n_w = rng.normal(size=3)
    n_w /= np.linalg.norm(n_w)
    d_w = 1.3
    n_s = T.rotation().T @ n_w + 0.01 * rng.normal(size=3)
    n_s /= np.linalg.norm(n_s)
    d_s = d_w + n_w @ T.translation() + 0.02
    e, Jp, Jl = lm.plane_factor(T, n_w, d_w, n_s, d_s)
    assert np.allclose(Jp, numerical_jacobian(lambda X: lm.plane_factor(X, n_w, d_w, n_s, d_s)[0], T), atol=1e-5)
    e_np, Jp_np, Jl_np = ln.plane_factor(T.transform(), n_w, d_w, n_s, d_s)
    assert np.allclose(e, e_np) and np.allclose(Jp, Jp_np, atol=1e-12) and np.allclose(Jl, Jl_np)


def test_between_factor_jacobians_match_finite_differences_and_numpy(lm, ln):
    rng = np.random.default_rng(2)
    Xi, Xj, Z = (random_pose(rng) for _ in range(3))
    e, Ji, Jj = lm.between_factor(Xi, Xj, Z)
    assert np.allclose(Ji, numerical_jacobian(lambda X: lm.between_factor(X, Xj, Z)[0], Xi), atol=1e-4)
    assert np.allclose(Jj, numerical_jacobian(lambda X: lm.between_factor(Xi, X, Z)[0], Xj), atol=1e-4)
    e_np, Ji_np, Jj_np = ln.between_factor(Xi.transform(), Xj.transform(), Z.transform())
    assert np.allclose(e, e_np, atol=1e-10) and np.allclose(Ji, Ji_np, atol=1e-8) and np.allclose(Jj, Jj_np, atol=1e-8)


@pytest.mark.parametrize("method", ["point_to_plane", "primitive"])
def test_registration_matches_the_numpy_backend(lm, ln, method):
    """Same scans, same start: both backends land on the same pose."""
    src, tgt, T_true, _ = scan_pair(lm, 0)
    xi0 = np.array([0.2, -0.1, 0.05, 0.03, -0.02, 0.05])
    T_m, info_m = lm.register(method, T_true.rplus(manif.SE3Tangent(xi0)), src, tgt, RegistrationConfig())
    T_n, info_n = ln.register(method, T_true.transform() @ ln.se3_exp(xi0), src, tgt, RegistrationConfig())
    assert info_m["iters"] == info_n["iters"]
    assert np.allclose(T_m.transform(), T_n, atol=1e-9)
    assert lm.pose_error(T_m, T_true)[0] < 0.01


def test_noise_free_extraction_scene_pair_is_identical_across_backends(lm, ln):
    """make_scan_pair consumes the RNG identically in both backends."""
    src_m, tgt_m, T_m, Tt_m = scan_pair(lm, 0)
    src_n, tgt_n, T_n, Tt_n = ln.make_scan_pair(corridor_scene(pillars=True), [0, 0],
                                                LidarConfig(n_azimuth=180, range_noise_std=0.005),
                                                ExtractionConfig(inlier_dist=0.05))
    assert np.allclose(T_m.transform(), T_n) and np.allclose(Tt_m.transform(), Tt_n)
    assert np.allclose(src_m.points, src_n.points) and np.allclose(tgt_m.points, tgt_n.points)


def test_plane_graph_matches_the_numpy_backend(lm, ln):
    """Odometry + plane landmarks over a short noise-free-ish synthetic run:
    both backends optimize to the same poses and landmarks."""
    rng = np.random.default_rng(3)
    truth = [manif.SE3.Identity()]
    for _ in range(4):
        truth.append(truth[-1].rplus(manif.SE3Tangent(np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.2])
                                                      + 0.05 * rng.normal(size=6))))
    planes_w = [(np.array([0.0, 0.0, 1.0]), 1.2), (np.array([0.0, 1.0, 0.0]), 3.0), (np.array([1.0, 0.0, 0.0]), 9.0)]
    patch = rng.uniform(-2, 2, (50, 2))
    scans = []
    for T in truth:
        planes, pts = [], []
        for n_w, d_w in planes_w:
            n_s, d_s = T.rotation().T @ n_w, d_w + n_w @ T.translation() + 0.003 * rng.normal()
            planes.append({"n": n_s, "d": d_s, "inliers": np.arange(50) + 50 * len(planes)})
            pts.append(-d_s * n_s + patch @ lm.s2_basis(n_s).T)
        scans.append(lm.Scan(np.vstack(pts), None, None, planes, []))
    Z = [truth[k].between(truth[k + 1]).rplus(manif.SE3Tangent(0.01 * rng.normal(size=6))) for k in range(4)]
    Omega = [np.eye(6) * 1e4] * 4
    init = [truth[0]]
    for Zk in Z:
        init.append(init[-1].compose(Zk))
    landmarks, observations = lm.associate_planes(scans, [T.rotation() for T in init], [T.translation() for T in init])
    est_m, lms_m, _ = lm.optimize_plane_graph(init, Z, Omega, scans, landmarks, observations, noise=0.01)
    est_n, lms_n, _ = ln.optimize_plane_graph([T.transform() for T in init], [T.transform() for T in Z], Omega,
                                              scans, landmarks, observations, noise=0.01)
    for T_m, T_n in zip(est_m, est_n):
        assert np.allclose(T_m.transform(), T_n, atol=1e-8)
    for (n_m, d_m), (n_n, d_n) in zip(lms_m, lms_n):
        assert np.allclose(n_m, n_n, atol=1e-8) and np.isclose(d_m, d_n, atol=1e-8)
