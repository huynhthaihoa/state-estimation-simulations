import numpy as np
import pytest

from state_estimation_simulations.lidar_utils import ExtractionConfig, LidarConfig, corridor_scene
from state_estimation_simulations.use_numpy import lidar_primitives as _lidar_primitives


@pytest.fixture
def lp():
    return _lidar_primitives


_PAIRS = {}


def scan_pair(lp, pillars):
    """Noise-free scan pair (cached: extraction takes about a second per scan)."""
    if pillars not in _PAIRS:
        _PAIRS[pillars] = lp.make_scan_pair(corridor_scene(pillars=pillars), [0, 0],
                                            LidarConfig(n_azimuth=180, range_noise_std=0.0), ExtractionConfig())
    return _PAIRS[pillars]


@pytest.fixture
def pair_pillars(lp):
    return scan_pair(lp, True)


@pytest.fixture
def pair_plain(lp):
    return scan_pair(lp, False)


EPS = 1e-6


def numerical_jacobian(lp, f, T):
    """d f(T Exp(xi)) / d xi at xi = 0, by forward differences."""
    e0 = f(T)
    return np.column_stack([(f(T @ lp.se3_exp(EPS * np.eye(6)[k])) - e0) / EPS for k in range(6)])


def random_pose(lp, rng):
    return lp.se3_exp(rng.normal(size=6))


def test_point_action_jacobian_matches_finite_differences(lp):
    rng = np.random.default_rng(0)
    T, P = random_pose(lp, rng), rng.normal(size=(4, 3))
    J = lp.point_action_jacobian(T, P).reshape(-1, 6)
    assert np.allclose(J, numerical_jacobian(lp, lambda X: lp.transform_points(X, P).ravel(), T), atol=1e-5)


def test_plane_and_cylinder_residual_jacobians_match_finite_differences(lp):
    rng = np.random.default_rng(1)
    T, P = random_pose(lp, rng), rng.normal(size=(6, 3))
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    _, J = lp.plane_point_residuals(T, P, n, 0.3)
    assert np.allclose(J, numerical_jacobian(lp, lambda X: lp.plane_point_residuals(X, P, n, 0.3)[0], T), atol=1e-5)
    a = rng.normal(size=3)
    a /= np.linalg.norm(a)
    c = rng.normal(size=3)
    c -= (a @ c) * a
    _, J = lp.cylinder_point_residuals(T, P, a, c, 0.4)
    assert np.allclose(J, numerical_jacobian(lp, lambda X: lp.cylinder_point_residuals(X, P, a, c, 0.4)[0], T),
                       atol=1e-5)


@pytest.mark.parametrize("method", ["point_to_point", "point_to_plane"])
def test_nearest_neighbor_residual_jacobians_match_finite_differences(lp, pair_pillars, method):
    src, tgt, T_true, _ = pair_pillars
    cfg = lp.RegistrationConfig()
    T = T_true @ lp.se3_exp(np.array([0.02, -0.01, 0.01, 0.002, 0.001, -0.003]))
    f = lp.RESIDUALS[method]
    e, J = f(T, src, tgt, cfg, cfg.max_iters)
    # the pairs must not change under the finite-difference step for this check to be meaningful
    J_num = numerical_jacobian(lp, lambda X: f(X, src, tgt, cfg, cfg.max_iters)[0], T)
    assert J_num.shape == J.shape
    assert np.allclose(J, J_num, atol=1e-4)


def test_primitives_match_one_to_one_at_the_truth(lp, pair_pillars):
    src, tgt, T_true, _ = pair_pillars
    matches = lp.match_primitives(T_true[:3, :3], T_true[:3, 3], src, tgt, lp.RegistrationConfig())
    planes = [(i, j) for k, i, j in matches if k == "plane"]
    assert len(planes) == len(src.planes) == len(tgt.planes) == 4
    R = T_true[:3, :3]
    for i, j in planes:
        assert R @ src.planes[i]["n"] @ tgt.planes[j]["n"] > 1 - 1e-9
    assert any(k == "cylinder" for k, _, _ in matches)


def test_primitive_registration_is_exact_noise_free(lp, pair_pillars):
    src, tgt, T_true, _ = pair_pillars
    T0 = T_true @ lp.se3_exp(np.array([0.3, -0.2, 0.1, 0.05, -0.03, 0.08]))
    T, info = lp.register("primitive", T0, src, tgt, lp.RegistrationConfig())
    et, er = lp.pose_error(T, T_true)
    assert info["converged"] and et < 1e-6 and er < 1e-7


def test_icp_variants_land_near_the_truth_noise_free(lp, pair_pillars):
    """Not exact: the two scans sample different surface points, so nearest
    neighbors are not true correspondences (a few mm for point-to-plane, about
    a centimeter for point-to-point with the coarse-to-fine gate)."""
    src, tgt, T_true, _ = pair_pillars
    for method, tol in (("point_to_plane", 0.01), ("point_to_point", 0.05)):
        T, _ = lp.register(method, T_true, src, tgt, lp.RegistrationConfig())
        assert lp.pose_error(T, T_true)[0] < tol


def test_planes_only_corridor_is_degenerate_along_its_axis(lp, pair_plain, pair_pillars):
    cfg = lp.RegistrationConfig()
    ratios = []
    for (src, tgt, T_true, T_tgt) in (pair_plain, pair_pillars):
        _, J = lp.residuals_primitive(T_true, src, tgt, cfg)
        evals, evecs = np.linalg.eigh(J.T @ J)
        ratios.append(evals[0] / evals[-1])
        axis = T_tgt[:3, :3].T @ np.array([1.0, 0.0, 0.0])
        weakest_t = T_true[:3, :3] @ evecs[:3, 0]
        assert abs(weakest_t @ axis) > 0.99 * np.linalg.norm(evecs[:3, 0])
    assert ratios[0] < 1e-10           # planes only: no information along the axis
    assert ratios[1] > 1e-4            # pillars restore it


def test_coarse_to_fine_gate_schedule(lp):
    cfg = lp.RegistrationConfig()
    gates = [lp.pair_gate(cfg, k) for k in range(20)]
    assert gates[0] == cfg.max_pair_dist and gates[-1] == cfg.min_pair_dist
    assert all(g1 <= g0 for g0, g1 in zip(gates, gates[1:]))


def test_side_check_rejects_an_arc_shifted_onto_the_far_side(lp):
    """The failure mode it guards against: the scanned (front) arc translated
    past the axis lands near the far side of the target pillar, where the bare
    distance residual is small; translation does not turn the normals, so
    they now point into the pillar and the side check drops every point."""
    a, c, r = np.array([0.0, 0.0, 1.0]), np.array([3.0, 0.0, 0.0]), 0.25
    q = np.linspace(np.pi - 1.0, np.pi + 1.0, 30)                  # arc facing the sensor at the origin
    radial = np.stack([np.cos(q), np.sin(q), np.zeros_like(q)], axis=1)
    P = c + r * radial
    cyl = {"a": a, "c": c, "r": r, "inliers": np.arange(len(P))}
    src = lp.Scan(P, radial, None, [], [cyl])
    tgt = lp.Scan(P, radial, None, [], [cyl])
    cfg = lp.RegistrationConfig()
    e, _ = lp.residuals_primitive(np.eye(4), src, tgt, cfg)
    assert len(e) == len(P) and np.allclose(e, 0.0)
    T_shift = np.eye(4)
    T_shift[0, 3] = 2 * r                                         # front apex -> far-side apex
    assert abs(lp.cylinder_point_residuals(T_shift, P[15:16], a, c, r)[0][0]) < 1e-3
    e, _ = lp.residuals_primitive(T_shift, src, tgt, cfg)
    assert len(e) == 0


def random_plane(rng):
    n = rng.normal(size=3)
    return n / np.linalg.norm(n), rng.uniform(0.5, 3.0)


def test_plane_factor_is_zero_at_truth_and_its_jacobians_match_finite_differences(lp):
    rng = np.random.default_rng(5)
    T = random_pose(lp, rng)
    n_w, d_w = random_plane(rng)
    n_s, d_s = T[:3, :3].T @ n_w, d_w + n_w @ T[:3, 3]           # exact measurement
    e, Jp, Jl = lp.plane_factor(T, n_w, d_w, n_s, d_s)
    assert np.allclose(e, 0.0)
    T = T @ lp.se3_exp(0.05 * rng.normal(size=6))                  # evaluate away from zero residual
    e, Jp, Jl = lp.plane_factor(T, n_w, d_w, n_s, d_s)
    assert np.allclose(Jp, numerical_jacobian(lp, lambda X: lp.plane_factor(X, n_w, d_w, n_s, d_s)[0], T),
                       atol=1e-5)
    f = lambda x: lp.plane_factor(T, lp.s2_exp(n_w, x[:2]), d_w + x[2], n_s, d_s)[0]
    Jl_num = np.column_stack([(f(EPS * np.eye(3)[k]) - e) / EPS for k in range(3)])
    assert np.allclose(Jl, Jl_num, atol=1e-5)


def test_between_factor_jacobians_match_finite_differences(lp):
    rng = np.random.default_rng(6)
    Xi, Xj, Z = (random_pose(lp, rng) for _ in range(3))
    _, Ji, Jj = lp.between_factor(Xi, Xj, Z)
    assert np.allclose(Ji, numerical_jacobian(lp, lambda X: lp.between_factor(X, Xj, Z)[0], Xi), atol=1e-4)
    assert np.allclose(Jj, numerical_jacobian(lp, lambda X: lp.between_factor(Xi, X, Z)[0], Xj), atol=1e-4)


def test_plane_fit_information_matches_the_scatter_of_noisy_fits(lp):
    """Monte Carlo: the covariance of (normal tangent, offset) over many noisy
    fits of the same patch matches the inverse of the predicted information."""
    from state_estimation_simulations.lidar_utils import fit_plane, s2_log
    rng = np.random.default_rng(7)
    n, d, noise = np.array([0.0, 0.0, 1.0]), 1.2, 0.01
    xy = rng.uniform(-2, 2, (300, 2))
    P0 = np.column_stack([xy, np.full(300, -d)])
    errs = []
    for _ in range(2000):
        n_e, d_e = fit_plane(P0 + noise * rng.normal(size=300)[:, None] * n)
        errs.append(np.concatenate([s2_log(n, n_e), [d_e - d]]))
    cov_mc = np.cov(np.array(errs).T)
    cov_pred = np.linalg.inv(lp.plane_fit_information(P0, n, noise))
    assert np.allclose(np.diag(cov_mc), np.diag(cov_pred), rtol=0.15)


def test_noise_free_plane_graph_recovers_the_truth(lp):
    """Perfect odometry and plane measurements, initialized off the truth:
    the graph returns exactly to the truth (pose 0 is the anchor)."""
    rng = np.random.default_rng(8)
    truth = [np.eye(4)]
    for _ in range(4):
        truth.append(truth[-1] @ lp.se3_exp(np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.2]) + 0.05 * rng.normal(size=6)))
    planes_w = [(np.array([0.0, 0.0, 1.0]), 1.2), (np.array([0.0, 1.0, 0.0]), 3.0), (np.array([1.0, 0.0, 0.0]), 9.0)]
    P_patch = np.column_stack([rng.uniform(-2, 2, 50), rng.uniform(-2, 2, 50), np.zeros(50)])
    scans = []
    for T in truth:
        planes = []
        for n_w, d_w in planes_w:
            n_s, d_s = T[:3, :3].T @ n_w, d_w + n_w @ T[:3, 3]
            B = lp.s2_basis(n_s)
            pts = -d_s * n_s + P_patch[:, :2] @ B.T                 # points on the measured plane
            planes.append({"n": n_s, "d": d_s, "inliers": np.arange(len(pts)) + 50 * len(planes)})
        all_pts = np.vstack([-pl["d"] * pl["n"] + P_patch[:, :2] @ lp.s2_basis(pl["n"]).T for pl in planes])
        scans.append(lp.Scan(all_pts, None, None, planes, []))
    Z = [lp.se3_inv(truth[k]) @ truth[k + 1] for k in range(4)]
    Omega = [np.eye(6) * 1e4] * 4
    init = [truth[0]] + [T @ lp.se3_exp(0.05 * rng.normal(size=6)) for T in truth[1:]]
    landmarks, observations = lp.associate_planes(scans, [T[:3, :3] for T in init], [T[:3, 3] for T in init])
    assert len(landmarks) == 3 and len(observations) == 15
    est, lms, _ = lp.optimize_plane_graph(init, Z, Omega, scans, landmarks, observations, noise=0.01)
    for T_e, T_t in zip(est, truth):
        et, er = lp.pose_error(T_e, T_t)
        assert et < 1e-7 and er < 1e-8
    for (n_e, d_e), (n_t, d_t) in zip(lms, planes_w):
        assert np.allclose(n_e, n_t, atol=1e-8) and np.isclose(d_e, d_t, atol=1e-7)


def test_l_corridor_trajectory_scans_see_every_plane_once(lp):
    """Scene sanity: at the corner the sensor sees floor, ceiling and all four walls
    (the two floor rectangles are one plane)."""
    from state_estimation_simulations.lidar_utils import l_corridor_scene, l_corridor_trajectory, simulate_scan, extract_primitives
    scene, traj = l_corridor_scene(), l_corridor_trajectory()
    assert len(traj) == 40
    R, t = traj[19]
    pts, _ = simulate_scan(scene, R, t, LidarConfig(n_azimuth=180, range_noise_std=0.0), np.random.default_rng(0))
    planes, _ = extract_primitives(pts, ExtractionConfig(), np.random.default_rng(0))
    assert len(planes) == 6
