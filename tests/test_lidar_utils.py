import numpy as np
import pytest

from state_estimation_simulations.lidar_utils import (
    Cylinder,
    ExtractionConfig,
    LidarConfig,
    Rect,
    Scene,
    corridor_scene,
    cylinder_residuals,
    cylinder_to_frame,
    estimate_normals,
    extract_primitives,
    plane_to_frame,
    random_sensor_pose,
    raycast,
    refine_cylinder,
    refine_cylinder_range,
    cylinder_range_residuals,
    s2_basis,
    s2_exp,
    s2_log,
    score_extraction,
    simulate_scan,
)


def unit(v):
    return v / np.linalg.norm(v)


def test_s2_basis_is_orthonormal_tangent():
    rng = np.random.default_rng(0)
    for n in [np.array([1.0, 0, 0]), np.array([0, 0, -1.0])] + [unit(rng.normal(size=3)) for _ in range(5)]:
        B = s2_basis(n)
        assert np.allclose(B.T @ B, np.eye(2))
        assert np.allclose(B.T @ n, 0.0)


def test_s2_exp_log_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = unit(rng.normal(size=3))
        delta = rng.uniform(-1.0, 1.0, 2)
        m = s2_exp(n, delta)
        assert np.isclose(np.linalg.norm(m), 1.0)
        assert np.isclose(np.arccos(np.clip(n @ m, -1, 1)), np.linalg.norm(delta))
        assert np.allclose(s2_log(n, m), delta, atol=1e-10)


def test_raycast_matches_analytic_distances():
    scene = Scene([Rect(np.array([5.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), 2.0, 2.0)],
                  [Cylinder(np.array([0, 3.0, -1.0]), np.array([0, 0, 1.0]), 0.5, 2.0)])
    dirs = np.array([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0], unit(np.array([1.0, 0.0, 1.0]))])
    s, label = raycast(scene, np.zeros(3), dirs)
    assert np.isclose(s[0], 5.0) and label[0] == 0
    assert np.isclose(s[1], 2.5) and label[1] == 1
    assert np.isinf(s[2]) and label[2] == -1
    # the 45-degree ray meets the rectangle's plane at height 5, outside its 2 m half extent
    assert np.isinf(s[3]) and label[3] == -1


def test_noise_free_scan_points_lie_on_their_primitive():
    scene = corridor_scene(pillars=True)
    R, t = random_sensor_pose(np.random.default_rng(2))
    pts, lab = simulate_scan(scene, R, t, LidarConfig(range_noise_std=0.0), np.random.default_rng(0))
    world = pts @ R.T + t
    for k, rect in enumerate(scene.rects):
        n, d = rect.plane()
        assert np.max(np.abs(world[lab == k] @ n + d), initial=0.0) < 1e-9
    for j, cyl in enumerate(scene.cylinders):
        sel = lab == len(scene.rects) + j
        assert np.max(np.abs(cylinder_residuals(world[sel], *cyl.params())), initial=0.0) < 1e-9


def test_frame_transforms_agree_with_transforming_points():
    rng = np.random.default_rng(3)
    R, t = random_sensor_pose(rng)
    n, d = unit(rng.normal(size=3)), 1.7
    on_plane = [p - (n @ p + d) * n for p in rng.normal(size=(10, 3))]
    n_s, d_s = plane_to_frame(n, d, R, t)
    assert np.allclose([n_s @ (R.T @ (p - t)) + d_s for p in on_plane], 0.0)
    a, c, r = unit(rng.normal(size=3)), np.zeros(3), 0.4
    c = rng.normal(size=3)
    c -= (a @ c) * a
    B = s2_basis(a)
    on_cyl = np.array([c + h * a + r * B @ [np.cos(q), np.sin(q)] for h, q in rng.uniform(-2, 2, (10, 2))])
    a_s, c_s, r_s = cylinder_to_frame(a, c, r, R, t)
    assert np.isclose(a_s @ c_s, 0.0)
    assert np.allclose(cylinder_residuals((on_cyl - t) @ R, a_s, c_s, r_s), 0.0)


def test_normals_point_toward_sensor():
    scene = corridor_scene()
    pts, lab = simulate_scan(scene, np.eye(3), np.array([0.0, 0.0, 1.2]), LidarConfig(range_noise_std=0.0),
                             np.random.default_rng(0))
    normals, curvature = estimate_normals(pts)
    for k, rect in enumerate(scene.rects):
        sel = lab == k
        assert np.median(normals[sel] @ rect.n) > 0.999   # identity pose: world normal = sensor normal
    assert np.median(curvature) < 1e-6


def test_refine_cylinder_recovers_truth_from_perturbed_start():
    rng = np.random.default_rng(4)
    a = unit(np.array([0.1, -0.2, 1.0]))
    c = np.array([1.0, 2.0, 0.0])
    c -= (a @ c) * a
    r = 0.3
    B = s2_basis(a)
    pts = np.array([c + h * a + r * B @ [np.cos(q), np.sin(q)] for h, q in zip(rng.uniform(-1, 1, 200),
                                                                               rng.uniform(-1.5, 1.5, 200))])
    a0 = s2_exp(a, [0.05, -0.03])
    c0 = c + 0.05 * rng.normal(size=3)
    a_e, c_e, r_e = refine_cylinder(pts, a0, c0 - (a0 @ c0) * a0, r + 0.04, n_iters=30)
    assert abs(abs(a_e @ a) - 1.0) < 1e-12
    assert np.isclose(r_e, r, atol=1e-9)
    w = c - c_e
    assert np.linalg.norm(w - (w @ a_e) * a_e) < 1e-9


@pytest.mark.parametrize("pose_seed", [0, 1])
def test_noise_free_extraction_finds_every_well_visible_primitive(pose_seed):
    scene = corridor_scene(pillars=True)
    R, t = random_sensor_pose(np.random.default_rng([pose_seed, 7]))
    pts, lab = simulate_scan(scene, R, t, LidarConfig(range_noise_std=0.0), np.random.default_rng(0))
    planes, cyls = extract_primitives(pts, ExtractionConfig(), np.random.default_rng(0))
    sc = score_extraction(scene, R, t, lab, planes, cyls, min_points=150)
    assert sc["false_positives"] == 0
    assert sc["planes_found"] == sc["planes_visible"] == 4
    assert sc["cylinders_found"] == sc["cylinders_visible"] > 0
    assert max(ang for ang, _ in sc["plane"]) < 1e-6
    assert max(off for _, off in sc["plane"]) < 1e-6
    assert max(abs(err) for e in sc["cylinder"] for err in e) < 1e-3


def pillar_scan(sigma, pose_seed, noise_seed):
    """Noise-free hits on the pillars of one scan + range noise along each ray,
    with every pillar's true (sensor-frame) parameters."""
    scene = corridor_scene(pillars=True)
    R, t = random_sensor_pose(np.random.default_rng([0, pose_seed]))
    pts, lab = simulate_scan(scene, R, t, LidarConfig(range_noise_std=0.0), np.random.default_rng(0))
    rng = np.random.default_rng(noise_seed)
    out = []
    for j, cyl in enumerate(scene.cylinders):
        P = pts[lab == len(scene.rects) + j]
        if len(P) >= 150:
            P = P + sigma * rng.standard_normal(len(P))[:, None] * P / np.linalg.norm(P, axis=1, keepdims=True)
            out.append((P, cylinder_to_frame(*cyl.params(), R, t)))
    return out


def test_range_residuals_vanish_at_truth_and_flag_misses():
    (P, (a, c, r)), *_ = pillar_scan(0.0, 0, 0)
    rays, ranges = P / np.linalg.norm(P, axis=1, keepdims=True), np.linalg.norm(P, axis=1)
    x = np.concatenate([np.zeros(2), s2_basis(a).T @ c, [r]])
    assert np.max(np.abs(cylinder_range_residuals(x, a, rays, ranges))) < 1e-9
    x_small = x.copy()
    x_small[4] = 0.5 * r                            # thinner cylinder: the edge rays now miss it
    res = cylinder_range_residuals(x_small, a, rays, ranges)
    assert np.any(res[len(P):] > 0)


def test_refine_cylinder_range_recovers_truth_noise_free():
    (P, (a, c, r)), *_ = pillar_scan(0.0, 1, 0)
    a0 = s2_exp(a, [0.02, -0.01])
    c0 = c + 0.02 * s2_basis(a) @ [1.0, -1.0]
    a_e, c_e, r_e = refine_cylinder_range(P, a0, c0 - (a0 @ c0) * a0, r - 0.02)
    assert abs(abs(a_e @ a) - 1.0) < 1e-10
    assert np.isclose(r_e, r, atol=1e-6)


def test_range_fit_reduces_the_orthogonal_fit_radius_bias():
    """Range noise along the ray biases the orthogonal-distance radius low;
    the range-space fit, started from it, removes most of that bias."""
    odr, rng_fit = [], []
    for k in range(8):
        for P, (a, c, r) in pillar_scan(0.04, k, k + 100):
            a1, c1, r1 = refine_cylinder(P, a, c, r, n_iters=30)
            odr.append(r1 - r)
            rng_fit.append(refine_cylinder_range(P, a1, c1, r1)[2] - r)
    assert np.mean(odr) < -0.01                     # about -17 mm at 4 cm noise
    assert abs(np.mean(rng_fit)) < 0.5 * abs(np.mean(odr))
