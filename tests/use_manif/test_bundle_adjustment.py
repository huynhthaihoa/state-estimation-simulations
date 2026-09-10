import numpy as np
import pytest

manif = pytest.importorskip("manifpy")


@pytest.fixture
def bundle_adjustment(import_module, use_manif_dir):
    return import_module(use_manif_dir, "bundle_adjustment")


K = (800.0, 800.0, 320.0, 240.0)


# ---------------------------------------------------------------------------
# generate_ground_truth_scene
# ---------------------------------------------------------------------------

def test_generate_ground_truth_scene_shapes_and_validity(bundle_adjustment):
    ba = bundle_adjustment
    rng = np.random.default_rng(0)
    T_true, P_true = ba.generate_ground_truth_scene(
        n_cameras=6, n_landmarks=30, camera_radius=5.0, arc_span_deg=180.0,
        landmark_spread=2.0, rng=rng)

    assert len(T_true) == 6
    assert P_true.shape == (30, 3)
    # Landmarks sampled uniformly in [-spread, spread]^3.
    assert np.all(np.abs(P_true) <= 2.0)

    centroid = P_true.mean(axis=0)
    for T in T_true:
        assert isinstance(T, manif.SE3)
        R = T.rotation()
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-8)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)
        # Camera sits at camera_radius from the landmark centroid (in-plane).
        assert np.isclose(np.linalg.norm(T.translation() - centroid), 5.0, atol=1e-8)


# ---------------------------------------------------------------------------
# camera_project + Jacobians (finite-difference check)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_camera_project_jacobians_match_finite_difference(bundle_adjustment, seed):
    ba = bundle_adjustment
    rng = np.random.default_rng(seed)
    # Build a random but valid camera pose (not the identity, so the test
    # actually exercises rotation coupling): a right-perturbation of the
    # identity by a random twist is guaranteed to be a valid manif.SE3.
    twist = rng.normal(0.0, 0.3, 6)
    T = manif.SE3.Identity() + manif.SE3Tangent(twist)

    # A world point placed in front of the camera at moderate depth.
    P = T.act(np.array([0.3, -0.2, 4.0]))

    pixel0, J_pose, J_point = ba.camera_project(T, P, K, with_jacobians=True)
    assert J_pose.shape == (2, 6)
    assert J_point.shape == (2, 3)

    epsilon = 1e-6

    # J_pose: perturb the pose via a right-perturbation T + SE3Tangent(eps * e_k),
    # matching the module's own docstring ("Jacobian of pixel wrt a
    # right-perturbation of T") and manif's rplus convention.
    for k in range(6):
        d = np.zeros(6)
        d[k] = epsilon
        T_plus = T + manif.SE3Tangent(d)
        T_minus = T + manif.SE3Tangent(-d)
        pixel_plus, _, _ = ba.camera_project(T_plus, P, K)
        pixel_minus, _, _ = ba.camera_project(T_minus, P, K)
        numeric = (pixel_plus - pixel_minus) / (2 * epsilon)
        assert np.allclose(numeric, J_pose[:, k], atol=1e-4), f"J_pose column {k} mismatch"

    # J_point: perturb the 3D point directly.
    for k in range(3):
        d = np.zeros(3)
        d[k] = epsilon
        pixel_plus, _, _ = ba.camera_project(T, P + d, K)
        pixel_minus, _, _ = ba.camera_project(T, P - d, K)
        numeric = (pixel_plus - pixel_minus) / (2 * epsilon)
        assert np.allclose(numeric, J_point[:, k], atol=1e-4), f"J_point column {k} mismatch"


def test_camera_project_without_jacobians_returns_none(bundle_adjustment):
    ba = bundle_adjustment
    T = manif.SE3.Identity()
    P = np.array([0.0, 0.0, 5.0])
    pixel, J_pose, J_point = ba.camera_project(T, P, K, with_jacobians=False)
    assert J_pose is None
    assert J_point is None
    assert pixel.shape == (2,)


def test_camera_project_matches_pinhole_formula(bundle_adjustment):
    ba = bundle_adjustment
    T = manif.SE3.Identity()
    P = np.array([1.0, -0.5, 4.0])
    pixel, _, _ = ba.camera_project(T, P, K)
    fx, fy, cx, cy = K
    expected = np.array([fx * 1.0 / 4.0 + cx, fy * (-0.5) / 4.0 + cy])
    assert np.allclose(pixel, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# compute_visibility
# ---------------------------------------------------------------------------

def test_compute_visibility_front_behind_and_fov(bundle_adjustment):
    ba = bundle_adjustment
    T_cam = manif.SE3.Identity()  # camera at origin, looking toward +z
    P_front_center = np.array([0.0, 0.0, 5.0])   # dead ahead -> visible
    P_behind = np.array([0.0, 0.0, -5.0])         # negative depth -> not visible
    P_outside_fov = np.array([10.0, 0.0, 5.0])    # wide angle -> not visible

    P_true = np.stack([P_front_center, P_behind, P_outside_fov])
    pairs = ba.compute_visibility([T_cam], P_true, fov_deg=60.0)

    assert (0, 0) in pairs
    assert (0, 1) not in pairs
    assert (0, 2) not in pairs
    assert len(pairs) == 1


# ---------------------------------------------------------------------------
# build_observations
# ---------------------------------------------------------------------------

def test_build_observations_raises_on_min_observations_below_2(bundle_adjustment):
    ba = bundle_adjustment
    rng = np.random.default_rng(0)
    T_true, P_true = ba.generate_ground_truth_scene(6, 20, 5.0, 180.0, 2.0, rng)
    with pytest.raises(ValueError):
        ba.build_observations(T_true, P_true, K, 1.0, 70.0, 1, rng)


def test_build_observations_drops_underobserved_landmarks(bundle_adjustment):
    ba = bundle_adjustment
    rng = np.random.default_rng(1)
    T_true, P_true = ba.generate_ground_truth_scene(6, 40, 5.0, 180.0, 2.0, rng)
    min_observations = 3
    P_kept, observations = ba.build_observations(T_true, P_true, K, 1.0, 70.0, min_observations, rng)

    counts = {}
    for i, j, z in observations:
        counts[j] = counts.get(j, 0) + 1
        assert z.shape == (2,)

    # Every kept landmark index actually appears, and with enough observers.
    assert set(counts.keys()) == set(range(len(P_kept)))
    assert all(c >= min_observations for c in counts.values())


# ---------------------------------------------------------------------------
# perturb_initial_guess
# ---------------------------------------------------------------------------

def test_perturb_initial_guess_produces_valid_but_different_poses(bundle_adjustment):
    ba = bundle_adjustment
    rng = np.random.default_rng(2)
    T_true, P_true = ba.generate_ground_truth_scene(4, 10, 5.0, 180.0, 2.0, rng)
    T_init, P_init = ba.perturb_initial_guess(T_true, P_true, pose_noise_std=0.05,
                                               landmark_noise_std=0.1, rng=rng)

    assert len(T_init) == len(T_true)
    assert P_init.shape == P_true.shape
    for T0, T1 in zip(T_true, T_init):
        R = T1.rotation()
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-8)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)
        # Genuinely different pose: the right-minus (log of the relative
        # motion) should be well away from zero.
        assert np.linalg.norm(T0.rminus(T1).coeffs()) > 1e-6
    assert not np.allclose(P_init, P_true, atol=1e-6)


# ---------------------------------------------------------------------------
# Full pipeline regression test
# ---------------------------------------------------------------------------

@pytest.fixture
def small_scene(bundle_adjustment):
    ba = bundle_adjustment
    rng = np.random.default_rng(42)
    T_true, P_true_all = ba.generate_ground_truth_scene(
        n_cameras=6, n_landmarks=40, camera_radius=5.0, arc_span_deg=180.0,
        landmark_spread=2.0, rng=rng)
    pixel_noise_std = 1.0
    P_true, observations = ba.build_observations(
        T_true, P_true_all, K, pixel_noise_std, fov_deg=70.0, min_observations=3, rng=rng)
    pose_noise_std = 0.05
    T_init, P_init = ba.perturb_initial_guess(
        T_true, P_true, pose_noise_std=pose_noise_std, landmark_noise_std=0.2, rng=rng)
    return {
        "T_true": T_true, "P_true": P_true, "observations": observations,
        "T_init": T_init, "P_init": P_init,
        "pose_noise_std": pose_noise_std, "pixel_noise_std": pixel_noise_std,
    }


def test_run_bundle_adjustment_converges_below_noise_floor(bundle_adjustment, small_scene):
    ba = bundle_adjustment
    s = small_scene
    T_est, P_est = ba.run_bundle_adjustment(
        s["T_init"], s["P_init"], s["observations"], K,
        s["pose_noise_std"], s["pixel_noise_std"], gn_tol=1e-8, gn_max_iters=30)

    T_aligned, P_aligned = ba.align_reconstruction_to_ground_truth(s["T_true"], T_est, P_est)

    rms_init = ba.reprojection_rms(s["T_init"], s["P_init"], s["observations"], K)
    rms_ba = ba.reprojection_rms(T_aligned, P_aligned, s["observations"], K)

    # BA should land close to the injected pixel-noise floor and be a large
    # improvement over the noisy initial guess.
    assert rms_ba < 2.0 * s["pixel_noise_std"]
    assert rms_ba < rms_init * 0.5

    rot_err_init, pos_err_init = ba.pose_errors(s["T_true"], s["T_init"])
    _, P_init_aligned = ba.align_reconstruction_to_ground_truth(s["T_true"], s["T_init"], s["P_init"])
    rot_err_ba, pos_err_ba = ba.pose_errors(s["T_true"], T_aligned)

    assert np.sqrt(np.mean(rot_err_ba ** 2)) < np.sqrt(np.mean(rot_err_init ** 2))
    assert np.sqrt(np.mean(pos_err_ba ** 2)) < np.sqrt(np.mean(pos_err_init ** 2))


def test_run_ba_landmarks_only_and_poses_only_reduce_error(bundle_adjustment, small_scene):
    ba = bundle_adjustment
    s = small_scene
    P_landmarks_only = ba.run_ba_landmarks_only(
        s["T_init"], s["P_init"], s["observations"], K, gn_tol=1e-8, gn_max_iters=30)
    T_poses_only = ba.run_ba_poses_only(
        s["T_init"], s["P_init"], s["observations"], K, gn_tol=1e-8, gn_max_iters=30)

    rms_init = ba.reprojection_rms(s["T_init"], s["P_init"], s["observations"], K)
    rms_landmarks_only = ba.reprojection_rms(s["T_init"], P_landmarks_only, s["observations"], K)
    rms_poses_only = ba.reprojection_rms(T_poses_only, s["P_init"], s["observations"], K)

    assert rms_landmarks_only < rms_init
    assert rms_poses_only < rms_init


# ---------------------------------------------------------------------------
# align_reconstruction_to_ground_truth: reprojection RMS is frame-invariant
# ---------------------------------------------------------------------------

def test_alignment_preserves_reprojection_rms(bundle_adjustment, small_scene):
    ba = bundle_adjustment
    s = small_scene
    rms_before = ba.reprojection_rms(s["T_init"], s["P_init"], s["observations"], K)
    T_aligned, P_aligned = ba.align_reconstruction_to_ground_truth(s["T_true"], s["T_init"], s["P_init"])
    rms_after = ba.reprojection_rms(T_aligned, P_aligned, s["observations"], K)
    assert np.isclose(rms_before, rms_after, atol=1e-6)


def test_compute_all_metrics_shape_and_ordering(bundle_adjustment, small_scene):
    ba = bundle_adjustment
    s = small_scene
    rows = [
        ("init", s["T_init"], s["P_init"]),
        ("true", s["T_true"], s["P_true"]),
    ]
    values = ba.compute_all_metrics(rows, s["T_true"], s["P_true"], s["observations"], K)
    assert values.shape == (2, 4)
    # Ground truth vs. itself: pose/landmark error should be exactly 0; the
    # reprojection RMS is against *noisy* observations, so it lands near the
    # injected pixel-noise std rather than 0.
    assert np.allclose(values[1, 0:3], 0.0, atol=1e-8)
    assert values[1, 3] < 2.0 * s["pixel_noise_std"]
    # Noisy init should have strictly larger errors on every metric.
    assert np.all(values[0] > values[1])
