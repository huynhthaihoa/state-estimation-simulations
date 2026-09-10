import math

import numpy as np
import pytest


@pytest.fixture
def baa(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "bundle_adjustment_advanced")


K = (800.0, 800.0, 320.0, 240.0)


# ---------------------------------------------------------------------------
# generate_ground_truth_trajectory / generate_landmark_corridor
# ---------------------------------------------------------------------------

def test_generate_ground_truth_trajectory_shapes_and_validity(baa):
    T_true = baa.generate_ground_truth_trajectory(n_keyframes=10, path_radius=15.0, arc_span_deg=90.0)
    assert len(T_true) == 10
    for T in T_true:
        R = T[0:3, 0:3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-8)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)
        assert np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
        # Every keyframe sits on the path_radius circle in the xy-plane.
        assert np.isclose(np.linalg.norm(T[0:3, 3][0:2]), 15.0, atol=1e-8)


def test_generate_landmark_corridor_shape(baa):
    T_true = baa.generate_ground_truth_trajectory(n_keyframes=6, path_radius=15.0, arc_span_deg=90.0)
    rng = np.random.default_rng(0)
    P = baa.generate_landmark_corridor(T_true, n_landmarks_per_keyframe=5, lateral_spread=2.0,
                                        vertical_spread=1.0, rng=rng)
    assert P.shape == (30, 3)


# ---------------------------------------------------------------------------
# compute_visibility (front/behind/FOV/range)
# ---------------------------------------------------------------------------

def test_compute_visibility_respects_fov_and_max_range(baa):
    T_cam = np.eye(4)  # camera at origin, looking toward +z
    P_near_center = np.array([0.0, 0.0, 3.0])     # in front, in FOV, within range
    P_behind = np.array([0.0, 0.0, -3.0])          # negative depth
    P_too_far = np.array([0.0, 0.0, 20.0])         # in front & in FOV but beyond max_view_range
    P_outside_fov = np.array([10.0, 0.0, 3.0])     # in front, within range, outside FOV cone

    P_true = np.stack([P_near_center, P_behind, P_too_far, P_outside_fov])
    pairs = baa.compute_visibility([T_cam], P_true, fov_deg=60.0, max_view_range=6.0)

    assert pairs == [(0, 0)]


# ---------------------------------------------------------------------------
# build_observations
# ---------------------------------------------------------------------------

def test_build_observations_raises_on_min_observations_below_2(baa):
    T_true = baa.generate_ground_truth_trajectory(10, 15.0, 90.0)
    rng = np.random.default_rng(0)
    P_true = baa.generate_landmark_corridor(T_true, 5, 2.0, 1.0, rng)
    with pytest.raises(ValueError):
        baa.build_observations(T_true, P_true, K, 1.0, 70.0, 6.0, 1, rng)


def test_build_observations_drops_underobserved_landmarks(baa):
    T_true = baa.generate_ground_truth_trajectory(10, 15.0, 90.0)
    rng = np.random.default_rng(1)
    P_all = baa.generate_landmark_corridor(T_true, 5, 2.0, 1.0, rng)
    min_observations = 3
    P_kept, observations = baa.build_observations(T_true, P_all, K, 1.0, 70.0, 6.0, min_observations, rng)

    counts = {}
    for i, j, z in observations:
        counts[j] = counts.get(j, 0) + 1
        assert z.shape == (2,)
    assert set(counts.keys()) == set(range(len(P_kept)))
    assert all(c >= min_observations for c in counts.values())


# ---------------------------------------------------------------------------
# triangulate_landmark / refine_landmark_gn
# ---------------------------------------------------------------------------

def test_triangulate_and_refine_landmark_recovers_true_point(baa):
    T_true = baa.generate_ground_truth_trajectory(6, 15.0, 90.0)
    T_obs_list = T_true[2:5]
    P_true_point = T_obs_list[1][0:3, 3] + T_obs_list[1][0:3, 0:3] @ np.array([0.2, 0.1, 4.0])

    # Noise-free observations: triangulation should be near-exact.
    z_list = [baa.camera_project(T, P_true_point, K)[0] for T in T_obs_list]
    P0 = baa.triangulate_landmark(T_obs_list, z_list, K)
    assert np.allclose(P0, P_true_point, atol=1e-6)

    # With small pixel noise, GN refinement from the closed-form guess should
    # still land close to the true point.
    rng = np.random.default_rng(3)
    z_noisy = [z + rng.normal(0.0, 0.5, 2) for z in z_list]
    P0_noisy = baa.triangulate_landmark(T_obs_list, z_noisy, K)
    P_refined = baa.refine_landmark_gn(T_obs_list, z_noisy, P0_noisy, K, gn_tol=1e-9, gn_max_iters=20)
    assert np.linalg.norm(P_refined - P_true_point) < 0.2


# ---------------------------------------------------------------------------
# point_depth / passes_cheirality
# ---------------------------------------------------------------------------

def test_point_depth_and_cheirality(baa):
    T_cam = np.eye(4)
    P_front = np.array([0.0, 0.0, 5.0])
    P_behind = np.array([0.0, 0.0, -5.0])

    assert baa.point_depth(T_cam, P_front) > 0.0
    assert baa.point_depth(T_cam, P_behind) < 0.0

    assert baa.passes_cheirality([T_cam], P_front)
    assert not baa.passes_cheirality([T_cam], P_behind)

    T_cam2 = np.eye(4)
    T_cam2[0:3, 3] = [0.0, 0.0, 20.0]  # facing the same way, further along +z
    # In front of cam1 but behind cam2 (T_cam2 looks toward +z from z=20;
    # a point at z=5 in world sits behind it in camera-frame terms).
    assert not baa.passes_cheirality([T_cam, T_cam2], P_front)


# ---------------------------------------------------------------------------
# cull_invalid_points
# ---------------------------------------------------------------------------

def test_cull_invalid_points_removes_only_invalid(baa):
    T_cam = np.eye(4)
    T_est = {0: T_cam}
    P_valid = np.array([0.0, 0.0, 5.0])
    P_invalid = np.array([0.0, 0.0, -5.0])
    P_est = {0: P_valid.copy(), 1: P_invalid.copy()}
    observers_by_landmark = {
        0: [(0, np.array([320.0, 240.0]))],
        1: [(0, np.array([320.0, 240.0]))],
    }

    baa.cull_invalid_points(T_est, P_est, observers_by_landmark)

    assert 0 in P_est
    assert 1 not in P_est
    assert np.allclose(P_est[0], P_valid)


# ---------------------------------------------------------------------------
# pose_errors_dict
# ---------------------------------------------------------------------------

def test_pose_errors_dict_zero_for_identical_poses(baa):
    T_true = baa.generate_ground_truth_trajectory(6, 15.0, 90.0)
    T_est = {i: T.copy() for i, T in enumerate(T_true)}
    rot_err, pos_err = baa.pose_errors_dict(T_true, T_est, range(6))
    assert rot_err.shape == (6,)
    assert pos_err.shape == (6,)
    assert np.allclose(rot_err, 0.0, atol=1e-8)
    assert np.allclose(pos_err, 0.0, atol=1e-8)


def test_pose_errors_dict_subset_of_keyframes(baa):
    T_true = baa.generate_ground_truth_trajectory(6, 15.0, 90.0)
    T_est = {i: T.copy() for i, T in enumerate(T_true)}
    T_est[3][0:3, 3] += np.array([1.0, 0.0, 0.0])
    rot_err, pos_err = baa.pose_errors_dict(T_true, T_est, [3])
    assert pos_err.shape == (1,)
    assert np.isclose(pos_err[0], 1.0, atol=1e-8)


# ---------------------------------------------------------------------------
# build_active_window
# ---------------------------------------------------------------------------

def test_build_active_window_includes_k_and_respects_anchors(baa):
    anchor_keyframes = {0, 1}
    # Build shared_count using the module's own pair_key helper (imported
    # from utils.py at module import time, so it's bound on the module).
    pair_key = baa.pair_key

    shared_count = {
        pair_key(2, 5): 3,
        pair_key(3, 5): 5,
        pair_key(0, 5): 10,  # anchor -- must never be selected as a neighbor
    }
    landmarks_by_keyframe = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 5: {2, 3}}
    observers_by_landmark = {
        2: [(2, np.zeros(2)), (5, np.zeros(2))],
        3: [(3, np.zeros(2)), (5, np.zeros(2))],
    }
    P_est = {2: np.array([0.0, 0.0, 3.0]), 3: np.array([0.0, 0.0, 3.0])}

    active_keyframes, fixed_keyframes, active_points = baa.build_active_window(
        k=5, shared_count=shared_count, landmarks_by_keyframe=landmarks_by_keyframe,
        observers_by_landmark=observers_by_landmark, P_est=P_est,
        min_shared_for_covisibility=2, max_window_keyframes=6, anchor_keyframes=anchor_keyframes)

    assert active_keyframes[0] == 5
    assert 0 not in active_keyframes  # anchor never becomes active
    assert set(active_keyframes) & anchor_keyframes == set()
    assert active_points == [2, 3]
    # Keyframes 2 and 3 observe the active points and are themselves active,
    # so they should not also show up as fixed anchors.
    assert fixed_keyframes.isdisjoint(active_keyframes)


# ---------------------------------------------------------------------------
# run_global_ba reduces reprojection error on a small hand-built map
# ---------------------------------------------------------------------------

def test_run_global_ba_reduces_reprojection_rms(baa):
    n_keyframes = 12
    T_true = baa.generate_ground_truth_trajectory(n_keyframes, 15.0, 90.0)
    rng = np.random.default_rng(7)
    P_all = baa.generate_landmark_corridor(T_true, 10, 2.0, 1.0, rng)
    P_true, observations = baa.build_observations(T_true, P_all, K, 1.0, 70.0, 8.0, 2, rng)
    assert len(observations) > 0  # sanity: this scene must actually produce triangulable landmarks

    observers_by_landmark = {}
    for i, j, z in observations:
        observers_by_landmark.setdefault(j, []).append((i, z))

    T_est = {i: T @ baa.se3_exp(rng.normal(0.0, 0.03, 6)) for i, T in enumerate(T_true)}
    P_est = {j: P_true[j] + rng.normal(0.0, 0.1, 3) for j in range(len(P_true))}

    anchor_keyframes = {0, 1}

    def total_rms(T_map, P_map):
        sq = []
        for i, j, z in observations:
            pred, _, _ = baa.camera_project(T_map[i], P_map[j], K)
            sq.append(np.sum((z - pred) ** 2))
        return float(np.sqrt(np.mean(sq)))

    rms_before = total_rms(T_est, P_est)
    rms_after = baa.run_global_ba(T_est, P_est, range(n_keyframes), list(P_est.keys()), observers_by_landmark, K,
                                   anchor_keyframes, gn_tol=1e-8, gn_max_iters=20)
    assert rms_after < rms_before


# ---------------------------------------------------------------------------
# Key regression test: run_incremental_local_ba, Global BA helps on most seeds
# ---------------------------------------------------------------------------

def _run_pair(baa, seed, n_keyframes=12):
    T_true = baa.generate_ground_truth_trajectory(n_keyframes=n_keyframes, path_radius=15.0, arc_span_deg=90.0)
    # landmarks_per_keyframe/max_view_range/min_observations tuned (verified
    # empirically) so this small a scene still yields plenty of covisible,
    # triangulable landmarks -- with the defaults (max_view_range=6,
    # min_observations=3) most seeds at this keyframe count produce *zero*
    # triangulable landmarks, which would make the Global-BA-vs-Local-only
    # comparison below vacuous (both runs trivially tie).
    P_all = baa.generate_landmark_corridor(T_true, n_landmarks_per_keyframe=10, lateral_spread=2.0,
                                            vertical_spread=1.0, rng=np.random.default_rng(seed))
    P_true, observations = baa.build_observations(
        T_true, P_all, K, pixel_noise_std=1.0, fov_deg=70.0, max_view_range=8.0,
        min_observations=2, rng=np.random.default_rng(seed))
    assert len(observations) > 0

    observations_by_keyframe = baa.group_observations_by_keyframe(observations, n_keyframes)

    common_kwargs = dict(
        T_true=T_true, observations_by_keyframe=observations_by_keyframe, K=K,
        relative_pose_noise_std=0.02, min_observations=2, min_shared_for_covisibility=2,
        max_window_keyframes=6, global_ba_interval=4, gn_tol=1e-6, gn_max_iters=15,
    )

    # Both runs get a fresh rng seeded identically, so the front-end
    # trajectory noise (the only randomness run_incremental_local_ba itself
    # consumes) is identical between the two -- an apples-to-apples
    # comparison, matching the module's own main().
    _, _, hist_local = baa.run_incremental_local_ba(
        use_global_ba=False, rng=np.random.default_rng(seed), **common_kwargs)
    _, _, hist_hybrid = baa.run_incremental_local_ba(
        use_global_ba=True, rng=np.random.default_rng(seed), **common_kwargs)

    return hist_local["traj_rms_pos_err"][-1], hist_hybrid["traj_rms_pos_err"][-1]


@pytest.mark.parametrize("seeds", [(0, 1, 2, 3, 4)])
def test_global_ba_beats_local_only_on_most_seeds(baa, seeds):
    results = [_run_pair(baa, seed) for seed in seeds]
    successes = sum(1 for rms_local, rms_hybrid in results if rms_hybrid <= rms_local)
    # The module's own docs note Global BA helps "on most, not all" noise
    # realizations on this open (non-looping) arc -- so require a majority,
    # not every single seed, to avoid a flaky test.
    assert successes >= math.ceil(0.6 * len(seeds)), (
        f"Global BA only beat Local-only on {successes}/{len(seeds)} seeds: {results}")
