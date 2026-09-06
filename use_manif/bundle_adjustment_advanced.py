'''
Local + Global Bundle Adjustment, contrasted -- manifpy version. See
use_numpy/bundle_adjustment_advanced.py's module docstring for the full
reasoning (identical here, just re-derived on manifpy's Jacobian
out-parameters instead of hand-rolled SE(3) math): a camera moving
sequentially through a scene, building a covisibility graph, and at every
new keyframe running BA over only a bounded *active window* (new keyframe +
covisible neighbors + fixed anchor keyframes) -- docs/bundle_adjustment.md
Section 13's diagram, made concrete -- contrasted against a periodic Global
BA pass over the whole map, the "occasional Global BA pass" Section 13 says
corrects Local BA's accumulated drift.

Poses are manif.SE3 camera-to-world transforms, consistent with every other
manif script in this codebase (T.act(p) convention). Camera pose Jacobians
are obtained by chaining manifpy's own Jacobian out-parameters --
`T.inverse(J_inv)` then `Tinv.act(P, J_self, J_point)`, chained as
`J_pose = J_proj @ J_self @ J_inv` -- the same pattern already used in
use_manif/bundle_adjustment.py's own camera_project. GN/LM pose updates use
manif's own `rplus` (`T + SE3Tangent(delta)`) and relative-motion
composition uses manif's `*` (`Ti.inverse() * Tj`), matching
use_manif/pose_graph.py's dead-reckoning chain. No manifold formula is
hand-rolled; rotation_matrix_to_quaternion_xyzw is a plain representation
conversion (Shepperd's method), not Lie-group math, matching the precedent
already set by use_manif/bundle_adjustment.py.
'''

import argparse
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import manifpy as manif


def rotation_matrix_to_quaternion_xyzw(R):
    """Converts a (3,3) rotation matrix to a manifpy-convention [x,y,z,w]
    quaternion (Shepperd's method) -- used only to build ground-truth
    manif.SE3 poses from a plain rotation matrix; not a Lie-group operation
    itself. Identical to bundle_adjustment.py's helper of the same name.
    Arguments:
        R: (3,3) rotation matrix
    Returns:
        quat_xyzw: (4,) array [x, y, z, w]
    """
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    quat = np.array([x, y, z, w])
    return quat / np.linalg.norm(quat)


def forward_facing_rotation(forward, up_hint=np.array([0.0, 0.0, 1.0])):
    """Builds a camera-to-world rotation whose local +z axis is `forward`
    (e.g. a path's tangent direction) -- analogous to bundle_adjustment.py's
    look_at_rotation, but taking the forward direction directly instead of
    deriving it from a target point.
    Arguments:
        forward: forward direction in world frame (3,)
        up_hint: approximate "up" direction in world frame (3,)
    Returns:
        R: (3,3) camera-to-world rotation (columns = camera x,y,z axes in world coords)
    """
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(forward, right)
    return np.column_stack([right, cam_up, forward])


def generate_ground_truth_trajectory(n_keyframes, path_radius, arc_span_deg):
    """Builds a ground-truth camera trajectory: n_keyframes placed along a
    circular arc, each facing forward along the path's tangent direction
    (direction of travel) rather than toward a fixed centroid -- a
    sequential "moving through a corridor" scene, distinct from
    bundle_adjustment.py's static look-at-centroid camera arc.
    Arguments:
        n_keyframes: number of keyframes along the path
        path_radius: radius of the arc the path follows (m)
        arc_span_deg: total angular span of the path (deg)
    Returns:
        T_true: list of n_keyframes ground-truth camera poses (manif.SE3), in time order
    """
    angles = np.linspace(-np.radians(arc_span_deg) / 2.0, np.radians(arc_span_deg) / 2.0, n_keyframes)
    T_true = []
    for a in angles:
        cam_pos = path_radius * np.array([np.cos(a), np.sin(a), 0.0])
        forward = np.array([-np.sin(a), np.cos(a), 0.0])
        R = forward_facing_rotation(forward)
        quat = rotation_matrix_to_quaternion_xyzw(R)
        T_true.append(manif.SE3(cam_pos, quat))
    return T_true


def generate_landmark_corridor(T_true, n_landmarks_per_keyframe, lateral_spread, vertical_spread, rng):
    """Scatters landmarks near the path centerline: n_landmarks_per_keyframe
    per keyframe station, offset along the local right/up/forward axes.
    Because the offsets are small relative to the whole path, a landmark is
    only geometrically visible from a short run of nearby keyframes --
    covisibility clusters emerge from geometry + field-of-view, not an
    artificial visibility-lifetime rule.
    Arguments:
        T_true: list of ground-truth camera poses (manif.SE3), in time order
        n_landmarks_per_keyframe: landmarks sampled per keyframe station
        lateral_spread: half-width of the lateral (right-axis) offset (m)
        vertical_spread: half-width of the vertical (up-axis) offset (m)
        rng: numpy random number generator
    Returns:
        P_true: (n_keyframes * n_landmarks_per_keyframe, 3) array of ground-truth landmark positions
    """
    positions = np.array([T.translation() for T in T_true])
    spacing = np.linalg.norm(positions[1] - positions[0]) if len(positions) > 1 else 1.0

    landmarks = []
    for T in T_true:
        cam_pos, R = T.translation(), T.rotation()
        right, up, forward = R[:, 0], R[:, 1], R[:, 2]
        for _ in range(n_landmarks_per_keyframe):
            offset = (rng.uniform(-lateral_spread, lateral_spread) * right +
                      rng.uniform(-vertical_spread, vertical_spread) * up +
                      rng.uniform(-spacing, spacing) * forward)
            landmarks.append(cam_pos + offset)
    return np.array(landmarks)


def camera_project(T, P, K, with_jacobians=False):
    """Pinhole reprojection of a world point through a camera pose:
    p_c = T^-1 @ P (world point into the camera frame), then
    u = fx*x/z + cx, v = fy*y/z + cy. Pose Jacobian obtained by chaining
    manifpy's own `inverse`/`act` Jacobian out-parameters. Identical to
    bundle_adjustment.py's camera_project (duplicated rather than imported
    -- this codebase never imports across sibling scripts).
    Arguments:
        T: camera pose, camera-to-world (manif.SE3)
        P: world point (3,)
        K: (fx, fy, cx, cy) camera intrinsics
        with_jacobians: if True, also return (J_pose, J_point)
    Returns:
        pixel: predicted (2,) pixel coordinates
        J_pose: (2,6) Jacobian of pixel wrt a right-perturbation of T, or None
        J_point: (2,3) Jacobian of pixel wrt P, or None
    """
    fx, fy, cx, cy = K

    J_pose = J_point = None
    if with_jacobians:
        J_inv = np.zeros((6, 6))
        T_inv = T.inverse(J_inv)
        J_act_self, J_act_point = np.zeros((3, 6)), np.zeros((3, 3))
        p_c = T_inv.act(P, J_act_self, J_act_point)
    else:
        p_c = T.inverse().act(P)

    x, y, z = p_c
    z_safe = z if abs(z) > 1e-9 else (1e-9 if z >= 0 else -1e-9)
    pixel = np.array([fx * x / z_safe + cx, fy * y / z_safe + cy])

    if with_jacobians:
        J_proj = np.array([
            [fx / z_safe, 0.0, -fx * x / z_safe ** 2],
            [0.0, fy / z_safe, -fy * y / z_safe ** 2],
        ])
        J_pose = J_proj @ J_act_self @ J_inv
        J_point = J_proj @ J_act_point
    return pixel, J_pose, J_point


def compute_visibility(T_true, P_true, fov_deg, max_view_range):
    """For each (keyframe, landmark) pair, decides whether the landmark is
    observed: in front of the camera, within its field of view, and within
    max_view_range -- a real feature detector only resolves a landmark out
    to some finite range, and without this cutoff a forward-facing camera
    looking straight down a corridor sees nearly every landmark from nearly
    every keyframe (it's within the FOV cone for the whole approach), which
    destroys the locality that makes covisibility graphs and bounded local
    windows meaningful in the first place.
    Arguments:
        T_true: list of ground-truth camera poses (manif.SE3)
        P_true: (n_landmarks, 3) array of ground-truth landmark positions
        fov_deg: full field-of-view angle (deg)
        max_view_range: maximum camera-to-landmark distance at which a landmark is detectable (m)
    Returns:
        pairs: list of (keyframe_idx, landmark_idx) visible pairs
    """
    half_fov_cos = np.cos(np.radians(fov_deg) / 2.0)
    pairs = []
    for i, T in enumerate(T_true):
        R, t = T.rotation(), T.translation()
        for j, P in enumerate(P_true):
            p_c = R.T @ (P - t)
            z = p_c[2]
            if z <= 0:
                continue
            range_ = np.linalg.norm(p_c)
            if range_ > max_view_range:
                continue
            cos_angle = z / range_
            if cos_angle >= half_fov_cos:
                pairs.append((i, j))
    return pairs


def build_observations(T_true, P_true, K, pixel_noise_std, fov_deg, max_view_range, min_observations, rng):
    """Determines visibility, drops landmarks seen by fewer than
    min_observations keyframes over the *whole* trajectory (never
    triangulable, so they're never promoted to a map point -- exactly like
    a real system would just never initialize them), and generates noisy
    pixel measurements for every remaining observed pair.
    Arguments:
        T_true: list of ground-truth camera poses (manif.SE3)
        P_true: (n_landmarks_all, 3) array of ground-truth landmark positions
        K: (fx, fy, cx, cy) camera intrinsics
        pixel_noise_std: std-dev of Gaussian pixel noise (px)
        fov_deg: full field-of-view angle (deg)
        max_view_range: maximum camera-to-landmark detection range (m)
        min_observations: minimum number of observing keyframes a landmark needs to be kept (must be >= 2)
        rng: numpy random number generator
    Returns:
        P_true_kept: (n_landmarks, 3) array, landmarks with enough observers only
        observations: list of (keyframe_idx, landmark_idx, z_ij), landmark_idx re-indexed into P_true_kept,
                       in the time order keyframes observe them (keyframe_idx ascending)
    """
    if min_observations < 2:
        raise ValueError(f"min_observations must be >= 2 (a landmark needs >= 2 views to be "
                          f"triangulable), got {min_observations}")

    raw_pairs = compute_visibility(T_true, P_true, fov_deg, max_view_range)

    observers = {}
    for i, j in raw_pairs:
        observers.setdefault(j, []).append(i)

    kept_landmark_ids = sorted(j for j, obs in observers.items() if len(obs) >= min_observations)
    remap = {old: new for new, old in enumerate(kept_landmark_ids)}
    P_true_kept = P_true[kept_landmark_ids]

    observations = []
    for i, j in raw_pairs:
        if j not in remap:
            continue
        pixel, _, _ = camera_project(T_true[i], P_true[j], K)
        z_ij = pixel + rng.normal(0.0, pixel_noise_std, 2)
        observations.append((i, remap[j], z_ij))
    return P_true_kept, observations


def group_observations_by_keyframe(observations, n_keyframes):
    """Groups a flat observation list by keyframe index -- the "online
    arrival order" the incremental loop reveals observations in.
    Arguments:
        observations: list of (keyframe_idx, landmark_idx, z_ij)
        n_keyframes: total number of keyframes
    Returns:
        by_keyframe: list of length n_keyframes, by_keyframe[k] = list of (k, landmark_idx, z_ij)
    """
    by_keyframe = [[] for _ in range(n_keyframes)]
    for i, j, z in observations:
        by_keyframe[i].append((i, j, z))
    return by_keyframe


def simulate_frontend_trajectory(T_true, relative_pose_noise_std, rng):
    """Simulates a front-end (visual-odometry-style) pose estimate for each
    keyframe: chains a noisy version of each true relative motion onto the
    previous estimate, so small per-step errors compound into growing drift
    over the trajectory -- the actual reason Local BA needs an occasional
    Global BA (or loop closure) correction. Same dead-reckoning-chain
    pattern as use_manif/pose_graph.py's simulate_noisy_edges/
    run_dead_reckoning (`Ti.inverse() * Tj` for the relative motion,
    `T + SE3Tangent(noise)` for the perturbation, `Tprev * Znoisy` to chain).
    Arguments:
        T_true: list of ground-truth camera poses (manif.SE3), in time order
        relative_pose_noise_std: std-dev of the se3 twist noise added to each relative motion (mixed m/rad)
        rng: numpy random number generator
    Returns:
        T_init: list of front-end pose estimates (manif.SE3), in time order
    """
    T_init = [T_true[0]]
    for k in range(1, len(T_true)):
        T_rel_true = T_true[k - 1].inverse() * T_true[k]
        T_rel_noisy = T_rel_true + manif.SE3Tangent(rng.normal(0.0, relative_pose_noise_std, 6))
        T_init.append(T_init[-1] * T_rel_noisy)
    return T_init


def pair_key(i, j):
    """Canonical (order-independent) key for an unordered keyframe pair."""
    return (i, j) if i < j else (j, i)


def triangulate_landmark(T_obs_list, z_list, K):
    """Closed-form multi-view ray-intersection initial guess for a new
    landmark: the least-squares point closest to every observing keyframe's
    back-projected ray (docs/bundle_adjustment.md Section 5's "bundle of
    rays" intuition, solved in closed form rather than iteratively).
    Arguments:
        T_obs_list: list of observing camera poses (manif.SE3)
        z_list: list of corresponding observed pixels (2,)
        K: (fx, fy, cx, cy) camera intrinsics
    Returns:
        P0: (3,) initial 3D landmark position guess
    """
    fx, fy, cx, cy = K
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for T, z in zip(T_obs_list, z_list):
        R, o = T.rotation(), T.translation()
        d_cam = np.array([(z[0] - cx) / fx, (z[1] - cy) / fy, 1.0])
        d = R @ d_cam
        d = d / np.linalg.norm(d)
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ o
    return np.linalg.solve(A + np.eye(3) * 1e-9, b)


def refine_landmark_gn(T_obs_list, z_list, P0, K, gn_tol, gn_max_iters):
    """A handful of Gauss-Newton iterations refining one newly-triangulated
    landmark against its observers so far, seeded from
    triangulate_landmark's closed-form guess. Same 3x3 normal-equations
    solve as bundle_adjustment.py's run_ba_landmarks_only, applied to a
    single point at the moment it first becomes triangulable.
    Arguments:
        T_obs_list: list of observing camera poses (manif.SE3)
        z_list: list of corresponding observed pixels (2,)
        P0: (3,) initial landmark position guess
        K: (fx, fy, cx, cy) camera intrinsics
        gn_tol: convergence tolerance on the correction step norm
        gn_max_iters: maximum number of iterations
    Returns:
        P: (3,) refined landmark position
    """
    p = P0.copy()
    for _ in range(gn_max_iters):
        H = np.zeros((3, 3))
        g = np.zeros(3)
        for T, z in zip(T_obs_list, z_list):
            pred, _, J_point = camera_project(T, p, K, with_jacobians=True)
            r = z - pred
            H += J_point.T @ J_point
            g += J_point.T @ r
        delta = np.linalg.solve(H + np.eye(3) * 1e-9, g)
        p = p + delta
        if np.linalg.norm(delta) < gn_tol:
            break
    return p


def point_depth(T, P):
    """Camera-frame z-depth of a world point (positive = in front of the camera)."""
    return T.inverse().act(P)[2]


def passes_cheirality(T_obs_list, P):
    """True if a 3D point has positive depth (point_depth) in every one of
    its observing camera frames. A freshly triangulated point with weak
    parallax (observers close together, as is common for landmarks near
    this corridor's forward-motion path) can fail this: pixel = f*x/z is
    invariant under negating a camera-frame point's x, y AND z together, so
    refine_landmark_gn's unguarded GN can converge to a point reflected
    behind a camera that still fits the pixel data. A point that fails this
    check should not be committed to the map -- see
    run_incremental_local_ba, which instead leaves it pending and retries
    once it gets another observation (more parallax).
    Arguments:
        T_obs_list: list of observing camera poses (manif.SE3)
        P: (3,) candidate landmark position
    Returns:
        True if P has positive depth in every camera in T_obs_list
    """
    return all(point_depth(T, P) > 0.0 for T in T_obs_list)


def cull_invalid_points(T_est, P_est, observers_by_landmark):
    """Removes from P_est any point that currently fails cheirality against
    one of its observers. This can happen even to a point that passed
    passes_cheirality at triangulation time: if a *different* window later
    refines one of that point's observing keyframes without the point
    itself being active in that window, the point's cheirality is never
    re-checked against the camera's new pose. run_windowed_gn_lm's own
    guard only refuses a step that keeps a violation it can see -- it can't
    repair a violation in a point it isn't touching -- so without this
    sweep an already-invalid point would report infinite reprojection cost
    forever in every future window it enters. Culling it lets
    run_incremental_local_ba retry triangulating it from scratch (see the
    "j not in P_est" check there) once it gets another observation.
    Arguments:
        T_est: dict {keyframe_idx: (manif.SE3) pose}, current estimates
        P_est: dict {landmark_idx: (3,) position}, mutated (bad entries removed)
        observers_by_landmark: dict {landmark_idx: list of (keyframe_idx, z_ij)}
    """
    for j in list(P_est.keys()):
        T_obs = [T_est[i] for i, _ in observers_by_landmark[j]]
        if not passes_cheirality(T_obs, P_est[j]):
            del P_est[j]


def build_active_window(k, shared_count, landmarks_by_keyframe, observers_by_landmark, P_est,
                         min_shared_for_covisibility, max_window_keyframes, anchor_keyframes):
    """Builds keyframe k's active BA window: the new keyframe plus its
    strongest covisible neighbors (active_keyframes), every already-
    triangulated landmark any of them observes (active_points), and every
    other keyframe that also observes one of those landmarks, held fixed as
    a rigid anchor (fixed_keyframes) -- directly implementing
    docs/bundle_adjustment.md Section 13's diagram.
    Arguments:
        k: the new keyframe index
        shared_count: dict {pair_key(i,j): shared landmark count}
        landmarks_by_keyframe: dict {keyframe_idx: set of observed landmark indices}
        observers_by_landmark: dict {landmark_idx: list of (keyframe_idx, z_ij)}
        P_est: dict {landmark_idx: (3,) position} of already-triangulated landmarks
        min_shared_for_covisibility: minimum shared-landmark count for a covisibility edge
        max_window_keyframes: maximum number of active keyframes (including k)
        anchor_keyframes: set of keyframe indices that are always hard-fixed (never active)
    Returns:
        active_keyframes: list of keyframe indices being optimized (k first)
        fixed_keyframes: set of keyframe indices held fixed as anchors
        active_points: sorted list of landmark indices being optimized
    """
    neighbors = sorted(
        (i for i in range(k)
         if i not in anchor_keyframes and shared_count.get(pair_key(i, k), 0) >= min_shared_for_covisibility),
        key=lambda i: -shared_count[pair_key(i, k)])
    active_keyframes = [k] + neighbors[:max_window_keyframes - 1]
    active_points = sorted({j for i in active_keyframes for j in landmarks_by_keyframe[i] if j in P_est})
    fixed_keyframes = {i for j in active_points for (i, _) in observers_by_landmark[j]} - set(active_keyframes)
    return active_keyframes, fixed_keyframes, active_points


def run_local_ba_step(T_est, P_est, active_keyframes, fixed_keyframes, active_points,
                       observers_by_landmark, K, gn_tol, gn_max_iters):
    """Solves one bounded local-BA window: refines only the active
    keyframes' poses and active landmarks' positions, holding every
    covisible-but-inactive (fixed) keyframe constant as a rigid anchor.
    T_est/P_est are updated in place for the active entries only.
    Arguments:
        T_est: dict {keyframe_idx: (manif.SE3) pose}, current estimates (mutated for active keyframes)
        P_est: dict {landmark_idx: (3,) position}, current estimates (mutated for active points)
        active_keyframes: list of keyframe indices being optimized
        fixed_keyframes: set of keyframe indices held fixed, contributing residuals only
        active_points: list of landmark indices being optimized
        observers_by_landmark: dict {landmark_idx: list of (keyframe_idx, z_ij)}
        K: (fx, fy, cx, cy) camera intrinsics
        gn_tol: convergence tolerance on the correction step norm
        gn_max_iters: maximum number of accepted GN steps
    Returns:
        rms_before: RMS reprojection error (px) over the window's observations, before the solve
        rms_after: RMS reprojection error (px) over the window's observations, after the solve
    """
    pose_idx = {i: k for k, i in enumerate(active_keyframes)}
    point_idx = {j: k for k, j in enumerate(active_points)}
    window_obs = [(i, j, z) for j in active_points for (i, z) in observers_by_landmark[j]]
    return run_windowed_gn_lm(T_est, P_est, pose_idx, point_idx, window_obs, K, gn_tol, gn_max_iters)


def run_windowed_gn_lm(T_est, P_est, pose_idx, point_idx, obs_list, K, gn_tol, gn_max_iters):
    """Levenberg-Marquardt-damped Gauss-Newton solve of a bundle-adjustment
    system restricted to the poses in pose_idx and the points in point_idx
    -- the shared solver core behind both run_local_ba_step's bounded window
    and run_global_ba's whole-map system. Any keyframe an observation refers
    to that is *not* in pose_idx is treated as fixed (contributes a
    residual/Jacobian against the point only, no pose block).

    Uses adaptive damping (accept a step only if it actually reduces total
    squared reprojection error; otherwise grow the damping and retry)
    instead of bundle_adjustment.py's fixed tiny damping. That script's
    single joint scene is densely observed and prior-anchored, so plain
    Gauss-Newton never leaves its convergence basin -- but a bounded local
    window can be weakly constrained early in the trajectory (few
    observations, little parallax), and plain GN there was found
    (empirically, running this script) to occasionally take a step that
    *increases* the error and then diverge explosively on the next
    iteration's bad linearization. Levenberg-Marquardt is the standard fix
    real solvers (Ceres, g2o, GTSAM) use for exactly this failure mode.

    Also gates out any observation that's already cheirality-inconsistent
    at entry (its point behind that camera given the *current* estimates,
    before this solve touches anything) -- docs/bundle_adjustment.md
    Section 13's table names exactly this ("chi-square gating") as part of
    real Local BA's outlier handling. It matters here because a keyframe
    entering its very first window still carries its raw, drift-compounded
    initial pose guess (see simulate_frontend_trajectory) -- occasionally
    drifted enough that an already-good, well-triangulated point appears
    behind it, which would otherwise deadlock the window forever (any
    accepted step must keep *every* included observation valid, but no
    small damped step can repair a grossly wrong initial orientation in one
    shot -- found empirically running this script). Gating the
    inconsistent pair out lets the window solve on its remaining valid
    observations, refine that keyframe's pose there, and pick the gated
    observation back up naturally in a later window once poses have
    improved enough to agree with it again.
    Arguments:
        T_est: dict {keyframe_idx: (manif.SE3) pose}, current estimates (mutated for keys in pose_idx)
        P_est: dict {landmark_idx: (3,) position}, current estimates (mutated for keys in point_idx)
        pose_idx: dict {keyframe_idx: column-block index} of poses being optimized
        point_idx: dict {landmark_idx: column-block index} of points being optimized
        obs_list: list of (keyframe_idx, landmark_idx, z_ij) observations in this system
        K: (fx, fy, cx, cy) camera intrinsics
        gn_tol: convergence tolerance on the correction step norm
        gn_max_iters: maximum number of accepted GN steps
    Returns:
        rms_before: RMS reprojection error (px) over the *gated* obs_list, before the solve
        rms_after: RMS reprojection error (px) over the *gated* obs_list, after the solve
    """
    obs_list = [(i, j, z) for i, j, z in obs_list if point_depth(T_est[i], P_est[j]) > 0.0]
    pose_dof = 6 * len(pose_idx)
    dof = pose_dof + 3 * len(point_idx)

    def cost_and_rms(T_map, P_map):
        # Cheirality guard: pixel = f*x/z is invariant under negating a
        # camera-frame point's x, y AND z together (reflecting it through
        # the camera center), so without an explicit positive-depth check a
        # weakly-constrained point (few observers, narrow parallax -- common
        # for landmarks near this corridor's forward-motion path) can flip
        # behind a camera to a wildly wrong 3D location while still fitting
        # that view's pixel almost exactly. Rejecting any trial that does
        # this (treating it as infinite cost, so LM shrinks the step and
        # retries) is standard practice in real BA solvers and is what
        # keeps this system from diverging the way plain, unguarded GN/LM
        # was found to (empirically) on this corridor's weak-parallax points.
        if not obs_list:
            return 0.0, 0.0
        sq = []
        for i, j, z in obs_list:
            T, P = T_map[i], P_map[j]
            if point_depth(T, P) <= 0.0:
                return float("inf"), float("inf")
            pred, _, _ = camera_project(T, P, K)
            sq.append(np.sum((z - pred) ** 2))
        return float(np.sum(sq)), float(np.sqrt(np.mean(sq)))

    cost, rms_before = cost_and_rms(T_est, P_est)
    if dof == 0 or not obs_list:
        return rms_before, rms_before

    lam = 1e-3
    for _ in range(gn_max_iters):
        H = np.zeros((dof, dof))
        g = np.zeros(dof)
        for i, j, z in obs_list:
            pred, J_pose, J_point = camera_project(T_est[i], P_est[j], K, with_jacobians=True)
            r = z - pred
            cl = pose_dof + 3 * point_idx[j]
            H[cl:cl + 3, cl:cl + 3] += J_point.T @ J_point
            g[cl:cl + 3] += J_point.T @ r
            if i in pose_idx:
                cp = 6 * pose_idx[i]
                H[cp:cp + 6, cp:cp + 6] += J_pose.T @ J_pose
                H[cp:cp + 6, cl:cl + 3] += J_pose.T @ J_point
                H[cl:cl + 3, cp:cp + 6] += J_point.T @ J_pose
                g[cp:cp + 6] += J_pose.T @ r

        diag_H = np.diag(H).copy()
        diag_H[diag_H < 1e-12] = 1e-12

        accepted = False
        delta = np.zeros(dof)
        for _retry in range(10):
            delta = np.linalg.solve(H + lam * np.diag(diag_H), g)
            T_trial = {i: T_est[i] + manif.SE3Tangent(delta[6 * k:6 * k + 6]) for i, k in pose_idx.items()}
            P_trial = {j: P_est[j] + delta[pose_dof + 3 * k:pose_dof + 3 * k + 3] for j, k in point_idx.items()}
            trial_cost, _ = cost_and_rms({**T_est, **T_trial}, {**P_est, **P_trial})
            if trial_cost < cost:
                T_est.update(T_trial)
                P_est.update(P_trial)
                cost = trial_cost
                lam = max(lam * 0.5, 1e-7)
                accepted = True
                break
            lam *= 2.0

        if not accepted or np.linalg.norm(delta) < gn_tol:
            break

    _, rms_after = cost_and_rms(T_est, P_est)
    return rms_before, rms_after


def run_global_ba(T_est, P_est, keyframe_ids, point_ids, observers_by_landmark, K,
                   anchor_keyframes, gn_tol, gn_max_iters):
    """Joint Levenberg-Marquardt bundle adjustment over the entire map built
    so far -- structurally identical to bundle_adjustment.py's
    run_bundle_adjustment, minus its prior-factor gauge fix: keyframes in
    anchor_keyframes are hard-fixed (never entered into the optimization),
    which already pins the gauge, so no prior factor is needed. This is the
    "occasional Global BA pass" Section 13 says corrects Local BA's
    accumulated drift, and its cost grows with len(keyframe_ids)/
    len(point_ids) -- the contrast to run_local_ba_step's bounded-window
    cost. Delegates the actual solve to run_windowed_gn_lm, the same
    LM-damped core run_local_ba_step uses.
    Arguments:
        T_est: dict {keyframe_idx: (manif.SE3) pose}, current estimates (mutated for non-anchor keyframes)
        P_est: dict {landmark_idx: (3,) position}, current estimates (mutated)
        keyframe_ids: iterable of every keyframe index processed so far
        point_ids: iterable of every landmark index triangulated so far
        observers_by_landmark: dict {landmark_idx: list of (keyframe_idx, z_ij)}
        K: (fx, fy, cx, cy) camera intrinsics
        anchor_keyframes: set of keyframe indices hard-fixed forever
        gn_tol: convergence tolerance on the correction step norm
        gn_max_iters: maximum number of accepted GN steps
    Returns:
        rms_after: RMS reprojection error (px) over every observation in the map, after the solve
    """
    optimizable_kf = [i for i in keyframe_ids if i not in anchor_keyframes]
    point_ids = list(point_ids)
    pose_idx = {i: k for k, i in enumerate(optimizable_kf)}
    point_idx = {j: k for k, j in enumerate(point_ids)}
    all_obs = [(i, j, z) for j in point_ids for (i, z) in observers_by_landmark[j]]
    _, rms_after = run_windowed_gn_lm(T_est, P_est, pose_idx, point_idx, all_obs, K, gn_tol, gn_max_iters)
    return rms_after


def pose_errors_dict(T_true, T_est, keyframe_ids):
    """Rotation geodesic error (deg) and position error (m) for a subset of
    keyframes, keyed by index -- used instead of bundle_adjustment.py's
    list-based pose_errors since keyframes here live in a dict keyed by
    keyframe index, and callers often want only a subset (e.g. excluding
    the hard-fixed anchor keyframes 0/1 from a drift metric).
    Arguments:
        T_true: list of true camera poses (manif.SE3)
        T_est: dict {keyframe_idx: (manif.SE3)} of estimated camera poses
        keyframe_ids: iterable of keyframe indices to evaluate
    Returns:
        rot_err: (n,) array of rotation errors (deg)
        pos_err: (n,) array of position errors (m)
    """
    ids = list(keyframe_ids)
    rot_err, pos_err = np.zeros(len(ids)), np.zeros(len(ids))
    for idx, i in enumerate(ids):
        R_true = manif.SO3(T_true[i].coeffs()[3:7])
        R_est = manif.SO3(T_est[i].coeffs()[3:7])
        rot_err[idx] = np.degrees(np.linalg.norm(R_est.rminus(R_true).coeffs()))
        pos_err[idx] = np.linalg.norm(T_true[i].translation() - T_est[i].translation())
    return rot_err, pos_err


def run_incremental_local_ba(T_true, observations_by_keyframe, K, relative_pose_noise_std,
                              min_observations, min_shared_for_covisibility, max_window_keyframes,
                              global_ba_interval, use_global_ba, gn_tol, gn_max_iters, rng):
    """Processes keyframes k = 0..n-1 in time order: reveals k's
    observations, triangulates any landmark that just became triangulable,
    builds k's active BA window (build_active_window) and solves it
    (run_local_ba_step), and -- if use_global_ba -- runs a full Global BA
    pass (run_global_ba) every global_ba_interval keyframes plus once at the
    end. Keyframes 0 and 1 are hard-fixed forever as the gauge anchor.
    Arguments:
        T_true: list of ground-truth camera poses (manif.SE3), in time order
        observations_by_keyframe: list, observations_by_keyframe[k] = list of (k, landmark_idx, z_ij)
        K: (fx, fy, cx, cy) camera intrinsics
        relative_pose_noise_std: std-dev of the se3 twist noise per frame-to-frame front-end estimate
        min_observations: minimum observer count before a landmark is triangulated (must be >= 2)
        min_shared_for_covisibility: minimum shared-landmark count for a covisibility edge
        max_window_keyframes: maximum number of active keyframes per local BA window
        global_ba_interval: run a Global BA pass every this many keyframes
        use_global_ba: whether to ever run the periodic Global BA pass at all
        gn_tol: convergence tolerance on the correction step norm
        gn_max_iters: maximum number of Gauss-Newton iterations per solve
        rng: numpy random number generator (used only to build the front-end trajectory)
    Returns:
        T_est: dict {keyframe_idx: (manif.SE3)} final pose estimates
        P_est: dict {landmark_idx: (3,)} final landmark position estimates
        history: dict of per-event metrics recorded during the run
    """
    n_keyframes = len(T_true)
    anchor_keyframes = {0, 1}

    T_est = {i: T for i, T in enumerate(simulate_frontend_trajectory(T_true, relative_pose_noise_std, rng))}
    P_est = {}

    observers_by_landmark = defaultdict(list)
    landmarks_by_keyframe = defaultdict(set)
    shared_count = defaultdict(int)

    history = {
        "local_step": [], "local_solve_time": [], "n_active_kf": [], "n_fixed_kf": [],
        "n_active_points": [], "reproj_rms_after": [],
        "global_step": [], "global_solve_time": [],
        "traj_step": [], "traj_rms_pos_err": [],
    }

    for k in range(n_keyframes):
        for (i, j, z) in observations_by_keyframe[k]:
            for (i_prev, _) in observers_by_landmark[j]:
                shared_count[pair_key(i_prev, k)] += 1
            observers_by_landmark[j].append((k, z))
            landmarks_by_keyframe[k].add(j)

            if j not in P_est and len(observers_by_landmark[j]) >= min_observations:
                obs = observers_by_landmark[j]
                T_obs = [T_est[ci] for ci, _ in obs]
                z_obs = [zi for _, zi in obs]
                P0 = triangulate_landmark(T_obs, z_obs, K)
                P_candidate = refine_landmark_gn(T_obs, z_obs, P0, K, gn_tol, gn_max_iters)
                # A candidate that fails cheirality (see passes_cheirality)
                # is a genuine weak-parallax triangulation failure, not a
                # usable point -- leave it pending rather than committing a
                # value that would poison every future window it enters.
                # The "j not in P_est" check above means this is retried
                # automatically the next time j gets another observation.
                if passes_cheirality(T_obs, P_candidate):
                    P_est[j] = P_candidate

        if k in anchor_keyframes:
            continue

        active_keyframes, fixed_keyframes, active_points = build_active_window(
            k, shared_count, landmarks_by_keyframe, observers_by_landmark, P_est,
            min_shared_for_covisibility, max_window_keyframes, anchor_keyframes)

        t0 = time.perf_counter()
        _, rms_after = run_local_ba_step(
            T_est, P_est, active_keyframes, fixed_keyframes, active_points,
            observers_by_landmark, K, gn_tol, gn_max_iters)
        cull_invalid_points(T_est, P_est, observers_by_landmark)
        history["local_step"].append(k)
        history["local_solve_time"].append(time.perf_counter() - t0)
        history["n_active_kf"].append(len(active_keyframes))
        history["n_fixed_kf"].append(len(fixed_keyframes))
        history["n_active_points"].append(len(active_points))
        history["reproj_rms_after"].append(rms_after)

        if use_global_ba and (k % global_ba_interval == 0 or k == n_keyframes - 1):
            t0 = time.perf_counter()
            run_global_ba(T_est, P_est, range(k + 1), list(P_est.keys()), observers_by_landmark, K,
                          anchor_keyframes, gn_tol, gn_max_iters)
            cull_invalid_points(T_est, P_est, observers_by_landmark)
            history["global_step"].append(k)
            history["global_solve_time"].append(time.perf_counter() - t0)

        # A trailing window rather than a cumulative range(2, k+1): RMS'd
        # over every keyframe seen so far would mechanically trend upward
        # just from averaging in more (typically nonzero-error) keyframes,
        # regardless of whether drift is actually growing or Global BA is
        # helping -- it would mask exactly the sawtooth this metric exists
        # to show.
        _, pos_err = pose_errors_dict(T_true, T_est, range(max(2, k - 9), k + 1))
        history["traj_step"].append(k)
        history["traj_rms_pos_err"].append(float(np.sqrt(np.mean(pos_err ** 2))))

    return T_est, P_est, history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--n-keyframes", type=int, default=50, help="Number of keyframes along the path")
    parser.add_argument("--path-radius", type=float, default=15.0, help="Radius of the arc the path follows (m)")
    parser.add_argument("--arc-span-deg", type=float, default=90.0, help="Total angular span of the path (deg)")
    parser.add_argument("--landmarks-per-keyframe", type=int, default=8, help="Landmarks scattered per keyframe station")
    parser.add_argument("--lateral-spread", type=float, default=2.0, help="Half-width of the lateral landmark offset (m)")
    parser.add_argument("--vertical-spread", type=float, default=1.0, help="Half-width of the vertical landmark offset (m)")
    parser.add_argument("--fov-deg", type=float, default=70.0, help="Camera full field-of-view angle (deg)")
    parser.add_argument("--max-view-range", type=float, default=6.0, help="Maximum camera-to-landmark detection range (m) -- bounds covisibility to nearby keyframes, like a real feature detector's effective range")

    parser.add_argument("--image-width", type=int, default=640, help="Image width in pixels (sets cx)")
    parser.add_argument("--image-height", type=int, default=480, help="Image height in pixels (sets cy)")
    parser.add_argument("--focal-length", type=float, default=800.0, help="Shared fx=fy focal length in pixels")

    parser.add_argument("--min-observations", type=int, default=3, help="Minimum number of observing keyframes a landmark needs to be kept/triangulated (must be >= 2)")
    parser.add_argument("--min-shared-for-covisibility", type=int, default=2, help="Minimum shared-landmark count for a covisibility edge between two keyframes")
    parser.add_argument("--max-window-keyframes", type=int, default=6, help="Maximum number of active keyframes per local BA window (including the new one)")
    parser.add_argument("--global-ba-interval", type=int, default=8, help="Run a full Global BA pass every this many keyframes (plus once at the end)")

    parser.add_argument("--relative-pose-noise-std", type=float, default=0.02, help="Std-dev of the se3 twist noise added to each frame-to-frame front-end pose estimate (mixed m/rad) -- compounds into drift over the trajectory")
    parser.add_argument("--pixel-noise-std", type=float, default=1.0, help="Std-dev of Gaussian pixel measurement noise (px)")

    parser.add_argument("--gn-tol", type=float, default=1e-6, help="Gauss-Newton convergence tolerance")
    parser.add_argument("--gn-max-iters", type=int, default=15, help="Maximum Gauss-Newton iterations per solve")

    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")

    args = parser.parse_args()

    K = (args.focal_length, args.focal_length, args.image_width / 2.0, args.image_height / 2.0)

    T_true = generate_ground_truth_trajectory(args.n_keyframes, args.path_radius, args.arc_span_deg)
    P_true_all = generate_landmark_corridor(
        T_true, args.landmarks_per_keyframe, args.lateral_spread, args.vertical_spread, np.random.default_rng(args.seed))
    P_true, observations = build_observations(
        T_true, P_true_all, K, args.pixel_noise_std, args.fov_deg, args.max_view_range, args.min_observations,
        np.random.default_rng(args.seed))
    observations_by_keyframe = group_observations_by_keyframe(observations, args.n_keyframes)

    print(f"Generated {args.n_keyframes} keyframes, {len(P_true)} triangulable landmarks "
          f"(of {len(P_true_all)} sampled), {len(observations)} observations.")

    print("Running incremental Local BA only (no Global BA correction)...")
    T_local, P_local, hist_local = run_incremental_local_ba(
        T_true, observations_by_keyframe, K, args.relative_pose_noise_std, args.min_observations,
        args.min_shared_for_covisibility, args.max_window_keyframes, args.global_ba_interval,
        False, args.gn_tol, args.gn_max_iters, np.random.default_rng(args.seed))

    print("Running incremental Local BA + periodic Global BA...")
    T_hybrid, P_hybrid, hist_hybrid = run_incremental_local_ba(
        T_true, observations_by_keyframe, K, args.relative_pose_noise_std, args.min_observations,
        args.min_shared_for_covisibility, args.max_window_keyframes, args.global_ba_interval,
        True, args.gn_tol, args.gn_max_iters, np.random.default_rng(args.seed))

    avg_local_time = np.mean(hist_hybrid["local_solve_time"])
    print(f"\nLocal BA:  {len(hist_hybrid['local_solve_time'])} calls, "
          f"avg window = {np.mean(hist_hybrid['n_active_kf']):.1f} active + {np.mean(hist_hybrid['n_fixed_kf']):.1f} fixed keyframes, "
          f"{np.mean(hist_hybrid['n_active_points']):.1f} active points, avg solve time = {avg_local_time * 1e3:.3f} ms")
    print(f"Global BA: {len(hist_hybrid['global_solve_time'])} calls, "
          f"solve time grows {hist_hybrid['global_solve_time'][0] * 1e3:.3f} ms -> {hist_hybrid['global_solve_time'][-1] * 1e3:.3f} ms "
          f"as the map grows to {args.n_keyframes} keyframes / {len(P_true)} landmarks")
    print(f"Final reprojection RMS (local windows, last 5 calls): "
          f"{np.mean(hist_hybrid['reproj_rms_after'][-5:]):.3f} px (pixel noise std = {args.pixel_noise_std:.3f} px)")
    print(f"Final RMS trajectory position error: Local-only = {hist_local['traj_rms_pos_err'][-1]:.4f} m, "
          f"Local + Global BA = {hist_hybrid['traj_rms_pos_err'][-1]:.4f} m")
    print("(This path never revisits a place -- no loop closure -- so periodic Global BA has no "
          "genuinely new constraint to exploit, only a joint re-solve of the same information; "
          "it still helps on most noise draws, just not as dramatically as it would with a loop.)")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_scene, ax_time, ax_window, ax_drift = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    true_pos = np.array([T.translation() for T in T_true])
    hybrid_pos = np.array([T_hybrid[i].translation() for i in range(args.n_keyframes)])
    ax_scene.plot(true_pos[:, 0], true_pos[:, 1], "k-", marker="^", markersize=4, label="Ground truth path")
    ax_scene.plot(hybrid_pos[:, 0], hybrid_pos[:, 1], "b--", marker="^", markersize=4, label="Local + Global BA estimate")
    ax_scene.scatter(P_true[:, 0], P_true[:, 1], color="gray", s=8, alpha=0.5, label="Landmarks (ground truth)")
    if P_hybrid:
        p_ids = sorted(P_hybrid.keys())
        p_hybrid_arr = np.array([P_hybrid[j] for j in p_ids])
        ax_scene.scatter(p_hybrid_arr[:, 0], p_hybrid_arr[:, 1], color="tab:blue", s=8, alpha=0.5, label="Landmarks (estimated)")
    ax_scene.set_xlabel("x (m)")
    ax_scene.set_ylabel("y (m)")
    ax_scene.axis("equal")
    ax_scene.set_title("Scene (top-down)")
    ax_scene.legend(fontsize=7, loc="best")

    ax_time.plot(hist_hybrid["local_step"], np.array(hist_hybrid["local_solve_time"]) * 1e3,
                 "o-", color="tab:blue", markersize=3, label="Local BA (bounded window)")
    ax_time.plot(hist_hybrid["global_step"], np.array(hist_hybrid["global_solve_time"]) * 1e3,
                 "s-", color="tab:red", markersize=5, label="Global BA (whole map)")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("Keyframe index")
    ax_time.set_ylabel("Solve time (ms)")
    ax_time.set_title("Solve time: bounded window vs. whole-map")
    ax_time.legend(fontsize=7)

    ax_window.plot(hist_hybrid["local_step"], hist_hybrid["n_active_kf"], color="tab:blue", label="Active keyframes")
    ax_window.plot(hist_hybrid["local_step"], hist_hybrid["n_fixed_kf"], color="tab:orange", label="Fixed keyframes")
    ax_window.set_xlabel("Keyframe index")
    ax_window.set_ylabel("Keyframe count")
    ax_window_pts = ax_window.twinx()
    ax_window_pts.plot(hist_hybrid["local_step"], hist_hybrid["n_active_points"], color="tab:green", label="Active points")
    ax_window_pts.set_ylabel("Point count")
    lines1, labels1 = ax_window.get_legend_handles_labels()
    lines2, labels2 = ax_window_pts.get_legend_handles_labels()
    ax_window.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax_window.set_title("Local BA window size over time (bounded)")

    ax_drift.plot(hist_local["traj_step"], hist_local["traj_rms_pos_err"], color="tab:red", label="Local BA only (drift accumulates)")
    ax_drift.plot(hist_hybrid["traj_step"], hist_hybrid["traj_rms_pos_err"], color="tab:blue", label="Local + periodic Global BA")
    for step in hist_hybrid["global_step"]:
        ax_drift.axvline(step, color="gray", linestyle=":", linewidth=0.7)
    ax_drift.set_xlabel("Keyframe index")
    ax_drift.set_ylabel("RMS trajectory position error (m)")
    ax_drift.set_title("Drift vs. periodic Global BA correction")
    ax_drift.legend(fontsize=7)

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
