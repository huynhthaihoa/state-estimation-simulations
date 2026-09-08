'''
Bundle Adjustment: jointly refines camera poses AND 3D landmarks against
reprojection error -- the classic multi-view structure-from-motion problem,
matching docs/bundle_adjustment.md's

    min_{T_i,P_j} sum_{(i,j) in O} ||z_ij - pi(T_i^-1 P_j)||^2

(O = the set of (camera, landmark) pairs actually observed -- not every
camera sees every landmark; a camera's field of view controls which pairs
land in O).

Poses T_i are (4,4) camera-to-world transforms, consistent with every other
script in this codebase (T.act(p) convention); the doc's own p=pi(TP)
shorthand omits the inverse, which is fine for an intro-level page but not
literally what's implemented here.

Three solvers make the doc's central "jointly" thesis (Section 4) concrete:

  - run_ba_landmarks_only: poses held fixed at their noisy initial guess,
    only landmarks refined (the "cameras are correct" strawman). Since poses
    are constants here, each landmark's residual only depends on that one
    landmark, so this decouples into independent 3x3 GN solves -- classic
    multi-view triangulation from known poses. No gauge freedom: the fixed
    poses already pin the coordinate frame and scale.

  - run_ba_poses_only: landmarks held fixed, only poses refined (the
    "points are correct" strawman). Symmetric: independent 6x6 GN solves per
    camera -- classic PnP-style resection. Also no gauge freedom, same
    reason.

  - run_bundle_adjustment: both refined jointly -- the actual thing the doc
    describes. This reintroduces the classic monocular BA gauge freedom
    (6-DoF rigid + 1-DoF scale: scaling every camera translation and every
    landmark by lambda about camera 0's position leaves every reprojection
    unchanged), fixed with a *prior factor* on T_0 and T_1 (mean = their own
    noisy initial guess), the same prior-factor pattern
    pointcloud_pose_tracking.py's run_batch_gn already uses for its own
    gauge freedom -- a finite prior, not a hard anchor, so camera 1's actual
    pose error is still correctable by its real observations; only the
    otherwise-unobservable gauge directions get pulled toward the initial
    guess.

Hand-rolled SE(3) math (skew/exp/log/inverse-right-Jacobian) comes from
lie_utils.py, shared with pointcloud_pose_tracking.py and pose_graph.py.
'''

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lie_utils import skew, se3_exp, se3_log, se3_inv, compute_se3_inv_right_jacobian, rotation_geodesic_error
from utils import look_at_rotation, umeyama_alignment, landmark_errors


def generate_ground_truth_scene(n_cameras, n_landmarks, camera_radius, arc_span_deg, landmark_spread, rng):
    """Builds a ground-truth scene: n_landmarks scattered in a cube around
    the origin, and n_cameras placed on a horizontal arc looking inward at
    the landmark cluster's centroid.
    Arguments:
        n_cameras: number of cameras
        n_landmarks: number of 3D landmarks
        camera_radius: radius of the camera arc, centered on the landmark centroid (m)
        arc_span_deg: total angular span of the camera arc (deg)
        landmark_spread: half-width of the cube landmarks are sampled in (m)
        rng: numpy random number generator
    Returns:
        T_true: list of n_cameras ground-truth camera poses (4,4)
        P_true: (n_landmarks, 3) array of ground-truth landmark positions
    """
    P_true = rng.uniform(-landmark_spread, landmark_spread, size=(n_landmarks, 3))
    centroid = P_true.mean(axis=0)

    angles = np.linspace(-np.radians(arc_span_deg) / 2.0, np.radians(arc_span_deg) / 2.0, n_cameras)
    T_true = []
    for a in angles:
        cam_pos = centroid + camera_radius * np.array([np.cos(a), np.sin(a), 0.0])
        R = look_at_rotation(cam_pos, centroid)
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = cam_pos
        T_true.append(T)
    return T_true, P_true


def camera_project(T, P, K, with_jacobians=False):
    """Pinhole reprojection of a world point through a camera pose:
    p_c = T^-1 @ P (world point into the camera frame), then
    u = fx*x/z + cx, v = fy*y/z + cy.
    Arguments:
        T: camera pose, camera-to-world (4,4)
        P: world point (3,)
        K: (fx, fy, cx, cy) camera intrinsics
        with_jacobians: if True, also return (J_pose, J_point)
    Returns:
        pixel: predicted (2,) pixel coordinates
        J_pose: (2,6) Jacobian of pixel wrt a right-perturbation of T ([dv,dw] order), or None
        J_point: (2,3) Jacobian of pixel wrt P, or None
    """
    fx, fy, cx, cy = K
    R, t = T[0:3, 0:3], T[0:3, 3]
    p_c = R.T @ (P - t)
    x, y, z = p_c
    z_safe = z if abs(z) > 1e-9 else (1e-9 if z >= 0 else -1e-9)

    pixel = np.array([fx * x / z_safe + cx, fy * y / z_safe + cy])

    J_pose = J_point = None
    if with_jacobians:
        J_proj = np.array([
            [fx / z_safe, 0.0, -fx * x / z_safe ** 2],
            [0.0, fy / z_safe, -fy * y / z_safe ** 2],
        ])
        # d(p_c)/d(right-perturbation of T) = [-I | skew(p_c)]: T(d)=T@Exp(d)
        # => T(d)^-1 = Exp(-d)@T^-1, so p_c(d) = Exp(-d)@p_c(0), and the
        # standard identity d(Exp(e)@q)/de|_0 = [I|-skew(q)] with e=-d gives
        # d(p_c)/dd = [I|-skew(p_c)] @ (-I) block-wise = [-I|skew(p_c)].
        J_pc_wrt_pose = np.hstack([-np.eye(3), skew(p_c)])
        J_pose = J_proj @ J_pc_wrt_pose
        J_point = J_proj @ R.T
    return pixel, J_pose, J_point


def compute_visibility(T_true, P_true, fov_deg):
    """For each (camera, landmark) pair, decides whether the landmark is
    observed: in front of the camera and within its field of view.
    Arguments:
        T_true: list of ground-truth camera poses (4,4)
        P_true: (n_landmarks, 3) array of ground-truth landmark positions
        fov_deg: full field-of-view angle (deg)
    Returns:
        pairs: list of (cam_idx, landmark_idx) visible pairs
    """
    half_fov_cos = np.cos(np.radians(fov_deg) / 2.0)
    pairs = []
    for i, T in enumerate(T_true):
        R, t = T[0:3, 0:3], T[0:3, 3]
        for j, P in enumerate(P_true):
            p_c = R.T @ (P - t)
            z = p_c[2]
            if z <= 0:
                continue
            cos_angle = z / np.linalg.norm(p_c)
            if cos_angle >= half_fov_cos:
                pairs.append((i, j))
    return pairs


def build_observations(T_true, P_true, K, pixel_noise_std, fov_deg, min_observations, rng):
    """Determines visibility, drops landmarks seen by fewer than
    min_observations cameras (not triangulable -- a single 2D observation
    only constrains a ray through 3D space, so min_observations must be
    >= 2 or run_ba_landmarks_only's per-landmark 3x3 normal-equations solve
    would be rank-deficient), and generates noisy pixel measurements for
    every remaining observed pair -- this is what makes the doc's O a real
    strict subset of all (i,j) pairs.
    Arguments:
        T_true: list of ground-truth camera poses (4,4)
        P_true: (n_landmarks_all, 3) array of ground-truth landmark positions
        K: (fx, fy, cx, cy) camera intrinsics
        pixel_noise_std: std-dev of Gaussian pixel noise (px)
        fov_deg: full field-of-view angle (deg)
        min_observations: minimum number of observing cameras a landmark
                           needs to be kept (must be >= 2)
        rng: numpy random number generator
    Returns:
        P_true_kept: (n_landmarks, 3) array, landmarks with enough observers only
        observations: list of (cam_idx, landmark_idx, z_ij), landmark_idx re-indexed into P_true_kept
    """
    if min_observations < 2:
        raise ValueError(f"min_observations must be >= 2 (a landmark needs >= 2 views to be "
                          f"triangulable), got {min_observations}")

    raw_pairs = compute_visibility(T_true, P_true, fov_deg)

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


def perturb_initial_guess(T_true, P_true, pose_noise_std, landmark_noise_std, rng):
    """Builds the noisy initial guess bundle adjustment must correct: every
    camera pose perturbed by a small random se3 twist, every landmark
    perturbed by a Gaussian offset -- both sides simultaneously wrong, the
    premise behind doc Section 4.
    Arguments:
        T_true: list of ground-truth camera poses (4,4)
        P_true: (n_landmarks, 3) array of ground-truth landmark positions
        pose_noise_std: std-dev of the se3 perturbation twist (mixed m and rad)
        landmark_noise_std: std-dev of the landmark position offset (m)
        rng: numpy random number generator
    Returns:
        T_init: list of perturbed initial camera poses (4,4)
        P_init: (n_landmarks, 3) array of perturbed initial landmark positions
    """
    T_init = [T @ se3_exp(rng.normal(0.0, pose_noise_std, 6)) for T in T_true]
    P_init = P_true + rng.normal(0.0, landmark_noise_std, P_true.shape)
    return T_init, P_init


def run_ba_landmarks_only(T_fixed, P_init, observations, K, gn_tol, gn_max_iters):
    """Refines only the landmarks, holding every camera pose fixed at
    T_fixed (the "cameras are correct" strawman from doc Section 4). Since
    poses are constants here, each landmark's residual depends only on that
    landmark, so this decouples into independent 3x3 GN solves -- classic
    triangulation from known poses. No gauge freedom.
    Arguments:
        T_fixed: list of fixed camera poses (4,4)
        P_init: (n_landmarks, 3) initial landmark guess
        observations: list of (cam_idx, landmark_idx, z_ij)
        K: (fx, fy, cx, cy) camera intrinsics
        gn_tol: convergence tolerance on the correction step norm
        gn_max_iters: maximum number of iterations
    Returns:
        P_est: (n_landmarks, 3) refined landmark positions
    """
    obs_by_landmark = {}
    for i, j, z_ij in observations:
        obs_by_landmark.setdefault(j, []).append((i, z_ij))

    P_est = P_init.copy()
    for j, obs in obs_by_landmark.items():
        p = P_est[j].copy()
        for _ in range(gn_max_iters):
            H = np.zeros((3, 3))
            g = np.zeros(3)
            for i, z_ij in obs:
                pred, _, J_point = camera_project(T_fixed[i], p, K, with_jacobians=True)
                r = z_ij - pred
                H += J_point.T @ J_point
                g += J_point.T @ r
            delta = np.linalg.solve(H + np.eye(3) * 1e-9, g)
            p = p + delta
            if np.linalg.norm(delta) < gn_tol:
                break
        P_est[j] = p
    return P_est


def run_ba_poses_only(T_init, P_fixed, observations, K, gn_tol, gn_max_iters):
    """Refines only the camera poses, holding every landmark fixed at
    P_fixed (the "points are correct" strawman from doc Section 4).
    Symmetric to run_ba_landmarks_only: each camera's residual depends only
    on that camera, so this decouples into independent 6x6 GN solves --
    classic PnP-style resection. No gauge freedom.
    Arguments:
        T_init: list of initial camera pose guesses (4,4)
        P_fixed: (n_landmarks, 3) fixed landmark positions
        observations: list of (cam_idx, landmark_idx, z_ij)
        K: (fx, fy, cx, cy) camera intrinsics
        gn_tol: convergence tolerance on the correction step norm
        gn_max_iters: maximum number of iterations
    Returns:
        T_est: list of refined camera poses (4,4)
    """
    obs_by_camera = {}
    for i, j, z_ij in observations:
        obs_by_camera.setdefault(i, []).append((j, z_ij))

    T_est = list(T_init)
    for i, obs in obs_by_camera.items():
        T = T_est[i]
        for _ in range(gn_max_iters):
            H = np.zeros((6, 6))
            g = np.zeros(6)
            for j, z_ij in obs:
                pred, J_pose, _ = camera_project(T, P_fixed[j], K, with_jacobians=True)
                r = z_ij - pred
                H += J_pose.T @ J_pose
                g += J_pose.T @ r
            delta = np.linalg.solve(H + np.eye(6) * 1e-9, g)
            T = T @ se3_exp(delta)
            if np.linalg.norm(delta) < gn_tol:
                break
        T_est[i] = T
    return T_est


def run_bundle_adjustment(T_init, P_init, observations, K, pose_noise_std, pixel_noise_std, gn_tol, gn_max_iters):
    """Joint Gauss-Newton bundle adjustment: refines every camera pose AND
    every landmark together against every observation's reprojection
    residual -- the actual thing doc Section 4 describes ("adjust
    everything together"). Unlike the two one-sided solvers above, jointly
    optimizing both reintroduces the classic monocular BA gauge freedom
    (6-DoF rigid + 1-DoF scale -- verified by hand: scaling every camera
    translation and every landmark by lambda about camera 0's position
    leaves every reprojection unchanged), fixed with a prior factor on T_0
    and T_1 (mean = their own noisy initial guess), the same prior-factor
    pattern pointcloud_pose_tracking.py's run_batch_gn already uses for its
    own gauge freedom -- a *finite* prior, not a hard anchor, so camera 1's
    actual pose error is still correctable by its real observations; only
    the otherwise-unobservable gauge directions get pulled toward the
    initial guess.
    Arguments:
        T_init: list of initial camera pose guesses (4,4)
        P_init: (n_landmarks, 3) initial landmark guess
        observations: list of (cam_idx, landmark_idx, z_ij)
        K: (fx, fy, cx, cy) camera intrinsics
        pose_noise_std: std-dev used to build the T_0/T_1 gauge-prior covariance
        pixel_noise_std: std-dev of the pixel measurement noise, used to weight reprojection factors
        gn_tol: convergence tolerance on the correction step norm
        gn_max_iters: maximum number of iterations
    Returns:
        T_est: list of refined camera poses (4,4)
        P_est: (n_landmarks, 3) refined landmark positions
    """
    n_cameras = len(T_init)
    n_landmarks = len(P_init)
    pose_dof = 6 * n_cameras
    dof = pose_dof + 3 * n_landmarks

    Omega_prior = np.eye(6) / pose_noise_std ** 2
    omega_pixel = 1.0 / pixel_noise_std ** 2
    T_prior = [T_init[0], T_init[1]]

    T_est = list(T_init)
    P_est = P_init.copy()

    for it in range(gn_max_iters):
        H = np.zeros((dof, dof))
        g = np.zeros(dof)

        # --- Gauge-fixing prior factors on T_0 and T_1 ---
        for k in (0, 1):
            e0 = se3_log(se3_inv(T_prior[k]) @ T_est[k])
            J = compute_se3_inv_right_jacobian(e0)
            c = 6 * k
            H[c:c + 6, c:c + 6] += J.T @ Omega_prior @ J
            g[c:c + 6] += -J.T @ Omega_prior @ e0

        # --- Reprojection factors ---
        total_reproj_sq = 0.0
        for i, j, z_ij in observations:
            pred, J_pose, J_point = camera_project(T_est[i], P_est[j], K, with_jacobians=True)
            r = z_ij - pred
            total_reproj_sq += r @ r

            cp, cl = 6 * i, pose_dof + 3 * j
            H[cp:cp + 6, cp:cp + 6] += omega_pixel * (J_pose.T @ J_pose)
            H[cl:cl + 3, cl:cl + 3] += omega_pixel * (J_point.T @ J_point)
            H[cp:cp + 6, cl:cl + 3] += omega_pixel * (J_pose.T @ J_point)
            H[cl:cl + 3, cp:cp + 6] += omega_pixel * (J_point.T @ J_pose)
            g[cp:cp + 6] += omega_pixel * (J_pose.T @ r)
            g[cl:cl + 3] += omega_pixel * (J_point.T @ r)

        H += np.eye(dof) * 1e-6
        delta = np.linalg.solve(H, g)

        for i in range(n_cameras):
            T_est[i] = T_est[i] @ se3_exp(delta[6 * i:6 * i + 6])
        for j in range(n_landmarks):
            P_est[j] = P_est[j] + delta[pose_dof + 3 * j:pose_dof + 3 * j + 3]

        step_norm = np.linalg.norm(delta)
        print(f"    Iteration {it + 1}: reprojection SSE = {total_reproj_sq:.4f}, |delta| = {step_norm:.6f}")
        if step_norm < gn_tol:
            print(f"    Converged after {it + 1} iteration(s) (|delta| < {gn_tol})")
            break
    else:
        print(f"    Reached max iterations ({gn_max_iters}) without full convergence")

    return T_est, P_est


def align_reconstruction_to_ground_truth(T_true, T_est, P_est):
    """Aligns a bundle-adjustment result to ground truth via the best-fit
    similarity (scale + rotation + translation) between estimated and true
    camera positions (Umeyama, 1991), then applies that same transform to
    the landmarks. This is standard practice for evaluating monocular BA/SfM
    output (e.g. absolute-trajectory-error benchmarks): joint BA only
    recovers the scene up to an unknown global similarity (its gauge
    freedom -- see run_bundle_adjustment's docstring), so comparing its raw
    output to ground truth would conflate genuine reconstruction error with
    an arbitrary, meaningless gauge offset. Reprojection error is unaffected
    by this alignment -- a global similarity applied consistently to both
    cameras and landmarks leaves every reprojection exactly unchanged.
    Arguments:
        T_true: list of true camera poses (4,4)
        T_est: list of estimated camera poses (4,4)
        P_est: (n_landmarks, 3) estimated landmark positions
    Returns:
        T_aligned: list of aligned camera poses (4,4)
        P_aligned: (n_landmarks, 3) aligned landmark positions
    """
    est_positions = np.array([T[0:3, 3] for T in T_est])
    true_positions = np.array([T[0:3, 3] for T in T_true])
    s, R, t = umeyama_alignment(est_positions, true_positions)

    T_aligned = []
    for T in T_est:
        T_new = np.eye(4)
        T_new[0:3, 0:3] = R @ T[0:3, 0:3]
        T_new[0:3, 3] = s * (R @ T[0:3, 3]) + t
        T_aligned.append(T_new)
    P_aligned = s * (P_est @ R.T) + t
    return T_aligned, P_aligned


def pose_errors(T_true_list, T_est_list):
    """Rotation geodesic error (deg) and position error (m), per camera.
    Arguments:
        T_true_list: list of true camera poses (4,4)
        T_est_list: list of estimated camera poses (4,4)
    Returns:
        rot_err: (n_cameras,) array of rotation errors (deg)
        pos_err: (n_cameras,) array of position errors (m)
    """
    n = len(T_true_list)
    rot_err, pos_err = np.zeros(n), np.zeros(n)
    for k in range(n):
        rot_err[k] = np.degrees(rotation_geodesic_error(T_true_list[k][0:3, 0:3], T_est_list[k][0:3, 0:3]))
        pos_err[k] = np.linalg.norm(T_true_list[k][0:3, 3] - T_est_list[k][0:3, 3])
    return rot_err, pos_err


def reprojection_rms(T_list, P_list, observations, K):
    """RMS pixel reprojection error over every observed (camera, landmark)
    pair -- directly implements docs/bundle_adjustment.md's own
    sum_{(i,j) in O} ||z_ij - pi(T_i P_j)||^2 objective.
    Arguments:
        T_list: list of camera poses (4,4)
        P_list: (n_landmarks, 3) array of landmark positions
        observations: list of (cam_idx, landmark_idx, z_ij)
        K: (fx, fy, cx, cy) camera intrinsics
    Returns:
        rms: RMS reprojection error (px)
    """
    sq_errors = []
    for i, j, z_ij in observations:
        pred, _, _ = camera_project(T_list[i], P_list[j], K)
        sq_errors.append(np.sum((z_ij - pred) ** 2))
    return float(np.sqrt(np.mean(sq_errors)))


def compute_all_metrics(rows, T_true, P_true, observations, K):
    """RMS pose-rotation, pose-position, landmark, and reprojection error
    for each (name, T_list, P_list) row.
    Arguments:
        rows: list of (name, T_list, P_list)
        T_true: list of true camera poses (4,4)
        P_true: (n_landmarks, 3) array of true landmark positions
        observations: list of (cam_idx, landmark_idx, z_ij)
        K: (fx, fy, cx, cy) camera intrinsics
    Returns:
        values: (n_rows, 4) array of [rot_rms_deg, pos_rms_m, landmark_rms_m, reproj_rms_px]
    """
    values = np.zeros((len(rows), 4))
    for k, (_, T_list, P_list) in enumerate(rows):
        rot_err, pos_err = pose_errors(T_true, T_list)
        land_err = landmark_errors(P_true, P_list)
        reproj = reprojection_rms(T_list, P_list, observations, K)
        values[k] = [np.sqrt(np.mean(rot_err ** 2)), np.sqrt(np.mean(pos_err ** 2)),
                     np.sqrt(np.mean(land_err ** 2)), reproj]
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--n-cameras", type=int, default=8, help="Number of cameras placed on the arc")
    parser.add_argument("--n-landmarks", type=int, default=60, help="Number of 3D landmarks sampled (before dropping under-observed ones)")
    parser.add_argument("--camera-radius", type=float, default=5.0, help="Radius of the camera arc, centered on the landmark centroid (m)")
    parser.add_argument("--arc-span-deg", type=float, default=180.0, help="Total angular span of the camera arc (deg)")
    parser.add_argument("--landmark-spread", type=float, default=2.0, help="Half-width of the cube landmarks are sampled in (m)")
    parser.add_argument("--fov-deg", type=float, default=70.0, help="Camera full field-of-view angle; controls which landmarks each camera observes (deg)")
    parser.add_argument("--min-observations", type=int, default=2, help="Minimum number of observing cameras a landmark needs to be kept (must be >= 2, since a landmark needs >= 2 views to be triangulable)")

    parser.add_argument("--image-width", type=int, default=640, help="Image width in pixels (sets cx)")
    parser.add_argument("--image-height", type=int, default=480, help="Image height in pixels (sets cy)")
    parser.add_argument("--focal-length", type=float, default=800.0, help="Shared fx=fy focal length in pixels")

    parser.add_argument("--pose-noise-std", type=float, default=0.1, help="Std-dev of the se3 twist used to perturb the initial camera-pose guess (mixed m/rad)")
    parser.add_argument("--landmark-noise-std", type=float, default=0.3, help="Std-dev of the Gaussian offset used to perturb the initial landmark guess (m)")
    parser.add_argument("--pixel-noise-std", type=float, default=1.0, help="Std-dev of Gaussian pixel measurement noise (px)")

    parser.add_argument("--gn-tol", type=float, default=1e-6, help="Gauss-Newton convergence tolerance")
    parser.add_argument("--gn-max-iters", type=int, default=30, help="Maximum Gauss-Newton iterations")

    parser.add_argument("--seed", type=int, default=0, help="RNG seed")

    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    K = (args.focal_length, args.focal_length, args.image_width / 2.0, args.image_height / 2.0)

    T_true, P_true_all = generate_ground_truth_scene(
        args.n_cameras, args.n_landmarks, args.camera_radius, args.arc_span_deg, args.landmark_spread, rng)

    P_true, observations = build_observations(
        T_true, P_true_all, K, args.pixel_noise_std, args.fov_deg, args.min_observations, rng)
    n_landmarks = len(P_true)

    print(f"Generated {args.n_cameras} cameras, {n_landmarks} triangulable landmarks "
          f"(of {args.n_landmarks} sampled), {len(observations)} observations "
          f"({len(observations) / (args.n_cameras * n_landmarks):.1%} of all possible camera-landmark pairs -- "
          f"this is the doc's O).")

    T_init, P_init = perturb_initial_guess(T_true, P_true, args.pose_noise_std, args.landmark_noise_std, rng)

    print("Refining landmarks only (cameras held fixed at noisy initial guess)...")
    P_landmarks_only = run_ba_landmarks_only(T_init, P_init, observations, K, args.gn_tol, args.gn_max_iters)

    print("Refining poses only (landmarks held fixed at noisy initial guess)...")
    T_poses_only = run_ba_poses_only(T_init, P_init, observations, K, args.gn_tol, args.gn_max_iters)

    print("Running full joint bundle adjustment...")
    T_ba, P_ba = run_bundle_adjustment(
        T_init, P_init, observations, K, args.pose_noise_std, args.pixel_noise_std, args.gn_tol, args.gn_max_iters)

    # Joint BA only recovers the scene up to an unknown global similarity
    # (its gauge freedom, see run_bundle_adjustment's docstring) -- align to
    # ground truth before comparing absolute pose/landmark error, standard
    # practice for evaluating monocular BA/SfM output. Reprojection error is
    # unaffected by this (see align_reconstruction_to_ground_truth).
    T_ba, P_ba = align_reconstruction_to_ground_truth(T_true, T_ba, P_ba)

    rows = [
        ("Noisy initial guess", T_init, P_init),
        ("Landmarks-only refinement", T_init, P_landmarks_only),
        ("Poses-only refinement", T_poses_only, P_init),
        ("Full joint bundle adjustment", T_ba, P_ba),
    ]
    values = compute_all_metrics(rows, T_true, P_true, observations, K)

    print("\nRMS errors (pose rotation/position, landmark position, reprojection):")
    for (name, _, _), row_vals in zip(rows, values):
        print(f"  {name:<29s} pose RMS rot={row_vals[0]:6.3f} deg, pos={row_vals[1]:7.4f} m | "
              f"landmark RMS={row_vals[2]:7.4f} m | reprojection RMS={row_vals[3]:7.3f} px")

    fig, (ax_scene, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))

    def xy(points):
        pts = np.asarray(points)
        return pts[:, 0], pts[:, 1]

    def camera_positions_and_headings(T_list, length=0.3):
        pos = np.array([T[0:3, 3] for T in T_list])
        heading = np.array([T[0:3, 0:3] @ np.array([0.0, 0.0, length]) for T in T_list])
        return pos, heading

    for T_list, P_list, color, label in [
        (T_true, P_true, "black", "ground truth"),
        (T_init, P_init, "tab:gray", "noisy init"),
        (T_ba, P_ba, "tab:blue", "joint BA"),
    ]:
        ax_scene.scatter(*xy(P_list), color=color, marker="o", s=12, alpha=0.6, label=f"Landmarks ({label})")
        pos, heading = camera_positions_and_headings(T_list)
        ax_scene.scatter(pos[:, 0], pos[:, 1], color=color, marker="^", s=70, label=f"Cameras ({label})")
        for p, h in zip(pos, heading):
            ax_scene.arrow(p[0], p[1], h[0], h[1], color=color, head_width=0.08, alpha=0.7)

    ax_scene.set_xlabel("x (m)")
    ax_scene.set_ylabel("y (m)")
    ax_scene.axis("equal")
    ax_scene.set_title("Scene (top-down): ground truth vs. noisy init vs. joint BA")
    ax_scene.legend(fontsize=7, loc="upper left")

    metric_names = ["Pose rot RMS\n(deg)", "Pose pos RMS\n(m)", "Landmark RMS\n(m)", "Reprojection RMS\n(px)"]
    row_names = [name for name, _, _ in rows]
    colors = ["tab:gray", "tab:orange", "tab:purple", "tab:blue"]

    x = np.arange(len(metric_names))
    width = 0.2
    for k, (name, color) in enumerate(zip(row_names, colors)):
        ax_bar.bar(x + (k - 1.5) * width, values[k], width, label=name, color=color)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names)
    ax_bar.set_yscale("log")
    ax_bar.set_title("RMS metrics by method")
    ax_bar.legend(fontsize=7)

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
