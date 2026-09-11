'''
Perspective-n-Point (PnP): recovering a camera pose from known 3D-2D
correspondences.

This is triangulation's exact inverse (see bundle_adjustment_advanced.py's
triangulate_landmark/refine_landmark_gn, and docs/frontend/triangulation_pnp.md):
triangulation holds camera poses fixed and solves for an unknown 3D point;
PnP holds a set of known 3D points fixed and solves for the unknown camera
pose that observed them. Both follow the same two-step recipe used
throughout this codebase -- a closed-form linear initial guess, then a
handful of Gauss-Newton iterations refining it against the true nonlinear
reprojection error.

linear_pnp_dlt solves the classic Direct Linear Transform (DLT) camera
resectioning problem specialized to *known* intrinsics: a calibrated ray is
parallel to its camera-frame point, so their cross product is zero, giving a
linear (homogeneous) constraint on the flattened world-to-camera
[R|t]. Solved via the smallest right-singular vector, then projected onto
the nearest proper rotation (SVD orthogonalization) to enforce R in SO(3),
with scale and sign fixed from that same decomposition and a
positive-depth check.

refine_pose_gn then runs ordinary Gauss-Newton on the true reprojection
residual, updating the pose via a right-multiplicative SE(3) correction
(T <- T @ se3_exp(delta)) -- the same style of pose-only GN loop used
elsewhere in this codebase (e.g. robot_imu_simulation.py's single-pose
correction), just with a reprojection residual instead of a point-cloud one.

Poses are represented as (4,4) numpy homogeneous transforms (camera-to-world,
matching bundle_adjustment_advanced.py's convention); tangent vectors follow
the [vx,vy,vz,wx,wy,wz] (translation-first) convention used throughout this
codebase.
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt

from lie_utils import skew, so3_exp, se3_exp, rotation_geodesic_error


def camera_project(T, P, K, with_jacobians=False):
    """Pinhole reprojection of a world point through a camera pose:
    p_c = T^-1 @ P (world point into the camera frame), then
    u = fx*x/z + cx, v = fy*y/z + cy. Identical to
    bundle_adjustment_advanced.py's camera_project (duplicated rather than
    imported -- this codebase never imports across sibling scripts, only
    through the shared lie_utils.py).
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
        J_pc_wrt_pose = np.hstack([-np.eye(3), skew(p_c)])
        J_pose = J_proj @ J_pc_wrt_pose
        J_point = J_proj @ R.T
    return pixel, J_pose, J_point


def generate_scene(n_points, K, rng):
    """A random ground-truth camera pose plus n_points random 3D points,
    sampled directly in front of it (camera-frame z in [3, 8]) so every
    point is guaranteed positive-depth by construction.
    Arguments:
        n_points: number of 3D-2D correspondences to generate
        K: (fx, fy, cx, cy) camera intrinsics (unused here, kept for a
           uniform generate_scene(n_points, K, rng) signature alongside the
           rest of this codebase's scene generators)
        rng: numpy random number generator
    Returns:
        T_true: ground-truth camera pose, camera-to-world (4,4)
        P_list: list of n_points world points (3,)
    """
    R_true = so3_exp(rng.normal(0.0, 0.3, 3))
    t_true = rng.normal(0.0, 1.0, 3)
    T_true = np.eye(4)
    T_true[0:3, 0:3] = R_true
    T_true[0:3, 3] = t_true

    P_list = []
    for _ in range(n_points):
        p_c = np.array([rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(3.0, 8.0)])
        P_list.append(R_true @ p_c + t_true)
    return T_true, P_list


def linear_pnp_dlt(P_list, z_list, K):
    """Closed-form linear initial guess for the camera pose: Direct Linear
    Transform (DLT) camera resectioning, specialized to known intrinsics.
    A calibrated ray d_i = [(u-cx)/fx, (v-cy)/fy, 1] is parallel to its
    camera-frame point R_cw@P_i + t_cw, so d_i x (R_cw@P_i + t_cw) = 0 is
    linear and homogeneous in the 12 flattened entries of [R_cw | t_cw]
    (world-to-camera). Stacking two independent rows of that cross-product
    constraint per correspondence gives an over-determined homogeneous
    system, solved via the smallest right-singular vector (docs/frontend/
    triangulation_pnp.md Section 3 has the full derivation).

    The raw 3x3 block recovered this way is only a *scaled, possibly
    reflected* rotation, so it's projected onto the nearest proper rotation
    via SVD orthogonalization; scale comes from that same decomposition's
    singular values, and sign is fixed by requiring the (scale-invariant)
    raw camera-frame depths to come out positive -- PnP's analogue of
    bundle_adjustment_advanced.py's passes_cheirality check.
    Arguments:
        P_list: list of world points (3,)
        z_list: list of corresponding observed pixels (2,)
        K: (fx, fy, cx, cy) camera intrinsics
    Returns:
        T0: initial camera pose guess, camera-to-world (4,4)
    """
    fx, fy, cx, cy = K
    n = len(P_list)
    A = np.zeros((2 * n, 12))
    for i, (P, z) in enumerate(zip(P_list, z_list)):
        d = np.array([(z[0] - cx) / fx, (z[1] - cy) / fy, 1.0])
        S = skew(d)[0:2, :]  # 2x3; always rank 2 since d's 3rd component is fixed at 1
        M = np.zeros((3, 12))
        for r in range(3):
            M[r, 3 * r:3 * r + 3] = P
            M[r, 9 + r] = 1.0
        A[2 * i:2 * i + 2, :] = S @ M

    _, _, Vt = np.linalg.svd(A)
    x = Vt[-1]
    R_raw = x[0:9].reshape(3, 3)
    t_raw = x[9:12]

    # Sign is ambiguous (x and -x both solve the homogeneous system); resolve
    # it via the scale-invariant sign of the raw camera-frame depth.
    depths_raw = np.array([(R_raw @ P + t_raw)[2] for P in P_list])
    if np.median(depths_raw) < 0.0:
        R_raw, t_raw = -R_raw, -t_raw

    U, singular_values, Vt2 = np.linalg.svd(R_raw)
    R_cw = U @ Vt2
    if np.linalg.det(R_cw) < 0.0:
        Vt2[-1, :] *= -1.0
        R_cw = U @ Vt2
    scale = np.mean(singular_values)
    t_cw = t_raw / scale

    T0 = np.eye(4)
    T0[0:3, 0:3] = R_cw.T
    T0[0:3, 3] = -R_cw.T @ t_cw
    return T0


def refine_pose_gn(T0, P_list, z_list, K, gn_tol, gn_max_iters):
    """A handful of Gauss-Newton iterations refining the camera pose against
    the true reprojection error, seeded from linear_pnp_dlt's closed-form
    guess. Same 6x6 normal-equations solve as pose_graph.py's per-node
    blocks, applied to a single pose against a reprojection residual instead
    of a relative-pose residual, updating via a right-multiplicative SE(3)
    correction (T <- T @ se3_exp(delta)).
    Arguments:
        T0: initial camera pose guess, camera-to-world (4,4)
        P_list: list of world points (3,)
        z_list: list of corresponding observed pixels (2,)
        K: (fx, fy, cx, cy) camera intrinsics
        gn_tol: convergence tolerance on the correction step norm
        gn_max_iters: maximum number of iterations
    Returns:
        T: refined camera pose, camera-to-world (4,4)
    """
    T = T0.copy()
    for _ in range(gn_max_iters):
        H = np.zeros((6, 6))
        g = np.zeros(6)
        for P, z in zip(P_list, z_list):
            pred, J_pose, _ = camera_project(T, P, K, with_jacobians=True)
            r = z - pred
            H += J_pose.T @ J_pose
            g += J_pose.T @ r
        delta = np.linalg.solve(H + np.eye(6) * 1e-9, g)
        T = T @ se3_exp(delta)
        if np.linalg.norm(delta) < gn_tol:
            break
    return T


def pose_error(T_est, T_true):
    """Rotation geodesic error (rad) and position error (m) between two poses.
    Arguments:
        T_est: estimated pose (4,4)
        T_true: true pose (4,4)
    Returns:
        rot_err: rotation error (rad)
        pos_err: position error (m)
    """
    rot_err = rotation_geodesic_error(T_true[0:3, 0:3], T_est[0:3, 0:3])
    pos_err = np.linalg.norm(T_true[0:3, 3] - T_est[0:3, 3])
    return rot_err, pos_err


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--n-points", type=int, default=20, help="Number of 3D-2D correspondences")
    parser.add_argument("--image-width", type=int, default=640, help="Image width in pixels (sets cx)")
    parser.add_argument("--image-height", type=int, default=480, help="Image height in pixels (sets cy)")
    parser.add_argument("--focal-length", type=float, default=800.0, help="Shared fx=fy focal length in pixels")
    parser.add_argument("--pixel-noise-std", type=float, default=1.0, help="Std-dev of Gaussian pixel noise (px)")

    parser.add_argument("--gn-tol", type=float, default=1e-8, help="Gauss-Newton convergence tolerance")
    parser.add_argument("--gn-max-iters", type=int, default=20, help="Maximum Gauss-Newton iterations")

    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    K = (args.focal_length, args.focal_length, args.image_width / 2.0, args.image_height / 2.0)

    T_true, P_list = generate_scene(args.n_points, K, rng)
    z_list = [camera_project(T_true, P, K)[0] + rng.normal(0.0, args.pixel_noise_std, 2) for P in P_list]

    print(f"Generated {args.n_points} 3D-2D correspondences, pixel noise std={args.pixel_noise_std} px.")

    T_dlt = linear_pnp_dlt(P_list, z_list, K)
    print("Solved linear PnP (DLT) initial guess.")

    T_gn = refine_pose_gn(T_dlt, P_list, z_list, K, args.gn_tol, args.gn_max_iters)
    print("Refined pose with Gauss-Newton.")

    rot_err_dlt, pos_err_dlt = pose_error(T_dlt, T_true)
    rot_err_gn, pos_err_gn = pose_error(T_gn, T_true)

    print("\nPose error vs. ground truth:")
    for name, rot_err, pos_err in [
        ("Linear DLT only", rot_err_dlt, pos_err_dlt),
        ("Gauss-Newton refined", rot_err_gn, pos_err_gn),
    ]:
        print(f"  {name:<24s} rot={np.degrees(rot_err):7.3f} deg, pos={pos_err:7.4f} m")

    P_arr = np.array(P_list)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P_arr[:, 0], P_arr[:, 1], P_arr[:, 2], color="tab:gray", label="3D points", alpha=0.6)

    axis_len = 0.5
    for T, color, label in [
        (T_true, "black", "Ground truth"),
        (T_dlt, "tab:orange", "Linear DLT"),
        (T_gn, "tab:blue", "Gauss-Newton refined"),
    ]:
        pos = T[0:3, 3]
        forward = T[0:3, 0:3] @ np.array([0.0, 0.0, 1.0]) * axis_len
        ax.scatter(*pos, color=color, marker="x", s=80, label=label)
        ax.plot(*zip(pos, pos + forward), color=color)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("PnP: recovering a camera pose from known 3D-2D correspondences")
    ax.legend()

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
