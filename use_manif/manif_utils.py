import numpy as np
import manifpy as manif


def camera_project(T, P, K, with_jacobians=False):
    """Pinhole reprojection of a world point through a camera pose:
    p_c = T^-1 @ P (world point into the camera frame), then
    u = fx*x/z + cx, v = fy*y/z + cy. Pose Jacobian obtained by chaining
    manifpy's own `inverse`/`act` Jacobian out-parameters.
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


def pose_errors(T_true_list, T_est_list):
    """Rotation geodesic error (deg) and position error (m), per pose.
    Arguments:
        T_true_list: list of true poses (manif.SE3)
        T_est_list: list of estimated poses (manif.SE3)
    Returns:
        rot_err: (N,) array of rotation errors (deg), one per pose
        pos_err: (N,) array of position errors (m), one per pose
    """
    n = len(T_true_list)
    rot_err, pos_err = np.zeros(n), np.zeros(n)
    for k in range(n):
        R_true = manif.SO3(T_true_list[k].coeffs()[3:7])
        R_est = manif.SO3(T_est_list[k].coeffs()[3:7])
        rot_err[k] = np.degrees(np.linalg.norm(R_est.rminus(R_true).coeffs()))
        pos_err[k] = np.linalg.norm(T_true_list[k].translation() - T_est_list[k].translation())
    return rot_err, pos_err
