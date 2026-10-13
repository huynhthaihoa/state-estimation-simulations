import numpy as np

from lie_utils import skew, rotation_geodesic_error


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


def pose_errors(T_true_list, T_est_list):
    """Rotation geodesic error (deg) and position error (m), per pose.
    Arguments:
        T_true_list: list of true poses (4,4)
        T_est_list: list of estimated poses (4,4)
    Returns:
        rot_err: (N,) array of rotation errors (deg), one per pose
        pos_err: (N,) array of position errors (m), one per pose
    """
    n = len(T_true_list)
    rot_err, pos_err = np.zeros(n), np.zeros(n)
    for k in range(n):
        rot_err[k] = np.degrees(rotation_geodesic_error(T_true_list[k][0:3, 0:3], T_est_list[k][0:3, 0:3]))
        pos_err[k] = np.linalg.norm(T_true_list[k][0:3, 3] - T_est_list[k][0:3, 3])
    return rot_err, pos_err
