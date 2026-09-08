import time
import tracemalloc
import numpy as np


def true_body_rates(t):
    """Smooth, persistently-exciting angular & linear body-frame rate profile.
    Arguments:
        t: time (s)
    Returns:
        omega: body-frame angular velocity (rad/s)
        v: body-frame linear velocity (m/s)
    """
    omega = np.array([
        0.6 * np.sin(0.7 * t),
        0.5 * np.cos(0.4 * t + 0.3),
        0.8 * np.sin(0.25 * t + 1.0),
    ])
    v = np.array([
        1.0 * np.cos(0.3 * t),
        0.5 * np.sin(0.2 * t),
        0.2 * np.sin(0.5 * t),
    ])
    return omega, v


def pair_key(i, j):
    """Canonical (order-independent) key for an unordered keyframe pair."""
    return (i, j) if i < j else (j, i)


def look_at_rotation(cam_pos, target, up_hint=np.array([0.0, 0.0, 1.0])):
    """Builds a camera-to-world rotation whose local +z axis points from
    cam_pos toward target.
    Arguments:
        cam_pos: camera position in world frame (3,)
        target: point the camera should look at, world frame (3,)
        up_hint: approximate "up" direction in world frame (3,)
    Returns:
        R: (3,3) camera-to-world rotation (columns = camera x,y,z axes in world coords)
    """
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(forward, right)
    return np.column_stack([right, cam_up, forward])


def forward_facing_rotation(forward, up_hint=np.array([0.0, 0.0, 1.0])):
    """Builds a camera-to-world rotation whose local +z axis is `forward`
    (e.g. a path's tangent direction) -- analogous to look_at_rotation, but
    taking the forward direction directly instead of deriving it from a
    target point.
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


def rotation_matrix_to_quaternion_xyzw(R):
    """Converts a (3,3) rotation matrix to a manifpy-convention [x,y,z,w]
    quaternion (Shepperd's method) -- used only to build ground-truth/
    aligned manif.SE3 poses from a plain rotation matrix; not a Lie-group
    operation itself.
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


def umeyama_alignment(P_est, P_true):
    """Least-squares similarity (scale, rotation, translation) that best
    maps P_est onto P_true (Umeyama, 1991): P_true ~= s * (R @ P_est.T).T + t.
    Arguments:
        P_est: (N,3) estimated points
        P_true: (N,3) corresponding true points
    Returns:
        s: scale
        R: (3,3) rotation
        t: (3,) translation
    """
    mu_est, mu_true = P_est.mean(axis=0), P_true.mean(axis=0)
    X, Y = P_est - mu_est, P_true - mu_true
    n = P_est.shape[0]
    cov = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_est = (X ** 2).sum() / n
    s = np.trace(np.diag(D) @ S) / var_est
    t = mu_true - s * (R @ mu_est)
    return s, R, t


def landmark_errors(P_true, P_est):
    """Per-landmark position error (m).
    Arguments:
        P_true: (n_landmarks, 3) ground-truth landmark positions
        P_est: (n_landmarks, 3) estimated landmark positions
    Returns:
        err: (n_landmarks,) array of per-landmark position errors (m)
    """
    return np.linalg.norm(P_true - P_est, axis=1)


def measure_performance(fn, *args, n_steps, **kwargs):
    """Runs `fn` once, measuring wall-clock time and peak memory allocated
    during the call (via tracemalloc), and reports both as per-step averages
    so the three approaches (different amounts of work per step) are
    comparable on the same footing.
    Arguments:
        fn: callable to run and measure
        *args, **kwargs: forwarded to fn
        n_steps: number of trajectory steps, used to normalize both metrics
    Returns:
        result: fn(*args, **kwargs)'s return value
        avg_time_per_step: wall-clock time / n_steps (s)
        avg_mem_per_step: peak traced memory / n_steps (bytes)
    """
    tracemalloc.start()
    t_start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed / n_steps, peak_mem / n_steps
