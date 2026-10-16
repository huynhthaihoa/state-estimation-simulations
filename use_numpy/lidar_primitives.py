'''
LiDAR scan registration with geometric primitives (planes + cylinders),
against the two classic ICP variants, on simulated scans of a corridor.

The scene, the ray-cast LiDAR, the per-point normals and the primitive
extraction (sequential RANSAC, planes vs. cylinders) live in the shared,
library-agnostic `lidar_utils.py` at the repo root. This script adds the
SE(3) part: residuals, their Jacobians, and one Gauss-Newton loop.

Registration estimates T, the pose of the source scan's sensor frame in the
target scan's sensor frame (target point = T @ source point). The truth is
T_true = inv(T_target) @ T_source. Three residuals, re-associated at every
iteration from the current estimate (nothing is handed over as known):

  - point-to-point (ICP, Besl & McKay 1992): each source point minus its
    nearest target point, 3 residuals per pair.
  - point-to-plane (Chen & Medioni 1992): the same nearest-neighbor pairs,
    but only the distance along the target point's PCA normal.
  - primitive: source primitives are matched to target primitives (planes by
    normal and offset, cylinders by axis direction, axis position and
    radius, after mapping the source primitive through T); the residuals are
    the distances of the source primitive's inlier points to the matched
    target surface.

Pairs or primitives farther apart than the gates in RegistrationConfig are
dropped, the same idea for every method. Poses are (4,4) homogeneous
transforms, perturbed on the right, T <- T Exp(xi), with the translation-first
[v, omega] tangent convention of lie_utils.py.

Experiments (run with --experiment):
  1  extraction accuracy (lidar_utils.run_extraction_experiment)
  2  convergence basin: success rate and final error of each method vs. the
     size of the initial-guess error, in the corridor with pillars
  3  degeneracy: eigen-structure of each method's Gauss-Newton Hessian in the
     plain corridor (planes only) vs. the corridor with pillars, and the
     error along the weakest direction
  4  planes as landmarks: along a trajectory through an L-shaped corridor,
     chained scan-to-scan odometry (primitive registration) vs. a factor
     graph of poses + world plane landmarks (odometry factors + plane
     measurement factors), solved by Levenberg-Marquardt
'''

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lie_utils import skew, se3_exp, se3_log, se3_inv, se3_adjoint, compute_se3_inv_right_jacobian
from lidar_utils import (METHODS, RELATIVE_MOTION, ExtractionConfig, LidarConfig, RegistrationConfig,  # noqa: F401
                         Scan, associate_planes, print_basin, print_degeneracy, print_extraction, print_landmarks,
                         corridor_scene, inlier_threshold, l_corridor_scene, l_corridor_trajectory,
                         match_primitives, pair_gate, plane_fit_information, prepare_scan, random_perturbation,
                         random_sensor_pose, run_extraction_experiment, s2_basis, s2_exp, simulate_scan)
from utils import umeyama_alignment

def make_pose(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def transform_points(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def point_action_jacobian(T, P):
    """d(T Exp(xi) p)/dxi at xi = 0 for every p: [R, -R skew(p)], shape (N,3,6)."""
    R = T[:3, :3]
    S = np.zeros((len(P), 3, 3))                     # skew(p) for every p, vectorized
    S[:, 0, 1], S[:, 0, 2], S[:, 1, 2] = -P[:, 2], P[:, 1], -P[:, 0]
    S -= S.transpose(0, 2, 1)
    J = np.empty((len(P), 3, 6))
    J[:, :, :3] = R
    J[:, :, 3:] = -np.einsum("ij,njk->nik", R, S)
    return J


# ---------------------------------------------------------------------------
# Residuals (each returns e (m,), J (m,6))
# ---------------------------------------------------------------------------

def nearest_pairs(T, src, tgt, cfg, it):
    """Source indices and their nearest target indices within the gate."""
    dist, idx = tgt.tree.query(transform_points(T, src.points), distance_upper_bound=pair_gate(cfg, it))
    keep = np.isfinite(dist)
    return np.flatnonzero(keep), idx[keep]


def residuals_point_to_point(T, src, tgt, cfg, it=0):
    i, j = nearest_pairs(T, src, tgt, cfg, it)
    e = (transform_points(T, src.points[i]) - tgt.points[j]).ravel()
    return e, point_action_jacobian(T, src.points[i]).reshape(-1, 6)


def residuals_point_to_plane(T, src, tgt, cfg, it=0):
    i, j = nearest_pairs(T, src, tgt, cfg, it)
    n = tgt.normals[j]
    e = np.sum(n * (transform_points(T, src.points[i]) - tgt.points[j]), axis=1)
    return e, np.einsum("ni,nij->nj", n, point_action_jacobian(T, src.points[i]))


def plane_point_residuals(T, P, n, d):
    """Distances of the source points P (mapped by T) to the target plane (n, d)."""
    return transform_points(T, P) @ n + d, point_action_jacobian(T, P).transpose(0, 2, 1) @ n


def cylinder_point_residuals(T, P, a, c, r):
    """Distances of the source points P (mapped by T) to the target cylinder (a, c, r)."""
    w = transform_points(T, P) - c
    q = w - (w @ a)[:, None] * a
    rho = np.linalg.norm(q, axis=1)
    e_hat = q / rho[:, None]
    return rho - r, np.einsum("ni,nij->nj", e_hat, point_action_jacobian(T, P))


def residuals_primitive(T, src, tgt, cfg, it=0):
    es, Js = [np.zeros(0)], [np.zeros((0, 6))]
    for kind, i, j in match_primitives(T[:3, :3], T[:3, 3], src, tgt, cfg):
        if kind == "plane":
            P = src.points[src.planes[i]["inliers"]]
            e, J = plane_point_residuals(T, P, tgt.planes[j]["n"], tgt.planes[j]["d"])
        else:
            inl = src.cylinders[i]["inliers"]
            P, ct = src.points[inl], tgt.cylinders[j]
            # side check: the distance to a cylinder does not know which side was scanned, so a scanned arc
            # shifted past the axis also fits the far side (a mirrored local minimum); keep a point only if
            # its normal, mapped by T, points outward from the target axis, as on the side a sensor sees
            w = transform_points(T, P) - ct["c"]
            P = P[np.sum((src.normals[inl] @ T[:3, :3].T) * (w - (w @ ct["a"])[:, None] * ct["a"]), axis=1) > 0]
            e, J = cylinder_point_residuals(T, P, ct["a"], ct["c"], ct["r"])
        es.append(e)
        Js.append(J)
    return np.concatenate(es), np.vstack(Js)


RESIDUALS = {
    "point_to_point": residuals_point_to_point,
    "point_to_plane": residuals_point_to_plane,
    "primitive": residuals_primitive,
}


# ---------------------------------------------------------------------------
# Gauss-Newton
# ---------------------------------------------------------------------------

def register(method, T0, src, tgt, cfg, history=None):
    """Gauss-Newton on SE(3), re-associating at every iteration. The step is
    the minimum-norm least-squares solution with singular values of J below
    cfg.rank_rcond (relative) treated as zero, so a direction the residuals
    do not constrain at all (H exactly singular up to round-off) is not
    moved. A merely weak direction is kept: this is numerical rank
    detection, not Zhang-Kaess-Singh solution remapping.
    Arguments:
        method: one of METHODS
        T0: initial guess (4,4)
        src, tgt: Scan
        cfg: RegistrationConfig
        history: optional list; the estimate after every iteration is appended to it
    Returns:
        T: estimate (4,4)
        info: dict with 'iters', 'converged', 'H' (6x6 J^T J at the final
              estimate), 'n_residuals'
    """
    T = T0.copy()
    converged = False
    for it in range(1, cfg.max_iters + 1):
        e, J = RESIDUALS[method](T, src, tgt, cfg, it - 1)
        if len(e) < 6:
            break
        delta = np.linalg.lstsq(J, -e, rcond=cfg.rank_rcond)[0]
        T = T @ se3_exp(delta)
        if history is not None:
            history.append(T)
        gate_done = method == "primitive" or pair_gate(cfg, it - 1) == cfg.min_pair_dist
        if np.linalg.norm(delta) < cfg.tol and gate_done:
            converged = True
            break
    e, J = RESIDUALS[method](T, src, tgt, cfg, it)
    return T, {"iters": it, "converged": converged, "H": J.T @ J, "n_residuals": len(e)}


def pose_error(T_est, T_true):
    """(translation error in m, rotation error in rad) of T_est vs. T_true."""
    xi = se3_log(se3_inv(T_true) @ T_est)
    return np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]), np.linalg.norm(xi[3:])


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def make_scan_pair(scene, seed, lcfg, ecfg, x_range=(-8.0, 8.0)):
    """Target and source scans of `scene`; the source sensor sits at
    RELATIVE_MOTION from the target sensor.
    Returns:
        src, tgt (Scan), T_true (4,4), T_target (4,4, world pose of the target sensor)
    """
    rng = np.random.default_rng(seed)
    T_tgt = make_pose(*random_sensor_pose(rng, x_range=x_range))
    T_true = se3_exp(RELATIVE_MOTION)
    T_src = T_tgt @ T_true
    scans = []
    for T in (T_tgt, T_src):
        pts, _ = simulate_scan(scene, T[:3, :3], T[:3, 3], lcfg, rng)
        scans.append(prepare_scan(pts, ecfg, rng))
    return scans[1], scans[0], T_true, T_tgt


def run_basin_experiment(levels, n_pairs, n_starts, seed, noise=0.01, n_azimuth=360,
                         success_trans=0.05, success_rot_deg=1.0, cfg=RegistrationConfig()):
    """Experiment 2: each method from initial guesses T_true Exp(perturbation)
    of growing size, in the corridor with pillars.
    Arguments:
        levels: list of (translation m, rotation deg) perturbation sizes
        n_pairs: scan pairs (different corridor poses)
        n_starts: random perturbation directions per pair and level
        success_trans, success_rot_deg: success = final error below both
    Returns:
        dict method -> dict of arrays over levels: 'success' (rate),
        'trans_mm' / 'rot_deg' (median final error over successes), 'iters' (median)
    """
    scene = corridor_scene(pillars=True)
    lcfg = LidarConfig(n_azimuth=n_azimuth, range_noise_std=noise)
    ecfg = ExtractionConfig(inlier_dist=inlier_threshold(noise))
    rows = {m: [[] for _ in levels] for m in METHODS}
    for p in range(n_pairs):
        src, tgt, T_true, _ = make_scan_pair(scene, [seed, p], lcfg, ecfg)
        for k, (tr, rot_deg) in enumerate(levels):
            rng = np.random.default_rng([seed, p, k])
            for _ in range(n_starts):
                T0 = T_true @ se3_exp(random_perturbation(rng, tr, np.deg2rad(rot_deg)))
                for m in METHODS:
                    T, info = register(m, T0, src, tgt, cfg)
                    et, er = pose_error(T, T_true)
                    rows[m][k].append((et, er, info["iters"]))
    out = {}
    for m in METHODS:
        res = {"success": [], "trans_mm": [], "rot_deg": [], "iters": []}
        for vals in rows[m]:
            v = np.asarray(vals)
            ok = (v[:, 0] < success_trans) & (v[:, 1] < np.deg2rad(success_rot_deg))
            res["success"].append(ok.mean())
            res["trans_mm"].append(1e3 * np.median(v[ok, 0]) if ok.any() else np.nan)
            res["rot_deg"].append(np.rad2deg(np.median(v[ok, 1])) if ok.any() else np.nan)
            res["iters"].append(np.median(v[:, 2]))
        out[m] = {k: np.asarray(v) for k, v in res.items()}
    return out


def run_degeneracy_experiment(n_pairs, seed, noise=0.01, n_azimuth=360, start_trans=0.2, start_rot_deg=2.0,
                              cfg=RegistrationConfig()):
    """Experiment 3: is the registration problem degenerate, and what does
    Gauss-Newton do about it? For each scene (plain corridor, corridor with
    pillars) and method:
      - diagnosis, from H = J^T J evaluated at the TRUE pose (with the final,
        tightest gate), so it describes the problem rather than wherever the
        solver ended up: the eigenvalue ratio, whether the weakest
        eigenvector is a translation along the corridor axis, and the
        predicted standard deviations noise * sqrt(g^T H^+ g) of the
        translation along the axis and across it (horizontal, in the
        target frame);
      - outcome, from registration started at a small random error: the
        signed error along the axis and the total errors.
    Returns:
        dict scene -> method -> dict of per-pair arrays: 'eig_ratio',
        'axis_alignment', 'axis_sigma_mm', 'lateral_sigma_mm',
        'axis_err_mm', 'trans_mm', 'rot_deg'
    """
    lcfg = LidarConfig(n_azimuth=n_azimuth, range_noise_std=noise)
    ecfg = ExtractionConfig(inlier_dist=inlier_threshold(noise))
    keys = ("eig_ratio", "axis_alignment", "axis_sigma_mm", "lateral_sigma_mm", "axis_err_mm", "trans_mm", "rot_deg")
    out = {}
    for name, pillars in (("corridor", False), ("corridor_pillars", True)):
        scene = corridor_scene(pillars=pillars)
        rows = {m: {k: [] for k in keys} for m in METHODS}
        for p in range(n_pairs):
            src, tgt, T_true, T_tgt = make_scan_pair(scene, [seed, p], lcfg, ecfg)
            axis = T_tgt[:3, :3].T @ np.array([1.0, 0.0, 0.0])        # corridor axis, target frame
            lateral = T_tgt[:3, :3].T @ np.array([0.0, 1.0, 0.0])
            rng = np.random.default_rng([seed, p, 99])
            T0 = T_true @ se3_exp(random_perturbation(rng, start_trans, np.deg2rad(start_rot_deg)))
            R_true = T_true[:3, :3]
            for m in METHODS:
                _, J = RESIDUALS[m](T_true, src, tgt, cfg, cfg.max_iters)
                H = J.T @ J
                evals, evecs = np.linalg.eigh(H)
                # a right perturbation's translation is in the source frame: R_true maps it to the target frame
                weakest_t = R_true @ evecs[:3, 0]
                H_pinv = np.linalg.pinv(H, rcond=1e-12, hermitian=True)
                sig = [1e3 * noise * np.sqrt(max(g @ H_pinv @ g, 0.0))
                       for g in (np.concatenate([R_true.T @ d, np.zeros(3)]) for d in (axis, lateral))]
                T, _ = register(m, T0, src, tgt, cfg)
                et, er = pose_error(T, T_true)
                r = rows[m]
                r["eig_ratio"].append(evals[0] / evals[-1])
                r["axis_alignment"].append(abs(weakest_t @ axis) / max(np.linalg.norm(evecs[:3, 0]), 1e-12))
                r["axis_sigma_mm"].append(sig[0])
                r["lateral_sigma_mm"].append(sig[1])
                r["axis_err_mm"].append(1e3 * (T[:3, 3] - T_true[:3, 3]) @ axis)
                r["trans_mm"].append(1e3 * et)
                r["rot_deg"].append(np.rad2deg(er))
        out[name] = {m: {k: np.asarray(v) for k, v in rows[m].items()} for m in METHODS}
    return out


# ---------------------------------------------------------------------------
# Planes as landmarks (experiment 4)
# ---------------------------------------------------------------------------

def plane_factor(T, n_w, d_w, n_s, d_s):
    """Residual and Jacobians of one plane measurement (n_s, d_s), extracted
    in the sensor frame of pose T, against the world landmark (n_w, d_w).
    The landmark predicted in the sensor frame is n = R^T n_w, d = d_w + n_w . t;
    the residual is [B(n_s)^T n (2), d - d_s (1)].
    Returns:
        e (3,), J_pose (3,6) wrt T Exp(xi), J_lm (3,3) wrt (s2_exp(n_w, dn), d_w + dd)
    """
    R, t = T[:3, :3], T[:3, 3]
    n = R.T @ n_w
    Bs, Bw = s2_basis(n_s), s2_basis(n_w)
    e = np.concatenate([Bs.T @ n, [d_w + n_w @ t - d_s]])
    J_pose = np.zeros((3, 6))
    J_pose[:2, 3:] = Bs.T @ skew(n)        # R^T -> (I - [dtheta]x) R^T: dn = [n]x dtheta
    J_pose[2, :3] = n                      # t -> t + R dv: dd = (R^T n_w) . dv
    J_lm = np.zeros((3, 3))
    J_lm[:2, :2] = Bs.T @ R.T @ Bw
    J_lm[2, :2] = t @ Bw
    J_lm[2, 2] = 1.0
    return e, J_pose, J_lm


def between_factor(Xi, Xj, Z):
    """Odometry residual Log(Z^-1 Xi^-1 Xj) and its Jacobians wrt Xi and Xj
    (right perturbations), as in pose_graph.py."""
    e = se3_log(se3_inv(Xi @ Z) @ Xj)
    return e, -compute_se3_inv_right_jacobian(-e) @ se3_adjoint(se3_inv(Z)), compute_se3_inv_right_jacobian(e)


def run_odometry(scans, T_start, init_rel, noise, cfg=RegistrationConfig()):
    """Chained scan-to-scan primitive registration.
    Arguments:
        scans: list of Scan along the trajectory
        T_start: world pose of the first sensor (the gauge, assumed known)
        init_rel: list of initial guesses for every relative pose (4,4)
        noise: range noise std, for the odometry information H / noise^2
    Returns:
        poses: list of world poses; Z: list of relative poses; Omega: list of 6x6 informations
    """
    poses, Z, Omega = [T_start], [], []
    for k in range(len(scans) - 1):
        T, info = register("primitive", init_rel[k], scans[k + 1], scans[k], cfg)
        Z.append(T)
        Omega.append(info["H"] / noise ** 2)
        poses.append(poses[-1] @ T)
    return poses, Z, Omega


def optimize_plane_graph(poses0, Z, Omega, scans, landmarks0, observations, noise, damping=1e-3, tol=1e-8,
                         max_iters=50):
    """Batch Levenberg-Marquardt over every pose and plane landmark: pose 0
    anchored (gauge), one between factor per odometry step (information
    Omega), one plane factor per observation (information from the fit).
    Accept/reject damping as in pose_graph.py.
    Returns:
        poses, landmarks (lists), n_iters
    """
    n_p, n_l = len(poses0), len(landmarks0)
    dim = 6 * n_p + 3 * n_l
    plane_info = {(k, i): plane_fit_information(scans[k].points[scans[k].planes[i]["inliers"]],
                                                scans[k].planes[i]["n"], noise)
                  for k, i, _ in observations}

    def cost(poses, lms):
        c = 0.0
        for k, (Zk, Ok) in enumerate(zip(Z, Omega)):
            e = se3_log(se3_inv(poses[k] @ Zk) @ poses[k + 1])
            c += e @ Ok @ e
        for k, i, l in observations:
            pl = scans[k].planes[i]
            e, _, _ = plane_factor(poses[k], lms[l][0], lms[l][1], pl["n"], pl["d"])
            c += e @ plane_info[(k, i)] @ e
        return c

    poses, lms = list(poses0), [list(lm) for lm in landmarks0]
    cur, lam = cost(poses, lms), damping
    for it in range(1, max_iters + 1):
        H = np.zeros((dim, dim))
        g = np.zeros(dim)
        H[:6, :6] += 1e12 * np.eye(6)                    # anchor pose 0
        for k, (Zk, Ok) in enumerate(zip(Z, Omega)):
            e, Ji, Jj = between_factor(poses[k], poses[k + 1], Zk)
            J = np.hstack([Ji, Jj])
            sl = slice(6 * k, 6 * k + 12)
            H[sl, sl] += J.T @ Ok @ J
            g[sl] -= J.T @ Ok @ e
        for k, i, l in observations:
            pl = scans[k].planes[i]
            e, Jp, Jl = plane_factor(poses[k], lms[l][0], lms[l][1], pl["n"], pl["d"])
            Lam = plane_info[(k, i)]
            ip, il = 6 * k, 6 * n_p + 3 * l
            H[ip:ip + 6, ip:ip + 6] += Jp.T @ Lam @ Jp
            H[il:il + 3, il:il + 3] += Jl.T @ Lam @ Jl
            H[ip:ip + 6, il:il + 3] += Jp.T @ Lam @ Jl
            H[il:il + 3, ip:ip + 6] += Jl.T @ Lam @ Jp
            g[ip:ip + 6] -= Jp.T @ Lam @ e
            g[il:il + 3] -= Jl.T @ Lam @ e
        diag = np.maximum(np.diag(H), 1e-12)
        accepted = False
        for _ in range(10):
            delta = np.linalg.solve(H + lam * np.diag(diag), g)
            trial_p = [poses[k] @ se3_exp(delta[6 * k:6 * k + 6]) for k in range(n_p)]
            trial_l = [[s2_exp(n, delta[6 * n_p + 3 * l:6 * n_p + 3 * l + 2]), d + delta[6 * n_p + 3 * l + 2]]
                       for l, (n, d) in enumerate(lms)]
            trial = cost(trial_p, trial_l)
            if trial < cur:
                poses, lms, cur = trial_p, trial_l, trial
                lam = max(lam * 0.5, 1e-9)
                accepted = True
                break
            lam *= 4.0
        if not accepted or np.linalg.norm(delta) < tol:
            break
    return poses, lms, it


def trajectory_errors(poses_est, poses_true):
    """ATE RMSE (m) after the best rigid (no scale) alignment of the
    positions, and the unaligned end-point position (m) and rotation (rad)
    errors (the start pose is known, so the end-point error is the drift)."""
    P_est = np.array([T[:3, 3] for T in poses_est])
    P_true = np.array([T[:3, 3] for T in poses_true])
    _, R, _ = umeyama_alignment(P_est, P_true)           # the rotation does not depend on the scale
    t = P_true.mean(axis=0) - R @ P_est.mean(axis=0)
    ate = np.sqrt(np.mean(np.sum((P_est @ R.T + t - P_true) ** 2, axis=1)))
    end_t, end_r = pose_error(poses_est[-1], poses_true[-1])
    return ate, end_t, end_r


def run_landmark_experiment(n_runs, seed, noise=0.01, n_azimuth=360, init_trans=0.1, init_rot_deg=2.0):
    """Experiment 4: n_runs noise realizations of the L-corridor trajectory.
    Each odometry step starts from the true relative pose perturbed by
    (init_trans, init_rot_deg) in a random direction, standing in for an
    IMU / wheel-odometry prior (the corner turns ~38 deg per step, more
    than the primitive matching gate tolerates as an initial error).
    Returns:
        dict 'odometry' / 'graph' -> dict of per-run arrays 'ate_mm',
        'end_mm', 'end_deg'; plus 'n_landmarks' (per run; the scene has 6
        distinct planes) and 'n_scans'
    """
    scene = l_corridor_scene()
    traj = [make_pose(R, t) for R, t in l_corridor_trajectory()]
    lcfg = LidarConfig(n_azimuth=n_azimuth, range_noise_std=noise)
    ecfg = ExtractionConfig(inlier_dist=inlier_threshold(noise))
    out = {m: {"ate_mm": [], "end_mm": [], "end_deg": []} for m in ("odometry", "graph")}
    out["n_landmarks"] = []
    for run in range(n_runs):
        rng = np.random.default_rng([seed, run])
        scans = [prepare_scan(simulate_scan(scene, T[:3, :3], T[:3, 3], lcfg, rng)[0], ecfg, rng) for T in traj]
        init_rel = [se3_inv(traj[k]) @ traj[k + 1] @ se3_exp(random_perturbation(rng, init_trans,
                                                                                 np.deg2rad(init_rot_deg)))
                    for k in range(len(traj) - 1)]
        odo, Z, Omega = run_odometry(scans, traj[0], init_rel, noise)
        landmarks, observations = associate_planes(scans, [T[:3, :3] for T in odo], [T[:3, 3] for T in odo])
        graph, _, _ = optimize_plane_graph(odo, Z, Omega, scans, landmarks, observations, noise)
        for name, est in (("odometry", odo), ("graph", graph)):
            ate, end_t, end_r = trajectory_errors(est, traj)
            out[name]["ate_mm"].append(1e3 * ate)
            out[name]["end_mm"].append(1e3 * end_t)
            out[name]["end_deg"].append(np.rad2deg(end_r))
        out["n_landmarks"].append(len(landmarks))
    for name in ("odometry", "graph"):
        out[name] = {k: np.asarray(v) for k, v in out[name].items()}
    out["n_landmarks"] = np.asarray(out["n_landmarks"])
    out["n_scans"] = len(traj)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=("1", "2", "3", "4", "all"), default="all")
    parser.add_argument("--n-trials", type=int, default=10, help="Experiment 1: sensor poses per cell")
    parser.add_argument("--n-pairs", type=int, default=10, help="Experiments 2-3: scan pairs")
    parser.add_argument("--n-starts", type=int, default=5, help="Experiment 2: initial guesses per pair and level")
    parser.add_argument("--n-runs", type=int, default=5, help="Experiment 4: noise realizations of the trajectory")
    parser.add_argument("--noise", type=float, default=0.01, help="Experiments 2-4: range noise std (m)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args()

    if args.experiment in ("1", "all"):
        noise_levels, azimuth_counts = [0.005, 0.01, 0.02, 0.04], [180, 360, 720]
        print("Experiment 1: extraction accuracy")
        print_extraction(run_extraction_experiment(corridor_scene(pillars=True), noise_levels, azimuth_counts,
                                                   args.n_trials, args.seed, min_points=150),
                         noise_levels, azimuth_counts)
    if args.experiment in ("2", "all"):
        levels = [(0.1, 2), (0.25, 5), (0.5, 10), (1.0, 20), (2.0, 30)]
        print("\nExperiment 2: convergence basin (corridor with pillars)")
        print_basin(run_basin_experiment(levels, args.n_pairs, args.n_starts, args.seed, args.noise), levels)
    if args.experiment in ("3", "all"):
        print("\nExperiment 3: degeneracy")
        print_degeneracy(run_degeneracy_experiment(args.n_pairs, args.seed, args.noise))
    if args.experiment in ("4", "all"):
        print("\nExperiment 4: planes as landmarks (L-shaped corridor)")
        print_landmarks(run_landmark_experiment(args.n_runs, args.seed, args.noise))


if __name__ == "__main__":
    main()
