'''
LiDAR scan registration with geometric primitives (planes + cylinders) using
manifpy: the same three registration methods, Gauss-Newton loop, plane-landmark
factor graph and experiments as use_numpy/lidar_primitives.py, with poses held
as manif.SE3 and every group operation (retraction, Log, composition) and the
factor Jacobians taken from manif instead of lie_utils.py.

Everything that never touches the group (scene, LiDAR, normals, primitive
extraction, primitive matching, plane-fit information, landmark association,
the printing) is the shared, library-agnostic lidar_utils.py at the repo
root, the same copy the use_numpy script imports.

One deliberate shortcut: the point residuals act on ~10k points per
iteration, so they read T.rotation()/T.translation() once and apply the
point-action Jacobian [R, -R skew(p)] vectorized, instead of calling
T.act(p, J) per point (a test checks the two agree). The plane and odometry
factors, a handful per pose, use manif's Jacobian out-parameters directly.

Experiments (run with --experiment): 1 extraction accuracy, 2 convergence
basin, 3 degeneracy, 4 planes as landmarks; see the use_numpy script.
'''

import argparse
import os
import sys

import numpy as np
import manifpy as manif
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lidar_utils import (METHODS, RELATIVE_MOTION, ExtractionConfig, LidarConfig, RegistrationConfig,  # noqa: F401
                         Scan, associate_planes, print_basin, print_degeneracy, print_extraction, print_landmarks,
                         corridor_scene, inlier_threshold, l_corridor_scene, l_corridor_trajectory,
                         match_primitives, pair_gate, plane_fit_information, prepare_scan, random_perturbation,
                         random_sensor_pose, run_extraction_experiment, s2_basis, s2_exp, simulate_scan)
from utils import umeyama_alignment


def make_pose(R, t):
    """manif.SE3 from a rotation matrix and a translation (manif has no
    rotation-matrix constructor, so the quaternion goes through scipy)."""
    return manif.SE3(np.concatenate([t, Rotation.from_matrix(R).as_quat()]))


def transform_points(T, P):
    return P @ T.rotation().T + T.translation()


def point_action_jacobian(T, P):
    """d(T Exp(xi) p)/dxi at xi = 0 for every p, shape (N,3,6): manif's
    T.act(p, J_self) Jacobian [R, -R skew(p)], built vectorized."""
    R = T.rotation()
    S = np.zeros((len(P), 3, 3))
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
    R = T.rotation()
    es, Js = [np.zeros(0)], [np.zeros((0, 6))]
    for kind, i, j in match_primitives(R, T.translation(), src, tgt, cfg):
        if kind == "plane":
            P = src.points[src.planes[i]["inliers"]]
            e, J = plane_point_residuals(T, P, tgt.planes[j]["n"], tgt.planes[j]["d"])
        else:
            inl = src.cylinders[i]["inliers"]
            P, ct = src.points[inl], tgt.cylinders[j]
            # side check (see the use_numpy script): keep a point only if its mapped normal points outward
            w = transform_points(T, P) - ct["c"]
            P = P[np.sum((src.normals[inl] @ R.T) * (w - (w @ ct["a"])[:, None] * ct["a"]), axis=1) > 0]
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
    """Gauss-Newton on SE(3) with manif's right-plus retraction, re-associating
    at every iteration; minimum-norm steps with numerical rank cutoff
    cfg.rank_rcond (see the use_numpy script).
    Returns:
        T (manif.SE3), info dict with 'iters', 'converged', 'H', 'n_residuals'
    """
    T = T0
    converged = False
    for it in range(1, cfg.max_iters + 1):
        e, J = RESIDUALS[method](T, src, tgt, cfg, it - 1)
        if len(e) < 6:
            break
        delta = np.linalg.lstsq(J, -e, rcond=cfg.rank_rcond)[0]
        T = T.rplus(manif.SE3Tangent(delta))
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
    xi = T_est.rminus(T_true).coeffs()
    return np.linalg.norm(T_est.translation() - T_true.translation()), np.linalg.norm(xi[3:])


# ---------------------------------------------------------------------------
# Experiments 2-3
# ---------------------------------------------------------------------------

def make_scan_pair(scene, seed, lcfg, ecfg, x_range=(-8.0, 8.0)):
    """Target and source scans of `scene`; the source sensor sits at
    RELATIVE_MOTION from the target sensor.
    Returns:
        src, tgt (Scan), T_true (manif.SE3), T_target (manif.SE3, world pose of the target sensor)
    """
    rng = np.random.default_rng(seed)
    T_tgt = make_pose(*random_sensor_pose(rng, x_range=x_range))
    T_true = manif.SE3Tangent(RELATIVE_MOTION).exp()
    T_src = T_tgt.compose(T_true)
    scans = []
    for T in (T_tgt, T_src):
        pts, _ = simulate_scan(scene, T.rotation(), T.translation(), lcfg, rng)
        scans.append(prepare_scan(pts, ecfg, rng))
    return scans[1], scans[0], T_true, T_tgt


def run_basin_experiment(levels, n_pairs, n_starts, seed, noise=0.01, n_azimuth=360,
                         success_trans=0.05, success_rot_deg=1.0, cfg=RegistrationConfig()):
    """Experiment 2 (see the use_numpy script for the full description)."""
    scene = corridor_scene(pillars=True)
    lcfg = LidarConfig(n_azimuth=n_azimuth, range_noise_std=noise)
    ecfg = ExtractionConfig(inlier_dist=inlier_threshold(noise))
    rows = {m: [[] for _ in levels] for m in METHODS}
    for p in range(n_pairs):
        src, tgt, T_true, _ = make_scan_pair(scene, [seed, p], lcfg, ecfg)
        for k, (tr, rot_deg) in enumerate(levels):
            rng = np.random.default_rng([seed, p, k])
            for _ in range(n_starts):
                T0 = T_true.rplus(manif.SE3Tangent(random_perturbation(rng, tr, np.deg2rad(rot_deg))))
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
    """Experiment 3 (see the use_numpy script for the full description)."""
    lcfg = LidarConfig(n_azimuth=n_azimuth, range_noise_std=noise)
    ecfg = ExtractionConfig(inlier_dist=inlier_threshold(noise))
    keys = ("eig_ratio", "axis_alignment", "axis_sigma_mm", "lateral_sigma_mm", "axis_err_mm", "trans_mm", "rot_deg")
    out = {}
    for name, pillars in (("corridor", False), ("corridor_pillars", True)):
        scene = corridor_scene(pillars=pillars)
        rows = {m: {k: [] for k in keys} for m in METHODS}
        for p in range(n_pairs):
            src, tgt, T_true, T_tgt = make_scan_pair(scene, [seed, p], lcfg, ecfg)
            axis = T_tgt.rotation().T @ np.array([1.0, 0.0, 0.0])      # corridor axis, target frame
            lateral = T_tgt.rotation().T @ np.array([0.0, 1.0, 0.0])
            rng = np.random.default_rng([seed, p, 99])
            T0 = T_true.rplus(manif.SE3Tangent(random_perturbation(rng, start_trans, np.deg2rad(start_rot_deg))))
            R_true = T_true.rotation()
            for m in METHODS:
                _, J = RESIDUALS[m](T_true, src, tgt, cfg, cfg.max_iters)
                H = J.T @ J
                evals, evecs = np.linalg.eigh(H)
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
                r["axis_err_mm"].append(1e3 * (T.translation() - T_true.translation()) @ axis)
                r["trans_mm"].append(1e3 * et)
                r["rot_deg"].append(np.rad2deg(er))
        out[name] = {m: {k: np.asarray(v) for k, v in rows[m].items()} for m in METHODS}
    return out


# ---------------------------------------------------------------------------
# Planes as landmarks (experiment 4)
# ---------------------------------------------------------------------------

def plane_factor(T, n_w, d_w, n_s, d_s):
    """Residual and Jacobians of one plane measurement (n_s, d_s) against the
    world landmark (n_w, d_w), as in the use_numpy script, with the pose
    Jacobian from manif: the predicted normal R^-1 n_w through SO3.inverse and
    SO3.act, the predicted offset d_w + n_w . t through SE3.act at the origin
    (whose Jacobian is d t / d xi).
    Returns:
        e (3,), J_pose (3,6) wrt T.rplus(xi), J_lm (3,3) wrt (s2_exp(n_w, dn), d_w + dd)
    """
    J_inv, J_rot = np.zeros((3, 3)), np.zeros((3, 3))
    n = manif.SO3(T.coeffs()[3:7]).inverse(J_inv).act(n_w, J_rot)
    J_t = np.zeros((3, 6))
    t = T.act(np.zeros(3), J_t)
    Bs, Bw = s2_basis(n_s), s2_basis(n_w)
    e = np.concatenate([Bs.T @ n, [d_w + n_w @ t - d_s]])
    J_pose = np.zeros((3, 6))
    J_pose[:2, 3:] = Bs.T @ J_rot @ J_inv      # SE3's right-plus rotates like SO3's: same 3 rotation coordinates
    J_pose[2, :] = n_w @ J_t
    J_lm = np.zeros((3, 3))
    J_lm[:2, :2] = Bs.T @ T.rotation().T @ Bw
    J_lm[2, :2] = t @ Bw
    J_lm[2, 2] = 1.0
    return e, J_pose, J_lm


def between_factor(Xi, Xj, Z):
    """Odometry residual Log(Z^-1 Xi^-1 Xj) = Xj.rminus(Xi.compose(Z)) and its
    Jacobians wrt Xi and Xj, chained from manif's compose/rminus Jacobians
    (as in use_manif/pose_graph.py)."""
    J_c = np.zeros((6, 6))
    T_pred = Xi.compose(Z, J_c)
    J_ej, J_ep = np.zeros((6, 6)), np.zeros((6, 6))
    e = Xj.rminus(T_pred, J_ej, J_ep).coeffs()
    return e, J_ep @ J_c, J_ej


def run_odometry(scans, T_start, init_rel, noise, cfg=RegistrationConfig()):
    """Chained scan-to-scan primitive registration (see the use_numpy script).
    Returns:
        poses (manif.SE3), Z (manif.SE3), Omega (6x6)
    """
    poses, Z, Omega = [T_start], [], []
    for k in range(len(scans) - 1):
        T, info = register("primitive", init_rel[k], scans[k + 1], scans[k], cfg)
        Z.append(T)
        Omega.append(info["H"] / noise ** 2)
        poses.append(poses[-1].compose(T))
    return poses, Z, Omega


def optimize_plane_graph(poses0, Z, Omega, scans, landmarks0, observations, noise, damping=1e-3, tol=1e-8,
                         max_iters=50):
    """Batch Levenberg-Marquardt over every pose (manif.SE3) and plane landmark
    (see the use_numpy script).
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
            e = poses[k + 1].rminus(poses[k].compose(Zk)).coeffs()
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
            trial_p = [poses[k].rplus(manif.SE3Tangent(delta[6 * k:6 * k + 6])) for k in range(n_p)]
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
    """ATE RMSE (m) after the best rigid alignment, and the end-point drift (m, rad)."""
    P_est = np.array([T.translation() for T in poses_est])
    P_true = np.array([T.translation() for T in poses_true])
    _, R, _ = umeyama_alignment(P_est, P_true)
    t = P_true.mean(axis=0) - R @ P_est.mean(axis=0)
    ate = np.sqrt(np.mean(np.sum((P_est @ R.T + t - P_true) ** 2, axis=1)))
    end_t, end_r = pose_error(poses_est[-1], poses_true[-1])
    return ate, end_t, end_r


def run_landmark_experiment(n_runs, seed, noise=0.01, n_azimuth=360, init_trans=0.1, init_rot_deg=2.0):
    """Experiment 4 (see the use_numpy script for the full description)."""
    scene = l_corridor_scene()
    traj = [make_pose(R, t) for R, t in l_corridor_trajectory()]
    lcfg = LidarConfig(n_azimuth=n_azimuth, range_noise_std=noise)
    ecfg = ExtractionConfig(inlier_dist=inlier_threshold(noise))
    out = {m: {"ate_mm": [], "end_mm": [], "end_deg": []} for m in ("odometry", "graph")}
    out["n_landmarks"] = []
    for run in range(n_runs):
        rng = np.random.default_rng([seed, run])
        scans = [prepare_scan(simulate_scan(scene, T.rotation(), T.translation(), lcfg, rng)[0], ecfg, rng)
                 for T in traj]
        init_rel = [traj[k].between(traj[k + 1]).rplus(
                        manif.SE3Tangent(random_perturbation(rng, init_trans, np.deg2rad(init_rot_deg))))
                    for k in range(len(traj) - 1)]
        odo, Z, Omega = run_odometry(scans, traj[0], init_rel, noise)
        landmarks, observations = associate_planes(scans, [T.rotation() for T in odo],
                                                   [T.translation() for T in odo])
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
