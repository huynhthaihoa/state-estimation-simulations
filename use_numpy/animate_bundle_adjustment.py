'''
Animated version of bundle_adjustment.py's default run (docs/optimization/
bundle_adjustment.md, Section 6): joint Gauss-Newton bundle adjustment over
camera poses AND landmarks, started from a noisy initial guess.

Left: top-down (x-y) scene. Black = ground truth, gray = the noisy initial
guess, blue = the current Gauss-Newton iterate. Joint BA only recovers the
scene up to a global similarity (its gauge freedom), so every iterate is
aligned to ground truth for display with align_reconstruction_to_ground_truth;
this does not change any reprojection error. Only 7 states exist (initial
guess + 6 iterations), so the motion between consecutive iterates is
interpolated for display only (positions linearly, camera heading by
rotating along the SO(3) geodesic via se3_exp/se3_log).

Right: reprojection RMS (px, log scale) per iteration, against two horizontal
references computed in the same run: the landmarks-only and poses-only
one-sided solvers, which hold the other half at its noisy initial guess.

The message (doc Section 4): optimizing poses and landmarks jointly reaches
about the pixel-noise level, while each one-sided solver stalls well above it.
Not claimed: that these RMS levels hold for other seeds or noise settings,
or anything about how the number of iterations scales.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from bundle_adjustment import (
    generate_ground_truth_scene, build_observations, perturb_initial_guess,
    run_ba_landmarks_only, run_ba_poses_only, run_bundle_adjustment,
    align_reconstruction_to_ground_truth, reprojection_rms, camera_project)
from lie_utils import se3_exp, se3_log, se3_inv, compute_se3_inv_right_jacobian


def run_bundle_adjustment_recorded(T_init, P_init, observations, K, pose_noise_std, pixel_noise_std,
                                   gn_tol, gn_max_iters):
    """Copy of bundle_adjustment.run_bundle_adjustment's loop (same helpers,
    same update) that snapshots (T_est, P_est) before the first step and
    after every Gauss-Newton update (the original updates them in place and
    returns only the last).
    Returns:
        snapshots: list of (T_list, P_array); index 0 is the initial guess
    """
    n_cameras, n_landmarks = len(T_init), len(P_init)
    pose_dof = 6 * n_cameras
    dof = pose_dof + 3 * n_landmarks
    Omega_prior = np.eye(6) / pose_noise_std ** 2
    omega_pixel = 1.0 / pixel_noise_std ** 2
    T_prior = [T_init[0], T_init[1]]

    T_est = list(T_init)
    P_est = P_init.copy()
    snapshots = [(list(T_est), P_est.copy())]

    for _ in range(gn_max_iters):
        H = np.zeros((dof, dof))
        g = np.zeros(dof)
        for k in (0, 1):
            e0 = se3_log(se3_inv(T_prior[k]) @ T_est[k])
            J = compute_se3_inv_right_jacobian(e0)
            c = 6 * k
            H[c:c + 6, c:c + 6] += J.T @ Omega_prior @ J
            g[c:c + 6] += -J.T @ Omega_prior @ e0
        for i, j, z_ij in observations:
            pred, J_pose, J_point = camera_project(T_est[i], P_est[j], K, with_jacobians=True)
            r = z_ij - pred
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
        snapshots.append((list(T_est), P_est.copy()))
        if np.linalg.norm(delta) < gn_tol:
            break
    return snapshots


def interp_state(A, B, s):
    """Display interpolation between two aligned states (T_list, P): linear
    positions, geodesic rotation (R_a Exp(s * Log(R_a^T R_b)))."""
    (Ta, Pa), (Tb, Pb) = A, B
    T_out = []
    for a, b in zip(Ta, Tb):
        T = np.eye(4)
        Rrel = np.eye(4)
        Rrel[:3, :3] = a[:3, :3].T @ b[:3, :3]
        w = se3_log(Rrel)[3:]
        Rs = np.eye(4)
        Rs[:3, :3] = a[:3, :3]
        T[:3, :3] = (Rs @ se3_exp(np.concatenate([np.zeros(3), s * w])))[:3, :3]
        T[:3, 3] = (1 - s) * a[:3, 3] + s * b[:3, 3]
        T_out.append(T)
    return T_out, (1 - s) * Pa + s * Pb


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-cameras", type=int, default=8)
    p.add_argument("--n-landmarks", type=int, default=60)
    p.add_argument("--camera-radius", type=float, default=5.0)
    p.add_argument("--arc-span-deg", type=float, default=180.0)
    p.add_argument("--landmark-spread", type=float, default=2.0)
    p.add_argument("--min-observations", type=int, default=2)
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--focal-length", type=float, default=800.0)
    p.add_argument("--pose-noise-std", type=float, default=0.1)
    p.add_argument("--landmark-noise-std", type=float, default=0.3)
    p.add_argument("--pixel-noise-std", type=float, default=1.0)
    p.add_argument("--gn-tol", type=float, default=1e-6)
    p.add_argument("--gn-max-iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--dpi", type=int, default=80)
    p.add_argument("--out", type=str, default="bundle_adjustment.gif")
    p.add_argument("--frames-png", type=str, default=None, help="Directory to save a few sample frames")
    a = p.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    K = (a.focal_length, a.focal_length, a.image_width / 2.0, a.image_height / 2.0)
    T_true, P_all = generate_ground_truth_scene(
        a.n_cameras, a.n_landmarks, a.camera_radius, a.arc_span_deg, a.landmark_spread, rng)
    P_true, obs = build_observations(T_true, P_all, K, a.pixel_noise_std, a.min_observations, rng)
    T_init, P_init = perturb_initial_guess(T_true, P_true, a.pose_noise_std, a.landmark_noise_std, rng)

    P_lm = run_ba_landmarks_only(T_init, P_init, obs, K, a.gn_tol, a.gn_max_iters)
    T_po = run_ba_poses_only(T_init, P_init, obs, K, a.gn_tol, a.gn_max_iters)
    rms_lm = reprojection_rms(T_init, P_lm, obs, K)
    rms_po = reprojection_rms(T_po, P_init, obs, K)

    snaps = run_bundle_adjustment_recorded(
        T_init, P_init, obs, K, a.pose_noise_std, a.pixel_noise_std, a.gn_tol, a.gn_max_iters)
    # Equivalence check against the module's own function (silence its prints).
    stdout, sys.stdout = sys.stdout, open(os.devnull, "w")
    try:
        T_ref, P_ref = run_bundle_adjustment(T_init, P_init, obs, K, a.pose_noise_std, a.pixel_noise_std,
                                             a.gn_tol, a.gn_max_iters)
    finally:
        sys.stdout.close()
        sys.stdout = stdout
    same = (np.allclose(P_ref, snaps[-1][1], atol=0, rtol=0) and
            all(np.allclose(x, y, atol=0, rtol=0) for x, y in zip(T_ref, snaps[-1][0])))
    print(f"copied loop == run_bundle_adjustment final result (exact): {same}")

    rms = [reprojection_rms(T, P, obs, K) for T, P in snaps]
    n_it = len(snaps) - 1
    print("reprojection RMS per state (px):", ", ".join(f"{r:.3f}" for r in rms))
    print(f"landmarks-only RMS = {rms_lm:.3f} px, poses-only RMS = {rms_po:.3f} px, joint final = {rms[-1]:.3f} px")

    aligned = [align_reconstruction_to_ground_truth(T_true, T, P) for T, P in snaps]
    for r, (T, P) in zip(rms, aligned):  # alignment must not change reprojection
        assert abs(reprojection_rms(T, P, obs, K) - r) < 1e-6 * max(1.0, r)

    # Fixed limits from all displayed positions (aligned iterates + truth).
    pts = np.vstack([np.array([T[:3, 3] for T in Ts])[:, :2] for Ts, _ in aligned] +
                    [P[:, :2] for _, P in aligned] +
                    [np.array([T[:3, 3] for T in T_true])[:, :2], P_true[:, :2]])
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.08 * (hi - lo)
    xlim, ylim = (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])

    # Frame schedule: (i, j, s) = interpolate aligned[i] -> aligned[j] at s.
    sched = [(0, 0, 0.0)] * 10
    for k in range(1, n_it + 1):
        n_tr, n_hold = (6, 6) if k <= 3 else (2, 2)
        sched += [(k - 1, k, (m + 1) / n_tr) for m in range(n_tr)]
        sched += [(k, k, 0.0)] * n_hold
    sched += [(n_it, n_it, 0.0)] * 8

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(10, 7.5), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.17, wspace=0.22)

    def heading(Ts, L=0.35):
        return np.array([T[:3, :3] @ np.array([0.0, 0.0, L]) for T in Ts])

    pos_true = np.array([T[:3, 3] for T in T_true])
    ax.scatter(P_true[:, 0], P_true[:, 1], c="k", s=12, alpha=0.6, label="landmarks (truth)")
    ax.scatter(pos_true[:, 0], pos_true[:, 1], c="k", marker="^", s=60, label="cameras (truth)")
    Th = heading(T_true)
    for pp, h in zip(pos_true, Th):
        ax.arrow(pp[0], pp[1], h[0], h[1], color="k", head_width=0.08, alpha=0.6)
    T0, P0 = aligned[0]
    pos0 = np.array([T[:3, 3] for T in T0])
    ax.scatter(P0[:, 0], P0[:, 1], c="tab:gray", s=12, alpha=0.5, label="initial guess (ghost)")
    ax.scatter(pos0[:, 0], pos0[:, 1], c="tab:gray", marker="^", s=60, alpha=0.5)
    for pp, h in zip(pos0, heading(T0)):
        ax.arrow(pp[0], pp[1], h[0], h[1], color="tab:gray", head_width=0.08, alpha=0.5)
    lm_sc = ax.scatter(P0[:, 0], P0[:, 1], c="tab:blue", s=14, alpha=0.8, label="current iterate")
    cam_sc = ax.scatter(pos0[:, 0], pos0[:, 1], c="tab:blue", marker="^", s=60, zorder=4)
    arrows = []
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Scene, top-down (aligned to truth)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    it_text = ax.text(0.98, 0.02, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
                      bbox=dict(fc="w", ec="0.7", alpha=0.9))

    axr.axhline(rms_lm, color="tab:orange", ls="--", label=f"landmarks-only solver: {rms_lm:.1f} px")
    axr.axhline(rms_po, color="tab:purple", ls="--", label=f"poses-only solver: {rms_po:.1f} px")
    axr.axhline(a.pixel_noise_std, color="0.5", ls=":", label=f"pixel noise std: {a.pixel_noise_std:.1f} px")
    (line,) = axr.plot([], [], "-o", color="tab:blue", ms=5, label="joint BA (Gauss-Newton)")
    axr.set_yscale("log")
    axr.set_xlim(-0.3, n_it + 0.3)
    axr.set_ylim(0.6, max(rms) * 2.5)
    axr.set_xlabel("Gauss-Newton iteration"); axr.set_ylabel("reprojection RMS (px)")
    axr.set_title("Reprojection error", fontsize=10)
    axr.legend(fontsize=7, loc="upper right")
    val_text = axr.text(0.97, 0.45, "", transform=axr.transAxes, ha="right", fontsize=11, color="tab:blue")

    fig.text(0.5, 0.045,
             "Joint optimization over poses AND landmarks reaches ~%.1f px; holding either half at its noisy guess stalls at %.0f-%.0f px.\n"
             "Motion between the 7 iterates is interpolated for display; blue is aligned to truth (gauge freedom)."
             % (rms[-1], min(rms_lm, rms_po), max(rms_lm, rms_po)),
             ha="center", va="center", fontsize=9)
    fig.suptitle("Bundle adjustment: pose-only and landmark-only vs joint", fontsize=12)

    def update(f):
        i, j, s = sched[f]
        T, P = interp_state(aligned[i], aligned[j], s) if i != j else aligned[i]
        pos = np.array([t[:3, 3] for t in T])
        lm_sc.set_offsets(P[:, :2])
        cam_sc.set_offsets(pos[:, :2])
        for ar in arrows:
            ar.remove()
        arrows.clear()
        for pp, h in zip(pos, heading(T)):
            arrows.append(ax.arrow(pp[0], pp[1], h[0], h[1], color="tab:blue", head_width=0.08, alpha=0.8, zorder=4))
        xs = list(range(j + 1)) if i == j else list(range(i + 1))
        ys = [rms[k] for k in xs]
        if i != j:  # log-linear interpolated tip
            xs.append(i + s)
            ys.append(np.exp((1 - s) * np.log(rms[i]) + s * np.log(rms[j])))
        line.set_data(xs, ys)
        cur = ys[-1]
        it_text.set_text("initial guess" if (i == 0 and j == 0) else f"iteration {j}" + ("" if i == j else f" (from {i})"))
        val_text.set_text(f"{cur:.1f} px")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(sched), blit=False)
    anim.save(a.out, writer=animation.PillowWriter(fps=a.fps), dpi=a.dpi)
    if a.frames_png:
        os.makedirs(a.frames_png, exist_ok=True)
        for name, f in [("early", 3), ("mid", 10 + 6 * 0 + 3), ("it2", 10 + 12 + 5), ("final", len(sched) - 1)]:
            update(f)
            fig.savefig(os.path.join(a.frames_png, f"frame_{name}.png"), dpi=a.dpi)
    print(f"Saved {a.out}: {len(sched)} frames, {os.path.getsize(a.out) / 1e6:.2f} MB, "
          f"{time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
