"""Generates one illustrative figure per simulation, from the simulations' own
output, into assets/<name>.png, plus a setup figure (<name>_concept) for the docs
that have one.

Nothing here re-implements an estimator: every trajectory, error and NEES
curve comes from calling the functions in use_numpy/ exactly the way each
script's main() does (same defaults, same seed, same order of random draws),
so the numbers on a figure match what the script itself prints. The only
exception is a non-default duration where the default is too short to show
anything, which the figure then states in its subtitle.

Usage:
    uv run python generator/make_figures.py            # all figures
    uv run python generator/make_figures.py pose_graph # just one (see FIGURES)
"""
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.join(ROOT, "assets")
SEED = 0

TRUTH = "#1f1f1f"
PRIOR = "#9a9a9a"
C = {"red": "#d1495b", "blue": "#2e86de", "green": "#2a9d8f", "purple": "#8e6bbf",
     "orange": "#f4a261", "gold": "#e9c46a"}


def style():
    plt.rcParams.update({
        "figure.figsize": (13, 5.2), "figure.dpi": 150, "savefig.dpi": 150,
        "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.fontsize": 8.5, "legend.frameon": False,
    })


def quiet(fn, *args, **kwargs):
    """Calls fn with its progress prints suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def plain_log_ticks(ax):
    """Log-scale y axis labelled 2, 3, 6, 10, ... instead of 2×10⁰."""
    ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 3.0, 6.0)))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def legend_below(ax, ncol=2):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=ncol, fontsize=8.5)


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def finish(fig, name, title, subtitle, bottom=0.0, tight=True):
    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.925, subtitle, fontsize=10.5, color="#555555", ha="left")
    if tight:
        fig.tight_layout(rect=(0, bottom, 1, 0.93))
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {os.path.relpath(path, ROOT)}")


def xy_of(T_list):
    pts = np.array([T[0:3, 3] for T in T_list])
    return pts[:, 0], pts[:, 1]


# --------------------------------------------------------------------------- figures

def fig_pose_tracking():
    """pointcloud_pose_tracking.py: six estimators on one point-cloud tracking run."""
    from state_estimation_simulations.use_numpy import pointcloud_pose_tracking as m
    from state_estimation_simulations.use_numpy.lie_utils import se3_exp

    duration, dt, n_points = 5.0, 0.1, 20
    vel_std, gyro_std, point_std, init_std = 0.05, 0.02, 0.03, 0.1
    rng = np.random.default_rng(SEED)
    body, T_true, u_meas, z = m.generate_ground_truth_and_data(duration, dt, n_points, vel_std, gyro_std, point_std, rng)
    T_init = T_true[0] @ se3_exp(rng.normal(0.0, init_std, 6))
    P_init = init_std ** 2 * np.eye(6)
    Q = dt ** 2 * np.diag([vel_std ** 2] * 3 + [gyro_std ** 2] * 3)

    T_dr = m.run_dead_reckoning(T_init, u_meas, dt)
    runs = [
        ("Dead reckoning (prior only)", T_dr, PRIOR, ":"),
        ("EKF", m.run_ekf(T_init, P_init, u_meas, z, body, dt, Q, point_std), C["red"], "-"),
        ("Invariant EKF", m.run_iekf(T_init, P_init, u_meas, z, body, dt, Q, point_std), C["green"], "--"),
        ("UKF", m.run_ukf(T_init, P_init, u_meas, z, body, dt, Q, point_std, 1.0, 2.0, -3.0), C["purple"], "-."),
        ("Vanilla KF (12-dim ambient)", m.run_vanilla_kf(T_init, P_init, u_meas, z, body, dt, Q, point_std), C["orange"], "-"),
        ("Batch Gauss-Newton", quiet(m.run_batch_gn, T_dr, T_init, P_init, u_meas, z, body, dt, Q, point_std, 1e-6, 20), C["blue"], "-"),
    ]

    fig, (ax_xy, ax_err) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.25]})
    ax_xy.plot(*xy_of(T_true), color=TRUTH, lw=2.4, label="Ground truth")
    t = np.arange(len(T_true)) * dt
    for name, T_est, col, ls in runs:
        _, pos_err = m.pose_errors(T_true, T_est)
        ax_xy.plot(*xy_of(T_est), color=col, ls=ls, lw=1.6, label=f"{name}  (RMSE {rms(pos_err) * 100:.1f} cm)")
        ax_err.plot(t, pos_err, color=col, ls=ls, lw=1.8)
        print(f"    {name:<28s} RMS pos = {rms(pos_err):.4f} m")
    ax_xy.set(title="Object trajectory (x-y)", xlabel="x (m)", ylabel="y (m)")
    ax_xy.axis("equal")
    fig.legend(*ax_xy.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=9,
               title="in brackets: position RMSE over the run", title_fontsize=8.5)
    ax_err.set(title="Position error", xlabel="time (s)", ylabel="position error (m)", yscale="log")
    finish(fig, "pose_tracking", "Point-cloud pose tracking on SE(3)",
           f"Noisy twist prior + {n_points} noisy body-frame points per step (z = T·p + noise); "
           "one run, seed 0, script defaults", bottom=0.14)


def fig_imu_propagation_position_fix():
    """imu_propagation_position_fix.py: 100 Hz IMU dead reckoning + 1 Hz position-only Gauss-Newton fix."""
    from state_estimation_simulations.use_numpy import imu_propagation_position_fix as m

    seconds, pos_std = 10, 0.05
    args = (0.01, None, 4, 1e-6, 10, 1.0, 0.5, pos_std, np.eye(3) / pos_std ** 2)

    def run(n):
        a = list(args)
        a[1] = n
        return quiet(m.run_simulation, *a, np.random.default_rng(SEED))

    # run_simulation draws its random numbers in a fixed order, so a run of n
    # seconds reproduces the first n seconds of a longer run exactly: its final
    # pose is the full run's state at the end of second n.
    T_true, T_est, pre, post = run(seconds)
    rot_err = []
    for n in range(1, seconds + 1):
        Tt, Te, _, _ = run(n)
        R_rel = Tt[0:3, 0:3].T @ Te[0:3, 0:3]
        rot_err.append(np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))))
    print(f"    position error before fix: {np.round(pre, 3)}")
    print(f"    position error after fix:  {np.round(post, 3)}")
    print(f"    rotation error (deg):      {np.round(rot_err, 3)}")

    s = np.arange(1, seconds + 1)
    fig, (ax_p, ax_r) = plt.subplots(1, 2)
    xs, ys = [0.0], [0.0]
    for k in range(seconds):
        xs += [s[k], s[k]]
        ys += [pre[k], post[k]]
    ax_p.plot(xs, ys, color=C["blue"], lw=1.2, alpha=0.5)
    ax_p.scatter(s, pre, color=C["red"], zorder=3, label="just before the 1 Hz fix (after 100 IMU steps)")
    ax_p.scatter(s, post, color=C["green"], zorder=3, label="just after the Gauss-Newton fix")
    ax_p.axhline(pos_std * np.sqrt(3), color=PRIOR, ls="--", lw=1,
                 label=f"3D GPS noise level, σ√3 = {pos_std * np.sqrt(3) * 100:.1f} cm (σ = {pos_std * 100:.0f} cm per axis)")
    ax_p.set(title="Position error: each fix resets it to about the GPS noise level", xlabel="time (s)",
             ylabel="position error (m)", ylim=(0, max(max(pre), max(post)) * 1.35))
    ax_p.legend(loc="upper left")
    ax_r.plot(s, rot_err, "o-", color=C["purple"], lw=1.8)
    ax_r.set(title="Orientation error: never corrected", xlabel="time (s)", ylabel="rotation error (deg)")
    ax_r.text(0.03, 0.95, "position-only fix: Jacobian [R | 0],\nso the rotation block gets no correction",
              transform=ax_r.transAxes, va="top", fontsize=9, color="#555555")
    finish(fig, "imu_propagation_position_fix", "IMU dead reckoning + GPS-style position fix",
           f"100 Hz noisy body twist on SE(3), 1 Hz position-only measurement; seed 0, --total-seconds {seconds} "
           "(default 3), other settings default")


def fig_imu_preintegration():
    """imu_preintegration.py: one bundle of 100 samples, then a bias update applied to first order."""
    from state_estimation_simulations.use_numpy import imu_preintegration as m

    hz = 100
    b_g, b_a = np.array([0.01, -0.01, 0.02]), np.array([0.05, 0.0, -0.05])
    v_cmd, w_cmd = np.array([1.0, 0.1, 0.0]), np.array([0.0, 0.0, 0.5])
    b_g_new, b_a_new = np.array([0.008, -0.009, 0.018]), np.array([0.045, 0.002, -0.048])
    dt = 1.0 / hz

    bundle = m.PreintegratedIMUBundle(b_g, b_a)
    path = [bundle.delta_p.copy()]
    for _ in range(hz):
        bundle.integrate_measurement(v_cmd + b_a, w_cmd + b_g, dt)
        path.append(bundle.delta_p.copy())
    path = np.array(path)
    c_R, c_v, c_p = bundle.get_corrected_measurement(b_g_new, b_a_new)

    # Reference for the first-order correction: re-integrate the same raw
    # samples from scratch with the new bias (what the correction avoids doing).
    ref = m.PreintegratedIMUBundle(b_g_new, b_a_new)
    for _ in range(hz):
        ref.integrate_measurement(v_cmd + b_a, w_cmd + b_g, dt)

    def rot_deg(Ra, Rb):
        return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))

    shift = [np.linalg.norm(ref.delta_p - bundle.delta_p), np.linalg.norm(ref.delta_v - bundle.delta_v),
             rot_deg(bundle.delta_R, ref.delta_R)]
    resid = [np.linalg.norm(c_p - ref.delta_p), np.linalg.norm(c_v - ref.delta_v), rot_deg(c_R, ref.delta_R)]
    print(f"    shift from bias update  |dp|,|dv|,dR = {shift}")
    print(f"    first-order residual    |dp|,|dv|,dR = {resid}")

    fig, (ax_p, ax_b) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.1, 1]})
    ax_p.plot(path[:, 0], path[:, 1], ".", color=C["blue"], ms=3.5, label=f"Δp after each of the {hz} samples")
    ax_p.scatter([0], [0], s=80, marker="s", color=TRUTH, zorder=3, label="keyframe i")
    ax_p.scatter([path[-1, 0]], [path[-1, 1]], s=80, marker="s", color=C["red"], zorder=3, label="keyframe j: one Δp_ij")
    ax_p.set(title="100 samples compressed into one relative motion", xlabel="Δp_x (m)", ylabel="Δp_y (m)")
    ax_p.axis("equal")
    ax_p.legend(loc="upper left")

    x = np.arange(3)
    w = 0.38
    ax_b.bar(x - w / 2, shift, w, color=C["orange"], label="change caused by the bias update")
    ax_b.bar(x + w / 2, resid, w, color=C["green"], label="first-order correction's error vs. re-integration")
    ax_b.set_xticks(x, ["Δp (m)", "Δv (m/s)", "ΔR (deg)"])
    ax_b.set(yscale="log", ylim=(1e-6, 20), title="Bias update: Jacobian correction vs. full re-integration")
    ax_b.legend(loc="upper left")
    finish(fig, "imu_preintegration", "IMU preintegration with bias Jacobians",
           "Script defaults (--linear-vel is a body-frame acceleration, integrated twice); the bias update is "
           f"applied as 1 Jacobian correction instead of re-integrating {hz} samples")


def fig_pose_graph():
    """pose_graph.py: square loop, odometry drift, one loop closure, Levenberg-Marquardt relaxation."""
    from state_estimation_simulations.use_numpy import pose_graph as m

    rng = np.random.default_rng(SEED)
    gt = m.generate_ground_truth_trajectory(2.0)
    odo, loop = m.simulate_noisy_edges(gt, 0.05, 0.01, 0.5, rng)
    T_dr = m.run_dead_reckoning(gt[0], odo)
    T_opt = quiet(m.run_pose_graph_optimization, T_dr, odo + loop, np.eye(6), 0.01, 1e-6, 10)
    _, e_dr = m.pose_errors(gt, T_dr)
    _, e_opt = m.pose_errors(gt, T_opt)
    print(f"    RMS pos: odometry {rms(e_dr):.4f} m, optimized {rms(e_opt):.4f} m")

    def closed(T_list):
        x, y = xy_of(T_list)
        return np.append(x, x[0]), np.append(y, y[0])

    fig, (ax_xy, ax_e) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.1, 1]})
    ax_xy.plot(*closed(gt), "o-", color=TRUTH, lw=2.2, ms=7, label="Ground truth")
    ax_xy.plot(*xy_of(T_dr), "o--", color=PRIOR, lw=1.8, ms=6, label="Odometry only (drifts)")
    ax_xy.plot(*xy_of(T_opt), "o-", color=C["blue"], lw=1.8, ms=6, label="After pose-graph optimization")
    (i, j, _), = loop
    a, b = T_opt[i][0:2, 3], T_opt[j][0:2, 3]
    ax_xy.plot([a[0], b[0]], [a[1], b[1]], ls=(0, (5, 3)), color=C["red"], lw=2.2,
               label=f"loop-closure edge X{i}–X{j}")
    for k, T in enumerate(gt):
        ax_xy.text(T[0, 3] - 0.22, T[1, 3] + 0.1, f"X{k}", fontsize=9)
    ax_xy.set(title="Square loop (x-y)", xlabel="x (m)", ylabel="y (m)")
    ax_xy.axis("equal")
    ax_xy.legend(loc="center", bbox_to_anchor=(0.56, 0.5), fontsize=8)
    ax_xy.set_xlim(-0.6, 2.6)

    k = np.arange(len(gt))
    w = 0.38
    ax_e.bar(k - w / 2, e_dr, w, color=PRIOR, label=f"Odometry only (RMSE {rms(e_dr) * 100:.1f} cm)")
    ax_e.bar(k + w / 2, e_opt, w, color=C["blue"], label=f"Optimized (RMSE {rms(e_opt) * 100:.1f} cm)")
    ax_e.set_xticks(k, ["X0\n(anchored)"] + [f"X{n}" for n in k[1:]])
    ax_e.set(title="Position error per pose", ylabel="position error (m)")
    ax_e.legend(loc="upper left")
    finish(fig, "pose_graph", "Pose-graph relaxation with one loop closure",
           "Noisy SE(3) odometry edges + one loop-closure edge, residual Log(Z_ij⁻¹ X_i⁻¹ X_j), "
           "Levenberg-Marquardt; seed 0, script defaults")


def fig_bundle_adjustment():
    """bundle_adjustment.py: landmarks-only vs poses-only vs joint BA."""
    from state_estimation_simulations.use_numpy import bundle_adjustment as m

    rng = np.random.default_rng(SEED)
    K = (800.0, 800.0, 320.0, 240.0)
    T_true, P_all = m.generate_ground_truth_scene(8, 60, 5.0, 180.0, 2.0, rng)
    P_true, obs = m.build_observations(T_true, P_all, K, 1.0, 2, rng)
    T_init, P_init = m.perturb_initial_guess(T_true, P_true, 0.1, 0.3, rng)
    P_lo = quiet(m.run_ba_landmarks_only, T_init, P_init, obs, K, 1e-6, 30)
    T_po = quiet(m.run_ba_poses_only, T_init, P_init, obs, K, 1e-6, 30)
    T_ba, P_ba = quiet(m.run_bundle_adjustment, T_init, P_init, obs, K, 0.1, 1.0, 1e-6, 30)
    T_ba, P_ba = m.align_reconstruction_to_ground_truth(T_true, T_ba, P_ba)
    rows = [("Noisy initial guess", T_init, P_init), ("Landmarks only", T_init, P_lo),
            ("Poses only", T_po, P_init), ("Joint bundle adjustment", T_ba, P_ba)]
    vals = np.array(m.compute_all_metrics(rows, T_true, P_true, obs, K))
    for (name, _, _), v in zip(rows, vals):
        print(f"    {name:<24s} rot {v[0]:.3f} deg, pos {v[1]:.4f} m, landmark {v[2]:.4f} m, reproj {v[3]:.3f} px")

    fig, (ax_s, ax_b) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.05, 1]})
    for T_list, P_list, col, label, mk in [(T_true, P_true, TRUTH, "ground truth", "o"),
                                            (T_init, P_init, PRIOR, "noisy initial guess", "x"),
                                            (T_ba, P_ba, C["blue"], "joint BA", "o")]:
        P = np.asarray(P_list)
        ax_s.scatter(P[:, 0], P[:, 1], s=14 if mk == "o" else 18, marker=mk, color=col, alpha=0.75,
                     facecolors="none" if col == C["blue"] else None, label=f"landmarks, {label}")
        cams = np.array([T[0:3, 3] for T in T_list])
        ax_s.scatter(cams[:, 0], cams[:, 1], s=40, marker="s", color=col, zorder=3)
        for T in T_list:
            p, h = T[0:3, 3], T[0:3, 0:3] @ np.array([0.0, 0.0, 0.6])
            ax_s.annotate("", xy=(p[0] + h[0], p[1] + h[1]), xytext=(p[0], p[1]),
                          arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4))
    ax_s.plot([], [], color=TRUTH, marker="s", ls="", ms=6, label="cameras (arrow = optical axis)")
    ax_s.set(title=f"Scene from above: {len(T_true)} cameras, {len(P_true)} landmarks, {len(obs)} observations",
             xlabel="x (m)", ylabel="y (m)")
    ax_s.axis("equal")
    ax_s.legend(loc="upper left", fontsize=7.5, ncol=1)

    names = ["pose rotation\n(deg)", "pose position\n(m)", "landmark\n(m)", "reprojection\n(px)"]
    cols = [PRIOR, C["orange"], C["purple"], C["blue"]]
    x = np.arange(4)
    w = 0.2
    for r, ((name, _, _), col) in enumerate(zip(rows, cols)):
        ax_b.bar(x + (r - 1.5) * w, vals[r], w, color=col, label=name)
    ax_b.set_xticks(x, names)
    ax_b.set(yscale="log", ylim=(1e-3, 1e5), title="RMSE by method (after similarity alignment)")
    ax_b.legend(loc="upper center", ncol=2)
    finish(fig, "bundle_adjustment", "Bundle adjustment: joint camera-pose and landmark refinement",
           "Synthetic scene, cameras on an arc looking inward, 640×480 image (f = 800 px), 1 px pixel noise; seed 0, script defaults")


def fig_bundle_adjustment_advanced():
    """bundle_adjustment_advanced.py: incremental Local BA, with and without periodic Global BA."""
    from state_estimation_simulations.use_numpy import bundle_adjustment_advanced as m

    K = (800.0, 800.0, 320.0, 240.0)
    n_kf = 50
    T_true = m.generate_ground_truth_trajectory(n_kf, 15.0, 90.0)
    P_all = m.generate_landmark_corridor(T_true, 8, 2.0, 1.0, np.random.default_rng(SEED), extend_ahead=6.0)
    P_true, obs = m.build_observations(T_true, P_all, K, 1.0, 6.0, 3, np.random.default_rng(SEED))
    by_kf = m.group_observations_by_keyframe(obs, n_kf)
    common = (T_true, by_kf, K, 0.02, 3, 2, 6, 8)
    T_loc, _, h_loc = quiet(m.run_incremental_local_ba, *common, False, 1e-6, 15, np.random.default_rng(SEED), 20)
    T_hyb, P_hyb, h_hyb = quiet(m.run_incremental_local_ba, *common, True, 1e-6, 15, np.random.default_rng(SEED), 20)
    fe = m.simulate_frontend_trajectory(T_true, 0.02, np.random.default_rng(SEED))
    print(f"    final trailing RMS: local-only {h_loc['traj_rms_pos_err'][-1]:.4f} m, "
          f"local+global {h_hyb['traj_rms_pos_err'][-1]:.4f} m")

    fig, (ax_s, ax_t, ax_d) = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax_s.scatter(P_true[:, 0], P_true[:, 1], s=6, color=PRIOR, alpha=0.5, label="landmarks (ground truth)")
    ax_s.plot(*xy_of(fe), ":", color=C["orange"], lw=1.6, label="front-end estimate (chained noisy motion)")
    ax_s.plot(*xy_of(T_true), "-", color=TRUTH, lw=2.2, label="ground-truth keyframes")
    ax_s.plot(*xy_of([T_loc[i] for i in range(n_kf)]), "-", color=C["red"], lw=1.4, label="Local BA only")
    ax_s.plot(*xy_of([T_hyb[i] for i in range(n_kf)]), "--", color=C["blue"], lw=1.4, label="Local + periodic Global BA")
    ax_s.set(title=f"Path and landmark corridor ({n_kf} keyframes)", xlabel="x (m)", ylabel="y (m)")
    ax_s.axis("equal")
    ax_s.legend(loc="lower left", fontsize=7.5)

    ax_t.plot(h_hyb["local_step"], np.array(h_hyb["local_solve_time"]) * 1e3, "o-", ms=3, color=C["blue"],
              label="Local BA (bounded window)")
    ax_t.plot(h_hyb["global_step"], np.array(h_hyb["global_solve_time"]) * 1e3, "s-", ms=5, color=C["red"],
              label="Global BA (whole map so far)")
    ax_t.set(yscale="log", title="Solve time per call", xlabel="keyframe", ylabel="time (ms, this machine)")
    ax_t.legend(loc="lower left")

    ax_d.plot(h_loc["traj_step"], h_loc["traj_rms_pos_err"], color=C["red"], lw=1.6, label="Local BA only")
    ax_d.plot(h_hyb["traj_step"], h_hyb["traj_rms_pos_err"], color=C["blue"], lw=1.6, label="Local + periodic Global BA")
    for k in h_hyb["global_step"]:
        ax_d.axvline(k, color=PRIOR, ls=":", lw=0.8)
    ax_d.set(title="Trajectory error (last 10 keyframes)", xlabel="keyframe", ylabel="position RMSE (m)")
    ax_d.legend(loc="upper left")
    finish(fig, "bundle_adjustment_advanced", "Local vs. Global bundle adjustment on a growing map",
           "Keyframes arrive one at a time; each triggers a Local BA over its covisible neighbors, and every 8th "
           "also a Global BA (dotted lines); seed 0, script defaults (open 90° arc, no loop closure)")


def fig_saltation_matrix_ekf():
    """saltation_matrix_ekf.py: naive (DR) vs saltation-corrected (Xi) covariance update at each bounce."""
    from state_estimation_simulations.use_numpy import saltation_matrix_ekf as m

    dt, e, g = 0.02, 0.85, 9.81
    pos_std, proc_std, imp_std, i_pos, i_vel, n_mc = 0.03, 0.3, 0.005, 0.1, 0.2, 500
    rng = np.random.default_rng(SEED)
    x_true, z = m.generate_ground_truth_and_data(5.0, dt, e, g, 5.0, [1.0, 0.5], pos_std, rng)
    n = len(x_true) - 1
    init_std = np.array([i_pos] * 3 + [i_vel] * 3)
    x_init = x_true[0] + rng.normal(0.0, 1.0, 6) * init_std
    P_init, R = np.diag(init_std ** 2), pos_std ** 2 * np.eye(3)
    r_naive = np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))
    r_salt = np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))
    x_naive, _ = m.run_ekf_naive(x_init, P_init, z, dt, n, e, g, proc_std, imp_std, R,
                                 detect_time_bias=0.0, detect_time_noise_std=0.0, detect_rng=r_naive)
    x_salt, _ = m.run_ekf_saltation(x_init, P_init, z, dt, n, e, g, proc_std, imp_std, R,
                                    detect_time_bias=0.0, detect_time_noise_std=0.0, detect_rng=r_salt)
    nees_naive, nees_salt = m.run_monte_carlo_consistency(x_true, dt, e, g, proc_std, imp_std, pos_std,
                                                          i_pos, i_vel, n_mc, rng, detect_time_bias=0.0,
                                                          detect_time_noise_std=0.0)
    bounces = m.detect_bounce_ticks(x_true)
    post_n, post_s = [], []
    for tick in bounces:
        post_n.extend(nees_naive[tick + 1:tick + 6])
        post_s.extend(nees_salt[tick + 1:tick + 6])
    print(f"    mean NEES naive {np.mean(nees_naive):.2f}, saltation {np.mean(nees_salt):.2f}; "
          f"post-bounce naive {np.mean(post_n):.2f}, saltation {np.mean(post_s):.2f}")

    t = np.arange(len(x_true)) * dt
    zz = np.asarray(z)
    fig, (ax_h, ax_n) = plt.subplots(1, 2)
    ax_h.plot(t, zz[:, 2], ".", color=PRIOR, ms=3, label="position measurements (σ = 3 cm)")
    ax_h.plot(t, x_true[:, 2] if isinstance(x_true, np.ndarray) else [x[2] for x in x_true],
              color=TRUTH, lw=2.2, label="ground truth")
    ax_h.plot(t, [x[2] for x in x_naive], color=C["red"], lw=1.4, label="EKF, naive bounce update (P⁺ = DR·P·DRᵀ)")
    ax_h.plot(t, [x[2] for x in x_salt], "--", color=C["blue"], lw=1.4, label="EKF, saltation update (P⁺ = Ξ·P·Ξᵀ)")
    for tick in bounces:
        ax_h.axvline(t[tick], color=C["gold"], lw=1, alpha=0.8)
    ax_h.set(title="Height of the bouncing point mass (bounces in yellow)", xlabel="time (s)", ylabel="p_z (m)")
    ax_h.legend(loc="upper right", fontsize=8)

    ax_n.plot(t, nees_naive, color=C["red"], lw=1.3, label=f"naive (mean {np.mean(post_n):.2f} just after bounces)")
    ax_n.plot(t, nees_salt, color=C["blue"], lw=1.3, label=f"saltation (mean {np.mean(post_s):.2f} just after bounces)")
    ax_n.axhline(6.0, color=TRUTH, lw=1, label="consistent filter: 6 (state dimension)")
    for tick in bounces:
        ax_n.axvline(t[tick], color=C["gold"], lw=1, alpha=0.8)
    ax_n.set(yscale="log", title=f"Consistency: NEES averaged over {n_mc} Monte Carlo runs", xlabel="time (s)", ylabel="NEES")
    plain_log_ticks(ax_n)
    ax_n.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=8)
    finish(fig, "saltation_matrix_ekf", "Hybrid system: EKF through ground-contact bounces",
           "Point mass in free fall, restitution e = 0.85; the two EKFs differ only in the covariance update at "
           "each bounce; seed 0, script defaults")


def fig_inchworm_zupt_ekf():
    """inchworm_zupt_ekf.py: never / always / phase-conditional zero-velocity updates."""
    from state_estimation_simulations.use_numpy import inchworm_zupt_ekf as m

    dt, t_a, t_e, t_r, v_e = 0.05, 1.0, 1.0, 0.2, 0.1
    pos_std, proc_std, zupt_std, i_pos, i_vel, n_mc = 0.02, 0.15, 0.01, 0.05, 0.05, 500
    rng = np.random.default_rng(SEED)
    x_true, z, is_anchor, is_cruise = m.generate_ground_truth_and_data(10.0, dt, t_a, t_e, t_r, v_e, pos_std, rng)
    n = len(x_true) - 1
    init_std = np.array([i_pos, i_vel])
    x_init = x_true[0] + rng.normal(0.0, 1.0, 2) * init_std
    P_init, R_pos, R_zupt = np.diag(init_std ** 2), pos_std ** 2, zupt_std ** 2
    ests = {
        "never": m.run_ekf_never_zupt(x_init, P_init, z, dt, n, proc_std, R_pos, R_zupt, is_anchor)[0],
        "always": m.run_ekf_always_zupt(x_init, P_init, z, dt, n, proc_std, R_pos, R_zupt, is_anchor)[0],
        "phase_conditional": m.run_ekf_phase_conditional_zupt(x_init, P_init, z, dt, n, proc_std, R_pos, R_zupt, is_anchor)[0],
    }
    nees = dict(zip(ests, m.run_monte_carlo_consistency(x_true, dt, proc_std, R_pos, R_zupt, is_anchor,
                                                         i_pos, i_vel, n_mc, rng)))
    for k in ests:
        print(f"    {k:<18s} NEES anchor {np.mean(nees[k][is_anchor]):.2f}, cruise {np.mean(nees[k][is_cruise]):.2f}")

    t = np.arange(len(x_true)) * dt
    cols = {"never": PRIOR, "always": C["red"], "phase_conditional": C["blue"]}
    labels = {"never": "never ZUPT", "always": "ZUPT every tick", "phase_conditional": "ZUPT on anchor ticks only"}

    def shade(ax):
        start = None
        for k in range(len(t) + 1):
            on = k < len(t) and is_anchor[k]
            if on and start is None:
                start = t[k]
            elif not on and start is not None:
                ax.axvspan(start, t[k - 1] + dt, color=C["green"], alpha=0.10, lw=0)
                start = None

    fig, (ax_v, ax_n) = plt.subplots(1, 2)
    shade(ax_v)
    ax_v.plot(t, x_true[:, 1], color=TRUTH, lw=2.4, label="true velocity")
    for k, x in ests.items():
        ax_v.plot(t, x[:, 1], color=cols[k], lw=1.3, label=labels[k])
    ax_v.axvspan(0, 0, color=C["green"], alpha=0.25, label="anchor phase (truly v = 0)")
    ax_v.set(title="Velocity: anchor / extend gait (one run)", xlabel="time (s)", ylabel="v (m/s)")
    legend_below(ax_v, ncol=3)

    shade(ax_n)
    for k in ests:
        ax_n.plot(t, nees[k], color=cols[k], lw=1.3,
                  label=f"{labels[k]} (anchor {np.mean(nees[k][is_anchor]):.1f}, cruise {np.mean(nees[k][is_cruise]):.1f})")
    ax_n.axhline(2.0, color=TRUTH, lw=1, label="consistent filter: 2 (state dimension)")
    ax_n.set(yscale="log", title=f"Consistency: NEES averaged over {n_mc} Monte Carlo runs", xlabel="time (s)", ylabel="NEES")
    plain_log_ticks(ax_n)
    legend_below(ax_n, ncol=2)
    finish(fig, "inchworm_zupt_ekf", "Zero-velocity updates on a known anchor/extend schedule",
           "1D crawler, state [p, v]: stationary on anchor, 0.1 m/s cruise on extend; three ZUPT policies share the "
           "same predict step and position measurements; seed 0, script defaults", bottom=0.02)


def fig_inchworm_slip_gate_ekf():
    """inchworm_slip_gate_ekf.py: anchor slip, the pad sensor, velocity estimates and NEES per policy, at the
    matched process noise (0.5 m/s^2), plus a gate-threshold sweep of the slip-window and cruise-window NEES."""
    from state_estimation_simulations.use_numpy import inchworm_slip_gate_ekf as m

    dt, t_a, t_e, t_r, v_e, dur = 0.05, 1.0, 1.0, 0.2, 0.1, 20.0
    pos_std, zupt_std, sens_std, n_mc = 0.02, 0.01, 0.0005, 500
    p_slip, v_slip, t_slip, accel = 0.5, 0.08, 0.6, 0.5
    init_std = np.array([0.05, 0.05])
    d_thr = 3.0 * sens_std
    R = (pos_std ** 2, zupt_std ** 2, (sens_std / dt) ** 2)
    cols = {"never": PRIOR, "schedule": C["blue"], "sensor_gate": C["red"], "schedule_and_sensor": C["purple"],
            "sensor_fused": C["green"]}
    labels = {"never": "never", "schedule": "ZUPT on schedule", "sensor_gate": "ZUPT if |d| <= threshold",
              "schedule_and_sensor": "ZUPT if scheduled and |d| <= threshold",
              "sensor_fused": "d / dt as a velocity measurement"}

    rng = np.random.default_rng(SEED)
    x_true, z, d, is_anchor, is_cruise, is_slip, is_clean = m.generate_ground_truth_and_data(
        dur, dt, t_a, t_e, t_r, v_e, pos_std, sens_std, p_slip, v_slip, t_slip, rng)
    n = len(x_true) - 1
    x_init = x_true[0] + rng.normal(0.0, 1.0, 2) * init_std
    P_init = np.diag(init_std ** 2)
    ests = {k: m.run_ekf(x_init, P_init, z, d, dt, n, accel, *R, d_thr, is_anchor, k)[0] for k in m.GATING_MODES}
    nees = m.run_monte_carlo_consistency(x_true, dt, accel, *R, d_thr, is_anchor, 0.05, 0.05, sens_std, n_mc, rng)
    for k in nees:
        print(f"    {k:<20s} NEES clean {np.mean(nees[k][is_clean]):.2f}, slip {np.mean(nees[k][is_slip]):.2f}, "
              f"cruise {np.mean(nees[k][is_cruise]):.2f}")
    t = np.arange(len(x_true)) * dt
    from state_estimation_simulations.utils import averaged_nees_bounds
    lo, hi = averaged_nees_bounds(2, n_mc)

    def below(ax):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5)

    def shade_slip(ax):
        edges = np.diff(np.concatenate([[False], is_slip, [False]]).astype(int))
        for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            ax.axvspan(t[a], t[b - 1] + dt, color=C["orange"], alpha=0.22, lw=0)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
    ax_d, ax_v, ax_n, ax_s = axes.ravel()

    ax_d.plot(t, d * 1e3, color=C["orange"], lw=1.1, label="pad sensor reading d")
    ax_d.axhspan(-d_thr * 1e3, d_thr * 1e3, color=C["red"], alpha=0.14, label="quiet band: |d| <= threshold")
    shade_slip(ax_d)
    ax_d.set(title="Pad sensor (shaded = slip pulses)", xlabel="time (s)", ylabel="displacement per tick (mm)")
    below(ax_d)

    ax_v.plot(t, x_true[:, 1], color=TRUTH, lw=2.4, label="true velocity")
    for k in ("schedule", "sensor_gate", "schedule_and_sensor"):
        ax_v.plot(t, ests[k][:, 1], color=cols[k], lw=1.2, label=labels[k])
    shade_slip(ax_v)
    ax_v.set(title="Velocity estimate (one run)", xlabel="time (s)", ylabel="v (m/s)")
    below(ax_v)

    for k in m.GATING_MODES:
        ax_n.plot(t, nees[k], color=cols[k], lw=1.3,
                  label=f"{labels[k]} (slip {np.mean(nees[k][is_slip]):.1f})")
    ax_n.axhspan(lo, hi, color=TRUTH, alpha=0.10, lw=0)
    ax_n.axhline(2.0, color=TRUTH, lw=1, label="consistent filter: 2 (state dimension)")
    shade_slip(ax_n)
    ax_n.set(yscale="log", title=f"NEES, {n_mc} runs, assumed {accel} m/s^2", xlabel="time (s)", ylabel="NEES")
    plain_log_ticks(ax_n)
    below(ax_n)

    sigmas = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    sweep = {k: {"slip": [], "cruise": []} for k in ("sensor_gate", "schedule_and_sensor")}
    for g in sigmas:
        res = m.run_monte_carlo_consistency(x_true, dt, accel, *R, g * sens_std, is_anchor, 0.05, 0.05, sens_std,
                                            200, np.random.default_rng(SEED + 1), modes=tuple(sweep))
        for k in sweep:
            sweep[k]["slip"].append(np.mean(res[k][is_slip]))
            sweep[k]["cruise"].append(np.mean(res[k][is_cruise]))
    speeds = [g * sens_std / dt for g in sigmas]
    for k in sweep:
        ax_s.plot(speeds, sweep[k]["slip"], "o-", color=cols[k], lw=1.4, label=f"{labels[k]}: slip window")
        ax_s.plot(speeds, sweep[k]["cruise"], "s--", color=cols[k], lw=1.1, alpha=0.8,
                  label=f"{labels[k]}: cruise window")
    ax_s.axhline(2.0, color=TRUTH, lw=1)
    ax_s.axvline(v_e, color=C["orange"], lw=1, ls=":")
    ax_s.text(v_e * 0.93, 3.0, "cruise speed ", color=C["orange"], va="bottom", ha="right", fontsize=8.5, rotation=90)
    ax_s.set(xscale="log", yscale="log", title="Threshold sweep (200 runs per point)",
             xlabel="gate threshold as a speed: |d| / dt (m/s)", ylabel="NEES")
    plain_log_ticks(ax_s)
    ax_s.set_xticks([0.005, 0.01, 0.02, 0.05, 0.1])
    ax_s.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_s.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    below(ax_s)
    finish(fig, "inchworm_slip_gate_ekf", "Anchor slip: gating ZUPT by schedule vs. by a pad sensor",
           "1D crawler [p, v], slip pulses in half the anchor phases, sensor noise 0.5 mm per tick, "
           "matched process noise; seed 0", bottom=0.02)


def fig_inchworm_two_pad_ekf():
    """inchworm_two_pad_ekf.py: top view, anchored-pad sensor, midpoint error and NEES per policy, at the
    script defaults (same draw order as its main())."""
    from state_estimation_simulations.use_numpy import inchworm_two_pad_ekf as m
    from state_estimation_simulations.utils import averaged_nees_bounds

    dt, dur, t_phase, t_ramp, n_mc = 0.05, 20.0, 1.0, 0.2, 500
    micro, sens = 0.01, 0.0005
    n_steps, n_phases = int(round(dur / dt)), int(np.ceil(dur / t_phase))
    noise = dict(micro_slip=micro, free=0.1, sensor=sens, move=0.02, gyro=0.02, closure=0.001, closure_lat=0.001,
                 d_thr=3.0 * np.hypot(sens, micro * dt))
    init_std = np.array([0.005] * 4 + [0.02])
    P_init = np.diag(init_std ** 2)
    cols = {"never": PRIOR, "schedule": C["blue"], "schedule_and_sensor": C["purple"], "pad_fused": C["green"]}
    labels = {"never": "never (loose anchor)", "schedule": "schedule: anchored = still",
              "schedule_and_sensor": "schedule + sensor veto", "pad_fused": "pad reading fused"}

    rng = np.random.default_rng(SEED)
    episodes = m.draw_slip_episodes(n_phases, t_phase, 0.6, 0.5, rng)
    is_slip, _ = m.window_flags(n_steps, dt, t_phase, episodes)
    sim_args = (n_steps, dt, t_phase, t_ramp, 0.1, 0.15, 0.3, episodes, 0.08, micro, 0.02, sens, 0.001, 0.001)

    def one_trial(r):
        x_true, gyro, d, z, back = m.simulate(*sim_args, r)
        return x_true, gyro, d, z, back, x_true[0] + r.normal(0.0, 1.0, m.STATE_DIM) * init_std

    x_true, gyro, d, z, back, x_init = one_trial(rng)
    ests = {k: m.run_ekf(x_init, P_init, gyro, d, z, back, dt, k, noise)[0] for k in m.POLICIES}
    sums = {k: np.zeros(n_steps + 1) for k in m.POLICIES}
    for _ in range(n_mc):
        xt, g, dd, zz, bb, xi = one_trial(rng)
        for k in m.POLICIES:
            xe, Pe = m.run_ekf(xi, P_init, g, dd, zz, bb, dt, k, noise)
            err = m.state_error(xt, xe)
            sums[k] += [err[i] @ np.linalg.solve(Pe[i], err[i]) for i in range(n_steps + 1)]
    nees = {k: s / n_mc for k, s in sums.items()}
    for k in nees:
        print(f"    {k:<20s} NEES overall {np.mean(nees[k]):.2f}")
    lo, hi = averaged_nees_bounds(m.STATE_DIM, n_mc)
    t = np.arange(n_steps + 1) * dt
    mid = lambda x: 0.5 * (x[:, m.B] + x[:, m.F])  # noqa: E731
    mid_true = mid(x_true)

    def shade_slip(ax):
        for _, a, b in episodes:
            ax.axvspan(a, b, color=C["orange"], alpha=0.22, lw=0)

    def below(ax):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
    ax_top, ax_d, ax_e, ax_n = axes.ravel()

    ax_top.plot(*mid_true.T, color=TRUTH, lw=2.4, label="true body midpoint")
    for k in m.POLICIES:
        ax_top.plot(*mid(ests[k]).T, color=cols[k], lw=1.2, label=labels[k])
    for i in range(0, n_steps + 1, 40):
        ax_top.plot([x_true[i, 0], x_true[i, 2]], [x_true[i, 1], x_true[i, 3]], color=TRUTH, lw=3, alpha=0.35)
    ax_top.set(title="Top view (one run; grey bars = the true body every 2 s)", xlabel="x (m)", ylabel="y (m)",
               aspect="equal")
    below(ax_top)

    ax_d.plot(t[1:], np.linalg.norm(d, axis=1) * 1e3, color=C["orange"], lw=1.1, label="|anchored-pad reading|")
    ax_d.axhline(noise["d_thr"] * 1e3, color=C["red"], lw=1, ls="--", label="veto threshold")
    shade_slip(ax_d)
    ax_d.set(title="Anchored-pad sensor (shaded = slip pulses)", xlabel="time (s)",
             ylabel="displacement per tick (mm)")
    below(ax_d)

    for k in m.POLICIES:
        ax_e.plot(t, np.linalg.norm(mid_true - mid(ests[k]), axis=1) * 1e3, color=cols[k], lw=1.2, label=labels[k])
    shade_slip(ax_e)
    ax_e.set(title="Midpoint position error (one run)", xlabel="time (s)", ylabel="error (mm)")
    below(ax_e)

    for k in m.POLICIES:
        ax_n.plot(t, nees[k], color=cols[k], lw=1.3, label=f"{labels[k]} (mean {np.mean(nees[k]):.1f})")
    ax_n.axhspan(lo, hi, color=TRUTH, alpha=0.10, lw=0)
    ax_n.axhline(m.STATE_DIM, color=TRUTH, lw=1, label=f"consistent filter: {m.STATE_DIM} (state dimension)")
    shade_slip(ax_n)
    ax_n.set(yscale="log", title=f"NEES, {n_mc} runs", xlabel="time (s)", ylabel="NEES")
    plain_log_ticks(ax_n)
    below(ax_n)
    finish(fig, "inchworm_two_pad_ekf", "Two-pad inchworm: trusting the anchored pad without a position fix",
           "planar [back pad, front pad, heading], gyro + kinematic closure, slip pulses in "
           f"{len(episodes)} of {n_phases} phases, no absolute position; seed 0", bottom=0.02)


def fig_inchworm_two_pad_fusion_ekf():
    """inchworm_two_pad_fusion_ekf.py: both pad readings with the faults (one run), midpoint RMSE and NEES per
    variant over the Monte Carlo runs at the script defaults (same draw order as its main()), plus a sweep of
    the moving sensor's noise."""
    from state_estimation_simulations.use_numpy import inchworm_two_pad_fusion_ekf as m
    from state_estimation_simulations.utils import averaged_nees_bounds

    args = m.build_parser().parse_args([])
    rng = np.random.default_rng(SEED)
    setup = m.build_setup(args, rng)
    n_steps, episodes, noise = setup["n_steps"], setup["episodes"], setup["noise"]
    trial = m.simulate_trial(args, setup, rng)
    stats, _ = m.run_monte_carlo(args, setup, args.n_mc_trials, rng)
    for v in m.VARIANTS:
        print(f"    {v:<14s} NEES overall {np.mean(stats[v]['nees']):.2f}")
    lo, hi = averaged_nees_bounds(m.STATE_DIM, args.n_mc_trials)

    sweep_std, sweep_vars, n_sweep = [0.0005, 0.001, 0.002, 0.005], ("random_walk", "moving_sensor", "cross_check"), 100
    sweep = {v: [] for v in sweep_vars}
    for std in sweep_std:
        a = m.build_parser().parse_args(["--moving-sensor-noise-std", str(std)])
        r = np.random.default_rng(SEED)
        setup_a = m.build_setup(a, r)
        m.simulate_trial(a, setup_a, r)  # same draw order as the script's main()
        st, _ = m.run_monte_carlo(a, setup_a, n_sweep, r, variants=sweep_vars)
        for v in sweep_vars:
            sweep[v].append(np.mean(st[v]["nees"]))

    cols = {"random_walk": PRIOR, "encoder_model": C["blue"], "moving_sensor": C["orange"], "both": C["purple"],
            "cross_check": C["green"]}
    labels = {"random_walk": "anchored sensor only", "encoder_model": "+ encoder model",
              "moving_sensor": "+ moving sensor", "both": "+ both, combined", "cross_check": "+ both, cross-checked"}
    t = np.arange(n_steps + 1) * args.dt
    def shade_slip(ax):
        for _, a, b in episodes:
            ax.axvspan(a, b, color=C["gold"], alpha=0.3, lw=0)

    def below(ax):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5, ncol=2)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
    ax_d, ax_e, ax_n, ax_s = axes.ravel()

    s = np.where(trial["back"], 1.0, -1.0)[:, None]
    stroke = np.array([m.commanded_change(trial["e_cmd"][k], trial["z"][k][0]) for k in range(n_steps)])
    ax_d.plot(t[1:], np.linalg.norm(trial["d_a"], axis=1) * 1e3, color=C["orange"], lw=1.1,
              label="|anchored reading|")
    ax_d.plot(t[1:], np.linalg.norm(trial["d_m"] - s * stroke, axis=1) * 1e3, color=C["blue"], lw=1.1,
              label="|moving reading − commanded stroke|")
    ax_d.axhline(noise["d_thr"] * 1e3, color=TRUTH, lw=1, ls="--", label="slip threshold")
    top = 1.05 * max(np.max(np.linalg.norm(trial["d_a"], axis=1)), np.max(np.linalg.norm(trial["d_m"] - s * stroke,
                                                                                     axis=1))) * 1e3
    for pad, kind, a, b, _ in trial["faults"]:
        ax_d.axvspan(a, b, ymin=0.92, ymax=1.0, color=C["red"] if kind == m.SPIKE else PRIOR, lw=0)
    ax_d.fill_between([], [], color=C["red"], label="spike (either sensor)")
    ax_d.fill_between([], [], color=PRIOR, label="dropout (either sensor)")
    ax_d.fill_between([], [], color=C["gold"], alpha=0.5, label="slip pulse")
    shade_slip(ax_d)
    ax_d.set(title="Both pad sensors (one run; top strip = sensor faults)", xlabel="time (s)",
             ylabel="per tick (mm)", ylim=(0, top / 0.9))
    below(ax_d)

    for v in m.VARIANTS:
        ax_e.plot(t, stats[v]["pos_rmse"] * 1e3, color=cols[v], lw=1.2, label=labels[v])
    shade_slip(ax_e)
    ax_e.set(title=f"Midpoint position RMSE, {args.n_mc_trials} runs", xlabel="time (s)", ylabel="RMSE (mm)")
    below(ax_e)

    for v in m.VARIANTS:
        ax_n.plot(t, stats[v]["nees"], color=cols[v], lw=1.3, label=f"{labels[v]} (mean {np.mean(stats[v]['nees']):.1f})")
    ax_n.axhspan(lo, hi, color=TRUTH, alpha=0.10, lw=0)
    ax_n.axhline(m.STATE_DIM, color=TRUTH, lw=1, label=f"consistent filter: {m.STATE_DIM}")
    shade_slip(ax_n)
    ax_n.set(yscale="log", title=f"NEES, {args.n_mc_trials} runs", xlabel="time (s)", ylabel="NEES")
    plain_log_ticks(ax_n)
    below(ax_n)

    x_mm = np.array(sweep_std) * 1e3
    for v in sweep_vars:
        ax_s.plot(x_mm, sweep[v], "o-", color=cols[v], lw=1.5, label=labels[v])
    ax_s.axhline(m.STATE_DIM, color=TRUTH, lw=1, label=f"consistent filter: {m.STATE_DIM}")
    ax_s.set(xscale="log", yscale="log", title=f"Noisier moving sensor ({n_sweep} runs each)",
             xlabel="moving sensor noise (mm per reading)", ylabel="NEES (run average)")
    ax_s.set_xticks(x_mm)
    ax_s.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_s.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    plain_log_ticks(ax_s)
    below(ax_s)
    finish(fig, "inchworm_two_pad_fusion_ekf", "Two-pad inchworm: reading both pad sensors",
           f"slip pulses in {len(episodes)} of {setup['n_phases']} phases, sensor faults on "
           f"{100 * args.p_fault:.0f}% of phases per sensor, 5% stroke error; seed 0", bottom=0.02)


def fig_gait_phase_ekf():
    """gait_phase_ekf.py: stance flags of the blocked leg, touchdown-offset estimates, Monte Carlo NEES per
    approach at the defaults (same draw order as its main()), and NEES vs terrain spread for the two
    acceptance rules (CPG window, phase-EKF sigma gate)."""
    from state_estimation_simulations.use_numpy import gait_phase_ekf as m
    from state_estimation_simulations.utils import averaged_nees_bounds

    args = m.build_parser().parse_args([])
    dt, n_mc = args.dt, args.n_mc_trials
    cols = {"schedule": C["blue"], "detector": C["red"], "cpg": C["purple"], "phase_ekf": C["green"], "oracle": TRUTH}
    labels = {"schedule": "schedule", "detector": "detector", "cpg": "CPG (0.1 s window)",
              "phase_ekf": "phase EKF (3σ gate)", "oracle": "true stance"}

    rng = np.random.default_rng(args.seed)
    x_true, x_init, r, z, flags, info = m.simulate_trial(args, rng)
    mc = m.run_monte_carlo(args, n_mc, rng)
    for k in m.ROWS:
        print(f"    {k:<10s} NEES {np.mean(mc[k]['nees']):.2f}, swing read/trial {mc[k]['swing']:.1f}")
    lo, hi = averaged_nees_bounds(6, n_mc)
    t = np.arange(len(x_true)) * dt
    leg = m.LEG_NAMES.index(args.blocked_leg)

    def below(ax):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5, ncol=2)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
    ax_g, ax_l, ax_n, ax_s = axes.ravel()

    rows = ("oracle", "schedule", "detector", "cpg", "phase_ekf")
    true_leg = flags["oracle"][:, leg]
    for y, a in enumerate(rows[::-1]):
        f = flags[a][:, leg]
        ax_g.fill_between(t, y - 0.35, y + 0.35, where=f, step="post", color=cols[a], alpha=0.75, lw=0)
        ax_g.fill_between(t, y - 0.35, y + 0.35, where=f & ~true_leg, step="post", color=C["gold"], lw=0)
    for (i, up, d, nxt), fk in zip(info["steps"], info["false_ticks"]):
        if i == leg and fk is not None:
            ax_g.plot(fk * dt, 2, "x", color=TRUTH, ms=7, mew=1.6)
    ax_g.axvspan(args.blocked_start, args.blocked_end, color=C["orange"], alpha=0.15, lw=0)
    ax_g.set_yticks(range(len(rows)), [labels[a] for a in rows[::-1]])
    ax_g.set_xlim(args.blocked_start - 3.0, args.blocked_end + 5.5)
    ax_g.set(title=f"Stance flags of leg {args.blocked_leg} (one run)", xlabel="time (s)")
    ax_g.grid(False)
    ax_g.plot([], [], "s", color=C["gold"], label="swinging foot read as planted")
    ax_g.plot([], [], "x", color=TRUTH, label="false trigger (detector row)")
    ax_g.fill_between([], [], color=C["orange"], alpha=0.3, label="blocked window")
    below(ax_g)

    td_t = np.array([d for (_, _, d, _) in info["steps"]]) * dt
    ax_l.plot(td_t, 1e3 * info["tau"], ".", color=PRIOR, ms=3, label="true offset τ of each step")
    ax_l.plot(td_t, 1e3 * info["lag"], color=TRUTH, lw=2, label="true systematic lag")
    for a in ("cpg", "phase_ekf"):
        ax_l.plot(td_t, 1e3 * info["lag_est"][a], color=cols[a], lw=1.2, label=f"{labels[a]} estimate")
    ax_l.fill_between(td_t, 1e3 * (info["lag_est"]["phase_ekf"] - 2 * info["ekf_std"]),
                      1e3 * (info["lag_est"]["phase_ekf"] + 2 * info["ekf_std"]), color=cols["phase_ekf"],
                      alpha=0.15, lw=0, label="phase EKF ±2σ")
    ax_l.set(title="Touchdown offset after the commanded time (one run)", xlabel="time (s)", ylabel="offset (ms)")
    below(ax_l)

    for a in m.ROWS:
        ax_n.plot(t, mc[a]["nees"], color=cols[a], lw=1.3, label=f"{labels[a]} (mean {np.mean(mc[a]['nees']):.1f})")
    ax_n.axhspan(lo, hi, color=TRUTH, alpha=0.10, lw=0)
    ax_n.axhline(6.0, color=TRUTH, lw=1)
    ax_n.set(yscale="log", title=f"Body NEES, {n_mc} runs (consistent: 6)", xlabel="time (s)", ylabel="NEES")
    plain_log_ticks(ax_n)
    below(ax_n)

    terrain = [0.0, 0.03, 0.06, 0.1]
    n_sw = 30
    curves = {"cpg window 0.1 s": [], "phase EKF 3σ": [], "phase EKF 2σ": []}
    for ts in terrain:
        for gate in (3.0, 2.0):
            a2 = m.build_parser().parse_args([])
            a2.terrain_std, a2.ekf_gate = ts, gate
            res = m.run_monte_carlo(a2, n_sw, np.random.default_rng(SEED + 1), rows=("cpg", "phase_ekf"))
            curves[f"phase EKF {gate:.0f}σ"].append(np.mean(res["phase_ekf"]["nees"]))
            if gate == 3.0:
                curves["cpg window 0.1 s"].append(np.mean(res["cpg"]["nees"]))
    styles = {"cpg window 0.1 s": (cols["cpg"], "o-"), "phase EKF 3σ": (cols["phase_ekf"], "s-"),
              "phase EKF 2σ": (cols["phase_ekf"], "s--")}
    for k, v in curves.items():
        print(f"    terrain sweep {k}: " + ", ".join(f"{x:.2f}" for x in v))
        ax_s.plot(1e3 * np.array(terrain), v, styles[k][1], color=styles[k][0], lw=1.4, label=k)
    ax_s.axhline(6.0, color=TRUTH, lw=1)
    ax_s.set(yscale="log", title=f"Acceptance width vs rougher ground ({n_sw} runs/point)",
             xlabel="terrain spread of touchdown times (ms)", ylabel="body NEES")
    plain_log_ticks(ax_s)
    below(ax_s)
    finish(fig, "gait_phase_ekf", "Gait phase: which feet are on the ground, when touchdowns are uncertain",
           "planar hexapod, tripod gait, known liftoffs; touchdowns lag (to 120 ms), spread 30 ms, one blocked leg, "
           "false triggers on 10% of steps; seed 0", bottom=0.02)


def fig_inchworm_zupt_zaru_ekf():
    """inchworm_zupt_zaru_ekf.py: heading and NEES under ZUPT + ZARU gating, at the script defaults
    (assumed accelerations under-sized) and at the matched noise level (0.5 m/s^2, 1.5 rad/s^2)."""
    from state_estimation_simulations.use_numpy import inchworm_zupt_zaru_ekf as m

    dt, t_a, t_e, t_r, v_e, om_e = 0.05, 1.0, 1.0, 0.2, 0.1, 0.3
    pos_std, zupt_std, zaru_std, n_mc = 0.02, 0.01, 0.02, 500
    init_std = np.array([0.05, 0.05, 0.1, 0.05, 0.05])
    cols = {"never": PRIOR, "always": C["red"], "phase_conditional": C["blue"],
            "phase_conditional_zupt_only": C["orange"]}
    labels = {"never": "never", "always": "ZUPT + ZARU every tick",
              "phase_conditional": "ZUPT + ZARU on anchor ticks",
              "phase_conditional_zupt_only": "ZUPT only on anchor ticks (no ZARU)"}
    regimes = [("script defaults", 0.15, 0.45), ("matched noise", 0.5, 1.5)]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    for row, (name, sa, sal) in enumerate(regimes):
        rng = np.random.default_rng(SEED)
        x_true, z, is_anchor, is_cruise = m.generate_ground_truth_and_data(
            10.0, dt, t_a, t_e, t_r, v_e, om_e, pos_std, rng)
        n = len(x_true) - 1
        x_init = x_true[0] + rng.normal(0.0, 1.0, 5) * init_std
        P_init = np.diag(init_std ** 2)
        R = (pos_std ** 2, zupt_std ** 2, zaru_std ** 2)
        ests = {k: m.run_ekf(x_init, P_init, z, dt, n, sa, sal, *R, is_anchor, k)[0] for k in m.GATING_MODES}
        nees = m.run_monte_carlo_consistency(x_true, dt, sa, sal, *R, is_anchor, init_std, n_mc, rng)
        for k in ests:
            print(f"    [{row}] {k:<28s} NEES anchor {np.mean(nees[k][is_anchor]):.2f}, "
                  f"cruise {np.mean(nees[k][is_cruise]):.2f}")
        t = np.arange(len(x_true)) * dt
        ax_h, ax_n = axes[row]
        for ax in (ax_h, ax_n):
            start = None
            for k in range(len(t) + 1):
                on = k < len(t) and is_anchor[k]
                if on and start is None:
                    start = t[k]
                elif not on and start is not None:
                    ax.axvspan(start, t[k - 1] + dt, color=C["green"], alpha=0.10, lw=0)
                    start = None
        ax_h.plot(t, x_true[:, 2], color=TRUTH, lw=2.4, label="true heading")
        for k, x in ests.items():
            ax_h.plot(t, x[:, 2], color=cols[k], lw=1.3, label=labels[k])
        ax_h.set(title=f"Heading (one run), {name}", xlabel="time (s)", ylabel="theta (rad)")
        for k in ests:
            ax_n.plot(t, nees[k], color=cols[k], lw=1.3,
                      label=f"{labels[k]} (anchor {np.mean(nees[k][is_anchor]):.1f}, "
                            f"cruise {np.mean(nees[k][is_cruise]):.1f})")
        ax_n.axhline(5.0, color=TRUTH, lw=1, label="consistent filter: 5 (state dimension)")
        ax_n.set(yscale="log", title=f"NEES, {n_mc} runs; assumed {sa} m/s^2, {sal} rad/s^2", xlabel="time (s)",
                 ylabel="NEES")
        plain_log_ticks(ax_n)
        if row == 0:
            legend_below(ax_h, ncol=2)
            legend_below(ax_n, ncol=1)
    finish(fig, "inchworm_zupt_zaru_ekf", "Heading and ZARU on a known anchor/extend schedule",
           "planar crawler [px, py, theta, v, omega], position-only measurements, turns 0.3 rad/s on extend; "
           "matched = assumed accelerations equal the true ramps; seed 0", bottom=0.02)


def fig_friction_anisotropic_ekf():
    """friction_anisotropic_ekf.py: isotropic / fixed-anisotropic / heading-aware process noise."""
    from state_estimation_simulations.use_numpy import friction_anisotropic_ekf as m

    duration, dt, v_cmd = 20.0, 0.05, 0.2
    omega = 2.0 * np.pi / duration
    s_grip, s_slip, s_th, pos_std, i_pos, i_th, n_mc = 0.02, 0.1, 0.01, 0.02, 0.05, 0.05, 500
    rng = np.random.default_rng(SEED)
    x_true, z = m.generate_ground_truth_and_data(duration, dt, v_cmd, omega, s_grip, s_slip, pos_std, rng)
    n = len(x_true) - 1
    theta_ref = x_true[0, 2]
    init_std = np.array([i_pos, i_pos, i_th])
    x_init = x_true[0] + rng.normal(0.0, 1.0, 3) * init_std
    P_init, R_pos = np.diag(init_std ** 2), pos_std ** 2 * np.eye(2)
    common = (x_init, P_init, z, dt, n, v_cmd, omega, s_grip, s_slip, s_th, R_pos)
    ests = {"isotropic": m.run_ekf_isotropic(*common)[0],
            "fixed_anisotropic": m.run_ekf_fixed_anisotropic(*common, theta_ref)[0],
            "heading_aware": m.run_ekf_heading_aware(*common)[0]}
    nees = dict(zip(ests, m.run_monte_carlo_consistency(x_true, dt, v_cmd, omega, s_grip, s_slip, s_th, R_pos,
                                                         theta_ref, i_pos, i_th, n_mc, rng)))
    for k in ests:
        print(f"    {k:<18s} mean NEES {np.mean(nees[k]):.2f}")

    cols = {"isotropic": PRIOR, "fixed_anisotropic": C["red"], "heading_aware": C["blue"]}
    labels = {"isotropic": "isotropic Q", "fixed_anisotropic": "anisotropic Q, frozen at t = 0",
              "heading_aware": "anisotropic Q, rotated with heading"}

    fig, (ax_p, ax_n) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.2]})
    ax_p.plot(x_true[:, 0], x_true[:, 1], color=TRUTH, lw=2.2, label="true path (exact circle, slip added)")
    scale = 60.0
    for k in range(0, n, n // 8):
        Q = m.anisotropic_Q_pos(x_true[k, 2], s_grip, s_slip, dt)
        vals, vecs = np.linalg.eigh(Q)
        ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        ax_p.add_patch(Ellipse(x_true[k, 0:2], 2 * scale * np.sqrt(vals[1]), 2 * scale * np.sqrt(vals[0]),
                               angle=ang, fc=C["orange"], ec=C["orange"], alpha=0.35))
        h = x_true[k, 2]
        ax_p.annotate("", xy=(x_true[k, 0] + 0.13 * np.cos(h), x_true[k, 1] + 0.13 * np.sin(h)),
                      xytext=tuple(x_true[k, 0:2]), arrowprops=dict(arrowstyle="-|>", color=TRUTH, lw=1.4))
    ax_p.add_patch(Ellipse((0, 0), 0, 0, fc=C["orange"], alpha=0.35,
                           label=f"per-step slip covariance (1σ, ×{scale:.0f}): wide = slip axis"))
    ax_p.plot([], [], color=TRUTH, marker=r"$\rightarrow$", ls="", ms=11, label="heading (grip axis)")
    ax_p.set(title="True slip noise turns with the robot", xlabel="p_x (m)", ylabel="p_y (m)")
    ax_p.axis("equal")
    legend_below(ax_p, ncol=1)

    t = np.arange(len(x_true)) * dt
    for k in ests:
        ax_n.plot(t, nees[k], color=cols[k], lw=1.3, label=f"{labels[k]} (mean {np.mean(nees[k]):.1f})")
    ax_n.axhline(3.0, color=TRUTH, lw=1, label="consistent filter: 3 (state dimension)")
    for frac, txt in [(0.25, "90°"), (0.5, "180°"), (0.75, "270°")]:
        ax_n.axvline(frac * duration, color=PRIOR, ls=":", lw=0.9)
        ax_n.text(frac * duration + 0.15, 0.03, f"turned {txt}", transform=ax_n.get_xaxis_transform(), fontsize=8, color="#555555")
    ax_n.set(yscale="log", title=f"Consistency: NEES averaged over {n_mc} Monte Carlo runs", xlabel="time (s)", ylabel="NEES")
    plain_log_ticks(ax_n)
    legend_below(ax_n, ncol=2)
    finish(fig, "friction_anisotropic_ekf", "Anisotropic slip: process noise must turn with the robot",
           "Unicycle on one full circle; slip std 0.02 m/s along the grip (forward) axis, 0.1 m/s along the slip "
           "(lateral) axis; three Q policies; seed 0, script defaults", bottom=0.02)


def fig_imu_integration_comparison():
    """imu_integration_comparison.py: naive Euler-angle vs SO(3) exp-map integration of the same noisy IMU."""
    from state_estimation_simulations.use_numpy import imu_integration_comparison as m

    t, rot_naive, rot_exp, pos_naive, pos_exp = m.run_simulation(20.0, 0.005, 0.02, 0.05, SEED)
    print(f"    final rot error naive {rot_naive[-1]:.3f} deg, exp {rot_exp[-1]:.3f} deg; "
          f"final pos error naive {pos_naive[-1]:.4f} m, exp {pos_exp[-1]:.4f} m")

    fig, (ax_r, ax_p) = plt.subplots(1, 2)
    for ax, naive, exp, unit, ylabel, title in [
            (ax_r, rot_naive, rot_exp, "°", "rotation error (deg)", "Attitude error"),
            (ax_p, pos_naive, pos_exp, " m", "position error (m)", "Position error")]:
        fmt = (lambda v: f"{v:.2f}{unit}") if unit == "°" else (lambda v: f"{v:.3f}{unit}")
        ax.plot(t, naive, color=C["red"], lw=1.6, label=f"naive: Euler angles added like a vector (final {fmt(naive[-1])})")
        ax.plot(t, exp, color=C["blue"], lw=1.6, label=f"exp-map: R ← R·Exp(ω·dt) on SO(3) (final {fmt(exp[-1])})")
        ax.set(yscale="log", title=title, xlabel="time (s)", ylabel=ylabel)
        ax.legend(loc="lower right", fontsize=8.5)
    finish(fig, "imu_integration_comparison", "IMU integration: Euler angles vs. the SO(3) exponential map",
           "Same noisy gyro/velocity samples (200 Hz, 20 s) integrated two ways, each against its own noise-free "
           "reference; seed 0, script defaults")


def fig_pose_graph_incremental():
    """pose_graph_incremental.py: batch re-solve per node vs incremental square-root SAM (Givens QR updates)."""
    from state_estimation_simulations.use_numpy import pose_graph_incremental as m
    from state_estimation_simulations.utils import measure_performance

    rng = np.random.default_rng(SEED)
    gt = m.generate_ground_truth_trajectory(2.0, 16)
    n = len(gt)
    odo, loop = m.simulate_noisy_edges(gt, 0.05, 0.01, 0.5, rng)
    T_dr = m.run_dead_reckoning(gt[0], odo)
    info = np.eye(6)
    (T_batch, relin_batch), t_batch, _ = quiet(measure_performance, m.run_batch_streaming, gt, odo, loop, info,
                                               0.01, 1e-6, 10, n_steps=n)
    (T_incr, relin_incr), t_incr, _ = quiet(measure_performance, m.run_incremental_pose_graph, gt, odo, loop, info,
                                            1e6, 8, 1e-6, 10, n_steps=n)
    errs = {name: m.pose_errors(gt, T)[1] for name, T in [("odo", T_dr), ("batch", T_batch), ("incr", T_incr)]}
    gap = max(np.linalg.norm(a[0:3, 3] - b[0:3, 3]) for a, b in zip(T_batch, T_incr))
    print(f"    RMS pos: odometry {rms(errs['odo']):.4f}, batch {rms(errs['batch']):.4f}, incremental "
          f"{rms(errs['incr']):.4f} m; max batch-incremental gap {gap:.2e} m")
    print(f"    avg per node: batch {t_batch * 1e3:.3f} ms, incremental {t_incr * 1e3:.3f} ms; "
          f"relinearizations {relin_batch} vs {relin_incr}")

    def closed(T_list):
        x, y = xy_of(T_list)
        return np.append(x, x[0]), np.append(y, y[0])

    fig, (ax_xy, ax_c) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.1, 1]})
    ax_xy.plot(*closed(gt), "-", color=TRUTH, lw=2.2, label="ground truth")
    ax_xy.plot(*xy_of(T_dr), "--", color=PRIOR, lw=1.6, label=f"odometry only (RMSE {rms(errs['odo']) * 100:.1f} cm)")
    ax_xy.plot(*xy_of(T_batch), "o-", color=C["blue"], ms=3, lw=1.4,
               label=f"batch re-solve per node (RMSE {rms(errs['batch']) * 100:.1f} cm)")
    ax_xy.plot(*xy_of(T_incr), "x:", color=C["orange"], ms=5, lw=1.4,
               label=f"incremental sqrt-SAM (RMSE {rms(errs['incr']) * 100:.1f} cm)")
    ax_xy.set(title=f"{n} poses streamed in one at a time, loop closed at the end", xlabel="x (m)", ylabel="y (m)")
    ax_xy.axis("equal")
    legend_below(ax_xy, ncol=2)

    names = ["batch re-solve\nper node", "incremental\nsqrt-SAM"]
    bars = ax_c.bar(names, [t_batch * 1e3, t_incr * 1e3], color=[C["blue"], C["orange"]], width=0.55)
    for b, relin in zip(bars, [relin_batch, relin_incr]):
        ax_c.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.1f} ms\n{relin} full solves",
                  ha="center", va="bottom", fontsize=9)
    ax_c.set(title="Average cost per streamed node (this machine)", ylabel="wall-clock time per node (ms)",
             ylim=(0, t_batch * 1e3 * 1.3))
    ax_c.text(0.98, 0.97, f"final answers differ by at most {gap * 1e3:.2g} mm", transform=ax_c.transAxes,
              ha="right", va="top", fontsize=9, color="#555555")
    finish(fig, "pose_graph_incremental", "Incremental vs. batch pose-graph solving",
           "Square loop with 16 nodes per side; incremental = Givens-rotation QR row updates, full relinearization "
           "every 8 nodes and at the loop closure; seed 0, script defaults", bottom=0.02)


def fig_bayes_tree_construction():
    """bayes_tree_construction.py: affected region of an odometry edge vs the loop-closure edge."""
    from state_estimation_simulations.use_numpy import bayes_tree_construction as m
    from state_estimation_simulations.utils import symbolic_eliminate, bayes_tree_affected_path

    n = len(m.generate_ground_truth_trajectory(side_length=2.0, nodes_per_side=4))
    odom, loop = m.ring_edges(n)
    separator, parent = symbolic_eliminate(n, odom + [loop], list(range(n)))
    root = next(v for v, p in parent.items() if p is None)
    children = {}
    for v, p in parent.items():
        if p is not None:
            children.setdefault(p, []).append(v)
    pos = m.layout_tree(children, root)
    aff_odom = bayes_tree_affected_path(parent, odom[-1])
    aff_loop = bayes_tree_affected_path(parent, loop)
    print(f"    affected: odometry edge {len(aff_odom)}/{n}, loop closure {len(aff_loop)}/{n}; "
          f"largest separator {max(len(s) for s in separator.values())}")

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(13, 5.2))
    m.plot_scenario(ax_a, parent, pos, aff_odom, odom[-1], f"New odometry edge x{odom[-1][0]}–x{odom[-1][1]}",
                    160 - 2 * n, True)
    m.plot_scenario(ax_b, parent, pos, aff_loop, loop, f"Loop-closure edge x{loop[0]}–x{loop[1]}", 160 - 2 * n, True)
    for ax in (ax_a, ax_b):
        ax.title.set_fontsize(10.5)
    ax_b.text(0.0, -0.18, f"root = x{root} (eliminated last) · stars = variables the new factor connects · "
              "red = variables that must be re-eliminated", transform=ax_b.transAxes, fontsize=9, color="#555555")
    finish(fig, "bayes_tree_construction", "Bayes-tree affected region: odometry vs. loop closure",
           f"Elimination tree of a {n}-node ring pose graph under oldest-first elimination (no reordering); "
           "script defaults")


def fig_pnp_estimation():
    """pnp_estimation.py: linear DLT initial guess vs Gauss-Newton refinement of a camera pose."""
    from state_estimation_simulations.use_numpy import pnp_estimation as m

    rng = np.random.default_rng(SEED)
    K = (800.0, 800.0, 320.0, 240.0)
    T_true, P_list = m.generate_scene(20, K, rng)
    z_list = [m.camera_project(T_true, P, K)[0] + rng.normal(0.0, 1.0, 2) for P in P_list]
    T_dlt = m.linear_pnp_dlt(P_list, z_list, K)
    T_gn = m.refine_pose_gn(T_dlt, P_list, z_list, K, 1e-8, 20)
    rows = []
    for name, T in [("linear DLT", T_dlt), ("Gauss-Newton refined", T_gn)]:
        rot, pos = m.pose_error(T, T_true)
        proj = np.array([m.camera_project(T, P, K)[0] for P in P_list])
        reproj = rms(np.linalg.norm(proj - np.array(z_list), axis=1))
        rows.append((name, np.degrees(rot), pos, reproj, proj))
        print(f"    {name:<22s} rot {np.degrees(rot):.3f} deg, pos {pos:.4f} m, reprojection RMS {reproj:.3f} px")

    z = np.array(z_list)
    fig, (ax_i, ax_b) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.15, 1]})
    ax_i.add_patch(plt.Rectangle((0, 0), 640, 480, fill=False, ec=PRIOR, lw=1.2))
    mag = 20
    ax_i.scatter(z[:, 0], z[:, 1], s=28, color=TRUTH, zorder=3, label="observed pixels (1 px noise)")
    for (name, _, _, reproj, proj), col in zip(rows, [C["orange"], C["blue"]]):
        d = (proj - z) * mag
        ax_i.quiver(z[:, 0], z[:, 1], d[:, 0], d[:, 1], angles="xy", scale_units="xy", scale=1, color=col,
                    width=0.005, label=f"reprojection residual ×{mag}, {name} pose (RMSE {reproj:.2f} px)")
    outside = int(np.sum((z[:, 0] < 0) | (z[:, 0] > 640) | (z[:, 1] < 0) | (z[:, 1] > 480)))
    print(f"    {outside} of {len(z)} observed pixels fall outside the 640x480 frame")
    frame_note = f", {outside} outside the 640×480 frame (gray)" if outside else " in the 640×480 frame (gray)"
    ax_i.set(title=f"Image plane: {len(P_list)} matches{frame_note}", xlabel="u (px)",
             ylabel="v (px)", xlim=(min(-30, z[:, 0].min() - 60), max(670, z[:, 0].max() + 60)),
             ylim=(max(510, z[:, 1].max() + 60), min(-30, z[:, 1].min() - 60)))
    ax_i.set_aspect("equal")
    ax_i.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8.5)

    x = np.arange(3)
    w = 0.36
    for k, ((name, rot, pos, reproj, _), col) in enumerate(zip(rows, [C["orange"], C["blue"]])):
        ax_b.bar(x + (k - 0.5) * w, [rot, pos * 100, reproj], w, color=col, label=name)
    ax_b.set_xticks(x, ["rotation error\n(deg)", "position error\n(cm)", "reprojection RMSE\n(px)"])
    ax_b.set(title="Closed-form guess vs. refined pose")
    ax_b.legend(loc="upper left")
    finish(fig, "pnp_estimation", "Perspective-n-Point: camera pose from known 3D–2D matches",
           "Linear DLT (closed form, SVD) as the initial guess, then Gauss-Newton on the reprojection error with "
           "T ← T·Exp(δ); seed 0, script defaults", bottom=0.02)


def fig_sliding_window_marginalization():
    """sliding_window_marginalization.py: bounded sliding window vs full batch re-solve, over trajectory length."""
    from state_estimation_simulations.use_numpy import sliding_window_marginalization as m
    from state_estimation_simulations.utils import measure_performance

    rng = np.random.default_rng(SEED)
    info, window = np.eye(6), 10
    res = []
    for nps in [2, 4, 8, 16, 32, 64]:
        gt = m.generate_ground_truth_trajectory(2.0, nodes_per_side=nps)
        n = len(gt)
        odo, _ = m.simulate_noisy_edges(gt, 0.05, 0.01, loop_noise_scale=1.0, rng=rng)
        (x_b, dof_b), t_b, _ = quiet(measure_performance, m.run_full_batch_growing, gt, odo, info, 1e-6, 10, n_steps=n)
        (x_w, dof_w), t_w, _ = quiet(measure_performance, m.run_sliding_window_pose_graph, gt, odo, info, window, 1e6,
                                     1e-6, 10, n_steps=n)
        res.append((n, dof_b, dof_w, t_b, t_w, rms(m.pose_errors(gt, x_b)[1]), rms(m.pose_errors(gt, x_w)[1])))
        print(f"    n_poses {n:4d}: dof {dof_b}/{dof_w}, {t_b * 1e6:.1f}/{t_w * 1e6:.1f} us/step, "
              f"RMS {res[-1][5]:.4f}/{res[-1][6]:.4f} m")
    n, dof_b, dof_w, t_b, t_w, e_b, e_w = map(np.array, zip(*res))

    fig, (ax_m, ax_t, ax_e) = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, yb, yw, ylabel, title in [
            (ax_m, dof_b ** 2 * 8 / 1024, dof_w ** 2 * 8 / 1024, "largest information matrix (KB)", "Memory of the solve"),
            (ax_t, t_b * 1e3, t_w * 1e3, "average time per new pose (ms, this machine)", "Time per new pose")]:
        ax.plot(n, yb, "o-", color=C["red"], lw=1.8, label="full batch re-solve (unbounded)")
        ax.plot(n, yw, "o-", color=C["blue"], lw=1.8, label=f"sliding window of {window} poses (bounded)")
        ax.set(xscale="log", yscale="log", title=title, xlabel="trajectory length (poses)", ylabel=ylabel)
        ax.legend(loc="upper left", fontsize=8.5)
    ax_e.plot(n, e_b * 100, "o-", color=C["red"], lw=1.8, label="full batch re-solve")
    ax_e.plot(n, e_w * 100, "o--", color=C["blue"], lw=1.8, label="sliding window")
    ax_e.set(xscale="log", title="Accuracy (no loop closures)", xlabel="trajectory length (poses)",
             ylabel="position RMSE (cm)")
    ax_e.legend(loc="upper left", fontsize=8.5)
    finish(fig, "sliding_window_marginalization", "Sliding-window marginalization: bounded cost on a growing trajectory",
           f"Odometry-only square path swept from {n[0]} to {n[-1]} poses; the window keeps {window} poses and folds "
           "older ones into a prior; seed 0, script defaults")


# --------------------------------------------------------------------------- concept illustrations
# Setup/mechanism figures for the top of each hybrid-filtering doc (no estimator runs). Same rule as
# above: every curve and number comes from the scripts' own functions at their defaults; the only
# non-data choice is the friction ellipses' visual magnification, which the figure states.

def cov_ellipse(ax, center, cov, scale, **kw):
    """Draws the 1-sigma ellipse of a 2x2 covariance, magnified by `scale`."""
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(center, 2 * scale * np.sqrt(vals[1]), 2 * scale * np.sqrt(vals[0]),
                         angle=angle, **kw))


def arrow(ax, p, q, color, lw=1.6, style="-|>", ms=12, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, **kw))


def fig_friction_anisotropic_ekf_concept():
    """friction_anisotropic_ekf.py: the slip ellipse is body-fixed, so it turns with the robot."""
    from state_estimation_simulations.use_numpy import friction_anisotropic_ekf as m

    duration, dt, v_cmd = 20.0, 0.05, 0.2
    omega = 2.0 * np.pi / duration  # script default: exactly one loop over the run
    s_grip, s_slip = 0.02, 0.1
    r = v_cmd / omega

    # commanded arc, from the script's own exact propagator, starting at the script's x0 = 0
    n = int(duration / dt)
    x = np.zeros(3)
    path = [x]
    for _ in range(n):
        x, _ = m.exact_arc_step(x, v_cmd, omega, dt)
        path.append(x)
    path = np.array(path)
    theta_ref = path[0, 2]  # what fixed_anisotropic is stuck with (0.0)

    fig, (ax_b, ax_w, ax_q) = plt.subplots(1, 3, figsize=(15, 5.6),
                                           gridspec_kw={"width_ratios": [0.8, 1.25, 1.1]})

    # (a) body frame: the pad is part of the robot
    ax_b.set_aspect("equal")
    ax_b.add_patch(Rectangle((-0.55, -0.35), 1.1, 0.7, fc="#e8eef6", ec="#5b6b7f", lw=1.4))
    for k in np.linspace(-0.45, 0.45, 10):  # scales run along the body axis
        ax_b.plot([-0.45, 0.45], [k * 0.62, k * 0.62], color="#aab7c6", lw=0.8)
    Qb = np.diag([s_grip ** 2, s_slip ** 2])
    cov_ellipse(ax_b, (0, 0), Qb, 5.0, fc=C["orange"], alpha=0.35, ec=C["orange"], lw=1.6)
    arrow(ax_b, (0, 0), (0.95, 0), TRUTH, lw=2)
    ax_b.text(0.97, 0.05, "forward\n(heading θ)", fontsize=9, va="bottom")
    arrow(ax_b, (-0.62, -0.72), (0.62, -0.72), C["green"], style="<|-|>")
    ax_b.text(0, -0.84, f"grip axis: σ_grip = {s_grip}", color=C["green"], ha="center", va="top", fontsize=9.5)
    arrow(ax_b, (-0.75, -0.55), (-0.75, 0.55), C["red"], style="<|-|>")
    ax_b.text(-0.82, 0, f"slip axis: σ_slip = {s_slip}", color=C["red"], rotation=90, ha="right",
              va="center", fontsize=9.5)
    ax_b.set(xlim=(-1.25, 1.6), ylim=(-1.45, 0.9), title="(a) Body frame: pad is part of the robot")
    ax_b.axis("off")
    ax_b.text(0.02, -0.02, "slip noise is drawn in this frame\n→ its ellipse is fixed to the body",
              transform=ax_b.transAxes, fontsize=9, color="#555555")

    # (b) world frame: one loop, true slip ellipse turns, the fixed one does not
    ax_w.set_aspect("equal")
    ax_w.plot(path[:, 0], path[:, 1], color=TRUTH, lw=1.6, label=f"commanded arc (r = v/ω = {r:.3f} m)")
    ell_scale = 0.9  # visual magnification (1-sigma of slip speed, m/s → drawn as m)
    for t_s in np.arange(0.0, duration, 2.5):
        k = int(round(t_s / dt))
        p, th = path[k, 0:2], path[k, 2]
        cov_ellipse(ax_w, p, m.anisotropic_Q_pos(theta_ref, s_grip, s_slip, 1.0), ell_scale,
                    fc="none", ec=C["red"], lw=1.3, ls="--")
        cov_ellipse(ax_w, p, m.anisotropic_Q_pos(th, s_grip, s_slip, 1.0), ell_scale,
                    fc=C["orange"], alpha=0.45, ec=C["orange"], lw=1.2)
        arrow(ax_w, p, p + 0.13 * np.array([np.cos(th), np.sin(th)]), TRUTH, lw=1.4, ms=9)
        ax_w.text(p[0] + 0.07 * np.cos(th - np.pi / 2) * 2.2, p[1] + 0.07 * np.sin(th - np.pi / 2) * 2.2,
                  f"{np.degrees(th):.0f}°", fontsize=8, ha="center", va="center", color="#444444")
    ax_w.plot([], [], color=C["orange"], lw=6, alpha=0.5, label="true slip ellipse (turns with heading)")
    ax_w.plot([], [], color=C["red"], ls="--", label="fixed_anisotropic Q (stuck at θ₀ = 0°)")
    ax_w.set(title="(b) World frame: one full loop, ω = 2π/T", xlabel="x (m)", ylabel="y (m)",
             xlim=(-0.95, 0.95), ylim=(-0.25, 1.5))
    ax_w.legend(loc="center", fontsize=7.5, frameon=False)

    # (c) the three q_policies at heading 90°, where the fixed one is worst
    k90 = int(round((np.pi / 2) / omega / dt))
    th90 = path[k90, 2]
    true_cov = m.anisotropic_Q_pos(th90, s_grip, s_slip, 1.0)
    policies = [
        ("isotropic", m.isotropic_Q_pos(s_grip, s_slip, 1.0), PRIOR, "same total variance,\nno direction"),
        ("fixed_anisotropic", m.anisotropic_Q_pos(theta_ref, s_grip, s_slip, 1.0), C["red"],
         "right shape,\nrotated 90° wrong"),
        ("heading_aware", m.anisotropic_Q_pos(th90, s_grip, s_slip, 1.0), C["blue"],
         "right shape,\nright orientation"),
    ]
    ax_q.set_aspect("equal")
    for i, (name, Q, col, note) in enumerate(policies):
        cx = i * 0.3
        cov_ellipse(ax_q, (cx, 0), true_cov, 1.2, fc=C["orange"], alpha=0.35, ec="none")
        cov_ellipse(ax_q, (cx, 0), Q, 1.2, fc="none", ec=col, lw=2)
        arrow(ax_q, (cx, -0.02), (cx, 0.06), TRUTH, lw=1.2, ms=8)
        ax_q.text(cx, 0.17, name, ha="center", fontsize=9, fontweight="bold", color=col)
        ax_q.text(cx, -0.17, note, ha="center", va="top", fontsize=8.5, color="#444444")
    ax_q.set(xlim=(-0.15, 0.75), ylim=(-0.38, 0.22),
             title=f"(c) Filter Q at heading {np.degrees(th90):.0f}° (t = {k90 * dt:.0f} s)")
    ax_q.axis("off")
    ax_q.text(0.0, -0.02, "filled: true slip covariance    outline: the filter's Q\n"
              "worst mismatch at 90° / 270°, aligned again at 180°",
              transform=ax_q.transAxes, fontsize=8.5, color="#555555")

    finish(fig, "friction_anisotropic_ekf_concept", "Friction-anisotropic pad: the noise ellipse turns with the robot",
              f"friction_anisotropic_ekf.py defaults: v = {v_cmd} m/s, ω = 2π/{duration:.0f} s, "
              f"σ_grip = {s_grip}, σ_slip = {s_slip} m/s. Ellipses drawn at 1σ, magnified ×{ell_scale} "
              f"in (b); the {s_slip / s_grip:.0f}:1 axis ratio is exact.", bottom=0.03)


def fig_inchworm_zupt_ekf_concept():
    """inchworm_zupt_ekf.py: known anchor/extend schedule, and when each variant applies ZUPT."""
    from state_estimation_simulations.use_numpy import inchworm_zupt_ekf as m

    dt, t_a, t_e, t_r, v_e = 0.05, 1.0, 1.0, 0.2, 0.1
    duration = 6.0  # 3 gait cycles of the default schedule (the script runs 10 s)
    x_true, _, is_anchor, _ = m.generate_ground_truth_and_data(duration, dt, t_a, t_e, t_r, v_e, 0.0,
                                                               np.random.default_rng(0))
    t = np.arange(len(x_true)) * dt
    tf = np.linspace(0, duration, 1201)
    v_fine = np.array([m.true_velocity(s, t_a, t_e, t_r, v_e) for s in tf])
    p_fine = np.concatenate([[0.0], np.cumsum(0.5 * (v_fine[1:] + v_fine[:-1]) * np.diff(tf))])

    fig, axes = plt.subplots(3, 1, figsize=(13, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.0, 0.85]})
    ax_p, ax_v, ax_z = axes

    def phases(ax, label=False):
        period = t_a + t_e
        for c0 in np.arange(0, duration, period):
            ax.axvspan(c0, c0 + t_a, color=C["green"], alpha=0.12, lw=0)
            for a, b in [(c0 + t_a, c0 + t_a + t_r), (c0 + period - t_r, c0 + period)]:
                ax.axvspan(a, b, color=C["gold"], alpha=0.18, lw=0)
            if label:
                ax.text(c0 + t_a / 2, 1.02, "ANCHOR", transform=ax.get_xaxis_transform(),
                        ha="center", fontsize=9, color=C["green"], fontweight="bold")
                ax.text(c0 + t_a + t_e / 2, 1.02, "EXTEND", transform=ax.get_xaxis_transform(),
                        ha="center", fontsize=9, color="#a67c00", fontweight="bold")

    phases(ax_p, label=True)
    ax_p.plot(tf, p_fine, color=TRUTH, lw=2.2)
    ax_p.set(ylabel="p (m)", title="")
    ax_p.text(0.005, 0.8, "position: a staircase - it holds its new value while anchored",
              transform=ax_p.transAxes, fontsize=9.5, color="#444444")

    phases(ax_v)
    ax_v.plot(tf, v_fine, color=TRUTH, lw=2.2)
    ax_v.set(ylabel="v (m/s)", ylim=(-0.02, 0.17))
    c0 = t_a
    ax_v.annotate("", (c0, 0.113), (c0 + t_r, 0.113), arrowprops=dict(arrowstyle="<->", lw=1))
    ax_v.text(c0 + t_r + 0.03, 0.111, f"t_ramp = {t_r} s", ha="left", fontsize=8.5)
    ax_v.annotate("", (c0, 0.145), (c0 + t_e, 0.145), arrowprops=dict(arrowstyle="<->", lw=1))
    ax_v.text(c0 + t_e / 2, 0.149, f"t_extend = {t_e} s", ha="center", fontsize=8.5)
    ax_v.annotate("", (0, 0.145), (t_a, 0.145), arrowprops=dict(arrowstyle="<->", lw=1))
    ax_v.text(t_a / 2, 0.149, f"t_anchor = {t_a} s", ha="center", fontsize=8.5)
    ax_v.text(c0 + t_e / 2, v_e - 0.012, f"v_extend\n= {v_e} m/s", ha="center", va="top", fontsize=8.5)
    ax_v.set_title("velocity: a continuous trapezoid - ramps, no instantaneous jump", loc="left",
                   fontsize=10, fontweight="normal")

    phases(ax_z)
    rows = [("never", np.zeros_like(is_anchor), PRIOR),
            ("always", np.ones_like(is_anchor), C["red"]),
            ("phase_conditional", is_anchor, C["blue"])]
    for i, (name, on, col) in enumerate(rows):
        y = 2 - i
        ax_z.scatter(t[on], np.full(on.sum(), y), marker="|", s=90, color=col, lw=1.4)
        ax_z.text(-0.08, y, name, ha="right", va="center", fontsize=9.5, color=col, fontweight="bold")
    ax_z.text(duration + 0.05, 1, "wrong: tells the filter\nv = 0 while it moves", va="center",
              fontsize=8.5, color=C["red"])
    ax_z.text(duration + 0.05, 0, "right: only when the known\nschedule says anchored", va="center",
              fontsize=8.5, color=C["blue"])
    ax_z.set(ylim=(-0.6, 2.6), yticks=[], xlabel="time (s)", xlim=(0, duration))
    ax_z.set_title(f"ZUPT (v = 0 pseudo-measurement) ticks, dt = {dt} s  "
                   "- all variants also get a position measurement every tick",
                   loc="left", fontsize=10, fontweight="normal")
    ax_z.grid(False)

    for ax in axes:
        ax.set_xlim(0, duration)
    ax_z.axvspan(0, 0, color=C["green"], alpha=0.3, label="anchor: truly v = 0")
    ax_z.axvspan(0, 0, color=C["gold"], alpha=0.35, label="ramp up / down")
    ax_z.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=2)

    finish(fig, "inchworm_zupt_ekf_concept", "Inchworm gait: a known anchor/extend schedule gates the ZUPT",
              "inchworm_zupt_ekf.py defaults, first 3 of the run's 5 cycles; curves from "
              "true_velocity / generate_ground_truth_and_data. State x = [p, v].", bottom=0.02)


def bounce_path(x0, e, g, t_end, n=2000):
    """Dense exact samples of the hybrid flow via the script's own flow/crossing_time/reset_map."""
    from state_estimation_simulations.use_numpy import saltation_matrix_ekf as m
    ts = np.linspace(0, t_end, n)
    out, events = [], []
    x, t0 = x0.copy(), 0.0
    tau = m.crossing_time(x, g)
    for s in ts:
        while tau is not None and t0 + tau <= s:
            x, _ = m.flow(x, tau, g)
            t0 += tau
            events.append((t0, x.copy()))
            x = m.reset_map(x, e)
            tau = m.crossing_time(x, g)
        xs, _ = m.flow(x, s - t0, g)
        out.append(xs)
    return ts, np.array(out), events


def fig_saltation_matrix_ekf_concept():
    """saltation_matrix_ekf.py: the hybrid system, and why the reset Jacobian alone misses the timing shift."""
    from state_estimation_simulations.use_numpy import saltation_matrix_ekf as m

    # (a)/(b): the script's default true trajectory
    duration, e, g, h0, vh = 5.0, 0.85, 9.81, 5.0, [1.0, 0.5]
    x0 = np.array([0.0, 0.0, h0, vh[0], vh[1], 0.0])
    ts, xs, events = bounce_path(x0, e, g, duration)

    # (c): hybrid_saltation_ekf.md §3 worked example
    xw = np.array([0.0, 0.0, 5.0, 0.0, 0.0, -2.0])
    dx = np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0])
    ew, T = 0.5, 1.0
    tw, xa, ev_a = bounce_path(xw, ew, g, T, 4000)
    _, xb, ev_b = bounce_path(xw + dx, ew, g, T, 4000)
    t_star, x_minus = ev_a[0]
    Phi_b = m.flow(xw, t_star, g)[1]
    Phi_a = m.flow(m.reset_map(x_minus, ew), T - t_star, g)[1]
    d_naive = Phi_a @ m.reset_jacobian(ew) @ Phi_b @ dx
    d_salt = Phi_a @ m.saltation_matrix(x_minus, ew, g) @ Phi_b @ dx
    d_true = xb[-1] - xa[-1]

    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 0.95])
    ax_p = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[1, 0], sharex=ax_p)
    ax_z = fig.add_subplot(gs[:, 1])
    ax_t = fig.add_subplot(gs[:, 2])

    t_ev = [te for te, _ in events]
    ax_p.plot(ts, xs[:, 2], color=C["blue"], lw=2)
    ax_p.scatter(t_ev, np.zeros(len(t_ev)), color=C["red"], zorder=3, s=22, label="guard: p_z = 0, falling")
    ax_p.set(ylabel="p_z (m)", title=f"(a) Default run: drop from {h0:.0f} m, e = {e}")
    ax_p.legend(loc="upper right")
    ax_v.plot(ts, xs[:, 5], color=C["orange"], lw=2)
    for te, xm in events:
        ax_v.plot([te, te], [xm[5], -e * xm[5]], color=C["red"], lw=1, ls=":")
    ax_v.set(xlabel="time (s)", ylabel="v_z (m/s)")
    ax_v.text(0.99, 0.05, "flow: v̇_z = −g   reset: v_z⁺ = −e·v_z⁻\n(p, v_x, v_y unchanged at the jump)",
              transform=ax_v.transAxes, ha="right", fontsize=8.5, color="#444444")
    ax_p.tick_params(labelbottom=False)
    ax_p.text(0.99, 0.62, f"v_x, v_y = {vh[0]}, {vh[1]} m/s throughout\nfirst impact t = {t_ev[0]:.3f} s",
              transform=ax_p.transAxes, ha="right", fontsize=8.5, color="#444444")

    ax_z.plot(tw, xa[:, 2], color=TRUTH, lw=1.8, label="A: nominal, p_z0 = 5 m")
    ax_z.plot(tw, xb[:, 2], color=C["purple"], lw=1.8, ls="--", label="B: 5 cm higher")
    ax_z.axvline(T, color="#777777", lw=1)
    ax_z.set(xlim=(0.6, 1.02), ylim=(-0.1, 2.3), xlabel="time (s)", ylabel="p_z (m)",
             title="(b) Why timing matters: two balls, one bounce")
    ax_z.text(T - 0.005, 2.05, "compare at\nfixed T = 1 s", ha="right", fontsize=8.5)
    t_b = ev_b[0][0]
    ax_z.annotate(f"B lands {1e3 * (t_b - t_star):.1f} ms later,\nfaster ({-ev_b[0][1][5]:.3f} vs "
                  f"{-x_minus[5]:.3f} m/s)", (t_star, 0.0), (0.845, 1.35), fontsize=8.5,
                  arrowprops=dict(arrowstyle="->", lw=0.8))
    ax_z.legend(loc="upper left")

    ax_t.axis("off")
    rows = [("", "Δv_z(T) (m/s)", "Δp_z(T) (m)"),
            ("naive  Φ·DR·Φ", f"{d_naive[5]:+.4f}", f"{d_naive[2]:+.4f}"),
            ("saltation  Φ·Ξ·Φ", f"{d_salt[5]:+.4f}", f"{d_salt[2]:+.4f}"),
            ("true (re-simulated)", f"{d_true[5]:+.4f}", f"{d_true[2]:+.4f}")]
    cols = [TRUTH, C["red"], C["blue"], TRUTH]
    for i, (a, b, c) in enumerate(rows):
        y = 0.86 - 0.1 * i
        w = "bold" if i == 0 else "normal"
        ax_t.text(0.0, y, a, fontsize=10, color=cols[i], fontweight="bold")
        ax_t.text(0.58, y, b, fontsize=10, ha="right", fontweight=w)
        ax_t.text(0.97, y, c, fontsize=10, ha="right", fontweight=w)
    ax_t.set_title("(c) Effect of δp_z = +5 cm at T", loc="left")
    ax_t.text(0.0, 0.42,
              "DR (reset Jacobian) has no notion of time: it\n"
              "carries the 5 cm offset through the bounce and\n"
              "predicts no velocity change.\n\n"
              "Ξ (saltation matrix) adds how the offset shifts\n"
              "the contact time - which here is the entire\n"
              "effect, and flips the sign of Δp_z.",
              fontsize=9.5, color="#444444", va="top")
    ax_t.text(0.0, 0.0, f"inputs from hybrid_saltation_ekf.md §3: x₀ = (0,0,5,0,0,−2),\n"
              f"e = {ew}, g = {g}; nominal impact t* = {t_star:.3f} s",
              fontsize=8, color="#777777")

    finish(fig, "saltation_matrix_ekf_concept", "Bouncing point mass: a hybrid system, and the saltation correction",
              "State x = (p, v) ∈ ℝ⁶, flow ẋ = (v, 0, 0, −g), guard g(x) = p_z, reset v_z ↦ −e·v_z. "
              "All curves and numbers from saltation_matrix_ekf.py's flow / crossing_time / reset_map / "
              "saltation_matrix.")
    print(f"    naive dv={d_naive[5]:+.4f} dp={d_naive[2]:+.4f} | salt dv={d_salt[5]:+.4f} dp={d_salt[2]:+.4f} "
          f"| true dv={d_true[5]:+.4f} dp={d_true[2]:+.4f} | t*={t_star:.4f} dt_b={1e3*(t_b-t_star):.2f}ms")


def fig_imu_preintegration_concept():
    """imu_preintegration.py: why preintegrate - fewer factors (problem A) and cheap bias updates (problem B)."""
    from state_estimation_simulations.use_numpy import imu_preintegration as m

    hz = 100
    b_g, b_a = np.array([0.01, -0.01, 0.02]), np.array([0.05, 0.0, -0.05])
    a_cmd, w_cmd = np.array([1.0, 0.1, 0.0]), np.array([0.0, 0.0, 0.5])
    b_g_new, b_a_new = np.array([0.008, -0.009, 0.018]), np.array([0.045, 0.002, -0.048])
    dt = 1.0 / hz

    bundle = m.PreintegratedIMUBundle(b_g, b_a)
    for _ in range(hz):
        bundle.integrate_measurement(a_cmd + b_a, w_cmd + b_g, dt)
    _, _, c_p = bundle.get_corrected_measurement(b_g_new, b_a_new)
    ref = m.PreintegratedIMUBundle(b_g_new, b_a_new)
    for _ in range(hz):
        ref.integrate_measurement(a_cmd + b_a, w_cmd + b_g, dt)
    yaw = np.degrees(np.arccos(np.clip((np.trace(bundle.delta_R) - 1) / 2, -1, 1)))

    fig, (ax, ax_b) = plt.subplots(1, 2, figsize=(14, 5.0), gridspec_kw={"width_ratios": [1.45, 1]})

    # (a) problem A: 100 samples between two keyframes become one factor
    t = np.arange(1, hz) * dt
    ax.vlines(t, 2.85, 3.15, color=C["orange"], lw=1)
    ax.text(-0.03, 3.0, f"IMU samples\n({hz} Hz)", ha="right", va="center", fontsize=9.5)
    for x, lbl in [(0.0, "i"), (1.0, "j")]:
        ax.scatter([x], [3.0], marker="s", s=90, color=TRUTH, zorder=3)
        ax.text(x, 3.35, f"keyframe {lbl}", ha="center", fontsize=9.5)
        ax.plot([x, x], [2.75, 1.25], color="#bbbbbb", lw=0.8, ls=":")
    ax.annotate("", (0.5, 1.45), (0.5, 2.65), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#555555"))
    ax.text(0.52, 2.05, f"integrate once\n({hz} micro-steps)", fontsize=9, color="#555555", va="center")
    ax.plot([0, 1], [1.0, 1.0], color=C["blue"], lw=2)
    for x, lbl in [(0.0, "x_i"), (1.0, "x_j")]:
        ax.scatter([x], [1.0], s=420, color="white", edgecolor=TRUTH, lw=1.6, zorder=3)
        ax.text(x, 1.0, lbl, ha="center", va="center", fontsize=9.5, zorder=4)
    ax.scatter([0.5], [1.0], marker="s", s=150, color=C["blue"], zorder=3)
    ax.text(0.5, 0.72, f"one preintegrated factor:  ΔR = {yaw:.1f}° yaw,  "
            f"|Δv| = {np.linalg.norm(bundle.delta_v):.3f} m/s,  |Δp| = {np.linalg.norm(bundle.delta_p):.3f} m",
            ha="center", va="top", fontsize=9.5, color=C["blue"])
    ax.set(xlim=(-0.25, 1.1), ylim=(0.3, 3.7), title="(a) Problem A: too many samples → one factor per keyframe pair")
    ax.axis("off")

    # (b) problem B: a bias update without touching the raw samples
    ax_b.barh([1, 0], [hz, 1], color=[C["orange"], C["green"]], height=0.5)
    ax_b.set_yticks([1, 0], [f"re-integrate all {hz}\nsamples with new bias", "first-order correction\nΔ + J·δb"])
    ax_b.set_xlabel("integration steps / matrix-vector products after a bias update")
    ax_b.text(hz + 2, 1, f"Δp = {np.round(ref.delta_p, 4)}", va="center", fontsize=8.5)
    ax_b.text(3, 0, f"Δp = {np.round(c_p, 4)}\n(differs by {np.linalg.norm(c_p - ref.delta_p):.1e} m)",
              va="center", fontsize=8.5)
    ax_b.set(xlim=(0, hz * 1.75), title="(b) Problem B: the bias changes every optimizer iteration")
    ax_b.grid(axis="y", visible=False)

    finish(fig, "imu_preintegration_concept", "IMU preintegration: what it saves",
           f"imu_preintegration.py defaults: 1 s at {hz} Hz, commanded acceleration {a_cmd.tolist()} m/s², "
           f"turn rate {w_cmd[2]} rad/s, then the script's default bias update")


def fig_imu_propagation_position_fix_concept():
    """imu_propagation_position_fix.py: the 1 Hz fix reads position only, so its Jacobian cannot touch orientation."""
    from state_estimation_simulations.use_numpy import imu_propagation_position_fix as m

    seconds, pos_std = 10, 0.05
    T_true, T_est, pre, post = quiet(m.run_simulation, 0.01, seconds, 4, 1e-6, 10, 1.0, 0.5, pos_std,
                                     np.eye(3) / pos_std ** 2, np.random.default_rng(SEED))
    J = m.position_observation_jacobian(T_est)

    fig, (ax_t, ax_j) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})

    # (a) one second of the timeline
    ax_t.vlines(np.arange(100) / 100, 1.85, 2.15, color=C["orange"], lw=1)
    ax_t.text(-0.03, 2.0, "IMU twist\n(100 Hz)", ha="right", va="center", fontsize=9.5)
    ax_t.text(0.5, 2.3, "dead reckoning: T ← T·Exp(ξ̃·dt), position and orientation both drift",
              ha="center", fontsize=9)
    ax_t.scatter([1.0], [1.0], marker="D", s=90, color=C["green"], zorder=3)
    ax_t.text(-0.03, 1.0, "position fix\n(1 Hz)", ha="right", va="center", fontsize=9.5)
    ax_t.text(0.97, 0.72, f"z = p_true + noise (σ = {pos_std * 100:.0f} cm per axis)\n"
              "Gauss-Newton on r = z − p_est\n→ position pulled back, orientation untouched",
              ha="right", va="top", fontsize=9, color=C["green"])
    ax_t.annotate("", (1.0, 1.12), (1.0, 1.8), arrowprops=dict(arrowstyle="-|>", color="#777777"))
    ax_t.set(xlim=(-0.32, 1.08), ylim=(0.0, 2.6), title="(a) One second: 100 IMU steps, then one fix")
    ax_t.axis("off")
    ax_t.text(0.0, 0.0, f"at the end of {seconds} s: position error {post[-1] * 100:.1f} cm after the fix, "
              "orientation drift never corrected", transform=ax_t.transAxes, fontsize=8.5, color="#555555")

    # (b) the Jacobian itself, at the run's final estimate
    ax_j.imshow(np.abs(J), cmap="Blues", vmin=0, vmax=np.abs(J).max() * 1.8)
    for (r, c), v in np.ndenumerate(J):
        ax_j.text(c, r, f"{v:.2f}" if v else "0", ha="center", va="center", fontsize=9,
                  color="#999999" if not v else TRUTH)
    ax_j.set_xticks(range(6), ["v_x", "v_y", "v_z", "ω_x", "ω_y", "ω_z"])
    ax_j.set_yticks(range(3), ["p_x", "p_y", "p_z"])
    ax_j.axvline(2.5, color=TRUTH, lw=1.5)
    ax_j.grid(False)
    ax_j.set_title("(b) ∂p/∂ξ = [R | 0] at the final estimate")
    ax_j.set_xlabel("R_est (rotation of the estimate)          exactly 0: a GN step can never reach ω", fontsize=9.5)

    finish(fig, "imu_propagation_position_fix_concept", "Why a position-only fix cannot correct orientation",
           f"imu_propagation_position_fix.py: Jacobian from position_observation_jacobian at the final pose of a {seconds} s "
           "run (seed 0, other settings default); ξ = [v, ω], right perturbation T·Exp(ξ)")


def fig_pose_tracking_dt_sweep():
    """pointcloud_pose_tracking.py: max gap of UKF and vanilla KF to the EKF trajectory vs the time step dt."""
    from state_estimation_simulations.use_numpy import pointcloud_pose_tracking as m
    from state_estimation_simulations.use_numpy.lie_utils import se3_exp, rotation_geodesic_error

    duration, n_points = 5.0, 20
    vel_std, gyro_std, point_std, init_std = 0.05, 0.02, 0.03, 0.1
    dts = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
    gap = {"UKF": [], "Vanilla KF": []}          # per dt: (first 1 s pos, after 1 s pos, after 1 s rot deg)
    for dt in dts:
        rng = np.random.default_rng(SEED)        # fresh data + T_init per dt, as fig_pose_tracking draws them
        body, T_true, u_meas, z = m.generate_ground_truth_and_data(duration, dt, n_points, vel_std, gyro_std, point_std, rng)
        T_init = T_true[0] @ se3_exp(rng.normal(0.0, init_std, 6))
        P_init = init_std ** 2 * np.eye(6)
        Q = dt ** 2 * np.diag([vel_std ** 2] * 3 + [gyro_std ** 2] * 3)
        T_ekf = m.run_ekf(T_init, P_init, u_meas, z, body, dt, Q, point_std)
        others = {"UKF": m.run_ukf(T_init, P_init, u_meas, z, body, dt, Q, point_std, 1.0, 2.0, -3.0),
                  "Vanilla KF": m.run_vanilla_kf(T_init, P_init, u_meas, z, body, dt, Q, point_std)}
        first = np.arange(len(T_ekf)) < int(np.floor(1.0 / dt + 1e-9))   # first floor(1/dt) poses; the pose at t = 1 s counts as "after 1 s"
        for name, T_x in others.items():
            pos = np.array([np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in zip(T_ekf, T_x)])
            rot = np.degrees([rotation_geodesic_error(a[:3, :3], b[:3, :3]) for a, b in zip(T_ekf, T_x)])
            gap[name].append((pos[first].max(), pos[~first].max(), rot[~first].max()))
    for name, rows in gap.items():
        print(f"    {name}: dt, first 1 s pos (m), after 1 s pos (m), after 1 s rot (deg)")
        for dt, (p1, p2, r2) in zip(dts, rows):
            print(f"      {dt:<6g} {p1:.1e}  {p2:.1e}  {r2:.1e}")

    style_of = {"UKF": C["purple"], "Vanilla KF": C["orange"]}
    fig, (ax_p, ax_r) = plt.subplots(1, 2)
    for name, rows in gap.items():
        g = np.array(rows)
        col = style_of[name]
        ax_p.plot(dts, g[:, 1], color=col, ls="-", marker="o", ms=4, lw=1.9, label=f"{name}, after 1 s")
        ax_p.plot(dts, g[:, 0], color=col, ls="--", marker="o", ms=4, lw=1.4, mfc="none", label=f"{name}, first 1 s")
        ax_r.plot(dts, g[:, 2], color=col, ls="-", marker="o", ms=4, lw=1.9, label=f"{name}, after 1 s")
    ref = np.array(dts)
    ax_p.plot(ref, 3.8e-3 * ref / 0.1, color=PRIOR, lw=1.0, label="reference slope ∝ dt")
    for ax in (ax_p, ax_r):
        ax.set(xscale="log", yscale="log", xlabel="time step dt (s)")
        ax.set_xticks(dts)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.axvline(0.1, color="#555555", ls=":", lw=1.1)
        ax.annotate("default", (0.1, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", fontsize=9, color="#555555")
        ax.tick_params(axis="x", labelsize=9)
    ax_p.set(title="Position gap to EKF", ylabel="max position gap (m)")
    ax_r.set(title="Rotation gap to EKF, after the first second", ylabel="max rotation gap (deg)")
    handles, labels = ax_p.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9)
    finish(fig, "pose_tracking_dt_sweep", "How far UKF and vanilla KF sit from EKF as the step size changes",
           "pointcloud_pose_tracking.py defaults, seed 0, 5 s run, fresh data per dt; gap = max over the window "
           "(first 1 s = initial correction; after 1 s = steady state)", bottom=0.08)


def fig_pose_tracking_concept():
    """pointcloud_pose_tracking.py: one rigid point cloud, observed as z = T·p + noise; world vs body residual."""
    from state_estimation_simulations.use_numpy import pointcloud_pose_tracking as m
    from state_estimation_simulations.use_numpy.lie_utils import se3_exp

    duration, dt, n_points = 5.0, 0.1, 20
    vel_std, gyro_std, point_std, init_std = 0.05, 0.02, 0.03, 0.1
    rng = np.random.default_rng(SEED)
    body, T_true, u_meas, z = m.generate_ground_truth_and_data(duration, dt, n_points, vel_std, gyro_std, point_std, rng)
    T_init = T_true[0] @ se3_exp(rng.normal(0.0, init_std, 6))
    T_dr = m.run_dead_reckoning(T_init, u_meas, dt)
    k = 10
    T_pred = T_dr[k]
    pred_world, _ = m.observation_model(T_pred, body)
    R, t = T_pred[0:3, 0:3], T_pred[0:3, 3]
    z_body = (z[k] - t) @ R  # T_pred^-1 · z, row-wise

    fig, (ax_w, ax_r, ax_b) = plt.subplots(1, 3, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.25, 1, 1]})

    # (a) the same rigid cloud carried along the true trajectory
    ax_w.plot(*xy_of(T_true), color=TRUTH, lw=1.6, label="true trajectory")
    for kk, col in [(0, C["blue"]), (len(T_true) - 1, C["purple"])]:
        pts, _ = m.observation_model(T_true[kk], body)
        ax_w.scatter(pts[:, 0], pts[:, 1], s=14, color=col, alpha=0.8, label=f"true cloud T·p at t = {kk * dt:.1f} s")
    ax_w.scatter([T_true[k][0, 3]], [T_true[k][1, 3]], marker="*", s=160, color=C["red"], zorder=3,
                 label=f"t = {k * dt:.1f} s, zoomed in (b)/(c)")
    ax_w.set(title=f"(a) {n_points} body points, carried rigidly by T", xlabel="x (m)", ylabel="y (m)")
    ax_w.axis("equal")
    legend_below(ax_w, ncol=2)

    # (b) world-frame residual (EKF)
    ax_r.scatter(pred_world[:, 0], pred_world[:, 1], s=18, facecolor="none", edgecolor=PRIOR, label="prediction T_pred·p")
    ax_r.scatter(z[k][:, 0], z[k][:, 1], s=14, color=C["red"], label=f"measurement z (σ = {point_std * 100:.0f} cm)")
    for a, b in zip(pred_world, z[k]):
        ax_r.plot([a[0], b[0]], [a[1], b[1]], color=C["red"], lw=0.7, alpha=0.6)
    ax_r.set(title="(b) EKF: r = z − T_pred·p (world frame)", xlabel="x (m)", ylabel="y (m)")
    ax_r.axis("equal")
    legend_below(ax_r, ncol=2)

    # (c) body-frame residual (IEKF)
    ax_b.scatter(body[:, 0], body[:, 1], s=18, facecolor="none", edgecolor=PRIOR, label="known body points p")
    ax_b.scatter(z_body[:, 0], z_body[:, 1], s=14, color=C["green"], label="T_pred⁻¹·z")
    for a, b in zip(body, z_body):
        ax_b.plot([a[0], b[0]], [a[1], b[1]], color=C["green"], lw=0.7, alpha=0.6)
    ax_b.set(title="(c) IEKF: r = T_pred⁻¹·z − p (body frame)", xlabel="body x (m)", ylabel="body y (m)")
    ax_b.axis("equal")
    legend_below(ax_b, ncol=2)
    lw_, lb_ = np.linalg.norm(z[k] - pred_world, axis=1), np.linalg.norm(z_body - body, axis=1)
    ax_b.text(0.02, 0.98, f"same residual lengths (max diff {np.abs(lw_ - lb_).max():.0e} m),\n"
              "only rotated by R_pred: same information, different frame",
              transform=ax_b.transAxes, fontsize=8.5, color="#555555", va="top")

    finish(fig, "pose_tracking_concept", "Point-cloud pose tracking: one measurement, two residual frames",
           f"pointcloud_pose_tracking.py defaults, seed 0; x-y projection. (b)/(c) at t = {k * dt:.1f} s with the "
           "twist-only prediction (run_dead_reckoning from the perturbed initial pose) as T_pred", bottom=0.06)


def fig_ekf_linearization_concept():
    """kf_ekf_iekf.md §2: why the EKF's tangent-line stand-in works near the estimate and fails far from it.
    Pure-math concept figure (no simulation): the unicycle's x-step f(θ) = v·Δt·cos θ with v·Δt = 1 m."""
    th_hat = np.radians(30.0)
    f0, slope = np.cos(th_hat), -np.sin(th_hat)  # slope in m per rad
    deg = np.linspace(-30, 120, 600)
    f = np.cos(np.radians(deg))
    tang = f0 + slope * (np.radians(deg) - th_hat)
    gap = {d: float(np.cos(th_hat + np.radians(d)) - (f0 + slope * np.radians(d))) for d in (5, 40)}

    fig = plt.figure(figsize=(13.5, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax_s, ax_l = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    # left: the curve, its tangent at the estimate, and the two uncertainty bands
    ax.axvspan(30 - 40, 30 + 40, color=C["orange"], alpha=0.14, lw=0, label="±40° (large uncertainty)")
    ax.axvspan(30 - 5, 30 + 5, color=C["blue"], alpha=0.28, lw=0, label="±5° (small uncertainty)")
    ax.plot(deg, f, color=TRUTH, lw=2.2, label="true f(θ) = cos θ")
    ax.plot(deg, tang, color=C["red"], lw=1.8, ls="--", label="EKF tangent: cos θ̂ − sin θ̂·(θ − θ̂)")
    ax.scatter([30], [f0], s=70, color=C["red"], zorder=4, label=f"estimate θ̂ = 30°, f = {f0:.3f} m")
    for d, col, dx in [(5, C["blue"], -4), (40, C["orange"], 3)]:
        x = 30 + d
        ax.plot([x, x], [np.cos(np.radians(x)), np.cos(th_hat) + slope * np.radians(d)], color=col, lw=3, zorder=3,
                solid_capstyle="butt")
    ax.annotate(f"gap at θ̂+5°: {abs(gap[5]) * 100:.1f} cm\n(invisible at this scale)", xy=(35, np.cos(np.radians(35))),
                xytext=(52, 1.0), fontsize=9, color="#2e5f9e", arrowprops=dict(arrowstyle="-", color="#2e5f9e", lw=0.9))
    ax.annotate(f"gap at θ̂+40°: {abs(gap[40]) * 100:.1f} cm\n(tangent is above the curve)",
                xy=(70, 0.5 * (np.cos(np.radians(70)) + f0 + slope * np.radians(40))), xytext=(82, 0.62), fontsize=9,
                color="#b8672a", arrowprops=dict(arrowstyle="-", color="#b8672a", lw=0.9))
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set(title="Linearize at the estimate", xlabel="heading θ (deg)", ylabel="x-step f(θ) (m)",
           xlim=(-30, 120), ylim=(-0.65, 1.2))
    ax.set_xticks(range(-30, 121, 30))
    ax.legend(loc="lower left", fontsize=8.5)

    # right: push N(θ̂, σ²) through f, true vs EKF's linearized Gaussian
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal(200_000)
    xs = np.linspace(-1.05, 1.05, 500)
    stats = {}
    for a, sd in [(ax_s, 5.0), (ax_l, 40.0)]:
        sr = np.radians(sd)
        samp = np.cos(th_hat + sr * z)
        m_true, s_true = float(samp.mean()), float(samp.std())
        m_ekf, s_ekf = float(f0), float(abs(slope) * sr)
        stats[sd] = (m_true, s_true, m_ekf, s_ekf)
        a.hist(samp, bins=np.linspace(-1.0, 1.0, 101), density=True, color=C["gold"], alpha=0.8,
               label=f"true (Monte Carlo): mean {m_true:.3f} m, std {s_true:.3f} m")
        a.plot(xs, np.exp(-0.5 * ((xs - m_ekf) / s_ekf) ** 2) / (s_ekf * np.sqrt(2 * np.pi)), color=C["red"], lw=2,
               label=f"EKF Gaussian: mean {m_ekf:.3f} m, std {s_ekf:.3f} m")
        a.axvline(m_true, color=TRUTH, lw=1.2, ls=":")
        a.axvline(m_ekf, color=C["red"], lw=1.2, ls=":")
        a.set(title=f"θ ~ N(30°, ({sd:.0f}°)²): output of f", ylabel="density", xlim=(-1.05, 1.05))
        a.legend(loc="upper left", fontsize=8.5)
        a.set_ylim(0, a.get_ylim()[1] * 1.45)
    ax_l.set_xlabel("x-step f(θ) (m)")
    ax_s.text(0.02, 0.5, "near-perfect match", transform=ax_s.transAxes, ha="left", fontsize=9, color="#555555")
    ax_l.text(0.02, 0.5, "EKF mean is off, shape is wrong", transform=ax_l.transAxes, ha="left", fontsize=9,
              color="#555555")
    for sd, (mt, st, me, se) in stats.items():
        print(f"    sigma={sd:.0f} deg: true mean {mt:.4f} std {st:.4f} | EKF mean {me:.4f} std {se:.4f}")
    print(f"    gap +5 deg: {gap[5] * 100:.3f} cm, +40 deg: {gap[40] * 100:.3f} cm")

    finish(fig, "ekf_linearization_concept", "EKF: linearize the nonlinear model at the current estimate",
           "f(θ) = cos θ: the unicycle's x-step per metre travelled (§2), estimate θ̂ = 30°")


def fig_bundle_adjustment_concept():
    """bundle_adjustment.py: one landmark's bundle of rays, before and after joint BA."""
    from state_estimation_simulations.use_numpy import bundle_adjustment as m

    rng = np.random.default_rng(SEED)
    K = (800.0, 800.0, 320.0, 240.0)
    T_true, P_all = m.generate_ground_truth_scene(8, 60, 5.0, 180.0, 2.0, rng)
    P_true, obs = m.build_observations(T_true, P_all, K, 1.0, 2, rng)
    T_init, P_init = m.perturb_initial_guess(T_true, P_true, 0.1, 0.3, rng)
    T_ba, P_ba = quiet(m.run_bundle_adjustment, T_init, P_init, obs, K, 0.1, 1.0, 1e-6, 30)
    T_ba, P_ba = m.align_reconstruction_to_ground_truth(T_true, T_ba, P_ba)

    counts = np.bincount([j for _, j, _ in obs], minlength=len(P_true))
    lm = int(np.argmax(counts))
    rays = [(i, zij) for i, j, zij in obs if j == lm]

    def ray(T, zij):
        d = T[0:3, 0:3] @ np.array([(zij[0] - K[2]) / K[0], (zij[1] - K[3]) / K[1], 1.0])
        return T[0:3, 3], d / np.linalg.norm(d)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.0), sharex=True, sharey=True)
    for ax, T_list, P_list, title in [(axes[0], T_init, P_init, "(a) Initial guess: the rays do not meet"),
                                      (axes[1], T_ba, P_ba, "(b) After joint BA: one bundle, one point")]:
        P = np.asarray(P_list)
        ax.scatter(P[:, 0], P[:, 1], s=8, color=PRIOR, alpha=0.5, label="all landmark estimates")
        for i, zij in rays:
            c, d = ray(T_list[i], zij)
            L = np.linalg.norm(P_true[lm] - T_true[i][0:3, 3]) * 1.25
            ax.plot([c[0], c[0] + L * d[0]], [c[1], c[1] + L * d[1]], color=C["orange"], lw=1.1)
            ax.scatter([c[0]], [c[1]], marker="s", s=36, color=C["blue"], zorder=3)
        ax.scatter([P_true[lm][0]], [P_true[lm][1]], marker="*", s=200, color=TRUTH, zorder=4, label="true landmark")
        ax.scatter([P[lm][0]], [P[lm][1]], marker="o", s=60, facecolor="none", edgecolor=C["red"], lw=1.8,
                   zorder=4, label="its current estimate P_j")
        ax.set(title=title, xlabel="x (m)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y (m)")
    axes[0].plot([], [], color=C["orange"], label="ray through each measured pixel z_ij")
    axes[0].scatter([], [], marker="s", color=C["blue"], label="camera estimates T_i")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8.5)

    finish(fig, "bundle_adjustment_concept", "Why it is called bundle adjustment",
           f"bundle_adjustment.py defaults, seed 0, top view: rays through landmark {lm}'s {len(rays)} measured "
           "pixels; (b) after joint BA + alignment", bottom=0.08)


def fig_bundle_adjustment_advanced_concept():
    """bundle_adjustment_advanced.py: the covisibility graph and one Local BA window, as the solver built them."""
    from state_estimation_simulations.use_numpy import bundle_adjustment_advanced as m

    K = (800.0, 800.0, 320.0, 240.0)
    n_kf, k_show = 50, 30
    T_true = m.generate_ground_truth_trajectory(n_kf, 15.0, 90.0)
    P_all = m.generate_landmark_corridor(T_true, 8, 2.0, 1.0, np.random.default_rng(SEED), extend_ahead=6.0)
    P_true, obs = m.build_observations(T_true, P_all, K, 1.0, 6.0, 3, np.random.default_rng(SEED))
    by_kf = m.group_observations_by_keyframe(obs, n_kf)

    # Record the windows exactly as run_incremental_local_ba builds them (no re-implementation).
    windows, original = {}, m.build_active_window
    def recorder(k, shared_count, *rest):
        out = original(k, shared_count, *rest)
        windows[k] = (out, dict(shared_count))
        return out
    m.build_active_window = recorder
    try:
        _, _, hist = quiet(m.run_incremental_local_ba, T_true, by_kf, K, 0.02, 3, 2, 6, 8, True, 1e-6, 15,
                           np.random.default_rng(SEED), 20)
    finally:
        m.build_active_window = original
    (active, fixed, points), shared = windows[k_show]

    fig, (ax_w, ax_c, ax_n) = plt.subplots(1, 3, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.2, 1, 1]})

    # (a) the window at keyframe k_show, drawn at ground-truth positions
    pos = np.array([T[0:3, 3] for T in T_true])
    ax_w.scatter(P_true[:, 0], P_true[:, 1], s=5, color=PRIOR, alpha=0.35, label="landmarks")
    ax_w.scatter(P_true[points, 0], P_true[points, 1], s=10, color=C["green"], label=f"active landmarks ({len(points)})")
    ax_w.plot(pos[:k_show + 1, 0], pos[:k_show + 1, 1], "o", ms=3, color="#bbbbbb", label="other keyframes so far")
    groups = [(sorted(fixed), C["orange"], f"fixed anchors of this window ({len(fixed)})"),
              (active[1:], C["blue"], f"active covisible neighbours ({len(active) - 1})"),
              ([k_show], C["red"], f"new keyframe {k_show}"), ([0, 1], TRUTH, "permanently fixed keyframes 0, 1")]
    for idx, col, lbl in groups:
        ax_w.plot(pos[idx, 0], pos[idx, 1], "o", ms=6, color=col, label=lbl)
    ax_w.set(title=f"(a) Window when keyframe {k_show} arrives", xlabel="x (m)", ylabel="y (m)")
    ax_w.set(xlim=(P_true[:, 0].min() - 1, P_true[:, 0].max() + 1),
             ylim=(min(P_true[:, 1].min(), pos[:, 1].min()) - 1, P_true[:, 1].max() + 1))
    ax_w.set_aspect("equal", adjustable="box")
    ax_w.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)

    # (b) covisibility: shared landmark counts between keyframes so far
    C_mat = np.zeros((k_show + 1, k_show + 1))
    for key, c in shared.items():
        a, b = key
        if a <= k_show and b <= k_show:
            C_mat[a, b] = C_mat[b, a] = c
    im = ax_c.imshow(np.where(C_mat > 0, C_mat, np.nan), cmap="Blues", interpolation="nearest")
    for i in active:
        ax_c.add_patch(Rectangle((i - 0.5, k_show - 0.5), 1, 1, fill=False, ec=C["red"], lw=1.4))
    fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.03, label="shared landmarks")
    ax_c.set(title="(b) Covisibility graph (banded, not dense)", ylabel="keyframe")
    ax_c.grid(False)
    ax_c.set_xlabel(f"keyframe\nred: keyframe {k_show}'s chosen neighbours (≥ 2 shared, strongest {len(active) - 1})")

    # (c) window size stays bounded as the map grows
    ax_n.plot(hist["local_step"], hist["n_active_kf"], color=C["blue"], lw=1.6, label="active keyframes (cap 6)")
    ax_n.plot(hist["local_step"], hist["n_fixed_kf"], color=C["orange"], lw=1.6, label="fixed keyframes")
    ax_n.plot(hist["local_step"], np.array(hist["local_step"]) + 1, color=PRIOR, ls="--", lw=1.2,
              label="all keyframes so far (Global BA)")
    ax_n.set(title="(c) Window size per Local BA call", xlabel="new keyframe", ylabel="keyframes in the problem")
    empty = [k for k in range(n_kf) if not by_kf[k]]
    if empty:
        ax_n.annotate(f"keyframes {empty[0]}-{empty[-1]} observe no landmarks\n(end of the corridor): empty window",
                      (empty[0], 0.5), (24, 22), fontsize=8.5, color="#555555", arrowprops=dict(arrowstyle="->", lw=0.8))
    ax_n.legend(loc="upper left")

    finish(fig, "bundle_adjustment_advanced_concept", "Local BA: a bounded window picked from the covisibility graph",
           "bundle_adjustment_advanced.py defaults, seed 0; windows recorded from run_incremental_local_ba's own "
           "build_active_window calls")


def fig_pose_graph_incremental_concept():
    """pose_graph_incremental.py: what one Givens row insertion touches in R, odometry edge vs loop closure."""
    from state_estimation_simulations.use_numpy import pose_graph_incremental as m
    from state_estimation_simulations.utils import qr_insert_row

    rng = np.random.default_rng(SEED)
    gt = m.generate_ground_truth_trajectory(2.0, 2)  # 8-pose square: small enough to read R entry by entry
    n = len(gt)
    odo, loop = m.simulate_noisy_edges(gt, 0.05, 0.01, 0.5, rng)
    x_lin = m.run_dead_reckoning(gt[0], odo)
    info, anchor = np.eye(6), 1e6
    sqrt_info = np.linalg.cholesky(info).T

    R, d = m.full_relinearize(x_lin[:n - 1], odo[:n - 2], [], info, anchor)
    dof = 6 * n
    R_a, d_a = np.zeros((dof, dof)), np.zeros(dof)
    R_a[:dof - 6, :dof - 6], d_a[:dof - 6] = R, d

    def insert(R0, d0, edge):
        R1, d1 = R0.copy(), d0.copy()
        A, b = m.edge_whitened_block(edge[0], edge[1], edge[2], x_lin[:n], dof, sqrt_info)
        for r in range(6):
            qr_insert_row(R1, d1, A[r], b[r])
        return R1, d1

    R_b, d_b = insert(R_a, d_a, odo[n - 2])
    R_c, _ = insert(R_b, d_b, loop[0])
    R_full, _ = m.full_relinearize(x_lin[:n], odo[:n - 1], [], info, anchor)
    match = np.abs(np.abs(R_b) - np.abs(R_full)).max()

    def show(ax, R_new, R_old, title):
        nz, changed = np.abs(R_new) > 1e-12, np.abs(R_new - R_old) > 1e-12
        img = np.ones((dof, dof, 3))
        img[nz] = (0.75, 0.85, 0.95)
        img[changed] = (0.82, 0.29, 0.36)
        ax.imshow(img, interpolation="nearest")
        for k in range(1, n):
            ax.axhline(6 * k - 0.5, color="#dddddd", lw=0.5)
            ax.axvline(6 * k - 0.5, color="#dddddd", lw=0.5)
        ax.set_xticks([6 * k + 2.5 for k in range(n)], [f"x{k}" for k in range(n)], fontsize=8)
        ax.set_yticks([6 * k + 2.5 for k in range(n)], [f"x{k}" for k in range(n)], fontsize=8)
        ax.grid(False)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(f"{changed.sum()} of {nz.sum()} nonzeros changed" if R_old is not None else "")
        return changed.sum()

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6))
    show(axes[0], R_a, R_a, f"(a) R after {n - 1} poses (chain)")
    axes[0].set_xlabel(f"{(np.abs(R_a) > 1e-12).sum()} nonzeros; new pose x{n - 1} = empty block")
    show(axes[1], R_b, R_a, f"(b) + odometry edge x{n - 2}→x{n - 1}: 6 Givens rows")
    show(axes[2], R_c, R_b, f"(c) + loop closure x{loop[0][0]}→x{loop[0][1]}: same insertion")
    axes[0].scatter([], [], marker="s", color=(0.75, 0.85, 0.95), label="nonzero, unchanged")
    axes[0].scatter([], [], marker="s", color=(0.82, 0.29, 0.36), label="changed by this insertion")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))

    finish(fig, "pose_graph_incremental_concept", "Incremental square-root SAM: what one new edge costs",
           f"full_relinearize / edge_whitened_block / qr_insert_row on an {n}-pose square, seed 0; "
           f"(b) matches a from-scratch QR to {match:.0e}", bottom=0.1)


def fig_sliding_window_marginalization_concept():
    """sliding_window_marginalization.py: the window slides, and each dropped pose becomes one prior block."""
    from state_estimation_simulations.use_numpy import sliding_window_marginalization as m

    info, window = np.eye(6), 10
    gt = m.generate_ground_truth_trajectory(2.0, nodes_per_side=4)
    n = len(gt)
    odo, _ = m.simulate_noisy_edges(gt, 0.05, 0.01, loop_noise_scale=1.0, rng=np.random.default_rng(SEED))

    # Record what run_sliding_window_pose_graph actually does (no re-implementation).
    sizes, margs = [], []
    orig_solve, orig_marg = m.solve_to_convergence, m.marginalize_oldest
    def rec_solve(x_window, *a):
        sizes.append(len(x_window))
        return orig_solve(x_window, *a)
    def rec_marg(*a):
        margs.append(a)
        return orig_marg(*a)
    m.solve_to_convergence, m.marginalize_oldest = rec_solve, rec_marg
    try:
        m.run_sliding_window_pose_graph(gt, odo, info, window, 1e6, 1e-6, 10)
    finally:
        m.solve_to_convergence, m.marginalize_oldest = orig_solve, orig_marg

    fig, (ax_w, ax_h, ax_n) = plt.subplots(1, 3, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.15, 1, 1]})

    # (a) window membership after each new pose (window always ends at the newest pose)
    img = np.ones((n - 1, n, 3))
    for row, (k, L) in enumerate(zip(range(1, n), sizes)):
        img[row, :k - L + 1] = (0.80, 0.80, 0.80)
        img[row, k - L + 1:k + 1] = (0.75, 0.85, 0.95)
        img[row, k] = (0.82, 0.29, 0.36)
    ax_w.imshow(img, interpolation="nearest", aspect="auto", extent=(-0.5, n - 0.5, n - 0.5, 0.5))
    ax_w.set(title=f"(a) The window slides (size {window})", xlabel="pose index", ylabel="after adding pose k")
    ax_w.grid(False)
    for col, lbl in [((0.80, 0.80, 0.80), "marginalized (frozen)"), ((0.75, 0.85, 0.95), "live in the window"),
                     ((0.82, 0.29, 0.36), "newest pose")]:
        ax_w.scatter([], [], marker="s", s=60, color=col, label=lbl)
    legend_below(ax_w, ncol=3)

    def spy(ax, args, title, note):
        H, _ = m.assemble_window_system(*args)
        img = np.ones(H.shape + (3,))
        img[np.abs(H) > 1e-12] = (0.75, 0.85, 0.95)
        img[0:6, 0:6][np.abs(H[0:6, 0:6]) > 1e-12] = (0.82, 0.29, 0.36)
        ax.imshow(img, interpolation="nearest")
        nb = H.shape[0] // 6
        ax.set_xticks([6 * b + 2.5 for b in range(nb)], [f"w{b}" for b in range(nb)], fontsize=8)
        ax.set_yticks([6 * b + 2.5 for b in range(nb)], [f"w{b}" for b in range(nb)], fontsize=8)
        ax.grid(False)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(note, fontsize=9)

    spy(ax_h, margs[0], "(b) H of the full window, before the first drop",
        "red: oldest pose x_a (window slot w0), about to be eliminated\nblock-tridiagonal: x_a touches only w1")
    spy(ax_n, margs[1], "(c) H of the next window",
        "red: the new prior on the new oldest pose,\nΛ_bb(edge ab) − Λ_ba Λ_aa⁻¹ Λ_ab: one block, no fill-in")

    finish(fig, "sliding_window_marginalization_concept", "Sliding-window marginalization: drop a pose, keep its information",
           f"sliding_window_marginalization.py, {n}-pose square (nodes_per_side 4), window {window}, seed 0; "
           "windows and H recorded from run_sliding_window_pose_graph's own calls", bottom=0.04)


def fig_pnp_estimation_concept():
    """pnp_estimation.py + bundle_adjustment_advanced.py: triangulation and PnP as mirror-image ray problems."""
    from state_estimation_simulations.use_numpy import bundle_adjustment_advanced as ba
    from state_estimation_simulations.use_numpy import pnp_estimation as m

    K = (800.0, 800.0, 320.0, 240.0)

    # (a) triangulation: known cameras, unknown point -- bundle_adjustment_advanced.py's triangulate_landmark /
    # refine_landmark_gn on bundle_adjustment.py's wide-baseline arc (its forward-moving corridor has rays too
    # close to parallel to read in a top view)
    from state_estimation_simulations.use_numpy import bundle_adjustment as bscene
    rng_a = np.random.default_rng(SEED)
    T_cam, P_all = bscene.generate_ground_truth_scene(8, 60, 5.0, 180.0, 2.0, rng_a)
    P_true, obs = bscene.build_observations(T_cam, P_all, K, 1.0, 2, rng_a)
    lm = int(np.argmax(np.bincount([j for _, j, _ in obs], minlength=len(P_true))))
    seen = [(i, z) for i, j, z in obs if j == lm]
    T_obs, z_obs = [T_cam[i] for i, _ in seen], [z for _, z in seen]
    P0 = ba.triangulate_landmark(T_obs, z_obs, K)
    P_ref = ba.refine_landmark_gn(T_obs, z_obs, P0, K, 1e-6, 15)

    # (b) PnP: known points, unknown camera (pnp_estimation.py's scene and functions, as in fig_pnp_estimation)
    rng = np.random.default_rng(SEED)
    T_true, P_list = m.generate_scene(20, K, rng)
    z_list = [m.camera_project(T_true, P, K)[0] + rng.normal(0.0, 1.0, 2) for P in P_list]
    T_dlt = m.linear_pnp_dlt(P_list, z_list, K)
    T_gn = m.refine_pose_gn(T_dlt, P_list, z_list, K, 1e-8, 20)

    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(13, 5.4))

    for T, z in zip(T_obs, z_obs):
        c = T[0:3, 3]
        d = T[0:3, 0:3] @ np.array([(z[0] - K[2]) / K[0], (z[1] - K[3]) / K[1], 1.0])
        L = np.linalg.norm(P_true[lm] - c) * 1.25 / np.linalg.norm(d)
        ax_t.plot([c[0], c[0] + L * d[0]], [c[1], c[1] + L * d[1]], color=C["orange"], lw=1)
        ax_t.scatter(*c[0:2], marker="s", s=40, color=TRUTH, zorder=3)
    ax_t.scatter(*P_true[lm][0:2], marker="*", s=220, color=TRUTH, zorder=4, label="true point")
    ax_t.scatter(*P0[0:2], marker="o", s=70, facecolor="none", edgecolor=C["orange"], lw=1.8, zorder=4,
                 label=f"closed-form ray intersection ({np.linalg.norm(P0 - P_true[lm]) * 100:.1f} cm off)")
    ax_t.scatter(*P_ref[0:2], marker="x", s=70, color=C["blue"], lw=2, zorder=5,
                 label=f"Gauss-Newton refined ({np.linalg.norm(P_ref - P_true[lm]) * 100:.1f} cm off)")
    ax_t.scatter([], [], marker="s", color=TRUTH, label=f"known cameras ({len(T_obs)})")
    ax_t.set(title="(a) Triangulation: known cameras → unknown point", xlabel="x (m)", ylabel="y (m)")
    ax_t.set_aspect("equal", adjustable="datalim")
    legend_below(ax_t, ncol=1)

    R0, t0 = T_true[0:3, 0:3], T_true[0:3, 3]
    in_cam = lambda X: R0.T @ (X - t0)  # drawn in the true camera's own x-z plane
    for P in P_list:
        q = in_cam(P)
        ax_p.plot([q[0], 0.0], [q[2], 0.0], color=C["orange"], lw=0.8)
        ax_p.scatter(q[0], q[2], s=16, color=C["green"], zorder=3)
    ax_p.scatter(0, 0, marker="*", s=220, color=TRUTH, zorder=4, label="true camera centre")
    for T, mk, col, name in [(T_dlt, "o", C["orange"], "linear DLT"), (T_gn, "x", C["blue"], "Gauss-Newton refined")]:
        q = in_cam(T[0:3, 3])
        style = dict(color=col) if mk == "x" else dict(facecolor="none", edgecolor=col)
        ax_p.scatter(q[0], q[2], marker=mk, s=70, lw=1.8, zorder=5, **style,
                     label=f"{name} ({np.linalg.norm(T[0:3, 3] - t0) * 100:.1f} cm off)")
    ax_p.scatter([], [], s=16, color=C["green"], label=f"known 3D points ({len(P_list)})")
    ax_p.set(title="(b) PnP: known points → unknown camera", xlabel="camera x (m)", ylabel="camera z, depth (m)")
    ax_p.set_aspect("equal", adjustable="datalim")
    legend_below(ax_p, ncol=1)

    finish(fig, "pnp_estimation_concept", "Triangulation and PnP: the same rays, solved for opposite unknowns",
           f"(a) triangulate_landmark / refine_landmark_gn on bundle_adjustment.py's scene, landmark {lm}, top view; "
           "(b) pnp_estimation.py, true camera's x-z plane; seed 0", bottom=0.04)


HEX_GAIT_COLORS = {"tripod": C["red"], "ripple": C["orange"], "wave": C["blue"]}


def hexapod_body(ax, center=(0.0, 0.0), down=None, scale=1.0, label_legs=True):
    """Draws the hexapod body (heading +x) and its six feet at NOMINAL_FOOT_OFFSETS.
    Feet in `down` are filled (stance), the others hollow (swing). Returns foot positions."""
    from state_estimation_simulations.use_numpy import hexapod_gait_ekf as m
    c = np.asarray(center)
    feet = c + scale * m.NOMINAL_FOOT_OFFSETS
    ax.add_patch(Ellipse(c, scale * 0.42, scale * 0.24, fc="#e8eef6", ec="#5b6b7f", lw=1.4, zorder=2))
    arrow(ax, c, c + scale * np.array([0.17, 0.0]), "#5b6b7f", lw=1.2, ms=8, zorder=3)
    for i, (name, f) in enumerate(zip(m.LEG_NAMES, feet)):
        hip = c + scale * np.array([m.NOMINAL_FOOT_OFFSETS[i, 0] * 0.6, np.sign(m.NOMINAL_FOOT_OFFSETS[i, 1]) * 0.1])
        on = down is None or i in down
        ax.plot([hip[0], f[0]], [hip[1], f[1]], color="#5b6b7f", lw=1.6, alpha=1.0 if on else 0.35, zorder=2)
        ax.plot(*f, "o", ms=8, mfc=TRUTH if on else "white", mec=TRUTH, mew=1.3, zorder=4)
        if label_legs:
            ax.text(*(f + np.array([0.025, 0.0]) * scale), name, ha="left", va="center", fontsize=8.5,
                    color="#444444")
    return feet


def fig_hexapod_gait_ekf_concept():
    """hexapod_gait_ekf.py: the three gait schedules, what each stance foot measures, and stance geometry."""
    from state_estimation_simulations.use_numpy import hexapod_gait_ekf as m

    dt, period = 0.02, 1.2
    tpc = int(round(period / dt))
    sigma = 0.02
    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.25, 1.0, 1.0], hspace=0.55, wspace=0.18,
                          left=0.04, right=0.99, top=0.80, bottom=0.08)

    # (a) gait diagram: stance (dark) / swing (blank) bars per leg over two cycles
    for row, gait in enumerate(m.GAITS):
        ax = fig.add_subplot(gs[row, 0])
        stance = m.stance_mask(2 * tpc, tpc, gait)
        t = np.arange(2 * tpc) * dt
        for i in range(6):
            y = 5 - i
            on = stance[:, i]
            edges = np.diff(np.concatenate([[False], on, [False]]).astype(int))
            for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
                ax.add_patch(Rectangle((t[s], y - 0.35), (e - s) * dt, 0.7, fc=HEX_GAIT_COLORS[gait], ec="none"))
        duty, _ = m.GAITS[gait]
        ax.set(xlim=(0, 2 * period), ylim=(-0.6, 5.6), yticks=range(6), yticklabels=m.LEG_NAMES[::-1])
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(False)
        ax.text(0.0, 1.04, f"{gait}: duty {duty:.2f}, {int(stance.sum(axis=1)[0])} feet down at every tick",
                transform=ax.transAxes, fontsize=10, fontweight="bold", color=HEX_GAIT_COLORS[gait])
        if row == 0:
            ax.set_title("(a) Gait diagrams: bar = foot down (stance), gap = swing", loc="left", pad=22)
        if row == 2:
            ax.set_xlabel(f"time (s), two {period} s cycles")
        else:
            ax.set_xticklabels([])

    # (b) every stance foot measures -(b + omega x r_i): different feet, different readings
    ax_b = fig.add_subplot(gs[:, 1])
    ax_b.set_aspect("equal")
    b, omega = np.array([0.1, 0.0]), 0.2
    feet = hexapod_body(ax_b)
    for i, f in enumerate(feet):
        rel = -(b + omega * m.J @ m.NOMINAL_FOOT_OFFSETS[i])  # drawn to scale: 1 m/s = 1 m
        arrow(ax_b, f, f + rel, C["green"], lw=1.8, ms=10, zorder=5)
        side = np.sign(m.NOMINAL_FOOT_OFFSETS[i, 1])
        ax_b.text(*(f + 0.5 * rel + np.array([0.0, 0.04 * side])), f"{np.linalg.norm(rel):.3f}",
                  ha="center", va="center", fontsize=8.5, color=C["green"], fontweight="bold")
    arrow(ax_b, (-0.25, -0.33), (-0.12, -0.33), TRUTH, lw=1.8)
    ax_b.text(-0.10, -0.33, f"body: u = {b[0]} m/s,\nturning left at ω = {omega} rad/s", va="center", fontsize=9)
    ax_b.set(xlim=(-0.34, 0.34), ylim=(-0.52, 0.3))
    ax_b.axis("off")
    ax_b.set_title("(b) What each stance foot measures", loc="left")
    ax_b.text(0.0, 0.0, "green: foot velocity relative to the body, −(b + ω × r_i),\n"
              "to scale, speed in m/s. Same body motion, six different\n"
              "readings: that is what separates forward speed from turn rate.",
              transform=ax_b.transAxes, fontsize=8.5, color="#555555", va="bottom")

    # (c) same count, different places
    ax_c = fig.add_subplot(gs[:, 2])
    ax_c.set_aspect("equal")
    layouts = [("tripod triangle", [0, 4, 2], (0.0, 0.0)), ("one side", [0, 1, 2], (0.0, -0.58))]
    for label, legs, center in layouts:
        hexapod_body(ax_c, center=center, down=legs, label_legs=False)
        g = m.twist_geometry_std(m.NOMINAL_FOOT_OFFSETS[legs][None], np.ones((1, 3), dtype=bool), sigma)[0]
        ax_c.text(center[0] + 0.3, center[1] + 0.06, label, fontsize=10, fontweight="bold", va="center")
        ax_c.text(center[0] + 0.3, center[1] - 0.04, f"σ_u = {g[0]:.4f} m/s\nσ_ω = {g[2]:.4f} rad/s",
                  fontsize=9, va="center")
    ax_c.set(xlim=(-0.3, 0.72), ylim=(-1.1, 0.3))
    ax_c.axis("off")
    ax_c.set_title("(c) Three feet down, two layouts", loc="left")
    ax_c.text(0.0, 0.02, f"filled = stance. From one tick's readings alone\n(σ = {sigma} m/s per axis), "
              "three feet in a line along one\nside pin down u and ω about 40% worse than\nthe tripod's triangle.",
              transform=ax_c.transAxes, fontsize=8.5, color="#555555")

    finish(fig, "hexapod_gait_ekf_concept", "Hexapod leg odometry: every stance foot is a velocity sensor",
           "hexapod_gait_ekf.py: stance_mask (dt = 0.02 s, cycle 1.2 s), NOMINAL_FOOT_OFFSETS and twist_geometry_std. "
           "Arrows in (b) drawn to scale (1 m/s = 1 m).", tight=False)


def fig_hexapod_gait_ekf_extensions():
    """hexapod_gait_ekf.py: random foot drift (doc §4) and contact-timing jitter (doc §5)."""
    from state_estimation_simulations.use_numpy import hexapod_gait_ekf as m

    dt, period, n_steps, n_mc = 0.02, 1.2, 600, 100
    tpc = int(round(period / dt))
    meas_std, drift_std = 0.02, 0.02
    gaits = ["tripod", "wave"]
    ext_args = (n_steps, dt, tpc, gaits, np.array([0.1, 0.0, 0.2]), 0.05, 0.1, meas_std,
                np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05]), 0.1)
    lo3, hi3 = __import__("utils").averaged_nees_bounds(3, n_mc)

    fig, (ax_d, ax_s, ax_j) = plt.subplots(1, 3, figsize=(15, 5.4), gridspec_kw={"width_ratios": [0.8, 1.0, 1.25]})

    # (a) drift: same rng stream as hexapod_gait_ekf.main (seed + 3) for both filters
    width = 0.36
    for j, (label, R_f, hatch) in enumerate((("ignores drift", None, None),
                                             ("models drift", meas_std ** 2 + drift_std ** 2, "//"))):
        mc = m.run_monte_carlo(n_mc, *ext_args, np.random.default_rng(SEED + 3), foot_drift_std=drift_std,
                               R_meas_filter=R_f)
        for i, gait in enumerate(gaits):
            v = mc[(gait, False)]["nees_twist"][1:].mean()
            x = i + (j - 0.5) * width
            ax_d.bar(x, v, width * 0.92, color=HEX_GAIT_COLORS[gait], alpha=1.0 if j == 0 else 0.45,
                     hatch=hatch, ec="white", label=label if i == 0 else None)
            ax_d.text(x, v + 0.08, f"{v:.2f}", ha="center", fontsize=9)
            print(f"    drift {gait:<6s} {label}: twist NEES {v:.2f}")
    ax_d.axhspan(lo3, hi3, color=PRIOR, alpha=0.2, lw=0, label=f"consistent: 3, 95% band [{lo3:.2f}, {hi3:.2f}]")
    ax_d.axhline(3.0, color=TRUTH, lw=1)
    ax_d.set(xticks=[0, 1], xticklabels=gaits, ylabel="twist NEES", ylim=(0, 5.6),
             title=f"(a) Every foot drifts at {drift_std} m/s (doc §4)")
    ax_d.grid(axis="x", visible=False)
    legend_below(ax_d, ncol=1)

    # (b) swing foot speed: straight run at the nominal 0.1 m/s, no noise
    x_line = m.simulate_body_trajectory(300, dt, np.array([0.1, 0.0, 0.0]), 0.0, 0.0, np.random.default_rng(0))
    for gait in gaits:
        st, r, rel = m.simulate_feet(x_line, dt, tpc, gait, 0.1)
        i = 0
        lift = int(np.flatnonzero(st[100:-1, i] & ~st[101:, i])[0]) + 101
        land = lift + int(np.flatnonzero(st[lift:, i])[0])
        speed = np.linalg.norm(rel[lift:land, i] + x_line[lift:land, 3:5]
                               + x_line[lift:land, 5:6] * (m.J @ r[lift:land, i].T).T, axis=1)
        t_sw = (np.arange(lift, land) - lift + 1) * dt
        ax_s.plot(np.concatenate([[0], t_sw, [(land - lift + 1) * dt]]), np.concatenate([[0], speed, [0]]),
                  "-o", ms=3, color=HEX_GAIT_COLORS[gait], label=f"{gait}: swing {land - lift} ticks")
        for k_back, mk in ((1, "s"), (2, "D")):
            ax_s.plot(t_sw[-k_back], speed[-k_back], mk, ms=9, mfc="none", mec=HEX_GAIT_COLORS[gait], mew=1.6)
            ax_s.annotate(f"{speed[-k_back]:.3f}", (t_sw[-k_back], speed[-k_back]), textcoords="offset points",
                          xytext=(8, 2), fontsize=8.5, color=HEX_GAIT_COLORS[gait])
        print(f"    swing {gait}: {land - lift} ticks, peak {speed.max():.3f}, 1 tick {speed[-1]:.3f}, 2 ticks {speed[-2]:.3f}")
    ax_s.axhline(meas_std, color=TRUTH, lw=1, ls="--", label=f"reading noise σ = {meas_std} m/s")
    ax_s.plot([], [], "s", mfc="none", mec=TRUTH, label="1 tick before touchdown")
    ax_s.plot([], [], "D", mfc="none", mec=TRUTH, label="2 ticks before touchdown")
    ax_s.set(xlabel="time since liftoff (s)", ylabel="foot speed in the world (m/s)",
             title="(b) A foot read too early is still moving (doc §5)")
    legend_below(ax_s, ncol=2)

    # (c) jitter sweep: same rng stream as main (seed + 4) for all three filters
    sds = [0.02, 0.05, 0.1]
    res = {(g, f): [] for g in gaits for f in ("trust", "gate", "trim")}
    for sd in sds:
        mc_j = m.run_monte_carlo(n_mc, *ext_args, np.random.default_rng(SEED + 4), detect_noise_std=sd,
                                 gating_modes=(False, True))
        mc_t = m.run_monte_carlo(n_mc, *ext_args, np.random.default_rng(SEED + 4), detect_noise_std=sd,
                                 trim_ticks=int(np.ceil(2.0 * sd / dt)))
        for g in gaits:
            res[(g, "trust")].append(mc_j[(g, False)]["nees_twist"][1:].mean())
            res[(g, "gate")].append(mc_j[(g, True)]["nees_twist"][1:].mean())
            res[(g, "trim")].append(mc_t[(g, False)]["nees_twist"][1:].mean())
    styles = {"trust": ("-", "o", "trust detected times"), "gate": ("--", "s", "per-foot χ² gate"),
              "trim": (":", "^", "trim near detected edges")}
    for (g, f), vals in res.items():
        ls, mk, lab = styles[f]
        ax_j.plot(sds, vals, ls=ls, marker=mk, color=HEX_GAIT_COLORS[g], lw=1.6, label=f"{g}, {lab}")
        print(f"    jitter {g:<6s} {f:<5s}: " + ", ".join(f"{v:.2f}" for v in vals))
    ax_j.axhspan(lo3, hi3, color=PRIOR, alpha=0.2, lw=0)
    ax_j.axhline(3.0, color=TRUTH, lw=1)
    ax_j.set(xscale="log", yscale="log", xticks=sds, xlabel="detection-time jitter σ_detect (s)",
             ylabel="twist NEES (consistent: 3)", title=f"(c) Contact-timing jitter, {n_mc} runs each (doc §5)")
    ax_j.set_xticklabels([f"{v:g}" for v in sds])
    ax_j.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    plain_log_ticks(ax_j)
    legend_below(ax_j, ncol=2)

    finish(fig, "hexapod_gait_ekf_extensions", "Hexapod leg odometry: foot drift and contact-timing errors",
           f"hexapod_gait_ekf.py defaults, seed 0, tripod and wave gaits: run_monte_carlo ({n_mc} trials, 12 s), "
           "same draws for every filter in a panel. (b): one swing of leg L1 on a straight 0.1 m/s walk.", bottom=0.06)


def fig_hexapod_gait_ekf():
    """hexapod_gait_ekf.py: gait effect on twist RMSE (vs. the (q r)^(1/4) estimate) and one slipping leg."""
    from state_estimation_simulations.use_numpy import hexapod_gait_ekf as m

    dt, period, n_steps, n_mc = 0.02, 1.2, 600, 100
    tpc = int(round(period / dt))
    twist0 = np.array([0.1, 0.0, 0.2])
    lin_std, ang_std, meas_std = 0.05, 0.1, 0.02
    init_std = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])
    gaits = list(m.GAITS)
    sim_args = (n_steps, dt, tpc, gaits, twist0, lin_std, ang_std, meas_std, init_std, 0.1)
    slip = (1, 4.0, 8.0, np.array([0.0, 0.1]))

    # same rng streams as hexapod_gait_ekf.main at seed 0
    _, _, feet = m.simulate_trial(*sim_args, np.random.default_rng(SEED))
    mc = m.run_monte_carlo(n_mc, *sim_args, np.random.default_rng(SEED + 1))
    mc_slip = m.run_monte_carlo(n_mc, *sim_args, np.random.default_rng(SEED + 2), slip=slip,
                                gating_modes=(False, True))

    fig, (ax_u, ax_w, ax_s) = plt.subplots(1, 3, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    q = {3: (lin_std * dt) ** 2, 5: (ang_std * dt) ** 2}
    for ax, col, unit, name in ((ax_u, 3, "m/s", "forward speed u"), (ax_w, 5, "rad/s", "turn rate ω")):
        for j, gait in enumerate(gaits):
            stance, r, _ = feet[gait]
            geo = np.nanmean(m.twist_geometry_std(r[1:], stance[1:], meas_std), axis=0)[col - 3]
            rmse = np.sqrt(mc[(gait, False)]["mse"][1:, col].mean())
            pred = (q[col] * geo ** 2) ** 0.25
            ax.bar(j, rmse, color=HEX_GAIT_COLORS[gait], width=0.6)
            ax.text(j, rmse * 0.5, f"{rmse:.4f}", ha="center", color="white", fontweight="bold", fontsize=9)
            ax.plot([j - 0.35, j + 0.35], [pred, pred], color=TRUTH, lw=2,
                    label="(q·s²)^¼ estimate" if j == 0 else None)
            print(f"    {gait:<7s} {name}: RMSE {rmse:.4f}, estimate {pred:.4f}")
        ax.set(xticks=range(3), xticklabels=[f"{g}\n({3 + i} feet)" for i, g in enumerate(gaits)],
               ylabel=f"RMSE ({unit})", title=f"{name}: RMSE over {n_mc} runs")
        ax.legend(loc="lower right", fontsize=8.5)
        ax.grid(axis="x", visible=False)

    t = np.arange(n_steps + 1) * dt
    ax_s.axvspan(slip[1], slip[2], color=C["gold"], alpha=0.25, lw=0, label="L2 slipping sideways (0.1 m/s)")
    for gait in ("tripod", "wave"):
        for gating, ls in ((False, "-"), (True, "--")):
            ax_s.plot(t, np.sqrt(mc_slip[(gait, gating)]["mse"][:, 4]), color=HEX_GAIT_COLORS[gait], ls=ls, lw=1.4,
                      label=f"{gait}, {'per-foot χ² gate' if gating else 'trust every foot'}")
    ax_s.set(xlabel="time (s)", ylabel="sideways velocity v RMSE (m/s)", ylim=(0, 0.036), xlim=(0, t[-1]),
             title=f"One slipping leg: RMSE over {n_mc} runs")
    ax_s.text(8.3, 0.0335, "trusting every foot:\ntripod: L2 is 1/3 of the update,\n  down half the time\n"
              "wave: 1/5 of the update,\n  down 5/6 of the time", fontsize=8.5, color="#444444", va="top")
    legend_below(ax_s, ncol=3)

    finish(fig, "hexapod_gait_ekf", "Hexapod leg odometry: the gait barely matters, a slipping foot does",
           f"hexapod_gait_ekf.py defaults, seed 0: run_monte_carlo ({n_mc} trials, 12 s). "
           "Bars: filter RMSE; black line: steady-state estimate (q·s²)^¼ with s = one tick's std for that gait.",
           bottom=0.03)


def fig_hexapod_contact_inekf():
    """hexapod_contact_inekf.py: yaw consistency (InEKF vs world EKF) and one slipping leg."""
    from dataclasses import replace
    from state_estimation_simulations.use_numpy import hexapod_contact_inekf as m
    from state_estimation_simulations.utils import averaged_nees_bounds

    cfg, n_mc = m.Config(), 50
    colors = {"inekf": C["blue"], "world": C["red"]}
    labels = {"inekf": "invariant EKF", "world": "world-frame EKF"}
    t = np.arange(cfg.n_steps + 1) * cfg.dt
    # same rng streams as hexapod_contact_inekf.main at seed 0 (sweep: seed + 4, slip: seed + 2)
    sweep = {rs: m.run_monte_carlo(replace(cfg, init_rot_std=rs), n_mc, ["tripod"], np.random.default_rng(SEED + 4))
             for rs in (0.02, 0.1, 0.3)}
    slip = (1, 4.0, 8.0, np.array([0.0, 0.1, 0.0]))
    mc_slip = m.run_monte_carlo(cfg, n_mc, ["tripod", "wave"], np.random.default_rng(SEED + 2), kinds=("inekf",),
                                gating_modes=(False, True), slip=slip)

    fig, (ax_y, ax_n, ax_s) = plt.subplots(1, 3, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.2, 1, 1.2]})
    for kind in m.FILTERS:
        a = sweep[0.1][("tripod", kind, False)]
        ax_y.plot(t, np.degrees(np.sqrt(a["mse_rot"][:, 2])), color=colors[kind], lw=1.6,
                  label=f"{labels[kind]}: actual RMSE")
        ax_y.plot(t, np.degrees(np.sqrt(a["yaw_var"])), color=colors[kind], lw=1.6, ls="--",
                  label=f"{labels[kind]}: reported σ")
        print(f"    0.1 rad {kind}: final yaw RMSE {np.degrees(np.sqrt(a['mse_rot'][-1, 2])):.2f} deg, "
              f"reported {np.degrees(np.sqrt(a['yaw_var'][-1])):.2f}")
    ax_y.set(xlabel="time (s)", ylabel="yaw error (deg)", ylim=(0, None), xlim=(0, t[-1]),
             title=f"Yaw, initial rotation σ 0.1 rad ({n_mc} runs)")
    legend_below(ax_y)

    lo, hi = averaged_nees_bounds(9, n_mc)
    ax_n.axhspan(lo, hi, color=PRIOR, alpha=0.25, lw=0, label="95% band for 9 DoF")
    for j, rs in enumerate(sweep):
        for dx, kind in ((-0.18, "inekf"), (0.18, "world")):
            val = sweep[rs][("tripod", kind, False)]["nees"][1:].mean()
            ax_n.bar(j + dx, val, width=0.34, color=colors[kind], label=labels[kind] if j == 0 else None)
            ax_n.text(j + dx, val * 1.08, f"{val:.1f}", ha="center", fontsize=8.5)
            print(f"    init {rs} rad {kind}: NEES {val:.2f}")
    ax_n.set_yscale("log")
    plain_log_ticks(ax_n)
    ax_n.set(xticks=range(3), xticklabels=[f"{rs} rad" for rs in sweep], xlabel="initial rotation σ",
             ylabel="average NEES of (R, v, p)", ylim=(5, 900), title="Consistency vs. initial rotation error")
    ax_n.grid(axis="x", visible=False)
    legend_below(ax_n)

    ax_s.axvspan(slip[1], slip[2], color=C["gold"], alpha=0.25, lw=0, label="L2 slipping sideways (0.1 m/s)")
    for gait, col in (("tripod", C["purple"]), ("wave", C["green"])):  # red/blue mean the filters elsewhere
        for gating, ls in ((False, "-"), (True, "--")):
            a = mc_slip[(gait, "inekf", gating)]
            ax_s.plot(t, np.sqrt(a["mse_vel"].sum(axis=1)), color=col, ls=ls, lw=1.4,
                      label=f"{gait}, {'per-foot χ² gate' if gating else 'trust every foot'}")
    ax_s.set(xlabel="time (s)", ylabel="velocity RMSE (m/s)", xlim=(0, t[-1]), ylim=(0, 0.05),
             title=f"InEKF, one slipping leg ({n_mc} runs)")
    legend_below(ax_s, ncol=3)

    finish(fig, "hexapod_contact_inekf", "Contact-aided EKF on a hexapod: only the invariant filter knows yaw is unobservable",
           f"hexapod_contact_inekf.py defaults, seed 0: run_monte_carlo ({n_mc} trials, 12 s, tripod unless noted). "
           "The two filters are about equally accurate; the world EKF reports far less yaw uncertainty than it has.",
           bottom=0.05)


def fig_lidar_primitives_concept():
    """lidar_primitives.py: what the extraction sees in one scan, and the L-corridor of experiment 4."""
    from state_estimation_simulations.use_numpy import lidar_primitives as m
    from state_estimation_simulations.lidar_utils import (ExtractionConfig, LidarConfig, corridor_scene, inlier_threshold, l_corridor_scene,
                             l_corridor_trajectory)

    noise = 0.01
    src, tgt, _, T_tgt = m.make_scan_pair(corridor_scene(pillars=True), [SEED, 0], LidarConfig(range_noise_std=noise),
                                          ExtractionConfig(inlier_dist=inlier_threshold(noise)))
    P = tgt.points @ T_tgt[:3, :3].T + T_tgt[:3, 3]
    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.6, 1]})
    used = np.zeros(len(P), bool)
    for pl in tgt.planes:
        used[pl["inliers"]] = True
    rest = ~used
    for cyl in tgt.cylinders:
        rest[cyl["inliers"]] = False
    ax_s.scatter(P[rest, 0], P[rest, 1], s=1, c=PRIOR, lw=0, label="not assigned")
    walls = [pl for pl in tgt.planes if abs((T_tgt[:3, :3] @ pl["n"])[2]) < 0.5]
    flat = [pl for pl in tgt.planes if abs((T_tgt[:3, :3] @ pl["n"])[2]) >= 0.5]
    for group, col, lab in ((flat, C["gold"], "floor / ceiling plane"), (walls, C["blue"], "wall plane")):
        idx = np.concatenate([pl["inliers"] for pl in group])
        ax_s.scatter(P[idx, 0], P[idx, 1], s=1, c=col, lw=0, label=f"{lab} ({len(group)})")
    idx = np.concatenate([cyl["inliers"] for cyl in tgt.cylinders])
    ax_s.scatter(P[idx, 0], P[idx, 1], s=3, c=C["green"], lw=0, label=f"cylinder ({len(tgt.cylinders)})")
    ax_s.plot(*T_tgt[:2, 3], marker="^", color="k", ms=8, ls="")
    c = T_tgt[:2, 3]
    ax_s.set(xlim=(c[0] - 16, c[0] + 16), ylim=(-3.6, 3.6), aspect="equal", xlabel="x (m)", ylabel="y (m)",
             title="One scan of the corridor with pillars, as extracted (seen from above; triangle: sensor)")
    ax_s.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, markerscale=6, fontsize=8.5)
    print(f"    scan: {len(P)} points, {len(tgt.planes)} planes, {len(tgt.cylinders)} cylinders, "
          f"{rest.sum()} unassigned")

    scene = l_corridor_scene()
    for r in scene.rects:
        if abs(r.n[2]) > 0.5:
            continue
        ends = np.array([r.c - r.half_u * r.u - r.half_v * r.v, r.c + r.half_u * r.u + r.half_v * r.v])
        ax_l.plot(ends[:, 0], ends[:, 1], color=C["blue"], lw=2)
    for cyl in scene.cylinders:
        ax_l.add_patch(plt.Circle(cyl.base[:2], cyl.r, color=C["green"]))
    traj = np.array([t for _, t in l_corridor_trajectory()])
    ax_l.plot(traj[:, 0], traj[:, 1], "o-", color=TRUTH, ms=2.5, lw=0.8, label=f"{len(traj)} sensor poses, 1 m apart")
    ax_l.set(xlim=(-23, 6), ylim=(-6, 23), aspect="equal", xlabel="x (m)", ylabel="y (m)",
             title="Experiment 4: L-shaped corridor (walls, pillars, path)")
    ax_l.legend(loc="lower right", fontsize=8.5)
    finish(fig, "lidar_primitives_concept", "LiDAR primitives: planes and cylinders extracted from one scan",
           "lidar_primitives.py, seed 0: make_scan_pair (32 rings x 360, 1 cm range noise), sequential RANSAC "
           "with competing plane and cylinder hypotheses.", bottom=0.02)


def fig_lidar_primitives_extraction():
    """lidar_primitives.py experiment 1: radius bias of the two cylinder fits, and detection vs. point density."""
    from state_estimation_simulations.use_numpy import lidar_primitives as m
    from state_estimation_simulations.lidar_utils import corridor_scene

    noise_levels, az = [0.005, 0.01, 0.02, 0.04], [180, 360, 720]
    out = m.run_extraction_experiment(corridor_scene(pillars=True), noise_levels, az, 10, SEED, min_points=150)
    x = 1e3 * np.array(noise_levels)
    fig, (ax_b, ax_p, ax_d) = plt.subplots(1, 3, figsize=(15, 5.2))
    cols = {180: C["purple"], 360: C["blue"], 720: C["orange"]}
    for j, n_az in enumerate(az):
        ax_b.plot(x, out["cyl_radius_bias_mm_mean"][:, j], "o-", color=cols[n_az], label=f"orthogonal fit, {n_az}/ring")
        ax_b.plot(x, out["cyl_radius_bias_mm_range_mean"][:, j], "s--", color=cols[n_az], label=f"range fit, {n_az}/ring")
        ax_p.plot(x, out["cyl_pos_mm_median"][:, j], "o-", color=cols[n_az])
        ax_p.plot(x, out["cyl_pos_mm_range_median"][:, j], "s--", color=cols[n_az])
        ax_d.plot(x, 100 * out["cyl_detection"][:, j], "o-", color=cols[n_az], label=f"{n_az} points per ring")
        print(f"    {n_az}/ring: radius bias ODR {np.round(out['cyl_radius_bias_mm_mean'][:, j], 1)} "
              f"range {np.round(out['cyl_radius_bias_mm_range_mean'][:, j], 1)} mm; axis pos median ODR "
              f"{np.round(out['cyl_pos_mm_median'][:, j], 1)} range {np.round(out['cyl_pos_mm_range_median'][:, j], 1)};"
              f" detection {np.round(out['cyl_detection'][:, j], 3)}; false positives {out['false_positives'][:, j]}")
    ax_b.axhline(0, color=TRUTH, lw=0.8)
    ax_b.set(xlabel="range noise σ (mm)", ylabel="mean radius error (mm)", xticks=x,
             title="Pillar radius: bias of the two fits")
    legend_below(ax_b, ncol=2)
    ax_p.set(xlabel="range noise σ (mm)", ylabel="median axis position error (mm)", xticks=x,
             title="Pillar axis position (solid: orthogonal, dashed: range)")
    ax_d.set(xlabel="range noise σ (mm)", ylabel="pillars detected (%)", xticks=x, ylim=(50, 102),
             title="Detection: more points per ring is not better")
    legend_below(ax_d, ncol=3)
    print(f"    planes: detection {out['plane_detection'].min():.2f}-{out['plane_detection'].max():.2f}, normal error "
          f"median {out['plane_angle_deg_median'].min():.3f}-{out['plane_angle_deg_median'].max():.3f} deg, offset "
          f"median {out['plane_offset_mm_median'].min():.2f}-{out['plane_offset_mm_median'].max():.2f} mm")
    finish(fig, "lidar_primitives_extraction", "Extraction: a range-only noise model biases the usual cylinder fit",
           "lidar_primitives.py experiment 1 (run_extraction_experiment, 10 sensor poses per cell, seed 0; pillars "
           "with at least 150 points). Both fits use the same RANSAC inliers.", bottom=0.06)


def fig_lidar_primitives():
    """lidar_primitives.py experiments 2-4: convergence basin, degeneracy, planes as landmarks."""
    from state_estimation_simulations.use_numpy import lidar_primitives as m
    from state_estimation_simulations.lidar_utils import METHODS

    labels = {"point_to_point": "point-to-point", "point_to_plane": "point-to-plane", "primitive": "primitives"}
    cols = {"point_to_point": C["red"], "point_to_plane": C["blue"], "primitive": C["green"]}
    levels = [(0.1, 2), (0.25, 5), (0.5, 10), (1.0, 20), (2.0, 30)]
    basin = m.run_basin_experiment(levels, 10, 5, SEED)
    deg = m.run_degeneracy_experiment(10, SEED)
    lmk = m.run_landmark_experiment(5, SEED)

    fig, (ax_b, ax_g, ax_l) = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw={"width_ratios": [1, 1.3, 0.9]})
    for meth in METHODS:
        ax_b.plot(range(len(levels)), 100 * basin[meth]["success"], "o-", color=cols[meth], label=labels[meth])
        print(f"    basin {meth}: success {basin[meth]['success']}, trans mm {np.round(basin[meth]['trans_mm'], 1)}, "
              f"iters {basin[meth]['iters']}")
    ax_b.set(xticks=range(len(levels)), xticklabels=[f"{t} m\n{r}°" for t, r in levels], ylim=(-3, 103),
             xlabel="initial error", ylabel="converged within 5 cm and 1° (%)", title="Convergence basin (50 starts)")
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=8.5)

    xs = np.arange(len(METHODS))
    for k, (scene, title) in enumerate((("corridor", "planes only"), ("corridor_pillars", "with pillars"))):
        for j, meth in enumerate(METHODS):
            r = deg[scene][meth]
            act, pred = np.median(np.abs(r["axis_err_mm"])), np.median(r["axis_sigma_mm"])
            x0 = k * 3.6 + j
            ax_g.bar(x0 - 0.2, act, width=0.38, color=cols[meth])
            ax_g.bar(x0 + 0.2, pred, width=0.38, color=cols[meth], alpha=0.35, hatch="//", edgecolor=cols[meth])
            fmt = lambda v, sig: f"{v:,.0f}" if v >= 1000 else f"{v:.{sig}g}"
            ax_g.text(x0 - 0.2, act * 1.3, fmt(act, 3), ha="center", fontsize=7.5)
            ax_g.text(x0 + 0.2, pred * 1.3, fmt(pred, 2), ha="center", fontsize=7.5, color="#555555")
            print(f"    degeneracy {scene} {meth}: |axis err| median {act:.3g} mm, predicted sigma {pred:.3g} mm, "
                  f"eig ratio {np.median(r['eig_ratio']):.2e}")
        ax_g.text(k * 3.6 + 1, 3e5, title, ha="center", fontsize=10, fontweight="bold")
    ax_g.set_yscale("log")
    ax_g.set(ylim=(0.05, 1e6), xticks=[k * 3.6 + j for k in range(2) for j in range(3)],
             xticklabels=[labels[meth].replace("-to-", "-to-\n") for _ in range(2) for meth in METHODS],
             ylabel="error along the corridor (mm)",
             title="Degeneracy: actual (solid) vs. predicted σ (hatched)")
    ax_g.tick_params(axis="x", labelsize=8)
    ax_g.grid(axis="x", visible=False)

    runs = np.arange(len(lmk["odometry"]["end_mm"]))
    for name, col, mk in (("odometry", C["orange"], "o"), ("graph", C["purple"], "s")):
        ax_l.plot(runs, lmk[name]["end_mm"], mk + "-", color=col, label=f"{name}: end-point drift")
        ax_l.plot(runs, lmk[name]["ate_mm"], mk + ":", color=col, label=f"{name}: ATE (aligned)")
        print(f"    landmarks {name}: end mm {np.round(lmk[name]['end_mm'], 2)}, ATE mm {np.round(lmk[name]['ate_mm'], 2)},"
              f" end deg {np.round(lmk[name]['end_deg'], 4)}")
    print(f"    landmarks per run {lmk['n_landmarks']}")
    ax_l.set(xticks=runs, xlabel="run (noise realization)", ylabel="error after 40 scans (mm)", ylim=(0, None),
             title="Planes as landmarks (L-corridor)")
    legend_below(ax_l, ncol=2)
    finish(fig, "lidar_primitives", "Scan registration with primitives: accurate, honest about degeneracy, narrower basin",
           "lidar_primitives.py experiments 2-4 at their defaults (seed 0, 1 cm range noise): run_basin_experiment, "
           "run_degeneracy_experiment, run_landmark_experiment.", bottom=0.04)


# --------------------------------------------------------------------------- optimization-doc figures
# These four figures illustrate docs/optimization/*.md. They use the docs' own
# toy problems and formulas; where a repo function exists it is called (utils.symbolic_eliminate,
# bundle_adjustment's scene), otherwise the formula is evaluated directly and printed.

def fig_gauss_newton_tangent():
    """gauss_newton.md §6/§9/§10: Gauss-Newton on r(x) = x^2 - 4 from x = 3."""
    r = lambda x: x ** 2 - 4.0
    J = lambda x: 2.0 * x
    x = 3.0
    xs = [x]
    for _ in range(3):
        x = x - J(x) * r(x) / J(x) ** 2        # Gauss-Newton step: J^T J dx = -J^T r (scalar)
        xs.append(x)
    res = [abs(r(v)) for v in xs]
    print("    iterates x_k      :", ", ".join(f"{v:.8g}" for v in xs))
    print("    |r(x_k)|          :", ", ".join(f"{v:.3g}" for v in res))
    print(f"    tangent at x=3: r = 5 + 6 (x - 3); root x = {xs[1]:.4f}")

    fig, (ax_c, ax_r) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.15, 1]})
    g = np.linspace(1.5, 3.4, 300)
    ax_c.plot(g, r(g), color=TRUTH, lw=2.2, label=r"$r(x)=x^2-4$")
    ax_c.plot(g, r(3.0) + J(3.0) * (g - 3.0), color=C["red"], lw=1.8, label=f"tangent at $x=3$ (root {xs[1]:.4f})")
    ax_c.plot(g, r(xs[1]) + J(xs[1]) * (g - xs[1]), color=C["blue"], lw=1.4, ls="--",
              label=f"next tangent at $x={xs[1]:.4f}$ (root {xs[2]:.4f})")
    ax_c.axhline(0, color=PRIOR, lw=1)
    ax_c.scatter([3.0], [r(3.0)], color=C["red"], zorder=4)
    ax_c.annotate(r"$x_0=3,\ r=5$", (3.0, 5.0), (2.35, 5.6), arrowprops=dict(arrowstyle="-", color=PRIOR), fontsize=9.5)
    ax_c.scatter([xs[1]], [0], color=C["red"], zorder=4)
    ax_c.annotate(f"$x_1={xs[1]:.4f}$", (xs[1], 0), (2.25, -2.3), arrowprops=dict(arrowstyle="-", color=PRIOR), fontsize=9.5)
    ax_c.scatter([2.0], [0], marker="*", s=170, color=C["gold"], edgecolor=TRUTH, zorder=5, label=r"true root $x=2$")
    ax_c.set(title="Gauss-Newton in 1D: jump to where the tangent hits zero", xlabel="x", ylabel="r(x)", ylim=(-3.5, 7.5))
    ax_c.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))

    k = np.arange(len(res))
    ax_r.semilogy(k, res, "o-", color=C["red"], lw=2, ms=7)
    for ki, v in zip(k, res):
        ax_r.annotate(f"{v:.3g}", (ki, v), (6, 7), textcoords="offset points", fontsize=9.5)
    ax_r.set(title=r"Residual magnitude $|r(x_k)|$ per iteration", xlabel="iteration k", ylabel=r"$|r(x_k)|$",
             xticks=k, ylim=(1e-5, 40))
    finish(fig, "gauss_newton_tangent", "Gauss-Newton on r(x) = x² − 4, starting from x = 3",
           "Residual and Jacobian J = 2x as in gauss_newton.md §6; each step solves J²Δx = −J·r; "
           "the iterates come from running that update")


def fig_lm_gradient_descent():
    """levenberg_marquardt.md §3: gradient descent on f = 1/2 (x^2 + 100 y^2) from (10, 1)."""
    tol, start = 0.01, np.array([10.0, 1.0])
    grad = lambda p: np.array([p[0], 100.0 * p[1]])

    def run(alpha, done, max_steps):
        p, path = start.copy(), [start.copy()]
        while len(path) <= max_steps and not done(np.linalg.norm(p)):
            p = p - alpha * grad(p)
            path.append(p.copy())
        return np.array(path)

    conv = run(0.019, lambda d: d < tol, 5000)
    div = run(0.021, lambda d: d > 1e8, 5000)
    n_conv, n_div = len(conv) - 1, len(div) - 1
    Jm = np.diag([1.0, 10.0]); e0 = np.array([start[0], 10.0 * start[1]])
    gn = start + np.linalg.solve(Jm.T @ Jm, -Jm.T @ e0)
    print(f"    alpha=0.019: {n_conv} steps to |p| < {tol}, final p = ({conv[-1, 0]:.5f}, {conv[-1, 1]:.2e}), |p| = {np.linalg.norm(conv[-1]):.5f}")
    print(f"    alpha=0.021: |p| exceeds 1e8 after {n_div} steps (|p| = {np.linalg.norm(div[-1]):.3g}); |p| at steps 0,10,50 = "
          + ", ".join(f"{np.linalg.norm(div[s]):.3g}" for s in (0, 10, 50)))
    print(f"    Gauss-Newton: one step from (10, 1) lands at ({gn[0]:.3g}, {gn[1]:.3g})")

    fig, (ax_c, ax_d) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.1, 1]})
    X, Y = np.meshgrid(np.linspace(-1.5, 11.5, 300), np.linspace(-3.2, 3.2, 300))
    ax_c.contour(X, Y, 0.5 * (X ** 2 + 100 * Y ** 2), levels=[1, 4, 10, 25, 50, 75], colors=PRIOR, linewidths=0.9)
    ax_c.plot(conv[:, 0], conv[:, 1], "-", color=C["blue"], lw=1.2, label=f"$\\alpha=0.019$ converges ({n_conv} steps)")
    ax_c.plot(conv[:12, 0], conv[:12, 1], "o", color=C["blue"], ms=3.5)
    m = 9
    ax_c.plot(div[:m + 1, 0], div[:m + 1, 1], "-s", color=C["red"], lw=1.2, ms=3.5,
              label=f"$\\alpha=0.021$ diverges (first {m} steps; $|y|$ grows 1.1× per step)")
    ax_c.annotate("", xy=gn, xytext=start, arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=2.2))
    ax_c.plot([], [], color=C["green"], lw=2.2, label="Gauss-Newton: 1 step")
    ax_c.scatter([0], [0], marker="*", s=190, color=C["gold"], edgecolor=TRUTH, zorder=5, label="minimum")
    ax_c.scatter(*start, color=TRUTH, zorder=5)
    ax_c.annotate("start (10, 1)", start, (7.2, 2.5), arrowprops=dict(arrowstyle="-", color=PRIOR), fontsize=9.5)
    ax_c.set(title=r"Contours of $f=\frac{1}{2}(x^2+100y^2)$ and the three paths", xlabel="x", ylabel="y", ylim=(-3.2, 3.2))
    ax_c.legend(loc="lower left", fontsize=8, bbox_to_anchor=(0.0, 0.0))

    ax_d.semilogy(np.linalg.norm(conv, axis=1), color=C["blue"], lw=2, label=r"$\alpha=0.019$")
    nd = min(n_div, 200)
    ax_d.semilogy(np.linalg.norm(div[:nd + 1], axis=1), color=C["red"], lw=2, label=r"$\alpha=0.021$")
    ax_d.axhline(tol, color=PRIOR, ls="--", lw=1)
    ax_d.text(5, tol * 1.35, f"tolerance {tol}", color="#555555", fontsize=9)
    ax_d.scatter([n_conv], [np.linalg.norm(conv[-1])], color=C["blue"], zorder=4)
    ax_d.annotate(f"{n_conv} steps", (n_conv, np.linalg.norm(conv[-1])), (-62, 18), textcoords="offset points", fontsize=9.5)
    ax_d.set(title="Distance to the minimum per step", xlabel="step", ylabel=r"$\|(x,y)\|$", xlim=(0, max(nd, n_conv) + 5))
    ax_d.legend(loc="center right")
    finish(fig, "lm_gradient_descent", "Gradient descent in a narrow valley: the step size must stay below 2/100",
           "f = ½(x² + 100y²) from (10, 1), fixed step size α; the converged path is plotted to |(x,y)| < 0.01")


def fig_sparse_fill_in_orderings():
    """sparse_cholesky_factorization.md §5-§9: ordering vs fill-in for a BA-like Hessian."""
    from state_estimation_simulations.use_numpy import bundle_adjustment as m
    from state_estimation_simulations.utils import symbolic_eliminate

    rng = np.random.default_rng(SEED)
    K = (800.0, 800.0, 320.0, 240.0)
    T_true, P_all = m.generate_ground_truth_scene(8, 60, 5.0, 180.0, 2.0, rng)
    P_true, obs = m.build_observations(T_true, P_all, K, 1.0, 2, rng)
    n_cam, n_lm = len(T_true), len(P_true)
    pairs = sorted({(i, j) for i, j, _ in obs})
    toy_pairs = [(i, j) for i in range(4) for j in range(12)]

    def orderings(nc, nl, prs):
        first = {}
        for i, j in sorted(prs):
            first.setdefault(j, i)
        inter = []
        for i in range(nc):
            inter += [("c", i)] + [("l", j) for j in range(nl) if first[j] == i]
        return {"poses first": [("c", i) for i in range(nc)] + [("l", j) for j in range(nl)],
                "landmarks first": [("l", j) for j in range(nl)] + [("c", i) for i in range(nc)],
                "interleaved": inter}

    def analyse(nc, nl, prs, order, scalar):
        """Returns (H_perm, L_perm, nnz_L, fill_in) over blocks (scalar=False) or 6/3-dof scalars."""
        size = {"c": 6 if scalar else 1, "l": 3 if scalar else 1}
        base = {}; n = 0
        for kind, cnt in (("c", nc), ("l", nl)):
            for idx in range(cnt):
                base[(kind, idx)] = list(range(n, n + size[kind])); n += size[kind]
        edges = set()
        for blk in base.values():
            edges |= {(a, b) for a in blk for b in blk if a < b}
        for i, j in prs:
            edges |= {(a, b) for a in base[("c", i)] for b in base[("l", j)]}
        scal_order = [s for blk in order for s in base[blk]]
        sep, _ = symbolic_eliminate(n, edges, scal_order)
        pos = {v: k for k, v in enumerate(scal_order)}
        H = np.eye(n, dtype=bool); L = np.eye(n, dtype=bool)
        for a, b in edges:
            H[pos[a], pos[b]] = H[pos[b], pos[a]] = True
        for v, nb in sep.items():
            for u in nb:
                L[max(pos[u], pos[v]), min(pos[u], pos[v])] = True
        nnz_H_lower = int(np.tril(H).sum()); nnz_L = int(L.sum())
        return H, L, nnz_L, nnz_L - nnz_H_lower, nnz_H_lower, n

    print("    scene: 8 cameras x 48 landmarks (bundle_adjustment defaults, seed 0), "
          f"{len(obs)} observations -> {len(pairs)} distinct camera-landmark pairs")
    print(f"    {'scene / level':<38s}{'ordering':<18s}{'n':>5s}{'nnz(H) lower':>14s}{'nnz(L)':>9s}{'fill-in':>9s}{'dense n(n+1)/2':>16s}")
    results = {}
    for scene, nc, nl, prs in (("BA scene", n_cam, n_lm, pairs), ("toy 4 cams x 12 lms", 4, 12, toy_pairs)):
        for scalar in (False, True):
            level = "6/3-dof scalars" if scalar else "blocks"
            for name, order in orderings(nc, nl, prs).items():
                res = analyse(nc, nl, prs, order, scalar)
                results[(scene, scalar, name)] = res
                H, L, nnz_L, fill, nnzH, n = res
                print(f"    {scene + ', ' + level:<38s}{name:<18s}{n:>5d}{nnzH:>14d}{nnz_L:>9d}{fill:>9d}{n * (n + 1) // 2:>16d}")

    fig, axes = plt.subplots(2, 3, figsize=(13, 9.2))
    for c, name in enumerate(["poses first", "landmarks first", "interleaved"]):
        H, L, nnz_L, fill, nnzH, n = results[("BA scene", True, name)]
        Hl = np.tril(H)
        ax = axes[0, c]
        ax.spy(Hl, markersize=1.3, color=C["blue"])
        ax.set_title(f"H, {name}\nnnz(lower H) = {nnzH}", fontsize=10.5)
        ax = axes[1, c]
        ax.spy(L & Hl, markersize=1.3, color=C["blue"])
        ax.spy(L & ~Hl, markersize=1.3, color=C["red"])
        ax.set_title(f"L, {name}\nnnz(L) = {nnz_L},  fill-in = {fill}", fontsize=10.5)
    for ax in axes.ravel():
        ax.grid(False)
        ax.tick_params(labelsize=8)
        for s in ax.spines.values():
            s.set_visible(True)
    axes[0, 0].set_ylabel("sparsity of H (lower triangle)")
    axes[1, 0].set_ylabel("sparsity of L (red = fill-in)")
    fig.subplots_adjust(top=0.84, bottom=0.04, left=0.07, right=0.98, hspace=0.3, wspace=0.18)
    finish(fig, "sparse_fill_in_orderings", "Elimination order decides the fill-in of the bundle-adjustment Cholesky factor",
           f"{n_cam} cameras (6-dof) + {n_lm} landmarks (3-dof) = {6 * n_cam + 3 * n_lm} unknowns, {len(pairs)} visible camera-landmark pairs "
           "(bundle_adjustment.py defaults, seed 0); fill-in from utils.symbolic_eliminate", tight=False)


def fig_robust_kernel_weights():
    """pose_graph_optimization.md §16: IRLS weights of L2, Huber, Cauchy, Geman-McClure, DCS."""
    delta = k = 1.0                     # §16.1 tiny example: delta = k = 1
    Phi = 12.59                         # §16.4/§16.5: chi-square(6) 95% quantile
    w_l2 = lambda e: np.ones_like(e)
    w_huber = lambda e: np.where(e <= delta, 1.0, delta / np.maximum(e, 1e-12))
    w_cauchy = lambda e: 1.0 / (1.0 + (e / k) ** 2)
    w_gm = lambda e: 1.0 / (1.0 + e ** 2) ** 2          # §16.4 table, unit scale
    s_dcs = lambda e2: np.minimum(1.0, 2.0 * Phi / (Phi + e2))
    w_dcs = lambda e: s_dcs(e ** 2) ** 2                # Omega_DCS = s^2 Omega

    print("    weight w(e) at e = 0.5, 3.548 (= sqrt(12.59)), 10:")
    for name, fn in (("L2", w_l2), ("Huber (delta=1)", w_huber), ("Cauchy (k=1)", w_cauchy),
                     ("Geman-McClure (unit scale)", w_gm), ("DCS s^2 (Phi=12.59)", w_dcs)):
        vals = fn(np.array([0.5, np.sqrt(Phi), 10.0]))
        print(f"      {name:<28s}" + "  ".join(f"{v:.4g}" for v in vals))
    print(f"    DCS s at e^2 = 12.59, 20, 100, 10^4: " + ", ".join(f"{float(s_dcs(v)):.4g}" for v in (Phi, 20.0, 100.0, 1e4)))
    print(f"    sqrt(Phi) = {np.sqrt(Phi):.3f}")

    fig, (ax_w, ax_s) = plt.subplots(1, 2)
    e = np.linspace(0.0, 10.0, 1000)
    for fn, col, ls, lab in ((w_l2, PRIOR, "-", r"plain least squares ($w=1$)"),
                             (w_huber, C["orange"], "-", r"Huber ($\delta=1$)"),
                             (w_cauchy, C["blue"], "-", r"Cauchy ($k=1$)"),
                             (w_gm, C["purple"], "--", r"Geman-McClure ($1/(1+e^2)^2$, unit scale)"),
                             (w_dcs, C["red"], "-", r"DCS, $s^2$ ($\Phi=12.59$)")):
        ax_w.semilogy(e, fn(e), color=col, ls=ls, lw=2, label=lab)
    ax_w.axvline(np.sqrt(Phi), color=PRIOR, ls=":", lw=1)
    ax_w.text(np.sqrt(Phi) + 0.1, 1.5e-4, r"$e=\sqrt{\Phi}=3.55$", fontsize=9, color="#555555")
    ax_w.set(title="Weight on the edge's information matrix", xlabel="residual magnitude e", ylabel="w(e)", ylim=(1e-4, 2.5))
    fig.legend(*ax_w.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9)

    e2 = np.linspace(0.0, 100.0, 1000)
    ax_s.plot(e2, s_dcs(e2), color=C["red"], lw=2.2, label=r"$s=\min(1,\ 2\Phi/(\Phi+e^2))$")
    ax_s.plot(e2, s_dcs(e2) ** 2, color=C["red"], lw=1.4, ls="--", label=r"$s^2$ (weight actually applied to $\Omega$)")
    ax_s.axvline(Phi, color=TRUTH, ls=":", lw=1.2)
    ax_s.scatter([Phi], [1.0], color=TRUTH, zorder=4)
    ax_s.annotate(r"$\Phi=\chi^2_{0.95}(6)=12.59$: $s$ starts to drop", (Phi, 1.0), (22, 0.88), fontsize=9.5,
                  arrowprops=dict(arrowstyle="-", color=PRIOR))
    ax_s.set(title=r"DCS scale $s$ against squared error", xlabel=r"squared error $e^2=r^\top\Omega r$", ylabel="s", ylim=(0, 1.08))
    ax_s.legend(loc="center right")
    finish(fig, "robust_kernel_weights", "Robust kernel weights (formulas from §16 of pose_graph_optimization.md)",
           "Huber δ = 1, Cauchy k = 1 (the §16.1 example values); Geman-McClure at unit scale (§16.4); DCS Φ = 12.59 (§16.5)",
           bottom=0.12)


FIGURES = {
    "pose_tracking": fig_pose_tracking,
    "imu_propagation_position_fix": fig_imu_propagation_position_fix,
    "imu_preintegration": fig_imu_preintegration,
    "pose_graph": fig_pose_graph,
    "bundle_adjustment": fig_bundle_adjustment,
    "bundle_adjustment_advanced": fig_bundle_adjustment_advanced,
    "saltation_matrix_ekf": fig_saltation_matrix_ekf,
    "inchworm_zupt_ekf": fig_inchworm_zupt_ekf,
    "inchworm_zupt_zaru_ekf": fig_inchworm_zupt_zaru_ekf,
    "inchworm_slip_gate_ekf": fig_inchworm_slip_gate_ekf,
    "inchworm_two_pad_ekf": fig_inchworm_two_pad_ekf,
    "inchworm_two_pad_fusion_ekf": fig_inchworm_two_pad_fusion_ekf,
    "gait_phase_ekf": fig_gait_phase_ekf,
    "friction_anisotropic_ekf": fig_friction_anisotropic_ekf,
    "imu_integration_comparison": fig_imu_integration_comparison,
    "pose_graph_incremental": fig_pose_graph_incremental,
    "bayes_tree_construction": fig_bayes_tree_construction,
    "pnp_estimation": fig_pnp_estimation,
    "sliding_window_marginalization": fig_sliding_window_marginalization,
    "friction_anisotropic_ekf_concept": fig_friction_anisotropic_ekf_concept,
    "inchworm_zupt_ekf_concept": fig_inchworm_zupt_ekf_concept,
    "saltation_matrix_ekf_concept": fig_saltation_matrix_ekf_concept,
    "imu_preintegration_concept": fig_imu_preintegration_concept,
    "imu_propagation_position_fix_concept": fig_imu_propagation_position_fix_concept,
    "pose_tracking_concept": fig_pose_tracking_concept,
    "ekf_linearization_concept": fig_ekf_linearization_concept,
    "pose_tracking_dt_sweep": fig_pose_tracking_dt_sweep,
    "bundle_adjustment_concept": fig_bundle_adjustment_concept,
    "bundle_adjustment_advanced_concept": fig_bundle_adjustment_advanced_concept,
    "pose_graph_incremental_concept": fig_pose_graph_incremental_concept,
    "sliding_window_marginalization_concept": fig_sliding_window_marginalization_concept,
    "pnp_estimation_concept": fig_pnp_estimation_concept,
    "hexapod_gait_ekf": fig_hexapod_gait_ekf,
    "hexapod_gait_ekf_concept": fig_hexapod_gait_ekf_concept,
    "hexapod_gait_ekf_extensions": fig_hexapod_gait_ekf_extensions,
    "hexapod_contact_inekf": fig_hexapod_contact_inekf,
    "lidar_primitives_concept": fig_lidar_primitives_concept,
    "lidar_primitives_extraction": fig_lidar_primitives_extraction,
    "lidar_primitives": fig_lidar_primitives,
    "gauss_newton_tangent": fig_gauss_newton_tangent,
    "lm_gradient_descent": fig_lm_gradient_descent,
    "sparse_fill_in_orderings": fig_sparse_fill_in_orderings,
    "robust_kernel_weights": fig_robust_kernel_weights,
}


def main():
    style()
    names = sys.argv[1:] or list(FIGURES)
    for name in names:
        print(f"{name}:")
        FIGURES[name]()


if __name__ == "__main__":
    main()
