"""Generates one illustrative figure per simulation, from the simulations' own
output, into assets/<name>.png.

Nothing here re-implements an estimator: every trajectory, error and NEES
curve comes from calling the functions in use_numpy/ exactly the way each
script's main() does (same defaults, same seed, same order of random draws),
so the numbers on a figure match what the script itself prints. The only
exception is a non-default duration where the default is too short to show
anything, which the figure then states in its subtitle.

Usage:
    uv run python assets/make_figures.py            # all figures
    uv run python assets/make_figures.py pose_graph # just one (see FIGURES)
"""
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter, LogLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "use_numpy"))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
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


def finish(fig, name, title, subtitle, bottom=0.0):
    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.925, subtitle, fontsize=10.5, color="#555555", ha="left")
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
    import pointcloud_pose_tracking as m
    from lie_utils import se3_exp

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
        ax_xy.plot(*xy_of(T_est), color=col, ls=ls, lw=1.6, label=f"{name}  (RMS {rms(pos_err) * 100:.1f} cm)")
        ax_err.plot(t, pos_err, color=col, ls=ls, lw=1.8)
        print(f"    {name:<28s} RMS pos = {rms(pos_err):.4f} m")
    ax_xy.set(title="Object trajectory (x-y)", xlabel="x (m)", ylabel="y (m)")
    ax_xy.axis("equal")
    fig.legend(*ax_xy.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=9,
               title="in brackets: position error, RMS over the run", title_fontsize=8.5)
    ax_err.set(title="Position error", xlabel="time (s)", ylabel="position error (m)", yscale="log")
    finish(fig, "pose_tracking", "Point-cloud pose tracking on SE(3)",
           f"Noisy twist prior + {n_points} noisy body-frame points per step (z = T·p + noise); "
           "one run, seed 0, script defaults", bottom=0.14)


def fig_robot_imu_tracking():
    """robot_imu_simulation.py: 100 Hz IMU dead reckoning + 1 Hz position-only Gauss-Newton fix."""
    import robot_imu_simulation as m

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
    finish(fig, "robot_imu_tracking", "IMU dead reckoning + GPS-style position fix",
           f"100 Hz noisy body twist on SE(3), 1 Hz position-only measurement; seed 0, --total-seconds {seconds} "
           "(default 3), other settings default")


def fig_imu_preintegration():
    """imu_preintegration.py: one bundle of 100 samples, then a bias update applied to first order."""
    import imu_preintegration as m

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
           "Script defaults (this demo integrates body-velocity samples, not accelerations); the bias update is "
           f"applied as 1 Jacobian correction instead of re-integrating {hz} samples")


def fig_pose_graph():
    """pose_graph.py: square loop, odometry drift, one loop closure, Levenberg-Marquardt relaxation."""
    import pose_graph as m

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
    ax_e.bar(k - w / 2, e_dr, w, color=PRIOR, label=f"Odometry only (RMS {rms(e_dr) * 100:.1f} cm)")
    ax_e.bar(k + w / 2, e_opt, w, color=C["blue"], label=f"Optimized (RMS {rms(e_opt) * 100:.1f} cm)")
    ax_e.set_xticks(k, ["X0\n(anchored)"] + [f"X{n}" for n in k[1:]])
    ax_e.set(title="Position error per pose", ylabel="position error (m)")
    ax_e.legend(loc="upper left")
    finish(fig, "pose_graph", "Pose-graph relaxation with one loop closure",
           "Noisy SE(3) odometry edges + one loop-closure edge, residual Log(Z_ij⁻¹ X_i⁻¹ X_j), "
           "Levenberg-Marquardt; seed 0, script defaults")


def fig_bundle_adjustment():
    """bundle_adjustment.py: landmarks-only vs poses-only vs joint BA."""
    import bundle_adjustment as m

    rng = np.random.default_rng(SEED)
    K = (800.0, 800.0, 320.0, 240.0)
    T_true, P_all = m.generate_ground_truth_scene(8, 60, 5.0, 180.0, 2.0, rng)
    P_true, obs = m.build_observations(T_true, P_all, K, 1.0, 70.0, 2, rng)
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
    ax_b.set(yscale="log", ylim=(1e-3, 1e5), title="RMS error by method (after similarity alignment)")
    ax_b.legend(loc="upper center", ncol=2)
    finish(fig, "bundle_adjustment", "Bundle adjustment: joint camera-pose and landmark refinement",
           "Synthetic scene, cameras on an arc looking inward, 70° field of view, 1 px pixel noise; seed 0, script defaults")


def fig_bundle_adjustment_advanced():
    """bundle_adjustment_advanced.py: incremental Local BA, with and without periodic Global BA."""
    import bundle_adjustment_advanced as m

    K = (800.0, 800.0, 320.0, 240.0)
    n_kf = 50
    T_true = m.generate_ground_truth_trajectory(n_kf, 15.0, 90.0)
    P_all = m.generate_landmark_corridor(T_true, 8, 2.0, 1.0, np.random.default_rng(SEED))
    P_true, obs = m.build_observations(T_true, P_all, K, 1.0, 70.0, 6.0, 3, np.random.default_rng(SEED))
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
    ax_d.set(title="Trajectory error (last 10 keyframes)", xlabel="keyframe", ylabel="RMS position error (m)")
    ax_d.legend(loc="upper left")
    finish(fig, "bundle_adjustment_advanced", "Local vs. Global bundle adjustment on a growing map",
           "Keyframes arrive one at a time; each triggers a Local BA over its covisible neighbors, and every 8th "
           "also a Global BA (dotted lines); seed 0, script defaults (open 90° arc, no loop closure)")


def fig_saltation_matrix_ekf():
    """saltation_matrix_ekf.py: naive (DR) vs saltation-corrected (Xi) covariance update at each bounce."""
    import saltation_matrix_ekf as m

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
    import inchworm_zupt_ekf as m

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


def fig_friction_anisotropic_ekf():
    """friction_anisotropic_ekf.py: isotropic / fixed-anisotropic / heading-aware process noise."""
    import friction_anisotropic_ekf as m

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
    import imu_integration_comparison as m

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
    import pose_graph_incremental as m
    from utils import measure_performance

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
    ax_xy.plot(*xy_of(T_dr), "--", color=PRIOR, lw=1.6, label=f"odometry only (RMS {rms(errs['odo']) * 100:.1f} cm)")
    ax_xy.plot(*xy_of(T_batch), "o-", color=C["blue"], ms=3, lw=1.4,
               label=f"batch re-solve per node (RMS {rms(errs['batch']) * 100:.1f} cm)")
    ax_xy.plot(*xy_of(T_incr), "x:", color=C["orange"], ms=5, lw=1.4,
               label=f"incremental sqrt-SAM (RMS {rms(errs['incr']) * 100:.1f} cm)")
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
    import bayes_tree_construction as m
    from utils import symbolic_eliminate, bayes_tree_affected_path

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
    import pnp_estimation as m

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
                    width=0.005, label=f"reprojection residual ×{mag}, {name} pose (RMS {reproj:.2f} px)")
    outside = int(np.sum((z[:, 0] < 0) | (z[:, 0] > 640) | (z[:, 1] < 0) | (z[:, 1] > 480)))
    print(f"    {outside} of {len(z)} observed pixels fall outside the 640x480 frame")
    ax_i.set(title=f"Image plane: {len(P_list)} matches, {outside} outside the 640×480 frame (gray)", xlabel="u (px)",
             ylabel="v (px)", xlim=(min(-30, z[:, 0].min() - 60), max(670, z[:, 0].max() + 60)),
             ylim=(max(510, z[:, 1].max() + 60), min(-30, z[:, 1].min() - 60)))
    ax_i.set_aspect("equal")
    ax_i.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8.5)

    x = np.arange(3)
    w = 0.36
    for k, ((name, rot, pos, reproj, _), col) in enumerate(zip(rows, [C["orange"], C["blue"]])):
        ax_b.bar(x + (k - 0.5) * w, [rot, pos * 100, reproj], w, color=col, label=name)
    ax_b.set_xticks(x, ["rotation error\n(deg)", "position error\n(cm)", "reprojection RMS\n(px)"])
    ax_b.set(title="Closed-form guess vs. refined pose")
    ax_b.legend(loc="upper left")
    finish(fig, "pnp_estimation", "Perspective-n-Point: camera pose from known 3D–2D matches",
           "Linear DLT (closed form, SVD) as the initial guess, then Gauss-Newton on the reprojection error with "
           "T ← T·Exp(δ); seed 0, script defaults", bottom=0.02)


def fig_sliding_window_marginalization():
    """sliding_window_marginalization.py: bounded sliding window vs full batch re-solve, over trajectory length."""
    import sliding_window_marginalization as m
    from utils import measure_performance

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
             ylabel="RMS position error (cm)")
    ax_e.legend(loc="upper left", fontsize=8.5)
    finish(fig, "sliding_window_marginalization", "Sliding-window marginalization: bounded cost on a growing trajectory",
           f"Odometry-only square path swept from {n[0]} to {n[-1]} poses; the window keeps {window} poses and folds "
           "older ones into a prior; seed 0, script defaults")


FIGURES = {
    "pose_tracking": fig_pose_tracking,
    "robot_imu_tracking": fig_robot_imu_tracking,
    "imu_preintegration": fig_imu_preintegration,
    "pose_graph": fig_pose_graph,
    "bundle_adjustment": fig_bundle_adjustment,
    "bundle_adjustment_advanced": fig_bundle_adjustment_advanced,
    "saltation_matrix_ekf": fig_saltation_matrix_ekf,
    "inchworm_zupt_ekf": fig_inchworm_zupt_ekf,
    "friction_anisotropic_ekf": fig_friction_anisotropic_ekf,
    "imu_integration_comparison": fig_imu_integration_comparison,
    "pose_graph_incremental": fig_pose_graph_incremental,
    "bayes_tree_construction": fig_bayes_tree_construction,
    "pnp_estimation": fig_pnp_estimation,
    "sliding_window_marginalization": fig_sliding_window_marginalization,
}


def main():
    style()
    names = sys.argv[1:] or list(FIGURES)
    for name in names:
        print(f"{name}:")
        FIGURES[name]()


if __name__ == "__main__":
    main()
