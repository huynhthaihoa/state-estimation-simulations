'''
Animated version of invariant_saltation_ekf.py's default run: a tumbling rigid
body carrying an IMU is dropped from 5 m and bounces on the ground, while
right-invariant EKFs on SE_2(3) track it from the IMU and from body-frame
sightings of four known landmarks. The filters differ only in how they carry
the covariance through a bounce (see docs/filtering/invariant_saltation_ekf.md).

Two panels:
  1. 3D scene: the true body (a box, tumbling at its constant body rate), its
     trail and ground shadow, the four landmarks, sight lines whenever a
     landmark measurement arrives, and each filter's position estimate;
  2. the Monte Carlo average NEES of each filter over time, drawn up to the
     current instant, with the consistent value 9 and its 95% interval.

The estimates stay within a few centimeters of the truth for every filter, so
their markers overlap in the scene: the difference is in panel 2, where the
invariant-naive filter becomes overconfident after each bounce. No single
state component's standard deviation is shown, on purpose: the naive filter's
mistake is a wrong rotation-velocity correlation, and its horizontal-velocity
standard deviation is in fact larger than the others' after a bounce.

The setup and random draws are the same as invariant_saltation_ekf.py's main(),
so the numbers match its output and figure. Writes a GIF with matplotlib's
pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from invariant_saltation_ekf import (FILTERS, LABELS, COLORS, LINESTYLES, se23_make, se23_parts, so3_exp,
                                     generate_ground_truth_and_data, initial_estimate, child_detect_rngs,
                                     run_filter, run_monte_carlo_consistency, averaged_nees_bounds,
                                     default_landmarks)


def box_faces(X, half_size):
    """The six faces of a box with the given half-dimensions, posed by X.
    Returns:
        list of six (4,3) arrays of world-frame corners
    """
    R, _, p = se23_parts(X)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = p + (signs * half_size) @ R.T
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    return [corners[f] for f in faces]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=5.0, help="Simulation length (s)")
    parser.add_argument("--dt", type=float, default=0.01, help="IMU step (s)")
    parser.add_argument("--meas-every", type=int, default=10, help="Landmark measurement period (IMU steps)")
    parser.add_argument("--restitution", type=float, default=0.8, help="Bounce restitution coefficient (0-1)")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity magnitude (m/s^2)")
    parser.add_argument("--drop-height", type=float, default=5.0, help="Initial height (m)")
    parser.add_argument("--init-velocity", type=float, nargs=3, default=[1.0, 0.5, 0.0],
                        metavar=("VX", "VY", "VZ"), help="Initial velocity (m/s)")
    parser.add_argument("--body-rate", type=float, nargs=3, default=[1.0, -0.6, 2.0],
                        metavar=("WX", "WY", "WZ"), help="Constant true body rate (rad/s)")
    parser.add_argument("--gyro-noise-std", type=float, default=0.02, help="Gyro noise std-dev per sample (rad/s)")
    parser.add_argument("--accel-noise-std", type=float, default=0.3, help="Accel noise std-dev per sample (m/s^2)")
    parser.add_argument("--meas-noise-std", type=float, default=0.05, help="Landmark measurement noise std-dev (m)")
    parser.add_argument("--init-rot-std-deg", type=float, default=5.0, help="Initial rotation error std-dev (deg)")
    parser.add_argument("--init-vel-std", type=float, default=0.2, help="Initial velocity error std-dev (m/s)")
    parser.add_argument("--init-pos-std", type=float, default=0.1, help="Initial position error std-dev (m)")
    parser.add_argument("--detect-time-bias", type=float, default=0.0,
                        help="Contact-detection time offset (s), positive = late; 0 = exact detection")
    parser.add_argument("--detect-time-noise-std", type=float, default=0.0,
                        help="Contact-detection time jitter std-dev (s); 0 = exact detection")
    parser.add_argument("--n-mc-trials", type=int, default=200, help="Monte Carlo consistency trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--stride", type=int, default=3, help="Filter ticks per GIF frame")
    parser.add_argument("--fps", type=int, default=20, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=72, help="GIF resolution")
    parser.add_argument("--out", type=str, default="invariant_saltation_ekf.gif", help="Output GIF path")
    args = parser.parse_args()

    # Same setup and order of random draws as invariant_saltation_ekf.main().
    rng = np.random.default_rng(args.seed)
    g_vec = np.array([0.0, 0.0, -args.gravity])
    landmarks = default_landmarks()
    body_rate = np.array(args.body_rate)
    X0 = se23_make(so3_exp(np.array([0.2, -0.1, 0.3])), np.array(args.init_velocity),
                   np.array([0.0, 0.0, args.drop_height]))
    P0_ri = np.diag([np.radians(args.init_rot_std_deg) ** 2] * 3 + [args.init_vel_std ** 2] * 3
                    + [args.init_pos_std ** 2] * 3)
    X_true, gyro, accel, meas, impact_ticks = generate_ground_truth_and_data(
        args.duration, args.dt, args.restitution, g_vec, X0, body_rate, landmarks, args.meas_every,
        args.gyro_noise_std, args.accel_noise_std, args.meas_noise_std, rng)
    X_init, P_init = initial_estimate(X0, P0_ri, rng)
    detect = (args.detect_time_bias, args.detect_time_noise_std)
    detect_rngs = child_detect_rngs(rng)
    # With exact detection "saltation_timing" is identical to "saltation", so it is left out.
    shown = [n for n in FILTERS if n != "saltation_timing" or any(detect)]
    est = {}
    for name in shown:
        X_list, _, _ = run_filter(name, X_init, P_init[name], gyro, accel, meas, args.dt, args.restitution, g_vec,
                                  landmarks, args.gyro_noise_std, args.accel_noise_std, args.meas_noise_std,
                                  *detect, detect_rngs[name])
        est[name] = np.array([se23_parts(X)[2] for X in X_list])
    print(f"Running Monte Carlo consistency check ({args.n_mc_trials} trials)...")
    avg_nees, _ = run_monte_carlo_consistency(
        X_true, impact_ticks, body_rate, args.dt, args.restitution, g_vec, landmarks, args.meas_every, P0_ri,
        args.gyro_noise_std, args.accel_noise_std, args.meas_noise_std, args.n_mc_trials, rng, *detect)
    lo, hi = averaged_nees_bounds(9, args.n_mc_trials)
    for name in shown:
        a = avg_nees[name]
        post = np.concatenate([a[k + 1:k + 1 + args.meas_every] for k in impact_ticks])
        print(f"  {LABELS[name]:<36s} before 1st bounce={np.nanmean(a[1:impact_ticks[0] - 1]):6.2f}  "
              f"after bounces={np.nanmean(post):6.2f}  peak={np.nanmax(a[impact_ticks[0]:]):6.2f}")

    pos_true = np.array([se23_parts(X)[2] for X in X_true])
    t = np.arange(len(X_true)) * args.dt
    half_size = np.array([0.4, 0.25, 0.12])

    fig = plt.figure(figsize=(10, 9.5))
    ax3d = fig.add_axes([-0.04, 0.40, 1.08, 0.55], projection="3d")
    ax_nees = fig.add_axes([0.09, 0.08, 0.86, 0.27])

    # --- 3D scene ---
    xlim = (min(landmarks[:, 0].min(), pos_true[:, 0].min()) - 0.5, max(landmarks[:, 0].max(), pos_true[:, 0].max()) + 0.5)
    ylim = (min(landmarks[:, 1].min(), pos_true[:, 1].min()) - 0.5, max(landmarks[:, 1].max(), pos_true[:, 1].max()) + 0.5)
    zlim = (0.0, max(landmarks[:, 2].max(), pos_true[:, 2].max()) + 0.5)
    gx, gy = np.meshgrid(xlim, ylim)
    ax3d.plot_surface(gx, gy, np.zeros_like(gx), color="saddlebrown", alpha=0.15, shade=False)
    ax3d.scatter(*landmarks.T, marker="*", s=160, color="black", depthshade=False, label="Landmarks (known)")
    for i, l in enumerate(landmarks):
        ax3d.plot([l[0], l[0]], [l[1], l[1]], [0.0, l[2]], color="black", linewidth=0.6, alpha=0.4)
    ax3d.set_xlim(*xlim)
    ax3d.set_ylim(*ylim)
    ax3d.set_zlim(*zlim)
    ax3d.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]), zoom=1.15)
    ax3d.view_init(elev=18, azim=-62)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.fill = False  # plain panes keep the GIF palette (and file size) small
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("height (m)")
    body = Poly3DCollection(box_faces(X_true[0], half_size), facecolor="limegreen", edgecolor="darkgreen",
                            alpha=0.45, linewidths=0.8)
    ax3d.add_collection3d(body)
    trail, = ax3d.plot([], [], [], color="green", linewidth=1, alpha=0.5)
    shadow, = ax3d.plot([], [], [], "o", color="gray", markersize=7, alpha=0.6)
    rays = [ax3d.plot([], [], [], "--", color="gray", linewidth=0.8)[0] for _ in landmarks]
    est_marks = {n: ax3d.plot([], [], [], "x", markersize=10, mew=2.5, color=COLORS[n], zorder=10)[0] for n in shown}
    ax3d.plot([], [], color="limegreen", linewidth=6, label="True body")
    ax3d.plot([], [], "--", color="gray", linewidth=0.8, label="Landmark sighting")
    ax3d.plot([], [], "x", color="dimgray", markersize=8, mew=2, label="Position estimates (colors below)")
    ax3d.legend(loc="upper left", bbox_to_anchor=(0.1, 0.92), fontsize=8)
    title = fig.suptitle("", y=0.985)
    bounce_text = ax3d.text2D(0.75, 0.82, "", transform=ax3d.transAxes, ha="center", fontsize=14,
                              color="crimson", fontweight="bold")

    # --- Monte Carlo NEES over time ---
    vals = np.concatenate([avg_nees[n][1:] for n in shown])
    vals = vals[np.isfinite(vals)]
    ax_nees.set_yscale("log")
    ax_nees.set_ylim(0.9 * min(vals.min(), lo), 1.15 * max(vals.max(), hi))
    ax_nees.set_xlim(0.0, t[-1])
    ax_nees.axhspan(lo, hi, color="gray", alpha=0.15, label=f"95% interval ({args.n_mc_trials}-trial average)")
    ax_nees.axhline(9.0, color="black", linewidth=1, label="Consistent value (9)")
    for k in impact_ticks:
        ax_nees.axvline(t[k], color="crimson", linestyle=":", linewidth=1)
    nees_lines = {n: ax_nees.plot([], [], linewidth=2, color=COLORS[n], linestyle=LINESTYLES.get(n, "-"),
                                  label=LABELS[n])[0] for n in shown}
    cursor = ax_nees.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax_nees.set_xlabel("Time (s)")
    ax_nees.set_ylabel("Monte Carlo avg. NEES")
    ax_nees.set_title("Is the reported uncertainty trustworthy? (above the band = overconfident; dotted: bounces)",
                      fontsize=10)
    ax_nees.legend(loc="upper right", fontsize=7.5, ncol=2)
    fig.text(0.5, 0.01, "All estimates (x) stay within a few cm of the truth, so they overlap in the scene; "
             "the filters differ in their covariance, i.e. in the NEES.", ha="center", fontsize=8.5)

    n_trail = 60
    frames = list(range(0, len(X_true), args.stride)) + [len(X_true) - 1] * args.fps  # hold the end ~1 s

    def update(k):
        title.set_text(f"Tumbling body with an IMU, right-invariant EKFs on SE_2(3)   t = {t[k]:.2f} s")
        body.set_verts(box_faces(X_true[k], half_size))
        lo_k = max(0, k - n_trail)
        trail.set_data_3d(pos_true[lo_k:k + 1, 0], pos_true[lo_k:k + 1, 1], pos_true[lo_k:k + 1, 2])
        shadow.set_data_3d([pos_true[k, 0]], [pos_true[k, 1]], [0.0])
        sighting = any(0 <= k - m < 2 * args.stride for m in meas)
        p = pos_true[k]
        for ray, l in zip(rays, landmarks):
            ray.set_data_3d(*(np.array([[p[0], l[0]], [p[1], l[1]], [p[2], l[2]]]) if sighting
                              else ([], [], [])))
        for n in shown:
            est_marks[n].set_data_3d([est[n][k, 0]], [est[n][k, 1]], [est[n][k, 2]])
            nees_lines[n].set_data(t[1:k + 1], avg_nees[n][1:k + 1])
        cursor.set_xdata([t[k], t[k]])
        bounce_text.set_text("BOUNCE" if any(0 <= k - b < 8 for b in impact_ticks) else "")
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
