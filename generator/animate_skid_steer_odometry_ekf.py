'''
Animated version of skid_steer_odometry_ekf.py, one run: a Clearpath Husky
drives its path (a straight, a full left loop, a straight, a full right loop)
at 0.5 m/s. Its wheels skid in turns, so it turns slower than its wheel
speeds say, by the ICR coefficient chi. Four estimators see the same wheel
encoders (and, for the EKFs, gyro): odo_chi1 (wheels, chi = 1), odo_driver
(wheels, the driver's chi = 1.875), ekf_fixed (EKF, chi fixed at 1.875) and
ekf_chi (EKF with chi as a state).

Three panels, drawn progressively:
  1. left: the whole scene from above. The true Husky is drawn to scale at the
     true position, with each row's estimated Husky (outline, to scale) and
     path. The Husky is about 1 m in a ~20 m scene, so a second panel follows
     it;
  2. top right: a close-up that follows the true Husky: body, four wheels and
     two arrows for the measured left and right wheel speeds (arrow length
     proportional to the speed; 1 m/s = 2 m). In a turn the outer side is
     faster;
  3. middle and bottom right: ekf_chi's chi estimate with its +-2 sigma band
     against the true chi and the fixed 1.875, then the position error of the
     four rows (log scale, meters) with ekf_fixed's own reported 2 sigma.

Message (doc docs/filtering/skid_steer_odometry_ekf.md): a wrong chi ruins
wheel odometry, and an EKF on wheels + gyro with chi fixed follows the wrong
wheels while reporting millimeters. With chi as a state, the filter learns
chi within about 2 s of the first turn.

True chi = 1.25 by default (--chi), a chosen value, not experiment 1's 1.875:
at 1.875 the driver's chi is right and odo_driver / ekf_fixed would be perfect,
so the four rows would not differ. 1.25 is one of the doc's sweep values
(experiment 2), chosen so the driver's 1.875 is wrong and the learning from
ekf_chi's 1.5 prior is visible (the estimate travels from 1.5 to 1.25).
Seed 0 by default (the simulation's own default seed); one run, not the doc's Monte Carlo averages.

Robot dimensions come from the doc's sources: body 0.99 x 0.67 m, wheelbase
0.512 m (Husky A200 user manual, ref. 3), track 0.555 m and wheel radius
0.1651 m (husky_description URDF, ref. 4). The wheel width (0.11 m) is not
sourced and is illustrative. The estimated robots are drawn with the same
dimensions at the estimated pose. Everything else is the simulation's own
simulate_trial / dead_reckoning / run_ekf output.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch, Polygon

from state_estimation_simulations.use_numpy.skid_steer_odometry_ekf import (
    DRIVER_CHI, DT, ROWS, TRACK, Config, dead_reckoning, run_ekf, simulate_trial, yaw_rate_profile)

BODY_L, BODY_W = 0.99, 0.67        # m, user manual (ref. 3)
WHEELBASE = 0.512                  # m, user manual (ref. 3)
WHEEL_D = 2 * 0.1651               # m, URDF wheel_radius (ref. 4)
WHEEL_W = 0.11                     # m, illustrative (not sourced)
ARROW_SCALE = 2.0                  # m of arrow per m/s of wheel speed
COLORS = dict(odo_chi1="tab:red", odo_driver="tab:orange", ekf_fixed="tab:green", ekf_chi="tab:blue")
LABELS = dict(odo_chi1="wheels, chi = 1", odo_driver="wheels, chi = 1.875 (driver)",
              ekf_fixed="EKF, chi fixed at 1.875", ekf_chi="EKF, chi as a state")


def box(pose, cx, cy, length, width):
    """Corners of a length x width rectangle centered at (cx, cy) in the robot frame, placed at pose (x, y, theta)."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    R = np.array([[c, -s], [s, c]])
    local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.5 * np.array([length, width]) + [cx, cy]
    return local @ R.T + pose[:2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chi", type=float, default=1.25, help="True ICR coefficient (chosen; the doc's experiment 1 uses 1.875)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (see the module docstring)")
    parser.add_argument("--frame-step", type=int, default=25, help="Simulation ticks (20 ms) per GIF frame")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.add_argument("--out", type=str, default="skid_steer_odometry_ekf.gif", help="Output GIF path")
    args = parser.parse_args()

    cfg = replace(Config(), chi=args.chi)
    trial = simulate_trial(cfg, np.random.default_rng(args.seed))
    omega = yaw_rate_profile(cfg)
    xs = {"odo_chi1": dead_reckoning(trial, 1.0), "odo_driver": dead_reckoning(trial, DRIVER_CHI)}
    full_fixed, P_fixed = run_ekf(trial, cfg, False)
    full_chi, P_chi = run_ekf(trial, cfg, True)
    xs["ekf_fixed"], xs["ekf_chi"] = full_fixed[:, :3], full_chi[:, :3]
    n = len(trial.v_l)
    t = np.arange(n + 1) * DT
    pos_err = {r: np.linalg.norm(trial.x[:, :2] - xs[r][:, :2], axis=1) for r in ROWS}
    # ekf_fixed's reported 2 sigma: twice the root of the largest eigenvalue of its position covariance
    fixed_2s = 2 * np.sqrt(np.linalg.eigvalsh(P_fixed[:, :2, :2])[:, -1])
    chi_est, chi_2s = full_chi[:, 5], 2 * np.sqrt(P_chi[:, 5, 5])

    k2 = int(round((cfg.straight + 2.0) / DT))
    print(f"true chi {args.chi} (chosen), seed {args.seed}, {cfg.duration:.0f} s, {cfg.speed * cfg.duration:.0f} m:")
    for r in ROWS:
        print(f"  {r:<10s} position error at 30 s {pos_err[r][int(30 / DT)]:.3f} m, at the end {pos_err[r][-1]:.3f} m")
    print(f"  ekf_chi largest position error {100 * pos_err['ekf_chi'].max():.1f} cm")
    print(f"  ekf_fixed reported 2 sigma: at the end {1000 * fixed_2s[-1]:.2f} mm, largest {1000 * fixed_2s.max():.2f} mm")
    print(f"  ekf_chi chi: prior {chi_est[0]:.3f} +-{chi_2s[0]:.3f}; {cfg.straight:.0f} s (first turn starts) {chi_est[int(cfg.straight / DT)]:.3f} "
          f"+-{chi_2s[int(cfg.straight / DT)]:.3f}; {cfg.straight + 2:.0f} s {chi_est[k2]:.3f} +-{chi_2s[k2]:.3f}; "
          f"end {chi_est[-1]:.3f} +-{chi_2s[-1]:.3f} (true {args.chi})")

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.0], height_ratios=[1.5, 1, 1], hspace=0.45, wspace=0.18)
    ax_m = fig.add_subplot(gs[:, 0])
    ax_z = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 1])
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.17)
    fig.suptitle(f"Skid-steer Husky, true chi = {args.chi}: a wrong chi ruins the wheels, estimating it fixes them",
                 fontsize=12, fontweight="bold")

    # whole-scene panel
    allx = np.concatenate([trial.x[:, 0]] + [xs[r][:, 0] for r in ROWS])
    ally = np.concatenate([trial.x[:, 1]] + [xs[r][:, 1] for r in ROWS])
    ax_m.set_aspect("equal")
    ax_m.set(xlim=(allx.min() - 1, allx.max() + 1), ylim=(ally.min() - 1, ally.max() + 1), xlabel="x (m)", ylabel="y (m)",
             title="Whole scene (Husky drawn to scale)")
    ax_m.plot(trial.x[:, 0], trial.x[:, 1], color="k", lw=1, ls="--", alpha=0.25)
    truth_path, = ax_m.plot([], [], color="k", lw=2, label="true path", zorder=5)
    est_paths = {r: ax_m.plot([], [], color=COLORS[r], lw=1.5, label=LABELS[r], zorder=4)[0] for r in ROWS}
    est_boxes = {r: ax_m.add_patch(Polygon(np.zeros((4, 2)), closed=True, fc="none", ec=COLORS[r], lw=1.3, zorder=6)) for r in ROWS}
    est_dots = {r: ax_m.plot([], [], "o", ms=4, color=COLORS[r], zorder=7)[0] for r in ROWS}
    true_box = ax_m.add_patch(Polygon(np.zeros((4, 2)), closed=True, fc="#c9ced6", ec="k", lw=1.3, zorder=8))
    true_dot, = ax_m.plot([], [], "o", ms=4, color="k", zorder=9)
    ax_m.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=8, ncol=2, frameon=False)
    status = ax_m.text(0.02, 0.98, "", transform=ax_m.transAxes, va="top", fontsize=9, fontweight="bold")

    # follow-the-robot close-up
    ax_z.set_aspect("equal")
    ax_z.set(xticks=[], yticks=[], title="Close-up following the true Husky; arrows: measured wheel speeds")
    ax_z.plot(trial.x[:, 0], trial.x[:, 1], color="k", lw=1, ls="--", alpha=0.25)
    zoom_path, = ax_z.plot([], [], color="k", lw=2)
    zoom_body = ax_z.add_patch(Polygon(np.zeros((4, 2)), closed=True, fc="#c9ced6", ec="k", lw=1.5, zorder=3))
    zoom_wheels = [ax_z.add_patch(Polygon(np.zeros((4, 2)), closed=True, fc="#333", ec="k", zorder=4)) for _ in range(4)]
    zoom_nose, = ax_z.plot([], [], color="k", lw=2, zorder=5)
    arrows = [ax_z.add_patch(FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", mutation_scale=14, lw=2.5, color=c, zorder=6))
              for c in ("tab:purple", "tab:cyan")]
    zoom_text = ax_z.text(0.02, 0.97, "", transform=ax_z.transAxes, va="top", fontsize=8.5, family="monospace")

    pos = ax_z.get_position(original=True)
    zoom_half_x = 1.25 * pos.width * fig.get_figwidth() / (pos.height * fig.get_figheight())   # fills the axes at equal aspect
    side_tags = [ax_z.text(0, 0, tag, color=c, fontsize=9, fontweight="bold", ha="center", va="center", zorder=7)
                 for tag, c in (("L", "tab:purple"), ("R", "tab:cyan"))]

    # chi panel
    ax_c.plot(t, chi_est, color="tab:blue", lw=1, alpha=0.2)
    chi_line, = ax_c.plot([], [], color="tab:blue", lw=1.8, label="ekf_chi estimate")
    band = None
    ax_c.axhline(args.chi, color="k", ls="--", lw=1.2, label=f"true chi = {args.chi}")
    ax_c.axhline(DRIVER_CHI, color="tab:green", ls=":", lw=1.5, label="fixed chi = 1.875 (ekf_fixed)")
    ax_c.axvline(cfg.straight, color="0.6", lw=0.8)
    ax_c.text(cfg.straight + 1, 0.93, "first turn", fontsize=8, color="0.4")
    lo = min((chi_est - chi_2s).min(), args.chi) - 0.05
    hi = max((chi_est + chi_2s).max(), DRIVER_CHI) + 0.05
    ax_c.set(xlim=(0, t[-1]), ylim=(lo, hi), xlabel="time (s)", ylabel="chi", title="ICR coefficient chi: ekf_chi estimate and its ±2σ (shaded)")
    ax_c.legend(loc="lower right", fontsize=8)

    # position-error panel (log scale; start at tick 1, the first error is zero)
    err_lines = {}
    for r in ROWS:
        ax_e.plot(t[1:], pos_err[r][1:], color=COLORS[r], lw=1, alpha=0.2)
        err_lines[r], = ax_e.plot([], [], color=COLORS[r], lw=1.8, ls="--" if r == "ekf_fixed" else "-")
    ax_e.plot(t[1:], fixed_2s[1:], color="tab:green", lw=1, alpha=0.2)
    sigma_line, = ax_e.plot([], [], color="tab:green", lw=1.5, ls=":")
    ax_e.set_yscale("log")
    ax_e.set(xlim=(0, t[-1]), ylim=(1e-4, 3 * max(e.max() for e in pos_err.values())), xlabel="time (s)",
             ylabel="position error (m)", title="Position error (log); dotted: ekf_fixed's own reported 2σ")
    ax_e.grid(True, which="major", alpha=0.3)

    fig.text(0.5, 0.012, f"True chi = {args.chi} (a chosen value, not the doc's default 1.875), seed {args.seed}, one run. "
             "Both wheel-only rows and the EKF with chi fixed at 1.875 follow the wrong wheels, meters off.\n"
             f"ekf_fixed reports a 2σ of at most {1000 * fixed_2s.max():.0f} mm while being meters off; ekf_chi's band collapses "
             f"within about 2 s of the first turn, and its error stays at most {100 * pos_err['ekf_chi'].max():.1f} cm.", ha="center", fontsize=9)

    def update(frame):
        k = min(frame * args.frame_step, n)
        sl = slice(0, k + 1)
        pose = trial.x[k]
        truth_path.set_data(trial.x[sl, 0], trial.x[sl, 1])
        true_box.set_xy(box(pose, 0, 0, BODY_L, BODY_W))
        true_dot.set_data([pose[0]], [pose[1]])
        for r in ROWS:
            est_paths[r].set_data(xs[r][sl, 0], xs[r][sl, 1])
            est_boxes[r].set_xy(box(xs[r][k], 0, 0, BODY_L, BODY_W))
            est_dots[r].set_data([xs[r][k, 0]], [xs[r][k, 1]])
        # close-up
        ax_z.set_xlim(pose[0] - zoom_half_x, pose[0] + zoom_half_x)
        ax_z.set_ylim(pose[1] - 1.25, pose[1] + 1.25)
        zoom_path.set_data(trial.x[max(0, k - 400):k + 1, 0], trial.x[max(0, k - 400):k + 1, 1])
        zoom_body.set_xy(box(pose, 0, 0, BODY_L, BODY_W))
        for w, (cx, cy) in zip(zoom_wheels, [(sx * WHEELBASE / 2, sy * TRACK / 2) for sx in (1, -1) for sy in (1, -1)]):
            w.set_xy(box(pose, cx, cy, WHEEL_D, WHEEL_W))
        c, s = np.cos(pose[2]), np.sin(pose[2])
        tip = pose[:2] + BODY_L / 2 * np.array([c, s])
        zoom_nose.set_data([pose[0], tip[0]], [pose[1], tip[1]])
        kk = min(k, n - 1)
        for arrow, v, side in zip(arrows, (trial.v_l[kk], trial.v_r[kk]), (+1, -1)):
            # side +1: robot's left (body y > 0), side -1: right; arrow starts mid-body, beside the wheels
            start = pose[:2] + side * 0.50 * np.array([-s, c])
            arrow.set_positions(start, start + ARROW_SCALE * v * np.array([c, s]))
        for tag, side in zip(side_tags, (+1, -1)):
            tag.set_position(pose[:2] + side * 0.50 * np.array([-s, c]) - 0.18 * np.array([c, s]))
        zoom_text.set_text(f"left wheel  {trial.v_l[kk]:.3f} m/s\nright wheel {trial.v_r[kk]:.3f} m/s\n"
                           f"true yaw rate {omega[kk]:+.3f} rad/s")
        # metrics
        nonlocal band
        chi_line.set_data(t[sl], chi_est[sl])
        if band is not None:
            band.remove()
        band = ax_c.fill_between(t[sl], (chi_est - chi_2s)[sl], (chi_est + chi_2s)[sl], color="tab:blue", alpha=0.2, lw=0)
        for r in ROWS:
            err_lines[r].set_data(t[1:k + 1], pos_err[r][1:k + 1])
        sigma_line.set_data(t[1:k + 1], fixed_2s[1:k + 1])
        status.set_text(f"t = {t[k]:.1f} s")
        return []

    n_frames = n // args.frame_step + 1
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {n_frames} frames to {args.out}")


if __name__ == "__main__":
    main()
