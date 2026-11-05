'''
Animated version of hexapod_joint_kinematics_inekf.py (experiment 2), one run:
the HEBI Daisy walks its turning tripod-gait path while two right-invariant
EKFs track it from an IMU plus its joint encoders through the leg kinematics.
Every joint has a fixed 2 deg zero offset (sigma_b = 2 deg) on top of 1 mrad
white noise. The filters see identical data and start from the same estimate;
one ignores the offsets ("joint"), the other estimates the 18 offsets as
states ("joint_offsets").

Three panels, drawn progressively:
  1. top-down view: Daisy's body (the hexagon through its six hip mounts) and
     its legs, each drawn joint by joint (hip, three joint origins, foot) from
     the script's leg_frames at the true joint angles, with the true path.
     Black dots: true stance feet. Orange segments: the foot-reading error
     R_true y~ + p - d (the simulated encoder reading, offsets included, minus
     the true foot), magnified 10 times;
  2. velocity error along the world x axis of each filter with its own +-2
     sigma band (shaded; the two bands are nearly the same);
  3. each filter's velocity NEES (squared velocity error over its reported
     covariance, 3 DoF) averaged over the run so far, from 1 s on; a
     consistent filter averages 3.

Message (doc docs/filtering/hexapod_joint_kinematics_inekf.md section 4): the
filter that ignores the offsets is overconfident, its velocity error is
larger than its band says; estimating the offsets keeps it consistent and
more accurate.

Seed 6 by default (not main()'s seed 0): the Monte Carlo result in the doc
(30 trials) is velocity NEES 6.93 (joint) vs 3.09 (joint_offsets) at 2 deg;
seed 6 gives about the same (see the printout), whereas seed 0 gives a milder
4.5 vs 2.8. Over seeds 0-7 the ignoring filter's run-average velocity NEES
ranges from 4.2 to 10.7 and the offset-state filter's from 2.7 to 3.3. The
joint angles of the stance legs are the simulation's own (inverse kinematics of
the true foot position, as in simulate_trial). The simulation has no swing feet: swing
legs are drawn faded at the nominal stance joint angles (illustrative only).
Everything else is the simulation's own simulate_trial / make_filter / step
output.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon

from state_estimation_simulations.utils import nees
from state_estimation_simulations.use_numpy.hexapod_contact_inekf import Config
from state_estimation_simulations.use_numpy.hexapod_joint_kinematics_inekf import (
    DAISY, HIP_ANGLE, HIP_OFFSET, daisy_config, iso_variances, leg_frames, make_filter, simulate_trial, stance_ik)

KINDS = ("joint", "joint_offsets")
COLORS = {"joint": "tab:blue", "joint_offsets": "tab:red"}
LABELS = {"joint": "ignores the offsets", "joint_offsets": "estimates the 18 offsets"}
MAGNIFY = 10.0   # foot-reading error magnification in the top-down view
SKIP_S = 1.0     # the doc's scores leave out the first second


def run(kind, trial, q_meas, cfg, sigma_q, sigma_b, iso_var, gait):
    """Drives one filter like run_filter, keeping the velocity error (invariant
    error, the one the covariance describes) with its reported std, the
    velocity NEES and the plain error v_hat - v per tick."""
    stance, y = trial.feet[gait]
    ekf = make_filter(kind, trial, cfg, sigma_q, sigma_b, iso_var)
    n_pts = len(stance)
    out = dict(vel_err=np.zeros((n_pts, 3)), vel_diff=np.zeros((n_pts, 3)), vel_sigma=np.zeros((n_pts, 3)), nees=np.zeros(n_pts))
    for k in range(n_pts):
        if k > 0:
            ekf.predict(trial.gyro_meas[k - 1], trial.accel_meas[k - 1])
        ekf.q_k = q_meas[gait][k]
        ekf.step(stance[k], y[k])
        err = ekf.base_error(trial.R[k], trial.v[k], trial.p[k])
        out["vel_err"][k] = err[3:6]
        out["vel_diff"][k] = ekf.v - trial.v[k]
        out["vel_sigma"][k] = np.sqrt(np.diag(ekf.P)[3:6])
        out["nees"][k] = nees(err[3:6], ekf.P[3:6, 3:6])
    return out


def true_joint_angles(trial, gait, cfg):
    """(N, 6, 3) true stance joint angles (NaN in swing): knee-up inverse
    kinematics of each true body-frame foot position, as in simulate_trial."""
    stance, _ = trial.feet[gait]
    d = trial.foot_pos[gait]
    q = np.full((len(stance), 6, 3), np.nan)
    for leg in range(6):
        q_prev = stance_ik(leg)
        for k in np.flatnonzero(stance[:, leg]):
            q_prev = DAISY.ik(trial.R[k].T @ (d[k, leg] - trial.p[k]), leg, q_prev, R=trial.R[k])
            q[k, leg] = q_prev
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gait", type=str, default="tripod", choices=["tripod", "ripple", "wave"], help="Gait")
    parser.add_argument("--sigma-q", type=float, default=1e-3, help="Encoder white-noise std per joint (rad)")
    parser.add_argument("--sigma-b-deg", type=float, default=2.0, help="Joint zero-offset std (deg)")
    parser.add_argument("--seed", type=int, default=6, help="RNG seed (see the module docstring)")
    parser.add_argument("--frame-step", type=int, default=4, help="Filter ticks per GIF frame")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.add_argument("--out", type=str, default="hexapod_joint_kinematics_inekf.gif", help="Output GIF path")
    args = parser.parse_args()

    cfg = daisy_config(Config())
    sigma_b = np.radians(args.sigma_b_deg)
    iso_var = iso_variances(args.sigma_q)
    trial, q_meas, b = simulate_trial(cfg, [args.gait], np.random.default_rng(args.seed), args.sigma_q, sigma_b)
    stance, y = trial.feet[args.gait]
    d_true = trial.foot_pos[args.gait]
    est = {kind: run(kind, trial, q_meas, cfg, args.sigma_q, sigma_b, iso_var, args.gait) for kind in KINDS}
    q_true = true_joint_angles(trial, args.gait, cfg)
    q_nominal = [stance_ik(leg) for leg in range(6)]
    n_steps = cfg.n_steps
    t = np.arange(n_steps + 1) * cfg.dt
    skip = int(round(SKIP_S / cfg.dt))
    # run average of the velocity NEES from 1 s on (NaN before)
    for kind in KINDS:
        nv = est[kind]["nees"]
        avg = np.full(len(nv), np.nan)
        avg[skip:] = np.cumsum(nv[skip:]) / np.arange(1, len(nv) - skip + 1)
        est[kind]["nees_avg"] = avg
    print(f"{args.gait}, encoder noise {args.sigma_q * 1e3:g} mrad, joint offsets {args.sigma_b_deg:g} deg, "
          f"seed {args.seed} (averages leave out the first {SKIP_S:g} s):")
    for kind in KINDS:
        e = est[kind]
        rmse = np.sqrt((e["vel_diff"][skip:] ** 2).sum(axis=1).mean()) * 1e3
        print(f"  {kind:<14s} velocity NEES {e['nees'][skip:].mean():5.2f} | velocity RMSE {rmse:5.2f} mm/s | "
              f"x error RMS {np.sqrt((e['vel_err'][skip:, 0] ** 2).mean()) * 1e3:4.2f} mm/s vs mean reported "
              f"1 sigma {e['vel_sigma'][skip:, 0].mean() * 1e3:4.2f} mm/s")
    err_world = np.array([[trial.R[k] @ y[k, leg] + trial.p[k] - d_true[k, leg] if stance[k, leg] else np.full(3, np.nan)
                           for leg in range(6)] for k in range(n_steps + 1)])
    print(f"  foot-reading error of the stance feet (true joint offsets, simulated): "
          f"mean norm {np.nanmean(np.linalg.norm(err_world, axis=2)) * 1e3:.1f} mm, shown x{MAGNIFY:g}")

    hip_xy = np.array([HIP_OFFSET[i] * np.array([np.cos(HIP_ANGLE[i]), np.sin(HIP_ANGLE[i]), 0.0]) for i in range(6)])
    hexagon = hip_xy[[0, 1, 2, 5, 4, 3]]   # left legs front to rear, then right legs rear to front

    fig = plt.figure(figsize=(12, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], hspace=0.45, wspace=0.25)
    ax_m = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_n = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.24)
    fig.suptitle(f"Daisy hexapod with {args.sigma_b_deg:g}° joint zero offsets: ignoring them makes the filter "
                 f"overconfident", fontsize=12, fontweight="bold")

    # robot panel
    pad = 0.8
    ax_m.set_aspect("equal")
    ax_m.set_xlim(trial.p[:, 0].min() - pad, trial.p[:, 0].max() + pad)
    ax_m.set_ylim(trial.p[:, 1].min() - pad, trial.p[:, 1].max() + pad)
    ax_m.set(xlabel="x (m)", ylabel="y (m)")
    ax_m.plot(trial.p[:, 0], trial.p[:, 1], color="green", lw=1, ls="--", alpha=0.3)
    truth_path, = ax_m.plot([], [], color="green", lw=2, label="true path", zorder=6)
    body = Polygon(hexagon[:, :2], closed=True, fc="#e8eef6", ec="#5b6b7f", lw=1.4, alpha=0.7, zorder=2)
    ax_m.add_patch(body)
    nose, = ax_m.plot([], [], color="#5b6b7f", lw=2, zorder=4)
    leg_lines = [ax_m.plot([], [], "-o", color="#5b6b7f", lw=2, ms=3.5, zorder=3)[0] for _ in range(6)]
    foot_marks = [ax_m.plot([], [], "o", ms=7, mec="black", zorder=5)[0] for _ in range(6)]
    err_lines = [ax_m.plot([], [], color="darkorange", lw=2.2, zorder=7)[0] for _ in range(6)]
    ax_m.plot([], [], "-o", color="#5b6b7f", lw=2, ms=3.5, label="leg: hip, 3 joints, foot (true angles)")
    ax_m.plot([], [], "o", ms=7, mfc="black", mec="black", label="true stance foot")
    ax_m.plot([], [], "-o", color="#5b6b7f", alpha=0.35, lw=2, ms=3.5, label="swing leg (illustrative)")
    ax_m.plot([], [], color="darkorange", lw=2.2, label=f"foot-reading error x{MAGNIFY:g}")
    ax_m.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8, ncol=2, frameon=False)
    status = ax_m.text(0.02, 0.98, "", transform=ax_m.transAxes, va="top", fontsize=9, fontweight="bold")

    # velocity error panel
    v_lines, v_bands = {}, {}
    for kind in KINDS:
        ax_v.plot(t, est[kind]["vel_err"][:, 0] * 1e3, color=COLORS[kind], lw=1, alpha=0.2)
        v_lines[kind], = ax_v.plot([], [], color=COLORS[kind], lw=1.8, label=LABELS[kind])
        v_bands[kind] = None
    ax_v.axhline(0.0, color="black", lw=0.8)
    # y range from the scored part (after the first second); the start-up transient is clipped
    lim = max(max(np.abs(est[kind]["vel_err"][skip:, 0]).max(), 2 * est[kind]["vel_sigma"][skip:, 0].max())
              for kind in KINDS) * 1e3 * 1.15
    ax_v.set(xlim=(0, t[-1]), ylim=(-lim, lim), xlabel="time (s)", ylabel="velocity error, x (mm/s)",
             title="Velocity error and the filter's own ±2σ (shaded)")
    ax_v.set_title(ax_v.get_title(), pad=22)
    ax_v.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, frameon=False)

    # NEES panel
    n_lines = {}
    for kind in KINDS:
        ax_n.plot(t, est[kind]["nees_avg"], color=COLORS[kind], lw=1, alpha=0.2)
        n_lines[kind], = ax_n.plot([], [], color=COLORS[kind], lw=1.8)
    ax_n.axhline(3.0, color="black", lw=0.8, ls="--")
    ax_n.text(0.99, 3.0, "consistent: 3", transform=ax_n.get_yaxis_transform(), ha="right", va="top", fontsize=8)
    ax_n.set(xlim=(0, t[-1]), ylim=(0, np.nanmax([est[kind]["nees_avg"][2 * skip:] for kind in KINDS]) * 1.3),
             xlabel="time (s)", ylabel="velocity NEES, run average", title="Velocity NEES averaged over the run so far (colors as above)")
    fig.text(0.5, 0.012, f"A fixed {args.sigma_b_deg:g}° offset on each joint moves every foot reading by about "
             f"{np.nanmean(np.linalg.norm(err_world, axis=2)) * 1e3:.0f} mm (orange), a slowly changing error that the "
             "white-noise model does not describe.\nThe filter that ignores it is overconfident (error larger than its "
             "band says, NEES above 3); estimating the 18 offsets keeps it consistent and more accurate.",
             ha="center", fontsize=9)

    def update(frame):
        k = min(frame * args.frame_step, n_steps)
        sl = slice(0, k + 1)
        R, p = trial.R[k], trial.p[k]
        poly = (hexagon @ R.T + p)[:, :2]
        body.set_xy(poly)
        tip = p + R @ np.array([0.33, 0.0, 0.0])
        nose.set_data([p[0], tip[0]], [p[1], tip[1]])
        for leg in range(6):
            planted = stance[k, leg]
            origins, _, foot = leg_frames(q_true[k, leg] if planted else q_nominal[leg], leg)
            pts = np.vstack([origins, foot]) @ R.T + p
            leg_lines[leg].set_data(pts[:, 0], pts[:, 1])
            leg_lines[leg].set_alpha(1.0 if planted else 0.35)
            foot_marks[leg].set_data([pts[-1, 0]], [pts[-1, 1]])
            foot_marks[leg].set_markerfacecolor("black" if planted else "white")
            foot_marks[leg].set_alpha(1.0 if planted else 0.35)
            if planted:
                tip_e = d_true[k, leg] + MAGNIFY * err_world[k, leg]
                err_lines[leg].set_data([d_true[k, leg, 0], tip_e[0]], [d_true[k, leg, 1], tip_e[1]])
            else:
                err_lines[leg].set_data([], [])
        truth_path.set_data(trial.p[sl, 0], trial.p[sl, 1])
        for kind in KINDS:
            e = est[kind]
            v_lines[kind].set_data(t[sl], e["vel_err"][sl, 0] * 1e3)
            if v_bands[kind] is not None:
                v_bands[kind].remove()
            s = 2e3 * e["vel_sigma"][sl, 0]
            v_bands[kind] = ax_v.fill_between(t[sl], -s, s, color=COLORS[kind], alpha=0.15, lw=0)
            n_lines[kind].set_data(t[sl], e["nees_avg"][sl])
        status.set_text(f"t = {t[k]:.1f} s")
        return []

    n_frames = n_steps // args.frame_step + 1
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {n_frames} frames to {args.out}")


if __name__ == "__main__":
    main()
