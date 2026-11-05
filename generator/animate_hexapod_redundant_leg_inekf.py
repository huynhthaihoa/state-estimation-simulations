'''
Animated version of hexapod_redundant_leg_inekf.py, one run: the CTU HAntR
hexapod (four joints per leg: coxa, trochanter, femur, tibia) walks the contact
doc's turning path (tripod gait) while the contact-aided invariant EKF with 24
joint-offset states (`joint_offsets`) tracks it from an IMU and the leg
encoders. The encoders carry constant zero offsets with a 2 deg std (the
filter's prior) plus the AX-12A quantization noise.

Three panels, drawn progressively:
  1. the robot in 3D (heading frame: yaw removed, tilt and bob kept), every leg
     drawn from the script's DH forward kinematics (HANTR.frames) at the true
     joint angles of the stance legs, with the ground and the true feet; one leg
     (R1, the front right) is highlighted with its four joints marked, and its
     null-direction motion is sketched around the true pose;
  2. the offset error along each leg's null direction n (J n = 0, at the default
     foothold, as in run_offsets) against the std the filter reports for it;
  3. the femur and tibia offset errors against the std the filter reports for
     them. Panels 2 and 3 show the root mean square over the six legs.

Message (doc docs/filtering/hexapod_redundant_leg_inekf.md section 4.2): a
fourth joint gives every leg a direction that does not move the foot, yet the
offset along it is mostly learned while walking (the reported std falls within
seconds); the femur and tibia offsets are learned poorly and the filter is
overconfident about them.

Seed 103 by default: it is the first trial of the script's own 2 deg offset
setting (main() runs it with rng seed 0 + 100 + 3), so the run is not picked
from several. A single run only has six legs per curve, so its numbers differ
from the 30-trial results in the doc; the printout says by how much.

Illustrative, not simulated: (a) a swing leg is drawn faded at the default
foothold (the script does not simulate swing feet); (b) the dotted ghost legs of
the highlighted leg are the true pose plus +-0.4 rad along the current null
direction (an exaggerated sketch: the foot moves only to second order, and the
printout gives how far). Everything else is the simulation's own
simulate_trial / make_filter / step output, and the true joint angles are the
simulate_trial inverse kinematics repeated tick by tick.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401  (registers the 3d projection)

from state_estimation_simulations.use_numpy.hexapod_contact_inekf import Config
from state_estimation_simulations.use_numpy.hexapod_gait_ekf import LEG_NAMES
from state_estimation_simulations.use_numpy.hexapod_joint_kinematics_inekf import run_filter, simulate_trial
from state_estimation_simulations.use_numpy.hexapod_redundant_leg_inekf import HANTR, MOUNT_DEG, SIGMA_Q_QUANT, iso_variances

JOINTS = ["coxa", "trochanter", "femur", "tibia"]
HIGHLIGHT = 3                  # R1: mounted at 0 deg, the front right leg
GHOST_ANGLE = 0.4              # rad, illustrative null-direction sketch


def true_joint_angles(trial, gait, model):
    """(N, 6, 4) true joint angles, NaN in swing: simulate_trial's own inverse
    kinematics, repeated tick by tick (it does not return them)."""
    stance, _ = trial.feet[gait]
    d = trial.foot_pos[gait]
    q = np.full((len(stance), 6, model.n_joints), np.nan)
    for leg in range(6):
        q_prev = model.stance_ik(leg)
        for k in np.flatnonzero(stance[:, leg]):
            q_prev = model.ik(trial.R[k].T @ (d[k, leg] - trial.p[k]), leg, q_prev, R=trial.R[k])
            q[k, leg] = q_prev
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gait", type=str, default="tripod", choices=["tripod", "ripple", "wave"], help="Gait")
    parser.add_argument("--sigma-b-deg", type=float, default=2.0, help="Joint zero-offset std (deg)")
    parser.add_argument("--sigma-q", type=float, default=SIGMA_Q_QUANT, help="Encoder white-noise std (rad)")
    parser.add_argument("--seed", type=int, default=103, help="RNG seed (see the module docstring)")
    parser.add_argument("--frame-step", type=int, default=4, help="Filter ticks per GIF frame")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.add_argument("--out", type=str, default="hexapod_redundant_leg_inekf.gif", help="Output GIF path")
    args = parser.parse_args()

    model, sb = HANTR, np.radians(args.sigma_b_deg)
    cfg = model.config(Config())
    n_steps = cfg.n_steps
    t = np.arange(n_steps + 1) * cfg.dt
    trial, q_meas, b = simulate_trial(cfg, [args.gait], np.random.default_rng(args.seed), args.sigma_q, sb, model)
    stance, _ = trial.feet[args.gait]
    nulls = np.array([model.null_direction(model.stance_ik(leg), leg) for leg in range(6)])
    out = run_filter("joint_offsets", trial, q_meas, args.gait, cfg, args.sigma_q, sb,
                     iso_variances(args.sigma_q, model), b, model, offset_dirs=nulls)
    q_true = true_joint_angles(trial, args.gait, model)
    q_default = [model.stance_ik(leg) for leg in range(6)]

    # per-tick metrics, root mean square over the six legs, in degrees
    e = out["b_err"].reshape(-1, 6, 4)
    s = out["b_std"].reshape(-1, 6, 4)
    deg = np.degrees
    null_act = deg(np.sqrt(np.mean(np.einsum("kli,li->kl", e, nulls) ** 2, axis=1)))
    null_rep = deg(np.sqrt(np.mean(out["dir_std"] ** 2, axis=1)))
    joint_act = deg(np.sqrt(np.mean(e ** 2, axis=1)))       # (N, 4)
    joint_rep = deg(np.sqrt(np.mean(s ** 2, axis=1)))
    iF, iT = JOINTS.index("femur"), JOINTS.index("tibia")

    # how far the illustrative null-motion sketch moves the highlighted foot
    ghost_move = 0.0
    for k in np.flatnonzero(stance[:, HIGHLIGHT]):
        n = model.null_direction(q_true[k, HIGHLIGHT], HIGHLIGHT)
        foot = model.fk(q_true[k, HIGHLIGHT], HIGHLIGHT)
        for sgn in (-1, 1):
            ghost_move = max(ghost_move, np.linalg.norm(model.fk(q_true[k, HIGHLIGHT] + sgn * GHOST_ANGLE * n, HIGHLIGHT) - foot))

    print(f"HAntR, {args.gait} gait, offsets {args.sigma_b_deg:g} deg std, encoder noise {args.sigma_q * 1e3:.2f} mrad, seed {args.seed}:")
    print(f"  null direction, RMS over 6 legs: start {null_act[0]:.2f} (reported {null_rep[0]:.2f}) -> end {null_act[-1]:.2f} "
          f"(reported {null_rep[-1]:.2f}) deg; reported falls below 1 deg at t = {t[np.argmax(null_rep < 1.0)]:.1f} s")
    for j, name in enumerate(JOINTS):
        print(f"  {name:<10s} end: actual {joint_act[-1, j]:.2f} deg vs reported {joint_rep[-1, j]:.2f} deg")
    print(f"  illustrative null sketch: +-{GHOST_ANGLE} rad moves the {LEG_NAMES[HIGHLIGHT]} foot at most {ghost_move * 1e3:.1f} mm")

    # --- geometry in the heading frame (yaw removed, tilt and bob kept)
    cy, sy = np.cos(-trial.yaw), np.sin(-trial.yaw)
    Rh = np.zeros((n_steps + 1, 3, 3))
    for k in range(n_steps + 1):
        Rh[k] = np.array([[cy[k], -sy[k], 0.0], [sy[k], cy[k], 0.0], [0.0, 0.0, 1.0]]) @ trial.R[k]
    order = np.argsort(MOUNT_DEG)                       # legs around the body, for the hexagon

    def leg_pts(k, leg, q):
        origins, _, foot = model.frames(q, leg)
        return Rh[k] @ np.vstack([origins, foot]).T     # (3, 5): coxa, trochanter, femur, tibia, foot

    fig = plt.figure(figsize=(12, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], hspace=0.45, wspace=0.22)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    ax_n = fig.add_subplot(gs[0, 1])
    ax_j = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(left=0.03, right=0.98, top=0.9, bottom=0.22)
    fig.suptitle(f"HAntR, four joints per leg: the null-direction offset is learned, the femur and tibia offsets are not",
                 fontsize=12, fontweight="bold")

    ax3.set_xlim(-0.3, 0.3)
    ax3.set_ylim(-0.3, 0.3)
    ax3.set_zlim(-0.13, 0.08)
    ax3.set_box_aspect((0.6, 0.6, 0.21), zoom=1.05)
    ax3.view_init(elev=22, azim=-58)
    ax3.set_axis_off()
    ground = Poly3DCollection([np.zeros((4, 3))], facecolor="#dfe7df", edgecolor="#9aa89a", alpha=0.5)
    ax3.add_collection3d(ground)
    gx = 0.27
    ground_xy = np.array([[-gx, -gx], [gx, -gx], [gx, gx], [-gx, gx]])
    body_line, = ax3.plot([], [], [], color="#5b6b7f", lw=2.5)
    leg_lines = [ax3.plot([], [], [], color="#5b6b7f", lw=2)[0] for _ in range(6)]
    foot_marks = [ax3.plot([], [], [], "o", ms=6, mec="black")[0] for _ in range(6)]
    hl_line, = ax3.plot([], [], [], color="tab:red", lw=4, solid_capstyle="round")
    hl_joints, = ax3.plot([], [], [], "o", ms=7, mfc="white", mec="tab:red", mew=2)
    hl_ghosts = [ax3.plot([], [], [], color="tab:red", lw=1.5, ls=":")[0] for _ in range(2)]
    ax3.plot([], [], [], color="#5b6b7f", lw=2, label="true leg (DH kinematics, true joint angles)")
    ax3.plot([], [], [], "o", ms=6, mfc="black", mec="black", label="true stance foot")
    ax3.plot([], [], [], "o", ms=6, mfc="white", mec="black", alpha=0.4, label="swing leg (illustrative)")
    ax3.plot([], [], [], color="tab:red", lw=4, label=f"{LEG_NAMES[HIGHLIGHT]} (front right), its 4 joints marked")
    ax3.plot([], [], [], color="tab:red", lw=1.5, ls=":", label=f"{LEG_NAMES[HIGHLIGHT]} turned ±{GHOST_ANGLE} rad along its null direction (illustrative)")
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8, ncol=1, frameon=False)
    status = fig.text(0.04, 0.88, "", fontsize=10, fontweight="bold")

    # null-direction panel
    ax_n.plot(t, null_act, color="tab:red", lw=1, alpha=0.2)
    ax_n.plot(t, null_rep, color="0.4", lw=1, ls="--", alpha=0.3)
    n_act, = ax_n.plot([], [], color="tab:red", lw=1.8, label="actual offset error (RMS over the 6 legs)")
    n_rep, = ax_n.plot([], [], color="0.3", lw=1.8, ls="--", label="std the filter reports")
    ax_n.axhline(args.sigma_b_deg, color="black", ls=":", lw=1, label=f"prior std ({args.sigma_b_deg:g}°)")
    ax_n.set(xlim=(0, t[-1]), ylim=(0, args.sigma_b_deg * 1.2), xlabel="time (s)", ylabel="offset error (deg)",
             title="Offset along each leg's null direction")
    ax_n.legend(loc="upper right", fontsize=8)

    # femur / tibia panel
    ax_j.axhline(args.sigma_b_deg, color="black", ls=":", lw=1)
    jcol = {"femur": "tab:green", "tibia": "tab:purple"}
    jl_act, jl_rep = {}, {}
    for name, i in (("femur", iF), ("tibia", iT)):
        ax_j.plot(t, joint_act[:, i], color=jcol[name], lw=1, alpha=0.2)
        ax_j.plot(t, joint_rep[:, i], color=jcol[name], lw=1, ls="--", alpha=0.3)
        jl_act[name], = ax_j.plot([], [], color=jcol[name], lw=1.8, label=f"{name}: actual offset error")
        jl_rep[name], = ax_j.plot([], [], color=jcol[name], lw=1.8, ls="--", label=f"{name}: std the filter reports")
    ymax = max(joint_act[:, [iF, iT]].max(), joint_rep[:, [iF, iT]].max(), args.sigma_b_deg) * 1.7
    ax_j.set(xlim=(0, t[-1]), ylim=(0, ymax), xlabel="time (s)", ylabel="offset error (deg)",
             title="Femur and tibia offsets (RMS over the 6 legs)")
    ax_j.legend(loc="upper right", fontsize=8)
    fig.text(0.5, 0.012, "A fourth joint leaves every leg a direction n that does not move the foot, yet the offset along n is mostly learned: "
             "walking probably keeps turning n (not tested).\nThe femur and tibia offsets end well above what the filter reports: it is overconfident about them. "
             "One run (the first of the 30-trial 2° case), see the doc for the Monte Carlo numbers.",
             ha="center", fontsize=9)

    def update(frame):
        k = min(frame * args.frame_step, n_steps)
        sl = slice(0, k + 1)
        z0 = -trial.p[k, 2]
        ground.set_verts([[(x, y, z0) for x, y in ground_xy]])
        hexa = np.array([leg_pts(k, leg, q_true[k, leg] if stance[k, leg] else q_default[leg])[:, 0] for leg in order])
        hexa = np.vstack([hexa, hexa[:1]])
        body_line.set_data_3d(hexa[:, 0], hexa[:, 1], hexa[:, 2])
        for leg in range(6):
            live = stance[k, leg]
            pts = leg_pts(k, leg, q_true[k, leg] if live else q_default[leg])
            leg_lines[leg].set_data_3d(*pts)
            leg_lines[leg].set_alpha(1.0 if live else 0.35)
            foot_marks[leg].set_data_3d([pts[0, 4]], [pts[1, 4]], [pts[2, 4]])
            foot_marks[leg].set_markerfacecolor("black" if live else "white")
            foot_marks[leg].set_alpha(1.0 if live else 0.4)
            if leg == HIGHLIGHT:
                hl_line.set_data_3d(*pts)
                hl_line.set_alpha(1.0 if live else 0.35)
                hl_joints.set_data_3d(pts[0, :4], pts[1, :4], pts[2, :4])
                hl_joints.set_alpha(1.0 if live else 0.35)
                q = q_true[k, leg] if live else q_default[leg]
                n = model.null_direction(q, leg)
                for g, sgn in zip(hl_ghosts, (-1, 1)):
                    g.set_data_3d(*leg_pts(k, leg, q + sgn * GHOST_ANGLE * n))
                    g.set_alpha(1.0 if live else 0.35)
        n_act.set_data(t[sl], null_act[sl])
        n_rep.set_data(t[sl], null_rep[sl])
        for name, i in (("femur", iF), ("tibia", iT)):
            jl_act[name].set_data(t[sl], joint_act[sl, i])
            jl_rep[name].set_data(t[sl], joint_rep[sl, i])
        status.set_text(f"t = {t[k]:.1f} s")
        return []

    n_frames = n_steps // args.frame_step + 1
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {n_frames} frames to {args.out}")


if __name__ == "__main__":
    main()
