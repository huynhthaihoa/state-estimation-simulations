'''
Animated version of pose_graph.py, one run at the script's defaults (seed 0):
a robot drives a 2 m square (four poses X0..X3), builds its trajectory by
composing noisy odometry edges (dead reckoning), then detects it is back at
the start and adds one loop-closure edge X3-X0. Levenberg-Marquardt pose-graph
relaxation then moves all poses together.

Story, in three stages:
  1. the dead-reckoned poses X0, X1, X2, X3 appear one by one next to the
     ground-truth square, and the drift builds (per-pose position error 0,
     3.3, 7.6, 10.3 cm);
  2. the loop-closure edge X3-X0 appears. Composing X3 with the edge's
     measurement predicts where X0 should be; that predicted position is not
     where X0 is (the "disagreement"; distance shown in cm);
  3. the optimizer's accepted steps 1, 2, 3, ... (iteration k = the pose list
     returned by run_pose_graph_optimization with gn_max_iters=k) are shown as
     discrete states, each held for a while. After iteration 3 the poses do not
     move any more at the precision shown; the run converges at iteration 5.

Four panels:
  1. left: the loop from above, to scale (x-y, meters): ground truth (black),
     dead reckoning (gray) and the current estimate (blue). The errors are
     centimeters on a ~2 m loop, so the other panels carry the numbers;
  2. top right: the error vector (estimate - truth, cm) of every pose, split
     into x, y, z. Gray outlines: dead reckoning; colored bars: current state.
     The top view in panel 1 cannot show z, which holds a large part of the
     error;
  3. middle right: the position error per pose (3-D norm, cm) as in
     pose_errors, dead reckoning (gray) vs the current state (blue);
  4. bottom right: the total chi-squared error per iteration (log scale).

Message (doc docs/optimization/pose_graph_optimization.md section 5): the
optimizer spreads the loop error over the chain instead of fixing only the
last pose. X3 improves a lot (10.3 -> 4.9 cm) and the RMS position error
drops (6.6 -> 5.2 cm), but X1 and X2 end slightly worse.

Data source: only use_numpy/pose_graph.py's own functions with the defaults
that script's main() and generator/make_figures.py use (side 2.0 m, position
noise 0.05 m, rotation noise 0.01 rad, loop noise scale 0.5, damping 0.01,
tolerance 1e-6, identity information matrix): generate_ground_truth_trajectory,
simulate_noisy_edges, run_dead_reckoning, run_pose_graph_optimization (its
stdout is discarded, except for the chi-squared lines parsed for a check) and
pose_errors. The pose lists for iteration k = 0..5 come from separate calls with
gn_max_iters = k. The total chi-squared error is not returned by that function,
so cost() below is a copy of its inner cost function; the script asserts that
it reproduces the chi-squared values the optimizer prints for iterations
1..5 (the cost before each step, i.e. the cost of states 0..4) to the printed 6
decimals, and that the state after 10 iterations (the script's default) equals
the state after 5 (converged). The "loop-edge disagreement" is the 3-D distance
between X0's position and the position of X3 @ Z_30 (Z_30 the loop-closure
measurement), which is the quantity the loop residual e = Log((X3 Z_30)^-1 X0)
compares; it is computed from the same poses and measurement.

Illustrative items: the stage timing (hold frames), the colors and the bar
layout. No pose is interpolated or magnified: every drawn pose is a returned
pose, and the error panels are in true cm.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse
import contextlib
import io
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Patch

from state_estimation_simulations.use_numpy.lie_utils import se3_inv, se3_log
from state_estimation_simulations.use_numpy.numpy_utils import pose_errors
from state_estimation_simulations.use_numpy.pose_graph import (
    generate_ground_truth_trajectory, run_dead_reckoning, run_pose_graph_optimization, simulate_noisy_edges)

# pose_graph.py's own defaults
SIDE, POS_STD, ROT_STD, LOOP_SCALE = 2.0, 0.05, 0.01, 0.5
DAMPING, GN_TOL, GN_MAX = 0.01, 1e-6, 10
N_STATES = 6          # iteration k = 0..5 (the default run converges at iteration 5)
SHOWN_UNTIL = 3       # last iteration at which poses still move at the precision shown
COMP_COLORS = ("tab:orange", "tab:green", "tab:purple")
HOLD = dict(intro=6, pose=9, loop=24, move=14, still=8, final=45)   # frames per stage (illustrative)


def run_optimizer(T_init, constraints, iters):
    """run_pose_graph_optimization with its stdout captured; returns (poses, printed text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        T = run_pose_graph_optimization(T_init, constraints, np.eye(6), DAMPING, GN_TOL, iters)
    return T, buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (the script's default is 0)")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.add_argument("--out", type=str, required=True, help="Output GIF path")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    gt = generate_ground_truth_trajectory(SIDE)
    odo, loop = simulate_noisy_edges(gt, POS_STD, ROT_STD, LOOP_SCALE, rng)
    constraints = odo + loop
    T_dr = run_dead_reckoning(gt[0], odo)
    Z_loop = loop[0][2]
    n = len(gt)

    # pose lists for iteration k = 0..5, and the optimizer's own printout for the chi-squared check
    states = [run_optimizer(T_dr, constraints, k)[0] for k in range(N_STATES)]
    final_default, printout = run_optimizer(T_dr, constraints, GN_MAX)
    printed = [float(v) for v in re.findall(r"total chi-squared error = ([0-9.]+)", printout)]

    def cost(T_list):
        # copy of run_pose_graph_optimization's inner cost() (identity information matrix)
        total = 0.0
        for i, j, Z_ij in constraints:
            e = se3_log(se3_inv(T_list[i] @ Z_ij) @ T_list[j])
            total += e @ e
        return total

    chi2 = [cost(T) for T in states]
    assert len(printed) == 5 and np.allclose(np.round(chi2[:5], 6), printed, atol=1e-9), (chi2, printed)
    assert all(np.allclose(a, b, atol=1e-9) for a, b in zip(states[5], final_default))
    assert all(np.allclose(a, b) for a, b in zip(states[0], T_dr))

    pos_err = [pose_errors(gt, T)[1] * 100 for T in states]                    # cm, 3-D norm
    comp_err = [np.array([(T[i] - gt[i])[0:3, 3] for i in range(n)]) * 100 for T in states]   # cm, (n, 3)
    assert np.allclose(np.linalg.norm(comp_err[5], axis=1), pos_err[5])
    gap = [np.linalg.norm((T[n - 1] @ Z_loop)[0:3, 3] - T[0][0:3, 3]) * 100 for T in states]   # cm
    rms = [np.sqrt(np.mean(e ** 2)) for e in pos_err]
    # the poses no longer move after iteration SHOWN_UNTIL at the precision shown (0.01 cm)
    assert all(np.abs(comp_err[k] - comp_err[SHOWN_UNTIL]).max() < 0.01 for k in range(SHOWN_UNTIL, N_STATES))

    # match the doc / script printout
    assert np.allclose(np.round(pos_err[0] / 100, 4), [0, 0.0333, 0.0761, 0.1035])
    assert round(rms[0] / 100, 4) == 0.0664 and round(rms[5] / 100, 4) == 0.0520
    assert round(pos_err[5][3] / 100, 4) == 0.0494

    print(f"seed {args.seed}, defaults of pose_graph.py; {n} poses, {len(odo)} odometry + 1 loop-closure edge")
    print("  position error per pose, cm (dead reckoning):", np.round(pos_err[0], 2))
    print("  position error per pose, cm (after iteration 5):", np.round(pos_err[5], 2))
    print(f"  RMS position error {rms[0]:.2f} cm -> {rms[5]:.2f} cm; X3 {pos_err[0][3]:.2f} -> {pos_err[5][3]:.2f} cm")
    print("  chi-squared per iteration:", " ".join(f"{c:.6f}" for c in chi2))
    print("  loop-edge disagreement, cm:", " ".join(f"{g:.2f}" for g in gap))

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.0], height_ratios=[1, 1, 1], hspace=0.62, wspace=0.18)
    ax_m = fig.add_subplot(gs[:, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_x = fig.add_subplot(gs[2, 1])
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.14)
    fig.suptitle("Pose-graph relaxation: one loop closure spreads the drift over the whole chain",
                 fontsize=12, fontweight="bold")

    def xy(T_list, count):
        return np.array([T[0:2, 3] for T in T_list[:count]])

    # panel 1: loop from above, to scale
    g_pts = xy(gt, n)
    ax_m.plot(*np.vstack([g_pts, g_pts[:1]]).T, "-", color="k", lw=1.2, alpha=0.5)
    ax_m.plot(*g_pts.T, "o", color="k", ms=8, label="ground truth")
    dr_line, = ax_m.plot([], [], "o--", color="tab:gray", lw=1.8, ms=6, label="dead reckoning (odometry only)")
    est_line, = ax_m.plot([], [], "o-", color="tab:blue", lw=1.8, ms=6, label="current estimate")
    loop_line, = ax_m.plot([], [], ls=(0, (5, 3)), color="tab:red", lw=2.6, label="loop-closure edge X3-X0")
    labels = [ax_m.text(g_pts[k, 0], g_pts[k, 1], f"X{k}", fontsize=10, fontweight="bold", visible=False,
                        ha="right" if k in (0, 3) else "left", va="top" if k in (0, 1) else "bottom") for k in range(n)]
    for k, lab in enumerate(labels):
        lab.set_position((g_pts[k, 0] + (-0.1 if k in (0, 3) else 0.1), g_pts[k, 1] + (-0.1 if k in (0, 1) else 0.1)))
    ax_m.set_aspect("equal")
    ax_m.set(xlim=(-0.55, 2.55), ylim=(-0.5, 2.5), xlabel="x (m)", ylabel="y (m)")
    ax_m.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8.5, ncol=2, frameon=False)
    ax_m.text(0.5, 0.5, "to scale: the errors are centimeters\non a 2 m loop (see the panels on the right)",
              transform=ax_m.transAxes, ha="center", va="center", fontsize=9, color="0.4")
    banner = ax_m.set_title("", fontsize=9.5, fontweight="bold", loc="left")

    # panel 2: error components
    xs = np.arange(n)
    bw = 0.25
    comp_dr_bars, comp_bars = [], []
    for c in range(3):
        comp_dr_bars.append(ax_c.bar(xs + (c - 1) * bw, comp_err[0][:, c], bw, fc="none", ec="tab:gray", lw=1.4, ls="--", visible=False))
        comp_bars.append(ax_c.bar(xs + (c - 1) * bw, comp_err[0][:, c], bw, color=COMP_COLORS[c],
                                  label="xyz"[c], visible=False))
    ax_c.axhline(0, color="k", lw=0.8)
    lim = 1.25 * max(np.abs(ce).max() for ce in comp_err)
    ax_c.set(xticks=xs, xticklabels=["X0 (anchored)"] + [f"X{k}" for k in range(1, n)], ylim=(-lim, lim), ylabel="estimate - truth (cm)")
    ax_c.set_title("Error vector per pose; gray outline: dead reckoning", fontsize=9.5, loc="left")
    ax_c.legend(handles=[Patch(fc=COMP_COLORS[c], label="xyz"[c]) for c in range(3)], loc="upper left", ncol=3,
                fontsize=8.5, frameon=False, handlelength=1, columnspacing=1)
    ax_c.grid(True, axis="y", alpha=0.3)

    # panel 3: position error per pose (3-D norm)
    w = 0.38
    bars_dr = ax_e.bar(xs - w / 2, pos_err[0], w, color="tab:gray", label="dead reckoning", visible=False)
    bars_cur = ax_e.bar(xs + w / 2, pos_err[0], w, color="tab:blue", label="current estimate", visible=False)
    txt_dr = [ax_e.text(k - w / 2, pos_err[0][k] + 0.3, f"{pos_err[0][k]:.2f}", ha="center", fontsize=8, visible=False) for k in range(n)]
    txt_cur = [ax_e.text(k + w / 2, 0.3, "", ha="center", fontsize=8, color="tab:blue", fontweight="bold", visible=False) for k in range(n)]
    ax_e.set(xticks=xs, xticklabels=["X0 (anchored)"] + [f"X{k}" for k in range(1, n)], ylim=(0, 1.65 * pos_err[0].max()),
             ylabel="position error (cm)")
    ax_e.set_title("Position error per pose (3-D distance to truth)", fontsize=9.5, loc="left")
    ax_e.legend(handles=[Patch(fc="tab:gray", label="dead reckoning"), Patch(fc="tab:blue", label="current estimate")],
                loc="upper left", fontsize=8.5, frameon=False)
    ax_e.grid(True, axis="y", alpha=0.3)
    rms_text = ax_e.text(0.99, 0.97, "", transform=ax_e.transAxes, va="top", ha="right", fontsize=9, fontweight="bold")

    # panel 4: chi-squared per iteration
    bars_x = ax_x.bar(np.arange(N_STATES), chi2, 0.6, color="tab:red", alpha=0.85, visible=False)
    txt_x = [ax_x.text(k, chi2[k] * 1.12, f"{chi2[k]:.6f}", ha="center", fontsize=8, visible=False) for k in range(N_STATES)]
    ax_x.set_yscale("log")
    ax_x.set(xticks=np.arange(N_STATES), xticklabels=["0 (dead\nreckoning)"] + [str(k) for k in range(1, N_STATES)],
             ylim=(chi2[-1] * 0.5, chi2[0] * 4), xlabel="optimizer iteration (accepted steps)", ylabel="chi-squared")
    ax_x.set_title("Total chi-squared error of all edges (log scale)", fontsize=9.5, loc="left")
    ax_x.grid(True, axis="y", which="major", alpha=0.3)

    fig.text(0.5, 0.012,
             f"Seed {args.seed}, pose_graph.py defaults, one run. Poses are the script's own outputs for gn_max_iters = 0..5, shown as discrete states "
             "(nothing interpolated); errors are 3-D (x, y, z).\n"
             f"X3: {pos_err[0][3]:.1f} -> {pos_err[5][3]:.1f} cm and RMS {rms[0]:.1f} -> {rms[5]:.1f} cm, but the error is shared along the chain: "
             f"X1 {pos_err[0][1]:.2f} -> {pos_err[5][1]:.2f} cm and X2 {pos_err[0][2]:.2f} -> {pos_err[5][2]:.2f} cm get slightly worse.",
             ha="center", fontsize=9)

    # ------------------------------------------------------------------ frames
    # each frame: (poses visible, loop edge visible, iteration state, banner)
    frames = [(0, False, 0, "Ground truth: the robot will drive X0 -> X1 -> X2 -> X3")] * HOLD["intro"]
    for nv in range(1, n + 1):
        text = ("Dead reckoning: X0 is anchored at the true start" if nv == 1 else
                f"Dead reckoning: X{nv - 1} = X{nv - 2} composed with the noisy\nodometry edge, {pos_err[0][nv - 1]:.1f} cm off the truth")
        frames += [(nv, False, 0, text)] * HOLD["pose"]
    frames += [(n, True, 0, f"Loop-closure edge X3-X0: X3 composed with it puts X0\n{gap[0]:.1f} cm from where X0 is, "
                            f"chi-squared = {chi2[0]:.6f}")] * HOLD["loop"]
    for k in range(1, N_STATES):
        if k <= SHOWN_UNTIL:
            text = (f"Optimizer iteration {k}: chi-squared {chi2[k - 1]:.6f} -> {chi2[k]:.6f}\n"
                    f"all poses move together, X0 stays; edge disagreement {gap[k]:.1f} cm")
            hold = HOLD["move"]
        else:
            text = (f"Iteration {k}: chi-squared {chi2[k]:.6f}, the poses do not move any more"
                    + ("\n(converged: step size below 1e-6)" if k == N_STATES - 1 else ""))
            hold = HOLD["still"]
        frames += [(n, True, k, text)] * hold
    frames += [(n, True, N_STATES - 1, f"Converged at iteration 5: chi-squared {chi2[0]:.6f} -> {chi2[-1]:.6f}\n"
                                       f"X3 {pos_err[0][3]:.1f} -> {pos_err[5][3]:.1f} cm, RMS {rms[0]:.1f} -> {rms[5]:.1f} cm")] * HOLD["final"]

    def update(i):
        nv, loop_on, k, text = frames[i]
        banner.set_text(text)
        for kk in range(n):
            labels[kk].set_visible(kk < nv)
        dr_line.set_data(*xy(T_dr, nv).T if nv else ([], []))
        moved = loop_on and k >= 1
        if moved:
            est_line.set_data(*xy(states[k], nv).T)
        else:
            est_line.set_data([], [])
        if loop_on:
            cur = states[k]
            loop_line.set_data([cur[n - 1][0, 3], cur[0][0, 3]], [cur[n - 1][1, 3], cur[0][1, 3]])
        else:
            loop_line.set_data([], [])
        # error components
        for c in range(3):
            for kk in range(n):
                comp_dr_bars[c][kk].set_visible(kk < nv)
                comp_bars[c][kk].set_height(comp_err[k][kk, c])
                comp_bars[c][kk].set_visible(kk < nv and moved)
        # position error per pose
        for kk in range(n):
            bars_dr[kk].set_visible(kk < nv)
            txt_dr[kk].set_visible(kk < nv)
            bars_cur[kk].set_visible(kk < nv and moved)
            bars_cur[kk].set_height(pos_err[k][kk])
            txt_cur[kk].set_visible(kk < nv and moved)
            txt_cur[kk].set_text(f"{pos_err[k][kk]:.2f}")
            txt_cur[kk].set_y(pos_err[k][kk] + 0.3)
        txt_cur[0].set_visible(False)       # X0 is anchored at the true start, error 0 for both
        if nv == n and not moved:
            rms_text.set_text(f"RMS over the 4 poses (dead reckoning): {rms[0]:.1f} cm")
        elif moved:
            rms_text.set_text(f"RMS over the 4 poses: {rms[0]:.1f} cm (gray) -> {rms[k]:.1f} cm (blue)")
        else:
            rms_text.set_text("")
        # chi-squared
        for kk in range(N_STATES):
            vis = loop_on and kk <= k
            bars_x[kk].set_visible(vis)
            txt_x[kk].set_visible(vis)
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
