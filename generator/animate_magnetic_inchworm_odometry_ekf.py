'''
Animated version of magnetic_inchworm_odometry_ekf.py, one run: MinchW, a
two-footed magnetic inchworm with six servos, climbs 10 floor steps and then
40 wall steps. On the wall its holding pad slides sideways on some steps; no
servo angle changes when it does. Three rows see the same servo readings:
servo_only (chains the forward-kinematics steps), hall_flag (adds slide noise
on the steps a Hall-sensor pad flags) and ext_fix_bias (hall_flag plus an
external position fix every 10 steps and a slide-bias state).

Three panels (the first has two parts), drawn progressively:
  1. the robot. Top: side view of the two feet and the arch between them,
     drawn from fk's own chain (the yaw-pitch-pitch chain, link lengths and
     feet spacing of the script), over the unrolled surface (floor, then
     wall). Bottom: the same robot in the wall plane (along the path x
     vertical, sideways y horizontal) with both feet, a trail of where the
     feet held, the Hall flag lamp when a slide is flagged, and a star where
     an external fix lands;
  2. the holding foot in the wall plane (sideways y in mm against along-path x
     in m): the truth (black) and each row's estimate, with the row's own
     sideways +-2 sigma band and, at the current step, its +-2 sigma bar;
  3. the sideways error (truth minus estimate) of each row with its own
     +-2 sigma (dashed), and the sum of the slides so far (dotted).

Message (doc docs/filtering/magnetic_inchworm_odometry_ekf.md): MinchW climbs
with only servos, and when the holding pad slides sideways no servo angle
changes, so servo odometry cannot see it. The slides all go one way and add
up. Slide noise on the flagged steps still leaves hall_flag's sideways error
outside its own band. An external fix every 10 steps and a slide-bias state
keep the error small and inside the band.

Seed 0 by default, the script's own default seed and the first Monte Carlo
trial (the run drawn in the static figure), chosen by a fixed rule before
looking at other seeds: use seed 0 if it has at least one flagged wall slide
(it has several). It is one run, not the doc's 200-trial averages: this run
has fewer slides than the 12 expected per run, and a large encoder-rounding
share of the sideways error; the printed summary gives the numbers.

Illustrative, not simulation output:
  - the foot motion between the two step poses (a straight interpolation,
    3 display frames per step by default), the slide happening over the first
    half of a step, and the arch drawn at the interpolated feet spacing with
    ik(d) (the script gives poses only at the end of a step). The real gait B
    slides rather than arches, and the assumed arch is taller than gait B's
    0.1875 m (doc section 1.2), so the arch is not MinchW's real shape;
  - the sideways scale of the wall-plane view is exaggerated (--exaggeration,
    default 30, labeled on the axes), and the feet there are markers, not to
    scale.
Everything else is the simulation's own simulate_trial / run_filter output
(truth, slides, flags, fixes, estimates and covariances).

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D

from state_estimation_simulations.use_numpy.magnetic_inchworm_odometry_ekf import (
    BODY, D_MIN, LEG, PAD_H, Config, fk, ik, run_filter, se2_compose, simulate_trial, step_plan)

ROWS = ("servo_only", "hall_flag", "ext_fix_bias")
COLORS = dict(servo_only="tab:red", hall_flag="tab:green", ext_fix_bias="tab:purple")
LW = dict(servo_only=4.0, hall_flag=1.6, ext_fix_bias=2.0)   # servo_only and hall_flag share one estimate: draw it wider underneath
FLOOR_COLOR, WALL_COLOR = "#e8d9b5", "#c5ccd6"


def arch_points(d):
    """(along, height) of the pad tips and joints of the arch whose feet are d apart: fk's own chain at q = ik(d)."""
    q = ik(d)
    a, s = 0.0, np.array([0.0, PAD_H])
    pts = [np.zeros(2), s.copy()]
    for ang, length in ((q[1], LEG), (q[2], BODY), (q[3], LEG)):
        a += ang
        s = s + length * np.array([np.cos(a), np.sin(a)])
        pts.append(s.copy())
    a += q[4]
    pts.append(pts[-1] + PAD_H * np.array([np.cos(a - np.pi / 2), np.sin(a - np.pi / 2)]))
    pts = np.array(pts)
    assert abs(pts[-1, 0] - fk(q)[0]) < 1e-9 and abs(pts[-1, 1]) < 1e-9      # ends on the surface at fk's separation
    return pts


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (see the module docstring)")
    parser.add_argument("--frames-per-step", type=int, default=3, help="Display frames per step (the in-between motion is illustrative)")
    parser.add_argument("--exaggeration", type=float, default=30.0, help="Sideways scale of the wall-plane view (illustrative)")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.add_argument("--out", type=str, default="magnetic_inchworm_odometry_ekf.gif", help="Output GIF path")
    parser.add_argument("--max-steps", type=int, default=None, help="Animate only the first steps (quick preview)")
    args = parser.parse_args()
    m, exag = args.frames_per_step, args.exaggeration

    cfg = Config()
    plan = step_plan(cfg)
    n = len(plan)
    trial = simulate_trial(cfg, np.random.default_rng(args.seed))
    est = {r: run_filter(r, trial, cfg) for r in ROWS}
    truth = trial.anchor                                         # (n+1, 3), before each slide
    hold_after = truth[:n].copy()                                # holder after its slide in step k
    hold_after[:, 1] += trial.slip
    start_moving = [se2_compose(truth[0], fk(ik(D_MIN)))] + [hold_after[k - 1] for k in range(1, n)]

    ty = 1000 * truth[:, 1]
    est_y = {r: 1000 * est[r][0][:, 1] for r in ROWS}
    est_x = {r: est[r][0][:, 0] for r in ROWS}
    sig2 = {r: 2000 * np.sqrt(est[r][1][:, 1, 1]) for r in ROWS}
    err = {r: ty - est_y[r] for r in ROWS}
    cum_slide = 1000 * np.concatenate([[0.0], np.cumsum(trial.slip)])
    steps = np.arange(n + 1)
    fix_steps = [k + 1 for k in range(n) if not np.isnan(trial.fix[k, 0])]
    x_corner = truth[cfg.n_floor, 0]

    wall0 = cfg.n_floor + 1
    outside = {r: int(np.sum(np.abs(err[r][wall0:]) > sig2[r][wall0:])) for r in ROWS}
    n_flag = int(trial.flag.sum())
    print(f"seed {args.seed}: {cfg.n_floor} floor + {cfg.n_wall} wall steps, {n_flag} flagged slides (expected {cfg.slip_rate * cfg.n_wall:.0f} per run), "
          f"slides sum {1000 * trial.slip.sum():.1f} mm (mean {1000 * trial.slip[trial.flag].mean():.1f} mm), largest {1000 * trial.slip.max():.1f} mm")
    print(f"flagged steps (1-based): {[int(k) + 1 for k in np.where(trial.flag)[0]]}; fixes after steps {fix_steps}")
    print(f"servo_only error at the end = slides {cum_slide[-1]:+.1f} mm + encoder rounding {err['servo_only'][-1] - cum_slide[-1]:+.1f} mm")
    for r in ROWS:
        print(f"  {r:<13s} sideways error at the end {err[r][-1]:+7.2f} mm (own 2 sigma {sig2[r][-1]:5.2f} mm), largest |error| {np.abs(err[r]).max():6.2f} mm, "
              f"wall steps outside own 2 sigma {outside[r]} of {cfg.n_wall}")

    # frames: (step k, progress s); the first frame is the start pose
    frames = [(0, 0.0)] + [(k, (i + 1) / m) for k in range(n if args.max_steps is None else args.max_steps) for i in range(m)]

    fig = plt.figure(figsize=(14, 8.1))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 0.95, 1.1], height_ratios=[1.0, 2.9], hspace=0.28, wspace=0.26)
    ax_s = fig.add_subplot(gs[0, 0])
    ax_w = fig.add_subplot(gs[1, 0])
    ax_p = fig.add_subplot(gs[:, 1])
    ax_e = fig.add_subplot(gs[:, 2])
    fig.subplots_adjust(left=0.045, right=0.985, top=0.905, bottom=0.215)
    fig.suptitle("MinchW climbing on servos only: the pad slide is invisible to the odometry", fontsize=14, fontweight="bold")

    # 1a side view
    ax_s.set_aspect("equal")
    ax_s.set(ylim=(-0.05, 0.31), yticks=[0, 0.1, 0.2, 0.3], ylabel="height (m)")
    ax_s.set_xlabel("along the unrolled surface x (m)", labelpad=1)
    ax_s.set_title("Side view: assumed arch at the interpolated spacing (illustrative)", fontsize=9)
    x_lo, x_hi = truth[0, 0] - 1.0, truth[-1, 0] + 1.0
    ax_s.fill_between([x_lo, x_corner], -0.05, 0, color=FLOOR_COLOR, lw=0)
    ax_s.fill_between([x_corner, x_hi], -0.05, 0, color=WALL_COLOR, lw=0)
    ax_s.plot([x_lo, x_corner], [0, 0], color="#8a7a50", lw=2.5)
    ax_s.plot([x_corner, x_hi], [0, 0], color="#667085", lw=2.5)
    ax_s.axvline(x_corner, color="0.4", ls=":", lw=1)
    arch_line, = ax_s.plot([], [], color="0.25", lw=5, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    arch_joints, = ax_s.plot([], [], "o", ms=5, mfc="white", mec="0.25", zorder=4)
    foot_s = [ax_s.plot([], [], "o", ms=15, mec="black", mew=1.6, zorder=5)[0] for _ in range(2)]
    surf_txt = ax_s.text(0.015, 0.93, "", transform=ax_s.transAxes, va="top", fontsize=9, fontweight="bold")

    # 1b wall plane
    y_lo, y_hi = min(ty.min(), 0) - 6, ty.max() + 8
    ax_w.set_aspect(1000.0 / exag, adjustable="box")
    ax_w.set(xlim=(y_lo, y_hi), xlabel=f"sideways y (mm), drawn x{exag:g} wider (illustrative)", ylabel="along the path x (m)")
    ax_w.set_title("Wall plane (feet are markers, not to scale; in-between foot motion illustrative)", fontsize=9)
    ax_w.axvline(0, color="0.55", ls="--", lw=1)
    ax_w.text(0, 0.0, " y = 0", transform=ax_w.get_xaxis_transform(), va="bottom", ha="left", fontsize=8, color="0.4")
    ax_w.axhline(x_corner, color="0.4", ls=":", lw=1)
    ax_w.text(0.99, x_corner, "floor | wall ", transform=ax_w.get_yaxis_transform(), ha="right", va="bottom", fontsize=8, color="0.4")
    foot_prints, = ax_w.plot([], [], "o", ms=4, color="0.6", zorder=2)
    slide_line, = ax_w.plot([], [], color="tab:red", lw=3, zorder=4, solid_capstyle="round")
    body_w, = ax_w.plot([], [], color="0.25", lw=3.5, solid_capstyle="round", zorder=3)
    foot_w = [ax_w.plot([], [], "o", ms=15, mec="black", mew=1.6, zorder=5)[0] for _ in range(2)]
    fix_star, = ax_w.plot([], [], "*", ms=17, mfc="gold", mec="black", mew=1.0, zorder=6)
    lamp = ax_w.text(0.98, 0.985, "", transform=ax_w.transAxes, ha="right", va="top", fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6"), zorder=8)
    fix_txt = ax_w.text(0.02, 0.985, "", transform=ax_w.transAxes, ha="left", va="top", fontsize=9.5, fontweight="bold", zorder=8)
    ax_w.legend(handles=[Line2D([], [], marker="o", ls="", ms=10, mfc="gold", mec="black", label="holding foot"),
                         Line2D([], [], marker="o", ls="", ms=10, mfc="tab:red", mec="black", label="holding, Hall flag on"),
                         Line2D([], [], marker="o", ls="", ms=10, mfc="white", mec="black", label="moving foot"),
                         Line2D([], [], marker="o", ls="", ms=5, color="0.6", label="where feet held (truth)"),
                         Line2D([], [], marker="*", ls="", ms=12, mfc="gold", mec="black", label="external fix lands")],
                loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False, handletextpad=0.3, columnspacing=1.0)

    # 2 holding foot in the wall plane
    p_lo = min(min((est_y[r] - sig2[r]).min() for r in ROWS), ty.min()) - 8
    p_hi = max(max((est_y[r] + sig2[r]).max() for r in ROWS), ty.max()) + 8
    ax_p.set(xlim=(p_lo, p_hi), ylim=(-0.2, truth[-1, 0] + 0.2), xlabel="sideways y of the holding foot (mm)", ylabel="along the path x (m)")
    ax_p.set_title("Holding foot: truth, estimates, own sideways ±2σ", fontsize=10)
    ax_p.axhline(x_corner, color="0.4", ls=":", lw=1)
    ax_p.axvline(0, color="0.8", lw=0.8)
    ax_p.grid(alpha=0.25)
    truth_line, = ax_p.plot([], [], color="k", lw=2.2, zorder=6)
    truth_dot, = ax_p.plot([], [], "o", color="k", ms=6, zorder=7)
    est_lines = {r: ax_p.plot([], [], color=COLORS[r], lw=LW[r], zorder=5 + (r == "hall_flag"))[0] for r in ROWS}
    est_dots = {r: ax_p.plot([], [], "D", color=COLORS[r], ms=6, mec="white", zorder=7)[0] for r in ROWS}
    est_bars = {r: ax_p.plot([], [], color=COLORS[r], lw=2.2, marker="|", ms=9, zorder=6)[0] for r in ROWS}
    bands = {r: None for r in ROWS}
    ax_p.legend(handles=[Line2D([], [], color="k", lw=2.2, label="truth"),
                         Line2D([], [], color="0.4", lw=6, alpha=0.25, label="shaded: own ±2σ"),
                         Line2D([], [], color=COLORS["servo_only"], lw=LW["servo_only"], label="servo_only (= hall_flag estimate)"),
                         Line2D([], [], color=COLORS["hall_flag"], lw=LW["hall_flag"], label="hall_flag"),
                         Line2D([], [], color=COLORS["ext_fix_bias"], lw=LW["ext_fix_bias"], label="ext_fix_bias")],
                loc="upper center", bbox_to_anchor=(0.5, -0.095), ncol=2, fontsize=8, frameon=False, handletextpad=0.4, columnspacing=1.0)

    # 3 sideways error
    e_all = np.concatenate([err[r] for r in ROWS] + [e * s for r in ROWS for e in (sig2[r],) for s in (1, -1)])
    ax_e.set(xlim=(0, n), ylim=(e_all.min() - 8, e_all.max() + 12), xlabel="step", ylabel="sideways error: truth minus estimate (mm)")
    ax_e.set_title("Sideways error and own ±2σ (dashed)", fontsize=10)
    ax_e.axvspan(0, cfg.n_floor, color=FLOOR_COLOR, alpha=0.35, lw=0)
    ax_e.text(cfg.n_floor / 2, e_all.max() + 6, "floor", ha="center", fontsize=8.5, color="0.35")
    ax_e.text((cfg.n_floor + n) / 2, e_all.max() + 6, "wall (slides)", ha="center", fontsize=8.5, color="0.35")
    for j in fix_steps:
        ax_e.axvline(j, color="tab:purple", lw=0.8, ls=":", alpha=0.6)
    ax_e.axhline(0, color="k", lw=0.8)
    ax_e.grid(alpha=0.25)
    err_lines = {r: ax_e.plot([], [], color=COLORS[r], lw=LW[r], zorder=5 + (r == "hall_flag"))[0] for r in ROWS}
    sig_lines = {r: [ax_e.plot([], [], color=COLORS[r], lw=1.1, ls="--", alpha=0.8)[0] for _ in range(2)] for r in ROWS}
    cum_line, = ax_e.plot([], [], color="k", lw=1.6, ls=":", zorder=4)
    cursor = ax_e.axvline(0, color="k", lw=1)
    ax_e.legend(handles=[Line2D([], [], color=COLORS[r], lw=LW[r], label=r) for r in ROWS] +
                [Line2D([], [], color="k", lw=1.6, ls=":", label="sum of the slides so far"),
                 Line2D([], [], color="tab:purple", lw=1, ls=":", label="external fix (ext_fix_bias)")],
                loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=8, framealpha=0.9)

    fig.text(0.5, 0.012,
             f"Slides change no servo angle: servo_only and hall_flag end {abs(err['servo_only'][-1]):.0f} mm off sideways (slides {cum_slide[-1]:.0f} mm, the rest encoder "
             f"rounding); hall_flag is outside its own ±2σ on {outside['hall_flag']} of {cfg.n_wall} wall steps.\n"
             f"A fix every 10 steps plus a slide-bias state (ext_fix_bias) ends {abs(err['ext_fix_bias'][-1]):.1f} mm off. Seed {args.seed}, one run. "
             "Foot motion between steps, the arch and the sideways scale are illustrative.", ha="center", fontsize=8.8)

    def update(frame):
        k, s = frames[frame]
        done = k + 1 if s >= 1.0 else k                           # steps completed (estimates revealed up to here)
        extend, wall = plan[k]
        w = smooth(s / 0.5) * trial.slip[k]                       # slide so far in this step (illustrative timing)
        hold = truth[k, :2] + [0.0, w]
        p0, p1 = start_moving[k][:2] + [0.0, w], truth[k + 1, :2]
        move = (1 - s) * p0 + s * p1
        rear, front = (hold, move) if extend else (move, hold)
        sliding = bool(trial.flag[k])

        # 1a side view
        d = front[0] - rear[0]
        pts = arch_points(d) + [rear[0], 0.0]
        arch_line.set_data(pts[:, 0], pts[:, 1])
        arch_joints.set_data(pts[1:-1, 0], pts[1:-1, 1])
        for ft, p, is_hold in zip(foot_s, (rear, front), (extend, not extend)):
            ft.set_data([p[0]], [0.0])
            ft.set_markerfacecolor(("tab:red" if sliding else "gold") if is_hold else "white")
        mid = 0.5 * (rear[0] + front[0])
        ax_s.set_xlim(mid - 0.95, mid + 0.95)
        surf_txt.set_text(f"step {min(k + 1, n)} of {n}, {'wall' if wall else 'floor'}, {'extend' if extend else 'close'}")

        # 1b wall plane
        for ft, p, is_hold in zip(foot_w, (rear, front), (extend, not extend)):
            ft.set_data([1000 * p[1]], [p[0]])
            ft.set_markerfacecolor(("tab:red" if sliding else "gold") if is_hold else "white")
        body_w.set_data([1000 * rear[1], 1000 * front[1]], [rear[0], front[0]])
        foot_prints.set_data(ty[1:done + 1], truth[1:done + 1, 0])
        if sliding and w > 0:
            slide_line.set_data([ty[k], 1000 * hold[1]], [hold[0], hold[0]])
        else:
            slide_line.set_data([], [])
        ax_w.set_ylim(mid - 0.65, mid + 0.65)
        fix_now = [j for j in fix_steps if (j == k + 1 and s >= 0.75) or (j == k and s <= 0.25)]
        if fix_now:
            j = fix_now[0]
            fix_star.set_data([1000 * trial.fix[j - 1, 1]], [trial.fix[j - 1, 0]])
        else:
            fix_star.set_data([], [])
        if sliding:
            lamp.set_text(f"Hall flag ON, slide {1000 * w:.1f} mm")
            lamp.set_color("tab:red")
            lamp.get_bbox_patch().set_facecolor("#fde0dc")
        else:
            lamp.set_text("Hall flag off")
            lamp.set_color("0.4")
            lamp.get_bbox_patch().set_facecolor("white")
        fix_txt.set_text("EXTERNAL FIX" if fix_now else "")

        # 2 holding foot
        sl = slice(0, done + 1)
        truth_line.set_data(ty[sl], truth[sl, 0])
        truth_dot.set_data([ty[done]], [truth[done, 0]])
        for r in ROWS:
            est_lines[r].set_data(est_y[r][sl], est_x[r][sl])
            est_dots[r].set_data([est_y[r][done]], [est_x[r][done]])
            est_bars[r].set_data([est_y[r][done] - sig2[r][done], est_y[r][done] + sig2[r][done]], [est_x[r][done]] * 2)
            if bands[r] is not None:
                bands[r].remove()
            bands[r] = ax_p.fill_betweenx(est_x[r][sl], (est_y[r] - sig2[r])[sl], (est_y[r] + sig2[r])[sl], color=COLORS[r], alpha=0.13, lw=0)

        # 3 sideways error
        for r in ROWS:
            err_lines[r].set_data(steps[sl], err[r][sl])
            sig_lines[r][0].set_data(steps[sl], sig2[r][sl])
            sig_lines[r][1].set_data(steps[sl], -sig2[r][sl])
        cum_line.set_data(steps[sl], cum_slide[sl])
        cursor.set_xdata([done, done])
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
