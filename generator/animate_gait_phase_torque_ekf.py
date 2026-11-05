'''
Animated version of gait_phase_torque_ekf.py, one run: the planar hexapod of
gait_phase_ekf.py walks a tripod gait whose feet land late or early, and the
contact detector reads joint torque instead of servo position error (HEBI
Daisy numbers). The estimated vertical foot force of each leg is a simulated
signal at 1 kHz: a load ramp after the true touchdown, plus one model-error
pulse per step in the first half of the swing-down, plus noise. Three torque
rows decide which feet are planted: torque_detector (the first threshold
crossing), torque_phase_ekf (gait_phase_ekf's phase EKF on the torque
triggers) and torque_fused (gait-phase prior plus force logistic).

Three panels, drawn progressively:
  1. top view (left): the true hexapod (body, six legs, feet at their
     simulated positions; filled = truly planted, hollow = in swing; the
     animated leg in orange). Next to each foot three small dots, one per
     torque row (red torque_detector, green torque_phase_ekf, blue
     torque_fused): filled = that row reads the foot as planted, hollow =
     reads it as swinging. A pink ring marks a swinging foot that at least one
     row reads as planted. The camera follows the body, the grid is fixed in
     the world;
  2. estimated vertical force of the animated leg (right, top), scrolling with
     time: the force signal, the true load (the ramp alone, no pulse or
     noise), the torque threshold, the first half of each swing-down (the
     window where the model-error pulse acts), the true touchdowns and the
     torque triggers (a red triangle for a trigger after the touchdown, a
     pink X for an early one);
  3. stance flags of that leg (right, bottom): the truth and the three torque
     rows, with gold where a swinging foot is read as planted; the blocked
     window is shaded.

Message (doc docs/filtering/gait_phase_torque_ekf.md section 3): a raw torque
threshold reads a swinging foot as planted whenever the unmodeled leg
dynamics (the model-error pulse) push the force over the threshold before the
touchdown; running gait_phase_ekf's phase EKF on the triggers removes those
early triggers; the fused log-odds is in between.

Seed 0 (the default of the simulation, same rng stream as its single run).
Over the 100-trial Monte Carlo of the doc, 2.4% of steps have an early torque
trigger and NEES is 14.09 (torque_detector), 6.30 (torque_phase_ekf) and 6.91
(torque_fused). In this run the window starts at 7.4 s, just before L2's
early trigger (8.84 s, 0.18 s before the touchdown), and ends 0.6 s after the
blocked window; main() prints the early-trigger rate of the whole run, which
is of the same size as the Monte Carlo's. Within the window the fused row's
only errors are on the blocked leg L2 (doc section 3). The body EKF is not
re-run (no NEES per run is shown).

Everything drawn is the simulation's own torque_trial output (signals,
triggers, stance flags, simulated foot positions). The force signal itself is
simulated with assumed model-error, load-ramp and noise values (HEBI
publishes none), not a measurement. The true load line recomputes the load
part with the script's own formula. The body outline and the hip positions
are schematic, as in animate_gait_phase_ekf.py. --n-mc-trials has no effect.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.ticker import MultipleLocator

from state_estimation_simulations.use_numpy import gait_phase_torque_ekf as gt
from state_estimation_simulations.use_numpy.hexapod_gait_ekf import LEG_NAMES, NOMINAL_FOOT_OFFSETS, rot

TORQUE = gt.TORQUE_ROWS
COLORS = {"torque_detector": "tab:red", "torque_phase_ekf": "tab:green", "torque_fused": "tab:blue",
          "oracle": "black"}
ROW_ORDER = ("oracle",) + TORQUE  # top to bottom
ROW_LABELS = {"oracle": "true (oracle)", "torque_detector": "torque_detector",
              "torque_phase_ekf": "torque_phase_ekf", "torque_fused": "torque_fused"}
BLOCK_COLOR = "tab:orange"
FALSE_COLOR = "deeppink"
FORCE_COLOR = "tab:purple"
BODY_POLY = np.array([[-0.17, -0.10], [0.17, -0.10], [0.24, 0.0], [0.17, 0.10], [-0.17, 0.10]])
HIPS = np.column_stack([0.6 * NOMINAL_FOOT_OFFSETS[:, 0], 0.09 * np.sign(NOMINAL_FOOT_OFFSETS[:, 1])])
VIEW_HALF = 0.6  # half-width of the following camera (m)
PAST, FUTURE = 2.5, 0.5  # scrolling window around the cursor (s)
DOT_GAP = 0.05  # spacing of the three row dots next to a foot (m)


def main():
    parser = gt.build_parser()
    parser.description = __doc__
    # The simulation already has --out (figure path): reuse it as the GIF path
    parser.set_defaults(out="gait_phase_torque_ekf.gif")
    for action in parser._actions:
        if action.dest == "out":
            action.help = "Output GIF path"
    parser.add_argument("--leg", type=str, default="L2", choices=LEG_NAMES, help="Leg whose force is shown")
    parser.add_argument("--t-start", type=float, default=7.4, help="First animated time (s)")
    parser.add_argument("--t-end", type=float, default=14.6, help="Last animated time (s)")
    parser.add_argument("--frame-step", type=int, default=3, help="Filter ticks per GIF frame")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    args = parser.parse_args()

    # Same rng stream as gait_phase_torque_ekf.main's single run
    rng = np.random.default_rng(args.seed)
    model = gt.TorqueModel(args)
    logistic = gt.train_logistic(args, model, args.seed + 1000)
    x_true, _, r, _, flags, info = gt.torque_trial(args, rng, model, logistic)
    dt = args.dt
    n_pts = len(x_true)
    t = np.arange(n_pts) * dt
    leg = LEG_NAMES.index(args.leg)
    blocked_on = args.blocked_delay > 0
    k0, k1 = int(round(args.t_start / dt)), min(int(round(args.t_end / dt)), n_pts - 1)
    if not 0 <= k0 < k1:
        raise SystemExit("need 0 <= t-start < t-end <= duration")
    t0, t1 = t[k0], t[k1]
    frames = list(range(k0, k1 + 1, args.frame_step))
    oracle = flags["oracle"]

    # Per-step data of the animated leg
    steps, td = info["steps"], info["td"]
    sig, trig = info["torque"]["signals"], info["torque"]["triggers"]
    wins = info["torque"]["windows"]
    pulse_len = (gt.swing_ticks_of(args) // 6) * dt
    mine = [j for j, s in enumerate(steps) if s[0] == leg]
    t_sig, f_sig, load = [], [], []
    for j in mine:
        ts, fs = sig[j]
        t_sig += [ts, [np.nan]]
        f_sig += [fs, [np.nan]]
        load += [model.share * np.clip((ts - td[j] * dt) / model.ramp, 0.0, 1.0), [np.nan]]
    t_sig, f_sig, load = np.concatenate(t_sig), np.concatenate(f_sig), np.concatenate(load)
    early_k, late_k = [], []
    for j in mine:
        early_k += [k for k in trig[j] if k < td[j]]
        late_k += [k for k in trig[j] if k >= td[j]]

    # Printed summary
    print(f"Leg {args.leg} force and stance, seed {args.seed}; animating t in [{t0:.1f}, {t1:.1f}] s; "
          f"threshold {model.threshold:.1f} N, load share {model.share:.1f} N, ramp {1e3 * model.ramp:.0f} ms, "
          f"model-error std {model.model_err_std:.0f} N")
    n_early = sum(any(k < d for k in tr) for tr, d in zip(trig, td))
    print(f"  whole run: {n_early} of {len(steps)} steps ({100 * n_early / len(steps):.1f}%) have a torque trigger "
          f"before the true touchdown (Monte Carlo of the doc: 2.4%)")
    win = slice(k0, k1 + 1)
    for j in range(len(steps)):
        early = [k for k in trig[j] if k < td[j]]
        if early and k0 <= early[0] <= k1:
            amp = model.model_err_std * info["torque"]["draws"][j][0]
            ts, fs = sig[j]
            peak = fs[ts < td[j] * dt].max()
            print(f"  early torque trigger in the window: {LEG_NAMES[steps[j][0]]} at {early[0] * dt:.2f} s, "
                  f"{(td[j] - early[0]) * dt:.2f} s before the true touchdown ({td[j] * dt:.2f} s); "
                  f"pulse amplitude {amp:+.1f} N, force peak before touchdown {peak:.1f} N")
    for a in TORQUE:
        bad = flags[a][win] & ~oracle[win]
        per_leg = ", ".join(f"{LEG_NAMES[i]} {int(bad[:, i].sum())}" for i in range(6) if bad[:, i].any())
        print(f"  {a:<17s} swinging feet read as planted in the window: {int(bad.sum())} ticks"
              + (f" ({per_leg})" if per_leg else ""))
        ks = np.flatnonzero(flags[a][win, leg] & ~oracle[win, leg]) + k0
        if len(ks):
            starts = ks[np.r_[True, np.diff(ks) > 1]]
            print(f"      {args.leg} read-as-planted runs start at t = " + ", ".join(f"{s * dt:.2f}" for s in starts) + " s")

    # --- Figure layout ---
    fig = plt.figure(figsize=(12, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.22)
    gs_r = gs[1].subgridspec(3, 1, height_ratios=[3.0, 2.2, 1.0], hspace=0.28)
    ax_xy = fig.add_subplot(gs[0])
    ax_f = fig.add_subplot(gs_r[0])
    ax_g = fig.add_subplot(gs_r[1], sharex=ax_f)
    ax_fl = fig.add_subplot(gs_r[2])
    ax_fl.axis("off")
    fig.subplots_adjust(left=0.05, right=0.985, top=0.91, bottom=0.27)
    fig.suptitle("Torque contact detection on a hexapod: a model-error pulse fools the raw threshold", fontsize=14,
                 fontweight="bold")

    # --- Top view ---
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.xaxis.set_major_locator(MultipleLocator(0.25))
    ax_xy.yaxis.set_major_locator(MultipleLocator(0.25))
    ax_xy.grid(alpha=0.3)
    ax_xy.tick_params(labelsize=9)
    ax_xy.set_xlabel("world x (m); the camera follows the body", fontsize=10)
    ax_xy.set_ylabel("world y (m)", fontsize=10)
    ax_xy.set_title("Top view: true stance vs what each torque row reads", fontsize=11)
    trail_n, trail_gap = 20, 10
    trail = [ax_xy.plot([], [], color="black", lw=3, solid_capstyle="round")[0] for _ in range(trail_n)]
    body = Polygon(BODY_POLY, closed=True, fc="#dfe7f2", ec="#4a5a70", lw=1.8, zorder=3)
    ax_xy.add_patch(body)
    legs, feet, names, halos, dots = [], [], [], [], []
    for i in range(6):
        isb = i == leg
        legs.append(ax_xy.plot([], [], color=BLOCK_COLOR if isb else "#5b6b7f", lw=3.2 if isb else 2.2, zorder=2)[0])
        halo = Circle((0, 0), 0.062, fc="none", ec=FALSE_COLOR, lw=2.5, zorder=4)
        ax_xy.add_patch(halo)
        halos.append(halo)
        feet.append(ax_xy.plot([], [], "o", ms=13 if isb else 11, mec="black", mew=1.8 if isb else 1.4, zorder=5)[0])
        dots.append([ax_xy.plot([], [], "o", ms=6.5, mew=1.6, color=COLORS[a], zorder=6)[0] for a in TORQUE])
        names.append(ax_xy.text(0, 0, LEG_NAMES[i], ha="center", va="center", fontsize=10 if isb else 8.5,
                                fontweight="bold" if isb else "normal", color="darkorange" if isb else "#33404f",
                                zorder=6))
    status = ax_xy.text(0.02, 0.98, "", transform=ax_xy.transAxes, ha="left", va="top", fontsize=10.5,
                        fontweight="bold", linespacing=1.4, zorder=7,
                        bbox=dict(fc="white", ec="none", alpha=0.8, pad=2))
    handles = [Line2D([], [], marker="o", ls="", ms=10, mfc="black", mec="black", label="foot planted (truly in stance)"),
               Line2D([], [], marker="o", ls="", ms=10, mfc="white", mec="black", label="foot in swing"),
               Line2D([], [], marker="o", ls="", ms=10, mfc="none", mec=FALSE_COLOR, mew=2.5,
                      label="swinging foot read as planted"),
               Line2D([], [], color="black", lw=2.5, label="body trail (last 4 s)")]
    handles += [Line2D([], [], marker="o", ls="", ms=7, mfc=COLORS[a], mec=COLORS[a], label=f"{a}: filled = planted")
                for a in TORQUE]
    ax_xy.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5, ncol=2, frameon=False)

    # --- Estimated vertical force of the animated leg ---
    for j in mine:
        kd = wins[j][1] * dt
        if t0 - PAST - 1 <= kd <= t1 + FUTURE:
            ax_f.axvspan(kd, kd + pulse_len, color="tab:gray", alpha=0.18, lw=0, zorder=0)
            ax_f.axvline(td[j] * dt, color="black", lw=1.0, ls=":", zorder=1)
    ax_f.plot(t_sig, load, color="black", lw=1.2, zorder=2)
    ax_f.plot(t_sig, f_sig, color=FORCE_COLOR, lw=1.3, zorder=3)
    ax_f.axhline(model.threshold, color="black", ls="--", lw=1.2, zorder=2)
    ax_f.plot(t[late_k], [model.threshold] * len(late_k), "v", color=COLORS["torque_detector"], ms=8, mec="black",
              mew=0.8, ls="", zorder=5)
    ax_f.plot(t[early_k], [model.threshold] * len(early_k), "X", color=FALSE_COLOR, ms=10, mec="black", mew=0.8,
              ls="", zorder=5)
    in_win = (t_sig >= t0 - PAST) & (t_sig <= t1 + FUTURE)
    lo, hi = np.nanmin(f_sig[in_win]), np.nanmax(f_sig[in_win])
    ax_f.set_ylim(min(lo, 0.0) - 0.06 * (hi - lo), hi + 0.08 * (hi - lo))
    ax_f.set_ylabel("force (N)", fontsize=10)
    ax_f.set_title(f"Estimated vertical foot force of leg {args.leg} (simulated signal, 1 kHz)", fontsize=11)
    ax_f.tick_params(labelsize=9)
    ax_f.grid(alpha=0.3)
    plt.setp(ax_f.get_xticklabels(), visible=False)
    ax_fl.legend(handles=[Line2D([], [], color=FORCE_COLOR, lw=2, label="estimated force"),
                          Line2D([], [], color="black", lw=1.2, label="true load (ramp alone)"),
                          Line2D([], [], color="black", ls="--", lw=1.2,
                                 label=f"torque threshold ({model.threshold:.1f} N)"),
                          Line2D([], [], color="black", ls=":", lw=1.2, label="true touchdown"),
                          Rectangle((0, 0), 1, 1, fc="tab:gray", alpha=0.3, lw=0,
                                    label="model-error window"),
                          Line2D([], [], marker="v", ls="", ms=8, mfc=COLORS["torque_detector"], mec="black",
                                 label="torque trigger"),
                          Line2D([], [], marker="X", ls="", ms=10, mfc=FALSE_COLOR, mec="black",
                                 label="early trigger"),
                          Line2D([], [], color="gold", lw=6, label="swing read as planted"),
                          Line2D([], [], color=BLOCK_COLOR, lw=6, alpha=0.3, label="blocked window")],
                 loc="center", ncol=3, fontsize=8.5, frameon=False, borderaxespad=0.0, columnspacing=1.2)

    # --- Stance flags of the animated leg ---
    ypos = {a: len(ROW_ORDER) - 1 - j for j, a in enumerate(ROW_ORDER)}
    for a in ROW_ORDER:
        y = ypos[a]
        ax_g.fill_between(t, y - 0.35, y + 0.35, where=flags[a][:, leg], step="post", color=COLORS[a], alpha=0.65, lw=0)
        bad = flags[a][:, leg] & ~oracle[:, leg]
        ax_g.fill_between(t, y - 0.35, y + 0.35, where=bad, step="post", color="gold", lw=0, zorder=3)
    ax_g.set_yticks([ypos[a] for a in ROW_ORDER], [ROW_LABELS[a] for a in ROW_ORDER], fontsize=9.5)
    ax_g.set_ylim(-0.6, len(ROW_ORDER) - 0.4)
    ax_g.set_xlabel("time (s)", fontsize=10)
    ax_g.set_title(f"Stance flags of leg {args.leg} (bar = read as planted)", fontsize=11)
    ax_g.tick_params(labelsize=9)

    # Blocked window, "future" cover and cursor on both time panels
    covers, cursors = [], []
    for ax in (ax_f, ax_g):
        if blocked_on:
            ax.axvspan(args.blocked_start, args.blocked_end, color=BLOCK_COLOR, alpha=0.15, lw=0, zorder=0)
        cover = Rectangle((0.0, 0.0), 100.0, 1.0, transform=ax.get_xaxis_transform(), fc="white", alpha=0.9, lw=0,
                          zorder=8)
        ax.add_patch(cover)
        covers.append(cover)
        cursors.append(ax.axvline(0.0, color="black", lw=1.5, zorder=9))

    fig.text(0.5, 0.012,
             "Unmodeled leg dynamics add a force pulse during the swing-down; when it crosses the threshold before touchdown, "
             "the raw torque_detector reads the swinging foot as planted.\n"
             "The phase EKF on the same triggers rejects the early one; the fused log-odds avoids it too, "
             "but reads the blocked leg's swing as planted.",
             ha="center", fontsize=9)

    def update(k):
        tk = t[k]
        p, th = x_true[k, :2], x_true[k, 2]
        R = rot(th)
        body.set_xy(p + BODY_POLY @ R.T)
        for i in range(6):
            hip = p + R @ HIPS[i]
            f = p + R @ r[k, i]
            legs[i].set_data([hip[0], f[0]], [hip[1], f[1]])
            planted = bool(oracle[k, i])
            feet[i].set_data([f[0]], [f[1]])
            feet[i].set_markerfacecolor((BLOCK_COLOR if i == leg else "black") if planted else "white")
            side = np.sign(HIPS[i, 1])
            for j, a in enumerate(TORQUE):
                c = f + R @ np.array([(j - 1) * DOT_GAP, side * 0.085])
                dots[i][j].set_data([c[0]], [c[1]])
                dots[i][j].set_markerfacecolor(COLORS[a] if flags[a][k, i] else "white")
            names[i].set_position(f + R @ np.array([0.0, side * 0.15]))
            halos[i].set_center(f)
            halos[i].set_visible(any(flags[a][k, i] and not planted for a in TORQUE))
        ax_xy.set_xlim(p[0] - VIEW_HALF, p[0] + VIEW_HALF)
        ax_xy.set_ylim(p[1] - VIEW_HALF, p[1] + VIEW_HALF)
        for j, ln in enumerate(trail):
            a, b = k - (trail_n - j) * trail_gap, k - (trail_n - j - 1) * trail_gap
            if a >= 0:
                ln.set_data(x_true[a:b + 1, 0], x_true[a:b + 1, 1])
                ln.set_alpha(0.05 + 0.6 * (j + 1) / trail_n)
            else:
                ln.set_data([], [])
        in_block = blocked_on and args.blocked_start <= tk < args.blocked_end
        wrong = [(a, LEG_NAMES[i]) for a in TORQUE for i in range(6) if flags[a][k, i] and not oracle[k, i]]
        line1 = f"t = {tk:.2f} s" + ("   BLOCKED WINDOW" if in_block else "")
        line2 = f"{args.leg}: " + ("planted" if oracle[k, leg] else "swinging")
        status.set_text(line1 + "\n" + line2)
        status.set_color("red" if in_block else "black")
        lines = [f"{a} reads {l} (swinging) as planted" for a, l in wrong]
        if lines:
            status.set_text(line1 + "\n" + line2 + "\n" + "\n".join(lines))
        ax_f.set_xlim(tk - PAST, tk + FUTURE)
        for cover, cur in zip(covers, cursors):
            cover.set_x(tk)
            cur.set_xdata([tk, tk])
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
