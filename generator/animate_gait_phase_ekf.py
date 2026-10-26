'''
Animated version of gait_phase_ekf.py, one run: a planar hexapod walks a tripod
gait whose feet land at the commanded touchdown time plus an unknown offset,
and one leg is blocked for a time window so its feet land late. Four ways of
deciding which feet are planted (schedule, detector, cpg, phase_ekf) feed the
same body EKF; what separates them is the stance flags.

Three panels, drawn progressively:
  1. top view (left): the true hexapod (body, six legs, feet at their
     simulated positions; filled = truly planted, hollow = in swing; the
     blocked leg in orange) with a fading trail of the body. The camera
     follows the body, the grid is fixed in the world;
  2. gait diagram of the blocked leg (right, top), scrolling with time: the
     stance flags of oracle (the truth), schedule, detector, cpg and phase_ekf.
     Gold marks ticks where a swinging foot is read as planted; a triangle
     marks each detector trigger and a pink X each false trigger; the blocked window is shaded;
  3. touchdown offset of each step of that leg (right, bottom): the true
     offset tau, the true systematic lag, the cpg and phase_ekf lag estimates
     (before each update) and the phase_ekf +-2 sigma band.

Message: the schedule reads the blocked leg as planted while it is still
swinging (late landing); the detector does so after every false trigger;
cpg and phase_ekf reject early triggers by their learned lag, and phase_ekf
keeps its lag estimate through the blocked steps by not learning from them.

Everything is the simulation's own simulate_trial output (same rng stream as
gait_phase_ekf.main's single run); no filter is re-run and no Monte Carlo is
needed. The simulation's flags are reused, except that --n-mc-trials has no
effect here. Detector triggers are info["triggers"] (every trigger of each
step); the ones listed in info["false_ticks"] are drawn as X, the others as
triangles. Default window: blocked-start - 3 s to blocked-end + 5.5 s.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import MultipleLocator

from state_estimation_simulations.use_numpy import gait_phase_ekf as gp
from state_estimation_simulations.use_numpy.gait_phase_ekf import LEG_NAMES
from state_estimation_simulations.use_numpy.hexapod_gait_ekf import NOMINAL_FOOT_OFFSETS, rot

COLORS = {"schedule": "tab:blue", "detector": "tab:red", "cpg": "tab:purple", "phase_ekf": "tab:green",
          "oracle": "black"}
ROW_ORDER = ("oracle", "schedule", "detector", "cpg", "phase_ekf")  # top to bottom
ROW_LABELS = {"oracle": "true (oracle)", "schedule": "schedule", "detector": "detector", "cpg": "cpg",
              "phase_ekf": "phase_ekf"}
BLOCK_COLOR = "tab:orange"
FALSE_COLOR = "deeppink"
BODY_POLY = np.array([[-0.17, -0.10], [0.17, -0.10], [0.24, 0.0], [0.17, 0.10], [-0.17, 0.10]])
HIPS = np.column_stack([0.6 * NOMINAL_FOOT_OFFSETS[:, 0], 0.09 * np.sign(NOMINAL_FOOT_OFFSETS[:, 1])])
VIEW_HALF = 0.6  # half-width of the following camera (m)
PAST, FUTURE = 5.0, 1.5  # scrolling gait-diagram window around the cursor (s)


def main():
    parser = gp.build_parser()
    parser.description = __doc__
    # The simulation already has --out (figure path): reuse it as the GIF path
    parser.set_defaults(out="gait_phase_ekf.gif")
    for action in parser._actions:
        if action.dest == "out":
            action.help = "Output GIF path"
    parser.add_argument("--t-start", type=float, default=None, help="First animated time (s); default blocked-start - 3")
    parser.add_argument("--t-end", type=float, default=None, help="Last animated time (s); default blocked-end + 5.5")
    parser.add_argument("--stride", type=int, default=4, help="Filter ticks per GIF frame")
    parser.add_argument("--fps", type=int, default=12, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    args = parser.parse_args()

    # Same rng stream as gait_phase_ekf.main's single run
    rng = np.random.default_rng(args.seed)
    x_true, _, r, _, flags, info = gp.simulate_trial(args, rng)
    dt = args.dt
    n_pts = len(x_true)
    t = np.arange(n_pts) * dt
    leg = LEG_NAMES.index(args.blocked_leg)
    blocked_on = args.blocked_delay > 0
    t0 = max(0.0, args.blocked_start - 3.0) if args.t_start is None else args.t_start
    t1 = min(t[-1], args.blocked_end + 5.5) if args.t_end is None else min(t[-1], args.t_end)
    k0, k1 = int(round(t0 / dt)), int(round(t1 / dt))
    if not 0 <= k0 < k1 < n_pts:
        raise SystemExit("need 0 <= t-start < t-end <= duration")
    frames = list(range(k0, k1 + 1, args.stride))

    # Per-step data of the blocked leg, at the true touchdown time
    steps = info["steps"]
    sel = np.array([s[0] == leg for s in steps])
    td_t = info["td"][sel] * dt
    tau, lag = 1e3 * info["tau"][sel], 1e3 * info["lag"][sel]
    est = {a: 1e3 * info["lag_est"][a][sel] for a in ("cpg", "phase_ekf")}
    band = 2e3 * info["ekf_std"][sel]
    blk = info["is_blocked"][sel]
    # Every detector trigger of this leg's steps, split into false and real ones
    false_k, real_k = [], []
    for trig, f, s in zip(info["triggers"], info["false_ticks"], steps):
        if s[0] == leg:
            false_k += [k for k in trig if k == f]
            real_k += [k for k in trig if k != f]

    oracle = flags["oracle"]
    print(f"{args.blocked_leg} blocked for t in [{args.blocked_start}, {args.blocked_end}) s (delay {args.blocked_delay} s), "
          f"seed {args.seed}; animating t in [{t0:.1f}, {t1:.1f}] s")
    win = slice(k0, k1 + 1)
    for a in gp.APPROACHES:
        bad = int((flags[a][win, leg] & ~oracle[win, leg]).sum())
        print(f"  {a:<10s} swinging {args.blocked_leg} read as planted: {bad} ticks in the window")

    # --- Figure layout ---
    fig = plt.figure(figsize=(12, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.24)
    gs_r = gs[1].subgridspec(3, 1, height_ratios=[3.0, 0.7, 2.5], hspace=0.12)
    ax_xy = fig.add_subplot(gs[0])
    ax_g = fig.add_subplot(gs_r[0])
    ax_gl = fig.add_subplot(gs_r[1])
    ax_gl.axis("off")
    ax_o = fig.add_subplot(gs_r[2], sharex=ax_g)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.89, bottom=0.2)
    fig.suptitle(f"Hexapod tripod gait: touchdown timing with leg {args.blocked_leg} blocked", fontsize=14,
                 fontweight="bold")

    # --- Top view ---
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.xaxis.set_major_locator(MultipleLocator(0.25))
    ax_xy.yaxis.set_major_locator(MultipleLocator(0.25))
    ax_xy.grid(alpha=0.3)
    ax_xy.tick_params(labelsize=9)
    ax_xy.set_xlabel("world x (m); the camera follows the body", fontsize=10)
    ax_xy.set_ylabel("world y (m)", fontsize=10)
    ax_xy.set_title("Top view: the true robot", fontsize=11)
    trail_n, trail_gap = 20, 10
    trail = [ax_xy.plot([], [], color="black", lw=3, solid_capstyle="round")[0] for _ in range(trail_n)]
    body = Polygon(BODY_POLY, closed=True, fc="#dfe7f2", ec="#4a5a70", lw=1.8, zorder=3)
    ax_xy.add_patch(body)
    legs, feet, names = [], [], []
    for i in range(6):
        isb = i == leg
        legs.append(ax_xy.plot([], [], color=BLOCK_COLOR if isb else "#5b6b7f", lw=3.2 if isb else 2.2, zorder=2)[0])
        feet.append(ax_xy.plot([], [], "o", ms=13 if isb else 11, mec="black", mew=1.8 if isb else 1.4, zorder=5)[0])
        names.append(ax_xy.text(0, 0, args.blocked_leg + " (blocked)" if isb else LEG_NAMES[i], ha="center",
                                va="center", fontsize=10 if isb else 8.5, fontweight="bold" if isb else "normal",
                                color="darkorange" if isb else "#33404f", zorder=6))
    status = ax_xy.text(0.02, 0.98, "", transform=ax_xy.transAxes, ha="left", va="top", fontsize=10.5,
                        fontweight="bold", linespacing=1.4, zorder=7,
                        bbox=dict(fc="white", ec="none", alpha=0.8, pad=2))
    handles = [Line2D([], [], marker="o", ls="", ms=10, mfc="black", mec="black", label="foot planted (truly in stance)"),
               Line2D([], [], marker="o", ls="", ms=10, mfc="white", mec="black", label="foot in swing"),
               Line2D([], [], marker="o", ls="", ms=10, mfc=BLOCK_COLOR, mec="black",
                      label=f"blocked leg {args.blocked_leg}, planted"),
               Line2D([], [], color="black", lw=2.5, label="body trail (last 4 s)")]
    ax_xy.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9, ncol=2, frameon=False)

    # --- Gait diagram of the blocked leg ---
    ypos = {a: len(ROW_ORDER) - 1 - j for j, a in enumerate(ROW_ORDER)}
    for a in ROW_ORDER:
        y = ypos[a]
        ax_g.fill_between(t, y - 0.35, y + 0.35, where=flags[a][:, leg], step="post", color=COLORS[a], alpha=0.65, lw=0)
        bad = flags[a][:, leg] & ~oracle[:, leg]
        ax_g.fill_between(t, y - 0.35, y + 0.35, where=bad, step="post", color="gold", lw=0, zorder=3)
    y_det = ypos["detector"] + 0.5
    ax_g.plot(t[real_k], [y_det] * len(real_k), "v", color=COLORS["detector"], ms=8, mec="black", mew=0.8, ls="",
              zorder=4)
    ax_g.plot(t[false_k], [y_det] * len(false_k), "X", color=FALSE_COLOR, ms=9, mec="black", mew=0.8, ls="", zorder=4)
    ax_g.set_yticks([ypos[a] for a in ROW_ORDER], [ROW_LABELS[a] for a in ROW_ORDER], fontsize=10)
    ax_g.set_ylim(-0.6, len(ROW_ORDER) - 0.3)
    ax_g.set_title(f"Stance flags of leg {args.blocked_leg} (bar = read as planted)", fontsize=11)
    plt.setp(ax_g.get_xticklabels(), visible=False)
    ax_gl.legend(handles=[Line2D([], [], color="gold", lw=6, label="swinging foot read as planted"),
                          Line2D([], [], marker="v", ls="", ms=8, mfc=COLORS["detector"], mec="black",
                                 label="detector trigger"),
                          Line2D([], [], marker="X", ls="", ms=9, mfc=FALSE_COLOR, mec="black", label="false trigger"),
                          Line2D([], [], color=BLOCK_COLOR, lw=6, alpha=0.3, label="blocked window")],
                 loc="center", ncol=2, fontsize=9.5, frameon=False, borderaxespad=0.0)

    # --- Touchdown offsets of that leg's steps ---
    ax_o.plot(td_t, lag, color="black", lw=1.6, label="true systematic lag")
    ax_o.plot(td_t[~blk], tau[~blk], "o", color="gray", ms=5, label="true offset tau")
    if blocked_on:
        ax_o.plot(td_t[blk], tau[blk], "o", color=BLOCK_COLOR, ms=6, mec="black", label="tau, blocked step")
    for a in ("cpg", "phase_ekf"):
        ax_o.plot(td_t, est[a], "-s", color=COLORS[a], lw=1.8, ms=4, label=f"{a} lag estimate")
    ax_o.fill_between(td_t, est["phase_ekf"] - band, est["phase_ekf"] + band, color=COLORS["phase_ekf"], alpha=0.2,
                      lw=0, label="phase_ekf $\\pm 2\\sigma$")
    ax_o.set_ylabel("touchdown offset (ms)", fontsize=10)
    ax_o.set_xlabel("time (s)", fontsize=10)
    ax_o.set_title(f"Touchdown offset of each {args.blocked_leg} step", fontsize=11)
    ax_o.tick_params(labelsize=9)
    ax_o.grid(alpha=0.3)
    ax_o.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, fontsize=9, frameon=False)
    in_win = (td_t >= t0 - PAST) & (td_t <= t1 + FUTURE)
    lo = min(tau[in_win].min(), (est["phase_ekf"] - band)[in_win].min(), 0.0)
    hi = max(tau[in_win].max(), (est["phase_ekf"] + band)[in_win].max())
    ax_o.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))

    # Blocked window, "future" cover and cursor on both time panels
    covers, cursors = [], []
    for ax in (ax_g, ax_o):
        if blocked_on:
            ax.axvspan(args.blocked_start, args.blocked_end, color=BLOCK_COLOR, alpha=0.15, lw=0, zorder=0)
        cover = Rectangle((0.0, 0.0), 100.0, 1.0, transform=ax.get_xaxis_transform(), fc="white", alpha=0.9, lw=0,
                          zorder=8)
        ax.add_patch(cover)
        covers.append(cover)
        cursors.append(ax.axvline(0.0, color="black", lw=1.5, zorder=9))

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
            lab = f + R @ np.array([0.0, np.sign(HIPS[i, 1]) * (0.13 if i == leg else 0.075)])
            names[i].set_position(lab)
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
        wrong = [a for a in gp.APPROACHES if flags[a][k, leg] and not oracle[k, leg]]
        line1 = f"t = {tk:.1f} s" + ("   BLOCKED WINDOW" if in_block else "")
        line2 = f"{args.blocked_leg}: " + ("planted" if oracle[k, leg] else "swinging")
        if wrong:
            line2 += "; read as planted by " + ", ".join(wrong)
        status.set_text(line1 + "\n" + line2)
        status.set_color("red" if in_block else "black")
        ax_g.set_xlim(tk - PAST, tk + FUTURE)
        for cover, cur in zip(covers, cursors):
            cover.set_x(tk)
            cur.set_xdata([tk, tk])
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
