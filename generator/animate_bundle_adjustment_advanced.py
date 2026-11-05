'''
Animated version of bundle_adjustment_advanced.py's Local-BA-only run
(docs/optimization/bundle_adjustment.md, section 13.6): a camera moves along
a 90-degree arc through a landmark corridor, and every new keyframe k = 2..49
triggers one Local BA solve over a bounded window picked from the covisibility
graph (run_incremental_local_ba with use_global_ba=False).

Panels:
  (a) top-down scene (x-y): the ground-truth keyframe arc and landmarks, the
      Local-BA estimate of every keyframe so far, and the current window drawn
      at ground-truth positions: new keyframe red, other active keyframes
      blue, fixed anchors orange, active landmarks green;
  (b) the same window magnified (illustrative: the view follows the window and
      is not to the scale of (a)); the estimate of each shown keyframe is the
      purple cross;
  (c) window size vs map size per frame, drawn progressively: keyframes
      (active, fixed anchors, whole map so far) and landmarks (active, whole
      triangulated map).

Message: the cost of one Local BA call is set by the window (a handful of
active keyframes and a few dozen active landmarks), which stays bounded, while
the map it lives in keeps growing. NO accuracy claim is made, and no solve
time is shown (it is machine-dependent). Whether a periodic Global BA pass
pays off is not shown either: the doc's 20-seed sweep has Global BA winning 11
of 20 seeds, so the seed-0 run is not representative.

Data source: use_numpy/bundle_adjustment_advanced.py at its defaults (seed 0).
Nothing is re-implemented. run_incremental_local_ba is called once, unmodified,
with two recording hooks monkeypatched in this file only (restored afterwards,
as generator/make_figures.py does for build_active_window):
  - build_active_window records, per call, (k, active keyframes, fixed
    keyframes, active landmarks, number of landmarks triangulated so far);
  - pose_errors_dict (called once at the end of each keyframe step k >= 2)
    snapshots every keyframe's estimated position after that step's solve.
Checks asserted at run time: the hooked run's final trailing RMSE equals that
of an unhooked run (and the doc's 2.1359 m), the last snapshot equals the
returned T_est, and there is one window and one snapshot per keyframe 2..49.

Illustrative items (labelled on the frames): panel (b)'s magnification and its
moving view; roles are drawn at ground-truth positions, not estimated ones.
The window of step k is the one built before that step's solve; the estimate
drawn for step k is the one after it.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse
import contextlib
import io

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D

from state_estimation_simulations.use_numpy import bundle_adjustment_advanced as m

K_CAM = (800.0, 800.0, 320.0, 240.0)   # the script's defaults: 640x480 image, f = 800 px
N_KF, RADIUS, ARC_DEG = 50, 15.0, 90.0
LM_PER_KF, LATERAL, VERTICAL, VIEW_RANGE = 8, 2.0, 1.0, 6.0
PIXEL_NOISE, MIN_OBS, MIN_SHARED, MAX_WINDOW = 1.0, 3, 2, 6
GLOBAL_INTERVAL, GN_TOL, GN_ITERS, FRONTEND_NOISE, LOOP_GAP = 8, 1e-6, 15, 0.02, 20
DOC_FINAL_RMSE = 2.1359   # docs/optimization/bundle_adjustment.md section 13.6, Local-only, seed 0

RED, BLUE, ORANGE, GREEN, PURPLE = "#d1495b", "#2e86de", "#f4a261", "#2a9d8f", "#8e6bbf"
GRAY = "#9a9a9a"


def run_recorded(seed):
    """Builds the default scene, runs run_incremental_local_ba (Local only)
    once unhooked and once with the recording hooks, and asserts they agree.
    Returns T_true, P_true, windows, snapshots, T_est, final trailing RMSE."""
    T_true = m.generate_ground_truth_trajectory(N_KF, RADIUS, ARC_DEG)
    P_all = m.generate_landmark_corridor(T_true, LM_PER_KF, LATERAL, VERTICAL, np.random.default_rng(seed),
                                         extend_ahead=VIEW_RANGE)
    P_true, obs = m.build_observations(T_true, P_all, K_CAM, PIXEL_NOISE, VIEW_RANGE, MIN_OBS,
                                       np.random.default_rng(seed))
    by_kf = m.group_observations_by_keyframe(obs, N_KF)

    def run():
        with contextlib.redirect_stdout(io.StringIO()):
            return m.run_incremental_local_ba(T_true, by_kf, K_CAM, FRONTEND_NOISE, MIN_OBS, MIN_SHARED, MAX_WINDOW,
                                              GLOBAL_INTERVAL, False, GN_TOL, GN_ITERS,
                                              np.random.default_rng(seed), LOOP_GAP)

    _, _, hist_plain = run()

    windows, snaps = [], []
    orig_window, orig_errors = m.build_active_window, m.pose_errors_dict

    def window_recorder(k, shared_count, landmarks_by_keyframe, observers_by_landmark, P_est, *rest):
        active, fixed, points = orig_window(k, shared_count, landmarks_by_keyframe, observers_by_landmark, P_est, *rest)
        windows.append(dict(k=k, active=list(active), fixed=sorted(fixed), points=list(points), n_map_pts=len(P_est)))
        return active, fixed, points

    def errors_recorder(T_true_, T_est, keyframe_ids):
        snaps.append({i: T_est[i][0:3, 3].copy() for i in T_est})
        return orig_errors(T_true_, T_est, keyframe_ids)

    m.build_active_window, m.pose_errors_dict = window_recorder, errors_recorder
    try:
        T_est, _, hist = run()
    finally:
        m.build_active_window, m.pose_errors_dict = orig_window, orig_errors

    rmse, rmse_plain = hist["traj_rms_pos_err"][-1], hist_plain["traj_rms_pos_err"][-1]
    assert np.isclose(rmse, rmse_plain, rtol=1e-9, atol=0.0), (rmse, rmse_plain)
    if seed == 0:
        assert round(rmse, 4) == DOC_FINAL_RMSE, rmse
    assert len(windows) == len(snaps) == N_KF - 2
    assert [w["k"] for w in windows] == list(range(2, N_KF))
    assert all(np.allclose(snaps[-1][i], T_est[i][0:3, 3]) for i in T_est)
    return T_true, P_true, windows, snaps, T_est, rmse


def view_limits(pts, min_half=1.5, pad=0.4):
    """Square axis limits around the points."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c, half = (lo + hi) / 2.0, max(float((hi - lo).max()) / 2.0 + pad, min_half)
    return (c[0] - half, c[0] + half), (c[1] - half, c[1] + half)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hold", type=int, default=6, help="frames to hold on the last keyframe")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--dpi", type=int, default=80)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--png-frames", type=int, nargs="*", default=[], help="also save these frame indices as PNG next to --out")
    a = p.parse_args()

    T_true, P_true, windows, snaps, T_est, rmse = run_recorded(a.seed)
    pos = np.array([T[0:3, 3] for T in T_true])
    n_steps = len(windows)
    frames = list(range(n_steps)) + [n_steps - 1] * a.hold

    # fixed scene limits: truth arc, landmarks, and every estimate ever drawn
    all_est = np.vstack([np.array([s[i][:2] for i in s]) for s in snaps])
    pts_all = np.vstack([pos[:, :2], P_true[:, :2], all_est])
    sx = (pts_all[:, 0].min() - 0.6, pts_all[:, 0].max() + 0.6)
    sy = (pts_all[:, 1].min() - 0.6, pts_all[:, 1].max() + 0.6)

    k_arr = np.array([w["k"] for w in windows])
    n_act = np.array([len(w["active"]) for w in windows])
    n_fix = np.array([len(w["fixed"]) for w in windows])
    n_pts = np.array([len(w["points"]) for w in windows])
    n_map_kf = k_arr + 1                                   # keyframes revealed so far
    n_map_pts = np.array([w["n_map_pts"] for w in windows])  # landmarks triangulated so far

    fig = plt.figure(figsize=(14.4, 6.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.8, 1.0, 1.0], left=0.045, right=0.985, top=0.855, bottom=0.215,
                          wspace=0.24, hspace=0.42)
    ax_s, ax_z = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
    ax_k, ax_p = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
    title = fig.suptitle("", fontsize=13, fontweight="bold", y=0.975)
    handles = [
        Line2D([], [], color="k", lw=1.2, marker="o", ms=3, label="ground-truth keyframes"),
        Line2D([], [], color=PURPLE, lw=1.4, label="Local-BA estimate (line; x in b)"),
        Line2D([], [], color=RED, marker="*", ls="", ms=11, label="new keyframe"),
        Line2D([], [], color=BLUE, marker="o", ls="", ms=6, label="active keyframes"),
        Line2D([], [], color=ORANGE, marker="s", ls="", ms=6, label="fixed anchors"),
        Line2D([], [], color=GREEN, marker=".", ls="", ms=9, label="active landmarks"),
        Line2D([], [], color=GRAY, marker=".", ls="", ms=6, label="other landmarks (truth)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.085))
    fig.text(0.5, 0.025,
             "Each new keyframe triggers one Local BA solve over a small window (blue = optimized, orange = held fixed). "
             "The window stays bounded\nwhile the map grows. Panel (b) is a moving magnification (illustrative). "
             "No accuracy or timing claim: Local BA only, seed %d." % a.seed,
             ha="center", va="bottom", fontsize=10)

    def draw_roles(ax, w, k, big):
        ms = 7 if big else 5
        pts = P_true[w["points"]]
        ax.scatter(pts[:, 0], pts[:, 1], s=14 if big else 8, color=GREEN, zorder=3)
        fx = pos[w["fixed"]]
        ax.plot(fx[:, 0], fx[:, 1], "s", color=ORANGE, ms=ms, mec="k", mew=0.4, zorder=4)
        act = [i for i in w["active"] if i != k]
        if act:
            ac = pos[act]
            ax.plot(ac[:, 0], ac[:, 1], "o", color=BLUE, ms=ms, mec="k", mew=0.4, zorder=5)
        ax.plot(pos[k, 0], pos[k, 1], "*", color=RED, ms=13 if big else 10, mec="k", mew=0.4, zorder=6)

    def update(f):
        w, snap = windows[f], snaps[f]
        k = w["k"]
        est = np.array([snap[i][:2] for i in range(k + 1)])

        ax_s.clear()
        ax_s.plot(pos[:, 0], pos[:, 1], "-", color="#dddddd", lw=1.0, zorder=1)
        ax_s.scatter(P_true[:, 0], P_true[:, 1], s=4, color=GRAY, alpha=0.4, zorder=1)
        ax_s.plot(pos[:k + 1, 0], pos[:k + 1, 1], "-o", color="k", lw=0.8, ms=2.2, zorder=2)
        ax_s.plot(est[:, 0], est[:, 1], "-", color=PURPLE, lw=1.4, zorder=2.5)
        draw_roles(ax_s, w, k, big=False)
        ax_s.set_xlim(sx), ax_s.set_ylim(sy), ax_s.set_aspect("equal")
        ax_s.set_xlabel("x (m)"), ax_s.set_ylabel("y (m)")
        ax_s.set_title("(a) Whole scene, top-down")

        ax_z.clear()
        shown = sorted(set(w["active"]) | set(w["fixed"]))
        zx, zy = view_limits(np.vstack([pos[shown, :2], P_true[w["points"], :2],
                                        np.array([snap[i][:2] for i in shown])]))
        ax_z.scatter(P_true[:, 0], P_true[:, 1], s=8, color=GRAY, alpha=0.4, zorder=1)
        ax_z.plot(pos[:k + 1, 0], pos[:k + 1, 1], "-", color="k", lw=0.8, zorder=2)
        ax_z.plot(est[:, 0], est[:, 1], "-", color=PURPLE, lw=1.0, alpha=0.6, zorder=2.5)
        draw_roles(ax_z, w, k, big=True)
        sh_est = np.array([snap[i][:2] for i in shown])
        ax_z.plot(sh_est[:, 0], sh_est[:, 1], "x", color=PURPLE, ms=5, mew=1.4, zorder=7)
        ax_z.set_xlim(zx), ax_z.set_ylim(zy), ax_z.set_aspect("equal")
        ax_z.set_xlabel("x (m)"), ax_z.set_ylabel("y (m)")
        ax_z.set_title("(b) Magnified window (illustrative zoom)")

        ax_k.clear()
        ax_k.plot(k_arr[:f + 1], n_map_kf[:f + 1], "-", color="k", lw=1.6, marker="o", ms=2.5, label="whole map")
        ax_k.plot(k_arr[:f + 1], n_fix[:f + 1], "-", color=ORANGE, lw=1.6, marker="o", ms=2.5, label="fixed anchors")
        ax_k.plot(k_arr[:f + 1], n_act[:f + 1], "-", color=BLUE, lw=1.6, marker="o", ms=2.5, label="active")
        ax_k.set_xlim(0, N_KF - 1), ax_k.set_ylim(0, n_map_kf.max() * 1.08)
        ax_k.set_ylabel("keyframes")
        ax_k.set_title("(c) Window size vs map size")
        ax_k.legend(fontsize=8.5, loc="upper left", frameon=False)
        ax_k.tick_params(labelbottom=False)

        ax_p.clear()
        ax_p.plot(k_arr[:f + 1], n_map_pts[:f + 1], "-", color="k", lw=1.6, marker="o", ms=2.5, label="whole map (triangulated)")
        ax_p.plot(k_arr[:f + 1], n_pts[:f + 1], "-", color=GREEN, lw=1.6, marker="o", ms=2.5, label="active")
        ax_p.set_xlim(0, N_KF - 1), ax_p.set_ylim(0, n_map_pts.max() * 1.08)
        ax_p.set_xlabel("new keyframe index k"), ax_p.set_ylabel("landmarks")
        ax_p.legend(fontsize=8.5, loc="upper left", frameon=False)

        title.set_text(f"Keyframe {k} arrives:  window = {n_act[f]} active + {n_fix[f]} fixed keyframes, "
                       f"{n_pts[f]} active landmarks   |   map = {n_map_kf[f]} keyframes, {n_map_pts[f]} landmarks")

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 // a.fps, blit=False)
    anim.save(a.out, writer=animation.PillowWriter(fps=a.fps), dpi=a.dpi)
    for fi in a.png_frames:
        update(frames[fi])
        base = a.out.rsplit(".", 1)[0]
        fig.savefig(f"{base}_frame{fi:03d}.png", dpi=a.dpi)

    print(f"Saved {a.out}: {len(frames)} frames ({n_steps} keyframes 2..{N_KF - 1} + {a.hold} hold)")
    print(f"Window (before each solve): active keyframes {n_act.min()}-{n_act.max()}, fixed anchors {n_fix.min()}-{n_fix.max()}, "
          f"active landmarks {n_pts.min()}-{n_pts.max()}")
    print(f"Map at the last step: {n_map_kf[-1]} keyframes, {n_map_pts[-1]} triangulated landmarks "
          f"(last window: {n_act[-1]} + {n_fix[-1]} keyframes, {n_pts[-1]} landmarks)")
    print(f"Hooked run final trailing RMSE {rmse:.4f} m == unhooked run (asserted; doc: {DOC_FINAL_RMSE} m)")


if __name__ == "__main__":
    main()
