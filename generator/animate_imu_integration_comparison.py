'''
Animated version of imu_integration_comparison.py, one run at its defaults
(20 s, dt = 0.005 s, gyro noise 0.02 rad/s, body-velocity noise 0.05 m/s):
a body is rotated by the same noisy gyro stream with two integrators, a naive
one (Euler angles treated as a flat vector, euler += omega * dt) and the SO(3)
Exp-map one (R <- R Exp(omega * dt)).

Three columns, drawn progressively:
  1. left, 3D: the naive estimator. A box with its body axes at the origin,
     the truth (wide pale gray box and pale axes) and the estimate (thin dashed
     colored box and axes). Only the orientation is drawn; the position is not;
  2. middle, 3D: the same for the Exp-map estimator;
  3. right: the rotation error (geodesic angle, deg, log scale) and the
     position error (m, log scale) over time, naive (red) vs Exp-map (blue).

The script uses TWO ground truths, one per estimator, each built from the
noise-free rates with that estimator's own composition rule (docs
optimization/imu_preintegration.md section 7.1, "Ground truths"). So each
estimate is drawn against its own truth: the naive estimate against
R_gt_naive (euler_to_R composition), the Exp-map estimate against R_gt_exp
(so3_exp composition). The two truths are therefore not the same attitude
(they differ by 0.45 deg at the end of the run, per the script's docstring), which
is invisible at this scale.

Message (doc docs/optimization/imu_preintegration.md): adding body rates to
Euler angles treats a curved space as flat, so the naive attitude ends tens of
degrees off (about 70.8 deg), while the Exp-map update, fed the same noisy
samples, ends within a fraction of a degree (about 0.47 deg).

Data source: the loop of imu_integration_comparison.run_simulation is copied
here (same RNG order, same updates) because run_simulation returns only the
error histories, not the orientations that are drawn. The copy is checked
in the generator: its four error histories must equal run_simulation's
(np.allclose), else the script stops.

Illustrative (not data): the box (1.0 x 0.6 x 0.3, a stand-in for the body)
and the axis length (1.0) are chosen shapes; the frames are subsampled (every
--frame-step simulation steps, default 25 = 0.125 s, 161 frames); the camera
is fixed. Only the orientation is drawn, so the position error appears in the
right-hand plot only. Everything else is the script's own output.

Seed 0 by default (the script's own default seed).

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from state_estimation_simulations.use_numpy.imu_integration_comparison import euler_to_R, run_simulation
from state_estimation_simulations.use_numpy.lie_utils import so3_exp, rotation_geodesic_error
from state_estimation_simulations.utils import true_body_rates

DURATION, DT, GYRO_STD, VEL_STD = 20.0, 0.005, 0.02, 0.05      # the script's defaults
BOX_HALF = np.array([0.5, 0.3, 0.15])    # illustrative box (1.0 x 0.6 x 0.3)
AXIS_LEN = 1.0                           # illustrative axis length
AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")


def simulate(duration, dt, gyro_std, vel_std, seed):
    """Copy of run_simulation's loop (same RNG order) that also keeps the orientations."""
    rng = np.random.default_rng(seed)
    n = int(duration / dt)
    Rgn, pgn = np.eye(3), np.zeros(3)
    Rge, pge = np.eye(3), np.zeros(3)
    eul = np.zeros(3)
    pen = np.zeros(3)
    Ree, pee = np.eye(3), np.zeros(3)
    out = dict(Rgn=np.zeros((n, 3, 3)), Ren=np.zeros((n, 3, 3)), Rge=np.zeros((n, 3, 3)), Ree=np.zeros((n, 3, 3)),
               t=np.zeros(n), rn=np.zeros(n), re=np.zeros(n), pn=np.zeros(n), pe=np.zeros(n))
    for k in range(n):
        t = k * dt
        omega, v = true_body_rates(t)
        Rgn = Rgn @ euler_to_R(omega * dt)
        pgn = pgn + Rgn @ v * dt
        Rge = Rge @ so3_exp(omega * dt)
        pge = pge + Rge @ v * dt
        om = omega + rng.normal(0.0, gyro_std, 3)
        vm = v + rng.normal(0.0, vel_std, 3)
        eul = eul + om * dt
        Ren = euler_to_R(eul)
        pen = pen + Ren @ vm * dt
        Ree = Ree @ so3_exp(om * dt)
        pee = pee + Ree @ vm * dt
        out["t"][k] = t
        out["Rgn"][k], out["Ren"][k], out["Rge"][k], out["Ree"][k] = Rgn, Ren, Rge, Ree
        out["rn"][k] = np.degrees(rotation_geodesic_error(Rgn, Ren))
        out["re"][k] = np.degrees(rotation_geodesic_error(Rge, Ree))
        out["pn"][k] = np.linalg.norm(pgn - pen)
        out["pe"][k] = np.linalg.norm(pge - pee)
    return out


def box_segments(R):
    """The 12 edges of the box rotated by R, as one NaN-separated 3D polyline (3 x N)."""
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * BOX_HALF
    c = c @ R.T
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
    pts = []
    for a, b in edges:
        pts += [c[a], c[b], [np.nan] * 3]
    return np.array(pts).T


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (the script's own default)")
    parser.add_argument("--frame-step", type=int, default=25, help="Simulation steps (5 ms) per GIF frame")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.add_argument("--out", type=str, required=True, help="Output GIF path")
    args = parser.parse_args()

    d = simulate(DURATION, DT, GYRO_STD, VEL_STD, args.seed)
    ref = run_simulation(DURATION, DT, GYRO_STD, VEL_STD, args.seed)
    for mine, theirs in zip((d["t"], d["rn"], d["re"], d["pn"], d["pe"]), ref):
        assert np.allclose(mine, theirs), "copied loop does not reproduce run_simulation"
    t, n = d["t"], len(d["t"])
    print(f"seed {args.seed}, {DURATION:.0f} s, dt {DT} s (copied loop reproduces run_simulation: np.allclose)")
    print(f"  final rotation error: naive {d['rn'][-1]:.2f} deg, Exp-map {d['re'][-1]:.2f} deg")
    print(f"  final position error: naive {d['pn'][-1]:.3f} m, Exp-map {d['pe'][-1]:.3f} m")
    print(f"  largest rotation error: naive {d['rn'].max():.2f} deg, Exp-map {d['re'].max():.2f} deg")
    print(f"  truths differ from each other at the end by "
          f"{np.degrees(rotation_geodesic_error(d['Rgn'][-1], d['Rge'][-1])):.2f} deg")

    fig = plt.figure(figsize=(14, 6.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05], hspace=0.5, wspace=0.12)
    fig.subplots_adjust(left=0.01, right=0.975, top=0.86, bottom=0.17)
    fig.suptitle("Integrating a noisy gyro: Euler angles treated as a flat vector vs the SO(3) Exp-map update",
                 fontsize=12.5, fontweight="bold")

    def frame_panel(spec, title):
        ax = fig.add_subplot(spec, projection="3d")
        ax.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-1.0, 1.0))
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=40)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        truth_box, = ax.plot(*box_segments(np.eye(3)), color="0.75", lw=4.5, ls="-")
        est_box, = ax.plot(*box_segments(np.eye(3)), color="k", lw=1.8, ls=(0, (3, 2)), zorder=10)
        truth_axes = [ax.plot([0, 0], [0, 0], [0, 0], color=c, lw=6, ls="-", alpha=0.35)[0] for c in AXIS_COLORS]
        est_axes = [ax.plot([0, 0], [0, 0], [0, 0], color=c, lw=2.2, ls=(0, (2.5, 1.5)), zorder=10)[0] for c in AXIS_COLORS]
        text = ax.text2D(0.5, 0.0, "", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, fontweight="bold")
        return dict(ax=ax, tb=truth_box, eb=est_box, ta=truth_axes, ea=est_axes, text=text)

    pn = frame_panel(gs[:, 0], "Naive: Euler angles, euler += omega dt")
    pe = frame_panel(gs[:, 1], "Exp-map: R <- R Exp(omega dt)")
    for p, color in ((pn, "tab:red"), (pe, "tab:blue")):
        p["eb"].set_color(color)
    # legend for the line styles, below the 3D panels
    fig.text(0.255, 0.115, "wide pale box and axes: truth    thin dashed box and axes: estimate    (axes: x red, y green, z blue)",
             ha="center", fontsize=8.5, color="0.2")

    ax_r = fig.add_subplot(gs[0, 2])
    ax_p = fig.add_subplot(gs[1, 2], sharex=ax_r)
    for ax, kn, ke, ylabel, title in ((ax_r, "rn", "re", "rotation error (deg)", "Rotation error (geodesic angle, log scale)"),
                                      (ax_p, "pn", "pe", "position error (m)", "Position error (log scale)")):
        ax.plot(t, d[kn], color="tab:red", lw=1, alpha=0.2)
        ax.plot(t, d[ke], color="tab:blue", lw=1, alpha=0.2)
        ax.set_yscale("log")
        ax.set(xlim=(0, DURATION), ylabel=ylabel, title=title)
        ax.title.set_fontsize(10)
        ax.grid(True, which="major", alpha=0.3)
    ax_p.set_xlabel("time (s)")
    plt.setp(ax_r.get_xticklabels(), visible=False)
    line_rn, = ax_r.plot([], [], color="tab:red", lw=1.9, label="naive (Euler angles)")
    line_re, = ax_r.plot([], [], color="tab:blue", lw=1.9, label="Exp-map (SO(3))")
    line_pn, = ax_p.plot([], [], color="tab:red", lw=1.9)
    line_pe, = ax_p.plot([], [], color="tab:blue", lw=1.9)
    ax_r.legend(loc="lower right", fontsize=8)
    status = fig.text(0.985, 0.915, "", ha="right", fontsize=10, fontweight="bold")

    fig.text(0.5, 0.012,
             f"Seed {args.seed}, one run, same noisy gyro stream for both. Each estimate is drawn against its own truth (the script scores each "
             "estimator against the noise-free run of its own update rule). Only orientation is drawn.\n"
             f"After 20 s the naive attitude is {d['rn'][-1]:.1f} deg off; the Exp-map attitude, fed the same samples, is {d['re'][-1]:.2f} deg off.",
             ha="center", fontsize=9)

    def set_frame(p, R_truth, R_est, err):
        p["tb"].set_data_3d(*box_segments(R_truth))
        p["eb"].set_data_3d(*box_segments(R_est))
        for lines, R in ((p["ta"], R_truth), (p["ea"], R_est)):
            for i, line in enumerate(lines):
                tip = AXIS_LEN * R[:, i]
                line.set_data_3d([0, tip[0]], [0, tip[1]], [0, tip[2]])
        p["text"].set_text(f"rotation error {err:.2f} deg")

    def update(frame):
        k = min(frame * args.frame_step, n - 1)
        set_frame(pn, d["Rgn"][k], d["Ren"][k], d["rn"][k])
        set_frame(pe, d["Rge"][k], d["Ree"][k], d["re"][k])
        sl = slice(0, k + 1)
        line_rn.set_data(t[sl], d["rn"][sl])
        line_re.set_data(t[sl], d["re"][sl])
        line_pn.set_data(t[sl], d["pn"][sl])
        line_pe.set_data(t[sl], d["pe"][sl])
        status.set_text(f"t = {t[k]:.1f} s")
        return []

    n_frames = -(-(n - 1) // args.frame_step) + 1   # last frame is the last step
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {n_frames} frames to {args.out}")


if __name__ == "__main__":
    main()
