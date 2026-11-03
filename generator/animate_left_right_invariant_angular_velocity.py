'''
Animated picture for docs/filtering/left_right_invariant.md §3.1 ("Picture
(gyro vs. ground watcher)"): body-frame and spatial angular velocity, and
which frame change each one ignores.

A rigid body spins with a constant body angular velocity w_b:
R(t) = R_init Exp(w_b t), so dR/dt = R hat(w_b). From R and dR/dt:

    body-frame (gyro)        hat(omega_b) = R^T dR/dt
    spatial (ground watcher) hat(omega_s) = dR/dt R^T

  - Left panel, rename the map axes: R -> R0 R (left-multiplication). Only
    the gray map axes turn; the body keeps spinning as before.
  - Right panel, re-bolt the body axes: R -> R R0 (right-multiplication).
    The colored sensor axes turn on the body; the body's motion is the same.

R0 = Exp(s a) is one fixed relabeling per frame, with s going 0 -> 1 -> 0
over the loop, so dR/dt relabels the same way (R0 dR/dt, dR/dt R0). Both
velocities are recomputed every frame from the relabeled matrices: omega_b
stays constant on the left and omega_s on the right (asserted to 1e-12); the
other one changes. No simulation is run.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from state_estimation_simulations.use_numpy.lie_utils import so3_exp
from state_estimation_simulations.utils import skew

KEEP_C, CHANGE_C = "#2a9d8f", "#d1495b"
AXIS_C = ("#d1495b", "#2a9d8f", "#2a7ab9")  # x, y, z of the sensor (body) axes


def vee(W):
    """3-vector of a skew-symmetric matrix (inverse of skew)."""
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def fmt(v):
    return "(" + ", ".join(f"{c:+.2f}" for c in v) + ")"


# a flat box, the spinning body, in body coordinates
BOX = np.array([[x, y, z] for x in (-0.6, 0.6) for y in (-0.35, 0.35) for z in (-0.15, 0.15)])
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(BOX[i] != BOX[j]) == 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=120, help="Frames per loop (s goes 0 -> 1 -> 0)")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=70, help="GIF resolution")
    parser.add_argument("--out", type=str, default="assets/left_right_invariant_angular_velocity.gif",
                        help="Output GIF path")
    args = parser.parse_args()

    w_b = np.array([0.0, 0.5, 1.2])                      # constant body angular velocity (rad/s)
    R_init = so3_exp(np.array([0.3, -0.2, 0.4]))
    a = np.radians(60.0) * np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)   # relabeling axis-angle at s = 1
    dt_frame = 1.0 / args.fps
    s_vals = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(args.frames) / args.frames))

    fig = plt.figure(figsize=(11.0, 6.6))
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.3, wspace=0.0)
    fig.suptitle("Gyro vs. ground watcher: each angular velocity ignores one kind of relabeling", fontsize=12)
    titles = ("Rename the map axes:  R → R₀R", "Re-bolt the body axes:  R → RR₀")
    panels = []
    for col, title in enumerate(titles):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), xticks=[], yticks=[], zticks=[])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-60)
        ax.set_title(title, fontsize=10.5)
        ax.text(-0.9, -0.9, -1.15, "map axes", color="0.45", fontsize=8.5)
        rows = [fig.text(0.04 + 0.5 * col, 0.25 - 0.04 * r, "", fontsize=9.5, family="monospace") for r in range(3)]
        panels.append(dict(ax=ax, rows=rows, dyn=[]))
    caption = ("A box spins at a constant body angular velocity ωb. ωb is what a gyro bolted to the box reads, in the "
               "sensor axes (colored arrows: x red, y green, z blue); ωs is how fast an observer on the ground sees it "
               "spin, in the map axes (gray). Left: only the map axes turn, so ωb is unchanged and ωs, written in the "
               "new map axes, changes. Right: only the sensor axes turn on the box, so ωs is unchanged and ωb changes. "
               "Recomputed from the matrices every frame (green = unchanged, red = changes).")
    fig.text(0.5, 0.015, textwrap.fill(caption, 150), ha="center", va="bottom", fontsize=8.5, color="#444444")

    def arrows(ax, o, M, colors, length, dyn, lw=2.0):
        for k in range(3):
            v = length * M[:, k]
            dyn.append(ax.quiver(*o, *v, color=colors[k], lw=lw, arrow_length_ratio=0.12))

    def update(i):
        s, t = s_vals[i], i * dt_frame
        R = R_init @ so3_exp(w_b * t)
        R_dot = R @ skew(w_b)
        R0 = so3_exp(s * a)
        w_s = vee(R_dot @ R.T)
        for p in panels:
            for d in p["dyn"]:
                d.remove()
            p["dyn"].clear()

        for col, (Rn, Rn_dot) in enumerate(((R0 @ R, R0 @ R_dot), (R @ R0, R_dot @ R0))):
            p = panels[col]
            ax, dyn = p["ax"], p["dyn"]
            wb_n, ws_n = vee(Rn.T @ Rn_dot), vee(Rn_dot @ Rn.T)
            keep_b = col == 0
            assert np.allclose(wb_n if keep_b else ws_n, w_b if keep_b else w_s, atol=1e-12)
            # drawn on the page in the ORIGINAL map frame: the box moves by R in both panels
            pts = (R @ BOX.T).T
            for i0, i1 in EDGES:
                dyn.append(ax.plot(*zip(pts[i0], pts[i1]), color="0.25", lw=1.2)[0])
            sensor = R if col == 0 else R @ R0          # sensor axes on the page
            arrows(ax, (0, 0, 0), sensor, AXIS_C, 0.8, dyn)
            map_axes = (R0.T if col == 0 else np.eye(3))  # map axes on the page
            arrows(ax, (-0.9, -0.9, -0.9), map_axes, ("0.55",) * 3, 0.7, dyn, lw=1.5)
            rows = [(f"ωb (gyro, sensor axes) = {fmt(wb_n)} rad/s  " + ("unchanged" if keep_b else "changes"),
                     KEEP_C if keep_b else CHANGE_C),
                    (f"ωs (ground, map axes)  = {fmt(ws_n)} rad/s  " + ("changes" if keep_b else "unchanged"),
                     CHANGE_C if keep_b else KEEP_C),
                    (f"relabeling R₀ = Exp({s:.2f} · a), a = 60° about (1, 1, 0)/√2", "#444444")]
            for txt, (string, colr) in zip(p["rows"], rows):
                txt.set_text(string)
                txt.set_color(colr)
        return []

    anim = animation.FuncAnimation(fig, update, frames=args.frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {args.out} ({args.frames} frames)")


if __name__ == "__main__":
    main()
