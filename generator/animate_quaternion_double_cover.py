'''
Animated companion to docs/foundations/quaternion.md section 10 ("q and -q are
the same rotation"): a box turns about the Z axis from 0 to 720 degrees.

Two panels:
  1. left: the 3D box (with its body axes), oriented by
     utils.quat_to_rotation(q), with a faint copy of the start orientation;
  2. right: the unit quaternion q = (w, 0, 0, z) = (cos(theta/2), 0, 0, sin(theta/2))
     (the doc's (w, x, y, z) order, section 2) as a point on the unit circle in the
     (w, z) plane, with its trail, and -q (hollow) at the opposite point. Readouts
     show theta, theta/2, q and -q.

Message: the box is back at its start after 360 degrees, but q = (-1, 0, 0, 0) is
-q of the start; only after 720 degrees does q return to (+1, 0, 0, 0). q and -q
give the same box at every frame (the double cover).

Data source: exact math with the repo's own helpers, no randomness (so no
--seed): q = utils.quat_exp([0, 0, theta]) and R = utils.quat_to_rotation(q).
The generator asserts, at every frame, that quat_to_rotation(q) equals
quat_to_rotation(-q) and that q matches (cos(theta/2), 0, 0, sin(theta/2)); and
that R(360 deg) = I, q(360 deg) = (-1, 0, 0, 0) and q(720 deg) = (+1, 0, 0, 0).

Illustrative items: the box dimensions and the camera angle (box drawing copied
from generator/make_figures.py draw_frame3d / clean3d), the 10 degree step per
frame, and the pauses at 360 and 720 degrees (repeated frames, no new data).

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from state_estimation_simulations.utils import quat_exp, quat_to_rotation

AXIS_RGB = ("#d1495b", "#2a9d8f", "#2e86de")  # body x, y, z
BOX = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.3, 0.3) for z in (-0.12, 0.12)])  # illustrative box


def draw_frame3d(ax, R, length=1.0, lw=2.5, box=True, alpha=1.0, axes=True):
    """Body axes (columns of R) as colored arrows, plus an optional flat box (copied from make_figures.py)."""
    if axes:
        for k in range(3):
            ax.quiver(0, 0, 0, *(length * R[:, k]), color=AXIS_RGB[k], lw=lw, arrow_length_ratio=0.15, alpha=alpha)
    if box:
        pts = (R @ BOX.T).T
        for i in range(8):
            for j in range(i + 1, 8):
                if np.sum(BOX[i] != BOX[j]) == 1:
                    ax.plot(*zip(pts[i], pts[j]), color="0.35", lw=1, alpha=alpha)


def clean3d(ax, lim=1.0):
    """Hide ticks, fix the limits and the camera (copied from make_figures.py)."""
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim), xticks=[], yticks=[], zticks=[])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-55)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=float, default=10.0, help="Rotation angle per GIF frame, degrees (illustrative)")
    parser.add_argument("--hold", type=int, default=6, help="Repeated frames at 360 and 720 degrees")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.add_argument("--out", type=str, required=True, help="Output GIF path")
    args = parser.parse_args()

    n_steps = int(round(720 / args.step))
    deg = np.arange(n_steps + 1) * args.step
    theta = np.deg2rad(deg)
    Q = np.array([quat_exp(np.array([0.0, 0.0, th])) for th in theta])      # (w, x, y, z)
    R = np.array([quat_to_rotation(q) for q in Q])
    R_neg = np.array([quat_to_rotation(-q) for q in Q])

    # exact-math checks (see the module docstring)
    assert np.allclose(R, R_neg, atol=1e-12), "R(q) != R(-q)"
    assert np.allclose(Q, np.stack([np.cos(theta / 2), 0 * theta, 0 * theta, np.sin(theta / 2)], axis=1), atol=1e-12)
    k360, k720 = int(round(360 / args.step)), n_steps
    assert np.isclose(deg[k360], 360) and np.isclose(deg[k720], 720)
    assert np.allclose(R[k360], np.eye(3), atol=1e-12), "R(360 deg) != I"
    assert np.allclose(Q[k360], [-1, 0, 0, 0], atol=1e-12) and np.allclose(Q[k720], [1, 0, 0, 0], atol=1e-12)

    # frame list: index into the angle table (pauses repeat a key frame)
    idx = [0] * 3
    for k in range(n_steps + 1):
        idx.append(k)
        if k in (k360, k720):
            idx += [k] * args.hold
    n_frames = len(idx)

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.12)
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    ax_q = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.17)
    fig.suptitle("q and -q are the same rotation: the box is back at 360 deg, q only at 720 deg",
                 fontsize=12, fontweight="bold")

    ax_q.set_aspect("equal")
    ax_q.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), xlabel="w  (= cos(theta/2))", ylabel="z  (= sin(theta/2))",
             title="Unit quaternion q = (w, 0, 0, z) on the unit circle")
    ang = np.linspace(0, 2 * np.pi, 200)
    ax_q.plot(np.cos(ang), np.sin(ang), color="0.8", lw=1.5)
    ax_q.axhline(0, color="0.9", lw=0.8)
    ax_q.axvline(0, color="0.9", lw=0.8)
    ax_q.plot([1], [0], "s", ms=7, color="k", mfc="none")
    ax_q.text(0.93, -0.1, "start and 720 deg:\nq = (1, 0, 0, 0)", fontsize=8, ha="right", va="top")
    ax_q.plot([-1], [0], "s", ms=7, color="k", mfc="none")
    ax_q.text(-0.93, -0.1, "360 deg:\nq = (-1, 0, 0, 0)\n= -q of the start", fontsize=8, ha="left", va="top")
    trail, = ax_q.plot([], [], color="tab:blue", lw=2.5)
    pt_q, = ax_q.plot([], [], "o", ms=10, color="tab:blue", label="q")
    pt_n, = ax_q.plot([], [], "o", ms=10, color="tab:orange", mfc="none", mew=2, label="-q (same box)")
    ax_q.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=9, ncol=2, frameon=False)
    readout = ax_q.text(0, 0.45, "", ha="center", va="center", fontsize=9, fontweight="bold")

    fig.text(0.5, 0.025, "Same box for q and -q at every frame (checked in the generator with utils.quat_to_rotation). "
             "q = (cos(theta/2), 0, 0, sin(theta/2)), so q(360 deg) = -q(0 deg).\n"
             "Illustrative: box shape, camera angle, 10 deg per frame, pauses at 360 and 720 deg.",
             ha="center", va="bottom", fontsize=9)

    def update(f):
        k = idx[f]
        ax3.cla()
        clean3d(ax3, 1.0)
        draw_frame3d(ax3, np.eye(3), length=0.8, lw=1, alpha=0.25, axes=True)   # faint start orientation
        draw_frame3d(ax3, R[k], length=0.8, lw=2.5)
        ax3.set_title("Box turning about Z, R = quat_to_rotation(q)", fontsize=10)
        if k == k360:
            note = "box = start orientation again (R = I), but q = -q(0)"
        elif k == k720:
            note = "q = +q(0) again: one full turn of q takes 720 deg"
        else:
            note = ""
        ax3.text2D(0.5, 0.02, note, transform=ax3.transAxes, ha="center", fontsize=9, color="tab:red", fontweight="bold")
        trail.set_data(Q[:k + 1, 0], Q[:k + 1, 3])
        pt_q.set_data([Q[k, 0]], [Q[k, 3]])
        pt_n.set_data([-Q[k, 0]], [-Q[k, 3]])
        q = Q[k]
        w, z = np.round(q[0], 3) + 0.0, np.round(q[3], 3) + 0.0   # round, then + 0.0 turns -0.0 into 0.0 for printing
        readout.set_text(f"theta = {deg[k]:.0f} deg\ntheta/2 = {deg[k] / 2:.0f} deg\n"
                         f"q  = ({w:+.3f}, 0, 0, {z:+.3f})\n-q = ({-w + 0.0:+.3f}, 0, 0, {-z + 0.0:+.3f})")
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)

    print(f"theta 0..720 deg in {args.step:g} deg steps; asserts passed (R(q) == R(-q) at all {len(deg)} angles, R(360) == I)")
    for k in (0, k360, k720):
        q = Q[k]
        print(f"  theta {deg[k]:4.0f} deg: q = ({q[0]:+.3f}, 0, 0, {np.round(q[3], 3) + 0.0:+.3f}), max|R - I| = {np.abs(R[k] - np.eye(3)).max():.1e}")
    print(f"Saved {n_frames} frames to {args.out}")


if __name__ == "__main__":
    main()
