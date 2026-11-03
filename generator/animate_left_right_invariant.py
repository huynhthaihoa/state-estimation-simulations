'''
Animated picture for docs/filtering/left_right_invariant.md §3 ("Picture
(re-drawing the map)"): which frame change each invariant error ignores.

A planar pose X in SE(2) (3x3 homogeneous matrix: rotation by theta, then
translation) maps body coordinates to world coordinates. Two fixed poses on
the page: the true pose X (solid) and our estimate X_hat (dashed).

  - Left panel, redraw the map: the world frame is relabeled by g(s), so
    every pose becomes g X (left-multiplication). The robots do not move on
    the page; only the map axes and the printed coordinates change.
  - Right panel, re-bolt the sensor: the body frame origin moves to the
    point d(s) of the body, so every pose becomes X g (right-multiplication,
    g a body-frame translation, plus a turn with --bolt-rotation-deg). Each
    robot's frame origin (dot) slides along its own heading, so the two dots
    move in different directions.

Both panels print both errors, eta_L = X_hat^-1 X and eta_R = X X_hat^-1,
computed every frame from the relabeled matrices: eta_L stays constant on
the left and eta_R on the right (asserted to 1e-12); the other one changes.
s goes 0 -> 1 -> 0, so the GIF loops. No simulation is run.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon

TRUE_C, EST_C, KEEP_C, CHANGE_C = "#1f1f1f", "#2a7ab9", "#2a9d8f", "#d1495b"


def se2(x, y, theta):
    """SE(2) matrix of the pose (x, y, theta)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def se2_params(T):
    """(x, y, theta) of an SE(2) matrix."""
    return T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])


def fmt(T):
    x, y, th = se2_params(T)
    return f"({x:+.2f}, {y:+.2f}, {np.degrees(th):+5.1f}°)"


def robot_outline(T, size=0.22):
    """World points of an arrow-shaped robot body at pose T (nose along body x)."""
    body = np.array([[1.0, 0.0], [-0.6, 0.55], [-0.35, 0.0], [-0.6, -0.55]]) * size
    return (T[:2, :2] @ body.T).T + T[:2, 2]


def draw_axes(ax, T_axes, dyn):
    """Gray map axes on the page, with origin and directions given by the pose T_axes; appends the artists to dyn."""
    o, R = T_axes[:2, 2], T_axes[:2, :2]
    for v, name in ((R[:, 0], "x"), (R[:, 1], "y")):
        dyn.append(ax.annotate("", xy=o + 0.55 * v, xytext=o, arrowprops=dict(arrowstyle="->", color="0.55", lw=1.5)))
        dyn.append(ax.text(*(o + 0.64 * v), name, color="0.45", fontsize=9, ha="center", va="center"))
    dyn.append(ax.plot([o[0]], [o[1]], "s", color="0.55", ms=5)[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=90, help="Frames per loop (s goes 0 -> 1 -> 0)")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=70, help="GIF resolution")
    parser.add_argument("--bolt-rotation-deg", type=float, default=0.0,
                        help="Also turn the re-bolted body frame by this angle (deg) at s = 1; 0 = translation only")
    parser.add_argument("--out", type=str, default="assets/left_right_invariant.gif", help="Output GIF path")
    args = parser.parse_args()

    X = se2(1.25, 0.95, np.radians(40.0))       # truth
    X_hat = se2(0.95, 0.55, np.radians(5.0))    # estimate
    eta_L0, eta_R0 = np.linalg.inv(X_hat) @ X, X @ np.linalg.inv(X_hat)

    def new_map(s):        # pose of the relabeled map frame on the page: origin shifted, axes rotated by 50°
        return se2(0.3 * s, -0.15 * s, np.radians(50.0) * s)

    def body_change(s):    # move the body-frame origin 0.35 m forward and 0.1 m left on the body (and turn it)
        return se2(0.35 * s, 0.1 * s, np.radians(args.bolt_rotation_deg) * s)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.4))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.33, wspace=0.08)
    fig.suptitle("Each invariant error ignores one kind of frame change", fontsize=12)
    titles = ("Redraw the map:  X → gX,  X̂ → gX̂", "Re-bolt the sensor:  X → Xg,  X̂ → X̂g")
    panels = []
    for ax, title in zip(axes, titles):
        ax.set_aspect("equal")
        ax.set(xlim=(-0.35, 2.05), ylim=(-0.35, 1.75), xticks=[], yticks=[])
        ax.set_title(title, fontsize=10.5)
        for T, col, ls in ((X, TRUE_C, "-"), (X_hat, EST_C, "--")):
            ax.add_patch(Polygon(robot_outline(T), closed=True, fill=False, ec=col, lw=2, ls=ls, zorder=3))
        ax.text(*(X[:2, 2] + [-0.12, 0.12]), "truth X", color=TRUE_C, fontsize=9, ha="right")
        ax.text(*(X_hat[:2, 2] + [0.05, -0.27]), "estimate X̂", color=EST_C, fontsize=9)
        lines = [ax.text(0.02, -0.04 - 0.075 * r, "", transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
                         family="monospace") for r in range(5)]
        panels.append(dict(ax=ax, lines=lines, dyn=[]))
    # left: the arrow from the estimate to the truth, a fixed physical relation on the page
    axes[0].annotate("", xy=X[:2, 2], xytext=X_hat[:2, 2],
                     arrowprops=dict(arrowstyle="->", color=KEEP_C, lw=2, shrinkA=0, shrinkB=0))
    # right: the map does not change; its axes stay in the corner
    draw_axes(axes[1], np.eye(3), [])
    caption = ("Poses are (x, y, θ) in SE(2). Left: the robots stay put; only the map axes (gray) and so the printed "
               "coordinates change, and the arrow from estimate to truth looks the same from the estimate. Right: "
               "each pose's frame origin (dot) moves to a new spot on its body, along its own heading, so the dots "
               "shift in different directions and their distance (red line) changes. Green = unchanged, red = "
               "changes; recomputed from the matrices every frame.")
    if args.bolt_rotation_deg:
        caption = caption.replace("moves to a new spot on its body, along its own heading,",
                                  f"moves to a new spot on its body, along its own heading, and its axes turn by "
                                  f"{args.bolt_rotation_deg:g}° (short arrows: each frame's new x-axis),")
    fig.text(0.5, 0.015, textwrap.fill(caption, 150), ha="center", va="bottom", fontsize=8.5, color="#444444")

    s_vals = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(args.frames) / args.frames))

    def show(panel, Xn, Xhn, keep, extra=None):
        eL, eR = np.linalg.inv(Xhn) @ Xn, Xn @ np.linalg.inv(Xhn)
        rows = [(f"X  = {fmt(Xn)}", TRUE_C), (f"X̂  = {fmt(Xhn)}", EST_C),
                (f"ηL = X̂⁻¹X = {fmt(eL)}" + ("  unchanged" if keep == "L" else "  changes"),
                 KEEP_C if keep == "L" else CHANGE_C),
                (f"ηR = XX̂⁻¹ = {fmt(eR)}" + ("  unchanged" if keep == "R" else "  changes"),
                 KEEP_C if keep == "R" else CHANGE_C),
                (extra or "", CHANGE_C)]
        for txt, (s, col) in zip(panel["lines"], rows):
            txt.set_text(s)
            txt.set_color(col)
        return eL, eR

    def update(i):
        s = s_vals[i]
        for p in panels:
            for d in p["dyn"]:
                d.remove()
            p["dyn"].clear()

        # left: relabel the world; new coordinates are g X with g the inverse of the new map frame's page pose
        M = new_map(s)
        g = np.linalg.inv(M)
        draw_axes(axes[0], M, panels[0]["dyn"])
        eL, _ = show(panels[0], g @ X, g @ X_hat, "L")
        assert np.allclose(eL, eta_L0, atol=1e-12)

        # right: re-bolt the body frame; each pose's frame origin is X h on the page
        h = body_change(s)
        Xn, Xhn = X @ h, X_hat @ h
        dyn = panels[1]["dyn"]
        for T0, T1, col in ((X, Xn, TRUE_C), (X_hat, Xhn, EST_C)):
            dyn.append(axes[1].plot([T0[0, 2], T1[0, 2]], [T0[1, 2], T1[1, 2]], ":", color=col, lw=1.2)[0])
            dyn.append(axes[1].plot([T1[0, 2]], [T1[1, 2]], "o", color=col, ms=8, mec="white", zorder=5)[0])
            if args.bolt_rotation_deg:
                dyn.append(axes[1].annotate("", xy=T1[:2, 2] + 0.3 * T1[:2, 0], xytext=T1[:2, 2],
                                            arrowprops=dict(arrowstyle="->", color=col, lw=1.8), zorder=6))
        dyn.append(axes[1].plot([Xn[0, 2], Xhn[0, 2]], [Xn[1, 2], Xhn[1, 2]], color=CHANGE_C, lw=1.5, zorder=4)[0])
        dist = np.linalg.norm(Xn[:2, 2] - Xhn[:2, 2])
        _, eR = show(panels[1], Xn, Xhn, "R", f"distance between the dots = {dist:.3f} m  changes")
        assert np.allclose(eR, eta_R0, atol=1e-12)
        return []

    anim = animation.FuncAnimation(fig, update, frames=args.frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {args.out} ({args.frames} frames)")


if __name__ == "__main__":
    main()
