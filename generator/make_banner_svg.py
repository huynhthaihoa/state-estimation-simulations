"""Draws assets/banner.svg (the pipeline banner at the top of the README).

Run from the repo root:
    uv run python generator/make_banner_svg.py            # rewrites assets/banner.svg
    uv run python generator/make_banner_svg.py out.svg    # writes elsewhere

The boxes, texts and arrows are laid out here, so editing a label or moving a box is a one-line change. The small
"trajectory + map" picture in the ESTIMATE box is scene data (its points, covariance-ellipse sizes, loop-closure
curve and star positions), kept as numbers below. The banner is also what the README's image alt text and the SVG's
own <desc> describe: when the content changes, change DESC and the alt text in README.md together.

This script was written to reproduce the banner that had been edited by hand; its output matches it byte for byte.
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "assets", "banner.svg")

W, H = 1200, 440
FONT = 'font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif"'
INK, MUTED = "#e2e8f0", "#94a3b8"
YEL, BLUE, PURP, GREEN, SLATE = "#facc15", "#38bdf8", "#a78bfa", "#34d399", "#64748b"

TITLE = "State Estimation Simulations"
SUBTITLE = "From-scratch pose estimation on Lie groups (SO(3), SE(3), SE\u2082(3)) \u00b7 NumPy and manif side by side"
DESC = 'Pipeline overview: sensors (including contact signals) feed a front-end (PnP, triangulation, visual-inertial initialization, LiDAR primitives), whose constraints go to two back-end families - Kalman filtering (including crawling and legged robots) and factor-graph optimization - producing a trajectory and map estimate, all built on Lie-group and Jacobian foundations.'
DEFS = '<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#16213b"/><stop offset="1" stop-color="#0b1222"/></linearGradient><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#94a3b8"/></marker></defs>'

out = []


def text(x, y, size, weight, fill, s, anchor="start", ls=None):
    extra = f' letter-spacing="{ls}"' if ls else ""
    out.append(f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}" font-weight="{weight}" fill="{fill}" '
               f'text-anchor="{anchor}"{extra}>{s}</text>')


def box(x, y, w, h, color, rx=14, fill_op=".10", stroke_op=".65"):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{color}" fill-opacity="{fill_op}" '
               f'stroke="{color}" stroke-opacity="{stroke_op}" stroke-width="2"/>')


def panel(x, y, w, h, color, title, lines, line_y0, title_y):
    """A titled box; the title sits at the baseline `title_y` and each line at its own baseline in `line_y0`."""
    box(x, y, w, h, color)
    text(x + 18, title_y, 14, 700, color, title, ls=1.5)
    for s, ly in zip(lines, line_y0):
        text(x + 18, ly, 16, 400, INK, s)


def arrow(x1, y1, x2, y2):
    out.append(f'<path d="M{x1:.1f} {y1:.1f} L{x2:.1f} {y2:.1f}" stroke="{MUTED}" stroke-width="2.5" fill="none" '
               f'marker-end="url(#ah)"/>')


def star(cx, cy, R=6.0, r=2.7):
    pts = []
    for k in range(10):
        a = math.radians(36 * k)
        rad = R if k % 2 == 0 else r
        pts.append((cx + rad * math.sin(a), cy - rad * math.cos(a)))
    d = " ".join(("M" if k == 0 else "L") + f"{x:.1f} {y:.1f}" for k, (x, y) in enumerate(pts))
    out.append(f'<path d="{d} Z" fill="{YEL}" fill-opacity=".85"/>')


# ---------------------------------------------------------------- frame and headings
out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
           f'aria-labelledby="t d">')
out.append(f'<title id="t">{TITLE}</title>')
out.append(f'<desc id="d">{DESC}</desc>')
out.append(DEFS)
out.append(f'<rect width="{W}" height="{H}" rx="18" fill="url(#bg)"/>')
text(48, 64, 34, 700, INK, TITLE)
text(48, 96, 17, 400, MUTED, SUBTITLE)

# ---------------------------------------------------------------- pipeline boxes
panel(48, 170, 190, 132, YEL, "SENSORS", ["IMU \u00b7 camera", "odometry \u00b7 contact", "point clouds"],
      (228, 252, 276), 200)
panel(278, 170, 214, 132, BLUE, "FRONT-END", ["PnP \u00b7 triangulation", "VI initialization", "LiDAR primitives"],
      (228, 252, 276), 200)
panel(532, 126, 336, 118, PURP, "FILTERING",
      ["KF \u00b7 EKF \u00b7 IEKF \u00b7 UKF", "saltation \u00b7 ZUPT \u00b7 slip \u00b7 friction", "legged \u00b7 gait phase"],
      (180, 202, 224), 154)
panel(532, 256, 336, 96, GREEN, "OPTIMIZATION",
      ["GN \u00b7 LM \u00b7 factor graphs \u00b7 BA", "iSAM \u00b7 marginalization"], (310, 332), 284)
box(908, 126, 244, 226, INK)
text(926, 156, 14, 700, INK, "ESTIMATE", ls=1.5)

arrow(240, 236, 274, 236)
arrow(494, 222, 528, 180)
arrow(494, 250, 528, 300)
arrow(870, 180, 904, 214)
arrow(870, 300, 904, 266)

# ---------------------------------------------------------------- the picture in the ESTIMATE box (scene data)
XS = [934.0 + 195.0 * k / 7 for k in range(8)]                           # evenly spaced along x
YS = [318.0, 256.3, 206.2, 176.9, 173.9, 198.1, 245.7, 308.5]            # the trajectory's height
RX = [8.5, 9.5, 13.5, 17.0, 16.5, 16.0, 11.0, 14.5]                      # covariance ellipse half-widths
LOOP_FROM, LOOP_TO, LOOP_CTRL = 6, 1, (1031.5, 296.3)                    # loop closure: pose 6 back to pose 1
STARS = [(958.0, 190.0), (1062.0, 232.0), (1112.0, 192.0), (975.0, 302.0), (1090.0, 296.0)]   # landmarks

for k, (x, y) in enumerate(zip(XS, YS)):
    out.append(f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{RX[k]:.1f}" ry="{RX[k] * 0.55:.1f}" '
               f'transform="rotate({20 * k} {x:.1f} {y:.1f})" fill="{PURP}" fill-opacity=".22" stroke="{PURP}" '
               f'stroke-opacity=".85"/>')
d = " ".join(("M" if k == 0 else "L") + f"{x:.1f} {y:.1f}" for k, (x, y) in enumerate(zip(XS, YS)))
out.append(f'<path d="{d}" stroke="{BLUE}" stroke-width="2.5" fill="none"/>')
out.append(f'<path d="M{XS[LOOP_FROM]:.1f} {YS[LOOP_FROM]:.1f} Q{LOOP_CTRL[0]:.1f} {LOOP_CTRL[1]:.1f} '
           f'{XS[LOOP_TO]:.1f} {YS[LOOP_TO]:.1f}" stroke="{GREEN}" stroke-width="2.5" stroke-dasharray="7 5" '
           f'fill="none"/>')
text(1031, 292, 12, 400, GREEN, "loop closure", anchor="middle")
for x, y in zip(XS, YS):
    out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#0b1222" stroke="{BLUE}" stroke-width="2.5"/>')
for cx, cy in STARS:
    star(cx, cy)
text(1030, 344, 14, 400, MUTED, "trajectory + map", anchor="middle")

# ---------------------------------------------------------------- foundations bar
box(48, 372, 1104, 42, SLATE, rx=12, fill_op=".14", stroke_op=".6")
text(66, 399, 14, 700, MUTED, "FOUNDATIONS", ls=1.5)
text(210, 399, 16, 400, INK, "Jacobians \u00b7 Lie groups (Exp / Log) \u00b7 quaternions \u00b7 Umeyama alignment")
out.append("</svg>")

if __name__ == "__main__":
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote", OUT)
