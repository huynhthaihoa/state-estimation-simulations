"""Draws docs/images/frontend_backend.svg and docs/images/frontend_backend_loop_closure.svg.

Run from the repo root:
    uv run python generator/make_frontend_backend_svgs.py            # writes into docs/images/
    uv run python generator/make_frontend_backend_svgs.py some/dir   # writes into some/dir

Every label is traced to a sentence in docs/frontend_backend.md or docs/slam_mental_map.md (see the figure captions in
the doc). Each text is measured with DejaVu Sans, the widest common fallback for the font stack the SVGs ask for, and
the script stops if a label would overflow its box. The loop-closure figure is computed from the doc's toy numbers
(0.4 m of drift spread as 0.1, 0.2, 0.3 and 0.4 m), so editing those numbers moves the drawing.
"""
import os
import sys
from xml.sax.saxutils import escape

import matplotlib
from PIL import ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "images")
FONT = 'font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif"'
FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")   # DejaVu ships with matplotlib
REG, BLD = os.path.join(FONT_DIR, "DejaVuSans.ttf"), os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")
INK, MUTED = "#e2e8f0", "#94a3b8"
YEL, BLUE, GREEN, PURP, GREY = "#facc15", "#38bdf8", "#34d399", "#a78bfa", "#94a3b8"


def width(s, size, bold=False, ls=0.0):
    return ImageFont.truetype(BLD if bold else REG, size).getlength(s) + ls * len(s)


class Svg:
    def __init__(self, w, h, title, desc):
        self.w, self.h = w, h
        self.parts = []
        self.head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
                     f'role="img" aria-labelledby="t d">\n<title id="t">{escape(title)}</title>\n'
                     f'<desc id="d">{escape(desc)}</desc>\n'
                     '<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#16213b"/>'
                     '<stop offset="1" stop-color="#0b1222"/></linearGradient>'
                     '<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                     'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#94a3b8"/></marker>'
                     '<marker id="ag" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                     'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#34d399"/></marker></defs>\n'
                     f'<rect width="{w}" height="{h}" rx="18" fill="url(#bg)"/>')

    def box(self, x, y, w, h, color, dash=False, op=".10", sw=2, rx=14):
        d = ' stroke-dasharray="7 5"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{color}" fill-opacity="{op}" '
                          f'stroke="{color}" stroke-opacity=".65" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=16, bold=False, fill=INK, anchor="start", ls=0.0, maxw=None):
        w = width(s, size, bold, ls)
        if maxw is not None:
            assert w <= maxw, f"too wide ({w:.0f} > {maxw}): {s}"
        extra = f' letter-spacing="{ls}"' if ls else ""
        self.parts.append(f'<text x="{x}" y="{y}" {FONT} font-size="{size}" font-weight="{700 if bold else 400}" '
                          f'fill="{fill}" text-anchor="{anchor}"{extra}>{escape(s)}</text>')
        return w

    def raw(self, s):
        self.parts.append(s)

    def arrow(self, x1, y1, x2, y2, green=False, sw=2.5, dash=False):
        d = ' stroke-dasharray="7 5"' if dash else ""
        self.parts.append(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{GREEN if green else GREY}" stroke-width="{sw}" '
                          f'fill="none"{d} marker-end="url(#{"ag" if green else "ah"})"/>')

    def write(self, path):
        open(path, "w").write(self.head + "\n" + "\n".join(self.parts) + "\n</svg>\n")


# ---------------------------------------------------------------- figure 1: pipeline
def pipeline(path):
    s = Svg(1200, 716, "Front-end and back-end in SLAM",
            "Sensors (camera, IMU, point cloud, wheel or visual odometry, contact and phase signals) feed the front-end "
            "(feature detection, matching and tracking, motion estimation, measurements and constraints; also data "
            "association, triangulation, PnP, keyframe selection and loop-closure detection; with a LiDAR, scan "
            "registration). The front-end hands constraints (relative poses, pixel observations, IMU bundles, priors) "
            "to the back-end, which solves a nonlinear least-squares problem (bundle adjustment, pose graph, factor "
            "graph) with Gauss-Newton, Levenberg-Marquardt or incremental iSAM solvers and robust costs, producing "
            "the optimized trajectory and map. Loop closure spans both stages: the front-end detects a candidate, the "
            "back-end verifies and applies it. Both stages can run online.")
    # sensors strip
    s.box(24, 20, 1152, 46, YEL)
    s.text(42, 49, "SENSORS", 14, True, YEL, ls=1.5)
    s.text(150, 49, "camera · IMU · point cloud · wheel / visual odometry · contact / phase", 16, maxw=1010)
    s.arrow(259, 68, 259, 98)
    s.text(272, 88, "raw sensor data", 14, fill=MUTED)

    top, hh = 102, 372
    # front-end
    s.box(24, top, 470, hh, BLUE)
    s.text(42, top + 30, "FRONT-END", 14, True, BLUE, ls=1.5)
    s.text(476, top + 30, "local · sequential", 14, fill=MUTED, anchor="end")
    s.text(42, top + 58, "“What happened between these observations?”", 15, fill=MUTED, maxw=434)
    chips = ["Feature detection", "Feature matching / tracking", "Motion estimation (visual odometry)",
             "Measurements / constraints"]
    cy = top + 76
    for i, c in enumerate(chips):
        y = cy + i * 52
        s.box(42, y, 280, 34, BLUE, op=".16", sw=1.5, rx=10)
        s.text(54, y + 23, c, 14, maxw=262)
        if i < len(chips) - 1:
            s.arrow(182, y + 36, 182, y + 50, sw=2)
    s.text(338, cy + 12, "also front-end:", 14, fill=MUTED, maxw=138)
    for j, t in enumerate(["data association", "triangulation · PnP", "keyframe selection", "loop-closure",
                           "detection"]):
        s.text(338, cy + 38 + j * 22, t, 14, maxw=138)
    ly = cy + 3 * 52 + 34 + 34
    s.text(42, ly, "With a LiDAR instead of a camera, matching", 14, fill=MUTED, maxw=434)
    s.text(42, ly + 20, "and motion estimation become scan registration.", 14, fill=MUTED, maxw=434)

    # constraints
    s.box(534, top, 170, hh, GREY, op=".07")
    s.text(552, top + 30, "CONSTRAINTS", 14, True, MUTED, ls=1.0, maxw=134)
    for j, t in enumerate(["relative poses", "pixel observations", "IMU bundles", "priors"]):
        s.text(552, top + 66 + j * 26, t, 14, maxw=134)
    s.box(544, top + 186, 152, 124, GREY, dash=True, op=".05", sw=1.5, rx=10)
    for j, t in enumerate(["IMU bundles come", "from IMU", "preintegration:", "front-end work,", "back-end factor"]):
        s.text(554, top + 210 + j * 20, t, 14, fill=MUTED, maxw=136)
    s.text(552, top + 336, "constraints, not", 14, fill=MUTED, maxw=134)
    s.text(552, top + 354, "a final answer", 14, fill=MUTED, maxw=134)
    s.arrow(496, top + 186, 532, top + 186)
    s.arrow(706, top + 186, 742, top + 186)

    # back-end
    bx = 744
    s.box(bx, top, 432, hh, GREEN)
    s.text(bx + 18, top + 30, "BACK-END", 14, True, GREEN, ls=1.5)
    s.text(bx + 414, top + 30, "global · accumulated", 14, fill=MUTED, anchor="end")
    s.text(bx + 18, top + 58, "“Given all observations, what is the most", 15, fill=MUTED, maxw=396)
    s.text(bx + 18, top + 78, "consistent trajectory and map?”", 15, fill=MUTED, maxw=396)
    s.text(bx + 18, top + 108, "What we optimize (problem formulations)", 14, True, MUTED, maxw=396)
    x = bx + 18
    for c, w in (("bundle adjustment", 150), ("pose graph", 108), ("factor graph", 118)):
        s.box(x, top + 118, w, 32, GREEN, op=".16", sw=1.5, rx=10)
        s.text(x + 10, top + 139, c, 14, maxw=w - 14)
        x += w + 10
    fx = bx + 18
    w1 = s.text(fx, top + 178, "all are nonlinear least squares:  min ½ Σ ‖ r", 15, maxw=396)
    s.raw(f'<text x="{fx + w1:.1f}" y="{top + 183}" {FONT} font-size="11" fill="{INK}">i</text>')
    s.text(fx + w1 + 6, top + 178, "(x) ‖²", 15)
    s.text(bx + 18, top + 214, "How we solve it (solvers)", 14, True, MUTED, maxw=396)
    s.text(bx + 18, top + 236, "Gauss-Newton · Levenberg-Marquardt", 15, maxw=396)
    s.text(bx + 18, top + 258, "iSAM: incremental re-solving", 15, maxw=396)
    s.text(bx + 18, top + 286, "Robust costs (Huber · Cauchy · DCS) down-weight", 14, fill=MUTED, maxw=396)
    s.text(bx + 18, top + 306, "a constraint that disagrees with the rest.", 14, fill=MUTED, maxw=396)
    s.box(bx + 18, top + 322, 396, 34, INK, op=".10", sw=1.5, rx=10)
    s.text(bx + 36, top + 344, "Output: optimized trajectory + map", 15, maxw=360)

    # loop closure band
    ly0 = top + hh + 26
    s.box(24, ly0, 1152, 146, GREEN, dash=True, op=".06")
    s.text(42, ly0 + 30, "LOOP CLOSURE", 14, True, GREEN, ls=1.5)
    s.text(196, ly0 + 30, "spans both stages", 14, fill=MUTED)
    s.text(42, ly0 + 60, "Front-end: detects it", 15, True, BLUE, maxw=440)
    s.text(42, ly0 + 82, "Place recognition matches the current view against", 14, maxw=440)
    s.text(42, ly0 + 102, "stored ones and outputs a candidate constraint.", 14, maxw=440)
    s.text(bx + 18, ly0 + 60, "Back-end: verifies and applies it", 15, True, GREEN, maxw=396)
    s.text(bx + 18, ly0 + 82, "Checks it against the other constraints (robust", 14, maxw=396)
    s.text(bx + 18, ly0 + 102, "costs down-weight a bad one) and folds it into", 14, maxw=396)
    s.text(bx + 18, ly0 + 122, "the global optimization.", 14, maxw=396)
    s.arrow(500, ly0 + 92, 738, ly0 + 92, green=True)
    s.text(619, ly0 + 82, "loop-closure candidate", 14, fill=GREEN, anchor="middle", maxw=220)

    # online/offline footer
    s.text(600, ly0 + 146 + 34, "Front-end ≠ real-time, back-end ≠ offline: both stages can run online (while data arrives).",
           14, fill=MUTED, anchor="middle", maxw=1100)
    s.write(path)


# ---------------------------------------------------------------- figure 2: loop closure toy
def loop_closure(path):
    K = 125.0  # px per metre
    s = Svg(1200, 610, "Loop closure: spreading the drift over the trajectory",
            "Toy case: a robot walks a square, poses x0 (A), x1 (B), x2 (C), x3 (D) and x4 (A prime). Odometry drift "
            "leaves the final pose x4 0.4 m away from x0. The front-end detects the loop closure, a constraint that "
            "x4 coincides with x0. The back-end spreads the 0.4 m over the 4 odometry steps, so x1, x2, x3 and x4 move "
            "back by 0.1, 0.2, 0.3 and 0.4 m.")
    true = [(0.0, 0.0), (2.0, 0.0), (2.0, -2.0), (0.0, -2.0), (0.0, 0.0)]
    e = (-0.097, -0.388)                      # end drift, |e| = 0.4 m
    assert abs((e[0] ** 2 + e[1] ** 2) ** 0.5 - 0.4) < 2e-3
    drift = [(px + k / 4 * e[0], py + k / 4 * e[1]) for k, (px, py) in enumerate(true)]
    fixed = [(dx - k / 4 * e[0], dy - k / 4 * e[1]) for k, (dx, dy) in enumerate(drift)]   # = true square

    def pix(p, ox, oy):
        return ox + p[0] * K, oy - p[1] * K

    names = ["x0 (A)", "x1 (B)", "x2 (C)", "x3 (D)", "x4 (A′)"]
    for panel, (px0, title) in enumerate(((24, "Before: the drifted estimate"),
                                          (616, "After: the drift spread over the steps"))):
        s.box(px0, 20, 560, 470, GREY, op=".05")
        s.text(px0 + 18, 50, title, 16, True, INK, maxw=520)
        ox, oy = px0 + 150, 112
        d = [pix(p, ox, oy) for p in drift]
        f = [pix(p, ox, oy) for p in fixed]
        if panel == 0:
            path_pts, col, lab = d, GREY, names
        else:
            path_pts, col, lab = f, BLUE, names
            ghost = " ".join(f"{x:.1f},{y:.1f}" for x, y in d)
            s.raw(f'<polyline points="{ghost}" fill="none" stroke="{GREY}" stroke-opacity=".45" stroke-width="2" '
                  f'stroke-dasharray="5 4"/>')
            for (x, y) in d[1:]:
                s.raw(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="none" stroke="{GREY}" stroke-opacity=".5" stroke-width="1.5"/>')
            for k in range(1, 5):
                (x1, y1), (x2, y2) = d[k], f[k]
                s.arrow(f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", green=True, sw=2.2)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in path_pts)
        s.raw(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        if panel == 0:
            (x0, y0), (x4, y4) = d[0], d[4]
            s.raw(f'<path d="M{x0:.1f} {y0:.1f} L{x4:.1f} {y4:.1f}" stroke="{GREEN}" stroke-width="3" '
                  f'stroke-dasharray="6 4" fill="none"/>')
        for k, (x, y) in enumerate(path_pts):
            if panel == 1 and k == 4:
                continue
            s.raw(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#0b1222" stroke="{col}" stroke-width="2.5"/>')
        # labels
        (x0, y0), (x1, y1), (x2, y2), (x3, y3), (x4, y4) = path_pts
        if panel == 1:
            s.raw(f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="6" fill="#0b1222" stroke="{GREEN}" stroke-width="2.5"/>')
            s.text(f"{x0 - 12:.1f}", f"{y0 - 14:.1f}", "x0 = x4", 14, anchor="end")
        else:
            s.text(f"{x0 - 12:.1f}", f"{y0 - 12:.1f}", "x0 (A)", 14, anchor="end")
            s.text(f"{x4 - 12:.1f}", f"{y4 + 18:.1f}", "x4 (A′)", 14, anchor="end")
        s.text(f"{x1 + 4:.1f}", f"{y1 - 14:.1f}", "x1 (B)", 14)
        s.text(f"{x2 + 14:.1f}", f"{y2 + 5:.1f}", "x2 (C)", 14)
        s.text(f"{x3 - 12:.1f}", f"{y3 + (5 if panel else 20):.1f}", "x3 (D)", 14, anchor="end")
        if panel == 0:
            s.text(f"{x4 + 14:.1f}", f"{y4 - 12:.1f}", "0.4 m", 14, fill=GREEN)
            s.text(f"{ox + 40:.1f}", f"{oy + 58:.1f}", "loop closure:", 14, fill=GREEN)
            s.text(f"{ox + 40:.1f}", f"{oy + 78:.1f}", "x4 should sit at x0", 14, fill=GREEN)
            s.text(f"{ox + 40:.1f}", f"{oy + 98:.1f}", "(relative pose ≈ identity)", 14, fill=GREEN)
        else:
            for k in range(1, 5):
                (xa, ya), (xb, yb) = d[k], f[k]
                if k == 3:
                    s.text(f"{xb - 14:.1f}", f"{(ya + yb) / 2 + 12:.1f}", f"{0.1 * k:.1f} m", 14, fill=GREEN, anchor="end")
                else:
                    s.text(f"{xb + 12:.1f}", f"{(ya + yb) / 2 + 18:.1f}", f"{0.1 * k:.1f} m", 14, fill=GREEN)
        # captions
        if panel == 0:
            s.text(px0 + 18, 440, "Front-end: place recognition detects the loop and", 14, fill=BLUE, maxw=524)
            s.text(px0 + 18, 460, "outputs a constraint between x0 and x4.", 14, fill=BLUE, maxw=524)
        else:
            s.text(px0 + 18, 440, "Back-end: the 0.4 m is spread over the 4 odometry", 14, fill=GREEN, maxw=524)
            s.text(px0 + 18, 460, "steps, 0.1 m each; later poses move more.", 14, fill=GREEN, maxw=524)
    # legend
    ly = 520
    s.raw(f'<polyline points="30,{ly} 62,{ly}" fill="none" stroke="{GREY}" stroke-width="2.5"/>')
    s.text(70, ly + 5, "odometry step (drifted)", 14, fill=MUTED)
    s.raw(f'<polyline points="270,{ly} 302,{ly}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>')
    s.text(310, ly + 5, "corrected", 14, fill=MUTED)
    s.raw(f'<path d="M410 {ly} L442 {ly}" stroke="{GREEN}" stroke-width="3" stroke-dasharray="6 4" fill="none"/>')
    s.text(450, ly + 5, "loop-closure constraint", 14, fill=MUTED)
    s.arrow(660, ly, 692, ly, green=True, sw=2.2)
    s.text(700, ly + 5, "correction of a pose", 14, fill=MUTED)
    s.text(30, 558, "Toy case from the text: translation only, x0 anchored, equal odometry weights, loop closure exact. "
                    "Distances: 125 px = 1 m; the", 14, fill=MUTED, maxw=1140)
    s.text(30, 578, "2 m square is an arbitrary choice, and the drift direction is schematic.", 14, fill=MUTED, maxw=1140)
    s.write(path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pipeline(os.path.join(OUT, "frontend_backend.svg"))
    loop_closure(os.path.join(OUT, "frontend_backend_loop_closure.svg"))
    print("wrote", OUT)
