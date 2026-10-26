"""Draws docs/images/factor_graph.svg and docs/images/factor_graph_vs_pose_graph.svg.

Run from the repo root:
    uv run python generator/make_factor_graph_svgs.py            # writes into docs/images/
    uv run python generator/make_factor_graph_svgs.py some/dir   # writes into some/dir

Both figures show the poses x0..x3 and the x0-x3 loop closure of docs/optimization/factor_graph.md §2, plus one landmark
l0 and each other kind of factor. Drawing convention is the one of §5: variables are circles, factors are squares, and
every edge joins a circle to a square. The factor kinds are the six of §6 (odometry, IMU, camera, GPS, loop closure,
prior). Each label is measured with DejaVu Sans, the widest common fallback for the font stack the SVGs ask for, and the
script stops if one would overflow its node.
"""
import os
import sys
from xml.sax.saxutils import escape

from make_frontend_backend_svgs import BLUE, GREEN, GREY, INK, MUTED, PURP, YEL, Svg, width

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "images")

R, HALF = 26, 12                                  # circle radius, half side of a factor square
XS = [160, 420, 680, 940]                         # pose columns (x0..x3), shared by both figures
YP = 250                                          # pose row (graph-local coordinates)
LM = (550, 85)                                    # landmark l0
BG = "#0b1222"


def mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def factors():
    """Every factor of the example: kind label, colour, square position, polyline through it, label anchor, dropped by a pose graph?"""
    p = [(x, YP) for x in XS]
    f = []
    f.append(dict(label="prior", col=YEL, sq=(XS[0], 168), path=[(XS[0], 168), p[0]], lab=(XS[0], 144, "middle"), drop=False))
    f.append(dict(label="GPS", col=YEL, sq=(XS[3], 168), path=[(XS[3], 168), p[3]], lab=(XS[3], 144, "middle"), drop=True))
    for i in range(3):
        m = mid(p[i], p[i + 1])
        f.append(dict(label="odometry", col=GREEN, sq=m, path=[p[i], p[i + 1]], lab=(m[0], YP - 22, "middle"), drop=False))
    f.append(dict(label="IMU", col=GREEN, sq=(550, 315), path=[p[1], (XS[1], 315), (XS[2], 315), p[2]],
                  lab=(550, 349, "middle"), drop=False))
    f.append(dict(label="loop closure", col=GREEN, sq=(550, 395), path=[p[0], (XS[0], 395), (XS[3], 395), p[3]],
                  lab=(550, 429, "middle"), drop=False))
    for i in (1, 2):
        m = mid(LM, p[i])
        f.append(dict(label="camera", col=GREEN, sq=m, path=[LM, p[i]],
                      lab=(m[0] - 20, m[1] + 5, "end") if i == 1 else (m[0] + 20, m[1] + 5, "start"), drop=True))
    return f


def factor_graph(s, oy, fade_dropped):
    """Draws the §2 example with its graph-local y shifted by oy. Items a pose graph drops are faded if fade_dropped."""
    def fade(drop):
        return ' opacity=".38"' if (fade_dropped and drop) else ""

    def pt(q):
        return f"{q[0]:.1f},{q[1] + oy:.1f}"

    fs = factors()
    # edges first, then variables, then factor squares and labels
    for f in fs:
        d = "M" + " L".join(pt(q) for q in f["path"])
        s.raw(f'<path d="{d}" fill="none" stroke="{GREY}" stroke-width="2.5" stroke-linejoin="round"{fade(f["drop"])}/>')
    nodes = [(f"x{i}", XS[i], YP, BLUE, False) for i in range(4)] + [("l0", LM[0], LM[1], PURP, True)]
    for name, x, y, col, drop in nodes:
        assert width(name, 16, True) <= 2 * R - 14, name
        s.raw(f'<circle cx="{x}" cy="{y + oy}" r="{R}" fill="{BG}"/>')
        s.raw(f'<g{fade(drop)}><circle cx="{x}" cy="{y + oy}" r="{R}" fill="{BG}" stroke="{col}" stroke-width="2.5"/>'
              f'<circle cx="{x}" cy="{y + oy}" r="{R}" fill="{col}" fill-opacity=".18"/></g>')
        s.text(x, y + oy + 5.5, name, 16, True, anchor="middle")
        if drop and fade_dropped:
            s.parts[-1] = s.parts[-1].replace("<text ", '<text opacity=".38" ', 1)
    for f in fs:
        x, y = f["sq"]
        s.raw(f'<rect x="{x - HALF}" y="{y + oy - HALF}" width="{2 * HALF}" height="{2 * HALF}" rx="3" fill="{BG}"/>')
        s.raw(f'<g{fade(f["drop"])}><rect x="{x - HALF}" y="{y + oy - HALF}" width="{2 * HALF}" height="{2 * HALF}" rx="3" '
              f'fill="{BG}" stroke="{f["col"]}" stroke-width="2.5"/><rect x="{x - HALF}" y="{y + oy - HALF}" '
              f'width="{2 * HALF}" height="{2 * HALF}" rx="3" fill="{f["col"]}" fill-opacity=".35"/></g>')
        lx, ly, anchor = f["lab"]
        s.text(f"{lx:.1f}", f"{ly + oy:.1f}", f["label"], 14, anchor=anchor)
        if f["drop"] and fade_dropped:
            s.parts[-1] = s.parts[-1].replace("<text ", '<text opacity=".38" ', 1)


def legend(s, y, x0=120):
    """Four-item key: pose, landmark, binary factor, unary factor, then the bipartite rule."""
    x = x0
    for kind, col, text in (("c", BLUE, "pose"), ("c", PURP, "landmark"),
                            ("s", GREEN, "binary factor: joins two variables"),
                            ("s", YEL, "unary factor: joins one variable")):
        if kind == "c":
            s.raw(f'<circle cx="{x + 10}" cy="{y}" r="10" fill="{BG}" stroke="{col}" stroke-width="2.5"/>'
                  f'<circle cx="{x + 10}" cy="{y}" r="10" fill="{col}" fill-opacity=".18"/>')
        else:
            s.raw(f'<rect x="{x}" y="{y - 10}" width="20" height="20" rx="3" fill="{BG}" stroke="{col}" stroke-width="2.5"/>'
                  f'<rect x="{x}" y="{y - 10}" width="20" height="20" rx="3" fill="{col}" fill-opacity=".35"/>')
        w = s.text(x + 30, y + 5, text, 14, fill=MUTED)
        x += 30 + w + 34
    assert x - 34 <= 1100 - 24, f"legend too wide: {x - 34}"
    s.text(550, y + 38, "Every edge joins a circle to a square: variables and factors alternate (a bipartite graph).",
           14, fill=MUTED, anchor="middle", maxw=1000)


def anatomy(path):
    s = Svg(1100, 540, "A factor graph with every factor kind",
            "Poses x0 to x3 in a row, joined in pairs by odometry factors (squares). An IMU factor runs in parallel to "
            "the odometry factor between x1 and x2. Landmark l0 is joined to x1 and x2 by camera factors. A loop-closure "
            "factor joins x0 and x3. A prior factor touches x0 only and a GPS factor touches x3 only. Variables are "
            "circles, factors are squares, and every edge joins a circle to a square.")
    factor_graph(s, 0, False)
    legend(s, 480)
    s.write(path)


def versus(path):
    s = Svg(1100, 853, "Factor graph versus pose graph",
            "Top: the factor graph of the first figure, with the parts a pose graph drops faded: landmark l0, the "
            "camera factors and the GPS factor. Bottom: the pose graph, which keeps poses x0 to x3 and the factors between "
            "them, drawn as edges: three odometry edges, an IMU edge between x1 and x2 and a loop-closure edge, with x0 "
            "anchored by its prior.")
    s.box(24, 20, 1052, 475, GREY, op=".05")
    s.text(42, 50, "Factor graph: variables and factors are both nodes", 16, True, maxw=700)
    s.text(1058, 50, "faded: dropped by a pose graph", 14, fill=MUTED, anchor="end", maxw=300)
    factor_graph(s, 25, True)
    s.box(24, 515, 1052, 318, GREY, op=".05")
    s.text(42, 545, "Pose graph: only poses; each edge is a factor between two poses", 16, True, maxw=700)
    y = 635
    for i in range(3):
        s.raw(f'<path d="M{XS[i]} {y} L{XS[i + 1]} {y}" stroke="{GREEN}" stroke-width="3.5"/>')
        s.text((XS[i] + XS[i + 1]) / 2, y - 14, "odometry", 14, anchor="middle")
    iy, ly = y + 48, y + 100
    s.raw(f'<path d="M{XS[1]} {y} L{XS[1]} {iy} L{XS[2]} {iy} L{XS[2]} {y}" fill="none" stroke="{GREEN}" '
          f'stroke-width="3.5" stroke-linejoin="round"/>')
    s.text(550, iy + 22, "IMU", 14, anchor="middle")
    s.raw(f'<path d="M{XS[0]} {y} L{XS[0]} {ly} L{XS[3]} {ly} L{XS[3]} {y}" fill="none" stroke="{GREEN}" '
          f'stroke-width="3.5" stroke-linejoin="round"/>')
    s.text(550, ly + 24, "loop closure", 14, anchor="middle")
    # anchor on x0: the prior, drawn as a ground symbol
    ax = XS[0]
    s.raw(f'<path d="M{ax} {y - R} L{ax} {y - R - 22} M{ax - 16} {y - R - 22} L{ax + 16} {y - R - 22}" fill="none" '
          f'stroke="{YEL}" stroke-width="3"/>')
    for k in (-12, 0, 12):
        s.raw(f'<path d="M{ax + k} {y - R - 22} L{ax + k - 6} {y - R - 30}" stroke="{YEL}" stroke-width="2.5"/>')
    s.text(ax + 24, y - R - 22, "anchor (prior on x0)", 14, fill=YEL, maxw=200)
    for i, x in enumerate(XS):
        s.raw(f'<circle cx="{x}" cy="{y}" r="{R}" fill="{BG}" stroke="{BLUE}" stroke-width="2.5"/>'
              f'<circle cx="{x}" cy="{y}" r="{R}" fill="{BLUE}" fill-opacity=".18"/>')
        s.text(x, y + 5.5, f"x{i}", 16, True, anchor="middle")
    s.text(42, 788, "The odometry, IMU and loop-closure squares have become edges, and the prior stays as an anchor on x0;",
           14, fill=MUTED, maxw=1000)
    s.text(42, 808, "the landmark, camera and GPS factors are left out.", 14, fill=MUTED, maxw=1000)
    s.write(path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    anatomy(os.path.join(OUT, "factor_graph.svg"))
    versus(os.path.join(OUT, "factor_graph_vs_pose_graph.svg"))
    print("wrote", OUT)
