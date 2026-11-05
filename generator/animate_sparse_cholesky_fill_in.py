'''
Animated companion of the static figure fig_sparse_fill_in_orderings
(generator/make_figures.py), for docs/optimization/sparse_cholesky_factorization.md
section 5: the Cholesky factor L of one bundle-adjustment Hessian H is built
column by column, for three elimination orders side by side (poses first,
landmarks first, interleaved).

Scene: the same as the static figure: bundle_adjustment.py's
generate_ground_truth_scene(8, 60, 5.0, 180.0, 2.0, rng) and build_observations
(K = (800, 800, 320, 240), pixel noise 1.0, min_observations 2), seed 0 (the
script's own default; the rng is used only by these two calls, as in the
static figure). 8 cameras (6 unknowns each) and 48 landmarks (3 each) remain,
192 unknowns. The sparsity pattern of H and the three orders are built exactly
as in the static figure, and the structure of L comes from utils.symbolic_eliminate
(the elimination game), whose separators give the nonzeros of every column of L.

Panels:
  1. top row, one panel per order: the lower-triangular pattern of L, revealed
     one block step at a time (one camera = 6 columns, one landmark = 3 columns,
     56 steps). Black = entries of L that are also nonzero in H, red = fill-in
     (nonzero in L, zero in H). The column band being built is shaded. Not yet
     built columns show the entries of H in gray and, in salmon, the fill-in
     that earlier eliminations have already created but whose column is not
     built yet. Under each panel: columns built, nnz(L) so far, fill-in drawn
     so far, fill-in created so far.
  2. bottom row, one panel per order: the two fill-in counts against the step,
     on a common vertical scale: fill-in drawn in the built columns (red) and
     fill-in created by the eliminations done so far (salmon, dashed).

Why two fill counts: eliminating a variable creates new edges between its
remaining neighbors, but the nonzero entry of L for such an edge is stored in
the column of the neighbor that is eliminated EARLIER of the two, which is
still to come. So fill created at one step is always drawn in a later column.
Poses first: the cameras' eliminations create the landmark-landmark fill, which
is drawn when the landmark columns are built. Landmarks first: the landmarks'
eliminations create the camera-camera fill, drawn only in the camera columns at
the end (no red before that, although the fill was created during the landmark
steps). The curves show this lag; the counts say nothing about wall-clock time.

Everything drawn comes from the repo's own functions. The fill-in created by
step v is derived from symbolic_eliminate's separators: an edge (a, b) with both
a, b in the separator of v, not an edge of H and not created at an earlier step,
is created at step v (this is the definition of the elimination game). The script
asserts that this derivation reproduces fill-in exactly (every fill entry of L is
created once, and the totals equal nnz(L) - nnz(lower H)) and that the final numbers
match the doc table: nnz(L) 15819 / 6720 / 13452, fill-in 10107 / 1008 / 7740,
nnz(lower H) 5712.

Illustrative: the order of block steps follows the elimination order; the stepping
speed (one block per frame) and the holds at the start and end are display choices.
Matrix images are 192 x 192 entries drawn into a few hundred pixels, so single
entries are not individually resolved. Seed 0 by default.

Writes a GIF with matplotlib's pillow writer (no ffmpeg needed).
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from state_estimation_simulations.use_numpy import bundle_adjustment as ba
from state_estimation_simulations.utils import symbolic_eliminate

NAMES = ["poses first", "landmarks first", "interleaved"]
EXPECTED = {"poses first": (15819, 10107), "landmarks first": (6720, 1008), "interleaved": (13452, 7740)}
EXPECTED_NNZ_H = 5712
CAM_C, LM_C = "#2e86de", "#2a9d8f"                      # repo palette (make_figures.C)
RED, PENDING, GRAY = (0.82, 0.29, 0.36, 1.0), (0.95, 0.63, 0.68, 1.0), (0.80, 0.80, 0.80, 1.0)
BLACK = (0.0, 0.0, 0.0, 1.0)
BAND = (1.0, 0.85, 0.2, 0.35)                           # shading of the column band being built


def orderings(nc, nl, prs):
    """Same three block orders as the static figure."""
    first = {}
    for i, j in sorted(prs):
        first.setdefault(j, i)
    inter = []
    for i in range(nc):
        inter += [("c", i)] + [("l", j) for j in range(nl) if first[j] == i]
    return {"poses first": [("c", i) for i in range(nc)] + [("l", j) for j in range(nl)],
            "landmarks first": [("l", j) for j in range(nl)] + [("c", i) for i in range(nc)],
            "interleaved": inter}


def analyse(nc, nl, prs, order):
    """Per-order data on the 6/3-dof scalar level (the static figure's scalar=True case)."""
    size = {"c": 6, "l": 3}
    base, n = {}, 0
    for kind, cnt in (("c", nc), ("l", nl)):
        for idx in range(cnt):
            base[(kind, idx)] = list(range(n, n + size[kind]))
            n += size[kind]
    edges = set()
    for blk in base.values():
        edges |= {(a, b) for a in blk for b in blk if a < b}
    for i, j in prs:
        edges |= {(a, b) for a in base[("c", i)] for b in base[("l", j)]}
    scal_order = [s for blk in order for s in base[blk]]
    sep, _ = symbolic_eliminate(n, edges, scal_order)
    pos = {v: k for k, v in enumerate(scal_order)}

    H = np.eye(n, dtype=bool)
    L = np.eye(n, dtype=bool)
    for a, b in edges:
        H[pos[a], pos[b]] = H[pos[b], pos[a]] = True
    for v, nb in sep.items():
        for u in nb:
            L[max(pos[u], pos[v]), min(pos[u], pos[v])] = True
    Hl = np.tril(H)
    fill = L & ~Hl

    # creation step (scalar elimination index) of every fill entry, from the separators
    big = n + 1
    created = np.full((n, n), big, dtype=int)           # created[row, col]; row > col in elimination positions
    for v in scal_order:
        nb = sep[v]
        for x in range(len(nb)):
            for y in range(x + 1, len(nb)):
                a, b = nb[x], nb[y]
                if (min(a, b), max(a, b)) in edges:
                    continue
                r, c = max(pos[a], pos[b]), min(pos[a], pos[b])
                if created[r, c] == big:
                    created[r, c] = pos[v]
    assert np.array_equal(created < big, fill), "created-fill derivation must reproduce the fill-in exactly"
    assert fill.sum() == L.sum() - np.tril(H).sum()
    return dict(n=n, Hl=Hl, L=L, fill=fill, created=created, nnz_H=int(Hl.sum()), nnz_L=int(L.sum()),
                n_fill=int(fill.sum()), cols=[len(base[b]) for b in order], is_cam=[b[0] == "c" for b in order],
                order=order)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (see the module docstring)")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    parser.add_argument("--hold-start", type=int, default=8, help="Frames held on the empty factor")
    parser.add_argument("--hold-end", type=int, default=24, help="Frames held on the finished factor")
    parser.add_argument("--out", type=str, required=True, help="Output GIF path")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    K = (800.0, 800.0, 320.0, 240.0)
    T_true, P_all = ba.generate_ground_truth_scene(8, 60, 5.0, 180.0, 2.0, rng)
    P_true, obs = ba.build_observations(T_true, P_all, K, 1.0, 2, rng)
    n_cam, n_lm = len(T_true), len(P_true)
    pairs = sorted({(i, j) for i, j, _ in obs})

    data = {}
    for name, order in orderings(n_cam, n_lm, pairs).items():
        data[name] = analyse(n_cam, n_lm, pairs, order)
    n = data["poses first"]["n"]
    print(f"scene: {n_cam} cameras x {n_lm} landmarks (seed {args.seed}), {len(obs)} observations, "
          f"{len(pairs)} camera-landmark pairs, {n} unknowns")
    for name in NAMES:
        d = data[name]
        print(f"  {name:<16s} nnz(lower H) {d['nnz_H']}, nnz(L) {d['nnz_L']}, fill-in {d['n_fill']}, steps {len(d['order'])}")
        if args.seed == 0:
            assert d["nnz_H"] == EXPECTED_NNZ_H and (d["nnz_L"], d["n_fill"]) == EXPECTED[name], (name, d["nnz_L"], d["n_fill"])
    assert (n_cam, n_lm, n) == (8, 48, 192) or args.seed != 0

    # per-step quantities (a step = one block = one camera or one landmark)
    n_steps = len(data["poses first"]["order"])
    for name in NAMES:
        d = data[name]
        ends = np.cumsum(d["cols"])                        # columns built after step k = ends[k-1]
        d["ncols"] = np.concatenate([[0], ends])
        colnnz = d["L"].sum(axis=0)
        colfill = d["fill"].sum(axis=0)
        d["nnz_built"] = np.concatenate([[0], np.cumsum([colnnz[s:e].sum() for s, e in zip(d["ncols"][:-1], ends)])])
        d["fill_built"] = np.concatenate([[0], np.cumsum([colfill[s:e].sum() for s, e in zip(d["ncols"][:-1], ends)])])
        cs = d["created"][d["fill"]]                       # creation scalar positions of all fill entries
        d["fill_created"] = np.array([(cs < nc).sum() for nc in d["ncols"]])
        # a fill entry is always created before its own column is built
        rr, cc = np.nonzero(d["fill"])
        assert (d["created"][rr, cc] < cc).all()
        assert d["fill_built"][-1] == d["fill_created"][-1] == d["n_fill"] and d["nnz_built"][-1] == d["nnz_L"]
        assert (d["fill_created"] >= d["fill_built"]).all()

    # --- figure ---
    fig = plt.figure(figsize=(13, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.3, 1.5], hspace=0.62, wspace=0.28)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.89, bottom=0.185)
    fig.suptitle("Building the Cholesky factor L column by column: the elimination order decides the fill-in",
                 fontsize=12.5, fontweight="bold")
    ymax = max(d["n_fill"] for d in data.values()) * 1.06
    ims, labels, hl_bands, lines_b, lines_c = {}, {}, {}, {}, {}
    for c, name in enumerate(NAMES):
        d = data[name]
        ax = fig.add_subplot(gs[0, c])
        ims[name] = ax.imshow(np.zeros((n, n, 4)), interpolation="nearest", extent=(0, n, n, 0))
        hl_bands[name] = ax.axvspan(0, 0, color=BAND[:3], alpha=BAND[3], lw=0)
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)
        ax.set_aspect("equal")
        ax.set_xticks([0, 48, 96, 144, 192])
        ax.set_yticks([0, 48, 96, 144, 192])
        ax.tick_params(labelsize=8)
        ax.grid(False)
        ax.set_title(name, fontsize=11.5, fontweight="bold", pad=14)
        if c == 0:
            ax.set_ylabel("row of L (in elimination order)", fontsize=9)
        ax.set_xlabel("column of L (in elimination order)", fontsize=9, labelpad=2)
        # strip above the panel: camera columns blue, landmark columns green
        strip = ax.inset_axes([0, 1.005, 1, 0.025])
        row = np.zeros((1, n, 3))
        col = 0
        for ncols, is_cam in zip(d["cols"], d["is_cam"]):
            row[0, col:col + ncols] = plt.matplotlib.colors.to_rgb(CAM_C if is_cam else LM_C)
            col += ncols
        strip.imshow(row, aspect="auto", interpolation="nearest")
        strip.axis("off")
        labels[name] = ax.text(0.5, -0.205, "", transform=ax.transAxes, ha="center", va="top", fontsize=8.8,
                               family="monospace", linespacing=1.35)

        axc = fig.add_subplot(gs[1, c])
        steps = np.arange(n_steps + 1)
        axc.plot(steps, d["fill_created"], color=PENDING[:3], lw=1, alpha=0.35)
        axc.plot(steps, d["fill_built"], color=RED[:3], lw=1, alpha=0.2)
        lines_c[name], = axc.plot([], [], color=PENDING[:3], lw=2.2, ls="--")
        lines_b[name], = axc.plot([], [], color=RED[:3], lw=2.2)
        axc.set(xlim=(0, n_steps), ylim=(0, ymax), xlabel="block step (one camera or one landmark)")
        axc.tick_params(labelsize=8)
        axc.xaxis.label.set_size(9)
        if c == 0:
            axc.set_ylabel("fill-in entries", fontsize=9)
        axc.set_title(f"final fill-in {d['n_fill']}", fontsize=10)

    handles = [Patch(color=BLACK[:3], label="L entry also in H"), Patch(color=RED[:3], label="L fill-in (zero in H)"),
               Patch(color=GRAY[:3], label="H entry, column not built yet"),
               Patch(color=PENDING[:3], label="fill-in created, column not built yet"),
               Patch(color=CAM_C, label="camera columns (6)"), Patch(color=LM_C, label="landmark columns (3)"),
               Line2D([], [], color=RED[:3], lw=2.2, label="fill-in drawn in built columns"),
               Line2D([], [], color=PENDING[:3], lw=2.2, ls="--", label="fill-in created by eliminations so far")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize=8.5, frameon=False)
    fig.text(0.5, 0.008, "Eliminating a variable creates fill-in between its neighbors, but each such nonzero of L sits in the column of a neighbor "
             "eliminated after that step, so it is drawn (red) only when that column is built.\n"
             f"Final: nnz(L) = 15819 / 6720 / 13452 and fill-in = 10107 / 1008 / 7740 for poses first / landmarks first / interleaved "
             f"({n} unknowns, nnz(lower H) = {data['poses first']['nnz_H']}, seed {args.seed}).", ha="center", fontsize=9)

    cols_idx = np.arange(n)[None, :]

    def image(d, k):
        nb = d["ncols"][k]
        img = np.zeros((n, n, 4))
        built = cols_idx < nb
        img[(d["Hl"] & ~built)] = GRAY
        pending = d["fill"] & ~built & (d["created"] < nb)
        img[pending] = PENDING
        img[d["L"] & ~d["fill"] & built] = BLACK
        img[d["fill"] & built] = RED
        return img

    def describe(d, k):
        who = "start: nothing eliminated yet" if k == 0 else (
            f"step {k}/{n_steps}: eliminate {'camera' if d['order'][k - 1][0] == 'c' else 'landmark'} {d['order'][k - 1][1]}")
        return (f"{who}\ncolumns built {d['ncols'][k]:3d}/{n}   nnz(L) so far {d['nnz_built'][k]:5d}\n"
                f"fill-in drawn {d['fill_built'][k]:5d}   fill-in created {d['fill_created'][k]:5d}")

    sequence = [0] * args.hold_start + list(range(n_steps + 1)) + [n_steps] * args.hold_end

    def update(frame):
        k = sequence[frame]
        for name in NAMES:
            d = data[name]
            ims[name].set_data(image(d, k))
            if 0 < k < n_steps:
                lo, hi = d["ncols"][k - 1], d["ncols"][k]
                hl_bands[name].set_x(lo)
                hl_bands[name].set_width(hi - lo)
            else:
                hl_bands[name].set_width(0)
            labels[name].set_text(describe(d, k))
            lines_b[name].set_data(np.arange(k + 1), d["fill_built"][:k + 1])
            lines_c[name].set_data(np.arange(k + 1), d["fill_created"][:k + 1])
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(sequence), interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Saved {len(sequence)} frames to {args.out}")
    # where the fill-in is created vs drawn, to quote in the doc: fraction of fill created / drawn by half of the steps
    for name in NAMES:
        d = data[name]
        h = n_steps // 2
        print(f"  {name:<16s} after {h}/{n_steps} steps: fill created {d['fill_created'][h]}, drawn {d['fill_built'][h]} "
              f"(of {d['n_fill']}); first step with red drawn: {int(np.argmax(d['fill_built'] > 0))}")


if __name__ == "__main__":
    main()
