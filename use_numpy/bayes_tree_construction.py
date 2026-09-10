'''
Bayes-tree construction and affected-region query for a pose graph -- the
docs/optimization/bayes_tree.md concept made concrete: build the actual tree
by symbolic elimination (`symbolic_eliminate`, utils.py) over this repo's
canonical square-loop pose-graph topology (`pose_graph.py`'s
generate_ground_truth_trajectory: a ring of `4 * nodes_per_side` nodes, one
odometry edge between each pair of consecutive nodes, one loop-closure edge
from the last node back to the first), then asks `bayes_tree_affected_path`
which cliques a new factor would invalidate, for two scenarios: an ordinary
odometry edge vs. the loop-closure edge.

Pure index/topology bookkeeping -- no pose math, SE(3), or Lie algebra is
involved at any point, so unlike every other script in this repo there is no
use_manif/ counterpart: the result would be structurally identical, since
the tree only depends on which node indices a factor connects, never on the
noisy relative-pose values themselves.

Elimination order is fixed as oldest-first (node 0 first, the newest node
last/root) -- a simple, deterministic choice, not iSAM2's dynamic COLAMD
reordering (out of scope here, see docs/optimization/isam2_optimization.md
Section 12). Worked out by hand before writing this script: under that
order, the loop-closure edge connects the *first*-eliminated node (the
deepest possible leaf) to the root, so the resulting tree is a straight
chain of ever-growing separators, and the loop closure invalidates the
*entire* chain -- the worst case a fill-reducing reordering like COLAMD
exists specifically to avoid. That worst case isn't hidden here; it's the
whole point of the comparison this script prints and plots.

This implements the Bayes tree's *symbolic construction* and *affected-
region query* only: no numeric fluid-relinearization solve (that is
pose_graph_incremental.py's square-root-SAM update, iSAM v1, not iSAM2) and
no dynamic variable reordering.
'''

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import symbolic_eliminate, bayes_tree_affected_path
from pose_graph import generate_ground_truth_trajectory


def ring_edges(n_poses):
    """Index-only edges for this repo's canonical closed-loop pose graph:
    one odometry edge between each pair of consecutive nodes, plus one
    loop-closure edge from the last node back to the first -- exactly
    pose_graph.py's simulate_noisy_edges topology, without the noisy Z_ij
    values this script never needs.
    Arguments:
        n_poses: number of nodes in the ring
    Returns:
        odom_edges: list of (i, i+1) pairs
        loop_edge: (n_poses - 1, 0)
    """
    odom_edges = [(k, k + 1) for k in range(n_poses - 1)]
    loop_edge = (n_poses - 1, 0)
    return odom_edges, loop_edge


def layout_tree(children, root):
    """Simple recursive layout, root on the left growing rightward: leaves
    get sequential y-positions in DFS order, internal nodes get y = mean of
    their children's y; x = depth from the root. Horizontal rather than
    vertical so a long chain (this script's default topology -- see the
    module docstring) reads left-to-right across a wide figure instead of
    top-to-bottom in a short one. Kept generic rather than hardcoded to that
    straight-chain shape, so it stays correct if the topology or
    elimination order ever produces real branching.
    Arguments:
        children: dict {variable: list of child variables}
        root: the tree's root variable
    Returns:
        pos: dict {variable: (x, y)}
    """
    pos = {}
    next_y = [0]

    def visit(v, depth):
        kids = children.get(v, [])
        if not kids:
            pos[v] = (depth, next_y[0])
            next_y[0] += 1
        else:
            for c in kids:
                visit(c, depth + 1)
            pos[v] = (depth, sum(pos[c][1] for c in kids) / len(kids))

    visit(root, 0)
    return pos


def plot_scenario(ax, parent, pos, affected, touched_vars, title, node_size, label_nodes):
    """Draws one Bayes-tree subplot: every clique/edge in gray, the
    affected region (the root-ward path from `touched_vars`) highlighted in
    red, and the touched variables themselves marked with a star.
    Arguments:
        ax: matplotlib axes to draw into
        parent: {variable: parent variable or None}, from symbolic_eliminate
        pos: {variable: (x, y)}, from layout_tree
        affected: set of variables on the affected root-ward path(s)
        touched_vars: the variables the new factor connects
        title: subplot title (the affected-fraction count is appended)
        node_size: marker area, scaled down for long chains so labels don't collide
        label_nodes: whether to draw per-node "x<i>" text (skipped past a size
                     where it would just overlap illegibly)
    """
    for v, p in parent.items():
        if p is None:
            continue
        (x0, y0), (x1, y1) = pos[v], pos[p]
        on_path = v in affected and p in affected
        ax.plot([x0, x1], [y0, y1], color="tab:red" if on_path else "0.75",
                 linewidth=2.5 if on_path else 1.0, zorder=1)

    for v, (x, y) in pos.items():
        on_path = v in affected
        ax.scatter(x, y, s=node_size * (1.8 if on_path else 1.0),
                   color="tab:red" if on_path else "0.85",
                   edgecolor="black", linewidth=0.6, zorder=2)
        if label_nodes:
            ax.annotate(f"x{v}", (x, y), ha="center", va="center", fontsize=7, zorder=3)

    for v in touched_vars:
        x, y = pos[v]
        ax.scatter(x, y, s=node_size * 3.2, marker="*", color="gold", edgecolor="black", zorder=4)

    ax.set_title(f"{title}\n{len(affected)}/{len(pos)} cliques affected")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes-per-side", type=int, default=4,
                         help="Nodes per side of the ring pose graph (same topology as pose_graph_incremental.py, "
                              "kept small by default since this plots an inspectable diagram, not a timing benchmark)")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")
    args = parser.parse_args()

    gt_poses = generate_ground_truth_trajectory(side_length=2.0, nodes_per_side=args.nodes_per_side)
    n_poses = len(gt_poses)
    odom_edges, loop_edge = ring_edges(n_poses)

    order = list(range(n_poses))  # oldest-first: node 0 eliminated first, newest node becomes the root
    separator, parent = symbolic_eliminate(n_poses, odom_edges + [loop_edge], order)
    root = next(v for v, p in parent.items() if p is None)

    children = {}
    for v, p in parent.items():
        if p is not None:
            children.setdefault(p, []).append(v)

    print(f"Built a Bayes tree over {n_poses} nodes ({args.nodes_per_side} per side of the loop), "
          f"oldest-first elimination order (root = node {root}).")
    print(f"Largest separator: {max(len(s) for s in separator.values())} "
          "(fill-in caused by the loop-closure edge)")

    newest_odom_edge = odom_edges[-1]
    affected_odom = bayes_tree_affected_path(parent, newest_odom_edge)
    affected_loop = bayes_tree_affected_path(parent, loop_edge)

    print(f"\nAffected region if odometry edge {newest_odom_edge} arrives: "
          f"{len(affected_odom)}/{n_poses} cliques ({100 * len(affected_odom) / n_poses:.1f}%)")
    print(f"Affected region if loop-closure edge {loop_edge} arrives:     "
          f"{len(affected_loop)}/{n_poses} cliques ({100 * len(affected_loop) / n_poses:.1f}%)")
    print("\nSame fixed elimination order, wildly different cost: this is exactly why real "
          "iSAM2 needs dynamic reordering (COLAMD) instead of a fixed one -- not implemented "
          "here, see docs/optimization/isam2_optimization.md Section 12.")

    pos = layout_tree(children, root)

    # A long chain gets a wider figure and progressively smaller markers/labels
    # so nodes don't overlap; this is an inspectable diagram, not a fixed-size
    # plot meant to stay crisp at arbitrary --nodes-per-side.
    fig_width = min(28.0, max(12.0, n_poses * 0.28))
    node_size = max(20, 160 - 2 * n_poses)
    label_nodes = n_poses <= 32

    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 7))
    plot_scenario(axes[0], parent, pos, affected_odom, newest_odom_edge,
                  "New odometry edge (local update)", node_size, label_nodes)
    plot_scenario(axes[1], parent, pos, affected_loop, loop_edge,
                  "Loop-closure edge (global update)", node_size, label_nodes)
    fig.suptitle("Bayes-tree affected region: small (odometry) vs. large (loop closure)")
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
