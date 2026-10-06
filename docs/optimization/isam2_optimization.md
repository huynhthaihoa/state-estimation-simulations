# iSAM2

## 1. The big idea

**iSAM2 = Incremental Smoothing and Mapping 2**

> **iSAM2 continuously solves a SLAM optimization problem without restarting from scratch every time a new measurement arrives.**

Suppose the robot has estimated four poses, then moves one more step:

```text
Pose 1 → Pose 2 → Pose 3 → Pose 4 → Pose 5
  ↓        ↓        ↓        ↓        ↓
  x1       x2       x3       x4       x5
```

- **Naive optimizer:** "New information! Let's optimize $x_1, \ldots, x_5$ all over again." That is expensive.
- **iSAM2:** "Most of the previous solution is still good. I'll update only the parts that actually need significant changes."

---

## 2. Recap: what normal SLAM optimization does

Take four poses $x_1, x_2, x_3, x_4$ and measurements $z_{12}, z_{23}, z_{34}$ between them:

```text
x1 ─── x2 ─── x3 ─── x4
   z12    z23    z34
```

We want the poses that best explain all measurements:

```math
X^*=\arg\min_X \sum_i \|r_i(X)\|^2
```

Each odometry measurement says roughly how far the robot moved. The true step was 1 m each time, but every measurement carries a little noise:

```text
x1 ── 1.02 m ── x2 ── 0.97 m ── x3 ── 1.05 m ── x4
```

On its own, this chain has nothing to reconcile: placing the poses exactly 1.02, 0.97 and 1.05 m apart satisfies every measurement with zero residual. Optimization only has real work once a measurement is **redundant**, say a second sensor measures the distance from $x_1$ to $x_4$ directly:

```text
x1 ── 1.02 m ── x2 ── 0.97 m ── x3 ── 1.05 m ── x4
│                                               │
└─────────────────── 2.90 m ────────────────────┘
```

The chain adds up to 3.04 m, but the direct measurement says 2.90 m. They can't all be exactly right, so optimization finds the poses that make **all measurements reasonably happy at once**, weighting each by how much it is trusted.

---

## 3. The problem with doing this repeatedly

A robot running at 10 Hz grows the graph with every measurement:

```text
t1: x1
t2: x1 → x2
t3: x1 → x2 → x3
...
t1000: 1000 poses
```

Batch optimization re-solves the whole graph after every new measurement ("new measurement → optimize EVERYTHING → repeat"), so the cost keeps growing with the trajectory. **iSAM** and then **iSAM2** avoid this.

---

## 4. iSAM's basic idea

iSAM (incremental Smoothing and Mapping) does not re-solve all variables each time. It **reuses the previous factorization**.

Each solve is a Gauss-Newton step $H\Delta x=-g$ with $H=J^\top J$ (see [`gauss_newton.md`](gauss_newton.md)):

```text
Factor graph → Jacobian J → H = JᵀJ → Factorize H → solve Δx
```

The expensive part is often the **factorization**. When a new pose arrives, most of the old information hasn't changed, so **iSAM keeps the old factorization and updates it**.

---

## 5. Think of it like editing a huge spreadsheet

A spreadsheet has 100,000 rows and we change one cell. We don't delete it and rebuild everything; we update the affected calculations. iSAM2 applies the same philosophy to SLAM:

```text
Old solution → new measurement → identify affected variables → update those parts → reuse everything else
```

---

## 6. But what makes iSAM2 special?

iSAM2 improves on the original iSAM with three ideas:

1. **Bayes tree** (§7-§9)
2. **Selective relinearization** (§10-§11)
3. **Variable reordering** (§12)

### 6.1 iSAM vs. iSAM2, side by side

iSAM2 solves the same problem as iSAM, on the same factorization: iSAM's square-root information matrix $R$ and iSAM2's Bayes tree hold the same numbers (for the same variable ordering and linearization point), just organized differently. What changes is how much each one can do *incrementally*:

| | iSAM (Kaess et al. 2008) | iSAM2 (Kaess et al. 2012) |
| --- | --- | --- |
| Data structure | A sparse upper-triangular matrix $R$ | A Bayes tree: the same factorization, grouped into cliques (§7) |
| New measurement | Folded into $R$ as a new row, with Givens rotations | The cliques it touches, and everything above them, are removed, re-eliminated and put back (§8-§9) |
| Relinearization | Periodic batch step: every variable at once | Selective: only variables that moved past a threshold (§10) |
| Variable reordering | Periodic batch step: COLAMD over every variable | Incremental: CCOLAMD over just the re-eliminated part (§12) |
| Loop closure | The Givens sweep runs through most of $R$, and the fill-in it adds stays until the next batch step | A large affected region, but re-eliminated right away with a fresh ordering (§9, §12) |
| Updating the estimate | Back-substitution through $R$ | Starts at the root and stops in branches where the changes are negligible (§13) |
| Periodic batch steps | Required: the only way to relinearize and reorder | None |

One idea sits behind every row. In matrix form, iSAM can't cheaply tell which part of $R$ a relinearization or a reordering would touch, so it periodically does both for everything. The Bayes tree makes those dependencies explicit (each clique knows its parent), so iSAM2 can redo exactly the affected part at every step and never needs a batch step.

This repo implements the iSAM column minus COLAMD reordering, with loop closures handled by a full rebuild ([`isam_optimization.md` §14](isam_optimization.md#14-where-this-is-implemented-in-this-repo)), and only the tree-building part of the iSAM2 column (§19).

---

## 7. Bayes tree - the most important intuition

Quick recap first: **Sparse Cholesky factorization** splits $H$ into $H = LL^\top$ ($L$ lower-triangular), so solving $H\Delta x=-g$ becomes two cheap triangular solves instead of one matrix inversion. "Sparse" means most of $H$ is already zero - two poses only interact if a factor directly connects them - so the factorization skips arithmetic on entries it already knows are zero. The one catch: eliminating a variable can turn some of those zeros into nonzeros ("fill-in"), which is why elimination/variable order matters (§12). Full derivation in [`sparse_cholesky_factorization.md`](sparse_cholesky_factorization.md); a worked elimination example in [`elimination_tree.md`](elimination_tree.md).

So after linearization, SLAM hands you exactly the $H\Delta x=-g$ system above, and that sparse-elimination process is run for real. Conceptually:

```text
Factor graph
     ↓
Sparse matrix
     ↓
Cholesky/QR
     ↓
Bayes tree
```

The **Bayes tree** is a tree representation of that factorization - built by eliminating variables one at a time and recording which remaining variables each elimination step's result depends on - that makes incremental updates easier: given a new factor, iSAM2 can walk straight to the affected part of the tree instead of recomputing everything.

This doc only needs that one-sentence version. For the full mechanism - how elimination produces the tree, cliques, the probabilistic interpretation, and a worked loop-closure example on a correctly-drawn tree - see [`bayes_tree.md`](bayes_tree.md).

---

## 8. Intuition: only the part near the new variable changes

The robot adds $x_5$ and receives a measurement involving it. Usually the new information mainly affects the solution around $x_5$, so iSAM2 conceptually touches only the top of the tree instead of recomputing everything. Schematically (the newest variable lives at the root, §12):

```text
Before:                           After the update:

   [x4]  ← root                      [x4 x5]  ← re-eliminated root, now with x5
    │                                 │
 [x1 x2 x3]  ← old, reused         [x1 x2 x3]  ← old, reused
```

Batch optimization would instead recompute all of $x_1, \ldots, x_5$.

---

## 9. But what about loop closure?

Suppose the robot drives around:

```text
x1 ─ x2 ─ x3
          │
x6 ─ x5 ─ x4
```

It realizes: "Wait! $x_6$ is actually close to $x_1$." We add a loop-closure factor:

```text
x1 ─ x2 ─ x3
┆         │
x6 ─ x5 ─ x4

┆ = new loop-closure factor (x6 ↔ x1)
```

The new measurement can now affect **many old poses**. iSAM2 doesn't blindly update only $x_6$; it identifies the affected region of the Bayes tree and redoes that computation:

```text
normal odometry → small affected region → small update

loop closure    → large affected region → larger update
```

Ordinary updates are small and loop closures are the occasional large ones, which is a big part of why iSAM2 suits real-time SLAM.

---

## 10. Selective relinearization

A nonlinear problem $f(x)$ is approximated around the current estimate:

```math
f(x+\Delta x) \approx f(x)+J\Delta x
```

**Intuition:** $J$ is a tangent line, accurate only near the point where it was built. Relinearizing builds a fresh tangent at the new point.

If $x$ changes significantly, the Jacobian $J$ becomes outdated and we must relinearize (new Jacobians, then optimize again). The naive approach relinearizes **everything**. iSAM2 asks: "Which variables actually moved enough that their linearization is no longer accurate?"

```text
x1   x2   x3   x4   x5
 │    │    │    │    │
small small small BIG small
             ↑
       relinearize
```

Only the important variables are relinearized. This is **selective relinearization**.

It isn't free for that one variable, though. Relinearizing $x_4$ changes every factor that touches $x_4$, so every Bayes-tree clique containing $x_4$, and everything on the path from there up to the root, has to be re-eliminated, exactly as if a new factor had arrived there (§8-§9). In iSAM2 (Kaess et al. 2012), a variable counts as having moved "enough" when its change from the point it was last linearized at exceeds a threshold. The variables marked this way, together with the ones touched by new factors, make up the affected region of each update.

---

## 11. Why selective updates pay off

Take a SLAM graph with 10,000 poses. After a new measurement, maybe only 50 variables have changed significantly. A batch optimizer might effectively reconsider all 10,000, while iSAM2 focuses on the affected portion:

```text
Batch optimization          iSAM2

████████████████████████    ████
████████████████████████      ███
████████████████████████        ██
                                   █
```

The exact behavior depends on the graph and ordering, but that is the intuition.

---

## 12. Variable ordering is also crucial

The amount of computation in sparse Cholesky depends heavily on **variable ordering**, which decides two things at once:

- **Fill-in**: how dense the factorization gets ([sparse_cholesky_factorization.md](sparse_cholesky_factorization.md)).
- **Where each variable sits in the Bayes tree.** The last variable eliminated becomes the root, and an update's affected region is the path from the touched cliques up to the root (§8-§9).

A fixed order can be fine for one and terrible for the other. This repo's own example ([bayes_tree.md §15](bayes_tree.md#15-where-this-is-implemented-in-this-repo)) eliminates a square loop of 16 poses oldest first. Fill-in stays small (no separator larger than 2), but the tree comes out as a single chain with the first pose at the bottom, so a loop closure touching the first pose invalidates all 16 variables.

iSAM2 therefore reorders as it goes. Each time it re-eliminates the affected part of the tree, it picks a new order for just those variables with **constrained COLAMD (CCOLAMD)**: a fill-reducing ordering, constrained so that the variables touched by the newest factors are eliminated last. They end up near the root, where the next measurement (which usually involves the same recent variables) disturbs only a few cliques.

```text
new measurement → affected variables → reorder them → re-eliminate affected region → update Bayes tree
```

This aims to keep both the fill-in and the next update small.

---

## 13. The complete iSAM2 picture

Putting everything together, one iSAM2 update (simplified from Algorithm 1 of Kaess et al. 2012):

```text
             new measurements
                    │
                    ▼
 1. Add the new factors (and any new variables)
                    │
                    ▼
 2. Mark the affected variables:
      - those touched by the new factors
      - those that moved past the relinearization threshold (§10)
                    │
                    ▼
 3. Remove the Bayes-tree cliques containing them,
    plus everything above them up to the root
                    │
                    ▼
 4. Relinearize the marked variables' factors (§10)
                    │
                    ▼
 5. Reorder the removed variables (CCOLAMD, §12) and
    re-eliminate them into a new top of the tree
                    │
                    ▼
 6. Update the estimate from the root downward, stopping
    in branches where the changes become negligible
                    │
                    ▼
           updated SLAM state ──► wait for the next measurements
```

Everything below the removed top of the tree keeps its factorization untouched (its estimates can still be updated in step 6).

**Intuition for step 6:**
- Each clique's estimate is computed from its parent's: in symbols, child $`\delta = (d - R_{sep}\,\delta_{parent}) / R_{cc}`$, where $d$ is the clique's own right-hand side and $R_{sep}$ couples it to its parent.
- If the parent's $\delta$ barely moved and the clique's own $d$ is unchanged, the child's $\delta$ does not change either.
- So the update can stop at that branch, and everything below it is skipped safely.
- (Conceptual only: this repo does not implement iSAM2.)

---

## 14. iSAM2 vs EKF

**EKF-SLAM** is fundamentally a **filtering** approach: "I maintain my current estimate and uncertainty, and update it when a measurement arrives." Old information is compressed into the current state.

```text
prediction → measurement → EKF update → new estimate
```

**iSAM2** is an **incremental smoothing/optimization** approach: "I keep the history and continuously optimize it." It retains historical variables and constraints.

```text
x1 ─ x2 ─ x3 ─ x4 ─ x5
│         │         │
measurements + loop closures
          ↓
   continuously optimize
```

---

## 15. iSAM2 vs batch optimization

|                        | Batch optimization    | iSAM2                |
| ---------------------- | --------------------- | -------------------- |
| New measurement        | Reoptimize everything | Incrementally update |
| History                | Kept                  | Kept                 |
| Nonlinear optimization | Yes                   | Yes                  |
| Sparse factorization   | Yes                   | Yes                  |
| Relinearization        | Broad                 | Selective            |
| Variable ordering      | Important             | Dynamically managed  |
| Loop closure           | Full re-solve, same as any other step | Large affected region; can approach a full re-solve (§9, §19) |
| Real-time suitability  | Lower                 | Higher               |

The key difference isn't a different SLAM objective; it is **how intelligently the solution is updated**.

---

## 16. Ordinary measurement vs. loop closure, in spreadsheet terms

Continuing the spreadsheet of §5:

- **Ordinary measurement:** like changing one cell. We find the few formulas that depend on it and recompute only those.
- **Loop-closure measurement:** like changing a cell that many formulas depend on. A much larger region must be recomputed, but still not the whole sheet.

---

## 17. A common misconception: iSAM2 only optimizes the newest pose

That is not correct. iSAM2 maintains a **global smoothing solution**. If a loop closure says $x_{100} \approx x_1$, the correction can propagate backward through the trajectory, so old poses can change:

```text
x1 ← x2 ← x3 ← ... ← x100
↑                      ↑
└──── loop closure ────┘
```

The cleverness is that iSAM2 determines **which parts need computational attention** rather than blindly recomputing the entire problem.

---

## 18. The one-sentence mental model

> **iSAM2 is an incremental nonlinear least-squares solver for SLAM that maintains a Bayes-tree factorization and efficiently updates only the parts of the solution affected by new information.**

**Bayes tree + selective relinearization + variable reordering → incremental update**

---

## 19. Where this is implemented in this repo

Partially - and it's worth being precise about which part:
- [`bayes_tree_construction.py`](../../use_numpy/bayes_tree_construction.py) builds the elimination tree underlying this doc's §7 Bayes tree (symbolic elimination over this repo's pose-graph topology, via `symbolic_eliminate`/`bayes_tree_affected_path` in `utils.py`).
- It keeps one node per variable and never merges them into cliques, so its affected counts are counts of variables, not cliques. The tree is built once from all edges, the loop-closure edge included, and then queried per edge.
- It quantifies §9's "small vs. large affected region" claim with a computed example instead of only prose: with the default 16 nodes (`--nodes-per-side 4`), the newest odometry edge affects 2/16 variables and the loop-closure edge affects 16/16.
- A genuinely useful finding (details in [`bayes_tree.md` §15](bayes_tree.md#15-where-this-is-implemented-in-this-repo)): with a *fixed* elimination order (oldest node first, since §12's variable reordering is explicitly not implemented), this repo's loop-closure edge produces the worst possible case - the entire tree, not just a subtree, gets invalidated.

**§10 selective relinearization and §12 variable reordering (CCOLAMD) remain unimplemented** - no numeric solve is integrated with the tree above. That's still [`pose_graph_incremental.py`](../../use_numpy/pose_graph_incremental.py) (both `use_numpy/` and `use_manif/`)'s job, and it implements the original **iSAM v1** mechanism described in [`isam_optimization.md`](isam_optimization.md) instead - incremental Givens-rotation QR row insertion into a running square-root-information matrix, plus periodic/loop-closure-triggered full relinearization, with no Bayes tree involved at all. Its own module docstring explicitly calls out the Bayes tree and COLAMD as out of scope; see [`isam_optimization.md` §14](isam_optimization.md#14-where-this-is-implemented-in-this-repo) for exactly where that line is drawn.

So this page's Bayes tree now has a real, runnable counterpart, but iSAM2 as a whole - the tree, selective relinearization, and dynamic reordering working together against an actual numeric solve - is still the conceptual target no single script in `use_numpy/`/`use_manif/` reaches.

---

## 20. References

1. Kaess, M., Johannsson, H., Roberts, R., Ila, V., Leonard, J. J., & Dellaert, F. (2012). *iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree*. International Journal of Robotics Research, 31(2), 216–235. https://doi.org/10.1177/0278364911430419 - the Bayes tree, fluid/selective relinearization, and dynamic variable reordering this whole doc walks through.
2. Kaess, M., Ranganathan, A., & Dellaert, F. (2008). *iSAM: Incremental Smoothing and Mapping*. IEEE Transactions on Robotics, 24(6), 1365–1378. https://doi.org/10.1109/TRO.2008.2006706 - the predecessor algorithm (incremental QR updates, no Bayes tree) that [`isam_optimization.md`](isam_optimization.md) covers and that `pose_graph_incremental.py` actually implements.

See also [`bayes_tree.md`](bayes_tree.md) for the full elimination-to-tree mechanism behind §7, and [`isam_optimization.md`](isam_optimization.md) for the non-Bayes-tree predecessor algorithm this doc builds on.
