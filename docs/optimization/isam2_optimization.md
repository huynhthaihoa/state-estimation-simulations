# iSAM2

## 1. The big idea

**iSAM2 = Incremental Smoothing and Mapping 2**

The easiest way to think about it is:

> **iSAM2 is a way to continuously solve a SLAM optimization problem without starting the whole optimization from scratch every time a new sensor measurement arrives.**

Imagine your robot has estimated:

```text
Pose 1 → Pose 2 → Pose 3 → Pose 4
  ↓        ↓        ↓        ↓
  x1       x2       x3       x4
```

Then the robot moves one more step:

```text
Pose 1 → Pose 2 → Pose 3 → Pose 4 → Pose 5
  ↓        ↓        ↓        ↓        ↓
  x1       x2       x3       x4       x5
```

A naive optimizer might say:

> "New information! Let's optimize $x_1$, $x_2$, $x_3$, $x_4$, $x_5$ all over again."

That's expensive.

**iSAM2 says:**

> "Most of the previous solution is still good. I'll update only the parts that actually need significant changes."

That's the core intuition.

---

## 2. First remember what normal SLAM optimization does

Suppose your robot has poses:

$$x_1,x_2,x_3,x_4$$

and measurements between them:

$$z_{12},z_{23},z_{34}$$

Your factor graph looks like:

```text
x1 ─── x2 ─── x3 ─── x4
   z12    z23    z34
```

You want to find the poses that best explain all measurements:

```math
X^*=\arg\min_X \sum_i \|r_i(X)\|^2
```

For example, each odometry measurement says roughly how far the robot moved between two poses. The true step was 1 m each time, but every measurement carries a little noise:

```text
x1 ── 1.02 m ── x2 ── 0.97 m ── x3 ── 1.05 m ── x4
```

On its own, this chain has nothing to reconcile: placing the poses exactly 1.02, 0.97 and 1.05 m apart satisfies every measurement perfectly, with zero residual. Optimization only has real work to do once a measurement is **redundant** - say a second sensor measures the distance from $x_1$ to $x_4$ directly:

```text
x1 ── 1.02 m ── x2 ── 0.97 m ── x3 ── 1.05 m ── x4
│                                               │
└─────────────────── 2.90 m ────────────────────┘
```

The chain adds up to 3.04 m, but the direct measurement says 2.90 m. They can't all be exactly right, so optimization finds the set of poses that makes **all measurements reasonably happy at the same time**, weighting each one by how much it's trusted.

---

## 3. The problem with doing this repeatedly

Now imagine a robot running at 10 Hz.

Every time a new measurement arrives:

```text
t1: x1
t2: x1 → x2
t3: x1 → x2 → x3
t4: x1 → x2 → x3 → x4
...
t1000: 1000 poses
```

If you use batch optimization:

```text
new measurement
      ↓
optimize EVERYTHING
      ↓
new measurement
      ↓
optimize EVERYTHING
      ↓
new measurement
      ↓
optimize EVERYTHING
```

The computational cost becomes increasingly painful.

This is where **iSAM** and then **iSAM2** come in.

---

## 4. iSAM's basic idea

iSAM stands for:

> **incremental Smoothing and Mapping**

Instead of repeatedly solving:

$$\text{all variables}$$

it tries to **reuse the previous factorization**.

Remember the Gauss-Newton step you learned:

$$H\Delta x=-g$$

where

$$H=J^\top J$$

and then we solve this linear system.

In batch SLAM:

```text
Factor graph
     ↓
Jacobian J
     ↓
H = JᵀJ
     ↓
Factorize H
     ↓
solve Δx
```

The expensive part is often the **factorization**.

When a new pose arrives, most of the old information hasn't changed.

So why throw away the old factorization?

**iSAM tries to reuse it.**

---

## 5. Think of it like editing a huge spreadsheet

Imagine you have a huge spreadsheet:

```text
100,000 rows
```

You change one cell.

Would you:

> Delete the entire spreadsheet and rebuild it?

No.

You would update the affected calculations.

iSAM2 applies a similar philosophy to SLAM.

```text
Old solution
     ↓
new measurement
     ↓
identify affected variables
     ↓
update those parts
     ↓
reuse everything else
```

---

## 6. But what makes iSAM2 special?

This is where **iSAM2** improves upon the original iSAM.

The key concepts are:

1. **Bayes tree** (§7-§9)
2. **Selective relinearization** (§10-§11)
3. **Variable reordering** (§12)

These three ideas are the heart of iSAM2.

### 6.1 iSAM vs. iSAM2, side by side

iSAM2 solves the same problem as iSAM, on the same factorization: iSAM's square-root information matrix $R$ and iSAM2's Bayes tree hold the same numbers, just organized differently. What changes is how much each one can do *incrementally*:

| | iSAM (Kaess et al. 2008) | iSAM2 (Kaess et al. 2012) |
| --- | --- | --- |
| Data structure | A sparse upper-triangular matrix $R$ | A Bayes tree: the same factorization, grouped into cliques (§7) |
| New measurement | Folded into $R$ as a new row, with Givens rotations | The cliques it touches, and everything above them, are removed, re-eliminated and put back (§8-§9) |
| Relinearization | Periodic batch step: every variable at once | Selective: only variables that moved past a threshold (§10) |
| Variable reordering | Periodic batch step: COLAMD over every variable | Incremental: CCOLAMD over just the re-eliminated part (§12) |
| Loop closure | The Givens sweep runs through most of $R$, and the fill-in it adds stays until the next batch step | A large affected region, but re-eliminated right away with a fresh ordering (§9, §12) |
| Updating the estimate | Back-substitution through $R$ | Starts at the root and stops in branches where the changes are negligible (§13) |
| Periodic batch steps | Required: the only way to relinearize and reorder | None |

One idea sits behind every row. In matrix form, iSAM can't cheaply tell which part of $R$ a relinearization or a reordering would touch, so it periodically does both for everything. The Bayes tree makes those dependencies explicit - each clique knows its parent - so iSAM2 can redo exactly the affected part at every step, and never needs a batch step.

This repo implements the iSAM column ([`isam_optimization.md` §14](isam_optimization.md#14-where-this-is-implemented-in-this-repo)) and only the tree-building part of the iSAM2 column (§19).

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

## 8. Here's the really useful intuition

Suppose your robot adds a new pose:

```text
x1 ─ x2 ─ x3 ─ x4 ─ x5
```

and receives a new measurement involving:

```text
x5
```

Usually, the new information primarily affects the part of the solution around $x_5$.

So iSAM2 might conceptually do:

```text
             OLD
        ┌─────────────┐
        │ x1 x2 x3 x4 │
        └─────────────┘
                 \
                  x5 ← NEW
```

Instead of:

```text
x1 x2 x3 x4 x5
 \  \  \  \ /
   RECOMPUTE ALL
```

it updates only the relevant part of the Bayes tree.

---

## 9. But what about loop closure?

This is where things get interesting.

Suppose your robot drives around:

```text
x1 ─ x2 ─ x3
          │
x6 ─ x5 ─ x4
```

The robot realizes:

> "Wait! $x_6$ is actually close to $x_1$."

So you add a loop-closure factor:

```text
x1 ─ x2 ─ x3
┆         │
x6 ─ x5 ─ x4

┆ = new loop-closure factor (x6 ↔ x1)
```

Now the new measurement can affect **many old poses**.

iSAM2 recognizes this.

It doesn't blindly update only $x_6$.

Instead, it identifies the affected region of the Bayes tree and redoes the necessary computation.

So:

```text
normal odometry
      ↓
small affected region
      ↓
small update
```

but:

```text
loop closure
      ↓
large affected region
      ↓
larger update
```

This is one reason iSAM2 works well for real-time SLAM.

---

## 10. Selective relinearization

This is another very important idea.

Remember nonlinear optimization:

$$f(x)$$

We approximate it around the current estimate:

$$f(x+\Delta x) \approx f(x)+J\Delta x$$

But if $x$ changes significantly, the Jacobian $J$ becomes outdated.

So periodically we need:

```text
old estimate
    ↓
relinearize
    ↓
new Jacobians
    ↓
optimize again
```

The naive approach would relinearize **everything**.

iSAM2 asks:

> "Which variables actually moved enough that their linearization is no longer accurate?"

For example:

```text
x1   x2   x3   x4   x5
 │    │    │    │    │
small small small BIG small
             ↑
       relinearize
```

Only the important variables need relinearization.

This is called **selective relinearization**.

It isn't free for that one variable, though. Relinearizing $x_4$ changes every factor that touches $x_4$, so every Bayes-tree clique containing $x_4$ - and everything on the path from there up to the root - has to be re-eliminated, exactly as if a new factor had arrived there (§8-§9). In iSAM2 (Kaess et al. 2012), a variable counts as having moved "enough" when its change from the point it was last linearized at exceeds a threshold. The variables marked this way, together with the ones touched by new factors, make up the affected region of each update.

---

## 11. Why is that powerful?

Imagine a SLAM graph containing 10,000 poses.

After a new measurement, maybe only $50$ variables have changed significantly.

A batch optimizer might effectively reconsider all 10,000.

iSAM2 tries to focus computation on the affected portion.

Conceptually:

```text
Batch optimization

████████████████████████████████
████████████████████████████████
████████████████████████████████


iSAM2

████
  ███
    ██
       █
```

Obviously the exact computational behavior depends on the graph and ordering, but that's the intuition.

---

## 12. Variable ordering is also crucial

Remember sparse Cholesky?

The amount of computation depends heavily on **variable ordering**.

The order decides two things at once:

- **Fill-in**: how dense the factorization gets ([sparse_cholesky_factorization.md](sparse_cholesky_factorization.md)).
- **Where each variable sits in the Bayes tree.** The last variable eliminated becomes the root, and an update's affected region is the path from the touched cliques up to the root (§8-§9).

A fixed order can be fine for one and terrible for the other. This repo's own example ([bayes_tree.md §15](bayes_tree.md#15-where-this-is-implemented-in-this-repo)) eliminates a square loop of 16 poses oldest first. Fill-in stays small (no separator larger than 2), but the tree comes out as a single chain with the first pose at the bottom, so a loop closure touching the first pose invalidates all 16 variables.

iSAM2 therefore reorders as it goes. Each time it re-eliminates the affected part of the tree, it picks a new order for just those variables with **constrained COLAMD (CCOLAMD)**: a fill-reducing ordering, constrained so that the variables touched by the newest factors are eliminated last. They end up near the root, where the next measurement - which usually involves the same recent variables - disturbs only a few cliques.

Very roughly:

```text
new measurement
      ↓
affected variables
      ↓
reorder them
      ↓
re-eliminate affected region
      ↓
update Bayes tree
```

This keeps both the fill-in and the next update small.

---

## 13. The complete iSAM2 picture

Now put everything together. One iSAM2 update, following Algorithm 1 of Kaess et al. (2012):

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

Everything below the removed top of the tree is reused untouched.

---

## 14. iSAM2 vs EKF

This distinction is particularly useful for understanding modern SLAM systems.

### EKF-SLAM

Think:

> "I maintain my current estimate and uncertainty, and update it when a measurement arrives."

```text
prediction
    ↓
measurement
    ↓
EKF update
    ↓
new estimate
```

It is fundamentally a **filtering** approach.

Old information is compressed into the current state.

---

### iSAM2

Think:

> "I keep the history and continuously optimize it."

```text
x1 ─ x2 ─ x3 ─ x4 ─ x5
│         │         │
measurements + loop closures
          ↓
   continuously optimize
```

So iSAM2 is an **incremental smoothing/optimization** approach.

It retains historical variables and constraints.

---

## 15. iSAM2 vs batch optimization

This is probably the most intuitive comparison:

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

The key difference isn't that iSAM2 uses a fundamentally different SLAM objective.

It's mostly about **how intelligently it updates the solution**.

---

## 16. A very simple analogy

Imagine you're solving a giant jigsaw puzzle.

### Batch optimization

Every time someone gives you a new puzzle piece:

> "Let's throw away our current arrangement and solve the entire puzzle again."

### iSAM2

Instead:

> "Where does this new piece connect?"

Then:

```text
new piece
   ↓
find affected region
   ↓
rearrange that region
   ↓
keep the rest
```

If someone gives you a **loop-closure piece**, it may force you to rearrange a much larger region.

That's essentially the spirit of iSAM2.

---

## 17. One subtle but important point

Don't think:

> **iSAM2 only optimizes the newest pose.**

That's not correct.

It maintains a **global smoothing solution**.

If a loop closure says:

$$
x_{100} \approx x_1
$$

then the correction can propagate backward through the trajectory:

```text
x1 ← x2 ← x3 ← ... ← x100
↑                      ↑
└──── loop closure ────┘
```

So old poses can change.

The cleverness is that iSAM2 determines **which parts need computational attention** rather than blindly recomputing the entire problem.

---

## 18. The one-sentence mental model

If you remember only one thing:

> **iSAM2 is an incremental nonlinear least-squares solver for SLAM that maintains a Bayes-tree factorization and efficiently updates only the parts of the solution affected by new information.**

And the three ideas to remember, plus what they add up to, are:

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
