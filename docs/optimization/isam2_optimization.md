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

> "New information! Let's optimize x1, x2, x3, x4, x5 all over again."

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
      z12     z23    z34
```

You want to find the poses that best explain all measurements:

$$X^*=\arg\min_X \sum_i \|r_i(X)\|^2$$

For example:

```text
                measurement
x1 ───────────────────────── x2
        "you moved 1 m"

x2 ───────────────────────── x3
        "you moved 1 m"

x3 ───────────────────────── x4
        "you moved 1 m"
```

But measurements contain noise.

So maybe the robot actually estimates:

```text
x1 ---- 1.02m ---- x2
x2 ---- 0.97m ---- x3
x3 ---- 1.05m ---- x4
```

Optimization finds the set of poses that makes **all measurements reasonably happy at the same time**.

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

> **incremental Smoothing And Mapping**

Instead of repeatedly solving:

$$\text{all variables}$$

it tries to **reuse the previous factorization**.

Remember the Gauss-Newton step you learned:

$$H\Delta x=-g$$

where

$$H=J^TJ$$

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

### ① Bayes tree

### ② Variable reordering

### ③ Selective relinearization

These three ideas are the heart of iSAM2.

---

## 7. Bayes tree — the most important intuition

You already learned about **Sparse Cholesky factorization**.

This is closely connected.

After linearization, SLAM gives you something like:

$$H\Delta x=-g$$

Because SLAM is sparse, we don't want to treat $H$ as a giant dense matrix.

We factorize it efficiently, and conceptually:

```text
Factor graph
     ↓
Sparse matrix
     ↓
Cholesky / QR
     ↓
Bayes tree
```

The **Bayes tree** is a tree representation of that factorization — built by eliminating
variables one at a time and recording which remaining variables each elimination step's
result depends on — that makes incremental updates easier: given a new factor, iSAM2 can walk
straight to the affected part of the tree instead of recomputing everything.

This doc only needs that one-sentence version. For the full mechanism — how elimination
produces the tree, cliques, the probabilistic interpretation, and a worked loop-closure
example on a correctly-drawn tree — see [`bayes_tree.md`](bayes_tree.md).

---

## 8. Here's the really useful intuition

Suppose your robot adds a new pose:

```text
x1 — x2 — x3 — x4 — x5
```

and receives a new measurement involving:

```text
x5
```

Usually, the new information primarily affects the part of the solution around $x5$.

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
 \  \  \  \  /
   RECOMPUTE ALL
```

it updates only the relevant part of the Bayes tree.

---

## 9. But what about loop closure?

This is where things get interesting.

Suppose your robot drives around:

```text
        x1 ─ x2 ─ x3
        |           |
        |           |
        x6 ─ x5 ─ x4
```

The robot realizes:

> "Wait! x6 is actually close to x1."

So you add a loop-closure factor:

```text
x1 ─ x2 ─ x3
│           │
x6 ─ x5 ─ x4
```

Now the new measurement can affect **many old poses**.

iSAM2 recognizes this.

It doesn't blindly update only x6.

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

---

## 11. Why is that powerful?

Imagine a SLAM graph containing:

$$
10,000
$$

poses.

After a new measurement, maybe only:

$$
50
$$

variables have changed significantly.

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

For example:

```text
x1 x2 x3 x4 x5
```

might produce much less fill-in than a bad ordering.

iSAM2 therefore dynamically manages the ordering of variables.

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

This helps preserve sparsity.

---

## 13. The complete iSAM2 picture

Now put everything together:

```text
        Sensor measurements
                │
                ▼
          Factor graph
                │
                ▼
       Nonlinear optimization
                │
                ▼
          Linearization
                │
                ▼
       Sparse linear system
                │
                ▼
        Bayes tree / factorization
                │
       ┌────────┴─────────┐
       │                  │
 new measurement     robot estimate
       │                  │
       ▼                  │
 affected variables       │
       │                  │
       ▼                  │
 selective                │
 relinearization          │
       │                  │
       ▼                  │
 variable ordering        │
       │                  │
       └────────┬─────────┘
                ▼
       Incremental update
                │
                ▼
         updated SLAM state
```

---

## 14. iSAM2 vs EKF

This distinction is particularly useful for your SLAM learning.

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

So iSAM2 is an **incremental smoothing / optimization** approach.

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
| Loop closure           | Expensive             | Efficiently handled  |
| Real-time suitability  | Lower                 | Higher               |

The key difference isn't that iSAM2 uses a fundamentally different SLAM objective.

It's mostly about **how intelligently it updates the solution**.

---

## 16. A very simple analogy

Imagine you're solving a giant jigsaw puzzle.

### Batch optimization

Every time someone gives you a new puzzle piece:

> "Let's throw away our current arrangement and solve the entire puzzle again."

😅

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
↑                         ↑
└──── loop closure ───────┘
```

So old poses can change.

The cleverness is that iSAM2 determines **which parts need computational attention** rather than blindly recomputing the entire problem.

---

## 18. The one-sentence mental model

If you remember only one thing:

> **iSAM2 is an incremental nonlinear least-squares solver for SLAM that maintains a Bayes-tree factorization and efficiently updates only the parts of the solution affected by new information.**

And the three keywords to remember are:

**Bayes tree → selective relinearization → incremental update**

---

## 19. Where this is implemented in this repo

Partially — and it's worth being precise about which part rather than leaving it as one bare
claim. [`bayes_tree_construction.py`](../../use_numpy/bayes_tree_construction.py) builds this
doc's §7 Bayes tree for real (symbolic elimination over this repo's pose-graph topology, via
`symbolic_eliminate`/`bayes_tree_affected_path` in `utils.py`) and quantifies §9's "small vs.
large affected region" claim with a computed example instead of only prose — see
[`bayes_tree.md` §15](bayes_tree.md#15-where-this-is-implemented-in-this-repo) for the details,
including a genuinely useful finding: with a *fixed* elimination order (oldest node first, since
§12's variable reordering is explicitly not implemented), this repo's loop-closure edge produces
the worst possible case — the entire tree, not just a subtree, gets invalidated.

**§10 selective relinearization and §12 variable reordering (COLAMD) remain unimplemented** — no
numeric solve is integrated with the tree above. That's still
[`pose_graph_incremental.py`](../../use_numpy/pose_graph_incremental.py) (both `use_numpy/` and
`use_manif/`)'s job, and it implements the original **iSAM v1** mechanism described in
[`isam_optimization.md`](isam_optimization.md) instead — incremental Givens-rotation QR row
insertion into a running square-root-information matrix, plus periodic/loop-closure-triggered
full relinearization, with no Bayes tree involved at all. Its own module docstring explicitly
calls out the Bayes tree and COLAMD as out of scope; see
[`isam_optimization.md` §14](isam_optimization.md#14-where-this-is-implemented-in-this-repo) for
exactly where that line is drawn.

So this page's Bayes tree now has a real, runnable counterpart, but iSAM2 as a whole — the tree,
selective relinearization, and dynamic reordering working together against an actual numeric
solve — is still the conceptual target no single script in `use_numpy/`/`use_manif/` reaches.

---

## 20. References

1. Kaess, M., Johannsson, H., Roberts, R., Ila, V., Leonard, J. J., & Dellaert, F. (2012). *iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree*. International Journal of Robotics Research, 31(2), 216–235. https://doi.org/10.1177/0278364911430419 - the Bayes tree, fluid/selective relinearization, and dynamic variable reordering this whole doc walks through.
2. Kaess, M., Ranganathan, A., & Dellaert, F. (2008). *iSAM: Incremental Smoothing and Mapping*. IEEE Transactions on Robotics, 24(6), 1365–1378. https://doi.org/10.1109/TRO.2008.2006706 - the predecessor algorithm (incremental QR updates, no Bayes tree) that [`isam_optimization.md`](isam_optimization.md) covers and that `pose_graph_incremental.py` actually implements.

See also [`bayes_tree.md`](bayes_tree.md) for the full elimination-to-tree mechanism behind
§7, and [`isam_optimization.md`](isam_optimization.md) for the non-Bayes-tree predecessor
algorithm this doc builds on.
