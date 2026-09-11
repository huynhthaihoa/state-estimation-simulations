# Bayes tree

The name sounds probabilistic, but for SLAM, you can understand it mainly as a **smart tree representation of the factorization of your optimization problem**.

## 1. Start with the factor graph

Suppose your robot has four poses:

```text
x1 ───── x2 ───── x3 ───── x4
   odom12   odom23   odom34
```

And suppose it observes a landmark:

```text
        l1
       /  \
      /    \
     x1    x3
```

The factor graph represents **constraints**:

* `x1 → x2`: odometry
* `x2 → x3`: odometry
* `x3 → x4`: odometry
* `x1 → l1`: observation
* `x3 → l1`: observation

The goal is still:

$$X^* = \arg\min_X \sum_i \|r_i(X)\|^2$$

---

## 2. Linearization gives us a big equation

After Gauss-Newton linearization, we get something like:

$$H\Delta x=-g$$

or equivalently:

$$A\Delta x=b$$

Now we need to solve this large sparse system.

This is where **Sparse Cholesky Factorization** becomes important — see [`pose_graph_optimization.md` §5](pose_graph_optimization.md#5-solving-the-linear-system-gauss-newton-step) if you want the recap: $H = LL^T$, solved via two triangular substitutions instead of a full inversion, exploiting the fact that $H$ is mostly zero.

We want to factorize the system efficiently.

---

## 3. Elimination is the key idea

Imagine we have:

```text
x1 ─── x2 ─── x3
```

Suppose we decide to eliminate `x1`.

Originally:

```text
x1 ─── x2
```

After eliminating `x1`, we no longer need `x1` in the remaining problem.

But its information has to go somewhere.

It gets **summarized into a new constraint involving x2**.

Conceptually:

```text
Before:

x1 ─── x2 ─── x3


Eliminate x1:

        x2 ─── x3
         ↑
    summarized
    information
    from x1
```

This is the fundamental idea behind elimination.

(Here `x1` is eliminated as part of building a solve order - it's still implicitly part of the problem and gets re-eliminated on the next update. [marginalization.md](marginalization.md) reuses this exact step for a different purpose: permanently discarding an old state to bound a sliding-window estimator's size.)

---

## 4. Why elimination creates a tree

Let's use a slightly bigger example:

```text
x1 ─── x2 ─── x3 ─── x4
```

Suppose we eliminate from left to right:

```text
x1
 ↓
x2
 ↓
x3
 ↓
x4
```

The resulting dependency structure can be represented as:

```text
x1
 |
x2
 |
x3
 |
x4
```

This structure tells us:

> **Which variables depend on which other variables after elimination.**

That is essentially what the **Bayes tree** captures.

---

## 5. But there's an important difference from an ordinary tree

A Bayes tree is not simply:

> "My robot trajectory looks like this."

It is a tree representing the **conditional dependencies produced by eliminating variables from the factor graph**.

That's why you shouldn't think:

```text
Bayes tree = SLAM trajectory
```

Instead think:

```text
Factor graph
     ↓
elimination
     ↓
Bayes tree
```

The factor graph describes the **problem**.

The Bayes tree describes an efficient **solution structure**.

---

## 6. The probabilistic interpretation

The name "Bayes tree" comes from probability.

Suppose we have variables:

$$x_1,x_2,x_3$$

Their joint probability can be decomposed as:

$$P(x_1,x_2,x_3) = P(x_1|x_2,x_3)P(x_2|x_3)P(x_3)$$

The tree represents these conditional relationships.

For SLAM, however, you don't need to become a probability expert to understand iSAM2.

A useful mental translation is:

> **Bayes tree = conditional dependency tree created by variable elimination.**

---

## 7. Here's where it becomes powerful for [iSAM2](isam2_optimization.md)

Suppose we have:

```text
x1 ─ x2 ─ x3 ─ x4 ─ x5
```

The robot gets a new measurement involving `x5`.

iSAM2 doesn't want to rebuild everything.

Because it has the Bayes tree, it can ask:

> "Which part of my factorization is affected by this new factor?"

For example:

```text
       x1
       |
       x2
       |
       x3
       |
       x4
       |
       x5  ← new information
```

It can update the relevant part.

---

## 8. Even more interesting: loop closure

Suppose we have:

```text
x1 ─ x2 ─ x3 ─ x4 ─ x5
│                    │
└──── loop closure ──┘
```

The new loop-closure factor connects `x1` and `x5`.

This can change the solution for many variables.

The Bayes tree lets iSAM2 identify the affected portion of the tree.

Conceptually:

```text
       x1
       │
       x2    ← affected
       │
       x3    ← affected
       │
       x4    ← affected
       │
       x5    ← affected
```

The affected section is removed/re-eliminated and then reinserted into the Bayes tree.

That's much smarter than rebuilding the entire factorization blindly.

---

## 9. Bayes tree + sparse Cholesky

This connects directly to [`pose_graph_optimization.md`'s sparse Cholesky factorization](pose_graph_optimization.md#5-solving-the-linear-system-gauss-newton-step) and to iSAM2's use of it in [`isam2_optimization.md` §7](isam2_optimization.md#7-bayes-tree--the-most-important-intuition).

You can think of the pipeline as:

```text
Factor Graph
     │
     ▼
Linearization
     │
     ▼
Sparse Linear System
     │
     ▼
Variable Elimination
     │
     ▼
Sparse Factorization
     │
     ▼
Bayes Tree
```

Then when a new measurement arrives:

```text
New Factor
    │
    ▼
Which variables are affected?
    │
    ▼
Find affected Bayes-tree region
    │
    ▼
Remove/relinearize/reorder
    │
    ▼
Re-eliminate
    │
    ▼
Updated Bayes Tree
```

This is the core mechanism behind **[iSAM2](isam2_optimization.md)'s incremental efficiency** - the older [iSAM](isam_optimization.md) algorithm gets its own incremental updates from QR row insertion alone, without a Bayes tree at all.

---

## 10. A very intuitive analogy

Imagine you have a company hierarchy:

```text
CEO
 │
Manager A
 │
Team Leader
 │
Engineer
```

Each level summarizes information from the level below.

If one engineer changes something, you don't need to reorganize the entire company.

You update:

```text
Engineer
   ↓
Team Leader
   ↓
Manager
```

while leaving unrelated branches alone.

The Bayes tree gives iSAM2 a similar ability to **localize the computational consequences of new information**.

---

## 11. Why not just use the factor graph directly?

You might ask:

> "If the factor graph already tells us the relationships, why do we need a Bayes tree?"

Because the factor graph is good at representing **constraints**, but not necessarily the most convenient structure for **incrementally solving the linearized system**.

Think:

```text
Factor graph
    =
"What measurements constrain what?"

Bayes tree
    =
"How can I efficiently solve these constraints?"
```

That's a very useful distinction.

---

## 12. Bayes tree vs factor graph

| Factor Graph                        | Bayes Tree                                       |
| ----------------------------------- | ------------------------------------------------ |
| Represents measurements/constraints | Represents elimination/conditional structure     |
| Bipartite graph                     | Tree structure                                   |
| Variables + factors                 | Conditional relationships                        |
| Describes the SLAM problem          | Describes its factorized solution                |
| Natural representation of SLAM      | Efficient representation for incremental solving |

---

## 13. One more important concept: cliques

A Bayes tree doesn't necessarily contain one variable per node.

Its nodes are actually **cliques**.

For example:

```text
        {x1}
          |
       {x2,x3}
          |
        {x4}
```

A clique contains variables that are conditionally related after elimination.

This is important because when iSAM2 updates the graph, it often works with **cliques/subtrees**, rather than individual variables.

You don't need to master cliques yet, but keep the word in mind.

---

## 14. The connection to [iSAM2](isam2_optimization.md) becomes very clean

Now you can understand iSAM2 as:

> **Maintain a factor graph + maintain its Bayes-tree factorization + incrementally modify the affected part when new information arrives.**

So:

```text
             iSAM2
               │
       ┌───────┴────────┐
       │                │
 Factor Graph       Bayes Tree
       │                │
 measurements      efficient
 constraints       factorization
       │                │
       └───────┬────────┘
               ↓
       incremental SLAM
```

---

## 15. Where this is implemented in this repo

Partially. [`bayes_tree_construction.py`](../../use_numpy/bayes_tree_construction.py) builds an actual Bayes tree from this repo's pose-graph topology - `symbolic_eliminate`/`bayes_tree_affected_path` in `utils.py` run the elimination-game construction from §3-§4 and the root-ward affected-path query from §7-§9 - and plots the "small vs. large affected region" contrast this doc argues for in prose (§7 vs. §8) using a real, computed example instead of an ASCII sketch. It fixes the elimination order (oldest node first) rather than choosing one dynamically, so **COLAMD-style variable reordering is still not implemented** - worth noting since, under that fixed order, this repo's loop-closure edge happens to produce the worst possible case (it invalidates the *entire* tree, not just a subtree), which is exactly the scenario dynamic reordering exists to avoid. There is also still no numeric fluid-relinearization solve integrated with this tree - [`pose_graph_incremental.py`](../../use_numpy/pose_graph_incremental.py) (both `use_numpy/` and `use_manif/`) separately implements the older [iSAM v1](isam_optimization.md) algorithm (incremental QR row insertion, no Bayes tree at all); see [`isam_optimization.md` §14](isam_optimization.md#14-where-this-is-implemented-in-this-repo) and [`isam2_optimization.md` §19](isam2_optimization.md#19-where-this-is-implemented-in-this-repo) for exactly where that numeric-solve line is drawn.

So this page's data structure now has a real, runnable counterpart - but the rest of [iSAM2](isam2_optimization.md) (dynamic reordering, fluid relinearization integrated with an actual solve) still exists only conceptually here.

---

## 16. References

1. Kaess, M., Johannsson, H., Roberts, R., Ila, V., Leonard, J. J., & Dellaert, F. (2012). *iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree*. International Journal of Robotics Research, 31(2), 216–235. https://doi.org/10.1177/0278364911430419 - the Bayes tree itself: its construction via variable elimination (§3-§4), the clique structure (§13), and how it localizes incremental updates (§7-§9).
2. Kaess, M., Ranganathan, A., & Dellaert, F. (2008). *iSAM: Incremental Smoothing and Mapping*. IEEE Transactions on Robotics, 24(6), 1365–1378. https://doi.org/10.1109/TRO.2008.2006706 - the predecessor algorithm that reaches incremental updates without a Bayes tree, contrasted in §7 and implemented by `pose_graph_incremental.py`.

See also [`isam2_optimization.md`](isam2_optimization.md) for how the Bayes tree fits into the full iSAM2 algorithm, and [`isam_optimization.md`](isam_optimization.md) for the non-Bayes-tree predecessor this repo actually implements.