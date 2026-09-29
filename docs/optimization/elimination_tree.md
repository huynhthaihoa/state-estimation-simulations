# Elimination trees

> An elimination tree tells us **which variables depend on which other variables during sparse Cholesky elimination**.

---

## 1. Start with the simplest example

Suppose our sparse matrix has this structure:

```text
1 ─ 2 ─ 3 ─ 4
```

Think of each number as a variable.

We eliminate variables in the order:

$$1 \rightarrow 2 \rightarrow 3 \rightarrow 4$$

When we eliminate variable 1, it interacts with 2.

So we get:

```text
1 → 2
```

Then eliminate 2:

```text
2 → 3
```

Then:

```text
3 → 4
```

This produces the **elimination tree**:

```text
4
▲
│
3
▲
│
2
▲
│
1
```

Here, 2 is the **parent** of 1, 3 is the parent of 2, etc.

---

## 2. Why call it a tree?

Because every variable, except the root, has a parent.

For example:

```text
       5
      / \
     3   4
    / \
   1   2
```

might represent dependencies created during elimination.

The important point is that the tree is **not necessarily the original graph**.

The original graph describes:

> "Who directly interacts with whom?"

The elimination tree describes:

> "Who depends on whom during factorization?"

That's a very important distinction.

---

## 3. Let's connect it to sparse Cholesky

We want:

$$H=LL^\top$$

During factorization, we eliminate variables one by one.

Suppose:

```text
Original graph:

1 ─ 2 ─ 3
    │
    4
```

If we eliminate 1 first:

```text
1 removed
```

then 2 becomes responsible for the information from 1. Since 1 only had one neighbor, this creates no fill-in.

Conceptually:

```text
2
▲
│
1
```

So:

$$\text{parent}(1)=2$$

Now eliminate 2. It still has **two** neighbors left, 3 and 4 - and they were never directly connected. Removing 2 forces them to pick up each other's dependency, so a new edge appears:

$$3 \leftrightarrow 4 \quad \text{(fill-in)}$$

$$\text{parent}(2)=3$$

Eliminating 3 next, its only remaining neighbor is 4 (via that new fill-in edge), so:

$$\text{parent}(3)=4$$

giving the full tree:

```text
4
▲
│
3
▲
│
2
▲
│
1
```

Notice this ends up the same *shape* as Section 1's tree, but for a different reason: Section 1's chain came from a graph that was already a chain, while here the 3–4 link only exists because eliminating 2 created it. That's exactly Section 2's point - the tree is not the original graph.

The tree tells the factorization algorithm where information flows.

---

## 4. The really useful intuition: information flows upward

Imagine each variable carries some information.

During elimination:

```text
parent
  ▲
  │
child
```

The child is eliminated, and its information is passed to its parent.

For example:

```text
 Pose 3
  ▲
  │
 Pose 2
  ▲
  │
 Pose 1
  ▲
  │
Landmark 1
```

You can think:

> "Landmark 1 has been eliminated, but its information still affects Pose 1."

Then Pose 1 is eliminated, and its resulting information affects Pose 2.

So information propagates **up the tree**.

---

## 5. Elimination tree vs. factor graph

This distinction is extremely useful for SLAM.

### Factor graph

Shows measurement relationships:

```text
x1 ─ factor ─ l1
x1 ─ factor ─ l2
x2 ─ factor ─ l2
x2 ─ factor ─ l3
```

It answers:

> **Which variables are connected by measurements?**

---

### Elimination tree

Shows computational dependency, here for the elimination order l1, l2, l3, x1, x2:

```text
        x2
       /  \
      x1   l3
     /  \
   l1    l2
```

($l_2$ touches both $x_1$ and $x_2$, so eliminating it creates a new fill-in link between them - which is why $x_1$ ends up as $x_2$'s child.)

It answers:

> **What must be computed before what?**

So:

```text
Factor graph
     ↓
Variable interactions
     ↓
Choose elimination ordering
     ↓
Elimination tree
     ↓
Sparse Cholesky computation
```

---

## 6. Why is this useful?

There are three major reasons.

### A. Understand fill-in

Recall the fill-in example from [`sparse_cholesky_factorization.md`](sparse_cholesky_factorization.md):

```text
1 ─ 2 ─ 3
```

Eliminate 2:

```text
1 ─── 3
```

We created fill-in.

The elimination tree helps represent the resulting dependency structure.

So the tree is closely related to how the sparse factor $L$ is structured.

---

### B. Efficient computation

Suppose the tree looks like:

```text
        8
       /  \
      6    7
     / \    \
    4   5    1
   / \
  2   3
```

Variables 2 and 3 can be processed before 4.

Similarly, 4 and 5 can potentially be processed independently before 6.

You can see computational dependencies immediately:

```text
2 ─┐
   ├──> 4 ─┐
3 ─┘       │
           ├──> 6 ─┐
5 ─────────┘       │
                   ├──> 8
1 ──> 7 ───────────┘
```

The two subtrees under 8, {6, 4, 5, 2, 3} and {7, 1}, share nothing until 8, so they can be processed fully in parallel. This exposes **parallelism**.

---

## 7. This becomes very important for iSAM2

iSAM2 uses a [**Bayes tree**](bayes_tree.md), which is built directly from the elimination tree.

Very roughly:

```text
Factor Graph
     │
     ▼
Variable elimination
     │
     ▼
Bayes Tree
```

Concretely, a Bayes tree is the elimination tree with some of its nodes merged into **cliques**, where each clique stores the conditional density produced when its variables were eliminated ([bayes_tree.md §13](bayes_tree.md#13-one-more-important-concept-cliques)).

You can think of:

### Elimination tree

```text
"What depends on what?"
```

### Bayes tree

```text
"What probabilistic information is summarized by each clique?"
```

This is why the Bayes tree is so useful for **incremental SLAM**.

---

## 8. A SLAM example

Imagine a robot trajectory:

```text
x1 ─ x2 ─ x3 ─ x4 ─ x5
```

and landmarks:

```text
     l1
     │
x1 ─ x2 ─ x3 ─ x4 ─ x5
     │         │
     l2        l3
```

Suppose we eliminate `x1` and the landmarks first, then the remaining poses oldest first: x1, l1, l2, l3, x2, x3, x4, x5.

The elimination tree is:

```text
              x5
              │
              x4
             /  \
           x3    l3
           │
           x2
          / | \
        x1  l1  l2
```

The exact tree depends heavily on the elimination ordering.

That's important:

> **There isn't one universally fixed elimination tree for a problem. Change the ordering, and you can change the tree.**

---

## 9. Why ordering matters so much

Consider two possible strategies.

### Bad ordering

```text
Eliminate:
Pose 1
Pose 2
Pose 3
...
Landmarks
```

In bundle adjustment, where each pose observes many landmarks, this creates lots of fill-in. Eliminating a pose connects every landmark it sees to every other one (Section 3's fill-in rule), so the landmark block fills in densely.

You might end up with:

```text
████████████████
 ███████████████
  ██████████████
   █████████████
```

Lots of computation.

---

### Better ordering

Eliminate the landmarks first:

```text
Landmarks
   ↓
Poses
```

Landmarks never touch each other directly, so eliminating one only connects the poses that observe it. The structure stays much sparser. This is exactly bundle adjustment's Schur-complement trick ([bundle_adjustment.md §12](bundle_adjustment.md#12-block-sparsity-and-the-schur-complement)): eliminate every landmark first, then solve the much smaller camera-only system.

The resulting $L$ has far fewer nonzeros (it's always $n \times n$), and the elimination tree is more manageable.

---

## 10. One subtle point

Don't think of the elimination tree as a picture of the original matrix's graph. It isn't one: Section 3's tree contains a 3–4 link that the original graph doesn't have.

> **The elimination tree represents the dependency structure of the Cholesky factorization.**

Precisely (Liu 1990), it's read off the sparsity of the **factor** $L$, with the variables numbered in elimination order:

```math
\text{parent}(j) = \min\{\, i > j : L_{ij} \neq 0 \,\}
```

That is, the parent of $j$ is the row of the first nonzero below the diagonal in column $j$ of $L$. $L$'s pattern depends on both the matrix's pattern and the elimination order, so the same matrix under two different orderings gives two different trees - Section 9's point.

---

## 11. The big picture

The concepts covered above connect into a bigger picture:

```text
                SLAM
                  │
                  ▼
             Factor Graph
                  │
                  ▼
           Linearization
                  │
                  ▼
              Jacobian J
                  │
                  ▼
             H = Jᵀ W J
                  │
                  ▼
          Sparse Hessian
                  │
          choose ordering
                  │
                  ▼
          Variable Elimination
                  │
             ┌────┴────┐
             ▼         ▼
         Fill-in   Elimination
                      Tree
             │         │
             └────┬────┘
                  ▼
         Sparse Cholesky
             H = LLᵀ
                  │
                  ▼
              Solve Δx
```

And then:

```text
Elimination structure
        ↓
    Bayes Tree
        ↓
      iSAM2
```

### The simplest mental model

If you remember only three things:

**Factor graph:**

> "Who talks to whom?"

**Sparse Cholesky:**

> "How can I solve the resulting system without wasting work on zeros?"

**Elimination tree:**

> "What computations depend on what other computations?"

That mental model will make **Bayes trees and iSAM2** much easier to understand.

---

## 12. References

1. Liu, J. W. H. (1990). *The Role of Elimination Trees in Sparse Factorization*. SIAM Journal on Matrix Analysis and Applications, 11(1), 134-172. https://doi.org/10.1137/0611010 - the original definition and analysis of the elimination tree this doc builds intuition for, including the parent/child dependency-structure framing used throughout.
