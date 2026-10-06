# Sparse Cholesky Factorization

Sparse Cholesky factorization is essentially **Cholesky factorization designed to avoid doing unnecessary work on zeros**.

For a symmetric positive-definite matrix $A$, ordinary Cholesky decomposes:

$$A = LL^\top$$

where $L$ is lower triangular. Column by column, its entries are:

$$L_{jj} = \sqrt{A_{jj} - \sum_{k=1}^{j-1} L_{jk}^2}, \qquad L_{ij} = \frac{1}{L_{jj}}\left(A_{ij} - \sum_{k=1}^{j-1} L_{ik}L_{jk}\right)\ \ (i>j)$$

That sum over earlier columns $k$ is the whole story behind fill-in (Section 3): even when $A_{ij}=0$, $L_{ij}$ can come out nonzero if $i$ and $j$ share an already-eliminated neighbor $k$ with $L_{ik}, L_{jk} \neq 0$.

For example:

```math
A =\begin{bmatrix} 4 & 2 & 0\\
2 & 5 & 3\\
0 & 3 & 6 \end{bmatrix}
```

becomes

```math
L = \begin{bmatrix} 2 & 0 & 0\\
1 & 2 & 0\\
0 & 1.5 & \sqrt{3.75} \end{bmatrix}
```

The key problem is that **large optimization problems often contain mostly zeros**.

---

## 1. Why does sparsity matter?

Consider a SLAM problem with 1,000 robot poses and 5,000 landmarks.

Each measurement connects only a few variables:

- a landmark observation connects one pose and one landmark
- an odometry or loop-closure factor connects two poses

So the graph of which variables share a measurement looks like this (landmark observations only, for clarity):

```text
Pose 1 ─ Landmark 1
      └ Landmark 2

Pose 2 ─ Landmark 2
      └ Landmark 3

Pose 3 ─ Landmark 3
      └ Landmark 4
...
```

Most variables don't directly interact, so the Hessian $H = J^\top \Omega J$ is **sparse**.

Instead of storing something like:

```text
████████████████████
████████████████████
████████████████████
████████████████████
████████████████████
```

we have something more like this (an illustrative sketch; the real pattern depends on the variable order):

```text
██
███
 ███
  ███
   ███
    ██
```

A dense Cholesky algorithm would waste enormous amounts of computation on the zeros. Sparse Cholesky tries to preserve and exploit this structure.

---

## 2. The basic idea

Suppose we're at the linear-solve step of Gauss-Newton or LM:

$$H\Delta x = -b$$

- $H\approx J^\top J$ and $b := J^\top r$, as in [gauss_newton.md](gauss_newton.md).
- Weighted, these become $H = J^\top\Omega J$ and $b = J^\top\Omega r$.
- $\Omega$ is the information matrix from [factor_graph.md §4](factor_graph.md#4-optimization-means-minimizing-all-those-errors), the same quantity as $W_i$ in [nonlinear_least_square.md §12](nonlinear_least_square.md#12-add-measurement-uncertainty).

We need $H$ to be symmetric positive definite, and that isn't automatic. If every factor is relative (odometry, loop closures, reprojections with no prior), some motion of the whole solution leaves the cost unchanged (a **gauge freedom**). $H$ is then only positive *semi*-definite, and Cholesky breaks down on a zero pivot.

- **Pose graphs:** the gauge is a rigid motion of the whole trajectory. Anchoring one pose fixes it ([pose_graph_optimization.md §7](pose_graph_optimization.md#7-the-mathematics-is-actually-quite-intuitive)).
- **Monocular BA:** the gauge also includes uniform rescaling of the scene (7 DoF), which one anchored pose can't pin. The repo's BA scripts fix two cameras instead ([bundle_adjustment.md §14](bundle_adjustment.md#14-evaluating-the-result-gauge-freedom-and-umeyama-alignment)).
- **Either case:** LM's damping $H + \lambda I$ also keeps the system solvable ([levenberg_marquardt.md](levenberg_marquardt.md)).

We then factor $H = LL^\top$, so solving becomes two triangular solves: $Ly=-b$, followed by $L^\top\Delta x=y$.

The important difference is:

> **Sparse Cholesky only stores and computes the entries of $L$ that can be nonzero (the symbolic pattern).**

The catch is fill-in.

---

## 3. The surprising part: Fill-in

Even if $H$ is sparse, $L$ is **not necessarily equally sparse**.

Consider:

```math
H=\begin{bmatrix} * & * & * \\
* & * & 0 \\
* & 0 & * \end{bmatrix}
```

There is no connection between variable 2 and variable 3.

But eliminating variable 1 first - the natural order - creates a new nonzero, $L_{32}\neq0$, so the factor becomes:

```math
L = \begin{bmatrix} * & 0 & 0 \\
* & * & 0 \\
* & [*] & * \end{bmatrix}
```

where $[*]$ marks the entry that was zero in $H$. That newly created nonzero is called **fill-in**. It comes straight from the formula at the top: $`L_{32} = (H_{32} - L_{31}L_{21}) / L_{22}`$, and $H_{32} = 0$ doesn't help when $L_{31}$ and $L_{21}$ are both nonzero.

Eliminate the hub last instead (order 2, 3, 1, so variable 1 becomes the last row and column), and $L$ has no fill at all. §5 turns this into a general rule.

So:

> **Sparse Cholesky = exploit existing sparsity + control newly created fill-in.**

---

## 4. Why does fill-in happen?

The intuitive way to see it is through **variable elimination**. Take this graph:

```text
1 ─ 2 ─ 3
```

Initially 1 is connected to 2, 2 is connected to 3, and 1 is **not** connected to 3.

Now eliminate variable 2. Because 2 connects both 1 and 3, after removing 2 we need to connect its neighbors:

```text
1 ───── 3
```

So elimination creates a new edge $1 \leftrightarrow 3$, which corresponds to a new nonzero in the Cholesky factor. That is **fill-in**.

---

## 5. Ordering becomes extremely important

Take §4's chain `1 ─ 2 ─ 3` again:

- Eliminate **2 first**: we create the edge `1 ─ 3`, so there is fill-in.
- Eliminate **1 first**: only `2 ─ 3` remains, so there is no fill-in.

So the **order in which variables are eliminated** dramatically affects the amount of computation and memory required.

Both examples follow the same rule: eliminating a variable connects all of its remaining neighbors, so eliminate variables with few neighbors first and hubs last. Finding the truly optimal order is NP-hard, so solvers use heuristics built on that rule:

- **AMD** (Approximate Minimum Degree): repeatedly eliminate the variable with the fewest remaining neighbors.
- **COLAMD**: the same idea, computed from the columns of $J$ without forming $H$. It is a common choice for batch solvers.
- **CCOLAMD**: constrained COLAMD, which can force chosen variables (for example the newest poses) to the end. iSAM2 uses it ([isam2_optimization.md §12](isam2_optimization.md#12-variable-ordering-is-also-crucial)).
- **Nested dissection**: split the graph with a small separator, order each half recursively, and put the separator last.

[elimination_tree.md §9](elimination_tree.md#9-why-ordering-matters-so-much) works through the effect of ordering on a SLAM example.

---

## 6. Connection to SLAM

Suppose our state is:

```math
x =\begin{bmatrix} x_1\\
x_2\\
x_3\\
\vdots\\
x_N\\
l_1\\
l_2\\
\vdots \end{bmatrix}
```

with robot poses $x_1, \dots, x_N$ followed by landmarks $l_1, l_2, \dots$. Linearizing the residuals and forming the normal equations gives exactly §2's system $H\Delta x = -b$, with $H = J^\top\Omega J$ and $b = J^\top\Omega r$, which is then factored as $H = LL^\top$.

Because each measurement only involves a small number of variables, $J$ and $H$ are sparse.

The pattern also stays fixed for a fixed graph. Every Gauss-Newton or LM iteration relinearizes at a new estimate, which changes the *values* in $H$, but the same factors connect the same variables, so the *nonzero pattern* doesn't change. Solvers exploit this by splitting the factorization in two: a **symbolic** step (choose the ordering, predict where $L$'s nonzeros go, allocate memory) done once for a fixed graph, and a **numeric** step (compute the values) repeated every iteration. CHOLMOD's `analyze`/`factorize` calls are this split. That's why an expensive ordering heuristic is affordable: its cost is paid once, not per iteration. (Incremental solvers add factors, so their pattern changes and they redo part of this step.)

---

## 7. Factor graph → Hessian → sparse Cholesky

The entire process looks like:

```text
Factor Graph
     │
     │ linearization
     ▼
Jacobian J
     │
     │ Jᵀ Ω J
     ▼
Sparse Hessian H
     │
     │ Sparse Cholesky
     ▼
L Lᵀ
     │
     │ triangular solve
     ▼
Δx
```

This is one of the core computational pipelines behind graph-based SLAM.

---

## 8. Sparse Cholesky vs. dense Cholesky

|                     | Dense Cholesky           | Sparse Cholesky             |
| ------------------- | ------------------------ | --------------------------- |
| Matrix              | Mostly dense             | Mostly zeros                |
| Stores zeros?       | Yes                      | No                          |
| Computation         | Lots of unnecessary work | Exploits sparsity           |
| Memory              | $O(n^2)$                 | Nonzeros of $L$, including fill-in |
| Time                | $O(n^3)$                 | Depends on sparsity/fill-in |
| Large SLAM          | Usually impractical      | Very useful                 |
| Ordering important? | No: a dense $L$ has no zeros left to fill | **Extremely important**     |

---

## 9. Two kinds of sparsity

There are two different kinds of sparsity:

- **Sparse Jacobian:** $J$ is sparse because each measurement depends on only a few variables.
- **Sparse Hessian:** $H=J^\top \Omega J$ is also sparse, but its pattern represents **variable interactions**.

For example:

```text
Measurement 1:
Pose 1 ─ Landmark 1

Measurement 2:
Pose 1 ─ Landmark 2

Measurement 3:
Pose 2 ─ Landmark 2
```

gives this Hessian structure:

```text
       P1 P2 L1 L2

P1     X     X  X
P2        X     X
L1     X     X
L2     X  X     X
```

The rule is exact: $H_{ij}$ is structurally nonzero exactly when variables $i$ and $j$ appear together in at least one factor. So $H$'s graph is the **factor graph** with each factor replaced by edges between all of its variables.

---

## 10. The key intuition

In one sentence:

> **Sparse Cholesky is a way of solving a large linear system exactly, doing arithmetic only on the entries that can be nonzero, while choosing the elimination order to keep fill-in small.**

It is still exact: nothing small is dropped. Dropping small entries on purpose is a different method, *incomplete* Cholesky, used as a preconditioner for iterative solvers.

And in SLAM:

> **The factor graph tells us which variables interact; elimination turns those interactions into a sparse factor $L$.**

This is why **variable ordering, fill-in, elimination trees, and sparse matrix structures** become so important in systems such as iSAM, GTSAM, g2o, and Ceres.

---

## 11. References

1. Liu, J. W. H. (1990). *The Role of Elimination Trees in Sparse Factorization*. SIAM Journal on Matrix Analysis and Applications, 11(1), 134-172. https://doi.org/10.1137/0611010 - the elimination-tree/fill-in relationship behind §3-§5 here, covered in full in [`elimination_tree.md`](elimination_tree.md).
