# Sparse Cholesky Factorization — intuitive explanation

Sparse Cholesky factorization is essentially **Cholesky factorization designed to avoid doing unnecessary work on zeros**.

For a symmetric positive-definite matrix $A$, ordinary Cholesky decomposes:

$$A = LL^T$$

where $L$ is lower triangular.

For example:

$$A =\begin{bmatrix}4 & 2 & 0\\
2 & 5 & 3\\
0 & 3 & 6\end{bmatrix}$$

becomes

$$L = \begin{bmatrix}2 & 0 & 0\\
1 & 2 & 0\\
0 & 1.5 & \sqrt{3.75}\end{bmatrix}$$

The key problem is that **large optimization problems often contain mostly zeros**.

---

## 1. Why does sparsity matter?

Consider a SLAM problem.

Suppose we have 1,000 robot poses and 5,000 landmarks.

Each measurement usually connects only:

* one pose
* one landmark

So the Hessian/information matrix might look conceptually like:

```text
Pose 1 ─ Landmark 1
      └ Landmark 2

Pose 2 ─ Landmark 2
      └ Landmark 3

Pose 3 ─ Landmark 3
      └ Landmark 4
...
```

Most variables don't directly interact.

Therefore the Hessian

$$H = J^T WJ$$

is **sparse**.

Instead of storing something like:

```text
████████████████████
████████████████████
████████████████████
████████████████████
████████████████████
```

we have something more like:

```text
██
███
 ███
  ██
   ███
    ██
```

A dense Cholesky algorithm would waste enormous amounts of computation on the zeros.

Sparse Cholesky tries to preserve and exploit this structure.

---

## 2. The basic idea

Suppose:

$$H\Delta x = -b$$

and $H$ is symmetric positive definite.

We factor:

$$H = LL^T$$

Then solving becomes two triangular solves:

$$Ly=-b$$

followed by

$$L^T\Delta x=y$$

The important difference is:

> **Sparse Cholesky only stores and computes the nonzero entries of $L$.**

However, there is an important complication.

---

## 3. The surprising part: Fill-in

This is probably the most important concept to understand.

Even if $H$ is sparse, $L$ is **not necessarily equally sparse**.

Consider:

$$H = \begin{bmatrix} \boxed{*} & * & * \\ * & \boxed{*} & 0 \\ * & 0 & \boxed{*} \end{bmatrix}$$

There is no connection between variable 2 and variable 3.

But eliminating variable 1 first — the standard order — creates a new nonzero:

$$L_{32}\neq0$$

so the factor becomes:

$$L = \begin{bmatrix} * & 0 & 0 \\ * & * & 0 \\ * & \boxed{*} & * \end{bmatrix}$$

That newly created nonzero is called **fill-in**.

So:

> **Sparse Cholesky = exploit existing sparsity + control newly created fill-in.**

---

## 4. Why does fill-in happen?

A very intuitive way to see it is through **variable elimination**.

Suppose we have this graph:

```text
1 ─ 2 ─ 3
```

Initially:

```text
1 connected to 2
2 connected to 3
1 NOT connected to 3
```

Now eliminate variable 2.

Because 2 connects both 1 and 3, after removing 2 we need to connect its neighbors:

```text
1 ───── 3
```

So elimination creates a new edge:

$$1 \leftrightarrow 3$$

That edge corresponds to a new nonzero in the Cholesky factor.

That's **fill-in**.

---

## 5. Ordering becomes extremely important

Here's where sparse Cholesky gets interesting.

Suppose we have:

```text
1 ─ 2 ─ 3
```

If we eliminate:

```text
2 first
```

we create:

```text
1 ─ 3
```

and get fill-in.

But if we eliminate:

```text
1 first
```

there is no fill-in:

```text
2 ─ 3
```

So the **order in which variables are eliminated** dramatically affects the amount of computation and memory required.

This is why SLAM systems use algorithms such as:

* AMD — Approximate Minimum Degree
* COLAMD
* nested dissection
* specialized SLAM variable orderings

---

## 6. Connection to SLAM

This is especially important for your SLAM/optimization work.

Suppose your state is:

$$x =\begin{bmatrix}x_1\\
x_2\\
x_3\\
\vdots\\
x_N\\
l_1\\
l_2\\
\vdots\end{bmatrix}$$

and your nonlinear least-squares problem is:

$$\min_x \sum_i \|e_i(x)\|^2.$$

After linearization:

$$J\Delta x \approx -e$$

and solving the normal equations gives:

$$J^TWJ\Delta x=-J^TWe$$

Define:

$$H=J^TWJ$$

so:

$$H\Delta x=-b$$

Now factor:

$$H=LL^T$$

This is where sparse Cholesky can be used.

Because each measurement only involves a small number of variables, $J$ and $H$ are sparse.

---

## 7. Factor graph → Hessian → sparse Cholesky

You can think about the entire process as:

```text
Factor Graph
     │
     │ linearization
     ▼
Jacobian J
     │
     │ Jᵀ W J
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
| Memory              | $O(n^2)$               | Depends on sparsity/fill-in |
| Large SLAM          | Usually impractical      | Very useful                 |
| Ordering important? | Less important           | **Extremely important**     |

---

## 9. One subtle but important distinction

There are actually two different kinds of sparsity you may encounter:

### Sparse Jacobian

$$J$$

is sparse because each measurement depends on only a few variables.

### Sparse Hessian

$$H=J^TWJ$$

is also sparse, but its sparsity pattern represents **variable interactions**.

For example:

```text
Measurement 1:
Pose 1 ─ Landmark 1

Measurement 2:
Pose 1 ─ Landmark 2

Measurement 3:
Pose 2 ─ Landmark 2
```

gives a Hessian structure roughly like:

```text
       P1 P2 L1 L2

P1     X     X  X
P2        X     X
L1     X     X
L2     X  X     X
```

The graph structure of $H$ is therefore closely related to the **factor graph**.

---

## 10. The key intuition

If you remember only one thing:

> **Sparse Cholesky is a way of solving a large linear system by factoring only the important nonzero interactions, while carefully choosing the elimination order to minimize new interactions (fill-in).**

And in SLAM:

> **The factor graph tells you which variables interact; elimination turns those interactions into a sparse factor $L$.**

This is why **variable ordering, fill-in, elimination trees, and sparse matrix structures** become so important in systems such as iSAM, GTSAM, g2o, and Ceres.