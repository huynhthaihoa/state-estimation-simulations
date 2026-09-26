# iSAM / incremental optimization

Instead of solving the entire factor graph from scratch every time a new measurement arrives, update the existing solution intelligently.

---

## 1. The problem with ordinary batch optimization

Imagine a robot moving:

```text
t0       t1       t2       t3
X0 ───── X1 ───── X2 ───── X3
```

At every timestep, you receive new measurements.

A traditional **batch** approach might do:

```text
Measurement at t0
       ↓
Optimize X0

Measurement at t1
       ↓
Optimize X0, X1

Measurement at t2
       ↓
Optimize X0, X1, X2

Measurement at t3
       ↓
Optimize X0, X1, X2, X3
```

So every time a new measurement arrives, you potentially solve the **whole problem again**.

For a large SLAM system:

```text
X0 X1 X2 ... X1000
```

re-solving everything repeatedly becomes expensive.

---

## 2. Incremental optimization says:

Instead:

```text
Old solution
     +
New measurement
     ↓
Update only what needs updating
```

Conceptually:

```text
Before:

X0 ─── X1 ─── X2 ─── X3
              ↑
        already solved


New measurement:

X3 ─── X4
       ↑
      new


Don't throw everything away!

Just update the existing solution.
```

This is the fundamental idea behind **iSAM (incremental Smoothing and Mapping)**.

---

## 3. Why is this possible?

This is where things get interesting.

Suppose we have a [factor graph](factor_graph.md):

```text
X0 ─── X1 ─── X2 ─── X3
              │
              L1
```

After linearization, the nonlinear optimization becomes approximately:

$$A\Delta x=b$$

Instead of directly solving this system every time, we can factorize it:

$$A \approx QR$$

or equivalently work with a related factorization of the information/Hessian system.

> **Note**: QR decomposition factors a matrix as $A=QR$, with $Q$ orthogonal and $R$ upper-triangular. Because $Q$ is orthogonal it doesn't change the least-squares solution, so solving $A\Delta x=b$ reduces to the cheap triangular solve $R\Delta x = Q^\top b$ - and, critically for iSAM, $R$ can be updated incrementally via Givens rotations when a new row (factor) arrives, instead of refactorizing $A$ from scratch. That incremental-update property is exactly what the next paragraph relies on.

The important point is:

> **When a new factor is added, much of the previous factorization is still useful.**

We don't necessarily need to throw away the previous computation.

---

## 4. An analogy: editing a spreadsheet

Imagine you have a huge spreadsheet:

```text
100,000 calculations
```

You change one cell.

A stupid approach would be:

> "Let's recalculate absolutely everything from scratch."

An incremental approach says:

> "Which calculations actually depend on this cell?"

Then update only those.

iSAM applies a similar philosophy to the optimization problem.

---

## 5. The SLAM example

Suppose your robot has estimated:

```text
X0 ── X1 ── X2 ── X3 ── X4
```

and each edge represents odometry.

Then you get a new camera observation at X4.

Maybe it observes landmark L0:

```text
                 L0
                 ●
                /
               /
X0 ●──X1──X2──X3──X4
                  ↑
             new camera
```

A batch optimizer says:

> Rebuild and solve the entire nonlinear problem.

An incremental optimizer says roughly:

> "The new factor primarily affects X4 and the variables connected to it. Update the existing solution accordingly."

---

## 6. But what about loop closure?

This is where incremental SLAM becomes especially interesting.

Suppose we have:

```text
X0 ── X1 ── X2 ── X3 ── X4
│                        │
└────────────────────────┘
           loop closure
```

The loop closure can affect **many previous poses**.

So incremental optimization cannot simply update X4.

It needs to determine:

> **Which variables are affected by this new constraint?**

Then update the relevant part of the solution.

This is why iSAM maintains and updates a structured factorization of the problem.

---

## 7. The key concept: Bayes tree

Modern incremental smoothing approaches, particularly **iSAM2**, use a structure called a **Bayes tree**.

You can think of it as a clever organization of the factor graph that tells the optimizer:

> **"If this new measurement arrives, these are the parts of the solution that need to be reconsidered."**

Very roughly:

```text
Factor graph
     ↓
Eliminate variables
     ↓
Bayes tree

        X0
        │
        X1
        │
        X2
        │
        X3
        │
        X4
```

(A straight chain here, not a branch - the underlying factor graph is itself a chain $X_0-X_1-X_2-X_3-X_4$ with no side edges, so eliminating it in order produces a chain elimination/Bayes tree too. Two variables joined by a real edge in the factor graph - like $X_2$ and $X_3$ here - can never end up as siblings with no ancestor/descendant relationship in a correctly-built Bayes tree.)

When a new factor connects X4 to X0:

```text
X0 ───────────── X4
```

the algorithm identifies the affected portion of the Bayes tree and **relinearizes/recalculates that portion**, rather than rebuilding everything.

That's the really powerful idea behind iSAM2.

---

## 8. Batch vs incremental

Here's the simplest comparison:

| Batch optimization              | Incremental optimization        |
| ------------------------------- | ------------------------------- |
| Add measurements                | Add measurements                |
| Rebuild/solve the whole problem | Reuse previous computation      |
| Potentially expensive           | Much more efficient             |
| Good for offline SLAM           | Good for online SLAM            |
| Simple conceptual model         | More complicated implementation |
| Example: standard Gauss-Newton  | Example: iSAM/iSAM2             |

Think:

```text
BATCH

new measurement
      ↓
┌───────────────────┐
│ Optimize ALL      │
│ variables again   │
└───────────────────┘
```

versus:

```text
INCREMENTAL

new measurement
      ↓
┌───────────────────┐
│ Find affected     │
│ variables         │
└─────────┬─────────┘
          ↓
   Update those parts
```

---

## 9. iSAM doesn't mean "never touch old variables"

This is an important misconception.

You might think:

> "Incremental means only optimize the newest pose."

**No.**

Old poses can absolutely be updated.

For example:

```text
Before loop closure:

X0 ── X1 ── X2 ── X3 ── X4
↑
slightly wrong


Loop closure arrives:

X4 ───────── X0
```

Now the loop closure tells us the trajectory is inconsistent.

The optimizer may change:

```text
X0
X1
X2
X3
X4
```

But it does so **selectively and efficiently**, using the existing structure.

That's why the term **smoothing** is important.

The system is not merely estimating:

$$X_t$$

It is continually refining:

$$X_0,\ldots,X_t$$

using all available information.

---

## 10. Filtering vs iSAM

This connects directly to **[filtering vs optimization/smoothing](../filtering_smoothing.md)**.

### EKF-style filtering

Conceptually:

```text
X0 → X1 → X2 → X3 → X4
                 ↑
            current state
```

Once you've processed X0, you largely summarize its information and move forward.

You primarily care about:

$$P(X_t|Z_{1:t})$$

---

### iSAM / smoothing

Instead:

```text
X0 ── X1 ── X2 ── X3 ── X4
│     │     │     │     │
└─────┴─────┴─────┴─────┘
       all history
```

You maintain a representation of the entire trajectory:

$$X_{0:t}$$

and continuously refine it.

So:

> **Filtering:** "What is my best estimate of the robot NOW?"

> **Smoothing:** "Given everything I've seen, what were the best estimates of the robot's states throughout the entire trajectory?"

---

## 11. Where iSAM fits into the SLAM mental map

You can now connect your previous topics like this:

```text
                    SLAM
                     │
             State estimation
                     │
          ┌──────────┴──────────┐
          │                     │
      Filtering             Smoothing
          │                     │
       EKF / KF           Factor graph
                                │
                       Nonlinear optimization
                                │
                    ┌───────────┴──────────┐
                    │                      │
                  Batch              Incremental
                    │                      │
             Gauss-Newton            iSAM / iSAM2
             Levenberg-Marquardt
```

And underneath the pose optimization:

```text
SE(3)
 ↓
Lie algebra
 ↓
Perturbation
 ↓
Jacobian
 ↓
Linearization
 ↓
Gauss-Newton
 ↓
Factorization
 ↓
iSAM / iSAM2
```

---

## 12. The most intuitive way to remember iSAM

Imagine you're drawing a map while walking.

### Batch approach

Every time you take one more step:

> "Let me redraw the entire map from scratch."

### Incremental approach

Every time you take one more step:

> "I already have a pretty good map. I'll incorporate this new information into it."

And when you recognize a place you've visited before:

> "Oh! This new observation conflicts with my old map. I need to adjust the affected parts."

That's **iSAM**.

## 13. One-sentence summary

> **iSAM is an incremental factor-graph smoothing algorithm that continuously updates the SLAM solution as new measurements arrive, reusing previous computations instead of repeatedly solving the entire problem from scratch.**

And **iSAM2** takes this further by using a **Bayes tree** to efficiently identify and update the parts of the solution affected by new measurements.

---

## 14. Where this is implemented in this repo

[`pose_graph_incremental.py`](../../use_numpy/pose_graph_incremental.py) (both `use_numpy/` and `use_manif/`) implements exactly §3's mechanism: new odometry edges are absorbed into a running square-root-information matrix via Givens-rotation row insertion (`qr_insert_row` in `utils.py`) instead of rebuilding the linear system from scratch, contrasted directly against a batch baseline that re-solves everything at every new node - reproducing §1/§8's batch-vs-incremental comparison and §6/§9's "loop closure needs a wide update" point empirically (the script always triggers a full relinearization on the loop-closure edge, plus periodically otherwise). **It does not implement §7's Bayes tree, nor any variable reordering at all** - but only the Bayes tree is genuinely iSAM2-specific. Variable reordering (via COLAMD) to bound fill-in is already part of *original* iSAM (Kaess et al. 2008, periodic batch reordering during full relinearization) - what iSAM2 actually adds on top is making that reordering *incremental/fluid* (reordering only as needed, tied to the Bayes tree) instead of a periodic full pass. This script skips variable reordering of either kind - a real simplification relative to even the 2008 original, not just relative to iSAM2. For the Bayes tree itself, see [`bayes_tree.md`](bayes_tree.md); for the full iSAM2 algorithm this repo doesn't implement, see [`isam2_optimization.md`](isam2_optimization.md).

### 14.1 The incremental math, concretely

The script solves the same pose graph as `pose_graph.py`. `linearize_edge` returns that script's residual ${e_{ij} = \mathrm{Log}(Z_{ij}^{-1} X_i^{-1} X_j)}$ and Jacobians $J_i$, $J_j$ (see [pose_graph_optimization.md §15.7](pose_graph_optimization.md#157-the-solver-concretely)). The difference is how the linear system is stored and updated. Instead of $H$ and $g$, the script keeps an upper-triangular square-root-information matrix $R$ and a right-hand side $d$. The step is always ${\boldsymbol{\delta} = R^{-1} d}$, computed by back substitution (`solve_triangular`).

**Whitened rows** (`edge_whitened_block`): each edge becomes 6 rows of a least-squares system ${A \boldsymbol{\delta} \approx b}$. With $S$ the upper-triangular square root of the edge information matrix $\Omega$ (`sqrt_info`):

$$S = \mathrm{chol}(\Omega)^\top, \quad S^\top S = \Omega, \qquad A_{ij} = S \begin{bmatrix} \cdots & J_i & \cdots & J_j & \cdots \end{bmatrix}, \qquad b_{ij} = -S \, e_{ij}$$

Then $`\lVert A_{ij}\boldsymbol{\delta} - b_{ij} \rVert^2 = (e_{ij} + J\boldsymbol{\delta})^\top \Omega \, (e_{ij} + J\boldsymbol{\delta})`$, the linearized edge cost. `main` sets $\Omega = I_6$, so $S = I_6$ in the default run.

**Anchor rows** (`full_relinearize`): node 0 gets 6 extra rows $`\sqrt{w} \, I_6`$ with right-hand side 0, where $w$ = `--anchor-weight` (default $10^6$). These add ${w I_6}$ to node 0's block of ${A^\top A}$, which is the same gauge fix as `pose_graph.py`'s ${H_{00} \mathrel{+}= 10^6 I_6}$. Like that one, it pulls node 0's step toward zero around its current estimate.

**Full rebuild** (`full_relinearize`): stack the anchor rows and every edge seen so far, all linearized at the current linearization point ${\bar{X}}$ (`x_lin`), then factorize once:

$$A = QR \quad (\texttt{np.linalg.qr}), \qquad d = Q^\top b$$

This gives ${R^\top R = A^\top A = H}$ and ${R^\top d = A^\top b = g}$, the same normal equations as `pose_graph.py` without damping.

**Relinearization loop** (`relinearize_to_convergence`): plain Gauss-Newton on the QR system, with no Levenberg-Marquardt damping and no accept/reject test:

$$\boldsymbol{\delta} = R^{-1} d, \qquad \bar{X}_k \leftarrow \bar{X}_k \, \mathrm{Exp}(\boldsymbol{\delta}_k), \qquad \text{rebuild } R, d \text{ at the new } \bar{X}$$

It stops when ${\lVert \boldsymbol{\delta} \rVert <}$ `--gn-tol` (default $10^{-6}$), or after `--gn-max-iters` steps (default 10). $R$ and $d$ are rebuilt before each convergence check, so the returned $R$ and $d$ always belong to the returned ${\bar{X}}$.

**Givens row insertion** (`qr_insert_row` in `utils.py`): this absorbs one new whitened row ${(\mathbf{a}, \beta)}$ into ${(R, d)}$ without refactorizing. For each column $c$ where ${\lvert a_c \rvert \ge 10^{-14}}$, a plane rotation mixes row $c$ of the system with the new row so that $a_c$ becomes 0:

$$\rho = \sqrt{R_{cc}^2 + a_c^2}, \qquad \gamma = \frac{R_{cc}}{\rho}, \qquad \sigma = \frac{a_c}{\rho}$$

$$\begin{bmatrix} R_{c,\,c:} & d_c \\ 
\mathbf{a}_{c:} & \beta \end{bmatrix} \leftarrow \begin{bmatrix} \gamma & \sigma \\ 
-\sigma & \gamma \end{bmatrix} \begin{bmatrix} R_{c,\,c:} & d_c \\ 
\mathbf{a}_{c:} & \beta \end{bmatrix}$$

Each rotation is orthogonal, so after the sweep ${R^\top R}$ has gained exactly ${\mathbf{a}\mathbf{a}^\top}$ and ${R^\top d}$ has gained $`\beta \, \mathbf{a}`$. $R$ stays upper triangular, and the leftover $\beta$ is discarded. Two details are easy to miss:

- The rotation also works when $R_{cc} = 0$. Then $\gamma = 0$ and $\sigma = \pm 1$, so the rotation just swaps the new row into row $c$. This is what happens at a new node's still-empty diagonal block.
- The docstring's "O(m)" is the cost of one rotation, not of one row. A row whose first nonzero is at column $c_0$ can trigger up to ${m - c_0}$ rotations, so the worst case is ${O(m^2)}$ per row. An odometry edge into node $k$ starts at ${c_0 = 6(k-1)}$, so each of its rows needs at most 12 rotations, each at most 12 columns wide, no matter how long the trajectory is. The loop still scans the leading zero columns, which is O(m) per row.

**Streaming a new node** (`run_incremental_pose_graph`): for node $k$ with odometry edge ${Z_{k-1,k}}$:

1. Dead-reckon its linearization point from the previous one: $`\bar{X}_k = \bar{X}_{k-1} Z_{k-1,k}`$.
2. Pad $R$ and $d$ with 6 zero rows and columns.
3. Insert the edge's 6 whitened rows one at a time with `qr_insert_row`.
4. Read out the estimate (`read_out`): ${\boldsymbol{\delta} = R^{-1} d}$, then $`X_k = \bar{X}_k \, \mathrm{Exp}(\boldsymbol{\delta}_k)`$ for every node.

Between rebuilds, the linearization points ${\bar{X}}$ of old nodes never move. Only $\boldsymbol{\delta}$ changes, so old rows keep the Jacobians they were built with.

**Relinearization triggers**: a full `relinearize_to_convergence` runs once at the start (node 0 only), then whenever ${k \bmod r = 0}$ with $r$ = `--relinearize-every` (default 8), and at the last node when a loop-closure edge exists. It starts from the read-out estimate $X$, not from ${\bar{X}}$. The loop-closure edge is never inserted with Givens rotations. It enters $R$ only through that final full rebuild. The module docstring gives the reason: a loop closure affects many old poses, so it would touch nearly every column of $R$ anyway. Here the edge connects the last node to node 0, so a Givens sweep would start at column 0.

**Defaults**: `--nodes-per-side 16` gives 64 nodes, which stream in as ${k = 1, \ldots, 63}$. That makes 9 full relinearizations: the initial one, 7 periodic ones (${k = 8, 16, \ldots, 56}$) and the loop-closure one at ${k = 63}$. The `run_batch_streaming` baseline instead runs `pose_graph.py`'s damped solver from scratch 63 times, starting at `--damping 0.01`. With `--seed 0`, both solvers finish at the same final pose error (0.0334 m). The `use_manif/` twin has the same flow, triggers and defaults. It retracts with `x_lin[k] + manif.SE3Tangent(delta_k)`.

---

## 15. References

1. Dellaert, F., & Kaess, M. (2006). *Square Root SAM: Simultaneous Localization and Mapping via Square Root Information Smoothing*. International Journal of Robotics Research, 25(12), 1181–1203. https://doi.org/10.1177/0278364906072768 - the sparse QR/square-root-information factorization behind §3's $A \approx QR$ and the claim that most of the previous factorization stays reusable.
2. Kaess, M., Ranganathan, A., & Dellaert, F. (2008). *iSAM: Incremental Smoothing and Mapping*. IEEE Transactions on Robotics, 24(6), 1365–1378. https://doi.org/10.1109/TRO.2008.2006706 - the original iSAM algorithm (incremental QR updates via Givens rotations, with periodic variable reordering) behind §1, §2, §4–§6, and §9.
3. Kaess, M., Johannsson, H., Roberts, R., Ila, V., Leonard, J. J., & Dellaert, F. (2012). *iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree*. International Journal of Robotics Research, 31(2), 216–235. https://doi.org/10.1177/0278364911430419 - the Bayes-tree data structure and fluid relinearization behind §7, §11, and §13's iSAM2 description. Covered in full in [`isam2_optimization.md`](isam2_optimization.md) and [`bayes_tree.md`](bayes_tree.md).
4. Dellaert, F., & Kaess, M. (2017). *Factor Graphs for Robot Perception*. Foundations and Trends in Robotics, 6(1–2), 1–139. https://doi.org/10.1561/2300000043 - general reference for the factor-graph formulation underlying §3 and §11.
