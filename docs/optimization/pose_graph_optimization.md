# Pose-graph optimization

**Pose-graph optimization (PGO)** is probably one of the easiest SLAM concepts to understand once you have the right mental picture.

The key idea is:

> **Pose-graph optimization adjusts all robot poses so that the relative-motion and loop-closure constraints are as consistent as possible.**

Think of it as **"fixing the robot's entire trajectory using a network of geometric constraints."**

---

## 1. First: what is a "pose"?

A robot's **pose** is its position + orientation.

For a 2D robot:

$${x_i = [x,\ y,\ \theta]}$$

For a 3D robot:

$${T_i \in SE(3)}$$

which contains:

* 3D position
* 3D orientation

Imagine the robot traveling:

```text
t₀       t₁       t₂       t₃       t₄

🚗───────🚗───────🚗───────🚗───────🚗
```

Each robot location is a **node**:

```text
x₀ ─── x₁ ─── x₂ ─── x₃ ─── x₄
```

That's the **pose graph**.

---

## 2. Where do the edges come from?

Suppose the robot moves from $x_0$ to $x_1$.

From [odometry](factor_graph.md#2-why-do-we-need-it) or visual odometry, we estimate:

> "The robot moved approximately 1 meter forward."

That's a constraint between the two poses:

```text
x₀ ───────── x₁
      Δ₀₁
```

Similarly:

```text
x₀ ──Δ₀₁── x₁ ──Δ₁₂── x₂ ──Δ₂₃── x₃
```

Each edge says:

> **"The relative transformation between these two poses should approximately equal this measurement."**

> **Note**: "odometry" and "visual odometry" aren't the same sensor, just the same *kind* of measurement from different sources - wheel odometry counts wheel rotations, while visual odometry (VO) tracks features across camera frames to estimate the same incremental relative pose. The distinction matters for this graph's edges specifically because a single (monocular) camera used this way can only recover relative motion up to an unknown **scale** factor - "the camera moved some distance" could mean 1 meter or 100 - see [vi_initialization.md §1](../frontend/vi_initialization.md#1-the-bootstrapping-problem) for why. This doc's $\Delta_{ij}$ edges are treated as already-metric (as from wheel odometry, stereo VO, or monocular VO with scale recovered via IMU fusion), which is exactly why - unlike monocular bundle adjustment's projective edges, whose unresolved scale is discussed in [umeyama_alignment.md](../foundations/umeyama_alignment.md) - a pose graph's only gauge freedom is the rigid 6-DoF one noted in [§7](#7-the-mathematics-is-actually-quite-intuitive), never a scale ambiguity.

---

## 3. Why do we need optimization?

Because measurements are noisy.

Suppose the robot actually walks in a square:

```text
      x₃ ───── x₄
      │         │
      │         │
      x₂         x₅
      │         │
      │         │
      x₁ ───── x₀
```

But odometry has small errors.

The robot might estimate:

```text
x₀ ───── x₁
          \
           x₂
            \
             x₃
               \
                x₄
```

After enough motion, small errors accumulate.

This is called **drift**.

---

## 4. The really important event: loop closure

Now suppose the robot eventually recognizes:

> "Hey! I've been here before."

For example, it recognizes the same visual landmark/place corresponding to $x_0$.

We obtain a loop-closure constraint:

```text
x₀ ───── x₁ ───── x₂ ───── x₃ ───── x₄
│                                   │
└──────────── loop closure ─────────┘
```

This is extremely valuable.

It says:

> **"According to this observation, $x_4$ should be near $x_0$ with approximately this relative orientation."**

But our accumulated odometry says otherwise.

Now we have a conflict.

---

## 5. Pose-graph optimization resolves the conflict

We have:

### Odometry says:

$$x_0 \to x_1 \to x_2 \to x_3 \to x_4$$

### Loop closure says:

$$x_4 \to x_0$$

The measurements aren't perfectly consistent because they're noisy.

So PGO asks:

> **"Can I slightly move all these poses so that all constraints are satisfied as well as possible?"**

This is the crucial intuition.

It doesn't necessarily say:

> "$x_4$ is wrong."

Instead, it says:

> "Maybe $x_1$, $x_2$, $x_3$ and $x_4$ are all slightly wrong. Let's distribute the error."

---

## 6. Imagine stretching a rubber band

This is my favorite analogy.

Imagine every pose is a bead:

```text
●────●────●────●────●
```

And every edge is a **rubber band** telling neighboring poses:

> "You should be approximately this far apart and oriented this way."

Then loop closure adds another rubber band:

```text
●────●────●────●────●
│                   │
└───────────────────┘
```

But the rubber bands are pulling in slightly conflicting directions.

If you release the system:

```text
      ●────●
     /      \
    ●        ●
     \      /
      ●────●
```

the beads settle into a configuration that best satisfies all the rubber bands.

That's essentially what optimization is doing.

---

## 7. The mathematics is actually quite intuitive

For every edge, we have:

$${z_{ij}}$$

which is the **measured relative transformation** between poses $i$ and $j$.

Given our current estimates $T_i$ and $T_j$, we can calculate what relative transformation they imply:

$${T_i^{-1}T_j}$$

Then compare:

$${\text{error}_{ij} = z_{ij}^{-1}(T_i^{-1}T_j)}$$

That comparison is still a **group element** ($SE(2)$/$`SE(3)`$, not a plain vector), so to actually measure "how big" it is - and to compute the Jacobians the optimizer needs - we take its **Log map**, which turns it into a tangent-space vector:

$${e_{ij} = \text{Log}(\text{error}_{ij}) = \text{Log}\big(z_{ij}^{-1}(T_i^{-1}T_j)\big)}$$

That vector $e_{ij}$ is the thing that actually gets squared below.

Conceptually:

```text
measured relationship
        ↓
      zᵢⱼ

estimated relationship
        ↓
   Tᵢ⁻¹ Tⱼ

        ↓
     compare

        ↓
      error (group element)
        ↓
     Log map
        ↓
   vector eᵢⱼ
```

Then PGO minimizes the total error over every edge in the graph (odometry edges plus loop-closure edges):

$${\boxed{\min_{T_0,\ldots,T_n}\sum_{(i,j) \in \mathcal{E}}\|e_{ij}\|^2}}$$

So in plain English:

> **Find the poses that make all the measured relative transformations agree as much as possible.**

One subtlety this formula hides: since every constraint is *relative*, rigidly translating and rotating the entire graph together leaves every $e_{ij}$ completely unchanged - the optimization has a flat direction with zero curvature, called **gauge freedom**. In practice this is fixed by anchoring one pose (usually $T_0$), e.g. by giving it an enormous information weight so the linear system solved at each step has a unique solution instead of infinitely many equally-good ones.

---

## 8. Why is this different from Bundle Adjustment?

This is an extremely important distinction.

### Bundle Adjustment

Optimizes:

```text
Camera poses
     +
3D landmarks
```

using:

```text
image observations
      ↓
reprojection error
```

Conceptually:

```text
        📷 T₀
       /  \
      /    \
    P₁      P₂
     \      /
      \    /
       📷 T₁
```

### Pose-graph optimization

Usually optimizes:

```text
Robot poses
     +
relative-pose constraints
```

without explicitly optimizing the 3D landmarks.

```text
T₀ ───── T₁ ───── T₂ ───── T₃
 \                         /
  └────── loop closure ───┘
```

So:

> **BA asks:**
> "Do my cameras and 3D points explain the images?"

> **PGO asks:**
> "Do my robot poses form a trajectory consistent with all the relative-pose measurements?"

---

## 9. Another useful analogy: GPS navigation

Imagine you are reconstructing someone's journey.

You have:

* odometry
* GPS
* landmarks
* loop closures

Your odometry says:

```text
Home → A → B → C → D
```

But accumulated error makes `D` appear 20 m away from Home.

Then GPS tells you:

> "D is actually very close to Home."

Instead of moving only `D`, you could distribute the correction:

```text
Before:

Home ─ A ─ B ─ C ───────── D
                         20m error


After:

Home ─ A ─ B ─ C ─────── D
       ↘   ↘   ↘   ↘
        small corrections
```

The trajectory becomes globally consistent.

That's essentially what PGO does.

---

## 10. Why loop closure is so powerful

Without loop closure:

```text
x₀ ─ x₁ ─ x₂ ─ x₃ ─ x₄ ─ x₅ ─ x₆
```

The graph is basically a chain.

There's not much opportunity to correct accumulated drift.

With loop closure:

```text
┌────────────────────────┐
↓                        │
x₀ ─ x₁ ─ x₂ ─ x₃ ─ x₄ ─ x₅
```

we suddenly have a **cycle**.

That cycle provides a powerful consistency check.

You can think of it as:

> **"If I follow the measurements around this loop, I should eventually come back to where I started."**

If I don't, there is accumulated error.

Optimization distributes that error.

---

## 11. What happens in a real SLAM system?

A typical pipeline looks roughly like:

```text
Camera / LiDAR / IMU
        │
        ▼
 Front-end estimation
        │
        ▼
Relative pose
        │
        ▼
   New pose xᵢ
        │
        ▼
   Pose graph
        │
        ├──────────────┐
        │              │
        ▼              ▼
Odometry edge     Loop closure
        │              │
        └──────┬───────┘
               ▼
        Graph optimization
               │
               ▼
     Corrected trajectory
```

The **front-end** says:

> "I think I moved like this."

The **back-end** says:

> "Let's see whether all those estimates make sense together."

This front-end/back-end separation is very important in SLAM.

---

## 12. One subtle point: PGO doesn't magically know the correct trajectory

Suppose you have:

```text
x₀ ───── x₁ ───── x₂
```

and noisy measurements.

Optimization isn't discovering some objectively "true" trajectory.

It's finding:

> **the trajectory that best satisfies the available constraints according to the chosen error model and weights.**

For example, if one measurement is considered highly reliable:

$${w_1 = 100}$$

and another is noisy:

$${w_2 = 1}$$

the optimizer will care much more about satisfying the first constraint.

So the more complete objective is something like:

$${\min_X \sum_{(i,j) \in \mathcal{E}} e_{ij}^\top \Omega_{ij} e_{ij}}$$

where $\Omega_{ij}$ is related to the **information/covariance** of the measurement.

This is why sensor uncertainty matters - though it's worth noting that `pose_graph.py` (both the `use_numpy/` and `use_manif/` versions), the toy implementations accompanying this doc, keep things simple: they share one identity `info_matrix` across every edge (odometry and loop-closure alike), so they don't actually exploit per-edge weighting the way $\Omega_{ij}$ above suggests - even though the loop-closure edge is generated with a different noise level than the odometry edges. Per-edge weighting like this is a natural extension, not something the default scripts do.

---

## 13. The most important intuition

The three concepts above connect like this:

```text
                SLAM
                 │
        ┌────────┴────────┐
        │                 │
    Filtering         Optimization
        │                 │
        │          ┌──────┴──────┐
        │          │             │
       EKF      Pose Graph       BA
                  │              │
                  │              │
             poses only     poses + landmarks
                  │              │
                  ▼              ▼
             relative       reprojection
              pose errors       errors
```

And the three questions become:

### Filtering

> **"Given everything I've seen so far, where am I now?"**

### Pose-graph optimization

> **"Given all these relative-pose constraints, what trajectory is most consistent?"**

### Bundle adjustment

> **"Given all these images, what camera trajectory and 3D structure best explain the observations?"**

---

## 14. The one-sentence mental model

If you remember only one thing:

> **Pose-graph optimization is like taking a trajectory made of slightly inaccurate pieces, connecting those pieces with constraints—including loop closures—and then moving the poses around until the entire graph becomes as geometrically consistent as possible.**

There's also a particularly important connection worth making explicit: **PGO is essentially a sparse nonlinear least-squares problem over poses on $SE(2)$ or $SE(3)$**. Once you understand that, the next natural step is understanding **why we need Lie groups / Lie algebra and how Gauss–Newton or Levenberg–Marquardt actually moves the poses during optimization**.

---

## 15. Mathematical breakdown of the error formulation and Lie algebra operations

Pose-Graph Optimization (PGO) formulates loop closure and drift correction as a non-linear least squares problem on the Special Euclidean Group $\mathrm{SE}(3)$ (or $\mathrm{SE}(2)$ for 2D). Because $\mathrm{SE}(3)$ is a non-Euclidean Lie group rather than a vector space, standard calculus operations like addition and subtraction do not apply directly. Instead, optimization is performed locally on its Lie algebra $\mathfrak{se}(3)$ using tangent spaces.

### 15.1 State Representation and Constraints

#### Poses as Lie Group Elements

A 3D pose consists of a rotation $R \in \mathrm{SO}(3)$ and a translation $p \in \mathbb{R}^3$, represented as a $4 \times 4$ matrix $T_i \in \mathrm{SE}(3)$:

$$T_i = \begin{bmatrix} R_i & p_i \\ 
\mathbf{0}^\top & 1 \end{bmatrix} \in \mathrm{SE}(3)$$

The full state vector containing all $N$ pose keyframes is $X = \{T_1, T_2, \dots, T_N\}$.

#### Relative Edge Measurements

An edge $e_{ij}$ between nodes $i$ and $j$ represents a relative transformation measurement

$${z_{ij} = {\tilde{T}_{ij} \in \mathrm{SE}(3)}}$$ 

(e.g., from ICP scan matching or visual odometry), accompanied by an information matrix ${\Omega_{ij} = {\Sigma_{ij}^{-1} \in \mathbb{R}^{6 \times 6}}}$ representing measurement confidence.

### 15.2 Residual Vector Formulation on $\mathrm{SE}(3)$

The expected relative transformation between pose $T_i$ and pose $T_j$ according to the current state estimate is:

$${\hat{T}_{ij} = T_i^{-1} T_j}$$

The error matrix ${E_{ij} \in \mathrm{SE}(3)}$ measures the relative deviation between the actual measurement ${\tilde{T}_{ij}}$ and the predicted state transformation ${T_i^{-1} T_j}$:

$${E_{ij} = \tilde{T}_{ij}^{-1} \left( T_i^{-1} T_j \right)}$$

#### Mapping Error to Tangent Space ${\mathfrak{se}(3)}$

Because optimization requires a 6-dimensional Euclidean vector space, the matrix error ${E_{ij}}$ is mapped to its local tangent space (Lie algebra ${\mathfrak{se}(3)}$) via the logarithmic map ${\log: \mathrm{SE}(3) \to \mathfrak{se}(3)}$, and flattened into a vector ${\mathbb{R}^6}$ using the **${\vee}$ operator** ${(\cdot)^\vee}$ - together, ${\mathrm{Log}(\cdot) = (\log(\cdot))^\vee: \mathrm{SE}(3) \to \mathbb{R}^6}$, mirroring the lowercase/uppercase convention already used for ${\exp}$/$`{\mathrm{Exp}}`$ below:

$${r_{ij}(X) = \left( \log \left( \tilde{T}_{ij}^{-1} T_i^{-1} T_j \right) \right)^\vee \in \mathbb{R}^6}$$

The residual vector:

$${r_{ij} = \left[ \boldsymbol{\rho}_{ij}^\top \theta_{ij}^\top \right]^\top}$$

captures 3D translational error:

$${\boldsymbol{\rho}_{ij}}$$

and rotational error:

$${\theta_{ij}}$$

### 15.3 Objective Function

The global optimization minimizes the sum of squared Mahalanobis distances over all edges ${\mathcal{E}}$ in the graph:

$${F(X) = \sum_{(i,j) \in \mathcal{E}} r_{ij}(X)^\top \Omega_{ij} \, r_{ij}(X)}$$

> **Note**: Mahalanobis distance measures how far a point is from the center (mean) of a distribution, accounting for the correlations and variances between variables. Here the "point" is the residual ${r_{ij}(X)}$, the "distribution" is the measurement noise model (mean $\mathbf{0}$, covariance ${\Sigma_{ij} = \Omega_{ij}^{-1}}$), and ${\Omega_{ij}}$ is exactly the inverse-covariance weighting that turns a plain squared-error sum into a squared Mahalanobis-distance sum. See the tilestats.com video and amit's Medium explainer cited in [§17](#17-references), and [the isotropic special case](../filtering/pointcloud_pose_tracking_empirical_note.md#a3-mahalanobis-distance-and-the-information-matrix), where this weighting collapses to a uniform scale factor.

### 15.4 Manifold Optimization and Linearization

Standard vector updates ${T_i \leftarrow T_i + \Delta x_i}$ break the matrix constraints of ${\mathrm{SE}(3)}$ (e.g., $R_i$ will cease to be orthogonal). Updates are applied using the exponential map ${\mathrm{Exp}: \mathbb{R}^6 \to \mathrm{SE}(3)}$ via local perturbations ${\boldsymbol{\xi}_i \in \mathbb{R}^6}$ acting on the tangent space.

#### Local Perturbation Model (Left / Right Multiplication)

Applying a local perturbation ${\boldsymbol{\xi}_i = \left[ \boldsymbol{\rho}^\top \;\; \boldsymbol{\phi}^\top \right]^\top \in \mathbb{R}^6}$ to state $T_i$:

$${T_i \oplus \boldsymbol{\xi}_i = T_i \cdot \mathrm{Exp}(\boldsymbol{\xi}_i)}$$

where ${\mathrm{Exp}(\boldsymbol{\xi}) = \exp(\boldsymbol{\xi}^\wedge) \in \mathrm{SE}(3)}$, and ${(\cdot)^\wedge}$ maps a 6D vector to a ${4 \times 4}$ Lie algebra element ${\mathfrak{se}(3)}$:

$${\boldsymbol{\xi}^\wedge = \begin{bmatrix} \boldsymbol{\phi}^\wedge & \boldsymbol{\rho} \\ 
\mathbf{0}^\top & 0 \end{bmatrix}, \quad \text{with } \boldsymbol{\phi}^\wedge = \begin{bmatrix} 0 & -\phi_z & \phi_y \\ 
\phi_z & 0 & -\phi_x \\ 
-\phi_y & \phi_x & 0 \end{bmatrix} \in \mathfrak{so}(3)}$$

#### First-Order Taylor Expansion

Linearizing the residual $r_{ij}$ with respect to local perturbations ${\boldsymbol{\xi}_i}$ and ${\boldsymbol{\xi}_j}$:

$$r_{ij}(X \oplus \boldsymbol{\delta}) \approx r_{ij}(X) + J_i \, \boldsymbol{\xi}_i + J_j \, \boldsymbol{\xi}_j$$

Where the Jacobians

$$
J_i = \frac{\partial r_{ij}}{\partial \pmb{\xi}_i}
$$

and ${J_j = \frac{\partial r_{ij}}{\partial \boldsymbol{\xi}_j}}$ are the exact right Jacobians of $\mathrm{Log}$. They are not a truncated approximation. `use_numpy/` computes ${J_r^{-1}}$ as the matrix inverse of a series (see [§15.7](#157-the-solver-concretely)), and `use_manif/` uses manif's closed-form Jacobians. For small residuals the two agree to machine precision. The series' truncation error grows with the residual's size, and is about $10^{-12}$ at a 1.6 rad rotation:

$${J_j = J_r^{-1}(r_{ij})}$$

$${J_i = - J_r^{-1}(r_{ij}) \, \mathrm{Ad}\left( T_j^{-1} T_i \right)}$$

Here, ${\mathrm{Ad}(T) \in \mathbb{R}^{6 \times 6}}$ is the **Adjoint transformation matrix** of ${\mathrm{SE}(3)}$, which transforms velocity/tangent vectors between frame coordinate systems:

$${\mathrm{Ad}\left(\begin{bmatrix} R & p \\ 
\mathbf{0}^\top & 1 \end{bmatrix}\right) = \begin{bmatrix} R & p^\wedge R \\ 
\mathbf{0} & R \end{bmatrix}}$$

and ${J_r^{-1}(\cdot)}$ is the inverse right Jacobian of ${\mathrm{SE}(3)}$.

### 15.5 Solving the Linear System (Gauss-Newton Step)

Stacking all residuals into a global residual vector $R(X)$ and Jacobians into a sparse Jacobian matrix $J$, the linearization takes the standard form:

$${H \, \boldsymbol{\delta}^* = -b}$$

* **Hessian Matrix:** ${H = J^\top \Omega J = \sum_{(i,j) \in \mathcal{E}} J_{ij}^\top \Omega_{ij} J_{ij} \in \mathbb{R}^{6N \times 6N}}$
* **Gradient Vector:** ${b = J^\top \Omega R(X) \in \mathbb{R}^{6N}}$
* **Update Vector:** ${\boldsymbol{\delta}^* = \left[ \boldsymbol{\xi}_1^\top \;\; \boldsymbol{\xi}_2^\top \;\; \dots \;\; \boldsymbol{\xi}_N^\top \right]^\top}$

Because edges only connect adjacent or loop-closing keyframes, $H$ is extremely **sparse** and block-structured. It is typically solved using Sparse Cholesky Factorization (${\mathrm{LL}^\top}$ or ${\mathrm{LDL}^\top}$) or Conjugate Gradients in solvers like GTSAM or g2o — see [`sparse_cholesky_factorization.md`](sparse_cholesky_factorization.md) for the full derivation.

**What that factorization is doing:** $H$ is symmetric positive-definite, and Cholesky factorization writes it as $H = LL^\top$ with $L$ lower-triangular. Solving $H\boldsymbol{\delta}^* = -b$ then becomes two cheap triangular solves instead of one general one:

$$Ly = -b \quad\text{(forward substitution)}, \qquad L^\top \boldsymbol{\delta}^* = y \quad\text{(back substitution)}$$

Each pass is just row-by-row substitution — no matrix inversion needed.

**Why "sparse" matters:** most pose pairs never share a constraint, so most of $H$'s off-diagonal blocks are exactly zero. Sparse Cholesky exploits that known zero pattern instead of doing dense arithmetic on entries it already knows are zero. One subtlety: eliminating a variable can turn some of those zeros into nonzeros — called **fill-in**. For example, eliminating $x_2$ out of a chain $x_1 - x_2 - x_3$ creates a new dependency between $x_1$ and $x_3$ even though they were never directly measured. Which order variables are eliminated in controls how much fill-in accumulates; see [`elimination_tree.md`](elimination_tree.md) for a worked elimination example, and [`isam2_optimization.md` §7](isam2_optimization.md#7-bayes-tree--the-most-important-intuition) for why iSAM2 cares about this at all.

In practice a pure Gauss-Newton step can overshoot or diverge far from the solution, so a **Levenberg-Marquardt** damping term $\lambda$ is added to the Hessian's diagonal before solving. Both `pose_graph.py` implementations (`use_numpy/` and `use_manif/`) use the Marquardt-scaled form, which scales $\lambda$ by $H$'s own diagonal instead of adding $\lambda I$:

$${\big(H + \lambda \, \mathrm{diag}(H)\big) \, \boldsymbol{\delta}^* = -b}$$

Larger $\lambda$ shrinks the step toward (diagonally scaled) gradient descent: safer, but slower. ${\lambda \to 0}$ recovers pure Gauss-Newton, which is faster near convergence. In both scripts $\lambda$ is **adaptive**, not fixed, and the two implementations are identical here:

1. Entries of $\mathrm{diag}(H)$ below $10^{-12}$ are clamped to $10^{-12}$.
2. A trial step is accepted only if it lowers the total cost $F$. On acceptance, $\lambda \leftarrow \max(0.5\lambda,\ 10^{-7})$.
3. On rejection, $\lambda \leftarrow \max(2\lambda,\ 10^{-6})$ and the solve is retried. Each iteration tries at most 10 values of $\lambda$ in total. If all 10 are rejected, the solver stops.

`--damping` (default 0.01) is only the starting $\lambda$.

### 15.6 Retraction / State Update

Once the increment vector ${\boldsymbol{\delta}^*}$ is computed, the system updates the trajectory states on the ${\mathrm{SE}(3)}$ manifold:

$${T_i^{(k+1)} = T_i^{(k)} \cdot \mathrm{Exp}\left(\boldsymbol{\xi}_i^*\right), \quad \forall i \in \{1, \dots, N\}}$$

This iteration repeats until the accepted step is small, ${\Vert{}\boldsymbol{\delta}^*\Vert{} < \epsilon}$ (`--gn-tol`, default $10^{-6}$). It also stops after `--gn-max-iters` accepted iterations (default 10), or when every damping retry is rejected. The scripts have no separate test on the change in cost ${\Delta F}$.

### 15.7 The solver, concretely

The whole solver is `run_pose_graph_optimization` in [`use_numpy/pose_graph.py`](../../use_numpy/pose_graph.py). It works on $4 \times 4$ pose matrices $X_k$. Tangent vectors are ordered translation first, $`\boldsymbol{\xi} = [\mathbf{v}^\top \;\; \boldsymbol{\omega}^\top]^\top`$, the same order as §15.4's $`[\boldsymbol{\rho}^\top \;\; \boldsymbol{\phi}^\top]^\top`$. The code numbers nodes from 0, while §15.1 numbers them from 1. Below, $X$ is §15's $T$ and $Z_{ij}$ is §15's ${\tilde{T}_{ij}}$.

**Measurements** (`simulate_noisy_edges`): each edge is the true relative pose with right-multiplied noise. The loop-closure edge scales both std-devs by `--loop-noise-scale` (default 0.5):

$$Z_{ij} = \left(X_i^{\text{true}}\right)^{-1} X_j^{\text{true}} \, \mathrm{Exp}(\mathbf{n}), \qquad \mathbf{n} \sim \mathcal{N}\!\left(\mathbf{0},\ \mathrm{diag}(\sigma_p^2 I_3,\ \sigma_r^2 I_3)\right)$$

The defaults are ${\sigma_p = 0.05}$ m (`--pos-noise-std`) and ${\sigma_r = 0.01}$ rad (`--rot-noise-std`). The initial guess is the dead-reckoned chain $X_{k+1} = X_k Z_{k,k+1}$ (`run_dead_reckoning`), which never uses the loop-closure edge.

**Residual** (the edge loop and `cost`): the code first predicts node $j$ from node $i$, then compares:

$$\hat{X}_j = X_i Z_{ij}, \qquad e_{ij} = \mathrm{Log}\big(\hat{X}_j^{-1} X_j\big) = \mathrm{Log}\big(Z_{ij}^{-1} X_i^{-1} X_j\big)$$

This is §15.2's $r_{ij}$. In code, $\hat{X}_j$ is `T_pred`.

**Jacobians** (the edge loop): the chain rule through `T_pred`, with right perturbations $`X \leftarrow X \, \mathrm{Exp}(\boldsymbol{\xi})`$:

$$J_c = \mathrm{Ad}\big(Z_{ij}^{-1}\big), \qquad J_a = J_r^{-1}(e_{ij}), \qquad J_b = -J_r^{-1}(-e_{ij})$$

$$J_i = J_b \, J_c, \qquad J_j = J_a$$

These are `Jc_self` ($\partial \hat{X}_j / \partial X_i$), `Ja` ($\partial e_{ij} / \partial X_j$) and `Jb` ($\partial e_{ij} / \partial \hat{X}_j$). `use_manif/` gets the same three matrices from `Xi.compose(Z_ij, Jc_self)` and `Xj.rminus(T_pred, Ja, Jb)`. The product equals §15.4's closed form:

$$-J_r^{-1}(-e_{ij}) \, \mathrm{Ad}\big(Z_{ij}^{-1}\big) = -J_r^{-1}(e_{ij}) \, \mathrm{Ad}\big(X_j^{-1} X_i\big)$$

This holds because $`J_r(-e) = J_l(e) = \mathrm{Ad}(\mathrm{Exp}(e)) \, J_r(e)`$ and ${\mathrm{Exp}(e_{ij})^{-1} = X_j^{-1} X_i Z_{ij}}$.

**Right Jacobian** (`se3_right_jacobian`, `compute_se3_inv_right_jacobian` in `use_numpy/lie_utils.py`): an 18-term series in the little adjoint (`se3_ad`), then a plain matrix inverse:

$$J_r(\boldsymbol{\xi}) = \sum_{n=0}^{17} \frac{\left(-\mathrm{ad}_{\boldsymbol{\xi}}\right)^n}{(n+1)!}, \qquad \mathrm{ad}_{\boldsymbol{\xi}} = \begin{bmatrix} \boldsymbol{\omega}^\wedge & \mathbf{v}^\wedge \\ 
\mathbf{0} & \boldsymbol{\omega}^\wedge \end{bmatrix}, \qquad J_r^{-1} = \big(J_r\big)^{-1}$$

The series needs no small-angle branch. `se3_adjoint` builds $\mathrm{Ad}$ exactly as in §15.4.

**Assembly** (the edge loop): each edge adds four blocks to $H$ and two to $g$, with $\Omega$ = `info_matrix`:

$$H_{ii} \mathrel{+}= J_i^\top \Omega J_i, \quad H_{jj} \mathrel{+}= J_j^\top \Omega J_j, \quad H_{ij} \mathrel{+}= J_i^\top \Omega J_j, \quad H_{ji} \mathrel{+}= J_j^\top \Omega J_i$$

$$g_i \mathrel{-}= J_i^\top \Omega \, e_{ij}, \qquad g_j \mathrel{-}= J_j^\top \Omega \, e_{ij}$$

So $g = -b$ in §15.5's notation, and the code solves $`(H + \lambda \, \mathrm{diag}(H)) \boldsymbol{\delta} = g`$. `main` sets $\Omega = I_6$ for every edge, including the loop closure.

**Anchor** (gauge fix): before the edges are added, ${H_{00} \mathrel{+}= 10^6 I_6}$, and nothing is added to $g$. This is a prior that pulls node 0's step ${\boldsymbol{\delta}_0}$ toward zero. It is centered on node 0's current estimate, not on a fixed pose. Two side effects are easy to miss:

- The anchor is not part of `cost`, so the accept/reject test ignores it.
- Because it sits in $\mathrm{diag}(H)$, the Marquardt term damps node 0 far more than the other nodes.

**Cost** (`cost`): $`F = \sum e_{ij}^\top \Omega \, e_{ij}`$, with no $\frac{1}{2}$ factor. It is the value printed as "total chi-squared error" and the value the accept test compares.

**Retraction**: every node, node 0 included, is updated as $`X_k \leftarrow X_k \, \mathrm{Exp}(\boldsymbol{\delta}_k)`$ (`se3_exp`). `use_manif/` writes the same update as `T + manif.SE3Tangent(delta_k)`.

**Defaults**: `--damping 0.01` (initial $\lambda$), `--gn-tol 1e-6`, `--gn-max-iters 10`, `--side-length 2.0`, one node per square corner (4 nodes, 3 odometry edges and 1 loop closure). With `--seed 0` the default run converges in 5 iterations.

---

## 16. Robust loss functions used to handle false loop closures

In Pose-Graph Optimization (PGO), standard non-linear least squares relies on an $L_2$ squared-error norm ${F(x) = \sum r_{ij}^\top \Omega_{ij} r_{ij}}$. Under an $L_2$ loss, a single false loop closure (a severe outlier) produces a massive residual $r_{ij}$ whose squared weight pulls the entire trajectory out of shape to satisfy the invalid edge.

Robust cost functions replace or reweight the standard $L_2$ norm to cap or reduce the influence of large residuals.

### 16.1 The M-Estimator Framework (Iteratively Reweighted Least Squares)

Instead of minimizing $\frac{1}{2} e^2$ (where $e = \sqrt{r^\top \Omega r}$ is the normalized residual scalar), M-estimators minimize a robust kernel $\rho(e)$:

$$\min_{X} \sum_{(i,j) \in \mathcal{E}} \rho\left( \sqrt{r_{ij}(X)^\top \Omega_{ij} \, r_{ij}(X)} \right)$$

To integrate this into standard Gauss-Newton or Levenberg-Marquardt solvers without modifying the core linear algebra solver, robust kernels use **Iteratively Reweighted Least Squares (IRLS)**. The robust cost is converted into a modified information matrix $\Omega_{ij}^\text{robust} = w(e) \cdot \Omega_{ij}$, where the weight function $w(e)$ is:

$$w(e) = \frac{1}{e} \frac{\partial \rho(e)}{\partial e}$$

### 16.2 Classical M-Estimators

#### Huber Loss

Huber acts as quadratic ($L_2$) for small residuals (inliers) and linear ($L_1$) for residuals exceeding a threshold $\delta$:

$$\rho(e) = \begin{cases} \frac{1}{2} e^2 & \text{if } \vert{}e\vert{} \le \delta \\ 
\delta \left( \vert{}e\vert{} - \frac{1}{2} \delta \right) & \text{if } \vert{}e\vert{} > \delta \end{cases}, \quad w(e) = \begin{cases} 1 & \text{if } \vert{}e\vert{} \le \delta \\ 
\frac{\delta}{\vert{}e\vert{}} & \text{if } \vert{}e\vert{} > \delta \end{cases}$$

* **Behavior:** Because $w(e) \propto \frac{1}{\vert{}e\vert{}}$, the gradient magnitude becomes constant ($\delta$) for outliers rather than growing infinitely.
* **Limitation in SLAM:** A linear error cost still grows indefinitely as $e \to \infty$. If a false loop closure has a massive initial error, Huber will still pull the graph significantly toward the false measurement.

#### Cauchy Loss

Cauchy uses a logarithmic tail that flattens out faster than Huber:

$$\rho(e) = \frac{k^2}{2} \ln\left(1 + \frac{e^2}{k^2}\right), \quad w(e) = \frac{1}{1 + \left(\frac{e}{k}\right)^2}$$

* **Behavior:** The weight falls off quadratically ($w(e) \propto \frac{1}{e^2}$), heavily suppressing high-residual edges.

### 16.3 Dynamic Covariance Scaling (DCS)

Dynamic Covariance Scaling (Agarwal et al., 2013) is specifically designed for pose-graph optimization. Instead of reweighting during every residual evaluation, DCS dynamically scales the information matrix based on an analytical closed-form solution derived from Switchable Constraints.

DCS adds a dynamic scaling parameter $s_{ij} \in (0, 1]$ directly to the information matrix $\Omega_{ij}$:

$$\Omega_{ij}^\text{DCS} = s_{ij}^2 \, \Omega_{ij}$$

The scaling factor $s_{ij}$ is calculated in closed form at each iteration using the current error $e_{ij}^2 = r_{ij}^\top \Omega_{ij} r_{ij}$ and an upper-bound parameter $\Phi$:

$$s_{ij} = \min\left(1, \; \frac{2 \Phi}{\Phi + e_{ij}^2}\right)$$

```
                       DCS Scaling Factor (s_ij)
          1.0 |────────────┐
              |             \
              |              \___   (s_ij decreases once e_ij^2 > Phi)
          0.0 +--------------------> e_ij^2 (Error)
              0            Phi

```

#### How DCS Handles Outliers:

1. **Inliers ($e_{ij}^2 \le \Phi$):** $s_{ij} = 1$. The edge retains full confidence and behaves as a standard quadratic term.
2. **Outliers ($e_{ij}^2 > \Phi$):** $s_{ij} = \frac{2 \Phi}{\Phi + e_{ij}^2} < 1$. As error $e_{ij}^2$ grows, $s_{ij}^2 \propto \frac{1}{e^4}$, causing the effective weight of the edge to drop rapidly to zero.
3. **No Extra State Variables:** Unlike original Switchable Constraints, DCS does not add auxiliary optimization variables to the Hessian matrix $H$, preserving graph sparsity without increasing matrix inversion costs.

### 16.4 Comparison of Robust Loss Functions

| Loss Function | Residual Cost Tail $\rho(e)$ | Weight Degeneration $w(e)$ | SLAM False Loop Rejection Power |
| ------------- | ------------- | ------------- | ------------- |
| **Standard $L_2$** | Unbounded Quadratic ($e^2$) | Constant ($1.0$) | **None** (1 outlier ruins the map) |
| **Huber** | Unbounded Linear ($\delta e$) | $\propto \frac{1}{e}$ | **Low/Moderate** (Dampens, but still pulls graph) |
| **Cauchy** | Logarithmic ($\ln e^2$) | $\propto \frac{1}{e^2}$ | **High** |
| **DCS** | Saturation / Bounded ($=\Phi$, see below) | $\propto \frac{1}{e^4}$ | **Very High** (Effectively turns off bad edges) |
| **Geman-McClure** | Saturation / Bounded | $\propto \frac{1}{(1 + e^2)^2}$ | **Very High** |

Note the distinction in the first column - but be careful what "DCS's cost" actually means here, since it's easy to under-count it. $s_{ij}^2 e_{ij}^2 = \frac{4\Phi^2 e^2}{(\Phi+e^2)^2}$ is only the *first* term of DCS's true objective - DCS comes from Switchable Constraints' augmented cost $\Psi(s, e) = s^2e^2 + \Phi(s-1)^2$ (the second term is the prior that keeps the switch $s$ near $1$ unless the data really justifies turning an edge off), and DCS's whole point is a closed-form $s$ that approximates the optimal solve of that *joint* cost, not $s^2e^2$ alone. Substituting $s=\frac{2\Phi}{\Phi+e^2}$ into the *full* $\Psi(s,e)$ (for $e^2>\Phi$) and simplifying: $s^2e^2 + \Phi(s-1)^2 = \frac{4\Phi^2e^2 + \Phi(\Phi-e^2)^2}{(\Phi+e^2)^2} = \frac{\Phi\left[4\Phi e^2 + (\Phi-e^2)^2\right]}{(\Phi+e^2)^2} = \frac{\Phi(\Phi+e^2)^2}{(\Phi+e^2)^2} = \Phi$ - a **constant**, independent of $e$, for every $e^2>\Phi$. So DCS's actual cost doesn't decay back to $0$ as $e\to\infty$ - it **saturates** at exactly $\Phi$, immediately upon crossing the threshold (a flatter, even more abrupt saturation than Geman-McClure's asymptotic approach to $1$). This is also the source of the DCS/Geman-McClure equivalence result (MacTavish & Barfoot, 2015): both cost functions saturate rather than diverge or decay, which is exactly why both make sense as $\frac{1}{e^4}$-weight, redescending M-estimators, and why both carry the same "Graduated Non-Convexity" caveat below.

### 16.5 Practical Considerations in Implementation

1. **Threshold Tuning ($\delta, k, \Phi$):** The parameters set the boundary between inliers and outliers. In $\mathrm{SE}(3)$ PGO, error $e^2$ follows a Chi-Square distribution ($\chi^2$) with 6 degrees of freedom. Setting $\Phi$ or $k^2$ corresponding to the 95% or 99% quantile of $\chi^2(6)$ (e.g., $\Phi \approx 12.59$) provides a sound baseline.
2. **Graduated Non-Convexity (GNC):** Highly non-convex robust functions (like DCS or Geman-McClure) can introduce local minima if applied from a poor initial guess. Modern solvers use GNC to start with a convex $L_2$ loss and gradually harden the robust kernel as iterations progress.

This section is theory only: neither `use_numpy/pose_graph.py` nor `use_manif/pose_graph.py` implements Huber, Cauchy, or DCS reweighting - both scripts still use a single, unweighted `info_matrix = np.eye(6)` shared by every edge, odometry and loop-closure alike (the same gap already noted for per-edge $\Omega_{ij}$ weighting earlier in this doc). Robust loss reweighting is a natural extension a reader could add, not something the accompanying scripts exercise.

---

## 17. References

1. Grisetti, G., Kümmerle, R., Stachniss, C., & Burgard, W. (2010). *A Tutorial on Graph-Based SLAM*. IEEE Intelligent Transportation Systems Magazine, 2(4), 31-43. https://doi.org/10.1109/MITS.2010.939925 - already cited in [frontend_backend.md §6](../frontend_backend.md#6-references); the general graph/error-formulation/optimization reference behind §1-§14 here.
2. tilestats.com (2021, February 3). *Euclidean Distance and the Mahalanobis Distance (and the Error Ellipse)* [Video]. YouTube. https://www.youtube.com/watch?v=xXhLvheEF7o - the Mahalanobis-distance intuition in §15.3.
3. amit (2024, October 30). *Understanding Mahalanobis Distance*. Medium. https://medium.com/@pamit2235/understanding-mahalanobis-distance-081bd765fcdb - the Mahalanobis-distance intuition in §15.3.
4. Huber, P. J. (1964). *Robust Estimation of a Location Parameter*. Annals of Mathematical Statistics, 35(1), 73-101. https://doi.org/10.1214/aoms/1177703732 - the Huber loss in §16.2.
5. Geman, S., & McClure, D. E. (1985). *Bayesian Image Analysis: An Application to Single Photon Emission Tomography*. Proceedings of the American Statistical Association, Statistical Computing Section, 12-18. - the Geman-McClure loss named in §16.4's comparison table.
6. Agarwal, P., Tipaldi, G. D., Spinello, L., Stachniss, C., & Burgard, W. (2013). *Robust Map Optimization Using Dynamic Covariance Scaling*. ICRA 2013, 62-69. https://doi.org/10.1109/ICRA.2013.6630557 - Dynamic Covariance Scaling in §16.3, already named inline there as "Agarwal et al., 2013".
7. MacTavish, K., & Barfoot, T. D. (2015). *At all Costs: A Comparison of Robust Cost Functions for Camera Correspondence Outliers*. 12th Conference on Computer and Robot Vision (CRV), 62-69. https://doi.org/10.1109/CRV.2015.52 - the DCS/Geman-McClure equivalence result named in §16.4.
