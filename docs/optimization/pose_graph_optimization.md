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

Suppose the robot moves from `x₀` to `x₁`.

From odometry or visual odometry, we estimate:

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

For example, it recognizes the same visual landmark/place corresponding to `x₀`.

We obtain a loop-closure constraint:

```text
x₀ ───── x₁ ───── x₂ ───── x₃ ───── x₄
│                                     │
└──────────── loop closure ───────────┘
```

This is extremely valuable.

It says:

> **"According to this observation, x₄ should be near x₀ with approximately this relative orientation."**

But our accumulated odometry says otherwise.

Now we have a conflict.

---

## 5. Pose-graph optimization resolves the conflict

We have:

### Odometry says:

```text
x₀ → x₁ → x₂ → x₃ → x₄
```

### Loop closure says:

```text
x₄ → x₀
```

The measurements aren't perfectly consistent because they're noisy.

So PGO asks:

> **"Can I slightly move all these poses so that all constraints are satisfied as well as possible?"**

This is the crucial intuition.

It doesn't necessarily say:

> "x₄ is wrong."

Instead, it says:

> "Maybe x₁, x₂, x₃ and x₄ are all slightly wrong. Let's distribute the error."

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

That comparison is still a **group element** ($SE(2)$ / $SE(3)$, not a plain vector), so to actually measure "how big" it is - and to compute the Jacobians the optimizer needs - we take its **Log map**, which turns it into a tangent-space vector:

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
    P₁     P₂
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

$${\min_X \sum_{(i,j) \in \mathcal{E}} e_{ij}^T \Omega_{ij} e_{ij}}$$

where $\Omega_{ij}$ is related to the **information/covariance** of the measurement.

This is why sensor uncertainty matters - though it's worth noting that `pose_graph.py` (both the `use_numpy/` and `use_manif/` versions), the toy implementations accompanying this doc, keep things simple: they share one identity `info_matrix` across every edge (odometry and loop-closure alike), so they don't actually exploit per-edge weighting the way $\Omega_{ij}$ above suggests - even though the loop-closure edge is generated with a different noise level than the odometry edges. Per-edge weighting like this is a natural extension, not something the default scripts do.

---

## 13. The most important intuition

You can connect the three concepts we've discussed like this:

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

And there's a particularly important connection to your SLAM research: **PGO is essentially a sparse nonlinear least-squares problem over poses on $SE(2)$ or $SE(3)$**. Once you understand that, the next natural step is understanding **why we need Lie groups / Lie algebra and how Gauss–Newton or Levenberg–Marquardt actually moves the poses during optimization**.

---

## 15. Mathematical breakdown of the error formulation and Lie algebra operations

Pose-Graph Optimization (PGO) formulates loop closure and drift correction as a non-linear least squares problem on the Special Euclidean Group $\mathrm{SE}(3)$ (or $\mathrm{SE}(2)$ for 2D). Because $\mathrm{SE}(3)$ is a non-Euclidean Lie group rather than a vector space, standard calculus operations like addition and subtraction do not apply directly. Instead, optimization is performed locally on its Lie algebra $\mathfrak{se}(3)$ using tangent spaces.

### 1. State Representation and Constraints

#### Poses as Lie Group Elements

A 3D pose consists of a rotation $R \in \mathrm{SO}(3)$ and a translation $p \in \mathbb{R}^3$, represented as a $4 \times 4$ matrix $T_i \in \mathrm{SE}(3)$:

$$T_i = \begin{bmatrix} R_i & p_i \\ 
\mathbf{0}^\top & 1 \end{bmatrix} \in \mathrm{SE}(3)$$

The full state vector containing all $N$ pose keyframes is $X = \{T_1, T_2, \dots, T_N\}$.

#### Relative Edge Measurements

An edge $e_{ij}$ between nodes $i$ and $j$ represents a relative transformation measurement

$${z_{ij} = {\tilde{T}_{ij} \in \mathrm{SE}(3)}}$$ 

(e.g., from ICP scan matching or visual odometry), accompanied by an information matrix ${\Omega_{ij} = {\Sigma_{ij}^{-1} \in \mathbb{R}^{6 \times 6}}}$ representing measurement confidence.

### 2. Residual Vector Formulation on $\mathrm{SE}(3)$

The expected relative transformation between pose $T_i$ and pose $T_j$ according to the current state estimate is:

$${\hat{T}_{ij} = T_i^{-1} T_j}$$

The error matrix ${E_{ij} \in \mathrm{SE}(3)}$ measures the relative deviation between the actual measurement ${\tilde{T}_{ij}}$ and the predicted state transformation ${T_i^{-1} T_j}$:

$${E_{ij} = \tilde{T}_{ij}^{-1} \left( T_i^{-1} T_j \right)}$$

#### Mapping Error to Tangent Space ${\mathfrak{se}(3)}$

Because optimization requires a 6-dimensional Euclidean vector space, the matrix error ${E_{ij}}$ is mapped to its local tangent space (Lie algebra ${\mathfrak{se}(3)}$) via the logarithmic map ${\log: \mathrm{SE}(3) \to \mathfrak{se}(3)}$, and flattened into a vector ${\mathbb{R}^6}$ using the **${\vee}$ operator** ${(\cdot)^\vee}$ - together, ${\mathrm{Log}(\cdot) = (\log(\cdot))^\vee: \mathrm{SE}(3) \to \mathbb{R}^6}$, mirroring the lowercase/uppercase convention already used for ${\exp}$ / ${\mathrm{Exp}}$ below:

$${r_{ij}(X) = \left( \log \left( \tilde{T}_{ij}^{-1} T_i^{-1} T_j \right) \right)^\vee \in \mathbb{R}^6}$$

The residual vector:

$${r_{ij} = \left[ \boldsymbol{\rho}_{ij}^\top \theta_{ij}^\top \right]^\top}$$

captures 3D translational error:

$${\boldsymbol{\rho}_{ij}}$$

and rotational error:

$${\theta_{ij}}$$

### 3. Objective Function

The global optimization minimizes the sum of squared Mahalanobis distances over all edges ${\mathcal{E}}$ in the graph:

$${F(X) = \sum_{(i,j) \in \mathcal{E}} r_{ij}(X)^\top \Omega_{ij} \, r_{ij}(X)}$$

### 4. Manifold Optimization and Linearization

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

and ${J_j = \frac{\partial r_{ij}}{\partial \boldsymbol{\xi}_j}}$ are derived using the **Right Inverse Baker-Campbell-Hausdorff (BCH) approximation**:

$${J_j = J_r^{-1}(r_{ij})}$$

$${J_i = - J_r^{-1}(r_{ij}) \, \mathrm{Ad}\left( T_j^{-1} T_i \right)}$$

Here, ${\mathrm{Ad}(T) \in \mathbb{R}^{6 \times 6}}$ is the **Adjoint transformation matrix** of ${\mathrm{SE}(3)}$, which transforms velocity/tangent vectors between frame coordinate systems:

$${\mathrm{Ad}\left(\begin{bmatrix} R & p \\ 
\mathbf{0}^\top & 1 \end{bmatrix}\right) = \begin{bmatrix} R & p^\wedge R \\ 
\mathbf{0} & R \end{bmatrix}}$$

and ${J_r^{-1}(\cdot)}$ is the inverse right Jacobian of ${\mathrm{SE}(3)}$.

### 5. Solving the Linear System (Gauss-Newton Step)

Stacking all residuals into a global residual vector $R(X)$ and Jacobians into a sparse Jacobian matrix $J$, the linearization takes the standard form:

$${H \, \boldsymbol{\delta}^* = -b}$$

* **Hessian Matrix:** ${H = J^\top \Omega J = \sum_{(i,j) \in \mathcal{E}} J_{ij}^\top \Omega_{ij} J_{ij} \in \mathbb{R}^{6N \times 6N}}$
* **Gradient Vector:** ${b = J^\top \Omega R(X) \in \mathbb{R}^{6N}}$
* **Update Vector:** ${\boldsymbol{\delta}^* = \left[ \boldsymbol{\xi}_1^\top \;\; \boldsymbol{\xi}_2^\top \;\; \dots \;\; \boldsymbol{\xi}_N^\top \right]^\top}$

Because edges only connect adjacent or loop-closing keyframes, $H$ is extremely **sparse** and block-structured. It is typically solved using Sparse Cholesky Factorization (${\mathrm{LL}^\top}$ or ${\mathrm{LDL}^\top}$) or Conjugate Gradients in solvers like GTSAM or g2o.

In practice a pure Gauss-Newton step can overshoot or diverge far from the solution, so a **Levenberg-Marquardt** damping term $\lambda$ is added to the Hessian's diagonal before solving:

$${(H + \lambda I) \, \boldsymbol{\delta}^* = -b}$$

Larger $\lambda$ shrinks the step toward gradient descent (safer, slower) while ${\lambda \to 0}$ recovers pure Gauss-Newton (faster near convergence). Both `pose_graph.py` implementations (`use_numpy/` and `use_manif/`) use this damped form with a fixed `damping` coefficient (`H += np.eye(dof) * damping`) rather than pure, undamped Gauss-Newton.

### 6. Retraction / State Update

Once the increment vector ${\boldsymbol{\delta}^*}$ is computed, the system updates the trajectory states on the ${\mathrm{SE}(3)}$ manifold:

$${T_i^{(k+1)} = T_i^{(k)} \cdot \mathrm{Exp}\left(\boldsymbol{\xi}_i^*\right), \quad \forall i \in \{1, \dots, N\}}$$

This iteration repeats until convergence (${\Vert{}\boldsymbol{\delta}^*\Vert{} < \epsilon}$ or ${\Vert{}\Delta F\Vert{} < \epsilon}$).

---

## 16. Robust loss functions used to handle false loop closures

In Pose-Graph Optimization (PGO), standard non-linear least squares relies on an $L_2$ squared-error norm ${F(x) = \sum r_{ij}^\top \Omega_{ij} r_{ij}}$. Under an $L_2$ loss, a single false loop closure (a severe outlier) produces a massive residual $r_{ij}$ whose squared weight pulls the entire trajectory out of shape to satisfy the invalid edge.

Robust cost functions replace or reweight the standard $L_2$ norm to cap or reduce the influence of large residuals.

### The M-Estimator Framework (Iteratively Reweighted Least Squares)

Instead of minimizing $\frac{1}{2} e^2$ (where $e = \sqrt{r^\top \Omega r}$ is the normalized residual scalar), M-estimators minimize a robust kernel $\rho(e)$:

$$\min_{X} \sum_{(i,j) \in \mathcal{E}} \rho\left( \sqrt{r_{ij}(X)^\top \Omega_{ij} \, r_{ij}(X)} \right)$$

To integrate this into standard Gauss-Newton or Levenberg-Marquardt solvers without modifying the core linear algebra solver, robust kernels use **Iteratively Reweighted Least Squares (IRLS)**. The robust cost is converted into a modified information matrix $\Omega_{ij}^\text{robust} = w(e) \cdot \Omega_{ij}$, where the weight function $w(e)$ is:

$$w(e) = \frac{1}{e} \frac{\partial \rho(e)}{\partial e}$$

### 1. Classical M-Estimators

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

### 2. Dynamic Covariance Scaling (DCS)

Dynamic Covariance Scaling (Agarwal et al., 2013) is specifically designed for pose-graph optimization. Instead of reweighting during every residual evaluation, DCS dynamically scales the information matrix based on an analytical closed-form solution derived from Switchable Constraints.

DCS adds a dynamic scaling parameter $s_{ij} \in (0, 1]$ directly to the information matrix $\Omega_{ij}$:

$$\Omega_{ij}^\text{DCS} = s_{ij}^2 \, \Omega_{ij}$$

The scaling factor $s_{ij}$ is calculated in closed form at each iteration using the current error $e_{ij}^2 = r_{ij}^\top \Omega_{ij} r_{ij}$ and an upper-bound parameter $\Phi$:

$$s_{ij} = \min\left(1, \; \frac{2 \Phi}{\Phi + e_{ij}^2}\right)$$

```
                       DCS Scaling Factor (s_ij)
          1.0 |────────────┐
              |              \
              |               \___   (s_ij decreases once e_ij^2 > Phi)
          0.0 +--------------------> e_ij^2 (Error)
              0            Phi

```

#### How DCS Handles Outliers:

1. **Inliers ($e_{ij}^2 \le \Phi$):** $s_{ij} = 1$. The edge retains full confidence and behaves as a standard quadratic term.
2. **Outliers ($e_{ij}^2 > \Phi$):** $s_{ij} = \frac{2 \Phi}{\Phi + e_{ij}^2} < 1$. As error $e_{ij}^2$ grows, $s_{ij}^2 \propto \frac{1}{e^4}$, causing the effective weight of the edge to drop rapidly to zero.
3. **No Extra State Variables:** Unlike original Switchable Constraints, DCS does not add auxiliary optimization variables to the Hessian matrix $H$, preserving graph sparsity without increasing matrix inversion costs.

### Comparison of Robust Loss Functions

| Loss Function | Residual Cost Tail $\rho(e)$ | Weight Degeneration $w(e)$ | SLAM False Loop Rejection Power |
| ------------- | ------------- | ------------- | ------------- |
| **Standard $L_2$** | Unbounded Quadratic ($e^2$) | Constant ($1.0$) | **None** (1 outlier ruins the map) |
| **Huber** | Unbounded Linear ($\delta e$) | $\propto \frac{1}{e}$ | **Low/Moderate** (Dampens, but still pulls graph) |
| **Cauchy** | Logarithmic ($\ln e^2$) | $\propto \frac{1}{e^2}$ | **High** |
| **DCS** | Redescending ($\to 0$) | $\propto \frac{1}{e^4}$ | **Very High** (Effectively turns off bad edges) |
| **Geman-McClure** | Saturation / Bounded | $\propto \frac{1}{(1 + e^2)^2}$ | **Very High** |

Note the distinction in the first column: Geman-McClure's cost $\rho(e)=e^2/(1+e^2)$ genuinely **saturates**, monotonically approaching a constant ($1$) as $e\to\infty$. DCS's effective cost $s_{ij}^2 e_{ij}^2 = \frac{4\Phi^2 e^2}{(\Phi+e^2)^2}$ is stronger than that - for $e^2>\Phi$ it is *decreasing* in $e$ and decays all the way back to $0$ as $e\to\infty$ (differentiate w.r.t. $x=e^2$: $\frac{d}{dx}\frac{4\Phi^2 x}{(\Phi+x)^2} = \frac{4\Phi^2(\Phi-x)}{(\Phi+x)^3} < 0$ for $x>\Phi$). This **redescending** behavior is why DCS suppresses extreme outliers even more aggressively than Geman-McClure, and also why it's every bit as non-convex - the "Graduated Non-Convexity" caveat below applies to it for exactly this reason.

### Practical Considerations in Implementation

1. **Threshold Tuning ($\delta, k, \Phi$):** The parameters set the boundary between inliers and outliers. In $\mathrm{SE}(3)$ PGO, error $e^2$ follows a Chi-Square distribution ($\chi^2$) with 6 degrees of freedom. Setting $\Phi$ or $k^2$ corresponding to the 95% or 99% quantile of $\chi^2(6)$ (e.g., $\Phi \approx 12.59$) provides a sound baseline.
2. **Graduated Non-Convexity (GNC):** Highly non-convex robust functions (like DCS or Geman-McClure) can introduce local minima if applied from a poor initial guess. Modern solvers use GNC to start with a convex $L_2$ loss and gradually harden the robust kernel as iterations progress.

This section is theory only: neither `use_numpy/pose_graph.py` nor `use_manif/pose_graph.py` implements Huber, Cauchy, or DCS reweighting - both scripts still use a single, unweighted `info_matrix = np.eye(6)` shared by every edge, odometry and loop-closure alike (the same gap already noted for per-edge $\Omega_{ij}$ weighting earlier in this doc). Robust loss reweighting is a natural extension a reader could add, not something the accompanying scripts exercise.
