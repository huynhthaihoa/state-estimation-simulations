# Marginalization and sliding-window smoothing

Marginalizing a variable permanently removes it from the optimization and keeps everything it taught us about its neighbors, packaged as one new **prior factor** (see [Prior](../glossary.md#prior-unary-factor); unary only in the pure-chain case of §8, in general it spans all of $x_a$'s neighbors). A real-time estimator can then bound its problem size without discarding what it used to know.

This doc builds on two earlier ones:
- **Variable elimination** from [`bayes_tree.md`](bayes_tree.md) §3 ("eliminate $x_1$ → its info gets summarized into a new constraint on $x_2$"). We reuse that mechanic for a different purpose.
- **Filtering vs. smoothing** from [filtering_smoothing.md §9](../filtering_smoothing.md#9-the-precise-distinction) (filtering marginalizes old information, smoothing keeps it). This doc is the mechanical middle ground, [fixed-lag smoothing](../glossary.md#sliding-window-and-fixed-lag-smoothing), that the diagram in [§10](../filtering_smoothing.md#10-where-the-modern-systems-fit) only names.

---

## 1. The problem sliding-window smoothing needs to solve

Full batch optimization (plain [bundle_adjustment.md](bundle_adjustment.md)/[pose_graph_optimization.md](pose_graph_optimization.md)) keeps every pose ever seen in the optimization, so the problem grows without bound as the robot keeps moving. [isam2_optimization.md](isam2_optimization.md) fixes the *recompute* cost (only touch the part of the Bayes tree a new factor actually affects) but not the *memory* cost - every variable is still in the graph, just efficiently re-solved.

A real-time visual-inertial odometry (VIO) or visual-inertial SLAM (VI-SLAM) estimator, which fuses camera and [IMU](../glossary.md#imu-inertial-measurement-unit) data (MSCKF = Multi-State Constraint Kalman Filter, VINS-Mono, OKVIS = Open Keyframe-based Visual-Inertial SLAM), can afford neither. It has fixed onboard memory and fixed per-frame compute, and it runs forever. It must **forget** old states: drop the *variable* but keep the *information* it contributed. That is marginalization.

---

## 2. Marginalization: keep the information, drop the variable

The elimination picture from `bayes_tree.md §3`:

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

That is marginalization. `bayes_tree.md` uses it to build a **solve order**, where nothing is thrown away for good. We use the same step to drop $x_1$ **permanently**, on purpose: it is the oldest pose in the window and the estimator will never touch it again.

### Why not just delete $`x_1`$?
Two naive shortcuts go wrong, in opposite directions. Marginalization is the third option:
- **Delete $x_1$ and its factors:** everything those factors told us about $x_2$ is lost, so we become less sure than we should be.
- **Cut $x_1$'s row and column out of the [information matrix](../glossary.md#information-matrix) $\Lambda$ (ignore the [Schur complement](../glossary.md#schur-complement) term):** for the 2×2 example of §4, $x_2$ keeps information $2$, as if $x_1$ were known exactly. The true value is $`\Lambda' = 1.5`$ ($`\Lambda'`$ = information after marginalizing, §4), so the variance is $0.5$ instead of $0.667$: overconfident.
- **Marginalize:** keeps exactly what $x_1$ taught us about $x_2$, as a prior on $x_2$.

---

## 3. Three flavors of elimination, compared

| | Structural (BA landmarks) | Solve-order (Bayes tree/iSAM2) | Temporal (this doc) |
|---|---|---|---|
| What's eliminated | 3D landmarks | any variable, in a chosen order | the oldest pose/state in the window |
| Is it recoverable? | **Yes** - back-substitution recovers it | **Yes** - it stays in the graph, and is re-eliminated whenever a later update affects it | **No** - gone for good |
| Why eliminate it | Landmarks outnumber cameras; cheap to invert | Reuse most of the last solve; minimize fill-in ([glossary](../glossary.md#fill-in)) | Bound memory/compute to a fixed window size |
| Where it's covered | [bundle_adjustment.md §12](bundle_adjustment.md#12-block-sparsity-and-the-schur-complement) | [bayes_tree.md](bayes_tree.md), [isam2_optimization.md](isam2_optimization.md) | §4-§7 below |

The math (§4) is the same Schur-complement elimination in all three rows; the rows differ only in the fate of the eliminated variable.

---

## 4. The math: turning an eliminated pose into a prior

**A two-variable example first.** Take $`\Lambda = \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}`$ and drop the first variable:
- Naively keeping only the second diagonal entry says the surviving variable has information 2.
- The Schur correction subtracts what $x_1$ borrowed from it: $`\Lambda' = 2 - (-1)(1/2)(-1) = 1.5`$.
- Check: $`\Sigma = \Lambda^{-1} = \begin{bmatrix} 2/3 & 1/3 \\ 1/3 & 2/3 \end{bmatrix}`$, so the marginal variance of the second variable is $`2/3 = 1/1.5`$. The correction is exactly what marginalizing means.
- $\Lambda'$ here is $\Lambda_b'$ of the general formula below, for the scalar case.

In general, suppose the current window has poses $x_a$ (the oldest, about to be dropped) and $x_b$ (everything still connected to it - [odometry](../glossary.md#odometry) neighbors, and any landmark/IMU-bias variables it shares factors with). After linearization, the joint Gaussian is described by an information matrix $\Lambda$ and information vector $\eta$, partitioned to match. $\Lambda$ is the window's Gauss-Newton Hessian $H = J^\top \Omega J$ (the information matrix of the linearized problem; §8.1 calls it $H$), where $J$ is the stacked Jacobian of all the window's residuals and $\Omega$ holds the factors' information matrices (see [normal equations](../glossary.md#normal-equations-and-hessian)). It is not the damping $\lambda$ of §8.1. $\eta$ is the linear (right-hand-side) term of the window's linearized system, $g$ in $H\delta = g$ (§8.1); together with $\Lambda$ it describes the Gaussian:

```math
\Lambda = \begin{bmatrix} \Lambda_{aa} & \Lambda_{ab} \\ \Lambda_{ba} & \Lambda_{bb} \end{bmatrix}, \qquad \eta = \begin{bmatrix} \eta_a \\ 
\eta_b \end{bmatrix}
```

Marginalizing out $x_a$ means integrating it out of the joint distribution. The closed form is the Schur complement of `bundle_adjustment.md §12`, in information form:

```math
\Lambda_b' = \Lambda_{bb} - \Lambda_{ba}\Lambda_{aa}^{-1}\Lambda_{ab}, \qquad \eta_b' = \eta_b - \Lambda_{ba}\Lambda_{aa}^{-1}\eta_a
```

For the prior, $\Lambda_{bb}$ and $\eta_b$ must contain only the factors that touch $x_a$. Factors among the $x_b$ variables alone stay in the graph unchanged, so folding their contribution to $\Lambda_{bb}$ into the prior would count it twice. §8.1 shows how the script handles this.

$(\Lambda_b', \eta_b')$ is a brand-new **prior factor** over exactly the variables $x_a$ used to connect to. It enters the graph like any other factor, as in the §2 diagram with $x_a = x_1$: $x_a$ disappears and the summarized information sits on its former neighbors.

The difference from BA's use of the same Schur complement (on the point block):
- BA computes $\Lambda_b'$ *only to solve the reduced system faster*, then back-substitutes to recover $x_a$ from $x_b$.
- Here there is no back-substitution: $x_a$ is never coming back. That is what makes this **temporal marginalization** rather than a solver optimization.

---

## 5. The fill-in consequence

$\Lambda_{ba}\Lambda_{aa}^{-1}\Lambda_{ab}$ is a dense block whenever $x_a$ touched more than one neighbor. Suppose $x_a$ was connected to three poses that were previously unconnected to each other. Marginalizing $x_a$ then connects all three of them mutually in the new prior:

```text
Before:                  After eliminating x_a:

     x_a                       x1 ─── x2
    / | \                       \    /
   x1 x2 x3                      \  /
                                  x3

(x1,x2,x3 only connect        (x1,x2,x3 now all
 through x_a)                  directly connected)
```

This is the **fill-in** that [elimination_tree.md §9](elimination_tree.md#9-why-ordering-matters-so-much) shows the elimination order can control, and that iSAM2's reordering ([isam2_optimization.md §12](isam2_optimization.md#12-variable-ordering-is-also-crucial)) keeps small. Sliding-window marginalization has no such choice: we always eliminate the oldest pose, whatever it is connected to. The system accepts that cost and keeps the window small: a window of $`W`$ poses costs $O(W^3)$ per step, and a bigger window lets the dense prior span more states.

---

## 6. The consistency gotcha: why FEJ exists

$(\Lambda_b', \eta_b')$ is computed by linearizing at the *current* estimates of $x_a$ and $x_b$ at the moment of marginalization. That linearization point is then frozen into the prior. The surviving variables in $x_b$ are relinearized at new estimates on every later iteration, so they move away from the point the prior was built at. (This repo's prior differs: it relinearizes at every solve; see §8.)

**Intuition (toy 2D range example):** a range factor between pose $`p`$ and landmark $`l`$ only measures their distance.
- Move both by the same shift and the range does not change. So that shift is [unobservable](../glossary.md#observability).
- Consistent linearization: the Jacobian is the unit vector $`u`$ from $`p`$ to $`l`$ on the $`l`$ side and $`-u`$ on the $`p`$ side. For a common shift they cancel, so the information along it is 0.
- Mixed points: $`p = (0,0)`$, $`l = (3,4)`$ give $`u = (0.6, 0.8)`$. Now linearize the pose side at $`p' = (0,1)`$, which gives $`u' = (0.707, 0.707)`$.
- A common shift of $`(1,0)`$ then predicts a range change of $`0.6 - 0.707 = -0.107`$ m. The true change is 0.
- So the system "sees" information where there is none, and becomes overconfident (a [consistency](../glossary.md#consistency-and-nees) failure).

In general, this mismatch injects spurious information into directions of the state that should be unobservable. [kf_ekf_iekf.md §2](../filtering/kf_ekf_iekf.md#2-extended-kalman-filter-the-world-is-nonlinear-so-ill-approximate-it-locally) describes the same class of problem for plain EKF-SLAM.

The standard fix is **[First-Estimate Jacobians (FEJ)](../glossary.md#fej-first-estimate-jacobians)**: once a variable has contributed to a marginalization, all *future* Jacobians involving it are evaluated at that same first-linearization point, not the newest estimate. The linearized system then keeps the correct unobservable directions, which improves consistency. Who uses it:
- The original MSCKF (Mourikis & Roumeliotis, 2007) predates FEJ. It was added in the later MSCKF 2.0 (Li & Mourikis, 2012/2013).
- VINS-Mono does not apply FEJ systematically: its marginalization prior keeps the linearization point it was built at (FEJ-like for that factor), while the other factors are relinearized. Its §VI-D notes that marginalization fixes linearization points early, which may give suboptimal estimates, and argues that since small drift is acceptable for VIO, the negative impact is not critical.

See Huang, Mourikis & Roumeliotis (2009) in the references below for the FEJ derivation itself - we don't reproduce it here.

---

## 7. Sliding-window and fixed-lag smoothing: the payoff

`filtering_smoothing.md §10`'s diagram names this strategy; here is one step of it, repeated for every new keyframe/measurement (the order matches `run_sliding_window_pose_graph`, §8.1):

```mermaid
flowchart TD
    A["New keyframe or measurement arrives"] --> B{"Window full?<br/>(window_size poses)"}
    B -- "yes" --> C["Marginalize the oldest state<br/>(§4: produces a new prior,<br/>§5: densifies its neighbors)"]
    B -- "no" --> D
    C --> D["Append the new variables<br/>and factors to the window"]
    D --> E["Solve the window"]
    E --> F["Next arrival: step k+1<br/>(repeat from the top)"]
```

The window never grows past a fixed size, so cost and memory stay **bounded by the window size**, independent of run time. For $W$ poses, a dense solve takes $O(W^3)$ time and $O(W^2)$ memory.
- Full batch smoothing is unbounded.
- iSAM2 is bounded by how much of the Bayes tree a new factor *changes*, not by a fixed window. A loop closure can still touch a large chunk of history (`isam2_optimization.md §9`).

This is what `filtering_smoothing.md §10`'s "fixed-lag smoothing" box and its MSCKF/VINS-Mono bullets refer to:
- MSCKF keeps a sliding window of camera poses. When a landmark's track ends, it [triangulates](../glossary.md#triangulation-and-pnp) the landmark, uses its observations to update the window poses, then discards it: the landmark never enters the state (see [kf_ekf_iekf.md](../filtering/kf_ekf_iekf.md)'s MSCKF paragraph).
- VINS-Mono keeps a sliding window of [keyframes](../glossary.md#keyframe) and marginalizes the oldest frame when the second-newest is a keyframe (otherwise it drops the second-newest frame's visual measurements) with the Schur-complement step in §4, with no systematic FEJ (§6).

![Three panels from sliding_window_marginalization.py on a 16-pose square with a 10-pose window: window membership after each new pose, the window's block-tridiagonal information matrix with the oldest pose's block about to be eliminated, and the next window's matrix where that information lands as one prior block on the new oldest pose](../../assets/sliding_window_marginalization_concept.png)

*Figure: windows and information matrices recorded from `use_numpy/sliding_window_marginalization.py`'s own `run_sliding_window_pose_graph` calls (16 poses, window 10, seed 0), plotted by `uv run python generator/make_figures.py sliding_window_marginalization_concept`.*

---

## 8. What this repo implements

[`sliding_window_marginalization.py`](../../use_numpy/sliding_window_marginalization.py) streams a chain of odometry edges one node at a time, keeps at most `window_size` poses live in memory, and marginalizes the oldest one out via §4's Schur complement whenever a new node would exceed that.
- **The marginal is a genuine prior factor** - a frozen reference pose plus an information matrix, relinearized against the *current* estimate every solve, exactly like an ordinary edge - rather than a frozen linear term. This is one valid representation (an alternative to storing a fixed linear factor), and it keeps the prior consistent as the surviving poses move across later windows.
- **It keeps §4's $\Lambda_b'$ as the prior's information matrix, but does not store $\eta_b'$.** Instead, the prior is anchored at a frozen reference pose, so its linear term is zero at the moment it is created. That is exact only when $\eta_b' = 0$, and it is here: see §8.1.

Two deliberate scope choices, both flagged in the script:
- **Pure odometry chain, no loop closures.** The oldest pose in a chain window is connected to exactly one surviving neighbor, so marginalizing it produces a *provably unary* prior (touching a single pose). A test verifies this by checking that the Schur-complement correction term is zero (to 1e-8) everywhere outside that one block. §5's fill-in appears once a marginalized node has *multiple* neighbors (a landmark, an IMU-bias variable, a loop closure). That is a different problem, covered by [`bayes_tree.md`](bayes_tree.md)/[`isam2_optimization.md`](isam2_optimization.md) and by `pose_graph_incremental.py`'s loop-closure handling. This script isolates the memory-*bounding* property alone.
- **No First-Estimate Jacobians (§6).** Every pose's Jacobian, including the prior factor's own (first bullet above), is re-evaluated at its newest estimate on every solve. That is the source of the overconfidence FEJ prevents (§6); the script does not measure its size. Like `pose_graph_incremental.py` with iSAM2, the script flags what it omits.

### 8.1 The math, concretely

Poses are written $x$ in §1-§7 and $X_k$ from here on: $4 \times 4$ [SE(3)](../glossary.md#so3-and-se3) matrices. Tangent vectors (6-number small motions of a pose, see [Lie algebra and tangent space](../glossary.md#lie-algebra-and-tangent-space)) follow the repo's `[vx, vy, vz, wx, wy, wz]` order. We solve every window by plain Gauss-Newton ([gauss_newton.md](gauss_newton.md)) over two factor types, both linearized at the current estimate on every iteration (§8).

**Odometry edge** (`linearize_edge`, reused from `pose_graph_incremental.py`). The residual $e_{ij}$ is a 6-vector: the mismatch between the predicted pose $X_i Z_{ij}$ and $X_j$, written as a tangent vector. $J_i$ and $J_j$ are its $6 \times 6$ derivatives with respect to a right perturbation of $X_i$ or $X_j$ ([pose_graph_optimization.md §15.4](pose_graph_optimization.md#154-manifold-optimization-and-linearization)). Here $J_r^{-1}$ is the SE(3) inverse [right Jacobian](../glossary.md#right-and-left-jacobians) (`compute_se3_inv_right_jacobian`), $\mathrm{Ad}$ is the [adjoint](../glossary.md#adjoint), $\mathrm{Log}$ is the [Log map](../glossary.md#exp-and-log-maps), and $`Z_{ij}`$ is the measured relative pose from $`X_i`$ to $`X_j`$ ($`Z_{ij} = \tilde T_{ij}`$ in [pose_graph_optimization.md §15](pose_graph_optimization.md#15-mathematical-breakdown-of-the-error-formulation-and-lie-algebra-operations)):

```math
e_{ij} = \mathrm{Log}\big((X_i Z_{ij})^{-1} X_j\big), \qquad
J_j = J_r^{-1}(e_{ij}), \qquad
J_i = -J_r^{-1}(-e_{ij})\,\mathrm{Ad}(Z_{ij}^{-1})
```

**Prior on the oldest window pose** (`assemble_window_system`). $X_{\text{ref}}$ is a frozen reference pose and $\Omega_p$ is the prior's information matrix. The prior factor penalizes pose 0 for differing from $X_{\text{ref}}$: $e_p$ measures how far $X_0$ has moved from $X_{\text{ref}}$, and $J_p$ is the derivative of $e_p$ with respect to the perturbation of $X_0$:

```math
e_p = \mathrm{Log}(X_{\text{ref}}^{-1} X_0), \qquad J_p = J_r^{-1}(e_p)
```

The first prior is the [gauge](../glossary.md#gauge-freedom) anchor, with $X_{\text{ref}}$ equal to the ground-truth (simulator's true) first pose and $\Omega_p = 10^6 I$ (`--anchor-weight`). Every later prior comes from marginalization (below).

**Gauss-Newton step** (`assemble_window_system`, `solve_to_convergence`). Each edge uses the information matrix $\Omega = I_6$. The script accumulates

```math
H = \sum J^\top \Omega\, J, \qquad g = -\sum J^\top \Omega\, e, \qquad H\delta = g, \qquad X_k \leftarrow X_k \mathrm{Exp}(\delta_k)
```

Here each $J$ is read as the blocks above ($J_i$, $J_j$ per edge, $J_p$ for the prior), and the prior enters as one more factor with $\Omega_p$ in place of $\Omega$. In the glossary's convention $g = -b$ (with $`b = J^\top \Omega\, e`$), and $\delta$ is the step called $\delta^*$ in [pose_graph_optimization.md §15.5](pose_graph_optimization.md#155-solving-the-linear-system-gauss-newton-step).

It stops when $\lVert\delta\rVert$ < `gn_tol` ($10^{-6}$) or after `gn_max_iters` (10) iterations. With $W$ poses in the window, $H$ is $6W \times 6W$.

**Marginalizing the oldest pose** (`marginalize_oldest`). The window has already converged. Let $a$ be the oldest pose, `x_window[0]`, and $b$ the next one, `x_window[1]`. In a pure chain, $a$ touches exactly two factors: its prior and the single a–b edge. The script asserts that there is exactly one such edge. The script takes $\Lambda_{aa}$, $\Lambda_{ab}$ and $\Lambda_{ba}$ from the full window $H$ ($\Lambda = H$, §4), since those blocks only involve factors touching $a$. It builds $\Lambda_{bb}^{(ab)}$ from a separate system containing only the a–b edge:

```math
\Omega_p' = \Lambda_{bb}^{(ab)} - \Lambda_{ba}\Lambda_{aa}^{-1}\Lambda_{ab}, \qquad X_{\text{ref}}' = X_b \text{ (current estimate, frozen)}
```

Taking $\Lambda_{bb}$ from the full $H$ instead would also include the b–c edge. That edge stays in the window, so it would be counted twice. $\Omega_p'$ equals §4's Schur complement of the system built from $a$'s prior plus the a–b edge.

**Why storing no $\eta_b'$ is exact here.** Because $X_{\text{ref}}' = X_b$, the new prior's residual is zero when it is created, so its linear term is zero too. §4's $\eta_b'$ really is zero in this setting. The chain has a single anchor and no loop closures, so every factor can be satisfied exactly. Each new pose also starts at $X_{\text{last}} Z_{ij}$ ($`X_{\text{last}}`$ = the newest pose in the window before it is added), which already zeroes its edge residual. So at the optimum, every residual, and therefore every gradient term, is zero. This depends on the pure-chain scope. With a loop closure or a shared landmark, residuals would not vanish, and dropping $\eta_b'$ would lose information.

**Ordering** (`run_sliding_window_pose_graph`). For each new node, the script:

1. Marginalizes the oldest pose first, if the window already holds `window_size` poses, using the linearization point from the previous solve.
2. Appends the new pose, initialized as $X_{\text{last}} Z_{ij}$, together with its edge.
3. Solves the window to convergence.

The solve therefore never exceeds `window_size` poses, so `max_dof` (the size, in unknowns, of the largest system any solve assembles; §9) is capped at 6 × `window_size`. A marginalized pose keeps the estimate it had when it was dropped.

**Baseline** (`run_full_batch_growing`). For $k = 1 \dots n$, the baseline re-solves the whole graph of the first $k$ poses from scratch. It uses `pose_graph.run_pose_graph_optimization` (adaptive Levenberg-Marquardt ([levenberg_marquardt.md](levenberg_marquardt.md)) starting at λ = 0, via `damping=0.0`; the damping λ is the extra term LM adds to $H$, see [glossary](../glossary.md#levenberg-marquardt-lm-and-damping)), starting from [dead reckoning](../glossary.md#dead-reckoning) from the first ground-truth pose, with node 0 anchored by $10^6 I$ inside that function. Its `max_dof` is $6n$.

#### Defaults
`--window-size 10`; `--nodes-per-side-sweep 2 4 8 16 32 64` (8 to 256 poses); `--side-length 2.0` m; `--pos-noise-std 0.05` m; `--rot-noise-std 0.01` rad; `--anchor-weight 1e6`; `--gn-tol 1e-6`; `--gn-max-iters 10`; `--seed 0`.

---

## 9. Empirical verification: bounded vs. unbounded

`sliding_window_marginalization.py` compares this bounded approach against `run_full_batch_growing`, the unbounded baseline that re-solves the entire graph from scratch at every new node (the strategy §1 opens with). Output bookkeeping (final pose estimates for every node, kept only for this script's own error reporting) is unavoidably $O(n)$ for *both* approaches. The metric that isolates the algorithmic claim is therefore the size of the largest dense information matrix that either one assembles and solves. The script reports it as `max_dof`, with an approximate byte count for holding that matrix densely ($`\text{max\_dof}^2 \times 8`$ bytes, float64):

| Trajectory length | Full-batch `max_dof` | Full-batch peak (~bytes) | Sliding-window `max_dof` (window_size=10) | Sliding-window peak (~bytes) |
| --- | --- | --- | --- | --- |
| 8 | 48 | 18,432 | 48 | 18,432 |
| 16 | 96 | 73,728 | 60 | 28,800 |
| 32 | 192 | 294,912 | 60 | 28,800 |
| 64 | 384 | 1,179,648 | 60 | 28,800 |
| 128 | 768 | 4,718,592 | 60 | 28,800 |
| 256 | 1536 | 18,874,368 | 60 | 28,800 |

Two measurements, only one of them deterministic:
- **Memory (`max_dof`):** full-batch's system size grows linearly with trajectory length (so its dense-matrix memory grows *quadratically* - visible directly in the table, roughly $4\times$ per doubling of length) and never stops. Sliding-window's caps at exactly $`6 \times \text{window\_size}`$ the moment the window first fills, and never moves again, confirmed identically across multiple seeds (`max_dof` depends only on trajectory length and `window_size`, not on the noise realization). `max_dof` (and the byte counts derived from it) is the deterministic part.
- **Average per-step wall-clock time** tells the same story less starkly, and absolute values depend on the machine and its load (middle panel below). In the committed run the full batch grows from about 0.5 ms to about 86 ms per new pose, while the window stays at about 0.5-2 ms. The window's time flattens once trajectories exceed `window_size`, while the full batch's keeps growing; individual runs differ, so do not read the exact values.

![Three panels from sliding_window_marginalization.py: largest information-matrix size, time per new pose and position RMSE against trajectory length from 8 to 256 poses, for a full batch re-solve and a 10-pose sliding window](../../assets/sliding_window_marginalization.png)

*Figure: `use_numpy/sliding_window_marginalization.py` at its defaults (seed 0), plotted by `uv run python generator/make_figures.py sliding_window_marginalization`.*

The same streaming, animated for a 64-pose run with a 10-pose window. The window's system size stops at 60 unknowns while a full batch re-solve keeps growing as 6 unknowns per pose, reaching 384 at the last pose. The path drifts and does not close the square because there are only odometry edges, so both solvers return the dead-reckoned path.

![Animation of a 64-pose odometry-only square path streaming in: top, the live 10-pose window in blue, marginalized poses frozen in gray and the pose being marginalized circled in red; bottom left, the system size of a full batch re-solve growing with the number of poses against the sliding window capped at 60 unknowns; bottom right, the window's block-tridiagonal information matrix with the prior block on its oldest pose](../../assets/sliding_window_marginalization.gif)

*Animation: `generator/animate_sliding_window_marginalization.py` at its defaults (seed 0), rendered by `uv run python generator/animate_sliding_window_marginalization.py --out assets/sliding_window_marginalization.gif`. The batch curve is computed (6 unknowns per pose), not re-solved.*

**Accuracy is identical here, but that's guaranteed by the setup, not evidence that marginalization loses nothing.**
- Final position [RMSE](../glossary.md#rmse-root-mean-square-error) is the same for full-batch and sliding-window at every trajectory length.
- The reason is the one in §8.1 (all residuals are zero): the least-squares optimum *is* the dead-reckoned trajectory. Both solvers return exactly that. Checked at seed 0, both match dead reckoning with a maximum difference of 0.0 at 8, 16 and 32 poses.
- So any correct solver would tie. The accuracy comparison can't reveal whether marginalization throws information away, and the figure's accuracy panel is simply dead-reckoning error. The memory bound above is the real result.
- The information claim is checked separately: `tests/use_numpy/test_sliding_window_marginalization.py` (`test_marginalized_window_covariance_matches_joint_batch_marginal`) compares the window covariance after two marginalizations against the matching marginal of the joint batch system, and it passes. The accuracy panel cannot show this.

The same scope silences §6's FEJ subtlety here: the inconsistency only shows up once a later loop closure or shared landmark reconnects to something already marginalized. A version with loop closures would actually test accuracy, and would be expected to show §5's fill-in cost and a real (if likely small) accuracy gap from skipping FEJ. That extension is not implemented.

---

## 10. One-sentence summary

> **Marginalization is the variable-elimination step from `bayes_tree.md`, used to permanently discard an old state.** It turns that state into a dense prior over whatever it was still connected to. That lets sliding-window/fixed-lag smoothers (VINS-Mono) and window filters (MSCKF) run in bounded memory and time forever. The price is fill-in that no reordering can avoid (§5) and a consistency subtlety (FEJ, §6) that full batch and iSAM2 never face.

---

## 11. References

1. Sibley, G., Matthies, L., & Sukhatme, G. (2010). *Sliding Window Filter with Application to Planetary Landing*. Journal of Field Robotics, 27(5), 587-608. https://doi.org/10.1002/rob.20360 - the sliding-window/delayed-state-marginalization formulation behind §4 and §7.
2. Huang, G. P., Mourikis, A. I., & Roumeliotis, S. I. (2009). *A First-Estimates Jacobian EKF for Improving SLAM Consistency*. In Experimental Robotics: The Eleventh International Symposium (pp. 373-382). Springer. https://doi.org/10.1007/978-3-642-00196-3_43 - the FEJ fix behind §6.
3. Mourikis, A. I., & Roumeliotis, S. I. (2007). *A Multi-State Constraint Kalman Filter for Vision-Aided Inertial Navigation*. ICRA 2007, 3565-3572. https://doi.org/10.1109/ROBOT.2007.364024 - the original MSCKF reference in §7 (predates FEJ, per §6), already cited in [filtering_smoothing.md §12](../filtering_smoothing.md#12-references).
4. Qin, T., Li, P., & Shen, S. (2018). *VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator*. IEEE Transactions on Robotics, 34(4), 1004-1020. https://doi.org/10.1109/TRO.2018.2853729 - the VINS-Mono reference in §7, already cited in [filtering_smoothing.md §12](../filtering_smoothing.md#12-references). Its §VI-D notes that marginalization fixes linearization points early and argues the impact is not critical for VIO (not a systematic FEJ scheme).
5. Li, M., & Mourikis, A. I. (2013). *High-Precision, Consistent EKF-Based Visual-Inertial Odometry*. The International Journal of Robotics Research, 32(6), 690-711. https://doi.org/10.1177/0278364913481251 - the later MSCKF revision (sometimes called "MSCKF 2.0") that adds FEJ, referenced in §6.
