# Marginalization and sliding-window smoothing

Marginalizing a variable means permanently removing it from the optimization while keeping everything it taught you about its neighbors, packaged as one new prior factor - so a real-time estimator can bound its problem size without lying to itself about what it used to know.

This builds directly on two things you've already seen:
- **Variable elimination** from [`elimination_tree.md`](elimination_tree.md) ("eliminate $x_1$ → its info gets summarized into a new constraint on $x_2$") - this doc reuses that exact mechanic for a different purpose.
- **Sparsity and full vs. fixed-lag smoothing** from [filtering_smoothing.md §9-10](../filtering_smoothing.md#9-one-subtle-but-very-important-point) (filtering marginalizes old information, smoothing keeps it) - this doc is the missing mechanical middle ground the diagram there only names.

---

## 1. The problem sliding-window smoothing needs to solve

Full batch optimization (plain [bundle_adjustment.md](bundle_adjustment.md) / [pose_graph_optimization.md](pose_graph_optimization.md)) keeps every pose ever seen in the optimization forever - the problem grows without bound as the robot keeps moving. [isam2_optimization.md](isam2_optimization.md) fixes the *recompute* cost (only touch the part of the Bayes tree a new factor actually affects) but not the *memory* cost - every variable is still in the graph, just efficiently re-solved.

A real-time VIO/VI-SLAM front-end (MSCKF, VINS-Mono, OKVIS) often can't afford either: fixed onboard memory, fixed per-frame compute budget, running forever. The fix is to actively **forget** old states - but forgetting a pose's *variable* while keeping the *information* it contributed is exactly what marginalization does.

---

## 2. Marginalization: keep the information, drop the variable

Recall the elimination picture from `bayes_tree.md §3`:

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

That's marginalization. `bayes_tree.md` uses this step to build a *solve order* - $x_1$ is eliminated, but conceptually every variable gets eliminated eventually, and nothing is thrown away for good (§3 below makes this precise). Sliding-window smoothing uses the *identical* elimination math for the opposite reason: to throw $x_1$ away **permanently**, on purpose, because it's the oldest pose in the window and the estimator will never touch it again.

---

## 3. Three flavors of elimination, compared

| | Structural (BA landmarks) | Solve-order (Bayes tree / iSAM2) | Temporal (this doc) |
|---|---|---|---|
| What's eliminated | 3D landmarks | any variable, in a chosen order | the oldest pose/state in the window |
| Is it recoverable? | Yes - back-substitution recovers it | Yes - re-eliminated on the next update | **No** - gone for good |
| Why eliminate it | Landmarks outnumber cameras; cheap to invert | Reuse most of the last solve; minimize fill-in | Bound memory/compute to a fixed window size |
| Where it's covered | [bundle_adjustment.md §12](bundle_adjustment.md#12-block-sparsity-and-the-schur-complement) | [bayes_tree.md](bayes_tree.md), [isam2_optimization.md](isam2_optimization.md) | §4-§7 below |

The math (§4) is the same Schur-complement elimination in all three rows. What differs is only what happens to the eliminated variable afterward.

---

## 4. The math: turning an eliminated pose into a prior

Suppose the current window has poses $x_a$ (the oldest, about to be dropped) and $x_b$ (everything still connected to it - odometry neighbors, and any landmark/IMU-bias variables it shares factors with). After linearization, the joint Gaussian is described by an information matrix $\Lambda$ and information vector $\eta$, partitioned to match:

$$
\Lambda = \begin{bmatrix} \Lambda_{aa} & \Lambda_{ab} \\ \Lambda_{ba} & \Lambda_{bb} \end{bmatrix}, \qquad \eta = \begin{bmatrix} \eta_a \\ 
\eta_b \end{bmatrix}$$

Marginalizing out $x_a$ means integrating it out of the joint distribution, which has a closed form - the same Schur complement `bundle_adjustment.md §12` uses on the point block, just kept in information form here instead of being back-substituted afterward:

$$
\Lambda_b' = \Lambda_{bb} - \Lambda_{ba}\Lambda_{aa}^{-1}\Lambda_{ab}, \qquad \eta_b' = \eta_b - \Lambda_{ba}\Lambda_{aa}^{-1}\eta_a
$$

One precision point: here $\Lambda$ and $\eta$ are built only from the factors that touch $x_a$. Factors among the $x_b$ variables alone stay in the graph unchanged, so if their contribution to $\Lambda_{bb}$ were also folded into the prior, it would be counted twice. §8.1 shows how the script does this.

$(\Lambda_b', \eta_b')$ is a brand-new **prior factor** over exactly the variables $x_a$ used to connect to - nothing else. It gets inserted into the graph like any other factor:

```text
Before marginalization:

x_a ──odom── x_b ──odom── x_c
 │
 └──landmark/IMU-bias factors

After marginalizing x_a:

            x_b ──odom── x_c
             ↑
        new prior factor
     (Λ_b', η_b'), no x_a left
```

The critical difference from BA's version of this trick: BA computes $\Lambda_b'$ *only to solve the reduced system faster*, then back-substitutes to recover $x_a$'s optimal value from $x_b$'s. Here there is no back-substitution step - $x_a$ is never coming back. That's what makes this *temporal* marginalization rather than a solver optimization.

---

## 5. The fill-in consequence

$\Lambda_{ba}\Lambda_{aa}^{-1}\Lambda_{ab}$ is a dense (or denser) block whenever $x_a$ touched more than one neighbor. If $x_a$ was connected to, say, three other poses that were previously unconnected to each other, marginalizing $x_a$ makes all three of them mutually connected in the new prior:

```text
Before:                  After eliminating x_a:

     x_a                       x1 ─── x2
    /  |  \                     \    /
   x1  x2  x3                    \  /
                                   x3

(x1,x2,x3 only connect        (x1,x2,x3 now all
 through x_a)                  directly connected)
```

This is exactly the **fill-in** phenomenon `isam2_optimization.md §7` mentions in passing and `bayes_tree.md` builds its whole variable-*ordering* strategy around minimizing. The difference here: sliding-window marginalization doesn't get to choose an elimination order to minimize fill-in - it must always eliminate the oldest pose, whatever it happens to be connected to. This is a real, accepted cost of the sliding-window approach (not a bug to fix), and it's why sliding-window systems keep the window itself small - a bigger window means a more richly-connected pose to eventually eliminate, and a denser resulting prior.

---

## 6. The consistency gotcha: why FEJ exists

One subtlety worth flagging, not deriving in full: $(\Lambda_b', \eta_b')$ is computed by linearizing around whatever the *current* estimates of $x_a$ and $x_b$ happened to be at the moment of marginalization. That linearization point is then frozen into the prior forever. But the surviving variables in $x_b$ keep getting relinearized at new estimates on every later optimizer iteration - at a *different* point than the one the frozen prior was built from.

This mismatch injects spurious information into directions of the state that should be unobservable (the same class of problem [pose_graph_optimization.md](pose_graph_optimization.md) and `kf_ekf_iekf.md` describe for plain EKF-SLAM), making the estimator overconfident. The standard fix is **First-Estimate Jacobians (FEJ)**: once a variable has contributed to a marginalization, all *future* Jacobians involving it are evaluated at that same first-linearization point, not the newest estimate - trading a small amount of accuracy for provable consistency. This is narrower than "MSCKF and VINS-Mono both use this," though: the original MSCKF (Mourikis & Roumeliotis, 2007) predates FEJ - it was added in the later MSCKF 2.0 (Li & Mourikis, 2012/2013) - and VINS-Mono's own paper explicitly says its linearization-point handling is *not* FEJ, arguing the inconsistency FEJ targets isn't critical enough in their VIO setting to justify it. See Huang, Mourikis & Roumeliotis (2009) in the references below for the FEJ derivation itself - it isn't reproduced here.

---

## 7. Sliding-window and fixed-lag smoothing: the payoff

This is the strategy `filtering_smoothing.md §10`'s diagram names but doesn't mechanize - now it can be stated as a loop:

```text
new keyframe/measurement arrives
        ↓
add its variables + factors to the window
        ↓
   is the window full?
    ↙          ↘
  no            yes
   │             ↓
   │      marginalize the oldest state
   │      (§4: produces a new prior,
   │       §5: densifies its neighbors)
   ↓             ↓
        repeat
```

Because the window never grows past a fixed size, both the per-step optimization cost and memory stay **O(window size)**, independent of how long the robot has been running - the property full batch smoothing doesn't have (unbounded) and iSAM2 doesn't quite give you either (iSAM2 is bounded by how much of the Bayes tree a new factor *changes*, not by a fixed window - a loop closure can still touch a large chunk of history, per `isam2_optimization.md §9`).

This is precisely what `filtering_smoothing.md §10`'s "fixed-lag smoothing" box and its MSCKF/VINS-Mono bullets refer to - MSCKF keeps a sliding window of camera poses and marginalizes a landmark's constraint into them once triangulated (see [kf_ekf_iekf.md](../filtering/kf_ekf_iekf.md)'s MSCKF paragraph); VINS-Mono keeps a sliding window of keyframes and marginalizes the oldest one using exactly the Schur-complement step in §4 - though, per §6's correction, without FEJ.

---

## 8. What this repo implements

[`sliding_window_marginalization.py`](../../use_numpy/sliding_window_marginalization.py) implements exactly the follow-up this section used to say was missing: it streams a chain of odometry edges one node at a time, keeps at most `window_size` poses live in memory, and marginalizes the oldest one out via §4's Schur complement whenever a new node would exceed that. The marginal is represented as a genuine prior factor - a frozen reference pose plus an information matrix, re-linearized against the *current* estimate every solve, exactly like an ordinary edge - rather than a frozen linear term, which is the only representation that stays correct as the surviving poses keep moving across later windows. That representation keeps §4's $\Lambda_b'$ as the prior's information matrix, but it does not store $\eta_b'$. Instead, the prior is anchored at a frozen reference pose, so its linear term is zero at the moment it is created. That is exact only when $\eta_b' = 0$, and it is here: see §8.1.

Two deliberate scope choices, both flagged directly in the script:
- **Pure odometry chain, no loop closures.** The oldest pose in a chain window is connected to exactly one surviving neighbor, so marginalizing it produces a *provably unary* prior - verified directly by a test that checks the Schur-complement correction term is exactly zero everywhere outside that one block. §5's fill-in problem (a real cost once a marginalized node had *multiple* neighbors - a landmark, an IMU-bias variable, or a loop closure) is a genuinely different problem, already covered by [`bayes_tree.md`](bayes_tree.md)/[`isam2_optimization.md`](isam2_optimization.md) and by `pose_graph_incremental.py`'s own loop-closure handling; this script isolates the memory-*bounding* property alone.
- **No First-Estimate Jacobians (§6).** Every pose's Jacobian, including the prior factor's own, is re-evaluated at its newest estimate on every solve - the textbook source of the mild overconfidence FEJ exists to fix. Not implemented here, the same way `pose_graph_incremental.py` explicitly flags what it doesn't implement relative to iSAM2.

### 8.1 The math, concretely

Poses $X_k$ are $4 \times 4$ SE(3) matrices, and tangent vectors follow the repo's `[vx, vy, vz, wx, wy, wz]` order. Every window is solved by plain Gauss-Newton over two factor types. Both are linearized at the current estimate on every iteration, and there are no First-Estimate Jacobians.

**Odometry edge** (`linearize_edge`, reused from `pose_graph_incremental.py`). Here $\mathcal{J}_r^{-1}$ is the inverse right Jacobian of SE(3) (`compute_se3_inv_right_jacobian`) and $\mathrm{Ad}$ is the adjoint:

$$
e_{ij} = \mathrm{Log}\big((X_i Z_{ij})^{-1} X_j\big), \qquad
J_j = \mathcal{J}_r^{-1}(e_{ij}), \qquad
J_i = -\mathcal{J}_r^{-1}(-e_{ij})\,\mathrm{Ad}(Z_{ij}^{-1})
$$

**Prior on the oldest window pose** (`assemble_window_system`). $X_{\text{ref}}$ is a frozen reference pose and $\Omega_p$ is the prior's information matrix:

$$
e_p = \mathrm{Log}(X_{\text{ref}}^{-1} X_0), \qquad J_p = \mathcal{J}_r^{-1}(e_p)
$$

The first prior is the gauge anchor, with $X_{\text{ref}}$ equal to the ground-truth first pose and $\Omega_p = 10^6 I$ (`--anchor-weight`). Every later prior comes from marginalization (below).

**Gauss-Newton step** (`assemble_window_system`, `solve_to_convergence`). Each edge uses the information matrix $\Omega = I_6$. The script accumulates

$$
H = \sum J^\top \Omega\, J, \qquad g = -\sum J^\top \Omega\, e, \qquad H\delta = g, \qquad X_k \leftarrow X_k \exp(\delta_k)
$$

It stops when $\lVert\delta\rVert$ < `gn_tol` ($10^{-6}$) or after `gn_max_iters` (10) iterations. With $W$ poses in the window, $H$ is $6W \times 6W$.

**Marginalizing the oldest pose** (`marginalize_oldest`). The window has already converged. Let $a$ be the oldest pose, `x_window[0]`, and $b$ the next one, `x_window[1]`. In a pure chain, $a$ touches exactly two factors: its prior and the single a–b edge. The script asserts that there is exactly one such edge. It takes $\Lambda_{aa}$, $\Lambda_{ab}$ and $\Lambda_{ba}$ from the full window $H$; those blocks only ever involve factors touching $a$. It builds $\Lambda_{bb}^{(ab)}$ from a separate system containing only the a–b edge:

$$
\Omega_p' = \Lambda_{bb}^{(ab)} - \Lambda_{ba}\Lambda_{aa}^{-1}\Lambda_{ab}, \qquad X_{\text{ref}}' = X_b \text{ (current estimate, frozen)}
$$

Taking $\Lambda_{bb}$ from the full $H$ instead would also include the b–c edge. That edge stays in the window, so it would be counted twice. $\Omega_p'$ equals §4's Schur complement of the system built from $a$'s prior plus the a–b edge.

**Why storing no $\eta_b'$ is exact here.** Because $X_{\text{ref}}' = X_b$, the new prior's residual is zero when it is created, so its linear term is zero too. §4's $\eta_b'$ really is zero in this setting. The chain has a single anchor and no loop closures, so every factor can be satisfied exactly. Each new pose also starts at $X_{\text{last}} Z_{ij}$, which already zeroes its edge residual. So at the optimum, every residual, and therefore every gradient term, is zero. This depends on the pure-chain scope. With a loop closure or a shared landmark, residuals would not vanish, and dropping $\eta_b'$ would lose information.

**Ordering** (`run_sliding_window_pose_graph`). For each new node, the script:

1. Marginalizes the oldest pose first, if the window already holds `window_size` poses, using the linearization point from the previous solve.
2. Appends the new pose, initialized as $X_{\text{last}} Z_{ij}$, together with its edge.
3. Solves the window to convergence.

The solve therefore never exceeds `window_size` poses, so `max_dof` is capped at 6 × `window_size`. A marginalized pose keeps the estimate it had when it was dropped.

**Baseline** (`run_full_batch_growing`). For $k = 1 \dots n$, the baseline re-solves the whole graph of the first $k$ poses from scratch. It uses `pose_graph.run_pose_graph_optimization` with `damping=0.0`, starting from dead reckoning from the first ground-truth pose, with node 0 anchored by $10^6 I$ inside that function. Its `max_dof` is $6n$.

**Defaults:** `--window-size 10`; `--nodes-per-side-sweep 2 4 8 16 32 64` (8 to 256 poses); `--side-length 2.0` m; `--pos-noise-std 0.05` m; `--rot-noise-std 0.01` rad; `--anchor-weight 1e6`; `--gn-tol 1e-6`; `--gn-max-iters 10`; `--seed 0`.

## 9. Empirical verification: bounded vs. unbounded, for real

`sliding_window_marginalization.py` compares this bounded approach against `run_full_batch_growing` - the unbounded baseline that re-solves the entire graph from scratch at every new node, exactly the strategy §1 opens with. Since output bookkeeping (final pose estimates for every node, kept only for this script's own error reporting) is unavoidably $O(n)$ for *both* approaches alike, the metric that actually isolates the algorithmic claim is the size of the largest dense information matrix either one ever assembles and solves - reported here as `max_dof`, with an approximate byte count for holding that matrix densely ($`\text{max\_dof}^2 \times 8`$ bytes, float64):

| Trajectory length | Full-batch `max_dof` | Full-batch peak (~bytes) | Sliding-window `max_dof` (window_size=10) | Sliding-window peak (~bytes) |
| --- | --- | --- | --- | --- |
| 8 | 48 | 18,432 | 48 | 18,432 |
| 16 | 96 | 73,728 | 60 | 28,800 |
| 32 | 192 | 294,912 | 60 | 28,800 |
| 64 | 384 | 1,179,648 | 60 | 28,800 |
| 128 | 768 | 4,718,592 | 60 | 28,800 |
| 256 | 1536 | 18,874,368 | 60 | 28,800 |

Full-batch's system size grows linearly with trajectory length (so its dense-matrix memory grows *quadratically* - visible directly in the table, roughly $4\times$ per doubling of length) and never stops; sliding-window's caps at exactly $`6 \times \text{window\_size}`$ the moment the window first fills, and never moves again, confirmed identically across multiple seeds (`max_dof` depends only on trajectory length and `window_size`, not on the noise realization). Average per-step wall-clock time tells the same story less starkly, and its absolute values depend on the machine and its load. One run gave full-batch 1.5 ms → 61 ms as length grows 8 → 256, and sliding-window 1.6 ms → 7.0 ms. A later run on a different machine state gave full-batch 2.4 ms → 317 ms and sliding-window 0.78 ms → 3.8 ms. In both runs, sliding-window time flattens once trajectories exceed `window_size`, while full-batch time keeps growing. Only the `max_dof` column above is deterministic.

**Accuracy is not sacrificed to get this bound - but the reason why is specific to this setup, not a general property of marginalization.** Final RMS position error is *identical* (to displayed precision, every trajectory length, every seed tried) between full-batch and sliding-window. This isn't a coincidence and isn't the general case: with no loop closures anywhere in this chain, no future edge ever reaches back to inform an already-marginalized pose, so there is nothing later solving could have taught an earlier pose that marginalization threw away - the two approaches are solving genuinely equivalent problems. This is exactly why §6's FEJ subtlety matters in general and is silent here: FEJ protects against *inconsistency* that only shows up once a later loop closure or shared landmark reconnects to something already marginalized, which this script's pure-chain scope never triggers. A loop-closure-carrying version of this same script would be expected to show both §5's fill-in cost and a real (if likely small) accuracy gap from skipping FEJ - a natural further extension, not implemented here.

---

## 10. One-sentence summary

> **Marginalization is the same variable-elimination step `bayes_tree.md` uses to build a solve order, aimed instead at permanently discarding an old state - turning it into a dense prior factor over whatever it was still connected to, which is exactly the trick that lets sliding-window/fixed-lag smoothers (MSCKF, VINS-Mono) run in bounded memory and time forever, at the cost of a fill-in penalty and a consistency subtlety that a later fix (FEJ, §6) targets but that full-batch and iSAM2 never have to deal with in the first place.**

---

## 11. References

1. Sibley, G., Matthies, L., & Sukhatme, G. (2010). *Sliding Window Filter with Application to Planetary Landing*. Journal of Field Robotics, 27(5), 587-608. https://doi.org/10.1002/rob.20360 - the sliding-window/delayed-state-marginalization formulation behind §4 and §7.
2. Huang, G. P., Mourikis, A. I., & Roumeliotis, S. I. (2009). *A First-Estimates Jacobian EKF for Improving SLAM Consistency*. In Experimental Robotics: The Eleventh International Symposium (pp. 373-382). Springer. https://doi.org/10.1007/978-3-642-00196-3_43 - the FEJ fix behind §6.
3. Mourikis, A. I., & Roumeliotis, S. I. (2007). *A Multi-State Constraint Kalman Filter for Vision-Aided Inertial Navigation*. ICRA 2007, 3565-3572. https://doi.org/10.1109/ROBOT.2007.364024 - the original MSCKF reference in §7 (predates FEJ, per §6), already cited in [filtering_smoothing.md §12](../filtering_smoothing.md#12-references).
4. Qin, T., Li, P., & Shen, S. (2018). *VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator*. IEEE Transactions on Robotics, 34(4), 1004-1020. https://doi.org/10.1109/TRO.2018.2853729 - the VINS-Mono reference in §7, already cited in [filtering_smoothing.md §12](../filtering_smoothing.md#12-references). Its own §III-C explicitly discusses why it does not adopt FEJ.
5. Li, M., & Mourikis, A. I. (2013). *High-Precision, Consistent EKF-Based Visual-Inertial Odometry*. The International Journal of Robotics Research, 32(6), 690-711. https://doi.org/10.1177/0278364913481251 - the later MSCKF revision (sometimes called "MSCKF 2.0") that adds FEJ, referenced in §6's correction.
