# Bundle Adjustment

**Bundle Adjustment (BA)** sounds intimidating, but the intuition is actually quite simple:

> **Bundle Adjustment = jointly adjusting camera poses and 3D points so that the observed image measurements are explained as accurately as possible.**

The key word is **jointly**.

---

## 1. Start with a simple camera + 3D point

Imagine a camera looking at a 3D point:

```text
        3D point
           ● P
          / 
         /
        /
       📷 Camera
```

The 3D point $P$ gets projected onto the camera image:

```text
3D world                 Image

   P ●                      • p
      \                    /
       \                  /
        \                /
         📷 ------------ image plane
```

If we know:

* camera pose
* camera intrinsics
* 3D point position

we can predict where $P$ should appear in the image.

Mathematically:

$${p = \pi(T^{-1}P)}$$

where:

* $P$ = 3D point, in world coordinates
* $T$ = camera pose, camera-to-world (it maps camera-frame points into the world), so $T^{-1}P$ is the point expressed in the camera frame
* $\pi$ = camera projection function
* $p$ = predicted 2D pixel location

---

## 2. But our estimates are imperfect

Suppose the actual image measurement is:

```text
observed point:       ●
predicted point:          ×
```

There is an error:

```text
        ● observed

           ↕ error

              × predicted
```

This is called the **reprojection error**.

For one observation:

$${e = p_{\text{observed}} - p_{\text{predicted}}}$$

BA tries to make this error as small as possible.

---

## 3. Now add multiple cameras

Suppose the camera moves:

```text
Camera 1          Camera 2          Camera 3

  📷                📷                📷
   \                 \                 \
    \                 \                 \
     ● P               ● P               ● P
```

The same physical 3D point is observed from multiple camera positions.

Each camera produces a 2D observation:

```text
Camera 1 → p₁
Camera 2 → p₂
Camera 3 → p₃
```

We want to find:

* Camera 1 pose
* Camera 2 pose
* Camera 3 pose
* 3D position of P

such that **all projections agree with the observations**.

---

## 4. Here's the important part: both cameras AND points are adjusted

Suppose our initial reconstruction is wrong:

```text
Camera poses:

📷₁       📷₂       📷₃

 \         |         /
  \        |        /
   \       |       /
       ● P
```

Maybe the camera poses are slightly wrong.

Maybe the 3D point is slightly wrong.

Maybe **both** are wrong.

BA doesn't say:

> "The cameras are correct; I'll fix the points."

or:

> "The points are correct; I'll fix the cameras."

Instead:

> **"I'll adjust everything together until the entire reconstruction explains the image measurements as well as possible."**

That's bundle adjustment.

---

## 5. Why is it called "bundle" adjustment?

There's a beautiful geometric intuition.

Each image observation defines a **ray** from the camera through the observed pixel:

```text
Camera 1
   📷
    \
     \
      \       ● 3D point
       \     /
        \   /
         \ /
```

Another camera gives another ray:

```text
📷₁ --------\
             \
              ● P
             /
📷₂ --------/
```

Ideally, the rays intersect exactly at the 3D point.

But because of noise and imperfect estimates:

```text
📷₁ --------\
             \       ●
              \
               \

📷₂ -----------\ 
```

They don't intersect perfectly.

You can think of BA as adjusting the **bundle of rays** and camera poses so that everything fits together better.

(Recovering a single point from a bundle of already-known-pose rays - rather than jointly adjusting everything - is triangulation on its own; see [frontend/triangulation_pnp.md](../frontend/triangulation_pnp.md) for the closed-form math this repo actually implements, plus its exact inverse problem, PnP.)

Hence:

> **Bundle Adjustment.**

---

## 6. A more useful SLAM example

Imagine a robot/camera moving through a room:

```text
t₀       t₁       t₂       t₃
📷       📷       📷       📷
 \        \        \        \
  \        \        \        \
   ● A      ● B      ● C      ● D
    \       |       / 
     \      |      /
        landmarks
```

The camera observes many landmarks:

```text
      ● L1
     /
📷₀ /— — — — — ● L2
    \          
     \        
      ● L3
```

Every observation creates a constraint:

```text
camera pose + 3D landmark
              ↓
       predicted pixel
              ↓
       compare with
       observed pixel
```

So the system has potentially **thousands or millions of constraints**.

BA solves:

$${\min_{\{T_i\},\{P_j\}} \sum_{(i,j) \in \mathcal{O}} \left\|z_{ij} - \pi(T_i^{-1} P_j)\right\|^2}$$

where:

* $T_i$ = pose of camera `i`, camera-to-world (as in Section 1)
* $P_j$ = 3D landmark `j`, in world coordinates
* $z_{ij}$ = observed pixel
* $\pi(T_i^{-1}P_j)$ = predicted pixel
* $\mathcal{O}$ = the set of (camera, landmark) pairs that were actually observed - not every camera sees every landmark, so the sum only runs over real observations, not all $i,j$ combinations

In plain English:

> **Find the camera poses and 3D points that make the predicted image points match the actual image points as closely as possible.**

Real implementations usually wrap the squared reprojection error in a **robust loss** (e.g. Huber) instead of squaring it directly, so a handful of bad feature matches can't drag the whole reconstruction toward them - see [pose_graph_optimization.md](pose_graph_optimization.md#16-robust-loss-functions-used-to-handle-false-loop-closures)'s "Robust loss functions" section for the exact same idea applied to pose graphs. `bundle_adjustment.py` keeps the plain, un-robustified squared error above, matching the objective as written here.

### 6.1 The BA math, concretely

`bundle_adjustment.py` minimizes the objective above with plain Gauss-Newton. Each pose $T_i$ is a $4\times4$ camera-to-world matrix with rotation $R_i$ and translation $t_i$. Each landmark $P_j$ is a world point. Pose corrections are 6-vectors $\delta = [\delta v, \delta\omega]$, translation first.

**Projection** (`camera_project`): move the point into the camera frame, then apply the pinhole model with intrinsics $(f_x, f_y, c_x, c_y)$:

$$
p_c = R_i^\top (P_j - t_i) = \begin{bmatrix} x \\ y \\ z \end{bmatrix}, \qquad
\pi(p_c) = \begin{bmatrix} f_x\, x/z + c_x \\ f_y\, y/z + c_y \end{bmatrix}
$$

The code clamps $|z|$ to at least $10^{-9}$, keeping its sign, so a point at the camera center can't divide by zero.

**Jacobians** (`camera_project(..., with_jacobians=True)`): both go through $p_c$ by the chain rule.

$$
J_{\text{proj}} = \frac{\partial \pi}{\partial p_c} = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{bmatrix}
$$

$$
J_{\text{pose}} = J_{\text{proj}} \begin{bmatrix} -I_3 & [p_c]_\times \end{bmatrix}, \qquad
J_{\text{point}} = J_{\text{proj}}\, R_i^\top
$$

$J_{\text{pose}}$ is for a right perturbation $`T_i \leftarrow T_i\,\mathrm{Exp}(\delta)`$. The block $`[-I_3 \;\; [p_c]_\times]`$ comes from inverting the perturbed pose, which puts the perturbation on the left with a minus sign:

$$
\left(T_i\,\mathrm{Exp}(\delta)\right)^{-1} P_j = \mathrm{Exp}(-\delta)\, p_c \approx p_c - \delta v - \delta\omega \times p_c = p_c - \delta v + [p_c]_\times \delta\omega
$$

**Gauss-Newton step.** The residual is $r_{ij} = z_{ij} - \pi(p_c)$. It linearizes to $r_{ij} - J\delta$, so every solver builds and solves the same normal equations:

$$
H\,\delta = g, \qquad H = \sum_{(i,j)} J^\top W J, \qquad g = \sum_{(i,j)} J^\top W r_{ij}
$$

| Solver | Unknowns | $J$ per observation | $W$ | Added to $H$ |
| --- | --- | --- | --- | --- |
| `run_ba_landmarks_only` | one landmark at a time (poses fixed): a separate $3\times3$ solve per landmark | $J_{\text{point}}$ | $1$ | $`10^{-9} I_3`$ |
| `run_ba_poses_only` | one camera at a time (landmarks fixed): a separate $6\times6$ solve per camera | $J_{\text{pose}}$ | $1$ | $`10^{-9} I_6`$ |
| `run_bundle_adjustment` | every pose and landmark: one dense system of size $6N + 3M$ | both blocks | $`\omega_{\text{px}} = 1/\sigma_{\text{px}}^2`$ | $`10^{-6} I`$ plus the gauge prior below |

All three solvers then apply $`T_i \leftarrow T_i\,\mathrm{Exp}(\delta_i)`$ and $P_j \leftarrow P_j + \delta_j$. They stop when the step norm $`\|\delta\|`$ drops below `gn_tol`. The joint system has the arrow-head block structure described in Section 12, but the script solves it densely.

**Gauge prior** (`run_bundle_adjustment`): joint BA has a 7-DoF gauge freedom (rigid motion plus scale). The script fixes it with a prior factor on cameras 0 and 1. The prior mean $\bar T_k$ is each camera's own noisy initial guess:

$$
e_k = \mathrm{Log}\left(\bar T_k^{-1} T_k\right), \qquad
J_k = J_r^{-1}(e_k), \qquad
\Omega_{\text{prior}} = \frac{I_6}{\sigma_{\text{pose}}^2}
$$

$$
H_{kk} \mathrel{+}= J_k^\top \Omega_{\text{prior}} J_k, \qquad g_k \mathrel{+}= -J_k^\top \Omega_{\text{prior}}\, e_k
$$

$J_r^{-1}$ is `compute_se3_inv_right_jacobian`. One prior pose would remove only the 6 rigid DoF. The second one also pins the distance between the two cameras, and that fixes scale. The prior is finite ($\sigma_{\text{pose}}$ is the same 0.1 used to perturb the initial guess), so camera 1's actual error can still be corrected by its observations. Because of the prior, `run_bundle_adjustment` minimizes the objective above weighted by $\omega_{\text{px}}$, plus these two prior terms.

**Initial guess** (`perturb_initial_guess`): $`T_i \leftarrow T_i\,\mathrm{Exp}(\xi)`$ with $`\xi \sim \mathcal{N}(0, \sigma_{\text{pose}}^2 I_6)`$, and $P_j \leftarrow P_j + \mathcal{N}(0, \sigma_{\text{lm}}^2 I_3)$.

Script defaults: 8 cameras on a 180° arc of radius 5 m, 60 landmarks sampled (kept only if at least 2 cameras see them, 70° field of view), $f_x = f_y = 800$ px on a 640×480 image, $\sigma_{\text{pose}} = 0.1$, $\sigma_{\text{lm}} = 0.3$ m, $\sigma_{\text{px}} = 1$ px, `gn_tol` = $10^{-6}$, at most 30 iterations. `reprojection_rms` reports the RMS of $`\|z_{ij} - \pi(T_i^{-1}P_j)\|`$ before the Umeyama alignment of Section 14.

---

## 7. Why BA is so powerful

Suppose your estimated trajectory looks like:

```text
Initial:

📷──📷──📷──📷──📷
               \
                \
                 ● landmarks
```

But the actual observations suggest that the cameras should be slightly different:

```text
Optimized:

📷
  \
   📷
     \
      📷
        \
         📷
           \
            📷
```

At the same time, the landmarks move too.

So BA might effectively do:

```text
             Before             After

Camera 1       ×                  ●
Camera 2       ×                  ●
Camera 3       ×                  ●
Camera 4       ×                  ●

Landmark A     ×                  ●
Landmark B     ×                  ●
Landmark C     ×                  ●
```

Everything moves together to minimize the total reprojection error.

---

## 8. Connection to SLAM optimization

This is exactly why BA is an **optimization-based SLAM technique**.

Remember our previous discussion:

> Filtering → maintain the current belief.

> Optimization → maintain many states and jointly improve them.

BA is the latter.

You have a giant optimization problem:

```text
        Camera poses
             ↓
     T₀ T₁ T₂ T₃ T₄
      ↘  ↓  ↙ ↘  ↓
       landmarks
      P₀ P₁ P₂ P₃
             ↓
       projection
             ↓
     predicted pixels
             ↓
     compare with data
             ↓
      total error
             ↓
       optimization
             ↓
      better poses +
      better points
```

---

## 9. BA vs Pose Graph Optimization

This distinction is particularly useful in SLAM.

### Pose graph optimization

Usually optimizes:

$${T_0,T_1,\ldots,T_n}$$

using relative pose constraints:

```text
T₀ ───── T₁ ───── T₂ ───── T₃
 \                         /
  └────── loop closure ───┘
```

The landmarks may already have been marginalized or aren't explicitly part of the optimization.

---

### Bundle Adjustment

Optimizes:

$${\boxed{\text{camera poses + 3D landmarks}}}$$

```text
        T₀       T₁       T₂
         \        |       /
          \       |      /
           P₁    P₂    P₃
```

using **image reprojection errors**.

So a useful mental distinction is:

> **Pose graph:** "Make the poses geometrically consistent."

> **Bundle adjustment:** "Make the entire camera + 3D structure explain the images."

---

## 10. Why BA can be computationally expensive

Imagine:

* 1,000 camera poses
* 100,000 landmarks
* millions of image observations

Then you're optimizing a huge number of variables.

But there's a very useful structure:

```text
Camera variables ─── Landmark variables
       ↕                    ↕
       └──── observations ──┘
```

Camera $T_1$ only directly interacts with the landmarks it observes.

That produces a **sparse optimization problem**.

This sparsity is one of the fundamental reasons efficient BA algorithms are possible.

---

## 11. The deepest intuition

Here's how I'd recommend thinking about BA:

Imagine you have a pile of photographs and you're trying to reconstruct a miniature 3D world.

You initially make a rough reconstruction:

```text
     camera       camera
       📷           📷
        \           /
         \         /
          ●       ●
          landmarks
```

Then you ask:

> "If this really were the correct 3D world, would these cameras really see these landmarks at exactly these pixels?"

If not, something is wrong.

Maybe:

* camera 1 is slightly misplaced
* camera 2 is rotated incorrectly
* landmark 1 is too far away
* landmark 2 is too high

So you continuously adjust:

```text
camera poses
     +
3D landmarks
     ↓
projection
     ↓
image error
     ↓
optimization
     ↓
repeat
```

until the reconstruction becomes as consistent with the images as possible.

That's **Bundle Adjustment**.

---

## 12. Block sparsity and the Schur complement

Section 10 already named the sparsity that makes large BA problems tractable; here's the mechanics of how solvers actually exploit it. The same structural fact drives everything below: a residual for point $P_j$ seen by camera $T_i$ depends **only** on $T_i$ and $P_j$ - it's completely independent of any other camera or point.

```text
Camera variables ─── Landmark variables
       ↕                    ↕
       └──── observations ──┘
```

That independence gives the linearized normal-equations Hessian ${H = J^\top J}$ a distinctive **arrow-head** block structure: block-diagonal camera-camera blocks, block-diagonal point-point blocks, and off-diagonal camera-point coupling blocks - with no direct camera-camera or point-point coupling anywhere.

```text
        Cameras          Points
      ┌──────────┬────────────────┐
Cams  │  block-  │                │
      │ diagonal │    coupling    │
      ├──────────┼────────────────┤
Pts   │ coupling │     block-     │
      │          │    diagonal    │
      └──────────┴────────────────┘
```

Since real scenes usually have far more points than cameras, solvers exploit this with the **Schur complement trick**: marginalize out the point block first (cheap, since it's block-diagonal - each point's own small block inverts independently), solve the much smaller reduced camera-only system, then cheaply back-substitute to recover the points. It's the same style of sparsity exploitation that makes [`sparse_cholesky_factorization.md`](sparse_cholesky_factorization.md)'s approach tractable at scale.

`bundle_adjustment.py` doesn't need this trick - its toy scenes are small enough (a handful of cameras and landmarks) that `run_bundle_adjustment` just solves the full dense joint system directly every iteration. Schur-complement marginalization is what a production solver (COLMAP, g2o, GTSAM, Ceres) does under the hood at real scene sizes, not something this demo implements.

$H = J^\top J$ above is the plain textbook form. The scripts' actual $H$ has extra terms. `run_bundle_adjustment` weights every reprojection block by $\omega_{\text{px}} = 1/\sigma_{\text{px}}^2$, adds the gauge-prior blocks on cameras 0 and 1, and adds $10^{-6} I$ (see [Section 6.1](#61-the-ba-math-concretely)). `run_windowed_gn_lm` in `bundle_adjustment_advanced.py` uses no weighting but solves $`(H + \lambda\,\mathrm{diag}(H))\,\delta = g`$ (see Section 13). All of these extra terms sit on diagonal blocks, so none of them changes the sparsity pattern.

---

## 13. Local vs. Global Bundle Adjustment (real systems)

Local BA and Global BA solve the *exact same* objective from Section 6 - they differ only in how much of the problem gets optimized at once, a choice driven by very different system constraints.

| Metric | **Local BA** (e.g. ORB-SLAM) | **Global BA** (e.g. COLMAP) |
| --- | --- | --- |
| System paradigm | Visual SLAM (real-time, online) | Structure-from-Motion (offline, batch) |
| Optimization scope | A local window of recent keyframes + covisible neighbors | Every registered camera pose and every 3D point |
| Input | Sequential video with continuous tracking | Unordered photo collections (or long video) |
| Frequency | Continuous, runs on every new keyframe | Periodic (e.g. every ~10-20% map growth) or a final pass |
| Outlier handling | Fast local robust cost (Huber) + chi-square gating | Heavy re-triangulation, track merging/filtering |
| Scaling | Roughly constant per window | Grows cubically with the number of camera poses |

### Local BA (ORB-SLAM)

Re-optimizing the entire map on every camera move is impossible in real time, so Local BA **trades global consistency for speed** by isolating a small subgraph:

```text
  [Fixed Keyframe]  sees -> (Fixed Map Point)
         |                             |
  (Covisible Link)               (Observed by)
         |                             v
 [Active Keyframe] < optimizes > [Active Map Point]
```

* **Active keyframes**: the new keyframe plus its neighbors in the **covisibility graph** (keyframes sharing many observed points).
* **Active points**: every 3D point observed by an active keyframe.
* **Fixed keyframes**: other keyframes that also see an active point, held fixed as rigid anchors so the local window can't drift the map's global frame.

Because a covisibility neighborhood's size stays roughly constant regardless of total map size, Local BA runs in bounded, real-time-friendly time - at the cost of letting small errors accumulate into global drift over a long trajectory. SLAM systems correct that separately, via loop closure + pose-graph optimization ([pose_graph_optimization.md](pose_graph_optimization.md)) or an occasional Global BA pass.

### Global BA (COLMAP)

Offline SfM pipelines sacrifice real-time speed for maximum accuracy: as COLMAP incrementally registers new images, it periodically re-optimizes **every** camera and **every** point jointly in one large least-squares problem, then uses the resulting global residuals to prune bad matches and re-triangulate points - something a local window can never do, since it never sees the whole map at once. The cost is that even after the Schur complement from Section 12, the reduced camera-only system still grows cubically with the number of camera poses - and assembling that Schur complement in the first place costs roughly linear time in the number of points/observations - so this can only run periodically or as a final step, not every frame.

### Choosing between them

Use **Local BA** for real-time robotics/AR/VR where sub-30ms latency matters more than perfect global consistency (loop closure repairs that later).

Use **Global BA** for offline reconstruction - meshes, NeRF/Gaussian-Splatting input scenes, photogrammetric surveys - where total geometric fidelity matters more than runtime.

`bundle_adjustment.py`'s three solvers (`run_ba_landmarks_only`, `run_ba_poses_only`, `run_bundle_adjustment`) are all single-batch joint solves over the whole toy scene - closest in spirit to a (tiny) Global BA pass. It has no windowing, no covisibility graph, and no incremental registration, so it doesn't model Local BA's real-time system behavior at all.

`bundle_adjustment_advanced.py` fills that gap: a camera moves keyframe-by-keyframe through a landmark corridor, a covisibility graph is built incrementally, and every new keyframe triggers a bounded local-BA solve over an active window (new keyframe + covisible neighbors, with every other observing keyframe held fixed as a rigid anchor - this section's diagram, made concrete), while a periodic Global BA pass over the whole map runs alongside it for direct comparison. Measuring wall-clock solve time confirms both halves of this section's "Scaling" row in code: the local window's cost stays roughly flat as the map grows, while the Global BA pass's cost grows with it. It also demonstrates *why* an occasional Global BA pass helps - accumulated front-end drift - though the size and even the direction of that help on the default open (non-looping) path is noisier than a single run can show: the default seed looks like a clean win (final RMS trajectory position error goes from 1.8652 m Local-only to 1.2692 m Local+Global, a 32% reduction), but a wider 15-seed sweep (this script's own regression test) puts the true picture close to a wash - a roughly 50% per-seed win rate, with the *median* difference near zero either way, and a handful of seeds diverging to thousands of meters in *either* mode from a rare bad local minimum in the windowed GN/LM solve that neither mode is protected from. Don't read the default seed's 32% number as "the" benefit of Global BA here; it's one noise realization, not the typical case. The covisibility bookkeeping, active-window builder, and Global BA solve are all already loop-closure-agnostic (none of them assume temporal locality), so a genuine loop closure needs no code changes - just a path that revisits a place: passing `--arc-span-deg` close to 360 (e.g. 350) swings the path's end back within view range of its own start, auto-detected and reported as `Loop closure detected at keyframe K ...` with the measured before/after trajectory RMS. That stress test doesn't showcase Global BA's payoff any more clearly than the open path above, either - across several seeds the RMS trajectory error right after the loop-closing Global BA pass ranges from a 9.9% improvement to a 6.2% regression (default seed: 5.478 m → 5.504 m, a −1.3% "reduction"). The loop-closure edge does supply a genuinely new constraint, as intended, but on this small scene it's one new residual competing against hundreds of others in the same joint solve, not a dominant correction - so neither the open path nor the loop-closure stress test is a reliable demonstration of Global BA's payoff on any single seed; the real story is in the aggregate statistics, not any one run's printed numbers.

**The windowed solver, concretely** (`run_windowed_gn_lm`). Local and Global BA call the same solver. They differ only in which poses and points are unknowns:

- **Local** (`build_active_window`, `run_local_ba_step`): the new keyframe $k$ plus up to `max_window_keyframes` − 1 of its covisible neighbors (keyframes sharing at least `min_shared_for_covisibility` landmarks with $k$, strongest first) are active. Every already-triangulated landmark those keyframes see is active too. Every other keyframe that observes an active landmark is held fixed.
- **Global** (`run_global_ba`): every keyframe processed so far and every triangulated landmark are unknowns.
- **Gauge**: keyframes 0 and 1 (`anchor_keyframes`) are never unknowns in either solve. Keyframe 0 starts at its true pose. Keyframe 1 keeps its front-end estimate. There is no prior factor.

It uses the residual and Jacobians from [Section 6.1](#61-the-ba-math-concretely), unweighted ($W = I$). An observation from a fixed keyframe adds only its $J_{\text{point}}$ block. Each iteration solves a Levenberg-Marquardt system:

$$
\left(H + \lambda\,\mathrm{diag}(H)\right)\delta = g, \qquad \text{cost} = \sum \|z - \pi(T^{-1}P)\|^2
$$

Diagonal entries of $H$ below $10^{-12}$ are floored to $10^{-12}$ before scaling. $\lambda$ starts at $10^{-3}$ in every call. A trial step is accepted only if it lowers the cost, and then $\lambda \leftarrow \max(\lambda/2, 10^{-7})$. Otherwise $\lambda \leftarrow 2\lambda$ and the solve is retried, up to 10 times. The loop stops when no step is accepted after 10 retries, when the step norm drops below `gn_tol`, or after `gn_max_iters` iterations.

Two cheirality rules protect the solve, using `point_depth` (the $z$ of $R^\top(P - t)$). First, any observation whose point is already behind its camera when the call starts is dropped for this call (gating). Second, a trial that puts any remaining point behind its camera gets infinite cost, so it is rejected. New landmarks are triangulated once they have `min_observations` observers (`triangulate_landmark`, then `refine_landmark_gn`, see [triangulation_pnp.md](../frontend/triangulation_pnp.md)). They are committed only if `passes_cheirality` holds. After each local and global solve, `cull_invalid_points` deletes any landmark that is now behind one of its observers, so it can be triangulated again later.

Script defaults: 50 keyframes on a 90° arc of radius 15 m, 8 landmarks per keyframe, 6 m view range, `min_observations` = 3, `min_shared_for_covisibility` = 2, `max_window_keyframes` = 6, a Global BA pass every 8 keyframes (plus one at the end), `gn_tol` = $10^{-6}$, `gn_max_iters` = 15, front-end se3 twist noise std 0.02 per step, $\sigma_{\text{px}} = 1$ px.

---

## 14. Evaluating the result: gauge freedom and Umeyama alignment

Monocular BA recovers the scene only up to an unknown similarity transform (rigid + scale) - shifting, rotating, or uniformly rescaling the whole reconstructed scene and every camera pose together leaves reprojection error completely unchanged. `bundle_adjustment.py`'s `run_bundle_adjustment` pins that freedom down to *some* solution with a soft gauge-prior factor on the first two camera poses, but the resulting frame still won't match ground truth's frame or scale exactly. So before computing pose/landmark error, the script Umeyama-aligns the solved cameras and landmarks onto ground truth with one shared scale+rotation+translation - see [umeyama_alignment.md](../foundations/umeyama_alignment.md) for how that alignment is computed and why it's needed. Reprojection error itself is reported *before* this alignment step and is unaffected by it.

`bundle_adjustment_advanced.py` sidesteps this entirely: hard-fixing two anchor keyframes (rather than a soft prior) removes all residual gauge freedom up front, so there's nothing left to align away before reporting its error.

---

## 15. One sentence to remember

> **Bundle Adjustment is the process of jointly refining camera poses and 3D landmarks so that their projections agree as closely as possible with the observed image features.**

And the conceptual hierarchy is:

```text
SLAM
 │
 ├── Filtering
 │     └── EKF-SLAM
 │
 └── Optimization / Smoothing
       │
       ├── Pose Graph Optimization
       │
       └── Bundle Adjustment
              │
              ├── optimize camera poses
              └── optimize 3D landmarks
```

For **visual SLAM**, BA is essentially the workhorse behind the idea of *"make my entire reconstructed 3D world and camera trajectory agree with all the pixels/features I've observed."*

---

## 16. References

1. Triggs, B., McLauchlan, P. F., Hartley, R. I., & Fitzgibbon, A. W. (2000). *Bundle Adjustment - A Modern Synthesis*. In B. Triggs, A. Zisserman, & R. Szeliski (Eds.), Vision Algorithms: Theory and Practice (Vol. 1883, pp. 298-372). Springer. https://doi.org/10.1007/3-540-44480-7_21 - the standard reference survey behind this whole doc's framing (joint pose+landmark refinement, the Schur complement in §12, and the "modern synthesis" of BA as a sparse nonlinear least-squares problem rather than a purely photogrammetric one).
