# Bundle Adjustment

**Bundle Adjustment (BA)** in one line:

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
     P ●              3D point, world frame
        \
         \            ray from P to the camera center
  ────────•────────   image plane: the ray crosses it at the pixel p
           \
            \
             📷       camera center
```

If we know:

- camera pose
- camera intrinsics (the camera's fixed calibration: focal lengths and image center)
- 3D point position

we can predict where $P$ should appear in the image.

Mathematically:

$$p = \pi(T^{-1}P)$$

where:

- $P$ = 3D point, in world coordinates
- $T$ = camera pose, camera-to-world (it maps camera-frame points into the world), so $T^{-1}P$ is the point expressed in the camera frame
- $\pi$ = camera projection function
- $p$ = predicted 2D pixel location

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
      \
       \   reprojection error e
        \
         × predicted
```

This is the **reprojection error**. For one observation:

$$e = p_{\text{observed}} - p_{\text{predicted}}$$

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

- Camera 1 pose
- Camera 2 pose
- Camera 3 pose
- 3D position of P

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

Maybe the camera poses are slightly wrong, maybe the 3D point is slightly wrong, maybe **both** are.

BA doesn't say "the cameras are correct, I'll fix the points" or "the points are correct, I'll fix the cameras". Instead:

> **"I'll adjust everything together until the entire reconstruction explains the image measurements as well as possible."**

To see why both must move, look at the [RMSE](../glossary.md#rmse-root-mean-square-error) panel of the figure in [§6.1](#61-the-ba-math-concretely). It compares landmarks-only, poses-only and joint bundle adjustment.

---

## 5. Why is it called "bundle" adjustment?

Each image observation defines a **ray** from the camera through the observed pixel. Two cameras give two rays, which ideally meet at the 3D point:

```text
📷₁ --------\
             \
              ● P
             /
📷₂ --------/
```

With noise and imperfect estimates, they don't intersect perfectly:

```text
📷₁ --------\
             \       ●
              \
               \

📷₂ -----------\
```

We can think of BA as adjusting the **bundle of rays** and the camera poses together until everything fits.

(Recovering a single point from rays with already-known poses, rather than adjusting everything jointly, is triangulation on its own. See [frontend/triangulation_pnp.md](../frontend/triangulation_pnp.md) for the closed-form math this repo implements, plus its dual problem, PnP: recovering a camera pose from known points.)

![Two top-view panels from bundle_adjustment.py's defaults: rays back-projected through one landmark's measured pixels from all eight cameras, scattered at the noisy initial guess and meeting at one point after joint bundle adjustment](../../assets/bundle_adjustment_concept.png)

*Figure: the bundle of rays for one landmark in `use_numpy/bundle_adjustment.py`'s default scene (seed 0), plotted by `uv run python assets/make_figures.py bundle_adjustment_concept`.*

---

## 6. A more useful SLAM example

A robot/camera moves through a room. Consecutive cameras see overlapping sets of landmarks (✓ = observed):

| | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| camera $t_0$ | ✓ | ✓ | | | |
| camera $t_1$ | | ✓ | ✓ | | |
| camera $t_2$ | | | ✓ | ✓ | |
| camera $t_3$ | | | | ✓ | ✓ |

Every observation creates a constraint, and the system can have **thousands or millions** of them.

Each constraint compares the predicted pixel (from camera pose + 3D landmark) with the observed pixel.

BA solves:

```math
\min_{\{T_i\},\{P_j\}} \sum_{(i,j) \in \mathcal{O}} \left\|z_{ij} - \pi(T_i^{-1} P_j)\right\|^2
```

where:

- $T_i$ = pose of camera `i`, camera-to-world (as in §1)
- $P_j$ = 3D landmark `j`, in world coordinates
- $z_{ij}$ = observed pixel
- $\pi(T_i^{-1}P_j)$ = predicted pixel
- $\mathcal{O}$ = the set of (camera, landmark) pairs that were actually observed - not every camera sees every landmark, so the sum only runs over real observations, not all $i,j$ combinations


In words: **find the camera poses and 3D points that make the predicted image points match the actual image points as closely as possible.**

Real implementations usually wrap the squared reprojection error in a **robust loss** (e.g. Huber) instead of squaring it directly, so a handful of bad **feature matches** (wrong pairings of the same image point across images) can't drag the whole reconstruction toward them. The ["Robust loss functions" section of pose_graph_optimization.md](pose_graph_optimization.md#16-robust-loss-functions-used-to-handle-false-loop-closures) applies the same idea to pose graphs. `bundle_adjustment.py` keeps the plain, un-robustified squared error above, matching the objective as written here.

### 6.1 The BA math, concretely

`bundle_adjustment.py` minimizes the objective above with plain Gauss-Newton ([glossary](../glossary.md#gauss-newton)). Each pose $T_i$ is an $SE(3)$ element ([glossary](../glossary.md#so3-and-se3)), stored as a $4\times4$ camera-to-world matrix with rotation $R_i$ and translation $t_i$. Each landmark $P_j$ is a world point. Pose corrections are 6-vectors $\delta = [\delta v, \delta\omega]$, translation first.

![Two panels from bundle_adjustment.py: a top-down view of cameras on an arc around a landmark cluster (ground truth, noisy initial guess, joint BA), and log-scale RMSE for the noisy initial guess, landmarks-only, poses-only and joint bundle adjustment](../../assets/bundle_adjustment.png)

*Figure: `use_numpy/bundle_adjustment.py` at its defaults (seed 0), plotted by `uv run python assets/make_figures.py bundle_adjustment`.*

The same run, animated over its Gauss-Newton iterations. Optimizing poses and landmarks jointly brings the reprojection error from 127.7 px to 1.2 px within three iterations, while the one-sided solvers, which hold either the landmarks or the poses at their noisy guess, stop at 100.1 px and 53.9 px.

![Animation of bundle_adjustment.py: left, a top-down view of cameras and landmarks moving from the noisy initial guess onto ground truth over the Gauss-Newton iterations; right, the reprojection RMSE falling to about 1.2 px on a log scale, below the landmarks-only (100 px) and poses-only (54 px) levels](../../assets/bundle_adjustment.gif)

*Animation: `use_numpy/animate_bundle_adjustment.py` at its defaults (seed 0), rendered by `uv run python use_numpy/animate_bundle_adjustment.py --out assets/bundle_adjustment.gif`. Each iterate is aligned to ground truth for display (**[gauge freedom](../glossary.md#gauge-freedom):** the whole scene can shift, rotate or rescale without changing any reprojection error), and the motion between the 7 iterates is interpolated.*

#### Projection and Jacobians

**Projection** (`camera_project`): move the point into the camera frame, then apply the **pinhole model** (an ideal camera whose rays all pass through one point) with intrinsics $(f_x, f_y, c_x, c_y)$:

```math
p_c = R_i^\top (P_j - t_i) = \begin{bmatrix} x \\ y \\ z \end{bmatrix}, \qquad
\pi(p_c) = \begin{bmatrix} f_x\, x/z + c_x \\ f_y\, y/z + c_y \end{bmatrix}
```

Here $`z`$ is the point's depth in the camera frame, not the pixel observation $`z_{ij}`$. The code clamps $|z|$ to at least $10^{-9}$, keeping its sign, so a point at the camera center can't divide by zero.

**Intuition (projection Jacobian,** [glossary](../glossary.md#jacobian)**):** it says how many pixels the image point moves per metre the 3D point moves, and far points move less.

- **Tiny example (projection Jacobian):** shift a point $`1`$ cm sideways, with $`f_x = 800`$. The pixel moves by $`f_x \cdot 0.01 / z`$.
- At $`z = 5`$ m that is $`1.6`$ px. At $`z = 10`$ m it is $`0.8`$ px. This is the $`f_x/z`$ entry of $`J_{\text{proj}}`$ below (the projection Jacobian $`\partial\pi/\partial p_c`$).
- Sign: if the camera moves right by $`d`$, the point moves left by $`d`$ in the camera frame. This is the $`-I_3`$ in the pose block below.

**Jacobians** (`camera_project(..., with_jacobians=True)`): both go through $p_c$ by the chain rule.

```math
J_{\text{proj}} = \frac{\partial \pi}{\partial p_c} = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{bmatrix}
```

```math
J_{\text{pose}} = J_{\text{proj}} \begin{bmatrix} -I_3 & [p_c]_\times \end{bmatrix}, \qquad
J_{\text{point}} = J_{\text{proj}}\, R_i^\top
```

Here $`[p_c]_\times`$ is the skew-symmetric **hat** matrix of $`p_c`$ ([glossary](../glossary.md#hat-operator)) and $`\mathrm{Exp}`$ is the **exponential map** ([glossary](../glossary.md#exp-and-log-maps)). $J_{\text{pose}}$ is for a right perturbation $`T_i \leftarrow T_i\,\mathrm{Exp}(\delta)`$. The block $`[-I_3 \;\; [p_c]_\times]`$ comes from inverting the perturbed pose, which puts the perturbation on the left with a minus sign:

```math
\left(T_i\,\mathrm{Exp}(\delta)\right)^{-1} P_j = \mathrm{Exp}(-\delta)\, p_c \approx p_c - \delta v - \delta\omega \times p_c = p_c - \delta v + [p_c]_\times \delta\omega
```

#### Gauss-Newton step and solvers

**Gauss-Newton step.** The residual is $r_{ij} = z_{ij} - \pi(p_c)$. It linearizes to $r_{ij} - J\delta$, so every solver builds and solves the same [normal equations](../glossary.md#normal-equations-and-hessian). Here $`\Omega`$ is the weight ([information](../glossary.md#information-matrix)) matrix of one pixel residual: $`I_2`$ for the one-sided solvers, and $`\Omega_{\text{px}} = \omega_{\text{px}} I_2`$ with $`\omega_{\text{px}} = 1/\sigma_{\text{px}}^2`$ for the joint solver, where $`\sigma_{\text{px}}`$ is the pixel-noise standard deviation (see the table). $`H`$ is the Gauss-Newton **Hessian** (the information matrix of the linearized problem):

```math
H\,\delta = g, \qquad H = \sum_{(i,j)} J^\top \Omega J, \qquad g = \sum_{(i,j)} J^\top \Omega\, r_{ij}
```

Because $`r`$ is measured minus predicted, the right-hand side is $`+g`$. In the glossary's notation (residual = prediction − measurement, $`H\,\delta = -b`$) this is $`g = -b`$; [gauss_newton.md](gauss_newton.md) uses the glossary sign and has a note on the difference.

| Solver | Unknowns | $J$ per observation | $\Omega$ | Added to $H$ |
| --- | --- | --- | --- | --- |
| `run_ba_landmarks_only` | one landmark at a time (poses fixed): a separate $3\times3$ solve per landmark | $J_{\text{point}}$ | $I_2$ | $`10^{-9} I_3`$ |
| `run_ba_poses_only` | one camera at a time (landmarks fixed): a separate $6\times6$ solve per camera | $J_{\text{pose}}$ | $I_2$ | $`10^{-9} I_6`$ |
| `run_bundle_adjustment` | every pose and landmark: one dense system of size $6N + 3M$ | both blocks | $`\Omega_{\text{px}} = \omega_{\text{px}} I_2`$ | $`10^{-6} I`$ plus the gauge prior below |

All three solvers then apply $`T_i \leftarrow T_i\,\mathrm{Exp}(\delta_i)`$ and $P_j \leftarrow P_j + \delta_j$. They stop when the step norm $`\|\delta\|`$ drops below `gn_tol`. The joint system has the arrow-head block structure described in §12, but the script solves it densely.

#### Gauge prior and initial guess

**Gauge prior** (`run_bundle_adjustment`): joint BA has a 7-[DoF](../glossary.md#degrees-of-freedom-dof) gauge freedom (rigid motion plus scale). The script fixes it with a prior factor on cameras 0 and 1. The prior mean $\bar T_k$ is each camera's own noisy initial guess:

```math
e_k = \mathrm{Log}\left(\bar T_k^{-1} T_k\right), \qquad
J_k = J_r^{-1}(e_k), \qquad
\Omega_{\text{prior}} = \frac{I_6}{\sigma_{\text{pose}}^2}
```

```math
H_{kk} \mathrel{+}= J_k^\top \Omega_{\text{prior}} J_k, \qquad g_k \mathrel{+}= -J_k^\top \Omega_{\text{prior}}\, e_k
```

$J_r^{-1}$ is the inverse **right Jacobian** ([glossary](../glossary.md#right-and-left-jacobians)), computed by `compute_se3_inv_right_jacobian`; $`\mathrm{Log}`$ is the **logarithm map** (inverse of $`\mathrm{Exp}`$, same glossary section) and $`\sigma_{\text{pose}}`$ is the standard deviation of the prior on camera pose.

- One prior pose would remove only the 6 rigid DoF. The second also pins the distance between the two cameras, which fixes scale.
- The prior is finite ($\sigma_{\text{pose}}$ is the same 0.1 used to perturb the initial guess), so camera 1's actual error can still be corrected by its observations.
- So `run_bundle_adjustment` minimizes the objective above weighted by $\omega_{\text{px}}$, plus these two prior terms.

**Initial guess** (`perturb_initial_guess`): $`T_i \leftarrow T_i\,\mathrm{Exp}(\xi)`$ with $`\xi \sim \mathcal{N}(0, \sigma_{\text{pose}}^2 I_6)`$, and $P_j \leftarrow P_j + \mathcal{N}(0, \sigma_{\text{lm}}^2 I_3)$.

#### Script defaults


- Scene: 8 cameras on a 180° arc of radius 5 m, 60 landmarks sampled. A camera sees a landmark only if it projects inside the image, and landmarks seen by fewer than 2 cameras are dropped. The defaults keep 48 landmarks and 292 observations.
- Camera: $f_x = f_y = 800$ px on a 640×480 image (about 44°×33° field of view).
- Noise: $\sigma_{\text{pose}} = 0.1$, $\sigma_{\text{lm}} = 0.3$ m, $\sigma_{\text{px}} = 1$ px.
- Solver: `gn_tol` = $10^{-6}$, at most 30 iterations.
- `reprojection_rms` reports the RMSE of $`\|z_{ij} - \pi(T_i^{-1}P_j)\|`$. It is unaffected by the Umeyama alignment of §14, because reprojection error is invariant to a **similarity transform** (rotation, translation and uniform scale).


---

## 7. Why BA is so powerful

Suppose the estimated trajectory is off. The observations say the cameras should be somewhere else, and the landmarks move too.

Everything moves together to minimize the total reprojection error.

---

## 8. Connection to SLAM optimization

BA is an **optimization-based SLAM technique**. Recall the two estimation strategies from [filtering_smoothing.md](../filtering_smoothing.md):

- Filtering: maintain the current belief.
- Optimization: maintain many states and jointly improve them.

BA is the latter. We solve one giant optimization problem:

```mermaid
flowchart TD
    A["Current estimate:<br/>camera poses T_i and landmarks P_j"] --> B["Projection:<br/>predicted pixel from T_i inverse applied to P_j"]
    B --> C["Residual r_ij = observed pixel z_ij - predicted pixel<br/>total error = sum over all observed pairs (i, j)"]
    C --> D["Gauss-Newton step:<br/>solve H delta = g"]
    D --> E["Better poses and better points:<br/>T_i updated with Exp(delta_i), P_j with P_j + delta_j<br/>repeat from the top until the step is tiny"]
```

---

## 9. BA vs Pose Graph Optimization

| | Optimizes | Constraints |
| --- | --- | --- |
| **Pose graph** | $`\{T_0, T_1, \ldots, T_n\}`$ only (landmarks already marginalized or not in the problem) | relative poses |
| **Bundle adjustment** | camera poses + 3D landmarks | image reprojection errors |

In a pose graph the edges are relative-pose constraints between poses, including loop closures. In BA there are no pose-pose edges: cameras are coupled only through the landmarks they observe (§10). For the graph comparison, see [pose_graph_optimization.md §8](pose_graph_optimization.md#8-why-is-this-different-from-bundle-adjustment).

- **Pose graph:** "Make the poses geometrically consistent."
- **Bundle adjustment:** "Make the entire camera + 3D structure explain the images."

---

## 10. Why BA can be computationally expensive

Imagine 1,000 camera poses, 100,000 landmarks and millions of image observations. That is a huge number of variables.

But there is useful structure:

```text
Camera variables ─── Landmark variables
       ↕                    ↕
       └──── observations ──┘
```

- Camera $T_1$ only directly interacts with the landmarks it observes.
- That makes the problem **sparse** ([glossary](../glossary.md#sparsity)), which is one of the fundamental reasons efficient BA algorithms are possible.

---

## 11. The intuition in one picture

Imagine a pile of photographs and a rough miniature 3D reconstruction. We ask: "If this really were the correct 3D world, would these cameras see these landmarks at exactly these pixels?" If not, something is wrong:

- camera 1 is slightly misplaced
- camera 2 is rotated incorrectly
- landmark 1 is too far away
- landmark 2 is too high

So we keep adjusting poses and landmarks, reprojecting and measuring the image error, until the reconstruction is as consistent with the images as possible (the pipeline in §8, repeated).

---

## 12. Block sparsity and the Schur complement

§10 named the sparsity that makes large BA tractable. Here are the mechanics. A residual for point $P_j$ seen by camera $T_i$ depends **only** on $T_i$ and $P_j$ (the diagram in §10).

That gives the linearized normal-equations Hessian $`H = J^\top J`$ an **arrow-head** block structure with three kinds of blocks:

- camera-camera: block-diagonal;
- point-point: block-diagonal;
- camera-point: nonzero only where that camera observes that point.

No two different cameras, and no two different points, are coupled directly.

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

Real scenes usually have far more points than cameras, so solvers use the **Schur complement trick**:

1. [Marginalize](../glossary.md#marginalization) out the point block (cheap, since it is block-diagonal and each point's small block inverts independently).
2. Solve the much smaller reduced camera-only system.
3. Back-substitute to recover the points.

**Tiny example (Schur complement):** a toy with 2 cameras and 1 point, one number each, and no prior.
- Let $`J = \begin{bmatrix}1 & 0 & 1\\ 0 & 1 & 1\end{bmatrix}`$ (columns: camera 1, camera 2, point). The point block is $`1 \cdot 1 + 1 \cdot 1 = 2`$.
- Eliminating the point gives the reduced camera matrix $`\begin{bmatrix}1 & 0\\ 0 & 1\end{bmatrix} - \tfrac{1}{2}\begin{bmatrix}1 & 1\\ 1 & 1\end{bmatrix} = \begin{bmatrix}0.5 & -0.5\\ -0.5 & 0.5\end{bmatrix}`$.
- The new $`-0.5`$ links the two cameras. They never touched each other, but they share a point (covisibility).
- This matrix is singular because the toy has no gauge prior (§6.1), as expected.
- In real BA each point block is only $`3 \times 3`$, so inverting all of them is cheap.

It is the same style of sparsity exploitation that makes [`sparse_cholesky_factorization.md`](sparse_cholesky_factorization.md) tractable at scale.

`bundle_adjustment.py` doesn't need this trick. Its toy scenes are small (8 cameras and 48 landmarks by default), so `run_bundle_adjustment` solves the full dense joint system every iteration. Schur-complement marginalization is what a production solver (COLMAP for SfM; g2o, GTSAM and Ceres as general sparse least-squares libraries) does at real scene sizes, not something this demo implements.

$H = J^\top J$ above is the plain textbook form. The scripts' actual $H$ adds pixel weights, the gauge prior and a small diagonal term in `run_bundle_adjustment` ([§6.1](#61-the-ba-math-concretely)). `run_windowed_gn_lm` in `bundle_adjustment_advanced.py` is unweighted but solves $`(H + \lambda\,\mathrm{diag}(H))\,\delta = g`$ (see [§13.6](#136-in-this-repo)), where $`\lambda`$ is the **LM damping** factor ([glossary](../glossary.md#levenberg-marquardt-lm-and-damping)). None of these changes the sparsity pattern: the prior and damping touch only diagonal blocks, and the weights rescale blocks that already exist.

---

## 13. Local vs. Global Bundle Adjustment (real systems)

Local BA and Global BA minimize the same reprojection objective from §6. Local BA uses only the window's own terms, with the other variables held fixed. The two differ in how much of the problem gets optimized at once, a choice driven by very different system constraints.

| Metric | **Local BA** (e.g. ORB-SLAM) | **Global BA** (e.g. COLMAP) |
| --- | --- | --- |
| System paradigm | Visual SLAM (real-time, online) | **Structure-from-Motion** (SfM: reconstructing 3D structure and camera poses from many photos; offline, batch) |
| Optimization scope | A local window of recent **keyframes** (selected frames kept in the optimization) + **covisible** neighbors (keyframes sharing landmarks) | Every registered camera pose and every 3D point |
| Input | Sequential video with continuous tracking | Unordered photo collections (or long video) |
| Frequency | Continuous, runs on every new keyframe | Periodic (e.g. COLMAP's default: after ~10% growth in registered images or points) or a final pass |
| Outlier handling | Fast local robust cost (Huber) + **chi-square gating** (rejecting observations whose weighted squared error exceeds a chi-square threshold) | Heavy re-triangulation, track merging/filtering |
| Scaling | Bounded per window (set by the window size, not the map) | Grows up to cubically with the number of camera poses (dense reduced system) |


The example systems name where each style is the *workhorse*, not the only one used: ORB-SLAM2 also runs a full BA after a [loop closure](../glossary.md#loop-closure), and COLMAP runs a local BA after every image it registers (§13.4).

### 13.1 Why not re-solve everything every time?

Picture the map as a long chain of cameras linked by shared landmarks.

- A new keyframe's measurements pull directly only on the landmarks it sees, and those landmarks are seen only by nearby keyframes.
- The pull reaches further back only through more shared landmarks, weakening at every hop, so cameras far behind barely move.
- Re-solving the whole map to move a handful of nearby poses wastes almost all of the work, and §10-§12 show how fast that work grows with the number of cameras.

Local BA exploits this. It re-solves only the neighborhood the new keyframe actually affects and holds everything else where it is. That neighborhood stays roughly the same size however big the map gets, so each solve takes bounded, real-time-friendly time.

The price is in "holds everything else where it is." Each window treats the poses on its border as exact, so any small error in them is frozen into the new estimates, and every window adds a little more. Over a long trajectory these small errors compound into **drift**, the same compounding as dead-reckoning drift in [pose_graph_optimization.md §3](pose_graph_optimization.md#3-why-do-we-need-optimization).

### 13.2 Choosing the window: covisibility, not recency

Which keyframes count as "the neighborhood"? The obvious answer, the last $`N`$ keyframes, fails whenever the camera turns around or revisits a place: an old keyframe that sees the same landmarks is more tightly coupled to the new one than a recent keyframe facing the other way.

The right measure of coupling falls out of §12:

- After the Schur complement eliminates the landmarks, the reduced camera system has a nonzero block between cameras $i$ and $j$ exactly when they share at least one landmark.
- The **covisibility graph** (keyframes as nodes, an edge wherever two keyframes share landmarks, weighted by how many) is that sparsity pattern, kept up to date as keyframes arrive.
- Choosing the new keyframe's strongest covisible neighbors means choosing the cameras its measurements are most strongly coupled to. (ORB-SLAM draws an edge at 15 or more shared points.)

```text
  [Fixed Keyframe]  sees -> (Fixed Map Point)
         |                             |
  (Covisible Link)               (Observed by)
         |                             v
 [Active Keyframe] < optimizes > [Active Map Point]
```

![Three panels from bundle_adjustment_advanced.py's defaults: the Local BA window when keyframe 30 arrives, with its new keyframe, five covisible neighbours, seven fixed anchors and five active landmarks; the banded covisibility matrix of shared landmark counts; and the window size per Local BA call staying bounded while the whole map grows](../../assets/bundle_adjustment_advanced_concept.png)

*Figure: windows recorded from `use_numpy/bundle_adjustment_advanced.py`'s own `build_active_window` calls at its defaults (seed 0), plotted by `uv run python assets/make_figures.py bundle_adjustment_advanced_concept`.*

- **Active keyframes:** the new keyframe plus its neighbors in the covisibility graph.
- **Active points:** every 3D point observed by an active keyframe.
- **Fixed keyframes:** other keyframes that also see an active point, held fixed as rigid anchors.


**Fixed, not marginalized.** The fixed keyframes do two jobs:

- They pin the gauge, so the window can't slide, rotate or rescale as a whole (§14). Rescaling is blocked only when at least 2 keyframes are fixed.
- Their observations of the active points add constraints.

But fixing a pose treats it as exactly known, which throws its uncertainty away. Two ways to handle the border:

- **Marginalize it**, keeping what it knew as a prior on the window ([marginalization.md §2](marginalization.md#2-marginalization-keep-the-information-drop-the-variable)), as sliding-window visual-inertial estimators such as OKVIS and VINS-Mono do. That keeps the information, but the prior is dense ([marginalization.md §5](marginalization.md#5-the-fill-in-consequence)) and its linearization point is frozen, which causes the consistency problem in [marginalization.md §6](marginalization.md#6-the-consistency-gotcha-why-fej-exists).
- **Fix it** - ORB-SLAM's cheaper route: accept the drift that causes, and correct the drift when it closes a loop.


### 13.3 What Local BA can't fix, and what can

Drift has two parts, and they need different cures.

- **Inconsistency between windows.** Each window was solved against a slightly different frozen border, so neighboring windows disagree a little about the landmarks they share. The measurements *can* see this: solved jointly, the disagreement shows up as reprojection error. A Global BA pass removes it.
- **Error the measurements can't see.** On a path that never revisits a place, nothing ties the end of the map to its start. The measurements only fix each part of the map relative to its neighbors, so the uncertainty in where the far end is grows with distance from the anchor. Even a perfect joint solve has this drift. A monocular (single-camera) setup adds **scale drift** on top, since images only determine the scene up to a scale factor (Strasdat et al. 2010).

Only new information cures the second part: a loop closure or another sensor, such as GPS, or an IMU, which makes roll, pitch and metric scale observable. So real systems split the work. ORB-SLAM runs Local BA on every keyframe. When it detects a loop, it estimates a similarity transform between the two ends (7 DoF, so scale drift is corrected too) and spreads the correction along the trajectory with pose-graph optimization ([pose_graph_optimization.md](pose_graph_optimization.md)). ORB-SLAM2 then runs a full BA in a separate thread to refine the result.


### 13.4 Global BA in practice (COLMAP)

Offline SfM pipelines sacrifice real-time speed for maximum accuracy. COLMAP, for example:
- runs a local BA around each newly registered image as it incrementally registers new images;
- once the model has grown by a set percentage, re-optimizes **every** camera and **every** point jointly in one large least-squares problem;
- then uses the resulting global residuals to prune bad matches and re-triangulate points - something a local window can never do, since it never sees the whole map at once.

The cost: even after the Schur complement from §12, the reduced camera-only system grows up to cubically with the number of camera poses. Assembling that Schur complement also costs roughly linear time in the number of points and observations. So Global BA can only run periodically or as a final step, not every frame.


### 13.5 Choosing between them

Use **Local BA** for real-time robotics/AR/VR (augmented/virtual reality) where low, predictable latency matters more than perfect global consistency (loop closure repairs that later).

Use **Global BA** for offline reconstruction - meshes, NeRF/Gaussian-Splatting input scenes (learned 3D scene models), photogrammetric surveys - where total geometric fidelity matters more than runtime.

### 13.6 In this repo
`bundle_adjustment.py`'s three solvers (`run_ba_landmarks_only`, `run_ba_poses_only`, `run_bundle_adjustment`) are single-batch joint solves over the whole toy scene, closest in spirit to a (tiny) Global BA pass. They have no windowing, no covisibility graph and no incremental registration, so they don't model Local BA's real-time behavior at all.

`bundle_adjustment_advanced.py` fills that gap. A camera moves keyframe-by-keyframe through a landmark corridor while a covisibility graph is built incrementally:

- **Local BA:** every new keyframe triggers a bounded solve over an active window (the window of §13.2, made concrete)
- **Global BA:** a periodic pass over the whole map runs alongside for comparison.

![Three panels from bundle_adjustment_advanced.py: the keyframe path through a landmark corridor with the Local-only and Local plus Global estimates, solve time per call for the bounded Local BA window versus Global BA, and trailing trajectory RMSE over keyframes for both modes](../../assets/bundle_adjustment_advanced.png)

*Figure: `use_numpy/bundle_adjustment_advanced.py` at its defaults (seed 0), plotted by `uv run python assets/make_figures.py bundle_adjustment_advanced`.*


**What it confirms:** the "Scaling" row above. In the default run (one BLAS thread), the measured wall-clock solve time of the local window stays bounded (8-470 ms, set by how many landmarks the window holds, with no trend in map size), while Global BA's grows from 0.36 s at keyframe 8 to about 1.4-1.9 s at keyframes 40-49.

**What it doesn't confirm:** a reliable accuracy payoff from Global BA on this small scene.

Four sweeps compare Local+Global with Local-only (final trajectory RMSE, the script's own metric):

| Sweep | Scene | Global BA wins | Local+Global minus Local-only |
| --- | --- | --- | --- |
| Regression test, 15 seeds | 12 keyframes, 10 landmarks per keyframe, 8 m view range, `min_observations` = 2, Global pass every 4 keyframes | 6/15 | median +0.06 m |
| Script defaults, 20 seeds | the defaults listed at the end of this section | 11/20 | median −0.02 m |
| Script defaults, seed 0 only | the defaults | loss | 2.1359 m → 2.7124 m (+27%) |
| Loop closure, seeds 0-4 | `--arc-span-deg 350 --n-keyframes 190` | 1/5 (one single-thread run) | varies widely, often much worse |

- **Open path (default):** the default seed happens to look like a loss for Global BA, and no single seed is representative. The 15-seed sweep is the script's regression test. The 20-seed sweep is a wash either way, plus an occasional seed that diverges to thousands of meters in *either* mode from a rare bad local minimum in the windowed GN/LM solve.
- **Loop closure:** the path swings back within view of its start, and the script reports `Loop closure detected at keyframe K ...`.
  - No code changes are needed, because the covisibility bookkeeping, window builder and Global BA solve don't assume temporal locality. The extra keyframes only keep the default run's ~0.5 m spacing (at 50 keyframes it grows to ~1.9 m, and too few landmarks get triangulated for the closure to register).
  - Over seeds 0-4 the closure is detected on all 5 paths (keyframes 183-186). The Local-only final error on the default seed is 13.46 m.
  - The Local+Global result on this long run is unstable. In our re-runs it varied widely across seeds (better than Local-only on only 1 of 5 seeds in one single-thread run, often much worse) and even changed with the **BLAS** (the linear-algebra library NumPy calls) thread count.
  - So this scenario doesn't show that a Global pass reliably helps after a loop closure.

So judge Global BA's benefit here from the aggregate statistics, not from any single run's printed numbers.

**Why Global BA rarely helps on the open path.** §13.3 predicts the answer: most of the error is the kind Global BA can't see. A one-off check by script call over seeds 0-4 at the script's defaults (not part of the test suite) confirms this, and finds a second, script-specific error source. Each row aligns the Local-only trajectory to ground truth with a single similarity transform (Umeyama, §14):

| Seed | Local-only final error (m) | RMSE raw → aligned (m) | Removed by alignment | Fitted scale | Rotation (°) | Final error, keyframe 1 at true pose (m) | Fitted scale, keyframe 1 at true pose |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.1359 | 1.272 → 0.441 | 65% | 1.083 | 4.8 | 4.1046 | 1.254 |
| 1 | 3.4752 | 2.212 → 0.066 | 97% | 0.846 | 1.7 | 0.2080 | 0.989 |
| 2 | 2.9208 | 1.903 → 0.064 | 97% | 0.900 | 7.3 | 1.3336 | 0.930 |
| 3 | 1.4611 | 0.920 → 0.489 | 47% | 0.948 | 0.9 | 8.8921 (mirrored local minimum) | 0.982 |
| 4 | 3.4821 | 2.076 → 0.921 | 56% | 0.814 | 5.9 | 0.8873 | 0.981 |

*Source: one-off check by script call, not part of the test suite. Final errors are the script's own metric, RMSE over the last 10 keyframes. The alignment columns use keyframes 2-49 on both sides. In the last two columns only keyframe 1's initial pose is replaced by the true one; the rest of the front-end chain is unchanged.*

- **Most of the final error is one transform of the whole map.** Alignment removes 47-97% of the Local-only position RMSE over keyframes 2-49 (default seed: 1.27 m → 0.44 m). What remains, the inconsistency Global BA can actually fix, is small.
- **That transform has two parts:**
  - **Orientation drift** of up to ~7° (0.9-7.3°), which stays even with a perfect anchor: §13.3's unobservable drift.
  - **Scale error.** The anchor pair (keyframes 0 and 1, §14) sets the map's scale, and keyframe 1 keeps its noisy front-end pose. Its errors (with keyframe 1's orientation error and window-to-window scale drift) contribute to the scale error. The Local-only fitted scale is off by up to ~19% (0.81-1.08).
  - With keyframe 1 at its true pose, the scale is within 3% of correct on three of the five seeds (the default seed stays at 1.25), and the median final Local-only error falls from 2.92 m to 1.33 m. The mean doesn't improve, because seed 3 falls into a mirrored local minimum (a 171° orientation offset) once the anchor changes.
- **Global BA can make the scale worse.** On seed 2 with the noisy anchor, Global BA pulls the rest of the map into agreement with the wrong baseline: the fitted scale goes from 0.90 to 0.67 and the final error from 2.92 m to 9.55 m. With keyframe 1 at its true pose, the same seed improves instead (1.33 m → 0.70 m).

The anchor choice mirrors real monocular SLAM, where the first two keyframes' baseline sets the map's otherwise arbitrary scale. That's why monocular results are normally scored after a similarity alignment (§14), and why this script's raw, unaligned error mostly measures drift and gauge rather than what BA itself can fix.

**The windowed solver** (`run_windowed_gn_lm`). Local and Global BA call the same solver. They differ only in which poses and points are unknowns:

- **Local** (`build_active_window`, `run_local_ba_step`): the window of §13.2, built from the new keyframe $k$ plus up to `max_window_keyframes` − 1 covisible neighbors (keyframes sharing at least `min_shared_for_covisibility` landmarks with $k$, strongest first). Only already-triangulated landmarks can be active.
- **Global** (`run_global_ba`): every keyframe processed so far and every triangulated landmark are unknowns.
- **Gauge:** no prior factor. Keyframes 0 and 1 (`anchor_keyframes`) are hard-fixed in both solves (keyframe 0 at its true pose, keyframe 1 at its front-end estimate; see §14).

It uses the residual and Jacobians from [§6.1](#61-the-ba-math-concretely), unweighted ($\Omega = I_2$). An observation from a fixed keyframe adds only its $J_{\text{point}}$ block. Each iteration solves a Levenberg-Marquardt system:

```math
\left(H + \lambda\,\mathrm{diag}(H)\right)\delta = g, \qquad \text{cost} = \sum \|z - \pi(T^{-1}P)\|^2
```

Diagonal entries of $H$ below $10^{-12}$ are floored to $10^{-12}$ before scaling. $\lambda$ starts at $10^{-3}$ in every call. A trial step is accepted only if it lowers the cost, and then $\lambda \leftarrow \max(\lambda/2, 10^{-7})$. Otherwise $\lambda \leftarrow 2\lambda$ and the solve is retried, up to 10 times. The loop stops when no step is accepted after 10 retries, when the step norm drops below `gn_tol`, or after `gn_max_iters` iterations.

**Tiny example (cheirality):** **cheirality** is the requirement that a point lie in front of the camera (positive depth). The pinhole $`u = f_x\,x/z + c_x`$ flips sign with $`z`$, so cheirality matters. Take $`f_x = 800`$, $`c_x = 320`$.
- A point in front, $`P_c = (-1, 0, 5)`$, gives $`u = 160`$.
- The point $`(1, 0, -5)`$ behind the camera gives the same $`u = 160`$.
- So a point behind the camera can look like a valid mirrored pixel. The solver must reject it explicitly.

Two cheirality rules protect the solve, using `point_depth` (the depth $`p_{c,z}`$, the third entry of $`R^\top(P - t)`$). First, any observation whose point is already behind its camera when the call starts is dropped for this call (gating). Second, a trial that puts any remaining point behind its camera gets infinite cost, so it is rejected. New landmarks are triangulated once they have `min_observations` observers (`triangulate_landmark`, then `refine_landmark_gn`, see [triangulation_pnp.md](../frontend/triangulation_pnp.md)). They are committed only if `passes_cheirality` holds. After each local and global solve, `cull_invalid_points` deletes any landmark that is now behind one of its observers, so it can be triangulated again later.

Script defaults: 50 keyframes on a 90° arc of radius 15 m, 8 landmarks per keyframe, 6 m view range, `min_observations` = 3, `min_shared_for_covisibility` = 2, `max_window_keyframes` = 6, a Global BA pass every 8 keyframes (plus one at the end), `gn_tol` = $10^{-6}$, `gn_max_iters` = 15, per-step front-end twist noise std 0.02 (6-vector, mixed m and rad), $\sigma_{\text{px}} = 1$ px.

---

## 14. Evaluating the result: gauge freedom and Umeyama alignment

Monocular BA recovers the scene only up to an unknown similarity transform (rigid + scale). Shifting, rotating or uniformly rescaling the whole scene and every camera pose together leaves reprojection error unchanged.

### Tiny example (scale)
Use $`f_x = 800`$ and $`c_x = 320`$.
- A point at $`(1, 0, 5)`$ gives $`u = 800 \cdot 1/5 + 320 = 480`$ px.
- Double everything to $`(2, 0, 10)`$ and $`u = 800 \cdot 2/10 + 320 = 480`$ px. Same pixel, so the image cannot tell the two scenes apart.
- Fixing the distance between two cameras gives the scene a ruler. That is why both BA scripts pin two cameras.

- **`bundle_adjustment.py`** pins that freedom with a soft gauge-prior factor on the first two camera poses, but the resulting frame still won't match ground truth's frame or scale exactly. So before computing pose/landmark error, the script fits one scale + rotation + translation by Umeyama on the camera positions only, then applies it to the cameras and the landmarks. See [umeyama_alignment.md](../foundations/umeyama_alignment.md) for how the alignment is computed and why it's needed. Reprojection error is unaffected by the alignment.
- **`bundle_adjustment_advanced.py`** hard-fixes two anchor keyframes instead of using a soft prior, which removes all gauge freedom up front, and it reports raw, unaligned error. The catch is *where* the gauge gets pinned: keyframe 0 sits at its true pose, but keyframe 1 keeps its noisy front-end pose, so the scale and orientation it fixes are wrong, and that error is part of what the script reports. [§13.6](#136-in-this-repo) measures it: up to ~19% off in scale over seeds 0-4.

---

## 15. Summary

> **Bundle Adjustment is the process of jointly refining camera poses and 3D landmarks so that their projections agree as closely as possible with the observed image features.**

The conceptual hierarchy:

```text
SLAM
 │
 ├── Filtering
 │     └── EKF-SLAM
 │
 └── Optimization/Smoothing
       │
       ├── Pose Graph Optimization
       │
       └── Bundle Adjustment
              │
              ├── optimize camera poses
              └── optimize 3D landmarks
```

For **visual SLAM**, BA is the workhorse behind *"make my entire reconstructed 3D world and camera trajectory agree with all the pixels/features I've observed."*

---

## 16. References

1. Triggs, B., McLauchlan, P. F., Hartley, R. I., & Fitzgibbon, A. W. (2000). *Bundle Adjustment - A Modern Synthesis*. In B. Triggs, A. Zisserman, & R. Szeliski (Eds.), Vision Algorithms: Theory and Practice (Vol. 1883, pp. 298-372). Springer. https://doi.org/10.1007/3-540-44480-7_21 - the standard reference survey behind this whole doc's framing (joint pose+landmark refinement, the Schur complement in §12, and the "modern synthesis" of BA as a sparse nonlinear least-squares problem rather than a purely photogrammetric one).
2. Mur-Artal, R., Montiel, J. M. M., & Tardós, J. D. (2015). *ORB-SLAM: A Versatile and Accurate Monocular SLAM System*. IEEE Transactions on Robotics, 31(5), 1147-1163. https://doi.org/10.1109/TRO.2015.2463671 - the Local BA window (covisible keyframes active, other observers fixed), the covisibility graph, and Sim(3) loop closure with pose-graph optimization in §13.
3. Mur-Artal, R., & Tardós, J. D. (2017). *ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo, and RGB-D Cameras*. IEEE Transactions on Robotics, 33(5), 1255-1262. https://doi.org/10.1109/TRO.2017.2705103 - the full BA run in a separate thread after loop closure, in §13.3.
4. Schönberger, J. L., & Frahm, J.-M. (2016). *Structure-from-Motion Revisited*. CVPR 2016, 4104-4113. https://doi.org/10.1109/CVPR.2016.445 - COLMAP's local BA after each registration and global BA after model growth, with re-triangulation and filtering, in §13.4.
5. Strasdat, H., Montiel, J. M. M., & Davison, A. J. (2010). *Scale Drift-Aware Large Scale Monocular SLAM*. Robotics: Science and Systems VI. https://doi.org/10.15607/RSS.2010.VI.010 - monocular scale drift and why loop closure must correct a 7-DoF similarity, in §13.3.
6. Leutenegger, S., Lynen, S., Bosse, M., Siegwart, R., & Furgale, P. (2015). *Keyframe-based visual-inertial odometry using nonlinear optimization*. The International Journal of Robotics Research, 34(3), 314-334. https://doi.org/10.1177/0278364914554813 - OKVIS, a keyframe window that marginalizes its border instead of fixing it, in §13.2.
7. Qin, T., Li, P., & Shen, S. (2018). *VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator*. IEEE Transactions on Robotics, 34(4), 1004-1020. https://doi.org/10.1109/TRO.2018.2853729 - the other marginalizing sliding-window estimator in §13.2, already cited in [marginalization.md §11](marginalization.md#11-references).
