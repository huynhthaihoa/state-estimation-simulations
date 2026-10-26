# Triangulation and PnP: two sides of one geometric problem

- **Triangulation:** given known camera poses and a 2D observation in each, where is the 3D point?
- **PnP:** given several known 3D points and their observed pixels, where is the camera?
- **Why PnP needs several points:** one point/pixel pair (a **correspondence**) pins down a ray, not a unique pose.
  - **P3P** (PnP with exactly 3 correspondences) gives up to four candidate poses, so a fourth point picks the right one.
  - Our own DLT (Direct Linear Transform, §3) implementation needs at least 6 (derived in §3).
- **Same recipe for both:** both reduce to the same reprojection residual. We solve each with a **closed-form** (direct formula, no iteration) linear initial guess, then a few [Gauss-Newton](../glossary.md#gauss-newton) iterations.

This builds directly on:
- The **[reprojection error](../glossary.md#reprojection-error)** and **"bundle of rays"** intuition from [bundle_adjustment.md §5](../optimization/bundle_adjustment.md#5-why-is-it-called-bundle-adjustment).
- **Gauss-Newton** from [gauss_newton.md](../optimization/gauss_newton.md).

---

## 1. The two problems, stated symmetrically

| | Triangulation | PnP (Perspective-n-Point) |
|---|---|---|
| Known | Camera poses, 2D observations | A 3D point (or several), 2D observations |
| Unknown | One 3D point | One camera pose (6 DOF) |
| Residual | Reprojection error (§2) | Reprojection error (§3) |
| Implemented in this repo | `bundle_adjustment_advanced.py`'s `triangulate_landmark`/`refine_landmark_gn` | `pnp_estimation.py`'s `linear_pnp_dlt`/`refine_pose_gn` |

Both are front-end problems in the [front-end/back-end split](../frontend_backend.md). The front-end turns "I see this pixel" into one of two things:

- a 3D map point (triangulation, to grow the map);
- a camera pose (PnP, to localize against an already-known map).

Only then can back-end optimization ([bundle_adjustment.md](../optimization/bundle_adjustment.md), [pose_graph_optimization.md](../optimization/pose_graph_optimization.md)) refine it further.

![Two panels: triangulation, where rays from eight known cameras through their measured pixels meet at one unknown point, and PnP, where rays from 20 known 3D points converge on the unknown camera centre recovered by the linear DLT and Gauss-Newton](../../assets/pnp_estimation_concept.png)

*Figure: (a) `triangulate_landmark`/`refine_landmark_gn` from `use_numpy/bundle_adjustment_advanced.py`, run on `use_numpy/bundle_adjustment.py`'s wide-baseline scene (Gauss-Newton minimizes reprojection (pixel) error, not distance to the true point, so on one noisy point it can land slightly farther from the truth than the closed form); (b) `use_numpy/pnp_estimation.py` at its defaults (seed 0), plotted by `uv run python generator/make_figures.py pnp_estimation_concept`.*

---

## 2. Triangulation, worked from the real code

`triangulate_landmark` in `bundle_adjustment_advanced.py` solves triangulation in closed form. It is the "closest point to N lines" problem:

- Each observing camera gives a calibrated ray: direction $d$ from the camera center through the observed pixel, converted to a direction with the camera intrinsics (defined below).
- The best point minimizes the sum of squared perpendicular distances to every ray.
- For one ray (unit $d$, through camera center $o$), the projector $M = I - dd^\top$ measures that perpendicular distance.
- Summing $M$ and $Mo$ over all observers and solving the resulting $3\times3$ linear system gives the point directly, with no iteration.
- With noise the rays are skew (they do not meet), so the closed-form point is the least-squares compromise (figure (a) in §1 draws the geometry).

Then we polish and validate:

- **Polish.** `refine_landmark_gn` runs a few Gauss-Newton iterations on the *true* nonlinear reprojection error, seeded from the closed-form guess (the same "linear guess, then GN polish" pattern used throughout this repo).
- **Validate.** `passes_cheirality` checks the **cheirality** condition: the point has positive depth (its $`z`$ coordinate in the camera frame) in every observing camera.
  - The reason: `pixel = f*x/z` is invariant to negating a camera-frame point's $x,y,z$ together.
  - A weakly-constrained point has few observers and little **parallax** (the angle between the rays from different cameras to the point).
  - For such a point, the Gauss-Newton step can walk to a *reflection* behind a camera that still fits every pixel almost exactly (see the `refine_landmark_gn` and `passes_cheirality` docstrings).
- **Reject.** A point failing the check is never committed to the map (see the [`bundle_adjustment_advanced.py` README description](../../README.md#7-local--global-bundle-adjustment-bounded-windows-vs-whole-map-re-solves)). It stays pending until another observation gives it more parallax.

**The triangulation math, concretely.** Poses are camera-to-world: observer $k$ has rotation $R_k$ and camera center $o_k$ (its translation). `triangulate_landmark` rotates each pixel's calibrated ray into the world frame and normalizes it:

```math
\tilde d_k = \begin{bmatrix} (u_k - c_x)/f_x \\ (v_k - c_y)/f_y \\ 1 \end{bmatrix}, \qquad
d_k = \frac{R_k \tilde d_k}{\left\| R_k \tilde d_k \right\|}
```

Here $`\tilde d_k`$ is the unnormalized camera-frame ray (its $`z`$ component is 1), $`d_k`$ is the same ray rotated into the world frame and scaled to unit length, $`u_k, v_k`$ are the pixel coordinates of the observation in camera $`k`$, $`(c_x, c_y)`$ is the principal point (the pixel where the optical axis meets the image) and $`(f_x, f_y)`$ are the focal lengths in pixels (the **intrinsics**).

In the next block, $`A`$ and $`b`$ are the normal-equation matrix and vector of the closest-point problem. The linear system $`AP = b`$ has the point that minimizes the summed squared distances to the rays as its solution. $`P_0`$ is the initial point estimate. This $`A`$ is not the DLT matrix of §3.

$$
A = \sum_k \left(I - d_k d_k^\top\right), \qquad
b = \sum_k \left(I - d_k d_k^\top\right) o_k, \qquad
P_0 = \left(A + 10^{-9} I\right)^{-1} b
$$

- **Conditioning** (how strongly noise is amplified into the answer). The depth error along the rays grows as the baseline (the distance between camera centers) shrinks, because a shorter baseline means less parallax.
- **Tiny example** (toy case: two rectified cameras (parallel optical axes, offset sideways), baseline $`B`$, focal length $`f`$, depth $`Z`$, pixel noise $`\sigma`$). Depth error is about $`Z^2\sigma/(fB)`$. With $`f=800`$ px, $`Z=10`$ m, $`\sigma=1`$ px: $`B=1`$ m gives 0.125 m of error, and $`B=0.1`$ m gives 1.25 m. A 10 times smaller baseline means a 10 times larger depth error.
- **The $10^{-9} I$ term.** It keeps the solve defined when all rays are parallel. The depth along the rays is then arbitrary (the term pins the point's coordinate along the ray direction to 0, measured from the world origin).

`refine_landmark_gn` then runs Gauss-Newton on the point alone (here $`P`$ is the 3D point being refined, $`\delta`$ is its step, $`z_k`$ is the observed pixel and $`\pi`$ is the **pinhole** projection (a camera model where every ray passes through the single camera center) that maps a camera-frame point $`(x,y,z)`$ to the pixel $`(f_x x/z + c_x,\; f_y y/z + c_y)`$; $`\delta`$ is a different step in the PnP refinement of §3), with the [Jacobian](../glossary.md#jacobian) $J_{\text{point}}$ (the $`2\times3`$ derivative of the predicted pixel with respect to the point) from [bundle_adjustment.md §6.1, "Projection and Jacobians"](../optimization/bundle_adjustment.md#projection-and-jacobians):

$$
\left(\sum_k J_{\text{point},k}^\top J_{\text{point},k} + 10^{-9} I\right)\delta = \sum_k J_{\text{point},k}^\top r_k, \qquad
r_k = z_k - \pi\left(R_k^\top (P - o_k)\right), \qquad
P \leftarrow P + \delta
$$

The residual is measured minus predicted, so the right-hand side is $`+J^\top r`$: the sign convention of [bundle_adjustment.md §6.1](../optimization/bundle_adjustment.md#gauss-newton-step-and-solvers), opposite to the glossary's $`H\Delta x = -b`$ ([glossary](../glossary.md#normal-equations-and-hessian)).

It stops when the step norm drops below `gn_tol`, or earlier if the point's depth becomes non-positive in any observer. `passes_cheirality` then requires a strictly positive depth in every observer, using `point_depth`:

$$
\left[R_k^\top (P - o_k)\right]_z > 0 \quad \text{for every observer } k
$$

In `bundle_adjustment_advanced.py`, a landmark is triangulated once it has `min_observations` = 3 observers (the default), with that script's `gn_tol` = $10^{-6}$ and `gn_max_iters` = 15.

---

## 3. PnP as the inverse problem

**Pose convention.** As in §2, poses are camera-to-world ($T_{wc}$): a camera with rotation $R_k$ and center $o_k$ has world-to-camera pose $R_{cw} = R_k^\top$, $t_{cw} = -R_k^\top o_k$. This section's unknown $(R_{cw}, t_{cw})$ is that world-to-camera form, and `refine_pose_gn`'s $T$ is $T_{wc}$.

`pnp_estimation.py` solves the exact mirror image. Given $n$ known 3D points $P_i$ and their observed pixels, first form the calibrated ray for each (the unnormalized camera-frame ray, §2's $`\tilde d`$): $d_i = \left[\frac{u_i-c_x}{f_x}, \frac{v_i-c_y}{f_y}, 1\right]$. Because $d_i$ is parallel to the camera-frame point $R_{cw}P_i + t_{cw}$ ($R_{cw}, t_{cw}$ the unknown world-to-camera pose), their cross product must vanish:

$$d_i \times (R_{cw}P_i + t_{cw}) = 0$$

**Intuition:** count the unknowns.
- The 12 entries of $`[R_{cw} \mid t_{cw}]`$ are only defined up to scale, so there are 11 unknowns.
- Each point gives 2 equations. 5 points give 10, which is less than 11. 6 points give 12, which is enough.

This is **linear and homogeneous** (right-hand side zero) in the 12 flattened entries of $[R_{cw} \mid t_{cw}]$ - exactly the classical **Direct Linear Transform (DLT)** camera-resectioning (recovering a camera's pose from known 3D points) setup, specialized to *known* intrinsics. Stacking two independent rows of this constraint per correspondence gives an overdetermined homogeneous system $Ax=0$ ($`x`$ = those 12 entries, spelled out below; here $`A`$ is the DLT matrix, not the matrix $`A`$ of §2), solved via `linear_pnp_dlt` as the smallest right-singular vector of $A$, taken from its **singular value decomposition (SVD)** (`np.linalg.svd`).

**Intuition:** the smallest singular vector is the direction that $A$ squashes the most. Toy case: $`A = \mathrm{diag}(3, 0.1)`$ stretches $`(1,0)`$ to length 3 but shrinks $`(0,1)`$ to 0.1, so $`(0,1)`$ is the best answer to $`Ax \approx 0`$.

> **Note:** SVD (Singular Value Decomposition) factors any matrix as $A=U\Sigma V^\top$, with $U, V$ orthogonal and $\Sigma$ diagonal (the singular values, ranking how much each orthogonal direction contributes to $A$).
> - The smallest right-singular vector of $A$ (the column of $V$ paired with the smallest singular value) is the least-squares null-space solution (the unit $`x`$ that makes $`\|Ax\|`$ smallest) `linear_pnp_dlt` uses above.
> - Taking $UV^\top$ from a matrix's own SVD gives the nearest orthogonal matrix (a rotation once $\det > 0$, handled below). The orthogonalization step below uses this.

Concretely, each correspondence adds two rows to $A$: the first two rows of the cross-product matrix $`[d_i]_\times`$, times a $`3\times12`$ matrix that maps $x$ to $R_{cw}P_i + t_{cw}$. Here $r_1, r_2, r_3$ are the rows of $R_{cw}$:

```math
A_i = \left([d_i]_\times\right)_{\text{rows }1,2}
\begin{bmatrix} P_i^\top & 0 & 0 & 1 & 0 & 0 \\ 0 & P_i^\top & 0 & 0 & 1 & 0 \\ 0 & 0 & P_i^\top & 0 & 0 & 1 \end{bmatrix}, \qquad
x = \begin{bmatrix} r_1 & r_2 & r_3 & t_{cw}^\top \end{bmatrix}^\top
```

The third row is dropped because it is a combination of the first two (the third entry of $d_i$ is always 1). $A$ is $2n\times12$ and needs rank 11 for a one-dimensional null space. This sets the minimum point count derived above.

The recovered $3\times3$ block is only a *scaled, possibly reflected* rotation - not yet a valid element of $SO(3)$ ([glossary](../glossary.md#so3-and-se3): the group of 3D rotations). `linear_pnp_dlt` turns $x$ into a pose in this order:

1. **Unpack.** $x$ is the last row of $V^\top$. Its first 9 entries, row-major, give $R_{\text{raw}}$. The last 3 give $t_{\text{raw}}$.
2. **Fix the sign.** $x$ and $-x$ both solve $Ax = 0$. If the median over all points of the raw depth $`(R_{\text{raw}}P_i + t_{\text{raw}})_z`$ is negative, both $`R_{\text{raw}}`$ and $`t_{\text{raw}}`$ are negated (see §4). This runs *before* orthogonalization.
3. **Orthogonalize.** With $R_{\text{raw}} = U\Sigma V^\top$, set $R_{cw} = UV^\top$. If $\det(R_{cw}) < 0$, the last row of $V^\top$ is negated and $R_{cw}$ is recomputed, so the result is a proper rotation.
4. **Recover scale.** $s = (\sigma_1 + \sigma_2 + \sigma_3)/3$, the mean of the singular values of $R_{\text{raw}}$. Then $t_{cw} = t_{\text{raw}}/s$.
5. **Invert.** The function returns the camera-to-world pose, with rotation $R_{cw}^\top$ and translation $-R_{cw}^\top t_{cw}$.

With noise-free pixels, this recovers the true pose to floating-point precision.

`refine_pose_gn` then runs Gauss-Newton on the true reprojection error. It follows `refine_landmark_gn`'s loop (minus its early exit on non-positive depth) and solves a $6\times6$ system for the pose instead of a $3\times3$ system for the point. Here $`T`$ is the camera-to-world pose $`T_{wc}`$, $`z_i`$ the observed pixel of point $`P_i`$ and $`\pi`$ the same projection as above. The pose is updated by a right-multiplicative correction $T \leftarrow T\cdot\mathrm{Exp}(\delta)$ ([Exp](../glossary.md#exp-and-log-maps) turns a small 6-vector into a pose). It uses the same Jacobian $J_{\text{pose}}$ as bundle adjustment (the $`2\times6`$ derivative of the predicted pixel with respect to the pose's right perturbation; [bundle_adjustment.md §6.1, "Projection and Jacobians"](../optimization/bundle_adjustment.md#projection-and-jacobians)), for the pose step $\delta = [\delta v, \delta\omega]$ (not the point step $\delta$ of §2). Here $`\delta v`$ is the translation part and $`\delta\omega`$ the rotation part of the pose correction, in that order:

$$
\left(\sum_i J_{\text{pose},i}^\top J_{\text{pose},i} + 10^{-9} I_6\right)\delta = \sum_i J_{\text{pose},i}^\top r_i, \qquad
r_i = z_i - \pi\left(T^{-1} P_i\right)
$$

The residual and right-hand side use the same sign convention as §2. It stops when the step norm drops below `gn_tol`. Script defaults: 20 points, sampled at camera-frame depth 3 to 8 m, $f_x = f_y = 800$ px on a 640×480 image, pixel noise $\sigma = 1$ px, `gn_tol` = $10^{-8}$, at most 20 iterations.

---

## 4. The shared reflection/cheirality risk

A pinhole projection is invariant to certain sign flips, and a linear solve sees only the projection's algebraic structure, not "in front of the camera". So both problems can return a geometrically nonsensical but numerically consistent answer. The remedy is the same positive-depth reasoning in both:

- **Triangulation:** `passes_cheirality` (§2) rejects a point unless its depth is positive in *every* observer. It guards the reflection that Gauss-Newton can produce.
- **PnP:** the sign fix in step 2 of §3 (`np.median(depths_raw) < 0.0`) is *analogous*, not identical. It guards a global $x$ versus $-x$ ambiguity of the DLT null vector, and it uses the *median* of the raw depths to choose a sign, where `passes_cheirality` demands all depths positive.

With noise, neither linear step is accurate enough alone; the Gauss-Newton refinement and the depth checks both matter.

---

## 5. What the accompanying scripts do and don't do

`linear_pnp_dlt` is the simplest *correct* version of linear PnP, not a production algorithm. It needs the minimum number of correspondences from §3 for $A$ to reach rank 11. Its limits are about accuracy, not cost:

- **Cost is not the issue.** A version linear in $n$ would take the SVD of the fixed $12\times12$ matrix $A^\top A$ (or use `full_matrices=False`). Our code calls `np.linalg.svd(A)` on the full $2n\times12$ matrix $A$ with NumPy's default `full_matrices=True`, which also builds a $2n\times2n$ factor $U$: fine at these sizes, but quadratic in $n$ in time and memory.
- **Accuracy is.** It solves for 12 unconstrained numbers and minimizes an **algebraic error** (the size of $`Ax`$, which is easy to minimize, rather than the pixel error), ignoring that 9 of them must form a rotation until the orthogonalization step afterwards. More points help, but with few points, noticeable noise, or points close to a plane it is less accurate than purpose-built solvers.
- **Exactly coplanar points break it.** $A$ then loses rank and the null space is no longer one-dimensional; planar scenes need a **homography**-based method (a homography is the projective map between a plane and its image) instead.

**EPnP** (Lepetit, Moreno-Noguer & Fua, 2009 - see References) expresses every 3D point as a weighted combination of four virtual control points, so the problem becomes recovering just those four points' camera-frame coordinates. The $`O(n)`$ in its title is relative to earlier non-iterative PnP methods (the paper cites $`O(n^5)`$ and $`O(n^8)`$), not to the DLT. Its advantage over the DLT is accuracy.

In practice, PnP commonly runs inside **RANSAC** (random sample consensus: fit to small random subsets, keep the fit most matches agree with), because some 2D-3D matches are wrong:

- A minimal solver (P3P, three points) generates candidate poses from random subsets.
- The pose with the most **inliers** (matches consistent with it) wins.
- A non-minimal solver and/or nonlinear refinement then uses all inliers.
- We implement the simpler DLT without RANSAC (our synthetic correspondences have no outliers), matching `triangulate_landmark`'s choice of a simple closed-form solve.

![Two panels from pnp_estimation.py: the image plane with observed pixels and 20-times-magnified reprojection residuals for the linear DLT pose and the Gauss-Newton-refined pose, and bars of rotation, position and reprojection error for both](../../assets/pnp_estimation.png)

*Figure: `use_numpy/pnp_estimation.py` at its defaults (seed 0), plotted by `uv run python generator/make_figures.py pnp_estimation`.*

---

## 6. Summary

> **Triangulation and PnP are the same reprojection problem run in opposite directions (poses fixed, solve a point; points fixed, solve a pose). We solve both with a closed-form linear guess followed by Gauss-Newton, guarded by positive-depth checks.**

---

## 7. References

1. Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. - the standard reference for DLT camera resectioning behind §3's derivation.
2. Lepetit, V., Moreno-Noguer, F., & Fua, P. (2009). *EPnP: An Accurate O(n) Solution to the PnP Problem*. International Journal of Computer Vision, 81(2), 155-166. https://doi.org/10.1007/s11263-008-0152-6 - the more accurate linear-time PnP algorithm contrasted with the DLT in §5.
3. Triggs, B., McLauchlan, P. F., Hartley, R. I., & Fitzgibbon, A. W. (2000). *Bundle Adjustment - A Modern Synthesis*. In Vision Algorithms: Theory and Practice (pp. 298-372). Springer. - already cited in [bundle_adjustment.md](../optimization/bundle_adjustment.md), covering the triangulation-within-BA context behind §2.
