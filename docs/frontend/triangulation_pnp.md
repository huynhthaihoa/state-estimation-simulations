# Triangulation and PnP: two sides of one geometric problem

Triangulation asks "given known camera poses and a 2D observation in each, where is the 3D point?" PnP asks the exact inverse: "given several known 3D points and their observed 2D pixels, where is the camera?" (a single point/pixel pair pins down a ray, not a unique pose - PnP needs at least 3 correspondences (P3P), and this repo's own DLT implementation needs at least 6, per §5 below). Both reduce to the same reprojection residual, solved the same way in this repo - a closed-form linear initial guess, then a few Gauss-Newton iterations.

This builds directly on:
- The **reprojection error** and **"bundle of rays"** intuition from [bundle_adjustment.md §5](../optimization/bundle_adjustment.md#5-why-is-it-called-bundle-adjustment).
- **Gauss-Newton** from [gauss_newton.md](../optimization/gauss_newton.md).

---

## 1. The two problems, stated symmetrically

| | Triangulation | PnP (Perspective-n-Point) |
|---|---|---|
| Known | Camera poses, 2D observations | A 3D point (or several), 2D observations |
| Unknown | One 3D point | One camera pose (6 DOF) |
| Residual | Reprojection error (§2) | Reprojection error (§3) |
| Implemented in this repo | `bundle_adjustment_advanced.py`'s `triangulate_landmark`/`refine_landmark_gn` | `pnp_estimation.py`'s `linear_pnp_dlt`/`refine_pose_gn` |

Both are front-end problems in the [front-end/back-end split](../frontend_backend.md): a front-end needs to turn "I see this pixel" into either a 3D map point (triangulation, to grow the map) or a camera pose (PnP, to localize against an already-known map) *before* any back-end optimization ([bundle_adjustment.md](../optimization/bundle_adjustment.md), [pose_graph_optimization.md](../optimization/pose_graph_optimization.md)) can refine it further.

---

## 2. Triangulation, worked from the real code

`bundle_adjustment_advanced.py`'s `triangulate_landmark` solves triangulation in closed form: for each observing camera, the calibrated ray direction $d$ (from the camera center through the observed pixel) defines a line in 3D space, and the best point is the one minimizing the sum of squared perpendicular distances to every observing ray - the "closest point to N lines" problem. For a single ray with direction $d$ (unit norm) through camera center $o$, the projector $M = I - dd^\top$ measures perpendicular distance to that ray; summing $M$ and $Mo$ across every observer and solving the resulting $3\times3$ linear system gives the point directly, no iteration needed.

```text
Camera 1               Camera 2
   📷                      📷
    \                      /
     \                    /
      \                  /
       \                /
        ●  <- closed-form least-squares
           intersection of both rays
```

`refine_landmark_gn` then runs a few Gauss-Newton iterations against the *true* nonlinear reprojection error, seeded from that closed-form guess - the same "linear guess, then GN polish" pattern used everywhere in this repo. Finally, `passes_cheirality` checks that the resulting point has positive depth in every observing camera: because `pixel = f*x/z` is invariant to negating a camera-frame point's $x,y,z$ together, a weakly-constrained point (few observers, little parallax) can converge to a *reflection* behind a camera that still fits every pixel almost exactly. A point failing this check is never committed to the map (see the [`bundle_adjustment_advanced.py` README description](../../README.md#7-local--global-bundle-adjustment-bounded-windows-vs-whole-map-re-solves)) - it's left pending until another observation gives it more parallax to resolve with.

**The triangulation math, concretely.** Poses are camera-to-world: observer $k$ has rotation $R_k$ and camera center $o_k$ (its translation). `triangulate_landmark` rotates each pixel's calibrated ray into the world frame and normalizes it:

```math
\tilde d_k = \begin{bmatrix} (u_k - c_x)/f_x \\ (v_k - c_y)/f_y \\ 1 \end{bmatrix}, \qquad
d_k = \frac{R_k \tilde d_k}{\left\| R_k \tilde d_k \right\|}
```

$$
A = \sum_k \left(I - d_k d_k^\top\right), \qquad
b = \sum_k \left(I - d_k d_k^\top\right) o_k, \qquad
P_0 = \left(A + 10^{-9} I\right)^{-1} b
$$

The tiny $10^{-9} I$ keeps the solve defined when all rays are parallel. `refine_landmark_gn` then runs Gauss-Newton on the point alone, with $J_{\text{point}}$ from [bundle_adjustment.md Section 6.1](../optimization/bundle_adjustment.md#61-the-ba-math-concretely):

$$
\left(\sum_k J_{\text{point},k}^\top J_{\text{point},k} + 10^{-9} I\right)\delta = \sum_k J_{\text{point},k}^\top r_k, \qquad
r_k = z_k - \pi\left(R_k^\top (P - o_k)\right), \qquad
P \leftarrow P + \delta
$$

It stops when the step norm drops below `gn_tol`. `passes_cheirality` then requires a strictly positive depth in every observer, using `point_depth`:

$$
\left[R_k^\top (P - o_k)\right]_z > 0 \quad \text{for every observer } k
$$

In `bundle_adjustment_advanced.py` a landmark is triangulated once it has `min_observations` = 3 observers (the default), with that script's `gn_tol` = $10^{-6}$ and `gn_max_iters` = 15.

---

## 3. PnP as the inverse problem

`pnp_estimation.py` solves the exact mirror image. Given $n$ known 3D points $P_i$ and their observed pixels, first form the calibrated ray for each: $d_i = \left[\frac{u_i-c_x}{f_x}, \frac{v_i-c_y}{f_y}, 1\right]$. Because $d_i$ is parallel to the camera-frame point $R_{cw}P_i + t_{cw}$ ($R_{cw}, t_{cw}$ the unknown world-to-camera pose), their cross product must vanish:

$$d_i \times (R_{cw}P_i + t_{cw}) = 0$$

This is **linear and homogeneous** in the 12 flattened entries of $[R_{cw} \mid t_{cw}]$ - exactly the classical **Direct Linear Transform (DLT)** camera-resectioning setup, specialized to *known* intrinsics. Stacking two independent rows of this constraint per correspondence gives an over-determined homogeneous system $Ax=0$, solved via `linear_pnp_dlt` as the smallest right-singular vector of $A$ (`np.linalg.svd`).

> **Note**: SVD (Singular Value Decomposition) factors any matrix as $A=U\Sigma V^\top$, with $U,V$ orthogonal and $\Sigma$ diagonal (the singular values, ranking how much each orthogonal direction contributes to $A$). Taking the smallest right-singular vector of $A$ - the column of $V$ paired with the smallest singular value - gives the least-squares null-space solution `linear_pnp_dlt` uses above; taking $UV^\top$ from a matrix's own SVD gives the nearest true rotation, which is exactly what the orthogonalization step below does.

Concretely, each correspondence adds two rows to $A$: the first two rows of the cross-product matrix $[d_i]_\times$, times a $3\times12$ matrix that maps $x$ to $R_{cw}P_i + t_{cw}$. Here $r_1, r_2, r_3$ are the rows of $R_{cw}$:

```math
A_i = \left([d_i]_\times\right)_{\text{rows }1,2}
\begin{bmatrix} P_i^\top & 0 & 0 & 1 & 0 & 0 \\ 0 & P_i^\top & 0 & 0 & 1 & 0 \\ 0 & 0 & P_i^\top & 0 & 0 & 1 \end{bmatrix}, \qquad
x = \begin{bmatrix} r_1 & r_2 & r_3 & t_{cw}^\top \end{bmatrix}^\top
```

The third row is dropped because it is a combination of the first two (the third entry of $d_i$ is always 1). $A$ is $2n\times12$ and needs rank 11 for a one-dimensional null space. That is why §5 asks for at least 6 points.

The recovered $3\times3$ block is only a *scaled, possibly reflected* rotation - not yet a valid element of $SO(3)$. `linear_pnp_dlt` turns $x$ into a pose in this order:

1. **Unpack.** $x$ is the last row of $V^\top$. Its first 9 entries, row-major, give $R_{\text{raw}}$. The last 3 give $t_{\text{raw}}$.
2. **Fix the sign.** $x$ and $-x$ both solve $Ax = 0$. If the median over all points of the raw depth $(R_{\text{raw}}P_i + t_{\text{raw}})_z$ is negative, both $R_{\text{raw}}$ and $t_{\text{raw}}$ are negated (§4). This runs *before* orthogonalization.
3. **Orthogonalize.** With $R_{\text{raw}} = U\Sigma V^\top$, set $R_{cw} = UV^\top$. If $\det(R_{cw}) < 0$, the last row of $V^\top$ is negated and $R_{cw}$ is recomputed, so the result is a proper rotation.
4. **Recover scale.** $s = (\sigma_1 + \sigma_2 + \sigma_3)/3$, the mean of the singular values of $R_{\text{raw}}$. Then $t_{cw} = t_{\text{raw}}/s$.
5. **Invert.** The function returns the camera-to-world pose, with rotation $R_{cw}^\top$ and translation $-R_{cw}^\top t_{cw}$.

With noise-free pixels this recovers the true pose to floating-point precision.

`refine_pose_gn` then runs Gauss-Newton on the true reprojection error, updating the pose via a right-multiplicative correction $T \leftarrow T\cdot\mathrm{Exp}(\delta)$ - mirroring `refine_landmark_gn`'s loop exactly, just solving a $6\times6$ system for the pose instead of a $3\times3$ system for the point. It uses the same $J_{\text{pose}}$ as bundle adjustment ([bundle_adjustment.md Section 6.1](../optimization/bundle_adjustment.md#61-the-ba-math-concretely)), for $\delta = [\delta v, \delta\omega]$:

$$
\left(\sum_i J_{\text{pose},i}^\top J_{\text{pose},i} + 10^{-9} I_6\right)\delta = \sum_i J_{\text{pose},i}^\top r_i, \qquad
r_i = z_i - \pi\left(T^{-1} P_i\right)
$$

It stops when the step norm drops below `gn_tol`. Script defaults: 20 points, sampled at camera-frame depth 3 to 8 m, $f_x = f_y = 800$ px on a 640×480 image, pixel noise $\sigma = 1$ px, `gn_tol` = $10^{-8}$, at most 20 iterations.

```text
Known point P                 Unknown pose?
     ●
      \   observed pixel
       \  ↙
        📷 ?
```

---

## 4. The shared reflection/cheirality risk

Both directions share the same underlying ambiguity: a pinhole camera's projection equation is invariant to certain sign flips, so a *linear* solve (which only sees the projection equation's algebraic structure, not "in front of the camera" as a constraint) can return a geometrically nonsensical but numerically consistent answer.

- Triangulation: `passes_cheirality` explicitly checks the recovered point has positive depth in every observer.
- PnP: `linear_pnp_dlt`'s sign-fix step (checking `np.median(depths_raw) < 0.0` before orthogonalization) is doing the *identical* check, just applied to fix the pose's sign ambiguity rather than reject a bad point outright.

Neither problem is "solved" by the linear step alone - both need this positive-depth reasoning layered on top before the result is trustworthy.

---

## 5. What the accompanying scripts do (and don't)

`linear_pnp_dlt` is the simplest *correct* version of linear PnP, not a production algorithm - it needs $n\geq 6$ well-conditioned correspondences to be numerically stable (rank-3+ data), and its accuracy degrades faster than purpose-built solvers as $n$ grows or points become near-coplanar. Production systems (OpenCV's own `solvePnP`, ORB-SLAM, COLMAP) typically use **EPnP** (Lepetit, Moreno-Noguer & Fua, 2009 - see References), an $O(n)$ algorithm that expresses every 3D point as a weighted combination of four virtual control points, turning the problem into recovering just those four points' camera-frame coordinates - more accurate and much cheaper at scale than a general DLT null-space solve. This repo implements the simpler DLT version for pedagogical clarity, matching `triangulate_landmark`'s own choice of a simple closed-form linear solve over a more sophisticated one.

---

## 6. One-sentence summary

> **Triangulation and PnP are the same reprojection problem run in opposite directions - one holds poses fixed to solve for a point, the other holds several known points fixed to solve for a pose - and this repo solves both the same way: a closed-form linear guess (ray intersection / DLT) followed by a few Gauss-Newton iterations, guarded against the same sign/reflection ambiguity in both directions.**

---

## 7. References

1. Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. - the standard reference for DLT camera resectioning behind §3's derivation.
2. Lepetit, V., Moreno-Noguer, F., & Fua, P. (2009). *EPnP: An Accurate O(n) Solution to the PnP Problem*. International Journal of Computer Vision, 81(2), 155-166. https://doi.org/10.1007/s11263-008-0152-6 - the production-grade PnP algorithm named as a contrast in §5.
3. Triggs, B., McLauchlan, P. F., Hartley, R. I., & Fitzgibbon, A. W. (2000). *Bundle Adjustment - A Modern Synthesis*. In Vision Algorithms: Theory and Practice (pp. 298-372). Springer. - already cited in [bundle_adjustment.md](../optimization/bundle_adjustment.md), covering the triangulation-within-BA context behind §2.
