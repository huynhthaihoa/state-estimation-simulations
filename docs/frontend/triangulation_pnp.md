# Triangulation and PnP: two sides of one geometric problem

> **Triangulation asks "given known camera poses and a 2D observation in each, where is the 3D point?" PnP asks the exact inverse: "given a known 3D point and its observed 2D pixel, where is the camera?" Both reduce to the same reprojection residual, solved the same way in this repo - a closed-form linear initial guess, then a few Gauss-Newton iterations.**

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

`bundle_adjustment_advanced.py`'s `triangulate_landmark` solves triangulation in closed form: for each observing camera, the calibrated ray direction $d$ (from the camera center through the observed pixel) defines a line in 3D space, and the best point is the one minimizing the sum of squared perpendicular distances to every observing ray - the "closest point to N lines" problem. For a single ray with direction $d$ (unit norm) through camera center $o$, the projector $M = I - dd^\top$ measures perpendicular distance to that ray; summing $M$ and $Mo$ across every observer and solving the resulting 3x3 linear system gives the point directly, no iteration needed.

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

---

## 3. PnP as the inverse problem

`pnp_estimation.py` solves the exact mirror image. Given $n$ known 3D points $P_i$ and their observed pixels, first form the calibrated ray for each: $d_i = \left[\frac{u_i-c_x}{f_x}, \frac{v_i-c_y}{f_y}, 1\right]$. Because $d_i$ is parallel to the camera-frame point $R_{cw}P_i + t_{cw}$ ($R_{cw}, t_{cw}$ the unknown world-to-camera pose), their cross product must vanish:

$$d_i \times (R_{cw}P_i + t_{cw}) = 0$$

This is **linear and homogeneous** in the 12 flattened entries of $[R_{cw} \mid t_{cw}]$ - exactly the classical **Direct Linear Transform (DLT)** camera-resectioning setup, specialized to *known* intrinsics. Stacking two independent rows of this constraint per correspondence gives an over-determined homogeneous system $Ax=0$, solved via `linear_pnp_dlt` as the smallest right-singular vector of $A$ (`np.linalg.svd`).

The recovered $3\times3$ block is only a *scaled, possibly reflected* rotation - not yet a valid element of $SO(3)$ - so `linear_pnp_dlt` projects it onto the nearest true rotation via SVD orthogonalization ($R_{cw} = UV^\top$ from $R_{raw}=U\Sigma V^\top$), recovers scale from the singular values, and fixes the sign using a positive-depth check (§4). `refine_pose_gn` then runs Gauss-Newton on the true reprojection error, updating the pose via a right-multiplicative correction $T \leftarrow T\cdot\mathrm{Exp}(\delta)$ - mirroring `refine_landmark_gn`'s loop exactly, just solving a $6\times6$ system for the pose instead of a $3\times3$ system for the point.

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

## 5. Where PnP shows up elsewhere in this monorepo

The exact same problem - known 3D points, known intrinsics, unknown pose - is solved via `cv2.solvePnP` in `camera-calibration/utils.py`, to recover each calibration image's extrinsics from its detected checkerboard corners. That call is a black box (OpenCV's own implementation, not derived), but it's worth noting as a real, working instance of this same math elsewhere in this repo, in a calibration context rather than a SLAM front-end context.

---

## 6. What the accompanying scripts do (and don't)

`linear_pnp_dlt` is the simplest *correct* version of linear PnP, not a production algorithm - it needs $n\geq 6$ well-conditioned correspondences to be numerically stable (rank-3+ data), and its accuracy degrades faster than purpose-built solvers as $n$ grows or points become near-coplanar. Production systems (OpenCV's own `solvePnP`, ORB-SLAM, COLMAP) typically use **EPnP** (Lepetit, Moreno-Noguer & Fua, 2009 - see References), an $O(n)$ algorithm that expresses every 3D point as a weighted combination of four virtual control points, turning the problem into recovering just those four points' camera-frame coordinates - more accurate and much cheaper at scale than a general DLT null-space solve. This repo implements the simpler DLT version for pedagogical clarity, matching `triangulate_landmark`'s own choice of a simple closed-form linear solve over a more sophisticated one.

---

## 7. References

1. Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. - the standard reference for DLT camera resectioning behind §3's derivation.
2. Lepetit, V., Moreno-Noguer, F., & Fua, P. (2009). *EPnP: An Accurate O(n) Solution to the PnP Problem*. International Journal of Computer Vision, 81(2), 155-166. https://doi.org/10.1007/s11263-008-0152-6 - the production-grade PnP algorithm named as a contrast in §6.
3. Triggs, B., McLauchlan, P. F., Hartley, R. I., & Fitzgibbon, A. W. (2000). *Bundle Adjustment - A Modern Synthesis*. In Vision Algorithms: Theory and Practice (pp. 298-372). Springer. - already cited in [bundle_adjustment.md](../optimization/bundle_adjustment.md), covering the triangulation-within-BA context behind §2.

---

## 8. One-sentence summary

> **Triangulation and PnP are the same reprojection problem run in opposite directions - one holds poses fixed to solve for a point, the other holds a point fixed to solve for a pose - and this repo solves both the same way: a closed-form linear guess (ray intersection / DLT) followed by a few Gauss-Newton iterations, guarded against the same sign/reflection ambiguity in both directions.**
