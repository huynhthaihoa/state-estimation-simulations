# LiDAR geometric primitives: planes and cylinders for scan registration

## Intuition

A LiDAR scan is a cloud of points. To turn two scans into a relative pose, the front-end ([glossary](../glossary.md#front-end-and-back-end)) has to say which part of one scan corresponds to which part of the other. [`pointcloud_pose_tracking.py`](../../use_numpy/pointcloud_pose_tracking.py) ([note](../filtering/pointcloud_pose_tracking_empirical_note.md)) skips that step: every measured point arrives paired with its body point. This doc does it for real, three ways:

- **point-to-point ICP** ([Iterative Closest Point](../foundations/umeyama_alignment.md), ref. 1): pair each source point with its nearest target point and pull them together;
- **point-to-plane ICP** (ref. 2): the same pairs, but only the distance along the target surface's normal counts, so a point may slide along the surface;
- **primitives**: first fit a few **geometric primitives** (infinite planes, cylinders) to each scan, then match primitives between scans and pull each source primitive's points onto its matched target surface.

Man-made scenes are full of primitives: floors, walls and ceilings are planes; pillars, pipes and tree trunks are close to cylinders. A primitive summarizes thousands of points with a handful of parameters that do not depend on where the LiDAR's rings happened to hit. That has three consequences, each measured below:

- **Accuracy.** Point pairs from two scans are not the same surface points (the ring pattern moves with the sensor), so ICP has an error floor even without noise; primitives do not.
- **Honesty about degeneracy.** In a corridor with only planes, nothing fixes the position along the corridor. The primitive method's Gauss-Newton Hessian ([glossary](../glossary.md#normal-equations-and-hessian)) says so; the ICP variants find spurious information and report 0.1-2 mm of uncertainty while being off by 12-29 cm.
- **Landmarks.** A plane seen from many poses is a natural **landmark** ([glossary](../glossary.md#landmark)) for a factor graph.

The price is a narrower convergence basin than point-to-plane ICP's: primitives must be matched before anything else works.

![Two panels from lidar_primitives.py: one scan of the corridor with pillars seen from above, with points colored by the primitive they were assigned to (two floor/ceiling planes, two wall planes, three cylinders, the rest unassigned); and the L-shaped corridor of experiment 4 with its walls, pillars and 40 sensor poses 1 m apart](../../assets/lidar_primitives_concept.png)

*Figure: one scan of the corridor with pillars as the extraction sees it, and the scene of §6. Plotted by `uv run python assets/make_figures.py lidar_primitives_concept`.*

**Scope.** Simulated scans only: an open-ended corridor (6 m wide, 3 m high), optionally with two staggered rows of floor-to-ceiling pillars (radius 0.25 m, every 8 m), and an L-shaped version of it; a spinning 32-ring LiDAR (±20°, 360 points per ring, 20 m range) with Gaussian noise on the range only. Static scenes, no motion distortion within a scan, no intensity. Run with `uv run python use_numpy/lidar_primitives.py` (several minutes per experiment; `--experiment 1|2|3|4` for one). [`use_manif/lidar_primitives.py`](../../use_manif/lidar_primitives.py) is the same with manif poses. It agrees with the numpy version to round-off, except in runs that pass through a nearly singular step (§4), where round-off decides the outcome: of the experiment 2 starts compared (2 pairs, 2 starts per size), one at 2 m and 30° converges in numpy and fails in manif.

---

## 1. Extraction

Everything that never touches a pose lives in the shared [`lidar_utils.py`](../../lidar_utils.py).

**Primitives.** A plane is $`(n, d)`$ with $`n^\top x + d = 0`$ and $`\|n\| = 1`$; in a scan's sensor frame $n$ points toward the sensor, so $`d > 0`$ is the sensor's distance to the plane. A cylinder is $`(a, c, r)`$: unit axis $a$, the axis point $c$ closest to the frame origin ($`a^\top c = 0`$), radius $r$. Unit vectors live on the sphere $`S^2`$, a 2D **manifold** ([glossary](../glossary.md#manifold)), so plane normals and cylinder axes are updated by a **retraction** ([glossary](../glossary.md#retraction)) along a tangent basis (`s2_exp`/`s2_log`) rather than by adding 3-vectors: 3 degrees of freedom per plane, 5 per cylinder.

**Normals.** Each point gets a normal from a PCA of its 12 nearest neighbors: the eigenvector of the smallest eigenvalue of their scatter matrix.

**Sequential RANSAC** ([`triangulation_pnp.md`](triangulation_pnp.md) uses RANSAC for outlier rejection; ref. 3). Every round draws the best plane hypothesis (from 3 random points) and the best cylinder hypothesis (from 2 points and their normals: the axis is $`n_1 \times n_2`$, and in the plane orthogonal to it the two normal lines meet at the axis). A point is an inlier if it is within 3σ of the surface (at least 5 cm) and its normal agrees within 25°. Both hypotheses are refined (PCA for the plane, Gauss-Newton on 5 parameters for the cylinder), and the one with more inliers wins; its inliers leave the pool. This competition is the selection rule of Schnabel et al. (ref. 4), and it was not optional here: extracting all planes first gave 7-8 "planes" per scan instead of 4, because within a few centimeters the side of a pillar that faces the sensor is flat enough to pass as a plane, and the nearest pillars, the most useful ones, disappeared.

Two more fixes came out of the tests:

- **Divergent cylinder fits.** A refinement can run off to a huge radius; a fit is now rejected unless its radius stays in $`[0.05, 1]`$ m and it keeps enough inliers.
- **Corner contamination.** Near the line where a wall meets the floor, normals average over both surfaces, so a few floor or ceiling points pass the wall's distance band with a "close enough" normal while sitting up to 4 cm off the wall. That tilted noise-free wall fits by up to $`1.5 \times 10^{-4}`$ rad. A final cut at 3 robust standard deviations of the residuals ($`1.4826 \times`$ the median absolute residual) brings it below $`2 \times 10^{-7}`$ rad; the cut points go back to the pool for the floor to claim.

---

## 2. Extraction accuracy (experiment 1)

10 random sensor poses per cell, seed 0, primitives with at least 150 points (`run_extraction_experiment`). Errors are medians; the radius error is the mean of the signed error, so a bias shows.

| Range noise σ | Planes found | Plane normal / offset error | Pillars found (180 / 360 / 720 points per ring) | Pillar axis position error | Pillar radius error, orthogonal fit |
|---|---|---|---|---|---|
| 5 mm | 100% | ≤ 0.003° / ≤ 0.04 mm | 100 / 100 / 97% | 0.6-1.1 mm | −0.2 to +0.2 mm |
| 10 mm | 100% | ≤ 0.005° / ≤ 0.13 mm | 100 / 95.5 / 94.7% | 1.8-2.5 mm | −0.6 to −1.4 mm |
| 20 mm | 100% | ≤ 0.016° / ≤ 0.19 mm | 100 / 100 / 87% | 6.4-7.5 mm | −2.7 to −5.8 mm |
| 40 mm | 100% | ≤ 0.024° / ≤ 0.47 mm | 100 / 100 / 61% | 23.9-29.3 mm | −15.7 to −20.1 mm |

Planes are easy: thousands of inliers average the noise away. Pillars show two effects worth a closer look.

### 2.1 More points per ring made pillar detection worse

At 40 mm noise, 720 points per ring finds only 61% of the pillars, with 12 false detections, where 180 points per ring finds all of them. The normals are the reason. A fixed 12-neighbor PCA covers a smaller patch when the points are denser, so the same range noise tilts the normal more: on the planes' points at 40 mm noise the median normal error is 5.3° at 180 points per ring and 11.9° at 720. The cylinder hypotheses and the 25° normal test both rely on those normals. A fixed neighborhood *radius* instead of a fixed count should remove the effect (not tested here); the script keeps the fixed count so the effect stays visible.

### 2.2 The usual cylinder fit underestimates the radius

The orthogonal-distance fit minimizes $`\sum_i (\|(I - aa^\top)(x_i - c)\| - r)^2`$, the distance of each point to the surface. It underestimates the radius, by a bias that grows with the square of the noise: about −5 mm at 20 mm noise, −17 mm at 40 mm. Fitting only the true pillar points, starting from the true parameters, isolates the cause (σ = 4 cm, 67 fits):

| Noise added to the true pillar points | Mean radius error |
|---|---|
| along each ray (what the simulated LiDAR does) | −16.8 mm, with the axis 22 mm closer to the sensor |
| perpendicular to the surface only | −0.3 mm |
| along the surface, around the pillar, only | +4.0 mm |
| the same in every direction | +3.6 mm |

So it is the estimator, not RANSAC or the inlier selection. The orthogonal fit is the maximum-likelihood fit only when the noise is the same in every direction, and even then it is slightly biased: noise around the pillar lengthens each point's distance to the axis by about $`\sigma^2 / (2r)`$ on average, 3.2 mm here, close to the measured +3.6 and +4.0 mm. LiDAR noise lies along the ray instead, and the sensor sees only the near side of a pillar. A plausible reading of the large negative bias: near the pillar's silhouette the rays graze the surface, so range noise slides points along it, and on a one-sided arc a smaller radius with the axis moved toward the sensor fits those points almost as well. The bias is measured here, not derived in closed form.

The **range fit** (`refine_cylinder_range`) instead compares each point's measured range with the range at which its ray meets the cylinder, which is the maximum-likelihood fit for range-only noise. A ray that misses the current cylinder gets its closest-approach range plus a second residual, its gap to the surface; both are continuous at tangency. Dropping missed rays instead would discard exactly the rays that say "the pillar is wider". A 2D check (a circle seen from 4 m, 300 trials) shows the difference:

| Radius | Orthogonal fit, σ 2 / 4 cm | Range fit, σ 2 / 4 cm |
|---|---|---|
| 0.25 m | −3.7 / −14.4 mm | +0.5 / +2.7 mm |
| 0.5 m | −2.0 / −8.7 mm | +0.1 / +1.1 mm |
| 1.0 m | −1.2 / −5.9 mm | −0.1 / +0.3 mm |

In 3D on the true pillar points the range fit gives −0.3 mm at 2 cm (orthogonal −4.4) and −4.1 mm at 4 cm (orthogonal −16.9), with a larger spread at 4 cm (7.6 against 5.0 mm). A first-order fix, weighting each orthogonal residual by how much range noise can move that point off the surface, only reached −11.8 mm: a second-order bias needs the exact residual.

Inside the full pipeline (same RANSAC inliers for both fits), the range fit's bias changes sign:

| Range noise σ | Mean radius error, orthogonal (180 / 360 / 720 per ring) | Range fit | Axis position median, orthogonal → range |
|---|---|---|---|
| 20 mm | −3.9 / −2.7 / −5.8 mm | +2.5 / +3.6 / +2.1 mm | 6.4 / 6.9 / 7.5 → 5.4 / 3.7 / 1.9 mm |
| 40 mm | −20.1 / −15.7 / −15.8 mm | +6.6 / +5.9 / +3.6 mm | 28.9 / 29.3 / 23.9 → 13.5 / 20.2 / 9.4 mm |

At 40 mm the range fit cuts the radius bias 2.7-4.4× and the axis error 1.5-2.5×; at 20 mm the two biases are of similar size (at 360 points per ring the range fit's is larger), while the axis error drops 1.2-3.9×. At 40 mm the range fit also has heavier tails (radius RMSE 38.5 against 19.7 mm at 360 per ring). The orthogonal fit stays the default (`ExtractionConfig.cylinder_fit = "odr"`); `"range"` adds the range fit. Why the sign flips inside the pipeline is not confirmed; the likeliest suspect is the RANSAC inlier band, which is defined with the orthogonal distance.

![Three panels from lidar_primitives.py experiment 1: mean pillar radius error against range noise for the orthogonal and range fits at 180, 360 and 720 points per ring; median pillar axis position error for both fits; and the percentage of pillars detected against range noise for the three densities](../../assets/lidar_primitives_extraction.png)

*Figure: experiment 1 at its defaults, plotted by `uv run python assets/make_figures.py lidar_primitives_extraction` (about 3 minutes).*

---

## 3. Three residuals, one solver

Registration estimates $`T \in SE(3)`$, the pose of the source scan's sensor in the target scan's sensor frame (target point $`= T \cdot`$ source point). All three methods share one Gauss-Newton loop ([glossary](../glossary.md#gauss-newton)) with the right perturbation $`T \leftarrow T\,\mathrm{Exp}(\xi)`$, $`\xi = [\delta v, \delta\theta]`$ ([glossary](../glossary.md#left-and-right-perturbation)), and every residual of a source point $p$ goes through the same point-action Jacobian

```math
\frac{\partial\, T\,\mathrm{Exp}(\xi)\, p}{\partial \xi}\Big|_{\xi = 0} = \begin{bmatrix} R & -R\,p^\wedge \end{bmatrix}
```

with $`p^\wedge`$ the skew-symmetric matrix of $p$ ([glossary](../glossary.md#hat-operator)). With $q$ a target point, $`n_q`$ its PCA normal, and $`(n, d)`$, $`(a, c, r)`$ a matched target plane and cylinder:

| Method | Residual of one source point | Pairs or matches |
|---|---|---|
| point-to-point | $`Tp - q`$ (3 rows) | nearest target point |
| point-to-plane | $`n_q^\top (Tp - q)`$ | nearest target point |
| primitives, plane | $`n^\top Tp + d`$ | the source point's plane, matched to a target plane |
| primitives, cylinder | $`\|(I - aa^\top)(Tp - c)\| - r`$ | the source point's cylinder, matched to a target cylinder |

**Association happens at every iteration**, from the current estimate; nothing is handed over. Primitives are matched one-to-one, cheapest first, inside gates: normals within 20° and offsets within 1 m for planes; axes within 20°, axis distance within 1 m and radii within 0.1 m for cylinders.

**A coarse-to-fine pairing gate.** Nearest neighbors farther apart than a gate are dropped. A fixed 1 m gate was a strawman: started exactly at the truth with no noise, point-to-plane drifted to 4-12 mm and 0.1-0.2° (4 scan pairs), and point-to-point kept only 85-98% of the true motion. The likely cause is pairs whose nearest neighbor lies on another surface, where the target scan did not sample that surface (between rings, behind a pillar): tightening the gate to 0.1 m shrank the point-to-plane error to 0.6-2.6 mm and 0.014-0.020°. A gate that starts at 1 m and shrinks by 0.7 per iteration down to 0.1 m, common ICP practice, keeps the 1 m reach for the first iterations; on one pair it brings the drift from the truth to 7 mm (point-to-point) and 2.6 mm / 0.02° (point-to-plane). What remains is the sampling floor: the two scans never hit the same surface points.

**A side check for cylinders.** The distance to a cylinder does not know which side was scanned. In 2 of 50 starts, primitive registration converged 0.34 m off along the corridor with the correct matches; at that solution every point of 2 of the 3 matched pillars sat on the far side of its target pillar (none did at the truth). A point now counts only if its normal, mapped by $T$, points away from the target axis, as on the side a sensor can see. That removed those failures.

**Steps.** Each step is the minimum-norm least-squares solution, with singular values below $`10^{-6}`$ of the largest treated as zero. That only drops directions with no information at all, up to round-off; a merely weak direction is kept, which matters in §5.

**Checked.** `tests/use_numpy/test_lidar_primitives.py` compares every registration and factor Jacobian with finite differences, and recovers the noise-free truth exactly (below a micrometer) with the primitive method from a 0.37 m, 5.7° start.

---

## 4. Convergence basin (experiment 2)

Corridor with pillars, 1 cm range noise, 10 scan pairs (the source sensor 0.82 m and 6.9° from the target sensor), 5 random initial errors per pair and size; success means within 5 cm and 1° of the truth.

| Initial error | 0.1 m, 2° | 0.25 m, 5° | 0.5 m, 10° | 1 m, 20° | 2 m, 30° | Final error (median of successes) | Iterations |
|---|---|---|---|---|---|---|---|
| point-to-point | 100% | 92% | 22% | 0% | 0% | 11-15 mm, 0.2° | 63-100 |
| point-to-plane | 100% | 100% | 100% | 98% | 16% | 1.3-2.4 mm, 0.013° | 13-20 (100 at 2 m) |
| primitives | 100% | 100% | 100% | 84% | 2% | 0.2-0.4 mm, 0.002° | 4-5 (1 at 2 m: no matches) |

The primitive method is the most accurate (sub-millimeter) and needs the fewest iterations, but point-to-plane ICP recovers from larger initial errors. Classifying every failed start by what was matched at the start:

- **At 1 m and 20° (8 failures).** 2 start with only planes matched, which leaves the corridor axis free (§5); 6 start with planes and 1-2 pillars, all matched correctly. In 7 of the 8, the first one or two Gauss-Newton steps jump, mostly along the corridor axis, by 3 m to 9 km, and every match is lost: the planes leave the axis free and 1-2 pillars hold it only weakly, and from a 20° error the linearization is poor. The 8th diverges in rotation instead. Limiting the step along weak directions (the solution remapping of §5, or a damped or trust-region step) targets this; neither is implemented.
- **At 2 m and 30° (49 failures).** 36 starts match nothing on the first iteration (the 20° gates), so there is nothing to solve; 9 match only planes (mostly two) and run off by 190 m to 10 km; 4 also match a pillar and stall 2-6 m off with the rotation still 30-67° wrong. The single success is round-off luck: the manif backend, identical up to $`10^{-16}`$ at the start, fails it (see Scope).

Point-to-point ICP is slow and stalls: from 0.5 m and 10°, 78% of its starts fail, and in the first 4 pairs 14 of the 20 starts end 10 cm or more off.

![Animation from animate_lidar_primitives.py: two LiDAR scans of the corridor with pillars seen from above, the source scan moved by each method's estimate after each Gauss-Newton iteration from the same start 0.5 m and 10 degrees off; point-to-point ICP stalls at 299.6 mm and 4.59 degrees after 100 iterations, point-to-plane converges to 0.4 mm in 12 iterations and the primitive method to 0.2 mm in 4; a zoom on the nearest pillar shows the point-to-point points off the true outline](../../assets/lidar_primitives.gif)

*Animation: pair 1 of experiment 2, start 2 at 0.5 m and 10°, one Gauss-Newton iteration per frame. `uv run python use_numpy/animate_lidar_primitives.py`.*

---

## 5. Degeneracy: what the Hessian knows (experiment 3)

In the plain corridor every surface is parallel to the corridor axis: floor, ceiling and both walls. Sliding the sensor along the axis changes no point-to-surface distance, so the position along the axis is not observable from these surfaces ([glossary](../glossary.md#observability)). The question is which method knows that.

Each method's Hessian $`H = J^\top J`$ is evaluated at the true pose (so it describes the problem, not wherever the solver ended), and its predicted standard deviation along the axis is $`\sigma \sqrt{g^\top H^{+} g}`$, with $`H^+`$ the pseudo-inverse, $g$ the axis direction and σ the range noise ([glossary](../glossary.md#information-matrix)). Using σ for every residual flatters the ICP variants: their residuals also carry the target point's noise (and, for point-to-plane, its normal's) and the sampling mismatch of §3, so their σ would be optimistic even without degeneracy. Separately, each method registers the scans from a 0.2 m, 2° initial error. 10 scan pairs, 1 cm noise:

| Scene | Method | Hessian eigenvalue ratio (min/max) | Weakest direction along the axis? | Predicted σ along / across the axis (median) | Actual error along the axis (median) | Ratio of the two medians |
|---|---|---|---|---|---|---|
| planes only | point-to-point | $`5.4 \times 10^{-2}`$ | no (\|cos\| 0.04) | 0.11 / 0.11 mm | 118 mm | about 1,100 |
| planes only | point-to-plane | $`3.9 \times 10^{-4}`$ | yes | 2.1 / 0.12 mm | 288 mm | about 140 |
| planes only | primitives | $`3.0 \times 10^{-12}`$ | yes | 13,400 / 0.13 mm | 36,800 mm | 2.7 |
| with pillars | point-to-point | $`5.7 \times 10^{-2}`$ | no (\|cos\| 0.06) | 0.11 / 0.11 mm | 12.5 mm | about 116 |
| with pillars | point-to-plane | $`1.1 \times 10^{-2}`$ | yes | 0.42 / 0.12 mm | 1.35 mm | 3.3 |
| with pillars | primitives | $`4.2 \times 10^{-3}`$ | yes | 0.49 / 0.11 mm | 0.36 mm | 0.74 |

For a consistent estimate ([glossary](../glossary.md#consistency-and-nees)) with Gaussian errors and a similar σ in every pair, the ratio of the two medians is about 0.67 (the median of a standard normal's absolute value). With 10 pairs this is a rough check.

Three different behaviors:

- **Point-to-point ICP does not see the degeneracy.** Its Hessian is as well conditioned in the plain corridor as with pillars, with the same 0.11 mm in every direction, because every pair pulls in all three directions, including along the wall. It is off by 118 mm along the corridor while reporting a tenth of a millimeter. Even with pillars it is off by 116 of its own σ.
- **Point-to-plane ICP sees it but underrates it.** Its weakest direction is the corridor axis, with a predicted σ 18 times the one across it. But the axis still gets spurious information, presumably because the PCA normals of a noisy wall are not exactly perpendicular to it: the predicted 2.1 mm against an actual 288 mm.
- **The primitive method sees what is there.** In the plain corridor its Hessian has essentially no information along the axis (eigenvalue ratio $`3 \times 10^{-12}`$, predicted σ 13 m), and plain Gauss-Newton does what that predicts: with nothing holding it, the step along the axis is driven by sub-millimeter fit errors divided by almost zero, and the estimate runs off by tens of meters (its median error is 2.7 of its own σ). The rank cutoff of §3 just misses it: the singular-value ratio, $`1.7 \times 10^{-6}`$, is above the $`10^{-6}`$ cutoff. With pillars it is consistent: a median of 0.74 σ, close to the 0.67 of a Gaussian.

The pillars fix the axis because a vertical cylinder constrains both horizontal directions: its axis is a point in the floor plan. Of the six ratios in the table (about 1,100, 140, 2.7, 116, 3.3 and 0.74), only one is close to the Gaussian 0.67: the primitive method's with pillars.

**What a real system does about it.** Zhang, Kaess and Singh (ref. 6) detect degeneracy from the eigenvalues of exactly this Hessian and then **remap the solution**: the update is applied only along well-constrained eigenvectors, and the weak directions keep their prior value (from odometry or an IMU). The script implements only the diagnosis.

---

## 6. Planes as landmarks (experiment 4)

**Scene.** The L-shaped corridor: two 6 m wide legs with pillars, 40 sensor poses 1 m apart along the center line, around the corner (figure in the Intuition section). After the turn, the second leg's walls face the first leg's axis, so a world-frame plane landmark can tie the two legs together.

**Odometry.** Each scan is registered to the previous one with the primitive method, and the poses are chained. Each step starts from the true relative pose perturbed by 0.1 m and 2° in a random direction, standing in for an IMU or wheel-odometry prior. The corner turns up to about 38° per step, beyond the 20° matching gate, so a constant-velocity guess would fail there for reasons unrelated to this experiment. The odometry information is $`H / \sigma^2`$ of each registration, which §5 found to be honest for this method.

**Factor graph** ([glossary](../glossary.md#factor-graph)). Variables: the 40 poses and one world plane $`(n_w, d_w)`$ per landmark. Factors: one odometry factor per step, $`\mathrm{Log}(Z^{-1} X_i^{-1} X_j)`$ as in [`pose_graph_optimization.md`](../optimization/pose_graph_optimization.md), and one plane factor per extracted plane. The landmark predicted in the sensor frame of pose $`(R, t)`$ is $`n = R^\top n_w`$, $`d = d_w + n_w^\top t`$, and the residual against the measured plane $`(n_s, d_s)`$ is

```math
e = \begin{bmatrix} B(n_s)^\top n \\ d - d_s \end{bmatrix}
```

with $`B(n_s)`$ a basis of the tangent plane at $`n_s`$. Each plane factor is weighted by its own fit: the information $`\sum_i J_i^\top J_i / \sigma^2`$ with $`J_i = [p_i^\top B(n_s),\ 1]`$ over the plane's inlier points (a Monte Carlo test shows it matches the scatter of repeated noisy fits within 15%). Landmarks are created online: an extracted plane, mapped to the world by its odometry pose, joins the closest landmark within 10° and 0.5 m, or starts a new one. Pose 0 is anchored (gauge, [glossary](../glossary.md#gauge-freedom)), and the graph is solved with Levenberg-Marquardt ([glossary](../glossary.md#levenberg-marquardt-lm-and-damping)) as in `pose_graph.py`. Cylinders are used for registration only, not as landmarks.

**Results** (5 noise realizations, 1 cm noise; ATE is the position RMSE after the best rigid alignment, [glossary](../glossary.md#umeyama-alignment-and-trajectory-rmse); drift is the unaligned error of the last pose):

| | Landmarks created | ATE (median) | End-point drift over the 39 m path (median, max) |
|---|---|---|---|
| odometry only | - | 1.7 mm | 5.6 mm, 0.007° (max 6.8 mm) |
| odometry + plane landmarks | 6 in every run (= the 6 distinct planes) | 1.3 mm | 4.3 mm, 0.008° (max 5.7 mm) |

The landmarks help in all 5 runs (end-point drift 4.4 → 2.4, 5.6 → 4.3, 6.7 → 5.7, 6.8 → 5.5, 4.9 → 4.2 mm), but only a little, because there is little left to fix: with pillars in view, primitive odometry drifts about 0.14 mm per meter, and rotation does not improve at all. Landmarks matter in proportion to the odometry's drift; this experiment shows the mechanism working (every plane associated to one landmark, no duplicates), not a large gain.

![Three panels from lidar_primitives.py experiments 2 to 4: success rate against initial error for the three methods; the median error along the corridor axis against the predicted sigma for each method in the planes-only corridor and the corridor with pillars, on a log scale; and the end-point drift and aligned trajectory error of odometry alone versus the plane-landmark graph for five noise realizations](../../assets/lidar_primitives.png)

*Figure: experiments 2-4 at their defaults, plotted by `uv run python assets/make_figures.py lidar_primitives`. The numbers match the script's printout.*

---

## 7. Where this sits on the map

- **Front-end.** Extraction and matching are front-end work ([`frontend_backend.md` §1](../frontend_backend.md#1-slam-front-end---extracting-constraints)); their output is a relative pose with an information matrix, or plane measurements, for the back-end.
- **The step the point-cloud toy skips.** [`pointcloud_pose_tracking_empirical_note.md`](../filtering/pointcloud_pose_tracking_empirical_note.md) hands every point its correspondence; this doc is that missing step, and §3 shows how much it costs (ICP's sampling floor).
- **Structural elimination.** A plane landmark can be eliminated the way a BA landmark is (this repo's graph solves densely) ([`bundle_adjustment.md` §12](../optimization/bundle_adjustment.md#12-block-sparsity-and-the-schur-complement)); BALM (ref. 9) does it in closed form for planes.
- **Consistency.** §5 adds an accuracy-vs-honesty case to the [mental map's §6.5](../slam_mental_map.md#65-consistency-not-just-accuracy): ICP's covariance is far too small exactly where the problem is degenerate.

---

## 8. Takeaway

- **Fit the shapes, not the points.** Primitives remove ICP's sampling floor (0.2-0.4 mm against 1.3-2.4 mm for point-to-plane and 11-15 mm for point-to-point) and converge in 4-5 iterations.
- **But they must be matched first, and the first step must not overshoot.** Point-to-plane ICP still has the wider basin (98% against 84% at 1 m and 20°); the primitive method fails when its matching gates are exceeded, or when a weakly constrained first step jumps along the corridor and loses every match.
- **Planes alone leave a corridor's axis free.** Only a structure whose surface is not parallel to the axis fixes it, here the pillars. The primitive method's Hessian reports the free direction; the ICP variants report confidence they do not have.
- **Check the noise model of a fit.** Range-only noise makes the usual orthogonal cylinder fit underestimate the radius by a bias that grows with the square of the noise. On the true points the range-space fit removes most of it; inside the pipeline it helps clearly only at high noise.
- **Denser is not always better.** With fixed-count neighborhoods for the normals, more points per ring made pillar detection worse at high noise.

---

## 9. References

1. Besl, P. J., & McKay, N. D. (1992). *A Method for Registration of 3-D Shapes*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 14(2), 239-256. https://doi.org/10.1109/34.121791 - point-to-point ICP.
2. Chen, Y., & Medioni, G. (1992). *Object Modelling by Registration of Multiple Range Images*. Image and Vision Computing, 10(3), 145-155. https://doi.org/10.1016/0262-8856(92)90066-C - point-to-plane registration.
3. Fischler, M. A., & Bolles, R. C. (1981). *Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography*. Communications of the ACM, 24(6), 381-395. https://doi.org/10.1145/358669.358692 - RANSAC.
4. Schnabel, R., Wahl, R., & Klein, R. (2007). *Efficient RANSAC for Point-Cloud Shape Detection*. Computer Graphics Forum, 26(2), 214-226. https://doi.org/10.1111/j.1467-8659.2007.01016.x - shape hypotheses from points and normals, and choosing among shape types by score.
5. Zhang, J., & Singh, S. (2017). *Low-drift and Real-time Lidar Odometry and Mapping*. Autonomous Robots, 41(2), 401-416. https://doi.org/10.1007/s10514-016-9548-2 - LOAM: edge and planar feature points with point-to-line and point-to-plane residuals, the feature-based alternative to fitting whole primitives.
6. Zhang, J., Kaess, M., & Singh, S. (2016). *On Degeneracy of Optimization-based State Estimation Problems*. 2016 IEEE International Conference on Robotics and Automation (ICRA), 809-816. https://doi.org/10.1109/ICRA.2016.7487211 - degeneracy from the Hessian's eigenvalues, and solution remapping (§5).
7. Kaess, M. (2015). *Simultaneous Localization and Mapping with Infinite Planes*. 2015 IEEE International Conference on Robotics and Automation (ICRA), 4605-4611. https://doi.org/10.1109/ICRA.2015.7139837 - planes as landmarks, with a minimal parameterization.
8. Geneva, P., Eckenhoff, K., Yang, Y., & Huang, G. (2018). *LIPS: LiDAR-Inertial 3D Plane SLAM*. 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 123-130. https://doi.org/10.1109/IROS.2018.8594463 - plane landmarks with the closest-point parameterization, fused with an IMU.
9. Liu, Z., & Zhang, F. (2021). *BALM: Bundle Adjustment for Lidar Mapping*. IEEE Robotics and Automation Letters, 6(2), 3184-3191. https://doi.org/10.1109/LRA.2021.3062815 - plane and edge landmarks eliminated in closed form, the LiDAR analogue of BA's Schur complement.
