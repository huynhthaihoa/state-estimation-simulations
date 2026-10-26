# Front-end & Back-end in SLAM

In **SLAM** ([glossary](glossary.md#slam-simultaneous-localization-and-mapping)), the terms **front-end** and **back-end** describe two major parts of the system.

Think of them as:

> **Front-end: "What happened between these observations?"**

> **Back-end: "Given all observations, what is the most consistent trajectory and map?"**

![Pipeline: sensors feed the front-end (feature detection, matching and tracking, motion estimation, measurements and constraints; also data association, triangulation, PnP, keyframe selection and loop-closure detection), which hands constraints (relative poses, pixel observations, IMU bundles, priors) to the back-end (nonlinear least squares: bundle adjustment, pose graph, factor graph; Gauss-Newton, Levenberg-Marquardt and iSAM solvers; robust costs), which outputs the optimized trajectory and map. Loop closure spans both stages: the front-end detects a candidate, the back-end verifies and applies it. Both stages can run online.](images/frontend_backend.svg)

*Figure: the two stages and what passes between them, drawn from §1-§5 (regenerate with `uv run python generator/make_frontend_backend_svgs.py`). The back-end is drawn through its optimization form (§2); filtering is the other back-end strategy ([filtering_smoothing.md](filtering_smoothing.md)). **IMU bundles** are preintegrated IMU measurements between keyframes ([glossary](glossary.md#imu-preintegration)), front-end work whose output is a back-end factor; **priors** are single-variable constraints such as one that pins the first pose ([glossary](glossary.md#prior-unary-factor)). **PnP** is Perspective-n-Point, **iSAM** incremental smoothing and mapping, **DCS** dynamic covariance scaling (a robust cost).*

---

## 1. SLAM Front-end - extracting constraints

The **front-end processes raw sensor data** and turns it into useful geometric information.

For a camera-based SLAM system:

```text
Camera images
     ↓
Feature detection
     ↓
Feature matching/tracking
     ↓
Motion estimation
     ↓
Measurements/constraints
     ↓
  Back-end
```

Typical front-end tasks:

- **Feature extraction** (finding distinctive image points, such as corners) - a **detector** finds distinctive image points and a **descriptor** is a numeric signature of the patch around a point, used to match it between frames; examples are ORB (Oriented FAST and Rotated BRIEF), FAST (Features from Accelerated Segment Test), SIFT (Scale-Invariant Feature Transform)
- **Feature tracking/matching** (finding the same feature again in later frames)
- **Data association** - matching each measurement to the landmark it observes, e.g. determining that a feature in frame $k$ is the same physical point seen in frame $k+1$
- **Visual odometry** (VO) - [odometry](glossary.md#odometry) computed from camera images
- **Depth estimation/triangulation** (recovering a 3D point's position from two or more views with known camera poses) - see [frontend/triangulation_pnp.md](frontend/triangulation_pnp.md) for the actual math, plus its inverse problem, Perspective-n-Point (PnP), which estimates the [pose](glossary.md#pose) from known points
- **[Keyframe](glossary.md#keyframe) selection** (choosing a subset of frames to keep for optimization)
- **[Loop-closure](glossary.md#loop-closure) detection**

With a LiDAR instead of a camera, feature matching and motion estimation become **scan registration**: aligning one point cloud with another, point by point or through fitted planes and cylinders. See [frontend/lidar_primitives.md](frontend/lidar_primitives.md).

For example, suppose the camera sees:

```text
Frame 1                  Frame 2
   ●                        ●
      ●                      ●
          ●                    ●
```

The front-end tracks the same points and estimates the relative camera pose

$$\tilde T_{12} \in \mathrm{SE}(3)$$

(rotation + translation, see [SE(3)](glossary.md#so3-and-se3); the tilde marks a measurement, as in [pose_graph_optimization.md](optimization/pose_graph_optimization.md)), meaning:

> "The camera moved approximately this much from frame 1 to frame 2."

So the front-end produces **constraints**, not necessarily the final globally optimal trajectory.

---

## 2. SLAM Back-end - solving the global problem

The **back-end takes those constraints and optimizes the robot's trajectory and map**.

For example, the front-end might produce:

$$\tilde T_{01}, \tilde T_{12}, \tilde T_{23}, \tilde T_{34}$$

and [landmark](glossary.md#landmark) observations such as:

$$z_{11},\ z_{12},\ z_{21},\ \ldots$$

where $z_{ij}$ is the pixel measurement of landmark $j$ from pose $i$ (the bundle-adjustment $z_{ij}$, not the relative-pose $z_{ij}$ of the pose-graph docs).

Two more kinds of constraint reach the back-end: **IMU bundles** (preintegrated IMU measurements between keyframes) and **priors** (single-variable constraints, such as one that pins the first pose).

The back-end asks:

> "What set of robot poses and landmarks best explains **all** these measurements simultaneously?"

This is usually formulated as an optimization problem.

For example:

```math
\min_{\mathbf{x}} \frac12\sum_i \|r_i(\mathbf{x})\|^2
```

where $\mathbf{x}$ stacks all pose (and landmark) variables and $r_i$ is the [residual](glossary.md#residual-error) of measurement $i$, information-weighted (whitened; see [filtering_smoothing.md §3](filtering_smoothing.md#3-optimization-keep-the-history)). The $\frac12$ is a convention that does not change the minimizer.

**Intuition:** the residual is the gap between prediction and measurement. Toy case: the model predicts a landmark at pixel 100, the camera sees it at 103, so the residual has size 3. The sign convention varies, but the squared cost does not care.

Two levels to keep apart:

**Problem formulations** (what we optimize):

- **[Bundle adjustment (BA)](optimization/bundle_adjustment.md)** - jointly refine camera poses and 3D points to minimize reprojection error (the pixel gap between observed and projected points)
- **[Pose-graph optimization](optimization/pose_graph_optimization.md)** - poses only, linked by relative-pose constraints (odometry and loop closures)
- **[Factor-graph optimization](optimization/factor_graph.md)**
- **[Nonlinear least squares](optimization/nonlinear_least_square.md)** - the general form behind all of them

**Solvers** (how we minimize it):

- **[Gauss-Newton](optimization/gauss_newton.md)** and **[Levenberg-Marquardt](optimization/levenberg_marquardt.md)** - how each iteration chooses its step: solve the linearized problem, with LM adding damping to keep the step from overshooting
- **[iSAM/incremental optimization](optimization/isam_optimization.md)** - a strategy for re-solving incrementally as new constraints arrive

---

## 3. Example: loop closure

Imagine a robot walking around a building:

```text
       A ───── B
       │       │
       │       │
       D ───── C
```

It estimates:

```text
A → B → C → D → A
```

But every motion estimate has a small error.

So the estimated trajectory [drifts](glossary.md#drift), and the final pose does not land back on A:

```text
     A ───── B
              \
    A′         \
     \          \
      D ─────────C
```

Call the poses $x_0$ (A), $x_1$ (B), $x_2$ (C), $x_3$ (D) and $x_4$ (A′, the estimate of the final pose). The step D → A′ is just another odometry constraint. When the robot is back at A, the **front-end detects a loop closure**:

> "This place looks like somewhere I've seen before."

It generates a constraint between the *non-consecutive* poses $x_0$ and $x_4$: a measured relative pose $\tilde T_{04}$, which is close to the identity if the robot returned to exactly the same spot.

The **back-end then optimizes the entire trajectory** so that all constraints are satisfied as well as possible.

**Tiny example** (toy case: translation only, $x_0$ anchored, odometry steps weighted equally, loop closure treated as exact):

- The loop closure says $x_4$ should sit back at $x_0$, but the drifted estimate is 0.4 m away.
- That 0.4 m is spread over the 4 odometry steps, 0.1 m per step.
- So $x_1, x_2, x_3, x_4$ move back by 0.1, 0.2, 0.3, 0.4 m. Later poses carry more of the drift, so they move more.

![Loop-closure toy case: on the left the drifted square x0 to x4 with x4 0.4 m from x0 and the dashed loop-closure constraint between them; on the right the same poses after the correction, with x1, x2, x3 and x4 moved back by 0.1, 0.2, 0.3 and 0.4 m so that x4 coincides with x0](images/frontend_backend_loop_closure.svg)

*Figure: the toy case above, to scale in the drift and the corrections (the 2 m square and the drift direction are arbitrary); same generator as the figure in the introduction.*

In practice, "loop closure" spans both stages. The front-end place-recognition module (which matches the current view against stored ones) detects the candidate match. The back-end checks it against the other constraints (robust costs down-weight a closure that disagrees strongly with them; see [§4](#4-front-end-vs-back-end)) and folds it into the global optimization.

So loop closure belongs fully to neither stage, and a diagram can draw it in either one.

---

## 4. Front-end vs Back-end

|                    | Front-end                                        | Back-end                                |
| ------------------ | ------------------------------------------------ | --------------------------------------- |
| Main job           | Understand sensor observations                   | Find globally consistent solution       |
| Input              | Raw sensor data                                  | Measurements/constraints                |
| Output             | Features, matches, relative poses, loop-closure candidates| Optimized poses/map; verifies and applies loop closures |
| Typical algorithms | Feature tracking, VO, matching                   | BA, pose graph, factor graph            |
| Focus              | Local/sequential                               | Global/accumulated                    |

### Front-end errors become constraints for the back-end
If data association or motion estimation is wrong, a plain least-squares optimizer tries to satisfy the bad constraint and can converge to a wrong solution. Back-ends therefore often use robust costs ([glossary](glossary.md#outlier-and-robust-kernel); Huber, Cauchy, Dynamic Covariance Scaling (DCS)) that down-weight constraints that disagree strongly with the rest; see [optimization/pose_graph_optimization.md §16](optimization/pose_graph_optimization.md#16-robust-loss-functions-used-to-handle-false-loop-closures).

---

## 5. Front-end ≠ real-time, back-end ≠ offline

The front-end **doesn't necessarily mean "real-time"**, and the back-end **doesn't necessarily mean "offline."** (Offline means run after all the data is collected; online means run while data arrives.)

In modern SLAM systems, both stages can operate online (see the pipeline in §1).

---

## 6. References

1. Cadena, C., Carlone, L., Carrillo, H., Latif, Y., Scaramuzza, D., Neira, J., Reid, I., & Leonard, J. J. (2016). *Past, Present, and Future of Simultaneous Localization and Mapping: Toward the Robust-Perception Age*. IEEE Transactions on Robotics, 32(6), 1309–1332. https://doi.org/10.1109/TRO.2016.2624754 - the standard survey that frames the front-end/back-end split used throughout this doc (§1, §2, §4).
2. Grisetti, G., Kümmerle, R., Stachniss, C., & Burgard, W. (2010). *A Tutorial on Graph-Based SLAM*. IEEE Intelligent Transportation Systems Magazine, 2(4), 31–43. https://doi.org/10.1109/MITS.2010.939925 - the back-end/pose-graph optimization tutorial behind §2 and §4.
3. Mur-Artal, R., Montiel, J. M. M., & Tardós, J. D. (2015). *ORB-SLAM: A Versatile and Accurate Monocular SLAM System*. IEEE Transactions on Robotics, 31(5), 1147–1163. https://doi.org/10.1109/TRO.2015.2463671 - a concrete system using ORB features, local BA and pose-graph loop closing; its Tracking / Local Mapping / Loop Closing threads only roughly follow the front-end/back-end split. Relevant to §1's ORB mention and as a worked loop-closure example for §3.
4. Gálvez-López, D., & Tardós, J. D. (2012). *Bags of Binary Words for Fast Place Recognition in Image Sequences*. IEEE Transactions on Robotics, 28(5), 1188–1197. https://doi.org/10.1109/TRO.2012.2197158 - the DBoW2 place-recognition method, an example of the place-recognition step in §3's account of loop-closure *detection*.
