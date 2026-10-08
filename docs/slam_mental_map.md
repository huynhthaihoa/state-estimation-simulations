# SLAM mental map

This doc places every other doc in this folder, and every script in [`use_numpy/`](../use_numpy/)/[`use_manif/`](../use_manif/), on one map. It adds no new theory; each node points to the doc section that explains it. For short definitions of the terms used here, see [glossary.md](glossary.md).

Several docs already draw a partial map of their own corner. We treat those as zoom-ins of this one:

- [extra_kf_variants.md](filtering/extra_kf_variants.md): the Kalman-filter family tree.
- [filtering_smoothing.md §10](filtering_smoothing.md#10-where-the-modern-systems-fit): filtering vs. smoothing, and where real systems sit.
- [nonlinear_least_square.md §15](optimization/nonlinear_least_square.md#15-your-slam-mental-map): factor graph → NLS → GN/LM, batch or incremental.
- [isam_optimization.md §11](optimization/isam_optimization.md#11-where-isam-fits-into-the-slam-mental-map): where iSAM sits, and the $SE(3)$ → Jacobian → factorization chain beneath it.
- [elimination_tree.md §11](optimization/elimination_tree.md#11-the-big-picture): factor graph → Jacobian → sparse linear algebra → elimination tree.

---

## 1. The whole picture

```text
  SENSORS    camera · IMU · point cloud · wheel/visual odometry · contact/phase
                                  │
                                  ▼
┌─ FRONT-END ─ "What happened between these observations?" ─────────────────┐
│  features, matching, data association    triangulation · PnP              │
│  visual-inertial initialization          IMU preintegration               │
│  keyframe selection                      loop-closure detection           │
└─────────────────────────────────┬─────────────────────────────────────────┘
                                  │  constraints: relative poses, pixel
                                  │  observations, IMU bundles, priors
                                  ▼
┌─ BACK-END ─ "What trajectory and map best explain ALL of them?" ──────────┐
│                                                                           │
│                 State estimation = probabilistic inference                │
│                                  │                                        │
│            ┌─────────────────────┴─────────────────────┐                  │
│            │                                           │                  │
│        FILTERING                                   SMOOTHING              │
│   keep only the current state             keep (a window of) the history  │
│            │                                           │                  │
│   KF → EKF → IEKF                          factor graph = NLS problem     │
│   UKF · ESKF · MSCKF                       PGO · BA · IMU factors         │
│   hybrid resets · ZUPT · shaped Q          GN/LM         (batch)          │
│                                            iSAM/iSAM2    (incremental)    │
│                                            marginalization (bounded)      │
└─────────────────────────────────┬─────────────────────────────────────────┘
                                  │
                                  ▼
  OUTPUT      trajectory + map
  EVALUATION  Umeyama alignment → RMS error  ·  NEES consistency

  FOUNDATIONS (used at every layer above)
  Jacobian · Lie groups SO(3)/SE(3) · quaternion · linear vs. nonlinear
```

In the diagram, $`Q`$ is the **process-noise covariance** of the filter's predict step ([glossary](glossary.md#6-kalman-filter-family)).

Three questions organize the whole map:

1. **What does the front-end hand over?** Constraints, not a final answer. See [frontend_backend.md §1](frontend_backend.md#1-slam-front-end---extracting-constraints).
2. **Does the back-end keep the past?** Filtering folds it away, smoothing keeps it and can revise it. See [filtering_smoothing.md §9](filtering_smoothing.md#9-the-precise-distinction).
3. **How is the smoothing problem solved?** Linearize, then solve a sparse linear system, either from scratch (batch) or by updating the previous solution (incremental). See [nonlinear_least_square.md §9](optimization/nonlinear_least_square.md#9-this-is-exactly-where-gaussnewton-comes-from) and [isam_optimization.md §2](optimization/isam_optimization.md#2-the-incremental-idea).

---

## 2. Foundations: the math every layer uses

```text
            linear vs. nonlinear?  ── "does the nonlinearity matter over my uncertainty?"
                     │
                     ▼
               Jacobian J  ─────────────  local linear model of a nonlinear function
                     │
   states that live on a curved space (rotations, poses):
                     │
     quaternion ── SO(3)/SE(3) Lie group ── Exp/Log ── tangent space (Lie algebra)
                     │
                     ▼
     "perturb on the manifold, compute in the tangent space"
     → used by: EKF/IEKF, GN/LM on poses, PGO residuals, IMU preintegration
```

| Concept | What it gives | Explained in | Used by |
|---|---|---|---|
| Linear vs. nonlinear | When a first-order model is good enough | [linear_nonlinear.md §3](filtering/linear_nonlinear.md#3-how-nonlinear-matters-in-practice), [§4](filtering/linear_nonlinear.md#4-how-do-you-know-whether-its-nonlinear-over-your-uncertainty-region) | choosing KF vs. EKF vs. UKF |
| Jacobian | The local linear model: how errors change when states move | [jacobian.md §2](foundations/jacobian.md#2-now-imagine-multiple-inputs-and-outputs), [§5](foundations/jacobian.md#5-why-does-ekf-need-it) | EKF, GN, LM, bias correction, saltation matrix |
| Lie group/algebra | Updating rotations and poses without leaving the manifold | [lie_algebra.md §5-§9](foundations/lie_algebra.md#5-lie-group-the-space-of-valid-transformations), [§11](foundations/lie_algebra.md#11-se3-this-is-where-robotics-gets-really-interesting) | IEKF, PGO, BA, preintegration |
| Left/right Jacobians | Jacobians on a curved space | [jacobian.md §11](foundations/jacobian.md#11-left-and-right-jacobians-sensitivity-on-a-curved-space), [lie_algebra.md §15](foundations/lie_algebra.md#15-the-connection-to-jacobians) | PGO error terms, preintegration |
| Quaternion | A compact, singularity-free orientation | [quaternion.md §2](foundations/quaternion.md#2-the-intuitive-idea-a-quaternion-is-an-axis--angle-in-disguise), [§11](foundations/quaternion.md#11-the-connection-to-kalman-filtering) | orientation states, output poses |
| Umeyama alignment | Removing gauge freedom before measuring error | [umeyama_alignment.md §1](foundations/umeyama_alignment.md#1-why-an-alignment-step-is-needed-at-all) | evaluating BA/SLAM output |

---

## 3. Front-end: from raw sensors to constraints

```text
 camera images ──► features/matching ──► relative motion  T_ij ───────────┐
                           │                                              │
                           ├──► triangulation   (known poses → 3D point)  │
                           ├──► PnP             (known 3D points → pose)  ├──► constraints
                           └──► loop-closure detection (place revisited) ─┤    for the
                                                                          │    back-end
 IMU samples (100+ Hz) ──► preintegration  (ΔR, Δv, Δp + bias Jacobians) ─┤
                                                                          │
 camera + IMU at start-up ──► VI initialization (scale, gravity, v, b_g) ─┘
                              (seeds the first linearization point)
```

Here $`\Delta R, \Delta v, \Delta p`$ are the preintegrated rotation, velocity and position changes, $`v`$ is the velocity, and $`b_g`$ is the gyro bias ([glossary](glossary.md#5-slam-system)).

| Step | What it produces | Doc | Script |
|---|---|---|---|
| Features, data association, keyframes | Tracked observations $`z_{ij}`$ (pixel measurement of landmark $`j`$ from pose $`i`$) | [frontend_backend.md §1](frontend_backend.md#1-slam-front-end---extracting-constraints) | - |
| Triangulation | A 3D landmark from known poses | [triangulation_pnp.md §2](frontend/triangulation_pnp.md#2-triangulation-worked-from-the-real-code) | `bundle_adjustment_advanced.py` |
| PnP | A camera pose from known landmarks | [triangulation_pnp.md §3](frontend/triangulation_pnp.md#3-pnp-as-the-inverse-problem) | `pnp_estimation.py` |
| Loop-closure detection | A constraint between non-consecutive poses | [frontend_backend.md §3](frontend_backend.md#3-example-loop-closure) | `pose_graph.py` (the constraint is given, not detected) |
| IMU preintegration | One relative-motion bundle per keyframe pair (bundle + bias Jacobians + $O(1)$ correction) | [imu_preintegration.md §3](optimization/imu_preintegration.md#3-what-gets-compressed), [§6](optimization/imu_preintegration.md#6-where-this-fits-in-a-slam-back-end) | `imu_preintegration.py` |
| VI initialization | Scale, gravity, velocities, gyro bias | [vi_initialization.md §2](frontend/vi_initialization.md#2-the-classic-linear-alignment-pipeline), [§4](frontend/vi_initialization.md#4-the-hand-off) | - (conceptual only) |

Two things sit on the boundary:

- **Loop closure** is detected by the front-end but verified and applied by the back-end ([frontend_backend.md §3](frontend_backend.md#3-example-loop-closure)).
- **IMU preintegration** lives in `optimization/` because its output is a back-end factor, but its job is to compress raw sensor data, which is front-end work.

---

## 4. Back-end, branch A: filtering

A filter carries only the current state and its covariance. Each new measurement is folded in, and the past is marginalized away ([filtering_smoothing.md §2](filtering_smoothing.md#2-filtering-one-step-at-a-time)).

```text
                         Bayes filter (predict → update)
                                     │
                     ┌───────────────┴────────────────┐
                Gaussian belief                   any belief
                     │                                │
         ┌───────────┴──────────┐                Particle Filter
      linear               nonlinear
         │              ┌───────┴────────┐
        KF             EKF         UKF (sigma points)
                        │
          ┌─────────────┼──────────────────┐
         ESKF          IEKF             MSCKF
     (error state)  (error on the     (sliding window of poses,
                     Lie group)        landmarks marginalized)

   When the MODEL itself is unusual (this repo's robot-motion toys):
     discrete jumps          → saltation matrix       (hybrid_saltation_ekf)
     known stationary phases → phase-gated ZUPT       (inchworm_zupt_ekf)
     direction-dependent Q   → heading-aware Q        (friction_anisotropic_ekf)
```

| Topic | Key idea | Doc | Script |
|---|---|---|---|
| KF → EKF → IEKF | Linear, then linearized, then geometry-aware error | [kf_ekf_iekf.md §1-§3](filtering/kf_ekf_iekf.md#1-standard-kalman-filter-everything-is-nicely-linear), [§5](filtering/kf_ekf_iekf.md#5-the-really-important-difference-how-do-you-define-error) | `pointcloud_pose_tracking.py` |
| Left- vs. right-invariant error | Which frame the error is defined in, and why it matters | [left_right_invariant.md §3](filtering/left_right_invariant.md#3-why-invariant---and-why-leftright) | `pointcloud_pose_tracking.py` (`run_iekf`) |
| UKF, ESKF, MSCKF, PF, ... | Each relaxes one KF assumption | [extra_kf_variants.md §1](filtering/extra_kf_variants.md#1-unscented-kalman-filter-ukf), [§2](filtering/extra_kf_variants.md#2-error-state-kalman-filter-eskf), [§7](filtering/extra_kf_variants.md#7-particle-filter-pf), [kf_ekf_iekf.md §9](filtering/kf_ekf_iekf.md#9-other-prominent-variants-worth-keeping-in-mind) (MSCKF) | `pointcloud_pose_tracking.py` (`run_ukf`; UKF only) |
| What differs in practice | EKF = IEKF (same update $K r$, the **Kalman gain** times the innovation, and covariance $`P`$ to round-off; [glossary](glossary.md#6-kalman-filter-family)); UKF ~1.4 mm at the first update, micrometers after; vanilla KF ~1-4 mm off from truncating the motion step to first order | [pointcloud_pose_tracking_empirical_note.md §2-§4](filtering/pointcloud_pose_tracking_empirical_note.md#2-ekf-vs-iekf-exact-by-construction) | `pointcloud_pose_tracking.py` |
| Hybrid events | Propagating covariance through a reset | [hybrid_saltation_ekf.md §2](filtering/hybrid_saltation_ekf.md#2-why-the-reset-maps-own-jacobian-is-not-enough), [§6](filtering/hybrid_saltation_ekf.md#6-the-empirical-finding-a-modest-real-gap-not-a-dramatic-one) | `saltation_matrix_ekf.py` |
| Phase-gated measurements | Free zero-velocity information, but wrong gating is catastrophic and a confident gated filter is hit harder by unmodeled acceleration | [inchworm_zupt_ekf.md §3](filtering/inchworm_zupt_ekf.md#3-the-finding-not-just-always-is-wrong-while-moving) | `inchworm_zupt_ekf.py` |
| Shaped process noise | A wrongly oriented $Q$ is worse than an isotropic one | [friction_anisotropic_ekf.md §2](filtering/friction_anisotropic_ekf.md#2-the-dominant-finding-fixed_anisotropic-is-dramatically-worse-everywhere), [§3](filtering/friction_anisotropic_ekf.md#3-a-secondary-finding-the-worst-mismatch-is-90-not-180) | `friction_anisotropic_ekf.py` |

---

## 5. Back-end, branch B: smoothing and optimization

Smoothing keeps past states as variables and re-solves them jointly. The chain from problem to solver is:

```text
  PROBLEM     factor graph: variables (poses, landmarks, v, biases) + factors
                    │        pose graph = poses only; BA = poses + landmarks
                    ▼
              nonlinear least squares:  min_x  Σ ‖r_i(x)‖²_Ω
                    │
  SOLVER            ▼
              linearize at the current estimate (Jacobians, on the manifold)
                    │
                    ▼
              normal equations  H Δx = -b      ◄── LM adds damping λ to H
                    │
  LINEAR            ▼
  ALGEBRA     sparse Cholesky/QR  (ordering controls fill-in)
                    │
                    ▼
              retract: x ← x ⊞ Δx, repeat until converged
                    │
        ┌───────────┼─────────────────────────┐
        ▼           ▼                         ▼
      BATCH     INCREMENTAL                BOUNDED
    re-solve    update the previous       keep a window,
    everything  factorization             marginalize the rest
     (GN/LM)    (iSAM → iSAM2)            (sliding window/fixed lag)
```

In the chain, $`r_i`$ is the residual of factor $`i`$ and $`\Omega`$ its **information matrix** ([glossary](glossary.md#2-uncertainty-and-probability)); $`H`$ is the **Gauss-Newton Hessian** (approximate), $`b`$ the gradient, and $`\Delta x`$ the step ([glossary](glossary.md#3-least-squares-optimization)); $`\boxplus`$ is the **retraction**, the manifold-aware "add" ([glossary](glossary.md#1-geometry-and-lie-groups)).

The bottom row is the solve strategy. GN/LM is the step rule, a separate axis: iSAM is another way to run the linear solves, not an alternative to GN/LM ([levenberg_marquardt.md §11](optimization/levenberg_marquardt.md#11-where-lm-fits)).

### 5.1 The problem

| Topic | Key idea | Doc | Script |
|---|---|---|---|
| Nonlinear least squares | The problem, independent of any solver | [nonlinear_least_square.md §4](optimization/nonlinear_least_square.md#4-the-general-form), [§14](optimization/nonlinear_least_square.md#14-why-slam-naturally-becomes-nls) | - |
| Factor graph | Who constrains whom; the total cost is the sum of factor errors | [factor_graph.md §3](optimization/factor_graph.md#3-a-factor-is-basically-an-error-function), [§6](optimization/factor_graph.md#6-the-really-important-distinction-variable-vs-factor) | - |
| Pose-graph optimization | Poses + relative constraints; loop closures spread the drift conflict over the loop | [pose_graph_optimization.md §4-§5](optimization/pose_graph_optimization.md#4-the-really-important-event-loop-closure), [§15](optimization/pose_graph_optimization.md#15-mathematical-breakdown-of-the-error-formulation-and-lie-algebra-operations) | `pose_graph.py` |
| Robust kernels | Down-weight outliers such as false loop closures | [pose_graph_optimization.md §16](optimization/pose_graph_optimization.md#16-robust-loss-functions-used-to-handle-false-loop-closures) | - |
| Bundle adjustment | Poses + landmarks against reprojection error | [bundle_adjustment.md §4](optimization/bundle_adjustment.md#4-heres-the-important-part-both-cameras-and-points-are-adjusted), [§9](optimization/bundle_adjustment.md#9-ba-vs-pose-graph-optimization) | `bundle_adjustment.py` |
| Local vs. global BA | A covisibility window every keyframe, a full re-solve occasionally | [bundle_adjustment.md §13](optimization/bundle_adjustment.md#13-local-vs-global-bundle-adjustment-real-systems) | `bundle_adjustment_advanced.py` |
| IMU factors | Preintegrated edges between (pose, v, bias) nodes | [imu_preintegration.md §6](optimization/imu_preintegration.md#6-where-this-fits-in-a-slam-back-end) | - (the script stops before building the factor) |

### 5.2 The solver

| Topic | Key idea | Doc | Script |
|---|---|---|---|
| Gauss-Newton | Linearize, solve the normal equations, repeat | [gauss_newton.md §4](optimization/gauss_newton.md#4-the-optimization-problem), [§8](optimization/gauss_newton.md#8-why-is-this-everywhere-in-slam) | `bundle_adjustment.py`, `pnp_estimation.py`, `robot_imu_simulation.py`, `pointcloud_pose_tracking.py` (batch GN) |
| Levenberg-Marquardt | GN with damping, so a bad linear model can't throw the step; still local | [levenberg_marquardt.md §4](optimization/levenberg_marquardt.md#4-the-damping-idea), [§6](optimization/levenberg_marquardt.md#6-how-does-lm-decide-whether-to-be-cautious) | `pose_graph.py`, `bundle_adjustment_advanced.py` |
| Sparse Cholesky | Exploit sparsity; ordering controls fill-in | [sparse_cholesky_factorization.md §3](optimization/sparse_cholesky_factorization.md#3-the-surprising-part-fill-in), [§5](optimization/sparse_cholesky_factorization.md#5-ordering-becomes-extremely-important), [§7](optimization/sparse_cholesky_factorization.md#7-factor-graph--hessian--sparse-cholesky) | - |
| Schur complement (BA) | Eliminate landmarks, solve a camera-only system | [bundle_adjustment.md §12](optimization/bundle_adjustment.md#12-block-sparsity-and-the-schur-complement) | - |

### 5.3 Batch, incremental or bounded

| Topic | Key idea | Doc | Script |
|---|---|---|---|
| iSAM | Update the factorization row by row; relinearize occasionally (this repo: full rebuild at the loop closure, no reordering) | [isam_optimization.md §2-§3](optimization/isam_optimization.md#2-the-incremental-idea), [§9](optimization/isam_optimization.md#9-isam-doesnt-mean-never-touch-old-variables) | `pose_graph_incremental.py` |
| Elimination tree | What must be eliminated before what | [elimination_tree.md §4](optimization/elimination_tree.md#4-the-really-useful-intuition-information-flows-upward), [§7](optimization/elimination_tree.md#7-this-becomes-very-important-for-isam2) | `bayes_tree_construction.py` |
| Bayes tree | Cliques; a new factor only touches its path to the root | [bayes_tree.md §7-§8](optimization/bayes_tree.md#7-heres-where-it-becomes-powerful-for-isam2), [§13](optimization/bayes_tree.md#13-one-more-important-concept-cliques) | `bayes_tree_construction.py` (elimination tree only) |
| iSAM2 | Bayes tree + selective relinearization + reordering | [isam2_optimization.md §6-§10](optimization/isam2_optimization.md#6-but-what-makes-isam2-special), [§13](optimization/isam2_optimization.md#13-the-complete-isam2-picture) | - (conceptual only) |
| Marginalization | Drop old states, keep their information as a prior | [marginalization.md §2](optimization/marginalization.md#2-marginalization-keep-the-information-drop-the-variable), [§4](optimization/marginalization.md#4-the-math-turning-an-eliminated-pose-into-a-prior), [§6](optimization/marginalization.md#6-the-consistency-gotcha-why-fej-exists) | `sliding_window_marginalization.py` |

---

## 6. The threads that run through every layer

The map is a tree, but a few ideas cut across its branches.

### 6.1 Linearization

Every estimator here that faces a nonlinear model, except the sigma-point UKF and the PF, replaces it with a Jacobian at some point:

```text
  EKF predict/update ─┐
  GN/LM step ─────────┤
  IMU bias correction ┼──►  f(x + Δx) ≈ f(x) + J Δx   (jacobian.md)
  saltation matrix ───┤
  marginalization ────┘     ...and the linearization point matters (FEJ, relinearization)
```

The EKF linearizes once per predict/update step, at the current estimate. GN/LM re-linearize every iteration. iSAM2 re-linearizes only variables that moved enough ([isam2_optimization.md §10](optimization/isam2_optimization.md#10-selective-relinearization)). Marginalization freezes a linearization point for good in general; this repo's prior re-linearizes the Jacobian and freezes only $\Omega$ and $X_\mathrm{ref}$ (the prior's frozen reference pose) ([marginalization.md §6](optimization/marginalization.md#6-the-consistency-gotcha-why-fej-exists)).

### 6.2 Where the error is defined

The same question appears on both branches: in which frame, and on which space, is the error measured?

- Filtering: EKF vs. IEKF differ only in how they define the error ([kf_ekf_iekf.md §5](filtering/kf_ekf_iekf.md#5-the-really-important-difference-how-do-you-define-error), [left_right_invariant.md §4](filtering/left_right_invariant.md#4-why-it-actually-matters-not-just-bookkeeping)).
- Smoothing: PGO's residual is the $\mathrm{Log}$ of a relative-pose mismatch ([pose_graph_optimization.md §7](optimization/pose_graph_optimization.md#7-the-mathematics-is-actually-quite-intuitive)).

### 6.3 Variable elimination: one operation, three uses

The Schur complement/variable elimination shows up three times with three different purposes ([marginalization.md §3](optimization/marginalization.md#3-three-flavors-of-elimination-compared)):

```text
                    variable elimination (Schur complement)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
   STRUCTURAL                  SOLVE-ORDER                  TEMPORAL
   BA landmarks                elimination tree →           oldest pose in
   (recovered later by         Bayes tree (iSAM2)           a sliding window
    back-substitution)         (re-eliminated when          (gone for good;
                                affected)                    leaves a prior)
```

A filter is the extreme case of the temporal column: it marginalizes every past state immediately ([filtering_smoothing.md §9](filtering_smoothing.md#9-the-precise-distinction)).

### 6.4 Loop closure

Loop closure is the event that separates SLAM from odometry, and every layer has something to say about it:

| Layer | What happens | Doc |
|---|---|---|
| Front-end | Detects a revisited place and outputs a relative pose | [frontend_backend.md §3](frontend_backend.md#3-example-loop-closure) |
| PGO | The drift conflict is spread over the whole loop | [pose_graph_optimization.md §5](optimization/pose_graph_optimization.md#5-pose-graph-optimization-resolves-the-conflict), [§10](optimization/pose_graph_optimization.md#10-why-loop-closure-is-so-powerful) |
| Robust costs | A false closure is down-weighted | [pose_graph_optimization.md §16](optimization/pose_graph_optimization.md#16-robust-loss-functions-used-to-handle-false-loop-closures) |
| iSAM | The new row sweeps through the factorization and causes fill-in; a batch relinearize-and-reorder step cleans it up | [isam_optimization.md §6](optimization/isam_optimization.md#6-but-what-about-loop-closure) |
| Bayes tree/iSAM2 | The affected path reaches much further toward the root | [bayes_tree.md §8](optimization/bayes_tree.md#8-even-more-interesting-loop-closure), [isam2_optimization.md §9](optimization/isam2_optimization.md#9-but-what-about-loop-closure) |
| Local vs. global BA | Drift the measurements can't see is cured only by a loop or an absolute sensor; Global BA also removes inter-window inconsistency. In this repo's loop run, Local+Global is unstable | [bundle_adjustment.md §13.3](optimization/bundle_adjustment.md#133-what-local-ba-cant-fix-and-what-can), [§13.6](optimization/bundle_adjustment.md#136-in-this-repo) |

### 6.5 Consistency, not just accuracy

Accuracy and covariance calibration are separate properties, and NEES measures the second ([linear_nonlinear.md §4.4](filtering/linear_nonlinear.md#44-after-the-fact-consistency-tests)). Three docs report a calibration problem:

- ZUPT: the one clean accuracy-vs-consistency trade. `phase_conditional` has better velocity but worse position and cruise NEES than `never` ([inchworm_zupt_ekf.md §3](filtering/inchworm_zupt_ekf.md#3-the-finding-not-just-always-is-wrong-while-moving)).
- Hybrid: both filters are underconfident, and the saltation matrix is slightly closer to consistent ([hybrid_saltation_ekf.md §6](filtering/hybrid_saltation_ekf.md#6-the-empirical-finding-a-modest-real-gap-not-a-dramatic-one)).
- Friction: `fixed_anisotropic` is overconfident ([friction_anisotropic_ekf.md §2](filtering/friction_anisotropic_ekf.md#2-the-dominant-finding-fixed_anisotropic-is-dramatically-worse-everywhere)).

---

## 7. Every script on the map

| Script | Layer | Branch/topic | Main doc |
|---|---|---|---|
| `imu_integration_comparison.py` | Foundations | Euler-angle vs. $SO(3)$ exp-map integration | [imu_preintegration.md §7](optimization/imu_preintegration.md#7-what-the-accompanying-scripts-actually-do-and-dont) |
| `robot_imu_simulation.py` | Back-end (toy) | Twist dead reckoning + 1 Hz GN position fix (no accel/bias) | [imu_preintegration.md §7](optimization/imu_preintegration.md#7-what-the-accompanying-scripts-actually-do-and-dont) |
| `imu_preintegration.py` | Front-end → back-end | Bundle + bias Jacobians + $O(1)$ correction (no graph factor) | [imu_preintegration.md](optimization/imu_preintegration.md) |
| `pnp_estimation.py` | Front-end | Pose from known 3D points | [triangulation_pnp.md §3](frontend/triangulation_pnp.md#3-pnp-as-the-inverse-problem) |
| `pointcloud_pose_tracking.py` | Back-end, both branches | EKF/IEKF/UKF/vanilla KF vs. batch GN | [pointcloud_pose_tracking_empirical_note.md](filtering/pointcloud_pose_tracking_empirical_note.md) |
| `saltation_matrix_ekf.py` | Back-end, filtering | Hybrid resets | [hybrid_saltation_ekf.md](filtering/hybrid_saltation_ekf.md) |
| `inchworm_zupt_ekf.py` | Back-end, filtering | Phase-gated ZUPT | [inchworm_zupt_ekf.md](filtering/inchworm_zupt_ekf.md) |
| `friction_anisotropic_ekf.py` | Back-end, filtering | Heading-dependent $Q$ | [friction_anisotropic_ekf.md](filtering/friction_anisotropic_ekf.md) |
| `pose_graph.py` | Back-end, smoothing (batch) | PGO with one loop closure, LM | [pose_graph_optimization.md](optimization/pose_graph_optimization.md) |
| `bundle_adjustment.py` | Back-end, smoothing (batch) | Joint pose + landmark BA, GN | [bundle_adjustment.md](optimization/bundle_adjustment.md) |
| `bundle_adjustment_advanced.py` | Front-end + back-end | Triangulation, local vs. global BA, LM (no Umeyama: raw error carries a similarity offset) | [bundle_adjustment.md §13](optimization/bundle_adjustment.md#13-local-vs-global-bundle-adjustment-real-systems) |
| `pose_graph_incremental.py` | Back-end, smoothing (incremental) | iSAM v1: Givens updates for odometry, full rebuild at loop closure, no reordering | [isam_optimization.md §14](optimization/isam_optimization.md#14-where-this-is-implemented-in-this-repo) |
| `bayes_tree_construction.py` | Back-end, smoothing (incremental) | Elimination tree + affected path | [bayes_tree.md §15](optimization/bayes_tree.md#15-where-this-is-implemented-in-this-repo) |
| `sliding_window_marginalization.py` | Back-end, smoothing (bounded) | Schur-complement marginalization | [marginalization.md §8](optimization/marginalization.md#8-what-this-repo-implements) |

`lie_utils.py` (in `use_numpy/`) and `utils.py` (at the repo root) are shared helpers, not stops on the map; see [README.md §C](../README.md#c-library-structure).

---

## 8. Reading paths

Each path follows one branch of the map from the root to its leaves. Start with [frontend_backend.md](frontend_backend.md) and [filtering_smoothing.md](filtering_smoothing.md) for any of them.

**Filtering.** [linear_nonlinear.md](filtering/linear_nonlinear.md) → [jacobian.md](foundations/jacobian.md) → [kf_ekf_iekf.md](filtering/kf_ekf_iekf.md) → [lie_algebra.md](foundations/lie_algebra.md) → [left_right_invariant.md](filtering/left_right_invariant.md) → [extra_kf_variants.md](filtering/extra_kf_variants.md) → [pointcloud_pose_tracking_empirical_note.md](filtering/pointcloud_pose_tracking_empirical_note.md)

**Batch smoothing.** [nonlinear_least_square.md](optimization/nonlinear_least_square.md) → [gauss_newton.md](optimization/gauss_newton.md) → [levenberg_marquardt.md](optimization/levenberg_marquardt.md) → [factor_graph.md](optimization/factor_graph.md) → [pose_graph_optimization.md](optimization/pose_graph_optimization.md) → [bundle_adjustment.md](optimization/bundle_adjustment.md) → [umeyama_alignment.md](foundations/umeyama_alignment.md)

**Incremental and bounded smoothing.** [sparse_cholesky_factorization.md](optimization/sparse_cholesky_factorization.md) → [isam_optimization.md](optimization/isam_optimization.md) → [elimination_tree.md](optimization/elimination_tree.md) → [bayes_tree.md](optimization/bayes_tree.md) → [isam2_optimization.md](optimization/isam2_optimization.md) → [marginalization.md](optimization/marginalization.md)

**Visual-inertial.** [quaternion.md](foundations/quaternion.md) → [lie_algebra.md](foundations/lie_algebra.md) → [imu_preintegration.md](optimization/imu_preintegration.md) → [triangulation_pnp.md](frontend/triangulation_pnp.md) → [vi_initialization.md](frontend/vi_initialization.md) → [marginalization.md](optimization/marginalization.md)

**Unusual robot motion.** [kf_ekf_iekf.md](filtering/kf_ekf_iekf.md) → [hybrid_saltation_ekf.md](filtering/hybrid_saltation_ekf.md) → [inchworm_zupt_ekf.md](filtering/inchworm_zupt_ekf.md) → [friction_anisotropic_ekf.md](filtering/friction_anisotropic_ekf.md) → [filtering_smoothing.md §11](filtering_smoothing.md#11-and-this-matters-a-lot-for-slam--state-estimation-for-resource-constrained-robots-with-discontinuoushybrid-motion)

---

## 9. One-sentence summary

> **A SLAM system turns sensor data into constraints (front-end), then finds the trajectory and map that best explain them (back-end), either by carrying only the current state forward (filtering) or by keeping the history as a factor graph and solving it as sparse nonlinear least squares, from scratch, incrementally or over a bounded window (smoothing), with Jacobians on Lie groups underneath every step.**
