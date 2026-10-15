# Docs index

Conceptual notes on the state-estimation and SLAM ideas behind the code in [`use_numpy/`](../use_numpy/) and [`use_manif/`](../use_manif/). Docs are grouped by role and, within each group, listed in reading order (later ones lean on earlier ones).

## Start here

- [frontend_backend.md](frontend_backend.md): the **front-end**/**back-end** split in SLAM; the widest overview of a SLAM pipeline.
- [filtering_smoothing.md](filtering_smoothing.md): **filtering** (EKF-style) vs. **optimization/smoothing** (factor-graph-style), the two strategies behind the `filtering/` and `optimization/` groups below.
- [slam_mental_map.md](slam_mental_map.md): every doc and script on one map, with cross-cutting ideas and suggested reading paths. Read it after the two docs above.
- [glossary.md](glossary.md): short definitions of the terms shared across these docs, each linked to the section that explains it.

## [`foundations/`](foundations/) - Reusable math, not SLAM-specific

- [jacobian.md](foundations/jacobian.md): what a Jacobian means and why linearization needs one.
- [lie_algebra.md](foundations/lie_algebra.md): Lie groups and algebras for optimizing on $SE(3)$ without singularities.
- [quaternion.md](foundations/quaternion.md): quaternions as a 3D-orientation representation.
- [umeyama_alignment.md](foundations/umeyama_alignment.md): the closed-form best-fit scale, rotation and translation between two point sets, used to align a reconstruction onto ground truth before measuring error.

## [`frontend/`](frontend/) - Turning raw sensor data into constraints for the back-end

- [triangulation_pnp.md](frontend/triangulation_pnp.md): recovering a 3D point from known camera poses (triangulation, implemented in `bundle_adjustment_advanced.py`) and the inverse, recovering a camera pose from known 3D points (PnP, implemented in `pnp_estimation.py`).
- [vi_initialization.md](frontend/vi_initialization.md): bootstrapping scale, gravity direction, and initial velocity and bias, which IMU preintegration and factor-graph optimization need before they can run. Conceptual only, no script.

## [`filtering/`](filtering/) - The Kalman-filter lineage

- [kf_ekf_iekf.md](filtering/kf_ekf_iekf.md): KF → EKF → Invariant EKF, in one progression.
- [linear_nonlinear.md](filtering/linear_nonlinear.md): how to tell whether a system is linear or nonlinear, and whether the nonlinearity matters over the region your uncertainty covers.
- [extra_kf_variants.md](filtering/extra_kf_variants.md): UKF, ESKF, MSCKF and other variants, and why each exists.
- [left_right_invariant.md](filtering/left_right_invariant.md): left- vs. right-invariant error formulations in the IEKF (builds on `kf_ekf_iekf.md` §3-§5).
- [pointcloud_pose_tracking_empirical_note.md](filtering/pointcloud_pose_tracking_empirical_note.md): an empirical note (not a concept explainer) on why `run_ekf`/`run_iekf` agree to round-off in `pointcloud_pose_tracking.py` while `run_ukf` and `run_vanilla_kf` differ.
- [hybrid_saltation_ekf.md](filtering/hybrid_saltation_ekf.md): filtering through *discontinuous* motion (hybrid systems, guards, reset maps) with the saltation matrix; implemented in `saltation_matrix_ekf.py`.
- [invariant_saltation_ekf.md](filtering/invariant_saltation_ekf.md): the same bounce on a Lie group: a right-invariant EKF on $`SE_2(3)`$ for a tumbling IMU body. Covers the saltation matrix in invariant-error coordinates, why the reset breaks the filter's [trajectory independence](glossary.md#group-affine-dynamics-trajectory-independence), and a timing-noise term that keeps it consistent under contact-detection jitter. Implemented in `invariant_saltation_ekf.py`.
- [inchworm_zupt_ekf.md](filtering/inchworm_zupt_ekf.md): a known-schedule anchor/dwell gait as a switched-linear system, and when to apply zero-velocity updates (ZUPT: correctly, always, or never); implemented in `inchworm_zupt_ekf.py`.
- [friction_anisotropic_ekf.md](filtering/friction_anisotropic_ekf.md): a crawling unicycle on a friction-anisotropic pad, where process-noise variance depends on heading relative to the pad's axis; implemented in `friction_anisotropic_ekf.py`.
- [hexapod_gait_ekf.md](filtering/hexapod_gait_ekf.md): planar hexapod leg odometry, where each stance foot measures the body's velocity and turn rate; how much the gait (tripod, ripple, wave) changes the estimate, catching a slipping leg with a per-foot $`\chi^2`$ test, random foot drift, and contact-timing jitter (which hurts fast-swing gaits most); implemented in `hexapod_gait_ekf.py`.
- [hexapod_contact_inekf.md](filtering/hexapod_contact_inekf.md): the 3D, IMU-driven follow-up, the contact-aided invariant EKF on $`SE_{K+2}(3)`$ with each stance foot's position in the state; yaw is unobservable, and only the invariant EKF's covariance respects that (the world-frame EKF reaches NEES 481 at a 0.3 rad initial rotation error), plus a slipping leg and IMU biases; implemented in `hexapod_contact_inekf.py`.

## [`optimization/`](optimization/) - The factor-graph/smoothing lineage

- [nonlinear_least_square.md](optimization/nonlinear_least_square.md): the problem every solver below solves: nonlinear least squares as a formulation, distinct from any particular algorithm.
- [gauss_newton.md](optimization/gauss_newton.md): the core linearize-and-solve algorithm for nonlinear least squares.
- [levenberg_marquardt.md](optimization/levenberg_marquardt.md): Gauss-Newton with Marquardt damping, whose strength adapts to whether each step actually lowered the cost.
- [factor_graph.md](optimization/factor_graph.md): the graph structure (variables and factors) in which SLAM's least-squares problems are usually organized.
- [pose_graph_optimization.md](optimization/pose_graph_optimization.md): factor graphs specialized to poses-only, relative-constraint problems.
- [bundle_adjustment.md](optimization/bundle_adjustment.md): factor graphs specialized to joint camera-pose and 3D-landmark refinement.
- [sparse_cholesky_factorization.md](optimization/sparse_cholesky_factorization.md): why real solvers exploit the sparsity of the Hessian $`H`$ ([glossary](glossary.md#normal-equations-and-hessian)) in a [Cholesky factorization](glossary.md#cholesky-factorization) (this repo's solves are dense), and the fill-in that variable-elimination order controls.
- [imu_preintegration.md](optimization/imu_preintegration.md): compressing raw high-rate IMU samples into one relative-motion edge, with bias-sensitivity Jacobians that correct it for bias changes without re-integrating.
- [isam_optimization.md](optimization/isam_optimization.md): incrementally updating the solution to a growing factor graph (iSAM) instead of re-solving from scratch each step. Implemented in `pose_graph_incremental.py` (iSAM v1 style: Givens-rotation QR updates for odometry edges, a full rebuild at loop closure, no reordering, no Bayes tree).
- [elimination_tree.md](optimization/elimination_tree.md): the dependency structure ("what must be computed before what") that variable elimination produces, distinct from the factor graph itself ("who talks to whom"); the setup for the Bayes tree.
- [bayes_tree.md](optimization/bayes_tree.md): the tree representation of variable-elimination order that iSAM2 uses to find which part of the solution a new factor affects. The underlying elimination tree (one node per variable, not merged into cliques) is built from this repo's pose graph in `bayes_tree_construction.py`.
- [isam2_optimization.md](optimization/isam2_optimization.md): iSAM's successor, adding a Bayes-tree factorization, selective relinearization and dynamic variable reordering. Only the elimination tree underlying the Bayes tree is built in this repo (`bayes_tree_construction.py`); the rest is conceptual only.
- [marginalization.md](optimization/marginalization.md): the same variable-elimination step, used to permanently discard an old state instead of reordering a solve. It is the mechanism behind sliding-window/fixed-lag smoothing (e.g. VINS-Mono; MSCKF is a filter that uses a related window). Implemented in `sliding_window_marginalization.py`, with a measured bounded-vs-unbounded memory and time comparison against full-batch re-solving.

## [`images/`](images/)

Diagrams referenced by the docs above, named `<doc>_<n>.jpg`. Each image's source is checked in its owning doc's "Image sources" section.
