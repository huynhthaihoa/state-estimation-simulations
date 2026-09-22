# Docs index

Conceptual/pedagogical notes on the state-estimation and SLAM ideas behind the code in[`use_numpy/`](../use_numpy/) and [`use_manif/`](../use_manif/). Grouped by role below; within each group, docs are listed in a sensible reading order (later ones lean on earlier ones).

## Start here

- [frontend_backend.md](frontend_backend.md): the **front-end**/**back-end** split in SLAM; the widest-angle overview, and a map of how everything else here fits together.
- [filtering_smoothing.md](filtering_smoothing.md): **filtering** (EKF-style) vs. **optimization/smoothing** (factor-graph-style) as the two general strategies for state estimation; the other natural entry point, and the one that motivates why both the `filtering/` and `optimization/` groups below exist.

## [`foundations/`](foundations/) - Reusable math, not SLAM-specific

- [jacobian.md](foundations/jacobian.md): what a Jacobian means and why linearization needs one.
- [lie_algebra.md](foundations/lie_algebra.md): Lie groups/algebras for optimizing on $SE(3)$ without singularities.
- [quaternion.md](foundations/quaternion.md): quaternions as a 3D-orientation representation.
- [umeyama_alignment.md](foundations/umeyama_alignment.md): the closed-form best-fit scale+rotation+translation between two point sets, used to align a reconstruction onto ground truth before measuring error.

## [`frontend/`](frontend/) - Turning raw sensor data into constraints for the back-end

- [triangulation_pnp.md](frontend/triangulation_pnp.md): recovering a 3D point from known camera poses (triangulation, implemented for real in `bundle_adjustment_advanced.py`) and its exact inverse - recovering a camera pose from known 3D points (PnP, implemented in `pnp_estimation.py`).
- [vi_initialization.md](frontend/vi_initialization.md): bootstrapping scale, gravity direction, and initial velocity/bias before IMU preintegration and factor-graph optimization can run at all; conceptual only, no accompanying script.

## [`filtering/`](filtering/) - The Kalman-filter lineage

- [kf_ekf_iekf.md](filtering/kf_ekf_iekf.md): KF → EKF → Invariant EKF, in one progression.
- [extra_kf_variants.md](filtering/extra_kf_variants.md): UKF, ESKF, MSCKF and other variants, and why each exists.
- [left_right_invariant.md](filtering/left_right_invariant.md): left- vs. right-invariant error formulations in the IEKF (builds on `extra_kf_variants.md`).
- [pointcloud_pose_tracking_empirical_note.md](filtering/pointcloud_pose_tracking_empirical_note.md): an empirical note (not a concept explainer) on why `run_ekf`/`run_iekf` produce bit-identical output in `pointcloud_pose_tracking.py`, why `run_ukf` only comes close but doesn't match either one exactly, and why `run_vanilla_kf` diverges from all three unconditionally, by changing the state representation rather than the linearization.
- [hybrid_saltation_ekf.md](filtering/hybrid_saltation_ekf.md): the one filtering-lineage doc about *discontinuous* motion rather than continuous motion — what a hybrid dynamical system/guard/reset map is, the saltation matrix that correctly propagates covariance through a discrete event (including a plausible-looking formula that turned out wrong, caught by finite-difference verification), and an empirical finding in `saltation_matrix_ekf.py` that runs against the naive expectation: the mathematically-exact saltation correction is *more* fragile to real tracking error than the naive reset-Jacobian-only update, not less — quantified further in §8 once contact-detection timing itself is made uncertain.
- [inchworm_zupt_ekf.md](filtering/inchworm_zupt_ekf.md): a known-schedule anchor/dwell (inchworm-style) gait — not a hybrid reset, but a switched-linear system where the interesting question is purely on the measurement side: exploiting the free zero-velocity information a stationary anchor phase provides (ZUPT), correctly, always, or never, implemented in `inchworm_zupt_ekf.py`. Same "counterintuitive empirical finding" pattern as the saltation doc: misusing ZUPT poisons consistency well beyond the phase it was misapplied in, and even *correct* ZUPT usage is measurably more fragile (though more accurate) than never using it at all.
- [friction_anisotropic_ekf.md](filtering/friction_anisotropic_ekf.md): a crawling unicycle on a friction-anisotropic pad, where process-noise variance depends on heading relative to the pad's own fixed axis — implemented in `friction_anisotropic_ekf.py`, applying `pointcloud_pose_tracking_empirical_note.md`'s isotropic/anisotropic-and-rotating-frame argument (there about measurement noise) to process noise instead. Knowing the pad is anisotropic but pointing that anisotropy at a fixed, stale heading is dramatically worse than not claiming any direction at all, and the worst mismatch sits at a 90° heading difference, not 180° — a covariance ellipse has period $\pi$, not $2\pi$.

## [`optimization/`](optimization/) - The factor-graph / smoothing lineage

- [nonlinear_least_square.md](optimization/nonlinear_least_square.md): the problem every solver below is trying to solve: NLS as a formulation, distinct from any particular algorithm.
- [gauss_newton.md](optimization/gauss_newton.md): the core linearize-and-solve algorithm for NLS.
- [levenberg_marquardt.md](optimization/levenberg_marquardt.md): Gauss-Newton with a damping term, for when the initial estimate is untrustworthy.
- [factor_graph.md](optimization/factor_graph.md): the graphical structure (variables + factors) that SLAM's NLS problems are usually organized as.
- [pose_graph_optimization.md](optimization/pose_graph_optimization.md): factor graphs specialized to poses-only, relative-constraint problems.
- [bundle_adjustment.md](optimization/bundle_adjustment.md): factor graphs specialized to joint camera-pose + 3D-landmark refinement.
- [sparse_cholesky_factorization.md](optimization/sparse_cholesky_factorization.md): why the linear solve inside a Gauss-Newton/LM step exploits $H$'s sparsity instead of a dense $LL^T$, and the fill-in problem that variable-elimination order controls.
- [imu_preintegration.md](optimization/imu_preintegration.md): compressing raw high-rate IMU samples into one relative-motion edge, with a right-Jacobian trick to correct it for bias changes without re-integrating.
- [isam_optimization.md](optimization/isam_optimization.md): incrementally updating the solution to a growing factor graph (iSAM) instead of re-solving it from scratch every step.
- [isam2_optimization.md](optimization/isam2_optimization.md): iSAM's successor - adds a Bayes-tree factorization, selective relinearization, and dynamic variable reordering; the tree itself is built in this repo, the rest is conceptual only.
- [elimination_tree.md](optimization/elimination_tree.md): the dependency structure ("what must be computed before what") that variable elimination produces, distinct from the factor graph itself ("who talks to whom") - the setup for the Bayes tree below.
- [bayes_tree.md](optimization/bayes_tree.md): the tree representation of variable-elimination order that iSAM2 relies on to know which part of the solution a new factor actually affects, built for real from this repo's pose graph in `bayes_tree_construction.py`.
- [marginalization.md](optimization/marginalization.md): the same variable-elimination step, aimed at permanently discarding an old state instead of reordering a solve - the mechanism behind sliding-window/fixed-lag smoothing (MSCKF, VINS-Mono), implemented for real in `sliding_window_marginalization.py` with a measured bounded-vs-unbounded memory/time comparison against full-batch re-solving.

## [`images/`](images/)

Diagrams referenced by the docs above, named `<doc>_<n>.jpg`. Each image's originally-claimed source is checked (not assumed) in its owning doc's "Image sources" section before being cited.
