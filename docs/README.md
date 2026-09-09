# Docs index

Conceptual/pedagogical notes on the state-estimation and SLAM ideas behind the code in[`use_numpy/`](../use_numpy/) and [`use_manif/`](../use_manif/). Grouped by role below; within each group, docs are listed in a sensible reading order (later ones lean on earlier ones).

## Start here

- [frontend_backend.md](frontend_backend.md): the **front-end**/**back-end** split in SLAM;the widest-angle overview, and a map of how everything else here fits together.
- [filtering_smoothing.md](filtering_smoothing.md): **filtering** (EKF-style) vs. **optimization/smoothing** (factor-graph-style) as the two general strategies for state estimation; the other natural entry point, and the one that motivates why both the `filtering/` and `optimization/` groups below exist.

## [`foundations/`](foundations/) - Reusable math, not SLAM-specific

- [jacobian.md](foundations/jacobian.md): what a Jacobian means and why linearization needs one.
- [lie_algebra.md](foundations/lie_algebra.md): Lie groups/algebras for optimizing on $SE(3)$ without singularities.
- [quaternion.md](foundations/quaternion.md): quaternions as a 3D-orientation representation.

## [`filtering/`](filtering/) - The Kalman-filter lineage

- [kf_ekf_iekf.md](filtering/kf_ekf_iekf.md): KF → EKF → Invariant EKF, in one progression.
- [extra_kf_variants.md](filtering/extra_kf_variants.md): UKF, ESKF, MSCKF and other variants, and why each exists.
- [left_right_invariant.md](filtering/left_right_invariant.md): left- vs. right-invariant error formulations in the IEKF (builds on `extra_kf_variants.md`).
- [ekf_iekf_ukf_empirical_note.md](filtering/ekf_iekf_ukf_empirical_note.md): an empirical note (not a concept explainer) on why `run_ekf`/`run_iekf` produce bit-identical output in `pointcloud_pose_tracking.py`, and why `run_ukf` only comes close but doesn't match either one exactly.

## [`optimization/`](optimization/) - The factor-graph / smoothing lineage

- [nonlinear_least_square.md](optimization/nonlinear_least_square.md): the problem every solver below is trying to solve: NLS as a formulation, distinct from any particular algorithm.
- [gauss_newton.md](optimization/gauss_newton.md): the core linearize-and-solve algorithm for NLS.
- [levenberg_marquardt.md](optimization/levenberg_marquardt.md): Gauss-Newton with a damping term, for when the initial estimate is untrustworthy.
- [factor_graph.md](optimization/factor_graph.md): the graphical structure (variables + factors) that SLAM's NLS problems are usually organized as.
- [pose_graph_optimization.md](optimization/pose_graph_optimization.md): factor graphs specialized to poses-only, relative-constraint problems.
- [bundle_adjustment.md](optimization/bundle_adjustment.md): factor graphs specialized to joint camera-pose + 3D-landmark refinement.
- [imu_preintegration.md](optimization/imu_preintegration.md): compressing raw high-rate IMU samples into one relative-motion edge, with a right-Jacobian trick to correct it for bias changes without re-integrating.
- [isam_optimization.md](optimization/isam_optimization.md): incrementally updating the solution to a growing factor graph(iSAM/iSAM2) instead of re-solving it from scratch every step.

## [`images/`](images/)

Diagrams referenced by the docs above, named `<doc>_<n>.jpg`. Each image's originally-claimed source is checked (not assumed) in its owning doc's "Image sources" section before being cited.
