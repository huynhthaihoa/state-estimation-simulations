# Getting started suggestion

A suggested reading-and-running order through this repo, for newcomers. Each phase names the [`docs/`](docs/) notes to read first, then the matching simulation(s) to run under [`use_numpy/`](use_numpy/) (or its [`use_manif/`](use_manif/) counterpart). Keep that order: each script exercises what its phase's reading just introduced.

Before starting, follow [README.md § D. Installation](README.md#d-installation) to get `uv sync` working. Every script below runs as `uv run python use_numpy/<script>.py [flags]`. The full flag list for each lives in [README.md § E. Library guideline](README.md#e-library-guideline); this guide links to the relevant numbered entry instead of repeating it.

After all nine phases, [`docs/README.md`](docs/README.md) is the fuller index for non-sequential reference, and [`docs/slam_mental_map.md`](docs/slam_mental_map.md) shows where each phase sits in the overall picture.

---

## Phase 0 - Orient yourself (read only)

- [`docs/frontend_backend.md`](docs/frontend_backend.md) and [`docs/filtering_smoothing.md`](docs/filtering_smoothing.md) - the two big-picture docs. The second matters most: it names the repo's central fork, **filtering** (recursive, Kalman-style) vs. **optimization/smoothing** (factor-graph-style), which organizes the rest of the repo.
- [`docs/slam_mental_map.md`](docs/slam_mental_map.md) §1 only - the one-diagram overview of how every doc and script fits together. Skim it for the layout; the later sections use terms the phases below introduce, so return to them afterward.
- [`docs/foundations/lie_algebra.md`](docs/foundations/lie_algebra.md) - the most important prerequisite in the repo: why orientation/pose live on $SO(3)$/$`SE(3)`$ and get updated via $\text{Exp}$/$`\text{Log}`$ instead of flat-vector addition. Most scripts depend on it.
- [`docs/foundations/quaternion.md`](docs/foundations/quaternion.md) - quaternions as the other common 3D-orientation representation, alongside the rotation matrices that `lie_algebra.md` and this repo's states otherwise use. Several `use_manif/` scripts build a `manif.SO3`/`manif.SE3` object from a quaternion (e.g. `euler_to_quat_xyzw` in `imu_integration_comparison.py`, `rotation_matrix_to_quaternion_xyzw` in the BA scripts); this doc is the background for that conversion.
- [`docs/foundations/jacobian.md`](docs/foundations/jacobian.md) - what a Jacobian is, and its §11 (left/right Jacobians) for later.

## Phase 1 - Prove the manifold matters

**Read:** nothing new yet.

**Run:** [README.md §1](README.md#1-naive-vs-exp-map-imu-integration) - [`use_numpy/imu_integration_comparison.py`](use_numpy/imu_integration_comparison.py) - naive Euler-angle vs. $SO(3)$-exp-map integration. A direct demonstration of Phase 0's claim: same noisy data, two integration schemes, and the flat one drifts.

## Phase 2 - Introduce batch optimization

**Read:** [`docs/optimization/nonlinear_least_square.md`](docs/optimization/nonlinear_least_square.md), [`docs/optimization/gauss_newton.md`](docs/optimization/gauss_newton.md).

**Run:** [README.md §2](README.md#2-imu-propagation--gauss-newton-position-correction) - [`use_numpy/robot_imu_simulation.py`](use_numpy/robot_imu_simulation.py) - IMU propagation plus a periodic Gauss-Newton correction against a GPS-like fix. A first look at "linearize, solve, retract" in a simple setting.

## Phase 3 - Compress measurements: preintegration

**Read:** [`docs/optimization/imu_preintegration.md`](docs/optimization/imu_preintegration.md) (leans on `jacobian.md` §11's right Jacobian). Optionally [`docs/frontend/vi_initialization.md`](docs/frontend/vi_initialization.md) as a conceptual-only follow-on (no script): what has to happen before any of this can start.

**Run:** [README.md §3](README.md#3-imu-preintegration-with-bias-jacobians) - [`use_numpy/imu_preintegration.py`](use_numpy/imu_preintegration.py) - bias-Jacobian bundle plus instant Taylor correction.

## Phase 4 - The Kalman-filter lineage

**Read, in order:** [`docs/filtering/kf_ekf_iekf.md`](docs/filtering/kf_ekf_iekf.md) (KF → EKF → IEKF), [`docs/filtering/linear_nonlinear.md`](docs/filtering/linear_nonlinear.md) (how to tell a linear system from a nonlinear one, and whether the nonlinearity matters at your uncertainty level), [`docs/filtering/extra_kf_variants.md`](docs/filtering/extra_kf_variants.md) (UKF/ESKF/MSCKF), [`docs/filtering/left_right_invariant.md`](docs/filtering/left_right_invariant.md).

**Run:** [README.md §4](README.md#4-point-cloud-pose-tracking-ekf-vs-invariant-ekf-vs-batch-gauss-newton-vs-ukf-vs-vanilla-kf) - [`use_numpy/pointcloud_pose_tracking.py`](use_numpy/pointcloud_pose_tracking.py) - EKF vs. IEKF vs. batch-GN vs. UKF vs. vanilla KF, head to head, with timing/memory numbers.

**Then read:** [`docs/filtering/pointcloud_pose_tracking_empirical_note.md`](docs/filtering/pointcloud_pose_tracking_empirical_note.md) - deliberately *after* running the script, since it explains that script's output after the fact: why EKF/IEKF agree to round-off, why UKF differs mainly during the initial correction, and why vanilla KF stays millimeters away.

**Side quest:** [`docs/filtering/hybrid_saltation_ekf.md`](docs/filtering/hybrid_saltation_ekf.md) + [README.md §11](README.md#11-saltation-matrix-ekf-tracking-a-point-mass-through-discrete-ground-contact-events) - [`use_numpy/saltation_matrix_ekf.py`](use_numpy/saltation_matrix_ekf.py). Everything above assumes smooth motion between measurements; this script does not. A point mass bounces off the ground mid-trajectory, and the question is how to propagate the *covariance* (not just the mean) through that discrete reset.
- **Read** the doc's derivation section first. It includes a formula that is correct for a different comparison than this filter needs; finite-difference checks show which one applies.
- **Run** the script and check the Monte Carlo NEES plot. Contrary to the expectation that the naive update is overconfident, with exact contact detection both filters are *under*confident after a bounce (post-bounce NEES about 2.4 and 2.5 against a consistent 6), the two are close, and which one is nearer 6 depends on the tuning.
- **Then** the doc's §8 quantifies what happens once contact *detection* timing itself is uncertain (`--detect-time-bias`/`--detect-time-noise-std`): both filters become overconfident as jitter grows, and the saltation-corrected one more so (about 1.4-1.6× the naive filter's NEES), a modest gap rather than a blow-up.

- **Next**, the same bounce on a Lie group: [`docs/filtering/invariant_saltation_ekf.md`](docs/filtering/invariant_saltation_ekf.md) + [README.md §15](README.md#15-invariant-ekf-on-se_23-through-a-bounce) - [`use_numpy/invariant_saltation_ekf.py`](use_numpy/invariant_saltation_ekf.py) (or its [`use_manif/`](use_manif/invariant_saltation_ekf.py) twin). Read it after this phase's invariant-EKF material ([`docs/filtering/left_right_invariant.md`](docs/filtering/left_right_invariant.md) and README §4). It combines the right-invariant error with the saltation matrix, and shows that the bounce breaks the invariant filter's [trajectory independence](docs/glossary.md#group-affine-dynamics-trajectory-independence).

**Two more side quests, same "hybrid/switched locomotion" theme, each isolating one further bio-inspired-locomotion question:**
- [`docs/filtering/inchworm_zupt_ekf.md`](docs/filtering/inchworm_zupt_ekf.md) + [README.md §12](README.md#12-inchworm-zupt-exploiting-a-known-anchordwell-schedule) - [`use_numpy/inchworm_zupt_ekf.py`](use_numpy/inchworm_zupt_ekf.py). Not a discontinuous reset like the bounce above, but an ordinary switched-linear system. The question is on the *measurement* side: using the free zero-velocity information of a known anchor/dwell phase correctly, always, or never.
- [`docs/filtering/friction_anisotropic_ekf.md`](docs/filtering/friction_anisotropic_ekf.md) + [README.md §13](README.md#13-friction-anisotropic-process-noise-fixed-vs-heading-aware-q) - [`use_numpy/friction_anisotropic_ekf.py`](use_numpy/friction_anisotropic_ekf.py). Applies the isotropic/anisotropic and rotating-frame argument of `pointcloud_pose_tracking_empirical_note.md` to *process* noise instead of measurement noise, on a crawling unicycle over a friction-anisotropic pad.

## Phase 5 - Graph-based batch smoothing

**Read:** [`docs/optimization/levenberg_marquardt.md`](docs/optimization/levenberg_marquardt.md), [`docs/optimization/factor_graph.md`](docs/optimization/factor_graph.md), [`docs/optimization/pose_graph_optimization.md`](docs/optimization/pose_graph_optimization.md).

**Run:** [README.md §5](README.md#5-3d-pose-graph-relaxation) - [`use_numpy/pose_graph.py`](use_numpy/pose_graph.py) - drift, loop closure, gauge-freedom anchoring, and the [chi-squared](docs/glossary.md#chi-square-distribution) convergence printout, all in one script.

## Phase 6 - Bundle adjustment

**Read:** [`docs/foundations/umeyama_alignment.md`](docs/foundations/umeyama_alignment.md) (needed to align monocular BA's gauge-free output to ground truth), [`docs/optimization/bundle_adjustment.md`](docs/optimization/bundle_adjustment.md).

**Run:** [README.md §6](README.md#6-bundle-adjustment-joint-pose--landmark-refinement) - [`use_numpy/bundle_adjustment.py`](use_numpy/bundle_adjustment.py) - landmarks-only vs. poses-only strawmen vs. full joint BA.

**Side quest:** [`docs/frontend/triangulation_pnp.md`](docs/frontend/triangulation_pnp.md) + [README.md §10](README.md#10-perspective-n-point-pnp-recovering-a-camera-pose-from-known-3d-2d-correspondences) - [`use_numpy/pnp_estimation.py`](use_numpy/pnp_estimation.py) - the inverse of triangulation, using the same closed-form-then-Gauss-Newton recipe.

## Phase 7 - Scaling BA to a real system

**Read:** `bundle_adjustment.md` §13 (Local vs. Global BA, already covered in Phase 6).

**Run:** [README.md §7](README.md#7-local--global-bundle-adjustment-bounded-windows-vs-whole-map-re-solves) - [`use_numpy/bundle_adjustment_advanced.py`](use_numpy/bundle_adjustment_advanced.py) - bounded local windows plus periodic Global BA, cheirality guards, LM-with-step-rejection. Try the `--arc-span-deg 350 --n-keyframes 190` loop-closure variant afterward to see a real loop closure detected (Global BA's payoff there is seed-dependent).

## Phase 8 - Incremental solving (the iSAM lineage)

**Read:** [`docs/optimization/isam_optimization.md`](docs/optimization/isam_optimization.md).

**Run:** [README.md §8](README.md#8-incremental-square-root-sam-vs-batch-pose-graph-solving) - [`use_numpy/pose_graph_incremental.py`](use_numpy/pose_graph_incremental.py) - batch re-solve vs. incremental square-root SAM (Givens updates for odometry edges, a full rebuild at loop closure, no reordering), reusing `pose_graph.py`'s own solver as the baseline.

**Read:** [`docs/optimization/elimination_tree.md`](docs/optimization/elimination_tree.md), [`docs/optimization/bayes_tree.md`](docs/optimization/bayes_tree.md).

**Run:** [README.md §9](README.md#9-bayes-tree-construction-and-affected-region-query) - [`use_numpy/bayes_tree_construction.py`](use_numpy/bayes_tree_construction.py) - builds the elimination tree underlying the Bayes tree over the same square-loop topology, and shows why a loop-closure edge invalidates the *whole* chain while an ordinary edge invalidates just 2 of 16 variables.

**Read:** [`docs/optimization/isam2_optimization.md`](docs/optimization/isam2_optimization.md) (what selective relinearization + reordering buy you beyond what you just built - conceptual only, no accompanying script) and [`docs/optimization/sparse_cholesky_factorization.md`](docs/optimization/sparse_cholesky_factorization.md) (why real solvers exploit sparsity; this repo's solves are dense).

**Then read:** [`docs/optimization/marginalization.md`](docs/optimization/marginalization.md) (sliding-window smoothing - ties VINS-Mono-style windows back to the elimination machinery you just learned).

**Run:** [README.md §14](README.md#14-sliding-window-marginalization-bounded-memory-pose-graph-smoothing) - [`use_numpy/sliding_window_marginalization.py`](use_numpy/sliding_window_marginalization.py) - marginalizes the oldest pose out of a bounded window via the Schur complement as an odometry chain streams in, and compares it with an unbounded from-scratch re-solve across a sweep of trajectory lengths.
- Full-batch system size grows linearly (dense-matrix memory quadratically).
- The sliding window's size caps flat once the window first fills.
- Accuracy ties by construction (the optimum of a pure odometry chain is the dead-reckoned trajectory), so the memory bound is the real result.

---

**Why this order:** each phase's docs are mostly the prerequisites for the next script, and complexity ramps up roughly as flat-vs-manifold → single-pose filtering → multi-pose graphs → joint pose and landmark → incremental/scalable. The conceptual-only docs (`vi_initialization.md`, `isam2_optimization.md`) sit where their ideas matter most.
