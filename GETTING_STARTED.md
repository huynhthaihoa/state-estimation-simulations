# Getting started suggestion

A suggested reading-and-running order through this repo, for anyone arriving fresh. Each phase names the [`docs/`](docs/) conceptual notes to read first, then the matching simulation(s) to run under [`use_numpy/`](use_numpy/) (or its [`use_manif/`](use_manif/) counterpart) - in that order, since each script is written to exercise exactly what its phase's reading just introduced.

Before starting, follow [README.md § D. Installation](README.md#d-installation) to get `uv sync` working. Every script below is invoked the same way: `uv run python use_numpy/<script>.py [flags]` - the full flag list for each one lives in [README.md § E. Library guideline](README.md#e-library-guideline); this guide only links to the relevant numbered entry there rather than repeating it.

Once you've been through all nine phases, [`docs/README.md`](docs/README.md) is the fuller index to come back to for non-linear reference use.

---

## Phase 0 - Orient yourself (read only)

- [`docs/frontend_backend.md`](docs/frontend_backend.md) and [`docs/filtering_smoothing.md`](docs/filtering_smoothing.md) - the two "big picture" docs. The second matters most: it names the repo's central fork - **filtering** (recursive, Kalman-style) vs. **optimization/smoothing** (batch, factor-graph-style) - which is exactly how the rest of this repo is organized.
- [`docs/foundations/lie_algebra.md`](docs/foundations/lie_algebra.md) - the single most load-bearing prerequisite in the whole repo: why orientation/pose live on $SO(3)$/$`SE(3)`$ and get updated via $\text{Exp}$/$`\text{Log}`$ instead of flat-vector addition. Every script depends on this.
- [`docs/foundations/quaternion.md`](docs/foundations/quaternion.md) - quaternions as the other common 3D-orientation representation, alongside the rotation-matrix convention `lie_algebra.md` and this repo's own state representations otherwise use. Several `use_manif/` scripts do construct a `manif.SO3`/`manif.SE3` object from a quaternion under the hood (`euler_to_quat_xyzw`, an intermediate step, not the state representation itself) - this doc is the background for why that conversion looks the way it does.
- [`docs/foundations/jacobian.md`](docs/foundations/jacobian.md) - what a Jacobian is, and its §11 (left/right Jacobians) for later.

## Phase 1 - Prove the manifold matters

**Read:** nothing new yet.

**Run:** [README.md §1](README.md#1-naive-vs-exp-map-imu-integration) - [`use_numpy/imu_integration_comparison.py`](use_numpy/imu_integration_comparison.py) - naive Euler-angle vs. $SO(3)$-exp-map integration. The cleanest possible demonstration of Phase 0's claim: same noisy data, two integration schemes, watch the flat one drift.

## Phase 2 - Introduce batch optimization

**Read:** [`docs/optimization/nonlinear_least_square.md`](docs/optimization/nonlinear_least_square.md), [`docs/optimization/gauss_newton.md`](docs/optimization/gauss_newton.md).

**Run:** [README.md §2](README.md#2-imu-propagation--gauss-newton-position-correction) - [`use_numpy/robot_imu_simulation.py`](use_numpy/robot_imu_simulation.py) - IMU propagation plus a periodic Gauss-Newton correction against a GPS-like fix. First taste of "linearize, solve, retract" in a simple setting.

## Phase 3 - Compress measurements: preintegration

**Read:** [`docs/optimization/imu_preintegration.md`](docs/optimization/imu_preintegration.md) (leans on `jacobian.md` §11's right Jacobian). Optionally [`docs/frontend/vi_initialization.md`](docs/frontend/vi_initialization.md) as a conceptual-only follow-on - no script for it, but it's the natural "what has to happen before any of this can even start" companion.

**Run:** [README.md §3](README.md#3-imu-pre-integration-with-bias-jacobians) - [`use_numpy/imu_preintegration.py`](use_numpy/imu_preintegration.py) - bias-Jacobian bundle plus instant Taylor correction.

## Phase 4 - The Kalman-filter lineage

The biggest single payoff script in the repo lives here.

**Read, in order:** [`docs/filtering/kf_ekf_iekf.md`](docs/filtering/kf_ekf_iekf.md) (KF → EKF → IEKF), [`docs/filtering/extra_kf_variants.md`](docs/filtering/extra_kf_variants.md) (UKF/ESKF/MSCKF), [`docs/filtering/left_right_invariant.md`](docs/filtering/left_right_invariant.md).

**Run:** [README.md §4](README.md#4-point-cloud-pose-tracking-ekf-vs-invariant-ekf-vs-batch-gauss-newton-vs-ukf-vs-vanilla-kf) - [`use_numpy/pointcloud_pose_tracking.py`](use_numpy/pointcloud_pose_tracking.py) - EKF vs. IEKF vs. batch-GN vs. UKF vs. vanilla KF, head to head, with timing/memory numbers.

**Then read:** [`docs/filtering/pointcloud_pose_tracking_empirical_note.md`](docs/filtering/pointcloud_pose_tracking_empirical_note.md) - deliberately *after* running the script, since it's written as a post-hoc explanation of exactly that script's output (why EKF/IEKF are bit-identical, why UKF is close-but-not-exact, and why vanilla KF diverges from all three unconditionally).

**Side quest:** [`docs/filtering/hybrid_saltation_ekf.md`](docs/filtering/hybrid_saltation_ekf.md) + [README.md §11](README.md#11-saltation-matrix-ekf-tracking-a-point-mass-through-discrete-ground-contact-events) - [`use_numpy/saltation_matrix_ekf.py`](use_numpy/saltation_matrix_ekf.py). Everything above this point assumes continuous, smooth motion between measurements; this one script in the whole repo doesn't - a point mass bounces off the ground mid-trajectory, and the interesting question is how to propagate *covariance* (not just the mean) correctly through that discrete reset. Read the doc's derivation section first (it includes a formula that's correct, just for a different comparison than this filter needs, caught only by finite-difference verification against the specific comparison it does need), then run the script and check the Monte Carlo NEES plot - the result runs slightly against the naive expectation (the "mathematically correct" saltation-corrected filter comes out a bit *less* consistent than the naive one here, not more, though the gap is modest), which the doc explains. The doc's §8 then quantifies what happens once contact *detection* timing itself is uncertain (`--detect-time-bias`/`--detect-time-noise-std`) - the gap between the two filters widens further as jitter grows, though it stays within a similar modest range rather than blowing up.

**Two more side quests, same "hybrid/switched locomotion" theme, each isolating one further bio-inspired-locomotion question:**
- [`docs/filtering/inchworm_zupt_ekf.md`](docs/filtering/inchworm_zupt_ekf.md) + [README.md §12](README.md#12-inchworm-zupt-exploiting-a-known-anchordwell-schedule) - [`use_numpy/inchworm_zupt_ekf.py`](use_numpy/inchworm_zupt_ekf.py). Not a discontinuous reset like the bounce above - an ordinary switched-linear system where the interesting question is purely on the *measurement* side: exploiting a known anchor/dwell phase's free zero-velocity information, correctly, always, or never.
- [`docs/filtering/friction_anisotropic_ekf.md`](docs/filtering/friction_anisotropic_ekf.md) + [README.md §13](README.md#13-friction-anisotropic-process-noise-fixed-vs-heading-aware-q) - [`use_numpy/friction_anisotropic_ekf.py`](use_numpy/friction_anisotropic_ekf.py). Applies `pointcloud_pose_tracking_empirical_note.md`'s isotropic/anisotropic-and-rotating-frame argument to *process* noise instead of measurement noise, on a crawling unicycle over a friction-anisotropic pad.

## Phase 5 - Graph-based batch smoothing

**Read:** [`docs/optimization/levenberg_marquardt.md`](docs/optimization/levenberg_marquardt.md), [`docs/optimization/factor_graph.md`](docs/optimization/factor_graph.md), [`docs/optimization/pose_graph_optimization.md`](docs/optimization/pose_graph_optimization.md).

**Run:** [README.md §5](README.md#5-3d-pose-graph-relaxation) - [`use_numpy/pose_graph.py`](use_numpy/pose_graph.py) - drift, loop closure, gauge-freedom anchoring, and the chi-squared convergence printout, all in one script.

## Phase 6 - Bundle adjustment

**Read:** [`docs/foundations/umeyama_alignment.md`](docs/foundations/umeyama_alignment.md) (needed to align monocular BA's gauge-free output to ground truth), [`docs/optimization/bundle_adjustment.md`](docs/optimization/bundle_adjustment.md).

**Run:** [README.md §6](README.md#6-bundle-adjustment-joint-pose--landmark-refinement) - [`use_numpy/bundle_adjustment.py`](use_numpy/bundle_adjustment.py) - landmarks-only vs. poses-only strawmen vs. full joint BA.

**Side quest:** [`docs/frontend/triangulation_pnp.md`](docs/frontend/triangulation_pnp.md) +
[README.md §10](README.md#10-perspective-n-point-pnp-recovering-a-camera-pose-from-known-3d-2d-correspondences) - [`use_numpy/pnp_estimation.py`](use_numpy/pnp_estimation.py) - triangulation's exact inverse problem, using the same closed-form-then-GN recipe.

## Phase 7 - Scaling BA to a real system

**Read:** `bundle_adjustment.md` §13 (Local vs. Global BA, already covered in Phase 6).

**Run:** [README.md §7](README.md#7-local--global-bundle-adjustment-bounded-windows-vs-whole-map-re-solves) - [`use_numpy/bundle_adjustment_advanced.py`](use_numpy/bundle_adjustment_advanced.py) - bounded local windows plus periodic Global BA, cheirality guards, LM-with-step-rejection. Try the `--arc-span-deg 350` loop-closure variant afterward to see Global BA actually earn its keep.

## Phase 8 - Incremental solving (the iSAM lineage)

**Read:** [`docs/optimization/isam_optimization.md`](docs/optimization/isam_optimization.md).

**Run:** [README.md §8](README.md#8-incremental-square-root-sam-vs-batch-pose-graph-solving) - [`use_numpy/pose_graph_incremental.py`](use_numpy/pose_graph_incremental.py) - batch re-solve vs. incremental square-root SAM, reusing `pose_graph.py`'s own solver as the baseline.

**Read:** [`docs/optimization/elimination_tree.md`](docs/optimization/elimination_tree.md), [`docs/optimization/bayes_tree.md`](docs/optimization/bayes_tree.md).

**Run:** [README.md §9](README.md#9-bayes-tree-construction-and-affected-region-query) - [`use_numpy/bayes_tree_construction.py`](use_numpy/bayes_tree_construction.py) - builds the actual Bayes tree over the same square-loop topology, and shows why a loop-closure edge invalidates the *whole* chain while an ordinary edge invalidates just 2 nodes.

**Read:** [`docs/optimization/isam2_optimization.md`](docs/optimization/isam2_optimization.md) (what selective relinearization + reordering buy you beyond what you just built - conceptual only, no accompanying script) and [`docs/optimization/sparse_cholesky_factorization.md`](docs/optimization/sparse_cholesky_factorization.md) (why every GN/LM solve above exploited sparsity).

**Then read:** [`docs/optimization/marginalization.md`](docs/optimization/marginalization.md) (sliding-window smoothing - ties MSCKF/VINS-Mono back to the elimination machinery you just learned).

**Run:** [README.md §14](README.md#14-sliding-window-marginalization-bounded-memory-pose-graph-smoothing) - [`use_numpy/sliding_window_marginalization.py`](use_numpy/sliding_window_marginalization.py) - marginalizes the oldest pose out of a bounded-size window via the Schur complement as a pure odometry chain streams in, and measures it against an unbounded from-scratch re-solve at every trajectory length in a sweep: full-batch's peak system size grows quadratically in memory, sliding-window's caps flat the moment the window first fills, at no accuracy cost in this pure-chain setting.

---

**Why this order:** each phase's docs are exactly the prerequisites the next script needs and nothing more, complexity ramps monotonically (flat-vs-manifold → single-pose filtering → multi-pose graphs → joint pose+landmark → incremental/scalable), and the conceptual-only docs without their own script (`vi_initialization.md`, `isam2_optimization.md`) land right where their ideas are most load-bearing even so.
