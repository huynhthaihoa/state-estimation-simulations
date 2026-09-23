# State Estimation Simulations

> New here? [GETTING_STARTED.md](GETTING_STARTED.md) is a suggested reading-and-running order through every doc and script in this repo, phase by phase.

## A. Introduction

A collection of from-scratch simulations exploring **pose/state estimation on manifolds**: 
 - How **orientation** and **pose** should be integrated and corrected on $SO(3)$ / $SE(3)$ rather than treated as flat vectors
 - How a **prior** (a motion model driven by noisy control/odometry inputs) can be fused with **measurements** either **recursively** (Kalman filtering) or in **batch** (Gauss-Newton optimization).

Two implementation styles run side by side for the core comparison scripts, split into sibling directories with matching filenames:
- [use_numpy/](use_numpy/): skew-symmetric matrices, $Exp$ / $Log$ maps, and Jacobians written out by hand (Rodrigues' formula, the $SE(3)$ exponential/logarithm, the analytical inverse right Jacobian), shared across scripts via [use_numpy/lie_utils.py](use_numpy/lie_utils.py).
- [use_manif/](use_manif/): the same math delegated to the [`manif`](https://github.com/artivis/manif) Lie-theory library's Python bindings (`T.rplus`, `T.rminus`, `T.act`, all with analytical Jacobians returned as out-parameters), so no manifold formula is hand-rolled.

Each `use_manif/<name>.py` is the manifpy counterpart of `use_numpy/<name>.py` of the same filename.

## B. Documentation

[docs/](docs/) has the conceptual write-ups behind these simulations:
- **Math foundations**: Jacobian, Lie algebra, quaternions
- The **Kalman-filter** family: KF/EKF/IEKF and variants
- The **factor-graph/smoothing** family: NLS, Gauss-Newton, Levenberg-Marquardt, factor graphs,
pose-graph optimization, bundle adjustment, iSAM 

Start at [docs/README.md](docs/README.md) for the full index, or [docs/frontend_backend.md](docs/frontend_backend.md) / [docs/filtering_smoothing.md](docs/filtering_smoothing.md) for the two entry-point overviews.

## C. Library structure

### [utils.py](utils.py) - Shared, backend-agnostic helpers

Lives at the repo root (not inside `use_numpy/`/`use_manif/`) since both backends import from it
directly: `true_body_rates` (the shared ground-truth angular/linear-velocity profile several scripts
simulate against), `rotation_matrix_to_quaternion_xyzw`, `umeyama_alignment`/`landmark_errors`
(gauge-freedom alignment for monocular BA, `docs/foundations/umeyama_alignment.md`),
`unscented_weights`/`unscented_sigma_offsets` (UKF sigma-point machinery), `qr_insert_row`
(Givens-rotation incremental QR update, `pose_graph_incremental.py`), `symbolic_eliminate`/
`bayes_tree_affected_path` (elimination-tree construction and affected-region query,
`bayes_tree_construction.py`), `look_at_rotation`/`forward_facing_rotation`/`pair_key` (camera-pose
and pairing helpers for the bundle-adjustment scripts), and `measure_performance` (the shared
timing/peak-memory measurement helper every script's benchmark printout uses).

### [use_numpy/](use_numpy/) - Plain numpy, hand-rolled Lie-group math

- [lie_utils.py](use_numpy/lie_utils.py): shared module of the hand-rolled Lie-group helpers (`skew`, `rotation_geodesic_error`, `so3_exp`, `so3_right_jacobian`, `se3_exp`, `se3_log`, `se3_inv`, `se3_adjoint`, `compute_so3_inv_right_jacobian`, `compute_se3_inv_right_jacobian`, `se3_right_jacobian`), imported by 10 of the other 14 scripts in this directory (`imu_integration_comparison.py`, `robot_imu_simulation.py`, `imu_preintegration.py`, `pointcloud_pose_tracking.py`, `pose_graph.py`, `pose_graph_incremental.py`, `bundle_adjustment.py`, `bundle_adjustment_advanced.py`, `pnp_estimation.py`, and `sliding_window_marginalization.py`) - none of them keep their own inline copies of the skew/Jacobian helpers anymore. The remaining 4 (`bayes_tree_construction.py`, `friction_anisotropic_ekf.py`, `inchworm_zupt_ekf.py`, `saltation_matrix_ekf.py`) don't need Lie-group math for what they do.
- [imu_integration_comparison.py](use_numpy/imu_integration_comparison.py): naive Euler-angle vs. $SO(3)$ exp-map orientation integration.
- [robot_imu_simulation.py](use_numpy/robot_imu_simulation.py): high-rate IMU propagation + a low-rate Gauss-Newton position-only correction against a noisy GPS-style fix (the correction's Jacobian has a structurally zero angular block, so orientation is never touched by it, only ever dead-reckoned by the IMU), reusing `lie_utils.py`'s $SE(3)$ $Exp$ math.
- [imu_preintegration.py](use_numpy/imu_preintegration.py): IMU pre-integration - compresses a burst of high-frequency IMU samples into one relative measurement plus first-order bias Jacobians, then shows an instant Taylor-expansion correction when the bias estimate changes, without re-integrating.
- [pointcloud_pose_tracking.py](use_numpy/pointcloud_pose_tracking.py): tracks a rigid object's pose from a motion-model prior (noisy control inputs) fused with noisy point-cloud measurements of its known geometry, comparing a recursive **EKF**, a recursive **invariant EKF** (body-frame residual, state-independent measurement Jacobian), a **batch Gauss-Newton** smoother over the whole trajectory, a recursive **UKF** (sigma-point unscented transform via right-perturbation retraction, no Jacobians at all), and a recursive **vanilla KF** (redundant 12-dim `[vec(R), t]` ambient state instead of the $SE(3)$ tangent state - exactly linear observation model, but needs a small-angle motion-model truncation and an explicit SVD re-projection back onto $SO(3)$ after every update), reusing `lie_utils.py`'s $SE(3)$ $Exp$ / $Log$ /inverse-right-Jacobian math plus a hand-rolled adjoint/right-Jacobian for the motion-model Jacobians. Also reports empirical per-step time and peak memory for each of the six approaches.
- [pose_graph.py](use_numpy/pose_graph.py): a small closed-loop 3D pose-graph relaxation (odometry drift + one loop closure), jointly optimized via Levenberg-Marquardt, reusing the same `lie_utils.py` $SE(3)$ $Exp$ / $Log$ / inverse-right-Jacobian/adjoint math as `pointcloud_pose_tracking.py`.
- [pose_graph_incremental.py](use_numpy/pose_graph_incremental.py): the [iSAM](docs/optimization/isam_optimization.md) counterpart to `pose_graph.py`'s batch solver - a longer loop streams in one node at a time, and an incremental square-root-SAM solver (Givens-rotation QR row insertion via `qr_insert_row` in `utils.py`, plus periodic/loop-closure-triggered full relinearization) is timed against a from-scratch batch re-solve at every step.
- [bayes_tree_construction.py](use_numpy/bayes_tree_construction.py): builds the actual [Bayes tree](docs/optimization/bayes_tree.md) `iSAM2` relies on - symbolic variable elimination (`symbolic_eliminate` in `utils.py`) over the same square-loop pose-graph topology, then a root-ward affected-path query (`bayes_tree_affected_path`) contrasting a local odometry edge against the loop-closure edge. Pure index/graph bookkeeping, no pose math at all, so there's no `use_manif/` counterpart - the result would be structurally identical either way.
- [bundle_adjustment.py](use_numpy/bundle_adjustment.py): jointly refines camera poses **and** 3D landmarks against pinhole reprojection error - cameras on an arc around a landmark cluster, with a field-of-view cutoff so not every camera observes every landmark. Compares three solvers: **landmarks-only refinement** and **poses-only refinement** (independent $3\times3$/$6\times6$ GN solves, each a "fix one side" strawman) against **full joint bundle adjustment** (coupled dense GN over poses + landmarks, gauge-fixed with a prior factor on the first two camera poses, then [Umeyama-aligned](docs/foundations/umeyama_alignment.md) to ground truth before reporting absolute error, since monocular BA only recovers the scene up to an unknown similarity transform).
- [bundle_adjustment_advanced.py](use_numpy/bundle_adjustment_advanced.py): Local **and** Global bundle adjustment, run back to back for direct comparison - the real-time-system-behavior counterpart to `bundle_adjustment.py`'s single-batch scene. A camera moves keyframe-by-keyframe along a forward-facing arc through a landmark corridor instead of sitting on a static ring; a covisibility graph builds incrementally, and every new keyframe triggers a bounded local Gauss-Newton/Levenberg-Marquardt solve over an active window (new keyframe + covisible neighbors, with every other observing keyframe held fixed as a rigid anchor), while a periodic Global BA pass jointly re-solves the whole map so far for contrast. Keyframes 0/1 are hard-fixed forever as the gauge anchor - no prior factor needed, unlike `bundle_adjustment.py`, since a hard anchor already pins the gauge with no residual freedom left to constrain. Guards against Gauss-Newton divergence (Levenberg-Marquardt damping) and the classic point-behind-camera reflection ambiguity (`passes_cheirality`/ `cull_invalid_points`) that a weakly-constrained, forward-motion scene can hit but `bundle_adjustment.py`'s densely-observed toy scene never does.
- [pnp_estimation.py](use_numpy/pnp_estimation.py): Perspective-n-Point (PnP) - triangulation's exact inverse ([docs/frontend/triangulation_pnp.md](docs/frontend/triangulation_pnp.md)): given known 3D points and their observed pixels, recovers the unknown camera pose via a closed-form linear DLT initial guess (specialized to known intrinsics), then a few Gauss-Newton iterations against the true reprojection error. No `use_manif/` counterpart (single-implementation, like `bayes_tree_construction.py`).
- [saltation_matrix_ekf.py](use_numpy/saltation_matrix_ekf.py): a bouncing point mass ([docs/filtering/hybrid_saltation_ekf.md](docs/filtering/hybrid_saltation_ekf.md)) tracked through discrete ground-contact events by a naive EKF (reset-Jacobian-only covariance handling) vs. a saltation-matrix-corrected EKF, plus a Monte Carlo NEES consistency check (new to this repo), and `--detect-time-bias`/`--detect-time-noise-std` params quantifying what happens once contact-detection timing itself is uncertain (doc §8). Plain $\mathbb{R}^6$ state (position/velocity, no rotation) - no `use_manif/` counterpart, for the same reason as `bayes_tree_construction.py`.
- [inchworm_zupt_ekf.py](use_numpy/inchworm_zupt_ekf.py): a 1D point mass crawling through a known anchor(dwell)/extend gait cycle ([docs/filtering/inchworm_zupt_ekf.md](docs/filtering/inchworm_zupt_ekf.md)), comparing three zero-velocity-update (ZUPT) policies - `never`, `always`, `phase_conditional` - via the same Monte Carlo NEES pattern as `saltation_matrix_ekf.py`. State $x=[p,v]\in\mathbb{R}^2$ - no `use_manif/` counterpart, for the same reason as `saltation_matrix_ekf.py`.
- [friction_anisotropic_ekf.py](use_numpy/friction_anisotropic_ekf.py): a crawling unicycle on a friction-anisotropic pad ([docs/filtering/friction_anisotropic_ekf.md](docs/filtering/friction_anisotropic_ekf.md)), comparing `isotropic`/`fixed_anisotropic`/`heading_aware` process-noise policies for the position block of $Q$ as the (exact, noise-free) heading rotates through a full loop. Ordinary EKF throughout (heading itself is exact, so no manifold/linearization question is at stake) - no `use_manif/` counterpart for the same reason.
- [sliding_window_marginalization.py](use_numpy/sliding_window_marginalization.py): gives [docs/optimization/marginalization.md](docs/optimization/marginalization.md) its accompanying script - streams a pure odometry chain, marginalizes the oldest pose via the Schur complement whenever a `--window-size` cap would be exceeded, and compares the resulting bounded-memory solve against `run_full_batch_growing`'s unbounded re-solve-from-scratch baseline across a sweep of trajectory lengths, reusing `pose_graph.py`/`pose_graph_incremental.py`'s building blocks. No `use_manif/` counterpart - the bounded-vs-unbounded scaling result is backend-agnostic.

### [use_manif/](use_manif/) - Same simulations, on `manifpy`

- [imu_integration_comparison.py](use_manif/imu_integration_comparison.py)
- [robot_imu_simulation.py](use_manif/robot_imu_simulation.py)
- [imu_preintegration.py](use_manif/imu_preintegration.py): same bias-Jacobian preintegration bundle, with the $SO(3)$ $Exp$ map / right Jacobian / skew(hat) math delegated to manif's `rplus` Jacobian out-parameters and `SO3Tangent.hat()` instead of hand-rolled formulas.
- [pointcloud_pose_tracking.py](use_manif/pointcloud_pose_tracking.py): same six solvers (dead-reckoning, EKF, invariant EKF, batch Gauss-Newton, UKF, vanilla KF) as the `use_numpy` version, with `motion_model`/`observation_model` built on `manifpy`'s `rplus`/`rminus`/`act` Jacobian out-parameters instead of `lie_utils.py` (the UKF itself uses only the bare `rplus`/`rminus` group operations, no Jacobian out-parameters). The vanilla KF's ambient-state recursion works directly on `T.rotation()`/`T.translation()` numpy arrays instead - it only touches manif for `SO3Tangent(.).hat()` and reconstructs the final pose via `scipy.spatial.transform.Rotation.from_matrix(...).as_quat()`, since manif's `SE3`/`SO3` expose no rotation-matrix constructor.
- [pose_graph.py](use_manif/pose_graph.py)
- [pose_graph_incremental.py](use_manif/pose_graph_incremental.py): same incremental-vs-batch comparison as the `use_numpy` version, with edge Jacobians obtained by chaining `manifpy`'s own `compose`/`rminus` Jacobian out-parameters instead of `lie_utils.py`.
- [bundle_adjustment.py](use_manif/bundle_adjustment.py): same three solvers as the `use_numpy` version, with camera-pose Jacobians obtained by chaining `manifpy`'s own `inverse`/`act` Jacobian out-parameters instead of a hand-rolled closed form; landmarks stay plain numpy $\mathbb{R}^3$ vectors (manif has no notion of those), same as `pointcloud_pose_tracking.py`.
- [bundle_adjustment_advanced.py](use_manif/bundle_adjustment_advanced.py): same Local + Global BA comparison as the `use_numpy` version, with camera-pose Jacobians obtained by chaining `manifpy`'s own `inverse`/ `act` Jacobian out-parameters, matching `bundle_adjustment.py`'s own manif version.

### Root

- [pyproject.toml](pyproject.toml): uv project file. `manifpy` is an optional dependency (the `manif` extra) sourced from the local sibling checkout `../manif` (see Installation below).

## D. Installation

- Install [uv](https://docs.astral.sh/uv/).
- `uv sync` alone installs everything the [use_numpy/](use_numpy/) scripts need, with no `manifpy`/`../manif` requirement at all.
- The [use_manif/](use_manif/) scripts additionally need `manifpy`, which isn't on PyPI: this project's `pyproject.toml` points `uv` at a local sibling checkout (`../manif`) and builds it automatically (requires a working Eigen3 + CMake toolchain) when you request the `manif` extra: `uv sync --extra manif`.
- Run any script with `uv run` from this repository's root directory, e.g.:
  `uv run python use_numpy/imu_integration_comparison.py`

### Running the tests

`tests/` has a pytest suite covering `utils.py`, `use_numpy/lie_utils.py`'s Lie-group math (round-trip and Jacobian-identity checks), and the testable functions in every `use_numpy/`/`use_manif/` simulation script (deterministic, fixed-seed regression checks - e.g. optimized error beats an uncorrected baseline, two solvers agree on the same input, a filter matches its own documented exact-agreement claim). `use_manif/` tests skip cleanly (not fail) if you ran plain `uv sync` (no `manif` extra, so `manifpy` isn't installed).

```
uv run pytest tests/ -v
```

## E. Library guideline

### 1. Naive vs. exp-map IMU integration

#### Purpose

Integrates the same noisy gyro + body-velocity stream two ways: attitude kept as a **flat Euler-angle vector** (`euler += omega*dt`) vs. attitude kept on **$SO(3)$ and updated via the exponential map** - to isolate the error the flat vector-space approximation introduces on its own. Ground truth is the exp-map integration of the noise-free rates. Plots rotation and position error (log scale) over time.

#### Scripts

- [use_numpy/imu_integration_comparison.py](use_numpy/imu_integration_comparison.py) 

- [use_manif/imu_integration_comparison.py](use_manif/imu_integration_comparison.py)


#### Usage

```
uv run python use_numpy/imu_integration_comparison.py --duration 20.0 --dt 0.005 --gyro-noise-std 0.02 --vel-noise-std 0.05 --seed 0 --out out.png
```

- `--duration`: simulation length in seconds (default `20.0`)
- `--dt`: IMU sample interval in seconds (default `0.005`)
- `--gyro-noise-std`: gyro noise std-dev, rad/s (default `0.02`)
- `--vel-noise-std`: body-velocity noise std-dev, m/s (default `0.05`)
- `--seed`: RNG seed (default `0`)
- `--out`: save the figure to this path instead of showing it (default: show)

### 2. IMU propagation + Gauss-Newton position correction

#### Purpose

Simulates a robot with a 100 Hz IMU (noisy body twist) and a 1 Hz noisy global position fix (e.g. GPS). Each second: propagate the pose estimate through 100 noisy IMU micro-steps on $SE(3)$, then run a Gauss-Newton correction against a genuinely noisy position-only measurement (its Jacobian wrt a right perturbation is $[R_{\text{est}} \mid 0]$ - the angular block is exactly zero, so this measurement structurally cannot correct orientation, however it's weighted; orientation is left entirely to the IMU's own dead-reckoning between corrections) until the correction step norm drops below `--gn-tol` or `--gn-max-iters` is hit. Prints pre/post correction error each second (no plot).

#### Scripts

- [use_numpy/robot_imu_simulation.py](use_numpy/robot_imu_simulation.py) 

- [use_manif/robot_imu_simulation.py](use_manif/robot_imu_simulation.py)

#### Usage

```
uv run python use_numpy/robot_imu_simulation.py --dt-imu 0.01 --total-seconds 3 --snapshots-per-second 4 --gn-tol 1e-6 --gn-max-iters 10 --max-linear-vel 1.0 --max-angular-vel 0.5 --pos-noise-std 0.05
```

- `--dt-imu`: IMU update interval in seconds (default `0.01`, i.e. 100 Hz)
- `--total-seconds`: total simulation duration in seconds (default `3`)
- `--snapshots-per-second`: number of intermediate IMU steps logged per second (default `4`)
- `--gn-tol`: Gauss-Newton convergence tolerance (default `1e-6`)
- `--gn-max-iters`: maximum Gauss-Newton iterations (default `10`)
- `--max-linear-vel`: max linear velocity magnitude for the random true twist, m/s (default `1.0`)
- `--max-angular-vel`: max angular velocity magnitude for the random true twist, rad/s (default `0.5`)
- `--pos-noise-std`: per-axis std-dev (m) of the Global Position Measurement's Gaussian noise (default `0.05`)

### 3. IMU pre-integration with bias Jacobians

#### Purpose

Streams 1 second of high-frequency IMU samples into a single `PreintegratedIMUBundle` (compressed relative rotation/velocity/position, plus their Jacobians wrt gyro/accel bias). Then simulates a graph-SLAM-style bias update and applies it to the bundle via a first-order Taylor correction - instant, versus re-running the whole integration loop.

#### Scripts

- [use_numpy/imu_preintegration.py](use_numpy/imu_preintegration.py) 

- [use_manif/imu_preintegration.py](use_manif/imu_preintegration.py)

#### Usage

```
uv run python use_numpy/imu_preintegration.py --frequency-hz 100 --gyro-bias 0.01 -0.01 0.02 --accel-bias 0.05 0.00 -0.05 --linear-vel 1.0 0.1 0.0 --angular-vel 0.0 0.0 0.5 --optimized-gyro-bias 0.008 -0.009 0.018 --optimized-accel-bias 0.045 0.002 -0.048
```

- `--frequency-hz`: IMU sampling frequency integrated over 1 simulated second (default `100`)
- `--gyro-bias`, `--accel-bias`: initial estimated sensor biases (default `0.01 -0.01 0.02` rad/s, `0.05 0.00 -0.05` $\text{m/s}^2$)
- `--linear-vel`, `--angular-vel`: true commanded body-frame twist (default `1.0 0.1 0.0` m/s, `0.0 0.0 0.5` rad/s)
- `--optimized-gyro-bias`, `--optimized-accel-bias`: post-optimization corrected biases to apply via the Taylor correction

### 4. Point-cloud pose tracking: EKF vs. invariant EKF vs. batch Gauss-Newton vs. UKF vs. vanilla KF

#### Purpose

Tracks a rigid object's $SE(3)$ pose from a combination of a motion-model prior (noisy control-input twist) and noisy point-cloud measurements of the object's known body-frame geometry (`z_i = T.act(p_i) + noise`). Two shared, documented functions - `motion_model` and `observation_model`, both with analytical Jacobians (hand-rolled $SE(3)$ $Exp$ / $Log$ /adjoint math via `lie_utils.py` in the `use_numpy` version; `manifpy`'s `rplus`/`rminus`/`act` out-parameters in the `use_manif` version) - feed six solvers: 
 1. A **prior-only dead-reckoning baseline**
 2. A recursive **EKF** (constant-size state, online)
 3. A recursive **invariant EKF** (same predict step, but the update expresses the residual in the estimate's body frame, making the measurement Jacobian state-independent instead of re-linearized around the current rotation every step; for this problem's [isotropic](docs/filtering/pointcloud_pose_tracking_empirical_note.md#a1-isotropic-and-anisotropic-noise) point-noise model this is provably equivalent to the plain EKF's corrections at every step, so the practical win here is a fixed, precomputed Jacobian rather than different accuracy)
 4. A **batch Gauss-Newton** smoother that jointly optimizes the whole trajectory at once against prior/motion/measurement factors
 5. A recursive **UKF** (sigma points sampled around the current estimate, retracted onto $SE(3)$ and pushed through the exact, non-linearized `motion_model`/`observation_model` - no Jacobians at all, unlike the other three)
 6. A recursive **vanilla (linear) KF** - reparameterizes the pose as a redundant 12-dim ambient state `[vec(R), t]` instead of the minimal 6-dim $SE(3)$ tangent state, which makes the point-cloud observation model exactly linear (a fixed `H`, no Jacobian at all) at the cost of two compromises the manifold-aware methods avoid: the motion model needs a first-order (small-angle) truncation of `Exp(w)` - a real, growing source of mean error the EKF's exact group composition doesn't have, most visible at larger per-step rotation (e.g. larger `--dt`) - and nothing constrains `R` to stay in $SO(3)$, so it's explicitly re-projected back onto $SO(3)$ via SVD after every update. Its core recursion steps outside $SE(3)$ entirely in both versions (plain numpy arrays from `T[0:3,0:3]`/`T[0:3,3]` in `use_numpy`, `T.rotation()`/`T.translation()` in `use_manif`) - the `use_manif` version only touches manif for `SO3Tangent(.).hat()` and reconstructs the final pose via `scipy`'s `Rotation.from_matrix(...).as_quat()`, since manif exposes no rotation-matrix constructor.

Prints final/RMS rotation+position error per method, plus each method's empirical average per-step wall-clock time and peak memory (`measure_performance`, via `time.perf_counter` + `tracemalloc`), and plots rotation error, position error, and the x-y trajectory of all seven (ground truth included).

#### Scripts

- [use_numpy/pointcloud_pose_tracking.py](use_numpy/pointcloud_pose_tracking.py)

- [use_manif/pointcloud_pose_tracking.py](use_manif/pointcloud_pose_tracking.py)

#### Usage

```
uv run python use_numpy/pointcloud_pose_tracking.py --duration 5.0 --dt 0.1 --n-points 20 --vel-noise-std 0.05 --gyro-noise-std 0.02 --point-noise-std 0.03 --init-pose-noise-std 0.1 --gn-tol 1e-6 --gn-max-iters 20 --ukf-alpha 1.0 --ukf-beta 2.0 --ukf-kappa -3.0 --seed 0 --out out.png
```

- `--duration`: simulation length in seconds (default `5.0`)
- `--dt`: motion/measurement step interval in seconds (default `0.1`)
- `--n-points`: number of body-frame point-cloud landmarks (default `20`)
- `--vel-noise-std`: input linear-velocity noise std-dev, m/s (default `0.05`)
- `--gyro-noise-std`: input angular-velocity noise std-dev, rad/s (default `0.02`)
- `--point-noise-std`: point-cloud measurement noise std-dev, m (default `0.03`)
- `--init-pose-noise-std`: std-dev used to perturb the initial pose guess, and to set the prior/EKF-init covariance (default `0.1`)
- `--gn-tol`: batch Gauss-Newton convergence tolerance (default `1e-6`)
- `--gn-max-iters`: maximum batch Gauss-Newton iterations (default `20`)
- `--ukf-alpha`, `--ukf-beta`, `--ukf-kappa`: UKF scaled-unscented-transform tuning parameters (default `1.0`, `2.0`, `-3.0` - the classical Julier-Uhlmann scaling for this script's 6-dim tangent state, not the textbook tiny-`alpha` default, which collapses the sigma-point spread at this dimensionality)
- `--seed`: RNG seed (default `0`)
- `--out`: save the figure to this path instead of showing it (default: show)

### 5. 3D pose-graph relaxation

#### Purpose

A robot drives a closed 4-node square loop, accumulating drift from noisy relative-pose ("odometry") edges between consecutive nodes, then detects it has returned to the start and adds one loop-closure edge back to node 0. All node poses are jointly refined by Levenberg-Marquardt against every edge's residual $e_{ij} = \log(Z_{ij}^{-1} X_i^{-1} X_j)$, using analytical `compose`/`rminus` Jacobians chained together (hand-rolled $SE(3)$ $Exp$ / $Log$ / adjoint math via `lie_utils.py` in the `use_numpy` version; `manifpy`'s out-parameters in the `use_manif` version) - the same motion-factor Jacobian-chaining pattern as `run_batch_gn` in `pointcloud_pose_tracking.py`, generalized from a twist-based motion model to a directly-measured relative pose. Prints per-iteration chi-squared error plus final/RMS rotation+position error (uncorrected odometry vs. optimized), and plots the XY trajectory against ground truth.

#### Scripts

- [use_numpy/pose_graph.py](use_numpy/pose_graph.py)

- [use_manif/pose_graph.py](use_manif/pose_graph.py)

#### Usage

```
uv run python use_numpy/pose_graph.py --side-length 2.0 --pos-noise-std 0.05 --rot-noise-std 0.01 --loop-noise-scale 0.5 --damping 0.01 --gn-tol 1e-6 --gn-max-iters 10 --seed 0 --out out.png
```

- `--side-length`: side length of the square ground-truth loop, m (default `2.0`)
- `--pos-noise-std`: odometry-edge translation noise std-dev, m (default `0.05`)
- `--rot-noise-std`: odometry-edge rotation noise std-dev, rad (default `0.01`)
- `--loop-noise-scale`: noise std-dev multiplier for the loop-closure edge (default `0.5`)
- `--damping`: Levenberg-Marquardt damping factor (default `0.01`)
- `--gn-tol`: convergence tolerance on the correction step norm (default `1e-6`)
- `--gn-max-iters`: maximum optimization iterations (default `10`)
- `--seed`: RNG seed (default `0`)
- `--out`: save the figure to this path instead of showing it (default: show)

### 6. Bundle adjustment: joint pose + landmark refinement

#### Purpose

`n_cameras` cameras are placed on a horizontal arc around a cluster of `n_landmarks` 3D landmarks, each looking inward via a look-at rotation. A camera field-of-view cutoff means not every camera observes every landmark (landmarks seen by fewer than `--min-observations` cameras are dropped as not triangulable) - this is `docs/optimization/bundle_adjustment.md`'s observed set ${\mathcal{O}}$, a real strict subset of all camera-landmark pairs, not "every camera sees everything." Every camera pose and every landmark is perturbed from ground truth to build a noisy initial guess, then three solvers are compared:

- **Landmarks-only refinement**: poses held fixed at their noisy initial guess, only landmarks refined (the "cameras are correct" strawman) - decouples into independent $3\times3$ Gauss-Newton solves per landmark (classic triangulation from known poses).
- **Poses-only refinement**: landmarks held fixed, only poses refined (the "points are correct" strawman) - decouples into independent $6\times6$ Gauss-Newton solves per camera (classic PnP-style resection).
- **Full joint bundle adjustment**: both refined together in one coupled dense Gauss-Newton solve over every pose and landmark - the actual thing bundle adjustment is. This reintroduces the classic monocular BA gauge freedom (6-DoF rigid + 1-DoF scale ambiguity), fixed with a prior factor on the first two camera poses (mean = their own noisy initial guess), the same prior-factor pattern `pointcloud_pose_tracking.py`'s `run_batch_gn` already uses for its own gauge freedom. Since this only recovers the scene up to an unknown similarity transform, the result is aligned to ground truth via Umeyama's least-squares similarity fit (standard practice for evaluating monocular BA/SfM output) before computing absolute pose/ landmark error - reprojection error is unaffected by this alignment.

Camera-pose Jacobians come from a hand-derived closed form ($\partial p_c/\partial(\text{right-perturbation of }T) = [-I \mid \mathrm{skew}(p_c)]$, finite-difference verified) in the `use_numpy` version, and from chaining `manifpy`'s own `inverse`/`act` Jacobian out-parameters in the `use_manif` version - no manifold formula hand-rolled there. Prints RMS pose rotation/position error, landmark error, and reprojection error for all four rows (noisy init, landmarks-only, poses-only, joint BA), and plots a top-down scene view (ground truth vs. noisy init vs. joint BA) alongside a grouped bar chart of the four RMS metrics.

#### Scripts

- [use_numpy/bundle_adjustment.py](use_numpy/bundle_adjustment.py)

- [use_manif/bundle_adjustment.py](use_manif/bundle_adjustment.py)

#### Usage

```
uv run python use_numpy/bundle_adjustment.py --n-cameras 8 --n-landmarks 60 --camera-radius 5.0 --arc-span-deg 180 --landmark-spread 2.0 --fov-deg 70 --image-width 640 --image-height 480 --focal-length 800 --pose-noise-std 0.1 --landmark-noise-std 0.3 --pixel-noise-std 1.0 --min-observations 2 --gn-tol 1e-6 --gn-max-iters 30 --seed 0 --out out.png
```

- `--n-cameras`: number of cameras placed on the arc (default `8`)
- `--n-landmarks`: number of 3D landmarks sampled, before dropping under-observed ones (default `60`)
- `--camera-radius`: radius of the camera arc, centered on the landmark centroid, m (default `5.0`)
- `--arc-span-deg`: total angular span of the camera arc, deg (default `180`)
- `--landmark-spread`: half-width of the cube landmarks are sampled in, m (default `2.0`)
- `--fov-deg`: camera full field-of-view angle; controls which landmarks each camera observes, deg (default `70`)
- `--image-width`, `--image-height`: image size in pixels, sets `cx`/`cy` (default `640`/`480`)
- `--focal-length`: shared `fx=fy` focal length in pixels (default `800`)
- `--pose-noise-std`: std-dev of the se3 twist used to perturb the initial camera-pose guess, mixed m/rad (default `0.1`)
- `--landmark-noise-std`: std-dev of the Gaussian offset used to perturb the initial landmark guess, m (default `0.3`)
- `--pixel-noise-std`: std-dev of Gaussian pixel measurement noise, px (default `1.0`)
- `--min-observations`: minimum observing cameras a landmark needs to be kept; must be `>= 2` (default `2`)
- `--gn-tol`: Gauss-Newton convergence tolerance (default `1e-6`)
- `--gn-max-iters`: maximum Gauss-Newton iterations (default `30`)
- `--seed`: RNG seed (default `0`)
- `--out`: save the figure to this path instead of showing it (default: show)

### 7. Local + Global bundle adjustment: bounded windows vs. whole-map re-solves

#### Purpose
`docs/optimization/bundle_adjustment.md`'s Local-vs-Global-BA section (§13) notes that `bundle_adjustment.py`'s solvers are all single-batch joint solves with no windowing, no covisibility graph, and no incremental registration - this script is what fills that gap. `n_keyframes` keyframes move one at a time along a forward-facing arc through a corridor of landmarks scattered near the path (not a static ring around a shared centroid), with a field-of-view **and** `--max-view-range` cutoff so a landmark is only visible from a short run of nearby keyframes - real feature detectors have a finite effective range, and without this cutoff a forward-facing camera would see nearly every landmark from nearly every keyframe, destroying the locality the whole exercise depends on. Each keyframe arrives with a front-end pose estimate built by chaining a noisy relative motion onto the previous one (`simulate_frontend_trajectory`, the same dead-reckoning pattern as `pose_graph.py`), so small per-step errors compound into real drift.

The incremental loop (`run_incremental_local_ba`) runs keyframe-by-keyframe:

- **Reveal + triangulate**: each keyframe's observations are revealed one at a time; a landmark is triangulated (closed-form ray intersection, then a few Gauss-Newton iterations) the moment it crosses `--min-observations` observers, and committed only if it passes a cheirality (positive-depth) check - a weakly-constrained point can otherwise converge to a reflection behind a camera that still fits that view's pixel almost exactly, since $\text{pixel} = fx/z$ is invariant under negating a whole camera-frame point.
- **Bounded local BA window** (`build_active_window` + `run_local_ba_step`): the new keyframe plus its strongest covisible neighbors (up to `--max-window-keyframes`) are optimized together; every other keyframe that also observes one of the window's landmarks is held fixed as a rigid anchor. Keyframes 0/1 are hard-fixed forever as the gauge anchor - no prior factor needed, since a hard anchor already pins the gauge.
- **Periodic Global BA** (`run_global_ba`): every `--global-ba-interval` keyframes (plus once at the end), every keyframe and landmark seen so far is jointly re-solved in one system, for direct comparison against the bounded local window.

Both solvers share one Levenberg-Marquardt core (`run_windowed_gn_lm`): a step is only accepted if it actually reduces total reprojection error, otherwise the damping grows and the step is retried - plain fixed-damping Gauss-Newton (as `bundle_adjustment.py` uses for its densely-observed, prior-anchored toy scene) was found to diverge explosively on this script's weakly-constrained early windows. The incremental loop runs twice off the same scene - once with Global BA disabled, once with it enabled - and prints/plots wall-clock solve time (should stay flat for the local window, grow for Global BA as the map grows) and a trailing-window RMS trajectory error (drift accumulating vs. periodically corrected). One finding worth being upfront about: on the default open path, Global BA has no *new* geometric constraint to exploit beyond what the overlapping local windows already used - and, across a wider seed sweep than any single run shows, it's close to a wash against Local-only here (~50% per-seed win rate, median RMS-trajectory-error difference near zero across 15 seeds - see this script's own regression test), not a reliable win the way an actual loop closure is. A few individual seeds also diverge to thousands of meters in *either* run mode - a rare bad local minimum in the windowed GN/LM solve that neither mode is protected from, and occasionally backend-specific (one seed diverges in the numpy backend's Local-only run but not the manif backend's, for the identical nominal scenario). The covisibility/window/Global-BA machinery itself is already loop-closure-agnostic, though - see "Loop closure example" below for a path that actually has one.

#### Scripts

- [use_numpy/bundle_adjustment_advanced.py](use_numpy/bundle_adjustment_advanced.py) 

- [use_manif/bundle_adjustment_advanced.py](use_manif/bundle_adjustment_advanced.py)

#### Usage

```
uv run python use_numpy/bundle_adjustment_advanced.py --n-keyframes 50 --path-radius 15.0 --arc-span-deg 90 --landmarks-per-keyframe 8 --lateral-spread 2.0 --vertical-spread 1.0 --fov-deg 70 --max-view-range 6.0 --image-width 640 --image-height 480 --focal-length 800 --min-observations 3 --min-shared-for-covisibility 2 --max-window-keyframes 6 --global-ba-interval 8 --relative-pose-noise-std 0.02 --pixel-noise-std 1.0 --gn-tol 1e-6 --gn-max-iters 15 --seed 0 --out out.png
```

- `--n-keyframes`: number of keyframes along the path (default `50`)
- `--path-radius`: radius of the arc the path follows, m (default `15.0`)
- `--arc-span-deg`: total angular span of the path, deg (default `90`)
- `--landmarks-per-keyframe`: landmarks scattered per keyframe station (default `8`)
- `--lateral-spread`, `--vertical-spread`: half-width of the lateral/vertical landmark offset, m (default `2.0`/`1.0`)
- `--fov-deg`: camera full field-of-view angle, deg (default `70`)
- `--max-view-range`: maximum camera-to-landmark detection range, m - bounds covisibility to nearby keyframes (default `6.0`)
- `--image-width`, `--image-height`: image size in pixels, sets `cx`/`cy` (default `640`/`480`)
- `--focal-length`: shared `fx=fy` focal length in pixels (default `800`)
- `--min-observations`: minimum observing keyframes before a landmark is triangulated; must be `>= 2` (default `3`)
- `--min-shared-for-covisibility`: minimum shared-landmark count for a covisibility edge between two keyframes (default `2`)
- `--max-window-keyframes`: maximum active keyframes per local BA window, including the new one (default `6`)
- `--global-ba-interval`: run a full Global BA pass every this many keyframes, plus once at the end (default `8`)
- `--loop-closure-min-gap`: minimum keyframe-index gap for a covisibility edge to count as a loop closure rather than ordinary local covisibility (default `20`)
- `--relative-pose-noise-std`: std-dev of the se3 twist noise added to each frame-to-frame front-end pose estimate, mixed m/rad - compounds into drift (default `0.02`)
- `--pixel-noise-std`: std-dev of Gaussian pixel measurement noise, px (default `1.0`)
- `--gn-tol`: Levenberg-Marquardt convergence tolerance (default `1e-6`)
- `--gn-max-iters`: maximum accepted GN/LM steps per solve (default `15`)
- `--seed`: RNG seed (default `0`)

#### Loop closure example

The default `--arc-span-deg 90` path never revisits a place, so Global BA never gets a genuinely new constraint. Pushing the span close to 360 degrees (at the *same* `--n-keyframes 50` - no other change needed) swings the path's end back within view range of its own start, so a late keyframe re-observes an early landmark - a real loop closure, picked up automatically by the same covisibility/window/Global-BA code with zero logic changes:

```
uv run python use_numpy/bundle_adjustment_advanced.py --arc-span-deg 350 --seed 0 --out loop_closure.png
```

This prints an extra line (in place of the "never revisits" caveat) identifying exactly which keyframe closed the loop and against which earlier one, plus the measured trajectory-RMS improvement once the next Global BA pass exploits it, and adds a labeled green "Loop closure" marker to the drift plot at that keyframe.

### 8. Incremental (square-root SAM) vs. batch pose-graph solving

#### Purpose

`docs/optimization/isam_optimization.md` describes iSAM's core mechanism - absorbing a new measurement into an existing square-root-information factorization instead of rebuilding the whole linear system from scratch - but `pose_graph.py`'s batch solver always rebuilds from scratch. This script fills that gap on a longer version of the same square loop (`--nodes-per-side` nodes per side instead of `pose_graph.py`'s fixed 4 corners), streamed in one node/odometry-edge at a time, and runs two solvers over the identical stream:

- **Batch streaming** (`run_batch_streaming`): the expensive baseline - every new node triggers a full re-solve of the entire graph so far, from a fresh dead-reckoning guess, by calling `pose_graph.py`'s own `run_pose_graph_optimization` unmodified.
- **Incremental square-root SAM** (`run_incremental_pose_graph`): absorbs each new odometry edge into a running upper-triangular square-root-information matrix via Givens-rotation row insertion (`qr_insert_row` in `utils.py`), touching only the new node's columns - no other row is re-evaluated. Every `--relinearize-every` nodes, and unconditionally on the loop-closure edge (which, unlike a sequential odometry edge, connects the newest node all the way back to the first one), it instead relinearizes fully: adopts the current estimate as a new linearization point and rebuilds the whole system from one `np.linalg.qr` call, iterating to Gauss-Newton convergence exactly like the batch solver does.

This deliberately does **not** implement iSAM2's Bayes tree (selective relinearization of only the affected subtree) or variable reordering - see the module docstring and `docs/optimization/isam_optimization.md` §14 for exactly what is/isn't in scope. Both solvers are timed via `utils.py`'s `measure_performance` and converge to matching final/RMS pose error; the printed relinearization counts and per-node wall-clock time are the actual point of the comparison.

#### Scripts

- [use_numpy/pose_graph_incremental.py](use_numpy/pose_graph_incremental.py)

- [use_manif/pose_graph_incremental.py](use_manif/pose_graph_incremental.py)

#### Usage

```
uv run python use_numpy/pose_graph_incremental.py --side-length 2.0 --nodes-per-side 16 --pos-noise-std 0.05 --rot-noise-std 0.01 --loop-noise-scale 0.5 --damping 0.01 --gn-tol 1e-6 --gn-max-iters 10 --anchor-weight 1e6 --relinearize-every 8 --seed 0 --out out.png
```

- `--side-length`: side length of the square ground-truth loop, m (default `2.0`)
- `--nodes-per-side`: nodes per side of the loop, streamed in one at a time (default `16`)
- `--pos-noise-std`: odometry-edge translation noise std-dev, m (default `0.05`)
- `--rot-noise-std`: odometry-edge rotation noise std-dev, rad (default `0.01`)
- `--loop-noise-scale`: noise std-dev multiplier for the loop-closure edge (default `0.5`)
- `--damping`: Levenberg-Marquardt damping for the batch-streaming baseline only (default `0.01`)
- `--gn-tol`: Gauss-Newton convergence tolerance (default `1e-6`)
- `--gn-max-iters`: maximum Gauss-Newton iterations per solve/relinearization (default `10`)
- `--anchor-weight`: information weight of the node-0 gauge-fixing prior (default `1e6`)
- `--relinearize-every`: force a full relinearization every this many new nodes (default `8`)
- `--seed`: RNG seed (default `0`)
- `--out`: save the figure to this path instead of showing it (default: show)

### 9. Bayes tree construction and affected-region query

#### Purpose

`docs/optimization/bayes_tree.md` and `docs/optimization/isam2_optimization.md` describe the Bayes tree iSAM2 builds from a factor graph's elimination order, and the key claim that makes it useful: a new factor only invalidates the affected part of the tree, not the whole thing. This script makes that concrete instead of leaving it to ASCII diagrams - it builds an actual Bayes tree over `pose_graph.py`'s square-loop topology via symbolic variable elimination (`symbolic_eliminate` in `utils.py`: eliminating a node connects its still-uneliminated neighbors, and the resulting clique's parent is whichever neighbor is eliminated next), fixing the elimination order as oldest-node-first (not COLAMD - that dynamic reordering is explicitly out of scope, see `docs/optimization/isam2_optimization.md` §12).

It then answers `bayes_tree_affected_path`'s (`utils.py`) "what's affected" query for two scenarios: an ordinary new odometry edge, and the loop-closure edge. Worked out by hand before writing the script and confirmed by its own printed output: under oldest-first elimination the tree is always a straight chain here, and because the loop-closure edge connects the *first*-eliminated node (the deepest leaf) to the root, it invalidates the *entire* chain - the worst case a fill-reducing reordering like COLAMD exists specifically to avoid, versus an ordinary odometry edge near the root which only invalidates 2 nodes regardless of graph size.

This is pure index/graph bookkeeping - no pose math, $SE(3)$, or Lie algebra anywhere - so, like 5 other numpy-only scripts in this repo that also have no Lie-group math to delegate to manif (`friction_anisotropic_ekf.py`, `inchworm_zupt_ekf.py`, `pnp_estimation.py`, `saltation_matrix_ekf.py`, `sliding_window_marginalization.py`), there is no `use_manif/` counterpart; the result would be structurally identical either way, since the tree only depends on which node indices a factor connects, never the noisy relative-pose values themselves. It implements the Bayes tree's *construction* and *affected-region query* only, not the numeric fluid-relinearization solve (`pose_graph_incremental.py`'s iSAM v1 square-root-SAM update handles that, without ever building a Bayes tree) or dynamic reordering.

#### Scripts

- [use_numpy/bayes_tree_construction.py](use_numpy/bayes_tree_construction.py)

#### Usage

```
uv run python use_numpy/bayes_tree_construction.py --nodes-per-side 4 --out out.png
```

- `--nodes-per-side`: nodes per side of the ring pose graph, same topology as `pose_graph_incremental.py` (default `4`; kept small since this plots an inspectable diagram, not a timing benchmark)
- `--out`: save the figure to this path instead of showing it (default: show)

### 10. Perspective-n-Point (PnP): recovering a camera pose from known 3D-2D correspondences

#### Purpose

`docs/frontend/triangulation_pnp.md` frames triangulation and PnP as the same reprojection problem run in opposite directions: triangulation (`bundle_adjustment_advanced.py`'s `triangulate_landmark`/`refine_landmark_gn`) holds camera poses fixed to solve for an unknown 3D point; PnP holds a set of known 3D points fixed and solves for the unknown camera pose that observed them. This script implements PnP with the same two-step recipe used throughout this codebase: a closed-form linear initial guess, then a few Gauss-Newton iterations against the true nonlinear reprojection error.

`linear_pnp_dlt` solves the classic Direct Linear Transform (DLT) camera-resectioning problem specialized to known intrinsics - each calibrated ray is parallel to its camera-frame point, giving a linear homogeneous constraint on the flattened world-to-camera $[R \mid t]$, solved via the smallest right-singular vector and then projected onto the nearest proper rotation (SVD orthogonalization), with scale and sign fixed from that same decomposition and a positive-depth check (this repo's PnP analogue of `passes_cheirality`). `refine_pose_gn` then runs ordinary Gauss-Newton on the true reprojection residual, updating the pose via a right-multiplicative $SE(3)$ correction, mirroring `refine_landmark_gn`'s loop.

#### Scripts

- [use_numpy/pnp_estimation.py](use_numpy/pnp_estimation.py)

#### Usage

```
uv run python use_numpy/pnp_estimation.py --n-points 20 --pixel-noise-std 1.0 --seed 0 --out out.png
```

- `--n-points`: number of 3D-2D correspondences (default `20`)
- `--image-width`/`--image-height`/`--focal-length`: pinhole intrinsics (defaults `640`/`480`/`800.0`)
- `--pixel-noise-std`: std-dev of Gaussian pixel noise added to each observation (px, default `1.0`)
- `--gn-tol`/`--gn-max-iters`: Gauss-Newton convergence tolerance and iteration cap (defaults `1e-8`/`20`)
- `--seed`: RNG seed
- `--out`: save the figure to this path instead of showing it (default: show)

### 11. Saltation-matrix EKF: tracking a point mass through discrete ground-contact events

#### Purpose

Every other filtering script in this repo assumes smooth, continuous motion between measurements. `docs/filtering/hybrid_saltation_ekf.md` introduces the one exception: a **hybrid dynamical system**, where a point mass in free-fall bounces (inelastically, restitution `e`) off the ground $p_z = 0$. Between bounces the dynamics is exactly linear (mean/covariance propagate exactly, no linearization at all), so any disagreement between the two EKF variants below is attributable entirely to how each one handles the bounce itself.

Three methods are compared: 
- A **dead-reckoning baseline**: propagates the exact dynamics from an uncertain initial guess, never looking at measurements - an initial-condition error still causes growing error, since bounce *timing* is sensitive to the state even though the dynamics model itself is exact
- An **EKF with naive bounce handling**: $P^+ = DR\,P^-\,DR^\top$, the reset map's own Jacobian alone - the common mistake, since it is not the correct linearization of "post-impact state as a function of pre-impact state" once a perturbed trajectory reaches the guard at a different time
- An **EKF with saltation-corrected bounce handling**: $P^+ = \Xi\,P^-\,\Xi^\top$, the true `saltation_matrix` composed with the ordinary flow Jacobians before/after the bounce (the fixed-tick comparison this filter actually needs), derived from first principles and verified against a finite-difference ground truth to ~7e-6 after a formula correct for a *different* comparison (each trajectory at its own crossing time, not a shared fixed tick) was checked against this same ground truth and found wrong by a wide margin

A Monte Carlo consistency check (`run_monte_carlo_consistency`, NEES - Normalized Estimation Error Squared - new to this repo) repeats both EKFs over many independent noise realizations of the same nominal trajectory. The finding runs slightly against "naive is overconfident, saltation fixes it": the saltation matrix reduces, but does not zero out, the guard-normal (height) direction's *reported* variance at every bounce ($Dg\,\Xi = -e\,Dg$ identically, verified) - most accurate only if the filter's own estimated bounce time exactly coincides with the true one, which it generally will not with any real tracking error. Empirically (reproduced across multiple seeds), this makes the saltation-corrected EKF's post-bounce NEES modestly, consistently *higher* than the naive EKF's (~5-6% at this script's defaults), not lower - a small but real instance of the same underlying gap: saltation matrices assume a known transition time, but contact/phase detection is itself uncertain (see the doc for the full mechanism and how the gap grows, still modestly, once detection jitter is added). Both EKFs' mean trajectories look nearly identical throughout regardless - this entire effect is invisible in the point estimate.

Plain $\mathbb{R}^6$ state (position/velocity, no rotation) - no `use_manif/` counterpart, for the same reason as `bayes_tree_construction.py`.

#### Scripts

- [use_numpy/saltation_matrix_ekf.py](use_numpy/saltation_matrix_ekf.py)

#### Usage

```
uv run python use_numpy/saltation_matrix_ekf.py --duration 5.0 --dt 0.02 --restitution 0.85 --gravity 9.81 --drop-height 5.0 --init-horizontal-vel 1.0 0.5 --pos-noise-std 0.03 --process-noise-std 0.3 --impact-noise-std 0.005 --init-pos-noise-std 0.1 --init-vel-noise-std 0.2 --detect-time-bias 0.0 --detect-time-noise-std 0.0 --n-mc-trials 500 --seed 0 --out out.png
```

- `--duration`: simulation length in seconds (default `5.0`) - kept comfortably below the trajectory's own Zeno settling time (bounce intervals shrink geometrically for a lossy bounce; the default stops after 3 well-separated bounces)
- `--dt`: filter/measurement step interval in seconds (default `0.02`)
- `--restitution`: bounce restitution coefficient, 0-1 (default `0.85`)
- `--gravity`: gravitational acceleration magnitude, $\text{m/s}^2$ (default `9.81`)
- `--drop-height`: initial height, m (default `5.0`)
- `--init-horizontal-vel`: initial horizontal velocity, m/s (default `1.0 0.5`)
- `--pos-noise-std`: position measurement noise std-dev, m (default `0.03`)
- `--process-noise-std`: assumed acceleration-disturbance noise std-dev, $\text{m/s}^2$ (default `0.3`)
- `--impact-noise-std`: std-dev of the regularizing floor added to `P` at each bounce, both EKF variants (default `0.005`) - no longer numerically load-bearing (both filters stay well-behaved even at `0`), but still shifts both filters' absolute NEES level together as it grows (see the doc §7)
- `--init-pos-noise-std`/`--init-vel-noise-std`: std-dev used to perturb the initial position/velocity guess (defaults `0.1`/`0.2`)
- `--detect-time-bias`: systematic contact-detection timing offset, s, positive = late detection (default `0.0`, exact detection) - models a real contact sensor's own detection latency, on top of whatever the filter's state estimate already gets wrong about the geometric crossing time
- `--detect-time-noise-std`: std-dev, s, of Gaussian contact-detection timing jitter added on top of `--detect-time-bias` each time a bounce is detected (default `0.0`) - see doc §8 for the measured saltation-vs-naive NEES gap this opens up as jitter grows
- `--n-mc-trials`: number of Monte Carlo consistency trials (default `500`)
- `--seed`: RNG seed
- `--out`: save the figure to this path instead of showing it (default: show)

### 12. Inchworm ZUPT: exploiting a known anchor/dwell schedule

#### Purpose

`docs/filtering/inchworm_zupt_ekf.md` works through a known-schedule anchor(dwell)/extend gait - a 1D point mass alternates between being exactly stationary (anchor) and moving at a commanded cruise speed (extend), state $x=[p,v]$. Unlike `saltation_matrix_ekf.py`, this is *not* a discontinuous-state-reset problem - velocity is externally commanded per phase, an ordinary switched-linear system - so the entire interesting question sits on the *measurement* side: whether the filter exploits the free zero-velocity information an anchor phase provides. Three ZUPT policies sharing an identical predict step and the same noisy position measurement every tick are compared via the same Monte Carlo NEES pattern as `saltation_matrix_ekf.py`:

- **`never`**: only the position measurement is ever fused - never wrong, but leaves free information on the table during every anchor tick.
- **`always`**: an unconditional ZUPT is fused every tick regardless of true phase - the schedule-unaware mistake.
- **`phase_conditional`**: ZUPT is fused only on ticks the known schedule marks as anchor - the correct policy.

The finding runs deeper than "`always` is wrong while moving": it's dramatically worse than `phase_conditional` in *both* the anchor and cruise windows, not just during motion, because the overconfidence from misapplying ZUPT during cruise bleeds into the next anchor phase. A second, more subtle finding echoes `saltation_matrix_ekf.py`'s own: `never`, despite discarding real information, ends up at least as well *calibrated* (NEES) as `phase_conditional`, even though `phase_conditional` has the best raw accuracy (lowest velocity RMS) of the three - see the doc for the full numbers and mechanism.

#### Scripts

- [use_numpy/inchworm_zupt_ekf.py](use_numpy/inchworm_zupt_ekf.py)

#### Usage

```
uv run python use_numpy/inchworm_zupt_ekf.py --duration 10.0 --dt 0.05 --t-anchor 1.0 --t-extend 1.0 --t-ramp 0.2 --v-extend 0.1 --pos-noise-std 0.02 --process-noise-std 0.15 --zupt-noise-std 0.01 --init-pos-noise-std 0.05 --init-vel-noise-std 0.05 --n-mc-trials 500 --seed 0 --out out.png
```

- `--duration`: simulation length in seconds (default `10.0`)
- `--dt`: filter/measurement step interval in seconds (default `0.05`)
- `--t-anchor`: anchor/dwell phase duration per cycle, s (default `1.0`)
- `--t-extend`: extend phase duration per cycle (including its ramps), s (default `1.0`)
- `--t-ramp`: linear ramp-up/ramp-down duration at each end of the extend phase, s (default `0.2`) - keeps `true_velocity` continuous everywhere; see the doc's §2 calibration trap for why this (and its interaction with `--process-noise-std`) matters
- `--v-extend`: commanded cruise speed during the extend phase, m/s (default `0.1`)
- `--pos-noise-std`: position measurement noise std-dev, m (default `0.02`)
- `--process-noise-std`: assumed acceleration-disturbance noise std-dev, $\text{m/s}^2$ (default `0.15`)
- `--zupt-noise-std`: ZUPT pseudo-measurement noise std-dev, m/s (default `0.01`)
- `--init-pos-noise-std`/`--init-vel-noise-std`: std-dev used to perturb the initial position/velocity guess (defaults `0.05`/`0.05`)
- `--n-mc-trials`: number of Monte Carlo consistency trials (default `500`)
- `--seed`: RNG seed
- `--out`: save the figure to this path instead of showing it (default: show)

### 13. Friction-anisotropic process noise: fixed vs. heading-aware Q

#### Purpose

`docs/filtering/friction_anisotropic_ekf.md` applies this repo's own `pointcloud_pose_tracking_empirical_note.md` argument (a noise ellipsoid fixed in a body frame looks anisotropic-and-rotating in the world frame) to *process* noise instead of measurement noise. A unicycle drives an exact, known circular arc (heading is exact and noise-free, driven by a known commanded turn rate) while true position picks up a random slip disturbance each tick, drawn anisotropically in the pad's own body frame - low variance along grip/forward, high variance along slip/lateral - then rotated into world coordinates by the *true* heading. Three ways of building the position block of the filter's process-noise covariance $Q$ are compared, sharing an identical predict step and the same noisy position measurement each tick:

- **`isotropic`**: direction-blind, same total noise budget as the other two.
- **`fixed_anisotropic`**: correctly anisotropic in shape, but fixed to the heading at $t=0$ and never updated - the mistake of knowing the pad is anisotropic without re-deriving $Q$ as the robot turns.
- **`heading_aware`**: correctly anisotropic and re-oriented every predict step using the filter's own current heading estimate - the correct policy.

`fixed_anisotropic` comes out dramatically worse than *both* alternatives at every heading checkpoint, and the worst mismatch sits at a 90° heading difference from its stale reference, not 180° - a covariance ellipse has period $\pi$, not $2\pi$. See the doc for the full Monte Carlo NEES table and an honestly-reported non-robust nuance near 180°.

#### Scripts

- [use_numpy/friction_anisotropic_ekf.py](use_numpy/friction_anisotropic_ekf.py)

#### Usage

```
uv run python use_numpy/friction_anisotropic_ekf.py --duration 20.0 --dt 0.05 --v-cmd 0.2 --sigma-grip 0.02 --sigma-slip 0.1 --sigma-theta 0.01 --pos-noise-std 0.02 --init-pos-noise-std 0.05 --init-theta-noise-std 0.05 --n-mc-trials 500 --seed 0 --out out.png
```

- `--duration`: simulation length in seconds (default `20.0`)
- `--dt`: filter/measurement step interval in seconds (default `0.05`)
- `--v-cmd`: constant commanded forward speed, m/s (default `0.2`)
- `--omega-cmd`: constant commanded turn rate, rad/s (default: $2\pi/\text{duration}$, i.e. exactly one full loop, so heading sweeps every orientation relative to the fixed reference)
- `--sigma-grip`/`--sigma-slip`: true body-frame slip std-dev along the grip/forward and slip/lateral axes, m/s (defaults `0.02`/`0.1`)
- `--sigma-theta`: assumed heading process-noise std-dev, all three filter variants, rad/s (default `0.01`)
- `--pos-noise-std`: position measurement noise std-dev, m (default `0.02`)
- `--init-pos-noise-std`/`--init-theta-noise-std`: std-dev used to perturb the initial position/heading guess (defaults `0.05`/`0.05`)
- `--n-mc-trials`: number of Monte Carlo consistency trials (default `500`)
- `--seed`: RNG seed
- `--out`: save the figure to this path instead of showing it (default: show)

### 14. Sliding-window marginalization: bounded-memory pose-graph smoothing

#### Purpose

`docs/optimization/marginalization.md` §4 derives the Schur complement that turns an eliminated pose into a prior factor over its surviving neighbors; this script is the accompanying implementation the doc used to say was missing. A pure odometry chain (deliberately no loop closures, so the marginal produced at each step is provably unary - isolating the memory-*bounding* property from the fill-in problem `bayes_tree.md`/`isam2_optimization.md` already cover) streams in one node at a time. Whenever the live window would exceed `--window-size`, the oldest pose is marginalized out via the Schur complement and dropped; every other pose keeps optimizing inside a bounded-size Gauss-Newton solve. This is compared against `run_full_batch_growing` - the unbounded baseline that re-solves the entire graph from scratch at every new node - across a sweep of trajectory lengths (`--nodes-per-side-sweep`). The measured result: full-batch's largest dense information matrix (`max_dof`) grows linearly with trajectory length (so its memory grows quadratically), while sliding-window's caps at exactly `6 * window_size` the moment the window first fills and never moves again - with *identical* final RMS position error between the two, a consequence specific to this script's no-loop-closure scope (see the doc §9 for the full numbers and why accuracy isn't sacrificed here).

#### Scripts

- [use_numpy/sliding_window_marginalization.py](use_numpy/sliding_window_marginalization.py)

#### Usage

```
uv run python use_numpy/sliding_window_marginalization.py --nodes-per-side-sweep 2 4 8 16 32 64 --side-length 2.0 --window-size 10 --pos-noise-std 0.05 --rot-noise-std 0.01 --anchor-weight 1e6 --gn-tol 1e-6 --gn-max-iters 10 --seed 0 --out out.png
```

- `--nodes-per-side-sweep`: sweep of nodes-per-side values, trajectory has 4x this many total poses per entry (default `2 4 8 16 32 64`) - the resource-constraint result is how cost scales across this sweep, not any single run
- `--side-length`: side length of the square ground-truth path, m (default `2.0`)
- `--window-size`: sliding-window size, poses kept live at once (default `10`)
- `--pos-noise-std`: odometry-edge translation noise std-dev, m (default `0.05`)
- `--rot-noise-std`: odometry-edge rotation noise std-dev, rad (default `0.01`)
- `--anchor-weight`: information weight of the node-0 gauge-fixing prior (default `1e6`)
- `--gn-tol`: Gauss-Newton convergence tolerance (default `1e-6`)
- `--gn-max-iters`: maximum Gauss-Newton iterations per solve (default `10`)
- `--seed`: RNG seed
- `--out`: save the figure to this path instead of showing it (default: show)
