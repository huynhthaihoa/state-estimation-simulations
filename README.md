# State Estimation Simulations

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

### [use_numpy/](use_numpy/) - Plain numpy, hand-rolled Lie-group math

- [lie_utils.py](use_numpy/lie_utils.py): shared module of the hand-rolled Lie-group helpers (`skew`, `rotation_geodesic_error`, `so3_exp`, `so3_right_jacobian`, `se3_exp`, `se3_log`, `se3_inv`, `se3_adjoint`, `compute_so3_inv_right_jacobian`, `compute_se3_inv_right_jacobian`, `se3_right_jacobian`), imported by every other script in this directory (`imu_integration_comparison.py`, `robot_imu_simulation.py`, `imu_preintegration.py`, `pointcloud_pose_tracking.py`, `pose_graph.py`, and `bundle_adjustment.py`) - none of them keep their own inline copies of the skew/Jacobian helpers anymore.
- [imu_integration_comparison.py](use_numpy/imu_integration_comparison.py): naive Euler-angle vs. $SO(3)$ exp-map orientation integration.
- [robot_imu_simulation.py](use_numpy/robot_imu_simulation.py): high-rate IMU propagation + a low-rate Gauss-Newton pose correction (à la a GPS fix), reusing `lie_utils.py`'s $SE(3)$ $Exp$ / $Log$ / inverse-right-Jacobian math.
- [imu_preintegration.py](use_numpy/imu_preintegration.py): IMU pre-integration - compresses a burst of high-frequency IMU samples into one relative measurement plus first-order bias Jacobians, then shows an instant Taylor-expansion correction when the bias estimate changes, without re-integrating.
- [pointcloud_pose_tracking.py](use_numpy/pointcloud_pose_tracking.py): tracks a rigid object's pose from a motion-model prior (noisy control inputs) fused with noisy point-cloud measurements of its known geometry, comparing a recursive **EKF**, a recursive **invariant EKF** (body-frame residual, state-independent measurement Jacobian), a **batch Gauss-Newton** smoother over the whole trajectory, and a recursive **UKF** (sigma-point unscented transform via right-perturbation retraction, no Jacobians at all), reusing `lie_utils.py`'s $SE(3)$ $Exp$ / $Log$ /inverse-right-Jacobian math plus a hand-rolled adjoint/right-Jacobian for the motion-model Jacobians. Also reports empirical per-step time and peak memory for each of the five approaches.
- [pose_graph.py](use_numpy/pose_graph.py): a small closed-loop 3D pose-graph relaxation (odometry drift + one loop closure), jointly optimized via Levenberg-Marquardt, reusing the same `lie_utils.py` $SE(3)$ $Exp$ / $Log$ / inverse-right-Jacobian/adjoint math as `pointcloud_pose_tracking.py`.
- [pose_graph_incremental.py](use_numpy/pose_graph_incremental.py): the [iSAM](docs/optimization/isam_optimization.md) counterpart to `pose_graph.py`'s batch solver - a longer loop streams in one node at a time, and an incremental square-root-SAM solver (Givens-rotation QR row insertion via `qr_insert_row` in `utils.py`, plus periodic/loop-closure-triggered full relinearization) is timed against a from-scratch batch re-solve at every step.
- [bayes_tree_construction.py](use_numpy/bayes_tree_construction.py): builds the actual [Bayes tree](docs/optimization/bayes_tree.md) `iSAM2` relies on - symbolic variable elimination (`symbolic_eliminate` in `utils.py`) over the same square-loop pose-graph topology, then a root-ward affected-path query (`bayes_tree_affected_path`) contrasting a local odometry edge against the loop-closure edge. Pure index/graph bookkeeping, no pose math at all, so there's no `use_manif/` counterpart - the result would be structurally identical either way.
- [bundle_adjustment.py](use_numpy/bundle_adjustment.py): jointly refines camera poses **and** 3D landmarks against pinhole reprojection error - cameras on an arc around a landmark cluster, with a field-of-view cutoff so not every camera observes every landmark. Compares three solvers: **landmarks-only refinement** and **poses-only refinement** (independent 3x3/6x6 GN solves, each a "fix one side" strawman) against **full joint bundle adjustment** (coupled dense GN over poses + landmarks, gauge-fixed with a prior factor on the first two camera poses, then [Umeyama-aligned](docs/foundations/umeyama_alignment.md) to ground truth before reporting absolute error, since monocular BA only recovers the scene up to an unknown similarity transform).
- [bundle_adjustment_advanced.py](use_numpy/bundle_adjustment_advanced.py): Local **and** Global bundle adjustment, run back to back for direct comparison - the real-time-system-behavior counterpart to `bundle_adjustment.py`'s single-batch scene. A camera moves keyframe-by-keyframe along a forward-facing arc through a landmark corridor instead of sitting on a static ring; a covisibility graph builds incrementally, and every new keyframe triggers a bounded local Gauss-Newton/Levenberg-Marquardt solve over an active window (new keyframe + covisible neighbors, with every other observing keyframe held fixed as a rigid anchor), while a periodic Global BA pass jointly re-solves the whole map so far for contrast. Keyframes 0/1 are hard-fixed forever as the gauge anchor - no prior factor needed, unlike `bundle_adjustment.py`, since a hard anchor already pins the gauge with no residual freedom left to constrain. Guards against Gauss-Newton divergence (Levenberg-Marquardt damping) and the classic point-behind-camera reflection ambiguity (`passes_cheirality`/ `cull_invalid_points`) that a weakly-constrained, forward-motion scene can hit but `bundle_adjustment.py`'s densely-observed toy scene never does.
- [pnp_estimation.py](use_numpy/pnp_estimation.py): Perspective-n-Point (PnP) - triangulation's exact inverse ([docs/frontend/triangulation_pnp.md](docs/frontend/triangulation_pnp.md)): given known 3D points and their observed pixels, recovers the unknown camera pose via a closed-form linear DLT initial guess (specialized to known intrinsics), then a few Gauss-Newton iterations against the true reprojection error. No `use_manif/` counterpart (single-implementation, like `bayes_tree_construction.py`).

### [use_manif/](use_manif/) - Same simulations, on `manifpy`

- [imu_integration_comparison.py](use_manif/imu_integration_comparison.py)
- [robot_imu_simulation.py](use_manif/robot_imu_simulation.py)
- [imu_preintegration.py](use_manif/imu_preintegration.py): same bias-Jacobian preintegration bundle, with the $SO(3)$ $Exp$ map / right Jacobian / skew(hat) math delegated to manif's `rplus` Jacobian out-parameters and `SO3Tangent.hat()` instead of hand-rolled formulas.
- [pointcloud_pose_tracking.py](use_manif/pointcloud_pose_tracking.py): same five solvers (dead-reckoning, EKF, invariant EKF, batch Gauss-Newton, UKF) as the `use_numpy` version, with `motion_model`/`observation_model` built on `manifpy`'s `rplus`/`rminus`/`act` Jacobian out-parameters instead of `lie_utils.py` (the UKF itself uses only the bare `rplus`/`rminus` group operations, no Jacobian out-parameters).
- [pose_graph.py](use_manif/pose_graph.py)
- [pose_graph_incremental.py](use_manif/pose_graph_incremental.py): same incremental-vs-batch comparison as the `use_numpy` version, with edge Jacobians obtained by chaining `manifpy`'s own `compose`/`rminus` Jacobian out-parameters instead of `lie_utils.py`.
- [bundle_adjustment.py](use_manif/bundle_adjustment.py): same three solvers as the `use_numpy` version, with camera-pose Jacobians obtained by chaining `manifpy`'s own `inverse`/`act` Jacobian out-parameters instead of a hand-rolled closed form; landmarks stay plain numpy R^3 vectors (manif has no notion of those), same as `pointcloud_pose_tracking.py`.
- [bundle_adjustment_advanced.py](use_manif/bundle_adjustment_advanced.py): same Local + Global BA comparison as the `use_numpy` version, with camera-pose Jacobians obtained by chaining `manifpy`'s own `inverse`/ `act` Jacobian out-parameters, matching `bundle_adjustment.py`'s own manif version.

### Root

- [main.py](main.py): unused `uv init` placeholder entry point.
- [pyproject.toml](pyproject.toml): uv project file. `manifpy` is sourced from the local sibling checkout `../manif` (see Installation below).

## D. Installation

- Install [uv](https://docs.astral.sh/uv/).
- `manifpy` is not on PyPI here - this project's `pyproject.toml` points `uv` at a local sibling checkout (`../manif`), which must already have its Python bindings built (`pip3 install --user ../manif`, requires a working Eigen3 + CMake toolchain) before `uv sync` can resolve it. The [use_numpy/](use_numpy/) scripts don't need this - only the [use_manif/](use_manif/) scripts do.
- Then sync dependencies: `uv sync`
- Run any script with `uv run` from this repository's root directory, e.g.:
  `uv run python use_numpy/imu_integration_comparison.py`

### Running the tests

`tests/` has a pytest suite covering `utils.py`, `use_numpy/lie_utils.py`'s Lie-group math (round-trip and Jacobian-identity checks), and the testable functions in every `use_numpy/`/`use_manif/` simulation script (deterministic, fixed-seed regression checks - e.g. optimized error beats an uncorrected baseline, two solvers agree on the same input, a filter matches its own documented exact-agreement claim). `use_manif/` tests skip cleanly (not fail) in an environment without `manifpy` built.

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

Simulates a robot with a 100 Hz IMU (noisy body twist) and a 1 Hz high-accuracy global position fix (e.g. GPS). Each second: propagate the pose estimate through 100 noisy IMU micro-steps on $SE(3)$, then run a Gauss-Newton correction against the position fix (an unbalanced information matrix trusts position far more than orientation) until the correction step norm drops below `--gn-tol` or `--gn-max-iters` is hit. Prints pre/post correction error each second (no plot).

#### Scripts

- [use_numpy/robot_imu_simulation.py](use_numpy/robot_imu_simulation.py) 

- [use_manif/robot_imu_simulation.py](use_manif/robot_imu_simulation.py)

#### Usage

```
uv run python use_numpy/robot_imu_simulation.py --dt-imu 0.01 --total-seconds 3 --snapshots-per-second 4 --gn-tol 1e-6 --gn-max-iters 10 --max-linear-vel 1.0 --max-angular-vel 0.5
```

- `--dt-imu`: IMU update interval in seconds (default `0.01`, i.e. 100 Hz)
- `--total-seconds`: total simulation duration in seconds (default `3`)
- `--snapshots-per-second`: number of intermediate IMU steps logged per second (default `4`)
- `--gn-tol`: Gauss-Newton convergence tolerance (default `1e-6`)
- `--gn-max-iters`: maximum Gauss-Newton iterations (default `10`)
- `--max-linear-vel`: max linear velocity magnitude for the random true twist, m/s (default `1.0`)
- `--max-angular-vel`: max angular velocity magnitude for the random true twist, rad/s (default `0.5`)

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
- `--gyro-bias`, `--accel-bias`: initial estimated sensor biases (default `0.01 -0.01 0.02` rad/s, `0.05 0.00 -0.05` m/s²)
- `--linear-vel`, `--angular-vel`: true commanded body-frame twist (default `1.0 0.1 0.0` m/s, `0.0 0.0 0.5` rad/s)
- `--optimized-gyro-bias`, `--optimized-accel-bias`: post-optimization corrected biases to apply via the Taylor correction

### 4. Point-cloud pose tracking: EKF vs. invariant EKF vs. batch Gauss-Newton vs. UKF

#### Purpose

Tracks a rigid object's $SE(3)$ pose from a combination of a motion-model prior (noisy control-input twist) and noisy point-cloud measurements of the object's known body-frame geometry (`z_i = T.act(p_i) + noise`). Two shared, documented functions - `motion_model` and `observation_model`, both with analytical Jacobians (hand-rolled $SE(3)$ $Exp$ / $Log$ /adjoint math via `lie_utils.py` in the `use_numpy` version; `manifpy`'s `rplus`/`rminus`/`act` out-parameters in the `use_manif` version) - feed five solvers: 
 1. A **prior-only dead-reckoning baseline**
 2. A recursive **EKF** (constant-size state, online)
 3. A recursive **invariant EKF** (same predict step, but the update expresses the residual in the estimate's body frame, making the measurement Jacobian state-independent instead of re-linearized around the current rotation every step; for this problem's isotropic point-noise model this is provably equivalent to the plain EKF's corrections at every step, so the practical win here is a fixed, precomputed Jacobian rather than different accuracy)
 4. A **batch Gauss-Newton** smoother that jointly optimizes the whole trajectory at once against prior/motion/measurement factors
 5. A recursive **UKF** (sigma points sampled around the current estimate, retracted onto $SE(3)$ and pushed through the exact, non-linearized `motion_model`/`observation_model` - no Jacobians at all, unlike the other three). 

Prints final/RMS rotation+position error per method, plus each method's empirical average per-step wall-clock time and peak memory (`measure_performance`, via `time.perf_counter` + `tracemalloc`), and plots rotation error, position error, and the x-y trajectory of all six (ground truth included).

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

A robot drives a closed 4-node square loop, accumulating drift from noisy relative-pose ("odometry") edges between consecutive nodes, then detects it has returned to the start and adds one loop-closure edge back to node 0. All node poses are jointly refined by Levenberg-Marquardt against every edge's residual ${e_{ij} = \log({Z_{ij}}^{-1} * {X_i}^{-1} * X_j)}$, using analytical `compose`/`rminus` Jacobians chained together (hand-rolled $SE(3)$ $Exp$ / $Log$ / adjoint math via `lie_utils.py` in the `use_numpy` version; `manifpy`'s out-parameters in the `use_manif` version) - the same motion-factor Jacobian-chaining pattern as `run_batch_gn` in `pointcloud_pose_tracking.py`, generalized from a twist-based motion model to a directly-measured relative pose. Prints per-iteration chi-squared error plus final/RMS rotation+position error (uncorrected odometry vs. optimized), and plots the XY trajectory against ground truth.

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

- **Landmarks-only refinement**: poses held fixed at their noisy initial guess, only landmarks refined (the "cameras are correct" strawman) - decouples into independent 3x3 Gauss-Newton solves per landmark (classic triangulation from known poses).
- **Poses-only refinement**: landmarks held fixed, only poses refined (the "points are correct" strawman) - decouples into independent 6x6 Gauss-Newton solves per camera (classic PnP-style resection).
- **Full joint bundle adjustment**: both refined together in one coupled dense Gauss-Newton solve over every pose and landmark - the actual thing bundle adjustment is. This reintroduces the classic monocular BA gauge freedom (6-DoF rigid + 1-DoF scale ambiguity), fixed with a prior factor on the first two camera poses (mean = their own noisy initial guess), the same prior-factor pattern `pointcloud_pose_tracking.py`'s `run_batch_gn` already uses for its own gauge freedom. Since this only recovers the scene up to an unknown similarity transform, the result is aligned to ground truth via Umeyama's least-squares similarity fit (standard practice for evaluating monocular BA/SfM output) before computing absolute pose/ landmark error - reprojection error is unaffected by this alignment.

Camera-pose Jacobians come from a hand-derived closed form (`d(p_c)/d(right-perturbation of T) = [-I | skew(p_c)]`, finite-difference verified) in the `use_numpy` version, and from chaining `manifpy`'s own `inverse`/`act` Jacobian out-parameters in the `use_manif` version - no manifold formula hand-rolled there. Prints RMS pose rotation/position error, landmark error, and reprojection error for all four rows (noisy init, landmarks-only, poses-only, joint BA), and plots a top-down scene view (ground truth vs. noisy init vs. joint BA) alongside a grouped bar chart of the four RMS metrics.

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

- **Reveal + triangulate**: each keyframe's observations are revealed one at a time; a landmark is triangulated (closed-form ray intersection, then a few Gauss-Newton iterations) the moment it crosses `--min-observations` observers, and committed only if it passes a cheirality (positive-depth) check - a weakly-constrained point can otherwise converge to a reflection behind a camera that still fits that view's pixel almost exactly, since `pixel = f*x/z` is invariant under negating a whole camera-frame point.
- **Bounded local BA window** (`build_active_window` + `run_local_ba_step`): the new keyframe plus its strongest covisible neighbors (up to `--max-window-keyframes`) are optimized together; every other keyframe that also observes one of the window's landmarks is held fixed as a rigid anchor. Keyframes 0/1 are hard-fixed forever as the gauge anchor - no prior factor needed, since a hard anchor already pins the gauge.
- **Periodic Global BA** (`run_global_ba`): every `--global-ba-interval` keyframes (plus once at the end), every keyframe and landmark seen so far is jointly re-solved in one system, for direct comparison against the bounded local window.

Both solvers share one Levenberg-Marquardt core (`run_windowed_gn_lm`): a step is only accepted if it actually reduces total reprojection error, otherwise the damping grows and the step is retried - plain fixed-damping Gauss-Newton (as `bundle_adjustment.py` uses for its densely-observed, prior-anchored toy scene) was found to diverge explosively on this script's weakly-constrained early windows. The incremental loop runs twice off the same scene - once with Global BA disabled, once with it enabled - and prints/plots wall-clock solve time (should stay flat for the local window, grow for Global BA as the map grows) and a trailing-window RMS trajectory error (drift accumulating vs. periodically corrected). One finding worth being upfront about: on the default open path, Global BA has no *new* geometric constraint to exploit beyond what the overlapping local windows already used - it still helps on most noise draws, just not as dramatically as an actual loop closure would (see `docs/optimization/pose_graph_optimization.md`). The covisibility/window/Global-BA machinery itself is already loop-closure-agnostic, though - see "Loop closure example" below for a path that actually has one.

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

This is pure index/graph bookkeeping - no pose math, `SE(3)`, or Lie algebra anywhere - so unlike every other script in this repo there is no `use_manif/` counterpart; the result would be structurally identical either way, since the tree only depends on which node indices a factor connects, never the noisy relative-pose values themselves. It implements the Bayes tree's *construction* and *affected-region query* only, not the numeric fluid-relinearization solve (`pose_graph_incremental.py`'s iSAM v1 square-root-SAM update handles that, without ever building a Bayes tree) or dynamic reordering.

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

`linear_pnp_dlt` solves the classic Direct Linear Transform (DLT) camera-resectioning problem specialized to known intrinsics - each calibrated ray is parallel to its camera-frame point, giving a linear homogeneous constraint on the flattened world-to-camera `[R|t]`, solved via the smallest right-singular vector and then projected onto the nearest proper rotation (SVD orthogonalization), with scale and sign fixed from that same decomposition and a positive-depth check (this repo's PnP analogue of `passes_cheirality`). `refine_pose_gn` then runs ordinary Gauss-Newton on the true reprojection residual, updating the pose via a right-multiplicative SE(3) correction, mirroring `refine_landmark_gn`'s loop.

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
