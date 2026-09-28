# Glossary

Short definitions of the terms that come up across several docs in this folder, plus a few that individual docs used to define in their own appendices, grouped by topic. Each entry gives a one- or two-sentence meaning, the symbol the docs use for it (if any), and a link to the doc section that explains it properly.

Notation follows the docs: $^\top$ is transpose, $\Omega$ is an information matrix, a pose $T$ is updated as $T\cdot\mathrm{Exp}(\Delta x)$, and a pose's tangent vector is $\xi = [v, \omega]$ (translation first).

---

## 1. Geometry and Lie groups

- **Pose** ($T$): position and orientation together. In 3D it is a $4\times4$ matrix holding a rotation $R$ and a translation $t$, i.e., an element of $SE(3)$. See [pose_graph_optimization.md §1](optimization/pose_graph_optimization.md#1-first-what-is-a-pose).
- **$SO(3)$ / $SE(3)$**: the group of 3D rotations and the group of 3D rigid motions (rotation + translation). They have 3 and 6 degrees of freedom. See [lie_algebra.md §5](foundations/lie_algebra.md#5-lie-group-the-space-of-valid-transformations) and [§11](foundations/lie_algebra.md#11-se3-this-is-where-robotics-gets-really-interesting).
- **Degrees of freedom (DoF)**: the number of independent numbers needed to describe something - 3 for a 3D rotation, 6 for a 3D pose. See [lie_algebra.md §11](foundations/lie_algebra.md#11-se3-this-is-where-robotics-gets-really-interesting).
- **Minimal vs. ambient (redundant) parameterization**: a minimal parameterization uses exactly as many numbers as there are degrees of freedom, e.g. the 6-vector $[v, \omega]$ for a pose. An ambient one uses more, e.g. the 9 entries of $R$ plus $t$ (12 numbers for 6 DoF). Going ambient can make a model linear, but it needs an external step to stay on the manifold (such as SVD re-projection of $R$), and its covariance isn't directly comparable to a minimal one. This is a separate question from the covariance's *shape* (§2). See [pointcloud_pose_tracking_empirical_note.md §4.1](filtering/pointcloud_pose_tracking_empirical_note.md#41-what-vanilla-buys-and-costs) and [§4.2](filtering/pointcloud_pose_tracking_empirical_note.md#42-three-genuine-sources-of-divergence).
- **Manifold**: a curved space that looks flat when you zoom in far enough. Rotations and poses live on one, which is why they can't be updated by plain addition. See [lie_algebra.md §4](foundations/lie_algebra.md#4-rotations-are-different).
- **Lie group**: a smooth set of transformations that can be composed and inverted, such as $SO(3)$ and $SE(3)$. See [lie_algebra.md §5](foundations/lie_algebra.md#5-lie-group-the-space-of-valid-transformations).
- **Lie algebra/tangent space**: the flat vector space of small motions around a point of a Lie group. This is where optimizers and filters compute their steps and covariances. See [lie_algebra.md §6](foundations/lie_algebra.md#6-lie-algebra-zoom-in-locally).
- **Hat operator** ($(\cdot)^\wedge$): turns a 3-vector into its skew-symmetric matrix, so that $a^\wedge b = a \times b$. See [lie_algebra.md §7](foundations/lie_algebra.md#7-the-weird-looking-skew-symmetric-matrix).
- **Exp / Log maps**: $\mathrm{Exp}$ turns a tangent vector (a small motion) into an actual rotation or pose, and $\mathrm{Log}$ goes back. See [lie_algebra.md §8](foundations/lie_algebra.md#8-the-exponential-map) and [§9](foundations/lie_algebra.md#9-the-logarithm-map).
- **Retraction**: how an estimate on a manifold takes a step: $T \leftarrow T\cdot\mathrm{Exp}(\Delta x)$ instead of $T + \Delta x$. See [gauss_newton.md §8](optimization/gauss_newton.md#8-why-is-this-everywhere-in-slam) and [pose_graph_optimization.md §15.6](optimization/pose_graph_optimization.md#156-retraction--state-update).
- **Adjoint** ($`\mathrm{Ad}(T)`$): the $6\times6$ matrix that re-expresses a tangent vector from one frame's coordinates in another's. See [pose_graph_optimization.md §15.4](optimization/pose_graph_optimization.md#154-manifold-optimization-and-linearization).
- **Quaternion**: four numbers $(w, x, y, z)$ encoding a 3D rotation as an axis plus a half-angle. $q$ and $-q$ represent the same rotation. See [quaternion.md §2](foundations/quaternion.md#2-the-intuitive-idea-a-quaternion-is-an-axis--angle-in-disguise) and [§10](foundations/quaternion.md#10-one-subtle-but-very-important-fact).
- **Invariant / equivariant**: an error (or a system) is *invariant* to a transformation if applying that transformation, e.g., to both the true and the estimated state, leaves it unchanged. Left- and right-invariant errors are the basis of the Invariant EKF. *Equivariant* is the related, weaker property: the quantity does change, but in a matching, predictable way. See [left_right_invariant.md §3](filtering/left_right_invariant.md#3-why-invariant---and-why-leftright).
- **Isotropic / anisotropic**: isotropic noise or uncertainty is the same in every direction: a covariance $\sigma^2 I$, a sphere that looks identical after any rotation ($R\,(\sigma^2 I)\, R^\top = \sigma^2 I$). Anisotropic noise is direction-dependent: an ellipsoid. See [pointcloud_pose_tracking_empirical_note.md §2.1](filtering/pointcloud_pose_tracking_empirical_note.md#21-intuitive-explanation) and [§5](filtering/pointcloud_pose_tracking_empirical_note.md#5-when-would-each-actually-diverge-more).

---

## 2. Uncertainty and probability

- **Gaussian noise**: measurement noise modelled as a zero-mean normal distribution. Under this assumption, minimizing weighted squared errors gives the most likely estimate. See [nonlinear_least_square.md §12](optimization/nonlinear_least_square.md#12-add-measurement-uncertainty).
- **Covariance** ($P$ or $\Sigma$): a matrix describing how uncertain an estimate or a measurement is, and how its errors are correlated. See [kf_ekf_iekf.md §1](filtering/kf_ekf_iekf.md#1-standard-kalman-filter-everything-is-nicely-linear).
- **Scalar, diagonal and full covariance**: the three shapes a covariance can take, in increasing generality: *scalar* $\sigma^2 I$ (isotropic), *diagonal* (a different variance per axis but no correlation between axes), and *full* (with correlations). Only the scalar case is unchanged by every rotation. See [pointcloud_pose_tracking_empirical_note.md §5](filtering/pointcloud_pose_tracking_empirical_note.md#5-when-would-each-actually-diverge-more).
- **Information matrix** ($\Omega = \Sigma^{-1}$): the inverse covariance. It weights each error so that precise measurements count more. [nonlinear_least_square.md](optimization/nonlinear_least_square.md) writes the same quantity as $W$. See [factor_graph.md §4](optimization/factor_graph.md#4-optimization-means-minimizing-all-those-errors).
- **Mahalanobis distance**: $d^2 = r^\top \Omega\, r$, an error measured in standard deviations while accounting for the noise's shape. For isotropic noise $\Omega = I/\sigma^2$, so it reduces to plain squared error divided by $\sigma^2$. See [pose_graph_optimization.md §15.3](optimization/pose_graph_optimization.md#153-objective-function) and [factor_graph.md §4](optimization/factor_graph.md#4-optimization-means-minimizing-all-those-errors).
- **Prior (unary factor)**: a factor on a single variable, e.g. one that pins the first pose (to remove gauge freedom) or one that carries information left behind by marginalization. See [filtering_smoothing.md §7](filtering_smoothing.md#7-this-is-where-factor-graphs-fit).
- **Consistency / NEES**: an estimator is consistent when its reported covariance matches the errors it actually makes. NEES, $(x_{\text{true}} - x_{\text{est}})^\top P^{-1}(x_{\text{true}} - x_{\text{est}})$, tests this: for an $n$-DoF state it should average $n$ over many trials; larger means overconfident, smaller means underconfident. See [hybrid_saltation_ekf.md §6](filtering/hybrid_saltation_ekf.md#6-the-empirical-finding-a-modest-real-gap-not-a-dramatic-one).
- **Monte Carlo trials**: repeating a randomized simulation many times with different noise to measure typical behavior instead of one lucky or unlucky run. See [hybrid_saltation_ekf.md §1.1](filtering/hybrid_saltation_ekf.md#11-the-filter-math-concretely).
- **Observability**: a direction of the state is unobservable when no measurement can determine it. A solver that wrongly gains information there becomes overconfident, which is the problem FEJ (First-Estimates Jacobian) addresses. See [marginalization.md §6](optimization/marginalization.md#6-the-consistency-gotcha-why-fej-exists).
- **Bias** ($b_g$, $b_a$): the slowly varying offset in an IMU's gyroscope and accelerometer readings, usually estimated as part of the state. See [vi_initialization.md §1](frontend/vi_initialization.md#1-the-bootstrapping-problem) and [imu_preintegration.md §4](optimization/imu_preintegration.md#4-the-bias-sensitivity-jacobians).

---

## 3. Least-squares optimization

- **Residual / error** ($e$ or $r$; both letters appear across docs): the difference between what the model predicts and what was measured. See [nonlinear_least_square.md §4](optimization/nonlinear_least_square.md#4-the-general-form).
- **(Nonlinear) least squares**: choosing the unknowns that minimize the sum of weighted squared residuals, $\sum_i e_i^\top \Omega_i e_i$. It is *nonlinear* when the residuals depend nonlinearly on the unknowns, as they do in SLAM. See [nonlinear_least_square.md §3](optimization/nonlinear_least_square.md#3-then-what-makes-it-nonlinear).
- **Jacobian** ($J$): the matrix of first derivatives of the residuals with respect to the unknowns - how much each error changes when each unknown is nudged. See [jacobian.md §2](foundations/jacobian.md#2-now-imagine-multiple-inputs-and-outputs).
- **Linearization**: replacing a nonlinear function by its first-order approximation around the current estimate (the *linearization point*), $e(x+\Delta x) \approx e(x) + J\Delta x$. See [nonlinear_least_square.md §7](optimization/nonlinear_least_square.md#7-the-key-trick-make-the-nonlinear-problem-locally-linear).
- **Initial guess**: the starting estimate an iterative solver refines, often from dead reckoning. A poor one can make Gauss-Newton diverge. See [vi_initialization.md §1](frontend/vi_initialization.md#1-the-bootstrapping-problem).
- **Normal equations / Hessian**: each Gauss-Newton step solves $H\Delta x = -b$ with $H = J^\top \Omega J$ and $b = J^\top \Omega e$. This $H$ approximates the Hessian (the second derivatives) of the cost. See [gauss_newton.md §4](optimization/gauss_newton.md#4-the-optimization-problem) and [§7](optimization/gauss_newton.md#7-why-is-it-called-gauss-newton).
- **Gradient descent**: the step $\Delta x = -\alpha J^\top e$, straight downhill with step size $\alpha$. Safe with a small step, but slow on badly scaled problems. See [levenberg_marquardt.md §3](optimization/levenberg_marquardt.md#3-gradient-descent-has-the-opposite-problem).
- **Gauss-Newton**: linearize, solve the normal equations, update, repeat. See [gauss_newton.md §1](optimization/gauss_newton.md#1-the-basic-idea).
- **Levenberg-Marquardt (LM) / damping** ($\lambda$): Gauss-Newton with a damping term added to $H$ ($\lambda I$, or $\lambda\,\mathrm{diag}(H)$ in Marquardt's form). A large $\lambda$ gives cautious, gradient-descent-like steps; a small one gives Gauss-Newton steps. See [levenberg_marquardt.md §4](optimization/levenberg_marquardt.md#4-lms-brilliant-idea) and [§5](optimization/levenberg_marquardt.md#5-what-does-lambda-actually-do).
- **Gauge freedom**: a direction in which the whole solution can move without changing any residual, e.g. shifting and rotating the entire trajectory and map (plus scale, for monocular BA). It is removed by anchoring, e.g. a prior on the first pose, or handled at evaluation time by alignment. See [gauss_newton.md §4](optimization/gauss_newton.md#4-the-optimization-problem) and [bundle_adjustment.md §14](optimization/bundle_adjustment.md#14-evaluating-the-result-gauge-freedom-and-umeyama-alignment).
- **Outlier / robust kernel**: an outlier is a grossly wrong measurement, such as a false loop closure. Robust kernels (Huber, Cauchy, DCS, ...) down-weight large residuals so outliers can't dominate the solution. See [pose_graph_optimization.md §16](optimization/pose_graph_optimization.md#16-robust-loss-functions-used-to-handle-false-loop-closures).
- **Relinearization**: recomputing the Jacobians at an updated estimate. iSAM2 does it selectively, only for variables whose estimate has moved enough. See [isam2_optimization.md §10](optimization/isam2_optimization.md#10-selective-relinearization).

---

## 4. Graphs, sparsity and solvers

- **Factor graph**: a graph of *variables* (the unknowns) and *factors* (measurements or constraints linking them). The total cost is the sum of all factor errors. See [factor_graph.md §1](optimization/factor_graph.md#1-the-core-idea) and [§6](optimization/factor_graph.md#6-the-really-important-distinction-variable-vs-factor).
- **Factor**: one error term connecting the variables it involves, e.g., odometry between two poses, or a camera observing a landmark. See [factor_graph.md §3](optimization/factor_graph.md#3-a-factor-is-basically-an-error-function).
- **Pose graph**: a factor graph whose variables are only poses, linked by relative-pose factors (odometry and loop closures). See [factor_graph.md §7](optimization/factor_graph.md#7-factor-graph-vs-pose-graph).
- **Sparsity**: most entries of $J$ and $H$ are zero, because each measurement involves only a few variables. Efficient solvers depend on it. See [sparse_cholesky_factorization.md §1](optimization/sparse_cholesky_factorization.md#1-why-does-sparsity-matter).
- **Cholesky factorization**: writing $H = LL^\top$ with $L$ lower-triangular, so the solve becomes two cheap triangular solves. See [sparse_cholesky_factorization.md §2](optimization/sparse_cholesky_factorization.md#2-the-basic-idea).
- **Variable elimination**: solving for one variable at a time in terms of its neighbors. It is the graph view of factorization, and the step behind Cholesky, elimination trees, Bayes trees and marginalization. See [bayes_tree.md §3](optimization/bayes_tree.md#3-elimination-is-the-key-idea).
- **Fill-in**: new nonzeros that elimination creates between variables that weren't directly connected. See [sparse_cholesky_factorization.md §3](optimization/sparse_cholesky_factorization.md#3-the-surprising-part-fill-in).
- **Variable ordering**: the order in which variables are eliminated. It decides how much fill-in appears; heuristics such as AMD and COLAMD choose it. See [sparse_cholesky_factorization.md §5](optimization/sparse_cholesky_factorization.md#5-ordering-becomes-extremely-important).
- **Elimination tree**: records which variables must be eliminated before which. See [elimination_tree.md §2](optimization/elimination_tree.md#2-why-call-it-a-tree).
- **Bayes tree**: a tree built from elimination whose nodes are *cliques* (groups of variables). It lets iSAM2 update only the part of the solution a new measurement affects. See [bayes_tree.md §13](optimization/bayes_tree.md#13-one-more-important-concept-cliques) and [isam2_optimization.md §7](optimization/isam2_optimization.md#7-bayes-tree---the-most-important-intuition).
- **Batch vs. incremental**: batch optimization re-solves the whole problem from scratch; incremental methods (iSAM, iSAM2) update the previous solution as new measurements arrive. See [isam_optimization.md §1](optimization/isam_optimization.md#1-the-problem-with-ordinary-batch-optimization) and [§2](optimization/isam_optimization.md#2-incremental-optimization-says).
- **Marginalization**: permanently removing a variable while keeping what it told you about its neighbors, as a (generally dense) prior on them. It is computed with the Schur complement, the same tool [bundle_adjustment.md §12](optimization/bundle_adjustment.md#12-block-sparsity-and-the-schur-complement) uses to eliminate landmarks. See [marginalization.md §2](optimization/marginalization.md#2-marginalization-keep-the-information-drop-the-variable).
- **Sliding window**: optimizing only the most recent states and marginalizing older ones, so the cost stays bounded. See [marginalization.md §7](optimization/marginalization.md#7-sliding-window-and-fixed-lag-smoothing-the-payoff).
- **Filtering vs. smoothing**: filtering estimates only the current state and folds the past away; smoothing keeps past states and can revise them. See [filtering_smoothing.md §1](filtering_smoothing.md#1-the-core-idea).

---

## 5. SLAM system

- **SLAM**: Simultaneous Localization and Mapping - estimating a robot's trajectory and a map of its environment at the same time. See [frontend_backend.md](frontend_backend.md).
- **Front-end/back-end**: the front-end turns raw sensor data into constraints (feature tracking, data association, loop detection, keyframe selection); the back-end optimizes the trajectory and map from those constraints. See [frontend_backend.md §1](frontend_backend.md#1-slam-front-end---extracting-constraints) and [§2](frontend_backend.md#2-slam-back-end---solving-the-global-problem).
- **Odometry**: an estimate of the robot's incremental motion between two nearby moments from onboard sensors (wheel, visual, or inertial). See [factor_graph.md §2](optimization/factor_graph.md#2-why-do-we-need-it).
- **Dead reckoning**: chaining odometry steps into a trajectory with no external correction, so errors accumulate. See [factor_graph.md §2](optimization/factor_graph.md#2-why-do-we-need-it).
- **Drift**: the build-up of small errors over a long trajectory. See [pose_graph_optimization.md §3](optimization/pose_graph_optimization.md#3-why-do-we-need-optimization).
- **Loop closure**: recognizing a previously visited place, which adds a constraint between two distant poses and corrects drift. See [pose_graph_optimization.md §4](optimization/pose_graph_optimization.md#4-the-really-important-event-loop-closure).
- **Landmark**: a point in the environment observed from several poses and estimated together with them. See [bundle_adjustment.md §1](optimization/bundle_adjustment.md#1-start-with-a-simple-camera--3d-point).
- **Keyframe**: one of a selected subset of frames kept in the optimization, instead of every frame. See [bundle_adjustment.md §13](optimization/bundle_adjustment.md#13-local-vs-global-bundle-adjustment-real-systems).
- **Bundle adjustment (BA)**: jointly refining camera poses and 3D landmarks to minimize reprojection error. See [bundle_adjustment.md §5](optimization/bundle_adjustment.md#5-why-is-it-called-bundle-adjustment).
- **Reprojection error**: the pixel difference between where a landmark was observed in an image and where the current estimates project it. See [bundle_adjustment.md §2](optimization/bundle_adjustment.md#2-but-our-estimates-are-imperfect).
- **Triangulation/PnP**: triangulation recovers a 3D point from known camera poses; PnP (Perspective-n-Point) recovers a camera pose from known 3D points. They are the same reprojection problem solved in opposite directions. See [triangulation_pnp.md §1](frontend/triangulation_pnp.md#1-the-two-problems-stated-symmetrically).
- **IMU preintegration**: summarizing all IMU samples between two keyframes into one relative-motion measurement ($\Delta R$, $\Delta v$, $\Delta p$), with bias Jacobians so a bias change can be corrected without re-integrating. See [imu_preintegration.md §3](optimization/imu_preintegration.md#3-what-gets-compressed) and [§5](optimization/imu_preintegration.md#5-the-payoff-correcting-for-a-bias-change-without-re-integrating).
- **Umeyama alignment/RMS trajectory error**: before comparing an estimated trajectory with ground truth, the two are aligned (best rotation, translation and, if needed, scale) to remove gauge freedom; the remaining root-mean-square position error is then reported. See [umeyama_alignment.md §1](foundations/umeyama_alignment.md#1-why-an-alignment-step-is-needed-at-all).

---

## 6. Kalman-filter family

- **Kalman filter (KF)**: alternates a *predict* step (motion model) and an *update* step (measurement), carrying a mean and a covariance. It is optimal for linear models with Gaussian noise. See [kf_ekf_iekf.md §1](filtering/kf_ekf_iekf.md#1-standard-kalman-filter-everything-is-nicely-linear).
- **EKF (Extended Kalman Filter)**: the Kalman filter for nonlinear models, linearizing them with Jacobians at the current estimate. See [kf_ekf_iekf.md §2](filtering/kf_ekf_iekf.md#2-extended-kalman-filter-the-world-is-nonlinear-so-ill-approximate-it-locally).
- **IEKF (Invariant EKF)**: an EKF that uses an invariant error defined on the Lie group. In this repo "IEKF" always means the Invariant EKF, not the Iterated EKF of [kf_ekf_iekf.md §9](filtering/kf_ekf_iekf.md#9-other-prominent-variants-worth-keeping-in-mind). See [kf_ekf_iekf.md §3](filtering/kf_ekf_iekf.md#3-invariant-ekf-lets-respect-the-geometry-of-the-problem).
- **ESKF (Error-State Kalman Filter)**: propagates a nominal state with the full nonlinear model, while the filter itself estimates only a small error state. See [extra_kf_variants.md §2](filtering/extra_kf_variants.md#2-error-state-kalman-filter-eskf).
- **UKF (Unscented Kalman Filter)**: pushes a set of *sigma points* through the nonlinear functions instead of linearizing them with Jacobians. See [extra_kf_variants.md §1](filtering/extra_kf_variants.md#1-unscented-kalman-filter-ukf).

---

## 7. Hybrid (discontinuous) systems

- **Hybrid system/guard condition/reset map**: a hybrid system alternates smooth motion with instantaneous jumps, like a bouncing ball. The *guard* $g(x) = 0$ is the switching surface that triggers a jump, and the *reset map* $R$ (not a rotation here) is the possibly discontinuous map applied when the trajectory reaches it. See [hybrid_saltation_ekf.md §1](filtering/hybrid_saltation_ekf.md#1-what-a-hybrid-dynamical-system-is-here).
- **Zeno behavior**: a hybrid system making infinitely many jumps in a finite time, e.g. a lossy ball bouncing ever faster as it settles. It is a standard pathology to guard against in hybrid-system simulation, not something specific to saltation matrices. See [hybrid_saltation_ekf.md §7](filtering/hybrid_saltation_ekf.md#7-two-things-worth-knowing-before-reusing-this-pattern).

---

## 8. References

1. Solà, J., Deray, J., & Atchuthan, D. (2018). *A micro Lie theory for state estimation in robotics*. arXiv:1812.01537. https://doi.org/10.48550/arXiv.1812.01537 - the $SO(3)$/$`SE(3)`$, Exp/Log and hat conventions behind §1.
2. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press. - the probabilistic and filtering vocabulary behind §2 and §6.
3. Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-40065-5 - the least-squares, Gauss-Newton, and Levenberg-Marquardt terms in §3.
4. Dellaert, F., & Kaess, M. (2017). *Factor Graphs for Robot Perception*. Foundations and Trends in Robotics, 6(1–2), 1–139. https://doi.org/10.1561/2300000043 - the factor-graph, elimination, and incremental-smoothing terms in §4 and §5.
