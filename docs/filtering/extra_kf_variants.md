# An overview of other prominent Kalman Filter variants

For **robotics, SLAM, visual-inertial estimation, and embedded systems**, several Kalman filter variants are worth knowing. We focus on **why each one exists**.

This builds directly on:
- The **standard KF, EKF, and IEKF** foundations from [kf_ekf_iekf.md](kf_ekf_iekf.md) - this doc surveys the wider family those three sit inside.
- **Lie groups and $SO(3)$/$`SE(3)`$** from [lie_algebra.md §5, §11](../foundations/lie_algebra.md#5-lie-group-the-space-of-valid-transformations) - the structure §2's ESKF and §4's IEKF sections both lean on.
- **Robust loss functions (Huber, etc.)** from [pose_graph_optimization.md §16](../optimization/pose_graph_optimization.md#16-robust-loss-functions-used-to-handle-false-loop-closures) - referenced directly in §9's Robust KF discussion.

A good mental map is:

```text
                     Bayes filter (recursive estimation)
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
   Kalman family                           Beyond Kalman
   (Gaussian belief)                       (any belief)
          │                                       │
   ┌──────┴──────┐                          Particle Filter
   │             │
 Linear      Nonlinear
   │             │
Standard KF  ┌───┴───────┐
             │           │
            EKF     UKF (sigma points)
             │
           ESKF
             │
           IEKF
```

The remaining variants deal with **noise, time, robustness, and computational constraints**. This is one corner of the repo-wide map in [slam_mental_map.md](../slam_mental_map.md), which places every doc and script on one picture.

---

## 1. Unscented Kalman Filter (UKF)

The EKF says:

> "I'll linearize the nonlinear function using a Jacobian."

The UKF says:

> "I don't want to calculate Jacobians. I'll pick a few carefully chosen points around my estimate and see how the nonlinear function transforms them."

These points are called **sigma points**. They aren't random samples: for an $n$-dimensional state there are $2n+1$ of them, placed deterministically at the mean and at $\pm$ the columns of a matrix square root of $`(n+\lambda)P`$ (this repo uses the Cholesky factor), where $`P`$ is the state covariance ([glossary](../glossary.md#2-uncertainty-and-probability)) and $`\lambda`$ is the UKF scaling parameter, not the LM damping $\lambda$ of the [glossary](../glossary.md#3-least-squares-optimization).

Picture the uncertainty as an ellipse:

```text
              •
           .     .
        .           .
       •      x      •
        .           .
           .     .
              •
```

Instead of approximating the nonlinear function with a tangent line like the EKF, the UKF sends these representative points through the actual nonlinear function, then reconstructs the new mean and covariance:

```text
     uncertainty          nonlinear transformation

       •                         •
    .     .                   .     .
   •   x   •       --->       •       •
    .     .                   .  x    .
       •                         •
```

**Why care?**

- **Accuracy.** For a Gaussian input, the transformed mean is accurate to second order, while the EKF's tangent line is only first-order accurate. So the UKF captures the bias that curvature adds to the mean, which the EKF misses entirely ([linear_nonlinear.md §4](linear_nonlinear.md#4-how-do-you-know-whether-its-nonlinear-over-your-uncertainty-region) shows how big that bias can get, and §4.2 mentions a sigma-point comparison as a cheaper variant of its nonlinearity check).
- **Strong nonlinearity.** It helps when the nonlinearities are strong over the region the uncertainty covers.
- **No Jacobians.** It is derivative-free, which is attractive when Jacobians are hard to compute.

The cost: it evaluates the model $2n+1$ times per step instead of once, which adds up for high-dimensional states.

See `run_ukf` in [`use_numpy/pointcloud_pose_tracking.py`](../../use_numpy/pointcloud_pose_tracking.py)/[`use_manif/pointcloud_pose_tracking.py`](../../use_manif/pointcloud_pose_tracking.py) for a runnable manifold-UKF implementation of this idea.

---

## 2. Error-State Kalman Filter (ESKF)

One of the most useful variants for SLAM/VIO work.

An ESKF doesn't estimate the entire state directly. It separates a **nominal state** from a **small error state**. For example, the true rotation is the nominal one times a small correction ($`\delta\theta`$ is a small rotation vector, and $`^\wedge`$ turns a 3-vector into its skew-symmetric matrix, see [glossary](../glossary.md#1-geometry-and-lie-groups)):

$$R = \hat R \exp(\delta\theta^\wedge)$$

- Nominal state: $`\hat X = (\hat R,\hat p,\hat v,\hat b_g,\hat b_a)`$ (rotation, position, velocity, gyro bias, accelerometer bias; [glossary](../glossary.md#5-slam-system))
- Error state, on which the EKF mainly operates: $`\delta x = (\delta\theta,\delta p,\delta v,\delta b_g,\delta b_a)`$

Each cycle has three steps:

1. **Predict**: integrate the nominal state with the full nonlinear model (for example raw IMU samples), and propagate the error covariance $P$ with the linearized error dynamics.
2. **Update**: a measurement produces an estimate of $\delta x$ through a normal EKF update.
3. **Inject and reset**: fold $\delta x$ into the nominal state (for rotation, $`\hat R \leftarrow \hat R\exp(\delta\theta^\wedge)`$), then set $\delta x = 0$.

**Intuition:**
- Think of a big map (the nominal state) plus a small sticky note (the error state) with the current correction.
- Predict: redraw the map using the full model. Update: only the sticky note is corrected.
- Inject and reset: copy the note onto the map, then use a blank note.

Why it helps: some states, especially **rotation**, live on manifolds rather than ordinary Euclidean vector spaces. The error state stays small and near zero, so its linearization is accurate, and $\delta\theta$ is a minimal 3-vector with no unit-norm constraint and no singularities near zero error, unlike a quaternion or Euler angles stored directly in the state.

The same filter goes by several names in visual-inertial odometry and inertial navigation: ESKF, error-state EKF, or indirect EKF. The **Multiplicative EKF (MEKF)** is the attitude-only version, and **right/left-invariant error-state filters** are §4's IEKF.

The **MSCKF** (Mourikis & Roumeliotis 2007), a classic VIO filter, is an ESKF whose state also holds a sliding window of past camera poses, with landmarks kept out of the state entirely. [kf_ekf_iekf.md §9](kf_ekf_iekf.md#9-other-prominent-variants-worth-keeping-in-mind) explains how it works.

### ESKF vs IEKF

They aren't competing alternatives: the IEKF *is* an error-state filter. In fact the ESKF rotation error above, $`\delta\theta = \log(\hat R^{-1}R)`$, already *is* the left-invariant error ($`\eta = \hat X^{-1}X`$; the right-invariant one is $`X\hat X^{-1}`$). What the IEKF adds is:

- **Estimate-independent error dynamics.** For group-affine dynamics (and suitable invariant measurements), the error dynamics and their Jacobians don't depend on the current estimate. This can fail, for example with IMU biases in the state.
- **Left vs right choice.** We pick the invariant error that matches the measurement.

---

## 3. Unscented vs Extended

|                             | EKF                 | UKF                                                                 |
| --------------------------- | ------------------- | ------------------------------------------------------------------- |
| Nonlinear system            | ✓                   | ✓                                                                   |
| Jacobians                   | Required            | No explicit Jacobian                                                |
| Linearizes function         | ✓                   | No explicit Jacobian (statistical linearization through sigma points) |
| Sigma points                | No                  | ✓                                                                   |
| Computational cost          | Lower               | Higher                                                              |
| Easy for complicated models | Sometimes difficult | Often easier                                                        |
| Common in robotics          | Very common         | Less dominant than EKF/ESKF                                         |

For robotics we usually learn KF, EKF, ESKF and IEKF before spending much time on the UKF (see §12).

---

## 4. Invariant EKF (IEKF)

[kf_ekf_iekf.md](kf_ekf_iekf.md) covers this one in depth; here we place it in the broader family.

**Naming**: in this repo "IEKF" always means the *Invariant* EKF. The literature also uses IEKF for the *Iterated* EKF, which relinearizes the update at the new estimate and repeats. That is an unrelated idea, covered in [kf_ekf_iekf.md §9](kf_ekf_iekf.md#9-other-prominent-variants-worth-keeping-in-mind).

The key idea:

> **Exploit the symmetry and Lie-group structure of the system.**

We treat $`R \in SO(3)`$ and $`T \in SE(3)`$ as group elements rather than as ordinary vectors. There are two formulations, using a left-invariant or a right-invariant error. For group-affine systems, both give error dynamics that are independent of the current state, which ordinary EKF linearizations don't. See [left_right_invariant.md](left_right_invariant.md) for what distinguishes the two and when to reach for each.

Particularly relevant to IMU navigation, VIO, SLAM, robotics and pose estimation.

---

## 5. Square-Root Kalman Filter (SR-KF)

This one is less about the **estimation philosophy** and more about **numerical stability**.

The standard KF stores the covariance $P$. The square-root version stores a factor $L$ with $P = LL^\top$, often obtained through Cholesky decomposition.

Why? A covariance matrix must be symmetric and positive semi-definite. In floating point, the standard update $`P \leftarrow (I - KH)P`$, where $`K`$ is the **Kalman gain** and $`H`$ the measurement Jacobian ([glossary](../glossary.md#6-kalman-filter-family)), subtracts nearly equal numbers, and rounding error can leave $P$ slightly asymmetric or with a negative eigenvalue. After that, the filter can diverge.

Carrying $L$ avoids this in two ways:

- $LL^\top$ is positive semi-definite **by construction**, whatever rounding happens to $L$.
- $L$'s condition number is the square root of $P$'s, so the same arithmetic keeps roughly twice as many significant digits.

**Tiny example** (toy case: two independent states):
- $P$ has variances 1 and $10^{-8}$, so the spread of its entries is $10^8$. That is too much for float32 (about 7 digits).
- $L$ has 1 and $10^{-4}$, so the spread is only $10^4$.

A cheaper, common partial fix is the **Joseph form** of the update, $`P \leftarrow (I-KH)P(I-KH)^\top + KRK^\top`$ ($`R`$ is the measurement noise covariance), which keeps $P$ symmetric positive semi-definite for any gain.

The same idea appears on the optimization side: iSAM keeps and updates the square-root *information* matrix $R$ (with $R^\top R = H$; here $`R`$ is the square-root information factor and $`H`$ the information matrix, not the noise covariance $`R`$ and measurement Jacobian $`H`$ above) rather than $H$ itself ([isam_optimization.md](../optimization/isam_optimization.md)).

### Intuition

Standard KF:

> "I'll carry the uncertainty matrix directly."

Square-root KF:

> "I'll carry a factor of the uncertainty matrix because it's numerically safer."

This matters most in long-running filters, single-precision or embedded arithmetic, and systems where some states are known far more precisely than others (a badly conditioned $P$).

---

## 6. Ensemble Kalman Filter (EnKF)

Instead of storing the mean and covariance $\mu, P$ explicitly, we keep an **ensemble of possible states**, each one a possible realization of the system:

```text
     •
        •
  •        •
       •
 •              •
          •
```

We propagate the members through the nonlinear model, with no Jacobians, and read the mean and covariance off the ensemble whenever we need them.

The update is still a **Kalman update**. The gain comes from the ensemble's sample covariance, and each member is shifted linearly toward the measurement. So the EnKF keeps the Kalman family's Gaussian, linear-update assumption. That is the key difference from the particle filter (§7), which reweights and resamples instead.

What it gets rid of is the $n \times n$ matrix $P$. When the state has $n \sim 10^6$ to $10^8$ entries (a gridded atmosphere or ocean), $P$ can't even be stored, but an ensemble of tens to ~100 members gives a usable low-rank approximation of it.

Especially popular in weather prediction, geophysical systems, ocean modeling and other very high-dimensional systems. It's generally **less central to robotics/SLAM** than EKF/ESKF/IEKF.

---

## 7. Particle Filter (PF)

This one drops the Gaussian assumption that most other variants in this list rely on.

Instead of a mean and covariance, our belief is a swarm of weighted samples ("particles"), each one a full hypothesis for the state:

$$\text{belief} \approx \lbrace (x^{(1)}, w^{(1)}), (x^{(2)}, w^{(2)}), \dots, (x^{(N)}, w^{(N)}) \rbrace$$

Each particle is propagated through the (possibly highly nonlinear) motion model, reweighted by how well it explains the latest measurement, and periodically resampled so particles that poorly explain the data get replaced by copies of the better ones:

```text
     particles              after motion           after weighting
                                                     + resampling

   •  •   •                  •    •                    •  •
  •    •     •     --->     •  •    •      --->        •••  •
     •    •                    •   •                       •
```

This lets a PF represent **multimodal** beliefs - "the robot is either in room A or room B, I genuinely don't know which" - something a single Gaussian, which every KF-family filter assumes, simply cannot express.

**Why care?**

- No linearity or Gaussian-noise assumption at all - works for arbitrarily nonlinear, non-Gaussian problems.
- Naturally represents multimodal beliefs (ambiguous data association, the kidnapped-robot problem, global localization).
- Classic robotics use case: **Monte Carlo Localization (MCL)** - localizing a robot on a known map from range/bearing measurements.

**The catch**: accuracy scales with particle count, and in high-dimensional state spaces (like a full SLAM state vector) you need an impractically large number of particles to cover the space adequately. That's why plain particle filters are common for low-dimensional localization but rare for full SLAM state estimation, where EKF/UKF/factor-graph approaches dominate instead.

The notable exception is the **Rao-Blackwellized particle filter**. Given the robot's trajectory, the landmarks become independent of each other, so each particle carries only a trajectory hypothesis plus one small EKF per landmark (**FastSLAM**, Montemerlo et al. 2002), or an occupancy grid (**GMapping**, Grisetti et al. 2007). The particles then cover only the low-dimensional pose, which is what makes particle-filter SLAM practical.

---

## 8. Adaptive Kalman Filter

Normally we assume the process noise covariance $Q$ and measurement noise covariance $R$ are known and fixed. In real life they change: an IMU, for example, behaves differently when the robot vibrates heavily.

An adaptive KF estimates or adjusts $Q$ and/or $R$ online. One family does this from the filter's own innovation sequence, the approach of Mehra (1970, reference 7). It is useful when the environment or sensor quality changes over time.

**Intuition:**
- Before each measurement, the filter predicts how big its surprise (the innovation) should be: the **innovation covariance** $`S`$ ([glossary](../glossary.md#6-kalman-filter-family)).
- Then it compares with the surprise it actually gets.
- Surprises much bigger than $`S`$ say that $Q$ or $R$ is too small. Much smaller says too big.
- So the filter nudges $Q$/$R$ until predicted and actual surprise agree.

---

## 9. Robust Kalman Filter

The standard KF essentially assumes:

> "My noise is reasonably well-behaved, approximately Gaussian."

But what if the camera occasionally produces a **terrible outlier**?

```text
measurements:

       • • • •
      • • • •
     • • • •
                     X  ← outlier
```

A conventional KF may be pulled toward that outlier. Robust filtering reduces the influence of bad measurements. The two common approaches are:

- **Innovation gating**, widely used. Before an update, compute the normalized innovation squared $`\nu^\top S^{-1}\nu`$ (with $`\nu`$ the innovation, measurement minus prediction) (the NIS from [linear_nonlinear.md §4.4](linear_nonlinear.md#44-after-the-fact-consistency-tests)). If it exceeds a chi-square threshold (for example the 99% bound for the measurement's dimension), reject the measurement outright.
- **Down-weighting**: keep the measurement but inflate its $R$ in proportion to how surprising it is. This is the filter's version of the Huber-style robust losses in [pose_graph_optimization.md §16](../optimization/pose_graph_optimization.md#16-robust-loss-functions-used-to-handle-false-loop-closures).

**Tiny example of gating** (1-D measurement, $S = 1$):
- A 3σ surprise gives $`\nu^\top S^{-1}\nu = 9`$.
- The 99% chi-square bound for 1 degree of freedom is 6.63, and $9 > 6.63$, so the measurement is rejected.

Particularly relevant to visual SLAM, feature tracking, GNSS, LiDAR and multi-sensor fusion.

In robotics, robust losses such as the **Huber loss** are also commonly used inside optimization-based estimators rather than relying exclusively on a robust KF.

---

## 10. Kalman Smoother

Not exactly another KF variant, but worth knowing.

A normal Kalman filter estimates $x_k | z_1,\ldots,z_k$:

> "What is the state **now**, given everything I've seen so far?"

A smoother can use **future measurements** too, estimating $x_k | z_1,\ldots,z_N$:

```text
Filtering:

t0 → t1 → t2 → t3 → t4
                    ↑
                 estimate (newest step,
                 past data only)


Smoothing:

t0 → t1 → t2 → t3 → t4
     ↑
     estimate using
     information from
     both past AND future
```

The classic example is the **Rauch–Tung–Striebel (RTS) smoother**, very useful for **offline SLAM and trajectory estimation**. [filtering_smoothing.md](../filtering_smoothing.md) covers the filtering-vs-smoothing distinction in depth.

---

## 11. Multi-rate/asynchronous Kalman filtering

Suppose we have:

```text
IMU       200 Hz
Camera     30 Hz
LiDAR      10 Hz
GPS         1 Hz
```

We don't want to force everything to run at the same frequency. Instead, the filter propagates with the IMU:

```text
IMU → predict
IMU → predict
IMU → predict
...
```

and updates whenever another sensor arrives:

```text
Camera → update
LiDAR  → update
GPS    → update
```

This is a fundamental pattern behind real-time sensor fusion.

---

## 12. Which ones should you prioritize learning?

For robotics work spanning **computer vision, SLAM, embedded systems, and resource-constrained platforms**, we don't need to learn every Kalman variant equally. A reasonable prioritization:

### Tier 1 - Must understand

#### 1. Standard KF

Understand:

- prediction
- measurement update
- covariance
- Kalman gain
- $Q$, $R$

↓

#### 2. EKF

Understand:

- nonlinear dynamics
- Jacobians
- local linearization

↓

#### 3. Error-State EKF/ESKF

Understand:

- nominal state
- error state
- perturbations
- why rotations need special treatment

↓

#### 4. IEKF

Understand:

- Lie groups
- $SO(3)$
- $SE(3)$
- left/right invariant errors
- system symmetries

### Tier 2 - Very useful

#### 5. UKF

Good alternative to EKF and useful for understanding nonlinear uncertainty propagation.

↓

#### 6. RTS smoother

Very useful for offline trajectory estimation and SLAM.

↓

#### 7. Square-root KF

Important for numerical stability and large-scale estimation.

### Tier 3 - Know the idea

#### 8. Adaptive KF

When $Q/R$ aren't fixed.

↓

#### 9. Robust KF

When measurements contain outliers.

↓

#### 10. EnKF

Mostly important in high-dimensional scientific applications.

↓

#### 11. Particle Filter

Mostly useful for low-dimensional, multimodal problems like global/Monte Carlo localization; rarely used for full high-dimensional SLAM state estimation.

---

## 13. Filtering vs optimization for SLAM

Modern robotics estimators fall into two big families ([filtering_smoothing.md](../filtering_smoothing.md) compares them in depth):

### Filtering

We maintain a state estimate recursively (§10's filtering diagram):

```text
IMU → KF/EKF/ESKF/IEKF → current state
```

### Optimization/smoothing

We optimize over the whole trajectory at once (§10's smoothing picture):

```text
           ┌────── camera ───────┐
           │                     │
x0 ──IMU── x1 ──IMU── x2 ──IMU── x3
                                 │
                                GPS

            ↓ nonlinear optimization over all of x0…x3
```

IMU factors (preintegrated, [imu_preintegration.md](../optimization/imu_preintegration.md)) connect consecutive states, a camera factor connects the states that observed the same landmark, and GPS is a unary factor on a single state.

Examples include:

- Bundle Adjustment
- Factor Graphs
- iSAM/iSAM2
- GTSAM-style smoothing
- pose-graph optimization

A useful learning progression:

$$\boxed{\text{KF} \rightarrow \text{EKF} \rightarrow \text{ESKF} \rightarrow \text{Lie groups} \rightarrow \text{IEKF} \rightarrow \text{Factor graphs/smoothing}}$$

That sequence gives a solid conceptual foundation for modern **VIO/SLAM and state estimation**.

---

## 14. One-sentence summary

> **Every variant in this list exists to relax one assumption of the standard KF - linearity (EKF/UKF), Euclidean state (ESKF/IEKF), Gaussian/unimodal belief (PF), an explicitly stored covariance (EnKF), known noise (Adaptive KF), clean measurements (Robust KF), numerical precision (SR-KF), causal-only information (RTS smoother), or synchronized sensors (multi-rate filtering) - and knowing which assumption a given problem violates is what tells you which variant to reach for.**

---

## 15. References

1. Julier, S. J., & Uhlmann, J. K. (1997). *New extension of the Kalman filter to nonlinear systems*. Proc. SPIE 3068, Signal Processing, Sensor Fusion, and Target Recognition VI, 182-193. https://doi.org/10.1117/12.280797 - the original UKF paper behind §1; already cited in [kf_ekf_iekf.md §11](kf_ekf_iekf.md#11-references).
2. Solà, J. (2017). *Quaternion kinematics for the error-state Kalman filter*. arXiv:1711.02508. https://arxiv.org/abs/1711.02508 - the standard ESKF reference behind §2; already cited in [kf_ekf_iekf.md §11](kf_ekf_iekf.md#11-references).
3. Barrau, A., & Bonnabel, S. (2017). *The Invariant Extended Kalman Filter as a Stable Observer*. IEEE Transactions on Automatic Control, 62(4), 1797-1812. https://arxiv.org/abs/1410.1465 - the IEKF paper behind §4; already cited in [kf_ekf_iekf.md §11](kf_ekf_iekf.md#11-references).
4. Bierman, G. J. (1977). *Factorization Methods for Discrete Sequential Estimation*. Academic Press. - the standard square-root/Cholesky-factor filtering reference behind §5.
5. Evensen, G. (1994). *Sequential data assimilation with a nonlinear quasi-geostrophic model using Monte Carlo methods to forecast error statistics*. Journal of Geophysical Research, 99(C5), 10143-10162. https://doi.org/10.1029/94JC00572 - the original Ensemble Kalman Filter paper behind §6.
6. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press. - covers both the particle filter and Monte Carlo Localization behind §7; already cited in [filtering_smoothing.md §12](../filtering_smoothing.md#12-references).
7. Mehra, R. K. (1970). *On the identification of variances and adaptive Kalman filtering*. IEEE Transactions on Automatic Control, 15(2), 175-184. https://doi.org/10.1109/TAC.1970.1099422 - the adaptive-noise-estimation reference behind §8.
8. Huber, P. J. (1964). *Robust Estimation of a Location Parameter*. Annals of Mathematical Statistics, 35(1), 73-101. https://doi.org/10.1214/aoms/1177703732 - the robust-loss reference behind §9; already cited in [pose_graph_optimization.md §17](../optimization/pose_graph_optimization.md#17-references).
9. Rauch, H. E., Tung, F., & Striebel, C. T. (1965). *Maximum likelihood estimates of linear dynamic systems*. AIAA Journal, 3(8), 1445-1450. https://doi.org/10.2514/3.3166 - the original RTS smoother paper behind §10.
10. Mourikis, A. I., & Roumeliotis, S. I. (2007). *A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation*. ICRA 2007, 3565-3572. https://doi.org/10.1109/ROBOT.2007.364024 - the MSCKF mentioned in §2, an ESKF-based VIO filter; already cited in [kf_ekf_iekf.md §11](kf_ekf_iekf.md#11-references).
11. Montemerlo, M., Thrun, S., Koller, D., & Wegbreit, B. (2002). *FastSLAM: A factored solution to the simultaneous localization and mapping problem*. Proc. AAAI-02, 593-598. - the Rao-Blackwellized particle-filter SLAM in §7.
12. Grisetti, G., Stachniss, C., & Burgard, W. (2007). *Improved techniques for grid mapping with Rao-Blackwellized particle filters*. IEEE Transactions on Robotics, 23(1), 34-46. https://doi.org/10.1109/TRO.2006.889486 - GMapping, §7.
