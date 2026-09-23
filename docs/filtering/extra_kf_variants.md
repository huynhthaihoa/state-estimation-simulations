# An overview of other prominent Kalman Filter variants

For **robotics, SLAM, visual-inertial estimation, and embedded systems**, there are several Kalman filter variants worth knowing. You don't need to master all of them, but it's useful to know **why each one exists**.

This builds directly on:
- The **standard KF, EKF, and IEKF** foundations from [kf_ekf_iekf.md](kf_ekf_iekf.md) - this doc surveys the wider family those three sit inside.
- **Lie groups and $SO(3)$ / $SE(3)$** from [lie_algebra.md §5, §11](../foundations/lie_algebra.md#5-lie-group-the-space-of-valid-transformations) - the structure §2's ESKF and §4's IEKF sections both lean on.
- **Robust loss functions (Huber, etc.)** from [pose_graph_optimization.md §16](../optimization/pose_graph_optimization.md#16-robust-loss-functions-used-to-handle-false-loop-closures) - referenced directly in §9's Robust KF discussion.

A good mental map is:

```text
                         Kalman Filter
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
       Linear              Nonlinear           Non-Gaussian /
          │                   │                  uncertain
          │                   │                    │
      Standard KF       ┌─────┴─────┐         Particle Filter
                        │           │
                       EKF     UKF (sigma points)
                        │
                      ESKF
                        │
                      IEKF
```

And then there are variants dealing with **noise, time, robustness, and computational constraints**.

---

## 1. Unscented Kalman Filter (UKF)

This is probably the **most important variant to learn after EKF**.

The EKF says:

> "I'll linearize the nonlinear function using a Jacobian."

The UKF says:

> "I don't want to calculate Jacobians. I'll sample a few carefully chosen points around my estimate and see how the nonlinear function transforms them."

These points are called **sigma points**.

Imagine your uncertainty is an ellipse:

```text
              •
           .     .
        .           .
       •      x      •
        .           .
           .     .
              •
```

Instead of approximating the nonlinear function with a tangent line like EKF, UKF sends these representative points through the actual nonlinear function:

```text
     uncertainty          nonlinear transformation

       •                         •
    .     .                   .     .
   •   x   •       --->       •       •
    .     .                   .  x    .
       •                         •
```

Then it reconstructs the new mean and covariance.

### EKF

$$\text{nonlinear function} \rightarrow \text{Jacobian} \rightarrow \text{linear approximation}$$

### UKF

$$\text{nonlinear function} \rightarrow \text{sigma points} \rightarrow \text{transformed sigma points} \rightarrow \text{new mean/covariance}$$

**Why care?**

UKF can be attractive when:

* the nonlinearities are strong,
* calculating Jacobians is difficult,
* you want a derivative-free method.

But it can be computationally more expensive than EKF, especially for high-dimensional states.

See `run_ukf` in [`use_numpy/pointcloud_pose_tracking.py`](../../use_numpy/pointcloud_pose_tracking.py) / [`use_manif/pointcloud_pose_tracking.py`](../../use_manif/pointcloud_pose_tracking.py) for a runnable manifold-UKF implementation of this idea.

---

## 2. Error-State Kalman Filter (ESKF)

This is one of the most important variants to know for SLAM/VIO work.

An ESKF doesn't estimate the entire state error directly.

Instead, it separates:

$$\text{nominal state} + \text{small error state}$$

For example:

$$R = \hat R \exp(\delta\theta^\wedge)$$

while the nominal state might contain:

$$\hat X = (\hat R,\hat p,\hat v,\hat b_g,\hat b_a)$$

and the EKF operates primarily on:

$$\delta x = (\delta\theta,\delta p,\delta v,\delta b_g,\delta b_a)$$

This is extremely useful because some states, especially **rotation**, live on manifolds rather than ordinary Euclidean vector spaces.

You'll see things like:

* ESKF
* Error-State EKF
* Multiplicative EKF (MEKF)
* Right/left invariant error-state filters

in visual-inertial odometry and inertial navigation.

### Important relationship

Don't think:

> ESKF vs IEKF

as completely unrelated alternatives.

They are **closely related ideas**.

IEKF is essentially taking the idea of defining the estimation error carefully and exploiting the **invariant structure/symmetries** of the system.

---

## 3. Unscented vs Extended: a useful comparison

|                             | EKF                 | UKF                         |
| --------------------------- | ------------------- | --------------------------- |
| Nonlinear system            | ✓                   | ✓                           |
| Jacobians                   | Required            | No                          |
| Linearizes function         | ✓                   | No                          |
| Sigma points                | No                  | ✓                           |
| Computational cost          | Lower               | Higher                      |
| Easy for complicated models | Sometimes difficult | Often easier                |
| Common in robotics          | Very common         | Less dominant than EKF/ESKF |

For robotics, the typical learning order is:

**KF → EKF → ESKF → IEKF**

before spending too much time on UKF.

---

## 4. Invariant EKF (IEKF)

We already discussed this one, but it's worth putting it into the broader family.

The important idea is:

> **Exploit the symmetry and Lie-group structure of the system.**

For example:

$$R \in SO(3)$$

and

$$T \in SE(3)$$

rather than treating everything as an ordinary vector.

There are particularly interesting formulations involving:

* left-invariant error
* right-invariant error

which can give the estimation error dynamics properties that are independent of the current state in ways that ordinary EKF linearizations aren't. See [left_right_invariant.md](left_right_invariant.md) for exactly what distinguishes the two and when to reach for each.

This is particularly relevant to:

* IMU navigation
* VIO
* SLAM
* robotics
* pose estimation

---

## 5. Square-Root Kalman Filter (SR-KF)

This one is less about changing the **estimation philosophy** and more about improving **numerical stability**.

Standard KF stores covariance:

$$P$$

The square-root version instead stores something like:

$$P = SS^\top$$

where $S$ is a square-root factor, often obtained through Cholesky decomposition.

Why?

Because covariance matrices should be:

* symmetric
* positive semi-definite

But floating-point numerical errors can cause problems.

SR-KF tends to be more numerically stable.

### Intuition

Standard KF:

> "I'll carry the uncertainty matrix directly."

Square-root KF:

> "I'll carry a factor of the uncertainty matrix because it's numerically safer."

This becomes particularly interesting in **large-scale estimation** and systems where numerical stability is critical.

---

## 6. Ensemble Kalman Filter (EnKF)

This is quite different.

Instead of explicitly representing the probability distribution with:

$$\mu, P$$

you maintain an **ensemble of possible states**:

```text
     •
        •
  •        •
       •
 •              •
          •
```

Each point represents a possible realization of the system.

You propagate them through the nonlinear model.

This is especially popular in:

* weather prediction
* geophysical systems
* ocean modeling
* very high-dimensional systems

It's generally **less central to robotics/SLAM** than EKF/ESKF/IEKF.

---

## 7. Particle Filter (PF)

This one drops the Gaussian assumption that every other variant in this list relies on.

Instead of a mean and covariance, your belief is a swarm of weighted samples ("particles"), each one a full hypothesis for the state:

```text
belief ≈ {(x⁽¹⁾, w⁽¹⁾), (x⁽²⁾, w⁽²⁾), ..., (x⁽ᴺ⁾, w⁽ᴺ⁾)}
```

Each particle is propagated through the (possibly highly nonlinear) motion model, reweighted by how well it explains the latest measurement, and periodically resampled so particles that poorly explain the data get replaced by copies of the better ones:

```text
     particles              after motion            after weighting
                              + resampling

   •  •   •                  •    •                    •  •
  •    •     •     --->     •  •    •      --->        •••  •
     •    •                    •   •                       •
```

This lets a PF represent **multimodal** beliefs - "the robot is either in room A or room B, I genuinely don't know which" - something a single Gaussian, which every KF-family filter assumes, simply cannot express.

**Why care?**

* No linearity or Gaussian-noise assumption at all - works for arbitrarily nonlinear, non-Gaussian problems.
* Naturally represents multimodal beliefs (ambiguous data association, the kidnapped-robot problem, global localization).
* Classic robotics use case: **Monte Carlo Localization (MCL)** - localizing a robot on a known map from range/bearing measurements.

**The catch**: accuracy scales with particle count, and in high-dimensional state spaces (like a full SLAM state vector) you need an impractically large number of particles to cover the space adequately. That's why particle filters are common for low-dimensional localization but rare for full SLAM state estimation, where EKF/UKF/factor-graph approaches dominate instead.

---

## 8. Adaptive Kalman Filter

Normally you assume:

$$Q = \text{known}$$

$$R = \text{known}$$

where:

* $Q$ = process noise covariance
* $R$ = measurement noise covariance

But in real life, those values may change.

For example:

> Your IMU might behave differently when the robot is vibrating heavily.

So an adaptive KF tries to estimate or adjust $Q$ and/or $R$ online.

This is useful when the environment or sensor quality changes over time.

---

## 9. Robust Kalman Filter

Standard KF essentially assumes:

> "My noise is reasonably well-behaved, approximately Gaussian."

But what if your camera occasionally produces a **terrible outlier**?

For example:

```text
measurements:

       • • • •
      • • • •
     • • • •
                     X  ← outlier
```

A conventional KF may be pulled toward that outlier.

Robust filtering tries to reduce the influence of bad measurements.

This is particularly relevant to:

* visual SLAM
* feature tracking
* GNSS
* LiDAR
* multi-sensor fusion

Although in robotics, robust losses such as **Huber loss** are also commonly used within optimization-based estimators rather than relying exclusively on a "robust KF."

---

## 10. Kalman Smoother

This isn't exactly another KF variant, but it's important enough to know.

A normal Kalman filter estimates:

$$x_k | z_1,\ldots,z_k$$

In other words:

> "What is the state **now**, given everything I've seen so far?"

A smoother can use **future measurements** too:

$$x_k | z_1,\ldots,z_N$$

So:

```text
Filtering:

t0 → t1 → t2 → t3 → t4
                 ↑
             estimate


Smoothing:

t0 → t1 → t2 → t3 → t4
       ↑
       estimate using
       information from
       both past AND future
```

The classic example is the **Rauch–Tung–Striebel (RTS) smoother**.

This is very important for **offline SLAM and trajectory estimation**.

---

## 11. Multi-rate / asynchronous Kalman filtering

This one is particularly practical for robotics.

Suppose you have:

```text
IMU       200 Hz
Camera     30 Hz
LiDAR      10 Hz
GPS         1 Hz
```

You don't want to force everything to run at the same frequency.

A filter can propagate using IMU:

```text
IMU → predict
IMU → predict
IMU → predict
...
```

and then update whenever another sensor arrives:

```text
Camera → update
LiDAR  → update
GPS    → update
```

This is one of the fundamental patterns behind real-time sensor fusion.

---

## 12. Which ones should you prioritize learning?

For robotics work spanning **computer vision, SLAM, embedded systems, and resource-constrained platforms**, it's not necessary to learn every Kalman variant equally.

A reasonable prioritization:

### Tier 1 - Must understand

#### 1. Standard KF

Understand:

* prediction
* measurement update
* covariance
* Kalman gain
* $Q$, $R$

↓

#### 2. EKF

Understand:

* nonlinear dynamics
* Jacobians
* local linearization

↓

#### 3. Error-State EKF / ESKF

Understand:

* nominal state
* error state
* perturbations
* why rotations need special treatment

↓

#### 4. IEKF

Understand:

* Lie groups
* $SO(3)$
* $SE(3)$
* left/right invariant errors
* system symmetries

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

## 13. And there's one more important distinction for SLAM

Once you get into modern robotics, you'll encounter two big families:

### Filtering

```text
IMU → KF/EKF/ESKF/IEKF → current state
```

You maintain a state estimate recursively.

### Optimization / smoothing

```text
        ┌── camera ──┐
        │            │
x0 ─── x1 ─── x2 ─── x3
 │      │      │      │
IMU    IMU    IMU    GPS
        ↓
   nonlinear optimization
```

Examples include:

* Bundle Adjustment
* Factor Graphs
* iSAM / iSAM2
* GTSAM-style smoothing
* pose-graph optimization

A particularly useful learning progression is:

$$\boxed{KF \rightarrow EKF \rightarrow ESKF \rightarrow Lie\ Groups \rightarrow IEKF \rightarrow Factor\ Graphs/Smoothing}$$

Once you understand that sequence, you'll have a pretty solid conceptual foundation for modern **VIO/SLAM and state estimation**.

---

## 14. One-sentence summary

> **Every variant in this list exists to relax one assumption of the standard KF - linearity (EKF/UKF), Euclidean state (ESKF/IEKF), Gaussian/unimodal belief (PF/EnKF), known noise (Adaptive KF), clean measurements (Robust KF), numerical precision (SR-KF), or causal-only information (RTS smoother) - and knowing which assumption a given problem violates is what tells you which variant to reach for.**

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
10. Mourikis, A. I., & Roumeliotis, S. I. (2007). *A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation*. ICRA 2007, 3565-3572. https://doi.org/10.1109/ROBOT.2007.364024 - a concrete filtering-family system combining several of §2/§4's ideas (ESKF-style error state) for VIO; already cited in [kf_ekf_iekf.md §11](kf_ekf_iekf.md#11-references).
