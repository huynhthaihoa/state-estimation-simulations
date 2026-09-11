# Visual-inertial initialization

> **Before a visual-inertial estimator can run its normal loop (preintegrate IMU, optimize against camera constraints), it has to solve a chicken-and-egg bootstrapping problem: figure out metric scale, gravity direction, initial velocity, and IMU bias, none of which a monocular camera or a bias-corrupted IMU can tell you on its own.**

This builds directly on:
- **IMU preintegration** ($\Delta R, \Delta v, \Delta p$ and their bias Jacobians) from [imu_preintegration.md](../optimization/imu_preintegration.md) - this doc explains what has to happen *before* that machinery can be trusted.
- **Factor graphs** from [factor_graph.md](../optimization/factor_graph.md) - the optimizer this initialization eventually hands off to.

---

## 1. The bootstrapping problem

Every other doc in this repo assumes a decent initial guess already exists: `pose_graph.py`'s optimizer starts from dead-reckoning, `bundle_adjustment_advanced.py`'s GN starts from a triangulated point, `pnp_estimation.py`'s GN starts from a linear DLT solve ([triangulation_pnp.md](triangulation_pnp.md)). Visual-inertial initialization is the one place in this doc set where *no* such starting point exists yet, and three specific unknowns have to be pinned down before anything else can proceed:

- **Scale**: a single monocular camera can recover structure and motion only up to an unknown positive scale factor - "this camera moved some distance $d$" could mean 1 meter or 100, and vision alone can never tell you which.
- **Gravity direction**: preintegration's $\Delta v$ accumulates the effect of true acceleration *and* gravity together; separating them requires knowing which way gravity points in whatever frame the estimator is working in.
- **Bias** ($b_g$, $b_a$): `imu_preintegration.md §1`'s "Problem B" - every raw sample was integrated using *some* bias estimate, and an uninitialized (zero) bias guess can be badly wrong, especially for the gyroscope.

Get any of these wrong at the start, and the optimizer's very first linearization is already off - undermining exactly the guarantees the rest of this doc set relies on.

---

## 2. The classic linear-alignment pipeline

The standard solution (popularized by VINS-Mono; see References) runs vision and inertial data through a short window of keyframes *before* switching over to the full nonlinear estimator, in three linear (closed-form) steps:

```text
Vision-only SfM over a short window
        ↓
(relative rotations/translations, scale unknown)
        ↓
Step 1: gyroscope bias
        ↓
Step 2: gravity + scale + velocities
        ↓
Hand off to IMU preintegration + factor-graph optimization
```

**Step 1 - gyroscope bias.** A short vision-only structure-from-motion pass gives relative rotations between keyframes that don't depend on scale (rotation has no scale ambiguity, unlike translation). Each such rotation should match the corresponding preintegrated $\Delta R$ from [imu_preintegration.md §3](../optimization/imu_preintegration.md#3-what-gets-compressed) - any mismatch is explained by the (still-unknown) gyro bias. Because $\Delta R$'s bias-sensitivity is already tracked as a Jacobian ($J_{R,b_g}$, [imu_preintegration.md §4](../optimization/imu_preintegration.md#4-the-bias-sensitivity-jacobians)), this becomes a small **linear least-squares** problem for $\delta b_g$ across all keyframe pairs in the window - solved once, in closed form, no iteration needed.

**Step 2 - gravity, scale, and velocities.** With the gyro bias corrected, the preintegrated $\Delta v$ and $\Delta p$ from every keyframe pair (still functions of the *unknown* scale $s$ and gravity vector $g$ in the vision frame) are combined with the vision-only relative poses into a second linear system, solved jointly for $s$, $g$, and every keyframe's velocity at once. Because the true gravity magnitude is known (~9.81 m/s²), the raw linear solution for $g$ is refined by re-parameterizing it as a magnitude-constrained direction (2 degrees of freedom on a sphere) rather than a free 3-vector - without this, noise in the linear solve can produce a gravity vector with the wrong magnitude, which would silently bias the recovered scale.

**Accelerometer bias** is deliberately left near zero at this stage: over a short initialization window its effect on $\Delta v/\Delta p$ is small relative to gravity and noise, so trying to estimate it here is poorly conditioned - it's refined later, once the full optimizer is running with much more data to constrain it.

---

## 3. Why this has to be linear at all

Every other estimator in this doc set (§1's list) linearizes *around* an existing decent guess and takes a Gauss-Newton/Levenberg-Marquardt step from there. Initialization can't do that - there is no existing guess to linearize around yet, for scale or gravity in particular. That's why every step above is deliberately solved as a **closed-form linear system** instead of an iterative nonlinear one: it's the only tool available before a trustworthy initial estimate exists to seed anything iterative.

---

## 4. The hand-off

Once §2 produces an initial scale, gravity direction, and per-keyframe velocity/bias estimate, the problem looks exactly like every other doc in this repo assumes: a reasonable starting point exists, `imu_preintegration.md`'s $(\Delta R, \Delta v, \Delta p)$ bundles (now correctly scaled and bias-corrected) become real edges, and [factor_graph.md](../optimization/factor_graph.md)-style nonlinear optimization takes over from there. This doc's job stops exactly at that hand-off.

---

## 5. What this repo implements

Conceptual only - there is no accompanying script for this doc. Unlike `triangulation_pnp.md`'s two problems, initialization isn't demonstrated numerically anywhere in `use_numpy/`/`use_manif/`; a reader wanting to exercise §2 for real would need synthetic vision-only SfM output plus a preintegrated IMU bundle to align against it, which is a substantially larger undertaking than the other scripts in this repo.

---

## 6. References

1. Qin, T., & Shen, S. (2017). *Robust Initialization of Monocular Visual-Inertial Estimation on Aerial Robots*. IROS 2017, 4225-4232. https://doi.org/10.1109/IROS.2017.8206284 - the source of the linear gyro-bias / gravity-scale-velocity alignment pipeline in §2.
2. Qin, T., Li, P., & Shen, S. (2018). *VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator*. IEEE Transactions on Robotics, 34(4), 1004-1020. https://doi.org/10.1109/TRO.2018.2853729 - the full system this initialization pipeline bootstraps; already cited in [filtering_smoothing.md §12](../filtering_smoothing.md#12-references) and [marginalization.md §9](../optimization/marginalization.md#9-references).

---

## 7. One-sentence summary

> **Visual-inertial initialization solves, once and in closed form, the one problem every other doc in this repo assumes is already solved - a trustworthy starting guess - by linearly aligning a short vision-only window against preintegrated IMU bundles to recover scale, gravity direction, and initial velocity/bias, before handing off to the ordinary preintegration + factor-graph machinery for good.**
