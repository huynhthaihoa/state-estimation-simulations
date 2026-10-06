# Visual-inertial initialization

> **Note**: A visual-inertial state estimator combines data from cameras and inertial measurement units (IMUs) to track the position, velocity, and orientation of a moving robot or device.

Before a visual-inertial estimator can run its normal loop (preintegrate IMU, optimize against camera constraints), we have to solve a chicken-and-egg bootstrapping problem: find metric scale, gravity direction, initial velocity, and IMU bias, none of which a monocular camera or a bias-corrupted IMU can give on its own.

This builds on:
- **IMU preintegration** ($\Delta R, \Delta v, \Delta p$ and their bias Jacobians) from [imu_preintegration.md](../optimization/imu_preintegration.md). This doc covers what must happen *before* that machinery can be trusted.
- **Factor graphs** from [factor_graph.md](../optimization/factor_graph.md), the optimizer we hand off to (§4).

---

## 1. The bootstrapping problem

Every other doc in this repo assumes a decent initial guess already exists: `pose_graph.py`'s optimizer starts from [dead-reckoning](../optimization/factor_graph.md#2-why-do-we-need-it), `bundle_adjustment_advanced.py`'s GN starts from a triangulated point, `pnp_estimation.py`'s GN starts from a linear DLT solve ([triangulation_pnp.md](triangulation_pnp.md)). Visual-inertial initialization is a place where *no* such starting point exists yet, and four unknowns must be pinned down before anything else can proceed:

- **Scale**: a monocular camera recovers structure and motion only up to an unknown positive scale factor. "This camera moved some distance $d$" could mean 1 meter or 100, and vision alone can't say which.
- **Gravity direction**: the accelerometer senses specific force (acceleration minus gravity), so gravity enters the preintegration residual (see [imu_preintegration.md §7](../optimization/imu_preintegration.md#7-what-the-accompanying-scripts-actually-do-and-dont), where the demo omits it). Handling it requires knowing which way gravity points in the estimator's frame.
- **Velocity**: the IMU measures acceleration, so integrating it gives velocity only up to an unknown starting value. Each keyframe's velocity must be recovered too.
- **Bias** ($b_g$, $b_a$): `imu_preintegration.md §1`'s "Problem B". Every raw sample was integrated using *some* bias estimate, and an uninitialized (zero) guess can be badly wrong, especially for the gyroscope.

If any of these is wrong at the start, the optimizer's very first linearization is already off.

---

## 2. The classic linear-alignment pipeline

The standard solution (popularized by VINS-Mono; see References) runs vision and inertial data through a short window of keyframes *before* switching over to the full nonlinear estimator, in two steps. Step 1 is closed-form; Step 2 is a linear solve followed by a short iterative gravity refinement:

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

> **Note**: Structure from Motion (SfM) is a computer-vision technique that recovers camera poses and sparse 3D point positions from a sequence of 2D images taken from different viewpoints.

With a single (monocular) camera, SfM translations and structure are correct only up to an unknown scale factor, while rotations are not affected. That is why §1 lists scale, not rotation, as an unknown.

**Intuition for Step 1** (toy case: one keyframe pair, 1 s apart):
- SfM says the camera turned 10.0°. The preintegrated gyro says 10.5°.
- The 0.5° gap, spread over 1 s, is a gyro bias of about 0.5°/s.
- The real step does this fit jointly over all keyframe pairs in the window.

**Step 1 - gyroscope bias.**
- Short vision-only SfM gives relative rotations between keyframes that don't depend on scale.
- Each should match the corresponding preintegrated $\Delta R$ ([imu_preintegration.md §3](../optimization/imu_preintegration.md#3-what-gets-compressed)); any mismatch is explained by the still-unknown gyro bias.
- $\Delta R$'s bias sensitivity is already a Jacobian ($J_{R,b_g}$, [imu_preintegration.md §4](../optimization/imu_preintegration.md#4-the-bias-sensitivity-jacobians)), so this is a small **linear least-squares** problem for $\delta b_g$ over all keyframe pairs in the window.
- It is solved once, in closed form, with no iteration, for the linearized (first-order) model.

**Step 2 - gravity, scale, and velocities.**
- **Tiny example (toy case: ignores gravity and starting velocity):** vision says the camera moved 2 units, and the IMU says the same move was 1 m. So the scale is $`s = 1\,\text{m} / 2\,\text{units} = 0.5`$ m/unit.
- The real solve repeats this match over many keyframe pairs. That also pins down gravity and the velocities.
- With the gyro bias corrected, $\Delta v$ and $\Delta p$ from every keyframe pair (still functions of the *unknown* scale $s$ and gravity vector $g$ in the vision frame) are combined with the vision-only relative poses into a second linear system.
- It is solved jointly for $s$, $g$, and every keyframe's velocity.
- The true gravity magnitude is known (~9.81 m/s²), so the raw linear $g$ is then refined by re-parameterizing it as a magnitude-constrained direction (2 degrees of freedom on a sphere) instead of a free 3-vector.
- Without this refinement, noise in the linear solve can give $g$ the wrong magnitude, which would corrupt the recovered scale.

**Accelerometer bias** is left near zero at this stage: over a short window its effect on $\Delta v/\Delta p$ is small relative to gravity and noise, so estimating it here is poorly conditioned. It is refined later, once the full optimizer has more data.

**What the alignment needs from the data.** The solve only works if the motion reveals the unknowns, so real systems check before trusting it:

- **Enough acceleration.** At constant velocity the accelerometer senses only gravity, and a scaled-up trajectory moving at a scaled-up constant speed fits the IMU data equally well, so scale is unobservable. The window needs acceleration that is both non-zero and sufficiently varied.
- **Enough translation.** Under pure rotation the vision-only SfM has no parallax to recover structure and translation from, so there is nothing for the IMU to align against.
- **Known camera-IMU extrinsics.** The alignment needs the rotation and offset between camera and IMU; VINS-Mono can estimate the extrinsic rotation online if it isn't calibrated.

VINS-Mono, for example, checks for sufficient parallax (and the paper describes an IMU-excitation check) and retries initialization with a later window if the current one isn't informative enough.

---

## 3. Why a mostly linear pipeline

Every other estimator in this doc set (§1's list) linearizes *around* an existing decent guess and takes a Gauss-Newton/Levenberg-Marquardt step. At initialization there is no guess to linearize around, for scale or gravity in particular. So we solve mostly with **linear systems**: a cheap, robust way to get *some* usable estimate before anything trustworthy exists to seed an iterative solve.

It isn't the only design real systems use:
- VINS-Mono's gravity refinement (Step 2) is itself a short iterative loop over the tangent-plane reparameterization.
- ORB-SLAM3 instead estimates scale, gravity direction, velocities and IMU biases with an inertial-only MAP (maximum a posteriori) optimization over keyframes from a vision-only initialization (Campos et al., 2020; see References), then refined jointly.

---

## 4. The hand-off

Once §2 produces an initial scale, gravity direction, per-keyframe velocities, and gyroscope bias (the accelerometer bias still at its near-zero starting value), we are in the situation every other doc in this repo assumes. A reasonable starting point exists, `imu_preintegration.md`'s $(\Delta R, \Delta v, \Delta p)$ bundles (now scaled and bias-corrected) become real edges, and [factor_graph.md](../optimization/factor_graph.md)-style nonlinear optimization takes over. This doc stops at that hand-off.

---

## 5. What this repo implements

Conceptual only: there is no accompanying script, and initialization isn't demonstrated numerically anywhere in `use_numpy/`/`use_manif/` (unlike `triangulation_pnp.md`'s two problems). Exercising §2 would need:
- synthetic vision-only SfM output
- a preintegrated IMU bundle to align against it

That is a much larger undertaking than the other scripts in this repo.

---

## 6. One-sentence summary

> **Visual-inertial initialization supplies the starting guess other docs assume: we align a short, sufficiently exciting vision-only window against preintegrated IMU bundles (a closed-form gyro-bias step, then a linear solve with a short gravity refinement) to recover scale, gravity direction, velocities, and gyroscope bias, then hand off to ordinary preintegration + factor-graph optimization.**

---

## 7. References

1. Qin, T., & Shen, S. (2017). *Robust Initialization of Monocular Visual-Inertial Estimation on Aerial Robots*. IROS 2017, 4225-4232. https://doi.org/10.1109/IROS.2017.8206284 - the source of the linear gyro-bias/gravity-scale-velocity alignment pipeline in §2.
2. Qin, T., Li, P., & Shen, S. (2018). *VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator*. IEEE Transactions on Robotics, 34(4), 1004-1020. https://doi.org/10.1109/TRO.2018.2853729 - the full system this initialization pipeline bootstraps; already cited in [filtering_smoothing.md §12](../filtering_smoothing.md#12-references) and [marginalization.md §11](../optimization/marginalization.md#11-references).
3. Campos, C., Montiel, J. M. M., & Tardós, J. D. (2020). *Inertial-Only Optimization for Visual-Inertial Initialization*. ICRA 2020, 51-57. https://doi.org/10.1109/ICRA40945.2020.9197334 - ORB-SLAM3's inertial-only MAP-estimation alternative to §2's linear pipeline, named in §3's contrast.
