# Error-state Kalman filter (ESKF): estimate the correction, not the state

## Intuition

An [IMU](../glossary.md#imu-inertial-measurement-unit)-driven filter has to carry an **orientation**, and an orientation is not an ordinary vector. Every way to store it has a catch:

| Representation | Numbers | Catch for a Kalman filter |
| --- | --- | --- |
| Euler angles | 3 | singular at some attitudes (gimbal lock) |
| Rotation matrix | 9 | 6 constraints (orthonormal columns) that an additive update breaks |
| Unit [quaternion](../glossary.md#quaternion) | 4 | 1 constraint (length 1) that an additive update breaks |

A plain EKF corrects its state by **adding** $`K r`$ (gain times residual). Added to a quaternion, the result is no longer unit length, so implementations renormalize it, and unless the [covariance](../glossary.md#covariance) $`P`$ is mapped through the same renormalization, it no longer matches the state (§4.2).

The **error-state Kalman filter** (**ESKF**) sidesteps this by splitting the job in two:

- The **nominal state** $`\hat x`$ carries the large motion: the rotation as a proper rotation matrix (or unit quaternion), plus velocity, position and the sensor [biases](../glossary.md#bias). It is integrated with the full nonlinear equations. It has no covariance of its own; the filter linearizes around it.
- The **error state** $`\delta x`$ is the small gap between the truth and the nominal state. It holds the rotation error as a **3-vector** $`\delta\theta`$ (axis × angle, rad), with no constraint and no singularity near zero. The Kalman filter estimates only $`\delta x`$.

After each measurement update, the estimated error is **injected** into the nominal state and **reset** to zero.

**Toy example (yaw only):**
- Nominal yaw 30°. A position fix suggests the error is $`\delta\theta = 2°`$.
- Inject: the nominal yaw becomes 32°.
- Reset: $`\delta\theta = 0`$ again.

Two things follow:
- The error state has one number per independent quantity (here 15: three each for position, velocity, rotation and the two biases), so its covariance has no direction that does not exist physically.
- The filter always linearizes around $`\delta x = 0`$, which is where the error actually is after each reset, so the first-order approximation stays accurate.

[`eskf_imu_position_fix.py`](../../use_numpy/eskf_imu_position_fix.py) runs this filter on a 3D body and compares it with the two shortcuts it avoids: leaving the biases out, and putting a quaternion directly in an EKF state.

### Scope
- This is the multiplicative/error-state family of [kf_ekf_iekf.md §5](kf_ekf_iekf.md#5-the-really-important-difference-how-do-you-define-error). How the invariant EKF differs is in §5 below.
- The conventions follow Solà (2017): Hamilton quaternions, scalar first, and the rotation error on the right (in the body frame), $`R = \hat R\,\mathrm{Exp}(\delta\theta)`$. Trawny and Roumeliotis (2005) use the JPL convention, which changes the order of quaternion products and some signs.

---

## 1. The setup

| Item | In the script | Default |
| --- | --- | --- |
| Motion | smooth 3D: body rate and world-frame acceleration are sums of sines with random phases | 0.4 rad/s, 0.5 m/s² amplitude, 30 s |
| **Input**: IMU | gyro reading $`\tilde\omega`$ (body angular rate) and accelerometer reading $`\tilde a`$ (**specific force**: acceleration minus gravity, in the body frame), each with white noise and a constant unknown bias | 100 Hz; noise 0.01 rad/s and 0.05 m/s² per sample; bias 1σ 0.01 rad/s and 0.05 m/s² |
| **Measurement**: position fix | $`z = p + n_p`$, the world-frame position plus noise | 1 Hz, 0.1 m per axis |
| Initial guess | truth perturbed in position, velocity and attitude; biases start at 0 | 0.1 m, 0.1 m/s, 2° per axis |

The IMU drives the prediction: its noise is **process noise**, not measurement noise. The true trajectory is integrated with the same discretization the filters use, so any model error comes from a filter's own assumptions (for example, `eskf_no_bias` ignoring the biases).

**State** (15 numbers in the error state):

| Symbol | Meaning | Units |
| --- | --- | --- |
| $`R`$ | orientation, body frame → world frame | rotation matrix |
| $`v`$, $`p`$ | velocity and position in the world frame | m/s, m |
| $`b_a`$, $`b_g`$ | accelerometer and gyro bias (constant offsets added to each reading) | m/s², rad/s |
| $`\delta p, \delta v`$ | position and velocity error, truth − nominal | m, m/s |
| $`\delta\theta`$ | rotation error, $`R = \hat R\,\mathrm{Exp}(\delta\theta)`$ | rad (3-vector) |
| $`\delta b_a, \delta b_g`$ | bias errors, truth − nominal | m/s², rad/s |

Here $`\mathrm{Exp}`$ turns a rotation vector into a rotation matrix and $`\mathrm{Log}`$ is its inverse ([glossary](../glossary.md#exp-and-log-maps)); $`[\,a\,]_\times`$ is the skew-symmetric matrix with $`[\,a\,]_\times b = a \times b`$ ([hat operator](../glossary.md#hat-operator)).

---

## 2. One ESKF cycle

### 2.1 Predict, nominal state (`eskf_predict`)

With the bias-corrected readings $`a = \tilde a - \hat b_a`$ and $`\omega = \tilde\omega - \hat b_g`$, gravity $`g = (0, 0, -9.81)`$ m/s² and the IMU interval $`\Delta t`$:

```math
\hat R \leftarrow \hat R\,\mathrm{Exp}(\omega\,\Delta t) \qquad \hat v \leftarrow \hat v + (\hat R a + g)\,\Delta t \qquad \hat p \leftarrow \hat p + \hat v\,\Delta t + \tfrac12(\hat R a + g)\,\Delta t^2
```

All right-hand sides use the values from the start of the step. The biases stay as they are. This is plain dead reckoning; no covariance is involved.

### 2.2 Predict, error state (`eskf_transition`)

The error evolves as $`\delta x \leftarrow F\,\delta x + L\,n`$, where $`n`$ stacks the accelerometer noise, gyro noise and the two bias random walks. The covariance follows as $`P \leftarrow F P F^\top + L Q L^\top`$. Over one step, to first order:

| Error | Driven by | Term in $`F`$ | In words |
| --- | --- | --- | --- |
| $`\delta p`$ | $`\delta v`$, and $`\delta\theta`$, $`\delta b_a`$ through $`\tfrac12\Delta t^2`$ | $`I\,\Delta t`$ | a velocity error becomes a position error |
| $`\delta v`$ | $`\delta\theta`$ | $`-\hat R\,[\,a\,]_\times\,\Delta t`$ | a tilt error rotates the measured specific force (mostly the 9.81 m/s² that holds the body up) slightly the wrong way |
| $`\delta v`$ | $`\delta b_a`$ | $`-\hat R\,\Delta t`$ | an accelerometer-bias error is integrated as a fake acceleration |
| $`\delta\theta`$ | $`\delta\theta`$ | $`\mathrm{Exp}(\omega\Delta t)^\top`$ | the body-frame error is re-expressed in the rotated body frame |
| $`\delta\theta`$ | $`\delta b_g`$ | $`-J_r(\omega\Delta t)\,\Delta t`$ | a gyro-bias error is integrated as a fake rotation |
| $`\delta b_a, \delta b_g`$ | themselves | $`I`$ | constant, apart from the assumed random walk |

$`J_r`$ is the right Jacobian of $`\mathrm{Exp}`$ ([glossary](../glossary.md#right-and-left-jacobians)); for one IMU step it is close to $`I`$.

**Worked numbers (why the error terms matter):**
- A 1° tilt error misplaces $`9.81 \sin 1° = 0.17`$ m/s² of gravity into the horizontal. Integrated twice for 10 s, that is $`\tfrac12 \cdot 0.17 \cdot 10^2 \approx 8.6`$ m of position error.
- A gyro-bias error of 0.01 rad/s turns into 0.57° of attitude error per second.

The coupling rows are also what makes the unmeasured states correctable: the position fix sees only $`p`$, but through these rows $`P`$ correlates the position error with $`\delta\theta`$ and the biases, so the update corrects them too.

### 2.3 Update (`eskf_update`)

The position fix measures $`p`$ directly, so the residual is $`r = z - \hat p`$ and $`H = [\,I \;\; 0 \;\; 0 \;\; 0 \;\; 0\,]`$ (3×15). The update is the ordinary Kalman update on $`\delta x`$, whose predicted value is 0:

```math
K = P H^\top (H P H^\top + \sigma_p^2 I)^{-1} \qquad \widehat{\delta x} = K r \qquad P \leftarrow (I - K H) P
```

### 2.4 Inject and reset

**Inject** the estimated error into the nominal state, with the rotation composed rather than added:

```math
\hat p \leftarrow \hat p + \widehat{\delta p} \qquad \hat v \leftarrow \hat v + \widehat{\delta v} \qquad \hat R \leftarrow \hat R\,\mathrm{Exp}(\widehat{\delta\theta}) \qquad \hat b \leftarrow \hat b + \widehat{\delta b}
```

**Reset** $`\delta x`$ to zero. Strictly, moving the nominal rotation also changes the frame the remaining rotation error is expressed in, so $`P`$ is mapped through the **reset Jacobian** $`G = I - \tfrac12[\,\widehat{\delta\theta}\,]_\times`$ on the rotation block: $`P \leftarrow G P G^\top`$ (Solà 2017, §6.3); $`G`$ differs from $`I`$ only to first order in the injected error. Here it makes no measurable difference: with `--no-reset-jacobian` ($`G = I`$) on the same trials, the ESKF's NEES is 16.6 instead of 16.4 and its attitude RMSE 1.36° instead of 1.35°.

| Step | Acts on | Linearized? |
| --- | --- | --- |
| Predict, nominal | $`\hat R, \hat v, \hat p`$ | no, full model |
| Predict, error | $`P`$ | yes, around $`\delta x = 0`$ |
| Update | $`\widehat{\delta x}`$, $`P`$ | yes (here $`H`$ is exactly linear) |
| Inject | $`\hat R, \hat v, \hat p, \hat b`$ | no, exact composition |
| Reset | $`\delta x \to 0`$, $`P`$ | first order in $`\widehat{\delta\theta}`$ |

---

## 3. The comparison filters

| Filter | State | What it does differently |
| --- | --- | --- |
| `eskf` | nominal $`(R, v, p, b_a, b_g)`$, error 15 | the filter of §2 |
| `eskf_no_bias` | nominal $`(R, v, p)`$, error 9 | assumes both biases are zero |
| `qekf_naive` | 16 numbers: $`p, v, q, b_a, b_g`$ with the quaternion $`q`$ in the state | one EKF on the full state; adds $`K r`$ to $`q`$, renormalizes $`q`$, leaves $`P`$ as it is |
| `qekf_projected` | same 16 | as `qekf_naive`, but also maps $`P`$ through the Jacobian of $`q \mapsto q/\lVert q\rVert`$, $`(I - u u^\top)/\lVert q\rVert`$ with $`u = q/\lVert q\rVert`$ |

For a fair comparison, the quaternion EKF's covariance is mapped to the same 15 error coordinates before computing [NEES](../glossary.md#consistency-and-nees), using $`\delta\theta \approx 2\,\mathrm{vec}(\hat q^{*} \otimes \delta q)`$ (twice the vector part of the error quaternion), where $`\delta q = q - \hat q`$ is the additive quaternion error. NEES is the squared error weighted by the filter's own covariance; averaged over trials, a **consistent** filter's NEES equals its number of states (15, or 9 for `eskf_no_bias`).

---

## 4. Results at the defaults

50 Monte Carlo trials (motion, biases, noise and initial error redrawn each time), averaged over the second half of the run (15-30 s). With 50 trials, the 95% band for a consistent average NEES is [13.5, 16.6] for 15 states and [7.9, 10.2] for 9.

| Filter | Attitude RMSE | Position RMSE | NEES | States |
| --- | --- | --- | --- | --- |
| `eskf` | 1.35° | 0.185 m | 16.4 | 15 |
| `eskf_no_bias` | 12.9° | 1.51 m | 1736 | 9 |
| `qekf_naive` | 1.74° | 0.189 m | 41.6 | 15 |
| `qekf_projected` | 1.35° | 0.185 m | 16.4 | 15 |

![Four panels from eskf_imu_position_fix.py at its defaults: attitude error RMSE and position error RMSE over 30 s for the ESKF, the ESKF without bias states, the naive quaternion EKF and the projected quaternion EKF; their average NEES over time against the consistent values 15 and 9; and their second-half NEES against the initial attitude error from 0.5 to 10 degrees](../../assets/eskf_imu_position_fix.png)

*Figure: `uv run python use_numpy/eskf_imu_position_fix.py --out assets/eskf_imu_position_fix.png` (about 3 min). The projected quaternion EKF (dashed) lies on top of the ESKF.*

### 4.1 Without bias states, the filter believes a drifting model

The biases are only 0.01 rad/s and 0.05 m/s² (1σ), yet `eskf_no_bias` averages 12.9° attitude error and 1.5 m position error over the second half, against 1.35° and 0.19 m with bias states. Its NEES is 1736 against a consistent 9, about 190 times too large: its covariance claims errors far smaller than the ones it makes, because its model has no term for the drift the biases cause. With bias states in the filter, the position fixes and the motion make the biases observable, and the filter estimates them instead.

### 4.2 The quaternion EKF: the problem is the renormalization, not the quaternion

`qekf_naive` is overconfident (NEES 41.6) and less accurate in attitude (1.74° vs 1.35°). `qekf_projected`, which differs only in also mapping $`P`$ through the renormalization, matches the ESKF to the shown digits.

What goes wrong in the naive version:
- A unit quaternion has 4 numbers but only 3 rotational degrees of freedom. The fourth direction, along $`q`$ itself, changes only the length of $`q`$.
- After an additive update and renormalization, $`P`$ keeps a small amount of uncertainty in that length direction, which no real rotation error has.
- $`R(q)\,a`$ changes with the length of $`q`$ (in this script's form it scales with $`\lVert q\rVert^2`$), so the transition matrix couples that fictitious direction into velocity with a lever of about $`2\lVert a\rVert \approx 20`$ m/s².
- The position fixes then appear to measure it, and the filter gains confidence in its attitude that it has not earned. In one check (5 trials, outside the script), the naive filter's reported attitude standard deviation at 30 s was 0.77° against an actual RMSE of 1.26°; the projected filter's was 0.98° against 0.68°.
- Removing only that coupling from the transition matrix (15 trials, outside the script) brought the naive filter's NEES from 52.3 to 15.7.

So a quaternion EKF can be made to work, but it needs this repair: its 4-number attitude has a direction (the length of $`q`$) that has to be kept out of $`P`$. The ESKF's 3-vector rotation error has no such direction and nothing to renormalize.

A single run can mislead here: in the script's single run, `qekf_naive` happened to end with a smaller attitude error (0.43°) than the ESKF (0.84°). Only the Monte Carlo average shows the gap.

### 4.3 Larger initial attitude errors

Second-half NEES, 10 trials per value (the 2° row is the 50-trial run):

| Initial attitude error (per axis) | `eskf` | `eskf_no_bias` | `qekf_naive` | `qekf_projected` |
| --- | --- | --- | --- | --- |
| 0.5° | 15.9 | 2348 | 20.6 | 15.9 |
| 2° | 16.4 | 1736 | 41.6 | 16.4 |
| 5° | 19.5 | 1699 | 89.1 | 19.5 |
| 10° | 27.7 | 8927 | 727 | 28.4 |

- The naive quaternion EKF's overconfidence grows quickly with the initial error (NEES 20.6 at 0.5°, 727 at 10°).
- The ESKF is not immune to large errors either: at 10° its NEES is 27.7 against 15. The likely cause (not checked separately) is that its linearization is first order, which is less accurate while the error is large.

---

## 5. ESKF vs. invariant EKF

The ESKF's rotation error $`\delta\theta = \mathrm{Log}(\hat R^\top R)`$ is already invariant for the rotation alone ([kf_ekf_iekf.md §5](kf_ekf_iekf.md#a-caveat)). The difference is in the rest of the state: the ESKF keeps $`\delta v`$ and $`\delta p`$ as plain differences in world coordinates, so its transition matrix depends on the estimate (the $`-\hat R[\,a\,]_\times`$ term above). The [invariant EKF](kf_ekf_iekf.md#3-invariant-ekf-lets-respect-the-geometry-of-the-problem) puts $`(R, v, p)`$ on one group, $`SE_2(3)`$, and defines all three errors by group multiplication, which removes that dependence for **group-affine** dynamics ([glossary](../glossary.md#group-affine-dynamics-trajectory-independence)). [hexapod_contact_inekf.md](hexapod_contact_inekf.md) compares an invariant EKF with a world-frame error-state EKF on the same data.

---

## 6. What this toy does not show

- **Constant biases.** The true biases are constant; real ones drift. The filters assume a small random walk, which is the usual model.
- **Matched discretization.** The truth uses the filters' own integration step, so integration error is absent. Real IMU data adds it.
- **A position fix only.** No attitude sensor (magnetometer, camera) and no lever arm between the IMU and the antenna.
- **Gaussian noise, no outliers.**

---

## 7. Takeaway

- The ESKF estimates a small error on top of a nominal state, so the filter works on a minimal, unconstrained error that is always near zero, while the nominal state keeps the large rotation exact.
- Here the bias states matter far more than anything else: without them the NEES is about 190 times its consistent value.
- A quaternion in the state works only if the covariance is kept in step with the renormalization; the ESKF never has that problem.

---

## 8. References

1. Solà, J. (2017). *Quaternion kinematics for the error-state Kalman filter*. arXiv:1711.02508. https://arxiv.org/abs/1711.02508 - the ESKF formulation, conventions and reset Jacobian used here.
2. Trawny, N., & Roumeliotis, S. I. (2005). *Indirect Kalman Filter for 3D Attitude Estimation*. Technical Report 2005-002, University of Minnesota. https://www.cse.sc.edu/~yiannisr/774/2023/indirect-kalman-filter-for-3d-attitude-estimation.pdf - the same filter for attitude, in the JPL quaternion convention.
3. Lefferts, E. J., Markley, F. L., & Shuster, M. D. (1982). *Kalman Filtering for Spacecraft Attitude Estimation*. Journal of Guidance, Control, and Dynamics, 5(5), 417-429. https://doi.org/10.2514/3.56190 - the multiplicative EKF, the attitude-only ancestor of the ESKF.
