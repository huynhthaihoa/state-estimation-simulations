# Invariant EKF on $`SE_2(3)`$ through a bounce: the saltation matrix on a Lie group

## Intuition

[hybrid_saltation_ekf.md](hybrid_saltation_ekf.md) tracks a bouncing **point mass**: a plain vector state, where the only subtlety at a bounce is timing - a perturbed trajectory hits the ground a little earlier or later, and the saltation matrix ([glossary](../glossary.md#saltation-matrix)) handles it. This note lifts the same system to a **tumbling rigid body carrying an IMU** ([glossary](../glossary.md#imu-inertial-measurement-unit)), whose state lives on the Lie group ([glossary](../glossary.md#lie-group)) $`SE_2(3)`$, and tracks it with a **right-invariant EKF (RIEKF)**, an invariant EKF ([glossary](../glossary.md#iekf-invariant-ekf)) built on the right-invariant error of §2. Between bounces, the RIEKF's error dynamics do not depend on the trajectory at all (**trajectory independence**), which is why invariant filters are a common choice for IMU-driven robots.

Why $`SE_2(3)`$: the IMU drives the velocity, so the state must hold it next to the pose. Storing it as a second vector column next to the position is what gives the RIEKF its trajectory independence ([lie_algebra.md §11.1](../foundations/lie_algebra.md#111-extended-poses-pose-plus-velocity)).

A bounce breaks trajectory independence. The reset (reverse the vertical velocity, scaled by the restitution coefficient $`e`$) is not a group automorphism (§3), so its effect on the **invariant error** (the filter's error measured on the group, §2) depends on the body's orientation and velocity at impact. Each bounce therefore needs two things right: the **timing correction** (§4: a perturbed body lands earlier or later, as for the point mass) and this **state dependence**. The script compares filters that get neither, one, or both right.

Implemented in [use_numpy/invariant_saltation_ekf.py](../../use_numpy/invariant_saltation_ekf.py) and, with the group math delegated to the `manif` library, [use_manif/invariant_saltation_ekf.py](../../use_manif/invariant_saltation_ekf.py), which orders the 9-vectors as (position, rotation, velocity) instead of the (rotation, velocity, position) used below; tests in [tests/use_numpy/test_invariant_saltation_ekf.py](../../tests/use_numpy/test_invariant_saltation_ekf.py) and [tests/use_manif/test_invariant_saltation_ekf.py](../../tests/use_manif/test_invariant_saltation_ekf.py).

## 1. The system

The body is a **hybrid system**: smooth motion (the flow) interrupted by instantaneous jumps (the bounces), each triggered by a guard and applied by a reset ([glossary](../glossary.md#hybrid-system-guard-and-reset-map)).

State, with $`R`$ body-to-world rotation, $`v`$ velocity, $`p`$ position (an **extended pose**, [glossary](../glossary.md#extended-pose); $`SE_2(3)`$ is the group of such matrices, introduced in [lie_algebra.md §11.1](../foundations/lie_algebra.md#111-extended-poses-pose-plus-velocity)):

```math
X = \begin{bmatrix} R & v & p \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \in SE_2(3)
```

- **Flow** (between bounces), driven by the gyro $`\omega`$ and accelerometer $`a`$ (body frame) and gravity $`g`$: $`\dot R = R\,\omega^\wedge`$, $`\dot v = R\,a + g`$, $`\dot p = v`$, with $`\omega^\wedge`$ the [hat operator](../glossary.md#hat-operator) applied to $`\omega`$. The true body spins at a constant **body rate** (its angular velocity in the body frame, the true $`\omega`$). In free flight the accelerometer reads zero **specific force** (it measures acceleration minus gravity, so a falling body reads zero), so translation is exactly ballistic.
- **Guard**: the ground, $`p_z = 0`$, crossed while descending. The crossing time is the root of a quadratic in closed form (`crossing_time`).
- **Reset**: $`(R, v, p) \mapsto (R,\, M v,\, p)`$ with $`M = I - (1+e)\,n n^\top`$ and $`n = e_z`$, the vertical unit vector - the normal velocity is reflected and scaled by the **coefficient of restitution** $`e`$, the ratio of rebound speed to impact speed. With the default 0.8, a body hitting the ground at 10 m/s vertically rebounds at 8 m/s. The horizontal velocity, rotation, position and body rate are unchanged (`reset_map`).

Sensors: noisy IMU readings every **tick** (one filter time step, one per 10 ms IMU sample), and every few ticks the body-frame positions of four known landmarks, $`y_i = R^\top (\ell_i - p) + \nu`$, where $`\ell_i`$ is landmark $`i`$'s world position and $`\nu`$ the measurement noise, with covariance $`N`$ (here $\nu$ is noise, not the innovation it denotes in the [glossary](../glossary.md#innovation-and-kalman-gain)).

## 2. Why the RIEKF is attractive between bounces

With $`\hat X`$ the filter's estimate, the right-invariant error is $`\eta = \hat X X^{-1}`$, parameterized by $`\xi = \log(\eta) = (\xi_R, \xi_v, \xi_p) \in \mathbb{R}^9`$ (so the truth is $`X = \exp(-\xi)\,\hat X`$; $`\exp`$/$`\log`$ are the group's exponential and logarithm maps, [glossary](../glossary.md#exp-and-log-maps)). This $`\eta`$ is the inverse of the glossary's $`\eta_R = X \hat X^{-1}`$ ([glossary](../glossary.md#invariant-and-equivariant), [left_right_invariant.md](left_right_invariant.md)); both are right-invariant, and this doc follows Barrau & Bonnabel (2017) and Hartley et al. (2020), as does the code. Two exact properties make it the natural filter here (Barrau & Bonnabel 2017; Hartley et al. 2020):

**Propagation is trajectory-independent.** IMU kinematics are [group-affine](../glossary.md#group-affine-dynamics-trajectory-independence). Over a tick of length $`h`$ with constant readings, $`X^+ = \Gamma_h\, f_h(X)\, \Upsilon_h`$ (in this section $`^+`$ means one tick later; from §4 on, just after the impact). Here:
- $`f_h(R,v,p) = (R, v, p + v h)`$ is a **group automorphism**: it respects the group product, $`f_h(X_1 X_2) = f_h(X_1)\,f_h(X_2)`$;
- $`\Gamma_h = (I, g h, \tfrac12 g h^2)`$ carries gravity;
- $`\Upsilon_h`$ carries the body-frame IMU increments (`propagate_mean`).

The error then evolves as $`\eta^+ = \Gamma_h f_h(\eta) \Gamma_h^{-1}`$, independently of $`X`$, and its logarithm evolves *exactly linearly*, even for large errors:

```math
\xi^+ = \Phi(h)\,\xi, \qquad \Phi(h) = e^{A h}, \qquad A = \begin{bmatrix} 0 & 0 & 0 \\ g^\wedge & 0 & 0 \\ 0 & I & 0 \end{bmatrix}
```

$`\Phi(h)`$ is the invariant error's **transition matrix** over one tick, and $`e^{Ah}`$ is the matrix exponential (not the restitution $e$). `propagation_matrix` builds $`\Phi`$; the test `test_invariant_error_propagates_exactly_even_for_large_errors` checks $`\log(\hat X^+ (X^+)^{-1}) = \Phi\,\xi`$ to $`10^{-10}`$ for a finite $`\xi`$. IMU noise enters in the body frame with covariance $`Q_b`$, so it is mapped by the **adjoint** $`\mathrm{Ad}_{\hat X^+}`$ ([glossary](../glossary.md#adjoint)), and the covariance $`P`$ of $`\xi`$ propagates as $`P^+ = \Phi P \Phi^\top + \mathrm{Ad}_{\hat X^+} Q_b \mathrm{Ad}_{\hat X^+}^\top`$.

**The landmark measurement is right-invariant**, i.e. of the form $`X^{-1} b`$ for a known $`b`$, which is what makes its Jacobian constant. Written in homogeneous coordinates (the 3-vector padded with extra entries so that the $`5 \times 5`$ matrix $X$ can act on it), $`y_i = X^{-1} b_i`$ with $`b_i = (\ell_i, 0, 1)`$. The residual $`r_i = \hat R\,y_i + \hat p - \ell_i`$ then satisfies $`r_i \approx \xi_R^\wedge \ell_i + \xi_p`$, i.e.

```math
H_i = \begin{bmatrix} -\ell_i^\wedge & 0 & I \end{bmatrix}
```

This $`H_i`$ is constant, whatever the estimate (`measurement_jacobian_ri`), with noise $`\hat R N \hat R^\top`$. The ordinary world-frame EKF's (the non-invariant baseline of §6) $`H`$ depends on $`\hat R`$ and $`\hat p`$ instead (`measurement_jacobian_world`).

## 3. What the bounce breaks

A map $`\phi`$ preserves this structure only if it is a group automorphism, $`\phi(X_1 X_2) = \phi(X_1)\,\phi(X_2)`$. For the reset, the velocity block of $`X_1 X_2`$ is $`v_1 + R_1 v_2`$; the rotation and position blocks are untouched by the reset, so the velocity block of the difference $`\phi(X_1 X_2) - \phi(X_1)\phi(X_2)`$ is all that remains:

```math
M (v_1 + R_1 v_2) - (M v_1 + R_1 M v_2) = M R_1 v_2 - R_1 M v_2
```

This vanishes for every $`v_2`$ only when $`M R_1 = R_1 M`$, i.e. when $`R_1`$ maps the vertical axis to itself or to its opposite: a rotation about the vertical axis, or a half-turn about a horizontal one. For a tumbling body it is not (`reset_is_automorphism_gap`, the largest entry of $`|\phi(X_1 X_2) - \phi(X_1)\phi(X_2)|`$: about 0.87 for a random pair in the test, exactly 0 for a vertical-axis rotation). Consequence: unlike $`\Phi`$, the Jacobian of a bounce in invariant coordinates depends on the state.

## 4. The saltation matrix on the group

The saltation matrix has its textbook form in coordinates where the state is a plain vector. Use world-perturbation coordinates around a base state, $`\delta = (\delta\theta, \delta v, \delta p)`$ with $`R' = \mathrm{Exp}(\delta\theta) R`$ ($`\mathrm{Exp}`$ maps a rotation vector to a rotation, [glossary](../glossary.md#exp-and-log-maps)), $`v' = v + \delta v`$, $`p' = p + \delta p`$. There the vector field is $`f = (R\omega,\; R a + g,\; v)`$, the guard gradient $`D g`$ picks $`\delta p_z`$, the reset Jacobian is $`DR = \mathrm{blockdiag}(I, M, I)`$ (the block-diagonal matrix with those blocks), and, with $`f^-`$ and $`f^+`$ the vector field just before and just after the impact and $\otimes$ the outer product (Kong et al. 2024, Def. 2, time-invariant guard and reset):

```math
\Xi_{\text{world}} = DR + \frac{\left(f^+ - DR\, f^-\right) \otimes D g}{D g \cdot f^-}
```

Across the reset, only the position-rate component of $`f`$ changes directly ($`v \to M v`$). In $`f^+ - DR f^-`$ the rotation-rate component contributes $`0`$, the velocity-rate component $`(I - M)(R a + g)`$ and the position-rate component $`(M - I)\,v^-`$ (`saltation_matrix_world`).

**Timing correction.** The second term of $`\Xi_{\text{world}}`$. $`DR`$ alone assumes every nearby trajectory bounces at the same instant, but a body that is slightly higher lands slightly later. Worked number, for a vertical drop at 7 m/s with $`e = 0.8`$ and the truth 1 cm above the estimate:
- the truth lands 1 cm / 7 m/s ≈ 1.4 ms later;
- by then the estimate has already bounced and risen 5.6 m/s × 1.4 ms ≈ 0.8 cm;
- so after both have bounced, the truth is about 0.8 cm *below* the estimate: the height error goes from +1 cm to about −0.8 cm ($`-e`$ times it, to first order), and a vertical-velocity error of about $`(1+e)\,g \times 1.4`$ ms ≈ 0.025 m/s appears.

$`DR`$ alone would keep +1 cm. In the fraction, $`D g \cdot \delta \,/\, D g \cdot f^-`$ turns the state error into that shift of the impact time, and $`f^+ - DR\,f^-`$ (how the motion differs just after versus just before the bounce) turns the time shift back into a state error. This timing error is *caused by the state*. §7's **timing term** handles an error in the *detected* impact time.

To move between the two coordinate systems: a right-invariant perturbation $`X' = \exp(\xi) X`$ has, to first order, $`\delta\theta = \xi_R`$, $`\delta v = \xi_v - v^\wedge \xi_R`$, $`\delta p = \xi_p - p^\wedge \xi_R`$, i.e. $`\delta = T(X)\,\xi`$ with

```math
T(X) = \begin{bmatrix} I & 0 & 0 \\ -v^\wedge & I & 0 \\ -p^\wedge & 0 & I \end{bmatrix}
```

(`world_to_invariant_map`). The chain rule, with the two sets of local coordinates based at the pre- and post-impact states $`X^-`$ and $`X^+`$, gives

```math
\Xi_{RI} = T(X^+)^{-1}\,\Xi_{\text{world}}\,T(X^-)
```

(`saltation_matrix_ri`). The two $`T`$ factors do not cancel. Even for the reset Jacobian alone, $`T(X^+)^{-1} DR\, T(X^-)`$ differs from the shortcut $`\mathrm{blockdiag}(I, M, I)`$ in one block: the velocity error picks up $`\big((M v^-)^\wedge - M\,(v^-)^\wedge\big)\,\xi_R`$. That term depends on the impact velocity: its largest entries are $`(1+e)\,|v_z^-|`$, 12.6 for a 7 m/s vertical impact speed. It is nonzero even when the velocity is purely vertical. This is the state dependence of §3, made explicit.

Three bounce treatments are compared in the RIEKF, identical in everything else:

| Treatment | Covariance at the bounce | Timing correction | State dependence |
|---|---|---|---|
| invariant-naive | $`\mathrm{blockdiag}(I, M, I)`$ | no | no |
| reset-Jacobian | $`T(X^+)^{-1} DR\, T(X^-)`$ | no | yes |
| saltation | $`\Xi_{RI}`$ | yes | yes |

The invariant-naive choice is the tempting one: it applies the reset to $`\xi_v`$ as if the reset were an automorphism.

## 5. Verification against the exact hybrid map

The ground truth for "the right Jacobian" is the exact **fixed-time hybrid map**: start above the ground, flow to the crossing, reset, flow to a fixed later time $`T`$ (a time, not the map $`T(X)`$ of §4), with every perturbed trajectory using the same IMU inputs. Call this map $`\Psi`$. Its Jacobian in invariant coordinates, by central finite differences of $`\log(\Psi(\exp(\xi) X_0)\,\Psi(X_0)^{-1})`$, is compared with $`\Phi(T - \tau)\, \Xi\, \Phi(\tau)`$, where $`\tau`$ is the crossing time and $`\Xi`$ each treatment's bounce matrix (`test_saltation_matrix_ri_matches_finite_differences`, three random seeds):

| Treatment | max abs. difference from finite differences |
|---|---|
| saltation | $`4 \times 10^{-10}`$ to $`1.5 \times 10^{-9}`$ |
| reset-Jacobian | 3.9 to 4.3 |
| invariant-naive | 8.2 |

The saltation matrix is exact up to the finite-difference step; the other two are wrong by order one. The landmark Jacobians are checked the same way.

## 6. Results: consistency through the bounces

Defaults: 5 m drop, restitution 0.8, body rate $`(1.0, -0.6, 2.0)`$ rad/s, IMU at 100 Hz (gyro noise 0.02 rad/s, accel noise 0.3 m/s² per sample), landmarks at 10 Hz (0.05 m noise), four true bounces at 1.01, 2.63, 3.92 and 4.96 s. A fourth filter is the non-invariant baseline: an error-state EKF ([glossary](../glossary.md#eskf-error-state-kalman-filter)) in world-perturbation coordinates with $`\Xi_{\text{world}}`$ at bounces. (The fifth filter of §7, saltation + timing noise, is identical to the saltation one with exact contact detection, so it is left out of this section's tables.)

**Mode matching.** At one or two ticks around each impact, the truth has bounced and the estimate has not yet (or vice versa); there the velocity error is of order $`(1+e)|v_z|`$ - tens of m/s - and **NEES** ([glossary](../glossary.md#consistency-and-nees)) reaches about $`10^5`$ for every filter alike. That is an artifact of comparing a hybrid estimate with a hybrid truth at fixed ticks, not of the covariance treatment ([hybrid_saltation_ekf.md](hybrid_saltation_ekf.md) reports the same shared spike). `run_monte_carlo_consistency` therefore scores a tick only when the filter has resolved the same number of bounces as the truth, and reports the excluded fraction: 0.14-0.16% of (trial, tick) pairs here, for all four filters.

Monte Carlo average NEES over mode-matched ticks, 200 trials, seed 0 (a consistent 9-DoF ([glossary](../glossary.md#degrees-of-freedom-dof)) filter averages 9; the 95% interval for a 200-trial average is [8.42, 9.60]):

| Filter | Before the 1st bounce | 10 ticks after each bounce | Overall |
|---|---|---|---|
| RIEKF, invariant-naive | 9.34 | **13.45** | 10.03 |
| RIEKF, reset-Jacobian | 9.34 | 9.33 | 9.11 |
| RIEKF, saltation | 9.34 | 8.89 | 9.08 |
| World-frame EKF, saltation | **14.01** | 9.35 | 10.18 |

With landmarks only every 25 ticks (`--meas-every 25`), so the post-bounce covariance has to carry the filter longer:

| Filter | Before the 1st bounce | 25 ticks after each bounce | Overall |
|---|---|---|---|
| RIEKF, invariant-naive | 10.22 | **16.44** | 11.84 |
| RIEKF, reset-Jacobian | 10.22 | 10.36 | 9.69 |
| RIEKF, saltation | 10.22 | 9.37 | 9.34 |
| World-frame EKF, saltation | **18.92** | 12.64 | 12.57 |

What this shows:
- **The invariant-naive shortcut is clearly overconfident after bounces** (NEES 1.5-1.8× the consistent value), and recovers only slowly, over several measurement updates.
- **Reset-Jacobian vs. saltation is a modest gap**, as for the point mass: both are close to consistent, and in the sparse-measurement case only the saltation-corrected filter's after-bounce NEES stays inside the 95% interval (before the first bounce, all three RIEKFs sit slightly above it, at 10.22). Getting the state dependence right matters more here than the timing correction.
- **The world-frame EKF is inconsistent before any bounce** (NEES 14-19), most likely because its propagation and measurement Jacobians are evaluated at a poor initial estimate, which the invariant filter's constant $`\Phi`$ and $`H`$ avoid - the usual invariant-filter advantage, independent of the hybrid event (not isolated further here).
- **RMSEs** ([glossary](../glossary.md#rmse-root-mean-square-error)) **are nearly identical** across the three RIEKFs (e.g. 1.29° / 0.12 m/s / 0.11 m in the single default run): as in [hybrid_saltation_ekf.md](hybrid_saltation_ekf.md), the effect is in whether the reported uncertainty can be trusted, not in the point estimate.

![Three panels from invariant_saltation_ekf.py: height of the bouncing body for the ground truth and the filters (the dashed saltation + timing noise line coincides with saltation here); single-run position error; Monte Carlo average NEES over time with the consistent value of 9 and its 95% interval, the invariant-naive filter spiking after each bounce and the world-frame EKF elevated before the first bounce](../../assets/invariant_saltation_ekf.png)

The same run, animated. The top panel shows the tumbling body bouncing among the four landmarks, with a sight line to each landmark whenever a measurement arrives. All four position estimates stay within a few centimeters of the truth, so their markers overlap. The difference is in the bottom panel: the invariant-naive filter's NEES jumps to about 18 at the first bounce and needs about 1 s to fall back inside the 95% band (about 0.4 s after the next two bounces, whose spikes are smaller). No single state component's standard deviation is plotted, because none shows the problem. The naive filter's mistake is the missing rotation-velocity coupling of §4. Its horizontal-velocity standard deviation after a bounce is actually *larger* than the others' (0.067 vs. 0.049 m/s just after the first one).

![Animation of the default run: top, a 3D view of the tumbling box bouncing on the ground among four landmarks, with dashed sight lines at each landmark measurement and overlapping position-estimate markers; bottom, the Monte Carlo average NEES of the four filters drawn up to the current time, with the invariant-naive filter spiking after each bounce and the world-frame EKF elevated before the first one](../../assets/invariant_saltation_ekf.gif)

## 7. Contact-detection jitter

Everything above assumes each filter applies the bounce exactly when its own estimate reaches the ground. A real contact detector (an accelerometer spike, as in [Čížek et al., 2018](hybrid_saltation_ekf.md#11-references), or a force threshold) adds its own latency and jitter. `--detect-time-bias`/`--detect-time-noise-std` model this as in [hybrid_saltation_ekf.md §8](hybrid_saltation_ekf.md#8-quantifying-contact-detection-timing-jitter). A bounce found at $`\tau`$ inside a tick is applied at $`\tau_{\text{detect}}`$ instead, with detection bias $`b_{\text{detect}}`$ and jitter standard deviation $`\sigma_{\text{detect}}`$ (the two flags):

```math
\tau_{\text{detect}} = \text{clip}\big(\tau + b_{\text{detect}} + \mathcal{N}(0,\,\sigma_{\text{detect}}^2)\,,\ 0,\ \tau_{\text{remain}}\big)
```

(`detected_crossing_time`; $`\tau_{\text{remain}}`$ is the time left in the tick). Mean and covariance are both reset there, so a late detection resets slightly below the ground and an early one slightly above it. Each filter draws its jitter from its own random generator, spawned from the run's seed, so a sweep at a fixed seed changes only the jitter: the IMU noise, measurements and initial errors stay the same. The descending-crossings-only rule of `crossing_time` keeps a below-ground reset from triggering a spurious second bounce.

**The timing term.** The saltation matrix accounts for how a *state* error moves the impact time, but not for an error in the detected time itself. Resetting $`\delta`$ seconds late (here $`\delta`$ is a time, not the perturbation of §4) means flowing with $`f^-`$ past the guard and then resetting, instead of resetting and flowing with $`f^+`$. To first order the post-impact state is then off by $`-w\,\delta`$, with

```math
w = T(X^+)^{-1}\left(f^+ - DR\, f^-\right)
```

in invariant coordinates (`impact_timing_direction_ri`; checked against a finite difference of the hybrid map with respect to the reset time, `test_impact_timing_direction_matches_finite_difference_of_a_late_reset`). It is the same vector the saltation correction of §4 is built from. A timing error with RMSE $`s`$ therefore adds

```math
P^+ \leftarrow \Xi_{RI}\, P^-\, \Xi_{RI}^\top + s^2\, w\, w^\top
```

at each bounce. That is the fifth filter, "saltation + timing noise" (reset mode `saltation_timing`), here tuned to the true detector: $`s = \sqrt{b_{\text{detect}}^2 + \sigma_{\text{detect}}^2}`$. With exact detection, $`s = 0`$ and it is the saltation filter.

**Sweep.** Mean NEES over the 10 ticks after each bounce, mode-matched, 200 trials per seed, averaged over seeds 0-3 (every seed is within 6% of the 4-seed mean in every cell; a consistent filter averages 9, and the 95% interval for one 200-trial average is [8.42, 9.60]):

| Detection error | RIEKF, invariant-naive | RIEKF, reset-Jacobian | RIEKF, saltation | RIEKF, saltation + timing noise | World-frame EKF, saltation |
|---|---|---|---|---|---|
| exact (as in §6, but a 4-seed mean) | 13.98 | 9.56 | 9.03 | 9.03 | 9.54 |
| $`\sigma_{\text{detect}} = 1`$ ms | 14.92 | 10.49 | 10.60 | **9.01** | 11.34 |
| $`\sigma_{\text{detect}} = 2`$ ms | 17.31 | 12.86 | 14.58 | **8.92** | 15.71 |
| $`\sigma_{\text{detect}} = 5`$ ms | 26.92 | 22.55 | 30.99 | **8.68** | 32.86 |
| $`b_{\text{detect}} = 2`$ ms, no jitter | 17.78 | 13.34 | 15.92 | **8.95** | 16.05 |

A tick is 10 ms, so 5 ms is half a tick. NEES before the first bounce does not change with the detection error (9.50 for the RIEKFs and 14.54 for the world-frame EKF, 4-seed means), and 0.14-0.21% of (trial, tick) pairs are excluded as mode-mismatched in every run.

![Three panels from invariant_saltation_ekf.py with 5 ms contact-detection jitter: height of the bouncing body; single-run position error; Monte Carlo average NEES over time, where every filter except saltation + timing noise jumps to between about 18 and 45 after each bounce and decays until the next one, while saltation + timing noise stays near 9](../../assets/invariant_saltation_ekf_jitter.png)

What this shows:
- **Without the timing term, every filter becomes overconfident after bounces**, more so as the jitter grows. That is expected: none of their covariances contains the detection error.
- **The point-mass ordering carries over.** From 2 ms of jitter on, the saltation filter is *more* overconfident than the reset-Jacobian one, in every seed (14.58 vs. 12.86 at 2 ms, 30.99 vs. 22.55 at 5 ms); at 1 ms the two are tied, with the seeds split 2-2. [hybrid_saltation_ekf.md §8](hybrid_saltation_ekf.md#8-quantifying-contact-detection-timing-jitter) found the same for the point mass and gave a likely reason: the saltation update commits more strongly to the detected impact time.
- **The timing term restores consistency at every setting tried**, inside the 95% interval in every seed, including the pure 2 ms delay. At 5 ms it is slightly conservative (8.68). One possible reason, not isolated here: clipping $`\tau_{\text{detect}}`$ into the tick cuts off the largest timing errors, so the actual timing variance is a little below $`s^2`$.
- **The bias row needs care.** The timing term only matches the second moment (the mean square) of the timing error. A known systematic delay is better subtracted from the detected time; the term is for the part that cannot be predicted.
- **Accuracy does not change on average.** Single-run position RMSE is within about 2% across the RIEKFs at every setting. The single-run velocity RMSE is dominated by the one or two mode-mismatched ticks around each impact, so it does not separate the filters.

## 8. Scope and caveats

- **No IMU biases** ([glossary](../glossary.md#bias)). Biases break the exact group-affine property: an invariant EKF that carries them anyway is called "imperfect", because its error dynamics then depend on the estimate again ([`hexapod_contact_inekf.md` §5](hexapod_contact_inekf.md#5-imu-biases-the-imperfect-invariant-ekf) runs one); leaving them out keeps the comparison about the bounce.
- **The impact changes velocity only.** A real impact also changes the angular velocity; here the body rate is an input, continuous across the bounce.
- **Contact detection is a timing error only** (§7), clipped into the current tick, and the timing term is given the true detector's $`s`$. A real filter would have to estimate $`s`$, and a detector can also miss an impact or fire without one, which is not modeled.
- **Short runs only.** `step_hybrid_ri` raises if an impact is slower than `min_bounce_speed` rather than modeling the settling phase, where the bounces get ever shorter (**Zeno** behavior, [glossary](../glossary.md#zeno-behavior)).
- **Few numerical settings.** §6 shows the defaults and one variation, and §7 sweeps only the detection error. The qualitative pattern held in the runs tried, but the other parameters are not swept.

## 9. References

1. Barrau, A., & Bonnabel, S. (2017). *The Invariant Extended Kalman Filter as a Stable Observer*. IEEE Transactions on Automatic Control, 62(4), 1797–1812. https://doi.org/10.1109/TAC.2016.2594085 (preprint: https://arxiv.org/abs/1410.1465) - group-affine systems, invariant observations and the trajectory-independent error dynamics used in §2.
2. Hartley, R., Ghaffari, M., Eustice, R. M., & Grizzle, J. W. (2020). *Contact-Aided Invariant Extended Kalman Filtering for Robot State Estimation*. The International Journal of Robotics Research, 39(4), 402–430. https://doi.org/10.1177/0278364919894385 - the right-invariant IMU filter on $`SE_2(3)`$ / $`SE_{K+2}(3)`$ (with $K$ contact points added to the state) that §2's propagation and measurement model follow. The $`SE_{K+2}(3)`$ version is implemented for a hexapod in [`hexapod_contact_inekf.md`](hexapod_contact_inekf.md).
3. Kong, N. J., Payne, J. J., Zhu, J., & Johnson, A. M. (2024). *Saltation Matrices: The Essential Tool for Linearizing Hybrid Dynamical Systems*. Proceedings of the IEEE, 112(6), 585–608. https://doi.org/10.1109/JPROC.2024.3440211 - the saltation-matrix definition used in §4.
