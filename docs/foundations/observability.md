# Observability: what no sensor can see, and why a standard EKF forgets it

## Intuition

Take a robot and the landmarks it measures, and move everything by 1 m. The robot is still in the same place relative to every landmark, so no measurement changes. Rotate everything by 90° about the origin and the same holds:

| | robot position | robot heading | landmark | measured (landmark in the robot's frame) |
| --- | --- | --- | --- | --- |
| original | (0, 0) | 0° | (3, 4) | (3, 4) |
| shifted by (1, 0) | (1, 0) | 0° | (4, 4) | (3, 4) |
| rotated by 90° | (0, 0) | 90° | (−4, 3) | (3, 4) |

So a robot that only measures landmarks relative to itself can never learn where the world frame's origin is or how it is turned: 2 shift directions and 1 rotation direction are **unobservable** (no sequence of measurements can reveal them; [glossary](../glossary.md#observability)). An honest filter keeps its uncertainty along those directions as wide as it started.

An [EKF](../glossary.md#ekf-extended-kalman-filter) (extended Kalman filter) evaluates its Jacobians at its own latest estimates, which change at every step. This doc shows, with one small simulation, that this makes the filter's linearized model believe it can see the rotation. It then gains information that does not exist and turns **overconfident**: its covariance claims less error than it really has ([consistency](../glossary.md#consistency-and-nees)). Two fixes are compared:

- **FEJ** (first-estimates Jacobian, [glossary](../glossary.md#fej-first-estimate-jacobians)): evaluate the Jacobians at the first estimates instead of the latest.
- **Invariant EKF** ([glossary](../glossary.md#iekf-invariant-ekf)): choose an error variable in which the unobservable directions are the same at every estimate.

Result of §5 in one line: the standard EKF's linearized model has a 2-dimensional unobservable subspace instead of 3 and adds information of about $`9\times10^{4}`$ along the rotation over a run (as if the world's rotation had been measured to ±0.19°), while FEJ and the invariant EKF add 0 (to 1e-25). The effect on the robot pose NEES is smaller (3.95 against 3.05 for the ideal EKF) and depends on the map and the noise (§5.4).

## 1. Observable and unobservable directions

The filter works with the **error state** $`e`$, the difference between the true state and the estimate. Its linearized model is

- propagation: $`e_k = \Phi_k\, e_{k-1}`$, with the **transition matrix** $`\Phi_k`$ (the Jacobian of the motion model with respect to the state) at step $`k`$;
- measurement: residual $`r_k = H_k\, e_k`$, with the **measurement Jacobian** $`H_k`$ (the Jacobian of the measurement function with respect to the state).

Stack what a window of steps $`k_0 \dots k_1`$ measures about the error at its start, the **local observability matrix** (ref. 1, eq. 20):

```math
M = \begin{bmatrix} H_{k_0} \\ H_{k_0+1}\Phi_{k_0+1} \\ H_{k_0+2}\Phi_{k_0+2}\Phi_{k_0+1} \\ \vdots \end{bmatrix}
```

A direction $`v`$ (a vector of the state's size) is **unobservable** over the window if $`M v = 0`$: moving the state along $`v`$ changes no residual, to first order. The directions with $`Mv = 0`$ form the **null space** of $`M`$ ([glossary](../glossary.md#null-space)); the number of unobservable directions is its dimension. The matrix is "local" because it depends on the points where each $`\Phi_k`$ and $`H_k`$ were evaluated. For a time-invariant system, that point does not matter. For an EKF, it does, and that is §3.

**Worked case.** A 1D robot at $`x`$ measures its distance to a landmark at $`p`$: $`z = p - x`$. The state is $`(x, p)`$, $`H = [-1,\ 1]`$ and $`\Phi = I`$ (the robot moves by a known odometry step; the landmark does not move). Every row of $`M`$ is $`[-1,\ 1]`$, so $`M`$ has rank 1 and its null space is spanned by $`v = (1, 1)`$: shift both. Of the 2 state directions, 1 is unobservable.

**Why a filter cares.** The Kalman update adds $`H^\top R^{-1} H`$ to the **information matrix** $`P^{-1}`$ (the inverse of the covariance $`P`$; $`R`$ is the measurement noise covariance; [glossary](../glossary.md#information-matrix)). The information added along $`v`$ is $`v^\top H^\top R^{-1} H\, v`$, which is 0 exactly when $`Hv = 0`$. If the filter's $`H`$ (or the $`v`$ it carries through its own $`\Phi`$) does not satisfy that, the covariance shrinks along a direction nobody measures.

## 2. The planar SLAM toy

The model is the unicycle EKF-SLAM of Barrau & Bonnabel (ref. 3, eqs. 9 and 11) with the scenario of their Sec. 7.1, which they chose to resemble the one in Huang et al.'s journal paper (not read here); ref. 2 describes a similar one (10 laps of a circle, 20 landmarks, 5 m sensing range).

| Symbol | Meaning | Units |
| --- | --- | --- |
| $`x_k`$ | robot position (2-vector) at step $`k`$ | m |
| $`\theta_k`$ | robot heading | rad |
| $`p_j`$ | position of landmark $`j`$ (static, unknown), $`j = 1 \dots K`$ | m |
| $`X`$ | state $`(x, \theta, p_1, \dots, p_K)`$, size $`3 + 2K`$ | |
| $`d_k,\ w_k`$ | odometry readings of step $`k`$: forward displacement, rotation | m, rad |
| $`R(\theta)`$ | rotation matrix of angle $`\theta`$ | |
| $`J`$ | turns a 2-vector by 90°: $`J(a, b) = (-b, a)`$ | |
| $`\hat X_{k\mid k-1}`$, $`\hat X_{k\mid k}`$ | estimate after the propagation of step $`k`$ (before its update) and after the update | |

```math
x_k = x_{k-1} + R(\theta_{k-1})\begin{bmatrix} d_k \\ 0 \end{bmatrix},\qquad
\theta_k = \theta_{k-1} + w_k,\qquad
z_{k,j} = \tilde h\big(R(\theta_k)^\top (p_j - x_k)\big) + \text{noise}
```

The function $`\tilde h`$ is the identity (the landmark's position in the robot frame, the default of both papers) or range and bearing (`--meas range_bearing`). The world frame is fixed only by the initial pose (known exactly, as in the papers), so the 3 unobservable directions of the system are

- two shifts, $`n_1 = (1,0,\ 0,\ 1,0,\ \dots,\ 1,0)`$ and $`n_2 = (0,1,\ 0,\ 0,1,\ \dots,\ 0,1)`$: add the same vector to the robot and every landmark;
- one rotation, $`n_3 = (Jx,\ 1,\ Jp_1,\ \dots,\ Jp_K)`$: turn the heading by 1 rad and rotate every position about the origin (first order, ref. 1 eq. 18 and ref. 3 Prop. 1).

The worked line for $`n_3`$: the measurement Jacobian of landmark $`j`$ is $`H_j = \nabla\tilde h\,R(\theta)^\top\,[\,-I,\ -J(p_j - x),\ I \text{ at } p_j\,]`$, with $`\nabla\tilde h`$ the Jacobian of $`\tilde h`$. Applied to $`n_3`$ it gives $`\nabla\tilde h\,R^\top(-Jx - J(p_j - x) + Jp_j) = 0`$.

**Truth and noise.** The true path is the model's own discrete circle (speed 1 m/s, turn rate 9°/s, steps of 1 s: a 40-gon of 12.7 m diameter) and the landmark map is one fixed random draw. Each [Monte Carlo trial](../glossary.md#monte-carlo-trials) redraws the odometry noise and the measurement noise, so the filter's errors come from the noise it models. Every filter in a trial processes the same data.

| Parameter | Value | Source |
| --- | --- | --- |
| speed, turn rate | 1 m/s, 9°/s | ref. 3 Sec. 7.1 |
| laps, landmarks | 10 (400 steps), 20 | ref. 3 Sec. 7.1 |
| odometry noise | each wheel 2% of the speed; forward $`\sigma/\sqrt2`$ = 1.41 cm per step, rotation $`\sqrt2\sigma/a`$ = 3.24° per step, wheel base $`a`$ = 0.5 m | ref. 3 Sec. 7.1 |
| measurement | relative position, noise 0.1 m per axis, landmark seen if within 5 m of the true robot, once per second | ref. 3 Sec. 7.1 |
| initial pose uncertainty | zero | ref. 3 Sec. 7.1 |
| new landmark | initialized from the first measurement with a prior std of 100 m ("very high compared to the size of the map") and then updated like the others | ref. 3 Sec. 7.1; the 100 m is a design choice |
| step length 1 s | the paper gives only the 1 Hz measurement rate | design choice |
| landmark map | 20 points drawn uniformly in an annulus of half-width 4 m around the path (fixed per seed) | design choice: the paper shows the map only in its Fig. 1 |
| range-bearing variant | range noise 0.1 m, bearing noise 1° | design choice, labeled variant |
| lateral odometry noise | none (a unicycle slips sideways by 0) | design choice |

The paper calls the loop "15 m in diameter", but 1 m/s at 9°/s is a circle of 12.7 m diameter; this repo follows the two speeds.

## 3. Why the standard EKF loses the rotation direction

Only the heading column of $`\Phi_k`$ depends on the estimate. For the model above, it is the effect of the heading on the position:

```math
\Phi_k = \begin{bmatrix} I_2 & J\,(\hat x_{k\mid k-1} - \hat x_{k-1\mid k-1}) & 0 \\ 0 & 1 & 0 \\ 0 & 0 & I \end{bmatrix}
```

(the Jacobian of $`R(\theta)[d,0]^\top`$ with respect to $`\theta`$ is $`JR(\theta)[d,0]^\top`$, the estimated displacement turned by 90°).

Carry the rotation direction $`n_3`$ through a run of these matrices. Its position part starts as $`Jx_0`$ and gains $`J(\hat x_{j\mid j-1} - \hat x_{j-1\mid j-1})`$ per step. Each term starts from the *updated* previous estimate, so it leaves out the correction $`c_{j-1} = \hat x_{j-1\mid j-1} - \hat x_{j-1\mid j-2}`$ that the update moved the position by. The terms would add up to $`J\hat x_{k\mid k-1}`$ without the corrections, so after $`k`$ steps the carried direction is $`\Big(J\big(\hat x_{k\mid k-1} - \textstyle\sum_{j<k} c_j\big),\ 1,\ \dots\Big)`$ and not $`(J\hat x_{k\mid k-1}, 1, \dots)`$, the rotation direction at the current estimate, which $`H_k`$ does annihilate (§2). The difference is made of update corrections (ref. 1 eq. 27 shows the same in closed form, including the changes of the landmark estimates). Any nonzero correction makes $`H_k n_3 \neq 0`$, so the rotation looks observable. Ref. 1 proves that this lowers the unobservable subspace of the standard EKF from 3 to 2 dimensions, and that the missing direction is the global rotation (eq. 28; ref. 2 states the dimension as its eq. 8).

What the code measures on a small scene (3 landmarks always visible, 10 steps, one noisy trial; `observability_demo`):

| Linearization | null-space dimension | $`\lVert M n_3\rVert / \lVert n_3\rVert`$ | $`\lVert M n_{1,2}\rVert`$ | 3rd smallest singular value / largest |
| --- | --- | --- | --- | --- |
| true state (ideal EKF) | 3 | 3.0e-16 | 0 | 4e-17 |
| standard EKF | **2** | **6.6e-2** | 0 | **1.4e-3** |
| FEJ-EKF | 3 | 3.7e-16 | 0 | 8e-17 |
| invariant EKF (in its $`\xi`$ coordinates, §4) | 3 | 0 | 0 | 1e-16 |
| FEJ, propagation rule only | 2 | 6.2e-2 | 0 | 1.3e-3 |
| FEJ, measurement rule only | 2 | 6.5e-3 | 0 | 2.4e-4 |

The null-space dimension counts singular values below $`10^{-9}`$ of the largest; the translations stay unobservable in every filter, and only the rotation is lost. The third singular value of the standard EKF is small but not zero: the rotation is weakly "seen". Run: `uv run python use_numpy/observability_fej.py` (printed first).

## 4. Two fixes

### 4.1 FEJ

Two changes to the Jacobians (ref. 1 eqs. 29 and 30), both needed:

1. **Propagation:** compute $`\Phi_k`$ with $`\hat x_{k\mid k-1} - \hat x_{k-1\mid k-2}`$, the earlier predicted position, instead of the earlier updated one. The corrections then cancel out in the carried direction.
2. **Measurement:** in $`H_j`$'s term $`J(p_j - x)`$, use the landmark estimate from the first time it was seen, $`\hat p_j^{\text{first}}`$. The robot position there is the predicted one, as in the standard EKF.

With these, the carried direction is $`(J\hat x_{k\mid k-1}, 1, J\hat p^{\text{first}}_j)`$ at every step and $`H_k`$ annihilates it. The Jacobians are less accurate than the standard ones, which use the latest estimates (ref. 1 says so too).

The code also runs each rule alone: with only the propagation rule or only the measurement rule, the null space is 2-dimensional again (§3 table), and the information along the rotation stays large (§5.1).

### 4.2 Invariant EKF

Barrau & Bonnabel (ref. 3) change the error variable instead of the evaluation point:

```math
\xi = \begin{bmatrix} \theta - \hat\theta \\ x - \hat x - (\theta - \hat\theta)\,J\hat x \\ p_j - \hat p_j - (\theta - \hat\theta)\,J\hat p_j \end{bmatrix}
```

(ref. 3 eq. 16; it is the first-order form of the **right-invariant error** $`\chi\hat\chi^{-1}`$ on the group $`SE_{K+1}(2)`$ of rotations and translations that moves the robot and all landmarks together, [glossary](../glossary.md#iekf-invariant-ekf)). In these coordinates:

- the transition matrix is the identity, so a direction carried through the propagation does not change;
- $`H_j = \nabla\tilde h\,R(\hat\theta)^\top[\,-I,\ 0,\ I \text{ at } p_j\,]`$ has no heading column, so $`H_j`$ annihilates the rotation $`(1, 0, 0, \dots)`$ and both shifts $`(0, u, u, \dots)`$ **at any estimate** (ref. 3 Thm. 1, Prop. 3);
- the information along these directions cannot increase (ref. 3 Thm. 2);
- the update is applied to the group: the heading adds $`\delta\theta`$ and every position $`q`$ becomes $`R(\delta\theta)q + B(\delta\theta)\delta q`$ (ref. 3 eq. 22).

Their $`G_n`$ (the Jacobian of the error with respect to the odometry noise; eqs. 17 and 21) corresponds to the noise entering before the input (their eq. 26). This repo uses $`\hat x_{n\mid n-1}`$ in it, the exact first-order $`G`$ for the process model (9) with the error (16), so that all four filters model the same odometry noise (finite-difference check in `tests/use_numpy/test_observability_fej.py`).

### 4.3 Comparison

| Filter | Jacobians evaluated at | Error variable | Realizable | Unobservable subspace of its linear model | Time per step (this toy) |
| --- | --- | --- | --- | --- | --- |
| standard | latest estimates | $`X - \hat X`$ | yes | 2 | 0.49 ms |
| FEJ | propagated robot position, first landmark estimates | $`X - \hat X`$ | yes | 3 | 0.50 ms |
| invariant | latest estimates (they do not matter for the rotation) | $`\xi`$ above | yes | 3 | 0.73 ms |
| ideal (reference) | the true state | $`X - \hat X`$ | no | 3 | 0.58 ms |

Times are from the default run on a laptop shared with other jobs, so only the ratio (the invariant EKF about 1.5 times the others; not profiled) is meaningful. All four use the same prediction of the state and are scored in the same error $`X - \hat X`$ (the invariant EKF's covariance is mapped back with the first-order map $`\xi \mapsto X - \hat X`$, so its NEES is comparable).

## 5. Simulation results

Everything below is one run of `uv run python use_numpy/observability_fej.py` (seed 0, 50 trials, 6 filters, about 80 s) unless another command is given. NEES is the **normalized estimation error squared** ([glossary](../glossary.md#consistency-and-nees)); for the robot pose (3 DoF), a consistent filter averages 3, and the 95% band of an average over 50 trials is [2.36, 3.72].

### 5.1 Observability and information

The null-space table is in §3. The same run measures the information each filter adds along the global rotation: $`\sum_k n_3^\top H_k^\top R^{-1} H_k\, n_3`$ with $`n_3`$ carried through the filter's own $`\Phi`$ (mean over trials; for the invariant EKF the direction is $`(1,0,0,\dots)`$ in $`\xi`$). "Equivalent std" is the standard deviation of a rotation of the whole world about the origin that this information alone would imply, $`1/\sqrt{\text{information}}`$.

| Filter | information along the rotation | equivalent std |
| --- | --- | --- |
| standard EKF | 9.3e4 | 0.19° |
| FEJ, propagation rule only | 5.2e4 | 0.25° |
| FEJ, measurement rule only | 9.8e4 | 0.18° |
| FEJ-EKF | 1.9e-25 | none |
| invariant EKF | 0 | none |
| ideal EKF | 1.4e-25 | none |

### 5.2 Consistency and accuracy

| Filter | robot NEES, all steps | last lap | steps above the 95% band | median trial / max trial | landmark NEES (2 DoF; consistent: 2) |
| --- | --- | --- | --- | --- | --- |
| standard EKF | **3.95** | 4.07 | 69% | 2.78 / 17.45 | **3.05** |
| FEJ-EKF | 3.18 | 3.13 | 5% | 2.76 / 9.00 | 2.11 |
| invariant EKF | 3.06 | 3.04 | 2% | 2.63 / 9.08 | 2.07 |
| ideal EKF | 3.05 | 3.04 | 2% | 2.67 / 8.89 | 2.04 |
| FEJ, propagation rule only | 3.81 | 3.91 | 63% | 3.09 / 13.88 | 2.87 |
| FEJ, measurement rule only | 3.40 | 3.35 | 21% | 2.79 / 12.95 | 2.33 |

(NEES is undefined at steps 0 and 1, where the robot covariance is singular; they are left out. "median trial / max trial" are the per-trial time-averaged NEES over the 50 trials.)

Paired difference of the per-trial time-averaged robot NEES, standard minus the other filter (mean ± standard error over the 50 trials): FEJ 0.77 ± 0.37, invariant 0.89 ± 0.39, ideal 0.89 ± 0.38.

| Filter (last lap, RMS over steps and trials) | position RMSE (m) | heading RMSE (°) | reported heading std (°) | landmark RMSE (m) |
| --- | --- | --- | --- | --- |
| standard EKF | 0.148 | 1.34 | 1.13 | 0.116 |
| FEJ-EKF | 0.114 | 1.21 | 1.15 | 0.085 |
| invariant EKF | 0.113 | 1.19 | 1.15 | 0.085 |
| ideal EKF | 0.113 | 1.19 | 1.15 | 0.085 |

- **The mean NEES of the standard EKF is raised by a few bad trials.** Its median trial (2.78) is as good as the others; its worst trial is 17.45, compared with 9.00 for FEJ. FEJ, the invariant EKF, and the ideal EKF also have bad trials (max 8.9 to 9.1), so the evidence is the mean over trials and the landmark NEES, not one trial.
- **It is the combination that matters.** Each FEJ rule alone leaves most of the excess (NEES 3.81 and 3.40, against 3.95 for the standard and 3.05 for the ideal EKF) and the information along the rotation large; both together match the ideal EKF.
- **Reported against actual heading error.** The standard EKF reports 1.13° but is off by 1.34° RMS (1.19 times its claim); the others report 1.15° against 1.19–1.21°.

### 5.3 Figure and animation

![Four panels: the circular path with the 20 landmarks and trial-0 estimates of three filters; robot pose NEES against time for four filters with the 95% band; the heading error of one trial with the reported 3-sigma envelopes; the information each filter adds along the global rotation, growing to about 5e4 for the standard EKF and 0 for the others](../../assets/observability_fej.png)

*Figure: `uv run python use_numpy/observability_fej.py --out assets/observability_fej.png` (seed 0, 50 trials). Grey lines: 3 and the 95% band of a 50-trial average. The information panel shows trial 0 only (about 5e4 at the end); the 9.3e4 of §5.1 is the mean over the 50 trials.*

![Animation: three scene panels (standard EKF, FEJ-EKF, invariant EKF) with the robot driving the circle, the landmark estimates and their 99% ellipses, and below them the NEES, the heading error against its reported envelope and the information along the rotation, growing with time](../../assets/observability_fej.gif)

*Animation: `uv run python generator/animate_observability_fej.py --out assets/observability_fej.gif` (seed 0, 50 trials; the displayed trial is the Monte Carlo trial 0 of the run above; landmark ellipses drawn 10 times larger, black circle = true position).*

### 5.4 Other maps, measurement, noise

Each row is one run of the default command with the flag shown (50 trials; the sweep 30 trials). Robot pose NEES over all steps, mean over trials with the median over trials of each trial's time average in brackets; consistent value 3, 95% band [2.36, 3.72] (sweep: [2.19, 3.94]). The last column is the information the standard EKF adds along the rotation (the other filters add 0).

| Run | standard | FEJ | invariant | ideal | information, standard |
| --- | --- | --- | --- | --- | --- |
| default (seed 0) | 3.95 (2.78) | 3.18 (2.76) | 3.06 (2.63) | 3.05 (2.67) | 9.3e4 |
| `--seed 1` (new map and noise) | 3.45 (2.83) | 3.35 (2.77) | 3.29 (2.73) | 3.25 (2.74) | 7.8e4 |
| `--seed 2` | 3.46 (2.86) | 3.27 (2.89) | 3.20 (2.81) | 3.36 (2.97) | 6.4e4 |
| `--meas range_bearing` | 2.95 (2.50) | 2.95 (2.66) | 2.80 (2.43) | 2.79 (2.44) | 2.7e5 |
| `--n-landmarks 12` (sparser map) | **5.40** (2.76) | 3.39 (2.85) | 3.04 (2.50) | 3.03 (2.48) | 1.5e5 |

Sweep of the odometry rotation noise (`--sweep-heading-noise 0.5 1 2 4`, multiples of the default 3.24° per step):

| Rotation noise per step | standard | FEJ | invariant | ideal | information, standard |
| --- | --- | --- | --- | --- | --- |
| 1.62° (0.5×) | 3.01 (2.66) | 2.89 (2.71) | 2.86 (2.71) | 2.86 (2.71) | 7.5e4 |
| 3.24° (1×) | 3.07 (2.80) | 2.94 (2.78) | 2.85 (2.70) | 2.86 (2.68) | 9.7e4 |
| 6.48° (2×) | 4.12 (3.67) | 3.83 (3.53) | 3.15 (3.11) | 3.21 (2.88) | 1.7e5 |
| 12.96° (4×) | 44.14 (18.13) | 4941.81 (25.43) | 8.56 (7.58) | 8.34 (7.87) | 7.3e5 |

The odometry noise draws are the same in every row (they are scaled afterwards), and the landmarks first seen at step 0 are initialized identically at every scale (maximum difference 0 m, printed by the sweep). The 1× row is the default setting with 30 trials instead of 50; its standard-EKF NEES (3.07 against 3.95 in the default run) shows how much that mean depends on a few bad trials (§5.2).

- **The NEES gap depends on the map and the noise.** With the paper's noise, it is 0.1 to 0.9 on the three seeds (the paired difference of seed 1 is 0.11 ± 0.07 for FEJ and 0.16 ± 0.07 for the invariant EKF), 2.4 with 12 landmarks, and it grows with the rotation noise.
- **More information along the rotation does not by itself mean a worse NEES.** With range-bearing measurements, the standard EKF adds the most information (2.7e5) and its NEES (2.95) is inside the band. The reason is not tested.
- **The invariant EKF is within 0.2 of the ideal EKF in every row.** At 4× noise, both are above the band (8.6 and 8.3): the linearization error of large heading errors is then a second source of inconsistency that neither fix removes.
- **FEJ breaks down before the others at large noise.** At 4× its mean NEES is 4942 (median trial 25.4): a few trials diverge. Its Jacobians use first estimates, taken before later measurements improved them (ref. 1 notes they are less accurate); whether that is the cause is not tested.

## 6. Every observability finding in the repo

Each doc's finding, by the direction it concerns.

| Doc | Unobservable (or weakly observed) | What the doc shows |
| --- | --- | --- |
| [kf_ekf_iekf.md §2](../filtering/kf_ekf_iekf.md#2-extended-kalman-filter-the-world-is-nonlinear-so-ill-approximate-it-locally) | global yaw (every measurement is relative) | a schematic of an EKF covariance shrinking along yaw; the invariant EKF keeps it unobservable; its own benchmark has no such direction, since the known point cloud fixes the whole pose. [left_right_invariant.md §4](../filtering/left_right_invariant.md#4-why-it-actually-matters-not-just-bookkeeping) repeats the statement in one sentence |
| [hexapod_contact_inekf.md](../filtering/hexapod_contact_inekf.md#why-only-one-filter-can-learn-yaw) | yaw and absolute position (roll and pitch are observable through gravity) | the invariant EKF's yaw variance never decreased (10 seeds × 3 gaits); the world-frame EKF reaches NEES 18.2 at an initial rotation std of 0.1 rad and 481 at 0.3 rad, against 9.3 and 9.4 (9 DoF) |
| [hexapod_gait_ekf.md](../filtering/hexapod_gait_ekf.md#what-is-never-measured) | position and heading | leg odometry only integrates the twist, so both drift |
| [hexapod_camera_fusion_inekf.md](../filtering/hexapod_camera_fusion_inekf.md#intuition) | yaw and position of the legs-only filter | with the camera frame's offset as states, yaw becomes observable (yaw error 0.25° against 0.74°) |
| [hexapod_joint_kinematics_inekf.md §4](../filtering/hexapod_joint_kinematics_inekf.md#4-joint-zero-offsets-the-filter-that-ignores-them-becomes-overconfident) | the 18 encoder zero offsets | only partly learned in 12 s of tripod walking; longer runs, other gaits and turning are not tested |
| [hexapod_redundant_leg_inekf.md §4](../filtering/hexapod_redundant_leg_inekf.md#4-joint-zero-offsets) | the offset along a 4-joint leg's null direction ($`Jn = 0`$) | still learned to about 0.6° of a 2° prior, likely because $`n`$ turns during a stance (not tested); the hard offsets are a femur-tibia combination |
| [eskf.md §4](../filtering/eskf.md#4-results-at-the-defaults) | IMU biases | position fixes plus motion make them observable; without bias states the NEES is 1736 against a consistent 9 |
| [skid_steer_odometry_ekf.md §2](../filtering/skid_steer_odometry_ekf.md#2-filters) | gyro bias and the slip coefficient | both are learnable: a straight part shows the bias alone, a turn shows the coefficient times the yaw rate |
| [magnetic_inchworm_odometry_ekf.md §2](../filtering/magnetic_inchworm_odometry_ekf.md#2-filters) | the mean sideways slide per slip | only the external position fix makes it observable; the servos see no slide |
| [inchworm_zupt_ekf.md §5](../filtering/inchworm_zupt_ekf.md#5-side-extension-heading-and-zaru) | heading, with position-only measurements | observable only through the direction of travel, not at all while anchored |
| [marginalization.md §6](../optimization/marginalization.md#6-the-consistency-gotcha-why-fej-exists) | a common shift of pose and landmark under a range measurement | a worked number: linearizing the two sides at different points predicts a range change of −0.107 m for a shift that changes nothing; FEJ is the fix, and the repo's sliding-window script does not use it |
| [bundle_adjustment.md §13.3](../optimization/bundle_adjustment.md#133-what-local-ba-cant-fix-and-what-can) and [§14](../optimization/bundle_adjustment.md#14-evaluating-the-result-gauge-freedom-and-umeyama-alignment) | the 7-DoF gauge (rigid motion and scale); the drift of the far end of an open path | the gauge is pinned by a prior or by anchor keyframes; orientation drift of 0.9-7.3° stays even with a perfect anchor |
| [umeyama_alignment.md §1.1](umeyama_alignment.md#11-align-only-whats-actually-unobservable) | per sensor: monocular position, orientation and scale (7); stereo, RGB-D, LiDAR position and orientation (6); visual-inertial position and yaw (4) | align only the unobservable degrees of freedom |
| [vi_initialization.md §2](../frontend/vi_initialization.md#2-the-classic-linear-alignment-pipeline) | metric scale and gravity direction of a monocular camera | scale is unobservable at constant velocity (acceleration is needed); pure rotation gives no parallax |
| [lidar_primitives.md §5](../frontend/lidar_primitives.md#5-degeneracy-what-the-hessian-knows-experiment-3) | translation along a plain corridor's axis | the primitive method's Hessian sees it (eigenvalue ratio $`3\times10^{-12}`$, predicted std 13 m); point-to-point ICP does not (predicts 0.11 mm, actual 118 mm) |

These are 15 docs; the other files that mention observability (`README.md`, `GETTING_STARTED.md`, `docs/README.md`, `generator/README.md`, `glossary.md`, `slam_mental_map.md`) are indexes.

## 7. What this toy does not show

- **One robot, flat ground, point landmarks with known identity.** No data association, no 3D rotation, no IMU. Ref. 3's Appendix A.4 derives a 3D invariant EKF-SLAM; the 3D case is not simulated here.
- **No observability-constrained EKF and no robot-centric filter.** Refs. 1 and 3 compare against both.
- **Large heading errors leave the linear regime.** At 4× the rotation noise, even the ideal EKF is outside the band, and FEJ has diverging trials (§5.4); the cause is not tested.
- **One map per seed.** The consistency check is conditional on that map; §5.4 repeats it on two more.
- **The NEES gap is modest at the paper's noise level.** The published gap is larger (ref. 2 Table 1: standard 12.79 against FEJ 3.68, with 15% distance-proportional measurement noise, 1°/s rotation noise and 0.25 m/s); the two setups differ in more than the noise level, and no run here reproduces that scenario.
- **The information measure is a diagnostic of this repo, not a quantity from the references.** It combines ref. 3's Theorem 2 with the filter's own carried direction.

## 8. Takeaway

- **Unobservable directions are a property of the model, and the filter must keep them wide.** Check with the null space of the observability matrix of the system the filter actually uses.
- **A standard EKF's linear model has one unobservable direction too few (2 instead of 3), the global rotation**, because its Jacobians use estimates that moved between steps. Here, that adds $`9\times10^4`$ of information along the rotation per run.
- **FEJ and the invariant EKF keep all 3.** FEJ needs both of its rules; the invariant EKF does it by construction, at the cost of the group-valued update.
- **The NEES effect is small at this noise level and map-dependent** (§5.2, §5.4). The information measure and the null-space dimension show the mechanism more clearly than NEES does.

## 9. References

1. Huang, G. P., Mourikis, A. I., & Roumeliotis, S. I. (2008). *Analysis and Improvement of the Consistency of Extended Kalman Filter-Based SLAM*. 2008 IEEE International Conference on Robotics and Automation (ICRA), 473-479. https://doi.org/10.1109/ROBOT.2008.4543252 - the observability matrix (eq. 20), the 3-dimensional null space of the true system (eq. 18) and of the ideal EKF, the 2-dimensional one of the standard EKF (eqs. 25-28), the FEJ rules (eqs. 29, 30). Their journal version (IJRR 2010, https://doi.org/10.1177/0278364909353640) could not be reached and was not used.
2. Huang, G. P., Mourikis, A. I., & Roumeliotis, S. I. (2009). *A First-Estimates Jacobian EKF for Improving SLAM Consistency*. Experimental Robotics: The Eleventh International Symposium, Springer Tracts in Advanced Robotics 54, 373-382. https://doi.org/10.1007/978-3-642-00196-3_43 - the same rank result (eq. 8) and FEJ rules in the conference version, and the NEES numbers in Table 1 quoted in §7.
3. Barrau, A., & Bonnabel, S. (2016). *An EKF-SLAM algorithm with consistency properties*. arXiv:1510.06263v3. https://arxiv.org/abs/1510.06263 - the SLAM model (eqs. 9, 11), the unobservable shifts (Prop. 1, 2), the invariant EKF-SLAM (Algorithm 3; eqs. 16, 21, 22; Thm. 1, 2), and the simulation scenario (Sec. 7.1). Their stability paper (IEEE Trans. Automatic Control 2017) is about observer convergence and has no SLAM consistency result.
