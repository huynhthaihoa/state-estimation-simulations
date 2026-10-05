# Empirical note on EKF vs. IEKF vs. UKF vs. vanilla KF in `pointcloud_pose_tracking.py`: what's actually identical, and what isn't

Applies to both [`use_numpy/pointcloud_pose_tracking.py`](../../use_numpy/pointcloud_pose_tracking.py) and [`use_manif/pointcloud_pose_tracking.py`](../../use_manif/pointcloud_pose_tracking.py), which implement four ways to turn the same predicted point-cloud measurements into a pose correction: `run_ekf`, `run_iekf`, `run_ukf`, `run_vanilla_kf`. It's tempting to lump "they all give similar numbers" into one claim, but on this benchmark three very different things are actually true at once:

- **EKF and IEKF are identical** (position and rotation both agree to ~1e-14, i.e. floating-point round-off) - a proven property of this specific problem setup, not a coincidence.
- **UKF is close to both, but is not the same algorithm and does not match bit-for-bit.** Its largest disagreement (~1.4 mm) happens at the very first update, while the filters are correcting the initial pose error; after that it settles to micrometers (~4e-6 to 2e-5 m at the default step), and it grows with the step size as the mechanism predicts (§3).
- **Vanilla KF is neither identical to nor merely "close" to the other three.** It changes the state representation rather than the linearization, residual frame, or sampling scheme. Its steady-state gap (~4 mm at the defaults) comes mainly from a first-order truncation of the motion step, so it shrinks roughly in proportion to $\Delta t$ (§4).

This doc keeps these three claims separate and explains the mechanism behind each.

---

## 1. The setup

- **State**: a single rigid pose $T \in SE(3)$.
- **Motion model**: $`T_{\text{pred}} = T_{\text{prev}}\exp(u\,\Delta t)`$ - constant body-frame twist $u$ over a step of length $\Delta t$. Composing with this known relative motion makes the predict step's linearization exact, not first-order. This is the simplest *group-affine* system (right-multiplication by a known group element, listed explicitly by Barrau & Bonnabel 2017, Remark 1); IMU position/velocity/attitude propagation is a richer example of the same class. This holds for EKF, IEKF, and UKF (not part of the equivalence argument below), but notably **not** for vanilla KF, which only approximates this composition (§4.2).
- **Observation model**: a fixed body-frame point cloud $p_i$, observed as $z_i = T\cdot p_i + \text{noise}$ ($T\cdot p_i$ is the pose applied to a point, `T.act(p_i)` in code), with **isotropic** Gaussian noise ($R = \sigma^2 I$, same variance in every direction, uncorrelated across x/y/z).

All four filters run the same two-step loop at every time step: **predict** the pose from the known twist $u$, then **update** it with the point-cloud measurement. They differ in *which* step they change and *how*, so it is easiest to compare them one step at a time.

**Predict - where the pose is propagated forward.** EKF, IEKF and UKF all use the exact motion model above on the minimal 6-dim $SE(3)$ tangent state; only vanilla KF changes it.

- **EKF and IEKF**: identical code. The covariance is propagated through the motion model's Jacobians, and because the motion is group-affine this is exact, not a first-order approximation.
- **UKF**: no Jacobians. Sigma points sampled around the current estimate are retracted onto $SE(3)$, pushed through the same exact `motion_model`, and recombined into a predicted covariance (equations in §1.1).
- **Vanilla KF**: the odd one out. It never calls `motion_model`; instead it swaps the pose for a redundant 12-dim ambient state $`x = [\text{vec}(R) \in \mathbb{R}^9,\ t \in \mathbb{R}^3]`$ (with $\text{vec}$ taken **row-major**, as `R.flatten()` does - see §1.1) and propagates it with a fixed linear transition matrix. That matrix is only exact for the first-order truncation $`\exp(\omega^\wedge) \approx I + \omega^\wedge`$ of the motion step - the main source of its gap (§4).

**Update - where the measured points correct the predicted pose.** Every filter compares the measured points $z_i$ with the predicted ones $T_{\text{pred}}\cdot p_i$ and turns the mismatch into a correction; they differ in the frame and the tool used for that comparison.

- **EKF**: residual and Jacobian in the **world frame**: $r_{\text{world}} = z_i - T_{\text{pred}}\cdot p_i$, $`H_{\text{world}} = R_{\text{pred}}\left[I \;\; -p_i^\wedge\right]`$. $H$ depends on the current rotation estimate $R_{\text{pred}}$, so it is rebuilt every step.
- **IEKF**: the same comparison pulled into the **object's own body frame**: $r_{\text{body}} = T_{\text{pred}}^{-1}\cdot z_i - p_i$, $`H_{\text{body}} = \left[I \;\; -p_i^\wedge\right]`$. $H$ now depends only on the object's known geometry, so it is fixed and precomputed once. This residual is **left-invariant**: redefining the world frame (left-multiplying both the true and the estimated pose by the same fixed transform) doesn't change it. The measurement shape $h(T) = T\cdot p_i$ is the $`X b`$ row of [left_right_invariant.md §4](left_right_invariant.md#4-why-it-actually-matters-not-just-bookkeeping)'s table, which pairs with the left-invariant choice.
- **UKF**: no Jacobian. Fresh sigma points around the predicted pose are pushed through the exact `observation_model`, and their spread gives the gain directly (equations in §1.1).
- **Vanilla KF**: in the 12-dim state the observation model is exactly linear, $`\text{pred}_i = R\,p_i + t`$, so $H$ is fixed and built once - stronger than IEKF's still-$`SE(3)`$-flavored fixed $H$. The catch: nothing keeps $\text{vec}(R)$ orthonormal, so after every update it is re-projected onto $SO(3)$ via SVD (§4).

**Putting the two steps together:** EKF and IEKF share the predict step and differ only in the frame the update is linearized in - two algebraically related linearizations of the same model, which is why they can be proven identical (§2). UKF avoids linearizing in either step, so it can only be shown to be *close* (§3). Vanilla KF changes the state representation itself, which touches both steps (§4).

![Three panels from pointcloud_pose_tracking.py's defaults: the same 20-point body cloud carried rigidly along the true trajectory, the world-frame EKF residual between the measured points and the twist-only prediction T_pred·p at t = 1 s, and the same residuals pulled into the body frame as T_pred⁻¹·z − p for the IEKF, with identical lengths](../../assets/pose_tracking_concept.png)

*Figure: the setup and the two residual frames at `use_numpy/pointcloud_pose_tracking.py`'s defaults (seed 0), plotted by `uv run python assets/make_figures.py pose_tracking_concept`.*

![Two panels from pointcloud_pose_tracking.py at its defaults: the x-y trajectories of ground truth, dead reckoning and five estimators (EKF, invariant EKF, UKF, vanilla KF, batch Gauss-Newton), and each one's position error over time on a log scale](../../assets/pose_tracking.png)

*Figure: `use_numpy/pointcloud_pose_tracking.py` at its defaults (seed 0), plotted by `uv run python assets/make_figures.py pose_tracking`.*

### 1.1 The filter math, concretely

This section maps each piece of math to the function in [`use_numpy/pointcloud_pose_tracking.py`](../../use_numpy/pointcloud_pose_tracking.py) that computes it. Lie-group helpers live in [`use_numpy/lie_utils.py`](../../use_numpy/lie_utils.py), and the UKF helpers live in [`utils.py`](../../utils.py). The `use_manif` version implements the same math with the `manif` library.

#### Conventions

A pose $T$ is a $4\times4$ matrix with rotation $R$ and translation $t$. A tangent vector is $`\xi = [v, \omega]`$, translation first. $\mathrm{Exp}$/$`\mathrm{Log}`$ are `se3_exp`/`se3_log`, and $a^\wedge$ is `skew(a)`. Every method except the vanilla KF perturbs and corrects on the right: $`T \leftarrow T\,\mathrm{Exp}(\delta)`$. `se3_log` returns zero rotation when the angle is below $10^{-6}$ rad (its small-angle branch). Superscripts mark the step: $T^-$, $P^-$ are the predicted pose and covariance ($T_{\text{pred}}$ in §1), and $T^+$, $P^+$ are the corrected ones.

#### Data and noise (`generate_ground_truth_and_data`, `main`)

The true twist $`u^{\text{true}}_k = [v, \omega]`$ comes from `true_body_rates(k*dt)`. The true pose starts at $T_0 = I$ and follows $`T_{k+1} = T_k\,\mathrm{Exp}(u^{\text{true}}_k\Delta t)`$, using the same `motion_model` the filters use. The filters receive $`u_k = u^{\text{true}}_k + n_k`$ with $`n_k \sim \mathcal N(0, Q_{\text{rate}})`$. The $M$ body points $p_i$ are drawn uniformly in $[-0.5, 0.5]^3$ (`make_body_point_cloud`). The measurements are $`z_{k,i} = T_k p_i + \nu`$ with $`\nu \sim \mathcal N(0, \sigma_p^2 I_3)`$, for $k = 0..N$. The filters use the generator's own noise levels, so $Q$ is matched to the data, not tuned:

```math
Q_{\text{rate}} = \mathrm{diag}(\sigma_v^2 I_3,\ \sigma_\omega^2 I_3), \qquad
Q = Q_{\text{tangent}} = \Delta t^2\, Q_{\text{rate}}, \qquad
R_{\text{diag}} = \sigma_p^2 I_{3M}
```

The factor $\Delta t^2$ appears because the noise is on the twist *rate*, while `motion_model` consumes the increment $u\Delta t$. Scaling a random vector by a constant scales its covariance by that constant squared: $`\mathrm{Cov}(n_k\Delta t) = \Delta t^2\,\mathrm{Cov}(n_k)`$. This is not the textbook $`Q \approx Q_c\,\Delta t`$, which discretizes continuous white noise with spectral density $Q_c$. The generator instead draws a fresh $n_k$ with a fixed standard deviation at every step, so $\Delta t^2$ is the matched choice, not a bug. One consequence: over a fixed duration $N\Delta t$, the summed increment noise has variance (to first order, per axis) $`N\,\Delta t^2\sigma^2 = (N\Delta t)\,\Delta t\,\sigma^2`$. A smaller `--dt` at the same `--vel-noise-std`/`--gyro-noise-std` therefore also means less dead-reckoning drift, not just finer sampling.

Every method starts from the same guess, $`T_0^{\text{est}} = T_0\,\mathrm{Exp}(\varepsilon)`$ with $\varepsilon \sim \mathcal N(0, \sigma_0^2 I_6)$, and $P_0 = \sigma_0^2 I_6$.

| Argument | Symbol | Default |
| --- | --- | --- |
| `--dt`, `--duration` | $\Delta t$, $N$ | 0.1 s, 5 s ($N = 50$ steps, 51 poses) |
| `--n-points` | $M$ | 20 |
| `--vel-noise-std`, `--gyro-noise-std` | $\sigma_v$, $\sigma_\omega$ | 0.05 m/s, 0.02 rad/s |
| `--point-noise-std` | $\sigma_p$ | 0.03 m |
| `--init-pose-noise-std` | $\sigma_0$ | 0.1 |
| `--ukf-alpha`, `--ukf-beta`, `--ukf-kappa` | $\alpha, \beta, \kappa$ | 1.0, 2.0, -3.0 |
| `--gn-tol`, `--gn-max-iters` | | 1e-6, 20 |

The four filters run in the same order each step: predict with $u_k$, then update with $z_{k+1}$. They never use $z_0$. Batch GN uses all of $z_0..z_N$.

#### Motion model (`motion_model`)

This is used by dead reckoning, EKF, IEKF, UKF and GN. With $w = u\Delta t$:

```math
T^- = T\,\mathrm{Exp}(w), \qquad
J_{\text{self}} = \mathrm{Ad}_{\mathrm{Exp}(-w)}, \qquad
J_\tau = J_r(w)
```

```math
\mathrm{Ad}_T = \begin{bmatrix} R & t^\wedge R \\ 0 & R \end{bmatrix}, \qquad
J_r(\xi) = \sum_{n=0}^{17} \frac{(-\mathrm{ad}_\xi)^n}{(n+1)!}, \qquad
\mathrm{ad}_\xi = \begin{bmatrix} \omega^\wedge & v^\wedge \\ 0 & \omega^\wedge \end{bmatrix}
```

$\mathrm{Ad}$ is `se3_adjoint`. $J_r$ is `se3_right_jacobian`, which sums the series to 18 terms instead of using a closed form. $J_r^{-1}$ (`compute_se3_inv_right_jacobian`) is the matrix inverse of that series, not a separate formula. $J_{\text{self}}$ is exact, because $`T\,\mathrm{Exp}(\delta)\,\mathrm{Exp}(w) = T\,\mathrm{Exp}(w)\,\mathrm{Exp}(\mathrm{Ad}_{\mathrm{Exp}(-w)}\delta)`$.

#### Dead reckoning (`run_dead_reckoning`)

Applies $`T_{k+1} = T_k\,\mathrm{Exp}(u_k\Delta t)`$ only. Its trajectory is also batch GN's initial guess.

#### EKF (`run_ekf`)

**Predict step.**

```math
P^- = J_{\text{self}}\, P\, J_{\text{self}}^\top + J_\tau\, Q\, J_\tau^\top
```

**Update step.** It stacks all $M$ points. `observation_model` returns $h_i(T) = R p_i + t$ and, for a right perturbation, $`H_i = [\,R \;\; -R\,p_i^\wedge\,]`$:

```math
r = z_{k+1} - h(T^-), \qquad S = H P^- H^\top + R_{\text{diag}}, \qquad K = P^- H^\top S^{-1}
```

```math
T^+ = T^-\,\mathrm{Exp}(K r), \qquad P^+ = (I - K H)\,P^-
```

The **covariance update** uses the simple $(I-KH)P^-$ form, not the Joseph form, and $S$ is inverted explicitly. $P^+$ is not transported by $J_r(Kr)$ after the retraction. This is a common simplification.

#### IEKF (`run_iekf`)

**Predict step.** The same as the EKF's.

**Update step.** It uses the body-frame residual and a fixed Jacobian, built once before the loop:

```math
r_i = (R^-)^\top (z_{k+1,i} - t^-) - p_i, \qquad H_i = \begin{bmatrix} I_3 & -p_i^\wedge \end{bmatrix}
```

Here $R^-$ and $t^-$ come from $T^-$, so $r_i = (T^-)^{-1} z_{k+1,i} - p_i$. The gain, the retraction $`T^+ = T^-\,\mathrm{Exp}(Kr)`$ and $P^+$ are the same as the EKF's. Under isotropic $R_{\text{diag}}$, the resulting correction $Kr$ and $P^+$ are identical to the EKF's (§2.2).

#### UKF (`run_ukf`, with `unscented_weights` and `unscented_sigma_offsets` from `utils.py`)

It works on the $n = 6$ tangent space and uses the scaled (Van der Merwe) weights:

$$
\lambda = \alpha^2 (n + \kappa) - n, \qquad
W^m_0 = \frac{\lambda}{n + \lambda}, \qquad
W^c_0 = W^m_0 + 1 - \alpha^2 + \beta, \qquad
W^m_i = W^c_i = \frac{1}{2(n + \lambda)}, \quad i = 1..2n
$$

The defaults give $\lambda = -3$, $n + \lambda = 3$, $W^m_0 = -1$, $W^c_0 = 1$ and $W_i = 1/6$. The negative central mean weight is intentional. The sigma offsets are $\chi_0 = 0$ and $\chi_i, \chi_{n+i} = \pm$ the $i$-th column of $L$. Here $L L^\top = (n+\lambda)(P_{\text{sym}} + 10^{-9} I)$, and $P_{\text{sym}}$ is the symmetrized $P$.

**Predict step.** It pushes each retracted sigma point through the exact `motion_model` $f$. The predicted mean is sigma point 0's own propagation, $\bar T^- = f(T, u)$:

```math
\zeta_i = \mathrm{Log}\!\left( (\bar T^-)^{-1} f\big(T\,\mathrm{Exp}(\chi_i), u\big) \right), \qquad
P^- = Q + \sum_{i=0}^{2n} W^c_i\, \zeta_i \zeta_i^\top
```

$Q$ is added outside the sum ("additive-noise" UKF), and the $\zeta_i$ (each propagated sigma point's deviation from $\bar T^-$) are not re-centered on their weighted mean.

**Update step.** It draws fresh offsets $\chi_i$ from $P^-$ rather than reusing the predict-step points. It passes them through the exact `observation_model` $h$:

```math
Z_i = h\big(\bar T^-\,\mathrm{Exp}(\chi_i)\big), \qquad
\hat z = \sum_i W^m_i Z_i
```

```math
P_{zz} = R_{\text{diag}} + \sum_i W^c_i (Z_i - \hat z)(Z_i - \hat z)^\top, \qquad
P_{xz} = \sum_i W^c_i\, \chi_i (Z_i - \hat z)^\top
```

```math
K = P_{xz} P_{zz}^{-1}, \qquad
T^+ = \bar T^-\,\mathrm{Exp}\big(K (z_{k+1} - \hat z)\big), \qquad
P^+ = P^- - K P_{zz} K^\top
```

$P_{xz}$ uses $\chi_i$ directly. Their weighted mean is exactly zero because the offsets come in $\pm$ pairs.

#### Vanilla KF (`run_vanilla_kf`)

The state is $x = [\mathrm{vec}(R), t] \in \mathbb R^{12}$. Here $\mathrm{vec}$ is **row-major**, as `R.flatten()` computes it: $\mathrm{vec}(R) = [R_{11}, R_{12}, R_{13}, R_{21}, \dots, R_{33}]$. This is not the column-stacking $\mathrm{vec}$ common in textbooks. With this layout the observation model $z_i = R p_i + t$ is exactly linear. $H$ is built once:

```math
H_i = \begin{bmatrix} I_3 \otimes p_i^\top & I_3 \end{bmatrix}
= \begin{bmatrix} p_i^\top & 0 & 0 & 1 & 0 & 0 \\ 0 & p_i^\top & 0 & 0 & 1 & 0 \\ 0 & 0 & p_i^\top & 0 & 0 & 1 \end{bmatrix}
```

The transition matrix (`vanilla_kf_transition_matrix`) uses $\omega = u_\omega \Delta t$ and $v = u_v \Delta t$. It implements $R^- = R(I + \omega^\wedge)$ and $t^- = t + R v$, which is linear in $x$ for a known input:

```math
A = \begin{bmatrix} I_3 \otimes (I + \omega^\wedge)^\top & 0 \\ I_3 \otimes v^\top & I_3 \end{bmatrix}
```

This truncates the exact step in two places. $\mathrm{Exp}(\omega^\wedge) \approx I + \omega^\wedge$ for rotation, and $V(\omega) \approx I$ for translation, where the exact increment is $`R\,V(\omega)\,v`$ (§4.2). The noise lift (`se3_tangent_to_ambient_jacobian`) maps a right tangent perturbation at $R$ into the ambient space:

```math
J(R) = \begin{bmatrix} 0 & \big[\mathrm{vec}(R e_1^\wedge)\ \ \mathrm{vec}(R e_2^\wedge)\ \ \mathrm{vec}(R e_3^\wedge)\big] \\ R & 0 \end{bmatrix}
```

The columns are $[v, \omega]$ and the rows are $[\mathrm{vec}(R), t]$. The 6-dim initial covariance is lifted once, $`P_0 = J(R_0)\, P_0^{\text{tan}}\, J(R_0)^\top`$.

**Predict step.**

```math
x^- = A x, \qquad
P^- = A P A^\top + J(R)\, Q\, J(R)^\top
```

**Update step.** The textbook linear KF update:

```math
K = P^- H^\top (H P^- H^\top + R_{\text{diag}})^{-1}, \qquad
x^+ = x^- + K (z_{k+1} - H x^-), \qquad
P^+ = (I - K H) P^-
```

$J$ is evaluated at the pre-predict $R$. The lifted $P_0$ has rank 6. After each update, the rotation block is projected back onto $SO(3)$ with an SVD: $U \Sigma V^\top = \mathrm{svd}(R^+)$, and $`R \leftarrow U\,\mathrm{diag}(1, 1, s)\,V^\top`$ with $s = \mathrm{sign}\det(U V^\top)$. $t^+$ and $P^+$ are left unchanged by that projection.

#### Batch Gauss-Newton (`run_batch_gn`)

It optimizes all poses $T_0..T_N$ jointly ($6(N+1)$ unknowns) against three factor types. $\bar T_0$ is the shared initial guess $T_0^{\text{est}}$:

```math
F = \big\| e_0 \big\|^2_{P_0^{-1}} + \sum_{k=1}^{N} \big\| e_k \big\|^2_{Q^{-1}} + \sum_{k=0}^{N} \sum_{i=1}^{M} \frac{\big\| z_{k,i} - T_k p_i \big\|^2}{\sigma_p^2}
```

```math
e_0 = \mathrm{Log}(\bar T_0^{-1} T_0), \qquad
e_k = \mathrm{Log}\Big( \big(T_{k-1}\,\mathrm{Exp}(u_{k-1}\Delta t)\big)^{-1} T_k \Big)
```

The Jacobians are taken with respect to right perturbations of each pose:

```math
\frac{\partial e_0}{\partial T_0} = J_r^{-1}(e_0), \qquad
\frac{\partial e_k}{\partial T_k} = J_r^{-1}(e_k), \qquad
\frac{\partial e_k}{\partial T_{k-1}} = -J_r^{-1}(-e_k)\, \mathrm{Ad}_{\mathrm{Exp}(-u_{k-1}\Delta t)}, \qquad
\frac{\partial (z_{k,i} - T_k p_i)}{\partial T_k} = -\begin{bmatrix} R_k & -R_k p_i^\wedge \end{bmatrix}
```

Each iteration solves the damped normal equations. It then updates every pose with $`T_k \leftarrow T_k\,\mathrm{Exp}(\delta_k)`$ and stops when $\lVert\delta\rVert$ drops below `gn_tol` or after `gn_max_iters`:

```math
\Big( \sum J^\top \Omega J + 10^{-6} I \Big)\, \delta = -\sum J^\top \Omega\, e
```

There is no line search or Levenberg-Marquardt schedule. It starts from the dead-reckoning trajectory, and with the defaults it converged in 5 iterations. The motion factor uses $Q$ directly as the covariance of $e_k$. It does not use the EKF predict's $J_\tau Q J_\tau^\top$.

#### Error metrics (`pose_errors`, `rotation_geodesic_error`)

- The **rotation error** is the angle $\theta$ of the relative rotation $`W = R^\top \hat R`$, in degrees. It is computed as $`\mathrm{atan2}(s, c)`$, where $`c = (\mathrm{tr}\,W - 1)/2 = \cos\theta`$ is read off the symmetric part of $W$ and $`s = \tfrac12 \lVert (W - W^\top)^\vee \rVert = \sin\theta`$ off its skew part. Unlike $\arccos(c)$, this stays accurate for tiny angles, down to round-off. The `use_manif` version computes the same angle as $`\lVert \hat R\ \text{rminus}\ R \rVert`$, and manif's $SO(3)$ $\mathrm{Log}$ also uses `atan2`, so the same accuracy holds there.
- The **position error** is $\lVert t - \hat t \rVert$.

"Final" is the error in the last pose, whereas "RMS" is calculated over all $N+1$ poses, including $k = 0$. At $k = 0$ the five recursive methods (dead reckoning, EKF, IEKF, UKF, vanilla KF) all still sit at the initial guess, so they share the same initial error in their RMS, while batch GN already re-optimizes $T_0$. Small RMS gaps between GN and the recursive filters therefore partly reflect this difference at $k = 0$.


---

## 2. EKF vs. IEKF: exact, by construction

### 2.1 Intuitive explanation

Both filters are answering the same question - "how far off is my predicted point cloud from what I actually measured, and what pose correction explains that gap?" - they just *describe* the mismatch in different coordinate systems:
 - **EKF** reports it in world coordinates ("2cm too far east");
 - **IEKF** reports the same physical mismatch in the object's own coordinates ("2.2cm too far toward the object's nose").

Converting between the two is just applying the current rotation estimate - a rigid relabeling of axes that doesn't stretch or distort anything. As long as both the *error* and the *sensitivity* (how a pose tweak would move the points) are converted consistently, the real-world correction you get back is the same either way - like reporting a distance in miles vs. km and converting back.

The one thing that *could* break this is if the measurement noise "looked different" depending on which direction you're facing (e.g. a sensor noisier sideways than in depth). But the noise here is **isotropic** - a perfect sphere of uncertainty around each point - and a sphere looks identical no matter how you rotate it. That's the actual ingredient that makes the two filters land on bit-identical corrections every step: rotating an isotropic covariance leaves it unchanged ($`R\,(\sigma^2 I)\,R^\top = \sigma^2 I`$ for any rotation $R$).

### 2.2 The algebra (for the curious)

Per point $p_i$:

```math
r_{\text{body},i} = T_{\text{pred}}^{-1}\cdot z_i - p_i = R_{\text{pred}}^\top\big(z_i - T_{\text{pred}}\cdot p_i\big) = R_{\text{pred}}^\top\, r_{\text{world},i}
```

```math
H_{\text{world}} = R_{\text{pred}}\,H_{\text{body}}
```

Stack over all $M$ points; let $R_{\text{big}}$ = block-diagonal repeat of $R_{\text{pred}}$, $M$ times (still orthogonal). Then $r_{\text{body}} = R_{\text{big}}^\top r_{\text{world}}$ and $`H_{\text{world}} = R_{\text{big}}\,H_{\text{body}}`$. Push this through the Kalman update:

```math
\begin{aligned}
S_{\text{world}} &= H_{\text{world}}\,P\,H_{\text{world}}^\top + R_{\text{diag}} \\ 
&= R_{\text{big}}\big(H_{\text{body}}\,P\,H_{\text{body}}^\top + R_{\text{diag}}\big)R_{\text{big}}^\top && \text{needs } R_{\text{big}}\,R_{\text{diag}}\,R_{\text{big}}^\top = R_{\text{diag}} \\ 
&= R_{\text{big}}\,S_{\text{body}}\,R_{\text{big}}^\top \\ 
K_{\text{world}} &= P\,H_{\text{world}}^\top S_{\text{world}}^{-1} \\ 
&= P\,H_{\text{body}}^\top R_{\text{big}}^\top\big(R_{\text{big}}\,S_{\text{body}}\,R_{\text{big}}^\top\big)^{-1} \\ 
&= K_{\text{body}}\,R_{\text{big}}^\top \\ 
\delta_{\text{world}} &= K_{\text{world}}\,r_{\text{world}} = K_{\text{body}}\,R_{\text{big}}^\top r_{\text{world}} = K_{\text{body}}\,r_{\text{body}} = \delta_{\text{body}}
\end{aligned}
```

The step $`R_{\text{big}}\,R_{\text{diag}}\,R_{\text{big}}^\top = R_{\text{diag}}`$ is exactly where isotropy is used - it's the only place the argument could fail. $P$ and $T_{\text{est}}$ update identically thereafter, every step, so the two trajectories never diverge. Note that nothing in this argument mentions sigma points or a specific noise-injection scheme - it's a pure statement about two *linearizations* of the same model agreeing, which is why it has no counterpart for UKF (§3).

### 2.3 Empirical verification

- Printed final/RMS rotation+position error rows are identical between EKF and IEKF across seed 0 (default args) and stress tests (`--init-pose-noise-std 0.5`/`0.8`, `--duration 8`).
- Direct numerical diff of the full trajectory (duration 5.0, seed 0, 51 poses): max position diff ~2e-14 m, and max rotation diff ~6e-13 deg (~1e-14 rad) from the script's own `rotation_geodesic_error`. Both are floating-point round-off. Don't measure this with $`\lVert\mathrm{Log}(R_{\text{EKF}}^\top R_{\text{IEKF}})\rVert`$: `se3_log`'s small-angle branch returns exactly 0 for any angle below $10^{-6}$ rad, so it can't show differences this small.
- Saved trajectory/error plots show the IEKF line drawn exactly on top of the EKF line everywhere (invisible because identical).

### 2.4 What actually differs between them: speed, not accuracy

$H_{\text{body}}$ is a fixed matrix (depends only on the object's known geometry), computed once outside the step loop. $H_{\text{world}}$ depends on $R_{\text{pred}}$, the *current* rotation estimate, so EKF rebuilds it every step. Measured ~30% faster per step for IEKF (e.g. 260 vs 369 microseconds/step in one run) at identical memory - the entire practical benefit on this benchmark.

---

## 3. UKF vs. EKF/IEKF: close, but not exact

### 3.1 Why they're close in the first place

`run_ukf` alternates predict/update exactly like `run_ekf` does, and on this benchmark the per-step twist increment ($`u\,\Delta t`$, with $\Delta t = 0.1$ and the modest angular rates from `true_body_rates`) is small, i.e. the region EKF linearizes around is close to flat. A first-order Jacobian is an excellent local approximation there, so it's not surprising the two land close together. What's worth being precise about is that "close" here is not the same claim as §2's "identical", and the two mechanisms below are the reason.

### 3.2 Two genuine sources of disagreement

1. **Additive vs. Jacobian-scaled process noise.** `run_ekf` propagates process noise through the motion model's own noise Jacobian: $`P_{\text{pred}} = J_{\text{self}}\,P\,J_{\text{self}}^\top + J_\tau\,Q_{\text{tangent}}\,J_\tau^\top`$, where $J_\tau$ is the right Jacobian of $`u\,\Delta t`$ (`se3_right_jacobian(twist * dt)` in code). `run_ukf`, by design (see its own docstring), instead adds $Q_{\text{tangent}}$ directly to the recombined covariance - the standard "additive-noise UKF" simplification, chosen so sigma points don't need extra dimensions for process noise. These two only agree exactly when $J_\tau \approx I$, which holds to first order for a small twist increment but is not an exact identity - $J_\tau$ genuinely departs from $I$ by a term of order $`u\,\Delta t`$.
2. **Second-order curvature of the observation model.** `observation_model` is nonlinear in the rotation (it composes through the $\exp$ map under the retraction). EKF's Jacobian captures only the *first-order* (tangent-plane) behavior of that nonlinearity by construction. UKF's sigma points instead sample the *exact* nonlinear function and reconstruct the posterior mean/covariance from those exact evaluations, which is precisely what lets a UKF outperform an EKF when nonlinearity is strong - here the nonlinearity is mild, so the correction from this term is small, but it is not zero.

Both effects shrink as the per-step rotation increment shrinks (smaller $\Delta t$, slower true angular rate, or a tighter prior needing a smaller correction) - which is also a testable prediction (§3.4).

### 3.3 Empirical verification

Direct numerical diff of the full trajectory against EKF (`use_numpy`, duration 5.0, seed 0, 51 poses, default UKF tuning $\alpha = 1.0,\ \beta = 2.0,\ \kappa = -3.0$). Rotation differences are measured with `rotation_geodesic_error` (§1.1):

```text
EKF vs IEKF: max pos diff = 1.6e-14 m                                     max rot diff = 5.5e-13 deg
EKF vs UKF : max pos diff = 1.445e-03 m (at k = 1)  final pos diff = 3.8e-06 m   max rot diff = 6.3e-02 deg
```

The UKF gap is real and structural, not floating-point noise, but *where* it happens matters. The maximum sits at $k = 1$, the first update, when every filter is correcting the initial pose error ($\sigma_0 = 0.1$, so a correction of order 0.1 m/0.1 rad): that's the largest correction in the run, and §3.2's second-order term scales with the size of the correction. Once the filters have converged the two agree far more closely: at the final pose the difference is 3.8e-6 m, about 0.1% of the final position error (0.0031 m, the same for both). The same pattern holds for seeds 1 and 2 (maximum at $k = 1$ each time). The "Final/RMS errors" table the script prints only shows 3-4 decimal places, which is why EKF, IEKF and UKF look identical there.

### 3.4 How the gap moves as the step size changes

Both mechanisms in §3.2 scale with the per-step increment, so the steady-state gap should grow with $\Delta t$. Because the first update dominates the maximum over the whole run, the sweep reports two numbers separately: the maximum position gap during the first second (the initial transient), and the maximum after it (the steady state). Same seed and duration throughout:

```text
dt      first 1 s (m)   after 1 s (m)   after 1 s, rot (deg)
0.001   7.2e-04         2.6e-04         4.4e-02
0.002   1.1e-03         1.4e-04         2.5e-02
0.005   6.1e-04         3.4e-05         6.7e-03
0.01    1.1e-03         1.1e-05         2.6e-03
0.02    8.9e-04         4.3e-06         1.5e-03
0.05    1.0e-03         8.6e-06         2.9e-03
0.10    1.4e-03         1.9e-05         9.5e-03
0.20    2.9e-03         6.4e-05         2.5e-02
0.40    4.4e-03         6.2e-04         3.3e-02
```

Three things show up:

- **The transient depends little on $\Delta t$** (6e-4 to 1.4e-3 m up to $\Delta t = 0.1$, rising to 4.4e-3 m only at the largest step): it's set mainly by the size of the initial correction.
- **From $\Delta t = 0.02$ upward, the steady-state gap grows with $\Delta t$**, as the mechanism predicts, for position and rotation alike.
- **Below $\Delta t = 0.02$ it grows again**, to 2.6e-4 m at $\Delta t = 0.001$. This is not floating-point precision. It comes from `unscented_sigma_offsets` ([`utils.py`](../../utils.py)), which adds a fixed $10^{-9} I$ to $P$ before taking its Cholesky factor. As $\Delta t$ shrinks, so does $Q = \Delta t^2 Q_{\text{rate}}$ and with it the converged $P$, until that fixed term is no longer small next to $P$ and visibly widens the sigma points. Setting it to $10^{-15}$ or $0$ (a local experiment; the code still uses $10^{-9}$) drops the gap at $\Delta t = 0.001$ from 2.8e-4 m to 4.2e-6 m, with $\sigma_0 = 0.01$. So the small-$`\Delta t`$ rise is an artifact of a numerical-safety constant, not of the EKF/UKF mechanisms.

Seeds 1 and 2 show the same pattern (steady state at $\Delta t = 0.001/0.02/0.1/0.4$: 2.3e-4/5.1e-6/2.4e-5/7.8e-4 m and 2.8e-4/6.0e-6/2.0e-5/3.9e-4 m).

### 3.5 What actually differs: a small accuracy gap, and a real cost gap

Unlike EKF vs. IEKF (§2.4, speed only, zero accuracy difference), UKF's disagreement with EKF/IEKF is a genuine (if tiny) accuracy difference in *both* directions - it's not that UKF is strictly more correct here, since both are approximations to the true nonlinear posterior for different reasons (EKF's is a linearization error; UKF's is the additive-noise simplification of §3.2's point 1, combined with a finite, unscented-only sample of the nonlinearity). What is unambiguous is the cost: `run_ukf` calls `motion_model` and `observation_model` $2n+1 = 13$ times per step (sigma points for $n=6$), redrawing a fresh set for the update, versus one Jacobian-inclusive call each for EKF/IEKF - measured at roughly 5-13x the per-step wall-clock time of EKF/IEKF on this benchmark, for an accuracy difference that (per §3.3) matters mainly during the initial correction and is about 0.1% of the final error once converged, at this problem's scale of nonlinearity.

---

## 4. Vanilla KF vs. EKF/IEKF/UKF: diverges by construction, not just approximation

### 4.1 What "vanilla" buys and costs

`run_vanilla_kf` takes a different tack than any of §2's/§3's methods: instead of choosing a residual frame (§2) or a linearization strategy (§3), it changes *what the state itself is*. Reparameterizing the pose as an ambient $x = [\text{vec}(R) \in \mathbb{R}^9,\ t \in \mathbb{R}^3]$ vector makes the point-cloud observation model $`\text{pred}_i = R\,p_i + t`$ **exactly linear** - a textbook linear-KF update with a fixed $H$, no Jacobian, ever. That's a stronger claim than IEKF's "fixed $`H`$" (§2.4): IEKF's $`H_{\text{body}}`$ is still built from $SE(3)$-aware machinery (it's the linearization of a manifold-valued observation model, just a state-independent one); vanilla KF's $H$ is linear in the literal sense, because the state it operates on is a plain Euclidean vector.

That gain isn't free. Three costs come bundled with it, none of them part of textbook linear-KF theory proper - they're bolted onto the recursion specifically to make a linear KF usable on a manifold-valued quantity at all (§4.2).

### 4.2 Three genuine sources of divergence

1. **First-order truncation of the mean itself.** Every other method here composes the mean exactly: $`T_{\text{pred}} = T_{\text{prev}}\exp(u\,\Delta t)`$. `vanilla_kf_transition_matrix` instead builds a transition matrix that is only exact for the first-order truncation $\exp(\omega^\wedge) \approx I + \omega^\wedge$, plus a matching truncation of the translation step: it uses $`t + R\,v`$, while the exact $SE(3)$ increment is $`t + R\,V(\omega)\,v`$ with $V(\omega) = I + \tfrac{1}{2}\omega^\wedge + O(\lVert\omega\rVert^2)$, i.e. it also assumes $V(\omega) \approx I$ (§1.1). So unlike §3.2's point 1 (UKF's additive-noise simplification, which only ever affects *covariance* propagation, never the mean), this mechanism is a real, growing error in vanilla KF's point estimate itself, with no counterpart in EKF, IEKF, UKF, or batch GN.
2. **Ambient (12-dim, redundant) vs. minimal (6-dim tangent) covariance.** $P$ is lifted from the initial $6\times6$ tangent covariance into the 12-dim ambient space via `se3_tangent_to_ambient_jacobian`, and the entire recursion - predict, update, gain - runs in that redundant linear space. This $P$ isn't directly comparable dimension-for-dimension to the other four methods' $6\times6$ tangent covariance (see [minimal vs. ambient parameterization](../glossary.md#1-geometry-and-lie-groups) in the glossary).
3. **Post-hoc SVD re-projection.** Nothing in a linear KF constrains a 9-vector to stay an orthonormal rotation matrix. After every update, $\text{vec}(R)$ is explicitly re-projected onto $SO(3)$ via SVD ($`U\,\text{diag}\big(1, 1, \mathrm{sign}(\det(UV^\top))\big)\,V^\top`$) and fed back into the recursion. This is a correction bolted onto the recursion from the outside, not a property of the KF math itself - EKF/IEKF/UKF never need it because they never leave the manifold in the first place ($\exp$/$`\log`$ for EKF/IEKF, retraction for UKF's sigma points).

Mechanism 1 is a small-angle approximation whose error per step grows with the per-step motion, so over a fixed duration it should shrink roughly in proportion to $\Delta t$. Mechanisms 2 and 3 don't depend on the step size, but they only bite hard when the linearization in the ambient space is poor, i.e. when the correction is large. The measurements (§4.4) bear this out: after the initial transient the gap shrinks roughly in proportion to $\Delta t$, so mechanism 1 dominates the steady state, while the first-second gap is large and irregular at every $\Delta t$, which is where mechanisms 2 and 3 show up.

**None of these three mechanisms involve noise shape, a residual frame, or a Jacobian-linearization choice at all** - this divergence axis is completely orthogonal to §2's/§5's isotropy argument. It is already fully present under this benchmark's default isotropic point-noise model, and would not be created or fixed by switching to anisotropic noise.

### 4.3 Empirical verification

Direct numerical diff of the full trajectory (`use_numpy`, duration 5.0, seed 0, 51 poses, default args), same protocol as §3.3:

```text
EKF vs UKF: max pos diff = 1.445e-03 m (at k = 1)   final pos diff = 3.8e-06 m   max rot diff = 6.3e-02 deg
EKF vs VKF: max pos diff = 1.349e-02 m (at k = 1)   final pos diff = 1.4e-03 m   max rot diff = 1.4e-01 deg
```

Vanilla KF's maximum is also at the first update, and about 9x UKF's. The bigger difference is at the end: after convergence UKF agrees with EKF to micrometers, while vanilla KF is still 1.4 mm away, about 45% of the final error.

The script's own printed "Final/RMS errors" table shows how easy this is to miss at a glance: at default args, EKF's final rot/pos reads `0.289 deg/0.0031 m` against vanilla KF's `0.314 deg/0.0026 m`, which looks like noise at 3-4 decimal places. Vanilla KF happens to land slightly closer to the truth in position at this particular final pose, but it follows a different trajectory, not the same one with rounding.

### 4.4 How the gap moves as the step size changes

Same sweep and split as §3.4:

```text
dt      first 1 s (m)   after 1 s (m)   after 1 s, rot (deg)
0.001   1.1e-02         5.0e-05         3.7e-03
0.002   2.1e-02         8.9e-05         1.4e-02
0.005   3.4e-03         2.4e-04         1.0e-02
0.01    2.0e-02         5.0e-04         2.1e-02
0.02    7.8e-03         8.9e-04         5.7e-02
0.05    8.9e-03         2.1e-03         1.4e-01
0.10    1.3e-02         3.8e-03         1.4e-01
0.20    8.1e-03         6.1e-03         6.2e-01
0.40    4.6e-03         9.1e-03         2.3e+00
```

- **The steady-state position gap shrinks roughly in proportion to $\Delta t$**, from 9.1e-3 m at $\Delta t = 0.4$ to 5.0e-5 m at $\Delta t = 0.001$, monotonically. That's mechanism 1, the truncated motion step, whose accumulated error over a fixed duration scales with $\Delta t$. Seeds 1 and 2 agree (5.5e-5/8.9e-4/3.3e-3/1.4e-2 m and 5.2e-5/8.8e-4/3.6e-3/1.3e-2 m at $\Delta t = 0.001/0.02/0.1/0.4$).
- **The steady-state rotation gap grows fast at large $\Delta t$** (0.14° → 0.62° → 2.3° from $\Delta t = 0.1$ to 0.4), consistent with the per-step truncation error growing with the per-step rotation.
- **The first-second gap is large and irregular at every $\Delta t$** (3e-3 to 2e-2 m). That's the initial correction, and it's consistent with the ambient-space linearization and the SVD re-projection (mechanisms 2 and 3) doing the most damage when the correction is large (not isolated separately here). Reporting only the maximum over the whole run would mix this transient into the step-size trend and hide it.

Unlike UKF, vanilla KF's steady-state gap is not limited by a numerical-safety constant; it keeps shrinking down to the smallest step tested.

### 4.5 What actually differs: a real but different cost trade-off

Measured on one run of `use_numpy/pointcloud_pose_tracking.py` at default args. Time is per step; memory is the whole-run peak (tracemalloc). Both depend on the machine:

```text
EKF (recursive)    avg time=  338.08 µs/step | peak mem=  135.117 KB
IEKF (invariant)   avg time=  245.18 µs/step | peak mem=  134.773 KB
UKF (unscented)    avg time= 2163.72 µs/step | peak mem=  174.981 KB
Vanilla KF         avg time=  216.26 µs/step | peak mem=  145.960 KB
```

Timing is noisy. Across four re-runs on the same machine, vanilla KF took 210-379 µs/step and IEKF 196-245 µs/step, so their order flipped between runs. What held in every run: EKF is slower than IEKF, and UKF is several times slower than everything else.

So vanilla KF runs at about IEKF's speed, not faster, despite needing no Jacobian (§4.1). It pays a different cost instead: building the $12\times12$ transition matrix $A$ and the $12\times6$ ambient-lift Jacobian $J$ (`se3_tangent_to_ambient_jacobian`) with basis-vector loops every step, plus the per-step SVD re-projection (§4.2, point 3). IEKF gets its speed with the same accuracy as EKF (§2.4); vanilla KF gets about the same speed with a worse estimate.

---

## 5. When would each actually diverge more?

The EKF/IEKF equivalence (§2) rests on **isotropic noise**, not on rigidity per se. Two ways it can break:

1. **Anisotropic sensor noise.** If measurement noise is direction-dependent in a *fixed world-frame* sense (e.g., a sensor that's noisier along the world's vertical axis than horizontal, regardless of the object's orientation), $R_{\text{diag}}$ no longer commutes with $R_{\text{big}}$, and EKF/IEKF give different corrections. A *diagonal* covariance isn't enough either: only a scalar one, $\sigma^2 I$, is unchanged by every rotation. This script's $R_{\text{diag}}$ is in fact scalar, despite its name.

2. **Non-rigid deformation tied to the object's own frame.** A rigid-body tracker can absorb small deformation as extra "noise" on top of the rigid assumption. If that wobble is itself isotropic (any-direction jitter), the two filters *still* match - isotropic noise is rotation-invariant regardless of its physical source. But real deformation is often *structured*: a flag flexing along its pole, a limb bending more along its length than sideways - an ellipsoid of uncertainty aligned with the object's own axes (its long axis, a hinge axis), not the world's.
   - **EKF (world frame)** sees that ellipsoid rotate with the object every step; if it doesn't re-derive its noise model to track that rotation, it's silently using the wrong noise shape.
   - **IEKF (body frame)** sees the same ellipsoid sitting still, since it's fixed relative to the object's own axes - it can use a fixed, correctly-shaped anisotropic covariance with no per-step rotation.
   - Here the two filters genuinely diverge, and **IEKF is arguably the more natural model**, not just the faster one.

3. **Genuinely non-rigid motion (not just noisy-around-a-rigid-mean).** If no single rigid transform $T$ reasonably explains the point cloud's motion at all, the question "does EKF or IEKF do better" isn't quite well-posed - both filters' core assumption ($z_i = T\cdot p_i + \text{noise}$ for one shared rigid $T$) has failed. That needs a richer state (pose plus some deformation/shape parameters), not a filter swap.

**Rule of thumb (EKF vs. IEKF):** it's not **rigid** vs. **non-rigid** that decides this - it's **whether the uncertainty is isotropic or anisotropic-and-tied-to-the-object's-frame**. Non-rigid objects are simply a natural, common source of the latter. None of this affects UKF's already-approximate agreement with either filter one way or the other, since §3's gap comes from linearization/noise-injection mechanics, not from the isotropy argument in §2.

**Rule of thumb (UKF vs. EKF/IEKF):** the gap is largest when the correction is large (the initial convergence here), and in steady state it grows with the per-step motion (bigger $\Delta t$, faster rotation), per §3.4. It has no isotropy precondition and never becomes bit-identical the way EKF/IEKF are. At very small $\Delta t$ in this code, a fixed $10^{-9} I$ safety term in the sigma-point construction, not the filters themselves, sets the size of the gap.

Vanilla KF's divergence (§4) doesn't belong on this isotropy spectrum at all - it isn't conditional on anisotropic noise the way the EKF/IEKF split is, and it isn't a linearization/sampling-mechanics gap the way the UKF split is. It comes from swapping out the state representation itself (ambient vs. minimal, §4.2 and the [glossary](../glossary.md#1-geometry-and-lie-groups)), and per §4's default-isotropic-noise measurements, it's already fully present without ever touching the noise model.

**Rule of thumb (Vanilla KF vs. the other three):** no isotropy precondition. In steady state its gap shrinks roughly in proportion to $\Delta t$, because it comes mainly from the truncated motion step (§4.4); during large corrections the ambient representation and SVD re-projection add a larger, irregular error at any step size. At the default step it stays millimeters away from the other three after convergence, where UKF is within micrometers.

---

## 6. References

1. Julier, S. J., & Uhlmann, J. K. (1997). *A New Extension of the Kalman Filter to Nonlinear Systems*. Proceedings of SPIE, 3068 (Signal Processing, Sensor Fusion, and Target Recognition VI), 182-193. - the original unscented transform/UKF this doc's §3 empirically compares against EKF/IEKF.

### Further reading

- Lee, D., Jung, M., Yang, W., & Kim, A. (2024). *LiDAR Odometry Survey: Recent Advancements and Remaining Challenges*. Intelligent Service Robotics, 17(2), 95-118. https://doi.org/10.1007/s11370-024-00515-8 - a survey of the problem this benchmark deliberately skips. Here every measured point $z_i$ arrives already paired with its body point $p_i$; a real LiDAR odometry pipeline has to find those correspondences (scan matching) before any filter or optimizer sees a residual.
