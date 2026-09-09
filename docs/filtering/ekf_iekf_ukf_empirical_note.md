# Empirical note of EKF vs. IEKF vs. UKF in `pointcloud_pose_tracking.py`: what's actually identical, and what isn't

Applies to both [`use_numpy/pointcloud_pose_tracking.py`](../../use_numpy/pointcloud_pose_tracking.py) and [`use_manif/pointcloud_pose_tracking.py`](../../use_manif/pointcloud_pose_tracking.py), which implement three ways to turn the same predicted point-cloud measurements into a pose correction: `run_ekf`, `run_iekf`, `run_ukf`. It's tempting to lump "they all give similar numbers" into one claim, but on this benchmark two very different things are actually true at once:

- **EKF and IEKF are bit-identical** (to ~1e-14, floating-point noise) - a proven property of this specific problem setup, not a coincidence.
- **UKF is close to both, but is not the same algorithm and does not match bit-for-bit** - the gap is small on this benchmark, but it is a real, structural difference (about 9 orders of magnitude larger than the EKF/IEKF floating-point gap), not noise either.

This doc keeps those two claims separate and explains the mechanism behind each.

## 1. The setup

- **State**: a single rigid pose `T` (SE(3)).
- **Motion model**: `T_pred = T_prev (+) Exp(twist * dt)` - constant body-frame twist. Composing with this known relative motion makes the predict step's linearization exact, not first-order (loosely "group-affine" in spirit, though that term technically describes richer coupled systems like IMU position/velocity/attitude propagation) - true for all three methods, not part of the equivalence argument below.
- **Observation model**: a fixed body-frame point cloud `p_i`, observed as `z_i = T.act(p_i) + noise`, with **isotropic** Gaussian noise (`R = sigma^2 * I`, same variance in every direction, uncorrelated across x/y/z).

Three ways to turn a predicted pose + point-cloud measurement into a correction:

- **EKF**: residual and Jacobian expressed in the **world frame**: `r_world = z_i - T_pred.act(p_i)`, `H_world = R_pred @ [I | -skew(p_i)]` (rebuilt every step from the current rotation estimate `R_pred`).
- **IEKF**: residual and Jacobian expressed in the **object's own body frame**: `r_body = T_pred^-1.act(z_i) - p_i`, `H_body = [I | -skew(p_i)]` (fixed - depends only on the object's known geometry, precomputed once).
- **UKF**: no Jacobian at all. Sigma points sampled around the current estimate are retracted onto SE(3) and pushed through the *exact* `motion_model`/`observation_model`, then recombined into a new mean/covariance (full derivation in `run_ukf`'s own docstring).

EKF and IEKF are two algebraically-related ways of *linearizing the same model*; UKF instead avoids linearizing it at all. That difference in kind is exactly why the first pair can be proven identical while the third can only be shown to be *close*.

---

## 2. EKF vs. IEKF: exact, by construction

### 2.1 Intuitive explanation

Both filters are answering the same question - "how far off is my predicted point cloud from what I actually measured, and what pose correction explains that gap?" - they just *describe* the mismatch in different coordinate systems:
 - **EKF** reports it in world coordinates ("2cm too far east");
 - **IEKF** reports the same physical mismatch in the object's own coordinates ("2.2cm too far toward the object's nose").

Converting between the two is just applying the current rotation estimate - a rigid relabeling of axes that doesn't stretch or distort anything. As long as both the *error* and the *sensitivity* (how a pose tweak would move the points) are converted consistently, the real-world correction you get back is the same either way - like reporting a distance in miles vs. km and converting back.

The one thing that *could* break this is if the measurement noise "looked different" depending on which direction you're facing (e.g. a sensor noisier sideways than in depth). But the noise here is **isotropic** - a perfect sphere of uncertainty around each point - and a sphere looks identical no matter how you rotate it. That's the actual ingredient that makes the two filters land on bit-identical corrections every step: rotating an isotropic covariance leaves it unchanged (`R @ (sigma^2 I) @ R.T = sigma^2 I` for any rotation `R`).

### 2.2 The algebra (for the curious)

Per point $p_i$:

```
r_body_i = T_pred^-1.act(z_i) - p_i = R_pred.T @ (z_i - T_pred.act(p_i)) = R_pred.T @ r_world_i

H_world  = R_pred @ H_body
```

Stack over all `M` points; let `Rbig` = block-diagonal repeat of `R_pred`, `M` times (still orthogonal). Then `r_body = Rbig.T @ r_world` and `H_world = Rbig @ H_body`. Push this through the Kalman update:

```
S_world = H_world @ P @ H_world.T + R_diag
        = Rbig @ (H_body @ P @ H_body.T + R_diag) @ Rbig.T   [needs Rbig @ R_diag @ Rbig.T = R_diag]
        = Rbig @ S_body @ Rbig.T

K_world = P @ H_world.T @ S_world^-1
        = P @ H_body.T @ Rbig.T @ (Rbig @ S_body @ Rbig.T)^-1
        = K_body @ Rbig.T

delta_world = K_world @ r_world = K_body @ Rbig.T @ r_world = K_body @ r_body = delta_body
```

The step `Rbig @ R_diag @ Rbig.T = R_diag` is exactly where isotropy is used - it's the only place the argument could fail. `P` and `T_est` update identically thereafter, every step, so the two trajectories never diverge. Note that nothing in this argument mentions sigma points or a specific noise-injection scheme - it's a pure statement about two *linearizations* of the same model agreeing, which is why it has no counterpart for UKF (§3).

### 2.3 Empirical verification

- Printed final/RMS rotation+position error rows are identical between EKF and IEKF across seed 0 (default args) and stress tests (`--init-pose-noise-std 0.5`/`0.8`, `--duration 8`).
- Direct numerical diff of the full trajectory (duration 5.0, seed 0, 51 poses): max position diff ~1.0e-14 m, max rotation diff ~1.7e-6 deg - floating-point noise, not a real difference.
- Saved trajectory/error plots show the IEKF line drawn exactly on top of the EKF line everywhere (invisible because identical).

### 2.4 What actually differs between them: speed, not accuracy

`H_body` is a fixed matrix (depends only on the object's known geometry), computed once outside the step loop. `H_world` depends on `R_pred`, the *current* rotation estimate, so EKF rebuilds it every step. Measured ~30% faster per step for IEKF (e.g. 260 vs 369 microseconds/step in one run) at identical memory - the entire practical benefit on this benchmark.

---

## 3. UKF vs. EKF/IEKF: close, but not exact

### 3.1 Why they're close in the first place

`run_ukf` alternates predict/update exactly like `run_ekf` does, and on this benchmark the per-step twist increment (`twist * dt`, with `dt=0.1` and the modest angular rates from `true_body_rates`) is small, i.e. the region EKF linearizes around is close to flat. A first-order Jacobian is an excellent local approximation there, so it's not surprising the two land close together. What's worth being precise about is that "close" here is not the same claim as §2's "identical", and the two mechanisms below are the reason.

### 3.2 Two genuine sources of disagreement

1. **Additive vs. Jacobian-scaled process noise.** `run_ekf` propagates process noise through the motion model's own noise Jacobian: `P_pred = J_self @ P @ J_self.T + J_tau @ Q_tangent @ J_tau.T`, where `J_tau = se3_right_jacobian(twist * dt)`. `run_ukf`, by design (see its own docstring), instead adds `Q_tangent` directly to the recombined covariance - the standard "additive-noise UKF" simplification, chosen so sigma points don't need extra dimensions for process noise. These two only agree exactly when `J_tau ≈ I`, which holds to first order for a small twist increment but is not an exact identity - `J_tau` genuinely departs from `I` by a term of order `twist * dt`.
2. **Second-order curvature of the observation model.** `observation_model` is nonlinear in the rotation (it composes through the Exp map under the retraction). EKF's Jacobian captures only the *first-order* (tangent-plane) behavior of that nonlinearity by construction. UKF's sigma points instead sample the *exact* nonlinear function and reconstruct the posterior mean/covariance from those exact evaluations, which is precisely what lets a UKF outperform an EKF when nonlinearity is strong - here the nonlinearity is mild, so the correction from this term is small, but it is not zero.

Both effects shrink as the per-step rotation increment shrinks (smaller `dt`, slower true angular rate, or a tighter prior needing a smaller correction) - which is also a testable prediction (§3.4).

### 3.3 Empirical verification

Direct numerical diff of the full trajectory (`use_numpy`, duration 5.0, seed 0, 51 poses, default UKF tuning `alpha=1.0, beta=2.0, kappa=-3.0`):

```
EKF  vs IEKF: max pos diff = 1.031e-14 m, max rot diff = 1.708e-06 deg
EKF  vs UKF : max pos diff = 1.445e-03 m, max rot diff = 6.298e-02 deg
IEKF vs UKF : max pos diff = 1.445e-03 m, max rot diff = 6.298e-02 deg
```

The EKF-vs-UKF gap is about **9 orders of magnitude larger** than the EKF-vs-IEKF gap - a real, structural difference, not floating-point noise, even though it is small in absolute terms relative to the actual pose error on this benchmark (final position error ~0.003 m, i.e. the UKF/EKF disagreement is roughly half a percent of the error itself). The "Final / RMS errors" table `pointcloud_pose_tracking.py` prints at the end of a run only shows 3-4 decimal places, which is why EKF/IEKF/UKF can look identical there even though only EKF/IEKF actually are.

### 3.4 How the gap moves as the step size changes (verified, not just predicted)

Since both mechanisms in §3.2 scale with the per-step rotation increment, the EKF/UKF gap should *grow* under conditions that make that increment larger, e.g. a bigger `dt` (fewer, larger steps covering the same `true_body_rates` profile). This is directly checkable - re-running the numerical diff (§3.3) at a sweep of `dt` values, same seed and duration:

```
dt=0.02  n_steps=250  max pos diff=8.906e-04 m  max rot diff=7.909e-02 deg
dt=0.05  n_steps=100  max pos diff=1.033e-03 m  max rot diff=4.482e-02 deg
dt=0.10  n_steps= 50  max pos diff=1.445e-03 m  max rot diff=6.298e-02 deg
dt=0.20  n_steps= 25  max pos diff=2.921e-03 m  max rot diff=8.555e-02 deg
dt=0.40  n_steps= 12  max pos diff=4.360e-03 m  max rot diff=3.663e-02 deg
dt=0.80  n_steps=  6  max pos diff=1.204e-02 m  max rot diff=7.072e-02 deg
```

The **position** gap grows cleanly and monotonically with `dt`, exactly as predicted. The **rotation** gap moves in the same broad direction but not monotonically - it's a max-over-the-whole-trajectory statistic, and larger `dt` also means fewer, coarser-discretized steps sampling a different subset of the `true_body_rates` profile each time, so which single step happens to realize the worst-case disagreement varies. Worth reporting honestly rather than smoothing over: the *mechanism* (§3.2) and the *position* evidence both point the same way; the rotation statistic is consistent with it but noisier, not a clean confirmation on its own.

### 3.5 What actually differs: a small accuracy gap, and a real cost gap

Unlike EKF vs. IEKF (§2.4, speed only, zero accuracy difference), UKF's disagreement with EKF/IEKF is a genuine (if tiny) accuracy difference in *both* directions - it's not that UKF is strictly more correct here, since both are approximations to the true nonlinear posterior for different reasons (EKF's is a linearization error; UKF's is the additive-noise simplification of §3.2's point 1, combined with a finite, unscented-only sample of the nonlinearity). What is unambiguous is the cost: `run_ukf` calls `motion_model` and `observation_model` `2n+1 = 13` times per step (sigma points for `n=6`) redrawing a fresh set for the update, versus one Jacobian-inclusive call each for EKF/IEKF - measured at roughly 5-13x the per-step wall-clock time of EKF/IEKF on this benchmark, for an accuracy difference that (per §3.3) is negligible at this problem's scale of nonlinearity.

---

## 4. When would each actually diverge more?

The EKF/IEKF equivalence (§2) rests on **isotropic noise**, not on rigidity per se. Two ways it can break:

1. **Anisotropic sensor noise.** If measurement noise is direction-dependent in a *fixed world-frame* sense (e.g. a sensor that's noisier along the world's vertical axis than horizontal, regardless of the object's orientation), `R_diag` no longer commutes with `Rbig`, and EKF/IEKF give different corrections.

2. **Non-rigid deformation tied to the object's own frame.** A rigid-body tracker can absorb small deformation as extra "noise" on top of the rigid assumption. If that wobble is itself isotropic (any-direction jitter), the two filters *still* match - isotropic noise is rotation-invariant regardless of its physical source. But real deformation is often *structured*: a flag flexing along its pole, a limb bending more along its length than sideways - an ellipsoid of uncertainty aligned with the object's own axes (its long axis, a hinge axis), not the world's.
   - **EKF (world frame)** sees that ellipsoid rotate with the object every step; if it doesn't re-derive its noise model to track that rotation, it's silently using the wrong noise shape.
   - **IEKF (body frame)** sees the same ellipsoid sitting still, since it's fixed relative to the object's own axes - it can use a fixed, correctly-shaped anisotropic covariance with no per-step rotation.
   - Here the two filters genuinely diverge, and **IEKF is arguably the more natural model**, not just the faster one.

3. **Genuinely non-rigid motion (not just noisy-around-a-rigid-mean).** If no single rigid transform `T` reasonably explains the point cloud's motion at all, the question "does EKF or IEKF do better" isn't quite well-posed - both filters' core assumption (`z_i = T.act(p_i) + noise` for one shared rigid `T`) has failed. That needs a richer state (pose plus some deformation/shape parameters), not a filter swap.

**Rule of thumb (EKF vs. IEKF):** it's not **rigid** vs. **non-rigid** that decides this - it's **whether the uncertainty is isotropic or anisotropic-and-tied-to-the-object's-frame**. Non-rigid objects are simply a natural, common source of the latter. None of this affects UKF's already- approximate agreement with either filter one way or the other, since §3's gap comes from linearization/noise-injection mechanics, not from the isotropy argument in §2.

**Rule of thumb (UKF vs. EKF/IEKF):** per §3.4, the gap widens with stronger per-step nonlinearity (bigger `dt`, faster rotation, larger corrections) and narrows toward the EKF/IEKF floor as the per-step motion shrinks - it does not have an isotropy precondition the way the EKF/IEKF equivalence does, and it never becomes bit-identical the way EKF/IEKF are, no matter how small the step gets (there's always a residual first-vs-second-order gap, it just shrinks toward zero).
