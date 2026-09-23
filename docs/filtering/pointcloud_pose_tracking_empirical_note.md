# Empirical note of EKF vs. IEKF vs. UKF vs. vanilla KF in `pointcloud_pose_tracking.py`: what's actually identical, and what isn't

Applies to both [`use_numpy/pointcloud_pose_tracking.py`](../../use_numpy/pointcloud_pose_tracking.py) and [`use_manif/pointcloud_pose_tracking.py`](../../use_manif/pointcloud_pose_tracking.py), which implement four ways to turn the same predicted point-cloud measurements into a pose correction: `run_ekf`, `run_iekf`, `run_ukf`, `run_vanilla_kf`. It's tempting to lump "they all give similar numbers" into one claim, but on this benchmark three very different things are actually true at once:

- **EKF and IEKF are bit-identical** (to ~1e-14, floating-point noise) - a proven property of this specific problem setup, not a coincidence.
- **UKF is close to both, but is not the same algorithm and does not match bit-for-bit** - small in *absolute* terms on this benchmark (~1.4mm), but a real, structural difference (about 11 orders of magnitude larger than the EKF/IEKF floating-point position gap, ~4.6 orders for rotation) that's actually a sizeable fraction (~47%) of the final position error itself, not noise.
- **Vanilla KF is neither identical to nor merely "close" to the other three - it structurally diverges from all of them, unconditionally.** Unlike the EKF/IEKF equivalence, it has no isotropy precondition; unlike the UKF gap, which does grow and shrink with `dt` over the range where the underlying linearization/sampling mechanisms actually dominate (§3.4), vanilla KF's gap is present at *every* `dt` tested, dominated by a completely different, state-representation-level mechanism (§4) - not a different linearization, residual frame, or sampling scheme.

This doc keeps these three claims separate and explains the mechanism behind each.

## 1. The setup

- **State**: a single rigid pose `T` (SE(3)).
- **Motion model**: `T_pred = T_prev (+) Exp(twist * dt)` - constant body-frame twist. Composing with this known relative motion makes the predict step's linearization exact, not first-order (loosely "group-affine" in spirit, though that term technically describes richer coupled systems like IMU position/velocity/attitude propagation) - true for EKF, IEKF, and UKF (not part of the equivalence argument below), but notably **not** for vanilla KF, which only approximates this composition (§4.2).
- **Observation model**: a fixed body-frame point cloud `p_i`, observed as `z_i = T.act(p_i) + noise`, with **isotropic** Gaussian noise (`R = sigma^2 * I`, same variance in every direction, uncorrelated across x/y/z).

Four ways to turn a predicted pose + point-cloud measurement into a correction:

- **EKF**: residual and Jacobian expressed in the **world frame**: `r_world = z_i - T_pred.act(p_i)`, `H_world = R_pred @ [I | -skew(p_i)]` (rebuilt every step from the current rotation estimate `R_pred`).
- **IEKF**: residual and Jacobian expressed in the **object's own body frame**: `r_body = T_pred^-1.act(z_i) - p_i`, `H_body = [I | -skew(p_i)]` (fixed - depends only on the object's known geometry, precomputed once).
- **UKF**: no Jacobian at all. Sigma points sampled around the current estimate are retracted onto SE(3) and pushed through the *exact* `motion_model`/`observation_model`, then recombined into a new mean/covariance (full derivation in `run_ukf`'s own docstring).
- **Vanilla KF**: no Jacobian either, but for a different reason - it never calls `motion_model`/`observation_model` at all. It reparameterizes the pose as a redundant 12-dim ambient state `x = [vec(R) (9,), t (3,)]` instead of the minimal 6-dim SE(3) tangent state the other three use, which makes the point-cloud observation model exactly linear (`pred_i = R@p_i + t`, a fixed `H` built once - stronger than IEKF's still-SE(3)-flavored fixed `H`). The price: its own transition matrix is only exact for a first-order truncation `Exp(w) ≈ I + skew(w)` of the true motion composition every other method uses exactly, and nothing keeps the 9-vector `vec(R)` orthonormal, so it's explicitly re-projected onto `SO(3)` via SVD after every update (full mechanism in §4).

EKF and IEKF are two algebraically-related ways of *linearizing the same model*; UKF instead avoids linearizing it at all. That difference in kind is exactly why the first pair can be proven identical while the third can only be shown to be *close*. Vanilla KF (§4) is a different move again - not a different linearization or a different sampling scheme, but a different *state representation* entirely.

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

The EKF-vs-UKF gap is about **11 orders of magnitude larger** than the EKF-vs-IEKF position gap (1.445e-3 / 1.031e-14 ≈ 1.4e11), and about **4.6 orders of magnitude** larger for rotation (6.298e-2 / 1.708e-6 ≈ 3.7e4) - not the same ratio, and neither is the "9 orders of magnitude" an earlier version of this note claimed. Both are a real, structural difference, not floating-point noise - though it's not negligible in absolute terms relative to the actual pose error on this benchmark: final position error is ~0.003 m, and the UKF/EKF disagreement (1.445e-3 m) is **~47%** of that (1.445e-3 / 0.0031 ≈ 0.466), not the "roughly half a percent" an earlier version of this note claimed. The "Final / RMS errors" table `pointcloud_pose_tracking.py` prints at the end of a run only shows 3-4 decimal places, which is why EKF/IEKF/UKF can look identical there even though only EKF/IEKF actually are.

### 3.4 How the gap moves as the step size changes (verified, not just predicted)

Since both mechanisms in §3.2 scale with the per-step rotation increment, the EKF/UKF gap should *grow* under conditions that make that increment larger, e.g. a bigger `dt` (fewer, larger steps covering the same `true_body_rates` profile). This is directly checkable - re-running the numerical diff (§3.3) at a sweep of `dt` values, same seed and duration:

```
dt=0.001 n_steps=5000  max pos diff=7.221e-04 m  max rot diff=4.915e-02 deg
dt=0.002 n_steps=2500  max pos diff=1.105e-03 m  max rot diff=8.060e-02 deg
dt=0.005 n_steps=1000  max pos diff=6.067e-04 m  max rot diff=2.533e-02 deg
dt=0.01  n_steps= 500  max pos diff=1.089e-03 m  max rot diff=5.871e-02 deg
dt=0.02  n_steps=250  max pos diff=8.906e-04 m  max rot diff=7.909e-02 deg
dt=0.05  n_steps=100  max pos diff=1.033e-03 m  max rot diff=4.482e-02 deg
dt=0.10  n_steps= 50  max pos diff=1.445e-03 m  max rot diff=6.298e-02 deg
dt=0.20  n_steps= 25  max pos diff=2.921e-03 m  max rot diff=8.555e-02 deg
dt=0.40  n_steps= 12  max pos diff=4.360e-03 m  max rot diff=3.663e-02 deg
dt=0.80  n_steps=  6  max pos diff=1.204e-02 m  max rot diff=7.072e-02 deg
```

The **position** gap grows cleanly and monotonically with `dt` from `dt=0.02` up to `0.80`, exactly as predicted - but going *below* `dt=0.02` (an earlier version of this doc didn't test that far), it stops shrinking: `dt=0.01`'s gap (1.089e-03 m) is larger than `dt=0.02`'s, and it stays noisy and non-monotonic all the way down to `dt=0.001`, never dropping meaningfully below the ~6e-4-to-1e-3 m range it first reaches around `dt=0.01`-`0.02` - confirmed across several seeds, not a one-off. This isn't simply "more accumulated steps over the same duration": re-running `dt=0.001` at much shorter durations (fewer total steps) shows the same ~6e-4-to-1e-3 m floor appears almost immediately, not only after thousands of steps have run - consistent with a floating-point precision floor tied to `dt` itself (e.g. `Q_tangent = dt**2 * Q_rate` becoming a very small number at `dt=0.001`, `dt**2=1e-6`) rather than a slowly-accumulating rounding-error effect. The **rotation** gap tells the same qualitative story - see §3.4's original note on it being a noisier, max-over-the-whole-trajectory statistic even in the large-`dt` regime, and it's equally non-monotonic once `dt` drops below `0.02`. Worth reporting honestly rather than smoothing over: the *mechanism* (§3.2) correctly predicts the gap grows with larger `dt`, and the evidence backs that up cleanly from `dt=0.02` to `0.80` - but it does **not** predict, and the data does **not** show, a clean continued shrink all the way to zero as `dt` keeps getting smaller. There's a floor down there instead, most likely numerical rather than a property of the EKF/UKF mismatch mechanisms themselves.

### 3.5 What actually differs: a small accuracy gap, and a real cost gap

Unlike EKF vs. IEKF (§2.4, speed only, zero accuracy difference), UKF's disagreement with EKF/IEKF is a genuine (if tiny) accuracy difference in *both* directions - it's not that UKF is strictly more correct here, since both are approximations to the true nonlinear posterior for different reasons (EKF's is a linearization error; UKF's is the additive-noise simplification of §3.2's point 1, combined with a finite, unscented-only sample of the nonlinearity). What is unambiguous is the cost: `run_ukf` calls `motion_model` and `observation_model` `2n+1 = 13` times per step (sigma points for `n=6`) redrawing a fresh set for the update, versus one Jacobian-inclusive call each for EKF/IEKF - measured at roughly 5-13x the per-step wall-clock time of EKF/IEKF on this benchmark, for an accuracy difference that (per §3.3) is small in absolute terms but not negligible relative to this benchmark's own final error (~47% of it) at this problem's scale of nonlinearity.

---

## 4. Vanilla KF vs. EKF/IEKF/UKF: diverges by construction, not just approximation

### 4.1 What "vanilla" buys and costs

`run_vanilla_kf` takes a different tack than any of §2's/§3's methods: instead of choosing a residual frame (§2) or a linearization strategy (§3), it changes *what the state itself is*. Reparameterizing the pose as an ambient `x = [vec(R) (9,), t (3,)]` vector makes the point-cloud observation model `pred_i = R@p_i + t` **exactly linear** - a textbook linear-KF update with a fixed `H`, no Jacobian, ever. That's a stronger claim than IEKF's "fixed `H`" (§2.4): IEKF's `H_body` is still built from SE(3)-aware machinery (it's the linearization of a manifold-valued observation model, just a state-independent one); vanilla KF's `H` is linear in the literal sense, because the state it operates on is a plain Euclidean vector.

That gain isn't free. Three costs come bundled with it, none of them part of textbook linear-KF theory proper - they're bolted onto the recursion specifically to make a linear KF usable on a manifold-valued quantity at all (§4.2).

### 4.2 Three genuine sources of divergence

1. **First-order truncation of the mean itself.** Every other method here composes the mean exactly: `T_pred = T_prev (+) Exp(twist*dt)`. `vanilla_kf_transition_matrix` instead builds a transition matrix that is only exact for the first-order truncation `Exp(w) ≈ I + skew(w)` - so unlike §3.2's point 1 (UKF's additive-noise simplification, which only ever affects *covariance* propagation, never the mean), this mechanism is a real, growing error in vanilla KF's point estimate itself, with no counterpart in EKF, IEKF, UKF, or batch GN.
2. **Ambient (12-dim, redundant) vs. minimal (6-dim tangent) covariance.** `P` is lifted from the initial 6x6 tangent covariance into the 12-dim ambient space via `se3_tangent_to_ambient_jacobian`, and the entire recursion - predict, update, gain - runs in that redundant linear space. This `P` isn't directly comparable dimension-for-dimension to the other four methods' 6x6 tangent covariance (see [Appendix A.5](#a5-ambientredundant-vs-minimal-state-parameterization)).
3. **Post-hoc SVD re-projection.** Nothing in a linear KF constrains a 9-vector to stay an orthonormal rotation matrix. After every update, `vec(R)` is explicitly re-projected onto `SO(3)` via SVD (`U @ diag([1,1,sign(det(UV^T))]) @ Vt`) and fed back into the recursion. This is a correction bolted onto the recursion from the outside, not a property of the KF math itself - EKF/IEKF/UKF never need it because they never leave the manifold in the first place (`Exp`/`Log` for EKF/IEKF, retraction for UKF's sigma points).

Only mechanism 1 is a genuine small-angle approximation that should shrink as the per-step rotation shrinks; mechanisms 2 and 3 are structural and present regardless of step size. That's the mechanical reason this gap doesn't shrink toward zero over the `dt` range where mechanisms 2/3 already dominate (§4.4) - a different, stronger claim than UKF's gap not shrinking to zero (§3.4): UKF's non-shrinking behavior only shows up once a numerical floor is hit at very small `dt`, while vanilla KF's gap fails to shrink across its *entire* tested range, including the large-`dt` region where UKF's gap tracks the predicted mechanism cleanly.

**None of these three mechanisms involve noise shape, a residual frame, or a Jacobian-linearization choice at all** - this divergence axis is completely orthogonal to §2's/§5's isotropy argument. It is already fully present under this benchmark's default isotropic point-noise model, and would not be created or fixed by switching to anisotropic noise.

### 4.3 Empirical verification

Direct numerical diff of the full trajectory (`use_numpy`, duration 5.0, seed 0, 51 poses, default args), same protocol as §3.3:

```
EKF  vs IEKF: max pos diff = 1.494e-14 m, max rot diff = 2.091e-06 deg
EKF  vs UKF : max pos diff = 1.445e-03 m, max rot diff = 6.298e-02 deg
EKF  vs VKF : max pos diff = 1.349e-02 m, max rot diff = 1.441e-01 deg
```

The EKF-vs-vanilla-KF position gap is about **9.3x larger** than the EKF-vs-UKF gap, and about **12 orders of magnitude** above the EKF/IEKF floating-point floor; the rotation gap is a more modest **2.3x larger** than UKF's - the two ratios aren't the same, and it's worth reporting both rather than rounding the smaller one up to match the bigger one.

The script's own printed "Final / RMS errors" table shows exactly how easy this is to miss at a glance: at default args, EKF's final rot/pos reads `0.289 deg / 0.0031 m` against vanilla KF's `0.314 deg / 0.0026 m` - close enough at 3-4 decimal places to look like noise. The actual full-trajectory max diff (`0.144 deg / 0.0135 m`, above) tells a different story: EKF/IEKF/UKF are printed identically to this precision (§3.3), but vanilla KF visibly is not, even in the truncated table.

### 4.4 How the gap moves as the step size changes (verified, not just predicted)

Sweeping `dt` the same way as §3.4, plus a smaller-`dt` extension (0.005, 0.01) to check whether the gap shrinks toward zero the way UKF's does over its own cleanly-predicted large-`dt` range (§3.4):

```
dt=0.005  n_steps=1000  max pos diff=3.375e-03 m  max rot diff=1.079e-01 deg
dt=0.01   n_steps= 500  max pos diff=1.990e-02 m  max rot diff=2.823e-01 deg
dt=0.02   n_steps= 250  max pos diff=7.821e-03 m  max rot diff=1.493e-01 deg
dt=0.05   n_steps= 100  max pos diff=8.860e-03 m  max rot diff=1.975e-01 deg
dt=0.10   n_steps=  50  max pos diff=1.349e-02 m  max rot diff=1.441e-01 deg
dt=0.20   n_steps=  25  max pos diff=8.118e-03 m  max rot diff=6.166e-01 deg
dt=0.40   n_steps=  12  max pos diff=9.058e-03 m  max rot diff=2.281e+00 deg
dt=0.80   n_steps=   6  max pos diff=1.721e-02 m  max rot diff=1.019e+01 deg
```

Two honest findings, reported without smoothing over either:

- **The gap does not shrink toward zero as `dt` shrinks, anywhere in this sweep** - a stronger and more consistent non-shrink than §3.4's UKF gap, which *does* track the predicted mechanism cleanly over most of its range (`dt=0.02` to `0.80`) and only stops shrinking once a much smaller numerical floor (~6e-4-to-1e-3 m) is hit below `dt≈0.02`. Vanilla KF's position gap instead stays in a ~3e-3-to-2e-2 m band - several times larger than UKF's floor - across the *entire* two-decade sweep, including at `dt=0.005`, exactly as predicted by §4.2: mechanisms 2 and 3 aren't step-size effects, so they set a floor mechanism 1 alone can't fall below, and that floor is high enough to dominate at every `dt` tested, not just the smallest ones.
- **Rotation only becomes cleanly monotonic once per-step rotation dominates** (`dt >= 0.10`: 0.144 deg -> 0.617 -> 2.281 -> 10.19 deg, roughly 4x per doubling of `dt`) - a much cleaner confirmation of "error growing with the per-step rotation magnitude" (the transition matrix's own docstring claim) than §3.4's noisier UKF rotation series. Below `dt=0.10` it is not monotonic (`dt=0.01`'s 0.282 deg is larger than `dt=0.02`'s, `dt=0.05`'s, or even `dt=0.10`'s own value) - the small-`dt` regime is dominated by mechanisms 2/3's structural noise floor rather than mechanism 1's shrinking truncation error, so there's no reason to expect monotonicity there. Position, by contrast, is not monotonic anywhere in this sweep. This is close to the reverse of §3.4's pattern (clean position trend, noisy rotation) - worth stating plainly rather than implying either series behaves like §3.4's.

### 4.5 What actually differs: a real but different cost trade-off

Measured on one run of `use_numpy/pointcloud_pose_tracking.py` at default args:

```
EKF (recursive)    avg time=  918.70 µs/step | avg peak mem=2.703 KB/step
IEKF (invariant)   avg time=  551.23 µs/step | avg peak mem=2.696 KB/step
UKF (unscented)    avg time= 6531.92 µs/step | avg peak mem=3.521 KB/step
Vanilla KF         avg time=  635.85 µs/step | avg peak mem=2.927 KB/step
```

Vanilla KF lands *between* IEKF and EKF/UKF, not at IEKF's floor - a naive reading of "no Jacobian, ever" (§4.1) might expect it to match IEKF's speed, but it pays a different cost instead: building the 12x12 transition matrix `A` and the 12x6 ambient-lift Jacobian `J` (`se3_tangent_to_ambient_jacobian`) via basis-vector loops every step, plus the per-step SVD re-projection (§4.2, point 3) that IEKF never needs. So the accuracy cost (§4.3-4.4) and the speed benefit here are both real, but neither mirrors IEKF's clean "same accuracy, pure speed win" story from §2.4 - vanilla KF trades a *worse* estimate for a *partial*, not full, speed gain.

---

## 5. When would each actually diverge more?

The EKF/IEKF equivalence (§2) rests on **isotropic noise**, not on rigidity per se. Two ways it can break:

1. **Anisotropic sensor noise.** If measurement noise is direction-dependent in a *fixed world-frame* sense (e.g. a sensor that's noisier along the world's vertical axis than horizontal, regardless of the object's orientation), `R_diag` no longer commutes with `Rbig`, and EKF/IEKF give different corrections.

2. **Non-rigid deformation tied to the object's own frame.** A rigid-body tracker can absorb small deformation as extra "noise" on top of the rigid assumption. If that wobble is itself isotropic (any-direction jitter), the two filters *still* match - isotropic noise is rotation-invariant regardless of its physical source. But real deformation is often *structured*: a flag flexing along its pole, a limb bending more along its length than sideways - an ellipsoid of uncertainty aligned with the object's own axes (its long axis, a hinge axis), not the world's.
   - **EKF (world frame)** sees that ellipsoid rotate with the object every step; if it doesn't re-derive its noise model to track that rotation, it's silently using the wrong noise shape.
   - **IEKF (body frame)** sees the same ellipsoid sitting still, since it's fixed relative to the object's own axes - it can use a fixed, correctly-shaped anisotropic covariance with no per-step rotation.
   - Here the two filters genuinely diverge, and **IEKF is arguably the more natural model**, not just the faster one.

3. **Genuinely non-rigid motion (not just noisy-around-a-rigid-mean).** If no single rigid transform `T` reasonably explains the point cloud's motion at all, the question "does EKF or IEKF do better" isn't quite well-posed - both filters' core assumption (`z_i = T.act(p_i) + noise` for one shared rigid `T`) has failed. That needs a richer state (pose plus some deformation/shape parameters), not a filter swap.

**Rule of thumb (EKF vs. IEKF):** it's not **rigid** vs. **non-rigid** that decides this - it's **whether the uncertainty is isotropic or anisotropic-and-tied-to-the-object's-frame**. Non-rigid objects are simply a natural, common source of the latter. None of this affects UKF's already-approximate agreement with either filter one way or the other, since §3's gap comes from linearization/noise-injection mechanics, not from the isotropy argument in §2.

**Rule of thumb (UKF vs. EKF/IEKF):** per §3.4, the gap widens with stronger per-step nonlinearity (bigger `dt`, faster rotation, larger corrections) and narrows as the per-step motion shrinks, over the range where that mechanism actually dominates (`dt=0.02` to `0.80` here) - but it does **not** keep shrinking all the way to zero as `dt -> 0`; below roughly `dt=0.02` a much smaller numerical floor (~6e-4-to-1e-3 m) takes over and the gap stops shrinking, sometimes growing back a little. It does not have an isotropy precondition the way the EKF/IEKF equivalence does, and it never becomes bit-identical the way EKF/IEKF are (there's always a residual first-vs-second-order gap) - but "it just shrinks toward zero" overstates the small-`dt` regime specifically; it shrinks toward a small floor, not to zero.

Vanilla KF's divergence (§4) doesn't belong on this isotropy spectrum at all - it isn't conditional on anisotropic noise the way the EKF/IEKF split is, and it isn't a linearization/sampling-mechanics gap the way the UKF split is. It comes from swapping out the state representation itself (ambient vs. minimal, §4.2/[A.5](#a5-ambientredundant-vs-minimal-state-parameterization)), and per §4's default-isotropic-noise measurements, it's already fully present without ever touching the noise model.

**Rule of thumb (Vanilla KF vs. the other three):** unlike UKF's gap, which shrinks cleanly with `dt` over its own dominant range before hitting a small numerical floor, this one has no isotropy precondition and no shrink-toward-zero trend *anywhere* in the tested range (§4.4) - it's present unconditionally, at a level several times above UKF's floor, from a change of state representation rather than a choice of linearization, frame, or sampling scheme.

---

## Appendix: Related terms

Short definitions of a few terms this doc leans on, gathered in one place rather than left implicit in §2/§4/§5.

### A.1 Isotropic and anisotropic noise

**Isotropic** noise/uncertainty is the same in every direction: a covariance of the form `Sigma = sigma^2 * I` (equal variance on every axis, zero cross-correlation), which geometrically is a perfect sphere. **Anisotropic** noise is direction-dependent - unequal diagonal entries and/or nonzero off-diagonal terms, an ellipsoid rather than a sphere. This is what §2 and §4 above lean on: a sphere looks identical after any rotation (`R @ (sigma^2 I) @ R.T = sigma^2 I`), which is the exact algebraic step that makes EKF's world-frame and IEKF's body-frame corrections agree.

### A.2 Invariance and equivariance

An error (or a system) is **invariant** to a transformation if it doesn't change when that transformation is applied. `run_iekf`'s body-frame residual `T_pred^-1.act(z_i) - p_i` is **left-invariant** in exactly this sense - it's unaffected by redefining the world frame (left-multiplying both the true and estimated pose by the same fixed transform). See [left_right_invariant.md](left_right_invariant.md) for the full left- vs. right-invariant treatment, including why body-frame point-cloud measurements (this script's case) naturally pair with the left-invariant choice.

### A.3 Mahalanobis distance and the information matrix

The **Mahalanobis distance** of a residual is its Euclidean distance rescaled by the inverse covariance (the **information matrix** `Omega = Sigma^-1`): `d^2 = r.T @ Omega @ r` - "how many standard deviations away, accounting for the noise model's shape." Isotropic noise is the special case where that rescaling doesn't distort anything: `Omega = I / sigma^2` is just a uniform scale factor, so Mahalanobis distance reduces to plain Euclidean distance divided by `sigma`. See [pose_graph_optimization.md §15.3](../optimization/pose_graph_optimization.md#153-objective-function) for the general definition and its use as a pose-graph objective function.

### A.4 Scalar, diagonal, and full covariance matrices

Covariance matrices used in this repo fall into three shapes, in increasing generality: **scalar** (`sigma^2 * I`, e.g. this script's own `R_diag` - isotropic, the A.1 case), **diagonal** (a different variance per axis but no cross-correlation - anisotropic but still axis-aligned), and **full** (an arbitrary positive-definite matrix, with nonzero off-diagonal terms encoding correlation between axes - anisotropic and not necessarily aligned with any coordinate axis). Only the scalar case is guaranteed to commute with an arbitrary rotation, which is why §2's EKF/IEKF equivalence needs isotropic (scalar) noise specifically, not merely "diagonal."

### A.5 Ambient/redundant vs. minimal state parameterization

A **minimal** parameterization uses exactly as many numbers as the state has degrees of freedom - `SE(3)`'s 6-dim tangent vector `[v, w]` for a 6-DoF pose, which is what `run_ekf`/`run_iekf`/`run_ukf`/`run_batch_gn`'s covariances and increments all use. An **ambient** (or **redundant**) parameterization uses more numbers than there are degrees of freedom - `run_vanilla_kf`'s 12-dim `x = [vec(R) (9,), t (3,)]` for the same 6-DoF pose, 9 numbers standing in for a 3-DoF rotation.

Going ambient is what buys §4.1's exact linearity of the observation model, but it comes with two costs that a minimal parameterization never needs: an explicit, external step to keep the redundant numbers consistent with the manifold they're supposed to represent (§4.2's SVD re-projection - nothing inside a linear KF's own math keeps a 9-vector orthonormal), and a covariance that lives in the redundant ambient space rather than the state's true 6-dim tangent space, so it isn't directly comparable dimension-for-dimension to the other methods' 6x6 `P`.

This is a different axis from A.4: A.4 is about covariance *generality* (scalar/diagonal/full) at a *fixed* state dimension; A.5 is about the *dimensionality and redundancy of the state itself*, independent of what shape of covariance you'd put on it.

---

## References

1. Julier, S. J., & Uhlmann, J. K. (1997). *A New Extension of the Kalman Filter to Nonlinear Systems*. Proceedings of SPIE, 3068 (Signal Processing, Sensor Fusion, and Target Recognition VI), 182-193. - the original unscented transform/UKF this doc's §3 empirically compares against EKF/IEKF.
