# Why EKF and IEKF produce identical output in `pointcloud_pose_tracking.py`

Applies to both `use_numpy/pointcloud_pose_tracking.py` and
`use_manif/pointcloud_pose_tracking.py` - `run_ekf` and `run_iekf` produce
bit-identical (to ~1e-14) pose estimates on this benchmark, verified across
multiple seeds and stress parameters. This is a proven property of the
problem setup, not a coincidence, and it stops holding once certain
assumptions break. This doc records the reasoning for later reference.

## 1. The setup

- **State**: a single rigid pose `T` (SE(3)).
- **Motion model**: `T_pred = T_prev (+) Exp(twist * dt)` - constant
  body-frame twist. Composing with this known relative motion makes the
  predict step's linearization exact, not first-order (loosely "group-affine"
  in spirit, though that term technically describes richer coupled systems
  like IMU position/velocity/attitude propagation) - true for both filters,
  not part of the equivalence argument below.
- **Observation model**: a fixed body-frame point cloud `p_i`, observed as
  `z_i = T.act(p_i) + noise`, with **isotropic** Gaussian noise
  (`R = sigma^2 * I`, same variance in every direction, uncorrelated across
  x/y/z).

Two ways to run the measurement update:

- **EKF**: residual and Jacobian expressed in the **world frame**:
  `r_world = z_i - T_pred.act(p_i)`, `H_world = R_pred @ [I | -skew(p_i)]`
  (rebuilt every step from the current rotation estimate `R_pred`).
- **IEKF**: residual and Jacobian expressed in the **object's own body
  frame**: `r_body = T_pred^-1.act(z_i) - p_i`, `H_body = [I | -skew(p_i)]`
  (fixed - depends only on the object's known geometry, precomputed once).

---

## 2. Intuitive explanation

Both filters are answering the same question - "how far off is my predicted point cloud from what I actually measured, and what pose correction explains that gap?" - they just *describe* the mismatch in different coordinate systems:
 - **EKF** reports it in world coordinates ("2cm too far east"); 
 - **IEKF** reports the same physical mismatch in the object's own coordinates ("2.2cm too far toward the object's nose").

Converting between the two is just applying the current rotation estimate -
a rigid relabeling of axes that doesn't stretch or distort anything. As long
as both the *error* and the *sensitivity* (how a pose tweak would move the
points) are converted consistently, the real-world correction you get back
is the same either way - like reporting a distance in miles vs. km and
converting back.

The one thing that *could* break this is if the measurement noise "looked
different" depending on which direction you're facing (e.g. a sensor
noisier sideways than in depth). But the noise here is **isotropic** - a
perfect sphere of uncertainty around each point - and a sphere looks
identical no matter how you rotate it. That's the actual ingredient that
makes the two filters land on bit-identical corrections every step: rotating
an isotropic covariance leaves it unchanged (`R @ (sigma^2 I) @ R.T =
sigma^2 I` for any rotation `R`).

---

## 3. The algebra (for the curious)

Per point $p_i$:

```
r_body_i = T_pred^-1.act(z_i) - p_i = R_pred.T @ (z_i - T_pred.act(p_i)) = R_pred.T @ r_world_i

H_world  = R_pred @ H_body
```

Stack over all `M` points; let `Rbig` = block-diagonal repeat of `R_pred`,
`M` times (still orthogonal). Then `r_body = Rbig.T @ r_world` and
`H_world = Rbig @ H_body`. Push this through the Kalman update:

```
S_world = H_world @ P @ H_world.T + R_diag
        = Rbig @ (H_body @ P @ H_body.T + R_diag) @ Rbig.T   [needs Rbig @ R_diag @ Rbig.T = R_diag]
        = Rbig @ S_body @ Rbig.T

K_world = P @ H_world.T @ S_world^-1
        = P @ H_body.T @ Rbig.T @ (Rbig @ S_body @ Rbig.T)^-1
        = K_body @ Rbig.T

delta_world = K_world @ r_world = K_body @ Rbig.T @ r_world = K_body @ r_body = delta_body
```

The step `Rbig @ R_diag @ Rbig.T = R_diag` is exactly where isotropy is
used - it's the only place the argument could fail. `P` and `T_est` update
identically thereafter, every step, so the two trajectories never diverge.

---

## 4. Empirical verification

- Printed final/RMS rotation+position error rows are identical between EKF
  and IEKF across seed 0 (default args) and stress tests
  (`--init-pose-noise-std 0.5`/`0.8`, `--duration 8`).
- Direct numerical diff of the full trajectory (duration 5.0, seed 0, 51
  poses): max position diff ~1.7e-14 m, max rotation diff ~9.1e-15 rad -
  floating-point noise, not a real difference.
- Saved trajectory/error plots show the IEKF line drawn exactly on top of
  the EKF line everywhere (invisible because identical).

---

## 5. What actually differs between them: speed, not accuracy

`H_body` is a fixed matrix (depends only on the object's known geometry),
computed once outside the step loop. `H_world` depends on `R_pred`, the
*current* rotation estimate, so EKF rebuilds it every step. Measured ~30%
faster per step for IEKF (e.g. 260 vs 369 microseconds/step in one run) at
identical memory - the entire practical benefit on this benchmark.

---

## 6. When would EKF and IEKF actually diverge?

The equivalence rests on **isotropic noise**, not on rigidity per se. Two
ways it can break:

1. **Anisotropic sensor noise.** If measurement noise is direction-dependent
   in a *fixed world-frame* sense (e.g. a sensor that's noisier along the
   world's vertical axis than horizontal, regardless of the object's
   orientation), `R_diag` no longer commutes with `Rbig`, and EKF/IEKF give
   different corrections.

2. **Non-rigid deformation tied to the object's own frame.** A rigid-body
   tracker can absorb small deformation as extra "noise" on top of the rigid
   assumption. If that wobble is itself isotropic (any-direction jitter),
   the two filters *still* match - isotropic noise is rotation-invariant
   regardless of its physical source. But real deformation is often
   *structured*: a flag flexing along its pole, a limb bending more along
   its length than sideways - an ellipsoid of uncertainty aligned with the
   object's own axes (its long axis, a hinge axis), not the world's.
   - EKF (world frame) sees that ellipsoid rotate with the object every
     step; if it doesn't re-derive its noise model to track that rotation,
     it's silently using the wrong noise shape.
   - IEKF (body frame) sees the same ellipsoid sitting still, since it's
     fixed relative to the object's own axes - it can use a fixed,
     correctly-shaped anisotropic covariance with no per-step rotation.
   - Here the two filters genuinely diverge, and IEKF is arguably the more
     natural model, not just the faster one.

3. **Genuinely non-rigid motion (not just noisy-around-a-rigid-mean).** If
   no single rigid transform `T` reasonably explains the point cloud's
   motion at all, the question "does EKF or IEKF do better" isn't quite
   well-posed - both filters' core assumption (`z_i = T.act(p_i) + noise`
   for one shared rigid `T`) has failed. That needs a richer state (pose
   plus some deformation/shape parameters), not a filter swap.

**Rule of thumb:** it's not rigid vs. non-rigid that decides this - it's
whether the uncertainty is isotropic or anisotropic-and-tied-to-the-object's-frame.
Non-rigid objects are simply a natural, common source of the latter.
