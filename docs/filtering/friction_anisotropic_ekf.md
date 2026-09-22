# Heading-dependent process noise for friction-anisotropic locomotion

## Intuition

"Anisotropic friction" means grips well in one direction, slides easily in the other - the canonical image is a snake's belly scales, or a ratchet:

```
        low-friction (slide) axis
                 ↑
                 │
  grip axis ──── ● ──── grip axis      (small slip noise along grip axis,
                 │                      large slip noise along slide axis)
                 ↓
        low-friction (slide) axis
```

That noise ellipse is fixed *to the pad*, i.e. to the robot's body frame. The catch this doc explores: as the robot turns, that ellipse turns with it in the world frame -

```
heading = 0°:        heading = 45°:       heading = 90°:
   ⬭ (flat)             ⬮ (tilted)            |‾| (rotated 90°)
```

So a filter whose process-noise model uses a *fixed* orientation for that ellipse (set once, e.g. at $t = 0$) is only correct at the instant it was set - as soon as the robot turns, it's confidently modeling slip in the wrong direction. The toy problem below (a unicycle looping through every heading) is built to expose exactly that: `fixed_anisotropic` (wrong orientation once turned) vs. `heading_aware` (ellipse re-oriented every step to match current heading) vs. `isotropic` (no directional claim at all, the safe fallback).

Consider this claim about friction-anisotropic platforms (pads that grip well in one direction and slide easily in another - "the way a snake's belly scales do"): 

> Anisotropic-friction slip variance is a function of heading relative to the pad's fixed friction axis (refs. 19–21 characterize this directly), not just which discrete contact mode is active - a mode-switching saltation matrix with an otherwise-isotropic $Q$ would treat slip along the low-friction axis the same as the high-friction axis

This doc works through the simplest concrete version of that claim, using [`friction_anisotropic_ekf.py`](../../use_numpy/friction_anisotropic_ekf.py)'s crawling unicycle as the toy problem.

**This isn't new theory** - it's a direct application of an argument this repo already made, just applied on the other side of the filter. [`pointcloud_pose_tracking_empirical_note.md`](pointcloud_pose_tracking_empirical_note.md) §A.1/§2 established that a noise ellipsoid fixed in an object's own frame looks anisotropic-and-rotating once expressed in the world frame, and that what actually matters is whether the filter's noise model tracks that rotation - not "rigid vs. non-rigid." That note was about *measurement* noise in a point-cloud registration EKF/IEKF. This script is the first to apply the identical argument to *process* noise for a robot that's actually moving and turning, and deliberately stays an ordinary EKF throughout - no new IEKF/Lie-group derivation, since the question here is purely "does $Q$'s shape track heading," not a linearization-frame question.

**Scope, stated up front**: heading itself is exact and noise-free, driven only by a known, constant commanded turn rate. Only the *position* picks up random friction-anisotropic slip. This isolates the one question this script is about (does the process-noise ellipse's orientation track heading correctly) from a second one (heading estimation itself), which this toy deliberately doesn't touch.

## 1. The setup

A unicycle robot drives at a *known* constant commanded forward speed $v_{\text{cmd}}$ and turn rate $\omega_{\text{cmd}}$, so its true path is an exact circular arc (`exact_arc_step`, closed-form, exact for any $dt$ - the same "exact propagation, not a linearization" convention as `saltation_matrix_ekf.flow`, though an EKF Jacobian is still needed here since heading enters the position update nonlinearly). $\omega_{\text{cmd}}$ defaults to $\dfrac{2\pi}{T}$ (where $T$ is the run duration), i.e. exactly one full loop over the run - so the heading sweeps through every possible orientation relative to a fixed reference.

On top of that exact commanded arc, true position picks up a random slip disturbance each tick, drawn in the **pad's own body frame** - low variance along the grip/forward axis, high variance along the slip/lateral axis - then rotated into world coordinates by the **true** heading at that instant.

Three ways of building the position block of the filter's process-noise covariance $Q$ are compared, all sharing the identical predict step and the same noisy position measurement each tick:

| `q_policy` | Shape | Orientation |
| --- | --- | --- |
| `isotropic` | Direction-blind | N/A - same total noise budget as the other two (a fair comparison, not a rigged one: $\sigma_{\text{iso}}^2 = \dfrac{\sigma_{\text{grip}}^2 + \sigma_{\text{slip}}^2}{2}$) |
| `fixed_anisotropic` | Correctly anisotropic | Fixed to the heading at $t=0$, never updated - the mistake of knowing the pad is anisotropic without re-deriving $Q$ as the robot turns |
| `heading_aware` | Correctly anisotropic | Re-oriented every predict step using the filter's own current heading estimate - the correct policy |

## 2. The dominant finding: `fixed_anisotropic` is dramatically worse, everywhere

Monte Carlo NEES (500 trials, seed 0, $dt = 0.02\,\text{s}$, one full loop over $20\,\text{s}$), binned by how far the true heading has rotated away from `fixed_anisotropic`'s fixed reference heading:

| Rotation away from reference | `isotropic` | `fixed_anisotropic` | `heading_aware` |
| --- | --- | --- | --- |
| 0-45° | 2.9 | 4.8 | 2.6 |
| 45-90° | 3.0 | 8.3 | 2.5 |
| 90-135° | 3.5 | 8.9 | 3.3 |
| 135-180° | 3.1 | 6.6 | 2.8 |

(A consistent 3-DoF filter should average $\text{NEES} \approx 3$ everywhere; `isotropic` and `heading_aware` both sit close to that at every checkpoint. `fixed_anisotropic` clears the $\chi^2$ 95% bound ($7.8$) in two of the four bins.)

This part is robust: verified across seeds 0-3, `fixed_anisotropic` is worse than *both* alternatives at *every* checkpoint, by a wide margin, in every seed tried - not just "eventually," from the very first bin. RMS position error tells a smaller but consistent story too: $`\text{heading\_aware}\ (0.0108\,\text{m}) < \text{isotropic}\ (0.0118\,\text{m}) < \text{fixed\_anisotropic}\ (0.0131\,\text{m})`$ - getting the shape right and pointed the right way is both more accurate and, as the table shows, far better calibrated.

## 3. A secondary finding, and where it stops being clean

The *worst* of `fixed_anisotropic`'s own four checkpoints is consistently the 90-135° bin, not the largest possible mismatch (135-180°) - this also held across all four seeds tried. The reason isn't an accident: a covariance ellipse $R(\theta)\,\mathrm{diag}(a,b)\,R(\theta)^\top$ has period $\pi$ in $\theta$, not $2\pi$ - rotating it by 180° gives back the identical ellipse. So a heading mismatch of 180° is, for the *orientation of the noise ellipse specifically*, no mismatch at all; the worst possible ellipse-orientation mismatch is at 90°, exactly where the empirical peak sits.

What happens **beyond** that peak, heading back out toward a full 180° difference, is *not* a clean story, and this doc says so rather than overselling one: in seeds 0, 1, and 3, `fixed_anisotropic`'s NEES does partially recover in the 135-180° bin (matching the ellipse-symmetry prediction); in seed 2, it keeps climbing all the way through. The most likely explanation is that the pure instantaneous-orientation-mismatch effect (which the ellipse-symmetry argument correctly predicts) is competing with a second, accumulated-trajectory-drift effect that grows with elapsed time/distance regardless of instantaneous heading - and depending on the particular noise realization, either one can dominate by the time a full loop has been driven. The NEES-vs-time plot shows this concretely: `fixed_anisotropic` (red) has repeated, roughly periodic bumps over the $20\,\text{s}$ loop rather than one clean single-peaked hump, consistent with a real periodic effect that isn't the *only* thing going on.

## 4. Takeaway for a real friction-anisotropic contact model

Both findings point the same direction for modeling a friction-anisotropic pad's slip in a real filter: the noise model needs to track the *current* heading, not just "know" the pad is anisotropic in the abstract. Getting the shape right but the orientation wrong (`fixed_anisotropic`) isn't a small, forgivable approximation - it's dramatically worse than the naive isotropic fallback almost everywhere, precisely because it makes a confident, specific directional claim that's actively false most of the time, and (per §3) that claim is worst exactly at the 90° mismatch a fixed reference heading is most likely to drift into. The safe fallback, if heading-tracking isn't available for some reason, is to not claim a direction at all (`isotropic`) rather than claim the wrong one.
