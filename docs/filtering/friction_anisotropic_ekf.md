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

> Anisotropic-friction slip variance is a function of heading relative to the pad's fixed friction axis, not just which discrete contact mode is active - a mode-switching saltation matrix with an otherwise-isotropic $Q$ would treat slip along the low-friction axis the same as the high-friction axis

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

### 1.1 The filter math, concretely

All three variants run the same EKF over $x = [p_x, p_y, \theta]^\top$. Each tick is one **predict step** followed by one **position update** (`run_ekf`). The only difference between the variants is how the position block of $Q$ is built inside the **predict step**.

**Predict, mean and Jacobian** (`exact_arc_step`). With constant commanded speed $v$ and turn rate $\omega$, the robot moves along an exact circular arc of radius $r = v/\omega$:

$$
\theta^{-} = \theta + \omega\,\Delta t, \qquad
p_x^{-} = p_x + r\left(\sin\theta^{-} - \sin\theta\right), \qquad
p_y^{-} = p_y - r\left(\cos\theta^{-} - \cos\theta\right)
$$

$$
F = \frac{\partial x^{-}}{\partial x} =
\begin{bmatrix}
1 & 0 & r\left(\cos\theta^{-} - \cos\theta\right) \\
0 & 1 & r\left(\sin\theta^{-} - \sin\theta\right) \\
0 & 0 & 1
\end{bmatrix}
$$

The mean is propagated exactly, with no linearization. $F$ is still needed because heading enters the position update nonlinearly: its third column says how an error in $\theta$ turns into a position error over one step. When $|\omega| < 10^{-9}$ the code switches to the straight-line limit, $`p^{-} = p + v\,\Delta t\,(\cos\theta, \sin\theta)`$, to avoid dividing by $\omega$.

**Predict, covariance** (`predict`):

$$
P^{-} = F\,P\,F^\top + Q, \qquad
Q = \begin{bmatrix} Q_{\text{pos}} & 0 \\ 0 & (\sigma_\theta\,\Delta t)^2 \end{bmatrix}
$$

$Q_{\text{pos}}$ is the 2×2 position block, and it is the only thing that changes between variants. Two functions build it: `isotropic_Q_pos` (a circle) and `anisotropic_Q_pos` (a rotated ellipse). The three variants are then defined by which function they call and, for the ellipse, which heading they pass in.

**Isotropic block** (`isotropic_Q_pos`). The same variance in every direction:

$$
Q_{\text{pos}}^{\text{iso}} = \sigma_{\text{iso}}^2\,\Delta t^2\,I_2,
\qquad
\sigma_{\text{iso}}^2 = \frac{\sigma_{\text{grip}}^2 + \sigma_{\text{slip}}^2}{2}
$$

$\sigma_{\text{iso}}^2$ is the average of the two body-frame slip variances. That choice gives the circle the same total variance (trace) as the ellipse below, so the isotropic variant isn't handicapped by assuming more or less noise overall - it only lacks direction. Because a circle looks the same at every rotation, this block needs no heading at all.

**Anisotropic block** (`anisotropic_Q_pos`). It starts from an ellipse in the pad's own body frame (small variance along the grip/forward axis, large variance along the slip/lateral axis) and rotates it into the world frame by some heading $\theta_Q$:

$$
Q_{\text{pos}}^{\text{aniso}}(\theta_Q) = R(\theta_Q)
\begin{bmatrix} \sigma_{\text{grip}}^2 & 0 \\ 0 & \sigma_{\text{slip}}^2 \end{bmatrix}
R(\theta_Q)^\top \Delta t^2,
\qquad
R(\theta_Q) = \begin{bmatrix} \cos\theta_Q & -\sin\theta_Q \\ \sin\theta_Q & \cos\theta_Q \end{bmatrix}
$$

The $\Delta t^2$ converts a slip-velocity standard deviation (m/s) into a per-tick position variance (m²). Written out, with $a = \sigma_{\text{grip}}^2 \Delta t^2$, $b = \sigma_{\text{slip}}^2 \Delta t^2$, $c = \cos\theta_Q$ and $s = \sin\theta_Q$:

$$
Q_{\text{pos}}^{\text{aniso}}(\theta_Q) =
\begin{bmatrix}
a c^2 + b s^2 & (a - b)\,c s \\
(a - b)\,c s & a s^2 + b c^2
\end{bmatrix}
$$

This function doesn't decide which heading to use - the caller does. That choice is what separates the two anisotropic variants.

**The three variants.** At predict step $k$ (propagating from tick $k-1$ to tick $k$):

- **`isotropic`**: uses $Q_{\text{pos}}^{\text{iso}}$ every step. It ignores the pad's anisotropy entirely.
- **`fixed_anisotropic`**: uses $Q_{\text{pos}}^{\text{aniso}}(\theta_{\text{ref}})$ every step, where $\theta_{\text{ref}}$ is the true heading at $t = 0$ (`x_true[0, 2]`, which is $0$ in the default run). $\theta_{\text{ref}}$ never changes, so the ellipse keeps pointing in its initial direction while the robot turns underneath it.
- **`heading_aware`**: uses $Q_{\text{pos}}^{\text{aniso}}(\hat\theta_{k-1})$, where $\hat\theta_{k-1}$ is the filter's own heading estimate after the previous tick's update (`x[2]` before this step's propagation). The ellipse is re-oriented every step to follow the estimated heading. It uses the estimate, not the true heading, because a real filter has no access to the truth. It also uses the heading at the *start* of the step rather than the propagated $\theta^{-}$; with the defaults the heading changes by only $\omega\Delta t \approx 0.9°$ per step, so this makes no practical difference.

In code, all three are a single branch in `predict` on `q_policy`:

| `q_policy` | $Q_{\text{pos}}$ | Heading used to orient it |
| --- | --- | --- |
| `isotropic` | $`Q_{\text{pos}}^{\text{iso}}`$ | none (a circle) |
| `fixed_anisotropic` | $`Q_{\text{pos}}^{\text{aniso}}(\theta_{\text{ref}})`$ | $\theta_{\text{ref}}$, the true heading at $t = 0$, never updated |
| `heading_aware` | $`Q_{\text{pos}}^{\text{aniso}}(\hat\theta_{k-1})`$ | $\hat\theta_{k-1}$, the filter's own heading estimate at the start of the step |

This mirrors how the simulator generates the truth (`generate_ground_truth_and_data`): each tick it draws slip in the body frame with standard deviations $\sigma_{\text{grip}}$ and $\sigma_{\text{slip}}$ (scaled by $\Delta t$) and rotates it by the *true* heading. `heading_aware` is the only variant whose noise model has the same shape and orientation as that process, apart from its own heading-estimate error.

**Position update** (`measurement_update`). Only position is measured:

$$
H = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}, \qquad R = \sigma_{\text{pos}}^2 I
$$

$$
r = z - H x^{-}, \qquad S = H P^{-} H^\top + R, \qquad K = P^{-} H^\top S^{-1}, \qquad
x^{+} = x^{-} + K r, \qquad P^{+} = (I - K H)\,P^{-}
$$

Heading is never measured directly. It gets corrected only through the position-heading cross-covariance that $F$'s third column builds up in $P^{-}$. That is where `heading_aware`'s $\hat\theta_{k-1}$ comes from.

Script defaults: $\sigma_{\text{grip}} = 0.02$ m/s, $\sigma_{\text{slip}} = 0.1$ m/s, $\sigma_\theta = 0.01$ rad/s, $\sigma_{\text{pos}} = 0.02$ m, $v = 0.2$ m/s, $\Delta t = 0.05$ s. The true heading is noise-free, so the small $\sigma_\theta$ term only keeps the filter's heading covariance from collapsing to zero.

Three things follow directly from these formulas:

- **All three variants carry the same total noise.** Rotation doesn't change a matrix's trace, so $Q_{\text{pos}}^{\text{aniso}}$ always has trace $(\sigma_{\text{grip}}^2 + \sigma_{\text{slip}}^2)\Delta t^2$. That is exactly $2\sigma_{\text{iso}}^2 \Delta t^2$, the trace of the isotropic version. The variants differ only in *direction*, not in how much noise they assume overall.
- **The off-diagonal term is what points the ellipse.** It is zero only when $\theta_Q$ is a multiple of 90°. With the defaults, the slip variance is 25 times the grip variance (a 5:1 ratio in standard deviation), so pointing the ellipse the wrong way matters a lot. When `fixed_anisotropic`'s ellipse is misaligned, the filter assumes little noise in a direction where the real slip is large. It becomes overconfident in exactly that direction, which drives the NEES gap in §2.
- **A 180° error costs nothing.** Replacing $\theta_Q$ with $\theta_Q + \pi$ flips the sign of both $c$ and $s$, which leaves $c^2$, $s^2$ and $cs$ unchanged. So $Q_{\text{pos}}^{\text{aniso}}$ repeats every 180°, which is the periodicity §3 builds on.

## 2. The dominant finding: `fixed_anisotropic` is dramatically worse, everywhere

Monte Carlo NEES (500 trials, seed 0, $dt = 0.05\,\text{s}$ - the script's own default, verified to reproduce the table below to two significant figures - one full loop over $20\,\text{s}$), binned by how far the true heading has rotated away from `fixed_anisotropic`'s fixed reference heading:

| Rotation away from reference | `isotropic` | `fixed_anisotropic` | `heading_aware` |
| --- | --- | --- | --- |
| 0-45° | 2.9 | 4.8 | 2.6 |
| 45-90° | 3.0 | 8.3 | 2.5 |
| 90-135° | 3.5 | 8.9 | 3.3 |
| 135-180° | 3.1 | 6.6 | 2.8 |

(A consistent 3-DoF filter should average $\text{NEES} \approx 3$ everywhere; `isotropic` and `heading_aware` both sit close to that at every checkpoint. `fixed_anisotropic` clears the $\chi^2$ 95% bound ($7.8$) in two of the four bins.)

This part is robust: verified across seeds 0-3, `fixed_anisotropic` is worse than *both* alternatives at *every* checkpoint, by a wide margin, in every seed tried - not just "eventually," from the very first bin. RMS position error tells a smaller but consistent story too: $`\text{heading\_aware}\ (0.0108\,\text{m}) < \text{isotropic}\ (0.0118\,\text{m}) < \text{fixed\_anisotropic}\ (0.0131\,\text{m})`$ - getting the shape right and pointed the right way is both more accurate and, as the table shows, far better calibrated.

## 3. A secondary finding, and where it stops being clean

The *worst* of `fixed_anisotropic`'s own four checkpoints is the 90-135° bin in three of the four seeds tried (0, 1, 3), not the largest possible mismatch (135-180°) - seed 2 is the exception (§ below). The reason isn't an accident: a covariance ellipse $R(\theta)\,\mathrm{diag}(a,b)\,R(\theta)^\top$ has period $\pi$ in $\theta$, not $2\pi$ - rotating it by 180° gives back the identical ellipse. So a heading mismatch of 180° is, for the *orientation of the noise ellipse specifically*, no mismatch at all; the worst possible ellipse-orientation mismatch is at 90°, exactly where the empirical peak sits for those three seeds.

What happens **beyond** that peak, heading back out toward a full 180° difference, is *not* a clean story, and this doc says so rather than overselling one: in seeds 0, 1, and 3, `fixed_anisotropic`'s NEES does partially recover in the 135-180° bin (matching the ellipse-symmetry prediction); in seed 2, it keeps climbing all the way through. The most likely explanation is that the pure instantaneous-orientation-mismatch effect (which the ellipse-symmetry argument correctly predicts) is competing with a second, accumulated-trajectory-drift effect that grows with elapsed time/distance regardless of instantaneous heading - and depending on the particular noise realization, either one can dominate by the time a full loop has been driven. The NEES-vs-time plot shows this concretely: `fixed_anisotropic` (red) has repeated, roughly periodic bumps over the $20\,\text{s}$ loop rather than one clean single-peaked hump, consistent with a real periodic effect that isn't the *only* thing going on.

## 4. Takeaway for a real friction-anisotropic contact model

Both findings point the same direction for modeling a friction-anisotropic pad's slip in a real filter: the noise model needs to track the *current* heading, not just "know" the pad is anisotropic in the abstract. Getting the shape right but the orientation wrong (`fixed_anisotropic`) isn't a small, forgivable approximation - it's dramatically worse than the naive isotropic fallback almost everywhere, precisely because it makes a confident, specific directional claim that's actively false most of the time, and (per §3) that claim is worst exactly at the 90° mismatch a fixed reference heading is most likely to drift into. The safe fallback, if heading-tracking isn't available for some reason, is to not claim a direction at all (`isotropic`) rather than claim the wrong one.

---

## 5. References

1. Hu, D. L., Nirody, J., Scott, T., & Shelley, M. J. (2009). *The Mechanics of Slithering Locomotion*. Proceedings of the National Academy of Sciences, 106(25), 10081-10085. https://doi.org/10.1073/pnas.0812533106 - the source of this doc's opening image (a snake's belly scales) and the physical phenomenon it models: friction fixed to the body/scale frame with different coefficients along vs. across the body axis. The paper's friction model is deterministic and doesn't itself use a stochastic slip-variance/covariance framing - that translation into an EKF process-noise ellipse is this doc's own extension, applying the world-frame-rotation argument from [`pointcloud_pose_tracking_empirical_note.md`](pointcloud_pose_tracking_empirical_note.md) (see §Intuition above) to process rather than measurement noise.
