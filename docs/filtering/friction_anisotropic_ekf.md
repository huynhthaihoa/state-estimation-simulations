# Heading-dependent process noise for friction-anisotropic locomotion

## Intuition

"Anisotropic friction" means a surface grips well along one axis and slides easily along the perpendicular one. The classic biological example is a snake's belly scales, which slide easily along the body and grip sideways (Hu et al. 2009). This doc's pad is set up the other way round: it grips along its forward axis and slips sideways, as in the drawing below. Which axis is which doesn't matter to the argument; swapping them only rotates the noise ellipse by 90°.

```text
        low-friction (slide) axis
                 ↑
                 │
  grip axis ──── ● ──── grip axis      (small slip noise along grip axis,
                 │                      large slip noise along slide axis)
                 ↓
        low-friction (slide) axis
```

That noise ellipse is fixed *to the pad*, i.e. to the robot's body frame. The catch this doc explores: as the robot turns, that ellipse turns with it in the world frame. Drawing the ellipse's long (slip) axis in world coordinates:

```text
heading = 0°        heading = 45°        heading = 90°
     ↕                    ⤡                    ↔
 (slip along y)    (slip along the       (slip along x)
                     diagonal)
```

So a filter whose process-noise model uses a *fixed* orientation for that ellipse (set once, e.g. at $t = 0$) is only correct at the instant it was set - as soon as the robot turns, it's confidently modeling slip in the wrong direction. The toy problem below (a unicycle looping through every heading) is built to expose exactly that: `fixed_anisotropic` (wrong orientation once turned) vs. `heading_aware` (ellipse re-oriented every step to match current heading) vs. `isotropic` (no directional claim at all, the safe fallback).

The question behind this comes from hybrid-systems filtering. There, friction is usually modeled as discrete contact modes (stuck or sliding), each switch with its own saltation matrix. Kong et al. (2024, §V-F) derive the saltation matrix for this stick-slip transition under Coulomb friction, whose single coefficient $\mu$ has no preferred direction. For an anisotropic pad, that's not enough: slip variance depends on heading relative to the pad's own friction axes, not only on which mode is active. A filter that switches modes correctly but keeps an isotropic $Q$ treats slip along the low-friction axis the same as along the high-friction axis.

This doc works through the simplest concrete version of that point, using [`friction_anisotropic_ekf.py`](../../use_numpy/friction_anisotropic_ekf.py)'s crawling unicycle as the toy problem.

![Three panels from friction_anisotropic_ekf.py's defaults: the robot's body frame with the pad's grip axis along the heading and its slip axis sideways; one full circular loop in the world frame, where the true slip ellipse turns with the heading while a fixed_anisotropic ellipse stays at the initial heading; and the isotropic, fixed_anisotropic and heading_aware process-noise ellipses against the true slip covariance at a 90° heading](../../assets/friction_anisotropic_ekf_concept.png)

*Figure: the setup at `use_numpy/friction_anisotropic_ekf.py`'s defaults, plotted by `uv run python assets/make_figures.py friction_anisotropic_ekf_concept`. Ellipses are enlarged to be visible; their 5:1 axis ratio is exact.*

**This isn't new theory** - it's a direct application of an argument this repo already made, just applied on the other side of the filter. [`pointcloud_pose_tracking_empirical_note.md`](pointcloud_pose_tracking_empirical_note.md) §2/§5 (terms in the [glossary](../glossary.md#1-geometry-and-lie-groups)) established that a noise ellipsoid fixed in an object's own frame looks anisotropic-and-rotating once expressed in the world frame, and that what actually matters is whether the filter's noise model tracks that rotation - not "rigid vs. non-rigid." That note was about *measurement* noise in a point-cloud registration EKF/IEKF. This script is the first to apply the identical argument to *process* noise for a robot that's actually moving and turning, and deliberately stays an ordinary EKF throughout - no new IEKF/Lie-group derivation, since the question here is purely "does $Q$'s shape track heading," not a linearization-frame question.

**Scope, stated up front**: heading itself is exact and noise-free, driven only by a known, constant commanded turn rate. Only the *position* picks up random friction-anisotropic slip. This isolates the one question this script is about (does the process-noise ellipse's orientation track heading correctly) from a second one (heading estimation itself), which this toy deliberately doesn't touch.

---

## 1. The setup

A unicycle robot drives at a *known* constant commanded forward speed $v_{\text{cmd}}$ and turn rate $\omega_{\text{cmd}}$, so its true path is an exact circular arc (`exact_arc_step`, closed-form, exact for any $dt$ - the same "exact propagation, not a linearization" convention as `saltation_matrix_ekf.flow`; §1.1 gives the formulas and the Jacobian). $\omega_{\text{cmd}}$ defaults to $\dfrac{2\pi}{T}$ (where $T$ is the run duration), i.e. exactly one full loop over the run - so the heading sweeps through every possible orientation relative to a fixed reference.

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

```math
\theta^{-} = \theta + \omega\,\Delta t, \qquad
p_x^{-} = p_x + r\left(\sin\theta^{-} - \sin\theta\right), \qquad
p_y^{-} = p_y - r\left(\cos\theta^{-} - \cos\theta\right)
```

```math
F = \frac{\partial x^{-}}{\partial x} =
\begin{bmatrix}
1 & 0 & r\left(\cos\theta^{-} - \cos\theta\right) \\
0 & 1 & r\left(\sin\theta^{-} - \sin\theta\right) \\
0 & 0 & 1
\end{bmatrix}
```

The mean is propagated exactly, with no linearization. $F$ is still needed because heading enters the position update nonlinearly: its third column says how an error in $\theta$ turns into a position error over one step. When $|\omega| < 10^{-9}$ the code switches to the straight-line limit, $`p^{-} = p + v\,\Delta t\,(\cos\theta, \sin\theta)`$, to avoid dividing by $\omega$.

**Predict, covariance** (`predict`):

```math
P^{-} = F\,P\,F^\top + Q, \qquad
Q = \begin{bmatrix} Q_{\text{pos}} & 0 \\ 0 & (\sigma_\theta\,\Delta t)^2 \end{bmatrix}
```

$Q_{\text{pos}}$ is the 2×2 position block, and it is the only thing that changes between variants. Two functions build it: `isotropic_Q_pos` (a circle) and `anisotropic_Q_pos` (a rotated ellipse). The three variants are then defined by which function they call and, for the ellipse, which heading they pass in.

**Isotropic block** (`isotropic_Q_pos`). The same variance in every direction:

```math
Q_{\text{pos}}^{\text{iso}} = \sigma_{\text{iso}}^2\,\Delta t^2\,I_2,
\qquad
\sigma_{\text{iso}}^2 = \frac{\sigma_{\text{grip}}^2 + \sigma_{\text{slip}}^2}{2}
```

$\sigma_{\text{iso}}^2$ is the average of the two body-frame slip variances. That choice gives the circle the same total variance (trace) as the ellipse below, so the isotropic variant isn't handicapped by assuming more or less noise overall - it only lacks direction. Because a circle looks the same at every rotation, this block needs no heading at all.

**Anisotropic block** (`anisotropic_Q_pos`). It starts from an ellipse in the pad's own body frame (small variance along the grip/forward axis, large variance along the slip/lateral axis) and rotates it into the world frame by some heading $\theta_Q$:

```math
Q_{\text{pos}}^{\text{aniso}}(\theta_Q) = R(\theta_Q)
\begin{bmatrix} \sigma_{\text{grip}}^2 & 0 \\ 0 & \sigma_{\text{slip}}^2 \end{bmatrix}
R(\theta_Q)^\top \Delta t^2,
\qquad
R(\theta_Q) = \begin{bmatrix} \cos\theta_Q & -\sin\theta_Q \\ \sin\theta_Q & \cos\theta_Q \end{bmatrix}
```

The $\Delta t^2$ converts a slip-velocity standard deviation (m/s) into a per-tick position variance (m²). This is a simple heuristic scaling (the script's comments flag it as one), not the $\sigma^2\Delta t$ of continuous-time white noise; it's a tuning choice, not what this doc studies. Written out, with $a = \sigma_{\text{grip}}^2 \Delta t^2$, $b = \sigma_{\text{slip}}^2 \Delta t^2$, $c = \cos\theta_Q$ and $s = \sin\theta_Q$:

```math
Q_{\text{pos}}^{\text{aniso}}(\theta_Q) =
\begin{bmatrix}
a c^2 + b s^2 & (a - b)\,c s \\
(a - b)\,c s & a s^2 + b c^2
\end{bmatrix}
```

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

```math
H = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}, \qquad R = \sigma_{\text{pos}}^2 I
```

```math
r = z - H x^{-}, \qquad S = H P^{-} H^\top + R, \qquad K = P^{-} H^\top S^{-1}, \qquad
x^{+} = x^{-} + K r, \qquad P^{+} = (I - K H)\,P^{-}
```

Heading is never measured directly. It gets corrected only through the position-heading cross-covariance that $F$'s third column builds up in $P^{-}$. That is where `heading_aware`'s $\hat\theta_{k-1}$ comes from.

Script defaults: $\sigma_{\text{grip}} = 0.02$ m/s, $\sigma_{\text{slip}} = 0.1$ m/s, $\sigma_\theta = 0.01$ rad/s, $\sigma_{\text{pos}} = 0.02$ m, $v = 0.2$ m/s, $\Delta t = 0.05$ s. The true heading is noise-free, so the small $\sigma_\theta$ term only keeps the filter's heading covariance from collapsing to zero.

Three things follow directly from these formulas:

- **All three variants carry the same total noise.** Rotation doesn't change a matrix's trace, so $Q_{\text{pos}}^{\text{aniso}}$ always has trace $(\sigma_{\text{grip}}^2 + \sigma_{\text{slip}}^2)\Delta t^2$. That is exactly $2\sigma_{\text{iso}}^2 \Delta t^2$, the trace of the isotropic version. The variants differ only in *direction*, not in how much noise they assume overall.
- **The off-diagonal term is what points the ellipse.** It is zero only when $\theta_Q$ is a multiple of 90°. With the defaults, the slip variance is 25 times the grip variance (a 5:1 ratio in standard deviation), so pointing the ellipse the wrong way matters a lot. When `fixed_anisotropic`'s ellipse is misaligned, the filter assumes little noise in a direction where the real slip is large. It becomes overconfident in exactly that direction, which drives the NEES gap in §2.
- **A 180° error costs nothing.** Replacing $\theta_Q$ with $\theta_Q + \pi$ flips the sign of both $c$ and $s$, which leaves $c^2$, $s^2$ and $cs$ unchanged. So $Q_{\text{pos}}^{\text{aniso}}$ repeats every 180°, which is the periodicity §3 builds on. (That holds for $Q$ itself; §3 shows the error a mismatch leaves behind in the heading estimate doesn't disappear when the ellipse realigns.)

---

## 2. The dominant finding: `fixed_anisotropic` is dramatically worse, everywhere

Monte Carlo NEES over 500 trials (seed 0, $`dt = 0.05\,\text{s}`$, one full loop over $`20\,\text{s}`$, all script defaults), binned by how far the true heading has rotated away from `fixed_anisotropic`'s fixed reference heading. Each trial draws a fresh slip realization, initial error and measurement noise. Redrawing the slip matters, because the slip is the process noise whose model $Q$ is being tested.

![Two panels from friction_anisotropic_ekf.py: the circular path with the per-step slip covariance ellipse turning with the heading, and Monte Carlo NEES over one full turn for isotropic, frozen anisotropic and heading-aware process noise](../../assets/friction_anisotropic_ekf.png)

*Figure: `use_numpy/friction_anisotropic_ekf.py` at its defaults (seed 0), plotted by `uv run python assets/make_figures.py friction_anisotropic_ekf`.*

| Rotation away from reference | `isotropic` | `fixed_anisotropic` | `heading_aware` |
| --- | --- | --- | --- |
| 0-45° | 3.5 | 11.1 | 3.1 |
| 45-90° | 3.7 | 20.1 | 3.1 |
| 90-135° | 3.7 | 23.8 | 3.0 |
| 135-180° | 3.6 | 18.2 | 3.1 |

A consistent 3-DoF filter averages $\text{NEES} = 3$. For an average over 500 independent trials, the 95% acceptance interval is $[2.79, 3.22]$ (the single-run bound of 7.81 doesn't apply to an average).

- `heading_aware` stays inside that interval in every bin: it is consistent.
- `isotropic` sits just above it (3.4-3.8) in every bin: mildly overconfident, because its circle assumes half the true variance along the slip axis.
- `fixed_anisotropic` is 3.7-8× too high: badly overconfident.

This is robust. Across seeds 0-3, `isotropic`'s and `heading_aware`'s bins move by at most 0.15 and `fixed_anisotropic`'s by at most 1.4, and `fixed_anisotropic` is worse than *both* alternatives in every bin of every seed, from the first bin on.

RMS position error from the script's single run (one trajectory, not the Monte Carlo) tells a smaller but consistent story: at seed 0, $`\text{heading\_aware}\ (0.0108\,\text{m}) < \text{isotropic}\ (0.0118\,\text{m}) < \text{fixed\_anisotropic}\ (0.0131\,\text{m})`$, and the same ordering holds in seeds 1-3. Getting the shape right and pointed the right way is both more accurate and, as the table shows, far better calibrated.

---

## 3. A secondary finding: the worst mismatch is 90°, not 180°

The *worst* of `fixed_anisotropic`'s four bins is 90-135° in all four seeds, not the largest possible mismatch (135-180°). The reason: a covariance ellipse $`R(\theta)\,\mathrm{diag}(a,b)\,R(\theta)^\top`$ has period $\pi$ in $\theta$, not $2\pi$, since rotating it by 180° gives back the identical ellipse. So for the *orientation of the noise ellipse*, a 180° heading difference is no mismatch at all, and the worst possible mismatch is 90°. (The binned peak lands in 90-135° rather than 45-90° because the filter's error takes about a second to build up, so its NEES trails the mismatch slightly.)

Past the peak, NEES recovers only partly: 18.2 in the 135-180° bin, still about 6× the consistent value, although the ellipse is realigned at 180°. Splitting the NEES into its position and heading parts shows why (a separate check over 200 trials, not something the script prints):

| Time (heading) | Position NEES (2 DoF, consistent = 2) | Heading NEES (1 DoF, consistent = 1) |
| --- | --- | --- |
| 0 s (0°) | 2.0 | 0.9 |
| 5 s (90°) | 6.9 | 8.8 |
| 10 s (180°) | 2.6 | 14.1 |
| 15 s (270°) | 9.9 | 11.1 |
| 20 s (360°) | 2.2 | 10.5 |

The position part follows the ellipse mismatch exactly: it peaks at 90° and 270° and returns to nearly consistent at 180° and 360°. The heading part doesn't come back. While the ellipse is misaligned, the filter explains real slip with the wrong position noise and pushes part of that error into its heading estimate. Heading is never measured directly, so that error drains away only slowly through the position-heading cross-covariance, and it persists long after the ellipse realigns. The figure's NEES curve shows both effects: two humps a half-turn apart, sitting on a floor that doesn't return to 3.

---

## 4. Takeaway for a real friction-anisotropic contact model

Both findings point the same direction for modeling a friction-anisotropic pad's slip in a real filter: the noise model needs to track the *current* heading, not just "know" the pad is anisotropic in the abstract. Getting the shape right but the orientation wrong (`fixed_anisotropic`) isn't a small, forgivable approximation. It's dramatically worse than the naive isotropic fallback in every bin, because it makes a confident, specific directional claim that's false most of the time. That claim does the most damage at a 90° mismatch (§3), and the damage outlasts the mismatch through the heading estimate. The safe fallback, if heading-tracking isn't available for some reason, is to not claim a direction at all (`isotropic`) rather than claim the wrong one.

---

## 5. References

1. Hu, D. L., Nirody, J., Scott, T., & Shelley, M. J. (2009). *The Mechanics of Slithering Locomotion*. Proceedings of the National Academy of Sciences, 106(25), 10081-10085. https://doi.org/10.1073/pnas.0812533106 - the source of this doc's opening image (a snake's belly scales) and the physical phenomenon it models: friction fixed to the body/scale frame with different coefficients along vs. across the body axis (for snakes, lower along the body than across it). The paper's friction model is deterministic and doesn't itself use a stochastic slip-variance/covariance framing - that translation into an EKF process-noise ellipse is this doc's own extension, applying the world-frame-rotation argument from [`pointcloud_pose_tracking_empirical_note.md`](pointcloud_pose_tracking_empirical_note.md) (see §Intuition above) to process rather than measurement noise.
2. Kong, N. J., Payne, J. J., Zhu, J., & Johnson, A. M. (2024). *Saltation Matrices: The Essential Tool for Linearizing Hybrid Dynamical Systems*. Proceedings of the IEEE, 112(6), 585-608. https://doi.org/10.1109/JPROC.2024.3440211 (preprint: arXiv:2306.06862) - §V-F derives the saltation matrix for the stick-slip friction transition under Coulomb friction with a single, direction-free coefficient; this doc's direction-dependent process noise is the part that model leaves out. Also reference 1 of [hybrid_saltation_ekf.md](hybrid_saltation_ekf.md#11-references).
