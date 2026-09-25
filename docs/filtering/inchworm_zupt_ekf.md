# Phase-conditional ZUPT: exploiting a known anchor/dwell schedule

## Intuition

An inchworm (real or robotic) doesn't move continuously - it cycles between two phases:

```
ANCHOR                    EXTEND                     ANCHOR
(grip, hold still)   (release, push/pull body)   (grip again, hold still)

  ▓▓●━━━━━●            ▓▓●╲     ╱●--->              ●━━━━━●▓▓
   front anchored        body extends/stretches       new anchor set,
   to ground, v = 0      forward, then contracts       v = 0 again
```

- **Anchor**: one end is planted on the ground - the robot is, by construction, perfectly stationary ($v = 0$). No ambiguity, no noise - it's known ground truth.
- **Extend**: the anchor releases, the body extends/pushes forward at some commanded speed, then the new end plants and grips.

The idea this doc explores: during anchor, "velocity = 0" is a *free, trustworthy measurement* (a ZUPT - Zero Velocity Potential Update) you can feed into the filter - but only during anchor. Feed it in during extend (when the robot is actually moving) and you're telling the filter a lie with high confidence, which is exactly the failure mode the `always` variant below demonstrates.

Consider an observation specific to inchworm-like locomotion as below:

> During the anchored/dwell portion of an inchworm cycle the robot is (by definition) stationary - close to free ground truth for a zero-velocity/zero-angular-rate pseudo-measurement (ZUPT/ZARU), and the one moment the accelerometer is a trustworthy gravity reference; during the anchor-release/extend burst neither holds

This doc works through the simplest concrete version of that idea, using [`inchworm_zupt_ekf.py`](../../use_numpy/inchworm_zupt_ekf.py)'s 1D crawling point mass as the toy problem.

**Scope, stated up front**: this is deliberately a small slice of the real idea. It models translation only - zero-velocity updates (ZUPT), not the full zero-angular-rate/orientation story (ZARU) [`hybrid_saltation_ekf.md`](hybrid_saltation_ekf.md) mentions as a Stretch item - since that needs an orientation state this toy doesn't carry. It also assumes the gait schedule (when anchor/extend happen) is *known*, not detected. That second question - what happens once the schedule itself is uncertain - is exactly the subject of [`hybrid_saltation_ekf.md` §8](hybrid_saltation_ekf.md#8-quantifying-contact-detection-timing-jitter), which this doc leans on rather than repeating.

## 1. Why this isn't a hybrid-reset problem

Unlike the bouncing point mass in `hybrid_saltation_ekf.md`, nothing here needs a guard, a reset map, or a saltation matrix. Velocity is externally commanded per phase - an ordinary **switched-linear system**, not a discontinuous *state* jump the filter's covariance has to be linearized through. The entire interesting question sits on the **measurement** side: does the filter correctly exploit (or wrongly misuse) the free zero-velocity information the anchor phase provides.

State is $x = [p, v]$ (1D). The gait cycle alternates:

- **anchor** (duration $t_{\text{anchor}}$): the point mass is exactly stationary, $v = 0$.
- **extend** (duration $t_{\text{extend}}$): the point mass ramps up to a commanded cruise speed $v_{\text{extend}}$, holds it, then ramps back down to exactly $0$ before the next anchor phase begins - continuous everywhere, no instantaneous jump (see §2 for why that matters).

Three ways of using a zero-velocity pseudo-measurement (ZUPT) are compared, sharing an identical predict step and the same noisy position measurement every tick - they differ *only* in when an additional ZUPT update runs:

| Variant | ZUPT applied when |
| --- | --- |
| `never` | Never - position-only, ignores the anchor phase entirely |
| `always` | Every tick, regardless of true phase - the schedule-unaware mistake |
| `phase_conditional` | Only on ticks the known schedule marks as anchor - the correct policy |

### 1.1 The filter math, concretely

All three variants run the same Kalman filter over $x = [p, v]^\top$. The script calls it an EKF to match the rest of this repo, but every model here is exactly linear, so there is no linearization error and it is really a plain KF. Each tick runs up to three steps, in this order: predict, position update, then (depending on the variant) a ZUPT update (`run_ekf`).

- **Predict** (`predict`): a constant-velocity model, identical for all variants and both gait phases. The filter doesn't know the commanded velocity, only that it's roughly constant between ticks:

  $$
  \Phi = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}, \qquad
  Q = \begin{bmatrix} \left(\tfrac{1}{2}\sigma_{\text{process}}\Delta t^2\right)^2 & 0 \\ 0 & \left(\sigma_{\text{process}}\Delta t\right)^2 \end{bmatrix}
  $$

  $$
  x^{-} = \Phi\,x, \qquad P^{-} = \Phi\,P\,\Phi^\top + Q
  $$

  $Q$ is a simple diagonal heuristic, the same one `saltation_matrix_ekf.py` uses. It is not the textbook continuous white-noise-acceleration $Q$, which would also have off-diagonal terms.

- **Measurement updates** (`_kf_update`): both updates go through the same standard linear-Gaussian update and differ only in $z$, $H$ and $R$:

  $$
  r = z - H x^{-}, \qquad S = H P^{-} H^\top + R, \qquad K = P^{-} H^\top S^{-1}
  $$

  $$
  x^{+} = x^{-} + K r, \qquad P^{+} = (I - K H)\,P^{-}
  $$

  | Update | Function | $z$ | $H$ | $R$ | Runs |
  | --- | --- | --- | --- | --- | --- |
  | Position | `measurement_update_position` | noisy position reading | $\begin{bmatrix} 1 & 0 \end{bmatrix}$ | $\sigma_{\text{pos}}^2$ | every tick |
  | ZUPT | `measurement_update_zupt` | $0$ (pseudo-measurement) | $\begin{bmatrix} 0 & 1 \end{bmatrix}$ | $\sigma_{\text{zupt}}^2$ | depends on the variant |

ZUPT is a *pseudo*-measurement: no sensor produces that $z = 0$. The filter is simply told "velocity is zero, give or take $\sigma_{\text{zupt}}$", which is only true if the robot really is stationary. The script's defaults are $\sigma_{\text{pos}} = 0.02$ m and $\sigma_{\text{zupt}} = 0.01$ m/s. Because the two measurement noises are independent, running the ZUPT update right after the position update in the same tick gives the same result as one joint update with both rows of $H$ stacked.

**The ZUPT update, written out.** With $H = \begin{bmatrix} 0 & 1 \end{bmatrix}$ everything reduces to scalars. Write the predicted covariance as

$$
P^{-} = \begin{bmatrix} P_{pp} & P_{pv} \\ P_{pv} & P_{vv} \end{bmatrix}
\quad\Longrightarrow\quad
S = P_{vv} + R_{\text{zupt}}, \qquad
K = \frac{1}{P_{vv} + R_{\text{zupt}}} \begin{bmatrix} P_{pv} \\ P_{vv} \end{bmatrix}, \qquad
r = 0 - v^{-} = -v^{-}
$$

Substituting into the update above:

$$
v^{+} = \frac{R_{\text{zupt}}}{P_{vv} + R_{\text{zupt}}}\,v^{-}, \qquad
p^{+} = p^{-} - \frac{P_{pv}}{P_{vv} + R_{\text{zupt}}}\,v^{-}
$$

$$
P_{vv}^{+} = \frac{P_{vv}\,R_{\text{zupt}}}{P_{vv} + R_{\text{zupt}}}, \qquad
P_{pp}^{+} = P_{pp} - \frac{P_{pv}^2}{P_{vv} + R_{\text{zupt}}}
$$

Three things follow directly from these formulas:

- **It shrinks velocity toward zero.** The velocity estimate is multiplied by a factor between 0 and 1. The more the filter already trusts its velocity estimate (small $P_{vv}$) relative to the pseudo-measurement, the less it moves; the more uncertain it is, the harder it is pulled to $0$.

- **It corrects position too, through the cross-covariance.** $P_{pv}$ is how the filter has learned that position and velocity errors move together. If the velocity estimate turns out too high, the position estimate has probably drifted ahead too, so $p$ gets pulled back as well. This is how a velocity-only pseudo-measurement also affects position.

- **It always makes the filter more confident, whether or not the claim is true.** $P_{vv}^{+}$ is always smaller than both $P_{vv}$ and $R_{\text{zupt}}$. Nothing in the update looks at whether the robot is actually stationary. So when `always` applies it during extend, the filter reports a velocity standard deviation of at most 0.01 m/s while the true velocity is up to $v_{\text{extend}} = 0.1$ m/s away from the zero it was just told. That built-in overconfidence is what drives the large NEES values for `always` in §3.

## 2. A calibration trap: an instantaneous velocity jump swamps the comparison

The first version of this toy used a hard step for `true_velocity` - $0$ during anchor, $v_{\text{extend}}$ the instant extend began. That produced nonsense: $\text{NEES}$ in the hundreds to thousands for *all three* filter variants, dominated by a shared spike at every phase transition that had nothing to do with ZUPT policy. The reason: an instantaneous jump is an effectively-infinite acceleration, and the predict step's constant-velocity assumption (with a finite process-noise budget) has no way to represent that, regardless of which measurements get fused afterward.

The fix was physical, not numerical: `true_velocity` now ramps linearly over a $t_{\text{ramp}}$ window at each end of the extend phase, so velocity is continuous everywhere. But *even a short ramp* isn't automatically enough - the true ramp acceleration ($v_{\text{extend}} / t_{\text{ramp}}$) still has to be checked against $\sigma_{\text{process}}$ (the filter's assumed acceleration disturbance), or the same swamping happens on a smaller scale. A first ramped attempt ($t_{\text{ramp}} = 0.1\,\text{s}$, $v_{\text{extend}} = 0.2\,\text{m/s}$, $\sigma_{\text{process}} = 0.05\,\text{m/s}^2$) implied a true/assumed acceleration ratio of $40\times$ - still enough to produce $\text{NEES}$ in the hundreds throughout an entire 6-tick cruise window, never converging. The defaults were recalibrated ($t_{\text{ramp}} = 0.2\,\text{s}$, $v_{\text{extend}} = 0.1\,\text{m/s}$, $\sigma_{\text{process}} = 0.15\,\text{m/s}^2$, $t_{\text{extend}} = 1.0\,\text{s}$) to bring that ratio down to a modest $\sim 3\times$ and give the cruise/anchor windows enough ticks (12 and 20 respectively) to actually settle before being measured. This is the same lesson as `saltation_matrix_ekf.py`'s Zeno-regime pitfall: pick simulation parameters with real margin under a hard failure mode, not by trial and error against the first numbers that come out.

The headline NEES comparison below also excludes the ramp ticks themselves (`is_cruise` explicitly excludes them, matching `is_anchor`'s own definition) - they're a shared transition cost all three variants pay alike, not the phenomenon being measured, the same convention `hybrid_saltation_ekf.md` uses for its own shared bounce-tick spike.

## 3. The finding: not just "always is wrong while moving"

The going-in expectation was that `always` would at least match `phase_conditional` during genuine anchor ticks (both apply the identical, correct update there) and only diverge during motion. Running it says otherwise (seeds 0-3, `n_trials=500`, mean NEES over the cruise-only/anchor-only windows):

| Variant | anchor-only NEES | cruise-only NEES | velocity RMS (single run) |
| --- | --- | --- | --- |
| `never` | ~6.0 | ~10.8 | ~0.055-0.060 m/s |
| `always` | ~299 | ~521 | ~0.060 m/s |
| `phase_conditional` | ~6.1 | ~16.4 | ~0.043-0.045 m/s |

(A consistent 2-DoF filter should average $\text{NEES} \approx 2$ everywhere; all three run somewhat above that even in their best window, a residual mild effect of the ramp-recovery cost discussed in §2 that even the *best* variant doesn't fully escape.)

**`always` is dramatically worse everywhere, not just during motion.** Misapplying ZUPT throughout every cruise phase leaves the filter so overconfident (velocity variance driven artificially tight around a wrong belief) that ~20 ticks of genuinely correct ZUPT evidence at the start of the next anchor phase isn't enough to recover before that phase ends - the corruption from one cycle bleeds into the next. Checked directly: within a single 20-tick anchor run, `always`'s NEES starts around 730 and decays only to ~266 by the last tick, while `phase_conditional`'s starts around 20 and settles near 4 within the same window. `always` also gets no accuracy benefit for the trouble - its velocity RMS (~0.060 m/s) is statistically indistinguishable from `never`'s, since being right half the time and badly wrong the other half roughly cancels out in aggregate error, even though it's *dangerously overconfident* the whole time.

**A second, more subtle finding echoes `hybrid_saltation_ekf.md`'s own §6/§8.** `phase_conditional` has the best raw *velocity accuracy* of the three (lowest velocity RMS) - it's the only variant that both uses the free anchor-phase information and never misapplies it. But that doesn't carry over to *position*: RMS position error (seeds 0-3) is `never` 0.0107-0.0147 m vs. `phase_conditional` 0.0157-0.0184 m - `phase_conditional` is actually the *worse* of the two on position, in every seed tested, even though it wins clearly on velocity. And on *calibration* (NEES), `never`, despite discarding the anchor-phase information entirely, ends up at least as well calibrated as `phase_conditional`, sometimes measurably better (cruise-only: ~10.8 vs. ~16.4). Correctly staking strong confidence on a pseudo-measurement is still more fragile than never staking it at all - the same theme as the saltation-corrected filter running slightly *worse*-calibrated than the naive one once its own timing assumption stops being exact (there: detection timing; here: the residual ramp-recovery cost) - see [`hybrid_saltation_ekf.md` §6/§8](hybrid_saltation_ekf.md#6-the-empirical-finding-a-modest-real-gap-not-a-dramatic-one). Accuracy and calibration are genuinely different axes, and "accuracy" itself isn't even one axis: `phase_conditional` wins clearly on velocity, loses on position, and ties or narrowly loses on calibration - it dominates `always` on all three, but not `never`.

## 4. Takeaway for a real phase-gated measurement update

This toy's `never`/`always`/`phase_conditional` split is a minimal stand-in for a general design question in any estimator that fuses a phase-dependent pseudo-measurement: which measurement terms are active should be gated by phase within the single update, not applied unconditionally and not left out for simplicity. §3's two findings both argue for taking that gating seriously. The blunter one: misapplying the constraint doesn't just cost accuracy on the ticks it's wrong - the overconfidence it creates bleeds into the next anchor phase and isn't fully undone before that phase ends, so a gating bug doesn't stay contained to the phase it occurs in. The subtler one: `phase_conditional` has the best *velocity* accuracy of the three, but not the best *position* accuracy (that's `never`, in every seed tested) and not reliably the best calibrated either - `never` sometimes narrowly beats it on NEES despite discarding real information, the same fragility theme as `hybrid_saltation_ekf.md`'s saltation-corrected filter running slightly worse-calibrated than the naive one, where a confident claim that wins on one axis in the idealized case is somewhat more exposed to real-world mismatch than never making the claim. So the payoff for correct gating is real (clearly better velocity tracking, and by far the best-behaved failure mode) but it isn't a free lunch on every axis at once - it still needs to be checked for calibration and for the specific accuracy metric that matters, not assumed to win across the board, once anything in the real setup deviates from the idealization.

---

## 5. References

1. Foxlin, E. (2005). *Pedestrian Tracking with Shoe-Mounted Inertial Sensors*. IEEE Computer Graphics and Applications, 25(6), 38-46. https://doi.org/10.1109/MCG.2005.140 - the original zero-velocity-update (ZUPT) technique this doc's `never`/`always`/`phase_conditional` split is built around, there gated by a stance-phase detector rather than this toy's known anchor/extend schedule.