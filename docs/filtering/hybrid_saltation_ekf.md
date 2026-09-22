# Saltation matrices: propagating uncertainty through a hybrid/discontinuous event

Every filter in [kf_ekf_iekf.md](kf_ekf_iekf.md) and [extra_kf_variants.md](extra_kf_variants.md) assumes the state evolves *continuously* between measurements. Bio-inspired locomotion (a footstep, an inchworm anchor/release cycle, a friction-anisotropic grip-then-slip transition) breaks that assumption on purpose, not by accident: the whole point of these platforms is to move in discrete, hybrid bursts. This doc works through the tool that lets an EKF cross one of those discontinuities without either (a) silently pretending nothing happened, or (b) discarding all uncertainty information at the jump - using [`saltation_matrix_ekf.py`](../../use_numpy/saltation_matrix_ekf.py)'s bouncing-point-mass toy problem as the concrete example.

> **Note (reading order)**: §1-§3 carry the core idea - why an ordinary reset Jacobian isn't enough, and a worked example showing it's wrong by a real, non-tiny amount. That's enough for a first read. §4 is a from-scratch derivation (with a documented wrong turn kept deliberately, as a teaching point) - useful once you want to trust the formula yourself, skippable if you're willing to take it on faith for now. §5-§8 are the practical payoff (a structural property, a counterintuitive empirical finding, tuning gotchas, and a quantified account of what happens once contact detection itself is uncertain) and are worth reading even if §4 is skipped. §9 is an explicit tangent into a different, related context (periodic-orbit stability) that this script's own use case never needs - safe to skip entirely unless that's what brought you here.

## 1. What a hybrid dynamical system is, here

A **hybrid dynamical system** alternates **continuous flow** with **instantaneous discrete jumps**, triggered by a **guard condition** and applied via a **reset map**:

- **State**: $x = (p, v) \in \mathbb{R}^6$ with $p, v \in \mathbb{R}^3$ - a point mass's position and velocity, plain $\mathbb{R}^6$ (no rotation, unlike most scripts in this repo).
- **Flow**: $`\dot x = f(x) = \begin{bmatrix} v \\ (0,0,-g) \end{bmatrix}`$ - ordinary free-fall, exactly linear in $x$.
- **Guard**: $g(x) = p_z$. The system jumps whenever a falling trajectory reaches $g(x) = 0$ (touches the ground). Its gradient $Dg = \partial g/\partial x$ - which shows up throughout this doc, starting in §2 - is the constant row vector $(0,0,1,0,0,0)$ here: it just picks out the $p_z$ component.
- **Reset map**: $R(x)$, applied at the guard - here, $v_z \mapsto -e\,v_z$ (an inelastic bounce with restitution $e$), position and horizontal velocity untouched. Throughout this doc, a superscript $-$ / $+$ on a state or vector field (e.g. $x^{-}$, $f^{-}$, $x^{+}$, $f^{+}$) means "evaluated just before/after the reset."

This is the textbook canonical example (the word "saltation" is Latin for "leaping"), and it is the simplest possible analog of a foot-strike/ground-contact impact - a discrete velocity reset at a discrete contact event - without $SE(3)$'s rotational complexity layered on top.

## 2. Why the reset map's own Jacobian is not enough

Suppose you want to propagate a covariance $P$ through a bounce. The reset map itself is simple and linear: $R(x) = \text{diag}(1,1,1,1,1,-e)\,x =: DR\,x$. The natural (and wrong) instinct is

$$P^{+} = DR\,P^{-}\,DR^\top$$

The problem is timing, not the reset map itself: $DR$ is the Jacobian of "apply the reset to a state already sitting exactly on the guard" - but a *perturbed* trajectory does not reach the guard at the same instant as the nominal one. Picture two point masses falling on very slightly different trajectories - nominal and perturbed by $\delta x$:

```text
nominal:    ●╲                         perturbed:    ●╲
              ╲                                        ╲
               ╲                                        ╲
────────────────●──── ground (p_z = 0) ──────────────────●── ground
                t*                                       t* + δt
```

Because their initial states differ, they reach $p_z = 0$ at slightly different times, $t^{\*}$ and $t^{\*}+\delta t$. During that sliver $\delta t$, the perturbed trajectory is still governed by the *pre-impact* flow $f$, exactly like the nominal one - but by the time it finally crosses, it has drifted further along $f$ than a naive "just apply $DR$ to the perturbation at time $t^{\*}$" calculation would credit it for. An ordinary Jacobian, evaluated only at the fixed instant $t^{\*}$, has no way to see this: it silently assumes both trajectories cross at the same instant. Once the guard is close to tangent to the flow (small $Dg\cdot f^{-}$, i.e. a shallow, near-grazing crossing), this timing effect can dominate the reset map's own contribution.

The correct sensitivity of "post-impact state, at its own natural post-impact time" with respect to "pre-impact state, at its own natural pre-impact time" has to account for that time-shift. That correct sensitivity is the **saltation matrix**, $\Xi$ - the ordinary reset Jacobian $DR$, corrected by a term for exactly this event-timing sensitivity. §4 below shows this correction is a rank-one projector removing exactly the "along-the-flow" component of a perturbation - the part that only changes *when* the guard is crossed, not *where*.

At a smooth (non-event) point, the ordinary Jacobian $A = \partial f/\partial x$ is all there is - perturbations evolve continuously, and there is no timing ambiguity to correct for. At a hybrid event, $\Xi$ plays that same role but must additionally carry the event-time correction:

| Smooth dynamics | Hybrid event |
| --- | --- |
| Perturbation evolves continuously | Perturbation can jump discontinuously |
| Linearization: $A = \partial f/\partial x$ | Linearization: saltation matrix $\Xi$ |
| No event-time correction needed | Must correct for the crossing-time shift $\delta t$ |

## 3. A worked example: naive vs. saltation, in numbers

It's worth seeing the two approaches disagree on an actual number, not just abstractly. Take two point masses, both falling with the same downward velocity ($-2\,\text{m/s}$), except the second starts $5\text{cm}$ higher ($\delta p_z = 0.05$, no velocity perturbation at all):

- **Naive prediction** ($DR$ applied to the perturbation carried linearly forward to the nominal crossing time): $DR$'s velocity row only scales the *velocity* perturbation by $-e$ - and that perturbation is exactly zero here (a pure position offset, under free-fall, stays a pure position offset all the way to the nominal crossing time; nothing pushes it into the velocity component). So the naive prediction for the change in post-bounce speed is **exactly $0$**.
- **True answer** (re-simulated: let the second ball reach its own crossing time, apply the reset there, compare): starting $5\text{cm}$ higher means falling for an extra $\sim\!5\text{ms}$ before impact, so it's moving faster when it lands, and bounces back faster too. True change in post-bounce vertical speed: **$+0.0242\ \text{m/s}$**.
- **Saltation-corrected prediction** ($\Xi$ applied the same way): **$+0.0243\ \text{m/s}$** - matches to within second-order error.

The naive approach isn't slightly off here; it's qualitatively wrong: it says a pure position perturbation can never affect the bounce speed, when in fact it's the *entire* effect in this example. $DR$ cannot see this because it has no notion of time at all - it only knows how to transform a state that is already sitting on the guard. $\Xi$ sees it because its correction term is built specifically to capture "this trajectory needed a different amount of falling time to get here."

(Computed directly from this repo's own `flow`, `crossing_time`, `reset_map`, `reset_jacobian`, and `saltation_matrix` functions in `saltation_matrix_ekf.py`, not hand-derived - reproducible with $x^{-}_0=(0,0,5,0,0,-2)$, $\delta x_0=(0,0,0.05,0,0,0)$, $e=0.5$, $g=9.81$. See the [Appendix](#worked-arithmetic-behind-the-bounce-speed-numbers) for these three numbers worked out by hand, with a calculator - no code required.)

## 4. Deriving $\Xi$ - including a wrong turn, caught by verification

**A formula that looks standard and is wrong.** Before deriving anything, it's worth recording what didn't work, because it's the kind of formula that's easy to misremember and repeat. A plausible-looking candidate is

$$\Xi_{\text{wrong}} = DR + \frac{\big[f^{+}(x^{+}) - DR\,f^{-}(x^{-})\big] \otimes Dg}{Dg \cdot f^{-}(x^{-})}$$

($f^{-}$, $f^{+}$ the pre-/post-impact vector fields, $Dg$ the guard's gradient, $\otimes$ an outer product). It's dimensionally sensible and structurally plausible. It is also **not** the saltation matrix: checked against a from-scratch finite-difference ground truth (perturb the pre-impact state, re-land it on the guard via the pre-impact flow, apply the reset, compare to nominal), the two disagree by $\max|\Xi_{\text{wrong}} - \Xi_{\text{numeric}}| \approx 1.28$ - nowhere near floating-point noise.

This formula is not a fabrication, though - it is a real, standard expression from the hybrid-systems literature, which is exactly why it's easy to reach for. It answers a genuinely different question: it's the correct linearization when two trajectories are compared at a *shared* reference time (as in composing saltation matrices across several events into a single Poincaré-map derivative - see [§9](#9-connection-to-poincaré-maps) below), where the post-event vector field $f^{+}$ needs to be folded back in to account for continued flow past the shared time marker. $\Xi$ above answers a different question - "compare each trajectory at its *own* natural post-event time" - which is what propagating a covariance one event at a time actually needs, and it has no such term. Using the shared-time formula here silently smuggles in a spurious correction for a time shift this problem doesn't have.

**The correct derivation.** Consider a one-parameter family of trajectories $x(t; p)$ ($p$ a perturbation parameter, $p=0$ nominal), each governed by $\dot x = f(x)$ until a $p$-dependent crossing time $t^{\*}(p)$ defined implicitly by $g\big(x(t^{\*}(p); p)\big) = 0$. Let $S(t) = \left.\dfrac{\partial x(t;p)}{\partial p}\right|_{p=0}$.

Implicit differentiation of the guard condition gives the crossing-time sensitivity:

$$\frac{dt^{\*}}{dp} = -\frac{Dg \cdot S(t^{\*})}{Dg \cdot f(x^{-})}$$

The pre-impact state's own sensitivity (accounting for both the explicit $p$-dependence and the fact that $t^{\*}$ itself moves) is then

$$\frac{d}{dp}\big[x(t^{\*}(p); p)\big] = f(x^{-})\,\frac{dt^{\*}}{dp} + S(t^{\*}) = B(x^{-})\,S(t^{\*})$$

where

$$B(x^{-}) = I - \frac{f(x^{-}) \otimes Dg}{Dg \cdot f(x^{-})}$$

$B$ is a projector that removes exactly the "along-the-flow" component of a perturbation - the piece that would otherwise double-count as a pure time-shift rather than a genuine change in the crossing state. Composing with the reset map's own Jacobian gives

$$\Xi(x^{-}) = DR(x^{-})\,B(x^{-}) = DR(x^{-})\left[I - \frac{f(x^{-}) \otimes Dg(x^{-})}{Dg(x^{-}) \cdot f(x^{-})}\right]$$

This was re-derived a second, independent way (differentiating the "project a nearby point back onto the guard via the pre-impact flow" operator directly) and cross-checked numerically to $\sim 2.9\times10^{-8}$ - machine precision - against the finite-difference ground truth. Both derivations and the check are implemented in `saltation_matrix`.

## 5. A structural property that matters in practice: $Dg\,\Xi = 0$

For this guard ($g(x) = p_z$), $\Xi$'s output row for $p_z$ is identically zero, for *any* $x^{-}$, $e$, $g$:

$$Dg\,\Xi(x^{-}) = 0$$

Equivalently, in code: `Dg @ saltation_matrix(x_minus, e, g)` returns `array([0., 0., 0., 0., 0., 0.])`. This is not a bug - it's forced by the setup. Every trajectory in the family satisfies $g(x) = 0$ exactly at its own crossing (that's what "crossing the guard" means), and the reset map doesn't touch position, so the post-impact height is *exactly* zero for every member of the family, with zero sensitivity to any perturbation. $\Xi$ is correctly reporting that.

The catch: this exact-zero claim is only trustworthy if the filter's *own* estimated crossing time coincides exactly with the true one. It generally will not, once there is any tracking error at all - and comparing a covariance with an (near-)exactly-zero entry against a true trajectory sampled at a fixed tick, not at its own exact crossing, is a direct route to an artificially huge Mahalanobis distance if the two don't line up. `step_hybrid` adds a small isotropic "impact noise" floor after every bounce (identically for both filter variants) specifically to keep this otherwise-correct projection numerically usable; see its own docstring.

## 6. The empirical finding: not "naive is overconfident, saltation fixes it"

The intuitive story going in was that the naive $P^{+} = DR\,P^{-}\,DR^\top$ update would be measurably overconfident (too-small reported uncertainty) right after each bounce compared to the saltation-corrected one, and a Monte Carlo NEES (Normalized Estimation Error Squared - a per-trial score for how far the true state falls from the estimate relative to how much uncertainty $P$ claims; a well-calibrated 6-DoF filter should average NEES $\approx 6$, and *persistently* higher values mean $P$ is too small for the errors actually being made - full formula in the [Appendix](#appendix-related-terms)) consistency check would show it. That is not what happens.

Running `saltation_matrix_ekf.py` at its defaults (300+ Monte Carlo trials, 3 well-separated bounces from a 5m drop at `e=0.85`) and looking at NEES in the few ticks immediately following each bounce (excluding the shared spike at the bounce tick itself, which both filters exhibit for the mundane reason that the true trajectory is also close to its own crossing at that tick):

```
Mean NEES, 5 ticks after each of 3 bounces (seed 0):
  EKF (naive)      ≈ 2.4
  EKF (saltation)  ≈ 3.6
```

**The saltation-corrected filter's post-bounce NEES is consistently *higher* than the naive filter's**, reproduced across multiple seeds (0, 1, 3 all show the same ~2.4 vs. ~3.6 pattern; one seed out of four tried landed in a rarer regime where both spike similarly, discussed in §7). Both stay well under the chi-squared 95% bound ($12.59$ for 6 DoF) at these settings - this is a modest, not catastrophic, effect, but it is the opposite direction from the naive expectation.

**Why**: §5's $Dg\,\Xi = 0$ property means the saltation-corrected update makes the *strongest possible claim* about the guard-normal direction - exactly zero residual uncertainty, beyond the small regularizing floor. The naive update, by contrast, just carries $P$'s existing height-variance forward unchanged ($DR$'s height row is untouched, $[0,0,1,0,0,0]$), a much more modest claim. Once the filter's own estimated bounce time inevitably differs even slightly from the true one - which is the normal case, not a corner case - the *stronger* claim is the one that gets punished harder. The mathematically exact *local* linearization is, in this specific practical sense, the more fragile one, precisely because it is exact only in the limit of infinitesimal tracking error, which real filters never have.

This is not a reason to prefer the naive update generally - averaged over the whole trajectory (not just the few ticks right after a bounce), the two filters are close, and the naive update has no principled derivation behind it at all, so its accidental "robustness" here isn't something to rely on either. A real Hybrid-InEKF implementation needs an explicit model of that detection uncertainty (e.g., an impact-timing noise term scaled by the velocity jump at the event, rather than this script's simple isotropic floor) - using the exact saltation matrix "as-is" is not automatically the safer choice once that uncertainty is real, which is always.
<!-- It is, however, a concrete, quantified instance of exactly the gap already flagged as [Open Consideration #1](../../../../private-notes/PhD_Topic/unified_phd_plan.md#open-considerations-from-technical-review) in the dissertation plan: *saltation matrices assume the transition time is known exactly; contact/phase detection is itself uncertain.*  -->

## 7. Two things worth knowing before reusing this pattern

- **Tune the regularizing floor deliberately, not just for numerical stability.** Sweeping `--impact-noise-std` from `0.0005` to `0.02` on this same problem flips which filter looks worse: at very small floors, saltation's exact zero-projection dominates and it comes out *dramatically* worse (NEES in the tens to hundreds vs. the naive filter's ~3); at larger floors (`≥0.02`), the floor itself swamps the structural difference and the two converge to nearly identical, unremarkable NEES. The `~2.4` vs. `~3.6` result quoted in §6 is specific to a floor sized to be "just barely enough" to avoid outright numerical pathology - a deliberate choice, not an arbitrary one, and worth re-checking whenever any of the other parameters change.
- **Stay well clear of the Zeno regime.** A lossy bounce (`e<1`) produces infinitely many, ever-faster bounces approaching a finite settling time (`~12.4s` for this script's defaults). Well before that time, bounce intervals become comparable to the fixed measurement step `dt`, and comparing a fixed-tick-sampled estimate against a true trajectory that's bouncing many times within a single tick breaks the entire comparison's premise (both filters' NEES explodes into the thousands, for reasons that have nothing to do with the saltation matrix). `--duration`'s default is deliberately chosen to stop after 3 clean, well-separated bounces and before the 4th starts crowding the settling regime.

## 8. Quantifying Open Consideration #1: contact-detection timing jitter

§6 already showed that ordinary state-estimation error alone (with detection otherwise exact) makes the saltation-corrected filter's post-bounce NEES *higher* than the naive filter's, not lower - the opposite of the naive intuition. That leaves open exactly how much worse a *real* contact sensor's own detection latency/jitter makes things, on top of that. `step_hybrid`'s `detect_time_bias`/`detect_time_noise_std` parameters model this directly (both default `0.0`, reproducing every result above): a bounce's reset is applied at a *detected* crossing time `tau_detect = clip(tau + detect_time_bias [+ N(0, detect_time_noise_std)], 0, remaining)` instead of the true geometric `tau`, so early detection resets the mean while still above ground and late detection resets it after the free-fall model has carried it slightly below `p_z = 0` - both real artifacts of a delayed or jittery contact detector (IMU spike, force threshold - Čížek et al. 2018), not numerical noise.

**A correctness subtlety this surfaced**: once a reset can land off-guard, `crossing_time` needs to reject *ascending* roots (the point mass rising back out of a below-ground detection artifact crosses `p_z = 0` again almost immediately, which is not a real impact) and keep only the next *descending* one. Every crossing this script computed before `detect_time_*` existed was already descending by construction, so this is a latent-bug fix with zero effect on any result above - but skipping it turns on nonsense: NEES and RMS position error explode by 4-5 orders of magnitude the instant any detection jitter is introduced, from a cascade of spurious re-bounces within a single step, not from the phenomenon actually being modeled. See `crossing_time`'s docstring for the concrete numbers that exposed it.

With that fixed, sweeping `--detect-time-noise-std` (with `--detect-time-bias 0.0`, i.e. jitter only, no systematic delay) at `dt = 0.02s`, averaged over seeds 0/1/3 (mean NEES over the same 5-tick post-bounce window as §6, individual seeds agreeing to within ~5% of each other at every setting):

| `detect_time_noise_std` (fraction of `dt`) | EKF (naive) | EKF (saltation) | saltation / naive |
| --- | --- | --- | --- |
| 0 (exact detection, §6's result) | 2.37 | 3.61 | 1.5x |
| 0.001s (5%) | 4.10 | 13.12 | 3.2x |
| 0.002s (10%) | 9.30 | 41.54 | 4.5x |
| 0.005s (25%) | 38.64 | 203.92 | 5.3x |
| 0.01s (50%) | 87.67 | 485.98 | 5.5x |

**The finding**: both filters' consistency degrades monotonically as detection jitter grows - unsurprising, since neither one's `P` accounts for this extra error source at all. What's not obvious in advance is that the *gap* between them widens sharply, from §6's already-inverted 1.5x at exact detection to over 5x once jitter reaches a quarter of the tick interval. The saltation-corrected filter's mathematically-exact local correction isn't just "somewhat more fragile" to detection uncertainty than the naive one, as §6 alone might suggest - it is dramatically more sensitive to it, because `Dg @ Xi = 0` (§5) means it stakes everything on the detected crossing being exactly right, and every bit of jitter is pure error in exactly the direction the filter claims zero uncertainty about. This is a direct, quantified instance of Open Consideration #1 for this toy problem: saltation matrices are the mathematically correct tool *given* a known transition time, but that assumption is not free, and how expensive it is to violate is now measured here rather than only asserted.

## 9. Connection to Poincaré maps

Saltation matrices show up in a second, related context: analyzing the stability of a *periodic* hybrid trajectory - e.g., a robot repeatedly bouncing (or, for legged locomotion, repeatedly striking the ground once per stride). Sampling the state once per cycle, at a chosen event, defines a discrete return map $x_{k+1} = P(x_k)$, and its derivative $DP$ governs whether nearby trajectories converge back to the periodic orbit or diverge from it - the hybrid-systems analogue of eigenvalue stability analysis for a fixed point.

Over one cycle, $DP$ is a product of continuous-flow Jacobians and saltation matrices, one of each per phase and event the cycle passes through. Write $\Phi$ for the flow's own state-transition Jacobian over a smooth phase - the finite-time integral of the same $A = \partial f/\partial x$ from §2's table, concretely the `Phi` that this repo's `flow()` function already returns alongside the propagated state - and $\Xi$ for the saltation matrix at each event; e.g., for a cycle with three flow phases and two events in between:

$$DP \approx \Phi_3\,\Xi_2\,\Phi_2\,\Xi_1\,\Phi_1$$

This is the setting §4's "wrong turn" formula is actually built for: composing several such factors requires comparing every trajectory in the family against a single shared timeline running through the whole cycle, which is exactly what that formula's extra $f^{+}$ term supplies. This script never needs that composition - it propagates one covariance forward through one event at a time - so $\Xi$ alone, without the shared-timeline correction, is the right and complete tool here.

## Appendix: related terms

- **Guard condition/reset map**: the switching-surface function $g(x)=0$ and the (possibly discontinuous) map $R$ applied when a trajectory reaches it - the two ingredients that make a system "hybrid" rather than purely continuous.
- **NEES (Normalized Estimation Error Squared)**: $(x_{\text{true}} - x_{\text{est}})^\top P^{-1} (x_{\text{true}} - x_{\text{est}})$. A well-calibrated $n$-DoF filter's NEES should average to $n$ across many independent trials; systematically larger values mean the filter's reported $P$ is too small (overconfident) for the errors it's actually making. See [pose_graph_optimization.md §15.3](../optimization/pose_graph_optimization.md#153-objective-function) for the closely-related Mahalanobis-distance framing already used elsewhere in this repo.
- **Zeno behavior**: a hybrid system undergoing infinitely many discrete transitions in a finite time interval - the generic long-run behavior of any lossy bouncing system, and a standard pathology to guard against in hybrid-system simulation, not specific to saltation matrices themselves.

### Worked arithmetic behind the bounce-speed numbers

§3's three numbers (naive $0$, true $+0.0242$, saltation $+0.0243$) worked out by hand, with a calculator - no code required. Uses §3's own reproducibility parameters: $x^{-}_0=(0,0,5,0,0,-2)$ (so $p_z^0=5$, $v_z^0=-2$), $\delta x_0=(0,0,0.05,0,0,0)$ (so $\delta p_z^0 = 0.05$), $e=0.5$, $g=9.81$.

**Nominal crossing.** Free-fall height is $p_z(t) = p_z^0 + v_z^0 t - \tfrac12 g t^2$. Solve $5 - 2t - 4.905t^2 = 0$ for the positive root:

$$t^{\*} = \frac{-2+\sqrt{2^2+4(4.905)(5)}}{2(4.905)} = \frac{-2+\sqrt{102.1}}{9.81} \approx 0.82614\ \text{s}$$

Velocity there: $v_z^{-} = -2 - 9.81(0.82614) \approx -10.1045\ \text{m/s}$. Post-bounce: $v_z^{+} = -e\,v_z^{-} \approx +5.0522\ \text{m/s}$.

**Naive prediction ($0$).** $DR=\text{diag}(1,1,1,1,1,-e)$ only touches the velocity slot, and $\delta x_0$'s velocity slot is $0$: a pure position offset stays a pure position offset under free-fall (position never feeds back into the dynamics), so $DR$ is handed $0$ in that slot and returns $0$. No simulation needed - just that $DR$'s only nonzero action lands on a component that's zero here.

**True answer ($+0.0242$).** Solve the same quadratic with $p_z^0=5.05$ (the perturbed ball's own crossing):

$$t^{\*}_{\text{pert}} = \frac{-2+\sqrt{4+4(4.905)(5.05)}}{9.81} \approx 0.83108\ \text{s}, \qquad \delta t \approx 0.004936\ \text{s (}\sim\!4.94\text{ms extra fall)}$$

$v_z^{-}(\text{pert}) = -2-9.81(0.83108)\approx -10.1529\ \text{m/s}$, so $v_z^{+}(\text{pert}) = -e\,v_z^{-}(\text{pert}) \approx +5.0764\ \text{m/s}$.

$$\text{True change} = 5.0764-5.0522 \approx +0.02421\ \text{m/s}$$

**Saltation prediction ($+0.0243$).** From §4's formula $\Xi = DR\left[I - \dfrac{f(x^{-})\otimes Dg}{Dg\cdot f(x^{-})}\right]$, the scalar $\dfrac{Dg\cdot\delta x_0}{Dg\cdot f(x^{-})} = \dfrac{0.05}{-10.1045}\approx -0.004948$ is precisely §4's linearized crossing-time-shift formula, $dt^{\*}/dp$, evaluated here - compare it to the true $-\delta t \approx -0.004936$ just above: close but not identical, since it's only a first-order estimate of the shift. That's the origin of §3's "matches to within second-order error."

The $B$-projector step turns that position offset into an *equivalent velocity offset*, using $f_5=-g$ (the $v_z$-component of $f$, i.e. gravity, is constant):

$$(B\,\delta x_0)_{v_z} = 0-(-9.81)(-0.004948) \approx -0.04854\ \text{m/s}$$

Read this as: *starting 5cm higher behaves, to first order, like starting at the same height but already falling ≈0.0485m/s faster* - which is a language $DR$ already knows how to handle. Applying $DR$'s bounce law to that equivalent velocity gives the saltation prediction:

$$\text{Saltation change} = -e\times(-0.04854) \approx +0.02427\ \text{m/s}$$
