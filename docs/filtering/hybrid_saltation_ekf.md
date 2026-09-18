# Saltation matrices: propagating uncertainty through a hybrid/discontinuous event

Every filter in [kf_ekf_iekf.md](kf_ekf_iekf.md) and [extra_kf_variants.md](extra_kf_variants.md) assumes the state evolves *continuously* between measurements. Bio-inspired locomotion (a footstep, an inchworm anchor/release cycle, a friction-anisotropic grip-then-slip transition) breaks that assumption on purpose, not by accident: the whole point of these platforms is to move in discrete, hybrid bursts. This doc works through the tool that lets an EKF cross one of those discontinuities without either (a) silently pretending nothing happened, or (b) discarding all uncertainty information at the jump - using [`saltation_matrix_ekf.py`](../../use_numpy/saltation_matrix_ekf.py)'s bouncing-point-mass toy problem as the concrete example.

## 1. What a hybrid dynamical system is, here

A **hybrid dynamical system** alternates continuous flow with instantaneous discrete jumps, triggered by a **guard condition** and applied via a **reset map**:

- **State**: $x = (p, v) \in \mathbb{R}^6$ with $p, v \in \mathbb{R}^3$ - a point mass's position and velocity, plain $\mathbb{R}^6$ (no rotation, unlike every other script in this repo).
- **Flow**: $\dot x = f(x) = \begin{bmatrix} v \\ (0,0,-g) \end{bmatrix}$ - ordinary free-fall, exactly linear in $x$.
- **Guard**: $g(x) = p_z$. The system jumps whenever a falling trajectory reaches $g(x) = 0$ (touches the ground).
- **Reset map**: $R(x)$, applied at the guard - here, $v_z \mapsto -e\,v_z$ (an inelastic bounce with restitution $e$), position and horizontal velocity untouched.

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

The correct sensitivity of "post-impact state, at its own natural post-impact time" with respect to "pre-impact state, at its own natural pre-impact time" has to account for that time-shift. That correct sensitivity is the **saltation matrix**, $\Xi$ - the ordinary reset Jacobian $DR$, corrected by a term for exactly this event-timing sensitivity. §3 below shows this correction is a rank-one projector removing exactly the "along-the-flow" component of a perturbation - the part that only changes *when* the guard is crossed, not *where*.

At a smooth (non-event) point, the ordinary Jacobian $A = \partial f/\partial x$ is all there is - perturbations evolve continuously, and there is no timing ambiguity to correct for. At a hybrid event, $\Xi$ plays that same role but must additionally carry the event-time correction:

| Smooth dynamics | Hybrid event |
| --- | --- |
| Perturbation evolves continuously | Perturbation can jump discontinuously |
| Linearization: $A = \partial f/\partial x$ | Linearization: saltation matrix $\Xi$ |
| No event-time correction needed | Must correct for the crossing-time shift $\delta t$ |

## 3. Deriving $\Xi$ - including a wrong turn, caught by verification

**A formula that looks standard and is wrong.** Before deriving anything, it's worth recording what didn't work, because it's the kind of formula that's easy to misremember and repeat. A plausible-looking candidate is

$$\Xi_{\text{wrong}} = DR + \frac{\big[f^{+}(x^{+}) - DR\,f^{-}(x^{-})\big] \otimes Dg}{Dg \cdot f^{-}(x^{-})}$$

($f^{-}$, $f^{+}$ the pre-/post-impact vector fields, $Dg$ the guard's gradient, $\otimes$ an outer product). It's dimensionally sensible and structurally plausible. It is also **not** the saltation matrix: checked against a from-scratch finite-difference ground truth (perturb the pre-impact state, re-land it on the guard via the pre-impact flow, apply the reset, compare to nominal), the two disagree by $\max|\Xi_{\text{wrong}} - \Xi_{\text{numeric}}| \approx 1.28$ - nowhere near floating-point noise.

This formula is not a fabrication, though - it is a real, standard expression from the hybrid-systems literature, which is exactly why it's easy to reach for. It answers a genuinely different question: it's the correct linearization when two trajectories are compared at a *shared* reference time (as in composing saltation matrices across several events into a single Poincaré-map derivative - see [§7](#7-connection-to-poincaré-maps) below), where the post-event vector field $f^{+}$ needs to be folded back in to account for continued flow past the shared time marker. $\Xi$ above answers a different question - "compare each trajectory at its *own* natural post-event time" - which is what propagating a covariance one event at a time actually needs, and it has no such term. Using the shared-time formula here silently smuggles in a spurious correction for a time shift this problem doesn't have.

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

## 4. A structural property that matters in practice: $Dg\,\Xi = 0$

For this guard ($g(x) = p_z$), $\Xi$'s output row for $p_z$ is identically zero, for *any* $x^{-}$, $e$, $g$:

$$Dg\,\Xi(x^{-}) = 0$$

Equivalently, in code: `Dg @ saltation_matrix(x_minus, e, g)` returns `array([0., 0., 0., 0., 0., 0.])`. This is not a bug - it's forced by the setup. Every trajectory in the family satisfies $g(x) = 0$ exactly at its own crossing (that's what "crossing the guard" means), and the reset map doesn't touch position, so the post-impact height is *exactly* zero for every member of the family, with zero sensitivity to any perturbation. $\Xi$ is correctly reporting that.

The catch: this exact-zero claim is only trustworthy if the filter's *own* estimated crossing time coincides exactly with the true one. It generally will not, once there is any tracking error at all - and comparing a covariance with an (near-)exactly-zero entry against a true trajectory sampled at a fixed tick, not at its own exact crossing, is a direct route to an artificially huge Mahalanobis distance if the two don't line up. `step_hybrid` adds a small isotropic "impact noise" floor after every bounce (identically for both filter variants) specifically to keep this otherwise-correct projection numerically usable; see its own docstring.

## 5. The empirical finding: not "naive is overconfident, saltation fixes it"

The intuitive story going in was that the naive $P^{+} = DR\,P^{-}\,DR^\top$ update would be measurably overconfident (too-small reported uncertainty) right after each bounce compared to the saltation-corrected one, and a Monte Carlo NEES (Normalized Estimation Error Squared) consistency check would show it. That is not what happens.

Running `saltation_matrix_ekf.py` at its defaults (300+ Monte Carlo trials, 3 well-separated bounces from a 5m drop at `e=0.85`) and looking at NEES in the few ticks immediately following each bounce (excluding the shared spike at the bounce tick itself, which both filters exhibit for the mundane reason that the true trajectory is also close to its own crossing at that tick):

```
Mean NEES, 5 ticks after each of 3 bounces (seed 0):
  EKF (naive)      ≈ 2.4
  EKF (saltation)  ≈ 3.6
```

**The saltation-corrected filter's post-bounce NEES is consistently *higher* than the naive filter's**, reproduced across multiple seeds (0, 1, 3 all show the same ~2.4 vs. ~3.6 pattern; one seed out of four tried landed in a rarer regime where both spike similarly, discussed in §6). Both stay well under the chi-squared 95% bound ($12.59$ for 6 DoF) at these settings - this is a modest, not catastrophic, effect, but it is the opposite direction from the naive expectation.

**Why**: §4's $Dg\,\Xi = 0$ property means the saltation-corrected update makes the *strongest possible claim* about the guard-normal direction - exactly zero residual uncertainty, beyond the small regularizing floor. The naive update, by contrast, just carries $P$'s existing height-variance forward unchanged ($DR$'s height row is untouched, $[0,0,1,0,0,0]$), a much more modest claim. Once the filter's own estimated bounce time inevitably differs even slightly from the true one - which is the normal case, not a corner case - the *stronger* claim is the one that gets punished harder. The mathematically exact *local* linearization is, in this specific practical sense, the more fragile one, precisely because it is exact only in the limit of infinitesimal tracking error, which real filters never have.

This is not a reason to prefer the naive update generally - averaged over the whole trajectory (not just the few ticks right after a bounce), the two filters are close, and the naive update has no principled derivation behind it at all, so its accidental "robustness" here isn't something to rely on either. It is, however, a concrete, quantified instance of exactly the gap already flagged as [Open Consideration #1](../../../../private-notes/PhD_Topic/unified_phd_plan.md#open-considerations-from-technical-review) in the dissertation plan: *saltation matrices assume the transition time is known exactly; contact/phase detection is itself uncertain.* A real Hybrid-InEKF implementation needs an explicit model of that detection uncertainty (e.g., an impact-timing noise term scaled by the velocity jump at the event, rather than this script's simple isotropic floor) - using the exact saltation matrix "as-is" is not automatically the safer choice once that uncertainty is real, which is always.

## 6. Two things worth knowing before reusing this pattern

- **Tune the regularizing floor deliberately, not just for numerical stability.** Sweeping `--impact-noise-std` from `0.0005` to `0.02` on this same problem flips which filter looks worse: at very small floors, saltation's exact zero-projection dominates and it comes out *dramatically* worse (NEES in the tens to hundreds vs. the naive filter's ~3); at larger floors (`≥0.02`), the floor itself swamps the structural difference and the two converge to nearly identical, unremarkable NEES. The `~2.4` vs. `~3.6` result quoted in §5 is specific to a floor sized to be "just barely enough" to avoid outright numerical pathology - a deliberate choice, not an arbitrary one, and worth re-checking whenever any of the other parameters change.
- **Stay well clear of the Zeno regime.** A lossy bounce (`e<1`) produces infinitely many, ever-faster bounces approaching a finite settling time (`~12.4s` for this script's defaults). Well before that time, bounce intervals become comparable to the fixed measurement step `dt`, and comparing a fixed-tick-sampled estimate against a true trajectory that's bouncing many times within a single tick breaks the entire comparison's premise (both filters' NEES explodes into the thousands, for reasons that have nothing to do with the saltation matrix). `--duration`'s default is deliberately chosen to stop after 3 clean, well-separated bounces and before the 4th starts crowding the settling regime.

## 7. Connection to Poincaré maps

Saltation matrices show up in a second, related context: analyzing the stability of a *periodic* hybrid trajectory - e.g., a robot repeatedly bouncing (or, for legged locomotion, repeatedly striking the ground once per stride). Sampling the state once per cycle, at a chosen event, defines a discrete return map $x_{k+1} = P(x_k)$, and its derivative $DP$ governs whether nearby trajectories converge back to the periodic orbit or diverge from it - the hybrid-systems analogue of eigenvalue stability analysis for a fixed point.

Over one cycle, $DP$ is a product of continuous-flow Jacobians and saltation matrices, one of each per phase and event the cycle passes through. Write $\Phi$ for the flow's own state-transition Jacobian over a smooth phase - the finite-time integral of the same $A = \partial f/\partial x$ from §2's table, concretely the `Phi` that this repo's `flow()` function already returns alongside the propagated state - and $\Xi$ for the saltation matrix at each event; e.g., for a cycle with three flow phases and two events in between:

$$DP \approx \Phi_3\,\Xi_2\,\Phi_2\,\Xi_1\,\Phi_1$$

This is the setting §3's "wrong turn" formula is actually built for: composing several such factors requires comparing every trajectory in the family against a single shared timeline running through the whole cycle, which is exactly what that formula's extra $f^{+}$ term supplies. This script never needs that composition - it propagates one covariance forward through one event at a time - so $\Xi$ alone, without the shared-timeline correction, is the right and complete tool here.

## Appendix: related terms

- **Guard condition/reset map**: the switching-surface function $g(x)=0$ and the (possibly discontinuous) map $R$ applied when a trajectory reaches it - the two ingredients that make a system "hybrid" rather than purely continuous.
- **NEES (Normalized Estimation Error Squared)**: $(x_{\text{true}} - x_{\text{est}})^\top P^{-1} (x_{\text{true}} - x_{\text{est}})$. A well-calibrated $n$-DoF filter's NEES should average to $n$ across many independent trials; systematically larger values mean the filter's reported $P$ is too small (overconfident) for the errors it's actually making. See [pose_graph_optimization.md §15.3](../optimization/pose_graph_optimization.md#153-objective-function) for the closely-related Mahalanobis-distance framing already used elsewhere in this repo.
- **Zeno behavior**: a hybrid system undergoing infinitely many discrete transitions in a finite time interval - the generic long-run behavior of any lossy bouncing system, and a standard pathology to guard against in hybrid-system simulation, not specific to saltation matrices themselves.
