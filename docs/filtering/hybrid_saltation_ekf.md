# Saltation matrices: propagating uncertainty through a hybrid/discontinuous event

## Intuition

A hybrid system alternates ordinary smooth motion (falling) with sudden, instantaneous jumps (bouncing), triggered the instant some condition is hit (touching the ground). The tricky part isn't the jump itself - it's that two nearby trajectories don't hit that condition at exactly the same moment:

```
ball A (nominal):     ●
                        ╲
                         ╲                    both start falling
                          ╲                    together...
ball B (started 5cm       ╲●
higher):                    ╲╲
                              ╲╲
──────────────────────────────●●──── ground
                               A B
                           A lands first, B a moment later -
                           having had farther to fall, B is
                           already moving faster when it lands
```

Ball A and ball B start almost identically - B is just 5 cm higher. But because B has slightly farther to fall, it lands a fraction of a second *later* than A, by which point B has picked up extra speed from that extra bit of falling time. So B's bounce isn't just "the same bounce rule applied to a slightly different starting state" - it's applied to a state that's had a bit more time to accelerate first.

An ordinary Jacobian (the reset map's own derivative, called $DR$ below) only captures the first part - how the bounce transforms a state already sitting at the ground - and implicitly assumes both balls land at the same instant. It has no way to see B's extra falling time. The **saltation matrix** ($\Xi$) is the correction for exactly that: it works out how much *extra time* a perturbed trajectory needs to reach the same ground condition, converts that timing difference into an equivalent velocity change, and only then applies the bounce. This doc's central finding is that skipping this correction isn't a minor rounding error - §3 below works a case where the naive approach predicts a position offset has *zero* effect on the bounce, when in fact it's the *entire* effect.

Every filter in [kf_ekf_iekf.md](kf_ekf_iekf.md) and [extra_kf_variants.md](extra_kf_variants.md) assumes the state evolves *continuously* between measurements. Bio-inspired locomotion (a footstep, an inchworm anchor/release cycle, a friction-anisotropic grip-then-slip transition) breaks that assumption on purpose, not by accident: the whole point of these platforms is to move in discrete, hybrid bursts. This doc works through the tool that lets an EKF cross one of those discontinuities without either (a) silently pretending nothing happened, or (b) discarding all uncertainty information at the jump - using [`saltation_matrix_ekf.py`](../../use_numpy/saltation_matrix_ekf.py)'s bouncing-point-mass toy problem as the concrete example.

> **Note (reading order)**: §1-§3 carry the core idea - why an ordinary reset Jacobian isn't enough, and a worked example showing it's wrong by a real, non-tiny amount. That's enough for a first read. §4 is a from-scratch derivation (with a documented wrong turn kept deliberately, as a teaching point - and one subtle enough that an earlier draft of this doc took the wrong turn itself and shipped it as the main formula for a while) - useful once you want to trust the formula yourself, skippable if you're willing to take it on faith for now. §5-§8 are the practical payoff (a structural property, a modest but real empirical finding, tuning gotchas, and a quantified account of what happens once contact detection itself is uncertain) and are worth reading even if §4 is skipped. §9 is an explicit tangent into a different, related context (periodic-orbit stability) that this script's own use case never needs - safe to skip entirely unless that's what brought you here.

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

The correct sensitivity has to account for that time-shift, whatever moment the two trajectories are ultimately compared at. That correct sensitivity is the **saltation matrix**, $\Xi$ - the ordinary reset Jacobian $DR$, corrected by a term for exactly this event-timing sensitivity. §4 below derives it in two stages: first a rank-one projector removing exactly the "along-the-flow" component of a perturbation - the part that only changes *when* the guard is crossed, not *where* - then one further term, needed once the two trajectories are compared at a shared later time rather than each at its own crossing moment (§3's worked example is built around exactly that shared-time comparison, since that's what this script's own filter needs).

At a smooth (non-event) point, the ordinary Jacobian $A = \partial f/\partial x$ is all there is - perturbations evolve continuously, and there is no timing ambiguity to correct for. At a hybrid event, $\Xi$ plays that same role but must additionally carry the event-time correction:

| Smooth dynamics | Hybrid event |
| --- | --- |
| Perturbation evolves continuously | Perturbation can jump discontinuously |
| Linearization: $A = \partial f/\partial x$ | Linearization: saltation matrix $\Xi$ |
| No event-time correction needed | Must correct for the crossing-time shift $\delta t$ |

## 3. A worked example: naive vs. saltation, in numbers

It's worth seeing the two approaches disagree on an actual number, not just abstractly - and worth being precise about *when* the two trajectories are compared, since that choice turns out to change which formula is correct (§4 below). `step_hybrid` always compares nominal and perturbed trajectories at a fixed external time (the next filter tick), so that's the comparison worked out here.

Take two point masses, both falling with the same downward velocity ($-2\,\text{m/s}$) from $5\,\text{m}$, except the second starts $5\text{cm}$ higher ($\delta p_z = 0.05$, no velocity perturbation at all), with restitution $e=0.5$. Both bounce once, and both are compared at a fixed later time $T=1.0\,\text{s}$ (long enough after the nominal bounce, at $t^{\*}\approx0.826\,\text{s}$, to leave $\approx\!0.174\,\text{s}$ of post-bounce flow for the timing difference to matter):

- **Naive prediction** ($DR$ alone, composed with the ordinary free-fall flow Jacobian before and after the bounce - exactly what `step_hybrid`'s `use_saltation=False` branch computes): change in $v_z(T)$: **exactly $0$**; change in $p_z(T)$: **$+0.05\,\text{m}$** (the original position offset, simply carried forward unchanged - $DR$'s position row is the identity, and it never sees the bounce at all).
- **True answer** (re-simulated: let the second ball reach its own crossing time, apply the reset there, then flow both forward to $T$ and compare): starting $5\text{cm}$ higher means falling for an extra $\sim\!4.9\text{ms}$ before impact, so it's moving faster when it lands and bounces back faster too - but it also spends *less* remaining time flowing back up before $T$. True change in $v_z(T)$: **$+0.0726\ \text{m/s}$**; true change in $p_z(T)$: **$-0.0126\,\text{m}$** (the perturbed ball, now moving faster, has climbed higher back up by $T$, so it's *lower* than nominal there — the opposite sign from the naive guess).
- **Saltation-corrected prediction** ($\Xi$, composed the same way as the naive case above - $\Phi_{\text{after}}\,\Xi\,\Phi_{\text{before}}$, exactly what `saltation_matrix` plus `step_hybrid`'s surrounding `flow` calls compute): change in $v_z(T)$: **$+0.0728\ \text{m/s}$**; change in $p_z(T)$: **$-0.0123\,\text{m}$** - both within second-order error of the true answer.

The naive approach isn't slightly off here; it's qualitatively wrong on both components: it says a pure position perturbation can never affect the bounce speed (true answer: it's the *entire* effect), and it gets the *sign* of the resulting position error backwards. $DR$ cannot see any of this because it has no notion of time at all - it only knows how to transform a state that is already sitting on the guard, and composing it with the ordinary flow Jacobians on either side doesn't fix that blind spot. $\Xi$ sees it because its correction term is built specifically to capture "this trajectory needed a different amount of falling time to get here, both before *and* after the bounce."

(Computed directly from this repo's own `flow`, `crossing_time`, `reset_map`, `reset_jacobian`, and `saltation_matrix` functions in `saltation_matrix_ekf.py`, not hand-derived - reproducible with $x_0=(0,0,5,0,0,-2)$, $\delta x_0=(0,0,0.05,0,0,0)$, $e=0.5$, $g=9.81$, $T=1.0$. Composing three matrices by hand isn't a natural calculator exercise the way a single scalar formula is, so unlike earlier drafts of this section there's no hand-arithmetic appendix for this version - see the [Appendix](#appendix-a-simpler-related-quantity-worked-by-hand) instead for a closely related, hand-workable quantity and how it connects to this one.)

## 4. Deriving $\Xi$ - including a wrong turn, caught by verification

**A formula that looks right and is right for a different question.** Before deriving the formula this script actually uses, it's worth recording what *doesn't* work here, because it's a natural first guess and it isn't simply wrong - it's the correct answer to a different, easily-conflated question. A first attempt at "the" saltation matrix, following the same implicit-differentiation idea as below but stopping short of accounting for the flow that continues *after* the reset, gives

$$\Xi_{\text{own-time}}(x^{-}) = DR(x^{-})\left[I - \frac{f(x^{-}) \otimes Dg(x^{-})}{Dg(x^{-}) \cdot f(x^{-})}\right]$$

This is the correct linearization of "post-impact state, compared at its *own* natural post-impact time" with respect to "pre-impact state, compared at its own natural pre-impact time" - verified against a from-scratch finite-difference ground truth built exactly that way (perturb the pre-impact state, re-land it on the guard via the pre-impact flow, apply the reset, compare to nominal, no further flow past the reset on either side) to $\sim\!2.9\times10^{-8}$, machine precision. It is a real, useful, correctly-derived quantity - it's the natural building block for comparing trajectories *event-to-event* rather than against a fixed external clock, which is exactly the [Poincaré-map](#9-connection-to-poincaré-maps) setting in §9 below.

It is, however, **not** what `step_hybrid` needs, and using it there was checked against this script's own finite-difference ground truth of the *actual* fixed-$`\Delta t`$ map (perturb the state at the start of a tick, run it through `step_hybrid`'s full-precision logic to the end of that same fixed-length tick, compare to nominal) and found wrong by a wide margin: $\max|\Phi_{\text{after}}\,\Xi_{\text{own-time}}\,\Phi_{\text{before}} - J_{\text{numeric}}| \approx 4.4$ - nowhere near floating-point noise. The reason is exactly the distinction §3 just worked a number for: `step_hybrid` (like any fixed-timestep filter) always compares nominal and perturbed states at the *same external clock time*, not at each trajectory's own crossing moment - and once the reference time is shared rather than per-trajectory, the flow continuing *after* the reset also depends on where exactly the reset landed, which $\Xi_{\text{own-time}}$ has no term for at all.

**The derivation this script actually needs.** Consider a one-parameter family of trajectories $x(t; p)$ ($p$ a perturbation parameter, $p=0$ nominal), each governed by $\dot x = f(x)$ until a $p$-dependent crossing time $t^{\*}(p)$ defined implicitly by $g\big(x(t^{\*}(p); p)\big) = 0$, where a reset $x^{+} = R(x^{-})$ is applied and the same flow resumes. Fix a later reference time $T$, independent of $p$, and ask for $\frac{d}{dp}\big[x(T;p)\big]$ - the quantity a fixed-tick filter's covariance propagation actually needs. Let $S(t) = \left.\dfrac{\partial x(t;p)}{\partial p}\right|_{p=0}$.

Implicit differentiation of the guard condition gives the same crossing-time sensitivity as before:

$$\frac{dt^{\*}}{dp} = -\frac{Dg \cdot S(t^{\*})}{Dg \cdot f^{-}(x^{-})}$$

Since $x(T;p) = \Phi\big(x^{+}(p),\,T-t^{\*}(p)\big)$ (the flow map, run for whatever time remains after the reset), the chain rule gives two contributions: how the post-reset *state* changes, and how the *remaining flow duration* changes as $t^{\*}(p)$ shifts:

$$\frac{d}{dp}\big[x(T;p)\big]\Big|_{p=0} = \Phi_{\text{after}}\cdot\frac{dx^{+}}{dp} - f_T\cdot\frac{dt^{\*}}{dp}$$

where $\Phi_{\text{after}} = D_1\Phi(x^{+}, T-t^{\*})$ is the ordinary flow Jacobian over the remaining duration (exactly `flow`'s own `Phi` output) and $f_T = f\big(x(T;0)\big)$ is the vector field at the nominal trajectory's own state at $T$. Expanding $\frac{dx^{+}}{dp} = DR\big[f^{-}(x^{-})\frac{dt^{\*}}{dp} + S(t^{\*})\big]$ (same reasoning as $\Xi_{\text{own-time}}$'s derivation) and using the standard flow identity $D_1\Phi(x,s)\cdot f(x) = f\big(\Phi(x,s)\big)$ - an autonomous flow's own linearization always carries its generating vector field forward to the vector field at the flowed-to point, so $\Phi_{\text{after}}\cdot f^{+}(x^{+}) = f_T$ - the $f_T$ terms combine and cancel the part $\Xi_{\text{own-time}}$ already had right, leaving exactly one new term:

$$\frac{d}{dp}\big[x(T;p)\big]\Big|_{p=0} = \Phi_{\text{after}}\left\lbrace DR(x^{-}) + \frac{\big[f^{+}(x^{+}) - DR(x^{-})\,f^{-}(x^{-})\big] \otimes Dg}{Dg \cdot f^{-}(x^{-})}\right\rbrace S(t^{\*}) = \Phi_{\text{after}}\,\Xi(x^{-})\,\Phi_{\text{before}}\,\delta x_0$$

where $\Phi_{\text{before}} = D_1\Phi(x_0, t^{\*})$ (so $S(t^{\*})=\Phi_{\text{before}}\,\delta x_0$) and

$$\boxed{\Xi(x^{-}) = DR(x^{-}) + \frac{\big[f^{+}(x^{+}) - DR(x^{-})\,f^{-}(x^{-})\big] \otimes Dg}{Dg \cdot f^{-}(x^{-})}}$$

is exactly $\Xi_{\text{own-time}}$ plus the extra correction term that formula was missing. This is what `saltation_matrix` computes, and $\Phi_{\text{after}}\,\Xi\,\Phi_{\text{before}}$ matches the finite-difference ground truth of the whole fixed-$`\Delta t`$ map to $\sim\!7\times10^{-6}$ (limited by the finite-difference step itself, not this formula) - see this module's tests for the check.

## 5. A structural property that matters in practice: $Dg\,\Xi = -e\,Dg$

For this guard ($g(x) = p_z$), $\Xi$'s output row for $p_z$ scales by exactly $-e$, for *any* $x^{-}$, $e$, $g$:

$$Dg\,\Xi(x^{-}) = -e\,Dg$$

Equivalently, in code: `Dg @ saltation_matrix(x_minus, e, g)` always equals `-e * Dg`. This is *reduced*, not zero: unlike $\Xi_{\text{own-time}}$ from §4 (whose guard-normal row *is* identically zero, since every trajectory in that comparison satisfies $g(x)=0$ exactly at its own crossing by construction), $\Xi$ compares trajectories at a fixed later time, by which point the guard-normal coordinate has evolved under real post-bounce flow and is no longer pinned to any particular value - so its sensitivity to a perturbation is genuinely nonzero, just scaled down by the restitution coefficient.

The practical consequence is the opposite of what an exactly-zero claim would produce: $\Xi$ does not drive $P$'s guard-normal row/column to a near-singular state at every bounce, so it needs no artificial regularizing floor to stay numerically well-behaved against a true trajectory sampled at fixed ticks rather than at its own exact crossing (§7 below measures this directly). `step_hybrid`'s small isotropic "impact noise" floor remains in the code as a simplified stand-in for the explicit impact-timing/model-uncertainty term a real contact-aided filter would carry - not, as an earlier draft of this section claimed, a numerical necessity forced by an exact-zero projection.

## 6. The empirical finding: a modest, real gap, not a dramatic one

The intuitive story going in was that the naive $P^{+} = DR\,P^{-}\,DR^\top$ update would be measurably overconfident (too-small reported uncertainty) right after each bounce compared to the saltation-corrected one, and a Monte Carlo NEES (Normalized Estimation Error Squared - a per-trial score for how far the true state falls from the estimate relative to how much uncertainty $P$ claims; a well-calibrated 6-DoF filter should average NEES $\approx 6$, and *persistently* higher values mean $P$ is too small for the errors actually being made - full formula in the [Appendix](#appendix-related-terms)) consistency check would show it. That is roughly true, but only barely.

Running `saltation_matrix_ekf.py` at its defaults (500 Monte Carlo trials, 3 well-separated bounces from a 5m drop at $e = 0.85$) and looking at NEES in the few ticks immediately following each bounce (excluding the shared spike at the bounce tick itself, which both filters exhibit for the mundane reason that the true trajectory is also close to its own crossing at that tick):

```
Mean NEES, 5 ticks after each of 3 bounces (seed 0):
  EKF (naive)      ≈ 2.35
  EKF (saltation)  ≈ 2.49
```

**The saltation-corrected filter's post-bounce NEES runs slightly *higher* than the naive filter's** - about a 5-6% gap, reproduced tightly across seeds 0-3 (naive 2.35-2.38, saltation 2.49-2.50 in every one). Both are far below the chi-squared 95% bound ($12.59$ for 6 DoF), and the gap itself is small enough that it would be easy to miss as noise if it weren't this consistent across seeds - it is real, but it is not the dramatic, order-of-magnitude-scale effect an incorrect formula (one missing §4's $f^{+}$ correction term) would produce here instead.

**Why**: §5's $Dg\,\Xi = -e\,Dg$ property means the saltation-corrected update makes a *somewhat stronger* claim about the guard-normal direction than the naive update does (which just carries $P$'s existing height-variance forward roughly unchanged, composed only with the ordinary flow Jacobians). Once the filter's own estimated bounce time inevitably differs even slightly from the true one - the normal case, not a corner case - the *slightly stronger* claim is the one that absorbs slightly more of that mismatch. This is the same qualitative story the (now-corrected) code always intended to tell, just far more muted in practice than an incorrect formula would have suggested: correctly and precisely modeling a real effect is not automatically free of some fragility to the modeling assumption (known transition time) not holding exactly - it's just that here, unlike with the rejected $\Xi_{\text{own-time}}$ formula, that fragility is modest rather than severe.

This is not a reason to prefer the naive update - the two filters are close both right after a bounce and averaged over the whole trajectory, and the naive update has no principled derivation behind it at all, so there's no real tradeoff to weigh here. A real Hybrid-InEKF implementation would still benefit from an explicit model of detection uncertainty (e.g., an impact-timing noise term scaled by the velocity jump at the event, rather than this script's simple isotropic floor) - the gap this section measures is small, not absent.

## 7. Two things worth knowing before reusing this pattern

- **The regularizing floor is no longer numerically load-bearing, but still worth tuning deliberately.** Sweeping `--impact-noise-std` from `0` (fully disabled) to `0.02` on this same problem no longer flips which filter looks worse the way an exact-zero-projection formula would: naive and saltation move together and stay close at every setting (`0`: 2.94 vs. 3.05; `0.0005`: 2.93 vs. 3.04; `0.005`, this script's default: 2.35 vs. 2.49; `0.02`: 2.28 vs. 2.32) - both a sign that §5's $Dg\,\Xi = -e\,Dg$ genuinely fixed the near-singularity problem the floor originally existed to paper over, and a reminder that the floor still shifts *both* filters' absolute NEES level together as it grows, so it's still a real modeling choice, just no longer one that can flip the qualitative story between the two filters.
- **Stay well clear of the Zeno regime.** A lossy bounce ($e<1$) produces infinitely many, ever-faster bounces approaching a finite settling time (`~12.4s` for this script's defaults). Well before that time, bounce intervals become comparable to the fixed measurement step $\Delta t$, and comparing a fixed-tick-sampled estimate against a true trajectory that's bouncing many times within a single tick breaks the entire comparison's premise (both filters' NEES explodes into the thousands, for reasons that have nothing to do with the saltation matrix). `--duration`'s default is deliberately chosen to stop after 3 clean, well-separated bounces and before the 4th starts crowding the settling regime.

## 8. Quantifying contact-detection timing jitter

§6 already showed that ordinary state-estimation error alone (with detection otherwise exact) makes the saltation-corrected filter's post-bounce NEES slightly *higher* than the naive filter's, not lower - a modest instance of the same gap explored further here. That leaves open exactly how much worse a *real* contact sensor's own detection latency/jitter makes things, on top of that. `step_hybrid`'s `detect_time_bias`/`detect_time_noise_std` parameters model this directly (both default $0$, reproducing every result above): a bounce's reset is applied at a *detected* crossing time

$$\tau_{\text{detect}} = \text{clip}\big(\tau + b_{\text{detect}} + \mathcal{N}(0,\,\sigma_{\text{detect}}^2)\,,\ 0,\ \tau_{\text{remain}}\big)$$

instead of the true geometric $\tau$, so early detection resets the mean while still above ground and late detection resets it after the free-fall model has carried it slightly below $p_z = 0$ - both real artifacts of a delayed or jittery contact detector (IMU spike, force threshold - Čížek et al. 2018), not numerical noise.

**A correctness subtlety this surfaced**: once a reset can land off-guard, `crossing_time` needs to reject *ascending* roots (the point mass rising back out of a below-ground detection artifact crosses $p_z = 0$ again almost immediately, which is not a real impact) and keep only the next *descending* one. Every crossing this script computed before `detect_time_*` existed was already descending by construction, so this is a latent-bug fix with zero effect on any result above - but skipping it turns on nonsense: NEES and RMS position error explode by 4-5 orders of magnitude the instant any detection jitter is introduced, from a cascade of spurious re-bounces within a single step, not from the phenomenon actually being modeled. See `crossing_time`'s docstring for the concrete numbers that exposed it.

With that fixed, sweeping $\sigma_{\text{detect}}$ (with $b_{\text{detect}} = 0$, i.e. jitter only, no systematic delay) at $\Delta t = 0.02\,\text{s}$, averaged over seeds 0/1/3 (mean NEES over the same 5-tick post-bounce window as §6, individual seeds agreeing to within ~5% of each other at every setting):

| $\sigma_{\text{detect}}$ (fraction of $\Delta t$) | EKF (naive) | EKF (saltation) | saltation / naive |
| --- | --- | --- | --- |
| $0$ (exact detection, §6's result) | 2.37 | 2.49 | $1.05\times$ |
| $0.001\,\text{s}$ (5%) | 4.10 | 5.25 | $1.28\times$ |
| $0.002\,\text{s}$ (10%) | 9.30 | 13.55 | $1.46\times$ |
| $0.005\,\text{s}$ (25%) | 38.64 | 60.66 | $1.57\times$ |
| $0.01\,\text{s}$ (50%) | 87.67 | 141.29 | $1.61\times$ |

**The finding**: both filters' consistency degrades monotonically as detection jitter grows - unsurprising, since neither one's $P$ accounts for this extra error source at all. The naive filter's absolute numbers here are unaffected by §4's formula correction (its own covariance update never used `saltation_matrix` in the first place); only the saltation column, and therefore the ratio, changed. The *gap* between them does still widen as jitter grows, from §6's already-present $1.05\times$ at exact detection to $1.6\times$ once jitter reaches half the tick interval - a real, monotonic trend, and a genuine (if now modest) instance of "saltation matrices assume a known transition time, and that assumption isn't free." What's no longer true is the earlier, much larger claim this section used to make: with §4's $f^{+}$ term correctly included, $Dg\,\Xi = -e\,Dg$ (§5), not exactly zero, so the saltation-corrected filter never stakes *everything* on the detected crossing being exactly right the way an incomplete formula would - it is somewhat more sensitive to detection jitter than the naive update, not dramatically so.

## 9. Connection to Poincaré maps

Saltation matrices show up in a second, related context: analyzing the stability of a *periodic* hybrid trajectory - e.g., a robot repeatedly bouncing (or, for legged locomotion, repeatedly striking the ground once per stride). Sampling the state once per cycle, at a chosen event, defines a discrete return map $x_{k+1} = P(x_k)$, and its derivative $DP$ governs whether nearby trajectories converge back to the periodic orbit or diverge from it - the hybrid-systems analogue of eigenvalue stability analysis for a fixed point.

Over one cycle, $DP$ is a product of continuous-flow Jacobians and saltation matrices, one of each per phase and event the cycle passes through: $DP \approx \Phi_3\,\Xi_2\,\Phi_2\,\Xi_1\,\Phi_1$ for a cycle with three flow phases and two events in between, where each $\Phi_i$ is the flow's own Jacobian over phase $i$ and each $\Xi_i$ the saltation matrix at event $i$. Unlike `step_hybrid`'s fixed-$`\Delta t`$ comparison (§3-§4), a Poincaré return map samples the state at the *same kind of event* every cycle - so every reference point in this composition, including each intermediate $\Phi_i$'s endpoint, is itself an event-crossing that a perturbed trajectory reaches at its own natural time, not a shared external clock. That is the setting $\Xi_{\text{own-time}}$ from §4 (the formula this script's own `saltation_matrix` no longer computes) is built for, and it's the quantity classically used in this composition in the hybrid-systems literature this doc draws on.

This script doesn't build or verify that composition itself - it propagates one covariance forward through one fixed-$`\Delta t`$ tick at a time, which is exactly the different setting §3-§4 work out in detail - so this section stays a pointer to where saltation matrices show up next, not a second worked derivation. Anyone extending this toy into a periodic-orbit-stability tool should re-derive and numerically verify $DP$'s formula directly (the same finite-difference discipline §4 uses here), rather than assuming either this script's $\Xi$ or $\Xi_{\text{own-time}}$ carries over to a multi-phase composition without a fresh check.

## Appendix: related terms

- **Guard condition/reset map**: the switching-surface function $g(x)=0$ and the (possibly discontinuous) map $R$ applied when a trajectory reaches it - the two ingredients that make a system "hybrid" rather than purely continuous.
- **NEES (Normalized Estimation Error Squared)**: $(x_{\text{true}} - x_{\text{est}})^\top P^{-1} (x_{\text{true}} - x_{\text{est}})$. A well-calibrated $n$-DoF filter's NEES should average to $n$ across many independent trials; systematically larger values mean the filter's reported $P$ is too small (overconfident) for the errors it's actually making. See [pose_graph_optimization.md §15.3](../optimization/pose_graph_optimization.md#153-objective-function) for the closely-related Mahalanobis-distance framing already used elsewhere in this repo.
- **Zeno behavior**: a hybrid system undergoing infinitely many discrete transitions in a finite time interval - the generic long-run behavior of any lossy bouncing system, and a standard pathology to guard against in hybrid-system simulation, not specific to saltation matrices themselves.

### Appendix: a simpler, related quantity worked by hand

§3's fixed-time comparison needs three composed matrices and isn't a natural hand-arithmetic exercise. $\Xi_{\text{own-time}}$ from §4 - the formula this script's `saltation_matrix` *used to* compute, still correct for the different, event-to-event question it answers - reduces to a single scalar correction and is worth working out by hand once, with a calculator, no code required. Uses the same setup as §3's opening (before it moves to a fixed comparison time): $x^{-}_0=(0,0,5,0,0,-2)$ (so $p_z^0=5$, $v_z^0=-2$), $\delta x_0=(0,0,0.05,0,0,0)$ (so $\delta p_z^0 = 0.05$), $e=0.5$, $g=9.81$, comparing each trajectory at its *own* post-bounce moment (no shared fixed time here, unlike §3).

**Nominal crossing.** Free-fall height is $p_z(t) = p_z^0 + v_z^0 t - \tfrac12 g t^2$. Solve $5 - 2t - 4.905t^2 = 0$ for the positive root:

$$t^{\*} = \frac{-2+\sqrt{2^2+4(4.905)(5)}}{2(4.905)} = \frac{-2+\sqrt{102.1}}{9.81} \approx 0.82614\ \text{s}$$

Velocity there: $v_z^{-} = -2 - 9.81(0.82614) \approx -10.1045\ \text{m/s}$. Post-bounce: $v_z^{+} = -e\,v_z^{-} \approx +5.0522\ \text{m/s}$.

**Naive prediction ($0$).** $DR=\text{diag}(1,1,1,1,1,-e)$ only touches the velocity slot, and $\delta x_0$'s velocity slot is $0$: a pure position offset stays a pure position offset under free-fall (position never feeds back into the dynamics), so $DR$ is handed $0$ in that slot and returns $0$. No simulation needed - just that $DR$'s only nonzero action lands on a component that's zero here. (This is also, incidentally, the same $0$ §3 quotes for its own naive prediction at $v_z(T)$ - not a coincidence, since $DR$'s blindness to a pure position offset doesn't depend on how far past the bounce the comparison happens.)

**True answer, at the ball's own crossing ($+0.0242$).** Solve the same quadratic with $p_z^0=5.05$ (the perturbed ball's own crossing):

$$t^{\*}_{\text{pert}} = \frac{-2+\sqrt{4+4(4.905)(5.05)}}{9.81} \approx 0.83108\ \text{s}, \qquad \delta t \approx 0.004936\ \text{s (}\sim\!4.94\text{ms extra fall)}$$

$v_z^{-}(\text{pert}) = -2-9.81(0.83108)\approx -10.1529\ \text{m/s}$, so $v_z^{+}(\text{pert}) = -e\,v_z^{-}(\text{pert}) \approx +5.0764\ \text{m/s}$.

$$\text{True change} = 5.0764-5.0522 \approx +0.02421\ \text{m/s}$$

(Note this is a *different* true answer from §3's $+0.0726\,\text{m/s}$ - that one compares both balls at a fixed later time $T=1.0\,\text{s}$, including whatever post-bounce flow happens before $T$; this one compares each ball right at its own bounce, with no post-bounce flow at all. Different questions, different true answers - which is exactly §4's point about $\Xi_{\text{own-time}}$ and $\Xi$ not being interchangeable.)

**$\Xi_{\text{own-time}}$ prediction ($+0.0243$).** From §4's formula $\Xi_{\text{own-time}} = DR\left[I - \dfrac{f(x^{-})\otimes Dg}{Dg\cdot f(x^{-})}\right]$, the scalar $\dfrac{Dg\cdot\delta x_0}{Dg\cdot f(x^{-})} = \dfrac{0.05}{-10.1045}\approx -0.004948$ is precisely §4's linearized crossing-time-shift formula, $dt^{\*}/dp$, evaluated here - compare it to the true $-\delta t \approx -0.004936$ just above: close but not identical, since it's only a first-order estimate of the shift.

Write $B(x^{-}) = I - \dfrac{f(x^{-})\otimes Dg}{Dg\cdot f(x^{-})}$ for the rank-one projector in brackets above (so $\Xi_{\text{own-time}} = DR\,B$) - the $B$-projector step turns that position offset into an *equivalent velocity offset*, using $f_5=-g$ (the $v_z$-component of $f$, i.e. gravity, is constant):

$$(B\,\delta x_0)_{v_z} = 0-(-9.81)(-0.004948) \approx -0.04854\ \text{m/s}$$

Read this as: *starting 5cm higher behaves, to first order, like starting at the same height but already falling $\approx 0.0485$ m/s faster* - which is a language $DR$ already knows how to handle. Applying $DR$'s bounce law to that equivalent velocity gives the $\Xi_{\text{own-time}}$ prediction:

$$\Xi_{\text{own-time}}\text{ change} = -e\times(-0.04854) \approx +0.02427\ \text{m/s}$$

matching the true (own-crossing-time) answer to within second-order error, same as before. This scalar shortcut doesn't extend to $\Xi$ (§4's boxed formula, what `saltation_matrix` actually computes): the extra $f^{+}$ term and the subsequent composition with $\Phi_{\text{before}}$/$`\Phi_{\text{after}}`$ genuinely need matrices, which is why §3 computes its numbers from code rather than by hand.

---

## References

1. Kong, N. J., Payne, J. J., Zhu, J., & Johnson, A. M. (2023). *Saltation Matrices: The Essential Tool for Linearizing Hybrid Dynamical Systems*. arXiv:2306.06862. https://doi.org/10.48550/arXiv.2306.06862 - a modern tutorial/survey on saltation matrices, covering the same $\Xi(x^-)$ derivation this doc works through in §4, including the own-crossing-time vs. fixed-external-time distinction at the center of §4's "wrong turn."
2. Kong, N. J., Payne, J. J., Council, G., & Johnson, A. M. (2021). *The Salted Kalman Filter: Kalman Filtering on Hybrid Dynamical Systems*. Automatica, 131, 109752. https://doi.org/10.1016/j.automatica.2021.109752 - the direct precedent for this doc's whole exercise: propagating a Kalman filter's covariance correctly through a hybrid guard-crossing event via the saltation matrix, exactly what `saltation_matrix_ekf.py` implements for a single bounce.
