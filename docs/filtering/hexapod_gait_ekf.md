# Hexapod leg odometry: what the gait changes

## Intuition

A planted foot doesn't move. So if the joint encoders say a stance foot is sliding backward under the body at 0.1 m/s, the body is moving forward at 0.1 m/s. That is **leg odometry**: every foot on the ground is a velocity sensor for the body ([glossary](../glossary.md#odometry)).

A hexapod has six legs, and its **gait** decides which ones are down on each **tick** (one filter time step, $`\Delta t = 0.02`$ s):

```text
            L1 ●  ● R1          tripod   duty 1/2   3 feet down   L1,R2,L3 alternate with R1,L2,R3
 (front)  L2 ●  body  ● R2      ripple   duty 2/3   4 feet down   one leg per side swings, back to front
            L3 ●  ● R3          wave     duty 5/6   5 feet down   one leg swings at a time
```

A foot on the ground is in **stance**; for the rest of the cycle it is in **swing**, moving through the air from **liftoff** to **touchdown** at its next foothold. The **duty factor** is the fraction of a cycle each foot spends in stance. More feet down means more velocity readings per tick, but a slower walk on a real robot.

![Three panels from hexapod_gait_ekf.py: gait diagrams of stance and swing per leg for the tripod, ripple and wave gaits over two cycles; the hexapod body turning left with an arrow on each foot showing its measured velocity relative to the body, six different readings; and two three-foot stance layouts, a tripod triangle and three feet on one side, with their one-tick standard deviations of forward speed and turn rate](../../assets/hexapod_gait_ekf_concept.png)

*Figure: the gait schedules and stance geometry of `use_numpy/hexapod_gait_ekf.py`, plotted by `uv run python assets/make_figures.py hexapod_gait_ekf_concept`.*

Four questions, answered with [`hexapod_gait_ekf.py`](../../use_numpy/hexapod_gait_ekf.py):

1. **No slip:** how much does the gait change the estimate? (§2)
2. **One leg slips:** a slipping foot breaks "a planted foot doesn't move". What does that do, and can the filter catch it? (§3)
3. **Every foot drifts a little:** what if planted feet are never perfectly still? (§4)
4. **Contact timing is uncertain:** what if the filter's touchdown and liftoff times are off? (§5)

**Scope.** Planar body, perfect kinematics (the filter knows where each stance foot is; only its velocity reading is noisy), a known stance schedule except in §5, no IMU ([glossary](../glossary.md#imu-inertial-measurement-unit)). §5 adds timing errors to touchdown and liftoff; how a detector finds them is out of scope (see [`hybrid_saltation_ekf.md` §8](hybrid_saltation_ekf.md#8-quantifying-contact-detection-timing-jitter)).

---

## 1. Model

**Why planar, not 1D.** In 1D every stance foot measures the same scalar velocity, so six legs only average noise: variance $`\sigma^2/n`$, with $n$ the number of stance feet and $\sigma$ the noise std of one reading. In the plane, a foot at body-frame offset $`r_i`$ (its position relative to the body centre) sees the body's velocity *plus* the turn: when the body turns in place at $`\omega = 0.2`$ rad/s, a foot 0.15 m to the left moves at 0.03 m/s and its mirror foot at −0.03 m/s. Feet in different places therefore separate forward speed from turn rate.

**State.** $`x = [p_x, p_y, \theta, u, v, \omega]`$: world position, heading, body-frame velocity $`b = [u, v]`$ (forward, left) and turn rate. Velocity and turn rate together are the body's **twist** ([glossary](../glossary.md#twist)). **Heading** $`\theta`$ (also called yaw) is the body's orientation in the plane: the angle from the world x-axis to the body's forward axis. It is not the direction of travel $`\operatorname{atan2}`$ of the world velocity. The two differ whenever the robot steps sideways ($`v \neq 0`$), which a hexapod can do. **Turn rate** $`\omega = \dot\theta`$ (yaw rate, rad/s, positive counter-clockwise, i.e. turning left) is how fast the body rotates, not how sharply its path curves: turning in place has $`\omega \neq 0`$ at zero speed.

**Predict** (`predict`): constant twist, pose (position and heading, [glossary](../glossary.md#pose)) integrated with it. Nonlinear in $`\theta`$, hence an EKF ([glossary](../glossary.md#ekf-extended-kalman-filter)):

```math
p_{k+1} = p_k + R(\theta_k)\, b_k\, \Delta t, \qquad \theta_{k+1} = \theta_k + \omega_k \Delta t, \qquad [b, \omega]_{k+1} = [b, \omega]_k + w_k
```

$`R(\theta)`$ is the 2D rotation by $\theta$, taking body-frame vectors to the world frame. $`w_k`$ is the process noise on the twist: each tick adds an independent Gaussian step, so the twist does a **random walk**, with step std $`\sigma_a \Delta t`$ for $`u, v`$ ($`\sigma_a = 0.05\ \text{m/s}^2`$) and $`\sigma_\alpha \Delta t`$ for $`\omega`$ ($`\sigma_\alpha = 0.1\ \text{rad/s}^2`$). The truth twist is drawn from exactly this random walk, so $Q$, the covariance of $`w_k`$ ([glossary](../glossary.md#predict-and-update-steps)), is matched and NEES is a fair test ([glossary](../glossary.md#consistency-and-nees)).

**Measure** (`measurement_update_feet`): each stance foot $i$ reports its velocity relative to the body, in the body frame:

```math
z_i = -\left(b + \omega\, J r_i\right) + n_i, \qquad J = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}, \qquad n_i \sim \mathcal{N}(0,\ \sigma^2 I), \qquad \sigma = 0.02\ \text{m/s}
```

$`\omega J r_i`$ is the planar $`\omega \times r_i`$. The model is linear in the state, and its measurement Jacobian $`H_i`$ ([glossary](../glossary.md#jacobian)) only involves the twist columns:

```math
H_i = \begin{bmatrix} 0 & 0 & 0 & -1 & 0 & r_{i,y} \\ 0 & 0 & 0 & 0 & -1 & -r_{i,x} \end{bmatrix}
```

All stance feet enter one **joint update** per tick: their readings are stacked into a single measurement.

**What is never measured.** No reading involves $`p_x, p_y, \theta`$, so they are unobservable ([glossary](../glossary.md#observability)): leg odometry integrates the twist, and position and heading drift. This matches the real-robot result that, with leg kinematics and an IMU, everything except global position and yaw is observable (Bloesch et al. 2013, ref. 3).

**Controlled comparison** (`simulate_trial`): each Monte Carlo trial ([glossary](../glossary.md#monte-carlo-trials)) draws one true trajectory, one initial-guess error and one noise stream per foot, and every gait uses the same draws. Only which feet are down, and where, changes between gaits.

**No saltation matrix here.** In [`hybrid_saltation_ekf.md` §4](hybrid_saltation_ekf.md#4-deriving-xi---including-a-wrong-turn-caught-by-verification), an impact is an event that *jumps the state*, and the saltation matrix carries the covariance across it:

```math
\Xi = DR + \frac{\big[f^{+} - DR\,f^{-}\big] \otimes Dg}{Dg \cdot f^{-}}
```

The symbols follow [`hybrid_saltation_ekf.md` §1](hybrid_saltation_ekf.md#1-what-a-hybrid-dynamical-system-is-here) ([glossary](../glossary.md#hybrid-system-guard-and-reset-map)): the **guard** $`g(x) = 0`$ triggers the jump and $Dg$ is its gradient; $R$ is the **reset map** applied at the jump (not the rotation $`R(\theta)`$ above) and $DR$ its Jacobian; $`f^{-}`$ and $`f^{+}`$ are the dynamics just before and after the jump; $\otimes$ is the outer product.

A touchdown here only changes *which feet are measured*. The legs have no mass and the schedule is set by the clock, so:

- the body state does not jump: $`DR = I`$;
- the body moves the same way before and after: $`f^{+} = f^{-}`$;
- the timing term exists because a perturbed state reaches the guard at a different moment; here the clock sets the touchdown time, so a perturbed body lands at the same moment (and the term's numerator, $`f^{+} - DR\, f^{-}`$, is 0 anyway).

So $`\Xi = DR = I`$ exactly, and the timing-noise term of [`invariant_saltation_ekf.md` §7](invariant_saltation_ekf.md#7-contact-detection-jitter), $`\sigma_{\text{detect}}^2\, w w^\top`$ with $`w = f^{+} - DR\, f^{-}`$ and $`\sigma_{\text{detect}}`$ the timing-error std, is exactly 0. A wrong touchdown time shows up in the *readings* instead (§5). A saltation matrix would matter only if touchdown changed the body state, e.g. an impact with height and pitch in the state, as in the bouncing-body docs ([`hybrid_saltation_ekf.md`](hybrid_saltation_ekf.md), [`invariant_saltation_ekf.md`](invariant_saltation_ekf.md)).

---

## 2. No slip: the gait matters, but less than you'd think

**Intuition first.** One tick's readings pin down the body velocity to about $`\sigma/\sqrt{n}`$; turn rate shrinks at the same rate. The filter, though, averages over many ticks against a twist that slowly wanders, and that averaging shrinks the gait's effect to about $`n^{-1/4}`$.

**One tick, no filter** (`twist_geometry_std`: std from one tick's readings alone, mean over ticks; deterministic):

| Gait | Feet down | $`u`$ std (m/s) | $`\omega`$ std (rad/s) | $`\sigma/\sqrt{n}`$ (m/s) |
| --- | --- | --- | --- | --- |
| tripod | 3 | 0.0117 | 0.0497 | 0.0115 |
| ripple | 4 | 0.0100 | 0.0431-0.0432 | 0.0100 |
| wave | 5 | 0.0090-0.0091 | 0.0386-0.0388 | 0.0089 |

- With these gaits the **count** explains almost everything: $`u`$ std sits at $`\sigma/\sqrt{n}`$, and wave/tripod $`\omega`$ std is $`0.0388/0.0497 = 0.78 \approx \sqrt{3/5}`$.
- **Where the feet are** matters when the stance is lopsided. Three feet, each at its **nominal offset** (its default position relative to the body centre, about which it sweeps during stance), in two layouts:

| 3 feet at nominal offsets | $`u`$ std (m/s) | $`v`$ std (m/s) | $`\omega`$ std (rad/s) |
| --- | --- | --- | --- |
| tripod triangle L1, R2, L3 | 0.0117 | 0.0115 | 0.0497 |
| one side L1, L2, L3 | 0.0164 | 0.0115 | 0.0700 |

  Three feet in a line along one side are about 40% worse in both $`u`$ and $`\omega`$: forward speed and turn rate both show up as along-body motion of those feet, so they are harder to tell apart. The tripod's alternating triangle avoids this. All three standard gaits keep their stance feet spread on both sides; within a cycle, the wave gait's one-tick $`\omega`$ std only ranges over 0.0375-0.0413 rad/s (`twist_geometry_std` per tick, seed 0).

**The filter** (`run_monte_carlo`, 100 trials, seeds 0-2, $`\Delta t = 0.02`$ s, 12 s). **RMSE** is the root-mean-square error ([glossary](../glossary.md#rmse-root-mean-square-error)): the squared error averaged over trials and ticks, then square-rooted; "final" RMSE uses only the last tick. Each range in this and the later results tables is the min-max over three runs of the script, `--seed` 0, 1 and 2.

| Gait | RMSE $`u`$ (m/s) | RMSE $`\omega`$ (rad/s) | final position RMSE (m) | final heading RMSE (rad) | NEES full / twist |
| --- | --- | --- | --- | --- | --- |
| tripod | 0.0034-0.0035 | 0.0100-0.0103 | 0.024-0.025 | 0.026-0.029 | 5.79-6.23 / 2.98-3.02 |
| ripple | 0.0031-0.0032 | 0.0093-0.0095 | 0.020-0.025 | 0.021-0.024 | 5.56-6.25 / 2.96-3.01 |
| wave | 0.0030 | 0.0088-0.0090 | 0.020-0.023 | 0.020-0.021 | 5.61-6.28 / 2.99-3.00 |

- **Consistent.** A consistent filter averages NEES 6 (full state) and 3 (twist); the 95% intervals for a 100-trial average are $`[5.34, 6.70]`$ and $`[2.54, 3.50]`$. Every gait is inside both.
- **Wave beats tripod by 12%, not 22%.** One tick: 22% lower std. Filter: 12% lower RMSE.
- **Why $`n^{-1/4}`$.** For a random walk with per-tick variance $q$, measured each tick with variance $`s^2`$ and $`q \ll s^2`$, the steady-state variance is about $`\sqrt{q\, s^2}`$, so the std scales as $`s^{1/2}`$, and $`s \propto 1/\sqrt{n}`$. Here $s$ is the one-tick std from the first table. Worked for tripod $`u`$: $`q = (0.05 \times 0.02)^2 = 10^{-6}`$, $`s = 0.0117`$, so $`\sqrt[4]{q\, s^2} = 0.0034`$ m/s, the RMSE in the table. The same formula gives 0.0100 rad/s for tripod $`\omega`$ and 0.0030 / 0.0088 for wave, all within 0.0003 of the Monte Carlo values. $`(3/5)^{1/4} = 0.88`$ is the 12%.
- **Drift is the same story.** Position and heading are integrals of the twist error, so wave drifts slightly less (heading 0.020-0.021 vs 0.026-0.029 rad after 12 s), but no gait stops the drift.

![Three panels from hexapod_gait_ekf.py: bar charts of forward-speed and turn-rate RMSE for the tripod, ripple and wave gaits, each with its (q s^2)^(1/4) steady-state estimate drawn as a line; and the sideways-velocity RMSE over time with leg L2 slipping between 4 and 8 s, for the tripod and wave gaits, trusting every foot versus the per-foot chi-squared gate](../../assets/hexapod_gait_ekf.png)

*Figure: `use_numpy/hexapod_gait_ekf.py` at its defaults (seed 0, 100 trials), plotted by `uv run python assets/make_figures.py hexapod_gait_ekf`. The right panel is §3's slip experiment.*

---

## 3. One slipping leg: the gait barely matters, gating does

L2 slips outward at 0.1 m/s for $`t \in [4, 8)`$ s whenever it is down. Its reading is then biased by 0.1 m/s, and the filter's model says it is zero.

**Intuition.** In the joint update, one bad foot out of $n$ pulls the estimate by about $`1/n`$ of its bias: $`0.1/3 = 0.033`$ m/s with a tripod, $`0.1/5 = 0.020`$ m/s with a wave gait. The peaks of the sideways velocity RMSE match (seed 2, 100 trials): 0.0316 (tripod), 0.0254 (ripple), 0.0206 m/s (wave). But the slipping foot is down for 1/2 of the cycle in a tripod and 5/6 in a wave gait, so the wave gait's smaller error lasts longer. Over the window the two largely cancel.

**Per-foot gating** (`gating=True`): before the joint update, test each stance foot's **innovation** $`\nu_i = z_i - H_i \hat{x}^-`$ against the predicted state $`\hat{x}^-`$ and covariance $`P^-`$ ([glossary](../glossary.md#innovation-and-kalman-gain), [predict step](../glossary.md#predict-and-update-steps)):

```math
\nu_i^\top \left(H_i P^- H_i^\top + \sigma^2 I\right)^{-1} \nu_i \le 9.21
```

9.21 is the 99% point of a $`\chi^2`$ with 2 DoF ([glossary](../glossary.md#chi-square-distribution)), so a good foot is wrongly dropped 1% of the time. Feet that fail are left out ([glossary](../glossary.md#outlier-and-robust-kernel)). Testing each foot against the prediction, not after other feet's updates, makes the result independent of foot order.

Results (100 trials, seeds 0-2; **naive** = trusts every stance foot, **gated** = per-foot gate; "in window" = $`t \in [4, 8)`$ s):

| Gait | Filter | in-window RMSE $`v`$ (m/s) | in-window twist NEES | final position RMSE (m) | slip readings used | good readings used |
| --- | --- | --- | --- | --- | --- | --- |
| tripod | naive | 0.0187-0.0189 | 33.9-34.6 | 0.075-0.082 | 100% | 100% |
| tripod | gated | 0.0037-0.0038 | 3.15-3.17 | 0.026-0.027 | 2.2-2.4% | 99.0% |
| ripple | naive | 0.0173-0.0175 | 33.7-34.5 | 0.070-0.076 | 100% | 100% |
| ripple | gated | 0.0034-0.0035 | 3.15-3.20 | 0.022-0.026 | 2.2% | 99.0% |
| wave | naive | 0.0170-0.0172 | 36.6-37.4 | 0.078-0.083 | 100% | 100% |
| wave | gated | 0.0032-0.0033 | 3.17-3.19 | 0.022-0.024 | 2.0-2.3% | 99.0% |

- **Naive: wrong and overconfident.** About 5× the no-slip velocity error and twist NEES near 34-37 instead of 3. Position error after 12 s triples (≈0.08 m vs ≈0.025 m). The gait changes little: wave is 9% better on velocity, and its NEES is slightly *worse*, since its extra feet make it more confident.
- **Gated: back to the no-slip numbers.** RMSE and NEES return to the §2 values for every gait. The gate drops about 98% of the slipping readings and 1.0% of the good ones, the false-alarm rate it was set for.
- **Why 2% get through.** A 0.1 m/s bias against a per-axis std of 0.02 m/s contributes about $`(0.1/0.02)^2 = 25`$ to the test value; noise adds about 2 on average, and sometimes pulls the total under 9.21.
- **What this needs.** Gating works because the *other* feet agree with the prediction. With a tripod, two good feet still carry 4 readings for 3 twist unknowns.
The same experiment, one run (seed 0, tripod gait), animated. The top-down view shows each stance foot at its simulated foothold, red while L2 slides; trusting every foot drags the sideways velocity off whenever L2 is down, and its estimated path drifts away (final position error 0.083 m vs 0.020 m with the gate). Swing feet are drawn faded at their simulated positions (§5 describes the swing model).

![Animation of one tripod-gait run: left, a top-down view of the hexapod walking its curved path with stance feet at their footholds, leg L2 turning red while it slides, and the estimated paths of the filter that trusts every foot and the gated filter; right, the true sideways body velocity and both estimates over time, with the slip window shaded](../../assets/hexapod_gait_ekf.gif)

*Animation: `use_numpy/animate_hexapod_gait_ekf.py` at its defaults, rendered by `uv run python use_numpy/animate_hexapod_gait_ekf.py --out assets/hexapod_gait_ekf.gif`.*

- **Small slips get through.** At `--slip-speed 0.03` (1.5× the reading std; seed 0, 100 trials) the gate lets 93.3-93.8% of the slipping readings through. Gated and naive then look alike: in-window $`v`$ RMSE 0.0056-0.0063 vs 0.0060-0.0066 m/s, twist NEES 5.6-5.8 vs 6.0-6.3, all about twice the no-slip values. A per-tick test cannot separate a slip of that size from noise.

---

## 4. Every foot drifts a little: model it as reading noise

So far a planted foot is perfectly still. Real feet creep: soil gives, rubber rolls. `--foot-drift-std` makes every stance foot slide at a small random velocity, white (independent from tick to tick) and independent per foot and per axis (std $`\sigma_{\text{drift}} = 0.02`$ m/s, as large as the reading noise).

**Intuition.** The filter only sees the foot's velocity relative to the body, so a random slide looks exactly like extra reading noise. A filter that models the drift just uses a larger reading variance, $`\sigma^2 + \sigma_{\text{drift}}^2`$. (With foot positions in the state, as in refs. 1-2, the same drift would be process noise on each foot.)

Results (tripod and wave, 100 trials, seeds 0-2; both filters see the same draws):

| Gait | Filter | NEES full | NEES twist | RMSE $`u`$ (m/s) | RMSE $`\omega`$ (rad/s) | final position RMSE (m) |
| --- | --- | --- | --- | --- | --- | --- |
| tripod | ignores drift ($`\sigma^2`$) | 8.46-8.49 | 4.51-4.67 | 0.0042 | 0.0125-0.0130 | 0.027-0.031 |
| tripod | models drift ($`\sigma^2 + \sigma_{\text{drift}}^2`$) | 5.89-6.07 | 2.96-3.05 | 0.0041 | 0.0121-0.0127 | 0.027-0.031 |
| wave | ignores drift | 7.86-8.43 | 4.51-4.63 | 0.0036-0.0037 | 0.0109-0.0112 | 0.022-0.027 |
| wave | models drift | 5.65-5.98 | 2.94-3.01 | 0.0035-0.0036 | 0.0106-0.0109 | 0.022-0.027 |

- **Ignoring drift costs calibration, not accuracy.** Twist NEES is about 1.5× too high; RMSE changes by at most 0.0005.
- **Why 1.5×.** The real one-tick std is $`\sqrt{2}`$ larger than the filter thinks, and by §2 the filter's variance scales with that std, so it is $`\sqrt{2} \approx 1.41`$ too small; its too-large gain adds a little real error on top.
- **Drift itself costs accuracy, as §2 predicts.** Doubling the reading variance should raise RMSE by about $`2^{1/4} = 1.19`$: tripod $`u`$ goes from 0.0034 (§2) to 0.0041-0.0042 (1.21-1.24×).

---

## 5. Contact timing is uncertain: a swinging foot read as planted

`--detect-time-noise-std` shifts every touchdown and liftoff the filter uses by its own Gaussian error, std $`\sigma_{\text{detect}}`$, rounded to ticks. The truth keeps the true schedule. When the filter thinks a foot is down but it is still swinging, it reads a moving foot as if planted.

**Intuition: the swing speed decides the damage.** In one gait cycle (period $`T = 1.2`$ s) the body travels $`u T = 0.12`$ m at the default $`u = 0.1`$ m/s, and a foot must catch up all of it during its swing. Its mean swing speed is therefore $`u / (1 - \text{duty})`$: 0.2 m/s for the tripod, 0.6 m/s for the wave gait. The script's swing (a straight line with a smooth $`1 - \cos`$ speed profile, zero speed at liftoff and touchdown) gives, measured:

| Gait | swing | peak foot speed | 1 tick from touchdown | 2 ticks |
| --- | --- | --- | --- | --- |
| tripod | 30 ticks (0.6 s) | 0.30 m/s | 0.031 m/s | 0.061 m/s |
| wave | 10 ticks (0.2 s) | 0.85 m/s | 0.24 m/s | 0.46 m/s |

One tick of timing error is 1.5× the reading noise for the tripod, but 12× for the wave gait.

Three filters, same draws: **trust** the detected flags; the **per-foot gate** of §3; or **trim**, skipping readings within $`2\sigma_{\text{detect}}`$ of every detected touchdown and liftoff. Twist NEES (consistent: 3, 95% interval $`[2.54, 3.50]`$) / final position RMSE in m, 100 trials, seeds 0-2:

| $`\sigma_{\text{detect}}`$ | tripod trust | tripod gate | tripod trim | wave trust | wave gate | wave trim |
| --- | --- | --- | --- | --- | --- | --- |
| 0.02 s (1 tick) | 3.20-3.35 / 0.025-0.026 | 3.13-3.25 / 0.023-0.025 | 2.98-3.07 / 0.024 | 10.5-11.8 / 0.057-0.062 | 3.06-3.12 / 0.020-0.022 | 3.05-3.09 / 0.021-0.023 |
| 0.05 s | 6.8-7.5 / 0.051-0.054 | 3.28-3.46 / 0.024-0.026 | 2.98-3.10 / 0.025-0.027 | 108-123 / 0.20-0.22 | 3.06-3.11 / 0.020-0.023 | 7.6-9.4 / 0.028-0.042 |
| 0.1 s | 38.6-43.8 / 0.146-0.155 | 3.47-3.65 / 0.027-0.028 | 3.19-4.14 / 0.034-0.037 | 476-545 / 0.45-0.49 | 3.06-3.11 / 0.020-0.023 | 186-222 / 0.21-0.26 |

- **Trusting the flags is fine for the tripod at 1 tick, never for the wave gait.** At 0.05 s the wave gait's twist NEES is 108-123 and its position error about 10× the gated value.
- **The gate handles both gaits at every level.** A wave foot read mid-swing is off by many reading stds, so the gate drops it: 0.1-0.4 such readings get through per run. Tripod swing readings near the ends are slow enough to pass (36-81 per run), and cost little; only at 0.1 s does the tripod reach the 3.50 bound (3.47-3.65).
- **Trimming works for the tripod, not the wave gait.** It only covers timing errors up to $`2\sigma_{\text{detect}}`$; the remaining ~5% of events still read a moving foot, harmless for the tripod's slow feet, costly for the wave's fast ones. At 0.1 s, a late liftoff and an early touchdown also add up to more than a 10-tick swing for 11.7% of wave swings: the swing vanishes from the detected flags, leaving no edge to trim around. Trimming also throws away good readings: at 0.1 s the trimmed tripod filter ends 0.034-0.037 m off, against 0.027-0.028 m with the gate.
- **No $`\Xi`$ or timing-noise term would fix this**: both are trivial here (§1). The fix belongs on the measurement side.

![Three panels from hexapod_gait_ekf.py: twist NEES bars for the tripod and wave gaits when every foot drifts, ignoring versus modeling the drift, against the consistent value 3; foot speed over one swing for the tripod (30 ticks, peak 0.30 m/s) and wave (10 ticks, peak 0.85 m/s) gaits, with the readings one and two ticks before touchdown marked; and twist NEES on log axes against contact-detection jitter from 0.02 to 0.1 s for trusting the detected times, the per-foot chi-squared gate and trimming, for both gaits](../../assets/hexapod_gait_ekf_extensions.png)

*Figure: `use_numpy/hexapod_gait_ekf.py` at its defaults (seed 0, 100 trials), plotted by `uv run python assets/make_figures.py hexapod_gait_ekf_extensions` (about 3 minutes). Panel (a) is §4's drift experiment.*

---

## 6. Takeaway

- **More feet down is a weak lever for the estimate.** With the filter averaging over time, the gait changes twist RMSE by $`n^{-1/4}`$ (12% from tripod to wave), not $`n^{-1/2}`$. On a real robot, the gait choice is driven by stability and speed.
- **Keep stance feet spread out.** Three feet on one side are about 40% worse than a tripod triangle. Standard gaits already avoid this; a damaged-leg or custom gait might not.
- **A slipping foot matters much more than the gait**, and a per-foot $`\chi^2`$ test against the prediction is a cheap, effective guard against large slips; small ones pass it. Real filters do the same: Bloesch et al. 2013 (ref. 3) add outlier rejection to the update step for a quadruped on slippery terrain.
- **Model foot drift as reading noise.** Ignoring a drift as large as the reading noise leaves accuracy intact but NEES about 1.5× too high.
- **Contact-timing errors hurt fast-swing gaits most.** The wave gait's feet swing 3× faster than the tripod's, so the same timing error reads a foot moving several times faster. Gate each foot's reading; trimming near detected touchdowns and liftoffs already fails for the wave gait at 0.05 s, a quarter of its swing.
- **Position and heading always drift** with leg odometry alone. A real system adds an IMU, which fixes roll and pitch (tilt about the forward and sideways axes) through gravity but not yaw, and an exteroceptive sensor (one that senses the surroundings, e.g. a camera or lidar) for the rest.

The next step up is a 3D filter that keeps each stance foot's position in the state (refs. 1-2) instead of using only its velocity: [`hexapod_contact_inekf.md`](hexapod_contact_inekf.md) builds ref. 2's invariant version and compares it with a world-frame EKF.

---

## 7. References

1. Bloesch, M., Hutter, M., Hoepflinger, M., Leutenegger, S., Gehring, C., Remy, C. D., & Siegwart, R. (2012). *State Estimation for Legged Robots - Consistent Fusion of Leg Kinematics and IMU*. Robotics: Science and Systems VIII. https://doi.org/10.15607/RSS.2012.VIII.003 - an EKF that fuses leg kinematics with an IMU and keeps the footholds' absolute positions in the state; the 3D, IMU-driven version of this toy's stance-foot measurement.
2. Hartley, R., Ghaffari, M., Eustice, R. M., & Grizzle, J. W. (2020). *Contact-Aided Invariant Extended Kalman Filtering for Robot State Estimation*. The International Journal of Robotics Research, 39(4), 402-430. https://doi.org/10.1177/0278364919894385 - an invariant EKF ([glossary](../glossary.md#iekf-invariant-ekf)) with the current contact points in the state, including how to add and remove contacts as feet land and lift.
3. Bloesch, M., Gehring, C., Fankhauser, P., Hutter, M., Hoepflinger, M. A., & Siegwart, R. (2013). *State Estimation for Legged Robots on Unstable and Slippery Terrain*. 2013 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 6058-6064. https://doi.org/10.1109/IROS.2013.6697236 - an unscented Kalman filter ([glossary](../glossary.md#ukf-unscented-kalman-filter)) with outlier rejection in the update step, tested on a quadruped trotting over slippery terrain; also shows that everything except global position and yaw is generally observable, even with one foot down. The real-robot counterpart of §3.
4. Khalili, H. H., Cheah, W., Garcia-Nathan, T. B., Carrasco, J., Watson, S., & Lennox, B. (2020). *Tuning and Sensitivity Analysis of a Hexapod State Estimator*. Robotics and Autonomous Systems, 129, 103509. https://doi.org/10.1016/j.robot.2020.103509 - an IMU + leg-kinematics EKF on a real hexapod (Corin) and how its accuracy depends on its noise parameters. This toy sidesteps that question by matching $Q$ and the reading noise to the truth.
