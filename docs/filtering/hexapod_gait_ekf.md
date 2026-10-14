# Hexapod leg odometry: what the gait changes

## Intuition

A planted foot doesn't move. So if the joint encoders say a stance foot is sliding backward under the body at 0.1 m/s, the body is moving forward at 0.1 m/s. That is **leg odometry**: every foot on the ground is a velocity sensor for the body ([glossary](../glossary.md#odometry)).

A hexapod has six legs, and its **gait** decides which ones are down on each tick:

```text
            L1 ●  ● R1          tripod   duty 1/2   3 feet down   L1,R2,L3 alternate with R1,L2,R3
 (front)  L2 ●  body  ● R2      ripple   duty 2/3   4 feet down   one leg per side swings, back to front
            L3 ●  ● R3          wave     duty 5/6   5 feet down   one leg swings at a time
```

The **duty factor** is the fraction of a cycle each foot spends on the ground. More feet down means more velocity readings per tick, but a slower walk on a real robot.

![Three panels from hexapod_gait_ekf.py: gait diagrams of stance and swing per leg for the tripod, ripple and wave gaits over two cycles; the hexapod body turning left with an arrow on each foot showing its measured velocity relative to the body, six different readings; and two three-foot stance layouts, a tripod triangle and three feet on one side, with their one-tick standard deviations of forward speed and turn rate](../../assets/hexapod_gait_ekf_concept.png)

*Figure: the gait schedules and stance geometry of `use_numpy/hexapod_gait_ekf.py`, plotted by `uv run python assets/make_figures.py hexapod_gait_ekf_concept`.*

Two questions, answered with [`hexapod_gait_ekf.py`](../../use_numpy/hexapod_gait_ekf.py):

1. **No slip:** how much does the gait change the estimate? (§2)
2. **One leg slips:** a slipping foot breaks "a planted foot doesn't move". What does that do, and can the filter catch it? (§3)

**Scope.** Planar body, perfect kinematics (the filter knows where each stance foot is; only its velocity reading is noisy), a known stance schedule, no IMU. Contact detection is a separate problem ([`hybrid_saltation_ekf.md` §8](hybrid_saltation_ekf.md#8-quantifying-contact-detection-timing-jitter)).

---

## 1. Model

**Why planar, not 1D.** In 1D every stance foot measures the same scalar velocity, so six legs only average noise: variance $`\sigma^2/n`$. In the plane, a foot at body-frame offset $`r_i`$ sees the body's velocity *plus* the turn: when the body turns in place at $`\omega = 0.2`$ rad/s, a foot 0.15 m to the left moves at 0.03 m/s and its mirror foot at −0.03 m/s. Feet in different places therefore separate forward speed from turn rate.

**State.** $`x = [p_x, p_y, \theta, u, v, \omega]`$: world position, heading, body-frame velocity $`b = [u, v]`$ (forward, left) and turn rate. Velocity and turn rate together are the body's **twist** ([glossary](../glossary.md#twist)).

**Predict** (`predict`): constant twist, pose integrated with it. Nonlinear in $`\theta`$, hence an EKF ([glossary](../glossary.md#ekf-extended-kalman-filter)):

```math
p_{k+1} = p_k + R(\theta_k)\, b_k\, \Delta t, \qquad \theta_{k+1} = \theta_k + \omega_k \Delta t, \qquad [b, \omega]_{k+1} = [b, \omega]_k + w_k
```

$`w_k`$ is the random-walk noise on the twist, std $`\sigma_a \Delta t`$ for $`u, v`$ ($`\sigma_a = 0.05\ \text{m/s}^2`$) and $`\sigma_\alpha \Delta t`$ for $`\omega`$ ($`\sigma_\alpha = 0.1\ \text{rad/s}^2`$). The truth twist is drawn from exactly this random walk, so $Q$ is matched and NEES is a fair test.

**Measure** (`measurement_update_feet`): each stance foot $i$ reports its velocity relative to the body, in the body frame:

```math
z_i = -\left(b + \omega\, J r_i\right) + n_i, \qquad J = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}, \qquad n_i \sim \mathcal{N}(0,\ 0.02^2 I)
```

$`\omega J r_i`$ is the planar $`\omega \times r_i`$. The model is linear in the state, and it only involves the twist columns:

```math
H_i = \begin{bmatrix} 0 & 0 & 0 & -1 & 0 & r_{i,y} \\ 0 & 0 & 0 & 0 & -1 & -r_{i,x} \end{bmatrix}
```

All stance feet enter one joint update per tick.

**What is never measured.** No reading involves $`p_x, p_y, \theta`$, so they are unobservable ([glossary](../glossary.md#observability)): leg odometry integrates the twist, and position and heading drift. This matches the real-robot result that, with leg kinematics and an IMU, everything except global position and yaw is observable (Bloesch et al. 2013, ref. 3).

**Controlled comparison** (`simulate_trial`): each Monte Carlo trial draws one true trajectory, one initial-guess error and one noise stream per foot, and every gait uses the same draws. Only which feet are down, and where, changes between gaits.

---

## 2. No slip: the gait matters, but less than you'd think

**Intuition first.** One tick's readings pin down the twist to about $`\sigma/\sqrt{n}`$. The filter, though, averages over many ticks against a twist that slowly wanders, and that averaging shrinks the gait's effect to about $`n^{-1/4}`$.

**One tick, no filter** (`twist_geometry_std`: std from one tick's readings alone, mean over ticks; deterministic):

| Gait | Feet down | $`u`$ std (m/s) | $`\omega`$ std (rad/s) | $`\sigma/\sqrt{n}`$ (m/s) |
| --- | --- | --- | --- | --- |
| tripod | 3 | 0.0117 | 0.0497 | 0.0115 |
| ripple | 4 | 0.0100 | 0.0431-0.0432 | 0.0100 |
| wave | 5 | 0.0090-0.0091 | 0.0386-0.0388 | 0.0089 |

- With these gaits the **count** explains almost everything: $`u`$ std sits at $`\sigma/\sqrt{n}`$, and wave/tripod $`\omega`$ std is $`0.0388/0.0497 = 0.78 \approx \sqrt{3/5}`$.
- **Where the feet are** matters when the stance is lopsided. Same three feet, two layouts:

| 3 feet at nominal offsets | $`u`$ std (m/s) | $`v`$ std (m/s) | $`\omega`$ std (rad/s) |
| --- | --- | --- | --- |
| tripod triangle L1, R2, L3 | 0.0117 | 0.0115 | 0.0497 |
| one side L1, L2, L3 | 0.0164 | 0.0115 | 0.0700 |

  Three feet in a line along one side are about 40% worse in both $`u`$ and $`\omega`$: forward speed and turn rate both show up as along-body motion of those feet, so they are harder to tell apart. The tripod's alternating triangle avoids this. All three standard gaits keep their stance feet spread on both sides; within a cycle, the wave gait's one-tick $`\omega`$ std only ranges over 0.0375-0.0413 rad/s (`twist_geometry_std` per tick, seed 0).

**The filter** (`run_monte_carlo`, 100 trials, seeds 0-2, $`\Delta t = 0.02`$ s, 12 s):

| Gait | RMSE $`u`$ (m/s) | RMSE $`\omega`$ (rad/s) | final position RMSE (m) | final heading RMSE (rad) | NEES full / twist |
| --- | --- | --- | --- | --- | --- |
| tripod | 0.0034-0.0035 | 0.0100-0.0103 | 0.024-0.025 | 0.026-0.029 | 5.79-6.23 / 2.98-3.02 |
| ripple | 0.0031-0.0032 | 0.0093-0.0095 | 0.020-0.025 | 0.021-0.024 | 5.56-6.25 / 2.96-3.01 |
| wave | 0.0030 | 0.0088-0.0090 | 0.020-0.023 | 0.020-0.021 | 5.61-6.28 / 2.99-3.00 |

- **Consistent.** A consistent filter averages NEES 6 (full state) and 3 (twist) ([glossary](../glossary.md#consistency-and-nees)); the 95% intervals for a 100-trial average are $`[5.34, 6.70]`$ and $`[2.54, 3.50]`$. Every gait is inside both.
- **Wave beats tripod by 12%, not 22%.** One tick: 22% lower std. Filter: 12% lower RMSE.
- **Why $`n^{-1/4}`$.** For a random walk with per-tick variance $q$, measured each tick with variance $r$ and $`q \ll r`$, the steady-state variance is about $`\sqrt{q\, r}`$, so the std scales as $`r^{1/4}`$ and $r \propto 1/n$. Worked for tripod $`u`$: $`q = (0.05 \times 0.02)^2 = 10^{-6}`$, $`r = 0.0117^2`$, so $`\sqrt[4]{q\,r} = 0.0034`$ m/s, the RMSE in the table. The same formula gives 0.0100 rad/s for tripod $`\omega`$ and 0.0030 / 0.0088 for wave, all within 0.0003 of the Monte Carlo values. $`(3/5)^{1/4} = 0.88`$ is the 12%.
- **Drift is the same story.** Position and heading are integrals of the twist error, so wave drifts slightly less (heading 0.020-0.021 vs 0.026-0.029 rad after 12 s), but no gait stops the drift.

![Three panels from hexapod_gait_ekf.py: bar charts of forward-speed and turn-rate RMSE for the tripod, ripple and wave gaits, each with its (q r)^(1/4) steady-state estimate drawn as a line; and the sideways-velocity RMSE over time with leg L2 slipping between 4 and 8 s, for the tripod and wave gaits, trusting every foot versus the per-foot chi-squared gate](../../assets/hexapod_gait_ekf.png)

*Figure: `use_numpy/hexapod_gait_ekf.py` at its defaults (seed 0, 100 trials), plotted by `uv run python assets/make_figures.py hexapod_gait_ekf`. The right panel is §3's slip experiment.*

---

## 3. One slipping leg: the gait barely matters, gating does

L2 slips outward at 0.1 m/s for $`t \in [4, 8)`$ s whenever it is down. Its reading is then biased by 0.1 m/s, and the filter's model says it is zero.

**Intuition.** In the joint update, one bad foot out of $n$ pulls the estimate by about $`1/n`$ of its bias: $`0.1/3 = 0.033`$ m/s with a tripod, $`0.1/5 = 0.020`$ m/s with a wave gait. The peaks of the sideways velocity RMSE match (seed 2, 100 trials): 0.0316 (tripod), 0.0254 (ripple), 0.0206 m/s (wave). But the slipping foot is down for 1/2 of the cycle in a tripod and 5/6 in a wave gait, so the wave gait's smaller error lasts longer. Over the window the two largely cancel.

**Per-foot gating** (`gating=True`): before the joint update, test each stance foot's **innovation** $`\nu_i = z_i - H_i \hat{x}^-`$ against the predicted state ([glossary](../glossary.md#innovation-and-kalman-gain)):

```math
\nu_i^\top \left(H_i P^- H_i^\top + R\right)^{-1} \nu_i \le 9.21
```

9.21 is the 99% point of a $`\chi^2`$ with 2 DoF ([glossary](../glossary.md#chi-square-distribution)), so a good foot is wrongly dropped 1% of the time. Feet that fail are left out ([glossary](../glossary.md#outlier-and-robust-kernel)). Testing each foot against the prediction, not after other feet's updates, makes the result independent of foot order.

Results (100 trials, seeds 0-2; "in window" = $`t \in [4, 8)`$ s):

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
- **Why 2% get through.** A 0.1 m/s bias against a per-axis std of 0.02 m/s gives an expected test value near 25; the reading's own noise sometimes pulls it under 9.21.
- **What this needs.** Gating works because the *other* feet agree with the prediction. With a tripod, two good feet still carry 4 readings for 3 twist unknowns.
The same experiment, one run (seed 0, tripod gait), animated. The top-down view shows each stance foot at its simulated foothold, red while L2 slides; trusting every foot drags the sideways velocity off whenever L2 is down, and its estimated path drifts away (final position error 0.083 m vs 0.020 m with the gate). Swing legs are drawn at their nominal offset as an illustration, since the script does not simulate swing.

![Animation of one tripod-gait run: left, a top-down view of the hexapod walking its curved path with stance feet at their footholds, leg L2 turning red while it slides, and the estimated paths of the filter that trusts every foot and the gated filter; right, the true sideways body velocity and both estimates over time, with the slip window shaded](../../assets/hexapod_gait_ekf.gif)

*Animation: `use_numpy/animate_hexapod_gait_ekf.py` at its defaults, rendered by `uv run python use_numpy/animate_hexapod_gait_ekf.py --out assets/hexapod_gait_ekf.gif`.*

- **Small slips get through.** At `--slip-speed 0.03` (1.5× the reading std; seed 0, 100 trials) the gate lets 93.3-93.8% of the slipping readings through. Gated and naive then look alike: in-window $`v`$ RMSE 0.0056-0.0063 vs 0.0060-0.0066 m/s, twist NEES 5.6-5.8 vs 6.0-6.3, all about twice the no-slip values. A per-tick test cannot separate a slip of that size from noise.

---

## 4. Takeaway

- **More feet down is a weak lever for the estimate.** With the filter averaging over time, the gait changes twist RMSE by $`n^{-1/4}`$ (12% from tripod to wave), not $`n^{-1/2}`$. On a real robot, the gait choice is driven by stability and speed; estimation follows.
- **Keep stance feet spread out.** Three feet on one side are about 40% worse than a tripod triangle. Standard gaits already avoid this; a damaged-leg or custom gait might not.
- **A slipping foot matters much more than the gait**, and a per-foot $`\chi^2`$ test against the prediction is a cheap, effective guard against large slips; small ones pass it. Real filters do the same: Bloesch et al. 2013 (ref. 3) add outlier rejection to the update step for a quadruped on slippery terrain.
- **Position and heading always drift** with leg odometry alone. A real system adds an IMU, which fixes roll and pitch through gravity but not yaw, and an exteroceptive sensor for the rest.

The next step up is a 3D filter that keeps each stance foot's position in the state (refs. 1-2) instead of using only its velocity.

---

## 5. References

1. Bloesch, M., Hutter, M., Hoepflinger, M., Leutenegger, S., Gehring, C., Remy, C. D., & Siegwart, R. (2012). *State Estimation for Legged Robots - Consistent Fusion of Leg Kinematics and IMU*. Robotics: Science and Systems VIII. https://doi.org/10.15607/RSS.2012.VIII.003 - an EKF that fuses leg kinematics with an IMU and keeps the footholds' absolute positions in the state; the 3D, IMU-driven version of this toy's stance-foot measurement.
2. Hartley, R., Ghaffari, M., Eustice, R. M., & Grizzle, J. W. (2020). *Contact-Aided Invariant Extended Kalman Filtering for Robot State Estimation*. The International Journal of Robotics Research, 39(4), 402-430. https://doi.org/10.1177/0278364919894385 - an invariant EKF ([glossary](../glossary.md#iekf-invariant-ekf)) with the current contact points in the state, including how to add and remove contacts as feet land and lift.
3. Bloesch, M., Gehring, C., Fankhauser, P., Hutter, M., Hoepflinger, M. A., & Siegwart, R. (2013). *State Estimation for Legged Robots on Unstable and Slippery Terrain*. 2013 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 6058-6064. https://doi.org/10.1109/IROS.2013.6697236 - an unscented Kalman filter with outlier rejection in the update step, tested on a quadruped trotting over slippery terrain; also shows that everything except global position and yaw is generally observable, even with one foot down. The real-robot counterpart of §3.
4. Khalili, H. H., Cheah, W., Garcia-Nathan, T. B., Carrasco, J., Watson, S., & Lennox, B. (2020). *Tuning and Sensitivity Analysis of a Hexapod State Estimator*. Robotics and Autonomous Systems, 129, 103509. https://doi.org/10.1016/j.robot.2020.103509 - an IMU + leg-kinematics EKF on a real hexapod (Corin) and how its accuracy depends on its noise parameters. This toy sidesteps that question by matching $Q$ and $R$ to the truth.
