# Gait phase: which feet are on the ground, when touchdowns are uncertain

## Intuition

[`hexapod_gait_ekf.md`](hexapod_gait_ekf.md) tracks a hexapod from its legs: every foot on the ground reads the body's velocity. The filter is told which feet those are on every **tick** (one filter step, 0.02 s); even its §5 timing jitter is noise around a correct clock. On a real robot, that is the hard part:

```text
commanded:  liftoff ─── swing (nominal 0.6 s) ───▶ touchdown? ─── stance ───▶ next liftoff
truth:                  the foot lands at the commanded time + τ   (τ: lag, uneven ground, a blocked leg)
detector:               flags the touchdown a little later, and sometimes fires falsely before it
```

- **Stance and swing.** A foot is in **stance** while it is on the ground and in **swing** while it is in the air. **Liftoff** and **touchdown** are the switches between them.
- **Liftoff is known.** The gait controller commands it.
- **Touchdown is not.** Each foot lands at the commanded time plus an unknown **touchdown offset** $`\tau`$. It can be late (a slower swing, lower ground, an obstacle in the way) or early (higher ground).
- **A wrong "planted" flag is expensive.** If the filter reads a foot that is still swinging, it takes that foot's motion for the body's: a confident, wrong velocity.

So the question is **which feet to read on each tick**: the **stance flags**. This doc compares four ways to set them from what the robot knows: the **commanded schedule** (the timetable of liftoffs and nominal touchdowns) and a **contact detector**. The body filter is `hexapod_gait_ekf.py`'s, unchanged; only the stance flags it receives differ.

**Short answer:** the schedule reads swinging feet whenever the feet land late, and a raw detector reads them whenever it fires falsely. Both make the body filter badly **overconfident** (it claims less uncertainty than its errors show): NEES ([glossary](../glossary.md#consistency-and-nees); 6 is consistent here) of 41 and 50. A **rhythm model** of when touchdowns usually come, either a learned offset in the style of a **CPG** ([central pattern generator](../glossary.md#cpg-central-pattern-generator): an internal rhythm generator for the legs) or a small Kalman filter, lets the filter reject triggers that come too early to be real. That brings NEES down to 6.3-6.6, next to 6.2 with the true stance. What decides success is not the type of estimator but **how wide its acceptance window is** (§4).

![Four panels from gait_phase_ekf.py: stance flags of the blocked leg for the true stance and the four approaches with swinging-foot reads highlighted; true touchdown offsets with the CPG and phase-EKF lag estimates; body NEES over time per approach from 100 Monte Carlo runs; and body NEES against terrain spread for the CPG window and the phase EKF at 2 and 3 sigma](../../assets/gait_phase_ekf.png)

*Figure: `use_numpy/gait_phase_ekf.py` at its defaults (random seed 0, 100 trials), plus a terrain sweep (30 trials per point), plotted by `uv run python generator/make_figures.py gait_phase_ekf` (about 2 minutes).*

### Scope
Planar hexapod in a tripod gait, as in `hexapod_gait_ekf.md`; only touchdown times are uncertain. The detector's error rates and the touchdown offsets are this repo's assumptions, not measured values. No paper we found combines a CPG with a Kalman filter on a legged robot, so the two rhythm-model approaches here are this repo's design (§8).

---

## 1. Model

### 1.1 Ground truth: when each foot really lands

| Part | What it does | Default |
| --- | --- | --- |
| **Gait** | Tripod, commanded cycle 1.2 s: two groups of three legs alternate, 0.6 s swing and 0.6 s stance each | as in `hexapod_gait_ekf.md` |
| **Systematic lag** | Every foot lands later and later: the lag grows linearly over the run, e.g. a swing that slows under load | 0 → 20% of the swing (120 ms) |
| **Terrain** | Each foot and step gets its own random offset: higher ground is touched early, lower ground late | std 30 ms |
| **Blocked leg** | One leg lands much later on every step in a time window (an obstacle in its path) | L2, 10-14 s, +200 ms |

A foot always swings at least 5 ticks and stands at least 1 tick before its commanded liftoff. The body motion and foot readings come from `hexapod_gait_ekf.py`; its foot model follows these true stance times.

### 1.2 The contact detector

Modeled on the CTU lab's servo **position-error detector** (Faigl & Čížek 2019): contact is declared when the commanded and measured positions of the **femur joint** (the joint that raises and lowers the leg) differ by more than a threshold. The paper reports no detection delays or error rates, so the numbers below are assumptions.

| Behavior | Model | Default |
| --- | --- | --- |
| **Delay** | Each true touchdown is flagged after a random delay (reading period plus the time for the position error to build up). The estimators know the mean delay, not each draw | mean 40 ms, std 20 ms |
| **False trigger** | Early in the **swing-down** (the last third of the swing, while the leg lowers), the position error has a start-up peak that a fixed threshold can mistake for contact (the paper's motivation for an adaptive threshold). Modeled as one false trigger 0.12-0.2 s before the commanded touchdown | 10% of steps |
| **Lost detection** | A detection that would come after the next liftoff is dropped | — |

### 1.3 Four ways to set the stance flags

Stance always ends at the commanded liftoff. The approaches differ only in when they let it start.

| Approach | Stance starts at | Knows |
| --- | --- | --- |
| `schedule` | The commanded touchdown (assumes $`\tau = 0`$) | The schedule |
| `detector` | The first detector trigger of the step | The detector |
| `cpg` | The first trigger no earlier than $`\mu - 0.1`$ s, where $`\mu`$ is a learned **typical detection offset** per tripod group | Both, plus the mean delay |
| `phase_ekf` | The first trigger no more than 3σ (three standard deviations of the predicted detection time) earlier than a Kalman filter's predicted detection | Both, plus the mean delay and the terrain and delay spreads |

`oracle` uses the true stance and is shown as a reference, not as an approach.

**`cpg`** follows the idea of Szadkowski et al. (2021), much simplified. Their CPG, kept in sync with the touchdowns, learns *at which phase* of its rhythm (point in its cycle) contact usually comes. Here that is one number $`\mu`$ per tripod group. After each accepted trigger it moves $`\mu`$ a fraction (0.3) of the way toward the observed offset. An error above 0.1 s is flagged as a **mistiming** and not learned from. The paper's CPG is a **Matsuoka oscillator** (a pair of model neurons that inhibit each other and so produce a rhythm) with learned neuron models on top; here it is one number per group.

**`phase_ekf`** is a two-state Kalman filter per tripod group, in the style of the gait-phase EKFs used for prostheses and exoskeletons (Medrano et al. 2023), but driven by touchdown events:

- **State** $`[\lambda, \dot\lambda]`$: the current systematic lag $`\lambda`$ (s) and how fast it changes $`\dot\lambda`$ (s per s).
- **Predict**, once per step of that group (interval $`\Delta`$, here 1.2 s): constant rate, with **white noise** (random, independent from step to step) on the rate.
- **Measurement**, at each accepted detection: $`y`$ = detection time − commanded touchdown − mean delay, with noise variance $`R`$ = terrain spread² + delay spread² + $`\mathrm{tick}^2/12`$ (the error from rounding times to whole ticks).
- **Gate:** the **innovation** $`y - \hat\lambda`$ is the measured lag minus the predicted one. A trigger whose normalized innovation $`(y - \hat\lambda)/\sqrt{S}`$ is below −3 is rejected as too early; one above +3 is a mistiming and is not used to update. The 3 is the gate width, set with `--ekf-gate`.

```math
F = \begin{bmatrix} 1 & \Delta \\ 0 & 1 \end{bmatrix}, \qquad
Q = q \begin{bmatrix} \Delta^3/3 & \Delta^2/2 \\ \Delta^2/2 & \Delta \end{bmatrix}, \qquad
S = P_{\lambda\lambda} + R
```

Here $`F`$ is the state transition, $`Q`$ its **process noise** (the uncertainty each predict step adds; [glossary](../glossary.md#predict-and-update-steps)) with rate-change intensity $`q`$, $`P_{\lambda\lambda}`$ the filter's lag variance and $`S`$ the predicted spread of the next measurement. Defaults: $`q = 10^{-5}`$ s²/s³ (`--ekf-rate-q`); the filter starts at $`[0, 0]`$ with std 50 ms on the lag (`--ekf-init-lag-std`) and 0.01 s/s on its rate (`--ekf-init-rate-std`).

---

## 2. Sanity check: nothing perturbed

With no lag, no terrain spread, no blocked leg and a perfect detector (40 [Monte Carlo trials](../glossary.md#monte-carlo-trials)), all four approaches produce exactly the true stance flags, and the body NEES is 6.92 for all of them (the band a consistent 40-run average falls in 95% of the time: [4.97, 7.12]). So every difference below comes from touchdown timing.

---

## 3. Results at the defaults

100 trials; consistent band [5.34, 6.70]. "Swinging foot read" counts ticks where a foot in the air was flagged as planted; "stance missed" counts ticks where a planted foot was not read.

| Approach | Body NEES | Swinging foot read (ticks per run) | Stance missed (ticks per run) | Forward-speed [RMSE](../glossary.md#rmse-root-mean-square-error) (m/s) |
| --- | --- | --- | --- | --- |
| `schedule` | 40.6 | 398 | 10 | 0.0111 |
| `detector` | 49.8 | 129 | 215 | 0.0114 |
| `cpg` | **6.28** | 0 | 250 | 0.0039 |
| `phase_ekf` | **6.60** | 2.2 | 241 | 0.0040 |
| `oracle` | 6.23 | 0 | 0 | 0.0038 |

#### The schedule reads feet that are still in the air
Whenever a foot lands late, the schedule flags it as planted before it touches down. As the lag grows, so does the damage: the schedule's NEES climbs over the run (figure, bottom left). The blocked leg adds four large late landings (figure, top left, gold blocks).

#### A raw detector reads them too, through its false triggers
A detector never flags a real touchdown early, so with no false triggers it would be safe (§4). But each false trigger starts the stance 0.12-0.2 s before the commanded touchdown, while the foot is still lowering. With false triggers on 10% of steps, that is enough for NEES 50.

#### A rhythm model rejects triggers that come too early
`cpg` and `phase_ekf` know roughly when a detection should come. A trigger far earlier than that is rejected, so they almost never read a swinging foot. Both track the growing lag to about 16 ms RMS (figure, top right), against a per-step terrain scatter of 30 ms.

#### The price: readings lost to the detection delay
Every detection-based approach starts stance only after the detector fires, about 2 ticks after the true touchdown on average. That costs about 240 readings per run (2 ticks × 120 steps), but the other two or three feet in stance still pin down the body, so the forward-speed RMSE barely changes (0.0039-0.0040 vs 0.0038 with the true stance).

The same default run, animated:

![Animation of gait_phase_ekf.py: top view of the hexapod with feet filled in stance and hollow in swing and the blocked leg highlighted; a scrolling gait diagram of the blocked leg for the true stance and the four approaches with swinging-foot reads highlighted and detector triggers marked; and the touchdown offsets with the CPG and phase-EKF lag estimates](../../assets/gait_phase_ekf.gif)

*Animation: `generator/animate_gait_phase_ekf.py` at its defaults (seed 0, one run, 7-19.5 s around the blocked window; L2's false trigger comes at 18.4 s), rendered by `uv run python generator/animate_gait_phase_ekf.py --out assets/gait_phase_ekf.gif`.*

---

## 4. What decides the result: the acceptance window

One factor at a time, 40 trials each, seed 0, everything else at the defaults. Each cell is the body NEES (consistent band for 40 runs: [4.97, 7.12]).

**Growing lag** (final lag as a share of the 0.6 s swing):

| Final lag | 0 | 10% (60 ms) | 20% (120 ms, default) | 30% (180 ms) | 50% (300 ms) |
| --- | --- | --- | --- | --- | --- |
| `schedule` | 12.9 | 19.0 | 40.7 | 93.5 | 319.1 |
| `detector` | 27.8 | 35.7 | 45.7 | 56.8 | 84.0 |
| `cpg` | 7.08 | 7.02 | 7.04 | 6.97 | 7.03 |
| `phase_ekf` | 7.25 | 7.19 | 7.69 | 7.70 | 8.10 |

Even with no lag, the schedule reads late landings from the terrain spread and the blocked leg (NEES 12.9).

**False triggers:**

| False-trigger rate | 0 | 10% (default) | 30% |
| --- | --- | --- | --- |
| `detector` | 7.04 | 45.7 | 216.2 |
| `cpg` | 7.04 | 7.04 | 7.04 |
| `phase_ekf` | 7.04 | 7.69 | 12.3 |

Without false triggers the raw detector is as good as the rhythm models: the delay alone never makes it read a swinging foot.

**Rougher ground and the width of the window.** Both rhythm models accept a trigger only if it is not too much earlier than expected. Too narrow, and they reject true early landings (lost readings, safe). Too wide, and false triggers get in (swinging feet read, overconfident).

| Acceptance rule | Terrain 30 ms | 60 ms | 100 ms |
| --- | --- | --- | --- |
| `cpg`, window 0.05 s | 7.08 (524 missed) | 6.80 (1209 missed) | 6.57 (1553 missed) |
| `cpg`, window 0.1 s (default) | 7.04 | 7.04 (378 missed) | 6.81 (673 missed) |
| `cpg`, window 0.2 s | 9.26 | 11.6 | 17.3 |
| `phase_ekf`, gate 2σ | 7.05 | 7.27 | 21.6 |
| `phase_ekf`, gate 3σ (default) | 7.69 | 13.0 | 45.6 |
| `oracle` | 6.92 | 7.03 | 7.01 |

*"Missed": true-stance ticks not read, per run; shown in parentheses where well above the ~240 lost to the detection delay alone.*

Why the 3σ gate fails on rough ground: its width comes from the predicted spread of *true* detections, $`\sqrt{S}`$, and that grows with the terrain spread (σ ≈ 36 ms at 30 ms terrain, 63 ms at 60 ms, 100 ms at 100 ms). The false triggers sit 0.16-0.36 s before the predicted detection. Once 3σ reaches that distance, they pass. At 30 ms terrain 3σ (108 ms) stays short of it; the few swinging-foot reads at the defaults (2.2 per run, §3) come mostly in the first 5 s, while the initial lag uncertainty still widens the gate (132 of 193 such ticks over 100 runs). A Kalman gate is built for the spread of real measurements; it has no model of false triggers. A tighter 2σ gate holds up to 60 ms of terrain spread, then fails too.

**A fairness note:** the CPG's 0.1 s window works at every terrain level here because it reaches 0.1 s before the expected detection, short of the false triggers (at least 0.16 s before it), and both numbers are this toy's choices. Its success means "a window that matches the false-trigger zone works", not "a CPG beats a Kalman filter".

---

## 5. Mistiming: flagging the blocked leg

Both rhythm models flag a detection that comes much later than expected as a **mistiming**: a possible obstacle, which a controller could answer with a reflex. At the defaults (100 trials), `cpg` flags 400 of the 400 blocked steps with 99 false alarms among 11600 other steps (0.9%); `phase_ekf` flags 387 of 400 with 78 false alarms.

On rougher ground (40 trials: 160 blocked steps, 4640 others) the same window-width trade-off appears:

| Terrain | `cpg` (0.1 s): hits / false alarms | `phase_ekf` 3σ: hits / false alarms | `phase_ekf` 2σ: hits / false alarms |
| --- | --- | --- | --- |
| 30 ms | 160 / 37 | 156 / 61 | 160 / 100 |
| 60 ms | 156 / 466 | 50 / 5 | 130 / 100 |
| 100 ms | 145 / 1162 | 15 / 6 | 64 / 143 |

The CPG's fixed threshold keeps catching the blocked leg but raises a false alarm on a quarter of all steps at 100 ms terrain. The EKF's threshold widens with the terrain spread: few false alarms, but a 200 ms block no longer stands out against a 3σ of about 300 ms.

---

## 6. What this toy does not show

- **A filter that models false triggers.** Treating each trigger as possibly false, with a probability, and weighing them (as in probabilistic data association) would replace the hand-set window. Not implemented here.
- **Phase inside the body filter.** The touchdown estimator and the body EKF are separate (a cascade); the body EKF never learns how sure the stance flags are. A joint filter (one state holding both the body and the lags) is the alternative.
- **Real detector behavior.** The delay and false-trigger rate are assumptions; the cited paper reports neither. Missed detections on soft ground are not modeled.
- **Unknown liftoff.** Liftoff is assumed exactly known because it is commanded; a controller that waits for touchdowns, or slips at liftoff, would loosen that.
- **A gait that waits.** Liftoffs stay on schedule even when the other tripod group has not landed yet, so a late landing leaves the robot with fewer than three feet down (in seed 0's run: 193 of 1201 ticks, 75 of them with no foot down). A real **statically stable** gait (one that can stand still at any moment without falling) would wait for the touchdowns, as the controller of Faigl & Čížek (2019) does. The body filter has fewer readings on those ticks (none on the 75), so the comparison of stance flags still holds, but the walking itself is not physical there.
- **The CPG itself.** The `cpg` approach keeps only the learned contact phase and the mistiming threshold of Szadkowski et al.; no oscillator dynamics, and no **entrainment** (the rhythm and the gait pulling each other into sync).

---

## 7. Takeaway

When touchdowns are uncertain, the commanded schedule and a raw detector both end up reading feet that are still in the air, and the body filter becomes confidently wrong. A model of when touchdowns usually come lets the filter reject triggers that are too early to be real, and also flags legs that land far too late. The deciding choice is the width of the acceptance window: it must be wider than the spread of true touchdowns and narrower than the distance to the false triggers. A Kalman σ-gate sets it from the first number only.

---

## 8. References

1. Faigl, J. & Čížek, P. (2019). *Adaptive locomotion control of hexapod walking robot for traversing rough terrains with position feedback only*. Robotics and Autonomous Systems, 116, 136-147. Preprint: https://comrob.fel.cvut.cz/papers/ras19adaptive.pdf - the position-error contact detector (femur only, during the swing-down) and its start-of-swing-down error peak; reports no detection delays or error rates.
2. Szadkowski, R., Prágr, M., & Faigl, J. (2021). *Self-Learning Event Mistiming Detector Based on Central Pattern Generator*. Frontiers in Neurorobotics, 15. https://doi.org/10.3389/fnbot.2021.629652 - a CPG entrained to touchdowns acts as an estimator of when contact happens, and flags mistimings; no Kalman filter.
3. Medrano, R. L., Thomas, G. C., Keais, C. G., Rouse, E. J., & Gregg, R. D. (2023). *Real-Time Gait Phase and Task Estimation for Controlling a Powered Ankle Exoskeleton on Extremely Uneven Terrain*. IEEE Transactions on Robotics, 39(3), 2170-2182. https://doi.org/10.1109/TRO.2023.3235584 - an EKF with gait phase and phase rate in its state, on a wearable robot.
4. Thatte, N., Shah, T., & Geyer, H. (2019). *Robust and Adaptive Lower Limb Prosthesis Stance Control via Extended Kalman Filter-Based Gait Phase Estimation*. IEEE Robotics and Automation Letters, 4(4), 3129-3136. https://doi.org/10.1109/LRA.2019.2924841 - EKF gait-phase estimation for a prosthesis, more robust than threshold-based event detection.
