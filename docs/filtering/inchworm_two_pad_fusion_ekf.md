# Two-pad inchworm: reading both pad sensors

## Intuition

[`inchworm_two_pad_ekf.md`](inchworm_two_pad_ekf.md) (**the two-pad doc** below) tracks a planar inchworm with a back pad and a front pad, a gyro, a **kinematic closure** (the front pad's position relative to the back pad, from the **motor encoders**, sensors that count how far each motor has turned) and no position fix. Its filter reads only the **anchored pad**'s optical displacement sensor (the pad that grips while the other moves), as the robot that motivates it does (Kubík & Faigl, reference 1). The **moving pad** has no motion model; the closure alone places it. Its best policy, `pad_fused`, moves the anchored pad by its sensor's reading. This doc keeps that and adds what the two-pad doc left out:

- **The moving pad's sensor.** In a sliding gait the moving pad stays on the ground, so its sensor can read that end's motion too.
- **An encoder model.** The moving pad can also be predicted from the commanded stroke (how far the body should extend or contract in one **tick**, a filter step of 0.05 s) and the steering.
- **Imperfect sensors.** A sensor can **spike** (report a backward motion that did not happen, like a slip) or **drop out** (report no motion at all, as when it loses track of the surface).

One fact makes the second sensor useful for more than extra measurements. When the anchored pad slips, the whole body slides with it: both pads move by the same amount, and the encoders see no change in the distance between them. So:

| Anchored-pad reading | Moving-pad reading minus the commanded stroke | Meaning |
| --- | --- | --- |
| about 0 | about 0 | Nothing happened |
| a slip | the same slip | **Slip:** both sensors agree the body slid |
| moved | about 0 | The anchored-pad sensor is wrong |
| about 0 | moved | The moving-pad sensor is wrong (or the anchored one dropped out during a slip) |

A real slip makes the two sensors **agree**; a faulty sensor makes them **disagree**. One sensor alone cannot tell a spike from a slip.

**Short answer:** without sensor faults, the moving pad's sensor gains little: the final position error drops from 2.3% to 1.8% of the distance traveled (`--p-fault 0`). With faults, reading it without a check makes things worse: NEES ([glossary](../glossary.md#consistency-and-nees); 5 is consistent) 47.7 against 17.4 for the anchored sensor alone, because every fault of either sensor now enters the filter. A **cross-check** that compares the two sensors before fusing them, and admits its doubt when they disagree, brings NEES to 4.7 (consistent) and the final error from 6.1% to 2.4%. It also cuts the false slip alarms raised during sensor faults from 10.4 to 0.8 per run. It stays consistent only if the moving pad's sensor is about as precise as the anchored one's: at 2 mm noise instead of 0.5 mm its NEES is 10.2, still better than ignoring that sensor (17.5) but overconfident.

![Four panels from inchworm_two_pad_fusion_ekf.py: both pad readings of one run with the slip threshold, slip pulses and sensor faults; the Monte Carlo midpoint position RMSE over time for the five variants; the Monte Carlo NEES of the five variants on a log scale against the consistent value 5; and NEES against the moving sensor's noise for the anchored sensor only, the unchecked moving sensor and the cross-check](../../assets/inchworm_two_pad_fusion_ekf.png)

*Figure: `use_numpy/inchworm_two_pad_fusion_ekf.py` at its defaults (seed 0, 300 trials), plus a sweep of the moving sensor's noise (100 trials per point), plotted by `uv run python generator/make_figures.py inchworm_two_pad_fusion_ekf` (about 2 minutes). The numbers match the script's printout.*

### Scope
The two-pad doc's planar toy (point pads, prescribed slip pulses, no position fix), with the additions above. The fault model, the fault rate and the moving sensor's noise are this repo's assumptions: the paper reads only the anchored pad's sensor and gives no reason, so whether a sliding pad's sensor is as reliable as an anchored one is **not verified** (§4 shows how much that matters). The cross-check is this repo's design.

---

## 1. Model

### 1.1 What the truth adds

The gait, turning, micro-slip and slip pulses are the two-pad doc's ([§1.1](inchworm_two_pad_ekf.md#11-ground-truth)). New:

| Part | What it does | Default |
| --- | --- | --- |
| **Stroke error** | In each phase the body executes the commanded stroke rate times a random gain, so the command is not exact. The encoders still measure the true length | gain **standard deviation** (std) 5% |
| **Spike** | For 0.3 s, a sensor's reading gains a backward displacement at 0.08 m/s: it looks like a slip | |
| **Dropout** | For 0.3 s, a sensor reads exactly 0 | |
| **Fault rate** | Each sensor, in each phase, has one fault episode with this probability; spike or dropout with equal probability | 20% |

The slip pulses are drawn once per **seed** (the starting value of the random-number generator) and held fixed over the [Monte Carlo trials](../glossary.md#monte-carlo-trials), as in the two-pad doc. Stroke gains, faults and all noise are redrawn every trial.

**Why these defaults.** No measured values exist for this robot, so each default is chosen to make one effect visible, and each has a flag:

| Default | Flag | Why this value |
| --- | --- | --- |
| Fault rate 20% per phase and sensor | `--p-fault` | About 8 fault episodes per 20 s run, enough to matter. Each sensor is faulty only about 6% of the time, so the two rarely fail together |
| Fault length 0.3 s (6 ticks) | `--t-fault` | Half a slip pulse: long enough to move the estimate, short enough to fit inside one phase |
| Spike speed 0.08 m/s | `--v-spike` | The peak slip speed, so a spike looks exactly like a slip to a single-sensor threshold: the worst case for the paper's detector. At 4 mm per tick it is about twice the threshold |
| Stroke error 5% | `--stroke-err-std` | A motor tracking error of a few percent |
| Moving-sensor noise 0.5 mm | `--moving-sensor-noise-std` | The same sensor as the anchored one; §4 tests a worse one |

Two options change the fault and stroke models:

- `--random-spike-speed`: each spike gets its own speed, uniform between 0.5 and 1.5 times `--v-spike`, and a random direction, instead of the same backward speed.
- `--stroke-err-per-cycle`: each contract phase reuses the gain of the extend phase before it. With the default per-phase gains, the body's length drifts by a random walk (std 1.8 cm after 20 phases, up to 4.9 cm, over 300 trials); with per-cycle gains it returns to its starting length every cycle.

### 1.2 Sensors

As in the two-pad doc ([§1.2](inchworm_two_pad_ekf.md#12-sensors)), plus the moving pad's sensor:

| Sensor | What it reads, every tick | Noise (default) |
| --- | --- | --- |
| Gyro | Body turn rate | 0.02 rad/s |
| **Anchored-pad sensor** | The anchored pad's displacement over the tick, in the body frame: $`d_a`$ | 0.5 mm per axis |
| **Moving-pad sensor** | The moving pad's displacement over the tick, in the body frame: $`d_m`$ | 0.5 mm per axis (assumed; §4) |
| Kinematic closure | Front pad relative to back pad, in the body frame | 1 mm per axis |

The **body frame** has x along the body (back pad to front pad) and y across it.

### 1.3 Five ways to move the moving pad

The state is the two-pad doc's, $`x = [b_x, b_y, f_x, f_y, \theta]`$ (back pad, front pad, heading), and so is the closure update. In every variant the anchored pad moves by $`d_a`$ (rotated into the world frame), as `pad_fused` does, except when `cross_check` blames the anchored sensor (§1.4). The variants differ only in how the moving pad moves:

| Variant | Moving pad moves by | Noise it assumes |
| --- | --- | --- |
| `random_walk` | Nothing: a loose random walk that the closure places. This is the two-pad doc's `pad_fused` exactly | 2 cm per tick |
| `encoder_model` | $`d_a + s\,e`$: the anchored pad's motion plus the commanded change of the body | anchored sensor + stroke error |
| `moving_sensor` | $`d_m`$ | moving sensor |
| `both` | The **inverse-variance combination** of the two above: each weighted by how precise it is | smaller than either |
| `cross_check` | As `both`, after the disagreement test of §1.4 | |

Here $`e`$ is the **commanded change** of the front-minus-back vector over the tick, in the body frame: the commanded stroke along the body plus the commanded turn across it. $`s = +1`$ when the front pad moves (extend) and $`-1`$ when the back pad moves (contract), because the moving pad's motion is the anchored pad's motion plus or minus the body's change.

`encoder_model` treats the stroke error as **white noise** (independent from tick to tick), with a std of 5% of the commanded stroke per tick. The true error is a gain that stays the same for a whole phase; §5 shows what that mismatch costs.

### 1.4 The cross-check

Each tick, `cross_check` compares the moving pad's reading with what the anchored reading and the command predict:

```math
r = d_m - (d_a + s\,e), \qquad \text{sensors disagree if } \; r^\top C_r^{-1} r > 3^2
```

$`r`$ is the **disagreement** (2 numbers, mm per tick); $`C_r`$ is its [covariance](../glossary.md#covariance) if both sensors are healthy (moving sensor + anchored sensor + stroke error); the test is a [Mahalanobis distance](../glossary.md#mahalanobis-distance) of 3, the same 3 that sets the slip threshold.

If the sensors agree, it fuses as `both`. If they disagree, one sensor is treated as faulty: the one whose reading implies the larger motion of the anchored pad, each measured against its own noise. In words, it trusts the sensor that agrees better with "the anchored pad is still":

- **Anchored sensor blamed** (its reading $`d_a`$ implies the larger motion): the anchored pad moves by $`d_m - s\,e`$, i.e. by what the moving sensor says minus the commanded stroke.
- **Moving sensor blamed:** the moving pad follows the encoder model.

What that rule gets right, case by case:

| Case | Blamed | Right? |
| --- | --- | --- |
| Spike on either sensor, no slip | The spiking sensor | Yes |
| Moving-sensor dropout, no slip | Moving | Yes |
| Spike on either sensor during a slip | The spiking sensor | Yes |
| Moving-sensor dropout during a slip | Moving, if the slip per tick is smaller than the stroke | Yes in that case; otherwise no |
| **Anchored-sensor dropout during a slip** | Moving | **No:** the anchored sensor reads 0 and the moving one reads the slip, exactly what a moving-sensor spike looks like |
| **Both sensors spike alike at once** | Neither: they agree | **No:** two equal spikes look exactly like a slip |

An anchored-sensor dropout with no slip needs no rule: the pad barely moved, so a 0 reading is nearly right and the sensors agree.

**Admitting the doubt.** The rule can blame the wrong sensor, and the two choices differ in where both pads moved by exactly $`r`$. So on every disagreement, $`r\,r^\top`$ (the disagreement times its own transpose, a 2×2 matrix whose size is that of $`r`$ squared) is added to the covariance of both pads' increments (and to their cross-covariance): the filter says "the pads are here, or $`r`$ away". The increments themselves are unchanged, and at the defaults the final error stays at 2.4%. It is on by default; `--no-cross-check-inflate` turns it off. §3 shows why it matters.

**Slip flags.** The paper's detector, here `threshold`, flags a slip when $`|d_a|`$ exceeds 3 standard deviations of a still pad's reading (2.12 mm per tick, as in the two-pad doc). `cross_check` flags it when the anchored-pad motion it actually used exceeds the same threshold.

---

## 2. Sanity check: no slip, no faults, no stroke error

`--p-slip 0 --p-fault 0 --stroke-err-std 0 --n-mc-trials 100`. A consistent filter's 100-run average NEES falls in [4.40, 5.64] 95% of the time (the **consistent band**).

| Variant | NEES | Final error (% of distance) | Heading [RMSE](../glossary.md#rmse-root-mean-square-error) (rad) |
| --- | --- | --- | --- |
| `random_walk` | 4.86 | 2.3 | 0.0242 |
| `encoder_model` | 5.32 | 1.9 | 0.0187 |
| `moving_sensor` | 5.13 | 1.9 | 0.0242 |
| `both` | 5.11 | 2.0 | 0.0235 |
| `cross_check` | 4.89 | 1.9 | 0.0233 |

All five are consistent, so every difference below comes from slip, faults or stroke error. "Final error" is the body midpoint's error at the end, averaged over the trials, as a percentage of the distance it traveled (about 0.8 m).

Reading a second sensor gains little here: the final error drops from 2.3% to 1.9%, because the closure already ties the moving pad to the anchored one within about 1 mm per tick. With an exact command, the encoder model also improves the heading by 23%, because the commanded turn adds information to the gyro's. The moving sensor's sideways reading does not: it sees the turn as the body length (0.15-0.23 m) times the turn angle, so its 0.5 mm noise is worth 2.2-3.3 mrad per tick, against the gyro's 1 mrad.

---

## 3. Results at the defaults

Defaults (slip in 5 of 20 phases, faults, 5% stroke error), 300 trials; consistent band [4.65, 5.36].

| Variant | NEES | Final error (% of distance) | Heading RMSE (rad) |
| --- | --- | --- | --- |
| `random_walk` | 17.40 | 6.1 | 0.0230 |
| `encoder_model` | 25.67 | 5.9 | 0.0292 |
| `moving_sensor` | 47.68 | 8.0 | 0.0256 |
| `both` | 44.23 | 7.6 | 0.0271 |
| `cross_check` | **4.67** | **2.4** | 0.0227 |

Over seeds 0-3 (100 trials each, 5 to 13 slip pulses), `cross_check` has NEES 4.7-5.3 and final errors of 2.7-3.3%, against 13.0-17.5 and 5.6-6.7% for `random_walk`.

#### The anchored sensor alone fuses every spike
`random_walk` fuses $`d_a`$ as it comes, so a spike on the anchored pad's sensor moves the estimate backward by about 4 mm per tick, for 0.3 s. Its NEES is 4.86 without faults and 17.4 with them.

#### Reading the moving sensor without a check makes it worse
`moving_sensor` and `both` also fuse $`d_m`$, so now the faults of both sensors enter. A moving-sensor dropout is the worst: the pad is extending at up to 5 mm per tick and the sensor reports 0. With dropouts only and no slip, `moving_sensor` reaches NEES 34.4 while `random_walk` stays at 4.86.

#### The cross-check catches most faults
It caught 1218 of 1222 spikes and 757 of 1145 dropouts. The dropouts it misses do little harm: with dropouts only and no slip, it misses 41% of them and stays consistent (4.20, table below). They are dropouts of a reading that was small anyway: an anchored pad barely moving, or a moving pad at the start or end of a stroke (a **ramp**), when the stroke is still small.

#### Why admitting the doubt matters
`--stroke-err-std 0 --n-mc-trials 100`, one fault type at a time (`--fault-type`), with and without the covariance inflation of §1.4:

| Faults | Slip | `random_walk` | `moving_sensor` | `cross_check` | `cross_check`, no inflation |
| --- | --- | --- | --- | --- | --- |
| Spikes only | no | 41.4 | 73.8 | 4.18 | 6.41 |
| Spikes only | yes | 41.5 | 74.0 | 4.17 | 6.38 |
| Random spikes only (`--random-spike-speed`) | no | 17.4 | 22.9 | 4.17 | 6.29 |
| Dropouts only | no | 4.86 | 34.4 | 4.20 | 5.04 |
| Dropouts only | yes | 6.32 | 27.4 | 4.62 | 7.71 |

Without inflation, the cross-check is overconfident in every row with spikes, and with dropouts during slip. Two kinds of error get into the estimate:

- **Undetected faults.** A few fault ticks pass the disagreement test: with equal spikes, mostly both sensors spiking alike at once; with random spikes, small or sideways spikes hidden in the noise. Random spikes almost never agree, yet the result barely changes (6.29 against 6.41), so the agreeing pairs are not the main cause.
- **Wrong blames.** An anchored-sensor dropout during a slip blames the moving sensor (§1.4), so the slip is missed. `random_walk` misses the same slips (6.32).

Either way the error stays: with no position fix, nothing corrects it, so most of the excess NEES shows up on later, fault-free ticks. Inflation covers every detected disagreement with its full size, which also absorbs the few undetected ticks on average. With spikes only it slightly over-covers (4.2, below the band [4.40, 5.64]); at the defaults it lands inside the band.

#### Slip detection

Defaults, 300 trials (1500 slip pulses; 11864 ticks in which the anchored pad truly moved more than the 2.12 mm threshold):

| Detector | Pulses detected | Slip ticks missed | False alarms during an anchored-sensor fault (per run) | Other false alarms (per run) |
| --- | --- | --- | --- | --- |
| `threshold` (paper) | 1500 | 1056 (8.9%) | 10.4 | 3.6 |
| `cross_check` | 1500 | 1095 (9.2%) | 0.8 | 3.3 |

*A false alarm is a flagged tick outside every slip pulse.*

Both detect every pulse and miss about the same slip ticks. Without faults both miss about 5.5% (`--p-fault 0`, 100 trials: noise near the threshold); the extra misses at the defaults come from anchored-sensor dropouts during a slip, which neither detector can see through. What the cross-check removes is the false alarms during faults: a spike on the anchored sensor looks like a slip to a threshold, but the moving sensor does not confirm it. The other false alarms are noise crossing the threshold, about 1% of ticks, as in the two-pad doc. The covariance inflation does not change the flags, which depend only on the readings.

The same default run, animated:

![Animation of the two-pad inchworm reading both pad sensors over 20 s: top view of the true pads and body with the anchored pad highlighted, pad outlines marking sensor spikes and dropouts, and three estimated midpoints; the anchored reading and the moving reading minus the commanded stroke with the slip threshold, fault episodes and the cross-check's detections; and the Monte Carlo NEES of the five variants on a log scale against the expected value 5](../../assets/inchworm_two_pad_fusion_ekf.gif)

*Animation: `generator/animate_inchworm_two_pad_fusion_ekf.py` at its defaults (seed 0, 300 trials), rendered by `uv run python generator/animate_inchworm_two_pad_fusion_ekf.py --out assets/inchworm_two_pad_fusion_ekf.gif`.*

---

## 4. The caveat: how good must the moving sensor be?

The paper reads only the anchored pad's sensor. If a sliding pad's sensor is noisier (a moving surface, vibration of the friction flaps), the cross-check weakens. `--moving-sensor-noise-std`, defaults otherwise, 100 trials each:

| Moving sensor noise (mm per reading) | `random_walk` | `moving_sensor` | `cross_check` | `cross_check`, no inflation | Spikes caught | Dropouts caught |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 (default) | 17.5 | 47.5 | **4.77** | 7.70 | 410 / 412 | 274 / 386 |
| 1 | 17.5 | 25.2 | **5.54** | 8.43 | 405 / 412 | 264 / 386 |
| 2 | 17.5 | 20.6 | 10.2 | 16.6 | 284 / 412 | 94 / 386 |
| 5 | 17.5 | 19.0 | 14.1 | 21.3 | 58 / 412 | 20 / 386 |

The disagreement test compares $`r`$ with the noise it expects. A spike adds 4 mm per tick; with a 2 mm sensor that is no longer clearly more than noise, and the test misses a third of the spikes and most dropouts. Undetected faults are exactly what the inflation cannot cover, so the cross-check turns overconfident, though it stays better than ignoring the moving sensor. **The cross-check stays consistent only if the moving pad's sensor is about as precise as the anchored one's (here up to about 1 mm).** That is the first thing to measure on the robot.

---

## 5. Other sweeps

**Fault rate** (`--p-fault`, 100 trials each):

| Fault rate per phase and sensor | `random_walk` NEES | `cross_check` NEES | False alarms during an anchored-sensor fault, `threshold` / `cross_check` (per run) |
| --- | --- | --- | --- |
| 0 | 4.86 | 5.28 | 0 / 0 |
| 10% | 9.38 | 4.77 | 5.4 / 0.2 |
| 20% (default) | 17.5 | 4.77 | 11.2 / 0.8 |
| 50% | 54.1 | 4.93 | 26.0 / 4.9 |

The cross-check stays consistent even when every sensor fails in half the phases; only its false alarms grow, as faults on both sensors start to overlap.

**Random spikes** (`--random-spike-speed`, defaults otherwise, 100 trials): `random_walk` 12.9, `moving_sensor` 27.9, `cross_check` 4.92. Spikes of random size and direction hurt the unchecked variants less, likely because their directions vary and their errors partly cancel; the cross-check stays consistent.

**Stroke error and the encoder model** (`--p-fault 0 --stroke-err-std`, 100 trials each; gyro alone gives a heading RMSE of 0.0242 rad):

| Stroke error (gain std) | `encoder_model` NEES | `encoder_model` heading RMSE (rad) | `cross_check` NEES |
| --- | --- | --- | --- |
| 0 | 5.32 | 0.0187 | 4.88 |
| 5% (default) | 9.14 | 0.0281 | 5.28 |
| 10% | 12.1 | 0.0426 | 5.59 |
| 20% | 24.4 | 0.0617 | 5.55 |

The stroke error is a gain that persists for a whole phase, but `encoder_model` treats it as white noise, so it underestimates how far the error adds up within a phase and turns overconfident. Through the commanded turn, the wrong stroke also makes its heading worse than the gyro alone at 5% and above. `cross_check` barely notices: the moving sensor gets at least half the weight in the combination (more as the stroke error grows), and the encoder model is only its fallback.

With per-cycle gains (`--stroke-err-per-cycle --p-fault 0`), `encoder_model` is at 8.60 instead of 9.14: keeping the body length from drifting hardly matters, because the error still persists through each phase.

---

## 6. What this toy does not show

- **The real moving-pad sensor.** Its noise and fault behavior while sliding are assumed (§4). The paper's sensor also has a surface-quality output, which would flag many dropouts directly; this toy does not model it.
- **Realistic faults.** The spikes are either all alike or, with `--random-spike-speed`, independent between the sensors. A common cause (vibration, lighting) could hit both sensors at once, in the same way, and look like a slip.
- **Fault isolation with more information.** An anchored-sensor dropout during a slip cannot be told from a moving-sensor spike with these three sources. A surface-quality flag, a third sensor, or a model of how often slips happen would break the tie; the inflation only admits the doubt.
- **Estimating the stroke gain.** Putting the per-phase gain in the state would let the encoder model learn the stroke error instead of treating it as white noise.
- **A position fix**, gyro bias, and slip physics, as in the two-pad doc ([§6](inchworm_two_pad_ekf.md#6-what-this-toy-does-not-show)).

---

## 7. Takeaway

Two pad sensors are worth more for checking each other than for measuring more. A slip moves both pads, so the sensors agree on it; a faulty sensor makes them disagree. Fusing the second sensor blindly adds its faults to the filter. Comparing the two first removes most faults and almost all the false slip alarms they cause, and admitting the doubt on a disagreement keeps the filter consistent, but only if the moving pad's sensor is about as precise as the anchored one's. Some cases stay out of reach of any such rule (two faults that agree, an anchored-sensor dropout during a slip); the filter can only say it is unsure.

---

## 8. References

1. Kubík, J. & Faigl, J. *Adaptive Friction-based Inchworm-like Locomotion*. CTU Computational Robotics Laboratory. https://comrob.fel.cvut.cz/papers/icra26looper.pdf - an inchworm robot with a laser-mouse displacement sensor in each pad, read on the anchored pad only, with a fixed slip threshold and a surface-quality output; lists Kalman fusion with motor and inertial data as future work.
