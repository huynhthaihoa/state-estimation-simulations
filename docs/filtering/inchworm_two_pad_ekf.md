# Two-pad inchworm: trusting the anchored pad without a position fix

## Intuition

[`inchworm_zupt_ekf.md` §6](inchworm_zupt_ekf.md#6-side-extension-anchor-slip-and-a-measured-gate) (**inchworm §6** below) asked when a slipping anchor still justifies "$`v = 0`$", using a 1D point mass, one sensor that reads the body in every phase, and a noisy position measurement every **tick** (one filter step, 0.05 s). A real inchworm robot has none of those simplifications. This doc drops them:

```text
EXTEND:    back pad anchored   ▓B━━━━━━━━F→   front pad slides forward, steering turns the body
CONTRACT:  front pad anchored     ←B━━━━F▓    back pad slides up behind it
```

- **Two pads.** The robot has a **back pad** and a **front pad**. In each phase one of them is the **anchored pad** (gripping, ideally still) and the other is the **moving pad**. They swap every phase.
- **One sensor per pad, read only while anchored.** Each pad has an optical displacement sensor, like a laser mouse, that reports how far the pad moved over the ground since the last reading. As on the robot that motivates this doc (Kubík & Faigl, reference 1), only the anchored pad's sensor is used.
- **No position fix.** Nothing measures where the robot is in the world. The filter can only add up small motions: **dead reckoning** ([glossary](../glossary.md#dead-reckoning)).
- **Turning.** The body turns while it extends, and a gyro measures the turn rate.

**The question stays the same as in inchworm §6:** when the **schedule** (the known timetable of which pad is anchored) says a pad is anchored, should the filter believe it is still?

**Short answer:** without a position fix, a slip that the filter does not track is never corrected. A sensor **veto** ("the pad moved, so don't trust 'still' on this tick") reduces the **overconfidence** (the filter claiming less uncertainty than its errors show) but leaves the position error almost as large as ignoring the sensor. Only **fusing** the reading, i.e. moving the anchored pad by the measured displacement, keeps both the position and the uncertainty right. Over four seeds (four different random slip patterns) that policy has NEES ([glossary](../glossary.md#consistency-and-nees); 5 is consistent here) of 4.84-5.01 and a final position error of 1.3-4.1% of the distance traveled, against 58-483 and 19-51% for "anchored, so still".

![Four panels from inchworm_two_pad_ekf.py at its defaults: top view of the true body midpoint and four estimates, the anchored-pad sensor reading with the veto threshold, the midpoint position error over time, and the Monte Carlo NEES per policy on a log scale against the consistent value 5](../../assets/inchworm_two_pad_ekf.png)

*Figure: `use_numpy/inchworm_two_pad_ekf.py` at its defaults (seed 0, 500 trials), plotted by `uv run python generator/make_figures.py inchworm_two_pad_ekf` (about 2 minutes). The numbers match the script's printout.*

### Scope
Planar motion; pads are points that never lift off; the body is a straight segment between them. Slip is a prescribed pulse, not friction physics. The numbers are illustrative, not the robot's. The paper has no filter; the filter here is this repo's design, built on the idea of a contact point held in the state from [`hexapod_contact_inekf.md`](hexapod_contact_inekf.md).

---

## 1. Model

### 1.1 Ground truth

| Part | What it does | Default |
| --- | --- | --- |
| **Gait** | Alternating phases of length $`t_{\text{phase}}`$. **Extend:** back pad anchored, body length $`l`$ grows. **Contract:** front pad anchored, $`l`$ shrinks back. The stroke rate $`\dot l`$ is a trapezoid (peak $`v_{\text{stroke}}`$, ramps $`t_{\text{ramp}}`$), as in the 1D toy. $`l_{\min}`$ is the body length at the start of extend (its shortest) | 1 s phases, 0.1 m/s, 0.2 s ramps, $`l_{\min}`$ = 0.15 m |
| **Turning** | During extend only, the heading $`\theta`$ (direction from back pad to front pad) turns at $`\dot\theta = \dot l \tan\phi / l_{\min}`$: bicycle-style steering, where $`\phi`$ is the steering angle and $`l_{\min}`$ plays the **wheelbase** (on a bicycle, the distance between the axles). The paper uses the same idea, with turn radius $`R = L / \tan\phi`$ for a wheelbase $`L`$ | $`\phi`$ = 0.3 rad |
| **Micro-slip** | The anchored pad is never perfectly still: it creeps by small random steps (**white noise** on its velocity: a random value, independent from tick to tick). This is the "approximately still" of the 1D doc's caveats | 0.01 m/s per axis |
| **Slip pulse** | With probability $`p_{\text{slip}}`$ per phase, the anchored pad slides backward along the body in one trapezoidal pulse (same shape as inchworm §6). The rest of the body slides with it | 0.5, peak 0.08 m/s, 0.6 s |

The slip pulses are drawn once per seed and held fixed over the [Monte Carlo trials](../glossary.md#monte-carlo-trials), as in inchworm §6. Everything else, including the micro-slip, is redrawn every trial, so the noise under test is resampled.

### 1.2 Sensors

| Sensor | What it reads, every tick | Noise (default) |
| --- | --- | --- |
| **Gyro** | Body turn rate $`\dot\theta`$ | 0.02 rad/s |
| **Pad sensor** | The anchored pad's displacement over the tick, in the **body frame** (x along the body, y across), $`d_k`$ | 0.5 mm per axis |
| **Kinematic closure** | Where the front pad is relative to the back pad, in the body frame, from the **motor encoders** (sensors that count how far each motor has turned): ideally $`[l, 0]`$ | 1 mm along, 1 mm across |

**Kinematic closure** means: the motors know how long the body is, so they give the front pad's position relative to the back pad, along the body. That "closes the loop" between the two pads. They say nothing about the heading in the world, which comes only from the gyro. It is the only measurement that says anything about the moving pad, because its own sensor is not used.

### 1.3 Filter

**State** $`x = [b_x, b_y, f_x, f_y, \theta]`$: back pad position $`(b_x, b_y)`$ and front pad position $`(f_x, f_y)`$ in meters (world frame), and heading $`\theta`$ in radians. Five numbers, so a consistent filter has NEES ≈ 5.

#### Predict: one tick of length $`\Delta t`$

**Process noise** is the uncertainty each prediction adds ([glossary](../glossary.md#predict-and-update-steps)); the last column gives its standard deviation.

| State part | Prediction | Process noise (std) |
| --- | --- | --- |
| Heading $`\theta`$ | Add the gyro reading $`\tilde\omega`$: $`\theta \mathrel{+}= \tilde\omega\,\Delta t`$ | gyro noise × $`\Delta t`$ |
| Moving pad | Unchanged: no motion model. The closure update places it | 2 cm per tick (loose) |
| Anchored pad | Depends on the policy, §1.4 | Depends on the policy |

#### Update, every tick: the kinematic closure
The predicted reading is the front-minus-back vector rotated into the body frame:

```math
h(x) = R(\theta)^\top (f - b), \qquad
H = \begin{bmatrix} -R(\theta)^\top & R(\theta)^\top & \dfrac{\partial R(\theta)^\top}{\partial \theta}(f - b) \end{bmatrix}
```

Here $`R(\theta)`$ is the 2×2 rotation from the body frame to the world frame, and $`H`$ is the Jacobian of $`h`$ with respect to the state ([glossary](../glossary.md#jacobian)). $`h`$ depends on $`\theta`$ through a rotation, so this is a genuine EKF ([glossary](../glossary.md#ekf-extended-kalman-filter)) update, unlike the 1D toy's linear one.

### 1.4 Four policies for the anchored pad

| Policy | What the filter assumes about the anchored pad on each tick | Process noise |
| --- | --- | --- |
| `never` | Nothing: it wanders loosely | 0.1 m/s × $`\Delta t`$ |
| `schedule` | "Anchored, so still": it stays put, up to micro-slip. Blind to slip pulses | 0.01 m/s × $`\Delta t`$ (matches the true micro-slip) |
| `schedule_and_sensor` | As `schedule`, unless the pad sensor reads more than the **veto threshold**; then as `never` for that tick | Tight, or loose on a veto |
| `pad_fused` | It moved by the sensor reading, rotated into the world frame: $`b \mathrel{+}= R(\theta)\, d_k`$ (or $`f`$) | The sensor noise, 0.5 mm |

The veto threshold is 3 standard deviations of a still pad's reading (sensor noise plus micro-slip): 2.12 mm per tick, a speed of 0.042 m/s.

The last two rows differ the way inchworm §6's gate and fusion do: a veto uses the reading as a yes/no flag, fusion uses its value.

---

## 2. Sanity check: no slip

With `--p-slip 0`, 500 trials. A consistent filter's 500-run average NEES falls in [4.73, 5.28] 95% of the time (the **consistent band**). Above it the filter is **overconfident** (it claims less uncertainty than its errors show); below it, **underconfident** (it claims more).

| Policy | NEES | Reading |
| --- | --- | --- |
| `never` | 3.46 | Underconfident, by design: it assumes a much looser anchor than the truth |
| `schedule` | 4.91 | Consistent |
| `schedule_and_sensor` | 4.35 | Slightly underconfident: about 1% of ticks cross the threshold by chance (false alarms), and each one loosens the filter |
| `pad_fused` | 4.92 | Consistent |

So with a still anchor, both `schedule` and `pad_fused` are honest, and any problem below comes from slip.

---

## 3. Results with slip

Defaults (`p_slip` 0.5), 500 trials per seed. "Final error" is the single run's body-midpoint error at the end, as a percentage of the distance the midpoint traveled (0.71-0.80 m).

| Seed (slip pulses of 20 phases) | `never` | `schedule` | `schedule_and_sensor` | `pad_fused` |
| --- | --- | --- | --- | --- |
| 0 (5) | 3.19 / 8.7% | 58.8 / 19.1% | 10.3 / 17.8% | **4.92 / 3.2%** |
| 1 (10) | 5.54 / 27.6% | 295.6 / 39.1% | 22.1 / 36.1% | **4.84 / 3.8%** |
| 2 (11) | 7.33 / 35.9% | 426.5 / 46.7% | 26.5 / 43.3% | **5.01 / 4.1%** |
| 3 (13) | 8.09 / 39.3% | 483.3 / 50.8% | 28.2 / 47.0% | **4.96 / 1.3%** |

*Each cell: overall Monte Carlo NEES / final error.*

The heading [RMSE](../glossary.md#rmse-root-mean-square-error) is the same for all four policies in every seed (0.008-0.037 rad): the heading comes only from the gyro, and no policy changes it.

#### One slip, in numbers
Before the four mechanisms below, one pulse: seed 0's first slip (1.0-1.6 s, a 3.2 cm backward slide), averaged over 300 trials. "Error" is the body midpoint's mean error along the body axis just after the pulse; "reported σ" is the filter's own standard deviation in that direction. Before the pulse every policy is within about 5-7 mm. (Computed with the script's functions; not part of its printout.)

| Policy | Error after the pulse | Reported σ | Error / σ | Reading |
| --- | --- | --- | --- | --- |
| `schedule` | 32.0 mm | 5.8 mm | 5.5 | Missed the whole slide, and is sure of itself |
| `schedule_and_sensor` | 29.8 mm | 14.9 mm | 2.0 | Flagged the slide on about 8 ticks and loosened, but did not move: still wrong, and not loose enough |
| `pad_fused` | 0.2 mm (spread 5.4 mm) | 5.8 mm | 0.03 (spread / σ ≈ 0.9) | Followed the slide |
| `never` | 23.7 mm | 28.7 mm | 0.8 | Wrong, but its loose anchor covers one slide |

#### The schedule is blind to slip, and the error stays
Each slip pulse moves the robot 3.2 cm backward (the area under the trapezoid: peak 0.08 m/s × (0.6 s − one 0.2 s ramp)). `schedule` assumes the pad stood still, so it misses the whole slide. In inchworm §6 the next position measurements pulled the estimate back. Here nothing does, so the error stays for the rest of the run, and the NEES climbs in steps, one per pulse (figure, bottom right). Between pulses the NEES drifts down slowly, because the filter's uncertainty keeps growing with dead reckoning while the error stays put.

This is also why the "clean" and "slip" NEES windows used in inchworm §6 no longer separate here (the script still prints them): an error made during a pulse persists into the clean phases after it.

#### A veto stops trusting the pad, but cannot say where it went
`schedule_and_sensor` sees the pulse (the reading jumps to 4-5 mm per tick, figure top right) and loosens the anchored pad on those ticks. That reduces the overconfidence (NEES 10-28 instead of 58-483), but the position error is almost as large as `schedule`'s. Loosening admits "I don't know where the pad is". It gives the filter no information about where the pad actually went, and with no position fix nothing else does either. In inchworm §6 the veto removed most of the damage (slip-window NEES about 47 → 3.5) only because the position measurements could correct the estimate afterward.

The veto also misses the start and end of each pulse. At the threshold speed (0.042 m/s) and the ramp's acceleration (0.08 m/s over 0.2 s), each ramp spends about 0.1 s below the threshold. That leaves about 2 mm of unflagged slide per ramp, about 4.5 mm per pulse, against the 0.5 mm per tick the filter allows (a computed estimate, ignoring sensor noise).

#### Fusing the reading keeps the position
`pad_fused` moves the anchored pad by what its sensor measured, so a slide is followed, not just flagged. Its NEES stays at 4.84-5.01 in every seed and its final error at 1.3-4.1% of the distance, whatever the number of pulses. This is the step the paper lists as future work: Kalman fusion of the pad sensor with motor and inertial data.

#### `never` is not safe either
`never`'s loose anchor (0.1 m/s) covers a slip pulse's speed, but not its direction: every pulse slides backward, while the filter models independent random steps. With more pulses, the bias adds up and `never` turns overconfident (NEES 3.19 → 8.09 from 5 to 13 pulses), with errors of 9-39%.

The same default run, animated:

![Animation of the two-pad inchworm over 20 s: top view of the true pads and body with the anchored pad highlighted and each policy's estimated midpoint, the anchored-pad sensor reading with the veto threshold, and the Monte Carlo NEES of the four policies on a log scale against the expected value 5](../../assets/inchworm_two_pad_ekf.gif)

*Animation: `generator/animate_inchworm_two_pad_ekf.py` at its defaults (seed 0, 500 trials), rendered by `uv run python generator/animate_inchworm_two_pad_ekf.py --out assets/inchworm_two_pad_ekf.gif`.*

---

## 4. Tuning the veto

The veto has two knobs: **how loose it makes the pad** (`--free-noise-std`) and **where the threshold sits** (`--gate-sigmas`). Seed 0, 200 trials each (consistent band for a 200-run average: [4.57, 5.45]):

| Setting | Veto NEES | Veto final error |
| --- | --- | --- |
| Defaults: 0.1 m/s, 3σ (500 trials) | 10.30 | 17.8% |
| Looser: 0.3 m/s | 3.57 | 11.4% |
| Much looser: 1.0 m/s | 3.12 | 4.8% |
| Lower threshold: 1.5σ (1.06 mm per tick) | 4.33 | 14.6% |

- **Consistency can be tuned.** Loosening more, or catching more of the ramps with a lower threshold, removes its overconfidence (NEES 3.1-4.3, now slightly below the band: cautious rather than overconfident).
- **The low error at 1.0 m/s should not be read as a property of the veto.** When the anchored pad is made much looser than the moving pad (2 cm per tick), the closure update hands the body's position to the moving pad's estimate. In this toy the moving pad has no motion model, so the result depends on how far the moving pad happens to travel during a pulse. A slower slip (0.04 m/s) gives 3.6%, a faster one (0.15 m/s) gives 12.9% (both at 1.0 m/s). A real filter would predict the moving pad from the encoders, and this effect would change.
- **Fusion needs no such tuning.** Its process noise is the sensor's own noise.

---

## 5. Experiments left out, and why

- **Kinematic closure on/off.** With the moving pad's sensor unused and no position fix, closure is the only measurement of the moving pad. Switched off, the moving pad's estimate wanders freely (2 cm per tick), and at the next phase it becomes the anchored pad. The filter would lose track of the body within a phase (expected, not run), which says nothing about slip. Closure is therefore always on.
- **Gyro noise or gyro on/off.** The heading comes only from the gyro and is the same for all four policies (§3), so a gyro study would not separate them. It is left for a later version.

---

## 6. What this toy does not show

- **The moving pad's sensor.** In a sliding gait the moving pad stays on the ground, so its sensor could measure that end's motion too: direct odometry of the moving end, on top of the encoders. The paper reads only the anchored pad, and so does this toy. [inchworm_two_pad_fusion_ekf.md](inchworm_two_pad_fusion_ekf.md) adds it.
- **A motion model for the moving pad** from the encoder stroke rate and the steering angle. Here it is a loose **random walk** (it drifts by independent random steps) placed by the closure update (§4 shows where that matters).
- **Real sensor behavior.** The paper's sensor reads every 10 ms at 1200 counts per inch and has a surface-quality output. This toy has white Gaussian noise every 50 ms, no quantization (rounding to whole counts), no dropout (missed readings) and no lift-off.
- **Slip physics.** Slip is a prescribed backward pulse along the body; the robot's real friction scales (3D-printed flaps that grip backward and slide forward) slip by amounts that depend on load and terrain.
- **Gyro bias** and estimating it.

---

## 7. Takeaway

Without a position fix, the only way to recover from anchor slip is to measure it. A schedule that assumes the anchor is still misses every slip, and the error stays. A sensor veto makes the filter admit it lost track, but it still loses track. Fusing the anchored pad's measured displacement keeps the filter both accurate and consistent, whatever the number of slips.

---

## 8. References

1. Kubík, J. & Faigl, J. *Adaptive Friction-based Inchworm-like Locomotion*. CTU Computational Robotics Laboratory. https://comrob.fel.cvut.cz/papers/icra26looper.pdf - an inchworm robot with a laser-mouse displacement sensor in each pad, read on the anchored pad to detect slip with a fixed threshold; lists Kalman fusion with motor and inertial data as future work. Turning uses $`R = L / \tan\phi`$ (turn radius $`R`$, wheelbase $`L`$, steering angle $`\phi`$).
2. Hartley, R., Ghaffari Jadidi, M., Grizzle, J. W., & Eustice, R. M. (2018). *Contact-Aided Invariant Extended Kalman Filtering for Legged Robot State Estimation*. RSS 2018. https://arxiv.org/abs/1805.10410 - a contact point kept in the state and assumed fixed in the world, with its velocity modeled as noise so that slip is allowed: the idea behind treating the anchored pad as a state here.
