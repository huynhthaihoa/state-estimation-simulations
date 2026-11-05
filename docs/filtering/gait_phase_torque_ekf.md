# Torque-based contact detection: what decides it, and when the gait phase helps

## Intuition

[`gait_phase_ekf.md`](gait_phase_ekf.md) (**the gait-phase doc** below) decides which feet of a hexapod are on the ground when touchdowns come late or early. Its **contact detector** follows the servo **position error** (the gap between the commanded and the measured joint angle) of Faigl & Čížek (ref. 1): it fires some time after touchdown and sometimes fires falsely early in the **swing-down** (the last third of a leg's swing, when the foot moves down toward the ground). The doc models that detector by its outputs only, a delay and a false-trigger rate, both assumed.

Robots built from **series-elastic actuators** ([glossary](../glossary.md#series-elastic-actuator): motor modules with a spring between the gearbox and the output, whose deflection gives the output torque), such as HEBI's X-Series in the **HEBI Daisy** hexapod (ref. 3), measure **joint torque** instead. A torque detector asks a different question: "is the leg carrying weight?" This doc models it one level down, as a signal, so that its delay and false triggers come out of the signal instead of being set by hand:

- The torque is turned into the **vertical foot force** the leg pushes with. It is near 0 while the foot swings and rises to the foot's share of the body weight after touchdown.
- A detector fires when that force crosses a threshold.

**Short answer.**

- **Small torque noise hardly matters.** Daisy's legs turn 1 Nm of torque noise into only 4.3 N of force noise, against a 101 N load per foot. With the gait-phase filter, the results stay the same up to 1 Nm, 100 times HEBI's published 0.01 Nm resolution. A raw threshold already fires early more often at 1 Nm. HEBI publishes no noise value, so the noise levels are assumed and swept.
- **Two unpublished quantities decide the detector:**
  - **Model error:** the leg dynamics the force estimate leaves out, which can look like force during the swing.
  - **Load ramp:** how fast the weight comes onto the leg after touchdown.

  The model error sets the false triggers, and the load ramp sets the delay.
- **A raw torque threshold is not enough.** Model error near the threshold makes it fire early, and the body filter becomes overconfident.
- **The gait-phase filter fixes it.** Fed with the torque triggers, the gait-phase doc's touchdown filter (`phase_ekf`) rejects almost every early trigger at every model-error size tried, and the body filter stays consistent.
- **Fusion helps only with slow load ramps.** Adding the gait phase as a prior to the force reading, after Bledt et al. (ref. 5), roughly halves the missed stance at a slow 200 ms ramp. On a leg that lands very late, its prior outvotes "no force" and reads the swinging foot as planted.

![Four panels from gait_phase_torque_ekf.py at its defaults: the estimated vertical foot force of one step against the threshold, with a false trigger from a model-error pulse early in the swing-down and the true trigger 20 ms after touchdown; and three sweeps of wrong stance ticks per trial (swinging foot read as planted plus true stance missed) against the model-error size, the load-ramp time and the torque noise, for the three torque rows, with the gait-phase doc's position-error detector and phase EKF as dashed lines](../../assets/gait_phase_torque_ekf.png)

*Figure: `uv run python use_numpy/gait_phase_torque_ekf.py --out assets/gait_phase_torque_ekf.png`.*

### Scope
The gait-phase doc's planar hexapod, uncertain touchdowns (growing lag, terrain scatter, one leg blocked 0.2 s late from 10 to 14 s), body EKF and scores, all unchanged. Its five rows (four ways to set the stance flags, plus the true stance) are kept as they are. Three torque rows are added, fed by a simulated torque signal sized to Daisy. Run with `uv run python use_numpy/gait_phase_torque_ekf.py` (about 2 minutes).

---

## 1. The torque signal

### 1.1 From joint torque to foot force
A leg's joint torques $`\tau`$ (3 numbers, Nm) and the force $`f`$ the foot pushes on the ground (3 numbers, N) are linked by the leg's **Jacobian** $`J`$ ([glossary](../glossary.md#jacobian)), the matrix of how fast the foot moves per radian of each joint. Following Camurri et al. (ref. 4):

```math
f = -\left(J^\top\right)^{-1} \left(\tau - h(q, \dot q)\right)
```

- $`h(q, \dot q)`$: the torque the leg needs for its own motion and weight (gravity, Coriolis and centrifugal terms), from a model of the leg. Here $`q`$ and $`\dot q`$ are the joint angles and their rates.
- Only the vertical part of $`f`$ is used.

With Daisy's geometry and nominal stance ([`hexapod_joint_kinematics_inekf.md`](hexapod_joint_kinematics_inekf.md) §1.1), white torque noise of std 1 Nm on each joint gives vertical-force noise of std **4.27 N**.

### 1.2 What the signal is made of
Per step, the script samples the estimated vertical force at 1 kHz, from the start of the swing-down (or the true touchdown, if earlier) to the next liftoff. The signal is the sum of three parts:

| Part | Model | Default | Source |
| --- | --- | --- | --- |
| **Load** | 0 before the true touchdown, then a linear ramp to the foot's share of the weight ([load share](../glossary.md#load-share): body weight / 3, tripod) over the **load ramp** time (tripod gait: three feet down at a time) | share 101.4 N (31 kg); ramp 50 ms | mass: HEBI `hex_config.xml` "body mass" (ref. 3); ramp: assumed |
| **Model error** | what $`h`$ misses (unmodeled swing dynamics): one half-sine pulse per step over the first half of the swing-down, the window where the gait-phase doc puts its false triggers; normal amplitude, random sign | std 15 N | assumed |
| **Noise** | white torque noise, mapped to force (×4.27 N/Nm) | 0.1 Nm → 0.43 N | assumed; HEBI resolution 0.01 Nm (ref. 2) |

**The numbers are assumptions.**
- HEBI publishes the torque resolution (0.01 Nm) and the 1 kHz feedback rate (ref. 2), but no torque noise, bandwidth, model error or load-ramp time. The torque noise, model error and load ramp are swept; bandwidth is not modeled.
- The mass is HEBI's controller setting. A third-party page lists Daisy at about 18 kg (ref. 6; 58.9 N per foot); the mass only scales the load share.
- At 31 kg, holding the share takes about 31-36 Nm at the second joint, close to the X8-16's 38 Nm peak (ref. 2).

---

## 2. Detectors

| Row | Stance starts at | New here? |
| --- | --- | --- |
| `schedule`, `detector`, `cpg`, `phase_ekf`, `oracle` | the gait-phase doc's rows, on its position-error triggers (mean delay 40 ms, 10% of steps with a false trigger: its assumptions) | no |
| `torque_detector` | the first **torque trigger** of the step: the force rises above 30% of the load share (30.4 N), **re-armed** (allowed to fire again) only after it falls below half that, at the first 50 Hz filter tick at or after the crossing | yes |
| `torque_phase_ekf` | the gait-phase doc's `phase_ekf` on the torque triggers. It knows the noise-free delay: load ramp × 30%, rounded up to a tick (20 ms at the defaults) | yes |
| `torque_fused` | the first filter tick at or after the first 1 kHz sample where a **fused contact probability** passes 0.5 (below) | yes |

**The fused probability** combines two numbers as [log-odds](../glossary.md#log-odds) (for a probability $`p`$, the log of $`p/(1-p)`$; log-odds add when pieces of evidence are independent):

- **Prior from the gait phase:** the probability that the foot has landed by time $`t`$, a normal distribution around the touchdown that `torque_phase_ekf` predicts. Its std is the predicted lag's std combined with the terrain scatter.
- **Evidence from the force:** a **logistic** ([glossary](../glossary.md#logistic-function-and-logistic-regression)) in the force, $`P(\text{contact} \mid f) = 1 / (1 + e^{-(\beta_1 f / F_\text{share} + \beta_0)})`$, as in Camurri et al. (ref. 4). Here $`F_\text{share}`$ is the foot's load share (101.4 N), and the weights $`\beta_1`$ (slope) and $`\beta_0`$ (offset) are fitted on one separate training trial, labelled by the true contact, and refitted for every sweep setting. The training data has its own **base rate** ([glossary](../glossary.md#base-rate)): the fraction of its samples at or after the true touchdown (0.65 at the defaults). The logistic has that rate built in, so its log-odds are subtracted before adding, to avoid counting a prior twice.

This is a simplified version of the idea of Bledt et al. (ref. 5), who fuse a force estimate with contact priors in a Kalman filter. The details here are this script's.

**Checked** (`tests/use_numpy/test_gait_phase_torque_ekf.py`, 7 tests):
- the 4.27 N/Nm factor from Daisy's stance;
- the load ramp and the pulse window;
- trigger ticks, re-arming, and dropping triggers after liftoff;
- the noise-free torque delay equals the delay the filter knows;
- the logistic fit recovers known weights;
- the position rows are identical to the gait-phase doc's for the same seed: the torque draws come last.

---

## 3. Results at the defaults (100 trials)

The body EKF's state has 6 numbers (planar position, heading and the three velocities). A consistent filter's run-average NEES ([glossary](../glossary.md#consistency-and-nees)) is about 6, and for 100 [Monte Carlo trials](../glossary.md#monte-carlo-trials) it falls in [5.34, 6.70] 95% of the time.

The two count columns are per trial, summed over the 6 legs and the run's 1200 ticks:
- **swing read:** a swinging foot read as planted;
- **missed:** a planted foot not read.

| Row | NEES | forward-speed RMSE (m/s) | turn-rate RMSE (rad/s) | swing read | missed |
| --- | --- | --- | --- | --- | --- |
| `schedule` | 42.97 | 0.0114 | 0.0200 | 400.4 | 10.3 |
| `detector` (position) | 58.45 | 0.0121 | 0.0275 | 137.5 | 212.7 |
| `cpg` (position) | 6.26 | 0.0039 | 0.0106 | 0.0 | 250.8 |
| `phase_ekf` (position) | 6.39 | 0.0039 | 0.0106 | 1.2 | 240.8 |
| `oracle` (true stance) | 6.25 | 0.0038 | 0.0104 | 0.0 | 0.0 |
| `torque_detector` | 14.09 | 0.0063 | 0.0156 | 31.2 | 117.1 |
| `torque_phase_ekf` | 6.30 | 0.0038 | 0.0105 | 0.1 | 122.6 |
| `torque_fused` | 6.91 | 0.0040 | 0.0108 | 21.3 | 115.8 |

Early triggers (a trigger before the true touchdown) hit 10.2% of steps for position and 2.4% for torque.

#### A raw torque threshold still reads swinging feet
A model-error pulse above 30.4 N fires `torque_detector` early. That happens on 2.4% of steps at a 15 N std, close to the 2.1% chance that a normal amplitude exceeds 30.4 / 15 = 2.03 std. Each early trigger reads a swinging foot as planted, and NEES rises to 14.

#### The gait-phase filter removes them
`torque_phase_ekf` rejects triggers that come too early for the predicted touchdown, as in the gait-phase doc. It keeps NEES at 6.30, inside the band.

#### Fewer missed ticks than position error: a comparison of two assumptions
`torque_phase_ekf` misses half as many stance ticks as the position `phase_ekf`. That follows from the two delays: 20 ms (this doc's assumed ramp and threshold) against 40 ms (the gait-phase doc's assumed position-error delay). Neither delay is measured, so this is not evidence that torque detection is faster on a real robot.

#### Fusion reads the blocked leg's swing as planted
`torque_fused` reads about 21 swinging-foot ticks per trial as planted, and NEES rises to 6.91, just above the band. A check over 5 trials puts all of them on the blocked leg L2, and none when blocking is turned off.

The force evidence has a ceiling. At zero force, the force evidence is only -4.7 in log-odds (the logistic's offset -4.10 minus the base rate's log-odds 0.64), because the samples just after a true touchdown also show almost no force while the load is still ramping. The prior has no such ceiling: its log-odds keep growing while the foot stays late. Once the time passes 2.4 std beyond the predicted touchdown, the prior outvotes the zero force. A foot 0.2 s late, against a std of a few tens of ms, passes that point.

The first trial of the default run (seed 0), animated for leg L2 from 7.4 to 14.6 s. The **model-error pulse** (the unmodeled leg dynamics, grey bands) is a bump in the force during the swing-down, and an **early trigger** is a torque trigger before the true touchdown:

![Animation of gait_phase_torque_ekf.py: top view of the hexapod with feet filled in stance and hollow in swing, three small dots beside each foot for what torque_detector, torque_phase_ekf and torque_fused read, and a pink ring on a swinging foot read as planted; the simulated vertical foot force of leg L2 scrolling against the 30.4 N threshold, with the true load, the model-error window, the true touchdowns and the torque triggers; and the stance flags of L2 for the true stance and the three torque rows, with swinging-foot reads highlighted](../../assets/gait_phase_torque_ekf.gif)

*Animation: `generator/animate_gait_phase_torque_ekf.py` at its defaults (seed 0, the same run as the script's single run; the window starts before L2's early trigger), rendered by `uv run python generator/animate_gait_phase_torque_ekf.py --out assets/gait_phase_torque_ekf.gif`. In this run 2 of 120 steps (1.7%) have an early trigger (2.4% over the 100 trials). At 8.84 s a +34.3 N pulse takes L2's force over the 30.4 N threshold 0.18 s before the true touchdown (9.02 s): `torque_detector` reads L2 as planted for 9 ticks, `torque_phase_ekf` for none and `torque_fused` for none. During the blocked window (10 to 14 s), `torque_fused` reads the swinging L2 as planted for 17 ticks, in four runs from 10.30, 11.54, 12.74 and 13.96 s; the other two rows read none. The force is the script's simulated signal with assumed values, not a measurement; the true-load line is its load part alone. The body outline and the hip positions are schematic. No body-filter result is shown for this single run.*

---

## 4. Sweeps: what decides the torque detector

Each sweep varies one quantity from the defaults and scores only the stance flags (no body EKF). It uses 30 trials, with the same trials and random draws for every setting.

**Wrong ticks** below means swinging feet read as planted plus planted feet missed, per trial. For reference, these 30 trials give 358.6 wrong ticks for the position `detector` and 243.9 for the position `phase_ekf` (the figure's dashed lines).

| Model-error std (N) | 0 | 7.5 | 15 | 22.5 | 30 | 45 |
| --- | --- | --- | --- | --- | --- | --- |
| steps with an early torque trigger | 0% | 0% | 2.2% | 8.9% | 15.8% | 25.1% |
| `torque_detector`: wrong ticks | 120.0 | 120.0 | 147.1 | 227.6 | 316.3 | 440.3 |
| `torque_phase_ekf`: wrong ticks | 122.9 | 122.9 | 123.2 | 123.2 | 127.3 | 122.9 |
| `torque_fused`: wrong ticks | 136.8 | 135.9 | 137.8 | 147.1 | 176.6 | 181.3 |

| Load ramp (ms) | 10 | 25 | 50 | 100 | 200 |
| --- | --- | --- | --- | --- | --- |
| `torque_detector`: wrong ticks | 147.1 | 147.1 | 147.1 | 261.4 | 425.5 |
| `torque_phase_ekf`: wrong ticks | 123.2 | 123.2 | 123.2 | 239.8 | 412.0 |
| `torque_fused`: wrong ticks | 134.2 | 135.7 | 137.8 | 164.7 | 219.2 |

| Torque noise (Nm) | 0.01 | 0.1 | 1 | 3 |
| --- | --- | --- | --- | --- |
| steps with an early torque trigger | 2.1% | 2.2% | 7.0% | 87.7% |
| `torque_detector`: wrong ticks | 145.6 | 147.1 | 201.7 | 989.5 |
| `torque_phase_ekf`: wrong ticks | 122.9 | 123.2 | 124.5 | 964.5 |
| `torque_fused`: wrong ticks | 138.0 | 137.8 | 139.8 | 1035.8 |

- **Model error sets the false triggers.** Below about a quarter of the threshold (7.5 N) the raw threshold did not fire early in these 30 trials. At 45 N it fires early on a quarter of the steps. Throughout, `torque_phase_ekf` stays at about 123 wrong ticks.
- **The load ramp sets the delay.** Up to 50 ms the force crosses 30% of the share within one tick (≤ 15 ms), so 10, 25 and 50 ms score the same. At 100 ms and 200 ms it needs 2 and 3 ticks. That is where fusion pays off: once the prior is high, the fused rule passes 0.5 at a force below the threshold, so `torque_fused` misses far fewer ticks (194.6 against 412.0 for `torque_phase_ekf` at 200 ms).
- **Torque noise matters only when large.** Up to 1 Nm, with Daisy's 4.27 N/Nm, the force noise (≤ 4.3 N) stays well below the 30 N threshold, and `torque_phase_ekf` and `torque_fused` stay flat. The raw threshold does not: at 1 Nm the noise adds to the model-error pulses, and early triggers rise from 2.2% to 7.0% of steps. At 3 Nm (12.8 N std), with the 1 kHz signal thresholded unfiltered, nearly every swing crosses it and every row fails. A real detector would low-pass the signal first; this one does not.

---

## 5. What this toy does not show

- **Measured values.** The model error, the load ramp and the torque noise are assumptions, and so is the gait-phase doc's position-error delay. The torque-vs-position comparison holds only for these values.
- **Real leg dynamics.** The model error is one pulse per step, not the output of a leg model with a wrong $`h`$. Spring hysteresis, impacts and the actuators' own torque bandwidth are not modeled.
- **Load sharing.** Every landed foot ramps to a third of the weight, whatever the other feet do. On a real tripod handover, the load moves between groups.
- **Bledt et al.'s filter.** The fusion here is one log-odds sum with a fitted logistic, not their discrete-time momentum observer and Kalman filter, so their 99.3% accuracy and 4-5 ms delay (ref. 5) say nothing about this script.
- **Signal filtering.** The detector thresholds the raw 1 kHz signal. A low-pass filter would likely trade the large-noise failure for more delay (not simulated).

---

## 6. Takeaway

On a Daisy-sized hexapod, at the torque-noise levels assumed here (up to 1 Nm), the sensor's own noise is not the problem: the leg's mechanical advantage shrinks it far below the weight each foot carries. What decides a torque contact detector is how well the leg's own dynamics are modeled (false triggers) and how fast the weight comes onto the foot (delay). The gait-phase filter of the gait-phase doc handles the first on its own. Fusing the gait phase into the detection itself helps with the second, at the cost of trusting the schedule over a foot that is clearly carrying no weight.

---

## 7. References

1. Faigl, J. & Čížek, P. (2019). *Adaptive locomotion control of hexapod walking robot for traversing rough terrains with position feedback only*. Robotics and Autonomous Systems, 116, 136-147. https://comrob.fel.cvut.cz/papers/ras19adaptive.pdf - the position-error detector the gait-phase doc models.
2. HEBI Robotics. *X-Series Actuators datasheet* (October 2021), https://docs.hebi.us/resources/datasheets/X-SeriesDatasheet.pdf - series-elastic; output torque sensed; torque resolution 0.01 Nm; peak torque 20 N-m (X8-9), 38 N-m (X8-16). *Hardware documentation*, https://docs.hebi.us/hardware.html - feedback at up to 1 kHz; torque from the spring's deflection and a calibrated spring constant.
3. HEBI Robotics. *hebi_description* (Apache-2.0), https://github.com/HebiRobotics/hebi_description - Daisy's `.hrdf` files and `config/daisy/gains/hex_config.xml` (body mass 31.00). The CTU Computational Robotics Laboratory's Daisy page, https://comrob.fel.cvut.cz/robots/daisy.html - 18 DOF, X8-9 and X8-16 actuators.
4. Camurri, M., Fallon, M., Bazeille, S., Radulescu, A., Barasuol, V., Caldwell, D. G., & Semini, C. (2017). *Probabilistic Contact Estimation and Impact Detection for State Estimation of Quadruped Robots*. IEEE Robotics and Automation Letters, 2(2), 1023-1030. https://doi.org/10.1109/LRA.2017.2652491 - foot force from joint torques, $`f = -(J^\top)^{-1}(\tau - h)`$; a logistic in the normal force for the contact probability; on the hydraulic HyQ.
5. Bledt, G., Wensing, P. M., Ingersoll, S., & Kim, S. (2018). *Contact Model Fusion for Event-Based Locomotion in Unstructured Terrains*. IEEE ICRA, 4399-4406. https://doi.org/10.1109/ICRA.2018.8460904 - a momentum-observer force estimate fused with contact priors in a Kalman filter, 99.3% accuracy and a 4-5 ms delay on MIT Cheetah 3 (from the abstract; the full text was not read for this doc).
6. RobotsGuide (IEEE Spectrum). *Daisy*. https://robotsguide.com/robots/daisy - weight 18 kg.
