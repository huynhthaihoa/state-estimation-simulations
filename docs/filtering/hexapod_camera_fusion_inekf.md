# Legs plus a tracking camera: how to use a pose that drifts

## Intuition

[`hexapod_contact_inekf.md`](hexapod_contact_inekf.md) (**the contact doc** below) tracks a walking hexapod from an IMU and its stance feet. That combination cannot observe the robot's absolute position or its **yaw** (heading about gravity): both drift without bound.

A **tracking camera** ([glossary](../glossary.md#tracking-camera)) such as the Intel RealSense T265, which the CTU Computational Robotics Laboratory mounts on its DARPA SubT hexapod and SCARAB robots (ref. 1), runs its own **visual-inertial odometry** (VIO: position and rotation tracked from camera images and a built-in IMU). It reports a pose and a velocity. Its pose is not the world pose, though. It is measured in the camera's own odometry frame, which starts at the robot's start pose and then drifts away. An independent test measured the T265 ending 0.7-2.0% of the distance walked or driven away from where it started (ref. 3).

So the question is how to use a pose that is accurate locally but drifts. This doc compares three ways on the same data:

| Filter | Camera reading used | Camera drift |
| --- | --- | --- |
| `legs` | none (the contact doc's filter) | - |
| `cam_vel` | velocity, in the body frame | ignored |
| `cam_pose` | position and rotation, as if they were the world pose | ignored |
| `cam_pose_drift` | position and rotation | **estimated**: the camera frame's position offset and yaw offset as filter states |

It then adds the failures the T265's documentation warns about (ref. 2): a sudden **pose jump** when the camera finds its pose inconsistent with an earlier one, and a **dropout** when it loses tracking. Each camera filter runs with and without a **gate**, a test that rejects a reading too far from the filter's prediction.

**Short answer.**

- **Treating the camera pose as the world pose is badly overconfident.** Position NEES ([glossary](../glossary.md#consistency-and-nees)) is about 8600, where 3 is consistent, and the position error is worse than with legs alone (12.2 cm against 8.3 cm). The filter trusts each reading to 5 mm, while the camera has drifted centimeters.
- **Estimating the drift fixes it.** With the camera frame's offset as stated, position NEES is 3.18, and the position error drops to 3.2 cm, less than half of legs alone. Yaw becomes observable too (yaw error 0.25° against 0.74°).
- **The camera velocity adds little and is slightly overconfident.** The velocity carries the camera's 1% scale error, which the filter does not model (position NEES 6.71). Without the scale error, it is consistent (2.84).
- **A late camera reading hurts, and the filter can learn the delay.** At 0.2 m/s, a 30 ms delay shifts the reading's position by 6.5 mm, as much as its 5 mm noise, and a 100 ms delay by 21.5 mm. Ignoring a 100 ms delay raises position NEES from 1.86 to 7.11 (straight walk) and, on a turning walk, yaw NEES from 1.18 to 51. A filter that is told the delay is consistent through 200 ms, and one that estimates it as a state, is within 5 ms of it in under 1 s and within 0.3 ms by the end of 20 s (§5; the delay values are a sweep, not a measured T265 latency).
- **A position jump does not hurt the drift-aware filter, and gating barely matters for it.** With one 10 cm pose jump and one 3 s dropout per run, `cam_pose_drift` stays consistent with or without a gate (position NEES 2.95 and 2.98) at 3.0 cm error. The gate rejects 17% of the readings after the jump but locks the camera out in none of the 30 runs. The filter that treats the camera pose as the world pose is the one in trouble: the gate locks its camera out in 23 of 30 runs, and it stays overconfident (NEES 1196).

![Four panels from hexapod_camera_fusion_inekf.py, 30 trials, 60 s of tripod walking: position RMSE and position NEES over time for legs only, camera velocity, camera pose as world pose and camera pose with drift states; then the same with one pose jump and one dropout per run, for the three camera filters with and without a gate](../../assets/hexapod_camera_fusion_inekf.png)

*Figure: `uv run python use_numpy/hexapod_camera_fusion_inekf.py --out assets/hexapod_camera_fusion_inekf.png` (about 8 minutes).*

The experiment 2 setting, one run, animated: a top-down view of the walking hexapod that follows it, with the camera's reported position and each filter's estimated body outline, then each filter's position error and position NEES over time, and a strip showing when the camera reading was available and when the gated `cam_pose` used it. Seed 2, not the script's seed 0: the seed sets the jump size, and seed 0 draws a 33 cm jump where the jump's std is 0.1 m per axis. Seed 2 is the first seed whose camera draw gives a jump of 8-13 cm (11 cm here, at 15.1 s; dropout 41.7-44.7 s); the rule uses the camera draws only, not the filter results, though the filter results of seeds 0-3 had been looked at before the rule was set. This is one draw, not one of the 30 trials above, so its numbers differ from the tables. Averages after the first second:

| Filter | Position NEES | Position error (cm) | At the end (cm) |
| --- | --- | --- | --- |
| `legs` | 1.05 | 4.22 | 8.71 |
| `cam_pose` | 2921.50 | 6.47 | 7.36 |
| `cam_pose`, gated | 423.76 | 3.08 | 2.97 |
| `cam_pose_drift` | 1.55 | 1.80 | 1.69 |

- The camera reading itself is 7.4 cm off at the end (scale error: -0.92%, yaw drift, and the jump); `cam_pose` follows it and steps up with the jump.
- The gate rejects 74% of the camera readings after the jump and uses none from the jump until 46.8 s, 2.1 s after the dropout ends. That is not a lockout as defined in section 4 (under 5% used; it used 26%), and the NEES stays at 424.
- The error ordering of `cam_pose` against `legs` depends on the seed: in seeds 0, 2, and 3 its mean error is above that of `legs`, in seed 1 below (4.4 against 6.3 cm). The NEES result does not: `cam_pose` 1300-47000 and `cam_pose_drift` 1.1-3.5 in seeds 0-3.
- In the NEES panel, the grey band, 0.22-9.35, holds 95% of the NEES values of a consistent filter at one instant (the chi-square 2.5% and 97.5% points for 3 degrees of freedom); the 30-trial band above is for an average.

![Animation of one 60 s tripod-gait run with a drifting tracking camera, one pose jump and one dropout: left, a top-down window following the hexapod, with its true stance feet, the camera's reported position and heading drifting away and stepping at the jump, and the estimated body outlines of the legs-only filter, the camera pose as world pose, the same with a gate, and the camera pose with drift states; right, each filter's position error, its position NEES on a log scale with the consistent value 3 marked, and a strip of when the camera reading was available and used, with the jump and the dropout marked](../../assets/hexapod_camera_fusion_inekf.gif)

*Animation: `uv run python generator/animate_hexapod_camera_fusion_inekf.py --out assets/hexapod_camera_fusion_inekf.gif` (seed 2 by default). Swing legs are drawn at fixed default positions relative to the body; the script does not simulate swing feet.*

### Scope
The contact doc's 3D walking body, IMU, stance feet, and right-invariant EKF ([IEKF](../glossary.md#iekf-invariant-ekf)), unchanged: isotropic foot-position noise, no IMU biases, the true stance schedule, tripod gait. The body walks faster and straighter than in the contact doc (0.2 m/s, no mean turn, 60 s, about 12 m), so the camera's drift has room to grow. The camera's numbers are T265-style assumptions, not a specific robot's; no published geometry was found for the DARPA hexapod or SCARAB, and the question does not depend on it.

---

## 1. The camera

### 1.1 What a T265 reports
| Fact | Source |
| --- | --- |
| Pose at 200 Hz | ref. 2 (librealsense T265 doc) |
| Each pose sample holds translation, velocity, acceleration, rotation, angular velocity and acceleration, and a **tracker confidence** from 0 (failed) to 3 (high) | ref. 2 (`rs2_pose`) |
| Its pose may "discontinuously jump" when it finds it is inconsistent with an earlier one (pose jumping, an option); this "will only affect the translation and not the rotation" | ref. 2 |
| It is blinded by light shining into it and cannot track in darkness | ref. 2 |
| It is "sensitive to high-frequency vibrations" and should be soft-mounted | ref. 4 (PX4 docs) |
| Final drift 0.61-9.01 m over 84-514 m (0.7-2.0%), mostly on paths that return to the start; its path is "smaller than the actual movements" ("scale inconsistency"); it failed on a 3 km drive | ref. 3 (Table I, §IV) |

### 1.2 The simulated camera
The readings come every filter tick (50 Hz), in a frame that starts at the robot's start pose. All values below are this doc's assumptions, chosen to match the facts above where there are any:

| Part | Model | Default |
| --- | --- | --- |
| Scale error | the camera's displacement from the start is $`(1 + s)`$ times the true one; $`s`$ is drawn once per run | std 1% (assumed; ref. 3's 0.7-2.0% is final drift, and its T265 scale was biased small, not zero-mean) |
| Yaw drift | the camera frame turns about the start position by a random walk that grows with the distance walked | 0.005 rad per $`\sqrt{\text{m}}`$ |
| Pose noise | white, per axis | 5 mm, 0.005 rad |
| Velocity | $`(1 + s) R^\top v`$ in the body frame (with $`R`$ the body rotation and $`v`$ its world velocity), plus white noise | 0.02 m/s |
| Pose jump (experiment 2) | at a random time in 12-36 s the camera frame shifts and stays shifted | std 0.1 m per horizontal axis; no rotation (ref. 2) |
| Dropout (experiment 2) | no readings for 3 s, starting at a random time in 36-48 s | - |

The vibration of walking would show up as larger noise; no published size was found, so it is not modeled separately.

---

## 2. Filters

All four are the contact doc's right-invariant EKF on the same legs and IMU data. The camera filters add a camera update after each tick's leg update; `cam_pose_drift` also carries the offset states below through the prediction and the leg update.

**Notation.** $`R`$, $`v`$, $`p`$ are the body's true rotation, world velocity, and world position; a hat ($`\hat R`$, $`\hat v`$, $`\hat p`$) marks the filter's estimate. $`y_v`$, $`y_p`$, $`y_R`$ are the camera's velocity, position and rotation readings, and $`n_v`$, $`n_p`$, $`n_R`$ their white noises. $`\mathrm{Exp}`$ turns a rotation vector into a rotation matrix and $`\log`$ is its inverse ([glossary](../glossary.md#exp-and-log-maps)). The **residual** is the difference between the reading and its prediction, and $`H`$ is its Jacobian: how the residual changes with the filter's error ([glossary](../glossary.md#residual-error)).

- **`cam_vel`** uses $`y_v = R^\top v + n_v`$. For a right-invariant EKF, this is an **invariant observation** ([glossary](../glossary.md#invariant-observation); the same kind as the leg readings): the residual $`\hat R y_v - \hat v`$ has the constant Jacobian $`[0, -I, 0]`$ on (rotation, velocity, position).
- **`cam_pose`** uses $`y_p = p + n_p`$ and $`y_R = \mathrm{Exp}(n_R) R`$. The rotation residual $`\log(\hat R y_R^\top)`$ has the Jacobian $`I`$ on the rotation error. The position residual $`\hat p - y_p`$ has $`-[\hat p]_\times`$ on rotation and $`I`$ on position, so it depends on the estimate. Here $`[\cdot]_\times`$ is the cross-product matrix.
- **`cam_pose_drift`** adds two **offset states**, the camera frame's position offset $`b_p`$ (3 numbers) and yaw offset $`b_\psi`$, so that $`y_p = p + b_p + n_p`$ and $`y_R = R_z(b_\psi)\,\mathrm{Exp}(n_R) R`$, where $`R_z(b_\psi)`$ is a rotation by $`b_\psi`$ about the vertical. Both are random walks whose variance grows with the distance walked (estimated speed × time step):
  - for the yaw offset, by the truth's 0.005 rad per $`\sqrt{\text{m}}`$;
  - for the position offset, by $`1.2 \times 10^{-3}`$ m² per m. That is chosen so that its std after the run's 12 m equals 1% of 12 m. The truth's scale error grows in proportion to the distance, not as a random walk, so this matches it only at the end of the run.

**The gate** (`gated`, [glossary](../glossary.md#gating-chi-square-gate)): a camera reading is used only if its **normalized innovation squared** (the residual weighted by its predicted covariance) is below the chi-squared 99% bound for its dimension: 11.34 for velocity (3), 16.81 for pose (6). Leg readings are never gated here.

**Scores**:
- **Position NEES:** physical position error against the position covariance mapped to world coordinates. Its consistent value is 3, and over 30 [Monte Carlo trials](../glossary.md#monte-carlo-trials), a consistent run average falls in [2.19, 3.94] 95% of the time.
- **Yaw NEES:** consistent value 1, band [0.56, 1.57].
- **Velocity NEES**, and position and yaw RMSE ([glossary](../glossary.md#rmse-root-mean-square-error)).

All are averaged after the first second.

**Checked** (`tests/use_numpy/test_hexapod_camera_fusion_inekf.py`, 9 tests here; the 15 delay tests are in §5.2):
- the camera draws leave the legs and IMU data unchanged;
- a camera without noise or drift reads the truth;
- the jump persists, and the dropout lasts 3 s;
- each filter's camera residual matches $`H`$ times a small error;
- the velocity Jacobian is constant;
- the offset variance grows with the distance;
- the gate rejects a 1 m jump.

---

## 3. Experiment 1: drift and noise only (30 trials, 60 s)

| Filter | Position NEES | Yaw NEES | Velocity NEES | Position RMSE (cm) | Position RMSE at the end (cm) | Yaw RMSE (°) |
| --- | --- | --- | --- | --- | --- | --- |
| `legs` | 2.93 | 0.40 | 2.78 | 8.30 | 11.83 | 0.74 |
| `cam_vel` | 6.71 | 0.40 | 2.76 | 8.68 | 12.53 | 0.74 |
| `cam_pose` | 8643.64 | 427.17 | 9.24 | 12.22 | 19.98 | 0.83 |
| `cam_pose_drift` | 3.18 | 1.09 | 2.94 | 3.21 | 5.13 | 0.25 |

#### Legs alone: drift from the first yaw error
Without the camera, yaw is never corrected, so the initial yaw error stays and turns into a sideways position error that grows with the distance walked. The yaw NEES of 0.40, below the band, comes from the draws: this experiment's 30 initial yaw errors happen to be small (RMS 0.77°, against the 1.15° the filter assumes; experiment 2's are 1.25°).

#### Camera pose as world pose: confidently wrong
Each reading claims 5 mm, so the filter's position uncertainty stays at a few millimeters while the camera's frame drifts by centimeters. NEES reaches thousands, and the position error ends worse than with legs alone (20.0 against 11.8 cm), because the filter follows the camera's drift.

#### Drift as states: consistent and twice as accurate
With the offsets as states, the camera's pose still fixes what the legs cannot: the yaw (0.25°) and the short-term position. The filter's uncertainty now grows as the camera's frame may have drifted. Position error ends at 5.1 cm against 11.8 cm for legs alone, and every NEES is in its band.

#### Camera velocity: no help, slightly overconfident
The velocity cannot fix yaw or position, and it carries the 1% scale error. With 50 velocity readings a second, the filter believes it knows the velocity better than the scale error allows, and position NEES rises to 6.71. A rerun without the scale error (`--scale-std 0 --n-mc-trials 10`) gives a consistent 2.84 and a slightly smaller error than legs alone (6.5 against 6.8 cm).

---

## 4. Experiment 2: one pose jump and one dropout per run

The jump shifts the camera's position only, as the T265 doc says its pose jumps do (ref. 2). Same seed scheme, different draws from experiment 1.

| Filter | Gate | Position NEES | Yaw NEES | Position RMSE (cm) | At the end (cm) | Camera readings used, before / after the jump | Runs locked out |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `legs` | - | 3.75 | 1.22 | 14.63 | 21.76 | - | - |
| `cam_vel` | no | 6.61 | 1.24 | 14.80 | 22.24 | 100% / 100% | 0 |
| `cam_vel` | yes | 6.45 | 1.24 | 14.80 | 22.27 | 99% / 99% | 0 |
| `cam_pose` | no | 14022.73 | 222.14 | 16.18 | 24.60 | 100% / 100% | 0 |
| `cam_pose` | yes | 1196.36 | 71.80 | 8.38 | 13.00 | 98% / 10% | 23 of 30 |
| `cam_pose_drift` | no | 2.95 | 0.88 | 2.99 | 4.47 | 100% / 100% | 0 |
| `cam_pose_drift` | yes | 2.98 | 0.86 | 2.99 | 4.47 | 100% / 83% | 0 |

"Readings used" counts only ticks outside the dropout. A run is **locked out** when it uses under 5% of the readings after the jump.

- **`cam_vel`: the gate changes almost nothing.** A pose jump shifts the camera frame, not its velocity; the gate rejects about 1% of readings both before and after the jump, the rate expected of a 99% bound, and NEES moves from 6.61 to 6.45.
- **`cam_pose` without a gate follows the jump** (its error curve steps up at the jumps in the figure; 24.6 cm at the end). With the gate, it uses only 10% of the readings after the jump and is locked out in 23 of 30 runs. Its error is then lower (13.0 cm at the end), but it remains overconfident (NEES 1196), as it already was without any jump in experiment 1.
- **`cam_pose_drift` is consistent with or without the gate** (2.95, 2.98), with the same error (4.47 cm at the end, against 21.8 cm for legs alone on these draws). The gate rejects 17% of the readings after the jump but locks out no run.
- **The dropout's cost is not measured separately**: every run in this experiment has one.

**A harsher, assumed case.** The T265 doc says its pose jumps leave the rotation unchanged. The script can add a yaw jump anyway (`--jump-yaw-deg`). With a 3° yaw-jump std (`--jump-yaw-deg 3`, same seeds), the result changes: `cam_pose_drift` without a gate becomes overconfident (position NEES 5.31), and with the gate it is consistent (3.13) but locked out in 12 of 30 runs. In that case the gate matters, and so does its lockout.

---

## 5. Experiment 3: the camera's readings arrive late

**Intuition.** At 0.2 m/s, a reading that is 30 ms late shows where the robot was 6 mm earlier, as large as the camera's own 5 mm position noise. At 100 ms it is 2 cm, 4 noise units. The gait's roll and pitch bob moves faster still: 30 ms shifts the reading's roll and pitch by 0.9°, three times the camera's 0.29° (0.005 rad) rotation noise. A filter that takes the reading as current is then wrong and sure of itself. This experiment asks what that costs and whether the filter can learn the delay.

### 5.1 The delay and the three filters

- **Delay $`d`$:** the time between the camera sampling the body and the filter receiving the reading. It is constant in a run. The reading at tick $`k`$ describes the body at time $`t_k - d`$ (position, rotation and velocity, all in the camera's drifting frame as before). Between two 20 ms ticks the body follows the IMU model with constant inputs per tick, which is how the truth is built. The first readings, from before the start, do not exist.
- **The delay value is a sweep, not a T265 value.** No measured T265 pose latency was found. Intel states a design target for the T265 of "latency under 6ms" (ref. 8; a target, for head tracking, not a measurement). Qin and Shen write that a camera-IMU time offset "usually ranges from several milliseconds to hundreds of milliseconds" (ref. 7). Pronto reports 150-300 ms for its own visual odometry, not for a T265 (ref. 5). The sweep 0, 10, 30, 100, 200 ms covers these ranges; it is an assumption.
- **Setup.** `cam_pose_drift` (§2) with no jump, dropout, or gate; 30 trials; 20 s of tripod walking (not 60 s, to keep the run near 9 minutes). Two walks: a **straight walk** (0.2 m/s, as above) and a **turning walk** (initial turn rate 0.2 rad/s, the contact doc's default; the turn rate then random-walks as in the contact doc). With no delay, the straight 60 s walk reproduces experiment 1's `cam_pose_drift` line exactly (`--delay-experiment --delay-duration 60 --delays-ms 0`: position NEES 3.18, yaw NEES 1.09, position RMSE 3.21 cm, yaw RMSE 0.25°).

| Filter | How the reading is compared | Delay |
| --- | --- | --- |
| `ignore` | with the current estimate (§2) | not modeled |
| `known` | with the estimate $`d`$ ago | given to the filter |
| `estimated` | with the estimate $`\hat d`$ ago | one more state, $`\hat d`$, starting at 0 with std 100 ms (a design choice: of the order of the sweep) |

**How much the delay moves the reading** (RMS over the run, same noise draws, against the no-delay reading; the camera's own noise is 5 mm and 0.29°):

| Delay (ms) | Position (mm, straight walk) | Roll and pitch (°, straight walk) | Yaw, straight walk (°) | Yaw, turning walk (°) |
| --- | --- | --- | --- | --- |
| 10 | 2.2 | 0.30 | 0.031 | 0.114 |
| 30 | 6.5 | 0.89 | 0.090 | 0.343 |
| 100 | 21.5 | 2.86 | 0.291 | 1.150 |
| 200 | 42.3 | 4.96 | 0.561 | 2.358 |

On the straight walk, the yaw part is below the camera's noise up to 30 ms and equal to it at 100 ms; the roll and pitch part is above it from 30 ms on. The roll and pitch bob (0.05 rad amplitude) is an assumption of the contact doc's toy body, so the size of this part is too.

### 5.2 How `known` and `estimated` work

- **Rewind.** The estimate $`d`$ ago is the current estimate run backwards through the stored IMU readings, the exact inverse of the propagation. If $`d`$ is not a whole number of ticks, the last part of a tick is run forward with that tick's IMU reading. The residual is the camera filter's residual (§2), computed with that rewound pose.
- **Jacobian.** The residual's Jacobian on the current error is its Jacobian at the rewound time times a 9×9 map from the current error to the error $`d`$ ago. The contact filter's error transition does not depend on the state, so the map is exact: it only involves gravity and the elapsed time.
- **Estimated delay.** The column of the Jacobian for $`\hat d`$ is minus the rate at which the predicted reading changes: for rotation, minus the rewound rotation times the gyro reading; for position, minus the rewound velocity. The estimate may go negative; the filter then runs the current estimate forward with the latest IMU reading. It is clipped to -0.2-1 s.
- **Not modeled** in `known`: the process noise accumulated over $`d`$ (from the IMU noise, 0.37 mm and 0.018° over 200 ms, against the camera's 5 mm and 0.29°; a side check, not in the script) and the leg updates since the reading was taken, which a full replay (as in Pronto, ref. 5) would redo.
- **Li and Mourikis (ref. 6).** Their delay $`t_d`$ is defined so that an image with timestamp $`t`$ was captured at $`t + t_d`$ (ref. 6, §3), so a late camera has $`t_d = -d`$. They compare the reading with the state estimated at $`t + \hat t_d`$ (ref. 6, eqs. 21-24) and put $`t_d`$ in the EKF state with Jacobian $`H_\theta \hat\omega + H_p \hat v`$ (eq. 26): the rotation and position Jacobians times the angular and linear velocity, which is the rate of change of the predicted reading. Ours is the same rate with the opposite sign ($`t_d = -d`$). Two differences: they propagate the filter to $`t + \hat t_d`$, so with a late camera their estimate lags (ref. 6, §4.4), while ours rewinds the current estimate only to predict the reading, so the estimate stays current; and their readings are image features with camera-IMU extrinsics, ours a pose with offsets. Their timestamp jitter term (eq. 32) is not modeled here.
- **Against an exact replay.** A reference filter that runs 100 ms (5 ticks) behind, takes each reading at its own tick, and is replayed to the present, has a position RMSE of 1.03 cm averaged over 8 runs of 10 s, `known` 0.89 cm and `ignore` 1.89 cm (a side check with the test file's replay helper, not in the script). Per run, `known` is lower than the replay in 6 of 8 runs and higher in 2 (by up to 0.3 cm): no sign that it is worse.

**Checked** (the 15 new tests in `tests/use_numpy/test_hexapod_camera_fusion_inekf.py`):
- a delayed camera reads the body at the earlier time, also for delays that are not a whole number of ticks;
- a `known` filter at the truth, with the true IMU readings, has a zero residual (10, 30, 100, 200 ms);
- the 9×9 map from the current error to the earlier error matches a propagation of a perturbed state;
- the Jacobian column of $`\hat d`$ matches finite differences (also for a negative $`\hat d`$), and so does the whole Jacobian on the error of rotation, velocity, position, the offsets and $`\hat d`$;
- with no delay, `known` and `estimated` (with a zero prior std on $`\hat d`$) are equal to `ignore`;
- `known` stays close to the replay on one run, and `estimated` recovers a 100 ms delay on a fixed seed.

### 5.3 Results (30 trials, 20 s)

`uv run python use_numpy/hexapod_camera_fusion_inekf.py --delay-experiment --out assets/hexapod_camera_fusion_inekf_delay.png`. Each cell is **NEES / RMSE**, averaged after the first second. Consistent run averages: position NEES in [2.19, 3.94], yaw NEES in [0.56, 1.57]. `known` at 0 ms is `ignore` (the same filter).

**Position** (NEES / RMSE in cm):

| Delay (ms) | Straight: `ignore` | `known` | `estimated` | Turning: `ignore` | `known` | `estimated` |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.86 / 1.21 | 1.86 / 1.21 | 1.86 / 1.22 | 3.04 / 1.51 | 3.04 / 1.51 | 2.95 / 1.49 |
| 10 | 1.76 / 1.20 | 1.78 / 1.21 | 1.77 / 1.21 | 2.96 / 1.50 | 2.88 / 1.48 | 2.90 / 1.49 |
| 30 | 2.24 / 1.31 | 1.77 / 1.22 | 1.73 / 1.21 | 3.33 / 1.58 | 2.72 / 1.47 | 2.70 / 1.46 |
| 100 | 7.11 / 2.21 | 1.76 / 1.27 | 1.70 / 1.25 | 8.91 / 2.54 | 2.55 / 1.46 | 2.60 / 1.47 |
| 200 | 16.77 / 3.44 | 1.65 / 1.27 | 2.18 / 1.39 | 25.01 / 4.50 | 2.50 / 1.51 | 3.32 / 1.67 |

**Yaw** (NEES / RMSE in degrees):

| Delay (ms) | Straight: `ignore` | `known` | `estimated` | Turning: `ignore` | `known` | `estimated` |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.58 / 0.12 | 0.58 / 0.12 | 0.57 / 0.12 | 1.18 / 0.17 | 1.18 / 0.17 | 1.12 / 0.17 |
| 10 | 0.60 / 0.12 | 0.60 / 0.12 | 0.61 / 0.12 | 1.93 / 0.21 | 1.14 / 0.17 | 1.08 / 0.16 |
| 30 | 0.62 / 0.12 | 0.60 / 0.12 | 0.60 / 0.12 | 6.53 / 0.38 | 1.07 / 0.16 | 1.08 / 0.16 |
| 100 | 1.09 / 0.17 | 0.57 / 0.12 | 0.56 / 0.12 | 51.38 / 1.07 | 1.09 / 0.17 | 1.09 / 0.17 |
| 200 | 2.72 / 0.28 | 0.56 / 0.12 | 0.58 / 0.12 | 176.81 / 2.07 | 1.06 / 0.17 | 1.09 / 0.17 |

![Six panels from hexapod_camera_fusion_inekf.py --delay-experiment, 30 trials, 20 s: position RMSE, position NEES and yaw RMSE against the true camera delay for the three filters on the straight and turning walks; the mean estimated delay over the first 5 s for each walk; and the filter std and the error of the delay estimate for four motions](../../assets/hexapod_camera_fusion_inekf_delay.png)

*Figure: `uv run python use_numpy/hexapod_camera_fusion_inekf.py --delay-experiment --out assets/hexapod_camera_fusion_inekf_delay.png` (about 9 minutes). Bottom row: mean estimate over the trials, with the filter's std shaded and the true delay dotted; then §5.5.*

- **Ignoring a delay costs little up to 10 ms and then grows.** On the straight walk, 30 ms leaves position NEES in the consistent band (2.24) with 8% more position error (1.31 against 1.21 cm). At 100 ms, NEES is 7.11, and the error is 2.21 cm; at 200 ms, 16.77 and 3.44 cm, 2.8 times the no-delay error.
- **The turning walk shows the yaw.** A yaw reading that lags by the turn rate times the delay: 0.114° at 10 ms to 2.358° at 200 ms (table above). Yaw NEES is 1.93 at 10 ms (band up to 1.57), 6.53 at 30 ms, and 176.81 at 200 ms, with yaw error 2.07° against 0.17°. On the straight walk, yaw is worse later: NEES 1.09 (inside the band) at 100 ms, 2.72 at 200 ms.
- **`known` removes the effect.** Its position error moves from 1.21 to 1.27 cm over the sweep (straight), its yaw error is not at all (0.12°), and its NEES stays in the no-delay range (1.65-1.78 straight, 2.50-2.88 turning).
- **`estimated` matches `known` up to 100 ms and is slightly worse at 200 ms** (position NEES 2.18 against 1.65 straight, 3.32 against 2.50 turning; after the first 5 s still 2.06 against 1.71 and 3.25 against 2.62), within the consistent band but above `known`. The cause is not isolated.
- **Cost per tick** (straight walk, from the same run): `ignore` 0.55 ms; `known` 0.61 ms (10 ms) to 0.78 ms (200 ms), `estimated` 0.65 to 0.84 ms; the rewind goes back one tick per 20 ms of delay.

### 5.4 The estimated delay

With 30 trials, `estimated` ends within 0.11 ms of the truth on average on both walks, with a scatter of 0.18-0.34 ms:

| True delay (ms) | Straight: final error, mean ± std (ms) | Turning: final error, mean ± std (ms) | Median time to stay within 5 ms (s), straight / turning | Mean estimate peaks at (ms, straight) |
| --- | --- | --- | --- | --- |
| 0 | -0.11 ± 0.19 | 0.00 ± 0.34 | 0.2 / 0.2 | 7 |
| 10 | -0.11 ± 0.19 | -0.00 ± 0.34 | 0.2 / 0.2 | 17 |
| 30 | -0.09 ± 0.18 | 0.01 ± 0.33 | 0.2 / 0.3 | 34 |
| 100 | -0.03 ± 0.24 | 0.05 ± 0.27 | 0.4 / 0.4 | 107 |
| 200 | -0.08 ± 0.27 | -0.06 ± 0.32 | 0.6 / 0.7 | 218 |

The filter's own std at the end is 0.30 ms in all rows, about the error scatter. The estimate overshoots (218 ms at 0.48 s for a true 200 ms) and then settles; the first readings arrive only after $`d`$, so the times include that wait.

**This precision comes from the toy.** The body bob is a deterministic sinusoid of 0.05 rad, the readings are white noise at 50 Hz, and there is no timestamp jitter. A real camera's delay will not be known to 0.3 ms.

### 5.5 Is the delay identifiable?

Li and Mourikis show that their time offset is locally identifiable "except in a small number of degenerate motion cases" (ref. 6, abstract). Their rotation condition (ref. 6, §6.4, Lemma 6.1, eq. 49) fails to identify it when the angular velocity obeys $`\dot\omega = k_2 \times \omega + k_1`$ for constant vectors $`k_1`$, $`k_2`$; they note that this includes no rotation and constant rotational velocity. With the acceleration condition added (ref. 6, §6.5, Lemma 6.2), the most important case is constant accelerometer readings, "moving at constant velocity and rotations occur only about the direction of gravity" among them. Their setting is a camera and an IMU with image features, not a pose with offset states, so this is a hint, not a proof for ours. The check: four motions, `estimated` at a true 100 ms, 10 trials each, 20 s (`--ident-trials`):

| Motion | Final error, RMS (ms) | Filter std at the end (ms) |
| --- | --- | --- |
| the default walk (random-walk speed and turn rate, roll, pitch and height bob) | 0.22 | 0.30 |
| no bob (random walks only) | 25.37 | 13.70 |
| constant velocity, no rotation | 49.37 | 15.81 |
| constant velocity and constant turn rate | 39.33 | 15.60 |

- **The delay is only identified with enough variation in the motion.** Without the body bob, the error is 25 ms; with a constant velocity or a constant turn rate, it is 39-49 ms, the degenerate cases of ref. 6. The default walk is not degenerate here, mainly because of the bob (an assumption of the toy, above); the prior std of 100 ms falls to 0.3 ms with it and to 14-16 ms without it.
- **A constant turn does not help.** A constant turn rate is "constant rotational velocity", which ref. 6 lists as degenerate, and the run agrees (39 ms). The turning walk of §5.3 is identifiable because of the bob and the random-walk turn rate, not because it turns.
- **In the degenerate cases, the filter is overconfident:** its std (14-16 ms) is below its RMS error (25-49 ms).
- **Why the std still falls from 100 ms to about 15 ms is likely** the known start: the camera frame starts at the world pose, and the position offset's prior std is 1 mm, so a 100 ms delay at 0.2 m/s (2 cm) is visible before the offset's variance grows. This was not isolated.

### 5.6 What §5 does not show

- **A real T265 delay.** The sweep is an assumption. The delay is constant, with no timestamp jitter (ref. 6 models it) and no camera-to-IMU clock drift.
- **The other filters and failures with a delay.** `cam_vel` and `cam_pose` are not run with a delay (`cam_vel` adds little, `cam_pose` is already overconfident), nor are the jump, the dropout, and the gate.
- **A full replay for every delay.** The replay reference is one 100 ms comparison on 8 short runs.
- **Longer runs.** The experiment is 20 s, not 60 s.

---

## 6. What this toy does not show

- **Real T265 numbers.** The drift model (one zero-mean scale error plus a yaw random walk), the noise levels, and the jump sizes are assumptions. Only the 0.7-2.0% drift range comes from a measurement (ref. 3), and that was not on a walking robot.
- **Correlated VIO errors.** A real VIO's errors are smooth over time, not white per reading. The camera is also read at the filter's 50 Hz, not at 200 Hz. White readings this fast make the filter more confident than real ones would justify.
- **Vibration and delay.** Walking vibration (ref. 4) is not modeled. The camera's latency is modeled only in experiment 3 (§5): a constant delay, as a sweep, for `cam_pose_drift` alone. Pronto (ref. 5) reports its visual odometry arriving 150-300 ms late and replays the filter from the reading's time.
- **The tracker confidence.** Readings with low confidence are not given less weight; dropouts are simply missing readings.
- **A cure for the gate's lockout.** Lockouts happen here for `cam_pose` (23 of 30 runs) and, with an assumed yaw jump, for `cam_pose_drift` (12 of 30). No re-anchoring step (accepting the camera again after a run of rejections, with its offset reset) is tried.
- **Relative-pose fusion.** Pronto fuses visual odometry as relative poses between frames, translation only, with fixed covariances, discarding failed frames (ref. 5). That alternative to the offset states is not run here.

---

## 7. Takeaway

A tracking camera gives a legged robot what its legs cannot: yaw and a position that stays put over short distances. It only helps if the filter knows the camera's frame drifts. Fed as the world pose, its readings make the filter confidently wrong. Fed with the frame's drift as states, they halve the position error and keep the filter consistent, through a 10 cm position jump too. A late reading is a further mismatch: ignored, a delay of 100 ms raises the position error from 1.21 to 2.21 cm and makes the filter overconfident (straight walk); given the delay, or estimating it as one more state, the filter is back to its no-delay accuracy (in this toy, where the motion varies enough for the delay to be identifiable). A chi-squared gate adds little in that case; it matters for jumps the offset states do not expect, such as an assumed yaw jump, where it can also lock the camera out.

---

## 8. References

1. CTU Computational Robotics Laboratory, robot pages: https://comrob.fel.cvut.cz/robots/darpa.html and https://comrob.fel.cvut.cz/robots/scarab2.html - both robots carry an Intel RealSense T265 "localization camera".
2. Intel RealSense. *librealsense* v2.53.1: `doc/t265.md`, https://github.com/IntelRealSense/librealsense/blob/v2.53.1/doc/t265.md (200 Hz poses; pose jumping, relocalization; blinded by light, no tracking in darkness), and `include/librealsense2/h/rs_types.h` (`rs2_pose`: translation, velocity, acceleration, rotation, angular velocity and acceleration, tracker and mapper confidence 0-3).
3. Kim, J., Song, M., Lee, Y., Jung, M., & Kim, P. (2022). *An Empirical Evaluation of Four Off-the-Shelf Proprietary Visual-Inertial Odometry Systems*. arXiv:2207.06780, https://arxiv.org/abs/2207.06780 - Table I: T265 final drift error per sequence; notes its "scale inconsistency problem".
4. PX4 User Guide (v1.14). *Intel RealSense T265 Tracking Camera (VIO)*. https://docs.px4.io/v1.14/en/peripherals/camera_t265_vio.html - "The camera is sensitive to high-frequency vibrations!"
5. Camurri, M., Ramezani, M., Nobili, S., & Fallon, M. (2020). *Pronto: A Multi-Sensor State Estimator for Legged Robots in Real-World Scenarios*. Frontiers in Robotics and AI, 7, 68. https://doi.org/10.3389/frobt.2020.00068 - visual odometry fused loosely as relative pose measurements (translation only, fixed covariance, failures discarded, 150-300 ms delay).
6. Li, M., & Mourikis, A. I. (2014). *Online temporal calibration for camera-IMU systems: Theory and algorithms*. The International Journal of Robotics Research, 33(7), 947-964. https://doi.org/10.1177/0278364913515286 - §3 (time-offset definition: an image with timestamp $`t`$ was captured at $`t + t_d`$), §4.3 eqs. 21-26 (residual at $`t + \hat t_d`$, Jacobian with respect to $`t_d`$) and eq. 32 (timestamp jitter), §4.4 (a delayed camera makes the estimate lag), §6.4 Lemma 6.1 and eq. 49-55 and §6.5 Lemma 6.2 (identifiability and its degenerate motions).
7. Qin, T., & Shen, S. (2018). *Online Temporal Calibration for Monocular Visual-Inertial Systems*. arXiv:1808.00692, https://arxiv.org/abs/1808.00692 - Introduction: "The time offset usually ranges from several milliseconds to hundreds of milliseconds."
8. Intel RealSense. *RealSense tracking camera T265 and depth cameras D400 series - Better together*. https://www.realsenseai.com/?p=4266 - "The tracking camera T265 was designed to provide tracking with latency under 6ms" (a design target in a head-tracking context, not a measurement).
