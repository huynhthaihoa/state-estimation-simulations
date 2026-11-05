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

- **Treating the camera pose as the world pose is badly overconfident.** Position NEES ([glossary](../glossary.md#consistency-and-nees)) is about 8600, where 3 is consistent, and the position error is worse than with legs alone (12.2 cm against 8.3 cm). The filter trusts each reading to 5 mm while the camera has drifted centimeters.
- **Estimating the drift fixes it.** With the camera frame's offset as states, position NEES is 3.18, and the position error drops to 3.2 cm, less than half of legs alone. Yaw becomes observable too (yaw error 0.25° against 0.74°).
- **The camera velocity adds little and is slightly overconfident.** The velocity carries the camera's 1% scale error, which the filter does not model (position NEES 6.71). Without the scale error it is consistent (2.84).
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

- The camera reading itself is 7.4 cm off at the end (scale error -0.92%, yaw drift and the jump); `cam_pose` follows it and steps up with the jump.
- The gate rejects 74% of the camera readings after the jump and uses none from the jump until 46.8 s, 2.1 s after the dropout ends. That is not a lockout as defined in section 4 (under 5% used; it used 26%), and the NEES stays at 424.
- The error ordering of `cam_pose` against `legs` depends on the seed: in seeds 0, 2 and 3 its mean error is above that of `legs`, in seed 1 below (4.4 against 6.3 cm). The NEES result does not: `cam_pose` 1300-47000 and `cam_pose_drift` 1.1-3.5 in seeds 0-3.
- In the NEES panel the grey band, 0.22-9.35, holds 95% of the NEES values of a consistent filter at one instant (the chi-square 2.5% and 97.5% points for 3 degrees of freedom); the 30-trial band above is for an average.

![Animation of one 60 s tripod-gait run with a drifting tracking camera, one pose jump and one dropout: left, a top-down window following the hexapod, with its true stance feet, the camera's reported position and heading drifting away and stepping at the jump, and the estimated body outlines of the legs-only filter, the camera pose as world pose, the same with a gate, and the camera pose with drift states; right, each filter's position error, its position NEES on a log scale with the consistent value 3 marked, and a strip of when the camera reading was available and used, with the jump and the dropout marked](../../assets/hexapod_camera_fusion_inekf.gif)

*Animation: `uv run python generator/animate_hexapod_camera_fusion_inekf.py --out assets/hexapod_camera_fusion_inekf.gif` (seed 2 by default). Swing legs are drawn at fixed default positions relative to the body; the script does not simulate swing feet.*

### Scope
The contact doc's 3D walking body, IMU, stance feet and right-invariant EKF ([IEKF](../glossary.md#iekf-invariant-ekf)), unchanged: isotropic foot-position noise, no IMU biases, the true stance schedule, tripod gait. The body walks faster and straighter than in the contact doc (0.2 m/s, no mean turn, 60 s, about 12 m) so the camera's drift has room to grow. The camera's numbers are T265-style assumptions, not a specific robot's; no published geometry was found for the DARPA hexapod or SCARAB, and the question does not depend on it.

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

**Notation.** $`R`$, $`v`$, $`p`$ are the body's true rotation, world velocity and world position; a hat ($`\hat R`$, $`\hat v`$, $`\hat p`$) marks the filter's estimate. $`y_v`$, $`y_p`$, $`y_R`$ are the camera's velocity, position and rotation readings, and $`n_v`$, $`n_p`$, $`n_R`$ their white noises. $`\mathrm{Exp}`$ turns a rotation vector into a rotation matrix and $`\log`$ is its inverse ([glossary](../glossary.md#exp-and-log-maps)). The **residual** is the difference between the reading and its prediction, and $`H`$ is its Jacobian: how the residual changes with the filter's error ([glossary](../glossary.md#residual-error)).

- **`cam_vel`** uses $`y_v = R^\top v + n_v`$. For a right-invariant EKF this is an **invariant observation** ([glossary](../glossary.md#invariant-observation); the same kind as the leg readings): the residual $`\hat R y_v - \hat v`$ has the constant Jacobian $`[0, -I, 0]`$ on (rotation, velocity, position).
- **`cam_pose`** uses $`y_p = p + n_p`$ and $`y_R = \mathrm{Exp}(n_R) R`$. The rotation residual $`\log(\hat R y_R^\top)`$ has the Jacobian $`I`$ on the rotation error. The position residual $`\hat p - y_p`$ has $`-[\hat p]_\times`$ on rotation and $`I`$ on position, so it depends on the estimate. Here $`[\cdot]_\times`$ is the cross-product matrix.
- **`cam_pose_drift`** adds two **offset states**, the camera frame's position offset $`b_p`$ (3 numbers) and yaw offset $`b_\psi`$, so that $`y_p = p + b_p + n_p`$ and $`y_R = R_z(b_\psi)\,\mathrm{Exp}(n_R) R`$, where $`R_z(b_\psi)`$ is a rotation by $`b_\psi`$ about the vertical. Both are random walks whose variance grows with the distance walked (estimated speed × time step):
  - for the yaw offset, by the truth's 0.005 rad per $`\sqrt{\text{m}}`$;
  - for the position offset, by $`1.2 \times 10^{-3}`$ m² per m. That is chosen so that its std after the run's 12 m equals 1% of 12 m. The truth's scale error grows in proportion to the distance, not as a random walk, so this matches it only at the end of the run.

**The gate** (`gated`, [glossary](../glossary.md#gating-chi-square-gate)): a camera reading is used only if its **normalized innovation squared** (the residual weighed by its predicted covariance) is below the chi-squared 99% bound for its dimension: 11.34 for velocity (3), 16.81 for pose (6). Leg readings are never gated here.

**Scores**:
- **Position NEES:** physical position error against the position covariance mapped to world coordinates. Its consistent value is 3, and over 30 [Monte Carlo trials](../glossary.md#monte-carlo-trials) a consistent run average falls in [2.19, 3.94] 95% of the time.
- **Yaw NEES:** consistent value 1, band [0.56, 1.57].
- **Velocity NEES**, and position and yaw RMSE ([glossary](../glossary.md#rmse-root-mean-square-error)).

All are averaged after the first second.

**Checked** (`tests/use_numpy/test_hexapod_camera_fusion_inekf.py`, 9 tests):
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
With the offsets as states, the camera's pose still fixes what the legs cannot, the yaw (0.25°) and the short-term position. The filter's uncertainty now grows as the camera's frame may have drifted. Position error ends at 5.1 cm against 11.8 cm for legs alone, and every NEES is in its band.

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
- **`cam_pose` without a gate follows the jump** (its error curve steps up at the jumps in the figure; 24.6 cm at the end). With the gate it uses only 10% of the readings after the jump and is locked out in 23 of 30 runs. Its error is then lower (13.0 cm at the end), but it remains overconfident (NEES 1196), as it already was without any jump in experiment 1.
- **`cam_pose_drift` is consistent with or without the gate** (2.95, 2.98), with the same error (4.47 cm at the end, against 21.8 cm for legs alone on these draws). The gate rejects 17% of the readings after the jump but locks out no run.
- **The dropout's cost is not measured separately**: every run in this experiment has one.

**A harsher, assumed case.** The T265 doc says its pose jumps leave the rotation unchanged. The script can add a yaw jump anyway (`--jump-yaw-deg`). With a 3° yaw-jump std (`--jump-yaw-deg 3`, same seeds), the result changes: `cam_pose_drift` without a gate becomes overconfident (position NEES 5.31), and with the gate it is consistent (3.13) but locked out in 12 of 30 runs. In that case the gate matters, and so does its lockout.

---

## 5. What this toy does not show

- **Real T265 numbers.** The drift model (one zero-mean scale error plus a yaw random walk), the noise levels and the jump sizes are assumptions. Only the 0.7-2.0% drift range comes from a measurement (ref. 3), and that was not on a walking robot.
- **Correlated VIO errors.** A real VIO's errors are smooth over time, not white per reading. The camera is also read at the filter's 50 Hz, not at 200 Hz. White readings this fast make the filter more confident than real ones would justify.
- **Vibration and delay.** Walking vibration (ref. 4) and the camera's latency are not modeled. Pronto (ref. 5) reports its visual odometry arriving 150-300 ms late and replays the filter from the reading's time.
- **The tracker confidence.** Readings with low confidence are not given less weight; dropouts are simply missing readings.
- **A cure for the gate's lockout.** Lockouts happen here for `cam_pose` (23 of 30 runs) and, with an assumed yaw jump, for `cam_pose_drift` (12 of 30). No re-anchoring step (accepting the camera again after a run of rejections, with its offset reset) is tried.
- **Relative-pose fusion.** Pronto fuses visual odometry as relative poses between frames, translation only, with fixed covariances, discarding failed frames (ref. 5). That alternative to the offset states is not run here.

---

## 6. Takeaway

A tracking camera gives a legged robot what its legs cannot: yaw and a position that stays put over short distances. It only helps if the filter knows the camera's frame drifts. Fed as the world pose, its readings make the filter confidently wrong. Fed with the frame's drift as states, they halve the position error and keep the filter consistent, through a 10 cm position jump too. A chi-squared gate adds little in that case; it matters for jumps the offset states do not expect, such as an assumed yaw jump, where it can also lock the camera out.

---

## 7. References

1. CTU Computational Robotics Laboratory, robot pages: https://comrob.fel.cvut.cz/robots/darpa.html and https://comrob.fel.cvut.cz/robots/scarab2.html - both robots carry an Intel RealSense T265 "localization camera".
2. Intel RealSense. *librealsense* v2.53.1: `doc/t265.md`, https://github.com/IntelRealSense/librealsense/blob/v2.53.1/doc/t265.md (200 Hz poses; pose jumping, relocalization; blinded by light, no tracking in darkness), and `include/librealsense2/h/rs_types.h` (`rs2_pose`: translation, velocity, acceleration, rotation, angular velocity and acceleration, tracker and mapper confidence 0-3).
3. Kim, J., Song, M., Lee, Y., Jung, M., & Kim, P. (2022). *An Empirical Evaluation of Four Off-the-Shelf Proprietary Visual-Inertial Odometry Systems*. arXiv:2207.06780, https://arxiv.org/abs/2207.06780 - Table I: T265 final drift error per sequence; notes its "scale inconsistency problem".
4. PX4 User Guide (v1.14). *Intel RealSense T265 Tracking Camera (VIO)*. https://docs.px4.io/v1.14/en/peripherals/camera_t265_vio.html - "The camera is sensitive to high-frequency vibrations!"
5. Camurri, M., Ramezani, M., Nobili, S., & Fallon, M. (2020). *Pronto: A Multi-Sensor State Estimator for Legged Robots in Real-World Scenarios*. Frontiers in Robotics and AI, 7, 68. https://doi.org/10.3389/frobt.2020.00068 - visual odometry fused loosely as relative pose measurements (translation only, fixed covariance, failures discarded, 150-300 ms delay).
