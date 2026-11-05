# Skid-steer wheel odometry: slip as a filter state

## Intuition

A **skid-steer** robot ([glossary](../glossary.md#skid-steering-and-the-icr-coefficient)) has fixed wheels and turns by driving its left and right wheels at different speeds. Its wheels must skid sideways to do that, so it turns slower than the wheel speeds suggest. [Odometry](../glossary.md#odometry) that assumes ideal wheels then overestimates every turn.

The Computational Robotics Laboratory of the Czech Technical University in Prague (CTU) runs a Clearpath **Husky A200**, a four-wheeled skid-steer robot, with a 128-line LiDAR and an Xsens MTi-30 IMU (refs. 1, 2). This doc asks how much the skid matters for its wheel odometry, and whether a gyro, or estimating the skid online, fixes it.

One established way to model the skid scales the track by an **ICR coefficient** $`\chi \ge 1`$ (refs. 6, 7): the robot turns as if its wheels were $`\chi`$ times farther apart than they are. Husky's own ROS driver multiplies its track by 1.875 (ref. 4), but the file does not say why. This doc compares four ways on the same data:

| Row | Wheels | Gyro | $`\chi`$ used |
| --- | --- | --- | --- |
| `odo_chi1` | yes | - | 1 (ideal differential drive) |
| `odo_driver` | yes | - | 1.875 (the driver's multiplier) |
| `ekf_fixed` | yes | yes | 1.875, fixed |
| `ekf_chi` | yes | yes | **estimated**, as a filter state |

**Short answer.**

- **A wrong $`\chi`$ ruins wheel odometry.** On a 60 m run, ideal differential drive (`odo_chi1`) on a robot whose true $`\chi`$ is 1.875 has a position [RMSE](../glossary.md#rmse-root-mean-square-error) of 5.5 m and a heading RMSE of 102°. The driver's 1.875 on a robot whose true $`\chi`$ is 1.47 gives 10.4 m.
- **With the right $`\chi`$, the wheels are nearly perfect here.** 0.01 cm RMSE. The only wheel error in this model is the encoder's rounding, and a constant, known skid adds nothing.
- **A gyro in an EKF does not rescue a wrong $`\chi`$.** `ekf_fixed` follows the wheels (10.4 m RMSE at a true $`\chi`$ of 1.47) and reports millimeters ([NEES](../glossary.md#consistency-and-nees) 2.4 × 10⁷, where 3 is consistent). In its model the gyro reading is 31 times noisier, in variance, than the wheels' yaw reading (about 108 times per unit of yaw rate, at $`\chi`$ = 1.875), so it trusts the wheels over the gyro.
- **Estimating $`\chi`$ works.** `ekf_chi` learns $`\chi`$ within about 2 s of the first turn (to 0.001 by the end), for every true $`\chi`$ from 1.0 to 2.0. Its position RMSE stays at about 3 cm, made in those first seconds: started with the true $`\chi`$, it drops to 0.05 cm.
- **Real Husky wheel odometry drifts 2.6-4.9%** ([RPE](../glossary.md#rpe-relative-pose-error) over path segments of 1-100 m; measured, ref. 9). This model's odometry with the correct $`\chi`$ drifts almost nothing. So the real drift comes from a $`\chi`$ that is wrong or changes with the path and the ground, or from what the model leaves out (sideways slip, rough ground); ref. 9 does not say which $`\chi`$ its odometry used.

![Four panels from skid_steer_odometry_ekf.py: one run's path for the truth and the four rows; position RMSE over time over 30 trials; the chi estimate of ekf_chi with its reported 2-sigma band; position RMSE of the four rows for true chi of 1.0, 1.25, 1.47, 1.875 and 2.0](../../assets/skid_steer_odometry_ekf.png)

*Figure: `uv run python use_numpy/skid_steer_odometry_ekf.py --out assets/skid_steer_odometry_ekf.png` (about 1.5 minutes).*

One run, animated, at a **true $`\chi`$ of 1.25** (one of experiment 2's sweep values; a chosen value, not experiment 1's 1.875, where the driver's $`\chi`$ is right and `odo_driver` and `ekf_fixed` would be perfect). Seed 0. The left panel draws the whole scene with the Husky to scale (body 0.99 × 0.67 m, wheelbase 0.512 m, track 0.555 m, refs. 3, 4) and each row's estimated Husky; the top right follows the true Husky and shows the measured left and right wheel speeds as arrows (the wheel width drawn, 0.11 m, is illustrative). At the end of this run `odo_chi1`, `odo_driver` and `ekf_fixed` are 11.5, 22.4 and 22.2 m off, and `ekf_fixed` reported a 2σ of at most 17 mm. `ekf_chi` is 5 cm off at the end (largest 10.2 cm); its $`\chi`$ goes from its prior, 1.50 ± 0.60 (2σ), at 8 s, when the first turn starts, to 1.253 ± 0.013 at 10 s and 1.251 ± 0.001 at the end.

![Animation of one run with true chi 1.25: left, a top-down view of the whole scene with the Husky drawn to scale on its true path and the estimated Husky and path of wheels with chi 1, wheels with the driver's chi, the EKF with chi fixed and the EKF with chi as a state; top right, a close-up following the Husky with its four wheels and arrows for the measured left and right wheel speeds; middle right, the chi estimate of ekf_chi with its shaded plus-minus two sigma band against the true chi and the fixed 1.875; bottom right, the position error of the four rows over time on a log scale, with the reported two sigma of ekf_fixed](../../assets/skid_steer_odometry_ekf.gif)

*Animation: `uv run python generator/animate_skid_steer_odometry_ekf.py --out assets/skid_steer_odometry_ekf.gif` (true $`\chi`$ 1.25 by default, `--chi`; seed 0 by default).*

### Scope
A planar robot on flat ground with the Husky's sourced numbers (§1). Not run: the CTU lab's other wheeled robot, the car-like **AgileX Hunter SE**. Its driver gives its geometry (ref. 10), but no source was found for its slip or for its odometry error, and it has no listed IMU.

---

## 1. The robot

### 1.1 What the sources give
| Fact | Value | Source |
| --- | --- | --- |
| Robot and payload at CTU | Husky A200; 128-line LiDAR, Intel NUC, three RGB cameras | ref. 1 |
| Sensors in the lab's Husky paper | Ouster OS0 LiDAR, Xsens MTi-30 IMU, Intel RealSense T265; wheel odometry is not among the listed sensors, and is named as a possible extra loosely-coupled source and for its "slippage" | ref. 2 |
| Track, wheelbase | 550 mm, 512 mm; the manual's kinematics use an "effective track" of 0.555 m | ref. 3 |
| Mass, top speed | 50 kg, 1 m/s | ref. 3 |
| Wheel encoders | 78,000 ticks per meter | ref. 3 |
| Driver track and wheel radius | `track` 0.555 m, `wheel_radius` 0.1651 m (URDF) | ref. 4 |
| Driver odometry | differential drive at 50 Hz (`publish_rate: 50`), track multiplied by `wheel_separation_multiplier: 1.875` | refs. 4, 5 |
| Skid-steer model | $`v = (v_l + v_r)/2`$, $`\omega = (v_r - v_l)/(\chi B)`$, with $`v_l`$, $`v_r`$ the left and right wheel speeds, $`v`$ the forward speed, $`\omega`$ the yaw rate and $`B`$ the track; $`\chi`$ "is equal to 1 when no slippage occurs" | ref. 6 (after ref. 7) |
| Measured $`\chi`$ | 1.41-1.47 on a Pioneer P3-AT (another skid-steer robot), falling as the turn radius grows | ref. 6, Table 3, Eq. 14 |
| Gyro (MTi 10-series) | [noise density](../glossary.md#noise-density) 0.03 °/s/√Hz, bias repeatability 0.2 °/s, in-run bias stability 18 °/h | ref. 8 |
| Husky wheel-odometry error | RPE (average translation error over path segments of 1, 2, 5, 10, 20, 50 and 100 m) 2.63-4.93% over four hardware runs (campus pavement; a park with grass and rough terrain) | ref. 9, Table II |

### 1.2 The simulated Husky
| Part | Model | Source or assumption |
| --- | --- | --- |
| Motion | $`v_{r,l} = v \pm \omega \chi B / 2`$, with $`B = 0.555`$ m | model from ref. 6; $`B`$ from refs. 3, 4 |
| True $`\chi`$ | constant per run; 1.875 in experiment 1, swept 1.0-2.0 in experiment 2 | **assumed**; 1.875 is the driver's multiplier, read as a slip factor (an inference) |
| Path | 8 s straight, a full left loop, 8 s straight, a full right loop; 0.5 m/s, 120 s, 60 m; loop radius 4.0 m | **assumed** (half the top speed, ref. 3) |
| Wheel speeds | each side's distance rounded down to whole encoder ticks (counts), differenced every 20 ms | refs. 3, 4 |
| Gyro | true yaw rate + a constant bias (std 0.2 °/s per run) + white noise (0.21 °/s per 20 ms sample) | ref. 8; **assumed**: the bias is constant, the leaflet's typical bias repeatability (0.2 °/s; max 0.5 °/s) is read as its std, and each sample is the noise averaged over 20 ms (the leaflet's bandwidth is 415 Hz) |

Also **assumed**: no sideways slip (the symmetric case of ref. 6's model), an exact wheel radius, flat ground. The gyro's in-run bias stability (18 °/h) is about 40 times smaller than its turn-on bias, and bias changes during a run are left out.

---

## 2. Filters

**Dead reckoning** (`odo_chi1`, `odo_driver`; [glossary](../glossary.md#dead-reckoning)): the driver's formula, $`v = (v_l + v_r)/2`$ and $`\omega = (v_r - v_l)/(\hat\chi B)`$, integrated step by step with $`\hat\chi`$ = 1 or 1.875. It has no covariance, so it has no NEES.

**EKF** (`ekf_fixed`, `ekf_chi`; [glossary](../glossary.md#ekf-extended-kalman-filter)). The state is the position $`(x, y)`$, the heading $`\theta`$, the yaw rate $`\omega`$, the gyro [bias](../glossary.md#bias) $`b`$ and, in `ekf_chi`, $`\chi`$.

- **Prediction:** the wheels' mean speed moves the position along the heading, and $`\omega`$ turns the heading. $`\omega`$ is a random walk of 0.1 rad/s per √s, a tuning choice that covers how fast the path's yaw rate changes (up to about 0.1 rad/s² in the turn ramps).
- **Two readings every 20 ms:**
  - the gyro, $`z_g = \omega + b`$ + noise;
  - the wheels' yaw reading, $`z_w = (v_r - v_l)/B = \chi\,\omega`$ + noise.
- **Why both parameters can be learned** ([observability](../glossary.md#observability)): on a straight part $`\omega \approx 0`$, so the gyro shows its bias alone. In a turn the wheels' reading is $`\chi`$ times the yaw rate.
- **Noise levels:**
  - gyro 0.0037 rad/s per sample, from the noise density;
  - wheels' yaw reading 0.00067 rad/s, from the encoder step (0.00064 m/s per 20 ms) treated as white **[quantization noise](../glossary.md#quantization-noise)**;
  - in variance, the gyro is 31 times noisier.
- **Priors:**
  - start pose known (std 10⁻⁴ m and rad);
  - yaw rate std 0.1 rad/s;
  - bias std 0.2 °/s;
  - $`\chi`$: mean 1.5, std 0.3 in `ekf_chi`, a design choice that spans 1.0-2.0.

**Scores**:
- **Position RMSE**, the **worst position error** (each run's largest error, RMS over the runs, as a percentage of the 60 m driven), and the heading RMSE.
- **NEES** of $`(x, y, \theta)`$. Its consistent value is 3. Over 30 [Monte Carlo trials](../glossary.md#monte-carlo-trials), a consistent run average falls in [2.19, 3.94] 95% of the time, and over 20 trials in [2.02, 4.17].
- All averages leave out the first second.

**Checked** (`tests/use_numpy/test_skid_steer_odometry_ekf.py`, 7 tests):
- the path turns one full loop each way;
- the wheel speeds follow the skid-steer model, to within the encoder step;
- the encoders count the distance to within one tick;
- dead reckoning with the true $`\chi`$ stays within 1 mm;
- the measurement Jacobian matches finite differences;
- `ekf_chi` learns $`\chi`$ and the bias;
- `ekf_chi` is consistent when the wheel noise really is white.

---

## 3. Experiment 1: slip coefficient 1.875 (30 trials)

| Row | Position RMSE (cm) | Worst error (% of distance) | Heading RMSE (°) | NEES |
| --- | --- | --- | --- | --- |
| `odo_chi1` | 545.54 | 17.06 | 102.28 | - |
| `odo_driver` | 0.01 | 0.00 | 0.00 | - |
| `ekf_fixed` | 0.02 | 0.00 | 0.00 | 0.02 |
| `ekf_chi` | 3.38 | 0.13 | 0.26 | 1.12 |

`ekf_chi` ends with $`\chi`$ = 1.875 (RMSE 0.001, reported std 0.001). Both EKFs end with the bias known to 0.0027 °/s, from a prior of 0.2 °/s.

#### Ideal differential drive: every turn overestimated
With $`\hat\chi = 1`$ the odometry turns 1.875 times too fast. Each 360° loop becomes 675°, so the estimated path draws circles of about half the true radius and leaves each loop pointing the wrong way (figure, top left).

#### `ekf_chi`: one turn to learn $`\chi`$
For the first 8 s the path is straight, so $`\chi`$ stays at its prior (figure, bottom left). Once the first turn starts, it converges within about 2 s. The heading error made in those seconds stays, and driving on with it causes the 3 cm position error. A check over 10 trials confirms this: started with the true $`\chi`$ (`--chi-prior 1.875 --chi-prior-std 1e-4 --n-mc-trials 10`), `ekf_chi`'s position RMSE drops from 4.23 cm (the same 10 trials with the default prior) to 0.05 cm.

#### Both EKFs report too much uncertainty here
NEES is 0.02 for `ekf_fixed` and 1.12 for `ekf_chi`, below the band. The filters treat the encoder rounding as white noise, which would add up over time. Real rounding does not add up. Each reading is the difference of two whole tick counts, so the rounding errors cancel when the readings are summed: the distance is never off by more than one tick.

With the encoder rounding, `ekf_chi`'s NEES is higher than `ekf_fixed`'s because most of its error comes from learning $`\chi`$, which its covariance does cover: started with the true $`\chi`$, its NEES falls to 0.03 (from 1.78 over the same 10 trials).

With white wheel noise of the same size instead (`--white-wheel-noise --n-mc-trials 10`), both are consistent: NEES 2.48 (`ekf_fixed`) and 2.02 (`ekf_chi`), band [1.68, 4.70]; a test checks it for `ekf_chi`.

---

## 4. Experiment 2: sweep of the slip coefficient (20 trials each)

Position RMSE (cm), and NEES for the EKFs:

| True $`\chi`$ | `odo_chi1` | `odo_driver` | `ekf_fixed` (NEES) | `ekf_chi` (NEES) |
| --- | --- | --- | --- | --- |
| 1.0 | 0.01 | 2155.31 | 2144.48 (1.3 × 10⁸) | 3.08 (1.18) |
| 1.25 | 893.11 | 1600.21 | 1591.20 (6.7 × 10⁷) | 2.90 (0.98) |
| 1.47 (P3-AT, ref. 6) | 1161.66 | 1041.26 | 1035.23 (2.4 × 10⁷) | 3.01 (0.89) |
| 1.875 (driver) | 545.54 | 0.01 | 0.02 (0.02) | 2.94 (0.92) |
| 2.0 | 446.46 | 286.32 | 284.96 (1.3 × 10⁶) | 3.24 (1.19) |

- **A fixed $`\chi`$ is only right where it was set.** Both fixed-$`\chi`$ dead-reckoning rows are near-perfect at their own value and meters off elsewhere. The error does not grow steadily with the mismatch: once the heading error passes 180° in a loop, the estimate starts to come back.
- **Where $`\chi`$ is wrong, `ekf_fixed` matches `odo_driver` to within 1%.** The wheels' yaw reading outweighs the gyro, so a wrong $`\chi`$ goes straight into the heading. The constant bias state cannot absorb an error that appears only in turns.
- **`ekf_chi` is the same at every $`\chi`$**: about 3 cm, with $`\chi`$ learned to 0.001. Its NEES (0.89-1.19) is below the band for the same reason as in experiment 1: cautious, not overconfident.

---

## 5. What this toy does not show

- **A $`\chi`$ that changes.** On the P3-AT, $`\chi`$ falls as the turn radius grows (ref. 6). It also depends on the ground. Here it is constant per run, and `ekf_chi` treats it as constant (no process noise).
- **Sideways slip, wheel radius error, rough ground.** All left out (§1.2). With them, even the correct-$`\chi`$ rows would drift. The measured 2.6-4.9% (ref. 9) cannot be split between these effects and a wrong $`\chi`$ from the information given.
- **A wheel reading with slip noise.** `ekf_fixed` trusts the wheels' yaw reading because its noise is only the encoder step. A filter that adds noise for slip would lean more on the gyro; that tuning is not run here.
- **The in-run gyro drift** (18 °/h, ref. 8) and the IMU's other axes.
- **LiDAR.** The lab fuses LiDAR, the IMU and visual odometry in a graph (ref. 2), and LiDAR odometry can also correct the wheels (ref. 9). Neither is run here.

---

## 6. Takeaway

In this model, the slip coefficient, not the sensors' noise, decides the error: with the right $`\chi`$ the wheels are off by 0.01 cm, with a wrong one by meters. A wrong $`\chi`$ throws wheel odometry off by meters within a minute, and an EKF that trusts the wheels' yaw reading keeps that error and reports it as millimeters. With $`\chi`$ as a filter state, one turn with a gyro is enough to learn it, at every slip tried here, and its run-averaged NEES stays at or below 3, so on average it never claims more accuracy than it has. A real Husky adds what this model leaves out, a $`\chi`$ that changes and sideways slip, so a constant $`\chi`$ state is a starting point, not the whole answer.

---

## 7. References

1. Computational Robotics Laboratory, Czech Technical University in Prague. *Husky* and *Hunter SE* robot pages. https://comrob.fel.cvut.cz/robots/husky.html, https://comrob.fel.cvut.cz/robots/hunter.html - Husky A200 "wheeled robot" with a "128-line LiDAR, Intel NUC computer, and three RGB global shutter cameras"; Hunter SE "AgileX car-like drive robot"; the Husky page lists ref. 2.
2. Hulchuk, V., Bayer, J., & Faigl, J. (2023). *Graph-based LiDAR-Inertial SLAM Enhanced by Loosely-Coupled Visual Odometry*. European Conference on Mobile Robots (ECMR), 1-8. https://doi.org/10.1109/ECMR59166.2023.10256360, PDF https://comrob.fel.cvut.cz/papers/ecmr23nav.pdf - "a four-wheeled skid-steered robot Husky", Ouster OS0, "a 9-axis IMU Xsens MTi-30", Intel RealSense T265; wheel odometry is not among the listed sensors, and is named as a possible extra loosely-coupled source and, in the introduction, for its "slippage".
3. Clearpath Robotics. *Husky A200 User Manual*. https://docs.clearpathrobotics.com/docs_robots/outdoor_robots/husky/a200/user_manual_husky - track width 550 mm, wheelbase 512 mm, mass 50 kg, maximum speed 1 m/s, "Wheel Encoders 78,000 ticks/m", kinematics with "W is the effective track of the robot, 0.555 m".
4. Clearpath Robotics. *husky* ROS packages (branch `noetic-devel`). https://github.com/husky/husky - `husky_description/urdf/husky.urdf.xacro` (`track` 0.555, `wheelbase` 0.5120, `wheel_radius` 0.1651); `husky_control/config/control.yaml` (`publish_rate: 50`, `wheel_separation_multiplier: 1.875 # default: 1.0`, under the comment "Wheel separation and radius multipliers").
5. ros-controls. *ros_controllers*, `diff_drive_controller/src/diff_drive_controller.cpp` (branch `noetic-devel`). https://github.com/ros-controls/ros_controllers/blob/noetic-devel/diff_drive_controller/src/diff_drive_controller.cpp - the wheel separation (read from the robot description when `wheel_separation` is not set) is multiplied by `wheel_separation_multiplier`, both for the odometry and for the wheel commands.
6. Wang, T., Wu, Y., Liang, J., Han, C., Chen, J., & Zhao, Q. (2015). *Analysis and Experimental Kinematics of a Skid-Steering Wheeled Robot Based on a Laser Scanner Sensor*. Sensors, 15(5), 9681. https://doi.org/10.3390/s150509681 - the ICR model and coefficient $`\chi`$ (Eq. 15); tread ICRs "always lie outside of the tread centerlines because of slippage"; Table 3: $`\chi`$ from 1.4662 to 1.4115 measured on a Pioneer P3-AT with an external laser scanner, as $`\lambda`$ goes from 0 to 7, where the turn radius is $`\lambda`$ times half the effective track (Eq. 14). The text calls increasing $`\lambda`$ "increasing the path curvature", which conflicts with Eq. 14; this doc follows the equation.
7. Mandow, A., Martínez, J. L., Morales, J., Blanco, J. L., García-Cerezo, A., & González, J. (2007). *Experimental kinematics for wheeled skid-steer mobile robots*. IEEE/RSJ IROS, 1222-1227. https://doi.org/10.1109/IROS.2007.4399139 - origin of the ICR-based skid-steer model that ref. 6 builds on; the full text was not read for this doc (closed access).
8. Xsens. *MTi 10-series* product leaflet (third-party copy; Xsens's own specification page could not be read). https://hkjdwchip.com/image//files/xsenstechnologies-mti202a5g4-datasheets-0241.pdf - gyroscope: noise density 0.03 °/s/√Hz (typical), bias repeatability (1 yr) 0.2 °/s, in-run bias stability 18 °/h.
9. Guadagnino, T., Mersch, B., Vizzo, I., Gupta, S., Malladi, M. V. R., Lobefaro, L., Doisy, G., & Stachniss, C. (2024). *Kinematic-ICP: Enhancing LiDAR Odometry with Kinematic Constraints for Wheeled Mobile Robots Moving on Planar Surfaces*. arXiv:2410.10277v3. https://arxiv.org/abs/2410.10277 - §IV (metrics: the KITTI average translation error over segments of 1-100 m) and Table II, Clearpath Husky A200 with a total-station ground truth: wheel odometry RPE 4.93, 4.71, 2.63 and 2.98% on four runs (Campus 0-2 on pavement, Palace in a park).
10. AgileX Robotics. *hunter_ros*, `hunter_base/include/hunter_base/hunter_params.hpp`. https://github.com/agilexrobotics/hunter_ros - `HunterSEParams`: track 0.52 m, wheelbase 0.55 m, wheel radius 0.1375 m, maximum central steering angle 0.283 rad.
