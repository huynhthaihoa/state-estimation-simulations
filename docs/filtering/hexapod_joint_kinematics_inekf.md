# Joint encoders on an 18-DOF hexapod: when the leg-kinematics noise model matters

## Intuition

[`hexapod_contact_inekf.md`](hexapod_contact_inekf.md) (**the contact doc** below) tracks a hexapod from an IMU (inertial measurement unit) and its stance feet (the feet on the ground). Each stance foot's position in the body frame is a reading with the same small error in every direction. On a real robot that reading comes from the leg's **joint encoders** (the sensors that measure each joint's angle) through the leg's **forward kinematics** (FK: the map from joint angles to the foot's position). So its error is not a fixed sphere:

- **It depends on the pose.** A joint far from the foot moves it more per radian, and the direction of that motion depends on how the leg is bent. The error is stretched along the directions the joints move the foot fastest.
- **It can be a fixed bias.** An encoder whose zero is off by 1° puts every reading of that joint off by 1°. The resulting foot error changes only slowly as the leg moves, so it is not the independent-each-tick noise a Kalman filter expects.

This doc uses the real geometry of the **HEBI Daisy**, a hexapod with 18 degrees of freedom (**DOF**: 6 legs × 3 joints) and one of the robots of the Computational Robotics Laboratory at CTU Prague (ref. 3). It asks two questions:

1. **White encoder noise:** does modelling the pose-dependent shape of the error, instead of a sphere, change the result?
2. **Encoder zero offsets:** what does a fixed offset do to the filter, and what fixes it?

**Short answer.** The shape barely matters. At any plausible encoder noise, a sphere with the right total size is as **consistent** (its claimed uncertainty matches its actual error) as the full model. The full model is only somewhat more accurate once the noise reaches about 5 mrad (0.3°). Offsets matter more: at 2° the filter that ignores them claims about half its actual velocity error variance (velocity NEES, [glossary](../glossary.md#consistency-and-nees): 6.9 where 3 is consistent). Calling the offsets extra white noise overcorrects (NEES 1.7, and less accurate). Estimating the 18 offsets as filter states keeps velocity consistent (NEES 3.1) and gives the lowest velocity error.

![Two panels: a top view of HEBI's Daisy at its nominal stance, six legs drawn joint by joint from the body hexagon to the feet on a 0.55 m radius, each foot with its encoder-noise ellipse drawn 150 times larger; and a side view of leg L2 with arrows showing how far a 1 degree offset on each joint moves the foot: 6.3 mm out of the page for joint 1, 8.3 mm for joint 2, 5.7 mm for joint 3](../../assets/hexapod_joint_kinematics_concept.png)

*Figure: Daisy's legs at the nominal stance, plotted by `uv run python generator/make_figures.py hexapod_joint_kinematics_concept`.*

### Scope
The contact doc's 3D walking body, IMU and filter, unchanged, at Daisy's size: body 0.21 m above the ground, feet landing around a nominal point on a 0.55 m radius. As in the contact doc's terms: the tripod gait (three feet down at a time), the true stance schedule, no IMU biases and no gating (every reading is used). The stance feet's readings come from Daisy's leg kinematics and simulated encoders. Only the right-invariant extended Kalman filter (EKF) is run (the contact doc's `InvariantContactEKF`); the filters below differ only in how they model the foot reading's error. Run with `uv run python use_numpy/hexapod_joint_kinematics_inekf.py` (about 10 minutes).

---

## 1. Model

### 1.1 Daisy's legs
Each leg is a chain of three HEBI actuators (motor modules with built-in sensors) and two links, taken from HEBI's own robot description files (`.hrdf`, ref. 1):

| Part | Value | Source |
| --- | --- | --- |
| Joint 1 | X8-9 actuator, vertical axis | `daisyLeg-Left.hrdf` / `-Right.hrdf` |
| Joint 2 | X8-16 actuator, horizontal axis, after a heavy bracket | same |
| Joint 3 | X8-9 actuator, horizontal axis, after a 0.325 m link | same |
| Foot | after a second 0.325 m link | same |
| Hip mounts | at ±30°, ±90°, ±150° around the body centre, 0.2375 m out (0.1875 m for the middle legs) | `Daisy.hrdf` |
| Nominal stance | feet on a 0.55 m radius, body 0.21 m above them | `hex_config.xml` |

`leg_fk` builds the chain from these parts. It matches HEBI's own robot model (the `hebi-py` 2.16.2 library reading `Daisy.hrdf`, ref. 5) to 8×10⁻⁸ m over 50 random 18-joint configurations; the tests keep six of those reference pairs, so `hebi-py` is not needed to run anything. HEBI publishes no description of the Lily hexapod (the CTU lab's other HEBI hexapod, ref. 3), so this doc uses Daisy only.

The leg's **Jacobian** $`J = \partial\,\mathrm{FK} / \partial q`$ ([glossary](../glossary.md#jacobian)) is the 3×3 matrix of how fast the foot moves per radian of each joint. At the nominal stance the three joints move the foot 5.5-6.3, 7.6-8.3 and 5.7 mm per degree (joints 1, 2, 3; front and rear legs vs middle legs). The truth's joint angles come from **inverse kinematics** (the joint angles that put the foot at a given point) on the **knee-up** branch, the one with the third joint above the body.

### 1.2 Encoder readings
Each tick, each joint of a stance leg reads

```math
\tilde q = q + b + n_q, \qquad n_q \sim \mathcal N(0, \sigma_q^2), \qquad b \sim \mathcal N(0, \sigma_b^2) \text{ once per run}
```

- $`q`$: the true joint angle (rad); $`\tilde q`$: the reading.
- $`n_q`$: **white encoder noise**, drawn fresh every tick, std $`\sigma_q`$.
- $`b`$: the joint's **zero offset**, drawn once per run and then fixed, std $`\sigma_b`$ (0 in §3).

The foot reading is $`\tilde y = \mathrm{FK}(\tilde q)`$. To first order, $`\tilde y \approx \mathrm{FK}(q) + J\,(n_q + b)`$: the white part has covariance $`\sigma_q^2 J J^\top`$, and the offset adds the slowly changing error $`J b`$.

**The numbers are assumptions.** HEBI does not publish the encoder resolution of its X-Series actuators, and no measured value was available for this doc, so $`\sigma_q`$ is swept from 0.5 to 50 mrad (default 1 mrad, 0.057°) and $`\sigma_b`$ from 0 to 2°. At 1 mrad, the foot error's std along its three principal directions is 0.51, 0.31 and 0.21 mm (front and rear legs) or 0.54, 0.36 and 0.19 mm (middle legs): far below the contact doc's 5 mm.

---

## 2. Filters

All filters are the contact doc's right-invariant EKF (the contact-aided invariant EKF of ref. 2) on identical data. The foot reading enters as the residual $`\hat R \tilde y + \hat p - \hat d`$, so its body-frame error covariance $`C`$ enters as $`\hat R C \hat R^\top`$, both in the update and when a new foot is added at touchdown. Here $`\hat R`$ and $`\hat p`$ are the filter's estimated body rotation and position and $`\hat d`$ its estimated foot position (world frame).

| Filter | Foot-reading error it assumes | Used in |
| --- | --- | --- |
| `iso` | a sphere, per-axis variance = the mean of the three eigenvalues of $`\sigma_q^2 J J^\top`$ at the nominal stance (0.37 mm std at 1 mrad): the right total size, no shape | §3 |
| `iso_max` | a sphere sized by the largest eigenvalue (0.52 mm std at 1 mrad): never too small | §3 |
| `joint` | $`\sigma_q^2 J J^\top`$, with $`J`$ at the reading's own angles $`\tilde q`$ | §3, §4 |
| `joint_inflated` | $`(\sigma_q^2 + \sigma_b^2) J J^\top`$: the offset treated as more white noise | §4 |
| `joint_offsets` | $`\sigma_q^2 J J^\top`$, plus the 18 offsets as filter states | §4 |

**The offset states.** `joint_offsets` adds $`\hat b`$ (18 numbers, prior std $`\sigma_b`$, constant) to the state and corrects each reading, $`\tilde y_c = \mathrm{FK}(\tilde q - \hat b)`$. With the offset error $`\zeta = \hat b - b`$, the residual changes by $`-\hat R J \zeta`$, so the update gets the extra Jacobian $`-\hat R J`$ on that leg's three offsets. A new foot's error at touchdown gets the same term. Like the IMU biases in the contact doc's §5, this term depends on the estimate $`\hat R`$, so the filter is an "imperfect" invariant EKF.

**Checked** (`tests/use_numpy/test_hexapod_joint_kinematics_inekf.py`): `leg_fk` against HEBI's model; the analytic Jacobian against finite differences; the inverse kinematics; the offset Jacobian against a finite difference of the residual; with $`\sigma_b = 0`$, `joint_offsets` matches `joint`'s estimates to 1e-10; the offsets change no other random draw; `simulate_feet`'s new `foot_offsets` argument changes nothing at its default. The two hooks added to `hexapod_contact_inekf.py` for this (`foot_offsets`, `kin_noise`) also leave that script's default figure byte-identical (checked once when they were added).

**Scored on what can be known.** Leg odometry plus an IMU cannot observe absolute position or yaw (the contact doc's Intuition), so their errors grow without bound whatever the noise model. The scores are **velocity** and **tilt** (roll and pitch): NEES of the velocity block (consistent value 3) and a tilt NEES, the squared roll and pitch errors each divided by its own reported variance (consistent value 2), and RMSE ([glossary](../glossary.md#rmse-root-mean-square-error)), averaged over the run after the first second. For 30 [Monte Carlo trials](../glossary.md#monte-carlo-trials), a consistent filter's run-average velocity NEES falls in [2.19, 3.94] 95% of the time, and its tilt NEES in [1.35, 2.78].

---

## 3. White encoder noise: the shape barely matters

30 trials per setting, seed 0 + setting index, $`\sigma_b = 0`$:

| $`\sigma_q`$ (mrad) | velocity NEES `iso` / `iso_max` / `joint` | velocity RMSE (mm/s) `iso` / `iso_max` / `joint` |
| --- | --- | --- |
| 0.5 | 2.99 / 2.84 / 3.00 | 4.99 / 5.01 / 4.98 |
| 1 (default) | 2.98 / 2.74 / 2.98 | 5.43 / 5.46 / 5.36 |
| 2 | 2.98 / 2.62 / 2.96 | 5.93 / 5.97 / 5.77 |
| 5 | 3.02 / 2.55 / 3.02 | 7.39 / 7.48 / 7.06 |
| 20 | 2.98 / 2.22 / 2.96 | 12.19 / 12.35 / 11.00 |
| 50 | 3.10 / **2.14** / 3.08 | 20.93 / 21.34 / 18.39 |

- **`iso` and `joint` are equally consistent at every noise level.** A sphere with the right total size claims the right uncertainty for velocity, even though its shape is wrong.
- **`iso_max` drifts toward too cautious** as the noise grows, and leaves the band at 50 mrad (2.14). A sphere sized by the worst direction claims too much error in the other two.
- **`joint` is the most accurate**, but the gain is small until the noise is large: 0.07 mm/s (1%) at 1 mrad, 0.33 mm/s (4.5%) at 5 mrad, 2.5 mm/s (12%) at 50 mrad. Tilt shows the same pattern (RMSE 0.0606° for all three at 1 mrad; 0.0961° vs 0.1013° for `joint` vs `iso` at 50 mrad), with tilt NEES 1.8-2.1 throughout.

Why so little: at 1 mrad the foot error is under 0.6 mm, while the foot itself drifts on the ground by the contact noise (0.01 m/s, so 0.2 mm per tick) and the IMU adds its own. The shape of a sub-millimetre error is not what limits the filter.

---

## 4. Joint zero offsets: the filter that ignores them becomes overconfident

$`\sigma_q`$ = 1 mrad, 30 trials per setting, seed 100 + setting index:

| $`\sigma_b`$ (deg) | velocity NEES `joint` / `joint_inflated` / `joint_offsets` | velocity RMSE (mm/s) `joint` / `joint_inflated` / `joint_offsets` |
| --- | --- | --- |
| 0 | 3.01 / 3.01 / 3.01 | 5.28 / 5.28 / 5.28 |
| 0.5 | 3.23 / **2.17** / 3.01 | 5.37 / 6.87 / 5.22 |
| 1 | **4.29** / **2.00** / 3.04 | 6.18 / 8.48 / 5.39 |
| 2 | **6.93** / **1.68** / 3.09 | 7.85 / 10.65 / 5.63 |

*Bold: outside the 95% band [2.19, 3.94].*

#### Ignoring the offsets: overconfident
`joint` leaves the band from about 1° on. An offset makes each foot's reading drift slightly as its leg moves through the stride. Over the stance that drift looks like the body moving, so the velocity estimate picks up an error that the filter's white-noise model does not account for. At 2° its velocity error variance is about twice what it claims.

#### Calling them white noise: too cautious and less accurate
`joint_inflated` adds the offset's size to every reading's noise. That noise is the right size but the wrong kind: a white error averages out over many readings, a fixed one does not, and the filter now trusts every reading less than it could. So it is too cautious (NEES 1.68 at 2°) and also less accurate (10.65 mm/s, against 7.85 mm/s for ignoring the offsets).

#### Estimating them: consistent velocity, but overconfident offsets
`joint_offsets` stays in the band at every offset size, and has the lowest velocity error at every non-zero offset (5.63 mm/s at 2°, close to the 5.28 mm/s with no offsets). Its tilt is also the most accurate at 2° (RMSE 0.0633° against 0.0721° for `joint`).

It does not fully learn the offsets, though, and is overconfident about them:

| $`\sigma_b`$ (deg) | offset RMSE at the end (deg) | std the filter reports (deg) |
| --- | --- | --- |
| 0.5 | 0.369 | 0.270 |
| 1 | 0.559 | 0.347 |
| 2 | 0.669 | 0.368 |

Each leg's three offsets are learned only from how that leg's foot readings vary while the leg moves through a stride. The leftover error variance is 1.9 to 3.3 times what the filter claims, so the offset estimates should not be read as a calibration. The velocity is still consistent because what matters for it is the foot error $`J(\hat b - b)`$, not each offset separately.

![Four panels from hexapod_joint_kinematics_inekf.py, tripod gait, 30 trials per point: velocity NEES and velocity RMSE against encoder white-noise std from 0.5 to 50 mrad for the isotropic, isotropic-largest-axis and joint models, where isotropic and joint stay at about 3 and joint is the most accurate; and velocity NEES and RMSE against joint zero-offset std from 0 to 2 degrees for the joint model, the joint model with offsets as extra white noise, and the joint model with 18 offset states, where only the offset-state filter stays inside the consistency band and it also has the lowest error](../../assets/hexapod_joint_kinematics_inekf.png)

*Figure: `use_numpy/hexapod_joint_kinematics_inekf.py` at its defaults, rendered by `uv run python use_numpy/hexapod_joint_kinematics_inekf.py --out assets/hexapod_joint_kinematics_inekf.png` (about 10 minutes). The numbers match the script's printout.*

The 2° case, one run, animated. Seed 6, not the script's seed 0: seed 0 gives a milder gap (velocity NEES 4.5 vs 2.8), while seed 6 is close to the 30-trial averages above (6.95 vs 3.13, against 6.93 vs 3.09). Over seeds 0-7 the filter that ignores the offsets runs from 4.2 to 10.7 and the offset-state filter from 2.7 to 3.3. Left: Daisy walking its curved path, each leg drawn joint by joint from `leg_frames` at the true joint angles, and the **foot-reading error** (the reading's world-frame position minus the true foot position, mostly the offsets' effect, 21 mm on average) magnified 10 times. Right: each filter's velocity error along x with its own ±2σ band (the two bands are nearly the same), and its velocity NEES averaged over the run so far. The ignoring filter's x-error RMS is 4.95 mm/s against a reported 1σ of 2.92 mm/s; the offset-state filter's is 2.96 against 2.96.

![Animation of one tripod-gait run of the Daisy hexapod with 2 degree joint zero offsets: left, a top-down view of the walking robot with its legs drawn joint by joint, the true stance feet, the true path and the foot-reading error magnified 10 times; right, the x velocity error of the filter that ignores the offsets and of the filter that estimates them, each with its shaded plus-minus two sigma band, and each filter's velocity NEES averaged over the run so far, about 7 for the ignoring filter and about 3 for the offset-state filter](../../assets/hexapod_joint_kinematics_inekf.gif)

*Animation: `uv run python generator/animate_hexapod_joint_kinematics_inekf.py --out assets/hexapod_joint_kinematics_inekf.gif` (seed 6 by default). Swing legs are drawn at the nominal stance joint angles; the script does not simulate swing feet.*

---

## 5. What this toy does not show

- **Series elasticity.** HEBI's X-Series actuators are **series-elastic** (a spring between motor and output, which lets them sense torque). Under load the spring deflects, which is a load-dependent joint error neither modelled nor estimated here.
- **Real encoder numbers.** $`\sigma_q`$ and $`\sigma_b`$ are swept assumptions. With a measured encoder noise and a calibration record, §3 and §4 would say which regime Daisy is in.
- **Other kinematic errors.** Link lengths, mounting angles and flex are exact here.
- **Other gaits and robots.** Only the tripod gait and Daisy's geometry are run; `--gait` switches gaits. The CTU lab's other 18-DOF hexapods (DARPA Sub-T hexapod with AX12/AX18 servos, SCARAB with XM430 servos, per their pages in ref. 3) are not modelled.
- **Observability of the offsets.** §4 shows the offsets are only partly learned in 12 s of tripod walking. Whether a longer run, other gaits or turning would pin them down is not tested. [`hexapod_redundant_leg_inekf.md`](hexapod_redundant_leg_inekf.md) runs the same filters on a 4-joint leg (CTU's HAntR) and looks at which offset combinations stay unlearned.

---

## 6. Takeaway

For a filter fed by joint encoders, the size of the foot-reading error matters more than its shape, and fixed errors matter more than either. A sphere of the right total size is as consistent as the full $`\sigma_q^2 J J^\top`$ model at any plausible encoder noise; the full model only buys accuracy when the encoders are coarse. A 1-2° zero offset, by contrast, makes the filter overconfident. Inflating the noise to cover it trades that for under-confidence and a larger error, while estimating the offsets keeps velocity consistent and accurate.

---

## 7. References

Refs. 1-3 and 5 are cited in the text; ref. 4 is further reading.

1. HEBI Robotics. *hebi_description* (GitHub repository, Apache-2.0): `config/daisy/hrdf/Daisy.hrdf`, `daisyLeg-Left.hrdf`, `daisyLeg-Right.hrdf` and `config/daisy/gains/hex_config.xml`. https://github.com/HebiRobotics/hebi_description - Daisy's leg chain, hip mounts and nominal stance used here.
2. Hartley, R., Ghaffari, M., Eustice, R. M., & Grizzle, J. W. (2020). *Contact-Aided Invariant Extended Kalman Filtering for Robot State Estimation*. The International Journal of Robotics Research, 39(4), 402-430. https://doi.org/10.1177/0278364919894385 - the contact-aided invariant EKF that the contact doc reproduces and this doc extends.
3. Computational Robotics Laboratory, Czech Technical University in Prague. *Robots*. https://comrob.fel.cvut.cz/robots/ and its per-robot pages (`daisy.html`, `lily1.html`, `lily2.html`, `darpa.html`, `scarab2.html`) - HEBI Daisy (18 DOF, X8-9 and X8-16 actuators), HEBI Lily 1 and 2 (hexapods, 18 DOF, R8-9 and R8-16), the DARPA Sub-T hexapod and SCARAB among the lab's robots.
4. Bloesch, M., Hutter, M., Hoepflinger, M., Leutenegger, S., Gehring, C., Remy, C. D., & Siegwart, R. (2012). *State Estimation for Legged Robots - Consistent Fusion of Leg Kinematics and IMU*. Robotics: Science and Systems VIII. https://doi.org/10.15607/RSS.2012.VIII.003 - the non-invariant predecessor: an EKF with the footholds in the state, fusing leg kinematics and an IMU.
5. HEBI Robotics. *hebi-py* 2.16.2 (Python API, PyPI). https://pypi.org/project/hebi-py/2.16.2/ - `hebi.robot_model.import_from_hrdf`, the independent check of this doc's leg chain.
