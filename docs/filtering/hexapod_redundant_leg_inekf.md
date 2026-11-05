# Four joints per leg: which joint offsets a walking hexapod can learn

## Intuition

[`hexapod_joint_kinematics_inekf.md`](hexapod_joint_kinematics_inekf.md) (**the 3-joint doc** below) reads each stance foot's position from the leg's joint encoders, on a hexapod with three joints per leg. Its main finding is that fixed **joint zero offsets** ([glossary](../glossary.md#joint-encoder-and-zero-offset): an encoder whose zero is off by a constant angle) make the filter overconfident, and that estimating the offsets as filter states fixes the body's velocity.

The **HAntR** hexapod of the CTU Computational Robotics Laboratory (ref. 1) has **four** joints per leg. A foot position has only three coordinates, so the leg is **redundant**: for any foot position there is a one-parameter family of joint angles that reaches it. In the language of the leg's **Jacobian** $`J`$ ([glossary](../glossary.md#jacobian)), now a 3×4 matrix, there is a **null direction** $`n`$ with $`J n = 0`$ (the matrix's [null space](../glossary.md#null-space)): a combination of the four joint angles that, to first order, does not move the foot.

That raises a question the 3-joint doc could not ask. An encoder offset along $`n`$ moves no foot reading at that pose. Can a walking robot learn it at all?

**Short answer.**

- **The null direction is not the problem.** With offsets drawn with a 2° std (the filter's **prior**, its starting uncertainty), the offset along each leg's null direction at the default foothold ends at about 0.6° of error. The filter claims 0.36°, so it is somewhat overconfident. The likely reason it is learned at all: as the leg moves through a stance its pose changes and $`n`$ turns, so the offset no longer hides from the foot reading. That reason is not tested separately.
- **The hard offsets are elsewhere.** The offsets of the two outer joints, the **femur** and **tibia** (the leg's two pitch joints, §1.1), end with 1.28° and 1.79° of error at a 2° prior, while the filter reports about 0.58° for both. The largest errors lie along a combination that is mostly tibia, against femur, which moves the foot only 85 mm per radian, less than any single joint (138-206 mm).
- **The body still comes out right.** Estimating the 24 offsets keeps the velocity consistent (NEES 3.26 at 2°, [glossary](../glossary.md#consistency-and-nees)) and gives the lowest velocity error, as on the 3-joint robot. Only the offset estimates themselves are overconfident.
- **White encoder noise:** as in the 3-joint doc, the shape of the noise model barely matters at the encoder's resolution.

![Four panels from hexapod_redundant_leg_inekf.py, HAntR, tripod gait, 30 trials per point: velocity NEES and velocity RMSE against the joint zero-offset std for ignoring the offsets, treating them as extra white noise and estimating them as 24 states; the reported std of each leg's offset along its null direction over the run, relative to the prior, with the actual error at the end; and per joint, the actual offset error at the end against the reported std at a 2 degree prior](../../assets/hexapod_redundant_leg_inekf.png)

*Figure: `uv run python use_numpy/hexapod_redundant_leg_inekf.py --out assets/hexapod_redundant_leg_inekf.png` (about 22 minutes).*

### Scope
The 3-joint doc's setup with HAntR's legs instead of Daisy's: the contact doc's 3D walking body, IMU and right-invariant EKF ([`hexapod_contact_inekf.md`](hexapod_contact_inekf.md)), tripod gait, true stance schedule ([stance](../glossary.md#stance-and-swing): the feet on the ground), no IMU biases, no gating, 12 s runs. The body is the contact doc's (its speed, turn rate and sway), at HAntR's height. Only the leg geometry, the encoders and the redundancy rule are HAntR's. Run with `uv run python use_numpy/hexapod_redundant_leg_inekf.py`.

---

## 1. Model

### 1.1 HAntR's legs
Each leg has four servos (Dynamixel AX-12A): **coxa**, **trochanter**, **femur** and **tibia**, turning about a roll, yaw, pitch and pitch axis. The paper gives the chain as **Denavit-Hartenberg (DH) parameters** ([glossary](../glossary.md#denavit-hartenberg-dh-parameters)), a standard list of four numbers per joint that fixes how each joint's frame sits on the previous one (ref. 1, Table 2):

| Joint | Twist $`\alpha`$ (rad) | Link length $`a`$ (mm) | Offset $`d`$ (mm) | Angle offset $`\theta_\text{off}`$ (rad) |
| --- | --- | --- | --- | --- |
| Coxa | π/2 | 0 | 0 | 0 |
| Trochanter | −π/2 | 25.4 | −18.5 | π/2 |
| Femur | 0 | 81.6 | 0 | −0.52 |
| Tibia | 0 | 205.5 | 0 | −1.06 |

| Part | Value | Source (ref. 1) |
| --- | --- | --- |
| Leg mounts | every 60° around the body, 102 mm from its centre, 38.5 mm above the body frame, coxa axis pointing outward | Eq. 3 |
| Default **footholds** (where each foot is placed) | **leg span** 240 mm from the body centre | Section IV, foothold $`(d\,s_\beta,\ d\,c_\beta,\ -h)`$ |
| Body height | **ground clearance** 100 mm: the footholds are 100 mm below the body frame | same |
| Gait | tripod: every other leg around the body moves together | Section IV |

The paper writes the default foothold with the sine first, $`(d \sin\beta, d \cos\beta, -h)`$, while its Eq. 3 places the leg mounts at $`(\cos\beta, \sin\beta)`$. Taken literally, the foot of the leg mounted at 0° would sit at 90°. This doc places each foot along its own mount direction, reading the difference as an axis convention.

**The redundancy rule.** The paper's **inverse kinematics** (the joint angles that put the foot at a given point) picks, among all joint angles that reach the foot, the one whose **tibia** (the last link, from the tibia joint to the foot) points as close as possible to straight down along gravity (ref. 1, Section III-C; gravity as that direction, Section IV). The truth here follows that rule at every tick, with gravity taken in the body's current tilt. The script solves it numerically: the foot position, plus the condition that turning the leg along its null direction cannot bring the tibia closer to vertical. At the default foothold the tibia is 12.3° from vertical, and the joint angles are coxa 0°, trochanter 0°, femur 62.2° and tibia −49.4° (all six legs alike).

**The null direction.** At the default foothold, every leg has $`n \approx (0.71, 0.71, 0, 0)`$ (coxa, trochanter, femur, tibia): equal parts coxa and trochanter. Those two axes cross at one point, and this $`n`$ turns the leg about the line from that point to the foot, which leaves the foot where it is (checked in the tests). During a stance the body moves over the foot, that line turns, and so does $`n`$.

**How far each joint moves the foot.** At the default foothold, 1 rad on a single joint moves the foot 138.5 mm (coxa), 138.0 mm (trochanter), 193.2 mm (femur) or 205.5 mm (tibia). These are the lengths of the Jacobian's four columns.

### 1.2 Encoder readings
As in the 3-joint doc, each joint reads

```math
\tilde q = q + b + n_q, \qquad n_q \sim \mathcal N(0, \sigma_q^2), \qquad b \sim \mathcal N(0, \sigma_b^2) \text{ once per run}
```

with $`q`$ the true angle, $`\tilde q`$ the reading, $`n_q`$ white noise drawn every tick and $`b`$ the joint's zero offset, drawn once per run and then fixed.

**The noise is the encoder's step.** The AX-12A reads its position in steps of 0.29° (ref. 2). Rounding to such a step ([quantization noise](../glossary.md#quantization-noise)) adds an error spread evenly over one step, whose std is the step divided by $`\sqrt{12}`$: 0.084° (1.46 mrad). That is the default $`\sigma_q`$, modeled as white noise. The offsets have no published value and are swept from 0 to 2°.

---

## 2. Filters

The filters are the 3-joint doc's ([§2](hexapod_joint_kinematics_inekf.md#2-filters)), unchanged except that each leg has four joints:

| Filter | Foot-reading error it assumes | Used in |
| --- | --- | --- |
| `iso`, `iso_max` | a sphere, sized by the mean or the largest eigenvalue of $`\sigma_q^2 J J^\top`$ at the default foothold | §3 |
| `joint` | $`\sigma_q^2 J J^\top`$ at the reading's own angles | §3, §4 |
| `joint_inflated` | $`(\sigma_q^2 + \sigma_b^2) J J^\top`$: the offsets treated as more white noise | §4 |
| `joint_offsets` | $`\sigma_q^2 J J^\top`$, plus the 24 offsets (6 legs × 4 joints) as filter states | §4 |

The code is shared: `hexapod_joint_kinematics_inekf.py` now takes a **leg model** (its geometry, number of joints, stance and inverse kinematics), with Daisy as the default and HAntR added by this doc's script. Daisy's outputs are unchanged by that (the same printout and the same figure file, checked on a short run).

**Checked** (`tests/use_numpy/test_hexapod_redundant_leg_inekf.py`, 24 tests):
- the zero-angle pose has the paper's roll-yaw-pitch-pitch axes;
- the link lengths;
- the Jacobian matches finite differences;
- $`J n = 0`$, and at the default foothold $`n`$ uses only the coxa and trochanter and turns the leg about the line from their crossing point to the foot;
- the inverse kinematics reaches the foot, and moving along $`n`$ only turns the tibia further from vertical;
- the offset filter's measurement Jacobian matches a finite difference of its residual, on all four offset columns;
- the single-joint foot motions of §1.1.

**Scores** are the 3-joint doc's: velocity NEES (consistent value 3) and tilt NEES (consistent value 2), plus RMSE ([glossary](../glossary.md#rmse-root-mean-square-error)), averaged over the run after the first second. Over 30 [Monte Carlo trials](../glossary.md#monte-carlo-trials), a consistent filter's run-average velocity NEES falls in [2.19, 3.94] 95% of the time, and its tilt NEES in [1.35, 2.78].

---

## 3. White encoder noise: a check

| $`\sigma_q`$ | Filter | Velocity NEES | Tilt NEES | Velocity RMSE (mm/s) |
| --- | --- | --- | --- | --- |
| 1.46 mrad (one quantization std) | `iso` | 3.00 | 2.06 | 5.21 |
| | `iso_max` | 2.79 | 2.04 | 5.24 |
| | `joint` | 3.01 | 2.05 | 5.17 |
| 5.06 mrad (one full 0.29° step) | `iso` | 3.14 | 2.14 | 6.42 |
| | `iso_max` | 2.74 | 2.10 | 6.50 |
| | `joint` | 3.12 | 2.13 | 6.15 |

The result is the same as on Daisy: all three are consistent. The full model is about 4% more accurate at the coarser noise.

---

## 4. Joint zero offsets

### 4.1 The body

| Offset std | `joint` (ignore): NEES / RMSE | `joint_inflated`: NEES / RMSE | `joint_offsets` (estimate): NEES / RMSE |
| --- | --- | --- | --- |
| 0.5° | 3.21 / 5.47 | 2.48 / 6.31 | 3.04 / 5.42 |
| 1° | 3.78 / 5.85 | 2.55 / 7.48 | 3.05 / 5.38 |
| 2° | 5.71 / 7.00 | 2.68 / 9.40 | 3.26 / 5.55 |

Velocity NEES and velocity RMSE (mm/s). With no offsets, all three filters give NEES 2.98 and RMSE 5.21 mm/s.

As on Daisy:
- ignoring the offsets makes the filter overconfident as they grow (outside the band at 2°);
- calling them white noise makes it cautious and less accurate;
- estimating them keeps it consistent and the most accurate.

### 4.2 The offsets themselves

Each leg's four offset errors at the end of the run, split three ways:

| Offset std (prior) | Along the null direction: actual / reported | Other three directions: actual / reported | Largest-error direction: actual / reported |
| --- | --- | --- | --- |
| 0.5° | 0.29° / 0.27° | 0.48° / 0.33° | 0.68° / 0.38° |
| 1° | 0.37° / 0.33° | 0.87° / 0.46° | 1.29° / 0.58° |
| 2° | 0.59° / 0.36° | 1.41° / 0.55° | 2.12° / 0.77° |

- **Actual:** RMSE over 30 trials × 6 legs.
- **Reported:** the square root of the mean variance the filter reports along the same direction.
- **Other three directions:** per direction, i.e. the total over the three directions orthogonal to $`n`$, divided by three.
- **Null direction:** each leg's $`n`$ at the default foothold.
- **Largest-error direction:** the direction of the four offsets with the largest actual error, pooled over legs and trials. Because it is picked after the fact as the worst, its actual RMSE is biased upward (2.12° at 2° is slightly above the prior for that reason). It is mostly tibia, against femur: (0.03, −0.08, −0.55, 0.83) at 2°, (0, −0.03, −0.38, 0.92) at 1° and (0, 0.06, −0.28, 0.96) at 0.5°. It moves the foot 85 mm per radian at 2° (125 and 149 mm at 1° and 0.5°).

Per joint at 2°, the actual error at the end is 0.52° (coxa), 1.09° (trochanter), 1.28° (femur) and 1.79° (tibia). The filter reports 0.31°, 0.51°, 0.59° and 0.58°.

#### The null direction is learned
The figure's lower-left panel shows the reported std along $`n`$ falling well below the prior within the first seconds. At the end the actual error is above it: 0.29° against 0.27° reported at a 0.5° prior, 0.37° against 0.33° at 1°, and 0.59° against 0.36° at 2° (NEES along it 2.67, where 1 is consistent).

#### The femur and tibia offsets are not
The femur and tibia offsets end at 1.28° and 1.79° at a 2° prior, while the filter reports 0.59° and 0.58°. Along the largest-error direction the filter reports 0.77°. This direction moves the foot less than any single joint: 85 mm per radian, against 138-206 mm. The femur and tibia turn about parallel axes, so turning one forward and the other back mostly cancels at the foot.

Why the filter believes it has learned this combination is not shown here. One possibility, not tested: each foot's position is itself a filter state (contact doc), so part of an offset's effect on the foot can be absorbed into that estimate instead.

#### One run, animated
The 2° case, one run, animated: HAntR walks (left, each leg drawn from the DH table of §1.1 at its true joint angles, the front right leg highlighted) while the two result panels fill in. The run is seed 103, the first trial of the script's own 2° case, so it was not picked from several. With only six legs per curve it is noisier than the 30-trial figures above, and its errors are somewhat larger. The offset error along $`n`$ (root mean square over the six legs) goes from 2.50° at the start to 0.70° at the end (30-trial figure: 0.59°), while the reported std goes from 2.00° to 0.35° and is below 1° from 0.9 s. At the end the femur offset error is 1.75° against 0.59° reported, and the tibia's 1.98° against 0.58° (30 trials: 1.28° and 1.79°): the same pattern as above, the null direction mostly learned and the femur and tibia not. This run's true femur and tibia offsets happen to be small (0.73° and 1.57° root mean square over the six legs, against the 2° prior), so their curves start low; the errors then grow, to 1.75° and 1.98°, while the reported std keeps falling.

![Animation of one tripod-gait run of the HAntR hexapod with 2 degree joint zero offsets: left, a 3D view of the hexapod walking with all six four-joint legs drawn at their true joint angles, the true stance feet on the ground and the front right leg highlighted with its four joints marked and a dotted sketch of that leg turned along its null direction; right, over time the offset error along the null direction against the std the filter reports, which falls quickly and stays below the error, and the femur and tibia offset errors against their reported std, where the errors stay between about 1.3 and 2 degrees while the reported std falls to about 0.6](../../assets/hexapod_redundant_leg_inekf.gif)

*Animation: `uv run python generator/animate_hexapod_redundant_leg_inekf.py --out assets/hexapod_redundant_leg_inekf.gif` (seed 103 by default, about 45 s). Swing legs are drawn at the default foothold (the script does not simulate swing feet), and the dotted legs are an exaggerated sketch of the highlighted leg turned ±0.4 rad along its null direction at that tick (its foot moves at most 8.5 mm): both are illustrative. The body is drawn in a heading frame (yaw removed, tilt and bob kept).*

---

## 5. What this toy does not show

- **The paper's foothold convention.** Each foot is placed along its own mount direction (§1.1); if the paper's sine-first foothold meant something else, the pose, and with it $`n`$ and the Jacobian, would change.
- **The paper's walking.** The body motion is the contact doc's, not HAntR's own controller (which moves the body continuously over the feet and can change the gait).
- **The AX-12A itself.** Its backlash, its compliance under load and the 24 ms it takes to read all joints (ref. 1) are left out. The quantization is treated as white noise, which it is only roughly.
- **A cure for the offset overconfidence.** The same overconfidence appears on Daisy (3-joint doc §4), whose second and third joints also turn about parallel axes. Whether the same combination is responsible there was not tested. Neither a longer run, richer motion (turning in place, shifting the body over planted feet) nor a calibration step was tried.

---

## 6. Takeaway

A fourth joint per leg gives the foot reading a blind direction at every pose, yet the offset along that direction is mostly learned while walking, likely because walking keeps turning it. The offsets that stay poorly learned are the femur's and tibia's, along a combination that moves the foot less than any single joint. For the body's velocity, estimating all 24 offsets is still the right choice. The offsets it reports should not be trusted as a calibration.

---

## 7. References

1. Čížek, P., Zoula, M., & Faigl, J. (2021). *Design, Construction, and Rough-Terrain Locomotion Control of Novel Hexapod Walking Robot With Four Degrees of Freedom Per Leg*. IEEE Access, 9, 17866-17881. https://doi.org/10.1109/ACCESS.2021.3053492 (open access, CC-BY 4.0; PDF on the lab's HAntR page, https://comrob.fel.cvut.cz/robots/hantr.html) - the leg's DH parameters (Table 2), leg mounts (Eq. 3), the inverse kinematics with the tibia kept closest to gravity (Section III-C), the default leg span and ground clearance (Section IV), 24 DoF and 2.9 kg (Table 1).
2. ROBOTIS. *AX-12/AX-12+/AX-12A e-Manual*. https://emanual.robotis.com/docs/en/dxl/ax/ax-12a/ - position resolution 0.29°, range 0-300°, position and load feedback.
