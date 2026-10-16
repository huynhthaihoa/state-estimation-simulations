# Contact-aided invariant EKF on a hexapod: which filter knows yaw is unobservable

## Intuition

An IMU ([glossary](../glossary.md#imu-inertial-measurement-unit)) plus leg kinematics is the standard **proprioceptive** setup for a legged robot: sensors that measure only the robot's own motion and joints, not its surroundings. The IMU drives the body forward in time. Each foot on the ground stays put, and the leg's joint encoders say where that foot is relative to the body. If the filter keeps the **world position** of every **stance foot** (a foot on the ground, as opposed to a **swing** foot in the air) in its state, each kinematic reading ties the body to a point that isn't moving. Hartley et al. (ref. 1) built this as an invariant EKF ([glossary](../glossary.md#iekf-invariant-ekf)), and this toy reproduces it on a hexapod.

What can such a robot know? Picture the whole world, robot and footholds included, **rotated about the gravity axis**:

- the gyro reads the same body rates,
- the accelerometer reads the same specific force (gravity is unchanged by that rotation),
- every leg reports the same foot position in the body frame.

No reading changes, so **yaw** (the heading, the rotation about gravity) is **unobservable**, and so is **absolute position** (shift the whole world sideways: same argument). **Roll** and **pitch**, together the body's **tilt**, are **observable**, because gravity points one way ([glossary](../glossary.md#observability)).

So a correct filter can only ever *lose* information about yaw. The experiments below check which filter respects that:

- the **invariant EKF**, whose error dynamics do not depend on the estimate;
- the ordinary **world-frame error-state EKF** ([glossary](../glossary.md#eskf-error-state-kalman-filter)), whose Jacobians are evaluated at the estimate.

Short answer: both are about equally *accurate*, but the world-frame EKF gains spurious yaw information. With a larger initial rotation error it becomes badly overconfident: NEES 18 instead of 9 at a 6° initial std, 481 at 17°.

![Three panels from hexapod_contact_inekf.py: yaw error over 12 s with an initial rotation std of 0.1 rad, where the invariant EKF's actual RMSE (6.2 deg) and reported sigma (5.7 deg) stay flat while the world-frame EKF's reported sigma falls to 1.9 deg below its actual 6.8 deg; average NEES bars for initial rotation std 0.02, 0.1 and 0.3 rad, invariant EKF 9.3, 9.3, 9.4 and world-frame EKF 9.6, 18.2, 481, against the 95% band for 9 DoF; and velocity RMSE of the invariant EKF over time with leg L2 slipping between 4 and 8 s, for the tripod and wave gaits, trusting every foot versus the per-foot chi-squared gate](../../assets/hexapod_contact_inekf.png)

*Figure: `use_numpy/hexapod_contact_inekf.py` at its defaults (seed 0, 50 trials), plotted by `uv run python assets/make_figures.py hexapod_contact_inekf` (about 2 minutes). The numbers match the script's printout.*

**Scope.** 3D body walking at fixed height on flat ground, with a small roll, pitch and up-down bob at twice the gait frequency; tripod, ripple and wave gaits and their footholds from [`hexapod_gait_ekf.md`](hexapod_gait_ekf.md); the true stance schedule; isotropic kinematic noise (no joint-encoder model). Biases are off except in §5. Run with `uv run python use_numpy/hexapod_contact_inekf.py` (about 4 minutes).

---

## 1. Model

**State.** Rotation $`R`$ (body to world), velocity $`v`$, position $`p`$ and the world positions $`d_1, \dots, d_k`$ of the $k$ feet on the ground, packed into one matrix:

```math
X = \begin{bmatrix} R & v & p & d_1 & \cdots & d_k \\ 0 & & I_{k+2} & & \end{bmatrix} \in SE_{k+2}(3)
```

This is the extended pose $`SE_2(3)`$ ([glossary](../glossary.md#extended-pose)) with one more column per foot. $k$ follows the gait: 3, 4 or 5 feet for **tripod**, **ripple** and **wave**. A foot joins the state at touchdown and leaves it at liftoff.

**Propagate** (`propagate_base`), with gyro $`\omega`$ and accelerometer $`a`$ over one step $`\Delta t`$, gravity $`g`$, and $`\mathrm{Exp}`$ turning a rotation vector into a rotation ([glossary](../glossary.md#exp-and-log-maps)):

```math
R^+ = R\,\mathrm{Exp}(\omega \Delta t), \quad v^+ = v + g\Delta t + R a \Delta t, \quad p^+ = p + v\Delta t + \tfrac12 g \Delta t^2 + \tfrac12 R a \Delta t^2
```

A stance foot stays put up to a **contact noise** $`w_d`$, which is a small random body-frame velocity (std 0.01 m/s): $`d^+ = d + R w_d \Delta t`$. The truth uses exactly these models, so the noise the filters assume is the noise that is there.

**Measure** (leg kinematics), for each stance foot, with noise $`n`$ (std $`\sigma_{\text{kin}} = 5`$ mm per axis):

```math
y_i = R^\top (d_i - p) + n
```

**Touchdown and liftoff.** A foot that lands is initialized from its first reading, $`\hat d = \hat p + \hat R y`$. A foot that lifts is dropped by deleting its rows and columns of the covariance $P$ ([glossary](../glossary.md#covariance)), which is exact marginalization of a Gaussian ([glossary](../glossary.md#marginalization)).

---

## 2. Two filters, one model

Both filters run the same mean propagation and the same bookkeeping. They differ in the error they track, and so in their Jacobians.

**Invariant EKF** (`InvariantContactEKF`). The right-invariant error $`\xi = \log(\hat X X^{-1})`$ ([glossary](../glossary.md#invariant-and-equivariant)). Three things come out independent of the estimate:

- **Propagation.** IMU kinematics are **group-affine** ([glossary](../glossary.md#group-affine-dynamics-trajectory-independence)), so the error transition is the constant $`SE_2(3)`$ matrix of [`invariant_saltation_ekf.md`](invariant_saltation_ekf.md#2-why-the-riekf-is-attractive-between-bounces), with the identity on every foot.
- **Measurement.** The residual $`r_i = \hat R y_i + \hat p - \hat d_i`$ ([glossary](../glossary.md#residual-error)) is, to first order, $`\xi_p - \xi_{d_i}`$: the Jacobian is $`H_i = [0, 0, I, \dots, -I, \dots]`$, with no rotation columns and no dependence on the estimate.
- **Touchdown.** The new foot's error is $`\xi_p + \hat R n`$: the position rows of $P$, plus the reading noise.

**World-frame EKF** (`WorldContactEKF`). The error $`(\delta\theta, \delta v, \delta p, \delta d)`$ with $`R = \mathrm{Exp}(\delta\theta)\hat R`$ and the rest additive, as in the baseline of [`invariant_saltation_ekf.md`](invariant_saltation_ekf.md). Its Jacobians all contain the estimate: the transition has $`-(\hat R a \Delta t)^\wedge`$ terms ($`(\cdot)^\wedge`$ is the skew-symmetric matrix of a vector, [glossary](../glossary.md#hat-operator)), the measurement Jacobian has a rotation column $`\hat R^\top (\hat d_i - \hat p)^\wedge`$, and a new foot's error depends on $`-(\hat R y)^\wedge \delta\theta`$.

**Why only one filter can learn yaw.** In the invariant EKF, a global yaw rotation by an angle $`\theta`$ is the fixed error direction $`\xi = (\theta e_z, 0, \dots, 0)`$, with $`e_z`$ the world's vertical unit vector. The measurement Jacobian has no rotation columns, and gravity has no yaw component, so the transition leaves that direction alone. No information can reach it: over 30 runs (3 gaits) its yaw variance never decreased on any step. In the world-frame EKF, the unobservable direction of the linearized system depends on the estimate at which each Jacobian is evaluated. Those estimates change from step to step, so the linearized system appears to observe yaw. This is the same mechanism that **FEJ** (first-estimate Jacobians, [glossary](../glossary.md#fej-first-estimate-jacobians)) addresses in sliding-window smoothers ([`marginalization.md` §6](../optimization/marginalization.md#6-the-consistency-gotcha-why-fej-exists)).

**Checked.** `tests/use_numpy/test_hexapod_contact_inekf.py` compares every Jacobian with finite differences (numerical derivatives from small perturbations of the state): the invariant transition is exact (error about $`10^{-9}`$) and identical at two random states; the world-frame one matches but changes with the state. With no noise, both filters track the truth to about $`10^{-15}`$.

---

## 3. No biases: equal accuracy, different honesty

**Default initial error** (rotation std 0.02 rad, about 1.1°). 50 trials, seed 0; NEES of $`(R, v, p)`$ ([glossary](../glossary.md#consistency-and-nees)) should be 9 (95% band $`[7.86, 10.21]`$) and yaw NEES 1 ($`[0.65, 1.43]`$). RMSE is the root-mean-square error over the trials:

| Gait | Filter | NEES | yaw NEES | final yaw RMSE / reported σ (°) | tilt RMSE (°) | velocity RMSE (m/s) | final position RMSE (m) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tripod | invariant | 9.06 | 1.13 | 1.219 / 1.154 | 0.261 | 0.0134 | 0.0320 |
| tripod | world | 9.27 | 1.33 | 1.263 / 1.057 | 0.261 | 0.0134 | 0.0323 |
| ripple | invariant | 9.07 | 1.13 | 1.221 / 1.154 | 0.259 | 0.0120 | 0.0273 |
| ripple | world | 9.24 | 1.28 | 1.249 / 1.043 | 0.261 | 0.0121 | 0.0274 |
| wave | invariant | 8.91 | 1.13 | 1.220 / 1.154 | 0.250 | 0.0113 | 0.0282 |
| wave | world | 9.15 | 1.35 | 1.298 / 1.034 | 0.252 | 0.0113 | 0.0292 |

- **Both are consistent here, and equally accurate.** Every NEES is inside the band.
- **But the world-frame EKF already "learns" yaw.** Its reported yaw σ ends at 1.03-1.06°, *below* the 1.146° it started with; the invariant EKF's grows to 1.154° (gyro noise only).
- **The gait barely matters**, as in [`hexapod_gait_ekf.md` §2](hexapod_gait_ekf.md#2-no-slip-the-gait-matters-but-less-than-youd-think): velocity RMSE 0.0134 → 0.0113 m/s from tripod to wave.

**Larger initial rotation error** (tripod, 50 trials each). The world-frame EKF's Jacobians are evaluated further from the truth, so the spurious yaw information grows:

| initial rotation std | Filter | NEES | yaw NEES | final yaw RMSE / reported σ (°) | final position RMSE (m) |
| --- | --- | --- | --- | --- | --- |
| 0.02 rad | invariant | 9.30 | 1.14 | 1.24 / 1.15 | 0.032 |
| 0.02 rad | world | 9.58 | 1.39 | 1.33 / 1.06 | 0.033 |
| 0.1 rad | invariant | 9.33 | 1.16 | 6.19 / 5.73 | 0.107 |
| 0.1 rad | world | **18.2** | **9.16** | 6.76 / **1.91** | 0.117 |
| 0.3 rad | invariant | 9.40 | 1.22 | 18.98 / 17.19 | 0.316 |
| 0.3 rad | world | **481** | **388** | 20.20 / **1.33** | 0.335 |

- **The invariant EKF stays consistent at every level**: its error dynamics do not depend on the estimate, so a bad estimate does not distort its covariance.
- **The world-frame EKF claims to know yaw to 1.3°** when its actual error is 20°.
- **Accuracy is close.** The world-frame EKF's yaw RMSE is 6-9% higher. Position error grows with the initial yaw error for both: a heading error turns the whole estimated path about its start, and 6° (0.108 rad) times the 0.94 m RMS distance walked in 12 s is about 0.10 m.
- **Why it matters.** An overconfident filter rejects good measurements in a gate and weights a later loop closure ([glossary](../glossary.md#loop-closure)) or GPS fix too little against its own wrong yaw.

The 0.1 rad case, one run, animated. Seed 4, not the script's seed 0: seed 0 happens to draw an initial yaw error of only 0.08°, too small to see. Here the world-frame EKF's ±2σ band shrinks from 11.5° to 4.1° while its updates drag the yaw from −1.9° to −6.8°, and its path rotates away (final position error 0.157 m vs 0.050 m). The invariant EKF's yaw error stays at −1.8°, inside a band that never shrinks.

![Animation of one tripod-gait run with 0.1 rad initial rotation std: left, a top-down view of the hexapod walking a curved path, its true stance feet and each filter's estimated body outline and estimated feet, with the true and both estimated paths; right, each filter's yaw error over time with its shaded plus-minus two sigma band, the world-frame EKF's band shrinking below its error, and each filter's position error over time](../../assets/hexapod_contact_inekf.gif)

*Animation: `uv run python use_numpy/animate_hexapod_contact_inekf.py --out assets/hexapod_contact_inekf.gif` (seed 4 by default). Swing legs are drawn at fixed default positions relative to the body; the script does not simulate swing feet.*

---

## 4. One slipping leg: gating, and what feet in the state change

L2 slides outward at 0.1 m/s for $`t \in [4, 8)`$ s whenever it is down, as in [`hexapod_gait_ekf.md` §3](hexapod_gait_ekf.md#3-one-slipping-leg-the-gait-barely-matters-gating-does). Invariant EKF only. The **gated** filter tests each stance foot's residual against the prediction before the joint update ([glossary](../glossary.md#outlier-and-robust-kernel)):

```math
r_i^\top \left(H_i P^- H_i^\top + \sigma_{\text{kin}}^2 I\right)^{-1} r_i \le 11.34
```

$`P^-`$ is the predicted covariance ([glossary](../glossary.md#predict-and-update-steps)). 11.34 is the 99% point of a $`\chi^2`$ with 3 DoF ([glossary](../glossary.md#chi-square-distribution)).

Results (50 trials; "in window" = $`t \in [4, 8)`$ s):

| Gait | Filter | in-window velocity RMSE (m/s) | in-window NEES | final position RMSE (m) | slip readings used | good readings used |
| --- | --- | --- | --- | --- | --- | --- |
| tripod | trust every foot | 0.0231 | 31.5 | 0.072 | 100% | 100% |
| tripod | gated | 0.0120 | 10.5 | 0.036 | 51.8% | 99.0% |
| ripple | trust every foot | 0.0241 | 38.5 | 0.067 | 100% | 100% |
| ripple | gated | 0.0097 | 9.8 | 0.030 | 34.2% | 99.0% |
| wave | trust every foot | 0.0231 | 41.9 | 0.073 | 100% | 100% |
| wave | gated | 0.0091 | 10.2 | 0.030 | 29.7% | 99.0% |

- **The gate halves the error and restores consistency** (NEES about 10 instead of 31-42).
- **But it lets 30-52% of the slipping readings through, against about 2% in the velocity-reading filter.** The reason is the foot in the state. The residual is the distance the foot has slid since touchdown, not its velocity. At 2 mm per tick (one 0.02 s filter step) the slide starts below the 5 mm reading noise: in the first 5 ticks after the slip begins, 96-100% of the slipping readings pass; after about 23 ticks (4.6 cm) almost none do. The tripod, with 30-tick stances, uses the most.
- **Those early readings cost little.** Gated velocity RMSE in the window (0.0091-0.0120 m/s) is close to the no-slip values of §3 (0.0113-0.0134 m/s, whole run).

---

## 5. IMU biases: the "imperfect" invariant EKF

Each trial draws a constant gyro **bias** ([glossary](../glossary.md#bias)) (std 0.01 rad/s) and accelerometer bias (std 0.05 m/s²); both filters add the six biases to the state, with that prior. The bias error enters the invariant error through $`-\mathrm{Ad}_{\hat X}`$ ([glossary](../glossary.md#adjoint)), which depends on the estimate. So exact invariance is lost: Hartley et al. call this the **imperfect InEKF** (InEKF is their abbreviation for the invariant EKF). The bias columns of both filters are a first-order discretization (their error shrinks 4× when $`\Delta t`$ halves; tested).

50 trials; bias NEES should be 6 (95% band $`[5.08, 7.00]`$):

| Gait | Filter | NEES | yaw NEES | bias NEES | final yaw RMSE / reported σ (°) | final position RMSE (m) |
| --- | --- | --- | --- | --- | --- | --- |
| tripod | invariant | 9.50 | 0.97 | 6.06 | 2.13 / 2.34 | 0.037 |
| tripod | world | 9.96 | 1.35 | 6.28 | 2.61 / 2.05 | 0.041 |
| ripple | invariant | 8.94 | 1.11 | 6.15 | 2.20 / 1.84 | 0.032 |
| ripple | world | 9.37 | **1.54** | 6.25 | 2.54 / 1.63 | 0.037 |
| wave | invariant | 8.89 | 0.84 | 6.27 | 1.47 / 1.62 | 0.028 |
| wave | world | 9.34 | 1.27 | 6.38 | 1.85 / 1.43 | 0.032 |

- **The invariant EKF stays consistent** on every measure, even though it is no longer exactly invariant. Its yaw variance can now shrink, but only slightly: at most 0.05% on one step over 30 runs.
- **The world-frame EKF is again mildly overconfident in yaw**: yaw NEES 1.27-1.54 against the invariant EKF's 0.84-1.11, above the 1.43 bound for ripple. This is at the default 0.02 rad initial error; §3 shows how it grows with a larger one.
- **A gyro bias makes yaw drift faster.** The reported yaw σ after 12 s roughly doubles compared with §3 (tripod 2.34° vs 1.15°). The final-tick values are from 50 runs, so their ratios are noisy; the time-averaged NEES is the reliable number.

---

## 6. Takeaway

- **Put the stance feet in the state and use an invariant error.** Then the measurement Jacobian and the touchdown step are constant, and the filter's unobservable directions (yaw and position) are the true ones. That is the main point of ref. 1.
- **The standard EKF is as accurate but not honest.** At a 1° initial rotation std the difference is small. At 6° its NEES doubles, and at 17° it is about 50× too high, with a reported yaw σ of 1.3°.
- **Biases cost exact invariance, not consistency**, in this setting.
- **Gating still works with feet in the state, but catches a slip later.** The residual measures how far a foot has slid, so the first centimetre passes.
- **Yaw and position still drift.** IMU plus legs fixes roll, pitch and velocity; an **exteroceptive** sensor, one that measures the surroundings (camera, lidar, GPS), is needed for the rest.

---

## 7. References

1. Hartley, R., Ghaffari, M., Eustice, R. M., & Grizzle, J. W. (2020). *Contact-Aided Invariant Extended Kalman Filtering for Robot State Estimation*. The International Journal of Robotics Research, 39(4), 402-430. https://doi.org/10.1177/0278364919894385 - the filter this toy reproduces: IMU plus contact points on $`SE_{K+2}(3)`$, adding and removing contacts, and the imperfect InEKF with biases.
2. Barrau, A., & Bonnabel, S. (2017). *The Invariant Extended Kalman Filter as a Stable Observer*. IEEE Transactions on Automatic Control, 62(4), 1797-1812. https://doi.org/10.1109/TAC.2016.2594085 - the theory behind the filter: group-affine dynamics, whose invariant error propagates independently of the estimate, and the resulting stability of the invariant EKF as an observer (a state estimator).
3. Bloesch, M., Hutter, M., Hoepflinger, M., Leutenegger, S., Gehring, C., Remy, C. D., & Siegwart, R. (2012). *State Estimation for Legged Robots - Consistent Fusion of Leg Kinematics and IMU*. Robotics: Science and Systems VIII. https://doi.org/10.15607/RSS.2012.VIII.003 - the non-invariant predecessor: an EKF with the footholds in the state, and an observability analysis of the same setup.
