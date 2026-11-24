# Getting started

A reading-and-running order for new learners in this repo. Each step names the [`docs/`](docs/) sections to read, then the simulation to run under [`use_numpy/`](use_numpy/) (or its [`use_manif/`](use_manif/) counterpart). Keep that order: each script exercises what the reading just introduced.

Before starting, follow [README.md § D. Installation](README.md#d-installation) to get `uv sync` working. Every script below runs as `uv run python use_numpy/<script>.py [flags]`. The full flag list for each lives in [README.md § E. Simulations](README.md#e-simulations); this guide links to the relevant numbered entry instead of repeating it.

## Before you start

- **Background assumed:** Python with numpy; matrices (multiply, transpose, inverse); mean, variance, and the Gaussian ("bell curve") distribution. Phase 3 opens with [`probability_basics.md`](docs/foundations/probability_basics.md), which builds the rest of the probability the repo uses (covariance, fusing estimates, Bayes' rule, NEES) from there.
- **Section numbers:** "§3" means the doc's heading numbered 3, and "§1-§3" means its sections 1 to 3. Read only the listed sections: the rest of a doc comes later in this guide or belongs to a deep dive.
- **Unknown term?** Every term used in the docs is defined in [`docs/glossary.md`](docs/glossary.md).

## The path at a glance

```text
Main route:   Phase 0 → 1 → 2 → 3 → 4 → 5 ──→ Track L (the destination)
 │
Smoothing track:             └─(after Phase 2)─→ S1 → S2 → S3     (optional, parallel)
Deep dives:   D1-D5, each when its "Needs" are met                (optional)
```

| Step | The question it answers | You run |
| --- | --- | --- |
| [Phase 0](#phase-0---the-big-picture-read-only) | What are the parts of a SLAM (localization and mapping) system, and what are the two ways to estimate a robot's state? | nothing (reading only) |
| [Phase 1](#phase-1---rotations-are-not-vectors) | Why can't you just add up rotation angles? | `imu_integration_comparison.py` |
| [Phase 2](#phase-2---jacobians-least-squares-gauss-newton) | How do you fit a model to noisy measurements? | `imu_propagation_position_fix.py` |
| [Phase 3](#phase-3---filter-basics-one-guess-or-several) | How does a filter blend a prediction with a measurement, and which filter fits which situation? | `probability_basics.py`, `particle_filter_localization.py` |
| [Phase 4](#phase-4---filters-on-rotations) | How does a filter handle orientation correctly? | `eskf_imu_position_fix.py`, `pointcloud_pose_tracking.py` |
| [Phase 5](#phase-5---is-the-filter-honest-consistency-and-observability) | Is the filter honest about its own uncertainty, and what can it never see? | `probability_basics.py --part nees`, `imu_allan_variance.py`, `observability_fej.py` |
| [Track L](#track-l---hybrid-and-bio-inspired-locomotion-the-destination) | How do you track a robot that bounces, crawls on anchors, walks, or slips? | a first pass of 5 scripts, then 11 more |

- **Main route (Phases 0-5):** everything [Track L](#track-l---hybrid-and-bio-inspired-locomotion-the-destination) builds on. Every Track L script is a filter, and every Track L doc checks whether its filter is honest about its own uncertainty (Phase 5's topic).
- **Smoothing track (S1-S3):** building maps by re-solving the whole history (the SLAM back-end used by graph-based systems). Track L does not need it; skip it on a first pass.
- **Deep dives (D1-D5):** more depth on one topic. Skip them on a first pass too.

## How to work through each step

1. **Read** the listed sections.
2. **Predict, then run.** Before running the script, guess what its plot or printout will show. A wrong guess is the most useful kind: it shows exactly what you hadn't understood yet.
3. **Check yourself** at the step's **Checkpoint** ("You're ready when you can explain …"). Explain it in two or three sentences, out loud or in writing, without looking; if you can't, reread just that part before moving on.
4. **Stuck?** Look the term up in the [glossary](docs/glossary.md); see where the step fits in [`slam_mental_map.md`](docs/slam_mental_map.md) §1; find the script's flags in its README entry.

---

## Main route

### Phase 0 - The big picture (read only)

**Why this step:** a map of the territory before you walk it.

1. [`docs/frontend_backend.md`](docs/frontend_backend.md) - what a SLAM (simultaneous localization and mapping) front-end and back-end each do.
2. [`docs/filtering_smoothing.md`](docs/filtering_smoothing.md) §1-§3 - the repo's central fork: **filtering** (keep only the current state) vs. **smoothing** (keep the history and re-solve it).
3. Skim `filtering_smoothing.md` §11: why robots with contacts and jumps make both harder. That is Track L's motivation; don't worry about its terms yet.
4. Skim [`docs/slam_mental_map.md`](docs/slam_mental_map.md) §1 for the layout only. Come back to its later sections after the main route.

**Checkpoint:** you're ready when you can explain the front-end vs. the back-end, and filtering vs. smoothing, in one sentence each.

### Phase 1 - Rotations are not vectors

**Why this step:** almost every script tracks an orientation, and orientations need their own arithmetic.

1. **Read** [`docs/foundations/lie_algebra.md`](docs/foundations/lie_algebra.md) §1-§10 and §13-§14 - why orientation lives on $`SO(3)`$ and is updated with $`\text{Exp}`$/$`\text{Log}`$ instead of vector addition.
2. **Run** [README.md §1](README.md#1-naive-vs-exp-map-imu-integration) - [`use_numpy/imu_integration_comparison.py`](use_numpy/imu_integration_comparison.py) - naive Euler-angle vs. $`SO(3)`$ exp-map integration: same noisy data, two integration schemes, and the flat one drifts.

**Checkpoint:** you're ready when you can explain why adding angle increments fails for 3D rotation, and what $`\text{Exp}`$ and $`\text{Log}`$ convert between.

### Phase 2 - Jacobians, least squares, Gauss-Newton

**Why this step:** every filter and optimizer in the repo works by approximating a curved model with a straight one near the current guess.

1. **Read** [`docs/foundations/jacobian.md`](docs/foundations/jacobian.md) §1-§8 - a Jacobian as a local linear model ("sensitivity map").
2. **Read** [`docs/optimization/nonlinear_least_square.md`](docs/optimization/nonlinear_least_square.md) §1-§8 - fitting a model to noisy measurements, and making a nonlinear problem locally linear.
3. **Read** [`docs/optimization/gauss_newton.md`](docs/optimization/gauss_newton.md) §1-§6 - fit a bowl, jump to its bottom, repeat.
4. **Run** [README.md §2](README.md#2-imu-propagation--gauss-newton-position-correction) - [`use_numpy/imu_propagation_position_fix.py`](use_numpy/imu_propagation_position_fix.py) - IMU propagation plus a periodic Gauss-Newton correction against a GPS-like fix: "linearize, solve, update" in a simple setting.

**Checkpoint:** you're ready when you can explain what a Jacobian tells you, and the linearize → solve → update loop of Gauss-Newton.

### Phase 3 - Filter basics: one guess or several

**Why this step:** a filter's belief about the state is a probability "landscape". Most filters assume one hill (one best guess with some spread); some situations need several hills (several places the robot could be).

1. **Read** [`docs/foundations/probability_basics.md`](docs/foundations/probability_basics.md) §1-§6 - uncertainty as a distribution, covariance, pushing a Gaussian through a map, fusing two estimates, Bayes' rule, and why least squares.
2. **Run** [README.md §35](README.md#35-probability-basics-fusing-propagating-and-checking-uncertainty) - `uv run python use_numpy/probability_basics.py --part fusion --part propagation` - where the Kalman filter's 70%/30% weights come from, and when a straight-line (Jacobian) approximation of a curved map holds.
3. **Read** [`docs/filtering/kf_ekf_iekf.md`](docs/filtering/kf_ekf_iekf.md) §1-§2 - the Kalman filter (predict, then correct) and the EKF (linearize, then do the same).
4. **Read** [`docs/filtering/linear_nonlinear.md`](docs/filtering/linear_nonlinear.md) - how to tell a linear system from a nonlinear one, and whether the nonlinearity matters at your uncertainty level.
5. **Read** [`docs/filtering/extra_kf_variants.md`](docs/filtering/extra_kf_variants.md) §1 - the UKF.
6. **Read** [`docs/filtering/kf_ekf_iekf.md`](docs/filtering/kf_ekf_iekf.md) §8 - which filter to pick: how many hills does the belief have, and how curved is the model?
7. **Read** [`docs/filtering/particle_filter.md`](docs/filtering/particle_filter.md) - a belief with several hills.
8. **Run** [README.md §8](README.md#8-particle-filter-localization-several-hills-against-one-hill) - [`use_numpy/particle_filter_localization.py`](use_numpy/particle_filter_localization.py) - a robot at an unknown start in a corridor of look-alike doors: a particle filter against an EKF and a UKF on the same data.

**Checkpoint:** you're ready when you can explain where the 70%/30% weights of two fused estimates come from, the predict and update steps, what the EKF linearizes, and when you need a UKF or a particle filter instead.

### Phase 4 - Filters on rotations

**Why this step:** Phase 3's filters treat the state as a flat vector; real robots have an orientation, and Phase 1 showed it isn't one.

1. **Read** [`docs/foundations/lie_algebra.md`](docs/foundations/lie_algebra.md) §11 - $`SE(3)`$: rotation and position together.
2. **Read** [`docs/filtering/eskf.md`](docs/filtering/eskf.md) - the error-state Kalman filter: a nominal state plus a small error state, injected into the nominal state and reset to zero after each measurement update.
3. **Run** [README.md §6](README.md#6-error-state-kalman-filter-eskf-vs-a-quaternion-in-the-state) - [`use_numpy/eskf_imu_position_fix.py`](use_numpy/eskf_imu_position_fix.py) - an IMU-driven ESKF, against leaving the biases out and against a quaternion in the EKF state.
4. **Read** [`docs/filtering/kf_ekf_iekf.md`](docs/filtering/kf_ekf_iekf.md) §3-§7 - the invariant EKF: what changes is *where the error is defined*.
5. **Read** [`docs/foundations/jacobian.md`](docs/foundations/jacobian.md) §9 - the same idea from the Jacobian side.
6. **Read** [`docs/filtering/left_right_invariant.md`](docs/filtering/left_right_invariant.md) - left- vs. right-invariant errors.
7. **Run** [README.md §5](README.md#5-point-cloud-pose-tracking-ekf-vs-invariant-ekf-vs-batch-gauss-newton-vs-ukf-vs-vanilla-kf) - [`use_numpy/pointcloud_pose_tracking.py`](use_numpy/pointcloud_pose_tracking.py) - EKF vs. IEKF vs. batch-GN vs. UKF vs. vanilla KF, head-to-head, with timing/memory numbers.
8. **Read** [`docs/filtering/pointcloud_pose_tracking_empirical_note.md`](docs/filtering/pointcloud_pose_tracking_empirical_note.md) - deliberately *after* running the script: why EKF/IEKF agree to round-off there, why the UKF differs mainly during the initial correction, and why the vanilla KF stays millimeters away.

**Checkpoint:** you're ready when you can explain why the ESKF resets its error state after injecting it, what the invariant EKF changes compared with the EKF, and why EKF and IEKF agree in the point-cloud benchmark.

### Phase 5 - Is the filter honest? Consistency and observability

**Why this step:** a filter reports an estimate *and* how sure it is. A filter that is too sure ignores good measurements and can't be trusted, and Track L keeps asking whether its filters are honest.

1. **Read** [`probability_basics.md`](docs/foundations/probability_basics.md) §7 - NEES: does the claimed uncertainty match the real errors?
2. **Run** `uv run python use_numpy/probability_basics.py --part nees` - an honest, an overconfident, and an underconfident estimator with the same errors: only the NEES tells them apart.
3. **Read** [`docs/foundations/noise_identification.md`](docs/foundations/noise_identification.md) - where a filter's IMU noise values come from.
4. **Run** [README.md §4](README.md#4-imu-noise-from-a-static-recording-allan-deviation) - [`use_numpy/imu_allan_variance.py`](use_numpy/imu_allan_variance.py) - the Allan deviation of a motionless IMU, and what a wrong noise-unit conversion does to how honest Phase 4's ESKF is (its NEES).
5. **Read** [`docs/foundations/observability.md`](docs/foundations/observability.md) - what no sensor can see.
6. **Run** [README.md §7](README.md#7-ekf-slam-observability-standard-ekf-vs-fej-vs-invariant-ekf) - [`use_numpy/observability_fej.py`](use_numpy/observability_fej.py) - a planar EKF-SLAM where the standard EKF's linearized model loses the global rotation as an unobservable direction and turns overconfident, against first-estimate Jacobians (FEJ) and the invariant EKF.

**Checkpoint:** you're ready when you can explain what NEES measures and what "consistent" means, why a noise density is not a per-sample standard deviation, and why the standard EKF turns overconfident about a direction no sensor can see.

**You've finished the main route.** Start Track L's first pass below.

---

## Track L - Hybrid and bio-inspired locomotion (the destination)

**Needs:** Phase 5. Every script in L1-L4 is a filter for a robot that plain smooth motion does not describe: something switches (a bounce, an anchor, a foot touching down), or, from L3.3 on, a constant error the encoders or camera hides (an offset, a drift, a delay). The question is what the filter may assume at a switch, or whether to estimate the constant error as a state. The groups go from one contact event to one anchor, to many legs; L5 adds wheels that slip.

**First pass:** L1.1 → L2.1 → L3.1 → L4.1 → L5.1, one step per group. Together they cover the track's main ideas: a reset, an anchor, many contacts, uncertain touchdowns, and slip. **Then** come back for the remaining steps, group by group, from top to bottom.

### L1 - Contact events: a reset in the motion

| Step | Read + run | The question | Builds on |
| --- | --- | --- | --- |
| L1.1 | [`hybrid_saltation_ekf.md`](docs/filtering/hybrid_saltation_ekf.md) + [README.md §19](README.md#19-saltation-matrix-ekf-tracking-a-point-mass-through-discrete-ground-contact-events) - [`saltation_matrix_ekf.py`](use_numpy/saltation_matrix_ekf.py) | A point mass bounces off the ground: how to carry the *covariance* (not just the mean) through that discrete reset | Phase 5 |
| L1.2 | [`invariant_saltation_ekf.md`](docs/filtering/invariant_saltation_ekf.md) + [README.md §20](README.md#20-invariant-ekf-on-se_23-through-a-bounce) - [`invariant_saltation_ekf.py`](use_numpy/invariant_saltation_ekf.py) (or its [`use_manif/`](use_manif/invariant_saltation_ekf.py) twin) | The same bounce on a Lie group: the right-invariant error with the saltation matrix, and why the bounce breaks the invariant filter's [trajectory independence](docs/glossary.md#group-affine-dynamics-trajectory-independence) | L1.1, plus [`left_right_invariant.md`](docs/filtering/left_right_invariant.md) and README §5 |

How to work through L1.1:
- **Read** the doc's derivation section first. It includes a formula that is correct for a different comparison than this filter needs; finite-difference checks show which one applies.
- **Predict, then run.** Before running, predict: after a bounce, is the naive filter (no saltation matrix) overconfident or underconfident, and is the saltation-corrected one better? Then run the script and check the Monte Carlo NEES plot against your prediction. The doc's results section explains the outcome.
- **Then** the doc's §8 adds uncertain contact *detection* timing (`--detect-time-bias`/`--detect-time-noise-std`). Predict again which filter suffers more as the timing jitter grows, then compare with §8.

**Checkpoint (L1):** you're ready when you can explain why a reset map's own Jacobian is not enough to carry the covariance through a bounce.

### L2 - Anchors: crawling on a held contact

Not a discontinuous reset like L1, but an ordinary switched-linear system: the question moves to the *measurement* side.

| Step | Read + run | The question | Builds on |
| --- | --- | --- | --- |
| L2.1 | [`inchworm_zupt_ekf.md`](docs/filtering/inchworm_zupt_ekf.md) §1-§4 + [README.md §21](README.md#21-inchworm-zupt-exploiting-a-known-anchordwell-schedule) - [`inchworm_zupt_ekf.py`](use_numpy/inchworm_zupt_ekf.py) | Using the free zero-velocity information (ZUPT) of a known anchor/dwell phase: correctly, always, or never | Phase 5 |
| L2.2 | Same doc, §5 - [`inchworm_zupt_zaru_ekf.py`](use_numpy/inchworm_zupt_zaru_ekf.py) | The gait in the plane with a heading: ZARU (zero angular rate) joins ZUPT on anchor ticks and keeps the unobserved heading from drifting at rest | L2.1 |
| L2.3 | Same doc, §6 - [`inchworm_slip_gate_ekf.py`](use_numpy/inchworm_slip_gate_ekf.py) | The anchor can slip, and a pad displacement sensor can veto the schedule's "v = 0" | L2.1 |
| L2.4 | [`inchworm_two_pad_ekf.md`](docs/filtering/inchworm_two_pad_ekf.md) + [README.md §23](README.md#23-two-pad-inchworm-trusting-the-anchored-pad-without-a-position-fix) - [`inchworm_two_pad_ekf.py`](use_numpy/inchworm_two_pad_ekf.py) | Two pads, turning, no position fix: a slip the filter does not track is never corrected, so only fusing the anchored pad's measured displacement keeps the estimate accurate and consistent | L2.3 |
| L2.5 | [`inchworm_two_pad_fusion_ekf.md`](docs/filtering/inchworm_two_pad_fusion_ekf.md) + [README.md §24](README.md#24-two-pad-inchworm-reading-both-pad-sensors) - [`inchworm_two_pad_fusion_ekf.py`](use_numpy/inchworm_two_pad_fusion_ekf.py) | The moving pad has a sensor too, and sensors fail: a real slip moves both pads so the two sensors agree, a fault makes them disagree, so a cross-check before fusing removes most faults (reading the second sensor blindly adds them) | L2.4 |
| L2.6 | [`friction_anisotropic_ekf.md`](docs/filtering/friction_anisotropic_ekf.md) + [README.md §25](README.md#25-friction-anisotropic-process-noise-fixed-vs-heading-aware-q) - [`friction_anisotropic_ekf.py`](use_numpy/friction_anisotropic_ekf.py) | A crawler on a friction-anisotropic pad: the isotropic/anisotropic and rotating-frame argument of `pointcloud_pose_tracking_empirical_note.md`, applied to *process* noise instead of measurement noise | Phase 5 |
| L2.7 | [`magnetic_inchworm_odometry_ekf.md`](docs/filtering/magnetic_inchworm_odometry_ekf.md) + [README.md §34](README.md#34-magnetic-climbing-inchworm-odometry-that-cannot-see-the-slide) - [`magnetic_inchworm_odometry_ekf.py`](use_numpy/magnetic_inchworm_odometry_ekf.py) | A magnetic climbing inchworm (CTU MinchW) has only its servos, and its holding pad slides sideways on a wall. What can servo odometry, a slip flag and an occasional outside fix tell about a slide no servo sees? | L2.3 |
| L2.8 | [`inchworm_slip_imm.md`](docs/filtering/inchworm_slip_imm.md) + [README.md §22](README.md#22-inchworm-anchor-slip-an-imm-instead-of-a-hard-gate) - [`inchworm_slip_imm.py`](use_numpy/inchworm_slip_imm.py) | Instead of gating the ZUPT on the pad reading, keep a probability that the body is still: two Kalman filters (still, moving) blended by an interacting multiple model, with and without the pad sensor | L2.3 |

**Checkpoint (L2):** you're ready when you can explain what a known anchor phase gives the filter for free, and why using it while the body still moves (or slips) hurts.

### L3 - Legs: many contacts at once

| Step | Read + run | The question | Builds on |
| --- | --- | --- | --- |
| L3.1 | [`hexapod_gait_ekf.md`](docs/filtering/hexapod_gait_ekf.md) + [README.md §26](README.md#26-hexapod-leg-odometry-what-the-gait-changes) - [`hexapod_gait_ekf.py`](use_numpy/hexapod_gait_ekf.py) | From one anchored segment to six legs: every stance foot is a velocity sensor, the gait decides how many there are, and a per-foot test catches a slipping leg | L2.1 |
| L3.2 | [`hexapod_contact_inekf.md`](docs/filtering/hexapod_contact_inekf.md) + [README.md §27](README.md#27-contact-aided-invariant-ekf-on-a-hexapod-which-filter-knows-yaw-is-unobservable) - [`hexapod_contact_inekf.py`](use_numpy/hexapod_contact_inekf.py) | The same hexapod in 3D with an IMU: each stance foot's position joins the state ($`SE_{K+2}(3)`$, Hartley et al.'s contact-aided invariant EKF), and a world-frame EKF on the same data shows why the invariant error matters: yaw is unobservable, and only the invariant EKF's covariance keeps it so | L3.1, L1.2 |
| L3.3 | [`hexapod_joint_kinematics_inekf.md`](docs/filtering/hexapod_joint_kinematics_inekf.md) + [README.md §29](README.md#29-joint-encoders-on-an-18-dof-hexapod-when-the-leg-kinematics-noise-model-matters) - [`hexapod_joint_kinematics_inekf.py`](use_numpy/hexapod_joint_kinematics_inekf.py) | Where the foot readings come from: joint encoders through the leg kinematics of an 18-DOF hexapod (HEBI Daisy). Does the pose-dependent shape of their error matter, and what do fixed joint zero offsets do to the filter? | L3.2 |
| L3.4 | [`hexapod_redundant_leg_inekf.md`](docs/filtering/hexapod_redundant_leg_inekf.md) + [README.md §31](README.md#31-four-joints-per-leg-which-joint-offsets-a-walking-hexapod-can-learn) - [`hexapod_redundant_leg_inekf.py`](use_numpy/hexapod_redundant_leg_inekf.py) | A fourth joint per leg (CTU HAntR): one joint combination never moves the foot at a given pose. Can walking still reveal its encoder offset, and which offsets stay hidden? | L3.3 |
| L3.5 | [`hexapod_camera_fusion_inekf.md`](docs/filtering/hexapod_camera_fusion_inekf.md) + [README.md §32](README.md#32-legs-plus-a-tracking-camera-how-to-use-a-pose-that-drifts) - [`hexapod_camera_fusion_inekf.py`](use_numpy/hexapod_camera_fusion_inekf.py) | Legs and an IMU cannot see position or yaw; a tracking camera reports both, but in a frame that drifts. How should the filter use it, what do its pose jumps do, and what does a late reading cost? | L3.2 |

**Checkpoint (L3):** you're ready when you can explain why every stance foot acts as a velocity sensor, and why yaw stays unobservable for legs plus an IMU.

### L4 - Gait timing: when does a foot land?

| Step | Read + run | The question | Builds on |
| --- | --- | --- | --- |
| L4.1 | [`gait_phase_ekf.md`](docs/filtering/gait_phase_ekf.md) + [README.md §28](README.md#28-gait-phase-which-feet-are-on-the-ground-when-touchdowns-are-uncertain) - [`gait_phase_ekf.py`](use_numpy/gait_phase_ekf.py) | The hexapod's liftoffs are commanded but its touchdowns are not: which feet may the body filter read when a foot can land late, and a detector can fire falsely? A schedule and a raw detector both read swinging feet; a learned touchdown offset or a small Kalman filter rejects triggers that come too early, if its acceptance window is the right width | L3.1 |
| L4.2 | [`gait_phase_torque_ekf.md`](docs/filtering/gait_phase_torque_ekf.md) + [README.md §30](README.md#30-torque-based-contact-detection-what-decides-it-and-when-the-gait-phase-helps) - [`gait_phase_torque_ekf.py`](use_numpy/gait_phase_torque_ekf.py) | The detector itself: on a robot that measures joint torque (HEBI Daisy), what decides when the contact detector fires, and does the gait-phase filter still help? | L4.1, L3.3 |

**Checkpoint (L4):** you're ready when you can explain why the body filter must not read a foot that is still swinging, and how a touchdown trigger that comes too early can be rejected.

### L5 - Wheels: slip in turns

The motion here is smooth (nothing switches); it sits in Track L with the CTU lab's other robots because, like an anchor or a foot, the wheels' contact with the ground decides what the odometry may assume.

| Step | Read + run | The question | Builds on |
| --- | --- | --- | --- |
| L5.1 | [`skid_steer_odometry_ekf.md`](docs/filtering/skid_steer_odometry_ekf.md) + [README.md §33](README.md#33-skid-steer-wheel-odometry-slip-as-a-filter-state) - [`skid_steer_odometry_ekf.py`](use_numpy/skid_steer_odometry_ekf.py) | A skid-steer robot (Clearpath Husky) turns slower than its wheel speeds say. How wrong is wheel odometry with the wrong slip coefficient, does a gyro fix it, and can the filter learn the coefficient? | Phase 5 |

**Checkpoint (L5):** you're ready when you can explain why wheel odometry misjudges a skid-steer turn, and what changes when the slip coefficient becomes a filter state.

A new locomotion type goes in the group whose question it shares, or in a new group (L6, ...) if it raises a new one.

---

## Smoothing track (optional, parallel)

**Needs:** Phase 2, plus [`lie_algebra.md`](docs/foundations/lie_algebra.md) §11 ($`SE(3)`$). This track keeps the whole history and re-solves it: the smoothing side of Phase 0's fork.

### S1 - Pose graphs

**Read:** [`docs/optimization/factor_graph.md`](docs/optimization/factor_graph.md), [`docs/optimization/levenberg_marquardt.md`](docs/optimization/levenberg_marquardt.md) §1-§9, [`docs/optimization/pose_graph_optimization.md`](docs/optimization/pose_graph_optimization.md) §1-§14.

**Run:** [README.md §12](README.md#12-3d-pose-graph-relaxation) - [`use_numpy/pose_graph.py`](use_numpy/pose_graph.py) - drift, loop closure, gauge-freedom anchoring, and the [chi-squared](docs/glossary.md#chi-square-distribution) convergence printout, all in one script.

**Then:** [`docs/optimization/robust_pose_graph.md`](docs/optimization/robust_pose_graph.md) + `pose_graph_optimization.md` §16 (the kernels) + [README.md §13](README.md#13-3d-pose-graph-with-false-loop-closures-plain-least-squares-vs-robust-kernels) - [`use_numpy/pose_graph_robust.py`](use_numpy/pose_graph_robust.py) (or its [`use_manif/`](use_manif/pose_graph_robust.py) twin) - the same pose graph over three laps with false loop closures: what plain least squares does with them, and how Huber, Cauchy and DCS limit the damage.

**Checkpoint:** you're ready when you can explain how a loop closure spreads the accumulated drift back along the trajectory, what the information matrix $`\Omega`$ decides, and why a false loop closure needs a robust kernel.

### S2 - Bundle adjustment

**Read:** [`docs/foundations/umeyama_alignment.md`](docs/foundations/umeyama_alignment.md) (needed to align monocular BA's gauge-free output to ground truth), [`docs/optimization/bundle_adjustment.md`](docs/optimization/bundle_adjustment.md) §1-§11 and §14.

**Run:** [README.md §14](README.md#14-bundle-adjustment-joint-pose--landmark-refinement) - [`use_numpy/bundle_adjustment.py`](use_numpy/bundle_adjustment.py) - landmarks-only vs. poses-only strawmen vs. full joint BA.

**Then read:** `bundle_adjustment.md` §13 (Local vs. Global BA).

**Run:** [README.md §15](README.md#15-local--global-bundle-adjustment-bounded-windows-vs-whole-map-re-solves) - [`use_numpy/bundle_adjustment_advanced.py`](use_numpy/bundle_adjustment_advanced.py) - bounded local windows plus periodic Global BA, cheirality guards, LM-with-step-rejection. Try the `--arc-span-deg 350 --n-keyframes 190` loop-closure variant afterward to see a real loop closure detected (Global BA's payoff there is seed-dependent).

**Checkpoint:** you're ready when you can explain why BA adjusts poses and landmarks jointly, and why a real system runs Local BA often and Global BA rarely.

### S3 - Incremental and bounded solving

**Read:** [`docs/optimization/isam_optimization.md`](docs/optimization/isam_optimization.md) §1-§14.

**Run:** [README.md §16](README.md#16-incremental-square-root-sam-vs-batch-pose-graph-solving) - [`use_numpy/pose_graph_incremental.py`](use_numpy/pose_graph_incremental.py) - batch re-solve vs. incremental square-root SAM (Givens updates for odometry edges, a full rebuild at loop closure, no reordering), reusing `pose_graph.py`'s own solver as the baseline.

**Then read:** [`docs/optimization/marginalization.md`](docs/optimization/marginalization.md) §1-§4 and §7-§9 - sliding-window smoothing: keep a bounded window and fold old states into a prior.

**Run:** [README.md §18](README.md#18-sliding-window-marginalization-bounded-memory-pose-graph-smoothing) - [`use_numpy/sliding_window_marginalization.py`](use_numpy/sliding_window_marginalization.py) - marginalizes the oldest pose out of a bounded window via the Schur complement as an odometry chain streams in, and compares it with an unbounded from-scratch re-solve across a sweep of trajectory lengths.
- Full-batch system size grows linearly (dense-matrix memory quadratically).
- The sliding window's size caps flat once the window first fills.
- Accuracy ties by construction (the optimum of a pure odometry chain is the dead-reckoned trajectory), so the memory bound is the real result.

**Checkpoint:** you're ready when you can explain why batch, incremental and sliding-window solving differ in cost but (without loop closures) not in accuracy, and why GN/LM is a separate choice from the solve strategy.

---

## Deep dives (optional)

Each one adds depth on one topic. Start one once its **Needs** are met.

### D1 - IMU preintegration and initialization

**Needs:** S1.

**Read:** [`docs/foundations/jacobian.md`](docs/foundations/jacobian.md) §11 (left/right Jacobians), then [`docs/optimization/imu_preintegration.md`](docs/optimization/imu_preintegration.md).

**Run:** [README.md §3](README.md#3-imu-preintegration-with-bias-jacobians) - [`use_numpy/imu_preintegration.py`](use_numpy/imu_preintegration.py) - bias-Jacobian bundle plus instant Taylor correction.

**Then read:** [`docs/frontend/vi_initialization.md`](docs/frontend/vi_initialization.md) - conceptual only (no script): what has to happen before a visual-inertial back-end can start.

### D2 - Front-ends

**Needs:** Phase 2, plus [`lie_algebra.md`](docs/foundations/lie_algebra.md) §11 ($`SE(3)`$, from Phase 4) for both docs' pose updates (`triangulation_pnp.md` also needs [`bundle_adjustment.md`](docs/optimization/bundle_adjustment.md) §5-§6.1 from S2; the LiDAR doc's §6 also needs S1).

- [`docs/frontend/triangulation_pnp.md`](docs/frontend/triangulation_pnp.md) + [README.md §9](README.md#9-perspective-n-point-pnp-recovering-a-camera-pose-from-known-3d-2d-correspondences) - [`use_numpy/pnp_estimation.py`](use_numpy/pnp_estimation.py) - the inverse of triangulation, using the same closed-form-then-Gauss-Newton recipe.
- [`docs/frontend/lidar_primitives.md`](docs/frontend/lidar_primitives.md) + [README.md §10](README.md#10-lidar-primitives-planes-and-cylinders-for-scan-registration) - [`use_numpy/lidar_primitives.py`](use_numpy/lidar_primitives.py) - the correspondence step that `pointcloud_pose_tracking.py` (Phase 4) skips: point-to-point and point-to-plane ICP against fitted planes and cylinders, one Gauss-Newton loop for all three. Its §5 shows a corridor's degeneracy and which method's Hessian admits it; its §6 uses S1's factor graph with planes as landmarks.

### D3 - Sparse solvers

**Needs:** S3.

**Read:**
- [`docs/optimization/sparse_cholesky_factorization.md`](docs/optimization/sparse_cholesky_factorization.md) - why real solvers exploit sparsity, what fill-in is, and why variable ordering matters (this repo's solves are dense).
- `bundle_adjustment.md` §12 (block sparsity and the Schur complement) and `marginalization.md` §5-§6 (fill-in from marginalization; the consistency issue that FEJ fixes).
- [`docs/optimization/elimination_tree.md`](docs/optimization/elimination_tree.md), [`docs/optimization/bayes_tree.md`](docs/optimization/bayes_tree.md).

**Run:** [README.md §17](README.md#17-bayes-tree-construction-and-affected-region-query) - [`use_numpy/bayes_tree_construction.py`](use_numpy/bayes_tree_construction.py) - builds the elimination tree underlying the Bayes tree over the same square-loop topology, and shows why a loop-closure edge invalidates the *whole* chain while an ordinary edge invalidates just 2 of 16 variables.

**Then read:** [`docs/optimization/isam2_optimization.md`](docs/optimization/isam2_optimization.md) - what selective relinearization and reordering buy beyond iSAM (conceptual only, no script).

### D4 - A filter that keeps its history: the RTS smoother

**Needs:** Phase 3 and S1.

[`docs/filtering_smoothing.md`](docs/filtering_smoothing.md) §8, last part + [README.md §11](README.md#11-rts-smoother-a-kalman-filter-its-smoother-and-batch-least-squares-on-one-data-set) - [`use_numpy/rts_smoother.py`](use_numpy/rts_smoother.py) - the same data estimated by a Kalman filter, its RTS smoother and batch least squares; the smoother's result equals the batch one.

### D5 - More background

**Needs:** the phase where each doc first appears.

- [`docs/foundations/quaternion.md`](docs/foundations/quaternion.md) - quaternions, the other common 3D-orientation representation. Read it before the `use_manif/` scripts, several of which build a `manif.SO3`/`manif.SE3` object from a quaternion.
- [`docs/filtering/extra_kf_variants.md`](docs/filtering/extra_kf_variants.md) §2-§13 and [`kf_ekf_iekf.md`](docs/filtering/kf_ekf_iekf.md) §9 - more Kalman-filter variants (square-root, ensemble, adaptive, robust, multi-rate, MSCKF) and filtering vs. optimization for SLAM.
- The remaining sections of [`lie_algebra.md`](docs/foundations/lie_algebra.md), [`jacobian.md`](docs/foundations/jacobian.md), [`nonlinear_least_square.md`](docs/optimization/nonlinear_least_square.md), [`gauss_newton.md`](docs/optimization/gauss_newton.md) and [`levenberg_marquardt.md`](docs/optimization/levenberg_marquardt.md).
- [`pose_graph_optimization.md`](docs/optimization/pose_graph_optimization.md) §15 - the full Lie-algebra math of the pose-graph error.

---

## After the main route

[`docs/slam_mental_map.md`](docs/slam_mental_map.md) shows where each step sits in the overall picture; read its later sections once you've finished the main route. [`docs/README.md`](docs/README.md) is the full index for non-sequential reference.

**Why this order:** each step's docs are mostly the prerequisites for its script. The main route ramps up as rotations → linearization → filters → filters on rotations → consistency, which is exactly what Track L's filters use. The smoothing track and the deep dives branch off where their prerequisites are met.
