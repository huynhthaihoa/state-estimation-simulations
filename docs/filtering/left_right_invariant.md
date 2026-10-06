# Left-invariant vs. Right-invariant errors in IEKF

What "left-invariant" and "right-invariant" errors mean in the **Invariant EKF**. This builds on the IEKF treatment in [kf_ekf_iekf.md §3-§5](kf_ekf_iekf.md#3-invariant-ekf-lets-respect-the-geometry-of-the-problem) (short overview: [extra_kf_variants.md §4](extra_kf_variants.md#4-invariant-ekf-iekf)).

---

## 1. The setup

State lives on a Lie group, e.g. a pose $X \in SE(3)$ mapping body coordinates to world coordinates. We have the true pose $X$ and our estimate $\hat X$. An ordinary EKF defines the error as the subtraction $X - \hat X$, but subtracting two poses leaves the manifold. IEKF uses group composition instead, in two natural ways:

$$\eta_L = \hat X^{-1} X \qquad \text{(left-invariant error)}$$
$$\eta_R = X \hat X^{-1} \qquad \text{(right-invariant error)}$$

**Convention note.** Barrau & Bonnabel (2017, Eqs. 5-6) define the inverses, $\eta^L = X^{-1}\hat X$ and $\eta^R = \hat X X^{-1}$. Inverting an error doesn't change which frame changes it ignores, so everything below holds for both; only the sign of the small error vector $\xi$ flips.

**Naming trap.** The perturbation $`X = \hat X\,\mathrm{Exp}(\xi)`$ used throughout this repo is what Solà et al. and the `manif` library call "right-plus" ($\hat X \oplus \xi$), because $\mathrm{Exp}(\xi)$ multiplies on the right. It is the **left**-invariant error: $\eta_L = \hat X^{-1}X = \mathrm{Exp}(\xi)$. "Right" in "right-plus" says where the increment is applied; "left" in "left-invariant" says which frame change the error ignores. This is why `run_iekf` in `pointcloud_pose_tracking.py` is a left-invariant filter.

---

## 2. The intuitive difference: *whose reference frame is the error measured in?*

**Left-invariant error ($\hat X^{-1}X$)** - the mismatch as seen **from inside our own estimated body frame**.

> "Sitting where I *think* I am, looking around - where does the true pose appear relative to me?"

**Right-invariant error ($X\hat X^{-1}$)** - the mismatch as seen **from a fixed observer in the world frame**.

> "Standing on the ground watching both the true robot and my estimate - how far apart are they, in map coordinates?"

---

## 3. Why "invariant" - and why left/right

Each error is unaffected by one specific kind of frame change, and the name says which:

- $\eta_L$ is unchanged if we left-multiply both $X$ and $\hat X$ by the same fixed transform $g$ (we redefine the *world/global* frame: rotate the map, shift the origin). That cancels out: $(g\hat X)^{-1}(gX) = \hat X^{-1}X$. So it's invariant to **global frame redefinition** - which makes sense, since a body-frame quantity shouldn't care how we labeled the world frame.
- $\eta_R$ is unchanged if we right-multiply both by $g$ (we redefine the *body* frame convention: recalibrate where the robot frame origin sits, e.g. sensor extrinsics). That cancels out too: $(Xg)(\hat Xg)^{-1} = X\hat X^{-1}$. So it's invariant to **body-frame redefinition**.

**Picture (re-drawing the map):**
- Redraw the world map: move the origin or rotate the axes. The true pose and our estimate both get new coordinates.
- But "where the truth sits as seen from my estimate" does not change, because both moved together. That is $`\eta_L`$.
- Now re-bolt the sensor to a different spot on the robot body (new body frame). Both poses shift in the same body way.
- The world positions now move by different amounts (each pose carries the offset in its own heading), so the plain distance between them changes.
- But the world-frame move that carries our estimate onto the truth, $`X\hat X^{-1}`$, does not change. That is $`\eta_R`$.

### 3.1 Same story, one level down: angular velocity on $SO(3)$

The pattern above is easiest to see on the rotation part alone, without the estimate/truth pair. For $R(t) \in SO(3)$, define body-frame and spatial (world-frame) angular velocity by

$$\hat\omega^b = R^\top\dot R \qquad \hat\omega^s = \dot RR^\top$$

(hat denotes the skew-symmetric matrix built from the corresponding vector). Redefining the world frame by a fixed rotation is left-multiplication $R\mapsto R_0R$; redefining the body frame is right-multiplication $R\mapsto RR_0$. Each redefinition cancels out of exactly one of the two velocities:

| | Left-mult. ($R\mapsto R_0R$) | Right-mult. ($R\mapsto RR_0$) |
|---|---|---|
| Physical meaning | Redefining the global/world frame | Redefining the local/body frame |
| Invariant quantity | Body-frame velocity $\hat\omega^b=R^\top\dot R$ | Spatial-frame velocity $\hat\omega^s=\dot RR^\top$ |
| Geometric term | Left-invariant vector field | Right-invariant vector field |

So $\hat\omega^b$ plays the same role as $\eta_L$ above (indifferent to how we label the world) and $\hat\omega^s$ plays the same role as $\eta_R$ (indifferent to how we label the body).

**Picture (gyro vs. ground watcher):**
- $`\hat\omega^b`$ is what a gyro bolted to the robot reads: how fast I spin, in my own axes.
- Rename the map axes ($`R\mapsto R_0R`$) and the gyro reading does not change: $`(R_0R)^\top R_0\dot R = R^\top\dot R`$.
- $`\hat\omega^s`$ is what an observer standing on the ground sees: how fast the robot spins, in map axes.
- Re-bolt the body axes ($`R\mapsto RR_0`$) and that view does not change: $`\dot RR_0\,R_0^\top R^\top = \dot RR^\top`$.

---

## 4. Why it actually matters (not just bookkeeping)

The whole point of IEKF is that with the *right* choice of error, the linearized error dynamics stop depending on the current state estimate. The Jacobians can still change over time with known inputs (for example the gyro and accelerometer readings), but not with the estimate, unlike an ordinary EKF's. That is what gives IEKF its convergence guarantees (Barrau & Bonnabel 2017, Theorem 1: estimate-independent, log-linear error propagation), and it also helps consistency.

Rule of thumb for picking one:

| Situation | Natural choice |
|---|---|
| Propagation driven by body-mounted sensors (IMU gyro/accel, wheel odometry) | either, for group-affine dynamics (see below) |
| World-frame measurement of a point fixed on the body: $`h(X) = X b`$ for a known body-frame point $b$ (GPS antenna, or this repo's `pointcloud_pose_tracking.py`, where $b = p_i$) | **left**-invariant |
| Body-mounted sensor measuring a landmark known in the world frame: $`h(X) = X^{-1} d`$ for a known world-frame point $d$ | **right**-invariant |

- **Propagation (row 1).** The two errors differ in how body-frame noise enters. With the **left** error it enters unchanged. With the **right** error it is multiplied by $`\mathrm{Ad}_{\hat X}`$, which depends on the estimate. Right-invariant filters such as Hartley et al.'s legged-robot IEKF accept that because their measurements are right-invariant.
- **$X b$ with the left error (row 2).** Substitute $`X = \hat X \eta_L`$ and form the residual in the body frame: $`\hat X^{-1} z - b = \eta_L b - b`$ plus rotated noise. With $`\eta_L = \exp(\xi)`$ the Jacobian is $`[\,I \;\; -b^\wedge\,]`$, which is constant. This is equivalent to `run_iekf`'s residual, `((z[k+1] - t_pred) @ R_pred - body_points)` in `pointcloud_pose_tracking.py`, which computes $`R_{\text{pred}}^\top(z - t_{\text{pred}}) - p_i`$ for all points at once.
- **$X^{-1}d$ with the right error (row 3).** Substitute $`X = \eta_R \hat X`$ and form the residual in the world frame: $`\hat X z - d = \eta_R^{-1} d - d`$ plus rotated noise. The Jacobian $`-[\,I \;\; -d^\wedge\,]`$ is constant too. This is the dual of row 2.
- **Why the pairing matters.** Rows 2 and 3 both involve a point and a world frame, but the measurement shapes differ ($X b$ vs. $X^{-1}d$) and pair with opposite errors (Barrau & Bonnabel call them left- and right-invariant observations). The wrong pairing does not give a constant Jacobian: $X b$ with $`X = \eta_R \hat X`$ gives $`h(X) = \eta_R \hat X b`$, whose Jacobian $`[\,I \;\; -(\hat X b)^\wedge\,]`$ contains the predicted point, hence the current estimate.
- **Same contrast, side by side.** The EKF-vs-IEKF table in [kf_ekf_iekf.md, EKF vs. IEKF equations side by side](kf_ekf_iekf.md#ekf-vs-iekf-equations-side-by-side) shows it for $Xb$: the world-frame (EKF) $H$ contains $`R_{\text{pred}}`$, the body-frame (IEKF) $H$ does not.
- **Repo check.** The point-cloud IEKF tests row 2. Its fixed $`H = [\,I \;\; -p_i^\wedge\,]`$ is in [pointcloud_pose_tracking_empirical_note.md §2](pointcloud_pose_tracking_empirical_note.md#2-ekf-vs-iekf-exact-by-construction), and [§1](pointcloud_pose_tracking_empirical_note.md#1-the-setup) of that note explains why the residual is left-invariant.
- **Propagation rarely decides.** For "group-affine" dynamics, such as IMU [dead-reckoning](../optimization/factor_graph.md#2-why-do-we-need-it) of pose and velocity, Barrau & Bonnabel (2017) show that both errors evolve independently of the estimate. So the measurement shape usually picks the error: papers choose whichever makes *their* sensor model's Jacobian trajectory-independent. That is the real design criterion, not a fixed rule.

---

## 5. References

1. Barrau, A., & Bonnabel, S. (2017). *The Invariant Extended Kalman Filter as a Stable Observer*. IEEE Transactions on Automatic Control, 62(4), 1797-1812. https://doi.org/10.1109/TAC.2016.2594085 - the main reference. Its Eqs. 5-6 define the errors as inverses of this doc's (§1); it also covers the observation shapes and group-affine dynamics in §4.
2. Bonnabel, S. (2007). *Left-invariant extended Kalman filter and attitude estimation*. 46th IEEE Conference on Decision and Control, 1027-1032. https://doi.org/10.1109/CDC.2007.4434662 - an early invariant-error formulation.
3. Hartley, R., Ghaffari, M., Eustice, R. M., & Grizzle, J. W. (2020). *Contact-aided invariant extended Kalman filtering for robot state estimation*. International Journal of Robotics Research, 39(4), 402-430. https://doi.org/10.1177/0278364919894385 - the right-invariant legged-robot filter (§4, propagation bullet).
4. Solà, J., Deray, J., & Atchuthan, D. (2018). *A micro Lie theory for state estimation in robotics*. arXiv:1812.01537. https://arxiv.org/abs/1812.01537 - the "right-plus" ($\oplus$) convention in §1, and the basis of `manif`.
