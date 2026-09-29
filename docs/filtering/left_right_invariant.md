# Left-invariant vs. Right-invariant errors in IEKF

An intuitive explanation of the difference between "left-invariant" and "right-invariant" error formulations in the **Invariant EKF**, building on the IEKF treatment in [kf_ekf_iekf.md §3-§5](kf_ekf_iekf.md#3-invariant-ekf-lets-respect-the-geometry-of-the-problem) (and its short overview in [extra_kf_variants.md §4](extra_kf_variants.md#4-invariant-ekf-iekf)).

---

## 1. The setup

State lives on a Lie group, e.g. a pose $X \in SE(3)$ mapping body coordinates to world coordinates. You have the true pose $X$ and your estimate $\hat X$. Ordinary EKF would define the error as a plain subtraction $X - \hat X$, but that doesn't make sense on a manifold - so IEKF defines the error using group composition instead. There are two natural ways to do it:

$$\eta_L = \hat X^{-1} X \qquad \text{(left-invariant error)}$$
$$\eta_R = X \hat X^{-1} \qquad \text{(right-invariant error)}$$

**Convention note.** Barrau & Bonnabel (2017, Eqs. 5-6) define the inverses, $\eta^L = X^{-1}\hat X$ and $\eta^R = \hat X X^{-1}$. Inverting an error doesn't change which frame changes it ignores, so everything below holds for both; only the sign of the small error vector $\xi$ flips.

**Naming trap.** The perturbation $X = \hat X\,\mathrm{Exp}(\xi)$ used throughout this repo is what Solà et al. and the `manif` library call "right-plus" ($\hat X \oplus \xi$), because $\mathrm{Exp}(\xi)$ multiplies on the right. It is the **left**-invariant error: $\eta_L = \hat X^{-1}X = \mathrm{Exp}(\xi)$. "Right" in "right-plus" says where the increment is applied; "left" in "left-invariant" says which frame change the error ignores. This is why `run_iekf` in `pointcloud_pose_tracking.py` is a left-invariant filter.

---

## 2. The intuitive difference: *whose reference frame is the error measured in?*

**Left-invariant error ($\hat X^{-1}X$)** - the mismatch as seen **from inside your own estimated body frame**.

> "Sitting where I *think* I am, looking around - where does the true pose appear relative to me?"

**Right-invariant error ($X\hat X^{-1}$)** - the mismatch as seen **from a fixed observer in the world frame**.

> "Standing on the ground watching both the true robot and my estimate - how far apart are they, in map coordinates?"

---

## 3. Why "invariant" - and why left/right

Each error is unaffected by one specific kind of frame change, and the name tells you which:

- $\eta_L$ is unchanged if you left-multiply both $X$ and $\hat X$ by the same fixed transform $g$ (i.e. you redefine the *world/global* frame - rotate your map, shift your origin). That cancels out: $(g\hat X)^{-1}(gX) = \hat X^{-1}X$. So it's invariant to **global frame redefinition** - which makes sense, since a body-frame quantity shouldn't care how you labeled the world frame.
- $\eta_R$ is unchanged if you right-multiply both by $g$ (i.e. you redefine the *body* frame convention - recalibrate where "robot frame origin" sits, e.g. sensor extrinsics). That cancels out too: $(Xg)(\hat Xg)^{-1} = X\hat X^{-1}$. So it's invariant to **body-frame redefinition**.

### 3.1 Same story, one level down: angular velocity on $SO(3)$

The pattern above is easiest to see on the rotation part alone, without the estimate/truth pair. For $R(t) \in SO(3)$, define body-frame and spatial (world-frame) angular velocity by

$$\hat\omega^b = R^\top\dot R \qquad \hat\omega^s = \dot RR^\top$$

(hat denotes the skew-symmetric matrix built from the corresponding vector). Redefining the world frame by a fixed rotation is left-multiplication $R\mapsto R_0R$; redefining the body frame is right-multiplication $R\mapsto RR_0$. Each redefinition cancels out of exactly one of the two velocities:

| | Left-mult. ($R\mapsto R_0R$) | Right-mult. ($R\mapsto RR_0$) |
|---|---|---|
| Physical meaning | Redefining the global/world frame | Redefining the local/body frame |
| Invariant quantity | Body-frame velocity $\hat\omega^b=R^\top\dot R$ | Spatial-frame velocity $\hat\omega^s=\dot RR^\top$ |
| Geometric term | Left-invariant vector field | Right-invariant vector field |

So $\hat\omega^b$ plays the same role as $\eta_L$ above (indifferent to how you label the world) and $\hat\omega^s$ plays the same role as $\eta_R$ (indifferent to how you label the body).

---

## 4. Why it actually matters (not just bookkeeping)

The whole point of IEKF is that with the *right* choice of error, the linearized error dynamics stop depending on the current state estimate. The Jacobians can still change over time with known inputs (for example the gyro and accelerometer readings), but not with the estimate, unlike an ordinary EKF's. That's what gives IEKF its better consistency properties.

Rule of thumb for picking one:

| Situation | Natural choice |
|---|---|
| Propagation driven by body-mounted sensors (IMU gyro/accel, wheel odometry) | either works for group-affine dynamics (see below); they differ in how the body-frame noise enters. With the **left** error it enters unchanged; with the **right** error it is multiplied by $`\mathrm{Ad}_{\hat X}`$, which depends on the estimate. Right-invariant filters such as Hartley et al.'s legged-robot IEKF accept that because their measurements are right-invariant |
| A world-frame measurement of a point fixed on the body ($`h(X) = X b`$ for a known body-frame point $b$). Examples: GPS measuring the antenna location, or this repo's `pointcloud_pose_tracking.py`, where an external sensor measures the object's known body-frame points $p_i$ in the world frame | **left**-invariant. Substituting $`X = \hat X \eta_L`$ and forming the residual in the body frame gives $`\hat X^{-1} z - b = \eta_L b - b`$ plus rotated noise. With $`\eta_L = \exp(\xi)`$ its Jacobian is $`[\,I \;\; -b^\wedge\,]`$, which is constant. This is exactly `run_iekf`'s residual `T_pred⁻¹.act(z_i) - p_i` |
| A body-mounted sensor measures, in its own frame, a landmark whose position is known in the world frame ($`h(X) = X^{-1} d`$ for a known world-frame point $d$) | **right**-invariant. Substituting $`X = \eta_R \hat X`$ and forming the residual in the world frame gives $`\hat X z - d = \eta_R^{-1} d - d`$ plus rotated noise. Its Jacobian is $`-[\,I \;\; -d^\wedge\,]`$, which is constant too. This is the dual of the row above |

These last two rows look alike, since both involve a point and a world frame. But they are different measurement shapes ($X b$ vs. $X^{-1}d$), and they pair with opposite errors (Barrau & Bonnabel, 2017, call them left- and right-invariant observations). Pairing a shape with the other error does not give a constant Jacobian. For example, $X b$ with $`X = \eta_R \hat X`$ gives $`h(X) = \eta_R \hat X b`$, whose Jacobian $`[\,I \;\; -(\hat X b)^\wedge\,]`$ contains the predicted point and so the current estimate. The point-cloud script's IEKF is the repo's own check of the $X b$ row. Its fixed $`H = [\,I \;\; -p_i^\wedge\,]`$ appears in [pointcloud_pose_tracking_empirical_note.md §2](pointcloud_pose_tracking_empirical_note.md#2-ekf-vs-iekf-exact-by-construction), and [§1](pointcloud_pose_tracking_empirical_note.md#1-the-setup) of that note explains why its residual is left-invariant.

Propagation alone usually doesn't decide the choice. For "group-affine" dynamics, such as IMU [dead-reckoning](../optimization/factor_graph.md#2-why-do-we-need-it) of pose and velocity, Barrau & Bonnabel (2017) show that both the left- and right-invariant errors evolve independently of the estimate. So the measurement shape is often what picks between them.

In practice, papers pick whichever one makes *their* sensor model's Jacobian trajectory-independent - that's the real design criterion, not a fixed rule. But the mental picture to keep is simple: **left = error viewed from your own cockpit, right = error viewed from a fixed point on the ground.**

---

## 5. References

1. Barrau, A., & Bonnabel, S. (2017). *The Invariant Extended Kalman Filter as a Stable Observer*. IEEE Transactions on Automatic Control, 62(4), 1797-1812. https://doi.org/10.1109/TAC.2016.2594085 - the main reference for the left-/right-invariant error framework this whole doc explains (its Eqs. 5-6 define the errors as inverses of this doc's, see §1), including the left-/right-invariant observation shapes in §4's table and the group-affine dynamics property in the paragraph after it.
2. Bonnabel, S. (2007). *Left-invariant extended Kalman filter and attitude estimation*. 46th IEEE Conference on Decision and Control, 1027-1032. https://doi.org/10.1109/CDC.2007.4434662 - an early formulation of the invariant-error idea that reference 1 builds on.
3. Hartley, R., Ghaffari, M., Eustice, R. M., & Grizzle, J. W. (2020). *Contact-aided invariant extended Kalman filtering for robot state estimation*. International Journal of Robotics Research, 39(4), 402-430. https://doi.org/10.1177/0278364919894385 - the right-invariant legged-robot filter in §4's first table row.
4. Solà, J., Deray, J., & Atchuthan, D. (2018). *A micro Lie theory for state estimation in robotics*. arXiv:1812.01537. https://arxiv.org/abs/1812.01537 - the "right-plus" ($\oplus$) convention in §1's naming note, and the basis of the `manif` library.
