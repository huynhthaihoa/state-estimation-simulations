# Left-invariant vs. Right-invariant errors in IEKF

An intuitive explanation of the difference between "left invariant" and "right invariant" error formulations in the **Invariant EKF**, building on the IEKF section of [extra_kf_variants.md](extra_kf_variants.md).

---

## 1. The setup

State lives on a Lie group, e.g. a pose $X \in SE(3)$ mapping body coordinates to world coordinates. You have the true pose $X$ and your estimate $\hat X$. Ordinary EKF would define the error as a plain subtraction $X - \hat X$, but that doesn't make sense on a manifold - so IEKF defines the error using group composition instead. There are two natural ways to do it:

$$\eta_L = \hat X^{-1} X \qquad \text{(left-invariant error)}$$
$$\eta_R = X \hat X^{-1} \qquad \text{(right-invariant error)}$$

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

The whole point of IEKF is that with the *right* choice of error, the linearized error dynamics stop depending on the current state estimate - the Jacobians become constant (or nearly so) instead of changing at every timestep like ordinary EKF. That's what gives IEKF its better consistency properties.

Rule of thumb for picking one:

| Situation | Natural choice |
|---|---|
| Propagation driven by body-mounted sensors (IMU gyro/accel, wheel odometry) | often **left**-invariant - error dynamics driven by body-frame noise become state-independent |
| A world-frame measurement of a point fixed on the body ($`h(X) = X b`$ for a known body-frame point $b$). Examples: GPS measuring the antenna location, or this repo's `pointcloud_pose_tracking.py`, where an external sensor measures the object's known body-frame points $p_i$ in the world frame | **left**-invariant. Substituting $`X = \hat X \eta_L`$ and forming the residual in the body frame gives $`\hat X^{-1} z - b = \eta_L b - b`$ plus rotated noise. With $`\eta_L = \exp(\xi)`$ its Jacobian is $`[\,I \;\; -b^\wedge\,]`$, which is constant. This is exactly `run_iekf`'s residual `T_pred⁻¹.act(z_i) - p_i` |
| A body-mounted sensor measures, in its own frame, a landmark whose position is known in the world frame ($`h(X) = X^{-1} d`$ for a known world-frame point $d$) | **right**-invariant. Substituting $`X = \eta_R \hat X`$ and forming the residual in the world frame gives $`\hat X z - d = \eta_R^{-1} d - d`$ plus rotated noise. Its Jacobian is $`-[\,I \;\; -d^\wedge\,]`$, which is constant too. This is the dual of the row above |

These last two rows look alike, since both involve a point and a world frame. But they are different measurement shapes ($X b$ vs. $X^{-1}d$), and they pair with opposite errors (Barrau & Bonnabel, 2017, call them left- and right-invariant observations). Pairing a shape with the other error does not give a constant Jacobian. For example, $X b$ with $`X = \eta_R \hat X`$ gives $`h(X) = \eta_R \hat X b`$, whose Jacobian $`[\,I \;\; -(\hat X b)^\wedge\,]`$ contains the predicted point and so the current estimate. An earlier version of this table had these pairings swapped. The point-cloud script's IEKF is the repo's own check of the $X b$ row. Its fixed $`H = [\,I \;\; -p_i^\wedge\,]`$ appears in [pointcloud_pose_tracking_empirical_note.md §2](pointcloud_pose_tracking_empirical_note.md#2-ekf-vs-iekf-exact-by-construction), and [appendix A.2](pointcloud_pose_tracking_empirical_note.md#a2-invariance-and-equivariance) of that note explains why its residual is left-invariant.

Propagation alone usually doesn't decide the choice. For "group-affine" dynamics, such as IMU [dead-reckoning](../optimization/factor_graph.md#2-why-do-we-need-it) of pose and velocity, Barrau & Bonnabel (2017) show that both the left- and right-invariant errors evolve independently of the estimate. So the measurement shape is often what picks between them.

In practice, papers pick whichever one makes *their* sensor model's Jacobian trajectory-independent - that's the real design criterion, not a fixed rule. But the mental picture to keep is simple: **left = error viewed from your own cockpit, right = error viewed from a fixed point on the ground.**

---

## 5. References

1. Barrau, A., & Bonnabel, S. (2017). *The Invariant Extended Kalman Filter as a Stable Observer*. IEEE Transactions on Automatic Control, 62(4), 1797-1812. https://doi.org/10.1109/TAC.2016.2594085 - the original left-/right-invariant error framework this whole doc explains, including the left-/right-invariant observation shapes in §4's table and the group-affine dynamics property in the paragraph after it.
