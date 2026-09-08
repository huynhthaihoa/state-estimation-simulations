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

---

## 4. Why it actually matters (not just bookkeeping)

The whole point of IEKF is that with the *right* choice of error, the linearized error dynamics stop depending on the current state estimate - the Jacobians become constant (or nearly so) instead of changing at every timestep like ordinary EKF. That's what gives IEKF its better consistency properties.

Rule of thumb for picking one:

| Situation | Natural choice |
|---|---|
| Propagation driven by body-mounted sensors (IMU gyro/accel, wheel odometry) | often **left**-invariant - error dynamics driven by body-frame noise become state-independent |
| Measurements given directly in the world frame (GPS, known landmark positions), or "group-affine" IMU dead-reckoning models (pose+velocity, as in legged-robot contact-aided IEKF) | often **right**-invariant - makes propagation/measurement Jacobians state-independent instead |

In practice, papers pick whichever one makes *their* sensor model's Jacobian trajectory-independent - that's the real design criterion, not a fixed rule. But the mental picture to keep is simple: **left = error viewed from your own cockpit, right = error viewed from a fixed point on the ground.**
