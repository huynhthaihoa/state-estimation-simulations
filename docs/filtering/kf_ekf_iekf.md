
# What are the difference between the standard Kalman Filter, Extended Kalman Filter, and Invariant Extended Kalman Filter? 

The most intuitive way to understand them is to start with one idea:

> **A Kalman Filter is basically a smart way of combining "what I predicted" with "what I measured."**

The three filters differ mainly in **what kind of system they assume** and **how they deal with nonlinear motion**.

---

## 1. Standard Kalman Filter: "Everything is nicely linear"

Imagine you're tracking a car.

You have:

* a previous estimate: `car is at x = 10 m, moving at 5 m/s`
* a motion model: `after 1 second, it should be around x = 15 m`
* a sensor measurement: `GPS says x = 14 m`

The Kalman Filter asks:

> **How much should I trust my prediction versus my measurement?**

If GPS is noisy, you might say:

> Prediction: 70%

> GPS: 30%

So the result might be around `14.7 m`.

The important assumption is that the relationship between the state and its evolution is **linear**.

For example:

$$x_{k+1} = Fx_k + w$$

and measurement:

$$z_k = Hx_k + v$$

where:

* $x$: state
* $z$: measurement
* $F$: linear motion model
* $H$: linear measurement model
* $w$: process noise (everything motion model $F$ doesn't capture like unmodeled dynamics, wind gusts, wheel slip, IMU bias drift, etc.)
* $v$: measurement noise (sensor imperfection like GPS jitter, camera pixel noise, IMU noise, etc.)

### Intuition

Think of the standard KF as:

> **"I have a straight ruler, and the world behaves approximately like a straight line."**

It's elegant and mathematically clean, but it doesn't work directly for things like rotations, camera poses, or nonlinear robot dynamics.

---

## 2. Extended Kalman Filter: "The world is nonlinear, so I'll approximate it locally"

Now suppose your robot is moving.

Its state might be: $x = [x,y,\theta]$

where $\theta$ is its orientation.

The motion might look like:
 - $x_{k+1}=x_k+v\cos\theta\,\Delta t$
 - $y_{k+1}=y_k+v\sin\theta\,\Delta t$

This is **nonlinear** because of the $\cos\theta$ and $\sin\theta$.

A standard KF can't handle this directly.

So the EKF says:

> "Okay, the system is nonlinear, but perhaps I can pretend it's linear **around my current estimate**."

It uses a **Jacobian** to locally approximate the nonlinear function.

Imagine a curved road:

```text
                 actual nonlinear function
                       /
                     /
                   /
                __/
             __/
          __/
       __/

        EKF approximation
       /
      /
     /
```

The EKF essentially says:

> "I don't need to understand the whole curve.

> I just need to approximate the curve around where I currently am."

Mathematically, if: $x_{k+1}=f(x_k,u_k)+w$

the EKF computes: $F_k = \frac{\partial f}{\partial x}$

That's the Jacobian.

Then it uses this local linear approximation inside the normal Kalman equations.

### Intuition

So:

#### KF

> "My world is linear."

#### EKF

> "My world is nonlinear, but I'll locally pretend it's linear."

This works surprisingly well and is probably one of the most widely used nonlinear filtering approaches.

---

## 3. But here's the problem with EKF

This becomes particularly important in **robotics and SLAM**.

Suppose your robot has a pose: $X=(R,p)$

where:

* $R$: rotation
* $p$: position

Rotations are not ordinary vectors.

In 2D, composing rotations is easy: rotating by 90° and then another 90° just adds the angles, $90^\circ + 90^\circ = 180^\circ$, and the order doesn't matter.

In 3D it's not that simple. Rotate an object 90° about the x-axis, then 90° about the y-axis, and you get a different orientation than doing it in the reverse order:

$$R_x(90^\circ)\,R_y(90^\circ) \neq R_y(90^\circ)\,R_x(90^\circ)$$

3D rotations don't commute, and there's no simple "add the numbers" operation the way there is for positions on a line. That's the sense in which rotations aren't ordinary vectors.

They live on a mathematical structure called a **Lie group**, typically:
 - $SO(3)$ for rotations
 - $SE(3)$ for 3D poses

But representing rotations awkwardly isn't actually the deep problem - plain EKF already has workarounds for that (quaternions, Euler angles, small additive perturbations around them; this is often called MEKF, and it's been standard in spacecraft attitude estimation since the 1960s-70s; see Lefferts, Markley, and Shuster, 1982).

The real problem is more subtle: **the EKF's Jacobians are evaluated at the current, drifting state estimate**, and that estimate is different every run and every step. Each time the filter linearizes at a slightly different point, it linearizes the system's symmetries slightly differently too. Concretely, this can make the EKF inject spurious information into directions of the state that should be unobservable - for example, a global orientation offset that no sensor can actually see - leaving the filter overconfident (inconsistent) in exactly those directions. This consistency problem, studied by Huang, Mourikis, and Roumeliotis (2010) for EKF-SLAM and VINS, is the concrete motivation Barrau and Bonnabel (2017) built the Invariant EKF to address.

And this is where the Invariant EKF becomes interesting.

---

## 4. Invariant EKF: "Let's respect the geometry of the problem"

The key insight is:

> **Don't treat a robot pose like an ordinary vector if it isn't one.**

This sounds subtle, but it is extremely important.

Imagine two robots:

```text
Robot A                    Robot B

     ↑                          ↑
     |                          |
     ●                          ●
```

Suppose both robots observe the **same relative motion**.

If you change the global coordinate system:

```text
Before                         After changing frame

     ↑                               ↗
     ●                               ●
```

the physical situation hasn't changed.

The robot doesn't suddenly behave differently just because **you changed your coordinate system**.

A good estimator should therefore behave consistently under these transformations.

This property is related to **invariance**.

---

## 5. The big difference

Here's perhaps the most useful mental model:

| Filter   | Mental model                                                                                   |
| -------- | ---------------------------------------------------------------------------------------------- |
| **KF**   | "The world is linear."                                                                         |
| **EKF**  | "The world is nonlinear, so I'll linearize it."                                                |
| **IEKF** | "The world is nonlinear, so I'll linearize it in a way that respects its geometry/symmetries." |

The IEKF is therefore **not simply "EKF but more accurate."**

It's a different way of constructing the error and performing the linearization.

---

## 6. The really important difference: how do you define error?

This is probably the most important concept for understanding IEKF.

Suppose your estimated robot orientation is: $\hat R$ and the true orientation is: $R$

### Ordinary EKF thinking

You might think: $R-\hat R$

But subtraction doesn't naturally make sense for rotations.

So EKF implementations often represent the rotation using something like Euler angles or a local perturbation and then linearize.

### IEKF thinking

Instead of asking:

> "What is the difference between these two rotations?"

in an ordinary vector sense, we ask:

> **"What rotation would transform my estimate into the true rotation?"**

For example:

$$R = \hat R \exp(\delta\theta^\wedge)$$

where:

$\delta\theta$ is a small rotation error.

This is much more natural geometrically.

You are essentially saying:

> "My estimate is here. What small motion on the rotation manifold would take me to the truth?"

That's a much more natural representation for robot motion.

**A caveat worth remembering**: using $R = \hat R \exp(\delta\theta^\wedge)$ instead of $R - \hat R$ is, by itself, just a *manifold* or *multiplicative* error representation - it is not automatically "invariant." Plenty of widely-used filters (ESKF, MEKF, the error-state formulations behind most VIO pipelines) already define their error this way without being IEKFs. What earns the name **invariant** is a further, more specific choice: picking the error so that it's built from the group action itself (left- or right-invariant), which makes the *linearized error dynamics* independent of the current state estimate. That's the property that actually fixes the consistency problem from §3 - not the exp/log notation on its own.

---

## 7. Why is this useful for SLAM?

This becomes especially relevant to your PhD topic because you're dealing with **SLAM and state estimation for robots with unusual motion**.

Consider a robot state:

$${X = \begin{bmatrix} R & p & v & b_g & b_a \end{bmatrix} }$$

where:

* $R$: orientation
* $p$: position
* $v$: velocity
* $b_g$: gyroscope bias
* $b_a$: accelerometer bias

An ordinary EKF has to repeatedly calculate Jacobians around the current estimate.

But the system has important geometric symmetries.

For example, changing the global reference frame shouldn't fundamentally change the robot's physical behavior.

The IEKF tries to construct the estimation error so that these symmetries are handled naturally.

That can give you much better behavior when the system is highly nonlinear.

---

## 8. A nice analogy

Imagine you're navigating Prague with a map. 🙂

### KF

You say:

> "The streets are straight."

Very easy.

### EKF

You realize:

> "Okay, streets curve. I'll approximate the street as a straight line around my current location."

Then you repeatedly update your approximation.

### IEKF

You realize:

> "Wait. I'm navigating on a **map**, and the important thing is how my position and orientation transform relative to the map."

So instead of pretending the map is just a giant flat vector space, you explicitly use the geometry of the navigation problem.

---

## 9. Another way to visualize the progression

Think about this:

```text
                 State estimation
                       │
                       ▼
              ┌─────────────────┐
              │ Linear system?  │
              └────────┬────────┘
                       │ Yes
                       ▼
                  Standard KF
                       
                       │ No
                       ▼
              ┌─────────────────┐
              │ Nonlinear       │
              │ system          │
              └────────┬────────┘
                       │
                       ▼
                     EKF
             "Linearize locally"
                       │
                       │
          But does the system have
          important geometric
          symmetries?
                       │
                       ▼
                     IEKF
       "Linearize while respecting
              the geometry"
```

---

## 10. The subtle but important point

One misconception is:

> **KF → EKF → IEKF = three levels of accuracy.**

That's not quite right.

It's better to think:

> **KF → different mathematical assumptions**

> **EKF → generic nonlinear approximation**

> **IEKF → geometry-aware nonlinear approximation**

In fact, an IEKF can sometimes have **better convergence and consistency properties** than a conventional EKF because the linearization is aligned with the system's inherent symmetries.

**A concrete data point**: in this repo's own point-cloud pose-tracking benchmark ([use_numpy/pointcloud_pose_tracking.py](../../use_numpy/pointcloud_pose_tracking.py), [use_manif/pointcloud_pose_tracking.py](../../use_manif/pointcloud_pose_tracking.py), EKF and IEKF were verified to produce **exactly identical** corrections under isotropic point-noise covariance - proven algebraically (the body-frame and world-frame residual/Jacobian pairs differ only by a per-point rotation that cancels exactly out of the Kalman gain) and confirmed numerically to ~1e-14 precision. IEKF's real, measured advantage there wasn't accuracy - both filters converged to the same estimate - it was **speed**: IEKF's fixed Jacobian made it ~35% faster per step than EKF, at identical memory. That's a good concrete reminder that "geometry-aware" doesn't always mean "more accurate" - sometimes it means "cheaper to compute the same answer," and the accuracy gap only opens up once the noise model or system structure breaks the symmetry that made them equivalent here.

---

## 11. If you remember only three sentences

I'd remember these:

### KF

> **"I can model my system as linear."**

### EKF

> **"My system is nonlinear, but I'll approximate it as linear around my current estimate."**

### IEKF

> **"My system is nonlinear and has geometric structure, so I'll define my errors and linearization in a way that respects that structure."**

And for your SLAM research, the last distinction is particularly important: **robot pose is not just a vector; it has geometry.** That's one of the main reasons IEKF is so attractive for inertial navigation, visual-inertial estimation, and SLAM.

---

## 12. Other prominent variants worth keeping in mind

KF, EKF, and IEKF aren't the whole landscape. A few others come up constantly in robotics/SLAM/VIO work, so it's worth knowing what each one buys you.

### UKF - "Don't linearize the function, sample around it instead"

Instead of a Jacobian, the UKF pushes a small, deterministic set of "sigma points" through the *exact* nonlinear function and reconstructs the mean/covariance from the results. No derivatives needed, and it captures curvature a first-order Jacobian misses - often more accurate than EKF at similar cost.

### ESKF - "Keep a big slow-changing state and a tiny error state that's always near zero"

Already touched on in §6: the ESKF splits the state into a "nominal" state (integrated directly with the raw nonlinear equations, never filtered) and a small additive "error state" that stays close to zero and is safe to linearize. It's the de facto backbone of most modern VIO pipelines.

### MSCKF - "Don't put landmarks in the state at all"

Instead of estimating landmark positions jointly with the pose (as EKF-SLAM does), the MSCKF keeps a sliding window of past camera poses and uses each landmark's multi-view geometry to build a constraint *between those poses*, then discards the landmark. Cost scales with the number of poses in the window, not the number of landmarks - the reason it scales to large scenes.

### Iterated EKF - "One linearization pass isn't enough - relinearize at the new estimate, repeat"

At each update, relinearize the measurement model around the *updated* state estimate and repeat until convergence - essentially Gauss-Newton applied inside a single EKF update. Reduces linearization error for measurements that are very nonlinear or very informative.

**A naming collision worth flagging**: "IEKF" is used in the literature for *both* Iterated EKF and Invariant EKF - two unrelated ideas that happen to share an acronym. This doc uses IEKF exclusively for Invariant EKF (§3-§6); when reading other material, check which one is meant.

### EqF - "Generalize the invariant idea to any symmetry, not just matrix Lie groups"

The Invariant EKF (§4-§6) exploits the geometry of matrix Lie groups like SO(3) and SE(3). The Equivariant Filter takes the same core idea - pick errors and linearizations that respect the system's symmetry - and extends it to systems whose natural symmetry group isn't a matrix Lie group at all. A reasonable "what comes after IEKF" pointer if you want to go further.

---

## 13. References

1. Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems*. Journal of Basic Engineering, 82(1), 35–45. https://doi.org/10.1115/1.3662552 - the original formulation behind §1's standard KF.
2. Huang, G. P., Mourikis, A. I., & Roumeliotis, S. I. (2010). *Observability-based Rules for Designing Consistent EKF SLAM Estimators*. International Journal of Robotics Research, 29(5), 502–528. https://journals.sagepub.com/doi/10.1177/0278364909353640 - the consistency analysis behind §3's "spurious information into unobservable directions" argument.
3. Lefferts, E. J., Markley, F. L., & Shuster, M. D. (1982). *Kalman Filtering for Spacecraft Attitude Estimation*. Journal of Guidance, Control, and Dynamics, 5(5), 417–429. https://doi.org/10.2514/3.56190 - the classic MEKF reference behind §3's spacecraft-attitude aside.
4. Barrau, A., & Bonnabel, S. (2017). *The Invariant Extended Kalman Filter as a Stable Observer*. IEEE Transactions on Automatic Control, 62(4), 1797–1812. https://arxiv.org/abs/1410.1465 - the paper that introduces the IEKF discussed in §3, §4, and §6.
5. Solà, J., Deray, J., & Atchuthan, D. (2018). *A micro Lie theory for state estimation in robotics*. arXiv:1812.01537. https://arxiv.org/abs/1812.01537 - background for the exp/log/hat (Lie group) notation used in §6, and the theoretical basis of the `manif` library used in this repo's own benchmark referenced in §10.
6. Julier, S. J., & Uhlmann, J. K. (1997). *New extension of the Kalman filter to nonlinear systems*. Proc. SPIE 3068, Signal Processing, Sensor Fusion, and Target Recognition VI, 182–193. https://doi.org/10.1117/12.280797 - the original UKF paper, referenced in §12.
7. Solà, J. (2017). *Quaternion kinematics for the error-state Kalman filter*. arXiv:1711.02508. https://arxiv.org/abs/1711.02508 - the standard ESKF reference, §6 and §12.
8. Mourikis, A. I., & Roumeliotis, S. I. (2007). *A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation*. ICRA 2007, 3565–3572. https://doi.org/10.1109/ROBOT.2007.364024 - the original MSCKF paper, §12.
9. Bell, B. M., & Cathey, F. W. (1993). *The iterated Kalman filter update as a Gauss-Newton method*. IEEE Transactions on Automatic Control, 38(2), 294–297. https://doi.org/10.1109/9.250476 - the Iterated EKF reference, §12.
10. van Goor, P., Hamel, T., & Mahony, R. (2020). *Equivariant Filter (EqF)*. arXiv:2010.14666. https://arxiv.org/abs/2010.14666 - the EqF paper, §12.
