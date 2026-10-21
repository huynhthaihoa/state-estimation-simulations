# Jacobian intuitive explanation

The **Jacobian** has a very intuitive meaning:

> **A Jacobian tells us how a small change in the input causes a small change in the output.**

---

## 1. Start with a simple function

Suppose:

$$y = 3x$$

If $x$ changes by a tiny amount $\Delta x = 0.1$, then $\Delta y = 3(0.1) = 0.3$. The derivative

$$\frac{dy}{dx} = 3$$

tells us:

> **"If I move $x$ a little bit, $y$ moves about 3 times as much."**

For a function with **one input and one output**, we call this a derivative.

---

## 2. Now imagine multiple inputs and outputs

Suppose:

```math
\begin{bmatrix} y_1 \\ 
y_2 \end{bmatrix}=f\left(\begin{bmatrix} x_1 \\ 
x_2 \end{bmatrix}\right)
```

Now every output depends on every input:

```text
       x1 ──────┐
                ├──→ y1
       x2 ──────┘

       x1 ──────┐
                ├──→ y2
       x2 ──────┘
```

We want four sensitivities: how each of $y_1, y_2$ changes when each of $x_1, x_2$ changes. To make this concrete, suppose:

$$y_1 = x_1 + x_2^2 \qquad y_2 = x_1 x_2$$

Then:

$$\frac{\partial y_1}{\partial x_1} = 1 \qquad \frac{\partial y_1}{\partial x_2} = 2x_2 \qquad \frac{\partial y_2}{\partial x_1} = x_2 \qquad \frac{\partial y_2}{\partial x_2} = x_1$$

We put all those derivatives into a matrix:

```math
{J =\begin{bmatrix} \frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} \\ 
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} \end{bmatrix} = \begin{bmatrix} 1 & 2x_2 \\ 
x_2 & x_1 \end{bmatrix}}
```

That's the **Jacobian**. Notice the convention: each **row** is one output ($y_i$), each **column** is one input ($x_j$) - $J_{ij} = \partial y_i/\partial x_j$.

---

## 3. Think of it as a "sensitivity map"

Suppose our robot's state is (called $`p`$ here, $`x`$ from §5 on):

```math
p = \begin{bmatrix} x \\ 
y \\ 
\theta \end{bmatrix}
```

and our camera produces some measurement:

```math
{z = \begin{bmatrix} u \\ 
v \end{bmatrix}}
```

The Jacobian might look like:

```math
H = \begin{bmatrix} \frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} & \frac{\partial u}{\partial \theta} \\
\frac{\partial v}{\partial x} & \frac{\partial v}{\partial y} & \frac{\partial v}{\partial \theta} \end{bmatrix}
```

This tells us:

> **If I slightly perturb the robot's $(x, y, \theta)$, how much will the camera measurement $(u, v)$ change?**

So we can think of the Jacobian as a **sensitivity table**. This $H$ is schematic: its entries depend on the camera model. §4 works out a concrete one, differentiating with respect to the 3D point rather than the pose, so it is labeled $J$.

---

## 4. Example: camera projection

Suppose a 3D point $P = (X, Y, Z)$ is projected onto the image:

$$u = f_x\frac{X}{Z} \qquad v = f_y\frac{Y}{Z}$$

Here $`f_x, f_y`$ are the camera's focal lengths (in pixels) and $`u, v`$ are pixel coordinates (the principal point is ignored). This is nonlinear because of the division by $Z$. We ask:

> "What happens to the image point if the 3D point moves slightly?"

The Jacobian answers that to first order. For example:

- $\partial u/\partial X = f_x/Z$: if $X$ changes slightly, $u$ changes by about $f_x/Z$ times that amount.
- $\partial u/\partial Z = -f_x X/Z^2$: moving the point in depth changes its image position, by an amount that depends on its current depth and horizontal position.

Putting every partial derivative together gives the full Jacobian:

```math
{J = \begin{bmatrix} \frac{\partial u}{\partial X} & \frac{\partial u}{\partial Y} & \frac{\partial u}{\partial Z} \\ 
\frac{\partial v}{\partial X} & \frac{\partial v}{\partial Y} & \frac{\partial v}{\partial Z} \end{bmatrix} =\begin{bmatrix} f_x/Z & 0 & -f_x X/Z^2 \\ 
0 & f_y/Z & -f_y Y/Z^2 \end{bmatrix}}
```

Notice the zeros: $\partial u/\partial Y = 0$ and $\partial v/\partial X = 0$, because horizontal image position ($u$) doesn't depend on vertical 3D position ($Y$) at all, and vice versa for $v$ and $X$. This is the projection Jacobian a visual-SLAM or bundle-adjustment system computes at every reprojected point, with $(X, Y, Z)$ the point in *camera* coordinates. It is then chained with the Jacobians of the world-to-camera transform to get derivatives with respect to the pose and the world-frame point.

---

## 5. Why does EKF need it?

Suppose the real system is nonlinear: $z = h(x)$. The [EKF (Extended Kalman Filter)](../glossary.md#ekf-extended-kalman-filter) still evaluates the exact $h$ at its estimate, to predict the measurement. What it can't do is push a whole probability distribution through $h$ exactly: the Kalman gain and covariance update need a linear model. So for those it asks:

> "Around my current estimate $\hat{x}$, can I approximate this nonlinear function with a linear one?"

**Intuition:** a Gaussian pushed through a straight line stays a Gaussian. Pushed through a curve, it bends out of shape.
- Toy case: $`y = 2x + 1`$ turns $`\mathcal{N}(0,1)`$ into $`\mathcal{N}(1,4)`$, still a bell.
- Toy case: $`y = x^2`$ turns $`\mathcal{N}(0,1)`$ into a lopsided shape that is never negative.
- So the EKF first flattens the curve into a line (the Jacobian), then uses the Kalman formulas.

The Jacobian gives us exactly that approximation:

$$h(x) \approx h(\hat{x}) + H(x - \hat{x}) \qquad H = \left.\frac{\partial h}{\partial x}\right|_{\hat{x}}$$

So the Jacobian is the **local slope of a multidimensional nonlinear function**.

---

## 6. The mountain picture

Imagine standing somewhere on a complicated mountain we can't see in full.

We can still ask local questions: "If I take one tiny step north, what happens to my altitude?" and the same for east. The Jacobian collects those local slopes, so around our position the mountain is approximated by a **flat plane**, the **local linear approximation**. The figure shows the one-dimensional version: the curve is the actual terrain, and the dashed tangent line is the approximation around our position.

![Left: cos θ with its tangent at the estimate θ̂ = 30° and ±5° and ±40° bands; the tangent is 0.3 cm off at +5° and 17.5 cm off at +40°. Right: Monte Carlo distribution of cos θ versus the EKF's Gaussian, matching for σ = 5° and clearly different for σ = 40°](../../assets/ekf_linearization_concept.png)

*Figure: $`\cos\theta`$ is the x-step of a robot moving 1 m at heading $`\theta`$. The tangent at the estimate (30°) is an excellent stand-in for the curve when the uncertainty is small (5°: 0.3 cm gap) and a poor one when it is large (40°: 17.5 cm gap).*

The EKF does exactly this around its current estimate.

---

## 7. Jacobian $\approx$ "local translator"

The mountain idea of §6, written as an equation. Let $\delta x$ be a slight change of the robot state ("I slightly changed my robot state"). The Jacobian converts it into $\delta z$, the approximate change in the sensor measurement:

$$\boxed{\delta z \approx H\delta x}$$

That is why Jacobians appear throughout EKF, [ESKF](../glossary.md#eskf-error-state-kalman-filter) (error-state Kalman filter), [IEKF](../glossary.md#iekf-invariant-ekf) (invariant EKF), bundle adjustment, [nonlinear least squares](../glossary.md#nonlinear-least-squares), factor graphs, visual odometry and SLAM.

---

## 8. One subtle point: Jacobian is LOCAL

Suppose $y = x^2$, so $dy/dx = 2x$. At $x = 1$ we get $J = 2$, and at $x = 10$ we get $J = 20$. The Jacobian changes depending on **where we are**, so it describes the local behavior of a nonlinear function.

This is also a key weakness of the EKF: if the function is highly nonlinear, the local approximation might become poor. How poor is "poor" depends on how far from the linearization point we need it to hold: for a filter, that's the region its uncertainty covers. [linear_nonlinear.md](../filtering/linear_nonlinear.md) shows how to tell linear from nonlinear, and how to check whether the nonlinearity matters at our uncertainty level.

---

## 9. Connecting this back to IEKF

The pieces above connect directly to the IEKF.

### KF

The system $x_{k+1} = Fx_k$ is already linear, so no Jacobian is necessary.

### EKF

The system $x_{k+1} = f(x_k)$ is nonlinear (here $`f`$ is the generic state-transition function again), so we linearize:

$$F_k = \left.\frac{\partial f}{\partial x}\right|_{\hat{x}_k}$$

The Jacobian tells us:

> "How does the nonlinear system behave **locally around my current estimate**?"

### IEKF

Same basic idea of linearization, but with a crucial difference:

> **The perturbation/error is defined according to the geometry and invariance of the system.**

So instead of blindly asking "What is the derivative with respect to my state vector?", we carefully ask:

> **"What is the derivative with respect to the appropriate local perturbation on the state manifold?"**

Here the **state manifold** is the curved (but locally flat) space the state lives in, so plain addition does not apply ([glossary](../glossary.md#manifold)). That's one reason Lie groups (the set of rotations or poses, which can be composed; [glossary](../glossary.md#lie-group)) and Jacobians become so tightly connected in modern SLAM.

A manifold perturbation on its own isn't yet what makes a filter **invariant**: error-state filters such as the ESKF and MEKF (multiplicative EKF, which perturbs attitude by composition) already perturb on the manifold. The IEKF goes one step further and picks a left- or right-[invariant](../glossary.md#invariant-and-equivariant) error. For **group-affine** dynamics ([glossary](../glossary.md#group-affine-dynamics-trajectory-independence)) this makes its linearization independent of the current estimate (see [lie_algebra.md §11.1](lie_algebra.md#111-extended-poses-pose-plus-velocity)). See [kf_ekf_iekf.md §5](../filtering/kf_ekf_iekf.md#5-the-really-important-difference-how-do-you-define-error) and [left_right_invariant.md](../filtering/left_right_invariant.md).

---

## 10. The one-sentence intuition

> **The Jacobian is a multidimensional "local sensitivity map": it tells us how small changes in one thing approximately translate into small changes in another thing.**

In the EKF specifically:

> **The Jacobian is the tool that lets us temporarily turn a nonlinear system into a locally linear one.**

---

## 11. Left and right Jacobians: sensitivity on a curved space

§9 mentioned that manifold-aware filters (error-state filters and the IEKF) perturb the state by composition instead of subtracting vectors. This section makes that concrete for rotations and explains where the **left Jacobian** $J_l$ and **right Jacobian** $J_r$ come from.

### 11.1 Why plain addition breaks

Everywhere above, perturbing an input meant simple addition: $x \to x + \delta x$. That works because $x$ lives in a vector space.

A rotation $R \in SO(3)$ ([glossary](../glossary.md#so3-and-se3): the set of 3D rotation matrices) does not. There is no such thing as $R + \delta R$ - the result generally isn't even a valid rotation. Instead, a small perturbation is a **rotation vector** $\delta\varphi \in \mathbb{R}^3$: its direction is the rotation axis and its length the angle in radians. The exponential map $\mathrm{Exp}(\delta\varphi)$ ([lie_algebra.md §8](lie_algebra.md#8-the-exponential-map), [glossary](../glossary.md#exp-and-log-maps)) turns it into the rotation matrix that turns by that angle about that axis, and that matrix is then **composed** (matrix-multiplied) with $R$:

```math
R' = \mathrm{Exp}(\delta\varphi)\,R \qquad \text{or} \qquad R' = R\,\mathrm{Exp}(\delta\varphi)
```

Both are valid ways to perturb $R$ - but they are not the same $R'$ in general, because matrix multiplication doesn't commute. That non-commutativity, together with the nonlinearity of $\mathrm{Exp}$ (§11.2), is the reason the left and right versions differ. On a vector space the distinction never comes up, because addition commutes.

### 11.2 The actual question being asked

Suppose we have a rotation built from $\varphi$, i.e. $\mathrm{Exp}(\varphi)$, and we nudge its argument: $\mathrm{Exp}(\varphi + \delta\varphi)$. Because $\mathrm{Exp}$ is a curved, nonlinear map (just like $u = f_x X/Z$ in §4), we cannot simply distribute the addition. We can still ask the usual question - "how does the output change?" - and express the answer as a small rotation composed onto $\mathrm{Exp}(\varphi)$, either on the left or on the right:

```math
\mathrm{Exp}(\varphi+\delta\varphi)\;\approx\;\mathrm{Exp}\big(J_l(\varphi)\,\delta\varphi\big)\cdot\mathrm{Exp}(\varphi) \qquad\text{(left)}
```

```math
\mathrm{Exp}(\varphi+\delta\varphi)\;\approx\;\mathrm{Exp}(\varphi)\cdot\mathrm{Exp}\big(J_r(\varphi)\,\delta\varphi\big) \qquad\text{(right)}
```

Here $`\varphi`$ is the rotation vector of the starting rotation, and $`\delta\varphi`$ changes $`\varphi`$ itself, unlike §11.1 where $`\delta\varphi`$ is applied to $`R`$ by composition. $`J_l(\varphi)\,\delta\varphi`$ (or $`J_r(\varphi)\,\delta\varphi`$) is that change re-expressed as a §11.1-style perturbation. $`J_l`$ and $`J_r`$ are $`3\times3`$ matrices (closed forms in §11.4).

So $J_l$ and $J_r$ are still Jacobians in the sense of §10, applied to the exponential map instead of a vector-valued function, and reported by which side the resulting perturbation attaches to.

### 11.3 The intuition: compass vs. steering wheel

- **Right Jacobian** $J_r(\varphi)$: the extra rotation multiplies $\mathrm{Exp}(\varphi)$ on the right, so it's a rotation about the object's **own current (body) axes**. Like turning a car's steering wheel a bit more - "a bit more" is relative to however the car is already pointed.
- **Left Jacobian** $J_l(\varphi)$: the extra rotation multiplies $\mathrm{Exp}(\varphi)$ on the left, so it's a rotation about the **fixed world/global axes**. Like someone nudging our heading by a fixed compass bearing, regardless of which way we're facing.

Near the identity ($\varphi \to 0$) there's no rotation yet to disagree about "whose frame," so $J_l(0) = J_r(0) = I$. This is the flat-tangent-plane picture of [lie_algebra.md §6](lie_algebra.md#6-lie-algebra-zoom-in-locally): at the identity the manifold looks flat. The distinction only shows up when we linearize away from the identity, around some existing rotation.

### 11.4 Closed form for $SO(3)$

With $\theta = \lVert\varphi\rVert$ and $[\varphi]_\times$ the skew-symmetric matrix of $\varphi$ (so that $`[\varphi]_\times b = \varphi \times b`$ for any 3-vector $`b`$):

$$J_l(\varphi) = I + \frac{1-\cos\theta}{\theta^2}[\varphi]_\times + \frac{\theta-\sin\theta}{\theta^3}[\varphi]_\times^2$$

$$J_r(\varphi) = I - \frac{1-\cos\theta}{\theta^2}[\varphi]_\times + \frac{\theta-\sin\theta}{\theta^3}[\varphi]_\times^2$$

**Where the formulas come from.** In short, $`J_l(\varphi)`$ is the average of the **rotation matrices** along the path from $`I`$ to $`\mathrm{Exp}(\varphi)`$. Write $`\Phi = [\varphi]_\times`$ and $`B = [\delta\varphi]_\times`$, so $`\mathrm{Exp}(\varphi) = e^{\Phi}`$. The left perturbation of §11.2 is $`Y = \mathrm{Exp}(\varphi+\delta\varphi)\,\mathrm{Exp}(\varphi)^{-1}`$. Grow it along $`t \in [0, 1]`$ as $`Y(t) = e^{t(\Phi+B)}e^{-t\Phi}`$, which starts at $`Y(0) = I`$:

```math
\dot Y(t) = e^{t(\Phi+B)}\,B\,e^{-t\Phi} \approx e^{t\Phi}\,B\,e^{-t\Phi} = \big[\mathrm{Exp}(t\varphi)\,\delta\varphi\big]_\times
```

The approximation drops terms of second order in $`\delta\varphi`$, and the last step uses $`R\,[b]_\times R^\top = [R\,b]_\times`$. Integrating from 0 to 1:

```math
Y(1) \approx I + \big[J_l(\varphi)\,\delta\varphi\big]_\times \approx \mathrm{Exp}\big(J_l(\varphi)\,\delta\varphi\big), \qquad J_l(\varphi) = \int_0^1 \mathrm{Exp}(t\varphi)\,dt
```

The closed form follows by integrating the Rodrigues form $`\mathrm{Exp}(t\varphi) = I + \frac{\sin t\theta}{\theta}\Phi + \frac{1-\cos t\theta}{\theta^2}\Phi^2`$ term by term: $`\int_0^1 \sin t\theta\,dt = \frac{1-\cos\theta}{\theta}`$ and $`\int_0^1 (1-\cos t\theta)\,dt = \frac{\theta-\sin\theta}{\theta}`$ give the two coefficients above (after dividing by $`\theta`$ and $`\theta^2`$ from the Rodrigues form). The same steps with the perturbation on the right, $`\mathrm{Exp}(\varphi)^{-1}\mathrm{Exp}(\varphi+\delta\varphi)`$, give $`J_r(\varphi) = \int_0^1 \mathrm{Exp}(-t\varphi)\,dt`$.

Useful identities, with $`R(\varphi) = \mathrm{Exp}(\varphi)`$ the rotation matrix of $`\varphi`$. Each follows in one line from the two integrals: flip the sign of $`\varphi`$; use $`\mathrm{Exp}(t\varphi)^\top = \mathrm{Exp}(-t\varphi)`$; and use $`\mathrm{Exp}(\varphi)\,\mathrm{Exp}(-t\varphi) = \mathrm{Exp}\big((1-t)\varphi\big)`$ (all three checked numerically against `so3_right_jacobian` and `so3_exp` in `use_numpy/lie_utils.py`):

```math
J_r(\varphi) = J_l(-\varphi) \qquad J_r(\varphi) = J_l(\varphi)^\top \qquad J_l(\varphi) = R(\varphi)\,J_r(\varphi)
```

The last one is the frame-conversion identity: $R(\varphi)$ turns a body-frame perturbation into a world-frame one, so it bridges $J_r$ and $J_l$, consistent with the compass/steering-wheel picture.

### 11.5 A worked example: 90° yaw

We take $\varphi = (0, 0, \theta)$, a pure rotation about $z$, with $\theta = \pi/2$. For a single-axis rotation, $[\varphi]_\times^2 = \theta^2(kk^\top - I) = \text{diag}(-\theta^2, -\theta^2, 0)$ with $k=(0,0,1)$, which keeps the algebra clean. Plugging $\theta=\pi/2$ ($\cos\theta=0$, $\sin\theta=1$) into the formulas above gives:

```math
J_l \approx \begin{bmatrix} 0.637 & -0.637 & 0\\ 
0.637 & 0.637 & 0\\ 
0 & 0 & 1\end{bmatrix} \qquad J_r \approx \begin{bmatrix} 0.637 & 0.637 & 0\\ 
-0.637 & 0.637 & 0\\ 
0 & 0 & 1 \end{bmatrix}
```

Notice $J_r = J_l^\top$, as the identity predicts. Plugging $\theta \to 0$ into the same formulas collapses both matrices to $I$, confirming §11.3.

### 11.6 Where this actually matters

- **[IMU preintegration](../glossary.md#imu-preintegration):** the effect of a small change in gyroscope bias is naturally expressed in the sensor's own (body) frame, so bias-correction Jacobians in preintegration use $J_r$ - worked out end-to-end in [imu_preintegration.md](../optimization/imu_preintegration.md).
- **Covariance propagation on the manifold:** a rotation's uncertainty is a covariance on the tangent vector $\delta\varphi$, and what it numerically means depends on whether $\delta\varphi$ is defined via $`R\,\mathrm{Exp}(\delta\varphi)`$ (right) or $`\mathrm{Exp}(\delta\varphi)\, R`$ (left).
  - Converting between them is a multiplication by $R$ (the [adjoint](lie_algebra.md#112-the-adjoint-moving-a-small-motion-between-frames), which for rotations is $R$ itself): $`\mathrm{Exp}(\delta\varphi_{\text{left}})\,R = R\,\mathrm{Exp}(\delta\varphi_{\text{right}})`$ gives $`\delta\varphi_{\text{left}} = R\,\delta\varphi_{\text{right}}`$.
  - Hence $`\Sigma_{\text{left}} = R\,\Sigma_{\text{right}}R^\top`$: a change of frame, not a change of the underlying uncertainty.
  - The §11.4 identity $`J_l = R\,J_r`$ follows from the same fact.
- **Factor graphs/bundle adjustment on $SE(3)$** ([glossary](../glossary.md#so3-and-se3)): residual Jacobians w.r.t. a pose depend on which perturbation convention (left vs. right) the library uses, and the update must apply the step with the same convention. Mixing them gives wrong step directions: the optimization can converge slowly, converge to the wrong place, or diverge, all while the code runs without error.

### 11.7 One-sentence intuition

> **$J_l$ and $J_r$ are the same "local sensitivity map" as every other Jacobian in this document, applied to the exponential map on a Lie group: a small change to the input can be reported in the world frame (left) or in the object's own frame (right), and the two answers only agree at $\varphi = 0$.**

---

## 12. References

1. Solà, J., Deray, J., & Atchuthan, D. (2018). *A micro Lie theory for state estimation in robotics*. arXiv:1812.01537. https://doi.org/10.48550/arXiv.1812.01537 - the standard modern reference for $J_l$/$`J_r`$ and the left/right perturbation conventions behind §11, written by (among others) the author of the `manif` library that this repo's `use_manif/` scripts are built on.
