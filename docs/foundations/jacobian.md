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

Suppose our robot's state is:

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

So we can think of the Jacobian as a **sensitivity table**. This $H$ is schematic: its entries depend on the camera model. Section 4 works out a concrete one, differentiating with respect to the 3D point rather than the pose, so it is labeled $J$.

---

## 4. Example: camera projection

Suppose a 3D point $P = (X, Y, Z)$ is projected onto the image:

$$u = f\frac{X}{Z} \qquad v = f\frac{Y}{Z}$$

This is nonlinear because of the division by $Z$. We ask:

> "What happens to the image point if the 3D point moves slightly?"

The Jacobian answers that to first order. For example:

- $\partial u/\partial X = f/Z$: if $X$ changes slightly, $u$ changes by about $f/Z$ times that amount.
- $\partial u/\partial Z = -fX/Z^2$: moving the point in depth changes its image position, by an amount that depends on its current depth and horizontal position.

Putting every partial derivative together gives the full Jacobian:

```math
{J = \begin{bmatrix} \frac{\partial u}{\partial X} & \frac{\partial u}{\partial Y} & \frac{\partial u}{\partial Z} \\ 
\frac{\partial v}{\partial X} & \frac{\partial v}{\partial Y} & \frac{\partial v}{\partial Z} \end{bmatrix} =\begin{bmatrix} f/Z & 0 & -fX/Z^2 \\ 
0 & f/Z & -fY/Z^2 \end{bmatrix}}
```

Notice the zeros: $\partial u/\partial Y = 0$ and $\partial v/\partial X = 0$, because horizontal image position ($u$) doesn't depend on vertical 3D position ($Y$) at all, and vice versa for $v$ and $X$. This is the projection Jacobian a visual-SLAM or bundle-adjustment system computes at every reprojected point, with $(X, Y, Z)$ the point in *camera* coordinates; it is then chained with the Jacobians of the world-to-camera transform to get derivatives with respect to the pose and the world-frame point.

---

## 5. Why does EKF need it?

Suppose the real system is nonlinear: $z = h(x)$. The EKF still evaluates the exact $h$ at its estimate, to predict the measurement. What it can't do is push a whole probability distribution through $h$ exactly: the Kalman gain and covariance update need a linear model. So for those it asks:

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

Imagine standing somewhere on a complicated mountain we can't see in full:

```text
                 /\       /\
        /\      /  \_____/  \
   ____/  \____/             \____
```

We can still ask local questions: "If I take one tiny step north, what happens to my altitude?" and the same for east. The Jacobian collects those local slopes, so around our position the mountain is approximated by a **flat plane**, the **local linear approximation**:

```text
             actual mountain
                /
               /
              /
             /________
            /
       you ●
```

The EKF does exactly this around its current estimate.

---

## 7. Jacobian $\approx$ "local translator"

The mountain idea of Section 6, written as an equation. Let $\delta x$ be a slight change of the robot state ("I slightly changed my robot state"). The Jacobian converts it into $\delta z$, the approximate change in the sensor measurement:

$$\boxed{\delta z \approx H\delta x}$$

That is why Jacobians are everywhere in EKF, ESKF, IEKF, bundle adjustment, nonlinear least squares, factor graphs, visual odometry and SLAM.

---

## 8. One subtle point: Jacobian is LOCAL

Suppose $y = x^2$, so $dy/dx = 2x$. At $x = 1$ we get $J = 2$, and at $x = 10$ we get $J = 20$. The Jacobian changes depending on **where we are**, so it describes the local behavior of a nonlinear function.

This is also a key weakness of the EKF: if the function is highly nonlinear, the local approximation might become poor. How poor is "poor" depends on how far from the linearization point we need it to hold: for a filter, that's the region its uncertainty covers. [linear_nonlinear.md](../filtering/linear_nonlinear.md) shows how to tell linear from nonlinear, and how to check whether the nonlinearity matters at our uncertainty level.

---

## 9. Connecting this back to IEKF

The pieces above connect back to the IEKF directly.

### KF

The system $x_{k+1} = Fx_k$ is already linear, so no Jacobian is necessary.

### EKF

The system $x_{k+1} = f(x_k)$ is nonlinear, so we linearize:

$$F_k = \left.\frac{\partial f}{\partial x}\right|_{\hat{x}_k}$$

The Jacobian tells us:

> "How does the nonlinear system behave **locally around my current estimate**?"

### IEKF

Same basic idea of linearization, but with a crucial difference:

> **The perturbation/error is defined according to the geometry and invariance of the system.**

So instead of blindly asking "What is the derivative with respect to my state vector?", we carefully ask:

> **"What is the derivative with respect to the appropriate local perturbation on the state manifold?"**

That's one reason Lie groups and Jacobians become so tightly connected in modern SLAM.

A manifold perturbation on its own isn't yet what makes a filter *invariant*: error-state filters such as the ESKF and MEKF already perturb on the manifold. The IEKF goes one step further and picks the specific error that ignores a change of world or body frame (left- or right-invariant), which is what makes its linearization independent of the current estimate for group-affine dynamics. See [kf_ekf_iekf.md §5](../filtering/kf_ekf_iekf.md#5-the-really-important-difference-how-do-you-define-error) and [left_right_invariant.md](../filtering/left_right_invariant.md).

---

## 10. The one-sentence intuition

> **The Jacobian is a multidimensional "local sensitivity map": it tells us how small changes in one thing approximately translate into small changes in another thing.**

In the EKF specifically:

> **The Jacobian is the tool that lets us temporarily turn a nonlinear system into a locally linear one.**

---

## 11. Left and right Jacobians: sensitivity on a curved space

Section 9 mentioned that manifold-aware filters (error-state filters and the IEKF) perturb the state by composition instead of just subtracting vectors. This section makes that concrete for rotations, and explains where the **left Jacobian** $J_l$ and **right Jacobian** $J_r$ come from.

### 11.1 Why plain addition breaks

Everywhere above, perturbing an input meant simple addition: $x \to x + \delta x$. That works because $x$ lives in a vector space.

A rotation $R \in SO(3)$ does not. There is no such thing as $R + \delta R$ - the result generally isn't even a valid rotation. Instead, a small perturbation $\delta\varphi \in \mathbb{R}^3$ is turned into a rotation via the exponential map, $\text{Exp}(\delta\varphi)$, and then **composed** (matrix-multiplied) with $R$:

```math
R' = \text{Exp}(\delta\varphi)\,R \qquad \text{or} \qquad R' = R\,\text{Exp}(\delta\varphi)
```

Both are valid ways to perturb $R$ - but they are not the same $R'$ in general, because matrix multiplication doesn't commute. That non-commutativity, together with the nonlinearity of $\text{Exp}$ (Section 11.2), is the reason the left and right versions differ. On a vector space the distinction never comes up, because addition commutes.

### 11.2 The actual question being asked

Suppose we have a rotation built from $\varphi$, i.e. $\text{Exp}(\varphi)$, and we nudge its argument: $\text{Exp}(\varphi + \delta\varphi)$. Because $\text{Exp}$ is a curved, nonlinear map (just like $u = fX/Z$ in Section 4), we cannot simply distribute the addition. We can still ask the usual sensitivity question - "how does the output change?" - and express the answer as a small rotation composed onto $\text{Exp}(\varphi)$, either on the left or on the right:

```math
\text{Exp}(\varphi+\delta\varphi)\;\approx\;\text{Exp}\big(J_l(\varphi)\,\delta\varphi\big)\cdot\text{Exp}(\varphi) \qquad\text{(left)}
```

```math
\text{Exp}(\varphi+\delta\varphi)\;\approx\;\text{Exp}(\varphi)\cdot\text{Exp}\big(J_r(\varphi)\,\delta\varphi\big) \qquad\text{(right)}
```

So $J_l$ and $J_r$ are still what Section 10 says a Jacobian is - a local sensitivity map, "small change in input → small change in output" - applied to the exponential map instead of a vector-valued function, and reported by which side the resulting perturbation attaches to.

### 11.3 The intuition: compass vs. steering wheel

- **Right Jacobian** $J_r(\varphi)$: the extra rotation is applied after $\text{Exp}(\varphi)$, so it's expressed in the object's **own current (body) frame**. Like turning a car's steering wheel a bit more - "a bit more" is relative to however the car is already pointed.
- **Left Jacobian** $J_l(\varphi)$: the extra rotation is applied before, so it's expressed in the **fixed world/global frame**. Like someone nudging our heading by a fixed compass bearing, regardless of which way we're facing.

Near the identity ($\varphi \to 0$) there's no rotation yet to disagree about "whose frame," so $J_l(0) = J_r(0) = I$. This is the flat-tangent-plane picture of Section 6: at the identity the manifold looks flat. The distinction only shows up when we linearize away from the identity, around some existing rotation.

### 11.4 Closed form for $SO(3)$

With $\theta = \lVert\varphi\rVert$ and $[\varphi]_\times$ the skew-symmetric matrix of $\varphi$:

$$J_l(\varphi) = I + \frac{1-\cos\theta}{\theta^2}[\varphi]_\times + \frac{\theta-\sin\theta}{\theta^3}[\varphi]_\times^2$$

$$J_r(\varphi) = I - \frac{1-\cos\theta}{\theta^2}[\varphi]_\times + \frac{\theta-\sin\theta}{\theta^3}[\varphi]_\times^2$$

Useful identities. The first two follow from $`[\varphi]_\times`$ being antisymmetric and $`[\varphi]_\times^2`$ symmetric; the third also needs the Rodrigues form $`R = I + \frac{\sin\theta}{\theta}[\varphi]_\times + \frac{1-\cos\theta}{\theta^2}[\varphi]_\times^2`$ (all three checked numerically against `so3_right_jacobian` and `so3_exp` in `use_numpy/lie_utils.py`):

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

Notice $J_r = J_l^\top$, as the identity predicts. Plugging $\theta \to 0$ into the same formulas collapses both matrices to $I$, confirming Section 11.3.

### 11.6 Where this actually matters

- **IMU preintegration**: the effect of a small change in gyroscope bias is naturally expressed in the sensor's own (body) frame, so bias-correction Jacobians in preintegration use $J_r$ - worked out end-to-end in [imu_preintegration.md](../optimization/imu_preintegration.md).
- **Covariance propagation on the manifold**: a rotation's uncertainty is a covariance on the tangent vector $\delta\varphi$, and what it numerically means depends on whether $\delta\varphi$ is defined via $`R\,\text{Exp}(\delta\varphi)`$ (right) or $`\text{Exp}(\delta\varphi)\, R`$ (left).
  - Converting between them is a multiplication by $R$ (for rotations, the adjoint is $R$ itself): $`\text{Exp}(\delta\varphi_{\text{left}})\,R = R\,\text{Exp}(\delta\varphi_{\text{right}})`$ gives $`\delta\varphi_{\text{left}} = R\,\delta\varphi_{\text{right}}`$.
  - Hence $`\Sigma_{\text{left}} = R\,\Sigma_{\text{right}}R^\top`$: a change of frame, not a change of the underlying uncertainty.
  - The Section 11.4 identity $`J_l = R\,J_r`$ follows from the same fact.
- **Factor graphs/bundle adjustment on $SE(3)$**: residual Jacobians w.r.t. a pose depend on which perturbation convention (left vs. right) the library uses, and the update must apply the step with the same convention. Mixing them computes wrong step directions: the optimization converges slowly, converges to the wrong place, or diverges, all while the code runs without error.

### 11.7 One-sentence intuition

> **$J_l$ and $J_r$ are the same "local sensitivity map" as every other Jacobian in this document, applied to the exponential map on a Lie group: a small change to the input can be reported in the world frame (left) or in the object's own frame (right), and the two answers only agree at $\varphi = 0$.**

---

## 12. References

1. Solà, J., Deray, J., & Atchuthan, D. (2018). *A micro Lie theory for state estimation in robotics*. arXiv:1812.01537. https://doi.org/10.48550/arXiv.1812.01537 - the standard modern reference for $J_l$/$`J_r`$ and the left/right perturbation conventions behind Section 11, written by (among others) the author of the `manif` library that this repo's `use_manif/` scripts are built on.
