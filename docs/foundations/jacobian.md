# Jacobian intuitive explanation

The **Jacobian** is one of those things that looks intimidating mathematically but has a very intuitive meaning:

> **A Jacobian tells you how a small change in the input will cause a small change in the output.**

That's basically it.

---

## 1. Start with a simple function

Suppose:

$$y = 3x$$

If $x$ changes by a tiny amount:

$$\Delta x = 0.1$$

then:

$$\Delta y = 3(0.1) = 0.3$$

The derivative:

$$\frac{dy}{dx} = 3$$

tells us:

> **"If I move $x$ a little bit, $y$ moves about 3 times as much."**

For a function with **one input and one output**, we call this a derivative.

---

## 2. Now imagine multiple inputs and outputs

Suppose:

$${\begin{bmatrix}y_1 \\ 
y_2 \end{bmatrix}=f\left(\begin{bmatrix}x_1 \\ 
x_2\end{bmatrix}\right)}$$

Now we have:

```text
       x1 ──────┐
                ├──→ y1
       x2 ──────┘

       x1 ──────┐
                ├──→ y2
       x2 ──────┘
```

We want to know:

* How does $y_1$ change when $x_1$ changes?
* How does $y_1$ change when $x_2$ changes?
* How does $y_2$ change when $x_1$ changes?
* How does $y_2$ change when $x_2$ changes?

To make this concrete, suppose:

$${y_1 = x_1 + x_2^2 \qquad y_2 = x_1 x_2}$$

Then:

$${\frac{\partial y_1}{\partial x_1} = 1 \qquad \frac{\partial y_1}{\partial x_2} = 2x_2 \qquad \frac{\partial y_2}{\partial x_1} = x_2 \qquad \frac{\partial y_2}{\partial x_2} = x_1}$$

So we put all those derivatives into a matrix:

$${J =\begin{bmatrix}\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} \\ 
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} \\ 
\end{bmatrix} = \begin{bmatrix}1 & 2x_2 \\ 
x_2 & x_1\end{bmatrix}}$$

That's the **Jacobian**. Notice the convention: each **row** is one output ($y_i$), each **column** is one input ($x_j$) - $J_{ij} = \partial y_i/\partial x_j$.

---

## 3. Think of it as a "sensitivity map"

This is probably the most useful intuition.

Suppose your robot's state is:

$${p = \begin{bmatrix} x \\ 
y \\ 
\theta \end{bmatrix}}$$

and your camera produces some measurement:

$${z = \begin{bmatrix} u \\ 
v \end{bmatrix}}$$

The Jacobian might look like:

$$
H =
\begin{bmatrix}
\frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} & \frac{\partial u}{\partial \theta} \\
\frac{\partial v}{\partial x} & \frac{\partial v}{\partial y} & \frac{\partial v}{\partial \theta}
\end{bmatrix}
$$

This tells you:

> **If I slightly perturb the robot's $(x, y, \theta)$, how much will the camera measurement $(u, v)$ change?**

So you can think of the Jacobian as a **sensitivity table**.

> This $H$ is schematic - its exact entries depend on the camera model, which isn't specified here. Section 4 below works out a concrete projection Jacobian end-to-end - though there the input is the 3D point being observed rather than the robot pose, so it's a related but distinct sensitivity matrix (labeled $J$, not $H$, for that reason).

---

## 4. A very intuitive example: camera projection

This works out a concrete instance of "camera measurement Jacobian" - but note it's differentiating with respect to the observed 3D point, not the robot pose from Section 3's $H$, which is why it gets its own symbol, $J$.

Suppose a 3D point is:

$$P = (X, Y, Z)$$

and the camera projects it onto the image:

$$u = f\frac{X}{Z}$$

$$v = f\frac{Y}{Z}$$

This is nonlinear because of the division by $Z$.

Now imagine:

> "What happens to the image point if the 3D point moves slightly?"

The Jacobian answers exactly that.

For example:

$$\frac{\partial u}{\partial X} = \frac{f}{Z}$$

Meaning:

> If $X$ changes slightly, $u$ changes approximately by $f/Z$ times that amount.

And:

$${\frac{\partial u}{\partial Z} = -\frac{fX}{Z^2}}$$

Meaning:

> Moving the point forward/backward in depth changes its image position, and the amount depends on its current depth and horizontal position.

Putting every partial derivative together gives the full Jacobian:

$${J = \begin{bmatrix} \frac{\partial u}{\partial X} & \frac{\partial u}{\partial Y} & \frac{\partial u}{\partial Z} \\ 
\frac{\partial v}{\partial X} & \frac{\partial v}{\partial Y} & \frac{\partial v}{\partial Z} \end{bmatrix} =\begin{bmatrix} f/Z & 0 & -fX/Z^2 \\ 
0 & f/Z & -fY/Z^2 \end{bmatrix}}$$

Notice the zeros: $\partial u/\partial Y = 0$ and $\partial v/\partial X = 0$, because horizontal image position ($u$) doesn't depend on vertical 3D position ($Y$) at all, and vice versa for $v$ and $X$. This is the actual Jacobian a visual-SLAM or bundle-adjustment system would compute at every reprojected point.

---

## 5. Why does EKF need it?

This is where it connects directly to your previous question.

Suppose the real system is nonlinear:

$$z = h(x)$$

The EKF doesn't want to deal with the full nonlinear function every time.

So it says:

> "Around my current estimate $\hat{x}$, can I approximate this nonlinear function with a linear one?"

The Jacobian gives us exactly that approximation:

$$h(x) \approx h(\hat{x}) + H(x - \hat{x})$$

where:

$$H = \left.\frac{\partial h}{\partial x}\right|_{\hat{x}}$$

So the Jacobian is essentially the **local slope of a multidimensional nonlinear function**.

---

## 6. Think about a mountain

This analogy is extremely useful.

Imagine you're standing somewhere on a mountain.

The actual mountain is complicated:

```text
                 /\       /\
        /\      /  \_____/  \
   ____/  \____/             \____
```

You don't know the entire mountain.

But you can ask:

> "If I take one tiny step north, what happens to my altitude?"

and:

> "If I take one tiny step east, what happens?"

The Jacobian tells you those local slopes.

At your current position, the complicated mountain can be approximated by a **flat plane**:

```text
             actual mountain
                /
               /
              /
             /________
            /
       you ●
```

The plane is the **local linear approximation**.

That's essentially what EKF does.

---

## 7. Jacobian ≈ "local translator"

Here's another mental model I really like for robotics - the same idea as the mountain in Section 6, just written as an equation instead of a picture.

Suppose you have:

$$\delta x$$

which means:

> "I slightly changed my robot state."

The Jacobian converts that into:

$$\delta z$$

which means:

> "Because of that change, my sensor measurement changed approximately this much."

In simplified form:

$$\boxed{\delta z \approx H\delta x}$$

So:

```text
small change in state
        │
        ▼
    [ Jacobian ]
        │
        ▼
small change in measurement
```

That's why Jacobians are everywhere in:

* EKF
* ESKF
* IEKF
* bundle adjustment
* nonlinear least squares
* factor graphs
* visual odometry
* SLAM

---

## 8. One subtle point: Jacobian is LOCAL

This is extremely important.

Suppose:

$$y = x^2$$

Then:

$$\frac{dy}{dx} = 2x$$

At $x = 1$:

$$J = 2$$

At $x = 10$:

$$J = 20$$

So the Jacobian changes depending on **where you are**.

That's why we say:

> **The Jacobian describes the local behavior of a nonlinear function.**

This is also the fundamental weakness of EKF.

If the function is highly nonlinear, the local approximation might become poor.

---

## 9. Connecting this back to IEKF

Now we can connect everything you've asked about.

### KF

System:

$$x_{k+1} = Fx_k$$

Already linear.

No Jacobian is necessary.

### EKF

System:

$$x_{k+1} = f(x_k)$$

Nonlinear.

So:

$$F_k = \left.\frac{\partial f}{\partial x}\right|_{\hat{x}_k}$$

The Jacobian tells us:

> "How does the nonlinear system behave **locally around my current estimate**?"

### IEKF

Same basic idea of linearization, but with a crucial difference:

> **The perturbation/error is defined according to the geometry and invariance of the system.**

So instead of blindly asking:

> "What is the derivative with respect to my state vector?"

you carefully ask:

> **"What is the derivative with respect to the appropriate local perturbation on the state manifold?"**

That's one reason Lie groups and Jacobians become so tightly connected in modern SLAM.

---

## 10. The one-sentence intuition

If you remember only one thing:

> **The Jacobian is a multidimensional "local sensitivity map": it tells you how small changes in one thing approximately translate into small changes in another thing.**

And in EKF specifically:

> **The Jacobian is the tool that lets us temporarily turn a nonlinear system into a locally linear one.**

---

## 11. Left and right Jacobians: sensitivity on a curved space

Section 9 mentioned that IEKF perturbs the state "according to the geometry of the manifold" instead of just subtracting vectors. This section makes that concrete for rotations, and explains where the **left Jacobian** $J_l$ and **right Jacobian** $J_r$ come from.

### 11.1 Why plain addition breaks

Everywhere above, perturbing an input meant simple addition: $x \to x + \delta x$. That works because $x$ lives in a vector space.

A rotation $R \in SO(3)$ does not. There is no such thing as $R + \delta R$ - the result generally isn't even a valid rotation. Instead, a small perturbation $\delta\varphi \in \mathbb{R}^3$ is turned into a rotation via the exponential map, $\text{Exp}(\delta\varphi)$, and then **composed** (matrix-multiplied) with $R$:

$$R' = \text{Exp}(\delta\varphi)\,R \qquad \text{or} \qquad R' = R\,\text{Exp}(\delta\varphi)$$

Both are valid ways to perturb $R$ - but they are *not* the same $R'$ in general, because matrix multiplication doesn't commute. That single fact - no commutativity - is the entire reason left and right Jacobians exist. On a vector space this distinction never comes up, because addition always commutes.

### 11.2 The actual question being asked

Suppose you already have a rotation built from $\varphi$, i.e. $\text{Exp}(\varphi)$, and you want to nudge the *argument*: $\text{Exp}(\varphi + \delta\varphi)$. Because $\text{Exp}$ is a curved, nonlinear map (just like $u = fX/Z$ was curved in Section 4), you cannot simply distribute the addition. What you *can* do is ask the same sensitivity question as always - "how does the output change?" - and express the answer as a small rotation composed onto $\text{Exp}(\varphi)$, either on the left or on the right:

$$\text{Exp}(\varphi+\delta\varphi)\;\approx\;\text{Exp}\big(J_l(\varphi)\,\delta\varphi\big)\cdot\text{Exp}(\varphi) \qquad\text{(left)}$$

$$\text{Exp}(\varphi+\delta\varphi)\;\approx\;\text{Exp}(\varphi)\cdot\text{Exp}\big(J_r(\varphi)\,\delta\varphi\big) \qquad\text{(right)}$$

So $J_l$ and $J_r$ are still exactly what Section 10 says a Jacobian always is - a local sensitivity map, "small change in input → small change in output" - just applied to the exponential map instead of to a vector-valued function, and reported as *which side* the resulting perturbation attaches to.

### 11.3 The intuition: compass vs. steering wheel

- **Right Jacobian** $J_r(\varphi)$: the extra rotation is applied *after* $\text{Exp}(\varphi)$, i.e. it's expressed in the object's **own current (body) frame**. Like turning a car's steering wheel a bit more - "a bit more" is always relative to however the car is already pointed.
- **Left Jacobian** $J_l(\varphi)$: the extra rotation is applied *before*, i.e. it's expressed in the **fixed world/global frame**. Like someone nudging your heading by a fixed compass bearing, regardless of which way you're currently facing.

Near the identity ($\varphi \to 0$), there's no rotation yet to disagree about "whose frame," so the two notions collapse: $J_l(0) = J_r(0) = I$. This is exactly the flat-tangent-plane picture from Section 6 - right at the point of linearization, the manifold looks flat and left/right don't matter yet. The distinction only shows up once you're linearizing *away* from the identity, i.e., around some existing rotation.

### 11.4 Closed form for $SO(3)$

With $\theta = \lVert\varphi\rVert$ and $[\varphi]_\times$ the skew-symmetric matrix of $\varphi$:

$$J_l(\varphi) = I + \frac{1-\cos\theta}{\theta^2}[\varphi]_\times + \frac{\theta-\sin\theta}{\theta^3}[\varphi]_\times^2$$

$$J_r(\varphi) = I - \frac{1-\cos\theta}{\theta^2}[\varphi]_\times + \frac{\theta-\sin\theta}{\theta^3}[\varphi]_\times^2$$

Useful identities (all follow from $`{[\varphi]_\times^2}`$ being symmetric and $`{[\varphi]_\times}`$ being antisymmetric):

$$J_r(\varphi) = J_l(-\varphi) \qquad J_r(\varphi) = J_l(\varphi)^\top \qquad J_l(\varphi) = R(\varphi)\,J_r(\varphi)$$

The last one is the frame-conversion identity: $R(\varphi)$ is exactly what turns a body-frame perturbation into a world-frame one, so it's the bridge between $J_r$ and $J_l$ - consistent with the compass/steering-wheel picture above.

### 11.5 A worked example: 90° yaw

Take $\varphi = (0, 0, \theta)$, a pure rotation about $z$, with $\theta = \pi/2$. For a single-axis rotation, $[\varphi]_\times^2 = \theta^2(kk^\top - I) = \text{diag}(-\theta^2, -\theta^2, 0)$ with $k=(0,0,1)$, which keeps the algebra clean. Plugging $\theta=\pi/2$ ($\cos\theta=0$, $\sin\theta=1$) into the formulas above gives:

$$J_l \approx \begin{bmatrix}0.637 & -0.637 & 0\\ 
0.637 & 0.637 & 0\\ 
0 & 0 & 1\end{bmatrix} \qquad J_r \approx \begin{bmatrix}0.637 & 0.637 & 0\\ 
-0.637 & 0.637 & 0\\ 
0 & 0 & 1\end{bmatrix}$$

Notice $J_r = J_l^\top$, exactly as the identity predicts. If you instead plug in $\theta \to 0$ in the same formulas, both matrices collapse to $I$, confirming Section 11.3's claim about the identity.

### 11.6 Where this actually matters

* **IMU preintegration**: the effect of a small change in gyroscope bias is naturally expressed in the sensor's own (body) frame, so bias-correction Jacobians in preintegration use $J_r$ - worked out end-to-end in [imu_preintegration.md](../optimization/imu_preintegration.md).
* **Covariance/uncertainty propagation on the manifold**: a rotation's uncertainty is stored as a covariance on the tangent vector $\delta\varphi$, but whether that $\delta\varphi$ is defined via $R\,\text{Exp}(\delta\varphi)$ (right) or $\text{Exp}(\delta\varphi)\, R$ (left) changes what the covariance numerically means. Converting between the two conventions is exactly a multiplication by $J_l$ or $J_r$ - a change of frame, not a change of the underlying uncertainty.
* **Factor graphs/bundle adjustment on $SE(3)$**: residual Jacobians w.r.t. a pose depend on which perturbation convention (left vs. right) the library uses; using the wrong one silently biases the optimization even though the code runs without error.

### 11.7 One-sentence intuition

> **$J_l$ and $J_r$ are the same "local sensitivity map" as every other Jacobian in this document - they just answer the question for the exponential map on a Lie group, where a small change to the input can be reported either in the world frame (left) or in the object's own frame (right), and those two answers only agree exactly at the identity.**
