# Nonlinear least squares (NLS)

We have several imperfect measurements, and we want to find the unknown parameters that make the total measurement error as small as possible.

---

## 1. Start with ordinary least squares

Suppose you want to fit a line to measurements:

```text
       •
    •     •
  •   •
──────────────
```

You have measurements:

$$(x_i,y_i)$$

and assume:

$$y = ax+b$$

The unknowns are:

$$x=\begin{bmatrix}a \\ 
b\end{bmatrix}$$

For each measurement, there is an error:

$$e_i = y_i-(ax_i+b)$$

We don't want to make **one particular error** zero.

We want to make **all errors collectively small**.

So we minimize:

$$\boxed{\min_{a,b}\sum_i e_i^2}$$

That's **least squares**.

---

## 2. Why square the errors?

Suppose the errors are:

$$[1,-2,3]$$

If we simply sum them:

$$1-2+3=2$$

positive and negative errors cancel.

Instead:

$$1^2+(-2)^2+3^2=14$$

Now every error contributes positively.

So:

$$\boxed{\text{total error}=\sum_i e_i^2}$$

---

## 3. Then what makes it "nonlinear"?

Here's the important distinction.

### Linear least squares

Suppose:

$$e_i = y_i-(ax_i+b)$$

The unknowns $a,b$ appear linearly.

That's a **linear least-squares** problem.

---

### Nonlinear least squares

Suppose instead:

$$y = ae^{bx}$$

Then:

$$e_i=y_i-ae^{bx_i}$$

Now $b$ appears inside an exponential.

Or in SLAM:

$$e_i = z_i-\pi(TX_i)$$

where:

* $T$ = camera pose
* $X_i$ = 3D landmark
* $\pi$ = camera projection

The relationship between the unknowns and measurements is nonlinear.

Therefore:

$$\boxed{\text{Nonlinear least squares}}$$

---

## 4. The general form

The standard NLS problem is:

$$\boxed{\min_x \frac12\sum_i \|e_i(x)\|^2}$$

or, stacking all residuals:

$$\boxed{\min_x \frac12\|e(x)\|^2}$$

where:

$$x =\begin{bmatrix}x_1\\ 
x_2\\ 
\vdots\\ 
x_n\end{bmatrix}$$

contains the unknowns.

Think of:

```text
Unknowns
   ↓
 x = [pose, landmark, bias, ...]
   ↓
Measurement model
   ↓
Predicted measurement
   ↓
Compare with actual measurement
   ↓
Residual
   ↓
Square + sum
   ↓
Total cost
```

---

## 5. SLAM example

Suppose a camera observes a landmark.

We have:

```text
          Landmark
             ● X
            /
           /
          /
  Camera ●
   T
```

We know the image measurement:

$$z$$

We predict where the landmark should appear:

$$\hat z=\pi(TX)$$

Therefore the reprojection error is:

$$\boxed{e(T,X)=z-\pi(TX)}$$

And we want:

$$\boxed{\min_{T,X}\|z-\pi(TX)\|^2}$$

This is nonlinear because of:

* camera projection $\pi(\cdot)$
* rotation inside $T$
* multiplication between pose and landmark

This is exactly the kind of problem encountered in **[bundle adjustment](bundle_adjustment.md)**.

---

## 6. Why not solve it directly?

Here's the fundamental problem.

For a simple linear equation:

$$Ax=b$$

we can solve directly using linear algebra.

But our SLAM problem looks more like:

$$e(x)=z-f(x)$$

where $f(x)$ is nonlinear.

For example:

$$f(x)=\begin{bmatrix}\sin x\\ e^x\\ x^2\end{bmatrix}$$

There's generally no simple closed-form solution for:

$$\min_x\|e(x)\|^2$$

So we use an iterative strategy.

---

## 7. The key trick: make the nonlinear problem locally linear

This is where **Jacobian** enters.

Suppose we're currently at:

$$x_k$$

We approximate the nonlinear residual using a first-order Taylor expansion:

$$e(x_k+\Delta x)\approx e(x_k)+J\Delta x$$

where:

$$J=\frac{\partial e}{\partial x}$$

is the Jacobian.

Visually:

```text
Nonlinear function

       ╭──────
      ╱
     ╱
    ●
   /
  /
```

Near the current point, we replace it with:

```text
Local linear approximation

    /
   /
  ●────────
```

We're saying:

> "I can't understand the whole nonlinear landscape, but I can approximate what's happening right around me."

---

## 8. This transforms NLS into a linear least-squares problem

Originally:

$$\min_x\|e(x)\|^2$$

After linearization:

$$\min_{\Delta x}\|e+J\Delta x\|^2$$

Now this is a **linear least-squares problem**.

We can solve it using the normal equations:

$$\boxed{J^TJ\Delta x=-J^Te}$$

Then update:

$$\boxed{x_{k+1}=x_k+\Delta x}$$

And repeat.

---

## 9. This is exactly where [Gauss–Newton](gauss_newton.md) comes from

So the relationship is:

```text
Nonlinear least squares
        ↓
Linearize residual
        ↓
e(x + Δx) ≈ e + JΔx
        ↓
Linear least squares
        ↓
JᵀJ Δx = -Jᵀe
        ↓
Gauss–Newton step
        ↓
Update x
        ↓
Repeat
```

Therefore:

> **Gauss–Newton is an algorithm for solving nonlinear least-squares problems.**

This distinction is important.

**NLS is the problem.**

**Gauss–Newton is one method for solving it.**

---

## 10. Where [Levenberg–Marquardt](levenberg_marquardt.md) fits

You just asked about LM.

Now its role becomes much clearer.

NLS gives us:

$$\min_x\|e(x)\|^2$$

Gauss–Newton gives:

$$J^TJ\Delta x=-J^Te$$

But GN can sometimes take bad steps.

LM modifies it:

$$\boxed{(J^TJ+\lambda I)\Delta x=-J^Te}$$

So:

```text
                NLS problem
                     │
                     ↓
               Linearization
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
    Gauss–Newton              LM
    JᵀJ Δx=-Jᵀe       (JᵀJ+λI)Δx=-Jᵀe
```

---

## 11. Now connect this to [factor graphs](factor_graph.md)

Suppose your SLAM graph contains:

```text
X0 ─── X1 ─── X2
│      │      │
L0     L1     L2
```

Each factor provides a residual:

$$e_1(x),e_2(x),e_3(x),...$$

The entire problem becomes:

$$\boxed{\min_x \sum_i \|e_i(x)\|^2}$$

That's **nonlinear least squares**.

So a factor graph is essentially a structured way of saying:

> "Here are all my variables and all the residuals connecting them."

Then an optimizer such as GN or LM tries to solve the resulting NLS problem.

---

## 12. Add measurement uncertainty

In real SLAM, measurements aren't equally reliable.

Suppose:

```text
Camera measurement → uncertain
IMU measurement    → relatively reliable
GPS measurement    → very reliable
```

We can weight their residuals:

$$\boxed{\min_x \sum_i e_i(x)^T W_i e_i(x)}$$

where $W_i$ represents how much we trust measurement $i$.

For Gaussian noise:

$$W_i=\Sigma_i^{-1}$$

where $\Sigma_i$ is the covariance.

So a useful interpretation is:

> **NLS doesn't just minimize errors; weighted NLS minimizes errors according to how trustworthy each measurement is.**

---

## 13. A really intuitive example

Imagine three sensors estimate your position.

```text
Sensor A:  10.0 m
Sensor B:  10.5 m
Sensor C:   9.8 m
```

You don't know the true position.

Instead of choosing one:

```text
10.0
10.5
 9.8
```

you find the position $x$ that minimizes:

$$(x-10.0)^2+(x-10.5)^2+(x-9.8)^2$$

The solution is a compromise.

Now imagine Sensor B is much noisier:

$$(x-10.0)^2+0.1(x-10.5)^2+(x-9.8)^2$$

Now Sensor B has less influence.

That's the intuition behind weighted least squares in SLAM.

---

## 14. Why SLAM naturally becomes NLS

This is the really important insight.

Almost every SLAM sensor gives you a statement like:

> **"Given these states, I should have observed this measurement."**

For example:

### Odometry

$$e_{odom}(X_i,X_j)$$

### IMU

$$e_{imu}(X_i,X_j,v_i,v_j,b_i,b_j)$$

### Camera

$$e_{cam}(X_i,L_j)$$

### GPS

$$e_{gps}(X_i)$$

### Loop closure

$$e_{loop}(X_i,X_j)$$

Put everything together:

$$
\boxed{
\min_x
\left(
\|e_{odom}\|^2+
\|e_{imu}\|^2+
\|e_{cam}\|^2+
\|e_{gps}\|^2+
\|e_{loop}\|^2
\right)
}
$$

That is one giant **nonlinear least-squares problem**.

---

## 15. Your SLAM mental map

At this point, I'd organize the concepts you've been studying like this:

```text
                         SLAM
                          │
                          ↓
                    Factor Graph
                          │
                          ↓
             Define measurement residuals
                          │
                          ↓
             Nonlinear Least Squares
                          │
              ┌───────────┴───────────┐
              ↓                       ↓
        Gauss–Newton           Levenberg–Marquardt
              │                       │
          fast/local             damped/robust
              │                       │
              └───────────┬───────────┘
                          ↓
                  Iterative optimization
```

And for online SLAM:

```text
Factor graph
     ↓
NLS problem
     ↓
GN / LM
     ↓
Batch optimization

        OR

Factor graph
     ↓
Incremental factorization
     ↓
iSAM / iSAM2
     ↓
Incremental smoothing
```

---

## 16. The one thing I want you to remember

**Nonlinear least squares is not an optimizer. It's the mathematical problem you're trying to solve.**

$$
\boxed{
\text{Find }x\text{ that minimizes the sum of squared measurement residuals}
}
$$

Then:

* **[Jacobian](../foundations/jacobian.md)** → tells you how residuals change when variables move.
* **[Gauss–Newton](gauss_newton.md)** → linearizes NLS and solves for a step.
* **[Levenberg–Marquardt](levenberg_marquardt.md)** → GN + damping for safer steps.
* **[Factor graph](factor_graph.md)** → organizes variables and residuals.
* **[iSAM/iSAM2](isam_optimization.md)** → solves/updates the factor-graph NLS problem incrementally.

That distinction—**problem formulation vs optimization algorithm**—is one of the most useful things to keep straight when learning SLAM.

---

## 17. References

1. Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer Series in Operations Research and Financial Engineering. Springer. ISBN 978-0-387-30303-1 - the general nonlinear-least-squares / Gauss–Newton treatment behind §4 and §7–9.
2. Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. ISBN 978-0-521-54051-3 - the camera-projection / reprojection-error formalism ($z - \pi(TX)$) behind §5.
3. Triggs, B., McLauchlan, P. F., Hartley, R. I., & Fitzgibbon, A. W. (2000). *Bundle Adjustment - A Modern Synthesis*. In Vision Algorithms: Theory and Practice (LNCS vol. 1883, pp. 298–372). Springer. https://doi.org/10.1007/3-540-44480-7_21 - the bundle-adjustment application referenced in §5.
4. Dellaert, F., & Kaess, M. (2017). *Factor Graphs for Robot Perception*. Foundations and Trends in Robotics, 6(1–2), 1–139. https://doi.org/10.1561/2300000043 - the factor-graph formulation of the NLS problem behind §11 and §14.

All four were verified against live search results before being added here (title, authors, venue/publisher, edition/volume/pages, and ISBN/DOI cross-checked), rather than cited from memory alone.
