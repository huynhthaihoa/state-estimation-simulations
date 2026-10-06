# Nonlinear least squares (NLS)

We have several imperfect measurements, and we want to find the unknown parameters that make the total measurement error as small as possible.

---

## 1. Start with ordinary least squares

Suppose we want to fit a line to measurements:

```text
       •
    •     •
  •   •
──────────────
```

We have measurements $`(t_i,y_i)`$ and assume $`y = at+b`$. The unknowns are:

```math
x=\begin{bmatrix} a \\ 
b \end{bmatrix}
```

(We call the data's input $t$ because $x$ is reserved for the *unknown parameter vector*.)

Each measurement has an error:

$$e_i = y_i-(at_i+b)$$

We don't want to zero **one particular error**; we want **all errors collectively small**. So we minimize:

$$\boxed{\min_{a,b}\sum_i e_i^2}$$

That's **least squares**. (§4 adds a factor $`\tfrac12`$ in front; it doesn't change the minimizer, and later sections drop it again.)

---

## 2. Why square the errors?

Take errors $`[1,-2,3]`$:

- Plain sum: $`1-2+3=2`$, so positive and negative errors cancel.
- Squares: $`1^2+(-2)^2+3^2=14`$, so every error contributes positively.

Absolute values, $`\sum_i|e_i|`$, would also stop the cancelling, so why squares? Two reasons:

- **Easy to solve.** Once we linearize $e$ (§7), each step is a linear least-squares problem with a closed-form solution (normal equations, §8). $`|e|`$ has a kink at zero and no such structure.
- **Statistically right.** For Gaussian measurement noise, minimizing squared error gives the maximum-likelihood estimate, which is also why §12's weights are $`W_i=\Sigma_i^{-1}`$.

So $`\text{total error}=\sum_i e_i^2`$.

---

## 3. Then what makes it "nonlinear"?

### Linear least squares

With $`e_i = y_i-(at_i+b)`$ the unknowns $a,b$ appear linearly: a **linear least-squares** problem.

### Nonlinear least squares

Suppose instead $`y = ae^{bt}`$. Then $`e_i=y_i-ae^{bt_i}`$, and $b$ sits inside an exponential.

Or in SLAM:

$$e_i = z_i-\pi(TX_i)$$

where:

- $T$ = camera pose, written here as world-to-camera, so $TX_i$ is the landmark in the camera frame. [bundle_adjustment.md](bundle_adjustment.md) uses the inverse convention: a camera-to-world $T$, and $\pi(T^{-1}P)$.
- $X_i$ = 3D landmark
- $\pi$ = camera projection

The relationship between unknowns and measurements is nonlinear, so this is **nonlinear least squares**.

---

## 4. The general form

The standard NLS problem is:

```math
\boxed{\min_x \frac12\sum_i \|e_i(x)\|^2}
```

or, stacking all residuals:

```math
\boxed{\min_x \frac12\|e(x)\|^2}
```

where:

```math
x =\begin{bmatrix} x_1\\ 
x_2\\ 
\vdots\\ 
x_n \end{bmatrix}
```

contains the unknowns. The pipeline from unknowns to cost:

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

Suppose a camera observes a landmark:

```text
          Landmark
             ● X
            /
           /
          /
  Camera ●
   T
```

We know the image measurement $`z`$ and predict where the landmark should appear, $`\hat z=\pi(TX)`$. The reprojection error is:

$$\boxed{e(T,X)=z-\pi(TX)}$$

and we want:

```math
\boxed{\min_{T,X}\|z-\pi(TX)\|^2}
```

This is nonlinear because of:

- camera projection $\pi(\cdot)$
- rotation inside $T$
- multiplication between pose and landmark

This is exactly the kind of problem encountered in **[bundle adjustment](bundle_adjustment.md)**.

---

## 6. Why not solve it directly?

A linear equation $`Ax=b`$ can be solved directly with linear algebra. Our SLAM problem looks more like $`e(x)=z-f(x)`$ with $f$ nonlinear, for example (here $x$ is a scalar input):

```math
f(x)=\begin{bmatrix} \sin x\\ e^x\\ x^2 \end{bmatrix}
```

There is generally no closed-form solution for $`\min_x\|e(x)\|^2`$, so we iterate.

---

## 7. The key trick: make the nonlinear problem locally linear

This is where the **Jacobian** enters. At the current estimate $`x_k`$ we use a first-order Taylor expansion of the residual:

$$e(x_k+\Delta x)\approx e(x_k)+J\Delta x,\qquad J=\frac{\partial e}{\partial x}$$

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

```math
\min_x\|e(x)\|^2
```

After linearization:

```math
\min_{\Delta x}\|e+J\Delta x\|^2
```

This is a **linear least-squares problem**, solved by the normal equations:

$$\boxed{J^\top J\Delta x=-J^\top e}$$

Then update, and repeat:

$$\boxed{x_{k+1}=x_k+\Delta x}$$

(For poses, $+$ becomes $\oplus$, a retraction onto the pose manifold.)

---

## 9. This is exactly where [Gauss–Newton](gauss_newton.md) comes from

The whole chain:

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

> **NLS is the problem. Gauss–Newton is one method for solving it.**

---

## 10. Where [Levenberg–Marquardt](levenberg_marquardt.md) fits

- NLS gives the problem $`\min_x\|e(x)\|^2`$.
- Gauss–Newton gives the step $`J^\top J\Delta x=-J^\top e`$, which can occasionally be a bad step.
- LM damps it:

$$\boxed{(J^\top J+\lambda I)\Delta x=-J^\top e}$$

Both share the same linearization (§9); they differ only in the damping term. Marquardt's variant, $`\lambda\,\mathrm{diag}(J^\top J)`$ in place of $`\lambda I`$, is what this repo's solvers use (`use_numpy/pose_graph.py` solves `np.linalg.solve(H + lam * np.diag(diag_H), g)`); see [levenberg_marquardt.md §10](levenberg_marquardt.md#10-marquardts-scaling).

---

## 11. Now connect this to [factor graphs](factor_graph.md)

Suppose our SLAM graph contains:

```text
x0 ─── x1 ─── x2
│      │      │
l0     l1     l2
```

Each factor provides a residual $`e_1(x),e_2(x),e_3(x),\dots`$, so the entire problem becomes:

```math
\boxed{\min_x \sum_i \|e_i(x)\|^2}
```

That's **nonlinear least squares**. A factor graph is a structured way of saying:

> "Here are all my variables and all the residuals connecting them."

An optimizer such as GN or LM then solves the resulting NLS problem.

---

## 12. Add measurement uncertainty

In real SLAM, measurements aren't equally reliable.

Suppose two position fixes for the same pose come from different receivers:

```text
consumer GPS fix → good to about ±5 m
RTK-GPS fix      → good to about ±2 cm
```

(Covariances let us compare sensors with different units: a camera measures pixels, an IMU measures rates, so each measurement carries its own $\Sigma_i$.)

We weight the residuals:

$$\boxed{\min_x \sum_i e_i(x)^\top W_i e_i(x)}$$

where $W_i$ says how much we trust measurement $i$. For Gaussian noise $`W_i=\Sigma_i^{-1}`$, with $\Sigma_i$ the covariance.

($W_i$ is the same quantity as the **information matrix** $\Omega_k$ used from [factor_graph.md §4](factor_graph.md#4-optimization-means-minimizing-all-those-errors) onward, including in [sparse_cholesky_factorization.md](sparse_cholesky_factorization.md); this doc uses $W_i$, those use $\Omega$.)

> **NLS doesn't just minimize errors; weighted NLS minimizes errors according to how trustworthy each measurement is.**

---

## 13. A worked example: three sensors

Suppose three sensors estimate our position.

```text
Sensor A:  10.0 m
Sensor B:  10.5 m
Sensor C:   9.8 m
```

We don't know the true position, and instead of choosing one reading we find the $x$ that minimizes:

$$(x-10.0)^2+(x-10.5)^2+(x-9.8)^2$$

The solution is a compromise: the average, $`x = 10.1`$ m.

If Sensor B is much noisier, we down-weight it:

$$(x-10.0)^2+0.1(x-10.5)^2+(x-9.8)^2$$

B now has less influence and the solution moves toward A and C: $`x = (10.0 + 0.1\cdot 10.5 + 9.8)/2.1 \approx 9.93`$ m.

That's the intuition behind weighted least squares in SLAM.

---

## 14. Why SLAM naturally becomes NLS

Almost every SLAM sensor gives a statement like:

> **"Given these states, I should have observed this measurement."**

| Sensor | Residual |
|---|---|
| Odometry | $`e_{odom}(X_i,X_j)`$ |
| IMU | $`e_{imu}(X_i,X_j,v_i,v_j,b_i,b_j)`$ |
| Camera | $`e_{cam}(X_i,L_j)`$ |
| GPS | $`e_{gps}(X_i)`$ |
| Loop closure | $`e_{loop}(X_i,X_j)`$ |

Putting everything together (each term weighted by its $W_i$ as in §12, left out here for readability):

```math
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
```

That is one giant **nonlinear least-squares problem**.

---

## 15. Your SLAM mental map

The concepts above organize into one map:

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
          fast/local             damped/safer
              │                       │
              └───────────┬───────────┘
                          ↓
                  Iterative optimization
```

The step rule (GN or LM) and the solve strategy (batch or incremental) are separate axes. Both strategies start from the same NLS problem:

```text
Factor graph
     ↓
NLS problem
     ↓
┌────┴───────────────────────┐
↓                            ↓
Batch: re-solve everything   Incremental: re-solve only
with GN/LM                   what changed (iSAM/iSAM2)
```

See [levenberg_marquardt.md §11](levenberg_marquardt.md#11-where-lm-fits) for how the two axes combine.

This is one corner of the repo-wide map in [slam_mental_map.md](../slam_mental_map.md), which places every doc and script on one picture.

---

## 16. The one thing to remember

**Nonlinear least squares is not an optimizer. It's the mathematical problem we're trying to solve.**

$$
\boxed{
\text{Find }x\text{ that minimizes the sum of squared measurement residuals}
}
$$

Then:

- **[Jacobian](../foundations/jacobian.md)** → tells us how residuals change when variables move.
- **[Gauss–Newton](gauss_newton.md)** → linearizes NLS and solves for a step.
- **[Levenberg–Marquardt](levenberg_marquardt.md)** → GN + damping for safer steps.
- **[Factor graph](factor_graph.md)** → organizes variables and residuals.
- **[iSAM/iSAM2](isam_optimization.md)** → solves/updates the factor-graph NLS problem incrementally.

Keeping **problem formulation vs optimization algorithm** straight is one of the most useful habits in SLAM.

---

## 17. References

1. Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer Series in Operations Research and Financial Engineering. Springer. ISBN 978-0-387-30303-1 - the general nonlinear-least-squares/Gauss–Newton treatment behind §4 and §7–9.
2. Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. ISBN 978-0-521-54051-3 - the camera-projection/reprojection-error formalism (${z - \pi(TX)}$) behind §5.
3. Triggs, B., McLauchlan, P. F., Hartley, R. I., & Fitzgibbon, A. W. (2000). *Bundle Adjustment - A Modern Synthesis*. In Vision Algorithms: Theory and Practice (LNCS vol. 1883, pp. 298–372). Springer. https://doi.org/10.1007/3-540-44480-7_21 - the bundle-adjustment application referenced in §5.
4. Dellaert, F., & Kaess, M. (2017). *Factor Graphs for Robot Perception*. Foundations and Trends in Robotics, 6(1–2), 1–139. https://doi.org/10.1561/2300000043 - the factor-graph formulation of the NLS problem behind §11 and §14.

<!-- All four were verified against live search results before being added here (title, authors, venue/publisher, edition/volume/pages, and ISBN/DOI cross-checked), rather than cited from memory alone. -->
