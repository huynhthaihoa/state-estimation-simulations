# Gauss-Newton Optimization

Gauss-Newton and its damped variants, like [Levenberg-Marquardt](levenberg_marquardt.md), are the core optimizers behind **SLAM, [bundle adjustment](bundle_adjustment.md), and [pose-graph optimization](pose_graph_optimization.md)**. This doc builds the intuition for Gauss-Newton.

---

## 1. The basic idea

We want parameters $x$ that make some measurements fit as well as possible. In SLAM, for example: which robot pose $x$ best explains these sensor measurements?

We define a **residual**:

$$r(x) = \text{prediction}(x) - \text{measurement}$$

and minimize the **total squared error**:

```math
\min_x \frac12 \|r(x)\|^2
```

We'll call this cost $f(x) = \tfrac12\|r(x)\|^2$.

The problem is that $r(x)$ is usually **nonlinear**. Gauss-Newton's answer:

> **"I can't solve this nonlinear problem directly, so around my current guess, I'll pretend it is linear, solve that easier problem, move there, and repeat."**

---

## 2. Intuition: fit a bowl, jump to its bottom

Imagine standing somewhere on a complicated landscape, looking for its lowest point. We can't see the whole landscape, only the ground right around us.

So we fit a simple **bowl** to that nearby ground and jump straight to the bottom of the bowl. The real ground isn't exactly a bowl, so we land near the low point, not on it. We fit a new bowl there and jump again.

Each jump uses the *shape* of the ground, not just its slope. That is what lets Gauss-Newton take big, well-aimed steps (§10).

---

## 3. Where does the "linear" part come from?

Around the current estimate $x$, a first-order Taylor expansion approximates the residual:

$$r(x+\Delta x)\approx r(x) + J\Delta x$$

where $J$ is the **[Jacobian](../foundations/jacobian.md)**. The nonlinear $r(x+\Delta x)$ is replaced locally by the linear $r(x)+J\Delta x$.

So instead of asking "which $x$ minimizes this nonlinear function?", we ask "which small change $\Delta x$ makes this linear approximation as small as possible?"

---

## 4. The optimization problem

Linearization turns the original problem into:

```math
\min_{\Delta x} \frac12 \|r + J\Delta x\|^2
```

This is a **linear least-squares problem**, so we can solve it analytically. Setting the derivative to zero gives the **Gauss-Newton equation**:

$$J^\top J\Delta x = -J^\top r$$

Then we update and repeat:

$$x_{\text{new}} = x_{\text{old}}+\Delta x$$

> **Note**: solving it needs $J^\top J$ to be invertible, i.e. $J$ must have full column rank. In SLAM it isn't by default: some motions of the whole solution change no residual (a **gauge freedom**), so $J^\top J$ stays singular until something pins the solution down.
>
> - **Pose graphs**: shifting or rotating the whole trajectory together (6 DoF in 3D). Anchoring the first pose fixes it - `pose_graph.py` does this, "heavily anchoring node 0".
> - **Monocular BA**: also uniformly rescaling the scene (7 DoF), which one pose can't pin. The BA scripts fix *two* cameras instead (a prior on cameras 0 and 1, or two hard-fixed keyframes); see [bundle_adjustment.md §14](bundle_adjustment.md#14-evaluating-the-result-gauge-freedom-and-umeyama-alignment).
>
> [Levenberg-Marquardt](levenberg_marquardt.md)'s damping (its §4-§5) also keeps the system solvable.

> **Note on signs**: some docs in this repo, such as [bundle_adjustment.md §6.1](bundle_adjustment.md#61-the-ba-math-concretely), write the residual the other way round, $r = \text{measurement} - \text{prediction}$, and take $J$ as the Jacobian of the *prediction*. That $J$ is minus the Jacobian of their residual, so their equation reads $J^\top J\Delta x = +J^\top r$. It's the same step as $J^\top J\Delta x = -J^\top r$ above, just written with different sign conventions.

---

## 5. The algorithm loop

```text
Current guess
     ↓
Calculate residuals
     ↓
Calculate Jacobian
     ↓
Pretend the nonlinear problem is locally linear
     ↓
Solve for the best Δx
     ↓
Update x
     ↓
Repeat
```

The Jacobian is what makes the loop work. It answers:

> **"If I slightly change each parameter, how will my errors change?"**

---

## 6. Tiny numerical example

Find $x$ with $x^2 = 4$, i.e. minimize $\tfrac12(x^2-4)^2$ with residual $r(x)=x^2-4$ and Jacobian $J = \frac{dr}{dx}=2x$.

Start at $x=3$, so $r = 9-4 = 5$ and $J = 6$. Everything is scalar, so the Gauss-Newton equation becomes:

$$36\Delta x=-6\cdot 5 \quad\Rightarrow\quad \Delta x=-\tfrac{5}{6}\approx-0.833$$

$$x_{\text{new}}=3-\tfrac{5}{6}\approx2.167$$

One step gets us much closer to $2$, and repeating converges rapidly.

---

## 7. Why is it called Gauss-Newton?

**Newton's method** uses the **second derivative** (the Hessian) $H$ and solves $H\Delta x=-\nabla f$. For nonlinear least squares, the exact Hessian decomposes as:

$$H=J^\top J + \sum_i r_i \nabla^2 r_i$$

Gauss-Newton drops the second term, $H \approx J^\top J$, and so solves

$$J^\top J\Delta x=-J^\top r$$

The right-hand sides are the same thing: the gradient of $f(x) = \tfrac12\|r(x)\|^2$ is exactly $\nabla f = J^\top r$. Gauss-Newton only changes the left-hand side. Needing only first derivatives makes it **cheaper and well suited to least-squares problems**.

> **Note**: dropping $\sum_i r_i \nabla^2 r_i$ is a good approximation when the residuals are small at the solution (the measurements fit well) or $r$ is only mildly nonlinear. With large residuals or strong nonlinearity, the Gauss-Newton step can overshoot and even increase the cost - exactly what [Levenberg-Marquardt's damping](levenberg_marquardt.md#4-the-damping-idea) guards against.

---

## 8. Why is this everywhere in SLAM?

A robot at pose $T_i$ observes landmark $p_j$. The predicted measurement is

$$
\hat z_{ij} = h(T_i,p_j)
$$

where:

- $T_i$ = robot pose
- $p_j$ = landmark
- $h(\cdot)$ = camera/measurement model
- $z_{ij}$ = actual measurement of landmark $j$ from pose $i$

The residual is $r_{ij} = h(T_i,p_j)-z_{ij}$, and the SLAM problem becomes:

```math
\min_{\{T_i\},\{p_j\}}
\sum_{(i,j)\in\mathcal{O}}\|r_{ij}\|^2
```

where $\mathcal{O}$ is the set of pose-landmark pairs that were actually observed (not every landmark is seen from every pose).

> **Note**: real systems also weight each residual by its measurement's information matrix, $`r_{ij}^\top\Omega_{ij}\,r_{ij}`$ instead of $\|r_{ij}\|^2$, so precise sensors count more - left out here for readability. See [nonlinear_least_square.md §12](nonlinear_least_square.md#12-add-measurement-uncertainty) and [factor_graph.md](factor_graph.md).

This is a huge nonlinear problem. Gauss-Newton linearizes every measurement function around the current estimates of the poses and landmarks:

$$r_{ij}(x+\Delta x) \approx r_{ij}(x)+J_{ij}\Delta x$$

All measurements then stack into one large system:

$$
J^\top J\Delta x=-J^\top r
$$

Solve it → update all poses and landmarks → repeat.

One caveat: landmarks $p_j\in\mathbb{R}^3$ really do just get $p_j \leftarrow p_j+\Delta p_j$. Poses $T_i$ don't - they live on the $SE(3)$ manifold, so the update is a **retraction** through the exponential map, $T_i \leftarrow T_i\cdot\mathrm{Exp}(\Delta x_i)$, not plain addition. See [pose_graph_optimization.md](pose_graph_optimization.md) for the full derivation.

This is the core optimization mechanism behind many **[bundle adjustment](bundle_adjustment.md) and [graph-SLAM](pose_graph_optimization.md) systems**.

---

## 9. Picture: following the tangent

Take §6's example: the residual $r(x) = x^2 - 4$ is a curve, and we're looking for where it hits zero. Gauss-Newton doesn't try to understand the whole curve. Near the current guess, it treats the curve as a straight line: the tangent, which is exactly the linearization $r(x) + J\Delta x$ from §3.

```text
  r(x)
  ↑
  │                                 ●  current guess (r > 0)
  │                                /
  │                            . /
  │                         .  /
  │    real curve r(x) → .   /        ← tangent line: r(x) + J·Δx
  │                   .    /
  │                .     /
  │             .      /
  ┼───────────●──────○────────────────────────→ x
              ↑      ↑
            root next guess
                 (where the tangent hits zero)
```

It moves to where that line says the residual is zero: from $x = 3$ to $x \approx 2.167$. The real curve bends, so that isn't the true root ($x = 2$), but it's much closer. A new tangent at $x \approx 2.167$ makes the next jump land even closer.

(With many residuals, $J\Delta x$ usually can't make every residual exactly zero at once; Gauss-Newton then moves to where the linearized residuals are *as small as possible* in the least-squares sense - the bottom of §2's bowl. The one-residual picture above is the special case where that minimum is exactly zero.)

---

## 10. Gauss-Newton vs Gradient Descent

### Gradient descent

Gradient descent asks:

> **"Which direction is downhill?"**

and takes a chosen step:

$$\Delta x=-\alpha\nabla f$$

where $\alpha$ is the learning rate.

### Gauss-Newton

Gauss-Newton asks:

> **"Given the local shape of the least-squares problem, what step should I take to approximately reach the minimum?"**

$$J^\top J\Delta x=-J^\top r$$

It uses much more information about the local geometry, so it can converge much faster near the solution - quadratically for zero-residual problems like §6's, where the term Gauss-Newton drops (§7) vanishes at the solution.

---

## 11. The family of methods

| Method                                            | Idea                                            |
| ------------------------------------------------- | ----------------------------------------------- |
| **Gradient Descent**                              | Follow the slope                                |
| **Newton**                                        | Use slope + curvature                           |
| **Gauss-Newton**                                  | Use Jacobian structure to approximate curvature |
| **[Levenberg-Marquardt](levenberg_marquardt.md)** | Gauss-Newton + damping for robustness           |

SLAM systems typically use **Gauss-Newton/Levenberg-Marquardt + a sparse linear solver**, because SLAM produces large, sparse least-squares problems.

---

## 12. The one sentence to remember

> **Gauss-Newton repeatedly approximates a nonlinear least-squares problem as a local linear least-squares problem, solves for the best parameter update, and repeats.**

Or more intuitively:

> **"I'm going to pretend the world is linear around where I currently am, take the best step according to that approximation, and then update my approximation."**

That idea is the bridge from **[Jacobian](../foundations/jacobian.md) → Gauss-Newton → [bundle adjustment](bundle_adjustment.md) → [pose-graph optimization](pose_graph_optimization.md) → SLAM**.

---

## 13. References

1. Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-40065-5 - the standard textbook treatment of the Gauss-Newton method (Chapter 10) behind this whole doc, including the relationship to Newton's method (§7) and gradient descent (§10) covered here.
