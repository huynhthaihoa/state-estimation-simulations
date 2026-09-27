# Gauss-Newton Optimization

For **SLAM, [bundle adjustment](bundle_adjustment.md), and [pose-graph optimization](pose_graph_optimization.md)**, Gauss-Newton is one of the most important optimization ideas to understand intuitively.

---

## 1. The basic idea

Suppose you want to find parameters $x$ that make some measurements fit as well as possible.

For example, in SLAM:

> "What robot pose $x$ best explains these sensor measurements?"

You define a **residual**:

$$r(x) = \text{prediction}(x) - \text{measurement}$$

and want to minimize the **total squared error**:

```math
\min_x \frac12 \|r(x)\|^2
```

We'll call this cost $f(x) = \tfrac12\|r(x)\|^2$.

The problem is that $r(x)$ is usually **nonlinear**.

Gauss-Newton says:

> **"I can't solve this nonlinear problem directly, so around my current guess, I'll pretend it is linear, solve that easier problem, move there, and repeat."**

That's essentially the whole algorithm.

---

## 2. Imagine you're lost on a mountain

Imagine you're standing somewhere on a complicated mountain landscape.

Your goal:

> **Find the lowest point.**

You don't know the entire landscape.

But you can look at the terrain immediately around you.

If the terrain looks approximately like a **tilted bowl**, you can estimate:

> "If I move this direction, I'll go downhill."

So you:

1. Look at the terrain around your current position.
2. Approximate it with something simpler.
3. Find the minimum of that approximation.
4. Move there.
5. Repeat.

That's Gauss-Newton.

---

## 3. Where does the "linear" part come from?

Suppose your current estimate is $x$.

We can approximate the residual using a first-order Taylor expansion:

$$r(x+\Delta x)\approx r(x) + J\Delta x$$

where $J$ is the **[Jacobian](../foundations/jacobian.md)**.

This is extremely important.

The nonlinear function:

$$r(x+\Delta x)$$

gets replaced locally by the linear approximation:

$$r(x)+J\Delta x$$

So instead of asking:

> "What $x$ minimizes this complicated nonlinear function?"

we ask:

> "What small change $\Delta x$ makes this local linear approximation as small as possible?"

---

## 4. The optimization problem

Originally:

```math
\min_x \frac12\|r(x)\|^2
```

After linearization:

```math
\min_{\Delta x} \frac12 \|r + J\Delta x\|^2
```

Now this is a **linear least-squares problem**.

We can solve it analytically.

Taking the derivative and setting it to zero gives:

$$J^\top J\Delta x = -J^\top r$$

This is the famous **Gauss-Newton equation**.

> **Note**: solving it needs $J^\top J$ to be invertible, i.e. $J$ must have full column rank. In SLAM it isn't by default: shifting or rotating the whole trajectory and map together changes no residual (a **gauge freedom**), so $J^\top J$ stays singular until something pins the solution down - typically a strong prior anchoring the first pose, which is what this repo's scripts do (e.g. `pose_graph.py` "heavily anchoring node 0"). Levenberg-Marquardt's damping (§11) also keeps the system solvable.

Then:

$$x_{\text{new}} = x_{\text{old}}+\Delta x$$

And repeat.

---

## 5. The most intuitive interpretation

You can think of Gauss-Newton as:

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

The Jacobian answers:

> **"If I slightly change each parameter, how will my errors change?"**

That's why the Jacobian is so important.

---

## 6. Tiny numerical example

Suppose we want to find $x$ such that:

$$x^2 = 4$$

Define the residual:

$$r(x)=x^2-4$$

We want:

$$\min_x (x^2-4)^2$$

Suppose our initial guess is:

$$x=3$$

The residual is:

$$r(3)=9-4=5$$

The Jacobian is simply the derivative:

$$J = \frac{dr}{dx}=2x$$

so:

$$J=6$$

Gauss-Newton solves:

$$J^\top J\Delta x=-J^\top r$$

Since everything is scalar:

$$6^2\Delta x=-6(5)$$

$$36\Delta x=-30$$

$$\Delta x=-\tfrac{5}{6}\approx-0.833$$

Therefore:

$$x_{\text{new}}=3-\tfrac{5}{6}\approx2.167$$

We're already much closer to $2$.

Repeat again, and it converges rapidly toward $2$.

---

## 7. Why is it called Gauss-Newton?

You may already know **Newton's method**.

Newton's method uses the **second derivative** (the Hessian) $H$.

For nonlinear least squares, the exact Hessian decomposes as:

$$H=J^\top J + \sum_i r_i \nabla^2 r_i$$

Gauss-Newton says:

> "Let's ignore the second term."

So:

$$H \approx J^\top J$$

and therefore instead of solving

$$H\Delta x=-\nabla f$$

we solve

$$J^\top J\Delta x=-J^\top r$$

The right-hand sides are the same thing: the gradient of $f(x) = \tfrac12\|r(x)\|^2$ is exactly $\nabla f = J^\top r$. Gauss-Newton only changes the left-hand side.

This makes Gauss-Newton **cheaper and particularly well suited to least-squares problems**.

> **Note**: dropping $\sum_i r_i \nabla^2 r_i$ is a good approximation when the residuals are small at the solution (the measurements fit well) or $r$ is only mildly nonlinear. With large residuals or strong nonlinearity, the Gauss-Newton step can overshoot and even increase the cost - exactly what Levenberg-Marquardt's damping (§11) guards against.

---

## 8. Why is this everywhere in SLAM?

This is where it becomes really relevant to you.

Suppose a robot has a pose:

$$T_i$$

and observes a landmark.

Your prediction might be something like:

$$
\hat z_{ij} = h(T_i,p_j)
$$

where:

- $T_i$ = robot pose
- $p_j$ = landmark
- $h(\cdot)$ = camera/measurement model
- $z_{ij}$ = actual measurement of landmark $j$ from pose $i$

Residual:

$$r_{ij} = h(T_i,p_j)-z_{ij}$$

Your SLAM problem becomes:

```math
{\min_{\{T_i\},\{p_j\}}
\sum_{(i,j)\in\mathcal{O}}\|r_{ij}\|^2}
```

where $\mathcal{O}$ is the set of pose-landmark pairs that were actually observed (not every landmark is seen from every pose).

> **Note**: real systems also weight each residual by its measurement's information matrix, $r_{ij}^\top\Omega_{ij}\,r_{ij}$ instead of $\|r_{ij}\|^2$, so precise sensors count more - left out here for readability. See [nonlinear_least_square.md §12](nonlinear_least_square.md#12-add-measurement-uncertainty) and [factor_graph.md](factor_graph.md).

That's a huge nonlinear optimization problem.

Gauss-Newton says:

> "Around my current estimates of the poses and landmarks, approximate all these measurement functions linearly."

So:

$$r_{ij}(x+\Delta x) \approx r_{ij}(x)+J_{ij}\Delta x$$

Then all measurements contribute to a large system:

$$
J^\top J\Delta x=-J^\top r
$$

Solve it → update all poses and landmarks → repeat.

One caveat: landmarks $p_j\in\mathbb{R}^3$ really do just get $p_j \leftarrow p_j+\Delta p_j$. Poses $T_i$ don't - they live on the $SE(3)$ manifold, so the update is a **retraction** through the exponential map, $T_i \leftarrow T_i\cdot\mathrm{Exp}(\Delta x_i)$, not plain addition. See [pose_graph_optimization.md](pose_graph_optimization.md) for the full derivation.

That's essentially the core optimization mechanism behind many **[bundle adjustment](bundle_adjustment.md) and [graph-SLAM](pose_graph_optimization.md) systems**.

---

## 9. A very useful mental picture

Picture §6's example: the residual $r(x) = x^2 - 4$ is a curve, and we're looking for where it hits zero.

Gauss-Newton doesn't try to understand the whole curve.

It says:

> "Near where I am, this curve looks approximately like a straight line."

That straight line is the tangent at the current guess - exactly the linearization $r(x) + J\Delta x$ from §3:

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

Then it moves to where that line says the residual is zero: from $x = 3$ to $x \approx 2.167$. The real curve bends, so that isn't the true root ($x = 2$) - but it's much closer.

From there it builds **another local approximation** (a new tangent at $x \approx 2.167$), and the next jump lands even closer.

So the key idea is:

> **Linearize → solve → move → linearize again.**

(With many residuals, $J\Delta x$ usually can't make every residual exactly zero at once; Gauss-Newton then moves to where the linearized residuals are *as small as possible* in the least-squares sense - the bottom of §2's bowl. The one-residual picture above is the special case where that minimum is exactly zero.)

---

## 10. Gauss-Newton vs Gradient Descent

This distinction is very useful.

### Gradient descent

Gradient descent asks:

> **"Which direction is downhill?"**

and takes a chosen step:

$$\Delta x=-\alpha\nabla f$$

where $\alpha$ is the learning rate.

---

### Gauss-Newton

Gauss-Newton asks:

> **"Given the local shape of the least-squares problem, what step should I take to approximately reach the minimum?"**

$$J^\top J\Delta x=-J^\top r$$

So Gauss-Newton uses much more information about the local geometry.

That's why it can converge much faster near the solution - quadratically for zero-residual problems like §6's, where the term Gauss-Newton drops (§7) vanishes at the solution.

---

## 11. Gauss-Newton vs Newton

A useful hierarchy:

| Method                  | Idea                                            |
| ----------------------- | ----------------------------------------------- |
| **Gradient Descent**    | Follow the slope                                |
| **Newton**              | Use slope + curvature                           |
| **Gauss-Newton**        | Use Jacobian structure to approximate curvature |
| **Levenberg-Marquardt** | Gauss-Newton + damping for robustness           |

In SLAM, you'll frequently encounter:

**Gauss-Newton / Levenberg-Marquardt + sparse linear solver**

because SLAM naturally produces large, sparse least-squares problems.

---

## 12. The one sentence to remember

If you remember only one thing:

> **Gauss-Newton repeatedly approximates a nonlinear least-squares problem as a local linear least-squares problem, solves for the best parameter update, and repeats.**

Or even more intuitively:

> **"I'm going to pretend the world is linear around where I currently am, take the best step according to that approximation, and then update my approximation."**

That idea is the bridge from **[Jacobian](../foundations/jacobian.md) → Gauss-Newton → [bundle adjustment](bundle_adjustment.md) → [pose-graph optimization](pose_graph_optimization.md) → SLAM**.

---

## 13. References

1. Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-40065-5 - the standard textbook treatment of the Gauss-Newton method (Chapter 10) behind this whole doc, including the relationship to Newton's method (§11) and gradient descent (§10) covered here.
