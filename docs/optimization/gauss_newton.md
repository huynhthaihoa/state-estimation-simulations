# Gauss-Newton Optimization

Since you’re working with **SLAM, [bundle adjustment](bundle_adjustment.md), and [pose-graph optimization](pose_graph_optimization.md)**, Gauss-Newton is one of the most important optimization ideas to understand intuitively.

---

## 1. The basic idea

Suppose you want to find parameters $x$ that make some measurements fit as well as possible.

For example, in SLAM:

> “What robot pose $x$ best explains these sensor measurements?”

You define a **residual**:

$$r(x) = \text{prediction}(x) - \text{measurement}$$

and want to minimize the total squared error:

$$\min_x \frac12 \|r(x)\|^2$$

The problem is that $r(x)$ is usually **nonlinear**.

Gauss-Newton says:

> **“I can't solve this nonlinear problem directly, so around my current guess, I'll pretend it is linear, solve that easier problem, move there, and repeat.”**

That's essentially the whole algorithm.

---

## 2. Imagine you're lost on a mountain

Imagine you're standing somewhere on a complicated mountain landscape.

Your goal:

> **Find the lowest point.**

You don't know the entire landscape.

But you can look at the terrain immediately around you.

If the terrain looks approximately like a **tilted bowl**, you can estimate:

> “If I move this direction, I'll go downhill.”

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

$$r(x)$$

gets replaced locally by the linear approximation:

$$r(x)+J\Delta x$$

So instead of asking:

> "What $x$ minimizes this complicated nonlinear function?"

we ask:

> "What small change $\Delta x$ makes this local linear approximation as small as possible?"

---

## 4. The optimization problem

Originally:

$$\min_x \frac12\|r(x)\|^2$$

After linearization:

$$\min_{\Delta x} \frac12 \|r + J\Delta x\|^2$$

Now this is a **linear least-squares problem**.

We can solve it analytically.

Taking the derivative and setting it to zero gives:

$$J^T J\Delta x = -J^T r$$

This is the famous **Gauss-Newton equation**.

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

$$J^TJ\Delta x=-J^Tr$$

Since everything is scalar:

$$6^2\Delta x=-6(5)$$

$$36\Delta x=-30$$

$$\Delta x=-0.833$$

Therefore:

$$x_{\text{new}}=3-0.833=2.167$$

We're already much closer to $2$.

Repeat again, and it converges rapidly toward $2$.

---

## 7. Why is it called Gauss-Newton?

You may already know **Newton's method**.

Newton's method uses the **second derivative** (the Hessian) $H$.

For nonlinear least squares, the exact Hessian decomposes as:

$$H=J^TJ + \sum_i r_i \nabla^2 r_i$$

Gauss-Newton says:

> "Let's ignore the second term."

So:

$$H \approx J^TJ$$

and therefore instead of solving

$$H\Delta x=-\nabla f$$

we solve

$$J^TJ\Delta x=-J^Tr$$

This makes Gauss-Newton **cheaper and particularly well suited to least-squares problems**.

---

## 8. Why is this everywhere in SLAM?

This is where it becomes really relevant to you.

Suppose a robot has a pose:

$$T_i$$

and observes a landmark.

Your prediction might be something like:

$$
\hat z = h(T_i,p_j)
$$

where:

* $T_i$ = robot pose
* $p_j$ = landmark
* $h(\cdot)$ = camera/measurement model
* $z$ = actual measurement

Residual:

$$r = h(T_i,p_j)-z$$

Your SLAM problem becomes:

$$
{\min_{\{T_i\},\{p_j\}}
\sum_{i,j}\|r_{ij}\|^2}
$$

That's a huge nonlinear optimization problem.

Gauss-Newton says:

> "Around my current estimates of the poses and landmarks, approximate all these measurement functions linearly."

So:

$$r_{ij}(\Delta x) \approx r_{ij}+J_{ij}\Delta x$$

Then all measurements contribute to a large system:

$$
J^TJ\Delta x=-J^Tr
$$

Solve it → update all poses and landmarks → repeat.

One caveat: landmarks $p_j\in\mathbb{R}^3$ really do just get $p_j \leftarrow p_j+\Delta p_j$. Poses $T_i$ don't - they live on the $SE(3)$ manifold, so the update is a **retraction** through the exponential map, $T_i \leftarrow T_i\cdot\mathrm{Exp}(\Delta x_i)$, not plain addition. See [pose_graph_optimization.md](pose_graph_optimization.md) for the full derivation.

That's essentially the core optimization mechanism behind many **[bundle adjustment](bundle_adjustment.md) and [graph-SLAM](pose_graph_optimization.md) systems**.

---

## 9. A very useful mental picture

Think of the real nonlinear function as a complicated curved road:

```text
        Real nonlinear function
             __
           /    \__
        __/        \___
     __/                \__
___/                       \____
              ↑
         current point
```

Gauss-Newton doesn't try to understand the whole road.

It says:

> "Near where I am, this road looks approximately like this."

```text
             /
            /
           /
----------●----------
       local approximation
```

Then it finds the best direction to move.

After moving:

```text
             /
            /
           /
----------●----------
              ↓
          new position
```

Then it builds **another local approximation**.

So the key idea is:

> **Linearize → solve → move → linearize again.**

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

$$J^TJ\Delta x=-J^Tr$$

So Gauss-Newton uses much more information about the local geometry.

That's why it can converge much faster near the solution.

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
