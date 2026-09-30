# Levenberg-Marquardt

**Levenberg-Marquardt (LM)** is especially important in SLAM because it sits between the two ideas you've already seen:

> **Gauss-Newton is fast but can be unstable. Gradient descent is stable but can be slow. LM tries to get the best of both.**

---

## 1. Start with the problem

In SLAM/[bundle adjustment](bundle_adjustment.md)/[factor graphs](factor_graph.md), we usually have:

```math
\min_x \sum_i \|e_i(x)\|^2
```

where:

- $x$ = unknown states, e.g. poses and landmarks
- $e_i(x)$ = measurement error from factor $i$

For example:

```text
       camera observation
              ↓
x1 ●──────── Factor ────────● l1
       "How inconsistent
        is this observation?"
```

We want to adjust $x_1$ and $l_1$ so that the error becomes smaller.

---

## 2. Why not just use [Gauss-Newton](gauss_newton.md)?

Gauss-Newton linearizes the error:

$$e(x+\Delta x)\approx e(x)+J\Delta x$$

Then solves:

$$\boxed{J^\top J\Delta x=-J^\top e}$$

and updates:

$$x\leftarrow x+\Delta x$$

(For a pose $T \in SE(3)$ the update is a retraction, $T \leftarrow T\cdot\mathrm{Exp}(\Delta x)$, not plain addition - see [gauss_newton.md §8](gauss_newton.md#8-why-is-this-everywhere-in-slam).)

This works extremely well **when you're already reasonably close to the solution**.

But suppose your initial estimate is terrible:

```text
True solution
      ★

                 Current estimate
                       ●
```

The local linear approximation is only valid near the current estimate, but Gauss-Newton trusts it fully - so the step it computes can be far too long and overshoot.

Gauss-Newton can then:

```text
Current
   ●
    \
     \
      \      ← huge step
       \
        ●
       diverge
```

(The step's *direction* is still downhill whenever $J$ has full column rank: for $f = \tfrac12\|e\|^2$, $\Delta x^\top\nabla f = -\|J\Delta x\|^2 < 0$ unless you're already at a stationary point. The problem is the step's *length*. And when $J^\top J$ is singular or nearly so, the step can become huge or not exist at all.)

---

## 3. Gradient descent has the opposite problem

Gradient descent says:

> "Just move in the direction that decreases the error."

For $f = \tfrac12\|e\|^2$ the gradient is $\nabla f = J^\top e$, and the step goes straight downhill:

$$\boxed{\Delta x = -\alpha J^\top e}$$

where $\alpha$ is the **step size**. It uses only the **slope**, with no $J^\top J$ (curvature) term.

**Safer:** to first order a step changes the cost by $-\alpha\|\nabla f\|^2 < 0$, so a small enough $\alpha$ always lowers it, and there's no matrix to invert.

**Slow:** $\alpha$ must stay small enough not to overshoot in the steepest direction, so in a long, narrow valley the steps zigzag across it and crawl along it:

```text
        ●
         ╲    ╱╲    ╱╲    ╱╲
          ╲  ╱  ╲  ╱  ╲  ╱  ╲
           ╲╱    ╲╱    ╲╱    ╲ · · · → ★
   ↕ steep across the valley, → flat along it
```

For example, $f = \tfrac12(x^2 + 100y^2)$ is 100× steeper in $y$, so $\alpha$ must stay below $2/100 = 0.02$. Starting from $(10, 1)$:

- $\alpha = 0.019$ needs **361 steps** to get within $0.01$ of the minimum.
- $\alpha = 0.021$ **diverges**.
- Gauss-Newton gets there in **one step**, because $J^\top J$ rescales each direction by its own curvature. Here the errors are $e = (x,\ 10y)$, so $J = \text{diag}(1, 10)$ and $J^\top J = \text{diag}(1, 100)$.

§5 shows that LM with a large $\lambda$ becomes exactly this step with $\alpha = 1/\lambda$, and §12 shows how Marquardt's scaling removes this valley problem when the valley runs along the parameter axes, as it does here.

---

## 4. LM's brilliant idea

LM introduces a **damping parameter**:

$$\boxed{(J^\top J+\lambda I)\Delta x=-J^\top e}$$

Compare:

### Gauss-Newton

$$J^\top J\Delta x=-J^\top e$$

### Levenberg-Marquardt

$$\boxed{(J^\top J+\lambda I)\Delta x=-J^\top e}$$

That tiny:

$$\lambda I$$

makes a huge conceptual difference.

It also fixes a problem Gauss-Newton can't handle on its own. $J^\top J$ is only positive *semi*-definite, and in SLAM it's often singular (see [gauss_newton.md §4](gauss_newton.md#4-the-optimization-problem)). For any $\lambda > 0$, $v^\top(J^\top J+\lambda I)v = \|Jv\|^2 + \lambda\|v\|^2 > 0$ for every $v \neq 0$, so the damped system always has a unique solution.

---

## 5. What does $\lambda$ actually do?

Think of $\lambda$ as a **caution knob**.

### Large $\lambda$

The algorithm becomes cautious:

$$J^\top J+\lambda I$$

is dominated by $\lambda I$, so the solution is approximately

$$\Delta x \approx -\frac{1}{\lambda}J^\top e$$

Since $J^\top e$ is the gradient of $\tfrac12\|e\|^2$, this is a gradient-descent step with step size $1/\lambda$ (compare [gauss_newton.md §10](gauss_newton.md#10-gauss-newton-vs-gradient-descent)'s $-\alpha\nabla f$) - and a large $\lambda$ means a *short* step.

The behavior becomes similar to **gradient descent**.

```text
Large λ

● → → → → → ★
small/cautious steps
```

### Small $\lambda$

The damping becomes almost irrelevant:

$$J^\top J+\lambda I\approx J^\top J$$

So LM behaves like **Gauss-Newton**.

```text
Small λ

● ───────→ ★
larger, faster steps
```

Therefore:

```math
\boxed{\text{LM} \approx \begin{cases} \text{Gradient Descent}, & \lambda\text{ large}\\ 
\text{Gauss-Newton}, & \lambda\text{ small}\end{cases}}
```

That's the most important intuition.

---

## 6. Think of driving a car

Imagine you're driving toward a destination, but your map is uncertain.

### Far from the destination

You don't trust your local estimate very much.

So:

> "Let's move carefully."

→ large $\lambda$

### Near the destination

Your local model is reliable.

> "Great, I can take a more direct step."

→ small $\lambda$

So LM slides along one dial, with gradient descent and Gauss-Newton as its two ends:

```text
CAUTIOUS                                AGGRESSIVE
large λ ←────────────── LM ──────────────→ small λ
gradient-descent-like            Gauss-Newton-like
```

---

## 7. How does LM decide whether to be cautious?

This is one of its nicest features.

Suppose we compute a step:

$$\Delta x$$

We try it.

### If the error decreases

```text
Before: error = 100
After:  error = 60
```

Great!

We **accept** the step and trust our local model more.

So:

$$\lambda \downarrow$$

The next iteration behaves more like Gauss-Newton.

---

### If the error gets worse

```text
Before: error = 100
After:  error = 150
```

Oops.

Our predicted step wasn't good, so we **reject** it - $x$ stays exactly where it was.

Then:

$$\lambda \uparrow$$

and we re-solve for a more conservative step from the same point, repeating until the error actually decreases (or a retry limit is hit and the solver stops). Because a bad step is never kept, the error never goes up from one accepted step to the next.

Conceptually:

```text
             Try step
                │
        ┌───────┴───────┐
        ↓               ↓
     Error ↓         Error ↑
        │               │
   accept step     reject step
   (x updated)    (x unchanged)
        │               │
   λ decreases     λ increases
        │               │
   more GN-like   re-solve from
                   the same x,
                  more cautious
```

---

## 8. An intuitive landscape

Imagine the optimization landscape:

```text
Error
  ↑
  │                            ●  current estimate (far away)
  │  \                   __   /
  │   \                 /  \_/
  │    \       __      /
  │     \     /  \____/
  │      \_★_/
  └──────────────────────────────────→ x
           ↑
        optimum
```

If you're far away, the landscape can be highly nonlinear.

Gauss-Newton says:

> "I'll approximate the landscape locally and take a potentially large step."

LM says:

> "Let's not trust that approximation too much yet."

As you get closer:

```text
Error
  ↑
  │ ●  current estimate (close)
  │  \         /
  │   \       /
  │    \__★__/
  └──────────────→ x
          ↑
       optimum
```

the local approximation becomes more trustworthy, and LM gradually behaves more like Gauss-Newton.

---

## 9. Why this matters enormously in SLAM

SLAM is highly nonlinear because of:

- rotations
- camera projection
- perspective
- 3D geometry
- $SE(3)$ transformations
- reprojection errors

For example:

$$u =\pi(TX)$$

where:

- $X$ = 3D landmark
- $T$ = camera pose, written here as world-to-camera, so $TX$ is the landmark in the camera frame. [bundle_adjustment.md](bundle_adjustment.md) uses the inverse convention: a camera-to-world $T$, and $\pi(T^{-1}P)$.
- $\pi$ = camera projection

The reprojection error is:

$$e = \pi(TX)-z$$

(prediction minus measurement, the same convention as [gauss_newton.md](gauss_newton.md)'s residual $r$)

and we optimize:

```math
\min_{T,X}\|\pi(TX)-z\|^2
```

This is nonlinear.

LM is therefore frequently useful for:

- bundle adjustment
- nonlinear least squares
- factor-graph optimization
- pose estimation
- calibration
- SLAM

---

## 10. LM and your factor graph

Take the small factor graph from [factor_graph.md §5](factor_graph.md#5-why-call-it-a-graph), where landmark $l_0$ is seen from $x_0$ and $x_1$, and $l_1$ from $x_1$ and $x_2$:

```text
       l0    l1
       ●     ●
      / \   / \
     /   \ /   \
    ●─────●─────●
    x0    x1    x2
```

Each factor produces an error:

$$e_1(x), e_2(x), e_3(x), ...$$

Together, each weighted by its measurement's information matrix $\Omega_i$ (the same weighting as [factor_graph.md §4](factor_graph.md#4-optimization-means-minimizing-all-those-errors)):

```math
E(x)=\sum_i e_i(x)^\top\Omega_i\, e_i(x)
```

Linearize:

$$e(x+\Delta x)\approx e(x)+J\Delta x$$

where $e$ and $J$ stack every factor's error and Jacobian, and $\Omega$ is block-diagonal with the $\Omega_i$ on its diagonal.

Then LM solves:

$$\boxed{(J^\top\Omega J+\lambda I)\Delta x = -J^\top\Omega e}$$

and updates the states.

So the pipeline is:

```text
Factor graph
      ↓
Measurement errors
      ↓
Nonlinear least squares
      ↓
Linearization
      ↓
Jacobian J
      ↓
LM
      ↓
Solve for Δx
      ↓
Update poses/landmarks
      ↓
Repeat
```

---

## 11. LM vs Gauss-Newton

The simplest comparison:

|                 | Gauss-Newton                        | Levenberg-Marquardt           |
| --------------- | ----------------------------------- | ----------------------------- |
| Basic idea      | Trust local quadratic approximation | Control how much you trust it |
| Step            | More aggressive                     | Adaptive                      |
| Stability       | Can be fragile far from solution    | Generally more robust         |
| Near solution   | Very fast                           | Approaches GN                 |
| Extra parameter | None                                | $\lambda$                   |
| SLAM usage      | Very common                         | Very common                   |

Think:

```text
                 LM
                /  \
               /    \
      Gradient Descent  Gauss-Newton
          ↑                 ↑
       cautious          aggressive
```

LM is the **middle ground**.

---

## 12. One subtle but important point

You may see the equation written as:

$$(J^\top J+\lambda I)\Delta x=-J^\top e$$

or:

$$(J^\top J+\lambda D)\Delta x=-J^\top e$$

where $D$ might be:

$$D=\text{diag}(J^\top J)$$

This is Marquardt's version: each parameter is damped in proportion to its own curvature, which makes the step independent of how you scale individual parameters (e.g. metres vs. millimetres) - plain $\lambda I$ doesn't have that property. With a large $\lambda$ it becomes a gradient-descent step scaled *per parameter*, rather than §5's plain one. For §3's example, $D = \text{diag}(J^\top J) = \text{diag}(1, 100)$, so with a large $\lambda$ the step becomes $-\tfrac{1}{\lambda}D^{-1}\nabla f = -\tfrac{1}{\lambda}(x,\ y)$: both directions shrink at the same rate, and the zigzag is gone. That only works because §3's valley runs along the parameter axes. A valley running diagonally to them keeps its zigzag, because a diagonal $D$ can't rescale a tilted direction.

One catch: a parameter that no measurement constrains has a zero on that diagonal and gets no damping at all, so implementations clamp the diagonal to a small positive floor.

Different implementations use slightly different damping schemes.

The **concept** remains the same:

> Add damping to make the optimization step safer.

---

## 13. Connection to everything you've learned

You can now build a nice mental hierarchy:

```text
                 SLAM
                  │
             Factor Graph
                  │
          Nonlinear least squares
                  │
             Linearization
                  │
             Jacobian J
                  │
        ┌─────────┴─────────┐
        │                   │
   Gauss-Newton             LM
        │                   │
   Fast/local        Adaptive damping
                            │
                 ┌──────────┴──────────┐
                 ↓                     ↓
          Gradient-descent-like    GN-like
             when λ is large          when λ is small
```

And then:

```text
Batch optimization
       │
       ├── Gauss-Newton
       └── Levenberg-Marquardt

Incremental optimization
       │
       └── iSAM/iSAM2
```

---

## 14. The one-sentence intuition

> **Levenberg-Marquardt is Gauss-Newton with a safety knob: when the nonlinear problem is difficult, it takes cautious gradient-descent-like steps; when the solution becomes trustworthy, it reduces the damping and behaves like fast Gauss-Newton.**

That's why **GN, LM, factor graphs, and iSAM** fit together so naturally in modern SLAM.

---

## 15. Where this is implemented in this repo

Every LM solver here uses §12's Marquardt form, $(H + \lambda\,\text{diag}(H))\Delta x = -J^\top\Omega e$ with $H = J^\top\Omega J$ (plus the pose-graph anchor below), clamps that diagonal to at least $10^{-12}$, and follows §7's rule literally: a trial step is kept only if it lowers the total cost; otherwise it's rejected, $\lambda$ doubles, and the step is re-solved from the same point (at most 10 tries per iteration). Poses are updated by retraction (§2), landmarks by plain addition.

- [`pose_graph.py`](../../use_numpy/pose_graph.py)'s `run_pose_graph_optimization` (both `use_numpy/` and `use_manif/`): starts from `--damping` (default $0.01$); after an accepted step $\lambda \leftarrow \max(\lambda/2,\ 10^{-7})$, after a rejected one $\lambda \leftarrow \max(2\lambda,\ 10^{-6})$. It also adds a large constant, $10^6 I$, to the first pose's block of $H$ - pinning that pose removes the gauge freedom, so $H$ isn't singular to begin with. [`pose_graph_incremental.py`](../../use_numpy/pose_graph_incremental.py) reuses it for its batch baseline (its incremental solver runs plain Gauss-Newton), and [`sliding_window_marginalization.py`](../../use_numpy/sliding_window_marginalization.py) calls it with an initial damping of $0$, so its first step is a pure Gauss-Newton step (after that, the floors above keep $\lambda \ge 10^{-7}$).
- [`bundle_adjustment_advanced.py`](../../use_numpy/bundle_adjustment_advanced.py) (both versions): starts from $\lambda = 10^{-3}$; halves it (floor $10^{-7}$) after an accepted step and doubles it (no floor) after a rejected one. All reprojection residuals are weighted equally, and a trial step that puts any point behind a camera counts as infinite cost, so it's always rejected.

---

## 16. References

1. Levenberg, K. (1944). *A Method for the Solution of Certain Non-Linear Problems in Least Squares*. Quarterly of Applied Mathematics, 2(2), 164–168. https://doi.org/10.1090/qam/10666 - the original damped least-squares method behind §4's $(J^\top J+\lambda I)\Delta x=-J^\top e$.
2. Marquardt, D. W. (1963). *An Algorithm for Least-Squares Estimation of Nonlinear Parameters*. Journal of the Society for Industrial and Applied Mathematics, 11(2), 431–441. https://doi.org/10.1137/0111030 - the scale-invariant diagonal-damping variant $D=\text{diag}(J^\top J)$ behind §12.
3. Triggs, B., McLauchlan, P. F., Hartley, R. I., & Fitzgibbon, A. W. (2000). *Bundle Adjustment - A Modern Synthesis*. In Vision Algorithms: Theory and Practice (LNCS vol. 1883, pp. 298–372). Springer. https://doi.org/10.1007/3-540-44480-7_21 - the reprojection-error/bundle-adjustment application of LM behind §9.
4. Dellaert, F., & Kaess, M. (2017). *Factor Graphs for Robot Perception*. Foundations and Trends in Robotics, 6(1–2), 1–139. https://doi.org/10.1561/2300000043 - general reference for the factor-graph formulation behind §1 and §10.
