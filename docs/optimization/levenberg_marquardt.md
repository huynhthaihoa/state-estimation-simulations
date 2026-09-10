# Levenberg–Marquardt

**Levenberg–Marquardt (LM)** is especially important in SLAM because it sits between the two ideas you've already seen:

> **Gauss–Newton is fast but can be unstable. Gradient descent is stable but can be slow. LM tries to get the best of both.**

---

## 1. Start with the problem

In SLAM / [bundle adjustment](bundle_adjustment.md) / [factor graphs](factor_graph.md), we usually have:

$$\min_x \sum_i \|e_i(x)\|^2$$

where:

* $x$ = unknown states, e.g. poses and landmarks
* $e_i(x)$ = measurement error from factor $i$

For example:

```text
       camera observation
              ↓
X1 ●──────── Factor ────────● L1
       "How inconsistent
        is this observation?"
```

We want to adjust $X_1$ and $L_1$ so that the error becomes smaller.

---

## 2. Why not just use [Gauss–Newton](gauss_newton.md)?

Gauss–Newton linearizes the error:

$$e(x+\Delta x)\approx e(x)+J\Delta x$$

Then solves:

$$\boxed{J^T J\Delta x=-J^Te}$$

and updates:

$$x\leftarrow x+\Delta x$$

This works extremely well **when you're already reasonably close to the solution**.

But suppose your initial estimate is terrible:

```text
True solution
      ★

                 Current estimate
                       ●
```

The local linear approximation might point in a bad direction.

Gauss–Newton can then:

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

---

## 3. Gradient descent has the opposite problem

Gradient descent says:

> "Just move in the direction that decreases the error."

It tends to be safer:

```text
        ★
      ↗
    ↗
  ●
```

But it can be painfully slow:

```text
● → → → → → → → ★
```

especially for problems with very different scales in different directions.

---

## 4. LM's brilliant idea

LM introduces a **damping parameter**:

$$\boxed{(J^TJ+\lambda I)\Delta x=-J^Te}$$

Compare:

### Gauss–Newton

$$J^TJ\Delta x=-J^Te$$

### Levenberg–Marquardt

$$\boxed{(J^TJ+\lambda I)\Delta x=-J^Te}$$

That tiny:

$$\lambda I$$

makes a huge conceptual difference.

---

## 5. What does $\lambda$ actually do?

Think of $\lambda$ as a **caution knob**.

### Large $\lambda$

The algorithm becomes cautious:

$$J^TJ+\lambda I$$

is dominated by $\lambda I$.

The behavior becomes similar to **gradient descent**.

```text
Large λ

● → → → → → ★
small/cautious steps
```

### Small $\lambda$

The damping becomes almost irrelevant:

$$J^TJ+\lambda I\approx J^TJ$$

So LM behaves like **Gauss–Newton**.

```text
Small λ

● ───────→ ★
larger, faster steps
```

Therefore:

$$\boxed{\text{LM} \approx \begin{cases} \text{Gradient Descent}, & \lambda\text{ large}\\ 
\text{Gauss–Newton}, & \lambda\text{ small}\end{cases}}$$

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

So LM dynamically changes between:

```text
CAUTIOUS
   ↓
Gradient Descent
   ↓
  LM
   ↓
Gauss–Newton
   ↓
AGGRESSIVE
```

---

## 7. How does LM decide whether to be cautious?

This is one of its nicest features.

Suppose we compute a step:

$$\Delta x$$

We try it.

### If the error decreases significantly

```text
Before: error = 100
After:  error = 60
```

Great!

We trust our local model more.

So:

$$\lambda \downarrow$$

Next iteration behaves more like Gauss–Newton.

---

### If the error gets worse

```text
Before: error = 100
After:  error = 150
```

Oops.

Our predicted step wasn't good.

So:

$$\lambda \uparrow$$

and we try a more conservative step.

Conceptually:

```text
             Try step
                │
        ┌───────┴───────┐
        ↓               ↓
    Error ↓          Error ↑
        │               │
    λ decreases      λ increases
        │               │
   more GN-like     more cautious
```

---

## 8. An intuitive landscape

Imagine the optimization landscape:

```text
Error
  ↑
  │          ●
  │         / \
  │        /   \
  │       /     \
  │   ●──/       \──────
  │
  └────────────────────→ x
              ★
           optimum
```

If you're far away, the landscape can be highly nonlinear.

Gauss–Newton says:

> "I'll approximate the landscape locally and take a potentially large step."

LM says:

> "Let's not trust that approximation too much yet."

As you get closer:

```text
             ★
           /   \
         /       \
       ●           \
```

the local approximation becomes more trustworthy, and LM gradually behaves more like Gauss–Newton.

---

## 9. Why this matters enormously in SLAM

SLAM is highly nonlinear because of:

* rotations
* camera projection
* perspective
* 3D geometry
* SE(3) transformations
* reprojection errors

For example:

$$u =\pi(TX)$$

where:

* $X$ = 3D landmark
* $T$ = camera pose
* $\pi$ = camera projection

The reprojection error is:

$$e = z-\pi(TX)$$

and we optimize:

$$\min_{T,X}\|z-\pi(TX)\|^2$$

This is nonlinear.

LM is therefore frequently useful for:

* bundle adjustment
* nonlinear least squares
* factor-graph optimization
* pose estimation
* calibration
* SLAM

---

## 10. LM and your factor graph

Remember our factor graph:

```text
       L1
       ●
      / \
     /   \
X0  ●─────● X1
     \   /
      \ /
       X2
```

Each factor produces an error:

$$e_1(x), e_2(x), e_3(x), ...$$

Together:

$$E(x)=\sum_i\|e_i(x)\|^2$$

Linearize:

$$e(x+\Delta x)\approx e(x)+J\Delta x$$

Then LM solves:

$$\boxed{(J^TWJ+\lambda I)\Delta x = -J^TWe}$$

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
Update poses / landmarks
      ↓
Repeat
```

---

## 11. LM vs Gauss–Newton

The simplest comparison:

|                 | Gauss–Newton                        | Levenberg–Marquardt           |
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

$$(J^TJ+\lambda I)\Delta x=-J^Te$$

or:

$$(J^TJ+\lambda D)\Delta x=-J^Te$$

where $D$ might be:

$$D=\text{diag}(J^TJ)$$

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
       └── iSAM / iSAM2
```

## 14. The one-sentence intuition

> **Levenberg–Marquardt is Gauss–Newton with a safety knob: when the nonlinear problem is difficult, it takes cautious gradient-descent-like steps; when the solution becomes trustworthy, it reduces the damping and behaves like fast Gauss–Newton.**

That's why **GN, LM, factor graphs, and iSAM** fit together so naturally in modern SLAM.

---

## 15. References

1. Levenberg, K. (1944). *A Method for the Solution of Certain Non-Linear Problems in Least Squares*. Quarterly of Applied Mathematics, 2(2), 164–168. https://doi.org/10.1090/qam/10666 - the original damped least-squares method behind §4's $(J^TJ+\lambda I)\Delta x=-J^Te$.
2. Marquardt, D. W. (1963). *An Algorithm for Least-Squares Estimation of Nonlinear Parameters*. Journal of the Society for Industrial and Applied Mathematics, 11(2), 431–441. https://doi.org/10.1137/0111030 - the scale-invariant diagonal-damping variant $D=\text{diag}(J^TJ)$ behind §12.
3. Triggs, B., McLauchlan, P. F., Hartley, R. I., & Fitzgibbon, A. W. (2000). *Bundle Adjustment - A Modern Synthesis*. In Vision Algorithms: Theory and Practice (LNCS vol. 1883, pp. 298–372). Springer. https://doi.org/10.1007/3-540-44480-7_21 - the reprojection-error / bundle-adjustment application of LM behind §9.
4. Dellaert, F., & Kaess, M. (2017). *Factor Graphs for Robot Perception*. Foundations and Trends in Robotics, 6(1–2), 1–139. https://doi.org/10.1561/2300000043 - general reference for the factor-graph formulation behind §1 and §10.
