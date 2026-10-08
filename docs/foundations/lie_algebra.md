# Lie Algebra

## 1. The big idea

> **Lie group = the actual transformations we can perform.**

> **Lie algebra = the small, local motions that generate those transformations.**

For robotics:

- Rotation matrix → an actual orientation
- Quaternion → an actual orientation ([quaternion.md](quaternion.md))
- $SE(3)$ transformation → an actual robot pose
- Lie algebra → a convenient way to describe **tiny changes** to those poses

An analogy:

> **Lie group = the curved globe itself.**

> **Lie algebra = a flat local map (the tangent plane) at one spot.**

No flat map covers the whole globe without distortion, but the local one is much easier to do calculus with.

---

## 2. Why do we need Lie algebra?

Take a robot orientation $R \in SO(3)$ that rotates by a small amount $\delta\theta$. We would like to update it as $R_{\text{new}} = R_{\text{old}} + \delta R$.

The problem: a rotation matrix must satisfy $R^\top R=I$ and $\det(R)=1$, and adding numbers to it generally breaks both. We could end up with

```math
\begin{bmatrix}1.01 & 0 & 0\\ 
0 & 1.02 & 0\\ 
0 & 0 & 1\end{bmatrix}
```

which isn't a valid rotation.

**Lie algebra lets us represent the small change while automatically respecting the geometry of rotations.**

---

## 3. Ordinary vectors

A robot position and a small movement are plain vectors:

```math
p =\begin{bmatrix}x\\ 
y\\ 
z\end{bmatrix},
\qquad
\delta p = \begin{bmatrix}\delta x\\ 
\delta y\\ 
\delta z\end{bmatrix}
```

so we can simply update $p_{\text{new}} = p + \delta p$. Position space is flat:

```text
       δp
   ──────────►
 ●────────────●
old          new
```

---

## 4. Rotations are different

Turning 10° and then another 10° about the *same* axis is harmless: the angles add up to 20°, in either order. The trouble starts when the axes differ, because **the meaning of a rotation depends on the current orientation**.

- $R_1R_2 \neq R_2R_1$ (the famous **non-commutativity of rotations**).
- 90° around X then 90° around Y is generally not the same as 90° around Y then 90° around X.

So rotation space is curved, not Euclidean.

---

## 5. Lie group: the space of valid transformations

For rotations, the Lie group is $SO(3)$: it contains **all valid 3D rotations**.

```text
                 SO(3)
          ┌─────────────────┐
          │                 │
          │   all possible  │
          │    rotations    │
          │                 │
          └─────────────────┘
```

A rotation matrix is a point in this space. Optimizing directly on it is inconvenient.

> **$SO(3)$ tells us where we are.**

---

## 6. Lie algebra: zoom in locally

Now suppose we are at some rotation $R$. Instead of thinking about all possible rotations, we zoom in around $R$, where small rotations behave approximately like ordinary vectors. Every nearby rotation can be written as $R\exp(\delta\theta^\wedge)$ for a small 3-vector (where $`\delta\theta^\wedge`$ is the **hat operator**'s skew-symmetric matrix, Section 7, and $`\exp`$ the **matrix exponential**, Section 8; see the [glossary](../glossary.md#1-geometry-and-lie-groups)):

```math
\delta\theta =\begin{bmatrix}\delta\theta_x\\ 
\delta\theta_y\\ 
\delta\theta_z \end{bmatrix}
```

The space these small motions live in is the Lie algebra $\mathfrak{so}(3)$.

> **$\mathfrak{so}(3)$ tells us how we can move from there.**

Technically:

- $\mathfrak{so}(3)$ is the tangent space *at the identity*; its elements are $3\times3$ skew-symmetric matrices.
- The 3-vector $\delta\theta$ holds their coordinates, related by the hat operator of Section 7.
- Near some other $R$ we reuse the same algebra by composing with $R$, as above, so no separate algebra is needed at every point.

---

## 7. The weird-looking skew-symmetric matrix

We will meet this matrix:

```math
\delta\theta^\wedge = \begin{bmatrix} 0 & -\delta\theta_z & \delta\theta_y\\
\delta\theta_z & 0 & -\delta\theta_x\\ 
-\delta\theta_y & \delta\theta_x & 0 \end{bmatrix}
```

The map $(\cdot)^\wedge : \mathbb{R}^3 \rightarrow \mathfrak{so}(3)$ is called the **hat operator**. The reason for the odd shape is its key property:

$$\delta\theta^\wedge v = \delta\theta \times v$$

so the matrix is just the **cross product** written as a matrix.

---

## 8. The exponential map

We have a rotation vector $\delta\theta$ (an axis times an angle) and want a real rotation matrix. The **exponential map** turns a local motion into an actual transformation.

**Intuition:**

- Picture spinning at a constant angular velocity $`\delta\theta`$ for 1 second. Where you end up is $`R`$.
- The axis of $`\delta\theta`$ is the spin axis. Its length is the angle turned.
- The result is always a valid rotation, never a stretched or skewed matrix.

In symbols:

$$R = \exp(\delta\theta^\wedge)$$

```text
Lie algebra                     Lie group

small rotation                  actual rotation
δθ                               R
 │                               │
 │       exponential             │
 └──────────────────────────────►│
```

**Notation.** Other docs in this repo, the Solà et al. reference, and the `manif` library write the same maps directly on vectors, with capital letters: $`\mathrm{Exp}(\delta\theta) = \exp(\delta\theta^\wedge)`$ and $`\mathrm{Log}(R) = \log(R)^\vee`$ (the vee operator is defined in Section 9).

This is exact for any rotation vector, not just small ones. With $\theta = \lVert\delta\theta\rVert$ it has a closed form (Rodrigues' formula), the same kind of expression as the Jacobians in [jacobian.md §11.4](jacobian.md#114-closed-form-for-so3):

```math
\exp(\delta\theta^\wedge) = I + \frac{\sin\theta}{\theta}\,\delta\theta^\wedge + \frac{1-\cos\theta}{\theta^2}\,(\delta\theta^\wedge)^2
```

For a small rotation it reduces to $R \approx I+\delta\theta^\wedge$, the first-order form behind the small update $R\exp(\delta\theta^\wedge)$ of Section 10.

---

## 9. The logarithm map

The **logarithm map** does the reverse of Section 8: it turns an actual transformation back into a local motion.

```text
Lie group                       Lie algebra

actual rotation                 small rotation
 R                               δθ
 │                               │
 │       logarithm               │
 └──────────────────────────────►│
```

Given two rotations $R_1$ and $R_2$, "what small rotation takes us from $R_1$ to $R_2$?" is answered by

$$\delta\theta^\wedge = \log(R_1^{-1}R_2)$$

and $\log$ undoes the exponential map: $\log(\exp(\delta\theta^\wedge)) = \delta\theta^\wedge$.

- This holds as long as $`\|\delta\theta\| < \pi`$ (at exactly $\pi$, $\log$ is ambiguous).
- Beyond that range, multiple $\delta\theta$'s map to the same $R$ (the exponential is periodic per rotation axis), and $\log$ recovers only the one on its principal branch.
- To pull the plain vector out of the skew-symmetric matrix we apply the inverse of the hat operator, the **vee operator** $(\cdot)^\vee$, so $\delta\theta = (\log(R))^\vee$.

**Worked number: a 90° yaw.** Take $\delta\theta = (0,0,\pi/2)$. Then

```math
R = \exp(\delta\theta^\wedge) = \begin{bmatrix} 0 & -1 & 0\\
1 & 0 & 0\\
0 & 0 & 1 \end{bmatrix},
\qquad
\mathrm{Log}(R) = (0,\,0,\,\pi/2)
```

so Exp builds the matrix and Log recovers the original 3-vector (checked numerically).

This is how we turn "the difference between two poses" into a plain vector we can measure, weight, and feed into a least-squares solver, which is what pose-graph optimization does with every edge residual.

The same idea applies to $SE(3)$: $\xi^\wedge = \log(T_1^{-1}T_2)$ (where $`\xi = [\rho, \phi]`$ is the 6-vector defined in Section 11; the [glossary](../glossary.md#1-geometry-and-lie-groups) writes it $[v, \omega]$) gives the 6D motion that separates two poses $T_1$ and $T_2$.

---

## 10. Why this matters for SLAM

Suppose the optimizer says our estimated orientation $R$ is wrong by a small amount. Instead of optimizing the 9 elements of $R$, we optimize only $\delta\theta \in \mathbb{R}^3$ and update

$$R_{\text{new}}=R\exp(\delta\theta^\wedge)$$

The optimizer sees an ordinary 3-vector, while the resulting rotation remains valid.

---

## 11. $SE(3)$: this is where robotics gets really interesting

A robot pose is a translation plus a rotation:

```math
T = \begin{bmatrix}R & t\\ 
0 & 1 \end{bmatrix} \in SE(3)
```

$SE(3)$ is the **Lie group of 3D rigid-body transformations**. Its Lie algebra is $\mathfrak{se}(3)$, and a small pose perturbation is

```math
\xi = \begin{bmatrix} \rho\\ 
\phi \end{bmatrix} \in \mathbb{R}^6
```

- $\rho$: translational part (the actual translation it produces is $V\rho$, where $`V`$ is the matrix explained below)
- $\phi$: tiny rotation

So one 6D vector represents a tiny change in the entire robot pose:

```text
ξ
│
├── translation:  ρ  (≈ Δx Δy Δz for small φ, since V ≈ I)
│
└── rotation:     φ  (Δrx Δry Δrz)
```

Just like $\mathfrak{so}(3)$ has a hat operator (Section 7), $\mathfrak{se}(3)$ has its own, turning the 6-vector $\xi$ into a $4\times4$ matrix, where $\phi^\wedge$ is the same $3\times3$ skew-symmetric block as before:

```math
\xi^\wedge = \begin{bmatrix} \phi^\wedge & \rho\\ 
0 & 0 \end{bmatrix}
```

**Easy to get wrong:** $\exp(\xi^\wedge)$ is *not* "exponentiate the rotation part and copy the translation unchanged." Rotation and translation are coupled: sweeping a small rotation while translating traces a curve, not a straight line. The closed form is

```math
\exp(\xi^\wedge) = \begin{bmatrix} \exp(\phi^\wedge) & V\rho\\ 
0 & 1 \end{bmatrix}
```

- $V$ is a $3\times3$ matrix, built purely from $\phi$, that "bends" the raw translation $\rho$ to account for the coupling. Its exact formula isn't the point here.
- We can't glue the $SO(3)$ exponential and the raw translation together; the $SE(3)$ exponential genuinely mixes the two.
- The log map has the mirror-image subtlety: recovering $\rho$ from a pose needs $V^{-1}$, not just the translation column.

**Tiny example:** drive forward at 1 m/s while turning left at 90°/s, for 1 s.

- Here $`\rho = (1,0,0)`$ and $`\phi = (0,0,\pi/2)`$. The robot traces a quarter circle.
- It ends at $`V\rho = (2/\pi,\ 2/\pi,\ 0) \approx (0.637,\ 0.637,\ 0)`$, and its heading has turned by 90°.
- So $`V`$ bends the straight line $`\rho`$ into an arc. The endpoint is not $`(1,0,0)`$.
- For small $`\phi`$ the arc is almost straight, so $`V \approx I`$.

The pose update is then

$$T_{\text{new}} = T_{\text{old}}\exp(\xi^\wedge)$$

This is the foundation of many **pose-graph optimization, bundle adjustment, visual-inertial estimation, and SLAM** implementations.

---

## 12. A physical analogy

Imagine holding a drone with pose $T$. Someone says:

> "Move forward 2 cm, left 1 cm, rotate 0.5° around X, and rotate 0.2° around Z."

With the common robotics body-frame convention (x forward, y left, z up), that is approximately a **Lie algebra vector**:

```math
\xi \approx \begin{bmatrix} 2\,\text{cm}\\ 
1\,\text{cm}\\ 
0\\ 
0.5^\circ\\ 
0\\ 
0.2^\circ \end{bmatrix}
```

- It describes a **small motion**, not a complete pose.
- The translation entries are $\rho$, so the actual translation $V\rho$ matches the quoted numbers only approximately (since $V \approx I$ for small $\phi$).
- The cm/degrees are for intuition only; plugging into $\exp(\xi^\wedge)$ requires meters and radians.

We then apply that motion to the drone's current pose:

$$\boxed{ \text{current pose} + \text{small motion} \rightarrow \text{new pose}}$$

except that the "+" is replaced by the appropriate Lie-group operation.

---

## 13. Why not just use Euler angles?

We could optimize $x,y,z,\text{roll},\text{pitch},\text{yaw}$ directly, but Euler angles bring singularities (gimbal lock), awkward composition, coordinate-dependent behavior, and problematic derivatives. The Lie algebra gives a **local minimal representation** of the perturbation while the state stays on the correct manifold.

---

## 14. The four objects

| Concept   | Intuition                 |
| --------- | ------------------------- |
| $SO(3)$ | All possible 3D rotations |
| $\mathfrak{so}(3)$ | Small rotational motions  |
| $SE(3)$ | All possible 3D poses     |
| $\mathfrak{se}(3)$ | Small 6-DoF pose motions  |

$$\boxed{SO(3) \underset{\exp}{\overset{\log}{\rightleftarrows}} \mathfrak{so}(3)}$$

$$\boxed{SE(3) \underset{\exp}{\overset{\log}{\rightleftarrows}} \mathfrak{se}(3)}$$

---

## 15. The connection to Jacobians

In optimization we typically have an error $\text{error} = f(T)$ and want to know: "if I slightly change the pose, how does the error change?" So we introduce $\delta\xi \in \mathbb{R}^6$ and approximate

$$f(T\exp(\delta\xi^\wedge)) \approx f(T)+J\delta\xi$$

where $J$ is the **Jacobian with respect to the Lie-algebra perturbation**. Whether the perturbation is applied on the right, as here, or on the left changes $J$; [jacobian.md §11](jacobian.md#11-left-and-right-jacobians-sensitivity-on-a-curved-space) explains the left and right Jacobians this leads to.

That's why Lie algebra appears everywhere in modern SLAM: it gives optimization algorithms a **locally Euclidean 6D space** to work in, while the actual pose remains a valid element of $SE(3)$.

---

## 16. The one-sentence intuition

> **A Lie group represents the actual robot state (rotation/pose), while its Lie algebra represents small changes around that state, allowing us to use ordinary vector calculus and optimization without breaking the geometry of the state.**

In pictures: the state sits on $SE(3)$ (curved), we zoom in to $\xi \in \mathbb{R}^6$ (flat, the tree of Section 11), and $\exp()$ brings the result back as a new valid pose.

---

## 17. References

1. Solà, J., Deray, J., & Atchuthan, D. (2018). *A micro Lie theory for state estimation in robotics*. arXiv:1812.01537. https://doi.org/10.48550/arXiv.1812.01537 - the standard modern reference for the $SO(3)$/$`SE(3)`$ $Exp$/$`Log`$ conventions used throughout this doc, written by (among others) the author of the `manif` library this repo's `use_manif/` scripts are built on.
