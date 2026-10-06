# Quaternion intuitive explanation

For **SLAM, state estimation, and robotics**, the most useful way to understand a quaternion is as a clever mathematical way to represent **3D orientation** without some of the problems of Euler angles.

---

## 1. First: what problem is a quaternion solving?

Take a robot. It can point **left/right** (yaw), tilt **up/down** (pitch), and roll sideways (roll). We need to describe that orientation in 3D. The obvious approach is Euler angles:

> "Rotate 30° around X, then 20° around Y, then 10° around Z."

But three angles cause two separate problems:

- **Gimbal lock.** At certain orientations (for the common yaw-pitch-roll order, pitch $= \pm 90°$) two of the three rotation axes line up, and one degree of freedom disappears: different angle combinations give the same orientation, and small orientation changes can need large angle jumps. It's a singularity of the three-angle description, not of rotation itself.
- **Order conventions.** Rotating around X and then Y is not the same as around Y and then X, so every set of Euler angles only means something together with its convention (XYZ, ZYX, ...). Interpolating the three angles separately also gives awkward paths.

Quaternions avoid gimbal lock, because a unit quaternion has no singular orientation. They don't make rotation order-independent (nothing can: composing rotations is inherently order-dependent, Section 13), but they make composition a single multiplication (once a convention is fixed).

---

## 2. The intuitive idea: a quaternion is an "axis + angle" in disguise

Suppose we say:

> "Rotate the robot by **90° around the Z-axis**."

That is enough to define an orientation change. We have an **axis** $(0, 0, 1)$ and an **angle** $90°$. A quaternion $q = (w,x,y,z)$ packs these two things into four numbers. For a rotation by angle $\theta$ around a unit axis $\mathbf{u}=(u_x,u_y,u_z)$:

$$
q =
\left(
\cos\frac{\theta}{2},
u_x\sin\frac{\theta}{2},
u_y\sin\frac{\theta}{2},
u_z\sin\frac{\theta}{2}
\right)
$$

Notice the **half angle**. For our running example, 90° around Z:

$$
q = (\cos45^\circ, 0, 0, \sin45^\circ) \approx (0.707,0,0,0.707)
$$

So a quaternion reads as **"rotate around this axis by this amount."** The four numbers are just a convenient encoding of that idea.

> **Watch the component order and the convention.**
>
> - **Component order.** This doc writes quaternions scalar-first, $(w, x, y, z)$. Many libraries store them scalar-last, $[x, y, z, w]$: scipy's `Rotation.as_quat()`, ROS messages, and this repo's own `utils.py`. Reading one ordering as the other silently gives a different rotation.
> - **Multiplication convention.** **Hamilton** (used here, and by most robotics libraries) and **JPL** (common in older aerospace and MSCKF-era VIO papers) compose in opposite orders. Solà (2017) compares them.
> - Always check both before mixing code or equations from different sources.

---

## 3. Why four numbers?

A 3D orientation seems to need only **three numbers**, like $(\text{roll}, \text{pitch}, \text{yaw})$, yet a quaternion has four: $(w, x, y, z)$. The catch is that **not every four-number quaternion is a rotation**. A rotation quaternion must satisfy

$$
w^2+x^2+y^2+z^2=1
$$

> 4 numbers + 1 constraint → 3 degrees of freedom.

---

## 4. Read it as a 3D rotation, not a 4D point

At first, it helps **not to think of a quaternion as "a point in 4D space"**. Think of it as a **compact representation of a 3D rotation**:

| Quaternion $(w,x,y,z)$ | Meaning |
| --- | --- |
| $(1,0,0,0)$ | No rotation |
| $(0.707,0,0,0.707)$ | About 90° around Z |
| $(0.707,0.707,0,0)$ | About 90° around X |

---

## 5. Why is there a $w$?

From $q = \left(\cos\frac{\theta}{2}, \mathbf{u}\sin\frac{\theta}{2}\right)$, the quaternion has two parts:

- $w=\cos(\theta/2)$ gives the **amount of rotation**: $\theta = 2\arccos(w)$. Check with the 90° example: $w = 0.707$ gives $\theta = 2 \cdot 45° = 90°$.
- $(x,y,z)=\mathbf{u}\sin(\theta/2)$ is the **rotation axis weighted by the amount of rotation**.

At zero rotation ($\theta = 0$), $w = \cos 0 = 1$ and $x = y = z = 0$, giving $q = (1, 0, 0, 0)$. As the rotation grows, the vector part grows.

---

## 6. Why are quaternions so useful in robotics?

Suppose the IMU says "the robot rotated slightly during this 10 ms interval." We update the orientation $R_{k+1}=R_k\Delta R$. A rotation matrix ($`R \in SO(3)`$) needs **9 numbers**, while a quaternion needs only $q = (w,x,y,z)$ with the unit constraint. More importantly, composing rotations becomes quaternion multiplication, $q_{\text{new}}=q_{\text{old}}\otimes\Delta q$:

> **Quaternion multiplication = "apply one rotation after another."**

This is the operation at the heart of IMU orientation integration.

---

## 7. A geometric intuition

Hold a phone with its "up" arrow pointing up:

```text
      ↑
   ┌─────┐
   │PHONE│
   └─────┘
```

Now rotate it +90° around the Z-axis (Z pointing out of the page, so positive is counterclockwise by the right-hand rule). The "up" arrow now points left:

```text
   ┌───┐
   │ P │
   │ H │
 ← │ O │
   │ N │
   │ E │
   └───┘
```

Instead of storing "roll = ?, pitch = ?, yaw = 90°", we describe the transformation as **rotate 90° around this axis**. The quaternion stores exactly that, in a form that is convenient for chaining rotations.

---

## 8. Why not just use rotation matrices?

> "If rotation matrices work, why bother with quaternions?"

A rotation matrix

```math
R= \begin{bmatrix} r_{11} & r_{12} & r_{13}\\
r_{21} & r_{22} & r_{23}\\
r_{31} & r_{32} & r_{33} \end{bmatrix}
```

has **9 elements**, even though a rotation has only 3 degrees of freedom, and they must satisfy several constraints: $R^\top R = I$ and $\det(R) = 1$. A quaternion has four numbers and one simple normalization constraint: $`\|q\| = 1`$. So quaternions are:

- more compact
- numerically convenient: rounding makes $\lVert q\rVert$ drift slowly from 1 over many multiplications, so implementations renormalize regularly, which is much cheaper than re-orthogonalizing a rotation matrix
- efficient for composing rotations
- easy to interpolate smoothly: slerp (spherical linear interpolation) $\text{slerp}(q_0, q_1, t)$ moves at constant angular speed along the shortest arc (choose the sign of $q_1$ so that $q_0 \cdot q_1 \ge 0$, see Section 10)
- free of gimbal lock

---

## 9. Quaternion vs Euler angles

Think of them as **different languages describing the same orientation**:

| Representation  | Intuition                         | Main problem                  |
| --------------- | --------------------------------- | ----------------------------- |
| Euler angles    | Roll + pitch + yaw                | Gimbal lock, order dependence |
| Rotation matrix | Transform coordinate axes         | 9 numbers + constraints       |
| Quaternion      | Axis + angle encoded in 4 numbers | Less intuitive initially      |

For our 90° around Z:

- **Euler:** $(\text{roll},\text{pitch},\text{yaw})=(0,0,90^\circ)$
- **Quaternion:** $q=(0.707,0,0,0.707)$
- **Rotation matrix:**

```math
R= \begin{bmatrix} 0 & -1 & 0\\
1 & 0 & 0\\
0 & 0 & 1 \end{bmatrix}
```

---

## 10. q and -q are the same rotation

Strangely, $q$ and $-q$ represent **the exact same physical orientation**. For example, $q = (0.707, 0, 0, 0.707)$ and $-q = (-0.707, 0, 0, -0.707)$ are the same rotation.

This matters for **optimization, SLAM, EKF, and Lie-group state estimation**, because treating quaternion components as ordinary Euclidean coordinates causes two problems:

- **Additive updates break the unit-norm constraint.** Adding a small correction to $q$ leaves the unit sphere.
- **The double cover makes averaging and differencing ambiguous.** Averaging $q$ and $-q$ (the same rotation) gives $(0,0,0,0)$, which is not a rotation. Their component-wise difference is as large as possible.

---

## 11. The connection to Kalman filtering

Quaternions matter most inside filters. In an EKF, we might have a state like

```math
\mathbf{x}= \begin{bmatrix} p\\ 
v\\ 
q\\ 
b_g\\ 
b_a \end{bmatrix}
```

where:

- $p$ = position
- $v$ = velocity
- $q$ = orientation quaternion
- $b_g$ = gyro bias
- $b_a$ = accelerometer bias

The tricky part is:

> **A quaternion does not live in ordinary 4D Euclidean space.**

It lives on the **unit quaternion manifold**. That's why modern VIO/SLAM systems often estimate a **small 3D rotational error** $\delta q$ rather than directly adding a 4D quaternion error.

The small error can be attached on either side, and both are common:

- $`q_{\text{true}} = \delta q\otimes \hat q`$: the error is expressed in the world frame. In the terminology of [left_right_invariant.md](../filtering/left_right_invariant.md) this is a *right*-invariant error.
- $`q_{\text{true}} = \hat q\otimes \delta q`$: the error is expressed in the body frame. This is the *left*-invariant error, the convention of Solà's ESKF and of this repo's `run_iekf` (the $\hat X\,\mathrm{Exp}(\xi)$ form).

This is the error-state idea of [extra_kf_variants.md §2](../filtering/extra_kf_variants.md#2-error-state-kalman-filter-eskf), and it leads directly into **SO(3), Lie groups, Lie algebra, and the Invariant EKF**.

---

## 12. Summary

> **A quaternion is a clever four-number representation of a 3D rotation, essentially encoding "rotate by this angle around this axis," in a form that makes chaining and estimating rotations much easier.**

In a robotics context, it's worth organizing it as:

$$
\boxed{ \text{Euler angles} \rightarrow \text{Quaternion} \rightarrow SO(3) \rightarrow \mathfrak{so}(3) \rightarrow \text{Lie-group state estimation} }
$$

That chain is a learning order, not a hierarchy: unit quaternions are a Lie group in their own right, and each rotation in $SO(3)$ corresponds to exactly two of them, $q$ and $-q$ (Section 10). Both groups share the same small-motion space, so the tools in [lie_algebra.md](lie_algebra.md) apply to either.

The next question is **why quaternion multiplication actually performs rotation**.

---

## 13. Why quaternion multiplication actually performs rotation

So far we've treated a quaternion as "axis + angle in disguise." That raises a fair question:

> "Why does *multiplying* four numbers together rotate a 3D vector?"

The answer comes down to four observations.

### Step 1: the multiplication rule hides a dot product and a cross product

Write a quaternion as a scalar part plus a vector part:

$$
q = (w, \mathbf{v}), \qquad \mathbf{v} = (x, y, z)
$$

**Tiny example:** the rule $`i^2 = j^2 = k^2 = ijk = -1`$ gives $`ij = k`$ but $`ji = -k`$. Swapping the order flips the sign.
- Expand $`(w_1 + \mathbf{v}_1)(w_2 + \mathbf{v}_2)`$ term by term and collect the pieces.
- The $`-1`$ from $`i^2`$ becomes the dot product, and the $`ij = k`$ terms become the cross product.

In symbols, quaternion multiplication (the Hamilton product, built on the rule $i^2 = j^2 = k^2 = ijk = -1$) works out to:

```math
q_1 \otimes q_2 = \left( w_1 w_2 - \mathbf{v}_1 \cdot \mathbf{v}_2,\; w_1 \mathbf{v}_2 + w_2 \mathbf{v}_1 + \mathbf{v}_1 \times \mathbf{v}_2 \right)
```

Look at what's inside:

- a **dot product** $\mathbf{v}_1 \cdot \mathbf{v}_2$
- a **cross product** $\mathbf{v}_1 \times \mathbf{v}_2$

Those are exactly the ingredients of 3D rotation formulas. The cross product also makes the multiplication **order-dependent** ($q_1 \otimes q_2 \neq q_2 \otimes q_1$ in general) - just like the rotations in Section 1.

### Step 2: put the vector inside a quaternion, then "sandwich" it

To rotate a vector $\mathbf{v}$, first turn it into a quaternion with zero scalar part (a **pure quaternion**):

```math
\mathbf{v} \;\rightarrow\; (0, \mathbf{v})
```

Then multiply by $q$ on the left and by its **conjugate** $q^{\ast} = (w, -x, -y, -z)$ on the right:

$$
\boxed{(0, \mathbf{v}') = q \otimes (0, \mathbf{v}) \otimes q^{\ast}}
$$

For a unit quaternion, $q \otimes q^{\ast} = (1, 0, 0, 0)$, so $q^{\ast}$ is also the inverse of $q$ - it undoes the rotation.

Why two multiplications? Why not just $q \otimes (0, \mathbf{v})$?

Because multiplying from **one side** leaks out of 3D whenever the vector has a component along the axis. Take our 90° Z rotation $q = (0.707, 0, 0, 0.707)$ and follow two vectors through, writing every result as $(w, x, y, z)$:

| Input vector | After $q \otimes (0, \mathbf{v})$ | After $q \otimes (0, \mathbf{v}) \otimes q^{\ast}$ |
| --- | --- | --- |
| X-axis $(1, 0, 0)$ | $(0, 0.707, 0.707, 0)$ | $(0, 0, 1, 0)$ |
| Z-axis $(0, 0, 1)$ | $(-0.707, 0, 0, 0.707)$ | $(0, 0, 0, 1)$ |

Two things to notice:

- The **Z-axis** (the rotation axis itself) picks up a nonzero $w = -0.707$ after one multiplication. It is no longer a pure quaternion, because the Z-axis is entirely along the rotation axis. The second multiplication by $q^{\ast}$ cancels the leak and returns it exactly to where it started, as a rotation should leave its own axis alone.
- The **X-axis** (perpendicular to the axis) stays pure and lands on $(0.707, 0.707, 0)$ after one multiplication, rotated by only **45°**. The second multiplication adds another 45°, giving $(0, 1, 0)$: the full **90°**.

So each side of the sandwich does **half** of the rotation. That is the reason for the half angle in Section 2.

### Step 3: why the sandwich rotates by exactly $\theta$

The example generalizes. From here on, a vector inside a product means its pure quaternion, e.g. $\mathbf{v}$ stands for $(0, \mathbf{v})$.

Let $q = \left(\cos\frac{\theta}{2}, \mathbf{u}\sin\frac{\theta}{2}\right)$ and split $\mathbf{v}$ into a part along the axis and a part perpendicular to it:

$$
\mathbf{v} = \mathbf{v}_{\parallel} + \mathbf{v}_{\perp}
$$

**Parallel part.** Two pure quaternions along the same direction commute (their cross product is zero). So $q$ commutes with $\mathbf{v}_{\parallel}$ and the sandwich collapses:

$$
q \otimes \mathbf{v}_{\parallel} \otimes q^{\ast} = \mathbf{v}_{\parallel} \otimes q \otimes q^{\ast} = \mathbf{v}_{\parallel}
$$

The axis doesn't move - exactly what a rotation around that axis should do.

**Perpendicular part.** Here $`\mathbf{u} \cdot \mathbf{v}_{\perp} = 0`$, and swapping the order of a cross product flips its sign. Plugging this into the Step 1 rule shows that $`\mathbf{u}`$ and $`\mathbf{v}_{\perp}`$ **anticommute**:

```math
\mathbf{u} \otimes \mathbf{v}_{\perp} = -\,\mathbf{v}_{\perp} \otimes \mathbf{u}
```

That sign flip lets $q^{\ast}$ move across $`\mathbf{v}_{\perp}`$, turning into $q$ on the way:

$$
\mathbf{v}_{\perp} \otimes q^{\ast} = q \otimes \mathbf{v}_{\perp}
\quad\Rightarrow\quad
q \otimes \mathbf{v}_{\perp} \otimes q^{\ast} = q \otimes q \otimes \mathbf{v}_{\perp}
$$

And multiplying $q$ by itself doubles its angle (Step 1 rule plus the double-angle identities):

$$
q \otimes q = \left(\cos\theta,\ \mathbf{u}\sin\theta\right)
$$

Multiplying that into $`\mathbf{v}_{\perp}`$ with the Step 1 rule (the dot product vanishes because $`\mathbf{u} \perp \mathbf{v}_{\perp}`$) gives:

```math
q \otimes \mathbf{v}_{\perp} \otimes q^{\ast} = \cos\theta\,\mathbf{v}_{\perp} + \sin\theta\,(\mathbf{u} \times \mathbf{v}_{\perp})
```

That is, $`\mathbf{v}_{\perp}`$ rotated by $\theta$ within the plane perpendicular to $`\mathbf{u}`$: $`\mathbf{v}_{\perp}`$ and $`\mathbf{u} \times \mathbf{v}_{\perp}`$ are perpendicular and the same length, so they act like the $x$ and $y$ axes of that plane.

Putting both parts together:

```math
\boxed{q \otimes \mathbf{v} \otimes q^{\ast} = \mathbf{v}_{\parallel} + \cos\theta\,\mathbf{v}_{\perp} + \sin\theta\,(\mathbf{u} \times \mathbf{v}_{\perp})}
```

This is exactly **Rodrigues' rotation formula** - the same rotation a rotation matrix would produce.

> **Each side of the sandwich turns the perpendicular part by $\theta/2$, so the quaternion stores $\theta/2$ in order to produce a rotation by $\theta$.**

### Step 4: two facts from earlier sections, now explained

**Chaining rotations (Section 6).** The conjugate of a product reverses the order: $(p \otimes q)^{\ast} = q^{\ast} \otimes p^{\ast}$. So rotating by $q$ and then by $p$ gives:

$$
p \otimes \left(q \otimes \mathbf{v} \otimes q^{\ast}\right) \otimes p^{\ast} = (p \otimes q) \otimes \mathbf{v} \otimes (p \otimes q)^{\ast}
$$

Two rotations in a row are the single rotation $p \otimes q$. That's why "quaternion multiplication = apply one rotation after another" works. Note that the **rightmost** quaternion acts first, just like the rightmost matrix in $R_k \Delta R$.

**$q$ and $-q$ (Section 10).** Flip the sign of $q$, and the sign appears twice in the sandwich, so it cancels:

$$
(-q) \otimes \mathbf{v} \otimes (-q)^{\ast} = q \otimes \mathbf{v} \otimes q^{\ast}
$$

So every rotation has exactly two quaternions, $q$ and $-q$. In angle terms: $\theta$ and $\theta + 360°$ are the same rotation, but the half angles $\theta/2$ and $\theta/2 + 180°$ flip the sign of every component of $q$.

### Why 2D doesn't need a sandwich

Compare with rotation in 2D using complex numbers. Multiplying by $e^{i\theta} = \cos\theta + i\sin\theta$ rotates a point by $\theta$ with **one** multiplication, because multiplying two complex numbers always gives another point in the same plane - nothing can leak out.

In 3D, one quaternion multiplication leaks out of 3D whenever the vector has a component along the axis (Step 2), so a second multiplication is needed to cancel the leak, and the rotation angle gets split between the two sides.

|  | 2D: complex numbers | 3D: unit quaternions |
| --- | --- | --- |
| Rotate a vector | $e^{i\theta} z$ | $q \otimes \mathbf{v} \otimes q^{\ast}$ |
| Multiplications | 1 | 2 (one per side) |
| Angle stored | $\theta$ | $\theta/2$ |

> **A unit quaternion rotates a vector by sandwiching it: each side turns the part perpendicular to the axis by $\theta/2$, the part along the axis is untouched, and the two halves add up to a full rotation by $\theta$.**

---

## 14. References

1. Diebel, J. (2006). *Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors*. Stanford University Technical Report. https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf - a widely-cited technical reference covering the quaternion/Euler-angle/rotation-vector conversions and conventions this doc builds intuition for.
2. Solà, J. (2017). *Quaternion kinematics for the error-state Kalman filter*. arXiv:1711.02508. https://arxiv.org/abs/1711.02508 - the Hamilton vs. JPL comparison in Section 2's note, and the body-frame quaternion error of Section 11.
