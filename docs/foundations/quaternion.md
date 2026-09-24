# Quaternion intuitive explanation

For **SLAM, state estimation, and robotics**, the most useful way to understand a quaternion is as a clever mathematical way to represent **3D orientation** without some of the problems of Euler angles.

## 1. First: what problem is a quaternion solving?

Imagine a robot:

* It can point **left/right** → yaw
* It can tilt **up/down** → pitch
* It can roll sideways → roll

So we need to describe its orientation in 3D.

The obvious approach is Euler angles:

> "Rotate 30° around X, then 20° around Y, then 10° around Z."

But there is a problem: **the rotations interact with each other**.

For example, rotating around X and then Y is not generally the same as rotating around Y and then X.

This creates problems such as **gimbal lock** and awkward interpolation.

A quaternion gives us another representation.

---

## 2. The intuitive idea: a quaternion is an "axis + angle" in disguise

Suppose I tell you:

> "Rotate the robot by **90° around the Z-axis**."

That's actually enough information to define an orientation change.

We have:

* **Axis:** $(0, 0, 1)$
* **Angle:** $90°$

A quaternion essentially packages these two things into four numbers:

$$
q = (w,x,y,z)
$$

For a rotation by angle $\theta$ around a unit axis

$$
\mathbf{u}=(u_x,u_y,u_z)
$$

the quaternion is

$$
q =
\left(
\cos\frac{\theta}{2},
u_x\sin\frac{\theta}{2},
u_y\sin\frac{\theta}{2},
u_z\sin\frac{\theta}{2}
\right)
$$

Notice the **half angle**.

For our 90° Z rotation:

$$
q =
\left(
\cos45^\circ,
0,
0,
\sin45^\circ
\right)
$$

so approximately

$$
q=(0.707,0,0,0.707)
$$

You can think of it as:

> **"Rotate around this axis by this amount."**

The four numbers are just a convenient mathematical encoding of that idea.

---

## 3. Why four numbers?

This is initially confusing because a 3D orientation seems like it should need only **three numbers**.

After all, $(\text{roll}, \text{pitch}, \text{yaw})$ are three numbers. But quaternions use four: $(w, x, y, z)$.

The important point is that **not every four-number quaternion represents a rotation**.

For a rotation quaternion, we require:

$$
w^2+x^2+y^2+z^2=1
$$

So there is a constraint.

Effectively:

> 4 numbers + 1 constraint → 3 degrees of freedom.

That's why a quaternion can represent a 3D orientation even though it has four components.

---

## 4. The most important intuition: don't think of it as a mysterious 4D object

When you're learning robotics, I'd recommend **not initially thinking of a quaternion as "a point in 4D space."**

Instead think:

> **Quaternion = compact representation of a 3D rotation.**

For example:

$$
q=(1,0,0,0)
$$

means:

> **No rotation.**

And:

$$
q=(0.707,0,0,0.707)
$$

means approximately:

> **90° rotation around Z.**

And:

$$
q=(0.707,0.707,0,0)
$$

means:

> **90° rotation around X.**

---

## 5. Why is there a $w$?

This is where the formula $q = \left(\cos\frac{\theta}{2}, \mathbf{u}\sin\frac{\theta}{2}\right)$ is useful.

The quaternion has two conceptual parts:

$$
\boxed{w=\cos(\theta/2)}
$$

and

$$
\boxed{(x,y,z)=\mathbf{u}\sin(\theta/2)}
$$

So:

**$w$** tells us something about the **amount of rotation**, while **$x,y,z$** encode the **rotation axis weighted by the amount of rotation**.

For example, at zero rotation ($\theta = 0$), $w = \cos 0 = 1$ and $x = y = z = 0$, giving $q = (1, 0, 0, 0)$.

As the rotation increases, the vector part grows.

---

## 6. Why are quaternions so useful in robotics?

This is where they become really interesting for SLAM/state-estimation work.

Suppose your IMU tells you:

> "The robot rotated slightly during this 10 ms interval."

You want to update:

$$
R_{k+1}=R_k\Delta R
$$

where $R$ is the robot's orientation.

You can represent $R$ as a rotation matrix (${R \in SO(3)}$), but that's **9 numbers**.

A quaternion only needs $q = (w,x,y,z)$ with the unit constraint.

So quaternions give you a compact representation of rotation.

More importantly, composing rotations becomes quaternion multiplication:

$${q_{\text{new}}=q_{\text{old}}\otimes\Delta q}$$

Conceptually:

> **Quaternion multiplication = "apply one rotation after another."**

That's incredibly useful for IMU integration.

---

## 7. A beautiful geometric intuition

Imagine holding a phone.

Initially:

```text
      ↑
      |
   ┌─────┐
   │PHONE│
   └─────┘
```

Now rotate it 90° around the Z-axis:

```text
          ┌─────┐
          │PHONE│
          └─────┘
              →
```

Instead of storing:

> "The phone has roll = ?, pitch = ?, yaw = 90°"

you can simply describe the transformation as:

> **Rotate 90° around this axis.**

The quaternion stores exactly that information in a form that is mathematically convenient for chaining rotations.

---

## 8. Why not just use rotation matrices?

You might reasonably ask:

> "If rotation matrices work, why bother with quaternions?"

A rotation matrix:

$$R= \begin{bmatrix} r_{11} & r_{12} & r_{13}\\
r_{21} & r_{22} & r_{23}\\
r_{31} & r_{32} & r_{33} \end{bmatrix}$$

has **9 elements**, even though a rotation has only 3 degrees of freedom.

And those 9 numbers must satisfy several constraints: $R^\top R = I$ and $\det(R) = 1$.

Quaternions have only four numbers and one simple normalization constraint: $\|q\| = 1$.

So they're generally:

* more compact
* numerically convenient
* efficient for composing rotations
* excellent for interpolation
* free of gimbal lock

---

## 9. Quaternion vs Euler angles

This is probably the most useful mental comparison:

| Representation  | Intuition                         | Main problem                  |
| --------------- | --------------------------------- | ----------------------------- |
| Euler angles    | Roll + pitch + yaw                | Gimbal lock, order dependence |
| Rotation matrix | Transform coordinate axes         | 9 numbers + constraints       |
| Quaternion      | Axis + angle encoded in 4 numbers | Less intuitive initially      |

Think of them as **different languages describing the same orientation**.

For example:

$${\text{90° around Z}}$$

can be represented as:

- **Euler**

$${(roll,pitch,yaw)=(0,0,90^\circ)}$$

- **Rotation matrix**

$$R= \begin{bmatrix} 0 & -1 & 0\\
1 & 0 & 0\\
0 & 0 & 1 \end{bmatrix}$$

- **Quaternion**

$${q=(0.707,0,0,0.707)}$$

Same physical rotation. Different mathematical representation.

---

## 10. One subtle but VERY important fact

There is a strange property: $q$ and $-q$ represent **the exact same physical orientation**.

For example, $q = (0.707, 0, 0, 0.707)$ and $-q = (-0.707, 0, 0, -0.707)$ represent the same rotation.

This becomes particularly important when working with **optimization, SLAM, EKF, and Lie-group state estimation**, because treating quaternion components as ordinary Euclidean coordinates can cause problems.

---

## 11. The connection to Kalman filtering

This is actually where quaternions become especially relevant.

In an EKF, you might have a state like

$${\mathbf{x}= \begin{bmatrix} p\\ 
v\\ 
q\\ 
b_g\\ 
b_a \end{bmatrix}}$$

where:

* $p$ = position
* $v$ = velocity
* $q$ = orientation quaternion
* $b_g$ = gyro bias
* $b_a$ = accelerometer bias

The tricky part is:

> **A quaternion does not live in ordinary 4D Euclidean space.**

It lives on the **unit quaternion manifold**, which represents rotations.

That's why modern VIO/SLAM systems often use a **small 3D orientation error** rather than directly adding a 4D quaternion error:

$${q_{\text{true}} = \delta q\otimes q_{\text{estimate}}}$$

where $\delta q$ represents a **small 3D rotational error**.

This idea leads directly into **SO(3), Lie groups, Lie algebra, and the Invariant EKF** you were asking about earlier. In the terminology of [left_right_invariant.md](../filtering/left_right_invariant.md), this particular $\delta q$ is a *right*-invariant error (the mismatch viewed from a fixed point in the world frame) - see that doc for when you'd instead want the left-invariant, body-frame version.

---

## 12. The one-sentence intuition

If you remember only one thing:

> **A quaternion is a clever four-number representation of a 3D rotation, essentially encoding "rotate by this angle around this axis," in a form that makes chaining and estimating rotations much easier.**

And in a robotics context, it's worth mentally organizing it as:

$${ \boxed{ \text{Euler angles} \rightarrow \text{Quaternion} \rightarrow SO(3) \rightarrow \mathfrak{so}(3) \rightarrow \text{Lie-group state estimation} } }
$$

The really interesting next step is **why quaternion multiplication actually performs rotation**, because that is the part that makes quaternions initially feel like magic. That's the next section.

---

## 13. Why quaternion multiplication actually performs rotation

So far we've treated a quaternion as "axis + angle in disguise." But that raises a fair question:

> "Why does *multiplying* four numbers together rotate a 3D vector?"

It isn't magic. It comes down to four observations.

### Step 1: the multiplication rule hides a dot product and a cross product

Write a quaternion as a scalar part plus a vector part:

$$
q = (w, \mathbf{v}), \qquad \mathbf{v} = (x, y, z)
$$

Quaternion multiplication (the Hamilton product, built on the rule $i^2 = j^2 = k^2 = ijk = -1$) then works out to:

$$
q_1 \otimes q_2 = \left( w_1 w_2 - \mathbf{v}_1 \cdot \mathbf{v}_2,\; w_1 \mathbf{v}_2 + w_2 \mathbf{v}_1 + \mathbf{v}_1 \times \mathbf{v}_2 \right)
$$

Look at what's inside:

* a **dot product** $\mathbf{v}_1 \cdot \mathbf{v}_2$
* a **cross product** $\mathbf{v}_1 \times \mathbf{v}_2$

Those are exactly the ingredients of 3D rotation formulas. The cross product also makes the multiplication **order-dependent** ($q_1 \otimes q_2 \neq q_2 \otimes q_1$ in general) - just like the rotations in Section 1.

### Step 2: put the vector inside a quaternion, then "sandwich" it

To rotate a vector $\mathbf{v}$, first turn it into a quaternion with zero scalar part (a **pure quaternion**):

$$
\mathbf{v} \;\rightarrow\; (0, \mathbf{v})
$$

Then multiply by $q$ on the left and by its **conjugate** $q^{\ast} = (w, -x, -y, -z)$ on the right:

$$
\boxed{(0, \mathbf{v}') = q \otimes (0, \mathbf{v}) \otimes q^{\ast}}
$$

For a unit quaternion, $q \otimes q^{\ast} = (1, 0, 0, 0)$, so $q^{\ast}$ is also the inverse of $q$ - it undoes the rotation.

Why two multiplications? Why not just $q \otimes (0, \mathbf{v})$?

Because multiplying from **one side** usually knocks the vector out of 3D. Take our 90° Z rotation $q = (0.707, 0, 0, 0.707)$ and follow two vectors through, writing every result as $(w, x, y, z)$:

| Input vector | After $q \otimes (0, \mathbf{v})$ | After $q \otimes (0, \mathbf{v}) \otimes q^{\ast}$ |
| --- | --- | --- |
| X-axis $(1, 0, 0)$ | $(0, 0.707, 0.707, 0)$ | $(0, 0, 1, 0)$ |
| Z-axis $(0, 0, 1)$ | $(-0.707, 0, 0, 0.707)$ | $(0, 0, 0, 1)$ |

Two things to notice:

* The **Z-axis** (the rotation axis itself) picks up a nonzero $w = -0.707$ after one multiplication. It is no longer a pure quaternion - it has leaked out of 3D space. The second multiplication by $q^{\ast}$ cancels the leak and returns it exactly to where it started, as a rotation should leave its own axis alone.
* The **X-axis** lands on $(0.707, 0.707, 0)$ after one multiplication - rotated by only **45°**. The second multiplication adds another 45°, giving $(0, 1, 0)$: the full **90°**.

So each side of the sandwich does **half** of the rotation. That is the real reason for the half angle in Section 2.

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

$$
\mathbf{u} \otimes \mathbf{v}_{\perp} = -\,\mathbf{v}_{\perp} \otimes \mathbf{u}
$$

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

$$
q \otimes \mathbf{v}_{\perp} \otimes q^{\ast} = \cos\theta\,\mathbf{v}_{\perp} + \sin\theta\,(\mathbf{u} \times \mathbf{v}_{\perp})
$$

That is, $`\mathbf{v}_{\perp}`$ rotated by $\theta$ within the plane perpendicular to $`\mathbf{u}`$: $`\mathbf{v}_{\perp}`$ and $`\mathbf{u} \times \mathbf{v}_{\perp}`$ are perpendicular and the same length, so they act like the $x$ and $y$ axes of that plane.

Putting both parts together:

$$
\boxed{q \otimes \mathbf{v} \otimes q^{\ast} = \mathbf{v}_{\parallel} + \cos\theta\,\mathbf{v}_{\perp} + \sin\theta\,(\mathbf{u} \times \mathbf{v}_{\perp})}
$$

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

In 3D, one quaternion multiplication *can* leak out of 3D space (Step 2), so a second multiplication is needed to cancel the leak, and the rotation angle gets split between the two sides.

|  | 2D: complex numbers | 3D: unit quaternions |
| --- | --- | --- |
| Rotate a vector | $e^{i\theta} z$ | $q \otimes \mathbf{v} \otimes q^{\ast}$ |
| Multiplications | 1 | 2 (one per side) |
| Angle stored | $\theta$ | $\theta/2$ |

> **A unit quaternion rotates a vector by sandwiching it: each side turns the part perpendicular to the axis by $\theta/2$, the part along the axis is untouched, and the two halves add up to a full rotation by $\theta$.**

---

## 14. References

1. Diebel, J. (2006). *Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors*. Stanford University Technical Report. https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf - a widely-cited technical reference covering the quaternion/Euler-angle/rotation-vector conversions and conventions this doc builds intuition for.
