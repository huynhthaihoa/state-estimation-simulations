# Lie Algebra

## 1. The big idea

The easiest way to think about **Lie groups and Lie algebras** is:

> **Lie group = the actual transformations you can perform.**

> **Lie algebra = the small, local motions that generate those transformations.**

For robotics:

* Rotation matrix → an actual orientation
* Quaternion → an actual orientation
* $SE(3)$ transformation → an actual robot pose
* Lie algebra → a convenient way to describe **tiny changes** to those poses

A useful analogy:

> **Lie group = a map of the whole Earth.**

> **Lie algebra = a local coordinate system around where you currently are.**

The local coordinate system is much easier to do calculus with.

---

## 2. Why do we need Lie algebra?

Consider a robot orientation.

You might represent it with a rotation matrix:

$$R \in SO(3)$$

Suppose the robot rotates by a small amount:

$$\delta\theta$$

You want to update:

$$R_{\text{new}} = R_{\text{old}} + \delta R$$

But there's a problem.

A rotation matrix has special constraints:

$$R^TR=I$$

and

$$\det(R)=1$$

Simply adding numbers to a rotation matrix will generally destroy these properties.

For example, you could accidentally obtain something like:

$$\begin{bmatrix}1.01 & 0 & 0\\
0 & 1.02 & 0\\
0 & 0 & 1\end{bmatrix}$$

which isn't a valid rotation.

**Lie algebra gives us a way to represent the small change while automatically respecting the geometry of rotations.**

---

## 3. Start with something familiar: ordinary vectors

Suppose your robot position is

$$p =\begin{bmatrix}x\\
y\\
z\end{bmatrix}$$

A small movement is simply:

$$\delta p = \begin{bmatrix}\delta x\\ 
\delta y\\
\delta z\end{bmatrix}$$

and we can update:

$$p_{\text{new}} = p + \delta p$$

Easy.

The space of positions is basically flat:

```text
       δp
   ──────────►
 ●────────────●
old          new
```

So ordinary vector addition works.

---

## 4. Rotations are different

Imagine an arrow pointing forward.

You rotate it by 10°.

Then another 10°.

Then another 10°.

Rotations don't behave like ordinary vectors because **the meaning of a rotation depends on the current orientation**.

For example:

$$R_1R_2 \neq R_2R_1$$

This is the famous **non-commutativity of rotations**.

Rotate:

> 90° around X, then 90° around Y

is not generally the same as:

> 90° around Y, then 90° around X.

So rotation space is curved/non-Euclidean in an important sense.

---

## 5. Lie group: the space of valid transformations

For rotations, the Lie group is:

$$SO(3)$$

It contains **all valid 3D rotations**.

You can imagine it as:

```text
                 SO(3)
          ┌─────────────────┐
          │                 │
          │   all possible  │
          │    rotations    │
          │                 │
          └─────────────────┘
```

A rotation matrix is a point somewhere in this space.

But doing optimization directly on this space is inconvenient.

---

## 6. Lie algebra: zoom in locally

Now imagine you're currently at some rotation $R$.

Instead of thinking about **all possible rotations**, zoom in around $R$.

Locally, small rotations behave approximately like ordinary vectors:

$$\delta\theta =\begin{bmatrix}\delta\theta_x\\
\delta\theta_y\\
\delta\theta_z \end{bmatrix}$$

This is the Lie algebra:

$$\mathfrak{so}(3)$$

So:

> **$SO(3)$ tells you where you are.**

> **$\mathfrak{so}(3)$ tells you how you can move from there.**

That's probably the single most useful intuition.

---

## 7. The weird-looking skew-symmetric matrix

You might encounter:

$$\delta\theta^\wedge = \begin{bmatrix} 0 & -\delta\theta_z & \delta\theta_y\\
\delta\theta_z & 0 & -\delta\theta_x\\
-\delta\theta_y & \delta\theta_x & 0 \end{bmatrix}$$

This is called the **hat operator**:

$$(\cdot)^\wedge$$

It converts:

$$\mathbb{R}^3 \rightarrow \mathfrak{so}(3)$$

Why?

Because this matrix has a very useful property:

$$\delta\theta^\wedge v = \delta\theta \times v$$

So the matrix is essentially a convenient way of representing the **cross product**.

---

## 8. The exponential map

Here's where Lie theory becomes extremely useful.

You have a small rotation:

$$\delta\theta$$

and you want to turn it into a real rotation matrix.

Use the exponential map:

$$R = \exp(\delta\theta^\wedge)$$

Conceptually:

```text
Lie algebra                     Lie group

small rotation                  actual rotation
δθ                              R
 │                               │
 │       exponential             │
 └──────────────────────────────►│
```

So:

> **Exponential map = turn a local motion into an actual transformation.**

For a small rotation:

$$R \approx I+\delta\theta^\wedge$$

This is particularly important for optimization and filtering.

---

## 9. The logarithm map

The exponential map turns a small motion into an actual transformation. The **logarithm map** does the reverse.

Suppose you have two rotations, $R_1$ and $R_2$, and want to know: "what small rotation takes me from $R_1$ to $R_2$?"

You compute:

$$\delta\theta^\wedge = \log(R_1^{-1}R_2)$$

This converts a group element back into a Lie-algebra element - undoing the exponential map:

$$\log(\exp(\delta\theta^\wedge)) = \delta\theta^\wedge$$

(This holds as long as $\|\delta\theta\| < \pi$. The exponential map is periodic per rotation axis, so beyond that range multiple $\delta\theta$'s map to the same $R$, and $\log$ can only recover the one on its principal branch.)

Conceptually:

```text
Lie group                       Lie algebra

actual rotation                 small rotation
R                                δθ
 │                               │
 │       logarithm               │
 └──────────────────────────────►│
```

So:

> **Logarithm map = turn an actual transformation back into a local motion.**

(To pull the plain vector $\delta\theta \in \mathbb{R}^3$ out of the skew-symmetric matrix $\delta\theta^\wedge$, apply the inverse of the hat operator - usually called the **vee operator** $(\cdot)^\vee$ - so $\delta\theta = (\log(R))^\vee$.)

This is exactly how you turn "the difference between two poses" into a plain vector you can measure, weight, and feed into a least-squares solver - which is precisely what pose-graph optimization does with every edge residual.

The same idea applies to $SE(3)$:

$$\xi^\wedge = \log(T_1^{-1}T_2)$$

gives the 6D motion that separates two poses $T_1$ and $T_2$.

---

## 10. Why this is beautiful for SLAM

Suppose your estimated robot orientation is:

$$R$$

and your optimizer says:

> "Your orientation is wrong by a small amount."

Instead of optimizing the 9 elements of $R$, you optimize only:

$$\delta\theta \in \mathbb{R}^3$$

Then update:

$$R_{\text{new}}=R\exp(\delta\theta^\wedge)$$

This is extremely convenient.

The optimizer works with an ordinary 3-dimensional vector:

$$\delta\theta$$

while the resulting rotation remains a valid rotation.

---

## 11. $SE(3)$: this is where robotics gets really interesting

A robot pose consists of:

* translation
* rotation

We represent it as:

$$T = \begin{bmatrix}R & t\\
0 & 1 \end{bmatrix}$$

where:

$$T \in SE(3)$$

This is the **Lie group of 3D rigid-body transformations**.

Its Lie algebra is:

$$\mathfrak{se}(3)$$

and a small pose perturbation can be represented as:

$$\xi = \begin{bmatrix} \rho\\
\phi \end{bmatrix} \in \mathbb{R}^6$$

where:

* $\rho$: tiny translation
* $\phi$: tiny rotation

So one 6D vector represents a tiny change in the entire robot pose:

```text
ξ
│
├── translation:  Δx Δy Δz
│
└── rotation:     Δrx Δry Δrz
```

Just like $\mathfrak{so}(3)$ had a hat operator turning a 3-vector into a skew-symmetric matrix (section 7), $\mathfrak{se}(3)$ has its own hat operator turning the 6-vector $\xi$ into a $4\times4$ matrix:

$$\xi^\wedge = \begin{bmatrix} \phi^\wedge & \rho\\ 
0 & 0 \end{bmatrix}$$

where $\phi^\wedge$ is that same $3\times3$ skew-symmetric block from before.

Here's the part that's easy to get wrong: **$\exp(\xi^\wedge)$ is *not* "exponentiate the rotation part and copy the translation part over unchanged."** Rotation and translation are coupled - sweeping a small rotation while translating traces a curve, not a straight line. The closed form is:

$$\exp(\xi^\wedge) = \begin{bmatrix} \exp(\phi^\wedge) & V\rho\\ 
0 & 1 \end{bmatrix}$$

where $V$ is a $3\times3$ matrix (built purely from $\phi$) that "bends" the raw translation $\rho$ to account for that coupling. The exact formula for $V$ isn't the point here - what matters is: **you can't just glue the $SO(3)$ exponential and the raw translation together; $SE(3)$'s exponential map genuinely mixes rotation and translation.** (The log map has the mirror-image subtlety: recovering $\rho$ from a pose requires $V^{-1}$, not just reading the translation column off directly.)

Then:

$$T_{\text{new}} = T_{\text{old}}\exp(\xi^\wedge)$$

This is the foundation of many **pose-graph optimization, bundle adjustment, visual-inertial estimation, and SLAM** implementations.

---

## 12. A very intuitive physical analogy

Imagine holding a drone.

Its pose is:

$$T$$

Now someone tells you:

> "Move forward 2 cm, left 1 cm, rotate 0.5° around X, and rotate 0.2° around Z."

That's essentially a **Lie algebra vector**:

$$\xi = \begin{bmatrix} 2\,\text{cm}\\
-1\,\text{cm}\\
0\\
0.5^\circ\\
0\\
0.2^\circ \end{bmatrix}$$

It describes a **small motion**, not a complete pose. (In cm/degrees purely for intuition - plugging into the actual $\exp(\xi^\wedge)$ formula requires consistent units, i.e. meters and radians.)

You then apply that motion to the drone's current pose.

That's the relationship:

$$\boxed{ \text{current pose} + \text{small motion} \rightarrow \text{new pose}}$$

except that the "+" is replaced by the appropriate Lie-group operation.

---

## 13. Why not just use Euler angles?

You might ask:

> "Why not just optimize $x,y,z,\text{roll},\text{pitch},\text{yaw}$?"

You can, but Euler angles have problems:

* singularities / gimbal lock
* awkward composition
* coordinate-dependent behavior
* derivatives can become problematic

Lie algebra gives you a **local minimal representation** of the perturbation while the actual state remains on the correct manifold.

---

## 14. The most important distinction

Keep these four things separate:

| Concept   | Intuition                 |
| --------- | ------------------------- |
| $SO(3)$ | All possible 3D rotations |
| $\mathfrak{so}(3)$ | Small rotational motions  |
| $SE(3)$ | All possible 3D poses     |
| $\mathfrak{se}(3)$ | Small 6-DoF pose motions  |

And:

$$\boxed{SO(3) \underset{\log}{\overset{\exp}{\rightleftarrows}} \mathfrak{so}(3)}$$

$$\boxed{SE(3) \underset{\log}{\overset{\exp}{\rightleftarrows}} \mathfrak{se}(3)}$$

---

## 15. The connection to Jacobians

This is particularly important for the SLAM topics you've been asking about.

When doing optimization, you typically have:

$$\text{error} = f(T)$$

and want to know:

> "If I slightly change the pose, how does the error change?"

So you introduce:

$$\delta\xi \in \mathbb{R}^6$$

and approximate:

$$f(T\exp(\delta\xi^\wedge)) \approx f(T)+J\delta\xi$$

where $J$ is the **Jacobian with respect to the Lie-algebra perturbation**.

That's why Lie algebra appears everywhere in modern SLAM.

It gives optimization algorithms a nice **locally Euclidean 6D space** to work in, while the actual pose remains a valid element of $SE(3)$.

---

## 16. The one-sentence intuition

If you remember only one thing:

> **A Lie group represents the actual robot state (rotation/pose), while its Lie algebra represents small changes around that state, allowing us to use ordinary vector calculus and optimization without breaking the geometry of the state.**

For SLAM, the mental picture I recommend is:

```text
             GLOBAL / ACTUAL STATE
                   Lie Group
                      SE(3)
                        ●
                       / \
                      /   \
                     /     \
                    /       \
             "zoom in locally"
                    ↓
             ┌───────────────┐
             │   Lie Algebra │
             │               │
             │  Δx Δy Δz     │
             │  Δrx Δry Δrz  │
             │               │
             └───────────────┘
                    ξ ∈ R⁶
                       │
                       │ exp()
                       ▼
                new valid pose
```

**Lie algebra is essentially the "local language of motion" for a nonlinear transformation space.**
