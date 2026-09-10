# Factor-graph Optimization

## 1. The core idea

Think of a **factor graph** as a way to represent:

> **“What unknowns do I have, and what pieces of evidence tell me about those unknowns?”**

For SLAM:

* **Variables** = things we don't know
  → robot poses, landmarks, sensor biases, etc.
* **Factors** = measurements/constraints
  → odometry, camera observations, IMU measurements, GPS, loop closures, etc.

![A bipartite factor graph for a SLAM-style problem: robot poses x0..xn connected by odometry-measurement factors u1..un, with landmarks l1, l2 connected to poses via landmark-measurement factors m1..m4, plus a small "variable node / factor node" legend](../images/factor_graph_1.jpg)

![A pose-graph diagram (in Korean) with keyframe poses P0..P6, a black PriorFactor on P0, blue BetweenFactors along the chain, a red BetweenFactor (loop closure) between P0 and P5, and green UnaryFactors on P5](../images/factor_graph_2.jpg)

![Two stacked example factor graphs from a mapping survey: a pose-only graph (top) with pose parameters X linked by odometry factors Tx and a loop-closure factor Tx, and a pose+landmark graph (bottom) that adds observation factors H connecting poses X to landmark parameters L](../images/factor_graph_3.jpg)

![A small factor graph with poses x0, x1, x2, x3 in a chain and landmarks l7, l16, l78, l71, l82 each connected to one or two poses](../images/factor_graph_4.jpg)

![A four-layer factor graph for 2D robot localization: landmarks L0-L2 at top connected via bearing factors to poses P0-P2, which are in turn chained together by odometry factors](../images/factor_graph_5.jpg)

*These diagrams are illustrative sketches pulled from different tutorials/papers - see [Image sources](#image-sources) below for exactly which claim is confirmed vs. still unverified.*

For example:

```text
        landmark L1
           ●
          / \
       camera observations
        /     \
       /       \
Pose X0 ●──────● X1 ──────● X2
          odometry
```

The circles are **variables**.

The connections are **factors**.

---

## 2. Why do we need it?

Suppose your robot moves:

```text
X0 → X1 → X2 → X3
```

The odometry says:

```text
X1 is 1m ahead of X0
X2 is 1m ahead of X1
X3 is 1m ahead of X2
```

But every measurement has error.

Maybe the real motion was:

```text
X0  →  X1  →  X2  →  X3
  1.02   0.97    1.05 m
```

If you simply integrate these measurements:

```text
estimated position ≈ 3.04 m
```

you accumulate error.

Now imagine that at X3 the camera recognizes a place it saw at X0.

That's a **loop closure**:

```text
X0 ●────────────────● X3
    \              /
     \            /
      ●────●────●
      X1   X2
```

The loop-closure measurement says:

> "Hey, X3 should actually be close to X0."

Now we have conflicting information.

The optimizer can adjust:

```text
X0, X1, X2, X3
```

so that **all measurements are as consistent as possible**.

That's the essence of factor-graph optimization.

---

## 3. A factor is basically an error function

This is the most important mathematical intuition.

Suppose we have two poses:

$$
X_i,\ X_j
$$

and odometry gives us a measured relative transformation:

$$
Z_{ij}
$$

We can calculate what the relative transformation *would be* according to our current estimates:

$$
\hat Z_{ij}=X_i^{-1}X_j
$$

Then compare:

$$\text{error}=Z_{ij}^{-1}\hat Z_{ij}$$

Conceptually:

```text
              measurement
                  ↓
Xi ─────────── Factor ─────────── Xj
                  ↓
             "How wrong
              are Xi,Xj?"
```

The factor answers:

> **Given my current estimates of the variables, how badly do they violate this measurement?**

---

## 4. Optimization means minimizing all those errors

Suppose we have:

```text
X0 ── odometry ── X1 ── odometry ── X2
 │                                  │
 └──────────── loop closure ────────┘
```

We have three errors:

$$
e_{01}(X_0,X_1)
$$

$$
e_{12}(X_1,X_2)
$$

$$
e_{02}(X_0,X_2)
$$

The optimizer tries to find poses that minimize:

$$
\boxed{
\min_{X_0,X_1,X_2}
\sum_k \|e_k\|^2
}
$$

More realistically, each measurement has different reliability:

$$
\boxed{
\min_X
\sum_k
e_k^T\Omega_k e_k
}
$$

where $\Omega_k$ is the **information matrix**.

So:

> **Factor-graph optimization = find the variable values that make all measurement factors as happy as possible.**

---

## 5. Why call it a "graph"?

Because the problem naturally looks like a graph.

For example:

```text
VARIABLES

X0       X1       X2       X3
●────────●────────●────────●
    F01      F12      F23
```

Adding landmarks:

```text
           L0
           ●
          / \
         /   \
X0 ●────●─────● X1
     \        /
      \      /
       ●────●
       L1   X2
```

Adding IMU:

```text
X0 ●────────● X1────────● X2
    \ IMU     \ IMU
     \          \
      camera     camera
       \          \
        L0         L1
```

Every measurement becomes a **factor connecting the variables it depends on**.

This is extremely powerful because the graph tells you the **structure of the estimation problem**.

---

## 6. The really important distinction: variable vs factor

This is worth memorizing.

### Variable

Something you're trying to estimate:

```text
X0 = robot pose
X1 = robot pose
L0 = landmark position
b = IMU bias
```

### Factor

Something that constrains those variables:

```text
Odometry factor
IMU factor
Camera reprojection factor
GPS factor
Loop-closure factor
Prior factor
```

For example:

```text
        camera measurement
                ↓
              Factor
             /      \
            /        \
          X1          L0
       robot pose   landmark
```

The camera measurement doesn't directly "set" X1 or L0.

Instead it says:

> "X1 and L0 should satisfy this observation."

---

## 7. Factor graph vs pose graph

This distinction is particularly important for SLAM.

A **[pose graph](pose_graph_optimization.md)** might look like:

```text
X0 ── X1 ── X2 ── X3
│                │
└────────────────┘
```

Usually:

* nodes = robot poses
* edges = relative pose constraints

A **factor graph** is more general:

```text
          L0
          ●
         / \
        /   \
X0 ●───●─────● X1
  │             │
  │             │
 IMU           Camera
  │             │
X2 ●───────────●
```

It can represent:

* poses
* landmarks
* velocity
* IMU biases
* calibration parameters
* time offsets
* sensor extrinsics
* etc.

So:

> **Pose graph is essentially a special/simple case of a factor graph.**

---

## 8. Where Gauss–Newton enters

This connects directly to what you asked about earlier.

Factor-graph optimization is usually a **nonlinear least-squares problem**.

We have:

$$\min_X \sum_i \|e_i(X)\|^2$$

But the errors are nonlinear because poses involve rotations and transformations.

So we linearize:

$$e(X+\Delta X)\approx e(X)+J\Delta X$$

Then **[Gauss–Newton](gauss_newton.md)** solves:

$$J^T W J\Delta X =-J^T W e$$

and updates:

$$X \leftarrow X \oplus\Delta X$$

For poses, that $\oplus$ is often implemented using **[Lie algebra](../foundations/lie_algebra.md) / SE(3)**.

So you can mentally connect everything you've been studying:

```text
Factor graph
     ↓
Nonlinear least squares
     ↓
Gauss–Newton / Levenberg–Marquardt
     ↓
Linearization
     ↓
Solve linear system
     ↓
Update poses/landmarks
     ↓
Repeat
```

---

## 9. A very intuitive analogy

Imagine several people trying to reconstruct the position of objects in a room.

You don't know where anything is.

But people give you statements:

> Person A: "B is approximately 2 meters east of me."

> Person B: "C is approximately 1 meter north of me."

> Person C: "I can see the same object that A sees."

> GPS: "A is approximately here."

Each statement becomes a **factor**.

You now have a giant network of constraints.

Your job is:

> **Move all the unknown positions around until the entire network of statements is as consistent as possible.**

That's factor-graph optimization.

---

## 10. The key SLAM insight

The beautiful thing about this approach is that **measurements don't have to be perfect**.

Suppose:

```text
Odometry says:
X1 should be here ─────────┐
                           │
Camera says:
X1 should be there ────────┤ → compromise
                           │
IMU says:
X1 should be somewhere else┘
```

The optimizer doesn't necessarily choose one measurement.

It finds the configuration that provides the **best global compromise**, weighted by measurement uncertainty.

And when a loop closure arrives:

```text
Before:

X0 ●──●──●──●──●
                 \
                  \
                   ● X5


After loop closure:

X0 ●──────────────● X5
 │                │
 └──●──●──●──●────┘
```

the optimizer can distribute the accumulated error across the entire trajectory.

That's why graph optimization is so effective for SLAM.

---

## 11. One mental model to remember

If you remember only one picture, remember this:

```text
          MEASUREMENTS
       ┌────┬────┬────┐
       ↓    ↓    ↓    ↓

      Factor Factor Factor
        │      │      │
        ↓      ↓      ↓

X0 ●──────●──────●──────● X3
    X1     X2

 ↑                         ↑
 └────── loop closure ─────┘
```

**Variables are what you want to know.**

**Factors are what your sensors tell you.**

**Optimization finds the variable values that best satisfy all factors simultaneously.**

And this gives you a very useful hierarchy:

> **SLAM** = estimation problem

> **Factor graph** = representation of that estimation problem

> **Factors** = sensor/measurement constraints

> **Nonlinear least squares** = mathematical formulation

> **Gauss–Newton / LM** = optimization method

> **Lie algebra** = convenient way to optimize poses on SE(3)

That is the conceptual bridge connecting essentially all the SLAM topics you've been asking about recently.

---

## 12. References

1. Kschischang, F. R., Frey, B. J., & Loeliger, H.-A. (2001). *Factor Graphs and the Sum-Product Algorithm*. IEEE Transactions on Information Theory, 47(2), 498–519. https://doi.org/10.1109/18.910572 - the original, general (non-robotics) definition of a factor graph as a bipartite graph of variable nodes and factor nodes, behind §1 and §6's variable/factor terminology. This paper does **not** contain the diagram in `images/factor_graph_1.jpg`; see Image sources below - that image's originally-recorded citation was checked against the paper's actual figures and found to be wrong.
2. Dellaert, F., & Kaess, M. (2017). *Factor Graphs for Robot Perception*. Foundations and Trends in Robotics, 6(1–2), 1–139. https://doi.org/10.1561/2300000043 - the standard robotics-focused reference for factor graphs in SLAM (poses, landmarks, IMU/camera/loop- closure factors), behind §1, §5, §6, and §7's pose-graph-vs-factor-graph distinction.
3. Racinskis, P., Arents, J., & Greitans, M. (2023). *Constructing Maps for Autonomous Robotics: An Introductory Conceptual Overview*. Electronics, 12(13), 2925. https://doi.org/10.3390/electronics12132925 - Figure 1 of this paper is the confirmed source of the diagram in `images/factor_graph_3.jpg` (see Image sources below).

### Image sources

<!-- 1. `images/factor_graph_1.jpg` - originally cited as https://ieeexplore.ieee.org/document/910572 (IEEE document 910572, i.e., Reference 1 above). **This citation is incorrect.** The paper was downloaded in full and every figure inspected; none of them show robot poses, landmarks, "Odometry measurement"/"Landmark measurement" labels, or the "Bipartite graph with variable nodes and factor nodes" legend seen in this image - the paper's figures are all abstract coding-theory examples ($x_1,\dots,x_5$ with generic factors $f_A,\dots,f_E$), Tanner graphs, trellises, and a scalar Kalman-filter derivation. A plausible alternative family of sources (Dellaert & Kaess's SLAM tutorials, which use this exact "Odometry measurement" / "Landmark measurement" phrasing with toy robot/furniture photos) was checked and did not match either - their version uses photographs, not the abstract $x_0,\dots,x_n$ / $l_1, l_2$ circles seen here. The true source of this image is **unidentified**; do not cite IEEE document 910572 for it. -->
2. `images/factor_graph_2.jpg` - originally cited as https://engcang.github.io/gtsam_tutorial.html. **Confirmed**: this is a pixel-for-pixel match for the second pose-graph figure on that page (image file `/assets/img/posts/230715_gtsam/graph2.png`), a Korean-language GTSAM tutorial blog post by Eungchang Mason Lee (page title "GTSAM 튜토리얼 | Eungchang Mason Lee").
3. `images/factor_graph_3.jpg` - originally cited as https://www.mdpi.com/2079-9292/12/13/2925. **Confirmed**: pixel-for-pixel match for Figure 1 of Reference 3 above (downloaded directly from MDPI's own PDF host, since the MDPI article page itself returns HTTP 403 to automated fetches).
4. `images/factor_graph_4.jpg` - originally cited as https://cmsc426.github.io/gtsam/. **Confirmed**: pixel-for-pixel match for the image `/assets/sfm/gtsam9.png` embedded on that page, part of the University of Maryland CMSC426 (Computer Vision) course's "Structure from Motion" lecture notes.
5. `images/factor_graph_5.jpg` - originally cited as https://symforce.org/. **Confirmed**: pixel-for-pixel match for the image `docs/static/images/robot_2d_localization/factor_graph.png` embedded on that page - the diagram from SymForce's (Skydio's symbolic-computation library for robotics) "Robot 2D Localization" example/tutorial, https://symforce.org/examples/robot_2d_localization/README.html.
