# Factor-graph Optimization

## 1. The core idea

Think of a **factor graph** as a way to represent:

> **"What unknowns do I have, and what pieces of evidence tell me about those unknowns?"**

For SLAM:

- **Variables** = things we don't know
  → robot poses, landmarks, sensor biases, etc.
- **Factors** = measurements/constraints
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
Pose X0 ●─────────● X1 ──────● X2
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

If you simply integrate the odometry's own numbers above (each step reported as "1m ahead"):

$$\text{estimated position} \approx 3.00\ \text{m}$$

Maybe the real motion, though, was:

```text
X0  →  X1  →  X2  →  X3
  1.02   0.97    1.05 m
```

$$\text{real position} \approx 3.04\ \text{m}$$

Integrating noisy per-step measurements never magically cancels their errors, so the estimate (3.00 m) and the real position (3.04 m) diverge - you accumulate error.

> **Note**: each of those per-step measurements (`X1 is 1m ahead of X0`, etc.) is what **odometry** actually provides - an estimate of the robot's *incremental* change in pose between two nearby moments, from onboard motion sensors (wheel encoders, IMU, visual odometry, ...). Chaining ("integrating") a sequence of these incremental measurements to track pose relative to a starting point, the way the `3.00 m` estimate above was computed, is called **dead reckoning**. Since every measurement carries a small error and dead reckoning sums them with no correction, the drift grows unboundedly the longer you integrate - exactly the problem the loop closure below fixes. The term is also used more loosely for propagating a known motion *model* forward from an initial guess without looking at any measurements (e.g. the `saltation_matrix_ekf.py` baseline in the README) - the same "no correction" idea, with the same growing error.

Now imagine that at X3 the camera re-observes a landmark it first saw from X0. Recognizing a previously seen place like this is a **loop closure**, and it gives a direct measurement between X0 and X3 - say, 3.03 m:

```text
X0 ── 1 m ── X1 ── 1 m ── X2 ── 1 m ── X3
│                                      │
└─────────────── 3.03 m ───────────────┘
```

The loop-closure measurement says:

> "X3 is 3.03 m from X0 - not the 3.00 m that the chained odometry claims."

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

Then compare them. $Z_{ij}^{-1}\hat Z_{ij}$ is itself a transform, equal to the identity exactly when the estimate agrees with the measurement. The log map turns it into a 6-vector (a translation part and a rotation part) that optimization can square and sum:

```math
e_{ij} = \mathrm{Log}\left(Z_{ij}^{-1}\hat Z_{ij}\right) = \mathrm{Log}\left(Z_{ij}^{-1} X_i^{-1} X_j\right) \in \mathbb{R}^6
```

This is the residual `pose_graph.py` uses ([pose_graph_optimization.md §15.7](pose_graph_optimization.md#157-the-solver-concretely)); [lie_algebra.md](../foundations/lie_algebra.md) explains Log.

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

```math
\boxed{
\min_{X_0,X_1,X_2}
\sum_k \|e_k\|^2
}
```

More realistically, each measurement has different reliability:

$$
\boxed{
\min_X
\sum_k
e_k^\top\Omega_k e_k
}
$$

where $\Omega_k$ is the **information matrix** - the same quantity [nonlinear_least_square.md §12](nonlinear_least_square.md#12-add-measurement-uncertainty) calls $W_i$; both are $\Sigma^{-1}$ for a factor's measurement; this doc's $\Omega$ notation is what everything built on top of it (including [sparse_cholesky_factorization.md](sparse_cholesky_factorization.md)) uses from here on.

> **Note**: $\Omega_k$ is the inverse of that measurement's covariance, $\Omega_k = \Sigma_k^{-1}$ - a precise sensor (small $\Sigma_k$) inverts to a *large* $\Omega_k$, so its error counts more in the sum, while a noisy sensor (large $\Sigma_k$) inverts to a *small* $\Omega_k$ and gets down-weighted. The term $e_k^\top\Omega_k e_k$ is a **Mahalanobis distance** - see the [glossary](../glossary.md#2-uncertainty-and-probability) for the definition and the isotropic special case where it collapses to plain squared error divided by a constant.

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
       L0    L1
       ●     ●
      / \   / \
     /   \ /   \
    ●─────●─────●
    X0    X1    X2
```

Here landmark L0 is seen from X0 and X1, and L1 from X1 and X2.

Adding IMU:

```text
X0          X1          X2
●────IMU────●────IMU────●
 \           \
  camera      camera
   \           \
    ●           ●
    L0          L1
```

Every measurement becomes a **factor connecting the variables it depends on**.

This is extremely powerful because the graph tells you the **structure of the estimation problem**.

---

## 6. The really important distinction: variable vs factor

This is worth memorizing.

### Variable

Something you're trying to estimate:

- $X_0$ = robot pose
- $X_1$ = robot pose
- $L_0$ = landmark position
- $b$ = IMU bias

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
│                 │
└─────────────────┘
```

Usually:

- nodes = robot poses
- edges = relative pose constraints

A **factor graph** is more general:

```text
              L0
               ●
              / \
       camera/   \camera
            /     \
X0 ●──IMU──●──IMU──● X2
          X1
```

IMU factors link consecutive poses, and each camera factor links a pose to the landmark it observes. A pose graph would keep only the poses and the relative-pose factors between them.

It can represent:

- poses
- landmarks
- velocity
- IMU biases
- calibration parameters
- time offsets
- sensor extrinsics
- etc.

So:

> **Pose graph is essentially a special/simple case of a factor graph.**

---

## 8. Where Gauss–Newton enters

Factor-graph optimization is usually a **nonlinear least-squares problem**.

We have:

```math
\min_X \sum_i e_i(X)^\top \Omega_i\, e_i(X)
```

But the errors are nonlinear because poses involve rotations and transformations.

So we linearize:

$$e(X+\Delta X)\approx e(X)+J\Delta X$$

Then **[Gauss–Newton](gauss_newton.md)** solves:

```math
J^\top \Omega J\,\Delta X = -J^\top \Omega\, e
```

with every factor's error stacked into $e$, and its $\Omega_i$ blocks into $\Omega$.

and updates:

$$X \leftarrow X \oplus\Delta X$$

For poses, that $\oplus$ is often implemented using **[Lie algebra](../foundations/lie_algebra.md) / SE(3)**.

Put together, the pieces above connect like this:

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
   │              │
   └──●──●──●──●──┘
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

That is the conceptual bridge connecting essentially all the SLAM topics covered in this doc set.

---

## 12. References

1. Kschischang, F. R., Frey, B. J., & Loeliger, H.-A. (2001). *Factor Graphs and the Sum-Product Algorithm*. IEEE Transactions on Information Theory, 47(2), 498–519. https://doi.org/10.1109/18.910572 - the original, general (non-robotics) definition of a factor graph as a bipartite graph of variable nodes and factor nodes, behind §1 and §6's variable/factor terminology. This paper does **not** contain the diagram in `images/factor_graph_1.jpg`; see Image sources below - that image's originally-recorded citation was checked against the paper's actual figures and found to be wrong.
2. Dellaert, F., & Kaess, M. (2017). *Factor Graphs for Robot Perception*. Foundations and Trends in Robotics, 6(1–2), 1–139. https://doi.org/10.1561/2300000043 - the standard robotics-focused reference for factor graphs in SLAM (poses, landmarks, IMU/camera/loop-closure factors), behind §1, §5, §6, and §7's pose-graph-vs-factor-graph distinction.
3. Racinskis, P., Arents, J., & Greitans, M. (2023). *Constructing Maps for Autonomous Robotics: An Introductory Conceptual Overview*. Electronics, 12(13), 2925. https://doi.org/10.3390/electronics12132925 - Figure 1 of this paper is the confirmed source of the diagram in `images/factor_graph_3.jpg` (see Image sources below).

### Image sources

<!-- 1. `images/factor_graph_1.jpg` - originally cited as https://ieeexplore.ieee.org/document/910572 (IEEE document 910572, i.e., Reference 1 above). **This citation is incorrect.** The paper was downloaded in full and every figure inspected; none of them show robot poses, landmarks, "Odometry measurement"/"Landmark measurement" labels, or the "Bipartite graph with variable nodes and factor nodes" legend seen in this image - the paper's figures are all abstract coding-theory examples ($x_1,\dots,x_5$ with generic factors $f_A,\dots,f_E$), Tanner graphs, trellises, and a scalar Kalman-filter derivation. A plausible alternative family of sources (Dellaert & Kaess's SLAM tutorials, which use this exact "Odometry measurement" / "Landmark measurement" phrasing with toy robot/furniture photos) was checked and did not match either - their version uses photographs, not the abstract $x_0,\dots,x_n$ / $l_1, l_2$ circles seen here. The true source of this image is **unidentified**; do not cite IEEE document 910572 for it. -->
2. `images/factor_graph_2.jpg` - originally cited as https://engcang.github.io/gtsam_tutorial.html, match for the second pose-graph figure on that page (image file `/assets/img/posts/230715_gtsam/graph2.png`), a Korean-language GTSAM tutorial blog post by Eungchang Mason Lee (page title "GTSAM 튜토리얼 | Eungchang Mason Lee").
3. `images/factor_graph_3.jpg` - originally cited as https://www.mdpi.com/2079-9292/12/13/2925, match for Figure 1 of Reference 3 above (downloaded directly from MDPI's own PDF host, since the MDPI article page itself returns HTTP 403 to automated fetches).
4. `images/factor_graph_4.jpg` - originally cited as https://cmsc426.github.io/gtsam/, match for the image `/assets/sfm/gtsam9.png` embedded on that page, part of the University of Maryland CMSC426 (Computer Vision) course's "Structure from Motion" lecture notes.
5. `images/factor_graph_5.jpg` - originally cited as https://symforce.org/, match for the image `docs/static/images/robot_2d_localization/factor_graph.png` embedded on that page - the diagram from SymForce's (Skydio's symbolic-computation library for robotics) "Robot 2D Localization" example/tutorial, https://symforce.org/examples/robot_2d_localization/README.html.
