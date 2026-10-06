# Factor-graph Optimization

## 1. The core idea

A **factor graph** represents:

> **"What unknowns do we have, and what pieces of evidence tell us about them?"**

It is a bipartite graph with two kinds of nodes (§6 spells out the distinction):

- **Variables** = things we don't know: robot poses, landmarks, sensor biases, etc.
- **Factors** = measurements/constraints: odometry, camera observations, IMU, GPS, loop closures, etc.

![A bipartite factor graph for a SLAM-style problem: robot poses x0..xn connected by odometry-measurement factors u1..un, with landmarks l1, l2 connected to poses via landmark-measurement factors m1..m4, plus a small "variable node/factor node" legend](../images/factor_graph_1.jpg)

![A pose-graph diagram (in Korean) with keyframe poses P0..P6, a black PriorFactor on P0, blue BetweenFactors along the chain, a red BetweenFactor (loop closure) between P0 and P5, and green UnaryFactors on P5](../images/factor_graph_2.jpg)

![Two stacked example factor graphs from a mapping survey: a pose-only graph (top) with pose parameters X linked by odometry factors Tx and a loop-closure factor Tx, and a pose+landmark graph (bottom) that adds observation factors H connecting poses X to landmark parameters L](../images/factor_graph_3.jpg)

![A small factor graph with poses x0, x1, x2, x3 in a chain and landmarks l7, l16, l78, l71, l82 each connected to one or two poses](../images/factor_graph_4.jpg)

![A four-layer factor graph for 2D robot localization: landmarks L0-L2 at top connected via bearing factors to poses P0-P2, which are in turn chained together by odometry factors](../images/factor_graph_5.jpg)

*These diagrams are illustrative sketches from different tutorials/papers; [Image sources](#image-sources) lists which attributions are confirmed and which source is unidentified.*

A text sketch of the same idea:

```text
        landmark l1
             ●
            / \
       camera observations
          /     \
         /       \
Pose x0 ●─────────● x1 ──────● x2
         odometry
```

- The circles are **variables**.
- In this sketch each labelled line stands for a **factor** joining the variables it touches (a binary factor). The images above draw factors as their own nodes (dots or squares), which is how a unary factor such as a prior or GPS fix fits in: it touches just one variable.

---

## 2. Why do we need it?

Suppose the robot moves:

```text
x0 → x1 → x2 → x3
```

The odometry says:

```text
x1 is 1m ahead of x0
x2 is 1m ahead of x1
x3 is 1m ahead of x2
```

But every measurement has error.

Integrating the odometry's own numbers (each step reported as "1m ahead") gives an estimate of $`3.00`$ m. Maybe the real motion, though, was:

```text
x0  →  x1  →  x2  →  x3
  1.02   0.97    1.05 m
```

so the real position is about $`3.04`$ m. Zero-mean per-step errors partly cancel, but nothing corrects them, so on average the drift keeps growing (like $`\sqrt n`$ for $n$ independent steps). Here the estimate (3.00 m) and the real position (3.04 m) diverge.

> **Note**: each per-step measurement ($x_1$ is 1m ahead of $x_0$, etc.) is **odometry**: an estimate of the *incremental* pose change from onboard motion sensors (wheel encoders, IMU, visual odometry, ...).
> - **Dead reckoning** chains ("integrates") these increments to track pose from a starting point, as the `3.00 m` estimate above was computed. With no correction, drift grows the longer we integrate; the loop closure below fixes exactly that.
> - The term is also used for propagating a motion *model* with no measurements (e.g. the `saltation_matrix_ekf.py` baseline in the README): same "no correction" idea, same growing error.

Now imagine that at $x_3$ the camera re-observes a landmark it first saw from $x_0$. Recognizing a previously seen place like this is a **loop closure**, and it gives a direct measurement between $x_0$ and $x_3$ - say, 3.03 m:

```text
x0 ── 1 m ── x1 ── 1 m ── x2 ── 1 m ── x3
│                                      │
└─────────────── 3.03 m ───────────────┘
```

The loop-closure measurement says:

> "$`x_3`$ is 3.03 m from $x_0$ - not the 3.00 m that the chained odometry claims."

Now we have conflicting information.

The optimizer can adjust:

```text
x0, x1, x2, x3
```

so that **all measurements are as consistent as possible**.

That's the essence of factor-graph optimization.

Worked number: the optimizer only sees the measurements, the three 1 m odometry steps and the 3.03 m loop closure (the true 1.02/0.97/1.05 m are unknown to it). Weighting all four equally, least squares lengthens each step by $`0.0075`$ m to 1.0075 m, so $`x_3 = 3.0225`$ m: a compromise between the odometry's 3.00 m and the loop closure's 3.03 m, and closer to the true 3.04 m than odometry alone.

---

## 3. A factor is basically an error function

Suppose we have two poses $`X_i, X_j`$, and odometry gives a measured relative transformation $`Z_{ij}`$. We can calculate what the relative transformation *would be* according to our current estimates:

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
x0 ── odometry ── x1 ── odometry ── x2
 │                                  │
 └──────────── loop closure ────────┘
```

We have three errors, $`e_{01}(x_0,x_1)`$, $`e_{12}(x_1,x_2)`$ and $`e_{02}(x_0,x_2)`$, and the optimizer looks for poses that minimize:

```math
\boxed{
\min_{x_0,x_1,x_2}
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

where $\Omega_k=\Sigma_k^{-1}$ is the **information matrix** (called $W_i$ in [nonlinear_least_square.md §12](nonlinear_least_square.md#12-add-measurement-uncertainty); downstream docs such as [sparse_cholesky_factorization.md](sparse_cholesky_factorization.md) use $\Omega$).

> **Note**: a precise sensor (small $\Sigma_k$) gives a *large* $\Omega_k$, so its error counts more in the sum; a noisy sensor (large $\Sigma_k$) gives a *small* $\Omega_k$ and is down-weighted. The term $e_k^\top\Omega_k e_k$ is a **squared Mahalanobis distance** - see the [glossary](../glossary.md#2-uncertainty-and-probability) for the definition and the isotropic special case where it collapses to plain squared error divided by a constant.

So:

> **Factor-graph optimization = find the variable values that make all measurement factors as happy as possible.**

---

## 5. Why call it a "graph"?

The problem naturally looks like a graph. Below, each labelled line is a factor joining the variables it touches (the images in §1 draw factors as nodes instead):

```text
VARIABLES

x0       x1       x2       x3
●────────●────────●────────●
    F01      F12      F23
```

Adding landmarks:

```text
       l0    l1
       ●     ●
      / \   / \
     /   \ /   \
    ●─────●─────●
    x0    x1    x2
```

Here landmark $l_0$ is seen from $x_0$ and $x_1$, and $l_1$ from $x_1$ and $x_2$.

Adding IMU:

```text
x0          x1          x2
●────IMU────●────IMU────●
 \           \
  camera      camera
   \           \
    ●           ●
    l0          l1
```

Every measurement becomes a **factor connecting the variables it depends on**, so the graph shows the **structure of the estimation problem**.

---

## 6. The really important distinction: variable vs factor

### Variable

Something we're trying to estimate:

- $x_0$ = robot pose
- $x_1$ = robot pose
- $l_0$ = landmark position
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
          x1          l0
       robot pose   landmark
```

The camera measurement doesn't directly "set" $x_1$ or $l_0$. Instead it says:

> "$`x_1`$ and $l_0$ should satisfy this observation."

---

## 7. Factor graph vs pose graph

A **[pose graph](pose_graph_optimization.md)** might look like:

```text
x0 ── x1 ── x2 ── x3
│                 │
└─────────────────┘
```

Usually:

- nodes = robot poses
- edges = relative pose constraints (each edge is a binary factor between two poses)

A **factor graph** is more general:

```text
              l0
               ●
              / \
       camera/   \camera
            /     \
x0 ●──IMU──●──IMU──● x2
          x1
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

$$e(X\oplus\Delta X)\approx e(X)+J\Delta X$$

Then **[Gauss–Newton](gauss_newton.md)** solves (every factor's error stacked into $e$, its $\Omega_i$ blocks into $\Omega$):

```math
J^\top \Omega J\,\Delta X = -J^\top \Omega\, e
```

and updates $`X \leftarrow X \oplus\Delta X`$. For poses, that $\oplus$ is often implemented using **[Lie algebra](../foundations/lie_algebra.md)/$`SE(3)`$**.

Put together, the pieces above connect like this:

```text
Factor graph
     ↓
Nonlinear least squares
     ↓
Gauss–Newton/Levenberg–Marquardt
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

## 9. An analogy: people locating objects

Imagine several people reconstructing the positions of objects in a room. We don't know where anything is, but people give us statements:

> Person A: "B is approximately 2 meters east of me."

> Person B: "C is approximately 1 meter north of me."

> Person C: "The lamp is 3 m west of me."

> GPS: "A is approximately here."

Each statement becomes a **factor**, and together they form a giant network of constraints. Our job is:

> **Move all the unknown positions around until the entire network of statements is as consistent as possible.**

That's factor-graph optimization.

---

## 10. Measurements don't have to be perfect

Suppose:

```text
Odometry says:
x1 should be here ─────────┐
                           │
Camera says:
x1 should be there ────────┤ → compromise
                           │
IMU says:
x1 should be somewhere else┘
```

The optimizer doesn't choose one measurement. It finds the configuration with the **best overall compromise**, weighted by measurement uncertainty (a local optimum, so it relies on a good initial guess).

And when a loop closure arrives:

```text
Before:

x0 ●──●──●──●──●
                \
                 \
                  ● x5


After loop closure:

x0 ●──────────────● x5
   │              │
   └──●──●──●──●──┘
```

the optimizer can distribute the accumulated error across the entire trajectory.

That's why graph optimization is so effective for SLAM.

---

## 11. The one picture to remember

```text
          MEASUREMENTS
       ┌────┬────┬────┐
       ↓    ↓    ↓    ↓

      Factor Factor Factor
        │      │      │
        ↓      ↓      ↓

x0 ●──────●──────●──────● x3
          x1     x2

 ↑                         ↑
 └────── loop closure ─────┘
```

- **Variables** are what we want to know.
- **Factors** are what our sensors tell us.
- **Optimization** finds the variable values that best satisfy all factors simultaneously.

The hierarchy:

> **SLAM** = estimation problem

> **Factor graph** = representation of that estimation problem

> **Factors** = sensor/measurement constraints

> **Nonlinear least squares** = mathematical formulation

> **Gauss–Newton/LM** = optimization method

> **Lie algebra** = convenient way to optimize poses on SE(3)

That is the conceptual bridge connecting the optimization-based SLAM topics in this doc set.

---

## 12. References

1. Kschischang, F. R., Frey, B. J., & Loeliger, H.-A. (2001). *Factor Graphs and the Sum-Product Algorithm*. IEEE Transactions on Information Theory, 47(2), 498–519. https://doi.org/10.1109/18.910572 - the original, general (non-robotics) definition of a factor graph as a bipartite graph of variable nodes and factor nodes, behind §1 and §6's variable/factor terminology. This paper does **not** contain the diagram in `images/factor_graph_1.jpg`; see Image sources below - that image's originally-recorded citation was checked against the paper's actual figures and found to be wrong.
2. Dellaert, F., & Kaess, M. (2017). *Factor Graphs for Robot Perception*. Foundations and Trends in Robotics, 6(1–2), 1–139. https://doi.org/10.1561/2300000043 - the standard robotics-focused reference for factor graphs in SLAM (poses, landmarks, IMU/camera/loop-closure factors), behind §1, §5, §6, and §7's pose-graph-vs-factor-graph distinction.
3. Racinskis, P., Arents, J., & Greitans, M. (2023). *Constructing Maps for Autonomous Robotics: An Introductory Conceptual Overview*. Electronics, 12(13), 2925. https://doi.org/10.3390/electronics12132925 - Figure 1 of this paper is the confirmed source of the diagram in `images/factor_graph_3.jpg` (see Image sources below).

### Image sources

1. `images/factor_graph_1.jpg` - **source unidentified.** It was originally cited as IEEE document 910572 (Reference 1 above), but that citation is incorrect: none of the paper's figures (abstract coding-theory factor graphs, Tanner graphs, trellises, a scalar Kalman-filter derivation) show robot poses, landmarks or this legend. Dellaert & Kaess's SLAM tutorials use similar "Odometry measurement"/"Landmark measurement" wording but with photographs, so they did not match either. Do not cite IEEE 910572 for this image.
2. `images/factor_graph_2.jpg` - originally cited as https://engcang.github.io/gtsam_tutorial.html, match for the second pose-graph figure on that page (image file `/assets/img/posts/230715_gtsam/graph2.png`), a Korean-language GTSAM tutorial blog post by Eungchang Mason Lee (page title "GTSAM 튜토리얼 | Eungchang Mason Lee").
3. `images/factor_graph_3.jpg` - originally cited as https://www.mdpi.com/2079-9292/12/13/2925, match for Figure 1 of Reference 3 above (downloaded directly from MDPI's own PDF host, since the MDPI article page itself returns HTTP 403 to automated fetches).
4. `images/factor_graph_4.jpg` - originally cited as https://cmsc426.github.io/gtsam/, match for the image `/assets/sfm/gtsam9.png` embedded on that page, part of the University of Maryland CMSC426 (Computer Vision) course's "Structure from Motion" lecture notes.
5. `images/factor_graph_5.jpg` - originally cited as https://symforce.org/, match for the image `docs/static/images/robot_2d_localization/factor_graph.png` embedded on that page - the diagram from SymForce's (Skydio's symbolic-computation library for robotics) "Robot 2D Localization" example/tutorial, https://symforce.org/examples/robot_2d_localization/README.html.
