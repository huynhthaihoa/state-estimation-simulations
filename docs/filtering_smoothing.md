# Filtering vs. Optimization/Smoothing in SLAM

The key to **Filtering vs. Optimization/Smoothing in SLAM** is **when the robot is allowed to change its mind about the past**.

---

## 1. The core idea

Suppose a robot moves through a room:

**$t_0 \to t_1 \to t_2 \to t_3 \to t_4$**

At every time step, it gets:

- IMU measurements
- camera/LiDAR observations
- wheel odometry, etc.

The robot wants to estimate:

> **Where am I? What does the environment look like?**

The fundamental difference is:

### Filtering

> **Estimate the current state using everything available up to now.**

### Optimization/Smoothing

> **Estimate a whole trajectory by jointly considering many states and measurements, including information that may arrive later.**

Two one-line mental models:

> **Filtering = "What do I believe right now?"**

> **Smoothing = "Given everything I've seen, what do I now believe my whole trajectory was?"**

---

## 2. Filtering: one step at a time

Imagine the robot is at time $t_3$.

A filter maintains something like:

```text
        past measurements
              ↓
z₀ → z₁ → z₂ → z₃
              ↓
        [ FILTER ]
              ↓
        current state x₃
```

It summarizes the past into the **current belief**.

For a Kalman filter, conceptually:

```text
prediction
    ↓
x₃ predicted
    ↓
new measurement z₃
    ↓
correction
    ↓
x₃ estimated
```

Then the robot moves to $t_4$.

The filter takes the estimate at $t_3$, propagates it forward, incorporates $z_4$, and produces the estimate at $t_4$.

### What the filter forgets

Once the filter has moved on, it generally **doesn't go back and reconsider old states**. It essentially says:

> "I have compressed everything before $t_3$ into my current belief. Now let's continue."

This makes filtering **naturally online and computationally efficient**.

### The big limitation of filtering

Suppose the robot sees:

```text
A → B → C → D → E
```

At the beginning, it thinks:

```text
A ---- B ---- C ---- D
```

But later, at `E`, it recognizes:

> "Wait! I've seen this place before. This is actually A."

That's a **loop closure**.

The previous trajectory must have been wrong: the new information says the robot ended up back near its starting position, not at a separate point `E`.

Therefore, **all those previous poses may need to move**:

```text
Before:

A ---- B ---- C ---- D
                      \
                       E


After:

       B --- C --- D
      /             \
     A ------------- E
```

A pure filtering mindset is uncomfortable with this because it has already compressed the past: the old poses are no longer in its state, so it cannot move them. (An EKF-SLAM filter still benefits from the loop closure. Through the correlations in its covariance, it corrects the current pose and the whole landmark map; see [§9](#9-the-precise-distinction).)

---

## 3. Optimization: keep the history

Optimization-based SLAM takes a different approach.

Instead of maintaining only:

```text
current state xₜ
```

it maintains many states:

```text
x₀   x₁   x₂   x₃   x₄
 |    |    |    |    |
```

and measurements connecting them:

```text
x₀ ── x₁ ── x₂ ── x₃ ── x₄
   measurements between each consecutive pair
```

The SLAM problem becomes:

> **Find the set of poses and landmarks that best explains all the measurements (a local optimum, in general).**

Mathematically, we write:

```math
{\mathbf{x}^*=\arg\min_{\mathbf{x}}\sum_i \|r_i(\mathbf{x})\|^2}
```

where:

- $\mathbf{x}$ = all poses/landmarks
- $r_i(\mathbf{x})$ = (whitened) measurement residual, i.e. weighted by the measurement covariance
- the solver (Gauss-Newton or a damped variant such as Levenberg-Marquardt) finds a local minimum of the total error, so a good initial guess matters.

This is the fundamental idea behind **bundle adjustment**, **pose-graph optimization**, and many modern SLAM systems.

### Revising the past after a loop closure

Suppose we have:

```text
x₀ → x₁ → x₂ → x₃ → x₄
```

with odometry constraints:

```text
x₀ ── x₁ ── x₂ ── x₃ ── x₄
```

But then we detect a loop:

```text
x₀ ── x₁ ── x₂ ── x₃ ── x₄
│                        │
└────────────────────────┘
          loop closure
```

Optimization says:

> "Let's adjust **all of these poses together** so that all constraints are satisfied as well as possible."

So the error can be distributed:

```text
Before:

x₀ ── x₁ ── x₂ ── x₃ ── x₄
│                        │
└────────────────────────┘

After optimization:

      x₁ --- x₂ --- x₃
     /               \
    x₀ -------------- x₄
```

The important point is:

**Optimization can revise the past.**

---

## 4. Smoothing: optimization with a probabilistic interpretation

**Smoothing** is closely related to optimization, but conceptually it is useful to distinguish them.

Filtering asks:

$${p(x_t \mid z_{0:t})}$$

> "What is my belief about the current state given measurements up to now?"

Smoothing asks something closer to:

$${p(x_{0:t} \mid z_{0:t})}$$

> "Given all measurements, what is my belief about the entire trajectory?"

So:

```text
FILTERING

z₀ z₁ z₂ z₃
     ↓
   x₃ only
```

versus:

```text
SMOOTHING

z₀ z₁ z₂ z₃
 ↓  ↓  ↓  ↓
x₀ x₁ x₂ x₃
 \  |  |  /
  joint estimate
```

Optimization is often used as the computational mechanism for obtaining this joint estimate. Strictly, least-squares optimization returns the single most likely trajectory (the MAP estimate, i.e. the peak of $`p(x_{0:t} \mid z_{0:t})`$), not the whole distribution; the uncertainty around it is usually approximated as a Gaussian whose information matrix is the Gauss-Newton Hessian $`J^\top \Sigma^{-1} J`$ at the solution (with whitened Jacobians, simply $`J^\top J`$).

**Tiny example** (toy case: one scalar $x$, measurement $r = x - z$ with $\sigma = 0.5$):
- Information $= 1/\sigma^2 = 4$, so the variance is $1/4 = 0.25$.
- A second identical measurement adds its information: $4 + 4 = 8$, and the variance halves to $0.125$.

---

## 5. An analogy

We reconstruct a person's route through a city.

- **Filtering:** at each intersection we take the previous answer, add the new clue (a street sign, a step count), and update "I'm probably here". We carry that belief forward and never reopen old answers.
- **Smoothing:** at the end of the day we collect all the GPS fixes, photos, timestamps and landmarks and ask, **"Given everything we know now, where were you at every point during the day?"** The person realizes the landmark seen at 2 PM is the one seen at 10 AM, so the 11 AM, 12 PM and 1 PM positions get revised too.

---

## 6. The computational difference

This gives us a useful trade-off:

|                    | Filtering                      | Optimization/Smoothing                   |
| ------------------ | ------------------------------ | ------------------------------------------ |
| Main question      | Where am I **now**?            | Where was I **throughout the trajectory**? |
| State maintained   | Current belief                 | Many historical states                     |
| Uses past          | Compressed into current belief | Explicitly retained                        |
| Can revise past?   | Limited                        | Yes                                        |
| Loop closure       | Harder (no past poses to move; EKF-SLAM corrects the map via correlations) | Natural |
| Computation        | Usually cheaper                | Usually more expensive                     |
| Memory             | Lower                          | Higher                                     |
| Online operation   | Excellent                      | Possible, but needs management             |
| Global consistency | Harder                         | Stronger                                   |
| Typical idea       | EKF-SLAM                       | Pose graph/factor graph/BA             |

**A caveat on cost**: "filtering is cheaper" holds for small, fixed-size problems, but it inverts at scale.
- **EKF-SLAM** keeps a *dense* joint covariance, so every update costs roughly $O(n^2)$ in the number of landmarks (Dissanayake et al., 2001).
- **Sparse factor-graph smoothing** exploits the sparsity of the graph instead. With an incremental solver such as iSAM2 (Kaess et al., 2012), an odometry measurement only re-eliminates a small part of the Bayes tree near the newest pose.
- **There is no guaranteed bound**: a loop closure can force re-elimination of most or all of the tree, and the cost depends on the fill-in of the sparse factorization. This repo's [`bayes_tree_construction.py`](../use_numpy/bayes_tree_construction.py) shows both cases on a 16-node square loop. Under its fixed oldest-first ordering, an odometry edge affects 2 of 16 variables and a loop-closure edge affects all 16; iSAM2 additionally reorders the affected part, so its real cost is lower (see [`bayes_tree.md` §15](optimization/bayes_tree.md#15-where-this-is-implemented-in-this-repo)).

The defensible claim is therefore that *most* updates in a typical SLAM graph (mostly odometry, occasionally a loop closure) are cheap under iSAM2, while EKF-SLAM pays $O(n^2)$ on every update. That is why large-scale SLAM moved from EKF-SLAM toward factor-graph smoothing.

---

## 7. This is where factor graphs fit

A **factor graph** is the bridge between the two worlds. It is bipartite: variables (poses, landmarks) are one kind of node, and each **factor** is its own node connected to the variables it constrains. A SLAM factor graph typically contains **three kinds of factors** (■ = factor node, ● = variable node):

- A **landmark factor** ties a pose to a landmark it observed (two such factors, from $x_0$ and $x_2$ to the same landmark):

```text
x₀ ● ──■── ● landmark ──■── ● x₂
```

- A **pose-to-pose factor** ties two consecutive (or, for a loop closure, non-consecutive) poses together via a relative measurement:

```text
x₀ ● ──■── ● x₁ ──■── ● x₂ ──■── ● x₃
```

- A **unary (prior) factor** constrains a single variable directly. It removes gauge freedom (the whole solution could otherwise shift and rotate together, and in monocular BA also rescale) or injects an absolute measurement like GPS:

```text
■ ── ● x₀ ──■── ● x₁ ──■── ● x₂
prior
```

In this repo the gauge is fixed as follows:

| Script | How the gauge is fixed |
| --- | --- |
| [`pose_graph.py`](../use_numpy/pose_graph.py) | strong prior on node 0's block of the information matrix |
| [`bundle_adjustment.py`](../use_numpy/bundle_adjustment.py) | soft prior factors on the first two cameras (also pins the monocular scale) |
| [`bundle_adjustment_advanced.py`](../use_numpy/bundle_adjustment_advanced.py) | no prior factor; two keyframes held constant |

Each measurement becomes a factor node imposing a constraint:

```text
IMU:             x₁ ●──■──● x₂
Visual odometry: x₂ ●──■──● x₃
Loop closure:    x₃ ●──■──● x₀
```

Optimization then asks:

> **"What configuration of $x_0$, $x_1$, $x_2$, $x_3$ best satisfies all these constraints?"**

---

## 8. Filtering vs smoothing in one picture

```text
FILTERING                                   SMOOTHING / OPTIMIZATION

                ┌──────────┐                measurements
measurements ──►│  FILTER  │──► state             │
                └──────────┘                      ▼
                     │                 x₀ ─── x₁ ─── x₂ ─── x₃
                     ▼                  \                   /
                summarize past           └──loop closure────┘
                                                  │
                                                  ▼
                                         JOINT OPTIMIZATION
                                                  │
                                                  ▼
                                         optimized trajectory
```

- The filter's past gets **compressed**.
- The smoother's past is **kept around so it can be reconsidered**.

---

## 9. The precise distinction

We are tempted to say:

> "Filtering is local, optimization is global."

That is **useful intuition, but not strictly correct**.

A filter can incorporate loop closures and other global information. For example, EKF-SLAM can update the entire state covariance/mean when a landmark is re-observed.

The more precise distinction is:

> **Filtering recursively represents the current posterior and marginalizes old information, whereas smoothing maintains a posterior over multiple states and can jointly revise them.**

**Tiny example (toy case, 1D):**
- Prior: $`x_0\sim N(0,1)`$, and $`x_1 = x_0 + 1`$ with extra noise of variance 1. So $`x_1`$ has mean 1 and variance 2.
- Later we learn $`x_0 = 0.4`$ with variance 0.01 (a sharp reading).
- Smoother (still holds $`x_0`$): $`x_0 = 40/101 \approx 0.396`$, so $`x_1 \approx 1.396`$, and var($`x_1`$) drops from 2 to about 1.01.
- Filter that already dropped $`x_0`$: it only has $`x_1\sim N(1,2)`$, so it cannot use a reading of $`x_0`$.

---

## 10. Where the modern systems fit

A rough map is:

```text
                State Estimation
                       │
          ┌────────────┴────────────┐
          │                         │
      Filtering                 Smoothing
          │                         │
    ┌─────┴─────┐             ┌─────┴─────┐
    │           │             │           │
   KF          EKF        Fixed-lag    Full
                           smoothing   smoothing
                               │           │
                               ▼           ▼
                         Factor graphs/Optimization
```

(KF and EKF stay in the dense, recursive filtering loop; fixed-lag and full smoothing are the methods usually solved via sparse factor-graph optimization.)

Examples:

- **EKF-SLAM** → filtering
- **MSCKF** → an EKF-based filter for visual-inertial estimation, run over a sliding window of poses. Landmarks never enter the state; each is used once as a constraint between window poses, then dropped. Because of the window it fits the fixed-lag box only loosely.
- **VINS-Mono/VINS-Fusion** → nonlinear optimization + sliding window
- **ORB-SLAM** → heavily optimization-based
- **GTSAM-based systems** → factor-graph optimization
- **iSAM/iSAM2** → incremental smoothing/optimization

(See [optimization/marginalization.md](optimization/marginalization.md) for the actual mechanics of how a state gets marginalized out and why that's what makes a bounded sliding window possible.)

This is one corner of the repo-wide map in [slam_mental_map.md](slam_mental_map.md), which places every doc and script on one picture.

---

## 11. And this matters a lot for SLAM + state estimation for resource-constrained robots with discontinuous/hybrid motion

A robot with discontinuous/hybrid motion (bio-inspired locomotion, legged contact events, and similar) might experience:

```text
normal motion
     ↓
contact
     ↓
jump/climb/discrete transition
     ↓
new contact
     ↓
normal motion
```

A filtering approach naturally asks:

> **"Given the state right now, what happens next?"**

A smoothing approach can instead ask:

> **"Given the measurements before and after this unusual transition, what was the most consistent trajectory and transition state?"**

That matters when the motion model is **hybrid/discontinuous**, because future observations may provide strong evidence about what actually happened during an ambiguous transition.

In summary:

> **Filtering is like continuously updating a belief about where the robot is.**

> **Smoothing/optimization is like periodically reopening the entire notebook and rewriting the past trajectory so that everything observed so far fits together as consistently as possible.**

That distinction links **EKF-SLAM → factor graphs → sliding-window VIO → pose-graph SLAM → incremental smoothing**.

---

## 12. References

1. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press. - the online-SLAM-vs-full-SLAM framing behind §1's and §6's filtering/smoothing distinction.
2. Dissanayake, M. W. M. G., Newman, P., Clark, S., Durrant-Whyte, H. F., & Csorba, M. (2001). *A Solution to the Simultaneous Localization and Map Building (SLAM) Problem*. IEEE Transactions on Robotics and Automation, 17(3), 229–241. https://doi.org/10.1109/70.938381 - the $O(n^2)$ dense-covariance growth of EKF-SLAM referenced in §6's caveat.
3. Dellaert, F., & Kaess, M. (2006). *Square Root SAM: Simultaneous Localization and Mapping via Square Root Information Smoothing*. International Journal of Robotics Research, 25(12), 1181–1203. https://doi.org/10.1177/0278364906072768 - the sparse smoothing/factor-graph approach behind §4, §7, and §6's caveat.
4. Kaess, M., Johannsson, H., Roberts, R., Ila, V., Leonard, J. J., & Dellaert, F. (2012). *iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree*. International Journal of Robotics Research, 31(2), 216–235. https://doi.org/10.1177/0278364911430419 - the incremental sparse solver behind §6's caveat and §10's iSAM/iSAM2 entry.
5. Dellaert, F., & Kaess, M. (2017). *Factor Graphs for Robot Perception*. Foundations and Trends in Robotics, 6(1–2), 1–139. https://doi.org/10.1561/2300000043 - general reference for the factor-graph formulation used throughout §7.
6. Mourikis, A. I., & Roumeliotis, S. I. (2007). *A Multi-State Constraint Kalman Filter for Vision-Aided Inertial Navigation*. ICRA 2007, 3565–3572. https://doi.org/10.1109/ROBOT.2007.364024 - the MSCKF reference in §10.
7. Qin, T., Li, P., & Shen, S. (2018). *VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator*. IEEE Transactions on Robotics, 34(4), 1004–1020. https://doi.org/10.1109/TRO.2018.2853729 - the VINS-Mono/VINS-Fusion reference in §10.
8. Mur-Artal, R., Montiel, J. M. M., & Tardós, J. D. (2015). *ORB-SLAM: A Versatile and Accurate Monocular SLAM System*. IEEE Transactions on Robotics, 31(5), 1147–1163. https://doi.org/10.1109/TRO.2015.2463671 - the ORB-SLAM reference in §10.

### Further reading

- Awesome Legged Robot Localization and Mapping (GitHub list maintained by KwanWaiPang). https://github.com/KwanWaiPang/Awesome-Legged-Robot-Localization-and-Mapping - a curated, ongoing list of localization and mapping work on legged robots, where §11's contact-driven, discontinuous motion shows up in practice. A link collection, not a peer-reviewed source.
