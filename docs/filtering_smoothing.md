# Filtering vs. Optimization/Smoothing in SLAM

The easiest way to understand **Filtering vs. Optimization/Smoothing in SLAM** is to think about **when the robot is allowed to change its mind about the past**.

---

## 1. The core idea

Suppose a robot moves through a room:

**t₀ → t₁ → t₂ → t₃ → t₄**

At every time, it gets:

* IMU measurements
* camera/LiDAR observations
* wheel odometry, etc.

The robot wants to estimate:

> **Where am I? What does the environment look like?**

The fundamental difference is:

### Filtering

> **Estimate the current state using everything available up to now.**

### Optimization / Smoothing

> **Estimate a whole trajectory by jointly considering many states and measurements, including information that may arrive later.**

This leads to a very useful mental model:

> **Filtering = "What do I believe right now?"**

> **Smoothing = "Given everything I've seen, what do I now believe my whole trajectory was?"**

---

## 2. Filtering: one step at a time

Imagine the robot is at time `t₃`.

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

Then the robot moves to `t₄`.

The filter takes the estimate at `t₃`, propagates it forward, incorporates `z₄`, and produces the estimate at `t₄`.

### Important property

Once the filter has moved on, it generally **doesn't go back and completely reconsider old states**.

For example:

```text
t₀       t₁       t₂       t₃
|--------|--------|--------|
                           ↑
                     current estimate
```

It essentially says:

> "I have compressed everything before `t₃` into my current belief. Now let's continue."

This makes filtering **naturally online and computationally efficient**.

### The big limitation of filtering

Here's where SLAM becomes interesting.

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

Now the robot realizes that its previous trajectory was wrong: the new information tells us it should have ended up back near its starting position, not off at a separate point `E`.

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

A pure filtering mindset is uncomfortable with this because it has already compressed the past.

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

> **Find the set of poses and landmarks that best explains all the measurements.**

Mathematically, you can think of:

$${\mathbf{x}^*=\arg\min_{\mathbf{x}}\sum_i \|r_i(\mathbf{x})\|^2}$$

where:

* $\mathbf{x}$ = all poses/landmarks
* $r_i(\mathbf{x})$ = measurement residual
* optimization finds the trajectory that minimizes the total error.

This is the fundamental idea behind **bundle adjustment**, **pose-graph optimization**, and many modern SLAM systems.

### Why optimization is powerful

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

### Optimization can revise the past.

That's a huge conceptual difference.

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

Optimization is often used as the computational mechanism for obtaining this joint estimate.

---

## 5. A very intuitive analogy

Imagine you're reconstructing a person's route through a city.

### Filtering

You ask the person at every intersection:

> "Where are you now?"

At intersection 1:

> "I'm probably here."

At intersection 2:

> "Now I'm probably here."

At intersection 3:

> "I'm probably here."

You keep moving forward.

You don't spend much time reconsidering previous answers.

---

### Smoothing

At the end of the day, you give them:

* GPS measurements
* photos
* timestamps
* landmarks
* the final destination

and ask:

> **"Given everything we know now, where were you at every point during the day?"**

Suddenly they realize:

> "Oh! That landmark I saw at 2 PM was actually the same place I saw at 10 AM."

So they revise their estimate of where they were at 11 AM, 12 PM, 1 PM, etc.

That's smoothing.

---

## 6. The computational difference

This gives us a useful trade-off:

|                    | Filtering                      | Optimization / Smoothing                   |
| ------------------ | ------------------------------ | ------------------------------------------ |
| Main question      | Where am I **now**?            | Where was I **throughout the trajectory**? |
| State maintained   | Current belief                 | Many historical states                     |
| Uses past          | Compressed into current belief | Explicitly retained                        |
| Can revise past?   | Limited                        | Yes                                        |
| Loop closure       | More difficult                 | Natural                                    |
| Computation        | Usually cheaper                | Usually more expensive                     |
| Memory             | Lower                          | Higher                                     |
| Online operation   | Excellent                      | Possible, but needs management             |
| Global consistency | Harder                         | Stronger                                   |
| Typical idea       | EKF-SLAM                       | Pose graph / factor graph / BA             |

**A caveat worth remembering**: "filtering is cheaper, optimization is more expensive" is the right intuition for small, fixed-size problems, but it inverts at scale. EKF-style filtering maintains a *dense* joint covariance over the state, so each update costs roughly $O(n^2)$ in the number of landmarks/poses (Dissanayake et al., 2001). Sparse factor-graph smoothing exploits the sparsity of the underlying graph, so incremental solvers like iSAM2 update in close to $O(1)$ – ${O(\log n)}$ amortized time even as the map grows (Kaess et al., 2012). That gap - not accuracy - is the actual reason large-scale SLAM systems moved from EKF-SLAM toward factor-graph smoothing: dense filtering simply doesn't scale to large maps the way sparse smoothing does.

---

## 7. This is where factor graphs fit

A **factor graph** is an excellent mental bridge between the two worlds.

A factor graph has two kinds of factors. A **landmark factor** ties a pose to a landmark it observed:

```text
      landmark
         ●
        / \
       /   \
 x₀   ●     ● x₂
```

A **pose-to-pose factor** ties two consecutive (or, for a loop closure, non-consecutive) poses together via a relative measurement:

```text
x₀ ● ── x₁ ● ── x₂ ● ── x₃ ●
```

Each measurement becomes a **factor** imposing a constraint.

For example:

```text
IMU factor:
x₁ ───────── x₂

Visual odometry:
x₂ ───────── x₃

Loop closure:
x₃ ───────── x₀
```

Optimization then asks:

> **"What configuration of x₀, x₁, x₂, x₃ best satisfies all these constraints?"**

This is why factor graphs are so common in modern SLAM.

---

## 8. Filtering vs smoothing in one picture

Think of the information flow:

### Filtering

```text
                    ┌──────────┐
measurements ──────►│  FILTER  │──────► current state
                    └──────────┘
                         │
                         ▼
                    summarize past
```

The past gets **compressed**.

### Smoothing / optimization

```text
measurements
     │
     ├─────────┐
     ├─────────┤
     ├─────────┤
     ▼         ▼
   x₀  ─── x₁ ─── x₂ ─── x₃
    \                 /
     └── loop closure┘
             │
             ▼
        JOINT OPTIMIZATION
             │
             ▼
     optimized trajectory
```

The past is **kept around so it can be reconsidered**.

---

## 9. One subtle but very important point

It's tempting to say:

> "Filtering is local, optimization is global."

That's **mostly useful intuitively, but not strictly correct**.

A filter can incorporate loop closures and other global information. For example, EKF-SLAM can update the entire state covariance/mean when a landmark is re-observed.

The more precise distinction is:

> **Filtering recursively represents the current posterior and marginalizes old information, whereas smoothing maintains a posterior over multiple states and can jointly revise them.**

This distinction becomes especially important when studying **VIO, SLAM, and state estimation**.

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
                         Factor graphs / Optimization
```

(KF and EKF stay in the dense, recursive filtering loop; fixed-lag and full smoothing are the methods usually solved via sparse factor-graph optimization.)

Examples you'll encounter:

* **EKF-SLAM** → filtering
* **MSCKF** → an EKF-based filter for visual-inertial estimation (a sliding window of poses, landmarks marginalized out rather than kept in the state)
* **VINS-Mono / VINS-Fusion** → nonlinear optimization + sliding window
* **ORB-SLAM** → heavily optimization-based
* **GTSAM-based systems** → factor-graph optimization
* **iSAM / iSAM2** → incremental smoothing/optimization

---

## 11. And this matters a lot for your PhD topic

For your planned work on **SLAM + state estimation for resource-constrained robots with discontinuous/hybrid motion**, this distinction is particularly important.

Your robot might experience:

```text
normal motion
     ↓
contact
     ↓
jump / climb / discrete transition
     ↓
new contact
     ↓
normal motion
```

A filtering approach naturally asks:

> **"Given my state right now, what happens next?"**

A smoothing approach can instead ask:

> **"Given the measurements before and after this unusual transition, what was the most consistent trajectory and transition state?"**

That can become very interesting when your motion model is **hybrid/discontinuous**, because future observations may provide strong evidence about what actually happened during an ambiguous transition.

So, if you remember only one thing:

> **Filtering is like continuously updating your belief about where you are.**

> **Smoothing/optimization is like periodically reopening the entire notebook and rewriting your past trajectory so that everything you've observed fits together as consistently as possible.**

And that distinction is one of the most useful conceptual foundations for understanding **EKF-SLAM → factor graphs → sliding-window VIO → pose-graph SLAM → incremental smoothing**.

---

## 12. References

1. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press. - the online-SLAM-vs-full-SLAM framing behind §1's and §6's filtering/smoothing distinction.
2. Dissanayake, M. W. M. G., Newman, P., Clark, S., Durrant-Whyte, H. F., & Csorba, M. (2001). *A Solution to the Simultaneous Localization and Map Building (SLAM) Problem*. IEEE Transactions on Robotics and Automation, 17(3), 229–241. https://doi.org/10.1109/70.938381 - the $O(n^2)$ dense-covariance growth of EKF-SLAM referenced in §8's caveat.
3. Dellaert, F., & Kaess, M. (2006). *Square Root SAM: Simultaneous Localization and Mapping via Square Root Information Smoothing*. International Journal of Robotics Research, 25(12), 1181–1203. https://doi.org/10.1177/0278364906072768 - the sparse smoothing/factor-graph approach behind §4, §9, and §8's caveat.
4. Kaess, M., Johannsson, H., Roberts, R., Ila, V., Leonard, J. J., & Dellaert, F. (2012). *iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree*. International Journal of Robotics Research, 31(2), 216–235. https://doi.org/10.1177/0278364911430419 - the incremental sparse solver behind §8's caveat and §12's iSAM/iSAM2 entry.
5. Dellaert, F., & Kaess, M. (2017). *Factor Graphs for Robot Perception*. Foundations and Trends in Robotics, 6(1–2), 1–139. https://doi.org/10.1561/2300000043 - general reference for the factor-graph formulation used throughout §9.
6. Mourikis, A. I., & Roumeliotis, S. I. (2007). *A Multi-State Constraint Kalman Filter for Vision-Aided Inertial Navigation*. ICRA 2007, 3565–3572. https://doi.org/10.1109/ROBOT.2007.364024 - the MSCKF reference in §12.
7. Qin, T., Li, P., & Shen, S. (2018). *VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator*. IEEE Transactions on Robotics, 34(4), 1004–1020. https://doi.org/10.1109/TRO.2018.2853729 - the VINS-Mono/VINS-Fusion reference in §12.
8. Mur-Artal, R., Montiel, J. M. M., & Tardós, J. D. (2015). *ORB-SLAM: A Versatile and Accurate Monocular SLAM System*. IEEE Transactions on Robotics, 31(5), 1147–1163. https://doi.org/10.1109/TRO.2015.2463671 - the ORB-SLAM reference in §12.
