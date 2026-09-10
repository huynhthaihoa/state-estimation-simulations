# Bundle Adjustment

**Bundle Adjustment (BA)** sounds intimidating, but the intuition is actually quite simple:

> **Bundle Adjustment = jointly adjusting camera poses and 3D points so that the observed image measurements are explained as accurately as possible.**

The key word is **jointly**.

---

## 1. Start with a simple camera + 3D point

Imagine a camera looking at a 3D point:

```text
        3D point
           ● P
          / 
         /
        /
       📷 Camera
```

The 3D point `P` gets projected onto the camera image:

```text
3D world                 Image

   P ●                      • p
      \                    /
       \                  /
        \                /
         📷 ------------ image plane
```

If we know:

* camera pose
* camera intrinsics
* 3D point position

we can predict where `P` should appear in the image.

Mathematically:

$${p = \pi(TP)}$$

where:

* `P` = 3D point
* `T` = camera pose
* `π` = camera projection function
* `p` = predicted 2D pixel location

---

## 2. But our estimates are imperfect

Suppose the actual image measurement is:

```text
observed point:       ●
predicted point:          ×
```

There is an error:

```text
        ● observed

           ↕ error

              × predicted
```

This is called the **reprojection error**.

For one observation:

$${e = p_{\text{observed}} - p_{\text{predicted}}}$$

BA tries to make this error as small as possible.

---

## 3. Now add multiple cameras

Suppose the camera moves:

```text
Camera 1          Camera 2          Camera 3

  📷                📷                📷
   \                 \                 \
    \                 \                 \
     ● P               ● P               ● P
```

The same physical 3D point is observed from multiple camera positions.

Each camera produces a 2D observation:

```text
Camera 1 → p₁
Camera 2 → p₂
Camera 3 → p₃
```

We want to find:

* Camera 1 pose
* Camera 2 pose
* Camera 3 pose
* 3D position of P

such that **all projections agree with the observations**.

---

## 4. Here's the important part: both cameras AND points are adjusted

Suppose our initial reconstruction is wrong:

```text
Camera poses:

📷₁       📷₂       📷₃

 \         |         /
  \        |        /
   \       |       /
       ● P
```

Maybe the camera poses are slightly wrong.

Maybe the 3D point is slightly wrong.

Maybe **both** are wrong.

BA doesn't say:

> "The cameras are correct; I'll fix the points."

or:

> "The points are correct; I'll fix the cameras."

Instead:

> **"I'll adjust everything together until the entire reconstruction explains the image measurements as well as possible."**

That's bundle adjustment.

---

## 5. Why is it called "bundle" adjustment?

There's a beautiful geometric intuition.

Each image observation defines a **ray** from the camera through the observed pixel:

```text
Camera 1
   📷
    \
     \
      \       ● 3D point
       \     /
        \   /
         \ /
```

Another camera gives another ray:

```text
📷₁ --------\
             \
              ● P
             /
📷₂ --------/
```

Ideally, the rays intersect exactly at the 3D point.

But because of noise and imperfect estimates:

```text
📷₁ --------\
             \       ●
              \
               \

📷₂ -----------\ 
```

They don't intersect perfectly.

You can think of BA as adjusting the **bundle of rays** and camera poses so that everything fits together better.

Hence:

> **Bundle Adjustment.**

---

## 6. A more useful SLAM example

Imagine a robot/camera moving through a room:

```text
t₀       t₁       t₂       t₃
📷       📷       📷       📷
 \        \        \        \
  \        \        \        \
   ● A      ● B      ● C      ● D
    \       |       / 
     \      |      /
        landmarks
```

The camera observes many landmarks:

```text
      ● L1
     /
📷₀ /— — — — — ● L2
    \          
     \        
      ● L3
```

Every observation creates a constraint:

```text
camera pose + 3D landmark
              ↓
       predicted pixel
              ↓
       compare with
       observed pixel
```

So the system has potentially **thousands or millions of constraints**.

BA solves:

$${\min_{\{T_i\},\{P_j\}} \sum_{(i,j) \in \mathcal{O}} \left\|z_{ij} - \pi(T_i P_j)\right\|^2}$$

where:

* $T_i$ = pose of camera `i`
* $P_j$ = 3D landmark `j`
* $z_{ij}$ = observed pixel
* $\pi(T_iP_j)$ = predicted pixel
* $\mathcal{O}$ = the set of (camera, landmark) pairs that were actually observed - not every camera sees every landmark, so the sum only runs over real observations, not all $i,j$ combinations

In plain English:

> **Find the camera poses and 3D points that make the predicted image points match the actual image points as closely as possible.**

Real implementations usually wrap the squared reprojection error in a **robust loss** (e.g. Huber) instead of squaring it directly, so a handful of bad feature matches can't drag the whole reconstruction toward them - see [pose_graph_optimization.md](pose_graph_optimization.md#16-robust-loss-functions-used-to-handle-false-loop-closures)'s "Robust loss functions" section for the exact same idea applied to pose graphs. `bundle_adjustment.py` keeps the plain, un-robustified squared error above, matching the objective as written here.

---

## 7. Why BA is so powerful

Suppose your estimated trajectory looks like:

```text
Initial:

📷──📷──📷──📷──📷
               \
                \
                 ● landmarks
```

But the actual observations suggest that the cameras should be slightly different:

```text
Optimized:

📷
  \
   📷
     \
      📷
        \
         📷
           \
            📷
```

At the same time, the landmarks move too.

So BA might effectively do:

```text
             Before             After

Camera 1       ×                  ●
Camera 2       ×                  ●
Camera 3       ×                  ●
Camera 4       ×                  ●

Landmark A     ×                  ●
Landmark B     ×                  ●
Landmark C     ×                  ●
```

Everything moves together to minimize the total reprojection error.

---

## 8. Connection to SLAM optimization

This is exactly why BA is an **optimization-based SLAM technique**.

Remember our previous discussion:

> Filtering → maintain the current belief.

> Optimization → maintain many states and jointly improve them.

BA is the latter.

You have a giant optimization problem:

```text
        Camera poses
             ↓
     T₀ T₁ T₂ T₃ T₄
      ↘  ↓  ↙ ↘  ↓
       landmarks
      P₀ P₁ P₂ P₃
             ↓
       projection
             ↓
     predicted pixels
             ↓
     compare with data
             ↓
      total error
             ↓
       optimization
             ↓
      better poses +
      better points
```

---

## 9. BA vs Pose Graph Optimization

This distinction is particularly useful in SLAM.

### Pose graph optimization

Usually optimizes:

$${T_0,T_1,\ldots,T_n}$$

using relative pose constraints:

```text
T₀ ───── T₁ ───── T₂ ───── T₃
 \                         /
  └────── loop closure ───┘
```

The landmarks may already have been marginalized or aren't explicitly part of the optimization.

---

### Bundle Adjustment

Optimizes:

$${\boxed{\text{camera poses + 3D landmarks}}}$$

```text
        T₀       T₁       T₂
         \        |       /
          \       |      /
           P₁    P₂    P₃
```

using **image reprojection errors**.

So a useful mental distinction is:

> **Pose graph:** "Make the poses geometrically consistent."

> **Bundle adjustment:** "Make the entire camera + 3D structure explain the images."

---

## 10. Why BA can be computationally expensive

Imagine:

* 1,000 camera poses
* 100,000 landmarks
* millions of image observations

Then you're optimizing a huge number of variables.

But there's a very useful structure:

```text
Camera variables ─── Landmark variables
       ↕                    ↕
       └──── observations ──┘
```

Camera `T₁` only directly interacts with the landmarks it observes.

That produces a **sparse optimization problem**.

This sparsity is one of the fundamental reasons efficient BA algorithms are possible.

---

## 11. The deepest intuition

Here's how I'd recommend thinking about BA:

Imagine you have a pile of photographs and you're trying to reconstruct a miniature 3D world.

You initially make a rough reconstruction:

```text
     camera       camera
       📷           📷
        \           /
         \         /
          ●       ●
          landmarks
```

Then you ask:

> "If this really were the correct 3D world, would these cameras really see these landmarks at exactly these pixels?"

If not, something is wrong.

Maybe:

* camera 1 is slightly misplaced
* camera 2 is rotated incorrectly
* landmark 1 is too far away
* landmark 2 is too high

So you continuously adjust:

```text
camera poses
     +
3D landmarks
     ↓
projection
     ↓
image error
     ↓
optimization
     ↓
repeat
```

until the reconstruction becomes as consistent with the images as possible.

That's **Bundle Adjustment**.

---

## 12. Block sparsity and the Schur complement

Section 10 already named the sparsity that makes large BA problems tractable; here's the mechanics of how solvers actually exploit it. The same structural fact drives everything below: a residual for point $P_j$ seen by camera $T_i$ depends **only** on $T_i$ and $P_j$ - it's completely independent of any other camera or point.

```text
Camera variables ─── Landmark variables
       ↕                    ↕
       └──── observations ──┘
```

That independence gives the linearized normal-equations Hessian ${H = J^\top J}$ a distinctive **arrow-head** block structure: block-diagonal camera-camera blocks, block-diagonal point-point blocks, and off-diagonal camera-point coupling blocks - with no direct camera-camera or point-point coupling anywhere.

```text
        Cameras          Points
      ┌──────────┬────────────────┐
Cams  │  block-  │                │
      │ diagonal │    coupling    │
      ├──────────┼────────────────┤
Pts   │ coupling │     block-     │
      │          │    diagonal    │
      └──────────┴────────────────┘
```

Since real scenes usually have far more points than cameras, solvers exploit this with the **Schur complement trick**: marginalize out the point block first (cheap, since it's block-diagonal - each point's own small block inverts independently), solve the much smaller reduced camera-only system, then cheaply back-substitute to recover the points. It's the same style of sparsity exploitation that makes [pose_graph_optimization.md](pose_graph_optimization.md#5-solving-the-linear-system-gauss-newton-step)'s sparse Cholesky factorization tractable at scale.

`bundle_adjustment.py` doesn't need this trick - its toy scenes are small enough (a handful of cameras and landmarks) that `run_bundle_adjustment` just solves the full dense joint system directly every iteration. Schur-complement marginalization is what a production solver (COLMAP, g2o, GTSAM, Ceres) does under the hood at real scene sizes, not something this demo implements.

---

## 13. Local vs. Global Bundle Adjustment (real systems)

Local BA and Global BA solve the *exact same* objective from Section 6 - they differ only in how much of the problem gets optimized at once, a choice driven by very different system constraints.

| Metric | **Local BA** (e.g. ORB-SLAM) | **Global BA** (e.g. COLMAP) |
| --- | --- | --- |
| System paradigm | Visual SLAM (real-time, online) | Structure-from-Motion (offline, batch) |
| Optimization scope | A local window of recent keyframes + covisible neighbors | Every registered camera pose and every 3D point |
| Input | Sequential video with continuous tracking | Unordered photo collections (or long video) |
| Frequency | Continuous, runs on every new keyframe | Periodic (e.g. every ~10-20% map growth) or a final pass |
| Outlier handling | Fast local robust cost (Huber) + chi-square gating | Heavy re-triangulation, track merging/filtering |
| Scaling | Roughly constant per window | Grows cubically with the number of camera poses |

### Local BA (ORB-SLAM)

Re-optimizing the entire map on every camera move is impossible in real time, so Local BA **trades global consistency for speed** by isolating a small subgraph:

```text
  [Fixed Keyframe]  sees -> (Fixed Map Point)
         |                             |
  (Covisible Link)               (Observed by)
         |                             v
 [Active Keyframe] < optimizes > [Active Map Point]
```

* **Active keyframes**: the new keyframe plus its neighbors in the **covisibility graph** (keyframes sharing many observed points).
* **Active points**: every 3D point observed by an active keyframe.
* **Fixed keyframes**: other keyframes that also see an active point, held fixed as rigid anchors so the local window can't drift the map's global frame.

Because a covisibility neighborhood's size stays roughly constant regardless of total map size, Local BA runs in bounded, real-time-friendly time - at the cost of letting small errors accumulate into global drift over a long trajectory. SLAM systems correct that separately, via loop closure + pose-graph optimization ([pose_graph_optimization.md](pose_graph_optimization.md)) or an occasional Global BA pass.

### Global BA (COLMAP)

Offline SfM pipelines sacrifice real-time speed for maximum accuracy: as COLMAP incrementally registers new images, it periodically re-optimizes **every** camera and **every** point jointly in one large least-squares problem, then uses the resulting global residuals to prune bad matches and re-triangulate points - something a local window can never do, since it never sees the whole map at once. The cost is that even after the Schur complement from Section 12, the reduced camera-only system still grows cubically with the number of camera poses - and assembling that Schur complement in the first place costs roughly linear time in the number of points/observations - so this can only run periodically or as a final step, not every frame.

### Choosing between them

Use **Local BA** for real-time robotics/AR/VR where sub-30ms latency matters more than perfect global consistency (loop closure repairs that later).

Use **Global BA** for offline reconstruction - meshes, NeRF/Gaussian-Splatting input scenes, photogrammetric surveys - where total geometric fidelity matters more than runtime.

`bundle_adjustment.py`'s three solvers (`run_ba_landmarks_only`, `run_ba_poses_only`, `run_bundle_adjustment`) are all single-batch joint solves over the whole toy scene - closest in spirit to a (tiny) Global BA pass. It has no windowing, no covisibility graph, and no incremental registration, so it doesn't model Local BA's real-time system behavior at all.

`bundle_adjustment_advanced.py` fills that gap: a camera moves keyframe-by-keyframe through a landmark corridor, a covisibility graph is built incrementally, and every new keyframe triggers a bounded local-BA solve over an active window (new keyframe + covisible neighbors, with every other observing keyframe held fixed as a rigid anchor - this section's diagram, made concrete), while a periodic Global BA pass over the whole map runs alongside it for direct comparison. Measuring wall-clock solve time confirms both halves of this section's "Scaling" row in code: the local window's cost stays roughly flat as the map grows, while the Global BA pass's cost grows with it. It also demonstrates *why* an occasional Global BA pass helps - accumulated front-end drift - and, just as informatively, that its benefit is modest without an actual loop closure to supply a genuinely new constraint (see the script's own module docstring for that finding).

---

## 14. Evaluating the result: gauge freedom and Umeyama alignment

Monocular BA recovers the scene only up to an unknown similarity transform (rigid + scale) - shifting, rotating, or uniformly rescaling the whole reconstructed scene and every camera pose together leaves reprojection error completely unchanged. `bundle_adjustment.py`'s `run_bundle_adjustment` pins that freedom down to *some* solution with a soft gauge-prior factor on the first two camera poses, but the resulting frame still won't match ground truth's frame or scale exactly. So before computing pose/landmark error, the script Umeyama-aligns the solved cameras and landmarks onto ground truth with one shared scale+rotation+translation - see [umeyama_alignment.md](../foundations/umeyama_alignment.md) for how that alignment is computed and why it's needed. Reprojection error itself is reported *before* this alignment step and is unaffected by it.

`bundle_adjustment_advanced.py` sidesteps this entirely: hard-fixing two anchor keyframes (rather than a soft prior) removes all residual gauge freedom up front, so there's nothing left to align away before reporting its error.

---

## 15. One sentence to remember

> **Bundle Adjustment is the process of jointly refining camera poses and 3D landmarks so that their projections agree as closely as possible with the observed image features.**

And the conceptual hierarchy is:

```text
SLAM
 │
 ├── Filtering
 │     └── EKF-SLAM
 │
 └── Optimization / Smoothing
       │
       ├── Pose Graph Optimization
       │
       └── Bundle Adjustment
              │
              ├── optimize camera poses
              └── optimize 3D landmarks
```

For **visual SLAM**, BA is essentially the workhorse behind the idea of *"make my entire reconstructed 3D world and camera trajectory agree with all the pixels/features I've observed."*
