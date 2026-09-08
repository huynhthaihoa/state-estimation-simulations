# Front-end & Back-end in SLAM

In **SLAM**, the terms **front-end** and **back-end** describe two major parts of the system.

A simple way to think about them is:

> **Front-end: “What happened between these observations?”**

> **Back-end: “Given all observations, what is the most consistent trajectory and map?”**

![Front-end (visual odometry: feature extraction, data association) feeding a back-end optimization stage that produces a mapped point-cloud trajectory, with loop closure looping back into the front-end](images/frontend_backend_1.jpg)

![A tracking / window-optimization visual-odometry pipeline feeding a front-end manager (location awareness, loop-closure constraint) into back-end pose-map optimization, alongside a keyframe / point-cloud / vocabulary dataset](images/frontend_backend_2.jpg)

![A "Front End SLAM" block (feature extraction, submaps, sensor processing, motion estimation) producing a pose, and a separate "Backend SLAM" block (loop closure, graph optimization, global map optimization) producing a map](images/frontend_backend_3.jpg)

![A "Frontend (online processing)" block (scene tracking, pose estimation) and "Backend (offline processing)" block (pose graph, mapping, loop closure, optimization over a cost landscape) producing an optimized trajectory](images/frontend_backend_4.jpg)

*These four diagrams are illustrative sketches - they don't fully agree with each other, or with the sections below, on exactly which stage owns loop-closure detection vs. correction, or on the online/offline framing. See §3 and §5 for that nuance.*

## 1. SLAM Front-end - extracting constraints

The **front-end processes raw sensor data** and turns it into useful geometric information.

For a camera-based SLAM system:

```text
Camera images
     ↓
Feature detection
     ↓
Feature matching / tracking
     ↓
Motion estimation
     ↓
Measurements / constraints
     ↓
       Backend
```

Typical front-end tasks:

* **Feature extraction** - ORB, FAST, SIFT, etc.
* **Feature tracking / matching**
* **Data association** - determining that a feature in frame $k$ is the same physical point seen in frame $k+1$
* **Visual odometry**
* **Depth estimation / triangulation**
* **Keyframe selection**
* **Loop-closure detection**

For example, suppose the camera sees:

```text
Frame 1                  Frame 2
   ●                        ●
      ●                      ●
          ●                    ●
```

The front-end tracks the same points and estimates:

$$T_{12}$$

meaning:

> "The camera moved approximately this much from frame 1 to frame 2."

So the front-end produces **constraints**, rather than necessarily producing the final globally optimal trajectory.

---

## 2. SLAM Back-end - solving the global problem

The **back-end takes those constraints and optimizes the robot's trajectory and map**.

For example, the front-end might produce:

$$T_{01}, T_{12}, T_{23}, T_{34}$$

and observations such as:

$$x_1^0,\ x_2^1,\ x_3^2,\ldots$$

(following the same pairing as $T_{01}, T_{12}, T_{23}$ above, each $x_n^{\,n-1}$ reads as *the observation made at frame $n$, expressed relative to frame $n-1$* - the doc doesn't spell this out explicitly, but the indexing pattern matches.)

The back-end asks:

> "What set of robot poses and landmarks best explains **all** these measurements simultaneously?"

This is usually formulated as an optimization problem.

For example:

$$\min_{\mathbf{x}} \frac12\sum_i \|r_i(\mathbf{x})\|^2$$

where $r_i$ is the error associated with a measurement.

Common back-end techniques include:

* **[Bundle Adjustment](optimization/bundle_adjustment.md)**
* **[Pose-graph optimization](optimization/pose_graph_optimization.md)**
* **[Factor-graph optimization](optimization/factor_graph.md)**
* **[Nonlinear least squares](optimization/nonlinear_least_square.md)**
* **[Gauss-Newton](optimization/gauss_newton.md)**
* **[Levenberg-Marquardt](optimization/levenberg_marquardt.md)**
* **[iSAM / incremental optimization](optimization/isam_optimization.md)**

---

## 3. The most intuitive example: loop closure

Imagine your robot walks around a building:

```text
       A ───── B
       │       │
       │       │
       D ───── C
```

It estimates:

```text
A → B → C → D → A
```

But every motion estimate has a small error.

So the estimated trajectory might become:

```text
       A ───── B
        \       \
         \       C
          D ─────
```

When the robot returns to A, the **front-end detects a loop closure**:

> "Hey! This place looks like somewhere I've seen before."

It generates a constraint:

$$T_{DA} \approx \text{known relationship}$$

The **back-end then optimizes the entire trajectory** so that all constraints are satisfied as well as possible.

In practice, "loop closure" really spans both stages: the front-end / place-recognition module detects the candidate match, while the back-end verifies it and folds it into the global optimization - which is why some of the diagrams above draw the "loop closure" box on the back-end side instead.

That's why you can think of:

> **Front-end = measurement generation**

> **Back-end = constraint optimization**

---

## 4. Front-end vs Back-end

|                    | Front-end                                        | Back-end                                |
| ------------------ | ------------------------------------------------ | --------------------------------------- |
| Main job           | Understand sensor observations                   | Find globally consistent solution       |
| Input              | Raw sensor data                                  | Measurements/constraints                |
| Output             | Features, matches, relative poses, loop closures | Optimized poses/map                     |
| Typical algorithms | Feature tracking, VO, matching                   | BA, pose graph, factor graph            |
| Focus              | Local / sequential                               | Global / accumulated                    |
| Question           | "What happened?"                                 | "What is the best overall explanation?" |

## 5. One important distinction

The front-end **doesn't necessarily mean "real-time"**, and the back-end **doesn't necessarily mean "offline."**

(Image 4 above labels the two stages "online" and "offline" respectively - a common simplification, but, as just stated, not a strict rule.)

In modern SLAM systems, both can operate online:

```text
              FRONT-END
Sensors ──→ Tracking ──→ Constraints
                          │
                          ↓
                    BACK-END
                   Optimization
                          │
                          ↓
                  Updated SLAM state
```

For your research topic, this distinction becomes particularly important because **front-end errors become constraints for the back-end**. If data association or motion estimation is wrong, even a very good optimizer can converge to the wrong solution.

A useful mental model is:

> **Front-end = perception**

> **Back-end = estimation/optimization**.

---

## 6. References

1. Cadena, C., Carlone, L., Carrillo, H., Latif, Y., Scaramuzza, D., Neira, J., Reid, I., &
   Leonard, J. J. (2016). *Past, Present, and Future of Simultaneous Localization and Mapping:
   Toward the Robust-Perception Age*. IEEE Transactions on Robotics, 32(6), 1309–1332.
   https://doi.org/10.1109/TRO.2016.2624754 - the standard survey that frames the front-end/
   back-end split used throughout this doc (§1, §2, §4).
2. Grisetti, G., Kümmerle, R., Stachniss, C., & Burgard, W. (2010). *A Tutorial on Graph-Based
   SLAM*. IEEE Intelligent Transportation Systems Magazine, 2(4), 31–43.
   https://doi.org/10.1109/MITS.2010.939925 - the back-end / pose-graph optimization tutorial
   behind §2 and §4.
3. Mur-Artal, R., Montiel, J. M. M., & Tardós, J. D. (2015). *ORB-SLAM: A Versatile and Accurate
   Monocular SLAM System*. IEEE Transactions on Robotics, 31(5), 1147–1163.
   https://doi.org/10.1109/TRO.2015.2463671 - a concrete worked system pairing an ORB-feature
   front-end with a local-BA + pose-graph back-end, behind §1's ORB mention and §3's loop-closure
   walkthrough.
4. Gálvez-López, D., & Tardós, J. D. (2012). *Bags of Binary Words for Fast Place Recognition in
   Image Sequences*. IEEE Transactions on Robotics, 28(5), 1188–1197.
   https://doi.org/10.1109/TRO.2012.2197158 - the DBoW2 place-recognition method behind §3's
   claim that loop-closure *detection* is a front-end (place-recognition) task.
5. Chen, W., Shang, G., Ji, A., Zhou, C., Wang, X., Xu, C., Li, Z., & Hu, K. (2022). *An Overview
   on Visual SLAM: From Tradition to Semantic*. Remote Sensing, 14(13), 3010.
   https://doi.org/10.3390/rs14133010 - the original source of the diagram in
   `images/frontend_backend_1.jpg` (see Image sources below).
6. Chen, W., Zhou, C., Shang, G., Wang, X., Li, Z., Xu, C., & Hu, K. (2022). *SLAM Overview:
   From Single Sensor to Heterogeneous Fusion*. Remote Sensing, 14(23), 6033.
   https://doi.org/10.3390/rs14236033 - the original source of the diagram in
   `images/frontend_backend_2.jpg` (see Image sources below).
7. Duan, R., Feng, Y., & Wen, C.-Y. (2022). *Deep Pose Graph-Matching-Based Loop Closure
   Detection for Semantic Visual SLAM*. Sustainability, 14(19), 11864.
   https://doi.org/10.3390/su141911864 - Figure 1 of this paper is the original source of the
   diagram in `images/frontend_backend_4.jpg` (see Image sources below).

### Image sources

The 4 diagrams above were originally embedded as hotlinks to an OpenAI-hosted CDN
(`images.openai.com`) and have since been downloaded into [`images/`](images/) for durability.

1. `images/frontend_backend_1.jpg`
   - Original CDN URL: https://images.openai.com/static-rsc-4/ohut01r4hXxy0FBo1CUweMJvMR0DsWr7j5TugX3IqHY3ySm7JWT-MdDMc2CRdBXXvXt09Nx7tRO552QRLOCFQVlHDCt1ECQBAvWQ3EM4vtQVfcF4qvfLvULQutQRozc5gRTmOfctTMUsz0aIsbjamEKvI7-cPBEbDYm31qYmsS9bfmdHnOGJdUH9Z1ke9Gg4?purpose=fullsize
   - Confirmed source: Reference 5 above (Chen et al., 2022, *An Overview on Visual SLAM: From
     Tradition to Semantic*).
2. `images/frontend_backend_2.jpg`
   - Original CDN URL: https://images.openai.com/static-rsc-4/fRJK_NxCUU5MX0xfOKYi4rlB7jQ2LD2ipmlYDd1XKT9PVkvHJ59nZQLTRowQ7zaPmc1Tzewjuf_rumoJ-G38d3DTs4WPCbo0OZ9-qlf_3j4KZgVNfSF75TNlFbV_dW2iWXO4jkKbovsJViao1gfCoFPnqfyO3C6b0bOfIQjQ2wl8FJoT509kwELbxeXlDrRy?purpose=fullsize
   - Confirmed source: Reference 6 above (Chen et al., 2022, *SLAM Overview: From Single Sensor
     to Heterogeneous Fusion*).
3. `images/frontend_backend_3.jpg`
   - Original CDN URL: https://images.openai.com/static-rsc-4/wSKnA5y3wi9kOy12TexCHpO7AOzmAIMkZP2Lubf4gLoaeo0jwVd2DipIoWO0Wl3INhlXLgBCQdyTZDQDPTZ_RdR9zltV7hoG-H7wiRY1Ja5iCt4PRRL85wEpuQhuOsc_bFhxps25YgL-sEyUaeU8fOFDOwQkEP_pkcA_9pUoqBfG5E6skjfrZ_g_hbGjQ_X6?purpose=fullsize
   - Source: not yet identified - no candidate has been found or confirmed. Do not assume it
     shares a source with the other three.
4. `images/frontend_backend_4.jpg`
   - Original CDN URL: https://images.openai.com/static-rsc-4/1oH3r36n4WxtILN9nlK9PXhy52VKoblXYiuWIZXpb-0MKArZ9UwGQ2MPtIRdMgIlUwEwt2pvxyX0Ri8_bukIvSur2fgcoVvciiGxRn3Dqtd1GXQ63CWXEJIAwd1Ua9qad043n0DNKLTnphMUzjYDmzneWRPZPEI_hn9jf-ij253TMJJs6oZh9k7FQ7KgA59-?purpose=fullsize
   - Confirmed source: Reference 7 above (Duan, Feng, & Wen, 2022, *Deep Pose Graph-Matching-
     Based Loop Closure Detection for Semantic Visual SLAM*), Figure 1.