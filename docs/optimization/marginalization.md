# Marginalization and sliding-window smoothing

> **Marginalizing a variable means permanently removing it from the optimization while keeping everything it taught you about its neighbors, packaged as one new prior factor - so a real-time estimator can bound its problem size without lying to itself about what it used to know.**

This builds directly on two things you've already seen:
- **Variable elimination** from [bayes_tree.md §3](bayes_tree.md#3-elimination-is-the-key-idea) ("eliminate x1 → its info gets summarized into a new constraint on x2") - this doc reuses that exact mechanic for a different purpose.
- **Sparsity and full vs. fixed-lag smoothing** from [filtering_smoothing.md §9-10](../filtering_smoothing.md#9-one-subtle-but-very-important-point) (filtering marginalizes old information, smoothing keeps it) - this doc is the missing mechanical middle ground the diagram there only names.

---

## 1. The problem sliding-window smoothing needs to solve

Full batch optimization (plain [bundle_adjustment.md](bundle_adjustment.md) / [pose_graph_optimization.md](pose_graph_optimization.md)) keeps every pose ever seen in the optimization forever - the problem grows without bound as the robot keeps moving. [isam2_optimization.md](isam2_optimization.md) fixes the *recompute* cost (only touch the part of the Bayes tree a new factor actually affects) but not the *memory* cost - every variable is still in the graph, just efficiently re-solved.

A real-time VIO/VI-SLAM front-end (MSCKF, VINS-Mono, OKVIS) often can't afford either: fixed onboard memory, fixed per-frame compute budget, running forever. The fix is to actively **forget** old states - but forgetting a pose's *variable* while keeping the *information* it contributed is exactly what marginalization does.

---

## 2. Marginalization: keep the information, drop the variable

Recall the elimination picture from `bayes_tree.md §3`:

```text
Before:

x1 ─── x2 ─── x3

Eliminate x1:

        x2 ─── x3
         ↑
    summarized
    information
    from x1
```

That's marginalization. `bayes_tree.md` uses this step to build a *solve order* - x1 is eliminated, but conceptually every variable gets eliminated eventually, and nothing is thrown away for good (§3 below makes this precise). Sliding-window smoothing uses the *identical* elimination math for the opposite reason: to throw x1 away **permanently**, on purpose, because it's the oldest pose in the window and the estimator will never touch it again.

---

## 3. Three flavors of elimination, compared

| | Structural (BA landmarks) | Solve-order (Bayes tree / iSAM2) | Temporal (this doc) |
|---|---|---|---|
| What's eliminated | 3D landmarks | any variable, in a chosen order | the oldest pose/state in the window |
| Is it recoverable? | Yes - back-substitution recovers it | Yes - re-eliminated on the next update | **No** - gone for good |
| Why eliminate it | Landmarks outnumber cameras; cheap to invert | Reuse most of the last solve; minimize fill-in | Bound memory/compute to a fixed window size |
| Where it's covered | [bundle_adjustment.md §12](bundle_adjustment.md#12-block-sparsity-and-the-schur-complement) | [bayes_tree.md](bayes_tree.md), [isam2_optimization.md](isam2_optimization.md) | §4-§7 below |

The math (§4) is the same Schur-complement elimination in all three rows. What differs is only what happens to the eliminated variable afterward.

---

## 4. The math: turning an eliminated pose into a prior

Suppose the current window has poses $x_a$ (the oldest, about to be dropped) and $x_b$ (everything still connected to it - odometry neighbors, and any landmark/IMU-bias variables it shares factors with). After linearization, the joint Gaussian is described by an information matrix $\Lambda$ and information vector $\eta$, partitioned to match:

$$
\Lambda = \begin{bmatrix}\Lambda_{aa} & \Lambda_{ab} \\ \Lambda_{ba} & \Lambda_{bb}\end{bmatrix}, \qquad \eta = \begin{bmatrix}\eta_a \\ \eta_b\end{bmatrix}
$$

Marginalizing out $x_a$ means integrating it out of the joint distribution, which has a closed form - the same Schur complement `bundle_adjustment.md §12` uses on the point block, just kept in information form here instead of being back-substituted afterward:

$$
\Lambda_b' = \Lambda_{bb} - \Lambda_{ba}\Lambda_{aa}^{-1}\Lambda_{ab}, \qquad \eta_b' = \eta_b - \Lambda_{ba}\Lambda_{aa}^{-1}\eta_a
$$

$(\Lambda_b', \eta_b')$ is a brand-new **prior factor** over exactly the variables $x_a$ used to connect to - nothing else. It gets inserted into the graph like any other factor:

```text
Before marginalization:

x_a ──odom── x_b ──odom── x_c
 │
 └──landmark/IMU-bias factors

After marginalizing x_a:

            x_b ──odom── x_c
             ↑
        new prior factor
     (Λ_b', η_b'), no x_a left
```

The critical difference from BA's version of this trick: BA computes $\Lambda_b'$ *only to solve the reduced system faster*, then back-substitutes to recover $x_a$'s optimal value from $x_b$'s. Here there is no back-substitution step - $x_a$ is never coming back. That's what makes this *temporal* marginalization rather than a solver optimization.

---

## 5. The fill-in consequence

$\Lambda_{ba}\Lambda_{aa}^{-1}\Lambda_{ab}$ is a dense (or denser) block whenever $x_a$ touched more than one neighbor. If $x_a$ was connected to, say, three other poses that were previously unconnected to each other, marginalizing $x_a$ makes all three of them mutually connected in the new prior:

```text
Before:                  After eliminating x_a:

     x_a                       x1 ─── x2
    /  |  \                     \    /
   x1  x2  x3                    \  /
                                   x3

(x1,x2,x3 only connect        (x1,x2,x3 now all
 through x_a)                  directly connected)
```

This is exactly the **fill-in** phenomenon `isam2_optimization.md §7` mentions in passing and `bayes_tree.md` builds its whole variable-*ordering* strategy around minimizing. The difference here: sliding-window marginalization doesn't get to choose an elimination order to minimize fill-in - it must always eliminate the oldest pose, whatever it happens to be connected to. This is a real, accepted cost of the sliding-window approach (not a bug to fix), and it's why sliding-window systems keep the window itself small - a bigger window means a more richly-connected pose to eventually eliminate, and a denser resulting prior.

---

## 6. The consistency gotcha: why FEJ exists

One subtlety worth flagging, not deriving in full: $(\Lambda_b', \eta_b')$ is computed by linearizing around whatever the *current* estimates of $x_a$ and $x_b$ happened to be at the moment of marginalization. That linearization point is then frozen into the prior forever. But the surviving variables in $x_b$ keep getting relinearized at new estimates on every later optimizer iteration - at a *different* point than the one the frozen prior was built from.

This mismatch injects spurious information into directions of the state that should be unobservable (the same class of problem [pose_graph_optimization.md](pose_graph_optimization.md) and `kf_ekf_iekf.md` describe for plain EKF-SLAM), making the estimator overconfident. The standard fix is **First-Estimate Jacobians (FEJ)**: once a variable has contributed to a marginalization, all *future* Jacobians involving it are evaluated at that same first-linearization point, not the newest estimate - trading a small amount of accuracy for provable consistency. MSCKF and VINS-Mono both use this. See Huang, Mourikis & Roumeliotis (2009) in the references below for the full derivation - it isn't reproduced here.

---

## 7. Sliding-window and fixed-lag smoothing: the payoff

This is the strategy `filtering_smoothing.md §10`'s diagram names but doesn't mechanize - now it can be stated as a loop:

```text
new keyframe/measurement arrives
        ↓
add its variables + factors to the window
        ↓
   is the window full?
    ↙          ↘
  no            yes
   │             ↓
   │      marginalize the oldest state
   │      (§4: produces a new prior,
   │       §5: densifies its neighbors)
   ↓             ↓
        repeat
```

Because the window never grows past a fixed size, both the per-step optimization cost and memory stay **O(window size)**, independent of how long the robot has been running - the property full batch smoothing doesn't have (unbounded) and iSAM2 doesn't quite give you either (iSAM2 is bounded by how much of the Bayes tree a new factor *changes*, not by a fixed window - a loop closure can still touch a large chunk of history, per `isam2_optimization.md §9`).

This is precisely what `filtering_smoothing.md §10`'s "fixed-lag smoothing" box and its MSCKF/VINS-Mono bullets refer to - MSCKF keeps a sliding window of camera poses and marginalizes a landmark's constraint into them once triangulated (see [kf_ekf_iekf.md](../filtering/kf_ekf_iekf.md)'s MSCKF paragraph); VINS-Mono keeps a sliding window of keyframes and marginalizes the oldest one using exactly the Schur-complement step in §4, with FEJ (§6) to keep it consistent.

---

## 8. What this repo implements (and doesn't)

This doc is conceptual only - unlike most docs in `optimization/`, there is no accompanying script in `use_numpy/` or `use_manif/` that performs marginalization. `pose_graph_incremental.py` (behind [isam2_optimization.md](isam2_optimization.md)) grows its graph but never drops a variable from it. A reader wanting to exercise §4 for real would extend that script to marginalize its oldest node once a fixed window size is exceeded - a natural follow-up, not something currently implemented here.

---

## 9. References

1. Sibley, G., Matthies, L., & Sukhatme, G. (2010). *Sliding Window Filter with Application to Planetary Landing*. Journal of Field Robotics, 27(5), 587-608. https://doi.org/10.1002/rob.20360 - the sliding-window/delayed-state-marginalization formulation behind §4 and §7.
2. Huang, G. P., Mourikis, A. I., & Roumeliotis, S. I. (2009). *A First-Estimates Jacobian EKF for Improving SLAM Consistency*. In Experimental Robotics: The Eleventh International Symposium (pp. 373-382). Springer. https://doi.org/10.1007/978-3-642-00196-3_43 - the FEJ fix behind §6.
3. Mourikis, A. I., & Roumeliotis, S. I. (2007). *A Multi-State Constraint Kalman Filter for Vision-Aided Inertial Navigation*. ICRA 2007, 3565-3572. https://doi.org/10.1109/ROBOT.2007.364024 - the MSCKF reference in §7, already cited in [filtering_smoothing.md §12](../filtering_smoothing.md#12-references).
4. Qin, T., Li, P., & Shen, S. (2018). *VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator*. IEEE Transactions on Robotics, 34(4), 1004-1020. https://doi.org/10.1109/TRO.2018.2853729 - the VINS-Mono reference in §7, already cited in [filtering_smoothing.md §12](../filtering_smoothing.md#12-references).

---

## 10. One-sentence summary

> **Marginalization is the same variable-elimination step `bayes_tree.md` uses to build a solve order, aimed instead at permanently discarding an old state - turning it into a dense prior factor over whatever it was still connected to, which is exactly the trick that lets sliding-window/fixed-lag smoothers (MSCKF, VINS-Mono) run in bounded memory and time forever, at the cost of a fill-in penalty and a consistency subtlety (FEJ) that full-batch and iSAM2 never have to deal with.**
