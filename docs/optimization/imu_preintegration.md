# IMU preintegration

> **IMU preintegration compresses thousands of raw, sensor-rate IMU samples between two keyframes into one relative-motion factor - and lets that factor be instantly recomputed when the bias estimate changes, without re-touching a single raw sample.**

This builds directly on two things you've already seen: 
- The **node/edge language** from [pose_graph_optimization.md](pose_graph_optimization.md#2-where-do-the-edges-come-from) ("an edge is a relative-motion constraint between two nodes")

- The **right Jacobian** from [jacobian.md §11](../foundations/jacobian.md#11-left-and-right-jacobians-sensitivity-on-a-curved-space) (" $J_r$ converts a tangent-space nudge into a body-frame-composed rotation"). 

IMU preintegration is where both ideas get used together for a real sensor.

---

## 1. The problem

An IMU reports raw angular rate and acceleration at 100-1000 Hz. Camera keyframes or loop closures, which is what a pose graph or bundle-adjustment back-end actually optimizes over, arrive far more slowly - often 1-10 Hz. That mismatch creates two separate problems:

- **Problem A - too many samples**: If every raw IMU sample became its own node in the graph, a single second of motion would add 100+ nodes and edges. The graph would be enormous, and most of that detail is irrelevant to the back-end, which only cares about the *net* relative motion between two keyframes.

- **Problem B - bias re-linearization**: IMUs have slowly-drifting gyro/accel biases ($b_g$, $b_a$) that the optimizer refines as part of the state. Every raw sample was integrated using *some* bias estimate. When the optimizer updates that estimate - which happens on essentially every iteration - the integration technically has to be redone with the new bias, because the raw samples were corrected using the old one. Redoing a 100+-sample integration loop every time the bias nudges, for every IMU segment in the graph, every optimizer iteration, is not something you can afford.

Preintegration solves both: 

- It folds every raw sample between two keyframes into a single **relative-motion bundle** ($\Delta R$, $\Delta v$, $\Delta p$) - solving **Problem A**. 

- It tracks, *as it integrates*, a set of **bias-sensitivity Jacobians** that let that bundle be corrected for a bias change with one matrix-vector multiply instead of a full re-integration - solving **Problem B**.

---

## 2. The analogy: a trip summary, not a replayed dashcam

Imagine you drove somewhere and want to know: how far did I go, and what would that distance have been if I'd assumed slightly different fuel efficiency?

You don't re-play the entire dashcam footage frame-by-frame every time someone tweaks the fuel-efficiency assumption. Instead, your trip computer keeps a running **summary** (total distance, average speed) *plus* a **sensitivity** ("a 5% change in my efficiency assumption would have shifted my logged consumption by about this much"). Tweaking the assumption afterward is one multiplication against that sensitivity - not a second drive.

$\Delta R, \Delta v, \Delta p$ are the trip summary. $J_{R,b_g}, J_{v,b_g}, J_{v,b_a}, J_{p,b_g}, J_{p,b_a}$ (defined in §4 below) are the sensitivities.

---

## 3. What gets compressed

At every raw IMU sample, [`use_numpy/imu_preintegration.py`](../../use_numpy/imu_preintegration.py)'s `PreintegratedIMUBundle.integrate_measurement` folds one micro-step into three running quantities, all expressed relative to the body frame at the *start* of the integration window:

$$\Delta R \leftarrow \Delta R \cdot \text{Exp}\big((\tilde\omega - b_g)\,dt\big)$$

$$\Delta v \leftarrow \Delta v + \Delta R\,(\tilde v - b_a)\,dt$$

$$\Delta p \leftarrow \Delta p + \Delta v\,dt + \tfrac{1}{2}\Delta R\,(\tilde v - b_a)\,dt^2$$

where $\tilde\omega, \tilde v$ are the raw sensor readings and $b_g, b_a$ are the bias estimates *in effect when this bundle started*. (The script's own comment flags a simplification worth knowing: a real IMU's second channel measures raw *acceleration*, which would need one more integration to reach velocity; this demo integrates a commanded *velocity* channel directly instead, purely to keep the manifold bookkeeping legible. The $\Delta R$ recursion - the part this doc is actually about - is unaffected either way.)

Notice $\Delta R$'s update is exactly the $\text{Exp}$-map composition from [jacobian.md §11.1](../foundations/jacobian.md#111-why-plain-addition-breaks): rotations don't add, so each micro-step's tiny rotation gets *composed onto* the running $\Delta R$, not added to it.

---

## 4. The bias-sensitivity Jacobians

This is the part that makes preintegration more than just "add up the samples." Alongside $\Delta R, \Delta v, \Delta p$, the bundle also propagates five Jacobian matrices - one for how each compressed quantity would shift per unit change in each bias:

$$J_{R,b_g} = \frac{\partial \Delta R}{\partial b_g} \qquad J_{v,b_g} = \frac{\partial \Delta v}{\partial b_g} \qquad J_{v,b_a} = \frac{\partial \Delta v}{\partial b_a} \qquad J_{p,b_g} = \frac{\partial \Delta p}{\partial b_g} \qquad J_{p,b_a} = \frac{\partial \Delta p}{\partial b_a}$$

These are exactly the "sensitivity map" idea from [jacobian.md §3](../foundations/jacobian.md#3-think-of-it-as-a-sensitivity-map) - "if I nudge the input a little, how much does the output move?" - just tracked incrementally, one micro-step at a time, instead of computed once from a closed-form function. Each step updates them using the *current* micro-step's rotation $dR = \text{Exp}((\tilde\omega-b_g)dt)$ and its right Jacobian $J_r = J_r\big((\tilde\omega-b_g)dt\big)$:

$$J_{R,b_g} \leftarrow dR^\top J_{R,b_g} - J_r\,dt$$

$$J_{v,b_g} \leftarrow J_{v,b_g} - \Delta R\,[\tilde v-b_a]_\times J_{R,b_g}\,dt \qquad J_{v,b_a} \leftarrow J_{v,b_a} - \Delta R\,dt$$

$$J_{p,b_g} \leftarrow J_{p,b_g} + J_{v,b_g}\,dt - \tfrac{1}{2}\Delta R\,[\tilde v-b_a]_\times J_{R,b_g}\,dt^2 \qquad J_{p,b_a} \leftarrow J_{p,b_a} + J_{v,b_a}\,dt - \tfrac{1}{2}\Delta R\,dt^2$$

The one worth staring at is $J_{R,b_g}$, because it's a direct instance of [jacobian.md §11.4](../foundations/jacobian.md#114-closed-form-for-so3)'s identities: $J_r\,dt$ is literally the right Jacobian of that micro-step's rotation, and $dR^\top$ is the frame-transport term (recall $J_l = R\,J_r$, i.e. left-multiplying by a rotation or its transpose is how a tangent-space sensitivity gets carried from one frame into the next). Every other $J_{\cdot,b_g}$ in this list inherits from $J_{R,b_g}$ - that's why a bias-Jacobian discussion for IMU preintegration is really a right-Jacobian discussion in disguise.

---

## 5. The payoff: correcting for a bias change without re-integrating

Once the bundle has absorbed, say, 100 raw samples, suppose the graph optimizer updates its bias estimate from $b_g \to b_g'$ (a small change $\delta b_g = b_g' - b_g$, since bias only drifts slowly between optimizer iterations - exactly the "small change" regime a Jacobian is built for, per [jacobian.md §8](../foundations/jacobian.md#8-one-subtle-point-jacobian-is-local)). `get_corrected_measurement` applies the first-order Taylor correction directly, no raw samples touched:

$$\Delta R' \approx \Delta R\cdot\text{Exp}\big(J_{R,b_g}\,\delta b_g\big) \qquad \Delta v' \approx \Delta v + J_{v,b_g}\,\delta b_g + J_{v,b_a}\,\delta b_a \qquad \Delta p' \approx \Delta p + J_{p,b_g}\,\delta b_g + J_{p,b_a}\,\delta b_a$$

Notice the asymmetry: $\Delta v'$ and $\Delta p'$ are corrected by plain addition, because velocity and position are ordinary vectors - but $\Delta R'$ has to go back through $\text{Exp}(\cdot)$ and get *composed* onto $\Delta R$, because rotations aren't. That's the same distinction [jacobian.md §11.1](../foundations/jacobian.md#111-why-plain-addition-breaks) opened with, showing up again here as a concrete asymmetry inside one function.

The whole point: this is one matrix-vector multiply per bias component, independent of how many raw samples fed the bundle - $O(1)$ instead of $O(N)$. The script's own demonstration frames it exactly this way: "took less than 1 microsecond to solve, completely bypassing the need to loop through the 100 raw sensor inputs again."

---

## 6. Where this fits in a SLAM back-end

A preintegrated bundle becomes one **edge** in the same sense as [pose_graph_optimization.md §2](pose_graph_optimization.md#2-where-do-the-edges-come-from) - except instead of connecting two pose nodes, it connects two *(pose, velocity, bias)* node sets:

```text
(pose_i, v_i, b_i) ──── ΔR, Δv, Δp ────  (pose_j, v_j, b_j)
                     preintegrated
                      IMU factor
```

A full VIO/VI-SLAM back-end (GTSAM's `CombinedImuFactor`, ORB-SLAM3, VINS-Mono, etc.) then builds a residual comparing this corrected bundle against what the two connected nodes' current estimates imply - the same "measured vs. predicted, then `Log`" pattern as [pose_graph_optimization.md §7](pose_graph_optimization.md#7-the-mathematics-is-actually-quite-intuitive). Because the bundle can be bias-corrected in $O(1)$, this residual and its Jacobian can be cheaply re-evaluated at every optimizer iteration without ever revisiting the raw 100+ Hz stream - which is the entire reason preintegration exists.

---

## 7. What the accompanying scripts actually do (and don't)

- **`imu_preintegration.py`** (both [`use_numpy/`](../../use_numpy/imu_preintegration.py) and [`use_manif/`](../../use_manif/imu_preintegration.py)) implements exactly §3-§5 above: compress raw samples into a bundle, track the bias Jacobians, then apply the $O(1)$ correction. The two versions are line-for-line equivalent - `use_manif`'s docstring spells out the exact correspondence (its `rplus()` call returns the same `dR`/right-Jacobian pair the numpy version computes by hand). **Neither script builds the graph residual described in §6** - both stop at demonstrating the compression + correction, which is the specific mechanism this doc covers.
- **`robot_imu_simulation.py`** is a related but different script: it dead-reckons raw IMU readings directly on $SE(3)$ every micro-step (chaining `se3_exp`, no bundle, no bias Jacobians at all), then periodically runs a small Gauss-Newton correction against a separate GPS-like sensor. It's about *strapdown integration + periodic on-manifold correction*, not preintegration.
- **`imu_integration_comparison.py`** is also related but different: like [ekf_iekf_equivalence.md](../filtering/ekf_iekf_equivalence.md), it's an empirical comparison note rather than a concept explainer - it plots naive Euler-angle (vector-space) integration against proper $SO(3)$ Exp-map integration to show why the latter doesn't drift the way the former does. No bias correction involved.

---

## 8. One-sentence summary

> **Preintegration turns "re-run 100+ raw IMU samples every time the bias estimate changes" into "one matrix-vector multiply," by tracking - while integrating - exactly how sensitive the compressed relative-motion bundle is to the bias, using the same right-Jacobian machinery that converts a tangent-space nudge into a body-frame rotation anywhere else in this codebase.**
