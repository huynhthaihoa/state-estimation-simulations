# IMU preintegration

IMU preintegration compresses thousands of raw, sensor-rate IMU samples between two keyframes into one relative-motion factor - and lets that factor be instantly recomputed when the bias estimate changes, without re-touching a single raw sample.

This builds directly on two things you've already seen:
- The **node/edge language** from [pose_graph_optimization.md](pose_graph_optimization.md#2-where-do-the-edges-come-from) ("an edge is a relative-motion constraint between two nodes")

- The **right Jacobian** from [jacobian.md §11](../foundations/jacobian.md#11-left-and-right-jacobians-sensitivity-on-a-curved-space) ($\text{Exp}(\varphi+\delta\varphi) \approx \text{Exp}(\varphi)\cdot\text{Exp}\big(J_r(\varphi)\,\delta\varphi\big)$ - the extra rotation composed on the *right*, i.e. expressed in the object's own body frame).

IMU preintegration is where both ideas get used together for a real sensor.

> **Note**: An IMU (Inertial Measurement Unit) is the physical sensor this doc is about - a gyroscope (angular rate) and accelerometer (linear acceleration) packaged together, sampled at the 100-1000 Hz rate described in §1 below.

---

## 1. The problem

An IMU reports raw angular rate and acceleration at 100-1000 Hz. Camera keyframes or loop closures, which is what a pose graph or bundle-adjustment back-end actually optimizes over, arrive far more slowly - often 1-10 Hz. That mismatch creates two separate problems:

- **Problem A - too many samples**: If every raw IMU sample became its own node in the graph, a single second of motion would add 100+ nodes and edges. The graph would be enormous, and most of that detail is irrelevant to the back-end, which only cares about the *net* relative motion between two keyframes.

- **Problem B - bias re-linearization**: IMUs have slowly drifting gyro/accel biases ($b_g$, $b_a$) that the optimizer refines as part of the state. Every raw sample was integrated using *some* bias estimate. When the optimizer updates that estimate, which happens on essentially every iteration, the integration technically has to be redone with the new bias because the raw samples were corrected using the old one. Redoing a 100+-sample integration loop every time the bias nudges, for every IMU segment in the graph, every optimizer iteration, is not something you can afford.

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

At every raw IMU sample, [`use_numpy/imu_preintegration.py`](../../use_numpy/imu_preintegration.py)'s `PreintegratedIMUBundle.integrate_measurement` folds one micro-step into three running quantities, all expressed relative to the body frame at the *start* of the integration window. The order below matters and matches the code exactly (`imu_preintegration.py:60-62`) - every quantity on the right-hand side is the value from *before* this step, never a value already updated earlier in the same step, which is why $\Delta p$ is updated first (it depends on the old $\Delta v$ and old $\Delta R$), then $\Delta v$ (depends on the old $\Delta R$), and $\Delta R$ last (depends on neither):

$$\Delta p \leftarrow \Delta p + \Delta v\,dt + \tfrac{1}{2}\Delta R\,(\tilde v - b_a)\,dt^2$$

$$\Delta v \leftarrow \Delta v + \Delta R\,(\tilde v - b_a)\,dt$$

$$\Delta R \leftarrow \Delta R \cdot \text{Exp}\big((\tilde\omega - b_g)\,dt\big)$$

where $\tilde\omega, \tilde v$ are the raw sensor readings and $b_g, b_a$ are the bias estimates *in effect when this bundle started*. (An earlier version of this doc listed $\Delta R$ first, $\Delta v$ second, $\Delta p$ last - which reads, taken as literal sequential assignment, as using the *already-updated* $\Delta R$/$`\Delta v`$ a step early, giving each a $1.5\times$ contribution instead of $1\times$ from that step's rotation/velocity increment. The code never does this - it saves `R_prev` before touching `self.delta_R`, and updates `delta_p`/`delta_v` from that saved old value first - and the order above now matches it.) (The script's own comment flags a simplification worth knowing: a real IMU's second channel measures raw *acceleration*, which would need one more integration to reach velocity; this demo integrates a commanded *velocity* channel directly instead, purely to keep the manifold bookkeeping legible. The $\Delta R$ recursion - the part this doc is actually about - is unaffected either way.)

Notice $\Delta R$'s update is exactly the $\text{Exp}$-map composition from [jacobian.md §11.1](../foundations/jacobian.md#111-why-plain-addition-breaks): rotations don't add, so each micro-step's tiny rotation gets *composed onto* the running $\Delta R$, not added to it.

---

## 4. The bias-sensitivity Jacobians

This is the part that makes preintegration more than just "add up the samples." Alongside $\Delta R, \Delta v, \Delta p$, the bundle also propagates five Jacobian matrices - one for how each compressed quantity would shift per unit change in each bias:

$$J_{R,b_g} = \frac{\partial \Delta R}{\partial b_g} \qquad J_{v,b_g} = \frac{\partial \Delta v}{\partial b_g} \qquad J_{v,b_a} = \frac{\partial \Delta v}{\partial b_a} \qquad J_{p,b_g} = \frac{\partial \Delta p}{\partial b_g} \qquad J_{p,b_a} = \frac{\partial \Delta p}{\partial b_a}$$

These are exactly the "sensitivity map" idea from [jacobian.md §3](../foundations/jacobian.md#3-think-of-it-as-a-sensitivity-map) - "if I nudge the input a little, how much does the output move?" - just tracked incrementally, one micro-step at a time, instead of computed once from a closed-form function. Each step updates them using the *current* micro-step's rotation $dR = \text{Exp}((\tilde\omega-b_g)dt)$ and its right Jacobian $J_r = J_r\big((\tilde\omega-b_g)dt\big)$ - and, matching §3's point above, in the same "everything on the right is the old value" order the code uses (`imu_preintegration.py:66-72`): the $p$-Jacobians first (they depend on the old $J_{v,\cdot}$, old $\Delta R$, and old $J_{R,b_g}$), then the $v$-Jacobians (depend on the old $\Delta R$ and old $J_{R,b_g}$), then $J_{R,b_g}$ itself last (depends on neither):

$$J_{p,b_g} \leftarrow J_{p,b_g} + J_{v,b_g}\,dt - \tfrac{1}{2}\Delta R\,[\tilde v-b_a]_\times J_{R,b_g}\,dt^2 \qquad J_{p,b_a} \leftarrow J_{p,b_a} + J_{v,b_a}\,dt - \tfrac{1}{2}\Delta R\,dt^2$$

$$J_{v,b_g} \leftarrow J_{v,b_g} - \Delta R\,[\tilde v-b_a]_\times J_{R,b_g}\,dt \qquad J_{v,b_a} \leftarrow J_{v,b_a} - \Delta R\,dt$$

$$J_{R,b_g} \leftarrow dR^\top J_{R,b_g} - J_r\,dt$$

**Implementation notes.** Both helpers in `lie_utils.py` switch to first-order forms when $\theta = \lVert(\tilde\omega-b_g)dt\rVert < 10^{-6}$: `so3_exp` returns $I + [\phi]_\times$ and `so3_right_jacobian` returns $I - \tfrac{1}{2}[\phi]_\times$. Otherwise they use the closed forms (Rodrigues, and the $J_r$ formula from [jacobian.md §11.4](../foundations/jacobian.md#114-closed-form-for-so3)). The demo also leaves out two things a real preintegrated factor needs. There is no gravity term; a real accelerometer's specific force must have gravity removed, usually in the residual. And there is no preintegrated noise covariance: the script propagates the bias Jacobians but not Forster et al.'s $\Sigma$ recursion, so the bundle carries no uncertainty. Finally, the demo builds its raw readings as the commanded twist plus the *estimated* bias, so $\tilde v - b_a$ and $\tilde\omega - b_g$ equal the commanded values exactly.

The one worth staring at is $J_{R,b_g}$, because it's a direct instance of [jacobian.md §11.4](../foundations/jacobian.md#114-closed-form-for-so3)'s identities: $J_r$ is the right Jacobian of that micro-step's rotation (the extra $dt$ is the chain-rule factor from $`\partial\big((\tilde\omega-b_g)dt\big)/\partial b_g = -dt\,I`$, which also supplies the minus sign), and $dR^\top$ is the frame-transport term (recall $J_l = R J_r$, i.e. left-multiplying by a rotation or its transpose is how a tangent-space sensitivity gets carried from one frame into the next). Every other $J_{\cdot,b_g}$ in this list inherits from $J_{R,b_g}$ - that's why a bias-Jacobian discussion for IMU preintegration is really a right-Jacobian discussion in disguise.

---

## 5. The payoff: correcting for a bias change without re-integrating

Once the bundle has absorbed, say, 100 raw samples, suppose the graph optimizer updates its bias estimate from $b_g \to b_g'$ (a small change $\delta b_g = b_g' - b_g$, since bias only drifts slowly between optimizer iterations - exactly the "small change" regime a Jacobian is built for, per [jacobian.md §8](../foundations/jacobian.md#8-one-subtle-point-jacobian-is-local)). `get_corrected_measurement` applies the first-order Taylor correction directly, no raw samples touched:

$$\Delta R' \approx \Delta R\cdot\text{Exp}\big(J_{R,b_g}\,\delta b_g\big) \qquad \Delta v' \approx \Delta v + J_{v,b_g}\,\delta b_g + J_{v,b_a}\,\delta b_a \qquad \Delta p' \approx \Delta p + J_{p,b_g}\,\delta b_g + J_{p,b_a}\,\delta b_a$$

Notice the asymmetry: $\Delta v'$ and $\Delta p'$ are corrected by plain addition, because velocity and position are ordinary vectors - but $\Delta R'$ has to go back through $\text{Exp}(\cdot)$ and get *composed* onto $\Delta R$, because rotations aren't. That's the same distinction [jacobian.md §11.1](../foundations/jacobian.md#111-why-plain-addition-breaks) opened with, showing up again here as a concrete asymmetry inside one function.

The whole point: this is one matrix-vector multiply per bias component, independent of how many raw samples fed the bundle - $O(1)$ instead of $O(N)$. The script's own demonstration frames it this way: "took less than 1 microsecond to solve, completely bypassing the need to loop through the 100 raw sensor inputs again." That timing is a hard-coded printed string, not a measurement. The script never times anything.

---

## 6. Where this fits in a SLAM back-end

A preintegrated bundle becomes one **edge** in the same sense as [pose_graph_optimization.md §2](pose_graph_optimization.md#2-where-do-the-edges-come-from) - except instead of connecting two pose nodes, it connects two *(pose, velocity, bias)* node sets:

```text
(pose_i, v_i, b_i) ──── ΔR, Δv, Δp ────  (pose_j, v_j, b_j)
                     preintegrated
                      IMU factor
```

A full VIO (Visual-Inertial Odometry)/VI-SLAM back-end (GTSAM's `CombinedImuFactor`, ORB-SLAM3, VINS-Mono, etc.) then builds a residual comparing this corrected bundle against what the two connected nodes' current estimates imply - the same "measured vs. predicted, then `Log`" pattern as [pose_graph_optimization.md §7](pose_graph_optimization.md#7-the-mathematics-is-actually-quite-intuitive). Because the bundle can be bias-corrected in $O(1)$, this residual and its Jacobian can be cheaply re-evaluated at every optimizer iteration without ever revisiting the raw 100+ Hz stream - which is the entire reason preintegration exists.

---

## 7. What the accompanying scripts actually do (and don't)

- **`imu_preintegration.py`** (both [`use_numpy/`](../../use_numpy/imu_preintegration.py) and [`use_manif/`](../../use_manif/imu_preintegration.py)) implements exactly §3-§5 above: compress raw samples into a bundle, track the bias Jacobians, then apply the $O(1)$ correction. The two versions are line-for-line equivalent - `use_manif`'s docstring spells out the exact correspondence (its `rplus()` call returns the same `dR`/right-Jacobian pair the numpy version computes by hand). **Neither script builds the graph residual described in §6** - both stop at demonstrating the compression + correction, which is the specific mechanism this doc covers.
- **`robot_imu_simulation.py`** is a related but different script: it [dead-reckons](factor_graph.md#2-why-do-we-need-it) noisy body-twist readings directly on $SE(3)$ every micro-step (chaining `se3_exp`, no bundle, no bias Jacobians at all). Its "IMU" reports a 6-D velocity twist, so there is no accelerometer, no bias, and no gravity. It then periodically runs a small damped Gauss-Newton correction against a separate GPS-like position fix. It's about *twist dead-reckoning + periodic on-manifold correction*, not preintegration.
- **`imu_integration_comparison.py`** is also related but different: like [pointcloud_pose_tracking_empirical_note.md](../filtering/pointcloud_pose_tracking_empirical_note.md), it's an empirical comparison note rather than a concept explainer - it plots naive Euler-angle (vector-space) integration against proper $SO(3)$ Exp-map integration. The Exp-map estimate still drifts from gyro noise, just far less: with the defaults it ends at about 0.47° rotation error vs. about 70.8° for naive. Each method is scored against its own noise-free ground truth (§7.1). No bias correction involved.

### 7.1 The comparison and correction math, concretely

Both scripts share `lie_utils.py`. The $SO(3)$ exponential `so3_exp` is Rodrigues' formula, with the $I + [\phi]_\times$ fallback below $\theta < 10^{-6}$ noted in §4:

$$\text{Exp}(\phi) = I + \frac{\sin\theta}{\theta}[\phi]_\times + \frac{1-\cos\theta}{\theta^2}[\phi]_\times^2, \qquad \theta = \lVert\phi\rVert$$

**`imu_integration_comparison.py`, measurements** (`run_simulation`). True body rates $\omega$, $v$ come from `utils.true_body_rates(t)`. Both estimators consume the *same* noisy readings each step:

$$\tilde\omega = \omega + n_\omega,\quad n_\omega \sim \mathcal N(0, \sigma_g^2 I), \qquad \tilde v = v + n_v,\quad n_v \sim \mathcal N(0, \sigma_v^2 I)$$

Defaults: $\sigma_g = 0.02$ rad/s, $\sigma_v = 0.05$ m/s, $dt = 0.005$ s, 20 s, seed 0.

**Naive update.** Attitude is kept as a ZYX Euler vector $\theta_{rpy}$ and rebuilt with `euler_to_R`, which returns $R_z(\text{yaw}) R_y(\text{pitch}) R_x(\text{roll})$:

$$`\theta_{rpy} \leftarrow \theta_{rpy} + \tilde\omega\,dt, \qquad R \leftarrow \texttt{euler\_to\_R}(\theta_{rpy}), \qquad p \leftarrow p + R\,\tilde v\,dt`$$

**Exp-map update.**

$$R \leftarrow R\,\text{Exp}(\tilde\omega\,dt), \qquad p \leftarrow p + R\,\tilde v\,dt$$

In both, the position step uses the rotation *after* this step's update. That is the opposite of the preintegration order in §3, which uses the old $\Delta R$.

**Ground truths.** There are two, one per estimator, each built from the noise-free rates with that estimator's own composition rule: $`R^{gt}_{\text{naive}} \leftarrow R^{gt}_{\text{naive}}\,\texttt{euler\_to\_R}(\omega\,dt)`$ and $`R^{gt}_{\text{exp}} \leftarrow R^{gt}_{\text{exp}}\,\text{Exp}(\omega\,dt)`$, with positions integrated as above. Rotation error is the geodesic angle (`rotation_geodesic_error`), plotted in degrees; position error is the Euclidean norm:

$$e_R = \arccos\!\left(\frac{\mathrm{tr}(R_{gt}^\top R_{est}) - 1}{2}\right),\qquad e_p = \lVert p_{gt} - p_{est}\rVert$$

**`robot_imu_simulation.py`, propagation** (`run_simulation`). The true twist $\xi = [v, \omega]$ (translation first) is drawn once, uniformly per axis from $\pm 1.0$ m/s and $\pm 0.5$ rad/s. Each IMU step ($dt = 0.01$ s, 100 steps per second):

$$T_{true} \leftarrow T_{true}\,\text{Exp}(\xi\,dt), \qquad T_{est} \leftarrow T_{est}\,\text{Exp}\big((\xi + n)\,dt\big),\quad n \sim \mathcal{N}(0, 0.02^2 I_6)$$

The 0.02 noise is hard-coded. `se3_exp` builds $\text{Exp}(\xi)$ with rotation $\text{Exp}(\omega)$ and translation $V v$, where

$$V = I + \frac{1-\cos\theta}{\theta^2}[\omega]_\times + \frac{\theta-\sin\theta}{\theta^3}[\omega]_\times^2$$

falling back to $`I + [\omega]_\times$ and $V = I + \tfrac{1}{2}[\omega]_\times`$ below $\theta < 10^{-6}$.

**Position fix.** Once per second: $z = t_{true} + n_z$ with $n_z \sim \mathcal N(0, \sigma_{pos}^2 I)$, $\sigma_{pos} = 0.05$ m by default. It measures translation only, never orientation.

**Residual and Jacobian** (`position_observation_jacobian`). For a right perturbation $T_{est} \text{Exp}(\delta\xi)$:

$$r = z - t_{est}, \qquad J = \frac{\partial t}{\partial \delta\xi} = \begin{bmatrix} R_{est} & 0_{3\times 3} \end{bmatrix}$$

**Solve and retract.** With $\Omega = I/\sigma_{pos}^2$ (built in `__main__` as `pos_info`):

$$\left(J^\top \Omega J + 10^{-4} I_6\right)\delta\xi = J^\top \Omega\, r, \qquad T_{est} \leftarrow T_{est}\,\text{Exp}(\delta\xi)$$

The loop stops when $\lVert\delta\xi\rVert$ drops below `gn_tol` ($10^{-6}$) or after `gn_max_iters` (10) iterations. With the defaults, it converges in 2 iterations each second.

The $10^{-4} I$ damping is required, not cosmetic. $J$'s angular block is zero, so $J^\top \Omega J$ has rank 3 and plain Gauss-Newton's normal matrix would be singular. The damping makes it solvable and leaves the rotational part of $\delta\xi$ at exactly zero. Orientation is therefore never corrected; only the IMU dead-reckoning sets it. Because $\Omega = 400 I$ dwarfs the damping, each solve moves $t_{est}$ essentially all the way onto the noisy $z$. So the post-correction error is about the size of the fix noise, and can be larger than the pre-correction error (seed 0, second 1: 0.0050 m before, 0.0785 m after).

---

## 8. One-sentence summary

> **Preintegration turns "re-run 100+ raw IMU samples every time the bias estimate changes" into "one matrix-vector multiply," by tracking - while integrating - exactly how sensitive the compressed relative-motion bundle is to the bias, using the same right-Jacobian machinery that converts a tangent-space nudge into a body-frame rotation anywhere else in this codebase.**

---

## 9. References

1. Forster, C., Carlone, L., Dellaert, F., & Scaramuzza, D. (2017). *On-Manifold Preintegration for Real-Time Visual-Inertial Odometry*. IEEE Transactions on Robotics, 33(1), 1-21. https://doi.org/10.1109/TRO.2016.2597321 - the on-manifold, right-Jacobian bias-correction preintegration formulation this whole doc walks through (§3-§5).
