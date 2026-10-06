# Linear vs. nonlinear systems: telling them apart, and when the nonlinearity matters

We can tell by looking at the model equations, not the data. Write the two models the filter uses:

```math
x_{k+1} = f(x_k, u_k) + w_k \qquad z_k = h(x_k) + v_k
```

The system is linear only if both $f$ and $h$ are linear in the state $x$, meaning they can be written as $f = F x + G u$ and $h = H x$, where $F$, $G$, $H$ don't depend on $x$. If either one isn't, the system is nonlinear, and the plain KF is no longer the exact answer.

## 1. Three ways to tell

1. **Superposition.** $f(a x_1 + b x_2) = a f(x_1) + b f(x_2)$ must hold for every $x_1, x_2, a, b$. A constant offset (making $f$ affine) fails this test but is still fine for a KF, since it just adds to the mean.
2. **The Jacobian doesn't depend on $x$.** Differentiate: if $\partial f / \partial x$ or $\partial h / \partial x$ still contains $x$, the system is nonlinear. This is the most practical test, and it is why the EKF has to recompute $F_k$ and $H_k$ at every step ([kf_ekf_iekf.md §2](kf_ekf_iekf.md#2-extended-kalman-filter-the-world-is-nonlinear-so-ill-approximate-it-locally)): they change as the estimate changes.
3. **Look for these patterns.** Any of them makes a model nonlinear:
   - **Trig of a state**: $\cos\theta$, $\sin\theta$ where $\theta$ is estimated. This is the unicycle in [kf_ekf_iekf.md §2](kf_ekf_iekf.md#2-extended-kalman-filter-the-world-is-nonlinear-so-ill-approximate-it-locally).
   - **States multiplied together**: $v \cdot \cos\theta$, or $`R\,p`$ when both $R$ and $p$ are estimated.
   - **Rotations or poses in minimal coordinates**: composing $T \cdot \mathrm{Exp}(\xi)$, or a pose parameterized by angles. The point-cloud observation $R\,p_i + t$ is the counter-example: it is exactly linear in the state $[\mathrm{vec}(R), t]$ ([pointcloud_pose_tracking_empirical_note.md §4.1](pointcloud_pose_tracking_empirical_note.md#41-what-vanilla-buys-and-costs)). The nonlinearity enters through the rotation constraint and the minimal error coordinates.
   - **Division by a state**: camera projection $u = f_x X/Z$.
   - **Norms and square roots**: range $\sqrt{(x-x_L)^2 + (y-y_L)^2}$.
   - **Switching or contact**: impacts, or stick/slip friction. These are non-smooth, which is what the saltation-matrix EKF in [hybrid_saltation_ekf.md](hybrid_saltation_ekf.md) handles.

The first two tests are exact. The third is a heuristic for spotting what will fail them.

## 2. Common points of confusion

- **Linearity is about the state, not time or inputs.** $F_k$ may change with $k$, or depend on a known control or measured signal, and the model is still linear (linear time-varying), so the KF is still exact. Example: $`x_{k+1} = x_k + v\cos\theta\,\Delta t`$ is linear if $\theta$ comes from a trusted compass as an input, and nonlinear if $\theta$ is part of the state we estimate.
- **Only one side needs to be nonlinear.** Linear motion with a range sensor (a nonlinear $h$) is already a nonlinear filtering problem.
- **Linearity depends on the choice of state coordinates.** This is the IEKF's point ([kf_ekf_iekf.md §5](kf_ekf_iekf.md#5-the-really-important-difference-how-do-you-define-error)). Pose dynamics are nonlinear when:
  - the state is a minimal parameterization (angles, tangent-space error) or must satisfy the $SO(3)$ constraint;
  - the angular velocity or a bias is itself a state, so it multiplies $R$.

  With a known body twist, $R_{k+1} = R_k\,\mathrm{Exp}(\omega\,\Delta t)$ is linear in $\mathrm{vec}(R_k)$. The vanilla KF in [pointcloud_pose_tracking_empirical_note.md §4](pointcloud_pose_tracking_empirical_note.md#4-vanilla-kf-vs-ekfiekfukf-diverges-by-construction-not-just-approximation) is inexact only because it truncates the exponential and needs an SVD re-projection onto $SO(3)$. For *group-affine* dynamics, a well-chosen invariant error evolves independently of the state estimate (Barrau & Bonnabel 2017, [kf_ekf_iekf.md §11](kf_ekf_iekf.md#11-references)).

## 3. "How nonlinear" matters in practice

Almost every real robot model is nonlinear. The useful question is whether it's nonlinear *over the region our uncertainty covers*. A quick numerical check: take a step $\delta$ about the size of our standard deviation and measure the linearization error

```math
\lVert f(\hat x + \delta) - f(\hat x) - J\delta \rVert
```

Compare that to the process or measurement noise.

- **Much smaller than the noise**: the EKF is fine.
- **Comparable to the noise or larger**: the Jacobian changes a lot within our uncertainty (for example, a large heading uncertainty, or a landmark close to the camera). Consider the UKF, an iterated update, or an invariant or manifold formulation.

$P$ changes over time, so the answer does too. Check at the worst moments:

- right after initialization, when $P$ is large;
- after long dead-reckoning;
- just before a loop closure;
- when landmarks are close to the camera.

A filter can be effectively linear in steady state and badly nonlinear at startup, which is where iterated updates, the UKF and invariant formulations earn their keep.

The same question comes up in optimization, where every [Gauss-Newton](../optimization/gauss_newton.md) step trusts a linearization up to the size of that step instead of up to our uncertainty. There it is answered after the fact, by checking whether the real cost went down: that's [Levenberg-Marquardt's accept/reject test](../optimization/levenberg_marquardt.md#6-how-does-lm-decide-whether-to-be-cautious). The rest of this page is the filter's version of the question.

## 4. How do you know whether it's nonlinear over your uncertainty region?

The test is always the same: compare the part the linearization leaves out with the noise the filter already expects. Small leftover means a good linear approximation over our uncertainty region. Four ways to check, from cheapest to most thorough:

### 4.1 Analytic: the second-order term

Expand the model around the estimate:

```math
f(\hat x + \delta) = f(\hat x) + J\delta + \tfrac12\,\delta^\top \nabla^2 f\,\delta + \dots
```

The EKF keeps only the first two terms. With $\delta \sim \mathcal N(0, P)$, what it drops is, per output component $i$:

- a mean bias of $`\tfrac12\,\mathrm{tr}(\nabla^2 f_i\, P)`$;
- an extra covariance of $`\tfrac12\,\mathrm{tr}(\nabla^2 f_i\, P\, \nabla^2 f_j\, P)`$ in entry $(i, j)$.

Compare both with the noise the model already carries: $R$ for $h$, $Q$ for $f$.

- **Small**: the bias and the extra spread $`\sqrt{\tfrac12\mathrm{tr}(\nabla^2 f_i\,P\,\nabla^2 f_i\,P)}`$ are both much smaller than $`\sqrt{R_{ii}}`$ (the sensor's own standard deviation), so the linearization is adequate.
- **Comparable or larger**, in either: the model is effectively nonlinear at this uncertainty.

**Range sensor example (2D)**: the Hessian of $`h = \lVert p - L\rVert`$ is $`(I - uu^\top)/r`$, where $u$ is the unit line-of-sight vector. It has curvature $1/r$ across the line of sight, so the bias is about $\sigma_p^2 / (2r)$ (in 3D there are two perpendicular directions, which gives $\sigma_p^2 / r$). With $\sigma_p = 1$ m and $\sigma_{\text{range}} = 0.1$ m:

| Range $r$ | Bias | Extra spread $\sqrt{\tfrac12 / r^2}$ | vs. 0.1 m noise |
| --- | --- | --- | --- |
| 2 m | 0.25 m | 0.354 m | 2.5× (bias), 3.5× (spread) → nonlinear |
| 50 m | 0.01 m | 0.014 m | 0.1× and 0.14× → effectively linear |

The spread matters even more than the bias here. The model is the same in both rows. Only the ratio of uncertainty to curvature changed.

Some quick scalings, each giving the dropped second-order term relative to the function's value (for $\cos\theta$, the value at $\hat\theta = 0$, since the ratio blows up where $\cos\hat\theta \approx 0$):

| Model | Second-order term ÷ value | Example |
| --- | --- | --- |
| Heading, $\cos\theta$ | $\sim \sigma_\theta^2/2$ | 0.1 rad → 0.5%; 0.5 rad → 12.5% |
| Camera, $X/Z$ | $\sim (\sigma_Z / Z)^2$ | depth uncertainty comparable to depth → trouble |
| Range (2D), $r$ | $\sim \tfrac12 (\sigma_p / r)^2$ | position uncertainty comparable to range → trouble |

**Tiny example (why the bias is one-sided):**
- Toy case: $\hat\theta = 0$, $\theta \sim \mathcal N(0, 0.5^2)$, so $\sigma_\theta = 0.5$ rad.
- $\cos\theta$ has a flat top at $0$: an error of $+0.5$ and an error of $-0.5$ both push it below $1$.
- The errors do not cancel. Exactly, $`E[\cos\theta] = e^{-\sigma_\theta^2/2} = 0.8825`$, about 11.8% below $1$.
- This matches the $\sigma_\theta^2/2 = 12.5\%$ in the table, which is the first-order estimate of the same bias.
- A linearization at $\hat\theta = 0$ predicts exactly $1$, so it misses this shift.

### 4.2 Numerical: sample and compare (works for any model)

Draw samples from the current belief, push them through the real function, and look at the residual that the linear model $`h(\hat x) + J(x - \hat x)`$ misses. Measuring the residual directly, rather than comparing sample means and covariances, cancels the first-order sampling noise, so a few thousand samples resolve even a small bias. For the range example above:

```python
import numpy as np

rng = np.random.default_rng(0)
L = np.array([0.0, 0.0])                                # landmark
h = lambda x: np.array([np.linalg.norm(x - L)])         # range measurement
x_hat = np.array([2.0, 0.0])                            # estimate, 2 m from L
P = np.eye(2) * 1.0**2                                  # sigma_p = 1 m
R = np.array([[0.1**2]])                                # sigma_range = 0.1 m
J = ((x_hat - L) / np.linalg.norm(x_hat - L))[None, :]  # dh/dx at x_hat

X = rng.multivariate_normal(x_hat, P, 5000)
res = np.array([h(x) - h(x_hat) - J @ (x - x_hat) for x in X])  # what linearization drops
print(np.abs(res.mean(0)) / np.sqrt(np.diag(R)),  # mean bias, in sensor sigmas
      res.std(0) / np.sqrt(np.diag(R)))          # extra spread, in sensor sigmas
```

This prints a bias of about 2.7σ and an extra spread of about 3.9σ at 2 m: clearly nonlinear. The bias is a little above §4.1's 0.25 m estimate because terms beyond second order still matter when $\sigma_p / r = 0.5$. With `x_hat = [50, 0]` it prints about 0.1σ and 0.14σ: effectively linear.

For a cheaper version, use 2n+1 sigma points instead of 5000 samples. If the UKF's predicted mean and covariance differ noticeably from the EKF's, nonlinearity matters.

### 4.3 Jacobian drift

Evaluate $J$ at $\hat x \pm \sigma$ along the main axes of $P$ (the eigenvectors, scaled by $\sqrt{\lambda}$). If $`\lVert J(\hat x + \sigma e) - J(\hat x)\rVert / \lVert J(\hat x)\rVert`$ is more than a few percent, the linear model varies a lot within our uncertainty.

### 4.4 After the fact: consistency tests

Run the filter and check whether its reported uncertainty is believable:

- **NIS**, the normalized innovation squared, $\nu^\top S^{-1}\nu$, needs no ground truth.
- **NEES**, the normalized estimation error squared, needs simulated ground truth. This repo reports a Monte Carlo NEES in three scripts, which test different things:
  - [`friction_anisotropic_ekf.py`](../../use_numpy/friction_anisotropic_ekf.py) ($n = 3$): a genuinely nonlinear motion model (heading enters through $\cos\theta$, $\sin\theta$);
  - [`inchworm_zupt_ekf.py`](../../use_numpy/inchworm_zupt_ekf.py) ($n = 2$): exactly linear;
  - [`saltation_matrix_ekf.py`](../../use_numpy/saltation_matrix_ekf.py) ($n = 6$): linear between bounces, with the nonlinearity concentrated in the bounce itself.

**Intuition for NEES:**
- Divide each error by its predicted spread. A consistent filter gives values around $\pm 1$.
- Square them and sum over the $n$ states: the total is about $n$.
- Well above $n$ means the filter is overconfident.

A consistent filter's NEES averages the state dimension $n$ (NIS: the measurement dimension). Which bound to compare against depends on what we look at:

- **One run**: a single NEES value is chi-square with $n$ degrees of freedom, so its 95% upper bound is wide (7.81 for $n = 3$).
- **An average over $N$ Monte Carlo runs**: $N$ times the average is chi-square with $nN$ degrees of freedom, so the interval is much tighter, about $`n \pm 1.96\sqrt{2n/N}`$. The scripts compute it with the Wilson-Hilferty approximation (`averaged_nees_bounds(n, N)`). For $N = 500$:

| Script | $n$ | 95% band |
| --- | --- | --- |
| `inchworm_zupt_ekf.py` | 2 | [1.83, 2.18] |
| `friction_anisotropic_ekf.py` | 3 | [2.79, 3.22] |
| `saltation_matrix_ekf.py` | 6 | [5.70, 6.31] |

Above the upper bound, the filter is overconfident. Below the lower bound, it's underconfident, which is what `saltation_matrix_ekf.py` shows after a bounce (NEES about 2.4 against 6). A nonzero mean in the innovations is a second warning sign.

These tests catch **any** mismatch between the filter's model and reality, not just linearization error:
- A wrong $Q$ or $R$, or dynamics the model leaves out, fail them just as well.
- `inchworm_zupt_ekf.py` is a clean example. Its model is exactly linear, yet its NEES is 6-16 against 2, mostly from a ramp acceleration the model doesn't include ([inchworm_zupt_ekf.md §3](inchworm_zupt_ekf.md#3-the-finding-not-just-always-is-wrong-while-moving)).
- So a failed test says *that* something is wrong, not what. To pin it on the linearization, use the direct checks in §4.1-§4.3, or remove the suspected cause and run again.
