# Linear vs. nonlinear systems: telling them apart, and when the nonlinearity matters

You can tell by looking at the model equations, not the data. Write the two models the filter uses:

```math
x_{k+1} = f(x_k, u_k) + w_k \qquad z_k = h(x_k) + v_k
```

The system is linear only if both $f$ and $h$ are linear in the state $x$, meaning they can be written as $f = F x + G u$ and $h = H x$, where $F$, $G$, $H$ don't depend on $x$. If either one isn't, the system is nonlinear, and the plain KF is no longer the exact answer.

## 1. Three ways to tell

1. **Superposition.** $f(a x_1 + b x_2) = a f(x_1) + b f(x_2)$ must hold for every $x_1, x_2, a, b$. A constant offset (making $f$ affine) fails this test but is still fine for a KF, since it just adds to the mean.
2. **The Jacobian is constant.** Differentiate: if $\partial f / \partial x$ and $\partial h / \partial x$ still contain $x$, the system is nonlinear. This is the most practical test. It's also why the EKF has to recompute $F_k$ and $H_k$ at every step ([kf_ekf_iekf.md §2](kf_ekf_iekf.md#2-extended-kalman-filter-the-world-is-nonlinear-so-ill-approximate-it-locally)): they change as the estimate changes.
3. **Look for these patterns.** Any of them makes a model nonlinear:
   - **Trig of a state**: $\cos\theta$, $\sin\theta$ where $\theta$ is estimated. This is the unicycle in [kf_ekf_iekf.md §2](kf_ekf_iekf.md#2-extended-kalman-filter-the-world-is-nonlinear-so-ill-approximate-it-locally).
   - **States multiplied together**: $v \cdot \cos\theta$, or $`R\,p`$ when both $R$ and $p$ are estimated.
   - **Rotations or poses in the state**: $R^\top(p - t)$ in point-cloud tracking, or composing $T \cdot \mathrm{Exp}(\xi)$.
   - **Division by a state**: camera projection $u = f_x X/Z$.
   - **Norms and square roots**: range $\sqrt{(x-x_L)^2 + (y-y_L)^2}$.
   - **Switching or contact**: impacts, or stick/slip friction. These are non-smooth, which is what the saltation-matrix EKF in [hybrid_saltation_ekf.md](hybrid_saltation_ekf.md) handles.

The first two tests are exact. The third is a quick heuristic for spotting what will fail them.

## 2. Common points of confusion

- **Linearity is about the state, not time or inputs.** $F_k$ may change with $k$, or depend on a known control or measured signal, and the model is still linear (linear time-varying), so the KF is still exact. Example: $`x_{k+1} = x_k + v\cos\theta\,\Delta t`$ is linear if $\theta$ comes from a trusted compass as an input, and nonlinear if $\theta$ is part of the state you're estimating.
- **Only one side needs to be nonlinear.** Linear motion with a range sensor (a nonlinear $h$) is already a nonlinear filtering problem.
- **Linearity depends on your choice of state coordinates.** This is the IEKF's point ([kf_ekf_iekf.md §5](kf_ekf_iekf.md#5-the-really-important-difference-how-do-you-define-error)): pose dynamics are nonlinear in $[R, t]$, but for *group-affine* dynamics a well-chosen invariant error on the group evolves independently of the state estimate (Barrau & Bonnabel 2017, [kf_ekf_iekf.md §11](kf_ekf_iekf.md#11-references)). The vanilla-KF result in [pointcloud_pose_tracking_empirical_note.md §4](pointcloud_pose_tracking_empirical_note.md#4-vanilla-kf-vs-ekfiekfukf-diverges-by-construction-not-just-approximation) shows what forcing a pose into a flat vector state costs.

## 3. "How nonlinear" matters in practice

Almost every real robot model is nonlinear. The useful question is whether it's nonlinear *over the region your uncertainty covers*. A quick numerical check: take a step $\delta$ about the size of your standard deviation and measure the linearization error

```math
\lVert f(\hat x + \delta) - f(\hat x) - J\delta \rVert
```

Compare that to the process or measurement noise.

- **Much smaller than the noise**: the EKF is fine.
- **Comparable to the noise or larger**: the Jacobian changes a lot within your uncertainty (for example, a large heading uncertainty, or a landmark close to the camera). Consider the UKF, an iterated update, or an invariant or manifold formulation.

The same question comes up in optimization, where every [Gauss-Newton](../optimization/gauss_newton.md) step trusts a linearization up to the size of that step instead of up to your uncertainty. There, it's answered after the fact, by checking whether the real cost actually went down: that's [Levenberg-Marquardt's accept/reject test](../optimization/levenberg_marquardt.md#7-how-does-lm-decide-whether-to-be-cautious). The rest of this page is about the filter's version of the question.

## 4. How do you know whether it's nonlinear over your uncertainty region?

Compare the part the linearization leaves out with the noise the filter already expects. If the leftover is small next to the noise, the linear approximation is good over the region your uncertainty covers. There are four ways to check this, from cheapest to most thorough.

### 4.1 Analytic: the second-order term

Expand the model around the estimate:

```math
f(\hat x + \delta) = f(\hat x) + J\delta + \tfrac12\,\delta^\top \nabla^2 f\,\delta + \dots
```

The EKF keeps only the first two terms. With $\delta \sim \mathcal N(0, P)$, what it drops is, per output component $i$:

- a mean bias of $`\tfrac12\,\mathrm{tr}(\nabla^2 f_i\, P)`$;
- an extra covariance of $`\tfrac12\,\mathrm{tr}(\nabla^2 f_i\, P\, \nabla^2 f_j\, P)`$ in entry $(i, j)$.

Compare these with the noise the model already carries: $R$ for $h$, $Q$ for $f$.

- **Small**: if the bias is much smaller than $`\sqrt{R_{ii}}`$ (the sensor's own standard deviation), the linearization is adequate.
- **Comparable or larger**: the model is effectively nonlinear at this uncertainty.

**Range sensor example (2D)**: the Hessian of $`h = \lVert p - L\rVert`$ is $`(I - uu^\top)/r`$, where $u$ is the unit line-of-sight vector. It has curvature $1/r$ across the line of sight, so the bias is about $\sigma_p^2 / (2r)$ (in 3D there are two perpendicular directions, which gives $\sigma_p^2 / r$). With $\sigma_p = 1$ m and $\sigma_{\text{range}} = 0.1$ m:

| Range $r$ | Bias | vs. 0.1 m noise |
| --- | --- | --- |
| 2 m | 0.25 m | 2.5× the noise → nonlinear |
| 50 m | 0.01 m | 0.1× the noise → effectively linear |

The model is the same in both rows. Only the ratio of uncertainty to curvature changed.

Some quick scalings, each giving the dropped second-order term relative to the function's own value:

| Model | Second-order term ÷ value | Example |
| --- | --- | --- |
| Heading, $\cos\theta$ | $\sim \sigma_\theta^2/2$ | 0.1 rad → 0.5%; 0.5 rad → 12% |
| Camera, $X/Z$ | $\sim (\sigma_Z / Z)^2$ | depth uncertainty comparable to depth → trouble |
| Range (2D), $r$ | $\sim \tfrac12 (\sigma_p / r)^2$ | position uncertainty comparable to range → trouble |

### 4.2 Numerical: sample and compare (works for any model)

Draw samples from your current belief, push them through the real function, and look at the residual that the linear model $`h(\hat x) + J(x - \hat x)`$ misses. Measuring the residual directly, rather than comparing sample means and covariances, cancels the first-order sampling noise, so a few thousand samples resolve even a small bias. For the range example above:

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

Evaluate $J$ at $\hat x \pm \sigma$ along the main axes of $P$ (the eigenvectors, scaled by $\sqrt{\lambda}$). If $`\lVert J(\hat x + \sigma e) - J(\hat x)\rVert / \lVert J(\hat x)\rVert`$ is more than a few percent, the linear model varies a lot within your uncertainty.

### 4.4 After the fact: consistency tests

Run the filter and check whether its reported uncertainty is believable:

- **NIS**, the normalized innovation squared, $\nu^\top S^{-1}\nu$, needs no ground truth.
- **NEES**, the normalized estimation error squared, needs simulated ground truth. This repo reports a Monte Carlo NEES in [`saltation_matrix_ekf.py`](../../use_numpy/saltation_matrix_ekf.py), [`inchworm_zupt_ekf.py`](../../use_numpy/inchworm_zupt_ekf.py) and [`friction_anisotropic_ekf.py`](../../use_numpy/friction_anisotropic_ekf.py).

Linearization error that matters shows up as:

- NIS or NEES above the chi-square bounds (the filter is overconfident);
- a nonzero mean in the innovations.

This test tells you *that* the approximation failed, not why.

### Where it matters most

$P$ changes over time, so the answer does too. Check at the worst moments:

- right after initialization, when $P$ is large;
- after long dead-reckoning;
- just before a loop closure;
- when landmarks are close to the camera.

A filter can be effectively linear in steady state and badly nonlinear at startup. That's why iterated updates, the UKF, and invariant formulations tend to earn their keep in those moments.
