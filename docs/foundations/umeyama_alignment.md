# Umeyama Alignment

**Umeyama alignment** answers a narrow but very practical question that shows up every time you evaluate a reconstruction:

> Given two point sets that are supposed to describe the same shape but live in different, unaligned coordinate frames, what's the best rigid-plus-scale transform that overlays one onto the other?

It's not a SLAM algorithm by itself - it's the evaluation tool you reach for *after* running one, whenever the algorithm's own output is only defined up to an ambiguity that has to be removed before comparing against ground truth.

---

## 1. Why an alignment step is needed at all

A monocular camera looking at a static scene can recover the *shape* of the scene and the *relative* motion of the cameras, but it cannot recover:

- **absolute position/orientation**: the whole reconstruction could be picked up and rigidly moved anywhere, and every reprojection error would stay identical, and
- **absolute scale**: shrinking the entire scene and every camera-to-point distance by the same factor, while doubling nothing else, leaves every projected pixel exactly where it was.

Together that's a **7-parameter similarity ambiguity**: 3 translation + 3 rotation + 1 scale. This is the same **gauge freedom** idea as in [pose_graph_optimization.md](../optimization/pose_graph_optimization.md)'s "one subtlety this formula hides" note - a direction the optimizer's cost function is completely blind to - except pose graphs only have the 6-DoF rigid version (their edges are *relative rigid* constraints, so scale is never in question), while monocular bundle adjustment's edges are *projective*, so scale is unobservable too.

You can't compute a meaningful "position error in meters" against ground truth while this ambiguity is still there - the reconstruction and the ground truth are simply expressed in two different (and differently-scaled) coordinate systems. Umeyama alignment is how you solve for the one similarity transform that brings them into the same frame before measuring error.

---

## 2. What it actually computes

Given $n$ corresponding point pairs $\{(x_i, y_i)\}$ - here, $x_i$ from the estimated reconstruction and $y_i$ the ground truth - Umeyama's method (1991) finds the scale $s \in \mathbb{R}$, rotation $R \in SO(3)$, and translation $t \in \mathbb{R}^3$ that solve:

$$\boxed{\min_{s,R,t} \sum_{i=1}^{n} \left\| s R x_i + t - y_i \right\|^2}$$

i.e. the least-squares best-fit similarity transform mapping the estimated points onto the ground-truth points. Unlike a generic nonlinear least-squares problem, this one has a **closed-form solution** via SVD - no Gauss-Newton iteration needed.

---

## 3. The closed-form solution

This repo's implementation, [`umeyama_alignment`](../../utils.py) in `utils.py`, is the textbook closed-form solution end to end:

1. **Center both point sets** on their own centroids:
   $$\mu_{\text{est}} = \frac{1}{n}\sum x_i \qquad \mu_{\text{true}} = \frac{1}{n}\sum y_i \qquad X = x_i - \mu_{\text{est}} \qquad Y = y_i - \mu_{\text{true}}$$
   Centering removes translation from the problem - it's handled separately, in step 4.

2. **Cross-covariance and its SVD:**
   $$\Sigma = \frac{1}{n} Y^\top X = U D V^\top$$

3. **Rotation, with a reflection guard:**
   $$R = U S V^\top, \qquad S = \begin{cases}I & \det(U)\det(V^\top) \ge 0\\ \text{diag}(1,1,-1) & \det(U)\det(V^\top) < 0\end{cases}$$
   Plain $UV^\top$ is the best-fit *orthogonal* matrix, but "orthogonal" includes reflections ($\det = -1$) as well as rotations ($\det = +1$). Since $R$ must be an actual rotation, $S$ flips the sign of the smallest-variance axis whenever the unconstrained best fit would have been a reflection - see the worked example in §4 for why this matters and what it costs.

4. **Scale and translation:**
   $$s = \frac{\text{tr}(DS)}{\text{var}(X)}, \qquad \text{var}(X) = \frac{1}{n}\sum \|X_i\|^2, \qquad t = \mu_{\text{true}} - s R \mu_{\text{est}}$$

That's exactly the four steps `umeyama_alignment` runs, in order.

---

## 4. Why the reflection guard matters

Without step 3's correction, the fit can silently return a **mirror image** instead of a rotation whenever the point geometry allows it. Concretely, take a small tetrahedron and its true mirror image across one plane (a genuine reflection, $\det = -1$ relative to the original - not something *any* rotation can reproduce exactly):

- **Uncorrected** ($R = UV^\top$): fits **perfectly** (residual $\approx 0$), but $\det(R) \approx -1$ - not a valid rotation, and physically meaningless as a camera/robot pose.
- **Corrected** ($R = USV^\top$): $\det(R) = +1$, a valid rotation, but can now only *approximate* the mirrored points - in this example the best achievable fit has a max residual of about $0.44$ (scale also drops from the uncorrected $1.0$ to $s \approx 0.78$ to partially compensate).

So the guard is a deliberate trade: it always returns a physically valid rotation, at the cost of no longer being able to claim a perfect fit on point sets that are actually mirror-related. For the well-conditioned, non-degenerate point sets this repo's scripts generate (cameras spread around a 3D landmark cluster), the reflection case essentially never triggers in practice - it matters most for near-coplanar or otherwise degenerate configurations.

---

## 5. A worked example: recovering a known transform exactly

Take four non-coplanar points and a known similarity transform - scale $s=2$, a $90°$ yaw about $z$, translation $t=(1,2,3)$:

$$X = \begin{bmatrix}0&0&0\\1&0&0\\0&1&0\\0&0&1\end{bmatrix}, \qquad R_{\text{true}} = \begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix}, \qquad Y = s\,(R_{\text{true}} X^\top)^\top + t$$

Running `umeyama_alignment(X, Y)` recovers $\hat{s} = 2.0$, $\hat{R} = R_{\text{true}}$, and $\hat{t} = (1, 2, 3)$ back out - to floating-point precision ($< 10^{-15}$ max error), since 4 well-spread non-coplanar points exactly determine a similarity transform with no noise to average out. With real (noisy) data from more than 4 points, the same four steps instead return the *least-squares best* $s, R, t$, exactly like fitting a line through noisy points.

---

## 6. Where this is (and isn't) used in this repo

- **[`bundle_adjustment.py`](../../use_numpy/bundle_adjustment.py)** (both `use_numpy/` and `use_manif/`) calls `umeyama_alignment` on the *camera positions* after solving, then applies the recovered $(s, R, t)$ to **both** the camera poses and the landmark positions before computing `pose_errors`/`landmark_errors` against ground truth. This is standard practice for evaluating monocular BA/SfM output - see [bundle_adjustment.md](../optimization/bundle_adjustment.md) for how the alignment step fits into the rest of that script. Note that reprojection error itself is computed *before* alignment and is unaffected by it - only the absolute pose/landmark error numbers depend on this step.
- **[`bundle_adjustment_advanced.py`](../../use_numpy/bundle_adjustment_advanced.py)** deliberately does **not** call it. That script hard-fixes two keyframes as a gauge anchor instead of using a soft gauge-prior factor; once $\ge 1$ keyframe is hard-fixed, there is no residual rigid *or* scale freedom left in the solution to align away, so the alignment step this doc describes simply doesn't apply there.

---

## 7. One-sentence intuition

> **Umeyama alignment is least-squares curve-fitting applied to whole point clouds instead of scalar points - it finds the one scale+rotation+translation that best overlays an estimate onto ground truth, which is exactly the piece missing before "position error in meters" against monocular reconstruction output means anything.**
