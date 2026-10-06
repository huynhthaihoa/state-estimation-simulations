# Umeyama Alignment

**Umeyama alignment** answers a narrow but very practical question that shows up every time we evaluate a reconstruction:

> Given two point sets that are supposed to describe the same shape but live in different, unaligned coordinate frames, what's the best rigid-plus-scale transform that overlays one onto the other?

It isn't a SLAM algorithm by itself. We focus on its most common use: the evaluation tool we reach for *after* running one, whenever the output is only defined up to an ambiguity that must be removed before comparing against ground truth.

The same closed-form fit also runs *inside* SLAM pipelines:

- point-to-point ICP uses it as the per-iteration alignment step (scale fixed to 1)
- monocular systems use the similarity version to align two maps at a loop closure

---

## 1. Why an alignment step is needed at all

A monocular camera looking at a static scene can recover the *shape* of the scene and the *relative* motion of the cameras, but it cannot recover:

- **absolute position/orientation**: the whole reconstruction could be picked up and rigidly moved anywhere, and every reprojection error would stay identical
- **absolute scale**: shrinking the entire scene and every camera-to-point distance by the same factor, while changing nothing else, leaves every projected pixel where it was

Together, that's a **7-parameter similarity ambiguity**: 3 translation + 3 rotation + 1 scale. This is the same **gauge freedom** idea as in [pose_graph_optimization.md](../optimization/pose_graph_optimization.md)'s "one subtlety this formula hides" note - a direction the optimizer's cost function is completely blind to - except pose graphs only have the 6-DoF rigid version (their edges are *relative rigid* constraints, so scale is never in question), while monocular bundle adjustment's edges are *projective*, so scale is unobservable too.

We can't compute a meaningful "position error in meters" against ground truth while this ambiguity remains: the two are expressed in different (and differently scaled) coordinate systems. Umeyama alignment solves for the one similarity transform that brings them into the same frame before we measure error.

### 1.1 Align only what's actually unobservable

The 7-DoF similarity transform is right for monocular output, but not for every system. The rule: align away the degrees of freedom the sensors can't observe, and no more. Aligning extra degrees of freedom hides real estimation error in the fitted transform.

| Sensor setup | Unobservable | Alignment |
| --- | --- | --- |
| Monocular camera | position, orientation, scale | 7-DoF similarity (this doc's full Umeyama) |
| Stereo, RGB-D, LiDAR | position, orientation | 6-DoF rigid: the same algorithm with $s$ fixed to 1 (the classic Kabsch/Horn fit) |
| Visual-inertial (camera + IMU) | position, yaw | 4-DoF: translation plus rotation about gravity only, because gravity makes roll and pitch observable |

Zhang & Scaramuzza (2018) give the full treatment, including the 4-DoF variant.

---

## 2. What it actually computes

Given $n$ corresponding point pairs $`\{(x_i, y_i)\}`$ - here, $x_i$ from the estimated reconstruction and $y_i$ the ground truth - Umeyama's method (1991) finds the scale $s > 0$, rotation $R \in SO(3)$, and translation $t \in \mathbb{R}^3$ that solve:

```math
\boxed{\min_{s,R,t} \sum_{i=1}^{n} \left\| s R x_i + t - y_i \right\|^2}
```

i.e., the least-squares best-fit similarity transform mapping the estimated points onto the ground-truth points. Unlike a generic nonlinear least-squares problem, this one has a **closed-form solution** via SVD - no Gauss-Newton iteration needed.

---

## 3. The closed-form solution

This repo's implementation, [`umeyama_alignment`](../../utils.py) in `utils.py`, is the textbook closed-form solution end-to-end:

**Step 1. Center both point sets** on their own centroids, and stack the centered points as the rows of two $n\times3$ matrices $X$ and $Y$:

- centroids: $`\mu_{\text{est}} = \frac{1}{n}\sum x_i`$ and $`\mu_{\text{true}} = \frac{1}{n}\sum y_i`$
- centered rows: $`X_i = x_i - \mu_{\text{est}}`$ and $`Y_i = y_i - \mu_{\text{true}}`$

Centering removes translation from the problem; step 4 recovers it.

**Step 2. Cross-covariance and its SVD:**

$$\Sigma = \frac{1}{n} Y^\top X = U D V^\top$$

**Step 3. Rotation, with a reflection guard:**

$$R = U S V^\top$$

```math
S = \begin{cases} 
I & \text{if } \det(U)\det(V^\top) > 0 \\ 
\text{diag}(1,1,-1) & \text{if } \det(U)\det(V^\top) < 0 
\end{cases}
```

Plain $UV^\top$ is the best-fit *orthogonal* matrix, but that includes reflections ($\det = -1$) as well as rotations ($\det = +1$). Here $\det(U)\det(V^\top)$ is $\pm1$, and the code flips when it is negative. $S$ then flips the axis paired with $\Sigma$'s smallest singular value (the last, since `np.linalg.svd` sorts in descending order) whenever the plain fit would be a reflection. §4 shows why this matters and what it costs.

**Step 4. Scale and translation:**

```math
s = \frac{\text{tr}(DS)}{\text{var}(X)}, \qquad \text{var}(X) = \frac{1}{n}\sum \|X_i\|^2, \qquad t = \mu_{\text{true}} - s R \mu_{\text{est}}
```

These are the four steps `umeyama_alignment` runs, in order.

---

## 4. Why the reflection guard matters

Without step 3's correction, the fit can return a **mirror image** instead of a rotation. Take the unit tetrahedron with corners $(0,0,0)$, $(1,0,0)$, $(0,1,0)$, $(0,0,1)$ (the same points as §5) and its mirror image across the $z=0$ plane, made by negating each $z$ ($\det = -1$ relative to the original, so no rotation reproduces it exactly):

- **Uncorrected** ($R = UV^\top$): fits **perfectly** (residual $\approx 0$), but $\det(R) \approx -1$, which is not a valid rotation and not a valid camera/robot pose.
- **Corrected** ($R = USV^\top$): $\det(R) = +1$, but it can only *approximate* the mirrored points. Here the best fit has a max per-point residual of about $0.77$ (at the origin point; the other three are off by about $0.31$), and scale drops from $1.0$ to $s = 7/9 \approx 0.78$ to partially compensate.

So the guard trades a perfect fit on mirror-related point sets for a valid rotation. For the point sets this repo's scripts generate (cameras spread around a 3D landmark cluster), we don't expect the reflection case to trigger (not measured separately). It is more likely for noisy or nearly degenerate configurations; for exactly coplanar points the flip costs nothing.

---

## 5. A worked example: recovering a known transform exactly

Take four non-coplanar points and a known similarity transform - scale $s=2$, a $90°$ yaw about $z$, translation $t=(1,2,3)$:

```math
X = \begin{bmatrix} 0 & 0 & 0\\
1 & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 1 \end{bmatrix}
```

```math
\qquad R_{\text{true}} = \begin{bmatrix} 0 & -1 & 0\\
1 & 0 & 0\\
0 & 0 & 1 \end{bmatrix}
```

```math
\qquad Y = s\,(R_{\text{true}} X^\top)^\top + t
```

Running `umeyama_alignment(X, Y)` recovers $\hat{s} = 2.0$, $\hat{R} = R_{\text{true}}$, and $\hat{t} = (1, 2, 3)$ back out - to floating-point precision ($< 10^{-15}$ max error), since 4 noise-free points are consistent with one similarity transform and there is no noise to average out. With real (noisy) data, the same four steps return the *least-squares best* $s, R, t$, like fitting a line through noisy points.

---

## 6. Where this is (and isn't) used in this repo

- **[`bundle_adjustment.py`](../../use_numpy/bundle_adjustment.py)** (both `use_numpy/` and `use_manif/`) calls `umeyama_alignment` on the *camera positions* after solving, then applies the recovered $(s, R, t)$ to **both** the camera poses and the landmark positions before computing `pose_errors`/`landmark_errors` against ground truth. This is standard practice for evaluating monocular BA/SfM (Structure from Motion) output; see [bundle_adjustment.md](../optimization/bundle_adjustment.md) for how it fits into that script.
  - Reprojection error is unaffected by the alignment (a similarity transform leaves every projected pixel unchanged), so only the absolute pose/landmark errors depend on this step.
- **[`bundle_adjustment_advanced.py`](../../use_numpy/bundle_adjustment_advanced.py)** does **not** call it. It hard-fixes **two** keyframes (poses 0 and 1) as a gauge anchor instead of using a soft gauge-prior factor:
  - One fixed keyframe removes only the 6 rigid DoF. Scale stays unobservable, since the reconstruction rescaled about that pose satisfies every constraint equally well.
  - A *second* fixed keyframe also fixes the *distance* between the anchors, so no gauge freedom is left.
  - That doesn't mean there's nothing to align. Keyframe 0 is fixed at its true pose, but keyframe 1 keeps its noisy front-end pose, so the scale and orientation it pins are wrong (up to ~19% in scale over seeds 0-4, see [bundle_adjustment.md §13.6](../optimization/bundle_adjustment.md#136-in-this-repo)). The reported raw error therefore includes a large similarity offset that Umeyama alignment would remove ([§14](../optimization/bundle_adjustment.md#14-evaluating-the-result-gauge-freedom-and-umeyama-alignment)).

---

## 7. One-sentence intuition

> **Umeyama alignment is a closed-form least-squares fit that finds the one scale + rotation + translation that best overlays an estimate onto ground truth, so that "position error in meters" means something for monocular output.**

---

## 8. References

1. Umeyama, S. (1991). *Least-Squares Estimation of Transformation Parameters Between Two Point Patterns*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(4), 376-380. https://doi.org/10.1109/34.88573 - the original closed-form SVD-based derivation behind §3, and the namesake of `umeyama_alignment` in `utils.py`.
2. Zhang, Z., & Scaramuzza, D. (2018). *A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry*. 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 7244-7251. https://doi.org/10.1109/IROS.2018.8593941 - which alignment to use for which sensor setup (§1.1), including the 4-DoF visual-inertial case.
