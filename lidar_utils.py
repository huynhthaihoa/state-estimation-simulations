'''
Library-agnostic LiDAR helpers shared by use_numpy/lidar_primitives.py and
use_manif/lidar_primitives.py: everything here is plain numpy (+ scipy's
cKDTree) and never touches an SE(3) Jacobian, so both backends import the
same copy.

  - Scene: finite planar rectangles + finite vertical-or-not cylinders.
  - LiDAR: a spinning multi-beam scanner, simulated by analytic ray casting
    (nearest hit, max range, Gaussian range noise along each ray). Each
    returned point carries the index of the primitive it hit -- used ONLY to
    score extraction and in tests, never handed to an estimator.
  - Extraction: per-point PCA normal + curvature from k nearest neighbors,
    then sequential RANSAC -- each round draws the best plane hypothesis
    (3 points) and the best cylinder hypothesis (2 points + their normals
    give the axis), refines both by least squares, and keeps whichever
    explains more of the remaining points (see extract_primitives).
  - S^2 helpers: tangent basis, exponential map and log for unit vectors
    (plane normals, cylinder axes).

Conventions:
  - A sensor pose is (R, t): world point = R @ sensor point + t.
  - A plane is (n, d) with n . x + d = 0, |n| = 1. Planes extracted in the
    sensor frame have n pointing toward the sensor, so d > 0 is the sensor's
    distance to the plane.
  - A cylinder is (a, c, r): unit axis a, c = the axis point closest to the
    frame origin (so a . c = 0), radius r. The sign of a is arbitrary.
'''

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# S^2 helpers
# ---------------------------------------------------------------------------

def s2_basis(n):
    """Deterministic orthonormal basis of the tangent plane of S^2 at n.
    Arguments:
        n: unit vector (3,)
    Returns:
        B: (3,2) matrix whose columns are orthonormal and orthogonal to n
    """
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    b1 = np.cross(n, helper)
    b1 /= np.linalg.norm(b1)
    return np.column_stack([b1, np.cross(n, b1)])


def s2_exp(n, delta):
    """Exponential map on S^2: moves n along the great circle in tangent
    direction B(n) delta by the angle |delta|.
    Arguments:
        n: unit vector (3,)
        delta: tangent coordinates (2,) in the basis s2_basis(n)
    Returns:
        unit vector (3,)
    """
    v = s2_basis(n) @ delta
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return n.copy()
    return np.cos(theta) * n + np.sin(theta) * v / theta


def s2_log(n, m):
    """Inverse of s2_exp: tangent coordinates at n of the unit vector m.
    Arguments:
        n: base unit vector (3,)
        m: target unit vector (3,), not antipodal to n
    Returns:
        delta: (2,) with s2_exp(n, delta) == m
    """
    w = m - (n @ m) * n
    s = np.linalg.norm(w)
    if s < 1e-12:
        return np.zeros(2)
    theta = np.arctan2(s, n @ m)
    return s2_basis(n).T @ (theta * w / s)


# ---------------------------------------------------------------------------
# Primitives and scenes
# ---------------------------------------------------------------------------

@dataclass
class Rect:
    """Finite planar rectangle: center c, in-plane unit axes u, v, half extents."""
    c: np.ndarray
    u: np.ndarray
    v: np.ndarray
    half_u: float
    half_v: float

    @property
    def n(self):
        return np.cross(self.u, self.v)

    def plane(self):
        """Infinite-plane parameters (n, d) of the rectangle, world frame."""
        n = self.n
        return n, -n @ self.c


@dataclass
class Cylinder:
    """Finite cylinder: base point on the axis, unit axis, radius, height."""
    base: np.ndarray
    a: np.ndarray
    r: float
    height: float

    def params(self):
        """Infinite-cylinder parameters (a, c, r) with a . c = 0, world frame."""
        return self.a, self.base - (self.a @ self.base) * self.a, self.r


@dataclass
class Scene:
    rects: list
    cylinders: list

    @property
    def n_primitives(self):
        return len(self.rects) + len(self.cylinders)

    def kind(self, label):
        """'plane' or 'cylinder' for a primitive label (rects come first)."""
        return "plane" if label < len(self.rects) else "cylinder"


def corridor_scene(length=60.0, width=6.0, height=3.0, pillars=False, pillar_radius=0.25,
                   pillar_spacing=8.0, pillar_offset=1.5):
    """Straight corridor along world x, open at both ends (no end walls), so
    that nothing constrains translation along x once the ends are out of
    range. With `pillars`, two rows of floor-to-ceiling cylinders at
    y = +/- pillar_offset, staggered by half a spacing.
    Arguments:
        length, width, height: corridor size (m); x in [-length/2, length/2]
        pillars: add the cylinders
        pillar_radius, pillar_spacing, pillar_offset: cylinder layout (m)
    Returns:
        Scene
    """
    ex, ey, ez = np.eye(3)
    L, W, H = length / 2, width / 2, height / 2
    rects = [
        Rect(np.array([0.0, 0.0, 0.0]), ex, ey, L, W),            # floor, n = +z
        Rect(np.array([0.0, 0.0, height]), ey, ex, W, L),         # ceiling, n = -z
        Rect(np.array([0.0, -W, H]), ez, ex, H, L),               # right wall, n = +y
        Rect(np.array([0.0, W, H]), ex, ez, L, H),                # left wall, n = -y
    ]
    cylinders = []
    if pillars:
        for side, shift in ((1.0, 0.0), (-1.0, pillar_spacing / 2)):
            for x in np.arange(-L + pillar_spacing / 2 + shift, L, pillar_spacing):
                cylinders.append(Cylinder(np.array([x, side * pillar_offset, 0.0]), ez, pillar_radius, height))
    return Scene(rects, cylinders)


def axis_rect(normal_axis, sign, offset, bounds):
    """Axis-aligned rectangle in the plane x[normal_axis] = offset, normal
    sign * e[normal_axis], spanning bounds[k] = (lo, hi) along each of the
    two other world axes k.
    Returns:
        Rect
    """
    E = np.eye(3)
    a, b = [k for k in range(3) if k != normal_axis]
    # (u, v) ordered so that u x v = sign * e[normal_axis]
    ua, va = (a, b) if np.cross(E[a], E[b])[normal_axis] * sign > 0 else (b, a)
    c = np.zeros(3)
    c[normal_axis] = offset
    for k in (a, b):
        c[k] = 0.5 * (bounds[k][0] + bounds[k][1])
    return Rect(c, E[ua], E[va], 0.5 * (bounds[ua][1] - bounds[ua][0]), 0.5 * (bounds[va][1] - bounds[va][0]))


def l_corridor_scene(leg=40.0, width=6.0, height=3.0, pillar_radius=0.25, pillar_spacing=8.0, pillar_offset=1.5):
    """L-shaped corridor: leg A along world x for x <= width/2, then leg B
    along world y for y >= -width/2, both open at their far ends, with two
    staggered rows of floor-to-ceiling pillars in each leg (none in the
    corner square). Floor and ceiling are one infinite plane each but two
    rectangles; the outer wall of the corner (x = +width/2) closes leg A.
    Arguments:
        leg: length of each leg from the corner (m)
        width, height, pillar_*: as in corridor_scene
    Returns:
        Scene
    """
    W = width / 2
    A, B = {0: (-leg, W), 1: (-W, W)}, {0: (-W, W), 1: (W, leg)}
    rects = [
        axis_rect(2, +1, 0.0, A), axis_rect(2, +1, 0.0, B),                  # floor (one plane)
        axis_rect(2, -1, height, A), axis_rect(2, -1, height, B),            # ceiling (one plane)
        axis_rect(1, +1, -W, {0: (-leg, W), 2: (0.0, height)}),              # leg A right wall
        axis_rect(1, -1, W, {0: (-leg, -W), 2: (0.0, height)}),              # leg A left wall (inner corner)
        axis_rect(0, +1, -W, {1: (W, leg), 2: (0.0, height)}),               # leg B left wall (inner corner)
        axis_rect(0, -1, W, {1: (-W, leg), 2: (0.0, height)}),               # leg B right wall (outer corner)
    ]
    ez = np.array([0.0, 0.0, 1.0])
    cylinders = []
    for side, shift in ((1.0, 0.0), (-1.0, pillar_spacing / 2)):
        for s in np.arange(W + 2.0 + shift, leg, pillar_spacing):
            cylinders.append(Cylinder(np.array([-s, side * pillar_offset, 0.0]), ez, pillar_radius, height))
            cylinders.append(Cylinder(np.array([side * pillar_offset, s, 0.0]), ez, pillar_radius, height))
    return Scene(rects, cylinders)


def l_corridor_trajectory(start=20.0, turn_radius=1.5, spacing=1.0, height=1.2, wobble_deg=2.0):
    """Sensor poses along the L-corridor's center line: leg A from
    x = -start to the corner, a quarter circle of radius turn_radius, then
    leg B to y = start, one pose every `spacing` meters of path. Yaw follows
    the path; roll, pitch and height wobble gently so every pose is full 6-DoF.
    Returns:
        list of (R, t)
    """
    r = turn_radius
    lengths = np.array([start - r, 0.5 * np.pi * r, start - r])
    poses = []
    for k, s in enumerate(np.arange(0.0, lengths.sum() + 1e-9, spacing)):
        if s <= lengths[0]:
            p, yaw = np.array([-start + s, 0.0]), 0.0
        elif s <= lengths[:2].sum():
            phi = (s - lengths[0]) / r
            p, yaw = np.array([-r + r * np.sin(phi), r - r * np.cos(phi)]), phi
        else:
            p, yaw = np.array([0.0, r + s - lengths[:2].sum()]), 0.5 * np.pi
        roll, pitch = np.deg2rad(wobble_deg) * np.array([np.sin(0.7 * k), np.cos(0.5 * k)])
        cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        poses.append((Rz @ Ry @ Rx, np.array([p[0], p[1], height + 0.05 * np.sin(0.3 * k)])))
    return poses


# ---------------------------------------------------------------------------
# LiDAR simulation
# ---------------------------------------------------------------------------

@dataclass
class LidarConfig:
    n_rings: int = 32
    elev_min_deg: float = -20.0
    elev_max_deg: float = 20.0
    n_azimuth: int = 360
    min_range: float = 0.3
    max_range: float = 20.0
    range_noise_std: float = 0.01


def lidar_ray_directions(cfg):
    """Unit ray directions in the sensor frame, ring-major.
    Arguments:
        cfg: LidarConfig
    Returns:
        (n_rings * n_azimuth, 3) array
    """
    elev = np.deg2rad(np.linspace(cfg.elev_min_deg, cfg.elev_max_deg, cfg.n_rings))
    az = np.linspace(0.0, 2 * np.pi, cfg.n_azimuth, endpoint=False)
    E, A = np.meshgrid(elev, az, indexing="ij")
    return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], axis=-1).reshape(-1, 3)


def raycast(scene, origin, dirs):
    """Nearest hit of each ray with the scene.
    Arguments:
        scene: Scene
        origin: ray origin (3,), world frame
        dirs: (m,3) unit ray directions, world frame
    Returns:
        s: (m,) hit distance (inf where nothing is hit)
        label: (m,) index of the primitive hit (-1 where nothing is hit)
    """
    m = len(dirs)
    s_best = np.full(m, np.inf)
    label = np.full(m, -1)
    for k, rect in enumerate(scene.rects):
        n = rect.n
        den = dirs @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (n @ (rect.c - origin)) / den
            q = origin + s[:, None] * dirs - rect.c
        ok = (np.abs(den) > 1e-12) & (s > 0) & (np.abs(q @ rect.u) <= rect.half_u) & (np.abs(q @ rect.v) <= rect.half_v)
        better = ok & (s < s_best)
        s_best[better], label[better] = s[better], k
    for j, cyl in enumerate(scene.cylinders):
        a = cyl.a
        o = (origin - cyl.base) - ((origin - cyl.base) @ a) * a
        dp = dirs - (dirs @ a)[:, None] * a
        A = np.sum(dp * dp, axis=1)
        B = 2 * dp @ o
        C = o @ o - cyl.r ** 2
        disc = B ** 2 - 4 * A * C
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (-B - np.sqrt(np.maximum(disc, 0.0))) / (2 * A)
        h = (origin - cyl.base) @ a + s * (dirs @ a)
        ok = (disc >= 0) & (A > 1e-12) & (s > 0) & (h >= 0) & (h <= cyl.height)
        better = ok & (s < s_best)
        s_best[better], label[better] = s[better], len(scene.rects) + j
    return s_best, label


def simulate_scan(scene, R, t, cfg, rng):
    """One LiDAR scan from sensor pose (R, t).
    Arguments:
        scene: Scene
        R, t: sensor-to-world rotation (3,3) and position (3,)
        cfg: LidarConfig
        rng: numpy Generator (range noise)
    Returns:
        points: (N,3) measured points, sensor frame
        labels: (N,) primitive index of each point (scoring/tests only)
    """
    dirs_s = lidar_ray_directions(cfg)
    s, label = raycast(scene, t, dirs_s @ R.T)
    keep = (label >= 0) & (s >= cfg.min_range) & (s <= cfg.max_range)
    s_meas = s[keep] + cfg.range_noise_std * rng.standard_normal(keep.sum())
    return s_meas[:, None] * dirs_s[keep], label[keep]


def plane_to_frame(n, d, R, t):
    """Expresses a world plane (n, d) in the frame of pose (R, t)."""
    return R.T @ n, d + n @ t


def cylinder_to_frame(a, c, r, R, t):
    """Expresses a world cylinder (a, c, r) in the frame of pose (R, t)."""
    a_s = R.T @ a
    q = R.T @ (c - t)
    return a_s, q - (a_s @ q) * a_s, r


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def estimate_normals(points, k=12):
    """Per-point PCA normal and curvature from the k nearest neighbors.
    Normals are oriented toward the sensor (the frame origin).
    Arguments:
        points: (N,3), sensor frame
        k: neighborhood size
    Returns:
        normals: (N,3) unit normals
        curvature: (N,) lambda_min / (lambda_0 + lambda_1 + lambda_2), 0 on a perfect plane
    """
    _, idx = cKDTree(points).query(points, k=k)
    nb = points[idx] - points[idx].mean(axis=1, keepdims=True)
    evals, evecs = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))
    normals = evecs[:, :, 0]
    flip = np.sum(normals * points, axis=1) > 0
    normals[flip] *= -1
    curvature = evals[:, 0] / np.maximum(evals.sum(axis=1), 1e-300)
    return normals, curvature


def fit_plane(points):
    """Least-squares (orthogonal-distance) plane through points, normal
    oriented toward the frame origin.
    Returns:
        (n, d)
    """
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid)
    n = Vt[-1]
    d = -n @ centroid
    return (-n, -d) if d < 0 else (n, d)


def cylinder_residuals(points, a, c, r):
    """Signed point-to-surface distances ||(I - a a^T)(x - c)|| - r."""
    w = points - c
    q = w - (w @ a)[:, None] * a
    return np.linalg.norm(q, axis=1) - r


def refine_cylinder(points, a, c, r, n_iters=10):
    """Gauss-Newton refinement of (a, c, r) on the point-to-surface residuals.
    a lives on S^2 (2 DoF), c moves in the plane orthogonal to a (2 DoF; its
    component along a is a free gauge, removed after every step), r is
    Euclidean (1 DoF).
    Returns:
        (a, c, r) with a . c = 0
    """
    for _ in range(n_iters):
        w = points - c
        along = w @ a
        q = w - along[:, None] * a
        rho = np.maximum(np.linalg.norm(q, axis=1), 1e-12)
        e_hat = q / rho[:, None]
        B = s2_basis(a)
        # de/da = -(a.w) e_hat^T, de/dc = -e_hat^T, de/dr = -1
        J = np.hstack([-(along[:, None] * e_hat) @ B, -e_hat @ B, -np.ones((len(points), 1))])
        e = rho - r
        delta = np.linalg.lstsq(J, -e, rcond=None)[0]
        c_new = c + B @ delta[2:4]
        a = s2_exp(a, delta[:2])
        c = c_new - (a @ c_new) * a
        r = r + delta[4]
        if np.linalg.norm(delta) < 1e-10:
            break
    return a, c, r


def cylinder_range_residuals(x, a0, rays, ranges):
    """Range-space residuals of a cylinder for rays cast from the frame
    origin (the sensor): measured range minus the range at which the ray
    meets the cylinder. A ray that misses the cylinder gets its
    closest-approach range there plus a second residual, its gap to the
    surface; both are continuous through tangency, so the cost is too.
    Arguments:
        x: (5,) local coordinates: S^2 step of the axis from a0 (2), axis
           point in s2_basis(a0) (2), radius (1)
        a0: reference axis (3,)
        rays: (N,3) unit ray directions; ranges: (N,) measured ranges
    Returns:
        (2N,) residuals: [range residuals, miss gaps (0 for a hit)]
    """
    a = s2_exp(a0, x[:2])
    c = s2_basis(a0) @ x[2:4]
    o = -(c - (c @ a) * a)
    dp = rays - (rays @ a)[:, None] * a
    A = np.sum(dp * dp, axis=1)
    B = 2 * dp @ o
    C = o @ o - x[4] ** 2
    disc = B ** 2 - 4 * A * C
    s_hit = (-B - np.sqrt(np.maximum(disc, 0.0))) / (2 * A)
    gap = np.sqrt(np.maximum(o @ o - B ** 2 / (4 * A), 0.0)) - x[4]
    return np.concatenate([ranges - s_hit, np.maximum(gap, 0.0)])


def refine_cylinder_range(points, a, c, r):
    """Maximum-likelihood cylinder fit for range-only noise: minimizes the
    measured-vs-predicted range along each ray (cylinder_range_residuals)
    instead of the orthogonal point-to-surface distance. LiDAR noise lies
    along the ray, which near a pillar's silhouette is almost tangent to
    the surface, so the orthogonal fit (refine_cylinder) underestimates
    the radius by a bias that grows with the square of the noise. Starts
    from (a, c, r), normally refine_cylinder's result; uses scipy's
    least_squares with a finite-difference Jacobian.
    Arguments:
        points: (N,3) cylinder inliers, sensor frame (the sensor is the origin)
        a, c, r: starting axis, axis point (a . c = 0), radius
    Returns:
        (a, c, r) with a . c = 0
    """
    ranges = np.linalg.norm(points, axis=1)
    x0 = np.concatenate([np.zeros(2), s2_basis(a).T @ c, [r]])
    x = least_squares(cylinder_range_residuals, x0, args=(a, points / ranges[:, None], ranges)).x
    a_new = s2_exp(a, x[:2])
    c_new = s2_basis(a) @ x[2:4]
    return a_new, c_new - (a_new @ c_new) * a_new, x[4]


def trim_by_residual(idx, residuals, k=3.0, floor=1e-6):
    """Keeps the inliers whose residual is within k robust sigmas
    (1.4826 * median absolute residual, floored at `floor`). Points near an
    intersection pass the fixed RANSAC band with a corner-blended normal while
    lying centimeters off the surface; this final cut removes them.
    Arguments:
        idx: inlier indices
        residuals: their residuals to the refined primitive
    Returns:
        the kept subset of idx
    """
    scale = 1.4826 * np.median(np.abs(residuals))
    return idx[np.abs(residuals) <= max(k * scale, floor)]


@dataclass
class ExtractionConfig:
    k_normals: int = 12
    inlier_dist: float = 0.05
    inlier_angle_deg: float = 25.0
    min_plane_points: int = 200
    min_cylinder_points: int = 60
    plane_iters: int = 300
    cylinder_iters: int = 500
    radius_range: tuple = (0.05, 1.0)
    max_planes: int = 10
    max_cylinders: int = 20
    cylinder_fit: str = "odr"   # "odr" (orthogonal distance) or "range" (+ refine_cylinder_range)


def extract_primitives(points, cfg, rng, normals=None):
    """Sequential RANSAC over two shape types. Every round draws the best
    plane hypothesis (3 points) and the best cylinder hypothesis (2 points +
    their normals), refines both, and keeps whichever explains more of the
    remaining points (the selection rule of Schnabel et al. 2007). Planes
    are not extracted first on purpose: within a few-cm band, the side of a
    pillar facing the sensor fits a plane too, and a planes-first pass
    would swallow it.
    Arguments:
        points: (N,3) one scan, sensor frame
        cfg: ExtractionConfig
        rng: numpy Generator (hypothesis sampling)
        normals: optional precomputed (N,3) normals (else estimate_normals)
    Returns:
        planes: list of dicts {n, d, inliers}
        cylinders: list of dicts {a, c, r, inliers}
        (inliers are index arrays into `points`)
    """
    if normals is None:
        normals, _ = estimate_normals(points, cfg.k_normals)
    cos_tol = np.cos(np.deg2rad(cfg.inlier_angle_deg))

    def plane_inliers(n, d, idx):
        P, N = points[idx], normals[idx]
        return idx[(np.abs(P @ n + d) < cfg.inlier_dist) & (np.abs(N @ n) > cos_tol)]

    def cylinder_inliers(a, c, r, idx):
        P, N = points[idx], normals[idx]
        w = P - c
        q = w - (w @ a)[:, None] * a
        rho = np.maximum(np.linalg.norm(q, axis=1), 1e-12)
        radial = np.abs(np.sum(N * q, axis=1)) / rho
        return idx[(np.abs(rho - r) < cfg.inlier_dist) & (radial > cos_tol)]

    def best_plane(remaining):
        best = np.array([], dtype=int)
        for _ in range(cfg.plane_iters):
            p0, p1, p2 = points[rng.choice(remaining, 3, replace=False)]
            n = np.cross(p1 - p0, p2 - p0)
            if np.linalg.norm(n) < 1e-9:
                continue
            n /= np.linalg.norm(n)
            inl = plane_inliers(n, -n @ p0, remaining)
            if len(inl) > len(best):
                best = inl
        if len(best) < cfg.min_plane_points:
            return None
        for _ in range(2):
            n, d = fit_plane(points[best])
            best = plane_inliers(n, d, remaining)
        n, d = fit_plane(points[best])
        best = trim_by_residual(best, points[best] @ n + d)
        if len(best) < cfg.min_plane_points:
            return None
        n, d = fit_plane(points[best])
        return {"n": n, "d": d, "inliers": best}

    def best_cylinder(remaining):
        best, best_params = np.array([], dtype=int), None
        for _ in range(cfg.cylinder_iters):
            i, j = rng.choice(remaining, 2, replace=False)
            a = np.cross(normals[i], normals[j])
            if np.linalg.norm(a) < 0.2:
                continue
            a /= np.linalg.norm(a)
            # in the plane orthogonal to a, the two normal lines meet at the axis
            B = s2_basis(a)
            p_i, p_j = B.T @ points[i], B.T @ points[j]
            m_i, m_j = B.T @ normals[i], B.T @ normals[j]
            M = np.column_stack([m_i, -m_j])
            if abs(np.linalg.det(M)) < 1e-6:
                continue
            s_i, s_j = np.linalg.solve(M, p_j - p_i)
            center2 = 0.5 * (p_i + s_i * m_i + p_j + s_j * m_j)
            r = 0.5 * (np.linalg.norm(p_i - center2) + np.linalg.norm(p_j - center2))
            if not cfg.radius_range[0] <= r <= cfg.radius_range[1]:
                continue
            inl = cylinder_inliers(a, B @ center2, r, remaining)
            if len(inl) > len(best):
                best, best_params = inl, (a, B @ center2, r)
        if len(best) < cfg.min_cylinder_points:
            return None
        a, c, r = best_params
        for _ in range(2):
            a, c, r = refine_cylinder(points[best], a, c, r)
            if not cfg.radius_range[0] <= r <= cfg.radius_range[1]:
                return None
            best = cylinder_inliers(a, c, r, remaining)
            if len(best) < cfg.min_cylinder_points:
                return None
        a, c, r = refine_cylinder(points[best], a, c, r)
        best = trim_by_residual(best, cylinder_residuals(points[best], a, c, r))
        if len(best) < cfg.min_cylinder_points:
            return None
        a, c, r = refine_cylinder(points[best], a, c, r)
        if cfg.cylinder_fit == "range":
            a, c, r = refine_cylinder_range(points[best], a, c, r)
        if not cfg.radius_range[0] <= r <= cfg.radius_range[1]:
            return None
        return {"a": a, "c": c, "r": r, "inliers": best}

    planes, cylinders = [], []
    remaining = np.arange(len(points))
    while True:
        plane = best_plane(remaining) if len(planes) < cfg.max_planes and len(remaining) >= cfg.min_plane_points else None
        cyl = (best_cylinder(remaining)
               if len(cylinders) < cfg.max_cylinders and len(remaining) >= cfg.min_cylinder_points else None)
        if plane is None and cyl is None:
            break
        if cyl is None or (plane is not None and len(plane["inliers"]) >= len(cyl["inliers"])):
            planes.append(plane)
            remaining = np.setdiff1d(remaining, plane["inliers"])
        else:
            cylinders.append(cyl)
            remaining = np.setdiff1d(remaining, cyl["inliers"])
    return planes, cylinders


# ---------------------------------------------------------------------------
# Scoring against the truth (experiment 1)
# ---------------------------------------------------------------------------

def plane_errors(n_est, d_est, n_true, d_true):
    """(normal angle error in rad, offset error in m) between two planes."""
    return np.arccos(np.clip(abs(n_est @ n_true), -1.0, 1.0)), abs(d_est - d_true * np.sign(n_est @ n_true))


def cylinder_errors(a_est, c_est, r_est, a_true, c_true, r_true):
    """(axis angle error in rad, axis position error in m, radius error in m).
    The position error is the distance from the true axis point c_true (the
    one closest to the frame origin) to the estimated axis line."""
    angle = np.arccos(np.clip(abs(a_est @ a_true), -1.0, 1.0))
    w = c_true - c_est
    return angle, np.linalg.norm(w - (w @ a_est) * a_est), abs(r_est - r_true)


def score_extraction(scene, R, t, labels, planes, cylinders, min_points, points=None):
    """Matches extracted primitives to the true ones by their inlier labels
    (majority vote) and returns their parameter errors in the sensor frame.
    Arguments:
        scene: Scene
        R, t: true sensor pose
        labels: (N,) true primitive index of every scan point
        planes, cylinders: output of extract_primitives
        min_points: a true primitive counts as visible with at least this many points
    Returns:
        dict with lists 'plane' [(angle, offset)], 'cylinder' [(angle, position, radius, signed radius)],
        and, when `points` (the scan) is given, 'cylinder_range': the same for
        refine_cylinder_range started from each extracted cylinder;
        counts 'planes_visible', 'planes_found', 'cylinders_visible', 'cylinders_found',
        'false_positives' (extracted primitives whose majority label has another type)
    """
    out = {"plane": [], "cylinder": [], "cylinder_range": [], "false_positives": 0}
    counts = np.bincount(labels, minlength=scene.n_primitives)
    visible = counts >= min_points
    found = set()
    for kind, prims in (("plane", planes), ("cylinder", cylinders)):
        for prim in prims:
            lab = np.bincount(labels[prim["inliers"]], minlength=scene.n_primitives).argmax()
            if scene.kind(lab) != kind or lab in found:
                out["false_positives"] += 1
                continue
            found.add(lab)
            if kind == "plane":
                n_t, d_t = plane_to_frame(*scene.rects[lab].plane(), R, t)
                out["plane"].append(plane_errors(prim["n"], prim["d"], n_t, d_t))
            else:
                a_t, c_t, r_t = cylinder_to_frame(*scene.cylinders[lab - len(scene.rects)].params(), R, t)
                out["cylinder"].append((*cylinder_errors(prim["a"], prim["c"], prim["r"], a_t, c_t, r_t),
                                        prim["r"] - r_t))
                if points is not None:
                    a_r, c_r, r_r = refine_cylinder_range(points[prim["inliers"]], prim["a"], prim["c"], prim["r"])
                    out["cylinder_range"].append((*cylinder_errors(a_r, c_r, r_r, a_t, c_t, r_t), r_r - r_t))
    n_rect = len(scene.rects)
    out["planes_visible"] = int(visible[:n_rect].sum())
    out["cylinders_visible"] = int(visible[n_rect:].sum())
    out["planes_found"] = sum(1 for lab in found if lab < n_rect and visible[lab])
    out["cylinders_found"] = sum(1 for lab in found if lab >= n_rect and visible[lab])
    return out


def random_sensor_pose(rng, x_range=(-10.0, 10.0), y_range=(-0.8, 0.8), z_range=(1.0, 1.5), tilt_deg=5.0):
    """Random sensor pose inside the corridor: uniform position, uniform yaw,
    roll and pitch uniform in +/- tilt_deg. The default y range keeps the
    sensor clear of corridor_scene's pillars.
    Returns:
        (R, t)
    """
    roll, pitch = np.deg2rad(rng.uniform(-tilt_deg, tilt_deg, 2))
    yaw = rng.uniform(-np.pi, np.pi)
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    t = np.array([rng.uniform(*x_range), rng.uniform(*y_range), rng.uniform(*z_range)])
    return Rz @ Ry @ Rx, t


def inlier_threshold(range_noise_std):
    """RANSAC distance threshold used by the experiments: 3 sigma of the
    range noise, but never below 5 cm (normals and fit residuals carry more
    than the raw range noise at low noise)."""
    return max(0.05, 3.0 * range_noise_std)


def run_extraction_experiment(scene, noise_levels, azimuth_counts, n_trials, seed, min_points=60):
    """Experiment 1: extraction accuracy and detection rate vs. range noise
    and horizontal point density. Every (noise, density) cell sees the same
    n_trials sensor poses.
    Arguments:
        scene: Scene (planes + cylinders)
        noise_levels: range noise standard deviations (m)
        azimuth_counts: points per ring (horizontal density)
        n_trials: random sensor poses per cell
        seed: RNG seed
        min_points: a true primitive counts as visible with at least this many points
    Returns:
        dict of (len(noise_levels), len(azimuth_counts)) arrays:
        plane_angle_deg / plane_offset_mm / cyl_angle_deg / cyl_pos_mm / cyl_radius_mm
        as (median, RMSE) pairs under keys '<name>_median' and '<name>_rmse'
        (cyl_radius_bias_mm: signed radius error, use its mean = '<name>_mean');
        every cyl_* name again with suffix '_range' for refine_cylinder_range
        on the same inliers;
        plane_detection, cyl_detection (found / visible), false_positives (total)
    """
    poses = [random_sensor_pose(np.random.default_rng([seed, k])) for k in range(n_trials)]
    cyl_names = ["cyl_angle_deg", "cyl_pos_mm", "cyl_radius_mm", "cyl_radius_bias_mm"]
    names = ["plane_angle_deg", "plane_offset_mm"] + cyl_names + [nm + "_range" for nm in cyl_names]
    shape = (len(noise_levels), len(azimuth_counts))
    out = {f"{nm}_{st}": np.full(shape, np.nan) for nm in names for st in ("median", "rmse", "mean")}
    for key in ("plane_detection", "cyl_detection", "false_positives"):
        out[key] = np.zeros(shape)
    for i, sigma in enumerate(noise_levels):
        ecfg = ExtractionConfig(inlier_dist=inlier_threshold(sigma))
        for j, n_az in enumerate(azimuth_counts):
            lcfg = LidarConfig(n_azimuth=n_az, range_noise_std=sigma)
            vals = {nm: [] for nm in names}
            counts = np.zeros(5)
            for k, (R, t) in enumerate(poses):
                rng = np.random.default_rng([seed, k, i, j])
                pts, lab = simulate_scan(scene, R, t, lcfg, rng)
                planes, cyls = extract_primitives(pts, ecfg, rng)
                sc = score_extraction(scene, R, t, lab, planes, cyls, min_points, points=pts)
                for ang, off in sc["plane"]:
                    vals["plane_angle_deg"].append(np.rad2deg(ang))
                    vals["plane_offset_mm"].append(1e3 * off)
                for suffix, key in (("", "cylinder"), ("_range", "cylinder_range")):
                    for ang, pos, rad, signed in sc[key]:
                        vals["cyl_angle_deg" + suffix].append(np.rad2deg(ang))
                        vals["cyl_pos_mm" + suffix].append(1e3 * pos)
                        vals["cyl_radius_mm" + suffix].append(1e3 * rad)
                        vals["cyl_radius_bias_mm" + suffix].append(1e3 * signed)
                counts += [sc["planes_found"], sc["planes_visible"], sc["cylinders_found"],
                           sc["cylinders_visible"], sc["false_positives"]]
            for nm in names:
                if vals[nm]:
                    v = np.asarray(vals[nm])
                    out[f"{nm}_median"][i, j] = np.median(v)
                    out[f"{nm}_rmse"][i, j] = np.sqrt(np.mean(v ** 2))
                    out[f"{nm}_mean"][i, j] = np.mean(v)
            out["plane_detection"][i, j] = counts[0] / max(counts[1], 1)
            out["cyl_detection"][i, j] = counts[2] / max(counts[3], 1)
            out["false_positives"][i, j] = counts[4]
    return out


# ---------------------------------------------------------------------------
# Scans and registration settings (shared by both backends)
# ---------------------------------------------------------------------------

@dataclass
class Scan:
    points: np.ndarray
    normals: np.ndarray
    tree: cKDTree
    planes: list
    cylinders: list


def prepare_scan(points, ecfg, rng):
    """Normals, k-d tree and primitives of one scan (sensor frame)."""
    normals, _ = estimate_normals(points, ecfg.k_normals)
    planes, cylinders = extract_primitives(points, ecfg, rng, normals)
    return Scan(points, normals, cKDTree(points), planes, cylinders)


@dataclass
class RegistrationConfig:
    max_pair_dist: float = 1.0          # nearest-neighbor gate (m) at the first iteration ...
    min_pair_dist: float = 0.1          # ... shrinking by gate_decay per iteration down to this (coarse-to-fine)
    gate_decay: float = 0.7
    plane_angle_gate_deg: float = 20.0  # primitive gates
    plane_offset_gate: float = 1.0
    cyl_angle_gate_deg: float = 20.0
    cyl_dist_gate: float = 1.0
    cyl_radius_gate: float = 0.1
    max_iters: int = 100
    tol: float = 1e-6
    rank_rcond: float = 1e-6            # singular values below rank_rcond * largest are treated as exactly zero


def pair_gate(cfg, it):
    """Coarse-to-fine nearest-neighbor gate at iteration `it` (0-based). A
    fixed 1 m gate left point-to-plane ICP 4-12 mm and 0.1-0.2 deg off even
    when started at the truth without noise, likely from pairs whose nearest
    neighbor lies on another surface (the target scan did not sample that
    surface there); a 0.1 m gate from the start would drop the pairs needed
    to correct a large initial error."""
    return max(cfg.min_pair_dist, cfg.max_pair_dist * cfg.gate_decay ** it)


def match_primitives(R, t, src, tgt, cfg):
    """Greedy one-to-one matching of source primitives (mapped by the pose
    (R, t) of the source frame in the target frame) to target primitives
    inside the gates, cheapest normalized cost first.
    Returns:
        list of ('plane', i_src, j_tgt) / ('cylinder', i_src, j_tgt)
    """
    candidates = []
    cos_p = np.cos(np.deg2rad(cfg.plane_angle_gate_deg))
    for i, ps in enumerate(src.planes):
        n = R @ ps["n"]
        d = ps["d"] - n @ t
        for j, pt in enumerate(tgt.planes):
            cos_ang = n @ pt["n"]
            dd = abs(d - pt["d"])
            if cos_ang > cos_p and dd < cfg.plane_offset_gate:
                ang = np.arccos(min(cos_ang, 1.0))
                candidates.append((ang / np.deg2rad(cfg.plane_angle_gate_deg) + dd / cfg.plane_offset_gate,
                                   "plane", i, j))
    cos_c = np.cos(np.deg2rad(cfg.cyl_angle_gate_deg))
    for i, cs in enumerate(src.cylinders):
        a = R @ cs["a"]
        c = R @ cs["c"] + t
        for j, ct in enumerate(tgt.cylinders):
            cos_ang = abs(a @ ct["a"])
            w = c - ct["c"]
            dist = np.linalg.norm(w - (w @ ct["a"]) * ct["a"])
            dr = abs(cs["r"] - ct["r"])
            if cos_ang > cos_c and dist < cfg.cyl_dist_gate and dr < cfg.cyl_radius_gate:
                ang = np.arccos(min(cos_ang, 1.0))
                candidates.append((ang / np.deg2rad(cfg.cyl_angle_gate_deg) + dist / cfg.cyl_dist_gate
                                   + dr / cfg.cyl_radius_gate, "cylinder", i, j))
    used_src, used_tgt, matches = set(), set(), []
    for _, kind, i, j in sorted(candidates, key=lambda x: x[0]):
        if (kind, i) not in used_src and (kind, j) not in used_tgt:
            used_src.add((kind, i))
            used_tgt.add((kind, j))
            matches.append((kind, i, j))
    return matches


def plane_fit_information(P, n, noise):
    """Information matrix of a fitted plane (n, d) in the local coordinates
    (S^2 tangent of n in s2_basis(n) (2), d (1)), from the point-to-plane
    residuals n . p + d of its inliers P with standard deviation `noise`:
    sum_i J_i^T J_i / noise^2 with J_i = [p_i^T B(n), 1]. Treats the range
    noise as if it were along the normal, which overstates the noise of
    oblique points (conservative).
    """
    J = np.hstack([P @ s2_basis(n), np.ones((len(P), 1))])
    return J.T @ J / noise ** 2


def associate_planes(scans, rotations, translations, angle_gate_deg=10.0, offset_gate=0.5):
    """Online landmark association: every extracted plane, mapped to the
    world by its scan's (odometry) pose, joins the closest existing landmark
    inside the gates (one landmark per plane per scan) or starts a new one.
    Arguments:
        scans: list of Scan
        rotations, translations: world pose (R, t) of every scan's sensor
    Returns:
        landmarks: list of [n_w, d_w] initial values
        observations: list of (scan k, plane index i, landmark l)
    """
    landmarks, observations = [], []
    cos_gate = np.cos(np.deg2rad(angle_gate_deg))
    for k, (scan, R, t) in enumerate(zip(scans, rotations, translations)):
        used = set()
        for i, pl in enumerate(scan.planes):
            n_w = R @ pl["n"]
            d_w = pl["d"] - n_w @ t
            best, best_cost = None, np.inf
            for l, (n_l, d_l) in enumerate(landmarks):
                c = n_w @ n_l
                if l in used or c < cos_gate or abs(d_w - d_l) > offset_gate:
                    continue
                cost = np.arccos(min(c, 1.0)) / np.deg2rad(angle_gate_deg) + abs(d_w - d_l) / offset_gate
                if cost < best_cost:
                    best, best_cost = l, cost
            if best is None:
                landmarks.append([n_w, d_w])
                best = len(landmarks) - 1
            used.add(best)
            observations.append((k, i, best))
    return landmarks, observations


def random_perturbation(rng, trans, rot):
    """Tangent vector with translation norm `trans` and rotation norm `rot` in random directions."""
    v, w = rng.normal(size=3), rng.normal(size=3)
    return np.concatenate([trans * v / np.linalg.norm(v), rot * w / np.linalg.norm(w)])


# ---------------------------------------------------------------------------
# Experiment settings and printing (shared by both backends)
# ---------------------------------------------------------------------------

METHODS = ("point_to_point", "point_to_plane", "primitive")
RELATIVE_MOTION = np.array([0.8, 0.2, 0.02, 0.01, -0.01, 0.12])   # source pose in the target frame, [v, omega]


def print_extraction(out, noise_levels, azimuth_counts):
    for key in ("plane_angle_deg_median", "plane_offset_mm_median", "plane_detection", "cyl_detection",
                "false_positives", "cyl_angle_deg_median", "cyl_angle_deg_range_median", "cyl_pos_mm_median",
                "cyl_pos_mm_range_median", "cyl_radius_bias_mm_mean", "cyl_radius_bias_mm_range_mean",
                "cyl_radius_mm_rmse", "cyl_radius_mm_range_rmse"):
        print(f"\n{key}  (rows: range noise {noise_levels} m; cols: points per ring {azimuth_counts})")
        print(np.array2string(out[key], precision=3, suppress_small=True))


def print_basin(out, levels):
    print("\nlevel (m, deg):        " + "  ".join(f"{t:>4.2f},{r:>3.0f}" for t, r in levels))
    for m in METHODS:
        for key, fmt in (("success", "{:9.2f}"), ("trans_mm", "{:9.1f}"), ("rot_deg", "{:9.3f}"),
                         ("iters", "{:9.0f}")):
            print(f"{m:15s} {key:8s}" + "".join(fmt.format(v) for v in out[m][key]))


def print_degeneracy(out):
    for name, res in out.items():
        print(f"\n{name}")
        for m in METHODS:
            r = res[m]
            print(f"  {m:15s} diagnosis at truth: eig min/max {np.median(r['eig_ratio']):.1e}, weakest-vs-axis |cos| "
                  f"{np.median(r['axis_alignment']):.3f}, predicted sigma axis {np.median(r['axis_sigma_mm']):.3g} mm "
                  f"/ lateral {np.median(r['lateral_sigma_mm']):.3g} mm")
            print(f"  {'':15s} outcome: |axis err| median {np.median(np.abs(r['axis_err_mm'])):.1f} mm (RMSE "
                  f"{np.sqrt(np.mean(r['axis_err_mm'] ** 2)):.1f}), trans err median {np.median(r['trans_mm']):.1f} mm,"
                  f" rot err median {np.median(r['rot_deg']):.3f} deg")


def print_landmarks(out):
    print(f"\n{out['n_scans']} scans; landmarks created per run: {out['n_landmarks'].tolist()} (6 distinct planes)")
    for name in ("odometry", "graph"):
        r = out[name]
        print(f"  {name:9s} ATE RMSE median {np.median(r['ate_mm']):7.1f} mm   end-point drift median "
              f"{np.median(r['end_mm']):7.1f} mm, {np.median(r['end_deg']):.3f} deg   (max {np.max(r['end_mm']):.1f} mm)")
