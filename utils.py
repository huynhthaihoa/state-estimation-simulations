import time
import tracemalloc
import numpy as np


def skew(v):
    """Returns the skew-symmetric matrix of a 3-vector, which is used to compute
    cross products and is essential in the exponential map for SO(3).
    For a vector v = [vx, vy, vz], the skew-symmetric matrix is:
    
        [[ 0   -vz   vy]
         [ vz   0   -vx]
         [-vy   vx   0 ]]


    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def true_body_rates(t):
    """Smooth, persistently-exciting angular & linear body-frame rate profile.
    Arguments:
        t: time (s)
    Returns:
        omega: body-frame angular velocity (rad/s)
        v: body-frame linear velocity (m/s)
    """
    omega = np.array([
        0.6 * np.sin(0.7 * t),
        0.5 * np.cos(0.4 * t + 0.3),
        0.8 * np.sin(0.25 * t + 1.0),
    ])
    v = np.array([
        1.0 * np.cos(0.3 * t),
        0.5 * np.sin(0.2 * t),
        0.2 * np.sin(0.5 * t),
    ])
    return omega, v


def pair_key(i, j):
    """Canonical (order-independent) key for an unordered keyframe pair."""
    return (i, j) if i < j else (j, i)


def look_at_rotation(cam_pos, target, up_hint=np.array([0.0, 0.0, 1.0])):
    """Builds a camera-to-world rotation whose local +z axis points from
    cam_pos toward target.
    Arguments:
        cam_pos: camera position in world frame (3,)
        target: point the camera should look at, world frame (3,)
        up_hint: approximate "up" direction in world frame (3,)
    Returns:
        R: (3,3) camera-to-world rotation (columns = camera x,y,z axes in world coords)
    """
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(forward, right)
    return np.column_stack([right, cam_up, forward])


def forward_facing_rotation(forward, up_hint=np.array([0.0, 0.0, 1.0])):
    """Builds a camera-to-world rotation whose local +z axis is `forward`
    (e.g. a path's tangent direction) -- analogous to look_at_rotation, but
    taking the forward direction directly instead of deriving it from a
    target point.
    Arguments:
        forward: forward direction in world frame (3,)
        up_hint: approximate "up" direction in world frame (3,)
    Returns:
        R: (3,3) camera-to-world rotation (columns = camera x,y,z axes in world coords)
    """
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(forward, right)
    return np.column_stack([right, cam_up, forward])


def rotation_matrix_to_quaternion_xyzw(R):
    """Converts a (3,3) rotation matrix to a manifpy-convention [x,y,z,w]
    quaternion (Shepperd's method) -- used only to build ground-truth/
    aligned manif.SE3 poses from a plain rotation matrix; not a Lie-group
    operation itself.
    Arguments:
        R: (3,3) rotation matrix
    Returns:
        quat_xyzw: (4,) array [x, y, z, w]
    """
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    quat = np.array([x, y, z, w])
    return quat / np.linalg.norm(quat)


def quat_multiply(a, b):
    """Hamilton product a (x) b of two scalar-first quaternions.
    Arguments:
        a, b: (4,) quaternions [w, x, y, z]
    Returns:
        (4,) quaternion a (x) b
    """
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_left_matrix(q):
    """4x4 matrix L(q) with q (x) p = L(q) p.
    Arguments:
        q: (4,) quaternion [w, x, y, z]
    Returns:
        (4,4) left-multiplication matrix
    """
    w, x, y, z = q
    return np.array([[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]])


def quat_right_matrix(q):
    """4x4 matrix M(q) with p (x) q = M(q) p.
    Arguments:
        q: (4,) quaternion [w, x, y, z]
    Returns:
        (4,4) right-multiplication matrix
    """
    w, x, y, z = q
    return np.array([[w, -x, -y, -z], [x, w, z, -y], [y, -z, w, x], [z, y, -x, w]])


def quat_exp(phi):
    """Unit quaternion of the rotation vector phi (axis * angle, rad).
    Arguments:
        phi: (3,) rotation vector
    Returns:
        (4,) unit quaternion [w, x, y, z]
    """
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        return np.array([1.0, *(0.5 * phi)])
    return np.array([np.cos(theta / 2), *(np.sin(theta / 2) * phi / theta)])


def quat_to_rotation(q):
    """Homogeneous rotation-matrix form (w^2 - v.v) I + 2 v v^T + 2 w [v]x; equals the
    rotation matrix of q when |q| = 1, and is the polynomial `d_rotate_dq` differentiates.
    Arguments:
        q: (4,) quaternion [w, x, y, z]
    Returns:
        (3,3) matrix
    """
    w, v = q[0], q[1:]
    return (w * w - v @ v) * np.eye(3) + 2.0 * np.outer(v, v) + 2.0 * w * skew(v)


def rotation_to_quat(R):
    """Unit quaternion [w, x, y, z] of a rotation matrix, scalar part >= 0.
    Arguments:
        R: (3,3) rotation matrix
    Returns:
        (4,) unit quaternion
    """
    x, y, z, w = rotation_matrix_to_quaternion_xyzw(R)
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


def d_rotate_dq(q, a):
    """Derivative of quat_to_rotation(q) @ a with respect to q.
    Arguments:
        q: (4,) quaternion [w, x, y, z]
        a: (3,) vector
    Returns:
        (3,4) Jacobian
    """
    w, v = q[0], q[1:]
    col_w = 2.0 * (w * a + np.cross(v, a))
    cols_v = 2.0 * ((v @ a) * np.eye(3) + np.outer(v, a) - np.outer(a, v) - w * skew(a))
    return np.column_stack([col_w, cols_v])


def quat_error_jacobian(q):
    """3x4 map from an additive quaternion error dq to the right rotation error
    dtheta = 2 vec(q^* (x) dq), to first order.
    Arguments:
        q: (4,) unit quaternion [w, x, y, z]
    Returns:
        (3,4) Jacobian
    """
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    return 2.0 * quat_left_matrix(q_conj)[1:, :]


def umeyama_alignment(P_est, P_true):
    """Least-squares similarity (scale, rotation, translation) that best
    maps P_est onto P_true (Umeyama, 1991): P_true ~= s * (R @ P_est.T).T + t.
    Arguments:
        P_est: (N,3) estimated points
        P_true: (N,3) corresponding true points
    Returns:
        s: scale
        R: (3,3) rotation
        t: (3,) translation
    """
    mu_est, mu_true = P_est.mean(axis=0), P_true.mean(axis=0)
    X, Y = P_est - mu_est, P_true - mu_true
    n = P_est.shape[0]
    cov = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_est = (X ** 2).sum() / n
    s = np.trace(np.diag(D) @ S) / var_est
    t = mu_true - s * (R @ mu_est)
    return s, R, t


def landmark_errors(P_true, P_est):
    """Per-landmark position error (m).
    Arguments:
        P_true: (n_landmarks, 3) ground-truth landmark positions
        P_est: (n_landmarks, 3) estimated landmark positions
    Returns:
        err: (n_landmarks,) array of per-landmark position errors (m)
    """
    return np.linalg.norm(P_true - P_est, axis=1)


def unscented_weights(n, alpha, beta, kappa):
    """Van der Merwe scaled unscented-transform weights for an n-dimensional
    tangent state.
    Arguments:
        n: state (tangent-space) dimension
        alpha: spread of the sigma points around the mean
        beta: prior-distribution knowledge (2 is optimal for Gaussian priors)
        kappa: secondary scaling parameter
    Returns:
        lambda_: scaling parameter used to build the sigma-point offsets
        w_m: (2n+1,) mean weights
        w_c: (2n+1,) covariance weights
    """
    lambda_ = alpha ** 2 * (n + kappa) - n
    w_m = np.full(2 * n + 1, 1.0 / (2.0 * (n + lambda_)))
    w_c = w_m.copy()
    w_m[0] = lambda_ / (n + lambda_)
    w_c[0] = lambda_ / (n + lambda_) + (1.0 - alpha ** 2 + beta)
    return lambda_, w_m, w_c


def unscented_sigma_offsets(P, lambda_):
    """Tangent-space sigma-point offsets around a zero mean with covariance P,
    for the scaled unscented transform. Row 0 is the zero (central) offset;
    rows 1..n and n+1..2n are the +/- columns of the scaled Cholesky factor.
    Arguments:
        P: (n,n) covariance (numpy array)
        lambda_: scaling parameter from unscented_weights
    Returns:
        offsets: (2n+1, n) array of tangent-space offset rows
    """
    n = P.shape[0]
    P_sym = 0.5 * (P + P.T) + 1e-9 * np.eye(n)
    L = np.linalg.cholesky((n + lambda_) * P_sym)
    offsets = np.zeros((2 * n + 1, n))
    offsets[1:n + 1] = L.T
    offsets[n + 1:2 * n + 1] = -L.T
    return offsets


def qr_insert_row(R, d, a_row, b_val):
    """Givens-rotation insertion of one new whitened measurement row into an
    existing upper-triangular square-root-information system, in place --
    the core "reuse the previous factorization" step behind incremental
    (iSAM-style) smoothing: absorbing one new row costs O(m) work touching
    only the columns from its first nonzero onward, not an O(m^3)
    refactorization of the whole system.
    Arguments:
        R: (m,m) upper-triangular sqrt-information matrix (modified in place)
        d: (m,) transformed right-hand side (modified in place)
        a_row: (m,) new row's coefficients (not modified; consumed via a local copy)
        b_val: scalar, the new row's right-hand side
    Returns:
        R, d: the same arrays passed in, returned for convenience/chaining
    """
    m = R.shape[0]
    a, b = a_row.copy(), b_val
    for c in range(m):
        if abs(a[c]) < 1e-14:
            continue
        r = np.hypot(R[c, c], a[c])
        cos_, sin_ = R[c, c] / r, a[c] / r
        R[c, c:], a[c:] = cos_ * R[c, c:] + sin_ * a[c:], -sin_ * R[c, c:] + cos_ * a[c:]
        d[c], b = cos_ * d[c] + sin_ * b, -sin_ * d[c] + cos_ * b
    return R, d


def symbolic_eliminate(n_vars, edges, order):
    """Symbolic variable elimination (the "elimination game"): the structural
    half of building a Bayes tree from a factor graph's sparsity pattern --
    which variables end up depending on which others after elimination,
    without touching any numeric factor values. Eliminating a variable
    connects all of its still-uneliminated neighbors to each other (fill-in),
    then removes it from the graph; the clique this produces is
    {variable} union {its neighbors at elimination time} (the separator). A
    clique's parent is whichever separator member is eliminated next -- the
    clique that will absorb this one's separator into its own.
    Arguments:
        n_vars: number of variables (0..n_vars-1)
        edges: iterable of (i, j) pairs, the factor graph's sparsity pattern
        order: sequence of all n_vars variable indices, the elimination order
               (order[0] eliminated first; order[-1] becomes the tree's root)
    Returns:
        separator: dict {variable: sorted list of separator variables}
        parent: dict {variable: parent variable index, or None for the root}
    """
    adjacency = {v: set() for v in range(n_vars)}
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)

    position = {v: k for k, v in enumerate(order)}
    separator, parent = {}, {}
    for v in order:
        neighbors = adjacency[v]
        separator[v] = sorted(neighbors)
        for a in neighbors:
            adjacency[a].update(neighbors - {a})
            adjacency[a].discard(v)
        parent[v] = min(neighbors, key=lambda u: position[u]) if neighbors else None
    return separator, parent


def bayes_tree_affected_path(parent, touched_vars):
    """Which cliques a set of newly touched variables invalidates: every
    clique on the root-ward path from each touched variable, via `parent`
    pointers from `symbolic_eliminate` -- Bayes-tree information only ever
    flows toward the root, so a change at a leaf can only affect its
    ancestors, never a sibling subtree.
    Arguments:
        parent: {variable: parent variable index or None}, from symbolic_eliminate
        touched_vars: iterable of variable indices a new factor touches
    Returns:
        affected: set of variable indices on the union of root-ward paths
    """
    affected = set()
    for v in touched_vars:
        while v is not None and v not in affected:
            affected.add(v)
            v = parent[v]
    return affected


def measure_performance(fn, *args, n_steps, **kwargs):
    """Runs `fn` twice -- once cleanly to measure wall-clock time, once more
    with tracemalloc active to measure peak memory -- and reports the
    former as a per-step average, the latter as a whole-call total.
    Two separate runs, not one: tracemalloc's own bookkeeping overhead is
    substantial and uneven across different algorithms (measured
    empirically at 2.7x-5.8x depending on how much allocation each one
    does), which would otherwise skew exactly the cross-method timing
    comparison this function exists to make fair. (Safe to call fn twice
    for every caller in this repo: none of them mutate shared state in a
    way a second call would corrupt, and any RNG argument they take is a
    fresh, single-use generator whose state nothing reads afterward.)
    Time is normalized by n_steps because it's a throughput quantity --
    meaningfully comparable per-step regardless of how much work each
    method does per step. Peak memory deliberately is *not* normalized
    this way: it's a high-water mark, not a rate, so dividing it by
    n_steps would make an O(1)-memory algorithm's reported number shrink
    as the trajectory grows, the opposite of what "resource usage" should
    mean here.
    Arguments:
        fn: callable to run and measure
        *args, **kwargs: forwarded to fn (twice)
        n_steps: number of trajectory steps, used to normalize the time metric
    Returns:
        result: fn(*args, **kwargs)'s return value (from the timing run)
        avg_time_per_step: wall-clock time / n_steps (s)
        peak_mem: peak traced memory over the whole call (bytes)
    """
    t_start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t_start

    tracemalloc.start()
    fn(*args, **kwargs)
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, elapsed / n_steps, peak_mem


def averaged_nees_bounds(dof, n_trials, z=1.959964):
    """Two-sided 95% acceptance interval for NEES averaged over n_trials
    independent runs. n_trials * (average NEES) is chi-squared with
    dof * n_trials degrees of freedom; its quantiles come from the
    Wilson-Hilferty approximation (accurate to well under 1% at the
    hundreds of degrees of freedom used here), avoiding a scipy dependency.
    The single-run bound (chi-squared with dof degrees of freedom) does not
    apply to an average.
    Arguments:
        dof: state dimension
        n_trials: number of Monte Carlo trials averaged
        z: standard-normal quantile (1.96 for a two-sided 95% interval)
    Returns:
        (lower, upper): bounds on the averaged NEES
    """
    k = dof * n_trials
    c = 2.0 / (9.0 * k)
    return tuple(k * (1.0 - c + s * z * np.sqrt(c)) ** 3 / n_trials for s in (-1.0, 1.0))


def nees(err, P):
    """Normalized Estimation Error Squared, err^T P^-1 err. A consistent
    filter's NEES averages to the state dimension over many independent
    trials; a persistently larger value means the reported covariance P is
    too small (overconfident) for the errors the filter actually makes.
    Arguments:
        err: estimation error (n,), e.g. x_true - x_est
        P: estimated covariance (n,n)
    Returns:
        scalar NEES value
    """
    return float(err @ np.linalg.solve(P, err))

class Quaternion:
    """Simple quaternion class for convenience in tests and examples."""
    def __init__(self, w, x, y, z):
        self.q = np.array([w, x, y, z])

    @classmethod
    def from_array(cls, arr):
        return cls(arr[0], arr[1], arr[2], arr[3])

    @classmethod
    def from_rotation(cls, R):
        return cls.from_array(rotation_to_quat(R))
    
    def as_array(self):
        return self.q

    def __mul__(self, other):
        return Quaternion.from_array(quat_multiply(self.q, other.q))

    def exp(self, phi):
        return Quaternion.from_array(quat_exp(phi))
    
    def left_matrix(self):
        return quat_left_matrix(self.q)
    
    def right_matrix(self):
        return quat_right_matrix(self.q)
    
    def to_rotation(self):
        return quat_to_rotation(self.q)
    
    def derivative_rotate(self, a):
        return d_rotate_dq(self.q, a)
    
    def error_jacobian(self):
        return quat_error_jacobian(self.q)
    
    def conjugate(self):
        w, x, y, z = self.q
        return Quaternion(w, -x, -y, -z)

    def normalize(self):
        norm = np.linalg.norm(self.q)
        if norm > 0:
            self.q /= norm