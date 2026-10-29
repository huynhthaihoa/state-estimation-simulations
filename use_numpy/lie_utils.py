import numpy as np
from state_estimation_simulations.utils import skew


def so3_exp(omega):
    """
    Rodrigues' formula: exact exponential map for a constant body rate over dt.
    Arguments:
        omega: 3-vector of angular velocity (rad/s) (numpy array)
    Returns:
        3x3 rotation matrix (numpy array)
    """
    theta = np.linalg.norm(omega)
    I3 = np.eye(3)
    if theta < 1e-6:
        return I3 + skew(omega)
    omega_skew = skew(omega)
    return I3 + (np.sin(theta) / theta) * omega_skew + ((1.0 - np.cos(theta)) / (theta ** 2)) * np.dot(omega_skew, omega_skew)


def so3_right_jacobian(omega):
    """
    The 3x3 (direct, non-inverse) right Jacobian of SO(3).
    Arguments:
        omega: 3-vector (numpy array)
    Returns:
        3x3 SO(3) right Jacobian (numpy array)
    """
    theta = np.linalg.norm(omega)
    I3 = np.eye(3)
    if theta < 1e-6:
        return I3 - 0.5 * skew(omega)
    omega_skew = skew(omega)
    omega_skew_sq = np.dot(omega_skew, omega_skew)
    return I3 - ((1.0 - np.cos(theta)) / (theta ** 2)) * omega_skew + ((theta - np.sin(theta)) / (theta ** 3)) * omega_skew_sq


def rotation_geodesic_error(R_a, R_b):
    """
    Angle (rad) of the relative rotation between two rotation matrices.
    Uses atan2(sin, cos) rather than arccos(cos): arccos has a precision
    floor of ~sqrt(2*eps) ~ 2e-8 rad near zero (its derivative blows up at
    1), whereas the skew part of R gives the sine to full relative
    precision, so tiny angles come out right instead of as 0 or 2.1e-8.
    Arguments:
        R_a: first rotation matrix (numpy array)
        R_b: second rotation matrix (numpy array)
    Returns:
        angle: angle (rad) of the relative rotation
    """
    W = R_a.T @ R_b
    c = (np.trace(W) - 1.0) / 2.0
    s = 0.5 * np.linalg.norm([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]])
    return np.arctan2(s, c)


def se3_exp(xi):
    """
    The Exponential Map for SE(3). Maps a 6-vector [v,omega] to a 4x4 matrix.
    Arguments:
        xi: 6-vector [v,omega] (numpy array)
    Returns:
        4x4 SE(3) transform (numpy array)
    """
    v = xi[0:3]
    omega = xi[3:6]
    theta = np.linalg.norm(omega)
    I3 = np.eye(3)
    T = np.eye(4)

    if theta < 1e-6:
        R = I3 + skew(omega)
        V = I3 + 0.5 * skew(omega)
    else:
        omega_skew = skew(omega)
        omega_skew_sq = np.dot(omega_skew, omega_skew)
        R = I3 + (np.sin(theta) / theta) * omega_skew + ((1.0 - np.cos(theta)) / (theta ** 2)) * omega_skew_sq
        V = I3 + ((1.0 - np.cos(theta)) / (theta ** 2)) * omega_skew + ((theta - np.sin(theta)) / (theta ** 3)) * omega_skew_sq

    T[0:3, 0:3] = R
    T[0:3, 3] = np.dot(V, v)
    return T


def se3_log(T):
    """
    The Logarithmic Map for SE(3). Extracts a 6-vector [v,omega] from a 4x4 matrix.
    Arguments:
        T: 4x4 SE(3) transform (numpy array)
    Returns:
        6-vector [v,omega] (numpy array)
    """
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    I3 = np.eye(3)

    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-6:
        omega = np.zeros(3)
        V_inv = I3
    else:
        if theta > np.pi - 1e-3:
            # Near theta=pi, the classic (R - R^T)/(2 sin(theta)) formula
            # below divides by sin(theta) -> 0, cancelling catastrophically
            # against a numerator that vanishes just as fast -- empirically,
            # its error is already ~1e-3 by theta = pi - 1e-6, well before
            # theta is actually close enough to pi for that to be
            # unavoidable. Rodrigues' formula gives, for *any* theta:
            #   Sym(R) := (R + R^T)/2 = cos(theta)*I + (1-cos(theta))*axis@axis.T
            # (the sin(theta)*skew(axis) term is antisymmetric and cancels
            # here), so axis@axis.T = (Sym(R) - cos(theta)*I)/(1-cos(theta))
            # exactly. That division is safe here (1-cos(theta) ~ 2 near
            # pi) -- unlike near theta=0, where it vanishes *quadratically*
            # and this formula would need its own small-angle branch; that's
            # exactly why this is gated to the near-pi case only, not used
            # in place of the regular branch below. Accurate to machine
            # precision down to the inherent (unavoidable) ~1e-8
            # conditioning floor of arccos itself within ~1e-7 of exactly
            # pi, where the log map's axis becomes a genuine, mathematically
            # real sign ambiguity, not a numerical one.
            axis_outer = (0.5 * (R + R.T) - cos_theta * I3) / (1.0 - cos_theta)
            diag = np.clip(np.diag(axis_outer), 0.0, None)
            k = int(np.argmax(diag))
            axis = axis_outer[:, k] / np.sqrt(diag[k])
            axis = axis / np.linalg.norm(axis)
            R_antisym = R - R.T
            sign_hint = np.array([-R_antisym[1, 2], R_antisym[0, 2], -R_antisym[0, 1]])
            if np.dot(axis, sign_hint) < 0:
                axis = -axis
            omega = axis * theta
        else:
            omega_hat = (theta / (2.0 * np.sin(theta))) * (R - R.T)
            omega = np.array([-omega_hat[1, 2], omega_hat[0, 2], -omega_hat[0, 1]])

        omega_skew = skew(omega)
        omega_skew_sq = np.dot(omega_skew, omega_skew)
        V_inv = I3 - 0.5 * omega_skew + (1.0 / (theta ** 2) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))) * omega_skew_sq

    v = np.dot(V_inv, t)
    return np.concatenate([v, omega])


def se3_inv(T):
    """
    Rigid inverse of a 4x4 SE(3) transform.
    Arguments:
        T: 4x4 SE(3) transform (numpy array)
    Returns:
        4x4 inverted SE(3) transform (numpy array)
    """
    R, t = T[0:3, 0:3], T[0:3, 3]
    T_inv = np.eye(4)
    T_inv[0:3, 0:3] = R.T
    T_inv[0:3, 3] = -R.T @ t
    return T_inv


def se3_adjoint(T):
    """
    The 6x6 SE(3) adjoint, block form [[R, skew(t)@R],[0, R]] for the
    [v,omega]-ordered tangent convention used throughout this codebase.
    Arguments:
        T: 4x4 SE(3) transform (numpy array)
    Returns:
        6x6 SE(3) adjoint (numpy array)
    """
    R, t = T[0:3, 0:3], T[0:3, 3]
    Adj = np.zeros((6, 6))
    Adj[0:3, 0:3] = R
    Adj[0:3, 3:6] = skew(t) @ R
    Adj[3:6, 3:6] = R
    return Adj

def compute_so3_inv_right_jacobian(omega):
    """
    Computes the 3x3 inverse right Jacobian of SO(3).
    Arguments:
        omega: 3-vector (numpy array)
    Returns:
        3x3 inverse right Jacobian (numpy array)
    """
    theta = np.linalg.norm(omega)
    I3 = np.eye(3)
    if theta < 1e-6:
        return I3 + 0.5 * skew(omega) + (1.0 / 12.0) * np.dot(skew(omega), skew(omega))
    omega_skew = skew(omega)
    omega_skew_sq = np.dot(omega_skew, omega_skew)
    coeff = (1.0 / (theta ** 2)) - ((1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta)))
    return I3 + 0.5 * omega_skew + coeff * omega_skew_sq


def so3_log(R):
    """Logarithm of a rotation matrix: the rotation vector phi with
    so3_exp(phi) = R (angle in [0, pi)).
    Arguments:
        R: (3,3) rotation matrix
    Returns:
        phi: (3,) rotation vector
    """
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    w = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-9:
        return w
    return theta / np.sin(theta) * w


def so3_left_jacobian(phi):
    """Left Jacobian of SO(3), J_l(phi) = sum_k (phi^)^k / (k+1)!.
    Arguments:
        phi: (3,) rotation vector
    Returns:
        J: (3,3) left Jacobian
    """
    theta = np.linalg.norm(phi)
    K = skew(phi)
    if theta < 1e-9:
        return np.eye(3) + 0.5 * K
    return (np.eye(3) + (1.0 - np.cos(theta)) / theta ** 2 * K
            + (theta - np.sin(theta)) / theta ** 3 * K @ K)


def se23_make(R, v, p):
    """Builds the (5,5) SE_2(3) matrix from its rotation, velocity and position.
    Arguments:
        R: (3,3) rotation, v: (3,) velocity, p: (3,) position
    Returns:
        X: (5,5) SE_2(3) element
    """
    X = np.eye(5)
    X[0:3, 0:3] = R
    X[0:3, 3] = v
    X[0:3, 4] = p
    return X


def se23_parts(X):
    """Splits an SE_2(3) matrix into (R, v, p) copies.
    Arguments:
        X: (5,5) SE_2(3) element
    Returns:
        R: (3,3), v: (3,), p: (3,)
    """
    return X[0:3, 0:3].copy(), X[0:3, 3].copy(), X[0:3, 4].copy()


def se23_inv(X):
    """Closed-form inverse: (R, v, p)^-1 = (R^T, -R^T v, -R^T p).
    Arguments:
        X: (5,5) SE_2(3) element
    Returns:
        X_inv: (5,5) inverse
    """
    R, v, p = se23_parts(X)
    return se23_make(R.T, -R.T @ v, -R.T @ p)


def se23_exp(xi):
    """Exponential map, xi = (xi_R, xi_v, xi_p) -> (Exp(xi_R), J_l xi_v, J_l xi_p).
    Arguments:
        xi: (9,) tangent vector
    Returns:
        X: (5,5) SE_2(3) element
    """
    J = so3_left_jacobian(xi[0:3])
    return se23_make(so3_exp(xi[0:3]), J @ xi[3:6], J @ xi[6:9])


def se23_log(X):
    """Logarithm map, the inverse of se23_exp.
    Arguments:
        X: (5,5) SE_2(3) element
    Returns:
        xi: (9,) tangent vector
    """
    R, v, p = se23_parts(X)
    phi = so3_log(R)
    J_inv = np.linalg.inv(so3_left_jacobian(phi))
    return np.concatenate([phi, J_inv @ v, J_inv @ p])


def se23_adjoint(X):
    """(9,9) adjoint, Ad_X xi = vee(X hat(xi) X^-1):
    [[R, 0, 0], [v^ R, R, 0], [p^ R, 0, R]].
    Arguments:
        X: (5,5) SE_2(3) element
    Returns:
        Ad: (9,9) adjoint matrix
    """
    R, v, p = se23_parts(X)
    Ad = np.zeros((9, 9))
    Ad[0:3, 0:3] = R
    Ad[3:6, 0:3] = skew(v) @ R
    Ad[3:6, 3:6] = R
    Ad[6:9, 0:3] = skew(p) @ R
    Ad[6:9, 6:9] = R
    return Ad


def sek3_make(R, cols):
    """Builds the (3+K, 3+K) SE_K(3) matrix [[R, c_1 .. c_K], [0, I_K]].
    SE_2(3) is K = 2 with cols = (v, p); a contact-aided state adds one
    column per foot (SE_{K+2}(3) with cols = (v, p, d_1 .. d_K)).
    Arguments:
        R: (3,3) rotation
        cols: (K,3) translation-like columns
    Returns:
        X: (3+K, 3+K) SE_K(3) element
    """
    cols = np.asarray(cols, dtype=float).reshape(-1, 3)
    X = np.eye(3 + len(cols))
    X[0:3, 0:3] = R
    X[0:3, 3:] = cols.T
    return X


def sek3_parts(X):
    """Splits an SE_K(3) matrix into (R, cols) copies.
    Arguments:
        X: (3+K, 3+K) SE_K(3) element
    Returns:
        R: (3,3), cols: (K,3)
    """
    return X[0:3, 0:3].copy(), X[0:3, 3:].T.copy()


def sek3_inv(X):
    """Closed-form inverse: (R, c_k)^-1 = (R^T, -R^T c_k).
    Arguments:
        X: (3+K, 3+K) SE_K(3) element
    Returns:
        X_inv: (3+K, 3+K) inverse
    """
    R, cols = sek3_parts(X)
    return sek3_make(R.T, -cols @ R)


def sek3_exp(xi):
    """Exponential map, xi = (xi_R, xi_1 .. xi_K) -> (Exp(xi_R), J_l xi_k).
    Arguments:
        xi: (3+3K,) tangent vector
    Returns:
        X: (3+K, 3+K) SE_K(3) element
    """
    J = so3_left_jacobian(xi[0:3])
    return sek3_make(so3_exp(xi[0:3]), xi[3:].reshape(-1, 3) @ J.T)


def sek3_log(X):
    """Logarithm map, the inverse of sek3_exp.
    Arguments:
        X: (3+K, 3+K) SE_K(3) element
    Returns:
        xi: (3+3K,) tangent vector
    """
    R, cols = sek3_parts(X)
    phi = so3_log(R)
    J_inv = np.linalg.inv(so3_left_jacobian(phi))
    return np.concatenate([phi, (cols @ J_inv.T).reshape(-1)])


def sek3_adjoint(X):
    """(3+3K, 3+3K) adjoint, Ad_X xi = vee(X hat(xi) X^-1): R on the
    diagonal, c_k^ R in the first block column of row block k.
    Arguments:
        X: (3+K, 3+K) SE_K(3) element
    Returns:
        Ad: (3+3K, 3+3K) adjoint matrix
    """
    R, cols = sek3_parts(X)
    n = 3 + 3 * len(cols)
    Ad = np.kron(np.eye(n // 3), R)
    for k, c in enumerate(cols):
        Ad[3 + 3 * k:6 + 3 * k, 0:3] = skew(c) @ R
    return Ad


def se3_ad(xi):
    """
    The 6x6 "little adjoint" ad(xi) of an SE(3) tangent vector -- the linear
    operator representing the Lie bracket [xi, .], used to build the right
    Jacobian from its definition. For the [v,omega] (translation-first)
    convention used throughout this codebase:
        ad(xi) = [[skew(omega), skew(v)   ],
                  [0,            skew(omega)]]
    Arguments:
        xi: 6-vector [v,omega] (numpy array)
    Returns:
        6x6 ad(xi) matrix (numpy array)
    """
    v, omega = xi[0:3], xi[3:6]
    omega_skew = skew(omega)
    ad = np.zeros((6, 6))
    ad[0:3, 0:3] = omega_skew
    ad[0:3, 3:6] = skew(v)
    ad[3:6, 3:6] = omega_skew
    return ad


def se3_right_jacobian(xi, n_terms=18):
    """
    The direct (non-inverse) SE(3) right Jacobian Jr(xi), computed from its
    definition as the alternating factorial series in the tangent-space
    adjoint operator: Jr(xi) = sum_{n=0}^inf (-ad(xi))^n / (n+1)!. Unlike a
    closed-form trigonometric expression, this series is exact by
    construction and needs no separate small-angle branch -- it degrades
    gracefully to the identity as xi -> 0 -- and converges fast (factorial-
    dominated) for any tangent-vector magnitude used in this codebase.
    Arguments:
        xi: 6-vector [v,omega] (numpy array)
        n_terms: number of series terms (18 already gives double-precision
                 accuracy well beyond any xi magnitude used here)
    Returns:
        6x6 SE(3) right Jacobian (numpy array)
    """
    ad = se3_ad(xi)
    power = np.eye(6)
    Jr = np.zeros((6, 6))
    sign, fact = 1.0, 1.0
    for n in range(n_terms):
        fact *= (n + 1)
        Jr += sign * power / fact
        power = power @ ad
        sign = -sign
    return Jr


def compute_se3_inv_right_jacobian(error_vector):
    """
    Computes the 6x6 inverse right Jacobian for an SE(3) error vector, as the
    matrix inverse of se3_right_jacobian -- guaranteed consistent with it by
    construction, rather than a second, independently-derived closed form
    that could silently drift out of sync with it.
    Arguments:
        error_vector: 6-vector [v,omega] (numpy array)
    Returns:
        6x6 inverse right Jacobian (numpy array)
    """
    return np.linalg.inv(se3_right_jacobian(error_vector))


class SO3:
    """Groups the SO(3) helpers above under short, consistent names
    (SO3.exp, SO3.hat, ...). Every method is the module-level function
    itself (a staticmethod alias), taking and returning the same plain numpy
    arrays -- nothing here holds state, and existing callers of the
    functions are unaffected. Not an object API on rotations: use_manif/
    already demonstrates that style via manifpy.
    """
    hat = staticmethod(skew)
    exp = staticmethod(so3_exp)
    right_jacobian = staticmethod(so3_right_jacobian)
    inv_right_jacobian = staticmethod(compute_so3_inv_right_jacobian)
    geodesic_error = staticmethod(rotation_geodesic_error)
    log = staticmethod(so3_log)
    left_jacobian = staticmethod(so3_left_jacobian)


class SE3:
    """Groups the SE(3) helpers above under short, consistent names
    (SE3.exp, SE3.log, SE3.inv, ...), [v,omega] tangent convention. Same
    staticmethod-alias design as SO3: identical functions on plain numpy
    arrays, no state.
    """
    exp = staticmethod(se3_exp)
    log = staticmethod(se3_log)
    inv = staticmethod(se3_inv)
    adjoint = staticmethod(se3_adjoint)
    ad = staticmethod(se3_ad)
    right_jacobian = staticmethod(se3_right_jacobian)
    inv_right_jacobian = staticmethod(compute_se3_inv_right_jacobian)


class SE23:
    """Groups the SE_2(3) helpers above under short, consistent names
    (SE23.exp, SE23.log, SE23.inv, ...), [xi_R, xi_v, xi_p] tangent
    convention (rotation first, unlike SE3). Same staticmethod-alias design
    as SO3: identical functions on plain numpy arrays, no state.
    """
    make = staticmethod(se23_make)
    parts = staticmethod(se23_parts)
    inv = staticmethod(se23_inv)
    exp = staticmethod(se23_exp)
    log = staticmethod(se23_log)
    adjoint = staticmethod(se23_adjoint)


class SEK3:
    """Groups the SE_K(3) helpers above under short, consistent names
    (SEK3.exp, SEK3.log, ...), [xi_R, xi_1 .. xi_K] tangent convention, so
    SE_2(3) is the K = 2 case of SE23. Same staticmethod-alias design as SO3.
    """
    make = staticmethod(sek3_make)
    parts = staticmethod(sek3_parts)
    inv = staticmethod(sek3_inv)
    exp = staticmethod(sek3_exp)
    log = staticmethod(sek3_log)
    adjoint = staticmethod(sek3_adjoint)
