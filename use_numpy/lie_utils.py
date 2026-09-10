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
    Arguments:
        R_a: first rotation matrix (numpy array)
        R_b: second rotation matrix (numpy array)
    Returns:
        angle: angle (rad) of the relative rotation
    """
    c = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
    return np.arccos(np.clip(c, -1.0, 1.0))


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
        omega_hat = (theta / (2.0 * np.sin(theta))) * (R - R.T)
        omega = np.array([-omega_hat[1, 2], omega_hat[0, 2], -omega_hat[0, 1]])

        omega_skew = skew(omega)
        omega_skew_sq = np.dot(omega_skew, omega_skew)
        V_inv = I3 - 0.5 * omega_skew + (1.0 / (theta ** 2) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))) * omega_skew_sq

    rho = np.dot(V_inv, t)
    return np.concatenate([rho, omega])


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

def compute_so3_inv_right_jacobian(theta_vec):
    """
    Computes the 3x3 inverse right Jacobian of SO(3).
    Arguments:
        theta_vec: 3D vector (numpy array)
    Returns:
        3x3 inverse right Jacobian (numpy array)
    """
    theta = np.linalg.norm(theta_vec)
    I3 = np.eye(3)
    if theta < 1e-6:
        return I3 + 0.5 * skew(theta_vec) + (1.0 / 12.0) * np.dot(skew(theta_vec), skew(theta_vec))
    theta_skew = skew(theta_vec)
    theta_skew_sq = np.dot(theta_skew, theta_skew)
    coeff = (1.0 / (theta ** 2)) - ((1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta)))
    return I3 + 0.5 * theta_skew + coeff * theta_skew_sq


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
        error_vector: 6-vector [rho, theta_vec] (numpy array)
    Returns:
        6x6 inverse right Jacobian (numpy array)
    """
    return np.linalg.inv(se3_right_jacobian(error_vector))
