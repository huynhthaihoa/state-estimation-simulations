'''
Observability and consistency in planar EKF-SLAM: why the standard EKF becomes
overconfident, and how FEJ (first-estimates Jacobian) and the invariant EKF
avoid it.

Model (the unicycle SLAM problem of Barrau & Bonnabel 2016, eqs. 9 and 11, and
the scenario of their Sec. 7.1, which is "deliberately similar" to the one of
Huang, Mourikis & Roumeliotis 2008). State X = [x, y, theta, p_1, ..., p_K]:
the robot pose and K static landmark positions, all in one world frame.
Odometry gives, per step, a forward displacement d and a rotation w; the robot
moves by x += R(theta) [d, 0], theta += w. Each visible landmark is measured in
the robot frame, z = h(R(theta)^T (p_j - x)) + noise, with h the identity
(relative position, the default of both papers) or range-bearing (`--meas`).
The world frame is fixed only by the robot's initial pose, so shifting or
rotating the whole scene (robot + all landmarks) changes no measurement:
3 unobservable directions, a translation (2) and a rotation (1).

Truth: a deterministic circle (the model's own discrete dynamics with the
nominal odometry) and a fixed random landmark map. The odometry readings and
the landmark measurements carry fresh noise in every Monte Carlo trial, so the
filter's error is driven by the same noise it models; the filters are never
told the truth (except the "ideal" reference below).

Filters (all share the mean propagation and the Joseph-form Kalman update; they
differ ONLY in where the Jacobians are evaluated, or in the error variable):

  standard  Jacobians at the latest estimates (Huang et al. 2008, eqs. 6, 13, 14)
  fej       FEJ-EKF (Huang et al. 2008, Sec. III, eqs. 29, 30): the propagation
            Jacobian uses the propagated robot position of the previous step
            instead of its updated value, and a landmark's measurement Jacobian
            uses its position estimate from the first time it was seen
  iekf      invariant EKF-SLAM (Barrau & Bonnabel 2016, Algorithm 3): error
            xi = [theta-theta^, x-x^-(theta-theta^) J x^, p_j-p^_j-(theta-theta^) J p^_j],
            F = I, measurement Jacobian independent of the robot position; the
            update is applied on the group (their eq. 22)
  ideal     the oracle EKF with Jacobians at the TRUE state (not realizable;
            the consistency reference)
  fej_phi, fej_h   ablations of fej: only the propagation rule / only the
            measurement rule

Landmarks are initialized from the first measurement with a very large prior
variance (`--init-landmark-std`, "very high compared to the size of the map",
Barrau & Bonnabel Sec. 7.1) and updated together with the others.

Scores (Monte Carlo over `--n-trials`, every noise source redrawn per trial):
  - robot pose NEES (3 DoF) per step and the 95% band of an average over the
    trials (`averaged_nees_bounds`); landmark NEES (2 DoF) averaged over seen
    landmarks, trials and steps; RMSE of robot position, heading and landmarks;
  - the filter's reported heading standard deviation next to the true heading
    RMS error in the last lap;
  - spurious information along the global rotation: the filter's own
    information gain, summed over the run, along the rotation direction carried
    through its own propagation Jacobians (Barrau & Bonnabel, Thm. 2): the
    update adds dX^T H^T R^-1 H dX to the information matrix, which must be 0
    for an unobservable dX. Reported also as the heading standard deviation
    that information alone would imply (1 / sqrt(sum)).
  - the local observability matrix of each filter's linearized system on a
    small scene where every landmark is always visible: rank, null-space
    dimension and the smallest singular values.

The findings are in docs/foundations/observability.md.
'''

import argparse
from dataclasses import dataclass, fields

import numpy as np
import matplotlib.pyplot as plt

from state_estimation_simulations.utils import measure_performance, averaged_nees_bounds

J2 = np.array([[0.0, -1.0], [1.0, 0.0]])
KINDS = ("standard", "fej", "iekf", "ideal")
ABLATIONS = ("fej_phi", "fej_h")
ALL_KINDS = KINDS + ABLATIONS


@dataclass
class Params:
    '''Scenario and noise parameters; defaults follow Barrau & Bonnabel Sec. 7.1 unless a
    comment says "design".'''
    speed: float = 1.0               # robot speed (m/s)
    turn_rate_deg: float = 9.0       # robot turn rate (deg/s)
    dt: float = 1.0                  # odometry and measurement period (s); design: the paper gives only the 1 Hz measurement
    loops: int = 10                  # number of laps
    n_landmarks: int = 20            # number of landmarks
    ring_halfwidth: float = 4.0      # design: landmarks drawn uniformly in an annulus of this half-width around the path (m)
    sense_range: float = 5.0         # a landmark is measured only if within this distance of the robot (m)
    wheel_noise_frac: float = 0.02   # std of each wheel's speed reading as a fraction of the speed
    wheel_base: float = 0.5          # distance between the drive wheels (m)
    meas: str = "position"           # "position" (relative position) or "range_bearing"
    pos_noise_std: float = 0.1       # relative-position noise std, per axis (m)
    range_noise_std: float = 0.1     # design: range noise std (m), only for range_bearing
    bearing_noise_std_deg: float = 1.0   # design: bearing noise std (deg), only for range_bearing
    init_landmark_std: float = 100.0     # design: prior std of a new landmark (m); "very high compared to the map"
    init_pos_std: float = 0.0        # initial robot position std (m); the paper uses 0
    init_heading_std: float = 0.0    # initial robot heading std (rad); the paper uses 0
    heading_noise_scale: float = 1.0     # multiplies the odometry rotation noise (1 = the paper's value)

    @property
    def omega(self):
        return np.deg2rad(self.turn_rate_deg) * self.dt

    @property
    def n_steps(self):
        return int(round(self.loops * 2.0 * np.pi / self.omega))

    @property
    def sigma_d(self):
        """Std of the forward displacement noise per step (m): the mean of two wheels, sigma/sqrt(2)."""
        return self.wheel_noise_frac * self.speed / np.sqrt(2.0) * self.dt

    @property
    def sigma_w(self):
        """Std of the rotation noise per step (rad): the difference of two wheels over the base, sqrt(2) sigma / a."""
        return self.heading_noise_scale * np.sqrt(2.0) * self.wheel_noise_frac * self.speed / self.wheel_base * self.dt

    def meas_cov(self):
        if self.meas == "position":
            return self.pos_noise_std ** 2 * np.eye(2)
        if self.meas == "range_bearing":
            return np.diag([self.range_noise_std ** 2, np.deg2rad(self.bearing_noise_std_deg) ** 2])
        raise ValueError("meas must be 'position' or 'range_bearing'")


# ----------------------------------------------------------------------------------------------
# Small geometry helpers
# ----------------------------------------------------------------------------------------------

def rot(a):
    """2x2 rotation matrix R(a) for angle a (rad)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def wrap(a):
    """Wraps an angle (or an array of angles) to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def B_matrix(a):
    """B(a) = [[sin a / a, -(1 - cos a) / a], [(1 - cos a) / a, sin a / a]] of the SE(2) exponential
    (Barrau & Bonnabel eq. 22); the limit a -> 0 is the identity.
    Arguments:
        a: angle (rad)
    Returns:
        (2, 2) matrix
    """
    if abs(a) < 1e-9:
        return np.eye(2) + 0.5 * a * J2
    return (np.sin(a) * np.eye(2) + (1.0 - np.cos(a)) * J2) / a


def meas_h(kind, rel):
    """Measurement function h~ applied to the robot-frame landmark position rel (m).
    Arguments:
        kind: "position" (identity) or "range_bearing" (range m, bearing rad)
        rel: (2,) landmark position in the robot frame (m)
    Returns:
        (2,) predicted measurement
    """
    if kind == "position":
        return rel.copy()
    return np.array([np.hypot(rel[0], rel[1]), np.arctan2(rel[1], rel[0])])


def meas_jac(kind, rel):
    """Jacobian of meas_h with respect to rel, (2, 2)."""
    if kind == "position":
        return np.eye(2)
    r2 = rel[0] ** 2 + rel[1] ** 2
    r = np.sqrt(r2)
    return np.array([[rel[0] / r, rel[1] / r], [-rel[1] / r2, rel[0] / r2]])


def meas_inv(kind, z):
    """Robot-frame landmark position from a measurement (inverse of meas_h)."""
    if kind == "position":
        return z.copy()
    return z[0] * np.array([np.cos(z[1]), np.sin(z[1])])


def meas_resid(kind, z, zhat):
    """Innovation z - zhat; the bearing difference is wrapped."""
    r = z - zhat
    if kind == "range_bearing":
        r[1] = wrap(r[1])
    return r


def f_motion(x, d, w):
    """Odometry propagation of the state vector (landmarks are static).
    Arguments:
        x: (3+2K,) state [x, y, theta, landmarks...]
        d: forward displacement over the step (m)
        w: rotation over the step (rad)
    Returns:
        (3+2K,) propagated state
    """
    y = x.copy()
    y[:2] = x[:2] + rot(x[2]) @ np.array([d, 0.0])
    y[2] = x[2] + w
    return y


def meas_state(kind, x, j):
    """h~(R(theta)^T (p_j - [x, y])) of landmark j for the state vector x (used by the finite-difference tests)."""
    return meas_h(kind, rot(x[2]).T @ (x[3 + 2 * j:5 + 2 * j] - x[:2]))


def group_act(xi, x):
    """Action of the error xi on the state (Barrau & Bonnabel eq. 22): theta += dtheta, and every position q
    (robot and landmarks) becomes R(dtheta) q + B(dtheta) dq. Landmarks never seen are zero and stay zero
    because their entries of xi are zero.
    Arguments:
        xi: (3+2K,) error [dx, dy, dtheta, dp_1, ...]
        x: (3+2K,) state [x, y, theta, p_1, ...]
    Returns:
        (3+2K,) updated state
    """
    dth = xi[2]
    R, Bm = rot(dth), B_matrix(dth)
    y = x.copy()
    y[2] = x[2] + dth
    y[:2] = R @ x[:2] + Bm @ xi[:2]
    for j in range((len(x) - 3) // 2):
        s = slice(3 + 2 * j, 5 + 2 * j)
        y[s] = R @ x[s] + Bm @ xi[s]
    return y


def xi_error(x_true, x_hat):
    """First-order invariant error (Barrau & Bonnabel eq. 16 / 23): [dtheta, x - x^ - dtheta J x^, p_j - p^_j - dtheta J p^_j],
    returned in the layout [dx, dy, dtheta, dp_1, ...].
    Arguments:
        x_true, x_hat: (3+2K,) true and estimated state
    Returns:
        (3+2K,) error xi
    """
    dth = wrap(x_true[2] - x_hat[2])
    xi = x_true - x_hat
    xi[2] = dth
    xi[:2] -= dth * (J2 @ x_hat[:2])
    for j in range((len(x_hat) - 3) // 2):
        s = slice(3 + 2 * j, 5 + 2 * j)
        xi[s] -= dth * (J2 @ x_hat[s])
    return xi


def noise_jacobian(filter_kind, th, pred, lm):
    """Jacobian G (3+2K, 3) of the propagated error with respect to the odometry noise w = [forward, lateral, rotation]
    (forward and lateral displacement noise in the robot frame, rotation noise).
    standard/fej/ideal: blockdiag(R(th), 1). iekf: the same plus the column of the rotation noise,
    [-J x^_{n|n-1}; 1; -J p^_j]: the exact first-order result for the process model (9) and the error (16) of
    Barrau & Bonnabel, checked by finite differences in the tests. (Their eq. 21 writes x^_{n-1|n-1} there, which is the
    parametrization where the noise enters before the input, their eq. 26.)
    Arguments:
        th: heading used in R(th) (rad)
        pred: (2,) propagated robot position x^_{n|n-1}
        lm: (2K,) landmark positions
    Returns:
        (3+2K, 3) matrix
    """
    G = np.zeros((3 + len(lm), 3))
    G[:2, :2] = rot(th)
    G[2, 2] = 1.0
    if filter_kind == "iekf":
        G[:2, 2] = -J2 @ pred
        G[3:, 2] = -(J2 @ lm.reshape(-1, 2).T).T.reshape(-1)
    return G


def jac_H(filter_kind, meas, th, pR, pL_for_J, rel_ref, j, n):
    """Measurement Jacobian row block of landmark j, (2, n), in the layout [x, y, theta, landmarks].
    standard/fej/ideal/ablations (Huang et al. 2008 eq. 13, 30): grad_h R(th)^T [-I, -J (pL_for_J - pR), I at p_j];
    iekf (Barrau & Bonnabel eq. 21): grad_h R(th)^T [-I, 0, I at p_j].
    Arguments:
        filter_kind: "iekf" or any other kind
        meas: measurement type
        th: heading used in R(th)^T (rad)
        pR: robot position used in the J term (m)
        pL_for_J: landmark position used in the J term (m); FEJ passes the first estimate here
        rel_ref: robot-frame landmark position at which grad_h is evaluated (m)
        j: landmark index; n: state dimension
    Returns:
        (2, n) matrix
    """
    Hrel = meas_jac(meas, rel_ref) @ rot(th).T
    H = np.zeros((2, n))
    H[:, :2] = -Hrel
    if filter_kind != "iekf":
        H[:, 2] = -Hrel @ (J2 @ (pL_for_J - pR))
    H[:, 3 + 2 * j:5 + 2 * j] = Hrel
    return H


# ----------------------------------------------------------------------------------------------
# Scene, data
# ----------------------------------------------------------------------------------------------

def make_scene(p, rng):
    """Truth: nominal odometry, the resulting true poses (a regular polygon approximating a circle), and a fixed
    landmark map drawn uniformly in an annulus around the circle.
    Arguments:
        p: Params
        rng: numpy Generator for the map
    Returns:
        dict with nominal (N, 2) [d, w], poses (N+1, 3) [x, y, theta], lms (K, 2) (m), n_steps
    """
    N = p.n_steps
    nominal = np.tile([p.speed * p.dt, p.omega], (N, 1))
    poses = np.zeros((N + 1, 3))
    for k in range(1, N + 1):
        poses[k, :2] = poses[k - 1, :2] + rot(poses[k - 1, 2]) @ np.array([nominal[k - 1, 0], 0.0])
        poses[k, 2] = poses[k - 1, 2] + nominal[k - 1, 1]
    radius = p.speed * p.dt / (2.0 * np.sin(p.omega / 2.0))
    center = np.array([0.5 * p.speed * p.dt, 0.5 * p.speed * p.dt / np.tan(p.omega / 2.0)])   # the first edge is along +x
    r_lo, r_hi = max(radius - p.ring_halfwidth, 0.0), radius + p.ring_halfwidth
    rr = np.sqrt(rng.uniform(r_lo ** 2, r_hi ** 2, p.n_landmarks))
    aa = rng.uniform(0.0, 2.0 * np.pi, p.n_landmarks)
    lms = center + np.stack([rr * np.cos(aa), rr * np.sin(aa)], axis=1)
    return {"nominal": nominal, "poses": poses, "lms": lms, "n_steps": N, "radius": radius, "center": center}


def visibility(p, scene):
    """(N+1, K) bool: landmark j is measured at step k if it is within `sense_range` of the TRUE robot position."""
    dist = np.linalg.norm(scene["lms"][None, :, :] - scene["poses"][:, None, :2], axis=2)
    return dist <= p.sense_range


def draw_trial(p, scene, rng):
    """Draws one trial's noise. The number and order of draws never depend on any parameter, so a sweep over
    one knob keeps every other random draw identical.
    Returns:
        dict with odom (N, 2) noisy [d, w] readings, z (N+1, K, 2) measurements of every landmark at every step
        (only the visible ones are used), init_err (3,) initial pose error
    """
    N, K = scene["n_steps"], p.n_landmarks
    n_odom, n_meas, n_init = rng.standard_normal((N, 2)), rng.standard_normal((N + 1, K, 2)), rng.standard_normal(3)
    odom = scene["nominal"] + n_odom * np.array([p.sigma_d, p.sigma_w])
    z = np.zeros((N + 1, K, 2))
    sd = np.sqrt(np.diag(p.meas_cov()))
    for k in range(N + 1):
        th = scene["poses"][k, 2]
        rel = (scene["lms"] - scene["poses"][k, :2]) @ rot(th)      # row j = R(th)^T (p_j - x)
        for j in range(K):
            z[k, j] = meas_h(p.meas, rel[j]) + sd * n_meas[k, j]
    init_err = n_init * np.array([p.init_pos_std, p.init_pos_std, p.init_heading_std])
    return {"odom": odom, "z": z, "init_err": init_err}


# ----------------------------------------------------------------------------------------------
# The filters
# ----------------------------------------------------------------------------------------------

def run_filter(filter_kind, p, scene, vis, trial, record=False, keep_states=False):
    """Runs one filter over one trial (propagation + update every step, measurements also at step 0).
    Arguments:
        filter_kind: one of ALL_KINDS
        p: Params
        scene: from make_scene; vis: from visibility
        trial: from draw_trial
        record: also return the linearization matrices of every step (used for the observability matrix)
        keep_states: also return, per step, the landmark estimates lm_hist (N+1, K, 2), their 2x2 covariances in the plain
            error coordinates lmP_hist (N+1, K, 2, 2) and the robot covariance robP_hist (N+1, 3, 3) (NaN for unseen
            landmarks); used by the animation
    Returns:
        dict of per-step arrays (length N+1): x_est (N+1, 3) robot pose, err (N+1, 3) true-minus-estimated pose,
        nees_robot, nees_lm_sum, nees_lm_cnt (sum and count over the landmarks seen), sq_lm (sum of squared landmark
        position errors), sig_theta (reported heading std, rad), info_gain (spurious-information gain per step),
        final landmark estimates lm_est (K, 2) and seen (K,); with record=True also Phi (list), H (list) and the first
        landmark estimates p_first.
    """
    if filter_kind not in ALL_KINDS:
        raise ValueError(f"unknown filter kind {filter_kind}")
    K, N = p.n_landmarks, scene["n_steps"]
    n = 3 + 2 * K
    poses, lms = scene["poses"], scene["lms"]
    iekf = filter_kind == "iekf"
    fej_phi_rule = filter_kind in ("fej", "fej_phi")
    fej_h_rule = filter_kind in ("fej", "fej_h")
    Rm = p.meas_cov()
    Q = np.diag([p.sigma_d ** 2, 0.0, p.sigma_w ** 2])

    x = np.zeros(n)
    x[:3] = poses[0] + trial["init_err"]
    P = np.zeros((n, n))
    P[:3, :3] = np.diag([p.init_pos_std ** 2, p.init_pos_std ** 2, p.init_heading_std ** 2])
    P[3:, 3:] = p.init_landmark_std ** 2 * np.eye(2 * K)
    seen = np.zeros(K, bool)
    p_first = np.zeros((K, 2))
    p_pred_prev = x[:2].copy()                       # p^_{k-1|k-2}, used by the FEJ propagation rule
    dx = np.zeros(n)                                 # rotation shift carried through the filter's own Jacobians
    dx[:2], dx[2] = J2 @ x[:2], 1.0
    out = {k: np.zeros(N + 1) for k in ("nees_robot", "nees_lm_sum", "nees_lm_cnt", "sq_lm", "sig_theta", "info_gain")}
    out["x_est"], out["err"] = np.zeros((N + 1, 3)), np.zeros((N + 1, 3))
    if keep_states:
        out["lm_hist"], out["lmP_hist"] = np.full((N + 1, K, 2), np.nan), np.full((N + 1, K, 2, 2), np.nan)
        out["robP_hist"] = np.zeros((N + 1, 3, 3))
    rec_Phi, rec_H = [], []

    for k in range(N + 1):
        if k > 0:
            d, w = trial["odom"][k - 1]
            th_prev = x[2]
            C_prev = rot(th_prev)
            pred = x[:2] + C_prev @ np.array([d, 0.0])
            if iekf:
                Phi_off = np.zeros(2)
            elif filter_kind == "ideal":
                Phi_off = J2 @ (poses[k, :2] - poses[k - 1, :2])
            elif fej_phi_rule:
                Phi_off = J2 @ (pred - p_pred_prev)
            else:
                Phi_off = J2 @ (pred - x[:2])
            G = noise_jacobian(filter_kind, poses[k - 1, 2] if filter_kind == "ideal" else th_prev, pred, x[3:])
            # P <- Phi P Phi^T + G Q G^T with Phi = I except Phi[:2, 2] = Phi_off
            P[:2, :] += np.outer(Phi_off, P[2, :])
            P[:, :2] += np.outer(P[:, 2], Phi_off)
            P += G @ Q @ G.T
            P = 0.5 * (P + P.T)
            if record:
                Phi_full = np.eye(n)
                Phi_full[:2, 2] = Phi_off
                rec_Phi.append(Phi_full)
            dx[:2] += Phi_off * dx[2]
            x[:2], x[2] = pred, th_prev + w
            p_pred_prev = pred.copy()

        idx = np.where(vis[k])[0]
        gain = 0.0
        if len(idx):
            for j in idx:
                if not seen[j]:
                    rel0 = meas_inv(p.meas, trial["z"][k, j])
                    x[3 + 2 * j:5 + 2 * j] = x[:2] + rot(x[2]) @ rel0
                    p_first[j] = x[3 + 2 * j:5 + 2 * j]
                    seen[j] = True
                    dx[3 + 2 * j:5 + 2 * j] = J2 @ (lms[j] if filter_kind == "ideal" else p_first[j])
            m = len(idx)
            H = np.zeros((2 * m, n))
            r = np.zeros(2 * m)
            for a, j in enumerate(idx):
                pL = x[3 + 2 * j:5 + 2 * j]
                rel_hat = rot(x[2]).T @ (pL - x[:2])
                if filter_kind == "ideal":
                    rel_ref = rot(poses[k, 2]).T @ (lms[j] - poses[k, :2])
                    H[2 * a:2 * a + 2] = jac_H(filter_kind, p.meas, poses[k, 2], poses[k, :2], lms[j], rel_ref, j, n)
                else:
                    pLJ = p_first[j] if fej_h_rule else pL
                    H[2 * a:2 * a + 2] = jac_H(filter_kind, p.meas, x[2], x[:2], pLJ, rel_hat, j, n)
                r[2 * a:2 * a + 2] = meas_resid(p.meas, trial["z"][k, j], meas_h(p.meas, rel_hat))
            Rbig = np.kron(np.eye(m), Rm)
            if iekf:
                shift = np.zeros(n)
                shift[2] = 1.0
            else:
                shift = dx
            Hs = H @ shift
            gain = float(Hs @ np.linalg.solve(Rbig, Hs))
            S = H @ P @ H.T + Rbig
            Kg = np.linalg.solve(S, H @ P).T
            IKH = np.eye(n) - Kg @ H
            P = IKH @ P @ IKH.T + Kg @ Rbig @ Kg.T
            P = 0.5 * (P + P.T)
            delta = Kg @ r
            if iekf:
                x = group_act(delta, x)
            else:
                x = x + delta
            if record:
                rec_H.append(H)
        elif record:
            rec_H.append(np.zeros((0, n)))

        # --- scores in the plain error coordinates X_true - X_hat for every filter ---
        err = np.zeros(3)
        err[:2] = poses[k, :2] - x[:2]
        err[2] = wrap(poses[k, 2] - x[2])
        PX = P[:3, :3].copy()
        if iekf:
            T = np.eye(3)
            T[:2, 2] = J2 @ x[:2]                    # dX_robot = [I, J x^] xi_robot
            PX = T @ P[:3, :3] @ T.T
        out["x_est"][k], out["err"][k] = x[:3], err
        out["nees_robot"][k] = float(err @ np.linalg.solve(PX, err)) if np.linalg.cond(PX) < 1e14 else np.nan
        out["sig_theta"][k] = np.sqrt(P[2, 2])
        if keep_states:
            out["robP_hist"][k] = PX
        out["info_gain"][k] = gain
        s_sum, s_cnt, sq = 0.0, 0, 0.0
        for j in np.where(seen)[0]:
            sl = slice(3 + 2 * j, 5 + 2 * j)
            e = lms[j] - x[sl]
            Pl = P[sl, sl].copy()
            if iekf:
                a = (J2 @ x[sl]).reshape(2, 1)
                Pth_p = P[2:3, sl]                   # (1, 2)
                Pl = Pl + a @ Pth_p + Pth_p.T @ a.T + a @ a.T * P[2, 2]
            if keep_states:
                out["lm_hist"][k, j], out["lmP_hist"][k, j] = x[sl], Pl
            s_sum += float(e @ np.linalg.solve(Pl, e))
            s_cnt += 1
            sq += float(e @ e)
        out["nees_lm_sum"][k], out["nees_lm_cnt"][k], out["sq_lm"][k] = s_sum, s_cnt, sq
    out["p_first"] = p_first.copy()
    out["lm_est"] = x[3:].reshape(K, 2).copy()
    out["seen"] = seen
    out["P_final"] = P
    if record:
        out["Phi"], out["H"] = rec_Phi, rec_H
    return out


# ----------------------------------------------------------------------------------------------
# Observability matrix
# ----------------------------------------------------------------------------------------------

def observability_matrix(Phis, Hs, k0=0, k1=None):
    """Local observability matrix over steps k0..k1 (Huang et al. 2008 eq. 20): rows H_k0, H_{k0+1} Phi_{k0+1},
    H_{k0+2} Phi_{k0+2} Phi_{k0+1}, ... with Phi_k the Jacobian that carries the error from step k-1 to k.
    Arguments:
        Phis: list, Phis[k-1] = Phi_k (n, n) for k = 1..N
        Hs: list, Hs[k] = H_k (m_k, n) for k = 0..N
        k0, k1: first and last step of the window (default: the whole run)
    Returns:
        (sum m_k, n) matrix
    """
    k1 = len(Hs) - 1 if k1 is None else k1
    n = Hs[0].shape[1]
    rows, cum = [], np.eye(n)
    for k in range(k0, k1 + 1):
        if k > k0:
            cum = Phis[k - 1] @ cum
        rows.append(Hs[k] @ cum)
    return np.vstack(rows)


def null_space_report(M, tol=1e-9):
    """Rank and null-space dimension of M with a relative tolerance on the singular values, plus the singular values.
    Arguments:
        M: (rows, n) observability matrix
        tol: singular values below tol * largest count as zero
    Returns:
        (null_dim, singular_values (n,) descending, padded with zeros if rows < n, right singular vectors V (n, n))
    """
    U, s, Vt = np.linalg.svd(M, full_matrices=True)
    s_full = np.zeros(M.shape[1])
    s_full[:len(s)] = s
    rank = int(np.sum(s_full > tol * s_full[0]))
    return M.shape[1] - rank, s_full, Vt.T


def rotation_shift(x_robot, lm_ref):
    """The infinitesimal rotation of the whole scene as a state direction (Barrau & Bonnabel Prop. 1):
    [J x, 1, J p_1, ..., J p_K] (shift of the position, heading, landmarks per radian).
    Arguments:
        x_robot: (3,) pose; lm_ref: (K, 2) landmark positions
    """
    v = np.concatenate([J2 @ x_robot[:2], [1.0]])
    return np.concatenate([v] + [J2 @ q for q in lm_ref])


def translation_shifts(K):
    """The two infinitesimal translations of the whole scene as state directions, (2, 3+2K)."""
    e = np.zeros((2, 3 + 2 * K))
    for a in range(2):
        e[a, a] = 1.0
        for j in range(K):
            e[a, 3 + 2 * j + a] = 1.0
    return e


def observability_demo(p, seed, n_window=10):
    """Local observability of each filter's linearized system on a small scene (3 landmarks inside the path, all
    always visible), first n_window steps, one noisy trial. Reports null-space dimension and the smallest singular
    values of the (column-normalized) observability matrix.
    Arguments:
        p: Params (a copy is modified: 3 landmarks, loops chosen for n_window steps, no range limit)
        seed: int
        n_window: number of steps in the window
    Returns:
        dict kind -> dict(null_dim, sv (descending), rot_resid, trans_resid, n_cols)
    """
    q = Params(**{f.name: getattr(p, f.name) for f in fields(Params)})
    q.n_landmarks, q.sense_range = 3, 1e9
    q.loops = max(1, int(np.ceil(n_window * q.omega / (2.0 * np.pi))))
    rng_map, rng_trial = np.random.default_rng(seed).spawn(2)
    scene = make_scene(q, rng_map)
    # landmarks inside the circle, near its centre
    scene["lms"] = scene["center"] + 0.35 * scene["radius"] * np.array([[1.0, 0.2], [-0.6, 0.7], [0.1, -0.9]])
    vis = visibility(q, scene)
    trial = draw_trial(q, scene, rng_trial)
    res = {}
    for kind in ALL_KINDS:
        o = run_filter(kind, q, scene, vis, trial, record=True)
        Phis, Hs = o["Phi"][:n_window], o["H"][:n_window + 1]
        if kind == "iekf":                       # F = I in the xi coordinates; H as the filter used it
            Phis = [np.eye(Phis[0].shape[0])] * len(Phis)
        M = observability_matrix(Phis, Hs)
        nd, s, _ = null_space_report(M)
        lm_ref = scene["lms"] if kind == "ideal" else o["p_first"]
        rot_dir = rotation_shift(scene["poses"][0], lm_ref) if kind != "iekf" else np.eye(M.shape[1])[2]
        tr = translation_shifts(q.n_landmarks)
        res[kind] = {"null_dim": nd, "sv": s, "n_cols": M.shape[1],
                     "rot_resid": float(np.linalg.norm(M @ rot_dir) / np.linalg.norm(rot_dir)),
                     "trans_resid": float(np.max(np.linalg.norm(M @ tr.T, axis=0)))}
    return res


# ----------------------------------------------------------------------------------------------
# Monte Carlo
# ----------------------------------------------------------------------------------------------

def run_monte_carlo(p, scene, vis, kinds, n_trials, rng):
    """Every trial draws fresh odometry noise, measurement noise and initial error and runs all `kinds` on the same data.
    Arguments:
        p: Params; scene, vis: from make_scene / visibility; kinds: filter kinds; n_trials: int; rng: Generator
    Returns:
        dict kind -> dict of averaged arrays: nees_robot (N+1,), nees_lm (N+1,), rmse_pos, rmse_head, rmse_lm (each (N+1,)),
        sig_theta_rms (N+1,), info_total (n_trials,), plus the first trial's raw output under "trial0".
    """
    N = scene["n_steps"]
    agg = {k: {"nees_robot": np.zeros(N + 1), "nees_lm_sum": np.zeros(N + 1), "nees_lm_cnt": np.zeros(N + 1),
               "sq_pos": np.zeros(N + 1), "sq_head": np.zeros(N + 1), "sq_lm": np.zeros(N + 1), "sig2": np.zeros(N + 1),
               "info_total": np.zeros(n_trials), "n_ok": np.zeros(N + 1), "trial_nees": np.zeros(n_trials)} for k in kinds}
    first = {}
    for t, g in enumerate(rng.spawn(n_trials)):
        trial = draw_trial(p, scene, g)
        for kind in kinds:
            o = run_filter(kind, p, scene, vis, trial)
            a = agg[kind]
            ok = np.isfinite(o["nees_robot"])
            a["n_ok"] += ok
            a["nees_robot"] += np.where(ok, o["nees_robot"], 0.0)
            a["nees_lm_sum"] += o["nees_lm_sum"]
            a["nees_lm_cnt"] += o["nees_lm_cnt"]
            a["sq_pos"] += np.sum(o["err"][:, :2] ** 2, axis=1)
            a["sq_head"] += o["err"][:, 2] ** 2
            a["sq_lm"] += o["sq_lm"]
            a["sig2"] += o["sig_theta"] ** 2
            a["info_total"][t] = o["info_gain"].sum()
            a["trial_nees"][t] = np.nanmean(o["nees_robot"])
            if t == 0:
                first[kind] = o
    res = {}
    for kind in kinds:
        a = agg[kind]
        res[kind] = {"nees_robot": np.where(a["n_ok"] == n_trials, a["nees_robot"] / n_trials, np.nan),
                     "nees_lm": np.where(a["nees_lm_cnt"] > 0, a["nees_lm_sum"] / np.maximum(a["nees_lm_cnt"], 1), np.nan),
                     "nees_lm_sum": a["nees_lm_sum"], "nees_lm_cnt": a["nees_lm_cnt"],
                     "rmse_pos": np.sqrt(a["sq_pos"] / n_trials), "rmse_head": np.sqrt(a["sq_head"] / n_trials),
                     "rmse_lm": np.sqrt(np.where(a["nees_lm_cnt"] > 0, a["sq_lm"] / np.maximum(a["nees_lm_cnt"], 1), np.nan) / 2.0),
                     "sig_theta_rms": np.sqrt(a["sig2"] / n_trials), "info_total": a["info_total"], "trial_nees": a["trial_nees"],
                     "trial0": first[kind]}
    return res


def summarize(res, lap_steps, n_trials):
    """Scalar scores per filter. Windows: all steps, and the last lap.
    Returns:
        dict kind -> dict of floats
    """
    out = {}
    lo, hi = averaged_nees_bounds(3, n_trials)
    for kind, r in res.items():
        last = slice(-lap_steps, None)
        nl = r["nees_lm_sum"].sum() / max(r["nees_lm_cnt"].sum(), 1)
        nl_last = r["nees_lm_sum"][last].sum() / max(r["nees_lm_cnt"][last].sum(), 1)
        out[kind] = {
            "nees_median_trial": float(np.median(r["trial_nees"])), "nees_max_trial": float(np.max(r["trial_nees"])),
            "nees_all": float(np.nanmean(r["nees_robot"])), "nees_last": float(np.nanmean(r["nees_robot"][last])),
            "frac_above": float(np.nanmean(np.where(np.isfinite(r["nees_robot"]), r["nees_robot"] > hi, np.nan))),
            "nees_lm": float(nl), "nees_lm_last": float(nl_last),
            "rmse_pos": float(np.sqrt(np.mean(r["rmse_pos"][last] ** 2))), "rmse_head": float(np.sqrt(np.mean(r["rmse_head"][last] ** 2))),
            "rmse_lm": float(np.sqrt(np.nanmean(r["rmse_lm"][last] ** 2))),
            "sig_theta_last": float(np.mean(r["sig_theta_rms"][last])),
            "info_mean": float(np.mean(r["info_total"])),
            "info_sigma_deg": float(np.rad2deg(1.0 / np.sqrt(np.mean(r["info_total"]))) if np.mean(r["info_total"]) > 0 else np.inf),
        }
    return out


def build_parser():
    d = Params()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", type=float, default=d.speed, help="Robot speed (m/s)")
    ap.add_argument("--turn-rate-deg", type=float, default=d.turn_rate_deg, help="Robot turn rate (deg/s)")
    ap.add_argument("--dt", type=float, default=d.dt, help="Odometry and measurement period (s)")
    ap.add_argument("--loops", type=int, default=d.loops, help="Number of laps of the circle")
    ap.add_argument("--n-landmarks", type=int, default=d.n_landmarks, help="Number of landmarks")
    ap.add_argument("--ring-halfwidth", type=float, default=d.ring_halfwidth,
                    help="Landmarks are drawn uniformly in an annulus of this half-width around the path (m)")
    ap.add_argument("--sense-range", type=float, default=d.sense_range, help="Landmark sensing range (m)")
    ap.add_argument("--wheel-noise-frac", type=float, default=d.wheel_noise_frac,
                    help="Std of each wheel's speed reading as a fraction of the speed")
    ap.add_argument("--wheel-base", type=float, default=d.wheel_base, help="Distance between the drive wheels (m)")
    ap.add_argument("--meas", choices=("position", "range_bearing"), default=d.meas, help="Landmark measurement type")
    ap.add_argument("--pos-noise-std", type=float, default=d.pos_noise_std, help="Relative-position noise std per axis (m)")
    ap.add_argument("--range-noise-std", type=float, default=d.range_noise_std, help="Range noise std (m), range_bearing only")
    ap.add_argument("--bearing-noise-std-deg", type=float, default=d.bearing_noise_std_deg,
                    help="Bearing noise std (deg), range_bearing only")
    ap.add_argument("--init-landmark-std", type=float, default=d.init_landmark_std, help="Prior std of a new landmark (m)")
    ap.add_argument("--init-pos-std", type=float, default=d.init_pos_std, help="Initial robot position std (m)")
    ap.add_argument("--init-heading-std", type=float, default=d.init_heading_std, help="Initial robot heading std (rad)")
    ap.add_argument("--heading-noise-scale", type=float, default=d.heading_noise_scale,
                    help="Multiplier of the odometry rotation noise (1 = the paper's value)")
    ap.add_argument("--n-trials", type=int, default=50, help="Monte Carlo trials")
    ap.add_argument("--sweep-heading-noise", type=float, nargs="*", default=[],
                    help="Optional sweep: odometry rotation-noise multipliers to run (empty = no sweep)")
    ap.add_argument("--sweep-trials", type=int, default=30, help="Monte Carlo trials per sweep value")
    ap.add_argument("--n-window", type=int, default=10, help="Steps of the observability-matrix window")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")
    return ap


def params_from_args(a):
    return Params(**{f.name: getattr(a, f.name) for f in fields(Params)})


def print_observability(res):
    names = {"standard": "standard EKF", "fej": "FEJ-EKF", "iekf": "invariant EKF", "ideal": "ideal EKF",
             "fej_phi": "FEJ, propagation rule only", "fej_h": "FEJ, measurement rule only"}
    print(f"  {'filter':<28s}{'null dim':>9s}{'resid. rot.':>13s}{'resid. trans.':>14s}   smallest 5 singular values / largest")
    for kind in ALL_KINDS:
        r = res[kind]
        s = r["sv"] / r["sv"][0]
        print(f"  {names[kind]:<28s}{r['null_dim']:9d}{r['rot_resid']:13.2e}{r['trans_resid']:14.2e}   "
              + " ".join(f"{v:9.2e}" for v in s[-5:][::-1]))


def main():
    args = build_parser().parse_args()
    p = params_from_args(args)
    rng = np.random.default_rng(args.seed)
    rng_map, rng_mc, rng_perf, rng_sweep = rng.spawn(4)
    scene = make_scene(p, rng_map)
    vis = visibility(p, scene)
    N = scene["n_steps"]
    lap = int(round(2.0 * np.pi / p.omega))
    lo, hi = averaged_nees_bounds(3, args.n_trials)
    print(f"Setup: {p.loops} laps x {lap} steps = {N} steps of {p.dt} s; circle radius {scene['radius']:.2f} m; {p.n_landmarks} landmarks, "
          f"{vis.sum(axis=1).mean():.1f} visible per step (range {p.sense_range} m); measurement '{p.meas}'; "
          f"odometry noise per step: forward {p.sigma_d * 100:.2f} cm, rotation {np.rad2deg(p.sigma_w):.2f} deg")

    print(f"\nLocal observability matrix of each filter's linearized system (small scene, 3 landmarks always visible, "
          f"first {args.n_window} steps; {3 + 2 * 3} columns, true null dimension 3):")
    demo = observability_demo(p, args.seed, args.n_window)
    print_observability(demo)

    kinds = ALL_KINDS
    print(f"\nRunning {args.n_trials} Monte Carlo trials x {len(kinds)} filters ...")
    res = run_monte_carlo(p, scene, vis, kinds, args.n_trials, rng_mc)
    sm = summarize(res, lap, args.n_trials)
    print(f"\nRobot pose NEES (3 DoF, consistent value 3, 95% band of a {args.n_trials}-trial average per step [{lo:.2f}, {hi:.2f}]):")
    print(f"  {'filter':<10s}{'all steps':>10s}{'last lap':>10s}{'steps above band':>18s}{'median trial':>14s}{'max trial':>11s}{'landmark NEES (2 DoF)':>24s}{'last lap':>10s}")
    for kind in kinds:
        s = sm[kind]
        print(f"  {kind:<10s}{s['nees_all']:10.2f}{s['nees_last']:10.2f}{100 * s['frac_above']:17.0f}%{s['nees_median_trial']:14.2f}{s['nees_max_trial']:11.2f}{s['nees_lm']:24.2f}{s['nees_lm_last']:10.2f}")
    print("  paired difference of the per-trial time-averaged NEES, standard minus the other filter (mean +- standard error over trials):")
    for kind in ("fej", "iekf", "ideal", "fej_phi", "fej_h"):
        dd = res["standard"]["trial_nees"] - res[kind]["trial_nees"]
        print(f"    standard - {kind:<8s}{dd.mean():7.2f} +- {dd.std(ddof=1) / np.sqrt(len(dd)):.2f}")
    print("\nAccuracy and reported uncertainty in the last lap (RMS over steps and trials):")
    print(f"  {'filter':<10s}{'pos RMSE (m)':>14s}{'heading RMSE (deg)':>20s}{'reported heading std (deg)':>28s}{'landmark RMSE (m)':>20s}")
    for kind in kinds:
        s = sm[kind]
        print(f"  {kind:<10s}{s['rmse_pos']:14.3f}{np.rad2deg(s['rmse_head']):20.2f}{np.rad2deg(s['sig_theta_last']):28.2f}{s['rmse_lm']:20.3f}")
    print("\nSpurious information along the global rotation (sum over the run of dX^T H^T R^-1 H dX, mean over trials):")
    for kind in kinds:
        s = sm[kind]
        print(f"  {kind:<10s}{s['info_mean']:14.4g}   -> equivalent heading std {s['info_sigma_deg']:.3g} deg")
    skipped = np.where(~np.isfinite(res["standard"]["nees_robot"]))[0]
    print(f"\nNEES is undefined at steps {skipped.tolist()} (the robot covariance is singular: initial pose known exactly and "
          f"no lateral odometry noise); they are left out of the averages.")

    print("\nAverage time (per step) and peak memory of one run per filter:")
    trial0 = draw_trial(p, scene, rng_perf)
    for kind in KINDS:
        _, t_step, mem = measure_performance(run_filter, kind, p, scene, vis, trial0, n_steps=N)
        print(f"  {kind:<10s} avg time={t_step * 1e6:9.1f} us/step | peak mem={mem / 1024.0:9.1f} KB")

    if args.sweep_heading_noise:
        lo_s, hi_s = averaged_nees_bounds(3, args.sweep_trials)
        print(f"\nSweep of the odometry rotation noise ({args.sweep_trials} trials per value, NEES band [{lo_s:.2f}, {hi_s:.2f}]); "
              f"columns: robot pose NEES over all steps (mean over trials, then the median over trials of each trial's time average in brackets)")
        print(f"  {'scale':>6s}{'rot. noise (deg/step)':>23s}" + "".join(f"{k:>18s}" for k in KINDS) + "   spurious info (standard)")
        firsts = []
        for sc_mult in args.sweep_heading_noise:
            q = Params(**{f.name: getattr(p, f.name) for f in fields(Params)})
            q.heading_noise_scale = sc_mult
            r_sw = run_monte_carlo(q, scene, vis, KINDS, args.sweep_trials, np.random.default_rng(args.seed).spawn(4)[3])
            sm_sw = summarize(r_sw, lap, args.sweep_trials)
            firsts.append(r_sw["standard"]["trial0"]["p_first"][vis[0]])
            print(f"  {sc_mult:6.2f}{np.rad2deg(q.sigma_w):23.2f}" + "".join(
                f"{sm_sw[k]['nees_all']:11.2f} ({sm_sw[k]['nees_median_trial']:4.2f})" for k in KINDS) + f"   {sm_sw['standard']['info_mean']:.3g}")
        print(f"  check: landmarks first seen at step 0 are initialized identically at every scale "
              f"(max difference {max(np.abs(f_ - firsts[0]).max() for f_ in firsts):.1e} m)")

    t = np.arange(N + 1) * p.dt
    colors = {"standard": "tab:red", "fej": "tab:blue", "iekf": "tab:green", "ideal": "black"}
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    ax = axs[0, 0]
    ax.plot(scene["poses"][:, 0], scene["poses"][:, 1], "k--", lw=1, label="true path")
    ax.plot(scene["lms"][:, 0], scene["lms"][:, 1], "k^", ms=5, label="landmarks")
    for kind in ("standard", "fej", "iekf"):
        e = res[kind]["trial0"]["x_est"]
        ax.plot(e[:, 0], e[:, 1], color=colors[kind], lw=1, label=f"{kind} (trial 0)")
    ax.set_aspect("equal")
    ax.set_title("Scene and trial-0 estimates")
    ax.legend(fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    ax = axs[0, 1]
    for kind in KINDS:
        ax.plot(t, res[kind]["nees_robot"], color=colors[kind], label=kind)
    ax.axhline(3, color="gray", lw=1)
    ax.axhline(lo, color="gray", ls="--", lw=1)
    ax.axhline(hi, color="gray", ls="--", lw=1)
    top = np.nanmax([np.nanmax(res[k]["nees_robot"]) for k in KINDS])
    ax.set_ylim(0.0, min(top * 1.05, 12.0))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"robot pose NEES ({args.n_trials} trials)")
    ax.set_title("Consistency (grey: 3 and the 95% band)")
    ax.legend(fontsize=7)
    ax = axs[1, 0]
    for kind in ("standard", "fej", "iekf"):
        o = res[kind]["trial0"]
        ax.plot(t, np.rad2deg(o["err"][:, 2]), color=colors[kind], lw=1, label=f"{kind}: heading error")
        ax.plot(t, 3 * np.rad2deg(o["sig_theta"]), color=colors[kind], lw=1, ls="--")
        ax.plot(t, -3 * np.rad2deg(o["sig_theta"]), color=colors[kind], lw=1, ls="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heading error (deg); dashed: +-3 std")
    ax.set_title("One trial: heading error and reported +-3 std")
    ax.legend(fontsize=7)
    ax = axs[1, 1]
    for kind in KINDS:
        ax.plot(t, np.cumsum(res[kind]["trial0"]["info_gain"]), color=colors[kind], label=kind)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("cumulative information along global rotation")
    ax.set_title("Spurious information along the rotation (trial 0)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=130)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
