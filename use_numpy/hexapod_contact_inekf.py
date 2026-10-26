'''
Contact-aided invariant EKF (Hartley et al. 2020) for a 3D hexapod: an IMU
drives the body, and every foot on the ground is a state that leg kinematics
measures. State

    X = [[R, v, p, d_1 .. d_K],
         [0, I_{K+2}           ]]   in SE_{K+2}(3)

(R: body-to-world rotation, v: velocity, p: position, d_i: world position of
the i-th foot in contact). K changes with the gait: a foot is added at
touchdown and removed at liftoff, so tripod / ripple / wave keep 3 / 4 / 5
feet in the state.

Models (all on the same discrete step dt):
  - IMU propagation, the same group-affine model as invariant_saltation_ekf.py:
        R+ = R Exp(gyro dt), v+ = v + g dt + R accel dt,
        p+ = p + v dt + 0.5 g dt^2 + R 0.5 accel dt^2;
    a stance foot stays put up to a small body-frame "contact noise":
        d+ = d + R w_d dt.
  - Leg kinematics (forward kinematics from joint encoders), per stance foot:
        y_i = R^T (d_i - p) + n_i  (body frame).

Two filters run on identical data:
  - "inekf": right-invariant EKF, error xi = log(X_hat X^-1). Its error
    transition Phi is the SE_2(3) one with identity on the feet -- the same
    for every trajectory -- and the kinematic residual R_hat y + p_hat - d_hat_i
    has the constant Jacobian H_i = [0, 0, I, .., -I (foot i), ..]. A new foot
    initialized as d_hat = p_hat + R_hat y has error xi_p + R_hat n, again
    independent of the state.
  - "world": the ordinary error-state EKF (R = Exp(dtheta) R_hat; v, p, d
    additive), as in invariant_saltation_ekf.py's baseline. Its F, H and
    touchdown Jacobian all depend on the estimate.

Leg odometry plus an IMU cannot observe absolute position or yaw (rotation
about gravity): those four directions should only ever lose information. The
experiments check which filter's covariance respects that.

Experiments (main()):
  1. Consistency, no biases: both filters, tripod / ripple / wave; Monte Carlo
     NEES of (rotation, velocity, position) and of yaw alone. Then a sweep of
     the initial rotation error (tripod): the larger it is, the more the
     world EKF's linearization errors let it "learn" yaw.
  2. One slipping leg (L2 by default): the InEKF trusting every stance foot vs.
     testing each foot's residual against the prediction (chi-squared, 3 DoF,
     99%) before the joint update.
  3. Constant IMU biases in the truth, gyro and accel bias states in both
     filters. With biases the InEKF is only "imperfect": the bias error
     enters the invariant error through -Ad_X_hat, which depends on the state.

Truth: the body walks a planar turning path (forward, sideways and turn rate
are random walks, as in hexapod_gait_ekf.py) at a fixed height, plus a small
roll / pitch / heave at twice the gait frequency, so gravity excites both
tilt axes. The IMU readings are the exact inputs that reproduce that path
under the filters' own discrete model, plus noise. Gait schedules and
footholds come from hexapod_gait_ekf.py, lifted to 3D (footholds on z = 0).
Every gait and both filters see the same truth and the same noise draws.

Simplifications: kinematics noise is isotropic in the body frame (no joint
encoder model); contact noise is in the body frame, not a per-foot contact
frame; stance flags are the true schedule; biases are constant (no bias
random walk). See docs/filtering/hexapod_contact_inekf.md.
'''

import argparse
from dataclasses import dataclass, replace

import numpy as np
import matplotlib.pyplot as plt

from state_estimation_simulations.utils import measure_performance, averaged_nees_bounds, nees  # noqa: E402
from state_estimation_simulations.use_numpy.lie_utils import skew, so3_exp, so3_log, sek3_make, sek3_exp, sek3_log, sek3_inv, sek3_adjoint  # noqa: E402
from state_estimation_simulations.use_numpy.hexapod_gait_ekf import GAITS, LEG_NAMES, NOMINAL_FOOT_OFFSETS, gait_phase, stance_mask  # noqa: E402


G_VEC = np.array([0.0, 0.0, -9.81])
FILTERS = ("inekf", "world")


@dataclass
class Config:
    """Simulation and filter settings (defaults = main()'s defaults)."""
    dt: float = 0.02                  # IMU and kinematics step (s)
    n_steps: int = 600                # steps per run
    ticks_per_cycle: int = 60         # gait cycle length in ticks
    twist0: tuple = (0.1, 0.0, 0.2)   # initial forward, sideways (m/s) and turn rate (rad/s)
    lin_accel_std: float = 0.05       # random walk of the horizontal speeds (m/s^2)
    ang_accel_std: float = 0.1        # random walk of the turn rate (rad/s^2)
    height: float = 0.12              # nominal body height (m)
    tilt_amp: float = 0.05            # roll and pitch amplitude (rad)
    heave_amp: float = 0.01           # body height oscillation amplitude (m)
    gyro_noise_std: float = 0.005     # per-sample gyro noise (rad/s)
    accel_noise_std: float = 0.05     # per-sample accel noise (m/s^2)
    contact_noise_std: float = 0.01   # stance-foot velocity noise, body frame (m/s)
    kin_noise_std: float = 0.005      # kinematic foot-position noise, per axis (m)
    init_rot_std: float = 0.02        # initial rotation error, per axis (rad)
    init_vel_std: float = 0.05        # initial velocity error, per axis (m/s)
    init_pos_std: float = 0.01        # initial position error, per axis (m)
    gyro_bias_std: float = 0.01       # truth gyro bias draw / filter prior (rad/s)
    accel_bias_std: float = 0.05      # truth accel bias draw / filter prior (m/s^2)
    gate: float = 11.34               # chi-squared 3 DoF, 99%

    @property
    def nominal_speed(self):
        return self.twist0[0]


# --- Truth and sensor data ----------------------------------------------------

def rot_z(a):
    """Rotation about the world z axis by a (rad)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_zyx(yaw, pitch, roll):
    """R = Rz(yaw) Ry(pitch) Rx(roll), body-to-world."""
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rot_z(yaw) @ Ry @ Rx


def propagate_base(R, v, p, gyro, accel, h):
    """Discrete IMU model shared by truth and both filters (group-affine):
    R+ = R Exp(gyro h), v+ = v + g h + R accel h,
    p+ = p + v h + 0.5 g h^2 + R 0.5 accel h^2."""
    return (R @ so3_exp(gyro * h),
            v + G_VEC * h + R @ (accel * h),
            p + v * h + 0.5 * G_VEC * h * h + R @ (0.5 * accel * h * h))


def simulate_body(cfg, rng):
    """True body motion and the exact IMU inputs that produce it.
    Planar part: forward / sideways speed (heading frame) and turn rate are
    random walks with per-tick std (lin, lin, ang) x dt. On top: roll =
    tilt_amp sin(phi), pitch = tilt_amp cos(phi), height = height +
    heave_amp sin(phi), phi = 2 x gait frequency x t. Each tick's gyro makes
    R hit the target exactly; each tick's accel makes v hit the target
    velocity exactly (deadbeat), and p follows from the discrete model.
    Returns:
        R: (N,3,3), v: (N,3), p: (N,3), yaw: (N,) true heading of the path
        gyro, accel: (N-1, 3) true IMU inputs of each step
    """
    n, dt = cfg.n_steps, cfg.dt
    t = np.arange(n + 1) * dt
    w = rng.normal(0.0, 1.0, (n, 3)) * np.array([cfg.lin_accel_std, cfg.lin_accel_std, cfg.ang_accel_std]) * dt
    twist = np.vstack([np.array(cfg.twist0), np.array(cfg.twist0) + np.cumsum(w, axis=0)])
    yaw = np.concatenate([[0.0], np.cumsum(twist[:-1, 2]) * dt])
    w_bob = 2.0 * 2.0 * np.pi / (cfg.ticks_per_cycle * dt)
    phi = w_bob * t
    R_tgt = np.array([rot_zyx(yaw[k], cfg.tilt_amp * np.cos(phi[k]), cfg.tilt_amp * np.sin(phi[k]))
                      for k in range(n + 1)])
    v_tgt = np.zeros((n + 1, 3))
    v_tgt[:, 0] = np.cos(yaw) * twist[:, 0] - np.sin(yaw) * twist[:, 1]
    v_tgt[:, 1] = np.sin(yaw) * twist[:, 0] + np.cos(yaw) * twist[:, 1]
    v_tgt[:, 2] = cfg.heave_amp * w_bob * np.cos(phi)

    R = np.zeros((n + 1, 3, 3))
    v = np.zeros((n + 1, 3))
    p = np.zeros((n + 1, 3))
    R[0], v[0], p[0] = R_tgt[0], v_tgt[0], np.array([0.0, 0.0, cfg.height])
    gyro = np.zeros((n, 3))
    accel = np.zeros((n, 3))
    for k in range(n):
        gyro[k] = so3_log(R[k].T @ R_tgt[k + 1]) / dt
        accel[k] = R[k].T @ ((v_tgt[k + 1] - v[k]) / dt - G_VEC)
        R[k + 1], v[k + 1], p[k + 1] = propagate_base(R[k], v[k], p[k], gyro[k], accel[k], dt)
    return R, v, p, yaw, gyro, accel


def simulate_feet(cfg, R, p, yaw, gait, contact_noise, slip=None):
    """World positions of the stance feet, per tick. A foot lands at its
    nominal offset shifted forward by half a stride (heading frame, on
    z = 0), then moves only by contact noise, d+ = d + R w_d dt, and by the
    slip if any. Feet already down at t=0 are placed as if they had landed
    earlier at the nominal stride (as in hexapod_gait_ekf.simulate_feet).
    Arguments:
        cfg: Config
        R, p, yaw: true rotations (N,3,3), positions (N,3), headings (N,)
        gait: "tripod", "ripple" or "wave"
        contact_noise: (N, 6, 3) body-frame foot velocity noise (m/s)
        slip: None, or (leg_index, t_start, t_end, slip_vel (3,)): while in
            stance and t_start <= t < t_end that foot also slides at slip_vel
            (heading frame, m/s)
    Returns:
        stance: (N, 6) bool
        d: (N, 6, 3) foot world positions (NaN in swing)
    """
    n_pts, dt, tpc = len(p), cfg.dt, cfg.ticks_per_cycle
    duty, _ = GAITS[gait]
    stance = stance_mask(n_pts, tpc, gait)
    psi = gait_phase(n_pts, tpc, gait)
    stride = cfg.nominal_speed * duty * tpc * dt
    touchdown = NOMINAL_FOOT_OFFSETS + np.array([stride / 2.0, 0.0])
    d = np.full((n_pts, 6, 3), np.nan)
    for k in range(n_pts):
        for i in np.flatnonzero(stance[k]):
            if k == 0 or not stance[k - 1, i]:
                off = touchdown[i].copy()
                if k == 0:
                    off[0] -= cfg.nominal_speed * (psi[0, i] - (1.0 - duty)) * tpc * dt
                d[k, i] = np.array([p[k, 0], p[k, 1], 0.0]) + rot_z(yaw[k]) @ np.array([off[0], off[1], 0.0])
            else:
                d[k, i] = d[k - 1, i] + R[k - 1] @ contact_noise[k - 1, i] * dt
                if slip is not None and slip[0] == i and slip[1] <= (k - 1) * dt < slip[2]:
                    d[k, i] += rot_z(yaw[k - 1]) @ slip[3] * dt
    return stance, d


@dataclass
class Trial:
    """One trial's truth and sensor data, shared by every gait and filter."""
    R: np.ndarray
    v: np.ndarray
    p: np.ndarray
    yaw: np.ndarray
    gyro_meas: np.ndarray
    accel_meas: np.ndarray
    bias: np.ndarray        # (6,) true gyro + accel bias
    xi0: np.ndarray         # (9,) initial right-invariant error of the estimate
    feet: dict              # gait -> (stance (N,6), y (N,6,3) kinematic readings)
    foot_pos: dict          # gait -> (N,6,3) true foot world positions (NaN in swing)


def simulate_trial(cfg, gaits, rng, slip=None, biases=False):
    """Draws one truth and one set of noise, in a fixed order so that every
    option (slip, biases) leaves all other draws unchanged."""
    n_pts = cfg.n_steps + 1
    R, v, p, yaw, gyro, accel = simulate_body(cfg, rng)
    imu_noise = rng.normal(0.0, 1.0, (cfg.n_steps, 6)) * np.array([cfg.gyro_noise_std] * 3 + [cfg.accel_noise_std] * 3)
    contact_noise = rng.normal(0.0, cfg.contact_noise_std, (n_pts, 6, 3))
    kin_noise = rng.normal(0.0, cfg.kin_noise_std, (n_pts, 6, 3))
    xi0 = rng.normal(0.0, 1.0, 9) * np.repeat([cfg.init_rot_std, cfg.init_vel_std, cfg.init_pos_std], 3)
    bias = rng.normal(0.0, 1.0, 6) * np.repeat([cfg.gyro_bias_std, cfg.accel_bias_std], 3)
    if not biases:
        bias = np.zeros(6)
    feet, foot_pos = {}, {}
    for gait in gaits:
        stance, d = simulate_feet(cfg, R, p, yaw, gait, contact_noise, slip)
        y = np.einsum("kji,kmj->kmi", R, d - p[:, None, :]) + kin_noise
        feet[gait] = (stance, y)
        foot_pos[gait] = d
    return Trial(R, v, p, yaw, gyro + bias[:3] + imu_noise[:, :3], accel + bias[3:] + imu_noise[:, 3:],
                 bias, xi0, feet, foot_pos)


# --- Filters --------------------------------------------------------------------

class ContactEKF:
    """Shared state and bookkeeping for both filters. Covariance layout:
    [rotation (3), velocity (3), position (3), gyro bias (3), accel bias (3)
    if biases, feet (3 each, in the order of self.legs)].
    Subclasses define the error coordinates via predict_jacobians,
    measurement, touchdown_jacobian and correct.
    """
    def __init__(self, R, v, p, P_base, cfg, biases=False, P_bias=None, gating=False):
        self.R, self.v, self.p = R.copy(), v.copy(), p.copy()
        self.d = np.zeros((0, 3))
        self.legs = []
        self.b = np.zeros(6)
        self.cfg = cfg
        self.biases = biases
        self.nb = 15 if biases else 9
        self.P = P_base.copy() if not biases else np.block([[P_base, np.zeros((9, 6))],
                                                            [np.zeros((6, 9)), P_bias]])
        self.gating = gating

    def foot_cols(self, j):
        return slice(self.nb + 3 * j, self.nb + 3 * j + 3)

    def group_index(self):
        """Indices of P that hold (rotation, velocity, position, feet), i.e.
        the SE_{K+2}(3) coordinates, skipping the bias block."""
        return np.r_[0:9, self.nb:self.nb + 3 * len(self.legs)]

    def predict(self, gyro, accel):
        """Propagates the mean with the bias-corrected IMU readings, then the
        covariance with the subclass's transition F and noise map L."""
        h = self.cfg.dt
        gyro_hat, accel_hat = gyro - self.b[:3], accel - self.b[3:]
        R_prev = self.R
        self.R, self.v, self.p = propagate_base(self.R, self.v, self.p, gyro_hat, accel_hat, h)
        F, L = self.predict_jacobians(R_prev, accel_hat, h)
        c = self.cfg
        Qn = np.diag([c.gyro_noise_std ** 2] * 3 + [c.accel_noise_std ** 2] * 3
                     + [c.contact_noise_std ** 2] * (3 * len(self.legs)))
        self.P = F @ self.P @ F.T + L @ Qn @ L.T

    @staticmethod
    def body_increment_map(n_feet, h):
        """(9+3K, 6+3K) map from (gyro noise, accel noise, contact noise)
        to body-frame increments of (R, v, p, d_1..d_K) over one step."""
        G = np.zeros((9 + 3 * n_feet, 6 + 3 * n_feet))
        G[0:3, 0:3] = h * np.eye(3)
        G[3:6, 3:6] = h * np.eye(3)
        G[6:9, 3:6] = 0.5 * h * h * np.eye(3)
        G[9:, 6:] = h * np.eye(3 * n_feet)
        return G

    def remove(self, leg):
        """Liftoff: drop the foot (marginalizing a Gaussian = deleting its rows/cols)."""
        j = self.legs.index(leg)
        keep = np.setdiff1d(np.arange(len(self.P)), np.arange(len(self.P))[self.foot_cols(j)])
        self.P = self.P[np.ix_(keep, keep)]
        self.d = np.delete(self.d, j, axis=0)
        del self.legs[j]

    def augment(self, leg, y):
        """Touchdown: add d = p_hat + R_hat y with its error rows M (P_new =
        [[P, P M^T], [M P, M P M^T + N]], N = kin noise in the error frame)."""
        M = self.touchdown_jacobian(y)
        N = self.cfg.kin_noise_std ** 2 * np.eye(3)
        PM = self.P @ M.T
        self.P = np.block([[self.P, PM], [PM.T, M @ PM + N]])
        self.d = np.vstack([self.d, self.p + self.R @ y])
        self.legs.append(leg)

    def update(self, legs, ys):
        """Joint update with the kinematic readings of `legs` (already in the
        state). With gating, each foot is first tested on its own against the
        prediction (NIS vs cfg.gate) and only the feet that pass are used.
        Returns:
            accepted: (len(legs),) bool
        """
        m = len(legs)
        accepted = np.ones(m, dtype=bool)
        if m == 0:
            return accepted
        N3 = self.cfg.kin_noise_std ** 2 * np.eye(3)
        rs, Hs = zip(*[self.measurement(self.legs.index(leg), y) for leg, y in zip(legs, ys)])
        if self.gating:
            for j in range(m):
                S = Hs[j] @ self.P @ Hs[j].T + N3
                accepted[j] = rs[j] @ np.linalg.solve(S, rs[j]) <= self.cfg.gate
            if not accepted.any():
                return accepted
        H = np.vstack([Hs[j] for j in range(m) if accepted[j]])
        r = np.concatenate([rs[j] for j in range(m) if accepted[j]])
        N = self.cfg.kin_noise_std ** 2 * np.eye(len(r))
        S = H @ self.P @ H.T + N
        K = np.linalg.solve(S, H @ self.P).T
        self.correct(K @ r)
        I_KH = np.eye(len(self.P)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ N @ K.T  # Joseph form
        return accepted

    def step(self, stance_k, y_k):
        """One tick of contact bookkeeping after predict: drop lifted feet,
        update with the feet that stay down, add the feet that just landed.
        Returns:
            accepted: (6,) bool, True where a stance foot's reading updated the state
        """
        for leg in [leg for leg in self.legs if not stance_k[leg]]:
            self.remove(leg)
        old = [leg for leg in self.legs if stance_k[leg]]
        accepted = np.zeros(6, dtype=bool)
        accepted[old] = self.update(old, y_k[old])
        for leg in np.flatnonzero(stance_k):
            if leg not in self.legs:
                self.augment(int(leg), y_k[leg])
        return accepted


class InvariantContactEKF(ContactEKF):
    """Right-invariant EKF on SE_{K+2}(3): X_hat = exp(xi) X, bias error
    zeta = b_hat - b."""

    def predict_jacobians(self, R_prev, accel_hat, h):
        """Phi: the SE_2(3) transition (state-independent) with identity on
        the feet; noise and bias enter as body-frame increments mapped by
        Ad_X_hat at the propagated estimate (biases: -Ad_X_hat G zeta, the only
        state-dependent term). R_prev is unused."""
        n, K = len(self.P), len(self.legs)
        gi = self.group_index()
        F = np.eye(n)
        F[3:6, 0:3] = h * skew(G_VEC)
        F[6:9, 0:3] = 0.5 * h * h * skew(G_VEC)
        F[6:9, 3:6] = h * np.eye(3)
        AdG = sek3_adjoint(self.group()) @ self.body_increment_map(K, h)
        L = np.zeros((n, 6 + 3 * K))
        L[gi] = AdG
        if self.biases:
            F[np.ix_(gi, np.arange(9, 15))] = -AdG[:, :6]
        return F, L

    def group(self):
        return sek3_make(self.R, np.vstack([self.v, self.p, self.d]))

    def measurement(self, j, y):
        """Residual r = R_hat y + p_hat - d_hat_j ~ H xi, H = [0, 0, I, .., -I, ..]."""
        H = np.zeros((3, len(self.P)))
        H[:, 6:9] = np.eye(3)
        H[:, self.foot_cols(j)] = -np.eye(3)
        return self.R @ y + self.p - self.d[j], H

    def touchdown_jacobian(self, y):
        """New foot error = xi_p + R_hat n: copies the position rows."""
        M = np.zeros((3, len(self.P)))
        M[:, 6:9] = np.eye(3)
        return M

    def correct(self, delta):
        X = sek3_exp(-delta[self.group_index()]) @ self.group()
        self.R = X[0:3, 0:3]
        cols = X[0:3, 3:].T
        self.v, self.p, self.d = cols[0], cols[1], cols[2:]
        if self.biases:
            self.b = self.b - delta[9:15]

    def base_error(self, R, v, p):
        """xi of (R, v, p) only: log(X_hat X^-1) restricted to SE_2(3)."""
        Xh = sek3_make(self.R, [self.v, self.p])
        return sek3_log(Xh @ sek3_inv(sek3_make(R, [v, p])))

    def bias_error(self, b):
        return self.b - b


class WorldContactEKF(ContactEKF):
    """Ordinary error-state EKF: R = Exp(dtheta) R_hat; v, p, d, b = estimate + delta."""

    def predict_jacobians(self, R_prev, accel_hat, h):
        """F and noise map at the pre-propagation estimate (both depend on it)."""
        n, K, R = len(self.P), len(self.legs), R_prev
        F = np.eye(n)
        F[3:6, 0:3] = -skew(R @ (accel_hat * h))
        F[6:9, 0:3] = -skew(R @ (0.5 * accel_hat * h * h))
        F[6:9, 3:6] = h * np.eye(3)
        if self.biases:
            F[0:3, 9:12] = -R * h
            F[3:6, 12:15] = -R * h
            F[6:9, 12:15] = -0.5 * R * h * h
        L = np.zeros((n, 6 + 3 * K))
        L[self.group_index()] = np.kron(np.eye(3 + K), R) @ self.body_increment_map(K, h)
        return F, L

    def measurement(self, j, y):
        """Innovation y - R_hat^T (d_hat - p_hat); H = [R^T (d-p)^, 0, -R^T, .., R^T, ..]."""
        H = np.zeros((3, len(self.P)))
        H[:, 0:3] = self.R.T @ skew(self.d[j] - self.p)
        H[:, 6:9] = -self.R.T
        H[:, self.foot_cols(j)] = self.R.T
        return y - self.R.T @ (self.d[j] - self.p), H

    def touchdown_jacobian(self, y):
        """d = p + R y: delta_d = delta_p - (R_hat y)^ dtheta (- R_hat n)."""
        M = np.zeros((3, len(self.P)))
        M[:, 0:3] = -skew(self.R @ y)
        M[:, 6:9] = np.eye(3)
        return M

    def correct(self, delta):
        self.R = so3_exp(delta[0:3]) @ self.R
        self.v = self.v + delta[3:6]
        self.p = self.p + delta[6:9]
        if self.biases:
            self.b = self.b + delta[9:15]
        self.d = self.d + delta[self.nb:].reshape(-1, 3)

    def base_error(self, R, v, p):
        return np.concatenate([so3_log(R @ self.R.T), v - self.v, p - self.p])

    def bias_error(self, b):
        return b - self.b


def initial_covariance(cfg):
    """Prior of the right-invariant error of (R, v, p)."""
    return np.diag(np.repeat([cfg.init_rot_std, cfg.init_vel_std, cfg.init_pos_std], 3) ** 2)


def world_from_invariant(v, p):
    """T(X): right-invariant error -> world error-state coordinates to first
    order (dtheta = xi_R, dv = xi_v - v^ xi_R, dp = xi_p - p^ xi_R), as
    invariant_saltation_ekf.world_to_invariant_map."""
    T = np.eye(9)
    T[3:6, 0:3] = -skew(v)
    T[6:9, 0:3] = -skew(p)
    return T


def make_filter(kind, trial, cfg, biases=False, gating=False):
    """Both filters start from the same estimate X_hat0 = exp(xi0) X0, each
    with the prior in its own coordinates."""
    X0 = sek3_make(trial.R[0], [trial.v[0], trial.p[0]])
    Xh = sek3_exp(trial.xi0) @ X0
    R, v, p = Xh[0:3, 0:3], Xh[0:3, 3], Xh[0:3, 4]
    P0 = initial_covariance(cfg)
    P_bias = np.diag(np.repeat([cfg.gyro_bias_std, cfg.accel_bias_std], 3) ** 2)
    if kind == "inekf":
        return InvariantContactEKF(R, v, p, P0, cfg, biases, P_bias, gating)
    if kind == "world":
        T = world_from_invariant(v, p)
        return WorldContactEKF(R, v, p, T @ P0 @ T.T, cfg, biases, P_bias, gating)
    raise ValueError(f"unknown filter {kind!r}")


def run_filter(kind, trial, gait, cfg, biases=False, gating=False):
    """Runs one filter over one trial and gait.
    Returns dict with:
        err: (N, 9) error of (R, v, p) in the filter's own coordinates
        P: (N, 9, 9) its covariance block
        rot_err: (N, 3) physical rotation error log(R_hat R^T) (world frame;
            [2] is yaw), vel_err, pos_err: (N, 3) estimate - truth
        yaw_var: (N,) reported variance of the yaw error
        bias_err: (N, 6), P_bias: (N, 6, 6) (only with biases)
        accepted: (N, 6) bool
        p_hat: (N, 3) estimated positions
    """
    stance, y = trial.feet[gait]
    ekf = make_filter(kind, trial, cfg, biases, gating)
    n_pts = len(stance)
    out = dict(err=np.zeros((n_pts, 9)), P=np.zeros((n_pts, 9, 9)), rot_err=np.zeros((n_pts, 3)),
               vel_err=np.zeros((n_pts, 3)), pos_err=np.zeros((n_pts, 3)), yaw_var=np.zeros(n_pts),
               bias_err=np.zeros((n_pts, 6)), P_bias=np.zeros((n_pts, 6, 6)),
               accepted=np.zeros((n_pts, 6), dtype=bool), p_hat=np.zeros((n_pts, 3)))
    for k in range(n_pts):
        if k > 0:
            ekf.predict(trial.gyro_meas[k - 1], trial.accel_meas[k - 1])
        out["accepted"][k] = ekf.step(stance[k], y[k])
        if k == 0:
            out["accepted"][k] = False
        out["err"][k] = ekf.base_error(trial.R[k], trial.v[k], trial.p[k])
        out["P"][k] = ekf.P[:9, :9]
        out["rot_err"][k] = so3_log(ekf.R @ trial.R[k].T)
        out["vel_err"][k] = ekf.v - trial.v[k]
        out["pos_err"][k] = ekf.p - trial.p[k]
        out["yaw_var"][k] = ekf.P[2, 2]
        out["p_hat"][k] = ekf.p
        if biases:
            out["bias_err"][k] = ekf.bias_error(trial.bias)
            out["P_bias"][k] = ekf.P[9:15, 9:15]
    return out


# --- Monte Carlo ---------------------------------------------------------------------

def run_monte_carlo(cfg, n_trials, gaits, rng, kinds=FILTERS, gating_modes=(False,), slip=None, biases=False):
    """Averages over fresh truths and noise. For each (gait, kind, gating):
    nees (9 DoF, R/v/p), nees_yaw (1 DoF), nees_bias (6 DoF, biases only),
    mean reported yaw variance, mean squared physical errors rot (3), vel (3),
    pos (3) per tick, and slip / good reading acceptance counts.
    Returns:
        dict (gait, kind, gating) -> dict of per-tick arrays and counts
    """
    n_pts = cfg.n_steps + 1
    t = np.arange(n_pts) * cfg.dt
    keys = [(g, kd, gm) for g in gaits for kd in kinds for gm in gating_modes]
    acc = {key: dict(nees=np.zeros(n_pts), nees_yaw=np.zeros(n_pts), nees_bias=np.zeros(n_pts),
                     yaw_var=np.zeros(n_pts), mse_rot=np.zeros((n_pts, 3)), mse_vel=np.zeros((n_pts, 3)),
                     mse_pos=np.zeros((n_pts, 3)), slip_used=0, slip_total=0, other_used=0, other_total=0)
           for key in keys}
    for _ in range(n_trials):
        trial = simulate_trial(cfg, gaits, rng, slip, biases)
        for gait, kind, gating in keys:
            o = run_filter(kind, trial, gait, cfg, biases, gating)
            a = acc[(gait, kind, gating)]
            for k in range(n_pts):
                a["nees"][k] += nees(o["err"][k], o["P"][k])
                if biases:
                    a["nees_bias"][k] += nees(o["bias_err"][k], o["P_bias"][k])
            a["nees_yaw"] += o["rot_err"][:, 2] ** 2 / o["yaw_var"]
            a["yaw_var"] += o["yaw_var"]
            a["mse_rot"] += o["rot_err"] ** 2
            a["mse_vel"] += o["vel_err"] ** 2
            a["mse_pos"] += o["pos_err"] ** 2
            stance = trial.feet[gait][0]
            slipping = np.zeros_like(stance)
            if slip is not None:
                slipping[:, slip[0]] = stance[:, slip[0]] & (t >= slip[1]) & (t < slip[2])
            updatable = stance.copy()
            updatable[1:] &= stance[:-1]  # a foot's touchdown reading initializes it, not updates
            updatable[0] = False
            a["slip_used"] += int(o["accepted"][slipping & updatable].sum())
            a["slip_total"] += int((slipping & updatable).sum())
            a["other_used"] += int(o["accepted"][updatable & ~slipping].sum())
            a["other_total"] += int((updatable & ~slipping).sum())
    for a in acc.values():
        for name in ("nees", "nees_yaw", "nees_bias", "yaw_var", "mse_rot", "mse_vel", "mse_pos"):
            a[name] /= n_trials
    return acc


def rms(x):
    return np.sqrt(np.mean(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--duration", type=float, default=d.n_steps * d.dt, help="Simulation length (s)")
    parser.add_argument("--dt", type=float, default=d.dt, help="IMU / kinematics step (s)")
    parser.add_argument("--cycle-period", type=float, default=d.ticks_per_cycle * d.dt,
                        help="Gait cycle period (s); must be a multiple of 6*dt")
    parser.add_argument("--forward-speed", type=float, default=d.twist0[0], help="Initial forward speed (m/s)")
    parser.add_argument("--turn-rate", type=float, default=d.twist0[2], help="Initial turn rate (rad/s)")
    parser.add_argument("--gyro-noise-std", type=float, default=d.gyro_noise_std, help="Per-sample gyro noise (rad/s)")
    parser.add_argument("--accel-noise-std", type=float, default=d.accel_noise_std,
                        help="Per-sample accel noise (m/s^2)")
    parser.add_argument("--contact-noise-std", type=float, default=d.contact_noise_std,
                        help="Stance-foot velocity noise, body frame (m/s)")
    parser.add_argument("--kin-noise-std", type=float, default=d.kin_noise_std,
                        help="Kinematic foot-position noise per axis (m)")
    parser.add_argument("--slip-leg", type=str, default="L2", choices=LEG_NAMES, help="Leg that slips in experiment 2")
    parser.add_argument("--slip-start", type=float, default=4.0, help="Slip window start (s)")
    parser.add_argument("--slip-end", type=float, default=8.0, help="Slip window end (s)")
    parser.add_argument("--slip-speed", type=float, default=0.1, help="Sideways (outward) slip speed (m/s)")
    parser.add_argument("--n-mc-trials", type=int, default=50, help="Monte Carlo trials per experiment")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")
    args = parser.parse_args()

    tpc = int(round(args.cycle_period / args.dt))
    if abs(tpc * args.dt - args.cycle_period) > 1e-9 or tpc % 6 != 0:
        raise ValueError("--cycle-period must be a multiple of 6*dt")
    cfg = Config(dt=args.dt, n_steps=int(round(args.duration / args.dt)), ticks_per_cycle=tpc,
                 twist0=(args.forward_speed, 0.0, args.turn_rate), gyro_noise_std=args.gyro_noise_std,
                 accel_noise_std=args.accel_noise_std, contact_noise_std=args.contact_noise_std,
                 kin_noise_std=args.kin_noise_std)
    gaits = list(GAITS)
    n_tr = args.n_mc_trials
    t_hist = np.arange(cfg.n_steps + 1) * cfg.dt
    lo9, hi9 = averaged_nees_bounds(9, n_tr)
    lo1, hi1 = averaged_nees_bounds(1, n_tr)
    lo6, hi6 = averaged_nees_bounds(6, n_tr)

    # --- single run ---
    trial = simulate_trial(cfg, gaits, np.random.default_rng(args.seed))
    single = {}
    print("Single run (no slip, no biases):")
    for gait in gaits:
        for kind in FILTERS:
            o, time_s, mem = measure_performance(run_filter, kind, trial, gait, cfg, n_steps=cfg.n_steps)
            single[(gait, kind)] = o
            print(f"  {gait:<7s} {kind:<6s} final pos err={np.linalg.norm(o['pos_err'][-1]):.4f} m, "
                  f"yaw err={np.degrees(abs(o['rot_err'][-1, 2])):.3f} deg "
                  f"(reported sigma {np.degrees(np.sqrt(o['yaw_var'][-1])):.3f}) | "
                  f"{time_s * 1e6:.0f} us/step, {mem / 1024.0:.0f} KB")

    # --- 1: consistency ---
    print(f"\n1. Consistency, no biases ({n_tr} trials). NEES (R, v, p) expects 9 [{lo9:.2f}, {hi9:.2f}], "
          f"yaw expects 1 [{lo1:.2f}, {hi1:.2f}]")
    mc = run_monte_carlo(cfg, n_tr, gaits, np.random.default_rng(args.seed + 1))
    for gait in gaits:
        for kind in FILTERS:
            a = mc[(gait, kind, False)]
            print(f"  {gait:<7s} {kind:<6s} NEES={a['nees'][1:].mean():6.2f}, yaw NEES={a['nees_yaw'][1:].mean():6.2f} | "
                  f"final yaw RMSE={np.degrees(rms(a['mse_rot'][-1, 2])):.3f} deg vs reported "
                  f"{np.degrees(np.sqrt(a['yaw_var'][-1])):.3f} | "
                  f"RMSE tilt={np.degrees(rms(a['mse_rot'][1:, :2].sum(axis=1))):.3f} deg, "
                  f"v={rms(a['mse_vel'][1:].sum(axis=1)):.4f} m/s | "
                  f"final pos RMSE={rms(a['mse_pos'][-1].sum()):.4f} m")

    rot_sweep = (0.02, 0.1, 0.3)
    print(f"   Initial rotation error sweep, tripod ({n_tr} trials each):")
    mc_rot = {}
    for rs in rot_sweep:
        mc_rot[rs] = run_monte_carlo(replace(cfg, init_rot_std=rs), n_tr, ["tripod"],
                                     np.random.default_rng(args.seed + 4))
        for kind in FILTERS:
            a = mc_rot[rs][("tripod", kind, False)]
            print(f"     init {rs:4.2f} rad {kind:<6s} NEES={a['nees'][1:].mean():7.2f}, "
                  f"yaw NEES={a['nees_yaw'][1:].mean():7.2f} | final yaw RMSE={np.degrees(rms(a['mse_rot'][-1, 2])):6.2f} deg "
                  f"vs reported {np.degrees(np.sqrt(a['yaw_var'][-1])):6.2f} | "
                  f"final pos RMSE={rms(a['mse_pos'][-1].sum()):.4f} m")

    # --- 2: slipping leg ---
    slip_leg = LEG_NAMES.index(args.slip_leg)
    side = 1.0 if args.slip_leg.startswith("L") else -1.0
    slip = (slip_leg, args.slip_start, args.slip_end, np.array([0.0, side * args.slip_speed, 0.0]))
    print(f"\n2. {args.slip_leg} slipping sideways at {args.slip_speed} m/s for t in "
          f"[{args.slip_start}, {args.slip_end}) s, InEKF ({n_tr} trials)")
    mc_slip = run_monte_carlo(cfg, n_tr, gaits, np.random.default_rng(args.seed + 2), kinds=("inekf",),
                              gating_modes=(False, True), slip=slip)
    win = (t_hist >= args.slip_start) & (t_hist < args.slip_end)
    for gait in gaits:
        for gating in (False, True):
            a = mc_slip[(gait, "inekf", gating)]
            print(f"  {gait:<7s} {'gated' if gating else 'naive'}: in window RMSE v={rms(a['mse_vel'][win].sum(axis=1)):.4f} m/s, "
                  f"NEES={a['nees'][win].mean():.2f} | final pos RMSE={rms(a['mse_pos'][-1].sum()):.4f} m | "
                  f"slip readings used {a['slip_used'] / max(a['slip_total'], 1):.1%}, "
                  f"good readings used {a['other_used'] / a['other_total']:.1%}")

    # --- 3: IMU biases ---
    print(f"\n3. Constant IMU biases (gyro std {cfg.gyro_bias_std} rad/s, accel std {cfg.accel_bias_std} m/s^2), "
          f"bias states in both filters ({n_tr} trials). Bias NEES expects 6 [{lo6:.2f}, {hi6:.2f}]")
    mc_b = run_monte_carlo(cfg, n_tr, gaits, np.random.default_rng(args.seed + 3), biases=True)
    for gait in gaits:
        for kind in FILTERS:
            a = mc_b[(gait, kind, False)]
            print(f"  {gait:<7s} {kind:<6s} NEES={a['nees'][1:].mean():6.2f}, yaw NEES={a['nees_yaw'][1:].mean():6.2f}, "
                  f"bias NEES={a['nees_bias'][1:].mean():6.2f} | "
                  f"final yaw RMSE={np.degrees(rms(a['mse_rot'][-1, 2])):.3f} deg vs reported "
                  f"{np.degrees(np.sqrt(a['yaw_var'][-1])):.3f} | final pos RMSE={rms(a['mse_pos'][-1].sum()):.4f} m")

    colors = {"inekf": "tab:blue", "world": "tab:red"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.plot(trial.p[:, 0], trial.p[:, 1], color="green", linestyle="--", linewidth=2, label="Ground truth")
    for kind in FILTERS:
        ax.plot(single[("tripod", kind)]["p_hat"][:, 0], single[("tripod", kind)]["p_hat"][:, 1],
                color=colors[kind], label=kind)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Single run, tripod: body path")
    ax.legend()

    ax = axes[0, 1]
    for kind in FILTERS:
        a = mc_rot[0.1][("tripod", kind, False)]
        ax.plot(t_hist, np.degrees(np.sqrt(a["mse_rot"][:, 2])), color=colors[kind], label=f"{kind}: actual RMSE")
        ax.plot(t_hist, np.degrees(np.sqrt(a["yaw_var"])), color=colors[kind], linestyle="--",
                label=f"{kind}: reported sigma")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Yaw error (deg)")
    ax.set_title(f"Yaw, tripod, initial rotation std 0.1 rad ({n_tr} trials)")
    ax.legend()

    ax = axes[1, 0]
    for rs, ls in ((0.02, ":"), (0.1, "--"), (0.3, "-")):
        for kind in FILTERS:
            ax.plot(t_hist[1:], mc_rot[rs][("tripod", kind, False)]["nees"][1:], color=colors[kind], linestyle=ls,
                    label=f"{kind}, init {rs} rad")
    ax.axhline(9.0, color="black", linewidth=1, label="Expected (9 DoF)")
    ax.axhline(lo9, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(hi9, color="black", linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NEES (R, v, p)")
    ax.set_title("Consistency vs. initial rotation error, tripod")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for gait, ls in (("tripod", "-"), ("wave", "--")):
        for gating in (False, True):
            ax.plot(t_hist, np.sqrt(mc_slip[(gait, "inekf", gating)]["mse_vel"].sum(axis=1)),
                    color="tab:purple" if gating else "tab:orange", linestyle=ls,
                    label=f"{gait}, {'gated' if gating else 'naive'}")
    ax.axvspan(args.slip_start, args.slip_end, color="gray", alpha=0.2, label=f"{args.slip_leg} slipping")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity RMSE (m/s)")
    ax.set_title("InEKF, one slipping leg: naive vs. gated")
    ax.legend()

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
