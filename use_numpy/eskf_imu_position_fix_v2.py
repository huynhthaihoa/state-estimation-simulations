'''
Error-state Kalman filter (ESKF) for an IMU-driven 3D body with a 1 Hz
position fix, against the two shortcuts it exists to avoid.

The body moves smoothly in 3D. A 100 Hz IMU (gyroscope + accelerometer, each
with white noise and a constant unknown bias) drives the prediction; a noisy
position fix arrives once per second. The true trajectory is integrated with
the same discretization the filters use, so the only model error is the one a
filter chooses (for example, leaving the biases out of its state).

Four filters run on the same data:

  - eskf: nominal state (R, v, p, b_a, b_g) integrated with the full model,
    15-dim error state (dp, dv, dtheta, db_a, db_g) with the rotation error
    defined on the right, R = R_hat Exp(dtheta). After each update the error
    is injected into the nominal state and reset to zero, with the reset
    Jacobian G = I - 0.5 [dtheta]x on the rotation block
    (--no-reset-jacobian uses G = I).

  - eskf_no_bias: the same filter with only (dp, dv, dtheta), 9 states; the
    biases are assumed zero.

  - qekf_naive: a full-state EKF that stores the unit quaternion q in its
    16-dim state (p, v, q, b_a, b_g) and corrects it additively. After each
    update q is renormalized; the covariance P is left as it is.

  - qekf_projected: the same, but P is also mapped through the Jacobian of
    the renormalization q -> q/|q|, (I - u u^T)/|q| with u = q/|q|.

Quaternions are Hamilton, scalar first, [w, x, y, z]. The quaternion EKF's
rotation matrix is the homogeneous form (w^2 - v.v) I + 2 v v^T + 2 w [v]x
(v = [x, y, z]), which equals the usual one at unit q and whose derivative
`d_rotate_dq` is exact off the unit sphere as well.

A Monte Carlo run (fresh motion, biases, noise and initial error per trial)
reports attitude and position error and NEES. The quaternion EKF's NEES uses
its 16x16 covariance mapped to the same 15-dim error as the ESKF's
(dtheta = 2 vec(q_hat^* (x) dq)). A smaller sweep repeats the Monte Carlo for
several initial attitude errors.
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt

from state_estimation_simulations.use_numpy.lie_utils import skew, so3_exp, so3_log, so3_right_jacobian
from state_estimation_simulations.utils import (
    averaged_nees_bounds, measure_performance, nees,
    quat_multiply, quat_left_matrix, quat_right_matrix, quat_exp, quat_to_rotation, rotation_to_quat, d_rotate_dq, quat_error_jacobian
)

GRAVITY = np.array([0.0, 0.0, -9.81])
FILTERS = ("eskf", "eskf_no_bias", "qekf_naive", "qekf_projected")
DOF = {"eskf": 15, "eskf_no_bias": 9, "qekf_naive": 15, "qekf_projected": 15}

# --------------------------------------------------------------------------- data

def generate_trial(duration, dt, fix_rate, rate_amp, accel_amp, gyro_noise_std, accel_noise_std,
                   gyro_bias_std, accel_bias_std, pos_noise_std, init_pos_noise_std,
                   init_vel_noise_std, init_att_noise_std, rng):
    """Draws one trial: a smooth 3D motion, the IMU readings it produces (noise plus a
    constant bias), the position fixes and a perturbed initial guess.
    The body rate and the world-frame acceleration are sums of sines with random
    phases; the truth is integrated with the filters' own discretization.
    Arguments:
        duration: length of the run (s)
        dt: IMU sample interval (s)
        fix_rate: position-fix rate (Hz)
        rate_amp: amplitude of each body-rate component (rad/s)
        accel_amp: amplitude of the horizontal world-frame acceleration (m/s^2)
        gyro_noise_std, accel_noise_std: per-sample IMU noise std-devs (rad/s, m/s^2)
        gyro_bias_std, accel_bias_std: std-devs of the constant true biases
        pos_noise_std: position-fix noise std-dev per axis (m)
        init_pos_noise_std, init_vel_noise_std: initial-guess error std-devs (m, m/s)
        init_att_noise_std: initial-guess attitude error std-dev per axis (rad)
        rng: numpy random Generator
    Returns:
        dict with true R, v, p (lists over n+1 steps), b_g, b_a, IMU arrays gyro and
        accel (n,3), fixes {step: z}, and the initial guess R0, v0, p0
    """
    n = int(round(duration / dt))
    R, v, p = np.eye(3), np.zeros(3), np.zeros(3)
    b_g = rng.normal(0.0, gyro_bias_std, 3)
    b_a = rng.normal(0.0, accel_bias_std, 3)
    phase = rng.uniform(0.0, 2.0 * np.pi, 6)
    Rs, vs, ps = [R], [v], [p]
    gyro, accel = np.zeros((n, 3)), np.zeros((n, 3))
    for k in range(n):
        t = k * dt
        omega = rate_amp * np.array([np.sin(0.5 * t + phase[0]), np.sin(0.7 * t + phase[1]),
                                     np.sin(0.3 * t + phase[2])])
        a_world = accel_amp * np.array([np.sin(0.4 * t + phase[3]), np.cos(0.3 * t + phase[4]),
                                        0.4 * np.sin(0.5 * t + phase[5])])
        specific_force = R.T @ (a_world - GRAVITY)
        gyro[k] = omega + b_g + rng.normal(0.0, gyro_noise_std, 3)
        accel[k] = specific_force + b_a + rng.normal(0.0, accel_noise_std, 3)
        acc = R @ specific_force + GRAVITY
        p = p + v * dt + 0.5 * acc * dt ** 2
        v = v + acc * dt
        R = R @ so3_exp(omega * dt)
        Rs.append(R)
        vs.append(v)
        ps.append(p)
    every = int(round(1.0 / (fix_rate * dt)))
    fixes = {k: ps[k] + rng.normal(0.0, pos_noise_std, 3) for k in range(every, n + 1, every)}
    R0 = Rs[0] @ so3_exp(rng.normal(0.0, init_att_noise_std, 3))
    v0 = vs[0] + rng.normal(0.0, init_vel_noise_std, 3)
    p0 = ps[0] + rng.normal(0.0, init_pos_noise_std, 3)
    return dict(R=Rs, v=vs, p=ps, b_g=b_g, b_a=b_a, gyro=gyro, accel=accel, fixes=fixes,
                R0=R0, v0=v0, p0=p0, dt=dt)


def initial_covariance(init_pos_noise_std, init_vel_noise_std, init_att_noise_std,
                       accel_bias_std, gyro_bias_std):
    """15x15 initial covariance of (dp, dv, dtheta, db_a, db_g).
    Arguments:
        init_pos_noise_std, init_vel_noise_std, init_att_noise_std: initial-guess std-devs
        accel_bias_std, gyro_bias_std: prior std-devs of the biases (initial estimate 0)
    Returns:
        (15,15) diagonal covariance
    """
    return np.diag(np.r_[[init_pos_noise_std ** 2] * 3, [init_vel_noise_std ** 2] * 3,
                         [init_att_noise_std ** 2] * 3, [accel_bias_std ** 2] * 3,
                         [gyro_bias_std ** 2] * 3])


def imu_noise_covariance(accel_noise_std, gyro_noise_std, accel_bias_rw, gyro_bias_rw):
    """12x12 covariance of (accel noise, gyro noise, accel-bias walk, gyro-bias walk).
    Arguments:
        accel_noise_std, gyro_noise_std: per-sample IMU noise std-devs
        accel_bias_rw, gyro_bias_rw: bias random-walk intensities assumed by the filters
            (m/s^2/sqrt(s), rad/s/sqrt(s))
    Returns:
        (12,12) diagonal covariance
    """
    return np.diag(np.r_[[accel_noise_std ** 2] * 3, [gyro_noise_std ** 2] * 3,
                         [accel_bias_rw ** 2] * 3, [gyro_bias_rw ** 2] * 3])


# --------------------------------------------------------------------------- ESKF
class ESKF:
    """A simple wrapper for the ESKF functions, to be used in the Monte Carlo runs.
    """
    def __init__(self, R0, v0, p0, P0, Q_imu, pos_noise_std, bias_states=True, reset_jacobian=True):
        """
        Arguments:
            R0: (3,3) initial rotation matrix
            v0: (3,) initial velocity
            p0: (3,) initial position
            P0: (15,15) initial covariance from initial_covariance
            Q_imu: (12,12) from imu_noise_covariance
            pos_noise_std: position-fix noise std-dev (m)
            bias_states: False for the 9-state filter
            reset_jacobian: False uses G = I at the reset
        """
        self.nominal = dict(R=R0.copy(), v=v0.copy(), p=p0.copy(),
                            b_a=np.zeros(3), b_g=np.zeros(3))
        
        self.Q_imu = Q_imu
        self.pos_noise_std = pos_noise_std
        self.bias_states = bias_states
        self.reset_jacobian = reset_jacobian
        
        self.nx = 15 if self.bias_states else 9
        
        self.P0 = P0[:self.nx, :self.nx].copy()
        
    def transition(self, gyro, accel, dt):
        R = self.nominal["R"]
        a = accel - self.nominal["b_a"]
        w = gyro - self.nominal["b_g"]
        Jr_dt = so3_right_jacobian(w * dt) * dt
        F = np.eye(self.nx)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = -0.5 * R @ skew(a) * dt ** 2
        F[3:6, 6:9] = -R @ skew(a) * dt
        F[6:9, 6:9] = so3_exp(w * dt).T
        L = np.zeros((self.nx, 12))
        L[0:3, 0:3] = -0.5 * R * dt ** 2
        L[3:6, 0:3] = -R * dt
        L[6:9, 3:6] = -Jr_dt
        if self.bias_states:
            F[0:3, 9:12] = -0.5 * R * dt ** 2
            F[3:6, 9:12] = -R * dt
            F[6:9, 12:15] = -Jr_dt
            L[9:12, 6:9] = np.eye(3) * np.sqrt(dt)
            L[12:15, 9:12] = np.eye(3) * np.sqrt(dt)
        return F, L, R, a, w
    
    def predict(self, gyro, accel, dt):
        F, L, R, a, w = self.transition(gyro, accel, dt)
        acc = R @ a + GRAVITY
        self.nominal = dict(R=R @ so3_exp(w * dt), v=self.nominal["v"] + acc * dt, p=self.nominal["p"] + self.nominal["v"] * dt + 0.5 * acc * dt ** 2,
               b_a=self.nominal["b_a"], b_g=self.nominal["b_g"])
        self.P0 = F @ self.P0 @ F.T + L @ self.Q_imu @ L.T

    def update(self, z):
        H = np.zeros((3, self.nx))
        H[:, 0:3] = np.eye(3)
        S = H @ self.P0 @ H.T + self.pos_noise_std ** 2 * np.eye(3)
        K = self.P0 @ H.T @ np.linalg.inv(S)
        err = z - self.nominal["p"]
        delta_x = K @ err
        self.nominal["p"] += delta_x[0:3]
        self.nominal["v"] += delta_x[3:6]
        dtheta = delta_x[6:9]
        self.nominal["R"] = self.nominal["R"] @ so3_exp(dtheta)
        if self.bias_states:
            self.nominal["b_a"] += delta_x[9:12]
            self.nominal["b_g"] += delta_x[12:15]
        self.P0 = (np.eye(self.nx) - K @ H) @ self.P0
        if self.reset_jacobian:
            G = np.eye(self.nx)
            G[6:9, 6:9] -= 0.5 * skew(dtheta)
            self.P0 = G @ self.P0 @ G.T
        self.P0 = 0.5 * (self.P0 + self.P0.T)
        
    def get_state(self):
        return self.nominal.copy(), self.P0.copy()


def run_eskf(trial, P0, Q_imu, pos_noise_std, bias_states=True, reset_jacobian=True):
    """Runs the ESKF over one trial.
    Arguments:
        trial: dict from generate_trial
        P0: (15,15) initial covariance from initial_covariance
        Q_imu: (12,12) from imu_noise_covariance
        pos_noise_std: position-fix noise std-dev (m)
        bias_states: False for the 9-state filter
        reset_jacobian: False uses G = I at the reset
    Returns:
        list over steps of (R, v, p, b_a, b_g, P) after each step
    """
    eskf = ESKF(trial["R0"], trial["v0"], trial["p0"], P0, Q_imu, pos_noise_std, bias_states, reset_jacobian)
    out = []
    for k in range(len(trial["gyro"])):
        eskf.predict(trial["gyro"][k], trial["accel"][k], trial["dt"])
        if k + 1 in trial["fixes"]:
            eskf.update(trial["fixes"][k + 1])
        nominal, P = eskf.get_state()
        out.append((nominal["R"], nominal["v"], nominal["p"], nominal["b_a"], nominal["b_g"], P))
    return out


# --------------------------------------------------------------------------- quaternion EKF
class QEKF:
    """A simple wrapper for the quaternion EKF functions, to be used in the Monte Carlo runs.
    """
    def __init__(self, R0, v0, p0, P0_15, Q_imu, pos_noise_std, project_covariance=False):
        """
        Arguments:
            R0: (3,3) initial rotation matrix
            v0: (3,) initial velocity
            p0: (3,) initial position
            P0_15: (15,15) initial covariance from initial_covariance
            Q_imu: (12,12) from imu_noise_covariance
            pos_noise_std: position-fix noise std-dev (m)
            project_covariance: True also maps P through the renormalization's Jacobian
        """
        self.x = np.r_[p0.copy(), v0.copy(), rotation_to_quat(R0), np.zeros(6)]
        to_quat = np.zeros((16, 15))
        to_quat[0:6, 0:6] = np.eye(6)
        to_quat[6:10, 6:9] = 0.5 * quat_left_matrix(self.x[6:10])[:, 1:]
        to_quat[10:16, 9:15] = np.eye(6)
        self.P = to_quat @ P0_15 @ to_quat.T
        self.Q_imu = Q_imu
        self.pos_noise_std = pos_noise_std
        self.project_covariance = project_covariance
    
    def transition(self, gyro, accel, dt):
        """
        Jacobians of one quaternion-EKF step, state (p, v, q, b_a, b_g), with respect to
        the state (F) and to the noise n = (accel noise, gyro noise, accel-bias walk,
        gyro-bias walk) (L). dq/dw uses the first-order increment 0.5 dt [0; I].
        Arguments:
            gyro, accel: (3,) IMU readings of this step
            dt: IMU sample interval (s)
        """
        q = self.x[6:10]
        a = accel - self.x[10:13]
        w = gyro - self.x[13:16]
        R = quat_to_rotation(q)
        dq_dw = 0.5 * dt * np.vstack([np.zeros(3), np.eye(3)])
        D = d_rotate_dq(q, a)
        F = np.eye(16)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:10] = 0.5 * D * dt ** 2
        F[3:6, 6:10] = D * dt
        F[0:3, 10:13] = -0.5 * R * dt ** 2
        F[3:6, 10:13] = -R * dt
        F[6:10, 6:10] = quat_right_matrix(quat_exp(w * dt))
        F[6:10, 13:16] = -quat_left_matrix(q) @ dq_dw
        L = np.zeros((16, 12))
        L[0:3, 0:3] = -0.5 * R * dt ** 2
        L[3:6, 0:3] = -R * dt
        L[6:10, 3:6] = -quat_left_matrix(q) @ dq_dw
        L[10:13, 6:9] = np.eye(3) * np.sqrt(dt)
        L[13:16, 9:12] = np.eye(3) * np.sqrt(dt)
        return F, L, R, a, w
    
    def predict(self, gyro, accel, dt):
        """One predict step of the full-state quaternion EKF, state (p, v, q, b_a, b_g).
        Arguments:
            gyro, accel: (3,) IMU readings of this step
            dt: IMU sample interval (s)
        """
        F, L, R, a, w = self.transition(gyro, accel, dt)
        acc = R @ a + GRAVITY
        self.x[0:3] += self.x[3:6] * dt + 0.5 * acc * dt ** 2
        self.x[3:6] += acc * dt
        self.x[6:10] = quat_multiply(self.x[6:10], quat_exp(w * dt))
        self.P = F @ self.P @ F.T + L @ self.Q_imu @ L.T

    def update(self, z):
        """Additive EKF update with a position fix, then renormalize q.
        Arguments:
            z: (3,) measured position (m)
        """
        H = np.zeros((3, 16))
        H[:, 0:3] = np.eye(3)
        S = H @ self.P @ H.T + self.pos_noise_std ** 2 * np.eye(3)
        K = self.P @ H.T @ np.linalg.inv(S)
        err = z - self.x[0:3]
        self.x += K @ err
        self.P = (np.eye(16) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        norm_q = np.linalg.norm(self.x[6:10])
        if self.project_covariance:
            u = self.x[6:10] / norm_q
            J = np.eye(16)
            J[6:10, 6:10] = (np.eye(4) - np.outer(u, u)) / norm_q
            self.P = J @ self.P @ J.T
        self.x[6:10] /= norm_q
    
    def get_state(self):
        return self.x.copy(), self.P.copy()

def run_qekf(trial, P0_15, Q_imu, pos_noise_std, project_covariance=False):
    """Runs the quaternion EKF over one trial. Its 16x16 initial covariance is the 15x15
    prior with the attitude block mapped by dq/dtheta = 0.5 L(q)[:, 1:].
    Arguments:
        trial: dict from generate_trial
        P0_15: (15,15) initial covariance from initial_covariance
        Q_imu: (12,12) from imu_noise_covariance
        pos_noise_std: position-fix noise std-dev (m)
        project_covariance: see qekf_update
    Returns:
        list over steps of (R, v, p, b_a, b_g, P15), P15 the covariance mapped to
        (dp, dv, dtheta, db_a, db_g)
    """
    qekf = QEKF(trial["R0"], trial["v0"], trial["p0"], P0_15, Q_imu, pos_noise_std, project_covariance)
    
    out = []
    for k in range(len(trial["gyro"])):
        qekf.predict(trial["gyro"][k], trial["accel"][k], trial["dt"])
        if k + 1 in trial["fixes"]:
            qekf.update(trial["fixes"][k + 1])
        x, P = qekf.get_state()
        to_err = np.zeros((15, 16))
        to_err[0:6, 0:6] = np.eye(6)
        to_err[6:9, 6:10] = quat_error_jacobian(x[6:10])
        to_err[9:15, 10:16] = np.eye(6)
        out.append((quat_to_rotation(x[6:10] / np.linalg.norm(x[6:10])), x[3:6], x[0:3],
                    x[10:13], x[13:16], to_err @ P @ to_err.T))
    return out


# --------------------------------------------------------------------------- evaluation

def trial_errors(trial, out, dof):
    """Attitude error (deg), position error (m) and NEES at every step of one run.
    The error is (p - p_hat, v - v_hat, Log(R_hat^T R), b_a - b_a_hat, b_g - b_g_hat),
    truncated to the first `dof` entries.
    Arguments:
        trial: dict from generate_trial
        out: list from run_eskf or run_qekf
        dof: 15, or 9 for the filter without bias states
    Returns:
        att_err, pos_err, nees_values: (n,) arrays
    """
    n = len(out)
    att, pos, ne = np.zeros(n), np.zeros(n), np.zeros(n)
    for k, (R, v, p, b_a, b_g, P) in enumerate(out):
        dtheta = so3_log(R.T @ trial["R"][k + 1])
        err = np.r_[trial["p"][k + 1] - p, trial["v"][k + 1] - v, dtheta,
                    trial["b_a"] - b_a, trial["b_g"] - b_g][:dof]
        att[k] = np.rad2deg(np.linalg.norm(dtheta))
        pos[k] = np.linalg.norm(trial["p"][k + 1] - p)
        ne[k] = nees(err, P[:dof, :dof])
    return att, pos, ne


def run_filter(name, trial, P0, Q_imu, pos_noise_std, reset_jacobian=True):
    """Dispatches one of FILTERS on a trial.
    Arguments:
        name: one of FILTERS
        trial: dict from generate_trial
        P0: (15,15) initial covariance
        Q_imu: (12,12) IMU noise covariance
        pos_noise_std: position-fix noise std-dev (m)
        reset_jacobian: passed to the two ESKFs
    Returns:
        list of per-step estimates (see run_eskf / run_qekf)
    """
    if name == "eskf":
        return run_eskf(trial, P0, Q_imu, pos_noise_std, True, reset_jacobian)
    if name == "eskf_no_bias":
        return run_eskf(trial, P0, Q_imu, pos_noise_std, False, reset_jacobian)
    return run_qekf(trial, P0, Q_imu, pos_noise_std, project_covariance=(name == "qekf_projected"))


def run_monte_carlo(n_trials, trial_kwargs, P0, Q_imu, pos_noise_std, rng, reset_jacobian=True):
    """Monte Carlo over fresh trials (motion, biases, noise and initial error redrawn).
    Arguments:
        n_trials: number of trials
        trial_kwargs: keyword arguments of generate_trial except rng
        P0: (15,15) initial covariance
        Q_imu: (12,12) IMU noise covariance
        pos_noise_std: position-fix noise std-dev (m)
        rng: numpy random Generator
        reset_jacobian: passed to the two ESKFs
    Returns:
        dict name -> dict with att_rmse, pos_rmse (RMS over trials, per step) and
        nees (mean over trials, per step)
    """
    acc = {f: {"att": [], "pos": [], "nees": []} for f in FILTERS}
    for _ in range(n_trials):
        trial = generate_trial(rng=rng, **trial_kwargs)
        for f in FILTERS:
            att, pos, ne = trial_errors(trial, run_filter(f, trial, P0, Q_imu, pos_noise_std,
                                                          reset_jacobian), DOF[f])
            acc[f]["att"].append(att)
            acc[f]["pos"].append(pos)
            acc[f]["nees"].append(ne)
    return {f: {"att_rmse": np.sqrt(np.mean(np.square(a["att"]), axis=0)),
                "pos_rmse": np.sqrt(np.mean(np.square(a["pos"]), axis=0)),
                "nees": np.mean(a["nees"], axis=0)} for f, a in acc.items()}


def second_half_mean(series):
    """Mean of the second half of a per-step series (after the filters have settled).
    Arguments:
        series: (n,) array
    Returns:
        scalar mean of series[n//2:]
    """
    return float(np.mean(series[len(series) // 2:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=30.0, help="Run length (s)")
    parser.add_argument("--dt", type=float, default=0.01, help="IMU sample interval (s)")
    parser.add_argument("--fix-rate", type=float, default=1.0, help="Position-fix rate (Hz)")
    parser.add_argument("--rate-amp", type=float, default=0.4, help="Body-rate amplitude per axis (rad/s)")
    parser.add_argument("--accel-amp", type=float, default=0.5,
                        help="Horizontal world-frame acceleration amplitude (m/s^2); vertical is 0.4x")
    parser.add_argument("--gyro-noise-std", type=float, default=0.01, help="Per-sample gyro noise std-dev (rad/s)")
    parser.add_argument("--accel-noise-std", type=float, default=0.05,
                        help="Per-sample accelerometer noise std-dev (m/s^2)")
    parser.add_argument("--gyro-bias-std", type=float, default=0.01,
                        help="Std-dev of the constant true gyro bias, also the filters' prior (rad/s)")
    parser.add_argument("--accel-bias-std", type=float, default=0.05,
                        help="Std-dev of the constant true accelerometer bias, also the filters' prior (m/s^2)")
    parser.add_argument("--gyro-bias-rw", type=float, default=1e-5,
                        help="Gyro-bias random walk assumed by the filters (rad/s/sqrt(s))")
    parser.add_argument("--accel-bias-rw", type=float, default=1e-4,
                        help="Accelerometer-bias random walk assumed by the filters (m/s^2/sqrt(s))")
    parser.add_argument("--pos-noise-std", type=float, default=0.1, help="Position-fix noise std-dev per axis (m)")
    parser.add_argument("--init-pos-noise-std", type=float, default=0.1, help="Initial position error std-dev (m)")
    parser.add_argument("--init-vel-noise-std", type=float, default=0.1, help="Initial velocity error std-dev (m/s)")
    parser.add_argument("--init-att-noise-deg", type=float, default=2.0,
                        help="Initial attitude error std-dev per axis (deg)")
    parser.add_argument("--n-mc-trials", type=int, default=50, help="Monte Carlo trials at --init-att-noise-deg")
    parser.add_argument("--sweep-att-deg", type=float, nargs="*", default=[0.5, 5.0, 10.0],
                        help="Extra initial attitude errors for the sweep (deg); none = no sweep")
    parser.add_argument("--n-sweep-trials", type=int, default=10, help="Monte Carlo trials per sweep value")
    parser.add_argument("--no-reset-jacobian", action="store_true",
                        help="Use G = I at the ESKF reset instead of I - 0.5 [dtheta]x")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    reset_jacobian = not args.no_reset_jacobian

    def trial_kwargs(att_deg):
        return dict(duration=args.duration, dt=args.dt, fix_rate=args.fix_rate, rate_amp=args.rate_amp,
                    accel_amp=args.accel_amp, gyro_noise_std=args.gyro_noise_std,
                    accel_noise_std=args.accel_noise_std, gyro_bias_std=args.gyro_bias_std,
                    accel_bias_std=args.accel_bias_std, pos_noise_std=args.pos_noise_std,
                    init_pos_noise_std=args.init_pos_noise_std, init_vel_noise_std=args.init_vel_noise_std,
                    init_att_noise_std=np.deg2rad(att_deg))

    def prior(att_deg):
        return initial_covariance(args.init_pos_noise_std, args.init_vel_noise_std, np.deg2rad(att_deg),
                                  args.accel_bias_std, args.gyro_bias_std)

    Q_imu = imu_noise_covariance(args.accel_noise_std, args.gyro_noise_std, args.accel_bias_rw, args.gyro_bias_rw)
    P0 = prior(args.init_att_noise_deg)
    n_steps = int(round(args.duration / args.dt))

    print(f"Single run ({args.duration:.0f} s, {1.0 / args.dt:.0f} Hz IMU, {args.fix_rate:g} Hz position fix, "
          f"initial attitude error std {args.init_att_noise_deg:g} deg per axis"
          f"{', reset Jacobian off' if not reset_jacobian else ''}):")
    trial = generate_trial(rng=rng, **trial_kwargs(args.init_att_noise_deg))
    timing = {}
    for f in FILTERS:
        out, t, mem = measure_performance(run_filter, f, trial, P0, Q_imu, args.pos_noise_std, reset_jacobian,
                                          n_steps=n_steps)
        att, pos, _ = trial_errors(trial, out, DOF[f])
        timing[f] = (t, mem)
        print(f"  {f:<15s} final attitude error={att[-1]:7.3f} deg | final position error={pos[-1]:6.3f} m")

    print(f"\nRunning Monte Carlo ({args.n_mc_trials} trials)...")
    mc = run_monte_carlo(args.n_mc_trials, trial_kwargs(args.init_att_noise_deg), P0, Q_imu, args.pos_noise_std,
                         rng, reset_jacobian)
    print(f"\nMonte Carlo, second half of the run (consistent NEES = state dimension; 95% band per step for "
          f"{args.n_mc_trials} trials: 15 -> [{averaged_nees_bounds(15, args.n_mc_trials)[0]:.1f}, "
          f"{averaged_nees_bounds(15, args.n_mc_trials)[1]:.1f}], 9 -> [{averaged_nees_bounds(9, args.n_mc_trials)[0]:.1f}, "
          f"{averaged_nees_bounds(9, args.n_mc_trials)[1]:.1f}]):")
    print(f"  {'':<15s}{'att RMSE (deg)':>15s}{'pos RMSE (m)':>14s}{'NEES':>10s}{'dof':>5s}")
    for f in FILTERS:
        r = mc[f]
        print(f"  {f:<15s}{second_half_mean(r['att_rmse']):15.3f}{second_half_mean(r['pos_rmse']):14.3f}"
              f"{second_half_mean(r['nees']):10.2f}{DOF[f]:5d}")

    sweep = {}
    if args.sweep_att_deg:
        print(f"\nInitial-attitude-error sweep ({args.n_sweep_trials} trials each; NEES, second half):")
        print(f"  {'att std (deg)':<15s}" + "".join(f"{f:>16s}" for f in FILTERS))
        values = sorted(set(args.sweep_att_deg) | {args.init_att_noise_deg})
        for att_deg in values:
            if att_deg == args.init_att_noise_deg:
                res = mc
            else:
                res = run_monte_carlo(args.n_sweep_trials, trial_kwargs(att_deg), prior(att_deg), Q_imu,
                                      args.pos_noise_std, rng, reset_jacobian)
            sweep[att_deg] = {f: second_half_mean(res[f]["nees"]) for f in FILTERS}
            print(f"  {att_deg:<15g}" + "".join(f"{sweep[att_deg][f]:16.2f}" for f in FILTERS))

    print("\nAverage time (per-step) and peak memory (whole-run) per filter, empirical:")
    for f in FILTERS:
        print(f"  {f:<15s} avg time={timing[f][0] * 1e6:9.2f} us/step | peak mem={timing[f][1] / 1024.0:9.3f} KB")

    t_axis = np.arange(1, n_steps + 1) * args.dt
    colors = {"eskf": "tab:blue", "eskf_no_bias": "tab:red", "qekf_naive": "tab:orange", "qekf_projected": "tab:green"}
    labels = {"eskf": "ESKF", "eskf_no_bias": "ESKF without bias states", "qekf_naive": "quaternion EKF, naive",
              "qekf_projected": "quaternion EKF, P projected"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for f in FILTERS:
        style = "--" if f == "qekf_projected" else "-"
        axes[0, 0].semilogy(t_axis, mc[f]["att_rmse"], style, color=colors[f], label=labels[f])
        axes[0, 1].semilogy(t_axis, mc[f]["pos_rmse"], style, color=colors[f], label=labels[f])
        axes[1, 0].semilogy(t_axis, mc[f]["nees"], style, color=colors[f], label=labels[f])
    for dof, name in ((15, "15 (consistent, 15 states)"), (9, "9 (consistent, 9 states)")):
        axes[1, 0].axhline(dof, color="k", ls=":", lw=1)
        axes[1, 0].text(t_axis[-1], dof, f" {name}", va="center", ha="right", fontsize=8,
                        bbox=dict(fc="white", ec="none", alpha=0.8))
    axes[0, 0].set_title(f"Attitude error RMSE ({args.n_mc_trials} trials)")
    axes[0, 0].set_ylabel("deg")
    axes[0, 1].set_title("Position error RMSE")
    axes[0, 1].set_ylabel("m")
    axes[1, 0].set_title("Average NEES")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("time (s)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    if sweep:
        vals = sorted(sweep)
        for f in FILTERS:
            style = "--o" if f == "qekf_projected" else "-o"
            axes[1, 1].semilogy(vals, [sweep[v][f] for v in vals], style, color=colors[f], label=labels[f])
        axes[1, 1].axhline(15, color="k", ls=":", lw=1)
        axes[1, 1].axhline(9, color="k", ls=":", lw=1)
        axes[1, 1].set_title("NEES (second half) vs. initial attitude error")
        axes[1, 1].set_xlabel("initial attitude error std per axis (deg)")
        axes[1, 1].grid(True, which="both", alpha=0.3)
    else:
        axes[1, 1].axis("off")
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
