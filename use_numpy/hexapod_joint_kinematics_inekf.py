'''
Contact-aided invariant EKF for an 18-DOF hexapod whose foot readings come
from joint encoders: hexapod_contact_inekf.py with the HEBI Daisy's leg
kinematics instead of an isotropic foot-position noise.

Each leg has three revolute joints (HEBI X8-9, X8-16, X8-9). The foot's
position in the body frame is the forward kinematics of the joint angles,
y = FK(q). The encoders read

    q~ = q + b + n_q,     n_q ~ N(0, sigma_q^2 I) per tick,   b ~ N(0, sigma_b^2 I) once per run,

white noise n_q plus a constant zero offset b per joint (a calibration
error). To first order the reading y~ = FK(q~) is

    y~ ~ FK(q) + J(q) (n_q + b),     J = dFK/dq (3x3, per leg),

so the white part has covariance sigma_q^2 J J^T, which depends on the leg's
pose, and the offset part is a slowly changing error J(q) b that no white
noise model describes.

Geometry: HEBI's Daisy description (HebiRobotics/hebi_description,
config/daisy/hrdf/Daisy.hrdf and daisyLeg-{Left,Right}.hrdf, Apache-2.0):
hip mounts at +-30, +-90, +-150 deg, 0.2375 / 0.1875 / 0.2375 m from the body
centre; nominal stance from its hex_config.xml (feet on a 0.55 m radius, body
0.21 m above them). leg_fk matches HEBI's own robot model (hebi-py 2.16.2,
import_from_hrdf of Daisy.hrdf) to 8.3e-8 m over 50 random 18-joint
configurations; tests/use_numpy/test_hexapod_joint_kinematics_inekf.py keeps
reference values, so hebi-py is not needed here.

Truth: hexapod_contact_inekf.py's walking body (simulate_body) at the Daisy
body height, with feet landing at Daisy's stance (simulate_feet with
foot_offsets). Each stance foot's true joint angles are the knee-up inverse
kinematics of its true body-frame position.

Filters: the right-invariant EKF of hexapod_contact_inekf.py
(InvariantContactEKF) on identical data; only the kinematic model differs.
  Experiment 1, white encoder noise only (sigma_b = 0), sigma_q swept:
  - "iso": isotropic noise, per-axis variance = the mean eigenvalue of
    sigma_q^2 J J^T at the nominal stance (same total variance, no shape);
  - "iso_max": isotropic with the largest eigenvalue (conservative);
  - "joint": R_hat sigma_q^2 J(q~) J(q~)^T R_hat^T at the reading's angles.
  Experiment 2, joint zero offsets (sigma_b swept, sigma_q fixed):
  - "joint": as above, ignores the offsets;
  - "joint_inflated": treats the offsets as extra white noise,
    R_hat (sigma_q^2 + sigma_b^2) J J^T R_hat^T;
  - "joint_offsets": estimates the 18 offsets as constant states with prior
    sigma_b. The reading is corrected, y_c = FK(q~ - b_hat); with offset error
    zeta = b_hat - b, the residual R_hat y_c + p_hat - d_hat has the extra
    Jacobian -R_hat J on that leg's three offsets, and so has a new foot's
    error at touchdown. Like the IMU biases of hexapod_contact_inekf.py, this
    term depends on the estimate ("imperfect" InEKF).

Metrics: velocity and tilt (roll, pitch) NEES and RMSE, which leg odometry plus
an IMU can observe; position and yaw drift without bound and are not scored.

Simplifications: no link flex or series-elastic deflection (HEBI X-Series
actuators are series-elastic); encoder noise and offsets have the same std on
every joint; HEBI publishes no encoder resolution, so sigma_q is an assumption
and is swept; no IMU biases, no gating, tripod gait by default. See
docs/filtering/hexapod_joint_kinematics_inekf.md.
'''

import argparse
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from state_estimation_simulations.utils import averaged_nees_bounds, nees  # noqa: E402
from state_estimation_simulations.use_numpy.lie_utils import so3_log, sek3_exp, sek3_make  # noqa: E402
from state_estimation_simulations.use_numpy.hexapod_contact_inekf import (  # noqa: E402
    Config, InvariantContactEKF, Trial, initial_covariance, simulate_body, simulate_feet)
from state_estimation_simulations.use_numpy.hexapod_gait_ekf import LEG_NAMES  # noqa: E402


# --- Daisy geometry (HEBI Daisy.hrdf; legs in LEG_NAMES order L1 L2 L3 R1 R2 R3) ---

HIP_ANGLE = np.radians([30.0, 90.0, 150.0, -30.0, -90.0, -150.0])   # Daisy legs 1, 3, 5, 2, 4, 6
HIP_OFFSET = np.array([0.2375, 0.1875, 0.2375, 0.2375, 0.1875, 0.2375])
IS_LEFT = np.array([True, True, True, False, False, False])
ACTUATOR_OUT = 0.0451          # X8 actuator: input face to output, along its axis (m)
LINK_EXT = 0.325               # X5 link extension (m)
LINK_EDGE = 0.02               # X5 link: tube edge to centre (m)
STANCE_RADIUS = 0.55           # hex_config.xml: feet on this radius around the body centre (m)
BODY_HEIGHT = 0.21             # hex_config.xml: body above the feet (m)
WHITE_KINDS = ("iso", "iso_max", "joint")
OFFSET_KINDS = ("joint", "joint_inflated", "joint_offsets")
LABELS = {"iso": "isotropic, same total variance", "iso_max": "isotropic, largest axis",
          "joint": "joint model σq² J Jᵀ", "joint_inflated": "joint model, offsets as extra white noise",
          "joint_offsets": "joint model + 18 offset states"}


def _tf(R=np.eye(3), t=(0.0, 0.0, 0.0)):
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class LegModel:
    """A hexapod's legs, all with n_joints revolute joints. A subclass gives
    frames(q, leg) -> (origins (n, 3), axes (n, 3), foot (3,)) in the body
    frame, stance_foot(leg), stance_ik(leg) and the body height; fk, jacobian
    and the Gauss-Newton ik follow from frames. ik takes the body rotation R
    (body to world) for legs whose redundancy depends on gravity; Daisy's
    ignores it."""
    name = ""
    n_joints = 0
    body_height = 0.0

    def frames(self, q, leg):
        raise NotImplementedError

    def fk(self, q, leg):
        """Foot position (3,) in the body frame for joint angles q of leg."""
        return self.frames(q, leg)[2]

    def jacobian(self, q, leg):
        """(3, n) dFK/dq: column i = axis_i x (foot - origin_i), revolute joints."""
        origins, axes, foot = self.frames(q, leg)
        return np.column_stack([np.cross(axes[i], foot - origins[i]) for i in range(self.n_joints)])

    def ik(self, target, leg, q0, R=None, tol=1e-12, max_iter=50):
        """Joint angles reaching target (body frame) by Gauss-Newton from q0 (the
        branch of q0 is kept). Raises if it does not converge."""
        q = np.array(q0, dtype=float)
        for _ in range(max_iter):
            r = target - self.fk(q, leg)
            if r @ r < tol:
                return q
            q = q + np.linalg.lstsq(self.jacobian(q, leg), r, rcond=None)[0]
        raise RuntimeError(f"leg_ik did not converge for leg {LEG_NAMES[leg]}: residual {np.linalg.norm(r):.2e} m")

    def stance_foot(self, leg):
        raise NotImplementedError

    def stance_ik(self, leg):
        raise NotImplementedError

    def foot_offsets(self):
        """(6, 2) nominal footholds (x forward, y left), LEG_NAMES order."""
        return np.array([self.stance_foot(leg)[:2] for leg in range(6)])

    def config(self, cfg):
        """The contact-InEKF settings at this robot's body height."""
        return replace(cfg, height=self.body_height)


class DaisyLeg(LegModel):
    """HEBI Daisy (see the module docstring)."""
    name = "Daisy"
    n_joints = 3
    body_height = BODY_HEIGHT

    def frames(self, q, leg):
        """Joint frames and foot of one Daisy leg, body frame.
        Chain (HRDF order): hip mount Rz(angle) then (offset, 0, 0); X8-9 (joint 1);
        X5HeavyLeftInside / RightInside bracket; X8-16 (joint 2); X5 link, extension
        0.325 m, twist pi; X8-9 (joint 3); X5 link, extension 0.325 m, twist 0; foot
        at that link's output. Each actuator is T(0, 0, 0.0451) then Rz(q_i).
        Returns:
            origins: (3, 3) joint origins, axes: (3, 3) joint axes (unit), foot: (3,)
        """
        T = _tf(_rz(HIP_ANGLE[leg]), _rz(HIP_ANGLE[leg]) @ np.array([HIP_OFFSET[leg], 0.0, 0.0]))
        bracket = (_tf(_rx(-np.pi / 2.0), (0.0, -0.0225, 0.055)) if IS_LEFT[leg]
                   else _tf(_rx(np.pi / 2.0), (0.0, 0.0225, 0.055)))
        after_joint = [bracket,
                       _tf(_rx(np.pi), (LINK_EXT, 0.0, 0.0)),
                       _tf(np.eye(3), (LINK_EXT, 0.0, 2.0 * LINK_EDGE))]
        origins, axes = np.zeros((3, 3)), np.zeros((3, 3))
        for i in range(3):
            T = T @ _tf(t=(0.0, 0.0, ACTUATOR_OUT))
            origins[i], axes[i] = T[0:3, 3], T[0:3, 2]
            T = T @ _tf(_rz(q[i])) @ after_joint[i]
        return origins, axes, T[0:3, 3]

    def stance_foot(self, leg):
        """Nominal stance foot (body frame): on STANCE_RADIUS along the hip direction, BODY_HEIGHT below."""
        a = HIP_ANGLE[leg]
        return np.array([STANCE_RADIUS * np.cos(a), STANCE_RADIUS * np.sin(a), -BODY_HEIGHT])

    def stance_ik(self, leg):
        """Knee-up joint angles at the nominal stance: of the IK solutions from a
        few starts, the one whose third joint (knee) is highest."""
        best = None
        for q0 in ([0.0, 0.5, 1.5], [0.0, -0.5, -1.5], [0.0, 1.0, 2.0], [0.0, -1.0, -2.0]):
            try:
                q = self.ik(self.stance_foot(leg), leg, q0)
            except RuntimeError:
                continue
            knee_z = self.frames(q, leg)[0][2, 2]
            if best is None or knee_z > best[0] + 1e-9:
                best = (knee_z, q)
        return best[1]


DAISY = DaisyLeg()


def leg_frames(q, leg):
    """Daisy's leg frames (DaisyLeg.frames)."""
    return DAISY.frames(q, leg)


def leg_fk(q, leg):
    """Daisy foot position (3,) in the body frame for joint angles q (3,) of leg."""
    return DAISY.fk(q, leg)


def leg_jacobian(q, leg):
    """Daisy (3, 3) dFK/dq."""
    return DAISY.jacobian(q, leg)


def leg_ik(target, leg, q0, tol=1e-12, max_iter=50):
    """Daisy joint angles reaching target (body frame), Gauss-Newton from q0."""
    return DAISY.ik(target, leg, q0, tol=tol, max_iter=max_iter)


def stance_foot(leg):
    """Daisy's nominal stance foot (body frame)."""
    return DAISY.stance_foot(leg)


def stance_ik(leg):
    """Daisy's knee-up joint angles at the nominal stance."""
    return DAISY.stance_ik(leg)


def nominal_joint_cov(leg, sigma_q, model=DAISY):
    """sigma_q^2 J J^T at the nominal stance of leg."""
    J = model.jacobian(model.stance_ik(leg), leg)
    return sigma_q ** 2 * J @ J.T


def iso_variances(sigma_q, model=DAISY):
    """Per-axis variances of the two isotropic models: the mean and the largest
    eigenvalue of sigma_q^2 J J^T at the nominal stance, averaged over the legs."""
    eig = np.array([np.linalg.eigvalsh(nominal_joint_cov(leg, sigma_q, model)) for leg in range(6)])
    return {"iso": float(eig.mean()), "iso_max": float(eig[:, -1].mean())}


# --- Truth with joint-encoder readings ----------------------------------------------

def daisy_config(cfg):
    """The contact-InEKF settings at Daisy's body height."""
    return DAISY.config(cfg)


def daisy_foot_offsets():
    """(6, 2) Daisy's nominal footholds (x forward, y left), LEG_NAMES order."""
    return DAISY.foot_offsets()


def simulate_trial(cfg, gaits, rng, sigma_q, sigma_b=0.0, model=DAISY):
    """One truth and one set of noise. Draw order follows
    hexapod_contact_inekf.simulate_trial (body, IMU noise, contact noise, then
    the encoder noise in place of its kinematic noise, initial error), with the
    joint offsets drawn last, so sigma_b changes no other draw.
    Returns (n = model.n_joints):
        trial: Trial whose feet[gait] = (stance, FK of the encoder readings)
        q_meas: dict gait -> (N, 6, n) encoder readings q + b + n (NaN in swing)
        b: (6, n) true joint offsets (rad)
    """
    n_pts, nj = cfg.n_steps + 1, model.n_joints
    R, v, p, yaw, gyro, accel = simulate_body(cfg, rng)
    imu_noise = rng.normal(0.0, 1.0, (cfg.n_steps, 6)) * np.array([cfg.gyro_noise_std] * 3 + [cfg.accel_noise_std] * 3)
    contact_noise = rng.normal(0.0, cfg.contact_noise_std, (n_pts, 6, 3))
    enc_noise = rng.normal(0.0, sigma_q, (n_pts, 6, nj))
    xi0 = rng.normal(0.0, 1.0, 9) * np.repeat([cfg.init_rot_std, cfg.init_vel_std, cfg.init_pos_std], 3)
    b = rng.normal(0.0, 1.0, (6, nj)) * sigma_b
    q_start = [model.stance_ik(leg) for leg in range(6)]
    feet, foot_pos, q_meas = {}, {}, {}
    for gait in gaits:
        stance, d = simulate_feet(cfg, R, p, yaw, gait, contact_noise, foot_offsets=model.foot_offsets())
        y = np.full((n_pts, 6, 3), np.nan)
        qm = np.full((n_pts, 6, nj), np.nan)
        for leg in range(6):
            q_prev = q_start[leg]
            for k in np.flatnonzero(stance[:, leg]):
                q_true = model.ik(R[k].T @ (d[k, leg] - p[k]), leg, q_prev, R=R[k])
                q_prev = q_true
                qm[k, leg] = q_true + b[leg] + enc_noise[k, leg]
                y[k, leg] = model.fk(qm[k, leg], leg)
        feet[gait] = (stance, y)
        foot_pos[gait] = d
        q_meas[gait] = qm
    trial = Trial(R, v, p, yaw, gyro + imu_noise[:, :3], accel + imu_noise[:, 3:], np.zeros(6), xi0, feet, foot_pos)
    return trial, q_meas, b


# --- Filters --------------------------------------------------------------------------

class JointKinematicsInEKF(InvariantContactEKF):
    """InvariantContactEKF with a kinematic-noise model per reading (kinds
    "iso", "iso_max": fixed isotropic; "joint": sigma_q^2; "joint_inflated":
    sigma_q^2 + sigma_b^2, times R_hat J(q~) J(q~)^T R_hat^T). The residual
    R_hat y + p_hat - d_hat and a new foot's error both carry R_hat times the
    body-frame noise. self.q_k holds this tick's encoder readings (6, n)."""
    def __init__(self, *args, kind, sigma_q, sigma_b=0.0, iso_var=None, model=DAISY, **kwargs):
        super().__init__(*args, **kwargs)
        self.kind, self.sigma_q, self.sigma_b, self.iso_var = kind, sigma_q, sigma_b, iso_var
        self.model = model
        self.q_k = None

    def kin_noise(self, leg, y):
        if self.kind in ("iso", "iso_max"):
            return self.iso_var[self.kind] * np.eye(3)
        var = self.sigma_q ** 2 + (self.sigma_b ** 2 if self.kind == "joint_inflated" else 0.0)
        RJ = self.R @ self.model.jacobian(self.q_k[leg], leg)
        return var * RJ @ RJ.T


class JointOffsetInEKF(JointKinematicsInEKF):
    """Adds the 6 n joint offsets (n joints per leg; 18 on Daisy) as constant
    states between the base (R, v, p) and the feet: P layout [rotation,
    velocity, position, offsets (6 legs x n), feet]. Offset error
    zeta = b_hat - b; readings are corrected to y_c = FK(q~ - b_hat). Residual
    and touchdown get -R_hat J on the leg's offsets; correct subtracts the
    estimated error. White noise: sigma_q^2."""
    def __init__(self, R, v, p, P_base, cfg, sigma_q, sigma_b, model=DAISY):
        super().__init__(R, v, p, P_base, cfg, kind="joint", sigma_q=sigma_q, sigma_b=sigma_b, model=model)
        nbq = 6 * model.n_joints
        self.nb = 9 + nbq
        self.P = np.block([[P_base, np.zeros((9, nbq))], [np.zeros((nbq, 9)), sigma_b ** 2 * np.eye(nbq)]])
        self.b_q = np.zeros((6, model.n_joints))

    def offset_cols(self, leg):
        n = self.model.n_joints
        return slice(9 + n * leg, 9 + n * (leg + 1))

    def corrected(self, leg):
        q = self.q_k[leg] - self.b_q[leg]
        return self.model.fk(q, leg), self.R @ self.model.jacobian(q, leg)

    def kin_noise(self, leg, y):
        RJ = self.corrected(leg)[1]
        return self.sigma_q ** 2 * RJ @ RJ.T

    def measurement(self, j, y):
        leg = self.legs[j]
        y_c, RJ = self.corrected(leg)
        r, H = super().measurement(j, y_c)
        H[:, self.offset_cols(leg)] = -RJ
        return r, H

    def augment(self, leg, y):
        y_c, RJ = self.corrected(leg)
        M = np.zeros((3, len(self.P)))
        M[:, 6:9] = np.eye(3)
        M[:, self.offset_cols(leg)] = -RJ
        PM = self.P @ M.T
        self.P = np.block([[self.P, PM], [PM.T, M @ PM + self.kin_noise(leg, y)]])
        self.d = np.vstack([self.d, self.p + self.R @ y_c])
        self.legs.append(leg)

    def correct(self, delta):
        super().correct(delta)
        self.b_q = self.b_q - delta[9:self.nb].reshape(6, self.model.n_joints)


def make_filter(kind, trial, cfg, sigma_q, sigma_b, iso_var, model=DAISY):
    """Every filter starts from the same estimate X_hat0 = exp(xi0) X0."""
    X0 = sek3_make(trial.R[0], [trial.v[0], trial.p[0]])
    Xh = sek3_exp(trial.xi0) @ X0
    R, v, p, P0 = Xh[0:3, 0:3], Xh[0:3, 3], Xh[0:3, 4], initial_covariance(cfg)
    if kind == "joint_offsets":
        return JointOffsetInEKF(R, v, p, P0, cfg, sigma_q, sigma_b, model=model)
    return JointKinematicsInEKF(R, v, p, P0, cfg, kind=kind, sigma_q=sigma_q, sigma_b=sigma_b, iso_var=iso_var,
                                model=model)


def run_filter(kind, trial, q_meas, gait, cfg, sigma_q, sigma_b=0.0, iso_var=None, b_true=None, model=DAISY,
               keep_offset_cov=False, offset_dirs=None):
    """One filter over one trial and gait.
    Returns dict with, per tick: nees_vel (velocity block, 3 DoF), tilt_err (N, 2)
    roll / pitch part of log(R_hat R^T), tilt_var (N, 2) its reported variances,
    vel_err (N, 3) v_hat - v, and for joint_offsets b_err (N, 6 n) and b_std
    (N, 6 n), n = model.n_joints; with keep_offset_cov also b_cov (6 n, 6 n),
    the offset block of P at the last tick; with offset_dirs ((6, n) unit
    vectors, one per leg) also dir_std (N, 6), the reported std of each leg's
    offset along its vector."""
    stance, y = trial.feet[gait]
    ekf = make_filter(kind, trial, cfg, sigma_q, sigma_b, iso_var, model)
    n_pts, nbq = len(stance), 6 * model.n_joints
    out = dict(nees_vel=np.zeros(n_pts), tilt_err=np.zeros((n_pts, 2)), tilt_var=np.zeros((n_pts, 2)),
               vel_err=np.zeros((n_pts, 3)), b_err=np.zeros((n_pts, nbq)), b_std=np.zeros((n_pts, nbq)))
    for k in range(n_pts):
        if k > 0:
            ekf.predict(trial.gyro_meas[k - 1], trial.accel_meas[k - 1])
        ekf.q_k = q_meas[gait][k]
        ekf.step(stance[k], y[k])
        err = ekf.base_error(trial.R[k], trial.v[k], trial.p[k])
        out["nees_vel"][k] = nees(err[3:6], ekf.P[3:6, 3:6])
        out["tilt_err"][k] = so3_log(ekf.R @ trial.R[k].T)[:2]
        out["tilt_var"][k] = np.diag(ekf.P)[0:2]
        out["vel_err"][k] = ekf.v - trial.v[k]
        if kind == "joint_offsets":
            out["b_err"][k] = (ekf.b_q - b_true).ravel()
            out["b_std"][k] = np.sqrt(np.diag(ekf.P)[9:9 + nbq])
            if offset_dirs is not None:
                if k == 0:
                    out["dir_std"] = np.zeros((n_pts, 6))
                for leg in range(6):
                    c = ekf.offset_cols(leg)
                    out["dir_std"][k, leg] = np.sqrt(offset_dirs[leg] @ ekf.P[c, c] @ offset_dirs[leg])
    if kind == "joint_offsets" and keep_offset_cov:
        out["b_cov"] = ekf.P[9:9 + nbq, 9:9 + nbq].copy()
    return out


def run_monte_carlo(cfg, n_trials, gait, sigma_q, sigma_b, kinds, rng, model=DAISY):
    """Per kind, per-tick averages over n_trials fresh truths and noise:
    nees_vel (3 DoF), nees_tilt (2 DoF, roll and pitch from the rotation block's
    diagonal), mse_vel (3,), mse_tilt (2,); for joint_offsets also mse_b and
    mean reported var_b (all 6 n offsets, averaged)."""
    n_pts = cfg.n_steps + 1
    iso_var = iso_variances(sigma_q, model)
    acc = {kd: dict(nees_vel=np.zeros(n_pts), nees_tilt=np.zeros(n_pts), mse_vel=np.zeros((n_pts, 3)),
                    mse_tilt=np.zeros((n_pts, 2)), mse_b=np.zeros(n_pts), var_b=np.zeros(n_pts)) for kd in kinds}
    for _ in range(n_trials):
        trial, q_meas, b = simulate_trial(cfg, [gait], rng, sigma_q, sigma_b, model)
        for kind, a in acc.items():
            o = run_filter(kind, trial, q_meas, gait, cfg, sigma_q, sigma_b, iso_var, b, model)
            a["nees_vel"] += o["nees_vel"]
            a["nees_tilt"] += np.sum(o["tilt_err"] ** 2 / o["tilt_var"], axis=1)
            a["mse_vel"] += o["vel_err"] ** 2
            a["mse_tilt"] += o["tilt_err"] ** 2
            a["mse_b"] += np.mean(o["b_err"] ** 2, axis=1)
            a["var_b"] += np.mean(o["b_std"] ** 2, axis=1)
    for a in acc.values():
        for name in a:
            a[name] /= n_trials
    return acc


def summarize(a, skip):
    """Run averages after the first `skip` ticks: velocity NEES, tilt NEES,
    velocity RMSE (m/s), tilt RMSE (deg)."""
    return (a["nees_vel"][skip:].mean(), a["nees_tilt"][skip:].mean(),
            np.sqrt(a["mse_vel"][skip:].sum(axis=1).mean()), np.degrees(np.sqrt(a["mse_tilt"][skip:].sum(axis=1).mean())))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    parser.add_argument("--duration", type=float, default=d.n_steps * d.dt, help="Simulation length (s)")
    parser.add_argument("--gait", type=str, default="tripod", choices=["tripod", "ripple", "wave"], help="Gait")
    parser.add_argument("--sigma-q", type=float, default=1e-3,
                        help="Encoder white-noise std per joint (rad); an assumption, HEBI publishes none")
    parser.add_argument("--sweep-q", type=float, nargs="+", default=[5e-4, 1e-3, 2e-3, 5e-3, 2e-2, 5e-2],
                        help="Experiment 1: encoder white-noise stds (rad)")
    parser.add_argument("--sweep-b-deg", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0],
                        help="Experiment 2: joint zero-offset stds (deg)")
    parser.add_argument("--n-mc-trials", type=int, default=30, help="Monte Carlo trials per setting")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")
    args = parser.parse_args()

    cfg = daisy_config(replace(d, n_steps=int(round(args.duration / d.dt))))
    skip = int(round(1.0 / cfg.dt))   # leave out the first second (initial transient)
    lo3, hi3 = averaged_nees_bounds(3, args.n_mc_trials)
    lo2, hi2 = averaged_nees_bounds(2, args.n_mc_trials)
    print(f"Daisy legs at the nominal stance (knee-up IK); foot noise std along the principal axes "
          f"for {args.sigma_q * 1e3:g} mrad encoder noise (mm):")
    for leg in range(6):
        sd = np.sqrt(np.linalg.eigvalsh(nominal_joint_cov(leg, args.sigma_q)))[::-1] * 1e3
        print(f"  {LEG_NAMES[leg]}: q = {np.round(np.degrees(stance_ik(leg)), 1)} deg -> {np.round(sd, 3)}")
    print(f"Consistent run averages ({args.n_mc_trials} trials): velocity NEES in [{lo3:.2f}, {hi3:.2f}], "
          f"tilt NEES in [{lo2:.2f}, {hi2:.2f}]; averages leave out the first second")

    exp1 = {}
    print(f"\nExperiment 1: white encoder noise ({args.gait})")
    for i, s in enumerate(args.sweep_q):
        exp1[s] = run_monte_carlo(cfg, args.n_mc_trials, args.gait, s, 0.0, WHITE_KINDS,
                                  np.random.default_rng(args.seed + i))
        for kind, a in exp1[s].items():
            nv, nt, rv, rt = summarize(a, skip)
            print(f"  sigma_q {s * 1e3:5.1f} mrad  {kind:<8s} vel NEES {nv:6.2f}  tilt NEES {nt:6.2f} | "
                  f"vel RMSE {rv * 1e3:6.2f} mm/s  tilt RMSE {rt:.4f} deg")

    exp2 = {}
    print(f"\nExperiment 2: joint zero offsets, encoder noise {args.sigma_q * 1e3:g} mrad ({args.gait})")
    for i, sb in enumerate(args.sweep_b_deg):
        exp2[sb] = run_monte_carlo(cfg, args.n_mc_trials, args.gait, args.sigma_q, np.radians(sb), OFFSET_KINDS,
                                   np.random.default_rng(args.seed + 100 + i))
        for kind, a in exp2[sb].items():
            nv, nt, rv, rt = summarize(a, skip)
            extra = ""
            if kind == "joint_offsets" and sb > 0:
                extra = (f" | offset RMSE {np.degrees(np.sqrt(a['mse_b'][-1])):.3f} deg vs reported "
                         f"{np.degrees(np.sqrt(a['var_b'][-1])):.3f} (prior {sb:g})")
            print(f"  offsets {sb:3.1f} deg  {kind:<15s} vel NEES {nv:6.2f}  tilt NEES {nt:6.2f} | "
                  f"vel RMSE {rv * 1e3:6.2f} mm/s  tilt RMSE {rt:.4f} deg{extra}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    colors = {"iso": "tab:orange", "iso_max": "tab:green", "joint": "tab:blue", "joint_inflated": "tab:purple",
              "joint_offsets": "tab:red"}
    xq = np.array(args.sweep_q) * 1e3
    for kind in WHITE_KINDS:
        stats = np.array([summarize(exp1[s][kind], skip) for s in args.sweep_q])
        axes[0, 0].plot(xq, stats[:, 0], "o-", color=colors[kind], label=LABELS[kind])
        axes[1, 0].plot(xq, stats[:, 2] * 1e3, "o-", color=colors[kind], label=LABELS[kind])
    for kind in OFFSET_KINDS:
        stats = np.array([summarize(exp2[sb][kind], skip) for sb in args.sweep_b_deg])
        axes[0, 1].plot(args.sweep_b_deg, stats[:, 0], "o-", color=colors[kind], label=LABELS[kind])
        axes[1, 1].plot(args.sweep_b_deg, stats[:, 2] * 1e3, "o-", color=colors[kind], label=LABELS[kind])
    for ax in axes[0]:
        ax.axhspan(lo3, hi3, color="0.85", label=f"95% band, consistent ({args.n_mc_trials} trials)")
        ax.set_ylabel("velocity NEES (run average, 3 DoF)")
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_ylabel("velocity RMSE (mm/s)")
    for ax in axes[:, 0]:
        ax.set(xscale="log", xlabel="encoder white-noise std σq (mrad)")
        ax.set_xticks(xq, [f"{v:g}" for v in xq])
        ax.minorticks_off()
    for ax in axes[:, 1]:
        ax.set(xlabel="joint zero-offset std σb (deg)")
    axes[0, 0].set_title(f"Experiment 1: white encoder noise ({args.gait})")
    axes[0, 1].set_title(f"Experiment 2: joint zero offsets, σq = {args.sigma_q * 1e3:g} mrad ({args.gait})")
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=120)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
