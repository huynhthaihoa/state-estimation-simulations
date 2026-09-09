'''
Tracks the pose of a rigid object over time from a combination of:

  1. A *prior*, coming from a motion model driven by noisy control/odometry
     inputs (a body-frame twist [v, omega], exactly like the IMU scripts in
     this codebase): T_pred = T_prev (+) Exp([v,omega]*dt).

  2. A *point-cloud measurement*: at every step, the object's known body-frame
     point cloud is observed transformed into the world frame (noisy 3D
     point-to-point correspondences), i.e. z_i = T.act(p_i) + noise.

Four ways to fuse the two into a pose estimate are implemented, all built on
the exact same `motion_model`/`observation_model` functions and manif's
built-in analytical Jacobians (no hand-rolled skew/exp/log math):

  - EKF (recursive): predict with the motion model + propagate a 6x6 tangent
    covariance, update with the point-cloud observation model + a Kalman
    gain. Constant-size state (T, P); only ever looks at the current step.

  - Invariant EKF / IEKF (recursive): same predict step as the EKF (already
    exact for this group-affine motion model), but the update expresses the
    residual in the estimate's body frame, which makes the point-cloud
    measurement Jacobian state-independent (depends only on the fixed body
    points, never on the current pose estimate) instead of being
    re-linearized around the current rotation every step.

  - Batch Gauss-Newton (smoother): jointly optimizes the *whole* trajectory
    T_0..T_N at once against three factor types (an initial-pose prior, N
    motion factors between consecutive poses, and (N+1)*n_points measurement
    factors), generalizing the single-pose GN correction loop already used in
    robot_imu_simulation_manif.py to a multi-pose graph. Has access to the
    full trajectory (not just causal history), so it can do at least as well
    as the EKF.

  - UKF (recursive): same predict/update structure as the EKF, but with no
    Jacobians at all -- sigma points sampled around the current estimate are
    retracted onto SE(3), pushed through the exact (not linearized)
    motion_model/observation_model, and recombined into a new mean/
    covariance. Genuinely different in character from the other three
    methods here, which all rely on an analytical Jacobian somewhere.

A pure dead-reckoning trajectory (motion model only, no point-cloud
correction at all) is carried along as the "uncorrected" baseline, and also
doubles as the batch solver's initial guess.
'''

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import manifpy as manif

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import measure_performance, true_body_rates, unscented_weights, unscented_sigma_offsets

def make_body_point_cloud(n_points, rng, half_extent=0.5):
    """A fixed, non-degenerate set of body-frame landmark points (the object's
    known geometry), sampled inside a small cube so the full 6-DoF pose stays
    observable from the point correspondences alone.
    Arguments:
        n_points: number of points to generate
        rng: numpy random number generator
        half_extent: half the side length of the cube in which to sample points
    Returns:
        body_points: (n_points, 3) array of points in the object's body frame
    """
    return rng.uniform(-half_extent, half_extent, size=(n_points, 3))


def motion_model(T_prev, twist, dt, J_self=None, J_tau=None):
    """Constant body-twist SE(3) motion model: T_pred = T_prev (+) Exp(twist*dt).
    twist = [vx,vy,vz,wx,wy,wz]. If given, J_self/J_tau are filled with the
    Jacobians of T_pred wrt T_prev and wrt the tangent increment, respectively
    (manif's rplus Jacobians -- used for EKF covariance propagation and for
    chaining the GN motion-factor Jacobian).
    Arguments:
        T_prev: previous pose (manif.SE3)
        twist: body-frame twist (6-vector)
        dt: time step (s)
        J_self: optional (6,6) array to fill with dT_pred/dT_prev
        J_tau: optional (6,6) array to fill with dT_pred/dtau
    Returns:
        T_pred: predicted pose (manif.SE3)
    """
    tangent = manif.SE3Tangent(twist * dt)
    if J_self is None:
        return T_prev.rplus(tangent)
    return T_prev.rplus(tangent, J_self, J_tau)


def observation_model(T, body_points, with_jacobian=False):
    """Point-cloud observation model: predicts every body point transformed
    into the world frame by T. Returns predicted points (M,3) and, if
    requested, the stacked (3M,6) Jacobian wrt a right-perturbation of T
    (manif's act() Jacobian -- used by both the EKF update and the GN
    measurement factors).
    Arguments:
        T: pose (manif.SE3)
        body_points: (M,3) array of points in the object's body frame
        with_jacobian: if True, also return the stacked Jacobian (3M,6)
    Returns:
        pred: (M,3) array of predicted points in the world frame
        J: (3M,6) Jacobian of pred wrt a right-perturbation of T, or None if with_jacobian=False
    """
    n_points = len(body_points)
    pred = np.zeros((n_points, 3))
    J = np.zeros((3 * n_points, 6)) if with_jacobian else None
    for i, p in enumerate(body_points):
        if with_jacobian:
            J_i = np.zeros((3, 6))
            pred[i] = T.act(p, J_i)
            J[3 * i:3 * i + 3, :] = J_i
        else:
            pred[i] = T.act(p)
    return pred, J


def generate_ground_truth_and_data(duration, dt, n_points, vel_noise_std, gyro_noise_std,
                                    point_noise_std, rng):
    """Builds the true trajectory (exp-map integration of noise-free inputs)
    and the noisy inputs/point-cloud measurements a tracker would actually
    receive.
    Arguments:
        duration: total simulation time (s)
        dt: time step (s)
        n_points: number of body-frame point-cloud landmarks
        vel_noise_std: std-dev of Gaussian noise added to the true body-frame linear velocity (m/s)
        gyro_noise_std: std-dev of Gaussian noise added to the true body-frame angular velocity (rad/s)
        point_noise_std: std-dev of Gaussian noise added to the predicted world-frame point-cloud measurements (m)
        rng: numpy random number generator
    Returns:
        body_points: (M,3) array of points in the object's body frame
        T_true: list of true poses (manif.SE3)
        u_meas: (N,6) array of noisy body-frame twists
        z: list of noisy point-cloud measurements (M,3)
    """
    n_steps = int(duration / dt)
    body_points = make_body_point_cloud(n_points, rng)

    T_true = [manif.SE3.Identity()]
    u_meas = np.zeros((n_steps, 6))
    z = [observation_model(T_true[0], body_points)[0]
         + rng.normal(0.0, point_noise_std, (n_points, 3))]

    for k in range(n_steps):
        t = k * dt
        omega_true, v_true = true_body_rates(t)
        twist_true = np.concatenate([v_true, omega_true])

        T_true.append(motion_model(T_true[-1], twist_true, dt))

        noise = np.concatenate([
            rng.normal(0.0, vel_noise_std, 3),
            rng.normal(0.0, gyro_noise_std, 3),
        ])
        u_meas[k] = twist_true + noise

        pred, _ = observation_model(T_true[-1], body_points)
        z.append(pred + rng.normal(0.0, point_noise_std, (n_points, 3)))

    return body_points, T_true, u_meas, z


def run_dead_reckoning(T_init, u_meas, dt):
    """Prior-only baseline: propagate the motion model, never look at the
    point-cloud measurements.
    Arguments:
        T_init: initial pose (manif.SE3)
        u_meas: (N,6) array of noisy body-frame twists
        dt: time step (s)
    Returns:
        T_list: list of predicted poses (manif.SE3)
    """
    T_list = [T_init]
    for k in range(len(u_meas)):
        T_list.append(motion_model(T_list[-1], u_meas[k], dt))
    return T_list


def run_ekf(T_init, P_init, u_meas, z, body_points, dt, Q_tangent, point_noise_std):
    """Recursive EKF: alternates a motion-model predict step (mean + 6x6
    tangent covariance) with a point-cloud observation-model update step.
    Only ever holds the current (T, P) -- no memory of past states.
    Arguments:
        T_init: initial pose (manif.SE3)
        P_init: initial 6x6 tangent covariance (numpy array)
        u_meas: (N,6) array of noisy body-frame twists
        z: list of noisy point-cloud measurements (M,3)
        body_points: (M,3) array of points in the object's body frame
        dt: time step (s)
        Q_tangent: 6x6 tangent covariance of the motion-model twist increment (numpy array)
        point_noise_std: std-dev of Gaussian noise added to the predicted world-frame point-cloud measurements (m)
    Returns:
        T_list: list of estimated poses (manif.SE3)
    """
    n_points = len(body_points)
    R_diag = point_noise_std ** 2 * np.eye(3 * n_points)

    T_est, P = T_init, P_init.copy()
    T_list = [T_est]
    for k in range(len(u_meas)):
        # --- Predict ---
        J_self, J_tau = np.zeros((6, 6)), np.zeros((6, 6))
        T_pred = motion_model(T_est, u_meas[k], dt, J_self, J_tau)
        P_pred = J_self @ P @ J_self.T + J_tau @ Q_tangent @ J_tau.T

        # --- Update ---
        pred, H = observation_model(T_pred, body_points, with_jacobian=True)
        r = (z[k + 1] - pred).reshape(-1)
        S = H @ P_pred @ H.T + R_diag
        K = P_pred @ H.T @ np.linalg.inv(S)
        delta = K @ r

        T_est = T_pred.rplus(manif.SE3Tangent(delta))
        P = (np.eye(6) - K @ H) @ P_pred

        T_list.append(T_est)

    return T_list


def run_iekf(T_init, P_init, u_meas, z, body_points, dt, Q_tangent, point_noise_std):
    """Left-invariant EKF: identical predict step to `run_ekf` (already exact for
    this group-affine motion model), but the update expresses the residual in the
    estimate's body frame -- z_body = T_pred^-1 @ z -- instead of the world frame.
    That makes the measurement Jacobian [I | -skew(p_i)] state-independent (only
    depends on the fixed body points, never on the current rotation estimate), so
    it's precomputed once outside the loop instead of being re-derived from R_pred
    every step.

    Note: with isotropic point-noise covariance (R_diag = sigma^2 * I), the
    world-frame (EKF) and body-frame (here) residual/Jacobian pairs differ only by
    a per-point orthogonal rotation (R_pred), which cancels exactly out of the
    Kalman gain and posterior covariance -- so this produces the *exact same*
    corrections as run_ekf every step, not just similar ones. The real benefit here
    is computational (H is fixed, no per-step Jacobian rebuild) and structural
    (state-independent linearization), not different accuracy on this benchmark.
    Arguments:
        T_init: initial pose (manif.SE3)
        P_init: initial 6x6 tangent covariance (numpy array)
        u_meas: (N,6) array of noisy body-frame twists
        z: list of noisy point-cloud measurements (M,3)
        body_points: (M,3) array of points in the object's body frame
        dt: time step (s)
        Q_tangent: 6x6 tangent covariance of the motion-model twist increment (numpy array)
        point_noise_std: std-dev of Gaussian noise added to the predicted world-frame point-cloud measurements (m)
    Returns:
        T_list: list of estimated poses (manif.SE3)
    """
    n_points = len(body_points)
    R_diag = point_noise_std ** 2 * np.eye(3 * n_points)

    # Precompute the state-independent measurement Jacobian (body-frame)
    H = np.zeros((3 * n_points, 6))
    for i, p in enumerate(body_points):
        H[3 * i:3 * i + 3, :] = np.hstack([np.eye(3), -manif.SO3Tangent(p).hat()])

    T_est, P = T_init, P_init.copy()
    T_list = [T_est]
    for k in range(len(u_meas)):
        # --- Predict ---
        J_self, J_tau = np.zeros((6, 6)), np.zeros((6, 6))
        T_pred = motion_model(T_est, u_meas[k], dt, J_self, J_tau)
        P_pred = J_self @ P @ J_self.T + J_tau @ Q_tangent @ J_tau.T

        # --- Update ---
        T_pred_inv = T_pred.inverse()
        z_body = np.array([T_pred_inv.act(p) for p in z[k + 1]])
        r_body = (z_body - body_points).reshape(-1)
        S = H @ P_pred @ H.T + R_diag
        K = P_pred @ H.T @ np.linalg.inv(S)
        delta = K @ r_body

        T_est = T_pred.rplus(manif.SE3Tangent(delta))
        P = (np.eye(6) - K @ H) @ P_pred

        T_list.append(T_est)

    return T_list


def run_ukf(T_init, P_init, u_meas, z, body_points, dt, Q_tangent, point_noise_std,
            ukf_alpha, ukf_beta, ukf_kappa):
    """Recursive UKF: a manifold unscented transform via right-perturbation
    retraction, alternating a predict step and an update step exactly like
    run_ekf/run_iekf, but with no Jacobians at all -- sigma points are
    retracted through the *exact* nonlinear motion_model/observation_model
    instead of a linearization of them (no manif rplus/rminus/act Jacobian
    out-parameters used here, only the bare group operations).

    Note: on a Lie group there is no linear average of poses, so the
    predicted mean is taken to be the central sigma point's own propagation
    (T_pred_mean = motion_model(T_est, u, dt)) -- this is exact, not an
    approximation, since that call *is* sigma point 0's propagation; the
    approximation is in measuring every other sigma point's spread relative
    to this reference via rminus() rather than an iterative Frechet mean, the
    standard simplification used by practical UKF-on-manifolds
    implementations. Sigma points are also *redrawn* at the update step from
    (T_pred_mean, P_pred) rather than reusing the predict-step sigma points,
    because Q_tangent is injected additively into P_pred after recombination
    (no augmented-noise sigma dimensions) -- reusing the predict sigma points
    would understate the true post-predict spread and bias the filter
    overconfident.
    Arguments:
        T_init: initial pose (manif.SE3)
        P_init: initial 6x6 tangent covariance (numpy array)
        u_meas: (N,6) array of noisy body-frame twists
        z: list of noisy point-cloud measurements (M,3)
        body_points: (M,3) array of points in the object's body frame
        dt: time step (s)
        Q_tangent: 6x6 tangent covariance of the motion-model twist increment (numpy array)
        point_noise_std: std-dev of Gaussian noise added to the predicted world-frame point-cloud measurements (m)
        ukf_alpha: unscented-transform spread parameter
        ukf_beta: unscented-transform prior-knowledge parameter (2 is optimal for Gaussian priors)
        ukf_kappa: unscented-transform secondary scaling parameter
    Returns:
        T_list: list of estimated poses (manif.SE3)
    """
    n = 6
    n_points = len(body_points)
    R_diag = point_noise_std ** 2 * np.eye(3 * n_points)
    lambda_, w_m, w_c = unscented_weights(n, ukf_alpha, ukf_beta, ukf_kappa)

    T_est, P = T_init, P_init.copy()
    T_list = [T_est]
    for k in range(len(u_meas)):
        # --- Predict ---
        offsets = unscented_sigma_offsets(P, lambda_)
        T_pred_mean = motion_model(T_est, u_meas[k], dt)

        xi_pred = np.zeros((2 * n + 1, n))
        for i in range(1, 2 * n + 1):
            T_sigma = T_est.rplus(manif.SE3Tangent(offsets[i]))
            T_pred_sigma = motion_model(T_sigma, u_meas[k], dt)
            xi_pred[i] = T_pred_sigma.rminus(T_pred_mean).coeffs()

        P_pred = Q_tangent.copy()
        for i in range(2 * n + 1):
            P_pred += w_c[i] * np.outer(xi_pred[i], xi_pred[i])

        # --- Update (fresh sigma points around the predicted mean) ---
        offsets_upd = unscented_sigma_offsets(P_pred, lambda_)
        z_sigma = np.zeros((2 * n + 1, 3 * n_points))
        for i in range(2 * n + 1):
            T_upd_sigma = T_pred_mean.rplus(manif.SE3Tangent(offsets_upd[i]))
            z_sigma_i, _ = observation_model(T_upd_sigma, body_points)
            z_sigma[i] = z_sigma_i.reshape(-1)

        z_hat = w_m @ z_sigma
        P_zz = R_diag.copy()
        P_xz = np.zeros((n, 3 * n_points))
        for i in range(2 * n + 1):
            dz = z_sigma[i] - z_hat
            P_zz += w_c[i] * np.outer(dz, dz)
            P_xz += w_c[i] * np.outer(offsets_upd[i], dz)

        r = z[k + 1].reshape(-1) - z_hat
        K = P_xz @ np.linalg.inv(P_zz)
        delta = K @ r

        T_est = T_pred_mean.rplus(manif.SE3Tangent(delta))
        P = P_pred - K @ P_zz @ K.T

        T_list.append(T_est)

    return T_list


def run_batch_gn(T_init_list, T_prior0, P_init, u_meas, z, body_points, dt, Q_tangent,
                  point_noise_std, gn_tol, gn_max_iters):
    """Batch Gauss-Newton smoother: jointly optimizes the whole trajectory
    T_0..T_N against a prior factor on T_0, N motion factors, and
    (N+1)*n_points measurement factors, all built from the same
    motion_model/observation_model as the EKF.
    Arguments:
        T_init_list: list of initial pose guesses (manif.SE3)
        T_prior0: prior pose on T_0 (manif.SE3)
        P_init: 6x6 tangent covariance of the prior on T_0 (numpy array)
        u_meas: (N,6) array of noisy body-frame twists
        z: list of noisy point-cloud measurements (M,3)
        body_points: (M,3) array of points in the object's body frame
        dt: time step (s)       
    Returns:
        T_est: list of estimated poses (manif.SE3)
    """
    n_poses = len(T_init_list)
    n_points = len(body_points)
    dof = 6 * n_poses

    Omega_prior = np.linalg.inv(P_init)
    Omega_motion = np.linalg.inv(Q_tangent)
    omega_point = 1.0 / point_noise_std ** 2

    T_est = list(T_init_list)

    for it in range(gn_max_iters):
        H = np.zeros((dof, dof))
        g = np.zeros(dof)

        # --- Prior factor on T_0 ---
        Jp = np.zeros((6, 6))
        e0 = T_est[0].rminus(T_prior0, Jp)
        e0 = e0.coeffs()
        H[0:6, 0:6] += Jp.T @ Omega_prior @ Jp
        g[0:6] += -Jp.T @ Omega_prior @ e0

        # --- Motion factors between consecutive poses ---
        for k in range(1, n_poses):
            Jc_self, Jc_tau = np.zeros((6, 6)), np.zeros((6, 6))
            T_pred = motion_model(T_est[k - 1], u_meas[k - 1], dt, Jc_self, Jc_tau)

            Ja, Jb = np.zeros((6, 6)), np.zeros((6, 6))
            e_k = T_est[k].rminus(T_pred, Ja, Jb).coeffs()

            J_prev = Jb @ Jc_self  # d e_k / d T_est[k-1], chained through T_pred
            J_curr = Ja            # d e_k / d T_est[k]

            c0, c1 = 6 * (k - 1), 6 * k
            H[c0:c0 + 6, c0:c0 + 6] += J_prev.T @ Omega_motion @ J_prev
            H[c0:c0 + 6, c1:c1 + 6] += J_prev.T @ Omega_motion @ J_curr
            H[c1:c1 + 6, c0:c0 + 6] += J_curr.T @ Omega_motion @ J_prev
            H[c1:c1 + 6, c1:c1 + 6] += J_curr.T @ Omega_motion @ J_curr
            g[c0:c0 + 6] += -J_prev.T @ Omega_motion @ e_k
            g[c1:c1 + 6] += -J_curr.T @ Omega_motion @ e_k

        # --- Measurement factors at every pose ---
        for k in range(n_poses):
            pred, Jm = observation_model(T_est[k], body_points, with_jacobian=True)
            e_meas = (z[k] - pred).reshape(-1)
            Jm = -Jm  # d(z - h(T))/dT = -dh/dT

            c = 6 * k
            H[c:c + 6, c:c + 6] += omega_point * (Jm.T @ Jm)
            g[c:c + 6] += -omega_point * (Jm.T @ e_meas)

        H += np.eye(dof) * 1e-6
        delta = np.linalg.solve(H, g)

        for k in range(n_poses):
            T_est[k] = T_est[k].rplus(manif.SE3Tangent(delta[6 * k:6 * k + 6]))

        step_norm = np.linalg.norm(delta)
        if step_norm < gn_tol:
            print(f"    Batch GN converged after {it + 1} iteration(s) (|delta| < {gn_tol})")
            break
    else:
        print(f"    Batch GN reached max iterations ({gn_max_iters}) without full convergence")

    return T_est


def pose_errors(T_true_list, T_est_list):
    """Rotation geodesic error (deg) and position error (m), per step.
    Arguments:
        T_true_list: list of true poses (manif.SE3)
        T_est_list: list of estimated poses (manif.SE3)
    Returns:
        rot_err: (N,) array of rotation errors (deg)
        pos_err: (N,) array of position errors (m)
    """
    n = len(T_true_list)
    rot_err, pos_err = np.zeros(n), np.zeros(n)
    for k in range(n):
        R_true = manif.SO3(T_true_list[k].coeffs()[3:7])
        R_est = manif.SO3(T_est_list[k].coeffs()[3:7])
        rot_err[k] = np.degrees(np.linalg.norm(R_est.rminus(R_true).coeffs()))
        pos_err[k] = np.linalg.norm(T_true_list[k].translation() - T_est_list[k].translation())
    return rot_err, pos_err


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("--duration", type=float, default=5.0, help="Simulation length in seconds")
    parser.add_argument("--dt", type=float, default=0.1, help="Motion/measurement step interval in seconds")
    parser.add_argument("--n-points", type=int, default=20, help="Number of body-frame point-cloud landmarks")
    
    parser.add_argument("--vel-noise-std", type=float, default=0.05, help="Input linear-velocity noise std-dev (m/s)")
    parser.add_argument("--gyro-noise-std", type=float, default=0.02, help="Input angular-velocity noise std-dev (rad/s)")
    parser.add_argument("--point-noise-std", type=float, default=0.03, help="Point-cloud measurement noise std-dev (m)")
    parser.add_argument("--init-pose-noise-std", type=float, default=0.1, help="Std-dev used to perturb the initial pose guess and set the prior/EKF-init covariance")
    
    parser.add_argument("--gn-tol", type=float, default=1e-6, help="Batch Gauss-Newton convergence tolerance")
    parser.add_argument("--gn-max-iters", type=int, default=20, help="Maximum batch Gauss-Newton iterations")

    parser.add_argument("--ukf-alpha", type=float, default=1.0, help="UKF unscented-transform spread parameter")
    parser.add_argument("--ukf-beta", type=float, default=2.0, help="UKF unscented-transform prior-knowledge parameter (2 is optimal for Gaussian priors)")
    parser.add_argument("--ukf-kappa", type=float, default=-3.0, help="UKF unscented-transform secondary scaling parameter (default gives n+kappa=3 for this script's 6-dim tangent state)")

    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")
    
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    body_points, T_true, u_meas, z = generate_ground_truth_and_data(
        args.duration, args.dt, args.n_points, args.vel_noise_std, args.gyro_noise_std,
        args.point_noise_std, rng,
    )

    # Shared, uncertain initial pose guess for every method (EKF init and GN prior).
    init_offset = rng.normal(0.0, args.init_pose_noise_std, 6)
    T_init = T_true[0].rplus(manif.SE3Tangent(init_offset))
    P_init = args.init_pose_noise_std ** 2 * np.eye(6)

    # Process noise covariance of the twist *rate*; the tangent increment fed to
    # motion_model is twist*dt, so its covariance is dt**2 times this.
    Q_rate = np.diag(np.concatenate([
        [args.vel_noise_std ** 2] * 3,
        [args.gyro_noise_std ** 2] * 3,
    ]))
    Q_tangent = args.dt ** 2 * Q_rate

    n_steps = len(u_meas)

    print("Running dead-reckoning baseline (motion model only, no point-cloud correction)...")
    T_dr, time_dr, mem_dr = measure_performance(
        run_dead_reckoning, T_init, u_meas, args.dt, n_steps=n_steps)

    print("Running recursive EKF (motion-model predict + point-cloud observation-model update)...")
    T_ekf, time_ekf, mem_ekf = measure_performance(
        run_ekf, T_init, P_init, u_meas, z, body_points, args.dt, Q_tangent, args.point_noise_std,
        n_steps=n_steps)

    print("Running invariant EKF (body-frame residual, state-independent measurement Jacobian)...")
    T_iekf, time_iekf, mem_iekf = measure_performance(
        run_iekf, T_init, P_init, u_meas, z, body_points, args.dt, Q_tangent, args.point_noise_std,
        n_steps=n_steps)

    print("Running batch Gauss-Newton (joint optimization over the whole trajectory)...")
    T_gn, time_gn, mem_gn = measure_performance(
        run_batch_gn, T_dr, T_init, P_init, u_meas, z, body_points, args.dt, Q_tangent,
        args.point_noise_std, args.gn_tol, args.gn_max_iters, n_steps=n_steps)

    print("Running UKF (manifold unscented transform via right-perturbation retraction)...")
    T_ukf, time_ukf, mem_ukf = measure_performance(
        run_ukf, T_init, P_init, u_meas, z, body_points, args.dt, Q_tangent, args.point_noise_std,
        args.ukf_alpha, args.ukf_beta, args.ukf_kappa, n_steps=n_steps)

    rot_err_dr, pos_err_dr = pose_errors(T_true, T_dr)
    rot_err_ekf, pos_err_ekf = pose_errors(T_true, T_ekf)
    rot_err_iekf, pos_err_iekf = pose_errors(T_true, T_iekf)
    rot_err_gn, pos_err_gn = pose_errors(T_true, T_gn)
    rot_err_ukf, pos_err_ukf = pose_errors(T_true, T_ukf)

    print("\nFinal / RMS errors:")
    for name, rot_err, pos_err in [
        ("Dead-reckoning", rot_err_dr, pos_err_dr),
        ("EKF (recursive)", rot_err_ekf, pos_err_ekf),
        ("IEKF (invariant)", rot_err_iekf, pos_err_iekf),
        ("Batch GN", rot_err_gn, pos_err_gn),
        ("UKF (unscented)", rot_err_ukf, pos_err_ukf),
    ]:
        print(f"  {name:<18s} final rot={rot_err[-1]:7.3f} deg, pos={pos_err[-1]:7.4f} m | "
              f"RMS rot={np.sqrt(np.mean(rot_err**2)):7.3f} deg, pos={np.sqrt(np.mean(pos_err**2)):7.4f} m")

    print("\nAverage time complexity + space complexity per approach (per-step, empirical):")
    for name, avg_time, avg_mem in [
        ("Dead-reckoning", time_dr, mem_dr),
        ("EKF (recursive)", time_ekf, mem_ekf),
        ("IEKF (invariant)", time_iekf, mem_iekf),
        ("Batch GN", time_gn, mem_gn),
        ("UKF (unscented)", time_ukf, mem_ukf),
    ]:
        print(f"  {name:<18s} avg time={avg_time * 1e6:9.2f} µs/step | avg peak mem={avg_mem / 1024.0:9.3f} KB/step")

    t_hist = np.arange(len(T_true)) * args.dt
    fig, (ax_rot, ax_pos, ax_traj) = plt.subplots(3, 1, figsize=(9, 11))

    for ax, err_dr, err_ekf, err_iekf, err_gn, err_ukf, ylabel in [
        (ax_rot, rot_err_dr, rot_err_ekf, rot_err_iekf, rot_err_gn, rot_err_ukf, "Rotation error (deg)"),
        (ax_pos, pos_err_dr, pos_err_ekf, pos_err_iekf, pos_err_gn, pos_err_ukf, "Position error (m)"),
    ]:
        ax.plot(t_hist, err_dr, label="Dead-reckoning (prior only)", color="tab:gray")
        ax.plot(t_hist, err_ekf, label="EKF (recursive)", color="tab:red")
        ax.plot(t_hist, err_iekf, label="IEKF (invariant)", color="tab:green")
        ax.plot(t_hist, err_gn, label="Batch Gauss-Newton", color="tab:blue")
        ax.plot(t_hist, err_ukf, label="UKF (unscented)", color="tab:purple")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.legend()
    ax_rot.set_title("Point-cloud pose tracking: prior-only vs. EKF vs. invariant EKF vs. batch Gauss-Newton vs. UKF")
    ax_pos.set_xlabel("Time (s)")

    def xy(T_list):
        pts = np.array([T.translation() for T in T_list])
        return pts[:, 0], pts[:, 1]

    ax_traj.plot(*xy(T_true), label="Ground truth", color="black", linewidth=2)
    ax_traj.plot(*xy(T_dr), label="Dead-reckoning (prior only)", color="tab:gray", linestyle="--")
    ax_traj.plot(*xy(T_ekf), label="EKF (recursive)", color="tab:red", linestyle="--")
    ax_traj.plot(*xy(T_iekf), label="IEKF (invariant)", color="tab:green", linestyle="--")
    ax_traj.plot(*xy(T_gn), label="Batch Gauss-Newton", color="tab:blue", linestyle="--")
    ax_traj.plot(*xy(T_ukf), label="UKF (unscented)", color="tab:purple", linestyle="--")
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.axis("equal")
    ax_traj.legend()

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
