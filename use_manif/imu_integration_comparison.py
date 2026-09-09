'''
Compares two ways of integrating a noisy IMU (gyro + body-frame velocity) stream
into a pose estimate:

  1. "naive" - orientation kept as a 3-vector of Euler angles (roll, pitch, yaw),
     updated by directly adding the measured body angular rate: euler += omega*dt.
     This treats SO(3) as if it were a flat vector space, which is only valid to
     first order and breaks down as soon as the attitude has any appreciable
     rotation (it ignores the nonlinear Euler-rate kinematic transform).

  2. "expmap" - orientation and position kept as a manif.SE3 pose, updated on the
     SE(3) manifold itself via T *= Exp([v, omega]*dt), manif's closed-form
     exponential map.

Both estimators consume the *same* noisy measurements each step, so any gap
between them isolates the error introduced by the naive vector-space
approximation itself (on top of the noise both share), rather than being an
artifact of different noise draws.

Ground truth is generated the same way as the exp-map estimator, by SE(3)
exp-map-integrating the noise-free body twist [v, omega]. This is exact under
the zero-order-hold assumption of a constant body twist within each dt: unlike
a decoupled "rotate then translate" Euler step, the SE(3) exponential map
couples the rotation and translation parts of the update, which is what makes
it the true closed-form solution for constant-twist motion.

This version fully delegates the manifold math to the manif library
(https://github.com/artivis/manif) via its Python bindings (manifpy), instead
of hand-rolled skew-symmetric / Rodrigues-formula numpy code. Correspondence:
  so3_exp(omega*dt), R *= ...        -> manif.SE3Tangent([v,omega]*dt).exp() /
                                          T.rplus(tangent)
  rotation_geodesic_error(Ra, Rb)    -> np.linalg.norm(Rb.rminus(Ra).coeffs())
  R @ v                              -> R.act(v)
manif's SE3 tangent vector layout is [vx, vy, vz, wx, wy, wz]; its SE3.coeffs()
layout is [tx, ty, tz, qx, qy, qz, qw], so the orientation sub-object is
recovered via manif.SO3(T.coeffs()[3:7]).

The naive estimator's own update rule (euler += omega*dt) is deliberately kept
as plain vector arithmetic - that flat-vector-space treatment of SO(3) is
exactly the thing being tested against manif's proper manifold integration.
Its resulting attitude is still wrapped in a manif.SO3 (via a small Euler ->
quaternion helper) so that error/rotation are computed with the same manif
operators used everywhere else.

Appendix:
- Gyroscopes: Measure angular velocity (rotation rate in degrees per second, °/s) around the X, Y, and Z axes.
    Integrating angular velocity over time gives you the change in orientation (angles like roll, pitch, and yaw).
- Accelerometers: Measure linear acceleration (m/s²) along the X, Y, and Z axes.
    By integrating acceleration over time, you can estimate velocity, and by integrating velocity, you can estimate position.
    However, this double integration can lead to significant drift over time due to noise and biases in the measurements.

'''

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import manifpy as manif

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import true_body_rates


def euler_to_quat_xyzw(rpy):
    """
    ZYX (roll-pitch-yaw) Euler angles -> quaternion [x, y, z, w].

    This is NOT a manifold operation - it's the naive estimator's own flat
    vector-space attitude state, converted just far enough to hand off to
    manif.SO3 for everything downstream (act, rminus, error metrics).
    """
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def run_simulation(duration, dt, gyro_noise_std, vel_noise_std, seed):
    """Simulation of naive vs. exp-map integration of noisy IMU measurements.

    Args:
        duration: Simulation length in seconds
        dt: IMU sample interval in seconds
        gyro_noise_std: Standard deviation of gyroscope noise
        vel_noise_std: Standard deviation of velocity noise
        seed: Random seed

    Returns:
        t_hist: Simulation history (time steps)
        rot_err_naive: Rotation error for naive estimator
        rot_err_exp: Rotation error for exp-map estimator
        pos_err_naive: Position error for naive estimator
        pos_err_exp: Position error for exp-map estimator
    """
    
    rng = np.random.default_rng(seed)
    n_steps = int(duration / dt)

    T_gt = manif.SE3.Identity()  # Ground truth pose, exact SE(3) exp-map integration

    # naive estimator's own attitude state (roll,pitch,yaw) and position
    naive_euler = np.zeros(3) #rotation angles (roll, pitch, yaw)
    p_naive = np.zeros(3)     # translation vector (x, y, z)

    # exp-map estimator pose, proper SE(3) manifold update (both rotation and translation)
    T_exp = manif.SE3.Identity()  
    
    t_hist = np.zeros(n_steps)

    # rotation error + position error for naive estimator
    rot_err_naive = np.zeros(n_steps)
    pos_err_naive = np.zeros(n_steps)

    # rotation error + position error for exp-map estimator
    rot_err_exp = np.zeros(n_steps)
    pos_err_exp = np.zeros(n_steps)

    for k in range(n_steps):

        t = k * dt

        # Ground truth: get the true body rates (angular velocity and linear velocity) at time t
        omega_gt, v_gt = true_body_rates(t)

        # Ground truth: exact SE(3) exp-map integration of the noise-free body twist.
        T_gt = T_gt.rplus(manif.SE3Tangent(np.concatenate([v_gt, omega_gt]) * dt))

        # Shared noisy IMU measurement.

        # Gyroscope noise: Gaussian noise added to the true angular velocity.
        # This simulates the effect of sensor noise on the angular velocity measurements from gyroscope.
        omega_meas = omega_gt + rng.normal(0.0, gyro_noise_std, 3)

        # Body-velocity noise: Gaussian noise added to the true body-frame velocity.
        # This simulates the effect of sensor noise on the linear velocity measurements from accelerometer.
        v_meas = v_gt + rng.normal(0.0, vel_noise_std, 3)

        # Naive: Euler angles treated as a flat vector space.
        naive_euler = naive_euler + omega_meas * dt
        R_naive = manif.SO3(euler_to_quat_xyzw(naive_euler))
        p_naive = p_naive + R_naive.act(v_meas) * dt

        # Exp-map: proper SE(3) manifold update.
        T_exp = T_exp.rplus(manif.SE3Tangent(np.concatenate([v_meas, omega_meas]) * dt))

        t_hist[k] = t
        
        R_gt = manif.SO3(T_gt.coeffs()[3:7])
        R_exp = manif.SO3(T_exp.coeffs()[3:7])
        
        rot_err_naive[k] = np.degrees(np.linalg.norm(R_naive.rminus(R_gt).coeffs()))
        rot_err_exp[k] = np.degrees(np.linalg.norm(R_exp.rminus(R_gt).coeffs()))
        
        pos_err_naive[k] = np.linalg.norm(T_gt.translation() - p_naive)
        pos_err_exp[k] = np.linalg.norm(T_gt.translation() - T_exp.translation())

    return t_hist, rot_err_naive, rot_err_exp, pos_err_naive, pos_err_exp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation length in seconds")
    parser.add_argument("--dt", type=float, default=0.005, help="IMU sample interval in seconds")
    parser.add_argument("--gyro-noise-std", type=float, default=0.02, help="Gyro noise std-dev (rad/s)")
    parser.add_argument("--vel-noise-std", type=float, default=0.05, help="Body-velocity noise std-dev (m/s)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Save the figure to this path instead of showing it")
    args = parser.parse_args()

    t, rot_err_naive, rot_err_exp, pos_err_naive, pos_err_exp = run_simulation(
        args.duration, args.dt, args.gyro_noise_std, args.vel_noise_std, args.seed,
    )

    fig, (ax_rot, ax_pos) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax_rot.plot(t, rot_err_naive, label="naive (Euler-angle, vector-space)", color="tab:red")
    ax_rot.plot(t, rot_err_exp, label="Exp-map (SO(3))", color="tab:blue")
    ax_rot.set_ylabel("Rotation error (deg)")
    ax_rot.set_yscale("log")
    ax_rot.legend()
    ax_rot.set_title("Attitude error: naive Euler-angle vs. SO(3) Exp-map integration")

    ax_pos.plot(t, pos_err_naive, label="naive (Euler-angle, vector-space)", color="tab:red")
    ax_pos.plot(t, pos_err_exp, label="Exp-map (SE(3))", color="tab:blue")
    ax_pos.set_ylabel("Position error (m)")
    ax_pos.set_xlabel("Time (s)")
    ax_pos.set_yscale("log")
    ax_pos.legend()

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
