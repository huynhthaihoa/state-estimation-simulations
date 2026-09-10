'''
This script simulates a robot attempting to track a moving target trajectory over
several timesteps under the same unbalanced noise profile
(highly accurate GPS translation, but imperfect IMU).

Same simulation as robot_imu_simulation.py, but the SE(3) manifold math (Exp/Log
maps, right-minus, inverse right Jacobian) is delegated to the manif library
(https://github.com/artivis/manif) instead of hand-rolled numpy formulas.

Key correspondence with the original script:
  se3_exp(xi)                         -> manif.SE3Tangent(xi).exp() / T.rplus(tangent)
  se3_log(inv(T_true) @ T_est)        -> T_est.rminus(T_true)
  compute_se3_inv_right_jacobian(e)   -> the J_self Jacobian rminus() writes out,
                                          i.e. the Jacobian of (T_est rminus T_true)
                                          with respect to a right-perturbation of T_est.
manif's tangent vector layout is [vx, vy, vz, wx, wy, wz], matching the original
script's [linear, angular] convention.
'''

import numpy as np
import argparse
import manifpy as manif


def run_simulation(dt_imu, total_seconds, snapshots_per_second, gn_tol, gn_max_iters,
                    max_linear_vel, max_angular_vel, omega_info, rng):
    """Runs the full IMU-propagation + low-rate correction timeline (100 Hz IMU
    guesses, 1 Hz Gauss-Newton correction against a precise GPS-style position
    measurement), printing the same interactive trace the original inline
    __main__ block did.
    Arguments:
        dt_imu: IMU update interval (s)
        total_seconds: total simulated duration (s)
        snapshots_per_second: number of IMU tracking snapshots to log per second
        gn_tol: Gauss-Newton convergence tolerance on the correction step norm
        gn_max_iters: maximum Gauss-Newton iterations per per-second correction
        max_linear_vel: max magnitude of the (fixed, randomly drawn) linear body-rate component (m/s)
        max_angular_vel: max magnitude of the (fixed, randomly drawn) angular body-rate component (rad/s)
        omega_info: (6,6) information (inverse-covariance) matrix for the correction step
        rng: numpy random Generator (drives the true twist velocity and IMU noise)
    Returns:
        T_true: final ground-truth pose (manif.SE3)
        T_est: final estimated pose (manif.SE3)
        pre_correction_errors: list of position error (m) before each second's correction
        post_correction_errors: list of position error (m) after each second's correction
    """
    # 1. Setup Initial States (Both ground-truth and robot estimate start at origin)
    T_true = manif.SE3.Identity()
    T_est = manif.SE3.Identity()

    # The true movement of the robot: a randomized 6D body-frame twist velocity
    # [vx, vy, vz, wx, wy, wz] -- linear velocity in m/s, angular velocity in rad/s
    true_twist_velocity = np.concatenate([
        rng.uniform(-max_linear_vel, max_linear_vel, 3),   # linear velocity ∈ [-max_linear_vel, max_linear_vel] m/s per axis
        rng.uniform(-max_angular_vel, max_angular_vel, 3)    # angular velocity ∈ [-max_angular_vel, max_angular_vel] rad/s per axis
    ])

    print("Starting IMU Propagation Timeline Simulation (manif backend)...")
    print("IMU frequency: 100 Hz | Main Global Sensor updates: 1 Hz")
    print(f"Initial State: X={T_est.translation()[0]:.2f}, Y={T_est.translation()[1]:.2f}\n")
    print("-" * 90)

    # Master Timeline Loop
    imu_steps_per_second = int(1.0 / dt_imu)
    snapshot_interval = max(1, imu_steps_per_second // snapshots_per_second)

    pre_correction_errors, post_correction_errors = [], []

    for second in range(1, total_seconds + 1):
        print(f"\n[START OF SECOND {second}] --- High Frequency IMU Guessing Phase ---")

        # --- PHASE 1: IMU VELOCITY PROPAGATION (100 Micro-Steps) ---
        for micro_step in range(imu_steps_per_second):
            # Move the real-world trajectory forward
            T_true = T_true.rplus(manif.SE3Tangent(true_twist_velocity * dt_imu))

            # Simulate a slightly imperfect IMU reading (adding minor motion noise)
            # The robot reads this velocity and estimates its position blindly
            imu_reading = true_twist_velocity + rng.normal(0, 0.02, 6)
            T_est = T_est.rplus(manif.SE3Tangent(imu_reading * dt_imu))

            # Print a few snippets of the fast intermediate tracking
            if micro_step % snapshot_interval == 0:
                est_pos = T_est.translation()
                print(f"  └─ IMU Step {micro_step:02d}: Est Pos = [{est_pos[0]:.3f}, {est_pos[1]:.3f}]")

        # --- PHASE 2: LOWER FREQUENCY POSITION FILTERING (1 Hz Update) ---
        print(f"\n[SENSOR TICK] Global Position Measurement Arrived!")
        pre_correction_error = np.linalg.norm(T_true.translation() - T_est.translation())
        pre_correction_errors.append(pre_correction_error)
        print(f"  Pre-Correction Error Distance: {pre_correction_error:.4f} meters")

        # Run local optimization cycles using manif's manifold operators, iterating
        # until the correction step shrinks below gn_tol (or gn_max_iters is hit)
        for opt_iter in range(gn_max_iters):
            # Compute tracking residual directly on the manifold: e = Log(T_true^-1 T_est),
            # expressed as a right-perturbation of T_est, together with its analytical Jacobian
            J = np.zeros((6, 6))
            e_vector = T_est.rminus(T_true, J).coeffs()

            # Solve normal system with our confidence profile
            H = np.dot(J.T, np.dot(omega_info, J)) + np.eye(6) * 1e-4
            g = -np.dot(J.T, np.dot(omega_info, e_vector))
            delta_xi = np.linalg.solve(H, g)

            # Correct the estimate matrix
            T_est = T_est.rplus(manif.SE3Tangent(delta_xi))

            step_norm = np.linalg.norm(delta_xi)
            print(f"    ├─ GN iter {opt_iter + 1}: |delta_xi| = {step_norm:.8f}")
            if step_norm < gn_tol:
                print(f"    └─ Converged after {opt_iter + 1} iteration(s) (|delta_xi| < {gn_tol})")
                break
        else:
            print(f"    └─ Reached max iterations ({gn_max_iters}) without full convergence")

        true_pos, est_pos = T_true.translation(), T_est.translation()
        post_correction_error = np.linalg.norm(true_pos - est_pos)
        post_correction_errors.append(post_correction_error)
        print(f"  Post-Correction Status: True X,Y = [{true_pos[0]:.3f}, {true_pos[1]:.3f}] | Est X,Y = [{est_pos[0]:.3f}, {est_pos[1]:.3f}]")
        print(f"  Final Post-Correction Error Distance: {post_correction_error:.4f} meters")
        print("-" * 90)

    print("\nTracking Loop Concluded.")

    return T_true, T_est, pre_correction_errors, post_correction_errors


# --- Main IMU & Sensor Update Simulation ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Simulation Parameters Configuration
    parser.add_argument("--dt-imu", type=float, default=0.01, help="IMU update interval in seconds (default: 0.01s for 100Hz)")
    parser.add_argument("--total-seconds", type=int, default=3, help="Total simulation duration in seconds")
    parser.add_argument("--snapshots-per-second", type=int, default=4, help="Number of IMU tracking snapshots to log per simulated second")

    # Gauss-Newton Convergence Criteria configuration
    parser.add_argument("--gn-tol", type=float, default=1e-6, help="Gauss-Newton convergence tolerance")
    parser.add_argument("--gn-max-iters", type=int, default=10, help="Maximum Gauss-Newton iterations")

    # Velocity Magnitude Configuration (for random twist generation)
    parser.add_argument("--max-linear-vel", type=float, default=1.0, help="Maximum linear velocity magnitude in m/s (default: 1.0)")
    parser.add_argument("--max-angular-vel", type=float, default=0.5, help="Maximum angular velocity magnitude in rad/s (default: 0.5)")

    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")

    args = parser.parse_args()

    np.set_printoptions(suppress=True, precision=4)

    rng = np.random.default_rng(args.seed)

    # Setup Correction Weights (Precision GPS for position, noisy gyro)
    unbalanced_cov = np.zeros((6, 6))
    np.fill_diagonal(unbalanced_cov, [0.001, 0.001, 0.001, 1000.0, 1000.0, 1000.0])
    omega_info = np.linalg.inv(unbalanced_cov)

    run_simulation(args.dt_imu, args.total_seconds, args.snapshots_per_second, args.gn_tol,
                   args.gn_max_iters, args.max_linear_vel, args.max_angular_vel, omega_info, rng)
