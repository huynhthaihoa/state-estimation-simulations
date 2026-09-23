'''
This script simulates a robot attempting to track a moving target trajectory over
several timesteps against a genuinely noisy, position-only "Global Position
Measurement" (GPS-style: it reads translation, never orientation) fused with an
imperfect IMU's dead-reckoned twist propagation.
'''

import numpy as np
import argparse

from lie_utils import se3_exp


def position_observation_jacobian(T_est):
    """d(position)/d(xi) for a right perturbation T_est <- T_est @ Exp(xi),
    xi=[v,omega] (this codebase's translation-first tangent convention): only
    the linear-velocity block affects translation to first order, so this is
    [R_est | 0] -- verified against finite differences in this module's tests.
    Factored out from run_simulation's correction loop so it's independently
    testable: this Jacobian's all-zero angular block is exactly what makes a
    position-only measurement structurally unable to correct orientation, no
    matter how it's weighted -- the property this module's tests check for
    directly, not just inferred from behavior.
    Arguments:
        T_est: current pose estimate (4,4)
    Returns:
        J: (3,6) Jacobian of position wrt xi
    """
    return np.hstack([T_est[0:3, 0:3], np.zeros((3, 3))])


def run_simulation(dt_imu, total_seconds, snapshots_per_second, gn_tol, gn_max_iters,
                    max_linear_vel, max_angular_vel, pos_noise_std, pos_info, rng):
    """Runs the full IMU-propagation + low-rate correction timeline (100 Hz IMU
    guesses, 1 Hz Gauss-Newton correction against a noisy GPS-style position
    measurement), printing the same interactive trace the original inline
    __main__ block did.

    The correction is a genuine position-only observation, z = T_true's
    translation + noise -- not T_true itself. Its Jacobian wrt a right
    perturbation xi=[v,omega] applied to T_est is [R_est | 0]: only the
    linear-velocity block affects position to first order (see module
    tests, verified against finite differences), so this measurement can
    only ever correct translation. Orientation is never touched by it and
    relies entirely on the IMU's own dead-reckoning between corrections --
    the realistic version of "precision GPS, imperfect gyro": GPS supplies
    no orientation information at all, rather than a full 6-DoF pose
    reading merely *weighted* to trust rotation less.
    Arguments:
        dt_imu: IMU update interval (s)
        total_seconds: total simulated duration (s)
        snapshots_per_second: number of IMU tracking snapshots to log per second
        gn_tol: Gauss-Newton convergence tolerance on the correction step norm
        gn_max_iters: maximum Gauss-Newton iterations per per-second correction
        max_linear_vel: max magnitude of the (fixed, randomly drawn) linear body-rate component (m/s)
        max_angular_vel: max magnitude of the (fixed, randomly drawn) angular body-rate component (rad/s)
        pos_noise_std: std-dev (m) of the position measurement's per-axis Gaussian noise
        pos_info: (3,3) information (inverse-covariance) matrix for the position correction
        rng: numpy random Generator (drives the true twist velocity, IMU noise, and GPS noise)
    Returns:
        T_true: final ground-truth pose (4,4)
        T_est: final estimated pose (4,4)
        pre_correction_errors: list of position error (m) before each second's correction
        post_correction_errors: list of position error (m) after each second's correction
    """
    # 1. Setup Initial States (Both ground-truth and robot estimate start at origin)
    T_true = np.eye(4)
    T_est = np.eye(4)

    # The true movement of the robot: a randomized 6D body-frame twist velocity
    # [vx, vy, vz, wx, wy, wz] -- linear velocity in m/s, angular velocity in rad/s
    true_twist_velocity = np.concatenate([
        rng.uniform(-max_linear_vel, max_linear_vel, 3),   # linear velocity ∈ [-max_linear_vel, max_linear_vel] m/s per axis
        rng.uniform(-max_angular_vel, max_angular_vel, 3)    # angular velocity ∈ [-max_angular_vel, max_angular_vel] rad/s per axis
    ])

    print("Starting IMU Propagation Timeline Simulation...")
    print("IMU frequency: 100 Hz | Main Global Sensor updates: 1 Hz")
    print(f"Initial State: X={T_est[0,3]:.2f}, Y={T_est[1,3]:.2f}\n")
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
            T_true = np.dot(T_true, se3_exp(true_twist_velocity * dt_imu))

            # Simulate a slightly imperfect IMU reading (adding minor motion noise)
            # The robot reads this velocity and estimates its position blindly
            imu_reading = true_twist_velocity + rng.normal(0, 0.02, 6)
            T_est = np.dot(T_est, se3_exp(imu_reading * dt_imu))

            # Print a few snippets of the fast intermediate tracking
            if micro_step % snapshot_interval == 0:
                print(f"  └─ IMU Step {micro_step:02d}: Est Pos = [{T_est[0,3]:.3f}, {T_est[1,3]:.3f}]")

        # --- PHASE 2: LOWER FREQUENCY POSITION FILTERING (1 Hz Update) ---
        print("\n[SENSOR TICK] Global Position Measurement Arrived!")
        # A genuine GPS-style reading: the true position plus per-axis Gaussian
        # noise -- not T_true itself. This is the one measurement T_est ever
        # gets corrected against, and it carries no orientation information.
        z_pos = T_true[0:3, 3] + rng.normal(0.0, pos_noise_std, 3)

        pre_correction_error = np.linalg.norm(T_true[0:3, 3] - T_est[0:3, 3])
        pre_correction_errors.append(pre_correction_error)
        print(f"  Pre-Correction Error Distance: {pre_correction_error:.4f} meters")

        # Run local optimization cycles using the manifold formulas, iterating
        # until the correction step shrinks below gn_tol (or gn_max_iters is hit)
        for opt_iter in range(gn_max_iters):
            # Position-only residual: r = z_pos - T_est's translation.
            r_vector = z_pos - T_est[0:3, 3]

            J = position_observation_jacobian(T_est)

            # Solve normal system with our confidence profile
            # H is the Gauss-Newton approximate Hessian
            H = np.dot(J.T, np.dot(pos_info, J)) + np.eye(6) * 1e-4
            g = np.dot(J.T, np.dot(pos_info, r_vector))
            delta_xi = np.linalg.solve(H, g)

            # T_est ← T_est · Exp(δξ): correct the estimate matrix
            T_est = np.dot(T_est, se3_exp(delta_xi))

            step_norm = np.linalg.norm(delta_xi)
            print(f"    ├─ GN iter {opt_iter + 1}: |delta_xi| = {step_norm:.8f}")
            if step_norm < gn_tol:
                print(f"    └─ Converged after {opt_iter + 1} iteration(s) (|delta_xi| < {gn_tol})")
                break
        else:
            print(f"    └─ Reached max iterations ({gn_max_iters}) without full convergence")

        post_correction_error = np.linalg.norm(T_true[0:3, 3] - T_est[0:3, 3])
        post_correction_errors.append(post_correction_error)
        print(f"  Post-Correction Status: True X,Y = [{T_true[0,3]:.3f}, {T_true[1,3]:.3f}] | Est X,Y = [{T_est[0,3]:.3f}, {T_est[1,3]:.3f}]")
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

    # Position measurement noise (the "precision GPS" -- precise, but not exact)
    parser.add_argument("--pos-noise-std", type=float, default=0.05,
                         help="Per-axis std-dev (m) of the Global Position Measurement's Gaussian noise (default: 0.05)")

    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")

    args = parser.parse_args()

    np.set_printoptions(suppress=True, precision=4)

    rng = np.random.default_rng(args.seed)

    pos_info = np.eye(3) / args.pos_noise_std ** 2

    run_simulation(args.dt_imu, args.total_seconds, args.snapshots_per_second, args.gn_tol,
                   args.gn_max_iters, args.max_linear_vel, args.max_angular_vel,
                   args.pos_noise_std, pos_info, rng)
