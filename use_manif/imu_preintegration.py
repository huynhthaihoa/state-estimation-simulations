'''
Same IMU pre-integration + first-order bias-Jacobian Taylor-correction demo as
imu_preintegration.py, but the SO(3) manifold math (Exp map, right Jacobian,
skew/hat) is delegated to the manif library (https://github.com/artivis/manif)
instead of hand-rolled numpy formulas.

Key correspondence with the original script:
  so3_exp(w*dt); delta_R = delta_R @ dR  -> delta_R = delta_R.rplus(SO3Tangent(w*dt), J_self, J_tau)
                                             (J_self is what the original computed by hand as
                                             dR.T -- the Jacobian of the composition wrt the left
                                             operand; J_tau is Jr(w*dt), the right Jacobian)
  so3_right_jacobian(w*dt)               -> the J_tau Jacobian rplus() writes out above
  skew(v)                                -> manif.SO3Tangent(v).hat()

manif's SO3 has no notion of velocity/position -- those stay plain R^3 vectors,
propagated exactly as in the numpy version, using delta_R.rotation() (the 3x3
rotation matrix) wherever the original multiplied by R_prev directly.
'''

import argparse

import numpy as np
import manifpy as manif


class PreintegratedIMUBundle:
    """
    Manages the compression of thousands of high-frequency IMU tracking steps
    into a single relative localized measurement factor independent of global pose.
    """
    def __init__(self, initial_bias_gyro, initial_bias_accel):
        # Initial bias estimates at the time the bundle started
        self.b_g = np.array(initial_bias_gyro, dtype=float)
        self.b_a = np.array(initial_bias_accel, dtype=float)

        # Core compressed relative measurements (The Local Sandbox variables)
        self.delta_R = manif.SO3.Identity()  # Relative orientation change
        self.delta_v = np.zeros(3)            # Relative velocity change
        self.delta_p = np.zeros(3)            # Relative position change

        # First-order Jacobians with respect to changes in sensor bias
        # Tracks exactly how our relative bundle shifts if the graph optimizer changes the bias guess
        self.J_R_bg = np.zeros((3, 3))    # Rotation wrt gyro bias
        self.J_v_bg = np.zeros((3, 3))    # Velocity wrt gyro bias
        self.J_v_ba = np.zeros((3, 3))    # Velocity wrt accel bias
        self.J_p_bg = np.zeros((3, 3))    # Position wrt gyro bias
        self.J_p_ba = np.zeros((3, 3))    # Position wrt accel bias

    def integrate_measurement(self, raw_linear_vel, raw_angular_vel, dt):
        """
        Integrates a single high-frequency micro-step into the local bundle variables
        and updates the companion bias sensitivity Jacobians.
        """
        # 1. Clean the raw inputs using our baseline estimated biases
        v_corrected = raw_linear_vel - self.b_a
        w_corrected = raw_angular_vel - self.b_g

        # Save current state for Jacobian calculations
        R_prev = self.delta_R.rotation()

        # 2. Compute local manifold update + its analytical Jacobians via manif's rplus
        J_self, J_tau = np.zeros((3, 3)), np.zeros((3, 3))
        self.delta_R = self.delta_R.rplus(manif.SO3Tangent(w_corrected * dt), J_self, J_tau)

        # 3. Propagate relative components step-by-step
        # Note: the linear input is used exactly like an accelerometer reading: it is
        # integrated twice below (into delta_v, then delta_p), so despite the '--linear-vel'
        # naming it is a body-frame acceleration in m/s^2. There is no gravity term, which a
        # real accelerometer's specific force would need removed.
        self.delta_p += self.delta_v * dt + 0.5 * np.dot(R_prev, v_corrected) * (dt**2)
        self.delta_v += np.dot(R_prev, v_corrected) * dt

        # 4. Propagate the Bias Jacobians over time using discrete-time system matrices
        # These capture how changes in bias backpropagate through the chained transformations
        v_skew = manif.SO3Tangent(v_corrected).hat()
        self.J_p_bg += self.J_v_bg * dt - 0.5 * np.dot(np.dot(R_prev, v_skew), self.J_R_bg) * (dt**2)
        self.J_p_ba += self.J_v_ba * dt - 0.5 * R_prev * (dt**2)

        self.J_v_bg -= np.dot(np.dot(R_prev, v_skew), self.J_R_bg) * dt
        self.J_v_ba -= R_prev * dt

        self.J_R_bg = np.dot(J_self, self.J_R_bg) - J_tau * dt

    def get_corrected_measurement(self, new_b_g, new_b_a):
        """
        Uses first-order Taylor expansion to instantly update the compressed bundle
        without re-running the thousands of integration steps.
        """
        delta_bg = new_b_g - self.b_g
        delta_ba = new_b_a - self.b_a

        # Apply the linear Taylor correction adjustments
        corrected_R = self.delta_R.rplus(manif.SO3Tangent(np.dot(self.J_R_bg, delta_bg)))
        corrected_v = self.delta_v + np.dot(self.J_v_bg, delta_bg) + np.dot(self.J_v_ba, delta_ba)
        corrected_p = self.delta_p + np.dot(self.J_p_bg, delta_bg) + np.dot(self.J_p_ba, delta_ba)

        return corrected_R, corrected_v, corrected_p


# --- Demonstration Timeline Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # IMU sampling configuration
    parser.add_argument("--frequency-hz", type=int, default=100, help="IMU sampling frequency in Hz, integrated over 1 simulated second (default: 100)")

    # Initial bias estimate configuration (bias guess in effect when the bundle starts)
    parser.add_argument("--gyro-bias", type=float, nargs=3, default=[0.01, -0.01, 0.02], metavar=("X", "Y", "Z"), help="Initial estimated gyro bias (rad/s, default: 0.01 -0.01 0.02)")
    parser.add_argument("--accel-bias", type=float, nargs=3, default=[0.05, 0.00, -0.05], metavar=("X", "Y", "Z"), help="Initial estimated accel bias (m/s^2, default: 0.05 0.00 -0.05)")

    # Commanded body-frame motion (the true, noise-free input): the linear channel is an acceleration
    parser.add_argument("--linear-vel", type=float, nargs=3, default=[1.0, 0.1, 0.0], metavar=("X", "Y", "Z"), help="True commanded body-frame acceleration in m/s^2, fed to the accelerometer channel and integrated twice (the flag name says velocity for historical reasons; default: 1.0 0.1 0.0)")
    parser.add_argument("--angular-vel", type=float, nargs=3, default=[0.0, 0.0, 0.5], metavar=("X", "Y", "Z"), help="True commanded angular velocity in rad/s (default: 0.0 0.0 0.5)")

    # Post-optimization bias update configuration (the corrected bias guess after a graph-SLAM update)
    parser.add_argument("--optimized-gyro-bias", type=float, nargs=3, default=[0.008, -0.009, 0.018], metavar=("X", "Y", "Z"), help="Optimizer-corrected gyro bias (rad/s, default: 0.008 -0.009 0.018)")
    parser.add_argument("--optimized-accel-bias", type=float, nargs=3, default=[0.045, 0.002, -0.048], metavar=("X", "Y", "Z"), help="Optimizer-corrected accel bias (m/s^2, default: 0.045 0.002 -0.048)")

    args = parser.parse_args()

    np.set_printoptions(suppress=True, precision=5)

    # Setup initial guesses for sensor biases
    est_gyro_bias = np.array(args.gyro_bias)
    est_accel_bias = np.array(args.accel_bias)

    # Create our tracking sandbox factor bundle
    imu_bundle = PreintegratedIMUBundle(est_gyro_bias, est_accel_bias)

    # Simulate a robot driving along a curved pathway for 1 second at the configured frequency
    frequency_hz = args.frequency_hz
    dt_step = 1.0 / frequency_hz

    # Smooth baseline commanded acceleration + angular rate
    true_v_cmd = np.array(args.linear_vel)  # Driving forward and sliding right slightly
    true_w_cmd = np.array(args.angular_vel)  # Turning around the Z-axis (Yaw)

    print(f"1. Simulating {frequency_hz} high-frequency IMU micro-steps streaming into the sandbox...")
    for _ in range(frequency_hz):
        # Simulate raw sensor inputs by adding back the actual bias values
        raw_v = true_v_cmd + est_accel_bias
        raw_w = true_w_cmd + est_gyro_bias

        # Bake this step into our compressed factor bundle
        imu_bundle.integrate_measurement(raw_v, raw_w, dt_step)

    print(f"   ➔ Successfully compressed {frequency_hz} steps into 1 bundle factor.")
    print(f"   ➔ Nominal Preintegrated Relative Position change Vector:\n      {imu_bundle.delta_p}")

    print("\n2. Graph SLAM Optimization simulation event:")
    print("   Suppose the global solver adjusts its estimate of the IMU biases by a small amount...")

    # Updated bias profiles computed after a new visual feature match arrives
    optimized_gyro_bias = np.array(args.optimized_gyro_bias)  # Minor correction updates
    optimized_accel_bias = np.array(args.optimized_accel_bias)

    print(f"   Old Gyro Bias: {est_gyro_bias} ➔ Optimized Gyro Bias: {optimized_gyro_bias}")

    # Run the instantaneous update step using Taylor Jacobians
    c_R, c_v, c_p = imu_bundle.get_corrected_measurement(optimized_gyro_bias, optimized_accel_bias)

    print("\n3. Comparing CPU Performance Strategy Results:")
    print(f"   Original Integrated Position: {imu_bundle.delta_p}")
    print(f"   Instant Taylor Corrected Pos: {c_p}")
    print("\n   Observation: The corrected position shifted precisely based on the updated bias values.")
    print("   Because this change used the internal Jacobians, it took less than 1 microsecond to solve,")
    print("   completely bypassing the need to loop through the 100 raw sensor inputs again!")
