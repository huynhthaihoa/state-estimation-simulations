import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture
def imu_integration_comparison(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "imu_integration_comparison")


@pytest.mark.parametrize("rpy", [
    (0.0, 0.0, 0.0),
    (np.pi / 2, 0.0, 0.0),
    (0.0, np.pi / 2, 0.0),
    (0.0, 0.0, np.pi / 2),
    (0.3, -0.5, 1.2),
])
def test_euler_to_R_matches_scipy_extrinsic_xyz(imu_integration_comparison, rpy):
    # euler_to_R composes Rz @ Ry @ Rx, which is scipy's extrinsic ("xyz",
    # lowercase) convention -- confirmed empirically, not assumed.
    R = imu_integration_comparison.euler_to_R(rpy)
    expected = Rotation.from_euler("xyz", rpy).as_matrix()
    assert np.allclose(R, expected, atol=1e-10)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_euler_to_R_single_axis_roll_matches_known_rotation(imu_integration_comparison):
    R = imu_integration_comparison.euler_to_R((np.pi / 2, 0.0, 0.0))
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    assert np.allclose(R, expected, atol=1e-10)


def test_run_simulation_with_zero_noise_exp_map_error_is_near_exact(imu_integration_comparison):
    # With zero noise, omega_meas == omega_gt and v_meas == v_gt exactly, so the
    # exp-map estimator's own state recursion is *identical* to the exp-map
    # ground-truth recursion -- their errors should be ~0 (float precision), not
    # merely "small". The naive estimator, however, composes Euler angles by
    # flat-vector-space addition while its own ground truth composes rotation
    # matrices properly -- these two formulas differ even with zero noise, so
    # rot_err_naive is NOT expected to vanish; it reflects the approximation
    # error the whole script exists to demonstrate.
    t, rot_err_naive, rot_err_exp, pos_err_naive, pos_err_exp = \
        imu_integration_comparison.run_simulation(
            duration=5.0, dt=0.01, gyro_noise_std=0.0, vel_noise_std=0.0, seed=0)

    assert np.max(rot_err_exp) < 1e-3   # degrees
    assert np.max(pos_err_exp) < 1e-6   # meters, exactly 0 up to float noise
    # The naive vector-space approximation error should dominate/exceed the
    # near-zero exp-map error by a wide margin, even without any noise driving it.
    assert rot_err_naive[-1] > rot_err_exp[-1] * 100


def test_run_simulation_with_realistic_noise_exp_map_beats_naive(imu_integration_comparison):
    t, rot_err_naive, rot_err_exp, pos_err_naive, pos_err_exp = \
        imu_integration_comparison.run_simulation(
            duration=5.0, dt=0.01, gyro_noise_std=0.02, vel_noise_std=0.05, seed=0)

    rms_rot_naive = np.sqrt(np.mean(rot_err_naive ** 2))
    rms_rot_exp = np.sqrt(np.mean(rot_err_exp ** 2))
    rms_pos_naive = np.sqrt(np.mean(pos_err_naive ** 2))
    rms_pos_exp = np.sqrt(np.mean(pos_err_exp ** 2))

    # The whole point of the comparison: proper SO(3)-manifold integration
    # should track substantially better than naive Euler-angle vector addition.
    assert rot_err_exp[-1] <= rot_err_naive[-1]
    assert rms_rot_exp <= rms_rot_naive
    assert pos_err_exp[-1] <= pos_err_naive[-1]
    assert rms_pos_exp <= rms_pos_naive


def test_run_simulation_output_shapes_and_monotonic_time(imu_integration_comparison):
    duration, dt = 2.0, 0.01
    t, rot_err_naive, rot_err_exp, pos_err_naive, pos_err_exp = \
        imu_integration_comparison.run_simulation(
            duration=duration, dt=dt, gyro_noise_std=0.01, vel_noise_std=0.02, seed=3)

    n_steps = int(duration / dt)
    for arr in (t, rot_err_naive, rot_err_exp, pos_err_naive, pos_err_exp):
        assert arr.shape == (n_steps,)
    assert np.allclose(t, np.arange(n_steps) * dt, atol=1e-12)
    assert np.all(rot_err_naive >= 0.0)
    assert np.all(rot_err_exp >= 0.0)
    assert np.all(pos_err_naive >= 0.0)
    assert np.all(pos_err_exp >= 0.0)


def test_run_simulation_is_deterministic_given_seed(imu_integration_comparison):
    result_a = imu_integration_comparison.run_simulation(
        duration=1.0, dt=0.01, gyro_noise_std=0.02, vel_noise_std=0.05, seed=42)
    result_b = imu_integration_comparison.run_simulation(
        duration=1.0, dt=0.01, gyro_noise_std=0.02, vel_noise_std=0.05, seed=42)
    for arr_a, arr_b in zip(result_a, result_b):
        assert np.array_equal(arr_a, arr_b)
