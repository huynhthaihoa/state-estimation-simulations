import numpy as np
import pytest


@pytest.fixture
def robot_imu_simulation(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "robot_imu_simulation")


def _default_omega_info():
    # Same "precision GPS, imperfect gyro" weighting the CLI default uses.
    unbalanced_cov = np.zeros((6, 6))
    np.fill_diagonal(unbalanced_cov, [0.001, 0.001, 0.001, 1000.0, 1000.0, 1000.0])
    return np.linalg.inv(unbalanced_cov)


def test_run_simulation_is_reproducible_given_a_fixed_seed(robot_imu_simulation):
    omega_info = _default_omega_info()

    def run():
        rng = np.random.default_rng(42)
        return robot_imu_simulation.run_simulation(
            dt_imu=0.01, total_seconds=2, snapshots_per_second=4,
            gn_tol=1e-6, gn_max_iters=10, max_linear_vel=1.0, max_angular_vel=0.5,
            omega_info=omega_info, rng=rng,
        )

    T_true_1, T_est_1, pre_1, post_1 = run()
    T_true_2, T_est_2, pre_2, post_2 = run()

    assert np.array_equal(T_true_1, T_true_2)
    assert np.array_equal(T_est_1, T_est_2)
    assert pre_1 == pre_2
    assert post_1 == post_2


def test_run_simulation_correction_reduces_error_each_second(robot_imu_simulation):
    rng = np.random.default_rng(0)
    omega_info = _default_omega_info()
    _, _, pre_correction_errors, post_correction_errors = robot_imu_simulation.run_simulation(
        dt_imu=0.01, total_seconds=3, snapshots_per_second=4,
        gn_tol=1e-6, gn_max_iters=10, max_linear_vel=1.0, max_angular_vel=0.5,
        omega_info=omega_info, rng=rng,
    )

    assert len(pre_correction_errors) == 3
    assert len(post_correction_errors) == 3
    for pre, post in zip(pre_correction_errors, post_correction_errors):
        assert post < pre  # the Gauss-Newton correction should always shrink the position error

    # Final error should be small: high-precision "GPS" correction every second
    # bounds how far the noisy IMU-only propagation can drift beforehand.
    assert post_correction_errors[-1] < 0.05


def test_run_simulation_different_seeds_give_different_trajectories(robot_imu_simulation):
    omega_info = _default_omega_info()
    kwargs = dict(dt_imu=0.01, total_seconds=1, snapshots_per_second=4,
                  gn_tol=1e-6, gn_max_iters=10, max_linear_vel=1.0, max_angular_vel=0.5,
                  omega_info=omega_info)

    T_true_a, _, _, _ = robot_imu_simulation.run_simulation(rng=np.random.default_rng(1), **kwargs)
    T_true_b, _, _, _ = robot_imu_simulation.run_simulation(rng=np.random.default_rng(2), **kwargs)

    assert not np.allclose(T_true_a, T_true_b)
