import numpy as np
import pytest
manif = pytest.importorskip("manifpy")


@pytest.fixture
def robot_imu_simulation(import_module, use_manif_dir):
    return import_module(use_manif_dir, "robot_imu_simulation")


def _default_pos_info(pos_noise_std=0.05):
    return np.eye(3) / pos_noise_std ** 2


def test_position_observation_jacobian_matches_finite_difference(robot_imu_simulation):
    rng = np.random.default_rng(0)
    manif_mod = robot_imu_simulation.manif
    T_est = manif_mod.SE3.Identity().rplus(manif_mod.SE3Tangent(rng.normal(size=6) * 0.3))

    J_analytic = robot_imu_simulation.position_observation_jacobian(T_est)

    eps = 1e-6
    J_numeric = np.zeros((3, 6))
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        T_pert = T_est.rplus(manif_mod.SE3Tangent(d))
        J_numeric[:, i] = (T_pert.translation() - T_est.translation()) / eps

    assert np.allclose(J_analytic, J_numeric, atol=1e-6)


def test_position_observation_jacobian_angular_block_is_exactly_zero(robot_imu_simulation):
    # The property this whole fix depends on: a position-only measurement's
    # Jacobian has a structurally zero angular block, so it can never correct
    # orientation, however it's weighted -- not just weighted low.
    rng = np.random.default_rng(1)
    manif_mod = robot_imu_simulation.manif
    for _ in range(5):
        T_est = manif_mod.SE3.Identity().rplus(manif_mod.SE3Tangent(rng.normal(size=6)))
        J = robot_imu_simulation.position_observation_jacobian(T_est)
        assert np.array_equal(J[:, 3:6], np.zeros((3, 3)))


def test_run_simulation_is_reproducible_given_a_fixed_seed(robot_imu_simulation):
    pos_info = _default_pos_info()

    def run():
        rng = np.random.default_rng(42)
        return robot_imu_simulation.run_simulation(
            dt_imu=0.01, total_seconds=2, snapshots_per_second=4,
            gn_tol=1e-6, gn_max_iters=10, max_linear_vel=1.0, max_angular_vel=0.5,
            pos_noise_std=0.05, pos_info=pos_info, rng=rng,
        )

    T_true_1, T_est_1, pre_1, post_1 = run()
    T_true_2, T_est_2, pre_2, post_2 = run()

    # T_true/T_est are manif.SE3 objects (no array_equal support of their own),
    # so bit-exact reproducibility is checked via their underlying coefficient
    # vectors ([tx,ty,tz,qx,qy,qz,qw]) instead of raw (4,4) matrices.
    assert np.array_equal(T_true_1.coeffs(), T_true_2.coeffs())
    assert np.array_equal(T_est_1.coeffs(), T_est_2.coeffs())
    assert np.array_equal(T_true_1.translation(), T_true_2.translation())
    assert np.array_equal(T_est_1.translation(), T_est_2.translation())
    assert pre_1 == pre_2
    assert post_1 == post_2


def test_run_simulation_correction_stays_bounded_near_gps_noise_floor(robot_imu_simulation):
    # The Global Position Measurement is now a genuine noisy reading (not T_true
    # itself -- see run_simulation's docstring), so per-tick correction can't be
    # guaranteed to always shrink error (a noisy GPS fix can occasionally pull the
    # estimate slightly further from ground truth than uncorrected IMU drift
    # already was that particular second -- real sensor fusion behavior). What a
    # working fusion *does* guarantee is that error stays bounded near the
    # measurement noise floor over time, instead of growing without bound the way
    # uncorrected IMU-only dead reckoning would (a pure random walk in the noise
    # integrated every micro-step).
    pos_noise_std = 0.05
    pos_info = _default_pos_info(pos_noise_std)
    rng = np.random.default_rng(0)
    _, _, pre_correction_errors, post_correction_errors = robot_imu_simulation.run_simulation(
        dt_imu=0.01, total_seconds=30, snapshots_per_second=1,
        gn_tol=1e-6, gn_max_iters=10, max_linear_vel=1.0, max_angular_vel=0.5,
        pos_noise_std=pos_noise_std, pos_info=pos_info, rng=rng,
    )

    assert len(pre_correction_errors) == 30
    assert len(post_correction_errors) == 30

    post = np.array(post_correction_errors)
    # Bounded near the measurement noise floor (empirically ~1.6x pos_noise_std
    # for this scenario/seed) -- generous margin, not a tight regression pin.
    assert post.mean() < 5 * pos_noise_std

    # Boundedness, not just a small mean: the second half of the run shouldn't be
    # systematically worse than the first half (which unbounded drift would show).
    first_half, second_half = post[:15], post[15:]
    assert second_half.mean() < 3 * first_half.mean()


def test_run_simulation_different_seeds_give_different_trajectories(robot_imu_simulation):
    pos_info = _default_pos_info()
    kwargs = dict(dt_imu=0.01, total_seconds=1, snapshots_per_second=4,
                  gn_tol=1e-6, gn_max_iters=10, max_linear_vel=1.0, max_angular_vel=0.5,
                  pos_noise_std=0.05, pos_info=pos_info)

    T_true_a, _, _, _ = robot_imu_simulation.run_simulation(rng=np.random.default_rng(1), **kwargs)
    T_true_b, _, _, _ = robot_imu_simulation.run_simulation(rng=np.random.default_rng(2), **kwargs)

    assert not np.allclose(T_true_a.translation(), T_true_b.translation())
    assert not np.allclose(T_true_a.coeffs(), T_true_b.coeffs())
