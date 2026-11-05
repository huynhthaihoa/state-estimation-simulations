import numpy as np
from dataclasses import replace

from state_estimation_simulations.use_numpy import magnetic_inchworm_odometry_ekf as m
from state_estimation_simulations.utils import averaged_nees_bounds


def test_chain_is_sized_to_the_robot_and_reaches_the_step():
    assert np.isclose(2 * m.LEG + m.BODY, 0.49)
    for d in (m.D_MIN, m.D_MIN + m.STEP_LEN):
        assert np.allclose(m.fk(m.ik(d)), [d, 0.0, 0.0], atol=1e-12)


def test_se2_inverse_and_compose():
    u = np.array([0.3, -0.1, 0.4])
    assert np.allclose(m.se2_compose(u, m.se2_inverse(u)), 0.0, atol=1e-12)


def test_slips_only_on_the_wall_one_sided_and_flagged():
    cfg = replace(m.Config(), slip_rate=0.5)
    trial = m.simulate_trial(cfg, np.random.default_rng(0))
    assert np.all(trial.slip[:cfg.n_floor] == 0.0)
    assert np.all(trial.slip >= 0.0) and np.any(trial.slip > 0.0)
    assert np.array_equal(trial.flag, trial.slip > 0.0)       # perfect flag by default



def test_random_sign_control_slides_both_ways_and_keeps_default_draws():
    cfg = replace(m.Config(), slip_rate=0.5)
    one = m.simulate_trial(cfg, np.random.default_rng(4))
    rnd = m.simulate_trial(replace(cfg, slip_sign="random"), np.random.default_rng(4))
    assert np.any(rnd.slip < 0.0) and np.any(rnd.slip > 0.0)
    assert np.allclose(np.abs(rnd.slip), one.slip)            # same sizes, signs drawn last
    assert np.array_equal(rnd.flag, one.flag)

def test_servo_odometry_misses_exactly_the_slips():
    """With exact servo readings, the servo-only odometry's sideways error at
    the end equals the sum of the slides it cannot see."""
    cfg = replace(m.Config(), slip_rate=0.5)
    trial = m.simulate_trial(cfg, np.random.default_rng(1))
    plan = m.step_plan(cfg)
    trial.q_meas = np.array([m.ik(m.D_MIN + m.STEP_LEN) if ext else m.ik(m.D_MIN) for ext, _ in plan])
    xs, _ = m.run_filter("servo_only", trial, cfg)
    assert np.isclose(trial.anchor[-1, 1] - xs[-1, 1], trial.slip.sum(), atol=1e-12)


def test_servo_odometry_is_consistent_without_slip():
    cfg = replace(m.Config(), slip_std=0.0)
    rng = np.random.default_rng(2)
    n_tr = 60
    vals = []
    for _ in range(n_tr):
        trial = m.simulate_trial(cfg, rng)
        xs, Ps = m.run_filter("servo_only", trial, cfg)
        e = trial.anchor[-1] - xs[-1]
        vals.append(m.nees(e, Ps[-1]))
    lo, hi = averaged_nees_bounds(3, n_tr)
    assert lo < np.mean(vals) < hi


def test_external_fix_bounds_the_sideways_error():
    cfg = replace(m.Config(), slip_rate=1.0)
    rng = np.random.default_rng(3)
    end = []
    for _ in range(20):
        trial = m.simulate_trial(cfg, rng)
        xs, _ = m.run_filter("ext_fix", trial, cfg)
        end.append(trial.anchor[-1, 1] - xs[-1, 1])
    assert np.sqrt(np.mean(np.square(end))) < 0.005
