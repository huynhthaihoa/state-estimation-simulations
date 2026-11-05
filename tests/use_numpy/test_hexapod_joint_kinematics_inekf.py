import numpy as np
import pytest

from state_estimation_simulations.use_numpy import hexapod_joint_kinematics_inekf as m
from state_estimation_simulations.use_numpy import hexapod_contact_inekf as c


# (leg in LEG_NAMES order, joint angles, foot in the body frame) from HEBI's own
# robot model: hebi-py 2.16.2, hebi.robot_model.import_from_hrdf("Daisy.hrdf")
# (HebiRobotics/hebi_description, config/daisy/hrdf), get_forward_kinematics("endeffector", q)
HEBI_REFERENCE = [
    (0, [1.721316619099806, -0.38403808930179695, 2.2531371815723604],
     [0.2440207204559278, 0.17089095283763786, 0.3789385941724836]),
    (1, [1.749135098575132, -1.9186415842058906, -0.209098100419983],
     [0.142174828812206, 0.27663479895617293, 0.727512318469639]),
    (2, [1.1466530692019, -2.263501901933922, -1.8855323730135107],
     [-0.3189010049748057, 0.11435896188297762, 0.4701304047878728]),
    (3, [0.4319136724091748, -2.2632222089662837, -2.421978978318367],
     [0.32435786377476683, -0.06689789618250437, -0.09867288253211898]),
    (4, [-0.5881886967587158, 1.9730482670885108, -2.092470856528834],
     [0.2312756148179341, 0.046631578275675906, 0.13981743255607354]),
    (5, [1.3627612497952022, -0.3181710657077921, -1.4310484722006673],
     [-0.0058747687070353355, -0.5293806833603611, 0.2899469714600463]),
]


@pytest.mark.parametrize("leg, q, foot", HEBI_REFERENCE)
def test_fk_matches_hebi_model(leg, q, foot):
    assert np.allclose(m.leg_fk(np.array(q), leg), foot, atol=1e-6)


@pytest.mark.parametrize("leg", range(6))
def test_jacobian_matches_finite_difference(leg):
    q = np.random.default_rng(leg).uniform(-2.0, 2.0, 3)
    h = 1e-7
    J_fd = np.column_stack([(m.leg_fk(q + e, leg) - m.leg_fk(q - e, leg)) / (2 * h) for e in np.eye(3) * h])
    assert np.allclose(m.leg_jacobian(q, leg), J_fd, atol=1e-7)


@pytest.mark.parametrize("leg", range(6))
def test_stance_ik_reaches_the_nominal_foot_knee_up(leg):
    q = m.stance_ik(leg)
    assert np.allclose(m.leg_fk(q, leg), m.stance_foot(leg), atol=1e-6)
    origins, _, _ = m.leg_frames(q, leg)
    assert origins[2, 2] > 0.0      # knee above the body frame's origin


def test_left_and_right_stances_mirror():
    for left, right in ((0, 3), (1, 4), (2, 5)):
        assert np.allclose(m.stance_ik(right), -m.stance_ik(left), atol=1e-8)


def test_simulate_feet_default_offsets_unchanged():
    cfg = c.Config(n_steps=60)
    rng = np.random.default_rng(0)
    R, v, p, yaw, _, _ = c.simulate_body(cfg, rng)
    noise = rng.normal(0.0, cfg.contact_noise_std, (cfg.n_steps + 1, 6, 3))
    s0, d0 = c.simulate_feet(cfg, R, p, yaw, "tripod", noise)
    s1, d1 = c.simulate_feet(cfg, R, p, yaw, "tripod", noise, foot_offsets=c.NOMINAL_FOOT_OFFSETS)
    assert np.array_equal(s0, s1) and np.array_equal(d0, d1, equal_nan=True)


def short_cfg():
    return m.daisy_config(c.Config(n_steps=90))


def test_offsets_change_no_other_draw():
    cfg = short_cfg()
    t0, q0, b0 = m.simulate_trial(cfg, ["tripod"], np.random.default_rng(3), 1e-3, 0.0)
    t1, q1, b1 = m.simulate_trial(cfg, ["tripod"], np.random.default_rng(3), 1e-3, np.radians(1.0))
    assert np.array_equal(t0.R, t1.R) and np.array_equal(t0.xi0, t1.xi0)
    assert np.allclose(b0, 0.0) and not np.allclose(b1, 0.0)
    stance = t0.feet["tripod"][0]
    k, leg = np.argwhere(stance)[5]
    assert np.allclose(q1["tripod"][k, leg] - q0["tripod"][k, leg], b1[leg], atol=1e-12)


def test_offset_measurement_jacobian():
    """d(residual)/d(b_hat) of the offset filter equals its H on that leg's offsets."""
    cfg = short_cfg()
    trial, q_meas, _ = m.simulate_trial(cfg, ["tripod"], np.random.default_rng(1), 1e-3, np.radians(1.0))
    stance, y = trial.feet["tripod"]
    f = m.make_filter("joint_offsets", trial, cfg, 1e-3, np.radians(1.0), None)
    f.q_k = q_meas["tripod"][0]
    f.step(stance[0], y[0])
    j, leg = 0, f.legs[0]
    f.b_q[leg] = [0.01, -0.02, 0.015]
    r0, H = f.measurement(j, y[0, leg])
    h = 1e-7
    for i in range(3):
        f.b_q[leg, i] += h
        r1, _ = f.measurement(j, y[0, leg])
        f.b_q[leg, i] -= h
        assert np.allclose((r1 - r0) / h, H[:, 9 + 3 * leg + i], atol=1e-5)


def test_zero_offset_prior_matches_joint_filter():
    """With sigma_b = 0 the offset states never move, so the estimate equals the joint filter's."""
    cfg = short_cfg()
    trial, q_meas, b = m.simulate_trial(cfg, ["tripod"], np.random.default_rng(2), 1e-3, 0.0)
    a = m.run_filter("joint", trial, q_meas, "tripod", cfg, 1e-3, 0.0, None, b)
    o = m.run_filter("joint_offsets", trial, q_meas, "tripod", cfg, 1e-3, 0.0, None, b)
    assert np.allclose(a["vel_err"], o["vel_err"], atol=1e-10)
    assert np.allclose(a["nees_vel"], o["nees_vel"], atol=1e-8)
