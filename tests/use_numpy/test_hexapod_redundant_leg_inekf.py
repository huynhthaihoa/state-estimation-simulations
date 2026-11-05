import numpy as np
import pytest

from state_estimation_simulations.use_numpy import hexapod_redundant_leg_inekf as m
from state_estimation_simulations.use_numpy import hexapod_joint_kinematics_inekf as e1
from state_estimation_simulations.use_numpy.hexapod_contact_inekf import Config

H = m.HANTR


def test_zero_pose_matches_the_papers_morphology():
    """At q = 0 (leg along +x, mount 0 deg): coxa axis radial (roll), trochanter
    axis vertical (yaw), femur and tibia axes horizontal (pitch), coxa and
    trochanter origins coincide, at (l_r, 0, l_z)."""
    origins, axes, _ = H.frames(np.zeros(4), 3)       # R1: mount 0 deg
    assert np.allclose(origins[0], [m.BODY_RADIUS, 0.0, m.COXA_HEIGHT])
    assert np.allclose(origins[0], origins[1])
    assert np.allclose(np.abs(axes[0]), [1, 0, 0]) and np.allclose(np.abs(axes[1]), [0, 0, 1])
    assert np.allclose(np.abs(axes[2]), [0, 1, 0]) and np.allclose(np.abs(axes[3]), [0, 1, 0])


def test_link_lengths():
    origins, _, foot = H.frames(np.array([0.3, -0.2, 0.5, -0.4]), 0)
    assert np.isclose(np.linalg.norm(origins[3] - origins[2]), 0.0816)
    assert np.isclose(np.linalg.norm(foot - origins[3]), 0.2055)
    assert np.isclose(np.linalg.norm(origins[2] - origins[1]), np.hypot(0.0254, 0.0185))


@pytest.mark.parametrize("leg", range(6))
def test_jacobian_matches_finite_difference(leg):
    q = np.random.default_rng(leg).uniform(-1.0, 1.0, 4)
    h = 1e-7
    J_fd = np.column_stack([(H.fk(q + e, leg) - H.fk(q - e, leg)) / (2 * h) for e in np.eye(4) * h])
    assert np.allclose(H.jacobian(q, leg), J_fd, atol=1e-7)


def test_null_direction_moves_no_foot():
    q = H.stance_ik(2)
    n = H.null_direction(q, 2)
    assert np.allclose(H.jacobian(q, 2) @ n, 0.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(n), 1.0)


@pytest.mark.parametrize("leg", range(6))
def test_ik_reaches_the_foot_with_the_tibia_closest_to_gravity(leg):
    """Foot reached; moving along the null space (foot fixed to first order)
    only turns the tibia further from vertical."""
    R = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(0.05), -np.sin(0.05)], [0.0, np.sin(0.05), np.cos(0.05)]])
    target = H.stance_foot(leg) + np.array([0.02, -0.015, 0.01])
    q = H.ik(target, leg, H.stance_ik(leg), R=R)
    assert np.allclose(H.fk(q, leg), target, atol=1e-6)
    o = R.T @ np.array([0.0, 0.0, -1.0])
    n = H.null_direction(q, leg)
    best = H.tibia(q, leg) @ o
    for s in (-1e-3, 1e-3):
        qs = H.ik(H.fk(q, leg), leg, q + s * n, R=R)     # back onto the target, nearby
        qn = q + s * n
        assert H.tibia(qn, leg) @ o <= best + 1e-9
        assert np.allclose(qs, q, atol=1e-6)


def test_redundant_offset_filter_jacobian():
    """d(residual)/d(b_hat) equals H on that leg's four offset columns."""
    cfg = H.config(Config(n_steps=60))
    trial, q_meas, _ = e1.simulate_trial(cfg, ["tripod"], np.random.default_rng(1), 1e-3, np.radians(1.0), H)
    stance, y = trial.feet["tripod"]
    f = e1.make_filter("joint_offsets", trial, cfg, 1e-3, np.radians(1.0), None, H)
    f.q_k = q_meas["tripod"][0]
    f.step(stance[0], y[0])
    leg = f.legs[0]
    f.b_q[leg] = [0.01, -0.02, 0.015, 0.005]
    r0, Hm = f.measurement(0, y[0, leg])
    cols = f.offset_cols(leg)
    assert cols == slice(9 + 4 * leg, 13 + 4 * leg)
    h = 1e-7
    for i in range(4):
        f.b_q[leg, i] += h
        r1, _ = f.measurement(0, y[0, leg])
        f.b_q[leg, i] -= h
        assert np.allclose((r1 - r0) / h, Hm[:, cols][:, i], atol=1e-5)


def test_null_split_sums_to_the_whole_error():
    rng = np.random.default_rng(0)
    e = rng.normal(size=24)
    A = rng.normal(size=(24, 24))
    C = A @ A.T
    s = m.null_split(e, C)
    for leg in range(6):
        sl = slice(4 * leg, 4 * leg + 4)
        assert np.isclose(s[0, leg] ** 2 + s[2, leg], e[sl] @ e[sl])
        assert np.isclose(s[1, leg] ** 2 + s[3, leg], np.trace(C[sl, sl]))


def test_single_joint_foot_motion_at_the_default_foothold():
    """Foot motion per radian of each joint alone (doc §1): coxa and trochanter
    about 138 mm, femur 193 mm, tibia 206 mm (the tibia length)."""
    for leg in range(6):
        J = H.jacobian(H.stance_ik(leg), leg)
        assert np.allclose(np.linalg.norm(J, axis=0) * 1e3, [138.5, 138.0, 193.2, 205.5], atol=0.5)


@pytest.mark.parametrize("leg", range(6))
def test_null_direction_spins_the_leg_about_the_hip_to_foot_line(leg):
    """Doc §1.1: at the default foothold n uses only the coxa and trochanter,
    whose axes cross at one point, and the resulting rotation is about the
    line from that point to the foot."""
    q = H.stance_ik(leg)
    origins, axes, foot = H.frames(q, leg)
    n = H.null_direction(q, leg)
    assert np.allclose(n[2:], 0.0, atol=1e-9)
    assert np.allclose(origins[0], origins[1])
    w = n[0] * axes[0] + n[1] * axes[1]
    assert np.allclose(np.cross(w, foot - origins[0]), 0.0, atol=1e-9)   # IK converges to ~1e-9
