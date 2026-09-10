import numpy as np
import pytest


@pytest.fixture
def imu_preintegration(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "imu_preintegration")


@pytest.fixture
def lie_utils(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "lie_utils")


def test_straight_line_motion_matches_closed_form(imu_preintegration):
    # Zero angular rate, constant "linear_vel" input: since integrate_measurement
    # treats raw_linear_vel as a body-frame acceleration-like quantity (delta_v
    # accumulates it, delta_p double-integrates it via semi-implicit Euler with
    # R_prev == I throughout), the discrete recursion telescopes exactly to the
    # continuous constant-acceleration formulas: delta_v(T) = v*T, delta_p(T) = 0.5*v*T^2.
    bundle = imu_preintegration.PreintegratedIMUBundle(
        initial_bias_gyro=np.zeros(3), initial_bias_accel=np.zeros(3))
    v = np.array([0.5, -0.3, 0.2])
    w = np.zeros(3)
    dt = 0.01
    n_steps = 100
    for _ in range(n_steps):
        bundle.integrate_measurement(v, w, dt)

    T = n_steps * dt
    assert np.allclose(bundle.delta_v, v * T, atol=1e-10)
    assert np.allclose(bundle.delta_p, 0.5 * v * T ** 2, atol=1e-10)
    assert np.allclose(bundle.delta_R, np.eye(3), atol=1e-10)


def test_constant_axis_rotation_matches_so3_exp_closed_form(imu_preintegration, lie_utils):
    # Rotating about a fixed axis each micro-step commutes/composes additively in
    # angle, so after n steps of dt, delta_R == so3_exp(w * n*dt) exactly (up to
    # floating point), independent of the per-step discretization.
    bundle = imu_preintegration.PreintegratedIMUBundle(
        initial_bias_gyro=np.zeros(3), initial_bias_accel=np.zeros(3))
    w = np.array([0.0, 0.0, 0.7])
    v = np.zeros(3)
    dt = 0.01
    n_steps = 100
    for _ in range(n_steps):
        bundle.integrate_measurement(v, w, dt)

    T = n_steps * dt
    expected_R = lie_utils.so3_exp(w * T)
    assert np.allclose(bundle.delta_R, expected_R, atol=1e-8)
    # sanity: still a valid rotation
    assert np.allclose(bundle.delta_R.T @ bundle.delta_R, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(bundle.delta_R), 1.0, atol=1e-10)


def test_get_corrected_measurement_with_same_bias_is_identity(imu_preintegration):
    # Correcting with the exact bias the bundle was integrated at (zero bias
    # delta) must reproduce the stored delta_R/delta_v/delta_p exactly.
    b_g = np.array([0.01, -0.02, 0.03])
    b_a = np.array([0.05, 0.01, -0.04])
    bundle = imu_preintegration.PreintegratedIMUBundle(b_g, b_a)

    rng = np.random.default_rng(0)
    dt = 0.01
    for _ in range(50):
        raw_v = rng.normal(size=3) * 0.5 + b_a
        raw_w = rng.normal(size=3) * 0.3 + b_g
        bundle.integrate_measurement(raw_v, raw_w, dt)

    c_R, c_v, c_p = bundle.get_corrected_measurement(b_g.copy(), b_a.copy())
    assert np.allclose(c_R, bundle.delta_R, atol=1e-12)
    assert np.allclose(c_v, bundle.delta_v, atol=1e-12)
    assert np.allclose(c_p, bundle.delta_p, atol=1e-12)


def _run_bundle(imu_preintegration, b_g, b_a, raw_vs, raw_ws, dt):
    bundle = imu_preintegration.PreintegratedIMUBundle(b_g, b_a)
    for raw_v, raw_w in zip(raw_vs, raw_ws):
        bundle.integrate_measurement(raw_v, raw_w, dt)
    return bundle


@pytest.mark.parametrize("bias_kind,axis", [
    ("gyro", 0), ("gyro", 1), ("gyro", 2),
    ("accel", 0), ("accel", 1), ("accel", 2),
])
def test_bias_correction_jacobian_matches_finite_difference(imu_preintegration, bias_kind, axis):
    # get_corrected_measurement uses a first-order Taylor expansion driven by
    # J_R_bg/J_v_bg/J_v_ba/J_p_bg/J_p_ba. Re-running the full integration loop
    # at a slightly perturbed initial bias gives the "ground truth" perturbed
    # bundle; the Taylor-corrected prediction from the *unperturbed* bundle
    # should match it to first order.
    rng = np.random.default_rng(1)
    b_g0 = np.array([0.01, -0.02, 0.005])
    b_a0 = np.array([0.05, -0.03, 0.02])
    dt = 0.01
    n_steps = 50
    raw_vs = [rng.normal(size=3) * 0.5 + b_a0 for _ in range(n_steps)]
    raw_ws = [rng.normal(size=3) * 0.3 + b_g0 for _ in range(n_steps)]

    bundle = _run_bundle(imu_preintegration, b_g0, b_a0, raw_vs, raw_ws, dt)

    eps = 1e-6
    delta = np.zeros(3)
    delta[axis] = eps
    if bias_kind == "gyro":
        new_b_g, new_b_a = b_g0 + delta, b_a0
    else:
        new_b_g, new_b_a = b_g0, b_a0 + delta

    perturbed_bundle = _run_bundle(imu_preintegration, new_b_g, new_b_a, raw_vs, raw_ws, dt)
    pred_R, pred_v, pred_p = bundle.get_corrected_measurement(new_b_g, new_b_a)

    assert np.allclose(pred_R, perturbed_bundle.delta_R, atol=1e-4)
    assert np.allclose(pred_v, perturbed_bundle.delta_v, atol=1e-4)
    assert np.allclose(pred_p, perturbed_bundle.delta_p, atol=1e-4)


def test_no_motion_leaves_bundle_at_identity(imu_preintegration):
    bundle = imu_preintegration.PreintegratedIMUBundle(np.zeros(3), np.zeros(3))
    for _ in range(20):
        bundle.integrate_measurement(np.zeros(3), np.zeros(3), 0.01)
    assert np.allclose(bundle.delta_R, np.eye(3), atol=1e-12)
    assert np.allclose(bundle.delta_v, np.zeros(3), atol=1e-12)
    assert np.allclose(bundle.delta_p, np.zeros(3), atol=1e-12)
