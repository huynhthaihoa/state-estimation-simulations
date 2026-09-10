import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from utils import (
    pair_key,
    look_at_rotation,
    forward_facing_rotation,
    rotation_matrix_to_quaternion_xyzw,
    umeyama_alignment,
    landmark_errors,
    unscented_weights,
    unscented_sigma_offsets,
    qr_insert_row,
    symbolic_eliminate,
    bayes_tree_affected_path,
    measure_performance,
)


def test_pair_key_order_independent():
    assert pair_key(3, 7) == pair_key(7, 3)
    assert pair_key(3, 7) == (3, 7)
    lo, hi = pair_key(9, 2)
    assert lo <= hi


def test_look_at_rotation_is_orthonormal_and_points_at_target():
    cam_pos = np.array([1.0, 2.0, 3.0])
    target = np.array([5.0, -1.0, 7.0])
    R = look_at_rotation(cam_pos, target)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)
    expected_forward = (target - cam_pos) / np.linalg.norm(target - cam_pos)
    assert np.allclose(R[:, 2], expected_forward, atol=1e-10)


def test_forward_facing_rotation_matches_look_at_rotation():
    cam_pos = np.array([0.0, 0.0, 0.0])
    target = np.array([3.0, 4.0, 0.0])
    R_look_at = look_at_rotation(cam_pos, target)
    R_forward = forward_facing_rotation(target - cam_pos)
    assert np.allclose(R_look_at, R_forward, atol=1e-12)


@pytest.mark.parametrize("euler", [
    (0.0, 0.0, 0.0),
    (0.3, -0.5, 1.2),             # generic case, trace > 0 branch
    (np.pi - 0.01, 0.1, 0.05),    # forces the R[0,0]-dominant branch
    (0.1, np.pi - 0.01, -0.2),    # forces the R[1,1]-dominant branch
    (0.1, 0.2, np.pi - 0.01),     # forces the R[2,2]-dominant branch
])
def test_rotation_matrix_to_quaternion_matches_scipy(euler):
    R = Rotation.from_euler("xyz", euler).as_matrix()
    quat = rotation_matrix_to_quaternion_xyzw(R)
    assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-10)
    expected = Rotation.from_matrix(R).as_quat()  # scipy is already [x,y,z,w]
    # quaternions represent the same rotation up to an overall sign
    assert np.allclose(quat, expected, atol=1e-6) or np.allclose(quat, -expected, atol=1e-6)


def test_umeyama_alignment_recovers_known_similarity_transform():
    rng = np.random.default_rng(0)
    P_est = rng.normal(size=(6, 3))
    s_true = 2.0
    R_true = Rotation.from_euler("xyz", [0.3, -0.2, 0.7]).as_matrix()
    t_true = np.array([1.0, 2.0, 3.0])
    P_true = s_true * (R_true @ P_est.T).T + t_true

    s, R, t = umeyama_alignment(P_est, P_true)
    assert np.isclose(s, s_true, atol=1e-8)
    assert np.allclose(R, R_true, atol=1e-8)
    assert np.allclose(t, t_true, atol=1e-8)


def test_umeyama_alignment_reflection_guard_returns_valid_rotation():
    # A genuinely mirrored point set (improper relationship, det=-1): no proper
    # rotation fits it exactly, so the raw SVD result U@Vt would be improper.
    # umeyama_alignment's S-correction must still return a valid rotation
    # (det=+1) -- exact recovery is impossible here by construction, validity
    # of the returned rotation is what's being guarded.
    P_est = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    P_true = P_est.copy()
    P_true[:, 2] *= -1.0  # mirror through the z=0 plane

    s, R, t = umeyama_alignment(P_est, P_true)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-8)


def test_landmark_errors():
    P_true = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    P_est = np.array([[3.0, 4.0, 0.0], [1.0, 1.0, 1.0]])
    assert np.allclose(landmark_errors(P_true, P_est), [5.0, 0.0])


def test_unscented_weights_sum_to_one_and_match_canonical_case():
    n, alpha, beta, kappa = 3, 1.0, 2.0, 0.0
    lambda_, w_m, w_c = unscented_weights(n, alpha, beta, kappa)
    assert np.isclose(np.sum(w_m), 1.0, atol=1e-12)
    assert np.isclose(w_c[0] - w_m[0], 1.0 - alpha ** 2 + beta, atol=1e-12)
    assert np.isclose(lambda_, 0.0, atol=1e-12)  # alpha=1,kappa=0 => lambda = n - n = 0
    assert np.allclose(w_m[1:], 1.0 / (2 * n), atol=1e-12)


def test_unscented_sigma_offsets_reconstruct_covariance():
    n = 3
    rng = np.random.default_rng(1)
    A = rng.normal(size=(n, n))
    P = A @ A.T + np.eye(n) * 0.1  # random SPD covariance
    lambda_, w_m, w_c = unscented_weights(n, alpha=1e-3, beta=2.0, kappa=0.0)
    offsets = unscented_sigma_offsets(P, lambda_)

    assert offsets.shape == (2 * n + 1, n)
    assert np.allclose(offsets[0], 0.0, atol=1e-12)
    assert np.allclose(offsets[1:n + 1], -offsets[n + 1:2 * n + 1], atol=1e-10)

    sigma_points = offsets  # mean is implicitly zero here
    reconstructed_mean = np.sum(w_m[:, None] * sigma_points, axis=0)
    reconstructed_cov = sum(
        w_c[i] * np.outer(sigma_points[i] - reconstructed_mean, sigma_points[i] - reconstructed_mean)
        for i in range(2 * n + 1)
    )
    assert np.allclose(reconstructed_mean, np.zeros(n), atol=1e-8)
    assert np.allclose(reconstructed_cov, P, atol=1e-6)


def test_qr_insert_row_matches_full_qr_refactorization():
    rng = np.random.default_rng(2)
    m = 5
    A0 = rng.normal(size=(m, m))
    b0 = rng.normal(size=m)
    Q0, R0 = np.linalg.qr(A0)
    d0 = Q0.T @ b0

    new_row = rng.normal(size=m)
    new_b = rng.normal()

    R, d = R0.copy(), d0.copy()
    qr_insert_row(R, d, new_row, new_b)

    A_full = np.vstack([A0, new_row])
    b_full = np.concatenate([b0, [new_b]])

    x_incremental = np.linalg.solve(R, d)
    x_full = np.linalg.lstsq(A_full, b_full, rcond=None)[0]
    assert np.allclose(x_incremental, x_full, atol=1e-8)
    assert np.allclose(R, np.triu(R), atol=1e-10)  # still upper-triangular


def test_symbolic_eliminate_and_affected_path_on_hand_verified_ring():
    # 4-node ring: 0-1,1-2,2-3,3-0 -- hand-derived and independently confirmed
    # earlier in this project (bayes_tree_construction.py's own verification).
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    order = [0, 1, 2, 3]
    separator, parent = symbolic_eliminate(4, edges, order)
    assert parent == {0: 1, 1: 2, 2: 3, 3: None}
    assert separator[0] == [1, 3]
    assert separator[1] == [2, 3]
    assert separator[2] == [3]
    assert separator[3] == []

    affected_local = bayes_tree_affected_path(parent, (2, 3))
    affected_global = bayes_tree_affected_path(parent, (0, 3))
    assert affected_local == {2, 3}
    assert affected_global == {0, 1, 2, 3}  # loop-closure-style edge invalidates the whole chain


def test_symbolic_eliminate_star_graph_root_has_empty_separator():
    # star: center 0 connected to leaves 1,2,3; eliminate leaves first, center last
    edges = [(0, 1), (0, 2), (0, 3)]
    order = [1, 2, 3, 0]
    separator, parent = symbolic_eliminate(4, edges, order)
    assert parent == {1: 0, 2: 0, 3: 0, 0: None}
    assert separator[1] == [0] and separator[2] == [0] and separator[3] == [0]
    assert separator[0] == []


def test_bayes_tree_affected_path_disjoint_subtrees_union_correctly():
    # two independent chains sharing only the root: 0->2 (root), 1->2 (root)
    parent = {0: 2, 1: 2, 2: None}
    assert bayes_tree_affected_path(parent, [0]) == {0, 2}
    assert bayes_tree_affected_path(parent, [1]) == {1, 2}
    assert bayes_tree_affected_path(parent, [0, 1]) == {0, 1, 2}
    assert bayes_tree_affected_path(parent, [2]) == {2}  # touching the root only affects itself


def test_measure_performance_matches_direct_call_and_reports_positive_metrics():
    def fn(x):
        return x * 2

    result, avg_time, avg_mem = measure_performance(fn, 21, n_steps=1)
    assert result == 42
    assert avg_time > 0
    assert avg_mem >= 0
