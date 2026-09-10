import pytest


@pytest.fixture
def bayes_tree_construction(import_module, use_numpy_dir):
    return import_module(use_numpy_dir, "bayes_tree_construction")


def test_ring_edges_known_small_case(bayes_tree_construction):
    odom_edges, loop_edge = bayes_tree_construction.ring_edges(5)
    assert odom_edges == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert loop_edge == (4, 0)


def test_ring_edges_minimal_two_node_case(bayes_tree_construction):
    odom_edges, loop_edge = bayes_tree_construction.ring_edges(2)
    assert odom_edges == [(0, 1)]
    assert loop_edge == (1, 0)


def test_layout_tree_straight_chain(bayes_tree_construction):
    # 0's parent is 1, 1's parent is 2 (root) -- a straight 3-node chain.
    children = {1: [0], 2: [1]}
    pos = bayes_tree_construction.layout_tree(children, root=2)

    assert pos.keys() == {0, 1, 2}
    # x = depth from the root
    assert pos[2][0] == 0
    assert pos[1][0] == 1
    assert pos[0][0] == 2
    # the single leaf gets y=0, and every ancestor's y is the mean of a
    # single child's y, so it propagates unchanged up the chain
    assert pos[0][1] == 0
    assert pos[1][1] == pos[0][1]
    assert pos[2][1] == pos[1][1]


def test_layout_tree_branching(bayes_tree_construction):
    # root "r" has children "a" (a leaf-parent with two leaf children a1, a2)
    # and "b" (a leaf) -- DFS order visits a1, a2, then b.
    children = {"r": ["a", "b"], "a": ["a1", "a2"]}
    pos = bayes_tree_construction.layout_tree(children, root="r")

    assert pos.keys() == {"r", "a", "b", "a1", "a2"}

    # leaves get sequential, distinct y-positions in DFS visitation order
    assert pos["a1"][1] == 0
    assert pos["a2"][1] == 1
    assert pos["b"][1] == 2

    # x = depth from the root
    assert pos["r"][0] == 0
    assert pos["a"][0] == 1
    assert pos["b"][0] == 1
    assert pos["a1"][0] == 2
    assert pos["a2"][0] == 2

    # an internal node's y is the mean of its children's y
    assert pos["a"][1] == (pos["a1"][1] + pos["a2"][1]) / 2
    assert pos["r"][1] == (pos["a"][1] + pos["b"][1]) / 2
