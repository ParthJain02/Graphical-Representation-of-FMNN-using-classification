"""Unit tests for the Fuzzy Min-Max Neural Network (FMNN) implementation."""

from __future__ import annotations

import numpy as np
import pytest

from src.fmnn import FMNN, HyperBox


def test_simpson_fig8_exact() -> None:
    """Reproduce Simpson (1992) Fig 8 step-by-step example exactly.

    Data:
      A1 = (0.2, 0.2), class 1
      A2 = (0.6, 0.6), class 2
      A3 = (0.5, 0.5), class 1
      A4 = (0.4, 0.3), class 2
    theta = 0.3, gamma = 4.0
    Expected:
      Box 1 (class 1): V=(0.2, 0.2), W=(0.45, 0.5)
      Box 2 (class 2): V=(0.45, 0.3), W=(0.6, 0.6)
    """
    Z = np.array([
        [0.2, 0.2],
        [0.6, 0.6],
        [0.5, 0.5],
        [0.4, 0.3],
    ])
    # Use classes 0 and 1 or 1 and 2
    y = np.array([1, 2, 1, 2])
    node_ids = [0, 1, 2, 3]

    fmnn = FMNN(theta=0.3, gamma=4.0)
    fmnn.fit(Z, y, node_ids, shuffle=False)

    assert fmnn.num_boxes() == 2
    b1 = fmnn.boxes[1]
    b2 = fmnn.boxes[2]

    assert b1.cls == 1
    assert b2.cls == 2

    np.testing.assert_allclose(b1.v, np.array([0.2, 0.2]), atol=1e-5)
    np.testing.assert_allclose(b1.w, np.array([0.45, 0.5]), atol=1e-5)

    np.testing.assert_allclose(b2.v, np.array([0.45, 0.3]), atol=1e-5)
    np.testing.assert_allclose(b2.w, np.array([0.6, 0.6]), atol=1e-5)


def test_membership_properties() -> None:
    """Check membership value properties: in [0, 1], 1 inside box, decreasing with distance."""
    fmnn = FMNN(theta=0.3, gamma=4.0)
    box = HyperBox(id=1, v=np.array([0.3, 0.3]), w=np.array([0.7, 0.7]), cls=0)

    # Points inside box
    inside_points = [
        np.array([0.3, 0.3]),
        np.array([0.5, 0.5]),
        np.array([0.7, 0.7]),
        np.array([0.4, 0.6]),
    ]
    for p in inside_points:
        mem = fmnn.membership(p, box)
        assert np.isclose(mem, 1.0), f"Expected 1.0 inside box, got {mem}"

    # Points outside box: decreasing with distance
    p_near = np.array([0.8, 0.5])
    p_far = np.array([0.95, 0.5])
    mem_near = fmnn.membership(p_near, box)
    mem_far = fmnn.membership(p_far, box)

    assert 0.0 <= mem_near <= 1.0
    assert 0.0 <= mem_far <= 1.0
    assert mem_near > mem_far, "Membership should decrease as point moves further away"


def test_membership_matrix_broadcasting_agreement() -> None:
    """Check that vectorized membership_matrix agrees with single-point membership."""
    rng = np.random.RandomState(42)
    Z = rng.rand(30, 8)
    boxes = [
        HyperBox(id=1, v=np.array([0.1] * 8), w=np.array([0.4] * 8), cls=0),
        HyperBox(id=2, v=np.array([0.5] * 8), w=np.array([0.8] * 8), cls=1),
        HyperBox(id=3, v=np.array([0.2] * 8), w=np.array([0.6] * 8), cls=0),
    ]

    fmnn = FMNN(gamma=4.0)
    mat = fmnn.membership_matrix(Z, boxes)

    assert mat.shape == (30, 3)
    for i in range(30):
        for j, b in enumerate(boxes):
            single = fmnn.membership(Z[i], b)
            assert np.isclose(mat[i, j], single, atol=1e-12)


def test_touching_boxes_do_not_overlap() -> None:
    """Verify that touching boxes (e.g. w_j == v_k) do NOT count as overlapping."""
    b1 = HyperBox(id=1, v=np.array([0.0, 0.0]), w=np.array([0.4, 0.5]), cls=0)
    b2 = HyperBox(id=2, v=np.array([0.4, 0.0]), w=np.array([0.8, 0.5]), cls=1)

    assert not FMNN.boxes_overlap(b1, b2), "Touching boxes should not count as overlapping"

    # Test contraction does not trigger on touching boxes
    fmnn = FMNN()
    fmnn.n_dims = 2
    contracted = fmnn._test_and_contract(b1, b2)
    assert not contracted, "Touching boxes should not trigger contraction"


def test_no_cross_class_overlap_after_fit() -> None:
    """Assert there are no overlapping boxes of different classes after fit on 2D data."""
    rng = np.random.RandomState(123)
    Z = rng.rand(80, 2)
    y = (Z[:, 0] + Z[:, 1] > 1.0).astype(int)
    node_ids = list(range(len(Z)))

    fmnn = FMNN(theta=0.2, gamma=4.0)
    fmnn.fit(Z, y, node_ids, shuffle=False)

    cross_overlaps = fmnn.cross_class_overlap_count()
    assert cross_overlaps == 0, f"Expected 0 cross-class overlaps, found {cross_overlaps}"


def test_theta_zero_behaviour() -> None:
    """theta = 0 should create one box per distinct point (nearest-neighbor-like)."""
    Z = np.array([
        [0.1, 0.1],
        [0.2, 0.2],
        [0.3, 0.3],
        [0.4, 0.4],
    ])
    y = np.array([0, 0, 1, 1])
    node_ids = [0, 1, 2, 3]

    fmnn = FMNN(theta=0.0, gamma=4.0)
    fmnn.fit(Z, y, node_ids, shuffle=False)

    assert fmnn.num_boxes() == 4, f"theta=0 should create 4 boxes, got {fmnn.num_boxes()}"


def test_larger_theta_fewer_or_equal_boxes() -> None:
    """Larger theta allows larger boxes, resulting in fewer or equal boxes."""
    rng = np.random.RandomState(42)
    Z = rng.rand(60, 4)
    y = (Z[:, 0] > 0.5).astype(int)
    node_ids = list(range(len(Z)))

    counts = []
    thetas = [0.05, 0.2, 0.5]
    for th in thetas:
        fmnn = FMNN(theta=th, gamma=4.0)
        fmnn.fit(Z, y, node_ids, shuffle=False)
        counts.append(fmnn.num_boxes())

    # Monotonically non-increasing
    assert counts[0] >= counts[1] >= counts[2], f"Expected non-increasing box counts, got {counts}"


def test_tie_handling_and_confidence() -> None:
    """Test deterministic tie handling (lowest class index) and predict_with_confidence."""
    fmnn = FMNN(theta=0.3, gamma=4.0)
    # Add symmetric boxes for class 0 and class 1 equidistant from test point
    fmnn.boxes[1] = HyperBox(id=1, v=np.array([0.2, 0.2]), w=np.array([0.4, 0.4]), cls=0)
    fmnn.boxes[2] = HyperBox(id=2, v=np.array([0.6, 0.6]), w=np.array([0.8, 0.8]), cls=1)
    fmnn.num_classes = 2
    fmnn.n_dims = 2

    # Point equidistant from both boxes
    mid_point = np.array([[0.5, 0.5]])
    scores = fmnn.class_scores(mid_point)
    assert np.isclose(scores[0, 0], scores[0, 1])

    # Predict should break tie by choosing lowest class index (0)
    pred = fmnn.predict(mid_point)
    assert pred[0] == 0

    # predict_with_confidence should flag tie as True
    preds, conf, is_tie = fmnn.predict_with_confidence(mid_point)
    assert preds[0] == 0
    assert is_tie[0] is True or is_tie[0] == 1


def test_member_assignments() -> None:
    """Check that get_assignments maps every training node to exactly one box."""
    Z = np.array([[0.1, 0.2], [0.15, 0.25], [0.8, 0.8]])
    y = np.array([0, 0, 1])
    node_ids = [101, 102, 103]

    fmnn = FMNN(theta=0.3, gamma=4.0)
    fmnn.fit(Z, y, node_ids)

    assignments = fmnn.get_assignments()
    assert set(assignments.keys()) == {101, 102, 103}
    for nid in node_ids:
        bid = assignments[nid]
        assert nid in fmnn.boxes[bid].members


# Verify fit_with_pseudo attaches covered nodes, expands within theta, and respects allow_new_boxes.
def test_fit_with_pseudo_coverage_and_expansion() -> None:
    Z_tr = np.array([[0.1, 0.1], [0.3, 0.3], [0.85, 0.85], [0.95, 0.95]])
    y_tr = np.array([0, 0, 1, 1])
    tr_ids = [0, 1, 2, 3]

    # Case 1: covered point [0.2, 0.2] + expandable point [0.4, 0.4] (L1 span 0.6 <= 2*0.4=0.8) -> 0 new boxes
    Z_ps_expand = np.array([[0.2, 0.2], [0.4, 0.4]])
    y_ps_expand = np.array([0, 0])
    ps_ids_expand = [10, 11]

    model_exp = FMNN(theta=0.4, gamma=4.0).fit_with_pseudo(
        Z_tr, y_tr, tr_ids, Z_ps_expand, y_ps_expand, ps_ids_expand, allow_new_boxes=True
    )
    assert model_exp.num_boxes() == 2
    assert model_exp.cross_class_overlap_count() == 0

    # Case 2: far outside point [0.05, 0.78] exceeds L1 budget 0.8 -> creates new box only if allow_new_boxes=True
    Z_ps_far = np.array([[0.2, 0.2], [0.05, 0.78]])
    y_ps_far = np.array([0, 0])
    ps_ids_far = [10, 12]

    model_allow = FMNN(theta=0.4, gamma=4.0).fit_with_pseudo(
        Z_tr, y_tr, tr_ids, Z_ps_far, y_ps_far, ps_ids_far, allow_new_boxes=True
    )
    assert model_allow.num_boxes() == 3

    model_no_new = FMNN(theta=0.4, gamma=4.0).fit_with_pseudo(
        Z_tr, y_tr, tr_ids, Z_ps_far, y_ps_far, ps_ids_far, allow_new_boxes=False
    )
    assert model_no_new.num_boxes() == 2
    assert len(model_no_new.boxes_of_class(0)) >= 1
    assert len(model_no_new.boxes_of_class(1)) >= 1
