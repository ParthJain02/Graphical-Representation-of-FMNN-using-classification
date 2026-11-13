"""Unit tests for graph-structure-aware hyperbox merging."""

from __future__ import annotations

import copy
import numpy as np
import pytest
import scipy.sparse as sp

from src.fmnn import FMNN, HyperBox
from src.merge import GraphContext, debug_merge_candidates, fast_gshr_merge, geometry_only_merge, gshr_merge


@pytest.fixture
def toy_graph_and_fmnn() -> tuple[FMNN, GraphContext, dict]:
    """Create a toy graph and FMNN for testing merge logic."""
    fmnn = FMNN(theta=0.5, gamma=4.0)
    fmnn.n_dims = 2
    fmnn.num_classes = 2

    # Two same-class boxes close and connected by an edge
    # Node 0 in Box 1, Node 1 in Box 2
    fmnn.boxes[1] = HyperBox(id=1, v=np.array([0.1, 0.1]), w=np.array([0.2, 0.2]), cls=0, members=[0])
    fmnn.boxes[2] = HyperBox(id=2, v=np.array([0.22, 0.22]), w=np.array([0.32, 0.32]), cls=0, members=[1])
    # Distant box of class 1 (node 2)
    fmnn.boxes[3] = HyperBox(id=3, v=np.array([0.8, 0.8]), w=np.array([0.9, 0.9]), cls=1, members=[2])
    fmnn.next_id = 4

    # Graph with edge (0, 1)
    edges = [(0, 1)]
    row = [0, 1]
    col = [1, 0]
    data = np.ones(2, dtype=np.float32)
    adj_csr = sp.coo_matrix((data, (row, col)), shape=(3, 3)).tocsr()
    degree = np.array(adj_csr.sum(axis=1), dtype=np.int64).flatten()
    y = np.array([0, 0, 1])
    train_mask = np.array([True, True, True])

    ctx = GraphContext(adj_csr=adj_csr, degree=degree, y=y, train_mask=train_mask)
    cfg_merge = {
        "alpha": 0.35,
        "beta": 0.35,
        "delta": 0.20,
        "eta": 0.10,
        "tau_merge": 0.4,
        "knn_candidates": 5,
        "max_volume_inflation": 0.8,
        "use_graph_edges_for_candidates": True,
        "theta": 0.5,
    }

    return fmnn, ctx, cfg_merge


def test_close_connected_boxes_merge(toy_graph_and_fmnn: tuple[FMNN, GraphContext, dict]) -> None:
    """Verify that close, edge-connected same-class boxes successfully merge."""
    fmnn, ctx, cfg = toy_graph_and_fmnn
    merged_fmnn, log = gshr_merge(fmnn, ctx, cfg)

    assert len(log) == 1
    assert log.boxes_before == 3
    assert log.boxes_after == 2
    assert merged_fmnn.num_boxes() == 2

    # Check that Box 1 and Box 2 were merged into a single box
    assert 1 not in merged_fmnn.boxes
    assert 2 not in merged_fmnn.boxes
    assert 3 in merged_fmnn.boxes

    new_box = [b for b in merged_fmnn.boxes.values() if b.cls == 0][0]
    assert set(new_box.members) == {0, 1}
    np.testing.assert_allclose(new_box.v, np.array([0.1, 0.1]))
    np.testing.assert_allclose(new_box.w, np.array([0.32, 0.32]))


def test_cannot_swallow_other_class_box() -> None:
    """Verify that two same-class boxes that would swallow an other-class box do NOT merge."""
    fmnn = FMNN(theta=0.9, gamma=4.0)
    fmnn.n_dims = 2
    fmnn.num_classes = 2

    # Box 1 and Box 2 of class 0 on opposite sides
    fmnn.boxes[1] = HyperBox(id=1, v=np.array([0.1, 0.1]), w=np.array([0.2, 0.2]), cls=0, members=[0])
    fmnn.boxes[2] = HyperBox(id=2, v=np.array([0.7, 0.7]), w=np.array([0.8, 0.8]), cls=0, members=[1])
    # Box 3 of class 1 directly in between
    fmnn.boxes[3] = HyperBox(id=3, v=np.array([0.4, 0.4]), w=np.array([0.5, 0.5]), cls=1, members=[2])
    fmnn.next_id = 4

    adj_csr = sp.csr_matrix((3, 3), dtype=np.float32)
    degree = np.zeros(3, dtype=np.int64)
    y = np.array([0, 0, 1])
    train_mask = np.ones(3, dtype=bool)
    ctx = GraphContext(adj_csr=adj_csr, degree=degree, y=y, train_mask=train_mask)

    cfg = {"alpha": 0.5, "beta": 0.0, "delta": 0.5, "eta": 0.0, "tau_merge": 0.1, "theta": 0.9}
    merged_fmnn, log = gshr_merge(fmnn, ctx, cfg)

    assert len(log) == 0, "Merge should be rejected due to cross-class overlap"
    assert log.rejections["cross_class_overlap"] >= 1
    assert merged_fmnn.num_boxes() == 3


def test_merge_exceeding_theta_rejected() -> None:
    """Verify that a candidate merge exceeding theta is rejected by size limit."""
    fmnn = FMNN(theta=0.2, gamma=4.0)
    fmnn.n_dims = 2
    fmnn.num_classes = 1

    # In 2D, sum of side lengths for [0.1, 0.1] to [0.5, 0.5] is 0.4 + 0.4 = 0.8 > 2*0.2 = 0.4
    fmnn.boxes[1] = HyperBox(id=1, v=np.array([0.1, 0.1]), w=np.array([0.2, 0.2]), cls=0, members=[0])
    fmnn.boxes[2] = HyperBox(id=2, v=np.array([0.4, 0.4]), w=np.array([0.5, 0.5]), cls=0, members=[1])
    fmnn.next_id = 3

    adj_csr = sp.csr_matrix((2, 2), dtype=np.float32)
    ctx = GraphContext(adj_csr=adj_csr, degree=np.zeros(2, dtype=np.int64), y=np.array([0, 0]), train_mask=np.ones(2, dtype=bool))
    cfg = {"tau_merge": 0.1, "theta": 0.2}

    merged_fmnn, log = gshr_merge(fmnn, ctx, cfg)
    assert len(log) == 0
    assert log.rejections["size_limit"] >= 1
    assert merged_fmnn.num_boxes() == 2


def test_merge_determinism(toy_graph_and_fmnn: tuple[FMNN, GraphContext, dict]) -> None:
    """Verify that running gshr_merge twice yields identical boxes and IDs."""
    fmnn, ctx, cfg = toy_graph_and_fmnn

    fmnn1, log1 = gshr_merge(fmnn, ctx, cfg)
    fmnn2, log2 = gshr_merge(fmnn, ctx, cfg)

    assert fmnn1.num_boxes() == fmnn2.num_boxes()
    assert sorted(fmnn1.boxes.keys()) == sorted(fmnn2.boxes.keys())
    for bid in fmnn1.boxes:
        np.testing.assert_allclose(fmnn1.boxes[bid].v, fmnn2.boxes[bid].v)
        np.testing.assert_allclose(fmnn1.boxes[bid].w, fmnn2.boxes[bid].w)
        assert fmnn1.boxes[bid].members == fmnn2.boxes[bid].members


def test_geometry_only_equivalence(toy_graph_and_fmnn: tuple[FMNN, GraphContext, dict]) -> None:
    """Verify that geometry_only_merge produces identical output to beta=0 without graph edges."""
    fmnn, ctx, cfg = toy_graph_and_fmnn

    geom_fmnn, geom_log = geometry_only_merge(fmnn, ctx, cfg)

    cfg_beta0 = copy.deepcopy(cfg)
    cfg_beta0["beta"] = 0.0
    cfg_beta0["use_graph_edges_for_candidates"] = False
    direct_fmnn, direct_log = gshr_merge(fmnn, ctx, cfg_beta0)

    assert geom_fmnn.num_boxes() == direct_fmnn.num_boxes()
    assert len(geom_log) == len(direct_log)
    for bid in geom_fmnn.boxes:
        assert bid in direct_fmnn.boxes
        np.testing.assert_allclose(geom_fmnn.boxes[bid].v, direct_fmnn.boxes[bid].v)


def test_members_preserved_and_no_cross_class_overlap(toy_graph_and_fmnn: tuple[FMNN, GraphContext, dict]) -> None:
    """Verify that all members are preserved and cross-class overlap is 0."""
    fmnn, ctx, cfg = toy_graph_and_fmnn
    orig_members = set()
    for b in fmnn.boxes.values():
        orig_members.update(b.members)

    merged_fmnn, _ = gshr_merge(fmnn, ctx, cfg)

    post_members = set()
    for b in merged_fmnn.boxes.values():
        post_members.update(b.members)

    assert orig_members == post_members
    assert merged_fmnn.cross_class_overlap_count() == 0
    assert merged_fmnn.num_boxes() <= fmnn.num_boxes()


# Verify fast_gshr_merge skips non-overlapping pairs and disabled merge, and debug_merge_candidates logs stats.
def test_fast_merge_and_debug_stats(toy_graph_and_fmnn: tuple[FMNN, GraphContext, dict]) -> None:
    fmnn, ctx, cfg = toy_graph_and_fmnn

    dbg = debug_merge_candidates(fmnn, ctx, cfg)
    assert dbg["total_same_class_pairs"] == 1
    assert dbg["candidate_pairs_count"] == 1
    assert dbg["pass_counts"]["same_class_geom_overlap"] == 0
    assert dbg["score_distribution"]["count"] == 1

    # Boxes 1 and 2 do not overlap geometrically, so fast_gshr_merge skips when require_same_class_overlap=True.
    cfg_fast = {**cfg, "enabled": True, "require_same_class_overlap": True}
    skipped_fmnn, skip_log = fast_gshr_merge(fmnn, ctx, cfg_fast)
    assert len(skip_log) == 0
    assert skip_log.rejections["no_geom_overlap"] == 1
    assert skipped_fmnn.num_boxes() == 3

    # Disabled merge exits immediately with zero merges.
    disabled_fmnn, dis_log = fast_gshr_merge(fmnn, ctx, {**cfg, "enabled": False})
    assert len(dis_log) == 0
    assert disabled_fmnn.num_boxes() == 3

    # Overlapping same-class pair DOES merge cleanly under fast_gshr_merge.
    fmnn_ov = copy.deepcopy(fmnn)
    fmnn_ov.boxes[2].v = np.array([0.15, 0.15])
    merged_ov, ov_log = fast_gshr_merge(fmnn_ov, ctx, cfg_fast)
    assert len(ov_log) == 1
    assert merged_ov.num_boxes() == 2
