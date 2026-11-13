"""Unit tests for hyperbox graph statistics and caching."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from src.fmnn import FMNN, HyperBox
from src.graph_stats import (
    BoxGraphCache,
    boundary_edges,
    build_box_adjacency,
    class_purity,
    conductance,
    edges_between,
    graph_association,
    internal_edges,
    neighbor_jaccard,
    neighbor_label_agreement,
    volume,
)


@pytest.fixture
def handmade_graph() -> dict:
    """Build a 6-node hand-crafted graph: triangle (0,1,2) and path (2,3,4,5).

    Edges:
      (0, 1), (1, 2), (2, 0)
      (2, 3), (3, 4), (4, 5)
    Degrees:
      node 0: 2
      node 1: 2
      node 2: 3
      node 3: 2
      node 4: 2
      node 5: 1
    Total volume: 12
    """
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5)]
    row = [e[0] for e in edges] + [e[1] for e in edges]
    col = [e[1] for e in edges] + [e[0] for e in edges]
    data = np.ones(len(row), dtype=np.float32)
    adj_csr = sp.coo_matrix((data, (row, col)), shape=(6, 6)).tocsr()
    degree = np.array(adj_csr.sum(axis=1), dtype=np.int64).flatten()

    y = np.array([0, 0, 1, 1, 0, 1])
    train_mask = np.array([True, True, True, True, False, False])

    return {
        "adj_csr": adj_csr,
        "degree": degree,
        "y": y,
        "train_mask": train_mask,
    }


def test_handmade_graph_exact_statistics(handmade_graph: dict) -> None:
    """Verify internal edges, boundary edges, volume, and conductance against hand-derived values."""
    adj = handmade_graph["adj_csr"]
    deg = handmade_graph["degree"]

    # S_A = {0, 1}:
    # Internal: (0, 1) -> 1
    # Volume: deg(0) + deg(1) = 2 + 2 = 4
    # Boundary: (0, 2), (1, 2) -> 2
    # Conductance: 2 / (2*1 + 2) = 2 / 4 = 0.5
    S_A = [0, 1]
    assert internal_edges(S_A, adj) == 1
    assert volume(S_A, deg) == 4
    assert boundary_edges(S_A, adj, deg) == 2
    assert np.isclose(conductance(S_A, adj, deg), 0.5)

    # S_B = {2, 3}:
    # Internal: (2, 3) -> 1
    # Volume: deg(2) + deg(3) = 3 + 2 = 5
    # Boundary: (2, 0), (2, 1), (3, 4) -> 3
    # Conductance: 3 / (2*1 + 3) = 3 / 5 = 0.6
    S_B = [2, 3]
    assert internal_edges(S_B, adj) == 1
    assert volume(S_B, deg) == 5
    assert boundary_edges(S_B, adj, deg) == 3
    assert np.isclose(conductance(S_B, adj, deg), 0.6)

    # S_C = {4, 5}:
    # Internal: (4, 5) -> 1
    # Volume: deg(4) + deg(5) = 2 + 1 = 3
    # Boundary: (4, 3) -> 1
    # Conductance: 1 / (2*1 + 1) = 1 / 3
    S_C = [4, 5]
    assert internal_edges(S_C, adj) == 1
    assert volume(S_C, deg) == 3
    assert boundary_edges(S_C, adj, deg) == 1
    assert np.isclose(conductance(S_C, adj, deg), 1.0 / 3.0)


def test_edges_between_and_graph_association(handmade_graph: dict) -> None:
    """Verify edges_between, graph_association symmetry, and bounds."""
    adj = handmade_graph["adj_csr"]
    deg = handmade_graph["degree"]

    S_A = [0, 1]
    S_B = [2, 3]

    # Between S_A and S_B: edges (0, 2) and (1, 2) -> 2
    e_ab = edges_between(S_A, S_B, adj)
    e_ba = edges_between(S_B, S_A, adj)
    assert e_ab == 2
    assert e_ba == 2

    # Graph association
    # 2 / sqrt(4 * 5) = 2 / sqrt(20) ~ 0.4472136
    assoc_ab = graph_association(S_A, S_B, adj, deg)
    assoc_ba = graph_association(S_B, S_A, adj, deg)
    assert np.isclose(assoc_ab, assoc_ba)
    assert np.isclose(assoc_ab, 2.0 / np.sqrt(20.0))
    assert 0.0 <= assoc_ab <= 1.0


def test_neighbor_jaccard(handmade_graph: dict) -> None:
    """Verify Jaccard similarity of neighbor sets."""
    adj = handmade_graph["adj_csr"]
    S_A = [0, 1]  # nbrs: {0, 1, 2}
    S_B = [2, 3]  # nbrs: {0, 1, 2, 3, 4}

    # Intersect: {0, 1, 2} (3 nodes)
    # Union: {0, 1, 2, 3, 4} (5 nodes)
    # Jaccard = 3 / 5 = 0.6
    jacc = neighbor_jaccard(S_A, S_B, adj)
    assert np.isclose(jacc, 3.0 / 5.0)


def test_conductance_edge_cases() -> None:
    """Check conductance edge cases: no edges, all internal edges, and all boundary edges."""
    # 1. No edges: isolated node
    adj_empty = sp.csr_matrix((4, 4), dtype=np.float32)
    deg_empty = np.zeros(4, dtype=np.int64)
    cond_empty = conductance([0, 1], adj_empty, deg_empty)
    assert not np.isnan(cond_empty)
    assert cond_empty == 0.0

    # 2. All edges are internal: isolated clique {0, 1, 2}
    edges = [(0, 1), (1, 2), (2, 0)]
    row = [e[0] for e in edges] + [e[1] for e in edges]
    col = [e[1] for e in edges] + [e[0] for e in edges]
    adj_clique = sp.coo_matrix((np.ones(6), (row, col)), shape=(3, 3)).tocsr()
    deg_clique = np.array(adj_clique.sum(axis=1), dtype=np.int64).flatten()
    cond_clique = conductance([0, 1, 2], adj_clique, deg_clique)
    assert cond_clique == 0.0

    # 3. All edges are boundary: single node connected to exterior
    # Node 0 has 2 edges, both to 1 and 2
    cond_boundary = conductance([0], adj_clique, deg_clique)
    # internal = 0, boundary = 2 -> 2 / (0 + 2) = 1.0
    assert np.isclose(cond_boundary, 1.0)


def test_purity_and_neighbor_agreement(handmade_graph: dict) -> None:
    """Verify class purity and neighbor label agreement."""
    adj = handmade_graph["adj_csr"]
    y = handmade_graph["y"]
    mask = handmade_graph["train_mask"]

    # S_A = [0, 1], class 0: both labeled 0
    purity_a = class_purity([0, 1], box_class=0, y=y, train_mask=mask)
    assert purity_a == 1.0

    # Neighbor agreement for S_A:
    # External neighbors of {0, 1} is {2}.
    # Node 2 is in train_mask (True), has label y[2] = 1.
    # Agreement with class 0: 0/1 = 0.0
    agr_a, count_a = neighbor_label_agreement([0, 1], box_class=0, adj_csr=adj, y=y, train_mask=mask)
    assert agr_a == 0.0
    assert count_a == 1

    # Empty neighbor case returns neutral 0.5 and count 0
    # S_C = [4, 5]: neighbors are {3}. Node 3 is in train_mask, label 1.
    # What if a node has no labeled neighbors?
    agr_empty, count_empty = neighbor_label_agreement([], box_class=0, adj_csr=adj, y=y, train_mask=mask)
    assert agr_empty == 0.5
    assert count_empty == 0


def test_build_box_adjacency_matches_brute_force(handmade_graph: dict) -> None:
    """Verify that build_box_adjacency (sparse B^T A B) exactly matches edges_between."""
    adj = handmade_graph["adj_csr"]

    fmnn = FMNN()
    # Create 3 boxes
    fmnn.boxes[1] = HyperBox(id=1, v=np.zeros(2), w=np.ones(2), cls=0, members=[0, 1])
    fmnn.boxes[2] = HyperBox(id=2, v=np.zeros(2), w=np.ones(2), cls=1, members=[2, 3])
    fmnn.boxes[3] = HyperBox(id=3, v=np.zeros(2), w=np.ones(2), cls=0, members=[4, 5])

    box_adj = build_box_adjacency(fmnn, adj)

    # Brute force checks
    bids = [1, 2, 3]
    for i in range(len(bids)):
        for j in range(i + 1, len(bids)):
            bid1, bid2 = bids[i], bids[j]
            s1 = fmnn.boxes[bid1].members
            s2 = fmnn.boxes[bid2].members
            exact_count = edges_between(s1, s2, adj)
            adj_count = box_adj.get((bid1, bid2), 0)
            assert exact_count == adj_count, f"Mismatch for pair ({bid1}, {bid2}): {exact_count} vs {adj_count}"


def test_box_graph_cache(handmade_graph: dict) -> None:
    """Test BoxGraphCache build_all, refresh, and drop."""
    adj = handmade_graph["adj_csr"]
    deg = handmade_graph["degree"]
    y = handmade_graph["y"]
    mask = handmade_graph["train_mask"]

    fmnn = FMNN()
    fmnn.boxes[1] = HyperBox(id=1, v=np.zeros(2), w=np.ones(2), cls=0, members=[0, 1])
    fmnn.boxes[2] = HyperBox(id=2, v=np.zeros(2), w=np.ones(2), cls=1, members=[2, 3])

    cache = BoxGraphCache(adj, deg, y, mask)
    cache.build_all(fmnn)

    stats1 = cache.get(1)
    assert stats1 is not None
    assert stats1.internal_edges == 1
    assert stats1.volume == 4
    assert stats1.purity == 1.0

    # Modify box 1 and refresh
    fmnn.boxes[1].members = [0]
    new_stats1 = cache.refresh(1, fmnn)
    assert new_stats1.internal_edges == 0
    assert new_stats1.volume == 2

    # Drop box 2
    cache.drop(2)
    assert cache.get(2) is None
