import numpy as np

from src.gshr.graph_stats import conductance, compute_boundary_and_internal_edges


def test_conductance_and_edge_counts():
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=int)
    members = {0, 1}
    internal, boundary = compute_boundary_and_internal_edges(members, edge_index)
    assert internal == 2
    assert boundary == 2
    score = conductance(members, edge_index)
    assert 0.0 < score < 1.0
