"""Unit tests for dataset loading, preprocessing, and structural properties."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data import GraphData, homophily_ratio, load_dataset


@pytest.fixture(scope="module")
def cora_data() -> GraphData:
    """Fixture to load Cora dataset once for the test module."""
    return load_dataset("Cora", root="./data")


def test_data_shapes(cora_data: GraphData) -> None:
    """Check that tensors and matrices have consistent shapes."""
    num_nodes = cora_data.num_nodes
    assert cora_data.name == "Cora"
    assert cora_data.x.dim() == 2
    assert cora_data.x.size(0) == num_nodes
    assert cora_data.y.dim() == 1
    assert cora_data.y.size(0) == num_nodes
    assert cora_data.edge_index.dim() == 2
    assert cora_data.edge_index.size(0) == 2
    assert cora_data.adj_csr.shape == (num_nodes, num_nodes)
    assert cora_data.degree.shape == (num_nodes,)
    assert cora_data.num_classes == 7


def test_disjoint_splits(cora_data: GraphData) -> None:
    """Check that train, val, and test indices are strictly disjoint."""
    train_set = set(cora_data.train_idx.cpu().tolist())
    val_set = set(cora_data.val_idx.cpu().tolist())
    test_set = set(cora_data.test_idx.cpu().tolist())

    assert len(train_set) > 0
    assert len(val_set) > 0
    assert len(test_set) > 0

    assert len(train_set & val_set) == 0, "Train and validation splits must be disjoint"
    assert len(train_set & test_set) == 0, "Train and test splits must be disjoint"
    assert len(val_set & test_set) == 0, "Validation and test splits must be disjoint"

    # Specific sizes expected for Cora public split
    assert len(train_set) == 140
    assert len(val_set) == 500
    assert len(test_set) == 1000


def test_adjacency_properties(cora_data: GraphData) -> None:
    """Check that adj_csr is symmetric, has zero diagonal, and has binary values."""
    adj = cora_data.adj_csr

    # Zero diagonal
    diag = adj.diagonal()
    assert np.all(diag == 0), "Adjacency matrix must have zero diagonal (no self-loops)"

    # Symmetric: diff with transpose has zero non-zero elements
    diff = adj - adj.T
    assert diff.nnz == 0, "Adjacency matrix must be symmetric"

    # Binary values
    assert np.all(adj.data == 1.0), "Adjacency matrix must be binary"


def test_degree_matches_adjacency(cora_data: GraphData) -> None:
    """Check that degree array exactly matches adj_csr row sums."""
    row_sums = np.array(cora_data.adj_csr.sum(axis=1)).flatten()
    assert np.array_equal(cora_data.degree, row_sums), "Degree array must match row sums of adj_csr"


def test_edge_index_consistency(cora_data: GraphData) -> None:
    """Check that edge_index has no self loops and matches adjacency nonzero count."""
    src = cora_data.edge_index[0].cpu().numpy()
    dst = cora_data.edge_index[1].cpu().numpy()

    # No self loops
    assert np.all(src != dst), "edge_index must have no self-loops"

    # Non-zero entries match
    assert cora_data.edge_index.size(1) == cora_data.adj_csr.nnz


def test_homophily_ratio(cora_data: GraphData) -> None:
    """Check that homophily ratio is a valid probability within [0, 1]."""
    homo = homophily_ratio(cora_data)
    assert 0.0 <= homo <= 1.0
    # For Cora, homophily is known to be around ~0.81
    assert 0.70 <= homo <= 0.90


def test_unsupported_dataset() -> None:
    """Check that requesting an unsupported dataset raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        load_dataset("UnsupportedBenchmark123", root="./data")
