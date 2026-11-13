"""Data loading, preprocessing, and dataset utilities for graph benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected


@dataclass
class GraphData:
    """Dataclass holding graph tensors and structural representations."""

    name: str
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    num_classes: int
    num_nodes: int
    adj_csr: sp.csr_matrix
    degree: np.ndarray


def homophily_ratio(data: GraphData) -> float:
    """Calculate edge homophily ratio using all ground-truth labels.

    WARNING:
    This function uses ground-truth labels across the entire graph and is strictly
    for offline dataset description and reporting. It must NEVER be called or used
    inside the GSHR method, model training, validation selection, merging, or pruning stages,
    as that would cause label leakage.
    """
    if data.edge_index.size(1) == 0:
        return 0.0

    src = data.edge_index[0]
    dst = data.edge_index[1]
    y_src = data.y[src]
    y_dst = data.y[dst]
    same_label_count = (y_src == y_dst).sum().item()
    return float(same_label_count / data.edge_index.size(1))


def load_dataset(name: str = "Cora", root: str = "./data") -> GraphData:
    """Load graph dataset using standard benchmarks and convert to GraphData.

    Args:
        name: Name of the benchmark dataset (e.g., 'Cora', 'CiteSeer', 'PubMed').
        root: Root directory for downloading and caching datasets.

    Returns:
        GraphData container with processed tensors, indices, and CSR adjacency.
    """
    name_clean = name.strip()
    name_lower = name_clean.lower()

    if name_lower in ["cora", "citeseer", "pubmed"]:
        standard_name = "Cora" if name_lower == "cora" else ("CiteSeer" if name_lower == "citeseer" else "PubMed")
        dataset = Planetoid(root=root, name=standard_name, split="public")
        pyg_data = dataset[0]

        # Extract node features, labels, and indices
        x = pyg_data.x.float()
        y = pyg_data.y.long().view(-1)
        num_nodes = int(x.size(0))
        num_classes = int(dataset.num_classes)

        train_idx = torch.nonzero(pyg_data.train_mask, as_tuple=False).view(-1).long()
        val_idx = torch.nonzero(pyg_data.val_mask, as_tuple=False).view(-1).long()
        test_idx = torch.nonzero(pyg_data.test_mask, as_tuple=False).view(-1).long()

        # Clean edge_index: remove self loops, ensure undirected, coalesce duplicates
        edge_index = pyg_data.edge_index
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)

        # Build SciPy CSR adjacency matrix (symmetric, binary, no self loops)
        row = edge_index[0].cpu().numpy()
        col = edge_index[1].cpu().numpy()
        data_vals = np.ones(len(row), dtype=np.float32)
        adj_coo = sp.coo_matrix((data_vals, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)
        adj_csr = adj_coo.tocsr()
        adj_csr.setdiag(0.0)
        adj_csr.eliminate_zeros()
        adj_csr.data = np.ones_like(adj_csr.data, dtype=np.float32)

        # Degree array as 1D numpy array
        degree = np.array(adj_csr.sum(axis=1), dtype=np.int64).flatten()

        return GraphData(
            name=standard_name,
            x=x,
            edge_index=edge_index,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            num_classes=num_classes,
            num_nodes=num_nodes,
            adj_csr=adj_csr,
            degree=degree,
        )

    elif name_lower == "chameleon":
        from torch_geometric.datasets import WikipediaNetwork
        dataset = WikipediaNetwork(root=root, name="chameleon", geom_gcn_preprocess=True)
        pyg_data = dataset[0]

        x = pyg_data.x.float()
        y = pyg_data.y.long().view(-1)
        num_nodes = int(x.size(0))
        num_classes = int(y.max().item()) + 1

        # Use first fixed split (split 0) from geom-gcn
        train_mask = pyg_data.train_mask[:, 0] if pyg_data.train_mask.dim() > 1 else pyg_data.train_mask
        val_mask = pyg_data.val_mask[:, 0] if pyg_data.val_mask.dim() > 1 else pyg_data.val_mask
        test_mask = pyg_data.test_mask[:, 0] if pyg_data.test_mask.dim() > 1 else pyg_data.test_mask

        train_idx = torch.nonzero(train_mask, as_tuple=False).view(-1).long()
        val_idx = torch.nonzero(val_mask, as_tuple=False).view(-1).long()
        test_idx = torch.nonzero(test_mask, as_tuple=False).view(-1).long()

        edge_index = pyg_data.edge_index
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)

        row = edge_index[0].cpu().numpy()
        col = edge_index[1].cpu().numpy()
        data_vals = np.ones(len(row), dtype=np.float32)
        adj_coo = sp.coo_matrix((data_vals, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)
        adj_csr = adj_coo.tocsr()
        adj_csr.setdiag(0.0)
        adj_csr.eliminate_zeros()
        adj_csr.data = np.ones_like(adj_csr.data, dtype=np.float32)

        degree = np.array(adj_csr.sum(axis=1), dtype=np.int64).flatten()

        return GraphData(
            name="Chameleon",
            x=x,
            edge_index=edge_index,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            num_classes=num_classes,
            num_nodes=num_nodes,
            adj_csr=adj_csr,
            degree=degree,
        )

    raise NotImplementedError(
        f"Dataset '{name}' is not supported yet in Part 1. "
        "Planetoid datasets 'Cora', 'CiteSeer', and 'PubMed' with public splits are currently supported."
    )


def print_dataset_summary(data: GraphData) -> None:
    """Print an informative summary of the loaded graph dataset."""
    homophily = homophily_ratio(data)
    num_edges_undirected = data.edge_index.size(1) // 2

    print("=" * 60)
    print(f"Dataset Summary: {data.name}")
    print("=" * 60)
    print(f"  Nodes:              {data.num_nodes}")
    print(f"  Features:           {data.x.size(1)}")
    print(f"  Classes:            {data.num_classes}")
    print(f"  Edges (undirected): {num_edges_undirected} (directed entries: {data.edge_index.size(1)})")
    print(f"  Train nodes:        {len(data.train_idx)}")
    print(f"  Val nodes:          {len(data.val_idx)}")
    print(f"  Test nodes:         {len(data.test_idx)}")
    print(f"  Homophily ratio:    {homophily:.4f}")
    print("=" * 60)
