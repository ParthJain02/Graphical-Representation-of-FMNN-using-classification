from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from torch_geometric.datasets import Planetoid
except Exception:  # pragma: no cover - fallback when PyG is unavailable
    Planetoid = None


@dataclass
class DatasetBundle:
    x: np.ndarray
    y: np.ndarray
    edge_index: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    num_classes: int
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_masks(
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    require_exhaustive: bool = True,
) -> None:
    if len(y) != len(train_mask) or len(y) != len(val_mask) or len(y) != len(test_mask):
        raise ValueError('Mask lengths must match the number of nodes.')
    combined = train_mask | val_mask | test_mask
    if require_exhaustive and not combined.all():
        raise ValueError('Each node must belong to exactly one split mask.')
    if np.any(train_mask & val_mask) or np.any(train_mask & test_mask) or np.any(val_mask & test_mask):
        raise ValueError('Split masks must be disjoint.')
    if y.shape[0] == 0:
        raise ValueError('Dataset cannot be empty.')


def load_planetoid_dataset(
    dataset_name: str,
    root: str | Path = 'data',
    split: str = 'public',
) -> DatasetBundle:
    if Planetoid is None:
        raise RuntimeError('torch_geometric is required to load real Planetoid datasets.')
    canonical_name = {'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed'}[dataset_name.lower()]
    dataset = Planetoid(root=str(root), name=canonical_name, split=split)
    data = dataset[0]
    x = np.asarray(data.x.numpy(), dtype=np.float32)
    y = np.asarray(data.y.numpy(), dtype=np.int64)
    edge_index = np.asarray(data.edge_index.numpy(), dtype=np.int64)
    train_mask = np.asarray(data.train_mask.numpy(), dtype=bool)
    val_mask = np.asarray(data.val_mask.numpy(), dtype=bool)
    test_mask = np.asarray(data.test_mask.numpy(), dtype=bool)
    validate_masks(y, train_mask, val_mask, test_mask, require_exhaustive=False)
    return DatasetBundle(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=int(data.y.max().item()) + 1,
        metadata={
            'split': split,
            'dataset': canonical_name,
            'source': 'Planetoid',
            'num_nodes': int(x.shape[0]),
            'num_features': int(x.shape[1]),
            'num_classes': int(data.y.max().item()) + 1,
        },
    )


def load_dataset(
    dataset_name: str = 'synthetic-cora',
    root: str | Path = 'data',
    num_nodes: int = 20,
    num_features: int = 8,
    num_classes: int = 4,
    seed: int = 0,
    split: str = 'public',
) -> DatasetBundle:
    name = str(dataset_name).lower()
    if name in {'cora', 'citeseer', 'pubmed'}:
        return load_planetoid_dataset(name, root=root, split=split)
    if name in {'synthetic-cora', 'synthetic'}:
        return load_cora_like_dataset(num_nodes=num_nodes, num_features=num_features, num_classes=num_classes, seed=seed)
    raise ValueError(f'Unsupported dataset name: {dataset_name}')


def load_cora_like_dataset(
    num_nodes: int = 20,
    num_features: int = 8,
    num_classes: int = 4,
    seed: int = 0,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(num_nodes, num_features))
    x = (x - x.min(axis=0)) / np.maximum((x.max(axis=0) - x.min(axis=0)), 1e-8)
    y = rng.integers(0, num_classes, size=num_nodes)

    edge_index: list[list[int]] = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            if rng.random() < 0.08:
                edge_index.append([i, j])
    if len(edge_index) == 0:
        edge_index = [[0, 1], [1, 0], [2, 3], [3, 2]]
    edge_index = np.asarray(edge_index, dtype=np.int64).T

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    idx = np.arange(num_nodes)
    rng.shuffle(idx)
    train_size = max(1, num_nodes // 3)
    val_size = max(1, num_nodes // 5)
    train_idx = idx[:train_size]
    val_idx = idx[train_size:train_size + val_size]
    test_idx = idx[train_size + val_size:]
    if len(test_idx) == 0:
        test_idx = np.array([num_nodes - 1], dtype=int)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    validate_masks(y, train_mask, val_mask, test_mask)
    return DatasetBundle(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=num_classes,
        metadata={'split': 'synthetic', 'seed': seed},
    )
