from __future__ import annotations

from typing import Any

from .data import DatasetBundle
from .gcn import GCNConfig, GCNModel


def train_two_layer_gcn(dataset: DatasetBundle, config: GCNConfig | None = None) -> tuple[GCNModel, dict[str, Any]]:
    model = GCNModel(num_features=dataset.x.shape[1], num_classes=dataset.num_classes, config=config)
    model.fit(dataset.x, dataset.y, dataset.train_mask, edge_index=dataset.edge_index)
    info = {'embedding_dim': model.embedding_dim, 'trained': True}
    return model, info
