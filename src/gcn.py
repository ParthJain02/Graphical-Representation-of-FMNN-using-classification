from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


@dataclass
class GCNConfig:
    hidden_dim: int = 64
    dropout: float = 0.5
    learning_rate: float = 1e-2
    weight_decay: float = 5e-4
    max_epochs: int = 200
    patience: int = 30


class GCNModel(nn.Module):
    """Two-layer GCN that supports both real graph training and the legacy scaffold API."""

    def __init__(self, num_features: int, num_classes: int, config: GCNConfig | None = None) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.config = config or GCNConfig()
        self.embedding_dim = self.config.hidden_dim
        self.conv1 = GCNConv(self.num_features, self.config.hidden_dim)
        self.conv2 = GCNConv(self.config.hidden_dim, self.num_classes)
        self.dropout = nn.Dropout(self.config.dropout)
        self._last_embeddings: np.ndarray | None = None
        self._last_edge_index: torch.Tensor | None = None

    def _make_edge_index(self, num_nodes: int, edge_index: np.ndarray | torch.Tensor | None = None) -> torch.Tensor:
        if edge_index is not None:
            edge = torch.as_tensor(edge_index, dtype=torch.long)
            if edge.numel() == 0:
                return torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])
            return edge
        idx = torch.arange(num_nodes, dtype=torch.long)
        return torch.stack([idx, idx])

    def forward(self, x: torch.Tensor | np.ndarray, edge_index: np.ndarray | torch.Tensor | None = None) -> torch.Tensor:
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        if x_tensor.ndim != 2:
            raise ValueError('Input must be a 2D matrix of node features.')
        if x_tensor.shape[1] != self.num_features:
            raise ValueError('Feature dimension mismatch.')
        edge = self._make_edge_index(x_tensor.shape[0], edge_index)
        hidden = self.conv1(x_tensor, edge)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        logits = self.conv2(hidden, edge)
        return logits

    def fit(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        train_mask: np.ndarray | torch.Tensor | None = None,
        edge_index: np.ndarray | torch.Tensor | None = None,
        val_mask: np.ndarray | torch.Tensor | None = None,
        max_epochs: int | None = None,
    ) -> 'GCNModel':
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.long)
        if x_tensor.ndim != 2:
            raise ValueError('Input must be a 2D matrix of node features.')
        if x_tensor.shape[0] != y_tensor.shape[0]:
            raise ValueError('Feature matrix and labels must have the same length.')
        if train_mask is None:
            train_mask = torch.ones(x_tensor.shape[0], dtype=torch.bool)
        train_mask_tensor = torch.as_tensor(train_mask, dtype=torch.bool)
        if val_mask is not None:
            val_mask_tensor = torch.as_tensor(val_mask, dtype=torch.bool)
        else:
            val_mask_tensor = None

        edge_tensor = self._make_edge_index(x_tensor.shape[0], edge_index)
        self._last_edge_index = edge_tensor

        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        criterion = nn.CrossEntropyLoss()
        best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        best_val_loss = float('inf')
        patience_remaining = self.config.patience
        epochs = self.config.max_epochs if max_epochs is None else max_epochs

        for _ in range(int(epochs)):
            self.train()
            optimizer.zero_grad()
            logits = self(x_tensor, edge_tensor)
            loss = criterion(logits[train_mask_tensor], y_tensor[train_mask_tensor])
            loss.backward()
            optimizer.step()

            if val_mask_tensor is not None and val_mask_tensor.any():
                with torch.no_grad():
                    self.eval()
                    val_logits = self(x_tensor, edge_tensor)
                    val_loss = criterion(val_logits[val_mask_tensor], y_tensor[val_mask_tensor])
                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                        patience_remaining = self.config.patience
                    else:
                        patience_remaining -= 1
                        if patience_remaining <= 0:
                            break

        if best_state:
            self.load_state_dict(best_state)

        self.eval()
        with torch.no_grad():
            self._last_embeddings = self.embed(x_tensor, edge_tensor)
        return self

    def embed(self, x: np.ndarray | torch.Tensor, edge_index: np.ndarray | torch.Tensor | None = None) -> np.ndarray:
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        edge = self._make_edge_index(x_tensor.shape[0], edge_index if edge_index is not None else self._last_edge_index)
        self.eval()
        with torch.no_grad():
            hidden = self.conv1(x_tensor, edge)
            hidden = F.relu(hidden)
            return hidden.cpu().numpy()

    def predict(self, x: np.ndarray | torch.Tensor, edge_index: np.ndarray | torch.Tensor | None = None) -> np.ndarray:
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        edge = self._make_edge_index(x_tensor.shape[0], edge_index if edge_index is not None else self._last_edge_index)
        self.eval()
        with torch.no_grad():
            logits = self(x_tensor, edge)
            return torch.argmax(logits, dim=1).cpu().numpy()
