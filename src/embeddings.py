"""Node embedding extraction and normalization utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.data import GraphData


def extract_embeddings(
    model: nn.Module,
    data: GraphData,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Extract node embeddings using the frozen GNN encoder.

    Args:
        model: Trained GNN model with embed method or encoder submodule.
        data: GraphData containing graph tensors.
        device: Torch device to execute embedding pass.

    Returns:
        Numpy array of shape (num_nodes, out_dim).
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        if hasattr(model, "embed"):
            embeddings = model.embed(x, edge_index)
        elif hasattr(model, "encoder"):
            embeddings = model.encoder(x, edge_index)
        else:
            raise AttributeError("Model must have an 'embed' method or 'encoder' submodule")

        return embeddings.detach().cpu().numpy()


# Extract raw pre-softmax logits for all nodes using frozen GNN weights.
def extract_logits(
    model: nn.Module,
    data: GraphData,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Compute raw logit scores from the frozen GNN classifier."""
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        return logits.double().detach().cpu().numpy()


# Extract GNN softmax probabilities for all nodes using frozen weights.
def extract_softmax_probs(
    model: nn.Module,
    data: GraphData,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Compute softmax class probabilities from the frozen GNN classifier."""
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        probs = torch.softmax(logits.double(), dim=-1)
        return probs.detach().cpu().numpy()


# Row-wise L2-normalise node embedding vectors safely without cross-node statistics.
def l2_normalize_embeddings(Z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Apply unit L2 normalization independently to each row of Z."""
    Z_arr = np.asarray(Z, dtype=np.float64)
    norms = np.linalg.norm(Z_arr, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, eps)
    return Z_arr / safe_norms


class EmbeddingScaler:
    """Min-max scaler that fits exclusively on training nodes and normalizes to [0, 1]."""

    def __init__(self, clip: bool = True, eps: float = 1e-12, l2_norm: bool = False) -> None:
        self.clip = clip
        self.eps = eps
        self.l2_norm = bool(l2_norm)
        self.min_val: np.ndarray | None = None
        self.max_val: np.ndarray | None = None

    def fit(self, Z_train: np.ndarray) -> EmbeddingScaler:
        """Store per-dimension min and max computed strictly on training nodes.

        Args:
            Z_train: Array of shape (num_train_nodes, dim).
        """
        if Z_train.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {Z_train.shape}")
        Z_work = l2_normalize_embeddings(Z_train, eps=self.eps) if self.l2_norm else np.asarray(Z_train, dtype=np.float64)
        self.min_val = np.min(Z_work, axis=0)
        self.max_val = np.max(Z_work, axis=0)
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """Map embeddings into [0, 1] using fitted training statistics.

        Args:
            Z: Array of shape (N, dim).
        """
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("EmbeddingScaler must be fitted on training data before calling transform")

        Z_work = l2_normalize_embeddings(Z, eps=self.eps) if self.l2_norm else np.asarray(Z, dtype=np.float64)
        span = self.max_val - self.min_val
        is_constant = span < self.eps

        # Safe denominator avoids divide by zero for constant dimensions
        denom = np.where(is_constant, 1.0, span + self.eps)
        Z_scaled = (Z_work - self.min_val) / denom

        if np.any(is_constant):
            Z_scaled[:, is_constant] = 0.5

        if self.clip:
            Z_scaled = np.clip(Z_scaled, 0.0, 1.0)

        return Z_scaled

    def fit_transform(self, Z_train: np.ndarray) -> np.ndarray:
        """Fit on Z_train and transform Z_train."""
        return self.fit(Z_train).transform(Z_train)


def fit_transform_all(
    Z: np.ndarray,
    train_idx: torch.Tensor | np.ndarray,
    clip: bool = True,
    l2_norm: bool = False,
) -> np.ndarray:
    """Scale embeddings of all nodes to [0, 1] using training nodes statistics only.

    Args:
        Z: Array of shape (N, dim) for all nodes.
        train_idx: Indices of training nodes.
        clip: Whether to clip values outside [0, 1].
        l2_norm: Whether to row-wise L2-normalize before min-max scaling.

    Returns:
        Array of shape (N, dim) with normalized embeddings.
    """
    Z_scaled, _ = fit_scaler_and_transform(Z, train_idx, clip=clip, l2_norm=l2_norm)
    return Z_scaled


# Fit an EmbeddingScaler strictly on training nodes and return both scaled embeddings and scaler.
def fit_scaler_and_transform(
    Z: np.ndarray,
    train_idx: torch.Tensor | np.ndarray,
    clip: bool = True,
    l2_norm: bool = False,
) -> tuple[np.ndarray, EmbeddingScaler]:
    if isinstance(train_idx, torch.Tensor):
        idx = train_idx.cpu().numpy()
    else:
        idx = np.asarray(train_idx)

    scaler = EmbeddingScaler(clip=clip, l2_norm=l2_norm)
    scaler.fit(Z[idx])
    return scaler.transform(Z), scaler
