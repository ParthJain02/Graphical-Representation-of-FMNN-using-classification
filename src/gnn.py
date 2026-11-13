"""GNN encoder architectures and training routines."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.nn import APPNP, GCN2Conv, GCNConv, SAGEConv

from src.data import GraphData
from src.utils import get_device, set_seed


class GCNEncoder(nn.Module):
    """Two-layer GCN encoder with intermediate ReLU and dropout."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass generating node embeddings."""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class SAGEEncoder(nn.Module):
    """Two-layer GraphSAGE encoder with intermediate ReLU and dropout."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass generating node embeddings."""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class APPNPEncoder(nn.Module):
    """Decoupled MLP followed by K-step personalized PageRank propagation."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        K: int = 10,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.prop = APPNP(K=K, alpha=alpha, cached=True, add_self_loops=True, normalize=True)
        self.dropout = dropout
        self._cache_key: tuple[int, int, int, str] | None = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass computing MLP features before Personalized PageRank diffusion."""
        key = (int(edge_index.data_ptr()), int(edge_index.size(1)), int(x.size(0)), str(edge_index.device))
        if self._cache_key != key:
            self.prop._cached_edge_index = None
            self._cache_key = key
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = F.relu(self.lin1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)
        return self.prop(h, edge_index)


class GCNIIEncoder(nn.Module):
    """Deep GCNII encoder with initial residual connections and identity mapping."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        num_layers: int = 8,
        alpha: float = 0.1,
        lam: float = 0.5,
    ) -> None:
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GCN2Conv(
                channels=hidden_dim,
                alpha=alpha,
                theta=lam,
                layer=i + 1,
                shared_weights=True,
                cached=True,
                add_self_loops=True,
                normalize=True,
            )
            for i in range(num_layers)
        ])
        self.lin_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
        self._cache_key: tuple[int, int, int, str] | None = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass across initial projection, stacked GCN2 layers, and output head."""
        key = (int(edge_index.data_ptr()), int(edge_index.size(1)), int(x.size(0)), str(edge_index.device))
        if self._cache_key != key:
            for conv in self.convs:
                conv._cached_edge_index = None
            self._cache_key = key
        x = F.dropout(x, p=self.dropout, training=self.training)
        h0 = F.relu(self.lin_in(x))
        h = h0
        for conv in self.convs:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(conv(h, h0, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin_out(h)


class H2GCNEncoder(nn.Module):
    """H2GCN-style encoder concatenating ego, 1-hop, and 2-hop representations without self-loops."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.prop = APPNP(K=1, alpha=0.0, cached=True, add_self_loops=False, normalize=True)
        self.lin_out = nn.Linear(3 * hidden_dim, out_dim)
        self.dropout = dropout
        self._cache_key: tuple[int, int, int, str] | None = None
        self._ei_no_loop: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass keeping ego features distinct from 1-hop and 2-hop neighbor averages."""
        key = (int(edge_index.data_ptr()), int(edge_index.size(1)), int(x.size(0)), str(edge_index.device))
        if self._cache_key != key:
            self.prop._cached_edge_index = None
            mask = edge_index[0] != edge_index[1]
            self._ei_no_loop = edge_index[:, mask] if not bool(torch.all(mask)) else edge_index
            self._cache_key = key
        ei_no_loop = self._ei_no_loop if self._ei_no_loop is not None else edge_index
        r0 = F.relu(self.lin_in(F.dropout(x, p=self.dropout, training=self.training)))
        r1 = self.prop(r0, ei_no_loop)
        r2 = self.prop(r1, ei_no_loop)
        cat = torch.cat([r0, r1, r2], dim=-1)
        cat = F.dropout(cat, p=self.dropout, training=self.training)
        return self.lin_out(cat)


class GNNClassifier(nn.Module):
    """GNN model wrapping an encoder with a linear classification head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_classes: int,
        model_type: str = "gcn",
        dropout: float = 0.5,
        encoder: nn.Module | None = None,
        appnp_k: int = 10,
        appnp_alpha: float = 0.1,
        gcnii_layers: int = 8,
        gcnii_alpha: float = 0.1,
        gcnii_lambda: float = 0.5,
    ) -> None:
        super().__init__()
        if encoder is not None:
            self.encoder = encoder
        else:
            model_type_clean = model_type.strip().lower()
            if model_type_clean == "gcn":
                self.encoder = GCNEncoder(in_dim, hidden_dim, out_dim, dropout=dropout)
            elif model_type_clean == "sage":
                self.encoder = SAGEEncoder(in_dim, hidden_dim, out_dim, dropout=dropout)
            elif model_type_clean == "appnp":
                self.encoder = APPNPEncoder(in_dim, hidden_dim, out_dim, dropout=dropout, K=appnp_k, alpha=appnp_alpha)
            elif model_type_clean == "gcnii":
                self.encoder = GCNIIEncoder(
                    in_dim,
                    hidden_dim,
                    out_dim,
                    dropout=dropout,
                    num_layers=gcnii_layers,
                    alpha=gcnii_alpha,
                    lam=gcnii_lambda,
                )
            elif model_type_clean == "h2gcn":
                self.encoder = H2GCNEncoder(in_dim, hidden_dim, out_dim, dropout=dropout)
            else:
                raise ValueError(
                    f"Unsupported model type '{model_type}'. Choose 'gcn', 'sage', 'appnp', 'gcnii', or 'h2gcn'."
                )

        self.classifier = nn.Linear(out_dim, num_classes)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute node representations from the encoder."""
        return self.encoder(x, edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute class logits."""
        z = self.embed(x, edge_index)
        return self.classifier(z)


# Train GNN encoder with cross-entropy loss and validation early stopping.
def train_gnn(
    data: GraphData,
    cfg: dict[str, Any],
    seed: int = 0,
    return_test_metrics: bool = True,
) -> tuple[GNNClassifier, list[dict[str, Any]], dict[str, float]] | tuple[GNNClassifier, list[dict[str, Any]]]:
    """Train GNN encoder using cross-entropy on training nodes with validation early stopping.

    Args:
        data: GraphData instance.
        cfg: Configuration dictionary.
        seed: Random seed for reproducibility.
        return_test_metrics: If True, returns (model, history, test_metrics); else (model, history).

    Returns:
        Trained model with best validation weights restored, training history, and optional test metrics.
    """
    set_seed(seed)
    device = get_device(cfg)

    gnn_cfg = cfg.get("gnn", {})
    model_type = gnn_cfg.get("model", "gcn")
    hidden_dim = int(gnn_cfg.get("hidden", 64))
    out_dim = int(gnn_cfg.get("out_dim", 64))
    dropout = float(gnn_cfg.get("dropout", 0.5))
    lr = float(gnn_cfg.get("lr", 0.01))
    weight_decay = float(gnn_cfg.get("weight_decay", 5e-4))
    max_epochs = int(gnn_cfg.get("max_epochs", 500))
    patience = int(gnn_cfg.get("patience", 50))
    early_stop_metric = str(gnn_cfg.get("early_stop_metric", "val_macro_f1")).strip().lower()
    if early_stop_metric not in {"val_macro_f1", "val_loss"}:
        raise ValueError(
            f"Unsupported gnn.early_stop_metric '{early_stop_metric}'. Choose 'val_macro_f1' or 'val_loss'."
        )

    in_dim = int(data.x.size(1))
    num_classes = int(data.num_classes)

    model = GNNClassifier(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_classes=num_classes,
        model_type=model_type,
        dropout=dropout,
        appnp_k=int(gnn_cfg.get("appnp_k", 10)),
        appnp_alpha=float(gnn_cfg.get("appnp_alpha", 0.1)),
        gcnii_layers=int(gnn_cfg.get("gcnii_layers", 8)),
        gcnii_alpha=float(gnn_cfg.get("gcnii_alpha", 0.1)),
        gcnii_lambda=float(gnn_cfg.get("gcnii_lambda", 0.5)),
    ).to(device)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_idx = data.train_idx.to(device)
    val_idx = data.val_idx.to(device)
    test_idx = data.test_idx.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_macro_f1 = -1.0
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0

    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_out = model(x, edge_index)
            val_loss = float(criterion(val_out[val_idx], y[val_idx]).item())
            val_preds = val_out[val_idx].argmax(dim=-1).cpu().numpy()
            val_targets = y[val_idx].cpu().numpy()
            val_acc = float((val_preds == val_targets).mean())
            val_macro_f1 = float(f1_score(val_targets, val_preds, average="macro", zero_division=0))

        history.append({
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
        })

        improved = (
            val_loss < best_val_loss
            if early_stop_metric == "val_loss"
            else val_macro_f1 > best_val_macro_f1
        )
        if improved:
            best_val_macro_f1 = val_macro_f1
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Restore best checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    val_metrics: dict[str, float] = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_loss": best_val_loss,
    }
    setattr(model, "val_metrics", val_metrics)

    if not return_test_metrics:
        return model, history

    # Compute test performance of best model once (Baseline B1)
    model.eval()
    with torch.no_grad():
        test_out = model(x, edge_index)
        test_preds = test_out[test_idx].argmax(dim=-1).cpu().numpy()
        test_targets = y[test_idx].cpu().numpy()
        test_acc = float((test_preds == test_targets).mean())
        test_macro_f1 = float(f1_score(test_targets, test_preds, average="macro", zero_division=0))

    test_metrics: dict[str, float] = {
        **val_metrics,
        "test_acc": test_acc,
        "test_macro_f1": test_macro_f1,
    }
    setattr(model, "test_metrics", test_metrics)
    return model, history, test_metrics


# Search GNN encoder hyperparameters using validation macro-F1 only (early stop configurable).
def search_encoder_hyperparams(
    data: GraphData,
    cfg: dict[str, Any],
    search_seeds: list[int] | None = None,
    hidden_grid: list[int] | None = None,
    dropout_grid: list[float] | None = None,
    weight_decay_grid: list[float] | None = None,
    lr_grid: list[float] | None = None,
    early_stop_metric: str = "val_loss",
    patience: int = 50,
    max_epochs: int = 300,
) -> dict[str, Any]:
    """Grid-search GNN encoder config selecting by mean validation macro-F1 without reading test labels."""
    from itertools import product

    seeds = [int(s) for s in (search_seeds if search_seeds is not None else [0, 1, 2])]
    h_grid = list(hidden_grid) if hidden_grid is not None else [64, 128]
    d_grid = list(dropout_grid) if dropout_grid is not None else [0.5, 0.6]
    wd_grid = list(weight_decay_grid) if weight_decay_grid is not None else [5e-4, 1e-3]
    l_grid = list(lr_grid) if lr_grid is not None else [0.01, 0.005]

    base_gnn_cfg = copy.deepcopy(cfg.get("gnn", {}))
    trials: list[dict[str, Any]] = []

    for hidden, dropout, weight_decay, lr in product(h_grid, d_grid, wd_grid, l_grid):
        cand_cfg = copy.deepcopy(cfg)
        cand_cfg.setdefault("gnn", {})
        cand_cfg["gnn"]["hidden"] = int(hidden)
        cand_cfg["gnn"]["dropout"] = float(dropout)
        cand_cfg["gnn"]["weight_decay"] = float(weight_decay)
        cand_cfg["gnn"]["lr"] = float(lr)
        cand_cfg["gnn"]["early_stop_metric"] = early_stop_metric
        cand_cfg["gnn"]["patience"] = int(patience)
        cand_cfg["gnn"]["max_epochs"] = int(max_epochs)

        f1_vals: list[float] = []
        acc_vals: list[float] = []
        loss_vals: list[float] = []
        epoch_vals: list[int] = []

        for seed in seeds:
            model, _ = train_gnn(data, cand_cfg, seed=seed, return_test_metrics=False)
            vm = getattr(model, "val_metrics")
            f1_vals.append(float(vm["best_val_macro_f1"]))
            acc_vals.append(float(vm["best_val_acc"]))
            loss_vals.append(float(vm["best_val_loss"]))
            epoch_vals.append(int(vm["best_epoch"]))

        trials.append({
            "hidden": int(hidden),
            "dropout": float(dropout),
            "weight_decay": float(weight_decay),
            "lr": float(lr),
            "val_macro_f1_mean": float(np.mean(f1_vals)),
            "val_macro_f1_std": float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0,
            "val_acc_mean": float(np.mean(acc_vals)),
            "val_loss_mean": float(np.mean(loss_vals)),
            "best_epoch_mean": float(np.mean(epoch_vals)),
        })

    # Maximize validation macro-F1, breaking ties deterministically by lower val_loss then config order.
    best_trial = min(
        trials,
        key=lambda t: (
            -t["val_macro_f1_mean"],
            t["val_loss_mean"],
            t["hidden"],
            t["dropout"],
            t["weight_decay"],
            t["lr"],
        ),
    )

    best_gnn_cfg = {
        **base_gnn_cfg,
        "hidden": best_trial["hidden"],
        "dropout": best_trial["dropout"],
        "weight_decay": best_trial["weight_decay"],
        "lr": best_trial["lr"],
        "early_stop_metric": early_stop_metric,
        "patience": int(patience),
        "max_epochs": int(max_epochs),
    }

    return {
        "dataset": data.name,
        "search_seeds": seeds,
        "selection_metric": "val_macro_f1_mean",
        "early_stop_metric": early_stop_metric,
        "patience": int(patience),
        "max_epochs": int(max_epochs),
        "best_config": best_gnn_cfg,
        "best_val_macro_f1_mean": best_trial["val_macro_f1_mean"],
        "best_val_acc_mean": best_trial["val_acc_mean"],
        "best_val_loss_mean": best_trial["val_loss_mean"],
        "all_trials": trials,
    }


# Merge a saved encoder_config_<dataset>.json into a copy of cfg.
def apply_encoder_config(cfg: dict[str, Any], encoder_payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of cfg with cfg['gnn'] updated from saved encoder search output."""
    updated = copy.deepcopy(cfg)
    best_gnn = encoder_payload.get("best_config", encoder_payload)
    updated.setdefault("gnn", {}).update(best_gnn)
    return updated


# Search validation hyperparameters across multiple encoder architectures without reading test labels.
def search_encoders_validation(
    data: GraphData,
    cfg: dict[str, Any],
    encoders: list[str],
    search_seeds: list[int] | None = None,
    hidden_grid: list[int] | None = None,
    dropout_grid: list[float] | None = None,
    weight_decay_grid: list[float] | None = None,
    lr_grid: list[float] | None = None,
    early_stop_metric: str | None = None,
    patience: int | None = None,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    """Run per-encoder validation hyperparameter search and pick the best encoder on val macro-F1."""
    gnn_cfg = cfg.get("gnn", {})
    es_metric = str(early_stop_metric or gnn_cfg.get("early_stop_metric", "val_loss"))
    pat = int(patience if patience is not None else gnn_cfg.get("patience", 50))
    m_ep = int(max_epochs if max_epochs is not None else gnn_cfg.get("max_epochs", 300))

    per_encoder: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    for enc_name in encoders:
        enc_cfg = copy.deepcopy(cfg)
        enc_cfg.setdefault("gnn", {})["model"] = enc_name.strip().lower()
        res = search_encoder_hyperparams(
            data=data,
            cfg=enc_cfg,
            search_seeds=search_seeds,
            hidden_grid=hidden_grid,
            dropout_grid=dropout_grid,
            weight_decay_grid=weight_decay_grid,
            lr_grid=lr_grid,
            early_stop_metric=es_metric,
            patience=pat,
            max_epochs=m_ep,
        )
        per_encoder[enc_name] = res
        summary_rows.append({
            "encoder": enc_name,
            "best_val_macro_f1_mean": res["best_val_macro_f1_mean"],
            "best_val_acc_mean": res["best_val_acc_mean"],
            "best_val_loss_mean": res["best_val_loss_mean"],
            "best_config": res["best_config"],
        })

    best_row = min(
        summary_rows,
        key=lambda r: (-r["best_val_macro_f1_mean"], r["best_val_loss_mean"], r["encoder"]),
    )
    return {
        "dataset": data.name,
        "encoders_tested": list(encoders),
        "best_encoder": best_row["encoder"],
        "best_config": best_row["best_config"],
        "best_val_macro_f1_mean": best_row["best_val_macro_f1_mean"],
        "best_val_acc_mean": best_row["best_val_acc_mean"],
        "per_encoder": per_encoder,
    }
