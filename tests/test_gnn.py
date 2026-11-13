"""Unit tests for GNN training, embedding extraction, and normalization."""

from __future__ import annotations

import copy
import numpy as np
import pytest
import torch

from src.data import GraphData, load_dataset
from src.embeddings import EmbeddingScaler, extract_embeddings, fit_transform_all
from src.gnn import (
    GCNEncoder,
    GNNClassifier,
    SAGEEncoder,
    search_encoder_hyperparams,
    search_encoders_validation,
    train_gnn,
)
from src.utils import load_config


@pytest.fixture(scope="module")
def cora_data() -> GraphData:
    """Fixture to load Cora dataset once."""
    return load_dataset("Cora", root="./data")


@pytest.fixture(scope="module")
def test_config() -> dict:
    """Fixture providing a fast configuration for tests."""
    cfg = load_config("configs/default.yaml")
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["gnn"]["max_epochs"] = 20
    cfg_copy["gnn"]["patience"] = 10
    cfg_copy["run"]["device"] = "cpu"
    return cfg_copy


def test_encoder_and_classifier_architectures(cora_data: GraphData) -> None:
    """Verify forward and embed shapes for GCN and GraphSAGE models."""
    in_dim = cora_data.x.size(1)
    hidden_dim = 32
    out_dim = 16
    num_classes = cora_data.num_classes

    # Test GCN
    gcn_model = GNNClassifier(in_dim, hidden_dim, out_dim, num_classes, model_type="gcn")
    emb_gcn = gcn_model.embed(cora_data.x, cora_data.edge_index)
    logits_gcn = gcn_model(cora_data.x, cora_data.edge_index)

    assert emb_gcn.shape == (cora_data.num_nodes, out_dim)
    assert logits_gcn.shape == (cora_data.num_nodes, num_classes)

    # Test SAGE
    sage_model = GNNClassifier(in_dim, hidden_dim, out_dim, num_classes, model_type="sage")
    emb_sage = sage_model.embed(cora_data.x, cora_data.edge_index)
    logits_sage = sage_model(cora_data.x, cora_data.edge_index)

    assert emb_sage.shape == (cora_data.num_nodes, out_dim)
    assert logits_sage.shape == (cora_data.num_nodes, num_classes)


def test_short_training_and_embedding_shape(cora_data: GraphData, test_config: dict) -> None:
    """Run short training and assert extracted embedding shape and history records."""
    model, history, test_metrics = train_gnn(cora_data, test_config, seed=42)

    assert len(history) <= 20
    assert "val_acc" in history[0]
    assert "val_macro_f1" in history[0]
    assert "test_acc" in test_metrics
    assert "test_macro_f1" in test_metrics

    # Extract embeddings
    Z = extract_embeddings(model, cora_data, device="cpu")
    out_dim = test_config["gnn"]["out_dim"]
    assert Z.shape == (cora_data.num_nodes, out_dim)

    # Check model parameters are frozen
    for p in model.parameters():
        assert not p.requires_grad


def test_scaled_embeddings_bounded_in_01(cora_data: GraphData, test_config: dict) -> None:
    """Assert that scaled train embeddings and val/test embeddings lie in [0, 1]."""
    model, _, _ = train_gnn(cora_data, test_config, seed=42)
    Z = extract_embeddings(model, cora_data, device="cpu")

    train_idx = cora_data.train_idx.cpu().numpy()
    val_idx = cora_data.val_idx.cpu().numpy()
    test_idx = cora_data.test_idx.cpu().numpy()

    scaler = EmbeddingScaler(clip=True)
    scaler.fit(Z[train_idx])
    Z_scaled = scaler.transform(Z)

    # Train embeddings must lie in [0, 1]
    assert np.all(Z_scaled[train_idx] >= 0.0)
    assert np.all(Z_scaled[train_idx] <= 1.0)

    # Val and test embeddings must also lie in [0, 1] due to clipping
    assert np.all(Z_scaled[val_idx] >= 0.0)
    assert np.all(Z_scaled[val_idx] <= 1.0)
    assert np.all(Z_scaled[test_idx] >= 0.0)
    assert np.all(Z_scaled[test_idx] <= 1.0)

    # Test fit_transform_all consistency
    Z_all = fit_transform_all(Z, cora_data.train_idx, clip=True)
    assert np.allclose(Z_scaled, Z_all)


def test_scaler_fitted_only_on_training_nodes(cora_data: GraphData) -> None:
    """Verify that changing non-training rows does not affect fitted min and max."""
    num_nodes = cora_data.num_nodes
    dim = 16
    train_idx = cora_data.train_idx.cpu().numpy()

    rng = np.random.RandomState(42)
    Z1 = rng.randn(num_nodes, dim)
    scaler1 = EmbeddingScaler(clip=True)
    scaler1.fit(Z1[train_idx])

    # Create Z2 with completely altered non-training nodes
    Z2 = Z1.copy()
    non_train_mask = np.ones(num_nodes, dtype=bool)
    non_train_mask[train_idx] = False
    Z2[non_train_mask] = Z2[non_train_mask] * 1000.0 + 5000.0

    scaler2 = EmbeddingScaler(clip=True)
    scaler2.fit(Z2[train_idx])

    # Min and max must be identical because they depend only on training nodes
    assert np.allclose(scaler1.min_val, scaler2.min_val)
    assert np.allclose(scaler1.max_val, scaler2.max_val)


def test_constant_dimension_handling() -> None:
    """Verify that constant feature dimensions are handled safely without division by zero."""
    train_data = np.array([
        [1.0, 5.0],
        [2.0, 5.0],
        [3.0, 5.0],
    ])
    scaler = EmbeddingScaler(clip=True)
    scaled = scaler.fit_transform(train_data)

    assert not np.isnan(scaled).any()
    assert not np.isinf(scaled).any()
    assert np.all(scaled[:, 1] == 0.5)


# Verify val_loss early stopping records val_loss and restores minimum validation loss checkpoint.
def test_val_loss_early_stopping(cora_data: GraphData, test_config: dict) -> None:
    cfg_vl = copy.deepcopy(test_config)
    cfg_vl["gnn"]["early_stop_metric"] = "val_loss"
    cfg_vl["gnn"]["max_epochs"] = 15
    cfg_vl["gnn"]["patience"] = 5

    model, history, test_metrics = train_gnn(cora_data, cfg_vl, seed=0)
    assert "val_loss" in history[0]
    best_ep = test_metrics["best_epoch"]
    assert history[best_ep - 1]["val_loss"] == pytest.approx(test_metrics["best_val_loss"])
    assert test_metrics["best_val_loss"] == pytest.approx(min(h["val_loss"] for h in history))


# Verify validation encoder search ignores test labels completely and returns deterministic config.
def test_encoder_search_no_test_leakage(cora_data: GraphData, test_config: dict) -> None:
    cfg_s = copy.deepcopy(test_config)
    cfg_s["gnn"]["max_epochs"] = 6
    cfg_s["gnn"]["patience"] = 3

    res_clean = search_encoder_hyperparams(
        data=cora_data,
        cfg=cfg_s,
        search_seeds=[0],
        hidden_grid=[16, 32],
        dropout_grid=[0.5],
        weight_decay_grid=[5e-4],
        lr_grid=[0.01],
        early_stop_metric="val_loss",
        patience=3,
        max_epochs=6,
    )

    corrupted = copy.deepcopy(cora_data)
    corrupted.y[corrupted.test_idx] = -999
    res_corrupt = search_encoder_hyperparams(
        data=corrupted,
        cfg=cfg_s,
        search_seeds=[0],
        hidden_grid=[16, 32],
        dropout_grid=[0.5],
        weight_decay_grid=[5e-4],
        lr_grid=[0.01],
        early_stop_metric="val_loss",
        patience=3,
        max_epochs=6,
    )

    assert res_clean["best_config"] == res_corrupt["best_config"]
    assert res_clean["best_val_macro_f1_mean"] == pytest.approx(res_corrupt["best_val_macro_f1_mean"])
    assert len(res_clean["all_trials"]) == 2


# Verify L2-normalized then min-max scaled embeddings stay bounded in [0, 1] without test leakage.
def test_l2_norm_then_minmax_scaler(cora_data: GraphData) -> None:
    rng = np.random.RandomState(7)
    Z = rng.randn(cora_data.num_nodes, 64) * 5.0
    train_idx = cora_data.train_idx.cpu().numpy()

    Z_l2 = fit_transform_all(Z, train_idx, clip=True, l2_norm=True)
    assert Z_l2.shape == (cora_data.num_nodes, 64)
    assert np.all(Z_l2 >= 0.0) and np.all(Z_l2 <= 1.0)

    Z_alt = Z.copy()
    non_train = np.ones(cora_data.num_nodes, dtype=bool)
    non_train[train_idx] = False
    Z_alt[non_train] = Z_alt[non_train] * 100.0 + 42.0
    Z_l2_alt = fit_transform_all(Z_alt, train_idx, clip=True, l2_norm=True)
    assert np.allclose(Z_l2[train_idx], Z_l2_alt[train_idx])


# Verify APPNP, GCNII, and H2GCN encoders run forward/backward and select strictly on validation.
def test_v4_encoders_forward_and_search(cora_data: GraphData, test_config: dict) -> None:
    in_dim = cora_data.x.size(1)
    hidden_dim = 16
    out_dim = 8
    num_classes = cora_data.num_classes

    for mtype in ["appnp", "gcnii", "h2gcn"]:
        mod = GNNClassifier(in_dim, hidden_dim, out_dim, num_classes, model_type=mtype, gcnii_layers=3)
        emb = mod.embed(cora_data.x, cora_data.edge_index)
        logits = mod(cora_data.x, cora_data.edge_index)
        assert emb.shape == (cora_data.num_nodes, out_dim)
        assert logits.shape == (cora_data.num_nodes, num_classes)
        logits.sum().backward()

    corrupted = copy.deepcopy(cora_data)
    corrupted.y[corrupted.test_idx] = -999
    cfg_s = copy.deepcopy(test_config)
    cfg_s["gnn"]["gcnii_layers"] = 2
    res_clean = search_encoders_validation(
        data=cora_data, cfg=cfg_s, encoders=["gcn", "h2gcn"],
        search_seeds=[0], hidden_grid=[16], dropout_grid=[0.5],
        weight_decay_grid=[5e-6], lr_grid=[0.01], max_epochs=4, patience=2,
    )
    res_corrupt = search_encoders_validation(
        data=corrupted, cfg=cfg_s, encoders=["gcn", "h2gcn"],
        search_seeds=[0], hidden_grid=[16], dropout_grid=[0.5],
        weight_decay_grid=[5e-6], lr_grid=[0.01], max_epochs=4, patience=2,
    )
    assert res_clean["best_encoder"] == res_corrupt["best_encoder"]
    assert res_clean["best_config"] == res_corrupt["best_config"]
