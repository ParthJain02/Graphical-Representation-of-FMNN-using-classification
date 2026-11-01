from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from hbbrain.numerical_data.incremental_learner.fmnn import FMNNClassifier

from .data import load_dataset
from .fmnn_adapter import create_fmnn_adapter
from .gcn import GCNConfig, GCNModel
from .gshr.hyperbox import HyperboxRecord
from .gshr.refinement import GraphStructureAwareRefinement
from .metrics import accuracy_score, macro_f1_score
from .normalization import MinMaxNormalizer
from .reproducibility import collect_software_versions, set_seed


@dataclass
class ExperimentConfig:
    seed: int = 0
    dataset_name: str = 'synthetic-cora'
    data_root: str = 'data'
    num_nodes: int = 20
    num_features: int = 8
    num_classes: int = 4
    hidden_dim: int = 64
    dropout: float = 0.5
    learning_rate: float = 1e-2
    weight_decay: float = 5e-4
    max_epochs: int = 200
    patience: int = 30
    fmnn_theta: float = 0.3
    fmnn_gamma: float = 1.0
    max_side_length: float = 1.0
    volume_limit: float = 1e6
    min_validation_purity: float = 0.2
    min_objective_gain: float = 0.0
    graph_enabled: bool = True
    merge_enabled: bool = True
    prune_enabled: bool = True
    class_purity_safeguards: bool = True


def fit_baseline(config: ExperimentConfig) -> tuple[Any, Any, np.ndarray, Any, MinMaxNormalizer]:
    cfg = config
    set_seed(cfg.seed)
    dataset = load_dataset(
        dataset_name=cfg.dataset_name,
        root=cfg.data_root,
        num_nodes=cfg.num_nodes,
        num_features=cfg.num_features,
        num_classes=cfg.num_classes,
        seed=cfg.seed,
    )
    model = GCNModel(
        num_features=dataset.x.shape[1],
        num_classes=dataset.num_classes,
        config=GCNConfig(
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            max_epochs=cfg.max_epochs,
            patience=cfg.patience,
        ),
    )
    model.fit(
        dataset.x,
        dataset.y,
        dataset.train_mask,
        edge_index=dataset.edge_index,
        val_mask=dataset.val_mask,
    )
    embeddings = model.embed(dataset.x, dataset.edge_index)
    normalizer = MinMaxNormalizer().fit(embeddings[dataset.train_mask])
    embeddings = normalizer.transform(embeddings, clip=True)
    fmnn = create_fmnn_adapter(theta=cfg.fmnn_theta, gamma=cfg.fmnn_gamma)
    masked_x = embeddings[dataset.train_mask]
    masked_y = dataset.y[dataset.train_mask]
    fmnn.fit(masked_x, masked_y)
    return dataset, model, embeddings, fmnn, normalizer


def _split_metrics(preds: np.ndarray, dataset: Any) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for name, mask in (
        ('train', dataset.train_mask),
        ('validation', dataset.val_mask),
        ('test', dataset.test_mask),
    ):
        y_true = dataset.y[mask]
        y_pred = preds[mask]
        metrics[name] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': macro_f1_score(y_true, y_pred, dataset.num_classes),
        }
    return metrics


def run_baseline_experiment(config: ExperimentConfig | None = None) -> dict[str, Any]:
    cfg = config or ExperimentConfig()
    dataset, model, embeddings, fmnn, normalizer = fit_baseline(cfg)
    return summarize_baseline(dataset, model, embeddings, fmnn, normalizer)


def run_softmax_experiment(config: ExperimentConfig | None = None) -> dict[str, Any]:
    cfg = config or ExperimentConfig()
    set_seed(cfg.seed)
    dataset = load_dataset(
        dataset_name=cfg.dataset_name,
        root=cfg.data_root,
        num_nodes=cfg.num_nodes,
        num_features=cfg.num_features,
        num_classes=cfg.num_classes,
        seed=cfg.seed,
    )
    model = GCNModel(
        num_features=dataset.x.shape[1],
        num_classes=dataset.num_classes,
        config=GCNConfig(
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            max_epochs=cfg.max_epochs,
            patience=cfg.patience,
        ),
    )
    model.fit(
        dataset.x,
        dataset.y,
        dataset.train_mask,
        edge_index=dataset.edge_index,
        val_mask=dataset.val_mask,
    )
    predictions = model.predict(dataset.x, dataset.edge_index)
    split_metrics = _split_metrics(predictions, dataset)
    return {
        'seed': cfg.seed,
        'dataset': dataset.metadata,
        'accuracy': split_metrics['test']['accuracy'],
        'macro_f1': split_metrics['test']['macro_f1'],
        'split_metrics': split_metrics,
        'num_hyperboxes': 0,
        'selected_epoch': model.selected_epoch,
        'versions': collect_software_versions(),
    }


def summarize_baseline(
    dataset: Any,
    model: GCNModel,
    embeddings: np.ndarray,
    fmnn: Any,
    normalizer: MinMaxNormalizer,
) -> dict[str, Any]:
    preds = fmnn.predict(embeddings)
    split_metrics = _split_metrics(preds, dataset)
    return {
        'dataset': dataset.metadata,
        'accuracy': split_metrics['test']['accuracy'],
        'macro_f1': split_metrics['test']['macro_f1'],
        'split_metrics': split_metrics,
        'num_hyperboxes': len(fmnn.get_boxes()),
        'selected_epoch': model.selected_epoch,
        'normalization': {
            'min': normalizer.min_.tolist(),
            'max': normalizer.max_.tolist(),
            'clip': True,
            'fit_split': 'train',
        },
        'versions': collect_software_versions(),
    }


def _convert_boxes_to_records(fmnn_model: Any, feature_matrix: np.ndarray, label_vector: np.ndarray, train_mask: np.ndarray | None = None) -> list[HyperboxRecord]:
    records: list[HyperboxRecord] = []
    node_ids = np.arange(len(label_vector)) if train_mask is None else np.where(train_mask)[0]
    train_features = feature_matrix if train_mask is None else feature_matrix[train_mask]
    for idx, box in enumerate(fmnn_model.get_boxes()):
        lower = np.asarray(box.lower, dtype=float)
        upper = np.asarray(box.upper, dtype=float)
        mask = (train_features >= lower).all(axis=1) & (train_features <= upper).all(axis=1)
        member_ids = node_ids[mask]
        records.append(
            HyperboxRecord(
                box_id=idx,
                lower=lower,
                upper=upper,
                label=int(box.label),
                members=set(int(node_id) for node_id in member_ids),
            )
        )
    return records


def _records_to_fmnn_model(boxes: list[HyperboxRecord], theta: float = 0.3, gamma: float = 1.0) -> FMNNClassifier:
    if not boxes:
        return FMNNClassifier(theta=theta, gamma=gamma)
    V = np.asarray([box.lower for box in boxes], dtype=float)
    W = np.asarray([box.upper for box in boxes], dtype=float)
    C = np.asarray([int(box.label) for box in boxes], dtype=int)
    return FMNNClassifier(theta=theta, gamma=gamma, V=V, W=W, C=C)


def _validation_objective(
    boxes: list[HyperboxRecord],
    embeddings: np.ndarray,
    dataset: Any,
    theta: float,
    gamma: float,
) -> float:
    if not boxes:
        return float('-inf')
    model = _records_to_fmnn_model(boxes, theta=theta, gamma=gamma)
    validation_mask = dataset.val_mask
    predictions = np.asarray(model.predict(embeddings[validation_mask]), dtype=int)
    labels = dataset.y[validation_mask]
    return float(
        0.5 * accuracy_score(labels, predictions)
        + 0.5 * macro_f1_score(labels, predictions, dataset.num_classes)
    )


def _evaluate_box_variant(model: FMNNClassifier, embeddings: np.ndarray, dataset: Any) -> dict[str, dict[str, float]]:
    y_pred = np.asarray(model.predict(embeddings), dtype=int)
    return _split_metrics(y_pred, dataset)


def run_refinement_ablation(
    config: ExperimentConfig | None = None,
    prune_rule: dict[str, float] | None = None,
    mode: str = 'full',
    fitted: tuple[Any, Any, np.ndarray, Any, MinMaxNormalizer] | None = None,
) -> dict[str, Any]:
    cfg = config or ExperimentConfig()
    valid_modes = {'baseline', 'full', 'merge', 'prune', 'geometry_merge', 'graph_merge', 'unsafe_full'}
    if mode not in valid_modes:
        raise ValueError(f'Unsupported refinement mode: {mode}')
    if fitted is None:
        fitted = fit_baseline(cfg)
    dataset, _, embeddings, fmnn, _ = fitted
    baseline_predictions = np.asarray(fmnn.predict(embeddings), dtype=int)
    baseline_split_metrics = _split_metrics(baseline_predictions, dataset)
    boxes = _convert_boxes_to_records(fmnn, embeddings, dataset.y, train_mask=dataset.train_mask)
    train_labels = {
        int(node_id): int(dataset.y[node_id])
        for node_id in np.where(dataset.train_mask)[0]
    }
    graph_enabled = cfg.graph_enabled and mode not in {'geometry_merge'}
    safeguards_enabled = cfg.class_purity_safeguards and mode != 'unsafe_full'
    refinement = GraphStructureAwareRefinement(
        boxes,
        edge_index=dataset.edge_index if graph_enabled else None,
        node_labels=train_labels,
    )
    rule = dict(prune_rule or {
        'max_loss_increase': 0.1,
        'min_class_coverage': 1,
        'min_graph_cohesion': 0.0,
        'min_prune_purity': 0.2,
    })
    if not safeguards_enabled:
        rule['min_prune_purity'] = 0.0
    merge_enabled = cfg.merge_enabled and mode not in {'baseline', 'prune'}
    prune_enabled = cfg.prune_enabled and mode not in {'merge', 'geometry_merge', 'graph_merge'}
    val_x = embeddings[dataset.val_mask]
    val_y = dataset.y[dataset.val_mask]
    objective = lambda candidate: _validation_objective(
        candidate,
        embeddings,
        dataset,
        cfg.fmnn_theta,
        cfg.fmnn_gamma,
    )
    merged_boxes = refinement.merge(
        max_side_length=cfg.max_side_length,
        volume_limit=cfg.volume_limit,
        validation_embeddings=val_x,
        validation_labels=val_y,
        min_validation_purity=cfg.min_validation_purity if safeguards_enabled else 0.0,
        validation_objective=objective,
        min_objective_gain=cfg.min_objective_gain,
    ) if merge_enabled else boxes
    pruned_boxes = refinement.prune(
        rule,
        validation_embeddings=val_x,
        validation_labels=val_y,
        validation_objective=objective,
        min_objective_gain=cfg.min_objective_gain,
    ) if prune_enabled else boxes
    full_boxes = GraphStructureAwareRefinement(
        merged_boxes,
        edge_index=dataset.edge_index if graph_enabled else None,
        node_labels=train_labels,
    ).prune(
        rule,
        validation_embeddings=val_x,
        validation_labels=val_y,
        validation_objective=objective,
        min_objective_gain=cfg.min_objective_gain,
    )
    if mode in {'geometry_merge', 'graph_merge'}:
        variants = {mode: merged_boxes}
    elif mode == 'merge':
        variants = {'merge': merged_boxes}
    elif mode == 'prune':
        variants = {'prune': pruned_boxes}
    elif mode == 'baseline':
        variants = {'baseline': boxes}
    elif mode == 'unsafe_full':
        variants = {'unsafe_full': full_boxes}
    else:
        variants = {'baseline': boxes, 'merge': merged_boxes, 'prune': pruned_boxes, 'full': full_boxes}
    summary = {
        'mode': mode,
        'seed': cfg.seed,
        'dataset': dataset.metadata,
        'baseline_accuracy': baseline_split_metrics['test']['accuracy'],
        'baseline_macro_f1': baseline_split_metrics['test']['macro_f1'],
        'baseline_split_metrics': baseline_split_metrics,
        'baseline_num_hyperboxes': len(boxes),
        'variants': [],
    }
    for name, variant_boxes in variants.items():
        model_variant = _records_to_fmnn_model(
            variant_boxes,
            theta=cfg.fmnn_theta,
            gamma=cfg.fmnn_gamma,
        )
        split_metrics = _evaluate_box_variant(model_variant, embeddings, dataset)
        summary['variants'].append(
            {
                'name': name,
                'box_count': len(variant_boxes),
                'accuracy': split_metrics['test']['accuracy'],
                'macro_f1': split_metrics['test']['macro_f1'],
                'split_metrics': split_metrics,
            }
        )
    return summary
