from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import (
    ExperimentConfig,
    _convert_boxes_to_records,
    _records_to_fmnn_model,
    _validation_objective,
    fit_baseline,
)
from src.gshr.refinement import GraphStructureAwareRefinement


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize the complete GCN + FMNN + GSHR pipeline.')
    parser.add_argument('--config', type=Path, default=Path('configs/cora.yaml'))
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/figures/cora_gcn_fmnn_gshr_representation.png'),
    )
    args = parser.parse_args()

    cfg_data = yaml.safe_load(args.config.read_text(encoding='utf-8')) or {}
    dataset_cfg = cfg_data.get('dataset', {})
    model_cfg = cfg_data.get('model', {})
    fmnn_cfg = cfg_data.get('fmnn', {})
    refinement_cfg = cfg_data.get('refinement', {})
    config = ExperimentConfig(
        seed=int(cfg_data.get('seed', 0)),
        dataset_name=str(dataset_cfg.get('name', 'Cora')),
        data_root=str(dataset_cfg.get('root', 'data')),
        hidden_dim=int(model_cfg.get('hidden_dim', 64)),
        dropout=float(model_cfg.get('dropout', 0.5)),
        learning_rate=float(model_cfg.get('learning_rate', 0.01)),
        weight_decay=float(model_cfg.get('weight_decay', 0.0005)),
        max_epochs=int(model_cfg.get('max_epochs', 200)),
        patience=int(model_cfg.get('patience', 30)),
        fmnn_theta=float(fmnn_cfg.get('theta', 0.3)),
        fmnn_gamma=float(fmnn_cfg.get('gamma', 1.0)),
        max_side_length=float(refinement_cfg.get('max_side_length', 1.0)),
        volume_limit=float(refinement_cfg.get('volume_limit', 1e6)),
        min_validation_purity=float(refinement_cfg.get('min_validation_purity', 0.2)),
        min_objective_gain=float(refinement_cfg.get('min_objective_gain', 0.0)),
    )
    dataset, _, embeddings, fmnn, _ = fit_baseline(config)
    boxes = _convert_boxes_to_records(fmnn, embeddings, dataset.y, train_mask=dataset.train_mask)
    train_labels = {int(i): int(dataset.y[i]) for i in np.where(dataset.train_mask)[0]}
    rule = dict(refinement_cfg.get('prune_rule', {
        'max_loss_increase': 0.1,
        'min_class_coverage': 1,
        'min_graph_cohesion': 0.0,
        'min_prune_purity': 0.2,
    }))
    validation_x = embeddings[dataset.val_mask]
    validation_y = dataset.y[dataset.val_mask]
    objective = lambda candidate: _validation_objective(
        candidate, embeddings, dataset, config.fmnn_theta, config.fmnn_gamma
    )
    refinement = GraphStructureAwareRefinement(
        boxes, edge_index=dataset.edge_index, node_labels=train_labels
    )
    merged = refinement.merge(
        max_side_length=config.max_side_length,
        volume_limit=config.volume_limit,
        validation_embeddings=validation_x,
        validation_labels=validation_y,
        min_validation_purity=config.min_validation_purity,
        validation_objective=objective,
        min_objective_gain=config.min_objective_gain,
    )
    refined = GraphStructureAwareRefinement(
        merged, edge_index=dataset.edge_index, node_labels=train_labels
    ).prune(
        rule,
        validation_embeddings=validation_x,
        validation_labels=validation_y,
        validation_objective=objective,
        min_objective_gain=config.min_objective_gain,
    )

    points = PCA(n_components=2, random_state=0).fit_transform(embeddings)
    projection = PCA(n_components=2, random_state=0).fit(embeddings)
    components = projection.components_
    axis = plt.subplots(figsize=(12, 8))[1]
    scatter = axis.scatter(
        points[:, 0], points[:, 1], c=dataset.y, cmap='tab10', s=12, alpha=0.35, linewidths=0
    )
    for box in refined:
        center = (box.lower + box.upper) / 2.0
        half_width = (box.upper - box.lower) / 2.0
        box_center = projection.transform(center.reshape(1, -1))[0]
        box_half_width = np.abs(components) @ half_width
        axis.add_patch(Rectangle(
            box_center - box_half_width,
            2 * box_half_width[0],
            2 * box_half_width[1],
            facecolor=scatter.cmap(scatter.norm(box.label)),
            edgecolor=scatter.cmap(scatter.norm(box.label)),
            alpha=0.2,
            linewidth=2,
        ))
        axis.text(box_center[0], box_center[1], str(box.label), ha='center', va='center', fontweight='bold')

    axis.set_title(f'Cora: GCN + FMNN + GSHR ({len(boxes)} → {len(refined)} hyperboxes)')
    axis.set_xlabel('PCA component 1')
    axis.set_ylabel('PCA component 2')
    axis.grid(alpha=0.2)
    axis.text(
        0.01, 0.01,
        'Dots = papers; regions = approximate 2D projections of refined 64D hyperboxes',
        transform=axis.transAxes, fontsize=9,
        bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'},
    )
    plt.colorbar(scatter, ax=axis, label='Cora topic label')
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=180)
    print({'output': str(args.output), 'baseline_boxes': len(boxes), 'gshr_boxes': len(refined)})


if __name__ == '__main__':
    main()
