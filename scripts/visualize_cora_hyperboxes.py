from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize Cora embeddings and FMNN hyperboxes.')
    parser.add_argument(
        '--run-dir',
        type=Path,
        default=Path('results/processed/baseline_seed_42_20261101T060632Z'),
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/figures/cora_hyperbox_representation.png'),
    )
    args = parser.parse_args()

    artifact = np.load(args.run_dir / 'embeddings_and_splits.npz')
    embeddings = artifact['embeddings']
    labels = artifact['labels']
    boxes = json.loads((args.run_dir / 'fmnn_boxes.json').read_text(encoding='utf-8'))

    projection = PCA(n_components=2, random_state=0)
    points = projection.fit_transform(embeddings)
    components = projection.components_
    point_min = points.min(axis=0)
    point_max = points.max(axis=0)
    point_span = np.maximum(point_max - point_min, 1e-8)

    fig, axis = plt.subplots(figsize=(12, 8))
    scatter = axis.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        cmap='tab10',
        s=12,
        alpha=0.35,
        linewidths=0,
    )

    for box in boxes:
        lower = np.asarray(box['lower'], dtype=float)
        upper = np.asarray(box['upper'], dtype=float)
        center = (lower + upper) / 2.0
        half_width = (upper - lower) / 2.0
        box_center = projection.transform(center.reshape(1, -1))[0]
        box_half_width = np.abs(components) @ half_width
        width, height = 2.0 * box_half_width

        # This rectangle encloses the projection of the original 64D box.
        axis.add_patch(
            Rectangle(
                box_center - box_half_width,
                width,
                height,
                facecolor=scatter.cmap(scatter.norm(box['label'])),
                edgecolor=scatter.cmap(scatter.norm(box['label'])),
                alpha=0.16,
                linewidth=1.5,
            )
        )
        axis.text(
            box_center[0],
            box_center[1],
            str(box['label']),
            fontsize=8,
            ha='center',
            va='center',
            fontweight='bold',
        )

    margin = 0.06 * point_span
    axis.set_xlim(point_min[0] - margin[0], point_max[0] + margin[0])
    axis.set_ylim(point_min[1] - margin[1], point_max[1] + margin[1])
    axis.set_xlabel('PCA component 1')
    axis.set_ylabel('PCA component 2')
    axis.set_title('Cora GCN Embeddings with FMNN Hyperboxes')
    axis.text(
        0.01,
        0.01,
        'Points = papers; colored regions = approximate 2D projections of 64D hyperboxes',
        transform=axis.transAxes,
        fontsize=9,
        bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'},
    )
    axis.grid(alpha=0.2)
    fig.colorbar(scatter, ax=axis, label='Cora topic label')
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == '__main__':
    main()
