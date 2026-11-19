from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_dataset
from src.fmnn_adapter import create_fmnn_adapter
from src.normalization import MinMaxNormalizer


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize basic FMNN on raw Cora features.')
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--theta', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/figures/cora_basic_fmnn_representation.png'),
    )
    args = parser.parse_args()

    dataset = load_dataset(dataset_name='Cora', root=args.data_root)
    normalizer = MinMaxNormalizer().fit(dataset.x[dataset.train_mask])
    features = normalizer.transform(dataset.x, clip=True)

    fmnn = create_fmnn_adapter(theta=args.theta, gamma=args.gamma)
    fmnn.fit(features[dataset.train_mask], dataset.y[dataset.train_mask])
    boxes = fmnn.get_boxes()

    projection = PCA(n_components=2, random_state=0)
    points = projection.fit_transform(features)
    components = projection.components_
    point_min = points.min(axis=0)
    point_max = points.max(axis=0)
    point_span = np.maximum(point_max - point_min, 1e-8)

    fig, axis = plt.subplots(figsize=(12, 8))
    scatter = axis.scatter(
        points[:, 0],
        points[:, 1],
        c=dataset.y,
        cmap='tab10',
        s=12,
        alpha=0.35,
        linewidths=0,
    )

    for box in boxes:
        center = (box.lower + box.upper) / 2.0
        half_width = (box.upper - box.lower) / 2.0
        box_center = projection.transform(center.reshape(1, -1))[0]
        box_half_width = np.abs(components) @ half_width
        axis.add_patch(
            Rectangle(
                box_center - box_half_width,
                2.0 * box_half_width[0],
                2.0 * box_half_width[1],
                facecolor=scatter.cmap(scatter.norm(box.label)),
                edgecolor=scatter.cmap(scatter.norm(box.label)),
                alpha=0.16,
                linewidth=1.5,
            )
        )
        axis.text(
            box_center[0],
            box_center[1],
            str(box.label),
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
    axis.set_title('Cora Raw Features with Basic FMNN Hyperboxes')
    axis.text(
        0.01,
        0.01,
        'No GCN used; points are raw features and regions are approximate 2D projections',
        transform=axis.transAxes,
        fontsize=9,
        bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'},
    )
    axis.grid(alpha=0.2)
    fig.colorbar(scatter, ax=axis, label='Cora topic label')
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(json.dumps({'output': str(args.output), 'hyperboxes': len(boxes)}))


if __name__ == '__main__':
    main()
