from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import ExperimentConfig, fit_baseline, summarize_baseline
from src.reporting import write_summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Train the configured GCN + FMNN baseline.')
    parser.add_argument('--config', default='configs/cora.yaml')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh) or {}
    experiment = ExperimentConfig(
        seed=int(cfg.get('seed', 0)),
        dataset_name=str(cfg.get('dataset', {}).get('name', 'synthetic-cora')),
        data_root=str(cfg.get('dataset', {}).get('root', 'data')),
        num_nodes=int(cfg.get('dataset', {}).get('num_nodes', 20)),
        num_features=int(cfg.get('dataset', {}).get('num_features', 8)),
        num_classes=int(cfg.get('dataset', {}).get('num_classes', 4)),
        hidden_dim=int(cfg.get('model', {}).get('hidden_dim', 64)),
        dropout=float(cfg.get('model', {}).get('dropout', 0.5)),
        learning_rate=float(cfg.get('model', {}).get('learning_rate', 0.01)),
        weight_decay=float(cfg.get('model', {}).get('weight_decay', 0.0005)),
        max_epochs=int(cfg.get('model', {}).get('max_epochs', 200)),
        patience=int(cfg.get('model', {}).get('patience', 30)),
        fmnn_theta=float(cfg.get('fmnn', {}).get('theta', 0.3)),
        fmnn_gamma=float(cfg.get('fmnn', {}).get('gamma', 1.0)),
    )
    dataset, model, embeddings, fmnn, normalizer = fit_baseline(experiment)
    result = summarize_baseline(dataset, model, embeddings, fmnn, normalizer)
    run_id = datetime.now(timezone.utc).strftime(f'baseline_seed_{experiment.seed}_%Y%m%dT%H%M%SZ')
    out_dir = Path('results/processed') / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), out_dir / 'gcn_state.pt')
    np.savez_compressed(
        out_dir / 'embeddings_and_splits.npz',
        embeddings=embeddings,
        labels=dataset.y,
        edge_index=dataset.edge_index,
        train_mask=dataset.train_mask,
        val_mask=dataset.val_mask,
        test_mask=dataset.test_mask,
    )
    fmnn.save(str(out_dir / 'fmnn_boxes.json'))
    with (out_dir / 'config.yaml').open('w', encoding='utf-8') as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    with (out_dir / 'normalizer.json').open('w', encoding='utf-8') as fh:
        json.dump(result['normalization'], fh, indent=2)
    with (out_dir / 'summary.json').open('w', encoding='utf-8') as fh:
        json.dump(result, fh, indent=2)
    write_summary(str(out_dir / 'summary.txt'), result)
    print(f'Artifacts written to {out_dir}')
    print(result)


if __name__ == '__main__':
    main()
