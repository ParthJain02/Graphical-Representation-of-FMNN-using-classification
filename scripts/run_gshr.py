from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import ExperimentConfig, run_refinement_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description='Run GSHR refinement in merge or prune mode.')
    parser.add_argument('--config', default='configs/cora.yaml')
    parser.add_argument('--mode', choices=['merge', 'prune', 'full'], default='full')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh) or {}
    dataset_cfg = cfg.get('dataset', {})
    model_cfg = cfg.get('model', {})
    fmnn_cfg = cfg.get('fmnn', {})
    refinement_cfg = cfg.get('refinement', {})
    experiment = ExperimentConfig(
        seed=int(cfg.get('seed', 0)),
        dataset_name=str(dataset_cfg.get('name', 'synthetic-cora')),
        data_root=str(dataset_cfg.get('root', 'data')),
        num_nodes=int(dataset_cfg.get('num_nodes', 20)),
        num_features=int(dataset_cfg.get('num_features', 8)),
        num_classes=int(dataset_cfg.get('num_classes', 4)),
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
    result = run_refinement_ablation(experiment)
    selected = next(item for item in result['variants'] if item['name'] == args.mode)
    payload = {
        'mode': args.mode,
        'seed': experiment.seed,
        'dataset': result['dataset'],
        'baseline': {
            'box_count': result['baseline_num_hyperboxes'],
            'accuracy': result['baseline_accuracy'],
            'macro_f1': result['baseline_macro_f1'],
        },
        'variants': result['variants'],
        'selected_variant': selected,
    }
    out = Path('results/processed')
    out.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime(f'gshr_{args.mode}_seed_{experiment.seed}_%Y%m%dT%H%M%SZ')
    output_path = out / f'{run_id}.json'
    with output_path.open('x', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
    print({'mode': args.mode, 'box_count': selected['box_count'], 'output': str(output_path)})


if __name__ == '__main__':
    main()
