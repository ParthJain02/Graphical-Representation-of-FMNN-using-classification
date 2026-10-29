from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import ExperimentConfig, run_baseline_experiment
from src.reporting import write_summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Train the synthetic baseline GCN + FMNN pipeline.')
    parser.add_argument('--config', default='configs/cora.yaml')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh) or {}
    experiment = ExperimentConfig(
        seed=int(cfg.get('seed', 0)),
        dataset_name=str(cfg.get('dataset', {}).get('name', 'synthetic-cora')),
        num_nodes=int(cfg.get('dataset', {}).get('num_nodes', 20)),
        num_features=int(cfg.get('dataset', {}).get('num_features', 8)),
        num_classes=int(cfg.get('dataset', {}).get('num_classes', 4)),
        hidden_dim=int(cfg.get('model', {}).get('hidden_dim', 64)),
    )
    result = run_baseline_experiment(experiment)
    out_dir = Path('results/raw')
    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(str(out_dir / 'baseline_summary.txt'), result)
    print(result)


if __name__ == '__main__':
    main()
