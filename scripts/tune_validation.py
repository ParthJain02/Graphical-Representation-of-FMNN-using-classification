from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import ExperimentConfig, run_refinement_ablation


def _validation_score(variant: dict[str, object]) -> float:
    metrics = variant['split_metrics']['validation']
    return 0.5 * float(metrics['accuracy']) + 0.5 * float(metrics['macro_f1'])


def main() -> None:
    parser = argparse.ArgumentParser(description='Tune GSHR configuration using validation metrics only.')
    parser.add_argument('--config', default='configs/experiments.yaml')
    parser.add_argument('--output', default='results/tables/validation_tuning.json')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as fh:
        experiment_config = yaml.safe_load(fh) or {}
    seeds = [int(item.get('seed', 0)) for item in experiment_config.get('experiments', [])]
    if not seeds:
        seeds = [0]

    candidates = [
        {'fmnn_theta': 0.1, 'fmnn_gamma': 1.0, 'max_side_length': 0.75},
        {'fmnn_theta': 0.3, 'fmnn_gamma': 1.0, 'max_side_length': 1.0},
        {'fmnn_theta': 0.5, 'fmnn_gamma': 1.0, 'max_side_length': 1.0},
        {'fmnn_theta': 0.3, 'fmnn_gamma': 2.0, 'max_side_length': 1.0},
    ]
    results = []
    for candidate in candidates:
        runs = []
        for seed in seeds:
            config = ExperimentConfig(seed=seed, dataset_name='Cora', **candidate)
            summary = run_refinement_ablation(config)
            merge = next(item for item in summary['variants'] if item['name'] == 'merge')
            runs.append({
                'seed': seed,
                'validation_score': _validation_score(merge),
                'validation_accuracy': merge['split_metrics']['validation']['accuracy'],
                'validation_macro_f1': merge['split_metrics']['validation']['macro_f1'],
                'box_count': merge['box_count'],
            })
        results.append({
            'configuration': candidate,
            'mean_validation_score': sum(item['validation_score'] for item in runs) / len(runs),
            'runs': runs,
        })

    selected = max(results, key=lambda item: item['mean_validation_score'])
    payload = {
        'selection_metric': 'mean(0.5 * validation_accuracy + 0.5 * validation_macro_f1)',
        'test_metrics_used_for_selection': False,
        'selected_configuration': selected['configuration'],
        'candidates': results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print({
        'selected_configuration': selected['configuration'],
        'mean_validation_score': selected['mean_validation_score'],
        'output': str(output),
    })


if __name__ == '__main__':
    main()
