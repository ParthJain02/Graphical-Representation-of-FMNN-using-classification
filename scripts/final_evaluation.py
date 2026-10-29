from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import ExperimentConfig, run_refinement_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate the frozen configuration on the held-out test split.')
    parser.add_argument('--tuning', default='results/tables/validation_tuning.json')
    parser.add_argument('--experiments', default='configs/experiments.yaml')
    parser.add_argument('--output', default='results/tables/final_evaluation.json')
    args = parser.parse_args()

    with open(args.tuning, 'r', encoding='utf-8') as fh:
        tuning = json.load(fh)
    with open(args.experiments, 'r', encoding='utf-8') as fh:
        experiment_config = yaml.safe_load(fh) or {}
    selected = tuning['selected_configuration']
    summaries = []
    for item in experiment_config.get('experiments', []):
        config = ExperimentConfig(
            seed=int(item.get('seed', 0)),
            dataset_name=str(item.get('dataset_name', 'Cora')),
            **selected,
        )
        summary = run_refinement_ablation(config)
        summaries.append(summary)

    variants = {}
    for summary in summaries:
        for variant in summary['variants']:
            variants.setdefault(variant['name'], []).append(variant)
    aggregate = {}
    for name, entries in variants.items():
        accuracies = [float(item['split_metrics']['test']['accuracy']) for item in entries]
        f1_scores = [float(item['split_metrics']['test']['macro_f1']) for item in entries]
        sample_count = len(entries)
        accuracy_std = statistics.stdev(accuracies) if sample_count > 1 else 0.0
        f1_std = statistics.stdev(f1_scores) if sample_count > 1 else 0.0
        t_critical = 2.776 if sample_count == 5 else 1.96
        aggregate[name] = {
            'mean_test_accuracy': sum(accuracies) / len(accuracies),
            'mean_test_macro_f1': sum(f1_scores) / len(f1_scores),
            'box_count_mean': sum(float(item['box_count']) for item in entries) / len(entries),
            'sample_count': sample_count,
            'test_accuracy_std': accuracy_std,
            'test_macro_f1_std': f1_std,
            'test_accuracy_95ci_half_width': t_critical * accuracy_std / math.sqrt(sample_count),
            'test_macro_f1_95ci_half_width': t_critical * f1_std / math.sqrt(sample_count),
        }
    baseline = aggregate['baseline']['mean_test_accuracy']
    for metrics in aggregate.values():
        metrics['test_accuracy_delta_vs_baseline'] = metrics['mean_test_accuracy'] - baseline
    payload = {
        'selection_source': args.tuning,
        'configuration': selected,
        'test_split_used_only_after_selection': True,
        'experiments': summaries,
        'aggregate': aggregate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print({'output': str(output), 'aggregate': aggregate})


if __name__ == '__main__':
    main()
