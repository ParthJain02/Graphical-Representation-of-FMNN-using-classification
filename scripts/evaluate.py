from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate experiment outputs.')
    parser.add_argument('--results-dir', default='results')
    args = parser.parse_args()
    root = Path(args.results_dir)
    files = sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())
    summary = {'results_dir': str(root), 'file_count': len(files), 'files': files[:10]}

    summary_path = root / 'tables' / 'ablation_summary.json'
    if summary_path.exists():
        with summary_path.open('r', encoding='utf-8') as fh:
            payload = json.load(fh)
        experiments = payload.get('experiments', [])
        summary['ablation_count'] = len(experiments)
        variant_scores: dict[str, list[float]] = {}
        for experiment in experiments:
            for variant in experiment.get('variants', []):
                variant_scores.setdefault(variant['name'], []).append(float(variant['accuracy']))
        summary['mean_test_accuracy_by_variant'] = {
            name: sum(scores) / len(scores) for name, scores in sorted(variant_scores.items())
        }
        summary['test_accuracy_std_by_variant'] = {
            name: statistics.stdev(scores) if len(scores) > 1 else 0.0
            for name, scores in sorted(variant_scores.items())
        }
        summary['mean_test_macro_f1_by_variant'] = {
            name: sum(
                float(variant['macro_f1'])
                for experiment in experiments
                for variant in experiment.get('variants', [])
                if variant['name'] == name
            ) / len(scores)
            for name, scores in sorted(variant_scores.items())
        }
        baseline_mean = summary['mean_test_accuracy_by_variant'].get('baseline', 0.0)
        summary['test_accuracy_delta_vs_baseline'] = {
            name: score - baseline_mean
            for name, score in summary['mean_test_accuracy_by_variant'].items()
        }
        summary['best_variant'] = max(
            summary['mean_test_accuracy_by_variant'],
            key=summary['mean_test_accuracy_by_variant'].get,
        ) if variant_scores else None

    print(summary)


if __name__ == '__main__':
    main()
