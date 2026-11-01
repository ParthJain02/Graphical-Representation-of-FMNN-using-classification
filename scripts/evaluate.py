from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate ablation outputs and generate research figures.')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--ablation', default='results/tables/ablation_summary_corrected.json')
    args = parser.parse_args()
    root = Path(args.results_dir)
    with Path(args.ablation).open('r', encoding='utf-8') as fh:
        payload = json.load(fh)
    rows = []
    for experiment in payload.get('experiments', []):
        for run in experiment.get('runs', []):
            result = run.get('selected_result', run.get('result', {}))
            if experiment.get('mode') == 'softmax':
                rows.append({
                    'variant': experiment['name'],
                    'seed': run['seed'],
                    'accuracy': float(result['accuracy']),
                    'macro_f1': float(result['macro_f1']),
                    'box_count': int(result.get('num_hyperboxes', 0)),
                })
            else:
                rows.append({
                    'variant': experiment['name'],
                    'seed': run['seed'],
                    'accuracy': float(result['accuracy']),
                    'macro_f1': float(result['macro_f1']),
                    'box_count': int(result['box_count']),
                })

    aggregates = {}
    for variant in sorted({row['variant'] for row in rows}):
        entries = [row for row in rows if row['variant'] == variant]
        accuracies = [row['accuracy'] for row in entries]
        macro_f1 = [row['macro_f1'] for row in entries]
        boxes = [row['box_count'] for row in entries]
        aggregates[variant] = {
            'sample_count': len(entries),
            'mean_accuracy': sum(accuracies) / len(accuracies),
            'mean_macro_f1': sum(macro_f1) / len(macro_f1),
            'mean_box_count': sum(boxes) / len(boxes),
            'accuracy_std': statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            'macro_f1_std': statistics.stdev(macro_f1) if len(macro_f1) > 1 else 0.0,
        }
    baseline = aggregates.get('E2', aggregates.get('baseline'))
    if baseline is not None:
        for metrics in aggregates.values():
            metrics['accuracy_delta_vs_fmnn_baseline'] = metrics['mean_accuracy'] - baseline['mean_accuracy']
            metrics['macro_f1_delta_vs_fmnn_baseline'] = metrics['mean_macro_f1'] - baseline['mean_macro_f1']

    tables = root / 'tables'
    figures = root / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    summary = {'source': args.ablation, 'rows': rows, 'aggregates': aggregates}
    (tables / 'evaluation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    labels = list(aggregates)
    plt.figure(figsize=(9, 5))
    plt.bar(labels, [aggregates[label]['mean_macro_f1'] for label in labels])
    plt.ylabel('Mean test Macro-F1')
    plt.title('E1-E8 test Macro-F1')
    plt.tight_layout()
    plt.savefig(figures / 'ablation_macro_f1.png', dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, [aggregates[label]['mean_box_count'] for label in labels])
    plt.ylabel('Mean hyperbox count')
    plt.title('E1-E8 classifier compactness')
    plt.tight_layout()
    plt.savefig(figures / 'ablation_box_count.png', dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(
        [aggregates[label]['mean_box_count'] for label in labels],
        [aggregates[label]['mean_macro_f1'] for label in labels],
    )
    for label in labels:
        plt.annotate(label, (aggregates[label]['mean_box_count'], aggregates[label]['mean_macro_f1']))
    plt.xlabel('Mean hyperbox count')
    plt.ylabel('Mean test Macro-F1')
    plt.title('Predictive quality versus compactness')
    plt.tight_layout()
    plt.savefig(figures / 'compactness_vs_macro_f1.png', dpi=160)
    plt.close()
    print({'summary': str(tables / 'evaluation_summary.json'), 'figures': str(figures), 'variants': labels})


if __name__ == '__main__':
    main()
