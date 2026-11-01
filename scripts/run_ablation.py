from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import ExperimentConfig, fit_baseline, run_refinement_ablation, run_softmax_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description='Run ablation variants for the research project.')
    parser.add_argument('--config', default='configs/experiments.yaml')
    parser.add_argument('--output', default='results/tables/ablation_summary.json')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as fh:
        data = yaml.safe_load(fh) or {}
    seed_experiments = data.get('experiments', [])
    variants = data.get('ablation_variants', [])
    if not variants:
        variants = [{'name': item.get('name', 'baseline'), 'mode': 'full'} for item in seed_experiments]
    out = Path('results/tables')
    out.mkdir(parents=True, exist_ok=True)

    results_by_variant = {variant['name']: [] for variant in variants}
    for seed_experiment in seed_experiments:
        fitted_by_parameters = {}
        for variant in variants:
            cfg = ExperimentConfig(
                seed=int(seed_experiment.get('seed', 0)),
                dataset_name=str(seed_experiment.get('dataset_name', variant.get('dataset_name', 'Cora'))),
                fmnn_theta=float(variant.get('fmnn_theta', 0.3)),
                fmnn_gamma=float(variant.get('fmnn_gamma', 1.0)),
                max_side_length=float(variant.get('max_side_length', 1.0)),
                volume_limit=float(variant.get('volume_limit', 1e6)),
                min_validation_purity=float(variant.get('min_validation_purity', 0.2)),
                min_objective_gain=float(variant.get('min_objective_gain', 0.0)),
                graph_enabled=bool(variant.get('graph_enabled', True)),
                class_purity_safeguards=bool(variant.get('class_purity_safeguards', True)),
            )
            mode = str(variant.get('mode', 'full'))
            if mode == 'softmax':
                result = run_softmax_experiment(cfg)
                selected_result = result
            else:
                fit_key = (cfg.seed, cfg.fmnn_theta, cfg.fmnn_gamma)
                if fit_key not in fitted_by_parameters:
                    fitted_by_parameters[fit_key] = fit_baseline(cfg)
                result = run_refinement_ablation(cfg, mode=mode, fitted=fitted_by_parameters[fit_key])
                selected_name = {'geometry_merge': 'geometry_merge', 'graph_merge': 'graph_merge', 'unsafe_full': 'unsafe_full'}.get(mode, mode)
                selected_result = next(item for item in result['variants'] if item['name'] == selected_name)
            results_by_variant[variant['name']].append({'seed': cfg.seed, 'result': result, 'selected_result': selected_result})
    summaries = [
        {'name': variant['name'], 'mode': variant.get('mode', 'full'), 'runs': results_by_variant[variant['name']]}
        for variant in variants
    ]

    payload = {'experiments': summaries}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    (out / 'ablation_variants.txt').write_text('\n'.join(v['name'] for v in variants), encoding='utf-8')
    print({'variant_count': len(variants), 'variants': [v['name'] for v in variants], 'summary_path': str(output_path)})


if __name__ == '__main__':
    main()
