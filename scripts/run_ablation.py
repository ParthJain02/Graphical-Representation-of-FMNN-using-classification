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


def main() -> None:
    parser = argparse.ArgumentParser(description='Run ablation variants for the research project.')
    parser.add_argument('--config', default='configs/experiments.yaml')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as fh:
        data = yaml.safe_load(fh) or {}
    variants = data.get('experiments', [])
    out = Path('results/tables')
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for variant in variants:
        cfg = ExperimentConfig(
            seed=int(variant.get('seed', 0)),
            dataset_name=str(variant.get('dataset_name', 'Cora')),
            fmnn_theta=float(variant.get('fmnn_theta', 0.3)),
            fmnn_gamma=float(variant.get('fmnn_gamma', 1.0)),
            max_side_length=float(variant.get('max_side_length', 1.0)),
            volume_limit=float(variant.get('volume_limit', 1e6)),
            min_validation_purity=float(variant.get('min_validation_purity', 0.2)),
            min_objective_gain=float(variant.get('min_objective_gain', 0.0)),
        )
        summary = run_refinement_ablation(cfg)
        summaries.append(summary)

    payload = {'experiments': summaries}
    (out / 'ablation_summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    (out / 'ablation_variants.txt').write_text('\n'.join(v['name'] for v in variants), encoding='utf-8')
    print({'variant_count': len(variants), 'variants': [v['name'] for v in variants], 'summary_path': str(out / 'ablation_summary.json')})


if __name__ == '__main__':
    main()
