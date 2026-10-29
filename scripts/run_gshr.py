from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gshr.hyperbox import HyperboxRecord
from src.gshr.refinement import GraphStructureAwareRefinement


def main() -> None:
    parser = argparse.ArgumentParser(description='Run GSHR refinement in merge or prune mode.')
    parser.add_argument('--config', default='configs/cora.yaml')
    parser.add_argument('--mode', choices=['merge', 'prune', 'full'], default='full')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh) or {}
    boxes = [
        HyperboxRecord(box_id=1, lower=[0.0, 0.0], upper=[1.0, 1.0], label=0, members={1, 2}),
        HyperboxRecord(box_id=2, lower=[0.2, 0.2], upper=[0.8, 0.8], label=0, members={3, 4}),
        HyperboxRecord(box_id=3, lower=[2.0, 2.0], upper=[3.0, 3.0], label=1, members={5, 6}),
    ]
    refinement = GraphStructureAwareRefinement(boxes)
    if args.mode in {'merge', 'full'}:
        boxes = refinement.merge(max_side_length=cfg.get('refinement', {}).get('max_side_length', 1.0), volume_limit=float(cfg.get('refinement', {}).get('volume_limit', 1e6)))
    if args.mode in {'prune', 'full'}:
        boxes = refinement.prune(cfg.get('refinement', {}).get('prune_rule', {'max_loss_increase': 0.1}))
    out = Path('results/processed')
    out.mkdir(parents=True, exist_ok=True)
    payload = [
        {'box_id': box.box_id, 'label': box.label, 'members': sorted(box.members), 'status': box.status}
        for box in boxes
    ]
    with (out / 'gshr_run.json').open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
    print({'mode': args.mode, 'box_count': len(boxes)})


if __name__ == '__main__':
    main()
