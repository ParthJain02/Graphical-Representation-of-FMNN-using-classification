"""CLI wrapper to run GNN softmax (B1) and plain GNN+FMNN (B2) baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_utils import (
    execute_variants,
    get_seeds,
    get_tables_dir,
    load_cfg_and_data,
    summarize_runs,
)
from src.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN softmax and plain FMNN baselines.")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset config YAML")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional list of seeds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, data = load_cfg_and_data(args.config)
    seeds = get_seeds(cfg, args.seeds)
    variants = ["B1", "B2"]

    runs = execute_variants(cfg, data, seeds, variants)
    summary = summarize_runs(runs)

    out_path = get_tables_dir(cfg) / f"baseline_{data.name.lower()}.json"
    payload = {
        "dataset": data.name,
        "config_path": args.config,
        "seeds": seeds,
        "variants": list(summary.keys()),
        "runs": runs,
        "summary": summary,
    }
    save_json(payload, out_path)

    print(f"Saved baseline results to {out_path}")
    for var_name, stats in summary.items():
        acc = stats["test_accuracy"]
        f1 = stats["test_macro_f1"]
        boxes = stats["num_boxes"]
        print(
            f"  {var_name:<16} | Acc: {acc['mean']:.4f} +/- {acc['std']:.4f} | "
            f"Macro-F1: {f1['mean']:.4f} +/- {f1['std']:.4f} | Boxes: {boxes['mean']:.1f}"
        )


if __name__ == "__main__":
    main()
