"""CLI wrapper to run GSHR graph-aware merge, prune, or full refinement."""

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

MODE_TO_VARIANT = {
    "merge": "B4",
    "prune": "B5",
    "full": "B6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GSHR hyperbox refinement pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset config YAML")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["merge", "prune", "full"],
        default="full",
        help="Refinement mode: merge only, prune only, or full (merge + prune).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional list of seeds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, data = load_cfg_and_data(args.config)
    seeds = get_seeds(cfg, args.seeds)
    variant = MODE_TO_VARIANT[args.mode]

    runs = execute_variants(cfg, data, seeds, [variant])
    summary = summarize_runs(runs)
    variant_name = next(iter(summary.keys()))

    out_path = get_tables_dir(cfg) / f"gshr_{data.name.lower()}_{args.mode}.json"
    payload = {
        "dataset": data.name,
        "mode": args.mode,
        "variant": variant_name,
        "config_path": args.config,
        "seeds": seeds,
        "runs": runs,
        "summary": summary,
    }
    save_json(payload, out_path)

    stats = summary[variant_name]
    acc = stats["test_accuracy"]
    f1 = stats["test_macro_f1"]
    boxes = stats["num_boxes"]
    print(f"Saved GSHR ({args.mode}) results to {out_path}")
    print(
        f"  {variant_name:<16} | Acc: {acc['mean']:.4f} +/- {acc['std']:.4f} | "
        f"Macro-F1: {f1['mean']:.4f} +/- {f1['std']:.4f} | Boxes: {boxes['mean']:.1f}"
    )


if __name__ == "__main__":
    main()
