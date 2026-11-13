"""Main experiment runner across seeds and model variants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from src.data import load_dataset
from src.pipeline import prepare_seed_context, run_single_seed
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GSHR node classification benchmark.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default="Cora", help="Dataset name (e.g. Cora, CiteSeer, PubMed)")
    parser.add_argument("--variants", nargs="+", default=["B1", "B2", "B3", "B4", "B5", "B6"], help="Variants to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="List of random seeds")
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset_name = args.dataset
    data_root = cfg.get("data", {}).get("root", "./data")
    data = load_dataset(dataset_name, root=data_root)

    seeds = args.seeds
    if seeds is None:
        seeds = cfg.get("run", {}).get("seeds", [0, 1, 2, 3, 4])

    variants = args.variants
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"GSHR Main Experiment: Dataset={dataset_name}, Seeds={seeds}, Variants={variants}")
    print("=" * 70)

    all_runs: list[dict] = []

    for seed in seeds:
        print(f"\n>>> Running Seed {seed}...")
        # Precompute and reuse GNN and base FMNN for fair comparison and efficiency
        seed_ctx = prepare_seed_context(cfg, data, seed)

        for var in variants:
            res = run_single_seed(cfg, data, seed, var, seed_ctx=seed_ctx)
            all_runs.append(res)
            print(f"  [{var:<4}] Test Acc: {res['test_accuracy']:.4f} | Test F1: {res['test_macro_f1']:.4f} | Boxes: {res['num_boxes']:3d} | Refine: {res['refine_time']:.4f}s")

    df_runs = pd.DataFrame(all_runs)
    runs_csv_path = out_dir / f"{dataset_name.lower()}_runs.csv"
    df_runs.to_csv(runs_csv_path, index=False)
    print(f"\nSaved raw run records to {runs_csv_path}")

    # Compute summary (mean and std across seeds per variant)
    summary_rows = []
    numeric_cols = [
        "test_accuracy", "test_macro_f1", "test_micro_f1",
        "val_accuracy", "val_macro_f1",
        "num_boxes", "mean_box_size",
        "gnn_train_time", "fmnn_fit_time", "refine_time",
        "pred_latency_per_node", "peak_memory_mb"
    ]

    for variant_name, group in df_runs.groupby("variant", sort=False):
        row = {"variant": variant_name, "num_seeds": len(group)}
        for col in numeric_cols:
            if col in group:
                row[f"{col}_mean"] = group[col].mean()
                row[f"{col}_std"] = group[col].std()
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = out_dir / f"{dataset_name.lower()}_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved aggregated summary to {summary_csv_path}\n")

    # Print readable summary table
    print("=" * 85)
    print(f"{'Variant':<22} | {'Test Acc (mean±std)':<22} | {'Test Macro-F1':<22} | {'Boxes':<7} | {'Refine(s)':<8}")
    print("-" * 85)
    for _, row in df_summary.iterrows():
        acc_str = f"{row['test_accuracy_mean']:.4f} ± {row['test_accuracy_std']:.4f}" if pd.notnull(row.get('test_accuracy_std')) else f"{row['test_accuracy_mean']:.4f}"
        f1_str = f"{row['test_macro_f1_mean']:.4f} ± {row['test_macro_f1_std']:.4f}" if pd.notnull(row.get('test_macro_f1_std')) else f"{row['test_macro_f1_mean']:.4f}"
        boxes_str = f"{row['num_boxes_mean']:.1f}"
        refine_str = f"{row['refine_time_mean']:.3f}"
        print(f"{row['variant']:<22} | {acc_str:<22} | {f1_str:<22} | {boxes_str:<7} | {refine_str:<8}")
    print("=" * 85)


if __name__ == "__main__":
    main()
