"""Validation-based GNN encoder search and B1/B2 v2 benchmark (saves to results/v2/)."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_utils import summarize_runs
from src.data import load_dataset
from src.gnn import apply_encoder_config, search_encoder_hyperparams
from src.pipeline import prepare_seed_context, run_single_seed
from src.utils import load_config, save_json


# Parse command-line options for encoder search and B1/B2 re-evaluation.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search GNN encoder hyperparameters and evaluate B1/B2.")
    parser.add_argument("--config", type=str, default="configs/cora_v2.yaml", help="Path to YAML config")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Evaluation seeds")
    parser.add_argument("--search-seeds", nargs="+", type=int, default=None, help="Validation search seeds")
    parser.add_argument("--early-stop-metric", type=str, default=None, choices=["val_loss", "val_macro_f1"])
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--max-epochs", type=int, default=None, help="Max GNN training epochs")
    parser.add_argument("--l2-norm", dest="l2_norm", action="store_true", default=None, help="Enable L2 normalization")
    parser.add_argument("--no-l2-norm", dest="l2_norm", action="store_false", help="Disable L2 normalization")
    parser.add_argument("--skip-search", action="store_true", help="Skip hyperparameter search and use config defaults")
    parser.add_argument("--out-dir", type=str, default="results/v2", help="Output directory for v2 artifacts")
    return parser.parse_args()


# Build effective configuration dictionary with CLI overrides applied.
def build_effective_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    if args.early_stop_metric is not None:
        cfg.setdefault("gnn", {})["early_stop_metric"] = args.early_stop_metric
    if args.patience is not None:
        cfg.setdefault("gnn", {})["patience"] = args.patience
    if args.max_epochs is not None:
        cfg.setdefault("gnn", {})["max_epochs"] = args.max_epochs
    if args.l2_norm is not None:
        cfg.setdefault("embed", {})["l2_norm"] = args.l2_norm
    return cfg


# Load old baseline reference rows from results/<dataset>_summary.csv without modifying it.
def load_old_baseline_rows(dataset_name: str) -> list[dict[str, Any]]:
    old_summary = Path("results") / f"{dataset_name.lower()}_summary.csv"
    if not old_summary.is_file():
        return []
    df = pd.read_csv(old_summary)
    rows: list[dict[str, Any]] = []
    for variant_id, label in [("B1_gnn_softmax", "Old B1 (gnn_softmax)"), ("B2_gnn_fmnn", "Old B2 (gnn_fmnn)")]:
        sub = df[df["variant"] == variant_id]
        if not sub.empty:
            r = sub.iloc[0]
            rows.append({
                "label": label,
                "acc_mean": float(r["test_accuracy_mean"]),
                "acc_std": float(r["test_accuracy_std"]),
                "f1_mean": float(r["test_macro_f1_mean"]),
                "f1_std": float(r["test_macro_f1_std"]),
                "boxes_mean": float(r["num_boxes_mean"]),
                "boxes_std": float(r["num_boxes_std"]),
            })
    return rows


# Evaluate B1 and B2 under a target encoder config while optionally testing L2-norm on/off.
def evaluate_v2_b1_b2(
    cfg: dict[str, Any],
    data: Any,
    seeds: list[int],
) -> list[dict[str, Any]]:
    all_runs: list[dict[str, Any]] = []
    use_l2 = bool(cfg.get("embed", {}).get("l2_norm", True))

    for seed in seeds:
        ctx_main = prepare_seed_context(cfg, data, seed)
        r_b1 = run_single_seed(cfg, data, seed, "B1", seed_ctx=ctx_main)
        r_b1["label"] = "New B1 (v2 encoder)"
        all_runs.append(r_b1)

        r_b2 = run_single_seed(cfg, data, seed, "B2", seed_ctx=ctx_main)
        r_b2["label"] = "New B2 (v2 enc + L2-minmax)" if use_l2 else "New B2 (v2 enc + minmax)"
        all_runs.append(r_b2)

        if use_l2:
            cfg_no_l2 = copy.deepcopy(cfg)
            cfg_no_l2.setdefault("embed", {})["l2_norm"] = False
            ctx_no_l2 = prepare_seed_context(cfg_no_l2, data, seed)
            r_b2_raw = run_single_seed(cfg_no_l2, data, seed, "B2", seed_ctx=ctx_no_l2)
            r_b2_raw["label"] = "New B2 (v2 enc, no L2)"
            all_runs.append(r_b2_raw)

    return all_runs


# Aggregate v2 runs by display label using mean and sample std across seeds.
def aggregate_by_label(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        grouped.setdefault(row["label"], []).append(row)

    rows: list[dict[str, Any]] = []
    for label, group in grouped.items():
        acc = np.asarray([r["test_accuracy"] for r in group], dtype=np.float64)
        f1 = np.asarray([r["test_macro_f1"] for r in group], dtype=np.float64)
        bx = np.asarray([r["num_boxes"] for r in group], dtype=np.float64)
        rows.append({
            "label": label,
            "acc_mean": float(np.mean(acc)),
            "acc_std": float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0,
            "f1_mean": float(np.mean(f1)),
            "f1_std": float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0,
            "boxes_mean": float(np.mean(bx)),
            "boxes_std": float(np.std(bx, ddof=1)) if len(bx) > 1 else 0.0,
        })
    return rows


# Print formatted before/after comparison table for old vs new B1 and B2 results.
def print_comparison_table(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 86)
    print(f"{'Model / Setting':<30} | {'Test Accuracy':<19} | {'Test Macro-F1':<19} | {'Boxes':<8}")
    print("-" * 86)
    for row in old_rows + new_rows:
        acc_str = f"{row['acc_mean']:.4f} +/- {row['acc_std']:.4f}"
        f1_str = f"{row['f1_mean']:.4f} +/- {row['f1_std']:.4f}"
        bx_str = f"{row['boxes_mean']:.1f} +/- {row['boxes_std']:.1f}"
        print(f"{row['label']:<30} | {acc_str:<19} | {f1_str:<19} | {bx_str:<8}")
    print("=" * 86)


# Main entry point running validation encoder search and B1/B2 v2 evaluation.
def main() -> None:
    args = parse_args()
    cfg = build_effective_config(args)

    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("name", "Cora")
    data = load_dataset(dataset_name, root=data_cfg.get("root", "./data"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_slug = data.name.lower()

    search_cfg = cfg.get("gnn_search", {})
    run_search = bool(search_cfg.get("enabled", True)) and not args.skip_search

    if run_search:
        search_seeds = args.search_seeds if args.search_seeds is not None else search_cfg.get("search_seeds", [0, 1, 2])
        gnn_cfg = cfg.get("gnn", {})
        search_result = search_encoder_hyperparams(
            data=data,
            cfg=cfg,
            search_seeds=search_seeds,
            hidden_grid=search_cfg.get("hidden", [64, 128]),
            dropout_grid=search_cfg.get("dropout", [0.5, 0.6]),
            weight_decay_grid=search_cfg.get("weight_decay", [5e-4, 1e-3]),
            lr_grid=search_cfg.get("lr", [0.01, 0.005]),
            early_stop_metric=str(gnn_cfg.get("early_stop_metric", "val_loss")),
            patience=int(gnn_cfg.get("patience", 50)),
            max_epochs=int(gnn_cfg.get("max_epochs", 300)),
        )
        cfg_v2 = apply_encoder_config(cfg, search_result)
        enc_json_path = out_dir / f"encoder_config_{ds_slug}.json"
        save_json(search_result, enc_json_path)
        bc = search_result["best_config"]
        print(f"Saved chosen encoder config to {enc_json_path}")
        print(
            f"  Best (val_macro_f1={search_result['best_val_macro_f1_mean']:.4f}, "
            f"val_loss={search_result['best_val_loss_mean']:.4f}): "
            f"hidden={bc['hidden']}, dropout={bc['dropout']}, weight_decay={bc['weight_decay']}, "
            f"lr={bc['lr']}, early_stop={bc['early_stop_metric']}"
        )
    else:
        cfg_v2 = copy.deepcopy(cfg)
        enc_json_path = out_dir / f"encoder_config_{ds_slug}.json"
        save_json({"dataset": data.name, "skipped_search": True, "best_config": cfg_v2["gnn"]}, enc_json_path)
        print(f"Search skipped; saved active config to {enc_json_path}")

    runs = evaluate_v2_b1_b2(cfg_v2, data, args.seeds)
    old_rows = load_old_baseline_rows(data.name)
    new_rows = aggregate_by_label(runs)

    print_comparison_table(old_rows, new_rows)

    runs_csv_path = out_dir / f"encoder_b1_b2_{ds_slug}.csv"
    pd.DataFrame(runs).to_csv(runs_csv_path, index=False)

    summary_json_path = out_dir / f"encoder_b1_b2_{ds_slug}.json"
    save_json(
        {
            "dataset": data.name,
            "seeds": args.seeds,
            "encoder_config": cfg_v2["gnn"],
            "embed_config": cfg_v2.get("embed", {}),
            "old_baseline_summary": old_rows,
            "new_summary": new_rows,
            "variant_summary": summarize_runs(runs),
            "runs": runs,
        },
        summary_json_path,
    )
    print(f"\nSaved v2 B1/B2 results to {summary_json_path} and {runs_csv_path}")


if __name__ == "__main__":
    main()
