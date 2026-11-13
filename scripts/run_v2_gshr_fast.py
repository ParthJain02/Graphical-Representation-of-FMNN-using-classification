"""Debug merge step and run B9_gshr_fast with per-setting validation and timing table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset
from src.gnn import apply_encoder_config
from src.merge import GraphContext, debug_merge_candidates
from src.pipeline import prepare_seed_context, run_single_seed
from src.utils import load_config, load_json, save_json


# Parse CLI flags so every fast-GSHR option can be configured or toggled off.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug merge step and evaluate B9_gshr_fast.")
    parser.add_argument("--config", type=str, default="configs/cora_v2.yaml", help="Path to YAML config")
    parser.add_argument(
        "--encoder-config",
        type=str,
        default="results/v2/encoder_config_cora.json",
        help="Optional saved encoder JSON",
    )
    parser.add_argument("--no-encoder-config", action="store_true", help="Disable saved encoder JSON")
    parser.add_argument("--l2-norm", dest="l2_norm", action="store_true", default=None)
    parser.add_argument("--no-l2-norm", dest="l2_norm", action="store_false")
    parser.add_argument("--no-merge", action="store_true", help="Disable merge inside B9_gshr_fast")
    parser.add_argument(
        "--no-require-overlap",
        action="store_true",
        help="Allow non-overlapping same-class pairs in fast merge prefilter",
    )
    parser.add_argument("--fixed-theta", type=float, default=None, help="Fix theta instead of grid search")
    parser.add_argument("--fixed-tau-prune", type=float, default=None, help="Fix tau_prune instead of grid search")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Evaluation seeds")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["B1", "B2", "B5", "B6", "B9_gshr_fast"],
        help="Variants to compare",
    )
    parser.add_argument("--tag", type=str, default="v2_enc", help="Output filename tag")
    parser.add_argument("--out-dir", type=str, default="results/v2", help="Output directory")
    return parser.parse_args()


# Build effective configuration applying optional encoder JSON and CLI switches.
def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    if not args.no_encoder_config and args.encoder_config:
        enc_path = Path(args.encoder_config)
        if enc_path.is_file():
            cfg = apply_encoder_config(cfg, load_json(enc_path))
    if args.l2_norm is not None:
        cfg.setdefault("embed", {})["l2_norm"] = args.l2_norm

    fast_cfg = cfg.setdefault("gshr_fast", {})
    if args.no_merge:
        fast_cfg["merge_enabled"] = False
    if args.no_require_overlap:
        fast_cfg["require_same_class_overlap"] = False
    if args.fixed_theta is not None:
        fast_cfg["theta"] = args.fixed_theta
    if args.fixed_tau_prune is not None:
        fast_cfg["tau_prune"] = args.fixed_tau_prune
    return cfg


# Print seed-0 merge debug stats: candidate count, safeguard pass counts, and score distribution.
def run_merge_debug_report(cfg: dict[str, Any], data: Any, seed: int = 0) -> dict[str, Any]:
    ctx = prepare_seed_context(cfg, data, seed)
    tr = data.train_idx.cpu().numpy()
    y_masked = np.full(data.num_nodes, -1, dtype=np.int64)
    y_masked[tr] = data.y[data.train_idx].cpu().numpy()
    train_mask = np.zeros(data.num_nodes, dtype=bool)
    train_mask[tr] = True
    graph_ctx = GraphContext(adj_csr=data.adj_csr, degree=data.degree, y=y_masked, train_mask=train_mask)

    m_cfg = dict(cfg.get("merge", {}))
    m_cfg["theta"] = cfg.get("fmnn", {}).get("theta", 0.3)
    m_cfg["safeguards"] = cfg.get("safeguards", {"class_purity": True, "graph_association": True})
    dbg = debug_merge_candidates(ctx.base_fmnn, graph_ctx, m_cfg)

    print("\n" + "=" * 78)
    print(f"MERGE STEP DEBUG REPORT (Dataset={data.name}, Seed={seed}, Base Boxes={dbg['num_boxes']})")
    print("=" * 78)
    print(f"  Total same-class pairs:  {dbg['total_same_class_pairs']}")
    print(f"  Candidate pairs checked: {dbg['candidate_pairs_count']}")
    pc = dbg["pass_counts"]
    print("  Safeguard pass counts:")
    print(f"    - same_class_geom_overlap : {pc['same_class_geom_overlap']} / {dbg['candidate_pairs_count']}")
    print(f"    - size_limit              : {pc['size_limit']} / {dbg['candidate_pairs_count']}")
    print(f"    - no_cross_class_overlap  : {pc['no_cross_class_overlap']} / {dbg['candidate_pairs_count']}")
    print(f"    - volume_inflation        : {pc['volume_inflation']} / {dbg['candidate_pairs_count']}")
    print(f"    - class_purity            : {pc['class_purity']} / {dbg['candidate_pairs_count']}")
    print(f"    - all_safeguards          : {pc['all_safeguards']} / {dbg['candidate_pairs_count']}")
    print(f"    - score >= tau_merge      : {pc['score_threshold']} / {dbg['candidate_pairs_count']}")
    sd = dbg["score_distribution"]
    if sd["count"] > 0:
        print(
            f"  Score M distribution (n={sd['count']}): "
            f"min={sd['min']:.4f}, p25={sd['p25']:.4f}, median={sd['median']:.4f}, "
            f"mean={sd['mean']:.4f}, p75={sd['p75']:.4f}, max={sd['max']:.4f}"
        )
    print("=" * 78)
    return dbg


# Aggregate B9 per-setting metrics across seeds and print the required settings table.
def build_and_print_settings_table(b9_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in b9_runs:
        for row in run.get("settings_table", []):
            grouped.setdefault(row["setting"], []).append(row)

    agg_rows: list[dict[str, Any]] = []
    for setting, group in grouped.items():
        agg_rows.append({
            "setting": setting,
            "theta": group[0]["theta"],
            "tau_prune": group[0]["tau_prune"],
            "safeguards": group[0]["safeguards"],
            "boxes_mean": float(np.mean([g["num_boxes"] for g in group])),
            "val_f1_mean": float(np.mean([g["val_macro_f1"] for g in group])),
            "refine_time_mean": float(np.mean([g["refine_time"] for g in group])),
            "merge_executed_mean": float(np.mean([g["merge_executed"] for g in group])),
        })

    print("\n" + "=" * 78)
    print("B9_gshr_fast Per-Setting Table (Mean over Seeds: Validation Only)")
    print(f"{'Setting':<28} | {'Boxes':<7} | {'Val F1':<8} | {'Refine Time (s)':<15} | {'Merges'}")
    print("-" * 78)
    for row in agg_rows:
        print(
            f"{row['setting']:<28} | {row['boxes_mean']:<7.1f} | {row['val_f1_mean']:<8.4f} | "
            f"{row['refine_time_mean']:<15.4f} | {row['merge_executed_mean']:.1f}"
        )
    print("=" * 78)
    return agg_rows


# Print variant comparison summary including accuracy, F1, boxes, chosen settings, and refine time.
def print_variant_comparison(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        grouped.setdefault(r["variant"], []).append(r)

    summary_rows: list[dict[str, Any]] = []
    print("\n" + "=" * 102)
    print(f"{'Variant':<20} | {'Test Acc':<17} | {'Test Macro-F1':<17} | {'Val F1':<8} | {'Boxes':<6} | {'RefineTime(s)':<14}")
    print("-" * 102)
    for var_name, group in grouped.items():
        acc = np.asarray([g["test_accuracy"] for g in group], dtype=np.float64)
        f1 = np.asarray([g["test_macro_f1"] for g in group], dtype=np.float64)
        vf1 = np.asarray([g["val_macro_f1"] for g in group], dtype=np.float64)
        bx = np.asarray([g["num_boxes"] for g in group], dtype=np.float64)
        rt = np.asarray([g["refine_time"] for g in group], dtype=np.float64)
        summary_rows.append({
            "variant": var_name,
            "num_seeds": len(group),
            "test_acc_mean": float(np.mean(acc)),
            "test_acc_std": float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0,
            "test_f1_mean": float(np.mean(f1)),
            "test_f1_std": float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0,
            "val_f1_mean": float(np.mean(vf1)),
            "boxes_mean": float(np.mean(bx)),
            "refine_time_mean": float(np.mean(rt)),
        })
        acc_str = f"{np.mean(acc):.4f} +/- {summary_rows[-1]['test_acc_std']:.4f}"
        f1_str = f"{np.mean(f1):.4f} +/- {summary_rows[-1]['test_f1_std']:.4f}"
        print(
            f"{var_name:<20} | {acc_str:<17} | {f1_str:<17} | "
            f"{np.mean(vf1):<8.4f} | {np.mean(bx):<6.1f} | {np.mean(rt):<14.4f}"
        )
    print("=" * 102)

    b9_runs = [r for r in runs if r["variant"] == "B9_gshr_fast"]
    if b9_runs:
        print("\nB9_gshr_fast Chosen Hyperparameters Per Seed (selected by val_macro_f1, ties->fewer boxes):")
        for r in b9_runs:
            print(
                f"  Seed {r['seed']}: {r['best_setting']} | "
                f"val_f1={r['val_macro_f1']:.4f} | boxes={r['num_boxes']} | "
                f"win_refine_time={r['best_setting_refine_time']:.4f}s | "
                f"total_grid_refine_time={r['refine_time']:.4f}s"
            )
    return summary_rows


# Main entry point running merge debug report and B9_gshr_fast evaluation saved under results/v2/.
def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    data_cfg = cfg.get("data", {})
    data = load_dataset(data_cfg.get("name", "Cora"), root=data_cfg.get("root", "./data"))

    debug_info = run_merge_debug_report(cfg, data, seed=args.seeds[0])

    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        ctx = prepare_seed_context(cfg, data, seed)
        for var in args.variants:
            res = run_single_seed(cfg, data, seed, var, seed_ctx=ctx)
            runs.append(res)

    b9_runs = [r for r in runs if r["variant"] == "B9_gshr_fast"]
    settings_summary = build_and_print_settings_table(b9_runs) if b9_runs else []
    variant_summary = print_variant_comparison(runs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_slug = data.name.lower()

    flat_runs = [{k: v for k, v in r.items() if k != "settings_table"} for r in runs]
    pd.DataFrame(flat_runs).to_csv(out_dir / f"b9_gshr_fast_{ds_slug}_{args.tag}.csv", index=False)
    if settings_summary:
        pd.DataFrame(settings_summary).to_csv(
            out_dir / f"b9_settings_table_{ds_slug}_{args.tag}.csv",
            index=False,
        )

    json_path = out_dir / f"b9_gshr_fast_{ds_slug}_{args.tag}.json"
    save_json(
        {
            "dataset": data.name,
            "tag": args.tag,
            "seeds": args.seeds,
            "config": cfg,
            "merge_debug_seed0": debug_info,
            "settings_summary": settings_summary,
            "variant_summary": variant_summary,
            "runs": runs,
        },
        json_path,
    )
    print(f"\nSaved B9_gshr_fast outputs under {out_dir}")


if __name__ == "__main__":
    main()
