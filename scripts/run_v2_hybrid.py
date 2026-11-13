"""Runner script for B7_hybrid and B8_hybrid_pseudo comparison saved to results/v2/."""

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
from src.pipeline import prepare_seed_context, run_single_seed
from src.utils import load_config, load_json, save_json


# Parse CLI flags so every option can be configured or switched off.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B7_hybrid and B8_hybrid_pseudo alongside B1, B2, B6.")
    parser.add_argument("--config", type=str, default="configs/cora_v2.yaml", help="Path to YAML config")
    parser.add_argument(
        "--encoder-config",
        type=str,
        default="results/v2/encoder_config_cora.json",
        help="Optional saved encoder search JSON to apply",
    )
    parser.add_argument("--no-encoder-config", action="store_true", help="Disable loading encoder JSON")
    parser.add_argument("--l2-norm", dest="l2_norm", action="store_true", default=None)
    parser.add_argument("--no-l2-norm", dest="l2_norm", action="store_false")
    parser.add_argument("--fixed-lam", type=float, default=None, help="Fix lambda instead of val search")
    parser.add_argument("--no-sanity-check", action="store_true", help="Disable strict lam=0/1 check")
    parser.add_argument("--no-pseudo", action="store_true", help="Disable pseudo-labeling for B8")
    parser.add_argument("--fixed-c", type=float, default=None, help="Fix confidence threshold c for B8")
    parser.add_argument("--exclude-val", action="store_true", help="Exclude val nodes from pseudo-label pool")
    parser.add_argument("--exclude-test", action="store_true", help="Exclude test nodes from pseudo-label pool")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Seeds to evaluate")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["B1", "B2", "B6", "B7_hybrid", "B8_hybrid_pseudo"],
        help="Variants to execute",
    )
    parser.add_argument("--tag", type=str, default="v2_enc", help="Output filename tag")
    parser.add_argument("--out-dir", type=str, default="results/v2", help="Output directory")
    return parser.parse_args()


# Build effective configuration applying optional saved encoder config and CLI flags.
def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    if not args.no_encoder_config and args.encoder_config:
        enc_path = Path(args.encoder_config)
        if enc_path.is_file():
            cfg = apply_encoder_config(cfg, load_json(enc_path))
    if args.l2_norm is not None:
        cfg.setdefault("embed", {})["l2_norm"] = args.l2_norm
    cfg.setdefault("hybrid", {})
    if args.fixed_lam is not None:
        cfg["hybrid"]["lambda"] = args.fixed_lam
    if args.no_sanity_check:
        cfg["hybrid"]["sanity_check"] = False
    cfg.setdefault("pseudo", {})
    if args.no_pseudo:
        cfg["pseudo"]["enabled"] = False
    if args.fixed_c is not None:
        cfg["pseudo"]["c"] = args.fixed_c
    if args.exclude_val:
        cfg["pseudo"]["exclude_val"] = True
    if args.exclude_test:
        cfg["pseudo"]["exclude_test"] = True
    return cfg


# Summarize per-variant test metrics, boxes, chosen lambda/c, pseudo stats, and coverage across seeds.
def build_summary_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        grouped.setdefault(row["variant"], []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for var_name, group in grouped.items():
        acc = np.asarray([r["test_accuracy"] for r in group], dtype=np.float64)
        f1 = np.asarray([r["test_macro_f1"] for r in group], dtype=np.float64)
        bx = np.asarray([r["num_boxes"] for r in group], dtype=np.float64)
        sz = np.asarray([r["mean_box_size"] for r in group], dtype=np.float64)
        ov = np.asarray([r["cross_class_overlap"] for r in group], dtype=np.float64)
        lams = [r["chosen_lam"] for r in group if r.get("chosen_lam") is not None]
        cs = [r["chosen_c"] for r in group if r.get("chosen_c") is not None]
        pseudos = [r["pseudo_count"] for r in group if r.get("pseudo_count") is not None]
        covs = [r["box_coverage"] for r in group if r.get("box_coverage") is not None]
        cov_accs = [r["covered_accuracy"] for r in group if r.get("covered_accuracy") is not None]
        uncov_accs = [r["uncovered_accuracy"] for r in group if r.get("uncovered_accuracy") is not None]

        summary_rows.append({
            "variant": var_name,
            "num_seeds": len(group),
            "test_acc_mean": float(np.mean(acc)),
            "test_acc_std": float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0,
            "test_f1_mean": float(np.mean(f1)),
            "test_f1_std": float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0,
            "boxes_mean": float(np.mean(bx)),
            "mean_box_size_mean": float(np.mean(sz)),
            "overlap_mean": float(np.mean(ov)),
            "chosen_lams": lams,
            "chosen_lam_mean": float(np.mean(lams)) if lams else None,
            "chosen_cs": cs,
            "chosen_c_mean": float(np.mean(cs)) if cs else None,
            "pseudo_counts": pseudos,
            "pseudo_count_mean": float(np.mean(pseudos)) if pseudos else None,
            "box_coverage_mean": float(np.mean(covs)) if covs else None,
            "covered_acc_mean": float(np.mean(cov_accs)) if cov_accs else None,
            "uncovered_acc_mean": float(np.mean(uncov_accs)) if uncov_accs else None,
        })
    return summary_rows


# Print summary table next to B1, B2, B6, B7 and log B8 pseudo-label, size, and overlap deltas.
def print_hybrid_table(summary_rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 112)
    print(
        f"{'Variant':<18} | {'Test Acc':<17} | {'Test Macro-F1':<17} | "
        f"{'Boxes':<6} | {'MeanSize':<8} | {'Chosen lam':<18} | {'c / Pseudo'}"
    )
    print("-" * 112)
    for row in summary_rows:
        acc_str = f"{row['test_acc_mean']:.4f} +/- {row['test_acc_std']:.4f}"
        f1_str = f"{row['test_f1_mean']:.4f} +/- {row['test_f1_std']:.4f}"
        bx_str = f"{row['boxes_mean']:.1f}"
        sz_str = f"{row['mean_box_size_mean']:.2f}"
        lam_str = f"{row['chosen_lam_mean']:.2f} {row['chosen_lams']}" if row["chosen_lams"] else "-"
        c_str = (
            f"c={row['chosen_cs']} ({row['pseudo_count_mean']:.0f})"
            if row["chosen_cs"]
            else "-"
        )
        print(
            f"{row['variant']:<18} | {acc_str:<17} | {f1_str:<17} | "
            f"{bx_str:<6} | {sz_str:<8} | {lam_str:<18} | {c_str}"
        )
    print("=" * 112)

    b7_runs = {r["seed"]: r for r in runs if r["variant"] == "B7_hybrid"}
    b2_runs = {r["seed"]: r for r in runs if r["variant"] == "B2_gnn_fmnn"}
    b8_runs = [r for r in runs if r["variant"] == "B8_hybrid_pseudo"]

    if b7_runs:
        print("\nBox Coverage & Stratified Accuracy (B7_hybrid, test split):")
        for r in sorted(b7_runs.values(), key=lambda x: x["seed"]):
            cov_acc = f"{r['covered_accuracy']:.4f}" if r["covered_accuracy"] is not None else "N/A"
            uncov_acc = f"{r['uncovered_accuracy']:.4f}" if r["uncovered_accuracy"] is not None else "N/A"
            print(
                f"  Seed {r['seed']}: lam={r['chosen_lam']:.2f} | "
                f"coverage={r['box_coverage']:.4f} ({r['covered_count']}/{r['covered_count'] + r['uncovered_count']}) | "
                f"covered_acc={cov_acc} | uncovered_acc={uncov_acc}"
            )

    if b8_runs:
        print("\nB8_hybrid_pseudo Diagnostics (train_size excludes pseudo-labels):")
        for r in b8_runs:
            seed = r["seed"]
            cov_acc = f"{r['covered_accuracy']:.4f}" if r["covered_accuracy"] is not None else "N/A"
            uncov_acc = f"{r['uncovered_accuracy']:.4f}" if r["uncovered_accuracy"] is not None else "N/A"
            b7_r = b7_runs.get(seed)
            b2_r = b2_runs.get(seed)
            d_box_b7 = f"{r['num_boxes'] - b7_r['num_boxes']:+d}" if b7_r else "N/A"
            d_sz_b7 = f"{r['mean_box_size'] - b7_r['mean_box_size']:+.3f}" if b7_r else "N/A"
            d_ov_b7 = f"{r['cross_class_overlap'] - b7_r['cross_class_overlap']:+d}" if b7_r else "N/A"
            d_box_b2 = f"{r['delta_num_boxes_vs_b2']:+d}" if b2_r else f"{r['delta_num_boxes_vs_b2']:+d}"
            d_sz_b2 = f"{r['delta_mean_box_size_vs_b2']:+.3f}"
            d_ov_b2 = f"{r['delta_cross_class_overlap_vs_b2']:+d}"
            print(
                f"  Seed {seed}: train_size={r['train_size']}, pseudo_count={r['pseudo_count']} (c={r['chosen_c']}, lam={r['chosen_lam']:.2f}) | "
                f"pseudo_base_boxes={r['pseudo_base_num_boxes']} -> refined={r['num_boxes']} | "
                f"coverage={r['box_coverage']:.4f} (cov_acc={cov_acc}, uncov_acc={uncov_acc})"
            )
            print(
                f"    Delta vs B2: boxes={d_box_b2}, mean_size={d_sz_b2}, overlap={d_ov_b2} | "
                f"Delta vs B7: boxes={d_box_b7}, mean_size={d_sz_b7}, overlap={d_ov_b7}"
            )


# Main entry point running variants and saving results to results/v2/.
def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    data_cfg = cfg.get("data", {})
    data = load_dataset(data_cfg.get("name", "Cora"), root=data_cfg.get("root", "./data"))

    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        ctx = prepare_seed_context(cfg, data, seed)
        for var in args.variants:
            res = run_single_seed(cfg, data, seed, var, seed_ctx=ctx)
            runs.append(res)

    summary_rows = build_summary_rows(runs)
    print_hybrid_table(summary_rows, runs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_slug = data.name.lower()
    json_path = out_dir / f"b8_hybrid_pseudo_{ds_slug}_{args.tag}.json"
    csv_path = out_dir / f"b8_hybrid_pseudo_{ds_slug}_{args.tag}.csv"

    pd.DataFrame(runs).to_csv(csv_path, index=False)
    save_json(
        {
            "dataset": data.name,
            "tag": args.tag,
            "seeds": args.seeds,
            "config": cfg,
            "summary": summary_rows,
            "runs": runs,
        },
        json_path,
    )
    print(f"\nSaved results to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
