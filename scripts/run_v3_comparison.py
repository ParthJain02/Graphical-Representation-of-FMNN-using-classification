"""Run v3 unified latency benchmark and 10-seed B11_final_lite comparison into results/v3/."""

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

from scripts.run_v2_final_comparison import (
    get_or_search_encoder,
    make_dataset_configs,
    safe_paired_ttest,
    summarize_dataset_runs,
)
from src.data import load_dataset
from src.pipeline import prepare_seed_context, run_single_seed, select_b10_components
from src.utils import load_config, load_json, save_json


LATENCY_VARIANTS = [
    "B1",
    "B2",
    "B6",
    "B7_hybrid",
    "B8_hybrid_pseudo",
    "B9_gshr_fast",
    "B10_final",
    "B11_final_lite",
]

V3_COMPARE_VARIANTS = [
    "B1",
    "B7_hybrid",
    "B9_gshr_fast",
    "B10_final",
    "B11_final_lite",
]


# Parse CLI flags for v3 latency measurement and 10-seed benchmark.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 unified latency and 10-seed comparison.")
    parser.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer", "Chameleon"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--latency-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--base-config", type=str, default="configs/default.yaml")
    parser.add_argument("--v2-config", type=str, default="configs/cora_v2.yaml")
    parser.add_argument("--v2-dir", type=str, default="results/v2")
    parser.add_argument("--out-dir", type=str, default="results/v3")
    parser.add_argument("--no-capped-pseudo", action="store_true")
    parser.add_argument("--no-new-boxes", action="store_true")
    parser.add_argument("--cap-per-class", type=int, default=None)
    parser.add_argument("--skip-latency-table", action="store_true")
    parser.add_argument("--skip-10seed", action="store_true")
    return parser.parse_args()


# Build variant config and seed context consistently while optionally enabling unified latency.
def _resolve_variant_cfg_ctx(
    variant: str,
    cfg_base: dict[str, Any],
    cfg_v2: dict[str, Any],
    ctx_base: Any,
    ctx_v2: Any,
    b10_sel: dict[str, Any],
    args: argparse.Namespace,
    unified_latency: bool = True,
) -> tuple[dict[str, Any], Any]:
    v_up = variant.strip().upper()
    if v_up in {"B1", "B2", "B6"}:
        cfg_use = copy.deepcopy(cfg_base)
        ctx_use = ctx_base
    elif v_up in {"B10", "B10_FINAL"}:
        use_best = bool(b10_sel["selected"]["use_best_encoder"])
        cfg_use = copy.deepcopy(cfg_v2 if use_best else cfg_base)
        cfg_use["b10_components"] = dict(b10_sel["selected"])
        ctx_use = ctx_v2 if use_best else ctx_base
    elif v_up in {"B11", "B11_FINAL_LITE", "FINAL_LITE"}:
        use_best = bool(b10_sel["selected"]["use_best_encoder"])
        cfg_use = copy.deepcopy(cfg_v2 if use_best else cfg_base)
        cfg_use["b11_lite"] = {
            "use_capped_pseudo": not bool(args.no_capped_pseudo),
            "use_fast_gshr": True,
            "use_hybrid": True,
            "allow_new_boxes": not bool(args.no_new_boxes),
            "cap_per_class_grid": [20, 50, 100],
            "cap_per_class": args.cap_per_class,
        }
        ctx_use = ctx_v2 if use_best else ctx_base
    else:
        cfg_use = copy.deepcopy(cfg_v2)
        ctx_use = ctx_v2

    cfg_use["unified_latency"] = bool(unified_latency)
    cfg_use["latency"] = {"warmup": 20, "repeats": 100}
    return cfg_use, ctx_use


# Load cached B10 validation component selection or compute if missing.
def _load_or_select_b10(
    data: Any,
    cfg_base: dict[str, Any],
    cfg_v2: dict[str, Any],
    v2_dir: Path,
) -> dict[str, Any]:
    b10_path = v2_dir / f"b10_components_{data.name.lower()}.json"
    if b10_path.is_file():
        return load_json(b10_path)
    b10_sel = select_b10_components(data, cfg_base, cfg_v2, seeds=[0, 1, 2, 3, 4])
    save_json(b10_sel, b10_path)
    return b10_sel


# Compute paired t-test comparisons between B11_final_lite and comparator variants.
def compute_b11_paired_stats(runs: list[dict[str, Any]], dataset_name: str) -> list[dict[str, Any]]:
    by_var: dict[str, dict[int, dict[str, Any]]] = {}
    for r in runs:
        by_var.setdefault(r["variant"], {})[int(r["seed"])] = r

    if "B11_final_lite" not in by_var:
        return []

    seeds = sorted(by_var["B11_final_lite"].keys())
    comparators = ["B1_gnn_softmax", "B7_hybrid", "B9_gshr_fast", "B10_final"]
    metrics = [
        "test_accuracy",
        "test_macro_f1",
        "num_boxes",
        "refine_time",
        "peak_memory_mb",
        "box_storage_bytes",
    ]
    rows: list[dict[str, Any]] = []
    for comp_var in comparators:
        if comp_var not in by_var:
            continue
        common = [s for s in seeds if s in by_var[comp_var]]
        if not common:
            continue
        row: dict[str, Any] = {
            "dataset": dataset_name,
            "baseline_variant": comp_var,
            "n_seeds": len(common),
        }
        for m in metrics:
            vals_b11 = [float(by_var["B11_final_lite"][s][m]) for s in common]
            vals_cmp = [float(by_var[comp_var][s][m]) for s in common]
            tt = safe_paired_ttest(vals_b11, vals_cmp)
            row[f"{m}_b11_mean"] = float(np.mean(vals_b11))
            row[f"{m}_cmp_mean"] = float(np.mean(vals_cmp))
            row[f"{m}_diff"] = float(tt["mean_diff"])
            row[f"{m}_t"] = float(tt["t_stat"])
            row[f"{m}_p"] = float(tt["p_value"])
        rows.append(row)
    return rows


# Write unified latency, 10-seed summary, and paired t-test results to markdown.
def write_v3_markdown(
    lat_summary: list[dict[str, Any]],
    comp_summary: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    b8_profile_path: Path,
    out_path: Path,
) -> None:
    lines: list[str] = ["# GNN + GSHR + FMNN v3 Results\n"]

    if b8_profile_path.is_file():
        prof = load_json(b8_profile_path)
        lines.append("## 1. B8_hybrid_pseudo Profiling Breakdown (Seed 0, c in {0.8, 0.9, 0.95})\n")
        lines.append("| Dataset | PseudoSel (s / %) | FMNNFit (s / %) | Merge (s / %) | Prune (s / %) | Total (s) | Overall Peak Mem (MB) |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for s in prof.get("summary", []):
            lines.append(
                f"| {s['dataset']} | {s['pseudo_selection_time_s']:.3f} ({s['pseudo_selection_pct']:.1f}%) "
                f"| {s['fmnn_fit_time_s']:.3f} ({s['fmnn_fit_pct']:.1f}%) "
                f"| {s['pairwise_merge_time_s']:.3f} ({s['pairwise_merge_pct']:.1f}%) "
                f"| {s['prune_time_s']:.3f} ({s['prune_pct']:.1f}%) "
                f"| {s['total_refine_time_s']:.3f} | {s['overall_peak_mb']:.2f} |"
            )
        lines.append("")

    if lat_summary:
        lines.append("## 2. Unified Prediction Latency (CPU, 20 warm-up, median of 100 full-test-batch runs)\n")
        lines.append("| Dataset | Variant | Latency (us/node) mean +/- std | Total Test Acc |")
        lines.append("| --- | --- | --- | --- |")
        for r in lat_summary:
            lat_us = f"{r['latency_us_mean']:.2f} +/- {r['latency_us_std']:.2f}"
            lines.append(f"| {r['dataset']} | {r['variant']} | {lat_us} | {r['test_accuracy_mean']:.4f} |")
        lines.append("")

    if comp_summary:
        lines.append("## 3. 10-Seed Comparison (Seeds 0..9)\n")
        lines.append(
            "| Dataset | Variant | Test Acc | Test Macro-F1 | Boxes | Refine (s) | Lat (us/node) | Peak Mem (MB) | Storage (B) | Pseudo Count (mean) |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in comp_summary:
            acc = f"{r['test_accuracy_mean']:.4f} +/- {r['test_accuracy_std']:.4f}"
            f1 = f"{r['test_macro_f1_mean']:.4f} +/- {r['test_macro_f1_std']:.4f}"
            bx = f"{r['num_boxes_mean']:.1f} +/- {r['num_boxes_std']:.1f}"
            rf = f"{r['refine_time_mean']:.3f} +/- {r['refine_time_std']:.3f}"
            lt = f"{r['pred_latency_per_node_mean'] * 1e6:.2f} +/- {r['pred_latency_per_node_std'] * 1e6:.2f}"
            mm = f"{r['peak_memory_mb_mean']:.2f} +/- {r['peak_memory_mb_std']:.2f}"
            st = f"{r['box_storage_bytes_mean']:.0f}"
            ps = f"{r.get('pseudo_count_mean', 0.0):.1f}"
            lines.append(f"| {r['dataset']} | {r['variant']} | {acc} | {f1} | {bx} | {rf} | {lt} | {mm} | {st} | {ps} |")
        lines.append("")

    if paired_rows:
        lines.append("## 4. Paired t-tests: B11_final_lite vs Comparators (10 Seeds)\n")
        lines.append(
            "| Dataset | Comparator | dAcc (p-val) | dF1 (p-val) | dBoxes (p-val) | dRefineTime_s (p-val) | dPeakMem_MB (p-val) |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in paired_rows:
            da = f"{r['test_accuracy_diff']:+.4f} (p={r['test_accuracy_p']:.4f})"
            df = f"{r['test_macro_f1_diff']:+.4f} (p={r['test_macro_f1_p']:.4f})"
            db = f"{r['num_boxes_diff']:+.2f} (p={r['num_boxes_p']:.4f})"
            dr = f"{r['refine_time_diff']:+.3f} (p={r['refine_time_p']:.4f})"
            dm = f"{r['peak_memory_mb_diff']:+.2f} (p={r['peak_memory_mb_p']:.4f})"
            lines.append(f"| {r['dataset']} | {r['baseline_variant']} | {da} | {df} | {db} | {dr} | {dm} |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# Run unified latency benchmark and 10-seed v3 comparison across datasets.
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    v2_dir = Path(args.v2_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v2_dir.mkdir(parents=True, exist_ok=True)

    base_tmpl = load_config(args.base_config)
    v2_tmpl = load_config(args.v2_config)

    lat_runs: list[dict[str, Any]] = []
    lat_summary: list[dict[str, Any]] = []
    comp_runs: list[dict[str, Any]] = []
    comp_summary: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for ds_name in args.datasets:
        print(f"\n>>> v3 Benchmark on {ds_name}")
        data = load_dataset(ds_name, root=base_tmpl.get("data", {}).get("root", "./data"))
        cfg_base, cfg_v2_raw = make_dataset_configs(data.name, base_tmpl, v2_tmpl)
        cfg_v2 = get_or_search_encoder(data, cfg_v2_raw, search_seeds=[0, 1, 2], out_dir=v2_dir, force=False)
        b10_sel = _load_or_select_b10(data, cfg_base, cfg_v2, v2_dir)

        needed_seeds = sorted(set(([] if args.skip_latency_table else args.latency_seeds) + ([] if args.skip_10seed else args.seeds)))
        seed_cache: dict[int, tuple[Any, Any]] = {}
        run_cache: dict[tuple[int, str], dict[str, Any]] = {}

        def _get_or_run(s: int, var: str) -> dict[str, Any]:
            if s not in seed_cache:
                seed_cache[s] = (
                    prepare_seed_context(cfg_base, data, s),
                    prepare_seed_context(cfg_v2, data, s),
                )
            key = (s, var)
            if key not in run_cache:
                ctx_base, ctx_v2 = seed_cache[s]
                cfg_u, ctx_u = _resolve_variant_cfg_ctx(var, cfg_base, cfg_v2, ctx_base, ctx_v2, b10_sel, args, unified_latency=True)
                res = run_single_seed(cfg_u, data, s, var, seed_ctx=ctx_u)
                res.pop("settings_table", None)
                run_cache[key] = res
            return copy.deepcopy(run_cache[key])

        if not args.skip_latency_table:
            ds_lat_runs: list[dict[str, Any]] = []
            for s in args.latency_seeds:
                print(f"  [{data.name}] unified latency seed={s}...", flush=True)
                for var in LATENCY_VARIANTS:
                    ds_lat_runs.append(_get_or_run(s, var))
            lat_runs.extend(ds_lat_runs)
            ds_lat_sum = summarize_dataset_runs(ds_lat_runs)
            for row in ds_lat_sum:
                row["latency_us_mean"] = row["pred_latency_per_node_mean"] * 1e6
                row["latency_us_std"] = row["pred_latency_per_node_std"] * 1e6
            lat_summary.extend(ds_lat_sum)

        if not args.skip_10seed:
            ds_comp_runs: list[dict[str, Any]] = []
            for s in args.seeds:
                print(f"  [{data.name}] running 10-seed v3 comparison seed={s}...", flush=True)
                for var in V3_COMPARE_VARIANTS:
                    ds_comp_runs.append(_get_or_run(s, var))

            comp_runs.extend(ds_comp_runs)
            ds_comp_sum = summarize_dataset_runs(ds_comp_runs)
            for row in ds_comp_sum:
                grp = [g for g in ds_comp_runs if g["variant"] == row["variant"]]
                p_vals = [float(g.get("pseudo_count", 0)) for g in grp]
                row["pseudo_count_mean"] = float(np.mean(p_vals)) if p_vals else 0.0
                row["chosen_k_values"] = [g.get("chosen_k") for g in grp if g.get("chosen_k") is not None]
            comp_summary.extend(ds_comp_sum)
            paired_rows.extend(compute_b11_paired_stats(ds_comp_runs, data.name))

    if lat_summary:
        print("\n=== Unified Prediction Latency Table (CPU, 20 warm-up, median of 100 batch runs) ===")
        print(f"{'Dataset':<10} | {'Variant':<18} | {'Lat(us/node)':<22} | {'Test Acc'}")
        print("-" * 70)
        for r in lat_summary:
            lat_s = f"{r['latency_us_mean']:.2f} +/- {r['latency_us_std']:.2f}"
            print(f"{r['dataset']:<10} | {r['variant']:<18} | {lat_s:<22} | {r['test_accuracy_mean']:.4f}")
        pd.DataFrame(lat_summary).to_csv(out_dir / "unified_latency.csv", index=False)
        save_json({"runs": lat_runs, "summary": lat_summary}, out_dir / "unified_latency.json")

    if comp_summary:
        print("\n=== v3 10-Seed Comparison Summary ===")
        hdr = (
            f"{'Dataset':<10} | {'Variant':<16} | {'Test Acc':<17} | {'Test F1':<17} | "
            f"{'Boxes':<12} | {'Refine(s)':<10} | {'PeakMB':<8} | {'Storage(B)':<10} | {'Pseudo'}"
        )
        print(hdr)
        print("-" * len(hdr))
        for r in comp_summary:
            acc = f"{r['test_accuracy_mean']:.4f} +/- {r['test_accuracy_std']:.4f}"
            f1 = f"{r['test_macro_f1_mean']:.4f} +/- {r['test_macro_f1_std']:.4f}"
            bx = f"{r['num_boxes_mean']:.1f} +/- {r['num_boxes_std']:.1f}"
            print(
                f"{r['dataset']:<10} | {r['variant']:<16} | {acc:<17} | {f1:<17} | "
                f"{bx:<12} | {r['refine_time_mean']:<10.3f} | {r['peak_memory_mb_mean']:<8.2f} | "
                f"{r['box_storage_bytes_mean']:<10.0f} | {r.get('pseudo_count_mean', 0.0):.1f}"
            )

        flat_runs = [
            {k: v for k, v in r.items() if k not in {"test_per_class_recall", "pseudo_per_class_counts"}}
            for r in comp_runs
        ]
        flat_sum = [
            {k: v for k, v in s.items() if "per_class_recall" not in k and k != "chosen_k_values"}
            for s in comp_summary
        ]
        pd.DataFrame(flat_runs).to_csv(out_dir / "v3_comparison_runs.csv", index=False)
        pd.DataFrame(flat_sum).to_csv(out_dir / "v3_comparison_summary.csv", index=False)
        pd.DataFrame(paired_rows).to_csv(out_dir / "v3_paired_tests.csv", index=False)
        save_json(
            {
                "datasets": args.datasets,
                "seeds": args.seeds,
                "summary": comp_summary,
                "paired_tests": paired_rows,
                "runs": comp_runs,
            },
            out_dir / "v3_comparison.json",
        )

    write_v3_markdown(lat_summary, comp_summary, paired_rows, out_dir / "b8_profile.json", out_dir / "v3_tables.md")
    print(f"\nSaved v3 outputs to {out_dir}")


if __name__ == "__main__":
    main()
