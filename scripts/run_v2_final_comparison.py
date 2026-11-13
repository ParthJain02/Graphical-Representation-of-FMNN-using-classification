"""Run final 5-seed comparison across Cora, CiteSeer, and Chameleon for B1..B10_final."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GraphData, homophily_ratio, load_dataset
from src.fmnn import FMNN
from src.gnn import apply_encoder_config, search_encoder_hyperparams
from src.metrics import classification_metrics
from src.pipeline import prepare_seed_context, run_single_seed, select_b10_components
from src.utils import load_config, load_json, save_json


BASELINE_VARIANTS = ["B1", "B2", "B6"]
V2_VARIANTS = ["B7_hybrid", "B8_hybrid_pseudo", "B9_gshr_fast"]


# Parse CLI arguments allowing dataset/seed overrides and option switches.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v2 final comparison across benchmarks.")
    parser.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer", "Chameleon"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--search-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--base-config", type=str, default="configs/default.yaml")
    parser.add_argument("--v2-config", type=str, default="configs/cora_v2.yaml")
    parser.add_argument("--force-encoder-search", action="store_true")
    parser.add_argument("--force-b10-search", action="store_true")
    parser.add_argument("--no-v2-encoder-on-b7b9", action="store_true")
    parser.add_argument("--out-dir", type=str, default="results/v2")
    return parser.parse_args()


# Build dataset-specific base and v2 config dictionaries without mutating templates.
def make_dataset_configs(
    dataset_name: str,
    base_template: dict[str, Any],
    v2_template: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg_base = copy.deepcopy(base_template)
    cfg_base.setdefault("data", {})["name"] = dataset_name
    cfg_v2 = copy.deepcopy(v2_template)
    cfg_v2.setdefault("data", {})["name"] = dataset_name
    return cfg_base, cfg_v2


# Load or run validation-only encoder hyperparameter search for a given dataset.
def get_or_search_encoder(
    data: GraphData,
    cfg_v2: dict[str, Any],
    search_seeds: list[int],
    out_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    enc_path = out_dir / f"encoder_config_{data.name.lower()}.json"
    if enc_path.is_file() and not force:
        payload = load_json(enc_path)
        if "best_config" in payload:
            print(f"  Using cached encoder config: {enc_path}")
            return apply_encoder_config(cfg_v2, payload)

    search_cfg = cfg_v2.get("gnn_search", {})
    gnn_cfg = cfg_v2.get("gnn", {})
    print(f"  Running validation encoder search on {data.name} (seeds={search_seeds})...")
    result = search_encoder_hyperparams(
        data=data,
        cfg=cfg_v2,
        search_seeds=search_seeds,
        hidden_grid=search_cfg.get("hidden", [64, 128]),
        dropout_grid=search_cfg.get("dropout", [0.5, 0.6]),
        weight_decay_grid=search_cfg.get("weight_decay", [5e-4, 1e-3]),
        lr_grid=search_cfg.get("lr", [0.01, 0.005]),
        early_stop_metric=str(gnn_cfg.get("early_stop_metric", "val_loss")),
        patience=int(gnn_cfg.get("patience", 50)),
        max_epochs=int(gnn_cfg.get("max_epochs", 300)),
    )
    save_json(result, enc_path)
    print(f"  Saved {enc_path} (best val_macro_f1={result['best_val_macro_f1_mean']:.4f})")
    return apply_encoder_config(cfg_v2, result)


# Scan FMNN theta using strictly validation accuracy/macro-F1 across precomputed seed contexts.
def scan_theta_validation(
    data: GraphData,
    ctxs: list[Any],
    gamma: float,
    thetas: list[float] | None = None,
) -> list[dict[str, Any]]:
    theta_vals = thetas if thetas is not None else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    tr_idx = data.train_idx.cpu().numpy()
    val_idx = data.val_idx.cpu().numpy()
    y_train = data.y[data.train_idx].cpu().numpy()
    y_val = data.y[data.val_idx].cpu().numpy()
    train_ids = tr_idx.tolist()

    rows: list[dict[str, Any]] = []
    for th in theta_vals:
        val_accs: list[float] = []
        val_f1s: list[float] = []
        box_counts: list[int] = []
        for ctx in ctxs:
            Z_tr = ctx.Z_scaled[tr_idx]
            Z_val = ctx.Z_scaled[val_idx]
            fmnn = FMNN(theta=float(th), gamma=gamma).fit(Z_tr, y_train, train_ids)
            vp = fmnn.predict(Z_val)
            vm = classification_metrics(y_val, vp, data.num_classes)
            val_accs.append(float(vm["accuracy"]))
            val_f1s.append(float(vm["macro_f1"]))
            box_counts.append(int(fmnn.num_boxes()))
        rows.append({
            "dataset": data.name,
            "theta": float(th),
            "val_accuracy_mean": float(np.mean(val_accs)),
            "val_accuracy_std": float(np.std(val_accs, ddof=1)) if len(val_accs) > 1 else 0.0,
            "val_macro_f1_mean": float(np.mean(val_f1s)),
            "val_macro_f1_std": float(np.std(val_f1s, ddof=1)) if len(val_f1s) > 1 else 0.0,
            "num_boxes_mean": float(np.mean(box_counts)),
            "num_boxes_std": float(np.std(box_counts, ddof=1)) if len(box_counts) > 1 else 0.0,
        })

    best_row = min(rows, key=lambda r: (-r["val_macro_f1_mean"], r["num_boxes_mean"], r["theta"]))
    for r in rows:
        r["is_val_best"] = bool(r["theta"] == best_row["theta"])
    return rows


# Run safe paired t-test comparing two paired per-seed arrays without NaN on zero differences.
def safe_paired_ttest(a: list[float], b: list[float]) -> dict[str, Any]:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    diffs = arr_a - arr_b
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    if len(diffs) < 2 or std_diff <= 1e-15:
        return {
            "test_name": "paired_ttest",
            "n": int(len(diffs)),
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "t_stat": 0.0,
            "p_value": 1.0,
        }
    t_res = stats.ttest_rel(arr_a, arr_b)
    return {
        "test_name": "paired_ttest",
        "n": int(len(diffs)),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": float(t_res.statistic),
        "p_value": float(t_res.pvalue),
    }


# Summarize all required metrics (mean, std) per variant for a dataset.
def summarize_dataset_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics_keys = [
        "test_accuracy",
        "test_macro_f1",
        "num_boxes",
        "refine_time",
        "pred_latency_per_node",
        "peak_memory_mb",
        "box_storage_bytes",
        "val_accuracy",
        "val_macro_f1",
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        grouped.setdefault(r["variant"], []).append(r)

    rows: list[dict[str, Any]] = []
    for var_name, group in grouped.items():
        row: dict[str, Any] = {
            "dataset": group[0]["dataset"],
            "variant": var_name,
            "num_seeds": len(group),
        }
        for key in metrics_keys:
            vals = np.asarray([float(g[key]) for g in group], dtype=np.float64)
            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        recalls = [g["test_per_class_recall"] for g in group if "test_per_class_recall" in g]
        if recalls:
            row["per_class_recall_mean"] = np.mean(np.asarray(recalls, dtype=np.float64), axis=0).tolist()
            row["per_class_recall_std"] = (
                np.std(np.asarray(recalls, dtype=np.float64), axis=0, ddof=1).tolist()
                if len(recalls) > 1 else [0.0] * len(recalls[0])
            )
        rows.append(row)
    return rows


# Compute B10_final vs B1 accuracy/F1 gap, B10 vs B2 box reduction %, and paired t-tests.
def compute_b10_comparisons(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_var: dict[str, dict[int, dict[str, Any]]] = {}
    for r in runs:
        by_var.setdefault(r["variant"], {})[int(r["seed"])] = r

    if "B10_final" not in by_var or "B1_gnn_softmax" not in by_var or "B2_gnn_fmnn" not in by_var:
        return {}

    seeds = sorted(by_var["B10_final"].keys())
    b10_acc = [float(by_var["B10_final"][s]["test_accuracy"]) for s in seeds]
    b1_acc = [float(by_var["B1_gnn_softmax"][s]["test_accuracy"]) for s in seeds]
    b10_f1 = [float(by_var["B10_final"][s]["test_macro_f1"]) for s in seeds]
    b1_f1 = [float(by_var["B1_gnn_softmax"][s]["test_macro_f1"]) for s in seeds]
    b10_boxes = [float(by_var["B10_final"][s]["num_boxes"]) for s in seeds]
    b2_boxes = [float(by_var["B2_gnn_fmnn"][s]["num_boxes"]) for s in seeds]

    acc_gap = float(np.mean(b10_acc) - np.mean(b1_acc))
    f1_gap = float(np.mean(b10_f1) - np.mean(b1_f1))
    b2_boxes_mean = float(np.mean(b2_boxes))
    b10_boxes_mean = float(np.mean(b10_boxes))
    box_red_pct = float((1.0 - b10_boxes_mean / b2_boxes_mean) * 100.0) if b2_boxes_mean > 0 else 0.0

    return {
        "b10_acc_mean": float(np.mean(b10_acc)),
        "b1_acc_mean": float(np.mean(b1_acc)),
        "acc_gap_b10_minus_b1": acc_gap,
        "b10_f1_mean": float(np.mean(b10_f1)),
        "b1_f1_mean": float(np.mean(b1_f1)),
        "f1_gap_b10_minus_b1": f1_gap,
        "b2_boxes_mean": b2_boxes_mean,
        "b10_boxes_mean": b10_boxes_mean,
        "box_reduction_vs_b2_pct": box_red_pct,
        "paired_test_accuracy": safe_paired_ttest(b10_acc, b1_acc),
        "paired_test_macro_f1": safe_paired_ttest(b10_f1, b1_f1),
    }


# Print concise per-dataset table of all metrics and paired statistics.
def print_dataset_table(dataset_name: str, summary_rows: list[dict[str, Any]], comp: dict[str, Any], b10_sel: dict[str, Any]) -> None:
    print("\n" + "=" * 124)
    print(f"FINAL COMPARISON: {dataset_name} (Seeds 0..4)")
    print(f"B10_final validation-selected components: {b10_sel.get('selected', {})}")
    print("-" * 124)
    print(
        f"{'Variant':<18} | {'Test Acc':<17} | {'Test Macro-F1':<17} | {'Boxes':<12} | "
        f"{'Refine(s)':<12} | {'Lat(us/node)':<12} | {'PeakMem(MB)':<12} | {'Storage(B)'}"
    )
    print("-" * 124)
    for r in summary_rows:
        acc_s = f"{r['test_accuracy_mean']:.4f} +/- {r['test_accuracy_std']:.4f}"
        f1_s = f"{r['test_macro_f1_mean']:.4f} +/- {r['test_macro_f1_std']:.4f}"
        bx_s = f"{r['num_boxes_mean']:.1f} +/- {r['num_boxes_std']:.1f}"
        rt_s = f"{r['refine_time_mean']:.3f}"
        lat_s = f"{r['pred_latency_per_node_mean'] * 1e6:.2f}"
        mem_s = f"{r['peak_memory_mb_mean']:.2f}"
        st_s = f"{r['box_storage_bytes_mean']:.0f}"
        print(
            f"{r['variant']:<18} | {acc_s:<17} | {f1_s:<17} | {bx_s:<12} | "
            f"{rt_s:<12} | {lat_s:<12} | {mem_s:<12} | {st_s}"
        )
    print("-" * 124)
    if comp:
        pa = comp["paired_test_accuracy"]
        pf = comp["paired_test_macro_f1"]
        print(
            f"B10_final vs B1: Acc gap={comp['acc_gap_b10_minus_b1']:+.4f} "
            f"(paired t={pa['t_stat']:+.3f}, p={pa['p_value']:.4f}) | "
            f"Macro-F1 gap={comp['f1_gap_b10_minus_b1']:+.4f} "
            f"(paired t={pf['t_stat']:+.3f}, p={pf['p_value']:.4f}) | "
            f"Box reduction vs B2={comp['box_reduction_vs_b2_pct']:.1f}%"
        )
    print("=" * 124)


# Main driver running encoder search, theta scan, B10 component selection, and 5-seed benchmarks.
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_template = load_config(args.base_config)
    v2_template = load_config(args.v2_config)

    all_runs: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    all_theta_scans: list[dict[str, Any]] = []
    dataset_reports: dict[str, Any] = {}

    for ds_name in args.datasets:
        print(f"\n>>> Processing dataset: {ds_name}")
        data = load_dataset(ds_name, root=base_template.get("data", {}).get("root", "./data"))
        cfg_base, cfg_v2_raw = make_dataset_configs(data.name, base_template, v2_template)
        cfg_v2 = get_or_search_encoder(
            data=data,
            cfg_v2=cfg_v2_raw,
            search_seeds=args.search_seeds,
            out_dir=out_dir,
            force=args.force_encoder_search,
        )

        b10_path = out_dir / f"b10_components_{data.name.lower()}.json"
        if b10_path.is_file() and not args.force_b10_search:
            cached_b10 = load_json(b10_path)
            if list(cached_b10.get("seeds", [])) == list(args.seeds):
                print(f"  Using cached B10_final components: {b10_path}")
                b10_sel = cached_b10
            else:
                print(f"  Selecting B10_final components on validation for {data.name}...")
                b10_sel = select_b10_components(data, cfg_base, cfg_v2, seeds=args.seeds)
                save_json(b10_sel, b10_path)
        else:
            print(f"  Selecting B10_final components on validation for {data.name}...")
            b10_sel = select_b10_components(data, cfg_base, cfg_v2, seeds=args.seeds)
            save_json(b10_sel, b10_path)

        ds_runs: list[dict[str, Any]] = []
        v2_ctxs: list[Any] = []

        for seed in args.seeds:
            print(f"  Running {data.name} seed {seed}...")
            ctx_base = prepare_seed_context(cfg_base, data, seed)
            ctx_v2 = prepare_seed_context(cfg_v2, data, seed)
            v2_ctxs.append(ctx_v2)

            for var in BASELINE_VARIANTS:
                ds_runs.append(run_single_seed(cfg_base, data, seed, var, seed_ctx=ctx_base))

            if args.no_v2_encoder_on_b7b9:
                cfg_var, ctx_var = cfg_base, ctx_base
            else:
                cfg_var, ctx_var = cfg_v2, ctx_v2

            for var in V2_VARIANTS:
                res = run_single_seed(cfg_var, data, seed, var, seed_ctx=ctx_var)
                res.pop("settings_table", None)
                ds_runs.append(res)

            use_best_enc = bool(b10_sel["selected"]["use_best_encoder"])
            cfg_b10 = copy.deepcopy(cfg_v2 if use_best_enc else cfg_base)
            cfg_b10["b10_components"] = b10_sel["selected"]
            ctx_b10 = ctx_v2 if use_best_enc else ctx_base
            ds_runs.append(run_single_seed(cfg_b10, data, seed, "B10_final", seed_ctx=ctx_b10))

        gamma = float(cfg_v2.get("fmnn", {}).get("gamma", 4.0))
        theta_rows = scan_theta_validation(data, v2_ctxs, gamma=gamma)
        all_theta_scans.extend(theta_rows)

        ds_summary = summarize_dataset_runs(ds_runs)
        ds_comp = compute_b10_comparisons(ds_runs)
        print_dataset_table(data.name, ds_summary, ds_comp, b10_sel)

        all_runs.extend(ds_runs)
        all_summaries.extend(ds_summary)
        dataset_reports[data.name] = {
            "homophily_ratio": homophily_ratio(data),
            "num_nodes": data.num_nodes,
            "num_classes": data.num_classes,
            "embed_dim": int(v2_ctxs[0].Z_scaled.shape[1]),
            "b10_components": b10_sel,
            "summary": ds_summary,
            "b10_vs_b1_comparison": ds_comp,
        }

    pd.DataFrame(all_theta_scans).to_csv(out_dir / "theta_sensitivity_val.csv", index=False)

    flat_runs = [{k: v for k, v in r.items() if k != "test_per_class_recall"} for r in all_runs]
    pd.DataFrame(flat_runs).to_csv(out_dir / "final_comparison_runs.csv", index=False)

    flat_sum = [{k: v for k, v in s.items() if "per_class_recall" not in k} for s in all_summaries]
    pd.DataFrame(flat_sum).to_csv(out_dir / "final_comparison_summary.csv", index=False)

    save_json(
        {
            "datasets": args.datasets,
            "seeds": args.seeds,
            "datasets_detail": dataset_reports,
            "summary": all_summaries,
            "runs": all_runs,
        },
        out_dir / "final_comparison.json",
    )
    print(f"\nSaved complete final comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
