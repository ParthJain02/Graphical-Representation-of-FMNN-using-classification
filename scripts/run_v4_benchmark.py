"""Run v4 encoder search, B12_ensemble, B13_gated_hybrid, B14_final, and 10-seed comparisons."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.make_report import build_v4_markdown_report, generate_v4_plots
from scripts.run_v2_final_comparison import make_dataset_configs, safe_paired_ttest, summarize_dataset_runs
from scripts.run_v3_comparison import _load_or_select_b10
from src.data import load_dataset
from src.gnn import apply_encoder_config, search_encoders_validation, train_gnn
from src.pipeline import prepare_seed_context, run_single_seed, select_b14_components
from src.utils import load_config, load_json, save_json


ALL_V4_DATASETS = ["Cora", "CiteSeer", "Chameleon", "PubMed"]


# Parse CLI flags for datasets, seeds, encoder overrides, and feature switches.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v4 benchmark into results/v4.")
    parser.add_argument("--datasets", nargs="+", default=ALL_V4_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--base-config", type=str, default="configs/default.yaml")
    parser.add_argument("--v2-config", type=str, default="configs/cora_v2.yaml")
    parser.add_argument("--v2-dir", type=str, default="results/v2")
    parser.add_argument("--out-dir", type=str, default="results/v4")
    parser.add_argument("--force-encoder-search", action="store_true")
    parser.add_argument("--force-b14-search", action="store_true")
    parser.add_argument("--force-runs", action="store_true")
    parser.add_argument("--no-pseudo", action="store_true")
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--finalize-only", action="store_true", help="Aggregate cached per-dataset JSONs only.")
    return parser.parse_args()


# Build full v4 config dictionary with default grids for fast GSHR, B12, B13, and smoothing.
def _build_v4_cfg(base_tmpl: dict[str, Any], v2_tmpl: dict[str, Any], ds_name: str) -> dict[str, Any]:
    cfg = copy.deepcopy(v2_tmpl)
    cfg.setdefault("data", {})["name"] = ds_name
    cfg.setdefault("gnn", {})["early_stop_metric"] = "val_macro_f1"
    cfg["gnn"]["max_epochs"] = 300
    cfg["gnn"]["patience"] = 50
    cfg.setdefault("embed", {})["l2_norm"] = True
    cfg["embed"]["clip"] = True
    cfg.setdefault("b12_ensemble", {"num_models": 5, "seed_stride": 100, "embed_mode": "auto", "use_fast_gshr": True})
    cfg.setdefault("b13_gated", {"T_grid": [0.5, 1.0, 1.5, 2.0], "use_fast_gshr": True})
    cfg.setdefault("smoothing", {"rho_grid": [0.0, 0.1, 0.2, 0.3, 0.5]})
    cfg.setdefault("b11_lite", {"cap_per_class_grid": [20, 50, 100], "min_conf": 0.8, "allow_new_boxes": True})
    cfg["unified_latency"] = True
    cfg["latency"] = {"warmup": 20, "repeats": 100}
    return cfg


# Run or load per-dataset validation search over gcn, sage, appnp, gcnii (and h2gcn on Chameleon).
def _get_or_search_encoders_v4(
    data: Any,
    cfg_v4: dict[str, Any],
    out_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    cache_path = out_dir / f"encoder_search_{data.name.lower()}.json"
    if cache_path.is_file() and not force:
        print(f"  Using cached v4 encoder search: {cache_path}", flush=True)
        return load_json(cache_path)

    is_chameleon = data.name.lower() == "chameleon"
    is_pubmed = data.name.lower() == "pubmed"
    encoders = ["gcn", "sage", "appnp", "gcnii", "h2gcn"] if is_chameleon else ["gcn", "sage", "appnp", "gcnii"]

    search_seeds = [0] if is_pubmed else [0, 1]
    hidden_grid = [64, 128]
    dropout_grid = [0.5, 0.6]
    wd_grid = [5e-6, 1e-4, 5e-4]
    lr_grid = [0.01, 0.005]
    max_epochs = 150 if is_pubmed else 200
    patience = 30 if is_pubmed else 40

    print(f"  Running v4 encoder validation search on {data.name} (encoders={encoders}, seeds={search_seeds})...", flush=True)
    search_res = search_encoders_validation(
        data=data,
        cfg=cfg_v4,
        encoders=encoders,
        search_seeds=search_seeds,
        hidden_grid=hidden_grid,
        dropout_grid=dropout_grid,
        weight_decay_grid=wd_grid,
        lr_grid=lr_grid,
        early_stop_metric="val_macro_f1",
        patience=patience,
        max_epochs=max_epochs,
    )
    save_json(search_res, cache_path)
    print(f"  Saved {cache_path} | Best encoder: {search_res['best_encoder']} (val_f1={search_res['best_val_macro_f1_mean']:.4f})", flush=True)
    return search_res


# Evaluate B1 across 10 seeds for each encoder's validation-selected config (without influencing selection).
def _evaluate_b1_per_encoder(
    data: Any,
    cfg_v4: dict[str, Any],
    search_res: dict[str, Any],
    seeds: list[int],
    out_dir: Path,
    force: bool = False,
) -> list[dict[str, Any]]:
    cache_path = out_dir / f"encoder_b1_{data.name.lower()}.json"
    if cache_path.is_file() and not force:
        return load_json(cache_path)["rows"]

    rows: list[dict[str, Any]] = []
    best_enc = search_res["best_encoder"]
    for enc_name, enc_payload in search_res["per_encoder"].items():
        cfg_enc = apply_encoder_config(cfg_v4, enc_payload)
        cfg_enc["gnn"]["max_epochs"] = 300
        cfg_enc["gnn"]["patience"] = 50
        v_accs, v_f1s, t_accs, t_f1s = [], [], [], []
        for s in seeds:
            _, _, tm = train_gnn(data, cfg_enc, seed=s, return_test_metrics=True)
            v_accs.append(float(tm["best_val_acc"]))
            v_f1s.append(float(tm["best_val_macro_f1"]))
            t_accs.append(float(tm["test_acc"]))
            t_f1s.append(float(tm["test_macro_f1"]))
        rows.append({
            "dataset": data.name,
            "encoder": enc_name,
            "is_val_best": bool(enc_name == best_enc),
            "val_acc_mean": float(np.mean(v_accs)),
            "val_acc_std": float(np.std(v_accs, ddof=1)) if len(v_accs) > 1 else 0.0,
            "val_macro_f1_mean": float(np.mean(v_f1s)),
            "val_macro_f1_std": float(np.std(v_f1s, ddof=1)) if len(v_f1s) > 1 else 0.0,
            "test_acc_mean": float(np.mean(t_accs)),
            "test_acc_std": float(np.std(t_accs, ddof=1)) if len(t_accs) > 1 else 0.0,
            "test_macro_f1_mean": float(np.mean(t_f1s)),
            "test_macro_f1_std": float(np.std(t_f1s, ddof=1)) if len(t_f1s) > 1 else 0.0,
            "best_config": enc_payload["best_config"],
        })

    save_json({"dataset": data.name, "rows": rows}, cache_path)
    return rows


# Compute 10-seed paired t-tests for v4 variants against both B1 baselines and B11_final_lite.
def compute_v4_paired_tests(runs: list[dict[str, Any]], dataset_name: str) -> list[dict[str, Any]]:
    by_var: dict[str, dict[int, dict[str, Any]]] = {}
    for r in runs:
        by_var.setdefault(r["variant"], {})[int(r["seed"])] = r

    targets = ["B12_ensemble", "B13_gated_hybrid", "B14_final"]
    baselines = ["B1_base_gcn", "B1_best_encoder", "B11_final_lite"]
    metrics = [
        "test_accuracy",
        "test_macro_f1",
        "num_boxes",
        "refine_time",
        "peak_memory_mb",
        "pred_latency_per_node",
    ]
    rows: list[dict[str, Any]] = []
    for var in targets:
        if var not in by_var:
            continue
        for base in baselines:
            if base not in by_var:
                continue
            common = sorted(s for s in by_var[var] if s in by_var[base])
            if not common:
                continue
            row: dict[str, Any] = {
                "dataset": dataset_name,
                "variant": var,
                "baseline_variant": base,
                "n_seeds": len(common),
            }
            for m in metrics:
                va = [float(by_var[var][s][m]) for s in common]
                vb = [float(by_var[base][s][m]) for s in common]
                tt = safe_paired_ttest(va, vb)
                row[f"{m}_var_mean"] = float(np.mean(va))
                row[f"{m}_cmp_mean"] = float(np.mean(vb))
                row[f"{m}_diff"] = float(tt["mean_diff"])
                row[f"{m}_t"] = float(tt["t_stat"])
                row[f"{m}_p"] = float(tt["p_value"])
            rows.append(row)
    return rows


# Summarize dataset runs including B13/B14 box-decision metrics safely with NaN fallback.
def summarize_v4_dataset_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_summary = summarize_dataset_runs(runs)
    extra_keys = [
        "box_decided_fraction",
        "box_decided_accuracy",
        "gnn_decided_accuracy",
        "fraction_changed_by_boxes",
        "b7_test_accuracy",
        "b7_test_macro_f1",
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        grouped.setdefault(r["variant"], []).append(r)

    for row in base_summary:
        grp = grouped.get(row["variant"], [])
        for ek in extra_keys:
            vals = [float(g[ek]) for g in grp if g.get(ek) is not None]
            if vals:
                row[f"{ek}_mean"] = float(np.mean(vals))
                row[f"{ek}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return base_summary


# Run full v4 workflow on a single dataset and cache its runs/metadata to disk.
def run_dataset_v4(
    ds_name: str,
    args: argparse.Namespace,
    base_tmpl: dict[str, Any],
    v2_tmpl: dict[str, Any],
    out_dir: Path,
    v2_dir: Path,
) -> dict[str, Any]:
    print(f"\n>>> v4 Benchmark on {ds_name}", flush=True)
    data = load_dataset(ds_name, root=base_tmpl.get("data", {}).get("root", "./data"))
    cfg_base, cfg_v2_raw = make_dataset_configs(data.name, base_tmpl, v2_tmpl)
    cfg_v4 = _build_v4_cfg(base_tmpl, v2_tmpl, data.name)

    # 1. Encoder search + 10-seed B1 per-encoder report
    enc_search = _get_or_search_encoders_v4(data, cfg_v4, out_dir, force=args.force_encoder_search)
    enc_b1_rows = _evaluate_b1_per_encoder(data, cfg_v4, enc_search, args.seeds, out_dir, force=args.force_encoder_search)

    print(f"\n  [{data.name}] B1 Encoder Comparison (Seeds {args.seeds[0]}..{args.seeds[-1]}):", flush=True)
    for r in enc_b1_rows:
        mark = "*" if r["is_val_best"] else " "
        print(
            f"   {mark} {r['encoder']:<6} | val_acc={r['val_acc_mean']:.4f}+/-{r['val_acc_std']:.4f} "
            f"| val_f1={r['val_macro_f1_mean']:.4f} | test_acc={r['test_acc_mean']:.4f}+/-{r['test_acc_std']:.4f}",
            flush=True,
        )

    cfg_best_enc = apply_encoder_config(cfg_v4, enc_search)
    cfg_best_enc["gnn"]["max_epochs"] = 300
    cfg_best_enc["gnn"]["patience"] = 50

    # 2. Validation component gating for B14_final
    b14_path = out_dir / f"b14_components_{data.name.lower()}.json"
    if b14_path.is_file() and not args.force_b14_search:
        print(f"  Using cached B14_final components: {b14_path}", flush=True)
        b14_sel = load_json(b14_path)
    else:
        print(f"  Selecting B14_final components strictly on validation for {data.name}...", flush=True)
        b14_sel = select_b14_components(data, cfg_best_enc, seeds=[0, 1, 2], ens_seeds_override=[0])
        save_json(b14_sel, b14_path)
        print(f"  B14_final selected: {b14_sel['selected']}", flush=True)

    b14_comp = copy.deepcopy(b14_sel["selected"])
    if args.no_pseudo:
        b14_comp["use_capped_pseudo"] = False
    if args.no_smoothing:
        b14_comp["use_smoothing"] = False
    if args.no_ensemble:
        b14_comp["use_ensemble"] = False

    # Prepare v2/v3 config for B11_final_lite comparator
    v2_enc_path = v2_dir / f"encoder_config_{data.name.lower()}.json"
    if v2_enc_path.is_file():
        cfg_v2_ds = apply_encoder_config(cfg_v2_raw, load_json(v2_enc_path))
        b10_sel = _load_or_select_b10(data, cfg_base, cfg_v2_ds, v2_dir)
        cfg_b11 = copy.deepcopy(cfg_v2_ds if b10_sel["selected"]["use_best_encoder"] else cfg_base)
    else:
        cfg_b11 = copy.deepcopy(cfg_v2_raw)
    cfg_b11["unified_latency"] = True
    cfg_b11["latency"] = {"warmup": 20, "repeats": 100}

    cfg_base_u = copy.deepcopy(cfg_base)
    cfg_base_u["unified_latency"] = True
    cfg_base_u["latency"] = {"warmup": 20, "repeats": 100}

    cfg_best_b1 = copy.deepcopy(cfg_best_enc)
    cfg_best_b1["embed"]["l2_norm"] = bool(b14_comp.get("use_l2_norm", True))

    cfg_b14 = copy.deepcopy(cfg_best_b1)
    cfg_b14["b14_components"] = b14_comp

    runs_path = out_dir / f"v4_runs_{data.name.lower()}.json"
    if runs_path.is_file() and not args.force_runs:
        print(f"  Using cached v4 runs: {runs_path}", flush=True)
        return load_json(runs_path)

    # 3. Run 10 seeds across all comparators & v4 variants
    ds_runs: list[dict[str, Any]] = []
    same_b11_cfg = (
        cfg_b11.get("gnn", {}).get("model") == cfg_base_u.get("gnn", {}).get("model")
        and cfg_b11.get("gnn", {}).get("hidden") == cfg_base_u.get("gnn", {}).get("hidden")
        and cfg_b11.get("gnn", {}).get("early_stop_metric") == cfg_base_u.get("gnn", {}).get("early_stop_metric")
        and cfg_b11.get("embed", {}).get("l2_norm") == cfg_base_u.get("embed", {}).get("l2_norm")
    )

    for s in args.seeds:
        print(f"  [{data.name}] running seed {s}...", flush=True)
        ctx_base = prepare_seed_context(cfg_base_u, data, s)
        ctx_b11 = ctx_base if same_b11_cfg else prepare_seed_context(cfg_b11, data, s)
        ctx_best = prepare_seed_context(cfg_best_b1, data, s)

        r_b1_base = run_single_seed(cfg_base_u, data, s, "B1", seed_ctx=ctx_base)
        r_b1_base["variant"] = "B1_base_gcn"
        ds_runs.append(r_b1_base)

        r_b1_best = run_single_seed(cfg_best_b1, data, s, "B1", seed_ctx=ctx_best)
        r_b1_best["variant"] = "B1_best_encoder"
        ds_runs.append(r_b1_best)

        r_b11 = run_single_seed(cfg_b11, data, s, "B11_final_lite", seed_ctx=ctx_b11)
        ds_runs.append(r_b11)

        r_b7 = run_single_seed(cfg_best_b1, data, s, "B7_hybrid", seed_ctx=ctx_best)
        ds_runs.append(r_b7)

        r_b13 = run_single_seed(cfg_best_b1, data, s, "B13_gated_hybrid", seed_ctx=ctx_best)
        ds_runs.append(r_b13)

        r_b12 = run_single_seed(cfg_best_b1, data, s, "B12_ensemble", seed_ctx=ctx_best)
        ds_runs.append(r_b12)

        r_b14 = run_single_seed(cfg_b14, data, s, "B14_final", seed_ctx=ctx_best)
        ds_runs.append(r_b14)

        # Free cached ensemble PyTorch modules between seeds to bound RAM.
        ctx_best.ensemble_models = None
        del ctx_base, ctx_b11, ctx_best

    ds_payload = {
        "dataset": data.name,
        "best_encoder": enc_search["best_encoder"],
        "encoder_b1_rows": enc_b1_rows,
        "b14_components": b14_sel,
        "runs": ds_runs,
    }
    save_json(ds_payload, runs_path)
    return ds_payload


# Aggregate all saved dataset results into summary tables, paired t-tests, and v4 plots.
def aggregate_and_report(datasets: list[str], seeds: list[int], out_dir: Path) -> None:
    all_enc_rows: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    all_paired: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    present_ds: list[str] = []

    for ds in datasets:
        p = out_dir / f"v4_runs_{ds.lower()}.json"
        if not p.is_file():
            continue
        present_ds.append(ds)
        pl = load_json(p)
        ds_runs = pl["runs"]
        ds_sum = summarize_v4_dataset_runs(ds_runs)
        ds_pt = compute_v4_paired_tests(ds_runs, pl["dataset"])
        all_enc_rows.extend(pl["encoder_b1_rows"])
        all_runs.extend(ds_runs)
        all_summaries.extend(ds_sum)
        all_paired.extend(ds_pt)
        detail[pl["dataset"]] = {
            "best_encoder": pl["best_encoder"],
            "b14_components": pl["b14_components"],
            "summary": ds_sum,
        }

    if not present_ds:
        return

    enc_flat = [{k: v for k, v in r.items() if k != "best_config"} for r in all_enc_rows]
    pd.DataFrame(enc_flat).to_csv(out_dir / "encoder_b1_comparison.csv", index=False)

    flat_runs = [
        {k: v for k, v in r.items() if k not in {"test_per_class_recall", "pseudo_per_class_counts", "b14_components"}}
        for r in all_runs
    ]
    pd.DataFrame(flat_runs).to_csv(out_dir / "v4_comparison_runs.csv", index=False)

    flat_sum = [{k: v for k, v in s.items() if "per_class_recall" not in k} for s in all_summaries]
    pd.DataFrame(flat_sum).to_csv(out_dir / "v4_comparison_summary.csv", index=False)
    pd.DataFrame(all_paired).to_csv(out_dir / "v4_paired_tests.csv", index=False)

    v4_json = {
        "datasets": present_ds,
        "seeds": seeds,
        "datasets_detail": detail,
        "summary": all_summaries,
        "paired_tests": all_paired,
        "runs": all_runs,
    }
    save_json(v4_json, out_dir / "v4_comparison.json")
    generate_v4_plots(out_dir, v4_json)
    md_v4 = build_v4_markdown_report(out_dir, v4_json)
    (out_dir / "report_tables.md").write_text(md_v4, encoding="utf-8")
    print(f"\nSaved complete v4 outputs and report_tables.md into {out_dir}", flush=True)


# Main entry point orchestrating encoder search, v4 runs, and aggregation.
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    v2_dir = Path(args.v2_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v2_dir.mkdir(parents=True, exist_ok=True)

    if not args.finalize_only:
        base_tmpl = load_config(args.base_config)
        v2_tmpl = load_config(args.v2_config)
        for ds in args.datasets:
            run_dataset_v4(ds, args, base_tmpl, v2_tmpl, out_dir, v2_dir)

    aggregate_and_report(ALL_V4_DATASETS, args.seeds, out_dir)


if __name__ == "__main__":
    main()
