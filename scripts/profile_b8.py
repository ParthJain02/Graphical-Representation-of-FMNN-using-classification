"""Profile B8_hybrid_pseudo time and peak memory breakdown across datasets."""

from __future__ import annotations

import argparse
import copy
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset
from src.embeddings import extract_softmax_probs
from src.fmnn import FMNN
from src.gnn import apply_encoder_config
from src.merge import gshr_merge
from src.metrics import classification_metrics
from src.pipeline import _build_pseudo_fit_set, _select_hybrid_lambda, prepare_seed_context
from src.prune import gshr_prune
from src.utils import load_config, load_json, save_json


# Measure elapsed time and peak traced memory for a single callable block.
def _measure_phase(fn: Any) -> tuple[Any, float, float]:
    tracemalloc.start()
    t0 = time.perf_counter()
    res = fn()
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return res, float(dt), float(peak / (1024 * 1024))


# Profile all stages of B8_hybrid_pseudo on a single dataset and seed.
def profile_dataset_b8(
    dataset_name: str,
    seed: int,
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = load_dataset(dataset_name, root=cfg.get("data", {}).get("root", "./data"))
    ctx = prepare_seed_context(cfg, data, seed)
    device = cfg.get("train", {}).get("device", "cpu")
    probs_all = extract_softmax_probs(ctx.gnn_model, data, device=device)

    val_idx = data.val_idx.cpu().numpy()
    y_val = data.y[data.val_idx].cpu().numpy()
    Z_val = ctx.Z_scaled[val_idx]
    probs_val = probs_all[val_idx]

    theta = float(cfg.get("fmnn", {}).get("theta", 0.3))
    gamma = float(cfg.get("fmnn", {}).get("gamma", 4.0))
    pseudo_cfg = cfg.get("pseudo", {})
    c_grid = [float(x) for x in pseudo_cfg.get("c_grid", [0.8, 0.9, 0.95])]
    tau_m_grid = [float(x) for x in cfg.get("merge", {}).get("tau_merge_grid", [0.3, 0.5, 0.7, 0.9])]
    tau_p_grid = [float(x) for x in cfg.get("prune", {}).get("tau_prune_grid", [0.0, 0.005, 0.01, 0.02])]
    lam_grid = [float(x) for x in cfg.get("hybrid", {}).get("lambda_grid", [0.0, 0.25, 0.5, 0.75, 1.0])]

    per_c_rows: list[dict[str, Any]] = []
    tot_pseudo_t = 0.0
    tot_fit_t = 0.0
    tot_merge_t = 0.0
    tot_prune_t = 0.0
    tot_hybrid_t = 0.0
    max_pseudo_mem = 0.0
    max_fit_mem = 0.0
    max_merge_mem = 0.0
    max_prune_mem = 0.0
    max_hybrid_mem = 0.0

    for c_cand in c_grid:
        print(f"  [{data.name} c={c_cand}] starting...", flush=True)
        (fit_idx, y_fit, _, pseudo_ctx, p_cnt), t_sel, mem_sel = _measure_phase(
            lambda: _build_pseudo_fit_set(data=data, probs_all=probs_all, threshold_c=c_cand)
        )
        Z_fit = ctx.Z_scaled[fit_idx]
        fit_ids = fit_idx.tolist()

        pseudo_base, t_fit, mem_fit = _measure_phase(
            lambda: FMNN(theta=theta, gamma=gamma).fit(Z_fit, y_fit, fit_ids)
        )
        print(
            f"    sel={t_sel:.3f}s fit={t_fit:.3f}s (fit_nodes={len(fit_idx)}, boxes={pseudo_base.num_boxes()})",
            flush=True,
        )

        # Pairwise similarity/merge checks over tau_merge_grid
        t_merge_c = 0.0
        mem_merge_c = 0.0
        merged_models: list[tuple[float, FMNN]] = []
        pairwise_checks_c = 0
        for tm in tau_m_grid:
            m_cfg = copy.deepcopy(cfg.get("merge", {}))
            m_cfg["tau_merge"] = tm
            m_cfg["theta"] = theta
            m_cfg["enabled"] = True
            n_b = pseudo_base.num_boxes()
            pairwise_checks_c += int(n_b * (n_b - 1) // 2)
            (m_model, _), dt_m, pm_m = _measure_phase(
                lambda mc=m_cfg: gshr_merge(pseudo_base, pseudo_ctx, mc)
            )
            t_merge_c += dt_m
            mem_merge_c = max(mem_merge_c, pm_m)
            merged_models.append((tm, m_model))

        # Prune sweeps over tau_prune_grid for each merged model
        t_prune_c = 0.0
        mem_prune_c = 0.0
        best_fmnn = pseudo_base
        best_val_f1 = -1.0
        best_boxes = pseudo_base.num_boxes()
        for tm, m_model in merged_models:
            for tp in tau_p_grid:
                p_cfg = copy.deepcopy(cfg.get("prune", {}))
                p_cfg["tau_prune"] = tp
                p_cfg["fast_mode"] = bool(cfg.get("prune", {}).get("fast_mode", True))
                (p_model, _), dt_p, pm_p = _measure_phase(
                    lambda mm=m_model, pc=p_cfg: gshr_prune(
                        mm, pseudo_ctx, pc, Z_fit, y_fit, fit_ids, Z_val, y_val
                    )
                )
                t_prune_c += dt_p
                mem_prune_c = max(mem_prune_c, pm_p)
                vf1 = float(classification_metrics(y_val, p_model.predict(Z_val), data.num_classes)["macro_f1"])
                if vf1 > best_f1_or_neg(best_val_f1) or (abs(vf1 - best_val_f1) < 1e-12 and p_model.num_boxes() < best_boxes):
                    best_val_f1 = vf1
                    best_fmnn = p_model
                    best_boxes = p_model.num_boxes()

        (_, t_hyb, mem_hyb) = _measure_phase(
            lambda bf=best_fmnn: _select_hybrid_lambda(
                probs_val=probs_val,
                mem_val=bf.class_scores(Z_val),
                y_val=y_val,
                num_classes=data.num_classes,
                lam_grid=lam_grid,
            )
        )

        row = {
            "dataset": data.name,
            "seed": seed,
            "c": c_cand,
            "pseudo_count": int(p_cnt),
            "fit_nodes": int(len(fit_idx)),
            "base_boxes": int(pseudo_base.num_boxes()),
            "pairwise_checks": pairwise_checks_c,
            "pseudo_sel_time_s": t_sel,
            "fmnn_fit_time_s": t_fit,
            "merge_time_s": t_merge_c,
            "prune_time_s": t_prune_c,
            "hybrid_eval_time_s": t_hyb,
            "total_time_s": t_sel + t_fit + t_merge_c + t_prune_c + t_hyb,
            "pseudo_sel_peak_mb": mem_sel,
            "fmnn_fit_peak_mb": mem_fit,
            "merge_peak_mb": mem_merge_c,
            "prune_peak_mb": mem_prune_c,
            "hybrid_peak_mb": mem_hyb,
            "max_phase_peak_mb": max(mem_sel, mem_fit, mem_merge_c, mem_prune_c, mem_hyb),
        }
        print(
            f"    merge={t_merge_c:.3f}s prune={t_prune_c:.3f}s hybrid={t_hyb:.3f}s peak={row['max_phase_peak_mb']:.2f}MB",
            flush=True,
        )
        per_c_rows.append(row)

        tot_pseudo_t += t_sel
        tot_fit_t += t_fit
        tot_merge_t += t_merge_c
        tot_prune_t += t_prune_c
        tot_hybrid_t += t_hyb
        max_pseudo_mem = max(max_pseudo_mem, mem_sel)
        max_fit_mem = max(max_fit_mem, mem_fit)
        max_merge_mem = max(max_merge_mem, mem_merge_c)
        max_prune_mem = max(max_prune_mem, mem_prune_c)
        max_hybrid_mem = max(max_hybrid_mem, mem_hyb)

        del merged_models, pseudo_base, Z_fit

    tot_all = tot_pseudo_t + tot_fit_t + tot_merge_t + tot_prune_t + tot_hybrid_t
    summary = {
        "dataset": data.name,
        "seed": seed,
        "pseudo_selection_time_s": tot_pseudo_t,
        "fmnn_fit_time_s": tot_fit_t,
        "pairwise_merge_time_s": tot_merge_t,
        "prune_time_s": tot_prune_t,
        "hybrid_eval_time_s": tot_hybrid_t,
        "total_refine_time_s": tot_all,
        "pseudo_selection_pct": 100.0 * tot_pseudo_t / max(tot_all, 1e-12),
        "fmnn_fit_pct": 100.0 * tot_fit_t / max(tot_all, 1e-12),
        "pairwise_merge_pct": 100.0 * tot_merge_t / max(tot_all, 1e-12),
        "prune_pct": 100.0 * tot_prune_t / max(tot_all, 1e-12),
        "hybrid_eval_pct": 100.0 * tot_hybrid_t / max(tot_all, 1e-12),
        "pseudo_sel_peak_mb": max_pseudo_mem,
        "fmnn_fit_peak_mb": max_fit_mem,
        "merge_peak_mb": max_merge_mem,
        "prune_peak_mb": max_prune_mem,
        "hybrid_peak_mb": max_hybrid_mem,
        "overall_peak_mb": max(max_pseudo_mem, max_fit_mem, max_merge_mem, max_prune_mem, max_hybrid_mem),
    }
    return per_c_rows, summary


# Helper comparison guard for validation F1 tie-breaking.
def best_f1_or_neg(val: float) -> float:
    return float(val)


# Parse arguments and run B8 profiling across requested datasets.
def main() -> None:
    parser = argparse.ArgumentParser(description="Profile B8_hybrid_pseudo time & peak memory.")
    parser.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer", "Chameleon"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--v2-config", type=str, default="configs/cora_v2.yaml")
    parser.add_argument("--out-dir", type=str, default="results/v3")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v2_tmpl = load_config(args.v2_config)

    all_c_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for ds in args.datasets:
        cfg = copy.deepcopy(v2_tmpl)
        cfg.setdefault("data", {})["name"] = ds
        enc_path = Path("results/v2") / f"encoder_config_{ds.lower()}.json"
        if enc_path.is_file():
            cfg = apply_encoder_config(cfg, load_json(enc_path))

        print(f"Profiling B8 on {ds} (seed={args.seed})...")
        c_rows, ds_sum = profile_dataset_b8(ds, args.seed, cfg)
        all_c_rows.extend(c_rows)
        summaries.append(ds_sum)

    print("\n=== B8_hybrid_pseudo Time & Memory Breakdown (Seed 0, summed across c in {0.8, 0.9, 0.95}) ===")
    hdr = (
        f"{'Dataset':<10} | {'PseudoSel(s/%)':<16} | {'FMNNFit(s/%)':<15} | "
        f"{'Merge(s/%)':<14} | {'Prune(s/%)':<15} | {'Total(s)':<9} | {'PeakMB (Stage)'}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        ps = f"{s['pseudo_selection_time_s']:.3f} ({s['pseudo_selection_pct']:.1f}%)"
        ff = f"{s['fmnn_fit_time_s']:.3f} ({s['fmnn_fit_pct']:.1f}%)"
        mg = f"{s['pairwise_merge_time_s']:.3f} ({s['pairwise_merge_pct']:.1f}%)"
        pr = f"{s['prune_time_s']:.3f} ({s['prune_pct']:.1f}%)"
        pk = f"{s['overall_peak_mb']:.2f} (prune={s['prune_peak_mb']:.2f}, fit={s['fmnn_fit_peak_mb']:.2f}, merge={s['merge_peak_mb']:.2f})"
        print(f"{s['dataset']:<10} | {ps:<16} | {ff:<15} | {mg:<14} | {pr:<15} | {s['total_refine_time_s']:<9.3f} | {pk}")

    pd.DataFrame(all_c_rows).to_csv(out_dir / "b8_profile_per_c.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "b8_profile.csv", index=False)
    save_json({"per_c": all_c_rows, "summary": summaries}, out_dir / "b8_profile.json")
    print(f"\nSaved B8 profile outputs to {out_dir}")


if __name__ == "__main__":
    main()
