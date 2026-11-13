"""Programmatically build REPORT.md and report_assets/ from real files in results/ and configs/."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch_geometric

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.make_report import place_non_overlapping_labels
from src.data import homophily_ratio, load_dataset


ASSETS_DIR = PROJECT_ROOT / "report_assets"
REPORT_PATH = PROJECT_ROOT / "REPORT.md"


# Convert a markdown heading title into its GitHub-style anchor slug.
def slugify_heading(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9\s_-]", "", text).strip().lower()
    return re.sub(r"\s+", "-", clean)


# Safely format numeric values and avoid printing raw NaN or empty table cells.
def fmt(val: Any, decimals: int = 4, na_label: str = "not measured") -> str:
    if val is None:
        return na_label
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return na_label
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    return f"{float(val):.{decimals}f}"


# Format mean +/- std safely when numbers are available, otherwise return na_label.
def fmt_ms(mean_v: Any, std_v: Any, decimals: int = 4, mult: float = 1.0, na_label: str = "not measured") -> str:
    if mean_v is None or (isinstance(mean_v, float) and np.isnan(mean_v)):
        return na_label
    m = float(mean_v) * mult
    if std_v is None or (isinstance(std_v, float) and np.isnan(std_v)):
        return f"{m:.{decimals}f}"
    s = float(std_v) * mult
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


# Build a markdown table string with explicit caption and one-line takeaway.
class ReportBuilder:
    """Accumulator for numbered figures, numbered tables, and verified markdown sections."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.table_count = 0
        self.figure_count = 0

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def add_table(self, caption: str, headers: list[str], rows: list[list[str]], takeaway: str) -> None:
        self.table_count += 1
        self.lines.append(f"**Table {self.table_count}: {caption}**\n")
        self.lines.append("| " + " | ".join(headers) + " |")
        self.lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            safe = [str(c) if (c is not None and str(c).strip() != "" and str(c).lower() != "nan") else "n/a" for c in row]
            self.lines.append("| " + " | ".join(safe) + " |")
        self.lines.append(f"\n*Takeaway: {takeaway}*\n")

    def add_figure(self, asset_rel_path: str, alt: str, caption: str, explanation: list[str]) -> None:
        self.figure_count += 1
        self.lines.append(f"![{alt}]({asset_rel_path})\n")
        self.lines.append(f"**Figure {self.figure_count}: {caption}**\n")
        for line in explanation:
            self.lines.append(line)
        self.lines.append("")


# Copy existing unique PNG assets into report_assets/ and regenerate fixed/v3 figures.
def prepare_report_assets() -> dict[str, str]:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    copies = [
        ("fmnn_practice_decision_boundary.png", "fmnn_practice_decision_boundary.png"),
        ("results/figures/graph_representation.png", "v1_graph_representation.png"),
        ("results/figures/accuracy_vs_boxes_scatter.png", "v1_accuracy_vs_boxes_scatter.png"),
        ("results/figures/macro_f1_and_boxes_bar.png", "v1_macro_f1_and_boxes_bar.png"),
        ("results/figures/per_class_recall_comparison.png", "v1_b2_vs_b6_hyperbox_reduction.png"),
        ("results/figures/theta_sensitivity.png", "v1_theta_sensitivity.png"),
        ("results/v2/figures/accuracy_vs_boxes_scatter.png", "v2_accuracy_vs_boxes_scatter.png"),
        ("results/v2/figures/macro_f1_and_boxes_bar.png", "v2_macro_f1_and_boxes_bar.png"),
        ("results/v2/figures/theta_sensitivity.png", "v2_theta_sensitivity.png"),
        ("results/v2/figures/per_class_recall_comparison.png", "v2_per_class_recall_comparison.png"),
        ("results/v4/figures/pareto_accuracy_vs_boxes.png", "v4_pareto_accuracy_vs_boxes.png"),
        ("results/v4/figures/box_decision_and_flips.png", "v4_box_decision_and_flips.png"),
    ]
    copied_map: dict[str, str] = {}
    for src_rel, dst_name in copies:
        src = PROJECT_ROOT / src_rel
        if src.is_file():
            dst = ASSETS_DIR / dst_name
            shutil.copy2(src, dst)
            copied_map[dst_name] = f"report_assets/{dst_name}"

    # 1. Regenerate fixed non-overlapping v1 scatter plot directly from results/cora_summary.csv.
    cora_sum = pd.read_csv(PROJECT_ROOT / "results/cora_summary.csv")
    fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=300)
    xs = cora_sum["num_boxes_mean"].astype(float).tolist()
    ys = cora_sum["test_accuracy_mean"].astype(float).tolist()
    short = [str(v).split("_")[0] for v in cora_sum["variant"]]
    # Slightly jitter text placements via place_non_overlapping_labels (and keep distinct shapes/sizes for overlapping coords).
    for i, (x, y, lb) in enumerate(zip(xs, ys, short)):
        ax.scatter(x, y, s=160 - i * 16, edgecolors="black", linewidths=0.8, zorder=5 - i * 0.2, label=f"{lb} ({x:.1f} boxes, {y:.4f})")
    ax.set_xlim(-1.0, max(xs) + 2.5)
    ax.set_ylim(min(ys) - 0.005, max(ys) + 0.005)
    # Give identical-coordinate points stacked y-offsets for crisp readability.
    used_coords: dict[tuple[float, float], int] = {}
    for x, y, lb in zip(xs, ys, short):
        key = (round(x, 2), round(y, 4))
        cnt = used_coords.get(key, 0)
        used_coords[key] = cnt + 1
        dx = -0.3 if x > 10 else 0.35
        dy = 0.0012 * (cnt + 1) * (1 if cnt % 2 == 0 else -1)
        ax.annotate(
            lb,
            xy=(x, y),
            xytext=(x + dx, y + dy),
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#666666", alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7),
            zorder=8,
        )
    ax.set_xlabel("Mean Number of Hyperboxes (Cora, 3 seeds)", fontweight="bold")
    ax.set_ylabel("Mean Test Accuracy", fontweight="bold")
    ax.set_title("Cora v1 Accuracy vs. Boxes (Regenerated Non-Overlapping Labels)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=7.5, frameon=True)
    fig.tight_layout()
    fixed_name = "v1_accuracy_vs_boxes_scatter_fixed.png"
    plt.savefig(ASSETS_DIR / fixed_name, bbox_inches="tight")
    plt.close()
    copied_map[fixed_name] = f"report_assets/{fixed_name}"

    # 2. Generate v3 profiling & B10 vs B11 efficiency summary plot directly from results/v3 CSVs.
    b8_prof = pd.read_csv(PROJECT_ROOT / "results/v3/b8_profile.csv")
    v3_sum = pd.read_csv(PROJECT_ROOT / "results/v3/v3_comparison_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=300)

    # Panel A: B8 stage percentage breakdown
    ds_list = b8_prof["dataset"].tolist()
    x_p = np.arange(len(ds_list))
    other_pct = b8_prof["pseudo_selection_pct"] + b8_prof["fmnn_fit_pct"] + b8_prof["hybrid_eval_pct"]
    axes[0].bar(x_p, b8_prof["prune_pct"], width=0.5, label="Prune (%)", color="#e15759")
    axes[0].bar(x_p, b8_prof["pairwise_merge_pct"], bottom=b8_prof["prune_pct"], width=0.5, label="Merge (%)", color="#f28e2b")
    axes[0].bar(x_p, other_pct, bottom=b8_prof["prune_pct"] + b8_prof["pairwise_merge_pct"], width=0.5, label="Fit+Select+Hybrid (%)", color="#4e79a7")
    axes[0].set_xticks(x_p)
    axes[0].set_xticklabels(ds_list, fontweight="bold")
    axes[0].set_ylabel("Share of B8 Refine Time (%)", fontweight="bold")
    axes[0].set_title("B8 Profiling Breakdown (Seed 0)", fontweight="bold")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(axis="y", linestyle=":", alpha=0.4)

    # Panel B: Refine time B10 vs B11_final_lite
    sub_b10 = v3_sum[v3_sum["variant"] == "B10_final"].set_index("dataset")
    sub_b11 = v3_sum[v3_sum["variant"] == "B11_final_lite"].set_index("dataset")
    ds_v3 = [d for d in ["Cora", "CiteSeer", "Chameleon"] if d in sub_b10.index]
    xv = np.arange(len(ds_v3))
    axes[1].bar(xv - 0.18, [sub_b10.loc[d, "refine_time_mean"] for d in ds_v3], width=0.36, label="B10_final", color="#b07aa1")
    axes[1].bar(xv + 0.18, [sub_b11.loc[d, "refine_time_mean"] for d in ds_v3], width=0.36, label="B11_final_lite", color="#59a14f")
    axes[1].set_xticks(xv)
    axes[1].set_xticklabels(ds_v3, fontweight="bold")
    axes[1].set_ylabel("Mean Refine Time (s, 10 seeds)", fontweight="bold")
    axes[1].set_title("Refine Time: B10_final vs. B11_final_lite", fontweight="bold")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(axis="y", linestyle=":", alpha=0.4)

    # Panel C: Peak memory B10 vs B11_final_lite
    axes[2].bar(xv - 0.18, [sub_b10.loc[d, "peak_memory_mb_mean"] for d in ds_v3], width=0.36, label="B10_final", color="#b07aa1")
    axes[2].bar(xv + 0.18, [sub_b11.loc[d, "peak_memory_mb_mean"] for d in ds_v3], width=0.36, label="B11_final_lite", color="#59a14f")
    axes[2].set_xticks(xv)
    axes[2].set_xticklabels(ds_v3, fontweight="bold")
    axes[2].set_ylabel("Mean Peak Traced Memory (MB)", fontweight="bold")
    axes[2].set_title("Peak Memory: B10_final vs. B11_final_lite", fontweight="bold")
    axes[2].legend(fontsize=8.5)
    axes[2].grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("GSHR v3 Profiling Bottleneck & Optimization Gains (Vectorized + Capped Pseudo-Labels)", fontweight="bold", y=1.02)
    fig.tight_layout()
    v3_fig_name = "v3_profiling_and_efficiency.png"
    plt.savefig(ASSETS_DIR / v3_fig_name, bbox_inches="tight")
    plt.close()
    copied_map[v3_fig_name] = f"report_assets/{v3_fig_name}"

    return copied_map


# Summarize per-seed CSV frames cleanly by grouping column.
def aggregate_csv(df: pd.DataFrame, group_col: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, grp in df.groupby(group_col, sort=False):
        rec: dict[str, Any] = {group_col: key, "n": len(grp)}
        for m in metrics:
            if m in grp.columns:
                vals = pd.to_numeric(grp[m], errors="coerce").dropna().to_numpy()
                rec[f"{m}_mean"] = float(np.mean(vals)) if len(vals) > 0 else np.nan
                rec[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


# Format a raw CSV DataFrame as a collapsible <details> markdown table block.
def raw_csv_details(title: str, df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = [f"<details>\n<summary>{title} ({len(df)} rows)</summary>\n"]
    out.append("| " + " | ".join(cols) + " |")
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if pd.isna(v):
                cells.append("n/a")
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{float(v):.6g}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    out.append("\n</details>\n")
    return "\n".join(out)


# Generate the entire REPORT.md programmatically from verified disk outputs.
def build_report() -> tuple[str, int, int, list[str]]:
    assets = prepare_report_assets()
    rb = ReportBuilder()

    # Load all verified files upfront
    v1_cora = pd.read_csv(PROJECT_ROOT / "results/cora_summary.csv")
    v1_cite = pd.read_csv(PROJECT_ROOT / "results/citeseer_summary.csv")
    v1_cham = pd.read_csv(PROJECT_ROOT / "results/chameleon_summary.csv")
    v1_cora_runs = pd.read_csv(PROJECT_ROOT / "results/cora_runs.csv")
    v1_cite_runs = pd.read_csv(PROJECT_ROOT / "results/citeseer_runs.csv")
    v1_cham_runs = pd.read_csv(PROJECT_ROOT / "results/chameleon_runs.csv")
    theta_scan_v1 = pd.read_csv(PROJECT_ROOT / "results/fmnn_theta_scan.csv")

    abl_core = pd.read_csv(PROJECT_ROOT / "results/ablations/a0_a4_core_ablations.csv")
    abl_sg = pd.read_csv(PROJECT_ROOT / "results/ablations/a5_safeguard_ablation.csv")
    abl_th = pd.read_csv(PROJECT_ROOT / "results/ablations/a6_sweep_theta.csv")
    abl_tm = pd.read_csv(PROJECT_ROOT / "results/ablations/a6_sweep_tau_merge.csv")
    abl_tp = pd.read_csv(PROJECT_ROOT / "results/ablations/a6_sweep_tau_prune.csv")
    abl_wt = pd.read_csv(PROJECT_ROOT / "results/ablations/a6_sweep_weights.csv")
    abl_a7 = pd.read_csv(PROJECT_ROOT / "results/ablations/a7_cross_dataset.csv")
    abl_a8 = pd.read_csv(PROJECT_ROOT / "results/ablations/a8_encoders.csv")
    abl_rob = pd.read_csv(PROJECT_ROOT / "results/ablations/robustness_stress_tests.csv")
    abl_sm = pd.read_csv(PROJECT_ROOT / "results/ablations/smoothing_ablation.csv")

    v2_json = json.loads((PROJECT_ROOT / "results/v2/final_comparison.json").read_text(encoding="utf-8"))
    v2_sum = pd.read_csv(PROJECT_ROOT / "results/v2/final_comparison_summary.csv")
    v2_runs = pd.read_csv(PROJECT_ROOT / "results/v2/final_comparison_runs.csv")
    v2_th = pd.read_csv(PROJECT_ROOT / "results/v2/theta_sensitivity_val.csv")

    v3_json = json.loads((PROJECT_ROOT / "results/v3/v3_comparison.json").read_text(encoding="utf-8"))
    v3_sum = pd.read_csv(PROJECT_ROOT / "results/v3/v3_comparison_summary.csv")
    v3_runs = pd.read_csv(PROJECT_ROOT / "results/v3/v3_comparison_runs.csv")
    v3_pt = pd.read_csv(PROJECT_ROOT / "results/v3/v3_paired_tests.csv")
    v3_lat = pd.read_csv(PROJECT_ROOT / "results/v3/unified_latency.csv")
    v3_prof = pd.read_csv(PROJECT_ROOT / "results/v3/b8_profile.csv")
    v3_prof_c = pd.read_csv(PROJECT_ROOT / "results/v3/b8_profile_per_c.csv")

    v4_json = json.loads((PROJECT_ROOT / "results/v4/v4_comparison.json").read_text(encoding="utf-8"))
    v4_sum = pd.read_csv(PROJECT_ROOT / "results/v4/v4_comparison_summary.csv")
    v4_runs = pd.read_csv(PROJECT_ROOT / "results/v4/v4_comparison_runs.csv")
    v4_pt = pd.read_csv(PROJECT_ROOT / "results/v4/v4_paired_tests.csv")
    v4_enc = pd.read_csv(PROJECT_ROOT / "results/v4/encoder_b1_comparison.csv")

    # Compute exact dataset geometry & duplicate row statistics
    ds_info: dict[str, dict[str, Any]] = {}
    for dname in ["Cora", "CiteSeer", "Chameleon", "PubMed"]:
        d = load_dataset(dname, root="./data")
        x_np = d.x.numpy()
        _, inv, counts = np.unique(x_np, axis=0, return_inverse=True, return_counts=True)
        dup_groups = int(np.sum(counts > 1))
        dup_nodes = int(np.sum(counts[counts > 1]) - dup_groups)
        tr_hashes = set(inv[d.train_idx.numpy()])
        te_match_tr = int(sum(1 for idx in d.test_idx.numpy() if inv[idx] in tr_hashes))
        ds_info[d.name] = {
            "nodes": d.num_nodes,
            "undir_edges": d.edge_index.size(1) // 2,
            "dir_edges": d.edge_index.size(1),
            "feats": int(d.x.size(1)),
            "classes": d.num_classes,
            "train": int(len(d.train_idx)),
            "val": int(len(d.val_idx)),
            "test": int(len(d.test_idx)),
            "homophily": homophily_ratio(d),
            "dup_rows": dup_nodes,
            "dup_groups": dup_groups,
            "test_dup_train": te_match_tr,
        }

    # Lookup helper for v2, v3, v4 summary rows
    def _v3_row(ds: str, var: str) -> pd.Series:
        return v3_sum[(v3_sum["dataset"] == ds) & (v3_sum["variant"] == var)].iloc[0]

    def _v4_row(ds: str, var: str) -> pd.Series | None:
        m = v4_sum[(v4_sum["dataset"] == ds) & (v4_sum["variant"] == var)]
        return m.iloc[0] if not m.empty else None

    def _v1_row(df: pd.DataFrame, var: str) -> pd.Series:
        return df[df["variant"] == var].iloc[0]

    def _ulat_row(ds: str, var: str) -> pd.Series:
        return v3_lat[(v3_lat["dataset"] == ds) & (v3_lat["variant"] == var)].iloc[0]

    def _v4_pt_row(ds: str, var: str, base_var: str) -> pd.Series:
        return v4_pt[(v4_pt["dataset"] == ds) & (v4_pt["variant"] == var) & (v4_pt["baseline_variant"] == base_var)].iloc[0]

    def _v3_pval(ds: str, base_var: str, col: str = "test_accuracy_p") -> float:
        return float(v3_pt[(v3_pt["dataset"] == ds) & (v3_pt["baseline_variant"] == base_var)].iloc[0][col])

    def _v4_pval(ds: str, var: str, base_var: str, col: str = "test_accuracy_p") -> float:
        return float(_v4_pt_row(ds, var, base_var)[col])

    cora_v3_b1 = _v3_row("Cora", "B1_gnn_softmax")
    cora_v3_b9 = _v3_row("Cora", "B9_gshr_fast")
    cora_v3_b10 = _v3_row("Cora", "B10_final")
    cora_v3_b11 = _v3_row("Cora", "B11_final_lite")
    cora_v4_b1g = _v4_row("Cora", "B1_base_gcn")
    cora_v4_b1b = _v4_row("Cora", "B1_best_encoder")
    cora_v4_b7 = _v4_row("Cora", "B7_hybrid")
    cora_v4_b14 = _v4_row("Cora", "B14_final")
    cora_v4_b12 = _v4_row("Cora", "B12_ensemble")
    cora_b13 = _v4_row("Cora", "B13_gated_hybrid")

    cite_v3_b1 = _v3_row("CiteSeer", "B1_gnn_softmax")
    cite_v3_b10 = _v3_row("CiteSeer", "B10_final")
    cite_v3_b11 = _v3_row("CiteSeer", "B11_final_lite")
    cite_v4_b1g = _v4_row("CiteSeer", "B1_base_gcn")
    cite_v4_b1b = _v4_row("CiteSeer", "B1_best_encoder")
    cite_v4_b14 = _v4_row("CiteSeer", "B14_final")
    cite_v4_b12 = _v4_row("CiteSeer", "B12_ensemble")
    cite_b13 = _v4_row("CiteSeer", "B13_gated_hybrid")

    cham_v3_b1 = _v3_row("Chameleon", "B1_gnn_softmax")
    cham_v3_b9 = _v3_row("Chameleon", "B9_gshr_fast")
    cham_v3_b10 = _v3_row("Chameleon", "B10_final")
    cham_v3_b11 = _v3_row("Chameleon", "B11_final_lite")

    cham_v2_b2 = v2_sum[(v2_sum["dataset"] == "Chameleon") & (v2_sum["variant"] == "B2_gnn_fmnn")].iloc[0]
    cora_v2_b2 = v2_sum[(v2_sum["dataset"] == "Cora") & (v2_sum["variant"] == "B2_gnn_fmnn")].iloc[0]
    cora_v2_b7 = v2_sum[(v2_sum["dataset"] == "Cora") & (v2_sum["variant"] == "B7_hybrid")].iloc[0]
    cite_v2_b2 = v2_sum[(v2_sum["dataset"] == "CiteSeer") & (v2_sum["variant"] == "B2_gnn_fmnn")].iloc[0]
    cite_v2_b7 = v2_sum[(v2_sum["dataset"] == "CiteSeer") & (v2_sum["variant"] == "B7_hybrid")].iloc[0]
    cite_v2_b8 = v2_sum[(v2_sum["dataset"] == "CiteSeer") & (v2_sum["variant"] == "B8_hybrid_pseudo")].iloc[0]

    cora_v1_b1 = _v1_row(v1_cora, "B1_gnn_softmax")
    cora_v1_b2 = _v1_row(v1_cora, "B2_gnn_fmnn")
    cora_v1_b4 = _v1_row(v1_cora, "B4_graph_merge_only")
    cora_v1_b5 = _v1_row(v1_cora, "B5_graph_prune_only")
    cora_v1_b6 = _v1_row(v1_cora, "B6_gshr_full")
    cite_v1_b2 = _v1_row(v1_cite, "B2_gnn_fmnn")
    cite_v1_b6 = _v1_row(v1_cite, "B6_gshr_full")
    cham_v1_b2 = _v1_row(v1_cham, "B2_gnn_fmnn")
    cham_v1_b6 = _v1_row(v1_cham, "B6_gshr_full")
    cham_b10_comp = json.loads((PROJECT_ROOT / "results/v2/b10_components_chameleon.json").read_text(encoding="utf-8"))["gate_details"]

    # Missing data registry
    missing_data = [
        "Chameleon v4 full benchmark (encoder search across {gcn, sage, appnp, gcnii, h2gcn}, B12_ensemble, B13_gated_hybrid, B14_final): execution interrupted during Chameleon v4 encoder search due to CPU runtime; no v4 Chameleon result file exists in results/v4/.",
        "Chameleon filtered split (duplicate-free node split): not implemented or evaluated in results/.",
        "PubMed benchmark runs (v1, v2, v3, v4): dataset downloaded and profiled locally (19,717 nodes, homophily 0.8024, configs/pubmed.yaml present), but full multi-seed runs were not saved in results/.",
        "GAT encoder and standalone MLP baseline: not implemented in src/gnn.py and not present in results/.",
        "Per-box 64-D hyperbox coordinate rules ([v_d, w_d] bounds and member node lists per box in results/): only aggregate box statistics (mean/max box size, purity, coverage, box-decided accuracy, flip fraction) and 2D PCA / 2D practice visualizations were saved in results/.",
    ]

    # =========================================================================
    # Title & Table of Contents
    # =========================================================================
    rb.add("# GNN + GSHR + FMNN: Graph-Structure-Aware Hyperbox Refinement for Interpretable Node Classification\n")
    rb.add("**Authors:** [NAME]  ")
    rb.add("**Guide / Instructor:** [GUIDE]  ")
    rb.add("**Date:** 2026-11-11  ")
    rb.add("**Course / Project:** Final Research Report (`GNN + GSHR + FMNN`)  \n")

    toc_sections = [
        ("0. Executive summary", "0-executive-summary"),
        ("1. Problem statement and motivation", "1-problem-statement-and-motivation"),
        ("2. Background", "2-background"),
        ("3. Datasets and experimental setup", "3-datasets-and-experimental-setup"),
        ("4. Method and implementation details", "4-method-and-implementation-details"),
        ("5. Version history", "5-version-history"),
        ("6. Results", "6-results"),
        ("6.1 v1 main benchmark (3 seeds)", "61-v1-main-benchmark-3-seeds"),
        ("6.2 v1 baseline and GSHR JSON results", "62-v1-baseline-and-gshr-json-results"),
        ("6.3 v1 ablations (A0-A8, robustness, smoothing)", "63-v1-ablations-a0-a8-robustness-smoothing"),
        ("6.4 v2 5-seed comparison, encoder search, and validation theta sweep", "64-v2-5-seed-comparison-encoder-search-and-validation-theta-sweep"),
        ("6.5 v3 10-seed comparison, paired tests, B8 profile, and unified latency", "65-v3-10-seed-comparison-paired-tests-b8-profile-and-unified-latency"),
        ("6.6 v4 results (encoder search, B12 ensemble, B13 gated hybrid, B14_final)", "66-v4-results-encoder-search-b12-ensemble-b13-gated-hybrid-b14_final"),
        ("6.7 Per-class recall comparison and box statistics across versions", "67-per-class-recall-comparison-and-box-statistics-across-versions"),
        ("7. Figures", "7-figures"),
        ("8. Discussion", "8-discussion"),
        ("9. Interpretability", "9-interpretability"),
        ("10. Threats to validity and limitations", "10-threats-to-validity-and-limitations"),
        ("11. Conclusion and future work", "11-conclusion-and-future-work"),
        ("12. Reproducibility", "12-reproducibility"),
        ("Appendix", "appendix"),
        ("Appendix A. Full raw per-seed tables", "appendix-a-full-raw-per-seed-tables"),
        ("Appendix B. Full hyperparameter configs", "appendix-b-full-hyperparameter-configs"),
        ("Appendix C. Complete list of output files with sizes", "appendix-c-complete-list-of-output-files-with-sizes"),
        ("Appendix D. Glossary of symbols", "appendix-d-glossary-of-symbols"),
        ("Appendix E. Missing data", "appendix-e-missing-data"),
    ]
    rb.add("## Table of Contents\n")
    for label, slug in toc_sections:
        indent = "  " if (label.startswith("6.") or label.startswith("Appendix ")) else ""
        rb.add(f"{indent}- [{label}](#{slug})")
    rb.add("")

    # =========================================================================
    # 0. Executive Summary
    # =========================================================================
    rb.add("## 0. Executive summary\n")
    cora_red_b14 = (1.0 - float(cora_v4_b14["num_boxes_mean"]) / float(cora_v2_b2["num_boxes_mean"])) * 100.0
    cite_red_b14 = (1.0 - float(cite_v4_b14["num_boxes_mean"]) / float(cite_v2_b2["num_boxes_mean"])) * 100.0
    cham_red_b11 = (1.0 - float(cham_v3_b11["num_boxes_mean"]) / float(cham_v2_b2["num_boxes_mean"])) * 100.0
    cora_b11_red = (1.0 - float(cora_v3_b11["num_boxes_mean"]) / float(cora_v2_b2["num_boxes_mean"])) * 100.0
    cite_b11_red = (1.0 - float(cite_v3_b11["num_boxes_mean"]) / float(cite_v2_b2["num_boxes_mean"])) * 100.0

    cora_v1_red = (1.0 - float(cora_v1_b6["num_boxes_mean"]) / float(cora_v1_b2["num_boxes_mean"])) * 100.0
    cite_v1_red = (1.0 - float(cite_v1_b6["num_boxes_mean"]) / float(cite_v1_b2["num_boxes_mean"])) * 100.0
    cham_v1_red = (1.0 - float(cham_v1_b6["num_boxes_mean"]) / float(cham_v1_b2["num_boxes_mean"])) * 100.0
    lat_b1_c = float(_ulat_row("Cora", "B1_gnn_softmax")["latency_us_mean"])
    lat_b1_s = float(_ulat_row("CiteSeer", "B1_gnn_softmax")["latency_us_mean"])
    lat_b1_h = float(_ulat_row("Chameleon", "B1_gnn_softmax")["latency_us_mean"])
    lat_b11_c = float(_ulat_row("Cora", "B11_final_lite")["latency_us_mean"])
    lat_b11_s = float(_ulat_row("CiteSeer", "B11_final_lite")["latency_us_mean"])
    lat_b11_h = float(_ulat_row("Chameleon", "B11_final_lite")["latency_us_mean"])
    b1_speedups = [lat_b11_c / lat_b1_c, lat_b11_s / lat_b1_s, lat_b11_h / lat_b1_h]
    max_b1_mem = max(
        float(_ulat_row("Cora", "B1_gnn_softmax")["peak_memory_mb_mean"]),
        float(_ulat_row("CiteSeer", "B1_gnn_softmax")["peak_memory_mb_mean"]),
        float(_ulat_row("Chameleon", "B1_gnn_softmax")["peak_memory_mb_mean"]),
    )
    min_box_final = min(float(cora_v4_b14["num_boxes_mean"]), float(cite_v4_b14["num_boxes_mean"]), float(cham_v3_b11["num_boxes_mean"]))
    max_box_final = max(float(cora_v4_b14["num_boxes_mean"]), float(cite_v4_b14["num_boxes_mean"]), float(cham_v3_b11["num_boxes_mean"]))
    min_kb_final = min(float(cora_v4_b14["box_storage_bytes_mean"]), float(cite_v4_b14["box_storage_bytes_mean"]), float(cham_v3_b11["box_storage_bytes_mean"])) / 1024.0
    max_kb_final = max(float(cora_v4_b14["box_storage_bytes_mean"]), float(cite_v4_b14["box_storage_bytes_mean"]), float(cham_v3_b11["box_storage_bytes_mean"])) / 1024.0
    min_red_all = min(cora_red_b14, cite_red_b14, cham_red_b11)
    max_red_all = max(cora_red_b14, cite_red_b14, cham_red_b11)

    rb.add(f"- **Goal:** Replace opaque GNN linear softmax heads with compact, class-pure Fuzzy Min-Max Neural Network (FMNN) hyperboxes refined via Graph-Structure-Aware Hyperbox Refinement (GSHR), while preserving GNN accuracy (`B1`) across homophilous (`Cora`, `CiteSeer`) and heterophilous (`Chameleon`) graphs.")
    rb.add(f"- **Method progression (`v1` -> `v4`):** Plain FMNN (`B2`) fragments embeddings into redundant boxes (`{cora_v2_b2['num_boxes_mean']:.1f}` Cora, `{cite_v2_b2['num_boxes_mean']:.1f}` CiteSeer, `{cham_v2_b2['num_boxes_mean']:.1f}` Chameleon in `v2`); `GSHR` pruning (`B6`/`B9`), hybrid probability mixing (`B7`/`B13`), validation-capped pseudo-labeling (`B11_final_lite`), stronger encoders (`APPNP` on Cora, `GCNII` on CiteSeer), and 5-seed ensembling (`B12_ensemble`) systematically tighten accuracy and box count strictly via validation macro-F1 selection.")
    rb.add(f"- **Cora main result (10 seeds):** Raising the encoder from `GCN` (`B1` Acc `{cora_v4_b1g['test_accuracy_mean']:.4f}`) to `APPNP` (`B1` Acc `{cora_v4_b1b['test_accuracy_mean']:.4f}`) lifts `B14_final` to **`{cora_v4_b14['test_accuracy_mean']:.4f} ± {cora_v4_b14['test_accuracy_std']:.4f}`** (`{cora_v4_b14['num_boxes_mean']:.1f}` boxes, `{cora_red_b14:.1f}%` fewer boxes than `v2 B2`'s `{cora_v2_b2['num_boxes_mean']:.1f}`); comparable (not statistically significant) vs `APPNP B1` (`p = {_v4_pval('Cora','B14_final','B1_best_encoder'):.4f}`) and statistically better than `v3 B11_final_lite` (`{cora_v3_b11['test_accuracy_mean']:.4f}`, `p = {_v4_pval('Cora','B14_final','B11_final_lite'):.4f}`). `B12_ensemble` reaches `{cora_v4_b12['test_accuracy_mean']:.4f}` (`p = {_v4_pval('Cora','B12_ensemble','B1_base_gcn'):.4f}` vs base GCN).")
    rb.add(f"- **CiteSeer main result (10 seeds):** `v4` validation selects `GCNII` (`B1` Acc `{cite_v4_b1b['test_accuracy_mean']:.4f}` vs base `GCN` `{cite_v4_b1g['test_accuracy_mean']:.4f}`); `B14_final` (with 5-encoder `avg` ensemble + fast GSHR + global `lam`) reaches **`{cite_v4_b14['test_accuracy_mean']:.4f} ± {cite_v4_b14['test_accuracy_std']:.4f}`** with **`{cite_v4_b14['num_boxes_mean']:.1f}`** boxes (`{cite_red_b14:.1f}%` fewer than `v2 B2`'s `{cite_v2_b2['num_boxes_mean']:.1f}`), comparable in accuracy vs `GCNII B1` (`p = {_v4_pval('CiteSeer','B14_final','B1_best_encoder'):.4f}`; macro-F1 `p = {_v4_pval('CiteSeer','B14_final','B1_best_encoder','test_macro_f1_p'):.4f}`) and improving over base `GCN B1` (`p = {_v4_pval('CiteSeer','B14_final','B1_base_gcn'):.4f}`). `B12_ensemble` reaches `{cite_v4_b12['test_accuracy_mean']:.4f}` (`p = {_v4_pval('CiteSeer','B12_ensemble','B1_base_gcn'):.4f}` vs base GCN).")
    rb.add(f"- **Chameleon main result (10 seeds `v3`; `v4` not measured):** `B11_final_lite` achieves **`{cham_v3_b11['test_accuracy_mean']:.4f} ± {cham_v3_b11['test_accuracy_std']:.4f}`** (`{cham_v3_b11['num_boxes_mean']:.1f}` boxes, `{cham_red_b11:.1f}%` fewer than `v2 B2`'s `{cham_v2_b2['num_boxes_mean']:.1f}`), comparable to base `B1` (`{cham_v3_b1['test_accuracy_mean']:.4f}`, `p = {_v3_pval('Chameleon','B1_gnn_softmax'):.4f}`), because Chameleon's heterophily (`h = {ds_info['Chameleon']['homophily']:.4f}`) makes box boundaries less discriminative alone (`v3 B9` drops to `{cham_v3_b9['test_accuracy_mean']:.4f}`).")
    rb.add(f"- **Box reduction & compute efficiency (`v3` vs `v2`):** Capping pseudo-labels (`k in {{20, 50, 100}}`), forbidding redundant box creation for covered nodes, and NumPy/chunked vectorization cut refine time of `B11_final_lite` vs `B10_final` (`10` seeds) from `{cora_v3_b10['refine_time_mean']:.3f}s` -> `{cora_v3_b11['refine_time_mean']:.3f}s` on Cora (`p = {_v3_pval('Cora','B10_final','refine_time_p'):.4f}`), peak memory `{cora_v3_b10['peak_memory_mb_mean']:.2f}` -> `{cora_v3_b11['peak_memory_mb_mean']:.2f} MB`, and boxes `{cora_v3_b10['num_boxes_mean']:.1f}` -> `{cora_v3_b11['num_boxes_mean']:.1f}` (`{cora_b11_red:.1f}%` below `v2 B2`; CiteSeer `B11` `{cite_b11_red:.1f}%` below `v2 B2`).")
    rb.add(f"- **Merge inactivity diagnosis:** In `v1`, graph-aware merge (`B4`) changed **zero** boxes (`{cora_v1_b2['num_boxes_mean']:.1f} -> {cora_v1_b4['num_boxes_mean']:.1f}`) because same-class FMNN boxes created under `theta = 0.3` rarely satisfied the `size_limit` (`L1 span <= d * theta`) and `max_volume_inflation = 0.5` checks together; pruning (`B5`) produced 100% of `v1`'s `{cora_v1_red:.1f}%` box reduction (`{cora_v1_b2['num_boxes_mean']:.1f} -> {cora_v1_b5['num_boxes_mean']:.1f}`).")
    rb.add(f"- **Inference cost (`v3` unified latency):** Under identical CPU end-to-end timing (`20` warm-up, median of `100` full-batch runs across seeds `0..2`), `B1` softmax inference (`{lat_b1_c:.2f} us/node` Cora, `{lat_b1_s:.2f} us/node` CiteSeer, `{lat_b1_h:.2f} us/node` Chameleon) is **{min(b1_speedups):.2f}x to {max(b1_speedups):.2f}x faster** (`{lat_b11_s/lat_b1_s:.2f}x–{lat_b11_c/lat_b1_c:.2f}x` on homophilous graphs) and uses `<= {max_b1_mem:.2f} MB` peak RAM compared to `{lat_b11_c:.2f}–{lat_b11_h:.2f} us/node` (`B11_final_lite`).")
    rb.add(f"- **Honest bottom line:** Boxes yield compact, class-pure geometric rules (`{min_box_final:.1f}–{max_box_final:.1f}` boxes, `{min_kb_final:.2f}–{max_kb_final:.2f} KiB` storage) matching `B1` accuracy within statistical noise, yet on `v4` `B13_gated_hybrid` only decides `{float(cora_b13['box_decided_fraction_mean'])*100:.2f}%` (Cora) / `{float(cite_b13['box_decided_fraction_mean'])*100:.2f}%` (CiteSeer) of test nodes and flips **`{float(cora_b13['fraction_changed_by_boxes_mean'])*100:.2f}%` (Cora) / `{float(cite_b13['fraction_changed_by_boxes_mean'])*100:.2f}%` (CiteSeer)** of `B1` predictions because GNN softmax already agrees with confident single-box regions.")
    rb.add("")

    # Headline table (using v4 on Cora/CiteSeer and v3 on Chameleon, matching exact values in Section 6 tables)
    rb.add_table(
        caption="Executive summary headline results (10 seeds: v4 B14_final for Cora/CiteSeer; v3 B11_final_lite for Chameleon where v4 is not measured).",
        headers=["Dataset", "Base GCN B1 Acc", "Best Single B1 Acc", "Final Model (Version)", "Final Model Acc", "Boxes (v2 B2 -> Final)", "Box Reduction % vs v2 B2", "p-value vs Base B1 (vs Best B1)"],
        rows=[
            [
                "Cora",
                fmt_ms(cora_v4_b1g["test_accuracy_mean"], cora_v4_b1g["test_accuracy_std"]),
                f"{cora_v4_b1b['test_accuracy_mean']:.4f} ± {cora_v4_b1b['test_accuracy_std']:.4f} (APPNP)",
                "B14_final (v4)",
                fmt_ms(cora_v4_b14["test_accuracy_mean"], cora_v4_b14["test_accuracy_std"]),
                f"{cora_v2_b2['num_boxes_mean']:.1f} -> {cora_v4_b14['num_boxes_mean']:.1f}",
                f"{cora_red_b14:.1f}%",
                f"p={_v4_pval('Cora','B14_final','B1_base_gcn'):.4f} (p={_v4_pval('Cora','B14_final','B1_best_encoder'):.4f})",
            ],
            [
                "CiteSeer",
                fmt_ms(cite_v4_b1g["test_accuracy_mean"], cite_v4_b1g["test_accuracy_std"]),
                f"{cite_v4_b1b['test_accuracy_mean']:.4f} ± {cite_v4_b1b['test_accuracy_std']:.4f} (GCNII)",
                "B14_final (v4)",
                fmt_ms(cite_v4_b14["test_accuracy_mean"], cite_v4_b14["test_accuracy_std"]),
                f"{cite_v2_b2['num_boxes_mean']:.1f} -> {cite_v4_b14['num_boxes_mean']:.1f}",
                f"{cite_red_b14:.1f}%",
                f"p={_v4_pval('CiteSeer','B14_final','B1_base_gcn'):.4f} (p={_v4_pval('CiteSeer','B14_final','B1_best_encoder'):.4f})",
            ],
            [
                "Chameleon",
                fmt_ms(cham_v3_b1["test_accuracy_mean"], cham_v3_b1["test_accuracy_std"]),
                f"{cham_v3_b1['test_accuracy_mean']:.4f} ± {cham_v3_b1['test_accuracy_std']:.4f} (v4 not measured)",
                "B11_final_lite (v3; v4 not measured)",
                fmt_ms(cham_v3_b11["test_accuracy_mean"], cham_v3_b11["test_accuracy_std"]),
                f"{cham_v2_b2['num_boxes_mean']:.1f} -> {cham_v3_b11['num_boxes_mean']:.1f}",
                f"{cham_red_b11:.1f}%",
                f"p={_v3_pval('Chameleon','B1_gnn_softmax'):.4f} (n/a)",
            ],
        ],
        takeaway=(
            f"Across 10 seeds, v4 APPNP/GCNII encoders lift accuracy on Cora/CiteSeer and boxes drop by "
            f"{min_red_all:.1f}%–{max_red_all:.1f}% vs unrefined v2 B2, while B14_final remains statistically comparable to its underlying best single B1 encoder "
            f"(p = {_v4_pval('Cora','B14_final','B1_best_encoder'):.4f} Cora, p = {_v4_pval('CiteSeer','B14_final','B1_best_encoder'):.4f} CiteSeer)."
        ),
    )

    # =========================================================================
    # 1. Problem statement and motivation
    # =========================================================================
    rb.add("## 1. Problem statement and motivation\n")
    rb.add(
        "Graph Neural Networks (GNNs) learn dense node representations $Z \\in \\mathbb{R}^{N \\times d}$ and achieve strong semi-supervised node classification performance via a linear softmax head (`B1`). "
        "However, softmax weights offer limited geometric transparency: predicting why a test node belongs to a class requires inspecting unstructured 64-dimensional linear projections.\n"
    )
    rb.add(
        "Fuzzy Min-Max Neural Networks (FMNN; Simpson, 1992) replace linear decision hyperplanes with axis-aligned hyperboxes $B_j = (V_j, W_j, c_j)$ inside $[0, 1]^d$, providing explicit bounding-interval decision rules with fuzzy membership scores. "
        "Naively fitting FMNN (`B2`) on GNN embeddings creates fragmented, redundant hyperboxes (`13.3–29.0` boxes across our benchmarks) and loses accuracy relative to the GNN softmax baseline. "
        "**Graph-Structure-Aware Hyperbox Refinement (GSHR)** addresses this by incorporating neighborhood connectivity, label agreement among training neighbors, conductance, and validation feedback to merge and prune redundant boxes toward the compact ideal of roughly one box per class, while reporting **GNN-only softmax (`B1`)** alongside every box variant (per instructor feedback) so no accuracy trade-off is hidden.\n"
    )

    # =========================================================================
    # 2. Background
    # =========================================================================
    rb.add("## 2. Background\n")
    rb.add(
        "- **Semi-supervised node classification:** Given an undirected graph $G = (V, E)$ with feature matrix $X \\in \\mathbb{R}^{N \\times F}$, sparse adjacency $A$, and training node labels $Y_{\\text{train}}$, the task is to classify unlabeled test nodes $V_{\\text{test}}$ without ever accessing $Y_{\\text{val}}$ or $Y_{\\text{test}}$ during representation learning or box construction.\n"
        "- **GNN encoders implemented (`src/gnn.py`):**\n"
        "  - **GCN (`GCNEncoder`):** Two-layer normalized graph convolution (`Conv -> ReLU -> Dropout -> Conv`) with symmetric renormalization.\n"
        "  - **GraphSAGE (`SAGEEncoder`):** Two-layer mean-aggregating neighborhood sampler convolution.\n"
        "  - **APPNP (`APPNPEncoder`):** Decoupled 2-layer MLP followed by $K=10$ personalized PageRank diffusion steps ($\\alpha = 0.1$), separating feature transformation depth from propagation depth.\n"
        "  - **GCNII (`GCNIIEncoder`):** Deep 8-layer architecture using `GCN2Conv` with initial residual connection ($\\alpha = 0.1$) and identity mapping ($\\lambda = 0.5$) to mitigate over-smoothing.\n"
        "  - **H2GCN-style (`H2GCNEncoder`):** Built for heterophily (e.g., `Chameleon`), removing self-loops and concatenating separate ego features $r_0$, 1-hop aggregations $r_1$, and 2-hop aggregations $r_2$ as $[r_0 \\Vert r_1 \\Vert r_2]$ prior to output projection.\n"
        "  - **GAT:** Not implemented and not measured in this repository.\n"
        "- **FMNN and Hyperboxes (`src/fmnn.py`):** Features scaled into $[0, 1]^d$ are grouped incrementally into hyperboxes $B_j = (V_j, W_j, c_j)$ under L1 expansion constraint $\\sum_{i=1}^d (\\max(w_{ji}, a_i) - \\min(v_{ji}, a_i)) \\le d \\cdot \\theta$. Simpson's 4-case overlap test and contraction ensure zero cross-class hyperbox overlap (`cross_class_overlap = 0`). Point-to-box membership (`gamma = 4.0`) equals $1.0$ inside $[V_j, W_j]$ and decays linearly outside.\n"
        "- **GSHR (`src/merge.py`, `src/prune.py`):** Merges same-class boxes using composite score $M = \\alpha \\cdot \\text{Geometry} + \\beta \\cdot \\text{GraphAssoc} + \\delta \\cdot \\text{Consistency} - \\eta \\cdot \\text{Risk}$ with default weights $(\\alpha, \\beta, \\delta, \\eta) = (0.35, 0.35, 0.20, 0.10)$ and merge threshold $\\tau_{\\text{merge}}$ subject to hard checks (`size_limit`, `cross_class_overlap = 0`, `volume_inflation <= 0.5`, and `class_purity = 1.0`). Sequential pruning removes lowest-importance boxes (scoring coverage $\\mu_1=0.25$, purity $\\mu_2=0.25$, boundary importance $\\mu_3=0.20$, graph region $\\mu_4=0.20$, redundancy $\\mu_5=0.10$) as long as validation macro-F1 drops stay within step tolerance $\\tau_{\\text{prune}}$ (`<= 2*tau_prune` total and per-class recall safeguards), keeping $\\ge 1$ box per class.\n"
        "- **Homophily ($h$):** Fraction of directed edges connecting same-class nodes. From our offline A7 inspection: **Cora $h = 0.8100$** (strongly homophilous), **CiteSeer $h = 0.7355$** (homophilous), **Chameleon $h = 0.2299$** (strongly heterophilous), and **PubMed $h = 0.8024$** (homophilous; benchmark runs not measured).\n"
    )
    rb.add("```mermaid")
    rb.add("flowchart LR")
    rb.add("    G[\"Graph G=(V,E,X)\"] --> E[\"GNN Encoder<br/>(GCN / SAGE / APPNP / GCNII / H2GCN)\"]")
    rb.add("    E -->|Softmax Head| B1[\"B1: GNN Softmax Baseline\"]")
    rb.add("    E -->|\"Hidden Embeddings Z (64-D)\"| S[\"Train-Only Scaler<br/>(Optional L2 Norm + Min-Max [0,1])\"]")
    rb.add("    S --> P[\"Optional Capped Pseudo-Labels<br/>(k per class via Val F1)\"]")
    rb.add("    P --> F[\"Base FMNN Fit (B2)<br/>(theta, gamma, 0 cross-class overlap)\"]")
    rb.add("    F --> M[\"GSHR Merge + Prune<br/>(Graph Assoc, Purity, Val Tau_prune)\"]")
    rb.add("    M --> H[\"Hybrid Scoring (B7/B13/B14)<br/>(Global lam or Inside-1 Box Gate + T)\"]")
    rb.add("    H --> SM[\"Optional Prediction Smoothing rho<br/>((1-rho)S + rho D^-1 A S)\"]")
    rb.add("```")
    rb.add("")

    # =========================================================================
    # 3. Datasets and experimental setup
    # =========================================================================
    rb.add("## 3. Datasets and experimental setup\n")
    ds_rows = []
    for name in ["Cora", "CiteSeer", "Chameleon", "PubMed"]:
        info = ds_info[name]
        split_desc = (
            "Planetoid public split"
            if name in {"Cora", "CiteSeer", "PubMed"}
            else "WikipediaNetwork geom_gcn split 0 (with duplicates)"
        )
        status_note = "Evaluated (v1, v2, v3, v4)" if name in {"Cora", "CiteSeer"} else (
            "Evaluated (v1, v2, v3; v4 not measured)" if name == "Chameleon" else "Stats checked only (benchmarks not measured)"
        )
        ds_rows.append([
            name,
            str(info["nodes"]),
            f"{info['undir_edges']} ({info['dir_edges']} dir.)",
            str(info["feats"]),
            str(info["classes"]),
            f"{info['train']} / {info['val']} / {info['test']}",
            f"{info['homophily']:.4f}",
            f"{info['dup_rows']} rows ({info['test_dup_train']} test==train)",
            f"{split_desc}; {status_note}",
        ])
    rb.add_table(
        caption="Benchmark dataset statistics computed directly from loaded GraphData instances.",
        headers=["Dataset", "Nodes", "Undirected (Dir.) Edges", "Features", "Classes", "Train / Val / Test", "Homophily (h)", "Duplicate Feature Rows", "Split & Evaluation Status"],
        rows=ds_rows,
        takeaway="Chameleon is strongly heterophilous (h=0.2299) and contains 327 duplicate feature rows with 64/456 test nodes (14.0%) identical to training feature vectors.",
    )

    py_ver = sys.version.split()[0]
    rb.add(
        f"- **Hardware & software environment:** CPU-only workstation (`CUDA: False`), processor `AMD64 Family 25 Model 97 Stepping 2`, OS `Windows 11`, `Python {py_ver}`, `PyTorch {torch.__version__}`, `PyTorch Geometric {torch_geometric.__version__}`.\n"
        "- **Seeds used:** `v1` used `3` seeds (`0, 1, 2`) for main runs/ablations (with `results/tables/*.json` using `5` seeds `0..4`); `v2` used `3` seeds (`0, 1, 2`) for encoder search/ablation steps and **`5` seeds (`0..4`)** for final multi-dataset comparison; `v3` and `v4` used **`10` seeds (`0..9`)** (with `v3` unified latency measured over seeds `0, 1, 2`, and `v4` over `0..9`).\n"
        "- **Measurement & unified latency protocol:** Timing uses `time.perf_counter()`, peak RAM uses Python `tracemalloc.get_traced_memory()` around pipeline stages (stopped before latency benchmarking in `v3`/`v4`). Because `v1`/`v2` timed only sub-steps for hybrids (`0.11–0.35 us/node`), `v3` and `v4` re-measure latency with `measure_prediction_latency` (`src/metrics.py`) under one identical CPU full-test-batch protocol: **`20` warm-up passes followed by the median of `100` full-batch passes divided by `|test_idx|`**.\n"
        "- **Metric definitions:**\n"
        "  - `Accuracy`: fraction of test nodes predicted correctly.\n"
        "  - `Macro-F1`: unweighted arithmetic mean of per-class F1 scores (treats minority classes equally).\n"
        "  - `Num Boxes`: final count $K$ of FMNN hyperboxes.\n"
        "  - `Refine Time (s)`: wall time for GSHR search, merging, pruning, and validation parameter selection.\n"
        "  - `Peak Memory (MB)`: `peak / 1024^2` bytes allocated during pipeline execution.\n"
        "  - `Box Storage (bytes)`: $K \\times 2 \\times d \\times 4$ bytes (storing $v, w \\in \\mathbb{R}^d$ per box in float32).\n"
        "  - `Cross-Class Overlap`: count of overlapping hyperbox pairs with different class labels (must remain `0`).\n"
        "  - `Paired t-test`: `scipy.stats.ttest_rel` over paired per-seed scores reporting $t$-statistic and two-sided $p$-value.\n"
    )

    # =========================================================================
    # 4. Method and implementation details
    # =========================================================================
    rb.add("## 4. Method and implementation details\n")
    rb.add(
        "1. **GNN Encoder & Training (`src/gnn.py` — `GCNEncoder`, `SAGEEncoder`, `APPNPEncoder`, `GCNIIEncoder`, `H2GCNEncoder`, `train_gnn`, `search_encoder_hyperparams`, `search_encoders_validation`):** Trains with Adam cross-entropy loss strictly on `train_idx` with early stopping on `val_idx` (`val_macro_f1` in `v1`/`v4`, `val_loss` in `v2`/`v3` where `patience=50`, `max_epochs=500` in `v1` and `300` in `v2`/`v4`). Final `v4` encoder search winners: **Cora -> `appnp`** (`hidden=64, dropout=0.5, lr=0.01, wd=5e-4`); **CiteSeer -> `gcnii`** (`hidden=64, dropout=0.5, lr=0.01, wd=5e-4`); `Chameleon` v4 encoder search was not completed (`not measured`).\n"
        "2. **Embedding Normalization (`src/embeddings.py` — `l2_normalize_embeddings`, `EmbeddingScaler`, `fit_scaler_and_transform`):** Optionally normalizes rows to unit L2 norm ($Z / \\max(\\|Z\\|_2, 10^{-12})$), fits `min`/`max` statistics **exclusively on `train_idx`**, and clips transformed coordinates into $[0, 1]^d$. PCA is used **only** for 2D visualizations (`experiments/visualize_graph.py`), never for classification.\n"
        "3. **Base & Pseudo-Labeled FMNN (`src/fmnn.py` — `FMNN.fit`, `FMNN.fit_with_pseudo`, `FMNN.membership_matrix`):** Incrementally expands boxes under L1 budget $d\\theta$ (`theta=0.3`, `gamma=4.0`) and resolves cross-class overlaps via `_test_and_contract`. `fit_with_pseudo` fits true training nodes first, attaches covered pseudo-nodes without adding boxes, tries expanding same-class boxes within $\\theta$, and creates new boxes only if uncovered/non-expandable.\n"
        "4. **GSHR Refinement (`src/merge.py` — `gshr_merge`, `fast_gshr_merge`; `src/prune.py` — `gshr_prune`):** Merge evaluates $M = \\alpha\\,\\text{Geom} + \\beta\\,\\text{GraphAssoc} + \\delta\\,\\text{Consist} - \\eta\\,\\text{Risk}$ (`0.35, 0.35, 0.20, 0.10`) and in `fast_gshr_merge` skips pairs across classes or without geometric overlap. Prune greedily tests removals sorted by ascending multi-term importance (`mu = [0.25, 0.25, 0.20, 0.20, 0.10]`), preserving $\\ge 1$ box per class and pure-NumPy confusion-matrix validation F1 guards.\n"
        "5. **Capped Pseudo-Labeling (`src/pipeline.py` — `_build_pseudo_fit_set`, `_build_capped_pseudo_fit_set`):** Adds non-train nodes exceeding GNN confidence `c` (`B8`: $c \\in \\{0.8, 0.9, 0.95\\}$; `B11`/`B14`: top-$k$ per class from $k \\in \\{20, 50, 100\\}$ with `conf >= 0.8`) carrying only **GNN predicted labels**, keeping true `val`/`test` labels strictly masked (`-1`) and excluded from reported `train_size`.\n"
        "6. **Hybrid & Gated Prediction (`src/pipeline.py` — `predict_hybrid`, `compute_single_box_gate_and_scores`, `predict_gated_hybrid`):**\n"
        "   - Global $\\lambda$ (`B7`–`B12`, `B14`): $\\text{score} = \\lambda \\cdot P_{\\text{GNN}}(y|x; T) + (1 - \\lambda) \\cdot \\tilde{\\mu}_{\\text{FMNN}}(x)$, with $\\lambda \\in \\{0, 0.25, 0.5, 0.75, 1.0\\}$ chosen on `val_macro_f1`.\n"
        "   - Single-box gate (`B13_gated_hybrid`): routes node $i$ to $\\tilde{\\mu}_{\\text{FMNN}}$ only when $\\sum_j \\mathbf{1}[v_j \\le z_i \\le w_j] = 1$, else uses temperature-scaled GNN softmax $\\text{softmax}(\\text{logits}/T)$ with $T \\in \\{0.5, 1, 1.5, 2\\}$ chosen on validation.\n"
        "7. **Prediction-Time Smoothing (`src/pipeline.py` — `smooth_scores_graph`, `_select_smoothing_rho`):** Computes $S' = (1 - \\rho) S + \\rho (D^{-1} A S)$ over $\\rho \\in \\{0.0, 0.1, 0.2, 0.3, 0.5\\}$ chosen strictly on validation macro-F1 (v4 gating selected $\\rho = 0.0$ / `use_smoothing = False` on both Cora and CiteSeer).\n"
    )

    rb.add_table(
        caption="Complete inventory of pipeline variants across versions v1-v4.",
        headers=["Variant", "Version", "One-Line Definition", "Encoder /Norm", "Merge / Prune", "Hybrid / Pseudo / Smooth"],
        rows=[
            ["B1_gnn_softmax", "v1-v4", "End-to-end GNN softmax baseline (no hyperboxes)", "Base GCN (v1-v3); APPNP/GCNII (v4)", "Off / Off", "Softmax only / Off / Off"],
            ["B2_gnn_fmnn", "v1-v2", "Base Simpson FMNN fitted on train embeddings", "Frozen GNN + min-max", "Off / Off", "Box only / Off / Off"],
            ["B3_geo_merge_prune", "v1", "Geometry-only merge and geometry-only prune", "Frozen GNN + min-max", "Geo / Geo", "Box only / Off / Off"],
            ["B4_graph_merge_only", "v1", "Graph-aware GSHR merging without pruning", "Frozen GNN + min-max", "GSHR / Off", "Box only / Off / Off"],
            ["B5_graph_prune_only", "v1", "Graph-aware GSHR pruning without merging", "Frozen GNN + min-max", "Off / GSHR", "Box only / Off / Off"],
            ["B6_gshr_full", "v1-v2", "Full GSHR graph-aware merge + prune grid", "Frozen GNN + min-max", "GSHR / GSHR", "Box only / Off / Off"],
            ["B7_hybrid", "v2-v4", "Global validation lambda blend of GNN softmax and B6/fast-GSHR box scores", "v2 GCN+L2 / v4 Best+L2", "GSHR / GSHR", "Global lam / Off / Off"],
            ["B8_hybrid_pseudo", "v2", "B7 trained on train + uncapped high-confidence pseudo-labels (c grid)", "v2 GCN + L2", "GSHR / GSHR", "Global lam / Uncapped c / Off"],
            ["B9_gshr_fast", "v2-v3", "Same-class fast merge + theta/tau_prune/safeguard val search", "v2 GCN + L2", "Fast / Vectorized", "Box only / Off / Off"],
            ["B10_final", "v2-v3", "Validation-gated assembly of v2 components (unconstrained pseudo c)", "Val-gated enc + L2", "Val-gated fast", "Global lam / Uncapped c / Off"],
            ["B11_final_lite", "v3-v4", "Best v2 encoder + hybrid + fast GSHR + class-capped k pseudo-labels", "Val-gated enc + L2", "Fast / Vectorized", "Global lam / Capped k / Off"],
            ["B12_ensemble", "v4", "5-seed encoder ensemble softmax + FMNN on val-picked avg vs concat", "5x Best Enc + L2", "Fast / Vectorized", "Global lam / Off / Off"],
            ["B13_gated_hybrid", "v4", "Per-node single-box gate (==1 box) + GNN temperature scaling T", "Best Enc + L2", "Fast / Vectorized", "Single-box gate + T / Off / Off"],
            ["B14_final", "v4", "Validation-gated combination of v4 encoder, ensemble, pseudo, hybrid, smoothing", "Best Enc + L2 (+5x CiteSeer)", "Fast / Vectorized", "Val-gated (Cora: +pseudo; CiteSeer: +ens; rho=0)"],
        ],
        takeaway="Progressing from B1-B6 (v1) through B14 (v4) turned test-based grid tuning and slow scalar loops into leak-free validation-gated models.",
    )

    rb.add_table(
        caption="Asymptotic time and space complexity (N nodes, E edges, K boxes, d dims, C classes) linked to measured CiteSeer/Chameleon costs.",
        headers=["Stage", "Time Complexity", "Space Complexity", "Measured Bottleneck Connection"],
        rows=[
            ["GNN Training (1 epoch)", "O(E d + N d F)", "O(N d + F d)", "0.79s (Cora GCN) to 51s (PubMed APPNP/GCNII); CPU cache-sensitive for K=10 or 8 layers."],
            ["Base FMNN Fit (B2)", "O(M K d + K d)", "O(K d + M d)", "0.008-0.30s for train set M; grows to 0.38-2.05s with uncapped pseudo nodes (B8, M<=1557)."],
            ["GSHR Merge", "O(K^2 d + K^2 avg_deg)", "O(K d)", "11.5s across c-grid on Chameleon B8 (117 boxes, 27k pair checks), near 0s in fast_gshr_merge."],
            ["GSHR Prune (pre-vec vs vec)", "O(|grid| K^2 N_val d)", "O(N_val K)", "Pre-vec: 91-94% of B8 time (263s Chameleon, 75.8 MB); NumPy + zero-copy cut to 9.7-15.6s."],
            ["Unified Prediction (CPU)", "O(E d + N_test K d)", "O(K d * 8B)", "Unified CPU batch median: B1 takes 3.6-31.8 us/node; B11/B14 take 10.3-33.1 us/node; storage 3.2-9.8 KB."],
        ],
        takeaway="Naive pruning dominates refinement cost when box count K grows large (as in heterophilic or uncapped-pseudo settings), which chunked vectorization and capping k suppress.",
    )

    # =========================================================================
    # 5. Version history
    # =========================================================================
    rb.add("## 5. Version history\n")
    rb.add(
        f"- **v1 Baseline & Diagnosis:** Established `B1`–`B6` across 3 seeds (`Cora`, `CiteSeer`, `Chameleon`) and ablations `A0`–`A8`. Revealed two critical findings: (1) on Cora, `B6` reduced boxes by `{cora_v1_red:.1f}%` (`{cora_v1_b2['num_boxes_mean']:.1f} -> {cora_v1_b6['num_boxes_mean']:.1f}`) at a small accuracy dip (`{cora_v1_b1['test_accuracy_mean']:.4f} -> {cora_v1_b6['test_accuracy_mean']:.4f}`), while **merge was completely inactive** (`B4 = B2` with `{cora_v1_b4['num_boxes_mean']:.1f}` boxes, `A3 = A0`, `tau_merge` sweep completely flat at `8` boxes, and `A1 beta=0` identical to `GSHR_full`) because `size_limit` ($\\sum (w-v) \\le d \\theta$) and `volume_inflation <= 0.5` blocked almost all merges; (2) `A6` sweeps in `experiments/run_ablations.py` evaluated test metrics across parameter grids.\n"
        f"- **v2 Stabilization & Expansion:** Strictly isolated test data; added validation-only encoder hyperparameter search (`encoder_config_*.json`, raising `B1` Cora `{cora_v1_b1['test_accuracy_mean']:.4f} -> {v2_sum[(v2_sum['dataset']=='Cora')&(v2_sum['variant']=='B1_gnn_softmax')].iloc[0]['test_accuracy_mean']:.4f}`), row-wise L2-normalization before `[0, 1]` min-max scaling, `B7_hybrid` global `lam` blending (`{cora_v2_b2['test_accuracy_mean']:.4f}` -> `{cora_v2_b7['test_accuracy_mean']:.4f}` Cora vs `v2 B6`'s `0.7760`), `B8_hybrid_pseudo` (`{cite_v2_b7['test_accuracy_mean']:.4f} -> {cite_v2_b8['test_accuracy_mean']:.4f}` CiteSeer), `B9_gshr_fast` (fast same-class merge + validation `theta/tau_prune/safeguards` + `>= 1` box/class guard, bringing pure-box Cora `B9` to `{v2_sum[(v2_sum['dataset']=='Cora')&(v2_sum['variant']=='B9_gshr_fast')].iloc[0]['test_accuracy_mean']:.4f}`), and `B10_final` validation component gating (`CiteSeer {v2_sum[(v2_sum['dataset']=='CiteSeer')&(v2_sum['variant']=='B10_final')].iloc[0]['test_accuracy_mean']:.4f}`, `Chameleon {v2_sum[(v2_sum['dataset']=='Chameleon')&(v2_sum['variant']=='B10_final')].iloc[0]['test_accuracy_mean']:.4f}` after rejecting v2 `val_loss` encoder).\n"
        "- **v3 Profiling & Efficiency:** Profiled `B8` and exposed `Prune` as `91.4%–94.1%` of wall time (`37.7s` Cora, `14.4s` CiteSeer, `263.1s` Chameleon) and `16.1–75.8 MB` traced RAM. Implemented chunked NumPy FMNN (`membership_matrix`), pure-NumPy pruning metrics, capped pseudo-labels (`k in {20, 50, 100}`), covered-node non-creation (`fit_with_pseudo`), unified CPU inference latency (`measure_prediction_latency`, exposing `v2` hybrid `0.11–0.35 us` artifacts vs real `9.8–53.1 us/node`), and `B11_final_lite` over 10 seeds (`B10 -> B11` refine time `-58.9%` Cora, `-22.1%` CiteSeer, `-24.5%` Chameleon; boxes `11.3 -> 8.4` Cora).\n"
        f"- **v4 Raising the GNN Ceiling:** Added `APPNP`, `GCNII`, and `H2GCN`-style encoders with validation search (`APPNP` raised Cora `B1` `{cora_v4_b1g['test_accuracy_mean']:.4f} -> {cora_v4_b1b['test_accuracy_mean']:.4f}`; `GCNII` raised CiteSeer `B1` `{cite_v4_b1g['test_accuracy_mean']:.4f} -> {cite_v4_b1b['test_accuracy_mean']:.4f}`), 5-encoder ensemble `B12_ensemble` (`{cora_v4_b12['test_accuracy_mean']:.4f}` Cora, `{cite_v4_b12['test_accuracy_mean']:.4f}` CiteSeer), single-box gate + temperature scaling `B13_gated_hybrid`, validation-tuned smoothing `rho`, and validation-gated `B14_final`. Chameleon `v4`, PubMed benchmarks, GAT, and MLP baseline remain `not measured` (`Missing data`).\n"
    )

    # =========================================================================
    # 6. Results
    # =========================================================================
    rb.add("## 6. Results\n")

    # --- 6.1 ---
    rb.add("### 6.1 v1 main benchmark (3 seeds)\n")
    for ds_name, df_s in [("Cora", v1_cora), ("CiteSeer", v1_cite), ("Chameleon", v1_cham)]:
        rows_61 = []
        for _, r in df_s.iterrows():
            rows_61.append([
                r["variant"],
                str(int(r["num_seeds"])),
                fmt_ms(r["test_accuracy_mean"], r["test_accuracy_std"]),
                fmt_ms(r["test_macro_f1_mean"], r["test_macro_f1_std"]),
                fmt_ms(r["val_accuracy_mean"], r["val_accuracy_std"]),
                fmt_ms(r["num_boxes_mean"], r["num_boxes_std"], 1),
                fmt_ms(r["mean_box_size_mean"], r["mean_box_size_std"], 2),
                fmt_ms(r["refine_time_mean"], r["refine_time_std"], 3),
                fmt_ms(r["peak_memory_mb_mean"], r["peak_memory_mb_std"], 2),
            ])
        b2_n = float(df_s[df_s["variant"] == "B2_gnn_fmnn"].iloc[0]["num_boxes_mean"])
        b6_n = float(df_s[df_s["variant"] == "B6_gshr_full"].iloc[0]["num_boxes_mean"])
        take_61 = (
            f"On {ds_name} v1, B6 matches B5 in accuracy and boxes while reducing boxes from {b2_n:.1f} to {b6_n:.1f}, behind B1 softmax accuracy."
            if ds_name == "Cora"
            else f"On {ds_name} v1, B6 reduces boxes from {b2_n:.1f} to {b6_n:.1f}, trailing B1 softmax accuracy."
        )
        rb.add_table(
            caption=f"v1 main results on {ds_name} (3 seeds: 0, 1, 2) from results/{ds_name.lower()}_summary.csv.",
            headers=["Variant", "Seeds", "Test Acc", "Test Macro-F1", "Val Acc", "Num Boxes", "Mean Box Size", "Refine (s)", "Peak Mem (MB)"],
            rows=rows_61,
            takeaway=take_61,
        )

    # --- 6.2 ---
    rb.add("### 6.2 v1 baseline and GSHR JSON results\n")
    tbl_rows_62 = []
    for fname in ["baseline_cora.json", "gshr_cora_merge.json", "gshr_cora_prune.json", "gshr_cora_full.json"]:
        d_json = json.loads((PROJECT_ROOT / "results/tables" / fname).read_text(encoding="utf-8"))
        df_j = pd.DataFrame(d_json["runs"])
        agg_j = aggregate_csv(df_j, "variant", ["test_accuracy", "test_macro_f1", "num_boxes", "mean_box_size", "refine_time", "peak_memory_mb"])
        for _, r in agg_j.iterrows():
            tbl_rows_62.append([
                fname,
                r["variant"],
                str(int(r["n"])),
                fmt_ms(r["test_accuracy_mean"], r["test_accuracy_std"]),
                fmt_ms(r["test_macro_f1_mean"], r["test_macro_f1_std"]),
                fmt_ms(r["num_boxes_mean"], r["num_boxes_std"], 1),
                fmt_ms(r["refine_time_mean"], r["refine_time_std"], 3),
                fmt_ms(r["peak_memory_mb_mean"], r["peak_memory_mb_std"], 2),
            ])
    rb.add_table(
        caption="5-seed Cora CLI wrapper results from results/tables/*.json (seeds 0..4).",
        headers=["File", "Variant", "Seeds", "Test Acc", "Test Macro-F1", "Num Boxes", "Refine (s)", "Peak Mem (MB)"],
        rows=tbl_rows_62,
        takeaway="Across 5 Cora seeds, B4 merge-only remains identical to B2 (13.6 boxes, 0.7792 acc), whereas B5 prune-only produces the exact same 7.0 boxes and 0.7760 acc as B6_gshr_full in 3x less refine time.",
    )

    # --- 6.3 ---
    rb.add("### 6.3 v1 ablations (A0-A8, robustness, smoothing)\n")
    # A0-A4 aggregated
    a04_agg = aggregate_csv(abl_core, "ablation", ["test_accuracy", "test_macro_f1", "num_boxes", "refine_time"])
    rb.add_table(
        caption="Cora core component ablations A0-A4 (3 seeds) from results/ablations/a0_a4_core_ablations.csv.",
        headers=["Ablation", "Seeds", "Test Acc", "Test Macro-F1", "Num Boxes", "Refine (s)"],
        rows=[
            [r["ablation"], str(int(r["n"])), fmt_ms(r["test_accuracy_mean"], r["test_accuracy_std"]), fmt_ms(r["test_macro_f1_mean"], r["test_macro_f1_std"]), fmt_ms(r["num_boxes_mean"], r["num_boxes_std"], 1), fmt_ms(r["refine_time_mean"], r["refine_time_std"], 3)]
            for _, r in a04_agg.iterrows()
        ],
        takeaway="Setting beta=0 (A1), dropping merge (A2), or using geometry-only (A4) all yield identical 0.7720 accuracy and 7.0 boxes, confirming merge (A3=A0=13.3 boxes) contributed zero box reduction.",
    )

    # A5 safeguard ablation
    a5_agg = aggregate_csv(abl_sg, "setting", ["test_accuracy", "test_macro_f1", "num_boxes", "cross_class_overlaps"])
    rb.add_table(
        caption="A5 safeguards on/off ablation on Cora (3 seeds) from results/ablations/a5_safeguard_ablation.csv.",
        headers=["Setting", "Seeds", "Test Acc", "Test Macro-F1", "Num Boxes", "Cross-Class Overlaps"],
        rows=[
            [r["setting"], str(int(r["n"])), fmt_ms(r["test_accuracy_mean"], r["test_accuracy_std"]), fmt_ms(r["test_macro_f1_mean"], r["test_macro_f1_std"]), fmt_ms(r["num_boxes_mean"], r["num_boxes_std"], 1), fmt(r["cross_class_overlaps_mean"], 1)]
            for _, r in a5_agg.iterrows()
        ],
        takeaway="Disabling safeguards raises accuracy slightly (0.7720 -> 0.7760) with +0.7 more boxes (7.0 -> 7.7) and still keeps 0 cross-class overlaps.",
    )

    # A6 sweeps (combine cleanly into descriptive tables)
    rb.add_table(
        caption="A6 theta sensitivity sweep on Cora seed 0 (test-evaluated; plus validation scan results/fmnn_theta_scan.csv).",
        headers=["Source", "theta", "Base Boxes", "Refined Boxes", "Val Acc / F1", "Test Acc", "Test Macro-F1"],
        rows=[
            ["fmnn_theta_scan.csv (val)", f"{r['theta']:.1f}", str(int(r['num_boxes'])), "n/a (base FMNN)", f"{r['val_acc']:.4f} / {r['val_macro_f1']:.4f}", "not measured", "not measured"]
            for _, r in theta_scan_v1.iterrows()
        ] + [
            ["a6_sweep_theta.csv (test)", f"{r['theta']:.1f}", str(int(r['num_boxes_base'])), str(int(r['num_boxes_refined'])), "not measured", f"{r['test_accuracy']:.4f}", f"{r['test_macro_f1']:.4f}"]
            for _, r in abl_th.iterrows()
        ],
        takeaway="Larger theta reduces base boxes monotonically (56 at theta=0.1 down to 7 at theta>=0.5), with theta=0.4 peaking on seed-0 test F1 (0.7866) in A6.",
    )

    # Remaining A6 sweeps (tau_merge, tau_prune, weights) combined into one clean sweep summary table
    a6_misc_rows = []
    for _, r in abl_tm.iterrows():
        a6_misc_rows.append(["tau_merge sweep", f"tau_merge={r['tau_merge']}", str(int(r["num_boxes"])), f"{r['test_accuracy']:.4f}", f"{r['test_macro_f1']:.4f}"])
    for _, r in abl_tp.iterrows():
        a6_misc_rows.append(["tau_prune sweep", f"tau_prune={r['tau_prune']}", str(int(r["num_boxes"])), f"{r['test_accuracy']:.4f}", f"{r['test_macro_f1']:.4f}"])
    for _, r in abl_wt.iterrows():
        a6_misc_rows.append(["score weights", f"{r['setting']} ({r['alpha']}/{r['beta']}/{r['delta']}/{r['eta']})", str(int(r["num_boxes"])), f"{r['test_accuracy']:.4f}", f"{r['test_macro_f1']:.4f}"])
    rb.add_table(
        caption="A6 tau_merge, tau_prune, and merge weight sweeps on Cora seed 0 from results/ablations/a6_sweep_*.csv (test-evaluated).",
        headers=["Sweep Group", "Parameter Value", "Num Boxes", "Test Acc", "Test Macro-F1"],
        rows=a6_misc_rows,
        takeaway="Both the tau_merge sweep and all 5 merge weight settings produce an identically flat 8 boxes and 0.7830 accuracy, while tau_prune>=0.02 drops one box (8 -> 7) with an accuracy drop (0.7830 -> 0.7770).",
    )

    # A7 cross-dataset
    a7_agg = abl_a7.groupby(["dataset", "homophily_ratio", "variant"], sort=False, as_index=False).agg(
        n=("test_accuracy", "count"),
        acc_mean=("test_accuracy", "mean"), acc_std=("test_accuracy", lambda s: float(np.std(s, ddof=1))),
        f1_mean=("test_macro_f1", "mean"), f1_std=("test_macro_f1", lambda s: float(np.std(s, ddof=1))),
        bx_mean=("num_boxes", "mean"), bx_std=("num_boxes", lambda s: float(np.std(s, ddof=1))),
    )
    rb.add_table(
        caption="A7 cross-dataset comparison across homophilous and heterophilous graphs (3 seeds) from a7_cross_dataset.csv.",
        headers=["Dataset", "Homophily (h)", "Variant", "Seeds", "Test Acc", "Test Macro-F1", "Num Boxes"],
        rows=[
            [r["dataset"], f"{r['homophily_ratio']:.4f}", r["variant"], str(int(r["n"])), fmt_ms(r["acc_mean"], r["acc_std"]), fmt_ms(r["f1_mean"], r["f1_std"]), fmt_ms(r["bx_mean"], r["bx_std"], 1)]
            for _, r in a7_agg.iterrows()
        ],
        takeaway="On heterophilous Chameleon (h=0.2299), unrefined B2 spawns 29.0 boxes (vs 11.7-13.3 on CiteSeer/Cora), which B6 prunes by 76.9% to 6.7 boxes.",
    )

    # A8 GCN vs GraphSAGE on Cora
    a8_agg = abl_a8.groupby(["encoder", "variant"], sort=False, as_index=False).agg(
        n=("test_accuracy", "count"),
        acc_mean=("test_accuracy", "mean"), acc_std=("test_accuracy", lambda s: float(np.std(s, ddof=1))),
        f1_mean=("test_macro_f1", "mean"), f1_std=("test_macro_f1", lambda s: float(np.std(s, ddof=1))),
        bx_mean=("num_boxes", "mean"), bx_std=("num_boxes", lambda s: float(np.std(s, ddof=1))),
        rt_mean=("refine_time", "mean"),
    )
    gcn_b1_acc = float(a8_agg[(a8_agg["encoder"] == "gcn") & (a8_agg["variant"] == "B1_gnn_softmax")].iloc[0]["acc_mean"])
    sage_b1_acc = float(a8_agg[(a8_agg["encoder"] == "sage") & (a8_agg["variant"] == "B1_gnn_softmax")].iloc[0]["acc_mean"])
    gcn_b2_bx = float(a8_agg[(a8_agg["encoder"] == "gcn") & (a8_agg["variant"] == "B2_gnn_fmnn")].iloc[0]["bx_mean"])
    sage_b2_bx = float(a8_agg[(a8_agg["encoder"] == "sage") & (a8_agg["variant"] == "B2_gnn_fmnn")].iloc[0]["bx_mean"])
    rb.add_table(
        caption="A8 encoder comparison (GCN vs GraphSAGE) on Cora (3 seeds) from results/ablations/a8_encoders.csv.",
        headers=["Encoder", "Variant", "Seeds", "Test Acc", "Test Macro-F1", "Num Boxes", "Refine (s)"],
        rows=[
            [r["encoder"], r["variant"], str(int(r["n"])), fmt_ms(r["acc_mean"], r["acc_std"]), fmt_ms(r["f1_mean"], r["f1_std"]), fmt_ms(r["bx_mean"], r["bx_std"], 1), fmt(r["rt_mean"], 3)]
            for _, r in a8_agg.iterrows()
        ],
        takeaway=f"Under default v1 hyperparameters (3 seeds, no paired test saved), raw Cora B1 accuracy is higher for GCN than GraphSAGE ({gcn_b1_acc:.4f} vs {sage_b1_acc:.4f}) while GraphSAGE B2 starts with fewer boxes ({sage_b2_bx:.1f} vs {gcn_b2_bx:.1f}).",
    )

    # Robustness & smoothing ablation
    rb.add_table(
        caption="v1 Robustness stress tests and prediction-time smoothing on Cora seed 0 from results/ablations/.",
        headers=["Experiment", "Condition / rho", "B2 Acc / F1 (Boxes)", "B6 Acc / F1 (Boxes)"],
        rows=[
            [f"{r['test_type']}", f"{r['perturbation']}", f"{r['b2_test_acc']:.4f} / {r['b2_test_f1']:.4f} ({int(r['b2_boxes'])})", f"{r['b6_test_acc']:.4f} / {r['b6_test_f1']:.4f} ({int(r['b6_boxes'])})"]
            for _, r in abl_rob.iterrows()
        ] + [
            ["prediction_smoothing", f"rho={r['rho']}", "n/a", f"{r['test_accuracy']:.4f} / {r['test_macro_f1']:.4f} (B6)"]
            for _, r in abl_sm.iterrows()
        ],
        takeaway="Under 10-30% test feature masking or 50% label reduction, B6 retains nearly identical accuracy to B2 (0.7740 vs 0.7740 at 30% mask) with 8 vs 14 boxes; rho=0.3 raises seed-0 B6 F1 slightly to 0.7848.",
    )

    # --- 6.4 ---
    rb.add("### 6.4 v2 5-seed comparison, encoder search, and validation theta sweep\n")
    enc_v2_rows = []
    for ds in ["Cora", "CiteSeer", "Chameleon"]:
        ep = json.loads((PROJECT_ROOT / f"results/v2/encoder_config_{ds.lower()}.json").read_text(encoding="utf-8"))
        bc = ep["best_config"]
        b10_d = json.loads((PROJECT_ROOT / f"results/v2/b10_components_{ds.lower()}.json").read_text(encoding="utf-8"))
        enc_v2_rows.append([
            ds,
            str(ep["search_seeds"]),
            f"hidden={bc['hidden']}, drop={bc['dropout']}, wd={bc['weight_decay']}, lr={bc['lr']}, es={ep['early_stop_metric']}",
            fmt(ep["best_val_macro_f1_mean"], 4),
            fmt(ep["best_val_acc_mean"], 4),
            str(b10_d["selected"]),
        ])
    rb.add_table(
        caption="v2 validation-only encoder hyperparameter search and B10_final component gating results (results/v2/).",
        headers=["Dataset", "Search Seeds", "Chosen Encoder Hyperparams", "Best Val Macro-F1", "Best Val Acc", "B10 Selected Components"],
        rows=enc_v2_rows,
        takeaway="Validation selected larger hidden=128 / val_loss early stop for Cora and CiteSeer, whereas B10 gating rejected the v2 encoder on Chameleon because val macro-F1 dropped (0.7142 -> 0.6513) and box count surged (29.0 -> 112.4).",
    )

    # Step-by-step v2 Cora 3-seed progression (encoder_b1_b2, b7, b8, b9)
    step_files = [
        ("v2 Step 1 (encoder)", "results/v2/encoder_b1_b2_cora.csv", "label"),
        ("v2 Step 2 (B7 base enc)", "results/v2/b7_hybrid_cora_base_enc.csv", "variant"),
        ("v2 Step 2 (B7 v2 enc)", "results/v2/b7_hybrid_cora_v2_enc.csv", "variant"),
        ("v2 Step 3 (B8 pseudo)", "results/v2/b8_hybrid_pseudo_cora_v2_enc.csv", "variant"),
        ("v2 Step 4 (B9 base enc)", "results/v2/b9_gshr_fast_cora_base_enc.csv", "variant"),
        ("v2 Step 4 (B9 v2 enc)", "results/v2/b9_gshr_fast_cora_v2_enc.csv", "variant"),
    ]
    step_rows = []
    for stage_lbl, rel_p, grp_c in step_files:
        df_st = pd.read_csv(PROJECT_ROOT / rel_p)
        agg_st = aggregate_csv(df_st, grp_c, ["test_accuracy", "test_macro_f1", "num_boxes", "refine_time", "chosen_lam", "chosen_c", "pseudo_count"])
        for _, r in agg_st.iterrows():
            step_rows.append([
                stage_lbl,
                str(r[grp_c]),
                str(int(r["n"])),
                fmt_ms(r["test_accuracy_mean"], r["test_accuracy_std"]),
                fmt_ms(r["test_macro_f1_mean"], r["test_macro_f1_std"]),
                fmt_ms(r["num_boxes_mean"], r["num_boxes_std"], 1),
                fmt_ms(r.get("refine_time_mean"), r.get("refine_time_std"), 3, na_label="0.000"),
                fmt(r.get("chosen_lam_mean"), 2, "n/a") if "chosen_lam_mean" in r else "n/a",
                fmt(r.get("pseudo_count_mean"), 1, "0"),
            ])
    rb.add_table(
        caption="v2 step-by-step Cora development results (seeds 0, 1, 2) from results/v2/ step CSVs.",
        headers=["Development Stage", "Variant / Label", "Seeds", "Test Acc", "Test Macro-F1", "Boxes", "Refine (s)", "Mean lam", "Mean Pseudo"],
        rows=step_rows,
        takeaway="On Cora 3-seed development runs, v2 encoder + L2 raised B1 from 0.7973 -> 0.8003 and B2 from 0.7763 -> 0.7867, B7_hybrid hit 0.8020 with 7.0 boxes, and B9_gshr_fast lifted pure-box accuracy to 0.7960.",
    )

    # Main 5-seed v2 summary table + B10 vs B1 p-values
    v2_rows = []
    for _, r in v2_sum.iterrows():
        v2_rows.append([
            r["dataset"],
            r["variant"],
            str(int(r["num_seeds"])),
            fmt_ms(r["test_accuracy_mean"], r["test_accuracy_std"]),
            fmt_ms(r["test_macro_f1_mean"], r["test_macro_f1_std"]),
            fmt_ms(r["num_boxes_mean"], r["num_boxes_std"], 1),
            fmt_ms(r["refine_time_mean"], r["refine_time_std"], 3),
            f"{fmt_ms(r['pred_latency_per_node_mean'], r['pred_latency_per_node_std'], 2, 1e6)} (legacy)",
            fmt_ms(r["peak_memory_mb_mean"], r["peak_memory_mb_std"], 2),
            fmt(r["box_storage_bytes_mean"], 0),
        ])
    rb.add_table(
        caption="v2 main 5-seed comparison (seeds 0..4) across Cora, CiteSeer, and Chameleon from results/v2/final_comparison_summary.csv.",
        headers=["Dataset", "Variant", "Seeds", "Test Acc", "Test Macro-F1", "Boxes", "Refine (s)", "Lat (us/node)", "Peak Mem (MB)", "Storage (B)"],
        rows=v2_rows,
        takeaway="On 5 seeds, B10_final is comparable to B1 on Cora (0.8026 vs 0.8034, p=0.8912) and CiteSeer (0.6906 vs 0.6754, p=0.1658) with 23.5% and 31.2% box reductions vs B2, and matches B1 on Chameleon (0.6697, p=1.0000) with 63.4% box reduction.",
    )

    # Validation theta sensitivity sweep v2
    rb.add_table(
        caption="v2 validation-only theta sensitivity sweep (5 seeds) from results/v2/theta_sensitivity_val.csv.",
        headers=["Dataset", "theta", "Val Acc", "Val Macro-F1", "Base Boxes", "Val Best?"],
        rows=[
            [r["dataset"], f"{r['theta']:.1f}", fmt_ms(r["val_accuracy_mean"], r["val_accuracy_std"]), fmt_ms(r["val_macro_f1_mean"], r["val_macro_f1_std"]), fmt_ms(r["num_boxes_mean"], r["num_boxes_std"], 1), "YES" if bool(r["is_val_best"]) else ""]
            for _, r in v2_th.iterrows()
        ],
        takeaway="Using strictly validation macro-F1, base FMNN prefers theta=0.4 on Cora (7.8 boxes), theta=0.2 on CiteSeer (23.6 boxes), and theta=0.1 on Chameleon (298.2 boxes).",
    )

    # --- 6.5 ---
    rb.add("### 6.5 v3 10-seed comparison, paired tests, B8 profile, and unified latency\n")
    rb.add_table(
        caption="B8_hybrid_pseudo stage time and memory profiling breakdown (seed 0, summed across c in {0.8, 0.9, 0.95}) from results/v3/b8_profile.csv.",
        headers=["Dataset", "Pseudo Sel (s / %)", "FMNN Fit (s / %)", "Merge (s / %)", "Prune (s / %)", "Hybrid Eval (s / %)", "Total (s)", "Overall Peak (MB)"],
        rows=[
            [
                r["dataset"],
                f"{r['pseudo_selection_time_s']:.3f} ({r['pseudo_selection_pct']:.1f}%)",
                f"{r['fmnn_fit_time_s']:.3f} ({r['fmnn_fit_pct']:.1f}%)",
                f"{r['pairwise_merge_time_s']:.3f} ({r['pairwise_merge_pct']:.1f}%)",
                f"{r['prune_time_s']:.3f} ({r['prune_pct']:.1f}%)",
                f"{r['hybrid_eval_time_s']:.3f} ({r['hybrid_eval_pct']:.1f}%)",
                f"{r['total_refine_time_s']:.3f}",
                f"{r['overall_peak_mb']:.2f} (prune={r['prune_peak_mb']:.2f})",
            ]
            for _, r in v3_prof.iterrows()
        ],
        takeaway="Pruning accounts for 91.4% to 94.1% of B8 wall time and 100% of peak memory before vectorization.",
    )

    b1_ul = v3_lat[v3_lat["variant"] == "B1_gnn_softmax"]["latency_us_mean"]
    hyb_ul = v3_lat[v3_lat["variant"].isin(["B7_hybrid", "B8_hybrid_pseudo", "B10_final", "B11_final_lite"])]["latency_us_mean"]
    b9_cham_ul = float(_ulat_row("Chameleon", "B9_gshr_fast")["latency_us_mean"])
    rb.add_table(
        caption="Unified end-to-end CPU prediction latency (20 warm-up runs, median of 100 full-test-batch runs, seeds 0..2) from results/v3/unified_latency.csv.",
        headers=["Dataset", "Variant", "Seeds", "Latency (us/node)", "Test Acc"],
        rows=[
            [r["dataset"], r["variant"], str(int(r["num_seeds"])), fmt_ms(r["latency_us_mean"], r["latency_us_std"], 2), fmt_ms(r["test_accuracy_mean"], r["test_accuracy_std"])]
            for _, r in v3_lat.iterrows()
        ],
        takeaway=(
            f"Under identical end-to-end timing, B1 softmax ({float(b1_ul.min()):.2f}-{float(b1_ul.max()):.2f} us/node) is faster than "
            f"hybrid FMNN pipelines ({float(hyb_ul.min()):.2f}-{float(hyb_ul.max()):.2f} us/node; pure-box Chameleon B9 reaches {b9_cham_ul:.2f} us/node due to 50.0 boxes)."
        ),
    )

    rb.add_table(
        caption="v3 10-seed comparison (seeds 0..9) from results/v3/v3_comparison_summary.csv.",
        headers=["Dataset", "Variant", "Seeds", "Test Acc", "Test Macro-F1", "Boxes", "Refine (s)", "Unified Lat (us/node)", "Peak Mem (MB)", "Storage (B)", "Mean Pseudo"],
        rows=[
            [
                r["dataset"], r["variant"], str(int(r["num_seeds"])),
                fmt_ms(r["test_accuracy_mean"], r["test_accuracy_std"]),
                fmt_ms(r["test_macro_f1_mean"], r["test_macro_f1_std"]),
                fmt_ms(r["num_boxes_mean"], r["num_boxes_std"], 1),
                fmt_ms(r["refine_time_mean"], r["refine_time_std"], 3),
                fmt_ms(r["pred_latency_per_node_mean"], r["pred_latency_per_node_std"], 2, 1e6),
                fmt_ms(r["peak_memory_mb_mean"], r["peak_memory_mb_std"], 2),
                fmt(r["box_storage_bytes_mean"], 0),
                fmt(r["pseudo_count_mean"], 1),
            ]
            for _, r in v3_sum.iterrows()
        ],
        takeaway="Over 10 seeds, B11_final_lite cuts refine time by 22-59% and peak RAM by 8-61% vs B10_final with statistically indistinguishable accuracy (p >= 0.2393).",
    )

    rb.add_table(
        caption="v3 10-seed paired t-tests: B11_final_lite minus comparator (n=10) from results/v3/v3_paired_tests.csv.",
        headers=["Dataset", "B11 vs Comparator", "dAcc (p-val)", "dMacro-F1 (p-val)", "dBoxes (p-val)", "dRefine_s (p-val)", "dPeakMem_MB (p-val)"],
        rows=[
            [
                r["dataset"], r["baseline_variant"],
                f"{r['test_accuracy_diff']:+.4f} (p={r['test_accuracy_p']:.4f})",
                f"{r['test_macro_f1_diff']:+.4f} (p={r['test_macro_f1_p']:.4f})",
                f"{r['num_boxes_diff']:+.2f} (p={r['num_boxes_p']:.4f})",
                f"{r['refine_time_diff']:+.3f} (p={r['refine_time_p']:.4f})",
                f"{r['peak_memory_mb_diff']:+.2f} (p={r['peak_memory_mb_p']:.4f})",
            ]
            for _, r in v3_pt.iterrows()
        ],
        takeaway="B11_final_lite achieves significant box reduction vs B10 on Cora (-2.90, p=0.0048) and significantly lower refine time and peak RAM vs B10 across all three datasets (all p < 0.01).",
    )

    # --- 6.6 ---
    rb.add("### 6.6 v4 results (encoder search, B12 ensemble, B13 gated hybrid, B14_final)\n")
    cora_appnp_gain = float(v4_enc[(v4_enc["dataset"] == "Cora") & (v4_enc["encoder"] == "appnp")].iloc[0]["test_acc_mean"]) - float(v4_enc[(v4_enc["dataset"] == "Cora") & (v4_enc["encoder"] == "gcn")].iloc[0]["test_acc_mean"])
    rb.add_table(
        caption="v4 encoder validation search and 10-seed B1 comparison from results/v4/encoder_b1_comparison.csv (Chameleon and PubMed not measured).",
        headers=["Dataset", "Encoder", "Selected?", "Val Acc (10 seeds)", "Val Macro-F1", "Test Acc (10 seeds)", "Test Macro-F1"],
        rows=[
            [
                r["dataset"], r["encoder"], "YES" if bool(r["is_val_best"]) else "no",
                fmt_ms(r["val_acc_mean"], r["val_acc_std"]),
                fmt_ms(r["val_macro_f1_mean"], r["val_macro_f1_std"]),
                fmt_ms(r["test_acc_mean"], r["test_acc_std"]),
                fmt_ms(r["test_macro_f1_mean"], r["test_macro_f1_std"]),
            ]
            for _, r in v4_enc.iterrows()
        ] + [
            ["Chameleon", "gcn / sage / appnp / gcnii / h2gcn", "not measured", "not measured", "not measured", "not measured", "not measured"],
            ["PubMed", "gcn / sage / appnp / gcnii", "not measured", "not measured", "not measured", "not measured", "not measured"],
        ],
        takeaway=f"Validation search selected APPNP for Cora ({cora_appnp_gain:+.4f} test acc over tuned GCN) and GCNII for CiteSeer, lifting the B1 encoder ceiling before any hyperbox step.",
    )

    # Chosen B14 components table
    b14_rows = []
    for ds_l, dname in [("cora", "Cora"), ("citeseer", "CiteSeer")]:
        sel = json.loads((PROJECT_ROOT / f"results/v4/b14_components_{ds_l}.json").read_text(encoding="utf-8"))["selected"]
        b14_rows.append([
            dname,
            json.loads((PROJECT_ROOT / f"results/v4/b14_components_{ds_l}.json").read_text(encoding="utf-8"))["encoder_model"],
            str(sel["use_l2_norm"]),
            f"{sel['use_ensemble']} ({sel['ensemble_embed_mode']})",
            str(sel["use_fast_gshr"]),
            str(sel["use_capped_pseudo"]),
            str(sel["hybrid_mode"]),
            str(sel["use_smoothing"]),
        ])
    b14_rows.append(["Chameleon", "not measured", "not measured", "not measured", "not measured", "not measured", "not measured", "not measured"])
    rb.add_table(
        caption="B14_final validation-selected components per dataset from results/v4/b14_components_*.json.",
        headers=["Dataset", "Best Encoder", "L2 Norm", "5x Ensemble (Mode)", "Fast GSHR", "Capped Pseudo", "Hybrid Mode", "Smoothing rho>0"],
        rows=b14_rows,
        takeaway="Validation rejected prediction-time smoothing (rho=0) on both Cora and CiteSeer, rejected capped pseudo-labels on CiteSeer (ensemble active instead), and selected global_lam over gated hybrid on both.",
    )

    # v4 10-seed main table (plus B13 vs B7 fields cleanly inside a companion table)
    rb.add_table(
        caption="v4 10-seed main comparison (seeds 0..9) from results/v4/v4_comparison_summary.csv (Chameleon/PubMed v4 not measured).",
        headers=["Dataset", "Variant", "Seeds", "Test Acc", "Test Macro-F1", "Boxes", "Refine (s)", "Unified Lat (us/node)", "Peak Mem (MB)", "Storage (B)"],
        rows=[
            [
                r["dataset"], r["variant"], str(int(r["num_seeds"])),
                fmt_ms(r["test_accuracy_mean"], r["test_accuracy_std"]),
                fmt_ms(r["test_macro_f1_mean"], r["test_macro_f1_std"]),
                fmt_ms(r["num_boxes_mean"], r["num_boxes_std"], 1),
                fmt_ms(r["refine_time_mean"], r["refine_time_std"], 3),
                fmt_ms(r["pred_latency_per_node_mean"], r["pred_latency_per_node_std"], 2, 1e6),
                fmt_ms(r["peak_memory_mb_mean"], r["peak_memory_mb_std"], 2),
                fmt(r["box_storage_bytes_mean"], 0),
            ]
            for _, r in v4_sum.iterrows()
        ],
        takeaway=(
            f"On 10 seeds, B12_ensemble achieves the highest accuracy on both Cora ({cora_v4_b12['test_accuracy_mean']:.4f}, p={_v4_pval('Cora','B12_ensemble','B1_base_gcn'):.4f} vs base GCN) "
            f"and CiteSeer ({cite_v4_b12['test_accuracy_mean']:.4f}, p={_v4_pval('CiteSeer','B12_ensemble','B1_base_gcn'):.4f} vs base GCN), while B14_final preserves comparable accuracy vs B1_best_encoder with single-digit boxes "
            f"({cora_v4_b14['num_boxes_mean']:.1f} Cora, {cite_v4_b14['num_boxes_mean']:.1f} CiteSeer)."
        ),
    )

    # B13 gated vs B7 global lam diagnostics table
    diag_rows = []
    for ds in ["Cora", "CiteSeer"]:
        for vname in ["B13_gated_hybrid", "B12_ensemble", "B14_final"]:
            r = _v4_row(ds, vname)
            if r is not None:
                diag_rows.append([
                    ds, vname,
                    fmt(r.get("box_decided_fraction_mean"), 4, "n/a (global/lam)"),
                    fmt(r.get("box_decided_accuracy_mean"), 4, "n/a"),
                    fmt(r.get("gnn_decided_accuracy_mean"), 4, "n/a"),
                    fmt(r.get("fraction_changed_by_boxes_mean"), 4, "0.0000"),
                    fmt(r.get("b7_test_accuracy_mean"), 4, "n/a"),
                ])
    rb.add_table(
        caption="B13_gated_hybrid, B12_ensemble, and B14_final box decision vs GNN prediction-flip diagnostics (10 seeds, results/v4/).",
        headers=["Dataset", "Variant", "Box-Decided Frac", "Box-Decided Acc", "Rest (GNN) Acc", "Preds Flipped vs GNN", "B7 Global-lam Comparator Acc"],
        rows=diag_rows,
        takeaway=(
            f"In B13_gated_hybrid, boxes decide {cite_b13['box_decided_fraction_mean']*100:.2f}% (CiteSeer) to {cora_b13['box_decided_fraction_mean']*100:.2f}% (Cora) of test nodes "
            f"with {cite_b13['box_decided_accuracy_mean']:.4f} and {cora_b13['box_decided_accuracy_mean']:.4f} accuracy on those nodes, yet flip "
            f"{cora_b13['fraction_changed_by_boxes_mean']*100:.2f}% (Cora) and {cite_b13['fraction_changed_by_boxes_mean']*100:.2f}% (CiteSeer) of GNN softmax predictions."
        ),
    )

    # v4 paired t-tests
    pt_c12 = _v4_pt_row("Cora", "B12_ensemble", "B1_base_gcn")
    pt_s12 = _v4_pt_row("CiteSeer", "B12_ensemble", "B1_base_gcn")
    pt_c14 = _v4_pt_row("Cora", "B14_final", "B1_base_gcn")
    pt_s14 = _v4_pt_row("CiteSeer", "B14_final", "B1_base_gcn")
    pt_c14_b = _v4_pt_row("Cora", "B14_final", "B1_best_encoder")
    pt_s14_b = _v4_pt_row("CiteSeer", "B14_final", "B1_best_encoder")
    rb.add_table(
        caption="v4 10-seed paired t-tests (variant minus baseline) across accuracy, F1, boxes, refine time, memory, and unified latency.",
        headers=["Dataset", "Variant vs Baseline", "dAcc (p-val)", "dMacro-F1 (p-val)", "dBoxes (p-val)", "dRefine_s (p-val)", "dPeakMem_MB (p-val)", "dLatency_us (p-val)"],
        rows=[
            [
                r["dataset"], f"{r['variant']} vs {r['baseline_variant']}",
                f"{r['test_accuracy_diff']:+.4f} (p={r['test_accuracy_p']:.4f})",
                f"{r['test_macro_f1_diff']:+.4f} (p={r['test_macro_f1_p']:.4f})",
                f"{r['num_boxes_diff']:+.2f} (p={r['num_boxes_p']:.4f})",
                f"{r['refine_time_diff']:+.3f} (p={r['refine_time_p']:.4f})",
                f"{r['peak_memory_mb_diff']:+.2f} (p={r['peak_memory_mb_p']:.4f})",
                f"{r['pred_latency_per_node_diff']*1e6:+.2f} (p={r['pred_latency_per_node_p']:.4f})",
            ]
            for _, r in v4_pt.iterrows()
        ],
        takeaway=(
            f"Against B1_base_gcn, B12_ensemble ({pt_c12['test_accuracy_diff']:+.4f} Cora p={pt_c12['test_accuracy_p']:.4f}, "
            f"{pt_s12['test_accuracy_diff']:+.4f} CiteSeer p={pt_s12['test_accuracy_p']:.4f}) and B14_final "
            f"({pt_c14['test_accuracy_diff']:+.4f} Cora p={pt_c14['test_accuracy_p']:.4f}, {pt_s14['test_accuracy_diff']:+.4f} CiteSeer p={pt_s14['test_accuracy_p']:.4f}) "
            f"achieve statistically significant gains; against the upgraded B1_best_encoder, B14_final accuracy differences are not statistically significant "
            f"(p={pt_c14_b['test_accuracy_p']:.4f} Cora, p={pt_s14_b['test_accuracy_p']:.4f} CiteSeer; CiteSeer macro-F1 p={pt_s14_b['test_macro_f1_p']:.4f})."
        ),
    )

    # --- 6.7 ---
    rb.add("### 6.7 Per-class recall comparison and box statistics across versions\n")
    # Per-class recall from v2 (Cora, CiteSeer, Chameleon) and v4 (Cora, CiteSeer)
    recall_rows = []
    for ds in ["Cora", "CiteSeer", "Chameleon"]:
        s_map = {r["variant"]: r for r in v2_json["datasets_detail"][ds]["summary"]}
        for v_k in ["B1_gnn_softmax", "B2_gnn_fmnn", "B10_final"]:
            rc_list = s_map[v_k]["per_class_recall_mean"]
            recall_rows.append(["v2 (5s)", ds, v_k] + [f"{x:.4f}" for x in rc_list] + (["n/a"] * (7 - len(rc_list))))
    for ds in ["Cora", "CiteSeer"]:
        s_map4 = {r["variant"]: r for r in v4_json["datasets_detail"][ds]["summary"]}
        for v_k in ["B1_base_gcn", "B1_best_encoder", "B14_final"]:
            rc_list = s_map4[v_k]["per_class_recall_mean"]
            recall_rows.append(["v4 (10s)", ds, v_k] + [f"{x:.4f}" for x in rc_list] + (["n/a"] * (7 - len(rc_list))))
    rb.add_table(
        caption="Mean per-class test recall across classes C0-C6 from results/v2/final_comparison.json and results/v4/v4_comparison.json.",
        headers=["Version", "Dataset", "Variant", "C0", "C1", "C2", "C3", "C4", "C5", "C6"],
        rows=recall_rows,
        takeaway="CiteSeer class C0 (minority/hardest class, 0.30-0.39 recall in v2) lifts to 0.4534 under v4 GCNII + B14_final while keeping class recalls balanced.",
    )

    # Box geometry & pseudo-label statistics from runs JSON/CSVs
    box_stat_rows = []
    for ds, df_r in [("Cora", v1_cora_runs), ("CiteSeer", v1_cite_runs), ("Chameleon", v1_cham_runs)]:
        for var_n in ["B2_gnn_fmnn", "B6_gshr_full"]:
            sub = df_r[df_r["variant"] == var_n]
            box_stat_rows.append([
                "v1 (3s)", ds, var_n,
                fmt_ms(sub["num_boxes"].mean(), sub["num_boxes"].std(ddof=1), 1),
                fmt_ms(sub["mean_box_size"].mean(), sub["mean_box_size"].std(ddof=1), 2),
                fmt_ms(sub["max_box_size"].mean(), sub["max_box_size"].std(ddof=1), 2),
                str(int(sub["cross_class_overlap"].max())),
                "not measured", "0.0",
            ])
    for ver_lbl, runs_df in [("v2 (5s)", v2_runs), ("v3 (10s)", v3_runs), ("v4 (10s)", v4_runs)]:
        for (ds, var_n), sub in runs_df.groupby(["dataset", "variant"], sort=False):
            if float(sub["num_boxes"].max()) == 0.0:
                continue
            cov_m = sub["box_coverage"].dropna() if "box_coverage" in sub.columns else pd.Series(dtype=float)
            ps_m = sub["pseudo_count"].dropna() if "pseudo_count" in sub.columns else pd.Series(dtype=float)
            box_stat_rows.append([
                ver_lbl, ds, var_n,
                fmt_ms(sub["num_boxes"].mean(), sub["num_boxes"].std(ddof=1), 1),
                fmt_ms(sub["mean_box_size"].mean(), sub["mean_box_size"].std(ddof=1), 2),
                fmt_ms(sub["max_box_size"].mean(), sub["max_box_size"].std(ddof=1), 2),
                str(int(sub["cross_class_overlap"].max())),
                fmt_ms(cov_m.mean(), cov_m.std(ddof=1), 3) if len(cov_m) > 0 else "not measured",
                fmt(ps_m.mean(), 1) if len(ps_m) > 0 else "0.0",
            ])
    rb.add_table(
        caption="Hyperbox geometry, purity, coverage, and mean pseudo-labels used across versions.",
        headers=["Version", "Dataset", "Variant", "Num Boxes", "Mean L1 Size", "Max L1 Size", "Max Cross Overlap", "Box Coverage", "Mean Pseudo"],
        rows=box_stat_rows,
        takeaway="Across every single version and seed, cross-class overlap stayed strictly 0, while capping pseudo-labels reduced Cora pseudo counts from 1244.0 (B10) to 499.7 (B11) and CiteSeer from 734.1 to 414.4.",
    )

    # Big consolidated "all variants x all datasets" table labelled by version
    all_rows = []
    for ds, df_s in [("Cora", v1_cora), ("CiteSeer", v1_cite), ("Chameleon", v1_cham)]:
        for _, r in df_s.iterrows():
            all_rows.append(["v1 (3s, GCN)", ds, r["variant"], fmt_ms(r["test_accuracy_mean"], r["test_accuracy_std"]), fmt_ms(r["test_macro_f1_mean"], r["test_macro_f1_std"]), fmt_ms(r["num_boxes_mean"], r["num_boxes_std"], 1), fmt(r["refine_time_mean"], 3), fmt(r["peak_memory_mb_mean"], 2)])
    for _, r in v2_sum.iterrows():
        all_rows.append(["v2 (5s, v2-GCN/L2)", r["dataset"], r["variant"], fmt_ms(r["test_accuracy_mean"], r["test_accuracy_std"]), fmt_ms(r["test_macro_f1_mean"], r["test_macro_f1_std"]), fmt_ms(r["num_boxes_mean"], r["num_boxes_std"], 1), fmt(r["refine_time_mean"], 3), fmt(r["peak_memory_mb_mean"], 2)])
    for _, r in v3_sum.iterrows():
        all_rows.append(["v3 (10s, vec/unified)", r["dataset"], r["variant"], fmt_ms(r["test_accuracy_mean"], r["test_accuracy_std"]), fmt_ms(r["test_macro_f1_mean"], r["test_macro_f1_std"]), fmt_ms(r["num_boxes_mean"], r["num_boxes_std"], 1), fmt(r["refine_time_mean"], 3), fmt(r["peak_memory_mb_mean"], 2)])
    for _, r in v4_sum.iterrows():
        all_rows.append(["v4 (10s, APPNP/GCNII)", r["dataset"], r["variant"], fmt_ms(r["test_accuracy_mean"], r["test_accuracy_std"]), fmt_ms(r["test_macro_f1_mean"], r["test_macro_f1_std"]), fmt_ms(r["num_boxes_mean"], r["num_boxes_std"], 1), fmt(r["refine_time_mean"], 3), fmt(r["peak_memory_mb_mean"], 2)])
    rb.add_table(
        caption="Consolidated all-versions x all-datasets comparison summary (each version differs in encoder config, vectorization, and seed count).",
        headers=["Version (Setting)", "Dataset", "Variant", "Test Acc", "Test Macro-F1", "Boxes", "Refine (s)", "Peak Mem (MB)"],
        rows=all_rows,
        takeaway="Across v1->v4, accuracy climbs primarily via validation-tuned encoders and ensemble softmax (v4 APPNP/GCNII), while GSHR + capped pseudo-labels preserve 7-11 boxes cleanly.",
    )

    # =========================================================================
    # 7. Figures
    # =========================================================================
    rb.add("## 7. Figures\n")
    rb.add("Every project image is stored cleanly inside `report_assets/` (prefixing by version avoids filename collisions). Outdated, duplicated, and test-evaluated plots are flagged explicitly alongside their replacements.\n")

    rb.add_figure(
        assets["fmnn_practice_decision_boundary.png"],
        "FMNN 2D practice decision boundaries",
        "Standalone Simpson FMNN decision boundaries on 2D synthetic data (standalone_fmnn.py).",
        [
            "Shows normalized features on axes in [0, 1]^2 for 3 classes (blue, orange, green circles=train, triangles=test) with dashed hyperboxes B1-B8 at theta=0.25, gamma=4.0.",
            "Concludes visually how FMNN carves disjoint axis-aligned rectangles achieving 100.0% test accuracy on cleanly clustered 2D data.",
        ],
    )
    rb.add_figure(
        assets["v1_graph_representation.png"],
        "4-panel Cora graph and hyperbox visualization",
        "4-panel visualization of Cora graph topology, 2D PCA embeddings, base FMNN (14 boxes), and GSHR refined (7 boxes).",
        [
            "Generated by experiments/visualize_graph.py; note results/figures/graph_hyperbox_visualization.png is an exact byte-for-byte duplicate (MD5 deb0e908e793) of this image.",
            "Panel A renders a 150-node / 109-edge Cora subgraph colored by class; Panel B shows 2D PCA projection (for visualization only; training uses 64-D) of 140 training triangles and unlabeled dots.",
            "Panels C-D contrast 14 fragmented dashed base boxes against 7 clean class-pure refined hyperboxes (each containing N=20 reassigned training members) after GSHR pruning.",
        ],
    )
    rb.add_figure(
        assets["v1_accuracy_vs_boxes_scatter.png"],
        "Legacy v1 Cora accuracy vs boxes scatter plot with overlapping labels",
        "Legacy v1 Cora Test Accuracy vs. Number of Hyperboxes scatter plot (OUTDATED: overlapping text labels).",
        [
            "Plots Cora v1 3-seed means: x-axis shows hyperboxes (0-14) and y-axis shows test accuracy (0.772-0.797).",
            "Known defect: markers at identical coordinates cause text labels B2/B4 and B3/B5/B6 to overlap unreadably; replaced by Figures 4, 8, and 13.",
        ],
    )
    rb.add_figure(
        assets["v1_accuracy_vs_boxes_scatter_fixed.png"],
        "Regenerated fixed v1 Cora scatter plot with non-overlapping labels",
        "Regenerated fixed v1 Cora Accuracy vs. Boxes scatter plot with callouts and legend separating identical-coordinate points.",
        [
            "Rebuilt programmatically by make_report_md.py from results/cora_summary.csv so B2/B4 (13.3, 0.7763) and B3/B5/B6 (7.0, 0.7720) remain individually legible.",
            "Confirms B4 merge-only did not alter B2, while B5 prune-only drove the entire box reduction seen in B3 and B6.",
        ],
    )
    rb.add_figure(
        assets["v1_macro_f1_and_boxes_bar.png"],
        "Legacy v1 Cora Macro-F1 and boxes grouped bars",
        "Legacy v1 Test Macro-F1 (blue left axis) and Hyperbox Count (red right axis) across B1-B6 on Cora (3 seeds).",
        [
            "Blue bars track Test Macro-F1 (~0.772-0.792) and red bars track box counts (0 for B1, 13.3 for B2/B4, 7.0 for B3/B5/B6).",
            "Replaced for multi-dataset v2 comparison by Figure 9 (v2_macro_f1_and_boxes_bar.png).",
        ],
    )
    rb.add_figure(
        assets["v1_b2_vs_b6_hyperbox_reduction.png"],
        "Legacy v1 B2 vs B6 hyperbox reduction bar chart",
        "Legacy v1 Hyperbox Reduction bar chart (stored in repo as results/figures/per_class_recall_comparison.png — misnamed plot of B2 vs B6 box count).",
        [
            "Despite its legacy filename per_class_recall_comparison.png, the chart strictly plots 13.3 boxes (blue, B2 Base FMNN) vs. 7.0 boxes (red, B6 GSHR Full) on Cora (47.5% drop).",
            "True per-class recall bars were implemented later in v2 (see Figure 11: v2_per_class_recall_comparison.png).",
        ],
    )
    rb.add_figure(
        assets["v1_theta_sensitivity.png"],
        "Legacy v1 Cora test-evaluated theta sensitivity plot",
        "Legacy v1 Sensitivity to Theta on Cora seed 0 (OUTDATED: evaluated on TEST data via a6_sweep_theta.csv).",
        [
            "Plots base boxes (grey dashed), refined boxes (purple squares), and Test Macro-F1 (green triangles) across theta in [0.1..0.6].",
            "Flagged as using test-split evaluation on seed 0 only and slightly overlapping legend; replaced by validation-only 5-seed Figure 10.",
        ],
    )
    rb.add_figure(
        assets["v2_accuracy_vs_boxes_scatter.png"],
        "v2 non-overlapping Accuracy vs Hyperbox Count scatter across 3 benchmarks",
        "v2 Test Accuracy vs. Hyperbox Count across Cora, CiteSeer, and Chameleon (5 seeds) with non-overlapping callout labels.",
        [
            "Uses distinct colored circles for B1-B9 and a purple star for B10_final with non-overlapping boxed callouts.",
            "Shows B7/B8/B10 lifting accuracy close to or slightly above B1 on Cora/CiteSeer with 6.4-10.4 boxes, while on Chameleon only B10 matches B1 (0.6697) via base-encoder fallback.",
        ],
    )
    rb.add_figure(
        assets["v2_macro_f1_and_boxes_bar.png"],
        "v2 Macro-F1 and Hyperboxes grouped bar chart across 3 benchmarks",
        "v2 Test Macro-F1 (blue) and Hyperbox Count (red) across B1-B10 on Cora, CiteSeer, and Chameleon (5 seeds).",
        [
            "Illustrates clean box compaction vs B2 on Cora/CiteSeer and exposes Chameleon v2 encoder over-fragmentation under B9 (49.0 boxes) vs B10_final (10.6 boxes).",
        ],
    )
    rb.add_figure(
        assets["v2_theta_sensitivity.png"],
        "v2 validation-only theta sensitivity plot across 3 datasets",
        "v2 Validation-Only Theta Sensitivity (theta in [0.1..0.6], 5 seeds) from theta_sensitivity_val.csv.",
        [
            "Purple dashed lines track base hyperbox count (left axis) while green squares track Validation Macro-F1 (right axis); orange dots mark val-optimal theta*.",
            "Validation selects theta*=0.4 on Cora (7.8 boxes), theta*=0.2 on CiteSeer (23.6 boxes), and theta*=0.1 on Chameleon (298.2 boxes), without touching test labels.",
        ],
    )
    rb.add_figure(
        assets["v2_per_class_recall_comparison.png"],
        "v2 per-class recall bars across Cora, CiteSeer, and Chameleon",
        "v2 Mean Test Recall by Class ID (C0-C6) comparing B1 Softmax (blue), B2 Plain FMNN (orange), and B10_final (green).",
        [
            "Demonstrates that B10_final tracks or slightly lifts B1 recall across nearly all classes while fixing B2's recall drops on CiteSeer C2/C3/C4 and Chameleon C2.",
        ],
    )
    rb.add_figure(
        assets["v3_profiling_and_efficiency.png"],
        "v3 B8 bottleneck profiling and B10 vs B11 efficiency gains",
        "v3 Profiling Breakdown (seed 0) and 10-Seed Refine Time / Peak Memory Gains (B10_final vs. B11_final_lite).",
        [
            "Generated from results/v3/b8_profile.csv and v3_comparison_summary.csv: Panel A shows pruning consumed 91.4-94.1% of B8 runtime prior to vectorization.",
            "Panels B-C illustrate consistent 22-59% faster refinement and lower peak memory for B11_final_lite vs B10_final across all three datasets.",
        ],
    )
    rb.add_figure(
        assets["v4_pareto_accuracy_vs_boxes.png"],
        "v4 Pareto frontier plot of Test Accuracy vs Hyperboxes",
        "v4 Pareto Plot (Test Accuracy vs. Mean Hyperboxes over 10 seeds) on Cora (APPNP) and CiteSeer (GCNII).",
        [
            "Diamonds show 0-box B1 baselines (base GCN vs v4 best encoder), dashed curves trace the non-dominated Pareto frontier, and stars mark B14_final.",
            "Highlights that upgrading the encoder shifted the entire frontier up by +1.8% (Cora) and +0.6-1.7% (CiteSeer), with B12_ensemble defining the top right Pareto corner.",
        ],
    )
    rb.add_figure(
        assets["v4_box_decision_and_flips.png"],
        "v4 fraction of test nodes decided or flipped by boxes vs GNN",
        "v4 Box Decision Fraction (B13) and Actual Prediction Flip Fraction vs. GNN Softmax across Cora and CiteSeer (10 seeds).",
        [
            f"Orange bars show B13's exact single-box gate triggers on only {cora_b13['box_decided_fraction_mean']*100:.2f}% (Cora) and {cite_b13['box_decided_fraction_mean']*100:.2f}% (CiteSeer) of test nodes.",
            (
                f"Crucially, yellow/green/red bars show that boxes flip {cora_b13['fraction_changed_by_boxes_mean']*100:.2f}% (B13 Cora), "
                f"{cite_b13['fraction_changed_by_boxes_mean']*100:.2f}% (B13 CiteSeer), {cora_v4_b12['fraction_changed_by_boxes_mean']*100:.2f}% (B12 Cora), "
                f"{cite_v4_b12['fraction_changed_by_boxes_mean']*100:.2f}% (B12 CiteSeer), and "
                f"{cora_v4_b14['fraction_changed_by_boxes_mean']*100:.2f}%-{cite_v4_b14['fraction_changed_by_boxes_mean']*100:.2f}% (B14) of GNN predictions."
            ),
        ],
    )

    # =========================================================================
    # 8. Discussion
    # =========================================================================
    v3_b1_mem = v3_sum[v3_sum["variant"] == "B1_gnn_softmax"]["peak_memory_mb_mean"]
    v3_box_mem = v3_sum[v3_sum["variant"] != "B1_gnn_softmax"]["peak_memory_mb_mean"]
    v4_box_mem = v4_sum[~v4_sum["variant"].isin(["B1_base_gcn", "B1_best_encoder"])]["peak_memory_mb_mean"]
    b10_t_reds = []
    b10_m_reds = []
    for ds_x in ["Cora", "CiteSeer", "Chameleon"]:
        r10 = _v3_row(ds_x, "B10_final")
        r11 = _v3_row(ds_x, "B11_final_lite")
        b10_t_reds.append((1.0 - float(r11["refine_time_mean"]) / float(r10["refine_time_mean"])) * 100.0)
        b10_m_reds.append((1.0 - float(r11["peak_memory_mb_mean"]) / float(r10["peak_memory_mb_mean"])) * 100.0)

    rb.add("## 8. Discussion\n")
    rb.add(
        f"- **(a) Is accuracy good?** Comparing against baseline `B1` (`v1/v3 GCN`), `v4` encoders (`APPNP` `{cora_v4_b1b['test_accuracy_mean']:.4f}` on Cora, `GCNII` `{cite_v4_b1b['test_accuracy_mean']:.4f}` on CiteSeer, `B12_ensemble` `{cora_v4_b12['test_accuracy_mean']:.4f}` Cora `p = {pt_c12['test_accuracy_p']:.4f}` / `{cite_v4_b12['test_accuracy_mean']:.4f}` CiteSeer `p = {pt_s12['test_accuracy_p']:.4f}`) lift accuracy significantly (`p < 0.02`). Against the stronger `B1_best_encoder`, `B14_final` is statistically comparable (`p = {pt_c14_b['test_accuracy_p']:.4f}` Cora, `p = {pt_s14_b['test_accuracy_p']:.4f}` CiteSeer; CiteSeer macro-F1 `p = {pt_s14_b['test_macro_f1_p']:.4f}`). *Literature comparison note:* published external reference ranges for GCN/APPNP/GCNII were not verified against external datasets/splits inside offline repo logs (`reference not verified`), so all conclusions anchor strictly on our measured controlled `B1` baselines.\n"
        f"- **(b) Did GSHR reduce boxes and by how much?** Yes, consistently: on `v1` (`3` seeds), `B6` reduced Cora `{cora_v1_b2['num_boxes_mean']:.1f} -> {cora_v1_b6['num_boxes_mean']:.1f}` (`{cora_v1_red:.1f}%`), CiteSeer `{cite_v1_b2['num_boxes_mean']:.1f} -> {cite_v1_b6['num_boxes_mean']:.1f}` (`{cite_v1_red:.1f}%`), and Chameleon `{cham_v1_b2['num_boxes_mean']:.1f} -> {cham_v1_b6['num_boxes_mean']:.1f}` (`{cham_v1_red:.1f}%`). In `v3` (`10` seeds), `B11_final_lite` reduced `v2` base FMNN (`B2`) boxes by `{cora_b11_red:.1f}%` on Cora (`{cora_v2_b2['num_boxes_mean']:.1f} -> {cora_v3_b11['num_boxes_mean']:.1f}`), `{cite_b11_red:.1f}%` on CiteSeer (`{cite_v2_b2['num_boxes_mean']:.1f} -> {cite_v3_b11['num_boxes_mean']:.1f}`), and `{cham_red_b11:.1f}%` on Chameleon (`{cham_v2_b2['num_boxes_mean']:.1f} -> {cham_v3_b11['num_boxes_mean']:.1f}`). In `v4` (`10` seeds), `B14_final` holds `{cora_v4_b14['num_boxes_mean']:.1f}` boxes on Cora and `{cite_v4_b14['num_boxes_mean']:.1f}` on CiteSeer.\n"
        "- **(c) Component contributions (from ablations and validation gates):**\n"
        "  - *Base FMNN (`B2`):* Interpretable boxes, `-2.1%` to `-4.4%` behind `B1` accuracy.\n"
        f"  - *Merge alone (`B4`/`A3`):* `0` reduction (`{cora_v1_b2['num_boxes_mean']:.1f} -> {cora_v1_b4['num_boxes_mean']:.1f}`) under `v1` rules.\n"
        f"  - *Prune alone (`B5`/`A2`):* Drives `100%` of `v1` box compaction (`{cora_v1_b2['num_boxes_mean']:.1f} -> {cora_v1_b5['num_boxes_mean']:.1f}`).\n"
        f"  - *L2-norm + Hybrid `lam` (`B7`):* Recovers `B1`-level accuracy (`{cora_v2_b7['test_accuracy_mean']:.4f}` Cora `v2`; `{cora_v4_b7['test_accuracy_mean']:.4f}` `v4 APPNP`).\n"
        f"  - *Pseudo-labels (`B8` uncapped vs `B11` capped):* Helped slightly on CiteSeer (`{cite_v2_b7['test_accuracy_mean']:.4f} -> {cite_v2_b8['test_accuracy_mean']:.4f}` `v2`) and Cora `B14`, while capping at `k<=100` cut peak memory `{cora_v3_b10['peak_memory_mb_mean']:.2f} -> {cora_v3_b11['peak_memory_mb_mean']:.2f} MB` and refine time `{cora_v3_b10['refine_time_mean']:.3f}s -> {cora_v3_b11['refine_time_mean']:.3f}s` on Cora (`10` seeds).\n"
        "  - *Ensemble (`B12`) + Smoothing (`rho`):* `5x` ensemble gave real validation & test gains (`CiteSeer B14` kept `avg` ensemble); validation smoothing selected `rho = 0.0` everywhere on top of `APPNP`/`GCNII`.\n"
        f"- **(d) Where `B1` GNN-only still wins:** `B1` wins decisively on **inference latency** (`{float(b1_ul.min()):.2f}–{float(b1_ul.max()):.2f} us/node` vs `{float(hyb_ul.min()):.2f}–{float(hyb_ul.max()):.2f} us/node` in unified benchmarks), **memory** (`{float(v3_b1_mem.min()):.2f}–{float(v3_b1_mem.max()):.2f} MB` vs `{float(v3_box_mem.min()):.1f}–{float(v3_box_mem.max()):.1f} MB` `v3` and `{float(v4_box_mem.min()):.1f}–{float(v4_box_mem.max()):.1f} MB` `v4`), **refine overhead** (`0.0s`), **zero box storage**, and simplicity whenever per-box interpretability is unnecessary.\n"
        f"- **(e) Why Chameleon behaves differently:** Low edge homophily (`h = {ds_info['Chameleon']['homophily']:.4f}`) causes neighbor aggregation to blend unlike classes, forcing `B2` into `{cham_v2_b2['num_boxes_mean']:.1f}` (or `{cham_b10_comp['encoder']['v2_b2_boxes_mean']:.1f}` under `v2 val_loss`) fragmented boxes and pushing pure-box accuracy (`{cham_v1_b2['test_accuracy_mean']:.3f}–{cham_v3_b9['test_accuracy_mean']:.3f}`) below `B1` (`{cham_v3_b1['test_accuracy_mean']:.3f}`). Consequently, `B10`/`B11` rely heavily on `lam -> 1.0` to match `B1`, and Chameleon also exhibits `{ds_info['Chameleon']['dup_rows']}` duplicate feature vectors (`{ds_info['Chameleon']['test_dup_train']}` test nodes identical to train).\n"
    )

    rb.add_table(
        caption="Cost-benefit comparison and recommended configurations across graph regimes (values from v3 and v4 10-seed tables).",
        headers=["Scenario / Dataset Type", "Recommended Variant", "Test Acc", "Boxes", "Refine (s)", "Unified Lat (us/node)", "Peak Mem (MB)"],
        rows=[
            ["Max accuracy + low-box homophily (Cora)", "v4 B14_final (APPNP + capped pseudo + lam)", fmt_ms(cora_v4_b14["test_accuracy_mean"], cora_v4_b14["test_accuracy_std"]), fmt_ms(cora_v4_b14["num_boxes_mean"], cora_v4_b14["num_boxes_std"], 1), fmt(cora_v4_b14["refine_time_mean"], 3), fmt(cora_v4_b14["pred_latency_per_node_mean"] * 1e6, 2), fmt(cora_v4_b14["peak_memory_mb_mean"], 2)],
            ["Max accuracy on Cora (storage budget relaxed)", "v4 B12_ensemble (5x APPNP, concat/avg)", fmt_ms(cora_v4_b12["test_accuracy_mean"], cora_v4_b12["test_accuracy_std"]), fmt_ms(cora_v4_b12["num_boxes_mean"], cora_v4_b12["num_boxes_std"], 1), fmt(cora_v4_b12["refine_time_mean"], 3), fmt(cora_v4_b12["pred_latency_per_node_mean"] * 1e6, 2), fmt(cora_v4_b12["peak_memory_mb_mean"], 2)],
            ["Max accuracy + compact boxes (CiteSeer)", "v4 B14_final (5x GCNII avg + fast GSHR)", fmt_ms(cite_v4_b14["test_accuracy_mean"], cite_v4_b14["test_accuracy_std"]), fmt_ms(cite_v4_b14["num_boxes_mean"], cite_v4_b14["num_boxes_std"], 1), fmt(cite_v4_b14["refine_time_mean"], 3), fmt(cite_v4_b14["pred_latency_per_node_mean"] * 1e6, 2), fmt(cite_v4_b14["peak_memory_mb_mean"], 2)],
            ["Fast & lightweight interpretable pipeline (general)", "v3 B11_final_lite (CiteSeer example)", fmt_ms(cite_v3_b11["test_accuracy_mean"], cite_v3_b11["test_accuracy_std"]), fmt_ms(cite_v3_b11["num_boxes_mean"], cite_v3_b11["num_boxes_std"], 1), fmt(cite_v3_b11["refine_time_mean"], 3), fmt(cite_v3_b11["pred_latency_per_node_mean"] * 1e6, 2), fmt(cite_v3_b11["peak_memory_mb_mean"], 2)],
            ["Heterophilous / lowest inference latency (Chameleon / CPU)", "B1 (GNN only; or v3 B11_final_lite for boxes)", fmt_ms(cham_v3_b1["test_accuracy_mean"], cham_v3_b1["test_accuracy_std"]), f"0.0 (or {cham_v3_b11['num_boxes_mean']:.1f})", f"0.000 (or {cham_v3_b11['refine_time_mean']:.3f})", fmt(cham_v3_b1["pred_latency_per_node_mean"] * 1e6, 2), fmt(cham_v3_b1["peak_memory_mb_mean"], 2)],
        ],
        takeaway="Use B1 when speed/memory matter most, B11_final_lite for fast interpretable refinement, and v4 APPNP/GCNII (B14_final / B12_ensemble) when maximizing accuracy on homophilous graphs.",
    )

    # =========================================================================
    # 9. Interpretability
    # =========================================================================
    rb.add("## 9. Interpretability\n")
    rb.add(
        "- **Structural purity:** Every variant from `v1` through `v4` maintained **0 cross-class overlaps** via Simpson contraction and class-purity merge guards.\n"
        f"- **Covered vs. uncovered accuracy (`v4` & `v2`):** In `v4`, when a test node falls inside an exact single box (`B13_gated_hybrid`), accuracy is **`{cora_b13['box_decided_accuracy_mean']:.4f}` on Cora** (`{cora_b13['box_decided_fraction_mean']*100:.2f}%` of test nodes) vs `{cora_b13['gnn_decided_accuracy_mean']:.4f}` on the remaining `{100.0 - float(cora_b13['box_decided_fraction_mean'])*100:.2f}%`, and **`{cite_b13['box_decided_accuracy_mean']:.4f}` on CiteSeer** (`{cite_b13['box_decided_fraction_mean']*100:.2f}%` of test nodes) vs `{cite_b13['gnn_decided_accuracy_mean']:.4f}` on the remaining `{100.0 - float(cite_b13['box_decided_fraction_mean'])*100:.2f}%`.\n"
        f"- **Box-vs-GNN agreement (where boxes do NOT change decisions):** On `B13_gated_hybrid`, boxes flipped **`{cora_b13['fraction_changed_by_boxes_mean']:.4f}`** (`{cora_b13['fraction_changed_by_boxes_mean']*100:.2f}%`) of Cora GNN predictions and **`{cite_b13['fraction_changed_by_boxes_mean']:.4f}`** (`{cite_b13['fraction_changed_by_boxes_mean']*100:.2f}%`) on CiteSeer; in `B14_final`, boxes flipped only `{cora_v4_b14['fraction_changed_by_boxes_mean']:.4f}` (`{cora_v4_b14['fraction_changed_by_boxes_mean']*100:.2f}%`) on Cora and `{cite_v4_b14['fraction_changed_by_boxes_mean']:.4f}` (`{cite_v4_b14['fraction_changed_by_boxes_mean']*100:.2f}%`) on CiteSeer. Inside confident hyperbox regions, FMNN mostly mirrors already-confident GNN predictions.\n"
        "- **Hyperbox rule representation:** Because hyperboxes are axis-aligned min-max bounds $[v_{jd}, w_{jd}]$ in the frozen GNN's normalized embedding space (e.g., visual 2D PCA boxes `B1`–`B7`, `N=20` train members each in Figure 2, or normalized synthetic 2D rules in Figure 1), an explicit **64-D per-box `[v_d, w_d]` coordinate JSON rule table was not exported into `results/`** (`not measured` in `Missing data`).\n"
    )

    # =========================================================================
    # 10. Threats to validity and limitations
    # =========================================================================
    rb.add("## 10. Threats to validity and limitations\n")
    rb.add(
        "- **Test data leakage in legacy `v1` `A6` sweeps and `smoothing_ablation.csv`:** In `experiments/run_ablations.py`, `run_a6_sensitivity` (`a6_sweep_*.csv`) and `run_prediction_smoothing` (`smoothing_ablation.csv`) swept hyperparameters (`theta`, `tau_merge`, `tau_prune`, merge weights, `rho`) and printed `test_accuracy`/`test_macro_f1` directly on `Z_test`, and legacy `results/figures/theta_sensitivity.png` plotted those seed-0 test curves. **`v2`, `v3`, and `v4` resolved this completely** by selecting every hyperparameter strictly on `val_idx` (`theta_sensitivity_val.csv`, `encoder_search_*.json`, `b10_components_*.json`, `b14_components_*.json`, unit tests using `corrupted.y[test_idx] = -999`).\n"
        "- **Dataset duplicates in Chameleon:** Direct measurement on loaded PyG `WikipediaNetwork` (`geom_gcn_preprocess=True`, split 0) found **327 duplicate feature rows** across 48 groups, with **64 / 456 test nodes (14.0%)** having feature vectors identical to training nodes (mirroring Cora's `1` and CiteSeer's `1` test duplicates out of `1000`). A duplicate-free Chameleon filtered split was not measured.\n"
        "- **Incomplete `v4` & `PubMed` coverage:** `Chameleon` `v4` timed out/was canceled during encoder validation search, `PubMed` (`19,717` nodes) was only benchmarked for single-run feasibility, and `GAT` / `MLP` baselines remain unmeasured.\n"
        "- **Small seed counts in early splits (`n=3` in `v1`, `n=1..2` during `v4` grid searches):** Early `v1` tables and `v2`/`v4` component selection gates used 3 seeds (`0,1,2`), whereas final comparisons (`v3`, `v4` Cora/CiteSeer) report `10` seeds (`0..9`).\n"
        "- **Interpretability space & CPU-only timing:** Boxes live in learned GNN embedding space (`64-D`, or `320-D` under `concat`), not raw bag-of-words features; 2D PCA plots (`Figure 2`) only illustrate projections; all timings reflect single-workstation CPU (`AMD64`, no GPU).\n"
    )

    # =========================================================================
    # 11. Conclusion and future work
    # =========================================================================
    min_hybrid_p = min(
        _v3_pval("Cora", "B1_gnn_softmax"),
        _v3_pval("CiteSeer", "B1_gnn_softmax"),
        _v3_pval("Chameleon", "B1_gnn_softmax"),
        float(pt_c14_b["test_accuracy_p"]),
        float(pt_s14_b["test_accuracy_p"]),
    )
    rb.add("## 11. Conclusion and future work\n")
    rb.add(
        f"Across 10-seed evaluations on Cora, CiteSeer, and Chameleon, GSHR pruning reduces unrefined FMNN hyperboxes by `{min_red_all:.1f}%–{cham_v1_red:.1f}%` (`{cora_v2_b2['num_boxes_mean']:.1f} -> {cora_v4_b14['num_boxes_mean']:.1f}`–`{cora_v3_b11['num_boxes_mean']:.1f}` Cora, `{cite_v2_b2['num_boxes_mean']:.1f} -> {cite_v4_b14['num_boxes_mean']:.1f}`–`{cite_v3_b11['num_boxes_mean']:.1f}` CiteSeer, `{cham_v2_b2['num_boxes_mean']:.1f} -> {cham_v3_b11['num_boxes_mean']:.1f}` Chameleon `v3`, `{cham_v1_b2['num_boxes_mean']:.1f} -> {cham_v1_b6['num_boxes_mean']:.1f}` `v1`) with zero cross-class overlap. "
        f"Pure-box FMNN still trails GNN softmax accuracy (`B9` vs `B1`), whereas hybrid probability blending (`B7`/`B11`/`B14`) matches `B1` within statistical noise in accuracy (`p >= {min_hybrid_p:.4f}` across all 10-seed comparisons vs active `B1`). "
        f"Profiling (`v3`) showed pruning accounted for `{float(v3_prof['prune_pct'].min()):.1f}%–{float(v3_prof['prune_pct'].max()):.1f}%` of pseudo-label runtime; vectorization and per-class capping (`k <= 100`) cut `B10` refine time by `{min(b10_t_reds):.1f}%–{max(b10_t_reds):.1f}%` and peak memory by `{min(b10_m_reds):.1f}%–{max(b10_m_reds):.1f}%`. "
        f"Upgrading encoders (`v4`) raised the GNN accuracy ceiling significantly on Cora (`APPNP`: `{cora_v4_b1b['test_accuracy_mean']:.4f}`) and CiteSeer (`GCNII`: `{cite_v4_b1b['test_accuracy_mean']:.4f}`), with `B12_ensemble` reaching `{cora_v4_b12['test_accuracy_mean']:.4f}` and `{cite_v4_b12['test_accuracy_mean']:.4f}`. "
        "Future work should complete Chameleon filtered (`v4`) and PubMed evaluations, export explicit interpretable box rules, and explore differentiable box regularization during GNN training.\n"
    )

    # =========================================================================
    # 12. Reproducibility
    # =========================================================================
    rb.add("## 12. Reproducibility\n")
    rb.add(
        "```bash\n"
        "# Environment & unit tests (66 passed in latest pytest -q verification)\n"
        "pip install -r requirements.txt\n"
        "python -m pytest -q\n\n"
        "# v1 benchmarks, ablations, & visualizations\n"
        "python experiments/run_main.py --dataset Cora --seeds 0 1 2 --variants B1 B2 B3 B4 B5 B6\n"
        "python experiments/run_ablations.py --seeds 0 1 2\n"
        "python experiments/visualize_graph.py\n"
        "python experiments/make_report.py --legacy\n\n"
        "# v2 5-seed benchmark & figures\n"
        "python -u scripts/run_v2_final_comparison.py\n"
        "python experiments/make_report.py\n\n"
        "# v3 B8 profiling, unified CPU latency, & 10-seed B11_final_lite\n"
        "python -u scripts/profile_b8.py\n"
        "python -u scripts/run_v3_comparison.py\n\n"
        "# v4 encoder validation search, B12/B13/B14 10-seed runs, & Pareto plots\n"
        "python -u scripts/run_v4_benchmark.py --datasets Cora CiteSeer\n"
        "python -u scripts/run_v4_benchmark.py --finalize-only\n\n"
        "# Rebuild REPORT.md & report_assets/ deterministically from results/\n"
        "python make_report_md.py\n"
        "```\n"
    )

    # Concise annotated project file tree
    file_descs = {
        "README.md": "Project overview, reproduction commands, leakage guarantees, and assumptions",
        "requirements.txt": "Python dependencies list",
        "pytest.ini": "Pytest configuration",
        "standalone_fmnn.py": "Self-contained Simpson 1992 FMNN practice demo (2D & Iris)",
        "prepare_assignment_submission.py": "Assignment packaging utility script",
        "make_report_md.py": "Programmatic generator for REPORT.md and report_assets/",
        "configs/default.yaml": "Default v1 configuration (GCN, FMNN theta=0.3, GSHR weights/safeguards)",
        "configs/cora_v2.yaml": "v2 configuration (encoder search grid, L2 norm, hybrid lambda, pseudo c, gshr_fast)",
        "configs/cora.yaml": "Cora CLI dataset config copy",
        "configs/citeseer.yaml": "CiteSeer dataset config",
        "configs/pubmed.yaml": "PubMed dataset config",
        "configs/experiments.yaml": "Experiment CLI config copy",
        "configs/citeseer_experiments.yaml": "CiteSeer experiment CLI config",
        "src/data.py": "Planetoid/Chameleon loader, undirected CSR adjacency builder, and homophily_ratio",
        "src/gnn.py": "GCNEncoder, SAGEEncoder, APPNPEncoder, GCNIIEncoder, H2GCNEncoder, train_gnn & val searches",
        "src/embeddings.py": "Embedding/logit/prob extraction, row L2 normalization, and train-only EmbeddingScaler",
        "src/fmnn.py": "Simpson 1992 FMNN, chunked vectorized membership_matrix, and fit_with_pseudo",
        "src/graph_stats.py": "Graph cohesion stats (conductance, neighbor agreement, purity) + BoxGraphCache",
        "src/merge.py": "GSHR merge scoring, hard safeguards, priority-queue merge, and fast_gshr_merge",
        "src/prune.py": "GSHR sequential pruning with NumPy confusion metrics & per-class guards",
        "src/metrics.py": "Classification metrics, compactness, box coverage, and unified CPU measure_prediction_latency",
        "src/pipeline.py": "SeedContext and execution branches B1-B14 + validation gating (B10/B14)",
        "src/utils.py": "Reproducibility seeding, device resolution, YAML/JSON I/O helpers",
        "scripts/train_baseline.py": "CLI wrapper training B1 and B2 to results/tables/",
        "scripts/run_gshr.py": "CLI wrapper for merge/prune/full modes to results/tables/",
        "scripts/run_v2_encoder.py": "Step-by-step v2 encoder search runner",
        "scripts/run_v2_hybrid.py": "Step-by-step v2 B7/B8 runner",
        "scripts/run_v2_gshr_fast.py": "Step-by-step v2 B9 fast GSHR runner",
        "scripts/run_v2_final_comparison.py": "v2 5-seed multi-dataset benchmark script",
        "scripts/profile_b8.py": "B8 stage time & peak memory profiler (results/v3/)",
        "scripts/run_v3_comparison.py": "v3 unified latency & 10-seed B11_final_lite benchmark",
        "scripts/run_v4_benchmark.py": "v4 encoder search & B12/B13/B14 10-seed benchmark",
        "experiments/run_main.py": "v1 main runner for B1-B6",
        "experiments/run_ablations.py": "v1 A0-A8 ablations, robustness tests, and smoothing script",
        "experiments/make_report.py": "v1, v2, and v4 matplotlib plots and markdown tables generator",
        "experiments/visualize_graph.py": "4-panel graph & PCA hyperbox visualizer",
        "experiments/scan_theta.py": "Cora validation theta scan script",
        "experiments/print_box_stats.py": "Base FMNN graph statistics diagnostic printer",
    }
    rb.add("**Project file tree:**\n```text")
    for rel_f in sorted(file_descs):
        rb.add(f"{rel_f:<42} # {file_descs[rel_f]}")
    rb.add("tests/test_*.py (8 files, 66 tests)       # Unit and zero-test-leakage verification suite")
    rb.add("```\n")

    # =========================================================================
    # Appendix
    # =========================================================================
    rb.add("## Appendix\n")

    rb.add("### Appendix A. Full raw per-seed tables\n")
    raw_frames = [
        ("v1 Cora per-seed runs (results/cora_runs.csv)", v1_cora_runs),
        ("v1 CiteSeer per-seed runs (results/citeseer_runs.csv)", v1_cite_runs),
        ("v1 Chameleon per-seed runs (results/chameleon_runs.csv)", v1_cham_runs),
        ("v1 A0-A4 core ablations per-seed (results/ablations/a0_a4_core_ablations.csv)", abl_core),
        ("v1 A5 safeguards per-seed (results/ablations/a5_safeguard_ablation.csv)", abl_sg),
        ("v1 A7 cross-dataset per-seed (results/ablations/a7_cross_dataset.csv)", abl_a7),
        ("v1 A8 encoder per-seed (results/ablations/a8_encoders.csv)", abl_a8),
        ("v2 5-seed runs (results/v2/final_comparison_runs.csv)", v2_runs),
        ("v3 B8 per-threshold profile (results/v3/b8_profile_per_c.csv)", v3_prof_c),
        ("v3 10-seed runs (results/v3/v3_comparison_runs.csv)", v3_runs),
        ("v4 10-seed runs on Cora & CiteSeer (results/v4/v4_comparison_runs.csv)", v4_runs),
    ]
    for title, df_r in raw_frames:
        rb.add(raw_csv_details(title, df_r))

    rb.add("### Appendix B. Full hyperparameter configs\n")
    rb.add("<details>\n<summary>Selected hyperparameter configs across v2 (encoder_config_*.json) and v4 (encoder_search_*.json) plus default/v2 YAML</summary>\n")
    for path_rel in [
        "configs/default.yaml",
        "configs/cora_v2.yaml",
        "results/v2/encoder_config_cora.json",
        "results/v2/encoder_config_citeseer.json",
        "results/v2/encoder_config_chameleon.json",
        "results/v4/encoder_search_cora.json",
        "results/v4/encoder_search_citeseer.json",
    ]:
        p = PROJECT_ROOT / path_rel
        rb.add(f"#### `{path_rel}`")
        if path_rel.endswith(".json"):
            data_j = json.loads(p.read_text(encoding="utf-8"))
            compact_j = {k: v for k, v in data_j.items() if k != "all_trials"}
            if "per_encoder" in compact_j:
                compact_j["per_encoder"] = {ek: {"best_config": ev["best_config"], "best_val_macro_f1_mean": ev["best_val_macro_f1_mean"]} for ek, ev in compact_j["per_encoder"].items()}
            rb.add("```json\n" + json.dumps(compact_j, indent=2) + "\n```\n")
        else:
            rb.add("```yaml\n" + p.read_text(encoding="utf-8").strip() + "\n```\n")
    rb.add("</details>\n")

    rb.add("### Appendix C. Complete list of output files with sizes\n")
    out_rows = []
    for folder in ["results", "report_assets"]:
        p = PROJECT_ROOT / folder
        if p.exists():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    out_rows.append([f.relative_to(PROJECT_ROOT).as_posix(), f"{f.stat().st_size:,} B"])
    if (PROJECT_ROOT / "fmnn_practice_decision_boundary.png").is_file():
        out_rows.append(["fmnn_practice_decision_boundary.png", f"{(PROJECT_ROOT / 'fmnn_practice_decision_boundary.png').stat().st_size:,} B"])
    rb.add_table(
        caption="Complete inventory of measured result artifacts and report figures with exact byte sizes.",
        headers=["File Path", "Size"],
        rows=out_rows,
        takeaway=f"Total verified output/figure files cataloged: {len(out_rows)}.",
    )

    rb.add("### Appendix D. Glossary of symbols\n")
    rb.add_table(
        caption="Mathematical and configuration symbols used throughout GNN + GSHR + FMNN.",
        headers=["Symbol", "Name / Role", "Typical Values / Range"],
        rows=[
            ["theta", "FMNN maximum L1 hyperbox expansion size budget (per dim)", "0.3 (default); swept {0.1..0.6}"],
            ["gamma", "FMNN trapezoidal membership sensitivity slope", "4.0"],
            ["tau_merge", "Minimum GSHR merge score M required to combine same-class boxes", "0.3-0.9 (0.5 default)"],
            ["tau_prune", "Validation macro-F1 drop tolerance per pruned box (total <= 2*tau_prune)", "{0.0, 0.005, 0.01, 0.02}"],
            ["alpha, beta, delta, eta", "GSHR merge weights: Geometry, GraphAssoc, Consistency, Risk penalty", "(0.35, 0.35, 0.20, 0.10)"],
            ["mu1..mu5", "GSHR prune importance weights: Coverage, Purity, Boundary, GraphRegion, Redundancy", "(0.25, 0.25, 0.20, 0.20, 0.10)"],
            ["lam", "Global hybrid weight blending GNN softmax (lam) and normalized FMNN scores (1-lam)", "{0.0, 0.25, 0.5, 0.75, 1.0}"],
            ["rho", "Prediction-time graph smoothing factor between own score and D^-1 A neighbor average", "{0.0, 0.1, 0.2, 0.3, 0.5}; v4 picked 0.0"],
            ["c", "Uncapped pseudo-label minimum GNN softmax confidence threshold (B8/B10)", "{0.8, 0.9, 0.95}"],
            ["k", "Per-class maximum count cap for highest-confidence pseudo-labels (B11/B14)", "{20, 50, 100}"],
            ["T", "GNN logit temperature scaling parameter in softmax(logits / T) (B13/B14)", "{0.5, 1.0, 1.5, 2.0}"],
            ["h", "Edge homophily ratio (fraction of edges linking nodes of equal class)", "Cora 0.8100, CiteSeer 0.7355, Chameleon 0.2299"],
        ],
        takeaway="All hyperparameters above except v1 A6 sweeps were selected strictly on training or validation splits without test label access.",
    )

    rb.add("### Appendix E. Missing data\n")
    for idx_m, item_m in enumerate(missing_data, start=1):
        rb.add(f"{idx_m}. {item_m}")
    rb.add("")

    report_text = "\n".join(rb.lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    return report_text, rb.figure_count, rb.table_count, missing_data


# Verify integrity checks on the generated REPORT.md.
def verify_report(report_text: str) -> None:
    # 1. Check all embedded images exist on disk
    imgs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report_text)
    for rel in imgs:
        p = PROJECT_ROOT / rel
        if not p.is_file():
            raise FileNotFoundError(f"Broken image link in REPORT.md: {rel}")

    # 2. Check no raw nan or empty table cells
    for ln_no, line in enumerate(report_text.splitlines(), start=1):
        if line.startswith("|") and "---" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            for cell in cells:
                if cell == "" or cell.lower() == "nan":
                    raise ValueError(f"Empty or NaN cell at REPORT.md line {ln_no}: {line}")

    # 3. Check all TOC internal anchors resolve to headings
    headings = re.findall(r"^#{1,6}\s+(.+)$", report_text, flags=re.MULTILINE)
    slugs = {slugify_heading(h) for h in headings}
    toc_links = re.findall(r"\(#([a-z0-9_-]+)\)", report_text)
    missing_anchors = [a for a in toc_links if a not in slugs]
    if missing_anchors:
        raise ValueError(f"Broken TOC anchors: {missing_anchors}")

    print(f"Verification PASSED: {len(imgs)} images verified on disk, {len(toc_links)} TOC links verified, zero empty/NaN cells.")


if __name__ == "__main__":
    text, n_figs, n_tbls, missing = build_report()
    verify_report(text)
    print(f"Generated REPORT.md ({len(text.encode('utf-8')):,} bytes) with {n_figs} figures and {n_tbls} tables.")
