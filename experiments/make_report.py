"""Report generation and visualization script for GSHR v1 and v2 benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import load_json


# Parse CLI flags so legacy v1 reporting remains available while v2/v4 are clean.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GSHR tables and figures.")
    parser.add_argument("--legacy", action="store_true", help="Run legacy v1 report into results/")
    parser.add_argument("--v2-dir", type=str, default="results/v2", help="Directory containing final_comparison.json")
    parser.add_argument("--v4-dir", type=str, default="", help="Optional v4 directory containing v4_comparison.json")
    return parser.parse_args()


# Place text annotations on a scatter plot iteratively so labels do not overlap points or each other.
def place_non_overlapping_labels(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    labels: list[str],
    x_span: float,
    y_span: float,
) -> None:
    offsets = [
        (0.03 * x_span, 0.04 * y_span),
        (0.03 * x_span, -0.05 * y_span),
        (-0.18 * x_span, 0.04 * y_span),
        (-0.18 * x_span, -0.05 * y_span),
        (0.04 * x_span, 0.09 * y_span),
        (-0.20 * x_span, 0.09 * y_span),
        (0.04 * x_span, -0.10 * y_span),
        (-0.20 * x_span, -0.10 * y_span),
    ]
    placed_boxes: list[tuple[float, float, float, float]] = []
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()

    for x, y, text in zip(xs, ys, labels):
        w_est = max(0.09 * x_span, 0.020 * len(text) * x_span)
        h_est = 0.055 * y_span
        best_pos = (x + 0.02 * x_span, y + 0.02 * y_span)
        best_score = float("inf")

        for dx, dy in offsets:
            tx = float(np.clip(x + dx, min_x + 0.01 * x_span, max_x - w_est - 0.01 * x_span))
            ty = float(np.clip(y + dy, min_y + 0.01 * y_span, max_y - h_est - 0.01 * y_span))
            rect = (tx, ty, tx + w_est, ty + h_est)

            overlap_penalty = 0.0
            for px0, py0, px1, py1 in placed_boxes:
                ox = max(0.0, min(rect[2], px1) - max(rect[0], px0))
                oy = max(0.0, min(rect[3], py1) - max(rect[1], py0))
                overlap_penalty += ox * oy / max(1e-9, w_est * h_est)

            # Penalize landing directly over any scatter point
            point_penalty = 0.0
            for qx, qy in zip(xs, ys):
                if rect[0] - 0.015 * x_span <= qx <= rect[2] + 0.015 * x_span and rect[1] - 0.015 * y_span <= qy <= rect[3] + 0.015 * y_span:
                    point_penalty += 1.0

            dist_penalty = abs(dx) / max(1e-9, x_span) + abs(dy) / max(1e-9, y_span)
            score = overlap_penalty * 100.0 + point_penalty * 10.0 + dist_penalty
            if score < best_score:
                best_score = score
                best_pos = (tx, ty)

        placed_boxes.append((best_pos[0], best_pos[1], best_pos[0] + w_est, best_pos[1] + h_est))
        ax.annotate(
            text,
            xy=(x, y),
            xytext=best_pos,
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#888888", alpha=0.85),
            arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7, shrinkA=2, shrinkB=3),
            zorder=6,
        )


# Generate the 4 required v2 plots across Cora, CiteSeer, and Chameleon into results/v2/figures.
def generate_v2_plots(v2_dir: Path, comp_data: dict[str, Any]) -> None:
    fig_dir = v2_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    datasets = comp_data["datasets"]
    detail = comp_data["datasets_detail"]

    variant_short = {
        "B1_gnn_softmax": "B1",
        "B2_gnn_fmnn": "B2",
        "B6_gshr_full": "B6",
        "B7_hybrid": "B7",
        "B8_hybrid_pseudo": "B8",
        "B9_gshr_fast": "B9",
        "B10_final": "B10",
    }
    palette = {
        "B1_gnn_softmax": "#4e79a7",
        "B2_gnn_fmnn": "#f28e2b",
        "B6_gshr_full": "#e15759",
        "B7_hybrid": "#76b7b2",
        "B8_hybrid_pseudo": "#59a14f",
        "B9_gshr_fast": "#edc948",
        "B10_final": "#b07aa1",
    }

    # (a) Accuracy vs number of boxes scatter across datasets with non-overlapping labels.
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 5.2), dpi=300)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sum_rows = detail[ds]["summary"]
        xs = [float(r["num_boxes_mean"]) for r in sum_rows]
        ys = [float(r["test_accuracy_mean"]) for r in sum_rows]
        lbls = [variant_short.get(r["variant"], r["variant"]) for r in sum_rows]
        for r, x, y, lb in zip(sum_rows, xs, ys, lbls):
            col = palette.get(r["variant"], "#333333")
            marker = "*" if r["variant"] == "B10_final" else "o"
            sz = 190 if r["variant"] == "B10_final" else 100
            ax.scatter(x, y, s=sz, c=col, marker=marker, edgecolors="black", linewidths=0.8, zorder=5)

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_pad = max(2.5, 0.22 * (x_max - x_min + 1.0))
        y_pad = max(0.012, 0.25 * (y_max - y_min + 0.01))
        ax.set_xlim(x_min - 0.15 * x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        place_non_overlapping_labels(
            ax, xs, ys, lbls,
            x_span=(ax.get_xlim()[1] - ax.get_xlim()[0]),
            y_span=(ax.get_ylim()[1] - ax.get_ylim()[0]),
        )
        ax.set_title(f"{ds} (h={detail[ds]['homophily_ratio']:.2f})", fontweight="bold")
        ax.set_xlabel("Number of Hyperboxes", fontweight="bold")
        ax.set_ylabel("Test Accuracy", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("Test Accuracy vs. Hyperbox Count Across Benchmarks", fontweight="bold", y=1.01)
    fig.tight_layout()
    plt.savefig(fig_dir / "accuracy_vs_boxes_scatter.png", bbox_inches="tight")
    plt.close()

    # (b) Grouped bars for Macro-F1 and Hyperboxes with distinct colours and clear legend.
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 5.0), dpi=300)
    if len(datasets) == 1:
        axes = [axes]
    handles_all, labels_all = None, None
    for ax1, ds in zip(axes, datasets):
        sum_rows = detail[ds]["summary"]
        x_pos = np.arange(len(sum_rows))
        f1_vals = [float(r["test_macro_f1_mean"]) for r in sum_rows]
        box_vals = [float(r["num_boxes_mean"]) for r in sum_rows]
        short_lbls = [variant_short.get(r["variant"], r["variant"]) for r in sum_rows]

        b1 = ax1.bar(x_pos - 0.18, f1_vals, width=0.36, color="#1f77b4", alpha=0.9, label="Test Macro-F1")
        ax1.set_ylabel("Test Macro-F1", color="#1f77b4", fontweight="bold")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        f1_min = max(0.0, min(f1_vals) - 0.05)
        f1_max = min(1.0, max(f1_vals) + 0.04)
        ax1.set_ylim(f1_min, f1_max)

        ax2 = ax1.twinx()
        b2 = ax2.bar(x_pos + 0.18, box_vals, width=0.36, color="#d62728", alpha=0.85, label="Num Hyperboxes")
        ax2.set_ylabel("Number of Hyperboxes", color="#d62728", fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax2.set_ylim(0, max(box_vals) * 1.25 + 1.0)

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(short_lbls, fontweight="bold")
        ax1.set_title(ds, fontweight="bold")
        ax1.grid(axis="y", linestyle=":", alpha=0.4)
        if handles_all is None:
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            handles_all, labels_all = h1 + h2, l1 + l2
    if handles_all:
        fig.legend(handles_all, labels_all, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Test Macro-F1 and Hyperbox Count by Variant", fontweight="bold", y=1.10)
    fig.tight_layout()
    plt.savefig(fig_dir / "macro_f1_and_boxes_bar.png", bbox_inches="tight")
    plt.close()

    # (c) Theta sensitivity plot chosen strictly by validation macro-F1.
    theta_csv = v2_dir / "theta_sensitivity_val.csv"
    if theta_csv.is_file():
        df_th = pd.read_csv(theta_csv)
        ds_list = [d for d in datasets if d in df_th["dataset"].unique()]
        fig, axes = plt.subplots(1, len(ds_list), figsize=(6.0 * len(ds_list), 4.8), dpi=300)
        if len(ds_list) == 1:
            axes = [axes]
        th_handles, th_labels = None, None
        for ax1, ds in zip(axes, ds_list):
            sub = df_th[df_th["dataset"] == ds].sort_values("theta")
            thetas = sub["theta"].to_numpy()
            vf1 = sub["val_macro_f1_mean"].to_numpy()
            nboxes = sub["num_boxes_mean"].to_numpy()

            ax1.plot(thetas, nboxes, "o--", color="#6a3d9a", lw=2.0, label="Base FMNN Boxes")
            ax1.set_xlabel("Max Hyperbox Size (theta)", fontweight="bold")
            ax1.set_ylabel("Number of Hyperboxes", color="#6a3d9a", fontweight="bold")
            ax1.tick_params(axis="y", labelcolor="#6a3d9a")

            ax2 = ax1.twinx()
            ax2.plot(thetas, vf1, "s-", color="#2ca02c", lw=2.0, label="Validation Macro-F1")
            ax2.set_ylabel("Validation Macro-F1", color="#2ca02c", fontweight="bold")
            ax2.tick_params(axis="y", labelcolor="#2ca02c")

            best_sub = sub[sub["is_val_best"] == True]
            if not best_sub.empty:
                b_th = float(best_sub.iloc[0]["theta"])
                b_vf1 = float(best_sub.iloc[0]["val_macro_f1_mean"])
                ax2.scatter([b_th], [b_vf1], s=140, color="#ff7f0e", edgecolors="black", zorder=7, label="Val Best theta*")
                ax2.axvline(b_th, color="#ff7f0e", linestyle=":", alpha=0.7)

            ax1.set_title(f"{ds} (Validation Only)", fontweight="bold")
            ax1.grid(True, linestyle=":", alpha=0.4)
            if th_handles is None:
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                th_handles, th_labels = h1 + h2, l1 + l2
        if th_handles:
            fig.legend(th_handles, th_labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.05))
        fig.suptitle("Theta Sensitivity Selected Strictly by Validation Macro-F1", fontweight="bold", y=1.10)
        fig.tight_layout()
        plt.savefig(fig_dir / "theta_sensitivity.png", bbox_inches="tight")
        plt.close()

    # (d) Per-class recall comparison for B1 vs B2 vs B10_final across datasets.
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.8), dpi=300)
    if len(datasets) == 1:
        axes = [axes]
    rc_colors = {"B1_gnn_softmax": "#4e79a7", "B2_gnn_fmnn": "#f28e2b", "B10_final": "#59a14f"}
    rc_labels = {"B1_gnn_softmax": "B1 (Softmax)", "B2_gnn_fmnn": "B2 (Plain FMNN)", "B10_final": "B10_final"}
    for ax, ds in zip(axes, datasets):
        sum_map = {r["variant"]: r for r in detail[ds]["summary"]}
        n_cls = int(detail[ds]["num_classes"])
        x_cls = np.arange(n_cls)
        width = 0.25
        for idx, var_key in enumerate(["B1_gnn_softmax", "B2_gnn_fmnn", "B10_final"]):
            if var_key in sum_map and "per_class_recall_mean" in sum_map[var_key]:
                recalls = sum_map[var_key]["per_class_recall_mean"]
                ax.bar(x_cls + (idx - 1) * width, recalls, width=width, color=rc_colors[var_key], label=rc_labels[var_key])
        ax.set_xticks(x_cls)
        ax.set_xticklabels([f"C{c}" for c in range(n_cls)], fontweight="bold")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Class ID", fontweight="bold")
        ax.set_ylabel("Mean Test Recall", fontweight="bold")
        ax.set_title(ds, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if ds == datasets[0]:
            ax.legend(loc="lower right", frameon=True, fontsize=8.5)
    fig.suptitle("Per-Class Test Recall Comparison: B1 vs B2 vs B10_final", fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.savefig(fig_dir / "per_class_recall_comparison.png", bbox_inches="tight")
    plt.close()

    print(f"Saved v2 plots into {fig_dir}")


# Generate markdown report tables directly from final_comparison.json and encoder configs.
def build_v2_markdown_report(v2_dir: Path, comp_data: dict[str, Any]) -> str:
    md: list[str] = []
    md.append("# GSHR v2 Experimental Report & Multi-Dataset Comparison\n")
    md.append(f"Evaluated on datasets `{', '.join(comp_data['datasets'])}` across seeds `{comp_data['seeds']}` (all selection on validation only; test evaluated once).\n")

    for ds in comp_data["datasets"]:
        d_info = comp_data["datasets_detail"][ds]
        md.append(f"## {ds} (Nodes={d_info['num_nodes']}, Classes={d_info['num_classes']}, Homophily h={d_info['homophily_ratio']:.4f})\n")
        sel = d_info["b10_components"]["selected"]
        md.append(
            f"**B10_final validation-selected components:** "
            f"`best_encoder={sel['use_best_encoder']}`, `l2_norm={sel['use_l2_norm']}`, "
            f"`fast_gshr={sel['use_fast_gshr']}`, `hybrid={sel['use_hybrid']}`, `pseudo={sel['use_pseudo']}`\n"
        )

        md.append("| Variant | Test Acc | Test Macro-F1 | Boxes | Refine Time (s) | Latency (us/node) | Peak Mem (MB) | Box Storage (B) |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in d_info["summary"]:
            acc_s = f"{r['test_accuracy_mean']:.4f} ± {r['test_accuracy_std']:.4f}"
            f1_s = f"{r['test_macro_f1_mean']:.4f} ± {r['test_macro_f1_std']:.4f}"
            bx_s = f"{r['num_boxes_mean']:.1f} ± {r['num_boxes_std']:.1f}"
            rt_s = f"{r['refine_time_mean']:.3f} ± {r['refine_time_std']:.3f}"
            lat_s = f"{r['pred_latency_per_node_mean'] * 1e6:.2f} ± {r['pred_latency_per_node_std'] * 1e6:.2f}"
            mem_s = f"{r['peak_memory_mb_mean']:.2f} ± {r['peak_memory_mb_std']:.2f}"
            st_s = f"{r['box_storage_bytes_mean']:.0f} ± {r['box_storage_bytes_std']:.0f}"
            md.append(f"| **{r['variant']}** | {acc_s} | {f1_s} | {bx_s} | {rt_s} | {lat_s} | {mem_s} | {st_s} |")
        md.append("")

        c_info = d_info.get("b10_vs_b1_comparison", {})
        if c_info:
            pa = c_info["paired_test_accuracy"]
            pf = c_info["paired_test_macro_f1"]
            md.append(
                f"- **B10_final vs B1:** Accuracy gap `{c_info['acc_gap_b10_minus_b1']:+.4f}` "
                f"(paired t-test: `t={pa['t_stat']:+.3f}`, `p={pa['p_value']:.4f}`, `n={pa['n']}`), "
                f"Macro-F1 gap `{c_info['f1_gap_b10_minus_b1']:+.4f}` (`t={pf['t_stat']:+.3f}`, `p={pf['p_value']:.4f}`)\n"
                f"- **Box reduction vs B2:** `{c_info['box_reduction_vs_b2_pct']:.1f}%` "
                f"(`{c_info['b2_boxes_mean']:.1f}` -> `{c_info['b10_boxes_mean']:.1f}`)\n"
            )

    theta_csv = v2_dir / "theta_sensitivity_val.csv"
    if theta_csv.is_file():
        df_th = pd.read_csv(theta_csv)
        md.append("## Validation-Only Theta Sensitivity (`theta_sensitivity_val.csv`)\n")
        md.append("| Dataset | Optimal theta* (Val) | Val Accuracy | Val Macro-F1 | Base Boxes |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        for ds in comp_data["datasets"]:
            sub = df_th[(df_th["dataset"] == ds) & (df_th["is_val_best"] == True)]
            if not sub.empty:
                r = sub.iloc[0]
                md.append(
                    f"| **{ds}** | {r['theta']:.1f} | {r['val_accuracy_mean']:.4f} | "
                    f"{r['val_macro_f1_mean']:.4f} | {r['num_boxes_mean']:.1f} |"
                )
        md.append("")

    return "\n".join(md)


# Legacy v1 plotting preserved behind --legacy flag.
def generate_plots(results_dir: Path, fig_dir: Path) -> None:
    """Generate publication-ready figures for legacy v1 GSHR report."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "cora_summary.csv"
    if summary_path.is_file():
        df_sum = pd.read_csv(summary_path)
        variants = df_sum["variant"].tolist()
        clean_vars = [v.replace("_", "\n") for v in variants]
        f1_means = df_sum["test_macro_f1_mean"].tolist()
        box_means = df_sum["num_boxes_mean"].tolist()

        fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
        color = "tab:blue"
        ax1.set_xlabel("Pipeline Variant", fontweight="bold")
        ax1.set_ylabel("Test Macro-F1", color=color, fontweight="bold")
        ax1.bar(np.arange(len(variants)) - 0.2, f1_means, width=0.4, color=color, alpha=0.85, label="Test Macro-F1")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_ylim(0.70, 0.83)

        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Number of Hyperboxes", color=color, fontweight="bold")
        ax2.bar(np.arange(len(variants)) + 0.2, box_means, width=0.4, color=color, alpha=0.85, label="Box Count")
        ax2.tick_params(axis="y", labelcolor=color)
        ax1.set_xticks(range(len(variants)))
        ax1.set_xticklabels(clean_vars, fontsize=9)
        plt.title("Test Macro-F1 and Hyperbox Count across Variants (Cora)", fontweight="bold")
        fig.tight_layout()
        plt.savefig(fig_dir / "macro_f1_and_boxes_bar.png")
        plt.close()


# Compute Pareto-optimal indices maximizing y (accuracy) while minimizing x (boxes).
def pareto_frontier_indices(xs: list[float], ys: list[float]) -> list[int]:
    order = sorted(range(len(xs)), key=lambda i: (xs[i], -ys[i]))
    front: list[int] = []
    best_y = -float("inf")
    for idx in order:
        if ys[idx] > best_y + 1e-12:
            front.append(idx)
            best_y = ys[idx]
    return front


# Generate Pareto plot (accuracy vs boxes with non-overlapping labels) and box-decision charts in results/v4/figures.
def generate_v4_plots(v4_dir: Path, v4_data: dict[str, Any]) -> None:
    fig_dir = v4_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    datasets = v4_data["datasets"]
    detail = v4_data["datasets_detail"]

    short_names = {
        "B1_base_gcn": "B1(gcn)",
        "B1_best_encoder": "B1(best)",
        "B11_final_lite": "B11",
        "B7_hybrid": "B7",
        "B12_ensemble": "B12",
        "B13_gated_hybrid": "B13",
        "B14_final": "B14",
    }
    colors = {
        "B1_base_gcn": "#7f7f7f",
        "B1_best_encoder": "#4e79a7",
        "B11_final_lite": "#b07aa1",
        "B7_hybrid": "#76b7b2",
        "B12_ensemble": "#59a14f",
        "B13_gated_hybrid": "#f28e2b",
        "B14_final": "#e15759",
    }

    # (1) Pareto plot: Test Accuracy vs Hyperbox Count with non-overlapping labels.
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.8 * len(datasets), 5.2), dpi=300)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sum_rows = detail[ds]["summary"]
        xs = [float(r["num_boxes_mean"]) for r in sum_rows]
        ys = [float(r["test_accuracy_mean"]) for r in sum_rows]
        lbls = [short_names.get(r["variant"], r["variant"]) for r in sum_rows]

        for r, x, y in zip(sum_rows, xs, ys):
            v = r["variant"]
            marker = "*" if v == "B14_final" else ("D" if "B1_" in v else "o")
            sz = 210 if v == "B14_final" else 105
            ax.scatter(x, y, s=sz, c=colors.get(v, "#333333"), marker=marker, edgecolors="black", linewidths=0.8, zorder=5)

        p_idx = pareto_frontier_indices(xs, ys)
        if p_idx:
            px = [xs[i] for i in p_idx]
            py = [ys[i] for i in p_idx]
            ax.plot(px, py, linestyle="--", color="#222222", lw=1.6, alpha=0.85, zorder=4, label="Pareto Frontier")

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_pad = max(2.5, 0.25 * (x_max - x_min + 1.0))
        y_pad = max(0.015, 0.28 * (y_max - y_min + 0.01))
        ax.set_xlim(x_min - 0.18 * x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        place_non_overlapping_labels(
            ax, xs, ys, lbls,
            x_span=(ax.get_xlim()[1] - ax.get_xlim()[0]),
            y_span=(ax.get_ylim()[1] - ax.get_ylim()[0]),
        )
        ax.set_title(f"{ds} ({detail[ds].get('best_encoder', 'enc')})", fontweight="bold")
        ax.set_xlabel("Mean Hyperboxes", fontweight="bold")
        ax.set_ylabel("Mean Test Accuracy (10 seeds)", fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="lower right", fontsize=8.5, frameon=True)

    fig.suptitle("Pareto Plot: Test Accuracy vs. Number of Hyperboxes (v4)", fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.savefig(fig_dir / "pareto_accuracy_vs_boxes.png", bbox_inches="tight")
    plt.close()

    # (2) Box decision & prediction flip fractions across variants where applicable.
    fig, ax = plt.subplots(figsize=(max(7.5, 2.2 * len(datasets)), 4.6), dpi=300)
    x_pos = np.arange(len(datasets))
    width = 0.20
    metrics_spec = [
        ("B13_gated_hybrid", "box_decided_fraction_mean", "B13 Box-Decided Frac", "#f28e2b"),
        ("B13_gated_hybrid", "fraction_changed_by_boxes_mean", "B13 Flipped vs GNN", "#edc948"),
        ("B12_ensemble", "fraction_changed_by_boxes_mean", "B12 Flipped vs Ens-GNN", "#59a14f"),
        ("B14_final", "fraction_changed_by_boxes_mean", "B14 Flipped vs Best-GNN", "#e15759"),
    ]
    for j, (vname, key, label, col) in enumerate(metrics_spec):
        vals = []
        for ds in datasets:
            smap = {r["variant"]: r for r in detail[ds]["summary"]}
            vals.append(float(smap.get(vname, {}).get(key, 0.0)))
        ax.bar(x_pos + (j - 1.5) * width, vals, width=width, color=col, label=label, alpha=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Fraction of Test Nodes", fontweight="bold")
    ax.set_title("Where Boxes Decide or Change Predictions vs. GNN Softmax", fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(frameon=True, fontsize=8.5, ncol=2)
    fig.tight_layout()
    plt.savefig(fig_dir / "box_decision_and_flips.png", bbox_inches="tight")
    plt.close()
    print(f"Saved v4 plots into {fig_dir}")


# Build markdown report tables summarizing v4 encoder search, gated hybrid, 10-seed stats, and p-values.
def build_v4_markdown_report(v4_dir: Path, v4_data: dict[str, Any]) -> str:
    md: list[str] = ["# GSHR v4 Experimental Report\n"]
    md.append(f"Datasets: `{', '.join(v4_data['datasets'])}` | Seeds: `{v4_data['seeds']}` | Selection strictly on validation macro-F1.\n")

    enc_csv = v4_dir / "encoder_b1_comparison.csv"
    if enc_csv.is_file():
        df_enc = pd.read_csv(enc_csv)
        md.append("## 1. Per-Dataset Encoder Comparison (Validation-Selected Configs, 10-Seed B1)\n")
        md.append("| Dataset | Encoder | Best Val Acc | Best Val F1 | Test Acc | Test Macro-F1 | Selected |")
        md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
        for _, r in df_enc.iterrows():
            md.append(
                f"| {r['dataset']} | {r['encoder']} | {r['val_acc_mean']:.4f} ± {r['val_acc_std']:.4f} "
                f"| {r['val_macro_f1_mean']:.4f} ± {r['val_macro_f1_std']:.4f} "
                f"| {r['test_acc_mean']:.4f} ± {r['test_acc_std']:.4f} "
                f"| {r['test_macro_f1_mean']:.4f} ± {r['test_macro_f1_std']:.4f} | {'YES' if r['is_val_best'] else ''} |"
            )
        md.append("")

    for ds in v4_data["datasets"]:
        d_info = v4_data["datasets_detail"][ds]
        md.append(f"## {ds} (Best Encoder: `{d_info.get('best_encoder', 'gcn')}`)\n")
        md.append(f"- **B14_final validation-selected components:** `{d_info.get('b14_components', {}).get('selected', {})}`\n")
        md.append("| Variant | Test Acc | Test Macro-F1 | Boxes | Refine (s) | Lat (us/node) | Peak Mem (MB) | Storage (B) |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in d_info["summary"]:
            acc_s = f"{r['test_accuracy_mean']:.4f} ± {r['test_accuracy_std']:.4f}"
            f1_s = f"{r['test_macro_f1_mean']:.4f} ± {r['test_macro_f1_std']:.4f}"
            bx_s = f"{r['num_boxes_mean']:.1f} ± {r['num_boxes_std']:.1f}"
            rt_s = f"{r['refine_time_mean']:.3f} ± {r['refine_time_std']:.3f}"
            lt_s = f"{r['pred_latency_per_node_mean'] * 1e6:.2f} ± {r['pred_latency_per_node_std'] * 1e6:.2f}"
            mm_s = f"{r['peak_memory_mb_mean']:.2f} ± {r['peak_memory_mb_std']:.2f}"
            st_s = f"{r['box_storage_bytes_mean']:.0f}"
            md.append(f"| **{r['variant']}** | {acc_s} | {f1_s} | {bx_s} | {rt_s} | {lt_s} | {mm_s} | {st_s} |")
        md.append("")

        # Box decision breakdown if present
        smap = {r["variant"]: r for r in d_info["summary"]}
        if "B13_gated_hybrid" in smap:
            b13 = smap["B13_gated_hybrid"]
            md.append(
                f"- **B13_gated_hybrid diagnostics:** box-decided fraction `{b13.get('box_decided_fraction_mean', 0.0):.3f}`, "
                f"accuracy on box-decided `{b13.get('box_decided_accuracy_mean', float('nan')):.4f}` vs rest "
                f"`{b13.get('gnn_decided_accuracy_mean', float('nan')):.4f}`, "
                f"predictions flipped vs GNN `{b13.get('fraction_changed_by_boxes_mean', 0.0):.4f}`, "
                f"vs B7 global lam Test Acc `{b13.get('b7_test_accuracy_mean', float('nan')):.4f}`.\n"
            )

    pt_csv = v4_dir / "v4_paired_tests.csv"
    if pt_csv.is_file():
        df_pt = pd.read_csv(pt_csv)
        md.append("## Paired t-Tests (10 Seeds vs B1 and B11)\n")
        md.append("| Dataset | Variant vs Baseline | dAcc (p-val) | dMacro-F1 (p-val) | dBoxes (p-val) | dRefineTime_s (p-val) | dPeakMem_MB (p-val) | dLatency_us (p-val) |")
        md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for _, r in df_pt.iterrows():
            md.append(
                f"| {r['dataset']} | {r['variant']} vs {r['baseline_variant']} "
                f"| {r['test_accuracy_diff']:+.4f} (p={r['test_accuracy_p']:.4f}) "
                f"| {r['test_macro_f1_diff']:+.4f} (p={r['test_macro_f1_p']:.4f}) "
                f"| {r['num_boxes_diff']:+.2f} (p={r['num_boxes_p']:.4f}) "
                f"| {r['refine_time_diff']:+.3f} (p={r['refine_time_p']:.4f}) "
                f"| {r['peak_memory_mb_diff']:+.2f} (p={r['peak_memory_mb_p']:.4f}) "
                f"| {r['pred_latency_per_node_diff'] * 1e6:+.2f} (p={r['pred_latency_per_node_p']:.4f}) |"
            )
        md.append("")

    return "\n".join(md)


# Main entry point creating v2 (or v4 when --v4-dir is set) figures and report_tables.md.
def main() -> None:
    args = parse_args()
    if args.legacy:
        results_dir = Path("results")
        generate_plots(results_dir, results_dir / "figures")
        print(f"Legacy figures regenerated in {results_dir / 'figures'}")
        return

    if args.v4_dir:
        v4_dir = Path(args.v4_dir)
        v4_path = v4_dir / "v4_comparison.json"
        if not v4_path.is_file():
            raise FileNotFoundError(f"Missing {v4_path}. Run scripts/run_v4_benchmark.py first.")
        v4_data = load_json(v4_path)
        generate_v4_plots(v4_dir, v4_data)
        md_v4 = build_v4_markdown_report(v4_dir, v4_data)
        report_v4 = v4_dir / "report_tables.md"
        report_v4.write_text(md_v4, encoding="utf-8")
        print(f"Saved v4 markdown report to {report_v4}")
        return

    v2_dir = Path(args.v2_dir)
    comp_path = v2_dir / "final_comparison.json"
    if not comp_path.is_file():
        raise FileNotFoundError(f"Missing {comp_path}. Run scripts/run_v2_final_comparison.py first.")

    comp_data = load_json(comp_path)
    generate_v2_plots(v2_dir, comp_data)
    md_text = build_v2_markdown_report(v2_dir, comp_data)
    report_path = v2_dir / "report_tables.md"
    report_path.write_text(md_text, encoding="utf-8")
    print(f"Saved markdown report to {report_path}")


if __name__ == "__main__":
    main()
