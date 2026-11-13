"""Ablation studies, robustness analyses, and smoothing experiments for GSHR."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedShuffleSplit

from src.data import GraphData, homophily_ratio, load_dataset
from src.embeddings import extract_embeddings, fit_transform_all
from src.fmnn import FMNN
from src.merge import GraphContext, gshr_merge
from src.metrics import classification_metrics, compactness_metrics
from src.pipeline import prepare_seed_context, run_single_seed
from src.prune import gshr_prune
from src.utils import get_device, load_config, set_seed


def run_a0_a4_ablations(cfg: dict[str, Any], data: GraphData, seeds: list[int], out_dir: Path) -> pd.DataFrame:
    """Run core ablations A0 (base), A1 (beta=0), A2 (prune only), A3 (merge only), A4 (geo only)."""
    print("\n--- Running Core Ablations A0 to A4 ---")
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        seed_ctx = prepare_seed_context(cfg, data, seed)

        # A0: base GNN -> ordinary FMNN (B2)
        r_a0 = run_single_seed(cfg, data, seed, "B2", seed_ctx=seed_ctx)
        r_a0["ablation"] = "A0_base_fmnn"
        rows.append(r_a0)

        # A1: beta = 0 in merge (no graph term in merge, full prune enabled)
        cfg_a1 = copy.deepcopy(cfg)
        cfg_a1["merge"]["beta"] = 0.0
        r_a1 = run_single_seed(cfg_a1, data, seed, "B6", seed_ctx=seed_ctx)
        r_a1["ablation"] = "A1_beta_zero"
        rows.append(r_a1)

        # A2: no graph-aware merging (prune only, B5)
        r_a2 = run_single_seed(cfg, data, seed, "B5", seed_ctx=seed_ctx)
        r_a2["ablation"] = "A2_no_merge_prune_only"
        rows.append(r_a2)

        # A3: no graph-aware pruning (merge only, B4)
        r_a3 = run_single_seed(cfg, data, seed, "B4", seed_ctx=seed_ctx)
        r_a3["ablation"] = "A3_no_prune_merge_only"
        rows.append(r_a3)

        # A4: geometry-only merge and prune (B3)
        r_a4 = run_single_seed(cfg, data, seed, "B3", seed_ctx=seed_ctx)
        r_a4["ablation"] = "A4_geometry_only"
        rows.append(r_a4)

        # Full GSHR for reference (B6)
        r_full = run_single_seed(cfg, data, seed, "B6", seed_ctx=seed_ctx)
        r_full["ablation"] = "GSHR_full"
        rows.append(r_full)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "a0_a4_core_ablations.csv", index=False)
    print(f"Saved core ablations to {out_dir / 'a0_a4_core_ablations.csv'}")
    return df


def run_a5_safeguard_ablation(cfg: dict[str, Any], data: GraphData, seeds: list[int], out_dir: Path) -> pd.DataFrame:
    """Ablation A5: Turn off class purity safeguard and relax cross-class overlap constraints."""
    print("\n--- Running Ablation A5: Safeguards Disabled ---")
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        seed_ctx = prepare_seed_context(cfg, data, seed)
        base_fmnn = seed_ctx.base_fmnn

        train_idx = data.train_idx.cpu().numpy()
        test_idx = data.test_idx.cpu().numpy()
        val_idx = data.val_idx.cpu().numpy()

        Z_train = seed_ctx.Z_scaled[train_idx]
        y_train = data.y[data.train_idx].cpu().numpy()
        train_ids = train_idx.tolist()

        Z_val = seed_ctx.Z_scaled[val_idx]
        y_val = data.y[data.val_idx].cpu().numpy()

        Z_test = seed_ctx.Z_scaled[test_idx]
        y_test = data.y[data.test_idx].cpu().numpy()

        # Leakage guard: non-train masked to -1
        y_masked = np.full(data.num_nodes, -1, dtype=np.int64)
        y_masked[train_idx] = y_train
        train_mask_arr = np.zeros(data.num_nodes, dtype=bool)
        train_mask_arr[train_idx] = True
        graph_ctx = GraphContext(adj_csr=data.adj_csr, degree=data.degree, y=y_masked, train_mask=train_mask_arr)

        # Safe run (default)
        r_safe = run_single_seed(cfg, data, seed, "B6", seed_ctx=seed_ctx)
        rows.append({
            "seed": seed,
            "setting": "Safeguards_ON (Default)",
            "test_accuracy": r_safe["test_accuracy"],
            "test_macro_f1": r_safe["test_macro_f1"],
            "num_boxes": r_safe["num_boxes"],
            "cross_class_overlaps": r_safe["cross_class_overlap"],
        })

        # Relaxed run: class_purity=False, high max_volume_inflation=1.0, and allow overlap during merge
        cfg_unsafe = copy.deepcopy(cfg)
        cfg_unsafe["safeguards"]["class_purity"] = False
        cfg_unsafe["merge"]["max_volume_inflation"] = 1.0
        cfg_unsafe["merge"]["tau_merge"] = 0.1  # very aggressive merge

        # Fit with relaxed checks
        merged_fmnn, _ = gshr_merge(base_fmnn, graph_ctx, cfg_unsafe["merge"])
        pruned_fmnn, _ = gshr_prune(merged_fmnn, graph_ctx, cfg_unsafe["prune"], Z_train, y_train, train_ids, Z_val, y_val)

        test_preds = pruned_fmnn.predict(Z_test)
        test_m = classification_metrics(y_test, test_preds, data.num_classes)
        compact = compactness_metrics(pruned_fmnn)

        rows.append({
            "seed": seed,
            "setting": "Safeguards_OFF (Aggressive)",
            "test_accuracy": test_m["accuracy"],
            "test_macro_f1": test_m["macro_f1"],
            "num_boxes": compact["num_boxes"],
            "cross_class_overlaps": compact["cross_class_overlap"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "a5_safeguard_ablation.csv", index=False)
    print(f"Saved A5 safeguard ablation to {out_dir / 'a5_safeguard_ablation.csv'}")
    return df


def run_a6_sensitivity(cfg: dict[str, Any], data: GraphData, seed: int, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Ablation A6: Sensitivity analysis across theta, tau_merge, tau_prune, and score weights."""
    print("\n--- Running Ablation A6: Sensitivity Sweeps ---")
    sweeps: dict[str, pd.DataFrame] = {}

    seed_ctx = prepare_seed_context(cfg, data, seed)
    train_idx = data.train_idx.cpu().numpy()
    val_idx = data.val_idx.cpu().numpy()
    test_idx = data.test_idx.cpu().numpy()

    Z_train = seed_ctx.Z_scaled[train_idx]
    y_train = data.y[data.train_idx].cpu().numpy()
    train_ids = train_idx.tolist()
    Z_val = seed_ctx.Z_scaled[val_idx]
    y_val = data.y[data.val_idx].cpu().numpy()
    Z_test = seed_ctx.Z_scaled[test_idx]
    y_test = data.y[data.test_idx].cpu().numpy()

    y_masked = np.full(data.num_nodes, -1, dtype=np.int64)
    y_masked[train_idx] = y_train
    train_mask_arr = np.zeros(data.num_nodes, dtype=bool)
    train_mask_arr[train_idx] = True
    graph_ctx = GraphContext(adj_csr=data.adj_csr, degree=data.degree, y=y_masked, train_mask=train_mask_arr)

    # 1. Sweep Theta
    theta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    theta_rows = []
    for th in theta_list:
        fmnn = FMNN(theta=th, gamma=4.0)
        fmnn.fit(Z_train, y_train, train_ids)
        m_cfg = dict(cfg["merge"])
        m_cfg["theta"] = th
        m_cfg["safeguards"] = cfg["safeguards"]
        merged_fmnn, _ = gshr_merge(fmnn, graph_ctx, m_cfg)
        p_cfg = dict(cfg["prune"])
        p_cfg["safeguards"] = cfg["safeguards"]
        pruned_fmnn, _ = gshr_prune(merged_fmnn, graph_ctx, p_cfg, Z_train, y_train, train_ids, Z_val, y_val)

        test_preds = pruned_fmnn.predict(Z_test)
        tm = classification_metrics(y_test, test_preds, data.num_classes)
        theta_rows.append({
            "theta": th,
            "num_boxes_base": fmnn.num_boxes(),
            "num_boxes_refined": pruned_fmnn.num_boxes(),
            "test_accuracy": tm["accuracy"],
            "test_macro_f1": tm["macro_f1"],
        })
    df_theta = pd.DataFrame(theta_rows)
    df_theta.to_csv(out_dir / "a6_sweep_theta.csv", index=False)
    sweeps["theta"] = df_theta

    # 2. Sweep tau_merge (fixed tau_prune=0.005)
    tau_merge_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    tau_m_rows = []
    for tm_val in tau_merge_list:
        m_cfg = dict(cfg["merge"])
        m_cfg["tau_merge"] = tm_val
        m_cfg["theta"] = 0.3
        m_cfg["safeguards"] = cfg["safeguards"]
        merged_fmnn, _ = gshr_merge(seed_ctx.base_fmnn, graph_ctx, m_cfg)
        p_cfg = dict(cfg["prune"])
        p_cfg["tau_prune"] = 0.005
        p_cfg["safeguards"] = cfg["safeguards"]
        pruned_fmnn, _ = gshr_prune(merged_fmnn, graph_ctx, p_cfg, Z_train, y_train, train_ids, Z_val, y_val)
        test_preds = pruned_fmnn.predict(Z_test)
        tm = classification_metrics(y_test, test_preds, data.num_classes)
        tau_m_rows.append({
            "tau_merge": tm_val,
            "num_boxes": pruned_fmnn.num_boxes(),
            "test_accuracy": tm["accuracy"],
            "test_macro_f1": tm["macro_f1"],
        })
    df_tau_m = pd.DataFrame(tau_m_rows)
    df_tau_m.to_csv(out_dir / "a6_sweep_tau_merge.csv", index=False)
    sweeps["tau_merge"] = df_tau_m

    # 3. Sweep tau_prune (fixed tau_merge=0.5)
    tau_prune_list = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
    tau_p_rows = []
    for tp_val in tau_prune_list:
        m_cfg = dict(cfg["merge"])
        m_cfg["tau_merge"] = 0.5
        m_cfg["theta"] = 0.3
        m_cfg["safeguards"] = cfg["safeguards"]
        merged_fmnn, _ = gshr_merge(seed_ctx.base_fmnn, graph_ctx, m_cfg)
        p_cfg = dict(cfg["prune"])
        p_cfg["tau_prune"] = tp_val
        p_cfg["safeguards"] = cfg["safeguards"]
        pruned_fmnn, _ = gshr_prune(merged_fmnn, graph_ctx, p_cfg, Z_train, y_train, train_ids, Z_val, y_val)
        test_preds = pruned_fmnn.predict(Z_test)
        tm = classification_metrics(y_test, test_preds, data.num_classes)
        tau_p_rows.append({
            "tau_prune": tp_val,
            "num_boxes": pruned_fmnn.num_boxes(),
            "test_accuracy": tm["accuracy"],
            "test_macro_f1": tm["macro_f1"],
        })
    df_tau_p = pd.DataFrame(tau_p_rows)
    df_tau_p.to_csv(out_dir / "a6_sweep_tau_prune.csv", index=False)
    sweeps["tau_prune"] = df_tau_p

    # 4. Sweep score weights (alpha, beta, delta, eta)
    weight_settings = [
        {"name": "Balanced (Default)", "alpha": 0.35, "beta": 0.35, "delta": 0.20, "eta": 0.10},
        {"name": "Geometry-Heavy", "alpha": 0.55, "beta": 0.15, "delta": 0.20, "eta": 0.10},
        {"name": "Graph-Heavy", "alpha": 0.15, "beta": 0.55, "delta": 0.20, "eta": 0.10},
        {"name": "Consistency-Heavy", "alpha": 0.25, "beta": 0.25, "delta": 0.40, "eta": 0.10},
        {"name": "High-Risk-Penalty", "alpha": 0.30, "beta": 0.30, "delta": 0.15, "eta": 0.25},
    ]
    w_rows = []
    for ws in weight_settings:
        m_cfg = dict(cfg["merge"])
        m_cfg["alpha"] = ws["alpha"]
        m_cfg["beta"] = ws["beta"]
        m_cfg["delta"] = ws["delta"]
        m_cfg["eta"] = ws["eta"]
        m_cfg["theta"] = 0.3
        m_cfg["safeguards"] = cfg["safeguards"]
        merged_fmnn, _ = gshr_merge(seed_ctx.base_fmnn, graph_ctx, m_cfg)
        p_cfg = dict(cfg["prune"])
        p_cfg["tau_prune"] = 0.005
        p_cfg["safeguards"] = cfg["safeguards"]
        pruned_fmnn, _ = gshr_prune(merged_fmnn, graph_ctx, p_cfg, Z_train, y_train, train_ids, Z_val, y_val)
        test_preds = pruned_fmnn.predict(Z_test)
        tm = classification_metrics(y_test, test_preds, data.num_classes)
        w_rows.append({
            "setting": ws["name"],
            "alpha": ws["alpha"],
            "beta": ws["beta"],
            "delta": ws["delta"],
            "eta": ws["eta"],
            "num_boxes": pruned_fmnn.num_boxes(),
            "test_accuracy": tm["accuracy"],
            "test_macro_f1": tm["macro_f1"],
        })
    df_weights = pd.DataFrame(w_rows)
    df_weights.to_csv(out_dir / "a6_sweep_weights.csv", index=False)
    sweeps["weights"] = df_weights

    print("Completed sensitivity sweeps.")
    return sweeps


def run_a7_dataset_comparison(cfg: dict[str, Any], seeds: list[int], out_dir: Path) -> pd.DataFrame:
    """Ablation A7: Comparison across homophilous and heterophilous graph datasets."""
    print("\n--- Running Ablation A7: Cross-Dataset Comparison ---")
    datasets = ["Cora", "CiteSeer", "Chameleon"]
    rows: list[dict[str, Any]] = []

    for dname in datasets:
        try:
            d = load_dataset(dname, root="./data")
            h_ratio = homophily_ratio(d)
            print(f"Dataset: {dname} (h={h_ratio:.4f})")

            for seed in seeds:
                seed_ctx = prepare_seed_context(cfg, d, seed)
                # B1 (GNN Softmax)
                r_b1 = run_single_seed(cfg, d, seed, "B1", seed_ctx=seed_ctx)
                r_b1["homophily_ratio"] = h_ratio
                rows.append(r_b1)
                # B2 (Base FMNN)
                r_b2 = run_single_seed(cfg, d, seed, "B2", seed_ctx=seed_ctx)
                r_b2["homophily_ratio"] = h_ratio
                rows.append(r_b2)
                # B6 (Full GSHR)
                r_b6 = run_single_seed(cfg, d, seed, "B6", seed_ctx=seed_ctx)
                r_b6["homophily_ratio"] = h_ratio
                rows.append(r_b6)

        except Exception as e:
            print(f"Skipping dataset {dname} due to error: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "a7_cross_dataset.csv", index=False)
    print(f"Saved A7 dataset comparison to {out_dir / 'a7_cross_dataset.csv'}")
    return df


def run_a8_encoders(cfg: dict[str, Any], data: GraphData, seeds: list[int], out_dir: Path) -> pd.DataFrame:
    """Ablation A8: Encoder architecture comparison (GCN vs GraphSAGE)."""
    print("\n--- Running Ablation A8: GCN vs GraphSAGE ---")
    rows: list[dict[str, Any]] = []

    for enc_name in ["gcn", "sage"]:
        cfg_enc = copy.deepcopy(cfg)
        cfg_enc["gnn"]["model"] = enc_name

        for seed in seeds:
            seed_ctx = prepare_seed_context(cfg_enc, data, seed)
            # Evaluate B1, B2, B6
            for v in ["B1", "B2", "B6"]:
                r = run_single_seed(cfg_enc, data, seed, v, seed_ctx=seed_ctx)
                r["encoder"] = enc_name
                rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "a8_encoders.csv", index=False)
    print(f"Saved A8 encoder comparison to {out_dir / 'a8_encoders.csv'}")
    return df


def run_robustness(cfg: dict[str, Any], data: GraphData, seed: int, out_dir: Path) -> pd.DataFrame:
    """Robustness stress tests: edge dropout, feature masking, and reduced training labels."""
    print("\n--- Running Robustness Stress Tests ---")
    rows: list[dict[str, Any]] = []

    seed_ctx = prepare_seed_context(cfg, data, seed)
    train_idx = data.train_idx.cpu().numpy()
    val_idx = data.val_idx.cpu().numpy()
    test_idx = data.test_idx.cpu().numpy()

    Z_scaled = seed_ctx.Z_scaled
    y_test = data.y[data.test_idx].cpu().numpy()

    # Base B2 and B6 models
    res_b2 = run_single_seed(cfg, data, seed, "B2", seed_ctx=seed_ctx)
    res_b6 = run_single_seed(cfg, data, seed, "B6", seed_ctx=seed_ctx)

    base_fmnn = seed_ctx.base_fmnn
    # Refined fmnn
    m_cfg = dict(cfg["merge"])
    m_cfg["theta"] = 0.3
    m_cfg["safeguards"] = cfg["safeguards"]
    y_masked = np.full(data.num_nodes, -1, dtype=np.int64)
    y_masked[train_idx] = data.y[data.train_idx].cpu().numpy()
    train_mask_arr = np.zeros(data.num_nodes, dtype=bool)
    train_mask_arr[train_idx] = True
    graph_ctx = GraphContext(adj_csr=data.adj_csr, degree=data.degree, y=y_masked, train_mask=train_mask_arr)

    merged_fmnn, _ = gshr_merge(base_fmnn, graph_ctx, m_cfg)
    p_cfg = dict(cfg["prune"])
    p_cfg["safeguards"] = cfg["safeguards"]
    gshr_fmnn, _ = gshr_prune(
        merged_fmnn, graph_ctx, p_cfg,
        Z_scaled[train_idx], data.y[data.train_idx].cpu().numpy(), train_idx.tolist(),
        Z_scaled[val_idx], data.y[data.val_idx].cpu().numpy(),
    )

    # 1. Feature Masking at Test Time (10%, 30%)
    rng = np.random.RandomState(seed)
    for mask_p in [0.0, 0.10, 0.30]:
        Z_test_masked = Z_scaled[test_idx].copy()
        if mask_p > 0.0:
            mask_entries = rng.rand(*Z_test_masked.shape) < mask_p
            Z_test_masked[mask_entries] = 0.0

        p_b2 = base_fmnn.predict(Z_test_masked)
        m_b2 = classification_metrics(y_test, p_b2, data.num_classes)

        p_b6 = gshr_fmnn.predict(Z_test_masked)
        m_b6 = classification_metrics(y_test, p_b6, data.num_classes)

        rows.append({
            "test_type": "feature_masking",
            "perturbation": f"{int(mask_p*100)}%",
            "b2_test_acc": m_b2["accuracy"],
            "b2_test_f1": m_b2["macro_f1"],
            "b2_boxes": base_fmnn.num_boxes(),
            "b6_test_acc": m_b6["accuracy"],
            "b6_test_f1": m_b6["macro_f1"],
            "b6_boxes": gshr_fmnn.num_boxes(),
        })

    # 2. Reduced Training Labels (100%, 50%, 25%)
    y_train_full = data.y[data.train_idx].cpu().numpy()
    for keep_ratio in [0.50, 0.25]:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=keep_ratio, random_state=seed)
        sub_idx_rel, _ = next(sss.split(train_idx, y_train_full))
        sub_train_idx = train_idx[sub_idx_rel]

        Z_train_sub = Z_scaled[sub_train_idx]
        y_train_sub = data.y[sub_train_idx].cpu().numpy()
        sub_train_ids = sub_train_idx.tolist()

        fmnn_sub = FMNN(theta=0.3, gamma=4.0)
        fmnn_sub.fit(Z_train_sub, y_train_sub, sub_train_ids)

        y_sub_masked = np.full(data.num_nodes, -1, dtype=np.int64)
        y_sub_masked[sub_train_idx] = y_train_sub
        sub_mask_arr = np.zeros(data.num_nodes, dtype=bool)
        sub_mask_arr[sub_train_idx] = True
        sub_graph_ctx = GraphContext(adj_csr=data.adj_csr, degree=data.degree, y=y_sub_masked, train_mask=sub_mask_arr)

        m_sub, _ = gshr_merge(fmnn_sub, sub_graph_ctx, m_cfg)
        p_sub, _ = gshr_prune(
            m_sub, sub_graph_ctx, p_cfg,
            Z_train_sub, y_train_sub, sub_train_ids,
            Z_scaled[val_idx], data.y[data.val_idx].cpu().numpy(),
        )

        p_b2_sub = fmnn_sub.predict(Z_scaled[test_idx])
        m_b2_sub = classification_metrics(y_test, p_b2_sub, data.num_classes)

        p_b6_sub = p_sub.predict(Z_scaled[test_idx])
        m_b6_sub = classification_metrics(y_test, p_b6_sub, data.num_classes)

        rows.append({
            "test_type": "reduced_labels",
            "perturbation": f"{int(keep_ratio*100)}%",
            "b2_test_acc": m_b2_sub["accuracy"],
            "b2_test_f1": m_b2_sub["macro_f1"],
            "b2_boxes": fmnn_sub.num_boxes(),
            "b6_test_acc": m_b6_sub["accuracy"],
            "b6_test_f1": m_b6_sub["macro_f1"],
            "b6_boxes": p_sub.num_boxes(),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "robustness_stress_tests.csv", index=False)
    print(f"Saved robustness tests to {out_dir / 'robustness_stress_tests.csv'}")
    return df


def run_prediction_smoothing(cfg: dict[str, Any], data: GraphData, seed: int, out_dir: Path) -> pd.DataFrame:
    """Optional Prediction-Time Graph Smoothing Ablation (Section 9)."""
    print("\n--- Running Prediction-Time Graph Smoothing Ablation ---")
    seed_ctx = prepare_seed_context(cfg, data, seed)
    train_idx = data.train_idx.cpu().numpy()
    val_idx = data.val_idx.cpu().numpy()
    test_idx = data.test_idx.cpu().numpy()

    Z_scaled = seed_ctx.Z_scaled
    y_test = data.y[data.test_idx].cpu().numpy()

    # Fit base and refine B6
    y_masked = np.full(data.num_nodes, -1, dtype=np.int64)
    y_masked[train_idx] = data.y[data.train_idx].cpu().numpy()
    train_mask_arr = np.zeros(data.num_nodes, dtype=bool)
    train_mask_arr[train_idx] = True
    graph_ctx = GraphContext(adj_csr=data.adj_csr, degree=data.degree, y=y_masked, train_mask=train_mask_arr)

    m_cfg = dict(cfg["merge"])
    m_cfg["theta"] = 0.3
    m_cfg["safeguards"] = cfg["safeguards"]
    merged_fmnn, _ = gshr_merge(seed_ctx.base_fmnn, graph_ctx, m_cfg)

    p_cfg = dict(cfg["prune"])
    p_cfg["safeguards"] = cfg["safeguards"]
    gshr_fmnn, _ = gshr_prune(
        merged_fmnn, graph_ctx, p_cfg,
        Z_scaled[train_idx], data.y[data.train_idx].cpu().numpy(), train_idx.tolist(),
        Z_scaled[val_idx], data.y[data.val_idx].cpu().numpy(),
    )

    # Compute raw class scores for ALL nodes in the graph
    all_scores = gshr_fmnn.class_scores(Z_scaled)  # (N, num_classes)

    # Normalized symmetric adjacency: D^{-1} A for neighbor average
    adj = data.adj_csr.astype(np.float32)
    deg = data.degree.astype(np.float32)
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    D_inv = sp.diags(deg_inv)
    P = D_inv @ adj  # row-stochastic neighbor transition

    neighbor_scores = P @ all_scores

    rows: list[dict[str, Any]] = []

    # Test with rho in [0.0, 0.1, 0.3, 0.5]
    for rho in [0.0, 0.1, 0.3, 0.5]:
        smoothed_scores = (1.0 - rho) * all_scores + rho * neighbor_scores
        test_preds = np.argmax(smoothed_scores[test_idx], axis=1)
        tm = classification_metrics(y_test, test_preds, data.num_classes)
        rows.append({
            "rho": rho,
            "test_accuracy": tm["accuracy"],
            "test_macro_f1": tm["macro_f1"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "smoothing_ablation.csv", index=False)
    print(f"Saved smoothing ablation to {out_dir / 'smoothing_ablation.csv'}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete GSHR ablations suite.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", type=str, default="results/ablations")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = load_dataset(args.dataset, root="./data")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STARTING GSHR ABLATION SUITE (5 Seeds)")
    print("=" * 70)

    # 1. Core ablations A0 to A4
    run_a0_a4_ablations(cfg, data, args.seeds, out_dir)

    # 2. A5: Safeguard relaxation
    run_a5_safeguard_ablation(cfg, data, args.seeds, out_dir)

    # 3. A6: Sensitivity sweeps
    run_a6_sensitivity(cfg, data, args.seeds[0], out_dir)

    # 4. A7: Homophilous vs heterophilous benchmarks
    run_a7_dataset_comparison(cfg, args.seeds[:3], out_dir)

    # 5. A8: GCN vs GraphSAGE
    run_a8_encoders(cfg, data, args.seeds[:3], out_dir)

    # 6. Robustness stress tests
    run_robustness(cfg, data, args.seeds[0], out_dir)

    # 7. Optional prediction smoothing
    run_prediction_smoothing(cfg, data, args.seeds[0], out_dir)

    print("\nALL ABLATION SUITES COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
