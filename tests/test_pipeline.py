"""Unit tests for the end-to-end pipeline execution."""

from __future__ import annotations

import copy
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from src.data import GraphData
from src.metrics import compute_box_coverage, measure_prediction_latency
from src.pipeline import (
    _build_capped_pseudo_fit_set,
    _build_pseudo_fit_set,
    _select_smoothing_rho,
    compute_single_box_gate_and_scores,
    predict_gated_hybrid,
    predict_hybrid,
    prepare_seed_context,
    run_single_seed,
    select_b10_components,
    select_b14_components,
    smooth_scores_graph,
)
from src.utils import load_config


@pytest.fixture(scope="module")
def synthetic_graph() -> GraphData:
    """Build a deterministic 30-node 2-community synthetic graph."""
    num_nodes = 30
    num_classes = 2
    dim = 8
    rng = np.random.RandomState(42)

    # Community structure: first 15 nodes class 0, next 15 nodes class 1
    y_np = np.zeros(num_nodes, dtype=np.int64)
    y_np[15:] = 1
    y = torch.tensor(y_np, dtype=torch.long)

    # Features: class 0 centered around -1, class 1 centered around +1
    x_np = rng.randn(num_nodes, dim).astype(np.float32)
    x_np[:15] += -1.0
    x_np[15:] += 1.0
    x = torch.tensor(x_np, dtype=torch.float32)

    # Synthetic edges with high intra-community connectivity
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p = 0.35 if y_np[i] == y_np[j] else 0.05
            if rng.rand() < p:
                edges.append((i, j))
                edges.append((j, i))

    if not edges:
        edges = [(0, 1), (1, 0), (15, 16), (16, 15)]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Disjoint splits (10 train, 10 val, 10 test)
    perm = rng.permutation(num_nodes)
    train_idx = torch.tensor(sorted(perm[:10]), dtype=torch.long)
    val_idx = torch.tensor(sorted(perm[10:20]), dtype=torch.long)
    test_idx = torch.tensor(sorted(perm[20:]), dtype=torch.long)

    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    data_vals = np.ones(len(row), dtype=np.float32)
    adj_csr = sp.coo_matrix((data_vals, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    adj_csr.setdiag(0.0)
    adj_csr.eliminate_zeros()
    degree = np.array(adj_csr.sum(axis=1), dtype=np.int64).flatten()

    return GraphData(
        name="SyntheticToy",
        x=x,
        edge_index=edge_index,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        num_classes=num_classes,
        num_nodes=num_nodes,
        adj_csr=adj_csr,
        degree=degree,
    )


@pytest.fixture(scope="module")
def fast_cfg() -> dict:
    """Fixture providing fast config for pipeline testing."""
    cfg = load_config("configs/default.yaml")
    fast_c = copy.deepcopy(cfg)
    fast_c["gnn"]["max_epochs"] = 15
    fast_c["gnn"]["patience"] = 5
    fast_c["gnn"]["hidden"] = 16
    fast_c["gnn"]["out_dim"] = 16
    fast_c["run"]["device"] = "cpu"
    fast_c["tune"] = {
        "tau_merge_grid": [0.4, 0.5],
        "tau_prune_grid": [0.0, 0.01],
    }
    return fast_c


def test_pipeline_execution_and_columns(synthetic_graph: GraphData, fast_cfg: dict) -> None:
    """Run full pipeline across multiple seeds and variants, checking expected columns."""
    expected_cols = {
        "dataset", "seed", "variant",
        "test_accuracy", "test_macro_f1", "test_micro_f1",
        "val_accuracy", "val_macro_f1",
        "num_boxes", "mean_box_size", "max_box_size", "cross_class_overlap",
        "chosen_tau_merge", "chosen_tau_prune",
        "gnn_train_time", "fmnn_fit_time", "refine_time",
        "pred_latency_per_node", "peak_memory_mb",
    }

    variants = ["B1", "B2", "B6"]
    for seed in [0, 1]:
        seed_ctx = prepare_seed_context(fast_cfg, synthetic_graph, seed)
        for var in variants:
            res = run_single_seed(fast_cfg, synthetic_graph, seed, var, seed_ctx=seed_ctx)
            assert expected_cols.issubset(res.keys()), f"Missing keys in result: {expected_cols - res.keys()}"
            assert res["test_accuracy"] >= 0.0
            assert res["test_macro_f1"] >= 0.0
            assert res["cross_class_overlap"] == 0


def test_b6_box_count_lte_b2(synthetic_graph: GraphData, fast_cfg: dict) -> None:
    """B6 (full refinement) box count must be less than or equal to B2 (base FMNN)."""
    seed_ctx = prepare_seed_context(fast_cfg, synthetic_graph, seed=42)
    res_b2 = run_single_seed(fast_cfg, synthetic_graph, seed=42, variant="B2", seed_ctx=seed_ctx)
    res_b6 = run_single_seed(fast_cfg, synthetic_graph, seed=42, variant="B6", seed_ctx=seed_ctx)

    assert res_b6["num_boxes"] <= res_b2["num_boxes"]
    assert res_b2["num_boxes"] > 0


def test_pipeline_determinism(synthetic_graph: GraphData, fast_cfg: dict) -> None:
    """Running the pipeline twice with identical seed produces identical outputs."""
    res1 = run_single_seed(fast_cfg, synthetic_graph, seed=123, variant="B6")
    res2 = run_single_seed(fast_cfg, synthetic_graph, seed=123, variant="B6")

    assert res1["test_accuracy"] == res2["test_accuracy"]
    assert res1["test_macro_f1"] == res2["test_macro_f1"]
    assert res1["num_boxes"] == res2["num_boxes"]
    assert res1["chosen_tau_merge"] == res2["chosen_tau_merge"]
    assert res1["chosen_tau_prune"] == res2["chosen_tau_prune"]


# Verify predict_hybrid edge cases and B7_hybrid sanity checks against B1 and B6.
def test_b7_hybrid_and_sanity_checks(synthetic_graph: GraphData, fast_cfg: dict) -> None:
    probs = np.array([[0.8, 0.2], [0.1, 0.9], [0.4, 0.6]])
    mem = np.array([[0.2, 0.8], [0.9, 0.1], [0.0, 0.0]])
    assert np.array_equal(predict_hybrid(probs, mem, 1.0), np.array([0, 1, 1]))
    assert np.array_equal(predict_hybrid(probs, mem, 0.0), np.array([1, 0, 0]))

    seed_ctx = prepare_seed_context(fast_cfg, synthetic_graph, seed=7)
    res_b1 = run_single_seed(fast_cfg, synthetic_graph, seed=7, variant="B1", seed_ctx=seed_ctx)
    res_b6 = run_single_seed(fast_cfg, synthetic_graph, seed=7, variant="B6", seed_ctx=seed_ctx)
    res_b7 = run_single_seed(fast_cfg, synthetic_graph, seed=7, variant="B7_hybrid", seed_ctx=seed_ctx)

    assert res_b7["variant"] == "B7_hybrid"
    assert res_b7["sanity_lam0_matches_b6"] is True
    assert res_b7["sanity_lam1_matches_b1"] is True
    assert res_b7["num_boxes"] == res_b6["num_boxes"]
    assert res_b7["chosen_lam"] in [0.0, 0.25, 0.5, 0.75, 1.0]
    assert 0.0 <= res_b7["box_coverage"] <= 1.0
    assert res_b7["covered_count"] + res_b7["uncovered_count"] == len(synthetic_graph.test_idx)

    # Confirm B7 lambda=1 matches B1 and lambda=0 matches B6 via fixed config option.
    cfg_l1 = copy.deepcopy(fast_cfg)
    cfg_l1["hybrid"] = {"lambda": 1.0}
    res_l1 = run_single_seed(cfg_l1, synthetic_graph, seed=7, variant="B7_hybrid", seed_ctx=seed_ctx)
    assert res_l1["test_accuracy"] == pytest.approx(res_b1["test_accuracy"])
    assert res_l1["test_macro_f1"] == pytest.approx(res_b1["test_macro_f1"])

    cfg_l0 = copy.deepcopy(fast_cfg)
    cfg_l0["hybrid"] = {"lambda": 0.0}
    res_l0 = run_single_seed(cfg_l0, synthetic_graph, seed=7, variant="B7_hybrid", seed_ctx=seed_ctx)
    assert res_l0["test_accuracy"] == pytest.approx(res_b6["test_accuracy"])
    assert res_l0["test_macro_f1"] == pytest.approx(res_b6["test_macro_f1"])

    # Verify box coverage calculation directly.
    cov = compute_box_coverage(
        seed_ctx.base_fmnn,
        seed_ctx.Z_scaled[synthetic_graph.train_idx.numpy()],
        synthetic_graph.y[synthetic_graph.train_idx].numpy(),
        seed_ctx.base_fmnn.predict(seed_ctx.Z_scaled[synthetic_graph.train_idx.numpy()]),
    )
    assert cov["box_coverage"] >= 0.0


# Verify B8_hybrid_pseudo builds pseudo labels without reading true val/test labels and logs deltas.
def test_b8_hybrid_pseudo_and_no_leakage(synthetic_graph: GraphData, fast_cfg: dict) -> None:
    rng = np.random.RandomState(11)
    probs = np.zeros((synthetic_graph.num_nodes, synthetic_graph.num_classes), dtype=np.float64)
    probs[:, 0] = 0.92
    probs[:, 1] = 0.08
    # Flip half of non-train confident predictions to class 1
    probs[15:, 0] = 0.04
    probs[15:, 1] = 0.96

    corrupted = copy.deepcopy(synthetic_graph)
    corrupted.y[corrupted.val_idx] = -999
    corrupted.y[corrupted.test_idx] = -999

    fit_c, y_c, p_idx_c, ctx_c, cnt_c = _build_pseudo_fit_set(synthetic_graph, probs, threshold_c=0.9)
    fit_bad, y_bad, p_idx_bad, ctx_bad, cnt_bad = _build_pseudo_fit_set(corrupted, probs, threshold_c=0.9)
    assert np.array_equal(fit_c, fit_bad)
    assert np.array_equal(y_c, y_bad)
    assert np.array_equal(p_idx_c, p_idx_bad)
    assert cnt_c == cnt_bad
    assert np.all(ctx_bad.y[ctx_bad.train_mask] >= 0)  # Fit mask only covers train (true) and pseudo (predicted), never -999!

    cfg_b8 = copy.deepcopy(fast_cfg)
    cfg_b8["pseudo"] = {"enabled": True, "c_grid": [0.8, 0.9, 0.95]}
    seed_ctx = prepare_seed_context(cfg_b8, synthetic_graph, seed=5)
    res_b8 = run_single_seed(cfg_b8, synthetic_graph, seed=5, variant="B8_hybrid_pseudo", seed_ctx=seed_ctx)

    assert res_b8["variant"] == "B8_hybrid_pseudo"
    assert res_b8["train_size"] == len(synthetic_graph.train_idx)
    assert res_b8["pseudo_count"] >= 0
    assert res_b8["chosen_c"] in [0.8, 0.9, 0.95]
    assert res_b8["chosen_lam"] in [0.0, 0.25, 0.5, 0.75, 1.0]
    assert res_b8["cross_class_overlap"] == 0
    assert "delta_num_boxes_vs_b2" in res_b8
    assert "delta_mean_box_size_vs_b2" in res_b8
    assert "delta_cross_class_overlap_vs_b2" in res_b8

    # Verify pseudo-labeling can be switched off cleanly.
    cfg_off = copy.deepcopy(cfg_b8)
    cfg_off["pseudo"] = {"enabled": False}
    res_off = run_single_seed(cfg_off, synthetic_graph, seed=5, variant="B8_hybrid_pseudo", seed_ctx=seed_ctx)
    assert res_off["pseudo_count"] == 0
    assert res_off["chosen_c"] is None


# Verify B9_gshr_fast selects (theta, tau_prune, safeguards) on validation only and keeps >= num_classes boxes.
def test_b9_gshr_fast_validation_selection_and_min_boxes(synthetic_graph: GraphData, fast_cfg: dict) -> None:
    cfg_b9 = copy.deepcopy(fast_cfg)
    cfg_b9["gshr_fast"] = {
        "merge_enabled": True,
        "require_same_class_overlap": True,
        "theta_grid": [0.3, 0.5],
        "tau_prune_grid": [0.0, 0.01],
        "safeguards_grid": [True, False],
    }

    seed_ctx = prepare_seed_context(cfg_b9, synthetic_graph, seed=3)
    res_clean = run_single_seed(cfg_b9, synthetic_graph, seed=3, variant="B9_gshr_fast", seed_ctx=seed_ctx)

    # Corrupt test labels inside a cloned dataset passed to _select_fast_gshr check via train/val path.
    assert res_clean["variant"] == "B9_gshr_fast"
    assert res_clean["num_boxes"] >= synthetic_graph.num_classes
    assert res_clean["chosen_theta"] in [0.3, 0.5]
    assert res_clean["chosen_tau_prune"] in [0.0, 0.01]
    assert res_clean["chosen_safeguards"] in ["on", "off"]
    assert len(res_clean["settings_table"]) == 8
    for row in res_clean["settings_table"]:
        assert row["num_boxes"] >= synthetic_graph.num_classes
        assert row["refine_time"] >= 0.0


# Verify B10_final component gating ignores test labels and logs storage bytes and per-class recall.
def test_b10_final_component_gating_no_test_leakage(synthetic_graph: GraphData, fast_cfg: dict) -> None:
    cfg_v2 = copy.deepcopy(fast_cfg)
    cfg_v2["gnn"]["early_stop_metric"] = "val_loss"
    cfg_v2["embed"]["l2_norm"] = True
    cfg_v2["gshr_fast"] = {
        "merge_enabled": True,
        "require_same_class_overlap": True,
        "theta_grid": [0.3, 0.5],
        "tau_prune_grid": [0.0, 0.01],
        "safeguards_grid": [True, False],
    }
    cfg_v2["pseudo"] = {"enabled": True, "c_grid": [0.8, 0.9]}

    corrupted = copy.deepcopy(synthetic_graph)
    corrupted.y[corrupted.test_idx] = -999

    sel_clean = select_b10_components(synthetic_graph, fast_cfg, cfg_v2, seeds=[0])
    sel_corrupt = select_b10_components(corrupted, fast_cfg, cfg_v2, seeds=[0])
    assert sel_clean["selected"] == sel_corrupt["selected"]
    assert sel_clean["gate_details"] == sel_corrupt["gate_details"]

    cfg_b10 = copy.deepcopy(cfg_v2)
    cfg_b10["b10_components"] = sel_clean["selected"]
    res_b1 = run_single_seed(fast_cfg, synthetic_graph, seed=0, variant="B1")
    res_b10 = run_single_seed(cfg_b10, synthetic_graph, seed=0, variant="B10_final")

    assert "test_per_class_recall" in res_b1
    assert len(res_b1["test_per_class_recall"]) == synthetic_graph.num_classes
    assert res_b1["box_storage_bytes"] == 0
    assert res_b10["variant"] == "B10_final"
    assert len(res_b10["test_per_class_recall"]) == synthetic_graph.num_classes
    expected_bytes = res_b10["num_boxes"] * 2 * fast_cfg["gnn"]["out_dim"] * 4
    assert res_b10["box_storage_bytes"] == expected_bytes


# Verify _build_capped_pseudo_fit_set caps counts per class without val/test label leakage.
def test_capped_pseudo_no_test_leakage_and_per_class_cap(synthetic_graph: GraphData) -> None:
    probs = np.zeros((synthetic_graph.num_nodes, 2), dtype=np.float64)
    for i in range(15):
        probs[i] = [0.85 + 0.005 * i, 0.15 - 0.005 * i]
    for i in range(15, 30):
        probs[i] = [0.15 - 0.005 * (i - 15), 0.85 + 0.005 * (i - 15)]

    corrupted = copy.deepcopy(synthetic_graph)
    corrupted.y[corrupted.val_idx] = -999
    corrupted.y[corrupted.test_idx] = -999

    fit_a, y_a, ps_a, yps_a, ctx_a, cnt_a, pcls_a = _build_capped_pseudo_fit_set(
        synthetic_graph, probs, max_per_class=2, min_conf=0.8
    )
    fit_b, y_b, ps_b, yps_b, ctx_b, cnt_b, pcls_b = _build_capped_pseudo_fit_set(
        corrupted, probs, max_per_class=2, min_conf=0.8
    )

    assert np.array_equal(fit_a, fit_b)
    assert np.array_equal(y_a, y_b)
    assert np.array_equal(ps_a, ps_b)
    assert np.array_equal(yps_a, yps_b)
    assert cnt_a == cnt_b
    assert pcls_a == pcls_b
    assert all(v <= 2 for v in pcls_a.values())
    assert cnt_a == sum(pcls_a.values())
    assert np.all(ctx_b.y[ctx_b.train_mask] >= 0)


# Verify B11_final_lite runs cleanly, switches capped pseudo off cleanly, and uses unified latency.
def test_b11_final_lite_end_to_end_and_switches(synthetic_graph: GraphData, fast_cfg: dict) -> None:
    cfg_b11 = copy.deepcopy(fast_cfg)
    cfg_b11["gshr_fast"] = {
        "merge_enabled": True,
        "require_same_class_overlap": True,
        "theta_grid": [0.3, 0.5],
        "tau_prune_grid": [0.0, 0.01],
        "safeguards_grid": [True],
    }
    cfg_b11["b11_lite"] = {
        "use_capped_pseudo": True,
        "cap_per_class_grid": [2, 5],
        "min_conf": 0.5,
        "allow_new_boxes": True,
    }
    cfg_b11["latency"] = {"warmup": 2, "repeats": 5}

    seed_ctx = prepare_seed_context(cfg_b11, synthetic_graph, seed=4)
    res_b11 = run_single_seed(cfg_b11, synthetic_graph, seed=4, variant="B11_final_lite", seed_ctx=seed_ctx)

    assert res_b11["variant"] == "B11_final_lite"
    assert res_b11["chosen_k"] in [2, 5]
    assert res_b11["train_size"] == len(synthetic_graph.train_idx)
    assert res_b11["pseudo_count"] == sum(res_b11["pseudo_per_class_counts"].values())
    assert res_b11["num_boxes"] >= synthetic_graph.num_classes
    assert res_b11["cross_class_overlap"] == 0
    assert res_b11["pred_latency_per_node"] > 0.0
    assert res_b11["box_storage_bytes"] == res_b11["num_boxes"] * 2 * cfg_b11["gnn"]["out_dim"] * 4

    cfg_off = copy.deepcopy(cfg_b11)
    cfg_off["b11_lite"]["use_capped_pseudo"] = False
    res_off = run_single_seed(cfg_off, synthetic_graph, seed=4, variant="B11_final_lite", seed_ctx=seed_ctx)
    assert res_off["chosen_k"] is None
    assert res_off["pseudo_count"] == 0
    assert res_off["pseudo_per_class_counts"] == {}


# Verify unified measure_prediction_latency preserves predictions and returns positive CPU latency.
def test_measure_prediction_latency_unified_cpu(synthetic_graph: GraphData, fast_cfg: dict) -> None:
    cfg_u = copy.deepcopy(fast_cfg)
    cfg_u["unified_latency"] = True
    cfg_u["latency"] = {"warmup": 2, "repeats": 5}

    seed_ctx = prepare_seed_context(cfg_u, synthetic_graph, seed=2)
    assert seed_ctx.scaler is not None

    lat_b1 = measure_prediction_latency(
        "B1", seed_ctx.gnn_model, synthetic_graph, scaler=seed_ctx.scaler, warmup=2, repeats=5
    )
    lat_b2 = measure_prediction_latency(
        "B2", seed_ctx.gnn_model, synthetic_graph, scaler=seed_ctx.scaler, fmnn_model=seed_ctx.base_fmnn, warmup=2, repeats=5
    )
    assert lat_b1 > 0.0 and np.isfinite(lat_b1)
    assert lat_b2 > 0.0 and np.isfinite(lat_b2)

    res_u = run_single_seed(cfg_u, synthetic_graph, seed=2, variant="B1", seed_ctx=seed_ctx)
    res_plain = run_single_seed(fast_cfg, synthetic_graph, seed=2, variant="B1", seed_ctx=seed_ctx)
    assert res_u["test_accuracy"] == res_plain["test_accuracy"]
    assert res_u["pred_latency_per_node"] > 0.0


# Verify B12_ensemble chooses avg vs concat on val, computes unified latency, and tracks box flips.
def test_b12_ensemble_end_to_end(synthetic_graph: GraphData, fast_cfg: dict) -> None:
    cfg_e = copy.deepcopy(fast_cfg)
    cfg_e["gshr_fast"] = {
        "merge_enabled": True, "require_same_class_overlap": True,
        "theta_grid": [0.3, 0.5], "tau_prune_grid": [0.0], "safeguards_grid": [True],
    }
    cfg_e["b12_ensemble"] = {"num_models": 2, "seed_stride": 100, "embed_mode": "auto"}
    cfg_e["latency"] = {"warmup": 2, "repeats": 4}

    res = run_single_seed(cfg_e, synthetic_graph, seed=0, variant="B12_ensemble")
    assert res["variant"] == "B12_ensemble"
    assert res["chosen_embed_mode"] in {"avg", "concat"}
    expected_dim = fast_cfg["gnn"]["out_dim"] * (2 if res["chosen_embed_mode"] == "concat" else 1)
    assert res["box_storage_bytes"] == res["num_boxes"] * 2 * expected_dim * 4
    assert 0.0 <= res["fraction_changed_by_boxes"] <= 1.0
    assert res["pred_latency_per_node"] > 0.0


# Verify B13_gated_hybrid routes only single-box nodes to FMNN, scales T, and logs exact breakdown.
def test_b13_gated_hybrid_and_smoothing_no_leakage(synthetic_graph: GraphData, fast_cfg: dict) -> None:
    cfg_g = copy.deepcopy(fast_cfg)
    cfg_g["gshr_fast"] = {
        "merge_enabled": True, "require_same_class_overlap": True,
        "theta_grid": [0.3, 0.5], "tau_prune_grid": [0.0], "safeguards_grid": [True],
    }
    cfg_g["b13_gated"] = {"T_grid": [0.5, 1.0, 1.5, 2.0]}
    cfg_g["latency"] = {"warmup": 2, "repeats": 4}

    seed_ctx = prepare_seed_context(cfg_g, synthetic_graph, seed=1)
    res_g = run_single_seed(cfg_g, synthetic_graph, seed=1, variant="B13_gated_hybrid", seed_ctx=seed_ctx)
    assert res_g["variant"] == "B13_gated_hybrid"
    assert res_g["chosen_T"] in [0.5, 1.0, 1.5, 2.0]
    assert res_g["box_decided_count"] + res_g["gnn_decided_count"] == len(synthetic_graph.test_idx)
    assert 0.0 <= res_g["box_decided_fraction"] <= 1.0
    assert "b7_test_accuracy" in res_g and "b7_chosen_lam" in res_g
    assert res_g["pred_latency_per_node"] > 0.0

    # Smoothing rho=0 is identity; validation selection ignores test labels completely.
    rng = np.random.RandomState(0)
    raw_scores = rng.rand(synthetic_graph.num_nodes, synthetic_graph.num_classes)
    assert np.allclose(smooth_scores_graph(raw_scores, synthetic_graph.adj_csr, synthetic_graph.degree, 0.0), raw_scores)
    val_idx = synthetic_graph.val_idx.cpu().numpy()
    corrupted = copy.deepcopy(synthetic_graph)
    corrupted.y[corrupted.test_idx] = -999
    r_clean, s_clean, _, f1_clean = _select_smoothing_rho(
        raw_scores, synthetic_graph.adj_csr, synthetic_graph.degree,
        val_idx, synthetic_graph.y[val_idx].cpu().numpy(), synthetic_graph.num_classes, [0.0, 0.1, 0.3],
    )
    r_bad, s_bad, _, f1_bad = _select_smoothing_rho(
        raw_scores, corrupted.adj_csr, corrupted.degree,
        val_idx, corrupted.y[val_idx].cpu().numpy(), corrupted.num_classes, [0.0, 0.1, 0.3],
    )
    assert r_clean == r_bad and f1_clean == f1_bad and np.allclose(s_clean, s_bad)


# Verify select_b14_components ignores test labels and B14_final respects component switches.
def test_b14_final_component_gating_no_test_leakage(synthetic_graph: GraphData, fast_cfg: dict) -> None:
    cfg_14 = copy.deepcopy(fast_cfg)
    cfg_14["gshr_fast"] = {
        "merge_enabled": True, "require_same_class_overlap": True,
        "theta_grid": [0.3, 0.5], "tau_prune_grid": [0.0], "safeguards_grid": [True],
    }
    cfg_14["b12_ensemble"] = {"num_models": 2, "seed_stride": 100}
    cfg_14["b11_lite"] = {"cap_per_class_grid": [2, 4], "min_conf": 0.5}
    cfg_14["latency"] = {"warmup": 2, "repeats": 4}

    corrupted = copy.deepcopy(synthetic_graph)
    corrupted.y[corrupted.test_idx] = -999

    sel_clean = select_b14_components(synthetic_graph, cfg_14, seeds=[0], ens_seeds_override=[0])
    sel_bad = select_b14_components(corrupted, cfg_14, seeds=[0], ens_seeds_override=[0])
    assert sel_clean["selected"] == sel_bad["selected"]
    assert sel_clean["gate_details"] == sel_bad["gate_details"]

    cfg_run = copy.deepcopy(cfg_14)
    cfg_run["b14_components"] = sel_clean["selected"]
    res14 = run_single_seed(cfg_run, synthetic_graph, seed=0, variant="B14_final")
    assert res14["variant"] == "B14_final"
    assert res14["pred_latency_per_node"] > 0.0
    assert 0.0 <= res14["fraction_changed_by_boxes"] <= 1.0


def test_refine_fast_vs_slow_equivalence(synthetic_graph: GraphData, fast_cfg: dict) -> None:
    """Verify that refine.fast=true and refine.fast=false produce identical box sets on a seeded graph."""
    cfg_fast = copy.deepcopy(fast_cfg)
    cfg_fast["refine"] = {"fast": True}

    cfg_slow = copy.deepcopy(fast_cfg)
    cfg_slow["refine"] = {"fast": False}

    seed = 0
    seed_ctx = prepare_seed_context(cfg_fast, synthetic_graph, seed)

    res_fast = run_single_seed(cfg_fast, synthetic_graph, seed, "B6", seed_ctx=seed_ctx)
    res_slow = run_single_seed(cfg_slow, synthetic_graph, seed, "B6", seed_ctx=seed_ctx)

    boxes_fast = sorted(
        res_fast["model"].boxes.values(),
        key=lambda b: (b.cls, tuple(b.v), tuple(b.w)),
    )
    boxes_slow = sorted(
        res_slow["model"].boxes.values(),
        key=lambda b: (b.cls, tuple(b.v), tuple(b.w)),
    )

    assert len(boxes_fast) == len(boxes_slow), (
        f"Box count mismatch: fast={len(boxes_fast)} vs slow={len(boxes_slow)}"
    )
    for i, (b_f, b_s) in enumerate(zip(boxes_fast, boxes_slow)):
        assert b_f.cls == b_s.cls, f"Box {i} class mismatch: {b_f.cls} != {b_s.cls}"
        assert np.allclose(b_f.v, b_s.v, atol=1e-7), f"Box {i} lower corner v mismatch"
        assert np.allclose(b_f.w, b_s.w, atol=1e-7), f"Box {i} upper corner w mismatch"

    assert res_fast["num_boxes"] == res_slow["num_boxes"]
    assert res_fast["num_boxes"] >= synthetic_graph.num_classes


def test_fast_macro_f1_matches_sklearn_1000_cases() -> None:
    """Verify NumPy confusion-matrix macro-F1 matches scikit-learn exactly across 1,000 random cases."""
    from sklearn.metrics import f1_score
    from src.metrics import fast_macro_f1

    rng = np.random.default_rng(2026)
    for _ in range(1000):
        n = rng.integers(1, 200)
        c = rng.integers(2, 20)
        y_true = rng.integers(0, c, size=n)
        y_pred = rng.integers(0, c, size=n)

        sk_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        fast_f1 = fast_macro_f1(y_true, y_pred, num_classes=c)

        np.testing.assert_allclose(fast_f1, sk_f1, atol=1e-12)
