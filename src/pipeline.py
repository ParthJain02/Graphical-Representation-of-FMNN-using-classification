"""End-to-end pipeline orchestration, hyperparameter selection, and leakage guards."""

from __future__ import annotations

import copy
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import KNeighborsClassifier

from src.data import GraphData
from src.embeddings import (
    EmbeddingScaler,
    extract_embeddings,
    extract_logits,
    extract_softmax_probs,
    fit_scaler_and_transform,
    fit_transform_all,
    l2_normalize_embeddings,
)
from src.fmnn import FMNN
from src.gnn import GNNClassifier, train_gnn
from src.merge import GraphContext, fast_gshr_merge, geometry_only_merge, gshr_merge
from src.metrics import classification_metrics, compactness_metrics, compute_box_coverage, measure_prediction_latency
from src.prune import geometry_only_prune, gshr_prune
from src.utils import get_device, set_seed


# Select hybrid mixing weight lambda strictly on validation macro-F1 (ties go to smaller lambda).
def _select_hybrid_lambda(
    probs_val: np.ndarray,
    mem_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    lam_grid: list[float],
    fixed_lam: float | None = None,
) -> tuple[float, float, float]:
    if fixed_lam is not None:
        chosen_lam = float(fixed_lam)
        vp = predict_hybrid(probs_val, mem_val, chosen_lam)
        vm = classification_metrics(y_val, vp, num_classes)
        return chosen_lam, float(vm["accuracy"]), float(vm["macro_f1"])

    lam_trials: list[dict[str, Any]] = []
    for lam_cand in lam_grid:
        vp = predict_hybrid(probs_val, mem_val, lam_cand)
        vm = classification_metrics(y_val, vp, num_classes)
        lam_trials.append({
            "lam": float(lam_cand),
            "val_accuracy": float(vm["accuracy"]),
            "val_macro_f1": float(vm["macro_f1"]),
        })

    best_trial = min(
        lam_trials,
        key=lambda item: (-item["val_macro_f1"], -item["val_accuracy"], item["lam"]),
    )
    return float(best_trial["lam"]), float(best_trial["val_accuracy"]), float(best_trial["val_macro_f1"])


# Build train + confident GNN pseudo-label fitting set without ever reading val or test true labels.
def _build_pseudo_fit_set(
    data: GraphData,
    probs_all: np.ndarray,
    threshold_c: float,
    exclude_val: bool = False,
    exclude_test: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, GraphContext, int]:
    train_idx = data.train_idx.cpu().numpy()
    val_idx = data.val_idx.cpu().numpy()
    test_idx = data.test_idx.cpu().numpy()
    y_train = data.y[data.train_idx].cpu().numpy()

    cand_mask = np.ones(data.num_nodes, dtype=bool)
    cand_mask[train_idx] = False
    if exclude_val:
        cand_mask[val_idx] = False
    if exclude_test:
        cand_mask[test_idx] = False

    max_probs = np.max(probs_all, axis=1)
    gnn_preds = np.argmax(probs_all, axis=1)
    pseudo_idx = np.nonzero(cand_mask & (max_probs > float(threshold_c)))[0]
    y_pseudo = gnn_preds[pseudo_idx].astype(np.int64)

    fit_idx = np.concatenate([train_idx, pseudo_idx]).astype(np.int64)
    y_fit = np.concatenate([y_train, y_pseudo]).astype(np.int64)

    # Mask all non-fit nodes to -1; pseudo nodes carry ONLY GNN predicted labels, never true labels.
    y_fit_masked = np.full(data.num_nodes, -1, dtype=np.int64)
    y_fit_masked[train_idx] = y_train
    y_fit_masked[pseudo_idx] = y_pseudo

    fit_mask_arr = np.zeros(data.num_nodes, dtype=bool)
    fit_mask_arr[fit_idx] = True

    pseudo_graph_ctx = GraphContext(
        adj_csr=data.adj_csr,
        degree=data.degree,
        y=y_fit_masked,
        train_mask=fit_mask_arr,
    )
    return fit_idx, y_fit, pseudo_idx, pseudo_graph_ctx, int(len(pseudo_idx))


# Select at most k highest-confidence non-train pseudo nodes per class without reading val/test labels.
def _build_capped_pseudo_fit_set(
    data: GraphData,
    probs_all: np.ndarray,
    max_per_class: int,
    min_conf: float = 0.8,
    exclude_val: bool = False,
    exclude_test: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, GraphContext, int, dict[int, int]]:
    train_idx = data.train_idx.cpu().numpy()
    val_idx = data.val_idx.cpu().numpy()
    test_idx = data.test_idx.cpu().numpy()
    y_train = data.y[data.train_idx].cpu().numpy()

    cand_mask = np.ones(data.num_nodes, dtype=bool)
    cand_mask[train_idx] = False
    if exclude_val:
        cand_mask[val_idx] = False
    if exclude_test:
        cand_mask[test_idx] = False

    max_probs = np.max(probs_all, axis=1)
    gnn_preds = np.argmax(probs_all, axis=1)

    selected_chunks: list[np.ndarray] = []
    per_class_counts: dict[int, int] = {}
    cap = max(0, int(max_per_class))
    for c in range(data.num_classes):
        cls_mask = cand_mask & (gnn_preds == c) & (max_probs >= float(min_conf))
        cls_nodes = np.nonzero(cls_mask)[0]
        if len(cls_nodes) > 0 and cap > 0:
            order = sorted(cls_nodes.tolist(), key=lambda nid: (-float(max_probs[nid]), int(nid)))
            chosen = np.asarray(order[:cap], dtype=np.int64)
        else:
            chosen = np.empty(0, dtype=np.int64)
        per_class_counts[int(c)] = int(len(chosen))
        if len(chosen) > 0:
            selected_chunks.append(chosen)

    if selected_chunks:
        pseudo_idx = np.concatenate(selected_chunks).astype(np.int64)
        # Process pseudo nodes globally in descending confidence order for deterministic box expansion.
        global_order = sorted(pseudo_idx.tolist(), key=lambda nid: (-float(max_probs[nid]), int(nid)))
        pseudo_idx = np.asarray(global_order, dtype=np.int64)
    else:
        pseudo_idx = np.empty(0, dtype=np.int64)

    y_pseudo = gnn_preds[pseudo_idx].astype(np.int64)
    fit_idx = np.concatenate([train_idx, pseudo_idx]).astype(np.int64)
    y_fit = np.concatenate([y_train, y_pseudo]).astype(np.int64)

    y_fit_masked = np.full(data.num_nodes, -1, dtype=np.int64)
    y_fit_masked[train_idx] = y_train
    y_fit_masked[pseudo_idx] = y_pseudo

    fit_mask_arr = np.zeros(data.num_nodes, dtype=bool)
    fit_mask_arr[fit_idx] = True

    pseudo_graph_ctx = GraphContext(
        adj_csr=data.adj_csr,
        degree=data.degree,
        y=y_fit_masked,
        train_mask=fit_mask_arr,
    )
    return fit_idx, y_fit, pseudo_idx, y_pseudo, pseudo_graph_ctx, int(len(pseudo_idx)), per_class_counts


# Combine GNN softmax probabilities and per-class box membership scores for hybrid prediction.
def predict_hybrid(probs: np.ndarray, membership: np.ndarray, lam: float) -> np.ndarray:
    mem = np.asarray(membership, dtype=np.float64)
    norm_mem = mem / (np.sum(mem, axis=1, keepdims=True) + 1e-9)
    score = float(lam) * np.asarray(probs, dtype=np.float64) + (1.0 - float(lam)) * norm_mem
    return np.argmax(score, axis=1)


# Compute temperature-scaled softmax probabilities in float64 without overflow.
def softmax_with_temperature(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    temp = max(1e-6, float(T))
    scaled = np.asarray(logits, dtype=np.float64) / temp
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exps = np.exp(scaled)
    return exps / np.sum(exps, axis=1, keepdims=True)


# Identify nodes strictly inside exact count==1 hyperbox and return normalized box memberships.
def compute_single_box_gate_and_scores(
    fmnn: FMNN,
    Z: np.ndarray,
    eps: float = 1e-12,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z_arr = np.asarray(Z, dtype=np.float64)
    n_pts = len(Z_arr)
    box_list = [fmnn.boxes[k] for k in sorted(fmnn.boxes.keys())]
    if not box_list or n_pts == 0:
        zeros = np.zeros((n_pts, fmnn.num_classes), dtype=np.float64)
        return np.zeros(n_pts, dtype=bool), zeros, zeros

    V = np.stack([b.v for b in box_list], axis=0)
    W = np.stack([b.w for b in box_list], axis=0)
    inside_counts = np.zeros(n_pts, dtype=np.int32)
    step = max(1, int(chunk_size))
    for start in range(0, n_pts, step):
        end = min(start + step, n_pts)
        zc = Z_arr[start:end, None, :]
        inside = np.all((zc >= V[None, :, :] - eps) & (zc <= W[None, :, :] + eps), axis=2)
        inside_counts[start:end] = np.sum(inside, axis=1)
    del V, W

    gate = inside_counts == 1
    mem = fmnn.class_scores(Z_arr)
    norm_mem = mem / (np.sum(mem, axis=1, keepdims=True) + 1e-9)
    return gate, norm_mem, mem


# Route prediction to FMNN membership when a node lies in exact count==1 box, else GNN probs.
def predict_gated_hybrid(
    probs_T: np.ndarray,
    gate: np.ndarray,
    norm_mem: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.where(gate[:, None], np.asarray(norm_mem, dtype=np.float64), np.asarray(probs_T, dtype=np.float64))
    return scores, np.argmax(scores, axis=1)


# Apply row-stochastic prediction-time score smoothing (1 - rho) * S + rho * P @ S.
def smooth_scores_graph(
    all_scores: np.ndarray,
    adj_csr: sp.csr_matrix,
    degree: np.ndarray,
    rho: float,
) -> np.ndarray:
    scores = np.asarray(all_scores, dtype=np.float64)
    r = float(rho)
    if r <= 0.0:
        return scores
    adj = adj_csr.astype(np.float64)
    deg = np.asarray(degree, dtype=np.float64)
    deg_inv = np.where(deg > 0.0, 1.0 / np.maximum(deg, 1.0), 0.0)
    P = sp.diags(deg_inv) @ adj
    return (1.0 - r) * scores + r * (P @ scores)


# Select prediction-time smoothing rho strictly on validation macro-F1 (ties favor smaller rho).
def _select_smoothing_rho(
    all_scores: np.ndarray,
    adj_csr: sp.csr_matrix,
    degree: np.ndarray,
    val_idx: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    rho_grid: list[float],
    fixed_rho: float | None = None,
) -> tuple[float, np.ndarray, float, float]:
    if fixed_rho is not None:
        r = float(fixed_rho)
        smoothed = smooth_scores_graph(all_scores, adj_csr, degree, r)
        vm = classification_metrics(y_val, np.argmax(smoothed[val_idx], axis=1), num_classes)
        return r, smoothed, float(vm["accuracy"]), float(vm["macro_f1"])

    adj = adj_csr.astype(np.float64)
    deg = np.asarray(degree, dtype=np.float64)
    deg_inv = np.where(deg > 0.0, 1.0 / np.maximum(deg, 1.0), 0.0)
    neighbor_scores = (sp.diags(deg_inv) @ adj) @ np.asarray(all_scores, dtype=np.float64)

    best_key: tuple | None = None
    best_payload: tuple[float, np.ndarray, float, float] | None = None
    for rho_cand in rho_grid:
        r = float(rho_cand)
        smoothed = np.asarray(all_scores, dtype=np.float64) if r <= 0.0 else (1.0 - r) * all_scores + r * neighbor_scores
        vm = classification_metrics(y_val, np.argmax(smoothed[val_idx], axis=1), num_classes)
        key = (-float(vm["macro_f1"]), -float(vm["accuracy"]), r)
        if best_key is None or key < best_key:
            best_key = key
            best_payload = (r, smoothed, float(vm["accuracy"]), float(vm["macro_f1"]))
    assert best_payload is not None
    return best_payload


# Select temperature T strictly on validation macro-F1 (ties break by validation cross-entropy).
def _select_gated_temperature(
    logits_val: np.ndarray,
    gate_val: np.ndarray,
    norm_mem_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    t_grid: list[float],
    fixed_T: float | None = None,
) -> tuple[float, float, float]:
    if fixed_T is not None:
        t_val = float(fixed_T)
        p_t = softmax_with_temperature(logits_val, T=t_val)
        _, vp = predict_gated_hybrid(p_t, gate_val, norm_mem_val)
        vm = classification_metrics(y_val, vp, num_classes)
        return t_val, float(vm["accuracy"]), float(vm["macro_f1"])

    y_val_arr = np.asarray(y_val, dtype=np.int64)
    row_idx = np.arange(len(y_val_arr))
    best_key: tuple | None = None
    best_res: tuple[float, float, float] | None = None
    for t_cand in t_grid:
        t_val = float(t_cand)
        p_t = softmax_with_temperature(logits_val, T=t_val)
        _, vp = predict_gated_hybrid(p_t, gate_val, norm_mem_val)
        vm = classification_metrics(y_val_arr, vp, num_classes)
        # Tie-break equal validation macro-F1 using validation negative log-likelihood cleanly.
        val_nll = float(-np.mean(np.log(np.clip(p_t[row_idx, y_val_arr], 1e-12, 1.0))))
        key = (-float(vm["macro_f1"]), -float(vm["accuracy"]), val_nll, abs(t_val - 1.0))
        if best_key is None or key < best_key:
            best_key = key
            best_res = (t_val, float(vm["accuracy"]), float(vm["macro_f1"]))
    assert best_res is not None
    return best_res


# Build averaged softmax probabilities and scaled ensemble embeddings ("avg" or "concat").
def _build_ensemble_representations(
    models: list[GNNClassifier],
    data: GraphData,
    embed_mode: str = "avg",
    clip: bool = True,
    l2_norm: bool = True,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, EmbeddingScaler]:
    probs_parts: list[np.ndarray] = []
    logits_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    for mod in models:
        lg = extract_logits(mod, data, device=device)
        logits_parts.append(lg)
        probs_parts.append(softmax_with_temperature(lg, T=1.0))
        z_raw = extract_embeddings(mod, data, device=device)
        if l2_norm:
            z_raw = l2_normalize_embeddings(z_raw)
        z_parts.append(z_raw)

    probs_ens = np.mean(probs_parts, axis=0)
    logits_ens = np.mean(logits_parts, axis=0)
    mode_clean = embed_mode.strip().lower()
    Z_comb = np.concatenate(z_parts, axis=1) if mode_clean == "concat" else np.mean(z_parts, axis=0)
    Z_scaled, scaler = fit_scaler_and_transform(Z_comb, data.train_idx, clip=clip, l2_norm=l2_norm)
    return probs_ens, logits_ens, Z_scaled, scaler


@dataclass
class SeedContext:
    """Precomputed representations and base models for a specific seed to allow reuse."""

    seed: int
    dataset_name: str
    gnn_model: GNNClassifier
    gnn_test_metrics: dict[str, float]
    gnn_train_time: float
    Z_scaled: np.ndarray
    base_fmnn: FMNN
    base_fit_time: float
    base_val_macro_f1: float
    base_val_acc: float
    scaler: EmbeddingScaler | None = None
    ensemble_models: list[GNNClassifier] | None = None
    ensemble_train_time: float = 0.0


# Return unified end-to-end test prediction latency when unified_latency is active.
def _maybe_measure_unified_latency(
    cfg: dict[str, Any],
    fallback_latency: float,
    variant_type: str,
    seed_ctx: SeedContext,
    data: GraphData,
    fmnn_model: FMNN | None = None,
    lam: float = 0.0,
    force: bool = False,
    T: float = 1.0,
    rho: float = 0.0,
    scaler_override: EmbeddingScaler | None = None,
    gnn_override: GNNClassifier | list[GNNClassifier] | None = None,
    ensemble_mode: str | None = None,
    ensemble_per_model_l2: bool = False,
    hybrid_mode: str | None = None,
) -> float:
    if not force and not bool(cfg.get("unified_latency", False)):
        return float(fallback_latency)

    scaler = scaler_override if scaler_override is not None else seed_ctx.scaler
    model_use = gnn_override if gnn_override is not None else seed_ctx.gnn_model
    if scaler is None and not isinstance(model_use, (list, tuple)):
        clip_setting = bool(cfg.get("embed", {}).get("clip", True))
        l2_norm_setting = bool(cfg.get("embed", {}).get("l2_norm", False))
        Z = extract_embeddings(model_use, data, device="cpu")
        _, scaler = fit_scaler_and_transform(Z, data.train_idx, clip=clip_setting, l2_norm=l2_norm_setting)
        del Z

    lat_cfg = cfg.get("latency", {})
    warmup = int(lat_cfg.get("warmup", 20))
    repeats = int(lat_cfg.get("repeats", 100))
    return measure_prediction_latency(
        variant_type=variant_type,
        gnn_model=model_use,
        data=data,
        scaler=scaler,
        fmnn_model=fmnn_model,
        lam=lam,
        warmup=warmup,
        repeats=repeats,
        T=T,
        rho=rho,
        ensemble_mode=ensemble_mode,
        ensemble_per_model_l2=ensemble_per_model_l2,
        hybrid_mode=hybrid_mode,
    )


# Train or reuse K independent GNN encoders for ensemble prediction.
def _get_or_train_ensemble(
    cfg: dict[str, Any],
    data: GraphData,
    seed: int,
    seed_ctx: SeedContext | None = None,
    num_models: int = 5,
    seed_stride: int = 100,
    evaluate_test: bool = False,
) -> tuple[list[GNNClassifier], float]:
    if seed_ctx is not None and seed_ctx.ensemble_models is not None and len(seed_ctx.ensemble_models) >= num_models:
        return list(seed_ctx.ensemble_models[:num_models]), float(seed_ctx.ensemble_train_time)

    t0 = time.perf_counter()
    models: list[GNNClassifier] = []
    for i in range(num_models):
        s_i = int(seed * seed_stride + i)
        if seed_ctx is not None and s_i == seed:
            models.append(seed_ctx.gnn_model)
        else:
            mod_i, *_ = train_gnn(data, cfg, seed=s_i, return_test_metrics=evaluate_test)
            models.append(mod_i)
    dt = float(time.perf_counter() - t0)
    if seed_ctx is not None:
        seed_ctx.ensemble_models = list(models)
        seed_ctx.ensemble_train_time = dt
    return models, dt


# Score an FMNN candidate on validation under 'none', 'global_lam', or 'gated' hybrid rules.
def _evaluate_candidate_scoring_val(
    fmnn_model: FMNN,
    Z_val: np.ndarray,
    logits_val: np.ndarray,
    probs_val: np.ndarray | None,
    y_val: np.ndarray,
    num_classes: int,
    hybrid_mode: str,
    T_grid: list[float],
    lam_grid: list[float],
    fixed_T: float | None = None,
    fixed_lam: float | None = None,
) -> dict[str, Any]:
    mode = hybrid_mode.strip().lower()
    if mode == "none":
        val_preds = fmnn_model.predict(Z_val)
        vm = classification_metrics(y_val, val_preds, num_classes)
        return {
            "chosen_T": 1.0,
            "chosen_lam": 0.0,
            "val_accuracy": float(vm["accuracy"]),
            "val_macro_f1": float(vm["macro_f1"]),
        }

    if mode == "gated":
        gate_val, norm_mem_val, _ = compute_single_box_gate_and_scores(fmnn_model, Z_val)
        t_val, v_acc, v_f1 = _select_gated_temperature(
            logits_val=logits_val,
            gate_val=gate_val,
            norm_mem_val=norm_mem_val,
            y_val=y_val,
            num_classes=num_classes,
            t_grid=T_grid,
            fixed_T=fixed_T,
        )
        return {
            "chosen_T": float(t_val),
            "chosen_lam": 0.0,
            "val_accuracy": float(v_acc),
            "val_macro_f1": float(v_f1),
        }

    # Global lambda blend with optional temperature scaling on logits.
    mem_val = fmnn_model.class_scores(Z_val)
    best_key: tuple | None = None
    best_out: dict[str, Any] | None = None
    t_candidates = [float(fixed_T)] if fixed_T is not None else [float(t) for t in T_grid]
    for t_cand in t_candidates:
        p_val = probs_val if (probs_val is not None and abs(t_cand - 1.0) < 1e-12) else softmax_with_temperature(logits_val, T=t_cand)
        lam_c, v_acc, v_f1 = _select_hybrid_lambda(
            probs_val=p_val,
            mem_val=mem_val,
            y_val=y_val,
            num_classes=num_classes,
            lam_grid=lam_grid,
            fixed_lam=fixed_lam,
        )
        key = (-v_f1, -v_acc, abs(t_cand - 1.0), lam_c)
        if best_key is None or key < best_key:
            best_key = key
            best_out = {
                "chosen_T": float(t_cand),
                "chosen_lam": float(lam_c),
                "val_accuracy": float(v_acc),
                "val_macro_f1": float(v_f1),
            }
    assert best_out is not None
    return best_out


# Fit FMNN with optional capped pseudo-labels and fast GSHR, selecting k on validation macro-F1.
def _fit_refine_scored_fmnn(
    seed_ctx: SeedContext,
    cfg: dict[str, Any],
    data: GraphData,
    graph_ctx: GraphContext,
    Z_scaled: np.ndarray,
    logits_all: np.ndarray,
    probs_all: np.ndarray | None,
    use_capped_pseudo: bool,
    use_fast_gshr: bool,
    hybrid_mode: str,
    T_grid: list[float],
    lam_grid: list[float],
    fixed_T: float | None = None,
    fixed_lam: float | None = None,
) -> dict[str, Any]:
    train_idx = data.train_idx.cpu().numpy()
    val_idx = data.val_idx.cpu().numpy()
    y_train = data.y[data.train_idx].cpu().numpy()
    y_val = data.y[data.val_idx].cpu().numpy()
    train_ids = train_idx.tolist()
    Z_train = Z_scaled[train_idx]
    Z_val = Z_scaled[val_idx]
    logits_val = logits_all[val_idx]
    probs_val = probs_all[val_idx] if probs_all is not None else None

    theta = float(cfg.get("fmnn", {}).get("theta", 0.3))
    gamma = float(cfg.get("fmnn", {}).get("gamma", 4.0))

    if not use_capped_pseudo:
        t0_f = time.perf_counter()
        base_fmnn = FMNN(theta=theta, gamma=gamma).fit(Z_train, y_train, train_ids)
        fit_t = time.perf_counter() - t0_f
        if use_fast_gshr:
            final_model, meta, _, _ = _select_fast_gshr(
                seed_ctx=seed_ctx,
                graph_ctx=graph_ctx,
                cfg=cfg,
                Z_train=Z_train,
                y_train=y_train,
                train_ids=train_ids,
                Z_val=Z_val,
                y_val=y_val,
                num_classes=data.num_classes,
                init_fmnn=base_fmnn,
            )
            th, tp, sg = meta["theta"], meta["tau_prune"], meta["safeguards"]
        else:
            final_model, th, tp, sg = base_fmnn, theta, None, None

        sc = _evaluate_candidate_scoring_val(
            final_model, Z_val, logits_val, probs_val, y_val, data.num_classes,
            hybrid_mode, T_grid, lam_grid, fixed_T=fixed_T, fixed_lam=fixed_lam,
        )
        return {
            "model": final_model,
            "chosen_k": None,
            "pseudo_count": 0,
            "pseudo_per_class": {},
            "chosen_theta": th,
            "chosen_tau_prune": tp,
            "chosen_safeguards": sg,
            "fit_time": fit_t,
            **sc,
        }

    b11_cfg = cfg.get("b11_lite", {})
    k_grid = [int(k) for k in b11_cfg.get("cap_per_class_grid", [20, 50, 100])]
    if b11_cfg.get("cap_per_class") is not None:
        k_grid = [int(b11_cfg["cap_per_class"])]
    min_conf = float(b11_cfg.get("min_conf", cfg.get("pseudo", {}).get("min_conf", 0.8)))
    allow_new_boxes = bool(b11_cfg.get("allow_new_boxes", True))
    exclude_val = bool(b11_cfg.get("exclude_val", False))
    exclude_test = bool(b11_cfg.get("exclude_test", False))
    p_source = probs_all if probs_all is not None else softmax_with_temperature(logits_all, T=1.0)

    best_key: tuple | None = None
    best_payload: dict[str, Any] | None = None
    for k_cand in k_grid:
        fit_idx, y_fit, pseudo_idx, y_pseudo, pseudo_ctx, p_cnt, per_cls = _build_capped_pseudo_fit_set(
            data=data, probs_all=p_source, max_per_class=k_cand, min_conf=min_conf,
            exclude_val=exclude_val, exclude_test=exclude_test,
        )
        Z_fit = Z_scaled[fit_idx]
        fit_ids = fit_idx.tolist()
        Z_ps = Z_scaled[pseudo_idx]
        ps_ids = pseudo_idx.tolist()

        t0_fk = time.perf_counter()
        p_base = FMNN(theta=theta, gamma=gamma).fit_with_pseudo(
            Z_train=Z_train, y_train=y_train, train_ids=train_ids,
            Z_pseudo=Z_ps, y_pseudo=y_pseudo, pseudo_ids=ps_ids,
            allow_new_boxes=allow_new_boxes,
        )
        fit_tk = time.perf_counter() - t0_fk
        if use_fast_gshr:
            mod_k, meta_k, _, _ = _select_fast_gshr(
                seed_ctx=seed_ctx, graph_ctx=pseudo_ctx, cfg=cfg,
                Z_train=Z_fit, y_train=y_fit, train_ids=fit_ids,
                Z_val=Z_val, y_val=y_val, num_classes=data.num_classes,
                init_fmnn=p_base,
                pseudo_args=(Z_train, y_train, train_ids, Z_ps, y_pseudo, ps_ids, allow_new_boxes),
            )
            th_k, tp_k, sg_k = meta_k["theta"], meta_k["tau_prune"], meta_k["safeguards"]
        else:
            mod_k, th_k, tp_k, sg_k = p_base, theta, None, None

        sc = _evaluate_candidate_scoring_val(
            mod_k, Z_val, logits_val, probs_val, y_val, data.num_classes,
            hybrid_mode, T_grid, lam_grid, fixed_T=fixed_T, fixed_lam=fixed_lam,
        )
        key = (-sc["val_macro_f1"], int(mod_k.num_boxes()), -sc["val_accuracy"], int(k_cand))
        if best_key is None or key < best_key:
            best_key = key
            best_payload = {
                "model": mod_k,
                "chosen_k": int(k_cand),
                "pseudo_count": int(p_cnt),
                "pseudo_per_class": per_cls,
                "chosen_theta": th_k,
                "chosen_tau_prune": tp_k,
                "chosen_safeguards": sg_k,
                "fit_time": float(fit_tk),
                **sc,
            }
        else:
            del mod_k
        if best_payload is not None and p_base is not best_payload["model"]:
            del p_base
        del Z_fit, Z_ps, pseudo_ctx

    assert best_payload is not None
    return best_payload


def prepare_seed_context(
    cfg: dict[str, Any],
    data: GraphData,
    seed: int,
    evaluate_test: bool = True,
) -> SeedContext:
    """Train GNN encoder, extract embeddings, and fit base FMNN for a single seed."""
    set_seed(seed)
    device = get_device(cfg)

    # 1. Train GNN with timing
    t0_gnn = time.perf_counter()
    if evaluate_test:
        gnn_model, _, gnn_test_metrics = train_gnn(data, cfg, seed=seed, return_test_metrics=True)
    else:
        gnn_model, _ = train_gnn(data, cfg, seed=seed, return_test_metrics=False)
        gnn_test_metrics = dict(getattr(gnn_model, "val_metrics", {}))
    gnn_train_time = time.perf_counter() - t0_gnn

    # 2. Extract and scale embeddings (scaler fitted strictly on training nodes)
    Z = extract_embeddings(gnn_model, data, device=device)
    clip_setting = bool(cfg.get("embed", {}).get("clip", True))
    l2_norm_setting = bool(cfg.get("embed", {}).get("l2_norm", False))
    Z_scaled, scaler = fit_scaler_and_transform(Z, data.train_idx, clip=clip_setting, l2_norm=l2_norm_setting)

    # Extract training arrays
    train_idx = data.train_idx.cpu().numpy()
    val_idx = data.val_idx.cpu().numpy()

    Z_train = Z_scaled[train_idx]
    y_train = data.y[data.train_idx].cpu().numpy()
    train_ids = train_idx.tolist()

    Z_val = Z_scaled[val_idx]
    y_val = data.y[data.val_idx].cpu().numpy()

    # 3. Fit base FMNN
    theta = float(cfg.get("fmnn", {}).get("theta", 0.3))
    gamma = float(cfg.get("fmnn", {}).get("gamma", 4.0))

    t0_fit = time.perf_counter()
    base_fmnn = FMNN(theta=theta, gamma=gamma)
    base_fmnn.fit(Z_train, y_train, train_ids)
    base_fit_time = time.perf_counter() - t0_fit

    val_preds_base = base_fmnn.predict(Z_val)
    val_metrics_base = classification_metrics(y_val, val_preds_base, data.num_classes)

    return SeedContext(
        seed=seed,
        dataset_name=data.name,
        gnn_model=gnn_model,
        gnn_test_metrics=gnn_test_metrics,
        gnn_train_time=gnn_train_time,
        Z_scaled=Z_scaled,
        base_fmnn=base_fmnn,
        base_fit_time=base_fit_time,
        base_val_macro_f1=val_metrics_base["macro_f1"],
        base_val_acc=val_metrics_base["accuracy"],
        scaler=scaler,
    )


def select_hyperparameters(
    base_fmnn: FMNN,
    graph_ctx: GraphContext,
    cfg: dict[str, Any],
    Z_train: np.ndarray,
    y_train: np.ndarray,
    train_ids: list[int],
    Z_val: np.ndarray,
    y_val: np.ndarray,
    variant: str,
    base_val_macro_f1: float,
) -> tuple[FMNN, float | None, float | None, float]:
    """Select best tau_merge and tau_prune from validation grid search.

    Selection rule:
      Among candidates whose val macro-F1 is >= (base_val_macro_f1 - 0.01),
      choose the one with the FEWEST boxes.
      Break ties by: higher validation macro-F1, then lower tau values.

    Returns:
        (best_model, best_tau_merge, best_tau_prune, refine_time)
    """
    t0_refine = time.perf_counter()
    num_classes = base_fmnn.num_classes

    # Grid definitions
    tau_merge_grid = cfg.get("tune", {}).get("tau_merge_grid", [0.3, 0.4, 0.5, 0.6])
    tau_prune_grid = cfg.get("tune", {}).get("tau_prune_grid", [0.0, 0.005, 0.01, 0.02])

    theta = float(cfg.get("fmnn", {}).get("theta", base_fmnn.theta))
    safeguards = cfg.get("safeguards", {})

    candidates: list[dict[str, Any]] = []

    if variant in ["B4", "graph_merge_only"]:
        for tau_m in tau_merge_grid:
            m_cfg = dict(cfg["merge"])
            m_cfg["tau_merge"] = tau_m
            m_cfg["theta"] = theta
            m_cfg["safeguards"] = safeguards
            cand_fmnn, _ = gshr_merge(base_fmnn, graph_ctx, m_cfg)

            val_preds = cand_fmnn.predict(Z_val)
            val_f1 = classification_metrics(y_val, val_preds, num_classes)["macro_f1"]
            candidates.append({
                "tau_merge": tau_m,
                "tau_prune": None,
                "val_f1": val_f1,
                "num_boxes": cand_fmnn.num_boxes(),
                "model": cand_fmnn,
            })

    elif variant in ["B5", "graph_prune_only"]:
        for tau_p in tau_prune_grid:
            p_cfg = dict(cfg["prune"])
            p_cfg["tau_prune"] = tau_p
            p_cfg["safeguards"] = safeguards
            cand_fmnn, _ = gshr_prune(
                fmnn=base_fmnn,
                graph_ctx=graph_ctx,
                cfg_prune=p_cfg,
                Z_train=Z_train,
                y_train=y_train,
                train_ids=train_ids,
                Z_val=Z_val,
                y_val=y_val,
            )
            val_preds = cand_fmnn.predict(Z_val)
            val_f1 = classification_metrics(y_val, val_preds, num_classes)["macro_f1"]
            candidates.append({
                "tau_merge": None,
                "tau_prune": tau_p,
                "val_f1": val_f1,
                "num_boxes": cand_fmnn.num_boxes(),
                "model": cand_fmnn,
            })

    elif variant in ["B6", "gshr_full"]:
        for tau_m in tau_merge_grid:
            m_cfg = dict(cfg["merge"])
            m_cfg["tau_merge"] = tau_m
            m_cfg["theta"] = theta
            m_cfg["safeguards"] = safeguards
            merged_model, _ = gshr_merge(base_fmnn, graph_ctx, m_cfg)

            for tau_p in tau_prune_grid:
                p_cfg = dict(cfg["prune"])
                p_cfg["tau_prune"] = tau_p
                p_cfg["safeguards"] = safeguards
                cand_fmnn, _ = gshr_prune(
                    fmnn=merged_model,
                    graph_ctx=graph_ctx,
                    cfg_prune=p_cfg,
                    Z_train=Z_train,
                    y_train=y_train,
                    train_ids=train_ids,
                    Z_val=Z_val,
                    y_val=y_val,
                )
                val_preds = cand_fmnn.predict(Z_val)
                val_f1 = classification_metrics(y_val, val_preds, num_classes)["macro_f1"]
                candidates.append({
                    "tau_merge": tau_m,
                    "tau_prune": tau_p,
                    "val_f1": val_f1,
                    "num_boxes": cand_fmnn.num_boxes(),
                    "model": cand_fmnn,
                })

    elif variant in ["B3", "geo_merge_prune"]:
        for tau_m in tau_merge_grid:
            m_cfg = dict(cfg["merge"])
            m_cfg["tau_merge"] = tau_m
            m_cfg["theta"] = theta
            m_cfg["safeguards"] = {"class_purity": True, "graph_association": False}
            merged_model, _ = geometry_only_merge(base_fmnn, graph_ctx, m_cfg)

            for tau_p in tau_prune_grid:
                p_cfg = dict(cfg["prune"])
                p_cfg["tau_prune"] = tau_p
                p_cfg["safeguards"] = {"class_purity": True, "graph_association": False}
                cand_fmnn, _ = geometry_only_prune(
                    fmnn=merged_model,
                    graph_ctx=graph_ctx,
                    cfg_prune=p_cfg,
                    Z_train=Z_train,
                    y_train=y_train,
                    train_ids=train_ids,
                    Z_val=Z_val,
                    y_val=y_val,
                )
                val_preds = cand_fmnn.predict(Z_val)
                val_f1 = classification_metrics(y_val, val_preds, num_classes)["macro_f1"]
                candidates.append({
                    "tau_merge": tau_m,
                    "tau_prune": tau_p,
                    "val_f1": val_f1,
                    "num_boxes": cand_fmnn.num_boxes(),
                    "model": cand_fmnn,
                })

    # Selection rule
    threshold_f1 = base_val_macro_f1 - 0.01
    valid_candidates = [c for c in candidates if c["val_f1"] >= threshold_f1]
    if not valid_candidates:
        valid_candidates = candidates

    # Sort key: fewest boxes, highest val_f1, lowest tau values
    def sort_key(c: dict[str, Any]) -> tuple:
        tm = c["tau_merge"] if c["tau_merge"] is not None else 0.0
        tp = c["tau_prune"] if c["tau_prune"] is not None else 0.0
        return (c["num_boxes"], -c["val_f1"], tm, tp)

    best = min(valid_candidates, key=sort_key)
    refine_time = time.perf_counter() - t0_refine

    return best["model"], best["tau_merge"], best["tau_prune"], refine_time


# Search (theta, tau_prune, safeguards) on validation macro-F1 using fast optional merge and fast prune.
def _select_fast_gshr(
    seed_ctx: SeedContext,
    graph_ctx: GraphContext,
    cfg: dict[str, Any],
    Z_train: np.ndarray,
    y_train: np.ndarray,
    train_ids: list[int],
    Z_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    init_fmnn: FMNN | None = None,
    pseudo_args: tuple[np.ndarray, np.ndarray, list[int], np.ndarray, np.ndarray, list[int], bool] | None = None,
) -> tuple[FMNN, dict[str, Any], list[dict[str, Any]], float]:
    t0_total = time.perf_counter()
    start_fmnn = init_fmnn if init_fmnn is not None else seed_ctx.base_fmnn
    fast_cfg = cfg.get("gshr_fast", {})
    theta_grid = [float(x) for x in fast_cfg.get("theta_grid", [0.3, 0.4, 0.5, 0.6])]
    tau_prune_grid = [float(x) for x in fast_cfg.get("tau_prune_grid", [0.0, 0.005, 0.01, 0.02])]
    safeguards_grid = [bool(x) for x in fast_cfg.get("safeguards_grid", [True, False])]
    if fast_cfg.get("theta") is not None:
        theta_grid = [float(fast_cfg["theta"])]
    if fast_cfg.get("tau_prune") is not None:
        tau_prune_grid = [float(fast_cfg["tau_prune"])]
    if fast_cfg.get("safeguards") is not None:
        safeguards_grid = [bool(fast_cfg["safeguards"])]

    gamma = float(cfg.get("fmnn", {}).get("gamma", 4.0))
    merge_enabled = bool(fast_cfg.get("merge_enabled", cfg.get("merge", {}).get("enabled", True)))
    require_same_ov = bool(fast_cfg.get("require_same_class_overlap", True))
    base_classes_present = {b.cls for b in start_fmnn.boxes.values()}

    settings_log: list[dict[str, Any]] = []
    best_rec: dict[str, Any] | None = None
    best_model: FMNN | None = None
    best_key: tuple | None = None

    for th in theta_grid:
        if abs(th - start_fmnn.theta) < 1e-12:
            fmnn_th = start_fmnn
        else:
            fmnn_th = FMNN(theta=th, gamma=gamma)
            if pseudo_args is not None:
                z_tr, y_tr, tr_ids, z_ps, y_ps, ps_ids, allow_new = pseudo_args
                fmnn_th.fit_with_pseudo(
                    z_tr, y_tr, tr_ids, z_ps, y_ps, ps_ids, allow_new_boxes=allow_new
                )
            else:
                fmnn_th.fit(Z_train, y_train, train_ids)

        for sg_on in safeguards_grid:
            sg_label = "on" if sg_on else "off"
            sg_dict = {"class_purity": sg_on, "graph_association": sg_on}
            m_cfg = dict(cfg.get("merge", {}))
            m_cfg["theta"] = th
            m_cfg["enabled"] = merge_enabled
            m_cfg["require_same_class_overlap"] = require_same_ov
            m_cfg["safeguards"] = sg_dict

            merged_model, merge_log = fast_gshr_merge(fmnn_th, graph_ctx, m_cfg)

            for tau_p in tau_prune_grid:
                p_cfg = dict(cfg.get("prune", {}))
                p_cfg["tau_prune"] = tau_p
                p_cfg["min_boxes_per_class"] = max(1, int(p_cfg.get("min_boxes_per_class", 1)))
                p_cfg["safeguards"] = sg_dict
                p_cfg["recall_safeguard"] = sg_on
                p_cfg["fast_mode"] = True

                cand_fmnn, prune_log = gshr_prune(
                    fmnn=merged_model,
                    graph_ctx=graph_ctx,
                    cfg_prune=p_cfg,
                    Z_train=Z_train,
                    y_train=y_train,
                    train_ids=train_ids,
                    Z_val=Z_val,
                    y_val=y_val,
                )

                # Enforce minimum box guard: every training class retains at least one box.
                assert {b.cls for b in cand_fmnn.boxes.values()} == base_classes_present
                assert cand_fmnn.num_boxes() >= len(base_classes_present)

                val_m = classification_metrics(y_val, cand_fmnn.predict(Z_val), num_classes)
                setting_time = float(merge_log.elapsed_time + prune_log.elapsed_time)
                rec = {
                    "setting": f"th={th}_tp={tau_p}_sg={sg_label}",
                    "theta": th,
                    "tau_prune": tau_p,
                    "safeguards": sg_label,
                    "num_boxes": int(cand_fmnn.num_boxes()),
                    "val_accuracy": float(val_m["accuracy"]),
                    "val_macro_f1": float(val_m["macro_f1"]),
                    "refine_time": setting_time,
                    "merge_executed": len(merge_log),
                }
                settings_log.append(rec)
                cand_key = (
                    -rec["val_macro_f1"],
                    rec["num_boxes"],
                    -rec["val_accuracy"],
                    rec["refine_time"],
                    rec["theta"],
                    rec["tau_prune"],
                    0 if rec["safeguards"] == "on" else 1,
                )
                if best_key is None or cand_key < best_key:
                    best_key = cand_key
                    best_rec = rec
                    best_model = cand_fmnn
                else:
                    del cand_fmnn

            del merged_model
        if fmnn_th is not start_fmnn:
            del fmnn_th

    assert best_model is not None and best_rec is not None
    total_time = float(time.perf_counter() - t0_total)
    return best_model, best_rec, settings_log, total_time


def run_single_seed(
    cfg: dict[str, Any],
    data: GraphData,
    seed: int,
    variant: str,
    seed_ctx: SeedContext | None = None,
) -> dict[str, Any]:
    """Execute a single pipeline run for a given seed and variant.

    Args:
        cfg: Configuration dictionary.
        data: GraphData benchmark instance.
        seed: Random seed.
        variant: Method variant name (e.g., 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9').
        seed_ctx: Optional precomputed seed context to avoid redundant retraining.

    Returns:
        Flat dictionary of results and metrics for reporting.
    """
    tracemalloc.start()

    variant_clean = variant.strip().upper()
    variant_aliases = {
        "GNN_SOFTMAX": "B1",
        "GNN_FMNN": "B2",
        "GEO_MERGE_PRUNE": "B3",
        "GRAPH_MERGE_ONLY": "B4",
        "GRAPH_PRUNE_ONLY": "B5",
        "GSHR_FULL": "B6",
        "B7": "B7_HYBRID",
        "B7_HYBRID": "B7_HYBRID",
        "HYBRID": "B7_HYBRID",
        "B8": "B8_HYBRID_PSEUDO",
        "B8_HYBRID_PSEUDO": "B8_HYBRID_PSEUDO",
        "HYBRID_PSEUDO": "B8_HYBRID_PSEUDO",
        "B9": "B9_GSHR_FAST",
        "B9_GSHR_FAST": "B9_GSHR_FAST",
        "GSHR_FAST": "B9_GSHR_FAST",
        "B10": "B10_FINAL",
        "B10_FINAL": "B10_FINAL",
        "B11": "B11_FINAL_LITE",
        "B11_FINAL_LITE": "B11_FINAL_LITE",
        "FINAL_LITE": "B11_FINAL_LITE",
        "B12": "B12_ENSEMBLE",
        "B12_ENSEMBLE": "B12_ENSEMBLE",
        "ENSEMBLE": "B12_ENSEMBLE",
        "B13": "B13_GATED_HYBRID",
        "B13_GATED_HYBRID": "B13_GATED_HYBRID",
        "GATED_HYBRID": "B13_GATED_HYBRID",
        "B14": "B14_FINAL",
        "B14_FINAL": "B14_FINAL",
        "KNN_EMBEDDING": "B_KNN",
    }
    canon_variant = variant_aliases.get(variant_clean, variant_clean)

    # 1. Prepare or reuse seed context
    if seed_ctx is None:
        seed_ctx = prepare_seed_context(cfg, data, seed)

    train_idx = data.train_idx.cpu().numpy()
    val_idx = data.val_idx.cpu().numpy()
    test_idx = data.test_idx.cpu().numpy()

    Z_scaled = seed_ctx.Z_scaled
    Z_train = Z_scaled[train_idx]
    y_train = data.y[data.train_idx].cpu().numpy()
    train_ids = train_idx.tolist()

    Z_val = Z_scaled[val_idx]
    y_val = data.y[data.val_idx].cpu().numpy()

    Z_test = Z_scaled[test_idx]

    # LEAKAGE GUARD: Build GraphContext with masked labels (-1 for non-train nodes)
    y_masked = np.full(data.num_nodes, -1, dtype=np.int64)
    y_masked[train_idx] = y_train

    # Assert that neither validation nor test labels are exposed in graph_ctx
    assert np.all(y_masked[val_idx] == -1), "Leakage guard violation: validation labels present in graph_ctx"
    assert np.all(y_masked[test_idx] == -1), "Leakage guard violation: test labels present in graph_ctx"

    train_mask_arr = np.zeros(data.num_nodes, dtype=bool)
    train_mask_arr[train_idx] = True

    graph_ctx = GraphContext(
        adj_csr=data.adj_csr,
        degree=data.degree,
        y=y_masked,
        train_mask=train_mask_arr,
    )

    # Initialize record fields
    chosen_tau_merge: float | None = None
    chosen_tau_prune: float | None = None
    refine_time = 0.0
    fmnn_fit_time = seed_ctx.base_fit_time
    gnn_train_time = seed_ctx.gnn_train_time

    val_macro_f1 = seed_ctx.base_val_macro_f1
    val_acc = seed_ctx.base_val_acc

    # 2. Execute Variant
    if canon_variant == "B1":
        # GNN softmax classifier
        test_acc = seed_ctx.gnn_test_metrics["test_acc"]
        test_macro_f1 = seed_ctx.gnn_test_metrics["test_macro_f1"]
        val_acc = seed_ctx.gnn_test_metrics["best_val_acc"]
        val_macro_f1 = seed_ctx.gnn_test_metrics["best_val_macro_f1"]

        # Latency and per-class recall at test evaluation
        device = get_device(cfg)
        t0_pred = time.perf_counter()
        with torch.no_grad():
            seed_ctx.gnn_model.eval()
            test_logits = seed_ctx.gnn_model(data.x.to(device), data.edge_index.to(device))[data.test_idx.to(device)]
        pred_latency_per_node = (time.perf_counter() - t0_pred) / len(test_idx)

        b1_test_preds = test_logits.argmax(dim=-1).cpu().numpy()
        y_test = data.y[data.test_idx].cpu().numpy()
        b1_test_m = classification_metrics(y_test, b1_test_preds, data.num_classes)

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        pred_latency_per_node = _maybe_measure_unified_latency(
            cfg=cfg,
            fallback_latency=pred_latency_per_node,
            variant_type="B1",
            seed_ctx=seed_ctx,
            data=data,
        )

        return {
            "dataset": data.name,
            "seed": seed,
            "variant": "B1_gnn_softmax",
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(test_macro_f1),
            "test_micro_f1": float(test_acc),
            "test_per_class_recall": b1_test_m["per_class_recall"],
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_macro_f1),
            "num_boxes": 0,
            "box_storage_bytes": 0,
            "mean_box_size": 0.0,
            "max_box_size": 0.0,
            "cross_class_overlap": 0,
            "chosen_tau_merge": None,
            "chosen_tau_prune": None,
            "gnn_train_time": gnn_train_time,
            "fmnn_fit_time": 0.0,
            "refine_time": 0.0,
            "pred_latency_per_node": pred_latency_per_node,
            "peak_memory_mb": peak_mem / (1024 * 1024),
        }

    elif canon_variant == "B2":
        # Plain FMNN without refinement
        final_model = seed_ctx.base_fmnn

    elif canon_variant == "B_KNN":
        # k-NN on scaled embeddings baseline
        t0_knn = time.perf_counter()
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(Z_train, y_train)
        refine_time = time.perf_counter() - t0_knn

        val_preds = knn.predict(Z_val)
        val_m = classification_metrics(y_val, val_preds, data.num_classes)
        val_acc = val_m["accuracy"]
        val_macro_f1 = val_m["macro_f1"]

        t0_pred = time.perf_counter()
        test_preds = knn.predict(Z_test)
        pred_latency_per_node = (time.perf_counter() - t0_pred) / len(test_idx)

        y_test = data.y[data.test_idx].cpu().numpy()
        test_m = classification_metrics(y_test, test_preds, data.num_classes)

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "dataset": data.name,
            "seed": seed,
            "variant": "B7_knn_embedding",
            "test_accuracy": test_m["accuracy"],
            "test_macro_f1": test_m["macro_f1"],
            "test_micro_f1": test_m["micro_f1"],
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "num_boxes": len(Z_train),
            "mean_box_size": 0.0,
            "max_box_size": 0.0,
            "cross_class_overlap": 0,
            "chosen_tau_merge": None,
            "chosen_tau_prune": None,
            "gnn_train_time": gnn_train_time,
            "fmnn_fit_time": fmnn_fit_time,
            "refine_time": refine_time,
            "pred_latency_per_node": pred_latency_per_node,
            "peak_memory_mb": peak_mem / (1024 * 1024),
        }

    elif canon_variant == "B7_HYBRID":
        # GNN softmax + GSHR-refined FMNN hybrid variant with validation lambda selection
        final_model, chosen_tau_merge, chosen_tau_prune, refine_time = select_hyperparameters(
            base_fmnn=seed_ctx.base_fmnn,
            graph_ctx=graph_ctx,
            cfg=cfg,
            Z_train=Z_train,
            y_train=y_train,
            train_ids=train_ids,
            Z_val=Z_val,
            y_val=y_val,
            variant="B6",
            base_val_macro_f1=seed_ctx.base_val_macro_f1,
        )

        probs_all = extract_softmax_probs(seed_ctx.gnn_model, data, device=get_device(cfg))
        probs_val = probs_all[val_idx]
        probs_test = probs_all[test_idx]

        mem_val = final_model.class_scores(Z_val)
        mem_test = final_model.class_scores(Z_test)

        hybrid_cfg = cfg.get("hybrid", {})
        lam_grid = [float(x) for x in hybrid_cfg.get("lambda_grid", [0.0, 0.25, 0.5, 0.75, 1.0])]
        fixed_lam = hybrid_cfg.get("lambda", None)
        sanity_check = bool(hybrid_cfg.get("sanity_check", True))

        chosen_lam, val_acc, val_macro_f1 = _select_hybrid_lambda(
            probs_val=probs_val,
            mem_val=mem_val,
            y_val=y_val,
            num_classes=data.num_classes,
            lam_grid=lam_grid,
            fixed_lam=fixed_lam,
        )

        y_test = data.y[data.test_idx].cpu().numpy()

        # Sanity checks: lam=0.0 matches B6, lam=1.0 matches B1
        preds_lam0 = predict_hybrid(probs_test, mem_test, 0.0)
        preds_b6 = final_model.predict(Z_test)
        m_lam0 = classification_metrics(y_test, preds_lam0, data.num_classes)
        m_b6 = classification_metrics(y_test, preds_b6, data.num_classes)
        lam0_ok = bool(
            np.array_equal(preds_lam0, preds_b6)
            and abs(m_lam0["accuracy"] - m_b6["accuracy"]) < 1e-12
            and abs(m_lam0["macro_f1"] - m_b6["macro_f1"]) < 1e-12
        )

        preds_lam1 = predict_hybrid(probs_test, mem_test, 1.0)
        m_lam1 = classification_metrics(y_test, preds_lam1, data.num_classes)
        b1_acc = float(seed_ctx.gnn_test_metrics["test_acc"])
        b1_f1 = float(seed_ctx.gnn_test_metrics["test_macro_f1"])
        lam1_ok = bool(
            abs(m_lam1["accuracy"] - b1_acc) < 1e-12
            and abs(m_lam1["macro_f1"] - b1_f1) < 1e-12
        )

        print(
            f"[B7_hybrid sanity seed={seed}] "
            f"lam=0.0 vs B6: acc {m_lam0['accuracy']:.4f}=={m_b6['accuracy']:.4f}, "
            f"f1 {m_lam0['macro_f1']:.4f}=={m_b6['macro_f1']:.4f} -> {'PASS' if lam0_ok else 'FAIL'} | "
            f"lam=1.0 vs B1: acc {m_lam1['accuracy']:.4f}=={b1_acc:.4f}, "
            f"f1 {m_lam1['macro_f1']:.4f}=={b1_f1:.4f} -> {'PASS' if lam1_ok else 'FAIL'}"
        )
        if sanity_check and (not lam0_ok or not lam1_ok):
            tracemalloc.stop()
            raise RuntimeError(
                f"B7_hybrid sanity check failed on seed {seed}: lam0_ok={lam0_ok}, lam1_ok={lam1_ok}"
            )

        t0_pred = time.perf_counter()
        test_preds = predict_hybrid(probs_test, mem_test, chosen_lam)
        pred_latency_per_node = (time.perf_counter() - t0_pred) / len(test_idx)

        test_metrics = classification_metrics(y_test, test_preds, data.num_classes)
        compact_m = compactness_metrics(final_model)
        cov_m = compute_box_coverage(final_model, Z_test, y_test, test_preds)

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        pred_latency_per_node = _maybe_measure_unified_latency(
            cfg=cfg,
            fallback_latency=pred_latency_per_node,
            variant_type="B7_HYBRID",
            seed_ctx=seed_ctx,
            data=data,
            fmnn_model=final_model,
            lam=chosen_lam,
        )

        return {
            "dataset": data.name,
            "seed": seed,
            "variant": "B7_hybrid",
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_micro_f1": test_metrics["micro_f1"],
            "test_per_class_recall": test_metrics["per_class_recall"],
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "num_boxes": compact_m["num_boxes"],
            "box_storage_bytes": int(compact_m["num_boxes"] * 2 * Z_scaled.shape[1] * 4),
            "mean_box_size": compact_m["mean_box_size"],
            "max_box_size": compact_m["max_box_size"],
            "cross_class_overlap": compact_m["cross_class_overlap"],
            "chosen_tau_merge": chosen_tau_merge,
            "chosen_tau_prune": chosen_tau_prune,
            "chosen_lam": chosen_lam,
            "box_coverage": cov_m["box_coverage"],
            "covered_count": cov_m["covered_count"],
            "uncovered_count": cov_m["uncovered_count"],
            "covered_accuracy": cov_m["covered_accuracy"],
            "uncovered_accuracy": cov_m["uncovered_accuracy"],
            "sanity_lam0_matches_b6": lam0_ok,
            "sanity_lam1_matches_b1": lam1_ok,
            "gnn_train_time": gnn_train_time,
            "fmnn_fit_time": fmnn_fit_time,
            "refine_time": refine_time,
            "pred_latency_per_node": pred_latency_per_node,
            "peak_memory_mb": peak_mem / (1024 * 1024),
        }

    elif canon_variant == "B8_HYBRID_PSEUDO":
        # Hybrid variant with FMNN fitted on train nodes plus confident GNN pseudo-labels
        probs_all = extract_softmax_probs(seed_ctx.gnn_model, data, device=get_device(cfg))
        probs_val = probs_all[val_idx]
        probs_test = probs_all[test_idx]

        theta = float(cfg.get("fmnn", {}).get("theta", 0.3))
        gamma = float(cfg.get("fmnn", {}).get("gamma", 4.0))
        hybrid_cfg = cfg.get("hybrid", {})
        lam_grid = [float(x) for x in hybrid_cfg.get("lambda_grid", [0.0, 0.25, 0.5, 0.75, 1.0])]
        fixed_lam = hybrid_cfg.get("lambda", None)

        pseudo_cfg = cfg.get("pseudo", {})
        pseudo_enabled = bool(pseudo_cfg.get("enabled", True))
        exclude_val = bool(pseudo_cfg.get("exclude_val", False))
        exclude_test = bool(pseudo_cfg.get("exclude_test", False))
        fixed_c = pseudo_cfg.get("c", None)
        if not pseudo_enabled:
            c_grid = [1.1]  # Strictly > max probability (1.0), yielding 0 pseudo-labels
        elif fixed_c is not None:
            c_grid = [float(fixed_c)]
        else:
            c_grid = [float(x) for x in pseudo_cfg.get("c_grid", [0.8, 0.9, 0.95])]

        b2_compact = compactness_metrics(seed_ctx.base_fmnn)
        c_trials: list[dict[str, Any]] = []
        t0_pseudo_total = time.perf_counter()

        for c_cand in c_grid:
            fit_idx, y_fit, _, pseudo_ctx, pseudo_count = _build_pseudo_fit_set(
                data=data,
                probs_all=probs_all,
                threshold_c=c_cand,
                exclude_val=exclude_val,
                exclude_test=exclude_test,
            )
            Z_fit = Z_scaled[fit_idx]
            fit_ids = fit_idx.tolist()

            t0_fit_c = time.perf_counter()
            pseudo_base_fmnn = FMNN(theta=theta, gamma=gamma)
            pseudo_base_fmnn.fit(Z_fit, y_fit, fit_ids)
            pseudo_fit_time_c = time.perf_counter() - t0_fit_c

            pseudo_base_compact = compactness_metrics(pseudo_base_fmnn)
            pseudo_base_val_preds = pseudo_base_fmnn.predict(Z_val)
            pseudo_base_val_f1 = classification_metrics(y_val, pseudo_base_val_preds, data.num_classes)["macro_f1"]

            cfg_b8_refine = copy.deepcopy(cfg)
            cfg_b8_refine.setdefault("prune", {})["fast_mode"] = bool(cfg.get("prune", {}).get("fast_mode", True))
            refined_c, tau_m_c, tau_p_c, refine_time_c = select_hyperparameters(
                base_fmnn=pseudo_base_fmnn,
                graph_ctx=pseudo_ctx,
                cfg=cfg_b8_refine,
                Z_train=Z_fit,
                y_train=y_fit,
                train_ids=fit_ids,
                Z_val=Z_val,
                y_val=y_val,
                variant="B6",
                base_val_macro_f1=pseudo_base_val_f1,
            )

            mem_val_c = refined_c.class_scores(Z_val)
            lam_c, val_acc_c, val_f1_c = _select_hybrid_lambda(
                probs_val=probs_val,
                mem_val=mem_val_c,
                y_val=y_val,
                num_classes=data.num_classes,
                lam_grid=lam_grid,
                fixed_lam=fixed_lam,
            )
            pure_fmnn_val_f1_c = classification_metrics(y_val, refined_c.predict(Z_val), data.num_classes)["macro_f1"]
            refined_compact_c = compactness_metrics(refined_c)

            c_trials.append({
                "c": float(c_cand),
                "pseudo_count": pseudo_count,
                "model": refined_c,
                "pseudo_base_compact": pseudo_base_compact,
                "refined_compact": refined_compact_c,
                "chosen_tau_merge": tau_m_c,
                "chosen_tau_prune": tau_p_c,
                "chosen_lam": lam_c,
                "val_accuracy": val_acc_c,
                "val_macro_f1": val_f1_c,
                "pure_fmnn_val_f1": float(pure_fmnn_val_f1_c),
                "pseudo_fit_time": pseudo_fit_time_c,
                "refine_time": refine_time_c,
            })

        # Select c on validation macro-F1; break ties by fewer boxes, pure-FMNN val F1, val acc, then c.
        best_c_trial = min(
            c_trials,
            key=lambda item: (
                -item["val_macro_f1"],
                item["refined_compact"]["num_boxes"],
                -item["pure_fmnn_val_f1"],
                -item["val_accuracy"],
                item["c"],
            ),
        )

        final_model = best_c_trial["model"]
        chosen_c = best_c_trial["c"] if pseudo_enabled else None
        pseudo_count = best_c_trial["pseudo_count"]
        chosen_lam = best_c_trial["chosen_lam"]
        chosen_tau_merge = best_c_trial["chosen_tau_merge"]
        chosen_tau_prune = best_c_trial["chosen_tau_prune"]
        val_acc = best_c_trial["val_accuracy"]
        val_macro_f1 = best_c_trial["val_macro_f1"]
        pseudo_base_compact = best_c_trial["pseudo_base_compact"]
        compact_m = best_c_trial["refined_compact"]
        refine_time = time.perf_counter() - t0_pseudo_total
        fmnn_fit_time = best_c_trial["pseudo_fit_time"]

        mem_test = final_model.class_scores(Z_test)
        y_test = data.y[data.test_idx].cpu().numpy()

        t0_pred = time.perf_counter()
        test_preds = predict_hybrid(probs_test, mem_test, chosen_lam)
        pred_latency_per_node = (time.perf_counter() - t0_pred) / len(test_idx)

        test_metrics = classification_metrics(y_test, test_preds, data.num_classes)
        cov_m = compute_box_coverage(final_model, Z_test, y_test, test_preds)

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        pred_latency_per_node = _maybe_measure_unified_latency(
            cfg=cfg,
            fallback_latency=pred_latency_per_node,
            variant_type="B8_HYBRID_PSEUDO",
            seed_ctx=seed_ctx,
            data=data,
            fmnn_model=final_model,
            lam=chosen_lam,
        )

        return {
            "dataset": data.name,
            "seed": seed,
            "variant": "B8_hybrid_pseudo",
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_micro_f1": test_metrics["micro_f1"],
            "test_per_class_recall": test_metrics["per_class_recall"],
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "num_boxes": compact_m["num_boxes"],
            "box_storage_bytes": int(compact_m["num_boxes"] * 2 * Z_scaled.shape[1] * 4),
            "mean_box_size": compact_m["mean_box_size"],
            "max_box_size": compact_m["max_box_size"],
            "cross_class_overlap": compact_m["cross_class_overlap"],
            "chosen_tau_merge": chosen_tau_merge,
            "chosen_tau_prune": chosen_tau_prune,
            "chosen_lam": chosen_lam,
            "chosen_c": chosen_c,
            "train_size": int(len(train_idx)),
            "pseudo_count": pseudo_count,
            "pseudo_base_num_boxes": pseudo_base_compact["num_boxes"],
            "pseudo_base_mean_box_size": pseudo_base_compact["mean_box_size"],
            "pseudo_base_cross_class_overlap": pseudo_base_compact["cross_class_overlap"],
            "delta_num_boxes_vs_b2": compact_m["num_boxes"] - b2_compact["num_boxes"],
            "delta_mean_box_size_vs_b2": compact_m["mean_box_size"] - b2_compact["mean_box_size"],
            "delta_cross_class_overlap_vs_b2": compact_m["cross_class_overlap"] - b2_compact["cross_class_overlap"],
            "delta_num_boxes_vs_pseudo_base": compact_m["num_boxes"] - pseudo_base_compact["num_boxes"],
            "delta_mean_box_size_vs_pseudo_base": compact_m["mean_box_size"] - pseudo_base_compact["mean_box_size"],
            "delta_cross_class_overlap_vs_pseudo_base": compact_m["cross_class_overlap"] - pseudo_base_compact["cross_class_overlap"],
            "box_coverage": cov_m["box_coverage"],
            "covered_count": cov_m["covered_count"],
            "uncovered_count": cov_m["uncovered_count"],
            "covered_accuracy": cov_m["covered_accuracy"],
            "uncovered_accuracy": cov_m["uncovered_accuracy"],
            "gnn_train_time": gnn_train_time,
            "fmnn_fit_time": fmnn_fit_time,
            "refine_time": refine_time,
            "pred_latency_per_node": pred_latency_per_node,
            "peak_memory_mb": peak_mem / (1024 * 1024),
        }

    elif canon_variant == "B9_GSHR_FAST":
        # Fast GSHR with optional geometric merge and validation search over theta, tau_prune, safeguards
        final_model, best_meta, settings_log, refine_time = _select_fast_gshr(
            seed_ctx=seed_ctx,
            graph_ctx=graph_ctx,
            cfg=cfg,
            Z_train=Z_train,
            y_train=y_train,
            train_ids=train_ids,
            Z_val=Z_val,
            y_val=y_val,
            num_classes=data.num_classes,
        )
        chosen_theta = best_meta["theta"]
        chosen_tau_prune = best_meta["tau_prune"]
        chosen_safeguards = best_meta["safeguards"]
        val_acc = best_meta["val_accuracy"]
        val_macro_f1 = best_meta["val_macro_f1"]

        t0_pred = time.perf_counter()
        test_preds = final_model.predict(Z_test)
        pred_latency_per_node = (time.perf_counter() - t0_pred) / len(test_idx)

        y_test = data.y[data.test_idx].cpu().numpy()
        test_metrics = classification_metrics(y_test, test_preds, data.num_classes)
        compact_m = compactness_metrics(final_model)
        cov_m = compute_box_coverage(final_model, Z_test, y_test, test_preds)

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        pred_latency_per_node = _maybe_measure_unified_latency(
            cfg=cfg,
            fallback_latency=pred_latency_per_node,
            variant_type="B9",
            seed_ctx=seed_ctx,
            data=data,
            fmnn_model=final_model,
        )

        return {
            "dataset": data.name,
            "seed": seed,
            "variant": "B9_gshr_fast",
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_micro_f1": test_metrics["micro_f1"],
            "test_per_class_recall": test_metrics["per_class_recall"],
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "num_boxes": compact_m["num_boxes"],
            "box_storage_bytes": int(compact_m["num_boxes"] * 2 * Z_scaled.shape[1] * 4),
            "mean_box_size": compact_m["mean_box_size"],
            "max_box_size": compact_m["max_box_size"],
            "cross_class_overlap": compact_m["cross_class_overlap"],
            "chosen_tau_merge": None,
            "chosen_tau_prune": chosen_tau_prune,
            "chosen_theta": chosen_theta,
            "chosen_safeguards": chosen_safeguards,
            "best_setting": best_meta["setting"],
            "best_setting_refine_time": best_meta["refine_time"],
            "settings_table": settings_log,
            "box_coverage": cov_m["box_coverage"],
            "covered_count": cov_m["covered_count"],
            "uncovered_count": cov_m["uncovered_count"],
            "covered_accuracy": cov_m["covered_accuracy"],
            "uncovered_accuracy": cov_m["uncovered_accuracy"],
            "gnn_train_time": gnn_train_time,
            "fmnn_fit_time": fmnn_fit_time,
            "refine_time": refine_time,
            "pred_latency_per_node": pred_latency_per_node,
            "peak_memory_mb": peak_mem / (1024 * 1024),
        }

    elif canon_variant == "B10_FINAL":
        # Validation-gated combination of best encoder + fast GSHR + hybrid + pseudo-labels
        b10_cfg = cfg.get("b10_components", {})
        use_fast_gshr = bool(b10_cfg.get("use_fast_gshr", True))
        use_hybrid = bool(b10_cfg.get("use_hybrid", True))
        use_pseudo = bool(b10_cfg.get("use_pseudo", False))

        theta = float(cfg.get("fmnn", {}).get("theta", 0.3))
        gamma = float(cfg.get("fmnn", {}).get("gamma", 4.0))
        hybrid_cfg = cfg.get("hybrid", {})
        lam_grid = [float(x) for x in hybrid_cfg.get("lambda_grid", [0.0, 0.25, 0.5, 0.75, 1.0])]
        fixed_lam = hybrid_cfg.get("lambda", None) if use_hybrid else 0.0

        probs_all = extract_softmax_probs(seed_ctx.gnn_model, data, device=get_device(cfg))
        probs_val = probs_all[val_idx]
        probs_test = probs_all[test_idx]

        t0_b10 = time.perf_counter()
        chosen_c: float | None = None
        pseudo_count = 0
        chosen_theta = theta
        chosen_tau_prune: float | None = None
        chosen_safeguards: str | None = None

        if use_pseudo:
            pseudo_cfg = cfg.get("pseudo", {})
            c_grid = [float(x) for x in pseudo_cfg.get("c_grid", [0.8, 0.9, 0.95])]
            if pseudo_cfg.get("c") is not None:
                c_grid = [float(pseudo_cfg["c"])]
            exclude_val = bool(pseudo_cfg.get("exclude_val", False))
            exclude_test = bool(pseudo_cfg.get("exclude_test", False))
            c_trials: list[dict[str, Any]] = []

            for c_cand in c_grid:
                fit_idx, y_fit, _, pseudo_ctx, p_cnt = _build_pseudo_fit_set(
                    data=data,
                    probs_all=probs_all,
                    threshold_c=c_cand,
                    exclude_val=exclude_val,
                    exclude_test=exclude_test,
                )
                Z_fit = Z_scaled[fit_idx]
                fit_ids = fit_idx.tolist()
                t0_fc = time.perf_counter()
                p_base = FMNN(theta=theta, gamma=gamma).fit(Z_fit, y_fit, fit_ids)
                p_fit_time = time.perf_counter() - t0_fc

                if use_fast_gshr:
                    model_c, meta_c, _, _ = _select_fast_gshr(
                        seed_ctx=seed_ctx,
                        graph_ctx=pseudo_ctx,
                        cfg=cfg,
                        Z_train=Z_fit,
                        y_train=y_fit,
                        train_ids=fit_ids,
                        Z_val=Z_val,
                        y_val=y_val,
                        num_classes=data.num_classes,
                        init_fmnn=p_base,
                    )
                    th_c = meta_c["theta"]
                    tp_c = meta_c["tau_prune"]
                    sg_c = meta_c["safeguards"]
                else:
                    model_c, th_c, tp_c, sg_c = p_base, theta, None, None

                mem_val_c = model_c.class_scores(Z_val)
                lam_c, v_acc_c, v_f1_c = _select_hybrid_lambda(
                    probs_val=probs_val,
                    mem_val=mem_val_c,
                    y_val=y_val,
                    num_classes=data.num_classes,
                    lam_grid=lam_grid,
                    fixed_lam=fixed_lam,
                )
                c_trials.append({
                    "c": c_cand,
                    "pseudo_count": p_cnt,
                    "model": model_c,
                    "theta": th_c,
                    "tau_prune": tp_c,
                    "safeguards": sg_c,
                    "lam": lam_c,
                    "val_accuracy": v_acc_c,
                    "val_macro_f1": v_f1_c,
                    "num_boxes": model_c.num_boxes(),
                    "fit_time": p_fit_time,
                })

            best_c = min(
                c_trials,
                key=lambda r: (-r["val_macro_f1"], r["num_boxes"], -r["val_accuracy"], r["c"]),
            )
            final_model = best_c["model"]
            chosen_c = float(best_c["c"])
            pseudo_count = int(best_c["pseudo_count"])
            chosen_theta = best_c["theta"]
            chosen_tau_prune = best_c["tau_prune"]
            chosen_safeguards = best_c["safeguards"]
            chosen_lam = float(best_c["lam"])
            val_acc = float(best_c["val_accuracy"])
            val_macro_f1 = float(best_c["val_macro_f1"])
            fmnn_fit_time = float(best_c["fit_time"])
        else:
            if use_fast_gshr:
                final_model, best_meta, _, _ = _select_fast_gshr(
                    seed_ctx=seed_ctx,
                    graph_ctx=graph_ctx,
                    cfg=cfg,
                    Z_train=Z_train,
                    y_train=y_train,
                    train_ids=train_ids,
                    Z_val=Z_val,
                    y_val=y_val,
                    num_classes=data.num_classes,
                )
                chosen_theta = best_meta["theta"]
                chosen_tau_prune = best_meta["tau_prune"]
                chosen_safeguards = best_meta["safeguards"]
            else:
                final_model = seed_ctx.base_fmnn

            mem_val = final_model.class_scores(Z_val)
            chosen_lam, val_acc, val_macro_f1 = _select_hybrid_lambda(
                probs_val=probs_val,
                mem_val=mem_val,
                y_val=y_val,
                num_classes=data.num_classes,
                lam_grid=lam_grid,
                fixed_lam=fixed_lam,
            )

        refine_time = time.perf_counter() - t0_b10
        mem_test = final_model.class_scores(Z_test)
        y_test = data.y[data.test_idx].cpu().numpy()

        t0_pred = time.perf_counter()
        test_preds = predict_hybrid(probs_test, mem_test, chosen_lam)
        pred_latency_per_node = (time.perf_counter() - t0_pred) / len(test_idx)

        test_metrics = classification_metrics(y_test, test_preds, data.num_classes)
        compact_m = compactness_metrics(final_model)
        cov_m = compute_box_coverage(final_model, Z_test, y_test, test_preds)

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        pred_latency_per_node = _maybe_measure_unified_latency(
            cfg=cfg,
            fallback_latency=pred_latency_per_node,
            variant_type="B10_FINAL",
            seed_ctx=seed_ctx,
            data=data,
            fmnn_model=final_model,
            lam=chosen_lam,
        )

        return {
            "dataset": data.name,
            "seed": seed,
            "variant": "B10_final",
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_micro_f1": test_metrics["micro_f1"],
            "test_per_class_recall": test_metrics["per_class_recall"],
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "num_boxes": compact_m["num_boxes"],
            "box_storage_bytes": int(compact_m["num_boxes"] * 2 * Z_scaled.shape[1] * 4),
            "mean_box_size": compact_m["mean_box_size"],
            "max_box_size": compact_m["max_box_size"],
            "cross_class_overlap": compact_m["cross_class_overlap"],
            "chosen_tau_merge": None,
            "chosen_tau_prune": chosen_tau_prune,
            "chosen_theta": chosen_theta,
            "chosen_safeguards": chosen_safeguards,
            "chosen_lam": chosen_lam,
            "chosen_c": chosen_c,
            "train_size": int(len(train_idx)),
            "pseudo_count": pseudo_count,
            "b10_components": dict(b10_cfg),
            "box_coverage": cov_m["box_coverage"],
            "covered_count": cov_m["covered_count"],
            "uncovered_count": cov_m["uncovered_count"],
            "covered_accuracy": cov_m["covered_accuracy"],
            "uncovered_accuracy": cov_m["uncovered_accuracy"],
            "gnn_train_time": gnn_train_time,
            "fmnn_fit_time": fmnn_fit_time,
            "refine_time": refine_time,
            "pred_latency_per_node": pred_latency_per_node,
            "peak_memory_mb": peak_mem / (1024 * 1024),
        }

    elif canon_variant == "B11_FINAL_LITE":
        # Lightweight final pipeline: best encoder + hybrid + fast GSHR + class-capped conservative pseudo-labels
        b11_cfg = cfg.get("b11_lite", {})
        use_capped_pseudo = bool(b11_cfg.get("use_capped_pseudo", not bool(cfg.get("no_pseudo", False))))
        use_fast_gshr = bool(b11_cfg.get("use_fast_gshr", True))
        use_hybrid = bool(b11_cfg.get("use_hybrid", True))
        allow_new_boxes = bool(b11_cfg.get("allow_new_boxes", True))
        min_conf = float(b11_cfg.get("min_conf", cfg.get("pseudo", {}).get("min_conf", 0.8)))

        k_grid = [int(k) for k in b11_cfg.get("cap_per_class_grid", [20, 50, 100])]
        if b11_cfg.get("cap_per_class") is not None:
            k_grid = [int(b11_cfg["cap_per_class"])]

        theta = float(cfg.get("fmnn", {}).get("theta", 0.3))
        gamma = float(cfg.get("fmnn", {}).get("gamma", 4.0))
        hybrid_cfg = cfg.get("hybrid", {})
        lam_grid = [float(x) for x in hybrid_cfg.get("lambda_grid", [0.0, 0.25, 0.5, 0.75, 1.0])]
        fixed_lam = hybrid_cfg.get("lambda", None) if use_hybrid else 0.0

        probs_all = extract_softmax_probs(seed_ctx.gnn_model, data, device=get_device(cfg))
        probs_val = probs_all[val_idx]
        probs_test = probs_all[test_idx]

        t0_b11 = time.perf_counter()
        chosen_k: int | None = None
        pseudo_count = 0
        pseudo_per_class: dict[int, int] = {}
        chosen_theta = theta
        chosen_tau_prune: float | None = None
        chosen_safeguards: str | None = None

        if use_capped_pseudo:
            exclude_val = bool(b11_cfg.get("exclude_val", cfg.get("pseudo", {}).get("exclude_val", False)))
            exclude_test = bool(b11_cfg.get("exclude_test", cfg.get("pseudo", {}).get("exclude_test", False)))

            best_k_key: tuple | None = None
            best_k_payload: dict[str, Any] | None = None

            for k_cand in k_grid:
                fit_idx, y_fit, pseudo_idx, y_pseudo, pseudo_ctx, p_cnt, per_cls = _build_capped_pseudo_fit_set(
                    data=data,
                    probs_all=probs_all,
                    max_per_class=k_cand,
                    min_conf=min_conf,
                    exclude_val=exclude_val,
                    exclude_test=exclude_test,
                )
                Z_fit = Z_scaled[fit_idx]
                fit_ids = fit_idx.tolist()
                Z_pseudo = Z_scaled[pseudo_idx]
                pseudo_ids = pseudo_idx.tolist()

                t0_fk = time.perf_counter()
                p_base = FMNN(theta=theta, gamma=gamma).fit_with_pseudo(
                    Z_train=Z_train,
                    y_train=y_train,
                    train_ids=train_ids,
                    Z_pseudo=Z_pseudo,
                    y_pseudo=y_pseudo,
                    pseudo_ids=pseudo_ids,
                    allow_new_boxes=allow_new_boxes,
                )
                p_fit_time = time.perf_counter() - t0_fk

                if use_fast_gshr:
                    model_k, meta_k, _, _ = _select_fast_gshr(
                        seed_ctx=seed_ctx,
                        graph_ctx=pseudo_ctx,
                        cfg=cfg,
                        Z_train=Z_fit,
                        y_train=y_fit,
                        train_ids=fit_ids,
                        Z_val=Z_val,
                        y_val=y_val,
                        num_classes=data.num_classes,
                        init_fmnn=p_base,
                        pseudo_args=(Z_train, y_train, train_ids, Z_pseudo, y_pseudo, pseudo_ids, allow_new_boxes),
                    )
                    th_k = meta_k["theta"]
                    tp_k = meta_k["tau_prune"]
                    sg_k = meta_k["safeguards"]
                else:
                    model_k, th_k, tp_k, sg_k = p_base, theta, None, None

                mem_val_k = model_k.class_scores(Z_val)
                lam_k, v_acc_k, v_f1_k = _select_hybrid_lambda(
                    probs_val=probs_val,
                    mem_val=mem_val_k,
                    y_val=y_val,
                    num_classes=data.num_classes,
                    lam_grid=lam_grid,
                    fixed_lam=fixed_lam,
                )
                k_key = (-v_f1_k, int(model_k.num_boxes()), -v_acc_k, int(k_cand))
                if best_k_key is None or k_key < best_k_key:
                    best_k_key = k_key
                    best_k_payload = {
                        "k": int(k_cand),
                        "pseudo_count": int(p_cnt),
                        "pseudo_per_class": per_cls,
                        "model": model_k,
                        "theta": th_k,
                        "tau_prune": tp_k,
                        "safeguards": sg_k,
                        "lam": float(lam_k),
                        "val_accuracy": float(v_acc_k),
                        "val_macro_f1": float(v_f1_k),
                        "fit_time": float(p_fit_time),
                    }
                else:
                    del model_k

                if best_k_payload is not None and p_base is not best_k_payload["model"]:
                    del p_base
                del Z_fit, Z_pseudo, mem_val_k, pseudo_ctx

            assert best_k_payload is not None
            final_model = best_k_payload["model"]
            chosen_k = best_k_payload["k"]
            pseudo_count = best_k_payload["pseudo_count"]
            pseudo_per_class = best_k_payload["pseudo_per_class"]
            chosen_theta = best_k_payload["theta"]
            chosen_tau_prune = best_k_payload["tau_prune"]
            chosen_safeguards = best_k_payload["safeguards"]
            chosen_lam = best_k_payload["lam"]
            val_acc = best_k_payload["val_accuracy"]
            val_macro_f1 = best_k_payload["val_macro_f1"]
            fmnn_fit_time = best_k_payload["fit_time"]
        else:
            if use_fast_gshr:
                final_model, best_meta, _, _ = _select_fast_gshr(
                    seed_ctx=seed_ctx,
                    graph_ctx=graph_ctx,
                    cfg=cfg,
                    Z_train=Z_train,
                    y_train=y_train,
                    train_ids=train_ids,
                    Z_val=Z_val,
                    y_val=y_val,
                    num_classes=data.num_classes,
                )
                chosen_theta = best_meta["theta"]
                chosen_tau_prune = best_meta["tau_prune"]
                chosen_safeguards = best_meta["safeguards"]
            else:
                final_model = seed_ctx.base_fmnn

            mem_val = final_model.class_scores(Z_val)
            chosen_lam, val_acc, val_macro_f1 = _select_hybrid_lambda(
                probs_val=probs_val,
                mem_val=mem_val,
                y_val=y_val,
                num_classes=data.num_classes,
                lam_grid=lam_grid,
                fixed_lam=fixed_lam,
            )
            del mem_val

        refine_time = time.perf_counter() - t0_b11
        mem_test = final_model.class_scores(Z_test)
        y_test = data.y[data.test_idx].cpu().numpy()
        test_preds = predict_hybrid(probs_test, mem_test, chosen_lam)
        del mem_test

        test_metrics = classification_metrics(y_test, test_preds, data.num_classes)
        compact_m = compactness_metrics(final_model)
        cov_m = compute_box_coverage(final_model, Z_test, y_test, test_preds)

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        pred_latency_per_node = _maybe_measure_unified_latency(
            cfg=cfg,
            fallback_latency=0.0,
            variant_type="B11_FINAL_LITE",
            seed_ctx=seed_ctx,
            data=data,
            fmnn_model=final_model,
            lam=chosen_lam,
            force=True,
        )

        return {
            "dataset": data.name,
            "seed": seed,
            "variant": "B11_final_lite",
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_micro_f1": test_metrics["micro_f1"],
            "test_per_class_recall": test_metrics["per_class_recall"],
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "num_boxes": compact_m["num_boxes"],
            "box_storage_bytes": int(compact_m["num_boxes"] * 2 * Z_scaled.shape[1] * 4),
            "mean_box_size": compact_m["mean_box_size"],
            "max_box_size": compact_m["max_box_size"],
            "cross_class_overlap": compact_m["cross_class_overlap"],
            "chosen_tau_merge": None,
            "chosen_tau_prune": chosen_tau_prune,
            "chosen_theta": chosen_theta,
            "chosen_safeguards": chosen_safeguards,
            "chosen_lam": chosen_lam,
            "chosen_k": chosen_k,
            "train_size": int(len(train_idx)),
            "pseudo_count": pseudo_count,
            "pseudo_per_class_counts": pseudo_per_class,
            "b11_lite_config": {
                "use_capped_pseudo": use_capped_pseudo,
                "use_fast_gshr": use_fast_gshr,
                "use_hybrid": use_hybrid,
                "allow_new_boxes": allow_new_boxes,
                "min_conf": min_conf,
            },
            "box_coverage": cov_m["box_coverage"],
            "covered_count": cov_m["covered_count"],
            "uncovered_count": cov_m["uncovered_count"],
            "covered_accuracy": cov_m["covered_accuracy"],
            "uncovered_accuracy": cov_m["uncovered_accuracy"],
            "gnn_train_time": gnn_train_time,
            "fmnn_fit_time": fmnn_fit_time,
            "refine_time": refine_time,
            "pred_latency_per_node": pred_latency_per_node,
            "peak_memory_mb": peak_mem / (1024 * 1024),
        }

    elif canon_variant == "B12_ENSEMBLE":
        # Ensemble of K encoders: average softmax probs, fit FMNN on val-selected avg vs concat embeddings.
        b12_cfg = cfg.get("b12_ensemble", {})
        num_models = int(b12_cfg.get("num_models", 5))
        seed_stride = int(b12_cfg.get("seed_stride", 100))
        embed_mode_opt = str(b12_cfg.get("embed_mode", "auto")).strip().lower()
        use_fast_gshr = bool(b12_cfg.get("use_fast_gshr", True))
        clip_setting = bool(cfg.get("embed", {}).get("clip", True))
        l2_norm_setting = bool(cfg.get("embed", {}).get("l2_norm", False))
        device = get_device(cfg)

        ens_models, ens_train_time = _get_or_train_ensemble(
            cfg=cfg, data=data, seed=seed, seed_ctx=seed_ctx,
            num_models=num_models, seed_stride=seed_stride, evaluate_test=False,
        )
        hybrid_cfg = cfg.get("hybrid", {})
        lam_grid = [float(x) for x in hybrid_cfg.get("lambda_grid", [0.0, 0.25, 0.5, 0.75, 1.0])]
        fixed_lam = hybrid_cfg.get("lambda", None)
        mode_candidates = ["avg", "concat"] if embed_mode_opt == "auto" else [embed_mode_opt]

        t0_b12 = time.perf_counter()
        best_key: tuple | None = None
        best_res: dict[str, Any] | None = None

        for m_mode in mode_candidates:
            probs_ens, logits_ens, Z_ens, scaler_ens = _build_ensemble_representations(
                ens_models, data, embed_mode=m_mode, clip=clip_setting, l2_norm=l2_norm_setting, device=device,
            )
            payload = _fit_refine_scored_fmnn(
                seed_ctx=seed_ctx, cfg=cfg, data=data, graph_ctx=graph_ctx,
                Z_scaled=Z_ens, logits_all=logits_ens, probs_all=probs_ens,
                use_capped_pseudo=False, use_fast_gshr=use_fast_gshr,
                hybrid_mode="global_lam", T_grid=[1.0], lam_grid=lam_grid,
                fixed_T=1.0, fixed_lam=fixed_lam,
            )
            mod_m = payload["model"]
            m_key = (-payload["val_macro_f1"], int(mod_m.num_boxes()), -payload["val_accuracy"], 0 if m_mode == "avg" else 1)
            if best_key is None or m_key < best_key:
                best_key = m_key
                best_res = {
                    **payload,
                    "embed_mode": m_mode,
                    "probs_ens": probs_ens,
                    "Z_ens": Z_ens,
                    "scaler_ens": scaler_ens,
                }
            else:
                del mod_m, Z_ens, scaler_ens

        assert best_res is not None
        refine_time = time.perf_counter() - t0_b12
        final_model = best_res["model"]
        Z_best = best_res["Z_ens"]
        probs_best = best_res["probs_ens"]
        chosen_embed_mode = str(best_res["embed_mode"])
        chosen_lam = float(best_res["chosen_lam"])

        mem_test = final_model.class_scores(Z_best[test_idx])
        test_preds = predict_hybrid(probs_best[test_idx], mem_test, chosen_lam)
        ens_softmax_preds = np.argmax(probs_best[test_idx], axis=1)
        fraction_changed_by_boxes = float(np.mean(test_preds != ens_softmax_preds))

        y_test = data.y[data.test_idx].cpu().numpy()
        test_metrics = classification_metrics(y_test, test_preds, data.num_classes)
        ens_sm_metrics = classification_metrics(y_test, ens_softmax_preds, data.num_classes)
        compact_m = compactness_metrics(final_model)
        cov_m = compute_box_coverage(final_model, Z_best[test_idx], y_test, test_preds)

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        pred_latency_per_node = _maybe_measure_unified_latency(
            cfg=cfg, fallback_latency=0.0, variant_type="B12_ENSEMBLE",
            seed_ctx=seed_ctx, data=data, fmnn_model=final_model, lam=chosen_lam, force=True,
            scaler_override=best_res["scaler_ens"], gnn_override=ens_models,
            ensemble_mode=chosen_embed_mode, ensemble_per_model_l2=l2_norm_setting,
        )

        return {
            "dataset": data.name,
            "seed": seed,
            "variant": "B12_ensemble",
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_micro_f1": test_metrics["micro_f1"],
            "test_per_class_recall": test_metrics["per_class_recall"],
            "val_accuracy": best_res["val_accuracy"],
            "val_macro_f1": best_res["val_macro_f1"],
            "num_boxes": compact_m["num_boxes"],
            "box_storage_bytes": int(compact_m["num_boxes"] * 2 * Z_best.shape[1] * 4),
            "mean_box_size": compact_m["mean_box_size"],
            "max_box_size": compact_m["max_box_size"],
            "cross_class_overlap": compact_m["cross_class_overlap"],
            "chosen_embed_mode": chosen_embed_mode,
            "chosen_lam": chosen_lam,
            "chosen_theta": best_res["chosen_theta"],
            "chosen_tau_prune": best_res["chosen_tau_prune"],
            "chosen_safeguards": best_res["chosen_safeguards"],
            "ens_softmax_test_accuracy": ens_sm_metrics["accuracy"],
            "ens_softmax_test_macro_f1": ens_sm_metrics["macro_f1"],
            "fraction_changed_by_boxes": fraction_changed_by_boxes,
            "box_coverage": cov_m["box_coverage"],
            "covered_count": cov_m["covered_count"],
            "uncovered_count": cov_m["uncovered_count"],
            "covered_accuracy": cov_m["covered_accuracy"],
            "uncovered_accuracy": cov_m["uncovered_accuracy"],
            "gnn_train_time": ens_train_time,
            "fmnn_fit_time": best_res["fit_time"],
            "refine_time": refine_time,
            "pred_latency_per_node": pred_latency_per_node,
            "peak_memory_mb": peak_mem / (1024 * 1024),
        }

    elif canon_variant == "B13_GATED_HYBRID":
        # Use box score only when node lies inside exact count==1 box; otherwise use T-scaled GNN probs.
        b13_cfg = cfg.get("b13_gated", {})
        use_fast_gshr = bool(b13_cfg.get("use_fast_gshr", True))
        t_grid = [float(t) for t in b13_cfg.get("T_grid", [0.5, 1.0, 1.5, 2.0])]
        fixed_T = b13_cfg.get("T", None)
        lam_grid = [float(x) for x in cfg.get("hybrid", {}).get("lambda_grid", [0.0, 0.25, 0.5, 0.75, 1.0])]

        device = get_device(cfg)
        logits_all = extract_logits(seed_ctx.gnn_model, data, device=device)
        probs_all_t1 = softmax_with_temperature(logits_all, T=1.0)

        t0_b13 = time.perf_counter()
        if use_fast_gshr:
            final_model, best_meta, _, _ = _select_fast_gshr(
                seed_ctx=seed_ctx, graph_ctx=graph_ctx, cfg=cfg,
                Z_train=Z_train, y_train=y_train, train_ids=train_ids,
                Z_val=Z_val, y_val=y_val, num_classes=data.num_classes,
            )
            chosen_theta = best_meta["theta"]
            chosen_tau_prune = best_meta["tau_prune"]
            chosen_safeguards = best_meta["safeguards"]
        else:
            final_model = seed_ctx.base_fmnn
            chosen_theta = float(cfg.get("fmnn", {}).get("theta", 0.3))
            chosen_tau_prune = None
            chosen_safeguards = None

        gate_val, norm_mem_val, _ = compute_single_box_gate_and_scores(final_model, Z_val)
        chosen_T, val_acc, val_macro_f1 = _select_gated_temperature(
            logits_val=logits_all[val_idx], gate_val=gate_val, norm_mem_val=norm_mem_val,
            y_val=y_val, num_classes=data.num_classes, t_grid=t_grid, fixed_T=fixed_T,
        )

        # Also evaluate B7 global lam on exact same boxes & encoder for direct comparison.
        mem_val_raw = final_model.class_scores(Z_val)
        b7_lam, _, _ = _select_hybrid_lambda(probs_all_t1[val_idx], mem_val_raw, y_val, data.num_classes, lam_grid)
        refine_time = time.perf_counter() - t0_b13

        gate_test, norm_mem_test, mem_test_raw = compute_single_box_gate_and_scores(final_model, Z_test)
        probs_test_T = softmax_with_temperature(logits_all[test_idx], T=chosen_T)
        _, test_preds = predict_gated_hybrid(probs_test_T, gate_test, norm_mem_test)
        gnn_test_preds = np.argmax(probs_test_T, axis=1)
        b7_test_preds = predict_hybrid(probs_all_t1[test_idx], mem_test_raw, b7_lam)

        y_test = data.y[data.test_idx].cpu().numpy()
        test_metrics = classification_metrics(y_test, test_preds, data.num_classes)
        b7_metrics = classification_metrics(y_test, b7_test_preds, data.num_classes)
        compact_m = compactness_metrics(final_model)
        cov_m = compute_box_coverage(final_model, Z_test, y_test, test_preds)

        dec_cnt = int(np.sum(gate_test))
        gnn_cnt = int(len(test_idx) - dec_cnt)
        box_dec_acc = float(np.mean(test_preds[gate_test] == y_test[gate_test])) if dec_cnt > 0 else None
        gnn_dec_acc = float(np.mean(test_preds[~gate_test] == y_test[~gate_test])) if gnn_cnt > 0 else None
        fraction_changed = float(np.mean(test_preds != gnn_test_preds))
        b7_fraction_changed = float(np.mean(b7_test_preds != np.argmax(probs_all_t1[test_idx], axis=1)))

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        pred_latency_per_node = _maybe_measure_unified_latency(
            cfg=cfg, fallback_latency=0.0, variant_type="B13_GATED_HYBRID",
            seed_ctx=seed_ctx, data=data, fmnn_model=final_model, T=chosen_T,
            hybrid_mode="gated", force=True,
        )

        return {
            "dataset": data.name,
            "seed": seed,
            "variant": "B13_gated_hybrid",
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_micro_f1": test_metrics["micro_f1"],
            "test_per_class_recall": test_metrics["per_class_recall"],
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "num_boxes": compact_m["num_boxes"],
            "box_storage_bytes": int(compact_m["num_boxes"] * 2 * Z_scaled.shape[1] * 4),
            "mean_box_size": compact_m["mean_box_size"],
            "max_box_size": compact_m["max_box_size"],
            "cross_class_overlap": compact_m["cross_class_overlap"],
            "chosen_T": chosen_T,
            "chosen_theta": chosen_theta,
            "chosen_tau_prune": chosen_tau_prune,
            "chosen_safeguards": chosen_safeguards,
            "box_decided_fraction": float(dec_cnt / max(1, len(test_idx))),
            "box_decided_count": dec_cnt,
            "gnn_decided_count": gnn_cnt,
            "box_decided_accuracy": box_dec_acc,
            "gnn_decided_accuracy": gnn_dec_acc,
            "fraction_changed_by_boxes": fraction_changed,
            "b7_chosen_lam": b7_lam,
            "b7_test_accuracy": b7_metrics["accuracy"],
            "b7_test_macro_f1": b7_metrics["macro_f1"],
            "b7_fraction_changed_by_boxes": b7_fraction_changed,
            "box_coverage": cov_m["box_coverage"],
            "covered_count": cov_m["covered_count"],
            "uncovered_count": cov_m["uncovered_count"],
            "covered_accuracy": cov_m["covered_accuracy"],
            "uncovered_accuracy": cov_m["uncovered_accuracy"],
            "gnn_train_time": gnn_train_time,
            "fmnn_fit_time": fmnn_fit_time,
            "refine_time": refine_time,
            "pred_latency_per_node": pred_latency_per_node,
            "peak_memory_mb": peak_mem / (1024 * 1024),
        }

    elif canon_variant == "B14_FINAL":
        # Assemble validation-selected v4 components with optional ensemble, capped pseudo-labels, gated/lam hybrid, and rho smoothing.
        b14_cfg = cfg.get("b14_components", {})
        use_ensemble = bool(b14_cfg.get("use_ensemble", False))
        ens_mode = str(b14_cfg.get("ensemble_embed_mode", "avg"))
        use_fast_gshr = bool(b14_cfg.get("use_fast_gshr", True))
        use_capped_pseudo = bool(b14_cfg.get("use_capped_pseudo", False))
        hybrid_mode = str(b14_cfg.get("hybrid_mode", "gated")).strip().lower()
        use_smoothing = bool(b14_cfg.get("use_smoothing", False))
        clip_setting = bool(cfg.get("embed", {}).get("clip", True))
        l2_norm_setting = bool(cfg.get("embed", {}).get("l2_norm", False))
        t_grid = [float(t) for t in cfg.get("b13_gated", {}).get("T_grid", [0.5, 1.0, 1.5, 2.0])]
        lam_grid = [float(x) for x in cfg.get("hybrid", {}).get("lambda_grid", [0.0, 0.25, 0.5, 0.75, 1.0])]
        rho_grid = [float(r) for r in cfg.get("smoothing", {}).get("rho_grid", [0.0, 0.1, 0.2, 0.3, 0.5])]
        fixed_rho = cfg.get("smoothing", {}).get("rho", None)
        device = get_device(cfg)

        if use_ensemble:
            ens_models, ens_t = _get_or_train_ensemble(
                cfg=cfg, data=data, seed=seed, seed_ctx=seed_ctx,
                num_models=int(cfg.get("b12_ensemble", {}).get("num_models", 5)),
                seed_stride=int(cfg.get("b12_ensemble", {}).get("seed_stride", 100)),
                evaluate_test=False,
            )
            probs_all, logits_all, Z_work, scaler_work = _build_ensemble_representations(
                ens_models, data, embed_mode=ens_mode, clip=clip_setting, l2_norm=l2_norm_setting, device=device,
            )
            gnn_time_used = ens_t
            gnn_for_lat: GNNClassifier | list[GNNClassifier] = ens_models
        else:
            logits_all = extract_logits(seed_ctx.gnn_model, data, device=device)
            probs_all = softmax_with_temperature(logits_all, T=1.0)
            Z_work = Z_scaled
            scaler_work = seed_ctx.scaler
            gnn_time_used = gnn_train_time
            gnn_for_lat = seed_ctx.gnn_model

        t0_b14 = time.perf_counter()
        payload = _fit_refine_scored_fmnn(
            seed_ctx=seed_ctx, cfg=cfg, data=data, graph_ctx=graph_ctx,
            Z_scaled=Z_work, logits_all=logits_all, probs_all=probs_all,
            use_capped_pseudo=use_capped_pseudo, use_fast_gshr=use_fast_gshr,
            hybrid_mode=hybrid_mode, T_grid=t_grid, lam_grid=lam_grid,
        )
        final_model = payload["model"]
        chosen_T = float(payload["chosen_T"])
        chosen_lam = float(payload["chosen_lam"])
        val_acc = float(payload["val_accuracy"])
        val_macro_f1 = float(payload["val_macro_f1"])

        # Build full-graph base score matrix matching hybrid_mode.
        if use_ensemble and abs(chosen_T - 1.0) > 1e-12:
            probs_T_all = np.mean([softmax_with_temperature(extract_logits(m, data, device=device), T=chosen_T) for m in gnn_for_lat], axis=0)  # type: ignore[arg-type]
        elif use_ensemble:
            probs_T_all = probs_all
        else:
            probs_T_all = probs_all if abs(chosen_T - 1.0) < 1e-12 else softmax_with_temperature(logits_all, T=chosen_T)

        if hybrid_mode == "gated":
            gate_all, norm_mem_all, _ = compute_single_box_gate_and_scores(final_model, Z_work)
            all_scores, _ = predict_gated_hybrid(probs_T_all, gate_all, norm_mem_all)
        else:
            mem_all = final_model.class_scores(Z_work)
            norm_mem_all = mem_all / (np.sum(mem_all, axis=1, keepdims=True) + 1e-9)
            if hybrid_mode == "none":
                gate_all = np.ones(data.num_nodes, dtype=bool)
                all_scores = norm_mem_all
            else:
                gate_all = np.any(mem_all >= 1.0 - 1e-12, axis=1)
                all_scores = chosen_lam * probs_T_all + (1.0 - chosen_lam) * norm_mem_all

        if use_smoothing:
            chosen_rho, final_scores, val_acc, val_macro_f1 = _select_smoothing_rho(
                all_scores, data.adj_csr, data.degree, val_idx, y_val, data.num_classes, rho_grid, fixed_rho=fixed_rho,
            )
        else:
            chosen_rho = 0.0
            final_scores = all_scores

        refine_time = time.perf_counter() - t0_b14
        test_preds = np.argmax(final_scores[test_idx], axis=1)
        gnn_only_test = np.argmax(probs_T_all[test_idx], axis=1)
        fraction_changed = float(np.mean(test_preds != gnn_only_test))

        y_test = data.y[data.test_idx].cpu().numpy()
        test_metrics = classification_metrics(y_test, test_preds, data.num_classes)
        compact_m = compactness_metrics(final_model)
        cov_m = compute_box_coverage(final_model, Z_work[test_idx], y_test, test_preds)

        gate_test = gate_all[test_idx]
        dec_cnt = int(np.sum(gate_test))
        gnn_cnt = int(len(test_idx) - dec_cnt)
        box_dec_acc = float(np.mean(test_preds[gate_test] == y_test[gate_test])) if dec_cnt > 0 else None
        gnn_dec_acc = float(np.mean(test_preds[~gate_test] == y_test[~gate_test])) if gnn_cnt > 0 else None

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        pred_latency_per_node = _maybe_measure_unified_latency(
            cfg=cfg, fallback_latency=0.0, variant_type="B14_FINAL",
            seed_ctx=seed_ctx, data=data, fmnn_model=final_model,
            lam=chosen_lam, T=chosen_T, rho=chosen_rho, force=True,
            scaler_override=scaler_work, gnn_override=gnn_for_lat,
            ensemble_mode=ens_mode if use_ensemble else None,
            ensemble_per_model_l2=l2_norm_setting,
            hybrid_mode=hybrid_mode,
        )

        return {
            "dataset": data.name,
            "seed": seed,
            "variant": "B14_final",
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_micro_f1": test_metrics["micro_f1"],
            "test_per_class_recall": test_metrics["per_class_recall"],
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "num_boxes": compact_m["num_boxes"],
            "box_storage_bytes": int(compact_m["num_boxes"] * 2 * Z_work.shape[1] * 4),
            "mean_box_size": compact_m["mean_box_size"],
            "max_box_size": compact_m["max_box_size"],
            "cross_class_overlap": compact_m["cross_class_overlap"],
            "chosen_T": chosen_T,
            "chosen_lam": chosen_lam,
            "chosen_rho": chosen_rho,
            "chosen_k": payload["chosen_k"],
            "pseudo_count": payload["pseudo_count"],
            "train_size": int(len(train_idx)),
            "chosen_theta": payload["chosen_theta"],
            "chosen_tau_prune": payload["chosen_tau_prune"],
            "chosen_safeguards": payload["chosen_safeguards"],
            "box_decided_fraction": float(dec_cnt / max(1, len(test_idx))),
            "box_decided_count": dec_cnt,
            "gnn_decided_count": gnn_cnt,
            "box_decided_accuracy": box_dec_acc,
            "gnn_decided_accuracy": gnn_dec_acc,
            "fraction_changed_by_boxes": fraction_changed,
            "b14_components": dict(b14_cfg),
            "box_coverage": cov_m["box_coverage"],
            "covered_count": cov_m["covered_count"],
            "uncovered_count": cov_m["uncovered_count"],
            "covered_accuracy": cov_m["covered_accuracy"],
            "uncovered_accuracy": cov_m["uncovered_accuracy"],
            "gnn_train_time": gnn_time_used,
            "fmnn_fit_time": payload["fit_time"],
            "refine_time": refine_time,
            "pred_latency_per_node": pred_latency_per_node,
            "peak_memory_mb": peak_mem / (1024 * 1024),
        }

    else:
        # Refinement variants (B3, B4, B5, B6)
        final_model, chosen_tau_merge, chosen_tau_prune, refine_time = select_hyperparameters(
            base_fmnn=seed_ctx.base_fmnn,
            graph_ctx=graph_ctx,
            cfg=cfg,
            Z_train=Z_train,
            y_train=y_train,
            train_ids=train_ids,
            Z_val=Z_val,
            y_val=y_val,
            variant=canon_variant,
            base_val_macro_f1=seed_ctx.base_val_macro_f1,
        )
        val_preds_final = final_model.predict(Z_val)
        val_m = classification_metrics(y_val, val_preds_final, data.num_classes)
        val_acc = val_m["accuracy"]
        val_macro_f1 = val_m["macro_f1"]

    # 3. Final Test Evaluation ONCE at the very end
    t0_pred = time.perf_counter()
    test_preds = final_model.predict(Z_test)
    pred_latency_per_node = (time.perf_counter() - t0_pred) / len(test_idx)

    # Access test labels strictly at final evaluation
    y_test = data.y[data.test_idx].cpu().numpy()
    test_metrics = classification_metrics(y_test, test_preds, data.num_classes)
    compact_m = compactness_metrics(final_model)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    pred_latency_per_node = _maybe_measure_unified_latency(
        cfg=cfg,
        fallback_latency=pred_latency_per_node,
        variant_type=canon_variant,
        seed_ctx=seed_ctx,
        data=data,
        fmnn_model=final_model,
    )

    variant_names = {
        "B2": "B2_gnn_fmnn",
        "B3": "B3_geo_merge_prune",
        "B4": "B4_graph_merge_only",
        "B5": "B5_graph_prune_only",
        "B6": "B6_gshr_full",
    }
    variant_name = variant_names.get(canon_variant, canon_variant)

    return {
        "dataset": data.name,
        "seed": seed,
        "variant": variant_name,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_micro_f1": test_metrics["micro_f1"],
        "test_per_class_recall": test_metrics["per_class_recall"],
        "val_accuracy": val_acc,
        "val_macro_f1": val_macro_f1,
        "num_boxes": compact_m["num_boxes"],
        "box_storage_bytes": int(compact_m["num_boxes"] * 2 * Z_scaled.shape[1] * 4),
        "mean_box_size": compact_m["mean_box_size"],
        "max_box_size": compact_m["max_box_size"],
        "cross_class_overlap": compact_m["cross_class_overlap"],
        "chosen_tau_merge": chosen_tau_merge,
        "chosen_tau_prune": chosen_tau_prune,
        "gnn_train_time": gnn_train_time,
        "fmnn_fit_time": fmnn_fit_time,
        "refine_time": refine_time,
        "pred_latency_per_node": pred_latency_per_node,
        "peak_memory_mb": peak_mem / (1024 * 1024),
    }


# Gate B10_final components on validation macro-F1 (ties -> fewer boxes) without reading test labels.
def select_b10_components(
    data: GraphData,
    cfg_base: dict[str, Any],
    cfg_v2: dict[str, Any],
    seeds: list[int],
) -> dict[str, Any]:
    train_idx = data.train_idx.cpu().numpy()
    val_idx = data.val_idx.cpu().numpy()
    y_train = data.y[data.train_idx].cpu().numpy()
    y_val = data.y[data.val_idx].cpu().numpy()
    train_ids = train_idx.tolist()

    y_masked = np.full(data.num_nodes, -1, dtype=np.int64)
    y_masked[train_idx] = y_train
    train_mask_arr = np.zeros(data.num_nodes, dtype=bool)
    train_mask_arr[train_idx] = True
    graph_ctx = GraphContext(adj_csr=data.adj_csr, degree=data.degree, y=y_masked, train_mask=train_mask_arr)

    device_base = get_device(cfg_base)
    device_v2 = get_device(cfg_v2)

    base_ctxs = [prepare_seed_context(cfg_base, data, s, evaluate_test=False) for s in seeds]
    v2_ctxs = [prepare_seed_context(cfg_v2, data, s, evaluate_test=False) for s in seeds]

    # Gate 1: Does v2 encoder + L2-minmax improve validation macro-F1 of FMNN (B2) over base encoder?
    f1_base_b2 = [c.base_val_macro_f1 for c in base_ctxs]
    f1_v2_b2 = [c.base_val_macro_f1 for c in v2_ctxs]
    f1_base_b1 = [float(c.gnn_test_metrics["best_val_macro_f1"]) for c in base_ctxs]
    f1_v2_b1 = [float(c.gnn_test_metrics["best_val_macro_f1"]) for c in v2_ctxs]
    boxes_base_b2 = [c.base_fmnn.num_boxes() for c in base_ctxs]
    boxes_v2_b2 = [c.base_fmnn.num_boxes() for c in v2_ctxs]

    use_best_encoder = bool(
        float(np.mean(f1_v2_b2)) > float(np.mean(f1_base_b2)) + 1e-12
        or (
            abs(float(np.mean(f1_v2_b2)) - float(np.mean(f1_base_b2))) <= 1e-12
            and float(np.mean(boxes_v2_b2)) < float(np.mean(boxes_base_b2))
        )
    )
    active_cfg = copy.deepcopy(cfg_v2 if use_best_encoder else cfg_base)
    active_ctxs = v2_ctxs if use_best_encoder else base_ctxs
    active_device = device_v2 if use_best_encoder else device_base
    curr_f1_mean = float(np.mean(f1_v2_b2 if use_best_encoder else f1_base_b2))
    curr_boxes_mean = float(np.mean(boxes_v2_b2 if use_best_encoder else boxes_base_b2))

    # Gate 2: Does fast GSHR improve validation macro-F1 (or tie with fewer boxes) vs plain FMNN?
    fast_models: list[FMNN] = []
    fast_f1s: list[float] = []
    fast_boxes: list[int] = []
    for ctx in active_ctxs:
        fm, meta, _, _ = _select_fast_gshr(
            seed_ctx=ctx,
            graph_ctx=graph_ctx,
            cfg=active_cfg,
            Z_train=ctx.Z_scaled[train_idx],
            y_train=y_train,
            train_ids=train_ids,
            Z_val=ctx.Z_scaled[val_idx],
            y_val=y_val,
            num_classes=data.num_classes,
        )
        fast_models.append(fm)
        fast_f1s.append(float(meta["val_macro_f1"]))
        fast_boxes.append(int(fm.num_boxes()))

    fg_f1_mean = float(np.mean(fast_f1s))
    fg_boxes_mean = float(np.mean(fast_boxes))
    use_fast_gshr = bool(
        fg_f1_mean > curr_f1_mean + 1e-12
        or (abs(fg_f1_mean - curr_f1_mean) <= 1e-12 and fg_boxes_mean < curr_boxes_mean)
    )
    if use_fast_gshr:
        stage2_models = fast_models
        curr_f1_mean = fg_f1_mean
        curr_boxes_mean = fg_boxes_mean
    else:
        stage2_models = [c.base_fmnn for c in active_ctxs]

    # Gate 3: Does hybrid prediction (lam > 0) improve validation macro-F1 over pure box predictions (lam=0)?
    lam_grid = [float(x) for x in active_cfg.get("hybrid", {}).get("lambda_grid", [0.0, 0.25, 0.5, 0.75, 1.0])]
    probs_val_list: list[np.ndarray] = []
    probs_all_list: list[np.ndarray] = []
    hyb_f1s: list[float] = []
    for ctx, mod in zip(active_ctxs, stage2_models):
        p_all = extract_softmax_probs(ctx.gnn_model, data, device=active_device)
        p_val = p_all[val_idx]
        probs_all_list.append(p_all)
        probs_val_list.append(p_val)
        m_val = mod.class_scores(ctx.Z_scaled[val_idx])
        _, _, h_f1 = _select_hybrid_lambda(p_val, m_val, y_val, data.num_classes, lam_grid)
        hyb_f1s.append(h_f1)

    hyb_f1_mean = float(np.mean(hyb_f1s))
    use_hybrid = bool(hyb_f1_mean > curr_f1_mean + 1e-12)
    if use_hybrid:
        curr_f1_mean = hyb_f1_mean

    # Gate 4: Do confident pseudo-labels improve validation macro-F1 (or tie with fewer boxes)?
    theta = float(active_cfg.get("fmnn", {}).get("theta", 0.3))
    gamma = float(active_cfg.get("fmnn", {}).get("gamma", 4.0))
    c_grid = [float(x) for x in active_cfg.get("pseudo", {}).get("c_grid", [0.8, 0.9, 0.95])]
    exclude_val = bool(active_cfg.get("pseudo", {}).get("exclude_val", False))
    exclude_test = bool(active_cfg.get("pseudo", {}).get("exclude_test", False))
    pseudo_f1s: list[float] = []
    pseudo_boxes: list[int] = []

    for ctx, p_all, p_val in zip(active_ctxs, probs_all_list, probs_val_list):
        c_trials = []
        for c_cand in c_grid:
            fit_idx, y_fit, _, pseudo_ctx, _ = _build_pseudo_fit_set(
                data=data, probs_all=p_all, threshold_c=c_cand,
                exclude_val=exclude_val, exclude_test=exclude_test,
            )
            Z_fit = ctx.Z_scaled[fit_idx]
            fit_ids = fit_idx.tolist()
            p_base = FMNN(theta=theta, gamma=gamma).fit(Z_fit, y_fit, fit_ids)
            if use_fast_gshr:
                mod_c, _, _, _ = _select_fast_gshr(
                    seed_ctx=ctx, graph_ctx=pseudo_ctx, cfg=active_cfg,
                    Z_train=Z_fit, y_train=y_fit, train_ids=fit_ids,
                    Z_val=ctx.Z_scaled[val_idx], y_val=y_val,
                    num_classes=data.num_classes, init_fmnn=p_base,
                )
            else:
                mod_c = p_base
            m_val_c = mod_c.class_scores(ctx.Z_scaled[val_idx])
            if use_hybrid:
                _, _, f1_c = _select_hybrid_lambda(p_val, m_val_c, y_val, data.num_classes, lam_grid)
            else:
                f1_c = float(classification_metrics(y_val, mod_c.predict(ctx.Z_scaled[val_idx]), data.num_classes)["macro_f1"])
            c_trials.append((f1_c, mod_c.num_boxes(), c_cand))
        best_c = min(c_trials, key=lambda x: (-x[0], x[1], x[2]))
        pseudo_f1s.append(float(best_c[0]))
        pseudo_boxes.append(int(best_c[1]))

    pseudo_f1_mean = float(np.mean(pseudo_f1s))
    pseudo_boxes_mean = float(np.mean(pseudo_boxes))
    use_pseudo = bool(
        pseudo_f1_mean > curr_f1_mean + 1e-12
        or (abs(pseudo_f1_mean - curr_f1_mean) <= 1e-12 and pseudo_boxes_mean < curr_boxes_mean)
    )

    return {
        "dataset": data.name,
        "seeds": [int(s) for s in seeds],
        "selected": {
            "use_best_encoder": use_best_encoder,
            "use_l2_norm": bool(active_cfg.get("embed", {}).get("l2_norm", False)) if use_best_encoder else False,
            "use_fast_gshr": use_fast_gshr,
            "use_hybrid": use_hybrid,
            "use_pseudo": use_pseudo,
        },
        "gate_details": {
            "encoder": {
                "base_b1_val_f1_mean": float(np.mean(f1_base_b1)),
                "v2_b1_val_f1_mean": float(np.mean(f1_v2_b1)),
                "base_b2_val_f1_mean": float(np.mean(f1_base_b2)),
                "v2_b2_val_f1_mean": float(np.mean(f1_v2_b2)),
                "base_b2_boxes_mean": float(np.mean(boxes_base_b2)),
                "v2_b2_boxes_mean": float(np.mean(boxes_v2_b2)),
                "kept": use_best_encoder,
            },
            "fast_gshr": {
                "before_val_f1_mean": float(np.mean(f1_v2_b2 if use_best_encoder else f1_base_b2)),
                "after_val_f1_mean": fg_f1_mean,
                "before_boxes_mean": float(np.mean(boxes_v2_b2 if use_best_encoder else boxes_base_b2)),
                "after_boxes_mean": fg_boxes_mean,
                "kept": use_fast_gshr,
            },
            "hybrid": {
                "before_val_f1_mean": fg_f1_mean if use_fast_gshr else float(np.mean(f1_v2_b2 if use_best_encoder else f1_base_b2)),
                "after_val_f1_mean": hyb_f1_mean,
                "kept": use_hybrid,
            },
            "pseudo": {
                "before_val_f1_mean": curr_f1_mean,
                "after_val_f1_mean": pseudo_f1_mean,
                "before_boxes_mean": curr_boxes_mean,
                "after_boxes_mean": pseudo_boxes_mean,
                "kept": use_pseudo,
            },
        },
    }


# Gate B14_final components strictly on validation macro-F1 (ties -> fewer boxes -> simpler) without reading test labels.
def select_b14_components(
    data: GraphData,
    cfg_enc: dict[str, Any],
    seeds: list[int] | None = None,
    ens_seeds_override: list[int] | None = None,
) -> dict[str, Any]:
    """Select only the v4 components that improve validation macro-F1 on a dataset."""
    val_seeds = [int(s) for s in (seeds if seeds is not None else [0, 1, 2])]
    train_idx = data.train_idx.cpu().numpy()
    val_idx = data.val_idx.cpu().numpy()
    y_train = data.y[data.train_idx].cpu().numpy()
    y_val = data.y[data.val_idx].cpu().numpy()
    train_ids = train_idx.tolist()

    y_masked = np.full(data.num_nodes, -1, dtype=np.int64)
    y_masked[train_idx] = y_train
    train_mask_arr = np.zeros(data.num_nodes, dtype=bool)
    train_mask_arr[train_idx] = True
    graph_ctx = GraphContext(adj_csr=data.adj_csr, degree=data.degree, y=y_masked, train_mask=train_mask_arr)

    device = get_device(cfg_enc)
    clip_setting = bool(cfg_enc.get("embed", {}).get("clip", True))
    t_grid = [float(t) for t in cfg_enc.get("b13_gated", {}).get("T_grid", [0.5, 1.0, 1.5, 2.0])]
    lam_grid = [float(x) for x in cfg_enc.get("hybrid", {}).get("lambda_grid", [0.0, 0.25, 0.5, 0.75, 1.0])]
    rho_grid = [float(r) for r in cfg_enc.get("smoothing", {}).get("rho_grid", [0.0, 0.1, 0.2, 0.3, 0.5])]

    # Prepare single-model contexts once (both l2_norm=False and l2_norm=True share the trained GNN model!).
    cfg_nol2 = copy.deepcopy(cfg_enc)
    cfg_nol2.setdefault("embed", {})["l2_norm"] = False
    cfg_l2 = copy.deepcopy(cfg_enc)
    cfg_l2.setdefault("embed", {})["l2_norm"] = True

    single_models: list[GNNClassifier] = []
    single_logits: list[np.ndarray] = []
    single_probs: list[np.ndarray] = []
    Z_single_raw: list[np.ndarray] = []
    ctxs_l2: list[SeedContext] = []
    ctxs_nol2: list[SeedContext] = []

    theta = float(cfg_enc.get("fmnn", {}).get("theta", 0.3))
    gamma = float(cfg_enc.get("fmnn", {}).get("gamma", 4.0))

    for s in val_seeds:
        ctx_t = prepare_seed_context(cfg_l2, data, s, evaluate_test=False)
        ctxs_l2.append(ctx_t)
        single_models.append(ctx_t.gnn_model)
        lg = extract_logits(ctx_t.gnn_model, data, device=device)
        single_logits.append(lg)
        single_probs.append(softmax_with_temperature(lg, T=1.0))
        Z_r = extract_embeddings(ctx_t.gnn_model, data, device=device)
        Z_single_raw.append(Z_r)

        Z_nl, sc_nl = fit_scaler_and_transform(Z_r, data.train_idx, clip=clip_setting, l2_norm=False)
        fm_nl = FMNN(theta=theta, gamma=gamma).fit(Z_nl[train_idx], y_train, train_ids)
        vm_nl = classification_metrics(y_val, fm_nl.predict(Z_nl[val_idx]), data.num_classes)
        ctxs_nol2.append(SeedContext(
            seed=s, dataset_name=data.name, gnn_model=ctx_t.gnn_model,
            gnn_test_metrics=ctx_t.gnn_test_metrics, gnn_train_time=ctx_t.gnn_train_time,
            Z_scaled=Z_nl, base_fmnn=fm_nl, base_fit_time=0.0,
            base_val_macro_f1=float(vm_nl["macro_f1"]), base_val_acc=float(vm_nl["accuracy"]),
            scaler=sc_nl,
        ))

    # Gate 1: Compare L2-norm False vs True on base single-encoder FMNN.
    f1_nol2 = float(np.mean([c.base_val_macro_f1 for c in ctxs_nol2]))
    bx_nol2 = float(np.mean([c.base_fmnn.num_boxes() for c in ctxs_nol2]))
    f1_l2 = float(np.mean([c.base_val_macro_f1 for c in ctxs_l2]))
    bx_l2 = float(np.mean([c.base_fmnn.num_boxes() for c in ctxs_l2]))
    use_l2_norm = bool(f1_l2 > f1_nol2 + 1e-12 or (abs(f1_l2 - f1_nol2) <= 1e-12 and bx_l2 < bx_nol2))

    active_cfg = copy.deepcopy(cfg_l2 if use_l2_norm else cfg_nol2)
    active_ctxs = ctxs_l2 if use_l2_norm else ctxs_nol2
    curr_f1 = f1_l2 if use_l2_norm else f1_nol2
    curr_boxes = bx_l2 if use_l2_norm else bx_nol2

    # Gate 2: Does 5-seed ensemble (avg vs concat) beat single encoder on validation macro-F1 (hybrid)?
    ens_eval_seeds = ens_seeds_override if ens_seeds_override is not None else val_seeds[:1]
    single_hyb_f1s = []
    for ctx, lg, pr in zip(active_ctxs[:len(ens_eval_seeds)], single_logits[:len(ens_eval_seeds)], single_probs[:len(ens_eval_seeds)]):
        sc_s = _evaluate_candidate_scoring_val(
            ctx.base_fmnn, ctx.Z_scaled[val_idx], lg[val_idx], pr[val_idx],
            y_val, data.num_classes, "global_lam", [1.0], lam_grid,
        )
        single_hyb_f1s.append(float(sc_s["val_macro_f1"]))
    single_hyb_mean = float(np.mean(single_hyb_f1s))

    best_ens_mode = "avg"
    best_ens_f1 = -1.0
    ens_mode_scores: dict[str, dict[str, float]] = {}
    num_ens = int(active_cfg.get("b12_ensemble", {}).get("num_models", 5))
    stride = int(active_cfg.get("b12_ensemble", {}).get("seed_stride", 100))
    ens_reps_cache: dict[int, tuple[list[GNNClassifier], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {}

    for s_idx, s in enumerate(ens_eval_seeds):
        e_mods, _ = _get_or_train_ensemble(active_cfg, data, s, seed_ctx=active_ctxs[s_idx], num_models=num_ens, seed_stride=stride)
        reps: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for m_mode in ["avg", "concat"]:
            pr_e, lg_e, Z_e, _ = _build_ensemble_representations(
                e_mods, data, embed_mode=m_mode, clip=clip_setting, l2_norm=use_l2_norm, device=device,
            )
            reps[m_mode] = (pr_e, lg_e, Z_e)
        ens_reps_cache[s] = (e_mods, reps)

    for m_mode in ["avg", "concat"]:
        e_f1s, e_bxs = [], []
        for s in ens_eval_seeds:
            pr_e, lg_e, Z_e = ens_reps_cache[s][1][m_mode]
            fm_e = FMNN(theta=theta, gamma=gamma).fit(Z_e[train_idx], y_train, train_ids)
            sc_e = _evaluate_candidate_scoring_val(
                fm_e, Z_e[val_idx], lg_e[val_idx], pr_e[val_idx],
                y_val, data.num_classes, "global_lam", [1.0], lam_grid,
            )
            e_f1s.append(float(sc_e["val_macro_f1"]))
            e_bxs.append(int(fm_e.num_boxes()))
        m_f1 = float(np.mean(e_f1s))
        m_bx = float(np.mean(e_bxs))
        ens_mode_scores[m_mode] = {"val_macro_f1": m_f1, "boxes": m_bx}
        if m_f1 > best_ens_f1 + 1e-12:
            best_ens_f1 = m_f1
            best_ens_mode = m_mode

    use_ensemble = bool(best_ens_f1 > single_hyb_mean + 1e-12)
    eval_seeds_use = ens_eval_seeds if use_ensemble else val_seeds

    def _get_repr(i: int, s: int) -> tuple[SeedContext, np.ndarray, np.ndarray, np.ndarray]:
        if use_ensemble:
            pr_e, lg_e, Z_e = ens_reps_cache[s][1][best_ens_mode]
            return active_ctxs[i], pr_e, lg_e, Z_e
        return active_ctxs[i], single_probs[i], single_logits[i], active_ctxs[i].Z_scaled

    # Gate 3: Fast GSHR refinement vs base FMNN.
    raw_f1s, raw_bxs, fg_f1s, fg_bxs = [], [], [], []
    for i, s in enumerate(eval_seeds_use):
        ctx_i, _, _, Z_i = _get_repr(i, s)
        fm_raw = FMNN(theta=theta, gamma=gamma).fit(Z_i[train_idx], y_train, train_ids)
        raw_vm = classification_metrics(y_val, fm_raw.predict(Z_i[val_idx]), data.num_classes)
        raw_f1s.append(float(raw_vm["macro_f1"]))
        raw_bxs.append(int(fm_raw.num_boxes()))
        fm_fg, meta_fg, _, _ = _select_fast_gshr(
            seed_ctx=ctx_i, graph_ctx=graph_ctx, cfg=active_cfg,
            Z_train=Z_i[train_idx], y_train=y_train, train_ids=train_ids,
            Z_val=Z_i[val_idx], y_val=y_val, num_classes=data.num_classes, init_fmnn=fm_raw,
        )
        fg_f1s.append(float(meta_fg["val_macro_f1"]))
        fg_bxs.append(int(fm_fg.num_boxes()))

    raw_f1_m, raw_bx_m = float(np.mean(raw_f1s)), float(np.mean(raw_bxs))
    fg_f1_m, fg_bx_m = float(np.mean(fg_f1s)), float(np.mean(fg_bxs))
    use_fast_gshr = bool(fg_f1_m > raw_f1_m + 1e-12 or (abs(fg_f1_m - raw_f1_m) <= 1e-12 and fg_bx_m < raw_bx_m))
    curr_f1 = fg_f1_m if use_fast_gshr else raw_f1_m
    curr_boxes = fg_bx_m if use_fast_gshr else raw_bx_m

    # Gate 4: Capped pseudo-labels (choosing k on validation macro-F1).
    no_ps_models: list[dict[str, Any]] = []
    yes_ps_models: list[dict[str, Any]] = []
    for i, s in enumerate(eval_seeds_use):
        ctx_i, pr_i, lg_i, Z_i = _get_repr(i, s)
        p_no = _fit_refine_scored_fmnn(
            ctx_i, active_cfg, data, graph_ctx, Z_i, lg_i, pr_i,
            use_capped_pseudo=False, use_fast_gshr=use_fast_gshr,
            hybrid_mode="global_lam", T_grid=t_grid, lam_grid=lam_grid,
        )
        p_yes = _fit_refine_scored_fmnn(
            ctx_i, active_cfg, data, graph_ctx, Z_i, lg_i, pr_i,
            use_capped_pseudo=True, use_fast_gshr=use_fast_gshr,
            hybrid_mode="global_lam", T_grid=t_grid, lam_grid=lam_grid,
        )
        no_ps_models.append(p_no)
        yes_ps_models.append(p_yes)

    nops_f1 = float(np.mean([p["val_macro_f1"] for p in no_ps_models]))
    nops_bx = float(np.mean([p["model"].num_boxes() for p in no_ps_models]))
    yps_f1 = float(np.mean([p["val_macro_f1"] for p in yes_ps_models]))
    yps_bx = float(np.mean([p["model"].num_boxes() for p in yes_ps_models]))
    use_capped_pseudo = bool(yps_f1 > nops_f1 + 1e-12 or (abs(yps_f1 - nops_f1) <= 1e-12 and yps_bx < nops_bx))

    chosen_payloads = yes_ps_models if use_capped_pseudo else no_ps_models

    # Gate 5: Hybrid mode ('none' vs 'gated' vs 'global_lam') on chosen boxes.
    mode_results: dict[str, dict[str, Any]] = {}
    for hm in ["none", "gated", "global_lam"]:
        hm_f1s, hm_details = [], []
        for i, s in enumerate(eval_seeds_use):
            _, pr_i, lg_i, Z_i = _get_repr(i, s)
            mod_i = chosen_payloads[i]["model"]
            sc_h = _evaluate_candidate_scoring_val(
                mod_i, Z_i[val_idx], lg_i[val_idx], pr_i[val_idx],
                y_val, data.num_classes, hm, t_grid, lam_grid,
            )
            hm_f1s.append(float(sc_h["val_macro_f1"]))
            hm_details.append(sc_h)
        mode_results[hm] = {"val_macro_f1": float(np.mean(hm_f1s)), "per_seed": hm_details}

    mode_priority = {"none": 0, "gated": 1, "global_lam": 2}
    best_hybrid_mode = min(
        ["none", "gated", "global_lam"],
        key=lambda m: (-mode_results[m]["val_macro_f1"], mode_priority[m]),
    )
    before_smooth_f1 = mode_results[best_hybrid_mode]["val_macro_f1"]

    # Gate 6: Prediction-time smoothing rho > 0 vs rho = 0.0.
    sm_f1s: list[float] = []
    for i, s in enumerate(eval_seeds_use):
        _, pr_i, lg_i, Z_i = _get_repr(i, s)
        mod_i = chosen_payloads[i]["model"]
        det = mode_results[best_hybrid_mode]["per_seed"][i]
        ch_T = float(det["chosen_T"])
        ch_lam = float(det["chosen_lam"])
        pr_T = pr_i if abs(ch_T - 1.0) < 1e-12 else softmax_with_temperature(lg_i, T=ch_T)
        if best_hybrid_mode == "gated":
            gate_i, norm_m_i, _ = compute_single_box_gate_and_scores(mod_i, Z_i)
            all_sc, _ = predict_gated_hybrid(pr_T, gate_i, norm_m_i)
        else:
            mem_i = mod_i.class_scores(Z_i)
            norm_m_i = mem_i / (np.sum(mem_i, axis=1, keepdims=True) + 1e-9)
            all_sc = norm_m_i if best_hybrid_mode == "none" else ch_lam * pr_T + (1.0 - ch_lam) * norm_m_i
        _, _, _, sm_f1 = _select_smoothing_rho(
            all_sc, data.adj_csr, data.degree, val_idx, y_val, data.num_classes, rho_grid,
        )
        sm_f1s.append(float(sm_f1))

    after_smooth_f1 = float(np.mean(sm_f1s))
    use_smoothing = bool(after_smooth_f1 > before_smooth_f1 + 1e-12)

    return {
        "dataset": data.name,
        "seeds": val_seeds,
        "encoder_model": str(cfg_enc.get("gnn", {}).get("model", "gcn")),
        "selected": {
            "use_l2_norm": use_l2_norm,
            "use_ensemble": use_ensemble,
            "ensemble_embed_mode": best_ens_mode,
            "use_fast_gshr": use_fast_gshr,
            "use_capped_pseudo": use_capped_pseudo,
            "hybrid_mode": best_hybrid_mode,
            "use_smoothing": use_smoothing,
        },
        "gate_details": {
            "l2_norm": {"f1_nol2": f1_nol2, "bx_nol2": bx_nol2, "f1_l2": f1_l2, "bx_l2": bx_l2, "kept": use_l2_norm},
            "ensemble": {"single_hyb_f1": single_hyb_mean, **ens_mode_scores, "kept": use_ensemble},
            "fast_gshr": {"raw_f1": raw_f1_m, "raw_bx": raw_bx_m, "fg_f1": fg_f1_m, "fg_bx": fg_bx_m, "kept": use_fast_gshr},
            "capped_pseudo": {"no_pseudo_f1": nops_f1, "no_pseudo_bx": nops_bx, "pseudo_f1": yps_f1, "pseudo_bx": yps_bx, "kept": use_capped_pseudo},
            "hybrid_mode": {k: v["val_macro_f1"] for k, v in mode_results.items()} | {"chosen": best_hybrid_mode},
            "smoothing": {"before_val_f1": before_smooth_f1, "after_val_f1": after_smooth_f1, "kept": use_smoothing},
        },
    }
