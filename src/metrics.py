"""Performance and model compactness evaluation metrics."""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score, recall_score

from src.embeddings import l2_normalize_embeddings
from src.fmnn import FMNN


def fast_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
) -> float:
    """Fast macro-averaged F1 score matching sklearn with zero_division=0.

    Computes F1 score averaged over the union of classes present in y_true and y_pred.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        num_classes: Optional total number of classes.

    Returns:
        Macro-averaged F1 score as float.
    """
    if hasattr(y_true, "cpu"):
        y_true = y_true.cpu().numpy()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.cpu().numpy()

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    if len(y_true_arr) == 0:
        return 0.0

    max_label = max(int(np.max(y_true_arr)), int(np.max(y_pred_arr)))
    if max_label < 0:
        return 0.0
    k = max_label + 1
    if num_classes is not None and num_classes > k:
        k = num_classes

    cm = np.bincount(y_true_arr * k + y_pred_arr, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    true_count = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    denom = true_count + pred_count
    present = denom > 0
    if not np.any(present):
        return 0.0
    f1_present = (2.0 * tp[present]) / denom[present]
    return float(np.mean(f1_present))


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    """Compute standard classification quality metrics.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        num_classes: Total number of classes.

    Returns:
        Dictionary with accuracy, macro_f1, micro_f1, and per-class recall list.
    """
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)

    acc = float((y_true_np == y_pred_np).mean()) if len(y_true_np) > 0 else 0.0
    macro_f1 = fast_macro_f1(y_true_np, y_pred_np, num_classes=num_classes)
    micro_f1 = float(f1_score(y_true_np, y_pred_np, average="micro", zero_division=0))

    class_labels = list(range(num_classes))
    recalls = recall_score(y_true_np, y_pred_np, average=None, labels=class_labels, zero_division=0)
    per_class_recall = [float(r) for r in recalls]

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_class_recall": per_class_recall,
    }


def compactness_metrics(fmnn: FMNN) -> dict[str, Any]:
    """Compute structural complexity and compactness metrics for an FMNN model.

    Args:
        fmnn: FMNN instance.

    Returns:
        Dictionary with num_boxes, boxes_per_class, mean_box_size, max_box_size,
        cross_class_overlap, and mean_side_length.
    """
    num_boxes = fmnn.num_boxes()
    boxes_per_class = fmnn.boxes_per_class()
    cross_overlap = fmnn.cross_class_overlap_count()
    assert cross_overlap == 0, f"FMNN must have 0 cross-class overlaps, found {cross_overlap}"

    if num_boxes == 0:
        return {
            "num_boxes": 0,
            "boxes_per_class": boxes_per_class,
            "mean_box_size": 0.0,
            "max_box_size": 0.0,
            "cross_class_overlap": 0,
            "mean_side_length": 0.0,
        }

    sizes = [fmnn.box_volume_proxy(b) for b in fmnn.boxes.values()]
    side_lengths = [b.w - b.v for b in fmnn.boxes.values()]
    all_sides = np.concatenate(side_lengths) if side_lengths else np.array([0.0])

    return {
        "num_boxes": num_boxes,
        "boxes_per_class": boxes_per_class,
        "mean_box_size": float(np.mean(sizes)),
        "max_box_size": float(np.max(sizes)),
        "cross_class_overlap": cross_overlap,
        "mean_side_length": float(np.mean(all_sides)),
    }


# Compute fraction of nodes inside at least one box and accuracy on covered vs uncovered nodes.
def compute_box_coverage(
    fmnn: FMNN,
    Z: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Measure test box coverage and accuracy stratified by covered vs uncovered nodes."""
    Z_arr = np.asarray(Z, dtype=np.float64)
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    total = int(len(Z_arr))

    if total == 0:
        return {
            "box_coverage": 0.0,
            "covered_count": 0,
            "uncovered_count": 0,
            "covered_accuracy": None,
            "uncovered_accuracy": None,
        }

    box_list = [fmnn.boxes[k] for k in sorted(fmnn.boxes.keys())]
    if not box_list:
        covered_mask = np.zeros(total, dtype=bool)
    else:
        V = np.stack([b.v for b in box_list], axis=0)
        W = np.stack([b.w for b in box_list], axis=0)
        inside = np.all((Z_arr[:, None, :] >= V[None, :, :] - eps) & (Z_arr[:, None, :] <= W[None, :, :] + eps), axis=2)
        covered_mask = np.any(inside, axis=1)

    cov_cnt = int(np.sum(covered_mask))
    uncov_cnt = int(total - cov_cnt)
    cov_acc = float(np.mean(y_p[covered_mask] == y_t[covered_mask])) if cov_cnt > 0 else None
    uncov_acc = float(np.mean(y_p[~covered_mask] == y_t[~covered_mask])) if uncov_cnt > 0 else None

    return {
        "box_coverage": float(cov_cnt / total),
        "covered_count": cov_cnt,
        "uncovered_count": uncov_cnt,
        "covered_accuracy": cov_acc,
        "uncovered_accuracy": uncov_acc,
    }


# Compute temperature-scaled softmax probabilities in float64 safely.
def _softmax_with_temperature_np(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    temp = max(1e-6, float(T))
    scaled = np.asarray(logits, dtype=np.float64) / temp
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exps = np.exp(scaled)
    return exps / np.sum(exps, axis=1, keepdims=True)


# Measure end-to-end test-batch prediction latency using 20 warm-up runs and median of 100 repeats on CPU.
def measure_prediction_latency(
    variant_type: str,
    gnn_model: torch.nn.Module | list[torch.nn.Module],
    data: Any,
    scaler: Any = None,
    fmnn_model: FMNN | None = None,
    lam: float = 0.0,
    warmup: int = 20,
    repeats: int = 100,
    T: float = 1.0,
    rho: float = 0.0,
    ensemble_mode: str | None = None,
    ensemble_per_model_l2: bool = False,
    hybrid_mode: str | None = None,
) -> float:
    is_ensemble = isinstance(gnn_model, (list, tuple))
    models_cpu = [copy.deepcopy(m).to("cpu").eval() for m in gnn_model] if is_ensemble else [copy.deepcopy(gnn_model).to("cpu").eval()]
    x_cpu = data.x.detach().to("cpu")
    ei_cpu = data.edge_index.detach().to("cpu")
    test_idx_cpu = data.test_idx.detach().to("cpu")
    test_idx_np = test_idx_cpu.numpy()
    num_test = max(1, int(len(test_idx_np)))
    vt = variant_type.strip().upper()
    use_gated = (hybrid_mode == "gated") or (vt in {"B13", "B13_GATED_HYBRID", "GATED"} and hybrid_mode is None)
    use_smoothing = float(rho) > 0.0

    # Precompute static normalized CSR transition matrix when graph smoothing rho > 0.0.
    P_csr = None
    if use_smoothing:
        adj = data.adj_csr.astype(np.float64)
        deg = np.asarray(data.degree, dtype=np.float64)
        deg_inv = np.where(deg > 0.0, 1.0 / np.maximum(deg, 1.0), 0.0)
        P_csr = sp.diags(deg_inv) @ adj

    box_list = [fmnn_model.boxes[k] for k in sorted(fmnn_model.boxes.keys())] if (fmnn_model is not None and use_gated) else []
    V_box = np.stack([b.v for b in box_list], axis=0) if box_list else None
    W_box = np.stack([b.w for b in box_list], axis=0) if box_list else None

    def _gate_mask(Z_sub: np.ndarray) -> np.ndarray:
        if V_box is None or W_box is None:
            return np.zeros(len(Z_sub), dtype=bool)
        counts = np.zeros(len(Z_sub), dtype=np.int32)
        step = 512
        for start in range(0, len(Z_sub), step):
            end = min(start + step, len(Z_sub))
            zc = Z_sub[start:end, None, :]
            inside = np.all((zc >= V_box[None, :, :] - 1e-12) & (zc <= W_box[None, :, :] + 1e-12), axis=2)
            counts[start:end] = np.sum(inside, axis=1)
        return counts == 1

    def predict_batch() -> np.ndarray:
        with torch.no_grad():
            if not is_ensemble and vt in {"B1", "B1_GNN_SOFTMAX", "SOFTMAX"} and not use_smoothing:
                logits = models_cpu[0](x_cpu, ei_cpu)[test_idx_cpu]
                return logits.argmax(dim=-1).numpy()

            # Compute node embeddings and classifier logits for 1 or K ensemble models.
            if not is_ensemble:
                z_all_raw = models_cpu[0].embed(x_cpu, ei_cpu)
                logits_all = models_cpu[0].classifier(z_all_raw)
                z_np = z_all_raw.numpy() if use_smoothing else z_all_raw[test_idx_cpu].numpy()
                logits_np = logits_all.numpy() if use_smoothing else logits_all[test_idx_cpu].numpy()
                probs_np = _softmax_with_temperature_np(logits_np, T=T)
            else:
                z_parts: list[np.ndarray] = []
                prob_parts: list[np.ndarray] = []
                for m_cpu in models_cpu:
                    z_m = m_cpu.embed(x_cpu, ei_cpu)
                    lg_m = m_cpu.classifier(z_m)
                    z_sub = z_m.numpy() if use_smoothing else z_m[test_idx_cpu].numpy()
                    lg_sub = lg_m.numpy() if use_smoothing else lg_m[test_idx_cpu].numpy()
                    if ensemble_per_model_l2:
                        z_sub = l2_normalize_embeddings(z_sub)
                    z_parts.append(z_sub)
                    prob_parts.append(_softmax_with_temperature_np(lg_sub, T=T))
                probs_np = np.mean(prob_parts, axis=0)
                if ensemble_mode == "concat":
                    z_np = np.concatenate(z_parts, axis=1)
                else:
                    z_np = np.mean(z_parts, axis=0)

            z_scaled_sub = scaler.transform(z_np)
            if (vt in {"B2", "B3", "B4", "B5", "B6", "B9", "FMNN"} or (lam == 0.0 and vt == "B10_PURE")) and not use_gated and not use_smoothing:
                assert fmnn_model is not None
                return fmnn_model.predict(z_scaled_sub)

            if hybrid_mode == "none" and not use_gated and not use_smoothing:
                assert fmnn_model is not None
                return fmnn_model.predict(z_scaled_sub)

            assert fmnn_model is not None
            mem = fmnn_model.class_scores(z_scaled_sub)
            norm_mem = mem / (np.sum(mem, axis=1, keepdims=True) + 1e-9)

            if use_gated:
                gate = _gate_mask(z_scaled_sub)
                scores = np.where(gate[:, None], norm_mem, probs_np)
            else:
                scores = float(lam) * probs_np + (1.0 - float(lam)) * norm_mem

            if use_smoothing and P_csr is not None:
                scores = (1.0 - float(rho)) * scores + float(rho) * (P_csr @ scores)
                return np.argmax(scores[test_idx_np], axis=1)

            return np.argmax(scores, axis=1)

    for _ in range(warmup):
        _ = predict_batch()

    elapsed_list: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = predict_batch()
        elapsed_list.append(time.perf_counter() - t0)

    del models_cpu, x_cpu, ei_cpu, test_idx_cpu
    return float(np.median(elapsed_list) / num_test)
