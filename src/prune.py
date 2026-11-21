"""Graph-Structure-Aware Hyperbox Pruning (GSHR Prune)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import time
from typing import Any, Iterable

import numpy as np

from src.fmnn import FMNN, HyperBox
from src.graph_stats import (
    class_purity,
    conductance,
    neighbor_label_agreement,
)
from src.merge import GraphContext, _unpack_graph_ctx


@dataclass
class RefinementMembershipContext:
    """Explicit refinement-local membership context caching exact evaluation matrices."""

    train_ids: list[int]
    val_ids: list[int]
    box_ids: list[int]
    id_to_col: dict[int, int]
    train_mem: np.ndarray | None
    val_mem: np.ndarray
    box_classes: np.ndarray
    box_v: np.ndarray
    box_w: np.ndarray

    @classmethod
    def build(
        cls,
        fmnn: FMNN,
        Z_train: np.ndarray,
        train_ids: list[int] | np.ndarray,
        Z_val: np.ndarray,
        val_ids: list[int] | np.ndarray,
    ) -> RefinementMembershipContext:
        """Construct exact membership context for the given model and splits."""
        box_list = sorted(fmnn.boxes.values(), key=lambda b: b.id)
        box_ids = [b.id for b in box_list]
        id_to_col = {bid: idx for idx, bid in enumerate(box_ids)}
        box_classes = np.array([b.cls for b in box_list], dtype=np.int64)
        box_v = (
            np.stack([b.v for b in box_list], axis=0)
            if box_list
            else np.empty((0, fmnn.n_dims), dtype=np.float64)
        )
        box_w = (
            np.stack([b.w for b in box_list], axis=0)
            if box_list
            else np.empty((0, fmnn.n_dims), dtype=np.float64)
        )

        val_mem = (
            fmnn.membership_matrix(Z_val, box_list)
            if len(Z_val) > 0 and len(box_list) > 0
            else np.zeros((len(Z_val), len(box_list)), dtype=np.float64)
        )
        train_mem = (
            fmnn.membership_matrix(Z_train, box_list)
            if len(Z_train) > 0 and len(box_list) > 0
            else None
        )

        return cls(
            train_ids=[int(x) for x in train_ids],
            val_ids=[int(x) for x in val_ids],
            box_ids=box_ids,
            id_to_col=id_to_col,
            train_mem=train_mem,
            val_mem=val_mem,
            box_classes=box_classes,
            box_v=box_v,
            box_w=box_w,
        )

    def is_valid_for(self, fmnn: FMNN) -> bool:
        """Verify that context geometry and box identities match the given model exactly."""
        box_list = sorted(fmnn.boxes.values(), key=lambda b: b.id)
        if len(box_list) != len(self.box_ids):
            return False
        for idx, b in enumerate(box_list):
            if b.id != self.box_ids[idx] or b.cls != self.box_classes[idx]:
                return False
            if np.max(np.abs(b.v - self.box_v[idx])) > 1e-12:
                return False
            if np.max(np.abs(b.w - self.box_w[idx])) > 1e-12:
                return False
        return True

    def compute_val_scores(
        self,
        active_box_ids: set[int] | list[int] | Any,
        num_classes: int,
    ) -> np.ndarray:
        """Compute validation class scores by slicing cached validation membership matrix."""
        num_val = len(self.val_ids)
        scores = np.zeros((num_val, num_classes), dtype=np.float64)
        active_set = set(active_box_ids) if not isinstance(active_box_ids, set) else active_box_ids
        for c in range(num_classes):
            c_cols = [
                self.id_to_col[bid]
                for bid in self.box_ids
                if bid in active_set and self.box_classes[self.id_to_col[bid]] == c
            ]
            if c_cols:
                scores[:, c] = np.max(self.val_mem[:, c_cols], axis=1)
        return scores

    def compute_val_predictions(
        self,
        active_box_ids: set[int] | list[int] | Any,
        num_classes: int,
    ) -> np.ndarray:
        """Compute deterministic validation predictions from active hyperboxes."""
        scores = self.compute_val_scores(active_box_ids, num_classes)
        return np.argmax(scores, axis=1)

    def reassign_train_members(
        self,
        fmnn: FMNN,
        active_box_ids: set[int] | list[int] | Any,
        y_train: np.ndarray,
    ) -> dict[int, int]:
        """Reassign training nodes using cached training membership matrix without recomputation."""
        return fmnn.reassign_members(
            Z_train=np.empty((0, fmnn.n_dims)),
            y_train=y_train,
            node_ids=self.train_ids,
            cached_train_mem=self.train_mem,
            id_to_col=self.id_to_col,
        )


class PruneLog(list):
    """Log container for pruning decisions with before/after validation metrics."""

    def __init__(
        self,
        visits: list[dict[str, Any]],
        boxes_before: int,
        boxes_after: int,
        pruned_count: int,
        elapsed_time: float,
        val_macro_f1_before: float,
        val_macro_f1_after: float,
        val_acc_before: float,
        val_acc_after: float,
        val_recalls_before: list[float],
        val_recalls_after: list[float],
    ) -> None:
        super().__init__(visits)
        self.boxes_before = boxes_before
        self.boxes_after = boxes_after
        self.pruned_count = pruned_count
        self.elapsed_time = elapsed_time
        self.val_macro_f1_before = val_macro_f1_before
        self.val_macro_f1_after = val_macro_f1_after
        self.val_acc_before = val_acc_before
        self.val_acc_after = val_acc_after
        self.val_recalls_before = val_recalls_before
        self.val_recalls_after = val_recalls_after

    def to_dict(self) -> dict[str, Any]:
        """Convert log metadata to dictionary format."""
        return {
            "visits": list(self),
            "boxes_before": self.boxes_before,
            "boxes_after": self.boxes_after,
            "pruned_count": self.pruned_count,
            "elapsed_time": self.elapsed_time,
            "val_macro_f1_before": self.val_macro_f1_before,
            "val_macro_f1_after": self.val_macro_f1_after,
            "val_acc_before": self.val_acc_before,
            "val_acc_after": self.val_acc_after,
            "val_recalls_before": self.val_recalls_before,
            "val_recalls_after": self.val_recalls_after,
        }


def _compute_box_redundancy(box: HyperBox, same_class_boxes: list[HyperBox]) -> float:
    """Compute maximum multi-dimensional overlap fraction with any same-class peer."""
    other_boxes = [b for b in same_class_boxes if b.id != box.id]
    if not other_boxes:
        return 0.0

    V_p = np.stack([peer.v for peer in other_boxes], axis=0)
    W_p = np.stack([peer.w for peer in other_boxes], axis=0)
    side = box.w - box.v
    zero_side = side < 1e-12

    ov_len = np.maximum(0.0, np.minimum(box.w[None, :], W_p) - np.maximum(box.v[None, :], V_p))
    # For zero-width sides check point containment, otherwise normalized overlap fraction.
    safe_side = np.where(zero_side, 1.0, side)[None, :]
    frac_pos = np.clip(ov_len / safe_side, 0.0, 1.0)
    frac_zero = np.where((V_p <= box.v[None, :]) & (box.v[None, :] <= W_p), 1.0, 0.0)
    dim_fracs = np.where(zero_side[None, :], frac_zero, frac_pos)

    avg_dim_frac = np.mean(dim_fracs, axis=1)
    return float(np.clip(np.max(avg_dim_frac), 0.0, 1.0))


# Compute accuracy, macro-F1, and per-class recalls identically to sklearn via NumPy confusion counts.
def _fast_confusion_metrics(
    y_true: np.ndarray,
    preds: np.ndarray,
    num_classes: int,
) -> tuple[float, float, np.ndarray]:
    cm = np.bincount(y_true * num_classes + preds, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    tp = np.diag(cm).astype(np.float64)
    actual = cm.sum(axis=1).astype(np.float64)
    pred_pos = cm.sum(axis=0).astype(np.float64)

    acc = float(tp.sum() / max(1, len(y_true)))
    recalls = np.where(actual > 0.0, tp / np.maximum(actual, 1.0), 0.0)
    denom = actual + pred_pos
    f1s = np.where(denom > 0.0, (2.0 * tp) / np.maximum(denom, 1.0), 0.0)
    present = (actual > 0.0) | (pred_pos > 0.0)
    macro_f1 = float(np.mean(f1s[present])) if np.any(present) else 0.0
    return acc, macro_f1, recalls


def _metrics_from_confusion_matrix(
    cm: np.ndarray,
    num_samples: int,
    num_classes: int,
) -> tuple[float, float, np.ndarray]:
    """Compute validation metrics from an already-maintained confusion matrix."""
    cm_float = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm_float)
    actual = cm_float.sum(axis=1)
    pred_pos = cm_float.sum(axis=0)

    acc = float(tp.sum() / max(1, num_samples))
    recalls = np.where(actual > 0.0, tp / np.maximum(actual, 1.0), 0.0)
    denom = actual + pred_pos
    f1s = np.where(denom > 0.0, (2.0 * tp) / np.maximum(denom, 1.0), 0.0)
    present = (actual > 0.0) | (pred_pos > 0.0)
    macro_f1 = float(np.mean(f1s[present])) if np.any(present) else 0.0
    return acc, macro_f1, recalls


def _compute_importance_score(
    box: HyperBox,
    fmnn: FMNN,
    ctx: GraphContext,
    Z_train: np.ndarray,
    node_to_train_idx: dict[int, int],
    mu1: float,
    mu2: float,
    mu3: float,
    mu4: float,
    mu5: float,
    boundary_ratio: float,
    graph_region_mix: float,
    use_graph_association: bool,
    train_mem_matrix: np.ndarray | None = None,
    id_to_col: dict[int, int] | None = None,
    class_to_other_cols: dict[int, list[int]] | None = None,
    fast_mode: bool = False,
) -> tuple[float, dict[str, float]]:
    """Compute normalized Importance(j) score and individual constituent terms."""
    c = box.cls
    same_class_boxes = fmnn.boxes_of_class(c)
    other_class_boxes = [b for b in fmnn.boxes.values() if b.cls != c]

    # 1. Coverage: relative member count within class scaled by class-balance factor (1 / num_boxes_in_class)
    max_class_members = max((len(b.members) for b in same_class_boxes), default=1)
    rel_size = len(box.members) / max(1, max_class_members)
    balance_factor = 1.0 / max(1, len(same_class_boxes))
    coverage = float(np.clip(rel_size * balance_factor, 0.0, 1.0))

    # 2. Purity (constant 1.0 in fast_mode; evaluated directly otherwise)
    if fast_mode:
        purity = 1.0
    else:
        purity = float(np.clip(class_purity(box.members, c, ctx.y, ctx.train_mask), 0.0, 1.0))

    # 3. Boundary Importance: fraction of member nodes close to competing other-class boxes
    if not box.members or not other_class_boxes:
        boundary_importance = 0.0
    elif train_mem_matrix is not None and id_to_col is not None and class_to_other_cols is not None:
        row_ids = [node_to_train_idx[nid] for nid in box.members if nid in node_to_train_idx]
        other_cols = class_to_other_cols.get(c, [])
        if not row_ids or not other_cols:
            boundary_importance = 0.0
        else:
            own_col = id_to_col[box.id]
            mem_own = train_mem_matrix[row_ids, own_col]
            mem_other = np.max(train_mem_matrix[np.ix_(row_ids, other_cols)], axis=1)
            close_count = int(np.sum(mem_other >= boundary_ratio * mem_own))
            boundary_importance = float(close_count / max(1, len(box.members)))
    else:
        close_count = 0
        for nid in box.members:
            if nid not in node_to_train_idx:
                continue
            z_u = Z_train[node_to_train_idx[nid]]
            mem_own = fmnn.membership(z_u, box)
            mem_other = max(fmnn.membership(z_u, b_oth) for b_oth in other_class_boxes)
            if mem_other >= boundary_ratio * mem_own:
                close_count += 1
        boundary_importance = float(close_count / max(1, len(box.members)))
    boundary_importance = float(np.clip(boundary_importance, 0.0, 1.0))

    # 4. Graph Region Importance
    if use_graph_association and box.members:
        cond = conductance(box.members, ctx.adj_csr, ctx.degree)
        agr, _ = neighbor_label_agreement(box.members, c, ctx.adj_csr, ctx.y, ctx.train_mask)
        graph_region = float(np.clip(graph_region_mix * (1.0 - cond) + (1.0 - graph_region_mix) * agr, 0.0, 1.0))
    else:
        graph_region = 0.0

    # 5. Redundancy
    redundancy = _compute_box_redundancy(box, same_class_boxes)

    if fast_mode:
        # Remove constant Purity term in fast mode
        importance = float(
            mu1 * coverage
            + mu3 * boundary_importance
            + mu4 * graph_region
            - mu5 * redundancy
        )
    else:
        importance = float(
            mu1 * coverage
            + mu2 * purity
            + mu3 * boundary_importance
            + mu4 * graph_region
            - mu5 * redundancy
        )

    terms = {
        "coverage": coverage,
        "purity": purity,
        "boundary_importance": boundary_importance,
        "graph_region": graph_region,
        "redundancy": redundancy,
    }
    return importance, terms


def compute_prune_importance(
    fmnn: FMNN,
    ctx: GraphContext,
    Z_train: np.ndarray,
    train_ids_list: list[int],
    cfg_prune: dict[str, Any],
    train_mem_matrix: np.ndarray | None = None,
    id_to_col: dict[int, int] | None = None,
    class_to_other_cols: dict[int, list[int]] | None = None,
    fast_mode: bool = True,
    mode: str = "gshr",
) -> list[tuple[float, int, dict[str, float]]]:
    """Compute and sort importance scores for all boxes once."""
    mu1 = float(cfg_prune.get("mu1", 0.25))
    mu2 = float(cfg_prune.get("mu2", 0.25))
    mu3 = float(cfg_prune.get("mu3", 0.20))
    mu4 = float(cfg_prune.get("mu4", 0.20))
    mu5 = float(cfg_prune.get("mu5", 0.10))
    boundary_ratio = float(cfg_prune.get("boundary_ratio", 0.5))
    graph_region_mix = float(cfg_prune.get("graph_region_mix", 0.5))
    safeguards = cfg_prune.get("safeguards", {})
    use_graph_association = safeguards.get("graph_association", True)
    if mode == "geometry_only":
        mu4 = 0.0
        use_graph_association = False

    node_to_train_idx = {nid: idx for idx, nid in enumerate(train_ids_list)}

    box_list = sorted(fmnn.boxes.values(), key=lambda b: b.id)
    if id_to_col is None:
        id_to_col = {b.id: idx for idx, b in enumerate(box_list)}
    if class_to_other_cols is None:
        class_to_other_cols = {
            c: [idx for idx, b in enumerate(box_list) if b.cls != c]
            for c in range(fmnn.num_classes)
        }

    scored_boxes: list[tuple[float, int, dict[str, float]]] = []
    for bid, box in sorted(fmnn.boxes.items()):
        if mode == "naive_coverage":
            imp = float(len(box.members))
            terms = {"coverage": imp, "purity": 1.0, "boundary_importance": 0.0, "graph_region": 0.0, "redundancy": 0.0}
        else:
            imp, terms = _compute_importance_score(
                box=box,
                fmnn=fmnn,
                ctx=ctx,
                Z_train=Z_train,
                node_to_train_idx=node_to_train_idx,
                mu1=mu1,
                mu2=mu2,
                mu3=mu3,
                mu4=mu4,
                mu5=mu5,
                boundary_ratio=boundary_ratio,
                graph_region_mix=graph_region_mix,
                use_graph_association=use_graph_association,
                train_mem_matrix=train_mem_matrix,
                id_to_col=id_to_col,
                class_to_other_cols=class_to_other_cols,
                fast_mode=fast_mode,
            )
        scored_boxes.append((imp, bid, terms))

    scored_boxes.sort(key=lambda item: (item[0], item[1]))
    return scored_boxes


def gshr_prune(
    fmnn: FMNN,
    graph_ctx: Any,
    cfg_prune: dict[str, Any],
    Z_train: np.ndarray,
    y_train: np.ndarray,
    train_ids: list[int] | np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    mode: str = "gshr",
    cached_val_mem: np.ndarray | None = None,
    cached_train_mem: np.ndarray | None = None,
    cached_importance: list[tuple[float, int, dict[str, float]]] | None = None,
    fast_mode: bool | None = None,
    init_val_scores: np.ndarray | None = None,
    init_val_preds: np.ndarray | None = None,
    init_cm: np.ndarray | None = None,
    init_metrics: tuple[float, float, np.ndarray] | None = None,
    defer_materialization: bool = False,
    mem_ctx: RefinementMembershipContext | None = None,
) -> tuple[FMNN | set[int], PruneLog]:
    """Execute Graph-Structure-Aware Hyperbox Pruning using validation feedback.

    Args:
        fmnn: Base FMNN model (will not be mutated).
        graph_ctx: GraphContext or GraphData container.
        cfg_prune: Pruning configuration dictionary.
        Z_train: Normalized training node embeddings.
        y_train: Training node ground-truth labels.
        train_ids: Original node IDs for training nodes.
        Z_val: Normalized validation node embeddings.
        y_val: Validation node ground-truth labels.
        mode: Pruning mode in {'gshr', 'geometry_only', 'naive_coverage'}.
        cached_val_mem: Optional precomputed validation membership matrix.
        cached_train_mem: Optional precomputed training membership matrix.
        cached_importance: Optional precomputed box importance scores.
        fast_mode: Optional boolean flag to enable fast pruning path.
        init_val_scores: Optional initial validation scores.
        init_val_preds: Optional initial validation predictions.
        init_cm: Optional initial validation confusion matrix.
        init_metrics: Optional initial validation metrics tuple (acc, macro_f1, recalls).
        defer_materialization: If True, returns active box ID set without copying FMNN.
        mem_ctx: Optional RefinementMembershipContext holding exact cached matrices.

    Returns:
        (new_fmnn_or_active_box_ids, prune_log)
    """
    start_time = time.perf_counter()
    new_fmnn = fmnn if defer_materialization else fmnn.copy()
    ctx = _unpack_graph_ctx(graph_ctx)

    train_ids_list = [int(nid) for nid in train_ids]
    node_to_train_idx = {nid: idx for idx, nid in enumerate(train_ids_list)}

    # Pruning hyperparameters
    mu1 = float(cfg_prune.get("mu1", 0.25))
    mu2 = float(cfg_prune.get("mu2", 0.25))
    mu3 = float(cfg_prune.get("mu3", 0.20))
    mu4 = float(cfg_prune.get("mu4", 0.20))
    mu5 = float(cfg_prune.get("mu5", 0.10))
    tau_prune = float(cfg_prune.get("tau_prune", 0.005))
    # Never allow min_boxes_per_class below 1 so pruning cannot drop the last box of any class.
    min_boxes_per_class = max(1, int(cfg_prune.get("min_boxes_per_class", 1)))
    boundary_ratio = float(cfg_prune.get("boundary_ratio", 0.5))
    graph_region_mix = float(cfg_prune.get("graph_region_mix", 0.5))
    if fast_mode is None:
        fast_mode = bool(cfg_prune.get("fast_mode", True) or cfg_prune.get("fast", True))
    recall_safeguard = bool(cfg_prune.get("recall_safeguard", True))

    safeguards = cfg_prune.get("safeguards", {})
    use_graph_association = safeguards.get("graph_association", True)
    if mode == "geometry_only":
        mu4 = 0.0
        use_graph_association = False

    num_classes = new_fmnn.num_classes
    num_val = len(Z_val)
    boxes_before = new_fmnn.num_boxes()
    initial_classes_present = {b.cls for b in new_fmnn.boxes.values()}

    # 1. Precompute validation and training membership matrices once
    if mem_ctx is not None:
        val_mem_matrix = mem_ctx.val_mem
        train_mem_matrix = mem_ctx.train_mem
        id_to_col = mem_ctx.id_to_col
        box_list = sorted(new_fmnn.boxes.values(), key=lambda b: b.id)
        all_box_ids = mem_ctx.box_ids
    else:
        box_list = sorted(new_fmnn.boxes.values(), key=lambda b: b.id)
        all_box_ids = [b.id for b in box_list]
        id_to_col = {bid: idx for idx, bid in enumerate(all_box_ids)}
        if cached_val_mem is not None:
            val_mem_matrix = cached_val_mem
        else:
            val_mem_matrix = new_fmnn.membership_matrix(Z_val, box_list)

        if cached_train_mem is not None:
            train_mem_matrix = cached_train_mem
        elif len(Z_train) > 0:
            train_mem_matrix = new_fmnn.membership_matrix(Z_train, box_list)
        else:
            train_mem_matrix = None

    class_to_other_cols = {
        c: [idx for idx, b in enumerate(box_list) if b.cls != c]
        for c in range(num_classes)
    }

    y_val_int = np.asarray(y_val, dtype=np.int64)

    # Track active validation column indices per class for zero-copy per-class score updates.
    active_cols_by_class: dict[int, list[int]] = {c: [] for c in range(num_classes)}
    for col_idx, b in enumerate(box_list):
        active_cols_by_class[b.cls].append(col_idx)

    if (
        init_val_scores is not None
        and init_val_preds is not None
        and init_cm is not None
        and init_metrics is not None
    ):
        base_scores = init_val_scores.copy()
        current_preds = init_val_preds.copy()
        current_cm = init_cm.copy()
        val_acc_init, val_macro_f1_init, val_recalls_init = init_metrics
    else:
        base_scores = np.zeros((num_val, num_classes), dtype=np.float64)
        for c in range(num_classes):
            cols_c = active_cols_by_class[c]
            if cols_c:
                base_scores[:, c] = np.max(val_mem_matrix[:, cols_c], axis=1)

        current_preds = np.argmax(base_scores, axis=1)
        current_cm = np.bincount(
            y_val_int * num_classes + current_preds,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
        val_acc_init, val_macro_f1_init, val_recalls_init = _metrics_from_confusion_matrix(
            current_cm,
            num_val,
            num_classes,
        )

    def eval_without_box(
        drop_bid: int | None = None,
        drop_cls: int | None = None,
    ) -> tuple[
        float,
        float,
        np.ndarray,
        np.ndarray | None,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Compute validation metrics when removing a single box without copying full matrices."""
        if drop_bid is None or drop_cls is None:
            return (
                val_acc_init,
                val_macro_f1_init,
                val_recalls_init,
                None,
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                current_cm.copy(),
            )

        drop_col = id_to_col[drop_bid]
        rem_cols = [col for col in active_cols_by_class[drop_cls] if col != drop_col]
        removed_scores = val_mem_matrix[:, drop_col]
        affected = np.flatnonzero(removed_scores >= base_scores[:, drop_cls] - 1e-12)
        if len(affected) == 0:
            return (
                current_val_acc,
                current_val_macro_f1,
                current_recalls,
                None,
                affected,
                np.empty(0, dtype=np.int64),
                current_cm,
            )

        if len(rem_cols) == 0:
            new_cls_scores = np.zeros(len(affected), dtype=np.float64)
        elif len(rem_cols) == 1:
            new_cls_scores = val_mem_matrix[affected, rem_cols[0]]
        else:
            new_cls_scores = np.max(val_mem_matrix[np.ix_(affected, rem_cols)], axis=1)

        cand_sub = np.flatnonzero(current_preds[affected] == drop_cls)
        if len(cand_sub) == 0:
            return (
                current_val_acc,
                current_val_macro_f1,
                current_recalls,
                new_cls_scores,
                affected,
                current_preds[affected],
                current_cm,
            )

        aff_cand = affected[cand_sub]
        cand_scores = base_scores[aff_cand].copy()
        cand_scores[:, drop_cls] = new_cls_scores[cand_sub]
        cand_preds = np.argmax(cand_scores, axis=1)

        changed_idx = np.flatnonzero(cand_preds != drop_cls)
        if len(changed_idx) == 0:
            return (
                current_val_acc,
                current_val_macro_f1,
                current_recalls,
                new_cls_scores,
                affected,
                current_preds[affected],
                current_cm,
            )

        aff_ch = aff_cand[changed_idx]
        new_preds_ch = cand_preds[changed_idx]
        trial_cm = current_cm.copy()
        np.add.at(trial_cm, (y_val_int[aff_ch], drop_cls), -1)
        np.add.at(trial_cm, (y_val_int[aff_ch], new_preds_ch), 1)

        trial_preds = current_preds[affected].copy()
        trial_preds[cand_sub[changed_idx]] = new_preds_ch

        acc, macro_f1, recalls = _metrics_from_confusion_matrix(
            trial_cm,
            num_val,
            num_classes,
        )
        return (
            acc,
            macro_f1,
            recalls,
            new_cls_scores,
            affected,
            trial_preds,
            trial_cm,
        )

    active_box_ids: set[int] = set(new_fmnn.boxes.keys())
    baseline_val_macro_f1 = val_macro_f1_init
    current_val_macro_f1 = val_macro_f1_init
    current_val_acc = val_acc_init
    current_recalls = val_recalls_init.copy()

    # Fast early exit when every class already sits at min_boxes_per_class.
    if fast_mode and all(len(new_fmnn.boxes_of_class(c)) <= min_boxes_per_class for c in initial_classes_present):
        elapsed_time = time.perf_counter() - start_time
        log = PruneLog(
            visits=[],
            boxes_before=boxes_before,
            boxes_after=boxes_before,
            pruned_count=0,
            elapsed_time=elapsed_time,
            val_macro_f1_before=val_macro_f1_init,
            val_macro_f1_after=val_macro_f1_init,
            val_acc_before=val_acc_init,
            val_acc_after=val_acc_init,
            val_recalls_before=val_recalls_init.tolist(),
            val_recalls_after=val_recalls_init.tolist(),
        )
        if defer_materialization:
            return active_box_ids, log
        return new_fmnn, log

    # Precompute initial training predictions for graph consistency check (skipped in fast_mode).
    node_preds_map: dict[int, int] = {}
    if not fast_mode:
        init_train_preds = new_fmnn.predict(Z_train)
        node_preds_map = {nid: int(init_train_preds[idx]) for idx, nid in enumerate(train_ids_list)}

    # 2. Score and sort candidate boxes
    if cached_importance is not None:
        scored_boxes = list(cached_importance)
    else:
        scored_boxes = []
        for bid, box in sorted(new_fmnn.boxes.items()):
            if mode == "naive_coverage":
                imp = float(len(box.members))
                terms = {"coverage": imp, "purity": 1.0, "boundary_importance": 0.0, "graph_region": 0.0, "redundancy": 0.0}
            else:
                imp, terms = _compute_importance_score(
                    box=box,
                    fmnn=new_fmnn,
                    ctx=ctx,
                    Z_train=Z_train,
                    node_to_train_idx=node_to_train_idx,
                    mu1=mu1,
                    mu2=mu2,
                    mu3=mu3,
                    mu4=mu4,
                    mu5=mu5,
                    boundary_ratio=boundary_ratio,
                    graph_region_mix=graph_region_mix,
                    use_graph_association=use_graph_association,
                    train_mem_matrix=train_mem_matrix,
                    id_to_col=id_to_col,
                    class_to_other_cols=class_to_other_cols,
                    fast_mode=fast_mode,
                )
            scored_boxes.append((imp, bid, terms))

        # Sort ascending by importance (lowest box ID wins ties)
        scored_boxes.sort(key=lambda item: (item[0], item[1]))

    visits: list[dict[str, Any]] = []
    pruned_count = 0

    # 3. Sequential candidate removal testing
    for imp, bid, terms in scored_boxes:
        box = new_fmnn.boxes[bid]
        c = box.cls

        # 3a. Safeguard: minimum boxes per class
        active_class_boxes = [b for b in active_box_ids if new_fmnn.boxes[b].cls == c]
        if len(active_class_boxes) <= min_boxes_per_class:
            visits.append({
                "id": bid,
                "cls": c,
                "importance": imp,
                **terms,
                "decision": "kept",
                "reason": "min_boxes_per_class",
                "val_f1_before": current_val_macro_f1,
                "val_f1_after": current_val_macro_f1,
                "nodes_lost_membership": 0,
                "consistency_change": 0.0,
            })
            continue

        # 3b. Fast temporary removal evaluation on validation split
        (
            temp_acc,
            temp_macro_f1,
            temp_recalls,
            new_cls_scores,
            affected_rows,
            trial_preds,
            trial_cm,
        ) = eval_without_box(bid, c)

        # 3c. Count member training nodes that lose their best same-class membership (membership < 1.0)
        lost_membership_count = 0
        consistency_change = 0.0
        if not fast_mode:
            rem_cls_cols = [col for col in active_cols_by_class[c] if col != id_to_col[bid]]
            mem_rows = [node_to_train_idx[nid] for nid in box.members if nid in node_to_train_idx]
            if mem_rows:
                if train_mem_matrix is not None and rem_cls_cols:
                    best_rem = np.max(train_mem_matrix[np.ix_(mem_rows, rem_cls_cols)], axis=1)
                    lost_membership_count = int(np.sum(best_rem < 1.0 - 1e-12))
                elif not rem_cls_cols:
                    lost_membership_count = int(len(mem_rows))

            # 3d. Check graph-local class consistency among training neighbors
            if box.members:
                sub = ctx.adj_csr[box.members, :]
                nbr_indices = np.unique(sub.indices)
                train_nbrs = [nid for nid in nbr_indices if nid in node_to_train_idx]
                if train_nbrs:
                    nbr_train_indices = [node_to_train_idx[nid] for nid in train_nbrs]
                    nbr_scores_after = np.zeros((len(train_nbrs), num_classes))
                    drop_col = id_to_col[bid]
                    for cl in range(num_classes):
                        cl_cols = [col for col in active_cols_by_class[cl] if col != drop_col]
                        if cl_cols and train_mem_matrix is not None:
                            nbr_scores_after[:, cl] = np.max(train_mem_matrix[np.ix_(nbr_train_indices, cl_cols)], axis=1)

                    nbr_preds_after = np.argmax(nbr_scores_after, axis=1)
                    nbr_preds_before = [node_preds_map[nid] for nid in train_nbrs]
                    true_nbr_labels = [int(ctx.y[nid]) for nid in train_nbrs]

                    share_before = float(np.mean([p == t for p, t in zip(nbr_preds_before, true_nbr_labels)]))
                    share_after = float(np.mean([p == t for p, t in zip(nbr_preds_after, true_nbr_labels)]))
                    consistency_change = share_after - share_before

        # 3e. Acceptance checks
        step_drop = current_val_macro_f1 - temp_macro_f1
        total_drop = baseline_val_macro_f1 - temp_macro_f1
        max_recall_drop = float(np.max(current_recalls - temp_recalls))

        accepted = True
        reject_reason = ""

        if step_drop > tau_prune:
            accepted = False
            reject_reason = "macro_f1_step_drop"
        elif total_drop > 2.0 * tau_prune:
            accepted = False
            reject_reason = "macro_f1_total_drop"
        elif recall_safeguard and max_recall_drop > 2.0 * tau_prune:
            accepted = False
            reject_reason = "per_class_recall_drop"

        if accepted:
            active_box_ids.remove(bid)
            active_cols_by_class[c].remove(id_to_col[bid])
            if new_cls_scores is not None:
                base_scores[affected_rows, c] = new_cls_scores
            if len(affected_rows) > 0:
                current_preds[affected_rows] = trial_preds
                current_cm = trial_cm
            pruned_count += 1
            visits.append({
                "id": bid,
                "cls": c,
                "importance": imp,
                **terms,
                "decision": "pruned",
                "reason": "accepted",
                "val_f1_before": current_val_macro_f1,
                "val_f1_after": temp_macro_f1,
                "nodes_lost_membership": lost_membership_count,
                "consistency_change": consistency_change,
            })
            current_val_macro_f1 = temp_macro_f1
            current_val_acc = temp_acc
            current_recalls = temp_recalls.copy()
        else:
            visits.append({
                "id": bid,
                "cls": c,
                "importance": imp,
                **terms,
                "decision": "kept",
                "reason": reject_reason,
                "val_f1_before": current_val_macro_f1,
                "val_f1_after": current_val_macro_f1,
                "nodes_lost_membership": lost_membership_count,
                "consistency_change": consistency_change,
            })

    # 4. Construct pruned FMNN and reassign training members
    if defer_materialization:
        elapsed_time = time.perf_counter() - start_time
        boxes_after = len(active_box_ids)
        log = PruneLog(
            visits=visits,
            boxes_before=boxes_before,
            boxes_after=boxes_after,
            pruned_count=pruned_count,
            elapsed_time=elapsed_time,
            val_macro_f1_before=val_macro_f1_init,
            val_macro_f1_after=current_val_macro_f1,
            val_acc_before=val_acc_init,
            val_acc_after=current_val_acc,
            val_recalls_before=val_recalls_init.tolist(),
            val_recalls_after=current_recalls.tolist(),
        )
        return active_box_ids, log

    for bid in list(new_fmnn.boxes.keys()):
        if bid not in active_box_ids:
            new_fmnn.remove_box(bid)

    new_fmnn.reassign_members(
        Z_train,
        y_train,
        train_ids_list,
        cached_train_mem=train_mem_matrix,
        id_to_col=id_to_col,
    )

    # Post-pruning assertions
    all_members = []
    for b in new_fmnn.boxes.values():
        all_members.extend(b.members)
    assert len(all_members) == len(train_ids_list), "Member count does not match total training nodes"
    assert len(all_members) == len(set(all_members)), "Duplicate training node member assignments found"
    assert new_fmnn.cross_class_overlap_count() == 0, "Cross-class overlap detected after pruning"
    assert new_fmnn.num_boxes() <= fmnn.num_boxes(), "Box count cannot increase during pruning"
    after_classes_present = {b.cls for b in new_fmnn.boxes.values()}
    assert after_classes_present == initial_classes_present, "Pruning must never remove the last box of any class"

    elapsed_time = time.perf_counter() - start_time
    boxes_after = new_fmnn.num_boxes()
    del val_mem_matrix, train_mem_matrix, base_scores

    log = PruneLog(
        visits=visits,
        boxes_before=boxes_before,
        boxes_after=boxes_after,
        pruned_count=pruned_count,
        elapsed_time=elapsed_time,
        val_macro_f1_before=val_macro_f1_init,
        val_macro_f1_after=current_val_macro_f1,
        val_acc_before=val_acc_init,
        val_acc_after=current_val_acc,
        val_recalls_before=val_recalls_init.tolist(),
        val_recalls_after=current_recalls.tolist(),
    )

    return new_fmnn, log


def geometry_only_prune(
    fmnn: FMNN,
    graph_ctx: Any,
    cfg_prune: dict[str, Any],
    Z_train: np.ndarray,
    y_train: np.ndarray,
    train_ids: list[int] | np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[FMNN, PruneLog]:
    """Execute geometry-only hyperbox pruning (GraphRegionImportance = 0)."""
    return gshr_prune(
        fmnn=fmnn,
        graph_ctx=graph_ctx,
        cfg_prune=cfg_prune,
        Z_train=Z_train,
        y_train=y_train,
        train_ids=train_ids,
        Z_val=Z_val,
        y_val=y_val,
        mode="geometry_only",
    )


def naive_prune_by_coverage(
    fmnn: FMNN,
    graph_ctx: Any,
    cfg_prune: dict[str, Any],
    Z_train: np.ndarray,
    y_train: np.ndarray,
    train_ids: list[int] | np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[FMNN, PruneLog]:
    """Execute naive hyperbox pruning sorted strictly by ascending member count."""
    return gshr_prune(
        fmnn=fmnn,
        graph_ctx=graph_ctx,
        cfg_prune=cfg_prune,
        Z_train=Z_train,
        y_train=y_train,
        train_ids=train_ids,
        Z_val=Z_val,
        y_val=y_val,
        mode="naive_coverage",
    )
