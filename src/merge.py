"""Graph-Structure-Aware Hyperbox Merging (GSHR Merge)."""

from __future__ import annotations

import copy
import heapq
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from src.fmnn import FMNN, HyperBox
from src.graph_stats import (
    BoxGraphCache,
    build_box_adjacency,
    class_purity,
    graph_association,
    neighbor_label_agreement,
)


@dataclass
class GraphContext:
    """Container holding graph structural and label data."""

    adj_csr: sp.csr_matrix
    degree: np.ndarray
    y: np.ndarray
    train_mask: np.ndarray


def _unpack_graph_ctx(graph_ctx: Any) -> GraphContext:
    """Normalize input graph context into a GraphContext dataclass."""
    if isinstance(graph_ctx, GraphContext):
        return graph_ctx

    if isinstance(graph_ctx, dict):
        return GraphContext(
            adj_csr=graph_ctx["adj_csr"],
            degree=np.asarray(graph_ctx["degree"], dtype=np.int64),
            y=np.asarray(graph_ctx["y"]),
            train_mask=np.asarray(graph_ctx["train_mask"], dtype=bool),
        )

    # Assume object has graph attributes (e.g. GraphData)
    adj_csr = graph_ctx.adj_csr
    degree = graph_ctx.degree
    y = graph_ctx.y.cpu().numpy() if hasattr(graph_ctx.y, "cpu") else np.asarray(graph_ctx.y)

    if hasattr(graph_ctx, "train_mask"):
        mask = graph_ctx.train_mask
        train_mask = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask, dtype=bool)
    elif hasattr(graph_ctx, "train_idx"):
        num_nodes = adj_csr.shape[0]
        train_mask = np.zeros(num_nodes, dtype=bool)
        idx = graph_ctx.train_idx.cpu().numpy() if hasattr(graph_ctx.train_idx, "cpu") else np.asarray(graph_ctx.train_idx)
        train_mask[idx] = True
    else:
        num_nodes = adj_csr.shape[0]
        train_mask = np.ones(num_nodes, dtype=bool)

    return GraphContext(adj_csr=adj_csr, degree=degree, y=y, train_mask=train_mask)


class MergeLog(list):
    """List of accepted merge events with summary counters and timings."""

    def __init__(
        self,
        merges: list[dict[str, Any]],
        rejections: dict[str, int],
        boxes_before: int,
        boxes_after: int,
        elapsed_time: float,
    ) -> None:
        super().__init__(merges)
        self.rejections = rejections
        self.boxes_before = boxes_before
        self.boxes_after = boxes_after
        self.elapsed_time = elapsed_time

    def to_dict(self) -> dict[str, Any]:
        """Convert log metadata and merges to dictionary format."""
        return {
            "merges": list(self),
            "rejections": self.rejections,
            "boxes_before": self.boxes_before,
            "boxes_after": self.boxes_after,
            "elapsed_time": self.elapsed_time,
        }


def _evaluate_candidate_pair(
    box_i: HyperBox,
    box_j: HyperBox,
    fmnn: FMNN,
    ctx: GraphContext,
    alpha: float,
    beta: float,
    delta: float,
    eta: float,
    tau_merge: float,
    max_volume_inflation: float,
    theta: float,
    class_purity_safeguard: bool,
    rejections: dict[str, int],
    fast: bool = True,
    box_edges: dict[tuple[int, int], int] | None = None,
    vol_dict: dict[int, int] | None = None,
    label_counts: dict[int, dict[int, int]] | None = None,
) -> tuple[bool, float, dict[str, float] | None, str]:
    """Evaluate whether two same-class hyperboxes can merge.

    Returns:
        (is_accepted, score_M, component_dict, reason)
    """
    assert box_i.cls == box_j.cls, "Cannot merge boxes of different classes"

    c = box_i.cls
    n_dims = fmnn.n_dims

    # Proposed merged box coordinates
    v_merged = np.minimum(box_i.v, box_j.v)
    w_merged = np.maximum(box_i.w, box_j.w)
    merged_members = list(set(box_i.members + box_j.members))
    merged_box = HyperBox(id=-1, v=v_merged, w=w_merged, cls=c, members=merged_members)

    if not fast:
        # Legacy behavior: check overlap and purity before score
        size_merged = float(np.sum(w_merged - v_merged))
        if size_merged > n_dims * theta:
            rejections["size_limit"] += 1
            return False, 0.0, None, "size_limit"

        other_boxes = [b for b in fmnn.boxes.values() if b.cls != c]
        for b_other in other_boxes:
            if fmnn.boxes_overlap(merged_box, b_other):
                rejections["cross_class_overlap"] += 1
                return False, 0.0, None, "cross_class_overlap"

        size_i = fmnn.box_volume_proxy(box_i)
        size_j = fmnn.box_volume_proxy(box_j)
        inflation = max(0.0, size_merged - size_i - size_j) / (size_merged + 1e-12)
        inflation = float(np.clip(inflation, 0.0, 1.0))
        if inflation > max_volume_inflation:
            rejections["volume_inflation"] += 1
            return False, 0.0, None, "volume_inflation"

        purity = class_purity(merged_members, c, ctx.y, ctx.train_mask)
        if class_purity_safeguard and purity < 1.0:
            rejections["class_purity"] += 1
            return False, 0.0, None, "class_purity"

        center_dist = float(np.linalg.norm(fmnn.center(box_i) - fmnn.center(box_j)))
        max_dist = np.sqrt(n_dims)
        geometry = float(np.clip(1.0 - (center_dist / (max_dist + 1e-12)), 0.0, 1.0))

        if beta > 0.0:
            graph_assoc = graph_association(box_i.members, box_j.members, ctx.adj_csr, ctx.degree)
        else:
            graph_assoc = 0.0
        graph_assoc = float(np.clip(graph_assoc, 0.0, 1.0))
        class_cons = float(np.clip(purity, 0.0, 1.0))

        agr, _ = neighbor_label_agreement(merged_members, c, ctx.adj_csr, ctx.y, ctx.train_mask)
        conflict = float(np.clip(1.0 - agr, 0.0, 1.0))
        risk = float(np.clip((inflation + conflict) / 3.0, 0.0, 1.0))

        M = float(alpha * geometry + beta * graph_assoc + delta * class_cons - eta * risk)
        if M < tau_merge:
            rejections["threshold"] += 1
            return False, M, None, "threshold"

        components = {
            "geometry": geometry,
            "graph": graph_assoc,
            "consistency": class_cons,
            "risk": risk,
        }
        return True, M, components, "accepted"

    # --- FAST OPTIMIZED MERGE PATH ---
    # 1. Cheap Hard Check: class and theta (Simpson expansion limit)
    size_merged = float(np.sum(w_merged - v_merged))
    if size_merged > n_dims * theta:
        rejections["size_limit"] += 1
        return False, 0.0, None, "size_limit"

    # Volume inflation calculation (sum of side lengths, not product)
    size_i = float(np.sum(box_i.w - box_i.v))
    size_j = float(np.sum(box_j.w - box_j.v))
    inflation = max(0.0, size_merged - size_i - size_j) / (size_merged + 1e-12)
    inflation = float(np.clip(inflation, 0.0, 1.0))

    # Class purity from cached label counts
    if class_purity_safeguard:
        if label_counts is not None:
            counts_i = label_counts.get(box_i.id, {})
            counts_j = label_counts.get(box_j.id, {})
            impure = any(lbl != c for lbl in counts_i) or any(lbl != c for lbl in counts_j)
            if impure:
                rejections["class_purity"] += 1
                return False, 0.0, None, "class_purity"
        else:
            purity = class_purity(merged_members, c, ctx.y, ctx.train_mask)
            if purity < 1.0:
                rejections["class_purity"] += 1
                return False, 0.0, None, "class_purity"

    # 2. Score Computation: Cheap terms before expensive cross-class overlap
    center_i = (box_i.v + box_i.w) * 0.5
    center_j = (box_j.v + box_j.w) * 0.5
    center_dist = float(np.linalg.norm(center_i - center_j))
    max_dist = np.sqrt(n_dims)
    geometry = float(np.clip(1.0 - (center_dist / (max_dist + 1e-12)), 0.0, 1.0))

    if beta > 0.0:
        if box_edges is not None and vol_dict is not None:
            pair_key = (min(box_i.id, box_j.id), max(box_i.id, box_j.id))
            e_ij = box_edges.get(pair_key, 0)
            vol_i = vol_dict.get(box_i.id, 0)
            vol_j = vol_dict.get(box_j.id, 0)
            denom = np.sqrt(float(vol_i * vol_j) + 1e-12)
            graph_assoc = float(np.clip(e_ij / denom, 0.0, 1.0))
        else:
            graph_assoc = graph_association(box_i.members, box_j.members, ctx.adj_csr, ctx.degree)
    else:
        graph_assoc = 0.0
    graph_assoc = float(np.clip(graph_assoc, 0.0, 1.0))

    # Neighbor agreement and risk
    agr, _ = neighbor_label_agreement(merged_members, c, ctx.adj_csr, ctx.y, ctx.train_mask)
    conflict = float(np.clip(1.0 - agr, 0.0, 1.0))
    risk = float(np.clip((inflation + conflict) / 3.0, 0.0, 1.0))

    # Remove constant ClassConsistency term (purity is 1.0 across all pure same-class boxes)
    M = float(alpha * geometry + beta * graph_assoc - eta * risk)

    if M < tau_merge:
        rejections["threshold"] += 1
        return False, M, None, "threshold"

    # 3. Run Cross-Class Overlap Check ONLY for candidates above tau_merge
    other_boxes = [b for b in fmnn.boxes.values() if b.cls != c]
    for b_other in other_boxes:
        if fmnn.boxes_overlap(merged_box, b_other):
            rejections["cross_class_overlap"] += 1
            return False, 0.0, None, "cross_class_overlap"

    # 4. Volume inflation check
    if inflation > max_volume_inflation:
        rejections["volume_inflation"] += 1
        return False, 0.0, None, "volume_inflation"

    components = {
        "geometry": geometry,
        "graph": graph_assoc,
        "consistency": 1.0,
        "risk": risk,
    }
    return True, M, components, "accepted"


def _generate_candidates_for_box(
    bid: int,
    fmnn: FMNN,
    ctx: GraphContext,
    knn_candidates: int,
    use_graph_edges: bool,
) -> set[tuple[int, int]]:
    """Generate candidate same-class merge pairs involving box bid."""
    target_box = fmnn.boxes[bid]
    c = target_box.cls
    same_class_boxes = [b for b in fmnn.boxes.values() if b.cls == c and b.id != bid]
    if not same_class_boxes:
        return set()

    pairs: set[tuple[int, int]] = set()

    # Center-distance k-NN candidates
    target_center = fmnn.center(target_box)
    scored_boxes = []
    for b in same_class_boxes:
        dist = np.linalg.norm(target_center - fmnn.center(b))
        scored_boxes.append((dist, b.id))

    # Sort deterministically by distance, then box id
    scored_boxes.sort(key=lambda item: (item[0], item[1]))
    k = min(knn_candidates, len(scored_boxes))
    for _, other_id in scored_boxes[:k]:
        pairs.add((min(bid, other_id), max(bid, other_id)))

    # Graph-adjacency candidates
    if use_graph_edges:
        target_members = target_box.members
        for b in same_class_boxes:
            # Check edge connection with target
            sub = ctx.adj_csr[target_members, :][:, b.members]
            if sub.nnz > 0:
                pairs.add((min(bid, b.id), max(bid, b.id)))

    return pairs


def gshr_merge(
    fmnn: FMNN,
    graph_ctx: Any,
    cfg_merge: dict[str, Any],
    train_data: Any = None,
    fast: bool | None = None,
) -> tuple[FMNN, MergeLog]:
    """Execute Graph-Structure-Aware Hyperbox Merging.

    Args:
        fmnn: Base FMNN model (will not be mutated).
        graph_ctx: GraphContext or GraphData container.
        cfg_merge: Merging configuration dictionary (merge section of config).
        train_data: Optional reference to training data.
        fast: Flag to enable fast merge path (default True or from config).

    Returns:
        (new_fmnn, merge_log)
    """
    start_time = time.perf_counter()
    new_fmnn = fmnn.copy()
    ctx = _unpack_graph_ctx(graph_ctx)

    if fast is None:
        fast = bool(cfg_merge.get("fast", True))

    # Config unpacking
    alpha = float(cfg_merge.get("alpha", 0.35))
    beta = float(cfg_merge.get("beta", 0.35))
    delta = float(cfg_merge.get("delta", 0.20))
    eta = float(cfg_merge.get("eta", 0.10))
    tau_merge = float(cfg_merge.get("tau_merge", 0.5))
    knn_candidates = int(cfg_merge.get("knn_candidates", 10))
    max_volume_inflation = float(cfg_merge.get("max_volume_inflation", 0.5))
    use_graph_edges = bool(cfg_merge.get("use_graph_edges_for_candidates", True))
    theta = float(cfg_merge.get("theta", new_fmnn.theta))

    # Safeguard overrides
    safeguards = cfg_merge.get("safeguards", {})
    if not safeguards.get("graph_association", True):
        beta = 0.0
        use_graph_edges = False
    class_purity_safeguard = safeguards.get("class_purity", True)

    rejections: dict[str, int] = {
        "size_limit": 0,
        "cross_class_overlap": 0,
        "volume_inflation": 0,
        "class_purity": 0,
        "threshold": 0,
    }

    boxes_before = new_fmnn.num_boxes()
    merges: list[dict[str, Any]] = []

    # Initialize graph statistics cache
    cache = BoxGraphCache(
        adj_csr=ctx.adj_csr,
        degree=ctx.degree,
        y=ctx.y,
        train_mask=ctx.train_mask,
    )
    cache.build_all(new_fmnn)

    # Fast structures: indicator product B.T @ A @ B, volumes, and label counts
    box_edges: dict[tuple[int, int], int] | None = None
    vol_dict: dict[int, int] | None = None
    label_counts: dict[int, dict[int, int]] | None = None

    if fast:
        box_edges = build_box_adjacency(new_fmnn, ctx.adj_csr)
        vol_dict = {
            bid: int(np.sum(ctx.degree[b.members])) if len(b.members) > 0 else 0
            for bid, b in new_fmnn.boxes.items()
        }
        label_counts = {}
        for bid, b in new_fmnn.boxes.items():
            counts: dict[int, int] = {}
            for u in b.members:
                if ctx.train_mask[u]:
                    lbl = int(ctx.y[u])
                    counts[lbl] = counts.get(lbl, 0) + 1
            label_counts[bid] = counts

    # Candidate pair set: (min_id, max_id)
    candidate_pairs: set[tuple[int, int]] = set()

    for bid in new_fmnn.boxes.keys():
        box_pairs = _generate_candidates_for_box(
            bid=bid,
            fmnn=new_fmnn,
            ctx=ctx,
            knn_candidates=knn_candidates,
            use_graph_edges=use_graph_edges,
        )
        candidate_pairs.update(box_pairs)

    # Priority queue: (-M, min_id, max_id)
    heap: list[tuple[float, int, int]] = []
    visited_pairs: set[tuple[int, int]] = set()

    for bid1, bid2 in sorted(candidate_pairs):
        visited_pairs.add((bid1, bid2))
        accepted, M, comps, reason = _evaluate_candidate_pair(
            box_i=new_fmnn.boxes[bid1],
            box_j=new_fmnn.boxes[bid2],
            fmnn=new_fmnn,
            ctx=ctx,
            alpha=alpha,
            beta=beta,
            delta=delta,
            eta=eta,
            tau_merge=tau_merge,
            max_volume_inflation=max_volume_inflation,
            theta=theta,
            class_purity_safeguard=class_purity_safeguard,
            rejections=rejections,
            fast=fast,
            box_edges=box_edges,
            vol_dict=vol_dict,
            label_counts=label_counts,
        )
        if accepted:
            heapq.heappush(heap, (-M, bid1, bid2))

    # Priority queue processing
    while heap:
        neg_M, bid1, bid2 = heapq.heappop(heap)
        M = -neg_M

        # Lazy invalidation
        if bid1 not in new_fmnn.boxes or bid2 not in new_fmnn.boxes:
            continue

        box_i = new_fmnn.boxes[bid1]
        box_j = new_fmnn.boxes[bid2]

        if not fast:
            # Re-verify acceptance against current state of model in slow mode
            temp_rejections: dict[str, int] = {k: 0 for k in rejections}
            accepted, current_M, comps, reason = _evaluate_candidate_pair(
                box_i=box_i,
                box_j=box_j,
                fmnn=new_fmnn,
                ctx=ctx,
                alpha=alpha,
                beta=beta,
                delta=delta,
                eta=eta,
                tau_merge=tau_merge,
                max_volume_inflation=max_volume_inflation,
                theta=theta,
                class_purity_safeguard=class_purity_safeguard,
                rejections=temp_rejections,
                fast=False,
            )
            if not accepted:
                for k, count in temp_rejections.items():
                    rejections[k] += count
                continue
            assert comps is not None
        else:
            # In fast mode: verify cross-class overlap against any boxes created since (bid1, bid2) was scored
            v_m = np.minimum(box_i.v, box_j.v)
            w_m = np.maximum(box_i.w, box_j.w)
            temp_box = HyperBox(id=-1, v=v_m, w=w_m, cls=box_i.cls, members=[])
            other_boxes = [b for b in new_fmnn.boxes.values() if b.cls != box_i.cls]
            if any(new_fmnn.boxes_overlap(temp_box, b_oth) for b_oth in other_boxes):
                rejections["cross_class_overlap"] += 1
                continue
            current_M = M
            comps = {"geometry": 0.0, "graph": 0.0, "consistency": 1.0, "risk": 0.0}

        # Execute merge
        new_id = new_fmnn.next_id
        new_fmnn.next_id += 1

        new_v = np.minimum(box_i.v, box_j.v)
        new_w = np.maximum(box_i.w, box_j.w)
        new_members = sorted(list(set(box_i.members + box_j.members)))

        merged_box = HyperBox(
            id=new_id,
            v=new_v,
            w=new_w,
            cls=box_i.cls,
            members=new_members,
        )

        new_fmnn.remove_box(bid1)
        new_fmnn.remove_box(bid2)
        new_fmnn.add_box(merged_box)

        if fast and box_edges is not None and vol_dict is not None and label_counts is not None:
            vol_dict[new_id] = vol_dict.get(bid1, 0) + vol_dict.get(bid2, 0)
            vol_dict.pop(bid1, None)
            vol_dict.pop(bid2, None)

            # Update edges: for every other remaining box k
            for k in list(new_fmnn.boxes.keys()):
                if k != new_id:
                    e1 = box_edges.pop((min(bid1, k), max(bid1, k)), 0)
                    e2 = box_edges.pop((min(bid2, k), max(bid2, k)), 0)
                    if e1 + e2 > 0:
                        box_edges[(min(new_id, k), max(new_id, k))] = e1 + e2

            # Update label counts
            lbls_new = dict(label_counts.get(bid1, {}))
            for lbl, cnt in label_counts.get(bid2, {}).items():
                lbls_new[lbl] = lbls_new.get(lbl, 0) + cnt
            label_counts[new_id] = lbls_new
            label_counts.pop(bid1, None)
            label_counts.pop(bid2, None)

        cache.drop(bid1)
        cache.drop(bid2)
        cache.refresh(new_id, new_fmnn)

        merges.append({
            "i": bid1,
            "j": bid2,
            "new_id": new_id,
            "M": current_M,
            "geometry": comps["geometry"],
            "graph": comps["graph"],
            "consistency": comps["consistency"],
            "risk": comps["risk"],
        })

        # Generate candidates for newly merged box
        new_candidates = _generate_candidates_for_box(
            bid=new_id,
            fmnn=new_fmnn,
            ctx=ctx,
            knn_candidates=knn_candidates,
            use_graph_edges=use_graph_edges,
        )

        for other_pair in sorted(new_candidates):
            if other_pair not in visited_pairs:
                visited_pairs.add(other_pair)
                acc, sc, _, reas = _evaluate_candidate_pair(
                    box_i=new_fmnn.boxes[other_pair[0]],
                    box_j=new_fmnn.boxes[other_pair[1]],
                    fmnn=new_fmnn,
                    ctx=ctx,
                    alpha=alpha,
                    beta=beta,
                    delta=delta,
                    eta=eta,
                    tau_merge=tau_merge,
                    max_volume_inflation=max_volume_inflation,
                    theta=theta,
                    class_purity_safeguard=class_purity_safeguard,
                    rejections=rejections,
                    fast=fast,
                    box_edges=box_edges,
                    vol_dict=vol_dict,
                    label_counts=label_counts,
                )
                if acc:
                    heapq.heappush(heap, (-sc, other_pair[0], other_pair[1]))

    # Post-merge consistency verification
    all_members = []
    for b in new_fmnn.boxes.values():
        all_members.extend(b.members)
    assert len(all_members) == len(set(all_members)), "Duplicate training node member assignments found"
    assert new_fmnn.cross_class_overlap_count() == 0, "Cross-class overlap detected after merge"

    elapsed_time = time.perf_counter() - start_time
    boxes_after = new_fmnn.num_boxes()

    log = MergeLog(
        merges=merges,
        rejections=rejections,
        boxes_before=boxes_before,
        boxes_after=boxes_after,
        elapsed_time=elapsed_time,
    )

    return new_fmnn, log


def geometry_only_merge(
    fmnn: FMNN,
    graph_ctx: Any,
    cfg_merge: dict[str, Any],
    train_data: Any = None,
) -> tuple[FMNN, MergeLog]:
    """Execute geometry-only hyperbox merging (beta=0, no graph edges for candidates)."""
    cfg_copy = copy.deepcopy(cfg_merge)
    cfg_copy["beta"] = 0.0
    cfg_copy["use_graph_edges_for_candidates"] = False
    return gshr_merge(fmnn, graph_ctx, cfg_copy, train_data=train_data)


# Inspect candidate pairs, per-safeguard pass counts, and score distribution without mutating fmnn.
def debug_merge_candidates(
    fmnn: FMNN,
    graph_ctx: Any,
    cfg_merge: dict[str, Any],
) -> dict[str, Any]:
    ctx = _unpack_graph_ctx(graph_ctx)
    alpha = float(cfg_merge.get("alpha", 0.35))
    beta = float(cfg_merge.get("beta", 0.35))
    delta = float(cfg_merge.get("delta", 0.20))
    eta = float(cfg_merge.get("eta", 0.10))
    tau_merge = float(cfg_merge.get("tau_merge", 0.5))
    knn_candidates = int(cfg_merge.get("knn_candidates", 10))
    max_volume_inflation = float(cfg_merge.get("max_volume_inflation", 0.5))
    use_graph_edges = bool(cfg_merge.get("use_graph_edges_for_candidates", True))
    theta = float(cfg_merge.get("theta", fmnn.theta))
    safeguards = cfg_merge.get("safeguards", {})
    if not safeguards.get("graph_association", True):
        beta = 0.0
        use_graph_edges = False

    n_dims = fmnn.n_dims
    same_class_total_pairs = sum(
        len(fmnn.boxes_of_class(c)) * (len(fmnn.boxes_of_class(c)) - 1) // 2
        for c in range(fmnn.num_classes)
    )
    candidate_pairs: set[tuple[int, int]] = set()
    for bid in sorted(fmnn.boxes.keys()):
        candidate_pairs.update(
            _generate_candidates_for_box(bid, fmnn, ctx, knn_candidates, use_graph_edges)
        )

    pass_counts = {
        "same_class_geom_overlap": 0,
        "size_limit": 0,
        "no_cross_class_overlap": 0,
        "volume_inflation": 0,
        "class_purity": 0,
        "all_safeguards": 0,
        "score_threshold": 0,
    }
    scores_all: list[float] = []
    pair_details: list[dict[str, Any]] = []

    for b1, b2 in sorted(candidate_pairs):
        bi, bj = fmnn.boxes[b1], fmnn.boxes[b2]
        v_m = np.minimum(bi.v, bj.v)
        w_m = np.maximum(bi.w, bj.w)
        merged_members = list(set(bi.members + bj.members))
        merged_box = HyperBox(id=-1, v=v_m, w=w_m, cls=bi.cls, members=merged_members)

        geom_ov = FMNN.boxes_overlap(bi, bj)
        size_m = float(np.sum(w_m - v_m))
        ok_size = size_m <= n_dims * theta
        ok_cross_ov = not any(
            fmnn.boxes_overlap(merged_box, bo) for bo in fmnn.boxes.values() if bo.cls != bi.cls
        )
        si = fmnn.box_volume_proxy(bi)
        sj = fmnn.box_volume_proxy(bj)
        infl = float(np.clip(max(0.0, size_m - si - sj) / (size_m + 1e-12), 0.0, 1.0))
        ok_infl = infl <= max_volume_inflation
        pur = class_purity(merged_members, bi.cls, ctx.y, ctx.train_mask)
        ok_pur = (not safeguards.get("class_purity", True)) or (pur >= 1.0)

        pass_counts["same_class_geom_overlap"] += int(geom_ov)
        pass_counts["size_limit"] += int(ok_size)
        pass_counts["no_cross_class_overlap"] += int(ok_cross_ov)
        pass_counts["volume_inflation"] += int(ok_infl)
        pass_counts["class_purity"] += int(ok_pur)
        all_ok = ok_size and ok_cross_ov and ok_infl and ok_pur
        pass_counts["all_safeguards"] += int(all_ok)

        dist = np.linalg.norm(fmnn.center(bi) - fmnn.center(bj))
        geo = float(np.clip(1.0 - dist / (np.sqrt(n_dims) + 1e-12), 0.0, 1.0))
        ga = graph_association(bi.members, bj.members, ctx.adj_csr, ctx.degree) if beta > 0.0 else 0.0
        ga = float(np.clip(ga, 0.0, 1.0))
        agr, _ = neighbor_label_agreement(merged_members, bi.cls, ctx.adj_csr, ctx.y, ctx.train_mask)
        risk = float(np.clip((infl + (1.0 - agr) + (0.0 if ok_cross_ov else 1.0)) / 3.0, 0.0, 1.0))
        score_M = float(alpha * geo + beta * ga + delta * pur - eta * risk)
        scores_all.append(score_M)
        if all_ok and score_M >= tau_merge:
            pass_counts["score_threshold"] += 1

        pair_details.append({
            "pair": (b1, b2),
            "cls": bi.cls,
            "same_class_geom_overlap": geom_ov,
            "size_merged": size_m,
            "size_limit": n_dims * theta,
            "ok_size": ok_size,
            "ok_cross_overlap": ok_cross_ov,
            "ok_volume_inflation": ok_infl,
            "ok_purity": ok_pur,
            "M": score_M,
        })

    if scores_all:
        arr = np.asarray(scores_all, dtype=np.float64)
        score_dist = {
            "count": len(scores_all),
            "min": float(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p75": float(np.percentile(arr, 75)),
            "max": float(np.max(arr)),
        }
    else:
        score_dist = {"count": 0, "min": None, "p25": None, "median": None, "mean": None, "p75": None, "max": None}

    return {
        "num_boxes": fmnn.num_boxes(),
        "total_same_class_pairs": same_class_total_pairs,
        "candidate_pairs_count": len(candidate_pairs),
        "pass_counts": pass_counts,
        "score_distribution": score_dist,
        "pair_details": pair_details,
    }


# Fast optional merge: evaluates same-class pairs only and skips scoring if no pair passes geometric overlap test.
def fast_gshr_merge(
    fmnn: FMNN,
    graph_ctx: Any,
    cfg_merge: dict[str, Any],
) -> tuple[FMNN, MergeLog]:
    start_time = time.perf_counter()
    new_fmnn = fmnn.copy()
    boxes_before = new_fmnn.num_boxes()
    rejections: dict[str, int] = {
        "size_limit": 0,
        "cross_class_overlap": 0,
        "volume_inflation": 0,
        "class_purity": 0,
        "threshold": 0,
        "no_geom_overlap": 0,
    }

    if not bool(cfg_merge.get("enabled", True)) or boxes_before <= 1:
        elapsed = time.perf_counter() - start_time
        return new_fmnn, MergeLog([], rejections, boxes_before, boxes_before, elapsed)

    theta = float(cfg_merge.get("theta", new_fmnn.theta))
    size_mult = float(cfg_merge.get("size_limit_multiplier", 1.0))
    effective_limit = new_fmnn.n_dims * theta * size_mult
    require_same_ov = bool(cfg_merge.get("require_same_class_overlap", True))

    # Inspect ONLY same-class pairs and check geometric overlap before graph caching or scoring.
    eligible_pairs: list[tuple[int, int]] = []
    for c in range(new_fmnn.num_classes):
        c_boxes = new_fmnn.boxes_of_class(c)
        if len(c_boxes) < 2:
            continue
        other_boxes = [b for b in new_fmnn.boxes.values() if b.cls != c]
        for idx_i in range(len(c_boxes)):
            bi = c_boxes[idx_i]
            for idx_j in range(idx_i + 1, len(c_boxes)):
                bj = c_boxes[idx_j]
                if require_same_ov and not FMNN.boxes_overlap(bi, bj):
                    rejections["no_geom_overlap"] += 1
                    continue
                v_m = np.minimum(bi.v, bj.v)
                w_m = np.maximum(bi.w, bj.w)
                if np.sum(w_m - v_m) > effective_limit:
                    rejections["size_limit"] += 1
                    continue
                m_box = HyperBox(id=-1, v=v_m, w=w_m, cls=c, members=[])
                if any(FMNN.boxes_overlap(m_box, bo) for bo in other_boxes):
                    rejections["cross_class_overlap"] += 1
                    continue
                eligible_pairs.append((bi.id, bj.id))

    if not eligible_pairs:
        elapsed = time.perf_counter() - start_time
        return new_fmnn, MergeLog([], rejections, boxes_before, boxes_before, elapsed)

    # Delegate full priority-queue merging only when at least one same-class pair passes geometry.
    cfg_run = copy.deepcopy(cfg_merge)
    cfg_run["theta"] = theta * size_mult
    return gshr_merge(new_fmnn, graph_ctx, cfg_run)
